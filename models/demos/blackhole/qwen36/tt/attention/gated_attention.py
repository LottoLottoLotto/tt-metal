# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5-9B gated full-attention layer — composes config/weights/prefill/decode."""

import os
import ttnn
from models.demos.blackhole.qwen36.tt.attention.config import AttentionConfig
from models.demos.blackhole.qwen36.tt.attention.decode import decode_forward
from models.demos.blackhole.qwen36.tt.attention.prefill import prefill_forward
from models.demos.blackhole.qwen36.tt.attention.weights import load_attention_weights


class Qwen36GatedAttention:
    """Gated Full Attention layer for Qwen3.5-9B with KV cache.

    Uses softmax SDPA with GQA (16 Q heads, 4 KV heads, head_dim=256)
    plus a sigmoid output gate derived from the 2x wide q_proj.
    Q and K are normalized with zero-centered RMSNorm before attention.
    """

    def __init__(self, mesh_device, config: AttentionConfig, state_dict, tensor_cache_path=None):
        self.device = mesh_device
        self.config = config

        self.weights = load_attention_weights(mesh_device, state_dict, tensor_cache_path)
        self.decode_program_configs = {}
        if os.environ.get("QWEN36_SINGLE_1D_DECODE") == "1":
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            grid_w = int(os.environ.get("QWEN36_DECODE_GRID_W", "11"))
            specs = {
                "q": (self.weights.q_proj, 44),
                "k": (self.weights.k_proj, 8),
                "v": (self.weights.v_proj, 8),
                "o": (self.weights.o_proj, 33),
            }
            if self.weights.qkv_fused is not None:
                specs["qkv"] = (self.weights.qkv_fused, 64)
            self.decode_program_configs = {
                name: tpc.create_matmul_1d_decode_progcfg(
                    1,
                    weight.shape[-2],
                    weight.shape[-1],
                    num_cores=num_cores,
                    grid_w=grid_w,
                )
                for name, (weight, num_cores) in specs.items()
            }

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # KV cache state (concat-based prefill)
        self.past_key = None
        self.past_value = None
        # Paged attention state (for vLLM integration)
        self.paged_kv_cache_key = None
        self.paged_kv_cache_value = None
        self.use_paged_attention = False

    def forward(
        self,
        x,
        cos,
        sin,
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
    ):
        T = x.shape[1]
        mc = ttnn.L1_MEMORY_CONFIG if T == 1 else None
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        # Branches are mutually exclusive on T; decode (T==1) is checked first to keep the hot path short.
        if self.use_paged_attention and T == 1:
            # Branch B — paged decode
            return decode_forward(
                x=x,
                cos=cos,
                sin=sin,
                weights=self.weights,
                config=self.config,
                device=self.device,
                ckc=ckc,
                mc=mc,
                position_tensor=position_tensor,
                page_table=page_table,
                paged_kv_cache_key=self.paged_kv_cache_key,
                paged_kv_cache_value=self.paged_kv_cache_value,
                program_configs=self.decode_program_configs,
                qkv_fused_weight=self.weights.qkv_fused,
            )
        elif self.use_paged_attention and T > 1 and chunk_page_table is not None:
            # Branch A — paged prefill
            return prefill_forward(
                x=x,
                cos=cos,
                sin=sin,
                weights=self.weights,
                config=self.config,
                device=self.device,
                ckc=ckc,
                mc=mc,
                paged_kv_cache_key=self.paged_kv_cache_key,
                paged_kv_cache_value=self.paged_kv_cache_value,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                use_paged_attention=True,
            )
        else:
            # Branch C — concat prefill
            output, new_key, new_value = prefill_forward(
                x=x,
                cos=cos,
                sin=sin,
                weights=self.weights,
                config=self.config,
                device=self.device,
                ckc=ckc,
                mc=mc,
                past_key=self.past_key,
                past_value=self.past_value,
                use_paged_attention=False,
            )
            self.past_key = new_key
            self.past_value = new_value
            return output

    def verify_two(
        self,
        x,
        cos,
        sin,
        position_tensor,
        page_table,
        shadow_fill_page_table,
        shadow_source_page,
        shadow_same_page,
    ):
        """Run causal M=2 attention with the second KV position isolated in a shadow page."""
        from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
            gated_attention_forward_ttnn,
        )

        output, key1, value1 = gated_attention_forward_ttnn(
            hidden_states=x,
            q_proj_weight=self.weights.q_proj,
            k_proj_weight=self.weights.k_proj,
            v_proj_weight=self.weights.v_proj,
            o_proj_weight=self.weights.o_proj,
            q_norm_weight=self.weights.q_norm,
            k_norm_weight=self.weights.k_norm,
            cos=cos,
            sin=sin,
            num_attention_heads=self.config.num_heads,
            num_key_value_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            device=self.device,
            norm_eps=self.config.norm_eps,
            compute_kernel_config=self.compute_kernel_config_decode,
            use_optimized_concat=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            norm_weights_pre_offset=True,
            cur_pos_tensor=position_tensor,
            page_table=page_table,
            paged_kv_cache_key=self.paged_kv_cache_key,
            paged_kv_cache_value=self.paged_kv_cache_value,
            q_decode_program_config=self.decode_program_configs.get("q"),
            k_decode_program_config=self.decode_program_configs.get("k"),
            v_decode_program_config=self.decode_program_configs.get("v"),
            o_decode_program_config=self.decode_program_configs.get("o"),
            qkv_fused_weight=self.weights.qkv_fused,
            qkv_fused_q_dim=self.weights.q_proj.shape[-1],
            qkv_fused_k_dim=self.weights.k_proj.shape[-1],
            qkv_fused_program_config=self.decode_program_configs.get("qkv"),
            verify_two=True,
            shadow_fill_page_table=shadow_fill_page_table,
            shadow_source_page=shadow_source_page,
            shadow_same_page=shadow_same_page,
        )
        return output, (key1, value1)

    def reset_cache(self):
        """Clear the concat KV cache for a new sequence."""
        self.past_key = None
        self.past_value = None

    def set_paged_kv_cache(self, k_cache, v_cache):
        """Attach externally-allocated paged KV cache (called once after allocate_kv_cache)."""
        self.paged_kv_cache_key = k_cache
        self.paged_kv_cache_value = v_cache
        self.use_paged_attention = True
