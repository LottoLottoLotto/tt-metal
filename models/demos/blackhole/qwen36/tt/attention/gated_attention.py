# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen gated full attention — composes config/weights/prefill/decode."""

import os
import ttnn
from models.demos.blackhole.qwen36.tt.attention.config import AttentionConfig, optimized_27b_attention
from models.demos.blackhole.qwen36.tt.attention.decode import decode_forward
from models.demos.blackhole.qwen36.tt.attention.prefill import prefill_forward
from models.demos.blackhole.qwen36.tt.attention.weights import load_attention_weights


class Qwen36GatedAttention:
    """Gated Full Attention with KV cache.

    Uses softmax SDPA with GQA (27B: 24 Q heads, 4 KV heads, head_dim=256)
    plus a sigmoid output gate derived from the 2x wide q_proj.
    Q and K are normalized with zero-centered RMSNorm before attention.
    """

    def __init__(self, mesh_device, config: AttentionConfig, state_dict, tensor_cache_path=None, pack_decode=False):
        """pack_decode: single-request 35B decode may serve the fused QKV / o_proj through decode-only bank-packed
        copies (QWEN36_PACKED_PROJ=qkv,o; PackedDecodeMatmul, bitwise the stock matmul)."""
        self.device = mesh_device
        self.config = config
        self.optimized_27b = optimized_27b_attention()
        if self.optimized_27b and (config.num_heads, config.num_kv_heads, config.head_dim) != (24, 4, 256):
            raise ValueError("Optimized 27B attention requires Q24/KV4/head256")

        self.weights = load_attention_weights(mesh_device, state_dict, tensor_cache_path)
        self.decode_program_configs = {}
        self.batched_program_configs = {}
        if self.optimized_27b or os.environ.get("QWEN36_SINGLE_1D_DECODE") == "1":
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            worker_grid = mesh_device.compute_with_storage_grid_size()
            grid_w = min(int(os.environ.get("QWEN36_DECODE_GRID_W", "11")), worker_grid.x)
            if self.optimized_27b:
                if grid_w < 8 or worker_grid.y < 10:
                    raise ValueError("Optimized 27B attention requires at least an 8x10 worker grid")
                specs = {
                    "q": (5120, 12288, 64), "k": (5120, 1024, 8),
                    "v": (5120, 1024, 8), "o": (6144, 5120, 44),
                    "qkv": (5120, 14336, 64),
                }
            else:
                specs = {
                    name: (weight.shape[-2], weight.shape[-1], cores)
                    for name, weight, cores in (
                        ("q", self.weights.q_proj, 44), ("k", self.weights.k_proj, 8),
                        ("v", self.weights.v_proj, 8), ("o", self.weights.o_proj, 33),
                        ("qkv", self.weights.qkv_fused, 64),
                    ) if weight is not None
                }
            self.decode_program_configs = {
                name: tpc.create_matmul_1d_decode_progcfg(
                    1, k, n, num_cores=num_cores, grid_w=grid_w,
                )
                for name, (k, n, num_cores) in specs.items()
            }
            if self.optimized_27b:
                # Double-buffered BF16 in0/BF8 in1, BF16 output + FP32 partials:
                # 2*4*2048 + 2*4*7*1088 + 7*(2048+4096) = 120320 B/core max.
                for pc in self.decode_program_configs.values():
                    pc.in0_block_w = 4

        self.decode_runners = {}
        if pack_decode and not self.optimized_27b and os.environ.get("QWEN36_SINGLE_1D_DECODE") == "1":
            from models.demos.blackhole.qwen36.tt.packed_matmul import packed_decode_runner, packed_projections

            decode_ckc = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True,
            )
            for name, weight in (("qkv", self.weights.qkv_fused), ("o", self.weights.o_proj)):
                if (name in packed_projections() and weight is not None and name in self.decode_program_configs
                        and weight.dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b)):
                    self.decode_runners[name] = packed_decode_runner(
                        weight, self.decode_program_configs[name], decode_ckc
                    )

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
        mc = ttnn.DRAM_MEMORY_CONFIG if self.optimized_27b else (ttnn.L1_MEMORY_CONFIG if T == 1 else None)
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
                decode_runners=self.decode_runners,
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

    def forward_batched(self, x, cos, sin, position_tensor, page_table):
        """Concurrent single-device decode, including its B1 bucket.

        Residuals are [1,1,B,2048], rotations [1,1,B,64], device positions
        [B] (-1 inactive), and page tables [B,width]. Unlike verify_many,
        rows are independent requests with independently writable KV pages.
        """
        from models.demos.blackhole.qwen36.tt.attention.batched import batched_decode_forward

        return batched_decode_forward(self, x, cos, sin, position_tensor, page_table)

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
            qkv_fused_q_dim=2 * self.config.num_heads * self.config.head_dim,
            qkv_fused_k_dim=self.config.num_kv_heads * self.config.head_dim,
            qkv_fused_program_config=self.decode_program_configs.get("qkv"),
            verify_two=True,
            decode_runners=self.decode_runners,
            shadow_fill_page_table=shadow_fill_page_table,
            shadow_source_page=shadow_source_page,
            shadow_same_page=shadow_same_page,
        )
        return output, (key1, value1)

    def verify_many(self, x, cos, sin, position_tensor, page_table, fork_position=0):
        """Verify 2..32 causal rows, leaving live KV unchanged until commit.

        QKV and output projections run once with the decode accumulation config.
        The returned single-use transaction owns candidate K/V and metadata.
        """
        from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
            gated_attention_forward_ttnn,
        )

        if not self.use_paged_attention or len(x.shape) != 3 or x.shape[0] != 1:
            raise ValueError("verify_many requires paged attention and [1,T,hidden] input")
        if not 2 <= int(x.shape[1]) <= 32:
            raise ValueError("verify_many supports 2..32 rows within one tile")
        output, transaction, _ = gated_attention_forward_ttnn(
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
            qkv_fused_q_dim=2 * self.config.num_heads * self.config.head_dim,
            qkv_fused_k_dim=self.config.num_kv_heads * self.config.head_dim,
            qkv_fused_program_config=self.decode_program_configs.get("qkv"),
            verify_many=True,
            tree_fork_position=fork_position,
        )
        return output, transaction

    def commit_verify_many(self, transaction, accepted_count, position_tensor, page_table):
        """Commit only rows 0..accepted_count into the stable live KV buffers."""
        from models.demos.blackhole.qwen36.tt.attention.verify import commit_verify_many

        commit_verify_many(
            transaction, accepted_count, position_tensor, page_table,
            self.paged_kv_cache_key, self.paged_kv_cache_value,
        )

    def reset_cache(self):
        """Clear the concat KV cache for a new sequence."""
        self.past_key = None
        self.past_value = None

    def set_paged_kv_cache(self, k_cache, v_cache):
        """Attach externally-allocated paged KV cache (called once after allocate_kv_cache)."""
        if self.optimized_27b:
            for cache in (k_cache, v_cache):
                if (
                    cache.dtype != ttnn.bfloat16 or cache.layout != ttnn.TILE_LAYOUT
                    or len(cache.shape) != 4 or cache.shape[1] != 4 or cache.shape[-1] != 256
                    or cache.shape[2] % 32 != 0
                    or cache.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                ):
                    raise ValueError("Optimized 27B KV cache must be tiled BF16 DRAM [blocks,4,page,256]")
        self.paged_kv_cache_key = k_cache
        self.paged_kv_cache_value = v_cache
        self.use_paged_attention = True
