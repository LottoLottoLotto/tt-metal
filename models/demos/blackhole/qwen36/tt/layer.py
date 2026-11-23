# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""

import os
import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.tt.rms_norm import bounded_norm_enabled, get_decode_norm_config
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.common import Mode


class Qwen36DecoderLayer:
    """Single transformer layer with hybrid attention dispatch.

    Pattern: x → attention_norm → attention → residual → ff_norm → MLP → residual
    Attention is either GatedAttention (full, with RoPE) or GatedDeltaNet (linear).
    """

    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        layer_num,
        tensor_cache_path=None,
        tt_ccl=None,
        state_prefix=None,
        force_full_attention=False,
    ):
        self.layer_num = layer_num
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        self._batched_single_device = self.num_devices == 1 and args.max_batch_size > 1
        grid = mesh_device.compute_with_storage_grid_size()
        # One-launch residual add + width-sharded RMSNorm (FusedResidualNorm): 27B under the bounded-norm
        # profile; 35B (dim 2048, native 8x8 [32,32] decode norm) whenever the generic baseline is not forced.
        self._packed_norm_enabled = (
            self.num_devices == 1 and (grid.x, grid.y) == (11, 10)
            and (
                (bounded_norm_enabled() and args.dim == 5120)
                or (args.dim == 2048 and os.environ.get("QWEN36_GENERIC_BASELINE") != "1")
            )
        )
        self._packed_norm_runners = {}
        self.is_full_attention = force_full_attention or args.is_full_attention_layer(layer_num)

        prefix = state_prefix or f"layers.{layer_num}"

        # Zero-centered RMSNorm (Qwen3.5): output = x_normed * (1 + weight). The
        # framework RMSNorm applies the +1 internally via add_unit_offset=True and
        # is mesh-aware (replicates the weight across a MeshDevice).
        #
        # Single device: plain RMSNorm on the full hidden state (validated path).
        # TP (27B on a (1,4) mesh): the residual stream is fractured along the
        # hidden dim, so each norm is wrapped in the framework DistributedNorm,
        # which all-gathers (PREFILL: distributed rmsnorm + gather; DECODE:
        # gather-then-norm) to hand the modules a replicated full-dim input —
        # exactly as models/demos/qwen35_27b does via the framework decoder.
        # Prefill fuses the norm all-gather into the in-proj matmul (all_gather_minimal_matmul_async):
        # GDN qkvzab and full-attn QKV. attention_norm then skips its post-norm AG (prefill only;
        # decode gathers pre-norm). Gates must match the module-side _fuse_agmm gates.
        self._fuse_norm_agmm = self.num_devices > 1 and (
            (not self.is_full_attention and getattr(args, "gdn_qkvz_weight_memcfg", None) is not None)
            or (self.is_full_attention and getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None)
        )
        self.attention_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            prefix,
            "input_layernorm",
            tensor_cache_path,
            tt_ccl,
            "attention_norm",
            enable_all_gather=not self._fuse_norm_agmm,
        )
        # Prefill: ff_norm skips AG (fused into gate/up AGMM); decode gathers pre-norm so this is a no-op there.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
        self.ffn_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            prefix,
            "post_attention_layernorm",
            tensor_cache_path,
            tt_ccl,
            "ff_norm",
            enable_all_gather=not self._fuse_ff_agmm,
        )


        if self.num_devices > 1:
            # Tensor-parallel modules (sharded weights from the raw substate).
            # Cache the sharded mesh weights to disk so re-runs skip the (slow,
            # single-threaded) reorder+shard of the full 27B.
            tp_cache = (tensor_cache_path / prefix / "tp") if tensor_cache_path else None
            if self.is_full_attention:
                from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp

                tw = load_attention_weights_tp(
                    mesh_device, substate(state_dict, f"{prefix}.self_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPAttention(mesh_device, args, tw, tt_ccl)
            else:
                from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp

                tw = load_gdn_weights_tp(
                    mesh_device, substate(state_dict, f"{prefix}.linear_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
        elif self.is_full_attention:
            attn_state = substate(state_dict, f"{prefix}.self_attn")
            attn_cache = (tensor_cache_path / prefix) if tensor_cache_path else None
            self.attention = Qwen36GatedAttention(
                mesh_device, AttentionConfig.from_args(args), attn_state, attn_cache,
                pack_decode=self.num_devices == 1 and not self._batched_single_device and args.dim == 2048,
            )
        else:
            gdn_state = substate(state_dict, f"{prefix}.linear_attn")
            gdn_cache = (tensor_cache_path / prefix) if tensor_cache_path else None
            self.attention = Qwen36GatedDeltaNet(
                mesh_device, GDNConfig.from_args(args), gdn_state, gdn_cache,
                pack_decode_mega=(
                    self.num_devices == 1 and args.dim == 2048
                    and os.environ.get("QWEN_GDN_MEGA_PACKED") != "1"
                ),
            )


        mlp_state = substate(state_dict, f"{prefix}.mlp")
        mlp_cache = (tensor_cache_path / prefix) if tensor_cache_path else None
        if getattr(args, "is_moe", False):
            from models.demos.blackhole.qwen36.tt.moe import Qwen36MoE

            self.feed_forward = Qwen36MoE(mesh_device, mlp_state, args, mlp_cache)
        else:
            self.feed_forward = Qwen36MLP(mesh_device, mlp_state, mlp_cache, args=args, tt_ccl=tt_ccl)

    def _make_norm(
        self,
        mesh_device,
        args,
        state_dict,
        state_prefix,
        weight_key,
        tensor_cache_path,
        tt_ccl,
        ag_key,
        enable_all_gather=True,
    ):
        """Build the per-layer RMSNorm; wrap in DistributedNorm when TP>1.

        On a single device this returns the same plain RMSNorm the validated 9B
        path used. The DistributedNorm wrapper (TP>1) mirrors tt_transformers
        decoder.py and handles the fractured->replicated transition.
        """
        norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key=weight_key,
            state_dict_prefix=f"{state_prefix}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=tt_ccl)
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            return DistributedNorm(
                norm, args, tt_ccl=tt_ccl, TG=args.is_galaxy, ag_config_key=ag_key, enable_all_gather=enable_all_gather
            )
        return norm

    def verify_two(
        self,
        x,
        cos,
        sin,
        position_tensor,
        page_table,
        shadow_fill_page_table=None,
        shadow_source_page=0,
        shadow_same_page=True,
    ):
        """Execute this layer at M=2 without committing speculative state."""
        attn_input = self._decode_norm(self.attention_norm, x)
        if self.is_full_attention:
            attn_output, candidates = self.attention.verify_two(
                attn_input,
                cos,
                sin,
                position_tensor,
                page_table,
                shadow_fill_page_table,
                shadow_source_page,
                shadow_same_page,
            )
        else:
            attn_output, recurrent_candidates, conv_candidates = self.attention.verify_two(attn_input)
            candidates = (recurrent_candidates, conv_candidates)
        ttnn.deallocate(attn_input)
        ff_input, h = self._decode_residual_norm(self.ffn_norm, x, attn_output)
        ttnn.deallocate(attn_output)
        # Keep both verification tokens in one tile row throughout the MLP.
        # Moving them to [2, 1, D] would materialize two separately padded rows.
        ff_output = self.feed_forward.forward(ttnn.reshape(ff_input, [1, 1, 2, ff_input.shape[-1]]))
        ttnn.deallocate(ff_input)
        output = ttnn.add(h, ttnn.reshape(ff_output, [1, 2, ff_output.shape[-1]]))
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)
        return output, candidates

    def verify_many(self, x, cos, sin, position_tensor, page_table, fork_position=0):
        """One shared tiled layer pass with transactional state for every prefix."""
        rows = int(x.shape[-2])
        if not 3 <= rows <= 32:
            raise ValueError("Wider verification requires 3..32 positions")
        attn_input = self._decode_norm(self.attention_norm, x)
        if self.is_full_attention:
            attn_output, candidates = self.attention.verify_many(
                attn_input, cos, sin, position_tensor, page_table, fork_position=fork_position
            )
        else:
            attn_output, states, conv = self.attention.verify_many(attn_input, fork_position=fork_position)
            candidates = (states, conv)
        ttnn.deallocate(attn_input)
        ff_input, h = self._decode_residual_norm(self.ffn_norm, x, attn_output)
        ttnn.deallocate(attn_output)
        ff_output = self.feed_forward.forward(
            ttnn.reshape(ff_input, [1, 1, rows, ff_input.shape[-1]])
        )
        ttnn.deallocate(ff_input)
        output = ttnn.add(h, ttnn.reshape(ff_output, [1, rows, ff_output.shape[-1]]))
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)
        return output, candidates

    def _decode_norm(self, norm, x):
        """Width-sharded RMSNorm for packed decode and verification rows.

        All supported packed M<=32 shapes occupy one tile row. The helper
        divides the full hidden width exactly and bounds per-core scratch.
        """
        if self._can_fuse_decode_norm(x):
            return self._fused_decode_norm(norm, x)[0]
        nc = get_decode_norm_config(self.args)
        shard_mc = nc["sharded_output_config"]
        rows = x.shape[-2]
        dim = x.shape[-1]
        sharded = ttnn.to_memory_config(ttnn.reshape(x, [1, 1, rows, dim]), shard_mc)
        normed = ttnn.rms_norm(
            sharded,
            epsilon=norm.eps,
            weight=norm.weight,
            program_config=nc["sharded_program_config"],
            memory_config=shard_mc,
            compute_kernel_config=norm.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(sharded)
        output = ttnn.reshape(ttnn.sharded_to_interleaved(normed, ttnn.DRAM_MEMORY_CONFIG), list(x.shape))
        ttnn.deallocate(normed)
        return output

    def _can_fuse_decode_norm(self, x):
        shape = tuple(x.shape)
        if len(shape) == 4 and shape[:2] == (1, 1) and self._batched_single_device and self.args.dim == 2048:
            # Batched decode rows [1,1,B,2048] are the same single tile row as [1,B,2048].
            shape = shape[1:]
        return (
            self._packed_norm_enabled and len(shape) == 3
            and shape[0] == 1 and 1 <= shape[1] <= 32 and shape[2] == self.args.dim
        )

    def _fused_decode_norm(self, norm, x, residual=None):
        # The same complete four BF16 tiles per width shard represent M=1,
        # MTP's M=2, DFlash/tree M<=32 and the batched decode buckets. No token interaction is added.
        shape = list(x.shape)
        if len(shape) == 4:
            flat = shape[1:]
            normed, summed = self._fused_decode_norm(
                norm, ttnn.reshape(x, flat), None if residual is None else ttnn.reshape(residual, flat)
            )
            return ttnn.reshape(normed, shape), (x if residual is None else ttnn.reshape(summed, shape))
        runner = self._packed_norm_runners.get(norm)
        if runner is None:
            from models.demos.blackhole.qwen36.tt.residual_norm.fused_residual_norm import FusedResidualNorm
            runner = FusedResidualNorm(
                self.device, get_decode_norm_config(self.args), norm.compute_kernel_config_hifi2
            )
            self._packed_norm_runners[norm] = runner
        return runner(x, norm.weight, norm.eps, residual=residual)

    def _decode_residual_norm(self, norm, x, residual):
        if self._can_fuse_decode_norm(x):
            if self.args.dim == 5120:
                return self._fused_decode_norm(norm, x, residual)
            # 35B: the in-launch add (separate add cores + semaphore) measured 44 us vs 18 us stock (runs/micro1
            # normbench); the stock add followed by the one-launch norm (7.3 vs 12.5 us) is bitwise the same.
            summed = ttnn.add(x, residual)
            return self._fused_decode_norm(norm, summed)[0], summed
        summed = ttnn.add(x, residual)
        return self._decode_norm(norm, summed), summed

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,  # = GDN long_prefill_chunk_size; the only size the chunk-seq prefill kernel supports
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
        gdn_collect=False,
        final_norm=None,
        moe_valid_rows=None,
    ):
        """moe_valid_rows (prefill only): the MoE ``prefill_valid_rows`` page for masked
        buckets; padding rows then take no expert work."""
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        if self.num_devices > 1:
            # TP: DistributedNorm uses the framework's per-norm memory configs.
            _attn_norm_config = self.args.get_norm_config("attn", _norm_mode)
            # PREFILL: distributed rmsnorm outputs in L1 so the fused in-proj AGMM gathers from L1, not DRAM.
            if _norm_mode == Mode.PREFILL:
                _attn_norm_config = {**_attn_norm_config, "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG}
            # DECODE ff_norm uses the attn_norm layout (act_shard_hidden, 32-core) so Qwen36MLP's input reshard is a no-op and the norm runs on 32 cores not 8; PREFILL keeps the framework ff config.
            if _norm_mode == Mode.DECODE:
                _ff_norm_config = self.args.get_norm_config("attn", _norm_mode)
            else:
                # ff_norm output stays DRAM: L1 keeps the full-width norm resident across the whole MLP,
                # clashing with each matmul's CBs (w1/w3/w2) for no gain. Verified dead end; keep DRAM.
                _ff_norm_config = self.args.get_norm_config("ff", _norm_mode)
        else:
            # Keep the generic decode output in DRAM. Full-27B framework norms
            # additionally use bounded sharded chunks, including prefill;
            # changing only output placement cannot bound the native CBs.
            _attn_norm_config = _ff_norm_config = (
                {"output_mem_config": ttnn.DRAM_MEMORY_CONFIG} if mode == "decode" else None
            )
        single_request_decode = (
            os.environ.get("QWEN36_GENERIC_BASELINE") != "1"
            and mode == "decode" and self.num_devices == 1 and x.shape[0] == 1
        )
        attn_input = (
            self._decode_norm(self.attention_norm, x)
            if single_request_decode
            else self.attention_norm(x, mode=_norm_mode, norm_config=_attn_norm_config)
        )

        if self.num_devices > 1:
            # TP modules: input is the gathered (full-dim) norm output [1,1,B/S,dim];
            # output is fractured along dim=3. cos/sin are in rope_tp format.
            if self.is_full_attention:
                if mode == "prefill":
                    # Contract/vLLM path supplies a page_table → paged KV prefill; the
                    # demo path (no page_table) uses the internal concat caches.
                    if page_table is not None:
                        attn_output = self.attention.forward_prefill_paged(
                            attn_input,
                            cos,
                            sin,
                            page_table,
                            chunk_page_table=chunk_page_table,
                            chunk_start_idx=chunk_start_idx if chunk_start_idx is not None else 0,
                            chunk_start_idx_tensor=chunk_start_idx_tensor,
                        )
                    else:
                        attn_output = self.attention.forward_prefill(attn_input, cos, sin)
                else:
                    attn_output = self.attention.forward_decode(
                        attn_input, position_tensor, cos, sin, page_table=page_table
                    )
            else:
                # GDN carries its recurrent/conv state internally (capture_state on
                # prefill, read on decode); it has no paged KV, so page_table is N/A.
                if mode == "prefill":
                    if gdn_collect:
                        # Batched per-user prefill: stash this user's from-scratch state for
                        # assembly into row u of the batched buffers (finalize_pending later).
                        attn_output = self.attention.forward_prefill_collect(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len
                        )
                    else:
                        attn_output = self.attention.forward_prefill(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len, capture_state=True
                        )
                else:
                    attn_output = self.attention.forward_decode(attn_input)
        elif self._batched_single_device and mode == "decode":
            if self.is_full_attention:
                attn_output = self.attention.forward_batched(attn_input, cos, sin, position_tensor, page_table)
            else:
                attn_output = self.attention.forward_batched(attn_input, position_tensor)
        elif self.is_full_attention:
            attn_output = self.attention.forward(
                attn_input,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)

        if single_request_decode:
            ff_input, h = self._decode_residual_norm(self.ffn_norm, x, attn_output)
            ttnn.deallocate(attn_output)
        else:
            h = ttnn.add(x, attn_output)
            ttnn.deallocate(attn_output)
            ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_ff_norm_config)

        if self._batched_single_device and mode == "decode":
            ff_output = self.feed_forward.forward_batched(ff_input)
        elif moe_valid_rows is not None and mode == "prefill":
            ff_output = self.feed_forward.forward(ff_input, valid_rows=moe_valid_rows)
        else:
            ff_output = self.feed_forward.forward(ff_input)
        ttnn.deallocate(ff_input)

        if final_norm is None:
            output = ttnn.add(h, ff_output)
        else:
            output = final_norm(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output
