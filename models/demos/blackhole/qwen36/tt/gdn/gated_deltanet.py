# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5-9B Gated DeltaNet layer — composes config/weights/state/prefill/decode.

Wraps the experimental ``gated_deltanet_forward_ttnn()`` and the on-device GDN prefill
kernel into a module that manages weight tensors, recurrent state, and conv state.
"""
import os
from loguru import logger
import ttnn
from models.demos.blackhole.qwen36.tt.gdn.config import GDNConfig
from models.demos.blackhole.qwen36.tt.gdn.decode import recurrent_forward
from models.demos.blackhole.qwen36.tt.gdn.state import init_recurrent_state, restore_split_conv_from_fused
from models.demos.blackhole.qwen36.tt.gdn.weights import load_gdn_weights

_LEGACY_NATIVE_FLAGS = (
    "QWEN36_GDN_RECURRENT_FUSED",
    "QWEN36_GDN_FRONTEND_FUSED",
    "QWEN36_GDN_MEGA_CORES",
)


def _effective_native_gdn_config():
    legacy = [name for name in _LEGACY_NATIVE_FLAGS if name in os.environ]
    if legacy:
        raise RuntimeError(
            f"Obsolete native GDN environment variable(s): {', '.join(legacy)}; "
            "use QWEN_GDN_RECURRENT_FUSED and QWEN_GDN_FRONTEND_FUSED"
        )
    recurrent = os.environ.get("QWEN_GDN_RECURRENT_FUSED", "0") == "1"
    frontend = os.environ.get("QWEN_GDN_FRONTEND_FUSED", "0") == "1"
    if frontend and not recurrent:
        raise RuntimeError("QWEN_GDN_FRONTEND_FUSED=1 requires QWEN_GDN_RECURRENT_FUSED=1")
    output = os.environ.get("QWEN_GDN_OUTPUT_FUSED", "0") == "1"
    if output and not frontend:
        raise RuntimeError("QWEN_GDN_OUTPUT_FUSED=1 requires QWEN_GDN_FRONTEND_FUSED=1")
    return recurrent, frontend, output


class Qwen36GatedDeltaNet:
    """Gated DeltaNet (linear attention) layer for Qwen3.5-9B.

    Maintains fixed-size recurrent state [B, H, K, V] that replaces the KV cache.
    Also maintains conv states [B, kernel_size-1, D] for causal conv1d history.
    Supports two modes:
      - "recurrent": single-token decode (T=1), O(1) memory
      - "chunk": multi-token prefill (T>1), chunked parallel processing
    """
    _logged_effective_config = False
    _shared_native_output_value = None
    _shared_native_output_value_m2 = None
    _shared_native_frontend_many = {}
    _shared_verify_many_inbox = {}
    _shared_m1_inbox = {}

    def __init__(self, mesh_device, config: GDNConfig, state_dict, tensor_cache_path=None):
        self.device = mesh_device
        self.cfg = config

        # Mirror config-derived scalar dims so the forward bodies read them directly.
        self.num_heads = config.num_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.norm_eps = config.norm_eps
        self.long_prefill_chunk_size = config.long_prefill_chunk_size

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

        self.native_recurrent, self.native_frontend, self.native_output = _effective_native_gdn_config()
        if self.native_output and config.output_gate_type not in ("swish", "silu"):
            raise ValueError("Native GDN output supports only the Swish gate")
        self.weights = load_gdn_weights(mesh_device, config, state_dict, tensor_cache_path)
        self.native_frontend_output = None
        self.native_frontend_output_m2 = None
        self.native_frontend_output_many = {}
        self._fused_verify_output_many = {}
        self._fused_m1_output = None
        self.native_output_output = None
        self.native_output_output_m2 = None
        self.mega_decode_program_config = None
        self.out_decode_program_config = None
        if os.environ.get("QWEN36_SINGLE_1D_DECODE") == "1":
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            grid_w = int(os.environ.get("QWEN36_DECODE_GRID_W", "11"))
            if self.weights.mega_fused_weight_decode is not None:
                if (
                    os.environ.get("QWEN_GDN_MEGA_PACKED") == "1"
                    and self.weights.mega_fused_weight_decode.dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b)
                ):
                    self.mega_decode_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(11, 10),
                        in0_block_w=(
                            2
                            if tuple(self.weights.mega_fused_weight_decode.shape) == (5120, 16512)
                            and self.weights.mega_fused_weight_decode.dtype == ttnn.bfloat8_b
                            else 4
                        ),
                        out_subblock_h=1,
                        out_subblock_w=2,
                        per_core_M=1,
                        per_core_N=6,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=True,
                    )
                else:
                    self.mega_decode_program_config = tpc.create_matmul_1d_decode_progcfg(
                        1,
                        self.weights.mega_fused_weight.shape[-2],
                        self.weights.mega_fused_weight.shape[-1],
                        num_cores=int(os.environ.get("QWEN_GDN_MEGA_CORES", "44")),
                        grid_w=grid_w,
                    )
            self.out_decode_program_config = tpc.create_matmul_1d_decode_progcfg(
                1,
                self.weights.o_proj_weight.shape[-2],
                self.weights.o_proj_weight.shape[-1],
                num_cores=(
                    tpc.proj_o_decode_cores(33)
                    if tuple(self.weights.o_proj_weight.shape) == (6144, 5120)
                    and self.weights.o_proj_weight.dtype == ttnn.bfloat8_b
                    else 33
                ),
                grid_w=grid_w,
            )
        if not Qwen36GatedDeltaNet._logged_effective_config:
            logger.info(
                "GDN effective config: implementation={} recurrent_fused={} frontend_fused={} output_fused={} "
                "value_heads={} persistent_state_placement={}",
                "native" if self.native_recurrent else "generic",
                self.native_recurrent,
                self.native_frontend,
                self.native_output,
                self.num_v_heads,
                "L1" if self.native_recurrent else "DRAM",
            )
            Qwen36GatedDeltaNet._logged_effective_config = True

        # ---- Runtime state (plain instance attributes, exact same names as before;
        # poked directly by the trace machinery in model.py / qwen36_vllm.py) ----
        self.recurrent_state = None
        # Conv states: ttnn tensors on device [B, kernel_size-1, D]
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        # Fused conv state [B, kernel_size-1, D_total] where D_total = q_dim + k_dim + v_dim
        self.fused_conv_state = None
        self.split_conv_state = None
        # Trace capture support
        self.use_inplace_state = False
        # When True (set during chunk-outer traced-prefill capture), the chunk (prefill)
        # path writes recurrent + conv state into the persistent external buffers IN PLACE
        # (ttnn.copy) instead of reassigning a fresh tensor, so the state carries across
        # execute_trace() replays (each replay re-runs the same baked buffer addresses).
        # Eager prefill keeps the reassign path. See Qwen36Model.capture_prefill_trace_chunked.
        self._chunk_inplace_state = False

    def forward(self, x, mode="recurrent", chunk_size=None, valid_len=None):
        return recurrent_forward(self, x, mode=mode, chunk_size=chunk_size, valid_len=valid_len)

    def verify_two(self, x):
        """Return M=2 output plus (S1,S2) recurrent and convolution candidates."""
        return recurrent_forward(self, x, mode="verify_two")

    def verify_many(self, x, fork_position=0, users=None, groups=None):
        """Return output and post-row candidate tuples, leaving live state unchanged.

        Recurrent candidates are transient DRAM allocations consumed by commit.
        Convolution candidates and frontend operands belong to the layer/runner;
        callers must not deallocate them.

        users=(recurrent_states, conv_states): row t is user t instead; each row reads and
        updates its own DRAM state pair and no candidates are returned.
        """
        if len(x.shape) != 3 or x.shape[0] != 1 or x.shape[2] != 5120 or not 2 <= int(x.shape[1]) <= 32:
            raise ValueError("GDN verify_many requires [1,T,5120], 2 <= T <= 32")
        if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
            raise ValueError("GDN verify_many requires BF16 TILE input")
        if not (self.native_recurrent and self.native_frontend and self.native_output):
            raise RuntimeError("GDN verify_many requires the exact native recurrent/frontend/output path")
        if self.out_decode_program_config is None or self.mega_decode_program_config is None:
            raise RuntimeError("GDN verify_many requires the packed mega and decode output projection configs")
        return recurrent_forward(self, x, mode="verify_many", fork_position=fork_position, users=users,
                                 groups=groups)

    def commit_verify_many(self, accepted_count, recurrent_candidates, convolution_candidates):
        """Copy candidate accepted_count into the stable live L1 state addresses."""
        if not hasattr(self, "_fused_mtp_commit"):
            from models.demos.blackhole.qwen36.tt.gdn.fused_commit import FusedCommit
            self._fused_mtp_commit = FusedCommit()
        self._fused_mtp_commit.commit_many(
            accepted_count, recurrent_candidates, convolution_candidates,
            self.recurrent_state, self.fused_conv_state,
        )
        for candidate in recurrent_candidates:
            ttnn.deallocate(candidate)

    def set_external_state(self, recurrent_state, conv_state):
        """Point layer at externally-allocated state buffers.
        Sets use_inplace_state=True so all forward passes write state inplace (preserving buffer addresses).
        Does NOT create split_conv_state — that happens after prefill when there is real data to split.
        """
        recurrent_shape = list(recurrent_state.shape)
        conv_shape = list(conv_state.shape)
        expected_tail = [self.num_v_heads, self.head_k_dim, self.head_v_dim]
        assert recurrent_shape[1:] == expected_tail, (
            f"recurrent_state shape mismatch: {recurrent_shape} "
            f"does not end in {expected_tail}"
        )
        assert recurrent_shape[0] == conv_shape[0], (
            f"state batch mismatch: recurrent={recurrent_shape[0]}, "
            f"conv={conv_shape[0]}"
        )
        assert conv_state.shape[1] == self.conv_kernel_size - 1, (
            f"conv_state dim 1 mismatch: {conv_state.shape[1]} != "
            f"{self.conv_kernel_size - 1}"
        )
        assert conv_state.dtype == ttnn.bfloat16, "convolution state must use BF16"
        if self.native_recurrent:
            assert recurrent_state.dtype == ttnn.bfloat16, "native recurrence requires BF16"
            assert recurrent_state.layout == ttnn.ROW_MAJOR_LAYOUT, "native recurrence requires row-major state"
            assert recurrent_state.memory_config().buffer_type == ttnn.BufferType.L1
            assert conv_state.memory_config().buffer_type == ttnn.BufferType.L1
        else:
            assert recurrent_state.dtype in (ttnn.float32, ttnn.bfloat16)
            assert recurrent_state.layout == ttnn.TILE_LAYOUT, "generic recurrence requires tiled state"
            assert recurrent_state.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            assert conv_state.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        self.recurrent_state = recurrent_state
        self.fused_conv_state = conv_state
        self._persistent_state_addresses = (
            int(recurrent_state.buffer_address()),
            int(conv_state.buffer_address()),
        )
        self.use_inplace_state = True

    def _restore_split_conv_from_fused(self):
        """Copy fused_conv_state slices into existing split_conv_state buffers.
        Preserves device addresses (critical for trace replay).
        Kept as a method because model.py calls it on the instance.
        """
        restore_split_conv_from_fused(self)

    def reset_state(self, batch_size=None):
        if batch_size is not None:
            init_recurrent_state(self, batch_size)
        else:
            self.recurrent_state = None
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        self.fused_conv_state = None
        self.split_conv_state = None
