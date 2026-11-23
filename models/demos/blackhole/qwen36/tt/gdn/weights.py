# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Load quantized GDN projections and precompute convolution/decay constants.

Optimized 27B keeps one canonical mega payload for prefill and T1/T2 decode;
the original separately quantized projections are released after fusion.
"""
from dataclasses import dataclass
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.config import GDNConfig
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt import weight_prefetch
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import create_chunk_masks_seq


@dataclass
class GDNWeights:
    # Projection weights
    qkv_proj_weight: ttnn.Tensor | None
    q_proj_weight: object  # None — dead split (op runs fused QKV); kept as kwarg
    k_proj_weight: object
    v_proj_weight: object
    a_proj_weight: ttnn.Tensor | None
    b_proj_weight: ttnn.Tensor | None
    g_proj_weight: ttnn.Tensor | None
    o_proj_weight: ttnn.Tensor
    # Conv weights (stay on HOST, ROW_MAJOR) + biases
    q_conv_weight: ttnn.Tensor
    k_conv_weight: ttnn.Tensor
    v_conv_weight: ttnn.Tensor
    q_conv_bias: object
    k_conv_bias: object
    v_conv_bias: object
    # 1D params
    A_log: ttnn.Tensor
    dt_bias: ttnn.Tensor
    o_norm_weight: ttnn.Tensor
    A_neg: ttnn.Tensor
    # Precomputed conv taps / bias on device
    q_weight_taps: list
    k_weight_taps: list
    v_weight_taps: list
    q_bias_dev: object
    k_bias_dev: object
    v_bias_dev: object
    fused_conv_weight_taps: list
    fused_conv_bias_dev: object
    native_frontend_weight_taps: object
    native_frontend_bias: object
    native_decay_bias: object
    native_decay_scale: object
    # Fused projection weights
    ab_proj_weight: ttnn.Tensor | None
    mega_fused_weight: object
    mega_fused_weight_decode: object
    mega_qkv_dim: object
    mega_a_dim: object
    mega_b_dim: object
    mega_g_dim: object
    # Chunk-parallel prefill kernel (gated_delta_attn_seq) — always on
    use_chunk_seq_prefill: bool
    chunk_seq_masks_long: object


def load_gdn_weights(mesh_device, config: GDNConfig, state_dict, tensor_cache_path=None) -> GDNWeights:
    """Load + precompute all device weights for one Gated DeltaNet layer.

    `state_dict` is the per-layer `linear_attn` substate (keys already stripped of
    the `layers.{n}.linear_attn.` prefix). `tensor_cache_path` points at the
    `layers.{n}` directory (or None) — cache file names re-add the `linear_attn.`
    prefix to match the original cache keys exactly.
    """
    num_heads = config.num_heads
    num_v_heads = config.num_v_heads
    head_k_dim = config.head_k_dim
    head_v_dim = config.head_v_dim
    conv_kernel_size = config.conv_kernel_size
    norm_eps = config.norm_eps
    generic_baseline = os.environ.get("QWEN36_GENERIC_BASELINE") == "1"

    linear_dtype = (
        ttnn.bfloat4_b
        if os.environ.get("QWEN36_BFP4_WEIGHTS") == "1"
        else ttnn.bfloat8_b
    )
    from models.demos.blackhole.qwen36.tt.model_config import evaluation_weight_dtype

    linear_dtype = evaluation_weight_dtype("gdn", tensor_cache_path, linear_dtype)
    output_dtype = evaluation_weight_dtype("gdn_output", tensor_cache_path, linear_dtype)
    def load_weight_2d(name):
        """Load 2D weight, transpose to [in, out] for ttnn.linear."""
        t = state_dict[name].T.contiguous()
        dtype = output_dtype if name == "out_proj.weight" else linear_dtype
        suffix = ""
        logical = t
        if name == "out_proj.weight" and not generic_baseline:
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            t, suffix = tpc.proj_o_rotating_pack(t, dtype)
        weight = ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"linear_attn.{name}{suffix}") if tensor_cache_path else None,
        )
        if (
            name == "out_proj.weight"
            and not generic_baseline
            and tuple(logical.shape) == (6144, 5120)
            and dtype == ttnn.bfloat8_b
            and weight_prefetch.enabled("proj_o")
        ):
            weight_prefetch.register(
                "proj_o", weight, logical, mesh_device,
                (tensor_cache_path / f"linear_attn.{name}.prefetch") if tensor_cache_path else None,
                in0_block_w=8,  # GDN out_decode_program_config's K block (M=1 decode and verification)
            )
        return weight

    def load_conv_weight(name):
        """Load conv1d weight — stays on HOST (not device), ROW_MAJOR layout."""
        t = state_dict[name]
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    def load_1d(name):
        """Load 1D param — must use TILE_LAYOUT on device like all other tensors."""
        t = state_dict[name]
        return ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"linear_attn.{name}") if tensor_cache_path else None,
        )

    # Fused QKV projection: one matmul instead of three
    qkv_key = "qkv_proj.weight"
    if qkv_key not in state_dict:
        raise ValueError(
            f"DeltaNet layer requires the combined qkv_proj weight "
            f"(key '{qkv_key}' missing; the split q/k/v_proj were removed in the weight refactor)."
        )
    t = state_dict[qkv_key].T.contiguous()  # [hidden, q_dim+k_dim+v_dim]
    qkv_proj_weight = ttnn.as_tensor(
        t,
        dtype=linear_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=(tensor_cache_path / "linear_attn.qkv_proj.weight") if tensor_cache_path else None,
    )
    # The operation receives these only for its split-projection fallback,
    # which this loader never selects.
    q_proj_weight = None
    k_proj_weight = None
    v_proj_weight = None
    a_proj_weight = load_weight_2d("in_proj_a.weight")
    b_proj_weight = load_weight_2d("in_proj_b.weight")
    g_proj_weight = load_weight_2d("in_proj_z.weight")
    o_proj_weight = load_weight_2d("out_proj.weight")

    q_conv_weight = load_conv_weight("q_conv.weight")
    k_conv_weight = load_conv_weight("k_conv.weight")
    v_conv_weight = load_conv_weight("v_conv.weight")

    def load_conv_bias_or_none(name):
        if name in state_dict:
            t = state_dict[name]
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        return None

    q_conv_bias = load_conv_bias_or_none("q_conv.bias")
    k_conv_bias = load_conv_bias_or_none("k_conv.bias")
    v_conv_bias = load_conv_bias_or_none("v_conv.bias")

    A_log = load_1d("A_log")
    dt_bias = load_1d("dt_bias")
    # DeltaNet output norm uses STANDARD RMSNorm (raw weights ~0.88),
    # NOT zero-centered like the decoder/attention norms (raw weights ~0.03).
    o_norm_weight = load_1d("norm.weight")

    # Precompute A_neg = -exp(A_log) once (constant per layer, saves 2 ops per decode step)
    A_neg = ttnn.neg(ttnn.exp(A_log))

    # ---- precompute helpers (bodies verbatim; self.device -> mesh_device, self.<dim> -> config.<dim>) ----

    def _precompute_weight_taps(conv_weight):
        """Pre-slice conv weight [D, 1, K] into K device tensors [1, 1, D] for FIR decode."""

        weight_torch = ttnn.to_torch(conv_weight)
        D = weight_torch.shape[0]
        taps = []
        for k in range(conv_kernel_size):
            w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
            taps.append(ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device))
        return taps

    def _precompute_bias_dev(conv_bias):
        """Pre-convert conv bias to [1, 1, D] device tensor."""
        if conv_bias is None:
            return None

        bias_torch = ttnn.to_torch(conv_bias)
        D = bias_torch.numel()
        bias_reshaped = bias_torch.reshape(1, 1, D).contiguous()
        return ttnn.from_torch(bias_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    def _precompute_fused_weight_taps():
        """Pre-concatenate Q+K+V conv weight taps into fused taps [1, 1, D_total]."""
        q_w = ttnn.to_torch(q_conv_weight)  # [D_q, 1, K]
        k_w = ttnn.to_torch(k_conv_weight)  # [D_k, 1, K]
        v_w = ttnn.to_torch(v_conv_weight)  # [D_v, 1, K]
        # Concatenate along D dimension: [D_total, 1, K]
        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        D_total = fused_w.shape[0]
        taps = []
        for k_idx in range(conv_kernel_size):
            w_k = fused_w[:, 0, k_idx].reshape(1, 1, D_total).contiguous()
            taps.append(ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device))
        return taps

    def _precompute_fused_bias_dev():
        """Pre-concatenate Q+K+V conv biases into fused bias [1, 1, D_total]."""
        parts = []
        for bias in [q_conv_bias, k_conv_bias, v_conv_bias]:
            if bias is not None:
                parts.append(ttnn.to_torch(bias))
            else:
                return None  # If any is None, skip fused bias
        fused = torch.cat(parts, dim=0)
        D_total = fused.numel()
        fused_reshaped = fused.reshape(1, 1, D_total).contiguous()
        return ttnn.from_torch(fused_reshaped, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    def _precompute_native_frontend_weights():
        # Source-head ownership consumes the same flat BF16 taps as generic FIR.
        return fused_conv_weight_taps, fused_conv_bias_dev



    def _precompute_mega_fused_weight():
        """Fuse QKV + padded a + padded b + g projections.

        27B output split: [10240 QKV | 64 a | 64 b | 6144 gate].
        Only the first 48 entries of either scalar segment are logical.
        """
        if qkv_proj_weight is None:
            return None, None
        qkv_w = ttnn.to_torch(qkv_proj_weight)
        a_w = ttnn.to_torch(a_proj_weight)
        b_w = ttnn.to_torch(b_proj_weight)
        g_w = ttnn.to_torch(g_proj_weight)
        padded_heads = ((num_v_heads + 31) // 32) * 32
        a_w = torch.nn.functional.pad(a_w, (0, padded_heads - num_v_heads))
        b_w = torch.nn.functional.pad(b_w, (0, padded_heads - num_v_heads))
        fused = torch.cat([qkv_w, a_w, b_w, g_w], dim=1).contiguous()
        if os.environ.get("QWEN_GDN_MEGA_PACKED") == "1":
            if linear_dtype not in (ttnn.bfloat4_b, ttnn.bfloat8_b):
                raise ValueError("packed GDN mega requires block-floating projection weights")
            packed_geometry = (86, 6) if tuple(fused.shape) == (5120, 16512) else (97, 4)
            fused = tpc.repack_for_worker_banks(
                fused, workers=packed_geometry[0], full_cols=packed_geometry[1]
            )
        # One canonical device payload for decode AND prefill. The packed-reader
        # host dispatch must select the bank-page reader for every token count.
        weight = ttnn.from_torch(
            fused, dtype=linear_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return weight, weight

    # Precompute conv weight taps and bias on device to avoid CPU round-trips during decode
    q_weight_taps = _precompute_weight_taps(q_conv_weight)
    k_weight_taps = _precompute_weight_taps(k_conv_weight)
    v_weight_taps = _precompute_weight_taps(v_conv_weight)
    q_bias_dev = _precompute_bias_dev(q_conv_bias)
    k_bias_dev = _precompute_bias_dev(k_conv_bias)
    v_bias_dev = _precompute_bias_dev(v_conv_bias)

    # Precompute fused QKV conv weight taps [1, 1, D_total] for fused conv decode
    fused_conv_weight_taps = _precompute_fused_weight_taps()
    fused_conv_bias_dev = _precompute_fused_bias_dev()
    native_frontend_weight_taps, native_frontend_bias = (
        _precompute_native_frontend_weights()
        if os.environ.get("QWEN_GDN_FRONTEND_FUSED") == "1" else (None, None)
    )
    native_decay_bias = native_decay_scale = None
    if os.environ.get("QWEN_GDN_FRONTEND_FUSED") == "1":
        def head_constants(value):
            return ttnn.from_torch(
                ttnn.to_torch(value).reshape(num_v_heads, 1, 1).contiguous(),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        native_decay_bias = head_constants(dt_bias)
        native_decay_scale = head_constants(A_neg)

    # Mega replaces every a/b consumer; generic baseline uses individual weights.
    ab_proj_weight = None

    # One matmul supplies QKV, tile-aligned scalar segments, and the output gate.
    mega_fused_weight, mega_fused_weight_decode = (
        (None, None) if generic_baseline else _precompute_mega_fused_weight()
    )
    if mega_fused_weight is not None:
        mega_qkv_dim = config.q_dim + config.k_dim + config.v_dim
        mega_a_dim = ((num_v_heads + 31) // 32) * 32
        mega_b_dim = mega_a_dim
        mega_g_dim = num_v_heads * head_v_dim
    else:
        mega_qkv_dim = None
        mega_a_dim = None
        mega_b_dim = None
        mega_g_dim = None
    if mega_fused_weight is not None:
        # Prefill and both decode token counts consume mega, so do not retain
        # another full copy of QKV/a/b/g (~4.3 GB over the 27B GDN layers).
        for projection in (qkv_proj_weight, a_proj_weight, b_proj_weight, g_proj_weight):
            ttnn.deallocate(projection)
        qkv_proj_weight = a_proj_weight = b_proj_weight = g_proj_weight = None

    # Chunk-parallel prefill via the C++ gated_delta_attn_seq kernel (float32) is
    # the default prefill path. The kernel hardcodes Ct=4 diagonal blocks, so it
    # ONLY supports chunk_size=128 (= long_prefill_chunk_size). Precompute its
    # float32 masks (incl. eye_32) once.
    use_chunk_seq_prefill = True
    chunk_seq_masks_long = create_chunk_masks_seq(config.long_prefill_chunk_size, mesh_device)

    return GDNWeights(
        qkv_proj_weight=qkv_proj_weight,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        a_proj_weight=a_proj_weight,
        b_proj_weight=b_proj_weight,
        g_proj_weight=g_proj_weight,
        o_proj_weight=o_proj_weight,
        q_conv_weight=q_conv_weight,
        k_conv_weight=k_conv_weight,
        v_conv_weight=v_conv_weight,
        q_conv_bias=q_conv_bias,
        k_conv_bias=k_conv_bias,
        v_conv_bias=v_conv_bias,
        A_log=A_log,
        dt_bias=dt_bias,
        o_norm_weight=o_norm_weight,
        A_neg=A_neg,
        q_weight_taps=q_weight_taps,
        k_weight_taps=k_weight_taps,
        v_weight_taps=v_weight_taps,
        q_bias_dev=q_bias_dev,
        k_bias_dev=k_bias_dev,
        v_bias_dev=v_bias_dev,
        fused_conv_weight_taps=fused_conv_weight_taps,
        fused_conv_bias_dev=fused_conv_bias_dev,
        native_frontend_weight_taps=native_frontend_weight_taps,
        native_decay_bias=native_decay_bias,
        native_decay_scale=native_decay_scale,
        native_frontend_bias=native_frontend_bias,
        ab_proj_weight=ab_proj_weight,
        mega_fused_weight=mega_fused_weight,
        mega_fused_weight_decode=mega_fused_weight_decode,
        mega_qkv_dim=mega_qkv_dim,
        mega_a_dim=mega_a_dim,
        mega_b_dim=mega_b_dim,
        mega_g_dim=mega_g_dim,
        use_chunk_seq_prefill=use_chunk_seq_prefill,
        chunk_seq_masks_long=chunk_seq_masks_long,
    )
