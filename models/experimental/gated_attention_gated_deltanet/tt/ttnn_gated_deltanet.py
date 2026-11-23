# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the Gated DeltaNet layer.
"""

import os

import torch

import ttnn

from .ttnn_delta_rule_ops import (
    l2_norm_ttnn,
    recurrent_gated_delta_rule_ttnn,
    recurrent_gated_delta_rule_decode_ttnn,
    recurrent_gated_delta_rule_decode_inplace_ttnn,
)
from .ttnn_delta_rule_seq import chunk_gated_delta_rule_seq_adapter

_L1_SEQ_THRESHOLD = 512


def _seq_memory_config(seq_len):
    """L1 for short sequences (faster), DRAM for long (avoids OOM)."""
    return ttnn.L1_MEMORY_CONFIG if seq_len <= _L1_SEQ_THRESHOLD else None


def rms_norm_gated_ttnn(x, gate, weight, eps=1e-5, memory_config=None):
    """RMSNorm + SiLU gate (trace-compatible). Clips gate to avoid overflow at long T."""
    mc = memory_config
    x_normed = ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=mc)
    gate_act = ttnn.silu(gate, memory_config=mc)
    gate_act = ttnn.clip(gate_act, min=-1e4, max=1e4)
    return ttnn.multiply(x_normed, gate_act, memory_config=mc)


def rms_norm_ttnn(x, weight, eps=1e-5, memory_config=None):
    """Standard RMSNorm (trace-compatible)."""
    return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=memory_config)


def _causal_conv1d_decode_t1_split(
    x, conv_state_list, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """T=1 decode conv+SiLU with split state (list of [B,1,D]); avoids slice ops.

    Returns output [B,1,D], new_state_list.
    """
    mc = memory_config

    # out = sum(weight_taps[k] * state[k]) + weight_taps[K-1] * x
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        term = ttnn.multiply(conv_state_list[k], weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # Shift state left, append x
    new_state_list = conv_state_list[1:] + [x]

    return ttnn.silu(out, memory_config=mc), new_state_list


def _causal_conv1d_decode_t1_split_inplace(
    x, conv_state_list, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """Split-state T=1 conv; inplace copy for trace-stable addresses."""
    mc = memory_config

    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        term = ttnn.multiply(conv_state_list[k], weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # Inplace shift via ttnn.copy
    for k in range(kernel_size - 2):
        ttnn.copy(conv_state_list[k + 1], conv_state_list[k])
    ttnn.copy(x, conv_state_list[kernel_size - 2])

    return ttnn.silu(out, memory_config=mc), conv_state_list


def _causal_conv1d_decode_t1(x, conv_state, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None):
    """T=1 decode conv+SiLU; taps 0..K-2 from state[:,k], tap K-1 from x."""
    mc = memory_config

    # out = sum(weight_taps[k] * state[k]) + weight_taps[K-1] * x
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        s_k = conv_state[:, k : k + 1, :]
        s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(s_k, weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # Drop oldest, append x
    new_state = ttnn.concat([conv_state[:, 1:, :], x], dim=1, memory_config=mc)
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)

    return ttnn.silu(out, memory_config=mc), new_state


def _causal_conv1d_decode_t1_inplace(
    x, conv_buffer, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """T=1 conv; copy-back to pre-allocated conv_buffer for trace capture."""
    mc = memory_config
    conv_interleaved = ttnn.to_memory_config(conv_buffer, ttnn.L1_MEMORY_CONFIG)

    # out = sum(weight_taps[k] * state[k]) + weight_taps[K-1] * x
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        s_k = conv_interleaved[:, k : k + 1, :]
        s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(s_k, weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # Keep the compact persistent history row-major: TILE layout pads three
    # history rows to 32 and costs 512 KiB/layer instead of 48 KiB/layer.
    x_row = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    new_state = ttnn.concat([conv_interleaved[:, 1:, :], x_row], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    new_state_sharded = ttnn.to_memory_config(new_state, conv_buffer.memory_config())
    ttnn.copy(new_state_sharded, conv_buffer)
    ttnn.deallocate(new_state_sharded)
    ttnn.deallocate(new_state)
    ttnn.deallocate(x_row)
    ttnn.deallocate(conv_interleaved)

    return ttnn.silu(out, memory_config=mc), conv_buffer


def _causal_conv1d_verify_two(
    x, conv_state, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """Two exact T=1 convolution transitions without mutating the live state."""
    outputs = []
    candidates = []
    state = conv_state
    for step in range(2):
        token = x[:, step : step + 1, :]
        state_interleaved = ttnn.to_memory_config(state, ttnn.L1_MEMORY_CONFIG)
        out = ttnn.multiply(token, weight_taps[kernel_size - 1], memory_config=memory_config)
        for tap in range(kernel_size - 1):
            state_tap = state_interleaved[:, tap : tap + 1, :]
            state_tap = ttnn.to_layout(state_tap, ttnn.TILE_LAYOUT)
            term = ttnn.multiply(state_tap, weight_taps[tap], memory_config=memory_config)
            next_out = ttnn.add(out, term, memory_config=memory_config)
            ttnn.deallocate(out)
            ttnn.deallocate(term)
            ttnn.deallocate(state_tap)
            out = next_out
        if bias_dev is not None:
            next_out = ttnn.add(out, bias_dev, memory_config=memory_config)
            ttnn.deallocate(out)
            out = next_out
        token_row = ttnn.to_layout(token, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        next_interleaved = ttnn.concat(
            [state_interleaved[:, 1:, :], token_row], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        next_state = ttnn.to_memory_config(next_interleaved, ttnn.DRAM_MEMORY_CONFIG)
        outputs.append(ttnn.silu(out, memory_config=memory_config))
        candidates.append(next_state)
        ttnn.deallocate(out)
        ttnn.deallocate(next_interleaved)
        ttnn.deallocate(token_row)
        ttnn.deallocate(state_interleaved)
        state = next_state
    return ttnn.concat(outputs, dim=1, memory_config=memory_config), candidates


def _causal_conv1d_fir(
    x,
    weight,
    bias,
    kernel_size,
    device,
    memory_config=None,
    conv_state=None,
    weight_taps=None,
    bias_dev=None,
    valid_len=None,
):
    """Depthwise causal conv1d + SiLU via K shifted multiply-accumulate slices.

    x [B,T,D]; conv_state [B,K-1,D] or list of [B,1,D]; weight_taps/bias_dev optional.
    Returns output [B,T,D], new_state [B,K-1,D].
    """
    mc = memory_config
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    # Fast path: T=1 decode with state + pre-sliced taps
    if T == 1 and conv_state is not None and weight_taps is not None:
        return _causal_conv1d_decode_t1(
            x, conv_state, kernel_size, device, memory_config=mc, weight_taps=weight_taps, bias_dev=bias_dev
        )

    if conv_state is not None:
        x_padded = ttnn.concat([conv_state, x], dim=1, memory_config=mc)
    else:
        pad = ttnn.zeros(
            [B, kernel_size - 1, D],
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
        )
        x_padded = ttnn.concat([pad, x], dim=1, memory_config=mc)

    # new_state: last K-1 tokens; land in DRAM (carry alive across downstream kernel CBs).
    total_len = (kernel_size - 1) + T
    if valid_len is None:
        new_state = x_padded[:, total_len - (kernel_size - 1) :, :]
        # to_layout then to_memory_config: slice keeps L1 if memory_config passed to to_layout
        new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
        new_state = ttnn.to_memory_config(new_state, ttnn.DRAM_MEMORY_CONFIG)
    else:
        # Fixed-bucket masking: x is right-padded to a bucket length T but only the first
        # valid_len positions are real; the decode conv window must come from the real tail
        # x[valid_len-(K-1):valid_len], i.e. x_padded[:, valid_len : valid_len+(K-1)] (x[i]
        # is at x_padded index (K-1)+i). A static slice there would compile a new program per
        # valid_len value — defeating the bounded-program goal — so select those rows with a
        # one-hot matmul instead: the program depends only on shapes (fixed per bucket), and
        # only the one-hot VALUES depend on valid_len.
        # valid_len may be a scalar (one length for all B rows) or a per-row list/tuple of length
        # B (batched prefill: each user's own real length picks that user's decode conv window).
        sel = torch.zeros(B, kernel_size - 1, total_len, dtype=torch.float32)
        if isinstance(valid_len, (list, tuple)):
            for bi in range(B):
                for j in range(kernel_size - 1):
                    sel[bi, j, int(valid_len[bi]) + j] = 1.0
        else:
            for j in range(kernel_size - 1):
                sel[:, j, valid_len + j] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=x_padded.dtype, layout=ttnn.TILE_LAYOUT, device=device)
        xp = ttnn.to_layout(x_padded, ttnn.TILE_LAYOUT)
        # cross-chunk carry -> DRAM
        new_state = ttnn.matmul(sel_tt, xp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sel_tt)

    # Precompute weight taps if not provided
    if weight_taps is None:
        pass

        weight_torch = ttnn.to_torch(weight)
        weight_taps = []
        for k in range(kernel_size):
            w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
            weight_taps.append(
                ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            )

    total_len = (kernel_size - 1) + T
    _dram = ttnn.DRAM_MEMORY_CONFIG
    # Depthwise K-tap FIR via multiply + addcmul; re-tilize k>=1 slices (only k=0 is tile-aligned).
    out = None
    for k in range(kernel_size):
        x_slice = x_padded[:, k : k + T]
        if k != 0:
            x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)
        if out is None:
            out = ttnn.multiply(x_slice, weight_taps[k], memory_config=mc)
        else:
            out = ttnn.addcmul(out, x_slice, weight_taps[k], memory_config=mc)

    # Bias (+ fused SiLU when a bias is present) else standalone SiLU. Conv output lands in DRAM.
    _silu = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]
    if bias_dev is not None:
        return ttnn.add(out, bias_dev, activations=_silu, memory_config=_dram), new_state
    if bias is not None:
        bias_torch = ttnn.to_torch(bias).reshape(1, 1, D).contiguous()
        bias_dev_tmp = ttnn.from_torch(
            bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        return ttnn.add(out, bias_dev_tmp, activations=_silu, memory_config=_dram), new_state
    # Conv output in DRAM (feeds gated_delta_attn_seq; MAC still ran in L1 when mc=L1)
    return ttnn.silu(out, memory_config=_dram), new_state


def causal_conv1d_ttnn(
    x,
    weight,
    bias,
    kernel_size,
    device,
    max_conv_len=512,
    memory_config=None,
    conv_state=None,
    weight_taps=None,
    bias_dev=None,
):
    """Depthwise causal conv1d + SiLU. FIR fallback when conv_state, T>max_conv_len, or D>2048."""
    B, T, D = x.shape[0], x.shape[1], x.shape[2]
    mc = memory_config

    # FIR when conv_state, T>max_conv_len, or D>2048 (native conv1d CBs overflow L1 at D=4096)
    if conv_state is not None or T > max_conv_len or D > 2048:
        return _causal_conv1d_fir(
            x,
            weight,
            bias,
            kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state,
            weight_taps=weight_taps,
            bias_dev=bias_dev,
        )

    # No state: native conv1d with zero padding
    if mc is not None:
        x = ttnn.to_memory_config(x, mc)
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    pad_zeros = ttnn.zeros(
        [B, kernel_size - 1, D],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=mc,
    )
    x_padded = ttnn.concat([pad_zeros, x_rm], dim=1, memory_config=mc)

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=True,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        config_tensors_in_dram=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
    )

    [out, out_length, _] = ttnn.conv1d(
        input_tensor=x_padded,
        weight_tensor=weight,
        in_channels=D,
        out_channels=D,
        device=device,
        bias_tensor=bias,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        batch_size=B,
        input_length=T + kernel_size - 1,
        groups=D,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    out = ttnn.sharded_to_interleaved(out, memory_config=mc)
    out = ttnn.reshape(out, [B, T, D])
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=mc)

    # Save last K-1 input tokens as conv state
    x_tile = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    if T >= kernel_size - 1:
        new_state = x_tile[:, -(kernel_size - 1) :, :]
        new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
    else:
        pad_needed = kernel_size - 1 - T
        pad_state = ttnn.zeros(
            [B, pad_needed, D],
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
        )
        new_state = ttnn.concat([pad_state, x_tile], dim=1, memory_config=mc)

    return out, new_state


def gated_deltanet_forward_ttnn(
    hidden_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    a_proj_weight,
    b_proj_weight,
    o_proj_weight,
    q_conv_weight,
    k_conv_weight,
    v_conv_weight,
    q_conv_bias,
    k_conv_bias,
    v_conv_bias,
    A_log,
    dt_bias,
    o_norm_weight,
    g_proj_weight=None,
    num_heads=4,
    num_v_heads=None,
    head_k_dim=256,
    head_v_dim=512,
    conv_kernel_size=4,
    use_gate=True,
    allow_neg_eigval=False,
    norm_eps=1e-5,
    device=None,
    recurrent_state=None,
    conv_state_q=None,
    conv_state_k=None,
    conv_state_v=None,
    mode="recurrent",
    chunk_size=64,
    q_weight_taps=None,
    k_weight_taps=None,
    v_weight_taps=None,
    q_bias_dev=None,
    k_bias_dev=None,
    v_bias_dev=None,
    qkv_proj_weight=None,
    q_dim=None,
    k_dim=None,
    compute_kernel_config=None,
    A_neg_precomputed=None,
    fused_conv_weight_taps=None,
    fused_conv_bias_dev=None,
    fused_conv_state=None,
    fused_conv_state_split=None,  # list of [B,1,D] for decode (no slice)
    ab_proj_weight=None,  # fused a+b (1 matmul)
    mega_fused_weight=None,  # QKV+a+b+g in one matmul
    mega_qkv_dim=None,
    mega_a_dim=None,
    mega_b_dim=None,
    mega_g_dim=None,
    mega_decode_program_config=None,
    out_decode_program_config=None,
    use_inplace_state=False,  # ttnn.copy for trace-stable state
    chunk_seq_masks=None,  # cached masks for gated_delta_attn_seq prefill
    valid_len=None,  # fixed-bucket padding; zeros padded positions in scan
    native_frontend_runner=None,
    native_output_runner=None,
    native_verify_output_runner=None,
):
    """Gated DeltaNet forward. mode: recurrent (decode T=1) or chunk (prefill T>1).

    Returns output [B,T,hidden], new_state [B,H,K,V], conv states.
    """
    if num_v_heads is None:
        num_v_heads = num_heads

    verify_two = mode == "verify_two"
    flat_decode = (
        not verify_two
        and len(hidden_states.shape) == 4
        and int(hidden_states.shape[0]) == 1
        and int(hidden_states.shape[1]) == 1
    )
    B = 1 if verify_two else (int(hidden_states.shape[-2]) if flat_decode else int(hidden_states.shape[0]))
    T = 2 if verify_two else (1 if flat_decode else int(hidden_states.shape[1]))
    # valid_len path forces DRAM (bucket 512 hits L1 CB clash on _seq_memory_config threshold)
    mc = ttnn.DRAM_MEMORY_CONFIG if verify_two else (None if valid_len is not None else _seq_memory_config(T))

    ckc = compute_kernel_config

    def decode_linear(x, weight, program_config=None):
        """Treat serving slots as M rows, not independent matmul batches."""
        if T == 1 and B > 1:
            x = ttnn.reshape(x, [1, 1, B, x.shape[-1]])
            out = ttnn.linear(
                x,
                weight,
                memory_config=mc,
                compute_kernel_config=ckc,
                program_config=program_config,
            )
            return ttnn.reshape(out, [B, 1, out.shape[-1]])
        return ttnn.linear(
            x,
            weight,
            memory_config=mc,
            compute_kernel_config=ckc,
            program_config=program_config,
        )

    # Mega-fused: one matmul for QKV+a+b+g (decode needs conv state; prefill needs fused taps)
    use_mega_fused = (
        mega_fused_weight is not None
        and mega_qkv_dim is not None
        and (T == 1 or fused_conv_weight_taps is not None)
        and (T > 1 or (fused_conv_state is not None and fused_conv_weight_taps is not None))
    )

    # Fused conv decode: QKV -> fused conv -> split (skip mega path)
    use_fused_conv = (
        not use_mega_fused
        and T == 1
        and fused_conv_weight_taps is not None
        and fused_conv_state is not None
        and qkv_proj_weight is not None
        and q_dim is not None
    )

    native_frontend_preexpanded = False
    if use_mega_fused and (T == 1 or verify_two) and native_frontend_runner is not None:
        # Keep the packed M=2 projection in L1. The native frontend reads tile
        # rows directly; the DRAM-interleaved form does not preserve the same
        # token-row addressing for its two-row logical shape.
        mega_out = ttnn.linear(
            hidden_states,
            mega_fused_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=ckc,
            program_config=mega_decode_program_config,
        )
        q, k, v, a_raw, b_raw = native_frontend_runner(
            mega_out, fused_conv_weight_taps, fused_conv_bias_dev, fused_conv_state
        )
        gate_raw = mega_out[..., mega_qkv_dim + mega_a_dim + mega_b_dim :]
        gate_raw = ttnn.to_layout(gate_raw, ttnn.TILE_LAYOUT)
        ttnn.deallocate(mega_out)
        new_fused_conv_state = (
            (native_frontend_runner.conv_s1, native_frontend_runner.conv_s2)
            if verify_two
            else fused_conv_state
        )
        new_conv_q = None
        new_conv_k = None
        new_conv_v = None
        _mega_extracted = True
        native_frontend_preexpanded = True
    elif use_mega_fused:
        mega_out = decode_linear(
            hidden_states,
            mega_fused_weight,
            program_config=mega_decode_program_config if (T == 1 or verify_two) else None,
        )
        # Split QKV | a | b | g
        qkv = mega_out[..., :mega_qkv_dim]
        qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT)
        a_raw = mega_out[..., mega_qkv_dim : mega_qkv_dim + mega_a_dim]
        a_raw = ttnn.to_layout(a_raw, ttnn.TILE_LAYOUT)
        b_raw = mega_out[..., mega_qkv_dim + mega_a_dim : mega_qkv_dim + mega_a_dim + mega_b_dim]
        b_raw = ttnn.to_layout(b_raw, ttnn.TILE_LAYOUT)
        gate_raw = mega_out[..., mega_qkv_dim + mega_a_dim + mega_b_dim :]
        gate_raw = ttnn.to_layout(gate_raw, ttnn.TILE_LAYOUT)
        ttnn.deallocate(mega_out)

        # Fused conv on QKV. Verification deliberately composes two exact
        # single-token transitions while retaining both candidate states.
        if verify_two:
            qkv, new_fused_conv_state = _causal_conv1d_verify_two(
                qkv,
                fused_conv_state,
                conv_kernel_size,
                device,
                memory_config=mc,
                weight_taps=fused_conv_weight_taps,
                bias_dev=fused_conv_bias_dev,
            )
            new_conv_q = None
            new_conv_k = None
            new_conv_v = None
        elif T > 1:
            # Prefill FIR conv
            qkv, new_fused_conv_state_raw = _causal_conv1d_fir(
                qkv,
                None,
                None,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=fused_conv_state,
                weight_taps=fused_conv_weight_taps,
                bias_dev=fused_conv_bias_dev,
                valid_len=valid_len,
            )
            new_fused_conv_state = new_fused_conv_state_raw
            # Per-stream conv states for decode handoff
            new_conv_q = new_fused_conv_state_raw[:, :, :q_dim]
            new_conv_q = ttnn.to_layout(new_conv_q, ttnn.TILE_LAYOUT)
            new_conv_k = new_fused_conv_state_raw[:, :, q_dim : q_dim + k_dim]
            new_conv_k = ttnn.to_layout(new_conv_k, ttnn.TILE_LAYOUT)
            new_conv_v = new_fused_conv_state_raw[:, :, q_dim + k_dim :]
            new_conv_v = ttnn.to_layout(new_conv_v, ttnn.TILE_LAYOUT)
        elif fused_conv_state_split is not None:
            # Split-state decode (no slice+to_layout)
            conv_fn = _causal_conv1d_decode_t1_split_inplace if use_inplace_state else _causal_conv1d_decode_t1_split
            qkv, new_fused_conv_state = conv_fn(
                qkv,
                fused_conv_state_split,
                conv_kernel_size,
                device,
                memory_config=mc,
                weight_taps=fused_conv_weight_taps,
                bias_dev=fused_conv_bias_dev,
            )
            new_conv_q = None
            new_conv_k = None
            new_conv_v = None
        else:
            # Decode with fused state
            conv_fn = _causal_conv1d_decode_t1_inplace if use_inplace_state else _causal_conv1d_decode_t1
            qkv, new_fused_conv_state = conv_fn(
                qkv,
                fused_conv_state,
                conv_kernel_size,
                device,
                memory_config=mc,
                weight_taps=fused_conv_weight_taps,
                bias_dev=fused_conv_bias_dev,
            )
            new_conv_q = None
            new_conv_k = None
            new_conv_v = None

        # Split QKV after conv
        q = qkv[:, :, :q_dim]
        k = qkv[:, :, q_dim : q_dim + k_dim]
        v = qkv[:, :, q_dim + k_dim :]
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        ttnn.deallocate(qkv)

        # a/b/g already from mega projection
        _mega_extracted = True
    elif use_fused_conv:
        # Fused decode: QKV proj -> fused conv -> split
        qkv = ttnn.linear(hidden_states, qkv_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        # Fused conv1d on concatenated QKV
        conv_fn = _causal_conv1d_decode_t1_inplace if use_inplace_state else _causal_conv1d_decode_t1
        qkv, new_fused_conv_state = conv_fn(
            qkv,
            fused_conv_state,
            conv_kernel_size,
            device,
            memory_config=mc,
            weight_taps=fused_conv_weight_taps,
            bias_dev=fused_conv_bias_dev,
        )
        # Split after conv
        q = qkv[:, :, :q_dim]
        k = qkv[:, :, q_dim : q_dim + k_dim]
        v = qkv[:, :, q_dim + k_dim :]
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        ttnn.deallocate(qkv)
        new_conv_q = None
        new_conv_k = None
        new_conv_v = None
        _mega_extracted = False
    elif qkv_proj_weight is not None and q_dim is not None:
        qkv = ttnn.linear(hidden_states, qkv_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        # Fused conv prefill on concatenated QKV
        if verify_two:
            qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT)
            qkv, new_fused_conv_state = _causal_conv1d_verify_two(
                qkv,
                fused_conv_state,
                conv_kernel_size,
                device,
                memory_config=mc,
                weight_taps=fused_conv_weight_taps,
                bias_dev=fused_conv_bias_dev,
            )
            q = qkv[:, :, :q_dim]
            k = qkv[:, :, q_dim : q_dim + k_dim]
            v = qkv[:, :, q_dim + k_dim :]
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
            v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
            ttnn.deallocate(qkv)
            new_conv_q = None
            new_conv_k = None
            new_conv_v = None
            _mega_extracted = False
        # Fused conv prefill on concatenated QKV
        elif T > 1 and fused_conv_weight_taps is not None:
            qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT)
            qkv, new_fused_conv_state_raw = _causal_conv1d_fir(
                qkv,
                None,
                None,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=fused_conv_state,
                weight_taps=fused_conv_weight_taps,
                bias_dev=fused_conv_bias_dev,
                valid_len=valid_len,
            )
            # Split after conv
            q = qkv[:, :, :q_dim]
            k = qkv[:, :, q_dim : q_dim + k_dim]
            v = qkv[:, :, q_dim + k_dim :]
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
            v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
            ttnn.deallocate(qkv)
            # Per-stream conv states from fused state
            D_total = (
                q_dim + k_dim + (qkv_proj_weight.shape[-1] - q_dim - k_dim)
                if hasattr(qkv_proj_weight, "shape")
                else None
            )
            new_conv_q = new_fused_conv_state_raw[:, :, :q_dim]
            new_conv_q = ttnn.to_layout(new_conv_q, ttnn.TILE_LAYOUT)
            new_conv_k = new_fused_conv_state_raw[:, :, q_dim : q_dim + k_dim]
            new_conv_k = ttnn.to_layout(new_conv_k, ttnn.TILE_LAYOUT)
            new_conv_v = new_fused_conv_state_raw[:, :, q_dim + k_dim :]
            new_conv_v = ttnn.to_layout(new_conv_v, ttnn.TILE_LAYOUT)
            new_fused_conv_state = new_fused_conv_state_raw
            _mega_extracted = False
        else:
            q = qkv[:, :, :q_dim]
            k = qkv[:, :, q_dim : q_dim + k_dim]
            v = qkv[:, :, q_dim + k_dim :]
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
            v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
            ttnn.deallocate(qkv)
            new_fused_conv_state = None
            _mega_extracted = False
            q, new_conv_q = causal_conv1d_ttnn(
                q,
                q_conv_weight,
                q_conv_bias,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=conv_state_q,
                weight_taps=q_weight_taps,
                bias_dev=q_bias_dev,
            )
            k, new_conv_k = causal_conv1d_ttnn(
                k,
                k_conv_weight,
                k_conv_bias,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=conv_state_k,
                weight_taps=k_weight_taps,
                bias_dev=k_bias_dev,
            )
            v, new_conv_v = causal_conv1d_ttnn(
                v,
                v_conv_weight,
                v_conv_bias,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=conv_state_v,
                weight_taps=v_weight_taps,
                bias_dev=v_bias_dev,
            )
    else:
        q = ttnn.linear(hidden_states, q_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        k = ttnn.linear(hidden_states, k_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        v = ttnn.linear(hidden_states, v_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        new_fused_conv_state = None
        _mega_extracted = False
        q, new_conv_q = causal_conv1d_ttnn(
            q,
            q_conv_weight,
            q_conv_bias,
            conv_kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state_q,
            weight_taps=q_weight_taps,
            bias_dev=q_bias_dev,
        )
        k, new_conv_k = causal_conv1d_ttnn(
            k,
            k_conv_weight,
            k_conv_bias,
            conv_kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state_k,
            weight_taps=k_weight_taps,
            bias_dev=k_bias_dev,
        )
        v, new_conv_v = causal_conv1d_ttnn(
            v,
            v_conv_weight,
            v_conv_bias,
            conv_kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state_v,
            weight_taps=v_weight_taps,
            bias_dev=v_bias_dev,
        )

    if not native_frontend_preexpanded:
        # Reshape to heads (explicit mc keeps decode in L1)
        q = ttnn.reshape(q, [B, T, num_heads, head_k_dim], memory_config=mc)
        k = ttnn.reshape(k, [B, T, num_heads, head_k_dim], memory_config=mc)
        v = ttnn.reshape(v, [B, T, num_v_heads, head_v_dim], memory_config=mc)

        # GVA: repeat q,k
        if num_v_heads > num_heads:
            repeats = num_v_heads // num_heads
            q = ttnn.repeat_interleave(q, repeats, dim=2)
            k = ttnn.repeat_interleave(k, repeats, dim=2)

    native_decode_frontend = (
        (T == 1 or verify_two)
        and os.environ.get("QWEN_GDN_RECURRENT_FUSED") == "1"
        and os.environ.get("QWEN_GDN_FRONTEND_FUSED") == "1"
        and recurrent_state is not None
        and A_neg_precomputed is not None
        and not allow_neg_eigval
    )

    # Beta and g. The native decode front end consumes raw a/b and computes
    # sigmoid(b), softplus(a + dt_bias), and the recurrent decay on Tensix.
    if _mega_extracted:
        a = a_raw
        beta = b_raw if native_decode_frontend else ttnn.sigmoid(b_raw, memory_config=mc)
    elif ab_proj_weight is not None:
        ab = ttnn.linear(hidden_states, ab_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        num_v = num_v_heads if num_v_heads is not None else num_heads
        a = ab[:, :, :num_v]
        a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
        b_raw = ab[:, :, num_v:]
        b_raw = ttnn.to_layout(b_raw, ttnn.TILE_LAYOUT)
        ttnn.deallocate(ab)
        beta = b_raw if native_decode_frontend else ttnn.sigmoid(b_raw, memory_config=mc)
    else:
        b_raw = ttnn.linear(hidden_states, b_proj_weight, memory_config=mc, compute_kernel_config=ckc)
        beta = b_raw if native_decode_frontend else ttnn.sigmoid(b_raw, memory_config=mc)
        a = ttnn.linear(hidden_states, a_proj_weight, memory_config=mc, compute_kernel_config=ckc)
    if native_decode_frontend:
        g = a
    else:
        if allow_neg_eigval:
            beta = ttnn.multiply(beta, 2.0, memory_config=mc)
        a_biased = ttnn.add(a, dt_bias, memory_config=mc)
        sp = ttnn.softplus(a_biased, memory_config=mc)
        if A_neg_precomputed is not None:
            g = ttnn.multiply(A_neg_precomputed, sp, memory_config=mc)
        else:
            A = ttnn.exp(A_log, memory_config=mc)
            A_neg = ttnn.neg(A, memory_config=mc)
            g = ttnn.multiply(A_neg, sp, memory_config=mc)

    output_prepared = False
    if verify_two:
        # Both paths preserve the two recurrent prefix states without committing.
        live_state = ttnn.reshape(recurrent_state, [num_v_heads, head_k_dim, head_v_dim])
        decay_bias_tokens = ttnn.reshape(dt_bias, [num_v_heads, 1, 1])
        decay_scale_tokens = ttnn.reshape(A_neg_precomputed, [num_v_heads, 1, 1])
        if native_verify_output_runner is not None and _mega_extracted:
            o, state_s1, state_s2 = native_verify_output_runner(
                q, k, v, a, b_raw, decay_bias_tokens, decay_scale_tokens,
                chunk_seq_masks["ones_32"], live_state, gate_raw, o_norm_weight,
            )
            output_prepared = True
        else:
            o_tokens, state_s1, state_s2 = ttnn.transformer.verify_two_gated_delta_rule(
                q, k, v, a, b_raw, decay_bias_tokens, decay_scale_tokens,
                chunk_seq_masks["ones_32"], live_state,
            )
            if native_output_runner is not None and _mega_extracted:
                o = o_tokens
            else:
                o = ttnn.permute(o_tokens, (1, 0, 2, 3))
        new_state = (
            ttnn.reshape(state_s1, [1, num_v_heads, head_k_dim, head_v_dim]),
            ttnn.reshape(state_s2, [1, num_v_heads, head_k_dim, head_v_dim]),
        )
    # Gated delta rule: chunk prefill (fp32 seq kernel) vs decode (optimized T=1) vs recurrent fallback.
    elif mode == "chunk" and T > 1:
        o, new_state = chunk_gated_delta_rule_seq_adapter(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            chunk_size=chunk_size,
            initial_state=recurrent_state,
            device=device,
            cached_masks=chunk_seq_masks,
            valid_len=valid_len,
        )
    elif T == 1:
        if os.environ.get("QWEN_GDN_RECURRENT_FUSED") == "1" and recurrent_state is not None:
            q_input = q if native_decode_frontend else l2_norm_ttnn(q, dim=-1)
            k_input = k if native_decode_frontend else l2_norm_ttnn(k, dim=-1)
            o, fused_state = ttnn.transformer.chunk_gated_delta_rule(
                q_input,
                k_input,
                v,
                g,
                beta,
                initial_state=recurrent_state,
                output_final_state=True,
                chunk_size=32,
                memory_config=mc,
                compute_kernel_config=ckc,
                fuse_decode_frontend=native_decode_frontend,
                decay_bias=dt_bias if native_decode_frontend else None,
                decay_scale=A_neg_precomputed if native_decode_frontend else None,
                ones=chunk_seq_masks["ones_32"] if native_decode_frontend else None,
            )
            o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=mc)
            new_state = fused_state
        elif use_inplace_state and recurrent_state is not None:
            o, new_state = recurrent_gated_delta_rule_decode_inplace_ttnn(
                q=q,
                k=k,
                v=v,
                beta=beta,
                g=g,
                state_buffer=recurrent_state,
                device=device,
            )
        else:
            o, new_state = recurrent_gated_delta_rule_decode_ttnn(
                q=q,
                k=k,
                v=v,
                beta=beta,
                g=g,
                initial_state=recurrent_state,
                device=device,
            )
    else:
        o, new_state = recurrent_gated_delta_rule_ttnn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            initial_state=recurrent_state,
            device=device,
        )

    # Output norm + projection (clip before o_proj to avoid sparse overflow).
    # The native decode path consumes the head-major recurrent output and the
    # unreshaped mega-projection gate, then emits projection layout.
    if not output_prepared:
        if (T == 1 or verify_two) and native_output_runner is not None and _mega_extracted:
            # The native kernel emits the packed projection input directly.
            o = native_output_runner(o, gate_raw, o_norm_weight)
        else:
            if use_gate and g_proj_weight is not None:
                if _mega_extracted:
                    gate = ttnn.reshape(gate_raw, [B, T, num_v_heads, head_v_dim], memory_config=mc)
                else:
                    gate = ttnn.linear(hidden_states, g_proj_weight, memory_config=mc, compute_kernel_config=ckc)
                    gate = ttnn.reshape(gate, [B, T, num_v_heads, head_v_dim], memory_config=mc)
                o = rms_norm_gated_ttnn(o, gate, o_norm_weight, eps=norm_eps, memory_config=mc)
            else:
                o = rms_norm_ttnn(o, o_norm_weight, eps=norm_eps, memory_config=mc)

            o = ttnn.clip(o, min=-1e4, max=1e4, memory_config=mc)
            o = ttnn.reshape(o, [B, T, num_v_heads * head_v_dim], memory_config=mc)
    o = decode_linear(
        o,
        o_proj_weight,
        program_config=out_decode_program_config if T == 1 else None,
    )

    if flat_decode:
        o = ttnn.reshape(o, [1, 1, B, o.shape[-1]])

    return o, new_state, new_conv_q, new_conv_k, new_conv_v, new_fused_conv_state
