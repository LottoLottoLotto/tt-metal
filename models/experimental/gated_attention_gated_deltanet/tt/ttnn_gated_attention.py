# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Gated Attention.
"""

import os as _os
from functools import lru_cache

import ttnn


@lru_cache(maxsize=32)
def _bounded_projection_program_config(tokens, input_dim, output_dim):
    """Bound prefill CBs independently of the bucket and full projection width."""
    from models.demos.blackhole.qwen36.tt import tp_common

    tuning = dict(tp_common.prefill_tuning(1))
    tuning["in0_block_w_cap"] = 4
    # All 11 grid columns (the GDN output projection's layout): 8 columns left 30 of the
    # 110 cores idle; the K blocking (in0_block_w) is independent of the column count.
    config = tp_common.create_prefill_mlp_matmul_program_config(
        tokens, input_dim, output_dim, max_cols=11, tuning=tuning
    )
    config.out_block_h = 1
    config.out_block_w = max(
        width for width in range(config.out_subblock_w, min(8, config.per_core_N) + 1, config.out_subblock_w)
        if config.per_core_N % width == 0
    )
    # BF16 input/output, BF8 weight, FP32 partials: <= 135168 bytes/core
    # (2*1*4*2048 + 2*4*8*1088 + 1*8*(2048+4096)).
    # 1572864 - resident 749568 - CB origin 111488 - margin 61440 = 650368 B/core.
    return config


def _prefetch_linear(x, weight, **kwargs):
    """ttnn.linear, routed through the Qwen DRISC weight prefetcher for registered decode weights."""
    from models.demos.blackhole.qwen36.tt import weight_prefetch

    return weight_prefetch.linear(x, weight, **kwargs)


def rotate_half_ttnn(x):
    """Rotates half the hidden dims of the input."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def apply_rotary_pos_emb_ttnn(q, k, cos, sin):
    """Apply RoPE to query and key tensors using TTNN ops."""
    # The fused program retains every BF16 rounding boundary and copies the
    # nonrotary dimensions unchanged. Other models/batches/layouts use the
    # general implementation below.
    shape = tuple(q.shape)
    tokens = shape[-2] if len(shape) == 4 else 0
    if (
        1 <= tokens <= 32
        and shape == (1, 24, tokens, 256)
        and tuple(k.shape) == (1, 4, tokens, 256)
        and tuple(q.padded_shape) == (1, 24, 32, 256)
        and tuple(k.padded_shape) == (1, 4, 32, 256)
        and all(tuple(t.padded_shape) == (*tuple(t.shape)[:-2], 32, 64) for t in (cos, sin))
        and all(tuple(t.shape) in ((1, tokens, 64), (1, 1, tokens, 64)) for t in (cos, sin))
        and all(
            t.dtype == ttnn.bfloat16
            and t.layout == ttnn.TILE_LAYOUT
            and t.tile == ttnn.Tile((32, 32))
            and t.memory_config() in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
            for t in (q, k, cos, sin)
        )
        and q.device().compute_with_storage_grid_size().x >= 11
        and q.device().compute_with_storage_grid_size().y >= 6
    ):
        from models.demos.blackhole.qwen36.tt.fused_rope import fused_rope

        return fused_rope(q, k, cos, sin)

    if len(cos.shape) < 4:
        cos = ttnn.unsqueeze(cos, 1)  # [B, 1, T, D]
        sin = ttnn.unsqueeze(sin, 1)

    rotary_dim = cos.shape[-1]
    full_dim = q.shape[-1]

    q_rot = q[..., :rotary_dim]
    k_rot = k[..., :rotary_dim]

    q_embed = ttnn.add(
        ttnn.multiply(q_rot, cos),
        ttnn.multiply(rotate_half_ttnn(q_rot), sin),
    )
    k_embed = ttnn.add(
        ttnn.multiply(k_rot, cos),
        ttnn.multiply(rotate_half_ttnn(k_rot), sin),
    )

    if rotary_dim < full_dim:
        q_pass = q[..., rotary_dim:]
        k_pass = k[..., rotary_dim:]
        q_embed = ttnn.concat([q_embed, q_pass], dim=-1)
        k_embed = ttnn.concat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


def rms_norm_zero_centered_ttnn(x, weight, eps=1e-6):
    """
    Zero-centered RMSNorm using TTNN: x * rsqrt(mean(x^2) + eps) * (1 + weight).
    """
    x_sq = ttnn.multiply(x, x)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True)
    inv_rms = ttnn.rsqrt(ttnn.add(variance, eps))
    x_normed = ttnn.multiply(x, inv_rms)
    scale = ttnn.add(weight, 1.0)
    return ttnn.multiply(x_normed, scale)


def _get_sdpa_program_config(device, seq_len, q_seq_len=None, chunk_start_idx=None):
    """Build SDPAProgramConfig with chunk sizes tuned to sequence length.

    For chunked paged prefill (chunk_start_idx >= 0), chunk sizes follow the
    tt_transformers reference (model_config.py:1433). For chunk_start_idx > 0,
    chunk sizes must divide chunk_start_idx (uses lowest-set-bit trick).
    For decode with pre-allocated cache (q_seq_len=1), use small chunks.
    For segmented prefill (q_seq_len < seq_len, both > 1), use small chunks.
    For regular prefill, scale chunks based on seq_len.
    """
    grid_size = device.compute_with_storage_grid_size()
    from models.demos.blackhole.qwen36.tt.attention.config import optimized_27b_attention
    if chunk_start_idx is not None and chunk_start_idx == 0:
        # First chunk of paged prefill: no divisibility constraint.
        # Use small chunks on Blackhole — chunked_sdpa has higher L1 overhead
        # than regular sdpa (page table indirection + block metadata).
        # 256/256 causes L1 clash; 64/64 is safe on Blackhole P150.
        q_chunk = 64
        k_chunk = 64
    elif chunk_start_idx is not None and chunk_start_idx > 0:
        # Subsequent chunks of paged prefill: chunk sizes must divide chunk_start_idx.
        # (x & -x) extracts the largest power-of-2 factor of x.
        # Use conservative sizes on Blackhole to avoid L1 clashes.
        if seq_len >= 2048:
            q_chunk = min(64, chunk_start_idx & -chunk_start_idx)
            k_chunk = min(64, chunk_start_idx & -chunk_start_idx)
        else:
            q_chunk = min(64, chunk_start_idx & -chunk_start_idx)
            k_chunk = min(64, chunk_start_idx & -chunk_start_idx)
        assert chunk_start_idx % q_chunk == 0, f"chunk_start_idx={chunk_start_idx} not divisible by q_chunk={q_chunk}"
    elif q_seq_len is not None and q_seq_len <= 1 and seq_len >= 512:
        # Decode with large pre-allocated cache: use small chunks to avoid L1 OOM
        q_chunk = 32
        k_chunk = 64
    elif q_seq_len is not None and q_seq_len > 1 and seq_len > q_seq_len:
        # Segmented prefill: Q shorter than KV, use small chunks to avoid L1 OOM
        q_chunk = 64
        k_chunk = 64
    elif seq_len >= 8192:
        q_chunk = 64
        k_chunk = 64
    elif seq_len >= 4096:
        q_chunk = 128
        k_chunk = 128
    elif seq_len >= 2048:
        q_chunk = 128
        k_chunk = 128
    else:
        q_chunk = 64
        k_chunk = 64
    if optimized_27b_attention():
        # BF16 Q/K/V and FP32 QK/sum CBs at 64/64: <=360448 B/core,
        # including a dense BF16 mask (causal-only is 348160), plus page table.
        q_chunk = min(q_chunk, 64)
        k_chunk = min(k_chunk, 64)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


def _get_flexible_sdpa_program_config(device):
    """Fixed SDPAProgramConfig for the FLEXIBLE chunked SDPA (chunk_start_idx supplied as a
    runtime device tensor). The chunk size must divide every chunk_start (all multiples of the
    2048-token GDN chunk) so ONE program serves all chunk positions — required so a single
    captured trace can be replayed per chunk in chunk-outer prefill.

    Default 128 (not 64): 128 still divides 2048, and a microbench showed q/k=128 is ~2x faster
    than 64 on the prefill SDPA shape (16 q-heads / 4 kv-heads, d=256, KV~32k) — numerically exact
    (only tiling granularity changes). Opt back to 64 with QWEN9B_SDPA_QK64=1 if 128 ever clashes
    with L1 in the paged chunked-SDPA path on a given prompt length.
    """
    from models.demos.blackhole.qwen36.tt.attention.config import optimized_27b_attention
    qk = 64 if optimized_27b_attention() or _os.environ.get("QWEN9B_SDPA_QK64") == "1" else 128
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qk,
        k_chunk_size=qk,
        exp_approx_mode=False,
    )


def _get_paged_sdpa_decode_program_config(device, max_seq_len):
    """Build SDPAProgramConfig for paged_scaled_dot_product_attention_decode.

    Uses small chunk sizes to avoid L1 OOM on Blackhole.
    Note: do NOT pass compute_kernel_config to paged_sdpa_decode on Blackhole —
    the default is compatible; custom configs can cause incorrect output.
    """
    from models.demos.blackhole.qwen36.tt.attention.config import optimized_27b_attention

    config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
    )
    if optimized_27b_attention():
        # Every 27B caller (M=1 decode, MTP-1/MTP-3/DFlash verification via verify_sdpa.py) splits
        # each (row, KV head)'s KV range over the same 8 cores and reduction tree, so a verification
        # row reproduces the single-row decode bit for bit. Uncapped, B=1 would take 16 cores/head.
        config.max_cores_per_head_batch = 8
    return config


def _get_sdpa_compute_kernel_config():
    """WormholeComputeKernelConfig for SDPA -- HiFi2 with fp32 accumulation."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _tile_heads_ok(t, T, width):
    """Tensor is a plain interleaved 32x32-tiled [B,T,X] / [B,H,T,D] buffer with tile-aligned
    T and head width, so head/token relayouts can move whole tiles."""
    return (
        T % 32 == 0
        and width % 32 == 0
        and t.layout == ttnn.TILE_LAYOUT
        and tuple(t.padded_shape) == tuple(t.shape)
        and t.tile == ttnn.Tile([32, 32])
        and not t.is_sharded()
    )


def gated_attention_forward_ttnn(
    hidden_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    device,
    norm_eps=1e-6,
    use_optimized_concat=False,
    past_key=None,
    past_value=None,
    compute_kernel_config=None,
    kv_cache_key=None,
    kv_cache_value=None,
    cache_pos=None,
    cache_len=None,
    memory_config=None,
    norm_weights_pre_offset=False,
    # Trace-compatible mode: save new K/V to buffers, write to staging pos, use full cache + mask
    trace_new_k_buf=None,
    trace_new_v_buf=None,
    trace_attn_mask=None,
    trace_kv_pad_zeros=None,
    trace_staging_pos=None,
    # Decode-optimized SDPA: pass cur_pos_tensor to use sdpa_decode (skips mask management)
    cur_pos_tensor=None,
    # Pre-allocated sharded buffers for paged_update_cache (avoids allocation during trace)
    paged_k_buf=None,
    paged_v_buf=None,
    # Paged attention with page_table (for vLLM integration)
    page_table=None,
    paged_kv_cache_key=None,  # DRAM paged cache [num_blocks, num_kv_heads, block_size, head_dim]
    paged_kv_cache_value=None,  # DRAM paged cache [num_blocks, num_kv_heads, block_size, head_dim]
    # Paged prefill: fill K/V into paged cache, attend via chunked SDPA
    chunk_page_table=None,  # [1, num_blocks_in_chunk] int32 — blocks for this chunk only
    chunk_start_idx=None,  # int — absolute position of this chunk in the full sequence
    chunk_start_idx_tensor=None,  # device tensor [1] int32 — runtime offset for trace-replay (flexible SDPA)
    q_decode_program_config=None,
    k_decode_program_config=None,
    v_decode_program_config=None,
    o_decode_program_config=None,
    qkv_fused_weight=None,
    qkv_fused_q_dim=None,
    qkv_fused_k_dim=None,
    qkv_fused_program_config=None,
    verify_two=False,
    shadow_fill_page_table=None,
    shadow_source_page=0,
    shadow_same_page=True,
    verify_many=False,
    tree_fork_position=0,
    verify_rows=None,
    verify_groups=None,
):
    """
    TTNN forward pass for Gated Attention with KV cache support.

    Uses ttnn.transformer.scaled_dot_product_attention (FlashAttention-2 kernel)
    with SDPAProgramConfig for tiling and WormholeComputeKernelConfig for precision.
    The fused kernel handles GQA (num_q_heads != num_kv_heads) internally.

    Args:
        hidden_states: ttnn.Tensor [B, T, hidden_size]
        *_proj_weight: ttnn.Tensor weight matrices in [in_features, out_features] format
                       (transposed from PyTorch convention)
        q_norm_weight, k_norm_weight: ttnn.Tensor [head_dim]
        cos, sin: ttnn.Tensor [B, T, head_dim] rotary embeddings
        num_attention_heads: number of Q heads
        num_key_value_heads: number of KV heads
        head_dim: dimension per head
        device: ttnn device
        norm_eps: RMSNorm epsilon
        use_optimized_concat: if True, use ttnn.transformer.concatenate_heads
        past_key: ttnn.Tensor [B, H_kv, S_past, D] or None
        past_value: ttnn.Tensor [B, H_kv, S_past, D] or None

    Returns:
        output: ttnn.Tensor [B, T, hidden_size]
        new_key: ttnn.Tensor [B, H_kv, S_total, D] updated KV cache key
        new_value: ttnn.Tensor [B, H_kv, S_total, D] updated KV cache value
    """
    from models.demos.blackhole.qwen36.tt.attention.config import optimized_27b_attention
    optimized = optimized_27b_attention()
    if optimized:
        if (
            (num_attention_heads, num_key_value_heads, head_dim) != (24, 4, 256)
            or hidden_states.shape[-1] != 5120 or cos.shape[-1] != 64 or sin.shape[-1] != 64
            or hidden_states.dtype != ttnn.bfloat16 or hidden_states.layout != ttnn.TILE_LAYOUT
        ):
            raise ValueError("Optimized 27B attention requires tiled BF16 hidden5120, Q24/KV4/head256/rotary64")
        for weight in (q_proj_weight, k_proj_weight, v_proj_weight, qkv_fused_weight, o_proj_weight):
            if weight is not None and (
                weight.dtype != ttnn.bfloat8_b or weight.layout != ttnn.TILE_LAYOUT
                or weight.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            ):
                raise ValueError("Optimized 27B attention requires interleaved tiled BF8 DRAM weights")
        for cache in (paged_kv_cache_key, paged_kv_cache_value, kv_cache_key, kv_cache_value, past_key, past_value):
            # Paged KV may be BFP8 for native-context serving (V also BFP4: the 4-bit V cache with a
            # BFP8 K cache); the other caches stay BF16.
            paged = cache is not None and (cache is paged_kv_cache_key or cache is paged_kv_cache_value)
            allowed = (ttnn.bfloat16, ttnn.bfloat8_b) if paged else (ttnn.bfloat16,)
            if cache is not None and cache is paged_kv_cache_value and paged_kv_cache_key is not None and (
                paged_kv_cache_key.dtype == ttnn.bfloat8_b
            ):
                allowed = (ttnn.bfloat8_b, ttnn.bfloat4_b)
            if cache is not None and (
                cache.dtype not in allowed or cache.layout != ttnn.TILE_LAYOUT
                or cache.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            ):
                raise ValueError("Optimized 27B attention requires tiled BF16 (paged: BF16/BFP8, V BFP4) DRAM caches")
        # Persistent native state already occupies 749568 bytes/core. Keep transient
        # activations in DRAM; only the explicit cache-update shards reside in L1.
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        if page_table is not None:
            if page_table.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise ValueError("Optimized 27B page tables must be DRAM interleaved")
            # Paged decode uses full 32x32 tiles for Q24 (not the Q16 half-tile
            # optimization). At 8x8 / KV4, M=1 has <=16 reducer workers/head:
            # 282624 fixed + (16-1)*20480 reduction = 589824 B/core.
            # M=2 uses <=8 workers/head, i.e. 425984 B/core. Both need one
            # aligned page-table row and two aligned current-position sticks.
            page_bytes = ((int(page_table.padded_shape[-1]) * 4 + 63) // 64) * 64
            pos_bytes = 0 if cur_pos_tensor is None else (
                (int(cur_pos_tensor.padded_shape[-1]) * 4 + 63) // 64
            ) * 64
            if page_bytes + 2 * pos_bytes + 589824 > 650368:
                raise ValueError("27B paged attention metadata exceeds the resident-state L1 budget")
    flat_decode = (
        not verify_two and not verify_many
        and len(hidden_states.shape) == 4
        and int(hidden_states.shape[0]) == 1
        and int(hidden_states.shape[1]) == 1
    )
    B = 1 if verify_two else (int(hidden_states.shape[-2]) if flat_decode else int(hidden_states.shape[0]))
    T = 2 if verify_two else (1 if flat_decode else int(hidden_states.shape[1]))
    if optimized and page_table is not None and T == 1:
        # Update-cache additionally holds both 16 KiB K/V shards. Its FP32
        # intermediate upper bound grows with packed batch even though SDPA's
        # reducer storage shrinks. M=2 writes are serialized B=1 calls.
        update_bytes = 200704 + B * 16384 + page_bytes
        if update_bytes > 650368:
            raise ValueError("27B packed cache update exceeds the resident-state L1 budget")
    scaling = head_dim**-0.5
    # 4-bit V cache (QWEN36_KV_CACHE_DTYPE=bfp8_v4r): V is rotated by R and stored as RNE-rounded
    # BFP4 (v_rotate.py); every attention output is rotated back by R^T before the gate. Only the
    # optimized single-P150 paths below implement it; the others refuse a BFP4 V cache.
    v4 = paged_kv_cache_value is not None and paged_kv_cache_value.dtype == ttnn.bfloat4_b
    if v4:
        from models.demos.blackhole.qwen36.tt.attention import v_rotate

    def v4_unsupported(path):
        if v4:
            raise ValueError(f"The 4-bit V cache (bfp8_v4r) is not implemented for the {path} attention path")

    # Full 27B stores only fused Q+gate/K/V, shared by decode, M=2 and prefill.
    # Other modes retain their existing projection layout and precision.
    ckc = compute_kernel_config

    def decode_linear(x, weight, program_config=None):
        """Treat every leading serving dimension as M rows."""
        original_shape = list(x.shape)
        rows = 1
        for dim in original_shape[:-1]:
            rows *= int(dim)
        if optimized:
            if x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if program_config is None or rows > 32:
                program_config = _bounded_projection_program_config(rows, int(weight.shape[-2]), int(weight.shape[-1]))
            x = ttnn.reshape(x, [1, 1, rows, original_shape[-1]])
            out = _prefetch_linear(
                x, weight, compute_kernel_config=ckc, memory_config=memory_config,
                program_config=program_config, dtype=ttnn.bfloat16,
            )
            return ttnn.reshape(out, original_shape[:-1] + [out.shape[-1]])
        if T == 1 and rows > 1:
            x = ttnn.reshape(x, [1, 1, rows, original_shape[-1]])
            out = _prefetch_linear(
                x,
                weight,
                compute_kernel_config=ckc,
                memory_config=memory_config,
                program_config=program_config,
            )
            return ttnn.reshape(out, original_shape[:-1] + [out.shape[-1]])
        return _prefetch_linear(
            x,
            weight,
            compute_kernel_config=ckc,
            memory_config=memory_config,
            program_config=program_config,
        )
    use_fused_qkv = (optimized or T == 1 or verify_two or verify_many) and qkv_fused_weight is not None
    _attention_layout_fused = (
        optimized and B == 1 and T == 1 and use_fused_qkv
        and not verify_two and not verify_many and norm_weights_pre_offset
        and (num_attention_heads, num_key_value_heads, head_dim) == (24, 4, 256)
        and (qkv_fused_q_dim, qkv_fused_k_dim) == (12288, 1024)
        and device.get_num_devices() == 1
        and (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (11, 10)
        and page_table is not None and paged_kv_cache_key is not None
        and paged_kv_cache_value is not None and cur_pos_tensor is not None
    )
    # MTP-1 serving verification (no shadow page): the same arithmetic as the
    # generic M=2 path below, with the layout glue replaced by bit-copy/RoPE
    # kernels that already serve M1 decode.
    _attention_verify_fused = (
        optimized and verify_two and use_fused_qkv and norm_weights_pre_offset
        and shadow_fill_page_table is None and not shadow_same_page
        and (num_attention_heads, num_key_value_heads, head_dim) == (24, 4, 256)
        and (qkv_fused_q_dim, qkv_fused_k_dim) == (12288, 1024)
        and tuple(hidden_states.shape) == (1, 2, 5120)
        and device.get_num_devices() == 1
        and (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (11, 10)
        and page_table is not None and paged_kv_cache_key is not None
        and paged_kv_cache_value is not None and cur_pos_tensor is not None
    )
    if _attention_verify_fused:
        from models.demos.blackhole.qwen36.tt.attention.split_qkg import split_qkg_two
        from models.demos.blackhole.qwen36.tt.attention.rope_to_sdpa import rope_to_sdpa_two

        qkv_fused = decode_linear(hidden_states, qkv_fused_weight, program_config=qkv_fused_program_config)
        query_states, key_states, gate = split_qkg_two(qkv_fused)
        query_normed = ttnn.rms_norm(query_states, weight=q_norm_weight, epsilon=norm_eps)
        ttnn.deallocate(query_states)
        key_normed = ttnn.rms_norm(key_states, weight=k_norm_weight, epsilon=norm_eps)
        ttnn.deallocate(key_states)
    if _attention_verify_fused and _os.environ.get("QWEN36_VERIFY_SDPA", "shared") == "shared":
        # MTP-1 rows p, p+1: RoPE + direct live-KV rows (the bytes rope_to_sdpa_two +
        # paged_update_cache store) and one shared-KV SDPA (verify_sdpa.py), bitwise equal to the
        # B=2 paged SDPA below while each KV chunk leaves DRAM once.
        from models.demos.blackhole.qwen36.tt.attention.rope_to_sdpa import rope_kv_many
        from models.demos.blackhole.qwen36.tt.attention.verify_sdpa import packed_q_shape, verify_sdpa

        q_packed = ttnn.empty(
            packed_q_shape(2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_rows = v_rotate.rotate_v_rows(qkv_fused) if v4 else None
        rope_kv_many(query_normed, key_normed, cos, sin, qkv_fused, q_packed,
                     paged_kv_cache_key, paged_kv_cache_value, cur_pos_tensor, page_table, v_rows=v_rows)
        ttnn.deallocate(query_normed)
        ttnn.deallocate(key_normed)
        ttnn.deallocate(qkv_fused)
        if v4:
            ttnn.deallocate(v_rows)
        attn_output = verify_sdpa(
            q_packed, paged_kv_cache_key, paged_kv_cache_value, cur_pos_tensor, page_table, scaling,
        )
        ttnn.deallocate(q_packed)
        if v4:
            rotated, attn_output = attn_output, v_rotate.unrotate_rows(attn_output)
            ttnn.deallocate(rotated)
        gate_act = ttnn.sigmoid(gate)
        ttnn.deallocate(gate)
        gated = ttnn.multiply(attn_output, gate_act)
        ttnn.deallocate(gate_act)
        ttnn.deallocate(attn_output)
        output = decode_linear(gated, o_proj_weight, program_config=o_decode_program_config)
        ttnn.deallocate(gated)
        return output, None, None
    if _attention_verify_fused:
        q_decode = ttnn.empty(
            (1, 2, num_attention_heads, head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_rows = v_rotate.rotate_v_rows(qkv_fused) if v4 else None
        kv_shards = [
            rope_to_sdpa_two(query_normed, key_normed, cos, sin, qkv_fused, token, q_decode, v_rows=v_rows)
            for token in (0, 1)
        ]
        ttnn.deallocate(query_normed)
        ttnn.deallocate(key_normed)
        ttnn.deallocate(qkv_fused)
        if v4:
            ttnn.deallocate(v_rows)
        rows = verify_rows if verify_rows is not None else tuple(
            (cur_pos_tensor[t : t + 1], page_table[t : t + 1, :]) for t in range(2)
        )
        # Positions p and p+1 share one cache tile: paged_update_cache is a
        # read-modify-write per core, so the two rows stay dependent calls.
        for cache, index in ((paged_kv_cache_key, 0), (paged_kv_cache_value, 1)):
            for shards, (position_row, table_row) in zip(kv_shards, rows):
                ttnn.experimental.paged_update_cache(
                    cache, shards[index], update_idxs_tensor=position_row, page_table=table_row
                )
        for key_shard, value_shard in kv_shards:
            ttnn.deallocate(key_shard)
            ttnn.deallocate(value_shard)
        attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_decode,
            paged_kv_cache_key,
            paged_kv_cache_value,
            cur_pos_tensor=cur_pos_tensor,
            page_table_tensor=page_table,
            is_causal=True,
            scale=scaling,
            program_config=_get_paged_sdpa_decode_program_config(
                device, paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
            ),
        )
        ttnn.deallocate(q_decode)
        attn_output = ttnn.reshape(attn_output, [1, 2, num_attention_heads * head_dim])
        if v4:
            rotated, attn_output = attn_output, v_rotate.unrotate_rows(attn_output)
            ttnn.deallocate(rotated)
        gate_act = ttnn.sigmoid(gate)
        ttnn.deallocate(gate)
        gated = ttnn.multiply(attn_output, gate_act)
        ttnn.deallocate(gate_act)
        ttnn.deallocate(attn_output)
        output = decode_linear(gated, o_proj_weight, program_config=o_decode_program_config)
        ttnn.deallocate(gated)
        return output, None, None
    # Wide MTP/DFlash verification (T=3..8 rows) whose caller accepts direct live-KV
    # writes (verify_groups given): rejected rows sit strictly beyond the
    # committed context and the next transaction rewrites them before any read,
    # exactly as in MTP-1. Same arithmetic as the transactional path; SDPA
    # stays in the reference's per-KV-head core split (groups of <=2 rows).
    _attention_verify_many_fused = (
        optimized and verify_many and verify_groups is not None and not tree_fork_position
        and use_fused_qkv and norm_weights_pre_offset
        and (num_attention_heads, num_key_value_heads, head_dim) == (24, 4, 256)
        and (qkv_fused_q_dim, qkv_fused_k_dim) == (12288, 1024)
        and len(hidden_states.shape) == 3 and 3 <= int(hidden_states.shape[1]) <= 8
        and int(hidden_states.shape[0]) == 1 and int(hidden_states.shape[2]) == 5120
        and device.get_num_devices() == 1
        and (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (11, 10)
        and page_table is not None and paged_kv_cache_key is not None
        and paged_kv_cache_value is not None and cur_pos_tensor is not None
    )
    if verify_many and verify_groups is not None and not _attention_verify_many_fused:
        raise ValueError("Direct-KV wide verification requires the fused single-P150 T=3..8 attention path")
    if _attention_verify_many_fused:
        from models.demos.blackhole.qwen36.tt.attention.split_qkg import split_qkg_many
        from models.demos.blackhole.qwen36.tt.attention.rope_to_sdpa import rope_kv_many, verify_row_groups
        from models.demos.blackhole.qwen36.tt.attention.merge_rows import merge_group_rows

        tokens = int(hidden_states.shape[1])
        if [rows for rows, _, _ in verify_groups] != verify_row_groups(tokens):
            raise ValueError("verify_groups must be the consecutive verify_row_groups covering every row")
        qkv_fused = decode_linear(hidden_states, qkv_fused_weight, program_config=qkv_fused_program_config)
        query_states, key_states, gate = split_qkg_many(qkv_fused)
        query_normed = ttnn.rms_norm(query_states, weight=q_norm_weight, epsilon=norm_eps)
        ttnn.deallocate(query_states)
        key_normed = ttnn.rms_norm(key_states, weight=k_norm_weight, epsilon=norm_eps)
        ttnn.deallocate(key_states)
        v_rows = v_rotate.rotate_v_rows(qkv_fused) if v4 else None
        shared_sdpa = _os.environ.get("QWEN36_VERIFY_SDPA", "shared") == "shared"
        if shared_sdpa:
            # DFlash / wide-MTP rows: one shared-KV SDPA launch (verify_sdpa.py), bitwise equal to
            # the grouped per-row SDPA below while each KV chunk leaves DRAM once per row pass.
            from models.demos.blackhole.qwen36.tt.attention.verify_sdpa import packed_q_shape, verify_sdpa

            q_packed = ttnn.empty(
                packed_q_shape(tokens), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            rope_kv_many(query_normed, key_normed, cos, sin, qkv_fused, q_packed,
                         paged_kv_cache_key, paged_kv_cache_value, cur_pos_tensor, page_table, v_rows=v_rows)
            ttnn.deallocate(query_normed)
            ttnn.deallocate(key_normed)
            ttnn.deallocate(qkv_fused)
            attn_output = verify_sdpa(
                q_packed, paged_kv_cache_key, paged_kv_cache_value, cur_pos_tensor, page_table, scaling,
            )
            ttnn.deallocate(q_packed)
        else:
            group_queries = [
                ttnn.empty(
                    (1, len(rows), num_attention_heads, head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for rows, _, _ in verify_groups
            ]
            # One launch: RoPE for every row, queries into the SDPA groups, K/V rows written
            # straight into the paged caches (the bytes per-row paged_update_cache stored).
            rope_kv_many(query_normed, key_normed, cos, sin, qkv_fused, group_queries,
                         paged_kv_cache_key, paged_kv_cache_value, cur_pos_tensor, page_table, v_rows=v_rows)
            ttnn.deallocate(query_normed)
            ttnn.deallocate(key_normed)
            ttnn.deallocate(qkv_fused)
            max_seq = paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
            group_outputs = []
            for (rows, group_positions, group_table), q_group in zip(verify_groups, group_queries):
                program_config = _get_paged_sdpa_decode_program_config(device, max_seq)
                # Every group keeps the two-row 8x8 reference split: 8 cores per (row, KV head).
                # Up to three rows fit an 11x9 grid (96 active cores) with that same split.
                reference = program_config.compute_with_storage_grid_size
                program_config.max_cores_per_head_batch = min(
                    program_config.max_cores_per_head_batch,
                    max(1, reference.x * reference.y // (2 * num_key_value_heads)),
                )
                if len(rows) == 3:
                    program_config.compute_with_storage_grid_size = ttnn.CoreCoord(11, 9)
                out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q_group,
                    paged_kv_cache_key,
                    paged_kv_cache_value,
                    cur_pos_tensor=group_positions,
                    page_table_tensor=group_table,
                    is_causal=True,
                    scale=scaling,
                    program_config=program_config,
                )
                ttnn.deallocate(q_group)
                group_outputs.append(out)
            # Bit copy equal to per-group reshape to [1,B,6144] plus a row concat.
            attn_output = merge_group_rows(group_outputs)
            for out in group_outputs:
                ttnn.deallocate(out)
        if v4:
            ttnn.deallocate(v_rows)
            rotated, attn_output = attn_output, v_rotate.unrotate_rows(attn_output)
            ttnn.deallocate(rotated)
        gate_act = ttnn.sigmoid(gate)
        ttnn.deallocate(gate)
        gated = ttnn.multiply(attn_output, gate_act)
        ttnn.deallocate(gate_act)
        ttnn.deallocate(attn_output)
        output = decode_linear(gated, o_proj_weight, program_config=o_decode_program_config)
        ttnn.deallocate(gated)
        return output, None, None
    if _attention_layout_fused:
        from models.demos.blackhole.qwen36.tt.attention.split_qkg import split_qkg
        from models.demos.blackhole.qwen36.tt.attention.rope_to_sdpa import rope_to_sdpa
        qkv_fused = decode_linear(hidden_states, qkv_fused_weight, program_config=qkv_fused_program_config)
        query_states, key_states, gate = split_qkg(qkv_fused)
        query_states = ttnn.rms_norm(query_states, weight=q_norm_weight, epsilon=norm_eps)
        key_states = ttnn.rms_norm(key_states, weight=k_norm_weight, epsilon=norm_eps)
        v_rows = v_rotate.rotate_v_rows(qkv_fused) if v4 else None
        q_decode, k_sharded, v_sharded = rope_to_sdpa(query_states, key_states, cos, sin, qkv_fused, v_rows=v_rows)
        ttnn.deallocate(query_states)
        ttnn.deallocate(key_states)
        ttnn.deallocate(qkv_fused)
        if v4:
            ttnn.deallocate(v_rows)
        query_states = q_decode
    else:
        prefill_v4 = v4 and T > 1 and chunk_page_table is not None and not verify_two and not verify_many
        if v4 and not prefill_v4:
            v4_unsupported("unfused decode")
        if use_fused_qkv:
            qkv_fused = decode_linear(
                hidden_states,
                qkv_fused_weight,
                program_config=qkv_fused_program_config if (T == 1 or verify_two or verify_many) else None,
            )
            qg = qkv_fused[..., :qkv_fused_q_dim]
            qg = ttnn.to_layout(qg, ttnn.TILE_LAYOUT)
            key_states = qkv_fused[
                ..., qkv_fused_q_dim : qkv_fused_q_dim + qkv_fused_k_dim
            ]
            key_states = ttnn.to_layout(key_states, ttnn.TILE_LAYOUT)
            if prefill_v4:
                # 4-bit V cache: one launch rotates, rounds (RNE) and writes this chunk's V straight
                # into the paged V cache (the V slice, typecast and paged_fill_cache it replaces).
                if (qkv_fused_q_dim, qkv_fused_k_dim) != (12288, 1024) or T % 32:
                    v4_unsupported("non-tile-aligned prefill")
                v_rotate.fill_v_paged(
                    qkv_fused, paged_kv_cache_value, chunk_page_table,
                    min(int(chunk_page_table.shape[1]) * int(paged_kv_cache_value.shape[2]), T),
                )
                value_states = None
            else:
                value_states = qkv_fused[..., qkv_fused_q_dim + qkv_fused_k_dim :]
                value_states = ttnn.to_layout(value_states, ttnn.TILE_LAYOUT)
            ttnn.deallocate(qkv_fused)
        else:
            qg = decode_linear(
                hidden_states,
                q_proj_weight,
                program_config=q_decode_program_config if (T == 1 or verify_two or verify_many) else None,
            )
        if (
            T > 1 and use_fused_qkv and len(qg.shape) == 3
            and _tile_heads_ok(qg, T, 2 * head_dim)
            and _tile_heads_ok(key_states, T, head_dim)
            and (prefill_v4 or _tile_heads_ok(value_states, T, head_dim))
        ):
            # Prefill: move whole tiles instead of padded [B,T,H,D] reshapes and transposes.
            # A flat [B,T,H*W] tile buffer is ordered [B,T/32,H,W/32], i.e. it is exactly a
            # [B,T/32,H,32,W] tile tensor; slicing W and permuting (T/32,H) yields the
            # head-major tensors. Pure data movement: values are unchanged.
            qg5 = ttnn.experimental.view(qg, [B, T // 32, num_attention_heads, 32, 2 * head_dim])
            q5 = ttnn.slice(qg5, [0, 0, 0, 0, 0], [B, T // 32, num_attention_heads, 32, head_dim])
            g5 = ttnn.slice(qg5, [0, 0, 0, 0, head_dim], [B, T // 32, num_attention_heads, 32, 2 * head_dim])
            ttnn.deallocate(qg)
            gate = ttnn.experimental.view(g5, [B, T, num_attention_heads * head_dim])

            def head_major(t5, heads):
                t = ttnn.permute(t5, (0, 2, 1, 3, 4))
                return ttnn.reshape(t, [B, heads, T, head_dim])

            query_states = head_major(q5, num_attention_heads)
            ttnn.deallocate(q5)
            k5 = ttnn.experimental.view(key_states, [B, T // 32, num_key_value_heads, 32, head_dim])
            key_states = head_major(k5, num_key_value_heads)
            if not prefill_v4:
                v5 = ttnn.experimental.view(value_states, [B, T // 32, num_key_value_heads, 32, head_dim])
                value_states = head_major(v5, num_key_value_heads)
            if norm_weights_pre_offset:
                query_states = ttnn.rms_norm(query_states, weight=q_norm_weight, epsilon=norm_eps)
                key_states = ttnn.rms_norm(key_states, weight=k_norm_weight, epsilon=norm_eps)
            else:
                query_states = rms_norm_zero_centered_ttnn(query_states, q_norm_weight, eps=norm_eps)
                key_states = rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps)
        else:
            if prefill_v4:
                v4_unsupported("generic-layout prefill")
            qg = ttnn.reshape(qg, [B, T, num_attention_heads, head_dim * 2])
            # Split into query and gate
            query_states, gate = ttnn.chunk(qg, 2, dim=-1)
            ttnn.deallocate(qg)
            gate = ttnn.reshape(gate, [B, T, num_attention_heads * head_dim])

            # Q norm + transpose to [B, H_q, T, D]
            if norm_weights_pre_offset:
                query_states = ttnn.rms_norm(query_states, weight=q_norm_weight, epsilon=norm_eps)
            else:
                query_states = rms_norm_zero_centered_ttnn(query_states, q_norm_weight, eps=norm_eps)
            query_states = ttnn.transpose(query_states, 1, 2)

            # K projection + norm + transpose to [B, H_kv, T, D]
            if not use_fused_qkv:
                key_states = decode_linear(
                    hidden_states,
                    k_proj_weight,
                    program_config=k_decode_program_config if (T == 1 or verify_two or verify_many) else None,
                )
            key_states = ttnn.reshape(key_states, [B, T, num_key_value_heads, head_dim])
            if norm_weights_pre_offset:
                key_states = ttnn.rms_norm(key_states, weight=k_norm_weight, epsilon=norm_eps)
            else:
                key_states = rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps)
            key_states = ttnn.transpose(key_states, 1, 2)

            # V projection + transpose to [B, H_kv, T, D]
            if not use_fused_qkv:
                value_states = decode_linear(
                    hidden_states,
                    v_proj_weight,
                    program_config=v_decode_program_config if (T == 1 or verify_two or verify_many) else None,
                )
            value_states = ttnn.reshape(value_states, [B, T, num_key_value_heads, head_dim])
            value_states = ttnn.transpose(value_states, 1, 2)

        # RoPE — for decode (T==1), use reshape (metadata-only) instead of unsqueeze (data movement)
        if T == 1 and len(cos.shape) == 3:
            cos_4d = ttnn.reshape(cos, [cos.shape[0], 1, 1, cos.shape[-1]])
            sin_4d = ttnn.reshape(sin, [sin.shape[0], 1, 1, sin.shape[-1]])
            query_states, key_states = apply_rotary_pos_emb_ttnn(query_states, key_states, cos_4d, sin_4d)
        else:
            query_states, key_states = apply_rotary_pos_emb_ttnn(query_states, key_states, cos, sin)

    # KV cache handling
    _use_sdpa_decode = False
    _paged_sdpa_done = False
    if verify_many:
        v4_unsupported("transactional verification")
        from models.demos.blackhole.qwen36.tt.attention.verify import causal_verify_many

        if verify_two or paged_kv_cache_key is None or paged_kv_cache_value is None or page_table is None:
            raise ValueError("verify_many requires paged KV caches and cannot be combined with verify_two")
        attn_output, transaction = causal_verify_many(
            query_states, key_states, value_states,
            paged_kv_cache_key, paged_kv_cache_value, cur_pos_tensor, page_table, scaling,
            _get_paged_sdpa_decode_program_config(
                device, paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
            ),
            fork_position=tree_fork_position,
        )
        new_key = transaction
        new_value = None
        _paged_sdpa_done = True
    elif verify_two:
        v4_unsupported("transactional MTP-1 verification")
        if paged_kv_cache_key is None or page_table is None:
            raise ValueError("verify_two requires paged KV caches and a two-row page table")
        if shadow_fill_page_table is not None:
            # Snapshot the destination logical block before provisional writes. Row 1
            # maps that logical block to the shadow physical page.
            key_block = paged_kv_cache_key[shadow_source_page : shadow_source_page + 1, :, :, :]
            value_block = paged_kv_cache_value[shadow_source_page : shadow_source_page + 1, :, :, :]
            ttnn.experimental.paged_fill_cache(
                paged_kv_cache_key, key_block, shadow_fill_page_table, batch_idx=0
            )
            ttnn.experimental.paged_fill_cache(
                paged_kv_cache_value, value_block, shadow_fill_page_table, batch_idx=0
            )

        # key_states/value_states are [1, H_kv, 2, D]. Positions p and p+1 share one
        # 32-row tile, and paged_update_cache does a per-core read-modify-write of that
        # tile, so a single two-row call races (one token's K/V is dropped
        # nondeterministically). Write the two tokens in two dependent single-row calls.
        worker_grid = device.compute_with_storage_grid_size()
        shard1 = ttnn.num_cores_to_corerangeset(1, worker_grid, row_wise=True)
        sharded1 = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard1, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )

        def provisional_write(cache, tokens, positions, table):
            for t in range(2):
                value = ttnn.to_memory_config(tokens[:, t : t + 1, :, :], sharded1)
                ttnn.experimental.paged_update_cache(
                    cache, value, update_idxs_tensor=positions[t : t + 1], page_table=table[t : t + 1, :]
                )
                ttnn.deallocate(value)

        k_tokens = ttnn.permute(key_states, (0, 2, 1, 3))  # [1, 2, H_kv, D]
        v_tokens = ttnn.permute(value_states, (0, 2, 1, 3))
        key1 = k_tokens[:, 1:2, :, :]
        value1 = v_tokens[:, 1:2, :, :]
        provisional_write(paged_kv_cache_key, k_tokens, cur_pos_tensor, page_table)
        provisional_write(paged_kv_cache_value, v_tokens, cur_pos_tensor, page_table)
        if shadow_same_page:
            # Token 0 must also be visible in the shadow page for token 1's causal read.
            pos0 = cur_pos_tensor[0:1]
            shadow_table = page_table[1:2, :]
            for cache, tokens in ((paged_kv_cache_key, k_tokens), (paged_kv_cache_value, v_tokens)):
                shadow0 = ttnn.to_memory_config(tokens[:, 0:1, :, :], sharded1)
                ttnn.experimental.paged_update_cache(
                    cache, shadow0, update_idxs_tensor=pos0, page_table=shadow_table
                )
                ttnn.deallocate(shadow0)

        # query_states is [1, H, 2, D]; paged decode wants [S=1, B=2, H, D].
        # A reshape would interleave heads across the two tokens.
        q_decode = ttnn.permute(query_states, (0, 2, 1, 3))
        q_decode = ttnn.to_memory_config(q_decode, ttnn.DRAM_MEMORY_CONFIG)
        attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_decode,
            paged_kv_cache_key,
            paged_kv_cache_value,
            cur_pos_tensor=cur_pos_tensor,
            page_table_tensor=page_table,
            is_causal=True,
            scale=scaling,
            program_config=_get_paged_sdpa_decode_program_config(
                device, paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
            ),
        )
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        new_key = key1
        new_value = value1
        _paged_sdpa_done = True
        S_total = paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif paged_kv_cache_key is not None and page_table is not None and T > 1 and chunk_page_table is not None:
        # Paged prefill: fill K/V into paged cache, then chunked SDPA.
        # Q/K/V stay bfloat16 — no typecast. Production models (Qwen3_VL) typecast to
        # bfloat8_b, but on Blackhole P150 this causes L1 clashes in downstream ops
        # (head concat, gate multiply, output projection) whose programs were compiled
        # for bfloat16. The core fix (paged KV) eliminates the growing-concat L1 issue;
        # the bfloat8_b optimization can be added later with matching downstream changes.

        if paged_kv_cache_key.dtype != key_states.dtype:
            # BFP8 native-context cache: paged_fill_cache stores its input format, so convert with
            # the packer's BF16->BFP8 conversion (as the decode-row updates do) BEFORE the page
            # slice, whose cache-dtype programs _warmup_paged_fill_widths compiles.
            key_cast = ttnn.typecast(key_states, paged_kv_cache_key.dtype)
            ttnn.deallocate(key_states)
            key_states = key_cast
            if not v4:  # the 4-bit V cache was already written by v_rotate.fill_v_paged
                value_cast = ttnn.typecast(value_states, paged_kv_cache_value.dtype)
                ttnn.deallocate(value_states)
                value_states = value_cast

        # Slice K/V to page_len to handle tile-padded tensors.
        block_size_cache = paged_kv_cache_key.shape[2]
        page_len = chunk_page_table.shape[1] * block_size_cache
        key_fill = key_states[:, :, :page_len, :] if page_len < key_states.shape[2] else key_states
        ttnn.experimental.paged_fill_cache(paged_kv_cache_key, key_fill, chunk_page_table, batch_idx=0)
        ttnn.deallocate(key_states)
        if not v4:
            value_fill = value_states[:, :, :page_len, :] if page_len < value_states.shape[2] else value_states
            ttnn.experimental.paged_fill_cache(paged_kv_cache_value, value_fill, chunk_page_table, batch_idx=0)
            ttnn.deallocate(value_states)

        if chunk_start_idx_tensor is not None:
            # Flexible chunked SDPA: chunk_start_idx is a runtime device tensor and the
            # program config is fixed (q/k_chunk=64), so a single captured trace replays
            # for every chunk position (chunk-outer per-chunk prefill).
            attn_output = ttnn.transformer.chunked_scaled_dot_product_attention(
                query_states,
                paged_kv_cache_key,
                paged_kv_cache_value,
                page_table,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                scale=scaling,
                program_config=_get_flexible_sdpa_program_config(device),
                compute_kernel_config=_get_sdpa_compute_kernel_config(),
            )
        else:
            attn_output = ttnn.transformer.chunked_scaled_dot_product_attention(
                query_states,
                paged_kv_cache_key,
                paged_kv_cache_value,
                page_table,
                chunk_start_idx,
                scale=scaling,
                program_config=_get_sdpa_program_config(device, T, chunk_start_idx=chunk_start_idx),
                compute_kernel_config=_get_sdpa_compute_kernel_config(),
            )
        if v4:
            # R^T per Q head, written token-major [B,T,H*D] (replaces the head relayout below).
            rotated, attn_output = attn_output, v_rotate.unrotate_heads_major(attn_output)
            ttnn.deallocate(rotated)

        new_key = paged_kv_cache_key
        new_value = paged_kv_cache_value
        _paged_sdpa_done = True
        S_total = paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif page_table is not None and paged_kv_cache_key is not None and T == 1:
        # Paged attention decode: write K/V via page_table, attend via paged SDPA.
        # paged_update_cache with page_table expects input [1, B, num_kv_heads, head_dim],
        # where page_table.shape[0] == input.shape[1] == B.
        # This is different from the non-paged path which uses [1, B*num_kv_heads, T, head_dim].
        if not _attention_layout_fused:
            k_for_paged = ttnn.reshape(key_states, [1, B, num_key_value_heads, head_dim])
            v_for_paged = ttnn.reshape(value_states, [1, B, num_key_value_heads, head_dim])
            # Shard to L1 (HEIGHT_SHARDED on B cores) across the full 2D worker grid.
            _worker_grid = device.compute_with_storage_grid_size()
            _shard_grid = ttnn.num_cores_to_corerangeset(B, _worker_grid, row_wise=True)
            _shard_spec = ttnn.ShardSpec(_shard_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
            _sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec)
            k_sharded = ttnn.to_memory_config(k_for_paged, _sharded_mc)
            v_sharded = ttnn.to_memory_config(v_for_paged, _sharded_mc)
        # Write K/V into paged cache via page_table
        ttnn.experimental.paged_update_cache(
            paged_kv_cache_key, k_sharded, update_idxs_tensor=cur_pos_tensor, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            paged_kv_cache_value, v_sharded, update_idxs_tensor=cur_pos_tensor, page_table=page_table
        )
        ttnn.deallocate(k_sharded)
        ttnn.deallocate(v_sharded)
        # Paged SDPA decode
        if not _attention_layout_fused:
            q_decode = ttnn.reshape(query_states, [1, B, num_attention_heads, head_dim])
            # paged_scaled_dot_product_attention_decode requires Q in DRAM when not sharded.
            q_decode = ttnn.to_memory_config(q_decode, ttnn.DRAM_MEMORY_CONFIG)
        attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_decode,
            paged_kv_cache_key,
            paged_kv_cache_value,
            cur_pos_tensor=cur_pos_tensor,
            page_table_tensor=page_table,
            is_causal=True,
            scale=scaling,
            program_config=_get_paged_sdpa_decode_program_config(
                device, paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
            ),
        )
        if v4:
            # [1,1,24,256] (rows = Q heads) -> R^T per head, the same row products as verification.
            rotated, attn_output = attn_output, v_rotate.unrotate_rows(attn_output)
            ttnn.deallocate(rotated)
        if not _attention_layout_fused:
            attn_output = ttnn.reshape(attn_output, [B, num_attention_heads, 1, head_dim])
        new_key = paged_kv_cache_key
        new_value = paged_kv_cache_value
        _paged_sdpa_done = True  # Skip the bottom SDPA section (attn_output already computed)
        S_total = paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif v4:
        v4_unsupported("non-paged")
    elif cur_pos_tensor is not None and kv_cache_key is not None and T == 1:
        # Trace-compatible SDPA decode: write K/V at dynamic position, attend up to cur_pos.
        # Uses paged_update_cache (takes device tensor for index, trace-compatible) +
        # sdpa_decode (reads cache[:, :, :cur_pos+1, :], no mask needed).
        #
        # paged_update_cache expects: input [1, N, seq, D] HEIGHT_SHARDED, cache [N, 1, max_seq, D].
        # N = batch * num_kv_heads. Pre-allocated sharded buffers (paged_k_buf, paged_v_buf)
        # are provided by the caller to avoid allocating during trace capture.
        N = B * num_key_value_heads
        max_seq = kv_cache_key.shape[2]
        k_for_cache = ttnn.reshape(key_states, [1, N, T, head_dim])
        v_for_cache = ttnn.reshape(value_states, [1, N, T, head_dim])
        # Convert to sharded for paged_update_cache. Use pre-allocated sharded buffers
        # when available (trace path) to avoid allocation during trace capture.
        if paged_k_buf is not None:
            k_sharded = ttnn.interleaved_to_sharded(
                k_for_cache, paged_k_buf.memory_config(), preallocated_output=paged_k_buf
            )
            v_sharded = ttnn.interleaved_to_sharded(
                v_for_cache, paged_v_buf.memory_config(), preallocated_output=paged_v_buf
            )
        else:
            # Eager path: allocate on the full 2D worker grid.
            _worker_grid = device.compute_with_storage_grid_size()
            _shard_grid = ttnn.num_cores_to_corerangeset(N, _worker_grid, row_wise=True)
            _shard_spec = ttnn.ShardSpec(_shard_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
            _sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec)
            k_sharded = ttnn.to_memory_config(k_for_cache, _sharded_mc)
            v_sharded = ttnn.to_memory_config(v_for_cache, _sharded_mc)
        kv_cache_key_reshaped = ttnn.reshape(kv_cache_key, [N, 1, max_seq, head_dim])
        kv_cache_value_reshaped = ttnn.reshape(kv_cache_value, [N, 1, max_seq, head_dim])
        ttnn.experimental.paged_update_cache(kv_cache_key_reshaped, k_sharded, update_idxs_tensor=cur_pos_tensor)
        ttnn.experimental.paged_update_cache(kv_cache_value_reshaped, v_sharded, update_idxs_tensor=cur_pos_tensor)
        new_key = kv_cache_key
        new_value = kv_cache_value
        _use_sdpa_decode = True
        S_total = kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif trace_new_k_buf is not None and trace_attn_mask is not None:
        # Legacy trace mode: staging position + mask (fallback when cur_pos_tensor not available)
        ttnn.copy(key_states, trace_new_k_buf)
        ttnn.copy(value_states, trace_new_v_buf)
        if trace_kv_pad_zeros is not None and trace_staging_pos is not None:
            k_padded = ttnn.concat([key_states, trace_kv_pad_zeros], dim=2)
            v_padded = ttnn.concat([value_states, trace_kv_pad_zeros], dim=2)
            ttnn.update_cache(kv_cache_key, k_padded, update_idx=trace_staging_pos)
            ttnn.update_cache(kv_cache_value, v_padded, update_idx=trace_staging_pos)
            ttnn.deallocate(k_padded)
            ttnn.deallocate(v_padded)
        key_states = kv_cache_key
        value_states = kv_cache_value
        new_key = kv_cache_key
        new_value = kv_cache_value
        S_total = kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif kv_cache_key is not None and cache_pos is not None and cur_pos_tensor is not None and T == 1:
        # Decode with SDPA decode variant: write K/V at cache_pos, attend with cur_pos_tensor.
        # sdpa_decode reads cache[:, :, :cur_pos+1, :] internally — no slicing or mask needed.
        ttnn.copy(key_states, kv_cache_key[:, :, cache_pos : cache_pos + T, :])
        ttnn.copy(value_states, kv_cache_value[:, :, cache_pos : cache_pos + T, :])

        new_key = kv_cache_key
        new_value = kv_cache_value
        _use_sdpa_decode = True
        S_total = kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif kv_cache_key is not None and cache_pos is not None:
        # Pre-allocated KV cache mode: write new K/V at cache_pos, read 0..cache_pos+T
        # Write new tokens into cache at position cache_pos
        # kv_cache_key shape: [B, H_kv, max_seq_len, D]
        ttnn.copy(key_states, kv_cache_key[:, :, cache_pos : cache_pos + T, :])
        ttnn.copy(value_states, kv_cache_value[:, :, cache_pos : cache_pos + T, :])

        # Read valid portion of cache for attention
        valid_len = cache_pos + T
        key_states = kv_cache_key[:, :, :valid_len, :]
        key_states = ttnn.to_layout(key_states, ttnn.TILE_LAYOUT)
        value_states = kv_cache_value[:, :, :valid_len, :]
        value_states = ttnn.to_layout(value_states, ttnn.TILE_LAYOUT)

        new_key = kv_cache_key
        new_value = kv_cache_value
        S_total = valid_len
        is_causal = T > 1  # Only causal during prefill
        segmented_attn_mask = None
    elif past_key is not None:
        # Legacy concat mode — used during segmented prefill
        past_len = past_key.shape[2]
        key_states = ttnn.concat([past_key, key_states], dim=2)
        value_states = ttnn.concat([past_value, value_states], dim=2)
        new_key = key_states
        new_value = value_states
        S_total = key_states.shape[2]
        if T > 1 and S_total > T:
            # Segmented prefill: Q_len != KV_len, need explicit causal mask
            # Each query position i (in current segment) can attend to:
            #   positions 0..past_len+i (all past tokens + tokens up to i in current segment)
            import torch as _torch

            row_idx = _torch.arange(T).unsqueeze(1)  # [T, 1]
            col_idx = _torch.arange(S_total).unsqueeze(0)  # [1, S_total]
            # Mask future positions: col > past_len + row means future token
            mask = (
                _torch.where(
                    col_idx > past_len + row_idx,
                    _torch.tensor(-1e4, dtype=_torch.bfloat16),
                    _torch.tensor(0.0, dtype=_torch.bfloat16),
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )  # [1, 1, T, S_total]
            segmented_attn_mask = ttnn.from_torch(
                mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            is_causal = False
        else:
            segmented_attn_mask = None
            is_causal = False
    else:
        # First call (prefill, no cache yet)
        new_key = key_states
        new_value = value_states
        S_total = key_states.shape[2]
        is_causal = True
        segmented_attn_mask = None

    # Fused scaled dot-product attention (skipped if paged SDPA already computed above)
    if _paged_sdpa_done:
        pass  # attn_output already set by paged SDPA decode path
    elif _use_sdpa_decode:
        # Decode-optimized SDPA: reads cache[:, :, :cur_pos+1, :] internally.
        # sdpa_decode expects Q as [1, B, H, D] (not [B, H, 1, D] like regular SDPA).
        # Transpose from [B=1, H, T=1, D] → [T=1, B=1, H, D].
        q_decode = ttnn.transpose(query_states, 1, 2)  # [1, 1, 16, 256]
        # cur_pos_tensor has B*H_kv elements; sdpa_decode needs B elements.
        sdpa_pos = cur_pos_tensor[:B] if cur_pos_tensor.shape[0] > B else cur_pos_tensor
        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode,
            kv_cache_key,
            kv_cache_value,
            cur_pos_tensor=sdpa_pos,
            scale=scaling,
            program_config=_get_sdpa_program_config(device, S_total, q_seq_len=T),
            compute_kernel_config=_get_sdpa_compute_kernel_config(),
        )
        # Transpose back to [B, H, T, D] for head concatenation
        attn_output = ttnn.transpose(attn_output, 1, 2)
    else:
        if trace_new_k_buf is not None:
            _attn_mask = trace_attn_mask
        elif past_key is not None and segmented_attn_mask is not None:
            _attn_mask = segmented_attn_mask
        else:
            _attn_mask = None
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=_attn_mask,
            is_causal=is_causal,
            scale=scaling,
            program_config=_get_sdpa_program_config(device, S_total, q_seq_len=T),
            compute_kernel_config=_get_sdpa_compute_kernel_config(),
        )
    ttnn.deallocate(query_states)

    # Convert from [B, H, T, D] back to [B, T, H*D]
    if v4 and T > 1:
        pass  # v_rotate.unrotate_heads_major already wrote the token-major [B,T,H*D] rows
    elif T == 1:
        # For a single query row [B, H, 1, D] the head concat is a pure reinterpretation:
        # the H tiles of one row are already contiguous in H*D order.
        attn_output = ttnn.reshape(attn_output, [B, 1, num_attention_heads * head_dim])
    elif use_optimized_concat and _os.environ.get("QWEN36_OPTIMIZED_27B") != "1":
        # concatenate_heads stages every head tile of a row on one core (~786 KB of
        # static CBs for H=24, D=256); that collides with the 749 KB/core L1-resident
        # GDN state of the full 27B, so it takes the interleaved transpose path below.
        attn_output = ttnn.transformer.concatenate_heads(attn_output)
    elif _tile_heads_ok(attn_output, T, head_dim):
        # [B,H,T,D] tiles are ordered [B,H,T/32,D/32]: exchange only the head and time-block
        # axes (whole tiles) and reinterpret [B,T/32,H,32,D] as the flat [B,T,H*D] buffer.
        heads = ttnn.reshape(attn_output, [B, num_attention_heads, T // 32, 32, head_dim])
        tokens = ttnn.permute(heads, (0, 2, 1, 3, 4))
        ttnn.deallocate(attn_output)
        attn_output = ttnn.experimental.view(tokens, [B, T, num_attention_heads * head_dim])
    else:
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, [B, T, num_attention_heads * head_dim])

    # Apply sigmoid gate
    gate = ttnn.sigmoid(gate)
    attn_output = ttnn.multiply(attn_output, gate)
    ttnn.deallocate(gate)

    # Output projection
    attn_output = decode_linear(
        attn_output,
        o_proj_weight,
        program_config=o_decode_program_config if (T == 1 or verify_two or verify_many) else None,
    )
    if flat_decode:
        attn_output = ttnn.reshape(attn_output, [1, 1, B, attn_output.shape[-1]])

    return attn_output, new_key, new_value
