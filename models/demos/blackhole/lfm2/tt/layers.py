# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM2 operator modules: RMSNorm, gated short conv (with per-sequence conv state), GQA attention with per-head q/k
RMSNorm + RoPE over a paged BFP8 KV cache, dense SwiGLU MLP.

Activations are [1, 1, T, H] BF16 tiles (T = prefill chunk rows, or the decode batch rows B).
"""
from __future__ import annotations

import math

import torch
import ttnn

from .common import CK_BFP8, CK_HIFI4, CK_NORM, free
from .conv_prefill import ConvPrefill

DRAM = ttnn.DRAM_MEMORY_CONFIG


_PREFILL_PC: dict = {}


def prefill_matmul_config(device, M: int, K: int, N: int):
    """2D-multicast program config for a prefill matmul [M, K] x [K, N] (BF16 x BFP8, FP32 accumulation) on the whole
    compute grid; None for decode-sized M (< 128 rows: the default config).  Bench: tests/bench_prefill_ops.py mm
    (2048 rows: conv in_proj 0.49 -> 0.31 ms, qkv 0.25 -> 0.17, 2048x2048 0.18 -> 0.12, dense w2 0.82 -> 0.46)."""
    if M < 128 or M % 32 or K % 32 or N % 32:
        return None
    key = (M, K, N)
    pc = _PREFILL_PC.get(key)
    if pc is None:
        grid = device.compute_with_storage_grid_size()
        mt, kt, nt = M // 32, K // 32, N // 32
        pm = -(-mt // min(grid.y, mt))
        pn = -(-nt // min(grid.x, nt))
        gy, gx = -(-mt // pm), -(-nt // pn)
        sw = max(d for d in (4, 2, 1) if pn % d == 0)
        sh = max(d for d in (4, 2, 1) if pm % d == 0 and d * sw <= 4)
        out_bytes = pm * pn * (2048 + 4096)  # output + FP32 partials
        ibw = next((d for d in (8, 4, 2, 1) if kt % d == 0
                    and out_bytes + 2 * d * (pm * 2048 + pn * 1088) <= 1350 * 1024), None)
        if ibw is None:
            pc = False
        else:
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
                per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None, fuse_batch=True)
        _PREFILL_PC[key] = pc
    return pc or None


def linear(x, w, dtype=ttnn.bfloat16, ck=CK_BFP8, memory_config=DRAM):
    pc = prefill_matmul_config(x.device(), x.shape[-2], x.shape[-1], w.shape[-1])
    return ttnn.linear(x, w, dtype=dtype, compute_kernel_config=ck, memory_config=memory_config, program_config=pc)


def norm_weight(device, w: torch.Tensor):
    """RMSNorm gamma in the [1, 1, dim/32, 32] row-major form ttnn.rms_norm takes."""
    return ttnn.from_torch(w.to(torch.bfloat16).reshape(1, 1, -1, 32), dtype=ttnn.bfloat16,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)


class RMSNorm:
    def __init__(self, device, w: torch.Tensor, eps: float):
        self.w, self.eps = norm_weight(device, w), eps

    def __call__(self, x, memory_config=DRAM):
        return ttnn.rms_norm(x, weight=self.w, epsilon=self.eps, compute_kernel_config=CK_NORM,
                             memory_config=memory_config)


class ShortConv:
    """Lfm2MoeShortConv: BCx = in_proj(x); Bx = B * x; y = C * causal_depthwise_conv3(Bx); out_proj(y).

    Conv state per sequence: the last two Bx rows, s0 = Bx[t-2], s1 = Bx[t-1] ([1, 1, B, H] BF16 tiles; row b = batch
    slot b).  Decode updates them in place (trace-safe); prefill reads them as the chunk's left context and writes
    the chunk's last two real rows back."""

    def __init__(self, device, cfg, src, layer: int, cache, batch: int = 1):
        p = f"model.layers.{layer}.conv"
        H = cfg.hidden
        self.H = H
        self.in_proj = cache.get(f"L{layer}.conv.in_proj", lambda: src.get(f"{p}.in_proj.weight").T, ttnn.bfloat8_b)
        self.out_proj = cache.get(f"L{layer}.conv.out_proj", lambda: src.get(f"{p}.out_proj.weight").T, ttnn.bfloat8_b)
        k = src.get(f"{p}.conv.weight").reshape(H, -1).to(torch.bfloat16)  # [H, 3]: taps for t-2, t-1, t
        self.taps = [ttnn.from_torch(k[:, j].reshape(1, 1, 1, H).contiguous(), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM) for j in range(3)]
        self.batch = batch
        tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
        self.s0, self.s1 = (ttnn.from_torch(torch.zeros(1, 1, batch, H, dtype=torch.bfloat16), **tile)
                            for _ in range(2))
        self.zero_row = ttnn.from_torch(torch.zeros(1, 1, 1, H, dtype=torch.bfloat16), **tile)
        # Per-slot row masks [1, 1, B, H] (1 on row b) and their complements for slot writes into the B-row state
        # tiles: state * (1 - m) + row * m is exact (products with 0 / 1 and sums with 0 do not round).
        self.masks, self.keep = [], []
        if batch > 1:
            for b in range(batch):
                m = torch.zeros(1, 1, batch, H, dtype=torch.bfloat16)
                m[:, :, b] = 1
                self.masks.append(ttnn.from_torch(m, **tile))
                self.keep.append(ttnn.from_torch(1 - m, **tile))
        self.fused_prefill = ConvPrefill(self)

    def _write_row(self, row, state, slot: int):
        """state[slot] = row ([1, 1, 1, H] tile), other rows unchanged (bit-exact)."""
        if self.batch == 1:
            ttnn.copy(row, state)
            return
        kept = ttnn.mul(state, self.keep[slot], memory_config=DRAM)
        ttnn.addcmul(kept, row, self.masks[slot], memory_config=DRAM, output_tensor=state)  # row broadcast over B
        ttnn.deallocate(kept)

    def reset(self, slot: int | None = None):
        for s in (self.s0, self.s1):
            if slot is None:
                ttnn.mul(s, 0.0, output_tensor=s)
            else:
                self._write_row(self.zero_row, s, slot)

    def save_slot(self, slot: int) -> torch.Tensor:
        """Host copy [2, H] (s0, s1 rows of `slot`; exact BF16)."""
        return torch.stack([ttnn.to_torch(s).reshape(self.batch, self.H)[slot] for s in (self.s0, self.s1)])

    def restore_slot(self, slot: int, rows: torch.Tensor):
        """Write a save_slot() copy back into `slot` (bit-exact)."""
        for s, r in zip((self.s0, self.s1), rows):
            row = ttnn.from_torch(r.reshape(1, 1, 1, self.H).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=s.device(), memory_config=DRAM)
            self._write_row(row, s, slot)
            ttnn.deallocate(row)

    def remap(self, idx: torch.Tensor):
        """Row r of the state takes old row idx[r] (rows beyond idx keep their content)."""
        for s in (self.s0, self.s1):
            host = ttnn.to_torch(s).reshape(self.batch, self.H)
            new = host.clone()
            new[: idx.numel()] = host[idx]
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(new.reshape(1, 1, self.batch, self.H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), s)

    def _bcx(self, x):
        bcx = linear(x, self.in_proj)
        T = bcx.shape[-2]
        H = self.H
        b = ttnn.slice(bcx, [0, 0, 0, 0], [1, 1, T, H])
        c = ttnn.slice(bcx, [0, 0, 0, H], [1, 1, T, 2 * H])
        xx = ttnn.slice(bcx, [0, 0, 0, 2 * H], [1, 1, T, 3 * H])
        ttnn.deallocate(bcx)
        bx = ttnn.mul(b, xx, memory_config=DRAM)
        free(b, xx)
        return bx, c

    def decode(self, x):
        bx, c = self._bcx(x)
        acc = ttnn.mul(bx, self.taps[2], memory_config=DRAM)
        acc = ttnn.addcmul(acc, self.s1, self.taps[1], memory_config=DRAM, output_tensor=acc)
        acc = ttnn.addcmul(acc, self.s0, self.taps[0], memory_config=DRAM, output_tensor=acc)
        ttnn.copy(self.s1, self.s0)
        ttnn.copy(bx, self.s1)
        y = ttnn.mul(c, acc, memory_config=DRAM)
        free(bx, c, acc)
        out = linear(y, self.out_proj)
        ttnn.deallocate(y)
        return out

    def prefill(self, x, valid: int, slot: int = 0, fresh: bool = True):
        """x: [1, 1, Tp, H] chunk (rows >= valid are padding) of the sequence in batch slot `slot`.  Left context:
        zeros for a sequence start (fresh), else the slot's state rows; afterwards the slot's state holds rows
        valid-2, valid-1 of [left0, left1, Bx].  The middle (B*x, 3-tap conv, C*, state rows) is one program
        (conv_prefill.ConvPrefill)."""
        bcx = linear(x, self.in_proj)
        y = self.fused_prefill(bcx, valid, slot=slot, fresh=fresh)
        ttnn.deallocate(bcx)
        out = linear(y, self.out_proj)
        ttnn.deallocate(y)
        return out


def rope_tables(cfg, max_pos: int):
    """HF default RoPE: cos/sin [max_pos, head_dim] in FP32, rounded to BF16 like HF's x.dtype cast."""
    d = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def rotate_half_matrix(d: int) -> torch.Tensor:
    """R with x @ R = rotate_half(x) = cat(-x[d/2:], x[:d/2])."""
    r = torch.zeros(d, d)
    h = d // 2
    for i in range(h):
        r[i + h, i] = -1.0
        r[i, i + h] = 1.0
    return r


class Rope:
    """Device RoPE tables (row-major for per-position gathers, tiled for prefill slices) and the rotate_half matrix."""

    def __init__(self, device, cfg, max_pos: int):
        cos, sin = rope_tables(cfg, max_pos)
        self.max_pos = max_pos
        kw = dict(dtype=ttnn.bfloat16, device=device, memory_config=DRAM)
        self.cos_rm = ttnn.from_torch(cos, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
        self.sin_rm = ttnn.from_torch(sin, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
        self.cos = ttnn.from_torch(cos.reshape(1, 1, max_pos, -1), layout=ttnn.TILE_LAYOUT, **kw)
        self.sin = ttnn.from_torch(sin.reshape(1, 1, max_pos, -1), layout=ttnn.TILE_LAYOUT, **kw)
        self.rot = ttnn.from_torch(rotate_half_matrix(cfg.head_dim).reshape(1, 1, cfg.head_dim, cfg.head_dim),
                                   layout=ttnn.TILE_LAYOUT, **kw)

    def prefill(self, start: int, rows: int):
        end = start + rows
        return (ttnn.slice(self.cos, [0, 0, start, 0], [1, 1, end, self.cos.shape[-1]]),
                ttnn.slice(self.sin, [0, 0, start, 0], [1, 1, end, self.sin.shape[-1]]))

    def decode(self, rope_idx):
        """rope_idx: [1, B] uint32 positions -> cos, sin [1, 1, B, D] tiles."""
        cos = ttnn.embedding(rope_idx, self.cos_rm, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_idx, self.sin_rm, layout=ttnn.TILE_LAYOUT)
        B, D = cos.shape[-2], cos.shape[-1]
        return ttnn.reshape(cos, [1, 1, B, D]), ttnn.reshape(sin, [1, 1, B, D])

    def apply(self, x, cos, sin):
        """x [.., T, D] * cos + rotate_half(x) * sin (cos/sin broadcast over heads)."""
        rx = ttnn.matmul(x, self.rot, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        out = ttnn.mul(x, cos, memory_config=DRAM)
        out = ttnn.addcmul(out, rx, sin, memory_config=DRAM, output_tensor=out)
        ttnn.deallocate(rx)
        return out


def sdpa_compute(fidelity=ttnn.MathFidelity.HiFi2):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fidelity, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=False)


# Prefill attention (chunked SDPA over the paged cache): HiFi4, q chunk 128, k chunk 128 for chunks starting below
# SDPA_LONG_START and 1024 beyond (always a divisor of the chunk start and padded length).  Against FP64 on real
# attention inputs (tests/sdpa_real.py, relative RMS error) HiFi4 beats the previous HiFi2 q/k 64 everywhere
# (start 0-2048: 0.89-1.0e-2 vs 1.20-1.37e-2; 14336-30720: k1024 0.72-0.98e-2 vs 1.21-1.88e-2); at long context the
# 1024 chunks are both the most accurate and the fastest (start 30720, 2048 rows: 18.4 -> 9.2 ms).  Below 16K the eval
# set (<= 4K contexts) prefers k 128 (all dPPL +0.29% vs +0.54% for k 512-1024; KL / top-1 equal), 0.50 ms at start 0.
SDPA_PREFILL_CK = sdpa_compute(ttnn.MathFidelity.HiFi4)
SDPA_LONG_START = 16384


class Attention:
    """Lfm2MoeAttention over a paged KV cache [blocks, n_kv, block, head_dim] (BFP8)."""

    def __init__(self, device, cfg, src, layer: int, cache):
        p = f"model.layers.{layer}.self_attn"
        self.device, self.cfg = device, cfg
        self.nh, self.nkv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wqkv = cache.get(f"L{layer}.attn.qkv", lambda: torch.cat(
            [src.get(f"{p}.{n}_proj.weight") for n in ("q", "k", "v")], dim=0).T, ttnn.bfloat8_b)
        self.wo = cache.get(f"L{layer}.attn.out_proj", lambda: src.get(f"{p}.out_proj.weight").T, ttnn.bfloat8_b)
        self.q_norm = norm_weight(device, src.get(f"{p}.q_layernorm.weight"))
        self.k_norm = norm_weight(device, src.get(f"{p}.k_layernorm.weight"))
        self.eps = cfg.norm_eps
        self.scale = self.hd ** -0.5
        self.k_cache = self.v_cache = None

    def set_kv_cache(self, k_cache, v_cache):
        self.k_cache, self.v_cache = k_cache, v_cache

    def _heads_norm_rope(self, qkv, cos, sin, rope):
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=self.nh, num_kv_heads=self.nkv,
                                                         transpose_k_heads=False, memory_config=DRAM)
        qn = ttnn.rms_norm(q, weight=self.q_norm, epsilon=self.eps, compute_kernel_config=CK_NORM, memory_config=DRAM)
        kn = ttnn.rms_norm(k, weight=self.k_norm, epsilon=self.eps, compute_kernel_config=CK_NORM, memory_config=DRAM)
        free(q, k)
        q, k = rope.apply(qn, cos, sin), rope.apply(kn, cos, sin)
        free(qn, kn)
        return q, k, v

    def prefill(self, x, start: int, rope, page_table, chunk_page_table):
        """x [1, 1, Tp, H] at positions start..start+Tp-1 (Tp % block_size == 0, start % chunk == 0)."""
        Tp = x.shape[-2]
        qkv = linear(x, self.wqkv)
        cos, sin = rope.prefill(start, Tp)
        q, k, v = self._heads_norm_rope(qkv, cos, sin, rope)
        free(qkv, cos, sin)
        for t, c in ((k, self.k_cache), (v, self.v_cache)):
            tc = ttnn.typecast(t, c.dtype) if t.dtype != c.dtype else t
            ttnn.experimental.paged_fill_cache(c, tc, chunk_page_table, batch_idx=0)
            if tc is not t:
                ttnn.deallocate(tc)
        free(k, v)
        span = start | Tp
        k_chunk = min(128 if start < SDPA_LONG_START else 1024, span & -span)
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q, self.k_cache, self.v_cache, page_table, start, scale=self.scale,
            program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                                                  q_chunk_size=min(128, Tp), k_chunk_size=k_chunk,
                                                  exp_approx_mode=False),
            compute_kernel_config=SDPA_PREFILL_CK)
        ttnn.deallocate(q)
        cat = ttnn.experimental.nlp_concat_heads(out, memory_config=DRAM)
        ttnn.deallocate(out)
        y = linear(cat, self.wo)
        ttnn.deallocate(cat)
        return y

    def decode(self, x, cos, sin, rope, cur_pos, page_table):
        """x [1, 1, B, H]; cos/sin [1, 1, B, D]; cur_pos [B] int32; page_table [B, blocks] int32."""
        B = x.shape[-2]
        qkv = linear(x, self.wqkv)
        q, k, v = self._heads_norm_rope(qkv, cos, sin, rope)  # [1, nh|nkv, B, D]
        ttnn.deallocate(qkv)
        grid = self.device.compute_with_storage_grid_size()
        shard = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                  ttnn.ShardSpec(ttnn.num_cores_to_corerangeset(B, grid, row_wise=True),
                                                 [32, self.hd], ttnn.ShardOrientation.ROW_MAJOR))
        for t, c in ((k, self.k_cache), (v, self.v_cache)):
            # BF16 rows: paged_update_cache converts into the BFP8 cache itself (it only accepts BF16/FP32 input).
            tb = ttnn.permute(t, (0, 2, 1, 3), memory_config=DRAM)  # [1, B, nkv, D]
            ts = ttnn.to_memory_config(tb, shard)
            ttnn.experimental.paged_update_cache(c, ts, update_idxs_tensor=cur_pos, page_table=page_table)
            free(tb, ts)
        free(k, v)
        qd = ttnn.permute(q, (0, 2, 1, 3), memory_config=DRAM)  # [1, B, nh, D]
        ttnn.deallocate(q)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qd, self.k_cache, self.v_cache, cur_pos_tensor=cur_pos, page_table_tensor=page_table, is_causal=True,
            scale=self.scale,
            program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                                                  q_chunk_size=32, k_chunk_size=64, exp_approx_mode=False),
            compute_kernel_config=sdpa_compute())
        ttnn.deallocate(qd)
        flat = ttnn.reshape(out, [1, 1, B, self.nh * self.hd])
        y = linear(flat, self.wo)
        free(out)
        return y


class DenseMLP:
    """Lfm2MoeMLP: w2(silu(w1 x) * w3 x), BFP8 weights."""

    def __init__(self, device, cfg, src, layer: int, cache):
        p = f"model.layers.{layer}.feed_forward"
        self.w1 = cache.get(f"L{layer}.mlp.w1", lambda: src.get(f"{p}.w1.weight").T, ttnn.bfloat8_b)
        self.w3 = cache.get(f"L{layer}.mlp.w3", lambda: src.get(f"{p}.w3.weight").T, ttnn.bfloat8_b)
        self.w2 = cache.get(f"L{layer}.mlp.w2", lambda: src.get(f"{p}.w2.weight").T, ttnn.bfloat8_b)

    def __call__(self, x):
        g = linear(x, self.w1)
        u = linear(x, self.w3)
        h = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)
        free(g, u)
        y = linear(h, self.w2)
        ttnn.deallocate(h)
        return y
