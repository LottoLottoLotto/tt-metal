# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM3 operator modules: RMSNorm, gated short conv (7 taps, per-slot conv state), GQA attention (per-head q/k RMSNorm,
per-layer-type RoPE, sigmoid head gate, full or 256-token sliding window) over a paged BFP8 KV cache, dense SwiGLU MLP.

Activations are [1, 1, T, H] BF16 tiles (T = prefill chunk rows, or the decode batch rows B).
Weight device formats come from the source per tensor (source.py: lfmq fmt, or the RTN plan of HFSource).
"""
from __future__ import annotations

import torch
import ttnn

from .common import CK_BFP8, CK_HIFI4, CK_NORM, DTYPES, compute_config, free
from .conv_prefill import ConvPrefill

DRAM = ttnn.DRAM_MEMORY_CONFIG


_PREFILL_PC: dict = {}


def prefill_matmul_config(device, M: int, K: int, N: int):
    """2D-multicast program config for a prefill matmul [M, K] x [K, N] (BF16 x BFP, FP32 accumulation) on the whole
    compute grid; None for decode-sized M (< 128 rows: the default config).  (lfm2 bench: 1.4-1.8x over default.)"""
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


# BFP4 weights in prefill matmuls: LoFi is bitwise HiFi2 there (the 3-bit weight mantissa fits the first fidelity
# phase; HiFi2's second phase multiplies zero bits) and 1.4x faster (logs/prefill/mm_sweep.json).
CK_BFP4_PREFILL = compute_config(ttnn.MathFidelity.LoFi)


def linear(x, w, dtype=ttnn.bfloat16, ck=CK_BFP8, memory_config=DRAM):
    pc = prefill_matmul_config(x.device(), x.shape[-2], x.shape[-1], w.shape[-1])
    if pc is not None and ck is CK_BFP8 and w.dtype == ttnn.bfloat4_b:
        ck = CK_BFP4_PREFILL
    return ttnn.linear(x, w, dtype=dtype, compute_kernel_config=ck, memory_config=memory_config, program_config=pc)


def weight_fmt(src, names) -> str:
    """Device format of a (possibly fused) Linear: every part must share it."""
    fmts = {src.fmt(n) for n in names}
    if len(fmts) != 1:
        raise ValueError(f"mixed formats {fmts} in fused weight {names}")
    return fmts.pop()


def load_linear(cache, src, key: str, names: list[str], pad_to: int = 0):
    """W^T [in, out] of the HF Linear(s) `names` concatenated along out (zero columns appended up to `pad_to`), in the
    source's per-tensor format; tensor cache key `key`."""
    fmt = weight_fmt(src, names)

    def build():
        w = torch.cat([src.get(n) for n in names], dim=0).T
        if pad_to > w.shape[1]:
            w = torch.nn.functional.pad(w, (0, pad_to - w.shape[1]))
        return w

    return cache.get(key, build, DTYPES[fmt])


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


def _taps_fp32(rows, taps):
    """sum_j rows[j] * taps[j] with exact BF16 x BF16 products and FP32 sums; rounded once to BF16 (the BF16 conv1d
    of HF accumulates in FP32 and rounds once)."""
    acc = ttnn.mul(rows[0], taps[0], dtype=ttnn.float32, memory_config=DRAM)
    for r, t in zip(rows[1:], taps[1:]):
        p = ttnn.mul(r, t, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.add(acc, p, dtype=ttnn.float32, memory_config=DRAM, output_tensor=acc)
        ttnn.deallocate(p)
    out = ttnn.typecast(acc, ttnn.bfloat16, memory_config=DRAM)
    ttnn.deallocate(acc)
    return out


class ShortConv:
    """Lfm3ShortConv: BCx = in_proj(x); Bx = B * x; y = C * causal_depthwise_conv_K(Bx) (K = conv_L_cache = 7 taps);
    out_proj(y).

    Conv state per sequence: the last K-1 = 6 Bx rows, s[j] = Bx[t-6+j] for j = 0..5 ([1, 1, B, H] BF16 tiles; row b =
    batch slot b).  Decode updates them in place (trace-safe: s[j] <- s[j+1], s[5] <- Bx[t]); prefill reads the slot's
    rows as the chunk's left context (zeros for a fresh sequence) and writes the chunk's last 6 rows back."""

    def __init__(self, device, cfg, src, layer: int, cache, batch: int = 1):
        p = f"model.layers.{layer}.conv"
        H = cfg.hidden
        self.H, self.K = H, cfg.conv_l_cache
        self.in_proj = load_linear(cache, src, f"L{layer}.conv.in_proj", [f"{p}.in_proj.weight"])
        self.out_proj = load_linear(cache, src, f"L{layer}.conv.out_proj", [f"{p}.out_proj.weight"])
        k = src.get(f"{p}.conv.weight").reshape(H, self.K).to(torch.bfloat16)  # tap j multiplies Bx[t-(K-1)+j]
        tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
        self.taps = [ttnn.from_torch(k[:, j].reshape(1, 1, 1, H).contiguous(), **tile) for j in range(self.K)]
        self.batch = batch
        self.s = [ttnn.from_torch(torch.zeros(1, 1, batch, H, dtype=torch.bfloat16), **tile) for _ in range(self.K - 1)]
        self.zero_row = ttnn.from_torch(torch.zeros(1, 1, 1, H, dtype=torch.bfloat16), **tile)
        self._prefill = None  # conv_prefill.ConvPrefill, built at the first prefill
        # Per-slot row masks [1, 1, B, H] (1 on row b) and their complements for slot writes into the B-row state
        # tiles: state * (1 - m) + row * m is exact (products with 0 / 1 and sums with 0 do not round).
        self.masks, self.keep = [], []
        if batch > 1:
            for b in range(batch):
                m = torch.zeros(1, 1, batch, H, dtype=torch.bfloat16)
                m[:, :, b] = 1
                self.masks.append(ttnn.from_torch(m, **tile))
                self.keep.append(ttnn.from_torch(1 - m, **tile))

    def _write_row(self, row, state, slot: int):
        """state[slot] = row ([1, 1, 1, H] tile), other rows unchanged (bit-exact)."""
        if self.batch == 1:
            ttnn.copy(row, state)
            return
        kept = ttnn.mul(state, self.keep[slot], memory_config=DRAM)
        ttnn.addcmul(kept, row, self.masks[slot], memory_config=DRAM, output_tensor=state)  # row broadcast over B
        ttnn.deallocate(kept)

    def reset(self, slot: int | None = None):
        for s in self.s:
            if slot is None:
                ttnn.mul(s, 0.0, output_tensor=s)
            else:
                self._write_row(self.zero_row, s, slot)

    def state_rows(self, slot: int) -> torch.Tensor:
        """Host copy [K-1, H] of slot's state (Bx[t-6] .. Bx[t-1]; exact BF16)."""
        return torch.stack([ttnn.to_torch(s).reshape(self.batch, self.H)[slot] for s in self.s])

    def restore_slot(self, slot: int, rows: torch.Tensor):
        """Write a state_rows() copy back into `slot` (bit-exact)."""
        for s, r in zip(self.s, rows):
            row = ttnn.from_torch(r.reshape(1, 1, 1, self.H).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=s.device(), memory_config=DRAM)
            self._write_row(row, s, slot)
            ttnn.deallocate(row)

    def remap(self, perm):
        """Row r of the state takes old row idx[r], on the device: perm is the permutation as a [1, 1, B, B] BF16 tile
        (perm[r, idx[r]] = 1).  One matmul per state tensor (each output element is one x * 1 product plus exact
        zeros: bit-exact at HiFi4 with FP32 accumulation; the padding rows of the state tiles are zeros) copied back
        in place."""
        for s in self.s:
            out = ttnn.matmul(perm, s, compute_kernel_config=CK_NORM, dtype=ttnn.bfloat16, memory_config=DRAM)
            ttnn.copy(out, s)
            ttnn.deallocate(out)

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
        conv = _taps_fp32(self.s + [bx], self.taps)
        for j in range(self.K - 2):
            ttnn.copy(self.s[j + 1], self.s[j])
        ttnn.copy(bx, self.s[-1])
        y = ttnn.mul(c, conv, memory_config=DRAM)
        free(bx, c, conv)
        out = linear(y, self.out_proj)
        ttnn.deallocate(y)
        return out

    def prefill(self, x, valid: int, slot: int = 0, fresh: bool = True):
        """x: [1, 1, Tp, H] chunk (rows >= valid are padding) of the sequence in batch slot `slot`.  Left context:
        zeros for a sequence start (fresh), else the slot's state rows; afterwards the slot's state holds
        Bx[valid-6 .. valid-1] of [left (6 rows), Bx].  The middle (B*x, taps, C*, state) is one fused program
        (conv_prefill.ConvPrefill, bitwise the ttnn op chain of decode's arithmetic)."""
        if self._prefill is None:
            self._prefill = ConvPrefill(self)
        bcx = linear(x, self.in_proj)
        y = self._prefill(bcx, valid, slot=slot, fresh=fresh)
        ttnn.deallocate(bcx)
        out = linear(y, self.out_proj)
        ttnn.deallocate(y)
        return out


def rope_tables(cfg, layer_type: str, max_pos: int):
    """HF Lfm3RotaryEmbedding (default rope, zero-tail partial rotary): cos/sin [max_pos, head_dim] in FP32, rounded
    to BF16 like HF's x.dtype cast.  Partial rotary keeps the first partial * head_dim / 2 frequencies of the full-dim
    progression and zeroes the rest (cos 1, sin 0 there)."""
    d = cfg.head_dim
    p = cfg.rope[layer_type]
    angles = int(p["partial_rotary_factor"] * d) // 2
    inv_freq = 1.0 / (p["rope_theta"] ** (torch.arange(0, 2 * angles, 2, dtype=torch.float) / d))
    inv_freq = torch.cat((inv_freq, torch.zeros(d // 2 - angles, dtype=torch.float)))
    freqs = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv_freq[None, :]
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
    """Device RoPE tables of one attention layer type (row-major for per-position decode gathers, tiled for the
    prefill front's per-row reads) and the rotate_half matrix."""

    def __init__(self, device, cfg, layer_type: str, max_pos: int):
        cos, sin = rope_tables(cfg, layer_type, max_pos)
        self.max_pos = max_pos
        kw = dict(dtype=ttnn.bfloat16, device=device, memory_config=DRAM)
        self.cos_rm = ttnn.from_torch(cos, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
        self.sin_rm = ttnn.from_torch(sin, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
        self.cos = ttnn.from_torch(cos.reshape(1, 1, max_pos, -1), layout=ttnn.TILE_LAYOUT, **kw)
        self.sin = ttnn.from_torch(sin.reshape(1, 1, max_pos, -1), layout=ttnn.TILE_LAYOUT, **kw)
        self.rot = ttnn.from_torch(rotate_half_matrix(cfg.head_dim).reshape(1, 1, cfg.head_dim, cfg.head_dim),
                                   layout=ttnn.TILE_LAYOUT, **kw)

    def decode(self, rope_idx):
        """rope_idx: [1, B] uint32 positions -> cos, sin [1, 1, B, D] tiles."""
        cos = ttnn.embedding(rope_idx, self.cos_rm, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_idx, self.sin_rm, layout=ttnn.TILE_LAYOUT)
        B, D = cos.shape[-2], cos.shape[-1]
        return ttnn.reshape(cos, [1, 1, B, D]), ttnn.reshape(sin, [1, 1, B, D])

    def apply(self, x, cos, sin):
        """x [.., T, D] * cos + rotate_half(x) * sin (cos/sin broadcast over heads).  rotate_half(x) = x @ R is exact
        (one +-1 product per element, FP32 accumulation); ttnn.experimental.rotate_half is bitwise the same but 7x
        slower (logs/prefill/window_rope.json)."""
        rx = ttnn.matmul(x, self.rot, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        out = ttnn.mul(x, cos, memory_config=DRAM)
        out = ttnn.addcmul(out, rx, sin, memory_config=DRAM, output_tensor=out)
        ttnn.deallocate(rx)
        return out


def sdpa_compute(fidelity=ttnn.MathFidelity.HiFi2):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fidelity, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=False)


GATE_PAD = 32  # gate_proj's 16 output columns padded to a tile in the fused q|k|v|gate weight


class Attention:
    """Lfm3Attention over a paged KV cache [blocks, n_kv, block, head_dim] (BFP8).

    qkv(+gate) = x @ [Wq | Wk | Wv | Wg | 0] (one matmul when the four share a format) -> heads -> q_norm/k_norm ->
    RoPE (the layer type's tables) -> attention (full causal, or causal within `sliding_window` keys incl. self) ->
    concat heads * sigmoid(gate) per head -> o_proj."""

    def __init__(self, device, cfg, src, layer: int, cache):
        p = f"model.layers.{layer}.self_attn"
        self.device, self.cfg, self.layer = device, cfg, layer
        self.nh, self.nkv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.window = cfg.sliding_window if cfg.is_sliding(layer) else None
        qkv = [f"{p}.{n}_proj.weight" for n in ("q", "k", "v")]
        gate = f"{p}.gate_proj.weight"
        self.qkv_width = (self.nh + 2 * self.nkv) * self.hd
        self.fused_gate = len({src.fmt(n) for n in qkv + [gate]}) == 1
        if self.fused_gate:
            self.wqkv = load_linear(cache, src, f"L{layer}.attn.qkvg", qkv + [gate], pad_to=self.qkv_width + GATE_PAD)
            self.wg = None
        else:
            self.wqkv = load_linear(cache, src, f"L{layer}.attn.qkv", qkv)
            self.wg = load_linear(cache, src, f"L{layer}.attn.gate", [gate], pad_to=GATE_PAD)
        self.wo = load_linear(cache, src, f"L{layer}.attn.o_proj", [f"{p}.o_proj.weight"])
        self.q_norm = norm_weight(device, src.get(f"{p}.q_norm.weight"))
        self.k_norm = norm_weight(device, src.get(f"{p}.k_norm.weight"))
        # Gate expansion E [32, nh * hd]: E[h, h*hd:(h+1)*hd] = 1 (rows >= nh zero), so sigmoid(g) @ E repeats each
        # head's gate over its hd output columns exactly (one nonzero product per column, FP32 accumulation).
        e = torch.zeros(GATE_PAD, self.nh * self.hd)
        for h in range(self.nh):
            e[h, h * self.hd:(h + 1) * self.hd] = 1.0
        self.gate_expand = ttnn.from_torch(e.reshape(1, 1, GATE_PAD, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                           device=device, memory_config=DRAM)
        self.eps = cfg.norm_eps
        self.scale = self.hd ** -0.5
        self.k_cache = self.v_cache = None
        self._sdpa = None  # prefill/sdpa.PrefillSDPA (built at the first prefill)
        self._fronts = {}  # prefill/attn_front.AttnFrontPrefill per Rope (built at the first prefill)

    def _front(self, rope):
        f = self._fronts.get(id(rope))
        if f is None:
            from .prefill.attn_front import AttnFrontPrefill

            f = self._fronts[id(rope)] = AttnFrontPrefill(self, rope)
        return f

    def set_kv_cache(self, k_cache, v_cache):
        self.k_cache, self.v_cache = k_cache, v_cache

    def _project(self, x):
        """-> (qkv [1, 1, T, qkv_width], gate logits [1, 1, T, 32])."""
        y = linear(x, self.wqkv)
        if not self.fused_gate:
            return y, linear(x, self.wg)
        T = y.shape[-2]
        qkv = ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, self.qkv_width])
        g = ttnn.slice(y, [0, 0, 0, self.qkv_width], [1, 1, T, self.qkv_width + GATE_PAD])
        ttnn.deallocate(y)
        return qkv, g

    def _gate(self, cat, g):
        """cat [1, 1, T, nh*hd] * sigmoid(g) repeated per head (HF: BF16 sigmoid of the BF16 gate_proj output)."""
        sg = ttnn.sigmoid(g, memory_config=DRAM)
        ge = ttnn.matmul(sg, self.gate_expand, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        out = ttnn.mul(cat, ge, memory_config=DRAM)
        free(sg, ge)
        return out

    def _heads_norm_rope(self, qkv, cos, sin, rope):
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=self.nh, num_kv_heads=self.nkv,
                                                         transpose_k_heads=False, memory_config=DRAM)
        qn = ttnn.rms_norm(q, weight=self.q_norm, epsilon=self.eps, compute_kernel_config=CK_NORM, memory_config=DRAM)
        kn = ttnn.rms_norm(k, weight=self.k_norm, epsilon=self.eps, compute_kernel_config=CK_NORM, memory_config=DRAM)
        free(q, k)
        q, k = rope.apply(qn, cos, sin), rope.apply(kn, cos, sin)
        free(qn, kn)
        return q, k, v

    def prefill(self, x, rope, page_table, chunk_page_table, start_tensor):
        """x [1, 1, Tp, H] at positions start..start+Tp-1 (Tp % 64 == 0; start in the [1] int32 device tensor
        start_tensor, a multiple of 64): qkvg matmul -> AttnFrontPrefill (q heads, q/k norms, RoPE, the chunk's K/V
        into the paged cache) -> PrefillSDPA (causal / sliding GQA attention over the paged cache, sigmoid gate,
        concat-heads layout) -> o_proj.  Programs depend only on Tp (and the page-table width)."""
        y = linear(x, self.wqkv)
        q = self._front(rope)(y, start_tensor, chunk_page_table)
        g = y if self.fused_gate else linear(x, self.wg)
        if self._sdpa is None:
            from .prefill.sdpa import PrefillSDPA

            self._sdpa = PrefillSDPA.for_attention(self)
        out = self._sdpa(q, self.k_cache, self.v_cache, page_table, start_tensor, gate=g)
        free(q, y)
        if g is not y:
            ttnn.deallocate(g)
        o = linear(out, self.wo)
        ttnn.deallocate(out)
        return o

    def decode(self, x, cos, sin, rope, cur_pos, page_table):
        """x [1, 1, B, H]; cos/sin [1, 1, B, D] (this layer type's); cur_pos [B] int32; page_table [B, blocks] int32."""
        B = x.shape[-2]
        qkv, g = self._project(x)
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
            scale=self.scale, sliding_window_size=self.window,
            program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                                                  q_chunk_size=32, k_chunk_size=64, exp_approx_mode=False),
            compute_kernel_config=sdpa_compute())
        ttnn.deallocate(qd)
        flat = ttnn.reshape(out, [1, 1, B, self.nh * self.hd])
        gated = self._gate(flat, g)
        free(out, g)
        y = linear(gated, self.wo)
        ttnn.deallocate(gated)
        return y


class DenseMLP:
    """Lfm3MLP: w2(silu(w1 x) * w3 x).  Decode: three stock matmuls (__call__); prefill: prefill/mlp.PrefillMLP (w1|w3
    + SwiGLU in one program over a panel-layout copy of w1 / w3, then w2)."""

    def __init__(self, device, cfg, src, layer: int, cache):
        from .prefill.mlp import PrefillMLP

        p = f"model.layers.{layer}.feed_forward"
        self.w1 = load_linear(cache, src, f"L{layer}.mlp.w1", [f"{p}.w1.weight"])
        self.w3 = load_linear(cache, src, f"L{layer}.mlp.w3", [f"{p}.w3.weight"])
        self.w2 = load_linear(cache, src, f"L{layer}.mlp.w2", [f"{p}.w2.weight"])
        self._prefill = PrefillMLP(self, cache=cache, key=f"L{layer}.mlp.w13p")

    def prefill(self, x):
        """x [1, 1, M, H] (M a multiple of 32) -> [1, 1, M, H]."""
        return self._prefill(x)

    def __call__(self, x):
        g = linear(x, self.w1)
        u = linear(x, self.w3)
        h = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)
        free(g, u)
        y = linear(h, self.w2)
        ttnn.deallocate(h)
        return y
