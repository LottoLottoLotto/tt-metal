# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused Inkling2 attention decode / multi-row verify (owner AttnFast): 3 launches per layer application instead of
~55-75 ttnn ops.

    fast = AttnDecodeFast(attn, weight_cache)        # attn: tt/attention.py Inkling2Attention (shares its caches,
                                                     # conv ring and weights; built once, before trace capture)
    y = fast(xq, xk, xv, tables, pass_idx)           # == attn.decode(xq, xk, xv, tables.cur_pos, pass_idx)
    y = fast.verify(xq, xk, xv, vtables, R, pass_idx)  # R = 1..8 rows at positions p .. p+R-1 (vtables:
                                                     # attention.AttnVerifyTables), == R decode steps
    y = fast.batch(xq, xk, xv, bt, R, pass_idx)      # batched serving (module with seqs / kv_blocks): R = 1..16
                                                     # rows, row b = its own sequence (bt: the batch tables:
                                                     # cur_pos / tau / slot / page_table), == that sequence's decode

1. projection (one generic_op, two worker groups): [q*s | gate | k | v] = xq|xq|xk|xv @ BFP8 W (StreamDecodeMatmul,
   32 workers; segments pick each worker's input) and the BF16 small projection (r / sink per group, full layers'
   dynamic k / v taps) on 16-24 more workers, rows 0..R-1.
2. AttnFront (kernels/fast/attn_front.cpp, 8 cores = kv heads, fused into launch 1): per row q / r clip [* tau],
   K / V short conv over the position-indexed conv ring + the earlier rows, k L2 clip, v RMSNorm, K / V cache row
   writes, conv-ring row writes, the SDPA query tiles (q | r | sink) of every row.
3. SDPADecodeInk (kernels/fast/attn_sdpa.cpp, on the cores left free: 48 sliding / 40 full): split-K decode attention
   over the ring / full cache with the relative bias in-kernel (augmented q.k), the sink as an extra partition,
   * sigmoid(gate); R rows, each at its own position (causal across rows: row j sees keys <= p + j).
4. o_proj: StreamDecodeMatmul (BFP8), output rows >= R zero.
Only tables.cur_pos (int32 word j = position of row j) and tables.tau (fp32 tile, element (0, j)) are read on device;
the cost of a full layer scales with the position (no full_len bucket).  All state is position-indexed (ring slot
pos % 1088, full row pos, conv ring row pos % 32): a verify writes only its rows' positions, so commit = advance.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import os

import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from ..common import CK_HIFI2, CK_HIFI4
from .common import DRAM, KERNELS, TILE_BYTES, CachedProgram, cb_descriptors, f32_bits, grid_cores, row_ranges, \
    unified_kernels
from .decode_matmul import Preset, StreamDecodeMatmul, cached_stream_weight

RING = 1024          # sliding window
RING_SLOTS = 1088    # ring modulus (attention.RING_SLOTS)
MAX_ROWS = 8         # verify rows
BATCH_ROWS = 16      # batch rows (batched serving: one row per sequence)
_FMT = {ttnn.bfloat16: 0, ttnn.bfloat8_b: 1, ttnn.bfloat4_b: 2}  # cache format codes of the kernels
ROWS_SEM = 6         # batch: semaphore of each front core = its rows done (attn_front.cpp; polled by attn_sdpa.cpp)
_ROUTE_CBS = (26, 27, 28, 29)  # front input CBs written by the projection workers: Qin, Rin, X, cIn
TBL_ROWS = 3136  # bias table rows per copy (zero | proj reversed | zero (ring hole) | proj reversed | zero tail)
# batch programs with >= XF_ROWS rows gather their input rows before streaming the projection / o_proj weights (both
# data-movement RISCs; decode_matmul.py x_first): measured faster from 8 rows (NOTES-attn.md "Batched attention, round 2")
XF_ROWS = int(os.environ.get("INK_XF_ROWS", "8"))

PRESETS = {
    "big": Preset(4, 1, 24),      # 2304 x 8192 BFP8: 32 workers x 8 columns (BIG_PERM), one column per subblock
    "qg": Preset(2, 1, 36),       # mixed formats: 2304 x 6144 q|gate (BFP4): 16 workers x 12 columns (QG_PERM)
    "kv": Preset(2, 1, 24),       # mixed formats: 2304 x 2048 k|v (BFP8): 16 workers x 4 columns
    "small": Preset(2, 1, 24),    # 2304 x 512 BF16 (sliding)
    "small_dyn": Preset(3, 1, 24),  # 2304 x 768 BF16 (full: + dynamic taps)
    "o": Preset(3, 3, 8),         # 3072 x 2304 BFP8: 24 workers x 3 columns
}


def _big_perm() -> list[int]:
    """Physical -> logical tile column of the [q (0..95) | gate (96..191) | k (192..223) | v (224..255)] projection:
    worker u (= 4 bank + core, physical columns 8u .. 8u+7, streamed in this order) gets 5 of the 160 q / k / v columns
    (consumed by the front) then 3 of the 96 gate columns (read by the SDPA reducers only): every DRAM bank serves 20 q /
    k / v + 12 gate columns and every worker finishes its routed columns first."""
    crit, gate = list(range(96)) + list(range(192, 256)), list(range(96, 192))
    return [c for u in range(32) for c in crit[5 * u:5 * u + 5] + gate[3 * u:3 * u + 3]]


BIG_PERM = _big_perm()  # attn_sdpa.cpp: gate column 96 + m at physical 8 (m / 3) + 5 + m % 3
# mixed formats, q|gate group: worker u (= 2 bank + core, physical columns 12u .. 12u+11) streams q columns 6u .. 6u+5
# (routed to the front) then gate columns 96 + 6u .. 96 + 6u + 5 (read by the SDPA reducers): gate column 96 + m at
# physical 12 (m / 6) + 6 + m % 6.  The k|v group keeps the logical order (k 192.., v 224..; worker = half of one bank's
# k or v columns).  Measured (NOTES-attn.md Round 6): 16 + 16 workers leave the SDPA its 40 full-layer cores (3 q|gate
# workers per bank: 32 cores, +11 % full-layer latency at 1M) at the same short-context latency.
QG_PERM = [c for u in range(16) for c in list(range(6 * u, 6 * u + 6)) + list(range(96 + 6 * u, 102 + 6 * u))]


def _torch(t) -> torch.Tensor:
    t = ttnn.to_torch(t)
    return t.reshape(t.shape[-2], t.shape[-1])


_SHARED: dict = {}


def _shared(device, key, build):
    """Device intermediates shared by every layer's AttnDecodeFast (programs run sequentially)."""
    k = (id(device), key)
    if k not in _SHARED:
        _SHARED[k] = build()
    return _SHARED[k]


def sdpa_consts(device):
    """[32, 32 * 255 * 8] BF16 TILE DRAM: (zeros), ones, A (A[r, k] = 1 for r < 4, k % 4 == r), then for v = 1..63
    live keys of the last block the prefix-mask pair (key tile t, column j: 0 if 32 t + j < v else -1e9), then for
    o = 1..63 dead keys at the start of the first block (ring wrap) the suffix-mask pair (0 if 32 t + j >= o else
    -1e9); every page replicated 8x (page p, replica r at p * 8 + r: the readers spread over the DRAM banks)."""
    def build():
        tiles = [torch.zeros(32, 32), torch.ones(32, 32), torch.zeros(32, 32)]
        for r in range(4):
            tiles[2][r, r::4] = 1.0
        j = torch.arange(64)
        for v in range(1, 64):
            row = torch.where(j < v, 0.0, -1e9)
            tiles += [row[:32].expand(32, 32), row[32:].expand(32, 32)]
        for o in range(1, 64):
            row = torch.where(j >= o, 0.0, -1e9)
            tiles += [row[:32].expand(32, 32), row[32:].expand(32, 32)]
        host = torch.cat([t for t in tiles for _ in range(8)], dim=1).to(torch.bfloat16)
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    return _shared(device, "sdpa_consts2", build)


def bias_table(proj: torch.Tensor) -> torch.Tensor:
    """proj [16, 1024] -> [16, TBL_ROWS * 16] BF16 (2 shift copies x 8 bank replicas): copy 0 row i = proj[:, 2047 - i]
    (1024 <= i < 2048), 0 (2048 <= i < 2112: distances 1024..1087 of the ring wrap), proj[:, 3135 - i]
    (2112 <= i < 3136), else 0; copy 1 = copy 0 shifted up one row (16-row runs start 64 B aligned in one copy).
    Key j of a step at bias position b (ring: slot pos % 1088, full: pos) uses row 2047 - b + j."""
    a = torch.zeros(TBL_ROWS + 1, 16)
    rev = proj.float().flip(1).T  # row k = proj[:, 1023 - k]
    a[1024:2048] = rev
    a[2112:3136] = rev
    copies = [a[:TBL_ROWS].reshape(-1), a[1:].reshape(-1)]
    # page = shift * 8 + r: every shift copy replicated on the 8 DRAM banks (104 SDPA cores read it)
    return torch.stack([copies[sh] for sh in range(2) for _ in range(8)]).to(torch.bfloat16)


class AttnDecodeFast:
    """See the module docstring.  kv dtype (BF16 / BFP8) follows the attention module's caches."""

    def __init__(self, attn, weight_cache, *, chunk_blocks: int = 16, v_blocks: int = 8, outstanding: int = 3,
                 fidelity=ttnn.MathFidelity.HiFi4, noc1_rows=(0, 1, 2, 4), k0: int = 2, mix: bool | None = None):
        self.attn, self.device, self.cfg = attn, attn.device, attn.cfg
        d, L, wc = attn.device, attn.layer, weight_cache
        self.full, self.fused = attn.full, attn.fused
        self.k_dtype, self.v_dtype = attn.k_cache[0].dtype, attn.v_cache[0].dtype
        if self.k_dtype not in _FMT or self.v_dtype not in _FMT:
            raise ValueError("AttnDecodeFast: BF16 / BFP8 / BFP4 caches")
        key = f"attn.L{L}.fast"
        # ---- 1. projections
        split = getattr(attn, "w_k", None) is not None  # separate k / v tensors (unfused layers, or fused layers whose
        # q|gate and k / v precision families differ: Prec6); the stream column order is [q | gate | k | v] either way
        # mix (default env INK_ATTN_MIX, on): q|gate and k|v in different formats stream in their own formats as two
        # worker groups (QG_PERM / PRESETS "qg" + "kv"); off: one group in the wider format (BFP4 values are exact in
        # BFP8: the same products, bitwise the same outputs)
        if mix is None:
            mix = os.environ.get("INK_ATTN_MIX", "1") != "0"
        self.mixed = bool(mix and split and attn.w_big.dtype != attn.w_k.dtype and attn.w_k.dtype == attn.w_v.dtype)

        def big():
            w = torch.cat([_torch(attn.w_big), _torch(attn.w_k), _torch(attn.w_v)], dim=1) if split else _torch(attn.w_big)
            return w.reshape(w.shape[0], -1, 32)[:, BIG_PERM].reshape(w.shape[0], -1)

        def qg():
            w = _torch(attn.w_big)
            return w.reshape(w.shape[0], -1, 32)[:, QG_PERM].reshape(w.shape[0], -1)

        def small():
            rs = _torch(attn.w_rs).float()  # [2304, 768]: head h cols h*32 + 0..15 = r, h*32 + 17 = sink
            n = 24 if self.full else 16
            w = torch.zeros(rs.shape[0], n * 32)
            for g in range(8):
                for h in range(3):
                    hh = 3 * g + h
                    c0 = (2 * g) * 32 + h * 16 if h < 2 else (2 * g + 1) * 32
                    w[:, c0:c0 + 16] = rs[:, hh * 32:hh * 32 + 16]
                    w[:, (2 * g + 1) * 32 + 16 + h] = rs[:, hh * 32 + 17]
            if self.full:
                w[:, 16 * 32:17 * 32] = _torch(attn.w_kd).float()
                w[:, 17 * 32:18 * 32] = _torch(attn.w_vd).float()
            return w

        ps = PRESETS["small_dyn" if self.full else "small"]
        # projection / o weight formats follow the attention module's (one group: BFP4 only if every source part is
        # BFP4); attn.wtag names the checkpoint variant (precision study) so differently quantized weights never share
        # a key
        wtag = getattr(attn, "wtag", "")
        gx = d.compute_with_storage_grid_size().x
        if self.mixed:  # q|gate (xq) on PRESETS["qg"] workers, k|v (xk | xv) on the next PRESETS["kv"] workers
            pq, pk = PRESETS["qg"], PRESETS["kv"]
            w_qg = cached_stream_weight(wc, f"{key}.qg.perm6{wtag}", pq, qg, attn.w_big.dtype)
            w_kv = cached_stream_weight(wc, f"{key}.kv{wtag}", pk,
                                        lambda: torch.cat([_torch(attn.w_k), _torch(attn.w_v)], dim=1), attn.w_k.dtype)
            self.mm_big = StreamDecodeMatmul(d, w_qg, pq, CK_HIFI2)
            self.mm_kv = StreamDecodeMatmul(d, w_kv, pk, CK_HIFI2, segments=(0, 32),
                                            first_core=len(self.mm_big.core_args))
            self.big_groups = [(self.mm_big, QG_PERM), (self.mm_kv, None)]  # (matmul, physical -> logical column)
            self.gate_cols = (12, 6, 6)  # attn_sdpa.cpp gate_stride / gate_div / gate_off of QG_PERM
        else:
            pb = PRESETS["big"]
            parts = [attn.w_big, attn.w_k, attn.w_v] if split else [attn.w_big]
            big_dt = ttnn.bfloat4_b if all(t.dtype == ttnn.bfloat4_b for t in parts) else ttnn.bfloat8_b
            w_big = cached_stream_weight(wc, f"{key}.big.perm{wtag}", pb, big, big_dt)
            col_seg = None if self.fused else [0 if c < 192 else (1 if c < 224 else 2) for c in BIG_PERM]
            self.mm_big = StreamDecodeMatmul(d, w_big, pb, CK_HIFI2, col_seg=col_seg)
            self.mm_kv = None
            self.big_groups = [(self.mm_big, BIG_PERM)]
            self.gate_cols = (8, 3, 5)  # BIG_PERM
        n_big = sum(len(mm.core_args) for mm, _ in self.big_groups)
        o_dt = ttnn.bfloat4_b if attn.w_o.dtype == ttnn.bfloat4_b else ttnn.bfloat8_b
        w_small = cached_stream_weight(wc, f"{key}.small", ps, small, ttnn.bfloat16)
        self.mm_small = StreamDecodeMatmul(d, w_small, ps, CK_HIFI4, n_out=(18 if self.full else 16) * 32,
                                           segments=(0, 16, 17) if self.full else (0,),
                                           first_core=-(-n_big // gx) * gx)
        w_o = cached_stream_weight(wc, f"{key}.o{wtag}", PRESETS["o"], lambda: _torch(attn.w_o), o_dt)
        self.mm_o = StreamDecodeMatmul(d, w_o, PRESETS["o"], CK_HIFI2, pad_zero=True, x_rm=True)
        # verify: the same weight on tile input rows 0..R-1 (att_rows), output rows >= R zero
        self.mm_o_rows = StreamDecodeMatmul(d, w_o, PRESETS["o"], CK_HIFI2, pad_zero=True)
        # ---- 2. front constants: fp32 taps tiles (g * 4 + lag; rows 0-3 k chunks, 4-7 v chunks)
        taps = [_torch(t).float().reshape(-1) for t in attn.taps]  # lag-indexed [2048]
        tt = torch.zeros(32, 32 * 32)
        for g in range(8):
            for lag in range(4):
                m = torch.zeros(32, 32)
                for c in range(4):
                    m[c] = taps[lag][128 * g + 32 * c:128 * g + 32 * c + 32]
                    m[4 + c] = taps[lag][1024 + 128 * g + 32 * c:1024 + 128 * g + 32 * c + 32]
                tt[:, (g * 4 + lag) * 32:(g * 4 + lag + 1) * 32] = m
        self.taps = ttnn.from_torch(tt.reshape(1, 1, 32, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d,
                                    memory_config=DRAM)
        proj = _torch(attn.p_aug).float()[:16, :RING]
        self.table = ttnn.from_torch(bias_table(proj), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=d,
                                     memory_config=DRAM)
        cfg = self.cfg
        post = attn.hd ** -0.5
        self.front_defines = [
            ("DYNAMIC", str(int(self.full))), ("RING", str(int(not self.full))),
            ("K_FMT", str(_FMT[self.k_dtype])), ("V_FMT", str(_FMT[self.v_dtype])),
            ("CONV_SCALE", hex(f32_bits(getattr(attn, "conv_scale", 1.0)))), ("EPS_BITS", hex(f32_bits(cfg.eps))),
            ("QK_CLIP_BITS", hex(f32_bits(cfg.qk_l2_clip))), ("Q_MUL_BITS", hex(f32_bits(cfg.qk_l2_clip * post))),
            ("Q_CAP_BITS", hex(f32_bits(post))), ("R_CLIP_BITS", hex(f32_bits(cfg.rel_l2_clip))),
        ]
        # ---- shared intermediates (rows 0..R-1 meaningful)
        z = lambda name, cols: _shared(d, name, lambda: ttnn.zeros(  # noqa: E731
            [1, 1, 32, cols], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=DRAM))
        self.P = z("P", 8192)
        self.S = z(f"S{24 if self.full else 16}", (24 if self.full else 16) * 32)
        self.q = z("q8", MAX_ROWS * 40 * 32)  # SDPA query tiles: row r, group g at tiles (8 r + g) * 5 + 0..4
        # batch (batched serving, attention module with slots / a pool): query tiles of up to BATCH_ROWS rows
        self.q_batch = z("q16", BATCH_ROWS * 40 * 32) if getattr(attn, "batched", False) else None
        # the gated attention row (SDPA output, o_proj input; row-major: one DRAM page)
        self.att = _shared(d, "att", lambda: ttnn.zeros([1, 1, 1, 3072], dtype=ttnn.bfloat16,
                                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=d, memory_config=DRAM))
        # verify (R > 1): the gated attention rows 0..R-1 as tiles (o_proj's tile input)
        self.att_rows = z("att_rows", 3072)
        self.consts = sdpa_consts(d)
        # ---- one program per application: projection workers + front cores + SDPA cores (every other core)
        self.outstanding, self.k0, self.chunk_blocks, self.v_cap = outstanding, k0, chunk_blocks, v_blocks
        self.fidelity, self.noc1_rows = fidelity, set(noc1_rows)
        # key blocks a row can span: positions < max_seq_len (single stream: the cache's blocks; batched: the page
        # table's columns, the pool may hold more or fewer blocks)
        self.max_blocks = attn.max_seq_len // 64 if self.full else RING_SLOTS // 64
        gx = d.compute_with_storage_grid_size().x
        gy = d.compute_with_storage_grid_size().y
        small_first = -(-n_big // gx) * gx
        front_first = -(-(small_first + 8 * self.mm_small.preset.cores_per_bank) // gx) * gx
        self.front_cores = grid_cores(d, 8, front_first)
        used = {(c.x, c.y) for mm, _ in self.big_groups for c, _ in mm.core_args}
        used |= {(c.x, c.y) for c, _ in self.mm_small.core_args}
        used |= {(c.x, c.y) for c in self.front_cores}
        free = [(i % gx, i // gx) for i in range(gx * gy) if (i % gx, i // gx) not in used]
        self.sdpa_cores = free[:8 * (len(free) // 8)]  # 8 W cores (W = SDPA workers per kv group)
        self._progs = {}
        # L1 weight prefetch (AttnPrefetch.attach): owner, {R: {pass: {"big" | "kv" | "small" | "o": per-worker
        # blocks}}}, deeper-ring copies of the stream matmuls (same weights / cores; "o_rows": verify's o_proj)
        self.pf, self.pf_plan, self.pf_mm = None, None, None

    def stream_mats(self) -> dict:
        """The stream matmuls by prefetch name (AttnPrefetch): "big" (q|gate|k|v, or q|gate when mixed), "kv" (mixed),
        "small", "o"."""
        mats = {"big": self.mm_big}
        if self.mm_kv is not None:
            mats["kv"] = self.mm_kv
        return dict(mats, small=self.mm_small, o=self.mm_o)

    def _geom(self, W):
        """(chunk blocks C, partitions pmax, V CB blocks) for W SDPA workers per kv group."""
        C = max(min(-(-self.max_blocks // W), self.chunk_blocks), self.outstanding + 1)
        pmax = W * -(-self.max_blocks // (W * C)) if self.max_blocks >= W else self.max_blocks
        return C, pmax, max(min(self.v_cap, C), self.outstanding + 1)

    # ------------------------------------------------------------------ programs
    def _routes(self, front_noc):
        """Projection worker routes into the front cores (decode_matmul_stream.cpp route mode): ([{worker: (dram mask,
        routes)} per big group], same small, routes expected per front core).  Targets: 0 Qin, 1 Rin, 2 X, 3 cIn
        (_ROUTE_CBS); the offsets are token 0's (token r: + r * 2048, decode_matmul_stream.cpp route_stride).  Big
        groups: physical column -> logical by the group's permutation (BIG_PERM; mixed: QG_PERM, k|v 192 + column)."""
        def rf(row, f):
            return ((row >> 4) * 2 + f) * 512 + (row & 15) * 32

        expected = [0] * 8

        def add(rl, j, face, zero, target, g, off):
            x, y = front_noc[g]
            rl.append((j, face, zero, target, x, y, off))
            expected[g] += 1

        big, small = [], {}
        for mm, perm in self.big_groups:  # perm None: logical column 192 + physical (the k|v group)
            routes = {}
            for i, (_, (_, _, nv, first, _)) in enumerate(mm.core_args):
                mask, rl = 0, []
                for j, col in enumerate(perm[c] if perm is not None else 192 + c for c in range(first, first + nv)):
                    if col < 96:  # q: head col // 4, chunk col % 4 -> Qin row 4 h + c of group head // 3
                        head, c = divmod(col, 4)
                        g, h = divmod(head, 3)
                        for f in range(2):
                            add(rl, j, f, 0, 0, g, rf(4 * h + c, f))
                    elif col < 192:  # gate: DRAM (the SDPA reducers read it)
                        mask |= 1 << j
                    else:  # k (192..223) | v (224..255) of group g, chunk c -> X tile 3 (token 0) row c | 4 + c
                        g, c = divmod((col - 192) % 32, 4)
                        row = c if col < 224 else 4 + c
                        for f in range(2):
                            add(rl, j, f, 0, 2, g, 3 * 2048 + rf(row, f))
                routes[i] = (mask, rl)
            big.append(routes)
        for i, (_, (_, _, nv, first, _)) in enumerate(self.mm_small.core_args):
            rl = []
            for j, col in enumerate(range(first, first + nv)):
                if col < 16:
                    g, odd = divmod(col, 2)
                    if not odd:  # r of heads 3g, 3g + 1 -> Rin rows 0, 4 (face 0); zero their cols 16..31
                        add(rl, j, 0, 0, 1, g, rf(0, 0))
                        add(rl, j, 1, 0, 1, g, rf(4, 0))
                        add(rl, j, 0, 1, 1, g, rf(0, 1))
                        add(rl, j, 0, 1, 1, g, rf(4, 1))
                    else:  # r of head 3g + 2 -> Rin row 8; sinks (cols 16..18) -> cIn [0, 32)
                        add(rl, j, 0, 0, 1, g, rf(8, 0))
                        add(rl, j, 0, 1, 1, g, rf(8, 1))
                        add(rl, j, 1, 0, 3, g, 0)
                elif col in (16, 17):  # dynamic k / v taps (col 4 g + lag): face g >= 4 -> cIn [64, 96) / [128, 160)
                    for g in range(8):
                        add(rl, j, int(g >= 4), 0, 3, g, 64 if col == 16 else 128)
            small[i] = (0, rl)
        return big, small, expected

    def _pt_page(self, bt) -> int:
        """Aligned page bytes of the batch page table (int32 [32, W] row-major: one page per row)."""
        return -(-int(bt.page_table.shape[-1]) * 4 // 64) * 64

    def _sdpa_parts(self, slot, tables, R, cores, ready, batch: bool = False):
        """SDPA kernels / CBs / semaphores (kernels/fast/attn_sdpa.cpp, fused mode) on `cores` (8 W logical (x, y)):
        gather semaphores 1 .. slots, front-ready semaphores ready .. ready + 7.  batch: tables = the batch tables
        (row b = its own sequence: slot / page-table rows, attn_sdpa.cpp BATCH)."""
        a, dev = self.attn, self.device
        nc = len(cores)
        W = nc // 8
        C, pmax_all, v_blocks = self._geom(W)
        slots = -(-8 * R // nc)
        gates = 1 + slots  # gate rows in P (front 0 relays the big workers' done count)
        if ready < gates + 1 or ready + 8 > 16:
            raise ValueError("SDPADecodeInk: semaphores")
        # 2-level reduction: each core merges its partitions of a group (compute) and sends one partial per segment;
        # a group's weight range spans at most nc / 8 + 2 cores -> the reducer gathers <= that many partials (+ sink)
        pmax = min(pmax_all, nc // 8 + 2)
        nkt = (4 * (pmax + 1) + 31) // 32
        k0 = self.k0
        # partitions per core: every row is split over the cores as a one-row decode (attn_sdpa.cpp vcore)
        max_parts = R * ((8 * (self.max_blocks + k0 * pmax_all) // nc + 1) // (1 + k0) + 2)
        noc = [dev.worker_core_from_logical_core(ttnn.CoreCoord(*c)) for c in cores]
        rt_args = [(ttnn.CoreCoord(*core), [c, int(core[1] in self.noc1_rows)]) for c, core in enumerate(cores)]
        core_set = row_ranges([ttnn.CoreCoord(*c) for c in cores])
        r64 = lambda n: (n + 63) // 64 * 64  # noqa: E731

        def cb(index, dtype, pages, page=None):
            return cb_descriptors([index], pages, core_set, dtype=dtype, page=page)

        def scratch(index, size):
            return [ttnn.CBDescriptor(total_size=r64(size), core_ranges=core_set,
                                      format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, r64(size))])]

        row_bytes = r64((48 if batch else 44) * R)  # Row[R] (attn_sdpa.cpp; BATCH: + the slot's ring base)
        # + BATCH: the page-table double buffer (full layers) and the rows-done poll word
        scratch_bytes = 64 + row_bytes + r64(4 * max_parts) + 3 * 4 * 2 * 64 + (256 + 64 if batch else 0)
        # + BATCH: the writer's gate-row staging (it reads the gate rows: reductions interleave with the rows)
        mail_bytes = slots * r64(16 * (pmax + 1)) + 128 + row_bytes + 128 * slots + (3 * 4 * 2 * 64 if batch else 0)
        bf, f32 = ttnn.bfloat16, ttnn.float32
        o = self.outstanding
        kd, vd = self.k_dtype, self.v_dtype
        # params: header + a word per partition (BATCH: + its row)
        cbs = (cb(0, bf, 10) + cb(1, kd, 8 * (o + 1)) + cb(2, vd, 8 * v_blocks)
               + cb(3, bf, 4) + cb(4, bf, 1) + cb(5, bf, 1) + cb(6, bf, 1)
               + scratch(7, 4 * (8 + (2 if batch else 1) * max_parts))
               + cb(8, bf, 2 * (o + 1)) + scratch(9, scratch_bytes) + scratch(10, mail_bytes) + cb(11, bf, nkt)
               + cb(12, bf, 4 * slots)
               + cb(16, f32, 2 * C) + cb(17, f32, 1) + cb(18, f32, slots) + cb(19, bf, 2 * C) + cb(20, f32, 5)
               + cb(21, f32, 5 * nkt * slots) + cb(22, bf, 4) + cb(23, f32, 1) + cb(24, f32, nkt * slots)
               + cb(25, f32, nkt) + cb(26, f32, 1) + cb(27, f32, 1) + cb(28, f32, 5))
        l1 = sum(c.total_size for c in cbs)
        if l1 > 1_200_000:
            raise ValueError(f"SDPADecodeInk: {l1} B of L1 per core")
        named = [("workers", W), ("gather_kt", nkt), ("outstanding", o), ("chunk_blocks", C), ("partitions", pmax),
                 ("cores", nc), ("slots", slots), ("k0", k0), ("max_parts", max_parts), ("ring", int(not self.full)),
                 ("k_bytes", TILE_BYTES[kd]), ("v_bytes", TILE_BYTES[vd]), ("rows", R), ("fused", 1), ("sem0", 1),
                 ("ready", ready), ("gates", gates), ("gate_stride", self.gate_cols[0]),
                 ("gate_div", self.gate_cols[1]), ("gate_off", self.gate_cols[2])]
        out = self.att if R == 1 else self.att_rows
        q = self.q_batch if batch else self.q
        common = [int(t.buffer_address()) for t in (q, a.k_cache[slot], a.v_cache[slot], tables.cur_pos,
                                                    self.table, self.consts, out, tables.cur_pos, self.P)]
        common += [v for c in noc for v in (c.x, c.y)]
        defines = [("EXP_ITERS", "2")]
        if batch:  # + the fronts' noc coordinates: the SDPA polls their rows-done semaphore (ROWS_SEM)
            front_noc = [dev.worker_core_from_logical_core(c) for c in self.front_cores]
            common += [int(tables.slot.buffer_address()), int(tables.page_table.buffer_address())]
            common += [v for c in front_noc for v in (c.x, c.y)]
            defines += [("BATCH", "1"), ("PT_PAGE", str(self._pt_page(tables))), ("ROWS_SEM", str(ROWS_SEM))]
        acc = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        kernels = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "attn_sdpa.cpp"), core_ranges=core_set, defines=defines,
            ncrisc_named_compile_time_args=named, brisc_named_compile_time_args=named,
            trisc_named_compile_time_args=named, ncrisc_compile_time_args=acc, brisc_compile_time_args=acc,
            ncrisc_common_runtime_args=common, brisc_common_runtime_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                                              math_approx_mode=False, dst_full_sync_en=True),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(ncrisc_args=rt_args, brisc_args=rt_args,
                                                                          trisc_args=rt_args),
        ).get_kernel_descriptors().kernels
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=core_set, initial_value=0)
                for s in list(range(1, 1 + slots)) + [gates] + list(range(ready, ready + 8))]
        return kernels, cbs, sems, noc, gates

    def _app_program(self, xs, slot, tables, R, pf: bool = False, batch: bool = False):
        """One program: projection workers (big: one group, or q|gate + k|v groups in their own formats; small BF16;
        rows 0..R-1), the 8 front cores and the SDPA cores; the SDPA reads the query tiles / newest blocks once the
        front of their group signals.  pf: part of the projection weights was prefetched into L1 (AttnPrefetch,
        plan pf_plan[R][slot]).  batch: tables = the batch tables, every row its own sequence (front / SDPA BATCH;
        slot = the pass)."""
        a, d = self.attn, self.device
        names = ["big", "kv"][:len(self.big_groups)]
        mmbs = [self.pf_mm[n] if pf else mm for n, (mm, _) in zip(names, self.big_groups)]
        mms = self.pf_mm["small"] if pf else self.mm_small
        plan = self.pf_plan[R][slot] if pf else {}
        fcores = self.front_cores
        fset = row_ranges(fcores)
        front_noc = [(n.x, n.y) for n in (d.worker_core_from_logical_core(c) for c in fcores)]
        r_big, r_small, expected = self._routes(front_noc)
        if self.mixed:  # q|gate: xq; k|v: xk | xv (segments 0 / 32)
            big_ins = [(xs[0],), (xs[0], xs[0]) if self.fused else (xs[1], xs[2])]
        else:
            big_ins = [(xs[0],) if self.fused else (xs[0], xs[1], xs[2])]
        small_in = (xs[0], xs[1], xs[2]) if self.full else (xs[0],)
        ready = 8  # front-ready semaphores 8..15 on the SDPA cores (gather: 1..slots)
        s_kernels, s_cbs, s_sems, s_noc, gates = self._sdpa_parts(slot, tables, R, self.sdpa_cores, ready, batch)
        done = 7  # front 0: count of group-0 workers (the gate rows' writers) with every write (incl. P) landed
        xf = batch and R >= XF_ROWS  # input rows gathered before the weight stream (decode_matmul_stream.cpp)
        kernels = []
        for gi, (mm, ins, routes, n) in enumerate(zip(mmbs, big_ins, r_big, names)):
            kernels += mm.kernel_descriptors(ins, self.P, R, routes, _ROUTE_CBS, 2048,
                                             (done,) + front_noc[0] if gi == 0 else None, plan.get(n), x_first=xf)
        kernels += mms.kernel_descriptors(small_in, self.S, R, r_small, _ROUTE_CBS, 2048, prefetch=plan.get("small"),
                                          x_first=xf)
        addrs = [int(t.buffer_address()) for t in (self.P, self.S, a.conv_ring[slot], self.taps, tables.cur_pos,
                                                   tables.tau, a.k_cache[slot], a.v_cache[slot],
                                                   self.q_batch if batch else self.q)]
        defines = self.front_defines + [("ROWS", str(R)), ("SDPA_SIGNAL", "1"), ("READY_SEM", str(ready)),
                                        ("DONE_SEM", str(done)), ("N_BIG", str(len(mmbs[0].core_args))),
                                        ("GATES_SEM", str(gates))]
        if batch:
            addrs += [int(tables.slot.buffer_address()), int(tables.page_table.buffer_address())]
            defines += [("BATCH", "1"), ("PT_PAGE", str(self._pt_page(tables))), ("ROWS_SEM", str(ROWS_SEM))]
        waddrs = addrs + [len(s_noc)] + [v for c in s_noc for v in (c.x, c.y)]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                               fp32_dest_acc_en=True, dst_full_sync_en=True)
        unpack = [ttnn.UnpackToDestMode.Default] * 64
        unpack[3] = ttnn.UnpackToDestMode.UnpackToDestFp32  # fp32 taps
        compute.unpack_to_dest_mode.extend(unpack)
        kernels += unified_kernels(KERNELS / "attn_front.cpp", fset, defines, waddrs, waddrs,
                                   [(c, [g, expected[g]]) for g, c in enumerate(fcores)], compute)
        kernels += s_kernels
        everyone = row_ranges([c for mm in mmbs + [mms] for c, _ in mm.core_args] + fcores)
        bf, f32 = ttnn.bfloat16, ttnn.float32
        # routed targets first, on every core: the same L1 address on the senders and the front cores
        cbs = []
        for idx, n in zip(_ROUTE_CBS, (R, R, 3 + R, R)):  # X: x[p-3], x[p-2], x[p-1], then the R tokens
            cbs += cb_descriptors([idx], n, everyone, dtype=bf)
        for mm, ins in zip(mmbs, big_ins):
            cbs += mm.cbs(mm.n_in(ins), xf)
        cbs += mms.cbs(1, xf)
        o = min(R, 2)  # per-token outputs double-buffered between compute and writer
        for idx, n, dt in ((3, 4, f32), (4, 1, bf), (5, 1, bf), (7, 5, bf), (16, 1, bf), (17, 3, f32), (18, 8, f32),
                           (19, o, bf), (20, o, bf), (21, o, self.k_dtype), (22, o, self.v_dtype)):
            cbs += cb_descriptors([idx], n, fset, dtype=dt)
        if batch:  # every row its own block-float tile group; the rows' conv-ring lag tiles (cLag) + their staging
            # (conv-ring tile images: 8 per row)
            cbs += cb_descriptors([6], 1, fset, dtype=bf, page=R * 8 * 1088)
            cbs += cb_descriptors([8], 1, fset, dtype=bf, page=1216 + 8 * 2048 * R)
            cbs += cb_descriptors([9], 3 * R, fset, dtype=bf)
        else:
            cbs += cb_descriptors([6], 1, fset, dtype=bf, page=16 * 1088)  # block-float cache tiles to patch (2 groups)
            cbs += cb_descriptors([8], 1, fset, dtype=bf, page=128 + 24 * 576)  # conv-ring rows staging
        cbs += s_cbs
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=everyone, initial_value=0),
                ttnn.SemaphoreDescriptor(id=done, core_ranges=fset, initial_value=0)] + s_sems
        if batch:  # each front's rows done (attn_front.cpp writer; polled by the SDPA cores)
            sems.append(ttnn.SemaphoreDescriptor(id=ROWS_SEM, core_ranges=fset, initial_value=0))
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)

    # ------------------------------------------------------------------ call
    def _check_rows(self, R):
        if not 1 <= R <= MAX_ROWS:
            raise ValueError(f"AttnDecodeFast: 1 <= R <= {MAX_ROWS} rows, got {R}")

    def attend(self, xq, xk, xv, tables, slot, R: int = 1, pf: bool = False, batch: bool = False):
        """Launch 1 (projections + front + SDPA of rows 0..R-1) -> the gated attention rows (R == 1: self.att [1, 1, 1,
        3072] BF16 row-major; else rows 0..R-1 of self.att_rows [1, 1, 32, 3072] BF16 TILE).  pf: read the prefetched
        weight blocks from L1 (only right after the glue programs that prefetched them: FastGlue.decode / verify).
        batch: tables = the batch tables (see batch())."""
        a = self.attn
        xs = (xq, xk, xv)
        ins = (xq,) if self.fused and not self.full else xs
        out = self.att if R == 1 else self.att_rows
        tensors = [*ins, *(mm.weight for mm, _ in self.big_groups), self.mm_small.weight, self.P, self.S,
                   a.conv_ring[slot], self.taps, tables.cur_pos, tables.tau, a.k_cache[slot], a.v_cache[slot],
                   self.q_batch if batch else self.q, self.table, self.consts, out]
        if batch:
            tensors += [tables.slot, tables.page_table]
        if pf:
            tensors.append(self.pf.buf)
        self._progs.setdefault(("batch" if batch else "app", slot, id(tables), R, pf), CachedProgram()).run(
            tensors, lambda: self._app_program(xs, slot, tables, R, pf, batch))

    def _single(self):
        if getattr(self.attn, "batched", False):
            raise ValueError("AttnDecodeFast: a batched attention module (slots / pool) decodes through batch()")

    def __call__(self, xq, xk, xv, tables, pass_idx: int = 0):
        """Decode one token: xq/xk/xv [1, 1, 32, 2304] BF16 TILE interleaved, one memory config (row 0 = the token;
        other rows ignored) -> [1, 1, 32, 2304] (row 0 = attention output, rows 1.. zero)."""
        self._single()
        pf = self.pf is not None and self.pf.active == 1
        self.attend(xq, xk, xv, tables, pass_idx, pf=pf)
        if pf:  # launch 2: [1, 1, 32, 2304], row 0 valid, rows 1.. zero
            return self.pf_mm["o"](self.att, prefetch=self.pf_plan[1][pass_idx]["o"], prefetch_buf=self.pf.buf,
                                   key=pass_idx)
        return self.mm_o(self.att)

    def verify(self, xq, xk, xv, tables, R: int, pass_idx: int = 0):
        """R (1..8) rows at consecutive positions p .. p+R-1 in one call (speculative verify): xq/xk/xv [1, 1, 32,
        2304] BF16 TILE (row j = the token at p + j; rows >= R ignored); tables: attention.AttnVerifyTables (word j of
        cur_pos = p + j, tau element (0, j)).  Equals R single-row decode steps at p .. p+R-1 BITWISE (attn_sdpa.cpp
        splits every row over its cores as a decode does): causal across rows (row j attends keys <= p + j: every row's
        K / V is written before the SDPA), per-row bias / tau / sink.  Writes only
        position-indexed state (ring slot / full row / conv-ring row of each row's position): after accepting k rows,
        the next decode / verify simply starts at p + k (the rejected rows' state is never read before rewritten).
        -> [1, 1, 32, 2304] (rows 0..R-1 = attention outputs, rows R.. zero).  Inside FastGlue.verify with AttnPrefetch
        (pf.active == R) the prefetched blocks of plan pf_plan[R] are read from L1 (the same bytes: bitwise unchanged)."""
        self._check_rows(R)
        self._single()
        pf = self.pf is not None and self.pf.active == R and R in self.pf_plan
        self.attend(xq, xk, xv, tables, pass_idx, R, pf=pf)
        if R == 1:
            if pf:
                return self.pf_mm["o"](self.att, prefetch=self.pf_plan[1][pass_idx]["o"], prefetch_buf=self.pf.buf,
                                       key=pass_idx)
            return self.mm_o(self.att)
        if pf:
            return self.pf_mm["o_rows"](self.att_rows, rows=R, prefetch=self.pf_plan[R][pass_idx]["o"],
                                        prefetch_buf=self.pf.buf, key=pass_idx)
        return self.mm_o_rows(self.att_rows, rows=R)

    def batch(self, xq, xk, xv, bt, R: int, pass_idx: int = 0):
        """Batched serving: R (1..16) rows, row b = the decode step of its own sequence: position bt.cur_pos[b] (tau:
        bt.tau element (0, b)), physical slot bt.slot[b] (its conv-ring rows and sliding K / V ring), full layers'
        keys and K / V row through bt.page_table row b (pool blocks).  xq/xk/xv [1, 1, 32, 2304] BF16 TILE (row b =
        row b's input; rows >= R ignored); bt: the batch tables (CONTRACT / fast/batch_tables.py).  Row b of the output
        and every state write of row b are BITWISE the one-row decode (__call__) of that sequence: same per-row
        programs (front math, the SDPA split of a one-row decode: attn_sdpa.cpp vcore), R == 1 the decode's row-major
        o_proj path.  Pad rows (replicas of row 0) write identical values to row 0's state.  -> [1, 1, 32, 2304] (rows
        0..R-1 = outputs, rows R.. zero)."""
        if not 1 <= R <= BATCH_ROWS:
            raise ValueError(f"AttnDecodeFast.batch: 1 <= R <= {BATCH_ROWS} rows, got {R}")
        if self.q_batch is None:
            raise ValueError("AttnDecodeFast.batch: the attention module is not batched (seqs / kv_blocks)")
        self.attend(xq, xk, xv, bt, pass_idx, R, batch=True)
        if R == 1:
            return self.mm_o(self.att)
        return self.mm_o_rows(self.att_rows, rows=R, x_first=R >= XF_ROWS)


# ------------------------------------------------------------------------------------------------ L1 weight prefetch
PF_SHARD_KB = 256  # L1 per prefetch core (a persistent buffer: every program's CB ceiling drops by this much)
# Share of every stream worker's weight blocks served from L1, per application class: "plain" = its PRE has no message
# gates (glue programs with a ~10 us DRAM-idle window), "gated" = loop passes 2 and 3 (~20-27 us windows).  Row-packed
# glue (NOTES-glue.md "Row-packed glue"): 7/16 everywhere (sweep: 1/2 +0.05 ms, 3/8 +0.03, 5/8 +0.1).
PF_FRACTION = {"plain": {"big": 7 / 16, "kv": 7 / 16, "small": 1 / 3, "o": 7 / 16},
               "gated": {"big": 7 / 16, "kv": 7 / 16, "small": 1 / 3, "o": 7 / 16}}
PF_RING = {"big": (6, 4), "kv": (6, 4), "small": None, "o": (6, 4)}  # (slots, outstanding) of the readers' in1 ring
PF_VERIFY = (2, 3, 4)  # verify row counts with a plan (FastGlue.verify: R glue instances; R <= the glue's max_rows)
# verify plans: PF_FRACTION x PF_VERIFY_SCALE[R] (then fitted, PF_SCALE), copies starting PF_VERIFY_DELAY[R] us into
# the R-instance glue programs (their own DRAM reads take longer; None = the decode glue's PF_GLUE starts).  Re-swept
# for the row-packed glue (NOTES-glue.md; verify ms vs the earlier x 0.4 @ 20 us: R 2 -0.26, R 3 -0.17, R 4 -0.08)
PF_VERIFY_SCALE = {2: 0.5, 3: 0.5, 4: 0.3}
PF_VERIFY_DELAY = {2: 10.0, 3: 14.0, 4: 20.0}
PF_SCALE = 0.85        # verify plans: fractions scaled by this until the blocks fit the cores outside the R glues
# Glue program kind -> (prefetch start after the program start, us: the glue's own DRAM reads come first; DRAM-idle
# window, us: splits an application's copies between the MID before it and its PRE).  NOTES-attn.md Round 4; starts
# re-swept for the row-packed glue (NOTES-glue.md: 10 / 10 / 10 -> 5 / 6 / 8 us; the windows' split ratio measured
# no effect, kept).
PF_GLUE = {"plain": (5.0, 12.0), "pre_gated": (6.0, 31.0), "mid_gated": (8.0, 28.0)}
AICLK_MHZ = 1350  # Blackhole wall clock (attn_prefetch.cpp delay)


def _every(n: int, f: float) -> list[int]:
    """Indices j < n at rate f, spread evenly (floor((j + 1) f) > floor(j f)): with f = 1/2 blocks 1, 3, 5, ... come
    from L1 and 0, 2, 4, ... from DRAM, so every worker's DRAM stream starts at once and never waits behind L1."""
    return [j for j in range(n) if int((j + 1) * f + 1e-9) > int(j * f + 1e-9)]


def _bank_cores(device, cores) -> list[list[int]]:
    """Prefetch cores (indices into `cores`) per DRAM bank: every bank gets cores of its half of the grid (the
    partition of moe.plan_noc0_workers), dealt one per bank per round, each the free core whose NOC0 response path
    (moe.noc0_flow_links) shares the fewest links with other banks' paths, then the fewest hops; until none is left."""
    from ..moe import noc0_flow_links

    g = device.compute_with_storage_grid_size()
    anchors = [(c.x, c.y) for c in device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
    starts = sorted({a[0] for a in anchors})
    half_of = {s: (s, ([t for t in starts if t > s] or [g.x])[0]) for s in starts}
    free = {(c.x, c.y): k for k, c in enumerate(cores)}
    used, out = {}, [[] for _ in anchors]
    dealt = True
    while dealt:
        dealt = False
        for b, anchor in enumerate(anchors):
            half = half_of[anchor[0]]
            best = None
            for c in sorted(free):
                if half[0] <= c[0] < half[1]:
                    links, hops = noc0_flow_links(anchor, c, half, g.y)
                    key = (sum(1 for ln in links if used.get(ln, b) != b), hops, c[1], c[0])
                    if best is None or key < best[0]:
                        best = (key, c, links)
            if best is not None:
                out[b].append(free.pop(best[1]))
                for ln in best[2]:
                    used.setdefault(ln, b)
                dealt = True
    return out


class AttnPrefetch:
    """L1 prefetch of the attention weights of the NEXT application during the glue programs before it (MID of the
    previous application, PRE of this one: DRAM mostly idle), so the attention programs stream only the rest from DRAM.

    - One L1 buffer (`buf`, HEIGHT_SHARDED, `shard_kb` per core) on every worker core outside the decode glue rectangle
      (grid.groups(1)), allocated in the constructor (trace safety) and shared by all applications (one at a time).
      The prefill programs need that L1 for their CBs (the rows glue's large layouts reach ~1.31 MB): `released()`
      frees it around a prefill forward and re-allocates it at the same address (prefill keeps no device buffer past
      the call, model.py trace safety), so the captured decode / verify traces stay valid; a moved address raises.
    - Per row count R (1 = decode; PF_VERIFY = speculative verify through FastGlue.verify, whose R glue instances
      occupy grid.groups(R): only the buffer cores outside them copy) and application (layer, pass): a fraction
      (PF_FRACTION, by application class; verify: scaled by PF_SCALE until it fits the fewer cores) of every stream
      worker's blocks (f.stream_mats(): projection group(s), small projection, o_proj; spread by `_every`) is placed
      on the least-filled shard of the block's DRAM bank's core set (`_bank_cores`: planned NOC0 response paths),
      nearest to the reading worker on ties.  plans[(R, layer, pass)]["tasks"][core] = that core's DRAM -> shard copies
      (kernels/fast/attn_prefetch.cpp); the first "mid"[core] of them run inside the previous application's MID, the
      rest inside this application's PRE (split by the two programs' DRAM-idle windows, PF_GLUE; application 0: all
      in its PRE).  The copies start PF_GLUE[kind][0] us into the glue program, after the glue's own DRAM reads.
    - The attention module reads those blocks from L1 (decode_matmul_stream.cpp `pf`) only while `active` == its row
      count (FastGlue.decode sets 1, FastGlue.verify R, around the glue programs carrying the copies); standalone
      calls stream everything from DRAM.  The values are the same bytes: outputs are bitwise unchanged.
    Order: the attention + o_proj of application i - 1 finish before MID(i - 1) overwrites the buffer; MoE / MSG / PRE
    programs in between cannot touch it (allocated buffer: no CB overlaps it)."""

    def __init__(self, device, cfg, fasts, grid, shard_kb: int = PF_SHARD_KB, fraction=None, glue=None,
                 verify_rows=PF_VERIFY):
        g = device.compute_with_storage_grid_size()
        ex = {(c.x, c.y) for c in grid.groups(1)[0].coords}
        self.cfg, self.apps = cfg, cfg.applications()
        self.cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x) if (x, y) not in ex]
        self.noc = [device.worker_core_from_logical_core(c) for c in self.cores]
        # copying cores per row count: indices into self.cores outside the R glue instances
        self.subset = {1: list(range(len(self.cores)))}
        for R in verify_rows:
            if 2 <= R <= grid.max_instances:
                exr = {(c.x, c.y) for grp in grid.groups(R) for c in grp.coords}
                self.subset[R] = [k for k, c in enumerate(self.cores) if (c.x, c.y) not in exr]
        self.core_sets = {R: row_ranges([self.cores[k] for k in ks]) for R, ks in self.subset.items()}
        self.bank_cores = {}
        for R, ks in self.subset.items():
            banks = _bank_cores(device, [self.cores[k] for k in ks])
            self.bank_cores[R] = [[ks[j] for j in b] for b in banks]
        self.shard = shard_kb * 1024
        rows = self.shard // 2048
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(self.core_sets[1], [rows, 1024], ttnn.ShardOrientation.ROW_MAJOR))
        self._alloc = lambda: ttnn.allocate_tensor_on_device(  # noqa: E731  (contents: written by a copy before use)
            ttnn.Shape([1, 1, len(self.cores) * rows, 1024]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, mc)
        self.buf = self._alloc()
        self.base = int(self.buf.buffer_address())
        self.fraction = {k: dict(v, **(fraction or {}).get(k, {})) for k, v in PF_FRACTION.items()}
        self.glue = dict(PF_GLUE, **(glue or {}))
        self.verify_scale, self.verify_delay = dict(PF_VERIFY_SCALE), dict(PF_VERIFY_DELAY)
        self.active = 0  # row count of the running decode (1) / verify (R) step whose glue programs copy; 0 = none
        self.plans, self.weights, self._kernels, self.verify_fraction = {}, {}, {}, {}
        for f in fasts:
            self.attach(f)

    @contextlib.contextmanager
    def released(self):
        """Free the buffer for the duration (prefill); re-allocate it at its address afterwards."""
        if self.buf is None:  # nested
            yield
            return
        ttnn.deallocate(self.buf)
        self.buf = None
        try:
            yield
        finally:
            self.buf = self._alloc()
            if int(self.buf.buffer_address()) != self.base:
                raise RuntimeError(f"AttnPrefetch: L1 buffer re-allocated at {int(self.buf.buffer_address())}, not "
                                   f"{self.base} (a device buffer allocated during the prefill outlived it): the "
                                   "captured decode traces would read a moved buffer")

    def kind(self, layer: int, p: int, site: str) -> str:
        """PF_GLUE kind of the glue program `site` ("pre" / "mid") of application (layer, p)."""
        return f"{site}_gated" if self.cfg.is_loop(layer) and p > 0 else "plain"

    def _place(self, mats, frac, R: int = 1):
        """Blocks of every worker at the given fractions, on the copying cores of row count R -> (per core copies,
        {matmul: {worker: (block mask, [(noc x, noc y, L1 address)])}})."""
        free = [self.shard if k in set(self.subset[R]) else 0 for k in range(len(self.cores))]
        tasks = [[] for _ in self.cores]
        items = []  # (matmul, worker, worker core, block index, block)
        for name, mm in mats.items():
            for i, (core, _) in enumerate(mm.core_args):
                blocks = mm.blocks(i)
                if len(blocks) > 32:
                    raise ValueError(f"AttnPrefetch: {name} worker {i}: {len(blocks)} blocks > 32")
                for j in _every(len(blocks), frac[name]):
                    items.append((name, i, core, j, blocks[j]))
        items.sort(key=lambda t: (-t[4][2], t[3]))  # big blocks first (packing)
        locs = {}
        for name, i, core, j, (bank, addr, nbytes) in items:
            fits = [k for k in self.bank_cores[R][bank] if free[k] >= nbytes]
            fits = fits or [k for k in self.subset[R] if free[k] >= nbytes]
            if not fits:
                raise ValueError(f"AttnPrefetch: {self.shard} B shards on {len(self.subset[R])} cores too small for "
                                 f"{frac} (rows {R})")
            k = min(fits, key=lambda k: (self.shard - free[k], abs(self.cores[k].x - core.x)
                                         + abs(self.cores[k].y - core.y), k))
            off = self.shard - free[k]
            free[k] -= nbytes
            tasks[k].append((bank, addr, nbytes, off))
            locs.setdefault((name, i), []).append((j, self.noc[k].x, self.noc[k].y, self.base + off))
        plan = {name: {} for name in mats}
        for (name, i), ls in locs.items():
            ls.sort()
            plan[name][i] = (sum(1 << j for j, *_ in ls), [(x, y, a) for _, x, y, a in ls])
        return tasks, plan

    def _place_rows(self, mats, frac, R: int):
        """_place; verify rows: the fractions scaled by PF_SCALE until the blocks fit (the chosen fractions are kept in
        verify_fraction[R, class name])."""
        while True:
            try:
                return self._place(mats, frac, R), frac
            except ValueError:
                if R == 1 or max(frac.values()) < 0.05:
                    raise
                frac = {k: v * PF_SCALE for k, v in frac.items()}

    def attach(self, f):
        """Plans of every application of f's layer and every row count; f (AttnDecodeFast) gets pf, pf_plan[R][pass],
        pf_mm (deeper-ring copies of its stream matmuls; "o_rows" = verify's tile-input o_proj, the same blocks)."""
        L = f.attn.layer
        mats = f.stream_mats()
        f.pf_mm = {}
        for name, mm in list(mats.items()) + [("o_rows", f.mm_o_rows)]:
            c = copy.copy(mm)
            ring = PF_RING["o" if name == "o_rows" else name]
            if ring:
                c.preset = dataclasses.replace(mm.preset, slots=ring[0], outstanding=ring[1])
            c._progs = {}
            f.pf_mm[name] = c
        f.pf, f.pf_plan = self, {}
        self.weights[L] = [mm.weight for mm in mats.values()]
        for R in self.subset:
            placed = {}
            for i, (layer, p) in enumerate(self.apps):
                if layer != L:
                    continue
                cls = "gated" if self.cfg.is_loop(L) and p > 0 else "plain"
                if cls not in placed:
                    scale = 1.0 if R == 1 else self.verify_scale.get(R, 1.0)
                    frac = {k: v * scale for k, v in self.fraction[cls].items() if k in mats}
                    placed[cls], self.verify_fraction[R, L, cls] = self._place_rows(mats, frac, R)
                tasks, plan = placed[cls]
                w_pre = self.glue[self.kind(L, p, "pre")][1]
                w_mid = self.glue[self.kind(*self.apps[i - 1], "mid")][1] if i else 0.0
                self.plans[R, L, p] = {"tasks": tasks,
                                       "mid": [round(len(ts) * w_mid / (w_mid + w_pre)) for ts in tasks],
                                       "bytes": sum(t[2] for ts in tasks for t in ts)}
                f.pf_plan.setdefault(R, {})[p] = plan
        self._kernels.clear()

    def _sel(self, plan, site: str):
        return [ts[:m] if site == "mid" else ts[m:] for ts, m in zip(plan["tasks"], plan["mid"])]

    def glue_extra(self, i: int, site: str, R: int = 1):
        """(tensors, build) for the glue program (R instances) that copies application i's blocks of `site`: "mid" =
        the MID of application i - 1, "pre" = the PRE of application i.  build() -> the prefetch kernel descriptors
        (one data-movement kernel on row count R's copying cores).  Runtime args are padded to the largest copy count of
        that site over all applications of R (compile-time `slots`): programs that hash alike (generic_op's program
        cache ignores runtime-arg values and counts) get equally long args."""
        L, p = self.apps[i]
        kind = self.kind(*self.apps[i - 1], "mid") if site == "mid" else self.kind(L, p, "pre")
        delay = self.verify_delay.get(R) if R > 1 else None
        delay = self.glue[kind][0] if delay is None else delay

        def build():
            key = (i, site, R)
            if key not in self._kernels:
                slots = max(len(s) for (r, *_), plan in self.plans.items() if r == R for s in self._sel(plan, site))
                sel = self._sel(self.plans[R, L, p], site)
                rt = ttnn.RuntimeArgs()
                for k in self.subset[R]:
                    c = self.cores[k]
                    rt[c.x][c.y] = [len(sel[k])] + [v for t in sel[k] for v in t] + [0] * (4 * (slots - len(sel[k])))
                self._kernels[key] = [ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "attn_prefetch.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.core_sets[R],
                    named_compile_time_args=[("slots", slots)],
                    common_runtime_args=[self.base, int(delay * AICLK_MHZ)], runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                             noc=ttnn.NOC.RISCV_0_default))]
            return self._kernels[key]

        return self.weights[L] + [self.buf], build
