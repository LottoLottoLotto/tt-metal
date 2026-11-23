# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Inkling2 attention (sliding-window and full layers) on one P150: decode (traceable) and chunked prefill.

    tables = AttnDecodeTables(mesh_device, cfg, max_seq_len=32768, full_len=4096)   # one per model (per bucket)
    attn = Inkling2Attention(mesh_device, source, layer_idx, cfg, max_seq_len=32768, tables=tables)
    out = attn.prefill(xq, xk, xv, start_pos, length, pass_idx)    # [1, 1, T, 2304] in -> [1, 1, T, 2304] out
    tables.update(pos)                                             # host -> device, before every decode (replay)
    out = attn.decode(xq, xk, xv, tables.cur_pos, pass_idx)        # [1, 1, 32, 2304] in (row 0) -> out (row 0)
    attn.reset()                                                   # new sequence: conv states of every pass slot

Batched serving (NOTES-batch.md): Inkling2Attention(..., seqs=S, kv_blocks=NB) keeps S sequence slots (sliding rings
[S, 8, 1152, 128], conv rings [S, 1, 32, 2048]: slot s = index s) and one full-attention pool [NB + 1, 8, 64, 128]
(block NB = zero block) per pass; attn.fast.batch(xq, xk, xv, bt, R, pass_idx) decodes R rows of different
sequences (bt: the batch tables), attn.prefill reads the slot from word 2 of prefill_pos and the sequence's page row
from attn.prefill_pt; attn.reset_slot(s) starts a new sequence in slot s.

Semantics = transformers PR #51050 `Inkling2LoopSlidingAttention` / `Inkling2LoopFullAttention` (one sequence):
xq / xk / xv are the already-normed q / k / v inputs, the output is the gated o_proj output BEFORE the post-attention
short conv.  Non-loop sliding layers (0-4, 18-22) use one fused projection of xq (the caller passes identical
xq / xk / xv there; xk / xv are ignored); every other layer projects xq, xk, xv separately.

    q = L2clip(Wq xq * q_scale, 22.63) [* tau(pos) on full layers]               (per head, fp32 math)
    k = L2clip(conv_k(Wk xk), 22.63),  v = rmsnorm(conv_v(Wv xv))                 (conv: fp32, kernel 4, residual,
        sliding: static depthwise taps, / sqrt(5); full: static + per-token taps from W_dyn xk / xv per kv head, / 3)
    r = L2clip(Wr xq * r_scale, 8) [* tau];  bias[h, q, k] = (r_q @ proj)[h, q - k] for 0 <= q - k < 1024, else 0
    sink[h, q] = Ws xq  (extra logit with a zero value);  scores = q k^T / sqrt(128) + bias (+ window / causal mask)
    out = Wo (softmax-attention * sigmoid(Wg xq)),   tau(pos) = 1 + 0.1 * log(max((pos + 1) / 8192, 1))

Device formulation (decode: ttnn ops; prefill: the kernels of tt/fast/attn_prefill.py with the same numerics):
- q_scale / r_scale are folded into the Wq / Wr rows (host fp32 product, then the weight format).  The 1/sqrt(128)
  scale is folded into the q clip factor (q is rounded to bf16 once); SDPA runs with scale 1.0.  Norm factors are fp32
  (sum of exact bf16 squares, rsqrt), applied in one multiply.
- Decode: bias, sink and masks are ONE additive SDPA mask, mask[h, slot] = r_aug[h] . P_aug[:, idx[slot]] =
  matmul(r_aug, embedding(idx, P_aug^T)^T) with the per-step index row from `AttnDecodeTables` (P_aug [32, 1056]:
  columns 0..1023 = proj[:, d] (distance d), 1024 = 0 (valid, no bias), 1025 = NEG (masked), 1026 = sink; r_aug =
  [r(16) | 1 | sink | 0..], P_aug[16, 1025] = NEG, P_aug[17, 1026] = 1: exact).  The sink is a key slot with
  K = V = 0 whose mask entry is the sink logit (q . 0 = 0), so SDPA needs no sink argument.
  Prefill: the flash kernels compute the bias / masks / sink per tile (fast/attn_prefill.py).
- K and V go through one 2048-wide conv (k | v channels).  Conv states per slot: rows x[t-3], x[t-2], x[t-1] of the
  [Wk xk | Wv xv] linear outputs ([1, 1, 1, 2048] bf16 each, exact: the linear outputs are bf16; the stock decode
  shifts them) and the fused path's position-indexed conv-input ring `conv_ring` ([1, 1, 32, 2048] bf16: row
  pos % 32 = x[pos]; a rejected verify row only writes its own position's row).  Prefill writes both (the shift states
  and the chunk's last <= 32 ring rows: FlashPrefill.conv_update).  Products and
  sums in fp32 on the SFPU (exact products); sliding layers fold the residual and 1/sqrt(5) into fp32 taps.
- Caches per pass slot (3 for loop layers 6-17, else 1); formats `dtype_policy["ring"]` (sliding) and
  `["k"]` / `["v"]` (full layers, independent: BF16 / BFP8 / BFP4; BF16 by default):
  sliding: ring [1, 8, 1088 + 64, 128] (slot = pos % 1088 (RING_SLOTS), keys at distance < 1024 valid, so the 64
  slots after the newest position are dead and up to 64 consecutive positions can be written before one is read;
  rows 1088.. stay zero: row 1088 is the stock decode's sink key);
  full: paged [max_seq_len / 64 + 1, 8, 64, 128] (identity block order; the last block stays zero = sink key block).
  Full layers keep positions < max_seq_len (constructor argument; default 32768).
- Decode = stock SDPA decode (non-causal, attn_mask): sliding over the 1152-row ring, full over the first
  `tables.full_len` positions (+ the zero block) of the paged cache.  Cost grows with full_len (the "bucket"), not pos:
  the caller picks full_len > every decoded position (one AttnDecodeTables + trace per bucket).

Decode inputs: xq/xk/xv [1, 1, 32, 2304] bf16 TILE (row 0 = the token; other rows finite, ignored); cur_pos = int32
[1] ROW_MAJOR device tensor (used for the full-layer cache write; `tables.cur_pos` is one, written by tables.update).
Per-step device tables (AttnDecodeTables.update(pos), shared by every layer, written before trace replay): ring_pos
(pos % 1088), ring_idx [1, 1152] / full_idx [1, full_len + 64] uint32 gather rows, tau (log scale) [1, 1, 1, 1] fp32.
Fused multi-row verify (tt/fast/attn_decode.py AttnDecodeFast.verify): `AttnVerifyTables` (rows p..p+R-1).
Output [1, 1, 32, 2304] bf16 (row 0 valid, rows 1..31 finite junk).

Prefill inputs: [1, 1, T, 2304] bf16 (T % 32 == 0), rows 0..length-1 = positions start_pos..start_pos+length-1 (any
start_pos; chunks must be fed in order per pass slot; rows >= length are padding: the K / V front reads them as zero,
so the cache / ring rows past the chunk are finite (decode masks them additively: a non-finite masked key or value
would still poison its softmax) and always masked or overwritten).  Output [1, 1, T, 2304] (rows >= length: finite
junk).  start / length reach the
kernels through `prefill_pos` (the model's RowsGlue.pos) or the module's own position tensor, so the chunk's
programs depend only on T.
"""
from __future__ import annotations

import math

import torch
import ttnn

from .common import CK_HIFI2, CK_HIFI4, DRAM, WeightCache, free
from .precision import attn_tag, attn_weight, attn_weight_map

NEG = -30000.0      # masked logit (exp(NEG - max) == 0 in fp32)
RING = 1024         # sliding window == relative-position extent
RING_SLOTS = 1088   # sliding ring modulus: slot = pos % 1088 (>= 1024 + 63: rows of the next 64 positions are dead)
RING_ROWS = RING_SLOTS + 64  # physical ring rows (the last 64 stay zero: the stock decode's sink block)
CONV_RING = 32      # rows of the position-indexed conv-input ring (row = pos % 32)
BLOCK = 64          # full-layer paged cache block; SDPA k chunk
PAD = 64            # zero key rows after the ring / one zero block after the full cache (row 0 = sink slot)
COLS = 1056         # P_aug columns: 1024 distances, ZERO, MASKED, SINK, padding
ZERO, MASKED, SINK = 1024, 1025, 1026
DEFAULT_POLICY = {"proj": ttnn.bfloat8_b, "o": ttnn.bfloat8_b, "small": ttnn.bfloat16, "k": ttnn.bfloat16,
                  "v": ttnn.bfloat16, "ring": ttnn.bfloat16}
SIGMOID = [ttnn.UnaryOpType.SIGMOID]


def _tau(pos: torch.Tensor, cfg) -> torch.Tensor:
    """Log-scaling factor (fp32), as Inkling2LoopLogScale."""
    return 1.0 + cfg.log_scaling_alpha * torch.log(((pos + 1).float() / cfg.log_scaling_n_floor).clamp(min=1.0))


def ring_index_row(pos: int) -> torch.Tensor:
    """[1152] gather row of the sliding ring at decode position `pos` (slot j holds the latest position == j mod 1088;
    valid at distance < 1024); entry 1088 = the sink (zero row)."""
    j = torch.arange(RING_SLOTS)
    d = (pos - j) % RING_SLOTS
    idx = torch.where((pos - d >= 0) & (d < RING), d, torch.full_like(d, MASKED))
    tail = torch.full((PAD,), MASKED, dtype=idx.dtype)
    tail[0] = SINK
    return torch.cat([idx, tail])


def full_index_row(pos: int, full_len: int) -> torch.Tensor:
    """[full_len + 64] gather row of the full cache at decode position `pos`."""
    d = pos - torch.arange(full_len)
    idx = torch.where(d < 0, MASKED, torch.where(d < RING, d, ZERO))
    tail = torch.full((PAD,), MASKED, dtype=idx.dtype)
    tail[0] = SINK
    return torch.cat([idx, tail])


class AttnDecodeTables:
    """Per-step device inputs of every attention layer's decode (shared; write with update(pos) before each replay).

    max_seq_len must equal the attention modules' max_seq_len (the full caches' zero block index).
    full_len: decode bucket of the full layers (multiple of 64, <= max_seq_len); decode positions must be < full_len.
    idx_rows: write the ring_idx / full_idx gather rows in update() (the stock decode path needs them; the fused path
    of tt/fast/attn_decode.py reads only cur_pos and tau).
    """

    def __init__(self, mesh_device, cfg, max_seq_len: int = 32768, full_len: int | None = None, idx_rows: bool = True):
        full_len = full_len or max_seq_len
        assert full_len % BLOCK == 0 and full_len <= max_seq_len and max_seq_len % BLOCK == 0
        self.device, self.cfg, self.max_seq_len, self.full_len = mesh_device, cfg, max_seq_len, full_len
        self.idx_rows = idx_rows
        self.pos = None
        hosts = self._host(0)
        self.cur_pos = ttnn.to_device(hosts[0], mesh_device, memory_config=DRAM)
        self.ring_pos = ttnn.to_device(hosts[1], mesh_device, memory_config=DRAM)
        self.ring_idx = ttnn.to_device(hosts[2], mesh_device, memory_config=DRAM)
        self.full_idx = ttnn.to_device(hosts[3], mesh_device, memory_config=DRAM)
        self.tau = ttnn.to_device(hosts[4], mesh_device, memory_config=DRAM)
        nb = full_len // BLOCK
        pt = torch.cat([torch.arange(nb), torch.tensor([max_seq_len // BLOCK])]).to(torch.int32).reshape(1, -1)
        self.full_page_table = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                               device=mesh_device, memory_config=DRAM)
        self.update(0)

    def _host(self, pos: int):
        def i32(v):
            return ttnn.from_torch(torch.tensor([v], dtype=torch.int32), dtype=ttnn.int32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT)

        def u32(row):
            return ttnn.from_torch(row.to(torch.int32).reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        tau = _tau(torch.tensor([pos]), self.cfg).reshape(1, 1, 1, 1)
        return (i32(pos), i32(pos % RING_SLOTS), u32(ring_index_row(pos)), u32(full_index_row(pos, self.full_len)),
                ttnn.from_torch(tau, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT))

    def update(self, pos: int):
        if pos >= self.full_len:
            raise ValueError(f"decode position {pos} >= full_len {self.full_len} (pick a larger bucket)")
        if not self.idx_rows:
            tau = _tau(torch.tensor([pos]), self.cfg).reshape(1, 1, 1, 1)
            for h, d in ((torch.tensor([pos], dtype=torch.int32), self.cur_pos),
                         (torch.tensor([pos % RING_SLOTS], dtype=torch.int32), self.ring_pos)):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(h, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), d)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(tau, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), self.tau)
            self.pos = pos
            return
        for h, d in zip(self._host(pos), (self.cur_pos, self.ring_pos, self.ring_idx, self.full_idx, self.tau)):
            ttnn.copy_host_to_device_tensor(h, d)
        self.pos = pos


class AttnVerifyTables:
    """Device inputs of the fused multi-row verify (AttnDecodeFast.verify; shared by every layer): rows j = 0..15 at
    positions p + j.  cur_pos int32 [16] ROW_MAJOR (word j = p + j), tau fp32 TILE [1, 1, 32, 32] (element (0, j) =
    tau(p + j)).  A verify of R rows reads words 0..R-1.  Write with update(p) before each replay."""

    ROWS = 16

    def __init__(self, mesh_device, cfg):
        self.device, self.cfg, self.pos = mesh_device, cfg, None
        h_pos, h_tau = self._host(0)
        self.cur_pos = ttnn.to_device(h_pos, mesh_device, memory_config=DRAM)
        self.tau = ttnn.to_device(h_tau, mesh_device, memory_config=DRAM)
        self.update(0)

    def _host(self, pos: int):
        p = torch.arange(pos, pos + self.ROWS)
        tau = torch.zeros(1, 1, 32, 32)
        tau[0, 0, 0, :self.ROWS] = _tau(p, self.cfg)
        return (ttnn.from_torch(p.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                ttnn.from_torch(tau, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT))

    def update(self, pos: int):
        for h, d in zip(self._host(pos), (self.cur_pos, self.tau)):
            ttnn.copy_host_to_device_tensor(h, d)
        self.pos = pos


def _dev(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    if dtype in (ttnn.uint32, ttnn.uint16, ttnn.int32):
        t = t.to(torch.int32)
    else:
        t = t.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=DRAM)


def _height_shard(device, rows: int = 32, cols: int = 128):
    grid = device.compute_with_storage_grid_size()
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(ttnn.num_cores_to_corerangeset(1, grid, row_wise=True), [rows, cols],
                                            ttnn.ShardOrientation.ROW_MAJOR))


def _sdpa_cfg(device, q_chunk=32, k_chunk=BLOCK):
    g = device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y), q_chunk_size=q_chunk,
                                  k_chunk_size=k_chunk, exp_approx_mode=False)


def _sumsq(x):
    """sum over the last dim of x^2 (exact bf16 squares, fp32 sum) -> fp32 [.., 1]."""
    sq = ttnn.mul(x, x, dtype=ttnn.float32, memory_config=DRAM)
    ss = ttnn.sum(sq, dim=-1, keepdim=True)
    ttnn.deallocate(sq)
    return ss


def _clip_factor(x, thr: float, post: float = 1.0, tau=None):
    """post * min(thr / ||x||_2, 1) [* tau] (fp32 [.., 1]); the reference's +1e-6 is below fp32 resolution at the
    clip and only matters for ||x|| ~ 0, where both give factor 1."""
    ss = _sumsq(x)
    f = ttnn.rsqrt(ss)
    ttnn.deallocate(ss)
    ttnn.mul(f, thr * post, output_tensor=f)
    out = ttnn.minimum(f, post)
    ttnn.deallocate(f)
    if tau is not None:
        ttnn.mul(out, tau, output_tensor=out)
    return out


class Inkling2Attention:
    """See the module docstring.  dtype_policy: {"proj": q|gate|k|v weights, "o": o_proj, "small": rel|sink|dynamic
    conv projections, "k" / "v": full-attention K / V caches (BF16 | BFP8 | BFP4, independent), "ring": sliding-window
    caches} (ttnn dtypes; defaults DEFAULT_POLICY).  flash: the model's fast/attn_prefill.FlashPrefill (the prefill
    kernels; None: this module builds its own for max_seq_len).
    weight_precision: {family: ExpertPrecision} of the projection weights (precision.attn_weight_map; None = all BFP8
    RTN); `wtag` = its WeightCache key suffix ("" by default), used by fast/attn_decode for its stream weights.
    Batched serving (NOTES-batch.md): seqs = S sequence slots (sliding rings [S, 8, 1152, 128], conv rings [S, 1, 32,
    2048], slot s = the pages of index s), kv_blocks = NB: the full caches are ONE pool [NB + 1, 8, 64, 128] shared by
    the slots (block NB = zero block) addressed through page tables (decode: the batch tables; prefill:
    `prefill_pt`).  seqs 1 with kv_blocks None is the single-stream module (today's tensors)."""

    def __init__(self, mesh_device, source, layer_idx: int, cfg, *, max_seq_len: int = 32768, dtype_policy=None,
                 tables: AttnDecodeTables | None = None, weight_cache: WeightCache | None = None,
                 flash=None, weight_precision: dict | None = None, seqs: int = 1, kv_blocks: int | None = None):
        assert max_seq_len % BLOCK == 0 and seqs >= 1
        self.device, self.cfg, self.layer, self.max_seq_len, self.tables = mesh_device, cfg, layer_idx, max_seq_len, tables
        self.seqs, self.kv_blocks = seqs, kv_blocks
        self.batched = seqs > 1 or kv_blocks is not None
        pol = dict(DEFAULT_POLICY, **(dtype_policy or {}))
        self.policy = pol
        self.full = cfg.is_full(layer_idx)
        self.nh, self.nkv, self.hd, self.dr = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_rel
        assert (self.nh, self.nkv, self.hd, self.dr, cfg.rel_extent, cfg.sliding_window) == (24, 8, 128, 16, RING, RING)
        self.kvw = self.nkv * self.hd
        self.qw = self.nh * self.hd
        self.fused = not self.full and not cfg.is_loop(layer_idx)  # xq is xk is xv
        self.n_slots = cfg.passes(layer_idx)
        wc = weight_cache or WeightCache(mesh_device)
        p = f"model.llm.layers.{layer_idx}.attn."
        get = lambda n: source.tensor(p + n)  # noqa: E731
        key = f"attn.L{layer_idx}"

        def W(name, build, dtype):
            return wc.get(f"{key}.{name}", build, dtype)

        wp = weight_precision or attn_weight_map("bfp8", cfg.n_layers)[layer_idx]
        self.wtag = attn_tag(wp)

        def WP(name, build, parts, dtype):  # projection weights under the weight precision map
            return attn_weight(wc, f"{key}.{name}", build, parts, wp, layer_idx, dtype)

        qg, kv = [("q", self.qw), ("gate", self.qw)], [("k", self.kvw), ("v", self.kvw)]

        # ---- projections: big = [Wq*s | Wg] (+ [Wk | Wv] when fused), small = W_rs (+ dyn) ----
        def big_q():
            wq = get("wq_du.weight").float() * get("q_l2_clip.scale").float().reshape(-1, 1)
            return torch.cat([wq, get("gate_attn.weight").float()], 0).T

        def w_rs():
            wr = get("wr_du.weight").float() * get("rel_l2_clip.scale").float().reshape(-1, 1)
            ws = get("attn_sink_proj.weight").float()
            w = torch.zeros(self.nh * 32, cfg.hidden)
            for h in range(self.nh):
                w[h * 32:h * 32 + 16] = wr[h * 16:(h + 1) * 16]
                w[h * 32 + 17] = ws[h]
            return w.T

        # fused layers keep one [q|gate|k|v] tensor unless q|gate and k|v have different formats (then separate
        # tensors as on the other layers, so each can stream in its own dtype); w_k / w_v None = not split
        self.w_k = self.w_v = None
        if self.fused and {wp["q"].fmt, wp["gate"].fmt} == {wp["k"].fmt, wp["v"].fmt}:
            self.w_big = WP("qgkv.s", lambda: torch.cat([big_q(), get("wk_dv.weight").T, get("wv_dv.weight").T], 1),
                            qg + kv, pol["proj"])
        else:
            self.w_big = WP("qg.s", big_q, qg, pol["proj"])
            self.w_k = WP("k", lambda: get("wk_dv.weight").T, kv[:1], pol["proj"])
            self.w_v = WP("v", lambda: get("wv_dv.weight").T, kv[1:], pol["proj"])
        self.w_rs = W("rs.s", w_rs, pol["small"])
        self.w_o = WP("o", lambda: get("wo_ud.weight").T, [("o", cfg.hidden)], pol["o"])
        if self.full:
            self.w_kd = W("kdyn", lambda: get("k_sconv.weight_proj").T, pol["small"])
            self.w_vd = W("vdyn", lambda: get("v_sconv.weight_proj").T, pol["small"])

        d = mesh_device
        f32 = ttnn.float32
        m16 = torch.zeros(self.nh, 32)
        m16[:, :16] = 1
        e16 = torch.zeros(self.nh, 32)
        e16[:, 16] = 1
        e17 = torch.zeros(self.nh, 32)
        e17[:, 17] = 1
        # decode: rows = heads ([1, 1, 24, 32])
        self.m16_d = _dev(m16.reshape(1, 1, 24, 32), d)
        self.e16_d = _dev(e16.reshape(1, 1, 24, 32), d)
        self.e17_d = _dev(e17.reshape(1, 1, 24, 32), d)
        proj = get("rel_logits_proj.proj").float()                              # [16, 1024]
        pa = torch.zeros(32, COLS)
        pa[:16, :RING] = proj
        pa[16, MASKED] = NEG
        pa[17, SINK] = 1.0
        self.p_aug = _dev(pa.reshape(1, 1, 32, COLS), d)                        # (fast/attn_decode reads proj here)
        self.p_aug_t = _dev(pa.T.contiguous(), d, layout=ttnn.ROW_MAJOR_LAYOUT)  # decode embedding table [1056, 32]
        # ---- K | V short conv (2048 channels): taps by lag (lag 0 = current token) ----
        if self.full:
            sk, sv = get("k_sconv.static_w").float(), get("v_sconv.static_w").float()  # [4 lags, 1024]
            self.taps = [_dev(torch.cat([sk[j], sv[j]]).reshape(1, 1, 1, -1), d) for j in range(4)]  # bf16, exact
            e = torch.zeros(64, 4 * 2 * self.kvw)  # [kd | vd] column h*4 + lag -> taps column lag*2048 + (k|v) + h*128
            for kv in range(2):
                for h in range(self.nkv):
                    for lag in range(4):
                        c0 = lag * 2 * self.kvw + kv * self.kvw + h * self.hd
                        e[kv * 32 + h * 4 + lag, c0:c0 + self.hd] = 1
            self.dyn_expand = _dev(e.reshape(1, 1, 64, -1), d)
            self.conv_scale = 1.0 / math.sqrt(2 * cfg.conv_kernel + 1)
        else:
            c = 1.0 / math.sqrt(cfg.conv_kernel + 1)
            kw = get("k_sconv.weight").float().reshape(self.kvw, 4)
            vw = get("v_sconv.weight").float().reshape(self.kvw, 4)
            w = torch.cat([kw, vw], 0)  # [2048, 4]; column j multiplies x[t - 3 + j]
            # fp32 taps with the residual and 1/sqrt(5) folded in: (sum_j w_j x_j + x) * c
            self.taps = [_dev(((w[:, 3 - lag] + (1.0 if lag == 0 else 0.0)) * c).reshape(1, 1, 1, -1), d, f32)
                         for lag in range(4)]
        # prefill K / V front (fast/attn_prefill.KVFront): the taps with rows replicated, lag row tiles [1, 1, 128, 2048]
        if self.full:
            host_taps, tap_dt = [torch.cat([sk[j], sv[j]]) for j in range(4)], ttnn.bfloat16
        else:
            host_taps, tap_dt = [(w[:, 3 - lag] + (1.0 if lag == 0 else 0.0)) * c for lag in range(4)], f32
        self.kv_taps = _dev(torch.stack([t.reshape(1, -1).expand(32, -1) for t in host_taps]).reshape(1, 1, 128, -1),
                            d, tap_dt)
        # ---- caches and conv states per pass slot ----
        kd, vd = (pol["k"], pol["v"]) if self.full else (pol["ring"], pol["ring"])
        z = lambda shape, dt=ttnn.bfloat16: ttnn.zeros(shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=d,  # noqa
                                                       memory_config=DRAM)
        if self.full:
            # single stream: identity block order over max_seq_len; batched: the pool (vLLM block ids)
            if seqs > 1 and kv_blocks is None:
                raise ValueError("Inkling2Attention: several sequence slots need the full-attention pool (kv_blocks)")
            self.nblocks = kv_blocks if kv_blocks is not None else max_seq_len // BLOCK
            shape = [self.nblocks + 1, self.nkv, BLOCK, self.hd]
            if kv_blocks is None:
                self.ident_pt = ttnn.from_torch(torch.arange(self.nblocks, dtype=torch.int32).reshape(1, -1),
                                                dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=d,
                                                memory_config=DRAM)
        else:
            shape = [seqs, self.nkv, RING_ROWS, self.hd]
        self.k_cache = [z(shape, kd) for _ in range(self.n_slots)]
        self.v_cache = [z(shape, vd) for _ in range(self.n_slots)]
        # x[t-3], x[t-2], x[t-1] of [Wk xk | Wv xv] (the stock decode's shift states: one sequence, shared scratch in
        # batched mode, the fused paths never read them)
        self.state = [[z([1, 1, 1, 2 * self.kvw]) for _ in range(3)] for _ in range(self.n_slots)]
        # fused path: row pos % 32 = x[pos] (position-indexed, see the module docstring); slot-major [seqs, ...]
        self.conv_ring = [z([seqs, 1, CONV_RING, 2 * self.kvw]) for _ in range(self.n_slots)]
        self.zero_ring = z([seqs, 1, CONV_RING, 2 * self.kvw])
        self.zero_slot = z([1, 1, CONV_RING, 2 * self.kvw]) if seqs > 1 else self.zero_ring  # reset_slot's source
        self.zero_row = z([1, 1, 1, 2 * self.kvw])
        self._shard = _height_shard(d)
        # the prefill kernels (fast/attn_prefill.FlashPrefill, shared by the model's layers) + this layer's bias table
        # (relative bias / masks / sink, 1.4 MB)
        from .fast.attn_prefill import FlashPrefill, bias_table

        self.flash = flash if flash is not None else FlashPrefill(d, cfg, max_seq_len)
        self.bias_tbl = bias_table(d, proj, sliding=not self.full)
        self.fast = None  # tt/fast/attn_decode.AttnDecodeFast (enable_fast): the fused decode path
        # prefill chunk position (int32 [1, 1, 1, 32] ROW_MAJOR: start, valid, sequence slot): the model sets
        # prefill_pos to its rows glue's tensor (written once per chunk); otherwise prefill writes this module's own
        self.prefill_pos = None
        # batched (pool): the page row of the sequence a prefill chunk belongs to (int32 [1, max_seq_len / 64]; the
        # model's tensor, host-written per sequence); single stream: unused (identity block order)
        self.prefill_pt = None
        self._own_pos = ttnn.from_torch(torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.int32,
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=d, memory_config=DRAM)

    def enable_fast(self, weight_cache: WeightCache):
        """Route decode() through the fused kernels of tt/fast/attn_decode.py (4 launches; reads only tables.cur_pos
        and tables.tau, cost grows with the position, not full_len).  Allocates its persistent device state now
        (call at construction, before any trace capture)."""
        from .fast.attn_decode import AttnDecodeFast

        self.fast = AttnDecodeFast(self, weight_cache)

    # ------------------------------------------------------------------ helpers
    def reset(self):
        """Start a new sequence: zero every pass slot's conv states and conv-input ring (every sequence slot's when
        batched; stale K/V rows are always masked)."""
        for st in self.state:
            for s in st:
                ttnn.copy(self.zero_row, s)
        for r in self.conv_ring:
            ttnn.copy(self.zero_ring, r)

    def reset_slot(self, s: int):
        """Batched serving, a new sequence in sequence slot s: zero s's conv-input rings (every pass; in place, no
        allocation).  Its K / V ring rows and pool blocks need no reset (masked by position until rewritten)."""
        if not 0 <= s < self.seqs:
            raise ValueError(f"reset_slot: slot {s} outside 0..{self.seqs - 1}")
        for r in self.conv_ring:
            ttnn.fill_cache(r, self.zero_slot, s)

    def _ck(self, dtype):
        return CK_HIFI2 if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) else CK_HIFI4

    def _project(self, xq, xk, xv):
        """-> qg [.., 6144] (q | gate), kv_lin [.., 2048] (Wk xk | Wv xv), rs [.., 768], dyn [.., 64] (None on
        sliding layers)."""
        lin = lambda x, w, ck: ttnn.linear(x, w, dtype=ttnn.bfloat16, compute_kernel_config=ck,  # noqa: E731
                                           memory_config=DRAM)
        ck_big = self._ck(self.policy["proj"])
        rs = lin(xq, self.w_rs, CK_HIFI4)
        if self.w_k is None:  # fused [q|gate|k|v]
            y = lin(xq, self.w_big, ck_big)
            T = y.shape[-2]
            qg = ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, 2 * self.qw])
            kv = ttnn.slice(y, [0, 0, 0, 2 * self.qw], [1, 1, T, 2 * self.qw + 2 * self.kvw])
            ttnn.deallocate(y)
            return qg, kv, rs, None
        if self.fused:  # split by the weight precision map: xq is xk is xv
            xk = xv = xq
        qg = lin(xq, self.w_big, ck_big)
        k = lin(xk, self.w_k, ck_big)
        v = lin(xv, self.w_v, ck_big)
        kv = ttnn.concat([k, v], dim=-1)
        free(k, v)
        dyn = None
        if self.full:
            kd = lin(xk, self.w_kd, CK_HIFI4)
            vd = lin(xv, self.w_vd, CK_HIFI4)
            dyn = ttnn.concat([kd, vd], dim=-1)
            free(kd, vd)
        return qg, kv, rs, dyn

    def _conv(self, rows, dyn):
        """Conv of the k | v channels: rows[lag] = x[t - lag] ([.., 2048] bf16).  Sliding: sum_lag rows * taps'
        (fp32 taps incl. residual and scale).  Full: (sum_lag rows * (static_lag + dyn_lag) + x) / 3, dyn =
        per-token expanded taps [.., 8192] (lag-major) or None.  fp32 math, one bf16 rounding."""
        acc = None
        n = 2 * self.kvw
        for lag in (3, 2, 1, 0):
            if dyn is None:
                w = self.taps[lag]
            else:
                T = dyn.shape[-2]
                dl = ttnn.slice(dyn, [0, 0, 0, lag * n], [1, 1, T, (lag + 1) * n])
                w = ttnn.add(self.taps[lag], dl, dtype=ttnn.float32, memory_config=DRAM)
                ttnn.deallocate(dl)
            last = dyn is None and lag == 0
            t = ttnn.mul(rows[lag], w, dtype=ttnn.float32, memory_config=DRAM)
            if dyn is not None:
                ttnn.deallocate(w)
            if acc is None:
                acc = t
            else:
                nxt = ttnn.add(acc, t, dtype=ttnn.bfloat16 if last else ttnn.float32, memory_config=DRAM)
                free(acc, t)
                acc = nxt
        if dyn is None:
            return acc
        ttnn.add(acc, rows[0], output_tensor=acc)
        out = ttnn.mul(acc, self.conv_scale, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(acc)
        return out

    def _dyn(self, dproj):
        """[.., T, 64] per-token taps ([kd | vd], column h*4 + lag) -> [.., T, 8192] (lag-major, exact)."""
        if dproj is None:
            return None
        return ttnn.matmul(dproj, self.dyn_expand, dtype=ttnn.bfloat16, compute_kernel_config=CK_HIFI4,
                           memory_config=DRAM)

    def _q_final(self, q, tau):
        """q bf16 heads [.., 128] (q_scale folded) -> bf16 L2clip(q) [* tau] / sqrt(128)."""
        f = _clip_factor(q, self.cfg.qk_l2_clip, self.hd ** -0.5, tau)
        out = ttnn.mul(q, f, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(f)
        return out

    def _r_aug(self, x, m16, e16, e17, tau):
        """x bf16 [.., 32] per head = [Wr x * r_scale (16) | 0 | sink | 0..] -> bf16 [L2clip(r, 8) [* tau] | 1 | sink
        | 0..]."""
        xr = ttnn.mul(x, m16, memory_config=DRAM)
        f = _clip_factor(xr, self.cfg.rel_l2_clip, 1.0, tau)
        r = ttnn.mul(xr, f, dtype=ttnn.bfloat16, memory_config=DRAM)
        free(xr, f)
        s = ttnn.mul(x, e17, memory_config=DRAM)
        ttnn.add(s, e16, output_tensor=s)
        ttnn.add(r, s, output_tensor=r)
        ttnn.deallocate(s)
        return r

    def _k_final(self, k, memory_config=DRAM):
        f = _clip_factor(k, self.cfg.qk_l2_clip)
        out = ttnn.mul(k, f, dtype=ttnn.bfloat16, memory_config=memory_config)
        ttnn.deallocate(f)
        return out

    def _v_final(self, v, memory_config=DRAM):
        ss = _sumsq(v)
        ttnn.mul(ss, 1.0 / self.hd, output_tensor=ss)
        ttnn.add(ss, self.cfg.eps, output_tensor=ss)
        f = ttnn.rsqrt(ss)
        ttnn.deallocate(ss)
        out = ttnn.mul(v, f, dtype=ttnn.bfloat16, memory_config=memory_config)
        ttnn.deallocate(f)
        return out

    def _gate_out(self, attn_flat, qg):
        """o_proj(attn * sigmoid(gate)): attn_flat [1, 1, R|1, 3072] bf16, qg [1, 1, R, 6144] -> [1, 1, R, 2304]."""
        R = qg.shape[-2]
        g = ttnn.slice(qg, [0, 0, 0, self.qw], [1, 1, R, 2 * self.qw])
        gated = ttnn.mul(g, attn_flat, input_tensor_a_activations=SIGMOID, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(g)
        out = ttnn.linear(gated, self.w_o, dtype=ttnn.bfloat16, compute_kernel_config=self._ck(self.policy["o"]),
                          memory_config=DRAM)
        ttnn.deallocate(gated)
        return out

    # ------------------------------------------------------------------ decode
    def decode(self, xq, xk, xv, cur_pos, pass_idx: int = 0):
        tb = self.tables
        assert tb is not None, "decode needs AttnDecodeTables (constructor kwarg tables=)"
        if self.fast is not None:
            return self.fast(xq, xk, xv, tb, pass_idx)
        if self.batched:
            raise ValueError("Inkling2Attention: batched modules decode through fast.batch (no stock batched decode)")
        slot = pass_idx
        tau = tb.tau if self.full else None
        qg, kv_lin, rs, dyn = self._project(xq, xk, xv)
        row = lambda t, c0, c1: ttnn.slice(t, [0, 0, 0, c0], [1, 1, 1, c1])  # noqa: E731
        # q heads (rows = heads)
        q0 = ttnn.reshape(row(qg, 0, self.qw), [1, 1, self.nh, self.hd])
        q = self._q_final(q0, tau)
        ttnn.deallocate(q0)
        # relative bias + sink -> mask
        r0 = ttnn.reshape(row(rs, 0, self.nh * 32), [1, 1, self.nh, 32])
        ttnn.deallocate(rs)
        r_aug = self._r_aug(r0, self.m16_d, self.e16_d, self.e17_d, tau)
        ttnn.deallocate(r0)
        idx = tb.full_idx if self.full else tb.ring_idx
        emb = ttnn.embedding(idx, self.p_aug_t, layout=ttnn.TILE_LAYOUT)
        emb = ttnn.reshape(emb, [1, 1, idx.shape[-1], 32])
        mask = ttnn.matmul(r_aug, emb, transpose_b=True, dtype=ttnn.bfloat16, compute_kernel_config=CK_HIFI4,
                           memory_config=DRAM)
        free(r_aug, emb)
        # K | V: conv with state, norms, cache writes
        x0 = row(kv_lin, 0, 2 * self.kvw)
        ttnn.deallocate(kv_lin)
        dyn0 = None
        if dyn is not None:
            d0 = row(dyn, 0, 64)
            dyn0 = self._dyn(d0)
            free(d0, dyn)
        st = self.state[slot]
        conv = self._conv([x0, st[2], st[1], st[0]], dyn0)
        ttnn.copy(st[1], st[0])
        ttnn.copy(st[2], st[1])
        ttnn.copy(x0, st[2])
        free(x0, dyn0)
        for i, (fn, cache) in enumerate(((self._k_final, self.k_cache[slot]), (self._v_final, self.v_cache[slot]))):
            part = ttnn.reshape(ttnn.slice(conv, [0, 0, 0, i * self.kvw], [1, 1, 1, (i + 1) * self.kvw]),
                                [1, 1, self.nkv, self.hd])
            hs = fn(part, self._shard)
            ttnn.deallocate(part)
            if self.full:
                ttnn.experimental.paged_update_cache(cache, hs, update_idxs_tensor=cur_pos, page_table=self.ident_pt)
            else:
                ttnn.experimental.paged_update_cache(cache, hs, update_idxs_tensor=tb.ring_pos)
            ttnn.deallocate(hs)
        ttnn.deallocate(conv)
        # attention
        if self.full:
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, self.k_cache[slot], self.v_cache[slot], tb.full_page_table, is_causal=False, attn_mask=mask,
                scale=1.0, program_config=_sdpa_cfg(self.device), compute_kernel_config=CK_HIFI4)
        else:
            out = ttnn.transformer.scaled_dot_product_attention_decode(
                q, self.k_cache[slot], self.v_cache[slot], is_causal=False, attn_mask=mask, scale=1.0,
                program_config=_sdpa_cfg(self.device), compute_kernel_config=CK_HIFI4)
        free(q, mask)
        flat = ttnn.reshape(out, [1, 1, 1, self.qw])
        y = self._gate_out(flat, qg)
        free(out, qg)
        return y

    # ------------------------------------------------------------------ prefill
    def _prefill_pos(self, s: int, length: int):
        """The chunk's position device tensor (int32 [1, 1, 1, 32]: start, valid, sequence slot): the model's shared
        one (`prefill_pos`, written once per chunk by the rows glue) or this module's own, written here (slot 0)."""
        if self.prefill_pos is not None:
            return self.prefill_pos
        host = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        host[0, 0, 0, 0], host[0, 0, 0, 1] = s, length
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                        self._own_pos)
        return self._own_pos

    def _project_rows(self, xq, xk, xv):
        """Prefill projections in the layout the prefill kernels read: qg (its first 192 column tiles = q | gate),
        k_lin / v_lin = (tensor, first column tile) of Wk xk / Wv xv, rs, kd / vd ([.., 32] dyn tap projections on
        full layers, else None), and the tensors to free after the attention."""
        lin = lambda x, w, ck: ttnn.linear(x, w, dtype=ttnn.bfloat16, compute_kernel_config=ck,  # noqa: E731
                                           memory_config=DRAM)
        ck_big = self._ck(self.policy["proj"])
        rs = lin(xq, self.w_rs, CK_HIFI4)
        if self.w_k is None:  # fused [q | gate | k | v]
            y = lin(xq, self.w_big, ck_big)
            c0 = 2 * self.qw // 32
            return y, (y, c0), (y, c0 + self.kvw // 32), rs, None, None, (y, rs)
        if self.fused:  # split by the weight precision map: xq is xk is xv
            xk = xv = xq
        qg = lin(xq, self.w_big, ck_big)
        k = lin(xk, self.w_k, ck_big)
        v = lin(xv, self.w_v, ck_big)
        kd = vd = None
        if self.full:
            kd = lin(xk, self.w_kd, CK_HIFI4)
            vd = lin(xv, self.w_vd, CK_HIFI4)
        return qg, (k, 0), (v, 0), rs, kd, vd, (qg, k, v, rs)

    def prefill(self, xq, xk, xv, start_pos: int, length: int, pass_idx: int = 0):
        """See the module docstring: projections, the K / V front (fast/attn_prefill KVFront: conv, clip, norm, cache
        / aligned-frame writes), the conv state update, the flash attention (q / r finalization, bias, masks, sink,
        gate), the ring write (sliding layers) and o_proj.  Programs depend only on T (start / valid: prefill_pos).
        Batched modules: the sequence slot is word 2 of prefill_pos (its rings / conv rings), full layers address the
        pool through `prefill_pt` (the sequence's page row, int32 [1, max_seq_len / 64], set by the model)."""
        T = xq.shape[-2]
        assert T % 32 == 0 and 0 < length <= T
        if self.full:
            assert start_pos + T <= self.max_seq_len, f"full-layer cache bound {self.max_seq_len} exceeded"
        slot, fl = pass_idx, self.flash
        slotted = self.seqs > 1  # slot-major rings (S = 1: the slot is 0, today's programs)
        pt = None
        if self.full and self.kv_blocks is not None:
            pt = self.prefill_pt
            if pt is None:
                raise ValueError("Inkling2Attention.prefill: the pool needs the sequence's page row (prefill_pt)")
        qg, k_lin, v_lin, rs, kd, vd, owned = self._project_rows(xq, xk, xv)
        pos = self._prefill_pos(start_pos, length)
        kc, vc, ring = self.k_cache[slot], self.v_cache[slot], self.conv_ring[slot]
        kv = fl.kv(k_lin, v_lin, kd, vd, ring, self.kv_taps, kc, vc, pos, sliding=not self.full,
                   nblocks=self.max_seq_len // BLOCK if self.full else 0, slotted=slotted, pt=pt)
        fl.conv_update(k_lin, v_lin, ring, self.state[slot], pos, slotted=slotted)  # after the K / V front's reads
        free(kd, vd)
        if self.full:
            attn = fl(qg, rs, kc, vc, self.bias_tbl, pos, sliding=False, zero_block=self.nblocks, pt=pt)
        else:
            kn, vn = kv
            attn = fl(qg, rs, kc, vc, self.bias_tbl, pos, sliding=True, kn=kn, vn=vn, slotted=slotted)
            fl.ring_write(kn, vn, kc, vc, pos, slotted=slotted)
            free(kn, vn)
        free(*owned)
        y = ttnn.linear(attn, self.w_o, dtype=ttnn.bfloat16, compute_kernel_config=self._ck(self.policy["o"]),
                        memory_config=DRAM)
        ttnn.deallocate(attn)
        return y
