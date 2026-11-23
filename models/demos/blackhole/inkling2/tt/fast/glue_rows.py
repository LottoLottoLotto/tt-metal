# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused glue with tokens as tile rows (GlueFast2): the prefill chunks' and the speculative verify's glue (every
non-attention / non-MoE op of the 48 applications) as ~2 generic_op programs per application
(kernels/fast/glue_rows.cpp), replacing layers.py's ListStates prefill path (~100 ttnn ops / application).

Programs (one kernel source, variants as compile-time args; NR = row tiles is a runtime argument):
  PRE  [post-MLP conv of the previous application + residual -> h]  -> block attention residual (1 / 3 queries over
       the block states and h; boundary layers append h)  -> input norm(s) -> xq (xk, xv)
  MID  post-attention conv + residual -> r (FP32) -> MLP block residual -> post_attention_layernorm -> z
  END  post-MLP conv + residual -> h (end of a loop pass)
  FIN  post-MLP conv + residual -> final block residual -> norm (the LM head input)
Message gates (passes 2 / 3 of the loop layers, 4 per application): the PRE / MID mix program also builds the gates'
messages m_k = s1 + w_k (s0 - s1) (the pass's per-token message weights w and states s0, s1 come from 4 ttnn ops per
pass) and writes [bf16(norm O) | bf16(norm m_k)] into one CAT tensor; ONE ttnn matmul with the layer's block
down-projection gives every gate's g = bf16(down(cat(norm x, norm m))); a GATE program applies
y = (1 - tanh(g Wf)) O + tanh(g Wg) m and the input / post-attention norm.

State: block states are the raw BF16 layer inputs of the boundary layers (X[i], the reference's block_states) and their
per-token 1 / rms (SC, FP32 [32, Tp]: row i = state i, column t = token); conv inputs live in position-indexed rings
(ring(layer, kind, p): FP32 [S, 1, 32, H], S = model.max_num_seqs sequence slots, slot s = tile pages s * 72 ..; row
pos % 32 of a slot = that sequence's conv input at absolute position pos) shared with the decode programs: a verify of
R rows writes its R rows at their positions and the next step simply reads positions < its own (commit = advance the
position).  A prefill chunk writes its last WR rows into the slot of pos word 2 (set_pos(start, valid, slot); S = 1:
always slot 0, today's tensors and binaries; S > 1: the programs are compiled with GR_SLOTS and read the slot word).

Numerics: the reference's FP32 normed states / partials (logits = s_i q . X_i with FP32 s_i, coefficients W s), dot
products / norms / mixes from exact BF16 products with hi / lo operands, FP32 accumulation; conv row shifts by
one-hot matmuls (exact).  BF16 outputs RNE-rounded like the reference's casts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import ttnn

from .common import KERNELS, f32_bits

KERNEL = KERNELS / "glue_rows.cpp"
DRAM = ttnn.DRAM_MEMORY_CONFIG
BF16, F32 = ttnn.bfloat16, ttnn.float32
HT = 72
WR = 8  # ring rows written per program (verify rows <= 8; a prefill chunk's last rows)
RING = 32
GATE_NAMES = (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp"))  # message kind -> checkpoint controller name

CB = dict(IN=0, PREV=1, BASE=2, TAP=3, X=4, Q=5, GAM=6, CONST=7, G=8, HV=9, RF=10, VH=11, VL=12, PF=13, PS=14, RAC=15,
          RAS=16, RB=17, DG=18, OH=19, OL=20, OUT=21, OO=22, SC=23, MF=24, MS=25, SV=26, STG=27, RSCR=28, ACK=29,
          W=30, WS=31, CSEL=32, WC=33, S1=34, DD=35, MH=36, ML=37, MO=38, RAM=39, NM=40, GG=41, WF=42, WG=43, XO=44,
          MI=45, MF2=46, MS2=47)
FP32_CBS = ("TAP", "PF", "RAS", "RB", "SC", "MF", "W", "WC", "S1", "DD", "RAM", "XO", "MI", "MF2")
K_SEL = 7  # K_I, K_S (3), K_SP (3), K_SEL ... (kernel order)


@dataclass(frozen=True)
class RVariant:
    pro: int         # 0 h given | 1 h = r + conv(x) | 2 r = [h +] conv(x)
    bound: int = 0   # PRO 2: r = conv(x)
    stop: int = 0    # END
    nq: int = 0      # block-residual queries (0: layer 0 / END); GATE: = ng
    ns: int = 0      # stored block states
    append: int = 0  # the partial h becomes state ns
    outm: int = 0    # 0 normed outputs | 2 O (FP32) + CAT [bf16(norm O) | bf16(norm m_k)] (message gates follow)
    nout: int = 1    # OUTM 0: output tensors (gamma rows)
    ng: int = 0      # OUTM 2: messages of ng gates | GATE: ng gates
    gate: int = 0    # the gate program
    gsame: int = 0   # GATE: all gates read O_0 (sliding layers)

    def ckey(self):
        """The constants' key (kernel CB_CONST layout)."""
        return (self.ns, self.nq, self.ng if self.outm == 2 else 0, self.gate, self.stop)


class RowsLayout:
    """NGRP groups of CG = GW x GH cores stacked at the grid origin (group g: rows g GH ..); core c = y GW + x of a
    group owns hidden column tiles [c TC, c TC + TC); group g handles row tiles g, g + NGRP, ...; the group's first
    core is the master of its rounds."""

    def __init__(self, device, gw: int, gh: int, ngrp: int | None = None):
        self.CG = gw * gh
        if HT % self.CG or self.CG > 32:
            raise ValueError(f"group of {self.CG} cores: must divide 72, at most 32")
        self.TC = HT // self.CG
        self.PF = 32 // self.CG
        g = device.compute_with_storage_grid_size()
        if gw > g.x:
            raise ValueError("group wider than the grid")
        self.NGRP = min(g.y // gh, ngrp or 99)
        self.gw, self.gh = gw, gh
        ranges, self.rt = [], []
        for grp in range(self.NGRP):
            y0 = grp * gh
            lo = ttnn.CoreCoord(0, y0)
            hi = ttnn.CoreCoord(gw - 1, y0 + gh - 1)
            ranges.append(ttnn.CoreRange(lo, hi))
            p0 = device.worker_core_from_logical_core(lo)
            p1 = device.worker_core_from_logical_core(hi)
            for y in range(gh):
                for x in range(gw):
                    self.rt.append((ttnn.CoreCoord(x, y0 + y), [grp, y * gw + x, p0.x, p0.y, p0.x, p0.y, p1.x, p1.y]))
        self.cores = ttnn.CoreRangeSet(ranges)
        self.key = (gw, gh, self.NGRP)

    def slots(self, ns: int, nq: int, nmg: int = 0, gate: int = 0, stop: int = 0):
        """(SA_L, SPQ, SA_M, NSLOT_A, NSLOT_B, NSEL) as the kernel computes them."""
        nlt = ns + 1 if nq and not gate else 0
        nsel = nlt * nq
        spq = -(-2 // self.PF)
        sa_l = -(-2 * nsel // self.PF)
        sa_m = sa_l + spq
        nsa = 0 if (gate or stop) else sa_m + nmg * spq
        return sa_l, spq, sa_m, nsa, nq * spq if nq else 0, nsel


def _consts(L: RowsLayout, ns: int, nq: int, nmg: int, gate: int, stop: int) -> torch.Tensor:
    """Constant tiles in the kernel's CB_CONST order -> host [1, 1, 32, 32 n] BF16."""
    sa_l, spq, sa_m, nsa, nsb, nsel = L.slots(ns, nq, nmg, gate, stop)
    tiles = [torch.eye(32)]                           # K_I
    for s in (1, 2, 3):                               # K_S: x_{t-s} from this tile
        m = torch.zeros(32, 32)
        for r in range(s, 32):
            m[r, r - s] = 1
        tiles.append(m)
    for s in (1, 2, 3):                               # K_SP: from the previous tile's last rows
        m = torch.zeros(32, 32)
        for r in range(s):
            m[r, 32 + r - s] = 1
        tiles.append(m)
    rho = [4 * (u // nq) + u % nq for u in range(nsel)]
    for u in range(nsel):                             # K_SEL: row broadcast of row rho
        m = torch.zeros(32, 32)
        m[:, rho[u]] = 1
        tiles.append(m)
    repx, repv, sumq, mask = (torch.zeros(32, 32) for _ in range(4))
    if nsel:
        for q in range(nq):
            for i in range(ns):
                repx[4 * i + q, i] = 1
            repv[4 * ns + q, 0] = 1
            for i in range(ns + 1):
                mask[4 * i + q, :] = 1
                for i2 in range(ns + 1):
                    sumq[4 * i2 + q, 4 * i + q] = 1
    tiles += [repx, repv, sumq, mask, 1 - mask]
    CG, PF = L.CG, L.PF
    for s in range(nsa):                              # K_EA: logits rows rho; sum v^2 / |m_k|^2 -> row 0
        m = torch.zeros(32, 32)
        for mm in range(PF):
            if s < sa_l:
                e = s * PF + mm
                if e >= 2 * nsel:
                    break
                row = rho[e // 2]
            else:
                if ((s - sa_l) % spq) * PF + mm >= 2:
                    break
                row = 0
            m[row, mm * CG: mm * CG + CG] = 1
        tiles.append(m)
    for s in range(nsb):                              # K_EB: DEST q, row 0
        m = torch.zeros(32, 32)
        for mm in range(PF):
            if (s % spq) * PF + mm >= 2:
                break
            m[0, mm * CG: mm * CG + CG] = 1
        tiles.append(m)
    k = torch.stack(tiles)
    return k.permute(1, 0, 2).reshape(1, 1, 32, 32 * len(tiles)).to(torch.bfloat16)


def _cb(index, dtype, tiles, cores):
    page = 4096 if dtype == F32 else 2048
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    return ttnn.CBDescriptor(total_size=max(int(tiles), 1) * page, core_ranges=cores,
                             format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])


ARGS = ["in", "base", "ring", "tap0", "tap1", "tap2", "tap3", "x0", "x1", "x2", "x3", "sc", "q", "gam0", "gam1",
        "gam2", "o0", "o1", "o2", "oh", "oo0", "oo1", "oo2", "pos", "const"]  # then NR (index 25)
ARGS2 = ["s1", "d", "w", "csel"]                                              # 26..29, then gc0..2 (30..32)
ARGS3 = ["cat", "m0", "m1", "m2", "g", "wf0", "wf1", "wf2", "wg0", "wg1", "wg2"]  # 33..43


class RowsProgram:
    """One call site: variant + layout; run(nr, gc, **tensors) (descriptor built once, the address arguments patched
    when a tensor moves: generic_op's program cache keys on everything but the argument values).  slots: the rings
    are slot-major (S > 1): compiled with GR_SLOTS (the programs read the slot from pos word 2)."""

    def __init__(self, layout: RowsLayout, v: RVariant, hidden: int, eps: float, const, slots: bool = False):
        self.L, self.v = layout, v
        self.hidden, self.eps = hidden, eps
        self.const = const
        self.slots = slots
        self.desc, self.args = None, None

    def _build(self, args):
        L, v = self.L, self.v
        TC = L.TC
        mix = v.nq > 0
        msg = v.outm == 2
        base_f32 = int(v.pro == 1 and not v.gate)
        named = dict(TC=TC, CG=L.CG, NGRP=L.NGRP, PRO=v.pro, BOUND=v.bound, STOP=v.stop, NQ=v.nq, NS=v.ns,
                     APPEND=v.append, OUTM=v.outm, NOUT=v.nout, PF=L.PF, WR=WR, INVH=f32_bits(1.0 / self.hidden),
                     EPS=f32_bits(self.eps), BASE_F32=base_f32, NG=v.ng, GATE=v.gate, GSAME=v.gsame)
        rt = ttnn.RuntimeArgs()
        for core, a in L.rt:
            rt[core.x][core.y] = a
        dm = dict(kernel_source=str(KERNEL), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                  core_ranges=L.cores, named_compile_time_args=list(named.items()), common_runtime_args=args)
        if self.slots:
            dm["defines"] = [("GR_SLOTS", "1")]
        fp32_cbs = [CB[k] for k in FP32_CBS] + ([CB["BASE"]] if base_f32 else [])
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, dst_full_sync_en=True)
        unpack = [ttnn.UnpackToDestMode.Default] * 64
        for i in fp32_cbs:
            unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode.extend(unpack)
        kernels = [
            ttnn.KernelDescriptor(**dm, runtime_args=rt, config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default)),
            ttnn.KernelDescriptor(**dm, runtime_args=rt, config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default)),
            ttnn.KernelDescriptor(**dm, runtime_args=rt, config=cfg),
        ]
        ns, nq, ng = v.ns, v.nq, v.ng
        nmg = ng if msg else 0
        _, _, _, nsa, nsb, nsel = L.slots(*v.ckey())
        nlt = nsel // nq if nsel else 0
        npa = nlt + 1 + nmg
        nconst = K_SEL + nsel + 5 + nsa + nsb
        no = (v.nout if v.outm == 0 else nq) if mix else 1
        rscr = 64 + 6 * TC * 64 + 2 * WR * TC * 128
        g_ = 1 if v.gate else 0
        # every FIFO CB must be sized to a multiple of the pages one row tile pushes (a reservation that straddles the
        # CB end is written / read past it): the GATE program has no round A, so its PF / PS / MF / MS hold exactly its
        # round B (NQ = 3 gates: 3 pages into a 7-page MF overwrote CB_MS on every group's 3rd row tile, i.e. Tp > 640)
        pf, ps = (nq, 2 * nq) if v.gate else (npa + nq, 2 * npa + 2 * nq)
        mf, ms = (nq, 2 * nq) if v.gate else ((4 + nq, 6 + 2 * nq) if mix else (2, 2))
        sizes = dict(IN=(BF16, TC), PREV=(BF16, TC), BASE=(F32 if base_f32 else BF16, TC), TAP=(F32, 4 * TC),
                     X=(BF16, max(ns, 1) * TC), Q=(BF16, 2 * TC), GAM=(BF16, v.nout * TC), CONST=(BF16, nconst),
                     G=(BF16, nsa + nsb), HV=(BF16, TC), RF=(F32, TC), VH=(BF16, TC), VL=(BF16, TC),
                     PF=(F32, pf), PS=(BF16, ps), RAC=(BF16, 2), RAS=(F32, 1),
                     RB=(F32, max(nq, 1)), DG=(BF16, max(2 * nsel, 1)), OH=(BF16, max(nq * TC, 1)),
                     OL=(BF16, max(nq * TC, 1)), OUT=(BF16, no * TC), OO=(F32, nq * TC if v.outm else 1), SC=(F32, 1),
                     MF=(F32, mf), MS=(BF16, ms), SV=(F32, 1), STG=(BF16, 1),
                     RSCR=(BF16, -(-rscr // 2048)), ACK=(BF16, 1),
                     W=(F32, 2 if msg else 1), WS=(BF16, 2 if msg else 1), CSEL=(BF16, nmg), WC=(F32, nmg),
                     S1=(F32, TC if msg else 1), DD=(F32, TC if msg else 1), MH=(BF16, nmg * TC), ML=(BF16, nmg * TC),
                     MO=(F32, TC if msg else 1), RAM=(F32, nmg), NM=(BF16, TC if msg else 1), GG=(BF16, 2 * ng * g_),
                     WF=(BF16, 2 * ng * TC * g_), WG=(BF16, 2 * ng * TC * g_),
                     XO=(F32, (1 if v.gsame else ng) * TC * g_), MI=(F32, ng * TC * g_), MF2=(F32, nmg),
                     MS2=(BF16, 2 * nmg))
        cbs = [_cb(CB[name], dt, n, L.cores) for name, (dt, n) in sizes.items()]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=L.cores, initial_value=0) for i in (0, 1)]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)

    def run(self, nr: int, gc=(), **t):
        t["const"] = self.const
        addr = lambda k: int(t[k].buffer_address()) if t.get(k) is not None else 0  # noqa: E731
        gcs = list(gc) + [0] * (3 - len(gc))
        args = [addr(k) for k in ARGS] + [nr] + [addr(k) for k in ARGS2] + gcs + [addr(k) for k in ARGS3]
        if self.desc is None:
            self.desc = self._build(args)
        elif args != self.args:
            for k in self.desc.kernels:
                k.common_runtime_args = args
        self.args = args
        tensors = [x for x in t.values() if x is not None]
        ttnn.generic_op(tensors, self.desc)


def _rows_rep(device, v: torch.Tensor, rows: int = 32, dtype=BF16):
    """[1, 1, rows, H] tensor with every row = v."""
    x = v.float().reshape(1, 1, 1, -1).expand(1, 1, rows, -1).contiguous()
    return ttnn.from_torch(x.to(torch.bfloat16 if dtype == BF16 else torch.float32), dtype=dtype,
                           layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def rows_query(device, q: torch.Tensor):
    """Block residual queries [nq, H] (FP32) -> [1, 1, 64, H] BF16: rows q = hi, rows 32 + q = lo of q H^-0.5."""
    q = q.double().reshape(-1, q.shape[-1])
    H = q.shape[-1]
    qs = q * H ** -0.5
    hi = qs.to(torch.bfloat16)
    lo = (qs - hi.double()).to(torch.bfloat16)
    x = torch.zeros(1, 1, 64, H, dtype=torch.bfloat16)
    x[0, 0, : q.shape[0]] = hi
    x[0, 0, 32: 32 + q.shape[0]] = lo
    return ttnn.from_torch(x, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def _empty(device, rows: int, cols: int, dtype=BF16):
    """Uninitialized device tensor (no host write: legal inside a trace capture; every buffer of forward() is
    written before it is read)."""
    return ttnn.empty([1, 1, rows, cols], dtype, ttnn.TILE_LAYOUT, device, DRAM)


def _zeros(device, rows: int, cols: int, dtype=BF16):
    return ttnn.from_torch(torch.zeros(1, 1, rows, cols, dtype=torch.bfloat16 if dtype == BF16 else torch.float32),
                           dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


class _Msgs:
    """Per-token message inputs of one pass (FP32): s1 = norm(the previous pass's last hidden), d = norm(the previous
    pass's state 2) - s1, w [Tp, 64] (column 4 j + k = weight of s0 in message (loop layer j, kind k))."""

    def __init__(self, mr, x2, h_prev):
        from ..layers import unorm

        s0 = unorm(x2)
        self.s1 = unorm(h_prev)
        self.w, self.d = mr.prepare(s0, self.s1)
        ttnn.deallocate(s0)

    def free(self):
        for t in (self.w, self.d, self.s1):
            ttnn.deallocate(t)


class RowsGlue:
    """The fused glue of an Inkling2Model for row-batched tokens (prefill chunks, verify rows).  forward(x, attend, moe,
    hook) runs the 48 applications on x = embed_norm(embeddings) [1, 1, Tp, H] BF16 (rows >= valid are padding;
    set_pos(start, valid, slot) first) and returns the final normed hidden [1, 1, Tp, H] BF16.  The conv rings are
    slot-major over the model's max_num_seqs sequence slots (reset_slot(s): a new sequence in slot s).  Construct
    after the model's glue modules; everything persistent is allocated here (trace safety)."""

    def __init__(self, model):
        from ..common import CK_HIFI4

        self.m = model
        self.ck = CK_HIFI4
        cfg = self.cfg = model.cfg
        dev = self.dev = model.device
        H = self.H = cfg.hidden
        src = model.src.tensor
        g = dev.compute_with_storage_grid_size()
        spec = os.environ.get("INK_GLUE_ROWS", "")
        self.layouts = {
            "one": RowsLayout(dev, min(8, g.x), 3, ngrp=1),     # 24 cores, 1 group (verify / short prompts)
            "small": RowsLayout(dev, min(9, g.x), 2),          # 18-core groups (TC 4)
            "large": RowsLayout(dev, min(9, g.x), 1),          # 9-core groups (TC 8)
        }
        if spec:
            gw, gh = (int(v) for v in spec.split("x"))
            self.layouts["fixed"] = RowsLayout(dev, gw, gh)
        self.boundaries = [b for b in range(cfg.n_layers) if cfg.is_boundary(b)]
        rep = lambda v: _rows_rep(dev, v)  # noqa: E731
        self.q_attn, self.q_mlp, self.g_pre, self.g_mid = {}, {}, {}, {}
        for layer in range(cfg.n_layers):
            p = f"model.llm.layers.{layer}"
            if layer:
                self.q_attn[layer] = rows_query(dev, src(f"model.llm.attn_res.attn_queries.{layer - 1}"))
            self.q_mlp[layer] = rows_query(dev, src(f"model.llm.attn_res.mlp_queries.{layer}"))
            if cfg.is_full(layer):
                self.g_pre[layer] = [rep(src(f"{p}.{k}_input_norm.weight")) for k in "qkv"]
            else:
                self.g_pre[layer] = [rep(src(f"{p}.attn_norm.weight"))]
            self.g_mid[layer] = [rep(src(f"{p}.mlp_norm.weight"))]
        self.q_fin = rows_query(dev, src("model.llm.attn_res.output_query"))
        self.g_fin = [rep(src("model.llm.norm.weight"))]
        # message gates: [x half; message halves] block down projections per loop layer (CAT x W -> every gate's g)
        self.wd_pre, self.wd_mid = {}, {}
        for layer in range(cfg.loop_start, cfg.loop_end):
            self.wd_pre[layer], self.wd_mid[layer] = self._down_blocks(layer)
        sel = torch.zeros(32, 32, 32)
        for c in range(32):
            sel[c, c, 0] = 1
        self.csel = ttnn.from_torch(sel.permute(1, 0, 2).reshape(1, 1, 32, 1024).to(torch.bfloat16), dtype=BF16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        self.S = getattr(model, "max_num_seqs", 1)  # sequence slots (batched serving; 1 = single stream)
        self.rings = {}
        for layer in range(cfg.n_layers):
            for kind in ("attn", "mlp"):
                for p in range(cfg.passes(layer)):
                    self.rings[layer, kind, p] = (_zeros(dev, RING, H, F32) if self.S == 1 else ttnn.from_torch(
                        torch.zeros(self.S, 1, RING, H), dtype=F32, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=DRAM))
        self._ring0 = _zeros(dev, RING, H, F32) if self.S > 1 else None  # reset_slot's source (one slot of zeros)
        self.pos = ttnn.from_torch(torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.int32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM)
        self._consts = {}
        for L in self.layouts.values():  # every program's constants now (trace safety)
            for v in self._all_variants():
                self.consts(L, v)
        self._progs = {}

    def _down_blocks(self, layer):
        """CAT x W: PRE rows [nx (1 or 3 blocks); nm_q; nm_k; nm_v] -> g_q | g_k | g_v; MID [nx; nm_mlp] -> g_mlp."""
        cfg, H = self.cfg, self.H
        src = self.m.src.tensor
        down = {kind: src(f"model.llm.layers.{layer}.controllers.{ck}.down_weight").float()
                for kind, ck in GATE_NAMES}  # [64, 2H]: [:, :H] x half, [:, H:] message half
        full = cfg.is_full(layer)

        def pre():
            nx = 3 if full else 1
            w = torch.zeros((nx + 3) * H, 192)
            for k, kind in enumerate("qkv"):
                r = k if full else 0
                w[r * H:(r + 1) * H, 64 * k:64 * k + 64] = down[kind][:, :H].T
                w[(nx + k) * H:(nx + k + 1) * H, 64 * k:64 * k + 64] = down[kind][:, H:].T
            return w.reshape(1, 1, *w.shape)

        def mid():
            return down["mlp"].T.reshape(1, 1, 2 * H, 64)

        c = self.m.cache
        return (c.get(f"rows_gate_down_pre_{layer}", pre, BF16), c.get(f"rows_gate_down_mid_{layer}", mid, BF16))

    def _gate_g(self, cat, w):
        """g = bf16(CAT x W) [1, 1, Tp, cols]."""
        return ttnn.matmul(cat, w, dtype=BF16, compute_kernel_config=self.ck, memory_config=DRAM)

    def _all_variants(self):
        """Every program variant forward() runs (for the persistent constants)."""
        cfg = self.cfg
        out = set()
        for layer, p in cfg.applications():
            gates = cfg.is_loop(layer) and p > 0
            full = cfg.is_full(layer)
            nq = 0 if layer == 0 else (3 if full else 1)
            ns = self.ns(layer)
            for pro in (0, 1):
                out.add(RVariant(pro=pro, nq=nq, ns=ns, outm=2 if gates else 0, ng=3 if gates else 0).ckey())
            out.add(RVariant(pro=2, nq=1, ns=ns + int(cfg.is_boundary(layer)), outm=2 if gates else 0,
                             ng=1 if gates else 0).ckey())
            if gates:
                out.add(RVariant(pro=0, nq=3, ng=3, gate=1).ckey())
                out.add(RVariant(pro=0, nq=1, ng=1, gate=1).ckey())
        out.add(RVariant(pro=1, stop=1).ckey())
        out.add(RVariant(pro=1, nq=1, ns=len(self.boundaries)).ckey())
        return [RVariant(pro=0, ns=k[0], nq=k[1], ng=k[2], outm=2 if k[2] else 0, gate=k[3], stop=k[4]) for k in out]

    # ------------------------------------------------------------------ helpers
    def consts(self, L: RowsLayout, v: RVariant):
        key = (L.key,) + v.ckey()
        if key not in self._consts:
            self._consts[key] = ttnn.from_torch(_consts(L, *v.ckey()), dtype=BF16, layout=ttnn.TILE_LAYOUT,
                                                device=self.dev, memory_config=DRAM)
        return self._consts[key]

    def layout(self, nr: int) -> RowsLayout:
        if "fixed" in self.layouts:
            return self.layouts["fixed"]
        if nr <= 1:
            return self.layouts["one"]
        return self.layouts["small" if nr <= self.layouts["small"].NGRP else "large"]

    def ring(self, layer: int, kind: str, p: int):
        """The conv ring of (layer, kind, pass p): FP32 [S, 1, 32, H] (slot s = pages s * 72 ..)."""
        return self.rings[layer, kind, p]

    def reset(self):
        for t in self.rings.values():
            ttnn.mul(t, 0.0, output_tensor=t)

    def reset_slot(self, s: int):
        """Zero slot s's pages of every conv ring (a new sequence in slot s; the other slots untouched)."""
        if not 0 <= s < self.S:
            raise ValueError(f"RowsGlue.reset_slot: slot {s} not in 0..{self.S - 1}")
        if self.S == 1:
            return self.reset()
        for t in self.rings.values():
            ttnn.fill_cache(t, self._ring0, s)

    def set_pos(self, start: int, valid: int, slot: int = 0):
        """Host write of the rows' start position, valid row count and sequence slot (words 0 / 1 / 2 of self.pos;
        before the programs run / a trace replays)."""
        if not 0 <= slot < self.S:
            raise ValueError(f"RowsGlue.set_pos: slot {slot} not in 0..{self.S - 1}")
        host = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        host[0, 0, 0, 0], host[0, 0, 0, 1], host[0, 0, 0, 2] = int(start), int(valid), int(slot)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                        self.pos)

    def ns(self, layer: int) -> int:
        return sum(b < layer for b in self.boundaries)

    def _prog(self, key, L, v: RVariant):
        k = (key, L.key)
        if k not in self._progs:
            self._progs[k] = RowsProgram(L, v, self.H, self.cfg.eps, self.consts(L, v), slots=self.S > 1)
        return self._progs[k]

    def _taps(self, layer: int, kind: str):
        conv = self.m.glue[layer].attn_conv if kind == "attn" else self.m.glue[layer].mlp_conv
        return {f"tap{i}": t for i, t in enumerate(conv.taps32)}

    def _gate_cols(self, layer, kinds):
        j = layer - self.cfg.loop_start
        return [4 * j + ("q", "k", "v", "mlp").index(k) for k in kinds]

    # ------------------------------------------------------------------ programs
    def pre(self, L, nr, i, layer, p, X, SC, h, mo=None, r=None, prev=None, outs=(), msgs=None, oo=(), cat=None,
            mt=()):
        """PRE of application i.  Direct (h given, mo None) or h <- bf16(r + conv(mo)) with prev = (layer, slot) of
        the previous application.  outs: the normed outputs (1 or 3); with msgs (message gates follow): oo = the FP32
        mixed rows, cat = the CAT tensor, mt = the 3 gates' FP32 messages."""
        cfg = self.cfg
        full = cfg.is_full(layer)
        nq = 0 if layer == 0 else (3 if full else 1)
        ns = self.ns(layer)
        gates = msgs is not None
        v = RVariant(pro=0 if mo is None else 1, nq=nq, ns=ns, append=int(cfg.is_boundary(layer)),
                     outm=2 if gates else 0, nout=1 if gates else len(outs), ng=3 if gates else 0)
        t = dict(base=h if mo is None else r, sc=SC, q=self.q_attn.get(layer), pos=self.pos)
        if mo is not None:
            t.update(self._taps(prev[0], "mlp"))
            t.update({"in": mo, "ring": self.ring(prev[0], "mlp", prev[1]), "oh": h})
        for k in range(ns):
            t[f"x{k}"] = X[k]
        gc = ()
        if gates:
            t.update({"s1": msgs.s1, "d": msgs.d, "w": msgs.w, "csel": self.csel, "cat": cat})
            t.update({f"oo{k}": o for k, o in enumerate(oo)})
            t.update({f"m{k}": o for k, o in enumerate(mt)})
            gc = self._gate_cols(layer, "qkv")
        else:
            for k, gm in enumerate(self.g_pre[layer][: len(outs)]):
                t[f"gam{k}"] = gm
            for k, o in enumerate(outs):
                t[f"o{k}"] = o
        self._prog(("pre", i), L, v).run(nr, gc, **t)

    def mid(self, L, nr, i, layer, p, X, SC, a, h, r, out=None, msgs=None, oo=None, cat=None, mt=None):
        cfg = self.cfg
        bound = int(cfg.is_boundary(layer))
        ns = self.ns(layer) + bound
        gates = msgs is not None
        v = RVariant(pro=2, bound=bound, nq=1, ns=ns, outm=2 if gates else 0, ng=1 if gates else 0)
        t = dict(self._taps(layer, "attn"))
        t.update({"in": a, "base": None if bound else h, "ring": self.ring(layer, "attn", p), "sc": SC,
                  "q": self.q_mlp[layer], "oh": r, "pos": self.pos})
        for k in range(ns):
            t[f"x{k}"] = X[k]
        gc = ()
        if gates:
            t.update({"s1": msgs.s1, "d": msgs.d, "w": msgs.w, "csel": self.csel, "cat": cat, "oo0": oo, "m0": mt})
            gc = self._gate_cols(layer, ("mlp",))
        else:
            t.update({"gam0": self.g_mid[layer][0], "o0": out})
        self._prog(("mid", i), L, v).run(nr, gc, **t)

    def gate(self, L, nr, key, layer, kinds, oo, mt, g, outs, gams):
        """GATE program: outs[k] = bf16(norm(y_k) gamma_k), y_k = (1 - tanh(g_k Wf_k)) oo[src] + tanh(g_k Wg_k) mt[k]."""
        ng = len(kinds)
        gsame = int(len(oo) == 1)
        v = RVariant(pro=0, nq=ng, ng=ng, gate=1, gsame=gsame, nout=ng)
        G = self.m.glue[layer]
        t = {"g": g, "pos": self.pos}
        for k, kind in enumerate(kinds):
            t[f"wf{k}"], t[f"wg{k}"] = G.gates[kind].wf, G.gates[kind].wg
            t[f"m{k}"], t[f"o{k}"], t[f"gam{k}"] = mt[k], outs[k], gams[k]
        for k, o in enumerate(oo):
            t[f"oo{k}"] = o
        self._prog(("gate", key), L, v).run(nr, (), **t)

    def end(self, L, nr, key, prev, mo, r, h_out):
        v = RVariant(pro=1, stop=1)
        t = dict(self._taps(prev[0], "mlp"))
        t.update({"in": mo, "base": r, "ring": self.ring(prev[0], "mlp", prev[1]), "oh": h_out, "pos": self.pos})
        self._prog(("end", key), L, v).run(nr, (), **t)

    def final(self, L, nr, prev, mo, r, X, SC, h_out, out):
        v = RVariant(pro=1, nq=1, ns=len(self.boundaries))
        t = dict(self._taps(prev[0], "mlp"))
        t.update({"in": mo, "base": r, "ring": self.ring(prev[0], "mlp", prev[1]), "oh": h_out, "sc": SC,
                  "q": self.q_fin, "gam0": self.g_fin[0], "o0": out, "pos": self.pos})
        for k in range(len(self.boundaries)):
            t[f"x{k}"] = X[k]
        self._prog(("fin",), L, v).run(nr, (), **t)

    # ------------------------------------------------------------------ gate applications
    def pre_gated(self, L, nr, i, layer, p, X, SC, h, msgs, outs, **kw):
        """PRE with the q / k / v message gates: mix + messages -> CAT -> g (one matmul) -> GATE -> outs."""
        dev, Tp, H = self.dev, X[0].shape[-2], self.H
        full = self.cfg.is_full(layer)
        nq = 3 if full else 1
        oo = [_empty(dev, Tp, H, F32) for _ in range(nq)]
        mt = [_empty(dev, Tp, H, F32) for _ in range(3)]
        cat = _empty(dev, Tp, (nq + 3) * H)
        self.pre(L, nr, i, layer, p, X, SC, h, msgs=msgs, oo=oo, cat=cat, mt=mt, **kw)
        g = self._gate_g(cat, self.wd_pre[layer])
        ttnn.deallocate(cat)
        gams = self.g_pre[layer] if full else self.g_pre[layer] * 3
        self.gate(L, nr, ("pre", i), layer, "qkv", oo, mt, g, outs, gams)
        for t in oo + mt + [g]:
            ttnn.deallocate(t)

    def mid_gated(self, L, nr, i, layer, p, X, SC, a, h, r, msgs, out):
        dev, Tp, H = self.dev, X[0].shape[-2], self.H
        oo, mt = _empty(dev, Tp, H, F32), _empty(dev, Tp, H, F32)
        cat = _empty(dev, Tp, 2 * H)
        self.mid(L, nr, i, layer, p, X, SC, a, h, r, msgs=msgs, oo=oo, cat=cat, mt=mt)
        g = self._gate_g(cat, self.wd_mid[layer])
        ttnn.deallocate(cat)
        self.gate(L, nr, ("mid", i), layer, ("mlp",), [oo], [mt], g, [out], self.g_mid[layer])
        for t in (oo, mt, g):
            ttnn.deallocate(t)

    # ------------------------------------------------------------------ the 48 applications
    def forward(self, x, attend, moe, hook=None):
        """x: embed_norm output [1, 1, Tp, H] BF16 (not consumed; rows >= the valid count are padding; set_pos(start,
        valid) first).  attend(layer, p, xq, xk, xv) -> [1, 1, Tp, H] BF16; moe(layer, z) -> [1, 1, Tp, H] BF16
        (both consumed here); hook(app, layer, p, h) gets every application's output.  Returns the final normed
        hidden [1, 1, Tp, H] BF16 (caller frees)."""
        m, cfg, dev, H = self.m, self.cfg, self.dev, self.H
        Tp = x.shape[-2]
        nr = Tp // 32
        L = self.layout(nr)
        zeros = lambda dt=BF16: _empty(dev, Tp, H, dt)  # noqa: E731
        SC = _empty(dev, 32, Tp, F32)
        r = zeros(F32)
        X = [x] + [zeros() for _ in self.boundaries[1:]]
        hbuf = [zeros(), zeros()]
        xo = [zeros() for _ in range(3)]
        z, h_end, fin = zeros(), zeros(), zeros()
        own = [SC, r, *X[1:], *hbuf, *xo, z, h_end]
        prev, mo, msgs, h_in = None, None, None, x
        apps = cfg.applications()
        for i, (layer, p) in enumerate(apps):
            if layer == cfg.loop_start and p > 0:  # pass restart: END of the previous pass, its messages
                self.end(L, nr, p, prev, mo, r, h_end)
                ttnn.deallocate(mo)
                if hook is not None:
                    hook(i - 1, prev[0], prev[1], h_end)
                if msgs is not None:
                    msgs.free()
                msgs = _Msgs(m.msg, X[2], h_end)
                h_in, prev, mo = X[1], None, None
            if layer == cfg.loop_end and msgs is not None:
                msgs.free()
                msgs = None
            h = h_in if h_in is not None else (X[self.ns(layer)] if cfg.is_boundary(layer) else hbuf[i % 2])
            full = cfg.is_full(layer)
            outs = xo[: (3 if full else 1)]
            kw = dict(mo=mo, r=r, prev=prev)
            if msgs is not None:
                self.pre_gated(L, nr, i, layer, p, X, SC, h, msgs, xo, **kw)
                outs = xo
            else:
                self.pre(L, nr, i, layer, p, X, SC, h, outs=outs, **kw)
            xq, xk, xv = (outs[0], outs[1], outs[2]) if len(outs) == 3 else (outs[0],) * 3
            if mo is not None:
                ttnn.deallocate(mo)
                if hook is not None:
                    hook(i - 1, prev[0], prev[1], h)
            a = attend(layer, p, xq, xk, xv)
            if msgs is not None:
                self.mid_gated(L, nr, i, layer, p, X, SC, a, h, r, msgs, z)
            else:
                self.mid(L, nr, i, layer, p, X, SC, a, h, r, out=z)
            ttnn.deallocate(a)
            mo = moe(layer, z)
            prev, h_in = (layer, p), None
        hl = hbuf[len(apps) % 2]
        self.final(L, nr, prev, mo, r, X, SC, hl, fin)
        ttnn.deallocate(mo)
        if hook is not None:
            hook(len(apps) - 1, prev[0], prev[1], hl)
        if msgs is not None:
            msgs.free()
        for t in own:
            ttnn.deallocate(t)
        return fin
