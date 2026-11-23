# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused decode glue (GlueFast): every non-attention / non-MoE op of a decode step in ~2 generic_op programs per layer
application (kernels/fast/glue_step.cpp), replacing the ~65 ttnn ops / application of layers.py's decode path.

Programs (one kernel source, structural variants as compile-time args):
  PRE  (before attention)  [post-MLP conv of the previous application + residual -> h | layer 0: embed_norm of the
       raw embedding row -> h]  -> block attention residual (1 or 3 queries over the block states + norm(h), ONE
       cross-core round; boundary layers append norm(h))  -> [q/k/v message gates]  -> input norm(s) -> xq (xk, xv)
  MID  (before the MoE)    post-attention conv + residual -> r (FP32)  -> MLP block residual -> [mlp message gate]
       -> post_attention_layernorm -> z
  MSG  (loop pass restart) post-MLP conv + residual of the previous pass's last application -> h_end, then the 48
       messages over [h_end, the previous pass's layer-12 block state] -> M (FP32) and bf16(norm(M)) (the gates read
       both; each gate computes its message half of the down projection)
  FIN  (after layer 23)    post-MLP conv + residual -> final block residual -> norm -> the LM head input

Numerics (see the kernel header): FP32 everywhere the reference is FP32; dot products / norms / softmax / mixes use
exact BF16 products with hi / lo splits of FP32 operands (~2^-17 relative); the partial state of a block residual is
the FP32-normed v (its 1/rms folded into the mixing coefficient), |O|^2 comes from the Gram matrix of the states; the
stored block states are BF16 (as the ttnn decode path); BF16 outputs are RNE-rounded like the reference's casts.

State: block states X (BF16 [1, 1, 32, H], row i = normed state i, written in place), the in-layer residual r (FP32),
h buffers, the output rows (row 0 of persistent [1, 1, 32, H] tensors, rows 1.. stay zero), the messages M / nM
(rows 0..47 of [1, 1, 64, H]); conv states are the position rings of model.rows_glue (row pos % 32 = the conv input
at pos; the programs read rows (p-3..p-1) % 32 and write row p % 32, p = cur_pos[0]; decode / verify use slot 0's
pages, batch row b slot bt.slot[b]'s with p = bt.cur_pos[b]).
Every persistent tensor is allocated in the constructor (trace safety, as tt/model.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import ttnn

from .common import KERNELS, CachedProgram, f32_bits

KERNEL = KERNELS / "glue_step.cpp"
DRAM = ttnn.DRAM_MEMORY_CONFIG
BF16, F32 = ttnn.bfloat16, ttnn.float32
HT = 72  # hidden tiles
NCONST = 29
NPT = 2  # BF16 parts per cross-core partial value (hi + lo; 3 = hi + mid + lo measured no more exact)
BATCH_ROWS = 16  # batch rows of FastGlue.batch (model.MAX_SEQS; the kernel reads rows < 16 of the position words)

# CB indices / unpack modes (glue_step.cpp)
CB = dict(IN=0, BASE=1, NH=2, ST=3, TAP=4, X=5, Q=6, GAM=7, MSG=8, NM=9, WD=10, WF=11, WG=12, CONST=13, G=14,
          RESF=15, RESB=16, D=17, HV=18, RF=19, VH=20, VL=21, PART=22, XF=23, OH=24, OL=25, NX=26, GK=27, OUT=28,
          SCR=29, PH=30, PL=31, STG=32, RSCR=33, WM=34, GE=35, ZQ=36, ZQL=37, MF=38, ORP=24, YR=25,  # ORP / YR: the OH / OL slots
          MRESF=39, MRESB=40)  # multi-row instances: a round master's own result tiles
GATE_KINDS = ("q", "k", "v", "mlp")


@dataclass(frozen=True)
class Variant:
    pro: int        # 0 h given | 1 h = r + conv(x) | 2 r = [h +] conv(x) | 3 h = embed_norm(e) (layer 0)
    bound: int = 0  # PRO 2: r = conv(x) (block boundary)
    nq: int = 0     # block residual queries (0: layer 0 / messages)
    ng: int = 0     # message gates
    gsame: int = 1  # gates read the mixed row 0 (sliding) / row k (full)
    nout: int = 1   # output rows
    msg: int = 0    # messages program
    in_dram: int = 1

    def slots(self):
        """Gather slots (kernel N_GSLOT; every partial value travels as NPT BF16 parts)."""
        has_v = self.nq == 0 and not self.msg
        return NPT * ((self.pro == 3) + has_v + ((self.nq + 5) if self.nq else 0) + 7 * self.msg + 3 * self.ng)

    def parts(self):
        """Partial tiles (kernel P_*)."""
        has_v = self.nq == 0 and not self.msg
        return NPT * ((self.pro == 3) + has_v + (self.nq > 0) + 3 * self.msg + 2 * self.ng + (self.ng > 0))

    def row_pages(self, T: int) -> dict:
        """Pages one row pushes into each per-row CB (kernel PR_*, the round-major RESF / RESB / PART blocks, the
        writer's staging): a multi-row launch (kernel MR) holds every row's pages, nothing is popped."""
        conv, mix, ng = self.pro in (1, 2), self.nq > 0, self.ng
        has_v = not mix and not self.msg
        need_base = self.pro in (0, 1, 3) or (self.pro == 2 and not self.bound)
        mixmsg = T if (mix or self.msg) else 0
        return dict(IN=int(conv), ST=int(conv), BASE=(1 if conv else T) if need_base else 0, X=mixmsg,
                    MSG=int(ng > 0), NM=ng * T, HV=T if self.pro == 3 else int(self.pro == 1),
                    VH=int(self.pro == 2), VL=int(self.pro == 2), RF=int(self.pro == 2), XF=mixmsg,
                    ZQ=T if mix else 0, ZQL=T if mix else 0, D=T if mix and self.pro == 2 else 0,
                    NX=T if ng else 0, ORP=2 if ng else 0, YR=2 if ng else 0, GK=2 * ng, PH=T if ng else 0,
                    PL=T if ng else 0, NH=1 if conv else T, OUT=2 * T if self.msg else (1 if mix else T),
                    MF=2 * T if self.msg else 0, SCR=4 if self.msg else 0, PART=self.parts(),
                    RESF=2 if self.msg else (self.pro == 3) + has_v + 2 * mix + (ng > 0),
                    RESB=2 * mix + 8 * self.msg + 2 * (ng > 0), STG=2)


class _Group:
    """One instance's cores: a gx x gy rectangle at logical (x0, y0); core k = y gx + x owns hidden tiles
    [k T, k T + T); core 0 is the master of the cross-core rounds."""

    def __init__(self, device, x0: int, y0: int, gx: int, gy: int):
        self.coords = [ttnn.CoreCoord(x0 + x, y0 + y) for y in range(gy) for x in range(gx)]
        self.range = ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + gx - 1, y0 + gy - 1))
        p0 = device.worker_core_from_logical_core(self.coords[0])
        p1 = device.worker_core_from_logical_core(self.coords[-1])
        self.master = (p0.x, p0.y)
        self.rect = (p0.x, p0.y, p1.x, p1.y)


class GlueGrid:
    """C = 72 / T cores per instance.  Instance 0 (the decode step) is a GX x GY rectangle at the worker-grid origin;
    the speculative verify runs one instance per token row on further disjoint rectangles (`groups(n)`)."""

    def __init__(self, device, tiles_per_core: int = 3):
        if HT % tiles_per_core or tiles_per_core > 3:
            raise ValueError("tiles_per_core must divide 72 and be <= 3 (row-packed tiles: rows 4 g + j, DEST slots)")
        self.T = tiles_per_core
        self.C = HT // tiles_per_core
        if self.C > 96:
            raise ValueError("at most 96 cores (three gather tiles per slot)")
        self.NGT = (self.C + 31) // 32  # gather tiles per slot (core k -> tile k // 32, row k % 32)
        g = device.compute_with_storage_grid_size()
        gx = next(x for x in range(min(g.x, self.C), 0, -1) if self.C % x == 0)
        self.GX, self.GY = gx, self.C // gx
        if self.GY > g.y:
            raise ValueError("grid too small")
        self.device = device
        # instance groups: the decode rectangle first, then greedy row-major placement of any C-core rectangle shape
        # (the default shape preferred) on the free cores
        shapes = [(gx, self.GY)] + [(a, self.C // a) for a in range(g.x, 0, -1)
                                    if self.C % a == 0 and a != gx and self.C // a <= g.y]
        free = [[True] * g.x for _ in range(g.y)]
        self._groups = []
        while True:
            spot = next(((x, y, w, h) for w, h in shapes for y in range(g.y - h + 1) for x in range(g.x - w + 1)
                         if all(free[y + b][x + a] for b in range(h) for a in range(w))), None)
            if spot is None:
                break
            x, y, w, h = spot
            for b in range(h):
                for a in range(w):
                    free[y + b][x + a] = False
            self._groups.append(_Group(device, x, y, w, h))
        self.max_instances = len(self._groups)
        self.cores = ttnn.CoreRangeSet({self._groups[0].range})
        self._consts = {}

    def groups(self, n: int):
        if not 1 <= n <= self.max_instances:
            raise ValueError(f"glue instances: 1..{self.max_instances} (T = {self.T}), got {n}")
        return self._groups[:n]

    def consts(self, ns: int, nq: int, msg: bool = False):
        """Constant tiles of a program (kernel K_* order, NCONST tiles): E_r (r < 5: row r), J (ones), SEL7 (rows
        0..7), softmax mask (rows < nq, columns <= ns; msg: columns < 16), COL0 / COLR (rows 0..7, column 0 /
        columns 1..ns), E00 ([0][0]), SX ([1 + i][i], i < ns), Q ([a][a], a < nq), QL ([a][4 + a]), SX3
        ([4 + i][i]), E30 ([3][0]), RM_k (k < 3: row k), ID, MX ([1][ns]), E0J_j / E3J_j (j < 4: [0][j] / [3][j]:
        row j of a row-packed tile -> row 0 / 3) -> BF16 [1, 1, 32, NCONST * 32] (tile i = columns 32 i ..), shared
        by the programs with the same key."""
        key = (ns, nq, bool(msg))
        if key not in self._consts:
            k = torch.zeros(NCONST, 32, 32)
            for r in range(5):
                k[r, r, :] = 1
            k[5] = 1
            k[6, :8, :] = 1
            if msg:
                k[7, :, :16] = 1
            else:
                k[7, :nq, : ns + 1] = 1
            k[8, :8, 0] = 1
            k[9, :8, 1: ns + 1] = 1
            k[10, 0, 0] = 1
            for i in range(ns):
                k[11, 1 + i, i] = 1
                k[14, 4 + i, i] = 1
            for a in range(nq):
                k[12, a, a] = 1
                k[13, a, 4 + a] = 1
            k[15, 3, 0] = 1
            for g in range(3):
                k[16 + g, g, :] = 1
            k[19] = torch.eye(32)
            k[20, 1, ns] = 1
            for j in range(4):
                k[21 + j, 0, j] = 1
                k[25 + j, 3, j] = 1
            host = k.permute(1, 0, 2).reshape(1, 1, 32, NCONST * 32).to(torch.bfloat16)
            self._consts[key] = ttnn.from_torch(host, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.device,
                                                memory_config=DRAM)
        return self._consts[key]


def _cb(index, dtype, tiles, cores):
    page = 4096 if dtype == F32 else 2048
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    return ttnn.CBDescriptor(total_size=max(tiles, 1) * page, core_ranges=cores,
                             format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])


# per-instance tensors (kernel per-core overrides PC_OVR.., in this order); every other tensor is shared
PER_INST = ("base", "x", "m", "nm", "oh", "mo", "nmo")


class GlueProgram:
    """One call site (application segment): variant + fixed per-site runtime values (NS, APPEND, message rows);
    run(n, ...) launches it on n instances (token rows; decode n = 1) with the given tensors (program rebuilt only
    when an address changes).  Per-instance tensors (PER_INST) are given as lists of n tensors; the shared input /
    output rows tensors carry instance r in row r (batch wave row0: row row0 + r; kernel MULTI 2: position / slot of
    the row from the batch tables `pos` / `slot`).  A batch of n > grid.max_instances rows is one multi-row launch
    (kernel MULTI 3): G = max_instances groups, group g runs the rows g, g + G, .. one after the other."""

    def __init__(self, grid: GlueGrid, v: Variant, hidden: int, eps: float, ns: int = 0, append: bool = False,
                 msg_rows=()):
        self.g, self.v = grid, v
        self.hidden, self.eps = hidden, eps
        self.ns, self.append = ns, int(append)
        self.msg_rows = list(msg_rows) + [0] * (3 - len(msg_rows))
        self.const = grid.consts(ns, v.nq, v.msg)
        self._progs = {}  # instances -> CachedProgram

    def _build(self, t: dict, n: int, bsh: int, extra=None, row0=None):
        g, v = self.g, self.v
        T = g.T
        base_f32 = int(v.pro == 1)
        g0 = g.groups(1)[0]
        mr = row0 is not None and n > g.max_instances  # multi-row instances
        mrn = -(-BATCH_ROWS // g.max_instances) if mr else 1  # MR: most rows per group (per-row CB capacity)
        multi = 3 if mr else (2 if row0 is not None else int(n > 1))
        named = dict(T=T, C=g.C, PRO=v.pro, BOUND=v.bound, NQ=v.nq, NG=v.ng, GSAME=v.gsame, NOUT=v.nout,
                     MSG=v.msg, NPT=NPT, INVH=f32_bits(1.0 / self.hidden), EPS=f32_bits(self.eps),
                     IN_DRAM=v.in_dram, BASE_F32=base_f32, MULTI=multi, MRN=mrn, BSH=bsh,
                     GX=g.GX, MX=g0.master[0], MY=g0.master[1], MCX0=g0.rect[0], MCY0=g0.rect[1], MCX1=g0.rect[2],
                     MCY1=g0.rect[3])
        addr = lambda k: int(t[k].buffer_address()) if t.get(k) is not None else 0  # noqa: E731
        rt = [addr("in"), 0, addr("ring"), addr("pos"), 0] + [addr(f"tap{i}") for i in range(4)]
        rt += [0, addr("q"), addr("gam"), 0, 0]
        rt += [addr(f"wd{i}") for i in range(3)] + [addr(f"wf{i}") for i in range(3)] + [addr(f"wg{i}") for i in range(3)]
        rt += [0] + [addr(f"o{i}") for i in range(3)]
        rt += [self.ns, self.append] + self.msg_rows + [addr("const")]
        rt += [addr(f"wm{i}") for i in range(3)] + [addr("ge"), 0, 0, addr("tapo"), addr("ptap")]
        if row0 is not None:  # batch wave: its first row, the slot words (kernel A_ROW0 / A_SLOT)
            rt += [row0, addr("slot")]
        groups = g.groups(min(n, g.max_instances) if mr else n)
        G = len(groups)
        cores = ttnn.CoreRangeSet([grp.range for grp in groups])
        dm = dict(kernel_source=str(KERNEL), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                  core_ranges=cores, named_compile_time_args=list(named.items()))
        ovr = [[int(t[key][r].buffer_address()) if t.get(key) is not None else 0 for key in PER_INST]
               for r in range(n)]
        if n == 1 and row0 is None:  # decode (MULTI 0): the per-instance tensors are common args
            for key, a in zip(PER_INST, ovr[0]):
                rt[{"base": 1, "x": 9, "m": 12, "nm": 13, "oh": 23, "mo": 37, "nmo": 38}[key]] = a
        else:
            per_core = ttnn.RuntimeArgs()
            noc = [[self.g.device.worker_core_from_logical_core(c) for c in grp.coords] for grp in groups]
            for r, grp in enumerate(groups):
                rows = list(range(r, n, G))  # MR: the group's rows; else [r]
                # MR: row count + the further rows' addresses, padded to the most rows a group can get (the per-core
                # argument counts must not depend on n: the programs of every n > max_instances share one binary /
                # program-cache entry, whose runtime arguments are updated in place), then the NOC x / y of row q's
                # round master (core q of the group), q < mrn
                more = [len(rows)] + [a for q in rows[1:] for a in ovr[q]] if mr else []
                more += [0] * ((1 + len(PER_INST) * (mrn - 1)) - len(more) if mr else 0)
                if mr:
                    more += [v for q in range(mrn) for v in (noc[r][q].x, noc[r][q].y)]
                for kk, c in enumerate(grp.coords):
                    sib = [v for s in range(G) if s != r for v in (noc[s][kk].x, noc[s][kk].y)]
                    per_core[c.x][c.y] = [kk, r, *grp.master, *grp.rect] + ovr[r] + [G - 1] + sib + more
            dm["runtime_args"] = per_core
        dm["common_runtime_args"] = rt
        fp32_cbs = [CB["IN"], CB["ST"], CB["TAP"], CB["MSG"], CB["RESF"]] + ([CB["BASE"]] if base_f32 else [])
        if mr:
            fp32_cbs.append(CB["MRESF"])
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, dst_full_sync_en=True)
        unpack = [ttnn.UnpackToDestMode.Default] * 64
        for i in fp32_cbs:
            unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode.extend(unpack)
        kernels = [
            ttnn.KernelDescriptor(**dm, config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default)),
            ttnn.KernelDescriptor(**dm, config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default)),
            ttnn.KernelDescriptor(**dm, config=cfg),
        ]
        ng, nm = v.ng, 2 if v.msg else 1
        conv = v.pro in (1, 2)  # conv prologue: row-packed IN / ST / TAP / BASE / HV / RF / VH / VL (kernel RP_*)
        # kernel RS_Q (pos / slot words, batch: the input rows' faces at RS_IN (MR: every row's; batch verify groups:
        # the in-group lags)) + a 1408 T B staging block per row
        rscr = -(-(128 + (1024 * T if row0 is not None else 0) + mrn * 1408 * T) // 2048)
        sizes = dict(IN=(F32, 1), BASE=(F32 if base_f32 else BF16, 1 if conv else T), NH=(BF16, T), ST=(F32, 1),
                     TAP=(F32, 1), X=(BF16, T), Q=(BF16, 4 * T if v.msg else T), GAM=(BF16, T), MSG=(F32, 1),
                     NM=(BF16, ng * T), WD=(BF16, ng * T * 2), WF=(BF16, ng * 2 * T), WG=(BF16, ng * 2 * T),
                     CONST=(BF16, NCONST), G=(BF16, v.slots() * g.NGT), RESF=(F32, 3), RESB=(BF16, 8),
                     D=(BF16, T), HV=(BF16, T), RF=(F32, 1), VH=(BF16, 1), VL=(BF16, 1), PART=(BF16, v.parts()),
                     XF=(BF16, T), NX=(BF16, T), GK=(BF16, 2 * ng),
                     OUT=(BF16, nm * T), SCR=(BF16, 8), PH=(BF16, T), PL=(BF16, T), STG=(BF16, 2),
                     RSCR=(BF16, rscr), WM=(BF16, ng * T * 2), GE=(BF16, T), ZQ=(BF16, T), ZQL=(BF16, T),
                     MF=(F32, 2 * T if v.msg else 1), ORP=(BF16, 2), YR=(BF16, 2))
        if mr:  # every row's pages of the per-row CBs (nothing is popped); the round masters' own result tiles
            for name, pages in v.row_pages(T).items():
                if pages:
                    sizes[name] = (sizes[name][0], mrn * pages)
            sizes.update(MRESF=(F32, 3), MRESB=(BF16, 8))
        cbs = [_cb(CB[name], dt, cnt, cores) for name, (dt, cnt) in sizes.items()]
        # 0 gather count, 1 round counter, 2 constants arrived (multicast by the master); instances: 3 sibling weight
        # shares arrived; MR: 4 row masters' results arrived at core 0, 5 round counter of the result multicasts
        nsem = 6 if mr else (4 if G > 1 else 3)
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in range(nsem)]
        if extra is not None:  # kernels of another job on other cores (attention weight prefetch, AttnPrefetch)
            kernels += extra[1]()
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)

    def run(self, n: int = 1, bsh: bool = False, extra=None, row0=None, **t):
        """n instances; PER_INST values: a tensor (n = 1) or a list of n tensors; bsh: `base` is a shared rows tensor
        (instance r reads row r; BF16).  extra = (tensors, build): build() -> more kernels for this program on cores
        outside the glue's (fast/attn_decode.py AttnPrefetch.glue_extra).  row0: a batch wave (instance r = batch row
        row0 + r; t["pos"] / t["slot"] = the batch tables' cur_pos / slot words)."""
        t["const"] = self.const
        for key in PER_INST:
            if key == "base" and bsh:
                t[key] = [t[key]] * n if t.get(key) is not None else None
            elif t.get(key) is not None and not isinstance(t[key], (list, tuple)):
                t[key] = [t[key]]
            if t.get(key) is not None and len(t[key]) != n:
                raise ValueError(f"GlueProgram: {key}: {n} instance tensors expected")
        if bsh and self.v.pro == 1:
            raise ValueError("GlueProgram: a shared base must be BF16 (PRO 0 / 2)")
        tensors = []
        for x in t.values():
            if isinstance(x, (list, tuple)):
                tensors += list(x)
            elif x is not None:
                tensors.append(x)
        if extra is not None:
            tensors += list(extra[0])
        self._progs.setdefault((n, bool(bsh), extra is not None, row0), CachedProgram()).run(
            tensors, lambda: self._build(t, n, int(bsh), extra, row0))


def _rows_tensor(device, rows: list, H: int, dtype=BF16, nrows: int = 32):
    """[1, 1, nrows, H] tensor with the given (row index, torch vector) rows, zeros elsewhere."""
    x = torch.zeros(1, 1, nrows, H, dtype=torch.float32)
    for r, v in rows:
        x[0, 0, r] = v.float()
    return ttnn.from_torch(x.to(torch.bfloat16 if dtype == BF16 else torch.float32), dtype=dtype,
                           layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def rp_tiles(device, vecs: list, T: int, dtype=BF16):
    """Row-packed per-core tiles of up to 8 [H] vectors: [1, 1, 32, 32 C] (C = H / (32 T)), tile k (core k) row
    4 g + j = vecs[g] over the core's column tile j (hidden columns (k T + j) 32 ..), zeros elsewhere."""
    H = vecs[0].shape[-1]
    C = H // (32 * T)
    x = torch.zeros(32, C, 32, dtype=torch.float32)
    for g, v in enumerate(vecs):
        x[4 * g: 4 * g + T] = v.float().reshape(C, T, 32).permute(1, 0, 2)
    x = x.reshape(1, 1, 32, C * 32)
    return ttnn.from_torch(x.to(torch.bfloat16 if dtype == BF16 else torch.float32), dtype=dtype,
                           layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def _hi_lo(q: torch.Tensor):
    """q [n, H] -> (hi, lo) BF16 halves of q * H^-0.5."""
    q = q.double()
    qs = q * q.shape[-1] ** -0.5
    hi = qs.to(torch.bfloat16)
    return hi, (qs - hi.double()).to(torch.bfloat16)


def query_tensor(device, q: torch.Tensor):
    """Block residual queries [nq, H] (FP32) -> rows 0..nq-1 = hi, rows 4..4+nq-1 = lo of q * H^-0.5 (BF16)."""
    q = q.reshape(-1, q.shape[-1])
    hi, lo = _hi_lo(q)
    rows = [(i, hi[i]) for i in range(q.shape[0])] + [(4 + i, lo[i]) for i in range(q.shape[0])]
    return _rows_tensor(device, rows, q.shape[-1])


def message_queries(src) -> torch.Tensor:
    """The 48 message queries [12 * 4, H] (row 4 j + k: loop layer j, k = q, k, v, mlp; FP32), merged as
    layers.MessageResidual: checkpoint queries rows (2j, 2j+1) = (q, mlp), kv_queries rows (2j, 2j+1) = (k, v)."""
    q = src("model.llm.middle_cycle_cross_pass_queries").float().unflatten(0, (-1, 2))
    kv = src("model.llm.middle_cycle_cross_pass_kv_queries").float().unflatten(0, (-1, 2))
    return torch.cat([q[:, :1], kv, q[:, 1:]], dim=1).reshape(-1, q.shape[-1])


def message_query_tensor(device, q: torch.Tensor):
    """48 message queries -> [1, 1, 128, H] BF16: rows 0..47 hi, rows 64..111 lo of q * H^-0.5."""
    hi, lo = _hi_lo(q)
    rows = [(i, hi[i]) for i in range(q.shape[0])] + [(64 + i, lo[i]) for i in range(q.shape[0])]
    return _rows_tensor(device, rows, q.shape[-1], nrows=128)


class _Inst:
    """One token's glue state (decode: instance 0; verify row r: instance r; batch row b: instance b)."""

    def __init__(self, zeros):
        self.X, self.r = zeros(), zeros(F32)
        self.hbuf = [zeros(), zeros()]
        self.pre_h, self.h_end = zeros(), zeros()
        self.M, self.nM = zeros(F32, 64), zeros(BF16, 64)  # messages of the current pass (rows 4 j + k)


class FastGlue:
    """The fused glue of an Inkling2Model's decode step.  decode(e, cur_pos) runs the 48 applications (calling the
    model's attention / MoE decode) on e = the raw embedding row (embed_norm is fused into layer 0's PRE) and returns
    the final normed hidden (row 0 of a persistent [1, 1, 32, H] BF16 tensor; the LM head input).
    verify(x, R, attend, moe, hook) runs the same programs for R token rows (speculative verify) as R instances on
    disjoint core groups: row r of every shared rows tensor = token r at position rows_glue.pos[0] + r; each row's
    result equals a decode step of that token (layer 0 starts from the normed embedding instead of fusing
    embed_norm).  batch(e, bt, R, attend, moe, hook) (batched serving) runs them for R rows of R sequences: row b at
    position bt.cur_pos[b] with slot bt.slot[b]'s conv rings, in waves of <= grid.max_instances instances; every row
    equals that sequence's decode step bitwise.  Construct after the model (it reads the model's glue modules, conv
    rings (model.rows_glue) and checkpoint source)."""

    def __init__(self, model, tiles_per_core: int | None = None):
        self.m = model
        cfg = self.cfg = model.cfg
        dev = self.dev = model.device
        H = self.H = cfg.hidden
        src = model.src.tensor
        self.grid = GlueGrid(dev, tiles_per_core or int(os.environ.get("INK_GLUE_T", "3")))
        zeros = lambda dt=BF16, n=32: _rows_tensor(dev, [], H, dt, n)  # noqa: E731
        # one state per instance: verify rows; batched models one per batch row (allocated now: trace safety)
        n_inst = max(self.grid.max_instances, BATCH_ROWS if getattr(model, "batched", False) else 0)
        self.insts = [_Inst(zeros) for _ in range(n_inst)]
        i0 = self.insts[0]  # the decode step's state (tests address it directly)
        self.X, self.r, self.hbuf, self.pre_h, self.h_end, self.M, self.nM = (
            i0.X, i0.r, i0.hbuf, i0.pre_h, i0.h_end, i0.M, i0.nM)
        self.xo = [zeros() for _ in range(3)]
        self.z, self.fin = zeros(), zeros()
        # verify / batch rows outputs (row r = instance r) and the applications' outputs for the tap hook
        self.vxo = [zeros() for _ in range(3)]
        self.vz, self.vfin, self.vtap = zeros(), zeros(), zeros()
        self.max_rows = self.grid.max_instances
        self.cur_pos = None
        self.slot_t = None  # batch: the batch tables' slot words (bt.slot)
        # batch R > max_instances: multi-row instances (default) or INK_BATCH_GLUE=waves (A/B: serial waves)
        self.batch_waves = os.environ.get("INK_BATCH_GLUE", "mr") == "waves"
        # conv taps per core as one row-packed FP32 tile (tile k: row 4 i + j = tap i over the core's column tile j)
        self.ptaps = {}
        for layer in range(cfg.n_layers):
            G = model.glue[layer]
            for kind, conv in (("attn", G.attn_conv), ("mlp", G.mlp_conv)):
                w = [ttnn.to_torch(t).float().reshape(-1, H)[0] for t in conv.taps]  # 4 x [H]
                self.ptaps[layer, kind] = rp_tiles(dev, w, self.grid.T, F32)
        self.boundaries = [b for b in range(cfg.n_layers) if cfg.is_boundary(b)]
        # per-layer constant tensors
        self.q_attn, self.q_mlp, self.g_pre, self.g_mid = {}, {}, {}, {}
        for layer in range(cfg.n_layers):
            p = f"model.llm.layers.{layer}"
            if layer:
                self.q_attn[layer] = query_tensor(dev, src(f"model.llm.attn_res.attn_queries.{layer - 1}"))
            self.q_mlp[layer] = query_tensor(dev, src(f"model.llm.attn_res.mlp_queries.{layer}"))
            if cfg.is_full(layer):
                ws = [src(f"{p}.{k}_input_norm.weight") for k in "qkv"]
            else:
                ws = [src(f"{p}.attn_norm.weight")] * 3
            # gammas: row-packed per core (rows 4 a + j; the mix programs), layer 0's as column tiles (rows a)
            self.g_pre[layer] = (rp_tiles(dev, ws, self.grid.T) if layer else
                                 _rows_tensor(dev, list(enumerate(ws)), H))
            self.g_mid[layer] = rp_tiles(dev, [src(f"{p}.mlp_norm.weight")], self.grid.T)
        self.q_fin = query_tensor(dev, src("model.llm.attn_res.output_query"))
        self.g_fin = rp_tiles(dev, [src("model.llm.norm.weight")], self.grid.T)
        self.g_emb = _rows_tensor(dev, [(0, src("model.llm.embed_norm.weight"))], H)
        self.q_msg = message_query_tensor(dev, message_queries(src))
        self._progs = {}
        self.prefetch = None  # fast/attn_decode.AttnPrefetch (set by the model): attention weight prefetch in MID / PRE
        for ns in range(len(self.boundaries) + 1):  # every program's constant tiles, allocated now (trace safety)
            for nq in (0, 1, 3):
                self.grid.consts(ns, nq)
        self.grid.consts(self.msg_row(), 0, True)

    # ------------------------------------------------------------------ program sites
    def ns(self, layer: int) -> int:
        """Block states before `layer`'s attention input (rows of X)."""
        return sum(b < layer for b in self.boundaries)

    def msg_row(self) -> int:
        """Row of X holding the previous pass's last block state (appended at layer 12): the messages' s0."""
        return self.ns(self.cfg.loop_start) + 1

    def _site(self, key, v: Variant, **kw):
        key = (key, v)
        if key not in self._progs:
            self._progs[key] = GlueProgram(self.grid, v, self.H, self.cfg.eps, **kw)
        return self._progs[key]

    def pre_variant(self, layer: int, p: int, direct: bool, embed: bool = True) -> Variant:
        """embed: layer 0 starts from the raw embedding (embed_norm fused, decode); else from the normed one."""
        cfg = self.cfg
        full = cfg.is_full(layer)
        ng = 3 if cfg.is_loop(layer) and p > 0 else 0
        nq = 0 if layer == 0 else (3 if full else 1)
        pro = 3 if layer == 0 and embed else (0 if direct else 1)
        return Variant(pro=pro, nq=nq, ng=ng, gsame=int(not full), nout=3 if (full or ng) else 1)

    def mid_variant(self, layer: int, p: int) -> Variant:
        cfg = self.cfg
        ng = 1 if cfg.is_loop(layer) and p > 0 else 0
        return Variant(pro=2, bound=int(cfg.is_boundary(layer)), nq=1, ng=ng, gsame=1, nout=1)

    def _conv_tensors(self, layer: int, kind: str, slot: int):
        conv = self.m.glue[layer].attn_conv if kind == "attn" else self.m.glue[layer].mlp_conv
        t = {f"tap{i}": tp for i, tp in enumerate(conv.taps)}
        t["ptap"] = self.ptaps[layer, kind]
        t.update(ring=self.m.rows_glue.ring(layer, kind, slot), pos=self.cur_pos)
        return t

    def _gate_tensors(self, layer: int, kinds, I):
        G = self.m.glue[layer]
        t = {"m": [s.M for s in I], "nm": [s.nM for s in I]}
        for i, kind in enumerate(kinds):
            gt = G.gates[kind]
            t[f"wd{i}"], t[f"wm{i}"], t[f"wf{i}"], t[f"wg{i}"] = gt.wdx, gt.wdm, gt.wf, gt.wg
        return t

    @staticmethod
    def _each(v, n):
        """A per-instance value: a tensor (n = 1) or a list of n tensors."""
        return v if v is None or isinstance(v, (list, tuple)) else [v] * n

    def _wave(self, n: int, row0):
        """The instance states of a launch: n instances from row0 (a batch wave) or from 0 (decode / verify)."""
        r0 = row0 or 0
        return self.insts[r0: r0 + n]

    def _run_prog(self, prog, n, t, row0, bsh=False, extra=None):
        if row0 is not None:  # batch wave: the kernel reads each row's slot (its conv rings' pages)
            t["slot"] = self.slot_t
        prog.run(n, bsh=bsh, extra=extra, row0=row0, **t)

    def pre(self, i: int, layer: int, p: int, h_in=None, prev=None, mo=None, h_out=None, n: int = 1,
            outs=None, bsh: bool = False, tapo=None, embed: bool = True, extra=None, row0=None):
        """PRE of application i over n instances: h_in (layer 0: the raw embedding row, or with embed=False the normed
        one; else direct h) or prev = (layer, pass) of the previous application with its MoE output mo (conv +
        residual from r -> h_out; layer 0: embed_norm -> h_out).  Per-instance h_in / h_out: lists of n tensors.
        bsh: h_in is a shared rows tensor (verify / batch layer 0: the instance's row).  outs: the 3 output rows
        tensors (default self.xo); tapo: h -> the instance's row of this tensor; row0: a batch wave (instances
        row0 .., rows row0 ..).  Returns (xq, xk, xv)."""
        cfg, I = self.cfg, self._wave(n, row0)
        outs = outs or self.xo
        v = self.pre_variant(layer, p, h_in is not None, embed=embed)
        j = layer - cfg.loop_start
        rows = [4 * j + k for k in range(3)] if v.ng else []
        prog = self._site(("pre", i), v, ns=self.ns(layer), append=cfg.is_boundary(layer), msg_rows=rows)
        t = dict(x=[s.X for s in I], q=self.q_attn.get(layer), gam=self.g_pre[layer], o0=outs[0],
                 o1=outs[1] if v.nout > 1 else None, o2=outs[2] if v.nout > 1 else None)
        if v.pro == 3:
            t.update(base=h_in if bsh else self._each(h_in, n), ge=self.g_emb, oh=self._each(h_out, n))
        elif h_in is not None:
            t["base"] = h_in if bsh else self._each(h_in, n)
        else:
            t.update(self._conv_tensors(prev[0], "mlp", prev[1]))
            t.update({"in": mo, "base": [s.r for s in I], "oh": self._each(h_out, n), "tapo": tapo})
        if v.ng:
            t.update(self._gate_tensors(layer, ("q", "k", "v"), I))
        self._run_prog(prog, n, t, row0, bsh=bsh, extra=extra)
        return (outs[0], outs[1], outs[2]) if v.nout > 1 else (outs[0],) * 3

    def mid(self, i: int, layer: int, p: int, a, h, n: int = 1, out=None, bsh: bool = False, extra=None, row0=None):
        """MID of application i: attention output a (rows), this application's input h (per instance; bsh: the shared
        rows tensor) -> z (MoE input rows; default self.z)."""
        cfg, I = self.cfg, self._wave(n, row0)
        out = out or self.z
        v = self.mid_variant(layer, p)
        j = layer - cfg.loop_start
        ns = self.ns(layer) + int(cfg.is_boundary(layer))
        prog = self._site(("mid", i), v, ns=ns, msg_rows=[4 * j + 3] if v.ng else [])
        t = dict(self._conv_tensors(layer, "attn", p))
        base = None if v.bound else (h if bsh else self._each(h, n))
        t.update({"in": a, "base": base, "x": [s.X for s in I], "q": self.q_mlp[layer], "gam": self.g_mid[layer],
                  "oh": [s.r for s in I], "o0": out})
        if v.ng:
            t.update(self._gate_tensors(layer, ("mlp",), I))
        self._run_prog(prog, n, t, row0, bsh=bsh and base is not None, extra=extra)
        return out

    def messages(self, key, prev, mo, n: int = 1, tapo=None, row0=None):
        """Pass restart: h_end = bf16(r + conv(mo)) of the previous application prev = (layer, pass) (-> h_end), then
        the messages over [h_end, X row msg_row()] -> M (FP32), nM (BF16 unweighted norm) of every instance."""
        I = self._wave(n, row0)
        prog = self._site(("msg", key), Variant(pro=1, msg=1, nout=0), ns=self.msg_row())
        t = self._conv_tensors(prev[0], "mlp", prev[1])
        t.update({"in": mo, "base": [s.r for s in I], "oh": [s.h_end for s in I], "x": [s.X for s in I],
                  "q": self.q_msg, "mo": [s.M for s in I], "nmo": [s.nM for s in I], "tapo": tapo})
        self._run_prog(prog, n, t, row0)
        return self.M, self.nM

    def final(self, prev, mo, n: int = 1, out=None, tapo=None, row0=None):
        """Final block residual + norm of the last application's output -> out (default self.fin)."""
        I = self._wave(n, row0)
        out = out or self.fin
        prog = self._site(("fin",), Variant(pro=1, nq=1, nout=1), ns=self.ns(self.cfg.n_layers))
        t = self._conv_tensors(prev[0], "mlp", prev[1])
        t.update({"in": mo, "base": [s.r for s in I], "oh": [s.hbuf[0] for s in I], "x": [s.X for s in I],
                  "q": self.q_fin, "gam": self.g_fin, "o0": out, "tapo": tapo})
        self._run_prog(prog, n, t, row0)
        return out

    # ------------------------------------------------------------------ the 48 applications
    def decode(self, e, cur_pos, attend=None, moe=None):
        """e: the raw embedding row (row 0 of a [1, 1, 32, H] BF16 tile), cur_pos: the decode position tensor (int32
        [1], element 0 = p) -> self.fin.  attend(layer, p, xq, xk, xv) / moe(layer, z) default to the model's
        modules (tests pass stand-ins).  self.prefetch (AttnPrefetch, set by the model): MID / PRE also copy part of
        the next attention application's weights into L1, and the attention modules read them there."""
        m = self.m
        self.cur_pos = cur_pos
        attend = attend or (lambda layer, p, q, k, v: m.attn[layer].decode(q, k, v, cur_pos, p))
        moe = moe or (lambda layer, z: m.moe[layer](z, decode=True))
        return self._with_prefetch(1, lambda: self._run(1, e, attend, moe, "decode"))

    def _with_prefetch(self, n: int, run):
        """run() with self.prefetch.active = n (its glue programs carry the copies of an n-row plan, the attention
        modules read them), when the prefetch has a plan for n rows."""
        pf = self.prefetch
        if pf is None or n not in pf.subset:
            return run()
        pf.active = n
        try:
            return run()
        finally:
            pf.active = 0

    def verify(self, x, R: int, attend, moe, hook=None, raw: bool = False):
        """Speculative-verify glue of R token rows (1 <= R <= self.max_rows) at positions p .. p+R-1 (p = word 0 of
        model.rows_glue.pos: rows_glue.set_pos(p, R) first): x = the embeddings (row r = token r; not consumed): raw
        (raw=True: embed_norm fused as in decode, so every row's glue equals a decode step's bitwise) or normed;
        attend(layer, p, xq, xk, xv) / moe(layer, z) take and return rows tensors (row r = token r; consumed here);
        hook(app, layer, pass, h) gets every application's output rows (self.vtap, valid until the next
        application).  Returns self.vfin (rows 0..R-1 = the LM head inputs).  Conv states: the position rings (row
        pos % 32), shared with prefill / decode: commit = advance the position.  self.prefetch: as in decode, with the
        R-row plan (copies on the cores outside the R glue instances)."""
        if not 1 <= R <= self.max_rows:
            raise ValueError(f"FastGlue.verify: 1 <= R <= {self.max_rows}, got {R}")
        self.cur_pos = self.m.rows_glue.pos
        return self._with_prefetch(R, lambda: self._run(R, x, attend, moe, "verify", hook=hook, raw=raw))

    def batch(self, e, bt, R: int, attend, moe, hook=None):
        """Batched decode glue of R rows (1 <= R <= 16; batched serving, NOTES-batch.md): row b = one sequence's decode
        step at position bt.cur_pos[b] (int32 [1, 32] words) with the conv rings of slot bt.slot[b] (int32 [1, 32]
        words; model.rows_glue rings, slot-major); e = the raw embedding rows (row b = token b; embed_norm fused in
        layer 0 as in decode; not consumed).  attend(layer, p, xq, xk, xv) / moe(layer, z) take and return rows
        tensors (row b = batch row b; consumed here); hook(app, layer, pass, h) as in verify.  The decode programs
        run as instances on the GlueGrid groups, one launch per site: R <= grid.max_instances: instance r = row r;
        more rows: multi-row instances (group g runs the rows g, g + G, .. one after the other, the shared inputs read
        once).  Every row has its own state insts[row].  Returns self.vfin (rows 0..R-1 = the LM head inputs); every
        row (outputs, taps, ring writes) is bitwise FastGlue.decode of that sequence.  Pad rows (replicas of row 0)
        write identical values to the same ring rows: safe (a row writes only ring row p % 32, reads p-3..p-1)."""
        if not 1 <= R <= min(len(self.insts), BATCH_ROWS):
            raise ValueError(f"FastGlue.batch: 1 <= R <= {min(len(self.insts), BATCH_ROWS)}, got {R}")
        self.cur_pos, self.slot_t = bt.cur_pos, bt.slot
        return self._run(R, e, attend, moe, "batch", hook=hook)

    def _run(self, n: int, x, attend, moe, mode: str, hook=None, raw: bool = True):
        """mode: "decode" (n = 1, outputs xo / z / fin), "verify" (n instances, rows tensors vxo / vz / vfin),
        "batch" (n rows, the verify rows tensors: one launch per site, multi-row instances when n > max_instances;
        INK_BATCH_GLUE=waves: waves of <= max_instances one-row instances instead)."""
        cfg = self.cfg
        verify = mode != "decode"  # rows tensors, layer 0 reads the shared embedding rows
        outs, zt, fin = (self.vxo, self.vz, self.vfin) if verify else (self.xo, self.z, self.fin)
        tapo = self.vtap if hook is not None else None
        mx = self.grid.max_instances
        if mode != "batch":
            waves = [(None, n)]
        elif self.batch_waves:
            waves = [(w0, min(mx, n - w0)) for w0 in range(0, n, mx)]
        else:
            waves = [(0, n)]
        prev, mo = None, None
        apps = cfg.applications()
        # with AttnPrefetch: MID(i) copies part of application i + 1's attention weights, PRE(i) the rest
        pf = (self.prefetch if mode != "batch" and self.prefetch is not None and self.prefetch.active == n
              else None)
        for i, (layer, p) in enumerate(apps):
            restart = layer == cfg.loop_start and p > 0
            if restart:  # pass restart: messages from the previous pass
                for w0, k in waves:
                    self.messages(p, prev, mo, k, tapo, row0=w0)
                ttnn.deallocate(mo)
                if hook is not None:
                    hook(i - 1, prev[0], prev[1], tapo)
                prev, mo = None, None
            bsh = verify and layer == 0  # verify / batch: layer 0 reads the instance's row of the embedding rows
            embed = layer == 0 and (raw or not verify)  # layer 0 fuses embed_norm (-> h = hbuf[0])
            pre_x = pf.glue_extra(i, "pre", n) if pf else None
            hs = {}
            for w0, k in waves:
                J = self._wave(k, w0)
                if layer == 0:
                    h_in = x
                    h = [s.hbuf[0] for s in J] if embed else x
                elif restart:
                    h = h_in = [s.pre_h for s in J]
                else:
                    h_in = None
                    h = [s.pre_h if layer == cfg.loop_start and p == 0 else s.hbuf[i % 2] for s in J]
                xq, xk, xv = self.pre(i, layer, p, h_in=h_in, prev=prev, mo=mo, h_out=h if layer or embed else None,
                                      n=k, outs=outs, bsh=bsh, tapo=tapo, embed=layer > 0 or embed, extra=pre_x,
                                      row0=w0)
                hs[w0] = h
            if mo is not None:
                ttnn.deallocate(mo)
                if hook is not None:
                    hook(i - 1, prev[0], prev[1], tapo)
            a = attend(layer, p, xq, xk, xv)
            mid_x = pf.glue_extra(i + 1, "mid", n) if pf and i + 1 < len(apps) else None
            for w0, k in waves:
                z = self.mid(i, layer, p, a, hs[w0], k, out=zt, bsh=bsh and not embed, extra=mid_x, row0=w0)
            ttnn.deallocate(a)
            mo = moe(layer, z)
            prev = (layer, p)
        for w0, k in waves:
            out = self.final(prev, mo, k, out=fin, tapo=tapo, row0=w0)
        ttnn.deallocate(mo)
        if hook is not None:
            hook(len(apps) - 1, prev[0], prev[1], tapo)
        return out
