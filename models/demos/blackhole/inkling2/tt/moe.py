# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini MoE block on one P150 (128 experts, top-8, softcapped SwiGLU, intermediate 1152, no shared expert).

`Inkling2MoE(mesh_device, source, layer_idx, cfg, *, expert_dtype)`; `y = moe(x, decode=...)` with x the
post_attention_layernorm output [1, 1, T, 2304] BF16 TILE and y the routed expert sum (before the post-MLP short
conv), BF16 TILE [1, 1, T, 2304].

Router (both paths, kernels/moe/moe_router.hpp): logits = x . W_r in FP32 SFPU multiply-adds (BF16 x BF16 products
are exact, the sums IEEE FP32: an FP32 reference up to summation order), scores = sigmoid(logits) (FP32 SFPU),
sel = scores + gate.bias.  The FPU matmul (HiFi4, FP32 DEST) accumulated at ~11-bit precision and flipped a
near-tie boundary expert on ~0.25% of rows; this router does not.
Selection (kernels): top-8 of `sel` by exact FP32 comparison, ties to the lower expert id (torch.topk's choice at
exact ties is unspecified); weights = scores[ids] / (sum + 1e-20) * route_scale in IEEE FP32.

Decode (`decode=True`, `rows` = R = 1..16 token rows of x, rows R..31 ignored and returned zero; R > 1: speculative
verify rows or the rows of independent batched sequences): one generic_op program (kernels/moe/moe_decode.cpp, see its
header) routes every row exactly and streams only the union of the rows' selected experts (ascending id; duplicate rows
add nothing) from the resident DRAM bank layout (moe_layout.py), each row weighted by its own routing weight; row r is
bitwise the R = 1 decode of that row (speculative verify == plain decode, batch row == the sequence alone).
Device-only and address-keyed: trace-capturable, no host reads; one cached program per R.
Prefill (`decode=False`): every row routed, experts grouped per 32-row group (moe_prefill.py).

Expert weights: `expert_dtype` ttnn.bfloat8_b (default) or ttnn.bfloat4_b, K-aligned BFP blocks (moe_layout.py),
NVFP4 layers factored (per-expert factor applied on device).  Per layer: BFP8 1.083 GB (gate_up 722 MB + down 361 MB),
BFP4 0.573 GB; `balance=True` adds gate_up_dup (1/9 of gate_up: BFP8 80 MB, BFP4 42 MB; kvplan.Plan.moe_balance).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import ttnn

from models.common.modules.moe.selected_experts import _CachedProgram, _row_ranges
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor

from . import moe_layout as lay

KERNELS = Path(__file__).resolve().parent / "kernels" / "moe"
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
TILE = 32
PREFIX = "model.llm.layers.{}.mlp."
MAX_ROWS = 16  # decode rows per call (speculative verify width / batched sequences)
ACT_GROUP = 6  # decode: intermediate tiles per activation page (pages spread over the L1 banks)
DOWN_AHEAD = 1  # decode: down weight blocks in flight (of the down ring): the gate/up streams get the banks first
DOWN_RING = {1: 6, 2: 6, 3: 8}  # decode: down weight ring blocks per R (12 from R = 4): a shallower down prefetch
# leaves the banks to gate/up at small R (MoE7 sweep: R=1 -2.0 us, R=2 -2.2, R=3 -1.4; deeper is better from R=4)
NOC1_BANKS = (4,)  # decode R <= NOC1_MAX_ROWS: banks whose stream cores read on NOC1 (bank 0 shares bank 4's NOC0 row)
NOC1_MAX_ROWS = 4  # (at R = 8 the NOC1 reads cost more than the shared row)
WIDE_ROUTER_GROUPS = 3  # decode R > 8: router row groups (router-side top-k; 3 x router_cores + 48 workers <= 110)


def down_ring(rows: int) -> int:
    """Down weight ring blocks of an R-row decode (DOWN_RING, 12 from R = 4)."""
    return DOWN_RING.get(rows, 12)


def router_layout(w: torch.Tensor) -> torch.Tensor:
    """gate.weight [E, H] -> the SFPU router's weight tiles: W^T [H, E] row-major BF16 cut into 1024-element pages
    (8 k rows x 128 experts each; moe_router.hpp)."""
    E, H = w.shape
    if E != 128 or H % 8:
        raise ValueError("the SFPU router layout needs 128 experts and H % 8 == 0")
    return w.to(torch.bfloat16).T.contiguous().reshape(H * E // 1024, 1024)


def fp32_unpack(*cbs: int) -> list:
    """unpack_to_dest_mode list: these CBs bypass SrcA (TF32) and unpack straight to DEST at FP32."""
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb in cbs:
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    return modes


def expert_fidelity(dtype):
    """BFP4 x BF16 is exact at LoFi; BFP8 (7-bit mantissas) needs HiFi2."""
    return ttnn.MathFidelity.LoFi if dtype == ttnn.bfloat4_b else ttnn.MathFidelity.HiFi2


def bank_memory_config(device, rows: int, width: int) -> ttnn.MemoryConfig:
    banks = device.dram_grid_size().x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                             ttnn.ShardSpec(grid, [rows, width // banks], ttnn.ShardOrientation.ROW_MAJOR))


def read_page(block_bytes: int, tile_bytes: int, burst: int = 16384) -> int:
    """Largest whole-tile NoC burst (<= 16 KB) dividing a streamed block."""
    page = (burst // tile_bytes) * tile_bytes
    while block_bytes % page:
        page -= tile_bytes
    return page


def _cb(cores, index, fmt, page, pages, tile=None):
    fd = (ttnn.CBFormatDescriptor(index, fmt, page, tile) if tile is not None
          else ttnn.CBFormatDescriptor(index, fmt, page))
    return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores, format_descriptors=[fd])


def _accessor(t) -> list[int]:
    a = ttnn.TensorAccessorArgs(t)
    if a.get_common_runtime_args():
        raise ValueError("accessors must be compile-time")
    return list(a.get_compile_time_args())


def noc0_flow_links(anchor: tuple, core: tuple, half: tuple, rows: int, south: int = 10) -> tuple[set, int]:
    """NOC0 links of the DRAM read responses from the bank at `anchor`'s row to `core` (logical coordinates).  NOC0
    routes east along the endpoint row, then south (wrapping through the two non-worker rows above logical row 0).
    Every worker of a bank lies in its half [x0, x1) of the grid (east of the bank's DRAM column), so the east run
    stays inside the half.  -> (links, hops); `south` weights a southward hop against an eastward one."""
    ax, ay = anchor
    x, y = core
    links = {("E", ay, c) for c in range(half[0], x + 1)}  # link entering column c of row ay
    dy = (y - ay) % rows
    links |= {("S", x, (ay + i + 1) % rows) for i in range(dy)}  # link entering row (ay + i + 1) of column x
    wrap = 2 if dy and y < ay else 0  # the physical rows 0 / 1 (Ethernet / router-only) on the way
    return links, (x - half[0] + 1) + south * (dy + wrap)


def plan_noc0_workers(grid: tuple, anchors: list, per_bank: int, south: int = 10) -> list[list[tuple]]:
    """`per_bank` workers per DRAM bank whose NOC0 response paths share no link with another bank's (MoE4).
    `anchors`: the banks' optimal NOC0 workers (first worker column east of the DRAM column, in the endpoint row);
    a bank's half runs from its anchor column to the next anchor column (or the grid edge).  Workers are dealt one per
    bank per round, each the free core of the bank's half with the fewest links used by other banks' flows, then the
    fewest hops (south hops weighted by `south`: the endpoint row first)."""
    gx, gy = grid
    starts = sorted({a[0] for a in anchors})
    half_of = {s: (s, ([t for t in starts if t > s] or [gx])[0]) for s in starts}
    free = {(x, y) for x in range(gx) for y in range(gy)}
    used: dict = {}  # link -> bank
    out = [[] for _ in anchors]
    for _ in range(per_bank):
        for b, anchor in enumerate(anchors):
            half = half_of[anchor[0]]
            best = None
            for c in sorted(free):
                if not half[0] <= c[0] < half[1]:
                    continue
                links, hops = noc0_flow_links(anchor, c, half, gy, south)
                key = (sum(1 for k in links if used.get(k, b) != b), hops, c[1], c[0])
                if best is None or key < best[0]:
                    best = (key, c, links)
            if best is None:
                raise ValueError(f"no free worker left for bank {b}")
            free.discard(best[1])
            for k in best[2]:
                used.setdefault(k, b)
            out[b].append(best[1])
    return out


def stream_workers(device, per_bank: int) -> list[list]:
    """Per DRAM bank, `per_bank` decode streaming workers with link-disjoint NOC0 response paths (plan_noc0_workers;
    a nearest-core placement made the banks sharing NOC0 links finish late, MoE4)."""
    grid = device.compute_with_storage_grid_size()
    anchors = [(c.x, c.y) for c in device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
    return [[ttnn.CoreCoord(*c) for c in row] for row in plan_noc0_workers((grid.x, grid.y), anchors, per_bank)]


class Inkling2MoE:
    def __init__(self, mesh_device, source, layer_idx: int, cfg, *, expert_dtype=ttnn.bfloat8_b, k_aligned: bool = False,
                 factored: bool = True, cache=None, weights: dict | None = None,
                 gate_splits: int = 3, down_cols: int = 3, router_cores: int = 16, fidelity=None,
                 gate_blk_k: int = 12, down_blk_k: int = 12, balance: bool = True):
        """`weights` (tests / synthetic): {'router' [E, H], 'bias' [E], 'gate' [E, I, H], 'up', 'down' [E, H, I],
        optional 'f13', 'f2' [E]} host tensors instead of reading `source`.  `cache`: common.WeightCache (keys
        L{layer}.moe.*) for the expert tensors.  `fidelity`: expert matmul fidelity (default expert_fidelity(dtype)).
        `gate_blk_k` / `down_blk_k`: decode stream blocks (k rows per block; the rings hold 8 / 12 blocks), results do
        not depend on them (same k order).  `balance` (kvplan.Plan.moe_balance): also store each expert's floating
        gate/up pair in the next bank (moe_layout.gate_up_dup_layout, +1/9 of gate_up: +1.0 GB BFP4 / +1.9 GB BFP8 over
        24 layers) so that decode streams the same number of gate/up units from every bank for any routing (results
        bitwise unchanged)."""
        self.device = mesh_device
        self.cfg = cfg
        self.layer = layer_idx
        self.E, self.top_k, self.H, self.I = cfg.n_experts, cfg.top_k, cfg.hidden, cfg.moe_intermediate
        self.route_scale = float(cfg.route_scale)
        self.dtype = expert_dtype
        self.fidelity = expert_fidelity(expert_dtype) if fidelity is None else fidelity
        self.k_aligned = k_aligned
        self.banks = mesh_device.dram_grid_size().x
        self.it = self.I // TILE
        self.pair_base, self.pair_odd = lay.gu_base(self.it, self.banks), lay.gu_odd(self.it, self.banks)
        self.down_cols = down_cols
        self.down_splits = self.H // TILE // self.banks // down_cols
        self.gate_splits = gate_splits
        self.router_cores = router_cores
        self.balance = bool(balance and self.pair_odd)  # bank-balanced gate/up units (needs gate_up_dup)
        self.gate_blk_k, self.down_blk_k = gate_blk_k, down_blk_k
        if self.it % ACT_GROUP or (self.H // TILE) % gate_blk_k or self.it % down_blk_k:
            raise ValueError("activation pages / stream blocks must split the k tiles evenly")
        if (self.H // TILE) % (self.banks * down_cols):
            raise ValueError("down columns must split evenly over banks and cores")
        if (self.H // 8) % router_cores or router_cores > 30:
            raise ValueError("router cores must split the H/8 router weight tiles evenly (and gather <= 31 rows)")
        p = PREFIX.format(layer_idx)
        if weights is None:
            weights = {"router": source.tensor(p + "gate.weight"), "bias": source.tensor(p + "gate.bias")}
            lazy = True
        else:
            lazy = False
        E, H, I = self.E, self.H, self.I
        # Router: weight tiles (router_layout) BF16 [H E / 1024, 1024], bias [1, 1, 1, E] FP32 row-major (512 B page),
        # factors [f13 | f2] FP32 row-major.
        self.router_w = ttnn.from_torch(router_layout(weights["router"]), dtype=ttnn.bfloat16,
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=DRAM)
        self.bias = ttnn.from_torch(weights["bias"].float().reshape(1, 1, 1, E), dtype=ttnn.float32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=DRAM)

        built = {}

        def experts():
            if not built:
                if lazy:
                    if factored and source.expert_format(layer_idx) == "nvfp4":
                        g, u, d, f13, f2 = source.experts_factored(layer_idx)
                    else:
                        g, u, d = source.experts(layer_idx)
                        f13 = f2 = None
                else:
                    g, u, d = weights["gate"], weights["up"], weights["down"]
                    f13, f2 = weights.get("f13"), weights.get("f2")
                built["f"] = torch.cat([torch.ones(E) if f13 is None else f13.float(),
                                        torch.ones(E) if f2 is None else f2.float()])
                g, u = g.to(torch.bfloat16), u.to(torch.bfloat16)
                built["gate_up"] = lay.gate_up_layout(g, u, self.banks, k_aligned)
                built["gate_up_dup"] = lay.gate_up_dup_layout(g, u, self.banks, k_aligned) if self.balance else None
                del g, u
                built["down"] = lay.down_layout(d.to(torch.bfloat16), self.banks, down_cols, k_aligned)
            return built

        tag = f"L{layer_idx}.moe.{'k' if k_aligned else 'n'}{'f' if factored else ''}.b{self.banks}c{down_cols}"
        gu_mc = bank_memory_config(mesh_device, E * H, 2 * I)
        dup_rows = E // 2 * 2 * H * self.banks // (2 * self.it)
        dup_mc = bank_memory_config(mesh_device, dup_rows, 2 * I)
        dn_mc = bank_memory_config(mesh_device, E * I, H)
        if cache is not None:
            self.gate_up = cache.get(f"{tag}.gate_up", lambda: experts()["gate_up"], expert_dtype,
                                     memory_config=gu_mc)
            self.down = cache.get(f"{tag}.down", lambda: experts()["down"], expert_dtype, memory_config=dn_mc)
            self.gate_up_dup = (cache.get(f"{tag}.gate_up_dup", lambda: experts()["gate_up_dup"], expert_dtype,
                                          memory_config=dup_mc) if self.balance else None)
            factors = cache.get(f"{tag}.factors", lambda: experts()["f"].reshape(1, 1, 1, 2 * E), ttnn.float32,
                                layout=ttnn.ROW_MAJOR_LAYOUT)
            self.factors = factors
        else:
            b = experts()
            self.gate_up = ttnn.from_torch(b["gate_up"], dtype=expert_dtype, layout=ttnn.TILE_LAYOUT,
                                           device=mesh_device, memory_config=gu_mc)
            self.down = ttnn.from_torch(b["down"], dtype=expert_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                        memory_config=dn_mc)
            self.gate_up_dup = (ttnn.from_torch(b["gate_up_dup"], dtype=expert_dtype, layout=ttnn.TILE_LAYOUT,
                                                device=mesh_device, memory_config=dup_mc) if self.balance else None)
            self.factors = ttnn.from_torch(b["f"].reshape(1, 1, 1, 2 * E), dtype=ttnn.float32,
                                           layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=DRAM)
        built.clear()
        self.tile_bytes = self.gate_up.get_tile().get_tile_size(expert_dtype)
        self._decode_prog: dict[int, _CachedProgram] = {}  # rows -> program
        self.keep_routing = False  # tests: decode also returns the [1,1,R,32+2E] uint32 routing rows
        self._prefill = None

    # ------------------------------------------------------------------ decode program
    def decode_cores(self, groups: int = 1):
        """(gate/up, down, router) core lists of the decode program: (logical core, bank | router index, part | router
        row group); `groups` router row groups of router_cores cores each (router_groups)."""
        dev = self.device
        workers = stream_workers(dev, self.gate_splits + self.down_splits)
        gate = [(row[p], b, p) for b, row in enumerate(workers) for p in range(self.gate_splits)]
        down = [(row[self.gate_splits + p], b, p) for b, row in enumerate(workers) for p in range(self.down_splits)]
        taken = {(c.x, c.y) for row in workers for c in row}
        grid = dev.compute_with_storage_grid_size()
        free = sorted(((x_, y_) for y_ in range(grid.y) for x_ in range(grid.x) if (x_, y_) not in taken),
                      key=lambda p: (abs(p[0] - grid.x // 2) + abs(p[1] - grid.y // 2), p[1], p[0]))
        n = self.router_cores
        router = [(ttnn.CoreCoord(*free[g * n + c]), c, g) for g in range(groups) for c in range(n)]
        return gate, down, router

    @staticmethod
    def router_groups(rows: int) -> int:
        """Router row groups of an R-row decode: R = 2..8 splits the rows into [0, (R+1)/2) and [(R+1)/2, R) (the
        workers' NCRISC / BRISC routing shares), each on its own router_cores cores: half the router MACs per core.
        R > 8 (router-side top-k, `RTK` in moe_decode.cpp): WIDE_ROUTER_GROUPS groups, rows split as evenly as
        possible."""
        return 1 if rows == 1 else (2 if rows <= 8 else WIDE_ROUTER_GROUPS)

    def slots(self, rows: int) -> int:
        """Largest expert union of `rows` rows (the decode program's slot capacity)."""
        return min(self.E, rows * self.top_k)

    def act_rows(self, rows: int) -> int:
        """Rows of an activation tile ([ar, 32], the down cores' in0 tiles): the tile height >= rows."""
        return next(h for h in (1, 2, 4, 8, 16) if h >= rows)

    def _decode_program(self, x, act, out, route, R):
        """The decode program (moe_decode.cpp): one kernel group per role (gate/up, down, router; binary = that
        role's code, MOE_ROLE), every CB and semaphore on ONE bounding rectangle of the cores, only common runtime
        args (a core finds its bank / part / router index in the core table; the stream cores of NOC1 banks swap their
        RISCs' sides at run time).  MoE7: a program's dispatch cost grows ~2.2 us per kernel group and per (CB core
        range); one all-role binary was slower in the model (NOTES-moe.md)."""
        dev = self.device
        G = self.router_groups(R)
        gate, down, router = self.decode_cores(G)
        cores = gate + down + router
        routers = self.router_cores
        grid = dev.compute_with_storage_grid_size()

        def noc(cs):
            out_ = []
            for c, _, _ in cs:
                n = dev.worker_core_from_logical_core(c)
                out_ += [n.x, n.y]
            return out_

        self.core_roles = {}  # (noc x, noc y) -> role (diagnostics: device profiles)
        for name, cs in (("gate_up", gate), ("down", down), ("router", router)):
            for c, b, p_ in cs:
                n = dev.worker_core_from_logical_core(c)
                self.core_roles[(n.x, n.y)] = f"{name}.b{b}.p{p_}"
        # core table: byte (y * grid.x + x) = 1 + index in [gate/up | down | router] cores, 0 = not a MoE core
        if len(cores) > 255:
            raise ValueError("the decode core table holds at most 255 cores")
        table = [0] * (-(-grid.x * grid.y // 4) * 4)
        for i, (c, _, _) in enumerate(cores):
            table[c.y * grid.x + c.x] = i + 1
        table_words = [sum(table[4 * w + j] << (8 * j) for j in range(4)) for w in range(len(table) // 4)]
        xs, ys = [c.x for c, *_ in cores], [c.y for c, *_ in cores]
        rect = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(min(xs), min(ys)), ttnn.CoreCoord(max(xs), max(ys)))])
        rt = self.H // 8 // routers  # router weight tiles (8 k each) per router core
        rbatch = max(d for d in range(1, 4) if rt % d == 0)  # pages per router weight batch (one trid each)
        gather_tiles = (routers + 8) // 8
        x_tiles = 8 * rt // TILE + 2  # x tiles covering one router core's k slice (upper bound)
        slots = self.slots(R)
        rp = R + R % 2
        rtk = R > 8  # router-side top-k (moe_decode.cpp RTK): a router core finishes one row, workers get records
        rg = 1 if rtk else ((R + 1) // 2 if G == 2 else R)  # rows finished per router core (gather regions)
        rows_area = R * 128 if rtk else R * 2 * self.E * 4  # cb_rows: records [R][128 B] | rows [R][sel | scores]
        wr = 4 if R <= 4 else (8 if R <= 8 else 16)  # row weights per slot entry (the SwiGLU's 4-row SFPU groups)
        ar = self.act_rows(R)
        max_units = -(-slots * (self.pair_base + self.pair_odd) // self.gate_splits)
        route_page = 16 * -(-4 * max(1 + max_units, slots * (1 + wr)) // 16)
        tb = self.tile_bytes
        named = [("experts", self.E), ("top_k", self.top_k), ("banks", self.banks), ("inter_tiles", self.it),
                 ("pair_base", self.pair_base), ("pair_odd", self.pair_odd), ("routers", routers),
                 ("router_tiles", rt), ("router_batch", rbatch), ("rows", R), ("row_pad", rp), ("w_rows", wr),
                 ("slots", slots), ("act_group", ACT_GROUP), ("act_rows", ar), ("balance", int(self.balance)),
                 ("router_groups", G), ("n_gate", len(gate)), ("n_down", len(down)), ("grid_x", grid.x),
                 ("table_words", len(table_words)),
                 ("noc1_mask", sum(1 << b for b in NOC1_BANKS) if R <= NOC1_MAX_ROWS else 0),
                 ("w_tile_bytes", tb), ("route_page", route_page),
                 ("route_scale_x1000", int(round(self.route_scale * 1000)))]
        # Stream roles: (k tiles, cols, k rows per streamed block, minimum blocks in the ring, blocks in flight, cores
        # per bank).  Down blocks of 12 k rows by default (36 tiles, 13 KB reads at BFP8; 9-row blocks streamed
        # slower).  Both roles share the weight CB (cb_w), sized to a whole number of blocks of each role (each role's
        # ring wraps where the CB does) and to the down ring of DOWN_RING (gate/up 9 / 12 / 18 blocks of 24 tiles,
        # down 6 / 8 / 12 of 36 at R = 1-2 / 3 / 4+ by default).
        roles = (("gate", (self.H // TILE, 2, self.gate_blk_k, 8, 6, self.gate_splits)),
                 ("down", (self.it, self.down_cols, self.down_blk_k, down_ring(R), DOWN_AHEAD, self.down_splits)))
        lcm = math.lcm(*(blk_k * cols for _, (_, cols, blk_k, *_) in roles))
        w_tiles = lcm * -(-max(n * blk_k * cols for _, (_, cols, blk_k, n, *_) in roles) // lcm)
        for name, (kt, cols, blk_k, _, ahead, splits) in roles:
            block_bytes = blk_k * cols * tb
            page = read_page(block_bytes, tb)
            named += [(f"{name}_k_tiles", kt), (f"{name}_cols", cols), (f"{name}_blk_k", blk_k),
                      (f"{name}_splits", splits), (f"{name}_pages", block_bytes // page),
                      (f"{name}_page_bytes", page), (f"{name}_buffers", w_tiles // (blk_k * cols)),
                      (f"{name}_ahead", ahead)]
        # Common runtime args (every RISC): the tensor addresses, then the core table and the NOC coordinates of the
        # gate/up, down and router cores.  (As positional compile-time args, read from the binary's read-only data,
        # the same lists made the model step 0.04 ms slower.)
        common = [x.buffer_address(), act.buffer_address(), out.buffer_address(), route.buffer_address(),
                  self.factors.buffer_address(), self.gate_up.buffer_address(), self.down.buffer_address(),
                  self.gate_up_dup.buffer_address() if self.balance else 0, self.router_w.buffer_address(),
                  self.bias.buffer_address()] + table_words + noc(gate) + noc(down) + noc(router)
        accessors = (_accessor(x) + _accessor(act) + _accessor(out) + _accessor(route) + _accessor(self.factors)
                     + _accessor(self.router_w) + _accessor(self.bias))
        kernels = []
        for role, cs in enumerate((gate, down, router)):
            compute = ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4 if role == 2 else self.fidelity, fp32_dest_acc_en=True,
                math_approx_mode=False)
            if role == 2:  # router: gathered partials unpacked straight to DEST at FP32
                compute.unpack_to_dest_mode.extend(fp32_unpack(7))
            kernels += UnifiedKernelDescriptor(
                kernel_source=str(KERNELS / "moe_decode.cpp"), core_ranges=_row_ranges([c for c, *_ in cs]),
                defines=[("MOE_ROLE", str(role)), ("K_ALIGNED", "1" if self.k_aligned else "0")],
                ncrisc_compile_time_args=accessors, brisc_compile_time_args=accessors,
                ncrisc_named_compile_time_args=named, brisc_named_compile_time_args=named,
                trisc_named_compile_time_args=named, trisc_compute_config=compute,
                ncrisc_common_runtime_args=common, brisc_common_runtime_args=common, trisc_common_runtime_args=common,
            ).get_kernel_descriptors().kernels
        tile = ttnn.TileDescriptor(ttnn.Tile([TILE, TILE]))
        dtile = ttnn.TileDescriptor(ttnn.Tile([ar, TILE]))  # down in0 / out: [ar, 32] tiles = the act page layout
        wtile = ttnn.TileDescriptor(self.gate_up.get_tile())
        bf16, fp32 = ttnn.bfloat16, ttnn.float32
        cbs = [  # indices: moe_decode.cpp cb_gx .. cb_rout
            _cb(rect, 0, bf16, 2048, self.H // TILE, tile),             # gate/up x tiles
            _cb(rect, 1, self.dtype, tb, w_tiles, wtile),                # weight blocks (both stream roles)
            _cb(rect, 2, bf16, 16384, 1),                                 # scratch
            _cb(rect, 3, bf16, route_page, 2),                            # TRISC plan / slot table
            _cb(rect, 4, bf16, 2048, 2, tile),                            # gate/up activation tile
            _cb(rect, 5, bf16, 2048, self.down_cols, tile),               # down output staging
            # router rows (R > 8: router records), flags, down counters: R x [sel E | scores E] FP32 (R > 8: R x
            # 128 B), stream / write side flag words (64 B), one 16 B word per slot
            _cb(rect, 6, bf16, rows_area + 64 + slots * 16, 1),
            _cb(rect, 7, fp32, 4096, rg * gather_tiles, tile),            # router gather (partials + bias)
            _cb(rect, 8, bf16, ar * 2 * TILE, 2 * self.it, dtile),        # down act tiles
            _cb(rect, 9, bf16, ar * 2 * TILE, self.down_cols, dtile),     # down output tiles
            _cb(rect, 10, bf16, x_tiles * 64 * rp, 1),                    # router x face rows
            _cb(rect, 11, bf16, 2048, rt, tile),                          # router weight tiles
            _cb(rect, 12, fp32, 4096, 1, tile),                           # router partials
            _cb(rect, 13, fp32, 4096, 2, tile),                           # router [sel | scores]
        ]
        # semaphore 0: router all-gather count; 1: down cores' counters zeroed (on the gate/up cores); R > 8: 2 = router
        # records landed (on the workers)
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=rect, initial_value=0) for s in range(3 if rtk else 2)]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)

    def decode(self, x, rows: int = 1, memory_config=L1):
        """x [1,1,32,H] BF16 TILE, rows 0..R-1 = R tokens (R = `rows`, 1..16; consecutive positions for a speculative
        verify, independent sequences in a batched step; rows R..31 are ignored) -> [1,1,32,H] BF16 TILE (rows 0..R-1 =
        their MoE outputs, rows R..31 zero).
        Row r of the output is bitwise the rows=1 output of row r.  One program: router (router_cores) + gate/up (24)
        + down (24) streaming the union of the rows' experts once; with keep_routing also the [1,1,R,32+2E] uint32
        routing rows (ids in words 0-7, FP32 weights in words 16-23, FP32 sel in words 32.., scores after)."""
        if tuple(x.shape) != (1, 1, TILE, self.H) or x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"decode takes a BF16 tile row block [1,1,32,{self.H}], got {tuple(x.shape)}")
        if not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"decode takes 1..{MAX_ROWS} rows, got {rows}")
        dev = self.device
        act = ttnn.empty([1, 1, self.slots(rows) * self.it // ACT_GROUP, ACT_GROUP * self.act_rows(rows) * TILE],
                         dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=L1)
        route = ttnn.empty([1, 1, rows, 32 + 2 * self.E], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                           memory_config=DRAM)
        out = ttnn.empty([1, 1, TILE, self.H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=memory_config)
        tensors = [x, self.router_w, self.bias, self.factors, route, act, out, self.gate_up, self.down]
        if self.balance:
            tensors.append(self.gate_up_dup)
        prog = self._decode_prog.setdefault(rows, _CachedProgram())
        prog.run(tensors, lambda: self._decode_program(x, act, out, route, rows))
        ttnn.deallocate(act)
        if self.keep_routing:
            return out, route
        ttnn.deallocate(route)
        return out

    # ------------------------------------------------------------------ entry
    def __call__(self, x, decode: bool = False, rows: int = 1):
        """decode=True: `rows` (1..16) token rows of x through the decode program (see decode); False: prefill."""
        if decode:
            return self.decode(x, rows=rows, memory_config=DRAM)
        from .moe_prefill import prefill

        return prefill(self, x)
