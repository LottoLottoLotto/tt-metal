# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini MoE prefill (T rows, T % 32 == 0): device-only grouped experts over the resident bank layout.

Four programs (+ the row-major copy of x), all on the whole worker grid:
  kernels/moe/moe_router_rows.cpp (route): exact FP32 router (SFPU, moe_router.hpp; the router weights multicast once
     into every core's L1) + top-8 + weights per token -> 32 B records; then 16 table cores count every expert and
     write the expert-group map (32-slot pages, slot = [token | k << 16, bf16 weight * f2], ascending expert id, token
     order within an expert) and the meta page (counts, group starts).  Replaces the stock single-core prefix and the
     pack program.
  kernels/moe/moe_grouped.cpp STAGE 0 (gate/up): each expert's groups in row blocks of <= RB; every core takes an
     equal contiguous range of the (row block, gate/up pair) units of its NOC half (west cores: the first pairs =
     banks 0-3, east cores the rest), gathers the block's x rows itself (tilized on TRISC) and streams each pair once
     per row block; softcapped SwiGLU x per-row weight -> act tiles.
  kernels/moe/moe_grouped.cpp STAGE 1 (down): same split over (row block, column group) units; FP32 rows written
     per (token, pick) into the combine buffer.
  kernels/moe/moe_combine.cpp: FP32 sum of the 8 picks per token in pick order, one BF16 rounding, tilize.
The arithmetic of every output element is the previous pipeline's (router, matmul k order, SwiGLU, FP32 pick-order
sum): outputs are bitwise those of the stock prefix / pack / combine version.  Programs are built once per shape;
every call only rewrites the tensor addresses (common runtime args).
Group capacity is the host bound min(8T, 8T/32 + E); only the device-side groups are processed.
"""
from __future__ import annotations

import os
import time

import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from .moe import DRAM, KERNELS, TILE, _accessor, _cb, fp32_unpack, read_page

N_TABLE = 16         # table cores (experts per table core = E / N_TABLE)
N_SEND = 8           # router weight senders (each reads w_tiles / N_SEND tiles and multicasts them to every core)
META_WORDS = 512     # meta page: counts [E], group starts [E + 1] (uint32)
W_BUFFERS = 3        # weight units in the stream ring (one ahead)
# groups per row block (x / act tiles held in L1): gate/up 147 KB per group, down 74 KB per group
ROW_BLOCK = {"gate_up": {ttnn.bfloat4_b: 4, ttnn.bfloat8_b: 3}, "down": {ttnn.bfloat4_b: 8, ttnn.bfloat8_b: 6}}
PROFILE = bool(int(os.environ.get("INK_MOE_PROFILE", "0")))


def _rm(dev, shape, dtype=ttnn.uint32):
    return ttnn.empty(shape, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM)


class _Program:
    """A program descriptor built once per shape (the first call's tensors give the compile-time accessors); every
    call writes the tensor addresses into the kernels' common runtime args ([NCRISC, BRISC] order of
    UnifiedKernelDescriptor): no per-call descriptor rebuild (generic_op's program hash ignores runtime-arg values)."""

    def __init__(self):
        self.desc = None

    def run(self, tensors, build, commons):
        if self.desc is None:
            self.desc = build()
        for k, args in zip(self.desc.kernels, commons):
            k.common_runtime_args = args
        ttnn.generic_op(tensors, self.desc)


class Prefill:
    def __init__(self, moe):
        self.moe = moe
        self.device = moe.device
        grid = self.device.compute_with_storage_grid_size()
        self.grid = grid
        self.cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        self.all = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self._programs: dict = {}
        self.times: dict = {}
        self._t = 0.0
        self.table_noc = []  # NOC x, y of the route program's table cores (its first N_TABLE cores)
        for c in self.cores[:N_TABLE]:
            p = self.device.worker_core_from_logical_core(c)
            self.table_noc += [p.x, p.y]
        lo = self.device.worker_core_from_logical_core(self.cores[0])
        hi = self.device.worker_core_from_logical_core(self.cores[-1])
        self.mcast = [lo.x, lo.y, hi.x, hi.y]  # NOC0 multicast rectangle of the grid
        # NOC halves of the expert programs: the cores west of the east banks' nearest cores stream banks
        # [0, banks / 2), the others banks [banks / 2, banks) (moe_grouped.cpp)
        anchors = self.device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        nb = len(anchors)
        x_east = min(c.x for c in anchors[nb // 2:])
        if max(c.x for c in anchors[:nb // 2]) >= x_east:
            raise ValueError("expected the first half of the DRAM banks west of the second")
        self.halves = [[c for c in self.cores if c.x < x_east], [c for c in self.cores if c.x >= x_east]]

    def capacity(self, tokens: int) -> int:
        k, E = self.moe.top_k, self.moe.E
        return min(tokens * k, tokens * k // TILE + E)

    def router_rows(self, tokens: int) -> int:
        """Rows per router item (1, 2, 4, 8; items dealt round-robin over every core): the least busiest-core cost
        items x (rows x mac + weight-tile unpack), relative per-tile costs 80 / 60 (larger rows on ties)."""
        n = len(self.cores)
        return min((8, 4, 2, 1), key=lambda r: -(-tokens // r // n) * (80 * r + 60) if tokens % r == 0 else 1 << 60)

    def _progs(self, key):
        if key not in self._programs:
            self._programs[key] = {n: _Program() for n in ("route", "s0", "s1", "combine")}
        return self._programs[key]

    # ------------------------------------------------------------------ route + table program
    def _route(self, rows, records, gmap, meta, dump):
        """rows: x row-major BF16 [1,1,T,H]; records [1,1,T/8,64] uint32 scratch; gmap [1,1,cap,64] uint32 group map;
        meta [1,1,1,512] uint32; dump: None or [1,1,T,2E] FP32 rows [sel | scores] (tests)."""
        m = self.moe
        dev = self.device
        tokens = int(rows.shape[-2])
        per_item = self.router_rows(tokens)
        cores = self.cores
        cs = self.all
        w_tiles = m.H * m.E // 1024
        block = w_tiles // m.router_cores  # k blocks = the decode router cores' slices (bitwise decode == prefill)
        common = [("experts", m.E), ("top_k", m.top_k), ("hidden", m.H), ("tokens", tokens), ("rows", per_item),
                  ("block_tiles", block),
                  ("route_scale_x1000", int(round(m.route_scale * 1000))), ("dump", int(dump is not None)),
                  ("n_table", N_TABLE), ("n_send", N_SEND)]
        sink = dump if dump is not None else records
        n = len(cores)
        args = [(c, [i, n]) for i, c in enumerate(cores)]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        compute.unpack_to_dest_mode.extend(fp32_unpack(7))
        k = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "moe_router_rows.cpp"), core_ranges=cs,
            ncrisc_compile_time_args=_accessor(rows) + _accessor(m.router_w),
            ncrisc_named_compile_time_args=common,
            brisc_compile_time_args=(_accessor(m.bias) + _accessor(m.factors) + _accessor(records) + _accessor(sink)
                                     + _accessor(gmap) + _accessor(meta)),
            brisc_named_compile_time_args=common,
            ncrisc_common_runtime_args=[0] * (8 + 2 * N_TABLE),
            brisc_common_runtime_args=[0] * (8 + 2 * N_TABLE),
            trisc_named_compile_time_args=common,
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(ncrisc_args=args, brisc_args=args,
                                                                          trisc_args=args),
        )
        tile = ttnn.TileDescriptor(ttnn.Tile([TILE, TILE]))
        bf16 = ttnn.bfloat16
        table_groups = m.E // N_TABLE * -(-tokens // TILE)  # an expert holds at most T rows
        cbs = [
            _cb(cs, 0, bf16, per_item * m.H * 2, 2),
            _cb(cs, 1, bf16, 2048, w_tiles, tile),        # resident router weights (multicast)
            _cb(cs, 2, bf16, 2 * m.E * 4 + 8 * 32, 1),
            _cb(cs, 4, ttnn.float32, 4096, 2, tile),
            _cb(cs, 7, ttnn.float32, 4096, 1, tile),
            _cb(cs, 8, bf16, tokens * 32, 1),             # table: every record
            _cb(cs, 9, bf16, META_WORDS * 4, 1),          # table: counts / starts / cursors (meta page)
            _cb(cs, 10, bf16, table_groups * 256, 1),     # table: the core's group pages
        ]
        # 0: records written (table cores), 1: router weight senders done, 2: cores started (senders)
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cs, initial_value=0) for s in range(3)]
        return ttnn.ProgramDescriptor(kernels=k.get_kernel_descriptors().kernels, cbs=cbs, semaphores=sems)

    # ------------------------------------------------------------------ grouped experts programs
    def _experts(self, stage, x, gmap, meta, out):
        """stage 0: x = row-major x [1,1,T,H], out = act tiles [1,1,cap*32,I]; stage 1: x = act tiles, out = combine
        rows [1,1,T/32 * col_groups * top_k, 32 * cols * 32] FP32 row-major."""
        m = self.moe
        gate_up = stage == 0
        resident = m.gate_up if gate_up else m.down
        kt = m.H // TILE if gate_up else m.it
        cols = 2 if gate_up else m.down_cols
        rb = ROW_BLOCK["gate_up" if gate_up else "down"][m.dtype]
        tb = m.tile_bytes
        unit_bytes = kt * cols * tb
        page = read_page(unit_bytes, tb)
        cores = self.cores
        C = len(cores)
        cs = self.all
        common = [("k_tiles", kt), ("cols", cols), ("rb", rb), ("experts", m.E), ("top_k", m.top_k),
                  ("inter_tiles", m.it), ("bank_cols", m.H // TILE // m.banks), ("pair_base", m.pair_base),
                  ("pair_odd", m.pair_odd), ("hidden_tiles", m.H // TILE)]
        nu = m.it if gate_up else m.H // TILE // cols
        split = round(nu * len(self.halves[0]) / len(self.cores))  # west cores' share of each row block's units
        win = [(0, split), (split, nu)]
        args = [(c, [i, len(half), *win[h]]) for h, half in enumerate(self.halves) for i, c in enumerate(half)]
        k = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "moe_grouped.cpp"), core_ranges=cs,
            defines=[("STAGE", str(stage)), ("K_ALIGNED", "1" if m.k_aligned else "0")],
            ncrisc_compile_time_args=_accessor(meta) + _accessor(m.factors),
            ncrisc_named_compile_time_args=common + [("w_tile_bytes", tb), ("w_page_bytes", page),
                                                     ("w_pages", unit_bytes // page), ("w_buffers", W_BUFFERS)],
            brisc_compile_time_args=_accessor(meta) + _accessor(gmap) + _accessor(x) + _accessor(out),
            brisc_named_compile_time_args=common,
            ncrisc_common_runtime_args=[0] * 5, brisc_common_runtime_args=[0] * 6,
            trisc_named_compile_time_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=m.fidelity,
                                                              fp32_dest_acc_en=True, math_approx_mode=False),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(ncrisc_args=args, brisc_args=args,
                                                                          trisc_args=args),
        )
        tile = ttnn.TileDescriptor(ttnn.Tile([TILE, TILE]))
        bf16 = ttnn.bfloat16
        cbs = [
            _cb(cs, 0, bf16, 2048, rb * kt, tile),
            _cb(cs, 1, m.dtype, tb, W_BUFFERS * kt * cols, ttnn.TileDescriptor(resident.get_tile())),
            _cb(cs, 2, bf16, META_WORDS * 4 + 2 * m.E * 4, 1),
            (_cb(cs, 4, bf16, 2048, 2, tile) if gate_up else _cb(cs, 4, ttnn.float32, 4096, 2 * cols, tile)),
            _cb(cs, 5, bf16, 64, 4),
            _cb(cs, 6, bf16, META_WORDS * 4, 1),
            _cb(cs, 8, bf16, rb * 256, 1),
        ]
        if gate_up:
            cbs += [_cb(cs, 3, bf16, 2048, rb, tile), _cb(cs, 7, bf16, 2048, kt, tile)]
        return ttnn.ProgramDescriptor(kernels=k.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    # ------------------------------------------------------------------ combine program
    def _combine(self, rows, out):
        m = self.moe
        tokens = int(out.shape[-2])
        cols, ht = m.down_cols, m.H // TILE
        items = tokens // TILE * (ht // cols)
        cores = self.cores[:min(items, len(self.cores))]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        n = len(cores)
        args = [(c, [i, n]) for i, c in enumerate(cores)]
        named = [("tokens", tokens), ("hidden_tiles", ht), ("cols", cols), ("top_k", m.top_k)]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        compute.unpack_to_dest_mode.extend(fp32_unpack(0))
        k = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "moe_combine.cpp"), core_ranges=cs,
            ncrisc_compile_time_args=_accessor(rows), ncrisc_named_compile_time_args=named,
            brisc_compile_time_args=_accessor(rows) + _accessor(out), brisc_named_compile_time_args=named,
            trisc_named_compile_time_args=named, trisc_compute_config=compute,
            ncrisc_common_runtime_args=[0] * 4, brisc_common_runtime_args=[0] * 4,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(ncrisc_args=args, brisc_args=args,
                                                                          trisc_args=args),
        )
        tile = ttnn.TileDescriptor(ttnn.Tile([TILE, TILE]))
        cbs = [_cb(cs, 0, ttnn.float32, 4096, 2 * m.top_k * cols, tile), _cb(cs, 2, ttnn.bfloat16, 2048, cols, tile),
               _cb(cs, 16, ttnn.bfloat16, 2048, 2 * cols, tile)]
        return ttnn.ProgramDescriptor(kernels=k.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    # ------------------------------------------------------------------ forward
    def __call__(self, x, dump=None):
        m = self.moe
        dev = self.device
        tokens, hidden = tuple(x.shape)[-2:]
        progs = self._progs((tokens, dump is not None))
        cap = self.capacity(tokens)
        cols = m.down_cols
        rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        records = _rm(dev, [1, 1, tokens // 8, 64])
        gmap = _rm(dev, [1, 1, cap, 64])
        meta = _rm(dev, [1, 1, 1, META_WORDS])
        rt = [rows, m.router_w, m.bias, m.factors, records, gmap, meta] + ([dump] if dump is not None else [])
        sink = dump if dump is not None else records
        a = lambda *ts: [0, 0] + [t.buffer_address() for t in ts]  # noqa: E731
        progs["route"].run(rt, lambda: self._route(rows, records, gmap, meta, dump),
                           [a(rows, m.router_w) + self.mcast + self.table_noc,
                            a(m.bias, m.factors, records, sink, gmap, meta) + self.table_noc])
        ttnn.deallocate(records)
        self._mark("route")
        act = ttnn.empty([1, 1, cap * TILE, m.I], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=DRAM)
        s0 = [rows, gmap, meta, act, m.factors, m.gate_up]
        progs["s0"].run(s0, lambda: self._experts(0, rows, gmap, meta, act),
                        [a(meta, m.factors, m.gate_up), a(meta, gmap, rows, act)])
        ttnn.deallocate(rows)
        self._mark("gate_up")
        picks = _rm(dev, [1, 1, tokens // TILE * (hidden // TILE // cols) * m.top_k, TILE * cols * TILE],
                    ttnn.float32)
        s1 = [act, gmap, meta, picks, m.factors, m.down]
        progs["s1"].run(s1, lambda: self._experts(1, act, gmap, meta, picks),
                        [a(meta, m.factors, m.down), a(meta, gmap, act, picks)])
        for t in (act, gmap, meta):
            ttnn.deallocate(t)
        self._mark("down")
        out = ttnn.empty([1, 1, tokens, hidden], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=DRAM)
        progs["combine"].run([picks, out], lambda: self._combine(picks, out), [a(picks, out), a(picks, out)])
        ttnn.deallocate(picks)
        self._mark("combine")
        return out

    def _mark(self, name):
        """INK_MOE_PROFILE=1: synchronize after each stage and accumulate wall ms in self.times (diagnostics)."""
        if not PROFILE:
            return
        ttnn.synchronize_device(self.device)
        now = time.perf_counter()
        self.times[name] = self.times.get(name, 0.0) + (now - self._t) * 1e3
        self._t = now


def prefill(moe, x, dump=None):
    """x [1,1,T,H] BF16 TILE -> [1,1,T,H] BF16 TILE.  dump (tests): [1,1,T,2E] FP32 row-major DRAM tensor that
    receives every row's FP32 [sel | scores]."""
    if x.shape[-2] % TILE:
        raise ValueError("prefill rows must be a multiple of 32")
    if x.memory_config() != DRAM:
        x = ttnn.to_memory_config(x, DRAM)
    if moe._prefill is None:
        moe._prefill = Prefill(moe)
    p = moe._prefill
    if PROFILE:
        ttnn.synchronize_device(p.device)
        p._t = time.perf_counter()
    return p(x, dump)
