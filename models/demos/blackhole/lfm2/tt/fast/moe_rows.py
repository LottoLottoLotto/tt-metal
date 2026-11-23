# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Multi-row decode MoE for LFM2 (64 experts, top-4, I = 1536) on one P150: B = 2..16 token rows, two launches.

The union of the rows' selected experts is streamed once (ascending expert ids); every row is computed against every
union expert with its own routing weight (zero where it did not select the expert: exact zero contributions), so each
row sums its own experts in ascending id order exactly as the one-row MoEDecodeV2 / MoEDecode do.

program 1 (kernels/moe_rows_router.cpp + moe_rows_experts.cpp stage 0):
  router role, 2 cores (one 32-expert column each): moe_decode2's bitwise router matmul on in0 = rows 0-15 of x, the
    two cores swap their logits tiles, the 4 router RISCs route contiguous row ranges (bitwise lfm_route, flags 3) and
    write each row's ids + BF16 weights into every gate/up worker's landing buffer (+ semaphore).
  gate/up workers, `gate_splits` per DRAM bank: union, per-row weight table, units (union expert, one gate/up column
    pair) round-robin; per row group of [tile_rows, 32] in0 tiles one transposed custom_mm + SiLU(g) * u * w_row
    (per-row weights by SFPU lane, batched_experts_kernel.cpp), the [tile_rows, 32] result into the L1-interleaved
    activation tensor (page (slot * groups + group) * banks + bank).
program 2 (moe_rows_experts.cpp stage 1): `down_splits` workers per bank stream their output columns of every union
  expert and accumulate them in FP32 DEST per row group (batched_experts_kernel.cpp's down role).

tile_rows = min(next power of two >= B, 8), groups = 1 (B <= 8) or 2 (B <= 16).  Same resident tensors as MoEDecodeV2
(expert banks, the `L{i}.router.stream-c1w1` router copy, Q24 bias, score table); no new weight copies.
"""
from __future__ import annotations

import ttnn

from models.common.modules.moe.batched_experts import _bank_workers
from models.common.modules.moe.selected_experts import _CachedProgram, _accessor_args, _row_ranges
from models.common.modules.moe.streaming_experts import _bank_args, _bank_cores, _brisc_accessors, _cb, _read_page, _scratch_cb
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from ..moe import ROUTING_FLAGS
from . import sigmoid_table
from .decode_matmul import PRESETS, cached_stream_weight, stream_weight
from .moe_decode import KERNELS, L1, score_table

LAND_CB = 7  # written remotely by the router cores (same index and address on every core of program 1)
ROUTE_SEM, CROSS_SEM = 0, 1


def row_tiling(rows: int) -> tuple[int, int]:
    """(tile_rows, groups) of a B-row bucket."""
    if not 1 <= rows <= 16:
        raise ValueError("MoEDecodeRows takes 1..16 rows")
    tr = 1
    while tr < min(rows, 8):
        tr *= 2
    return tr, -(-rows // tr)


class MoEDecodeRows:
    """x [1,1,B,H] BF16 tile (L1 or DRAM interleaved, 2 <= B <= 16) -> MoE output [1,1,B,H] BF16 tile (L1)."""

    def __init__(self, device, router, bias, gate_up, down, *, rows: int, hidden: int, intermediate: int,
                 num_experts: int, top_k: int, flags: int = ROUTING_FLAGS, router_stream=None, gate_splits: int = 3,
                 down_splits: int = 4, gate_buffers: int = 3, down_buffers: int = 6, router_kb: int = 8,
                 router_group: int = 4, bank_rows: str = "auto"):
        banks = device.dram_grid_size().x
        self.device, self.bias, self.gate_up, self.down = device, bias, gate_up, down
        self.rows, self.hidden, self.intermediate, self.E, self.top_k = rows, hidden, intermediate, num_experts, top_k
        self.banks = banks
        self.half = intermediate // 32 // banks
        self.tile_rows, self.groups = row_tiling(rows)
        if flags != 3:
            raise ValueError("MoEDecodeRows implements the flags-3 routing (BF16 logits and scores) only")
        if (intermediate % (32 * banks) or hidden % (32 * banks) or num_experts != 64 or top_k > 4
                or tuple(router.shape)[-2:] != (hidden, num_experts)):
            raise ValueError("unsupported MoE geometry for the decode streams")
        down_cols = hidden // 32 // banks
        if down_cols % down_splits or down_cols // down_splits * self.groups > 4:
            raise ValueError("down columns per core x row groups must fit four FP32 DEST slots")
        for t, k, n in ((gate_up, hidden, 2 * intermediate), (down, intermediate, hidden)):
            mc = t.memory_config()
            if (tuple(t.shape)[-2:] != (num_experts * k, n) or t.layout != ttnn.TILE_LAYOUT
                    or mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED or mc.buffer_type != ttnn.BufferType.DRAM):
                raise ValueError("expert weights must be the resident WIDTH_SHARDED DRAM bank layout")
        self.gate_splits, self.down_splits = gate_splits, down_splits
        self.gate_buffers, self.down_buffers = gate_buffers, down_buffers
        p = PRESETS["router"]
        if (p.cores_per_bank, p.w, p.k_split) != (1, 1, 2):
            raise ValueError("the router role reproduces the c1w1, k_split=2 router preset")
        self.router_w = router_stream if router_stream is not None else stream_weight(
            router, p, device=device, dtype=ttnn.bfloat16)
        kt = hidden // 32
        if kt % router_kb or router_kb % 2 or not 1 <= router_group <= 4:
            raise ValueError("bad router geometry")
        self.router_kb, self.router_group = router_kb, router_group
        self.selected = min(rows * top_k, num_experts)
        self.width = (self.selected + 1 + 31) // 32 * 32
        self.rows_per_risc = -(-rows // 4)
        self.senders = -(-rows // self.rows_per_risc)
        self.bank_cores = list(_bank_cores(device))
        if bank_rows == "auto":
            # one DRAM bank per worker row (moe_decode2.py: two banks' streams in one row ran one at ~3/4 speed)
            grid_y = device.compute_with_storage_grid_size().y
            for b, a in enumerate(self.bank_cores):
                used = {c.y for i, c in enumerate(self.bank_cores) if i != b}
                if a.y in {c.y for c in self.bank_cores[b + 1:]}:
                    y = min((y for y in range(grid_y) if y not in used), key=lambda y: (abs(y - a.y), y))
                    self.bank_cores[b] = ttnn.CoreCoord(a.x, y)
        self.table = score_table(device)
        self._progs = [_CachedProgram(), _CachedProgram()]
        self.keep_routing = False  # tests: __call__ also returns (logits, selection, routing)

    @classmethod
    def from_moe(cls, moe, rows: int, cache=None, layer: int | None = None, **kw) -> "MoEDecodeRows":
        cfg = moe.cfg
        if cache is not None:
            kw["router_stream"] = cached_stream_weight(cache, f"L{layer}.router", PRESETS["router"],
                                                       dtype=ttnn.bfloat16)
        return cls(moe.device, moe.router, moe.bias, moe.gate_up, moe.down, rows=rows, hidden=cfg.hidden,
                   intermediate=cfg.moe_intermediate, num_experts=cfg.n_experts, top_k=cfg.top_k, **kw)

    # ------------------------------------------------------------------ placement
    def _workers(self, splits):
        """[(core, bank, part, vc)]: per bank the bank core and its row neighbours (distinct VCs)."""
        _, vcs = _bank_args(self.bank_cores)
        rows = _bank_workers(self.device, self.bank_cores, splits)
        return [(row[p], bank, p, (vc + p) & 0x3) for bank, (row, (_, vc)) in enumerate(zip(rows, vcs))
                for p in range(splits)]

    def _routers(self, workers):
        """Router c beside a worker row, nearest to DRAM bank c's core (moe_decode2.py)."""
        grid = self.device.compute_with_storage_grid_size()
        taken = {(c.x, c.y) for c, *_ in workers}
        occupied = set(taken)
        routers = []
        for col in range(self.E // 32):
            a = self.bank_cores[col]
            free = [(x, y) for x in range(grid.x) for y in range(grid.y) if (x, y) not in taken]
            beside = [c for c in free if (c[0] - 1, c[1]) in occupied or (c[0] + 1, c[1]) in occupied]
            best = min(beside or free, key=lambda c: (abs(c[0] - a.x) + abs(c[1] - a.y), c[1], c[0]))
            taken.add(best)
            occupied.add(best)
            routers.append((ttnn.CoreCoord(*best), col))
        return routers

    # ------------------------------------------------------------------ programs
    def _common(self, fused, splits):
        kt = (self.hidden if fused else self.intermediate) // 32
        bank_cols = (2 * self.intermediate if fused else self.hidden) // self.banks // 32
        cols = 2 if fused else bank_cols // splits
        return kt, bank_cols, cols, [
            ("k_tiles", kt), ("bank_cols", bank_cols), ("core_cols", cols), ("splits", splits),
            ("banks", self.banks), ("top_k", self.top_k), ("rows", self.rows), ("tile_rows", self.tile_rows),
            ("row_groups", self.groups), ("metadata_width", self.width), ("inter_tiles", self.intermediate // 32),
            ("route_sem", ROUTE_SEM), ("senders", self.senders)]

    def _role_kernel(self, fused, workers, resident, brisc_tensors, writer_args):
        splits = self.gate_splits if fused else self.down_splits
        kt, bank_cols, cols, common = self._common(fused, splits)
        wt = resident.get_tile()
        tb = wt.get_tile_size(resident.dtype)
        run_bytes = kt * (1 if fused else cols) * tb  # gate/up: one column per run (two runs per unit)
        page = _read_page(run_bytes, tb)
        buffers = self.gate_buffers if fused else self.down_buffers
        block_tiles = kt * cols
        row_bytes = self.tile_rows * 64
        row_tile = ttnn.TileDescriptor(ttnn.Tile([self.tile_rows, 32]))
        cs = _row_ranges([c for c, *_ in workers])
        if fused:
            cbs = [_cb(cs, 0, ttnn.bfloat16, row_bytes, self.groups * kt, row_tile),
                   _cb(cs, 1, resident.dtype, tb, buffers * block_tiles, ttnn.TileDescriptor(wt)),
                   _scratch_cb(cs, 3, max(64, self.width * 2)),
                   _cb(cs, 4, ttnn.bfloat16, row_bytes, 2 * self.groups, row_tile),
                   _scratch_cb(cs, 5, self.rows * self.width * 2 + 64)]
        else:
            cbs = [_cb(cs, 0, ttnn.bfloat16, row_bytes, 4 * self.groups * kt, row_tile),
                   _cb(cs, 1, resident.dtype, tb, buffers * block_tiles, ttnn.TileDescriptor(wt)),
                   _scratch_cb(cs, 2, max(64, self.width * 2)),
                   _scratch_cb(cs, 3, cols * 2048),
                   _cb(cs, 4, ttnn.bfloat16, row_bytes, cols * self.groups, row_tile)]
        addresses = [t.buffer_address() for t in brisc_tensors]
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "moe_rows_experts.cpp"), core_ranges=cs,
            defines=[("MOE_STAGE", "0" if fused else "1")],
            ncrisc_named_compile_time_args=common + [
                ("weight_addr", resident.buffer_address()), ("weight_page_bytes", page),
                ("weight_pages", run_bytes // page), ("weight_run_bytes", run_bytes),
                ("weight_expert_bytes", kt * bank_cols * tb), ("weight_buffers", buffers)],
            brisc_compile_time_args=_brisc_accessors(brisc_tensors),
            brisc_named_compile_time_args=common,
            trisc_named_compile_time_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=[(core, [bank, vc, part]) for core, bank, part, vc in workers],
                brisc_args=[(core, addresses + [bank, part] + writer_args(bank, part))
                            for core, bank, part, _ in workers],
                trisc_args=[(core, [part]) for core, _, part, _ in workers]),
        )
        return kernel.get_kernel_descriptors().kernels, cbs

    def _router_kernel(self, routers, gate, x, logits, pos):
        kt = self.hidden // 32
        rows = 16  # in0 [16, 32] tiles: x rows 0-15
        in0_tile = ttnn.Tile([rows, 32])
        table_bytes = int(self.table.shape[-1]) * 2
        if table_bytes != 2 * (sigmoid_table.E_HI - sigmoid_table.E_LO + 1) * 128 * 2:
            raise ValueError("score table size does not match sigmoid_table.E_LO/E_HI")
        area = (table_bytes + self.E * 4 + 63) // 64 * 64
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _ in routers])

        def cb(index, dtype, page, pages, tile):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page,
                                                                                 ttnn.TileDescriptor(tile))])

        cbs = [cb(0, ttnn.bfloat16, rows * 64, kt // 2, in0_tile),
               cb(2, ttnn.bfloat16, rows * 64, kt // 2, in0_tile),
               cb(1, ttnn.bfloat16, 2048, kt, ttnn.Tile([32, 32])),
               _scratch_cb(cs, 3, 2 * area + 2048 + 2 * self.rows_per_risc * 32 + 2 * 64),  # + cur_pos per RISC
               cb(16, ttnn.float32, rows * 128, 1, in0_tile),
               cb(24, ttnn.float32, rows * 128, 1, in0_tile)]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        mode[24] = ttnn.UnpackToDestMode.UnpackToDestFp32  # exact reload of the FP32 partial
        compute.unpack_to_dest_mode.extend(mode)
        common = [("kt", kt), ("kb", self.router_kb), ("ks", 2), ("group", self.router_group)]
        dm = common + [("top_k", self.top_k), ("experts", self.E), ("score_e_lo", sigmoid_table.E_LO),
                       ("score_e_hi", sigmoid_table.E_HI), ("out_face_bytes", rows * 64),
                       ("in0_tile_bytes", rows * 64), ("rows", self.rows), ("rows_per_risc", self.rows_per_risc),
                       ("route_sem", ROUTE_SEM), ("cross_sem", CROSS_SEM), ("has_pos", int(pos is not None))]
        named_t = [("x", x), ("logits", logits), ("table", self.table), ("bias", self.bias)]
        if pos is not None:
            named_t.append(("pos", pos))
        acc = {name: _accessor_args(t) for name, t in named_t}

        def accessors(names):
            ct, named = [], []
            for name in names:
                if name in acc:
                    named.append((f"{name}_cta", len(ct)))
                    ct += acc[name]
            if pos is None:  # the kernel names pos_cta in a discarded branch: alias the bias accessor
                named.append(("pos_cta", dict(named)["bias_cta"]))
            return ct, named

        nc_ct, nc_named = accessors(("x", "table", "bias", "pos"))
        br_ct, br_named = accessors(("x", "logits", "table", "bias", "pos"))
        addrs = [int(t.buffer_address()) for t in (self.router_w, x, logits, self.table, self.bias)]
        addrs.append(int(pos.buffer_address()) if pos is not None else 0)
        noc = [self.device.worker_core_from_logical_core(c) for c, *_ in gate]
        wxy = [len(noc)] + [v for n in noc for v in (n.x, n.y)]
        rnoc = [self.device.worker_core_from_logical_core(c) for c, _ in routers]
        per_core = []
        for (core, col), _ in zip(routers, rnoc):
            other = rnoc[1 - col]
            per_risc = [other.x, other.y] + wxy
            per_core.append((core, [col] + per_risc + per_risc))
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "moe_rows_router.cpp"), core_ranges=cs,
            ncrisc_compile_time_args=nc_ct, ncrisc_named_compile_time_args=dm + nc_named,
            brisc_compile_time_args=br_ct, brisc_named_compile_time_args=dm + br_named,
            trisc_named_compile_time_args=common,
            ncrisc_common_runtime_args=addrs, brisc_common_runtime_args=addrs,
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=per_core, brisc_args=per_core, trisc_args=[(c, [col]) for c, col in routers]),
        )
        return kernel.get_kernel_descriptors().kernels, cbs

    def _program1(self, tensors, pos):
        x, logits, sel, routing, act = tensors[:5]
        gate = self._workers(self.gate_splits)
        routers = self._routers(gate)
        gk, gcbs = self._role_kernel(True, gate, self.gate_up, (x, sel, routing, act),
                                     lambda bank, part: [int(bank == 0 and part == 0)])
        rk, rcbs = self._router_kernel(routers, gate, x, logits, pos)
        every = _row_ranges([c for c, *_ in gate] + [c for c, _ in routers])
        # landing buffer first on every core: the same L1 address everywhere
        land = _scratch_cb(every, LAND_CB, max(64, self.rows * 32))
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=every, initial_value=0) for s in (ROUTE_SEM, CROSS_SEM)]
        return ttnn.ProgramDescriptor(kernels=gk + rk, cbs=[land] + gcbs + rcbs, semaphores=sems)

    def _program2(self, tensors):
        act, sel, out = tensors[:3]
        down = self._workers(self.down_splits)
        dk, dcbs = self._role_kernel(False, down, self.down, (act, sel, out), lambda bank, part: [])
        return ttnn.ProgramDescriptor(kernels=dk, cbs=dcbs, semaphores=[])

    # ------------------------------------------------------------------ forward
    def forward(self, x, cur_pos=None):
        """-> (out, logits FP32 [1,1,B,E], selection UINT16 [1,1,1,width] (union ids, count at width - 1), routing
        BF16 [1,1,B,width] (row weights per union slot)).  cur_pos (optional, the step's [B] int32): rows with
        cur_pos < 0 select no expert: their output rows are zero while any row is active (all rows inactive: the
        output is undefined); logits are still computed."""
        B = self.rows
        if (tuple(x.shape) != (1, 1, B, self.hidden) or x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT
                or x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            raise ValueError(f"MoEDecodeRows takes interleaved BF16 tiles [1,1,{B},{self.hidden}]")
        dev = self.device
        logits = ttnn.empty([1, 1, B, self.E], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=L1)
        sel = ttnn.empty([1, 1, 1, self.width], dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                         memory_config=L1)
        routing = ttnn.empty([1, 1, B, self.width], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                             memory_config=L1)
        act = ttnn.empty([1, 1, self.selected * self.groups * self.banks, self.tile_rows * self.intermediate // self.banks],
                         dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=L1)
        t1 = [x, logits, sel, routing, act, self.gate_up, self.router_w, self.bias, self.table]
        if cur_pos is not None:
            t1.append(cur_pos)
        self._progs[0].run(t1, lambda: self._program1(t1, cur_pos))
        out = ttnn.empty([1, 1, B, self.hidden], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=L1)
        t2 = [act, sel, out, self.down]
        self._progs[1].run(t2, lambda: self._program2(t2))
        ttnn.deallocate(act)
        return out, logits, sel, routing

    def __call__(self, x, cur_pos=None):
        out, logits, sel, routing = self.forward(x, cur_pos)
        if self.keep_routing:
            return out, logits, sel, routing
        for t in (logits, sel, routing):
            ttnn.deallocate(t)
        return out
