# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM2 MoE prefill programs.

`GroupedExpertsM`: models.common.modules.moe GroupedExperts' inputs and per-tile math (bitwise values) with row-group
reuse: every (expert, DRAM bank) weight column block is streamed once per item of up to `max_rows` consecutive 32-row
groups of that expert instead of once per group, the work is spread over the whole compute grid, and the down
projection writes row-major FP32 rows (kernels/grouped_experts_m.cpp).

`CombineRows`: the stock grouped combine program on those rows (grouped_combine minus its to_layout).

`RoutePrefill`: drop-in for moe.SigmoidGroupedRouting on whole 32-row tiles (same packed dict, bitwise): the
sigmoid routing and the grouped prefix run as one many-core program (kernels/moe_route_prefill.cpp) instead of a
routing program plus the single-core stock prefix; the stock pack follows.
"""
from __future__ import annotations

from pathlib import Path

import ttnn

from models.common.modules.moe.grouped_routing import _empty, _pack_program
from models.common.modules.moe.selected_experts import _CachedProgram, _accessor_args
from models.common.modules.moe.streaming_experts import _brisc_accessors, _cb, _read_page, _scratch_cb
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

KERNELS = Path(__file__).resolve().parent / "kernels"


def _grid_cores(device):
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]


def _ranges(cores) -> ttnn.CoreRangeSet:
    rows = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    out = []
    for y, xs in sorted(rows.items()):
        xs.sort()
        start = prev = xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                out.append(ttnn.CoreRange(ttnn.CoreCoord(start, y), ttnn.CoreCoord(prev, y)))
                start = x
            prev = x
        out.append(ttnn.CoreRange(ttnn.CoreCoord(start, y), ttnn.CoreCoord(prev, y)))
    return ttnn.CoreRangeSet(out)


class GroupedExpertsM:
    """packed x [1,1,G*32,H] BF16 tiles, packed route weights [1,1,G*32,32], grouped-routing metadata row -> FP32
    [1,1,G*32,H] per-assignment outputs (GroupedExperts' values) as row-major rows, the layout CombineRows reads (no
    to_layout).  max_rows: row groups per item (gate/up, down; down <= 4: one matmul_block in half the FP32 DEST)."""

    def __init__(self, device, gate_up, down, hidden: int, intermediate: int, max_rows=(6, 4)):
        self.device, self.gate_up, self.down = device, gate_up, down
        self.hidden, self.intermediate = hidden, intermediate
        self.banks = device.dram_grid_size().x
        self.cores = _grid_cores(device)
        self.max_rows = max_rows
        # balancing cost of one (item, bank) weight stream in eighths of a 32-row group's cost (gate/up: 442 KB of
        # weights vs 128 KB of x per group; down: 221 KB vs 96 KB of x + 32 KB of FP32 rows; swept 0-48: the
        # traffic ratios measure best within ~4%)
        self.unit_cost = (40, 20)
        self._programs = [None, None]
        self._signatures = [None, None]

    def _build(self, stage, x, route_weights, experts, out, resident):
        fused = stage == 0
        kt = x.shape[-1] // 32
        nt = out.shape[-1] // 32
        per_bank_out = nt // self.banks
        block_k = min(16, kt)
        if kt % block_k:
            raise ValueError("projection K must divide into 16-tile streaming blocks")
        weight_tile = resident.get_tile()
        weight_bytes = weight_tile.get_tile_size(resident.dtype)
        block_bytes = block_k * weight_bytes
        page = _read_page(block_bytes, weight_bytes)
        col_tiles = kt * (2 if fused else 1)
        untilize = stage == 1
        res_tiles = (per_bank_out * kt) if untilize else col_tiles
        max_rows = self.max_rows[stage]
        tile = ttnn.TileDescriptor(ttnn.Tile([32, 32]))
        output_bytes = ttnn.Tile([32, 32]).get_tile_size(out.dtype)
        meta_bytes = experts.shape[-1] * 4
        cs = _ranges(self.cores)
        cbs = [
            _cb(cs, 0, ttnn.bfloat16, 2048, max_rows * kt, tile),
            _cb(cs, 1, resident.dtype, weight_bytes, 2 * res_tiles, ttnn.TileDescriptor(weight_tile)),
            _scratch_cb(cs, 2, meta_bytes),
            _cb(cs, 4, out.dtype, output_bytes, max_rows * per_bank_out if untilize else 2, tile),
        ]
        if fused:
            cbs.append(_cb(cs, 5, ttnn.bfloat16, 2048, max_rows, tile))
        common = [
            ("num_tiles_k", kt),
            ("out_tiles", nt),
            ("per_bank_out", per_bank_out),
            ("subblock_k", block_k),
            ("fused_swiglu", int(fused)),
            ("banks", self.banks),
            ("max_rows", max_rows),
            ("workers", len(self.cores)),
            ("untilize_out", int(untilize)),
            ("unit_cost", self.unit_cost[stage]),
        ]
        inputs = (x, experts, out, route_weights)
        addrs = [t.buffer_address() for t in inputs]
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "grouped_experts_m.cpp"),
            core_ranges=cs,
            ncrisc_named_compile_time_args=common + [
                ("weight_addr", resident.buffer_address()),
                ("weight_page_bytes", page),
                ("weight_pages", block_bytes // page),
                ("weight_block_bytes", block_bytes),
            ],
            brisc_compile_time_args=_brisc_accessors(inputs),
            brisc_named_compile_time_args=common + [("metadata_bytes", meta_bytes)],
            trisc_named_compile_time_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
            ),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=[(c, [w, w & 3]) for w, c in enumerate(self.cores)],
                brisc_args=[(c, addrs + [w]) for w, c in enumerate(self.cores)],
                trisc_args=[(c, [w]) for w, c in enumerate(self.cores)],
            ),
        )
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, packed_x, packed_weights, experts):
        rows = packed_x.shape[-2]
        activated = ttnn.empty([1, 1, rows, self.intermediate], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.empty([1, 1, rows, self.hidden], dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                         device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for stage, x, result, resident in ((0, packed_x, activated, self.gate_up), (1, activated, out, self.down)):
            tensors = [x, experts, result, resident, packed_weights]
            signature = tuple((int(t.buffer_address()), tuple(t.shape)) for t in tensors)
            if self._programs[stage] is None or signature != self._signatures[stage]:
                self._programs[stage] = self._build(stage, x, packed_weights, experts, result, resident)
                self._signatures[stage] = signature
            ttnn.generic_op(tensors, self._programs[stage])
        ttnn.deallocate(activated)
        return out


class CombineRows:
    """grouped_combine without its to_layout: FP32 row-major per-assignment rows [1,1,G*32,H] + inverse [1,1,T,32]
    -> BF16 [1,1,T,H] tiles (the stock grouped_routing_combine.cpp program: FP32 sum in slot order, one BF16
    rounding; bitwise grouped_combine on the same values).  Programs are cached per tensor addresses."""

    def __init__(self, device, top_k: int, chunk: int = 4):
        self.device, self.selected, self.chunk = device, top_k, chunk
        self._program = _CachedProgram()

    def _build(self, tensors, tokens, hidden):
        from models.common.modules.moe.grouped_routing import _KERNELS as STOCK_KERNELS, _workers
        from models.common.modules.moe.selected_experts import _cb as tile_cb

        width_tiles = hidden // 32
        chunk = self.chunk
        items = ((tokens + 31) // 32) * (width_tiles // chunk)
        cores, cs = _workers(self.device, items)
        args = []
        for t in tensors:
            args.extend(_accessor_args(t))
        addresses = [t.buffer_address() for t in tensors]
        runtime = [(c, addresses + [i, len(cores)]) for i, c in enumerate(cores)]
        named = [("tokens", tokens), ("width_tiles", width_tiles), ("selected", self.selected), ("chunk", chunk)]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        unpack = [ttnn.UnpackToDestMode.Default] * 64
        unpack[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode.extend(unpack)
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(STOCK_KERNELS / "grouped_routing_combine.cpp"), core_ranges=cs,
            brisc_compile_time_args=args, ncrisc_compile_time_args=args,
            brisc_named_compile_time_args=named, ncrisc_named_compile_time_args=named,
            trisc_named_compile_time_args=named, trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=runtime, ncrisc_args=runtime, trisc_args=runtime))
        tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))

        def buffer(index, dtype, page, pages):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        return ttnn.ProgramDescriptor(
            kernels=kernel.get_kernel_descriptors().kernels,
            cbs=[buffer(0, ttnn.float32, 4096, 2 * self.selected * chunk), tile_cb(1, 2, cs),
                 buffer(2, ttnn.bfloat16, 2048, chunk), buffer(16, ttnn.bfloat16, 2048, 2 * chunk)],
            semaphores=[])

    def __call__(self, rows, inverse):
        tokens, hidden = inverse.shape[-2], rows.shape[-1]
        out = _empty(self.device, [1, 1, tokens, hidden])
        tensors = [rows, inverse, out]
        self._program.run(tensors, lambda: self._build(tensors, tokens, hidden))
        return out


class RoutePrefill:
    """x [1,1,T,H] BF16 tiles (T % 32 == 0), FP32 logits [1,1,T,E], Q24 bias page -> the grouped-routing packed dict
    (x, weights, experts, inverse) of SigmoidGroupedRouting (routing flags 3 only)."""

    def __init__(self, device, num_experts: int, top_k: int):
        from .fast import sigmoid_table
        from .fast.moe_decode import score_table

        if num_experts % 32 or num_experts > 64 or top_k > 4:
            raise ValueError("RoutePrefill supports E <= 64 (multiple of 32), top_k <= 4")
        self.device, self.num_experts, self.top_k = device, num_experts, top_k
        self.table = score_table(device)
        self.e_lo, self.e_hi = sigmoid_table.E_LO, sigmoid_table.E_HI
        self._shape = None

    def capacity(self, tokens: int) -> int:
        return min(tokens * self.top_k, (tokens * self.top_k + 31) // 32 + self.num_experts)

    def _program(self, tensors, tokens, expert_width):
        grid = self.device.compute_with_storage_grid_size()
        n = grid.x * grid.y  # every core: token shares for routing, the first E also own an expert's groups
        if n < self.num_experts:
            raise ValueError("RoutePrefill needs at least one core per expert")
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]
        cs = _ranges(cores)
        noc = [self.device.worker_core_from_logical_core(c) for c in cores]
        table_bytes = int(self.table.shape[-1]) * 2
        max_tokens = -(-tokens // n)
        max_groups = -(-tokens // 32)  # an expert takes each token at most once
        args = [self.num_experts, self.top_k, tokens, n, self.e_lo, self.e_hi, expert_width, table_bytes,
                max_tokens, max_groups]
        for t in tensors:
            args.extend(_accessor_args(t))
        addrs = [t.buffer_address() for t in tensors]
        everyone = [v for c in noc for v in (c.x, c.y)]
        runtime = [(c, addrs + [i] + everyone) for i, c in enumerate(cores)]
        E, k = self.num_experts, self.top_k
        scratch = (E * 4 + k * max_tokens * 16 + (E // 32) * 4096 + ((table_bytes + 63) & ~63) + 2 * E * 4
                   + max_tokens * 16 + n * E * 4 + 2 * E * 4 + max(max_tokens * 128, max_groups * 256)
                   + ((expert_width * 4 + 63) & ~63) + tokens * 16 + 256)
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "moe_route_prefill.cpp"), core_ranges=cs, compile_time_args=args,
            runtime_args=runtime, config=ttnn.ReaderConfigDescriptor())
        cb = ttnn.CBDescriptor(total_size=(scratch + 2047) // 2048 * 2048, core_ranges=cs,
                               format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048)])
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cs, initial_value=0) for s in range(2)]
        return ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb], semaphores=sems)

    def __call__(self, x, logits, bias) -> dict:
        tokens, hidden = tuple(x.shape)[-2:]
        if tokens % 32 or tuple(logits.shape) != (1, 1, tokens, self.num_experts) or logits.dtype != ttnn.float32:
            raise ValueError(f"need whole 32-row tiles and FP32 logits [1,1,{tokens},{self.num_experts}]")
        if self._shape != (tokens, hidden):
            self._route, self._pack = _CachedProgram(), _CachedProgram()
            self._shape = (tokens, hidden)
        groups = self.capacity(tokens)
        expert_width = ((groups + 32) // 32) * 32
        rm = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        packed = _empty(self.device, [1, 1, groups * 32, hidden])
        weights = _empty(self.device, [1, 1, groups * 32, 32])
        experts = _empty(self.device, [1, 1, 1, expert_width], **rm)
        inverse = _empty(self.device, [1, 1, tokens, 32], **rm)
        group_map = _empty(self.device, [1, 1, groups, 64], **rm)
        tensors = [logits, bias, self.table, group_map, experts, inverse]
        self._route.run(tensors, lambda: self._program(tensors, tokens, expert_width))
        rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pack_tensors = [rows, group_map, experts, packed, weights]
        self._pack.run(pack_tensors, lambda: _pack_program(self.device, pack_tensors, groups, hidden, expert_width))
        for t in (rows, group_map):
            ttnn.deallocate(t)
        return {"x": packed, "weights": weights, "experts": experts, "inverse": inverse}
