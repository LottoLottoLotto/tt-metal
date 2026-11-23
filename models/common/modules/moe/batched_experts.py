# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small-row union-expert decode using the existing native bank-shuffled weights.

A selected expert is read once per projection across the entire bucket. Each DRAM bank
feeds several neighbouring worker cores in its own row (placement near the bank matters:
a compact rectangle measured ~45% slower) so the math keeps up with the weight stream;
one core per bank is unpack-bound beyond ~8 rows:

- gate/up: the bank's cores take the union's experts round-robin; each streams whole
  expert bank blocks and applies them to every row group (B16: two 8-row subtiles).
- down: the bank's cores split its output columns; each streams its column slice of
  every expert once and accumulates all experts in FP32 DEST, so every row sums its
  experts in union order exactly as the one-row path does.

The compact activation between the stages is L1-interleaved in pages of one gate/up
bank's intermediate tiles per (expert, row group), which spreads each expert's down
input across many source cores. For H=2048/I=512/E=256/top_k=8 it is at most 2.02 MiB;
per-core CBs stay under 600 KiB (gate/up) and 340 KiB (down). No additional persistent
weight tensor or device state is used. Program descriptors specialize on bucket and
addresses; warm each used bucket before trace capture. Large prefill deliberately
remains GroupedExperts.
"""
from __future__ import annotations

from pathlib import Path

import ttnn

from models.common.modules.moe.selected_experts import _row_ranges
from models.common.modules.moe.streaming_experts import (
    _bank_args, _bank_cores, _brisc_accessors, _cb, _read_page, _scratch_cb,
)
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor,
)

_KERNEL = Path(__file__).resolve().parent / "kernels" / "streaming" / "batched_experts_kernel.cpp"
_GATE_SPLITS = 3  # cores per bank sharing the union's gate/up experts (measured best of 2-5)
_DOWN_ACCUMULATORS = 4  # FP32 DEST slots in half-sync: columns per core x row groups


def _bank_workers(device, bank_cores, splits):
    """``splits`` distinct worker cores per bank: the bank's own core, then its row neighbours."""
    grid = device.compute_with_storage_grid_size()
    taken = {(c.x, c.y) for c in bank_cores}
    workers = []
    for core in bank_cores:
        chosen = [core]
        for x in [*range(core.x + 1, grid.x), *range(core.x - 1, -1, -1)]:
            if len(chosen) == splits:
                break
            if (x, core.y) not in taken:
                taken.add((x, core.y))
                chosen.append(ttnn.CoreCoord(x, core.y))
        if len(chosen) != splits:
            raise ValueError(f"cannot place {splits} batched-expert workers beside bank core {core}")
        workers.append(chosen)
    return workers


class BatchedExperts:
    """BF16 [1,1,B,H] plus BatchedTopKRouting metadata -> BF16 [1,1,B,H]."""

    def __init__(self, device, gate_up, down, *, hidden, intermediate, stored_experts, top_k, rows):
        if rows not in (1, 2, 4, 8, 16):
            raise ValueError("batched experts require a 1/2/4/8/16-row bucket")
        self.device, self.gate_up, self.down = device, gate_up, down
        self.hidden, self.intermediate, self.rows = hidden, intermediate, rows
        self.tile_rows = min(rows, 8)
        self.row_groups = rows // self.tile_rows
        self.selected = rows * top_k + 1
        self.metadata_width = ((self.selected + 1 + 31) // 32) * 32
        bank_cores = _bank_cores(device)
        banks = device.dram_grid_size().x
        if len(bank_cores) != banks or hidden % (32 * banks) or intermediate % (32 * banks):
            raise ValueError("batched experts require whole hidden/intermediate tiles per DRAM bank")
        self.banks = banks
        if 2 * intermediate // banks // 32 > 4:
            raise ValueError("gate/up bank columns must fit four FP32 DEST slots")
        down_cols = hidden // banks // 32
        # Fewest down cores whose columns x row groups fit the FP32 accumulators.
        down_splits = next(
            (s for s in range(1, down_cols + 1)
             if down_cols % s == 0 and down_cols // s * self.row_groups <= _DOWN_ACCUMULATORS),
            None,
        )
        if down_splits is None:
            raise ValueError("down bank columns cannot be split into FP32 DEST accumulators")
        self.splits = (_GATE_SPLITS, down_splits)
        for tensor, k, n in ((gate_up, hidden, 2 * intermediate), (down, intermediate, hidden)):
            config = tensor.memory_config()
            if (tuple(tensor.shape)[-2:] != (stored_experts * k, n)
                    or tensor.dtype != ttnn.bfloat4_b or tensor.layout != ttnn.TILE_LAYOUT
                    or config.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
                    or config.buffer_type != ttnn.BufferType.DRAM):
                raise ValueError("batched experts require the resident bank-shuffled BFP4 weights")
        self._bank_cores = bank_cores
        self._programs = [None, None]
        self._signatures = [None, None]

    def _build(self, x, selection, routing, out, resident, fused):
        splits = self.splits[0 if fused else 1]
        _, vcs = _bank_args(self._bank_cores)
        # (core, bank, part, vc); a bank's cores use distinct request VCs.
        workers = [
            (row[p], bank, p, (vc + p) & 0x3)
            for bank, (row, (_, vc)) in enumerate(zip(_bank_workers(self.device, self._bank_cores, splits), vcs))
            for p in range(splits)
        ]
        kt = (self.hidden if fused else self.intermediate) // 32
        per_core_n = (2 * self.intermediate if fused else self.hidden) // self.banks // 32
        cols = per_core_n if fused else per_core_n // splits
        weight_tile = resident.get_tile()
        weight_bytes = weight_tile.get_tile_size(resident.dtype)
        block_bytes = kt * cols * weight_bytes
        page = _read_page(block_bytes, weight_bytes)
        row_tile = ttnn.TileDescriptor(ttnn.Tile([self.tile_rows, 32]))
        row_bytes = self.tile_rows * 64
        cs = _row_ranges([core for core, *_ in workers])
        groups_kt = self.row_groups * kt
        if fused:
            cbs = [
                _cb(cs, 0, ttnn.bfloat16, row_bytes, groups_kt, row_tile),
                _cb(cs, 1, resident.dtype, weight_bytes, 3 * kt * cols, ttnn.TileDescriptor(weight_tile)),
                _scratch_cb(cs, 2, self.metadata_width * 2),
                _scratch_cb(cs, 3, kt * max(self.rows, 2) * 64),
                _cb(cs, 4, ttnn.bfloat16, row_bytes, self.row_groups * per_core_n, row_tile),
                _scratch_cb(cs, 5, self.rows * self.metadata_width * 2),
            ]
        else:
            cbs = [
                _cb(cs, 0, ttnn.bfloat16, row_bytes, 4 * groups_kt, row_tile),
                _cb(cs, 1, resident.dtype, weight_bytes, 8 * kt * cols, ttnn.TileDescriptor(weight_tile)),
                _scratch_cb(cs, 2, self.metadata_width * 2),
                _scratch_cb(cs, 3, cols * 2048),
                _cb(cs, 4, ttnn.bfloat16, row_bytes, cols * self.row_groups, row_tile),
            ]
        common = [
            ("k_tiles", kt), ("per_core_n", per_core_n), ("core_cols", cols), ("splits", splits),
            ("banks", self.banks), ("rows", self.rows), ("tile_rows", self.tile_rows),
            ("row_groups", self.row_groups), ("selected", self.selected),
            ("metadata_width", self.metadata_width), ("fused_swiglu", int(fused)),
            ("inter_tiles", self.intermediate // 32),
        ]
        tensors = (x, selection, routing, out)
        addresses = [t.buffer_address() for t in tensors]
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(_KERNEL), core_ranges=cs,
            ncrisc_named_compile_time_args=common + [
                ("weight_addr", resident.buffer_address()), ("weight_page_bytes", page),
                ("weight_pages", block_bytes // page), ("weight_block_bytes", block_bytes),
                ("weight_expert_bytes", kt * per_core_n * weight_bytes),
            ],
            brisc_compile_time_args=_brisc_accessors(tensors),
            brisc_named_compile_time_args=common,
            trisc_named_compile_time_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False,
            ),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=[(core, [bank, vc, part]) for core, bank, part, vc in workers],
                brisc_args=[(core, addresses + [bank, part]) for core, bank, part, _ in workers],
                trisc_args=[(core, [part]) for core, _, part, _ in workers],
            ),
        )
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, x, selection, routing):
        if (tuple(x.shape) != (1, 1, self.rows, self.hidden) or x.dtype != ttnn.bfloat16
                or x.layout != ttnn.TILE_LAYOUT or tuple(x.get_tile().tile_shape) != (32, 32)
                or x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            raise ValueError("batched x must be interleaved BF16 32x32 tiles [1,1,B,H]")
        for tensor, shape, dtype in (
            (selection, (1, 1, 1, self.metadata_width), ttnn.uint16),
            (routing, (1, 1, self.rows, self.metadata_width), ttnn.bfloat16),
        ):
            if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                    or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
                    or tensor.memory_config() != ttnn.L1_MEMORY_CONFIG):
                raise ValueError("selection and routing must come from BatchedTopKRouting")
        # Page (expert * row_groups + group) * banks + bank: that gate/up bank's
        # intermediate row tiles for one expert and row group.
        activated = ttnn.empty(
            [1, 1, self.selected * self.row_groups * self.banks, self.tile_rows * self.intermediate // self.banks],
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = ttnn.empty(
            [1, 1, self.rows, self.hidden], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for stage, source, result, resident in (
            (0, x, activated, self.gate_up), (1, activated, out, self.down),
        ):
            tensors = [source, selection, routing, result, resident]
            signature = tuple((int(t.buffer_address()), str(t.memory_config())) for t in tensors)
            if self._programs[stage] is None or signature != self._signatures[stage]:
                self._programs[stage] = self._build(source, selection, routing, result, resident, stage == 0)
                self._signatures[stage] = signature
            ttnn.generic_op(tensors, self._programs[stage])
        ttnn.deallocate(activated)
        return out
