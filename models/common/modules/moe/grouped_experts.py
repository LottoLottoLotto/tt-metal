# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-tile, expert-grouped prefill using the resident decode weight banks.

Each metadata entry selects one expert and 32 packed token rows. Gate/up and down
stream that expert's bank-shuffled BFP4 columns once for all 32 rows. The routed
SwiGLU stays FP32 in DEST until the weighted intermediate is packed to BF16;
down preserves FP32 assignment outputs for the subsequent expert reduction.
Only the device-side actual group count is processed, not the capacity padding.
Each DRAM bank feeds several row-neighbour workers that take the groups round-robin:
one full-tile worker per bank is compute-bound, several keep up with the weight stream.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from models.common.modules.moe.batched_experts import _bank_workers
from models.common.modules.moe.selected_experts import _row_ranges
from models.common.modules.moe.streaming_experts import (
    _bank_args,
    _bank_cores,
    _brisc_accessors,
    _cb,
    _read_page,
    _scratch_cb,
)
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = Path(__file__).resolve().parent / "kernels" / "streaming" / "grouped_experts_kernel.cpp"
_SPLITS = 4  # workers per DRAM bank sharing the groups


class GroupedExperts:
    """Project packed groups with the existing bank-shuffled gate/up and down tensors."""

    def __init__(self, device, gate_up, down, hidden, intermediate, stored_experts):
        self.device = device
        self.gate_up = gate_up
        self.down = down
        self.hidden = hidden
        self.intermediate = intermediate
        self.cores = _bank_cores(device)
        self.banks = device.dram_grid_size().x
        if len(self.cores) != self.banks:
            raise ValueError("grouped experts require one worker per DRAM bank")
        if hidden % (32 * self.banks) or intermediate % (32 * self.banks):
            raise ValueError("hidden and intermediate dimensions must split into whole tiles per bank")
        for tensor, k, n in ((gate_up, hidden, 2 * intermediate), (down, intermediate, hidden)):
            if tuple(tensor.shape)[-2:] != (stored_experts * k, n):
                raise ValueError(f"resident weight shape must end in [{stored_experts * k}, {n}]")
            config = tensor.memory_config()
            if (tensor.dtype != ttnn.bfloat4_b or tensor.layout != ttnn.TILE_LAYOUT
                    or config.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
                    or config.buffer_type != ttnn.BufferType.DRAM):
                raise ValueError("grouped experts require resident WIDTH_SHARDED DRAM BFP4 weights")
        self.splits = _SPLITS
        self._programs = [None, None]
        self._signatures = [None, None]

    @staticmethod
    def _check_tiles(tensor, shape, name):
        config = tensor.memory_config()
        if (tuple(tensor.shape) != shape or tensor.dtype != ttnn.bfloat16
                or tensor.layout != ttnn.TILE_LAYOUT
                or config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
                or config.buffer_type != ttnn.BufferType.DRAM):
            raise ValueError(f"{name} must be interleaved DRAM BF16 tiles {shape}")

    def _build(self, x, route_weights, experts, out, resident, fused):
        kt = x.shape[-1] // 32
        nt = out.shape[-1] // 32
        per_bank_out = nt // self.banks
        # Three 16-tile weight blocks pipeline NCRISC reads against full-tile math.
        # At H=2048, all CBs together use <170 KiB on each bank worker.
        subblock_k = min(16, kt)
        if kt % subblock_k:
            raise ValueError("projection K must divide into 16-tile streaming blocks")
        weight_tile = resident.get_tile()
        weight_bytes = weight_tile.get_tile_size(resident.dtype)
        block_bytes = subblock_k * weight_bytes
        page = _read_page(block_bytes, weight_bytes)
        tile = ttnn.TileDescriptor(ttnn.Tile([32, 32]))
        output_bytes = out.get_tile().get_tile_size(out.dtype)
        _, vcs = _bank_args(self.cores)
        # (core, bank, part, vc); a bank's workers use distinct request VCs.
        workers = [
            (row[p], bank, p, (vc + p) & 0x3)
            for bank, (row, (_, vc)) in enumerate(zip(_bank_workers(self.device, self.cores, self.splits), vcs))
            for p in range(self.splits)
        ]
        cs = _row_ranges([core for core, *_ in workers])
        cbs = [
            _cb(cs, 0, ttnn.bfloat16, 2048, kt, tile),
            _cb(cs, 1, resident.dtype, weight_bytes, 3 * subblock_k, ttnn.TileDescriptor(weight_tile)),
            _scratch_cb(cs, 2, 64),  # first metadata cache line, including runtime group count
            _scratch_cb(cs, 3, 64),  # current expert's metadata cache line
            _cb(cs, 4, out.dtype, output_bytes, 2, tile),
        ]
        if fused:
            cbs.append(_cb(cs, 5, ttnn.bfloat16, 2048, 1, tile))
        common = [
            ("num_tiles_k", kt),
            ("out_tiles", nt),
            ("per_bank_out", per_bank_out),
            ("subblock_k", subblock_k),
            ("fused_swiglu", int(fused)),
            ("splits", self.splits),
        ]
        # Both specializations share the accessor signature: C++ checks the
        # non-template if-constexpr branch even when down does not read routing.
        inputs = (x, experts, out, route_weights)
        runtime = [tensor.buffer_address() for tensor in inputs]
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(_KERNEL),
            core_ranges=cs,
            ncrisc_named_compile_time_args=common + [
                ("weight_addr", resident.buffer_address()),
                ("weight_page_bytes", page),
                ("weight_pages", block_bytes // page),
                ("weight_block_bytes", block_bytes),
            ],
            brisc_compile_time_args=_brisc_accessors(inputs),
            brisc_named_compile_time_args=common + [("metadata_bytes", experts.shape[-1] * 4)],
            trisc_named_compile_time_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
            ),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=[(core, [bank, vc, part]) for core, bank, part, vc in workers],
                brisc_args=[(core, runtime + [bank, part]) for core, bank, part, _ in workers],
                trisc_args=[(core, [part]) for core, _, part, _ in workers],
            ),
        )
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, packed_x, packed_weights, experts):
        rows = packed_x.shape[-2]
        if rows == 0 or rows % 32:
            raise ValueError("packed token capacity must be a positive multiple of 32")
        self._check_tiles(packed_x, (1, 1, rows, self.hidden), "packed_x")
        self._check_tiles(packed_weights, (1, 1, rows, 32), "packed_weights")
        metadata_width = ((rows // 32 + 1 + 31) // 32) * 32
        config = experts.memory_config()
        if (tuple(experts.shape) != (1, 1, 1, metadata_width) or experts.dtype != ttnn.uint32
                or experts.layout != ttnn.ROW_MAJOR_LAYOUT
                or config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
                or config.buffer_type != ttnn.BufferType.DRAM):
            raise ValueError("experts must be the interleaved DRAM UINT32 grouped-routing metadata row")
        activated = ttnn.empty(
            [1, 1, rows, self.intermediate], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.empty(
            [1, 1, rows, self.hidden], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for stage, x, result, resident in (
            (0, packed_x, activated, self.gate_up),
            (1, activated, out, self.down),
        ):
            tensors = [x, experts, result, resident, packed_weights]
            signature = tuple((int(t.buffer_address()), tuple(t.shape)) for t in tensors)
            if self._programs[stage] is None or signature != self._signatures[stage]:
                self._programs[stage] = self._build(x, packed_weights, experts, result, resident, stage == 0)
                self._signatures[stage] = signature
            ttnn.generic_op(tensors, self._programs[stage])
        ttnn.deallocate(activated)
        return out
