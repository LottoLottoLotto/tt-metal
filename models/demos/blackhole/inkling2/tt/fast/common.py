# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the fused decode kernels (tt/fast): cached generic_op programs, core ranges, CB descriptors.

Ported from the LFM2.5 port (lfm2/tt/fast/conv_decode.py).  Owner: AttnFast (imported by GlueFast / HeadFast).
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

KERNELS = Path(__file__).resolve().parent.parent / "kernels" / "fast"
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048, ttnn.float32: 4096}


class CachedProgram:
    """generic_op program rebuilt only when a tensor's address / shape / memory config changes (trace-safe)."""

    def __init__(self):
        self.program, self.signature = None, None

    def run(self, tensors, build):
        sig = tuple((int(t.buffer_address()), tuple(t.shape), tuple(t.padded_shape), str(t.memory_config()))
                    for t in tensors)
        if self.program is None or sig != self.signature:
            self.program, self.signature = build(), sig
        return ttnn.generic_op(tensors, self.program)


def row_ranges(cores) -> ttnn.CoreRangeSet:
    """Cores as row-contiguous CoreRanges (dispatch cost grows with the range count)."""
    ranges = []
    for c in sorted(cores, key=lambda c: (c.y, c.x)):
        if ranges and ranges[-1][1] == c.y and ranges[-1][2] + 1 == c.x:
            ranges[-1][2] = c.x
        else:
            ranges.append([c.x, c.y, c.x])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y), ttnn.CoreCoord(x1, y)) for x0, y, x1 in ranges])


def grid_cores(device, n: int, first: int = 0) -> list:
    """Logical cores first .. first + n - 1 of the worker grid, row-major."""
    g = device.compute_with_storage_grid_size()
    if first + n > g.x * g.y:
        raise ValueError(f"{first + n} cores > grid {g.x} x {g.y}")
    return [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(first, first + n)]


def cb_descriptors(indices, tiles, cores, dtype=ttnn.bfloat16, page=None):
    page = page or TILE_BYTES[dtype]
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    return [ttnn.CBDescriptor(total_size=tiles * page, core_ranges=cores,
                              format_descriptors=[ttnn.CBFormatDescriptor(i, dtype, page, tile)]) for i in indices]


def unified_kernels(source, cores, defines, reader_common, writer_common, per_core, compute):
    """Reader (NCRISC, NOC0) / writer (BRISC, NOC1) / compute descriptors of one single-source kernel."""
    def rt(args):
        r = ttnn.RuntimeArgs()
        for core, a in args:
            r[core.x][core.y] = list(a)
        return r

    dm = dict(kernel_source=str(source), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=cores,
              defines=defines)
    return [
        ttnn.KernelDescriptor(**dm, common_runtime_args=reader_common, runtime_args=rt(per_core),
                              config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                                       noc=ttnn.NOC.RISCV_0_default)),
        ttnn.KernelDescriptor(**dm, common_runtime_args=writer_common, runtime_args=rt(per_core),
                              config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                       noc=ttnn.NOC.RISCV_1_default)),
        ttnn.KernelDescriptor(**dm, runtime_args=rt(per_core), config=compute),
    ]


def f32_bits(v: float) -> int:
    return int(torch.tensor([v], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
