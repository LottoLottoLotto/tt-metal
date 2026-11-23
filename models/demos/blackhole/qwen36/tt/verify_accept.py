# SPDX-License-Identifier: Apache-2.0
"""Native exact greedy verification for the linear DFlash cycle.

Replaces untilize + ttnn.argmax + ~40 scalar TTNN ops with two launches:
  1. 110 cores x (BRISC, NCRISC) scan the live rows of the tiled BF16 logits
     (only faces 0/1 of each tile are read) with ttnn.argmax's order;
  2. one core merges, forms the accepted prefix and writes all cycle scalars.
Outputs have fixed addresses (trace-safe) and are borrowed until the next call.
"""
from pathlib import Path

import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("verify_accept_kernel.cpp").resolve())
_CORES = 110


def _i32(device, shape, dtype=ttnn.int32):
    return ttnn.from_torch(torch.zeros(shape, dtype=torch.int32), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class NativeVerifyAccept:
    def __init__(self, device, rows, vocab):
        if not 2 <= rows <= 8 or vocab % 32:
            raise ValueError("Native verify accept supports 2..8 rows and a tile-multiple vocabulary")
        grid = device.compute_with_storage_grid_size()
        if grid.x * grid.y < _CORES:
            raise ValueError("Native verify accept requires 110 worker cores")
        self.device, self.rows, self.vocab = device, rows, vocab
        self.coordinates = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(_CORES)]
        self.workers = 2 * _CORES
        self.partial = _i32(device, (self.workers, 16), ttnn.uint32)
        self.head = _i32(device, (1, 2 * rows - 1))
        # accepted, next_token, advance, next_position, feature start (old position 0)
        self.scalars = [_i32(device, (1, 1)) for _ in range(5)]

    def _launch(self, tensors, mode, coordinates, ncrisc):
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in coordinates})
        tasks = [(core, [i]) for i, core in enumerate(coordinates)]
        addresses = [int(t.buffer_address()) for t in tensors]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL, core_ranges=cores,
            defines=[("VA_ROWS", str(self.rows)), ("VA_MODE", str(mode)), ("VA_WORKERS", str(self.workers)),
                     ("VA_VOCAB_TILES", str(self.vocab // 32))],
            brisc_common_runtime_args=addresses,
            ncrisc_common_runtime_args=addresses if ncrisc else [],
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=tasks, **({"ncrisc_args": tasks} if ncrisc else {})),
        )
        size = 16384 if mode else 2048
        cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(
            index, ttnn.bfloat16, 2048, ttnn.TileDescriptor(ttnn.Tile((32, 32))))]) for index in ((0,) if mode else (0, 1))]
        program = ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])
        ttnn.generic_op(tensors, program)

    def __call__(self, logits, tokens, positions):
        """Returns (targets+flags head [1,2R-1], accepted, next_token, advance, next_position, start);
        positions is advanced in place to next_position + [0, R)."""
        shape = tuple(int(x) for x in logits.padded_shape)
        if (shape[-1] != self.vocab or shape[-2] != 32 or int(logits.shape[-2]) != self.rows
                or logits.layout != ttnn.TILE_LAYOUT or logits.dtype != ttnn.bfloat16
                or logits.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError(f"Native verify accept requires DRAM tiled BF16 [.., {self.rows}, {self.vocab}] logits")
        self._launch([logits, self.partial], 0, self.coordinates, True)
        self._launch([self.partial, tokens, positions, self.head, *self.scalars], 1, self.coordinates[:1], False)
        return (self.head, *self.scalars)

    def release(self):
        for tensor in (self.partial, self.head, *self.scalars):
            ttnn.deallocate(tensor)
