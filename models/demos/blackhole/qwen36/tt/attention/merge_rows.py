# SPDX-License-Identifier: Apache-2.0
"""Wide-verification SDPA group merge: one BF16 bitcopy launch replaces the
per-group reshape plus row concat (untilize/concat/tilize) of [1,B,24,256]
outputs into the [1,T,6144] attention row block."""

from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("merge_rows_kernel.cpp").resolve())
_TILE_BYTES = 2048
_GRID_X = 11
_WORKERS = 96  # 192 output tiles, two per worker.


def _program(groups, output, tokens):
    core_list = [ttnn.CoreCoord(index % _GRID_X, index // _GRID_X) for index in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    tasks = [(core, [index * 2, 2]) for index, core in enumerate(core_list)]
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    cbs = [
        ttnn.CBDescriptor(
            total_size=_TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, _TILE_BYTES, tile)],
        )
        for index in (0, 1)
    ]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        defines=[("MERGE_TOKENS", str(tokens)), ("MERGE_GROUP", str(int(groups[0].shape[1])))],
        brisc_common_runtime_args=[int(t.buffer_address()) for t in (output, *groups, *[groups[-1]] * (4 - len(groups)))],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=tasks, brisc_args=tasks, trisc_args=tasks,
        ),
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])


def merge_group_rows(groups):
    """groups: SDPA outputs [1,G,24,256]... and a last [1,1..G,24,256]; returns [1,T,6144]."""
    if not 1 <= len(groups) <= 4:
        raise ValueError("merge_group_rows expects the 1..4 SDPA row groups of T=3..8")
    sizes = [int(g.shape[1]) for g in groups]
    tokens = sum(sizes)
    if any(size != sizes[0] for size in sizes[:-1]) or not 1 <= sizes[-1] <= sizes[0]:
        raise ValueError("merge_group_rows expects equal consecutive row groups, the last one possibly shorter")
    for group, size in zip(groups, sizes):
        if (tuple(int(v) for v in group.shape) != (1, size, 24, 256)
                or tuple(int(v) for v in group.padded_shape) != (1, size, 32, 256)
                or group.dtype != ttnn.bfloat16 or group.layout != ttnn.TILE_LAYOUT
                or group.tile != ttnn.Tile((32, 32))
                or group.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError("merge_group_rows requires BF16 tiled interleaved DRAM [1,B,24,256] groups")
    grid = groups[0].device().compute_with_storage_grid_size()
    if grid.x < _GRID_X or grid.y * grid.x < _WORKERS:
        raise ValueError("merge_group_rows requires an 11-wide grid with 96 workers")
    output = ttnn.empty(
        (1, tokens, 6144), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=groups[0].device(), memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.generic_op([*groups, output], _program(groups, output, tokens))
    return output
