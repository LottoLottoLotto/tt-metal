# SPDX-License-Identifier: Apache-2.0
"""M1-only BF16 Q/gate/K layout fusion; raw QKV remains owned by the caller.

The pinned SDK reshapes Q+gate to [B,T,24,512] before chunking the last
axis: each head's Q256 precedes its gate256, not all queries then all gates.
Only logical token row zero is copied; every output padding bit is zero.
"""

from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("split_qkg_kernel.cpp").resolve())
_KERNEL_TWO = str(Path(__file__).with_name("split_qkg2_kernel.cpp").resolve())
_TILE_BYTES = 2048
_GRID_X = 11
_WORKERS = 110


def _validate(qkv, tokens=1):
    shape = tuple(int(value) for value in qkv.shape)
    padded = tuple(int(value) for value in qkv.padded_shape)
    if shape != (1, tokens, 14336) or padded != (1, 32, 14336):
        raise ValueError(
            f"split_qkg requires logical [1,{tokens},14336], padded [1,32,14336]; got {shape}/{padded}"
        )
    if qkv.dtype != ttnn.bfloat16 or qkv.layout != ttnn.TILE_LAYOUT:
        raise ValueError("split_qkg requires tiled BF16 input")
    if qkv.tile != ttnn.Tile((32, 32)):
        raise ValueError("split_qkg requires standard non-transposed 32x32 tiles")
    if qkv.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("split_qkg requires interleaved DRAM input")
    grid = qkv.device().compute_with_storage_grid_size()
    if grid.x < _GRID_X or grid.y < 10:
        raise ValueError("split_qkg requires at least an 11x10 logical worker grid")


def _program(qkv, q, k, gate):
    core_list = [ttnn.CoreCoord(index % _GRID_X, index // _GRID_X) for index in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    tasks = []
    for index, core in enumerate(core_list):
        if index < 8:
            task = [0, index, 1]  # Q: one complete 24-head-row tile per worker.
        elif index < 16:
            task = [1, index - 8, 1]  # K: one complete four-head-row tile.
        else:
            worker = index - 16
            # 192 gate tiles across 94 workers: four own three, 90 own two.
            task = [2, worker * 2 + min(worker, 4), 3 if worker < 4 else 2]
        tasks.append((core, task))
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # Output tile plus 24 pairs of aligned 64-byte DRAM read staging slots.
    # NCRISC/TRISC are empty; BRISC alone owns both scratch and complete writes.
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * _TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, _TILE_BYTES, tile)],
        )
        for index, pages in ((0, 1), (1, 2))
    ]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        brisc_common_runtime_args=[int(tensor.buffer_address()) for tensor in (qkv, q, k, gate)],
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


def split_qkg(qkv):
    """Return DRAM TILE BF16 Q[1,1,24,256], K[1,1,4,256], gate[1,1,6144].

    One generic_op invocation, no tensor math or persistent host/device state.
    Fresh output allocations and runtime addresses are supplied on every call.
    The caller must retain qkv for the later V gather; this helper never frees it.
    """
    _validate(qkv)
    outputs = [
        ttnn.empty(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=qkv.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for shape in ((1, 1, 24, 256), (1, 1, 4, 256), (1, 1, 6144))
    ]
    q, k, gate = outputs
    ttnn.generic_op([qkv, q, k, gate], _program(qkv, q, k, gate))
    return q, k, gate


def _program_two(qkv, q, k, gate):
    core_list = [ttnn.CoreCoord(index % _GRID_X, index // _GRID_X) for index in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    tasks = []
    for index, core in enumerate(core_list):
        if index < 16:
            task = [0, index % 8, 1, index // 8]  # Q: token index // 8, one tile each.
        elif index < 32:
            task = [1, (index - 16) % 8, 1, (index - 16) // 8]  # K: four head rows per tile.
        else:
            worker = index - 32
            # 192 two-row gate tiles across 78 workers: 36 own three, 42 own two.
            task = [2, worker * 2 + min(worker, 36), 3 if worker < 36 else 2, 0]
        tasks.append((core, task))
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * _TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, _TILE_BYTES, tile)],
        )
        for index, pages in ((0, 1), (1, 2))
    ]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL_TWO,
        core_ranges=cores,
        brisc_common_runtime_args=[int(tensor.buffer_address()) for tensor in (qkv, q, k, gate)],
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


def split_qkg_two(qkv):
    """M=2 verification split: Q[1,2,24,256], K[1,2,4,256], gate[1,2,6144] (DRAM TILE BF16).

    Bit copies of the same values the generic slice/reshape/chunk path produces;
    one launch replaces three slices, two reshapes and two chunk slices.
    """
    _validate(qkv, tokens=2)
    outputs = [
        ttnn.empty(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=qkv.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for shape in ((1, 2, 24, 256), (1, 2, 4, 256), (1, 2, 6144))
    ]
    q, k, gate = outputs
    ttnn.generic_op([qkv, q, k, gate], _program_two(qkv, q, k, gate))
    return q, k, gate


_KERNEL_MANY = str(Path(__file__).with_name("split_qkgn_kernel.cpp").resolve())


def _program_many(qkv, q, k, gate, tokens):
    core_list = [ttnn.CoreCoord(index % _GRID_X, index // _GRID_X) for index in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    # Q (24 head rows per tile) dominates the row copies: split each Q column tile's token
    # rows over four workers and each K tile's over two; the rest share the 192 gate tiles.
    token_splits = []
    for parts in (4, 2):
        bounds = [tokens * p // parts for p in range(parts + 1)]
        token_splits.append([(bounds[p], bounds[p + 1]) for p in range(parts)])
    plan = [[0, tile, 1, *span] for tile in range(8) for span in token_splits[0]]
    plan += [[1, tile, 1, *span] for tile in range(8) for span in token_splits[1]]
    gate_workers = _WORKERS - len(plan)
    base, extra = divmod(192, gate_workers)
    first = 0
    for worker in range(gate_workers):
        count = base + (1 if worker < extra else 0)
        plan.append([2, first, count, 0, 0])
        first += count
    tasks = list(zip(core_list, plan))
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * _TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, _TILE_BYTES, tile)],
        )
        for index, pages in ((0, tokens), (1, 3 if tokens <= 4 else 6))
    ]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL_MANY,
        core_ranges=cores,
        defines=[("SPLIT_TOKENS", str(tokens))],
        brisc_common_runtime_args=[int(tensor.buffer_address()) for tensor in (qkv, q, k, gate)],
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


def split_qkg_many(qkv):
    """Wide verification split for T=3..8 rows: Q[1,T,24,256], K[1,T,4,256], gate[1,T,6144].

    Bit copies of the values the generic slice/reshape/chunk path produces.
    """
    tokens = int(qkv.shape[-2])
    if not 3 <= tokens <= 8:
        raise ValueError("split_qkg_many supports three to eight token rows")
    _validate(qkv, tokens=tokens)
    q, k, gate = (
        ttnn.empty(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=qkv.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for shape in ((1, tokens, 24, 256), (1, tokens, 4, 256), (1, tokens, 6144))
    )
    ttnn.generic_op([qkv, q, k, gate], _program_many(qkv, q, k, gate, tokens))
    return q, k, gate
