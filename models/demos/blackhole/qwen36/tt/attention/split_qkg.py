# SPDX-License-Identifier: Apache-2.0
"""M1/M2 BF16 Q/gate/K layout fusion; raw QKV remains owned by the caller.

The fused projection row is ``[Q+gate per head | K | V]``: the SDK reshapes Q+gate to
[B,T,H,512] before chunking the last axis, so each head's Q256 precedes its gate256.
Only logical token rows 0..T-1 (T = 1 or 2) are copied; every output padding bit is zero.
"""

from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("split_qkg_kernel.cpp").resolve())
_TILE_BYTES = 2048
_GRID_X = 11
_WORKERS = 110
_HEAD_DIM = 256
_HEAD_TILES = _HEAD_DIM // 32


def _validate(qkv, num_heads, num_kv_heads):
    if not (1 <= num_kv_heads <= num_heads <= 32):
        raise ValueError(f"split_qkg supports 1 <= kv heads <= heads <= 32, got {num_heads}/{num_kv_heads}")
    width = (2 * num_heads + 2 * num_kv_heads) * _HEAD_DIM
    shape = tuple(int(value) for value in qkv.shape)
    padded = tuple(int(value) for value in qkv.padded_shape)
    tokens = shape[1] if len(shape) == 3 else 0
    if tokens not in (1, 2) or shape != (1, tokens, width) or padded != (1, 32, width):
        raise ValueError(f"split_qkg requires logical [1,1|2,{width}], padded [1,32,{width}]; got {shape}/{padded}")
    if qkv.tile != ttnn.Tile((32, 32)):
        raise ValueError("split_qkg requires standard non-transposed 32x32 tiles")
    if qkv.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("split_qkg requires interleaved DRAM input")
    grid = qkv.device().compute_with_storage_grid_size()
    if grid.x < _GRID_X or grid.y < 10:
        raise ValueError("split_qkg requires at least an 11x10 logical worker grid")
    return tokens


def _program(qkv, q, k, gate, num_heads, num_kv_heads, tokens):
    core_list = [ttnn.CoreCoord(index % _GRID_X, index // _GRID_X) for index in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    gate_tiles = num_heads * _HEAD_TILES
    gate_workers = _WORKERS - 2 * _HEAD_TILES
    base, extra = divmod(gate_tiles, gate_workers)
    tasks = []
    for index, core in enumerate(core_list):
        if index < _HEAD_TILES:
            task = [0, index, 1]  # Q: one complete head-row tile per worker.
        elif index < 2 * _HEAD_TILES:
            task = [1, index - _HEAD_TILES, 1]  # K: one complete KV-head-row tile.
        else:
            worker = index - 2 * _HEAD_TILES
            # Gate tiles spread evenly; the first `extra` workers own one more.
            task = [2, worker * base + min(worker, extra), base + (1 if worker < extra else 0)]
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
        brisc_common_runtime_args=[int(tensor.buffer_address()) for tensor in (qkv, q, k, gate)]
        + [num_heads, num_kv_heads, 2 * num_heads * _HEAD_TILES, tokens],
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


def split_qkg(qkv, num_heads, num_kv_heads):
    """Return DRAM TILE BF16 Q[T,1,H,256], K[T,1,Hkv,256], gate[1,T,H*256] for T = 1 or 2.

    One generic_op invocation, no tensor math or persistent host/device state.
    Fresh output allocations and runtime addresses are supplied on every call.
    The caller must retain qkv for the later V gather; this helper never frees it.
    """
    tokens = _validate(qkv, num_heads, num_kv_heads)
    outputs = [
        ttnn.empty(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=qkv.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for shape in (
            (tokens, 1, num_heads, _HEAD_DIM),
            (tokens, 1, num_kv_heads, _HEAD_DIM),
            (1, tokens, num_heads * _HEAD_DIM),
        )
    ]
    q, k, gate = outputs
    ttnn.generic_op([qkv, q, k, gate], _program(qkv, q, k, gate, num_heads, num_kv_heads, tokens))
    return q, k, gate
