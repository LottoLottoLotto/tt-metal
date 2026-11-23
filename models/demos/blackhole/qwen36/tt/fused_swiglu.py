# SPDX-License-Identifier: Apache-2.0
"""P150 packed [up|gate] BF16 SwiGLU, including the SiLU BF16 rounding boundary."""

from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("fused_swiglu_kernel.cpp").resolve())
_GRID_X = 11
_TILE_BYTES = 2048


def _make_program(packed, output):
    half_tiles = int(packed.shape[-1]) // 64
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # Preserve standalone SiLU's BF16 pack before SFPU multiplication.
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    unpack[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    unpack[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode.extend(unpack)

    # 27B: 104*5 + 6*4 = 544 tiles on all 110 workers, one round.
    # 9B retains its original 96 four-tile workers.
    groups = ((0, 104, 5, 0), (104, 6, 4, 520)) if half_tiles == 544 else ((0, 96, 4, 0),)
    kernels = []
    cbs = []
    for first_worker, workers, tiles_per_core, first_tile in groups:
        core_list = [
            ttnn.CoreCoord(i % _GRID_X, i // _GRID_X)
            for i in range(first_worker, first_worker + workers)
        ]
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
        cbs.extend(
            ttnn.CBDescriptor(
                total_size=tiles_per_core * _TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, _TILE_BYTES, tile)],
            )
            for index in (0, 1, 2, 16)
        )
        offsets = [(core, [first_tile + i * tiles_per_core]) for i, core in enumerate(core_list)]
        descriptor = UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=cores,
            defines=[("QWEN_SWIGLU_HALF_TILES", str(half_tiles)), ("QWEN_SWIGLU_TILES", str(tiles_per_core))],
            ncrisc_common_runtime_args=[int(packed.buffer_address())],
            brisc_common_runtime_args=[int(output.buffer_address())],
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=offsets,
                brisc_args=offsets,
            ),
        )
        kernels.extend(descriptor.get_kernel_descriptors().kernels)
    return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])


def fused_swiglu(packed: ttnn.Tensor) -> ttnn.Tensor:
    """Return [1,1,M,MLP width] BF16 L1 for M=1/2, computing all padded rows.

    Build fresh runtime descriptors so no cached address can outlive its tensor.
    TTNN generic_op caches the compiled program by descriptor structure (runtime
    address values are excluded) and reapplies current runtime args on cache hits.
    Trace capture records the allocated output and these invocation's addresses.
    """
    shape = tuple(int(v) for v in packed.shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or not 1 <= shape[2] <= 32 or shape[3] not in (24576, 34816):
        raise ValueError(f"fused_swiglu requires [1,1,M,2*MLP], 1<=M<=32; got {shape}")
    if packed.dtype != ttnn.bfloat16 or packed.layout != ttnn.TILE_LAYOUT:
        raise ValueError("fused_swiglu requires tiled BF16 input")
    if packed.memory_config() != ttnn.L1_MEMORY_CONFIG:
        raise ValueError("fused_swiglu requires interleaved L1 input")
    if tuple(int(v) for v in packed.padded_shape) != (1, 1, 32, shape[3]):
        raise ValueError("fused_swiglu requires one standard 32-row tile row")
    grid = packed.device().compute_with_storage_grid_size()
    required_rows = 10 if shape[3] == 34816 else 9
    if grid.x < 11 or grid.y < required_rows:
        raise ValueError(f"fused_swiglu requires a P150 worker grid of at least 11x{required_rows}")
    output = ttnn.empty(
        [1, 1, shape[2], shape[3] // 2],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=packed.device(),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.generic_op([packed, output], _make_program(packed, output))
    return output
