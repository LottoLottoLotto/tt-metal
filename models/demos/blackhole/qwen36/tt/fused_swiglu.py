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
_WORKERS = 96
_TILES_PER_CORE = 4
_HALF_TILES = 384
_TILE_BYTES = 2048


def _make_program(packed, output):
    core_list = [ttnn.CoreCoord(i % _GRID_X, i // _GRID_X) for i in range(_WORKERS)]
    cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8)),
        }
    )
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    cbs = [
        ttnn.CBDescriptor(
            total_size=_TILES_PER_CORE * _TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, _TILE_BYTES, tile)],
        )
        for index in (0, 1, 2, 16)
    ]
    # unary_program_factory: BF16 SiLU uses Default unpack, HiFi4, APPROX=false,
    # fp32_dest_acc_en=false. binary_ng's default MUL is SFPU (not FPU), with
    # the SAME compute settings but UnpackToDestFp32 for both BF16 operands.
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    unpack[1] = ttnn.UnpackToDestMode.UnpackToDestFp32  # up
    unpack[2] = ttnn.UnpackToDestMode.UnpackToDestFp32  # BF16 SiLU intermediate
    compute.unpack_to_dest_mode.extend(unpack)
    offsets = [(core, [i * _TILES_PER_CORE]) for i, core in enumerate(core_list)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        ncrisc_common_runtime_args=[int(packed.buffer_address())],
        brisc_common_runtime_args=[int(output.buffer_address())],
        trisc_compute_config=compute,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=offsets,
            brisc_args=offsets,
        ),
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])


def fused_swiglu(packed: ttnn.Tensor) -> ttnn.Tensor:
    """Return a fresh [1,1,M,12288] L1 tensor for M=1/2; all padded rows are computed.

    Build fresh runtime descriptors so no cached address can outlive its tensor.
    TTNN generic_op caches the compiled program by descriptor structure (runtime
    address values are excluded) and reapplies current runtime args on cache hits.
    Trace capture records the allocated output and these invocation's addresses.
    """
    shape = tuple(int(v) for v in packed.shape)
    if shape not in ((1, 1, 1, 24576), (1, 1, 2, 24576)):
        raise ValueError(f"fused_swiglu requires [1,1,M,24576], M=1 or 2; got {shape}")
    if packed.dtype != ttnn.bfloat16 or packed.layout != ttnn.TILE_LAYOUT:
        raise ValueError("fused_swiglu requires tiled BF16 input")
    if packed.memory_config() != ttnn.L1_MEMORY_CONFIG:
        raise ValueError("fused_swiglu requires interleaved L1 input")
    if tuple(int(v) for v in packed.padded_shape) != (1, 1, 32, 24576):
        raise ValueError("fused_swiglu requires one standard 32-row tile row")
    grid = packed.device().compute_with_storage_grid_size()
    if grid.x < 11 or grid.y < 9:
        raise ValueError("fused_swiglu requires a P150 worker grid of at least 11x9")
    output = ttnn.empty(
        [1, 1, shape[2], _HALF_TILES * 32],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=packed.device(),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.generic_op([packed, output], _make_program(packed, output))
    return output
