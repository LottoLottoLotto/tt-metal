# SPDX-License-Identifier: Apache-2.0
"""Exact BF16 partial RoPE for single-batch P150 decode within one row tile."""

from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("fused_rope_kernel.cpp").resolve())
_TILE_BYTES = 2048
_WORKERS = 56


def _shape(tensor, padded=False):
    return tuple(int(value) for value in (tensor.padded_shape if padded else tensor.shape))


def _validate(q, k, cos, sin):
    tokens = _shape(q)[-2]
    expected = ((1, 24, tokens, 256), (1, 4, tokens, 256))
    for name, tensor, shape in zip(("q", "k"), (q, k), expected):
        if _shape(tensor) != shape or _shape(tensor, True) != (*shape[:2], 32, 256):
            raise ValueError(f"RoPE fusion {name}: unsupported logical/padded shape {_shape(tensor)}/{_shape(tensor, True)}")
    for name, tensor in (("cos", cos), ("sin", sin)):
        shape = _shape(tensor)
        # M1 metadata reshape and M2's original unsqueeze both describe the
        # same two physical tiles. Do not unsqueeze/copy or broadcast pad rows.
        if shape not in ((1, tokens, 64), (1, 1, tokens, 64)):
            raise ValueError(f"RoPE fusion {name}: expected [1,T,64] or [1,1,T,64], got {shape}")
        if _shape(tensor, True) != (*shape[:-2], 32, 64):
            raise ValueError(f"RoPE fusion {name}: unsupported padded shape {_shape(tensor, True)}")
    for name, tensor in (("q", q), ("k", k), ("cos", cos), ("sin", sin)):
        if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"RoPE fusion {name}: requires tiled BF16")
        if tensor.tile != ttnn.Tile((32, 32)):
            raise ValueError(f"RoPE fusion {name}: requires standard non-transposed 32x32 tiles")
        if tensor.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
            raise ValueError(f"RoPE fusion {name}: requires interleaved DRAM or L1")
        if tensor.device() != q.device():
            raise ValueError(f"RoPE fusion {name}: device differs from q")
    grid = q.device().compute_with_storage_grid_size()
    if grid.x < 11 or grid.y < 6:
        raise ValueError("RoPE fusion requires at least an 11x6 logical worker grid")


def _program(q, k, cos, sin, q_out, k_out):
    core_list = [ttnn.CoreCoord(index % 11, index // 11) for index in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # 8 single-tile CBs + one three-tile DMA-only CB = 22528 B/core.
    # 749568 persistent + 111488 CB origin + 22528 = 883584 B/core,
    # leaving 689280 B before the existing 61440-byte safety margin.
    cbs = [
        ttnn.CBDescriptor(
            total_size=(3 if index == 17 else 1) * _TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, _TILE_BYTES, tile)],
        )
        for index in (0, 1, 2, 3, 4, 5, 6, 16, 17)
    ]
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for index in (0, 2, 3, 4):
        unpack[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode.extend(unpack)
    tasks = [(core, [index]) for index, core in enumerate(core_list)]
    # Memory placement is compile-time accessor structure and therefore hashed;
    # all addresses and all three RISCs' task arguments are fresh on every call.
    defines = [
        (f"ROPE_{name}_DRAM", "1" if tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG else "0")
        for name, tensor in (("Q", q), ("K", k), ("COS", cos), ("SIN", sin))
    ]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        defines=defines,
        ncrisc_common_runtime_args=[int(t.buffer_address()) for t in (q, k, cos, sin)],
        brisc_common_runtime_args=[int(t.buffer_address()) for t in (q_out, k_out)],
        trisc_compute_config=compute,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=tasks,
            brisc_args=tasks,
            trisc_args=tasks,
        ),
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])


def fused_rope(q, k, cos, sin):
    """Compute all 32 physical rows, including nonlogical padding, without FMA."""
    if len(_shape(q)) != 4 or not 1 <= _shape(q)[-2] <= 32:
        raise ValueError("fused_rope requires 1..32 single-sequence decode rows")
    _validate(q, k, cos, sin)
    # Pinned concat.cpp defaults to DRAM, not the first input's placement.
    q_out = ttnn.empty(_shape(q), dtype=q.dtype, layout=q.layout, device=q.device(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_out = ttnn.empty(_shape(k), dtype=k.dtype, layout=k.layout, device=k.device(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.generic_op([q, k, cos, sin, q_out, k_out], _program(q, k, cos, sin, q_out, k_out))
    return q_out, k_out

