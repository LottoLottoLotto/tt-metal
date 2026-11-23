# SPDX-License-Identifier: Apache-2.0
"""Exact FP32 decay differences and mask/clip/exp preparation.

Read compact [batch,128] decay; expand row/column operands locally rather than
materializing their reshapes and [batch,128,128] broadcast subtraction.
"""
from functools import lru_cache
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("fused_decay_kernel.cpp").resolve())
_WORKERS = 96


def supports(decay, tril_mask):
    """Metadata-only guard; unsupported configurations retain the TTNN path."""
    shape = tuple(decay.shape)
    if len(shape) != 2 or shape[1] != 128:
        return False
    if not 32 <= shape[0] <= 512 or shape[0] % 32:
        return False
    if tuple(decay.padded_shape) != shape:
        return False
    if tuple(tril_mask.shape) != (1, 128, 128) or tuple(tril_mask.padded_shape) != (1, 128, 128):
        return False
    for tensor in (decay, tril_mask):
        if tensor.dtype != ttnn.float32 or tensor.layout != ttnn.TILE_LAYOUT:
            return False
        if tensor.tile != ttnn.Tile((32, 32)):
            return False
        if tensor.memory_config() not in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
            return False
    device = decay.device()
    if tril_mask.device() != device or not ttnn.device.is_blackhole(device):
        return False
    if isinstance(device, ttnn.MeshDevice) and device.get_num_devices() != 1:
        return False
    grid = device.compute_with_storage_grid_size()
    return grid.x >= 11 and grid.y >= 9


def decay_mask(decay, tril_mask, memory_config=None):
    """Return L_mask, preserving inputs and each original FP32 pack boundary.

    Local integer copies reproduce the original tiled reshapes and broadcasts.
    Subtract -> multiply(mask) -> clip[-20,+0] -> accurate exp -> multiply(mask).
    """
    if not supports(decay, tril_mask):
        raise ValueError("unsupported fused decay configuration; use original TTNN operations")
    mc = decay.memory_config() if memory_config is None else memory_config
    if mc not in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
        raise ValueError("fused decay output must be interleaved L1 or DRAM")
    output = ttnn.empty((int(decay.shape[0]), 128, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=decay.device(), memory_config=mc)
    tensors = [decay, tril_mask, output]
    placements = tuple(int(t.memory_config().buffer_type == ttnn.BufferType.DRAM) for t in tensors)
    addresses = tuple(int(t.buffer_address()) for t in tensors)
    program = _make_program(id(decay.device()), int(decay.shape[0]), placements, addresses)
    ttnn.generic_op(tensors, program)
    return output


@lru_cache(maxsize=128)
def _make_program(device_key, batch, placements, addresses):
    # Cache contains descriptors only; every current tensor address is keyed.
    core_list = [ttnn.CoreCoord(i % 11, i // 11) for i in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8)),
    })
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # Expanded operands, mask, FP32 staging, private reader scratch, and output.
    cbs = [ttnn.CBDescriptor(total_size=count * 4096, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.float32, 4096, tile)])
           for index, count in ((0, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (7, 1), (16, 2))]
    compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for index in (0, 1, 2, 3, 4, 5, 6):
        unpack[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode.extend(unpack)
    common = [("tiles", batch * 16), ("workers", _WORKERS)]
    worker_args = [(core, [i]) for i, core in enumerate(core_list)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        ncrisc_named_compile_time_args=common + [("input_dram", placements[0]), ("mask_dram", placements[1])],
        brisc_named_compile_time_args=common + [("output_dram", placements[2])],
        trisc_named_compile_time_args=common,
        ncrisc_common_runtime_args=list(addresses[:2]),
        brisc_common_runtime_args=[addresses[2]],
        trisc_compute_config=compute,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=worker_args, brisc_args=worker_args, trisc_args=worker_args),
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])
