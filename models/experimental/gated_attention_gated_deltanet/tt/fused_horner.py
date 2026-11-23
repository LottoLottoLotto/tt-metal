# SPDX-License-Identifier: Apache-2.0
"""Native one-tile FP32 Horner recurrence with the TTNN operation boundaries."""
from functools import lru_cache
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("fused_horner_kernel.cpp").resolve())
_WORKERS = 96


def supports(neg_n, eye):
    """Metadata-only guard; no host reads or changes to either input."""
    shape = tuple(neg_n.shape)
    # Batch is a runtime argument strided over a fixed 96-core grid with constant per-core
    # CBs, so nothing scales with it. 27B at a 2048-token chunk needs 48 heads x 16 chunks x 4
    # diagonal blocks = 3072; the earlier 2048 cap forced the 30-iteration generic Python
    # Horner (34 batched matmuls) and tripled the layer time.
    if len(shape) != 3 or shape[1:] != (32, 32) or not 1 <= shape[0] <= 65536:
        return False
    if tuple(neg_n.padded_shape) != shape:
        return False
    if tuple(eye.shape) != (1, 32, 32) or tuple(eye.padded_shape) != (1, 32, 32):
        return False
    for tensor in (neg_n, eye):
        if tensor.dtype != ttnn.float32 or tensor.layout != ttnn.TILE_LAYOUT:
            return False
        if tensor.tile != ttnn.Tile((32, 32)):
            return False
        if tensor.memory_config() not in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
            return False
    device = neg_n.device()
    if eye.device() != device or not ttnn.device.is_blackhole(device):
        return False
    if isinstance(device, ttnn.MeshDevice) and device.get_num_devices() != 1:
        return False
    grid = device.compute_with_storage_grid_size()
    return grid.x >= 11 and grid.y >= 9


def horner(neg_n, eye, memory_config=None):
    """Return eye + neg_n @ R after R=eye+neg_n and exactly 30 iterations.

    Each matmul packs FP32 before the separate full-FP32 SFPU addition. The
    matmul inputs retain TTNN's default SrcA/SrcB (TF32) unpack semantics;
    FP32 destination accumulation does not mean full-FP32 multiplicands.
    Diagonal normalization, negation and final column scaling belong to caller.
    """
    if not supports(neg_n, eye):
        raise ValueError("unsupported fused Horner configuration; use original TTNN operations")
    mc = neg_n.memory_config() if memory_config is None else memory_config
    if mc not in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
        raise ValueError("fused Horner output must be interleaved L1 or DRAM")
    output = ttnn.empty(tuple(neg_n.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=neg_n.device(), memory_config=mc)
    tensors = [neg_n, eye, output]
    placements = tuple(int(t.memory_config().buffer_type == ttnn.BufferType.DRAM) for t in tensors)
    addresses = tuple(int(t.buffer_address()) for t in tensors)
    program = _make_program(id(neg_n.device()), int(neg_n.shape[0]), placements, addresses)
    ttnn.generic_op(tensors, program)
    return output


@lru_cache(maxsize=2048)  # every layer x bucket x masked/full stays warm after warmup
def _make_program(device_key, batch, placements, addresses):
    # Descriptors retain no tensors. Address keys prevent stale runtime pointers.
    # Batch and worker ID are runtime arguments, so all batch sizes use the same
    # binaries for a given input/output placement rather than per-core binaries.
    core_list = [ttnn.CoreCoord(i % 11, i // 11) for i in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8)),
    })
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # CB 0 and CB 3 alias the same cached neg_n SRAM, with independent queues
    # advanced together: default matmul unpack versus exact SFPU input unpack.
    cbs = [ttnn.CBDescriptor(total_size=2 * 4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.float32, 4096, tile)
                            for index in (0, 3)])]
    cbs.extend(ttnn.CBDescriptor(total_size=count * 4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.float32, 4096, tile)])
        for index, count in ((1, 1), (2, 1), (4, 1), (16, 2)))
    compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for index in (1, 3, 4):
        unpack[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode.extend(unpack)
    common = [("workers", _WORKERS)]
    worker_args = [(core, [i, batch]) for i, core in enumerate(core_list)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        ncrisc_named_compile_time_args=common + [("input_dram", placements[0]), ("eye_dram", placements[1])],
        brisc_named_compile_time_args=common + [("output_dram", placements[2])],
        trisc_named_compile_time_args=common,
        ncrisc_common_runtime_args=list(addresses[:2]),
        brisc_common_runtime_args=[addresses[2]],
        trisc_compute_config=compute,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=worker_args, brisc_args=worker_args, trisc_args=worker_args),
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])
