# SPDX-License-Identifier: Apache-2.0
"""FP32 delta-rule preparation, retaining every original binary pack boundary."""
from functools import lru_cache
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("fused_normalization_kernel.cpp").resolve())
_WORKERS = 96


def _compatible(tensors, memory_config):
    try:
        if memory_config is not None and memory_config not in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
            return False
        device = tensors[0].device()
        if not ttnn.device.is_blackhole(device):
            return False
        if isinstance(device, ttnn.MeshDevice) and device.get_num_devices() != 1:
            return False
        grid = device.compute_with_storage_grid_size()
        if grid.x < 11 or grid.y < 9:
            return False
        return all(
            t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT
            and tuple(t.tile.tile_shape) == (32, 32)
            and not t.tile.transpose_within_face and not t.tile.transpose_of_faces
            and t.memory_config() in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
            and t.device() == device
            for t in tensors
        )
    except (RuntimeError, TypeError, AttributeError):
        return False


def _matrix(tensor):
    shape = tuple(tensor.shape)
    return len(shape) == 3 and 32 <= shape[0] <= 512 and shape[1:] == (128, 128) and tuple(tensor.padded_shape) == shape


def _eye(tensor):
    return tuple(tensor.shape) == (1, 128, 128) and tuple(tensor.padded_shape) == (1, 128, 128)


def supports(kk, L_mask, eye, alpha=0.25, legacy_doubling=False, memory_config=None):
    """Unsupported alpha/doubling/layout/device configurations require caller fallback."""
    return (
        not legacy_doubling and alpha in (0.0, 0.25, 1.0)
        and _matrix(kk) and tuple(L_mask.shape) == tuple(kk.shape)
        and tuple(L_mask.padded_shape) == tuple(kk.shape) and _eye(eye)
        and _compatible((kk, L_mask, eye), memory_config)
    )


def supports_normalize(L_mat, D_mat, D_inv_row, eye, memory_config=None):
    shape = tuple(L_mat.shape)
    return (
        _matrix(L_mat) and tuple(D_mat.shape) == shape and tuple(D_mat.padded_shape) == shape
        and tuple(D_inv_row.shape) == (shape[0], 128, 1)
        and tuple(D_inv_row.padded_shape) == (shape[0], 128, 32) and _eye(eye)
        and _compatible((L_mat, D_mat, D_inv_row, eye), memory_config)
    )


def regularize(kk, L_mask, eye, alpha=0.25, memory_config=None):
    """Return (L_mat, D_mat); caller retains original sum and reciprocal operations.

    Inputs remain allocated and unmodified. Legacy doubling must bypass this API.
    """
    if not supports(kk, L_mask, eye, alpha=alpha, memory_config=memory_config):
        raise ValueError("unsupported fused regularization configuration; use original path")
    mc = memory_config if memory_config is not None else kk.memory_config()
    outputs = tuple(ttnn.empty(tuple(kk.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                               device=kk.device(), memory_config=mc) for _ in range(2))
    _launch((kk, L_mask, eye, *outputs), 0, {0.0: 0, 0.25: 1, 1.0: 2}[alpha])
    return outputs


def normalize(L_mat, D_mat, D_inv_row, eye, memory_config=None):
    """Return L_unit using supplied original D_mat and reciprocal output, without shortcuts."""
    if not supports_normalize(L_mat, D_mat, D_inv_row, eye, memory_config=memory_config):
        raise ValueError("unsupported fused normalization configuration; use original path")
    mc = memory_config if memory_config is not None else L_mat.memory_config()
    output = ttnn.empty(tuple(L_mat.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=L_mat.device(), memory_config=mc)
    _launch((L_mat, D_mat, D_inv_row, eye, output), 1, 0)
    return output


def _launch(tensors, stage, alpha_mode):
    addresses = tuple(int(t.buffer_address()) for t in tensors)
    placements = tuple(int(t.memory_config().buffer_type == ttnn.BufferType.DRAM) for t in tensors)
    descriptor = _program(id(tensors[0].device()), int(tensors[0].shape[0]), stage,
                          alpha_mode, placements, addresses)
    ttnn.generic_op(list(tensors), descriptor)


@lru_cache(maxsize=128)
def _program(device_key, batch, stage, alpha_mode, placements, addresses):
    # Addresses participate in the bounded key; descriptors retain no tensors/devices.
    core_list = [ttnn.CoreCoord(i % 11, i // 11) for i in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8)),
    })
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17) if stage == 0 else (0, 1, 2, 3, 4, 5, 16)
    cbs = [ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.float32, 4096, tile)])
           for index in indices]
    # Match binary_ng's ComputeConfigDescriptor defaults, with FP32 accumulation/unpack.
    compute = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for index in indices:
        unpack[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode.extend(unpack)
    common = [("tiles", batch * 16), ("workers", _WORKERS), ("stage", stage), ("alpha_mode", alpha_mode)]
    input_count = 3 if stage == 0 else 4
    worker_args = [(core, [i]) for i, core in enumerate(core_list)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        ncrisc_named_compile_time_args=common + [(f"input{k}_dram", placements[k] if k < input_count else 0) for k in range(4)],
        brisc_named_compile_time_args=common + [(f"output{k}_dram", placements[input_count + k] if k < 5 - input_count else 0) for k in range(2)],
        trisc_named_compile_time_args=common,
        ncrisc_common_runtime_args=list(addresses[:input_count]),
        brisc_common_runtime_args=list(addresses[input_count:]),
        trisc_compute_config=compute,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=worker_args, brisc_args=worker_args, trisc_args=worker_args),
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])
