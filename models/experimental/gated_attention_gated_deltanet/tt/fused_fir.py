# SPDX-License-Identifier: Apache-2.0
"""Four-tap BF16 prefill FIR with the original operation rounding boundaries.

Blackhole, B=1, D=8192, BF16, 32..2048 tokens in multiples of32.
The first multiply uses UnpackToDestFp32; subsequent addcmul operations use
Default unpack and scalar1.0, with BF16 pack/reload after every operation.
Activation, history selection and persistent state updates remain in the caller.
"""
from functools import lru_cache
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("fused_fir_kernel.cpp").resolve())
_WORKERS = 96


def _interleaved(tensor):
    return tensor.memory_config() in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)


def supports(x_padded, tokens, taps, kernel_size=4):
    if kernel_size != 4 or tokens < 32 or tokens > 2048 or tokens % 32 or len(taps) != 4:
        return False
    if tuple(x_padded.shape) != (1, tokens + 3, 8192):
        return False
    if x_padded.layout == ttnn.TILE_LAYOUT and tuple(x_padded.padded_shape) != (1, ((tokens + 34) // 32) * 32, 8192):
        return False
    if any(tuple(t.shape) != (1, 1, 8192) or tuple(t.padded_shape) != (1, 32, 8192) for t in taps):
        return False
    if x_padded.dtype != ttnn.bfloat16 or x_padded.layout not in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT) or not _interleaved(x_padded):
        return False
    if any(t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or not _interleaved(t) for t in taps):
        return False
    device = x_padded.device()
    if not ttnn.device.is_blackhole(device):
        return False
    if isinstance(device, ttnn.MeshDevice) and device.get_num_devices() != 1:
        return False
    if any(t.device() != device for t in taps):
        return False
    grid = device.compute_with_storage_grid_size()
    return grid.x >= 11 and grid.y >= 9


def fir_mac(x_padded, tokens, taps, memory_config=None):
    """Compute MAC only; callers retain existing activation and state code."""
    if not supports(x_padded, tokens, taps):
        raise ValueError("unsupported fused FIR shape/layout/dtype; call original implementation")
    mc = memory_config if memory_config is not None else x_padded.memory_config()
    if mc not in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
        raise ValueError("fused FIR output must be interleaved L1 or DRAM")
    if x_padded.layout != ttnn.TILE_LAYOUT:
        x_padded = ttnn.to_layout(x_padded, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.empty((1, tokens, 8192), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=x_padded.device(), memory_config=mc)
    tensors = [x_padded, *taps, output]
    placements = tuple(int(t.memory_config().buffer_type == ttnn.BufferType.DRAM) for t in tensors)
    addresses = tuple(int(t.buffer_address()) for t in tensors)
    program = _make_program(id(x_padded.device()), tokens, placements, addresses)
    ttnn.generic_op(tensors, program)
    return output


@lru_cache(maxsize=128)
def _make_program(device_key, tokens, placements, addresses):
    """Bounded address/shape/placement cache; retains no tensors or devices.

    Every buffer address, including fresh output, participates in the key.
    Reusing an allocator address is safe because the current tensor is passed
    to generic_op. Eviction affects host construction only, not TTNN compilation.
    """
    core_list = [ttnn.CoreCoord(i % 11, i // 11) for i in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8)),
    })
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # k0 input/weight, k1..3 input/weight, BF16 accumulator, source scratch, output.
    cbs = [ttnn.CBDescriptor(total_size=count * 2048, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, 2048, tile)])
           for index, count in ((0, 1), (1, 1), (2, 3), (3, 3), (4, 1), (6, 1), (16, 2))]
    compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=False, dst_full_sync_en=False)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    unpack[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    unpack[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode.extend(unpack)
    common = [("rows", tokens // 32), ("workers", _WORKERS)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        ncrisc_named_compile_time_args=common + [("input_dram", placements[0])]
            + [(f"weight{k}_dram", placements[k + 1]) for k in range(4)],
        brisc_named_compile_time_args=common + [("output_dram", placements[5])],
        trisc_named_compile_time_args=common,
        ncrisc_common_runtime_args=list(addresses[:5]),
        brisc_common_runtime_args=[addresses[5]],
        trisc_compute_config=compute,
        per_core_compile_time_descriptors=[PerCoreCompileTimeDescriptor(
            "worker", [(core, i) for i, core in enumerate(core_list)], 1)],
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])
