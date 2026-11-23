# SPDX-License-Identifier: Apache-2.0
"""Four-tap BF16 prefill FIR with the original operation rounding boundaries.

Blackhole, B=1, D a multiple of 32, BF16, 32..2048 tokens in multiples of 32.
fir_mac reads the concatenated [history; x]; fir_mac_split reads x and the 3-row history
separately (no concatenation) with identical arithmetic.
The first multiply uses UnpackToDestFp32; subsequent addcmul operations use
Default unpack and scalar1.0, with BF16 pack/reload after every operation.
Activation, history selection and persistent state updates remain in the caller.
"""
from functools import lru_cache
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("fused_fir_kernel.cpp").resolve())
_WORKERS = 96


def _interleaved(tensor):
    return tensor.memory_config() in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)


def _operands_ok(inputs, taps, width):
    if any(tuple(t.shape) != (1, 1, width) or tuple(t.padded_shape) != (1, 32, width) for t in taps):
        return False
    if any(t.dtype != ttnn.bfloat16 or not _interleaved(t) for t in (*inputs, *taps)):
        return False
    if any(t.layout != ttnn.TILE_LAYOUT for t in taps):
        return False
    device = inputs[0].device()
    if not ttnn.device.is_blackhole(device):
        return False
    if isinstance(device, ttnn.MeshDevice) and device.get_num_devices() != 1:
        return False
    if any(t.device() != device for t in (*inputs, *taps)):
        return False
    grid = device.compute_with_storage_grid_size()
    return grid.x >= 11 and grid.y >= 9


def _width_ok(tokens, taps, kernel_size, width):
    return (kernel_size == 4 and 32 <= tokens <= 2048 and tokens % 32 == 0 and len(taps) == 4
            and width % 32 == 0 and 32 <= width <= 65536)


def supports(x_padded, tokens, taps, kernel_size=4):
    shape = tuple(x_padded.shape)
    if len(shape) != 3 or shape[:2] != (1, tokens + 3) or not _width_ok(tokens, taps, kernel_size, shape[2]):
        return False
    width = shape[2]
    if x_padded.layout == ttnn.TILE_LAYOUT and tuple(x_padded.padded_shape) != (1, ((tokens + 34) // 32) * 32, width):
        return False
    if x_padded.layout not in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
        return False
    return _operands_ok((x_padded,), taps, width)


def supports_split(x, history, tokens, taps, kernel_size=4):
    shape = tuple(x.shape)
    if len(shape) != 3 or shape[:2] != (1, tokens) or not _width_ok(tokens, taps, kernel_size, shape[2]):
        return False
    width = shape[2]
    if x.layout != ttnn.TILE_LAYOUT or tuple(x.padded_shape) != shape:
        return False
    if history.layout != ttnn.TILE_LAYOUT or tuple(history.shape) != (1, 3, width) \
            or tuple(history.padded_shape) != (1, 32, width):
        return False
    return _operands_ok((x, history), taps, width)


def _output_config(memory_config, source):
    mc = memory_config if memory_config is not None else source.memory_config()
    if mc not in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
        raise ValueError("fused FIR output must be interleaved L1 or DRAM")
    return mc


def _run(source, history, tokens, taps, mc):
    width = int(source.shape[2])
    output = ttnn.empty((1, tokens, width), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=source.device(), memory_config=mc)
    tensors = [source, *taps] + ([history] if history is not None else []) + [output]
    placement = lambda t: int(t.memory_config().buffer_type == ttnn.BufferType.DRAM)
    placements = (placement(source), *(placement(t) for t in taps), placement(output),
                  placement(history) if history is not None else 0)
    addresses = (int(source.buffer_address()), *(int(t.buffer_address()) for t in taps),
                 int(output.buffer_address()), int(history.buffer_address()) if history is not None else 0)
    program = _make_program(id(source.device()), tokens, width, int(history is not None), placements, addresses)
    ttnn.generic_op(tensors, program)
    return output


def fir_mac(x_padded, tokens, taps, memory_config=None):
    """Compute MAC only; callers retain existing activation and state code."""
    if not supports(x_padded, tokens, taps):
        raise ValueError("unsupported fused FIR shape/layout/dtype; call original implementation")
    mc = _output_config(memory_config, x_padded)
    if x_padded.layout != ttnn.TILE_LAYOUT:
        x_padded = ttnn.to_layout(x_padded, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return _run(x_padded, None, tokens, taps, mc)


def fir_mac_split(x, history, tokens, taps, memory_config=None):
    """fir_mac over concat([history, x], dim=1) without materializing the concatenation."""
    if not supports_split(x, history, tokens, taps):
        raise ValueError("unsupported fused FIR shape/layout/dtype; call original implementation")
    return _run(x, history, tokens, taps, _output_config(memory_config, x))


@lru_cache(maxsize=2048)  # every layer x bucket x masked/full stays warm after warmup
def _make_program(device_key, tokens, width, split, placements, addresses):
    """Bounded address/shape/placement cache; retains no tensors or devices.

    Every buffer address, including fresh output, participates in the key.
    Reusing an allocator address is safe because the current tensor is passed
    to generic_op. Eviction affects host construction only, not TTNN compilation.
    placements/addresses: (input, weight0..3, output, history-or-0).
    """
    core_list = [ttnn.CoreCoord(i % 11, i // 11) for i in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8)),
    })
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # k0 input/weight, k1..3 input/weight, BF16 accumulator, source scratch (next/current and
    # previous row tiles), output.
    cbs = [ttnn.CBDescriptor(total_size=count * 2048, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, 2048, tile)])
           for index, count in ((0, 1), (1, 1), (2, 3), (3, 3), (4, 1), (6, 2), (16, 2))]
    compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=False, dst_full_sync_en=False)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    unpack[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    unpack[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode.extend(unpack)
    common = [("rows", tokens // 32), ("cols", width // 32), ("workers", _WORKERS), ("split", split)]
    worker_args = [(core, [i]) for i, core in enumerate(core_list)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        ncrisc_named_compile_time_args=common + [("input_dram", placements[0]), ("history_dram", placements[6])]
            + [(f"weight{k}_dram", placements[k + 1]) for k in range(4)],
        brisc_named_compile_time_args=common + [("output_dram", placements[5])],
        trisc_named_compile_time_args=common,
        ncrisc_common_runtime_args=[*addresses[:5], addresses[6]],
        brisc_common_runtime_args=[addresses[5]],
        trisc_compute_config=compute,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=worker_args, brisc_args=worker_args, trisc_args=worker_args),
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])
