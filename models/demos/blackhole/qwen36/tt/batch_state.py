# SPDX-License-Identifier: Apache-2.0
"""Exact device-only slot transfers for row-major batched recurrent state."""
import math
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

_KERNEL = "models/demos/blackhole/qwen36/tt/batch_state_kernel.cpp"


def copy_state_rows(source, destination, source_rows, destination_rows):
    """Copy disjoint request slots; overlapping in-place permutations need staging."""
    if source is destination:
        raise ValueError("State row copies require distinct source and destination buffers")
    if len(source_rows) != len(destination_rows) or not source_rows:
        raise ValueError("State row mappings must be nonempty and equally sized")
    for tensor in (source, destination):
        if tensor.layout != ttnn.ROW_MAJOR_LAYOUT or tensor.dtype != ttnn.bfloat16 or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("State copies require interleaved BF16 row-major DRAM")
    if tuple(source.shape)[1:] != tuple(destination.shape)[1:]:
        raise ValueError("State slot geometries must match")
    if len(set(destination_rows)) != len(destination_rows):
        raise ValueError("State row destinations must be unique")
    if any(not 0 <= row < int(source.shape[0]) for row in source_rows) or any(not 0 <= row < int(destination.shape[0]) for row in destination_rows):
        raise ValueError("State row mapping exceeds allocated slots")
    row_bytes = int(source.shape[-1]) * 2
    pages_per_slot = math.prod(int(n) for n in tuple(source.shape)[1:-1])
    if row_bytes % 256:
        raise ValueError("State rows must be 256-byte aligned")
    grid = source.device().compute_with_storage_grid_size()
    workers = min(int(grid.x) * int(grid.y), pages_per_slot * (row_bytes // 256) * len(source_rows))
    coords = [ttnn.CoreCoord(i % int(grid.x), i // int(grid.x)) for i in range(workers)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in coords})
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        brisc_named_compile_time_args=[("row_bytes", row_bytes), ("pages_per_slot", pages_per_slot),
                                      ("copies", len(source_rows)), ("workers", workers)],
        brisc_common_runtime_args=[int(source.buffer_address()), int(destination.buffer_address()),
                                   *source_rows, *destination_rows],
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            brisc_args=[(core, [index]) for index, core in enumerate(coords)]),
        trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=False, dst_full_sync_en=False),
    )
    cb = ttnn.CBDescriptor(total_size=2048, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048, ttnn.TileDescriptor(ttnn.Tile((32, 32))))])
    program = ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=[cb], semaphores=[])
    ttnn.generic_op([source, destination], program)
