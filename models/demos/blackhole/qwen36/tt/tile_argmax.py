# SPDX-License-Identifier: Apache-2.0
"""Row argmax of BF16 TILE logits without the untilize (replaces to_layout(ROW_MAJOR) + ttnn.argmax).

For 1-16 logit rows (MTP-1 verify / draft heads, batched decode buckets): only the face rows holding the tokens are
read from each tile; 110 cores x 2 data-movement RISCs scan contiguous column ranges and core 0 gathers. Ties resolve to the lowest
column, as ttnn.argmax does; the result is the same uint32 row-major [.., rows] tensor.
"""
import math

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

_KERNEL = "models/demos/blackhole/qwen36/tt/tile_argmax_kernel.cpp"


class TileArgmax:
    def __init__(self):
        self.signature = None
        self.program = None

    def _make_program(self, logits, out, rows, tiles):
        device = logits.device()
        grid = device.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        workers = 2 * len(cores)
        if tiles < workers:
            raise ValueError("TileArgmax needs at least one tile per worker")
        core_set = ttnn.CoreRangeSet({ttnn.CoreRange(cores[0], cores[-1])})
        gather = device.worker_core_from_logical_core(cores[0])
        face_bytes = -(-rows * 32 // 64) * 64
        slot_bytes = -(-rows * 8 // 16) * 16
        named = [("rows", rows), ("tiles", tiles), ("workers", workers), ("gather_x", gather.x), ("gather_y", gather.y),
                 ("face_bytes", face_bytes), ("slot_bytes", slot_bytes)]
        common = [int(logits.buffer_address()), int(out.buffer_address())]
        per_core = [(c, [i]) for i, c in enumerate(cores)]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL, core_ranges=core_set,
            brisc_named_compile_time_args=named, ncrisc_named_compile_time_args=named,
            brisc_common_runtime_args=common, ncrisc_common_runtime_args=common,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(brisc_args=per_core, ncrisc_args=per_core),
        )
        span = max(1, math.ceil(tiles / workers)) * 2 * face_bytes
        cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=core_set,
                                 format_descriptors=[ttnn.CBFormatDescriptor(cb, ttnn.uint32, size)])
               for cb, size in ((0, span), (1, span), (2, workers * slot_bytes))]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=core_set, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=sems)

    def __call__(self, logits):
        shape = list(logits.shape)
        rows = math.prod(shape[:-1])
        width = shape[-1]
        if (logits.dtype != ttnn.bfloat16 or logits.layout != ttnn.TILE_LAYOUT or not 1 <= rows <= 16
                or width % 32 or int(logits.padded_shape[-2]) != 32 or math.prod(list(logits.padded_shape)[:-2]) != 1
                or logits.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError("TileArgmax needs interleaved DRAM BF16 TILE logits with 1-16 rows in one tile row")
        out = ttnn.empty(shape[:-1], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=logits.device(),
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        signature = (int(logits.buffer_address()), int(out.buffer_address()), tuple(shape))
        if signature != self.signature:
            self.program = self._make_program(logits, out, rows, width // 32)
            self.signature = signature
        ttnn.generic_op([logits, out], self.program)
        return out
