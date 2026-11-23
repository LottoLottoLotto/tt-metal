# SPDX-License-Identifier: Apache-2.0
"""Single-P150 accepted-prefix GDN commit; no arithmetic on state values."""
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

_KERNEL = "models/demos/blackhole/qwen36/tt/gdn/fused_commit_kernel.cpp"


def _physical(tensor):
    spec = tensor.memory_config().shard_spec
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError("fused commit requires row-major shard ordering")
    return [tensor.device().worker_core_from_logical_core(c) for c in ttnn.corerange_to_cores(spec.grid, row_wise=True)]


def _make_program(flag, recurrent_candidates, convolution_candidates, state, conv):
    count = len(recurrent_candidates)
    if not 2 <= count <= 32 or len(convolution_candidates) != count:
        raise ValueError("fused commit requires 2..32 paired prefix candidates")
    for tensor in recurrent_candidates:
        if tuple(tensor.shape) != (1, 48, 128, 128) or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("fused commit requires interleaved BF16 tiled DRAM [1,48,128,128] candidates")
    for tensor in convolution_candidates:
        if tuple(tensor.shape) != (1, 3, 10240) or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("fused commit requires interleaved BF16 row-major DRAM [1,3,10240] conv candidates")
    if tuple(flag.shape) != (1, 1) or flag.dtype != ttnn.int32 or flag.layout != ttnn.ROW_MAJOR_LAYOUT or flag.is_sharded():
        raise ValueError("fused commit requires interleaved int32 [1,1] acceptance flag")
    for tensor, shape, width in ((state, (1, 48, 128, 128), 128), (conv, (1, 3, 10240), 10240)):
        mc = tensor.memory_config()
        if tuple(tensor.shape) != shape or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT or mc.buffer_type != ttnn.BufferType.L1 or mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED or mc.shard_spec.shape[1] != width:
            raise ValueError("fused commit requires compact BF16 height-sharded live L1 state")
    rec_cores, conv_cores = _physical(state), _physical(conv)
    if len(conv_cores) != 1 or conv.memory_config().shard_spec.shape[0] != 3:
        raise ValueError("fused commit requires all three convolution rows on one core")
    shard_rows = state.memory_config().shard_spec.shape[0]
    if shard_rows != 56 or len(rec_cores) != 110:
        raise ValueError("fused commit requires 110 recurrent shards of 56 rows")
    grid = state.device().compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 6:
        raise ValueError("fused commit requires an 8x6 worker grid")
    if any(t.device() != state.device() for t in (flag, *recurrent_candidates, *convolution_candidates, conv)):
        raise ValueError("fused commit tensors must share one device")
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))})
    args = [int(t.buffer_address()) for t in (flag, *recurrent_candidates, *convolution_candidates, state, conv)]
    args += [c.x for c in rec_cores] + [c.y for c in rec_cores] + [c.x for c in conv_cores] + [c.y for c in conv_cores]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        brisc_named_compile_time_args=[("flag_dram", int(flag.memory_config().buffer_type == ttnn.BufferType.DRAM)), ("state_dram", 1), ("conv_dram", 1), ("shard_rows", shard_rows), ("state_cores", len(rec_cores)), ("num_candidates", count)],
        brisc_common_runtime_args=args,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(brisc_args=[(ttnn.CoreCoord(i % 8, i // 8), [i]) for i in range(48)]),
        trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, dst_full_sync_en=False),
    )
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    cbs = [ttnn.CBDescriptor(total_size=count * 2048, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(cb, ttnn.bfloat16, 2048, tile)]) for cb, count in ((0, 4), (16, 4), (2, 1))]
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])


class FusedCommit:
    def __init__(self):
        self.signature = None
        self.program = None

    def __call__(self, flag, s1, s2, c1, c2, state, conv):
        self.commit_many(flag, (s1, s2), (c1, c2), state, conv)

    def commit_many(self, accepted_count, recurrent_candidates, convolution_candidates, state, conv):
        recurrent_candidates = tuple(recurrent_candidates)
        convolution_candidates = tuple(convolution_candidates)
        tensors = (accepted_count, *recurrent_candidates, *convolution_candidates, state, conv)
        signature = (len(recurrent_candidates), len(convolution_candidates), tuple(
            (t.buffer_address(), tuple(t.shape), str(t.memory_config()), t.dtype, t.layout) for t in tensors
        ))
        if signature != self.signature:
            self.program = _make_program(accepted_count, recurrent_candidates, convolution_candidates, state, conv)
            self.signature = signature
        ttnn.generic_op(list(tensors), self.program)
