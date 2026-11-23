# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape single-P150 MTP1 GDN commit; no arithmetic on state values."""
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

_KERNEL = "models/demos/blackhole/qwen36/tt/gdn/fused_commit_kernel.cpp"


def _physical(tensor):
    spec = tensor.memory_config().shard_spec
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError("fused commit requires row-major shard ordering")
    return [tensor.device().worker_core_from_logical_core(c) for c in ttnn.corerange_to_cores(spec.grid, row_wise=True)]


def _make_program(flag, s1, s2, c1, c2, state, conv):
    for tensor in (s1, s2):
        if tuple(tensor.shape) != (1, 32, 128, 128) or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT or tensor.is_sharded():
            raise ValueError("fused commit requires interleaved BF16 tiled [1,32,128,128] candidates")
    for tensor in (c1, c2):
        if tuple(tensor.shape) != (1, 3, 8192) or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT or tensor.is_sharded():
            raise ValueError("fused commit requires interleaved BF16 row-major [1,3,8192] conv candidates")
    if s1.memory_config() != s2.memory_config() or c1.memory_config() != c2.memory_config():
        raise ValueError("candidate pairs must use identical memory configurations")
    if tuple(flag.shape) != (1, 1) or flag.dtype != ttnn.int32 or flag.layout != ttnn.ROW_MAJOR_LAYOUT or flag.is_sharded():
        raise ValueError("fused commit requires interleaved int32 [1,1] acceptance flag")
    for tensor, shape, width in ((state, (1, 32, 128, 128), 128), (conv, (1, 3, 8192), 8192)):
        mc = tensor.memory_config()
        if tuple(tensor.shape) != shape or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT or mc.buffer_type != ttnn.BufferType.L1 or mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED or mc.shard_spec.shape[1] != width:
            raise ValueError("fused commit requires compact BF16 height-sharded live L1 state")
    rec_cores, conv_cores = _physical(state), _physical(conv)
    if len(conv_cores) != 3 or conv.memory_config().shard_spec.shape[0] != 1:
        raise ValueError("fused commit requires one convolution row per shard")
    shard_rows = state.memory_config().shard_spec.shape[0]
    if shard_rows * len(rec_cores) < 4096:
        raise ValueError("recurrent shards do not cover all rows")
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
    args = [int(t.buffer_address()) for t in (flag, s1, s2, c1, c2, state, conv)]
    args += [c.x for c in rec_cores] + [c.y for c in rec_cores] + [c.x for c in conv_cores] + [c.y for c in conv_cores]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        brisc_named_compile_time_args=[("flag_dram", int(flag.memory_config().buffer_type == ttnn.BufferType.DRAM)), ("state_dram", int(s1.memory_config().buffer_type == ttnn.BufferType.DRAM)), ("conv_dram", int(c1.memory_config().buffer_type == ttnn.BufferType.DRAM)), ("shard_rows", shard_rows), ("state_cores", len(rec_cores))],
        brisc_common_runtime_args=args,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(brisc_args=[(ttnn.CoreCoord(i % 8, i // 8), [i]) for i in range(32)]),
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
        tensors = (flag, s1, s2, c1, c2, state, conv)
        signature = tuple((t.buffer_address(), tuple(t.shape), str(t.memory_config()), t.dtype, t.layout) for t in tensors)
        if signature != self.signature:
            self.program = _make_program(*tensors)
            self.signature = signature
        ttnn.generic_op(list(tensors), self.program)
