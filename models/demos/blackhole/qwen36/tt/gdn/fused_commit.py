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
    heads = int(state.shape[1])
    conv_width = int(conv.shape[-1])
    state_shape = (1, heads, 128, 128)
    for tensor in recurrent_candidates:
        if tuple(tensor.shape) != state_shape or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError(f"fused commit requires interleaved BF16 tiled DRAM {list(state_shape)} candidates")
    for tensor in convolution_candidates:
        if tuple(tensor.shape) != (1, 3, conv_width) or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError(f"fused commit requires interleaved BF16 row-major DRAM [1,3,{conv_width}] conv candidates")
    if tuple(flag.shape) != (1, 1) or flag.dtype != ttnn.int32 or flag.layout != ttnn.ROW_MAJOR_LAYOUT or flag.is_sharded():
        raise ValueError("fused commit requires interleaved int32 [1,1] acceptance flag")
    for tensor, shape, width in ((state, state_shape, 128), (conv, (1, 3, conv_width), conv_width)):
        mc = tensor.memory_config()
        if tuple(tensor.shape) != shape or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT or mc.buffer_type != ttnn.BufferType.L1 or mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED or mc.shard_spec.shape[1] != width:
            raise ValueError("fused commit requires compact BF16 height-sharded live L1 state")
    rec_cores, conv_cores = _physical(state), _physical(conv)
    if len(conv_cores) != 1 or conv.memory_config().shard_spec.shape[0] != 3:
        raise ValueError("fused commit requires all three convolution rows on one core")
    shard_rows = state.memory_config().shard_spec.shape[0]
    if shard_rows * len(rec_cores) < heads * 128:
        raise ValueError("fused commit: recurrent shards do not cover the state")
    # One worker per head; each also copies one 512-byte stripe of the conv rows, so the
    # heads must cover the conv width.
    if heads * 512 < conv_width * 2:
        raise ValueError("fused commit: head workers cannot cover the convolution rows")
    grid = state.device().compute_with_storage_grid_size()
    if grid.x < 8 or heads > 8 * grid.y:
        raise ValueError(f"fused commit needs an 8-wide grid with {heads} head workers")
    if any(t.device() != state.device() for t in (flag, *recurrent_candidates, *convolution_candidates, conv)):
        raise ValueError("fused commit tensors must share one device")
    worker_cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(heads)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in worker_cores})
    args = [int(t.buffer_address()) for t in (flag, *recurrent_candidates, *convolution_candidates, state, conv)]
    args += [c.x for c in rec_cores] + [c.y for c in rec_cores] + [c.x for c in conv_cores] + [c.y for c in conv_cores]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        brisc_named_compile_time_args=[("flag_dram", int(flag.memory_config().buffer_type == ttnn.BufferType.DRAM)), ("state_dram", 1), ("conv_dram", 1), ("shard_rows", shard_rows), ("state_cores", len(rec_cores)), ("num_candidates", count), ("conv_width", conv_width)],
        brisc_common_runtime_args=args,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(brisc_args=[(c, [i]) for i, c in enumerate(worker_cores)]),
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
