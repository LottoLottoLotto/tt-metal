# SPDX-License-Identifier: Apache-2.0
"""Single-P150 MTP/DFlash GDN commit for many layers per launch; no arithmetic on state values.

Byte-identical to one FusedCommit per layer: the selected TILE candidate is untilized
into the compact row-major live L1 state and the selected convolution rows are copied.
Ninety-six workers (two per head) pipeline reads, untilize and writes across layers.
"""
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

_KERNEL = "models/demos/blackhole/qwen36/tt/gdn/batched_commit_kernel.cpp"
_LAYERS_PER_LAUNCH = 12
_MAX_COMMON_ARGS = 300  # below the 341 runtime-arg limit, with room for per-core args


def _physical(tensor):
    spec = tensor.memory_config().shard_spec
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError("batched commit requires row-major shard ordering")
    return [tensor.device().worker_core_from_logical_core(c) for c in ttnn.corerange_to_cores(spec.grid, row_wise=True)]


def _validate(flag, layer, candidates):
    if len(layer) != 2 * candidates + 2:
        raise ValueError("batched commit layer entries are (states..., convs..., live state, live conv)")
    states, convs = layer[:candidates], layer[candidates:2 * candidates]
    state, conv = layer[-2:]
    for tensor in states:
        if tuple(tensor.shape) != (1, 48, 128, 128) or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("batched commit requires interleaved BF16 tiled DRAM [1,48,128,128] candidates")
    for tensor in convs:
        if tuple(tensor.shape) != (1, 3, 10240) or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("batched commit requires interleaved BF16 row-major DRAM [1,3,10240] conv candidates")
    if tuple(flag.shape) != (1, 1) or flag.dtype != ttnn.int32 or flag.layout != ttnn.ROW_MAJOR_LAYOUT or flag.is_sharded():
        raise ValueError("batched commit requires an interleaved int32 [1,1] acceptance flag")
    for tensor, shape, width in ((state, (1, 48, 128, 128), 128), (conv, (1, 3, 10240), 10240)):
        mc = tensor.memory_config()
        if tuple(tensor.shape) != shape or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT or mc.buffer_type != ttnn.BufferType.L1 or mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED or mc.shard_spec.shape[1] != width:
            raise ValueError("batched commit requires compact BF16 height-sharded live L1 state")
    if len(_physical(conv)) != 1 or conv.memory_config().shard_spec.shape[0] != 3:
        raise ValueError("batched commit requires all three convolution rows on one core")
    if state.memory_config().shard_spec.shape[0] != 56 or len(_physical(state)) != 110:
        raise ValueError("batched commit requires 110 recurrent shards of 56 rows")


def _make_program(flag, layers, candidates):
    rec_cores = _physical(layers[0][-2])
    for layer in layers:
        _validate(flag, layer, candidates)
        if _physical(layer[-2]) != rec_cores:
            raise ValueError("batched commit requires every live state on the same shard cores")
    grid = flag.device().compute_with_storage_grid_size()
    if grid.x < 11 or grid.y < 9:
        raise ValueError("batched commit requires an 11x9 worker grid")
    workers = [ttnn.CoreCoord(i % 11, i // 11) for i in range(96)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in workers})
    args = [int(flag.buffer_address())]
    for layer in layers:
        args += [int(t.buffer_address()) for t in layer]
    args += [c.x | (c.y << 16) for c in rec_cores]
    args += [(lambda c: c.x | (c.y << 16))(_physical(layer[-1])[0]) for layer in layers]
    per_core = [(c, [i // 2, i % 2]) for i, c in enumerate(workers)]
    flag_dram = int(flag.memory_config().buffer_type == ttnn.BufferType.DRAM)
    dataflow_args = [("layers", len(layers)), ("shard_rows", 56), ("state_cores", 110), ("flag_dram", flag_dram),
                     ("candidates", candidates)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        ncrisc_named_compile_time_args=dataflow_args,
        brisc_named_compile_time_args=dataflow_args,
        trisc_named_compile_time_args=[("layers", len(layers)), ("shard_rows", 56), ("state_cores", 110)],
        ncrisc_common_runtime_args=args,
        brisc_common_runtime_args=args,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(ncrisc_args=per_core, brisc_args=per_core),
        trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, dst_full_sync_en=False),
    )
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    cbs = [ttnn.CBDescriptor(total_size=count * 2048, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(cb, ttnn.bfloat16, 2048, tile)]) for cb, count in ((0, 8), (16, 8), (2, 1), (3, 1))]
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])


class BatchedCommit:
    def __init__(self):
        self.signatures = {}
        self.programs = {}

    def __call__(self, flag, layers):
        """layers: per GDN layer (state candidates..., conv candidates..., live_state, live_conv).

        The int32 flag selects candidate min(flag, C-1) for C=2 (accept flag) or
        candidate flag for C>2 (accepted_count), as FusedCommit does.
        """
        layers = [tuple(layer) for layer in layers]
        candidates = (len(layers[0]) - 2) // 2
        # Common args: flag, per-layer (2C+2) addresses, 110 state cores, per-layer conv core.
        per_launch = min(_LAYERS_PER_LAUNCH, (_MAX_COMMON_ARGS - 111) // (2 * candidates + 3))
        for start in range(0, len(layers), per_launch):
            chunk = layers[start:start + per_launch]
            tensors = [flag] + [t for layer in chunk for t in layer]
            signature = tuple((t.buffer_address(), tuple(t.shape), str(t.memory_config()), t.dtype, t.layout) for t in tensors)
            if self.signatures.get(start) != signature:
                self.programs[start] = _make_program(flag, chunk, candidates)
                self.signatures[start] = signature
            ttnn.generic_op(tensors, self.programs[start])
