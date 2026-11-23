# SPDX-License-Identifier: Apache-2.0
"""Single-P150 MTP-1 commit of all GDN layers in one launch (the per-layer FusedCommit copies, batched).

Same selection rule and the same lossless BF16 untilize/copy as FusedCommit, so the committed live state is
bitwise identical; one program deals all layer x head jobs over every worker core instead of 30 serial launches.
"""
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

_KERNEL = "models/demos/blackhole/qwen36/tt/gdn/fused_commit_all_kernel.cpp"


def _xy(core):
    return int(core.x) | (int(core.y) << 16)


def _physical(tensor):
    spec = tensor.memory_config().shard_spec
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError("fused commit requires row-major shard ordering")
    return [tensor.device().worker_core_from_logical_core(c) for c in ttnn.corerange_to_cores(spec.grid, row_wise=True)]


class FusedCommitAll:
    """commit(flag, [(s1, s2, c1, c2, live_state, live_conv), ...]) for every GDN layer of an MTP-1 cycle."""

    def __init__(self):
        self.signature = None
        self.program = None

    def _make_program(self, flag, layers):
        if not layers:
            raise ValueError("fused commit needs at least one layer")
        state0, conv0 = layers[0][4], layers[0][5]
        heads = int(state0.shape[1])
        conv_width = int(conv0.shape[-1])
        state_shape = (1, heads, 128, 128)
        shard_cores = _physical(state0)
        shard_rows = state0.memory_config().shard_spec.shape[0]
        if shard_rows * len(shard_cores) < heads * 128:
            raise ValueError("fused commit: recurrent shards do not cover the state")
        if heads * 512 < conv_width * 2:
            raise ValueError("fused commit: head workers cannot cover the convolution rows")
        if tuple(flag.shape) != (1, 1) or flag.dtype != ttnn.int32 or flag.layout != ttnn.ROW_MAJOR_LAYOUT or flag.is_sharded():
            raise ValueError("fused commit requires interleaved int32 [1,1] acceptance flag")
        device = state0.device()
        args = [int(flag.buffer_address())]
        for s1, s2, c1, c2, state, conv in layers:
            for tensor in (s1, s2):
                if (tuple(tensor.shape) != state_shape or tensor.dtype != ttnn.bfloat16
                        or tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
                    raise ValueError(f"fused commit requires interleaved BF16 tiled DRAM {list(state_shape)} candidates")
            for tensor in (c1, c2):
                if (tuple(tensor.shape) != (1, 3, conv_width) or tensor.dtype != ttnn.bfloat16
                        or tensor.layout != ttnn.ROW_MAJOR_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
                    raise ValueError(f"fused commit requires interleaved BF16 row-major DRAM [1,3,{conv_width}] conv candidates")
            for tensor, shape, width in ((state, state_shape, 128), (conv, (1, 3, conv_width), conv_width)):
                mc = tensor.memory_config()
                if (tuple(tensor.shape) != shape or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
                        or mc.buffer_type != ttnn.BufferType.L1
                        or mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED or mc.shard_spec.shape[1] != width):
                    raise ValueError("fused commit requires compact BF16 height-sharded live L1 state")
            if (state.memory_config().shard_spec.shape[0] != shard_rows
                    or [(c.x, c.y) for c in _physical(state)] != [(c.x, c.y) for c in shard_cores]):
                raise ValueError("fused commit: every layer's live state must share one shard placement")
            conv_cores = _physical(conv)
            if len(conv_cores) != 1 or conv.memory_config().shard_spec.shape[0] != 3:
                raise ValueError("fused commit requires all three convolution rows on one core")
            if any(t.device() != device for t in (flag, s1, s2, c1, c2, conv)):
                raise ValueError("fused commit tensors must share one device")
            args += [int(t.buffer_address()) for t in (s1, s2, c1, c2, state, conv)] + [_xy(conv_cores[0])]
        args += [_xy(c) for c in shard_cores]
        grid = device.compute_with_storage_grid_size()
        workers = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][: len(layers) * heads]
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in workers})
        named = [("heads", heads), ("layers", len(layers)), ("workers", len(workers))]
        per_core = [(c, [i]) for i, c in enumerate(workers)]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL, core_ranges=cores,
            brisc_named_compile_time_args=named + [
                ("flag_dram", int(flag.memory_config().buffer_type == ttnn.BufferType.DRAM)), ("conv_width", conv_width)],
            ncrisc_named_compile_time_args=named + [("shard_rows", shard_rows)],
            trisc_named_compile_time_args=named,
            brisc_common_runtime_args=args,
            ncrisc_common_runtime_args=args,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=per_core, ncrisc_args=per_core, trisc_args=per_core),
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False,
                dst_full_sync_en=False),
        )
        tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
        cbs = [ttnn.CBDescriptor(total_size=tiles * 2048, core_ranges=cores,
                                 format_descriptors=[ttnn.CBFormatDescriptor(cb, ttnn.bfloat16, 2048, tile)])
               for cb, tiles in ((0, 8), (16, 8), (2, 1))]
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, flag, layers):
        layers = [tuple(entry) for entry in layers]
        tensors = [flag] + [t for entry in layers for t in entry]
        signature = tuple((t.buffer_address(), tuple(t.shape), str(t.memory_config()), t.dtype, t.layout) for t in tensors)
        if signature != self.signature:
            self.program = self._make_program(flag, layers)
            self.signature = signature
        ttnn.generic_op(tensors, self.program)
