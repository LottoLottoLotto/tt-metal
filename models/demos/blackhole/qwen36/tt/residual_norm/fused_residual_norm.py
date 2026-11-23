# SPDX-License-Identifier: Apache-2.0
"""Single-launch P150 residual/add + native width-sharded RMSNorm + DRAM output.

No native arithmetic kernel is edited. Add uses separate cores so its BF16 DEST
configuration and pack boundary do not become the norm's FP32 intermediates.
The residual DRAM materialization is retained because the caller needs that sum.
Decode and speculative verification share the same complete 32-row physical
tile. Logical row count changes metadata only; the native descriptor is built
with that count, and no state/cache operation is performed here.
"""
from pathlib import Path
import math

import ttnn

_ROOT = Path(__file__).resolve().parent
_ADD_COMPUTE = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp"
_NORM_COMPUTE = "layernorm_sharded.cpp"
_READERS = {"reader_mcast_sender_unary_sharded_ln.cpp", "reader_mcast_receiver_unary_sharded_ln.cpp"}
_WRITERS = {"writer_unary_sharded_ln_rm_gb.cpp", "writer_unary_sharded_ln.cpp"}


def _core_set(coords):
    return ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in coords})


def _accessors(*tensors):
    result = []
    for tensor in tensors:
        args = ttnn.TensorAccessorArgs(tensor)
        if args.get_common_runtime_args():
            raise ValueError("Residual norm requires compile-time tensor accessors")
        result.extend(args.get_compile_time_args())
    return result


def _coords(core_set):
    result = []
    for core_range in core_set.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                result.append(ttnn.CoreCoord(x, y))
    return sorted(result, key=lambda c: (c.y, c.x))


class FusedResidualNorm:
    """Packed [1,T,W], 1 <= T <= 32, BF16 TILE residual/norm runner.

    The supplied get_decode_norm_config(args) controls the *native factory*.
    Admission is limited to a native width-sharded single-tile-row geometry whose
    norm grid is a gx x gy rectangle at the origin of the P150 11x10 worker grid
    (27B: 8x5 [32,128], W=5120; 35B: 8x8 [32,32], W=2048).
    Each call dispatches one generic_op, not separate add/shard/norm/unshard ops.
    With residual=None, residual_sum is x itself and its ownership is unchanged.
    Otherwise residual_sum and normed are fresh, independently owned DRAM tensors.
    No input, weight, returned tensor, or persistent scratch is cached/deallocated.
    """

    def __init__(self, device, norm_config, compute_kernel_config):
        self.device = device
        self.memory = norm_config["sharded_output_config"]
        self.program_config = norm_config["sharded_program_config"]
        self.compute_kernel_config = compute_kernel_config
        grid = device.compute_with_storage_grid_size()
        if (grid.x, grid.y) != (11, 10):
            raise ValueError("FusedResidualNorm requires a single P150 11x10 worker grid")
        if hasattr(device, "get_num_devices") and device.get_num_devices() != 1:
            raise ValueError("FusedResidualNorm does not support multi-device meshes")
        spec = self.memory.shard_spec
        pc = self.program_config
        gx, gy = pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y
        expected = [(x, y) for y in range(gy) for x in range(gx)]
        shard_w = tuple(spec.shape)[1] if spec is not None else 0
        if (
            self.memory.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
            or self.memory.buffer_type != ttnn.BufferType.L1
            or spec is None
            or tuple(spec.shape)[0] != 32 or shard_w == 0 or shard_w % 32 != 0
            or spec.orientation != ttnn.ShardOrientation.ROW_MAJOR
            or [(c.x, c.y) for c in _coords(spec.grid)] != expected
            or gx > 11 or gy > 10
            or (pc.block_h, pc.block_w) != (1, shard_w // 32)
            or pc.inplace
            or pc.use_welford
        ):
            raise ValueError("Unsupported norm geometry: need native width-sharded [32,32k] on an origin rectangle")
        self.tiles = shard_w // 32
        self.width = shard_w * len(expected)
        self.norm_cores = [ttnn.CoreCoord(x, y) for x, y in expected]
        # Separate add cores: rows below the norm rectangle first, then the columns right of it. Each add core
        # produces the residual tiles of `group` consecutive norm cores (27B 8x5: one each, at (x, y+5)).
        free = [(x, y) for y in range(gy, 10) for x in range(gx)] + [(x, y) for x in range(gx, 11) for y in range(10)]
        n = len(expected)
        self.group = next(k for k in range(1, n + 1) if n % k == 0 and n // k <= len(free))
        self.add_cores = [ttnn.CoreCoord(x, y) for x, y in free[: n // self.group]]
        self.norm_set = _core_set(self.norm_cores)
        self.add_set = _core_set(self.add_cores)
        self.all_set = _core_set(self.norm_cores + self.add_cores)
        self.norm_physical = [device.worker_core_from_logical_core(c) for c in self.norm_cores]

    def _check_tensor(self, tensor, name):
        W = self.width
        if (
            len(tensor.shape) != 3
            or tensor.shape[0] != 1 or not 1 <= tensor.shape[1] <= 32 or tensor.shape[2] != W
            or tuple(tensor.padded_shape) != (1, 32, W)
            or tensor.dtype != ttnn.bfloat16
            or tensor.layout != ttnn.TILE_LAYOUT
            or tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
            or tensor.memory_config().buffer_type not in (ttnn.BufferType.DRAM, ttnn.BufferType.L1)
            or tensor.device() != self.device
        ):
            raise ValueError(f"{name} must be interleaved BF16 TILE [1,T,{W}], 1 <= T <= 32 on the runner device")

    def _native_descriptor(self, weight, eps, rows):
        # Tensor metadata is required by the native descriptor factory, but none
        # of its pinned input/output buffers remain in the emitted descriptor.
        # This transient allocation is not a data movement operation, and avoids
        # 128 per-norm permanent scratch allocations in the full decoder.
        metadata = ttnn.empty(
            [1, 1, rows, self.width], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=self.memory,
        )
        try:
            params = ttnn.LayerNormParams()
            params.norm_type = ttnn.LayerNormType.RMSNORM
            params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
            params.eps = eps
            params.output_mem_config = self.memory
            params.program_config = self.program_config
            params.compute_kernel_config = self.compute_kernel_config
            params.dtype = ttnn.bfloat16
            inputs = ttnn.LayerNormInputs(metadata)
            inputs.weight = weight
            native = ttnn.LayerNormShardedProgramFactory.create_descriptor(params, inputs, metadata)
            # Both endpoints become ordinary local CBs of identical BF16 size.
            # All intermediate/statistic CB formats, sizes and allocation order
            # remain exactly those chosen by the pinned native factory.
            cbs = list(native.cbs)
            for cb in cbs:
                indices = {f.buffer_index for f in cb.format_descriptors}
                if indices in ({0}, {16}):
                    cb.set_buffer_from_cb(ttnn.CBDescriptor())
                elif cb.has_buffer():
                    raise ValueError(f"Unexpected pinned native norm CB: {indices}")
            native.cbs = cbs
            return native
        finally:
            ttnn.deallocate(metadata)

    def _program(self, x, weight, eps, residual, residual_sum, normed):
        native = self._native_descriptor(weight, eps, int(x.shape[1]))
        gx = self.program_config.compute_with_storage_grid_size.x
        kernels = list(native.kernels)
        for kernel in kernels:
            basename = Path(kernel.kernel_source).name
            extra = [("rn_tiles", self.tiles)]
            if kernel.common_runtime_args:
                raise ValueError("Unexpected native norm common runtime args")
            if basename in _READERS:
                kernel.defines = list(kernel.defines) + (
                    [("RN_SENDER", "1")] if "sender" in basename else []
                ) + ([("RN_RESIDUAL", "1")] if residual is not None else [])
                extra.append(("rn_accessor_offset", len(kernel.compile_time_args)))
                kernel.compile_time_args = list(kernel.compile_time_args) + _accessors(residual_sum)
                kernel.common_runtime_args = [int(residual_sum.buffer_address())]
                cores = _coords(kernel.core_ranges)
                lengths = {len(kernel.runtime_args[c.x][c.y]) for c in cores}
                if len(lengths) != 1:
                    raise ValueError("Unexpected native reader per-core argument lengths")
                extra.append(("rn_rt_offset", lengths.pop()))
                for c in cores:
                    kernel.runtime_args[c.x].append(c.y, (c.y * gx + c.x) * self.tiles)
                kernel.kernel_source = str(_ROOT / "norm_reader.cpp")
            elif basename in _WRITERS:
                if ("SKIP_WRITE_BACK", "1") not in list(kernel.defines):
                    raise ValueError("Native norm writer must skip reshard write-back")
                if basename.endswith("_rm_gb.cpp"):
                    kernel.defines = list(kernel.defines) + [("RN_WEIGHT_RM", "1")]
                extra.append(("rn_accessor_offset", len(kernel.compile_time_args)))
                kernel.compile_time_args = list(kernel.compile_time_args) + _accessors(normed)
                kernel.common_runtime_args = [int(normed.buffer_address())]
                kernel.kernel_source = str(_ROOT / "norm_writer.cpp")
            elif basename == _NORM_COMPUTE:
                if any(name == "FUSE_PRE_ADD" for name, _ in kernel.defines):
                    raise ValueError("Native FUSE_PRE_ADD changes the BF16 residual pack boundary")
                kernel.kernel_source = str(_ROOT / "norm_compute.cpp")
            else:
                raise ValueError(f"Unexpected native RMSNorm kernel: {basename}")
            kernel.named_compile_time_args = list(kernel.named_compile_time_args) + extra
        native.kernels = kernels
        if residual is None:
            return native

        native.semaphores = list(native.semaphores) + [
            ttnn.SemaphoreDescriptor(id=3, core_ranges=self.all_set, initial_value=0)
        ]
        per_add = self.tiles * self.group
        producer_reader = ttnn.KernelDescriptor(
            kernel_source=str(_ROOT / "add_reader.cpp"), core_ranges=self.add_set,
            compile_time_args=[per_add] + _accessors(x, residual),
            runtime_args=[(c, [int(x.buffer_address()), int(residual.buffer_address()), index * per_add])
                          for index, c in enumerate(self.add_cores)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        producer_writer = ttnn.KernelDescriptor(
            kernel_source=str(_ROOT / "add_writer.cpp"), core_ranges=self.add_set,
            compile_time_args=[per_add, self.group] + _accessors(residual_sum),
            runtime_args=[(c, [int(residual_sum.buffer_address()), index * per_add] + [
                v for k in range(self.group) for v in (self.norm_physical[index * self.group + k].x,
                                                       self.norm_physical[index * self.group + k].y)])
                          for index, c in enumerate(self.add_cores)],
            config=ttnn.WriterConfigDescriptor(),
        )
        # Same no-broadcast FPU kernel, one tile per cycle and FP32_DEST disabled
        # as native interleaved BF16 ttnn.add. Separate cores allow native RMSNorm
        # to retain its own (usually FP32_DEST-enabled) compute configuration.
        producer_compute = ttnn.KernelDescriptor(
            kernel_source=_ADD_COMPUTE, core_ranges=self.add_set,
            compile_time_args=[1], runtime_args=[(c, [per_add]) for c in self.add_cores],
            defines=[("BINARY_OP", "add_tiles"), ("BINARY_OP_TYPE", "EltwiseBinaryType::ELWADD"),
                     ("PROCESS_LHS_ACTIVATIONS(i)", ""), ("PROCESS_RHS_ACTIVATIONS(i)", ""),
                     ("PROCESS_POST_ACTIVATIONS(i)", "")],
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False),
        )
        tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
        add_cbs = [ttnn.CBDescriptor(
            total_size=per_add * 2048, core_ranges=self.add_set,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, 2048, tile)],
        ) for index in (0, 1, 2)]
        native.kernels = list(native.kernels) + [producer_reader, producer_writer, producer_compute]
        native.cbs = list(native.cbs) + add_cbs
        return native

    def __call__(self, x, weight, eps, residual=None):
        self._check_tensor(x, "x")
        if residual is not None:
            self._check_tensor(residual, "residual")
            if tuple(residual.shape) != tuple(x.shape):
                raise ValueError("residual and x must have identical logical shapes")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("epsilon must be finite and positive")
        if (weight.device() != self.device or weight.dtype != ttnn.bfloat16
                or weight.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
                or weight.memory_config().buffer_type not in (ttnn.BufferType.DRAM, ttnn.BufferType.L1)):
            raise ValueError("weight must be interleaved BF16 on the runner device")
        W = self.width
        if weight.layout == ttnn.ROW_MAJOR_LAYOUT:
            if tuple(weight.shape) != (1, 1, W // 32, 32) or tuple(weight.padded_shape) != (1, 1, W // 32, 32):
                raise ValueError(f"Native row-major norm weight must be [1,1,{W // 32},32]")
        elif weight.layout == ttnn.TILE_LAYOUT:
            if math.prod(weight.shape) != W or tuple(weight.padded_shape)[-2:] != (32, W):
                raise ValueError(f"Tiled norm weight must be one {W}-wide row")
        else:
            raise ValueError("Unsupported norm weight layout")
        normed = ttnn.empty(tuple(x.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        residual_sum = x if residual is None else ttnn.empty(
            tuple(x.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tensors = [x, weight, normed]
        if residual is not None:
            tensors.extend((residual, residual_sum))
        try:
            program = self._program(x, weight, float(eps), residual, residual_sum, normed)
            ttnn.generic_op(tensors, program)
        except BaseException:
            ttnn.deallocate(normed)
            if residual is not None:
                ttnn.deallocate(residual_sum)
            raise
        return normed, residual_sum
