# SPDX-License-Identifier: Apache-2.0
"""P150 M1 recurrence/output with bulk state transport, overlap and mega gate.

Ordinary single-token decode only: the supplied live state is updated in place.
Verification uses FusedVerifyOutput and explicit transactional commit instead.
Arithmetic is the native HiFi2/FP32 recurrence and HiFi4/BF16 output; state
transport is bit-preserving UInt16 and the BF16 pack boundary is retained.
"""

import math
from pathlib import Path
import struct

import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor
from models.demos.blackhole.qwen36.tt.gdn.native_output import _cb_descriptors

_ROOT = Path(__file__).resolve().parent


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _cores(coords):
    return ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in coords})


def _accessors(tensors):
    args = []
    for tensor in tensors:
        accessor = ttnn.TensorAccessorArgs(tensor)
        if accessor.get_common_runtime_args():
            raise ValueError("M1 accessors require compile-time metadata")
        args.extend(accessor.get_compile_time_args())
    return args


def _state_routes(accessor_args):
    """Decode the immutable image's accessor metadata, not recurrence placement.

    ROW_MAJOR shard s is logical (s % 11, s // 11); recurrence head h is
    logical (h // 10, h % 10), which is unrelated. TensorAccessorArgs already
    resolves shard endpoints through virtual_core_from_logical_core, including
    device harvesting/virtualization. Reuse those exact NoC-zero coordinates
    rather than assuming a physical grid offset or a worker API's semantics.
    """
    args = tuple(int(value) for value in accessor_args)
    if len(args) < 4 or args[0] != 1 or args[1] != 256:
        raise ValueError("Fast state requires static L1-sharded accessor metadata, 256-byte pages")
    rank, banks = args[2:4]
    if not 1 <= rank <= 8 or banks != 110 or len(args) != 4 + 2 * rank + 55:
        raise ValueError("Fast state requires 110 round-robin banks with compile-time coordinates")
    tensor_shape = args[4:4 + rank]
    shard_shape = args[4 + rank:4 + 2 * rank]
    # BufferDistributionSpec squeezes singleton dimensions. Accept only the
    # equivalent one-dimensional 6144-stick tensor / 56-stick shard geometry.
    if [(size, shard) for size, shard in zip(tensor_shape, shard_shape)
            if (size, shard) != (1, 1)] != [(6144, 56)]:
        raise ValueError("Fast state requires exactly 6144 rows distributed in 56-row shards")
    packed_pairs = args[4 + 2 * rank:]
    endpoints = tuple((packed_pairs[s // 2] >> (16 * (s % 2))) & 0xffff for s in range(110))
    if len(set(endpoints)) != 110:
        raise ValueError("Fast state requires 110 distinct shard endpoints")
    routes = []
    for head in range(48):
        first_shard, first_row = divmod(head * 128, 56)
        last_shard = (head * 128 + 127) // 56
        # Three-shard heads never use word four. Duplicate their last valid
        # coordinate, including head 47: do not index nonexistent shard 110.
        routes.append([*(endpoints[min(first_shard + i, last_shard)] for i in range(4)), first_row])
    return routes


class FusedM1Output:
    """Keep ordinary recurrence arithmetic and update the supplied state in place.

    Forty-eight HiFi2/FP32 recurrence workers send one rounded BF16 row each
    to forty-eight disjoint HiFi4/BF16 output workers. The shared mailbox is
    256 bytes per worker. Request and ready semaphores return to zero after
    every invocation, including trace replay; layers may share the inbox.
    """

    def __init__(self, device, inbox=None, gate_column_offset=0):
        self.device = device
        if not isinstance(gate_column_offset, int) or gate_column_offset < 0 or gate_column_offset % 32:
            raise ValueError("gate column offset must be a nonnegative multiple of 32")
        self._gate_page_offset = gate_column_offset // 32
        self._gate_width = gate_column_offset + 6144
        if hasattr(device, "get_num_devices") and device.get_num_devices() != 1:
            raise ValueError("fused M1 requires a single-device mesh")
        grid = device.compute_with_storage_grid_size()
        if grid.x != 11 or grid.y != 10:
            raise ValueError("fused M1 requires the P150 11x10 worker grid")
        self.recurrence_cores = [ttnn.CoreCoord(h // 10, h % 10) for h in range(48)]
        self.output_cores = [ttnn.CoreCoord(5 + h % 6, h // 6) for h in range(48)]
        self.output_set = _cores(self.output_cores)
        self.inbox_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.output_set, [1, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        state_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))})
        self.state_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(state_grid, [56, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        if inbox is not None:
            self._check_inbox(inbox)
        self.inbox = inbox if inbox is not None else ttnn.from_torch(
            torch.zeros(1, 1, 48, 128, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=self.inbox_memory_config,
        )
        self.program = None
        self.signature = None

    def _check_inbox(self, inbox):
        if (
            tuple(inbox.shape) != (1, 1, 48, 128)
            or tuple(inbox.padded_shape) != (1, 1, 48, 128)
            or inbox.dtype != ttnn.bfloat16
            or inbox.layout != ttnn.ROW_MAJOR_LAYOUT
            or inbox.memory_config() != self.inbox_memory_config
            or inbox.device() != self.device
        ):
            raise ValueError("M1 inbox must be BF16 row-major [1,1,48,128], output-core sharded [1,128]")

    def _make_program(self, inputs, gate, gamma, output):
        q, k, v, a, b, state, db, ds, ones = inputs
        rec_set = _cores(self.recurrence_cores)
        all_set = _cores(self.recurrence_cores + self.output_cores)
        rec_phys = [self.device.worker_core_from_logical_core(c) for c in self.recurrence_cores]
        out_phys = [self.device.worker_core_from_logical_core(c) for c in self.output_cores]
        # C++ uses 1.0f / std::sqrt(float(K)): round sqrt first, then division.
        sqrt_k = struct.unpack("<f", struct.pack("<f", math.sqrt(128.0)))[0]
        ct = [4, 4, 1, _bits(1.0 / sqrt_k), _bits(1e-6)]
        reader = ttnn.KernelDescriptor(
            kernel_source=str(_ROOT / "m1_reader_overlap.cpp"),
            core_ranges=rec_set,
            compile_time_args=ct + _accessors(inputs),
            runtime_args=[
                (c, [h, *[int(t.buffer_address()) for t in inputs]]) for h, c in enumerate(self.recurrence_cores)
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=str(_ROOT / "m1_writer_overlap.cpp"),
            core_ranges=rec_set,
            compile_time_args=[4, 4] + _accessors((state,)),
            runtime_args=[
                (c, [h, int(state.buffer_address()), out_phys[h].x, out_phys[h].y, int(self.inbox.buffer_address())])
                for h, c in enumerate(self.recurrence_cores)
            ],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=str(_ROOT / "recurrent_overlap.cpp"),
            core_ranges=rec_set,
            compile_time_args=ct,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        )
        tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))

        def cb(indices, tiles):
            return ttnn.CBDescriptor(
                total_size=tiles * 2048,
                core_ranges=rec_set,
                format_descriptors=[ttnn.CBFormatDescriptor(i, ttnn.bfloat16, 2048, tile) for i in indices],
            )

        recurrence_cbs = [cb((i,), n) for i, n in ((0, 4), (1, 4), (2, 4), (3, 1), (4, 1))]
        recurrence_cbs.append(cb((5, 17), 16))
        recurrence_cbs.extend(
            cb((i,), n)
            for i, n in (
                (16, 4),
                (18, 16),
                (19, 4),
                (20, 4),
                (21, 4),
                (22, 4),
                (23, 16),
                (24, 16),
                (25, 1),
                (6, 1),
                (7, 1),
                (26, 1),
                (27, 4),
                (28, 1),
                (29, 1),
                (30, 4),
                (31, 4),
            )
        )
        cb_args = [
            ("cb_scaler", 3),
            ("cb_eps", 4),
            ("cb_in", 0),
            ("cb_inb", 15),
            ("cb_out", 8),
            ("cb_gamma", 2),
            ("cb_beta", 15),
            ("cb_xmm", 0),
            ("cb_ex", 5),
            ("cb_ex2", 6),
            ("cb_xmm2", 7),
            ("cb_ex2pe", 9),
            ("cb_fusion", 10),
            ("cb_in_rm", 15),
        ]
        token_args = [("num_tokens", 1), ("packed_rows", 0)]
        norm = UnifiedKernelDescriptor(
            kernel_source=str(_ROOT / "mega_gate_kernel.cpp"),
            core_ranges=self.output_set,
            ncrisc_named_compile_time_args=token_args,
            brisc_named_compile_time_args=[
                ("gate_is_dram", int(gate.memory_config().buffer_type == ttnn.BufferType.DRAM)),
                ("gamma_is_dram", int(gamma.memory_config().buffer_type == ttnn.BufferType.DRAM)),
                ("output_is_dram", 0),
                *token_args,
            ],
            trisc_compile_time_args=[4, 4, 1, 0, 0, 0, 0, 128, 32],
            trisc_named_compile_time_args=[*cb_args, *token_args],
            brisc_common_runtime_args=[
                int(self.inbox.buffer_address()),
                int(gate.buffer_address()),
                int(gamma.buffer_address()),
                int(output.buffer_address()),
            ],
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=[(c, [h, rec_phys[h].x, rec_phys[h].y]) for h, c in enumerate(self.output_cores)],
                trisc_args=[(c, [1]) for c in self.output_cores],
            ),
            defines=[("RMSNORM", "1")],
        )
        norm_cbs = _cb_descriptors()
        for descriptor in norm_cbs:
            descriptor.core_ranges = self.output_set
        program = ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute, *norm.get_kernel_descriptors().kernels],
            cbs=recurrence_cbs + norm_cbs,
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=all_set, initial_value=0) for i in (0, 1)],
        )

        kernels = list(program.kernels)
        reader, writer, compute, output_reader, output_writer, output_compute = kernels
        routes = _state_routes(writer.compile_time_args[2:])
        output_writer.named_compile_time_args = [
            *output_writer.named_compile_time_args,
            ("gate_page_offset", self._gate_page_offset),
        ]
        # RuntimeArgsView is indexed by logical x/y, not an iterable of pairs.
        reader.runtime_args = [
            (core, [*reader.runtime_args[core.x][core.y],
                    *writer.runtime_args[core.x][core.y][2:4]])
            for core in self.recurrence_cores
        ]
        output_writer.runtime_args = [
            (core, [*output_writer.runtime_args[core.x][core.y],
                    int(inputs[5].buffer_address()), *routes[head]])
            for head, core in enumerate(self.output_cores)
        ]
        tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
        scratch_cbs = [
            ttnn.CBDescriptor(
                total_size=16 * 2048,
                core_ranges=self.output_set,
                format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.uint16, 2048, tile) for index in indices],
            )
            for indices in ((20, 23), (21, 22))
        ]
        all_set = _cores(self.recurrence_cores + self.output_cores)
        # pybind vector properties may return copies; reassign complete lists.
        program.kernels = kernels
        program.cbs = [*program.cbs, *scratch_cbs]
        program.semaphores = [
            *program.semaphores,
            *(ttnn.SemaphoreDescriptor(id=index, core_ranges=all_set, initial_value=0)
              for index in (2, 3, 4, 5, 6)),
        ]
        return program

    def __call__(self, q, k, v, a, b, db, ds, ones, state, gate, gamma, output=None):
        if any(tuple(t.shape) != (1, 1, 48, 128) or tuple(t.padded_shape) != (1, 1, 64, 128) for t in (q, k, v)):
            raise ValueError("fused M1 requires q/k/v [1,1,48,128], padded [1,1,64,128]")
        scalar_shapes = {(1, 1, 48): (1, 32, 64), (1, 1, 1, 48): (1, 1, 32, 64)}
        if any(tuple(t.shape) not in scalar_shapes or tuple(t.padded_shape) != scalar_shapes[tuple(t.shape)] for t in (a, b)):
            raise ValueError("fused M1 requires a/b [1,1,48] or [1,1,1,48] with 32x64 tile padding")
        if any(tuple(t.shape) != (48, 1, 1) or tuple(t.padded_shape) != (48, 32, 32) for t in (db, ds)):
            raise ValueError("fused M1 requires 48 separate decay bias/scale tiles [48,1,1]")
        if math.prod(ones.shape) != 1024 or math.prod(ones.padded_shape) != 1024 or tuple(ones.shape)[-2:] != (32, 32):
            raise ValueError("fused M1 requires one shared 32x32 normalization tile")
        if (
            tuple(state.shape) != (1, 48, 128, 128)
            or tuple(state.padded_shape) != (1, 48, 128, 128)
            or state.dtype != ttnn.bfloat16
            or state.layout != ttnn.ROW_MAJOR_LAYOUT
            or state.memory_config() != self.state_memory_config
        ):
            raise ValueError("fused M1 requires live BF16 row-major [1,48,128,128], 110 L1 shards [56,128]")
        if (tuple(gate.shape) not in ((1, 1, self._gate_width), (1, 1, 1, self._gate_width))
                or tuple(gate.padded_shape) not in ((1, 32, self._gate_width), (1, 1, 32, self._gate_width))):
            raise ValueError("fused M1 requires a single tiled row ending in 6144 gate columns")
        if math.prod(gamma.shape) != 128 or gamma.shape[-1] != 128 or math.prod(gamma.padded_shape) != 4096 or tuple(gamma.padded_shape)[-2:] != (32, 128):
            raise ValueError("fused M1 requires a shared 128-element gamma padded to four tiles")
        for tensor in (q, k, v, a, b, db, ds, ones, gate, gamma):
            if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                raise ValueError("fused M1 operands must be interleaved BF16 tiles")
            if tensor.memory_config().buffer_type not in (ttnn.BufferType.L1, ttnn.BufferType.DRAM):
                raise ValueError("fused M1 operands must reside in L1 or DRAM")
        if any(t.memory_config().buffer_type != ttnn.BufferType.DRAM for t in (q, k, v, gamma)):
            raise ValueError("fused M1 q/k/v and gamma must reside in DRAM")
        inputs = (q, k, v, a, b, state, db, ds, ones)
        if any(t.device() != self.device for t in (*inputs, gate, gamma)):
            raise ValueError("fused M1 tensors must share the runner's device")
        self._check_inbox(self.inbox)
        if output is None:
            output = ttnn.empty(
                [1, 1, 6144],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        if (
            tuple(output.shape) != (1, 1, 6144)
            or tuple(output.padded_shape) != (1, 32, 6144)
            or output.dtype != ttnn.bfloat16
            or output.layout != ttnn.TILE_LAYOUT
            or output.memory_config() != ttnn.L1_MEMORY_CONFIG
            or output.device() != self.device
        ):
            raise ValueError("fused M1 output must be interleaved BF16 TILE L1 [1,1,6144]")
        tensors = [*inputs, gate, gamma, self.inbox, output]
        # Addresses alone are insufficient: accessors specialize on storage layout.
        signature = tuple(
            (int(t.buffer_address()), tuple(t.shape), tuple(t.padded_shape), str(t.memory_config()), t.dtype, t.layout)
            for t in tensors
        )
        if self.program is None or signature != self.signature:
            self.program = self._make_program(inputs, gate, gamma, output)
            self.signature = signature
        ttnn.generic_op(tensors, self.program)
        return output, state
