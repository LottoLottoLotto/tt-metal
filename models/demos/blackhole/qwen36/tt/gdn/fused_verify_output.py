# SPDX-License-Identifier: Apache-2.0
"""Fixed P150 wide MTP recurrence and exact output preparation in one program."""

import math
import struct
import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor
from models.demos.blackhole.qwen36.tt.gdn.native_output import _cb_descriptors

_ROOT = "models/demos/blackhole/qwen36/tt/gdn/"
_RECURRENT = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/"


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _cores(coords):
    return ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in coords})


def _accessors(tensors):
    args = []
    for tensor in tensors:
        accessor = ttnn.TensorAccessorArgs(tensor)
        assert not accessor.get_common_runtime_args(), "Verifier accessor requires compile-time metadata"
        args.extend(accessor.get_compile_time_args())
    return args


class FusedVerifyOutput:
    """Pipeline recurrence and output preparation without changing either compute mode.

    Disjoint core groups allow HiFi2/FP32 recurrence and HiFi4/BF16 RMSNorm.
    Each head transfers one rounded BF16 row through a 256-byte L1 mailbox.
    The request/ready handshake consumes both semaphores back to zero after
    each token, so repeated trace execution needs no host semaphore reset.
    """

    def __init__(self, device, inbox=None, fork_position=0, value_heads=48):
        self.device = device
        if value_heads not in (32, 48):
            raise ValueError("fused verify supports 32 or 48 value heads")
        self.value_heads = value_heads
        if fork_position and not 1 <= fork_position <= 7:
            raise ValueError("Tree fork must be in 1..7")
        self.fork_position = fork_position
        grid = device.compute_with_storage_grid_size()
        if grid.x < 11 or grid.y < 10:
            raise ValueError("fused verify requires the P150 11x10 worker grid")
        self.recurrence_cores = [ttnn.CoreCoord(h // 10, h % 10) for h in range(value_heads)]
        self.output_cores = [ttnn.CoreCoord(5 + h % 6, h // 6) for h in range(value_heads)]
        self.output_set = _cores(self.output_cores)
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.output_set, [1, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        if inbox is not None and (
            tuple(inbox.shape) != (1, 1, value_heads, 128)
            or inbox.dtype != ttnn.bfloat16
            or inbox.layout != ttnn.ROW_MAJOR_LAYOUT
            or inbox.memory_config() != mc
            or inbox.device() != device
        ):
            raise ValueError("shared verifier inbox must be BF16 row-major [1,1,HV,128] on the output core group")
        # Layers execute sequentially; one model-owned mailbox can serve all
        # runners while each program retains its own request/ready semaphores.
        self.inbox = inbox if inbox is not None else ttnn.from_torch(
            torch.zeros(1, 1, value_heads, 128, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=mc,
        )
        self.program = None
        self.signature = None

    def _make_program(self, inputs, gate, gamma, output, candidates):
        q, k, v, a, b, state, db, ds, ones = inputs
        num_tokens = int(q.shape[0])
        verify_defines = [("VERIFY_TWO", "1")] if num_tokens == 2 else [("VERIFY_TOKENS", str(num_tokens))]
        if self.fork_position:
            if num_tokens != 16 - self.fork_position:
                raise ValueError("Tree recurrence row count does not match fork")
            verify_defines.append(("TREE_FORK_POSITION", str(self.fork_position)))
        rec_set = _cores(self.recurrence_cores)
        all_set = _cores(self.recurrence_cores + self.output_cores)
        rec_phys = [self.device.worker_core_from_logical_core(c) for c in self.recurrence_cores]
        out_phys = [self.device.worker_core_from_logical_core(c) for c in self.output_cores]
        sqrt_k = struct.unpack("<f", struct.pack("<f", math.sqrt(128.0)))[0]
        ct = [4, 4, 1, _bits(1.0 / sqrt_k), _bits(1e-6)]
        reader = ttnn.KernelDescriptor(
            kernel_source=_RECURRENT + "dataflow/reader_recurrent_gated_delta_rule.cpp",
            core_ranges=rec_set,
            compile_time_args=ct + _accessors(inputs) + (
                _accessors((candidates[0],)) if self.fork_position else []
            ),
            defines=verify_defines,
            runtime_args=[
                (c, [h, *[int(t.buffer_address()) for t in inputs], self.value_heads,
                     *([int(candidates[self.fork_position - 1].buffer_address())] if self.fork_position else [])])
                for h, c in enumerate(self.recurrence_cores)
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=_ROOT + "fused_verify_writer.cpp",
            core_ranges=rec_set,
            compile_time_args=[4, 4, num_tokens] + _accessors((candidates[0],)),
            defines=verify_defines,
            runtime_args=[
                (
                    c,
                    [
                        h,
                        *[int(t.buffer_address()) for t in candidates],
                        out_phys[h].x,
                        out_phys[h].y,
                        int(self.inbox.buffer_address()),
                    ],
                )
                for h, c in enumerate(self.recurrence_cores)
            ],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=_RECURRENT + "compute/recurrent_gated_delta_rule.cpp",
            core_ranges=rec_set,
            compile_time_args=ct,
            defines=verify_defines,
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
        recurrence_cbs.append(cb((8,), 4))  # row-major state staging: one tile row of sticks
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
        token_args = [
            ("num_tokens", num_tokens),
            ("value_heads", self.value_heads),
            ("head_major", 0),
            ("gate_tile_offset", 0),
            ("gate_row_tiles", int(gate.shape[-1]) // 32),
        ]
        norm = UnifiedKernelDescriptor(
            kernel_source=_ROOT + "fused_verify_output.cpp",
            core_ranges=self.output_set,
            ncrisc_named_compile_time_args=token_args,
            brisc_named_compile_time_args=[
                ("gate_is_dram", int(gate.memory_config().buffer_type == ttnn.BufferType.DRAM)),
                ("gamma_is_dram", int(gamma.memory_config().buffer_type == ttnn.BufferType.DRAM)),
                ("output_is_dram", 0),
                *token_args,
            ],
            trisc_compile_time_args=[4, 4, 1, 0, 0, 0, 0, 128, 32],
            # The mailbox publishes one token per handshake, not all packed
            # rows at once. Run native output compute once per token; our
            # dataflow kernel gathers/writes that token's packed row itself.
            trisc_named_compile_time_args=[*cb_args, *token_args, ("packed_rows", 0)],
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
        norm_cbs = _cb_descriptors(self.value_heads)
        for descriptor in norm_cbs:
            descriptor.core_ranges = self.output_set
        return ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute, *norm.get_kernel_descriptors().kernels],
            cbs=recurrence_cbs + norm_cbs,
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=all_set, initial_value=0)
                        for i in ((0, 1, 2) if self.fork_position else (0, 1))],
        )

    def __call__(self, q, k, v, a, b, db, ds, ones, state, gate, gamma):
        num_tokens = int(q.shape[0])
        heads = self.value_heads
        padded_heads = ((heads + 31) // 32) * 32
        if not 2 <= num_tokens <= 32 or tuple(q.shape) != (num_tokens, 1, heads, 128) or tuple(k.shape) != tuple(q.shape) or tuple(v.shape) != tuple(q.shape):
            raise ValueError("fused verify requires q/k/v [T,1,HV,128], 2 <= T <= 32")
        if any(tuple(t.padded_shape) != (num_tokens, 1, padded_heads, 128) for t in (q, k, v)):
            raise ValueError("fused verify requires q/k/v with tile-padded heads")
        if tuple(a.shape) != (1, 1, num_tokens, heads) or tuple(b.shape) != tuple(a.shape):
            raise ValueError("fused verify requires a/b [1,1,T,HV]")
        if any(tuple(t.padded_shape) != (1, 1, 32, padded_heads) for t in (a, b)):
            raise ValueError("fused verify requires tile-padded a/b")
        if tuple(db.shape) != (heads, 1, 1) or tuple(ds.shape) != tuple(db.shape):
            raise ValueError("fused verify requires one [1,1] decay tile per head")
        if any(tuple(t.padded_shape) != (heads, 32, 32) for t in (db, ds)):
            raise ValueError("fused verify requires separate decay tiles per head")
        if math.prod(ones.shape) != 1024 or tuple(ones.shape)[-2:] != (32, 32):
            raise ValueError("fused verify requires one shared 32x32 normalization tile")
        if tuple(state.shape) not in ((heads, 128, 128), (1, heads, 128, 128)):
            raise ValueError("fused verify requires live recurrent state [1,HV,128,128]")
        state_mc = state.memory_config()
        if state.dtype != ttnn.bfloat16 or state.layout != ttnn.ROW_MAJOR_LAYOUT or state_mc.buffer_type != ttnn.BufferType.L1 or state_mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            raise ValueError("fused verify requires BF16 row-major height-sharded live L1 state")
        state_spec = state_mc.shard_spec
        if state_spec.shape[1] != 128 or state_spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
            raise ValueError("fused verify requires row-major 128-wide recurrent shards")
        if tuple(gate.shape) != (1, num_tokens, heads * 128) or tuple(gate.padded_shape) != (1, 32, heads * 128):
            raise ValueError("fused verify requires packed gate rows [1,T,HV*128]")
        if math.prod(gamma.shape) != 128 or gamma.shape[-1] != 128:
            raise ValueError("fused verify requires a shared 128-element tiled gamma vector")
        for tensor in (q, k, v, a, b, db, ds, ones, gate, gamma):
            if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT or tensor.is_sharded():
                raise ValueError("fused verify operands must be interleaved BF16 tiles")
        if any(t.device() != self.device for t in (q, k, v, a, b, db, ds, ones, state, gate, gamma)):
            raise ValueError("fused verify tensors must share the runner's device")
        inputs = (q, k, v, a, b, state, db, ds, ones)
        output = ttnn.empty(
            [1, num_tokens, heads * 128],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        candidates = tuple(
            ttnn.empty(
                [1, heads, 128, 128],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(num_tokens)
        )
        tensors = [*inputs, gate, gamma, self.inbox, output, *candidates]
        signature = tuple((int(t.buffer_address()), tuple(t.shape), tuple(t.padded_shape), str(t.memory_config()), t.dtype, t.layout) for t in tensors)
        if self.program is None or signature != self.signature:
            self.program = self._make_program(inputs, gate, gamma, output, candidates)
            self.signature = signature
        ttnn.generic_op(tensors, self.program)
        return output, *candidates
