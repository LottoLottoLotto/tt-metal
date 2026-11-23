# SPDX-License-Identifier: Apache-2.0
"""Fixed P150 MTP1 recurrence and output preparation in one device program."""

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

    def __init__(self, device):
        self.device = device
        self.recurrence_cores = [ttnn.CoreCoord(h // 10, h % 10) for h in range(32)]
        self.output_cores = [ttnn.CoreCoord(4 + h % 4, h // 4) for h in range(32)]
        self.output_set = _cores(self.output_cores)
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.output_set, [1, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.inbox = ttnn.from_torch(
            torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=mc,
        )
        self.program = None
        self.signature = None

    def _make_program(self, inputs, gate, gamma, output, s1, s2):
        q, k, v, a, b, state, db, ds, ones = inputs
        rec_set = _cores(self.recurrence_cores)
        all_set = _cores(self.recurrence_cores + self.output_cores)
        rec_phys = [self.device.worker_core_from_logical_core(c) for c in self.recurrence_cores]
        out_phys = [self.device.worker_core_from_logical_core(c) for c in self.output_cores]
        sqrt_k = struct.unpack("<f", struct.pack("<f", math.sqrt(128.0)))[0]
        ct = [4, 4, 1, _bits(1.0 / sqrt_k), _bits(1e-6)]
        reader = ttnn.KernelDescriptor(
            kernel_source=_RECURRENT + "dataflow/reader_recurrent_gated_delta_rule.cpp",
            core_ranges=rec_set,
            compile_time_args=ct + _accessors(inputs),
            defines=[("VERIFY_TWO", "1")],
            runtime_args=[
                (c, [h, *[int(t.buffer_address()) for t in inputs], 32]) for h, c in enumerate(self.recurrence_cores)
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=_ROOT + "fused_verify_writer.cpp",
            core_ranges=rec_set,
            compile_time_args=[4, 4] + _accessors((s1, s2)),
            runtime_args=[
                (
                    c,
                    [
                        h,
                        int(s1.buffer_address()),
                        int(s2.buffer_address()),
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
            defines=[("VERIFY_TWO", "1")],
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
        token_args = [("num_tokens", 2), ("packed_rows", 1)]
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
        return ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute, *norm.get_kernel_descriptors().kernels],
            cbs=recurrence_cbs + norm_cbs,
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=all_set, initial_value=0) for i in (0, 1)],
        )

    def __call__(self, q, k, v, a, b, db, ds, ones, state, gate, gamma):
        assert tuple(q.shape) == tuple(k.shape) == tuple(v.shape) == (2, 1, 32, 128)
        assert tuple(a.shape) == tuple(b.shape) == (1, 1, 2, 32)
        assert tuple(state.shape) == (32, 128, 128) and state.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tuple(gate.shape) == (1, 2, 4096)
        inputs = (q, k, v, a, b, state, db, ds, ones)
        output = ttnn.empty(
            [1, 2, 4096],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        s1, s2 = [
            ttnn.empty(
                [32, 128, 128],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]
        tensors = [*inputs, gate, gamma, self.inbox, output, s1, s2]
        signature = tuple(int(t.buffer_address()) for t in tensors)
        if self.program is None or signature != self.signature:
            self.program = self._make_program(inputs, gate, gamma, output, s1, s2)
            self.signature = signature
        ttnn.generic_op(tensors, self.program)
        return output, s1, s2
