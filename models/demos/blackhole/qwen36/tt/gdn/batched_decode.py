# SPDX-License-Identifier: Apache-2.0
"""Single-device serving decode: M-row projection, independent DRAM state per slot.

State is externally owned, compact BF16 ROW_MAJOR DRAM in the canonical (key-row major)
layout shared with legacy/prefill. A bucket covers the first B slots; position -1 masks
both history and recurrence writes, without any host readback. Only transient operands
are shared across sequential layers (the L1 q|k|v rows also across buckets).

Slots stay tile rows end to end: the frontend activates each 32-channel QKV column for
every slot at once, the recurrence stripes (slot, head) jobs over the whole grid with
double-buffered operands and state moved as whole DRAM-bank runs, and its output rows
are already o_proj input rows.
"""
import math
import struct

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.gdn.native_output import NativeOutput, native_output

_ROOT = "models/demos/blackhole/qwen36/tt/gdn/"
_BUCKETS = (1, 2, 4, 8, 16)
_QKV_TILES = 8192 // 32


def validate_batched_state(recurrent, convolution):
    if recurrent is None or convolution is None:
        raise ValueError("Batched GDN requires externally bound recurrent and convolution state")
    capacity = int(recurrent.shape[0])
    if not 1 <= capacity <= 16 or tuple(recurrent.shape) != (capacity, 32, 128, 128):
        raise ValueError("Batched recurrent state must be [capacity,32,128,128], capacity <= 16")
    if tuple(convolution.shape) != (capacity, 3, 8192):
        raise ValueError("Batched convolution state must be [capacity,3,8192]")
    for state in (recurrent, convolution):
        if (state.dtype != ttnn.bfloat16 or state.layout != ttnn.ROW_MAJOR_LAYOUT
                or state.is_sharded() or state.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                or tuple(state.padded_shape) != tuple(state.shape)):
            raise ValueError("Batched GDN state must be compact interleaved BF16 ROW_MAJOR DRAM")
    return capacity


def _grid(device, count):
    """First `count` cores row-major, as at most two rectangles (binaries multicast)."""
    grid = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
    rows, rest = divmod(count, grid.x)
    ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, rows - 1))] if rows else []
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(rest - 1, rows)))
    return cores, ttnn.CoreRangeSet(set(ranges))


def _workers(device, jobs):
    grid = device.compute_with_storage_grid_size()
    count = min(jobs, grid.x * grid.y)
    cores, core_set = _grid(device, count)
    args = [(c, [i, count, (jobs - i + count - 1) // count]) for i, c in enumerate(cores)]
    return core_set, args


def _is_dram(tensor):
    return int(tensor.memory_config().buffer_type == ttnn.BufferType.DRAM)


def _cbs(cores, sizes):
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    return [ttnn.CBDescriptor(total_size=count * 2048, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, 2048, tile)])
            for index, count in sizes]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _positions_metadata(positions):
    return [
        ("positions_is_dram", int(positions.memory_config().buffer_type == ttnn.BufferType.DRAM)),
        ("positions_page_bytes", int(positions.padded_shape[-1]) * 4),
    ]


def _zeros(device, shape, memory=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory)


class BatchedWorkspace:
    def __init__(self, device, batch, qkv):
        # Activated q|k|v columns and recurrence rows; slot b is tile row b throughout.
        # qkv is one L1 tile row shared by every bucket: the recurrence gathers single
        # 32-byte rows from it, which must not queue behind state traffic in DRAM.
        self.qkv = qkv
        self.recurrent = _zeros(device, (1, batch, 4096))
        self.zero_bias = _zeros(device, (1, 1, 8192))
        self.output = _zeros(device, (1, batch, 4096), ttnn.L1_MEMORY_CONFIG)


class BatchedDecode:
    def __init__(self, device, batch, workspace):
        self.device, self.batch, self.workspace = device, batch, workspace
        self.frontend_program = self.recurrent_program = None
        self.frontend_signature = self.recurrent_signature = None
        self.output = NativeOutput(value=workspace.output)

    def _frontend(self, mega, taps, bias, history, positions):
        ws = self.workspace
        if bias is None:
            bias = ws.zero_bias
        tensors = [mega, *taps, bias, history, positions, ws.qkv]
        signature = tuple(int(t.buffer_address()) for t in tensors) + tuple(_positions_metadata(positions))
        if self.frontend_program is None or signature != self.frontend_signature:
            # Contiguous column runs: every worker reads each slot's history run in one piece.
            grid = self.device.compute_with_storage_grid_size()
            per = -(-_QKV_TILES // (grid.x * grid.y))
            cores, core_set = _grid(self.device, -(-_QKV_TILES // per))
            runs = [(c, [i * per, min(per, _QKV_TILES - i * per)]) for i, c in enumerate(cores)]
            activations = [int(t.buffer_address()) for t in (mega, history, positions)]
            params = [int(t.buffer_address()) for t in (ws.qkv, *taps, bias)]
            kernels = [
                ttnn.KernelDescriptor(kernel_source=_ROOT + "batched_frontend_reader.cpp",
                    core_ranges=core_set,
                    compile_time_args=[self.batch, _is_dram(mega),
                                       *[value for _, value in _positions_metadata(positions)]],
                    runtime_args=[(c, [*run, *activations]) for c, run in runs],
                    config=ttnn.ReaderConfigDescriptor()),
                ttnn.KernelDescriptor(kernel_source=_ROOT + "batched_frontend_writer.cpp",
                    core_ranges=core_set, compile_time_args=[_is_dram(ws.qkv), _is_dram(taps[0]), _is_dram(bias)],
                    runtime_args=[(c, [*run, *params]) for c, run in runs],
                    config=ttnn.WriterConfigDescriptor()),
                ttnn.KernelDescriptor(kernel_source=_ROOT + "batched_frontend_compute.cpp",
                    core_ranges=core_set, compile_time_args=[self.batch],
                    runtime_args=[(c, [run[1]]) for c, run in runs],
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False, fp32_dest_acc_en=False, dst_full_sync_en=False)),
            ]
            # Row-major history, mega, raw/broadcast params, tilized history, scratch, temp,
            # positions, activated output.
            sizes = ((0, 3 * per), (1, per), (2, 5 * per), (3, 5 * per), (4, 3 * per),
                     (5, 1), (6, 1), (7, 1), (16, per))
            self.frontend_program = ttnn.ProgramDescriptor(
                kernels=kernels, cbs=_cbs(core_set, sizes), semaphores=[])
            self.frontend_signature = signature
        ttnn.generic_op(tensors, self.frontend_program)

    def _recurrence(self, mega, state, bias, scale, positions):
        ws = self.workspace
        tensors = [ws.qkv, mega, state, bias, scale, positions, ws.recurrent]
        signature = tuple(int(t.buffer_address()) for t in tensors) + tuple(_positions_metadata(positions))
        if self.recurrent_program is None or signature != self.recurrent_signature:
            cores, jobs = _workers(self.device, self.batch * 32)
            readers = [(c, [*args, int(state.buffer_address())]) for c, args in jobs]
            writer_inputs = [int(t.buffer_address()) for t in (ws.qkv, mega, bias, scale, ws.recurrent, state,
                                                                positions)]
            writers = [(c, [*args, *writer_inputs]) for c, args in jobs]
            sqrt_k = struct.unpack("<f", struct.pack("<f", math.sqrt(128.0)))[0]
            kernels = [
                ttnn.KernelDescriptor(kernel_source=_ROOT + "batched_recurrent_reader.cpp",
                    core_ranges=cores, compile_time_args=[], runtime_args=readers,
                    config=ttnn.ReaderConfigDescriptor()),
                ttnn.KernelDescriptor(kernel_source=_ROOT + "batched_recurrent_writer.cpp",
                    core_ranges=cores,
                    compile_time_args=[_is_dram(mega), *[value for _, value in _positions_metadata(positions)]],
                    runtime_args=writers, config=ttnn.WriterConfigDescriptor()),
                ttnn.KernelDescriptor(kernel_source=_ROOT + "batched_recurrent_compute.cpp",
                    core_ranges=cores, compile_time_args=[4, 4, 1, _bits(1.0 / sqrt_k), _bits(1e-6)],
                    runtime_args=[(c, [args[2]]) for c, args in jobs],
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2,
                                                       fp32_dest_acc_en=True, math_approx_mode=False)),
            ]
            # Fixed workspace per worker; no persistent state lives in these CBs. Operands
            # (0-4, 6-7) and the output row (16) are double-buffered; state rows in (8)
            # hold two heads and out (17) one, so DRAM traffic overlaps neighbouring jobs.
            sizes = ((0, 8), (1, 8), (2, 8), (3, 2), (4, 2), (5, 16), (6, 2), (7, 2), (8, 32), (9, 1),
                     (10, 3), (16, 8), (17, 16), (18, 16), (19, 4), (20, 4), (21, 4), (22, 4), (23, 16),
                     (24, 16), (25, 1), (26, 1), (27, 4), (28, 1), (29, 1), (30, 4), (31, 4))
            self.recurrent_program = ttnn.ProgramDescriptor(kernels=kernels, cbs=_cbs(cores, sizes), semaphores=[])
            self.recurrent_signature = signature
        ttnn.generic_op(tensors, self.recurrent_program)

    def __call__(self, gdn, x, positions):
        w = gdn.weights
        if gdn.mega_decode_runner is not None:
            # Bank-packed decode mega weight, dual-NoC packed reader: bitwise the stock matmul below.
            mega = gdn.mega_decode_runner(x)
        else:
            mega = ttnn.linear(x, w.mega_fused_weight_decode, memory_config=ttnn.L1_MEMORY_CONFIG,
                               compute_kernel_config=gdn.compute_kernel_config_decode,
                               program_config=gdn.mega_decode_program_config)
        self._frontend(mega, w.native_frontend_weight_taps, w.native_frontend_bias,
                       gdn.fused_conv_state, positions)
        self._recurrence(mega, gdn.recurrent_state, w.native_decay_bias, w.native_decay_scale, positions)
        gate = ttnn.reshape(mega, (1, self.batch, 12352))
        value = native_output(self.workspace.recurrent, gate, w.o_norm_weight, self.output, 8256)
        projected = ttnn.linear(value, w.o_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                                compute_kernel_config=gdn.compute_kernel_config_decode,
                                program_config=gdn.out_decode_program_config)
        ttnn.deallocate(mega)
        return ttnn.reshape(projected, (1, 1, self.batch, 2048))


def forward_batched(gdn, x, position_tensor):
    """Decode [1,1,B,2048] BF16 TILE; return same shape, B in {1,2,4,8,16}.

    Positions are interleaved device INT32 ROW_MAJOR [B], -1 for inactive slots.
    They identify activity, not state addressing: row i always uses bound state slot i.
    Caller binds [capacity,32,128,128] / [capacity,3,8192] compact BF16 DRAM state,
    with capacity >= B. Neither persistent allocation nor host synchronization occurs.
    """
    shape = tuple(int(v) for v in x.shape)
    batch = shape[-2]
    if shape != (1, 1, batch, 2048) or batch not in _BUCKETS:
        raise ValueError("Batched GDN input must be [1,1,B,2048], B in {1,2,4,8,16}")
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT or x.is_sharded():
        raise ValueError("Batched GDN requires interleaved BF16 TILE input")
    if gdn.device.get_num_devices() != 1 or (gdn.num_heads, gdn.num_v_heads,
            gdn.head_k_dim, gdn.head_v_dim, gdn.conv_kernel_size) != (16, 32, 128, 128, 4):
        raise ValueError("Batched GDN requires single-device Qwen3.6-35B-A3B geometry")
    dram = gdn.device.dram_grid_size()
    if dram.x * dram.y != 8:
        raise ValueError("Batched GDN state streaming assumes 8 DRAM banks (P150)")
    capacity = validate_batched_state(gdn.recurrent_state, gdn.fused_conv_state)
    if batch > capacity:
        raise ValueError("GDN bucket exceeds the externally reserved state capacity")
    if (tuple(position_tensor.shape) != (batch,) or position_tensor.dtype != ttnn.int32
            or position_tensor.layout != ttnn.ROW_MAJOR_LAYOUT or position_tensor.is_sharded()):
        raise ValueError("Batched GDN positions must be interleaved device INT32 ROW_MAJOR [B]")
    w = gdn.weights
    if (w.mega_fused_weight_decode is None or w.native_frontend_weight_taps is None
            or w.native_decay_bias is None or w.native_decay_scale is None
            or gdn.cfg.output_gate_type not in ("swish", "silu") or gdn.norm_eps != 1e-6):
        raise ValueError("Batched GDN requires native frontend/output weights and RMS epsilon 1e-6")
    if not hasattr(gdn, "_batched_decode"):
        gdn._batched_decode = {}
    if batch not in gdn._batched_decode:
        key = (id(gdn.device), batch)
        shared = type(gdn)._shared_batched_workspace
        if key not in shared:
            rows_key = (id(gdn.device), "qkv")
            if rows_key not in shared:
                shared[rows_key] = _zeros(gdn.device, (1, 1, 32, 8192), ttnn.L1_MEMORY_CONFIG)
            shared[key] = BatchedWorkspace(gdn.device, batch, shared[rows_key])
        gdn._batched_decode[batch] = BatchedDecode(gdn.device, batch, shared[key])
    return gdn._batched_decode[batch](gdn, x, position_tensor)
