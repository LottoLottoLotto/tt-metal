# SPDX-License-Identifier: Apache-2.0
"""Single-P150 fused GDN core for the 35B geometry (H=16 key heads, HV=32 value heads, 128x128 heads): the native
frontend (four-tap FIR + SiLU + GQA scatter), the gated delta-rule recurrence and the native output preparation
(RMSNorm * clamp(SiLU(gate)), clamped) in ONE program over 96 cores, for decode (T=1, live state updated in place)
and MTP-1 verification (T=2, prefix candidates, live state untouched).

Arithmetic is that of the three unfused programs, op for op:
  * recurrence: the image's own compute kernel (recurrent_gated_delta_rule.cpp, HiFi2/FP32, VERIFY_TWO for T=2),
    run per (value head h, V-half j) with Vt = 2 local V tiles. Every recurrence op is per V-column tile (state
    decay, k^T S and q^T S column accumulations over the same K tiles, delta, outer product, add, tilize/untilize),
    and the per-head scalars (q/k L2 norms, sigmoid(b), decay) are recomputed identically on both halves, so each
    output/state element is computed by the identical tile op sequence;
  * frontend: native_frontend_kernel.cpp's elementwise tile ops, two source heads per output worker in one pass
    (row placement does not change any element's math; silu/mul_binary arithmetic and inits ignore APPROX);
  * output: native_output_kernel.cpp (stock layernorm compute with RMSNORM, SiLU, clamp, SFPU multiply).
Data moves core to core through a shared L1 inbox with two semaphores (each consumed back to zero every run):
frontend -> recurrence (q/k/v rows; a/b are read in place from the mega row) and recurrence -> output (o rows).
"""
from pathlib import Path

import numpy as np
import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

_ROOT = "models/demos/blackhole/qwen36/tt/gdn/core35/"
_RECURRENT = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/recurrent_gated_delta_rule.cpp"
_KT, _VT_LOCAL, _VT = 4, 2, 4
_KEY_HEADS, _VALUE_HEADS = 16, 32
_QKV_WIDTH = (2 * _KEY_HEADS + _VALUE_HEADS) * 128
_A_TILE = _QKV_WIDTH // 32  # mega tiles: [qkv | a (1 tile) | b (1 tile) | gate]
_GATE_TILE = _A_TILE + 2
_MEGA_TILES = _GATE_TILE + _VALUE_HEADS * 4
_REC_INBOX_TOKEN_BYTES = (2 * _KT + _VT_LOCAL) * 64
_OUT_INBOX_TOKEN_BYTES = 2 * _VT_LOCAL * 64
_INBOX_ROWS = 8  # per-core shard of the shared inbox: 8 x 128 BF16 = 2 KiB (>= 2 tokens x 640 B)


def _f32_bits(value):
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _accessors(tensors):
    args = []
    for tensor in tensors:
        accessor = ttnn.TensorAccessorArgs(tensor)
        if accessor.get_common_runtime_args():
            raise ValueError("fused GDN core accessors require compile-time metadata")
        args.extend(accessor.get_compile_time_args())
    return args


def _placement():
    """Rectangular core groups (dispatch multicasts binaries per CoreRange: scattered single cores cost ~80 us per
    launch). Recurrence R(h, j): the 8x8 block x 0..7, y 0..7 (r = 2h + j row-major); output O(h): x 0..7, y 8..9
    for h < 16 and x 8..9, y 0..7 for h >= 16."""
    rec, out = {}, {}
    for h in range(_VALUE_HEADS):
        for j in range(2):
            r = 2 * h + j
            rec[(h, j)] = ttnn.CoreCoord(r % 8, r // 8)
        out[h] = ttnn.CoreCoord(h % 8, 8 + h // 8) if h < 16 else ttnn.CoreCoord(8 + (h - 16) % 2, (h - 16) // 2)
    return rec, out


_REC_RANGES = [((0, 0), (7, 7))]
_OUT_RANGES = [((0, 8), (7, 9)), ((8, 0), (9, 7))]


def _ranges(rects):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in rects})


def _sources(h):
    """The two frontend source heads computed on O(h): its GQA key head's q (even h) or k (odd h), and v of h."""
    kh = h // 2
    return (kh if h % 2 == 0 else _KEY_HEADS + kh), 2 * _KEY_HEADS + h


def supports(device, cfg_heads, head_dims, mega_dims):
    """35B single-P150 geometry only."""
    grid = device.compute_with_storage_grid_size()
    return (
        device.get_num_devices() == 1 and (grid.x, grid.y) == (11, 10)
        and cfg_heads == (_KEY_HEADS, _VALUE_HEADS) and head_dims == (128, 128)
        and mega_dims == (_QKV_WIDTH, 32, 32, _VALUE_HEADS * 128)
    )


class FusedGdnCore:
    """One layer's fused core for one token count (1: decode, 2: MTP-1 verify)."""

    _shared_inbox = {}
    _shared_output = {}
    _shared_zero_bias = {}

    def __init__(self, device, num_tokens):
        if num_tokens not in (1, 2):
            raise ValueError("fused GDN core supports T=1 decode and T=2 verification")
        self.device = device
        self.num_tokens = num_tokens
        self.rec, self.out = _placement()
        self.rec_list = [self.rec[(h, j)] for h in range(_VALUE_HEADS) for j in range(2)]
        self.out_list = [self.out[h] for h in range(_VALUE_HEADS)]
        all_cores = self.rec_list + self.out_list
        key = id(device)
        inbox = type(self)._shared_inbox.get(key)
        if inbox is None:
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                ttnn.ShardSpec(_ranges(_REC_RANGES + _OUT_RANGES), [_INBOX_ROWS, 128], ttnn.ShardOrientation.ROW_MAJOR),
            )
            inbox = ttnn.from_torch(
                torch.zeros(1, 1, len(all_cores) * _INBOX_ROWS, 128, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc,
            )
            type(self)._shared_inbox[key] = inbox
        self.inbox = inbox
        # Layers run sequentially and o_proj consumes the prepared rows right away: one output per token count
        # (zero padding rows, as the native output buffers).
        output = type(self)._shared_output.get((key, num_tokens))
        if output is None:
            output = ttnn.from_torch(
                torch.zeros((1, num_tokens, _VALUE_HEADS * 128), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            type(self)._shared_output[(key, num_tokens)] = output
        self.output = output
        zero_bias = type(self)._shared_zero_bias.get(key)
        if zero_bias is None:
            zero_bias = ttnn.from_torch(torch.zeros((1, 1, _QKV_WIDTH), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            type(self)._shared_zero_bias[key] = zero_bias
        self.zero_bias = zero_bias
        # Verification never mutates live history: per-token conv prefix candidates (as the native frontend's).
        self.conv_candidates = tuple(
            ttnn.from_torch(torch.zeros((1, 3, _QKV_WIDTH), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(num_tokens)
        ) if num_tokens > 1 else ()
        self.program = None
        self.signature = None

    # ------------------------------------------------------------------ program
    def _make_program(self, mega, weights, conv_state, state, db, ds, ones, gamma, candidates):
        dev = self.device
        T = self.num_tokens
        phys = lambda c: dev.worker_core_from_logical_core(c)
        rec_set, out_set = _ranges(_REC_RANGES), _ranges(_OUT_RANGES)
        all_set = _ranges(_REC_RANGES + _OUT_RANGES)
        inbox = int(self.inbox.buffer_address())

        # ---- recurrence workers
        q_scale = np.float32(1.0) / np.sqrt(np.float32(128.0))
        ct = [_KT, _VT_LOCAL, 1, _f32_bits(q_scale), _f32_bits(1.0e-6)]
        verify = [("VERIFY_TWO", "1")] if T == 2 else []
        reader = ttnn.KernelDescriptor(
            kernel_source=_ROOT + "core_reader.cpp", core_ranges=rec_set,
            compile_time_args=[_KT, _VT_LOCAL, T, _A_TILE, _REC_INBOX_TOKEN_BYTES, 3]
            + _accessors((mega, state, db, ds, ones)),
            runtime_args=[
                (self.rec[(h, j)], [h, j, int(mega.buffer_address()), int(state.buffer_address()),
                                    int(db.buffer_address()), int(ds.buffer_address()), int(ones.buffer_address()), inbox])
                for h in range(_VALUE_HEADS) for j in range(2)
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        dst = (state,) if T == 1 else (candidates[0],)
        dst_addrs = [int(state.buffer_address())] if T == 1 else [int(c.buffer_address()) for c in candidates]
        writer = ttnn.KernelDescriptor(
            kernel_source=_ROOT + "core_writer.cpp", core_ranges=rec_set,
            compile_time_args=[_KT, _VT_LOCAL, T, _OUT_INBOX_TOKEN_BYTES, _VT] + _accessors(dst),
            runtime_args=[
                (self.rec[(h, j)], [h, j, phys(self.out[h]).x, phys(self.out[h]).y, inbox, *dst_addrs])
                for h in range(_VALUE_HEADS) for j in range(2)
            ],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=_RECURRENT, core_ranges=rec_set, compile_time_args=ct, defines=verify,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False),
        )
        tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))

        def rcb(indices, tiles):
            return ttnn.CBDescriptor(
                total_size=tiles * 2048, core_ranges=rec_set,
                format_descriptors=[ttnn.CBFormatDescriptor(i, ttnn.bfloat16, 2048, tile) for i in indices],
            )

        kt, vt = _KT, _VT_LOCAL
        kv = kt * vt
        stream = 2 if T == 2 else 1  # the verify factory double-buffers the per-token operand CBs
        rec_cbs = [rcb((0,), stream * kt), rcb((1,), stream * kt), rcb((2,), stream * vt), rcb((3,), stream),
                   rcb((4,), stream), rcb((5, 17), kv),
                   # Row-major state staging sized for every tile row: the reader never waits on compute.
                   rcb((8,), kv)]
        rec_cbs += [rcb((i,), n) for i, n in (
            (16, vt), (18, kv), (19, vt), (20, vt), (21, vt), (22, kt), (23, kv), (24, kv), (25, 1),
            (6, 1), (7, 1), (26, 1), (27, kt), (28, 1), (29, 1), (30, kt), (31, kt))]

        # ---- output / frontend workers

        def xy(core):
            p = phys(core)
            return int(p.x) | (int(p.y) << 16)

        def dests(worker):
            if worker < 2 * _KEY_HEADS:
                kh = worker % _KEY_HEADS
                return [xy(self.rec[(2 * kh + g, j)]) for g in range(2) for j in range(2)]
            h = worker - 2 * _KEY_HEADS
            return [xy(self.rec[(h, 0)]), xy(self.rec[(h, 1)]), 0, 0]

        brisc_args = []
        for h in range(_VALUE_HEADS):
            w0, w1 = _sources(h)
            brisc_args.append((self.out[h], [h, w0, w1, *dests(w0), *dests(w1), h]))
        named = [
            ("num_tokens", T), ("qkv_width", _QKV_WIDTH), ("key_heads", _KEY_HEADS), ("gqa", 2),
            ("value_heads", _VALUE_HEADS), ("rec_inbox_token_bytes", _REC_INBOX_TOKEN_BYTES),
            ("out_inbox_token_bytes", _OUT_INBOX_TOKEN_BYTES),
        ]
        conv_logical = ttnn.corerange_to_cores(conv_state.memory_config().shard_spec.grid, row_wise=True)
        if len(conv_logical) != 1:
            raise ValueError("fused GDN core requires the layer's convolution history on one core")
        conv_core = phys(conv_logical[0])
        is_dram = lambda t: int(t.memory_config().buffer_type == ttnn.BufferType.DRAM)
        cb_args = [
            ("cb_scaler", 3), ("cb_eps", 4), ("cb_in", 0), ("cb_inb", 15), ("cb_out", 8), ("cb_gamma", 2),
            ("cb_beta", 15), ("cb_xmm", 0), ("cb_ex", 5), ("cb_ex2", 6), ("cb_xmm2", 7), ("cb_ex2pe", 9),
            ("cb_fusion", 10), ("cb_in_rm", 15),
        ]
        out_kernel = UnifiedKernelDescriptor(
            kernel_source=_ROOT + "core_output_kernel.cpp",
            core_ranges=out_set,
            ncrisc_named_compile_time_args=named + [("gamma_is_dram", is_dram(gamma)), ("gate_tile_offset", _GATE_TILE)],
            brisc_named_compile_time_args=named + [("output_is_dram", 0)],
            trisc_compile_time_args=[4, 4, 1, 0, 0, 0, 0, 128, 32],
            trisc_named_compile_time_args=[*cb_args, *named],
            brisc_common_runtime_args=[
                int(mega.buffer_address()), int(weights.buffer_address()), 0, 0, 0, 0,
                int(conv_state.buffer_address()), int(conv_core.x), int(conv_core.y),
                int(self.output.buffer_address()), inbox,
                *([int(t.buffer_address()) for t in self.conv_candidates] or [0, 0]),
            ],
            ncrisc_common_runtime_args=[int(mega.buffer_address()), int(gamma.buffer_address())],
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=False,
                dst_full_sync_en=False),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=brisc_args,
                ncrisc_args=[(self.out[h], [h]) for h in range(_VALUE_HEADS)],
                trisc_args=[(self.out[h], [1]) for h in range(_VALUE_HEADS)],
            ),
            defines=[("RMSNORM", "1")],
        )
        def ocb(index, tiles, aliases=()):
            return ttnn.CBDescriptor(total_size=tiles * 2048, core_ranges=out_set, format_descriptors=[
                ttnn.CBFormatDescriptor(i, ttnn.bfloat16, 2048, tile) for i in (index, *aliases)])

        # native_output.py's CBs, except that the clipped gate (12) no longer aliases x (0): it is computed while
        # the recurrence runs.
        out_cbs = [ocb(0, 4), ocb(12, 4), ocb(1, 4), ocb(2, 4, (16,)), ocb(7, 4, (10,)), ocb(8, 4)]
        out_cbs += [ocb(i, 1) for i in (3, 4, 5, 6, 9, 11, 15)]
        staging = 2 * (3 + T) * 256  # two sources' FIR rows: three history rows + T token rows each
        front_cbs = [ocb(17, 4), ocb(18, 5), ocb(19, 1), ocb(21, 1), ocb(22, 1), ttnn.CBDescriptor(
            total_size=staging, core_ranges=out_set,
            format_descriptors=[ttnn.CBFormatDescriptor(20, ttnn.bfloat16, 64, ttnn.TileDescriptor(ttnn.Tile((1, 32))))])]
        return ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute, *out_kernel.get_kernel_descriptors().kernels],
            cbs=rec_cbs + out_cbs + front_cbs,
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=all_set, initial_value=0) for i in (0, 1)],
        )

    def _arranged_weights(self, taps, bias):
        """The FIR weight tiles every output core's frontend pass consumes, built once from the layer's taps/bias:
        tile 5c + p (p = tap 0..3, 4 = bias) of core c has, on row 8 * slot + 2 * dim (slot = source * T + token),
        that source's row-0 weights for channels 32 dim .. 32 dim + 31 and zeros elsewhere -- the tiles
        native_frontend_kernel.cpp assembles per launch."""
        key = tuple(int(t.buffer_address()) for t in (*taps, bias))
        if getattr(self, "_weights_key", None) == key:
            return self._weights
        T = self.num_tokens
        rows = [ttnn.to_torch(t).reshape(-1)[:_QKV_WIDTH].to(torch.bfloat16) for t in (*taps, bias)]
        arranged = torch.zeros(_VALUE_HEADS * 5, 32, 32, dtype=torch.bfloat16)
        for h in range(_VALUE_HEADS):
            for s, w in enumerate(_sources(h)):
                for p in range(5):
                    for token in range(T):
                        slot = s * T + token
                        for d in range(4):
                            arranged[h * 5 + p, 8 * slot + 2 * d] = rows[p][w * 128 + 32 * d: w * 128 + 32 * d + 32]
        self._weights = ttnn.from_torch(arranged.reshape(1, 1, _VALUE_HEADS * 5 * 32, 32), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=self.device,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._weights_key = key
        return self._weights

    # ------------------------------------------------------------------ call
    def __call__(self, mega, taps, bias, conv_state, state, db, ds, ones, gamma):
        """mega [1,1,T,12352] (or [1,T,12352]) L1 tiles; state: live row-major L1 [1,32,128,128]. Returns
        (prepared o_proj input, state result, conv result): T=1 -> (o [1,1,4096], state, conv_state) updated in
        place; T=2 -> (o [1,2,4096], (S1, S2) DRAM tile candidates, conv prefix candidates)."""
        T = self.num_tokens
        if int(mega.padded_shape[-1]) != _MEGA_TILES * 32 or int(np.prod(list(mega.shape)[:-1])) != T:
            raise ValueError(f"fused GDN core requires {T} packed mega rows of width {_MEGA_TILES * 32}")
        if mega.memory_config().buffer_type != ttnn.BufferType.L1 or mega.is_sharded():
            raise ValueError("fused GDN core reads the mega row from interleaved L1")
        candidates = tuple(
            ttnn.empty([1, _VALUE_HEADS, 128, 128], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(T)
        ) if T > 1 else ()
        weights = self._arranged_weights(taps, bias)
        tensors = [mega, weights, conv_state, state, db, ds, ones, gamma, self.inbox, self.output,
                   *self.conv_candidates, *candidates]
        signature = tuple((int(t.buffer_address()), str(t.memory_config())) for t in tensors)
        if self.program is None or signature != self.signature:
            self.program = self._make_program(mega, weights, conv_state, state, db, ds, ones, gamma, candidates)
            self.signature = signature
        ttnn.generic_op(tensors, self.program)
        if T == 1:
            return self.output, state, conv_state
        return self.output, candidates, self.conv_candidates
