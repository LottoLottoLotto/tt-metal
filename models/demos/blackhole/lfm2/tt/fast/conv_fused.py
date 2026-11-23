# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Whole LFM2 gated short-conv decode operator in one generic_op: in_proj + conv tail + out_proj, optionally with
the operator norm (residual add + RMSNorm) fused in front.

Replaces the 3-launch fast sequence (StreamDecodeMatmul conv_in -> ConvDecode -> StreamDecodeMatmul conv_out, itself
bitwise to ``layers.ShortConv.decode``), and with ``norm`` also the preceding ResidualRMSNorm launch::

    op = ConvOpDecode(conv, w_in, w_out)              # or ConvOpDecode.from_cache(conv, cache, layer)
    out = op(n)                                       # n [1, 1, B, 2048] BF16 TILE interleaved (L1 or DRAM)

    opn = ConvOpDecode(conv, w_in, w_out, norm=(operator_norm_weight, eps), placement=(0, 6))
    h, out = opn(a, b)                                # h = a + b, out = conv(rms_norm(h) * w)   (= rn(a, b); op(n))
    a_, out = opn(a)                                  # no add: out = conv(rms_norm(a) * w)

with the same numerics (bitwise out, s0, s1, h): in_proj / out_proj are the stream kernel's matmul_block sequence
(whole K in FP32 DEST, BF16 pack) on the same stream-layout weights (``cached_stream_weight`` keys
``L{i}.conv.in_proj.stream-c4w4`` / ``L{i}.conv.out_proj.stream-c4w2``: no new weight copies); the tail is
conv_decode.cpp's op sequence with the ttnn 16-bit-DEST SFPU arithmetic (RNE to BF16 after every op); the norm is
norm_matmul.NormPrologue (kernels/norm_prologue.hpp, bitwise ResidualRMSNorm).  Conv state is updated in place in
the ShortConv's s0/s1 tensors (layout unchanged: prefill / reset keep working).

Program (kernels/conv_fused.cpp): 32 workers, 4 per DRAM bank: placement "rows" = StreamDecodeMatmul's conv_in
workers (the first 32 row-major) + one hub core completing rows 0-2 of the 11-wide P150 grid; placement (x, y) = an
8 x 4 rectangle at (x, y) whose first worker is also the hub (required with ``norm``: the prologue runs on the 8 x 4
rectangle).  Each worker streams its in_proj run (6 bcx columns) from t = 0, sends every bcx tile to the owner of
its hidden column (2 columns per worker; L1 write + semaphore), runs the tail of its 2 columns, writes its y rows
into the hub; the hub multicasts all y tiles to every worker; the workers' out_proj runs (2 out columns each) were
streamed into L1 meanwhile.  Only the logical rows move; padding rows of the output are undefined (ttnn padding).
Measured: tests/fast/test_conv_fused.py, tests/fast/prof_conv_fused.py.
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from .decode_matmul import PRESETS, TILE_BYTES, cached_stream_weight, stream_columns, stream_cores

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "conv_fused.cpp")
_BANKS = 8
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
# CB ids (kernels/conv_fused.cpp)
CB_X, CB_WIN, CB_WOUT, CB_Y = 0, 1, 2, 3
K_B, K_X, K_C, K_T2, K_S1, K_T1, K_S0, K_T0 = 4, 5, 6, 7, 8, 9, 10, 11
K_BX, K_BXOUT, K_ACC, K_ACC2, K_ACC3, K_Y, CB_BCX, CB_OUT = 16, 17, 18, 19, 20, 21, 22, 23
_FP32_UNPACK = (K_B, K_X, K_T2, K_C, K_BX, K_ACC3)  # the SFPU multiplies' operands (as conv_decode.py)
_SEMS = 3  # 0: bcx tiles in, 1: hub gather, 2: y ready; the norm prologue takes 3-6
_NORM_SEMS = (3, 4, 5, 6)
_BRISC_COMMON = 13  # BRISC common runtime args before the norm prologue's


class _CachedProgram:
    def __init__(self):
        self.program, self.signature = None, None

    def run(self, tensors, build):
        sig = tuple((int(t.buffer_address()), tuple(t.shape), str(t.memory_config())) for t in tensors)
        if self.program is None or sig != self.signature:
            self.program, self.signature = build(), sig
        ttnn.generic_op(tensors, self.program)


class ConvOpDecode:
    """out = ShortConv.decode(x) for one conv layer in one launch (trace-safe, device-only; program cached per
    buffer addresses).  w_in / w_out: the layer's stream-layout weights (``cached_stream_weight`` with
    PRESETS["conv_in"] / PRESETS["conv_out"]; pass StreamDecodeMatmul's tensors to share them, or use
    ``from_cache``).  kb_* / slots / outstanding: streaming geometry (the layouts are fixed by the presets; slots
    default 4, + 8 with the norm so the in_proj ring covers the prologue).  placement: "rows" or the (x, y) origin
    of an 8 x 4 worker rectangle.  norm: (operator-norm weight [2048] torch, eps) -> fused residual add + RMSNorm
    (needs an 8 x 4 placement).  rows: the decode rows (default: the conv state's rows); fewer rows than the state
    run a decode-width bucket on state rows [0, rows) (the other slots' state rows are left unchanged).  taps: share
    another instance's replicated taps tensor (same layer).  verify: a candidate tensor [1, 1, 32, 2048] BF16 TILE
    DRAM -> speculative-verification mode (kernels/conv_fused.cpp CF_VERIFY): the rows (<= 16) are consecutive
    tokens of the sequence in state row 0, every row's Bx goes to `verify`, the state is left unchanged."""

    def __init__(self, conv, w_in, w_out, kb_in: int = 4, kb_out: int = 8, slots: int | None = None,
                 outstanding: int = 1, placement="rows", norm=None, output_memory_config=DRAM,
                 h_memory_config=L1, rows: int | None = None, taps=None, verify=None):
        H = conv.H
        self.conv, self.H = conv, H
        self.device = dev = conv.s0.device()
        state_rows = conv.s0.shape[-2]
        self.rows = state_rows if rows is None else rows
        self.cand = verify
        if verify is not None:
            if not 1 <= self.rows <= 16 or tuple(verify.padded_shape) != (1, 1, 32, H) \
                    or verify.dtype != ttnn.bfloat16 or verify.memory_config() != DRAM:
                raise ValueError("ConvOpDecode: verify needs <= 16 rows and a [1,1,32,H] BF16 TILE DRAM tensor")
        elif not 1 <= self.rows <= state_rows:
            raise ValueError(f"ConvOpDecode: {self.rows} rows for a {state_rows}-row conv state")
        self.keep_row = int(verify is None and self.rows < state_rows and self.rows < 16 and self.rows % 2 == 1)
        pin, pout = PRESETS["conv_in"], PRESETS["conv_out"]
        self.kt = H // 32
        self.n_workers = _BANKS * pin.cores_per_bank
        slots = slots or (12 if norm is not None else 4)
        if (H != 2048 or self.rows > 32 or pin.cores_per_bank != pout.cores_per_bank
                or self.kt % self.n_workers or self.kt % kb_in or self.kt % kb_out):
            raise ValueError("ConvOpDecode: hidden 2048, <= 32 rows, kb dividing 64, conv_in/out same cores")
        for w, n in ((w_in, 3 * H), (w_out, H)):
            if w.dtype != ttnn.bfloat8_b or tuple(w.padded_shape)[-2:] != (H, n) or w.memory_config() != DRAM:
                raise ValueError("ConvOpDecode: stream weights must be BFP8 DRAM-interleaved [2048, 6144 / 2048]")
        if not 1 <= outstanding < slots:
            raise ValueError("ConvOpDecode: need 1 <= outstanding < slots")
        if norm is not None and placement == "rows":
            raise ValueError("ConvOpDecode: the norm prologue needs an 8 x 4 placement, e.g. (0, 6)")
        self.w_in, self.w_out, self.pin, self.pout = w_in, w_out, pin, pout
        self.kb_in, self.kb_out, self.slots, self.outstanding = kb_in, kb_out, slots, outstanding
        self.per = self.kt // self.n_workers
        self.output_memory_config, self.h_memory_config = output_memory_config, h_memory_config
        if taps is None:
            k = torch.stack([ttnn.to_torch(t).reshape(H) for t in conv.taps])  # [3, H]: taps for t-2, t-1, t
            rep = k.reshape(1, 1, 1, 3 * H).expand(1, 1, 32, 3 * H).to(torch.bfloat16).contiguous()
            taps = ttnn.from_torch(rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=DRAM)
        self.taps = taps
        grid = dev.compute_with_storage_grid_size()
        if placement == "rows":
            # workers: StreamDecodeMatmul's (first workers row-major); hub: the next core; together whole rows
            _, bank_cores = stream_cores(dev, pin.cores_per_bank)
            n_rows = -(-(self.n_workers + 1) // grid.x)
            if n_rows > grid.y or n_rows * grid.x != self.n_workers + 1:
                raise ValueError("ConvOpDecode: the workers + hub must fill whole grid rows (P150: 32 + 1 = 3 x 11)")
            self.rect = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, n_rows - 1))
            self.hub = ttnn.CoreCoord(self.n_workers % grid.x, self.n_workers // grid.x)
        else:
            ox, oy = placement
            gx, gy = 8, self.n_workers // 8
            if self.n_workers % 8 or ox + gx > grid.x or oy + gy > grid.y:
                raise ValueError(f"ConvOpDecode: an 8 x {gy} rectangle at {placement} does not fit the grid")
            self.rect = ttnn.CoreRange(ttnn.CoreCoord(ox, oy), ttnn.CoreCoord(ox + gx - 1, oy + gy - 1))
            anchors = dev.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
            free = {(ox + i, oy + j) for i in range(gx) for j in range(gy)}
            bank_cores = [[] for _ in range(_BANKS)]
            for _ in range(pin.cores_per_bank):  # per round, each bank takes its nearest free core
                for b, a in enumerate(anchors):
                    best = min(free, key=lambda c: (abs(c[0] - a.x) + abs(c[1] - a.y), c[1], c[0]))
                    free.remove(best)
                    bank_cores[b].append(ttnn.CoreCoord(*best))
            self.hub = None  # worker 0
        self.norm = None
        if norm is not None:
            from .norm_matmul import NormPrologue

            weight, eps = norm
            self.norm = NormPrologue(dev, torch.as_tensor(weight), eps, origin=tuple(placement), gx=8,
                                     gy=self.n_workers // 8, sems=_NORM_SEMS)
            if self.norm.kt != self.kt:
                raise ValueError("ConvOpDecode: norm width != hidden")
        tb = TILE_BYTES[ttnn.bfloat8_b]
        # (core, bank, in run bytes, first in col, in cols, out run bytes, first out col, out cols); idx = list order
        cols_in = stream_columns(3 * self.kt, pin.cores_per_bank)
        cols_out = stream_columns(self.kt, pout.cores_per_bank)
        self.workers = []
        for b in range(_BANKS):
            run_in = run_out = 0
            for c, core in enumerate(bank_cores[b]):
                (fi, ni), (fo, no) = cols_in[b][c], cols_out[b][c]
                self.workers.append((core, b, run_in * tb, fi, ni, run_out * tb, fo, no))
                run_in += ni * self.kt
                run_out += no * self.kt
        self.max_out_tiles = max(-(-w[7] // pout.w) * pout.w * self.kt for w in self.workers)
        self._progs = {}
        self.prof = False  # CF_PROF: device-profiler zones (tests/fast/prof_conv_fused.py)
        self.out_tile_rows = 0  # 0: automatic (small_tile_rows); 32 forces full tiles

    @classmethod
    def from_cache(cls, conv, cache, layer: int, **kw):
        """Stream weights through the WeightCache under the keys StreamDecodeMatmul uses (no new entries when they
        exist; note each call loads its own device copy: share tensors when both paths are built)."""
        return cls(conv, cached_stream_weight(cache, f"L{layer}.conv.in_proj", PRESETS["conv_in"]),
                   cached_stream_weight(cache, f"L{layer}.conv.out_proj", PRESETS["conv_out"]), **kw)

    def small_tile_rows(self, rows: int) -> int:
        """out_proj in0 / out tile height (8, 16 or 32): the smallest that holds the rows (self.out_tile_rows
        overrides)."""
        if self.out_tile_rows:
            return self.out_tile_rows
        return 8 if rows <= 8 else 16 if rows <= 16 else 32

    def _build(self, x, out, rows, b=None, h=None):
        """x: the input (with the norm: a); b / h: the norm's residual input / h output (None: no add)."""
        from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
            PerCoreRuntimeArgsDescriptor,
            UnifiedKernelDescriptor,
        )

        dev, nm = self.device, self.norm
        residual = b is not None
        cs = ttnn.CoreRangeSet([self.rect])
        tb = TILE_BYTES[ttnn.bfloat8_b]
        wi, wo = self.pin.w, self.pout.w
        th = self.small_tile_rows(rows)

        def cb(index, dtype, page, pages, h_=32):
            tile = ttnn.TileDescriptor(ttnn.Tile([h_, 32]))
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        bf = ttnn.bfloat16
        cbs = [cb(CB_X, bf, 2048, self.kt), cb(CB_WIN, ttnn.bfloat8_b, tb, self.slots * self.kb_in * wi),
               cb(CB_WOUT, ttnn.bfloat8_b, tb, self.max_out_tiles), cb(CB_Y, bf, th * 64, self.kt, th),
               cb(CB_BCX, bf, 2048, 2 * wi), cb(CB_OUT, bf, th * 64, 2 * wo, th)]
        cbs += [cb(i, bf, 2048, self.per) for i in (K_B, K_X, K_C, K_T2, K_S1, K_T1, K_S0, K_T0, K_BX, K_BXOUT, K_ACC,
                                                    K_ACC2, K_ACC3, K_Y)]
        # the norm's RMS reduction needs HiFi4; the matmuls keep HiFi2 through explicit-fidelity calls (mm_fid)
        fid = ttnn.MathFidelity.HiFi4 if nm is not None else ttnn.MathFidelity.HiFi2
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        for i in _FP32_UNPACK:
            mode[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode.extend(mode)

        def noc(c):
            return dev.worker_core_from_logical_core(c)

        hub_core = self.hub if self.hub is not None else self.workers[0][0]
        lo, hi, hub = noc(self.rect.start), noc(self.rect.end), noc(hub_core)
        owner = [noc(w[0]) for w in self.workers for _ in range(self.per)]  # owner of hidden column j
        n_rect = (self.rect.end.x - self.rect.start.x + 1) * (self.rect.end.y - self.rect.start.y + 1)
        common = [("kt", self.kt), ("ht", self.kt), ("rows", rows), ("per", self.per), ("w_in", wi),
                  ("kb_in", self.kb_in), ("w_out", wo), ("kb_out", self.kb_out), ("th", th), ("keep_row", self.keep_row)]
        accessors = (list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
                     + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()))
        brisc_common = ([int(t.buffer_address()) for t in (x, out, self.taps, self.conv.s0, self.conv.s1)]
                        + [hi.x, hi.y, lo.x, lo.y, n_rect - 1]  # NOC1 multicast: start/end swapped
                        + [self.n_workers, hub.x, hub.y])
        assert len(brisc_common) == _BRISC_COMMON
        if self.cand is not None:
            brisc_common.append(int(self.cand.buffer_address()))  # CF_VERIFY: index 13, before the prologue's
        np_dm, np_c, sems = [], [], []
        if nm is not None:
            np_dm = nm.named_args(rows, residual, cb_x=CB_X)  # (NCRISC too: the header's constants read them)
            np_c = nm.named_args(rows, residual, cb_x=CB_X, dm=False) + nm.compute_args()
            accessors += nm.accessor_args(x, b, h)
            brisc_common += nm.runtime_args(x, b, h)
            cbs += nm.cbs(residual)
            sems = nm.semaphores()
        nc, bc, tc = [], [], []
        for idx, (core, bank, run_in, fi, ni, run_out, fo, no) in enumerate(self.workers):
            nc.append((core, [bank, run_in, ni, run_out, no]))
            own = [v for t in range(fi, fi + ni) for v in (owner[t % self.kt].x, owner[t % self.kt].y)]
            bc.append((core, [idx, fi, ni, fo, no, int(self.hub is None and idx == 0)] + own))
            tc.append((core, [ni, no]))
        if self.hub is not None:  # in_ncols 0: the hub-only role
            nc.append((self.hub, [0, 0, 0, 0, 0]))
            bc.append((self.hub, [0, 0, 0, 0, 0, 1]))
            tc.append((self.hub, [0, 0]))
        kernel = UnifiedKernelDescriptor(
            kernel_source=KERNEL,
            core_ranges=cs,
            defines=[("CF_PROF", str(int(self.prof))), ("CF_NORM", str(int(nm is not None))),
                     ("CF_VERIFY", str(int(self.cand is not None)))],
            ncrisc_named_compile_time_args=common + np_dm + [("wt_bytes", tb), ("slots", self.slots),
                                                             ("outstanding", self.outstanding)],
            brisc_named_compile_time_args=common + np_dm,
            brisc_compile_time_args=accessors,
            trisc_named_compile_time_args=common + np_c + [("mm_fid", 2)],
            ncrisc_common_runtime_args=[int(self.w_in.buffer_address()), int(self.w_out.buffer_address())],
            brisc_common_runtime_args=brisc_common,
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(ncrisc_args=nc, brisc_args=bc,
                                                                          trisc_args=tc),
        )
        sems += [ttnn.SemaphoreDescriptor(id=s, core_ranges=cs, initial_value=0) for s in range(_SEMS)]
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=sems)

    def _check(self, t, name):
        shape = tuple(t.shape)
        if (shape != (1, 1, self.rows, self.H) or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT
                or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            raise ValueError(f"ConvOpDecode: {name} must be BF16 TILE interleaved [1,1,{self.rows},{self.H}], "
                             f"got {shape}")

    def __call__(self, x, b=None, output_memory_config=None, h_memory_config=None):
        """Without norm: x [1, 1, B, 2048] BF16 TILE interleaved (L1 or DRAM; B = the conv state's rows) -> out
        [1, 1, B, 2048] BF16 TILE (output_memory_config, default the constructor's).  With norm: x = a (and the
        residual b) -> (h, out), h = a + b (new tensor, h_memory_config) or a itself when b is None, out = the conv
        operator on rms_norm(h) * w.  s0/s1 shifted in place (verify mode: unchanged; every row's Bx -> the
        candidate tensor)."""
        self._check(x, "x")
        if b is not None:
            if self.norm is None:
                raise ValueError("ConvOpDecode: a residual input needs the norm prologue")
            self._check(b, "b")
        out = ttnn.empty([1, 1, self.rows, self.H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                         memory_config=output_memory_config or self.output_memory_config)
        weights = [self.w_in, self.w_out, self.taps, self.conv.s0, self.conv.s1]
        if self.cand is not None:
            weights.append(self.cand)
        if self.norm is None:
            self._progs.setdefault("op", _CachedProgram()).run(
                [x] + weights + [out], lambda: self._build(x, out, self.rows))
            return out
        if b is None:
            self._progs.setdefault("norm", _CachedProgram()).run(
                [x] + weights + [self.norm.gamma, out], lambda: self._build(x, out, self.rows))
            return x, out
        h = ttnn.empty([1, 1, self.rows, self.H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=h_memory_config or self.h_memory_config)
        self._progs.setdefault("add+norm", _CachedProgram()).run(
            [x, b] + weights + [self.norm.gamma, h, out], lambda: self._build(x, out, self.rows, b, h))
        return h, out
