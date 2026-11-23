# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Whole LFM3 gated short-conv decode operator in one generic_op: in_proj + 7-tap conv tail + out_proj, optionally
with the sandwich residual + operator norm fused in front.

Replaces the 3-launch fast sequence (StreamDecodeMatmul conv_in -> ConvDecode -> StreamDecodeMatmul conv_out, itself
bitwise to ``layers.ShortConv.decode``), and with ``norm`` also the preceding ResidualRMSNorm launch::

    op = ConvOpDecode(conv, w_in, w_out)              # or ConvOpDecode.from_cache(conv, cache, layer)
    out = op(x)                                       # x [1, 1, rows, 2048] BF16 TILE interleaved (L1 or DRAM)

    opn = ConvOpDecode(conv, w_in, w_out, norm=(operator_norm_weight, eps), post=post_ffn_norm_weight_prev)
    h, out = opn(a, b)          # h = a + bf16(rms_norm(b) * post) (post None: a + b), out = conv(rms_norm(h) * w)
    a_, out = opn(a)            # no add (layer 0, the embedding): out = conv(rms_norm(a) * w)

with the same numerics (bitwise out, h and the six state tensors): in_proj / out_proj are the stream kernel's
matmul_block sequence (whole K in FP32 DEST, BF16 pack) on the same stream-layout weights (``cached_stream_weight``
keys ``L{i}.conv.in_proj.stream-c4w4`` / ``L{i}.conv.out_proj.stream-c4w2``, BFP8 or BFP4: the weights' own
format); the tail is conv_tail.hpp (bitwise the stock 7-tap FP32-accumulated conv); the norm is
norm_matmul.NormPrologue (kernels/norm_prologue.hpp, bitwise ResidualRMSNorm(post_weight=post)).  The conv state is
updated in place in the ShortConv's six state tensors (layout unchanged: prefill / reset / save / restore keep working).

Program (kernels/conv_fused.cpp): 32 workers, 4 per DRAM bank: placement "rows" = StreamDecodeMatmul's conv_in
workers (the first 32 row-major) + one hub core completing rows 0-2 of the 11-wide P150 grid; placement (x, y) = an
8 x 4 rectangle at (x, y) whose first worker is also the hub (required with ``norm``: the prologue runs on the 8 x 4
rectangle; default (0, 6)).  Each worker reads its 2 tail columns' taps / state rows and computes their FP32 state
partial sum first, streams its in_proj run (6 bcx columns), sends every bcx tile to the owner of its hidden column
(L1 write + semaphore), finishes the tail of its 2 columns, writes its y rows into the hub, shifts its state columns;
the hub multicasts all y tiles to every worker; the workers' out_proj runs (2 out columns each) were streamed into
L1 meanwhile.  Only the logical rows move; padding rows of the output are undefined (ttnn padding).
Measured: tests/fast/test_conv_fused.py.
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from .conv_decode import check_conv, replicated_taps
from .decode_matmul import PRESETS, TILE_BYTES, cached_stream_weight, stream_columns, stream_cores

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "conv_fused.cpp")
_BANKS = 8
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
# CB ids (kernels/conv_fused.cpp; 12-15 and 24-29 are the norm prologue's)
CB_X, CB_WIN, CB_WOUT, CB_Y = 0, 1, 2, 3
K_B, K_X, K_C, K_T, K_T6, K_S, K_ACC, K_BXOUT, K_Y, CB_BCX, CB_OUT = 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18
_FP32_UNPACK = (K_B, K_X, K_C, K_T, K_T6, K_S, K_ACC)  # the tail's operands (conv_tail.hpp)
K_RAW = 19  # CF_VERIFY: the state rows as read + the row map (BRISC scratch)
_SEMS = 3  # 0: bcx tiles in, 1: hub gather, 2: y ready; the norm prologue takes 3-6
_NORM_SEMS = (3, 4, 5, 6)
_BRISC_COMMON = 17  # BRISC common runtime args before the norm prologue's (conv_fused.cpp np_dm<..., 17>)
_N_STATE = 6


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
    PRESETS["conv_in"] / PRESETS["conv_out"] and the weights' real dtype, BFP8 or BFP4; pass StreamDecodeMatmul's
    tensors to share them, or use ``from_cache``).  kb_* / slots / outstanding: streaming geometry (the layouts are
    fixed by the presets; slots: in_proj ring depth in kb_in x 4-column blocks, default 24 = 418 KB at BFP8, most of
    a worker's 6-column run, so the stream never waits for TRISC (state partial / norm prologue first); measured
    sweep: 4 -> 24 slots 46.1 -> 39.7 us at B = 1, 61.1 -> 43.7 at B = 16).  placement:
    "rows" or the (x, y) origin of an 8 x 4 worker rectangle (default: "rows" without norm, (0, 6) with).  norm:
    (operator-norm weight [2048] torch, eps) -> fused residual add + RMSNorm (needs an 8 x 4 placement); post: the
    previous layer's post_ffn_norm weight [2048] (sandwich residual h = a + rms_norm(b) * post; None: h = a + b).
    rows: the decode rows (default: the conv state's rows); fewer rows than the state run a decode-width bucket on
    state rows [0, rows) (the other slots' state rows are left unchanged).  taps: ``conv_decode.replicated_taps``
    of the same layer, shared between widths (default: built here).  verify: a candidate tensor [1, 1, 32, 2048] BF16
    TILE DRAM -> speculative-verification mode (kernels/conv_fused.cpp CF_VERIFY, with the row map `vmap`, [1, 16]
    uint32 ROW_MAJOR DRAM): the rows (<= 16) are groups of consecutive tokens, one per sequence (row r's left context
    = its group's conv-state slot and the group's rows before r), every row's Bx goes to `verify`, the state is left
    unchanged."""

    def __init__(self, conv, w_in, w_out, kb_in: int = 4, kb_out: int = 8, slots: int = 24,
                 outstanding: int = 1, placement=None, norm=None, post=None, output_memory_config=DRAM,
                 h_memory_config=L1, rows: int | None = None, taps=None, verify=None, vmap=None):
        H = conv.H
        self.conv, self.H = conv, H
        self.device = dev = conv.s[0].device()
        self.rows, keep = check_conv(conv, rows, "ConvOpDecode", verify, vmap)
        self.cand, self.vmap = verify, vmap
        self.keep_row = int(keep)
        pin, pout = PRESETS["conv_in"], PRESETS["conv_out"]
        self.kt = H // 32
        self.n_workers = _BANKS * pin.cores_per_bank
        if placement is None:
            placement = (0, 6) if norm is not None else "rows"
        if (pin.cores_per_bank != pout.cores_per_bank or self.kt % self.n_workers or self.kt % kb_in
                or self.kt % kb_out):
            raise ValueError("ConvOpDecode: kb dividing 64, conv_in/out same cores")
        for w, n in ((w_in, 3 * H), (w_out, H)):
            if (w.dtype not in TILE_BYTES or tuple(w.padded_shape)[-2:] != (H, n)
                    or w.memory_config().buffer_type != ttnn.BufferType.DRAM
                    or w.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
                raise ValueError("ConvOpDecode: stream weights must be BFP8/BFP4/BF16 DRAM-interleaved "
                                 "[2048, 6144 / 2048]")
        if not 1 <= outstanding < slots:
            raise ValueError("ConvOpDecode: need 1 <= outstanding < slots")
        if norm is not None and placement == "rows":
            raise ValueError("ConvOpDecode: the norm prologue needs an 8 x 4 placement, e.g. (0, 6)")
        if post is not None and norm is None:
            raise ValueError("ConvOpDecode: post needs norm")
        self.w_in, self.w_out, self.pin, self.pout = w_in, w_out, pin, pout
        self.kb_in, self.kb_out, self.slots, self.outstanding = kb_in, kb_out, slots, outstanding
        self.per = self.kt // self.n_workers
        self.output_memory_config, self.h_memory_config = output_memory_config, h_memory_config
        self.taps = replicated_taps(conv) if taps is None else taps
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
                                     gy=self.n_workers // 8, sems=_NORM_SEMS,
                                     post_weight=None if post is None else torch.as_tensor(post))
            if self.norm.kt != self.kt:
                raise ValueError("ConvOpDecode: norm width != hidden")
        tb_in, tb_out = TILE_BYTES[w_in.dtype], TILE_BYTES[w_out.dtype]
        # (core, bank, in run bytes, first in col, in cols, out run bytes, first out col, out cols); idx = list order
        cols_in = stream_columns(3 * self.kt, pin.cores_per_bank)
        cols_out = stream_columns(self.kt, pout.cores_per_bank)
        self.workers = []
        for b in range(_BANKS):
            run_in = run_out = 0
            for c, core in enumerate(bank_cores[b]):
                (fi, ni), (fo, no) = cols_in[b][c], cols_out[b][c]
                self.workers.append((core, b, run_in * tb_in, fi, ni, run_out * tb_out, fo, no))
                run_in += ni * self.kt
                run_out += no * self.kt
        self.max_out_tiles = max(-(-w[7] // pout.w) * pout.w * self.kt for w in self.workers)
        self._progs = {}
        self.prof = False  # CF_PROF: device-profiler zones
        self.out_tile_rows = 0  # 0: automatic (small_tile_rows); 32 forces full tiles

    @classmethod
    def from_cache(cls, conv, cache, layer: int, dtype_in=ttnn.bfloat8_b, dtype_out=ttnn.bfloat8_b, **kw):
        """Stream weights through the WeightCache under the keys StreamDecodeMatmul uses (dtype_*: the weights' real
        formats; no new entries when they exist; each call loads its own device copy: share tensors when both paths
        are built)."""
        return cls(conv, cached_stream_weight(cache, f"L{layer}.conv.in_proj", PRESETS["conv_in"], dtype_in),
                   cached_stream_weight(cache, f"L{layer}.conv.out_proj", PRESETS["conv_out"], dtype_out), **kw)

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
        wi, wo = self.pin.w, self.pout.w
        th = self.small_tile_rows(rows)

        def cb(index, dtype, page, pages, h_=32):
            tile = ttnn.TileDescriptor(ttnn.Tile([h_, 32]))
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        bf, per = ttnn.bfloat16, self.per
        tbi, tbo = TILE_BYTES[self.w_in.dtype], TILE_BYTES[self.w_out.dtype]
        cbs = [cb(CB_X, bf, 2048, self.kt), cb(CB_WIN, self.w_in.dtype, tbi, self.slots * self.kb_in * wi),
               cb(CB_WOUT, self.w_out.dtype, tbo, self.max_out_tiles), cb(CB_Y, bf, th * 64, self.kt, th),
               cb(CB_BCX, bf, 2048, 2 * wi), cb(CB_OUT, bf, th * 64, 2 * wo, th),
               cb(K_T, bf, 2048, _N_STATE * per), cb(K_S, bf, 2048, _N_STATE * per),
               cb(K_ACC, ttnn.float32, 4096, per)]
        cbs += [cb(i, bf, 2048, per) for i in (K_B, K_X, K_C, K_T6, K_BXOUT, K_Y)]
        if self.cand is not None:  # CF_VERIFY: the state rows as read + the row map (BRISC scratch)
            cbs.append(cb(K_RAW, bf, 2048, _N_STATE * per + 1))
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
        owner = [noc(w[0]) for w in self.workers for _ in range(per)]  # owner of hidden column j
        n_rect = (self.rect.end.x - self.rect.start.x + 1) * (self.rect.end.y - self.rect.start.y + 1)
        common = [("kt", self.kt), ("ht", self.kt), ("rows", rows), ("per", per), ("w_in", wi),
                  ("kb_in", self.kb_in), ("w_out", wo), ("kb_out", self.kb_out), ("th", th), ("keep_row", self.keep_row)]
        accessors = (list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
                     + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()))
        brisc_common = ([int(t.buffer_address()) for t in (x, out, self.taps, *self.conv.s)]
                        + [hi.x, hi.y, lo.x, lo.y, n_rect - 1]  # NOC1 multicast: start/end swapped
                        + [self.n_workers, hub.x, hub.y])
        assert len(brisc_common) == _BRISC_COMMON
        if self.cand is not None:  # CF_VERIFY: indices 17 / 18, before the prologue's
            brisc_common += [int(self.cand.buffer_address()), int(self.vmap.buffer_address())]
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
            ncrisc_named_compile_time_args=common + np_dm + [("wt_in_bytes", tbi), ("wt_out_bytes", tbo),
                                                             ("slots", self.slots), ("outstanding", self.outstanding)],
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
        """Without norm: x [1, 1, rows, 2048] BF16 TILE interleaved (L1 or DRAM) -> out [1, 1, rows, 2048] BF16 TILE
        (output_memory_config, default the constructor's).  With norm: x = a (and the residual b) -> (h, out),
        h = a + post_norm(b) (new tensor, h_memory_config) or a itself when b is None, out = the conv operator on
        rms_norm(h) * w.  State rows [0, rows) shifted in place."""
        self._check(x, "x")
        if b is not None:
            if self.norm is None:
                raise ValueError("ConvOpDecode: a residual input needs the norm prologue")
            self._check(b, "b")
        out = ttnn.empty([1, 1, self.rows, self.H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                         memory_config=output_memory_config or self.output_memory_config)
        weights = [self.w_in, self.w_out, self.taps, *self.conv.s] + (
            [self.cand, self.vmap] if self.cand is not None else [])
        if self.norm is None:
            self._progs.setdefault("op", _CachedProgram()).run(
                [x] + weights + [out], lambda: self._build(x, out, self.rows))
            return out
        if b is None:
            self._progs.setdefault("norm", _CachedProgram()).run(
                [x] + weights + self.norm.tensors() + [out], lambda: self._build(x, out, self.rows))
            return x, out
        h = ttnn.empty([1, 1, self.rows, self.H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=h_memory_config or self.h_memory_config)
        self._progs.setdefault("add+norm", _CachedProgram()).run(
            [x, b] + weights + self.norm.tensors() + [h, out], lambda: self._build(x, out, self.rows, b, h))
        return h, out
