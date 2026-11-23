# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Prefill sandwich-norm residual for LFM3: one generic_op per layer boundary (kernels/residual_norm_prefill.cpp).

Replaces, in ``model.Layer`` prefill, the stock chain at every residual (48 per chunk)::

    p = post(o); h = ttnn.add(x, p); n = nxt(h)        # 3 programs (rms_norm, add, rms_norm)

with ``h, n = PrefillResidualNorm(post, nxt)(x, o)`` (post / nxt: layers.RMSNorm; nxt None -> n None, e.g. after the
last layer): h and n bitwise the stock chain (tests/prefill/norm.py).  x, o, h, n: [1, 1, Tp, 2048] BF16 TILE DRAM
interleaved, Tp % 32 == 0; x and o are not freed.  Typical use::

    h, n2 = PrefillResidualNorm(layer.post_op_norm, layer.ffn_norm)(x, o)        # operator residual + ffn norm
    out, n = PrefillResidualNorm(layer.post_ffn_norm, next.op_norm)(h, f)        # ffn residual + next op norm

Numerics: the stock rms_norm arithmetic op for op (layernorm.cpp RMSNORM under CK_NORM) with the row sum of squares
accumulated in DEST over the row's 64 column tiles in column order even when a row is split over several cores
(the raw FP32 partial DEST tile travels to the next core and is unpacked straight into DEST); see the kernel header.

Work split: the Tp/32 x 64 tiles in flat row-major order, one contiguous range of <= 64 tiles per core over
``n_cores`` cores (default: DEFAULT_CORES per bucket, measured, else ~16 tiles per core; capped at the grid).  The
program, per-core arguments and CBs are cached per Tp; a call only rebuilds the descriptor's address arguments.
"""
from __future__ import annotations

import struct
from pathlib import Path

import ttnn

from ..fast.conv_decode import CachedProgram, cb_descriptors, row_ranges

KERNEL = str(Path(__file__).resolve().parent.parent / "kernels" / "residual_norm_prefill.cpp")
DRAM = ttnn.DRAM_MEMORY_CONFIG
HIDDEN = 2048
COLS = HIDDEN // 32
MAX_RANGE = 64        # tiles per core (the kernel holds o, x, h, n, gamma rows of its range in L1; <= 2 segments)
IN, OUT = 1, 2        # segment flags (kernel)
# kernel CB ids
CB_O, CB_X, CB_G1, CB_G2, CB_SC, CB_EPS, CB_PIN1, CB_PIN2, CB_RR1, CB_RR2 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
CB_H, CB_N, CB_P, CB_POUT1, CB_POUT2, CB_RS1, CB_RS2, CB_SQ, CB_EX, CB_RL = 16, 17, 18, 19, 20, 21, 22, 24, 25, 26
# cores per prefill bucket (logs/prefill2/nf_v5.json, nf_v8.json: fastest measured); other Tp: ~16 tiles per core
DEFAULT_CORES = {128: 16, 256: 32, 512: 64, 1024: 64, 2048: 110}


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def plan(row_tiles: int, n_cores: int) -> list[dict]:
    """Per core: its tile range [a, a + T) of the flat row-major tile order and the kernel's segment roles."""
    units = row_tiles * COLS
    n = min(n_cores, units)
    bounds = [c * units // n for c in range(n + 1)]
    if max(b - a for a, b in zip(bounds[:-1], bounds[1:])) > MAX_RANGE:
        raise ValueError(f"PrefillResidualNorm: {n} cores give > {MAX_RANGE} tiles per core at {row_tiles} row tiles")
    owner = [0] * units
    for c in range(n):
        owner[bounds[c]:bounds[c + 1]] = [c] * (bounds[c + 1] - bounds[c])
    cores = []
    for c in range(n):
        a, b = bounds[c], bounds[c + 1]
        cut = min(b, (a // COLS + 1) * COLS)
        segs = [(a, cut)] + ([(cut, b)] if cut < b else [])
        fl = [(IN if s % COLS else 0) | (OUT if e % COLS else 0) for s, e in segs]
        peers = []
        if fl[0] == IN:  # finishes a row that started on earlier cores: they get its rstd
            peers = list(range(owner[a // COLS * COLS], c))
        cores.append({"a": a, "T": b - a, "segs": segs, "flags": fl, "next": c + 1 if fl[-1] & OUT else None,
                      "peers": peers})
    return cores


def _compute_config():
    """CK_NORM (HiFi4, FP32 DEST, exact SFPU), the incoming partial DEST tiles unpacked straight to DEST (FP32)."""
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, dst_full_sync_en=False)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for i in (CB_PIN1, CB_PIN2):
        unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode.extend(unpack)
    return cfg


class PrefillResidualNorm:
    """(h, n) = op(x, o): h = x + rms_norm(o) * post.w, n = rms_norm(h) * nxt.w (n None when nxt is None).
    post / nxt: layers.RMSNorm (gamma [1, 1, 64, 32] row-major BF16 DRAM, eps).  n_cores: override DEFAULT_CORES.
    extra_defines: kernel defines (tests: RN_ABL_NOMATH)."""

    def __init__(self, post, nxt=None, n_cores: int | None = None):
        self.post, self.nxt = post, nxt
        self.device = post.w.device()
        for norm in (post, nxt):
            if norm is not None and (tuple(norm.w.shape) != (1, 1, COLS, 32) or norm.w.layout != ttnn.ROW_MAJOR_LAYOUT
                                     or norm.w.dtype != ttnn.bfloat16):
                raise ValueError("PrefillResidualNorm: norm weights must be [1, 1, 64, 32] row-major BF16")
        grid = self.device.compute_with_storage_grid_size()
        self.grid_x, self.grid_cores = grid.x, grid.x * grid.y
        self.n_cores = n_cores
        self.extra_defines: list[tuple[str, str]] = []
        self._progs: dict[int, CachedProgram] = {}
        self._static: dict[int, dict] = {}

    def cores_for(self, Tp: int) -> int:
        units = Tp // 32 * COLS
        n = self.n_cores or DEFAULT_CORES.get(Tp) or max(-(-units // MAX_RANGE), units // 16)
        return min(n, self.grid_cores)

    def _static_parts(self, Tp: int) -> dict:
        """Everything but the tensor addresses: cores, the common-arg tail (split + NOC table), CBs, semaphores,
        compile args.  The kernel derives each core's range and roles from its logical coordinates (= plan())."""
        if Tp in self._static:
            return self._static[Tp]
        P = plan(Tp // 32, self.cores_for(Tp))
        core_list = [ttnn.CoreCoord(i % self.grid_x, i // self.grid_x) for i in range(len(P))]
        cores = row_ranges(core_list)
        tail = [Tp // 32, len(P), self.grid_x]
        for c in core_list:
            xy = self.device.worker_core_from_logical_core(c)
            tail += [xy.x, xy.y]
        has_next = self.nxt is not None
        T = max(p["T"] for p in P)
        sq = -(-max(e - s for p in P for s, e in p["segs"]) // 4) * 4
        cbs = []
        for i, k in ((CB_O, T), (CB_X, T), (CB_G1, T), (CB_H, T), (CB_SC, 1), (CB_EPS, 2), (CB_P, 8)):
            cbs += cb_descriptors([i], k, cores)
        if has_next:
            cbs += cb_descriptors([CB_G2], T, cores) + cb_descriptors([CB_N], T, cores)
        for i in (CB_PIN1, CB_PIN2, CB_RR1, CB_RR2, CB_POUT1, CB_POUT2, CB_RS1, CB_RS2, CB_EX, CB_RL):
            cbs += cb_descriptors([i], 1, cores, dtype=ttnn.float32, page=4096)
        cbs += cb_descriptors([CB_SQ], sq, cores, dtype=ttnn.float32, page=4096)
        eps2 = self.nxt.eps if has_next else self.post.eps
        self._static[Tp] = {
            "cores": cores, "tail": tail, "cbs": cbs,
            "sems": [ttnn.SemaphoreDescriptor(id=s, core_ranges=cores, initial_value=0) for s in range(4)],
            "ct": [int(has_next), f32_bits(1.0 / HIDDEN), f32_bits(self.post.eps), f32_bits(eps2)],
        }
        return self._static[Tp]

    def _build(self, x, o, h, n):
        st = self._static_parts(x.shape[-2])
        addr = lambda t: int(t.buffer_address()) if t is not None else 0  # noqa: E731
        common = [addr(o), addr(x), addr(self.post.w), addr(self.nxt.w if self.nxt else None), addr(h), addr(n)]
        dm = dict(kernel_source=KERNEL, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                  core_ranges=st["cores"], compile_time_args=st["ct"], common_runtime_args=common + st["tail"],
                  defines=list(self.extra_defines))
        kernels = [  # reads on NOC 0 and writes on NOC 1 (the reverse measured 1.5x slower)
            ttnn.KernelDescriptor(**dm, config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default)),
            ttnn.KernelDescriptor(**dm, config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default)),
            ttnn.KernelDescriptor(**dm, config=_compute_config()),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=st["cbs"], semaphores=st["sems"])

    def __call__(self, x, o):
        """x (residual stream), o (operator / MLP output): [1, 1, Tp, 2048] BF16 TILE DRAM -> (h, n) new tensors."""
        Tp = x.shape[-2]
        for t in (x, o):
            if (tuple(t.padded_shape) != (1, 1, Tp, HIDDEN) or Tp % 32 or t.dtype != ttnn.bfloat16
                    or t.layout != ttnn.TILE_LAYOUT or t.memory_config() != DRAM):
                raise ValueError(f"PrefillResidualNorm: x, o must be [1, 1, Tp, {HIDDEN}] BF16 TILE DRAM (same Tp)")
        mk = lambda: ttnn.empty([1, 1, Tp, HIDDEN], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                device=self.device, memory_config=DRAM)
        h = mk()
        n = mk() if self.nxt is not None else None
        tensors = [x, o, self.post.w] + ([self.nxt.w, n] if n is not None else []) + [h]
        self._progs.setdefault(Tp, CachedProgram()).run(tensors, lambda: self._build(x, o, h, n))
        return h, n
