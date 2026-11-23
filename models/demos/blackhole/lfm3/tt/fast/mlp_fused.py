# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Dense MLP decode in ONE program: `FusedMLPDecode` = NormStreamDecodeMatmul(ffn norm + w1|w3 SwiGLU) followed by
StreamDecodeMatmul(w2), bitwise (kernels/mlp_fused.cpp).

    h, f = FusedMLPDecode(dev, ffn_norm_w, eps, w13_stream, w13_preset, w2_stream, w2_preset, post_weight=post_op)(x, o)
      h = x + rms_norm(o) * post_op      (LFM3 sandwich; post_weight None: x + o; o None: h = x, norm only)
      f = w2(SiLU(n @ W1) * (n @ W3)),   n = rms_norm(h) * ffn_norm_w

One gx = 8 by gy = cores_per_bank rectangle: every core runs the norm prologue, is a w13 worker of DRAM bank b and the
w2 worker of the same bank.  Its NCRISC streams the w13 run from kernel start and then the w2 run (the ring holds most of it,
so DRAM never waits for g; optionally gated on every core's w13 reads); g tiles (8 / 16 / 32-row tiles by the row count) go
to the rectangle's master and come back as one multicast; w2 then runs over the full g block.  Saves the second launch and its pipeline fill / drain.
Same accumulation order and packs as the two-launch pair, so h, f are bitwise NormStreamDecodeMatmul(w13, glu) +
StreamDecodeMatmul(w2).  Measured in tests/fast/test_mlp_fused.py (logs/fast_mlp_fused.json).
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from .decode_matmul import TILE_BYTES, Preset, _CachedProgram, stream_columns
from .norm_matmul import _FID, NormPrologue, rect_bank_cores

_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "mlp_fused.cpp")
_BANKS = 8
_SEMS = (2, 3, 4, 5)  # norm prologue
_G_SEMS = (6, 7)  # g gather (count of cores), g ready
_W2_SEMS = (8, 9)  # w13 reads done (count of cores), w2 reads go
W2_RING_BYTES = 300 * 1024  # w2 ring cap per core (L1)
# Tuned in the fused program (tests/fast/test_mlp_fused.py, B = 1, 8 x 8 rectangle at (1, 2)): ~9-11 KB reads with
# one in flight per core keep both streams at ~475 GB/s (BFP4 w1|w3) / ~485 GB/s (BFP8); the larger in-flight reads
# of the standalone presets (w2 8, 1, 40, 3, 2: 43.5 KB x 2) left banks unevenly served (some cores 1.6x later).
# The w13 layouts (c8w4-glu) are those of PRESETS_BFP4["mlp_w13"]; W2_PRESET has the layout of PRESETS["mlp_w2"].
W13_PRESETS = {ttnn.bfloat4_b: Preset(8, 4, 4, 2, 1, glu=True), ttnn.bfloat8_b: Preset(8, 4, 2, 2, 1, glu=True)}
W2_PRESET = Preset(8, 1, 10, 2, 1)


def fused_presets(w13_dtype) -> tuple[Preset, Preset]:
    """(w13 preset, w2 preset) of FusedMLPDecode for w1|w3 weights of `w13_dtype` (BFP4 or BFP8)."""
    return W13_PRESETS[w13_dtype], W2_PRESET


class FusedMLPDecode:
    """(h, f) of the decode MLP half-layer in one generic_op (trace-safe).

    w13: glu stream weight (cached_stream_weight(cache, "L{i}.mlp.w1", w13_preset, dtype, key3="L{i}.mlp.w3")),
    w2: stream weight of w2_preset; both presets k_split 0 with the same cores_per_bank (2, 4 or 8: the 8 x cpb
    rectangle), w13_preset glu; tuned: fused_presets(w13 dtype).  BFP4 / BFP8 / BF16 weights (each its own format).  compute_kernel_config: the
    matmul fidelity (CK_BFP8 = HiFi2, as the decode MLP).  origin: the rectangle's first core (moved up to fit the
    grid; 8 x 8 at (1..3, 2) measured 72.3-72.5 us, (0, 2) 76, (0, 0) 88).  w2_gate: w2 reads wait for every core's
    w13 reads (measured +1.1 us with W2_PRESET, -6 us with the standalone w2 preset).  x, o: [1, 1, rows <= 32, 2048] BF16 TILE interleaved (DRAM or L1).  h: new tensor
    (h_memory_config, default L1; o None: h is x).  f: [1, 1, rows, 2048] BF16 (output_memory_config, default L1)."""

    def __init__(self, device, norm_weight: torch.Tensor, eps: float, w13, w13_preset: Preset, w2, w2_preset: Preset,
                 compute_kernel_config, post_weight: torch.Tensor | None = None, origin=(1, 2),
                 output_memory_config=ttnn.L1_MEMORY_CONFIG, h_memory_config=ttnn.L1_MEMORY_CONFIG,
                 slots: int | None = None, slots2: int | None = None, w2_gate: bool = False):
        p, p2 = w13_preset, w2_preset
        if not p.glu or p.k_split or p2.glu or p2.k_split or p.cores_per_bank != p2.cores_per_bank \
                or p.cores_per_bank not in (2, 4, 8):
            raise ValueError("FusedMLPDecode: glu w13 and plain w2 presets, k_split 0, equal cores_per_bank 2/4/8")
        self.device, self.p13, self.p2 = device, p, p2
        self.kt = int(w13.padded_shape[-2]) // 32
        self.n13 = int(w13.padded_shape[-1]) // 32
        self.kt2 = int(w2.padded_shape[-2]) // 32
        self.n2 = int(w2.padded_shape[-1]) // 32
        if self.kt2 * 2 != self.n13 or self.kt % p.kb or self.kt2 % p2.kb or p.w % 2:
            raise ValueError(f"FusedMLPDecode: shapes {self.kt}x{self.n13} / {self.kt2}x{self.n2} vs {p} / {p2}")
        for wt in (w13, w2):
            if (wt.dtype not in TILE_BYTES or wt.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
                    or wt.memory_config().buffer_type != ttnn.BufferType.DRAM):
                raise ValueError("FusedMLPDecode: weights must be DRAM-interleaved BFP4/BFP8/BF16")
        self.w13, self.w2 = w13, w2
        self.tb13, self.tb2 = TILE_BYTES[w13.dtype], TILE_BYTES[w2.dtype]
        self.slots = slots or min(p.slots + 8, max(p.slots, 400 * 1024 // (p.kb * p.w * self.tb13)))
        cols13 = stream_columns(self.n13, p.cores_per_bank, unit=p.w)
        cols2 = stream_columns(self.n2, p.cores_per_bank)
        blocks2 = max(-(-n // p2.w) for cols in cols2 for _, n in cols) * (self.kt2 // p2.kb)
        self.slots2 = slots2 or max(p2.outstanding + 1, min(blocks2, W2_RING_BYTES // (p2.kb * p2.w * self.tb2)))
        if not 1 <= p.outstanding < self.slots or not 1 <= p2.outstanding < self.slots2:
            raise ValueError("FusedMLPDecode: ring slots must exceed outstanding reads")
        gy = p.cores_per_bank
        grid = device.compute_with_storage_grid_size()
        origin = (origin[0], min(origin[1], grid.y - gy))
        self.norm = NormPrologue(device, norm_weight, eps, origin=origin, gx=_BANKS, gy=gy, sems=_SEMS,
                                 post_weight=post_weight)
        if self.norm.kt != self.kt:
            raise ValueError("FusedMLPDecode: norm width != w13 K")
        self.mm_fid = _FID[compute_kernel_config.math_fidelity]
        self.output_memory_config, self.h_memory_config = output_memory_config, h_memory_config
        cores = rect_bank_cores(device, self.norm, gy)
        self.core_args = []
        for b in range(_BANKS):
            run13 = run2 = 0
            for (f13, n13), (f2, n2), core in zip(cols13[b], cols2[b], cores[b]):
                self.core_args.append((core, [b, run13 * self.tb13, n13, f13 // 2, run2 * self.tb2, n2, f2]))
                run13 += n13 * self.kt
                run2 += n2 * self.kt2
        self.w2_gate = w2_gate  # w2 reads wait for every core's w13 reads (kernels/mlp_fused.cpp)
        self.prof = False  # MF_PROF: device-profiler zones (milestones from kernel start), debug only
        self._cache = {}

    def _build(self, a, b, h, f, rows):
        from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
            PerCoreRuntimeArgsDescriptor,
            UnifiedKernelDescriptor,
        )

        p, p2, cs, nm = self.p13, self.p2, self.norm.core_set, self.norm
        residual = b is not None
        tile = ttnn.TileDescriptor(ttnn.Tile([32, 32]))
        bf = ttnn.bfloat16

        th = next(t for t in (8, 16, 32) if rows <= t)  # g / f tile height (kernels/mlp_fused.cpp)

        def cb(index, dtype, page, pages, h_=32):
            desc = tile if h_ == 32 else ttnn.TileDescriptor(ttnn.Tile([h_, 32]))
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, desc)])

        cbs = [cb(0, bf, 2048, self.kt), cb(1, self.w13.dtype, self.tb13, self.slots * p.kb * p.w),
               cb(2, self.w2.dtype, self.tb2, self.slots2 * p2.kb * p2.w), cb(3, bf, th * 64, self.kt2, th),
               cb(16, bf, 2048, 2 * p.w), cb(17, bf, th * 64, 2 * p2.w, th)] + nm.cbs(residual)
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        common = [("kt", self.kt), ("w", p.w), ("kb", p.kb), ("kt2", self.kt2), ("w2", p2.w), ("kb2", p2.kb),
                  ("rows", rows), ("th", th)]
        np_dm = nm.named_args(rows, residual, cb_x=0)
        np_c = nm.named_args(rows, residual, cb_x=0, dm=False)
        accessors = list(ttnn.TensorAccessorArgs(f).get_compile_time_args()) + nm.accessor_args(a, b, h)
        per_core = [(c, list(args)) for c, args in self.core_args]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=cs,
            defines=[("MF_PROF", str(int(self.prof)))],
            ncrisc_named_compile_time_args=common + np_dm + [
                ("in1_tile_bytes", self.tb13), ("slots", self.slots), ("outstanding", p.outstanding),
                ("in2_tile_bytes", self.tb2), ("slots2", self.slots2), ("outstanding2", p2.outstanding),
                ("w2_gate", int(self.w2_gate)), ("sem_w13_done", _W2_SEMS[0]), ("sem_w2_go", _W2_SEMS[1])],
            brisc_named_compile_time_args=common + np_dm + [("sem_g_gather", _G_SEMS[0]),
                                                            ("sem_g_ready", _G_SEMS[1])],
            brisc_compile_time_args=accessors,
            trisc_named_compile_time_args=common + np_c + nm.compute_args() + [("mm_fid", self.mm_fid)],
            ncrisc_common_runtime_args=[int(self.w13.buffer_address()), int(self.w2.buffer_address())],
            brisc_common_runtime_args=[int(f.buffer_address())] + nm.runtime_args(a, b, h),
            trisc_common_runtime_args=[],
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=per_core, brisc_args=per_core, trisc_args=per_core),
        )
        sems = nm.semaphores() + [ttnn.SemaphoreDescriptor(id=i, core_ranges=cs, initial_value=0)
                                  for i in _G_SEMS + _W2_SEMS]
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=sems)

    def _check(self, t, name):
        if (len(t.shape) != 4 or tuple(t.shape)[:2] != (1, 1) or not 1 <= t.shape[2] <= 32
                or t.shape[3] != self.kt * 32 or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT
                or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            raise ValueError(f"{name} must be an interleaved BF16 TILE [1, 1, rows<=32, {self.kt * 32}] tensor")

    def __call__(self, a, b=None, output_memory_config=None, h_memory_config=None):
        self._check(a, "a")
        if b is not None:
            self._check(b, "b")
            if tuple(b.shape) != tuple(a.shape):
                raise ValueError("a and b must have the same shape")
        rows = int(a.shape[2])
        f = ttnn.empty([1, 1, rows, self.n2 * 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=output_memory_config or self.output_memory_config)
        if b is None:
            h, tensors = a, [a, self.w13, self.w2, f] + self.norm.tensors()
        else:
            h = ttnn.empty(list(a.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=h_memory_config or self.h_memory_config)
            tensors = [a, b, self.w13, self.w2, h, f] + self.norm.tensors()
        cache = self._cache.setdefault((b is None, rows), _CachedProgram())
        cache.run(tensors, lambda: self._build(a, b, h if b is not None else None, f, rows))
        return h, f
