# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""One-launch residual add + RMSNorm for LFM2 decode rows (generic_op, trace-safe).

    h = a + b                                   (BF16, bitwise ttnn.add)
    n = h / sqrt(mean(h^2, -1) + eps) * w       (LFM2 RMSNorm: weight w, no 1 + w)

Replaces in model.Layer.decode the pairs
    h = ttnn.add(x, o);  n2 = self.ffn_norm(h)                 -> h, n2 = ffn_rn(x, o)
    out = ttnn.add(h, f); n = next_layer.op_norm(out)           -> out, n = next_rn(h, f)
and (b=None) a plain norm, e.g. layer 0's op_norm(embedding) -> _, n = rn(x).

Inputs a, b: [1, 1, T, W] BF16 TILE, interleaved (DRAM or L1), 1 <= T <= 32 (one tile row; rows are normalised
independently; the global reduction only gathers rows < T, padded rows r >= T get rstd = 1/sqrt(eps)).
Outputs: h (BF16 TILE, interleaved DRAM or L1) and n (BF16 TILE, interleaved DRAM / L1, or any L1/DRAM sharded
layout TensorAccessor handles with compile-time args, e.g. WIDTH_SHARDED [32, W/k] on k cores).

Layout: C cores (C in {16, 32, 64}: an 8 x C/8 rectangle at the worker-grid origin), each owning W/32/C consecutive
width tiles. Each core reads its a/b/gamma tiles, packs h (written out at once) and its partial row sums of h^2
(diagonal of sum_i h_i h_i^T, one HiFi4 matmul pass); the partials (FP32, T values per core) go to the master
core (0, 0) as inline NOC writes + a semaphore; the master reduces them (scaler 1/W), adds eps and takes rsqrt on
the SFPU (column faces only) and multicasts the FP32 1/rms column tile; every core then computes
(h * rstd[col bcast]) * gamma in DEST and writes its n tiles. One generic_op, no per-core runtime args.

Numerics: CK_NORM-equivalent (HiFi4, FP32 DEST, BF16-truncated eps, non-legacy rsqrt, rstd and h*rstd entering
the FPU as TF32 like the native kernel). h is bitwise ttnn.add. n vs ttnn.rms_norm: bitwise for most rows; rows
whose 1/rms lands on a TF32 rounding boundary differ by 1 BF16 ulp in some elements (different summation order).
"""
from __future__ import annotations

import struct
from pathlib import Path

import torch
import ttnn

_KERNELS = Path(__file__).resolve().parent / "kernels"
_TILE = 2048
_GRIDS = {16: (8, 2), 32: (8, 4), 64: (8, 8)}


def _accessor(tensor) -> list[int]:
    args = ttnn.TensorAccessorArgs(tensor)
    if args.get_common_runtime_args():
        raise ValueError("residual norm needs compile-time tensor accessors")
    return list(args.get_compile_time_args())


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def gamma_tile(device, w: torch.Tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """[W] gamma -> [1, 1, 32, W] BF16 TILE with every row = w (row-broadcast or elementwise operand)."""
    g = w.to(torch.bfloat16).reshape(1, 1, 1, -1).expand(1, 1, 32, -1).contiguous()
    return ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _bf16_bits(x: float) -> int:
    """float -> bf16 bit pattern (truncation, as the native kernels build their constant tiles)."""
    return _f32_bits(x) >> 16


class _CachedProgram:
    """Rebuild the program descriptor only when a tensor's address / shape / memory config changes."""

    def __init__(self):
        self.program = None
        self.signature = None

    def run(self, tensors, build):
        sig = tuple((int(t.buffer_address()), tuple(t.padded_shape), tuple(t.shape), str(t.memory_config()))
                    for t in tensors)
        if self.program is None or sig != self.signature:
            self.program = build()
            self.signature = sig
        ttnn.generic_op(list(tensors), self.program)


class ResidualRMSNorm:
    """Fused residual add + RMSNorm (one launch). Construct once per norm weight (e.g. per layer ffn_norm, per
    layer op_norm, final norm).

    __call__(a, b=None, n_memory_config=None, h_memory_config=None) -> (h, n)
      a, b: [1, 1, T, W] BF16 TILE interleaved; b=None -> h is a itself (norm only, no add, no h write).
      h: new tensor (h_memory_config, default DRAM interleaved), bitwise ttnn.add(a, b).
      n: new tensor (n_memory_config, default DRAM interleaved), rms_norm(h) * w.
    """

    def __init__(self, device, weight: torch.Tensor, eps: float, num_cores: int = 32,
                 h_memory_config=ttnn.DRAM_MEMORY_CONFIG, n_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                 profile: bool = False):
        self.device = device
        self.width = int(weight.numel())
        wt = self.width // 32
        if (self.width % 32 or self.width & (self.width - 1) or num_cores not in _GRIDS or wt % num_cores
                or wt // num_cores > 4):
            # power-of-two width: the 1/W reduce scaler is exact in BF16; <= 4 tiles per core: one DEST block
            raise ValueError(f"unsupported width {self.width} / core count {num_cores}")
        self.eps = float(eps)
        self.tiles = wt // num_cores
        self.num_cores = num_cores
        self.h_memory_config = h_memory_config
        self.n_memory_config = n_memory_config
        self.gamma = gamma_tile(device, weight.float())
        gx, gy = _GRIDS[num_cores]
        grid = device.compute_with_storage_grid_size()
        if gx > grid.x or gy > grid.y:
            raise ValueError("core grid too small")
        # Debug only: per-core phase timestamps (kernels/residual_norm_prof.hpp), see read_profile().
        self.prof = (ttnn.from_torch(torch.zeros(1, 1, 2 * num_cores, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                     if profile else None)
        self.core_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        p0 = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        p1 = device.worker_core_from_logical_core(ttnn.CoreCoord(gx - 1, gy - 1))
        self.master = (p0.x, p0.y)
        self.mcast = (p0.x, p0.y, p1.x, p1.y)
        self._cache = {}

    def _check(self, t, name):
        if (len(t.shape) != 4 or tuple(t.shape)[:2] != (1, 1) or not 1 <= t.shape[2] <= 32
                or t.shape[3] != self.width or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT
                or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            raise ValueError(f"{name} must be an interleaved BF16 TILE [1, 1, T<=32, {self.width}] tensor")

    def _build(self, a, b, h, n):
        rows = int(a.shape[2])
        residual = b is not None
        prof = self.prof is not None
        defines = ([] if residual else [("RN_NO_RESIDUAL", "1")]) + ([("RN_PROFILE", "1")] if prof else [])
        prof_args = _accessor(self.prof) if prof else []
        prof_rt = [int(self.prof.buffer_address())] if prof else []
        t, C = self.tiles, self.num_cores
        m_tiles = (C + 31) // 32  # gathered partial tiles (column k = core k)
        gx = _GRIDS[C][0]
        reader = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "residual_norm_reader.cpp"), core_ranges=self.core_set,
            compile_time_args=[t, C, rows, 0, 1, gx, *self.master, _bf16_bits(1.0 / self.width) * 0x10001]
            + _accessor(a) + _accessor(b if residual else a) + _accessor(self.gamma) + prof_args,
            common_runtime_args=[int(a.buffer_address()), int(b.buffer_address()) if residual else 0,
                                 int(self.gamma.buffer_address())] + prof_rt,
            defines=defines, config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "residual_norm_writer.cpp"), core_ranges=self.core_set,
            compile_time_args=[t, C, 1, gx, *self.mcast] + _accessor(h if residual else n) + _accessor(n) + prof_args,
            common_runtime_args=[int(h.buffer_address()) if residual else 0, int(n.buffer_address())] + prof_rt,
            defines=defines, config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "residual_norm_compute.cpp"), core_ranges=self.core_set,
            # eps as the native kernel sees it: a BF16 (truncated) constant
            compile_time_args=[t, m_tiles, _bf16_bits(self.eps) << 16, rows],
            defines=defines,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                math_approx_mode=False),
        )
        tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))

        def cb(index, dtype, tiles):
            size = 4096 if dtype == ttnn.float32 else 2048
            return ttnn.CBDescriptor(total_size=size * tiles, core_ranges=self.core_set,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, size, tile)])

        bf, f32 = ttnn.bfloat16, ttnn.float32
        cbs = [cb(0, bf, t), cb(1, bf, t), cb(2, bf, t), cb(3, bf, 1), cb(16, bf, t), cb(17, bf, t),
               cb(25, f32, 1), cb(26, f32, m_tiles), cb(28, f32, 1)]
        if prof:
            cbs.append(cb(30, bf, 1))
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.core_set, initial_value=0) for i in (0, 1)]
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs, semaphores=sems)

    def __call__(self, a, b=None, n_memory_config=None, h_memory_config=None):
        self._check(a, "a")
        if b is not None:
            self._check(b, "b")
            if tuple(b.shape) != tuple(a.shape):
                raise ValueError("a and b must have the same shape")
        shape = list(a.shape)
        n = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=n_memory_config or self.n_memory_config)
        if b is None:
            tensors, h = [a, self.gamma, n], a
        else:
            h = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=h_memory_config or self.h_memory_config)
            tensors = [a, b, self.gamma, h, n]
        if self.prof is not None:
            tensors.append(self.prof)
        key = (b is None, int(a.shape[2]))
        cache = self._cache.setdefault(key, _CachedProgram())
        cache.run(tensors, lambda: self._build(a, b, h if b is not None else None, n))
        return h, n

    def read_profile(self) -> torch.Tensor:
        """[cores, 2, 16] int64 wall-clock samples of the last launch (row 0: reader RISC, row 1: writer RISC)."""
        return ttnn.to_torch(self.prof).reshape(self.num_cores, 2, 16).to(torch.int64) & 0xFFFFFFFF
