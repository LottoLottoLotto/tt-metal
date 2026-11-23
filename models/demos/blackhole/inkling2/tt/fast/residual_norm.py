# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""One-launch residual add + RMSNorm of decode rows (generic_op, trace-safe); copied from the LFM2.5 port
(lfm2/tt/fast/residual_norm.py, kernels/fast/residual_norm_*.cpp) for the drafter (tt/drafter.py), with grids for
non-power-of-two widths (Inkling's 2304 = 72 tiles: 24 cores x 3 tiles).

    rn = ResidualRMSNorm(device, w, eps)          # w: [W] gamma
    h, n = rn(a, b)                               # h = a + b (BF16, bitwise ttnn.add), n = rms_norm(h) * w (BF16)
    _, n = rn(a)                                  # norm only

Inputs a, b: [1, 1, T, W] BF16 TILE interleaved, 1 <= T <= 32 (rows normalised independently).  Layout: C cores (an
8 x C/8 rectangle at the grid origin), each owning W/32/C consecutive width tiles; per-core partial row sums of h^2
(diagonal of sum_i h_i h_i^T, HiFi4) gather on the master core (0, 0), which reduces them with the scaler 1/W (BF16,
round-to-nearest: exact for powers of two, 2.6e-4 relative for W = 2304), adds eps, takes rsqrt and multicasts the FP32
1/rms column; every core then writes (h * rstd) * gamma.
"""
from __future__ import annotations

import struct

import torch
import ttnn

from .common import KERNELS, CachedProgram

_GRIDS = {16: (8, 2), 24: (8, 3), 32: (8, 4), 64: (8, 8)}


def _accessor(tensor) -> list[int]:
    args = ttnn.TensorAccessorArgs(tensor)
    if args.get_common_runtime_args():
        raise ValueError("residual norm needs compile-time tensor accessors")
    return list(args.get_compile_time_args())


def _bf16_bits(x: float, rne: bool = False) -> int:
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    if rne:
        b += 0x7FFF + ((b >> 16) & 1)
    return (b >> 16) & 0xFFFF


class ResidualRMSNorm:
    def __init__(self, device, weight: torch.Tensor, eps: float, num_cores: int | None = None):
        self.device = device
        self.width = int(weight.numel())
        wt = self.width // 32
        if num_cores is None:
            num_cores = next((c for c in sorted(_GRIDS) if wt % c == 0 and wt // c <= 4), 0)
        if self.width % 32 or num_cores not in _GRIDS or wt % num_cores or wt // num_cores > 4:
            raise ValueError(f"unsupported width {self.width} / core count {num_cores}")
        self.eps = float(eps)
        self.tiles = wt // num_cores
        self.num_cores = num_cores
        g = weight.float().to(torch.bfloat16).reshape(1, 1, 1, -1).expand(1, 1, 32, -1).contiguous()
        self.gamma = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gx, gy = _GRIDS[num_cores]
        self.grid_x = gx
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
        defines = [] if residual else [("RN_NO_RESIDUAL", "1")]
        t, C = self.tiles, self.num_cores
        m_tiles = (C + 31) // 32
        scaler = _bf16_bits(1.0 / self.width, rne=True) * 0x10001
        reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "residual_norm_reader.cpp"), core_ranges=self.core_set,
            compile_time_args=[t, C, rows, 0, 1, self.grid_x, *self.master, scaler]
            + _accessor(a) + _accessor(b if residual else a) + _accessor(self.gamma),
            common_runtime_args=[int(a.buffer_address()), int(b.buffer_address()) if residual else 0,
                                 int(self.gamma.buffer_address())],
            defines=defines, config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "residual_norm_writer.cpp"), core_ranges=self.core_set,
            compile_time_args=[t, C, 1, self.grid_x, *self.mcast] + _accessor(h if residual else n) + _accessor(n),
            common_runtime_args=[int(h.buffer_address()) if residual else 0, int(n.buffer_address())],
            defines=defines, config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "residual_norm_compute.cpp"), core_ranges=self.core_set,
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
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.core_set, initial_value=0) for i in (0, 1)]
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs, semaphores=sems)

    def __call__(self, a, b=None):
        self._check(a, "a")
        if b is not None:
            self._check(b, "b")
            if tuple(b.shape) != tuple(a.shape):
                raise ValueError("a and b must have the same shape")
        shape = list(a.shape)
        dram = ttnn.DRAM_MEMORY_CONFIG
        n = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=dram)
        if b is None:
            tensors, h = [a, self.gamma, n], a
        else:
            h = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=dram)
            tensors = [a, b, self.gamma, h, n]
        key = (b is None, int(a.shape[2]))
        self._cache.setdefault(key, CachedProgram()).run(tensors, lambda: self._build(a, b, h if b is not None else None, n))
        return h, n
