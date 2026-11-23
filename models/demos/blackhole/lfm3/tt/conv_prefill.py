# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused prefill middle of the LFM3 gated short conv (kernels/conv_prefill.cpp): one launch between in_proj and
out_proj instead of the ttnn op chain (slices, B*x, row-major concat / 7 shifted slices / re-tilize, 7 FP32 tap
products + 6 adds + typecast, C*, 6 masked state-row writes).  Bitwise that chain's y and slot state
(tests/test_conv_prefill.py).  Work: the 64 x Tt (hidden column, row tile) units split evenly over every core of the
grid (each core one contiguous column-major range = at most two column segments).
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from .fast.conv_decode import CachedProgram, cb_descriptors, core_grid_rows, row_ranges, unified_kernels

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "conv_prefill.cpp")
DRAM = ttnn.DRAM_MEMORY_CONFIG
_IN_CBS = (0, 1, 2, 3, 4, 16)  # B, x, C, taps, shifted Bx, Bx: SFPU operands, unpacked to DEST as FP32
_CBS = {0: 2, 1: 2, 2: 3, 3: 7, 4: 12, 16: 2, 17: 2, 18: 2, 8: 8}  # CB index -> tiles
_COLS = 64  # hidden 2048 / 32


def _compute_config():
    """32-bit DEST (the FP32 tap sums), HiFi4, FP32 unpack of every SFPU operand."""
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, dst_full_sync_en=False)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for i in _IN_CBS:
        unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode.extend(unpack)
    return cfg


def segments(row_tiles: int, n_cores: int) -> list[list[tuple[int, int, int]]]:
    """Per core: its (column, r0, r1) segments of the column-major unit range [c U / n, (c + 1) U / n), U = 64 Tt."""
    units = _COLS * row_tiles
    out = []
    for c in range(n_cores):
        a, b = c * units // n_cores, (c + 1) * units // n_cores
        segs = []
        while a < b:
            j, r0 = divmod(a, row_tiles)
            r1 = min(row_tiles, r0 + (b - a))
            segs.append((j, r0, r1))
            a += r1 - r0
        if len(segs) > 2:
            raise ValueError("ConvPrefill: a core range spans more than two columns")
        out.append(segs)
    return out


class ConvPrefill:
    """y = C * sum_j taps[j] * Bx[t-6+j], Bx = B*x, over a prefill chunk of one sequence in batch slot `slot`, with that
    slot's conv state (6 rows) as left context (zeros when fresh), replaced by Bx[valid-6 .. valid-1] of
    [state, Bx].  Shares the ShortConv's state tensors (conv.s); taps re-laid out once ([1, 1, 32, 7H], rows
    replicated)."""

    def __init__(self, conv, n_cores: int | None = None):
        if conv.H != 2048 or conv.K != 7:
            raise ValueError("ConvPrefill: hidden 2048 and 7 taps")
        self.conv, self.H = conv, conv.H
        self.device = conv.s[0].device()
        grid = self.device.compute_with_storage_grid_size()
        self.n_cores = n_cores or grid.x * grid.y
        if conv.s[0].shape[-2] > 32:
            raise ValueError("ConvPrefill: at most 32 conv-state slots")
        k = torch.stack([ttnn.to_torch(t).reshape(self.H) for t in conv.taps])  # [7, H]: tap j for Bx[t-6+j]
        rep = k.reshape(1, 1, 1, conv.K * self.H).expand(1, 1, 32, conv.K * self.H).to(torch.bfloat16).contiguous()
        self.taps = ttnn.from_torch(rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                                    memory_config=DRAM)
        self._prog = CachedProgram()
        self._key = None

    def _build(self, bcx, y, valid, slot, fresh):
        row_tiles = bcx.shape[-2] // 32
        n = min(self.n_cores, _COLS * row_tiles)
        grid = self.device.compute_with_storage_grid_size()
        core_list = core_grid_rows(n, grid.x)
        cores = row_ranges(core_list)
        per_core = []
        for c, (core, segs) in enumerate(zip(core_list, segments(row_tiles, n))):
            args = [len(segs)] + [v for s in segs for v in s]
            per_core.append((core, args + [0] * (7 - len(args)) + [c]))
        table = [v for core in core_list for xy in [self.device.worker_core_from_logical_core(core)]
                 for v in (xy.x, xy.y)]
        kernels = unified_kernels(
            KERNEL, cores, [],
            reader_common=[int(t.buffer_address()) for t in (bcx, self.taps)],
            writer_common=[int(t.buffer_address()) for t in (*self.conv.s, y)] + [valid, slot, int(fresh), n] + table,
            per_core=per_core,
            compute=_compute_config())
        cbs = [d for i, k in _CBS.items() for d in cb_descriptors((i,), k, cores)]
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cores, initial_value=0) for s in (0, 1)]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)

    def __call__(self, bcx, valid: int, slot: int = 0, fresh: bool = True):
        """bcx: in_proj output [1, 1, Tp, 3*H] BF16 TILE DRAM (rows >= valid are padding) -> y [1, 1, Tp, H]."""
        Tp = bcx.shape[-2]
        if Tp % 32 or bcx.shape[-1] != 3 * self.H or not 1 <= valid <= Tp:
            raise ValueError(f"ConvPrefill: bad bcx {tuple(bcx.shape)} / valid {valid}")
        if not 0 <= slot < self.conv.batch:
            raise ValueError(f"ConvPrefill: slot {slot} outside the {self.conv.batch} conv-state slots")
        y = ttnn.empty([1, 1, Tp, self.H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=DRAM)
        key = (valid, slot, bool(fresh))
        if key != self._key:  # runtime arguments live in the cached descriptor
            self._prog = CachedProgram()
            self._key = key
        self._prog.run([bcx, self.taps, *self.conv.s, y], lambda: self._build(bcx, y, valid, slot, fresh))
        return y
