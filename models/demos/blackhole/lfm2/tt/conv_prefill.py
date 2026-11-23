# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused prefill middle of the LFM2 gated short conv (kernels/conv_prefill.cpp): one launch between in_proj and
out_proj instead of ShortConv.prefill's ~20 programs (slices, B*x, row-major concat / shifted slices / re-tilize, taps,
C*, the state row writes).  Bitwise ShortConv.prefill's y and slot state (tests/test_conv_prefill.py).
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from .fast.conv_decode import CachedProgram, cb_descriptors, core_grid_rows, eltwise_compute_config, row_ranges
from .fast.conv_decode import unified_kernels

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "conv_prefill.cpp")
DRAM = ttnn.DRAM_MEMORY_CONFIG
_FP32_UNPACK = (0, 1, 2, 7, 16, 20)  # the SFPU multiplies' operands (B, x, t2, C, Bx, acc3)
_CBS = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21)


class ConvPrefill:
    """y = C * (t0*Bx[t-2] + t1*Bx[t-1] + t2*Bx[t]), Bx = B*x, over a prefill chunk of one sequence in batch slot
    `slot`, with that slot's conv state as left context (zeros when fresh) and replaced by the chunk's rows
    valid-2, valid-1.  Shares the ShortConv's state tensors; taps are re-laid out once (rows replicated)."""

    def __init__(self, conv, n_cores: int = 64):
        if conv.H != 2048 or 64 % n_cores:
            raise ValueError("ConvPrefill: hidden 2048 and n_cores dividing 64")
        self.conv, self.H, self.n_cores = conv, conv.H, n_cores
        self.device = conv.s0.device()
        if conv.s0.shape[-2] > 32:
            raise ValueError("ConvPrefill: at most 32 conv-state slots")
        k = torch.stack([ttnn.to_torch(t).reshape(self.H) for t in conv.taps])  # [3, H]: taps for t-2, t-1, t
        rep = k.reshape(1, 1, 1, 3 * self.H).expand(1, 1, 32, 3 * self.H).to(torch.bfloat16).contiguous()
        self.taps = ttnn.from_torch(rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                                    memory_config=DRAM)
        self._prog = CachedProgram()
        self._key = None

    def _build(self, bcx, y, valid, slot, fresh):
        per = 64 // self.n_cores
        grid = self.device.compute_with_storage_grid_size()
        core_list = core_grid_rows(self.n_cores, grid.x)
        cores = row_ranges(core_list)
        defines = [("PER_CORE", str(per)), ("ROW_TILES", str(bcx.shape[-2] // 32))]
        s0, s1 = self.conv.s0, self.conv.s1
        kernels = unified_kernels(
            KERNEL, cores, defines,
            reader_common=[int(t.buffer_address()) for t in (bcx, self.taps)],
            writer_common=[int(t.buffer_address()) for t in (s0, s1, y)] + [valid, slot, int(fresh)],
            per_core=[(c, [i * per]) for i, c in enumerate(core_list)],
            compute=eltwise_compute_config(_FP32_UNPACK))
        cbs = cb_descriptors(_CBS, 2, cores) + cb_descriptors((8,), 3, cores)
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])

    def __call__(self, bcx, valid: int, slot: int = 0, fresh: bool = True):
        """bcx: in_proj output [1, 1, Tp, 3*H] BF16 TILE DRAM (rows >= valid are padding) -> y [1, 1, Tp, H]."""
        Tp = bcx.shape[-2]
        if Tp % 32 or bcx.shape[-1] != 3 * self.H or not 1 <= valid <= Tp:
            raise ValueError(f"ConvPrefill: bad bcx {tuple(bcx.shape)} / valid {valid}")
        y = ttnn.empty([1, 1, Tp, self.H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=DRAM)
        key = (valid, slot, bool(fresh))
        if key != self._key:  # runtime arguments live in the cached descriptor
            self._prog = CachedProgram()
            self._key = key
        self._prog.run([bcx, self.taps, self.conv.s0, self.conv.s1, y],
                       lambda: self._build(bcx, y, valid, slot, fresh))
        return y
