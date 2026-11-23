# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused decode tail of the LFM2 gated short conv: one generic_op after in_proj.

Replaces, in ``layers.ShortConv.decode`` (everything between the two matmuls)::

    b, c, xx = slice(bcx) x3;  bx = mul(b, xx)
    acc = mul(bx, t2); acc = addcmul(acc, s1, t1); acc = addcmul(acc, s0, t0)
    copy(s1 -> s0); copy(bx -> s1); y = mul(c, acc)

with ``y = ConvDecode(conv)(bcx)``: same LLK operations, operand order and BF16 rounding points (bitwise identical
y and state), conv state updated in place (trace-safe; layout unchanged: ``[1, 1, B, 2048]`` BF16 TILE DRAM, so
``ShortConv.prefill`` / ``reset`` keep working on the same tensors).

    bcx = linear(x, conv.in_proj)      # [1, 1, B, 6144] BF16 TILE DRAM (B <= 32)
    y = fast(bcx)                      # [1, 1, B, 2048] BF16 TILE DRAM
    out = linear(y, conv.out_proj)

Kernel: kernels/conv_decode.cpp (reader/compute/writer in one source), one core per ``64 / n_cores`` hidden tile
columns.  Only tile rows [0, B) (rounded up to even, B <= 16) move through DRAM; y's padding rows are zero.
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "conv_decode.cpp")
TILE_BYTES = 2048
DRAM = ttnn.DRAM_MEMORY_CONFIG

# CB index -> UnpackToDestFp32 (operands of the SFPU multiplies, like binary_ng's BF16 mul)
_FP32_UNPACK = (0, 1, 2, 7, 16, 20)
_CBS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21)


class CachedProgram:
    """generic_op program rebuilt only when a tensor's address / shape / memory config changes (trace-safe)."""

    def __init__(self):
        self.program, self.signature = None, None

    def run(self, tensors, build):
        sig = tuple((int(t.buffer_address()), tuple(t.padded_shape), str(t.memory_config())) for t in tensors)
        if self.program is None or sig != self.signature:
            self.program, self.signature = build(), sig
        return ttnn.generic_op(tensors, self.program)


def core_grid_rows(n: int, width: int) -> list:
    return [ttnn.CoreCoord(i % width, i // width) for i in range(n)]


def row_ranges(cores) -> ttnn.CoreRangeSet:
    """Cores as row-contiguous CoreRanges (dispatch cost grows with the range count)."""
    ranges = []
    for c in sorted(cores, key=lambda c: (c.y, c.x)):
        if ranges and ranges[-1][1] == c.y and ranges[-1][2] + 1 == c.x:
            ranges[-1][2] = c.x
        else:
            ranges.append([c.x, c.y, c.x])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y), ttnn.CoreCoord(x1, y)) for x0, y, x1 in ranges])


def unified_kernels(source, cores, defines, reader_common, writer_common, per_core, compute):
    """Reader (NCRISC, NOC0) / writer (BRISC, NOC1) / compute descriptors of one single-source kernel."""
    def rt(args):
        r = ttnn.RuntimeArgs()
        for core, a in args:
            r[core.x][core.y] = list(a)
        return r

    dm = dict(kernel_source=source, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=cores,
              defines=defines)
    return [
        ttnn.KernelDescriptor(**dm, common_runtime_args=reader_common, runtime_args=rt(per_core),
                              config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                                       noc=ttnn.NOC.RISCV_0_default)),
        ttnn.KernelDescriptor(**dm, common_runtime_args=writer_common, runtime_args=rt(per_core),
                              config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                       noc=ttnn.NOC.RISCV_1_default)),
        ttnn.KernelDescriptor(**dm, runtime_args=rt(per_core), config=compute),
    ]


def cb_descriptors(indices, tiles, cores, dtype=ttnn.bfloat16, page=TILE_BYTES):
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    return [ttnn.CBDescriptor(total_size=tiles * page, core_ranges=cores,
                              format_descriptors=[ttnn.CBFormatDescriptor(i, dtype, page, tile)]) for i in indices]


def eltwise_compute_config(fp32_unpack_cbs):
    """binary_ng / ternary BF16 config: HiFi4, 16-bit DEST, UnpackToDestFp32 on the SFPU operands' CBs."""
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=False, dst_full_sync_en=False)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for i in fp32_unpack_cbs:
        unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode.extend(unpack)
    return cfg


class ConvDecode:
    """y = C * (t0*s0 + t1*s1 + t2*(B*x)) and the state shift, from the in_proj output, for one ShortConv.

    Constructed once per conv layer from the existing ``layers.ShortConv`` (reads its taps back once; shares its
    s0/s1 state tensors).  ``__call__(bcx)`` is trace-safe (device-only; the program is cached per buffer
    addresses).  rows: the decode rows (default: the state's rows); fewer run a decode-width bucket on state rows
    [0, rows), leaving the other slots' rows unchanged.  taps: share another instance's replicated taps."""

    def __init__(self, conv, n_cores: int = 64, rows: int | None = None, taps=None):
        H = conv.H
        if H != 2048 or 64 % n_cores:
            raise ValueError("ConvDecode: hidden 2048 and n_cores dividing 64")
        self.conv, self.H, self.n_cores = conv, H, n_cores
        self.device = conv.s0.device()
        state_rows = conv.s0.shape[-2]
        rows = state_rows if rows is None else rows
        if rows > 32 or not 1 <= rows <= state_rows:
            raise ValueError(f"ConvDecode: {rows} decode rows for a {state_rows}-row state (<= 32)")
        self.rows = rows
        # odd rows < 16 are moved as rows + 1: keep that row of a wider state unchanged (kernel KEEP_ROW)
        self.keep_row = rows < state_rows and rows < 16 and rows % 2 == 1
        if taps is None:
            k = torch.stack([ttnn.to_torch(t).reshape(H) for t in conv.taps])  # [3, H]: taps for t-2, t-1, t
            rep = k.reshape(1, 1, 1, 3 * H).expand(1, 1, 32, 3 * H).to(torch.bfloat16).contiguous()
            taps = ttnn.from_torch(rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                                   memory_config=DRAM)
        self.taps = taps
        self._prog = CachedProgram()

    def _build(self, bcx, y):
        per = 64 // self.n_cores
        grid = self.device.compute_with_storage_grid_size()
        core_list = core_grid_rows(self.n_cores, grid.x)
        cores = row_ranges(core_list)
        rows = 32 if self.rows > 16 else self.rows + (self.rows & 1)
        cb_tiles = 2 if per > 1 else 1
        defines = [("ROWS", str(rows)), ("PER_CORE", str(per)), ("CB_TILES", str(cb_tiles))]
        if self.keep_row:
            if per != 1:
                raise ValueError("ConvDecode: odd bucket rows need one column per core")
            defines.append(("KEEP_ROW", str(self.rows)))
        s0, s1 = self.conv.s0, self.conv.s1
        kernels = unified_kernels(
            KERNEL, cores, defines,
            reader_common=[int(t.buffer_address()) for t in (bcx, self.taps, s0, s1)],
            writer_common=[int(t.buffer_address()) for t in (s0, s1, y)],
            per_core=[(c, [i * per]) for i, c in enumerate(core_list)],
            compute=eltwise_compute_config(_FP32_UNPACK))
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cb_descriptors(_CBS, cb_tiles, cores), semaphores=[])

    def __call__(self, bcx):
        """bcx: in_proj output [1, 1, B, 3*H] BF16 TILE DRAM -> y [1, 1, B, H] BF16 TILE DRAM; s0/s1 shifted."""
        shape = tuple(bcx.shape)
        if (shape != (1, 1, self.rows, 3 * self.H) or bcx.dtype != ttnn.bfloat16 or bcx.layout != ttnn.TILE_LAYOUT
                or bcx.memory_config() != DRAM):
            raise ValueError(f"ConvDecode: bcx must be [1,1,{self.rows},{3 * self.H}] BF16 TILE DRAM, got {shape}")
        y = ttnn.empty([1, 1, self.rows, self.H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=DRAM)
        self._prog.run([bcx, self.taps, self.conv.s0, self.conv.s1, y], lambda: self._build(bcx, y))
        return y
