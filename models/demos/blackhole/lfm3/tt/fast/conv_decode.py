# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused decode tail of the LFM3 gated short conv (7 taps, 6 state rows): one generic_op after in_proj.

Replaces, in ``layers.ShortConv.decode`` (everything between the two matmuls)::

    b, c, xx = slice(bcx) x3;  bx = mul(b, xx)
    conv = _taps_fp32([s0, ..., s5, bx], [t0, ..., t6])     # FP32 products / sums, rounded once to BF16
    copy(s_{j+1} -> s_j) (j < 5); copy(bx -> s5); y = mul(c, conv)

with ``y = ConvDecode(conv)(bcx)``: the same SFPU operations in the same order (kernels/conv_tail.hpp), so y and the
six state tensors are bitwise identical; the state is updated in place (trace-safe; layout unchanged: six
``[1, 1, B, 2048]`` BF16 TILE DRAM tensors, so ``ShortConv.prefill`` / ``reset`` / save / restore keep working).

    bcx = linear(x, conv.in_proj)      # [1, 1, rows, 6144] BF16 TILE interleaved (DRAM or L1)
    y = fast(bcx)                      # [1, 1, rows, 2048] BF16 TILE interleaved (DRAM default)
    out = linear(y, conv.out_proj)

Kernel: kernels/conv_decode.cpp (reader/compute/writer in one source), ``64 / n_cores`` hidden tile columns per core.
Only tile rows [0, rows) (64 B granules) move; y's padding rows are undefined (as ttnn padding).
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "conv_decode.cpp")
TILE_BYTES = 2048
DRAM = ttnn.DRAM_MEMORY_CONFIG
TAPS = 7  # conv_L_cache: 6 state rows

# kernels/conv_decode.cpp CB ids: B, x, C, t0..t5, t6, s0..s5, FP32 state partial (all UnpackToDestFp32); verify
# scratch (state rows as read + row map); y, Bx
_CB_B, _CB_X, _CB_C, _CB_T, _CB_T6, _CB_S, _CB_ACC, _CB_RAW, _CB_Y, _CB_BX = 0, 1, 2, 3, 4, 5, 6, 7, 16, 17
_FP32_UNPACK = (_CB_B, _CB_X, _CB_C, _CB_T, _CB_T6, _CB_S, _CB_ACC)


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


def eltwise_compute_config(fp32_unpack_cbs, fp32_dest: bool = False):
    """binary_ng / ternary config: HiFi4, 16-bit DEST (fp32_dest: FP32 DEST), UnpackToDestFp32 on the given CBs."""
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=fp32_dest, dst_full_sync_en=False)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for i in fp32_unpack_cbs:
        unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode.extend(unpack)
    return cfg


def replicated_taps(conv):
    """The conv taps as one [1, 1, 32, 7 * H] BF16 TILE DRAM tensor: tap k of hidden column c at column k * H + c,
    replicated over the 32 tile rows (the decode kernels' per-row operand).  One per layer; share it between the
    ConvDecode / ConvOpDecode instances of the decode widths (``taps=``)."""
    H = conv.H
    k = torch.stack([ttnn.to_torch(t).reshape(-1)[:H] for t in conv.taps])  # [7, H]: tap k multiplies Bx[t-6+k]
    rep = k.reshape(1, 1, 1, TAPS * H).expand(1, 1, 32, TAPS * H).to(torch.bfloat16).contiguous()
    return ttnn.from_torch(rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=conv.s[0].device(),
                           memory_config=DRAM)


def check_conv(conv, rows, owner: str, verify=None, vmap=None) -> tuple[int, bool]:
    """(rows, keep_row) of a decode-width module on `conv` (layers.ShortConv: hidden 2048, 7 taps).  verify: the
    speculative-verification candidate tensor ([1, 1, 32, 2048] BF16 TILE DRAM) or None; vmap: the verify row map
    ([1, 16] uint32 ROW_MAJOR DRAM, tt/speculate.py map_word).  In verify mode the rows are groups of consecutive
    tokens, one per sequence, each reading its group's conv-state slot (any 1..16 rows, the state untouched)."""
    if conv.H != 2048 or conv.K != TAPS or len(conv.s) != TAPS - 1:
        raise ValueError(f"{owner}: hidden 2048 and a 7-tap conv (6 state tensors) required")
    state_rows = conv.s[0].shape[-2]
    rows = state_rows if rows is None else rows
    if verify is not None:
        if (not 1 <= rows <= 16 or tuple(verify.padded_shape) != (1, 1, 32, conv.H)
                or verify.dtype != ttnn.bfloat16 or verify.layout != ttnn.TILE_LAYOUT or verify.memory_config() != DRAM):
            raise ValueError(f"{owner}: verify needs 1..16 rows and a [1, 1, 32, {conv.H}] BF16 TILE DRAM candidate")
        if (vmap is None or tuple(vmap.shape) != (1, 16) or vmap.dtype != ttnn.uint32
                or vmap.layout != ttnn.ROW_MAJOR_LAYOUT or vmap.memory_config() != DRAM):
            raise ValueError(f"{owner}: verify needs the [1, 16] uint32 ROW_MAJOR DRAM row map (vmap)")
        return rows, False
    if not 1 <= rows <= min(state_rows, 32):
        raise ValueError(f"{owner}: {rows} decode rows for a {state_rows}-row conv state (<= 32)")
    # an odd row count moves rows + 1 (64 B granules): with a wider state, that row belongs to another slot and is
    # written back unchanged (kernel KEEP / keep_row)
    return rows, rows % 2 == 1 and rows < state_rows


class ConvDecode:
    """y = C * conv7(s0..s5, B*x) and the state shift, from the in_proj output, for one ShortConv.

    Constructed once per conv layer and decode width from the existing ``layers.ShortConv`` (shares its six state
    tensors).  ``__call__(bcx)`` is trace-safe (device-only; the program is cached per buffer addresses).  rows: the
    decode rows (default: the state's rows); fewer run a decode-width bucket on state rows [0, rows), leaving the
    other slots' rows unchanged.  taps: ``replicated_taps(conv)`` to share (default: built here).  verify: a
    candidate tensor [1, 1, 32, 2048] BF16 TILE DRAM -> speculative-verification mode (kernels/conv_decode.cpp VERIFY,
    with the row map `vmap`): the rows are groups of consecutive tokens, one per sequence (row r's left context = its
    group's conv-state slot and the group's rows before r), every row's Bx goes to `verify`, the state is left
    unchanged."""

    def __init__(self, conv, n_cores: int = 64, rows: int | None = None, taps=None, verify=None, vmap=None):
        H = conv.H
        self.rows, self.keep_row = check_conv(conv, rows, "ConvDecode", verify, vmap)
        self.cand, self.vmap = verify, vmap
        if 64 % n_cores:
            raise ValueError("ConvDecode: n_cores must divide 64")
        self.conv, self.H, self.n_cores = conv, H, n_cores
        self.device = conv.s[0].device()
        self.taps = replicated_taps(conv) if taps is None else taps
        self._prog = CachedProgram()

    def _build(self, bcx, y):
        per = 64 // self.n_cores
        grid = self.device.compute_with_storage_grid_size()
        core_list = core_grid_rows(self.n_cores, grid.x)
        cores = row_ranges(core_list)
        dram = ttnn.BufferType.DRAM
        defines = [("ROWS", str(self.rows)), ("PER_CORE", str(per)), ("KEEP", str(int(self.keep_row))),
                   ("BCX_DRAM", str(int(bcx.memory_config().buffer_type == dram))),
                   ("Y_DRAM", str(int(y.memory_config().buffer_type == dram))),
                   ("VERIFY", str(int(self.cand is not None)))]
        s = [int(t.buffer_address()) for t in self.conv.s]
        verify = self.cand is not None
        cand = [int(self.cand.buffer_address())] if verify else []
        vmap = [int(self.vmap.buffer_address())] if verify else []
        kernels = unified_kernels(
            KERNEL, cores, defines,
            reader_common=[int(bcx.buffer_address()), int(self.taps.buffer_address())] + s + vmap,
            writer_common=[int(y.buffer_address())] + s + cand,
            per_core=[(c, [i * per]) for i, c in enumerate(core_list)],
            compute=eltwise_compute_config(_FP32_UNPACK, fp32_dest=True))
        n = TAPS - 1
        cbs = (cb_descriptors((_CB_B, _CB_X, _CB_C, _CB_T6, _CB_Y, _CB_BX), per, cores)
               + cb_descriptors((_CB_T, _CB_S), n * per, cores)
               + cb_descriptors((_CB_ACC,), per, cores, dtype=ttnn.float32, page=2 * TILE_BYTES))
        if verify:  # the state rows as read + the row map (reader scratch)
            cbs += cb_descriptors((_CB_RAW,), n * per + 1, cores)
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])

    def __call__(self, bcx, output_memory_config=DRAM):
        """bcx: in_proj output [1, 1, rows, 3*H] BF16 TILE interleaved -> y [1, 1, rows, H] BF16 TILE interleaved;
        state rows [0, rows) shifted in place (verify mode: unchanged; every row's Bx -> the candidate tensor)."""
        shape = tuple(bcx.shape)
        if (shape != (1, 1, self.rows, 3 * self.H) or bcx.dtype != ttnn.bfloat16 or bcx.layout != ttnn.TILE_LAYOUT
                or bcx.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            raise ValueError(f"ConvDecode: bcx must be [1,1,{self.rows},{3 * self.H}] BF16 TILE interleaved, "
                             f"got {shape}")
        y = ttnn.empty([1, 1, self.rows, self.H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=output_memory_config)
        extra = [self.cand, self.vmap] if self.cand is not None else []
        self._prog.run([bcx, self.taps, *self.conv.s, y] + extra, lambda: self._build(bcx, y))
        return y
