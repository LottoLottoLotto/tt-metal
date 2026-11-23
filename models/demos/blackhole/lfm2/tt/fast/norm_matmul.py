# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Residual add + RMSNorm fused into the following decode op (one program): `NormPrologue` (host side of
kernels/norm_prologue.hpp, reusable by any unified decode kernel on a gx x gy rectangle) and
`NormStreamDecodeMatmul` = ResidualRMSNorm + StreamDecodeMatmul in one launch, bitwise.

    h, y = NormStreamDecodeMatmul(...)(a, b)      # h = a + b (bitwise ttnn.add), y = rms_norm(h) * w @ W
    _, y = op(a)                                  # norm only (b = None): y = rms_norm(a) * w @ W

replaces `h, n = rn(a, b); y = mm(n)` of the round-1 chain.  The weight starts streaming at kernel start (NCRISC),
while BRISC/TRISC run the norm on the same 32 cores; the matmul then consumes n from L1 (no n tensor, no second
launch).  Measured in tests/fast/test_norm_matmul.py.
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from .decode_matmul import TILE_BYTES, Preset, _CachedProgram, stream_columns
from .residual_norm import _accessor, _bf16_bits, gamma_tile

_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "decode_matmul_norm.cpp")
_BANKS = 8
_FID = {ttnn.MathFidelity.LoFi: 0, ttnn.MathFidelity.HiFi2: 2, ttnn.MathFidelity.HiFi3: 3,
        ttnn.MathFidelity.HiFi4: 4}

# CB / semaphore ids of the prologue (defaults leave 0-11, 16-23 and semaphores 0-1 to the host kernel)
PROLOGUE_CBS = {"a": 12, "b": 13, "g": 14, "scaler": 15, "h": 24, "n": 25, "part": 26, "m": 27, "rstd": 28}
PROLOGUE_SEMS = (2, 3, 4, 5)  # partial gather, rstd, n gather, n ready


class NormPrologue:
    """Host side of norm_prologue.hpp for one norm weight on the rectangle origin..origin+(gx-1, gy-1)
    (gx * gy cores, each owning W/32/(gx*gy) width tiles, <= 4).  Bitwise ResidualRMSNorm(device, weight, eps)
    with the same core count (32 -> its 8 x 4 layout).

    A host kernel includes the header, calls np_dm<ACC, RT>() on its non-streaming data-movement RISC and
    np_compute() first on TRISC (compute config HiFi4, FP32 DEST), and adds to its program:
    named_args(...) to every RISC's named compile-time args (+ compute_args() on TRISC), accessor_args(...) at
    positional offset ACC of the DM RISC, runtime_args(...) at common-runtime offset RT of the DM RISC, cbs() and
    semaphores()."""

    def __init__(self, device, weight: torch.Tensor, eps: float, origin=(0, 0), gx: int = 8, gy: int = 4,
                 cbs: dict | None = None, sems=PROLOGUE_SEMS):
        self.device = device
        self.width = int(weight.numel())
        self.cores = gx * gy
        wt = self.width // 32
        if self.width % 32 or self.width & (self.width - 1) or wt % self.cores or wt // self.cores > 4:
            raise ValueError(f"unsupported width {self.width} for {self.cores} cores")
        self.tiles = wt // self.cores
        self.kt = wt
        self.eps = float(eps)
        self.gamma = gamma_tile(device, weight.float())
        self.origin, self.gx, self.gy = origin, gx, gy
        grid = device.compute_with_storage_grid_size()
        if origin[0] + gx > grid.x or origin[1] + gy > grid.y:
            raise ValueError("prologue rectangle outside the worker grid")
        c0 = ttnn.CoreCoord(*origin)
        c1 = ttnn.CoreCoord(origin[0] + gx - 1, origin[1] + gy - 1)
        self.core_set = ttnn.CoreRangeSet({ttnn.CoreRange(c0, c1)})
        p0 = device.worker_core_from_logical_core(c0)
        p1 = device.worker_core_from_logical_core(c1)
        self.master = (p0.x, p0.y)
        self.mcast = (p0.x, p0.y, p1.x, p1.y)
        self.cb_ids = dict(PROLOGUE_CBS if cbs is None else cbs)
        self.sem_ids = tuple(sems)

    def core_index(self, core) -> int:
        return (core.y - self.origin[1]) * self.gx + (core.x - self.origin[0])

    def named_args(self, rows: int, residual: bool, cb_x: int, write_n: bool = False, dm: bool = True,
                   embed: bool = False) -> list:
        c = self.cb_ids
        args = [("np_tiles", self.tiles), ("np_cores", self.cores), ("np_rows", rows), ("np_gx", self.gx),
                ("np_x0", self.origin[0]), ("np_y0", self.origin[1]), ("np_residual", int(residual)),
                ("np_embed", int(embed)),
                ("np_kt", self.kt), ("np_cb_x", cb_x)] + [(f"np_cb_{k}", v) for k, v in c.items()]
        if dm:
            args += [("np_master_x", self.master[0]), ("np_master_y", self.master[1]),
                     ("np_mc_x0", self.mcast[0]), ("np_mc_y0", self.mcast[1]), ("np_mc_x1", self.mcast[2]),
                     ("np_mc_y1", self.mcast[3]), ("np_scaler", _bf16_bits(1.0 / self.width) * 0x10001),
                     ("np_sem_gather", self.sem_ids[0]), ("np_sem_rstd", self.sem_ids[1]),
                     ("np_sem_xg", self.sem_ids[2]), ("np_sem_x", self.sem_ids[3]),
                     ("np_write_n", int(write_n))]
        return args

    def compute_args(self) -> list:
        return [("np_eps", _bf16_bits(self.eps) << 16)]

    def accessor_args(self, a, b=None, h=None, n=None) -> list:
        def pick(t):
            return _accessor(t if t is not None else a)

        return _accessor(a) + pick(b) + _accessor(self.gamma) + pick(h) + pick(n)

    def runtime_args(self, a, b=None, h=None, n=None) -> list:
        return [int(a.buffer_address()), int(b.buffer_address()) if b is not None else 0,
                int(self.gamma.buffer_address()), int(h.buffer_address()) if h is not None else 0,
                int(n.buffer_address()) if n is not None else 0]

    def cbs(self, residual: bool, embed: bool = False) -> list:
        tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
        cs, c, t = self.core_set, self.cb_ids, self.tiles
        m_tiles = (self.cores + 31) // 32

        def cb(index, dtype, pages, page=None):
            page = page or (4096 if dtype == ttnn.float32 else 2048)
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        bf, f32 = ttnn.bfloat16, ttnn.float32
        out = [cb(c["g"], bf, t), cb(c["scaler"], bf, 1), cb(c["h"], bf, t), cb(c["n"], bf, t),
               cb(c["part"], f32, 1), cb(c["m"], f32, m_tiles), cb(c["rstd"], f32, 1)]
        if residual:
            out += [cb(c["a"], bf, t), cb(c["b"], bf, t)]
        if embed:
            out.append(cb(c["b"], bf, 2))  # token ids + table chunks scratch
        return out

    def semaphores(self) -> list:
        return [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.core_set, initial_value=0) for i in self.sem_ids]


def rect_bank_cores(device, prologue: NormPrologue, cores_per_bank: int):
    """cores[b][c] of the prologue rectangle (gx * gy == banks * cores_per_bank): per round, each bank takes the
    free core nearest its preferred worker (as decode_matmul.stream_cores, on this rectangle)."""
    anchors = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    if len(anchors) != _BANKS or prologue.cores != _BANKS * cores_per_bank:
        raise ValueError(f"rectangle of {prologue.cores} cores != {_BANKS} banks x {cores_per_bank}")
    free = {(c.x, c.y) for c in ttnn.corerange_to_cores(prologue.core_set, row_wise=True)}
    cores = [[] for _ in range(_BANKS)]
    for _ in range(cores_per_bank):
        for b, a in enumerate(anchors):
            best = min(free, key=lambda c: (abs(c[0] - a.x) + abs(c[1] - a.y), c[1], c[0]))
            free.remove(best)
            cores[b].append(ttnn.CoreCoord(*best))
    return cores


class NormStreamDecodeMatmul:
    """(h, y) = (a + b, rms_norm(a + b) * w @ W) in one generic_op on an 8 x 4 rectangle at `origin`.

    weight / preset / compute_kernel_config / out_dtype / output_memory_config: as StreamDecodeMatmul (stream-layout
    weight from cached_stream_weight; presets with cores_per_bank 4 and K = 2048, k_split 0).  norm_weight, eps: as
    ResidualRMSNorm.  slots: weight ring depth in kb x w blocks (default preset.slots + 8, <= ~400 KB: covers the
    ~6 us prologue).  origin: the 8 x 4 rectangle's first core; (0, 6) streams ~3.5 us faster than (0, 0) on the
    P150 (DRAM-bank anchors sit in columns 0 and 6; measured conv_in / attn_qkv).  a, b: [1, 1, rows<=32, 2048] BF16 TILE interleaved (DRAM or L1).  h: new tensor
    (h_memory_config, default L1); b=None: no add, h is a itself.  y: as StreamDecodeMatmul's output."""

    def __init__(self, device, norm_weight: torch.Tensor, eps: float, weight, preset: Preset, compute_kernel_config,
                 out_dtype=ttnn.bfloat16, output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                 h_memory_config=ttnn.L1_MEMORY_CONFIG, slots: int | None = None, origin=(0, 6)):
        p = preset
        self.device, self.preset = device, p
        self.k_tiles = int(weight.padded_shape[-2]) // 32
        self.n_tiles = int(weight.padded_shape[-1]) // 32
        if p.k_split or p.cores_per_bank != 4:
            raise ValueError("NormStreamDecodeMatmul: presets with k_split 0 and 4 cores per bank only")
        self.tile_bytes = TILE_BYTES[weight.dtype]
        # ring covering the prologue (~6 us of weight per core), within ~400 KB of L1
        self.slots = slots or min(p.slots + 8, max(p.slots, 400 * 1024 // (p.kb * p.w * self.tile_bytes)))
        if self.k_tiles % p.kb or not 1 <= p.outstanding < self.slots or (p.glu and p.w % 2):
            raise ValueError(f"bad preset {p} for K tiles {self.k_tiles}")
        self.norm = NormPrologue(device, norm_weight, eps, origin=origin)
        if self.norm.kt != self.k_tiles:
            raise ValueError("norm width != matmul K")
        self.weight = weight
        self.out_dtype = out_dtype
        self.output_memory_config = output_memory_config
        self.h_memory_config = h_memory_config
        self.mm_fid = _FID[compute_kernel_config.math_fidelity]
        n_valid = self.n_tiles
        self.n_out = self.n_tiles // 2 if p.glu else n_valid
        columns = stream_columns(self.n_tiles, p.cores_per_bank, unit=p.w if p.glu else 1)
        cores = rect_bank_cores(device, self.norm, p.cores_per_bank)
        self.core_args = []
        for b, (cols, bank_cores) in enumerate(zip(columns, cores)):
            run = 0
            for (first, n), core in zip(cols, bank_cores):
                if n <= 0:
                    raise ValueError("every core needs weight columns")
                self.core_args.append((core, [b, run * self.tile_bytes, n, first // 2 if p.glu else first]))
                run += n * self.k_tiles
        self._cache = {}

    def _build(self, a, b, h, out, rows, embed=False):
        from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
            PerCoreRuntimeArgsDescriptor,
            UnifiedKernelDescriptor,
        )

        p, cs, nm = self.preset, self.norm.core_set, self.norm
        residual = b is not None and not embed
        out_tile_bytes = 4096 if self.out_dtype == ttnn.float32 else 2048
        tile = ttnn.TileDescriptor(ttnn.Tile([32, 32]))

        def cb(index, dtype, page, pages):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        cbs = [cb(0, ttnn.bfloat16, 2048, self.k_tiles),
               cb(1, self.weight.dtype, self.tile_bytes, self.slots * p.kb * p.w),
               cb(16, self.out_dtype, out_tile_bytes, 2 * p.w)] + nm.cbs(residual, embed)
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        common = [("kt", self.k_tiles), ("w", p.w), ("kb", p.kb), ("rows", rows), ("ks", 0), ("glu", int(p.glu))]
        np_dm = nm.named_args(rows, residual, cb_x=0, embed=embed)  # (NCRISC too: the DM template reads them)
        np_c = nm.named_args(rows, residual, cb_x=0, dm=False, embed=embed)
        accessors = list(ttnn.TensorAccessorArgs(out).get_compile_time_args()) + nm.accessor_args(a, b, h)
        addrs = [int(self.weight.buffer_address()), 0, int(out.buffer_address())]
        per_core = [(c, list(args)) for c, args in self.core_args]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=cs,
            ncrisc_named_compile_time_args=common + np_dm + [("in1_tile_bytes", self.tile_bytes),
                                                            ("slots", self.slots), ("outstanding", p.outstanding)],
            brisc_named_compile_time_args=common + np_dm + [("out_tile_bytes", out_tile_bytes)],
            brisc_compile_time_args=accessors,
            trisc_named_compile_time_args=common + np_c + nm.compute_args() + [("mm_fid", self.mm_fid)],
            ncrisc_common_runtime_args=addrs,
            brisc_common_runtime_args=addrs + nm.runtime_args(a, b, h),
            trisc_common_runtime_args=addrs,
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=per_core, brisc_args=per_core, trisc_args=per_core),
        )
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs,
                                      semaphores=nm.semaphores())

    def _check(self, t, name):
        if (len(t.shape) != 4 or tuple(t.shape)[:2] != (1, 1) or not 1 <= t.shape[2] <= 32
                or t.shape[3] != self.k_tiles * 32 or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT
                or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            raise ValueError(f"{name} must be an interleaved BF16 TILE [1, 1, rows<=32, K] tensor")

    def __call__(self, a, b=None, output_memory_config=None, h_memory_config=None):
        self._check(a, "a")
        if b is not None:
            self._check(b, "b")
            if tuple(b.shape) != tuple(a.shape):
                raise ValueError("a and b must have the same shape")
        rows = int(a.shape[2])
        out = ttnn.empty([1, 1, rows, self.n_out * 32], dtype=self.out_dtype, layout=ttnn.TILE_LAYOUT,
                         device=self.device, memory_config=output_memory_config or self.output_memory_config)
        if b is None:
            h, tensors = a, [a, self.weight, self.norm.gamma, out]
        else:
            h = ttnn.empty(list(a.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=h_memory_config or self.h_memory_config)
            tensors = [a, b, self.weight, self.norm.gamma, h, out]
        cache = self._cache.setdefault((b is None, rows), _CachedProgram())
        cache.run(tensors, lambda: self._build(a, b, h if b is not None else None, out, rows))
        return h, out

    def embed(self, tok, table, output_memory_config=None, h_memory_config=None):
        """Layer-0 form with the embedding gather folded in: x = table[tok] (the residual input, [1, 1, B, K] BF16
        tile, h_memory_config) and y = rms_norm(x) * w @ W; replaces `x = reshape(ttnn.embedding(tok, table))`
        + `_, y = self(x)`.  tok: [1, B] uint32 ROW_MAJOR interleaved; table: [vocab, K] BF16 ROW_MAJOR
        interleaved (the model's embedding table)."""
        rows = int(tok.shape[-1])
        if (tok.layout != ttnn.ROW_MAJOR_LAYOUT or table.layout != ttnn.ROW_MAJOR_LAYOUT or table.dtype != ttnn.bfloat16
                or int(table.shape[-1]) != self.k_tiles * 32 or not 1 <= rows <= 16):
            raise ValueError("embed: tok [1, B<=16] uint32 row-major, table [vocab, K] BF16 row-major")
        out = ttnn.empty([1, 1, rows, self.n_out * 32], dtype=self.out_dtype, layout=ttnn.TILE_LAYOUT,
                         device=self.device, memory_config=output_memory_config or self.output_memory_config)
        x = ttnn.empty([1, 1, rows, self.k_tiles * 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                       device=self.device, memory_config=h_memory_config or self.h_memory_config)
        cache = self._cache.setdefault(("embed", rows), _CachedProgram())
        cache.run([table, tok, self.weight, self.norm.gamma, x, out],
                  lambda: self._build(table, tok, x, out, rows, embed=True))
        return x, out
