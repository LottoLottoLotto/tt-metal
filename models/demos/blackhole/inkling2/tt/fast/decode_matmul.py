# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Decode (one tile row) matmuls streamed from DRAM with bank-contiguous weights (ported from LFM2.5
lfm2/tt/fast/decode_matmul.py; owner AttnFast, shared with GlueFast / HeadFast).

out[1, 1, rows<=32, N] = x[1, 1, rows, K] @ W^T[K, N], one generic_op (kernels/fast/decode_matmul_stream.cpp):
banks * cores_per_bank workers; weight tile column block b (N/8 columns) lives in DRAM bank b; each worker owns a few
consecutive output columns whose weight tiles are ONE contiguous run in its bank (`stream_layout`, a pure tile
permutation: block-float tiles keep their values), streamed by NCRISC in kb x w blocks; BRISC gathers the logical
rows of x and writes the logical output rows; TRISC accumulates K in FP32 DEST (whole K, or the stock in0_block_w
order with `k_split`).

Inkling extension: `segments` = output tile columns where a new input starts (e.g. [0, 192, 224] for [q|g | k | v]
of xq, xk, xv); `__call__(x0, x1, ...)` then runs every input's projection in one launch (a worker's columns must lie
inside one segment).  `col_seg` instead gives the input index of every weight column (a subblock's columns share one
input; <= 16 subblocks per worker): a worker may then mix inputs (its subblocks pick their input).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .common import DRAM, KERNELS, TILE_BYTES, CachedProgram

_KERNEL = str(KERNELS / "decode_matmul_stream.cpp")
_BANKS = 8  # P150 DRAM banks


@dataclass(frozen=True)
class Preset:
    """Program geometry: banks * cores_per_bank workers, subblocks of w weight columns, kb-row K blocks streamed
    with `slots` ring slots / `outstanding` reads in flight; k_split: stock accumulation order; glu: fused SwiGLU."""

    cores_per_bank: int
    w: int
    kb: int
    slots: int = 3
    outstanding: int = 2
    k_split: int = 0
    glu: bool = False

    @property
    def tag(self) -> str:
        """Weight-layout tag for cache keys (the layout depends on cores_per_bank, w and glu only)."""
        return f"stream-c{self.cores_per_bank}w{self.w}" + ("-glu" if self.glu else "")


# LFM2.5-tuned presets (K = 2048); Inkling presets are added by their users (K = 2304 = 72 tiles: kb | 72).
PRESETS = {
    "conv_in": Preset(4, 4, 4, 2, 1),
    "conv_out": Preset(4, 2, 8),
    "attn_qkv": Preset(4, 3, 8),
    "attn_out": Preset(4, 2, 8),
    "mlp_w13": Preset(4, 4, 4, 2, 1, glu=True),
    "mlp_w2": Preset(4, 2, 8, 2, 1),
    "lm_head": Preset(4, 4, 4, 2, 1),
    "router": Preset(1, 1, 32, k_split=2),
}


def stream_columns(n_tiles: int, cores_per_bank: int, banks: int = _BANKS, unit: int = 1):
    """Per bank, the (first_col, ncols) of each of its cores: bank b serves weight tile columns
    [b*n/banks, (b+1)*n/banks), split as evenly as possible over its cores in multiples of `unit` columns."""
    if n_tiles % (banks * unit):
        raise ValueError(f"stream layout needs N tiles divisible by {banks * unit}, got {n_tiles}")
    per_bank = n_tiles // banks // unit
    if not 1 <= cores_per_bank <= per_bank:
        raise ValueError(f"cores_per_bank must be in [1, {per_bank}], got {cores_per_bank}")
    q, r = divmod(per_bank, cores_per_bank)
    out = []
    for b in range(banks):
        first, cols = b * per_bank * unit, []
        for c in range(cores_per_bank):
            n = (q + (c < r)) * unit
            cols.append((first, n))
            first += n
        out.append(cols)
    return out


def stream_layout(weight: torch.Tensor, cores_per_bank: int, w: int, unit: int = 1, banks: int = _BANKS,
                  n_valid: int | None = None):
    """[K, N] W^T -> [K, N] tile permutation (see the module docstring).  A core's subblocks cover its valid
    columns (< n_valid tiles) first, then its zero-padding columns (the kernel streams only the valid ones)."""
    k_tiles, n_tiles = weight.shape[-2] // 32, weight.shape[-1] // 32
    n_valid = n_tiles if n_valid is None else n_valid
    src = []
    for cols in stream_columns(n_tiles, cores_per_bank, banks, unit):
        seq = []
        for first, n in cols:
            nv = max(0, min(n, n_valid - first))
            for a, b in ((0, nv), (nv, n)):
                for s0 in range(a, b, w):
                    j = torch.arange(first + s0, first + min(b, s0 + w)).reshape(1, -1)
                    k = torch.arange(k_tiles).reshape(-1, 1)
                    seq.append((k * n_tiles + j).reshape(-1))
        src.append(torch.cat(seq))
    target_to_source = torch.stack(src, dim=1).reshape(-1)  # page p = bank + banks * bank_page
    tiles = weight.reshape(k_tiles, 32, n_tiles, 32).permute(0, 2, 1, 3).reshape(k_tiles * n_tiles, 32, 32)
    return tiles[target_to_source].reshape(k_tiles, n_tiles, 32, 32).permute(0, 2, 1, 3).reshape(weight.shape)


def glu_weight(w1: torch.Tensor, w3: torch.Tensor, w: int) -> torch.Tensor:
    """[K, I] W1^T and W3^T -> [K, 2I] in groups of w tiles: w/2 gate tiles, then the matching w/2 up tiles."""
    k, i = w1.shape
    h = w // 2
    if i % (32 * h):
        raise ValueError(f"intermediate {i} must split into groups of {h} tiles")
    g = w1.reshape(k, i // (32 * h), 1, h * 32)
    u = w3.reshape(k, i // (32 * h), 1, h * 32)
    return torch.cat([g, u], dim=2).reshape(k, 2 * i)


def _as_torch(t) -> torch.Tensor:
    t = t if isinstance(t, torch.Tensor) else ttnn.to_torch(t)
    return t.reshape(t.shape[-2], t.shape[-1]).to(torch.bfloat16)


def stream_weight_torch(weight, preset: Preset, w3=None) -> torch.Tensor:
    """Host [K, N_pad] BF16 in the preset's stream layout from a [K, N] W^T (torch or ttnn tensor; BF16 rounding
    first, as WeightCache.get).  N is zero-padded to a multiple of 8 tiles; padding columns are never read."""
    w = _as_torch(weight)
    if preset.glu:
        w = glu_weight(w, _as_torch(w3), preset.w)
    unit = preset.w if preset.glu else 1
    n_valid = -(-w.shape[1] // 32)
    pad = -w.shape[1] % (32 * _BANKS * unit)
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return stream_layout(w, preset.cores_per_bank, preset.w, unit, n_valid=n_valid).contiguous()


def cached_stream_weight(cache, key: str, preset: Preset, build, dtype=ttnn.bfloat8_b, build3=None):
    """Device stream weight through the runtime's WeightCache under f"{key}.{preset.tag}"; build() / build3()
    return the [K, N] W^T torch (glu: W1^T / W3^T)."""
    return cache.get(f"{key}.{preset.tag}",
                     lambda: stream_weight_torch(build(), preset, build3() if build3 else None), dtype)


def stream_cores(device, cores_per_bank: int, banks: int = _BANKS, first: int = 0):
    """(core_set, cores[b][c]): banks * cores_per_bank workers starting at row-major grid index `first`, each bank's
    cores greedily nearest to the bank's preferred worker."""
    grid = device.compute_with_storage_grid_size()
    anchors = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    n = banks * cores_per_bank
    if len(anchors) != banks or first + n > grid.x * grid.y:
        raise ValueError(f"need {banks} bank anchors and {n} workers")
    pool = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(first, first + n)]
    from .common import row_ranges
    core_set = row_ranges(pool)
    free = {(c.x, c.y) for c in pool}
    cores = [[] for _ in range(banks)]
    for _ in range(cores_per_bank):
        for b, a in enumerate(anchors):
            best = min(free, key=lambda c: (abs(c[0] - a.x) + abs(c[1] - a.y), c[1], c[0]))
            free.remove(best)
            cores[b].append(ttnn.CoreCoord(*best))
    return core_set, cores


class StreamDecodeMatmul:
    """out[1, 1, rows, n_out] = x[1, 1, rows<=32, K] @ W, one generic_op (trace-safe, no host reads).

    weight: stream layout (`stream_weight_torch` / `cached_stream_weight`), DRAM interleaved BFP8/BFP4/BF16.
    compute_kernel_config: its math fidelity is used, with FP32 DEST accumulation.  n_out: logical output width
    (default: the weight's).  segments: see the module docstring.  first_core: row-major grid index of the first
    worker (the workers are the next banks * cores_per_bank cores).
    x_rm: x is one ROW-MAJOR BF16 row [1, 1, 1, K] (DRAM, one page): one read per worker instead of 2 per K tile
    (decode_matmul_stream.cpp; rows = 1)."""

    def __init__(self, device, weight, preset: Preset, compute_kernel_config, out_dtype=ttnn.bfloat16,
                 output_memory_config=ttnn.DRAM_MEMORY_CONFIG, n_out: int | None = None, segments=(0,),
                 first_core: int = 0, pad_zero: bool = False, x_rm: bool = False, col_seg=None):
        p = preset
        self.device, self.preset = device, p
        self.k_tiles = int(weight.padded_shape[-2]) // 32
        self.n_tiles = int(weight.padded_shape[-1]) // 32
        if (weight.dtype not in TILE_BYTES or weight.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
                or weight.memory_config().buffer_type != ttnn.BufferType.DRAM):
            raise ValueError("StreamDecodeMatmul: weight must be DRAM-interleaved BFP8/BFP4/BF16")
        if (self.k_tiles % p.kb or not 1 <= p.w <= 4 or not 1 <= p.outstanding < p.slots
                or (p.k_split and p.kb % p.k_split) or (p.glu and (p.w % 2 or p.k_split))):
            raise ValueError(f"StreamDecodeMatmul: bad preset {p} for K tiles {self.k_tiles}")
        self.weight = weight
        self.out_dtype = out_dtype
        self.output_memory_config = output_memory_config
        self.fidelity = compute_kernel_config.math_fidelity
        n_valid = self.n_tiles if p.glu else (n_out or self.n_tiles * 32) // 32
        self.n_out = self.n_tiles // 2 if p.glu else n_valid
        self.tile_bytes = TILE_BYTES[weight.dtype]
        self.segments = list(segments)
        self.col_seg = None if col_seg is None else list(col_seg)
        self.pad_zero = pad_zero
        self.x_rm = x_rm
        if pad_zero and out_dtype != ttnn.bfloat16:
            raise ValueError("StreamDecodeMatmul: pad_zero needs BF16 output")
        columns = stream_columns(self.n_tiles, p.cores_per_bank, unit=p.w if p.glu else 1)
        core_set, cores = stream_cores(device, p.cores_per_bank, first=first_core)
        # per core: bank, run offset (bytes, bank-relative), weight columns, first output column, input index
        self.core_args = []
        for b, (cols, bank_cores) in enumerate(zip(columns, cores)):
            run = 0
            for (first, n), core in zip(cols, bank_cores):
                nv = min(n, n_valid - first)
                if nv > 0 and self.col_seg is not None:  # packed 2-bit input index per subblock
                    seg = 0
                    for si, c0 in enumerate(range(first, first + nv, p.w)):
                        ids = {self.col_seg[c] for c in range(c0, min(first + nv, c0 + p.w))}
                        if len(ids) != 1 or si >= 16:
                            raise ValueError(f"StreamDecodeMatmul: subblock at column {c0}: one input, <= 16 subblocks")
                        seg |= ids.pop() << (2 * si)
                elif nv > 0:  # workers of zero-padding columns are not instantiated
                    seg = sum(1 for s in self.segments if s <= first) - 1
                    if seg + 1 < len(self.segments) and first + nv > self.segments[seg + 1]:
                        raise ValueError(f"StreamDecodeMatmul: worker columns [{first}, {first + nv}) cross a segment")
                if nv > 0:
                    self.core_args.append((core, [b, run * self.tile_bytes, nv, first // 2 if p.glu else first, seg]))
                run += n * self.k_tiles
        if len(self.core_args) < len(columns) * p.cores_per_bank:
            from .common import row_ranges
            core_set = row_ranges([c for c, _ in self.core_args])
        self.core_set = core_set
        self._progs = {}  # rows -> CachedProgram (rows is compiled in)

    def blocks(self, i: int):
        """Worker i's streamed blocks in consumption order: [(DRAM bank, bank address, bytes)]."""
        p = self.preset
        bank, run, ncols = self.core_args[i][1][:3]
        nsub = -(-ncols // p.w)
        rem = ncols - (nsub - 1) * p.w
        addr, out = int(self.weight.buffer_address()) + run, []
        for s in range(nsub):
            nbytes = p.kb * (rem if s == nsub - 1 else p.w) * self.tile_bytes
            for _ in range(self.k_tiles // p.kb):
                out.append((bank, addr, nbytes))
                addr += nbytes
        return out

    def kernel_descriptors(self, xs, out, rows, routes=None, route_cbs=(0, 0, 0, 0), route_stride: int = 2048,
                           done=None, prefetch=None, x_first: bool = False):
        """routes (fused consumers, see decode_matmul_stream.cpp): {worker index: (dram mask, [(local column, face,
        zero, target, noc x, noc y, byte offset)])}; route_cbs: the 4 target CB ids (declared first on every core);
        rows > 1: row r of a route goes to byte offset + r * route_stride.  done = (semaphore id, noc x, noc y): every
        worker increments it once all its writes landed.  prefetch: {worker index: (block mask, [(noc x, noc y, L1
        address)] per set bit)}: those blocks are read from L1 (AttnPrefetch) instead of DRAM.  x_first: the weight
        stream starts once the input rows are gathered (many-row inputs; the same values)."""
        from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
            PerCoreRuntimeArgsDescriptor,
            UnifiedKernelDescriptor,
        )

        p = self.preset
        out_tile_bytes = TILE_BYTES[self.out_dtype]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        if p.k_split:
            mode = [ttnn.UnpackToDestMode.Default] * 64
            mode[24] = ttnn.UnpackToDestMode.UnpackToDestFp32  # exact reload of the FP32 partials
            compute.unpack_to_dest_mode.extend(mode)
        if x_first and (self.x_rm or not all(x.memory_config().buffer_type == ttnn.BufferType.DRAM for x in xs)):
            raise ValueError("StreamDecodeMatmul(x_first): tile inputs in DRAM (2 KB pages on the 8 banks)")
        common = [("kt", self.k_tiles), ("w", p.w), ("kb", p.kb), ("rows", rows), ("ks", p.k_split),
                  ("glu", int(p.glu)), ("seg_mode", int(self.col_seg is not None)), ("n_in", self.n_in(xs)),
                  ("x_first", int(x_first))]
        accessors = (list(ttnn.TensorAccessorArgs(xs[0]).get_compile_time_args())
                     + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()))
        addrs = [int(self.weight.buffer_address()), int(out.buffer_address())] + [int(x.buffer_address()) for x in xs]
        per_core, nc_args = [], []
        # prefetch args padded to the largest count (compile-time `pf_slots`): equally hashed programs get equally
        # long runtime args (generic_op's program cache ignores their values and counts)
        npf = max((len(v[1]) for v in prefetch.values()), default=0) if prefetch is not None else 0
        for i, (c, a) in enumerate(self.core_args):
            args = list(a)
            if prefetch is not None:
                mask, locs = prefetch.get(i, (0, []))
                nc_args.append((c, args + [mask] + [v for x, y, l1 in locs for v in (x << 16 | y, l1)]
                                + [0] * (2 * (npf - len(locs)))))
            if routes is not None:
                mask, rl = routes.get(i, ((1 << 32) - 1, []))
                args += [mask, len(rl)]
                for j, face, zero, target, x, y, off in rl:
                    args += [j | face << 8 | zero << 9 | target << 12, x, y, off]
            per_core.append((c, args))
        return UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=self.core_set,
            ncrisc_named_compile_time_args=common + [("in1_tile_bytes", self.tile_bytes), ("slots", p.slots),
                                                     ("outstanding", p.outstanding), ("pf", int(prefetch is not None)),
                                                     ("pf_slots", npf)],
            brisc_named_compile_time_args=common + [("out_tile_bytes", out_tile_bytes),
                                                    ("pad_zero", int(self.pad_zero)), ("route", int(routes is not None)),
                                                    ("x_rm", int(self.x_rm)), ("x_rowbuf", 2),
                                                    ("route_stride", route_stride),
                                                    ("done_sem", done[0] if done else 16),
                                                    ("done_x", done[1] if done else 0), ("done_y", done[2] if done else 0)]
            + [(f"route_cb{i}", c) for i, c in enumerate(route_cbs)],
            brisc_compile_time_args=accessors,
            trisc_named_compile_time_args=common,
            ncrisc_common_runtime_args=addrs,
            brisc_common_runtime_args=addrs,
            trisc_common_runtime_args=addrs,
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=nc_args if prefetch is not None else per_core, brisc_args=per_core, trisc_args=per_core),
        ).get_kernel_descriptors().kernels

    def n_in(self, xs) -> int:
        """in0 input slots per worker (col_seg: every input; else the worker's one)."""
        return len(xs) if self.col_seg is not None else 1

    def cbs(self, n_in: int = 1, x_first: bool = False):
        p, cs = self.preset, self.core_set
        tile = ttnn.TileDescriptor(ttnn.Tile([32, 32]))

        def cb(index, dtype, page, pages):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        out = [cb(0, ttnn.bfloat16, 2048, self.k_tiles * n_in),
               cb(1, self.weight.dtype, self.tile_bytes, p.slots * p.kb * p.w),
               cb(16, self.out_dtype, TILE_BYTES[self.out_dtype], 2 * p.w)]
        if p.k_split:
            out.append(cb(24, ttnn.float32, 4096, p.w))
        if self.x_rm:  # the row buffer (64 B per K tile)
            out.append(cb(2, ttnn.bfloat16, self.k_tiles * 64, 1))
        if x_first:  # NCRISC's half of the input gather landed (decode_matmul_stream.cpp cb_xdone)
            out.append(cb(3, ttnn.bfloat16, 2048, 1))
        return out

    def _check(self, xs, rows=None):
        if self.x_rm:
            for x in xs:
                if (tuple(x.shape) != (1, 1, 1, self.k_tiles * 32) or x.dtype != ttnn.bfloat16
                        or x.layout != ttnn.ROW_MAJOR_LAYOUT or x.memory_config() != DRAM):
                    raise ValueError("StreamDecodeMatmul(x_rm): x must be a [1, 1, 1, K] BF16 ROW_MAJOR DRAM row")
            return 1
        rows = rows or xs[0].shape[-2]
        if len(xs) != len(self.segments):
            raise ValueError(f"StreamDecodeMatmul: {len(self.segments)} inputs expected, got {len(xs)}")
        for x in xs:
            shape = list(x.shape)
            if (len(shape) != 4 or shape[:2] != [1, 1] or shape[-2] < rows or rows > 32
                    or shape[-1] != self.k_tiles * 32 or x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT
                    or x.memory_config() != xs[0].memory_config()
                    or x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
                raise ValueError("StreamDecodeMatmul: inputs must be BF16 TILE interleaved [1, 1, rows<=32, K] "
                                 "with one memory config")
        return rows

    def __call__(self, *xs, output_memory_config=None, out=None, rows=None, prefetch=None, prefetch_buf=None,
                 key=None, x_first: bool = False):
        """rows: logical rows to compute (default x's); pad_zero: rows <= 16 and the output has x's row count
        (rows 0..rows-1 computed, the rest zero).  prefetch (+ the L1 buffer tensor holding the blocks): see
        kernel_descriptors; key: names the prefetch plan (one cached program per plan); x_first: see
        kernel_descriptors."""
        rows = self._check(xs, rows)
        if self.pad_zero and rows > 16:
            raise ValueError("StreamDecodeMatmul: pad_zero computes up to 16 rows")
        mc = output_memory_config or self.output_memory_config
        if out is None:
            out = ttnn.empty([1, 1, 32 if self.x_rm else (xs[0].shape[-2] if self.pad_zero else rows), self.n_out * 32], dtype=self.out_dtype, layout=ttnn.TILE_LAYOUT,
                             device=self.device, memory_config=mc)
        extra = [prefetch_buf] if prefetch is not None else []
        self._progs.setdefault((rows, prefetch is not None, key, x_first), CachedProgram()).run(
            [*xs, self.weight, out, *extra], lambda: ttnn.ProgramDescriptor(
                kernels=self.kernel_descriptors(xs, out, rows, prefetch=prefetch, x_first=x_first),
                cbs=self.cbs(self.n_in(xs), x_first), semaphores=[]))
        return out
