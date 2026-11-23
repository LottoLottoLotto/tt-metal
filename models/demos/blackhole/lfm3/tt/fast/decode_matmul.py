# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Decode (one tile row) matmuls for LFM3 on the P150: DRAM-streaming matmul with bank-contiguous weights.

out[1, 1, rows<=32, N] = x[1, 1, rows, K] @ W^T[K, N] for every decode projection (BFP8 / BFP4 weights, BF16 out
with CK_BFP8, bitwise `layers.linear`; the tied LM head with FP32 out + CK_HIFI4 (tests) or BF16 out (serving)),
plus the dense MLP's gate/up projection fused with SwiGLU. Replaces `layers.linear(x, w)` /
`ttnn.linear(x, head, dtype=..., ...)` for M = 1 tile row (decode only; prefill keeps ttnn.linear).

Program (kernels/decode_matmul_stream.cpp, one generic_op): banks * cores_per_bank workers (the first workers
row-major: one or two rectangles, so the kernel binaries are multicast at launch). Weight tile column block b
(N/8 columns) lives in DRAM bank b; each worker owns a few consecutive output columns and its weight tiles form
ONE contiguous run in its bank (`stream_layout`, a pure tile permutation of the [K, N] tensor zero-padded to a
multiple of 8 tile columns, stored DRAM interleaved, block-float values unchanged (verified when the cache entry is
built); prefill keeps the original layout, so both are resident), which its NCRISC streams in kb x w tile blocks
(transaction-id pipelined). Workers stream only their valid (non-padding) columns. BRISC gathers only the logical
rows of x's K tiles (x DRAM or L1 interleaved; from 9 rows one worker gathers and multicasts them) and writes only
the logical output rows. TRISC accumulates K in FP32 DEST with the stock matmul_block / fidelity: whole K (the
tuned presets: BF16 outputs equal ttnn.linear except rare 1-ulp elements), or the stock in0_block_w = 2 order
(`exact_preset`, k_split = 2: bitwise ttnn.linear for BF16 and FP32 outputs).
Measured: tests/fast/test_decode_matmul.py (logs/fast_decode_matmul.json).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
import ttnn

_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "decode_matmul_stream.cpp")
_BANKS = 8  # P150 DRAM banks
# From this many x rows one worker gathers x and multicasts it (StreamDecodeMatmul x_mcast_rows default).
X_MCAST_ROWS = 9
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}


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


# Tuned on the P150 with real LFM3 weights (tests/fast/test_decode_matmul.py --sweep, logs/fast_decode_matmul_sweep.json;
# rep_us at B = 1, x L1 interleaved; GB/s = weight bytes / time).  Keys are the LFM3 decode projections (W^T shapes,
# WeightCache keys of layers.py).  NormStreamDecodeMatmul needs cores_per_bank 2 / 4 / 8.
PRESETS = {
    "conv_in": Preset(4, 4, 8, 2, 1),  # 2048 x 6144   L{i}.conv.in_proj        31.2 us, 429 GB/s
    "conv_out": Preset(4, 2, 8, 4, 3),  # 2048 x 2048   L{i}.conv.out_proj      11.8 us, 377 GB/s
    "attn_qkv": Preset(4, 3, 8),  # 2048 x 3104   L{i}.attn.qkvg (q|k|v|gate|0; 97 tiles -> 104)  18.3 us, 369 GB/s
    "attn_out": Preset(4, 2, 8, 4, 3),  # 2048 x 2048   L{i}.attn.o_proj        (= conv_out)
    "mlp_w13": Preset(4, 4, 4, 2, 1, glu=True),  # 2048 x 2*7680  SiLU(x W1) * (x W3)   78.8 us, 424 GB/s
    "mlp_w2": Preset(8, 1, 40, 3, 2),  # 7680 x 2048  L{i}.mlp.w2                  40.6 us, 412 GB/s
    "lm_head": Preset(4, 4, 4, 2, 1),  # 2048 x 128000 (FP32 out + CK_HIFI4, or BF16 out)  554 us, 502 GB/s
}
# BFP4 weights (576 B tiles; the GPTQ checkpoint's w1 / w3): separately tuned where it pays, else the BFP8 preset.
PRESETS_BFP4 = {
    # 44.9 us standalone (399 GB/s; the BFP8 preset: 54 us), 42.4 us fused with the sandwich norm (8 x 8 at (0, 2))
    "mlp_w13": Preset(8, 4, 4, 2, 1, glu=True),
}
# The stock ttnn.linear of these decode shapes accumulates K in in0_block_w = 2 tile chunks (packer L1 accumulation
# in FP32): k_split = 2 reproduces it bit for bit (proven with FP32 outputs, where every order difference shows).
# Whole-K accumulation (k_split 0, the tuned presets) differs from it only in FP32 rounding, i.e. after the BF16
# output rounding in rare elements by 1 ulp (mlp_w2: ~1e-5 of the outputs); it costs +5 us (conv_in) / +11 us (w2)
# / +134 us (LM head) to be exact.
STOCK_K_SPLIT = 2


def preset_for(name: str, dtype) -> Preset:
    """The tuned preset of projection `name` for weights of `dtype` (ttnn.bfloat8_b / ttnn.bfloat4_b)."""
    if dtype == ttnn.bfloat4_b and name in PRESETS_BFP4:
        return PRESETS_BFP4[name]
    return PRESETS[name]


def exact_preset(preset: Preset) -> Preset:
    """`preset` with the stock accumulation order (bitwise ttnn.linear; same weight layout; not for glu presets)."""
    if preset.glu or preset.kb % STOCK_K_SPLIT:
        raise ValueError(f"no stock-order variant of {preset}")
    return replace(preset, k_split=STOCK_K_SPLIT)


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
    """[K, N] W^T -> [K, N] tile permutation. Stored DRAM-interleaved (tile page p lives in bank p % banks at bank
    page p // banks), every core's weight tiles are one contiguous run in its bank, ordered by subblock (w columns,
    the last one narrower), then K, then column (consecutive [kb, w] K-major blocks for any kb). Columns >= n_valid
    (zero padding) of a core come after its valid ones, in their own subblocks, so a worker streams just the prefix
    of its run that holds valid columns. Pure tile permutation: block-float tiles keep their exact values."""
    k_tiles, n_tiles = weight.shape[-2] // 32, weight.shape[-1] // 32
    n_valid = n_tiles if n_valid is None else n_valid
    src = []
    for cols in stream_columns(n_tiles, cores_per_bank, banks, unit):
        seq = []
        for first, n in cols:
            split = min(max(n_valid - first, 0), n)
            for lo, hi in ((first, first + split), (first + split, first + n)):
                for s0 in range(lo, hi, w):
                    j = torch.arange(s0, min(hi, s0 + w)).reshape(1, -1)
                    k = torch.arange(k_tiles).reshape(-1, 1)
                    seq.append((k * n_tiles + j).reshape(-1))
        src.append(torch.cat(seq))
    target_to_source = torch.stack(src, dim=1).reshape(-1)  # page p = bank + banks * bank_page
    tiles = weight.reshape(k_tiles, 32, n_tiles, 32).permute(0, 2, 1, 3).reshape(k_tiles * n_tiles, 32, 32)
    return tiles[target_to_source].reshape(k_tiles, n_tiles, 32, 32).permute(0, 2, 1, 3).reshape(weight.shape)


def glu_weight(w1: torch.Tensor, w3: torch.Tensor, w: int) -> torch.Tensor:
    """[K, I] gate W1^T and up W3^T -> [K, 2I] with columns in groups of w tiles: w/2 gate tiles, then the matching
    w/2 up tiles (StreamDecodeMatmul with a glu preset forms SiLU(gate) * up of each group in DEST)."""
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
    """Host [K, N_pad] BF16 in the preset's stream layout from a [K, N] W^T (torch, or a ttnn host/device tensor
    such as the WeightCache entry; BFP8 / BFP4 values convert exactly). glu presets take W1^T as `weight` and W3^T
    as `w3`. N is zero-padded to a multiple of 8 tiles (one column block per DRAM bank; the 97-tile qkvg -> 104);
    the padding columns are neither read nor written."""
    w = _as_torch(weight)
    if preset.glu:
        w = glu_weight(w, _as_torch(w3), preset.w)
    unit = preset.w if preset.glu else 1
    n_valid = -(-w.shape[1] // 32)
    pad = -w.shape[1] % (32 * _BANKS * unit)
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return stream_layout(w, preset.cores_per_bank, preset.w, unit, n_valid=n_valid).contiguous()


def exact_block_float(t: torch.Tensor, dtype) -> torch.Tensor:
    """t (BF16 values of a `dtype` tensor, permuted by whole tiles) after a host round trip through `dtype` must be
    unchanged: the stream copy holds exactly the original block-float values. Returns t; raises otherwise."""
    back = ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).reshape(t.shape)
    if not torch.equal(back.to(torch.bfloat16), t.to(torch.bfloat16)):
        n = int((back.float() != t.float()).sum())
        raise ValueError(f"stream weight: {n} values change in {dtype} (source is not that block-float format)")
    return t


def stream_weight(weight, preset: Preset, device=None, dtype=None, w3=None):
    """`stream_weight_torch` as a TILE tensor of `dtype` (default: the ttnn weight's own dtype): host (device=None)
    or DRAM-interleaved on `device`."""
    if dtype is None:
        if isinstance(weight, torch.Tensor):
            raise ValueError("stream_weight: dtype is required for a torch weight")
        dtype = weight.dtype
    host = ttnn.from_torch(stream_weight_torch(weight, preset, w3), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return host if device is None else ttnn.to_device(host, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def cached_stream_weight(cache, key: str, preset: Preset, dtype, key3: str | None = None, build=None, build3=None):
    """Device stream weight through the runtime's WeightCache under f"{key}.{preset.tag}" (dtype: the weight's real
    format, e.g. DTYPES[src.fmt(name)] / the runtime tensor's .dtype), built once from the WeightCache's host file of
    `key` (+ `key3`, the W3 entry, for glu presets) or, when that file is absent, from `build()` / `build3()` (the
    [K, N] W^T torch the runtime's own cache.get would use); the built copy is checked to hold exactly the source's
    block-float values (exact_block_float); later loads read only the new entry."""
    def source(k, b):
        path = cache.path(k, dtype)
        if path.exists():
            return ttnn.load_tensor(str(path))
        if b is None:
            raise FileNotFoundError(f"cached_stream_weight: no {path.name} and no build function")
        return b()

    return cache.get(f"{key}.{preset.tag}", lambda: exact_block_float(stream_weight_torch(
        source(key, build), preset, source(key3, build3) if key3 else None), dtype), dtype)


def stream_cores(device, cores_per_bank: int, banks: int = _BANKS):
    """(core_set, cores[b][c]): the first banks * cores_per_bank workers row-major (one or two rectangles: kernel
    binaries are multicast per rectangle at launch, scattered single cores cost ~0.6 us each), each bank's cores
    greedily nearest to the bank's preferred worker."""
    grid = device.compute_with_storage_grid_size()
    anchors = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    n = banks * cores_per_bank
    if len(anchors) != banks or n > grid.x * grid.y:
        raise ValueError(f"need {banks} bank anchors and {n} workers")
    core_set = ttnn.num_cores_to_corerangeset(n, grid, row_wise=True)
    free = {(c.x, c.y) for c in ttnn.corerange_to_cores(core_set, row_wise=True)}
    cores = [[] for _ in range(banks)]
    for _ in range(cores_per_bank):
        for b, a in enumerate(anchors):
            best = min(free, key=lambda c: (abs(c[0] - a.x) + abs(c[1] - a.y), c[1], c[0]))
            free.remove(best)
            cores[b].append(ttnn.CoreCoord(*best))
    return core_set, cores


class _CachedProgram:
    """Program descriptor rebuilt only when a tensor's address/shape/memory config changes (trace-safe). The logical
    shape is part of the key: the row count is a compile-time argument."""

    def __init__(self):
        self.program, self.signature = None, None

    def run(self, tensors, build):
        sig = tuple((int(t.buffer_address()), tuple(t.shape), str(t.memory_config())) for t in tensors)
        if self.program is None or sig != self.signature:
            self.program, self.signature = build(), sig
        ttnn.generic_op(tensors, self.program)


class StreamDecodeMatmul:
    """out[1, 1, rows, n_out] = x[1, 1, rows<=32, K] @ W, one generic_op (trace-safe, no host reads).

    weight: `stream_weight(W^T, preset)` / `cached_stream_weight(...)` (DRAM interleaved, BFP8/BFP4/BF16).
    compute_kernel_config: its math fidelity is used, with FP32 DEST accumulation (CK_BFP8 -> HiFi2, CK_HIFI4).
    n_out: logical output width, a multiple of 32 (default: the stream weight's padded width; the 3104-wide qkvg
    passes 3104 -- its stream copy is 3328 wide). glu presets: n_out = I and
    out = SiLU(x @ W1) * (x @ W3) in FP32 DEST, rounded once to out_dtype.
    x: BF16 TILE [1, 1, rows, K], DRAM or L1 interleaved; only its logical rows are read: by every worker, or, at
    rows >= x_mcast_rows (None: never), by worker (0, 0) only, which multicasts them to the others (the per-worker
    gather is NoC-bound at many rows).  Output: out_dtype TILE, interleaved `output_memory_config` (per-call
    override); only the logical rows are written (the padding rows of the output tiles are left undefined, as ttnn
    padding)."""

    def __init__(self, device, weight, preset: Preset, compute_kernel_config, out_dtype=ttnn.bfloat16,
                 output_memory_config=ttnn.DRAM_MEMORY_CONFIG, n_out: int | None = None,
                 x_mcast_rows: int | None = X_MCAST_ROWS):
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
        n_valid = self.n_tiles if p.glu else (n_out or self.n_tiles * 32) // 32  # weight columns with data
        self.n_out = self.n_tiles // 2 if p.glu else n_valid
        self.tile_bytes = TILE_BYTES[weight.dtype]
        columns = stream_columns(self.n_tiles, p.cores_per_bank, unit=p.w if p.glu else 1)
        core_set, cores = stream_cores(device, p.cores_per_bank)
        # per core: bank, run offset (bytes, bank-relative), weight columns, first output column.  core_args: the
        # workers with valid columns; the program runs on every worker of the rectangle(s) (a worker of zero-padding
        # columns only takes part in the x multicast), so its kernel binaries stay rectangle-multicast.
        self.core_args, self._workers = [], []
        for b, (cols, bank_cores) in enumerate(zip(columns, cores)):
            run = 0
            for (first, n), core in zip(cols, bank_cores):
                nv = max(0, min(n, n_valid - first))
                args = (core, [b, run * self.tile_bytes, nv, first // 2 if p.glu else first])
                self._workers.append(args)
                if nv > 0:
                    self.core_args.append(args)
                run += n * self.k_tiles
        self.core_set = core_set
        self.x_mcast_rows = x_mcast_rows
        # x multicast: worker (0, 0) gathers x and multicasts it to every rectangle of the program
        self._sender = ttnn.CoreCoord(0, 0)
        self._rects = []
        for r in core_set.ranges():
            a, z = device.worker_core_from_logical_core(r.start), device.worker_core_from_logical_core(r.end)
            size = (r.end.x - r.start.x + 1) * (r.end.y - r.start.y + 1)
            inside = r.start.x <= 0 <= r.end.x and r.start.y <= 0 <= r.end.y
            self._rects += [a.x, a.y, z.x, z.y, size - int(inside)]
        self._cache = _CachedProgram()

    def _build(self, x, out, rows):
        from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
            PerCoreRuntimeArgsDescriptor,
            UnifiedKernelDescriptor,
        )

        p, cs = self.preset, self.core_set
        out_tile_bytes = 4096 if self.out_dtype == ttnn.float32 else 2048
        tile = ttnn.TileDescriptor(ttnn.Tile([32, 32]))

        def cb(index, dtype, page, pages):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        cbs = [cb(0, ttnn.bfloat16, 2048, self.k_tiles),
               cb(1, self.weight.dtype, self.tile_bytes, p.slots * p.kb * p.w),
               cb(16, self.out_dtype, out_tile_bytes, 2 * p.w)]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        if p.k_split:
            cbs.append(cb(24, ttnn.float32, 4096, p.w))
            mode = [ttnn.UnpackToDestMode.Default] * 64
            mode[24] = ttnn.UnpackToDestMode.UnpackToDestFp32  # exact reload of the FP32 partials
            compute.unpack_to_dest_mode.extend(mode)
        common = [("kt", self.k_tiles), ("w", p.w), ("kb", p.kb), ("rows", rows), ("ks", p.k_split),
                  ("glu", int(p.glu))]
        accessors = (list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
                     + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()))
        addrs = [int(self.weight.buffer_address()), int(x.buffer_address()), int(out.buffer_address())]
        xm = self.x_mcast_rows is not None and rows >= self.x_mcast_rows
        per_core = [(c, list(a)) for c, a in self._workers]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=cs,
            ncrisc_named_compile_time_args=common + [("in1_tile_bytes", self.tile_bytes), ("slots", p.slots),
                                                     ("outstanding", p.outstanding)],
            brisc_named_compile_time_args=common + [("out_tile_bytes", out_tile_bytes), ("xm", int(xm)),
                                                    ("xm_x", self._sender.x), ("xm_y", self._sender.y)],
            brisc_compile_time_args=accessors,
            trisc_named_compile_time_args=common,
            ncrisc_common_runtime_args=addrs,
            brisc_common_runtime_args=addrs + [len(self._rects) // 5] + self._rects,
            trisc_common_runtime_args=addrs,
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=per_core, brisc_args=per_core, trisc_args=per_core),
        )
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cs, initial_value=0)] if xm else []
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=sems)

    def __call__(self, x, output_memory_config=None):
        mc = output_memory_config or self.output_memory_config
        shape = list(x.shape)
        rows = shape[-2]
        if (len(shape) != 4 or shape[:2] != [1, 1] or rows > 32 or shape[-1] != self.k_tiles * 32
                or x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT
                or x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            raise ValueError("StreamDecodeMatmul: x must be BF16 TILE interleaved [1, 1, rows<=32, K]")
        out = ttnn.empty([1, 1, rows, self.n_out * 32], dtype=self.out_dtype, layout=ttnn.TILE_LAYOUT,
                         device=self.device, memory_config=mc)
        self._cache.run([x, self.weight, out], lambda: self._build(x, out, rows))
        return out
