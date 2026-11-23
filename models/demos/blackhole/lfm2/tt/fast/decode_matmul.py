# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Decode (one tile row) matmuls for LFM2.5 on the P150: DRAM-streaming matmul with bank-contiguous weights.

out[1, 1, rows<=32, N] = x[1, 1, rows, K] @ W^T[K, N] for every decode projection (BFP8 weights, BF16 out with
CK_BFP8; the tied LM head with FP32 out and CK_HIFI4; the BF16 MoE router with FP32 out, bitwise to ttnn.linear),
plus the dense MLP's gate/up projection fused with SwiGLU. Replaces `layers.linear(x, w)` /
`ttnn.linear(x, head, dtype=float32, ...)` for M = 1 tile row (decode only; prefill keeps ttnn.linear).

Program (kernels/decode_matmul_stream.cpp, one generic_op): banks * cores_per_bank workers (the first workers
row-major: one or two rectangles, so the kernel binaries are multicast at launch). Weight tile column block b
(N/8 columns) lives in DRAM bank b; each worker owns a few consecutive output columns and its weight tiles form
ONE contiguous run in its bank (`stream_layout`, a pure tile permutation of the [K, N] tensor, same size, stored
DRAM interleaved; prefill keeps the original layout, so both are resident), which its NCRISC streams in kb x w tile
blocks (transaction-id pipelined). BRISC gathers only the logical rows of x's K tiles (x DRAM
or L1 interleaved; L1 is 4-6 us faster, 27 us for the K=11776 w2) and writes only the logical output rows.
TRISC accumulates K in FP32 DEST with the stock matmul_block / fidelity: whole K (bitwise ttnn.linear for the BF16
outputs measured here), or the stock in0_block_w order (`k_split`, bitwise for FP32 outputs).
Measured: tests/fast/test_decode_matmul.py (logs/fast_decode_matmul.json).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import ttnn

_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "decode_matmul_stream.cpp")
_BANKS = 8  # P150 DRAM banks
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


# Tuned on the P150 with real LFM2.5 weights (tests/fast/test_decode_matmul.py; rep_ms, x L1 interleaved).
PRESETS = {
    "conv_in": Preset(4, 4, 4, 2, 1),  # 2048 x 6144   ShortConv.in_proj
    "conv_out": Preset(4, 2, 8),  # 2048 x 2048   ShortConv.out_proj
    "attn_qkv": Preset(4, 3, 8),  # 2048 x 3072   Attention.wqkv
    "attn_out": Preset(4, 2, 8),  # 2048 x 2048   Attention.wo
    "mlp_w13": Preset(4, 4, 4, 2, 1, glu=True),  # 2048 x 2*11776  DenseMLP SiLU(w1 x) * (w3 x)
    "mlp_w2": Preset(4, 2, 8, 2, 1),  # 11776 x 2048  DenseMLP.w2
    "lm_head": Preset(4, 4, 4, 2, 1),  # 2048 x 128000 (FP32 out, CK_HIFI4)
    "router": Preset(1, 1, 32, k_split=2),  # 2048 x 64 BF16 (FP32 out, CK_HIFI4; bitwise ttnn.linear)
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


def stream_layout(weight: torch.Tensor, cores_per_bank: int, w: int, unit: int = 1, banks: int = _BANKS):
    """[K, N] W^T -> [K, N] tile permutation. Stored DRAM-interleaved (tile page p lives in bank p % banks at bank
    page p // banks), every core's weight tiles are one contiguous run in its bank, ordered by subblock (w columns,
    the last one narrower), then K, then column (consecutive [kb, w] K-major blocks for any kb). Pure tile
    permutation: block-float tiles keep their exact values."""
    k_tiles, n_tiles = weight.shape[-2] // 32, weight.shape[-1] // 32
    src = []
    for cols in stream_columns(n_tiles, cores_per_bank, banks, unit):
        seq = []
        for first, n in cols:
            for s0 in range(0, n, w):
                j = torch.arange(first + s0, first + min(n, s0 + w)).reshape(1, -1)
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
    such as the WeightCache entry; BFP8 values convert exactly). glu presets take W1^T as `weight` and W3^T as `w3`.
    N is zero-padded to a multiple of 8 tiles (one column block per DRAM bank; the 64-wide router -> 256); the
    padding columns are neither read nor written."""
    w = _as_torch(weight)
    if preset.glu:
        w = glu_weight(w, _as_torch(w3), preset.w)
    unit = preset.w if preset.glu else 1
    pad = -w.shape[1] % (32 * _BANKS * unit)
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return stream_layout(w, preset.cores_per_bank, preset.w, unit).contiguous()


def stream_weight(weight, preset: Preset, device=None, dtype=ttnn.bfloat8_b, w3=None):
    """`stream_weight_torch` as a TILE tensor of `dtype`: host (device=None) or DRAM-interleaved on `device`."""
    host = ttnn.from_torch(stream_weight_torch(weight, preset, w3), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return host if device is None else ttnn.to_device(host, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def cached_stream_weight(cache, key: str, preset: Preset, dtype=ttnn.bfloat8_b, key3: str | None = None,
                         build=None, build3=None):
    """Device stream weight through the runtime's WeightCache under f"{key}.{preset.tag}", built once from the
    WeightCache's host file of `key` (+ `key3`, the W3 entry, for glu presets) or, when that file is absent, from
    `build()` / `build3()` (the [K, N] W^T torch the runtime's own cache.get would use); later loads read only
    the new entry."""
    def source(k, b):
        path = cache.path(k, dtype)
        return ttnn.load_tensor(str(path)) if path.exists() else b()

    return cache.get(f"{key}.{preset.tag}", lambda: stream_weight_torch(
        source(key, build), preset, source(key3, build3) if key3 else None), dtype)


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
    n_out: logical output width (default: the weight's; the router passes 64). glu presets: n_out = I and
    out = SiLU(x @ W1) * (x @ W3) in FP32 DEST, rounded once to out_dtype.
    x: BF16 TILE [1, 1, rows, K], DRAM or L1 interleaved; only its logical rows are read. Output: out_dtype TILE,
    interleaved `output_memory_config` (per-call override); only the logical rows are written (the padding rows of
    the output tiles are left undefined, as ttnn padding)."""

    def __init__(self, device, weight, preset: Preset, compute_kernel_config, out_dtype=ttnn.bfloat16,
                 output_memory_config=ttnn.DRAM_MEMORY_CONFIG, n_out: int | None = None):
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
        # per core: bank, run offset (bytes, bank-relative), weight columns, first output column
        self.core_args = []
        for b, (cols, bank_cores) in enumerate(zip(columns, cores)):
            run = 0
            for (first, n), core in zip(cols, bank_cores):
                nv = min(n, n_valid - first)
                if nv > 0:  # workers of zero-padding columns are not instantiated
                    self.core_args.append((core, [b, run * self.tile_bytes, nv, first // 2 if p.glu else first]))
                run += n * self.k_tiles
        if len(self.core_args) < len(columns) * p.cores_per_bank:
            core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _ in self.core_args])
        self.core_set = core_set
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
        per_core = [(c, list(a)) for c, a in self.core_args]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=cs,
            ncrisc_named_compile_time_args=common + [("in1_tile_bytes", self.tile_bytes), ("slots", p.slots),
                                                     ("outstanding", p.outstanding)],
            brisc_named_compile_time_args=common + [("out_tile_bytes", out_tile_bytes)],
            brisc_compile_time_args=accessors,
            trisc_named_compile_time_args=common,
            ncrisc_common_runtime_args=addrs,
            brisc_common_runtime_args=addrs,
            trisc_common_runtime_args=addrs,
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=per_core, brisc_args=per_core, trisc_args=per_core),
        )
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

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
