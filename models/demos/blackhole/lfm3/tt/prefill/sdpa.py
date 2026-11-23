# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM3 prefill attention in one program (kernels/sdpa_prefill.cpp): causal or sliding-window GQA attention of a
prefill chunk over the paged BFP8 KV cache, with the per-head sigmoid gate and the head concat in the epilogue.

Replaces, per attention layer, the full-layer chunked_scaled_dot_product_attention (or the sliding layers' left-window
gather + zero query rows + windowed SDPA) + nlp_concat_heads + Attention._gate:

    out = PrefillSDPA.for_attention(attn)(q, attn.k_cache, attn.v_cache, page_table, start_tensor, gate=qkvg)
    # out [1, 1, Tp, 2048] BF16 = concat_h(softmax(q_h K^T / sqrt(128)) V) * sigmoid(gate[:, h]) -> o_proj

Design (details in the kernel header): a work unit is (kv head, RT query row tiles) with the group's 4 q heads stacked,
so K/V are read once per 4 heads (the stock op reads them once per head and q chunk); the units' key tiles are cut
into equal contiguous shares over the cores (split units are reduced through a DRAM scratch slot per core: balanced
causal triangle, no idle tail); sliding layers read only the window's keys straight from the cache pages.  Every
matmul product is exact (transposed formulation at HiFi2: BFP8 K / V in srcB, BF16 Q / P in srcA), scores stay FP32
in DEST up to the exp, the softmax numerator / denominator accumulate in FP32.  No running max: the logits are bounded
by |q.k| <= head_dim * max|q_norm.weight| * max|k_norm.weight| (RMSNorm outputs, RoPE is a rotation), so
exp(s * scale - OFFSET) stays inside the FP32 range for every key; the constructor checks the bound.  The exp of the
unmasked key tiles is an SFPLOADMACRO pipeline (5 issued SFPU instructions per row; SFPU and matmul instructions
contend for the core, so the exp's instruction count sets the matmul speed too); full layers stack 4 query row tiles
per unit for chunks of >= 1024 rows (K/V read once per 16 query tiles), 2 for shorter chunks.

Programs depend only on (Tp, page-table width, gate tensor width); the chunk start is read from the device tensor.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import ttnn

from ..fast.conv_decode import CachedProgram, cb_descriptors, row_ranges, unified_kernels

KERNEL = str(Path(__file__).resolve().parent.parent / "kernels" / "sdpa_prefill.cpp")
DRAM = ttnn.DRAM_MEMORY_CONFIG
N_HEADS, N_KV, HEAD_DIM, BLOCK = 16, 4, 128, 64
GATE_COL = 3072  # gate logits of head h at column 3072 + h of the qkvg projection
BFP8_TILE = 1088
# exp(s * scale - OFFSET): with |s * scale| <= B the largest P is e^(B - OFFSET) and the row maximum is at least
# e^(-B - OFFSET); B <= MAX_BOUND keeps sums of 2^17 keys x |V| far below FP32 max (e^(B - 6) <= e^60) and every
# row's leading terms normal (>= e^-72, 1e-31) with room for 1e-7 relative weights.
OFFSET = 6.0
MAX_BOUND = 66.0
# masked key tiles (diagonal / window edge) exp polynomial: 2^f = 1 + f (K1 + f (K2 + f K3)), f in [0, 1) (through 1
# and 2 - 2^-20; rel. error 1.03e-4); the kernel's f is the 23-bit integer fraction, so K_i carry 2^(-23 i)
_POLY = (0.695422379732132, 0.2263160648345947, 0.07826060175895694)
# unmasked tiles: t = s * scale * log2 e + EXP_M (EXP_M = 2175 - OFFSET log2 e) in [2048, 4096) for |s scale| <= 66,
# so t's mantissa is z + 127 (z = log2 P) in fixed point with 12 fraction bits; 2^f ~ ((Q3 v + Q2) v + Q1) v + Q0 for
# v = 1 + f: the relative-minimax cubic on the 2^14 grid values of v with p(1) = 1 (max rel. error 8.6e-5; FP32
# Horner keeps p in [1, 2) on every grid value: end to end rel. error <= 2.3e-4 incl. the 2^-12 fraction rounding)
_QPOLY = (0.45546457171440125, 0.471019983291626, -0.003550118999555707, 0.07706556469202042)


def logit_bound(q_gamma: torch.Tensor, k_gamma: torch.Tensor, head_dim: int = HEAD_DIM, margin: float = 1.05) -> float:
    """Upper bound of |q.k| for q = RMSNorm(x) * q_gamma, k = RMSNorm(y) * k_gamma (|RMSNorm(x)|^2 <= head_dim per
    head, RoPE preserves norms); `margin` covers BF16 / BFP8 rounding of q and the cached k."""
    return head_dim * float(q_gamma.float().abs().max()) * float(k_gamma.float().abs().max()) * margin


def _f32(x: float) -> str:
    return f"{x:.9e}f"


_CONSTS: dict = {}
_SCRATCH: dict = {}


def _consts(device):
    """[1, 1, 32, 32 * 20] BF16 TILE DRAM: zeros, ones, the diagonal mask (S^T layout: key row u, query column v;
    -1e30 where u > v), the window-edge mask (-1e30 where u <= v), E_0 .. E_15 (E_c: column c ones)."""
    key = id(device)
    if key not in _CONSTS:
        u = torch.arange(32)[:, None]
        v = torch.arange(32)[None, :]
        big = -1e30
        tiles = [torch.zeros(32, 32), torch.ones(32, 32), torch.where(u > v, big, 0.0), torch.where(u <= v, big, 0.0)]
        for c in range(16):
            e = torch.zeros(32, 32)
            e[:, c] = 1.0
            tiles.append(e)
        host = torch.cat(tiles, dim=1).reshape(1, 1, 32, -1).to(torch.bfloat16)
        _CONSTS[key] = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                       memory_config=DRAM)
    return _CONSTS[key]


def _scratch(device, tiles: int):
    """FP32 [1, 1, 32 tiles, 32] DRAM scratch shared by every PrefillSDPA program on the device (split-unit partials,
    produced and consumed within one program)."""
    key = id(device)
    t = _SCRATCH.get(key)
    if t is None or t.shape[-2] < tiles * 32:
        if t is not None:
            ttnn.deallocate(t)
        t = ttnn.empty([1, 1, tiles * 32, 32], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
                       memory_config=DRAM)
        _SCRATCH[key] = t
    return t


class PrefillSDPA:
    """Gated GQA prefill attention of one LFM3 attention layer type (16 q / 4 kv heads, head_dim 128, BFP8 paged
    K/V in blocks of 64 tokens).  window: None (full causal) or the sliding window (multiple of 32: key j visible to
    query i iff i - window < j <= i).  qk_bound: the layer's |q.k| bound (logit_bound / for_attention).

    rows: query row tiles per head per work unit (default: full layers 4 for Tp >= 1024 (16 query tiles share each K/V
    read, ~2.2 GB of K/V per layer at 128K) else 2; sliding layers 1).  k_chunk: key tiles resident in L1 per step.
    cores: grid cores used (default all).  min_seg: smallest per-core share in key tiles (tiny chunks use fewer
    cores)."""

    def __init__(self, device, window: int | None = None, *, qk_bound: float, rows: int | None = None,
                 k_chunk: int | None = None, cores: int | None = None, min_seg: int = 4,
                 scale: float = HEAD_DIM ** -0.5):
        if window is not None and (window <= 0 or window % 32):
            raise ValueError("PrefillSDPA: window must be a positive multiple of 32")
        bound = qk_bound * scale
        if bound > MAX_BOUND:
            raise ValueError(f"PrefillSDPA: logit bound {bound:.1f} > {MAX_BOUND} (the fixed softmax offset needs "
                             "|q.k| * scale within the FP32 exp range)")
        self.device, self.window, self.scale, self.bound = device, window, scale, bound
        self.rows = rows
        # sliding: one chunk holds a unit's whole window (window / 32 + rows key tiles)
        self.k_chunk = k_chunk or ((window // 32 + (rows or 1)) if window else 16)
        self.min_seg = min_seg
        grid = device.compute_with_storage_grid_size()
        self.cores = min(cores or grid.x * grid.y, grid.x * grid.y)
        self.consts = _consts(device)
        self._scratch_tiles = self.cores * 5 * 4 * (rows or 4)
        self._progs = {}

    @classmethod
    def for_attention(cls, attn, **kw):
        """From a layers.Attention (window, q_norm / k_norm gammas read back once)."""
        qg = ttnn.to_torch(attn.q_norm).float()
        kg = ttnn.to_torch(attn.k_norm).float()
        return cls(attn.device, attn.window, qk_bound=logit_bound(qg, kg, attn.hd), **kw)

    def _rows_for(self, Tp: int) -> int:
        """Query row tiles per unit: `rows` if given, else 1 (sliding) / 4 for Tp >= 1024 and 2 below (full layers:
        more K/V reuse per unit, but a split unit's finalizer adds (cores per unit) x 5 x 4 rows partial tiles, which
        dominates short chunks; measured sweep in src/NOTES.md section 16)."""
        rows = self.rows or (1 if self.window else (4 if Tp >= 1024 else 2))
        rt = min(rows, Tp // 32)
        while (Tp // 32) % rt:
            rt -= 1
        return rt

    def _build(self, q, k_cache, v_cache, page_table, start, gate, scratch, out, Tp, rt):
        dev = self.device
        grid = dev.compute_with_storage_grid_size()
        nc = self.cores
        core_list = [ttnn.CoreCoord(n % grid.x, n // grid.x) for n in range(nc)]
        cores = row_ranges(core_list)
        kc = self.k_chunk
        qt = 4 * rt
        ptw = int(page_table.shape[-1])
        nu = 4 * (Tp // 32) // rt
        gate_w = int(gate.shape[-1]) // 32 if gate is not None else 0
        gate_col = (GATE_COL // 32 if gate_w * 32 >= GATE_COL + 32 else 0) if gate is not None else 0
        log2e = 1.0 / math.log(2.0)
        defines = [("SP_TPT", str(Tp // 32)), ("SP_RT", str(rt)), ("SP_KC", str(kc)),
                   ("SP_WT", str((self.window or 0) // 32)), ("SP_NC", str(nc)), ("SP_MINSEG", str(self.min_seg)),
                   ("SP_GATE", "1" if gate is not None else "0"), ("SP_GATE_W", str(max(gate_w, 1))),
                   ("SP_GATE_COL", str(gate_col)), ("SP_PT_W", str(ptw)),
                   ("EXP_A", _f32(self.scale * log2e)), ("EXP_B", _f32(127.0 - OFFSET * log2e)),
                   ("EXP_K1", _f32(_POLY[0] * 2.0 ** -23)), ("EXP_K2", _f32(_POLY[1] * 2.0 ** -46)),
                   ("EXP_K3", _f32(_POLY[2] * 2.0 ** -69)), ("EXP_M", _f32(2175.0 - OFFSET * log2e))]
        defines += [(f"EXP_Q{i}", _f32(c)) for i, c in enumerate(_QPOLY)]
        if os.environ.get("LFM_SP_DEBUG"):  # kernel progress prints (DPRINT; debugging only)
            defines.append(("SP_DEBUG", "1"))
        if os.environ.get("LFM_SP_PROF"):  # per-thread phase timers (profiling only: tests/prefill/sdpa_prof.py)
            defines.append(("SP_PROF", "1"))
        noc = [dev.worker_core_from_logical_core(c) for c in core_list]
        common = [int(t.buffer_address()) for t in (q, k_cache, v_cache, page_table, start, self.consts, out,
                                                     gate if gate is not None else q, scratch)]
        common += [v for c in noc for v in (c.x, c.y)]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                               fp32_dest_acc_en=True, dst_full_sync_en=False)
        unpack = [ttnn.UnpackToDestMode.Default] * 64
        for i in (16, 17, 18, 19):
            unpack[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode.extend(unpack)
        kernels = unified_kernels(KERNEL, cores, defines, reader_common=common, writer_common=common,
                                  per_core=[(c, [i]) for i, c in enumerate(core_list)], compute=compute)
        bf16, bfp8, f32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32

        def cb(index, tiles, dtype=bf16, page=2048):
            return cb_descriptors((index,), tiles, cores, dtype=dtype, page=page)

        def raw(index, size):
            size = (size + 63) // 64 * 64
            return [ttnn.CBDescriptor(total_size=size, core_ranges=cores,
                                      format_descriptors=[ttnn.CBFormatDescriptor(index, bf16, size)])]

        plan_bytes = 4 * (2 + 4 * nu)
        cbs = (cb(0, 16) + cb(1, 2 * kc * 4, bfp8, BFP8_TILE) + cb(2, 2 * kc * 4, bfp8, BFP8_TILE)
               + cb(3, kc * 4) + cb(4, 2 * kc * 4) + cb(5, 1) + cb(6, 1) + cb(7, 2) + cb(8, 16)
               + cb(9, rt)
               + raw(10, plan_bytes) + cb(11, 4) + cb(12, 8) + raw(13, 64 + ptw * 4 + 128)
               + raw(14, 64 + plan_bytes) + raw(20, 64) + raw(21, 64) + cb(22, qt * 4)
               + cb(16, qt * 4, f32, 4096) + cb(17, qt, f32, 4096) + cb(18, 1, f32, 4096) + cb(19, 8, f32, 4096))
        l1 = sum(c.total_size for c in cbs)
        if l1 > 1_350_000:
            raise ValueError(f"PrefillSDPA: {l1} B of L1 per core (rows {rt}, k_chunk {kc}, page table {ptw})")
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)

    def __call__(self, q, k_cache, v_cache, page_table, start_tensor, gate=None):
        """q [1, 16, Tp, 128] BF16 TILE DRAM (post q_norm + RoPE); k_cache / v_cache [blocks, 4, 64, 128] BFP8 holding
        positions 0 .. start + Tp - 1 of this sequence (page_table [1, W] int32 ROW_MAJOR DRAM: page i = positions
        64 i .. 64 i + 63); start_tensor [1] int32 device tensor (multiple of 32); gate: None or the qkvg projection
        [1, 1, Tp, >= 3104] (or [1, 1, Tp, 32] logits) BF16 TILE DRAM.  -> [1, 1, Tp, 2048] BF16 TILE DRAM (rows >= the
        chunk's valid length: padding, causal-harmless)."""
        Tp = int(q.shape[-2])
        if tuple(q.shape) != (1, N_HEADS, Tp, HEAD_DIM) or q.dtype != ttnn.bfloat16 or Tp % 32:
            raise ValueError(f"PrefillSDPA: q must be [1, 16, Tp, 128] BF16, got {tuple(q.shape)}")
        if tuple(k_cache.shape)[1:] != (N_KV, BLOCK, HEAD_DIM) or k_cache.dtype != ttnn.bfloat8_b:
            raise ValueError("PrefillSDPA: caches must be [blocks, 4, 64, 128] BFP8")
        if gate is not None and (tuple(gate.shape)[:3] != (1, 1, Tp) or gate.dtype != ttnn.bfloat16
                                 or int(gate.shape[-1]) not in (32,) and int(gate.shape[-1]) < GATE_COL + 32):
            raise ValueError(f"PrefillSDPA: gate must be [1, 1, {Tp}, >= 3104] BF16, got {tuple(gate.shape)}")
        rt = self._rows_for(Tp)
        out = ttnn.empty([1, 1, Tp, N_HEADS * HEAD_DIM], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=self.device, memory_config=DRAM)
        key = (Tp, int(page_table.shape[-1]), None if gate is None else int(gate.shape[-1]))
        prog = self._progs.setdefault(key, CachedProgram())
        scratch = _scratch(self.device, self._scratch_tiles)
        tensors = [q, k_cache, v_cache, page_table, start_tensor, self.consts, scratch]
        if gate is not None:
            tensors.append(gate)
        tensors.append(out)
        prog.run(tensors, lambda: self._build(q, k_cache, v_cache, page_table, start_tensor, gate, scratch, out, Tp,
                                              rt))
        return out
