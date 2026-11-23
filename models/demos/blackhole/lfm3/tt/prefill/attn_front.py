# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused attention prefill front end for LFM3 (q/k RMSNorm + RoPE + paged K/V fill): one generic_op per chunk.

Replaces, in ``layers.Attention.prefill`` (between the q|k|v|gate projection and the SDPA)::

    qkv = slice(y, 0..3072)                                                     (_project; the gate slice stays)
    q, k, v = nlp_create_qkv_heads(qkv); qn, kn = rms_norm(q), rms_norm(k)      (_heads_norm_rope)
    cos, sin = rope.prefill(start, Tp); q, k = rope.apply(qn, ..), rope.apply(kn, ..)
    paged_fill_cache(k_cache, typecast(k, BFP8), chunk_page_table); same for v

with ``q = AttnFrontPrefill(att, rope)(y, start_tensor, chunk_page_table)``: bitwise the same K/V cache contents and q
([1, 16, Tp, 128] BF16 TILE DRAM), except q elements below ~3e-36 (the full layers' dims whose q_norm weight is
~1e-33: the HiFi4 product (x * rstd) * gamma falls below 2^-118 there and its result can differ by 2^-126 between
programs; tests/prefill/attn_front.py).  The program depends only on Tp (and the tensor
addresses): the chunk start is read at run time from the [1] int32 device tensor (model.prefill's start_tensor),
the rope rows from the layer type's tiled tables (Rope.cos / Rope.sin), the cache blocks from the chunk page table
([1, Tp / 64] int32 row-major, entry i = block of positions start + 64 i ..; padded rows go wherever it points: the
trash block in production).  Chunk starts must be multiples of the 64-token block (as paged_fill_cache's).

    y = linear(x, att.wqkv)                           # [1, 1, Tp, 3104] BF16 TILE DRAM (gate logits at 3072..3087)
    q = front(y, start_tensor, chunk_page_table)      # writes att.k_cache / att.v_cache in place

Kernel: kernels/attn_front_prefill.cpp.  Work units (row tile, head) over the whole compute grid in contiguous
weighted ranges (q/k units norm + rope, v units only convert to BFP8).
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from ..fast.attn_decode import QKV_WIDTH, check_attention, f32_bits, front_compute
from ..fast.conv_decode import CachedProgram, cb_descriptors, row_ranges, unified_kernels

KERNEL = str(Path(__file__).resolve().parent.parent / "kernels" / "attn_front_prefill.cpp")
DRAM = ttnn.DRAM_MEMORY_CONFIG
BFP8_TILE = 1088
BLOCK = 64            # KV page (tokens)
UNITS_PER_ROW = 24    # 16 q heads, 4 k heads, 4 v heads per 32-row tile
NORM_UNITS = 20       # q and k units (norm + rope)
V_WEIGHT = 0.25       # relative cost of a v unit (copy + BFP8 pack) in the core split
IO_TILES = 8          # input / output CB depth (tiles; 4 per unit)


def split_units(row_tiles: int, n_cores: int, v_weight: float = V_WEIGHT) -> list[tuple[int, int]]:
    """Contiguous unit ranges [u0, u1) of about equal weight (empty ranges dropped)."""
    w = [1.0 if j < NORM_UNITS else v_weight for _ in range(row_tiles) for j in range(UNITS_PER_ROW)]
    total = sum(w)
    bounds, acc, k = [0], 0.0, 1
    for i, wi in enumerate(w):
        acc += wi
        while k < n_cores and acc >= total * k / n_cores - 1e-9:
            bounds.append(i + 1)
            k += 1
    bounds += [len(w)] * (n_cores + 1 - len(bounds))
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _compute():
    """HiFi4, FP32 DEST, full-sync DEST (the decode front's config) + precise BFP8 packing (ttnn.typecast's
    bfp8_pack_precise: round half away from zero; the default FP32-DEST packer mode does not reproduce typecast's
    BFP8 values)."""
    cfg = front_compute()
    cfg.bfp8_pack_precise = True
    return cfg


def _cbs(cores):
    cbs = []
    for i, n in ((0, IO_TILES), (2, 1), (3, 1), (5, 8), (6, 8), (7, 8), (11, 1), (16, 4), (24, IO_TILES)):
        cbs += cb_descriptors([i], n, cores)
    for i, n in ((19, 1), (20, 4), (21, 1), (22, 4)):
        cbs += cb_descriptors([i], n, cores, dtype=ttnn.float32, page=4096)
    cbs += cb_descriptors([25], IO_TILES, cores, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
    return cbs


class AttnFrontPrefill:
    """q/k RMSNorm + RoPE + K/V paged-cache fill of one prefill chunk for one ``layers.Attention`` (constructed once
    per layer).  att: layers.Attention (caches bound via set_kv_cache); rope: the layer type's layers.Rope
    (model.rope[layer type]).  n_cores: worker cores (default: the whole compute grid)."""

    def __init__(self, att, rope, n_cores: int | None = None):
        check_attention(att)
        self.att, self.rope, self.device = att, rope, att.device
        grid = self.device.compute_with_storage_grid_size()
        self.grid_x = grid.x
        self.n_cores = n_cores or grid.x * grid.y
        if not 1 <= self.n_cores <= grid.x * grid.y:
            raise ValueError(f"AttnFrontPrefill: n_cores {n_cores} outside the {grid.x}x{grid.y} grid")
        self.eps = float(att.eps)
        qw = ttnn.to_torch(att.q_norm).reshape(-1).to(torch.bfloat16)
        kw = ttnn.to_torch(att.k_norm).reshape(-1).to(torch.bfloat16)
        g = torch.zeros(1, 1, 32, 256, dtype=torch.bfloat16)
        g[0, 0, 0] = torch.cat([qw, kw])
        self.gamma = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                                     memory_config=DRAM)
        self.table_tiles = rope.cos.shape[-2] // 32
        self._progs: dict[int, CachedProgram] = {}

    def _build(self, y, start_tensor, cpt, q):
        a = self.att
        Tp = y.shape[-2]
        R = Tp // 32
        ranges = split_units(R, self.n_cores, V_WEIGHT)
        core_list = [ttnn.CoreCoord(i % self.grid_x, i // self.grid_x) for i in range(len(ranges))]
        cores = row_ranges(core_list)
        cpt_bytes = cpt.shape[-1] * 4
        defines = [("Y_WT", str(y.padded_shape[-1] // 32)), ("ROW_TILES", str(R)),
                   ("CPT_READ", str(-(-cpt_bytes // 64) * 64))]
        addr = lambda t: int(t.buffer_address())  # noqa: E731
        kernels = unified_kernels(
            KERNEL, cores, defines,
            reader_common=[addr(y), addr(self.gamma), addr(self.rope.cos), addr(self.rope.sin), addr(start_tensor),
                           self.table_tiles, f32_bits(self.eps)],
            writer_common=[addr(q), addr(a.k_cache), addr(a.v_cache), addr(cpt)],
            per_core=[(c, list(r)) for c, r in zip(core_list, ranges)],
            compute=_compute())
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=_cbs(cores), semaphores=[])

    def __call__(self, y, start_tensor, chunk_page_table):
        """y [1, 1, Tp, W] BF16 TILE DRAM (W >= 3072: the qkv(g) projection), Tp a multiple of 64; start_tensor [1]
        int32 (the chunk start, a multiple of 64); chunk_page_table [1, >= Tp / 64] int32 row-major DRAM -> q
        [1, 16, Tp, 128] BF16 TILE DRAM; att.k_cache / att.v_cache blocks of the chunk written in place."""
        Tp = y.shape[-2]
        if (len(y.shape) != 4 or tuple(y.shape)[:2] != (1, 1) or y.shape[-1] < QKV_WIDTH or Tp % BLOCK
                or y.dtype != ttnn.bfloat16 or y.layout != ttnn.TILE_LAYOUT or y.memory_config() != DRAM):
            raise ValueError(f"AttnFrontPrefill: y must be [1,1,Tp%64==0,>=3072] BF16 TILE DRAM, got {tuple(y.shape)}")
        if (chunk_page_table.dtype != ttnn.int32 or chunk_page_table.layout != ttnn.ROW_MAJOR_LAYOUT
                or chunk_page_table.shape[-1] < Tp // BLOCK or chunk_page_table.memory_config() != DRAM):
            raise ValueError(f"AttnFrontPrefill: chunk_page_table must be [1, >= {Tp // BLOCK}] int32 row-major DRAM")
        if start_tensor.dtype != ttnn.int32 or start_tensor.memory_config() != DRAM:
            raise ValueError("AttnFrontPrefill: start_tensor must be a [1] int32 DRAM tensor")
        a = self.att
        q = ttnn.empty([1, 16, Tp, 128], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=DRAM)
        tensors = [y, self.gamma, self.rope.cos, self.rope.sin, start_tensor, chunk_page_table, a.k_cache, a.v_cache,
                   q]
        prog = self._progs.setdefault(Tp, CachedProgram())
        prog.run(tensors, lambda: self._build(y, start_tensor, chunk_page_table, q))
        return q
