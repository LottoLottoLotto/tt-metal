# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused attention decode front end (q/k RMSNorm + RoPE + paged KV update) for LFM2 (head_dim 64): one generic_op.

Replaces, in ``layers.Attention.decode`` (between ``qkv = linear(x, wqkv)`` and the SDPA call)::

    q, k, v = nlp_create_qkv_heads(qkv); qn, kn = rms_norm(q), rms_norm(k)        (_heads_norm_rope)
    q, k = rope.apply(qn, cos, sin), rope.apply(kn, cos, sin)                     (matmul R + mul + addcmul)
    for k, v: permute -> to_memory_config(height sharded) -> paged_update_cache   (BF16 rows -> BFP8 cache)
    qd = permute(q)

with ``qd = AttnDecode(attn, rope)(qkv, cur_pos, page_table)``: bitwise the same SDPA query ([1, B, 32, 64] BF16 TILE
DRAM) and the same K/V cache contents (row pos % 64 of block page_table[b][pos // 64], BFP8), for B <= 16 rows.
Cache layout is unchanged ([blocks, 8, 64, 64] BFP8 TILE, prefill's paged_fill_cache keeps writing it).  RoPE
cos/sin come from the module's own [max_pos, 128] table (row p = [cos_p | sin_p with the first half negated], the
same BF16 values as Rope's tables) indexed by cur_pos, so Rope.decode's gathers are not needed for attention;
cur_pos < 0 marks an inactive row (no KV write).

    qkv = linear(x, attn.wqkv)                       # [1, 1, B, 3072] BF16 TILE DRAM
    qd = fast(qkv, cur_pos, page_table)              # writes attn.k_cache / attn.v_cache in place
    out = paged_scaled_dot_product_attention_decode(qd, attn.k_cache, attn.v_cache, ...)  (unchanged)

Kernel: kernels/attn_decode.cpp.  Workers per row b: ``n_q`` query cores (32 / n_q heads each), one key core, one
value core; one row of 11 cores holds the first token.
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from ..layers import rope_tables
from .conv_decode import CachedProgram, cb_descriptors, row_ranges, unified_kernels

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "attn_decode.cpp")
DRAM = ttnn.DRAM_MEMORY_CONFIG
BFP8_TILE = 1088
_TABLES: dict = {}


def rope_table(device, cfg, max_pos: int):
    """[max_pos, 128] BF16 row-major: [cos | sin_signed], sin_signed = [-sin[:, :32], sin[:, 32:]] (shared)."""
    key = (id(device), max_pos)
    if key not in _TABLES:
        cos, sin = rope_tables(cfg, max_pos)
        h = cfg.head_dim // 2
        ssin = torch.cat([-sin[:, :h], sin[:, h:]], dim=-1)
        _TABLES[key] = ttnn.from_torch(torch.cat([cos, ssin], dim=-1).contiguous(), dtype=ttnn.bfloat16,
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
    return _TABLES[key]


class AttnDecode:
    """q/k RMSNorm + RoPE + K/V paged-cache update for one ``layers.Attention`` (constructed once per layer)."""

    def __init__(self, attn, rope, n_q: int = 4):
        cfg = attn.cfg
        if (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads) != (64, 32, 8) or 32 % n_q or 32 // n_q > 16:
            raise ValueError("AttnDecode: head_dim 64, 32 q heads, 8 kv heads; n_q in {2, 4, 8, 16, 32}")
        self.attn, self.device, self.n_q = attn, attn.device, n_q
        self.eps = float(attn.eps)
        qw = ttnn.to_torch(attn.q_norm).reshape(-1).to(torch.bfloat16)
        kw = ttnn.to_torch(attn.k_norm).reshape(-1).to(torch.bfloat16)
        g = torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16)
        g[0, 0, 0] = torch.cat([qw, kw])
        self.gamma = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                                     memory_config=DRAM)
        self.table = rope_table(self.device, cfg, rope.max_pos)
        self._prog = CachedProgram()

    def _build(self, qkv, cur_pos, page_table, q_out, k_cache, v_cache, rows):
        grid = self.device.compute_with_storage_grid_size()
        per_row = self.n_q + 2
        qh = 32 // self.n_q
        tasks = []
        for t in range(rows):
            tasks += [(0, i * qh, qh, t) for i in range(self.n_q)] + [(1, 0, 8, t), (2, 0, 8, t)]
        if len(tasks) > grid.x * grid.y:
            raise ValueError(f"AttnDecode: {rows} rows need {len(tasks)} cores")
        core_list = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(tasks))]
        cores = row_ranges(core_list)
        defines = [("TABLE_PAGE", str(self.table.shape[-1] * 2)), ("POS_PAGE", str(cur_pos.shape[-1] * 4)),
                   ("PT_PAGE", str(page_table.shape[-1] * 4))]
        eps_bits = int(torch.tensor([self.eps], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                               fp32_dest_acc_en=True, dst_full_sync_en=False)
        compute.unpack_to_dest_mode.extend([ttnn.UnpackToDestMode.Default] * 64)
        kernels = unified_kernels(
            KERNEL, cores, defines,
            reader_common=[int(t.buffer_address()) for t in (qkv, self.gamma, self.table, cur_pos, page_table,
                                                             k_cache, v_cache)] + [eps_bits],
            writer_common=[int(t.buffer_address()) for t in (q_out, k_cache, v_cache)],
            per_core=[(c, list(a)) for c, a in zip(core_list, tasks)],
            compute=compute)
        bf16 = [(0, 2), (2, 1), (3, 1), (5, 2), (16, 2), (6, 2), (7, 2), (8, 2), (9, 1), (11, 1), (24, 2)]
        cbs = []
        for i, n in bf16:
            cbs += cb_descriptors([i], n, cores)
        for i, n in ((19, 1), (20, 2), (21, 1), (22, 2)):
            cbs += cb_descriptors([i], n, cores, dtype=ttnn.float32, page=4096)
        cbs += cb_descriptors([10], 16, cores, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
        cbs += cb_descriptors([25], 2, cores, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])

    def __call__(self, qkv, cur_pos, page_table):
        """qkv [1, 1, B, 3072] BF16 TILE DRAM; cur_pos [B] int32; page_table [B, blocks] int32 (both row-major DRAM)
        -> SDPA query [1, B, 32, 64] BF16 TILE DRAM; attn.k_cache / attn.v_cache updated in place."""
        rows = qkv.shape[-2]
        if (tuple(qkv.shape) != (1, 1, rows, 3072) or qkv.dtype != ttnn.bfloat16 or qkv.layout != ttnn.TILE_LAYOUT
                or qkv.memory_config() != DRAM or rows > 16):
            raise ValueError(f"AttnDecode: qkv must be [1,1,B<=16,3072] BF16 TILE DRAM, got {tuple(qkv.shape)}")
        k_cache, v_cache = self.attn.k_cache, self.attn.v_cache
        q_out = ttnn.empty([1, rows, 32, 64], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=DRAM)
        tensors = [qkv, self.gamma, self.table, cur_pos, page_table, k_cache, v_cache, q_out]
        self._prog.run(tensors, lambda: self._build(qkv, cur_pos, page_table, q_out, k_cache, v_cache, rows))
        return q_out
