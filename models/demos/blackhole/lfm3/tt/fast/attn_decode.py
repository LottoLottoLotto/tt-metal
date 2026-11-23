# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused attention decode front end for LFM3 (q/k RMSNorm + RoPE + paged KV update + gate hand-off): one generic_op.

Replaces, in ``layers.Attention.decode`` (between the q|k|v|gate projection and the SDPA call)::

    qkv, g = slices of the qkvg projection                                        (_project)
    q, k, v = nlp_create_qkv_heads(qkv); qn, kn = rms_norm(q), rms_norm(k)        (_heads_norm_rope)
    q, k = rope.apply(qn, cos, sin), rope.apply(kn, cos, sin)                     (matmul R + mul + addcmul)
    for k, v: permute -> to_memory_config(height sharded) -> paged_update_cache   (BF16 rows -> BFP8 cache)
    qd = permute(q)

with ``qd, gate = AttnDecode(attn, rope)(qkvg, cur_pos, page_table)``: bitwise the same SDPA query ([1, B, 16, 128]
BF16 TILE DRAM) and K/V cache contents (row pos % 64 of block page_table[b][pos // 64], BFP8), plus ``gate`` =
[1, 1, B, 32] BF16 TILE DRAM holding the raw gate_proj logits (columns 0..15 = heads; the projection's own values,
sigmoid NOT applied: SDPADecode applies sigmoid(gate) in its epilogue, attn_fused.stock_sdpa_gated runs the stock
Attention._gate on it).  B <= 16 rows; cur_pos < 0 marks an inactive row (no KV write; its outputs are undefined).
Cache layout unchanged ([blocks + 1, 4, 64, 128] BFP8 TILE; prefill's paged_fill_cache keeps writing it).  RoPE
cos/sin are read at cur_pos from the layer type's ``layers.Rope`` row-major tables (Rope.decode's gathers are not
needed for attention).

    qkvg = linear(x, attn.wqkv)                      # [1, 1, B, >= 3104] BF16 TILE DRAM (gate at columns 3072..3087)
    qd, gate = fast(qkvg, cur_pos, page_table)       # writes attn.k_cache / attn.v_cache in place
    flat = SDPADecode(...)(qd, k_cache, v_cache, cur_pos, page_table, gate=gate, window=attn.window)

When q/k/v and gate_proj do not share a format (``attn.fused_gate`` False), pass the separate projection:
``fast(linear(x, attn.wqkv), cur_pos, page_table, gate=linear(x, attn.wg))`` (the gate output is still a copy).

Kernel: kernels/attn_decode.cpp (FRONT_FUSED 0).  Workers per row b (``front_layout``): n_q query cores (16 / n_q heads
each), n_k key cores (4 / n_k heads each: they patch and write 16 / n_k cache tiles), one value core (which also
copies the gate row).
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from .conv_decode import CachedProgram, cb_descriptors, row_ranges, unified_kernels

KERNEL = str(Path(__file__).resolve().parent / "kernels" / "attn_decode.cpp")
DRAM = ttnn.DRAM_MEMORY_CONFIG
BFP8_TILE = 1088
MAX_ROWS = 16
QKV_WIDTH = 3072   # q 2048 | k 512 | v 512
GATE_TILE = QKV_WIDTH // 32  # tile column of the gate logits in the fused qkvg projection


def f32_bits(v: float) -> int:
    return int(torch.tensor([v], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF


def check_attention(attn):
    cfg = attn.cfg
    if (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads) != (128, 16, 4):
        raise ValueError("attention decode kernels: head_dim 128, 16 q heads, 4 kv heads")


def front_cbs(cores, fused: bool, input_cb: bool = True):
    """The front workers' CBs (kernels/attn_decode.cpp); input_cb False: the input tiles (CB FRONT_IN_CB) are
    allocated by the caller."""
    cbs = cb_descriptors([0], 4, cores) if input_cb else []
    for i, n in ((2, 1), (3, 1), (5, 4), (16, 4), (6, 4), (7, 4), (9, 1), (11, 1), (24, 4)):
        cbs += cb_descriptors([i], n, cores)
    if not fused:
        cbs += cb_descriptors([8], 2, cores)
    for i, n in ((19, 1), (20, 4), (21, 1), (22, 4)):
        cbs += cb_descriptors([i], n, cores, dtype=ttnn.float32, page=4096)
    cbs += cb_descriptors([10], 16, cores, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
    cbs += cb_descriptors([25], 4, cores, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
    return cbs


def front_compute():
    compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, dst_full_sync_en=True)
    compute.unpack_to_dest_mode.extend([ttnn.UnpackToDestMode.Default] * 64)
    return compute


# (q workers, k workers) per row, in order of preference.  Each k worker patches and writes 16 / n_k cache tiles; the
# compute per worker is the same 4 head tiles whatever its head count.  AttnDecode gathers its head rows from DRAM
# (latency per row read: more q workers help); the fused fronts receive them (measured best: (2, 2) at B = 1..16,
# logs/fast_attn_fused_r2.json "front layouts").
LAYOUTS = ((4, 4), (4, 2), (4, 1), (2, 2), (2, 1), (1, 1))
FUSED_LAYOUTS = ((2, 2), (2, 1), (1, 1))


def front_layout(rows: int, free: int, layout=None, prefer=LAYOUTS):
    """(n_q, n_k) for `rows` rows on `free` cores: `layout` if given (checked), else the first of `prefer` that
    fits."""
    for nq, nk in [layout] if layout else prefer:
        if 16 % nq or 4 % nk:
            raise ValueError(f"front layout {layout}: n_q must divide 16, n_k 4")
        if rows * (nq + nk + 1) <= free:
            return nq, nk
    raise ValueError(f"front: {rows} rows do not fit {free} cores (layout {layout})")


def front_tasks(rows: int, n_q: int, n_k: int):
    """Per front worker, token-major: (role, first head, heads, token); n_q q workers, n_k k workers, 1 v worker."""
    qh, kh = 16 // n_q, 4 // n_k
    tasks = []
    for t in range(rows):
        tasks += ([(0, i * qh, qh, t) for i in range(n_q)] + [(1, i * kh, kh, t) for i in range(n_k)]
                  + [(2, 0, 4, t)])
    return tasks


class AttnDecode:
    """q/k RMSNorm + RoPE + K/V paged-cache update + gate copy for one ``layers.Attention`` (constructed once per
    layer).  attn: layers.Attention (caches bound via set_kv_cache); rope: the layer type's layers.Rope (model.rope[
    layer type]).  layout: (n_q, n_k) workers per row (default: the first of LAYOUTS that fits the grid)."""

    max_rows = MAX_ROWS

    def __init__(self, attn, rope, layout=None):
        check_attention(attn)
        self.attn, self.rope, self.device, self.layout = attn, rope, attn.device, layout
        self.eps = float(attn.eps)
        qw = ttnn.to_torch(attn.q_norm).reshape(-1).to(torch.bfloat16)
        kw = ttnn.to_torch(attn.k_norm).reshape(-1).to(torch.bfloat16)
        g = torch.zeros(1, 1, 32, 256, dtype=torch.bfloat16)
        g[0, 0, 0] = torch.cat([qw, kw])
        self.gamma = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                                     memory_config=DRAM)
        self._progs = {}

    def reader_common(self, qkv, gsrc, gate, cur_pos, page_table):
        """Reader common args of kernels/attn_decode.cpp (qkv / gsrc None in the fused mode)."""
        a = self.attn
        addr = lambda t: int(t.buffer_address()) if t is not None else 0  # noqa: E731
        return [addr(qkv), addr(self.gamma), addr(self.rope.cos_rm), addr(self.rope.sin_rm), addr(cur_pos),
                addr(page_table), addr(a.k_cache), addr(a.v_cache), f32_bits(self.eps), addr(gsrc), addr(gate)]

    def _build(self, qkv, gsrc, gtile, cur_pos, page_table, q_out, gate, rows):
        grid = self.device.compute_with_storage_grid_size()
        n_q, n_k = front_layout(rows, grid.x * grid.y, self.layout)
        tasks = front_tasks(rows, n_q, n_k)
        core_list = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(tasks))]
        cores = row_ranges(core_list)
        defines = [("POS_PAGE", str(cur_pos.shape[-1] * 4)), ("PT_PAGE", str(page_table.shape[-1] * 4)),
                   ("FRONT_FUSED", "0"), ("GATE_TILE", str(gtile))]
        kernels = unified_kernels(
            KERNEL, cores, defines,
            reader_common=self.reader_common(qkv, gsrc, gate, cur_pos, page_table),
            writer_common=[int(t.buffer_address()) for t in (q_out, self.attn.k_cache, self.attn.v_cache)],
            per_core=[(c, list(a) + [0]) for c, a in zip(core_list, tasks)],
            compute=front_compute())
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=front_cbs(cores, fused=False), semaphores=[])

    def __call__(self, qkv, cur_pos, page_table, gate=None):
        """qkv [1, 1, B, W] BF16 TILE DRAM: the qkvg projection (W >= 3104, e.g. 3104 or a stream matmul's padded
        3328) or the plain qkv projection (W == 3072) together with `gate` = its [1, 1, B, 32] gate projection;
        cur_pos [B] int32; page_table [B, blocks] int32 (both row-major DRAM) -> (q [1, B, 16, 128] BF16 TILE DRAM,
        gate [1, 1, B, 32] BF16 TILE DRAM raw logits); attn.k_cache / attn.v_cache updated in place."""
        rows = qkv.shape[-2]
        w = qkv.shape[-1]
        if (len(qkv.shape) != 4 or tuple(qkv.shape)[:2] != (1, 1) or qkv.dtype != ttnn.bfloat16
                or qkv.layout != ttnn.TILE_LAYOUT or qkv.memory_config() != DRAM or not 1 <= rows <= MAX_ROWS):
            raise ValueError(f"AttnDecode: qkv must be [1,1,B<=16,W] BF16 TILE DRAM, got {tuple(qkv.shape)}")
        if w == QKV_WIDTH:
            if gate is None or tuple(gate.shape) != (1, 1, rows, 32) or gate.memory_config() != DRAM:
                raise ValueError("AttnDecode: a 3072-wide qkv needs gate=[1,1,B,32] BF16 TILE DRAM")
            gsrc, gtile = gate, 0
        elif w >= QKV_WIDTH + 16 and gate is None:
            gsrc, gtile = qkv, GATE_TILE
        else:
            raise ValueError(f"AttnDecode: qkv width {w} (want >= 3104, or 3072 with gate=)")
        q_out = ttnn.empty([1, rows, 16, 128], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=DRAM)
        g_out = ttnn.empty([1, 1, rows, 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=DRAM)
        a = self.attn
        tensors = [qkv, self.gamma, self.rope.cos_rm, self.rope.sin_rm, cur_pos, page_table, a.k_cache, a.v_cache,
                   q_out, g_out] + ([gsrc] if gsrc is not qkv else [])
        prog = self._progs.setdefault(gtile, CachedProgram())
        prog.run(tensors, lambda: self._build(qkv, gsrc, gtile, cur_pos, page_table, q_out, g_out, rows))
        return q_out, g_out
