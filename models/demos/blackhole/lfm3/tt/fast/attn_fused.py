# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM3 attention decode: fused qkvg + front end (optionally with the sandwich norm), custom paged SDPA decode with the
head gate, stock-SDPA fallback.

AttnQKVFront: the qkvg projection (StreamDecodeMatmul preset attn_qkv on the stream copy of `L{i}.attn.qkvg`) and the
AttnDecode front end (q/k RMSNorm, RoPE, paged BFP8 K/V write, SDPA query, gate logits) in ONE program.  The matmul
workers (kernels/attn_qkv_mm.cpp: decode_matmul_stream.cpp's weight stream and math) send each output column's token
rows straight into the input tiles of the front workers (kernels/attn_decode.cpp, FRONT_FUSED 1), which meanwhile
read their position, RoPE rows, page-table entry and the cache tiles to patch; the worker of the gate column writes
the gate logits.  Same values as StreamDecodeMatmul + AttnDecode, so bitwise the same q, gate and K/V caches.

AttnNormQKVFront: the same program with NormStreamDecodeMatmul's prologue on the matmul rectangle (sandwich residual
h = a + post(b) + operator RMSNorm while the weight streams): (h, q, gate) bitwise ResidualRMSNorm(post_weight) +
AttnQKVFront.

SDPADecode: paged causal (or sliding-window) SDPA decode for 1..16 rows with the per-head sigmoid gate applied in its
epilogue; output = the flattened, gated attention [1, 1, B, 2048] that o_proj takes.

Decode call sequence (FastStep):

    h, qd, gate = AttnNormQKVFront(...)(a, b, cur_pos, page_table)   # or: _, n = op_rn(a, b);
                                                                     #     qd, gate = AttnQKVFront(...)(n, cur_pos, pt)
                                                                     # or: qkvg = matmul(n); qd, gate = AttnDecode(...)(
                                                                     #     qkvg, cur_pos, page_table)
    flat = SDPADecode(dev, rows=B)(qd, attn.k_cache, attn.v_cache, cur_pos, page_table, gate=gate, window=attn.window)
    #   (fallback: flat = stock_sdpa_gated(attn, qd, gate, cur_pos, page_table))
    y = StreamDecodeMatmul(..., PRESETS["attn_out"], ...)(flat)
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from ..common import CK_BFP8, free
from ..layers import DRAM, sdpa_compute
from .attn_decode import (BFP8_TILE, FUSED_LAYOUTS, MAX_ROWS, AttnDecode, check_attention, front_cbs, front_compute,
                          front_layout, front_tasks)
from .conv_decode import CachedProgram, cb_descriptors, row_ranges, unified_kernels
from .decode_matmul import PRESETS, StreamDecodeMatmul, cached_stream_weight

KERNELS = Path(__file__).resolve().parent / "kernels"
QKVG_OUT = 3104  # q 2048 | k 512 | v 512 | gate 16 + 16 zero columns
L1 = ttnn.L1_MEMORY_CONFIG


def _slot_of(col: int, n_q: int, n_k: int) -> int | None:
    """qkvg output tile column -> front worker slot (q 0..n_q-1, k n_q..n_q+n_k-1, v n_q+n_k); None: gate column."""
    h = col // 4
    if h >= 24:
        return None
    return h // (16 // n_q) if h < 16 else (n_q + (h - 16) // (4 // n_k) if h < 20 else n_q + n_k)


def _check_x(x, name):
    rows = x.shape[-2]
    if (tuple(x.shape) != (1, 1, rows, 2048) or x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT
            or not 1 <= rows <= MAX_ROWS):
        raise ValueError(f"{name}: [1,1,B<=16,2048] BF16 TILE expected, got {tuple(x.shape)}")
    return rows


class _QKVFrontBase:
    """Shared program assembly of AttnQKVFront / AttnNormQKVFront: matmul workers (kernels/attn_qkv_mm.cpp) on
    self.mm's cores, front workers (kernels/attn_decode.cpp, FRONT_FUSED 1) on free cores."""

    max_rows = MAX_ROWS
    verify = False  # FRONT_VERIFY (AttnNormQKVFront(verify=True, vmap=...)): rows = groups of consecutive tokens
    vmap = None  # the verify row map ([1, 16] uint32 ROW_MAJOR DRAM; reader common arg 11)

    def _init_front(self, attn, rope, layout):
        check_attention(attn)
        if not attn.fused_gate:
            raise ValueError(f"{type(self).__name__}: needs the fused q|k|v|gate weight (attn.fused_gate)")
        self.attn, self.device, self.layout = attn, attn.device, layout
        self.preset = p = PRESETS["attn_qkv"]
        if p.k_split or p.glu:
            raise ValueError(f"{type(self).__name__}: the attn_qkv preset must be whole-K, non-glu")
        self.front = AttnDecode(attn, rope)  # gamma / eps / rope tables (read-only)
        self._progs = {}

    def _front_part(self, rows, front_pool, cur_pos, page_table, q_out, g_out, in_cb):
        """(front kernels, front core set, front NOC xy list, n_q, n_k, writers per slot)."""
        dev, mm = self.device, self.mm
        n_q, n_k = front_layout(rows, len(front_pool), self.layout, prefer=FUSED_LAYOUTS)
        tasks = front_tasks(rows, n_q, n_k)
        per_tok = n_q + n_k + 1
        front_cores = front_pool[:len(tasks)]
        writers = [0] * per_tok
        for _, (_, _, ncols, first) in mm.core_args:
            for s in {_slot_of(c, n_q, n_k) for c in range(first, first + ncols)} - {None}:
                writers[s] += 1
        noc = [dev.worker_core_from_logical_core(c) for c in front_cores]
        f = self.front
        defines = [("POS_PAGE", str(cur_pos.shape[-1] * 4)), ("PT_PAGE", str(page_table.shape[-1] * 4)),
                   ("FRONT_FUSED", "1"), ("GATE_TILE", "0"), ("FRONT_IN_CB", str(in_cb)),
                   ("FRONT_VERIFY", str(int(self.verify)))]
        front_set = row_ranges(front_cores)
        per_core = []
        for i, (c, a) in enumerate(zip(front_cores, tasks)):
            args = list(a) + [writers[i % per_tok]]
            if self.verify:  # NOC (x, y) of the same-slot worker of every token (the cache-tile leaders)
                args += [v for u in range(rows) for v in (noc[u * per_tok + i % per_tok].x,
                                                          noc[u * per_tok + i % per_tok].y)]
            per_core.append((c, args))
        reader_common = f.reader_common(None, None, g_out, cur_pos, page_table)
        if self.verify:  # FRONT_VERIFY: arg 11, the row map
            reader_common.append(int(self.vmap.buffer_address()))
        kernels = unified_kernels(
            str(KERNELS / "attn_decode.cpp"), front_set, defines,
            reader_common=reader_common,
            writer_common=[int(t.buffer_address()) for t in (q_out, self.attn.k_cache, self.attn.v_cache)],
            per_core=per_core,
            compute=front_compute())
        return kernels, front_cores, front_set, [v for c in noc for v in (c.x, c.y)], n_q, n_k

    def _outputs(self, rows):
        q_out = ttnn.empty([1, rows, 16, 128], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=DRAM)
        g_out = ttnn.empty([1, 1, rows, 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=DRAM)
        return q_out, g_out

    def _front_tensors(self, cur_pos, page_table, q_out, g_out):
        a, f = self.attn, self.front
        return [f.gamma, f.rope.cos_rm, f.rope.sin_rm, cur_pos, page_table, a.k_cache, a.v_cache, q_out, g_out] + (
            [self.vmap] if self.verify else [])


class AttnQKVFront(_QKVFrontBase):
    """x [1, 1, B, 2048] -> (SDPA query [1, B, 16, 128] BF16 TILE DRAM, gate logits [1, 1, B, 32] BF16 TILE DRAM) +
    K/V cache rows, one generic_op (B <= 16).

    attn: layers.Attention with the fused q|k|v|gate weight (attn.fused_gate; caches bound via set_kv_cache); rope:
    the layer type's layers.Rope; cache: the runtime WeightCache; layer: layer index; weight: the qkvg stream copy
    (default cached_stream_weight(cache, f"L{layer}.attn.qkvg", PRESETS["attn_qkv"], dtype=attn.wqkv.dtype)).
    layout: (n_q, n_k) front workers per row (default: the first of attn_decode.FUSED_LAYOUTS that fits)."""

    def __init__(self, attn, rope, cache, layer: int, layout=None, weight=None):
        self._init_front(attn, rope, layout)
        if weight is None:
            weight = cached_stream_weight(cache, f"L{layer}.attn.qkvg", self.preset, dtype=attn.wqkv.dtype)
        self.mm = StreamDecodeMatmul(self.device, weight, self.preset, CK_BFP8, n_out=QKVG_OUT)

    def _build(self, x, cur_pos, page_table, q_out, g_out, rows):
        dev, mm, p = self.device, self.mm, self.preset
        grid = dev.compute_with_storage_grid_size()
        mm_cores = [c for c, _ in mm.core_args]
        taken = {(c.x, c.y) for c in mm_cores}
        pool = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(grid.x * grid.y)
                if (i % grid.x, i // grid.x) not in taken]
        front_kernels, front_cores, _, front_xy, n_q, n_k = self._front_part(rows, pool, cur_pos, page_table, q_out,
                                                                             g_out, 0)
        everyone = row_ranges(mm_cores + front_cores)
        common = [("kt", mm.k_tiles), ("w", p.w), ("kb", p.kb), ("rows", rows)]
        per_core = [(c, list(a)) for c, a in mm.core_args]
        mm_kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "attn_qkv_mm.cpp"),
            core_ranges=row_ranges(mm_cores),
            ncrisc_named_compile_time_args=common + [("in1_tile_bytes", mm.tile_bytes), ("slots", p.slots),
                                                     ("outstanding", p.outstanding)],
            brisc_named_compile_time_args=common + [("nq", n_q), ("nk", n_k)],
            brisc_compile_time_args=list(ttnn.TensorAccessorArgs(x).get_compile_time_args()),
            trisc_named_compile_time_args=common,
            ncrisc_common_runtime_args=[int(mm.weight.buffer_address())],
            brisc_common_runtime_args=[int(x.buffer_address()), 0, int(g_out.buffer_address())] + front_xy,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=mm.fidelity, fp32_dest_acc_en=True,
                                                              math_approx_mode=False),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=per_core, brisc_args=per_core, trisc_args=per_core),
        ).get_kernel_descriptors().kernels
        # every CB on every core: uniform L1 addresses (the matmul workers write the front's CB 0 remotely)
        cbs = front_cbs(everyone, fused=True)
        cbs += cb_descriptors([12], mm.k_tiles, everyone)
        cbs += cb_descriptors([13], p.slots * p.kb * p.w, everyone, dtype=mm.weight.dtype, page=mm.tile_bytes)
        cbs += cb_descriptors([14], 2 * p.w, everyone)
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=everyone, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=mm_kernel + front_kernels, cbs=cbs, semaphores=sems)

    def __call__(self, x, cur_pos, page_table):
        """x [1, 1, B, 2048] BF16 TILE interleaved (DRAM or L1); cur_pos [B] int32; page_table [B, blocks] int32
        -> (q [1, B, 16, 128] BF16 TILE DRAM, gate [1, 1, B, 32] BF16 TILE DRAM raw logits); attn.k_cache / v_cache
        updated in place."""
        rows = _check_x(x, "AttnQKVFront x")
        q_out, g_out = self._outputs(rows)
        tensors = [x, self.mm.weight] + self._front_tensors(cur_pos, page_table, q_out, g_out)
        prog = self._progs.setdefault(rows, CachedProgram())
        prog.run(tensors, lambda: self._build(x, cur_pos, page_table, q_out, g_out, rows))
        return q_out, g_out


class AttnNormQKVFront(_QKVFrontBase):
    """(a, b) [1, 1, B, 2048] -> (h, SDPA query, gate logits) + K/V cache rows, one generic_op (B <= 16):

        h = a + bf16(rms_norm(b) * post)          (post None: h = a + b; b None: h = a, no add)
        n = rms_norm(h) * norm_weight              (the operator norm; never materialized)
        q, gate, K/V = AttnQKVFront on n

    bitwise ResidualRMSNorm(device, norm_weight, eps, post_weight=post) + AttnQKVFront (= NormStreamDecodeMatmul's
    prologue on the 8 x 4 matmul rectangle at `origin`, then the front on cores outside it).  norm_weight / post:
    torch [2048] (operator_norm weight; the previous layer's post_ffn_norm weight).  cache / layer / weight: as
    AttnQKVFront.  h: new tensor in h_memory_config (default L1).  verify: speculative verification
    (kernels/attn_decode.cpp FRONT_VERIFY) with the row map `vmap` ([1, 16] uint32 ROW_MAJOR DRAM): the rows are groups
    of consecutive tokens, one per sequence, one worker per 32-row cache tile of a group patches all of its rows.
    share: another instance of the same layer whose matmul half (stream weight, prologue) and front tables this one
    reuses."""

    def __init__(self, attn, rope, norm_weight: torch.Tensor | None = None, eps: float | None = None,
                 post: torch.Tensor | None = None, cache=None, layer: int | None = None, weight=None, layout=None,
                 origin=(0, 6), h_memory_config=L1, verify: bool = False, share=None, vmap=None):
        from .norm_matmul import NormStreamDecodeMatmul

        if verify and vmap is None:
            raise ValueError("AttnNormQKVFront: verify needs the row map (vmap)")
        self.verify, self.vmap = verify, vmap
        if share is not None:
            self.attn, self.device, self.layout, self.preset = share.attn, share.device, share.layout, share.preset
            self.front, self.mm, self.norm = share.front, share.mm, share.norm
            self._progs = {}
            self.h_memory_config = share.h_memory_config
            return
        self._init_front(attn, rope, layout)
        if weight is None:
            weight = cached_stream_weight(cache, f"L{layer}.attn.qkvg", self.preset, dtype=attn.wqkv.dtype)
        # the norm + matmul half: NormStreamDecodeMatmul's placement / prologue / ring (its kernel is not used)
        self.mm = NormStreamDecodeMatmul(self.device, norm_weight, eps, weight, self.preset, CK_BFP8, origin=origin,
                                         post_weight=post, n_out=QKVG_OUT)
        self.norm = self.mm.norm
        self.h_memory_config = h_memory_config

    def _front_pool(self):
        """Free cores, the block right of the rectangle (same rows) first, then the rest row-major."""
        grid = self.device.compute_with_storage_grid_size()
        nm = self.norm
        rect = {(c.x, c.y) for c in ttnn.corerange_to_cores(nm.core_set, row_wise=True)}
        ox, oy = nm.origin
        pref = [(x, y) for y in range(oy, oy + nm.gy) for x in range(ox + nm.gx, grid.x)]
        rest = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in rect and (x, y) not in pref]
        return [ttnn.CoreCoord(x, y) for x, y in pref + rest]

    def _build(self, a, b, h, cur_pos, page_table, q_out, g_out, rows):
        mm, nm, p = self.mm, self.norm, self.preset
        residual = b is not None
        front_kernels, front_cores, front_set, front_xy, n_q, n_k = self._front_part(
            rows, self._front_pool(), cur_pos, page_table, q_out, g_out, 30)
        rect = nm.core_set
        everyone = row_ranges(list(ttnn.corerange_to_cores(rect, row_wise=True)) + front_cores)
        common = [("kt", mm.k_tiles), ("w", p.w), ("kb", p.kb), ("rows", rows)]
        np_dm = nm.named_args(rows, residual, cb_x=0)
        np_c = nm.named_args(rows, residual, cb_x=0, dm=False)
        addrs = [int(mm.weight.buffer_address()), 0, int(g_out.buffer_address())]
        per_core = [(c, list(args)) for c, args in mm.core_args]
        mm_kernels = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "attn_qkv_mm.cpp"), core_ranges=rect, defines=[("QKV_NORM", "1")],
            ncrisc_named_compile_time_args=common + np_dm + [("in1_tile_bytes", mm.tile_bytes), ("slots", mm.slots),
                                                            ("outstanding", p.outstanding)],
            brisc_named_compile_time_args=common + np_dm + [("nq", n_q), ("nk", n_k)],
            brisc_compile_time_args=nm.accessor_args(a, b, h),
            trisc_named_compile_time_args=common + np_c + nm.compute_args() + [("mm_fid", mm.mm_fid)],
            ncrisc_common_runtime_args=addrs,
            brisc_common_runtime_args=addrs + nm.runtime_args(a, b, h) + front_xy,
            trisc_common_runtime_args=addrs,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                              fp32_dest_acc_en=True, math_approx_mode=False),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=per_core, brisc_args=per_core, trisc_args=per_core),
        ).get_kernel_descriptors().kernels
        tile = ttnn.TileDescriptor(ttnn.Tile([32, 32]))

        def cb(index, dtype, page, pages, cores):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        # CB 30 first and on every core: the same L1 address on the matmul and the front cores
        cbs = [cb(30, ttnn.bfloat16, 2048, 4, everyone),
               cb(0, ttnn.bfloat16, 2048, mm.k_tiles, rect),
               cb(1, mm.weight.dtype, mm.tile_bytes, mm.slots * p.kb * p.w, rect),
               cb(16, ttnn.bfloat16, 2048, 2 * p.w, rect)] + nm.cbs(residual)
        cbs += front_cbs(front_set, fused=True, input_cb=False)
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=everyone, initial_value=0)] + nm.semaphores()
        if self.verify:  # the cache-tile leaders' receive slots (kHeadTiles pages per follower), arrival counter
            cbs += cb_descriptors([26], 4 * max(1, rows - 1), front_set, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
            sems.append(ttnn.SemaphoreDescriptor(id=1, core_ranges=front_set, initial_value=0))
        return ttnn.ProgramDescriptor(kernels=mm_kernels + front_kernels, cbs=cbs, semaphores=sems)

    def __call__(self, a, b, cur_pos, page_table):
        """a, b: [1, 1, B, 2048] BF16 TILE interleaved (DRAM or L1), b may be None -> (h, q, gate): h [1, 1, B, 2048]
        BF16 (h_memory_config; a itself when b is None), q [1, B, 16, 128] and gate [1, 1, B, 32] BF16 TILE DRAM;
        attn.k_cache / v_cache updated in place."""
        rows = _check_x(a, "AttnNormQKVFront a")
        if b is not None and tuple(b.shape) != tuple(a.shape):
            raise ValueError("AttnNormQKVFront: a and b must have the same shape")
        q_out, g_out = self._outputs(rows)
        if b is None:
            h = a
            tensors = [a, self.mm.weight]
        else:
            h = ttnn.empty(list(a.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=self.h_memory_config)
            tensors = [a, b, self.mm.weight, h]
        tensors += self.norm.tensors() + self._front_tensors(cur_pos, page_table, q_out, g_out)
        prog = self._progs.setdefault((b is None, rows), CachedProgram())
        prog.run(tensors, lambda: self._build(a, b, h if b is not None else None, cur_pos, page_table, q_out, g_out,
                                              rows))
        return h, q_out, g_out


def stock_sdpa_gated(attn, q, gate, cur_pos, page_table):
    """Stock fallback for the SDPA + gate part of layers.Attention.decode: paged_scaled_dot_product_attention_decode
    (window attn.window) on the front's query, then Attention._gate (sigmoid -> head expansion -> mul) with the
    front's gate logits: bitwise the stock decode for the same q / gate / caches.  -> [1, 1, B, 2048] BF16 DRAM."""
    B = q.shape[1]
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q, attn.k_cache, attn.v_cache, cur_pos_tensor=cur_pos, page_table_tensor=page_table, is_causal=True,
        scale=attn.scale, sliding_window_size=attn.window,
        program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                                              q_chunk_size=32, k_chunk_size=64, exp_approx_mode=False),
        compute_kernel_config=sdpa_compute())
    flat = ttnn.reshape(out, [1, 1, B, attn.nh * attn.hd])
    gated = attn._gate(flat, gate)
    free(out)
    return gated


_CONSTS: dict = {}


def sdpa_consts(device):
    """[32, 32 * 254 * 8] BF16 TILE DRAM (one tile row; page p * 8 + r = replica r of tile p, so the readers spread over
    the 8 banks): zeros (unused), ones, then for v = 1..63 the tail mask pair of a last block with v live tokens (token
    tile t, column j: 0 if 32 t + j < v else -1e9), then for v = 1..63 the head mask pair of a window's first block
    starting at token v (-1e9 if 32 t + j < v else 0)."""
    key = id(device)
    if key not in _CONSTS:
        tiles = [torch.zeros(32, 32), torch.ones(32, 32)]
        j = torch.arange(64)
        for head in (False, True):
            for v in range(1, 64):
                row = torch.where((j < v) == head, -1e9, 0.0)
                tiles += [row[:32].expand(32, 32), row[32:].expand(32, 32)]
        host = torch.cat([t for t in tiles for _ in range(8)], dim=1).to(torch.bfloat16)
        _CONSTS[key] = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                       memory_config=DRAM)
    return _CONSTS[key]


class SDPADecode:
    """Paged causal SDPA decode for LFM3 (16 q heads, 4 kv heads, head_dim 128, BFP8 K/V in blocks of 64 tokens) with
    the sigmoid head gate:

        out[1, 1, B, 2048] = concat_h(softmax(q_h K^T * scale) V) * sigmoid(gate[:, h])   (q head h at columns
        [128 h, 128 h + 128): the flattened layout o_proj takes)

    from the SDPA query q [1, B, 16, 128] BF16 TILE DRAM (AttnDecode / AttnQKVFront), the K/V caches
    ([blocks, 4, 64, 128] BFP8), cur_pos [B] int32, page_table [B, blocks] int32 (DRAM) and the gate logits (a
    [1, 1, B, 32] BF16 TILE DRAM tensor, or the qkvg projection itself: columns 3072..3087), B = rows <= 16.  window
    None: causal over [0, cur_pos]; window w (a multiple of 64): keys (cur_pos - w, cur_pos] only, and only the
    blocks covering them are read (sliding layers cost ~constant in the position).  One generic_op on `cores` cores;
    kernels/attn_sdpa.cpp: each (row, kv head) group's live blocks are split into partitions that depend only on that
    row's position (a row's output does not depend on the other rows: batch-invariant), the partitions of all groups
    are spread over the cores by weight (blocks + k0 per partition; rows with cur_pos < 0 have none and cost nothing),
    per-partition exact two-pass softmax (accurate FP32 exp, FP32 scale), the group's reducer (core G % cores) combines
    the partitions' [O | l] with weights exp(m_p - M) in one matmul (fixed order), out = O * sigmoid(gate) / l.  Rows
    with cur_pos < 0: output row undefined.

    workers: the partition count of a group up to chunking (min(workers, blocks) partitions while they hold <=
    chunk_blocks blocks).  chunk_blocks: the most KV blocks one partition holds in L1 (S, P, V); longer contexts get
    workers x ceil(n / (workers x chunk_blocks)) partitions (any page-table width, e.g. the 2048 blocks of
    max_model_len 131072).  Every width of one step must use the same workers / chunk_blocks / table width (batch
    invariance across widths).  k0: per-partition overhead in block units (schedule weights).  cores: the first
    `cores` grid cores (row-major) run the schedule; default every core (110; B = 1 at 131071: 345 vs 361 us on 104).
    A row's output does not depend on `cores`.  noc1_rows: grid rows whose KV reader runs on BRISC / NOC1."""

    def __init__(self, device, rows: int = 1, workers: int = 13, fidelity=ttnn.MathFidelity.HiFi2,
                 scale: float = 128 ** -0.5, outstanding: int = 4, noc1_rows=(0, 1, 2, 4), chunk_blocks: int = 32,
                 k0: int = 2, cores: int | None = None):
        if not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"SDPADecode: 1..16 rows, got {rows}")
        self.workers = workers
        self.device, self.rows, self.fidelity, self.scale = device, rows, fidelity, scale
        self.outstanding, self.chunk_blocks, self.k0 = outstanding, chunk_blocks, k0
        self.noc1_rows = set(range(noc1_rows)) if isinstance(noc1_rows, int) else set(noc1_rows)
        self.cores = cores
        self.consts = sdpa_consts(device)
        self._progs = {}

    def _sdpa_part(self, q, k_cache, v_cache, cur_pos, page_table, gate, gate_tile, window, out, sdpa_cores):
        """(kernels, cbs, semaphores) of the SDPA program on `sdpa_cores` ((x, y) logical, the schedule's order)."""
        dev, W, B = self.device, self.workers, self.rows
        max_blocks = int(page_table.shape[-1])
        pt_page = max_blocks * 4
        pos_page = int(cur_pos.shape[-1]) * 4
        # partitions of at most C blocks (S / P / V of one partition live in L1); a group has at most pmax =
        # W * ceil(max_blocks / (W C)) partitions (kernel: row_work()).  Contexts up to W C blocks get min(W, n)
        # partitions whatever C is, so narrow test tables match the serving width there.
        if self.chunk_blocks < self.outstanding + 1:
            raise ValueError("SDPADecode: chunk_blocks must exceed the outstanding KV reads")
        C = max(min((max_blocks + W - 1) // W, self.chunk_blocks), self.outstanding + 1)
        pmax = W * -(-max_blocks // (W * C)) if max_blocks >= W else max_blocks
        nkt = (4 * pmax + 31) // 32
        nc = len(sdpa_cores)
        groups = 4 * B
        slots = -(-groups // nc)  # reducer slots per core (group G -> core G % nc, slot G // nc)
        k0 = self.k0
        # partitions whose weighted start lies in one core's share (every partition weighs >= 1 + k0)
        max_parts = (groups * (max_blocks + k0 * pmax) // nc + 1) // (1 + k0) + 2
        noc = [dev.worker_core_from_logical_core(ttnn.CoreCoord(*c)) for c in sdpa_cores]
        rt_args = [(ttnn.CoreCoord(*core), [c, int(core[1] in self.noc1_rows)])  # KV reader on BRISC / NOC1 there
                   for c, core in enumerate(sdpa_cores)]
        core_set = row_ranges([ttnn.CoreCoord(*c) for c in sdpa_cores])

        def cb(index, dtype, page, pages):
            return cb_descriptors([index], pages, core_set, dtype=dtype, page=page)

        r64 = lambda n: (n + 63) // 64 * 64  # noqa: E731

        def scratch(index, size):
            return [ttnn.CBDescriptor(total_size=size, core_ranges=core_set,
                                      format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, size)])]

        row_bytes = r64(B * 32)  # kernel struct Row[B]
        # kernel layouts: scratch = cur_pos 64 | page-table slice | Row[B] | partition info; mail = slots x mailbox |
        # cur_pos 64 | stage 64 | gate rows slots x 64 | Row[B]
        scratch_bytes = 64 + (pt_page + 127) // 64 * 64 + row_bytes + 4 * max_parts
        mail_bytes = slots * r64(16 * pmax) + 128 + 64 * slots + row_bytes
        f32, bf16, bfp8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
        # Q (4 tiles) and the mask unit (4 pages) double-buffered: the reader stages a segment's Q / a partition's
        # masks while compute still works on the previous one
        cbs = (cb(0, bf16, 2048, 8) + cb(1, bfp8, BFP8_TILE, 8 * (self.outstanding + 1))
               + cb(2, bfp8, BFP8_TILE, 8 * C)
               + cb(3, bf16, 2048, 8) + cb(4, bf16, 2048, 1) + cb(5, bf16, 2048, 1) + cb(6, bf16, 2048, 1)
               + scratch(7, r64(4 * (8 + max_parts))) + cb(8, bf16, 2048, slots) + scratch(9, r64(scratch_bytes))
               + scratch(10, r64(mail_bytes))
               + cb(16, f32, 4096, 2 * C) + cb(17, f32, 4096, 1) + cb(18, f32, 4096, slots)
               + cb(19, bf16, 2048, 2 * C) + cb(20, f32, 4096, 5)
               + cb(21, f32, 4096, 5 * nkt * slots)
               + cb(22, bf16, 2048, 4) + cb(23, f32, 4096, 1)
               + cb(24, f32, 4096, nkt * slots) + cb(25, f32, 4096, nkt))
        named = [("workers", W), ("gather_kt", nkt), ("rows", B), ("pos_page", pos_page), ("pt_page", pt_page),
                 ("outstanding", self.outstanding), ("v_tiles", 8 * C), ("chunk_blocks", C), ("partitions", pmax),
                 ("cores", nc), ("slots", slots), ("k0", k0), ("max_parts", max_parts)]
        common = [int(t.buffer_address()) for t in (q, k_cache, v_cache, cur_pos, page_table, self.consts, out,
                                                    cur_pos, gate)]
        common += [v for c in noc for v in (c.x, c.y)]
        defines = [("WINDOW", str(window or 0)), ("SCALE_F", f"{self.scale:.9e}f"), ("GATE_TILE", str(gate_tile))]
        l1 = sum(c.total_size for c in cbs)
        if l1 > 1_300_000:
            raise ValueError(f"SDPADecode: {l1} B of L1 per core for {max_blocks} blocks / {B} rows")
        kernels = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "attn_sdpa.cpp"), core_ranges=core_set, defines=defines,
            ncrisc_named_compile_time_args=named, brisc_named_compile_time_args=named,
            trisc_named_compile_time_args=named,
            ncrisc_compile_time_args=list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            brisc_compile_time_args=list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            ncrisc_common_runtime_args=common, brisc_common_runtime_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                                              math_approx_mode=False, dst_full_sync_en=True),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(ncrisc_args=rt_args, brisc_args=rt_args,
                                                                          trisc_args=rt_args),
        ).get_kernel_descriptors().kernels
        # gather counters of the reducer slots
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=core_set, initial_value=0) for s in range(slots)]
        return kernels, cbs, sems

    def _build(self, q, k_cache, v_cache, cur_pos, page_table, gate, gate_tile, window, out):
        grid = self.device.compute_with_storage_grid_size()
        nc = min(self.cores or grid.x * grid.y, grid.x * grid.y)
        kernels, cbs, sems = self._sdpa_part(q, k_cache, v_cache, cur_pos, page_table, gate, gate_tile, window, out,
                                             [(n % grid.x, n // grid.x) for n in range(nc)])
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)

    def _check(self, q, k_cache, gate, window):
        B = self.rows
        if tuple(q.shape) != (1, B, 16, 128) or q.dtype != ttnn.bfloat16 or q.memory_config() != DRAM:
            raise ValueError(f"SDPADecode: q must be [1,{B},16,128] BF16 TILE DRAM, got {tuple(q.shape)}")
        if tuple(k_cache.shape)[1:] != (4, 64, 128) or k_cache.dtype != ttnn.bfloat8_b:
            raise ValueError("SDPADecode: caches must be [blocks, 4, 64, 128] BFP8")
        gw = int(gate.shape[-1])
        if (tuple(gate.shape)[:3] != (1, 1, B) or gate.dtype != ttnn.bfloat16 or gate.memory_config() != DRAM
                or not (gw == 32 or gw >= 3104)):
            raise ValueError(f"SDPADecode: gate must be [1,1,{B},32] (or qkvg) BF16 TILE DRAM, got {tuple(gate.shape)}")
        if window is not None and (window <= 0 or window % 64):
            raise ValueError("SDPADecode: window must be a positive multiple of 64")
        return 0 if gw == 32 else 96

    def __call__(self, q, k_cache, v_cache, cur_pos, page_table, gate, window=None, memory_config=None):
        """q [1, B, 16, 128] BF16 TILE DRAM; gate [1, 1, B, 32] (or the [1, 1, B, >= 3104] qkvg projection) BF16 TILE
        DRAM; window None (full causal) or the sliding window (multiple of 64) -> the gated attention output
        [1, 1, B, 2048] BF16 TILE interleaved (default L1)."""
        gate_tile = self._check(q, k_cache, gate, window)
        out = ttnn.empty([1, 1, self.rows, 2048], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                         memory_config=memory_config or ttnn.L1_MEMORY_CONFIG)
        tensors = [q, k_cache, v_cache, cur_pos, page_table, gate, out]
        prog = self._progs.setdefault((window or 0, gate_tile, int(k_cache.buffer_address())), CachedProgram())
        prog.run(tensors, lambda: self._build(q, k_cache, v_cache, cur_pos, page_table, gate, gate_tile, window, out))
        return out

