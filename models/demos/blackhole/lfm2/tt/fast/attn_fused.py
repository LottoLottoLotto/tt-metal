# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM2 attention decode, round 2: fewer launches (owner LfmKern.AttnFuse).

AttnQKVFront: the qkv projection (StreamDecodeMatmul preset attn_qkv, same stream weight cache entry) and the AttnDecode
front end (q/k RMSNorm, RoPE, paged BFP8 K/V write, SDPA query) in ONE program.  The 32 matmul workers
(kernels/attn_qkv_mm.cpp: decode_matmul_stream.cpp's weight stream and math) send each output column's token rows
straight into the input tiles of the front workers (kernels/attn_front_fused.cpp: attn_decode.cpp with that input),
which meanwhile read their position, RoPE row, page-table entry and the cache tiles to patch.  Same values as
StreamDecodeMatmul + AttnDecode, so bitwise the same q and K/V caches.

    qd = fused(x, cur_pos, page_table)      # replaces  qkv = qkv_dec(x); qd = AttnDecode(qkv, cur_pos, page_table)
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from ..common import CK_BFP8
from .attn_decode import BFP8_TILE, AttnDecode
from .conv_decode import CachedProgram, cb_descriptors, row_ranges, unified_kernels
from .decode_matmul import PRESETS, StreamDecodeMatmul, cached_stream_weight

KERNELS = Path(__file__).resolve().parent / "kernels"
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _slot_of(col: int, n_q: int) -> int:
    """qkv output tile column -> front worker slot (0..n_q-1 q, n_q k, n_q+1 v)."""
    h = col // 2
    return h // (32 // n_q) if h < 32 else (n_q if h < 40 else n_q + 1)


class AttnQKVFront:
    """x [1, 1, B, 2048] -> SDPA query [1, B, 32, 64] BF16 TILE + K/V cache rows, one generic_op (B <= 8).

    attn: layers.Attention (caches bound via set_kv_cache); rope: layers.Rope; cache: the runtime WeightCache;
    layer: layer index (the qkv stream weight is the round-1 cache entry f"L{layer}.attn.qkv.stream-c4w3").
    verify: speculative verification (kernels/attn_front_fused.cpp FRONT_VERIFY): the rows are consecutive tokens of
    one sequence, one worker per cache tile patches all of its rows.  share: another instance of the same layer whose
    weight and front tables this one reuses."""

    def __init__(self, attn, rope, cache, layer: int, n_q: int = 4, weight=None, verify: bool = False, share=None):
        self.attn, self.device, self.n_q, self.verify = attn, attn.device, n_q, verify
        self.preset = PRESETS["attn_qkv"]
        if share is not None:
            self.front, self.mm = share.front, share.mm
        else:
            self.front = AttnDecode(attn, rope, n_q=n_q)  # gamma / RoPE table / eps (read-only)
            if weight is None:
                weight = cached_stream_weight(cache, f"L{layer}.attn.qkv", self.preset)
            self.mm = StreamDecodeMatmul(self.device, weight, self.preset, CK_BFP8)
        self._prog = CachedProgram()

    def _build(self, x, cur_pos, page_table, q_out, k_cache, v_cache, rows):
        dev, n_q, mm, p = self.device, self.n_q, self.mm, self.preset
        grid = dev.compute_with_storage_grid_size()
        per_tok = n_q + 2
        qh = 32 // n_q
        n_mm = len(mm.core_args)
        front_tasks = []
        for t in range(rows):
            front_tasks += [(0, i * qh, qh, t) for i in range(n_q)] + [(1, 0, 8, t), (2, 0, 8, t)]
        n_all = n_mm + len(front_tasks)
        if n_all > grid.x * grid.y or [c for c, _ in mm.core_args] and max(
                c.y * grid.x + c.x for c, _ in mm.core_args) >= n_mm:
            raise ValueError(f"AttnQKVFront: {rows} rows need {n_all} cores (matmul workers must be the first {n_mm})")
        front_cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n_mm, n_all)]
        mm_cores = [c for c, _ in mm.core_args]
        everyone = row_ranges(mm_cores + front_cores)
        mm_set, front_set = row_ranges(mm_cores), row_ranges(front_cores)
        # matmul writers per front slot (same for every token)
        writers = [0] * per_tok
        for _, (_, _, ncols, first) in mm.core_args:
            for s in sorted({_slot_of(c, n_q) for c in range(first, first + ncols)}):
                writers[s] += 1
        noc = [dev.worker_core_from_logical_core(c) for c in front_cores]
        front_xy = [v for c in noc for v in (c.x, c.y)]

        common = [("kt", mm.k_tiles), ("w", p.w), ("kb", p.kb), ("rows", rows)]
        mm_kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "attn_qkv_mm.cpp"),
            core_ranges=mm_set,
            ncrisc_named_compile_time_args=common + [("in1_tile_bytes", mm.tile_bytes), ("slots", p.slots),
                                                     ("outstanding", p.outstanding)],
            brisc_named_compile_time_args=common + [("nq", n_q)],
            brisc_compile_time_args=list(ttnn.TensorAccessorArgs(x).get_compile_time_args()),
            trisc_named_compile_time_args=common,
            ncrisc_common_runtime_args=[int(mm.weight.buffer_address())],
            brisc_common_runtime_args=[int(x.buffer_address())] + front_xy,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=mm.fidelity, fp32_dest_acc_en=True,
                                                              math_approx_mode=False),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=[(c, list(a)) for c, a in mm.core_args], brisc_args=[(c, list(a)) for c, a in mm.core_args],
                trisc_args=[(c, list(a)) for c, a in mm.core_args]),
        ).get_kernel_descriptors().kernels

        f = self.front
        defines = [("TABLE_PAGE", str(f.table.shape[-1] * 2)), ("POS_PAGE", str(cur_pos.shape[-1] * 4)),
                   ("PT_PAGE", str(page_table.shape[-1] * 4)), ("FRONT_VERIFY", str(int(self.verify)))]
        per_core = []
        for i, (c, a) in enumerate(zip(front_cores, front_tasks)):
            args = list(a) + [writers[i % per_tok]]
            if self.verify:  # NOC (x, y) of the same-role worker of every token (the cache-tile leaders)
                k = n_q + (a[0] == 2)
                args += [v for u in range(rows) for v in (noc[u * per_tok + k].x, noc[u * per_tok + k].y)]
            per_core.append((c, args))
        eps_bits = int(torch.tensor([f.eps], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                               fp32_dest_acc_en=True, dst_full_sync_en=False)
        compute.unpack_to_dest_mode.extend([ttnn.UnpackToDestMode.Default] * 64)
        front_kernels = unified_kernels(
            str(KERNELS / "attn_front_fused.cpp"), front_set, defines,
            reader_common=[0] + [int(t.buffer_address()) for t in (f.gamma, f.table, cur_pos, page_table, k_cache,
                                                                    v_cache)] + [eps_bits],
            writer_common=[int(t.buffer_address()) for t in (q_out, k_cache, v_cache)],
            per_core=per_core,
            compute=compute)

        # every CB on every core: uniform L1 addresses (the matmul workers write the front's CB 0 remotely)
        cbs = []
        for i, n in ((0, 2), (2, 1), (3, 1), (5, 2), (16, 2), (6, 2), (7, 2), (9, 1), (11, 1), (24, 2)):
            cbs += cb_descriptors([i], n, everyone)
        for i, n in ((19, 1), (20, 2), (21, 1), (22, 2)):
            cbs += cb_descriptors([i], n, everyone, dtype=ttnn.float32, page=4096)
        cbs += cb_descriptors([10], 16, everyone, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
        cbs += cb_descriptors([25], 2, everyone, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
        cbs += cb_descriptors([12], mm.k_tiles, everyone)
        cbs += cb_descriptors([13], p.slots * p.kb * p.w, everyone, dtype=mm.weight.dtype, page=mm.tile_bytes)
        cbs += cb_descriptors([14], 2 * p.w, everyone)
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=everyone, initial_value=0)]
        if self.verify:  # the leaders' receive slots (2 packed pages per follower) and arrival counter
            cbs += cb_descriptors([26], 2 * max(1, rows - 1), everyone, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
            sems.append(ttnn.SemaphoreDescriptor(id=1, core_ranges=everyone, initial_value=0))
        return ttnn.ProgramDescriptor(kernels=mm_kernel + front_kernels, cbs=cbs, semaphores=sems)

    def __call__(self, x, cur_pos, page_table):
        """x [1, 1, B, 2048] BF16 TILE interleaved (DRAM or L1); cur_pos [B] int32; page_table [B, blocks] int32
        -> SDPA query [1, B, 32, 64] BF16 TILE DRAM; attn.k_cache / v_cache updated in place."""
        rows = x.shape[-2]
        if (tuple(x.shape) != (1, 1, rows, 2048) or x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT
                or rows > 8):
            raise ValueError(f"AttnQKVFront: x must be [1,1,B<=8,2048] BF16 TILE, got {tuple(x.shape)}")
        k_cache, v_cache = self.attn.k_cache, self.attn.v_cache
        q_out = ttnn.empty([1, rows, 32, 64], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=DRAM)
        tensors = [x, self.mm.weight, self.front.gamma, self.front.table, cur_pos, page_table, k_cache, v_cache,
                   q_out]
        self._prog.run(tensors, lambda: self._build(x, cur_pos, page_table, q_out, k_cache, v_cache, rows))
        return q_out



class AttnNormQKVFront:
    """Residual add + operator RMSNorm + qkv projection + AttnDecode front in ONE generic_op:
    (h, qd) = fused(a, b, cur_pos, page_table) replaces
        h, n = op_rn(a, b); qkv = qkv_dec(n); qd = AttnDecode(qkv, cur_pos, page_table)
    and NormStreamDecodeMatmul's (h, qkv) = nmm(a, b); qd = AttnDecode(qkv, ...): bitwise the same h, q and K/V cache.
    Kernels: attn_norm_qkv_mm.cpp (decode_matmul_norm.cpp's norm prologue + weight stream on the 8 x 4 rectangle at
    `origin`, BRISC sending the qkv columns to the front workers) + attn_front_fused.cpp on 6 cores per token outside
    the rectangle.  b = None: h = a (no add).  a, b: [1, 1, B <= 8, 2048] BF16 TILE interleaved (DRAM or L1)."""

    def __init__(self, attn, rope, norm_weight: torch.Tensor, eps: float, cache=None, layer: int | None = None,
                 n_q: int = 4, weight=None, origin=(0, 6), h_memory_config=ttnn.L1_MEMORY_CONFIG):
        from .norm_matmul import NormStreamDecodeMatmul

        self.attn, self.device, self.n_q = attn, attn.device, n_q
        self.front = AttnDecode(attn, rope, n_q=n_q)
        self.preset = PRESETS["attn_qkv"]
        if weight is None:
            weight = cached_stream_weight(cache, f"L{layer}.attn.qkv", self.preset)
        # the norm + matmul half: NormStreamDecodeMatmul's placement / ring / prologue (its kernel is not used)
        self.nmm = NormStreamDecodeMatmul(self.device, norm_weight, eps, weight, self.preset, CK_BFP8, origin=origin)
        self.h_memory_config = h_memory_config
        self._progs = {}

    def _front_cores(self, rows):
        grid = self.device.compute_with_storage_grid_size()
        rect = {(c.x, c.y) for c in ttnn.corerange_to_cores(self.nmm.norm.core_set, row_wise=True)}
        ox, oy = self.nmm.norm.origin
        gx, gy = self.nmm.norm.gx, self.nmm.norm.gy
        # first the block right of the rectangle (same rows), then everything else row-major
        pref = [(x, y) for y in range(oy, oy + gy) for x in range(ox + gx, grid.x)]
        rest = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in rect and (x, y) not in pref]
        need = rows * (self.n_q + 2)
        free = [c for c in pref + rest if c not in rect]
        if need > len(free):
            raise ValueError(f"AttnNormQKVFront: {rows} rows need {need} front cores")
        return [ttnn.CoreCoord(x, y) for x, y in free[:need]]

    def _build(self, a, b, h, cur_pos, page_table, q_out, k_cache, v_cache, rows):
        dev, n_q, nm, mm, p = self.device, self.n_q, self.nmm.norm, self.nmm, self.preset
        residual = b is not None
        per_tok = n_q + 2
        qh = 32 // n_q
        front_cores = self._front_cores(rows)
        front_set = row_ranges(front_cores)
        rect = nm.core_set
        everyone = row_ranges(list(ttnn.corerange_to_cores(rect, row_wise=True)) + front_cores)
        writers = [0] * per_tok
        for _, (_, _, ncols, first) in mm.core_args:
            for sl in sorted({_slot_of(c, n_q) for c in range(first, first + ncols)}):
                writers[sl] += 1
        noc = [dev.worker_core_from_logical_core(c) for c in front_cores]
        front_xy = [v for c in noc for v in (c.x, c.y)]
        front_tasks = []
        for t in range(rows):
            front_tasks += [(0, i * qh, qh, t) for i in range(n_q)] + [(1, 0, 8, t), (2, 0, 8, t)]

        common = [("kt", mm.k_tiles), ("w", p.w), ("kb", p.kb), ("rows", rows), ("ks", 0), ("glu", 0)]
        np_dm = nm.named_args(rows, residual, cb_x=0)
        np_c = nm.named_args(rows, residual, cb_x=0, dm=False)
        addrs = [int(mm.weight.buffer_address()), 0, 0]
        per_core = [(c, list(args)) for c, args in mm.core_args]
        mm_kernels = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "attn_norm_qkv_mm.cpp"),
            core_ranges=rect,
            ncrisc_named_compile_time_args=common + np_dm + [("in1_tile_bytes", mm.tile_bytes),
                                                            ("slots", mm.slots), ("outstanding", p.outstanding)],
            brisc_named_compile_time_args=common + np_dm + [("nq", n_q)],
            brisc_compile_time_args=nm.accessor_args(a, b, h if residual else None),
            trisc_named_compile_time_args=common + np_c + nm.compute_args() + [("mm_fid", mm.mm_fid)],
            ncrisc_common_runtime_args=addrs,
            brisc_common_runtime_args=addrs + nm.runtime_args(a, b, h if residual else None) + front_xy,
            trisc_common_runtime_args=addrs,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                              fp32_dest_acc_en=True, math_approx_mode=False),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=per_core, brisc_args=per_core, trisc_args=per_core),
        ).get_kernel_descriptors().kernels

        f = self.front
        defines = [("TABLE_PAGE", str(f.table.shape[-1] * 2)), ("POS_PAGE", str(cur_pos.shape[-1] * 4)),
                   ("PT_PAGE", str(page_table.shape[-1] * 4)), ("FRONT_IN_CB", "30")]
        eps_bits = int(torch.tensor([f.eps], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                               fp32_dest_acc_en=True, dst_full_sync_en=False)
        compute.unpack_to_dest_mode.extend([ttnn.UnpackToDestMode.Default] * 64)
        front_kernels = unified_kernels(
            str(KERNELS / "attn_front_fused.cpp"), front_set, defines,
            reader_common=[0] + [int(t.buffer_address()) for t in (f.gamma, f.table, cur_pos, page_table, k_cache,
                                                                    v_cache)] + [eps_bits],
            writer_common=[int(t.buffer_address()) for t in (q_out, k_cache, v_cache)],
            per_core=[(c, list(t) + [writers[k % per_tok]]) for k, (c, t) in enumerate(zip(front_cores, front_tasks))],
            compute=compute)

        tile = ttnn.TileDescriptor(ttnn.Tile([32, 32]))

        def cb(index, dtype, page, pages, cores):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        # CB 30 first and on every core: the same L1 address on the matmul and the front cores
        cbs = [cb(30, ttnn.bfloat16, 2048, 2, everyone),
               cb(0, ttnn.bfloat16, 2048, mm.k_tiles, rect), cb(1, mm.weight.dtype, mm.tile_bytes,
                                                                mm.slots * p.kb * p.w, rect),
               cb(16, ttnn.bfloat16, 2048, 2 * p.w, rect)] + nm.cbs(residual)
        for i, n in ((2, 1), (3, 1), (5, 2), (16, 2), (6, 2), (7, 2), (9, 1), (11, 1), (24, 2)):
            cbs += cb_descriptors([i], n, front_set)
        for i, n in ((19, 1), (20, 2), (21, 1), (22, 2)):
            cbs += cb_descriptors([i], n, front_set, dtype=ttnn.float32, page=4096)
        cbs += cb_descriptors([10], 16, front_set, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
        cbs += cb_descriptors([25], 2, front_set, dtype=ttnn.bfloat8_b, page=BFP8_TILE)
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=everyone, initial_value=0)] + nm.semaphores()
        return ttnn.ProgramDescriptor(kernels=mm_kernels + front_kernels, cbs=cbs, semaphores=sems)

    def __call__(self, a, b, cur_pos, page_table):
        """-> (h, qd): h = a + b ([1, 1, B, 2048] BF16, h_memory_config; a itself when b is None), qd = SDPA query
        [1, B, 32, 64] BF16 TILE DRAM; attn.k_cache / v_cache updated in place."""
        rows = int(a.shape[2])
        if tuple(a.shape) != (1, 1, rows, 2048) or a.dtype != ttnn.bfloat16 or a.layout != ttnn.TILE_LAYOUT or rows > 8:
            raise ValueError(f"AttnNormQKVFront: a must be [1,1,B<=8,2048] BF16 TILE, got {tuple(a.shape)}")
        k_cache, v_cache = self.attn.k_cache, self.attn.v_cache
        q_out = ttnn.empty([1, rows, 32, 64], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=DRAM)
        f = self.front
        if b is None:
            h = a
            tensors = [a, self.nmm.weight, self.nmm.norm.gamma, f.gamma, f.table, cur_pos, page_table, k_cache,
                       v_cache, q_out]
        else:
            h = ttnn.empty(list(a.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                           memory_config=self.h_memory_config)
            tensors = [a, b, self.nmm.weight, self.nmm.norm.gamma, h, f.gamma, f.table, cur_pos, page_table, k_cache,
                       v_cache, q_out]
        prog = self._progs.setdefault((b is None, rows), CachedProgram())
        prog.run(tensors, lambda: self._build(a, b, h if b is not None else None, cur_pos, page_table, q_out,
                                              k_cache, v_cache, rows))
        return h, q_out

def _f32_bits(v: float) -> int:
    return int(torch.tensor([v], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF


_CONSTS: dict = {}


def sdpa_consts(device):
    """[32, 32 * 129 * 8] BF16 TILE DRAM (one tile row; page p * 8 + r = replica r of tile p, so the 104 readers
    spread over the 8 banks): zeros (unused), ones, A (A[r, k] = 1 for r < 4, k % 4 == r), then for v = 1..63 live
    tokens the last block's mask pair (token tile t, column j: 0 if 32 t + j < v else -1e9)."""
    key = id(device)
    if key not in _CONSTS:
        tiles = [torch.zeros(32, 32), torch.ones(32, 32), torch.zeros(32, 32)]
        for r in range(4):
            tiles[2][r, r::4] = 1.0
        j = torch.arange(64)
        for v in range(1, 64):
            row = torch.where(j < v, 0.0, -1e9)
            tiles += [row[:32].expand(32, 32), row[32:].expand(32, 32)]
        host = torch.cat([t for t in tiles for _ in range(8)], dim=1).to(torch.bfloat16)
        _CONSTS[key] = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                       memory_config=DRAM)
    return _CONSTS[key]


class SDPADecode:
    """Paged causal SDPA decode for LFM2 (32 q heads, 8 kv heads, head_dim 64, BFP8 K/V in blocks of 64 tokens):
    out[1, 1, B, 2048] (q head h at columns [64 h, 64 h + 64): the flattened stock output, no reshape needed) from
    the stock SDPA query q [1, B, 32, 64] BF16 TILE DRAM, the K/V caches, cur_pos [B] int32 and page_table
    [B, blocks] int32 (DRAM), B = rows <= 16.  One generic_op on `cores` cores; kernels/attn_sdpa.cpp: each
    (row, kv head) group's live blocks are split into partitions that depend only on that row's position (so a row's
    output does not depend on the other rows: batch-invariant), the partitions of all groups are spread over the
    cores by weight (blocks + k0 per partition; rows with cur_pos < 0 have none and cost nothing), per-partition
    exact two-pass softmax, the group's reducer (core G % cores) combines the partitions' [O | l] with weights
    exp(m_p - M) in one matmul (fixed order), out = O / l.  Rows with cur_pos < 0: output row undefined.

    workers: the partition count of a group up to chunking (min(workers, blocks) partitions while they hold <=
    chunk_blocks blocks; default 13).
    chunk_blocks: the most KV blocks one partition holds in L1 (S, P, V); longer contexts get workers x
    ceil(n / (workers x chunk_blocks)) partitions (any page-table width, e.g. the 2048 blocks of max_model_len 131072).
    k0: per-partition overhead in block units (schedule weights).  cores: the first `cores` grid cores (row-major)
    run the schedule; default 8 x workers (104) for one row (a long row's 8 x workers x nch partitions are an exact
    multiple: 328 vs 346 us at 122880 tokens on 110 cores), every core for more rows (B = 8 at 122880: 2381 vs
    2561 us on 104).  A row's output does not depend on `cores`.
    exp_approx: the stock kernel's Schraudolph exp for the scores (faster, ~1e-3 relative) instead of the accurate
    SFPU exp.  noc1_rows: grid rows whose KV reader runs on BRISC / NOC1 (NOC0 serves rows 0-2 poorly)."""

    def __init__(self, device, rows: int = 1, workers: int | None = None, exp_approx: bool = False,
                 fidelity=ttnn.MathFidelity.HiFi2, scale: float = 0.125, outstanding: int = 4, exp_iters: int = 2,
                 noc1_rows=(0, 1, 2, 4), chunk_blocks: int = 40, k0: int = 2, cores: int | None = None):
        if not 1 <= rows <= 16:
            raise ValueError(f"SDPADecode: 1..16 rows, got {rows}")
        self.workers = workers or 13
        self.device, self.rows, self.exp_approx, self.fidelity, self.scale = device, rows, exp_approx, fidelity, scale
        self.outstanding, self.exp_iters, self.chunk_blocks, self.k0 = outstanding, exp_iters, chunk_blocks, k0
        self.noc1_rows = set(range(noc1_rows)) if isinstance(noc1_rows, int) else set(noc1_rows)
        self.cores = cores
        self.consts = sdpa_consts(device)
        self._prog = CachedProgram()

    def _build(self, q, k_cache, v_cache, cur_pos, page_table, out):
        dev, W, B = self.device, self.workers, self.rows
        grid = dev.compute_with_storage_grid_size()
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
        smax = 2 * C
        nc = min(self.cores or (8 * W if B == 1 else grid.x * grid.y), grid.x * grid.y)
        slots = -(-8 * B // nc)  # reducer slots per core (group G -> core G % nc, slot G // nc)
        k0 = self.k0
        # partitions whose weighted start lies in one core's share (every partition weighs >= 1 + k0)
        max_parts = (8 * B * (max_blocks + k0 * pmax) // nc + 1) // (1 + k0) + 2
        sdpa_cores = [(n % grid.x, n // grid.x) for n in range(nc)]
        noc = [dev.worker_core_from_logical_core(ttnn.CoreCoord(*c)) for c in sdpa_cores]
        args = {core: [c, int(core[1] in self.noc1_rows)]  # KV reader on BRISC / NOC1 on the noc1 rows
                for c, core in enumerate(sdpa_cores)}
        cores = [ttnn.CoreCoord(*c) for c in args]
        core_set = row_ranges(cores)
        rt_args = [(ttnn.CoreCoord(*c), a) for c, a in args.items()]

        def cb(index, dtype, page, pages):
            return cb_descriptors([index], pages, core_set, dtype=dtype, page=page)

        r64 = lambda n: (n + 63) // 64 * 64  # noqa: E731

        def scratch(index, size):
            return [ttnn.CBDescriptor(total_size=size, core_ranges=core_set,
                                      format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, size)])]

        row_bytes = r64(B * 24)  # kernel struct Row[B]
        # kernel layouts: scratch = cur_pos 64 | page-table slice | Row[B] | partition info; mail = slots x mailbox |
        # cur_pos 64 | stage 64 | Row[B]
        scratch_bytes = 64 + (pt_page + 127) // 64 * 64 + row_bytes + 4 * max_parts
        mail_bytes = slots * r64(16 * pmax) + 128 + row_bytes
        v_tiles = 4 * C
        # Q and the mask pair double-buffered: the reader stages a segment's Q / a partition's mask while compute
        # still works on the previous one
        cbs = (cb(0, ttnn.bfloat16, 2048, 4) + cb(1, ttnn.bfloat8_b, BFP8_TILE, 4 * (self.outstanding + 1))
               + cb(2, ttnn.bfloat8_b, BFP8_TILE, v_tiles)
               + cb(3, ttnn.bfloat16, 2048, 4) + cb(4, ttnn.bfloat16, 2048, 1) + cb(5, ttnn.bfloat16, 2048, 1)
               + cb(6, ttnn.bfloat16, 2048, 1) + scratch(7, r64(4 * (8 + max_parts))) + scratch(9, r64(scratch_bytes))
               + scratch(10, r64(mail_bytes)) + cb(11, ttnn.bfloat16, 2048, nkt)
               + cb(16, ttnn.float32, 4096, smax) + cb(17, ttnn.float32, 4096, 1)
               + cb(18, ttnn.float32, 4096, slots)
               + cb(19, ttnn.bfloat16, 2048, smax) + cb(20, ttnn.float32, 4096, 3)
               + cb(21, ttnn.float32, 4096, 3 * nkt * slots)
               + cb(22, ttnn.bfloat16, 2048, 2) + cb(23, ttnn.float32, 4096, 1)
               + cb(24, ttnn.float32, 4096, nkt * slots) + cb(25, ttnn.float32, 4096, nkt))
        named = [("workers", W), ("gather_kt", nkt), ("rows", B), ("pos_page", pos_page), ("pt_page", pt_page),
                 ("outstanding", self.outstanding), ("v_tiles", v_tiles), ("chunk_blocks", C), ("partitions", pmax),
                 ("cores", nc), ("slots", slots), ("k0", k0), ("max_parts", max_parts)]
        common = [int(t.buffer_address()) for t in (q, k_cache, v_cache, cur_pos, page_table, self.consts, out,
                                                    cur_pos)]
        common += [v for c in noc for v in (c.x, c.y)]
        defines = [("EXP_APPROX", "1" if self.exp_approx else "0"), ("EXP_ITERS", str(self.exp_iters)),
                   ("SCALE_FP32", hex(_f32_bits(self.scale))), ("SCALE_BF16", hex(_f32_bits(self.scale) >> 16))]
        sem_ids = tuple(range(slots))  # gather counters of the reducer slots
        l1 = sum(c.total_size for c in cbs)
        if l1 > 1_100_000:
            raise ValueError(f"SDPADecode: {l1} B of L1 per core for {max_blocks} blocks / {B} rows")
        kernels = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "attn_sdpa.cpp"), core_ranges=core_set, defines=defines,
            ncrisc_named_compile_time_args=named, brisc_named_compile_time_args=named,
            trisc_named_compile_time_args=named,
            ncrisc_compile_time_args=list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            brisc_compile_time_args=list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            ncrisc_common_runtime_args=common, brisc_common_runtime_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                                              math_approx_mode=False),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(ncrisc_args=rt_args, brisc_args=rt_args,
                                                                          trisc_args=rt_args),
        ).get_kernel_descriptors().kernels
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=core_set, initial_value=0) for s in sem_ids]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)

    def __call__(self, q, k_cache, v_cache, cur_pos, page_table, memory_config=None):
        """q [1, B, 32, 64] BF16 TILE DRAM -> the attention output [1, 1, B, 2048] BF16 TILE interleaved (default
        L1)."""
        B = self.rows
        if tuple(q.shape) != (1, B, 32, 64) or q.dtype != ttnn.bfloat16 or q.memory_config() != DRAM:
            raise ValueError(f"SDPADecode: q must be [1,{B},32,64] BF16 TILE DRAM, got {tuple(q.shape)}")
        if tuple(k_cache.shape)[1:] != (8, 64, 64) or k_cache.dtype != ttnn.bfloat8_b:
            raise ValueError("SDPADecode: caches must be [blocks, 8, 64, 64] BFP8")
        out = ttnn.empty([1, 1, B, 2048], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                         memory_config=memory_config or ttnn.L1_MEMORY_CONFIG)
        tensors = [q, k_cache, v_cache, cur_pos, page_table, out]
        self._prog.run(tensors, lambda: self._build(q, k_cache, v_cache, cur_pos, page_table, out))
        return out
