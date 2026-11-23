# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Attention prefill (attention.py Inkling2Attention.prefill): the K / V front and flash attention with the
relative-position bias, masks, sink, q / r finalization and the output gate computed in the kernels.  Per (layer
application, chunk): KVFront (one program), conv_update, the table shift pre-pass + FlashPrefill (two programs),
ring_write (sliding layers).  Every program depends only on the chunk shape Tp (start / valid come from the device
tensor `pos`, int32 [1, 1, 1, 32] ROW_MAJOR: word 0 = start, word 1 = valid rows), so a prefill chunk is traceable.

    flash = FlashPrefill(device, cfg, max_seq_len)                    # one per model
    table = bias_table(device, proj, sliding)                         # per layer (attention.py: attn.bias_tbl)
    kn, vn = flash.kv(k_lin, v_lin, kd, vd, conv_ring, taps, kc, vc, pos, sliding=..)   # (None on full layers)
    flash.conv_update(k_lin, v_lin, conv_ring, states, pos)
    out = flash(qg, rs, kc, vc, table, pos, sliding=.., kn=kn, vn=vn)  # [1, 1, Tp, 3072] gated o_proj input
    flash.ring_write(kn, vn, kc, vc, pos)                             # sliding: the chunk's rows into the ring

Numerics = attention.py's former ttnn chunk path, op for op (K / V cache rows and q / r_aug bitwise): row sums of
squares as ttnn.sum(ttnn.mul(x, x, fp32)) (exact fp32 squares on the SFPU, FPU row reduce whose source register
keeps TF32), the factor chain on the SFPU (rsqrt, clip, tau), x * factor on the SFPU (the factor broadcast across
the row exactly: three TF32 pieces times a 0/1 matrix), sigmoid(gate) rounded to BF16 before the product (ttnn's
LHS activation), full-layer conv (acc truncated to TF32) + x as the ttnn FPU add did.  Sliding layers use the exact
exp (as the ttnn SDPA the old path ran), full layers the approximate one (exact_exp_full=True: exact; 2.2x the time).

Kernels (tt/kernels/fast/attn_prefill_*.cpp).  Flash work item = (q head, q chunk of Sq tiles):
- bias: B[r][t] = r_aug(rows of q tile r) . proj[:, distance] for the distance tiles t = -1 .. 32 of the window,
  computed once per item as sum_k diag(r_aug[:, k]) @ W_t,k (W = per-layer Toeplitz table `bias_table`: W_t,k[m, j]
  = proj[k, 32 t + m - j], k = 16: NEG where masked), packed BF16, added to the scores where the key is in the
  window (NEG for future / outside-window keys).  A start with o = start % 32 != 0 reads the table shifted by o
  rows (W'_t[m] = W_t[m + o] / W_{t+1}[m + o - 32]: the pre-pass attn_prefill_tshift.cpp writes it to a scratch
  table; the reader picks it when o != 0), so the queries stay in the chunk's row frame;
- the sink logit = one more softmax term (stock SDPA attention-sink step), zero value;
- online softmax as the stock ttnn SDPA (fp32 DEST, BF16 running output).
Keys: full layers read every key [0, start + Tp + 32) from the paged cache (KVFront writes the chunk first);
sliding layers positions [p0 - 1024, start) from the ring (slot tile = position tile % 34) and the chunk's
aligned-frame K / V (`kn` / `vn`: row tile i = position tile start / 32 + i, Tp + 32 rows) for positions >= the
start's tile -- the ring is rewritten afterwards (`ring_write`), so a chunk of any length can be attended before
its rows overwrite the window.

Batched serving (attention modules with seqs / kv_blocks): `slotted` = slot-major rings / conv rings, the sequence
slot = word 2 of `pos` (kernels: define SLOTTED); `pt` = full layers' page row (int32 [1, max_seq_len / 64]; define
PAGED, one more runtime arg): the caches are the pool, logical block j = pool block pt[j]; KVFront writes only the
tiles holding a valid row and the flash reads the zero block for the others (their pool blocks may belong to another
sequence; such keys are masked for every valid row).  Single-stream programs are unchanged (no defines).
"""
from __future__ import annotations

import struct

import torch
import ttnn

from .common import KERNELS, TILE_BYTES, row_ranges

NQH, NKH, DHT = 24, 8, 4
BAND = 32           # last distance tile of the 1024 window
SLOT = 18           # table tiles per slot (k = 0..17)
NBAND = BAND + 3    # band slots: distance tile t = -1 .. 33 at slot t + 1
SINK, NEG_SLOT = NBAND, NBAND + 1
NSLOTS = NBAND + 2
RING_TILES, RING_MOD = 36, 34
NEG = -30000.0


def _largest_subblock(h: int, w: int, dst: int) -> tuple[int, int]:
    """ttnn sdpa_subblock_utils determine_largest_subblock_size."""
    for sh, sw in ((2, 4), (4, 2), (1, 8), (8, 1), (1, 7), (7, 1), (2, 3), (3, 2), (1, 6), (6, 1), (1, 5), (5, 1),
                   (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1)):
        if sh * sw <= dst and h % sh == 0 and w % sw == 0:
            return sh, sw
    return 1, 1


def _granularity(n: int, mx: int) -> int:
    g = min(n, mx)
    while g > 1 and n % g:
        g -= 1
    return g


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _dm(src, cores, ct, rt, proc, noc, defines=None):
    kw = {"defines": defines} if defines else {}  # batched serving only (single-stream programs: no defines)
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores, compile_time_args=ct, runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc), **kw)


def _batch_defines(slotted: bool, pt=None) -> list:
    """Kernel defines of a batched-serving prefill program: SLOTTED (the slot from pos word 2), PAGED (full layers:
    the pool through the page row `pt`)."""
    return ([("SLOTTED", "1")] if slotted else []) + ([("PAGED", "1")] if pt is not None else [])


class _CBs:
    """Circular buffers of one program (name -> index)."""

    def __init__(self, cores):
        self.cores, self.cbs, self.ids = cores, [], {}
        self.tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))

    def __call__(self, name, tiles, dtype):
        i = len(self.cbs)
        page = TILE_BYTES[dtype]
        self.cbs.append(ttnn.CBDescriptor(total_size=tiles * page, core_ranges=self.cores,
                                          format_descriptors=[ttnn.CBFormatDescriptor(i, dtype, page, self.tile)]))
        self.ids[name] = i

    def __getitem__(self, name):
        return self.ids[name]


def bias_table_host(proj: torch.Tensor, sliding: bool) -> torch.Tensor:
    """[37 * 18 * 32, 32] fp32: slot s = t + 1 (distance tile t = -1 .. 33), 35: sink, 36: NEG; tile k, element
    (m, j).  proj: [16, 1024] (rel_logits_proj.proj)."""
    t = torch.zeros(NSLOTS, SLOT, 32, 32)
    m = torch.arange(32).reshape(32, 1)
    j = torch.arange(32).reshape(1, 32)
    for s in range(NBAND):
        d = 32 * (s - 1) + m - j
        inside = (d >= 0) & (d < 1024)
        t[s, :16] = torch.where(inside, proj.float()[:, d.clamp(0, 1023)], torch.zeros(()))
        masked = (d < 0) | ((d >= 1024) & sliding)
        t[s, 16] = torch.where(masked, torch.full((), NEG), torch.zeros(()))
    t[SINK, 16] = torch.where(j == 0, torch.zeros(()), torch.full((), NEG)).expand(32, 32)
    t[SINK, 17] = torch.where(j == 0, torch.ones(()), torch.zeros(())).expand(32, 32)
    t[NEG_SLOT, 0] = NEG
    return t.reshape(-1, 32)


def bias_table(device, proj: torch.Tensor, sliding: bool):
    """The per-layer bias table on the device (BF16 TILE, one page per tile, 1.4 MB)."""
    host = bias_table_host(proj, sliding).reshape(1, 1, -1, 32)
    return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


class FlashPrefill:
    """See the module docstring.  One per model.  cfg / max_seq_len: the log-scale table tau(pos) (full layers; FP32
    [max_seq_len / 32 + 2 pages of 32 positions]) and the shifted-table scratch, allocated now (trace safety).
    k_chunk: keys per online-softmax step (multiple of 32); max_q_chunk: q rows per work item (<= 128: DEST / L1);
    fidelity: every matmul (HiFi4: exact BF16 products; the ttnn path's reduce fidelity); exact_exp_full: the exact
    exponential on full layers too."""

    def __init__(self, device, cfg, max_seq_len: int, k_chunk: int = 256, max_q_chunk: int = 128,
                 fidelity=ttnn.MathFidelity.HiFi4, exact_exp_full: bool = False):
        assert k_chunk % 32 == 0 and max_q_chunk % 32 == 0 and max_q_chunk <= 128
        self.device, self.Sk, self.max_Sq, self.fidelity = device, k_chunk // 32, max_q_chunk // 32, fidelity
        self.exact_exp_full = exact_exp_full
        g = device.compute_with_storage_grid_size()
        self.grid = (g.x, g.y)
        self.cfg = cfg
        self.kv = KVFront(device, cfg)  # the K / V front of the same prefill
        pos = torch.arange(max_seq_len + 64)
        tau = 1.0 + cfg.log_scaling_alpha * torch.log(((pos + 1).float() / cfg.log_scaling_n_floor).clamp(min=1.0))
        self.tau = ttnn.from_torch(tau.reshape(1, 1, -1, 32), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.shifted = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, NSLOTS * SLOT * 32, 32]), ttnn.bfloat16,
                                                      ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    def _sq(self, Tpt: int) -> int:
        """Largest power-of-two q chunk (tiles) dividing Tpt that still gives ~one work item per core."""
        gx, gy = self.grid
        s = self.max_Sq
        while s > 1 and (Tpt % s or NQH * Tpt // s < 0.85 * gx * gy):
            s //= 2
        return s

    def __call__(self, qg, rs, k_src, v_src, table, pos, *, sliding: bool, kn=None, vn=None, zero_block: int = 0,
                 qg_col0: int = 0, slotted: bool = False, pt=None):
        """qg [1, 1, Tp, W]: column tiles qg_col0 .. + 192 = q | gate projections (q_scale folded), rs [1, 1, Tp,
        768] (per head [Wr x (16) | 0 | Ws x | 0 ..], r_scale folded), pos: the chunk position device tensor ->
        gated attention [1, 1, Tp, 3072] BF16 (the o_proj input).  Full layers: k_src / v_src = the paged caches
        (the chunk written by kv() first); sliding: the rings and kn / vn = kv()'s aligned-frame chunk K / V.
        Batched serving: slotted = the rings are slot-major [S, 8, 1152, 128] (slot = pos word 2); pt = full layers'
        page row (int32 [1, W]: the caches are the pool)."""
        Tp = qg.shape[-2]
        assert Tp % 32 == 0 and qg.shape[-1] >= 32 * qg_col0 + 2 * NQH * 128, qg.shape
        assert tuple(rs.shape) == (1, 1, Tp, NQH * 32), rs.shape
        if sliding:
            assert kn is not None and vn is not None and kn.dtype == k_src.dtype and vn.dtype == v_src.dtype
            assert (tuple(kn.shape) == (1, NKH, Tp + 32, 128) and tuple(k_src.shape)[1:] == (NKH, 32 * RING_TILES, 128)
                    and (slotted or k_src.shape[0] == 1))
        else:
            kn, vn = k_src, v_src
        self._shift(table, pos)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, Tp, NQH * 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.device, ttnn.DRAM_MEMORY_CONFIG)
        ins = [qg, rs, k_src, v_src, table, self.shifted, pos, self.tau, out] + ([kn, vn] if sliding else [])
        ins += [pt] if pt is not None else []
        ttnn.generic_op(ins, self._program(qg, rs, k_src, v_src, kn, vn, table, pos, out, sliding, zero_block,
                                           qg_col0, slotted, pt))
        return out

    def _shift(self, table, pos):
        """The table shifted by o = start % 32 rows into self.shifted (band slots t = -1 .. 32, k = 0 .. 16; no-op
        for o = 0: the flash reader then reads `table`)."""
        units = (BAND + 2) * 17
        gx, gy = self.grid
        ncores = min(units, gx * gy)
        base, extra = divmod(units, ncores)
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(ncores)]
        core_set = row_ranges(cores)
        rt, u = ttnn.RuntimeArgs(), 0
        head = [int(t.buffer_address()) for t in (table, self.shifted, pos)]
        for i, c in enumerate(cores):
            n = base + (1 if i < extra else 0)
            rt[c.x][c.y] = head + [u, n]
            u += n
        cb = _CBs(core_set)
        cb("buf", 4, ttnn.bfloat16)
        k = _dm("attn_prefill_tshift.cpp", core_set, [SLOT, cb["buf"]], rt, ttnn.DataMovementProcessor.RISCV_1,
                ttnn.NOC.RISCV_0_default)
        ttnn.generic_op([table, self.shifted, pos], ttnn.ProgramDescriptor(kernels=[k], cbs=cb.cbs, semaphores=[]))

    def _program(self, qg, rs, kc, vc, kn, vn, table, pos, out, sliding: bool, zero_block: int, qg_col0: int,
                 slotted: bool = False, pt=None):
        Tpt = qg.shape[-2] // 32
        Sq, Sk = self._sq(Tpt), self.Sk
        qn = Tpt // Sq
        total = NQH * qn
        gx, gy = self.grid
        ncores = min(total, gx * gy)
        base, extra = divmod(total, ncores)
        per_core = base + (1 if extra else 0)
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(ncores)]
        core_set = row_ranges(cores)
        starts, s0 = [], 0
        for i in range(ncores):
            n = base + (1 if i < extra else 0)
            starts.append((s0, n))
            s0 += n

        kd, vd = kc.dtype, vc.dtype
        bf, f32 = ttnn.bfloat16, ttnn.float32
        dst = 4  # fp32 DEST, half sync
        qk_h, qk_w = _largest_subblock(Sq, Sk, dst)
        out_h, out_w = _largest_subblock(Sq, DHT, dst)
        dbl = 2 if per_core > 1 else 1
        cb = _CBs(core_set)
        cb("qr", Sq * DHT * dbl, bf)      # projection rows: q
        cb("rr", Sq, bf)                  # r
        cb("g", Sq * DHT, bf)             # gate (single-buffered: read at the item start, used at its end; L1)
        cb("tau", 1 if sliding else Sq * 2, f32)  # full layers: tau per q tile in column 0 (reader zero background)
        cb("ra", Sq, bf)                  # finished r_aug -> reader
        cb("q", Sq * DHT, bf)             # finished q
        cb("k", Sk * DHT * 2, kd)
        cb("v", Sk * DHT * 2, vd)
        cb("qt", Sq * SLOT, bf)
        cb("tbl", SLOT * 2, bf)
        cb("bias", Sq * (BAND + 2) + 1 + Sq, bf)  # band tiles t = -1 .. 32, NEG, sink tiles
        cb("scale", 1, bf)
        cb("col", 1, bf)
        cb("qk", Sq * Sk, f32)
        # running output in bf16 as the stock op: an fp32 CB breaks the packer L1 accumulation of the rescaled
        # previous output (lost whenever a fully masked K chunk follows a contributing one)
        cb("oA", Sq * DHT, bf)
        cb("oB", Sq * DHT, bf)
        cb("mA", Sq, bf)
        cb("mB", Sq, bf)
        cb("sA", Sq, f32)
        cb("sB", Sq, f32)
        cb("emd", Sq, bf)
        cb("out", Sq * DHT, bf)
        cb("fin", Sq * DHT, bf)
        cb("c", 3, bf)
        cb("p", 1, bf)
        cb("x", 2, bf)
        cb("sq", DHT, f32)
        cb("fr", 1, f32)
        cb("one", 1, f32)  # SUM reduce scaler in the squares' format (as ttnn's reduce of an FP32 tensor)

        reader_ct = [NQH, NKH, DHT, Sq, Sk, qn, Tpt] + [cb[n] for n in (
            "qr", "rr", "g", "tau", "ra", "k", "v", "qt", "tbl", "c", "p")] + [TILE_BYTES[kd], TILE_BYTES[vd],
                                                                               int(sliding)]
        one = 0x3F803F80  # two bf16 1.0
        writer_ct = [NQH, DHT, Sq, qn, cb["fin"], cb["scale"], cb["col"], one, cb["one"]]
        cfg = self.cfg
        compute_ct = [Sq, Sk, DHT, qn, int(sliding), _f32_bits(1.0)]
        compute_ct += [cb[n] for n in ("qr", "rr", "g", "tau", "ra", "q", "k", "v", "qt", "tbl", "bias", "scale",
                                       "col", "qk", "oA", "oB", "mA", "mB", "sA", "sB", "emd", "out", "fin", "c",
                                       "p", "x", "sq", "fr", "one")]
        compute_ct += [qk_w, qk_h, out_w, out_h]
        post = 128 ** -0.5
        compute_ct += [_f32_bits(cfg.qk_l2_clip * post), _f32_bits(post), _f32_bits(cfg.rel_l2_clip), _f32_bits(1.0)]
        defines = [("STATS_GRANULARITY", str(_granularity(Sq, dst))),
                   ("SUB_EXP_GRANULARITY", str(_granularity(Sk, dst))),
                   ("MUL_BCAST_GRANULARITY", str(_granularity(Sq * Sk, dst))),
                   ("DHT_GRANULARITY", str(_granularity(DHT, dst))),
                   ("REDUCE_GRANULARITY", str(_granularity(Sq, dst // 2))),
                   ("EXP_APPROX_MODE", "0")]
        if sliding or self.exact_exp_full:
            defines.append(("EXACT_EXP", "1"))

        addrs = [int(t.buffer_address()) for t in (qg, rs, kc, vc, kn, vn, table, self.shifted, pos, self.tau)]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c, (gs, gn) in zip(cores, starts):
            reader_rt[c.x][c.y] = addrs + [zero_block, qg.shape[-1] // 32, qg_col0, gs, gn] + (
                [int(pt.buffer_address())] if pt is not None else [])
            writer_rt[c.x][c.y] = [int(out.buffer_address()), gs, gn]
            compute_rt[c.x][c.y] = [gs, gn]

        compute = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                               math_approx_mode=False, dst_full_sync_en=False)
        unpack = [ttnn.UnpackToDestMode.Default] * 64
        unpack[cb["tau"]] = ttnn.UnpackToDestMode.UnpackToDestFp32  # exact tau / factor reloads
        unpack[cb["fr"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode.extend(unpack)
        kernels = [
            _dm("attn_prefill_reader.cpp", core_set, reader_ct, reader_rt, ttnn.DataMovementProcessor.RISCV_1,
                ttnn.NOC.RISCV_0_default, _batch_defines(slotted and sliding, pt)),
            _dm("attn_prefill_writer.cpp", core_set, writer_ct, writer_rt, ttnn.DataMovementProcessor.RISCV_0,
                ttnn.NOC.RISCV_1_default),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "attn_prefill_compute.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core_set,
                compile_time_args=compute_ct, defines=defines, runtime_args=compute_rt, config=compute),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cb.cbs, semaphores=[])

    def ring_write(self, kn, vn, k_ring, v_ring, pos, slotted: bool = False):
        """Sliding layers, after the chunk's attention: the aligned-frame chunk K / V kn / vn [1, 8, Tp + 32, 128]
        (the ring's format; row tile i = position tile start / 32 + i, start from the device tensor `pos`) into ring
        slot tiles (start / 32 + i) % 34: the last min(Tp / 32 + 1, 34) row tiles (rows past the chunk land in the
        dead slots of the next 64 positions; tile 0 rows before an unaligned start hold the ring's own rows).  An
        aligned start (from `pos`) skips the last tile: KVFront leaves it unwritten (uninitialized DRAM, possibly
        non-finite), and a non-finite dead slot would poison the decode softmax.  slotted: slot-major rings [S, 8,
        1152, 128], the sequence slot from pos word 2 (batched serving)."""
        assert kn.dtype == k_ring.dtype and vn.dtype == v_ring.dtype
        Tq = kn.shape[-2] // 32  # Tpt + 1
        first = max(0, Tq - RING_MOD)
        n = Tq - first
        units = NKH * n
        gx, gy = self.grid
        ncores = min(units, 64)
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(ncores)]
        per = -(-units // ncores)
        rt = ttnn.RuntimeArgs()
        addrs = [int(t.buffer_address()) for t in (kn, vn, k_ring, v_ring, pos)]
        for ci, c in enumerate(cores):
            u0, u1 = ci * per, min(units, (ci + 1) * per)
            rt[c.x][c.y] = addrs + [first, u0, max(u1 - u0, 0)]
        core_set = row_ranges(cores)
        kb, vb = TILE_BYTES[kn.dtype], TILE_BYTES[vn.dtype]
        cb = _CBs(core_set)
        cb("buf", 2 * DHT + 1, ttnn.bfloat16)
        k = _dm("attn_prefill_ring.cpp", core_set, [NKH, DHT, Tq, n, kb, vb, cb["buf"]], rt,
                ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_0_default, _batch_defines(slotted))
        ttnn.generic_op([kn, vn, k_ring, v_ring, pos], ttnn.ProgramDescriptor(kernels=[k], cbs=cb.cbs, semaphores=[]))

    def conv_update(self, k_lin, v_lin, conv_ring, states, pos, slotted: bool = False):
        """After the K / V front read them: the chunk's last <= 32 valid positions of the k / v linear outputs
        (k_lin / v_lin as in KVFront) into the conv-input ring (row pos % 32) and the 3 shift states (stock decode),
        start / valid from the device tensor pos (kernels/fast/attn_prefill_conv.cpp).  slotted: slot-major conv
        rings [S, 1, 32, 2048], the sequence slot from pos word 2 (batched serving)."""
        Tpt = k_lin[0].shape[-2] // 32
        gx, gy = self.grid
        ncores = 32
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(ncores)]
        core_set = row_ranges(cores)
        rt = ttnn.RuntimeArgs()
        head = [int(k_lin[0].buffer_address()), k_lin[1], k_lin[0].shape[-1] // 32,
                int(v_lin[0].buffer_address()), v_lin[1], v_lin[0].shape[-1] // 32]
        head += [int(t.buffer_address()) for t in (conv_ring, *states, pos)]
        for i, c in enumerate(cores):
            rt[c.x][c.y] = head + [2 * i, 2]
        cb = _CBs(core_set)
        cb("buf", 5, ttnn.bfloat16)
        kd = _dm("attn_prefill_conv.cpp", core_set, [Tpt, cb["buf"]], rt, ttnn.DataMovementProcessor.RISCV_1,
                 ttnn.NOC.RISCV_0_default, _batch_defines(slotted))
        ins = list({id(t): t for t in (k_lin[0], v_lin[0], pos, *states, conv_ring)}.values())
        ttnn.generic_op(ins, ttnn.ProgramDescriptor(kernels=[kd], cbs=cb.cbs, semaphores=[]))


class KVFront:
    """The chunk's K / V (attention.py semantics: short conv with the conv-input ring's left context, k L2 clip,
    v RMSNorm) in ONE program, written in the aligned frame (row tile i' = position tile start / 32 + i', start from
    the device tensor `pos`; tile 0 rows before the start keep the cache's rows; chunk rows >= valid are read as zero,
    so every row written is finite): full layers straight into the paged K / V caches (their formats), sliding layers
    into new [1, 8, Tp + 32, 128] tensors in the ring's format (the last row tile only for an unaligned start).
    Kernels: kernels/fast/attn_prefill_kv_{reader,compute,writer}.cpp.  The conv-input ring / shift states are NOT
    updated here (they are read by this program; FlashPrefill.conv_update)."""

    def __init__(self, device, cfg):
        self.device, self.cfg = device, cfg
        g = device.compute_with_storage_grid_size()
        self.grid = (g.x, g.y)

    def __call__(self, k_lin, v_lin, kd, vd, conv_ring, taps, k_dst, v_dst, pos, *, sliding: bool, nblocks: int = 0,
                 slotted: bool = False, pt=None):
        """k_lin / v_lin: (tensor [1, 1, Tp, W], first column tile) of the Wk xk / Wv xv linear outputs (1024 columns
        each), kd / vd: full layers the dyn tap projections [1, 1, Tp, 32] (else None), conv_ring [1, 1, 32, 2048],
        taps [1, 1, 128, 2048] (lag row tiles: sliding FP32 incl. residual / scale, full BF16 static).
        k_dst / v_dst: full: the paged caches (written in place, returns None); sliding: the rings (read for the
        rows before an unaligned start), returns the new (kn, vn).  nblocks: cache blocks of positions (max_seq_len /
        64).  Batched serving: slotted = slot-major conv rings / rings (slot = pos word 2); pt = full layers' page
        row (int32 [1, W]): the caches are the pool, tiles past the chunk's valid rows are not written."""
        Tp = k_lin[0].shape[-2]
        Tpt = Tp // 32
        if sliding:
            shape = ttnn.Shape([1, NKH, Tp + 32, 128])
            kn = ttnn.allocate_tensor_on_device(shape, k_dst.dtype, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG)
            vn = ttnn.allocate_tensor_on_device(shape, v_dst.dtype, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG)
            outs = (kn, vn)
        else:
            outs = (k_dst, v_dst)
        dk_t, dv_t = (kd, vd) if kd is not None else (k_lin[0], k_lin[0])
        units = 2 * NKH * (Tpt + 1)
        gx, gy = self.grid
        ncores = min(units, gx * gy)
        base, extra = divmod(units, ncores)
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(ncores)]
        core_set = row_ranges(cores)
        okd, ovd = outs[0].dtype, outs[1].dtype
        bf, f32 = ttnn.bfloat16, ttnn.float32
        tap_dt = taps.dtype
        nmats = 12 if sliding else 44
        cb = _CBs(core_set)
        cb("x", 6, bf)
        cb("tap", 8, tap_dt)
        cb("old", 4, okd)
        cb("oldv", 4, ovd)
        cb("dyn", 4, bf)
        cb("m", nmats, bf)
        cb("p", 1, bf)
        cb("e0", 1, bf)
        cb("e", 4, bf)
        cb("y", 4, bf)
        cb("sq", 4, f32)
        cb("fr", 1, f32)
        cb("r16", 2, bf)
        cb("one", 1, f32)  # the reduce scaler in the squares' format (as ttnn's reduce of an FP32 tensor)
        cb("ok", 8, okd)
        cb("ov", 8, ovd)
        cb("rs", 1, bf)
        cb("ws", 1, bf)
        cb("w", 4, f32)
        cb("acc", 1, f32)
        cb("x0", 1, bf)
        cfg = self.cfg
        reader_ct = [Tpt] + [cb[n] for n in ("x", "tap", "old", "oldv", "dyn", "m", "p")] + [
            TILE_BYTES[okd], TILE_BYTES[ovd], int(sliding), cb["rs"]]
        writer_ct = [Tpt, cb["ok"], cb["ov"], TILE_BYTES[okd], TILE_BYTES[ovd], int(sliding), cb["ws"], cb["one"]]
        compute_ct = [Tpt, int(sliding)] + [cb[n] for n in ("x", "tap", "old", "oldv", "dyn", "m", "p", "e0", "e",
                                                             "y", "sq", "fr", "r16", "one", "ok", "ov")]
        compute_ct += [_f32_bits(cfg.qk_l2_clip), _f32_bits(1.0 / 128), _f32_bits(cfg.eps),
                       _f32_bits(1.0 / (2 * cfg.conv_kernel + 1) ** 0.5), int(okd != bf), int(ovd != bf),
                       cb["w"], cb["acc"], cb["x0"]]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        u = 0
        for i, c in enumerate(cores):
            n = base + (1 if i < extra else 0)
            reader_rt[c.x][c.y] = ([int(k_lin[0].buffer_address()), k_lin[1], k_lin[0].shape[-1] // 32,
                                    int(v_lin[0].buffer_address()), v_lin[1], v_lin[0].shape[-1] // 32]
                                   + [int(t.buffer_address()) for t in (conv_ring, taps, dk_t, dv_t, k_dst, v_dst, pos)]
                                   + [u, n] + ([int(pt.buffer_address())] if pt is not None else []))
            writer_rt[c.x][c.y] = [int(outs[0].buffer_address()), int(outs[1].buffer_address()),
                                   int(pos.buffer_address()), nblocks, u, n] + (
                [int(pt.buffer_address())] if pt is not None else [])
            compute_rt[c.x][c.y] = [u, n]
            u += n

        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False, dst_full_sync_en=True)
        unpack = [ttnn.UnpackToDestMode.Default] * 64
        unpack[cb["fr"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        unpack[cb["w"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        if tap_dt == f32:
            unpack[cb["tap"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode.extend(unpack)
        kernels = [
            _dm("attn_prefill_kv_reader.cpp", core_set, reader_ct, reader_rt, ttnn.DataMovementProcessor.RISCV_1,
                ttnn.NOC.RISCV_0_default, _batch_defines(slotted, pt)),
            _dm("attn_prefill_kv_writer.cpp", core_set, writer_ct, writer_rt, ttnn.DataMovementProcessor.RISCV_0,
                ttnn.NOC.RISCV_1_default, _batch_defines(False, pt)),
            ttnn.KernelDescriptor(kernel_source=str(KERNELS / "attn_prefill_kv_compute.cpp"),
                                  source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core_set,
                                  compile_time_args=compute_ct, runtime_args=compute_rt, config=compute),
        ]
        ins = list({id(t): t for t in (k_lin[0], v_lin[0], conv_ring, taps, pos, k_dst, v_dst, dk_t, dv_t)}.values())
        ins += list(outs) if sliding else []
        ins += [pt] if pt is not None else []
        ttnn.generic_op(ins, ttnn.ProgramDescriptor(kernels=kernels, cbs=cb.cbs, semaphores=[]))
        return outs if sliding else None
