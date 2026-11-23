# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused device ops of the block drafter (tt/drafter.py; owner Spec).

    rm = RowMove(device, moves, workers=64)         # moves: see below (host-static)
    rm(tensors, pos=None)                           # one generic_op (kernels/fast/draft_rowmove.cpp)
    ma = MarkovArgmax(device, fold_w, preset, valid, brow, ck, users=1)
    ma(latent, base, fold_b, out, log)              # one generic_op (kernels/fast/draft_markov.cpp)

RowMove: copies 32-value tile-row segments between BF16 TILE DRAM-interleaved tensors.  A move is
(src, src_row, src_ct, dst, dst_row, dst_ct, ntiles): `ntiles` column tiles starting at column tile src_ct of
padded row src_row of tensors[src] -> padded row dst_row from column tile dst_ct of tensors[dst]; rows count every
leading dim folded in (a [1, U, 32, 128] tensor: row u*32 + h).  dst_row = ("cache", user, head, nkv): the row of
position pos[user] (int32 row-major device tensor, <= 32 entries) of kv head `head` in a paged [blocks, nkv, 64, D]
cache with the identity page table.  Each face row is one 32 B write, so moves into the same cache tile never race.

MarkovArgmax: token = argmax_v bf16(base[brow, v] + fold_b[v] + latent[0] . fold[:, v]) over v < valid (ties ->
lowest id), the fold streamed once from DRAM (stream layout of fast/decode_matmul.py) with the add + argmax fused:
no FP32 [32, V] tensors.  base: [1, 1, rows, >= Vp] BF16 TILE (the draft base logits), fold_b: [1, 1, 32, >= Vp] BF16
TILE with row 0 = fold_b, latent: [1, 1, >=1, rank] BF16 TILE (row 0), out: uint32 row-major (word 0 = the id), log:
uint32 row-major [>= brow, 32]: the id also goes to word 0 of row brow - 1 (brow >= 1; the drafter's one-read log).
users = U (2..4, batched drafter, block rows interleaved: row r of user u = r * U + u): U argmaxes in one fold stream,
user u = latent row u and base / fold_b row brow * U + u (fold_b must then hold the bias in EVERY row) -> word u of out
and of log row brow - 1 (each row computed exactly as the users = 1 program computes row 0).

MarkovRNN: one recurrent step (kernels/fast/draft_rnn.cpp, 8 cores): raw = ctx[crow] + [state | w1[prev]] W_sp (FP32),
latent = tanh(raw[512:768]), state' = gate * state + (1 - gate) * tanh(raw[256:512]) with gate = sigmoid(raw[0:256]);
prev = word 0 of a uint32 row (the previous MarkovArgmax output), w1 the row-major [V, 256] BF16 table, state /
latent [1, 1, 32, 256] BF16 TILE (row 0), ctx FP32 [1, 1, rows, 768] TILE, W_sp [512, 768] BF16 TILE.
users = U (<= 4): user u = prev word u, state / latent row u, ctx row crow * U + u.
"""
from __future__ import annotations

import ttnn

from .common import DRAM, KERNELS, TILE_BYTES, CachedProgram, row_ranges
from .decode_matmul import Preset, stream_columns, stream_cores

_ROWMOVE = str(KERNELS / "draft_rowmove.cpp")
_MARKOV = str(KERNELS / "draft_markov.cpp")
_RNN = str(KERNELS / "draft_rnn.cpp")


def _ukd():
    from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
        PerCoreRuntimeArgsDescriptor,
        UnifiedKernelDescriptor,
    )
    return UnifiedKernelDescriptor, PerCoreRuntimeArgsDescriptor


class RowMove:
    def __init__(self, device, moves, workers: int = 64):
        self.device = device
        grid = device.compute_with_storage_grid_size()
        ncores = min(-(-workers // 2), grid.x * grid.y)
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
        self.core_set = row_ranges(self.cores)
        enc = []
        for mv in moves:
            src, srow, sct, dst, drow, dct, nt = mv
            assert 0 <= src < 16 and 0 <= dst < 16 and 1 <= nt < 256 and sct < 65536 and dct < 65536
            if isinstance(drow, tuple):
                _, user, head, nkv = drow
                assert user < 32 and head < 256 and nkv < 256
                w0, w3 = src | dst << 4 | 1 << 8 | nt << 16, user | head << 8 | nkv << 16
            else:
                w0, w3 = src | dst << 4 | nt << 16, int(drow)
            enc.append((nt, [w0, int(srow), sct | dct << 16, w3]))
        # balance face-row segments over 2 * ncores workers (BRISC / NCRISC of each core)
        nw = 2 * ncores
        loads, lists = [0] * nw, [[] for _ in range(nw)]
        for nt, words in sorted(enc, key=lambda e: -e[0]):
            w = min(range(nw), key=lambda i: loads[i])
            loads[w] += nt
            lists[w] += words
        self.args = [[len(lists[w]) // 4] + lists[w] for w in range(nw)]
        self.buf_bytes = 128 + 64 + 2 * 64 * max(loads)
        self._prog = CachedProgram()
        self._dummy = None

    def _build(self, tensors, pos):
        UKD, PCR = _ukd()
        common = [int(pos.buffer_address()) if pos is not None else 0]
        for t in tensors:
            common += [int(t.buffer_address()), int(t.padded_shape[-1]) // 32]
        brisc = [(c, self.args[2 * i]) for i, c in enumerate(self.cores)]
        ncrisc = [(c, self.args[2 * i + 1]) for i, c in enumerate(self.cores)]
        k = UKD(kernel_source=_ROWMOVE, core_ranges=self.core_set,
                brisc_common_runtime_args=common, ncrisc_common_runtime_args=common,
                per_core_runtime_args_descriptor=PCR(brisc_args=brisc, ncrisc_args=ncrisc))
        cbs = [ttnn.CBDescriptor(total_size=self.buf_bytes, core_ranges=self.core_set,
                                 format_descriptors=[ttnn.CBFormatDescriptor(cb, ttnn.uint32, self.buf_bytes)])
               for cb in (0, 1)]
        return ttnn.ProgramDescriptor(kernels=k.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, tensors, pos=None):
        for t in tensors:
            if (t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or t.memory_config() != DRAM):
                raise ValueError("RowMove: tensors must be BF16 TILE DRAM-interleaved")
        io = list(tensors) + ([pos] if pos is not None else [])
        self._prog.run(io, lambda: self._build(tensors, pos))


class MarkovArgmax:
    def __init__(self, device, fold_w, preset: Preset, valid: int, brow: int, compute_kernel_config, users: int = 1):
        p = preset
        if p.glu or p.k_split or 2 * p.w > 4:
            raise ValueError(f"MarkovArgmax: bad preset {p}")
        self.device, self.preset, self.w = device, p, fold_w
        self.k_tiles = int(fold_w.padded_shape[-2]) // 32
        n_tiles = int(fold_w.padded_shape[-1]) // 32
        if self.k_tiles % p.kb:
            raise ValueError("MarkovArgmax: kb must divide the K tiles")
        self.valid, self.brow, self.users = int(valid), int(brow), int(users)
        if not 1 <= self.users <= 4 or self.brow * self.users + self.users - 1 >= 32:
            raise ValueError(f"MarkovArgmax: users {users} / brow {brow} out of range")
        n_valid = -(-self.valid // 32)
        self.tile_bytes = TILE_BYTES[fold_w.dtype]
        self.fidelity = compute_kernel_config.math_fidelity
        columns = stream_columns(n_tiles, p.cores_per_bank)
        _, cores = stream_cores(device, p.cores_per_bank)
        self.core_args = []
        for b, (cols, bank_cores) in enumerate(zip(columns, cores)):
            run = 0
            for (first, n), core in zip(cols, bank_cores):
                nv = min(n, n_valid - first)
                if nv > 0:
                    self.core_args.append((core, [b, run * self.tile_bytes, nv, first, len(self.core_args)]))
                run += n * self.k_tiles
        self.core_set = row_ranges([c for c, _ in self.core_args])
        self.gather = device.worker_core_from_logical_core(self.core_args[0][0])
        self._prog = CachedProgram()

    def _build(self, latent, base, fold_b, out, log):
        UKD, PCR = _ukd()
        p, cs = self.preset, self.core_set
        workers = len(self.core_args)
        common = [("kt", self.k_tiles), ("w", p.w), ("kb", p.kb)]
        addrs = [int(t.buffer_address()) for t in (self.w, latent, base, fold_b, out, log)]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        per_core = [(c, a) for c, a in self.core_args]
        k = UKD(kernel_source=_MARKOV, core_ranges=cs,
                ncrisc_named_compile_time_args=common + [("in1_tile_bytes", self.tile_bytes), ("slots", p.slots),
                                                         ("outstanding", p.outstanding)],
                brisc_named_compile_time_args=common + [("brow", self.brow), ("users", self.users),
                                                        ("valid", self.valid), ("workers", workers),
                                                        ("gather_x", self.gather.x), ("gather_y", self.gather.y)],
                trisc_named_compile_time_args=common,
                ncrisc_common_runtime_args=addrs, brisc_common_runtime_args=addrs, trisc_common_runtime_args=addrs,
                trisc_compute_config=compute,
                per_core_runtime_args_descriptor=PCR(ncrisc_args=per_core, brisc_args=per_core, trisc_args=per_core))
        tile = ttnn.TileDescriptor(ttnn.Tile([32, 32]))

        def cb(index, dtype, page, pages):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        slot = 16 if self.users <= 2 else 32  # gather bytes per worker: (key, idx) per user
        ring = 2 if self.users == 1 else 3    # base / fold_b subblocks (users > 1: reads of s + 2 overlap the scan of s)
        cbs = [cb(0, ttnn.bfloat16, 2048, self.k_tiles),
               cb(1, self.w.dtype, self.tile_bytes, p.slots * p.kb * p.w),
               cb(3, ttnn.bfloat16, 2048, ring * p.w),
               cb(4, ttnn.bfloat16, 2048, ring * p.w),
               ttnn.CBDescriptor(total_size=max(256, 2 * p.w * 64), core_ranges=cs,
                                 format_descriptors=[ttnn.CBFormatDescriptor(5, ttnn.uint32, max(256, 2 * p.w * 64))]),
               ttnn.CBDescriptor(total_size=slot * workers, core_ranges=cs,
                                 format_descriptors=[ttnn.CBFormatDescriptor(6, ttnn.uint32, slot * workers)]),
               cb(16, ttnn.float32, 4096, 2 * p.w)]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cs, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=k.get_kernel_descriptors().kernels, cbs=cbs, semaphores=sems)

    def __call__(self, latent, base, fold_b, out, log):
        for t in (latent, base, fold_b):
            if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or t.memory_config() != DRAM:
                raise ValueError("MarkovArgmax: latent / base / fold_b must be BF16 TILE DRAM")
        if int(latent.padded_shape[-1]) != self.k_tiles * 32:
            raise ValueError("MarkovArgmax: latent width != fold K")
        if self.brow and int(log.padded_shape[0]) < self.brow:
            raise ValueError("MarkovArgmax: the draft log needs a page per draft")
        self._prog.run([self.w, latent, base, fold_b, out, log],
                       lambda: self._build(latent, base, fold_b, out, log))
        return out


class MarkovRNN:
    def __init__(self, device, w_sp, crow: int, users: int = 1):
        if (int(w_sp.padded_shape[-2]), int(w_sp.padded_shape[-1])) != (512, 768) or w_sp.dtype != ttnn.bfloat16:
            raise ValueError("MarkovRNN: W_sp must be BF16 [512, 768]")
        self.device, self.w, self.crow, self.users = device, w_sp, int(crow), int(users)
        if not 1 <= self.users <= 4 or self.crow * self.users + self.users - 1 >= 32:
            raise ValueError(f"MarkovRNN: users {users} / crow {crow} out of range")
        self.cores = [ttnn.CoreCoord(i, 0) for i in range(8)]
        self.core_set = row_ranges(self.cores)
        self._prog = CachedProgram()

    def _build(self, prev, w1, state, ctx, latent, state_out):
        UKD, PCR = _ukd()
        cs = self.core_set
        addrs = [int(t.buffer_address()) for t in (prev, w1, state, ctx, self.w, latent, state_out)]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        mode[2] = ttnn.UnpackToDestMode.UnpackToDestFp32  # ctx rows: exact FP32 into DEST
        compute.unpack_to_dest_mode.extend(mode)
        per_core = [(c, [j]) for j, c in enumerate(self.cores)]
        k = UKD(kernel_source=_RNN, core_ranges=cs,
                brisc_named_compile_time_args=[("crow", self.crow), ("users", self.users)],
                ncrisc_common_runtime_args=addrs, brisc_common_runtime_args=addrs, trisc_common_runtime_args=addrs,
                trisc_compute_config=compute,
                per_core_runtime_args_descriptor=PCR(ncrisc_args=per_core, brisc_args=per_core, trisc_args=per_core))
        tile = ttnn.TileDescriptor(ttnn.Tile([32, 32]))

        def cb(index, dtype, page, pages):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

        stage = max(1024, 128 + 512 * self.users)  # ids row + one w1 row per user (+ 64 B alignment)
        cbs = [cb(0, ttnn.bfloat16, 2048, 16), cb(1, ttnn.bfloat16, 2048, 48), cb(2, ttnn.float32, 4096, 3),
               cb(3, ttnn.bfloat16, 2048, 1),
               ttnn.CBDescriptor(total_size=stage, core_ranges=cs,
                                 format_descriptors=[ttnn.CBFormatDescriptor(4, ttnn.uint32, stage)]),
               cb(16, ttnn.bfloat16, 2048, 1), cb(17, ttnn.bfloat16, 2048, 1)]
        return ttnn.ProgramDescriptor(kernels=k.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, prev, w1, state, ctx, latent, state_out):
        if state is state_out:
            raise ValueError("MarkovRNN: state in / out must differ")
        ts = [prev, w1, state, ctx, self.w, latent, state_out]
        self._prog.run(ts, lambda: self._build(prev, w1, state, ctx, latent, state_out))
        return latent
