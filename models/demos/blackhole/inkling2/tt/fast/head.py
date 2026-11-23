# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fused LM head: logits = 30 tanh((x @ unembed^T) / 30) streamed from DRAM, + the greedy argmax, in one launch.

    head = FastHead(device, cfg, head_weight(cache, build))   # stream-layout BFP8 unembedding (WeightCache
                                                               # "unembed.<layout tag>")
    lg = head(x, rows=1, ids=tok)                # x BF16 TILE [1, 1, >=rows, H] (rows 0..rows-1 used, <= 32)
                                                 # -> FP32 TILE [1, 1, rows, V_pad]; ids (optional, rows <= 16):
                                                 # uint32 ROW_MAJOR DRAM, word r = argmax of logit row r

kernels/fast/head_stream.cpp: banks * cores_per_bank workers (placed by moe.stream_workers: link-disjoint NOC0
response paths) stream their bank-contiguous column runs of the unembedding (decode_matmul.py's stream layout),
accumulate K in the stock linear's order (in0_block_w = 2, FP32 packer L1 accumulation, then the bias stage's srcA
pass) on 16-row tiles, apply the softcap in DEST and write the FP32 logit rows; the writer scans each row's maximum
while the stream runs and worker 0 merges the per-worker maxima (ties -> lowest column, as torch.argmax).  The values
equal the stock head (layers.LMHead: ttnn.linear + pad bias, mul / tanh / mul) bitwise, padding columns included
(-30): tests/test_fast_head.py.  So greedy decoding is unchanged by construction, and the device sampler reads the
same logits.  Traced: 1.07 ms (rows = 1, argmax included) vs 1.91 ms for stock head + slice + LogitArgmax.
"""
from __future__ import annotations

import struct

import torch
import ttnn

from .common import CachedProgram, KERNELS, TILE_BYTES, row_ranges
from .decode_matmul import Preset, cached_stream_weight, stream_columns

_KERNEL = str(KERNELS / "head_stream.cpp")
# 2304 x 200064 (K = 72 tiles): 5 workers per DRAM bank, w = 4 columns per subblock, kb = 4 K tiles per block
# (sweep: c5 1.066 ms, c6 1.118, c8 1.17; the stream alone ~1.04 ms = 470 GB/s)
HEAD_PRESET = Preset(5, 4, 4, 3, 2)


def _bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def unembed_t(source, cfg):
    """[H, V_pad] W^T of the unembedding (columns >= unpadded vocab zero), BF16 as the checkpoint."""
    w = source.tensor("model.llm.unembed.weight")[: cfg.unpadded_vocab].T
    return torch.nn.functional.pad(w, (0, cfg.padded_unembed - cfg.unpadded_vocab))


def head_weight(cache, build, preset: Preset = HEAD_PRESET, dtype=ttnn.bfloat8_b):
    """The stream-layout unembedding on the device (WeightCache "unembed.<layout tag>"); build() -> [H, V_pad] W^T
    torch, called only when the cache misses."""
    return cached_stream_weight(cache, "unembed", preset, build, dtype)


class FastHead:
    """weight: head_weight(cache, build, preset) (the same preset)."""

    def __init__(self, device, cfg, weight, preset: Preset = HEAD_PRESET):
        from ..moe import stream_workers

        p = preset
        if p.k_split or p.glu or p.kb % 2:
            raise ValueError(f"FastHead: preset {p} (k_split / glu are fixed by the kernel; kb must be even)")
        self.device, self.cfg, self.preset = device, cfg, p
        self.cap = float(cfg.logit_softcap)
        self.w = weight
        self.k_tiles = int(self.w.padded_shape[-2]) // 32
        if self.k_tiles % p.kb:
            raise ValueError(f"FastHead: kb {p.kb} must divide K tiles {self.k_tiles}")
        self.n_out = cfg.padded_unembed
        n_valid = self.n_out // 32
        self.tile_bytes = TILE_BYTES[self.w.dtype]
        columns = stream_columns(int(self.w.padded_shape[-1]) // 32, p.cores_per_bank)
        cores = stream_workers(device, p.cores_per_bank)
        self.workers = []  # (first column, core, [bank, run bytes, columns, first column]) in column order
        for b, (cols, bank_cores) in enumerate(zip(columns, cores)):
            run = 0
            for (first, n), core in zip(cols, bank_cores):
                nv = min(n, n_valid - first)
                if nv > 0:
                    self.workers.append((first, core, [b, run * self.tile_bytes, nv, first]))
                run += n * self.k_tiles
        self.workers.sort(key=lambda w: w[0])
        self.core_set = row_ranges([c for _, c, _ in self.workers])
        self._progs = {}

    def _program(self, x, out, ids, rows: int):
        from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
            PerCoreRuntimeArgsDescriptor,
            UnifiedKernelDescriptor,
        )

        p, cs = self.preset, self.core_set
        n = len(self.workers)
        gather = self.device.worker_core_from_logical_core(self.workers[0][1])
        argmax = ids is not None
        slot_bytes = ((rows * 8 + 15) // 16) * 16
        # activation / output CB tiles [th, 32]: the matmul computes th rows instead of 32 (row values unchanged)
        th = 16 if rows <= 16 else 32
        common = [("kt", self.k_tiles), ("w", p.w), ("kb", p.kb), ("rows", rows), ("th", th)]
        per_core = [(c, a + [i]) for i, (_, c, a) in enumerate(self.workers)]
        addrs = [int(self.w.buffer_address()), int(out.buffer_address()), int(x.buffer_address()),
                 int(ids.buffer_address()) if argmax else 0]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        kernels = UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=cs,
            ncrisc_named_compile_time_args=common + [("in1_tile_bytes", self.tile_bytes), ("slots", p.slots),
                                                     ("outstanding", p.outstanding)],
            brisc_named_compile_time_args=common + [
                ("valid", self.cfg.unpadded_vocab), ("argmax", int(argmax)), ("workers", n),
                ("gather_x", gather.x), ("gather_y", gather.y),
                ("ids_page", int(ids.padded_shape[-1]) * 4 if argmax else 4)],
            trisc_named_compile_time_args=common + [("inv_cap", _bits(1.0 / self.cap)), ("cap", _bits(self.cap))],
            ncrisc_common_runtime_args=addrs,
            brisc_common_runtime_args=addrs,
            trisc_common_runtime_args=addrs,
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=per_core, brisc_args=per_core, trisc_args=per_core),
        ).get_kernel_descriptors().kernels

        def cb(index, dtype, page, pages, h=None):
            fmt = (ttnn.CBFormatDescriptor(index, dtype, page, ttnn.TileDescriptor(ttnn.Tile([h, 32]))) if h
                   else ttnn.CBFormatDescriptor(index, dtype, page))
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs, format_descriptors=[fmt])

        cbs = [cb(0, ttnn.bfloat16, th * 64, self.k_tiles, th),
               cb(1, self.w.dtype, self.tile_bytes, p.slots * p.kb * p.w, 32),
               cb(3, ttnn.bfloat16, th * 64, 1, th),
               cb(4, ttnn.uint32, slot_bytes * n, 1),
               cb(5, ttnn.uint32, slot_bytes, 1),
               cb(16, ttnn.float32, th * 128, 2 * p.w, th),
               cb(24, ttnn.float32, th * 128, p.w, th)]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cs, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)

    def __call__(self, x, rows: int | None = None, ids=None, out=None):
        """x: BF16 TILE interleaved [1, 1, >= rows, K]; returns FP32 TILE [1, 1, rows, V_pad] (DRAM; rows 0..rows-1
        written).  ids: write the per-row argmax (rows <= 16) to words 0..rows-1 of its first page."""
        rows = int(rows or x.shape[-2])
        if (x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT or int(x.shape[-1]) != self.k_tiles * 32
                or not 1 <= rows <= min(32, int(x.shape[-2])) or int(x.padded_shape[-2]) != 32
                or x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            raise ValueError("FastHead: x must be BF16 TILE interleaved [1, 1, rows<=32, H] in one tile row")
        if ids is not None and (rows > 16 or ids.layout != ttnn.ROW_MAJOR_LAYOUT
                                or ids.dtype not in (ttnn.uint32, ttnn.int32) or int(ids.padded_shape[-1]) < rows
                                or ids.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError("FastHead: ids must be a DRAM uint32 ROW_MAJOR row and rows <= 16")
        if out is None:
            out = ttnn.empty([1, 1, rows, self.n_out], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                             device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tensors = [x, self.w, out] + ([ids] if ids is not None else [])
        self._progs.setdefault((rows, ids is not None), CachedProgram()).run(
            tensors, lambda: self._program(x, out, ids, rows))
        return out
