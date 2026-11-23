# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Batch tables of the batched decode step (tt/NOTES-batch.md section 3) and their device-side advance.

    bt = BatchTables(device, cfg, max_seq_len)       # model constructor (batched models): persistent device tensors
    bt.write(slots, positions, pages, R)             # host rows 0..R-1 (B = len(slots) real rows, pads = row 0)
    adv = BatchAdvance(device, bt, log_pages)        # one launch: rows b < R from p_b to p_b + 1 (+ token log)
    adv(R, tok, sel, log, from_sel)                  # traceable
    bt.advanced(R)                                   # host mirror of what the replayed advance did

Device tensors (the shared contract; rows b >= R are never read):
  cur_pos     int32 ROW_MAJOR [1, 32] DRAM: word b = position of row b's input token;
  slot        int32 ROW_MAJOR [1, 32] DRAM: word b = physical slot of row b;
  tau         FP32 TILE [1, 1, 32, 32] DRAM: element (0, b) = tau(cur_pos[b]) (the values of decode_loop.tau_table,
              i.e. attention._tau evaluated per position as the single-stream tables do);
  page_table  int32 ROW_MAJOR [32, W] DRAM (W = max_seq_len // 64): row b column j = pool block of positions
              64 j .. 64 j + 63 of row b's sequence.
  tau_table   FP32 ROW_MAJOR [1, max_seq_len] (BatchAdvance's source of tau).
The host keeps a mirror of every device value and writes a tensor only when one of its words changes: cur_pos /
slot / tau (<= 4 KB each) as whole tensors, the page table as a whole tensor when any row changes (a sequence's row
changes when vLLM appends a block: once per 64 tokens per sequence).
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from ..common import DRAM
from .decode_loop import tau_table
from .token_ops import CachedProgram

KERNEL = str(Path(__file__).resolve().parent.parent / "kernels" / "fast" / "batch_advance.cpp")
MAX_ROWS = 16
WORDS = 32
LOG_WORDS = 64  # token log page: tokens 0..15 | flags 0..15 | positions 0..15 | 0


def _i32(device, t):
    return ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)


class BatchTables:
    """See the module docstring."""

    def __init__(self, device, cfg, max_seq_len: int):
        if max_seq_len % 64:
            raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of 64")
        self.device, self.cfg, self.max_seq_len = device, cfg, max_seq_len
        self.W = max_seq_len // 64
        self.tau_host = tau_table(cfg, max_seq_len).reshape(-1)
        self.tau_table = ttnn.from_torch(self.tau_host.reshape(1, -1), dtype=ttnn.float32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
        # host mirrors of the device tensors (exactly what the device holds while `known`)
        self.h_pos = torch.zeros(1, WORDS, dtype=torch.int32)
        self.h_slot = torch.zeros(1, WORDS, dtype=torch.int32)
        self.h_tau = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
        self.h_tau[0, 0, 0, :] = self.tau_host[0]
        self.h_pt = torch.zeros(WORDS, self.W, dtype=torch.int32)
        self.cur_pos = _i32(device, self.h_pos)
        self.slot = _i32(device, self.h_slot)
        self.tau = ttnn.from_torch(self.h_tau, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
                                   memory_config=DRAM)
        self.page_table = _i32(device, self.h_pt)
        self.writes = {"cur_pos": 0, "slot": 0, "tau": 0, "page_table": 0}  # host write counts (bench / tests)
        self._known = self._pt_known = True  # the mirrors equal the device tensors

    def invalidate(self):
        """Forget the mirrors (another writer touched the tables): the next write() writes every tensor."""
        self._known = self._pt_known = False

    # ------------------------------------------------------------------ host writes
    def write(self, slots, positions, pages=None, R: int | None = None):
        """Rows 0..R-1: row b < B = len(slots) = (slots[b], positions[b], pages[b]); rows B..R-1 replicate row 0.
        pages[b]: the sequence's page row (sequence of ints / 1-D int tensor, <= W entries; missing columns keep
        don't-care zeros); pages None: the page table is not touched.  Writes only the tensors that change."""
        B = len(slots)
        R = R or B
        if not 1 <= B <= R <= MAX_ROWS or len(positions) != B or (pages is not None and len(pages) != B):
            raise ValueError(f"BatchTables.write: 1 <= B ({B}) <= R ({R}) <= {MAX_ROWS}, one position / page row per row")
        src = list(range(B)) + [0] * (R - B)
        pos = torch.tensor([int(positions[i]) for i in src], dtype=torch.int32)
        if int(pos.min()) < 0 or int(pos.max()) >= self.max_seq_len:
            raise ValueError(f"positions {pos.tolist()} outside 0..{self.max_seq_len - 1}")
        slot = torch.tensor([int(slots[i]) for i in src], dtype=torch.int32)
        force = not self._known
        if force or not torch.equal(self.h_pos[0, :R], pos):
            self.h_pos[0, :R] = pos
            self._put(self.h_pos, self.cur_pos, "cur_pos")
            tau = self.tau_host[pos.long()]
            if force or not torch.equal(self.h_tau[0, 0, 0, :R], tau):
                self.h_tau[0, 0, 0, :R] = tau
                self._put(self.h_tau, self.tau, "tau", layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        if force or not torch.equal(self.h_slot[0, :R], slot):
            self.h_slot[0, :R] = slot
            self._put(self.h_slot, self.slot, "slot")
        self._known = True
        if pages is not None:
            dirty = not self._pt_known
            for b, i in enumerate(src):
                row = torch.as_tensor(pages[i], dtype=torch.int32).reshape(-1)
                n = row.numel()
                if n > self.W:
                    raise ValueError(f"page row of {n} columns > W = {self.W}")
                if not torch.equal(self.h_pt[b, :n], row):
                    self.h_pt[b, :n] = row
                    dirty = True
            if dirty:
                self._put(self.h_pt, self.page_table, "page_table")
                self._pt_known = True

    def _put(self, host, dev, name, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=layout), dev)
        self.writes[name] += 1

    def advanced(self, R: int):
        """Host mirror of one BatchAdvance of rows 0..R-1 (cur_pos + 1, tau of the new position)."""
        q = self.h_pos[0, :R] + 1
        self.h_pos[0, :R] = q
        ok = q < self.max_seq_len
        self.h_tau[0, 0, 0, :R] = torch.where(ok, self.tau_host[q.clamp(max=self.max_seq_len - 1).long()],
                                              self.h_tau[0, 0, 0, :R])

    def read(self) -> dict:
        """Device values (tests): cur_pos / slot words, tau (0, b) as fp32 bits, page table."""
        return {"cur_pos": ttnn.to_torch(self.cur_pos).reshape(-1).to(torch.int32),
                "slot": ttnn.to_torch(self.slot).reshape(-1).to(torch.int32),
                "tau": ttnn.to_torch(self.tau).float().reshape(32, 32)[0].contiguous().view(torch.int32),
                "page_table": ttnn.to_torch(self.page_table).reshape(WORDS, self.W).to(torch.int32)}

    def mirror(self) -> dict:
        return {"cur_pos": self.h_pos.reshape(-1).clone(), "slot": self.h_slot.reshape(-1).clone(),
                "tau": self.h_tau[0, 0, 0].contiguous().view(torch.int32).clone(), "page_table": self.h_pt.clone()}


class BatchAdvance:
    """One launch (kernels/fast/batch_advance.cpp): rows b < R of `tables` from p_b to p_b + 1 (cur_pos, tau), the
    tokens of the step -> tok words 0..R-1 (from_sel: the device sampler's sel pages; else already there: the LM
    head's argmax) and the token log page (q_0 % log_pages) = [tokens, flags, q_b]."""

    def __init__(self, device, tables: BatchTables, log_pages: int):
        self.device, self.tables, self.log_pages = device, tables, log_pages
        self._progs = {}

    def _build(self, R, tok, sel, log, from_sel):
        t = self.tables
        core = ttnn.CoreCoord(0, 0)
        cs = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
        named = [("rows", R), ("max_seq", t.max_seq_len), ("log_pages", self.log_pages), ("from_sel", int(from_sel)),
                 ("sel_page", int(sel.padded_shape[-1]) * 4), ("tok_page", int(tok.padded_shape[-1]) * 4)]
        common = [int(x.buffer_address()) for x in (t.cur_pos, t.tau, t.tau_table, tok, sel, log)]
        kernel = UnifiedKernelDescriptor(
            kernel_source=KERNEL, core_ranges=cs,
            brisc_named_compile_time_args=named, ncrisc_named_compile_time_args=named,
            brisc_common_runtime_args=common, ncrisc_common_runtime_args=common,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(brisc_args=[(core, [0])],
                                                                          ncrisc_args=[(core, [0])]),
        )
        cbs = [ttnn.CBDescriptor(total_size=4096, core_ranges=cs,
                                 format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, 4096)])]
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, R: int, tok, sel, log, from_sel: bool):
        if not 1 <= R <= MAX_ROWS:
            raise ValueError(f"BatchAdvance: 1 <= R <= {MAX_ROWS}")
        if int(log.padded_shape[-1]) != LOG_WORDS or int(sel.padded_shape[-2]) < R:
            raise ValueError(f"BatchAdvance: log pages of {LOG_WORDS} words, sel with >= R pages")
        t = self.tables
        tensors = [t.cur_pos, t.tau, t.tau_table, tok, sel, log]
        self._progs.setdefault((R, bool(from_sel)), CachedProgram()).run(
            tensors, lambda: self._build(R, tok, sel, log, from_sel))
