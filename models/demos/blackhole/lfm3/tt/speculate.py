# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Prompt-lookup speculative decoding for LFM3 on one P150, in shared rounds (any max_num_seqs).  Ported from the
LFM2.5 runtime (lfm2/tt/speculate.py, one sequence) and extended to TensorFold-style shared
rounds: the conv state has six rows (7 taps), the dense step's cost grows with its decode width, not its live rows.

Drafts come from each request's own tokens (prompt + output): the longest earlier occurrence of the context's last
n tokens (n_max .. n_min) proposes the tokens that followed it (PromptLookup, one per stream).  A verify round is one
traced step of W = 2, 4, 8 or 16 rows of the fast decode step in verify mode (tt/fast/NOTES.md 7).  Its rows are
grouped by sequence; every running stream rides the round, as its pending token plus its drafts (or the pending
token alone):

    group g = rows r0 .. r0 + n - 1: input tok[r0 + j] (tok[r0] = the stream's last committed token, then its drafts)
        at position p_g + j, the stream's page-table row, its conv-state slot sigma_g (= its decode row)
    row map (FastStep.vmap, map_word): every row's offset j in its group, the group length n and the slot sigma_g
    FastStep.decode_forward(verify=True): KV written for every row into its sequence's pages (one worker per 32-row
        cache tile of a group patches the group's rows), conv rows chained inside the group from its slot's 6-row state
        (row j's left context = the state and the group's rows j-6 .. j-1), every row's Bx -> per-layer candidates, the
        conv state untouched; each row's logits are bitwise those of its stream's sequential 1-row steps, whatever the
        other groups (every module of the step is row-invariant)
    row tokens: SpecArgmax (greedy; ties -> lowest id, as the 1-row device argmax) or the exact device sampler
        (tt/sampling.py) with row j's penalty history counting its group's drafts 1..j and the draw at position
        p_g + j + 1 keyed on the stream's seed
    SpecCommit: per group a = the first row whose token differs from the next draft (or is flagged for a host redraw,
        or the group's last row); slot sigma_g's conv state <- the last six rows of [state, Bx_r0 .. Bx_{r0+a}] for
        all 13 conv layers; each penalized group's sampler history += its emitted tokens; result page [a per group,
        every row's token and flags]

A group emits its drafts 1..a and row r0 + a's token.  Sampling uses the same counter-based uniform per position as
the non-speculative path, so accepting a draft exactly when the row's own draw equals it is lossless speculative
sampling for a deterministic proposal (accept d with probability p(d); on rejection the row's draw is distributed as p
restricted to the other tokens): a seeded request gives the same tokens with and without speculation, alone or in a
round with others.  Rejected rows' KV is overwritten before any later step reads it (a query at position q reads keys
<= q of its own sequence only), and their conv rows never reach the state.

Draft pricing (RoundPlanner; TensorFold's depth rule for shared rounds, with proportional fairness across streams):
per stream, acceptance by depth (decayed counts of drafts accepted / offered at each depth, backed off to the stream's
estimate for its match-length bucket): q(s, i) = P(drafts 1..i of stream s accepted).  A round of N streams costs
StepCost: the device step of the width holding its rows (dense part: the weights stream once per step), every live
row's K/V reads (linear in its position) and the host time on its critical path (decode-ahead hides it for plain
rounds, a verify round keeps it).  The planner takes the drafts in order of marginal gain and keeps the width and
prefix with the largest sum over streams of log(expected tokens per ms) = sum_s log(1 + Q_s) - N log(cost) (Q_s: the
sum of stream s's q's), against the plain round -N log(cost(plain)) with a 3% margin per stream: drafts whose
expected gain is below the marginal width and row cost are not verified, nor rounds whose gain sits in a few streams
while the others ride a slower step; when no stream has a draft worth its price the round is a plain decode step.  A
stream whose drafts keep failing gets one probe draft every PROBE x N of its proposals.
"""
from __future__ import annotations

import heapq
import math
import time
from pathlib import Path

import numpy as np
import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

KERNELS = Path(__file__).resolve().parent / "kernels"
DRAM = ttnn.DRAM_MEMORY_CONFIG
VERIFY_WIDTHS = (2, 4, 8, 16)
MAX_ROWS = 16
MAX_DRAFTS = 15
MODES = ("argmax", "exact", "exact_full")
RESULT_WORDS = 64  # kernels/spec_commit.cpp: a per group start row, ids, flags, groups, live rows
CONV_STATE_ROWS = 6
MAP_ACTIVE = 1 << 31


def map_word(offset: int, length: int, slot: int) -> int:
    """Row-map word of a verify row (kernels/conv_tail.hpp, attn_decode.cpp, spec_commit.cpp): its offset in its group,
    the group length and the group's conv-state slot."""
    return MAP_ACTIVE | (slot << 16) | (length << 8) | offset


def _core_set(cores):
    return ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores})


class EarlyRead:
    """Non-blocking copies of a step's small outputs to fixed host tensors, enqueued before work the host need not
    wait for (the drafter commit after a target step): read(handle) waits for the copies only (an event recorded
    right after them), not for the commands queued behind."""

    def __init__(self, device):
        self.device = device
        self._host = {}

    def start(self, tensors) -> tuple:
        hosts = []
        for t in tensors:
            h = self._host.get(id(t))
            if h is None:
                h = self._host[id(t)] = ttnn.allocate_tensor_on_host(t.shape, t.dtype, t.layout, self.device)
            ttnn.copy_device_to_host_tensor(t, h, blocking=False)
            hosts.append(h)
        return hosts, ttnn.record_event(self.device, 0)

    @staticmethod
    def read(handle) -> list:
        hosts, event = handle
        ttnn.event_synchronize(event)
        return [ttnn.to_torch(h) for h in hosts]


class SpecArgmax:
    """Row argmax (rows <= 16) of BF16 TILE logits [1, 1, rows, vocab] -> u32 [rows] (ties -> lowest id)."""

    def __init__(self, device):
        self.device = device
        self._progs = {}

    def _build(self, logits, out, rows, tiles):
        dev = self.device
        grid = dev.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        workers = 2 * len(cores)
        if tiles < workers:
            raise ValueError("SpecArgmax needs at least one tile per worker")
        core_set = ttnn.CoreRangeSet({ttnn.CoreRange(cores[0], cores[-1])})
        gather = dev.worker_core_from_logical_core(cores[0])
        named = [("rows", rows), ("tiles", tiles), ("workers", workers), ("gather_x", gather.x),
                 ("gather_y", gather.y)]
        common = [int(logits.buffer_address()), int(out.buffer_address())]
        per_core = [(c, [i]) for i, c in enumerate(cores)]
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "spec_argmax.cpp"), core_ranges=core_set,
            brisc_named_compile_time_args=named, ncrisc_named_compile_time_args=named,
            brisc_common_runtime_args=common, ncrisc_common_runtime_args=common,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(brisc_args=per_core, ncrisc_args=per_core),
        )
        rb = (rows * 32 + 63) // 64 * 64
        span = max(1, math.ceil(tiles / workers)) * 2 * rb
        slot = (rows * 8 + 15) // 16 * 16
        cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=core_set,
                                 format_descriptors=[ttnn.CBFormatDescriptor(cb, ttnn.uint32, size)])
               for cb, size in ((0, span), (1, span), (2, workers * slot))]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=core_set, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=sems)

    def __call__(self, logits, rows: int, out):
        if (logits.dtype != ttnn.bfloat16 or logits.layout != ttnn.TILE_LAYOUT or not 1 <= rows <= 16
                or int(logits.padded_shape[-2]) != 32 or logits.memory_config() != DRAM):
            raise ValueError("SpecArgmax needs DRAM BF16 TILE logits with <= 16 rows in one tile row")
        tiles = int(logits.padded_shape[-1]) // 32
        sig = (int(logits.buffer_address()), int(out.buffer_address()), rows, tiles)
        if self._progs.get(rows, (None,))[0] != sig:
            self._progs[rows] = (sig, self._build(logits, out, rows, tiles))
        ttnn.generic_op([logits, out], self._progs[rows][1])
        return out


class SpecCommit:
    """kernels/spec_commit.cpp: per group accepted count, conv-state commit of its consumed rows into its slot, the
    result page, the penalized groups' sampler histories."""

    BATCH = 4  # (layer, column tile) jobs per read round and worker

    def __init__(self, model, fast, workers: int = 66):
        self.device = model.device
        self.layers = [(fast.cand[layer.i], *layer.op.s) for layer in model.layers if not layer.attn]
        if any(len(entry) != 1 + CONV_STATE_ROWS for entry in self.layers):
            raise ValueError("SpecCommit: six conv-state rows per layer expected")
        self.tiles = model.cfg.hidden // 32
        grid = self.device.compute_with_storage_grid_size()
        n = min(workers // 2, grid.x * grid.y)
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]
        self._progs = {}

    def _build(self, tensors, hists, rows, stride, sampled):
        jobs = len(self.layers) * self.tiles
        workers = 2 * len(self.cores)
        common = [int(t.buffer_address()) for t in tensors] + [int(h.buffer_address()) for h in hists]
        for entry in self.layers:
            common += [int(t.buffer_address()) for t in entry]
        args = {}
        for w in range(workers):
            first, last = w * jobs // workers, (w + 1) * jobs // workers
            args[w] = [first, last - first, int(w == 0)]
        core_set = _core_set(self.cores)
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "spec_commit.cpp"), core_ranges=core_set,
            defines=[("SC_ROWS", str(rows)), ("SC_STRIDE", str(stride)), ("SC_SAMPLED", str(int(sampled))),
                     ("SC_TILES", str(self.tiles)), ("SC_SLOTS", str(len(hists))), ("SC_BATCH", str(self.BATCH))],
            brisc_common_runtime_args=common, ncrisc_common_runtime_args=common,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=[(c, args[2 * i]) for i, c in enumerate(self.cores)],
                ncrisc_args=[(c, args[2 * i + 1]) for i, c in enumerate(self.cores)]),
        )
        # tok 64 | pos 64 | map 64 | ids 16 x 64 | result 256 | hist 128 | params 128 | jobs x (cand + 6 state) x 1 KB
        scratch = 64 * 3 + 16 * 64 + 256 + 128 + 128 + self.BATCH * 1024 * (1 + CONV_STATE_ROWS)
        cbs = [ttnn.CBDescriptor(total_size=scratch, core_ranges=core_set,
                                 format_descriptors=[ttnn.CBFormatDescriptor(cb, ttnn.uint32, scratch)])
               for cb in (0, 1)]
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, tok, pos, ids, result, params, vmap, hists, rows: int, stride: int, sampled: bool):
        tensors = (tok, pos, ids, result, params, vmap)
        key = (rows, stride, sampled)
        sig = tuple(int(t.buffer_address()) for t in (*tensors, *hists))
        if self._progs.get(key, (None,))[0] != sig:
            self._progs[key] = (sig, self._build(tensors, hists, rows, stride, sampled))
        ttnn.generic_op([tok, pos, ids, result], self._progs[key][1])


class Verifier:
    """Fixed-address inputs and the captured verify traces (width x mode) of an Lfm3Model with the fast step.

    mode: "argmax" (greedy, no penalties), "exact" (the exact device sampler), "exact_full" (the sampler with the
    row-maximum program: unpenalized top_k-disabled rows).  page_blocks: the decode page-table width (the plain
    step's: SDPADecode's partitions depend on it, and the verify rows must match the plain step bitwise)."""

    def __init__(self, model, page_blocks: int, widths=VERIFY_WIDTHS, sampler=None):
        if model.fast is None or model.batch > MAX_ROWS:
            raise ValueError(f"speculative verification needs the fast decode step and at most {MAX_ROWS} slots")
        self.m, self.device, self.widths, self.sampler = model, model.device, tuple(widths), sampler
        self.page_blocks = page_blocks
        model.fast.enable_verify(self.widths)
        self.vmap = model.fast.vmap
        self.argmax = SpecArgmax(self.device)
        self.commit = SpecCommit(model, model.fast)
        u32 = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, memory_config=DRAM)
        self.inputs, self.result, self.ids, self._pt_host = {}, {}, {}, {}
        for w in self.widths:
            tok = ttnn.from_torch(torch.zeros(1, w, dtype=torch.int32), dtype=ttnn.uint32, **u32)
            pos = ttnn.from_torch(torch.full((w,), -1, dtype=torch.int32), dtype=ttnn.int32, **u32)
            ridx = ttnn.from_torch(torch.zeros(1, w, dtype=torch.int32), dtype=ttnn.uint32, **u32)
            pt = ttnn.from_torch(torch.zeros(w, page_blocks, dtype=torch.int32), dtype=ttnn.int32, **u32)
            self.inputs[w] = (tok, pos, ridx, pt)
            self.result[w] = ttnn.from_torch(torch.zeros(1, RESULT_WORDS, dtype=torch.int32), dtype=ttnn.uint32,
                                             **u32)
            self.ids[w] = ttnn.from_torch(torch.zeros(w, dtype=torch.int32), dtype=ttnn.uint32, **u32)
            self._pt_host[w] = None
        self._map_host = None
        self.traces, self.logits = {}, {}
        self.dummy = ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32, **u32)

    def modes(self):
        return MODES if self.sampler is not None else MODES[:1]

    def width(self, rows: int) -> int:
        return next(w for w in self.widths if w >= rows)

    # ---------------------------------------------------------------- device sequence
    def _forward(self, w, mode):
        tok, pos, ridx, pt = self.inputs[w]
        logits = self.m.fast.decode_forward(tok, pos, ridx, pt, verify=True)
        if mode == "argmax":
            self.argmax(logits, w, self.ids[w])
            self.commit(tok, pos, self.ids[w], self.result[w], self.dummy, self.vmap, [], w, 1, False)
        else:
            s = self.sampler
            s.sample_decode(logits, w, pos, full=mode == "exact_full", aux=tok)
            self.commit(tok, pos, s.out, self.result[w], s.params, self.vmap, s.hist, w, 4, True)
        return logits

    def capture(self, log=print):
        """Compile then capture every (width, mode) verify trace.  The compile runs write garbage into the trash
        block's KV and the conv state: reset the model afterwards."""
        trash = int(self.m.trash_block)
        for w in self.widths:
            self.stage(w, [0] * w, list(range(w)), torch.full((w, self.page_blocks), trash, dtype=torch.int32),
                       [map_word(j, w, 0) for j in range(w)])
            for mode in self.modes():
                if self.sampler is not None:
                    self.sampler.set_params([None] * w)
                t0 = time.time()
                old = self.logits.pop((w, mode), None)
                if old is not None:
                    ttnn.deallocate(old)
                ttnn.deallocate(self._forward(w, mode))
                ttnn.synchronize_device(self.device)
                trace = ttnn.begin_trace_capture(self.device, cq_id=0)
                self.logits[w, mode] = self._forward(w, mode)
                ttnn.end_trace_capture(self.device, trace, cq_id=0)
                self.traces[w, mode] = trace
                log(f"LFM3 speculation: captured the verify trace (width {w}, {mode}) in {time.time() - t0:.1f}s")
        if self.sampler is not None:
            self.sampler.invalidate()

    def stage(self, w, tokens, positions, page_table, words):
        """Copy one round's inputs into width w's trace inputs and the row map: tokens / positions / map words per row
        (int lists [w]; position -1 and word 0 = inactive row), page_table int32 [w, page_blocks] (copied only when it
        changed; the row map likewise).  The rope-index input is not read by the verify step (its attention fronts
        take the positions)."""
        tok, pos, _, pt = self.inputs[w]
        t = torch.tensor(tokens, dtype=torch.int32).reshape(1, w)
        p = torch.tensor(positions, dtype=torch.int32).reshape(w)
        rk = dict(layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=ttnn.uint32, **rk), tok)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(p, dtype=ttnn.int32, **rk), pos)
        if self._pt_host[w] is None or not torch.equal(self._pt_host[w], page_table):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(page_table, dtype=ttnn.int32, **rk), pt)
            self._pt_host[w] = page_table
        words = list(words) + [0] * (16 - len(words))
        if any(x and ((x >> 16) & 0xFF) >= w for x in words):
            # the verify kernels stage the state rows [0, w): a group's slot is its row index (< streams <= w)
            raise ValueError(f"LFM3 speculation: a group slot beyond the width {w}")
        if self._map_host != words:
            m = torch.tensor(np.asarray(words, dtype=np.uint32).view(np.int32)).reshape(1, 16)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(m, dtype=ttnn.uint32, **rk), self.vmap)
            self._map_host = words

    def read_result(self, w, early=None):
        """The result page of the last width-w round (torch int64 [RESULT_WORDS]); early: its EarlyRead handle (the
        copy enqueued before the drafter commit), None: a blocking read."""
        page = EarlyRead.read(early)[0] if early is not None else ttnn.to_torch(self.result[w])
        return page.reshape(-1).to(torch.int64) & 0xFFFFFFFF

    def row_logits(self, w, mode, row):
        return ttnn.to_torch(self.logits[w, mode]).reshape(-1, self.m.cfg.vocab)[row]


class PromptLookup:
    """N-gram index over one request's tokens (prompt + output), incremental: n-grams ending at positions < L are
    indexed (key -> latest end); a query looks up the context's last n tokens for n = n_max .. n_min."""

    _P = np.uint64(0x9E3779B97F4A7C15)  # 64-bit polynomial hash multiplier (matches are verified on the tokens)
    MATCH_CAP = 10  # the match length is only extended this far (match_bucket saturates above 9)

    def __init__(self, n_min: int = 2, n_max: int = 4):
        self.n_min, self.n_max = n_min, n_max
        self.owner = None
        self.tokens = np.zeros(0, dtype=np.int64)
        self.length = 0
        self.indexed = 0  # n-grams ending at positions [.., indexed) are in the index
        self.index = {}
        self._pending = None  # (L, [(n, key)]): the n-grams ending at L, keyed by prepare() for a later lookup

    def _key(self, seq):
        h = 0
        for t in seq:
            h = (h * 0x9E3779B97F4A7C15 + int(t) + 1) & 0xFFFFFFFFFFFFFFFF
        return h

    def _bulk(self, end):
        """Index every n-gram ending at positions [n, end) (numpy; the request's first call)."""
        t = self.tokens[:end].astype(np.uint64) + np.uint64(1)
        with np.errstate(over="ignore"):
            for n in range(self.n_min, self.n_max + 1):
                if end <= n:
                    continue
                h = np.zeros(end - n, dtype=np.uint64)
                for i in range(n):
                    h = h * self._P + t[i: end - n + i]
                ends = np.arange(n, end, dtype=np.int64)
                uniq, first = np.unique(h[::-1], return_index=True)  # the latest occurrence of every key
                d = self.index.setdefault(n, {})
                d.update(zip(uniq.tolist(), ends[::-1][first].tolist()))

    def _sync(self, owner, prompt, output):
        total = len(prompt) + len(output)
        if owner != self.owner or total < self.length:
            self.owner, self.length, self.indexed, self.index, self._pending = owner, 0, 0, {}, None
            self.tokens = np.zeros(max(1024, 2 * total), dtype=np.int64)
        if total > self.tokens.size:
            grown = np.zeros(2 * total, dtype=np.int64)
            grown[: self.length] = self.tokens[: self.length]
            self.tokens = grown
        if self.length < len(prompt):
            self.tokens[self.length: len(prompt)] = np.asarray(prompt[self.length:], dtype=np.int64)
            self.length = len(prompt)
        start = self.length - len(prompt)
        if start < len(output):
            self.tokens[self.length: total] = np.asarray(output[start:], dtype=np.int64)
            self.length = total

    def append(self, tokens):
        """Extend the context by emitted tokens (the round's read, before the plugin's lists catch up)."""
        n = len(tokens)
        if self.length + n > self.tokens.size:
            grown = np.zeros(2 * (self.length + n), dtype=np.int64)
            grown[: self.length] = self.tokens[: self.length]
            self.tokens = grown
        self.tokens[self.length: self.length + n] = np.asarray(tokens, dtype=np.int64)
        self.length += n

    def _index_to(self, end):
        if self.indexed == 0:
            self._bulk(end)
        else:
            for e in range(max(self.indexed, self.n_min), end):
                for n in range(self.n_min, min(self.n_max, e) + 1):
                    self.index.setdefault(n, {})[self._key(self.tokens[e - n: e])] = e
        self.indexed = max(self.indexed, end)

    def prepare(self):
        """Index the context up to L and key the n-grams ending at L (they use only tokens already known; they enter
        the index at the next lookup once the context has grown), so the query after a round's read costs only its own
        keys (decode-ahead: called while the device runs the round)."""
        L = self.length
        if L == 0 or (self._pending is not None and self._pending[0] == L):
            return
        self._index_to(L)
        self._pending = (L, [(n, self._key(self.tokens[L - n: L])) for n in range(self.n_min, min(self.n_max, L) + 1)])

    def lookup(self, k: int):
        """(drafts, match length) for the current context (index it first)."""
        L = self.length
        pending, self._pending = self._pending, None
        if pending is not None and pending[0] == self.indexed and pending[0] < L:
            for n, key in pending[1]:
                self.index.setdefault(n, {})[key] = pending[0]
            self.indexed = pending[0] + 1
        if self.indexed < L or self.indexed == 0:
            self._index_to(L)
        t = self.tokens
        for n in range(min(self.n_max, L - 1), self.n_min - 1, -1):
            e = self.index.get(n, {}).get(self._key(t[L - n: L]))
            if e is None or not np.array_equal(t[e - n: e], t[L - n: L]):
                continue
            m = n
            while m < self.MATCH_CAP and e - m - 1 >= 0 and t[e - m - 1] == t[L - m - 1]:
                m += 1
            return t[e: min(e + k, L)].tolist(), m
        return [], 0


def match_bucket(m: int) -> int:
    """Acceptance-statistics bucket of a match length."""
    return 0 if m <= 2 else 1 if m <= 3 else 2 if m <= 5 else 3 if m <= 9 else 4


class StreamStats:
    """One stream's draft acceptance: per match-length bucket a decayed geometric estimate alpha (accepted / offered
    positions up to the first rejection, a window of ~1 / (1 - decay) verify rounds, a prior worth `weight`
    positions), and per depth d the decayed conditional acceptance P(draft d accepted | drafts 1..d-1 accepted),
    backed off to the bucket's alpha with `depth_weight` pseudo-observations."""

    PRIOR = (0.35, 0.45, 0.55, 0.65, 0.75)

    def __init__(self, decay: float = 0.95, weight: float = 2.0, depth_weight: float = 4.0):
        self.decay, self.weight, self.depth_weight = decay, weight, depth_weight
        self.succ = [0.0] * 5
        self.trials = [0.0] * 5
        self.dsucc = [0.0] * (MAX_DRAFTS + 1)
        self.dtrials = [0.0] * (MAX_DRAFTS + 1)
        self.skipped = 0  # proposals with candidates the planner gave no draft (probe counter)

    def alpha(self, bucket: int) -> float:
        return (self.succ[bucket] + self.PRIOR[bucket] * self.weight) / (self.trials[bucket] + self.weight)

    def q(self, bucket: int, k: int) -> list[float]:
        """[P(drafts 1..i accepted) for i = 1..k]."""
        ab, w = self.alpha(bucket), self.depth_weight
        out, p = [], 1.0
        for d in range(1, k + 1):
            p *= (self.dsucc[d] + w * ab) / (self.dtrials[d] + w)
            out.append(p)
        return out

    def update(self, bucket: int, offered: int, accepted: int):
        if offered <= 0:
            return
        dec = self.decay
        self.succ[bucket] = self.succ[bucket] * dec + accepted
        self.trials[bucket] = self.trials[bucket] * dec + accepted + (1 if accepted < offered else 0)
        for d in range(1, MAX_DRAFTS + 1):
            self.dsucc[d] *= dec
            self.dtrials[d] *= dec
        for d in range(1, accepted + 1):
            self.dsucc[d] += 1.0
            self.dtrials[d] += 1.0
        if accepted < offered:
            self.dtrials[accepted + 1] += 1.0


# ---------------------------------------------------------------------------------------------------- round costs
# Device ms per round by decode width at position 0 (the dense part: weights streamed once per step, the sampler, the
# verify commit), per mode ("argmax": greedy rows only, "exact": the exact device sampler) plus KV_MS_PER_1K ms per
# live row and 1000 positions (its K/V reads: 6 full-attention layers + the sliding window): the least-squares fit of
# tests/test_speculate.py --timing (back-to-back trace replays, GPTQ checkpoint, max_model_len 131072, every row live,
# positions 128 .. 120000; logs/spec13/timing.json: 136 points, largest error 1.5%, rms 0.6%).  Host time on the
# critical path: plain rounds overlap vLLM's step with the device (decode-ahead: period = device step + ~0.2 ms, the
# read-to-launch gap incl. the next round's plan); a verify round pays vLLM's whole per-step work (schedule, input
# build, the round's staging, output processing) with the device idle: HOST_VERIFY_MS + HOST_VERIFY_ROW_MS per stream
# (16-slot server step profiles, period - device estimate: ~0.7 ms with 1 stream, ~1.45 ms with 6-8 streams;
# logs/spec13/prof/serve2.log, prof_chat8.csv).
PLAIN_MS = {"argmax": {1: 3.52, 2: 3.54, 4: 3.66, 8: 3.88, 16: 4.37},
            "exact": {1: 3.59, 2: 3.63, 4: 3.73, 8: 3.95, 16: 4.46}}
VERIFY_MS = {"argmax": {2: 3.65, 4: 3.78, 8: 4.14, 16: 4.80},
             "exact": {2: 3.73, 4: 3.89, 8: 4.27, 16: 4.99}}
KV_MS_PER_1K = 0.0144
HOST_PLAIN_MS = 0.18
HOST_VERIFY_MS = 0.55
HOST_VERIFY_ROW_MS = 0.11


class StepCost:
    """Cost in ms of a plain or verify round (see the tables above): base[mode][width] + sum over live rows of
    kv(position) + host time on the critical path."""

    def __init__(self, plain_widths, verify_widths=VERIFY_WIDTHS, plain=None, verify=None, kv_per_1k=KV_MS_PER_1K,
                 host_plain=HOST_PLAIN_MS, host_verify=HOST_VERIFY_MS, host_verify_row=HOST_VERIFY_ROW_MS):
        self.plain_widths = sorted(plain_widths)
        self.verify_widths = sorted(verify_widths)
        self.plain_ms, self.verify_ms = plain or PLAIN_MS, verify or VERIFY_MS
        self.kv_per_1k = kv_per_1k
        self.host_plain, self.host_verify, self.host_verify_row = host_plain, host_verify, host_verify_row

    @staticmethod
    def _base(table, w):
        if w in table:
            return table[w]
        ks = sorted(table)
        lo = max([k for k in ks if k <= w], default=ks[0])
        hi = min([k for k in ks if k >= w], default=ks[-1])
        if lo == hi:
            return table[lo]
        return table[lo] + (table[hi] - table[lo]) * (w - lo) / (hi - lo)

    def kv(self, position: int) -> float:
        return self.kv_per_1k * max(position, 0) / 1000.0

    def plain(self, mode: str, positions) -> float:
        n = len(positions)
        w = next((x for x in self.plain_widths if x >= n), self.plain_widths[-1])
        return self._base(self.plain_ms[mode], w) + sum(self.kv(p) for p in positions) + self.host_plain

    def verify(self, mode: str, width: int, positions, streams: int) -> float:
        """positions: every live row's (the drafts' rows included)."""
        return (self._base(self.verify_ms[mode], width) + sum(self.kv(p) for p in positions) + self.host_verify
                + self.host_verify_row * streams)


class RoundPlanner:
    """Draft allocation of one round (see the module docstring): items [(q, candidates, position, stats)] for the
    streams that have candidate drafts (q: [P(drafts 1..i accepted)] for i = 1..len(candidates); stats: the
    StreamStats whose probe counter the item drives, None: no probes), `positions` of every stream of the round (one
    row each) -> drafts per item (0: none).  extra_ms: added to every verify round's cost (a draft pass the round
    would need).  verify_only: the baseline is the narrowest verify round without drafts, no margin (the round is
    already a verify round).  After a call, `objective` / `plain_objective` hold the chosen round's and the baseline's
    value.

    A round is priced by proportional fairness over its N streams: sum over streams of log(expected tokens per ms) =
    sum_s log(1 + Q_s) - N log(cost), Q_s = the expected accepted drafts of stream s (the sum of its q's).  Drafts are
    taken greedily by marginal gain log(1 + Q_s + q) - log(1 + Q_s) (each stream's drafts in depth order); every
    verify width and draft prefix is priced and the best one must beat the plain round (Q = 0, its own cost) by MARGIN
    per stream.  With one stream this is the most expected tokens per ms; with several, a round whose gain sits in a
    few streams must also pay for slowing the streams it carries without drafts (pure throughput pricing slowed the
    slowest of 8 chat streams by 5%: logs/spec13/prof)."""

    PROBE = 32
    MARGIN = 0.03

    def __init__(self, cost: StepCost, max_rows: int = MAX_ROWS):
        self.cost, self.max_rows = cost, max_rows
        self.objective = self.plain_objective = 0.0

    def plan(self, mode: str, items, positions, extra_ms: float = 0.0, probes: bool = True, verify_only: bool = False):
        n = len(positions)
        counts = [0] * len(items)
        if verify_only:
            # the alternative is a verify round without drafts (a round already scheduled as a verify round)
            w0 = next(w for w in self.cost.verify_widths if w >= n)
            best_obj = -n * math.log(self.cost.verify(mode, w0, positions, n) + extra_ms)
        else:
            best_obj = -n * math.log(self.cost.plain(mode, positions) / (1 + self.MARGIN))
        self.objective = self.plain_objective = best_obj
        if not items or n >= self.max_rows:
            return counts
        best = None
        # the drafts in greedy order of marginal gain: [(item, depth, position, q)]
        qs = [q[: len(cand)] for q, cand, _, _ in items]
        Q, depth, order = [0.0] * len(items), [0] * len(items), []
        heap = [(-math.log1p(q[0]), i) for i, q in enumerate(qs) if q]
        heapq.heapify(heap)
        while heap and len(order) < self.max_rows - n:
            _, i = heapq.heappop(heap)
            q = qs[i][depth[i]]
            Q[i] += q
            depth[i] += 1
            order.append((i, depth[i], items[i][2] + depth[i], q))
            if depth[i] < len(qs[i]):
                heapq.heappush(heap, (-(math.log1p(Q[i] + qs[i][depth[i]]) - math.log1p(Q[i])), i))
        for w in self.cost.verify_widths:
            if w < n + 1 or w > self.max_rows:
                continue
            c = self.cost.verify(mode, w, positions, n) + extra_ms
            gain, Qw = 0.0, [0.0] * len(items)
            for m in range(min(w - n, len(order))):
                i, _, p, q = order[m]
                gain += math.log1p(Qw[i] + q) - math.log1p(Qw[i])
                Qw[i] += q
                c += self.cost.kv(p)
                obj = gain - n * math.log(c)
                if obj > best_obj + 1e-9:
                    best_obj, best = obj, (w, m + 1)
        self.objective = best_obj
        if best is not None:
            for i, d, _, _ in order[: best[1]]:
                counts[i] = max(counts[i], d)
            free = best[0] - n - best[1]
        else:
            free = 0
        # probes: a stream whose candidates keep getting no draft re-measures with one (riding free rows of a verify
        # round, or forcing one every PROBE x n of its proposals)
        for idx, (_, _, _, stats) in enumerate(items):
            if stats is None or not probes:
                continue
            if counts[idx]:
                stats.skipped = 0
                continue
            stats.skipped += 1
            if stats.skipped >= self.PROBE * max(1, n) or (free > 0 and stats.skipped >= self.PROBE):
                if best is None:
                    best = (min(w for w in self.cost.verify_widths if w >= n + 1), 0)
                    free = best[0] - n
                if free > 0:
                    counts[idx] = 1
                    free -= 1
                    stats.skipped = 0
        return counts


# ---------------------------------------------------------------------------------------------------- the drafter
# tt/drafter.py (LFM3_SPEC_DRAFTER=1 with a drafter export): per stream and round, prompt lookup when its match is at
# least LOOKUP_MIN_MATCH tokens, else (when the stream's drafter context is complete) the drafter, its depth chosen by
# the planner from per-draft acceptance estimates (Calibration of the drafter's top-1 / top-2 logit margin).  A draft
# pass serves up to Drafter.max_streams streams; it runs when the round it enables beats the best round without it
# by the planner's objective, priced with the pass and the streams' expected acceptance (DraftStats).
#
# The pass is planned at the adapter's read point (decode-ahead on; without it the drafter does not run): it is
# launched right after a round's read and read only when the next round is launched (Speculation.resolve), so its
# device time runs while vLLM finishes its step and plans the next (the device otherwise idle).  The plan hands vLLM
# placeholder drafts (the stream's limit, at most DRAFTS); resolve replaces them by the pass's drafts and re-plans the
# round with their calibrated acceptance (drafts scheduled but not verified count as rejected).  Priced (pass_ms):
# DRAFT_HOST_MS (stage + launch + read) + the pass's device time (DRAFT_MS: ms at an empty window + ms per 1K window
# positions, by stream count) less DRAFT_HIDDEN_MS (the host time between launch and read).
#
# The drafter's context (its K/V of every position) is committed after a target step while at most
# COMMIT_MAX_STREAMS streams run and one of them is not in a run of COPY_RUN prompt-lookup rounds that each accepted
# at least COPY_ACCEPTED drafts; a stream with an uncommitted position in its window waits WINDOW_ROWS positions.  The
# step's tokens / result page are copied to the host before its commit (EarlyRead): the commit runs while the host
# reads them and plans.
LOOKUP_MIN_MATCH = 3
DRAFT_MS = {1: (0.651, 0.060), 2: (0.759, 0.127)}  # tests/test_drafter.py --timing (logs/drafter/t4.json)
DRAFT_HOST_MS = 0.17  # stage + launch + read of a pass (test_drafter --timing host_*, serving summaries)
DRAFT_HIDDEN_MS = 0.28  # host time between a pass's launch and its read (serving summaries: device - read wait)
COMMIT_MAX_STREAMS = 4
COPY_RUN = 6
COPY_ACCEPTED = 4
WINDOW_ROWS = 2040  # context rows a draft pass reads (tt/drafter.py WINDOW - 8)
DRAFTS = 7  # drafts per drafter pass (tt/drafter.py BLOCK_ROWS - 1)


def pass_ms(positions) -> float:
    """Critical-path cost of a draft pass for streams at these anchor positions (see above)."""
    base, per_1k = DRAFT_MS.get(len(positions), DRAFT_MS[max(DRAFT_MS)])
    window = max(min(int(p), WINDOW_ROWS) for p in positions) / 1000
    return max(0.0, base + per_1k * window - DRAFT_HIDDEN_MS) + DRAFT_HOST_MS


class DraftStats:
    """One stream's drafter acceptance by depth: decayed P(draft d accepted | drafts 1..d-1 accepted), prior PRIOR
    with `weight` pseudo-observations (estimates before a pass, which decide whether to run it)."""

    PRIOR = (0.60, 0.68, 0.70, 0.70, 0.70, 0.70, 0.70)

    def __init__(self, decay: float = 0.9, weight: float = 3.0):
        self.decay, self.weight = decay, weight
        self.succ = [0.0] * len(self.PRIOR)
        self.trials = [0.0] * len(self.PRIOR)

    def q(self, k: int) -> list[float]:
        out, p = [], 1.0
        for d in range(min(k, len(self.PRIOR))):
            p *= (self.succ[d] + self.weight * self.PRIOR[d]) / (self.trials[d] + self.weight)
            out.append(p)
        return out

    def update(self, offered: int, accepted: int):
        for d in range(len(self.PRIOR)):
            self.succ[d] *= self.decay
            self.trials[d] *= self.decay
        for d in range(min(accepted, len(self.PRIOR))):
            self.succ[d] += 1.0
            self.trials[d] += 1.0
        if accepted < offered and accepted < len(self.PRIOR):
            self.trials[accepted] += 1.0


class Calibration:
    """P(draft d accepted | drafts 1..d-1 accepted) from the drafter's top-1 / top-2 logit margin, per sampler mode
    family: the drafter export's calibration.json {"edges": [margin bin edges], "argmax": [[p per bin] per depth],
    "exact": [...]} (drafter/calibrate.py on the held-out set; greedy texts for argmax, the liquid preset's expected
    acceptance for exact)."""

    def __init__(self, path):
        import json
        d = json.loads(Path(path).read_text())
        self.edges = [float(x) for x in d["edges"]]
        self.p = {m: [[float(x) for x in row] for row in d[m]] for m in ("argmax", "exact")}

    def q(self, margins, mode: str) -> list[float]:
        table = self.p["argmax" if mode == "argmax" else "exact"]
        out, p = [], 1.0
        for d, m in enumerate(margins):
            b = sum(1 for e in self.edges if m >= e)
            p *= table[min(d, len(table) - 1)][b]
            out.append(p)
        return out


class Stream:
    """One request's speculation state: proposer, acceptance statistics, its plugin limits, the last proposal and its
    drafter context state."""

    __slots__ = ("owner", "lookup", "stats", "remaining", "depth", "stop", "q", "bucket", "kind", "dstats",
                 "copy_run", "hole_end", "ctx_row", "pt_row", "pending")

    def __init__(self, owner, n_min, n_max):
        self.owner = owner
        self.lookup = PromptLookup(n_min, n_max)
        self.stats = StreamStats()
        self.remaining = 0  # tokens the request may still emit (max_tokens / max_model_len)
        self.depth = MAX_DRAFTS
        self.stop = frozenset()  # tokens that end the request (empty while min_tokens / ignore_eos hold)
        self.q = []  # q of the drafts in the current plan
        self.bucket = 0
        self.kind = None  # the current plan's proposer: "lookup" / "drafter" / None
        self.dstats = DraftStats()
        self.copy_run = 0  # consecutive lookup rounds with >= COPY_ACCEPTED accepted drafts
        self.hole_end = 0  # drafter context: every position >= hole_end committed (0: all)
        self.ctx_row = None  # the stream's row in the last committed step (Drafter.ctx32), None: not committed
        self.pt_row = None  # its page-table row (the last round's)
        self.pending = None  # anchor position of the deferred pass whose drafts this stream's placeholders await


class Speculation:
    """The serving side of shared-round speculation for an Lfm3Model (any number of slots): per-stream proposers and
    statistics, the round planner, the verifier, stats.

    Rows: self.rows[slot] = the Stream decoding in conv-state slot `slot` (= its decode row after the runner's remap),
    kept in step with the adapter's remaps / prefills.  Plans: `register` (vllm_tt_plugin's call after every step:
    every scheduled request with its slot, context and limits) returns each request's drafts for the next round:
    the plan `precompute` made at the read point of the step just run (the adapter's decode-ahead launches the next
    plain step there when the plan has no drafts), else (no precompute: decode-ahead off) a fresh plan.  `resolve`
    fills in a deferred draft pass's drafts, `round` verifies a shared round."""

    def __init__(self, model, page_blocks: int, sampler=None, widths=VERIFY_WIDTHS, plain_widths=None, log=print,
                 cost: StepCost | None = None, calibration=None):
        self.m, self.log = model, log
        self.verifier = Verifier(model, page_blocks, widths=widths, sampler=sampler)
        self.widths = self.verifier.widths
        self.max_rows = self.widths[-1]
        self.cost = cost or StepCost(plain_widths or [model.batch], self.widths)
        self.planner = RoundPlanner(self.cost, self.max_rows)
        self.rows = [None] * model.batch
        self.streams = {}
        self.plan = None  # {owner: (context length, drafts)} for the next round (precompute)
        self.mode = "argmax"  # the last round's sampler mode family ("argmax" / "exact"): prices the next round
        self.stats = self._zero_stats()
        self._recent = {}  # verify width -> recent trace + read ms
        self._words = {}  # (RowSampling, slot) -> its sampler params row (uint32 [32])
        # the drafter (tt/drafter.py) when the model has one, with its margin calibration
        self.drafter = getattr(model, "drafter", None)
        self.calibration = None
        self.pending = None  # the deferred draft pass awaiting resolve
        self.early = EarlyRead(model.device)  # non-blocking reads of step outputs (the adapter's too)
        if self.drafter is not None:
            self.calibration = Calibration(calibration or (self.drafter.path / "calibration.json"))

    def capture(self):
        self.verifier.capture(log=self.log)
        d = self.drafter
        if d is not None:
            for w in self.widths:
                _, pos, _, pt = self.verifier.inputs[w]
                d.capture_commit(("verify", w), w, pos, pt, verify=True)
            d.capture(log=self.log)
        self.stats = self._zero_stats()

    @staticmethod
    def _zero_stats():
        return {"verify": 0, "groups": 0, "drafted": 0, "rows": 0, "offered": 0, "accepted": 0, "plain": 0,
                "host_redraws": 0, "trimmed": 0, "ms": 0.0, "ms_prep": 0.0, "ms_stage": 0.0, "passes": 0,
                "pass_streams": 0, "pass_ms": 0.0, "pass_wait_ms": 0.0, "deferred": 0, "dropped": 0,
                "d_offered": 0, "d_accepted": 0, "d_groups": 0, "commits": 0, "steps": 0}

    # ---------------------------------------------------------------- drafter context (commit after a target step)
    def commit_wanted(self, slots) -> bool:
        """Commit the drafter context after a target step over these slots' streams?  (The drafter is on, at most
        COMMIT_MAX_STREAMS streams run, and one of them is unknown or may still draft: not in a prompt-lookup copy
        run, and its context hole, if any, leaves the window before the request's token limit.)"""
        if self.drafter is None:
            return False
        sts = [self.rows[s] if 0 <= s < len(self.rows) else None for s in slots]
        return len(sts) <= COMMIT_MAX_STREAMS and any(
            st is None or (st.copy_run < COPY_RUN
                           and (not st.hole_end or st.hole_end + WINDOW_ROWS < st.lookup.length + st.remaining))
            for st in sts)

    def note_step(self, rows, committed: bool):
        """A target step was launched: rows [(slot, first position, row count)] of its streams; committed: its
        drafter commit too.  Uncommitted positions end the stream's drafter context until they leave its window."""
        self.stats["steps"] += 1
        self.stats["commits"] += int(committed)
        if committed:
            return
        for slot, first, k in rows:
            st = self.rows[slot] if 0 <= slot < len(self.rows) else None
            if st is not None:
                st.hole_end = max(st.hole_end, first + k)
                st.ctx_row = None

    def _draft_ok(self, st, position: int) -> bool:
        """Can a draft pass serve this stream at anchor `position` (its context committed through position - 1)?"""
        d = self.drafter
        return (d is not None and st.ctx_row is not None and st.pt_row is not None
                and st.hole_end <= max(0, position - WINDOW_ROWS)
                and position + DRAFTS < min(self.m.max_seq_len, d.page_blocks * 64))

    def _draft_defer(self, picked):
        """Launch one draft pass for picked [(stream, anchor position)], its top-2 page copied to the host behind it;
        resolve reads it."""
        self.drop_pending()
        d = self.drafter
        t0 = time.perf_counter()
        n = len(picked)
        d.stage(n, [(p, int(st.lookup.tokens[st.lookup.length - 1]), st.ctx_row, st.pt_row) for st, p in picked])
        d.launch(n)
        handle = self.early.start([d.outs[n]["top2"]])
        for st, p in picked:
            st.pending = p
        self.pending = {"n": n, "picked": list(picked), "handle": handle}
        self.stats["passes"] += 1
        self.stats["deferred"] += 1
        self.stats["pass_streams"] += n
        self.stats["pass_ms"] += (time.perf_counter() - t0) * 1000

    def drop_pending(self):
        """Forget a deferred pass not resolved (its round became a plain step, or a new plan replaces it)."""
        pend, self.pending = self.pending, None
        if pend is not None:
            self.stats["dropped"] += 1
            for st, _ in pend["picked"]:
                st.pending = None

    def resolve(self, drafts, positions):
        """A shared round's scheduled drafts per row (rows = slots after the remap) -> the drafts to verify: the
        placeholders of the streams a deferred pass served become that pass's drafts and the round is re-planned with
        their calibrated acceptance (verify rounds only; at most the scheduled number per stream).  Placeholders whose
        pass does not apply are dropped."""
        pend, self.pending = self.pending, None
        if pend is None:
            return drafts
        t0 = time.perf_counter()
        ids, margins = self.drafter.read(pend["n"], EarlyRead.read(pend["handle"])[0])
        self.stats["pass_wait_ms"] += (time.perf_counter() - t0) * 1000
        served = {id(st): (p, ds, mg) for (st, p), ds, mg in zip(pend["picked"], ids, margins)}
        drafts = [list(d) for d in drafts]
        positions = [int(p) for p in positions]
        items, rows, hit = [], [], False
        for g, ds in enumerate(drafts):
            st = self.rows[g] if g < len(self.rows) else None
            if st is None or not ds:
                continue
            e = served.get(id(st))
            if e is not None and st.pending == positions[g] == e[0]:
                k = min(len(ds), len(e[1]))
                q = self.calibration.q(e[2][:k], self.mode)
                items.append((q, list(e[1][:k]), positions[g], None))
                hit = True
            elif e is not None or st.pending is not None:
                drafts[g] = []  # placeholders of a pass that does not apply
                continue
            else:
                k = len(ds)
                q = st.q[:k] if len(st.q) >= k else [0.5 ** (j + 1) for j in range(k)]
                items.append((q, ds, positions[g], None))
            rows.append(g)
        for st, _ in pend["picked"]:
            st.pending = None
        if not hit:
            return drafts
        counts = self.planner.plan(self.mode, items, positions, probes=False, verify_only=True)
        for g, (q, cand, _, _), c in zip(rows, items, counts):
            drafts[g] = list(cand[:c])
            self.rows[g].q = q[:c]
        return drafts

    # ---------------------------------------------------------------- rows
    def remap(self, remap):
        """Decode rows moved (the runner's condense): new row r holds old slot remap[r]."""
        old = self.rows
        idx = [int(i) for i in remap]
        self.rows = [old[i] if i < len(old) else None for i in idx] + old[len(idx):]

    def admit(self, slots):
        """Prefills into these slots: their streams are unknown until the next register."""
        for s in slots:
            if 0 <= s < len(self.rows):
                self.rows[s] = None
        self.plan = None

    def _stream(self, owner, n_min, n_max):
        st = self.streams.get(owner)
        if st is None or (st.lookup.n_min, st.lookup.n_max) != (n_min, n_max):
            st = self.streams[owner] = Stream(owner, n_min, n_max)
        return st

    def _max_drafts(self, st, position: int, rows: int) -> int:
        """Most drafts the stream may verify from `position` (its next input) in a round of `rows` other rows."""
        return max(0, min(st.depth, MAX_DRAFTS, st.remaining - 1, self.m.max_seq_len - 1 - position,
                          self.max_rows - rows))

    def register(self, entries, fresh: bool):
        """vllm_tt_plugin's plan call after a step: entries [(owner, slot, prompt ids, output ids, remaining, depth,
        stop ids, n_min, n_max)] for every scheduled request that may speculate (decode phase, not stopped) -> drafts
        per entry.  fresh: no precomputed plan applies and none may be made now (False: a prefill step, or a plain
        step already launched ahead) -> plan now when True."""
        plan, self.plan = self.plan, None
        streams, totals = [], []
        for owner, slot, prompt, output, remaining, depth, stop, n_min, n_max in entries:
            st = self._stream(owner, n_min, n_max)
            st.lookup._sync(owner, prompt, output)
            st.remaining, st.depth, st.stop = int(remaining), int(depth), frozenset(stop)
            if 0 <= slot < len(self.rows):
                self.rows[slot] = st
            streams.append(st)
            totals.append(len(prompt) + len(output))
        live = {id(st) for st in self.rows if st is not None}
        for owner in [o for o, st in self.streams.items() if id(st) not in live]:
            del self.streams[owner]
        if plan is not None:
            out = []
            for st, total in zip(streams, totals):
                got = plan.get(st.owner)
                drafts = got[1] if got is not None and got[0] == total else []
                out.append(drafts[: self._max_drafts(st, total - 1, 0)])
            return out
        if not fresh or not streams:
            return [[] for _ in streams]
        drafts = self._plan([(st, total - 1) for st, total in zip(streams, totals)], [])
        return [drafts.get(st.owner, []) for st in streams]

    def prepare(self, slots):
        """Before a round's read: index every live stream's n-grams up to its context, so the query after the read
        costs only its key lookups."""
        for s in slots:
            st = self.rows[s] if 0 <= s < len(self.rows) else None
            if st is not None and st.lookup.length:
                st.lookup.prepare()

    def precompute(self, rows, page_table=None, ctx_rows=None) -> bool:
        """Right after a round's read: rows [(slot, emitted tokens, input position of the round)] of every live row ->
        append the tokens to each stream's context and plan the next round (cached for the plugin's register call).
        page_table: the round's int32 [>= slots, blocks] (row = slot; the drafter's pages), ctx_rows: {slot: the
        stream's row in the drafter's ctx32 (None: not committed)} after a plain step (a verify round's are set by
        collect).  True when the plan verifies drafts (the next round is a verify round: no plain step launched
        ahead)."""
        live, others = [], []
        for slot, emitted, position in rows:
            st = self.rows[slot] if 0 <= slot < len(self.rows) else None
            nxt = position + len(emitted)
            if st is None or st.lookup.length != position + 1:
                others.append(nxt)  # unknown or unsynced stream: one row, no drafts
                continue
            st.lookup.append(emitted)
            st.remaining -= len(emitted)
            if ctx_rows is not None:
                st.ctx_row = ctx_rows.get(slot)
            if page_table is not None and slot < page_table.shape[0]:
                st.pt_row = torch.as_tensor(page_table[slot]).clone()
            if st.stop and any(int(t) in st.stop for t in emitted):
                continue  # the request ends with this round
            live.append((st, nxt))
        self.plan = None
        self.drop_pending()
        if not live:
            return False
        drafts = self._plan(live, others, drafting=True)
        self.plan = {st.owner: (st.lookup.length, drafts.get(st.owner, [])) for st, _ in live}
        return any(drafts.values())

    def _plan(self, live, others, drafting: bool = False) -> dict:
        """live [(stream, next input position)], others: positions of rows without a known stream -> {owner: drafts}.
        drafting: the drafter may serve streams whose lookup match is shorter than LOOKUP_MIN_MATCH (the read point
        of a round: their drafter context rows are known) with a deferred pass (placeholder drafts until resolve)."""
        positions = [p for _, p in live] + list(others)
        n = len(positions)
        items, owners, dcand = [], [], []
        for st, p in live:
            st.q, st.kind = [], None
            limit = self._max_drafts(st, p, n)
            if limit <= 0:
                st.copy_run = 0
                continue
            cand, m = st.lookup.lookup(limit)
            if m < LOOKUP_MIN_MATCH:
                st.copy_run = 0
                if drafting and self._draft_ok(st, p):
                    dcand.append((st, p, min(limit, DRAFTS)))
            if not cand:
                continue
            st.bucket = match_bucket(m)
            items.append((st.stats.q(st.bucket, len(cand)), cand, p, st.stats))
            owners.append(st)
        counts = self.planner.plan(self.mode, items, positions, probes=not dcand)
        kinds = ["lookup"] * len(items)
        if dcand:
            # the streams whose drafter rounds are expected to pay most: a pass for the best 1 .. max_streams of them,
            # the stream count whose round has the best objective (a wider pass costs more)
            dcand.sort(key=lambda c: -sum(c[0].dstats.q(c[2])))
            best = None
            best_obj = self.planner.objective
            for m in range(1, min(len(dcand), self.drafter.max_streams) + 1):
                picked = dcand[:m]
                ids = {id(st) for st, _, _ in picked}
                keep = [i for i, st in enumerate(owners) if id(st) not in ids]
                est = [(st.dstats.q(k), [0] * k, p, None) for st, p, k in picked]
                est_counts = self.planner.plan(self.mode, [items[i] for i in keep] + est, positions,
                                               extra_ms=pass_ms([p for _, p, _ in picked]), probes=False)
                if self.planner.objective > best_obj + 1e-9:
                    best_obj, best = self.planner.objective, (picked, keep, est, est_counts)
            if best is not None:
                # placeholders (the anchor token) up to each stream's limit: resolve picks the verified prefix
                picked, keep, est, est_counts = best
                self._draft_defer([(st, p) for st, p, _ in picked])
                owners = [owners[i] for i in keep] + [st for st, _, _ in picked]
                kinds = ["lookup"] * len(keep) + ["drafter"] * len(picked)
                items = [items[i] for i in keep] + [(q, [int(st.lookup.tokens[st.lookup.length - 1])] * k, p, None)
                                                    for (q, _, _, _), (st, p, k) in zip(est, picked)]
                counts = est_counts[: len(keep)] + [k for _, _, k in picked]
            else:
                counts = self.planner.plan(self.mode, items, positions)
        out = {}
        for st, (q, cand, _, _), kind, k in zip(owners, items, kinds, counts):
            if k:
                out[st.owner] = list(cand[:k])
                st.q, st.kind = q[:k], kind
        return out

    # ---------------------------------------------------------------- rounds
    def trim(self, drafts):
        """Fit a round into the widest verify step: drop the drafts with the lowest q (latest first) until
        sum(1 + k) <= max_rows.  drafts: per stream (lists) -> the trimmed lists."""
        drafts = [list(d) for d in drafts]
        over = sum(1 + len(d) for d in drafts) - self.max_rows
        if over <= 0:
            return drafts
        self.stats["trimmed"] += over
        while over > 0:
            # the last draft of the stream whose last draft has the smallest q (unknown q: by depth)
            best, key = None, None
            for i, d in enumerate(drafts):
                if not d:
                    continue
                st = self.rows[i] if i < len(self.rows) else None
                q = st.q[len(d) - 1] if st is not None and len(st.q) >= len(d) else 0.5 ** len(d)
                if key is None or q < key:
                    best, key = i, q
            drafts[best].pop()
            over -= 1
        return drafts

    def round(self, tokens, drafts, positions, page_table, requests=None, outputs=None) -> list[list[int]]:
        """Verify one shared round (rows = slots 0..N-1 after the adapter's remap): tokens / positions [N], drafts per
        stream (lists, possibly empty; fits max_rows), page_table int32 [>= N, blocks] (row = slot), requests: the
        exact sampler's SamplingRequest per stream (None: greedy device argmax for all), outputs: each stream's
        output ids before this round (default request.outputs) -> the emitted tokens per stream."""
        return self.collect(self.launch(tokens, drafts, positions, page_table, requests, outputs))

    def launch(self, tokens, drafts, positions, page_table, requests=None, outputs=None) -> dict:
        from .sampling import F_NOCOMMIT, PARAM_ROWS, W_EXTRA, W_EXTRA_BASE

        t0 = time.perf_counter()
        v = self.verifier
        n = len(tokens)
        drafts = [[int(d) for d in ds] for ds in drafts]
        rows = sum(1 + len(d) for d in drafts)
        if rows > self.max_rows or n > len(self.rows):
            raise ValueError(f"LFM3 speculation: a round of {rows} rows / {n} streams")
        w = v.width(rows)
        toks, poss, words, starts, src = [], [], [], [], []
        for g in range(n):
            ds = drafts[g]
            k = len(ds) + 1
            if positions[g] + k > self.m.max_seq_len:
                raise ValueError(f"LFM3 speculation: drafts beyond max_seq_len at position {positions[g]}")
            starts.append(len(toks))
            toks += [int(tokens[g])] + ds
            poss += [int(positions[g]) + j for j in range(k)]
            words += [map_word(j, k, g) for j in range(k)]
            src += [g] * k
        pad = w - rows
        toks += [0] * pad
        poss += [-1] * pad
        pt_src = torch.as_tensor(page_table)
        pt_src = pt_src.to(torch.int32) if pt_src.dtype != torch.int32 else pt_src
        pt = torch.zeros(w, v.page_blocks, dtype=torch.int32)
        cols = min(v.page_blocks, pt_src.shape[1])
        pt[:rows, :cols] = pt_src[torch.tensor(src, dtype=torch.long), :cols]
        if requests is None or all(r is None for r in requests):
            mode, requests = "argmax", None
        else:
            s = v.sampler
            if outputs is None:
                outputs = [r.outputs for r in requests]
            mode = "exact_full" if any(r.spec.full and not r.spec.penalized for r in requests) else "exact"
            # params rows: the stream's words (cached per spec and slot) with F_NOCOMMIT, row j counting its group's
            # drafts 1..j (aux tokens [r0 + 1, r0 + 1 + j))
            table = np.zeros((PARAM_ROWS, 32), dtype=np.uint32)
            if len(self._words) > 4096:
                self._words.clear()
            for g, r in enumerate(requests):
                spec = r.spec
                base = self._words.get((spec, g))
                if base is None:
                    base = self._words[spec, g] = np.asarray(spec.words(g), dtype=np.uint64).astype(np.uint32)
                r0, k = starts[g], len(drafts[g]) + 1
                table[r0: r0 + k] = base
                table[r0: r0 + k, 0] |= F_NOCOMMIT
                table[r0: r0 + k, W_EXTRA] = np.arange(k, dtype=np.uint32)
                table[r0: r0 + k, W_EXTRA_BASE] = r0 + 1
                if spec.penalized:
                    s.ensure_history(g, r.owner, outputs[g], r.prompt or (), spec.uses_prompt)
                else:
                    s.invalidate(g)
            s.set_param_table(table)
        t1 = time.perf_counter()
        v.stage(w, toks, poss, pt, words)
        ttnn.execute_trace(v.device, v.traces[w, mode], cq_id=0, blocking=False)
        commit = self.commit_wanted(range(n))
        early = None
        if commit:
            early = self.early.start([v.result[w]])  # read before the commit queued behind it
            self.drafter.launch_commit(("verify", w))
        self.note_step([(g, int(positions[g]), len(drafts[g]) + 1) for g in range(n)], commit)
        t2 = time.perf_counter()
        self.mode = "argmax" if mode == "argmax" else "exact"
        return {"tokens": [int(t) for t in tokens], "drafts": drafts, "positions": [int(p) for p in positions],
                "starts": starts, "requests": requests, "outputs": outputs, "w": w, "mode": mode, "rows": rows,
                "commit": commit, "early": early, "t0": t0, "ms_prep": (t1 - t0) * 1000, "ms_stage": (t2 - t1) * 1000}

    def collect(self, rec: dict) -> list[list[int]]:
        """Read a launched round -> the emitted tokens per stream (accepted drafts + one)."""
        from .sampling import R_HOST, host_sample

        v = self.verifier
        w, mode, requests = rec["w"], rec["mode"], rec["requests"]
        t2 = time.perf_counter()
        res = v.read_result(w, rec["early"])
        t3 = time.perf_counter()
        n = len(rec["drafts"])
        if int(res[48]) != n:
            raise RuntimeError(f"LFM3 speculation: bad verify result ({int(res[48])} groups for {n})")
        emitted = []
        for g, ds in enumerate(rec["drafts"]):
            r0, k = rec["starts"][g], len(ds)
            a = int(res[r0])
            ids, flags = res[16 + r0: 16 + r0 + k + 1].tolist(), res[32 + r0: 32 + r0 + k + 1].tolist()
            want = 0
            while want < k and not (flags[want] & 1) and ids[want] == ds[want]:
                want += 1
            if a != want:
                raise RuntimeError(f"LFM3 speculation: group {g} accepted {a}, rows say {want}")
            final = int(ids[a])
            if requests is not None:
                r = requests[g]
                spec, outputs = r.spec, rec["outputs"][g]
                if flags[a] & R_HOST:
                    row = v.row_logits(w, mode, r0 + a)
                    final = host_sample(row, spec, list(outputs) + ds[:a], r.prompt or (), rec["positions"][g] + a + 1)
                    v.sampler.invalidate(g)
                    self.stats["host_redraws"] += 1
                elif spec.penalized:
                    v.sampler.note_committed(g, r.owner, len(outputs) + a + 1)
            st = self.rows[g] if g < len(self.rows) else None
            if st is not None:
                if k and st.kind == "drafter":
                    st.dstats.update(k, a)
                    self.stats["d_groups"] += 1
                    self.stats["d_offered"] += k
                    self.stats["d_accepted"] += a
                elif k:
                    st.stats.update(st.bucket, k, a)
                st.copy_run = st.copy_run + 1 if k and st.kind == "lookup" and a >= COPY_ACCEPTED else 0
                st.ctx_row = r0 + a if rec["commit"] else None
            emitted.append(ds[:a] + [final])
        st = self.stats
        st["verify"] += 1
        st["groups"] += n
        st["drafted"] += sum(1 for d in rec["drafts"] if d)
        st["rows"] += rec["rows"]
        st["offered"] += sum(len(d) for d in rec["drafts"])
        st["accepted"] += sum(len(e) - 1 for e in emitted)
        st["ms"] += rec["ms_prep"] + rec["ms_stage"] + (time.perf_counter() - t2) * 1000
        st["ms_prep"] += rec["ms_prep"]
        st["ms_stage"] += rec["ms_stage"]
        self._recent.setdefault(w, []).append((t3 - rec["t0"]) * 1000 - rec["ms_prep"] - rec["ms_stage"])
        del self._recent[w][:-256]
        if st["verify"] % 256 == 0:
            self.log(self.summary())
        return emitted

    def note_plain_step(self):
        self.stats["plain"] += 1
        self.drop_pending()  # a planned verify round became a plain step

    def summary(self) -> str:
        st = self.stats
        n = max(st["verify"], 1)
        drafted = max(st["drafted"], 1)
        return (f"LFM3 speculation: {st['verify']} verify rounds ({st['groups'] / n:.2f} streams, {st['rows'] / n:.2f} "
                f"rows each; launch to emitted tokens {st['ms'] / n:.2f} ms, of it host sampler setup "
                f"{st['ms_prep'] / n:.2f} + input staging {st['ms_stage'] / n:.2f}), drafts accepted "
                f"{st['accepted']}/{st['offered']} ({st['accepted'] / max(st['offered'], 1):.3f}), acceptance length "
                f"{1 + st['accepted'] / drafted:.2f} per drafted stream per verify round; {st['plain']} plain rounds; "
                f"trimmed {st['trimmed']}; host redraws {st['host_redraws']}; trace + read median ms by width "
                + ", ".join(f"{r}: {np.median(t):.2f}" for r, t in sorted(self._recent.items()))
                + (f"; drafter: {st['passes']} passes ({st['pass_streams'] / max(st['passes'], 1):.2f} streams, "
                   f"stage + launch {st['pass_ms'] / max(st['passes'], 1):.2f} ms, read wait "
                   f"{st['pass_wait_ms'] / max(st['passes'] - st['dropped'], 1):.2f} ms each; {st['dropped']} "
                   f"unused), drafts accepted {st['d_accepted']}/{st['d_offered']} in {st['d_groups']} groups, "
                   f"commits {st['commits']}/{st['steps']} steps"
                   if self.drafter is not None else ""))
