# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Prompt-lookup speculative decoding for LFM2.5 on one P150 (one sequence: max_num_seqs 1).

Drafts come from the request's own tokens (prompt + output): the longest earlier occurrence of the context's last
n tokens (n_max .. n_min) proposes the tokens that followed it.  One traced verify step runs the K drafts and the last
committed token as K + 1 rows of the fast decode step at consecutive positions (decode width W = 2, 4 or 8; unused
rows inactive):

    rows j = 0..K: input tok[j] (tok[0] = the last committed token, tok[j] = draft j) at position p + j
    FastStep.decode_forward(verify=True): KV written for every row (each cache tile patched by one worker), conv rows
        chained inside the step (row j's left context = rows j-1, j-2), every row's Bx -> per-layer candidates,
        the conv state untouched; each row's logits are bitwise those of the sequential 1-row steps
    row tokens: SpecArgmax (greedy; ties -> lowest id, as the 1-row device argmax) or the exact device sampler
        (tt/sampling.py) with row j's penalty history counting drafts 1..j and the draw at position p + j + 1
    SpecCommit: a = the first row whose token differs from the next draft (or is flagged for a host redraw, or K);
        conv state <- candidate rows a - 1, a for all 30 layers; sampler history += the emitted tokens (penalized
        rows); result page [a, final token, flags, live rows, row tokens]

The step emits drafts 1..a and row a's token.  Sampling uses the same counter-based uniform per position as the
non-speculative path, so accepting a draft exactly when the row's own draw equals it is lossless speculative
sampling for a deterministic proposal (accept d with probability p(d); on rejection the row's draw is distributed as
p restricted to the other tokens): a seeded request gives the same tokens with and without speculation.  Rejected
rows' KV is overwritten before any later step reads it.

Adaptive K (SpecPolicy): per match-length bucket a decayed acceptance estimate alpha; K = 0 (the plain 1-row step)
unless some K's expected emitted tokens per millisecond, sum_{i<=K} alpha^i / cost(K + 1 rows), beats the 1-row step.
A step's cost grows with its live rows (each row routes to its own experts), not with the decode width.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

KERNELS = Path(__file__).resolve().parent / "kernels"
DRAM = ttnn.DRAM_MEMORY_CONFIG
VERIFY_WIDTHS = (2, 4, 8)
MODES = ("argmax", "exact", "exact_full")
RESULT_WORDS = 32


def _core_set(cores):
    return ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores})


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
    """kernels/spec_commit.cpp: accepted count, conv-state commit of the accepted row, result page, sampler history."""

    def __init__(self, model, fast, workers: int = 66):
        self.device = model.device
        self.layers = [(fast.cand[layer.i], layer.op.s0, layer.op.s1) for layer in model.layers if not layer.attn]
        self.tiles = model.cfg.hidden // 32
        grid = self.device.compute_with_storage_grid_size()
        n = min(workers // 2, grid.x * grid.y)
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]
        self._progs = {}

    def _build(self, tensors, rows, stride, sampled):
        tok, pos, ids, result, params, hist = tensors
        jobs = len(self.layers) * self.tiles
        workers = 2 * len(self.cores)
        common = [int(t.buffer_address()) for t in (tok, pos, ids, result, params, hist)]
        for cand, s0, s1 in self.layers:
            common += [int(cand.buffer_address()), int(s0.buffer_address()), int(s1.buffer_address())]
        args = {}
        for w in range(workers):
            first, last = w * jobs // workers, (w + 1) * jobs // workers
            args[w] = [first, last - first, int(w == 0)]
        core_set = _core_set(self.cores)
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "spec_commit.cpp"), core_ranges=core_set,
            defines=[("SC_ROWS", str(rows)), ("SC_STRIDE", str(stride)), ("SC_SAMPLED", str(int(sampled))),
                     ("SC_TILES", str(self.tiles))],
            brisc_common_runtime_args=common, ncrisc_common_runtime_args=common,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=[(c, args[2 * i]) for i, c in enumerate(self.cores)],
                ncrisc_args=[(c, args[2 * i + 1]) for i, c in enumerate(self.cores)]),
        )
        scratch = 128 + 16 * 64 + 128 + 128 + 8 * (1024 + 256)
        cbs = [ttnn.CBDescriptor(total_size=scratch, core_ranges=core_set,
                                 format_descriptors=[ttnn.CBFormatDescriptor(cb, ttnn.uint32, scratch)])
               for cb in (0, 1)]
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, tok, pos, ids, result, params, hist, rows: int, stride: int, sampled: bool):
        tensors = (tok, pos, ids, result, params, hist)
        key = (rows, stride, sampled)
        sig = tuple(int(t.buffer_address()) for t in tensors)
        if self._progs.get(key, (None,))[0] != sig:
            self._progs[key] = (sig, self._build(tensors, rows, stride, sampled))
        ttnn.generic_op(list(tensors[:4]), self._progs[key][1])


class Verifier:
    """Fixed-address inputs and the captured verify traces (width x mode) of one 1-slot Lfm2Model with the fast step.

    mode: "argmax" (greedy, no penalties), "exact" (the exact device sampler), "exact_full" (the sampler with the
    row-maximum program: unpenalized top_k-disabled rows)."""

    def __init__(self, model, page_blocks: int, widths=VERIFY_WIDTHS, sampler=None):
        if model.fast is None or model.batch != 1:
            raise ValueError("speculative verification needs the fast decode step and one conv-state slot")
        self.m, self.device, self.widths, self.sampler = model, model.device, tuple(widths), sampler
        self.page_blocks = page_blocks
        model.fast.enable_verify(self.widths)
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
            self.result[w] = ttnn.from_torch(torch.zeros(1, RESULT_WORDS, dtype=torch.int32), dtype=ttnn.uint32, **u32)
            self.ids[w] = ttnn.from_torch(torch.zeros(w, dtype=torch.int32), dtype=ttnn.uint32, **u32)
            self._pt_host[w] = None
        self.traces, self.logits = {}, {}
        self.dummy = ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32, **u32)

    def modes(self):
        return MODES if self.sampler is not None else MODES[:1]

    # ---------------------------------------------------------------- device sequence
    def _forward(self, w, mode):
        tok, pos, ridx, pt = self.inputs[w]
        logits = self.m.fast.decode_forward(tok, pos, ridx, pt, verify=True)
        if mode == "argmax":
            self.argmax(logits, w, self.ids[w])
            self.commit(tok, pos, self.ids[w], self.result[w], self.dummy, self.dummy, w, 1, False)
        else:
            s = self.sampler
            s.sample_decode(logits, w, pos, full=mode == "exact_full", aux=tok)
            self.commit(tok, pos, s.out, self.result[w], s.params, s.hist[0], w, 4, True)
        return logits

    def capture(self, log=print):
        """Compile then capture every (width, mode) verify trace.  The compile runs write garbage into the trash
        block's KV and the conv state: reset the model afterwards."""
        trash = int(self.m.trash_block)
        for w in self.widths:
            self.stage(w, [0] * w, list(range(w)), [trash] * self.page_blocks)
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
                log(f"LFM2 speculation: captured the verify trace (width {w}, {mode}) in {time.time() - t0:.1f}s")
        if self.sampler is not None:
            self.sampler.invalidate()

    def stage(self, w, tokens, positions, page_row):
        """Copy one step's inputs into width w's trace inputs (the page table only when it changed)."""
        tok, pos, ridx, pt = self.inputs[w]
        t = torch.tensor(tokens, dtype=torch.int32).reshape(1, w)
        p = torch.tensor(positions, dtype=torch.int32).reshape(w)
        rk = dict(layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=ttnn.uint32, **rk), tok)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(p, dtype=ttnn.int32, **rk), pos)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(p.clamp(min=0).reshape(1, w), dtype=ttnn.uint32, **rk), ridx)
        row = torch.as_tensor(page_row, dtype=torch.int32).reshape(-1)
        if self._pt_host[w] is None or not torch.equal(self._pt_host[w], row):
            full = torch.zeros(self.page_blocks, dtype=torch.int32)
            full[: min(row.numel(), self.page_blocks)] = row[: self.page_blocks]
            host = ttnn.from_torch(full.reshape(1, -1).expand(w, -1).contiguous(), dtype=ttnn.int32, **rk)
            ttnn.copy_host_to_device_tensor(host, pt)
            self._pt_host[w] = row.clone()

    def run(self, w, mode, tokens, positions, page_row):
        """One verify step -> the result page (torch int64 [RESULT_WORDS]).  self.stage_ms: its input staging."""
        t0 = time.perf_counter()
        self.stage(w, tokens, positions, page_row)
        self.stage_ms = (time.perf_counter() - t0) * 1000
        ttnn.execute_trace(self.device, self.traces[w, mode], cq_id=0, blocking=False)
        return ttnn.to_torch(self.result[w]).reshape(-1).to(torch.int64) & 0xFFFFFFFF

    def row_logits(self, w, mode, row):
        return ttnn.to_torch(self.logits[w, mode]).reshape(-1, self.m.cfg.vocab)[row]


class PromptLookup:
    """N-gram index over one request's tokens (prompt + output), incremental: n-grams ending at positions < L are
    indexed (key -> latest end); a query looks up the context's last n tokens for n = n_max .. n_min."""

    _P = np.uint64(0x9E3779B97F4A7C15)  # 64-bit polynomial hash multiplier (matches are verified on the tokens)

    def __init__(self, n_min: int = 2, n_max: int = 4):
        self.n_min, self.n_max = n_min, n_max
        self.owner = None
        self.tokens = np.zeros(0, dtype=np.int64)
        self.length = 0
        self.indexed = 0  # n-grams ending at positions [.., indexed) are in the index
        self.index = {}

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
            self.owner, self.length, self.indexed, self.index = owner, 0, 0, {}
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

    def propose(self, owner, prompt, output, k: int):
        """(drafts, match length) for the context prompt + output; ([], 0) without a match."""
        self._sync(owner, prompt, output)
        L = self.length
        end = L  # index n-grams ending before L
        if self.indexed == 0:
            self._bulk(end)
        else:
            for e in range(max(self.indexed, self.n_min), end):
                for n in range(self.n_min, min(self.n_max, e) + 1):
                    self.index.setdefault(n, {})[self._key(self.tokens[e - n: e])] = e
        self.indexed = end
        t = self.tokens
        for n in range(min(self.n_max, L - 1), self.n_min - 1, -1):
            e = self.index.get(n, {}).get(self._key(t[L - n: L]))
            if e is None or not np.array_equal(t[e - n: e], t[L - n: L]):
                continue
            m = n  # the match's full length (up to 32)
            while m < 32 and e - m - 1 >= 0 and t[e - m - 1] == t[L - m - 1]:
                m += 1
            drafts = t[e: min(e + k, L)].tolist()
            return drafts, m
        return [], 0


def match_bucket(m: int) -> int:
    """Acceptance-statistics bucket of a match length."""
    return 0 if m <= 2 else 1 if m <= 3 else 2 if m <= 5 else 3 if m <= 9 else 4


class SpecPolicy:
    """Adaptive draft count.  cost(rows, position): ms per step with `rows` live rows (1 = the plain decode step)
    incl. host overhead; alpha per match bucket: decayed counts of accepted / offered draft positions (a window of
    ~1 / (1 - decay) verify steps) with a prior worth `weight` positions.  K maximizes the expected emitted tokens per
    ms, sum_{i<=K} alpha^i / cost(K + 1); K = 0 when the plain step is best (the 1-draft break-even is alpha ~0.22),
    so a bucket whose drafts keep failing stops speculating until a probe (one draft every `probe` skipped
    proposals) succeeds."""

    PRIOR = (0.35, 0.45, 0.55, 0.65, 0.75)

    def __init__(self, max_drafts: int = VERIFY_WIDTHS[-1] - 1, cost=None, decay: float = 0.95, weight: float = 2.0,
                 probe: int = 32):
        self.max_drafts = max_drafts
        self.cost = cost or default_cost
        self.decay, self.weight, self.probe = decay, weight, probe
        self.succ = [0.0] * 5
        self.trials = [0.0] * 5
        self.skipped = 0

    def alpha(self, bucket):
        return (self.succ[bucket] + self.PRIOR[bucket] * self.weight) / (self.trials[bucket] + self.weight)

    def choose(self, bucket: int, available: int, position: int) -> int:
        """Drafts to verify (0: the plain step)."""
        if available <= 0:
            return 0
        al = self.alpha(bucket)
        best, best_k = 1.0 / self.cost(1, position), 0
        expected, term = 1.0, 1.0
        for k in range(1, min(available, self.max_drafts) + 1):
            term *= al
            expected += term
            rate = expected / self.cost(k + 1, position)
            if rate > best * 1.0001:
                best, best_k = rate, k
        if best_k == 0:
            self.skipped += 1
            if self.skipped >= self.probe:  # re-measure a bucket whose estimate fell below break-even
                self.skipped = 0
                return 1
        return best_k

    def update(self, bucket: int, offered: int, accepted: int):
        self.succ[bucket] = self.succ[bucket] * self.decay + accepted
        self.trials[bucket] = self.trials[bucket] * self.decay + accepted + (1 if accepted < offered else 0)


def default_cost(rows: int, position: int) -> float:
    """Measured step cost (ms, device + host) with `rows` live rows, interpolated in the position."""
    c = COST_MS[rows]
    if position <= COST_POSITIONS[0]:
        return c[0]
    for i in range(1, len(COST_POSITIONS)):
        if position <= COST_POSITIONS[i]:
            p0, p1 = COST_POSITIONS[i - 1], COST_POSITIONS[i]
            return c[i - 1] + (c[i] - c[i - 1]) * (position - p0) / (p1 - p0)
    return c[-1]


# Step wall at the model level (tests/test_speculate.py --timing, logs/spec/t5.json: liquid preset, real prompt
# tokens as inputs and drafts; the MoE cost grows with the rows' distinct experts) + 0.4 ms of server overhead per
# step.  rows -> ms at COST_POSITIONS.
COST_POSITIONS = (128, 2048, 8192, 16384, 131008)
COST_MS = {1: (5.98, 5.99, 6.14, 6.36, 9.37), 2: (7.09, 7.28, 7.58, 8.01, 14.13), 3: (8.15, 8.38, 8.93, 9.64, 18.64),
           4: (9.11, 9.38, 10.07, 10.84, 22.69), 5: (10.02, 10.54, 11.31, 12.79, 30.4),
           6: (11.01, 11.31, 12.45, 13.71, 30.87), 7: (11.53, 12.09, 13.28, 14.74, 34.88),
           8: (12.13, 12.72, 14.13, 15.73, 38.37)}


class Speculation:
    """The serving side of prompt-lookup speculation for a 1-slot Lfm2Model: proposer + policy + verifier + stats.
    `drafts` proposes the next step's drafts (possibly none: the plain decode step runs); `step` verifies them."""

    def __init__(self, model, page_blocks: int, sampler=None, widths=VERIFY_WIDTHS, log=print):
        self.m, self.log = model, log
        self.verifier = Verifier(model, page_blocks, widths=widths, sampler=sampler)
        self.widths = self.verifier.widths
        self.proposer = PromptLookup()
        self.policy = SpecPolicy(max_drafts=self.widths[-1] - 1)
        self._pending = None  # (drafts, bucket) of the last proposal
        self.stats = self._zero_stats()
        self._recent = {}  # live rows -> recent trace + read ms

    def capture(self):
        self.verifier.capture(log=self.log)
        self.stats = self._zero_stats()

    @staticmethod
    def _zero_stats():
        return {"verify": 0, "offered": 0, "accepted": 0, "plain": 0, "host_redraws": 0, "ms": 0.0, "ms_prep": 0.0,
                "ms_stage": 0.0, "ms_run": 0.0}

    def drafts(self, owner, prompt, output, max_drafts: int, n_min: int, n_max: int) -> list[int]:
        p = self.proposer
        if (p.n_min, p.n_max) != (n_min, n_max):
            self.proposer = p = PromptLookup(n_min, n_max)
        position = len(prompt) + len(output) - 1  # the next step's input position
        limit = min(max_drafts, self.widths[-1] - 1, self.m.max_seq_len - 1 - position)
        cand, m = p.propose(owner, prompt, output, max(limit, 0))
        k = self.policy.choose(match_bucket(m), len(cand), position) if cand and limit > 0 else 0
        self._pending = (tuple(cand[:k]), match_bucket(m)) if k else None
        return cand[:k]

    def note_plain_step(self):
        self.stats["plain"] += 1

    def step(self, token: int, drafts, position: int, page_row, request=None) -> list[int]:
        """Verify `drafts` after the committed `token` at `position` -> the emitted tokens (accepted drafts + one).
        request: the exact sampler's SamplingRequest (None: greedy device argmax)."""
        from .sampling import F_NOCOMMIT, R_HOST, W_EXTRA, host_sample

        t0 = time.perf_counter()
        drafts = [int(d) for d in drafts]
        k = len(drafts)
        w = next(x for x in self.widths if x >= k + 1)
        tokens = [int(token)] + drafts + [0] * (w - k - 1)
        positions = [position + j for j in range(k + 1)] + [-1] * (w - k - 1)
        v = self.verifier
        if request is None:
            mode = "argmax"
        else:
            spec, s = request.spec, v.sampler
            mode = "exact_full" if spec.full and not spec.penalized else "exact"
            rows = []
            for j in range(w):
                if j > k:
                    rows.append(None)
                    continue
                words = list(spec.words(0))
                words[0] |= F_NOCOMMIT
                words[W_EXTRA] = j
                rows.append(words)
            s.set_params(rows)
            if spec.penalized:
                s.ensure_history(0, request.owner, request.outputs, request.prompt or (), spec.uses_prompt)
            else:
                s.invalidate(0)
        t1 = time.perf_counter()
        res = v.run(w, mode, tokens, positions, page_row)
        t2 = time.perf_counter()
        a, final, flags = int(res[0]), int(res[1]), int(res[2])
        if not 0 <= a <= k or int(res[3]) != k + 1:
            raise RuntimeError(f"LFM2 speculation: bad verify result {res[:4].tolist()} for {k} drafts")
        if request is not None:
            if flags & R_HOST:
                row = v.row_logits(w, mode, a)
                final = host_sample(row, spec, list(request.outputs) + drafts[:a], request.prompt or (),
                                    position + a + 1)
                v.sampler.invalidate(0)
                self.stats["host_redraws"] += 1
            elif spec.penalized:
                v.sampler.note_committed(0, request.owner, len(request.outputs) + a + 1)
        bucket = self._pending[1] if self._pending is not None and list(self._pending[0]) == drafts else 1
        self._pending = None
        self.policy.update(bucket, k, a)
        st = self.stats
        st["verify"] += 1
        st["offered"] += k
        st["accepted"] += a
        st["ms"] += (time.perf_counter() - t0) * 1000
        st["ms_prep"] += (t1 - t0) * 1000
        st["ms_stage"] += v.stage_ms
        st["ms_run"] += (t2 - t1) * 1000
        self._recent.setdefault(k + 1, []).append((t2 - t1) * 1000 - v.stage_ms)
        del self._recent[k + 1][:-256]
        if st["verify"] % 256 == 0:
            self.log(self.summary())
        return drafts[:a] + [final]

    def summary(self) -> str:
        st = self.stats
        n = max(st["verify"], 1)
        steps = st["verify"] + st["plain"]
        tokens = st["verify"] + st["accepted"] + st["plain"]
        return (f"LFM2 speculation: {st['verify']} verify steps ({st['ms'] / n:.2f} ms each: sampler setup "
                f"{st['ms_prep'] / n:.2f}, input staging {st['ms_stage'] / n:.2f}, trace + read "
                f"{(st['ms_run'] - st['ms_stage']) / n:.2f}), drafts accepted "
                f"{st['accepted']}/{st['offered']} ({st['accepted'] / max(st['offered'], 1):.3f}), "
                f"{1 + st['accepted'] / n:.2f} tokens per verify step; {st['plain']} plain steps; "
                f"{tokens / max(steps, 1):.2f} tokens per step overall; host redraws {st['host_redraws']}; "
                f"trace + read median ms by live rows "
                + ", ".join(f"{r}: {np.median(t):.2f}" for r, t in sorted(self._recent.items())))
