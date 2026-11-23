# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Batched traced decode (tt/NOTES-batch.md section 4): one device step advances every running sequence.

    dec = BatchDecoder(model)          # model = Inkling2Model(max_num_seqs=S, kv_blocks=NB); before any capture
    dec.capture()                      # buckets (1, 2, 4, 8, 16 capped at S, + S) x sampling kinds; ends with reset()
    toks = dec.step(rows)              # rows: list[BatchRow] (row b = vLLM decode row b) -> next token of every row
    lg = dec.step_logits(rows)         # host-sampling fallback for the whole step -> FP32 [B, unpadded_vocab]
    dec.invalidate(slot=None)          # forget the host mirror (of one slot / all) after a prefill / reset

Per step (B rows, bucket R = the smallest bucket >= B; rows B..R-1 = pads, replicas of row 0):
  main trace[R]:      x = embedding(tok words 0..R-1) (device BF16 table; embed="host": the host-uploaded rows) ->
                      model.batch_decode(x, bt, R, ids=tok) -> FP32 logits [1, 1, R, V_pad]; the LM head writes row
                      b's argmax to tok word b;
  tail trace[R, kind]: kind = the widest of the rows' kinds (fast/sampling.kind_of; greedy < sample < full <
                      full_reject): greedy: BatchAdvance(tokens from tok); else DeviceSampler on R rows (params page b
                      = row b's RowSampling.words(slot=slot[b]); greedy rows as greedy params, pads / unpenalized rows
                      without history commit; drawn position cur_pos[b] + 1; [token, flags, ..] -> sel page b), then
                      BatchAdvance(tokens from sel -> tok): cur_pos[b] + 1, tau (0, b), token log;
  rm trace[R]:        (step_logits / host redraws) the R logit rows untilized on the device.
The host keeps a mirror of what the device holds for every row (slot, token, position via the batch tables, page
row, sampling params / histories) and writes only what differs from the rows of the next step: a steady batch costs
two replays and one read of R ids (+ flags) per step.  An R_INEXACT row (truncated top-k tie group / 4 rejected
full-vocabulary draws) is redrawn on the host from its logit row (host_sample, exact) and its token rewritten.

Trace safety (model.py): every persistent tensor is allocated in the constructor; capture() compiles every bucket and
kind before capturing any trace; between steps only host writes into those tensors and eager prefill (call-local
tensors) run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import ttnn

from ..common import DRAM
from .batch_tables import LOG_WORDS, MAX_ROWS, BatchAdvance
from .decode_loop import LOG_PAGES
from .sampling import P_HOST, P_SAMPLE, R_INEXACT, DeviceSampler, RowSampling, host_sample, kind_of

BUCKETS = (1, 2, 4, 8, 16)
FEATURE_APPS = (4, 22, 40)  # drafter taps: outputs of these applications (= inputs of 5, 23, 41); tt/spec.py
KINDS = ("greedy", "sample", "full", "full_reject")  # ascending width: a wider tail serves every narrower row
_GREEDY = RowSampling(temperature=0.0, top_k=0)


@dataclass
class BatchRow:
    """One decode row: the sequence in physical slot `slot` decodes `token` at position `pos`; pages = its vLLM
    block-table row (pool block of positions 64 j .. at column j); sampling None = greedy (device argmax).  owner /
    outputs / prompt: the request identity and its output / prompt token ids (penalty history; outputs includes
    `token`)."""

    slot: int
    token: int
    pos: int
    pages: Sequence[int] | torch.Tensor
    sampling: RowSampling | None = None
    owner: object = None
    outputs: Sequence[int] = field(default_factory=tuple)
    prompt: Sequence[int] = field(default_factory=tuple)


def buckets_for(S: int) -> tuple[int, ...]:
    """Step buckets of a model with S slots: 1, 2, 4, 8, 16 up to S, plus S."""
    return tuple(sorted({b for b in BUCKETS if b <= S} | {S}))


def _u32(device, rows, cols):
    return ttnn.from_torch(torch.zeros(rows, cols, dtype=torch.int32), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)


class BatchDecoder:
    """See the module docstring.  sampler: build the device sampler (S history slots, <= 16 rows; None: greedy only,
    host fallback for the rest); embed: "device" (BF16 table on the device, 0.86 GiB) | "host" (the step's embedding
    rows uploaded from the host before every replay).  taps (batched speculation, tt/spec.py BatchSpec): every main
    step also copies the drafter's target features of its rows (outputs of FEATURE_APPS, the final hidden) into
    `feat` (4 BF16 [1, 1, 32, H]); extra_buckets: more step buckets (verify row counts)."""

    def __init__(self, model, *, sampler: bool = True, log_pages: int = LOG_PAGES, embed: str = "device",
                 taps: bool = False, extra_buckets: Sequence[int] = ()):
        self.model = m = model
        if not m.batched or m.bt is None:
            raise RuntimeError("BatchDecoder needs a batched model (Inkling2Model(max_num_seqs > 1 or kv_blocks))")
        if m._traces:
            raise RuntimeError("BatchDecoder must be built before any decode trace capture (persistent tensors)")
        if embed not in ("device", "host"):
            raise ValueError(f"embed must be 'device' or 'host', got {embed!r}")
        dev, cfg = m.device, m.cfg
        self.device, self.cfg, self.bt = dev, cfg, m.bt
        self.V = cfg.unpadded_vocab
        self.S = m.max_num_seqs
        self.buckets = tuple(sorted(set(buckets_for(self.S)) | {b for b in extra_buckets if b <= MAX_ROWS}))
        self.embed_mode = embed
        self.table = (m.cache.get("embed_rm", lambda: m.embed.table, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                      if embed == "device" else None)
        self.emb = (ttnn.to_device(m.embed.rows([0], 32), dev, memory_config=DRAM) if embed == "host" else None)
        self.tok = _u32(dev, 1, 32)          # word b = row b's input token (the embedding index row)
        self.sel = _u32(dev, MAX_ROWS, 16)   # page b = the sampler's [token, flags, n, 0] of row b
        self.log = _u32(dev, log_pages, LOG_WORDS)
        self.log_pages = log_pages
        self.sampler = (DeviceSampler(dev, cfg.padded_unembed, self.V, num_slots=self.S, rows=max(self.buckets))
                        if sampler else None)
        self.advance = BatchAdvance(dev, self.bt, log_pages)
        self.feat = ([ttnn.zeros([1, 1, 32, cfg.hidden], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=DRAM) for _ in range(4)] if taps else None)
        self._main = {}   # R -> (trace id, device logits)
        self._tail = {}   # (R, kind) -> trace id
        self._rm = {}     # R -> (trace id, ROW_MAJOR logits)
        self.trace_bytes = {}  # R -> trace-region bytes of the bucket's traces (main + tails + rm)
        self._tok = None  # host mirror of tok words 0..31 (None: unknown)
        self._emb_toks = None  # host mode: tokens of the uploaded embedding rows
        self._words = {}  # (spec, slot, nocommit) -> params page
        self.stats = {"steps": 0, "tok_writes": 0, "redraws": 0}
        m.slot_listeners.append(self.invalidate)

    @property
    def kinds(self):
        return KINDS if self.sampler is not None else ("greedy",)

    # ------------------------------------------------------------------ device programs (untraced forms)
    def _main_ops(self, R: int):
        m = self.model
        kw = {} if self.feat is None else {"hook": self._tap, "fin_out": self.feat[3]}
        if self.table is None:
            return m.batch_decode(self.emb, self.bt, R, ids=self.tok, **kw)
        x = ttnn.embedding(self.tok, self.table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
        x = ttnn.reshape(x, [1, 1, 32, self.cfg.hidden])
        lg = m.batch_decode(x, self.bt, R, ids=self.tok, **kw)  # the LM head writes row b's argmax to tok word b
        ttnn.deallocate(x)
        return lg

    def _tap(self, app, layer, p, h):
        """Drafter features (batched speculation): the outputs of FEATURE_APPS -> feat[0..2] (rows = the step's rows)."""
        if app in FEATURE_APPS:
            ttnn.copy(h, self.feat[FEATURE_APPS.index(app)])

    def _tail_ops(self, R: int, kind: str, lg):
        if kind == "greedy":
            self.advance(R, self.tok, self.sel, self.log, from_sel=False)
        else:
            self.sampler.sample(lg, self.sel, self.tok, self.bt.cur_pos, rows=R, kind=kind)
            self.advance(R, self.tok, self.sel, self.log, from_sel=True)

    # ------------------------------------------------------------------ capture
    def capture(self, buckets: Sequence[int] | None = None):
        """Compile every bucket's main / tail / rm programs, then capture them (needs the device's trace region:
        trace_bytes[R] after capture).  The compile runs decode rows of slot 0 at position 0 through page row 0
        (vLLM's null block): call between sequences; ends with model.reset().  Re-capturing releases the previous
        traces."""
        m, dev = self.model, self.device
        self.release()
        buckets = tuple(buckets or self.buckets)
        if self.sampler is not None:
            self.sampler.set_params([RowSampling(temperature=1.0, top_k=20, top_p=0.95, seed=1).words(nocommit=True)]
                                    * MAX_ROWS)
        zero_pages = [torch.zeros(self.bt.W, dtype=torch.int32)]
        for R in buckets:  # compile everything first (model.py trace safety rule 2)
            for kind in self.kinds:
                self.bt.write([0], [0], zero_pages, R)
                self._write_tok([0] * 32)
                lg = self._main_ops(R)
                self._tail_ops(R, kind, lg)
                ttnn.deallocate(ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM))
                ttnn.deallocate(lg)
                self.bt.invalidate()
        ttnn.synchronize_device(dev)
        for R in buckets:
            before = _trace_bytes(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            lg = self._main_ops(R)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            self._main[R] = (tid, lg)
            for kind in self.kinds:
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                self._tail_ops(R, kind, lg)
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                self._tail[(R, kind)] = tid
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            self._rm[R] = (tid, rm)
            after = _trace_bytes(dev)
            self.trace_bytes[R] = None if before is None or after is None else after - before
        ttnn.synchronize_device(dev)
        m.reset()
        self.invalidate()

    def release(self):
        for tid, lg in self._main.values():
            ttnn.release_trace(self.device, tid)
            ttnn.deallocate(lg)
        for tid in self._tail.values():
            ttnn.release_trace(self.device, tid)
        for tid, rm in self._rm.values():
            ttnn.release_trace(self.device, tid)
            ttnn.deallocate(rm)
        self._main.clear()
        self._tail.clear()
        self._rm.clear()
        self.trace_bytes.clear()

    # ------------------------------------------------------------------ host mirror
    def invalidate(self, slot: int | None = None):
        """Forget what the device holds (slot None: everything; else slot s's sampler history, e.g. after
        reset_slot(s) / a prefill into s).  Rows are compared with the mirror every step, so a slot's next step
        re-seeds what changed."""
        if self.sampler is not None:
            self.sampler.invalidate(slot)
        if slot is None:
            self.bt.invalidate()
            self._tok = self._emb_toks = None

    def bucket(self, B: int) -> int:
        R = next((r for r in self.buckets if r >= B), None)
        if R is None or B < 1:
            raise ValueError(f"{B} rows: 1..{self.buckets[-1]} (max_num_seqs)")
        return R

    def _write_tok(self, words):
        row = torch.tensor([list(words)], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(row, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                        self.tok)
        self._tok = list(words)
        self.stats["tok_writes"] += 1

    def _params(self, spec: RowSampling, slot: int, nocommit: bool):
        key = (spec, slot, nocommit)
        w = self._words.get(key)
        if w is None:
            if len(self._words) > 4096:
                self._words.clear()
            w = self._words[key] = spec.words(slot=slot, nocommit=nocommit)
        return w

    def _prepare(self, rows: Sequence[BatchRow], greedy_only: bool = False, groups: bool = False):
        """Host writes so that the device's next replay decodes `rows` (+ pads) -> (R, kind).  groups (verify, batched
        speculation): a slot may hold several rows, consecutive in `rows` at consecutive positions (a verify group);
        sampled rows are drawn without history commits (unpenalized only)."""
        B = len(rows)
        R = self.bucket(B)
        slots = [int(r.slot) for r in rows]
        if groups:
            seen = {}
            for b, r in enumerate(rows):
                s = int(r.slot)
                if s in seen and (seen[s][0] != b - 1 or seen[s][1] + 1 != int(r.pos)):
                    raise ValueError(f"verify rows of slot {s} must be consecutive rows at consecutive positions")
                seen[s] = (b, int(r.pos))
                if r.sampling is not None and r.sampling.penalized:
                    raise ValueError("verify rows: penalized sampling is not speculated")
            if not all(0 <= s < self.S for s in seen):
                raise ValueError(f"slots outside 0..{self.S - 1}: {sorted(seen)}")
        elif len(set(slots)) != B or not all(0 <= s < self.S for s in slots):
            raise ValueError(f"rows need distinct slots in 0..{self.S - 1}, got {slots}")
        for r in rows:
            if not 0 <= r.pos or r.pos + 1 >= self.model.max_seq_len:
                raise ValueError(f"position {r.pos + 1} reaches max_seq_len {self.model.max_seq_len}")
        src = list(range(B)) + [0] * (R - B)
        self.bt.write(slots, [int(r.pos) for r in rows], [r.pages for r in rows], R)
        toks = [int(rows[i].token) for i in src]
        if self._tok is None or self._tok[:R] != toks:
            self._write_tok(toks + (self._tok[R:] if self._tok is not None else [0] * (32 - R)))
        if self.table is None and self._emb_toks != toks:
            ttnn.copy_host_to_device_tensor(self.model.embed.rows(toks, 32), self.emb)
            self._emb_toks = toks
        kinds = [kind_of(r.sampling) for r in rows]
        kind = "greedy" if greedy_only else max(kinds, key=KINDS.index)
        if kind != "greedy":
            if self.sampler is None:
                raise RuntimeError("BatchDecoder built without the device sampler")
            words = []
            for b, i in enumerate(src):
                r = rows[i]
                sp = r.sampling if kinds[i] != "greedy" else _GREEDY
                words.append(self._params(sp, r.slot, nocommit=b >= B or not sp.penalized))
                if b < B and sp.penalized:
                    self.sampler.ensure_history(r.slot, r.owner, r.outputs, r.prompt, sp.uses_prompt)
            self.sampler.set_params(words)
        return R, kind

    # ------------------------------------------------------------------ steps
    def step(self, rows: Sequence[BatchRow], after=None) -> list[int]:
        """One device step of every row -> the next token of each row (greedy, or sampled with its RowSampling).
        after(R): more device work enqueued behind the step before the host waits for its tokens (verify())."""
        R, kind = self._prepare(rows)
        B = len(rows)
        if R not in self._main:
            raise RuntimeError(f"no batch decode trace for bucket {R}: BatchDecoder.capture()")
        dev = self.device
        ttnn.execute_trace(dev, self._main[R][0], cq_id=0, blocking=False)
        ttnn.execute_trace(dev, self._tail[(R, kind)], cq_id=0, blocking=False)
        self.bt.advanced(R)
        self.stats["steps"] += 1
        src = self.tok if kind == "greedy" else self.sel
        if after is not None:
            h = ttnn.from_device(src, blocking=False)
            ev = ttnn.record_event(dev, 0)
            after(R)
            ttnn.event_synchronize(ev)
            src = h
        if kind == "greedy":
            toks = [int(x) for x in ttnn.to_torch(src).reshape(-1)[:R].tolist()]
            self._tok[:R] = toks
            return toks[:B]
        sel = ttnn.to_torch(src).reshape(MAX_ROWS, 16)[:R, :2].to(torch.int64)
        toks = [int(x) for x in sel[:, 0].tolist()]
        inexact = [bool(int(f) & R_INEXACT) for f in sel[:, 1].tolist()]
        redraw = [b for b in range(B) if inexact[b]]
        if redraw:
            lg = self._read_logits(R)
            for b in redraw:
                r = rows[b]
                toks[b] = host_sample(lg[b], r.sampling, r.outputs, r.prompt, r.pos + 1,
                                      P_HOST if r.sampling.full else P_SAMPLE)
                self.stats["redraws"] += 1
            for b in range(B, R):  # pads follow row 0
                toks[b] = toks[0]
            self._write_tok(toks + self._tok[R:])
        else:
            self._tok[:R] = toks
        if self.sampler is not None:
            for b in range(B):
                r = rows[b]
                if r.sampling is not None and kind_of(r.sampling) != "greedy" and r.sampling.penalized \
                        and not inexact[b]:
                    self.sampler.note_committed(r.slot, r.owner, len(r.outputs) + 1)
        return toks[:B]

    def step_logits(self, rows: Sequence[BatchRow], groups: bool = False) -> torch.Tensor:
        """Host-sampling fallback: decode `rows` and return their FP32 logits [B, unpadded_vocab] without choosing
        tokens on the device (positions do not advance: the next step re-seeds from its rows).  groups: rows may hold
        verify groups (verify())."""
        R, _ = self._prepare(rows, greedy_only=True, groups=groups)
        if R not in self._main:
            raise RuntimeError(f"no batch decode trace for bucket {R}: BatchDecoder.capture()")
        ttnn.execute_trace(self.device, self._main[R][0], cq_id=0, blocking=False)
        self._tok = None  # the LM head wrote the argmax ids
        self.stats["steps"] += 1
        return self._read_logits(R)[: len(rows)]

    def verify(self, rows: Sequence[BatchRow], after=None) -> list[int]:
        """Batched speculation: one step over rows that may hold verify groups (a slot's rows consecutive at
        consecutive positions: its anchor and drafts; `_prepare(groups=True)`) -> the target's token after every
        row: greedy rows the argmax (no tail: positions do not advance), sampled rows (unpenalized) a device draw at
        position pos + 1 without history commits (the plain decoder's counter: a seeded row draws what plain decoding
        draws at that position), R_INEXACT rows redrawn on the host from the row's logits.  after(R): called once the
        step's device work is enqueued, before the host waits for its tokens (more device work: the drafter
        context of the rows)."""
        R, kind = self._prepare(rows, groups=True)
        B = len(rows)
        if R not in self._main:
            raise RuntimeError(f"no batch decode trace for bucket {R}: BatchDecoder.capture()")
        dev = self.device
        ttnn.execute_trace(dev, self._main[R][0], cq_id=0, blocking=False)
        self.stats["steps"] += 1
        if kind != "greedy":
            ttnn.execute_trace(dev, self._tail[(R, kind)], cq_id=0, blocking=False)
            self.bt.advanced(R)
        # non-blocking read of the tokens (argmax ids / sampler pages) + event: waiting for it does not wait for the
        # device work after() enqueues behind it
        h = ttnn.from_device(self.tok if kind == "greedy" else self.sel, blocking=False)
        ev = ttnn.record_event(dev, 0)
        if after is not None:
            after(R)
        ttnn.event_synchronize(ev)
        if kind == "greedy":
            toks = [int(x) for x in ttnn.to_torch(h).reshape(-1)[:R].tolist()]
            self._tok[:R] = toks
            return toks[:B]
        sel = ttnn.to_torch(h).reshape(MAX_ROWS, 16)[:R, :2].to(torch.int64)
        toks = [int(x) for x in sel[:, 0].tolist()]
        redraw = [b for b in range(B) if int(sel[b, 1]) & R_INEXACT and rows[b].sampling is not None]
        if redraw:
            lg = self._read_logits(R)
            for b in redraw:
                r = rows[b]
                toks[b] = host_sample(lg[b], r.sampling, (), r.prompt, r.pos + 1,
                                      P_HOST if r.sampling.full else P_SAMPLE)
                self.stats["redraws"] += 1
        self._tok[:R] = toks
        return toks[:B]

    def _read_logits(self, R: int) -> torch.Tensor:
        """FP32 [R, unpadded_vocab] logits of the last main replay of bucket R (device untilize + read)."""
        tid, rm = self._rm[R]
        ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
        return ttnn.to_torch(rm).float().reshape(R, -1)[:, : self.V]

    def generate(self, rows: Sequence[BatchRow], n: int) -> list[list[int]]:
        """n steps back to back from `rows` with one log read at the end -> n lists of the B rows' tokens (bench /
        tests; histories are not committed on the host, so penalized rows and host redraws are not allowed)."""
        if not 1 <= n <= self.log_pages:
            raise ValueError(f"1 <= n <= log_pages ({self.log_pages})")
        if any(r.sampling is not None and r.sampling.penalized for r in rows):
            raise ValueError("generate: penalized rows need per-step history bookkeeping (use step())")
        if any(r.pos + n >= self.model.max_seq_len for r in rows):
            raise ValueError("generate: positions reach max_seq_len")
        if self.table is None:
            raise RuntimeError("generate needs the device embedding table")
        R, kind = self._prepare(rows)
        B = len(rows)
        for _ in range(n):
            ttnn.execute_trace(self.device, self._main[R][0], cq_id=0, blocking=False)
            ttnn.execute_trace(self.device, self._tail[(R, kind)], cq_id=0, blocking=False)
            self.bt.advanced(R)
        self.stats["steps"] += n
        log = ttnn.to_torch(self.log).reshape(self.log_pages, LOG_WORDS).to(torch.int64)
        q0 = int(rows[0].pos)
        out = []
        for i in range(n):
            page = log[(q0 + 1 + i) % self.log_pages]
            if int(page[32]) != q0 + 1 + i:
                raise RuntimeError("token log positions do not match the replayed steps")
            if bool((page[16:16 + B] & R_INEXACT).any()):
                raise RuntimeError("a sampled step needs a host redraw: use step()")
            out.append([int(x) for x in page[:B].tolist()])
        self._tok = None
        return out


def _trace_bytes(dev) -> int | None:
    """Trace-region bytes in use (all banks)."""
    try:
        v = ttnn.get_memory_view(dev, ttnn.BufferType.TRACE)
    except Exception:  # noqa: BLE001
        return None
    return int(v.num_banks * v.total_bytes_allocated_per_bank)
