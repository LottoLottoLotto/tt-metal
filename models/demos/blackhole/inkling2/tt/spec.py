# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for Inkling-2-Mini on one P150: the bundled block drafter (tt/drafter.py) + a multi-row
target VERIFY step + greedy or sampled acceptance + exact commit of the accepted rows' state.

    spec = SpecDecoder(model, rows=(2, 3, 4), table=dec.table, plain=dec)   # adaptive R per cycle (an int: fixed R);
                                             # after Inkling2Model(...) and the TokenDecoder dec, before any trace
                                             # capture (allocates every persistent table / buffer and the drafter)
    spec.capture()                           # compile + capture the verify / context / draft traces (dec captured)
    toks, stats = spec.generate(prompt_ids, n_new)   # greedy; equals plain greedy decoding up to multi-row rounding
    toks, stats = spec.generate(prompt_ids, n_new, sampling=RowSampling(...))  # sampled cycles (below)

Cycle (anchor token a at position p, drafts d_1..d_m from the previous draft, R = 1 + m):
  1. verify (trace): the rows [a, d_1..d_m] (embedded on the device from the anchor id + the drafter's draft log) at
     positions p..p+m run through all 48 applications in ONE pass (model._forward with ctx mode "verify") ->
     softcapped logits of the R rows -> per-row argmax g_0..g_m (fused into the LM head, fast/head.py); the drafter's
     target features of the R rows (outputs of applications 4 / 22 / 40 = inputs of 5 / 23 / 41, final hidden) are
     copied into persistent buffers.  A non-blocking read of g follows.
  2. context (trace): drafter context K/V of the R verify rows (positions p..p+m; rows past the accepted ones are
     rewritten by the next block before any read) -- runs while the host waits for g.
  3. host: accepted k = longest prefix with d_i == g_{i-1}; emitted d_1..d_k, g_k; n = k + 1 committed rows; the
     next R from the decayed acceptance estimate (pick()).
  4. draft (trace; with INK_GLUE=ttnn preceded by the conv-state commit): block at p' = p + n with anchor g_k, f_rnn
     from feature row k -> drafts d'_1..d'_{R'-1} into the draft log.  The next cycle's verify + context are enqueued
     right behind it (the drafts' read does not wait for them), so host work and vLLM's per-token steps overlap device
     work.
Per-op formulation of the verify (VerifyOps, R rows in rows 0..R-1 of [1, 1, 32, H] tiles):
- attention: the fused R-row verify of fast/attn_decode.py (AttnDecodeFast.verify: == R decode steps, causal across
  rows; all attention state is position-indexed, so its commit is "continue at p + n").
- glue: R decode-glue instances (FastGlue.verify, R <= 4) or the fused rows glue (fast/glue_rows.py); conv inputs in
  position rings, nothing to commit; INK_GLUE=ttnn: the prefill-form ttnn glue (ListStates, FP32 mixing) with the
  short convs over [3 state rows; R rows] (read-only; the 3 + R rows are saved for commit).
- MoE: the R-row decode kernel (tt/moe.py rows=R: union of the rows' experts streamed once, row r bitwise the
  single-row decode MoE).
Greedy output equals spec-off decoding up to multi-row rounding (tests/test_spec.py, NOTES-spec.md).
Sampled cycles (set_sampling(spec) before begin(); requests the device sampler takes, fast/sampling.device_eligible):
after step 1 the device sampler draws every verify row (row r: position p + r + 1, penalties counting d_1..d_r) and
an accept step commits the accepted prefix to the device history; step 3 accepts while d_i == the row-(i-1) sample.
This is speculative rejection sampling with the drafter's (argmax) proposal q = delta(d): accept d with probability
min(1, p(d) / q(d)) = p(d), on rejection a sample of the residual max(0, p - q) = p without d; the emitted tokens have
exactly the target distribution, and because every position draws with the plain decoder's counter (seed,
position), a seeded request emits exactly the plain decoder's tokens (tests/test_spec_sampled.py).  A row the device
cannot decide (R_INEXACT) is redrawn on the host from the persistent verify logits.
"""
from __future__ import annotations

import os
import time

import torch
import ttnn

from .attention import AttnVerifyTables
from .common import CK_HIFI4, DRAM, free
from .drafter import BLOCK as DBLOCK, Drafter
from .fast.batch_decode import FEATURE_APPS, BatchRow
from .fast.sampling import P_HOST, P_SAMPLE, R_INEXACT, RowSampling, host_sample, kind_of
from .layers import BF16, F32, ShortConv

KINDS = ("sample", "full", "full_reject")  # sampled-cycle sampler kinds (fast/sampling.kind_of)


def _dev_zeros(device, shape, dtype=BF16, layout=ttnn.TILE_LAYOUT):
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        tdt = torch.int32 if dtype in (ttnn.uint32, ttnn.int32) else torch.bfloat16
        return ttnn.from_torch(torch.zeros(shape, dtype=tdt), dtype=dtype, layout=layout, device=device,
                               memory_config=DRAM)
    return ttnn.zeros(shape, dtype=dtype, layout=layout, device=device, memory_config=DRAM)


def _concat_into(parts, out):
    """Row concat (dim 2) into the persistent tensor `out` (ttnn.concat takes no output tensor)."""
    cat = ttnn.concat(parts, dim=2, memory_config=DRAM)
    ttnn.copy(cat, out)
    ttnn.deallocate(cat)


# ---------------------------------------------------------------------------------------------- per-cycle tables
class VerifyTables:
    """Persistent device inputs of a verify / commit of up to R rows (host-written per cycle) and the verify outputs.
    table: the device BF16 embedding table ([vocab, H] ROW_MAJOR, TokenDecoder.table) -> the verify embeds its rows on
    the device (enable_device_embed: row 0 = the anchor id, rows 1.. = the drafter's draft log), so a verify can be
    enqueued before the host has read the drafts; None: the host uploads the embedding rows (write_verify)."""

    EMB_CHUNK = 9  # column tiles per RowMove move of an embedding row (72 = 8 moves: spread over the movers)

    def __init__(self, model, R: int, keep_logits: bool = False, table=None):
        d, H = model.device, model.cfg.hidden
        self.model, self.R = model, R
        self.emb = _dev_zeros(d, [1, 1, 32, H])
        self.vt = AttnVerifyTables(d, model.cfg)
        self.sel_glue = [_dev_zeros(d, [1, 1, 32, 64], F32) for _ in range(3)]
        self.feat = [_dev_zeros(d, [1, 1, 32, H]) for _ in range(4)]
        self.ids = _dev_zeros(d, [1, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        # verify logits (diagnostics only: one FP32 [32, V] copy per verify)
        self.logits = _dev_zeros(d, [1, 1, R, model.cfg.padded_unembed], F32) if keep_logits else None
        # the verify's softcapped logits (persistent: the device sampler of sampled cycles and host redraws read them)
        self.vlogits = _dev_zeros(d, [1, 1, R, model.cfg.padded_unembed], F32)
        self.pos = None
        self.table = table
        self.anchor_ids = self.draft_log = None
        self.mv_emb = {}

    def enable_device_embed(self, Rs, anchor_ids, draft_log):
        """Device embedding of verify rows: row 0 = word 0 of anchor_ids, row r = word 0 of draft_log row r - 1."""
        from .fast.draft_ops import RowMove

        self.anchor_ids, self.draft_log = anchor_ids, draft_log
        nct, c = self.model.cfg.hidden // 32, self.EMB_CHUNK
        for R in Rs:
            dst = 1 if R == 1 else 2
            mv = [(0, 0, j, dst, 0, j, c) for j in range(0, nct, c)]
            mv += [(1, 32 * i, j, dst, i + 1, j, c) for i in range(R - 1) for j in range(0, nct, c)]
            self.mv_emb[R] = RowMove(self.model.device, mv)

    def device_embed(self, R: int):
        """Traceable: the R verify rows' embeddings into self.emb (rows >= R keep stale rows: every verify consumer
        reads rows 0..R-1 only)."""
        kw = {"layout": ttnn.TILE_LAYOUT, "dtype": BF16, "memory_config": DRAM}
        parts = [ttnn.embedding(self.anchor_ids, self.table, **kw)]
        if R > 1:
            parts.append(ttnn.embedding(self.draft_log, self.table, **kw))
        self.mv_emb[R](parts + [self.emb])
        free(*parts)

    def write_tables(self, pos: int, R: int):
        """Position inputs of a verify of R rows at pos .. (attention tables, glue rows)."""
        m = self.model
        self.vt.update(pos)
        if m.rows_glue is not None:
            m.rows_glue.set_pos(pos, R)
        self.pos = pos

    def write_verify(self, tokens: list[int], pos: int):
        """Inputs of a verify of len(tokens) (<= R) rows at positions pos ... (host embedding rows)."""
        assert 1 <= len(tokens) <= self.R
        ttnn.copy_host_to_device_tensor(self.model.embed.rows(tokens, 32), self.emb)
        self.write_tables(pos, len(tokens))

    def write_commit(self, n: int):
        """n accepted rows (1..R) of the last verify (glue conv states; nothing with the rows glue)."""
        if self.model.rows_glue is not None:
            return
        for j in range(3):
            g = torch.zeros(1, 1, 32, 64)
            g[..., n + j] = 1
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(g, dtype=F32, layout=ttnn.TILE_LAYOUT), self.sel_glue[j])

    def read_logits(self, rows: int) -> torch.Tensor:
        V = self.model.cfg.unpadded_vocab
        return ttnn.to_torch(self.logits).float().reshape(-1, self.logits.padded_shape[-1])[:rows, :V]

    def read_vlogits(self, row: int) -> torch.Tensor:
        """Host FP32 logits [unpadded_vocab] of verify row `row` (host redraws of sampled cycles)."""
        V = self.model.cfg.unpadded_vocab
        return ttnn.to_torch(self.vlogits).float().reshape(-1, self.vlogits.padded_shape[-1])[row, :V]

    def read_ids(self, rows: int) -> list[int]:
        return [int(x) for x in ttnn.to_torch(self.ids).reshape(-1)[:rows]]


# ---------------------------------------------------------------------------------------------- multi-row ops
class VerifyOps:
    """ctx["ops"] of model._forward in mode "verify": attention / conv / MoE on R rows (<= tables.R); the ttnn-glue
    convs are read-only on state and save what commit() needs in persistent buffers."""

    def __init__(self, model, tables: VerifyTables, R: int):
        self.m, self.t, self.R = model, tables, R
        d, H = model.device, model.cfg.hidden
        if not all(getattr(a, "fast", None) is not None for a in model.attn):
            raise ValueError("speculative verify needs the fused attention (Inkling2Attention.enable_fast)")
        self.glue_ext = {}
        if model.rows_glue is None:
            # saved conv inputs for commit: 64-row buffers [3 state rows | 32 block rows | 29 zero rows]
            for g in model.glue:
                for conv in (g.attn_conv, g.mlp_conv):
                    for p in range(len(conv.s)):
                        self.glue_ext[id(conv), p] = _dev_zeros(d, [1, 1, 64, H], F32)
            self.tail_h = _dev_zeros(d, [1, 1, 64 - 35, H], F32)
        self.debug = bool(int(__import__("os").environ.get("INK_SPEC_DEBUG", "0")))

    def _dbg(self, stage: str):
        if self.debug:
            ttnn.synchronize_device(self.m.device)
            print(f"[verify] {stage} ok", flush=True)

    def attend(self, layer: int, p: int, xq, xk, xv):
        y = self.m.attn[layer].fast.verify(xq, xk, xv, self.t.vt, self.R, p)
        self._dbg(f"attn L{layer} p{p}")
        return y

    def conv(self, conv: ShortConv, x, p: int):
        H = conv.H
        st = conv.s[p]
        ext = self.glue_ext[id(conv), p]
        heads = [ttnn.slice(s, [0, 0, 0, 0], [1, 1, 1, H]) for s in st]
        xf = ttnn.typecast(x, F32, memory_config=DRAM)
        _concat_into(heads + [xf, self.tail_h], ext)
        free(*heads)
        rows = [ttnn.slice(ext, [0, 0, j, 0], [1, 1, j + 32, H]) for j in range(3)]
        y = ShortConv._sum(rows + [xf], conv.taps)
        free(*rows, xf)
        return y

    def moe(self, layer: int, z):
        y = self.m.moe[layer](z, decode=True, rows=self.R)
        self._dbg(f"moe L{layer}")
        return y

    def commit(self):
        """Glue conv states after the n accepted rows (tables.write_commit(n) before); attention: nothing."""
        t = self.t
        for g in self.m.glue:
            for conv in (g.attn_conv, g.mlp_conv):
                for p in range(len(conv.s)):
                    ext = self.glue_ext.get((id(conv), p))
                    if ext is None:
                        continue
                    for j in range(3):
                        ttnn.matmul(t.sel_glue[j], ext, dtype=F32, compute_kernel_config=CK_HIFI4,
                                    optional_output_tensor=conv.s[p][j])
        self._dbg("commit glue")


class _Tap:
    """model._forward tap: copies the outputs of FEATURE_APPS into the persistent feature rows."""

    def __init__(self, feat):
        self.feat = feat

    def __call__(self, app, layer, p, h):
        if app in FEATURE_APPS:
            ttnn.copy(h, self.feat[FEATURE_APPS.index(app)])


# ---------------------------------------------------------------------------------------------- the decoder
class SpecDecoder:
    """Speculative decoding of one sequence, greedy or sampled (host loop over device traces).  rows = R = 1 + m verified rows
    (m = drafts verified per cycle, 1..7): an int (fixed R) or a tuple of row counts (adaptive: every cycle verifies the
    R of the tuple with the most expected tokens per ms, from a decayed estimate of the per-draft acceptance rate of
    this sequence and the per-R cycle costs measured at capture; see pick()).

    API:  logits = prefill(ids)            target prefill + drafter context of the prompt (host FP32 logits [V])
          begin(token, pos)                first draft for the anchor `token` at pos = len(ids)
          emitted = step(token, pos)       one cycle: the target's greedy tokens of positions pos+1 .. pos+len(emitted);
                                           the next cycle continues at (emitted[-1], pos + len(emitted))
          generate(ids, n)                 prefill + begin + steps (tests)
    table (TokenDecoder.table, the device embedding table): the verify embeds its rows on the device (anchor id + the
    drafter's draft log), so each cycle's next verify + drafter context is enqueued right behind its draft and the
    host's acceptance work overlaps device work (pipelined; None: host-uploaded rows, verify after the drafts read).
    plain (TokenDecoder): capture() times its decode step for the hand-off rule; generate() continues with it.
    `active` is False when no cycle can continue (fits(pos) False: a cycle needs pos + max R + 8 <= max_seq_len) or
    after a hand-off (`handoff` True: over the last HANDOFF_CYCLES cycles the emitted tokens per estimated ms stayed
    below HANDOFF_RATE x the plain decode step's rate: the drafter is not paying on this sequence).  After any cycle the
    model state equals plain decoding of the consumed tokens, so plain decode (fast/decode_loop TokenDecoder) may
    continue from (emitted[-1], pos + len(emitted)); spec cannot resume after plain steps (the drafter context would
    miss those rows) until the next prefill.  Sampled sequences (set_sampling) run sampled cycles with plain's
    DeviceSampler (module docstring).  keep_logits (fixed R only): the verify also copies its FP32 logits into t.logits
    (diagnostics; on_cycle can read them)."""

    ACC_PRIOR = (1.1, 0.9)  # (accepted, rejected) pseudo-counts of a new sequence: acceptance rate 0.55
    ACC_DECAY = 0.9         # per-cycle decay of the counts (~10-cycle memory)
    HANDOFF_CYCLES = 64     # hand-off window (cycles) ...
    HANDOFF_RATE = 0.9      # ... and the fraction of the plain rate below which a sequence hands off (irreversible)

    def __init__(self, model, rows: int | tuple[int, ...] = 4, log=print, keep_logits: bool = False,
                 drafter: Drafter | None = None, table=None, plain=None):
        self.Rs = tuple(sorted({rows} if isinstance(rows, int) else set(rows)))
        if not self.Rs or not all(2 <= r <= DBLOCK for r in self.Rs):
            raise ValueError(f"rows must be in 2..{DBLOCK}, got {rows}")
        if keep_logits and len(self.Rs) > 1:
            raise ValueError("keep_logits needs a fixed R")
        self.m, self.R, self.log = model, self.Rs[-1], log
        cfg, d = model.cfg, model.device
        self.t = VerifyTables(model, self.R, keep_logits, table=table)
        self.ops = {R: VerifyOps(model, self.t, R) for R in self.Rs}
        self.tap = _Tap(self.t.feat)
        # drafter: may be shared between decoders (built with ctx_widths covering every R)
        self.dr = drafter or Drafter(d, cfg, model.cache, model.max_seq_len, hf_dir=model.src.hf_dir,
                                     source=model.src, ctx_widths=self.Rs)
        if not set(self.Rs) <= set(self.dr.ctx_widths):
            raise ValueError(f"drafter context widths {self.dr.ctx_widths} miss some of rows {self.Rs}")
        self.pipelined = table is not None
        if self.pipelined:
            self.t.enable_device_embed(self.Rs, self.dr.inp.anchor_id, self.dr.log)
        self._queued = None  # ((R, pos), ids getter) of a verify already enqueued
        if model.prefill_chunk % 64:
            raise ValueError(f"prefill_chunk {model.prefill_chunk}: the drafter's context fill needs a multiple of 64")
        self.traces = {}
        # ms of a cycle of R rows (pick()): verify + context + draft replays measured by capture() (the pipelined cycle
        # hides the host share)
        self.cost = {R: 4.0 + 4.0 * R for R in self.Rs}
        self.plain = plain
        self.plain_ms = None    # TokenDecoder step ms (capture(); None: no hand-off)
        # sampled cycles: the plain decoder's DeviceSampler (shared history slot 0) on the verify rows
        self.sampler = plain.sampler if plain is not None and table is not None else None
        self.vsel = (_dev_zeros(d, [8, 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if self.sampler is not None
                     else None)
        self.sampling: RowSampling | None = None  # this sequence's sampling (None: greedy cycles)
        self._owner, self._outputs, self._prompt = None, [], []
        self.cost_s = dict(self.cost)  # sampled cycles (verify + sampler + context + draft) ms per R
        self.plain_ms_s = None         # plain sampled step ms (full-vocabulary kind)
        self.eos = cfg.eos_token_id
        self.on_cycle = None  # tests: on_cycle(pos, verify tokens, verify argmax ids, accepted k) after each verify
        self._compiled = False
        self._first_sel = None  # feature row of the last prompt token in t.feat (set by prefill)
        self.active = False     # a draft is pending for (anchor, pos) on this sequence's drafter context
        self.handoff = False    # the last sequence stopped speculating (see the class docstring)
        self.anchor = self.pos = None
        self.drafts: list[int] = []
        self.next_R = self.R    # rows of the next verify (the pending draft has next_R - 1 drafts)
        self.acc = list(self.ACC_PRIOR)
        self._window: list[tuple[int, float]] = []  # (emitted, estimated ms) of the last HANDOFF_CYCLES cycles
        self.stats = self._new_stats()

    @staticmethod
    def _new_stats():
        return {"cycles": 0, "emitted": 0, "accepted_hist": [], "rows_hist": [], "host_s": 0.0, "cycle_s": 0.0}

    def fits(self, pos: int) -> bool:
        """A cycle at pos stays inside max_seq_len: verify rows pos .. pos+R-1, next block at <= pos+R .. pos+R+7."""
        return pos + self.R + DBLOCK <= self.m.max_seq_len

    def pick(self) -> int:
        """Rows of the next verify: argmax over R of E[tokens / cycle] / cost[R], E = 1 + a + .. + a^(R-1) with a =
        the decayed acceptance rate of this sequence's drafts (geometric model: draft i is accepted with probability
        a given draft i-1 was)."""
        if len(self.Rs) == 1:
            return self.R
        a = self.acc[0] / (self.acc[0] + self.acc[1])
        cost = self.cost_s if self.sampling is not None else self.cost
        return max(self.Rs, key=lambda R: sum(a ** i for i in range(R)) / cost[R])

    def _observe(self, k: int, R: int):
        """Cycle of R rows accepted k of its R - 1 drafts: k successes, one failure unless all were accepted."""
        d = self.ACC_DECAY
        self.acc = [d * self.acc[0] + k, d * self.acc[1] + (k < R - 1)]

    def _account(self, R: int, n: int):
        """Hand-off rule: the last HANDOFF_CYCLES cycles emitted fewer tokens per (capture-estimated) ms than
        HANDOFF_RATE x the plain decode rate."""
        sampled = self.sampling is not None
        plain_ms = self.plain_ms_s if sampled else self.plain_ms
        if plain_ms is None:
            return
        self._window = (self._window + [(n, (self.cost_s if sampled else self.cost)[R])])[-self.HANDOFF_CYCLES:]
        if len(self._window) == self.HANDOFF_CYCLES:
            toks, ms = sum(w[0] for w in self._window), sum(w[1] for w in self._window)
            if toks * plain_ms < self.HANDOFF_RATE * ms:
                self.handoff = True

    # ------------------------------------------------------------------ device steps (traceable)
    def _verify(self, R: int):
        m, t = self.m, self.t
        if self.pipelined:
            t.device_embed(R)
        x = m.embed(t.emb)
        # emb: the raw rows, so the fused glue's layer 0 fuses embed_norm exactly as the decode step does
        out = m._forward(x, {"mode": "verify", "ops": self.ops[R], "tap": self.tap, "emb": t.emb})
        ttnn.copy(out, t.feat[3])
        # fused LM head: softcapped logits of rows 0..R-1 (persistent t.vlogits) + per-row argmax
        lg = m.head(out, rows=R, ids=t.ids, out=t.vlogits)
        ttnn.deallocate(out)
        if t.logits is not None:
            ttnn.copy(lg, t.logits)

    def _vsample(self, R: int, kind: str):
        """Sampled cycles: the device sampler on the R verify rows (row r: position p + r + 1, penalties counting
        drafts d_1..d_r from the draft log) -> vsel pages; accept: history += the accepted prefix + the row-k token,
        the R sampled tokens -> t.ids (fast/sampling.py)."""
        t, smp = self.t, self.sampler
        smp.sample(t.vlogits, self.vsel, t.ids, t.vt.cur_pos, rows=R, kind=kind, aux=self.dr.log, spec=True)
        smp.accept(t.vlogits, self.vsel, t.ids, t.vt.cur_pos, self.dr.log, R)

    def _context(self, R: int):
        """Drafter context K/V of the R verify rows (positions dr.ctx_inp[R]); independent of the acceptance: rows past
        the accepted ones are rewritten by the next block before any read."""
        self.dr.context(self.t.feat, R)

    def _commit_draft(self, R: int, steps: int):
        """Commit of the last verify of R rows (ttnn glue: conv states) + the next draft of `steps` drafts."""
        self.ops[R].commit()
        self.dr.draft(self.t.feat, steps=steps)

    def _draft(self, steps: int):
        self.dr.draft(self.t.feat, steps=steps)

    def _after_key(self, R: int, steps: int):
        """Trace key of the post-acceptance program of a cycle of R rows followed by `steps` drafts: commit + draft
        (ttnn glue: conv states to commit), else the draft alone (the rows glue's position rings commit nothing)."""
        return ("commit_draft", R, steps) if self.m.rows_glue is None else ("draft", steps)

    def _programs(self):
        """(trace key, fn) of every spec trace: verify and context per R, the post-acceptance program per (R done,
        drafts of the next R), the first draft per drafts count."""
        steps = sorted({R - 1 for R in self.Rs})
        out = [(("verify", R), lambda R=R: self._verify(R)) for R in self.Rs]
        out += [(("context", R), lambda R=R: self._context(R)) for R in self.Rs]
        if self.m.rows_glue is None:
            out += [(("commit_draft", R, s), lambda R=R, s=s: self._commit_draft(R, s)) for R in self.Rs for s in steps]
        out += [(("draft", s), lambda s=s: self._draft(s)) for s in steps]
        if self.sampler is not None:
            out += [(("vsample", R, kd), lambda R=R, kd=kd: self._vsample(R, kd)) for R in self.Rs for kd in KINDS]
        return out

    # ------------------------------------------------------------------ traces
    def compile(self):
        """Run every program once untraced (verify, context, commit + draft, the prompt context of every prefill chunk
        length): call before capturing ANY trace of the device (a program compiled after a capture can place device
        state inside that trace's intermediates).  Leaves the model reset."""
        self.t.write_commit(1)
        self.dr.inp.write(torch.zeros(DBLOCK, self.m.cfg.hidden), 1, 0, 0)
        for R in self.Rs:
            self.t.write_verify([0] * R, 0)
            self._verify(R)
            self.dr.ctx_inp[R].write(torch.arange(R))
            self._context(R)
            self._commit_draft(R, self.R - 1)
        if self.sampler is not None:  # sampled cycles' sampler programs (params / history contents irrelevant)
            self.sampler.set_params([RowSampling(temperature=1.0, top_k=0, top_p=0.9, seed=1).words(nocommit=True,
                                                                                                extra=r)
                                     for r in range(self.R)])
            for R in self.Rs:
                for kd in KINDS:
                    self._vsample(R, kd)
            self.sampler.invalidate()
        # the drafter's prompt context of every prefill chunk length (no program may compile after a capture)
        H = self.m.cfg.hidden
        for Tp in self.m.warmup_shapes():
            fz = [_dev_zeros(self.m.device, [1, 1, Tp, H]) for _ in range(4)]
            self.dr.context_fill(fz, 0)
            free(*fz)
        ttnn.synchronize_device(self.m.device)
        self.m.reset()
        self.active = False
        self._compiled = True

    def capture(self):
        """Capture every spec trace (compile() first if needed: all compiles must precede every capture of the
        device), then time each R's cycle (verify + context + commit / draft replays) for pick() and, with `plain`
        (captured before), its decode step (smallest bucket) for the hand-off rule."""
        if not self._compiled:
            self.compile()
        self.release()
        d = self.m.device
        for key, fn in self._programs():
            tid = ttnn.begin_trace_capture(d, cq_id=0)
            fn()
            ttnn.end_trace_capture(d, tid, cq_id=0)
            self.traces[key] = tid
        ttnn.synchronize_device(d)

        def replay_ms(tids, n=3):
            for tid in tids:
                ttnn.execute_trace(d, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(n):
                for tid in tids:
                    ttnn.execute_trace(d, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(d)
            return (time.perf_counter() - t0) / n * 1e3

        if self.sampler is not None:
            self.sampler.set_params([RowSampling(temperature=1.0, top_k=0, seed=1).words(nocommit=True, extra=r)
                                     for r in range(self.R)])
        for R in self.Rs:
            self.t.write_verify([0] * R, 0)
            keys = [("verify", R), ("context", R), self._after_key(R, R - 1)]
            self.cost[R] = replay_ms([self.traces[k] for k in keys])
            if self.sampler is not None:
                self.cost_s[R] = replay_ms([self.traces[k] for k in keys[:1] + [("vsample", R, "full")] + keys[1:]])
        if self.plain is not None and self.plain._main:
            b = min(self.plain._main)
            self.m._use_bucket(b).update(0)
            self.plain_ms = replay_ms([self.plain._main[b][0], self.plain._tail[(b, "greedy")]])
            if (b, "full") in self.plain._tail:
                self.plain.sampler.set_params([RowSampling(temperature=1.0, top_k=0, seed=1).words(nocommit=True)])
                self.plain_ms_s = replay_ms([self.plain._main[b][0], self.plain._tail[(b, "full")]])
            self.plain.invalidate()
        if self.sampler is not None:
            self.sampler.invalidate()
        self.log(f"spec: cycle ms per R {({R: round(c, 2) for R, c in self.cost.items()})}, plain step ms "
                 f"{self.plain_ms and round(self.plain_ms, 2)}; sampled (full vocabulary) cycle ms per R "
                 f"{({R: round(c, 2) for R, c in self.cost_s.items()}) if self.sampler else None}, plain step ms "
                 f"{self.plain_ms_s and round(self.plain_ms_s, 2)}")
        self.m.reset()
        self.active = False

    def release(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.m.device, tid)
        self.traces.clear()

    def _run(self, key, fn):
        tid = self.traces.get(key)
        if tid is None:
            fn()
        else:
            ttnn.execute_trace(self.m.device, tid, cq_id=0, blocking=False)

    # ------------------------------------------------------------------ prefill with drafter context
    def prefill(self, ids, start: int = 0, restore=None, at=None) -> torch.Tensor:
        """Target prefill of the prompt (model.prefill's chunks) + the drafter's context K/V of every prompt row.
        Returns the host FP32 logits [unpadded_vocab] of the last prompt token (== model.prefill's).  Prefix resume
        (NOTES-prefix.md): start > 0 (a multiple of the model's prefill chunk, so the chunks are a full prefill's)
        continues from the prefix ids[:start]: restore() puts its snapshot into the model instead of the reset (the
        target's full caches and the drafter context of rows < start are the identity caches' rows); at(pos) as
        model.prefill."""
        m, dr = self.m, self.dr
        ids = torch.as_tensor(ids, dtype=torch.int64).reshape(-1)
        L = ids.numel()
        if not 0 <= start < L <= m.max_seq_len:
            raise ValueError(f"prompt rows {start}..{L - 1} outside 0..{m.max_seq_len - 1}")
        if start:
            if start % m.prefill_chunk or restore is None:
                raise ValueError(f"resumed prefill at {start}: a multiple of the prefill chunk {m.prefill_chunk}, "
                                 "with the prefix snapshot's restore")
            restore()
        else:
            m.reset()
        self.abandon()
        self.active = False
        self.stats = self._new_stats()
        feats = {}

        def tap(app, layer, p, h):
            if app in FEATURE_APPS:
                feats[app] = ttnn.clone(h)

        last = None
        for rel, n in m.prefill_schedule(L - start):  # chunk starts: multiples of PREFILL_TAIL (64-aligned)
            pos = start + rel
            if at is not None:
                at(pos)
            Tp = -(-n // 32) * 32
            x = m.chunk_input(ids[pos:pos + n], Tp)
            ctx = {"mode": "prefill", "start": pos, "valid": n, "tap": tap}
            if m.rows_glue is None:
                ctx["conv_idx"] = ShortConv.state_index(m.device, n, m.cfg.conv_kernel)
            out = m._forward(x, ctx)
            if "conv_idx" in ctx:
                ttnn.deallocate(ctx["conv_idx"])
            fl = [feats[a] for a in FEATURE_APPS] + [out]
            dr.context_fill(fl, pos)
            if pos + n == L:
                # the logits of the last prompt token (== model.prefill's), and the features of its 32-row block
                # (row r = position L-1: the first draft's f_rnn row) into the verify feature buffers
                last = m.rows_logits(out, [n - 1])[0]
                s = (n - 1) // 32 * 32
                H = m.cfg.hidden
                rows = fl if Tp == 32 else [ttnn.slice(f, [0, 0, s, 0], [1, 1, s + 32, H]) for f in fl]
                for i, f in enumerate(rows):
                    ttnn.copy(f, self.t.feat[i])
                self._first_sel = n - 1 - s
                if rows is not fl:
                    free(*rows)
            free(*fl)
            feats.clear()
        if at is not None:
            at(L)
        self.pos = L
        return last

    # ------------------------------------------------------------------ sampling
    def set_sampling(self, spec: RowSampling | None, *, owner=None, output_ids=(), prompt_ids=()):
        """Sampling of this sequence's cycles (before begin()): None / greedy without penalties -> greedy cycles
        (verify argmax); else every verify row is sampled on the device (history = output_ids counts + prompt flags,
        kept current by the cycles' accept step)."""
        if spec is not None and spec.greedy and not spec.penalized:
            spec = None
        if spec is not None and self.sampler is None:
            raise RuntimeError("sampled speculative cycles need the pipelined decoder with the plain DeviceSampler")
        self.sampling = spec
        self._owner, self._outputs, self._prompt = owner, list(output_ids), list(prompt_ids)
        if spec is not None:
            self.sampler.set_params([spec.words(nocommit=True, extra=r) for r in range(self.R)])
            self.sampler.ensure_history(0, owner, self._outputs, self._prompt, spec.uses_prompt)

    def abandon(self):
        """Drop an enqueued next cycle (the sequence ends or continues plainly): a sampled one has committed its
        accepted tokens to the device history, which then no longer matches the host's record."""
        if self._queued is not None and self.sampling is not None:
            self.sampler.invalidate(0)
        self._queued = None

    def _read_sel_async(self, rows: int):
        """Non-blocking read of the sampled cycle's per-row [token, flags] (vsel pages)."""
        h = ttnn.from_device(self.vsel, blocking=False)
        ev = ttnn.record_event(self.m.device, 0)

        def get():
            ttnn.event_synchronize(ev)
            x = ttnn.to_torch(h).reshape(8, 16)[:rows].to(torch.int64)
            return [int(v) for v in x[:, 0]], [int(v) for v in x[:, 1]]
        return get

    # ------------------------------------------------------------------ cycles
    def _read_async(self, tensor, rows: int):
        """Non-blocking read of word 0.. of a uint32 row-major device tensor -> get(): waits for this read only (not
        for device work enqueued after it) and returns the first `rows` of its first column (or of its first row
        when it has one row)."""
        h = ttnn.from_device(tensor, blocking=False)
        ev = ttnn.record_event(self.m.device, 0)

        def get() -> list[int]:
            ttnn.event_synchronize(ev)
            x = ttnn.to_torch(h).reshape(-1, 32)
            v = x[0] if x.shape[0] == 1 else x[:, 0]
            return [int(i) for i in v[:rows]]
        return get

    def _enqueue_verify(self, R: int, pos: int, toks):
        """Enqueue a cycle's acceptance-independent device work: the verify of R rows at pos (toks: the host rows,
        unused when the device embeds them), a non-blocking read of its argmax ids, then the drafter context of its
        rows (which runs while the host waits for the ids and prepares the next block) -> the ids getter."""
        if self.pipelined:
            self.t.write_tables(pos, R)
        else:
            self.t.write_verify(toks, pos)
        self._run(("verify", R), lambda: self._verify(R))
        if self.sampling is not None:
            kd = kind_of(self.sampling)
            self._run(("vsample", R, kd), lambda: self._vsample(R, kd))
            get = self._read_sel_async(R)
        else:
            get = self._read_async(self.t.ids, R)
        self.dr.ctx_inp[R].write(torch.arange(pos, pos + R))
        self._run(("context", R), lambda: self._context(R))
        return get

    def _queue_next(self, steps: int):
        """After a draft of `steps` drafts is enqueued: (pipelined) enqueue the next cycle's verify + context behind
        it, then return the drafts (their read does not wait for that verify)."""
        get = self._read_async(self.dr.log, steps)
        if self.pipelined and self.active:
            self._queued = (self.next_R, self.pos), self._enqueue_verify(self.next_R, self.pos, None)
        return get()

    def begin(self, token: int, pos: int):
        """First draft of the sequence prefilled by prefill(): anchor `token` (the first generated token) at pos."""
        sel = self._first_sel
        if sel is None or pos != self.pos:
            raise RuntimeError(f"begin({pos}) must follow prefill() of {pos} tokens")
        if self.sampling is not None and not self.pipelined:
            raise RuntimeError("sampled speculative cycles need the pipelined decoder")
        self._first_sel = None
        self.handoff = False
        self._window = []
        self._queued = None
        self.active = self.fits(pos)
        if not self.active:
            return
        self.acc = list(self.ACC_PRIOR)
        R = self.next_R = self.pick()
        self.anchor, self.pos = int(token), pos
        self.dr.inp.write(self._block_input(token), pos, token, sel)
        self._run(("draft", R - 1), lambda: self._draft(R - 1))
        self.drafts = self._queue_next(R - 1)

    def step(self, token: int, pos: int) -> list[int]:
        """One verify / commit / draft cycle for the anchor (token, pos) -> emitted tokens (positions pos+1 ..)."""
        if not self.active or (int(token), pos) != (self.anchor, self.pos):
            raise RuntimeError(f"spec step at ({token}, {pos}) but the drafter holds ({self.anchor}, {self.pos}), "
                               f"active={self.active}")
        t, dr, R = self.t, self.dr, self.next_R
        c0 = time.perf_counter()
        toks = [int(token)] + self.drafts[: R - 1]
        if self._queued is not None and self._queued[0] == (R, pos):
            get_ids = self._queued[1]
        else:
            get_ids = self._enqueue_verify(R, pos, toks)
        self._queued = None
        # while the verify runs: the anchor-independent block inputs of every possible acceptance
        pre = [dr.inp.host_pos(pos + j + 1, j) for j in range(R)]
        c1 = time.perf_counter()
        sampled = self.sampling is not None
        g, flags = get_ids() if sampled else (get_ids(), [0] * R)
        c2 = time.perf_counter()
        # greedy: the drafts the argmax rows confirm; sampled: the drafts the rows' samples equal (speculative rejection
        # sampling of the drafter's point-mass proposal, fast/sampling.py), up to a row the device could not decide
        k = 0
        while k < R - 1 and not flags[k] & R_INEXACT and self.drafts[k] == g[k]:
            k += 1
        n = k + 1
        if sampled:
            if flags[k] & R_INEXACT:  # exact host redraw of row k (the device committed drafts 1..k only)
                sp = self.sampling
                g = list(g)
                g[k] = host_sample(t.read_vlogits(k), sp, self._outputs + self.drafts[:k], self._prompt,
                                   pos + k + 1, P_HOST if sp.full else P_SAMPLE)
                self._outputs += self.drafts[:k] + [g[k]]
                self.sampler.invalidate(0)
                self.sampler.ensure_history(0, self._owner, self._outputs, self._prompt, sp.uses_prompt)
                self.stats["host_redraws"] = self.stats.get("host_redraws", 0) + 1
            else:
                self._outputs += self.drafts[:k] + [g[k]]
                self.sampler.note_committed(0, self._owner, len(self._outputs))
        emitted = self.drafts[:k] + [g[k]]
        if self.on_cycle is not None:
            self.on_cycle(pos, toks, g, k)
        a, pnew = g[k], pos + n
        self._observe(k, R)
        self._account(R, n)
        R2 = self.next_R = self.pick()
        # commit the n rows (ttnn glue), next block with the R2 - 1 drafts of the next verify
        t.write_commit(n)
        dr.inp.write_pairs(pre[k] + dr.inp.host_anchor(self._block_input(a), a))
        c3 = time.perf_counter()
        self._run(self._after_key(R, R2 - 1), lambda: self._commit_draft(R, R2 - 1))
        self.anchor, self.pos = a, pnew
        self.active = self.fits(pnew) and not self.handoff
        self.drafts = self._queue_next(R2 - 1)
        c4 = time.perf_counter()
        st = self.stats
        st["cycles"] += 1
        st["emitted"] += n
        st["accepted_hist"].append(k)
        st["rows_hist"].append(R)
        st["host_s"] += (c1 - c0) + (c3 - c2)
        st["cycle_s"] += c4 - c0
        return emitted

    def generate(self, prompt_ids, n_new: int, stop_at_eos: bool = True, sampling: RowSampling | None = None):
        """Spec decoding (greedy, or `sampling`: the first token from the prefill logits by host_sample at the prompt
        length, then sampled cycles): returns (tokens, stats).  Stops early when the sequence reaches max_seq_len; after
        a hand-off the sequence continues on `plain` (as the adapter does)."""
        t0 = time.perf_counter()
        lg = self.prefill(prompt_ids)
        prompt = [int(x) for x in torch.as_tensor(prompt_ids).reshape(-1)]
        sampled = sampling is not None and not (sampling.greedy and not sampling.penalized)
        a = host_sample(lg, sampling, [], prompt, len(prompt)) if sampled else int(lg.argmax())
        t_prefill = time.perf_counter() - t0
        p = len(prompt_ids)
        out = [a]
        owner = object()
        self.set_sampling(sampling, owner=owner, output_ids=out, prompt_ids=prompt)
        t1 = time.perf_counter()
        self.begin(a, p)
        while len(out) < n_new and not (stop_at_eos and self.eos in out) and self.active:
            em = self.step(a, p)
            out += em
            p += len(em)
            a = em[-1]
        handoff_at = len(out) if self.handoff else None
        self.abandon()
        if self.handoff:
            self.plain.invalidate()  # seeds from (a, p): the spec cycles wrote the state behind its back
            self.plain.set_sampling(self.sampling, owner=owner, output_ids=out, prompt_ids=prompt)
            while len(out) < n_new and not (stop_at_eos and self.eos in out):
                a = self.plain.decode(a, p)
                out.append(a)
                p += 1
        dt = time.perf_counter() - t1
        if stop_at_eos and self.eos in out:
            out = out[: out.index(self.eos) + 1]
        out = out[:n_new]
        st = self.stats
        cycles = st["cycles"]
        stats = {"prefill_s": round(t_prefill, 3), "decode_s": round(dt, 3), "cycles": cycles,
                 "tok_per_cycle": round(st["emitted"] / max(cycles, 1), 3), "accepted_hist": list(st["accepted_hist"]),
                 "rows_hist": list(st["rows_hist"]), "handoff_at": handoff_at,
                 "ms_per_cycle": round(1000 * dt / max(cycles, 1), 2) if not self.handoff else None,
                 "step_ms_per_cycle": round(1000 * st["cycle_s"] / max(cycles, 1), 2),
                 "host_ms_per_cycle": round(1000 * st["host_s"] / max(cycles, 1), 3),
                 "host_redraws": st.get("host_redraws", 0),
                 "tok_s": round((len(out) - 1) / dt, 2) if dt > 0 else None}
        return out, stats

    def _block_input(self, a: int) -> torch.Tensor:
        """Drafter block input [8, H] BF16: row 0 = embed_norm(embed[a]) (host fp32 RMSNorm, the reference's math),
        rows 1..7 = mask_embed."""
        m = self.m
        x0 = self.dr.mask_embed.expand(DBLOCK, m.cfg.hidden).clone()
        e = m.embed.table[a].float()
        x0[0] = (e * torch.rsqrt(e.pow(2).mean() + m.cfg.eps) * self.dr.embed_norm_w).to(torch.bfloat16)
        return x0


# ---------------------------------------------------------------------------------------------- batched speculation
class _Seq:
    """Speculation state of one physical slot (BatchSpec): the device state holds the sequence up to `front` (the
    anchor token `anchor` at position `front` is the next input), vLLM is `len(queue)` tokens behind it (queue =
    [(position, token in, token out)] of target tokens vLLM has not asked for yet); `drafts` = the drafter's tokens
    for the next verify of this slot (empty: the next device row is a plain one)."""

    __slots__ = ("front", "anchor", "queue", "drafts", "acc")

    def __init__(self):
        self.reset()

    def reset(self):
        self.front = self.anchor = None
        self.queue, self.drafts = [], []
        self.acc = list(SpecDecoder.ACC_PRIOR)


class BatchSpec:
    """Speculative decoding shared across the sequences of the batched server (tt/NOTES-spec.md "Batched
    speculation").  One device step verifies every speculating sequence's anchor + drafts as a VERIFY GROUP (its rows
    consecutive in the step at consecutive positions: fast/batch_decode.BatchDecoder.verify; the batched attention /
    glue kernels give each row its group's earlier rows exactly as the single-stream verify does, so every row equals
    that sequence's decode step bitwise) next to plain rows of other sequences, then accepts per sequence (greedy:
    drafts equal to the argmax of the row before; sampled, unpenalized: drafts equal to the row before's device draw
    at its own position = speculative rejection sampling of the drafter's point-mass proposal, seeded rows emit the
    plain decoder's tokens), writes the drafter context of every row, and drafts the next block of every
    speculating sequence (batched drafter: <= MAX_USERS sequences per pass).  State is position-indexed everywhere, so
    commit = advance each sequence's position by its accepted rows + 1.

        dec = BatchDecoder(model, taps=True, extra_buckets=SPEC_BUCKETS)
        bs = BatchSpec(model, dec)            # before any capture (persistent tensors, drafter)
        bs.compile(); dec.capture(); bs.capture()
        logits = bs.prefill(ids, start, slot, page_row)      # target prefill + drafter context of the chunk rows
        toks = bs.decode(rows)                # rows: vLLM's decode rows (BatchRow, one per sequence) -> one token each
        lg = bs.decode_logits(rows)           # host-sampling step (logprobs, ...): plain, queues dropped

    vLLM asks one token per sequence per step: decode() answers from each sequence's queue and runs a device step
    only when some sequence's queue is empty; that step also cycles every other speculating sequence whose queue is
    shorter than QCAP (the sequences cycle in lockstep: a step's fixed cost is shared by all of them), so the device
    state runs ahead of vLLM by up to QCAP + 3 tokens (+ the drafter block of 8): the KV pool blocks of those
    positions come from vLLM's lookahead allocation (`lookahead` = the plugin's TT_DECODE_LOOKAHEAD: a decode step at
    position pos has the blocks of positions < pos + 1 + lookahead; the page row's columns past them are stale ids of
    other requests, never written); a sequence whose blocks do not cover a cycle decodes plainly.
    Policy per active-sequence count B: "spec" (every eligible sequence speculates, m drafts each from its decayed
    acceptance) or "plain" (no drafts; rows of one token), chosen from the measured served throughput of each mode at
    this B (decode-step wall clock incl. vLLM's host time: EWMA per window of WINDOW steps; explores the other mode
    one window in EXPLORE), so batched speculation does not stay on where it is slower.  Any sequence that cannot
    speculate (penalized sampling, a host-sampled step, near max_seq_len) makes the step plain: one plain row per
    step would pay the step's fixed cost every step anyway."""

    QCAP = 4
    MAX_USERS = 4
    M_OPTIONS = (1, 2, 3)
    # relative cost of one more verify row of a sequence at B sequences, ROW_COST[0] + ROW_COST[1] B (m choice:
    # E(m) / (1 + cost m); a row costs ~2-3 ms at any B, a sequence's share of the step falls with B)
    ROW_COST = (0.12, 0.06)
    WINDOW = 32         # decode steps per throughput sample
    WARM = 8            # steps after a mode / batch change that are not sampled (queues and drafts settling)
    EXPLORE = 16        # windows of the better mode per window of the other
    PRIOR_SPEC_MAX = 2  # an unmeasured batch size starts speculating up to this many sequences (bench_batch_spec)
    LOG_EVERY = 2000    # decode steps between stats lines

    def __init__(self, model, dec, drafter: Drafter | None = None, log=print, modes: str = "auto",
                 lookahead: int | None = None):
        if dec.feat is None:
            raise ValueError("BatchSpec needs BatchDecoder(taps=True)")
        self.m, self.dec, self.log = model, dec, log
        cfg, d = model.cfg, model.device
        self.dr = drafter or Drafter(d, cfg, model.cache, model.max_seq_len, hf_dir=model.src.hf_dir,
                                     source=model.src, slots=model.max_num_seqs, kv_blocks=model.kv_blocks)
        self.S = model.max_num_seqs
        if model.snaps is not None:
            # prefix caching (NOTES-prefix.md): the drafter's local-layer rings are per-slot running state like the
            # target's sliding rings, so every snapshot carries the slot's rings and restore_slot puts them back (a
            # resumed request's drafter context is then exactly the publisher's).  Registered here, in the adapter's
            # constructor: the pool tensors are allocated now, before any trace capture (PrefixSnapshots.register).
            # The drafter's full-context layer (1) is a pool addressed by vLLM block ids like the target's
            # full-attention pool: the resumed request shares the publisher's blocks of [0, S), which the publisher's
            # prefill (context_fill) wrote and nothing rewrites, so it needs no snapshot.  Everything else is
            # per-step input (ctx_rows_inp / users_inp, host-written) or the per-sequence queues reset by the slot
            # listener.
            model.snaps.register(self.dr.slot_tensors())
        self.lookahead = int(os.environ.get("TT_DECODE_LOOKAHEAD", "0")) if lookahead is None else lookahead
        self.seqs = [_Seq() for _ in range(self.S)]
        self.modes = modes  # "auto" | "spec" | "plain" (tests / benches: force a mode)
        self.traces = {}
        self.rate = {}      # (B, mode) -> EWMA tokens / s of decode steps
        self._win = None    # [B, mode, steps, tokens, seconds]
        self._wins = {}     # B -> windows run in the exploit mode since the last exploration
        self._t_last = None
        self.stats = {"steps": 0, "device_steps": 0, "spec_steps": 0, "rows": 0, "verify_rows": 0, "emitted": 0,
                      "accepted_hist": [0] * (max(self.M_OPTIONS) + 1), "drafts": 0, "plain_fallback": 0,
                      "queue_drops": 0, "cut": 0, "cap_limited": 0, "verify_s": 0.0, "accept_s": 0.0, "draft_s": 0.0}
        model.slot_listeners.append(self._slot_reset)

    # ------------------------------------------------------------------ programs / traces
    def _ctx(self, N: int):
        self.dr.context_rows(self.dec.feat, N)

    def _draft(self, U: int, steps: int):
        self.dr.draft_users(self.dec.feat, U, steps)

    def _programs(self):
        out = [(("ctx", N), lambda N=N: self._ctx(N)) for N in self.dec.buckets]
        out += [(("draft", U, s), lambda U=U, s=s: self._draft(U, s))
                for U in range(1, self.MAX_USERS + 1) for s in self.M_OPTIONS]
        return out

    def _dummy_inputs(self):
        W = self.m.max_seq_len // 64
        zero = torch.zeros(W, dtype=torch.int32)
        for N in self.dec.buckets:
            self.dr.ctx_rows_inp[N].write([0] * N, list(range(N)), [zero] * N)
        users = [(0, 64 * u, 0, self._block_input(0), u, zero) for u in range(self.MAX_USERS)]
        self.dr.users_inp.write(users)

    def compile(self):
        """Run every drafter program once untraced (context of every bucket, draft of every (users, steps), the prompt
        context of every prefill chunk length): before ANY trace capture of the device (model.py trace safety).  The
        compile writes into slot 0's drafter state and the pool's null block 0 only."""
        self._dummy_inputs()
        for _, fn in self._programs():
            fn()
        H = self.m.cfg.hidden
        zero = torch.zeros(self.m.max_seq_len // 64, dtype=torch.int32)
        for Tp in self.m.warmup_shapes():
            fz = [ttnn.zeros([1, 1, Tp, H], dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.m.device,
                             memory_config=DRAM) for _ in range(4)]
            self.dr.context_fill(fz, 0, slot=0, page_row=zero)
            free(*fz)
        ttnn.synchronize_device(self.m.device)

    def capture(self):
        """Capture the drafter traces (after compile() and the BatchDecoder's capture) and time them."""
        self.release()
        d = self.m.device
        self._dummy_inputs()
        for key, fn in self._programs():
            tid = ttnn.begin_trace_capture(d, cq_id=0)
            fn()
            ttnn.end_trace_capture(d, tid, cq_id=0)
            self.traces[key] = tid
        ttnn.synchronize_device(d)
        ms = {}
        for key in (("ctx", max(self.dec.buckets)), ("draft", self.MAX_USERS, max(self.M_OPTIONS)), ("draft", 1, 1)):
            ttnn.execute_trace(d, self.traces[key], cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(3):
                ttnn.execute_trace(d, self.traces[key], cq_id=0, blocking=False)
            ttnn.synchronize_device(d)
            ms[str(key)] = round((time.perf_counter() - t0) / 3 * 1e3, 3)
        self.log(f"batched spec: drafter traces captured, ms {ms}")
        for s in self.seqs:
            s.reset()

    def release(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.m.device, tid)
        self.traces.clear()

    def _run(self, key, fn):
        tid = self.traces.get(key)
        if tid is None:
            fn()
        else:
            ttnn.execute_trace(self.m.device, tid, cq_id=0, blocking=False)

    # ------------------------------------------------------------------ sequences
    def _slot_reset(self, s: int):
        self.seqs[s].reset()

    def prefill(self, token_ids, start: int, slot: int, page_row, at=None) -> torch.Tensor:
        """Batched prefill chunk of slot `slot` (model.prefill) + the drafter context of its rows (context_fill per
        chunk; chunk starts must be 64-aligned: the batched scheduler's chunks start at multiples of 2048).  at(pos):
        model.prefill's chunk-boundary hook (prefix caching publishes the slot's state, drafter rings included: each
        chunk's context_fill runs before the next boundary's at())."""
        feats = {}

        def tap(app, layer, p, h):
            if app in FEATURE_APPS:
                feats[app] = ttnn.clone(h)

        def on_chunk(pos, n, out):
            fl = [feats.pop(a) for a in FEATURE_APPS]
            if pos % 64 == 0:
                self.dr.context_fill(fl + [out], pos, slot=slot, page_row=page_row)
            free(*fl)

        self.seqs[slot].reset()
        return self.m.prefill(token_ids, start, slot=slot, page_row=page_row, at=at, tap=tap, on_chunk=on_chunk)

    def _capacity(self, r: BatchRow) -> int:
        """Positions < this have KV blocks in the row's page row: vLLM allocated the blocks of the scheduled token
        and `lookahead` more (clamped to max_seq_len); later columns hold stale block ids (vLLM never clears a block
        table row), the first null block (0) ends the row early."""
        row = torch.as_tensor(r.pages).reshape(-1)
        z = (row == 0).nonzero()
        filled = 64 * (int(z[0]) if z.numel() else row.numel())
        return min(filled, -(-(int(r.pos) + 1 + self.lookahead) // 64) * 64, self.m.max_seq_len - 1)

    def _eligible(self, r: BatchRow) -> bool:
        return r.sampling is None or not r.sampling.penalized

    def _pick_m(self, seq: _Seq, room: int, B: int) -> int:
        """Drafts of the sequence's next verify: argmax E(m) / (1 + cost m) under the geometric acceptance model
        (SpecDecoder.pick), cost = ROW_COST at B sequences, m <= room."""
        a = seq.acc[0] / (seq.acc[0] + seq.acc[1])
        cost = self.ROW_COST[0] + self.ROW_COST[1] * B
        best, val = 0, 0.0
        for m in self.M_OPTIONS:
            if m > room:
                break
            v = sum(a ** i for i in range(m + 1)) / (1.0 + cost * m)
            if v > val:
                best, val = m, v
        return best

    # ------------------------------------------------------------------ policy
    def _mode(self, B: int) -> str:
        if self.modes != "auto":
            return self.modes
        if B * 2 > self.dec.buckets[-1]:
            return "plain"  # verify groups of 2+ rows for every sequence do not fit the 16-row step
        sp, pl = self.rate.get((B, "spec")), self.rate.get((B, "plain"))
        if sp is None and pl is None:
            return "spec" if B <= self.PRIOR_SPEC_MAX else "plain"
        if sp is None or pl is None:  # explore the unmeasured mode once
            return "spec" if sp is None else "plain"
        best, other = ("spec", "plain") if sp >= pl else ("plain", "spec")
        n = self._wins.get(B, 0)
        return other if n >= self.EXPLORE else best

    def _account(self, B: int, mode: str, tokens: int):
        """Decode-step wall clock (from the previous decode call, prefill turns excluded by note_prefill()) ->
        throughput windows per (B, mode)."""
        now = time.perf_counter()
        last, self._t_last = self._t_last, now
        if last is None:
            return
        w = self._win
        if w is None or w[0] != B or w[1] != mode:
            w = self._win = [B, mode, -self.WARM, 0, 0.0]
        w[2] += 1
        if w[2] <= 0:
            return
        w[3] += tokens
        w[4] += now - last
        if w[2] >= self.WINDOW:
            r = w[3] / max(w[4], 1e-9)
            old = self.rate.get((B, mode))
            self.rate[(B, mode)] = r if old is None else 0.7 * old + 0.3 * r
            sp, pl = self.rate.get((B, "spec")), self.rate.get((B, "plain"))
            best = "spec" if pl is None or (sp is not None and sp >= pl) else "plain"
            self._wins[B] = 0 if mode != best else self._wins.get(B, 0) + 1
            self._win = None

    def note_prefill(self):
        """A prefill turn ran: the next decode step's wall clock includes it (not a decode sample)."""
        self._t_last = None

    # ------------------------------------------------------------------ steps
    def decode(self, rows) -> list[int]:
        """vLLM's decode step: rows = one BatchRow per running sequence (slot, its input token / position as vLLM
        sees them, page row, sampling) -> the next token of each (from the sequence's queue or a device step)."""
        B = len(rows)
        st = self.stats
        st["steps"] += 1
        for r in rows:
            seq = self.seqs[r.slot]
            if seq.queue:
                q = seq.queue[0]
                if (q[0], q[1]) != (int(r.pos), int(r.token)):
                    raise RuntimeError(f"batched spec: slot {r.slot} queue holds ({q[1]}, {q[0]}), vLLM sent "
                                       f"({r.token}, {r.pos})")
            elif seq.front != int(r.pos) or seq.anchor != int(r.token):
                seq.front, seq.anchor, seq.drafts = int(r.pos), int(r.token), []
        need = [r for r in rows if not self.seqs[r.slot].queue]
        all_ok = all(self._eligible(r) for r in rows)
        mode = self._mode(B) if all_ok else "plain"
        if not all_ok:
            st["plain_fallback"] += 1
        if need:
            if mode == "plain":
                for r in rows:  # spec -> plain: drop pending drafts (queued tokens stay valid)
                    self.seqs[r.slot].drafts = []
                self._step([r for r in need], mode)
            else:
                part = [r for r in rows if len(self.seqs[r.slot].queue) < self.QCAP]
                self._step(part, mode)
        out = []
        for r in rows:
            q = self.seqs[r.slot].queue.pop(0)
            out.append(q[2])
        self._account(B, mode, B)
        if st["steps"] % self.LOG_EVERY == 0:
            self.log(f"batched spec: {st}, tok/s per (sequences, mode) "
                     f"{({f'{b}:{md}': round(v, 1) for (b, md), v in sorted(self.rate.items())})}")
        return out

    def _step(self, part, mode: str):
        """One device step over the participating sequences: verify groups (drafts pending) or plain rows, the
        drafter context of every row, acceptance, then the next drafts (mode "spec")."""
        st = self.stats
        st["device_steps"] += 1
        rows, groups = [], []
        cap = {}
        for r in part:
            seq = self.seqs[r.slot]
            cap[r.slot] = self._capacity(r)
            toks = [seq.anchor] + seq.drafts
            room = cap[r.slot] - 1 - seq.front  # rows must stay below the covered positions
            if len(toks) - 1 > room or self.dec.buckets[-1] - len(rows) - (len(part) - len(groups) - 1) < len(toks):
                st["cut"] += len(toks) > 1
                toks = toks[:1]
            groups.append((r, len(rows), len(toks)))
            for j, t in enumerate(toks):
                rows.append(BatchRow(r.slot, t, seq.front + j, r.pages, r.sampling, owner=r.owner,
                                     outputs=r.outputs, prompt=r.prompt))
        if any(R > 1 for _, _, R in groups):
            st["spec_steps"] += 1
        st["rows"] += len(rows)
        st["verify_rows"] += sum(R - 1 for _, _, R in groups)

        def ctx(Rb):
            src = list(range(len(rows))) + [0] * (Rb - len(rows))
            self.dr.ctx_rows_inp[Rb].write([rows[i].slot for i in src], [rows[i].pos for i in src],
                                           [rows[i].pages for i in src])
            self._run(("ctx", Rb), lambda: self._ctx(Rb))

        # one row per sequence: the plain batched step (penalized rows commit their sampler history there); verify
        # groups: the verify step (no history commits; penalized sequences never speculate)
        t0 = time.perf_counter()
        g = (self.dec.step(rows, after=ctx) if len(rows) == len(groups) else self.dec.verify(rows, after=ctx))
        t1 = time.perf_counter()
        drafting = []
        for r, b0, R in groups:
            seq = self.seqs[r.slot]
            d = seq.drafts[: R - 1]
            k = 0
            while k < R - 1 and d[k] == g[b0 + k]:
                k += 1
            toks_in = [seq.anchor] + d[:k]
            toks_out = d[:k] + [g[b0 + k]]
            seq.queue += [(seq.front + j, toks_in[j], toks_out[j]) for j in range(k + 1)]
            st["emitted"] += k + 1
            if R > 1:
                st["accepted_hist"][k] += 1
                dec_ = SpecDecoder.ACC_DECAY
                seq.acc = [dec_ * seq.acc[0] + k, dec_ * seq.acc[1] + (k < R - 1)]
            seq.front += k + 1
            seq.anchor = toks_out[-1]
            seq.drafts = []
            if mode == "spec":
                room = min(cap[r.slot] - 8 - seq.front, self.m.max_seq_len - DBLOCK - 1 - seq.front)
                m = self._pick_m(seq, room, len(part))
                if m > 0:
                    drafting.append((r, b0 + k, m))
                elif room < 1:
                    st["cap_limited"] += 1  # the page row does not cover the next block (no vLLM lookahead?)
        t2 = time.perf_counter()
        for i in range(0, len(drafting), self.MAX_USERS):
            self._draft_pass(drafting[i:i + self.MAX_USERS])
        st["verify_s"] += t1 - t0
        st["accept_s"] += t2 - t1
        st["draft_s"] += time.perf_counter() - t2

    def _draft_pass(self, users):
        """Drafts for <= MAX_USERS sequences: block at each anchor (feature row sel = the step's row of the accepted
        token) -> seq.drafts."""
        steps = max(m for _, _, m in users)
        inp = []
        for r, sel, m in users:
            seq = self.seqs[r.slot]
            inp.append((r.slot, seq.front, seq.anchor, self._block_input(seq.anchor), sel, r.pages))
        self.dr.users_inp.write(inp)
        U = len(users)
        self._run(("draft", U, steps), lambda: self._draft(U, steps))
        drafts = self.dr.read_drafts_users(U, steps)
        for (r, _, m), dl in zip(users, drafts):
            self.seqs[r.slot].drafts = list(dl[:m])
        self.stats["drafts"] += U

    def decode_logits(self, rows) -> torch.Tensor:
        """Host-sampling step (some row needs vLLM's host sampler): every row decodes plainly at vLLM's (token,
        position) (state is position-indexed: recomputing a queued position writes the same values), queues and
        drafts are dropped, the drafter context of the rows is written; -> FP32 logits [B, unpadded_vocab]."""
        for r in rows:
            seq = self.seqs[r.slot]
            if seq.queue:
                self.stats["queue_drops"] += 1
            seq.queue, seq.drafts = [], []
            seq.front, seq.anchor = int(r.pos) + 1, None  # the host's token arrives with the next step
        lg = self.dec.step_logits(rows)
        Rb = self.dec.bucket(len(rows))
        src = list(range(len(rows))) + [0] * (Rb - len(rows))
        self.dr.ctx_rows_inp[Rb].write([rows[i].slot for i in src], [rows[i].pos for i in src],
                                       [rows[i].pages for i in src])
        self._run(("ctx", Rb), lambda: self._ctx(Rb))
        self._t_last = None
        return lg

    def _block_input(self, a: int) -> torch.Tensor:
        """Drafter block input [8, H] BF16 (SpecDecoder._block_input)."""
        m = self.m
        x0 = self.dr.mask_embed.expand(DBLOCK, m.cfg.hidden).clone()
        e = m.embed.table[a].float()
        x0[0] = (e * torch.rsqrt(e.pow(2).mean() + m.cfg.eps) * self.dr.embed_norm_w).to(torch.bfloat16)
        return x0
