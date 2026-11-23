# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Self-feeding traced decode for Inkling-2-Mini: the device picks the next token, feeds it to the next step and
advances the position / attention tables itself; the host only replays traces and reads token ids.

    dec = TokenDecoder(model)              # after Inkling2Model(...), before any trace capture (allocates persistent
                                           # tensors: device embedding table, token / log rows, tau table, sampler)
    dec.capture()                          # one main trace per decode bucket + small select/advance tail traces
    logits = model.prefill(prompt)         # host FP32 logits of the last prompt token (unchanged)
    tok = int(logits.argmax())             # (or the host / vLLM sampler for the first token)
    nxt = dec.decode(tok, pos=len(prompt)) # one step: token at position pos in -> next token id (greedy)
    nxt = dec.decode(nxt, pos + 1)         # continues on the device state without host writes
    toks = dec.generate(tok, pos, n)       # n steps back to back (no host round trip between steps), n <= log_pages
    dec.set_sampling(RowSampling(...), output_ids=[...], prompt_ids=[...])  # sampled steps (None = greedy)
    lg = dec.step_logits(tok, pos)         # host-sampling fallback: host FP32 logits of the step, no device choice

Per step on the device (bucket b = the smallest decode bucket > pos):
  main trace[b]:  x = embedding(tok_row, table) (ttnn.embedding, BF16 table on the device; embed="host": the model's
                  decode input row, uploaded by the host after each token read) -> model.decode(x, cur_pos, ids=tok)
                  -> FP32 softcapped logits (the model's own decode: bitwise the host-fed path); its fused LM head
                  also writes the greedy token (argmax) to tok_row word 0;
  tail trace[b, mode]: greedy: nothing more; sample | full | full_reject (fast/sampling.kind_of: top-K rows, full-
                  vocabulary rows, + the rejection launches of top-p / top-k > 64): DeviceSampler -> tok_row word 0
                  (overwrites) + sel;
                  then DecodeAdvance(tables[b]): cur_pos / ring_pos / ring_idx / full_idx window / tau -> pos + 1, and
                  log[(pos + 1) % log_pages] = [token, flags, n, pos + 1];
  rm trace[b]:    (step_logits only) the logits untilized on the device for a fast host read.
The host mirrors the position (self.pos = position of the token held in tok_row); a different (token, pos) than the
device holds (a new request, a host-sampled token, a bucket change, a host write of the tables by another path) is
written from the host first (`seed`).

Greedy tokens are the argmax of the same FP32 logits the host path reads (fast/head.py; ties -> lowest id, as
torch.argmax), so greedy decoding is identical to `model.decode_traced` + host argmax (tests/test_fast_loop.py).
Sampled tokens equal fast/sampling.host_sample (top-K rows: vLLM's sampler with the device's Philox draw) or
host_sample_full (full-vocabulary rows: the integer-weight model) on those logits.
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from ..attention import _tau
from ..common import DRAM, free
from .sampling import P_HOST, P_SAMPLE, R_INEXACT, DeviceSampler, RowSampling, host_sample, kind_of
from .token_ops import CachedProgram

KERNELS = Path(__file__).resolve().parent.parent / "kernels" / "fast"
LOG_PAGES = 1024


def _u32(device, rows, cols, dtype=ttnn.uint32):
    return ttnn.from_torch(torch.zeros(rows, cols, dtype=torch.int32), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=device, memory_config=DRAM)


class DecodeAdvance:
    """One launch: the attention decode tables `tables` (AttnDecodeTables) from position p (its cur_pos) to p + 1, as
    tables.update(p + 1) would write them (the tables must hold p's values), + the token log row."""

    def __init__(self, device, tables, tau_table, log_pages: int):
        self.device, self.tables, self.tau_table, self.log_pages = device, tables, tau_table, log_pages
        self._prog = CachedProgram()

    def _build(self, sel, log):
        t = self.tables
        core = ttnn.CoreCoord(0, 0)
        cs = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
        named = [("full_len", t.full_len), ("max_seq", t.max_seq_len), ("log_pages", self.log_pages),
                 ("ring_page", int(t.ring_idx.padded_shape[-1]) * 4),
                 ("full_page", int(t.full_idx.padded_shape[-1]) * 4)]
        common = [int(x.buffer_address()) for x in (t.cur_pos, t.ring_pos, t.ring_idx, t.full_idx, t.tau,
                                                    self.tau_table, sel, log)]
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "decode_advance.cpp"), core_ranges=cs,
            brisc_named_compile_time_args=named, ncrisc_named_compile_time_args=named,
            brisc_common_runtime_args=common, ncrisc_common_runtime_args=common,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(brisc_args=[(core, [0])],
                                                                          ncrisc_args=[(core, [0])]),
        )
        cbs = [ttnn.CBDescriptor(total_size=16384, core_ranges=cs,
                                 format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, 16384)])]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cs, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=sems)

    def __call__(self, sel, log):
        t = self.tables
        tensors = [t.cur_pos, t.ring_pos, t.ring_idx, t.full_idx, t.tau, self.tau_table, sel, log]
        self._prog.run(tensors, lambda: self._build(sel, log))


def tau_table(cfg, max_seq_len: int) -> torch.Tensor:
    """[1, max_seq_len] fp32: tau(pos) exactly as AttnDecodeTables computes it (one scalar evaluation per position
    where tau != 1; tau == 1 exactly below log_scaling_n_floor - 1)."""
    out = torch.ones(1, max_seq_len, dtype=torch.float32)
    start = max(0, int(cfg.log_scaling_n_floor) - 1)
    for p in range(start, max_seq_len):
        out[0, p] = _tau(torch.tensor([p]), cfg).reshape(())
    return out


class TokenDecoder:
    """See the module docstring.  model: Inkling2Model (its decode buckets / tables / reset are used).
    sampler: also build the on-device sampler (sampled requests); log_pages: token log rows (generate() limit).
    embed: "device" (BF16 table on the device, 0.86 GiB: the token never leaves the device between steps) | "host"
    (the model's host table: each step's embedding row is uploaded into the model's decode input after the token is
    read; everything else stays on the device; generate() is then a loop of decode())."""

    def __init__(self, model, *, sampler: bool = True, log_pages: int = LOG_PAGES, embed: str = "device"):
        self.model = m = model
        dev, cfg = m.device, m.cfg
        self.device, self.cfg = dev, cfg
        self.V = cfg.unpadded_vocab
        if m._traces:
            raise RuntimeError("TokenDecoder must be built before any decode trace capture (persistent tensors)")
        if embed not in ("device", "host"):
            raise ValueError(f"embed must be 'device' or 'host', got {embed!r}")
        self.embed_mode = embed
        # BF16 embedding table on the device ([vocab, H] ROW_MAJOR, ttnn.embedding's weight layout): 0.93 GB
        self.table = (m.cache.get("embed_rm", lambda: m.embed.table, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                      if embed == "device" else None)
        self.tok = _u32(dev, 1, 32)      # embedding index row: word 0 = the token at position self.pos
        self.sel = _u32(dev, 1, 32)      # sampler result [token, flags, n, 0]
        self.log = _u32(dev, log_pages, 16)
        self.log_pages = log_pages
        self.tau = ttnn.from_torch(tau_table(cfg, m.max_seq_len), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=dev, memory_config=DRAM)
        self.sampler = DeviceSampler(dev, cfg.padded_unembed, self.V) if sampler else None
        self.advance = {b: DecodeAdvance(dev, m.tables[b], self.tau, log_pages) for b in m.buckets}
        self._main = {}   # bucket -> (trace id, device logits)
        self._tail = {}   # (bucket, mode) -> trace id
        self._rm = {}     # bucket -> (trace id, ROW_MAJOR copy of the main trace's logits)
        self.pos = None   # position of the token held by self.tok (host mirror); None = unknown
        self._bucket = None  # bucket whose tables hold self.pos
        self._token = self._written = None  # token held by self.tok; position last written to the tables by us
        self.spec = None  # RowSampling of sampled steps; None = greedy
        self._owner, self._outputs, self._prompt = None, [], []

    # ------------------------------------------------------------------ device programs (untraced forms)
    def _main_ops(self, b: int):
        m = self.model
        if self.table is None:  # host mode: the model's persistent decode input (row 0 = the embedding row)
            return m.decode(m._emb, m.tables[b].cur_pos, ids=self.tok)
        x = ttnn.embedding(self.tok, self.table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
        x = ttnn.reshape(x, [1, 1, 32, self.cfg.hidden])
        lg = m.decode(x, m.tables[b].cur_pos, ids=self.tok)  # the LM head writes the greedy token to tok
        ttnn.deallocate(x)
        return lg

    def _upload_embedding(self, token: int):
        """Host mode: the next step's input row (the model's decode input tensor)."""
        ttnn.copy_host_to_device_tensor(self.model.decode_inputs_host(int(token)), self.model._emb)

    def _tail_ops(self, b: int, mode: str, lg):
        if mode == "greedy":  # the main trace's LM head wrote the greedy token to tok
            self.advance[b](self.tok, self.log)
        else:  # sample (top-K rows) | full | full_reject (full-vocabulary rows): fast/sampling.kind_of
            self.sampler.sample(lg, self.sel, self.tok, self.model.tables[b].cur_pos, kind=mode)
            self.advance[b](self.sel, self.log)

    @property
    def modes(self):
        return ("greedy", "sample", "full", "full_reject") if self.sampler is not None else ("greedy",)

    # ------------------------------------------------------------------ capture
    def capture(self, buckets: list[int] | None = None):
        """Compile + capture every bucket's main trace and tail traces (needs the device's trace region; the model's
        own decode traces may coexist).  Each compile run decodes position 0 (writes cache position 0, shifts conv
        states): call between sequences; ends with model.reset().  Re-capturing releases the previous traces."""
        m = self.model
        self.release()
        if self.sampler is not None:
            self.sampler.set_params([RowSampling(temperature=1.0, top_k=20, top_p=0.95, seed=1).words()])
            self.sampler.invalidate()
        for b in buckets or m.buckets:
            tb = m._use_bucket(b)
            for mode in self.modes:  # compile everything once
                tb.update(0)
                lg = self._main_ops(b)
                self._tail_ops(b, mode, lg)
                ttnn.deallocate(lg)
            tb.update(0)
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            lg = self._main_ops(b)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self._main[b] = (tid, lg)
            for mode in self.modes:
                tid = ttnn.begin_trace_capture(self.device, cq_id=0)
                self._tail_ops(b, mode, lg)
                ttnn.end_trace_capture(self.device, tid, cq_id=0)
                self._tail[(b, mode)] = tid
            # host-sampling fallback: the logits untilized on the device (a TILE read untilizes on the host: ~5 ms)
            ttnn.deallocate(ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM))
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self._rm[b] = (tid, rm)
        ttnn.synchronize_device(self.device)
        m.reset()
        self.pos = self._bucket = None
        if self.sampler is not None:
            self.sampler.invalidate()

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

    # ------------------------------------------------------------------ host state
    def seed(self, token: int, pos: int):
        """Host write: the device's next step decodes `token` at position `pos` (token row + pos's bucket tables)."""
        b = self.model.bucket(pos)
        row = torch.zeros(1, 32, dtype=torch.int32)
        row[0, 0] = int(token)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(row, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.tok)
        if self.table is None:
            self._upload_embedding(token)
        self.model._use_bucket(b).update(pos)
        self.pos, self._bucket, self._token, self._written = pos, b, int(token), pos

    def invalidate(self):
        """Forget the device state (after reset / prefill): the next decode() seeds from the host."""
        self.pos = self._bucket = None

    def set_sampling(self, spec: RowSampling | None, *, owner=None, output_ids=(), prompt_ids=()):
        """Sampling of the following steps: None / greedy spec without penalties -> device argmax; else the device
        sampler with `spec` (history = output_ids counts + prompt flags; kept current on the device while steps run).
        owner: request identity (history reuse across calls)."""
        if spec is not None and spec.greedy and not spec.penalized:
            spec = None
        if spec is not None and self.sampler is None:
            raise RuntimeError("TokenDecoder built without the device sampler")
        self.spec = spec
        self._owner, self._outputs, self._prompt = owner, list(output_ids), list(prompt_ids)
        if spec is not None:
            self.sampler.set_params([spec.words()])
            if spec.penalized:
                self.sampler.ensure_history(0, owner, self._outputs, self._prompt, spec.uses_prompt)

    # ------------------------------------------------------------------ steps
    def _replay(self):
        """Enqueue one step at self.pos (non-blocking); the device then holds the next token at self.pos + 1."""
        b = self.model.bucket(self.pos)
        if b != self._bucket:  # entering a larger bucket: its tables are written from the host once
            self.model._use_bucket(b).update(self.pos)
            self._bucket, self._written = b, self.pos
        if b not in self._main:
            raise RuntimeError(f"no decode trace for bucket {b}: TokenDecoder.capture()")
        mode = kind_of(self.spec)
        ttnn.execute_trace(self.device, self._main[b][0], cq_id=0, blocking=False)
        ttnn.execute_trace(self.device, self._tail[(b, mode)], cq_id=0, blocking=False)
        self.pos += 1
        if self.pos >= self.model.tables[b].full_len:
            self._bucket = None  # the advance stops at the bucket end: the next bucket is seeded from the host

    def _ensure(self, token: int, pos: int):
        # tables[b].pos is the last host-written position: another writer (the host-fed decode path) means stale state
        stale = self._bucket is None or self.model.tables[self._bucket].pos != self._written
        if stale or self.pos != pos or self._token != int(token):
            self.seed(token, pos)

    def decode(self, token: int, pos: int) -> int:
        """Decode `token` at position `pos` -> the next token (greedy, or sampled with set_sampling's spec)."""
        if pos + 1 >= self.model.max_seq_len:
            raise ValueError(f"position {pos + 1} reaches max_seq_len {self.model.max_seq_len}")
        self._ensure(token, pos)
        self._replay()
        if self.spec is None:
            nxt = int(ttnn.to_torch(self.tok).reshape(-1)[0])
        else:
            sel = ttnn.to_torch(self.sel).reshape(-1)
            nxt = int(sel[0])
            inexact = bool(int(sel[1]) & R_INEXACT)
            if inexact:  # truncated top-k tie group / 4 rejected full-vocabulary draws: exact host redraw
                nxt = host_sample(self.logits(), self.spec, self._outputs, self._prompt, pos + 1,
                                  P_HOST if self.spec.full else P_SAMPLE)
                row = torch.zeros(1, 32, dtype=torch.int32)
                row[0, 0] = nxt
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(row, dtype=ttnn.uint32,
                                                                layout=ttnn.ROW_MAJOR_LAYOUT), self.tok)
            self._outputs.append(nxt)
            if self.spec.penalized:
                if inexact:  # the device did not commit the token: rewrite the history
                    self.sampler.ensure_history(0, self._owner, self._outputs, self._prompt, self.spec.uses_prompt)
                else:
                    self.sampler.note_committed(0, self._owner, len(self._outputs))
        if self.table is None:
            self._upload_embedding(nxt)
        self._token = nxt
        return nxt

    def step_logits(self, token: int, pos: int) -> torch.Tensor:
        """Host-sampling fallback (logprobs, structured output, ...): decode `token` at `pos` and return the host FP32
        logits [unpadded_vocab] without choosing a token on the device (the next call seeds from the host)."""
        self.seed(token, pos)
        b = self.model.bucket(pos)
        ttnn.execute_trace(self.device, self._main[b][0], cq_id=0, blocking=False)
        ttnn.execute_trace(self.device, self._rm[b][0], cq_id=0, blocking=False)
        self.pos = self._bucket = None
        return ttnn.to_torch(self._rm[b][1]).float().reshape(-1)[: self.V]

    def generate(self, token: int, pos: int, n: int) -> list[int]:
        """n steps back to back from (token, pos) with one read at the end -> the n generated tokens (positions
        pos + 1 .. pos + n).  Greedy, or sampled when no step can need a host redraw (checked: raises if one did)."""
        if not 1 <= n <= self.log_pages:
            raise ValueError(f"1 <= n <= log_pages ({self.log_pages})")
        if pos + n >= self.model.max_seq_len:
            raise ValueError(f"positions up to {pos + n} reach max_seq_len {self.model.max_seq_len}")
        if self.table is None:  # host embedding: every step needs the token on the host
            toks = []
            for i in range(n):
                token = self.decode(token, pos + i)
                toks.append(token)
            return toks
        self._ensure(token, pos)
        for _ in range(n):
            self._replay()
        log = ttnn.to_torch(self.log).reshape(self.log_pages, 16).to(torch.int64)
        rows = [(pos + 1 + i) % self.log_pages for i in range(n)]
        out = log[rows]
        if not torch.equal(out[:, 3], torch.arange(pos + 1, pos + n + 1)):
            raise RuntimeError("token log positions do not match the replayed steps")
        if bool((out[:, 1] & R_INEXACT).any()):
            raise RuntimeError("a sampled step needs a host redraw (truncated top-k tie group): use decode()")
        toks = out[:, 0].tolist()
        if self.spec is not None:
            self._outputs += toks
            if self.spec.penalized:
                self.sampler.note_committed(0, self._owner, len(self._outputs))
        self._token = toks[-1]
        return toks

    def logits(self) -> torch.Tensor:
        """FP32 logits [unpadded_vocab] of the last replayed step (host read; debugging / host fallbacks)."""
        b = self.model.bucket(self.pos - 1)
        return ttnn.to_torch(self._main[b][1]).float().reshape(-1)[: self.V]
