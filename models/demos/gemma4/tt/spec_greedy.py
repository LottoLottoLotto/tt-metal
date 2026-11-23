# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact greedy speculative decoding with the gemma-4 it-assistant drafter (single user, one device).

One traced cycle, replayed per iteration:

  1. seed: the drafter's recurrent hidden = row ``m`` of the previous cycle's verify hidden (gathered on
     device from a row-major copy; host writes ``m``), token ``c`` = the last committed token, which sits
     at position ``p`` and has not been through the target yet.
  2. draft: K assistant steps at the fixed position ``p`` (HF ``SinglePositionMultiTokenCandidateGenerator``
     contract: input cat(target_embed(tok), hidden), shared target KV of the last sliding / full layer),
     on-device argmax chained into the next step. The drafter attends the target KV rows this request's
     cycles wrote: [0, p] when the previous cycle verified a draft at ``p`` (rejected: HF's cache holds
     its row too), else [0, p - 1] (after prefill / a restart / all drafts accepted, row ``p`` holds prefill
     padding or another request's KV). RoPE stays at ``p``.
  3. verify: the target runs rows [c, d1..dK] at positions p..p+K through ``Gemma4Model.ttnn_rows_forward``,
     whose per-row logits are bit-identical to batch=1 decode, then on-device argmax per row.

Host: m = number of leading drafts equal to the target argmax of the previous row; emits d1..dm and the
target token t_m; next c = t_m at p+m+1, next seed row m. Every emitted token is a target argmax of a
row whose KV history equals plain greedy decode, so the stream is token-identical to non-speculative
greedy decode. Rejected rows' KV (positions > p+m) is overwritten by the next cycle before it is read.
The first cycle after prefill seeds the drafter with a zero hidden (it only costs acceptance). The drafter
never reads a KV row the sequence has not written, so its drafts (and with them sampled outputs, whose
path depends on the drafts) are a function of the sequence alone: a seeded request reproduces.

Confidence-gated draft length at long context (``GEMMA4_SPEC_GATE`` = "ctx:tau,..." breakpoints,
default ``GATE_SCHEDULE``; "0" disables): from the first breakpoint's position on, the cycle runs as K
one-step drafter traces and K verify traces (P = 2..K+1 rows). After each drafter step the host reads
the drafter's softmax max-probability of its proposed token and keeps drafting only while it is >= tau
(the last breakpoint <= position); the draft that falls below is still verified. Verify rows grow
expensive with context (full-attention SDPA per row: ~0.3 ms/row at 128 tokens, ~11 ms/row at 256K),
so dropping low-confidence drafts pays there; at short context the fused trace is cheaper. Emitted
tokens are still target argmaxes of rows whose KV history equals plain greedy decode, for every draft
length (every verify P keeps the batch=1 flash-decode split), so the stream is unchanged.

Lossless speculative sampling (random / penalized requests, device_sampling.py): the cycle always runs the
split traces, each followed by a sampler add-on trace. After drafter step i the add-on draws x_i from the
processed draft distribution q_i (vLLM's sampler on the drafter logits, penalties counting x_1..x_{i-1}) and
writes it over the step's argmax draft, so step i+1 and the verify rows consume x_i. After verify P = k+1
the add-on runs the rejection chain over p_0..p_k (penalties counting the drafts in front of each row):
accept x_i with probability min(1, p_{i-1}(x_i)/q_i(x_i)), at the first rejection emit a draw from
normalize(max(0, p_{i-1} - q_i)), after k acceptances the bonus from p_k. Gate: at the same thresholds,
drafting stops after the first step whose max q is below tau. The draft count only depends on drafter
outputs before the verification, so each emitted token is still distributed exactly as non-speculative
sampling given the emitted prefix (lossless); the gate only trades acceptance for verify rows.
"""

import os
import struct
import time

import torch

import ttnn
from models.demos.gemma4.tt.attention.decode import verify_sdpa_rows

# Breakpoints chosen by measured decode tok/s per context and threshold (p3d-pc, K=5): gating loses
# a little below ~4K tokens (verify rows ~0.3 ms each there), tau 0.85 is best at 8K-32K, and a
# stricter 0.95 at 64K-256K where each verify row costs 3-11 ms.
GATE_SCHEDULE = "4096:0.85,49152:0.95"


def _gate_schedule():
    spec = os.environ.get("GEMMA4_SPEC_GATE", GATE_SCHEDULE).strip()
    if spec in ("", "0"):
        return []
    return sorted((int(c), float(t)) for c, t in (x.split(":") for x in spec.split(",")))


CONF_W = 1024  # confidence reductions run on the logits row folded to [vocab / CONF_W, CONF_W]


class SpecGreedyDecoder:
    def __init__(self, model, assistant, mesh_device, kv_cache, page_tables, draft_len, page_tables_rep=None):
        self.model = model
        self.assistant = assistant
        self.mesh = mesh_device
        self.kv_cache = kv_cache
        # Persistent device page tables ({layer_type: tensor} or one tensor), owned and updated in
        # place by the caller before a replay.
        self.page_tables = page_tables
        self.K = draft_len
        self.P = draft_len + 1
        # One-launch verify SDPA (GEMMA4_VERIFY_SDPA=batched, rows = P): the row positions go in as one
        # [P] tensor (one host write per cycle instead of P) with the caller's persistent page tables
        # replicated to P rows (``page_tables_rep``, same structure, updated in place with them).
        self.batched = verify_sdpa_rows() == self.P
        if self.batched and page_tables_rep is None:
            raise ValueError("batched verify SDPA needs the replicated page tables (page_tables_rep)")
        self.page_tables_rep = page_tables_rep
        self.vocab = model.vocab_size
        self.hidden = model.hidden_size
        self.shared_kv = model.get_shared_kv_caches()
        self.trace_id = None
        self.out = None
        self.buf = None
        self.state = None  # staged (token c, position p) of the next cycle
        # Gated cycle (split traces): needs the batched verify for every P (its replicated page
        # tables are sliced to P rows inside each verify trace).
        self.gate = _gate_schedule() if self.batched and self.K > 1 else []
        self.steps = None  # [(trace_id, [draft, hidden, confidence, logits rows])] drafter step i = 1..K
        self.verifies = None  # {P: (trace_id, [vx, targets, both, logits rows])} for P = 2..K+1
        self.drafted = 0  # drafts verified since the last stats read (serving log)
        # Exact device sampler (device_sampling.DeviceSampler, attached by the runtime before any capture): the
        # sampled cycle replays the split traces, each followed by a sampler add-on trace (draft_addons[i - 1]
        # after drafter step i, verify_addons[P] after verify P).
        self.sampler = None
        self.point_draft = os.environ.get("GEMMA_SPEC_DRAFT_MODE", "sampled") == "point"
        self.draft_addons = None
        self.verify_addons = None

    # ---- host-side input staging -------------------------------------------------------------
    def _host(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout)

    def _host_inputs(self, tok, row, pos, row_p_written=False):
        """``d_pi``: position p (the sampler's RNG position base); ``d_kv``: the last target KV row the drafter
        attends, p when this sequence's previous cycle verified a draft at p (``row_p_written``), else p - 1."""
        P = self.P
        d_pu = torch.zeros(1, 32, dtype=torch.int64)
        d_pu[0, 0] = pos
        v_pu = torch.zeros(1, 32, dtype=torch.int64)
        v_pu[0, :P] = torch.arange(pos, pos + P)
        h = {
            "tok": self._host(torch.tensor([[tok]], dtype=torch.int64), ttnn.uint32),
            "row": self._host(torch.tensor([[row]], dtype=torch.int64), ttnn.uint32),
            "d_pu": self._host(d_pu, ttnn.uint32),
            "d_pi": self._host(torch.tensor([pos], dtype=torch.int32), ttnn.int32),
            "d_kv": self._host(torch.tensor([pos if row_p_written else pos - 1], dtype=torch.int32), ttnn.int32),
            "v_pu": self._host(v_pu, ttnn.uint32),
        }
        if self.batched:
            h["v_pall"] = self._host(torch.arange(pos, pos + P, dtype=torch.int32), ttnn.int32)
        else:
            for b in range(P):
                h[f"v_pi{b}"] = self._host(torch.tensor([pos + b], dtype=torch.int32), ttnn.int32)
        return h

    def _write(self, tok, row, pos, row_p_written=False):
        for k, v in self._host_inputs(tok, row, pos, row_p_written).items():
            ttnn.copy_host_to_device_tensor(v, self.buf[k])

    # ---- traced cycle ------------------------------------------------------------------------
    def _argmax_rows(self, logits, rows, keep=False):
        """[1, 1, rows] uint32 argmax of the first ``rows`` logits rows (torch.argmax ties: lowest id). ``keep``
        also returns the untilized bf16 rows [1, 1, rows, V] (read by the sampler add-on traces)."""
        rm = ttnn.untilize_with_unpadding(logits, output_tensor_end=[0, 0, rows - 1, self.vocab - 1])
        logits.deallocate(True)
        idx = ttnn.argmax(rm, dim=-1, keepdim=False)  # [1, 1, rows] uint32
        if keep:
            return idx, rm
        rm.deallocate(True)
        return idx

    def _body(self):
        b = self.buf
        h = ttnn.unsqueeze_to_4D(ttnn.embedding(b["row"], b["hsrc"], layout=ttnn.TILE_LAYOUT))  # [1,1,1,H]
        tok = b["tok"]
        drafts = []
        for _ in range(self.K):
            logits, h_next = self.assistant.step(tok, h, self.shared_kv, self.page_tables, b["d_pu"], b["d_kv"])
            h.deallocate(True)
            h = h_next
            tok = ttnn.reshape(self._argmax_rows(logits, 1), (1, 1))
            drafts.append(tok)
        h.deallocate(True)
        vx = ttnn.concat([b["tok"]] + drafts, dim=1) if drafts else ttnn.clone(b["tok"])  # [1, P] uint32
        for d in drafts:
            d.deallocate(True)
        if self.batched:
            vlogits, vhidden = self.model.ttnn_rows_forward(vx, b["v_pu"], None, self.page_tables, kv_cache=self.kv_cache,
                                                            batched=(b["v_pall"], self.page_tables_rep))
        else:
            pos_rows = [b[f"v_pi{i}"] for i in range(self.P)]
            vlogits, vhidden = self.model.ttnn_rows_forward(vx, b["v_pu"], pos_rows, self.page_tables, kv_cache=self.kv_cache)
        targets = self._argmax_rows(vlogits, self.P)
        hrm = ttnn.to_layout(vhidden, ttnn.ROW_MAJOR_LAYOUT)  # [1,1,P,H]
        vhidden.deallocate(True)
        ttnn.copy(ttnn.reshape(hrm, (self.P, self.hidden)), b["hsrc"])
        hrm.deallocate(True)
        # [1, 2P] = drafts row then target argmaxes: one device->host read per cycle.
        both = ttnn.concat([vx, ttnn.reshape(targets, (1, self.P))], dim=1)
        return vx, targets, both

    # ---- gated cycle: per-step drafter traces + per-P verify traces ----------------------------
    def _confidence(self, logits):
        """Softmax max-probability of the drafter's [1,1,1,V] logits row ([1,1,1,1] TILE). The row
        is folded to [V/W, W] so the reductions spread over cores (a [1, V] row reduces on one core)."""
        x = ttnn.reshape(logits, (1, 1, self.vocab // CONF_W, CONF_W))
        mx = ttnn.max(x, dim=-1, keepdim=True)
        top = ttnn.max(mx, dim=-2, keepdim=True)
        mx.deallocate(True)
        sh = ttnn.subtract(x, top)
        x.deallocate(True)
        top.deallocate(True)
        e = ttnn.exp(sh)
        sh.deallocate(True)
        s1 = ttnn.sum(e, dim=-1, keepdim=True)
        e.deallocate(True)
        s = ttnn.sum(s1, dim=-2, keepdim=True)
        s1.deallocate(True)
        conf = ttnn.reciprocal(s)
        s.deallocate(True)
        return conf

    def _step_body(self, prev):
        """One drafter step: seeded from row m of ``hsrc`` (first step) or the previous step's
        outputs. Returns [draft [1,1] uint32, next hidden, confidence, logits row [1,1,1,V] bf16]."""
        b = self.buf
        if prev is None:
            h = ttnn.unsqueeze_to_4D(ttnn.embedding(b["row"], b["hsrc"], layout=ttnn.TILE_LAYOUT))
            tok = b["tok"]
        else:
            tok, h = prev[0], prev[1]
        logits, h_next = self.assistant.step(tok, h, self.shared_kv, self.page_tables, b["d_pu"], b["d_kv"])
        if prev is None:
            h.deallocate(True)
        conf = self._confidence(logits)
        tok, rm = self._argmax_rows(logits, 1, keep=True)
        return [ttnn.reshape(tok, (1, 1)), h_next, conf, rm]

    def _verify_body(self, P, steps):
        """Verify rows [c, d1..d_{P-1}] (drafts from the step traces); hidden rows go to the first P
        rows of ``hsrc``. Returns [vx, targets, both [1, 2P]]."""
        b = self.buf
        vx = ttnn.concat([b["tok"]] + [s[0] for s in steps[: P - 1]], dim=1)
        full = P == self.P  # (a full-range slice may alias its input: use the persistent tensors)
        pos = b["v_pall"] if full else ttnn.slice(b["v_pall"], [0], [P])
        rep = self.page_tables_rep if full else {k: ttnn.slice(t, [0, 0], [P, t.shape[-1]]) for k, t in self.page_tables_rep.items()}
        vlogits, vhidden = self.model.ttnn_rows_forward(vx, b["v_pu"], None, self.page_tables, kv_cache=self.kv_cache,
                                                        batched=(pos, rep))
        if not full:
            pos.deallocate(True)
            for t in rep.values():
                t.deallocate(True)
        targets, rm = self._argmax_rows(vlogits, P, keep=True)
        hrm = ttnn.to_layout(vhidden, ttnn.ROW_MAJOR_LAYOUT)
        vhidden.deallocate(True)
        hp = ttnn.reshape(hrm, (P, self.hidden))
        if P < self.P:
            hp = ttnn.pad(hp, [(0, self.P - P), (0, 0)], 0.0)
            hrm.deallocate(True)
        ttnn.copy(hp, b["hsrc"])
        hp.deallocate(True)
        both = ttnn.concat([vx, ttnn.reshape(targets, (1, P))], dim=1)
        return [vx, targets, both, rm]

    # Sampler add-ons (exact device sampling): draft step i overwrites its own draft token (outs[0], read by
    # step i+1 and by the verify rows) with x_i ~ q_i and keeps q_i; verify P runs the rejection chain.
    def _draft_addon(self, i, outs):
        self.sampler.draft(outs[3], i, pos=self.buf["d_pi"], tokdst=outs[0], point_draft=self.point_draft)

    def _verify_addon(self, P, outs):
        self.sampler.verify_rows(outs[3], P, tokens=outs[0], pos=self.buf["d_pi"])

    def _traced(self, phase, body):
        """Run ``body`` eagerly (compile phase) or capture it as its own trace; returns (trace id, result)."""
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0) if phase == "capture" else None
        out = body()
        if tid is not None:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        return tid, out

    def _capture_gated(self, tok, pos):
        """Compile pass (eager, chained), then capture the K step traces and the K verify traces
        (with the sampler: each followed by its add-on trace); each captured trace reads the
        persistent outputs of the ones captured before it."""
        sampler = self.sampler
        for phase in ("compile", "capture"):
            self._write(tok, 0, pos)
            steps, verifies, prev = [], {}, None
            draft_addons, verify_addons = [], {}
            for i in range(1, self.K + 1):
                tid, prev = self._traced(phase, lambda: self._step_body(prev))
                steps.append((tid, prev))
                if sampler is not None:
                    draft_addons.append(self._traced(phase, lambda: self._draft_addon(i, prev))[0])
            for P in range(2, self.P + 1):
                tid, out = self._traced(phase, lambda: self._verify_body(P, [o for _, o in steps]))
                verifies[P] = (tid, out)
                if sampler is not None:
                    verify_addons[P] = self._traced(phase, lambda: self._verify_addon(P, out))[0]
            if phase == "compile":
                for _, outs in steps + list(verifies.values()):
                    for t in outs:
                        t.deallocate(True)
        self.steps, self.verifies = steps, verifies
        if sampler is not None:
            self.draft_addons, self.verify_addons = draft_addons, verify_addons

    def _gate_tau(self, pos):
        """Confidence threshold at ``pos`` (None: below the first breakpoint, fused cycle)."""
        tau = None
        for ctx, t in self.gate:
            if pos < ctx:
                break
            tau = t
        return tau

    def _draft_gated(self, tau):
        """Replay drafter steps while the drafter's confidence stays >= tau; returns the draft count."""
        for k, (tid, outs) in enumerate(self.steps, start=1):
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            if k == self.K or ttnn.to_torch(outs[2]).reshape(-1)[0].item() < tau:
                return k

    def prepare_eager(self, tok, pos):
        """Allocate and stage the persistent inputs without capturing (profiling / eager runs)."""
        if self.buf is not None:
            self._write(tok, 0, pos)
            return
        host = self._host_inputs(tok, 0, pos)
        self.buf = {k: ttnn.to_device(v, device=self.mesh) for k, v in host.items()}
        self.buf["hsrc"] = ttnn.from_torch(
            torch.zeros(self.P, self.hidden, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
        )

    def capture(self, tok, pos):
        """Compile run, then capture the cycle trace; the next ``step`` restarts with a zero seed."""
        self.prepare_eager(tok, pos)
        for t in self._body():  # compile run
            t.deallocate(True)
        self._write(tok, 0, pos)
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out = self._body()
        ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        if self.split_traces():
            self._capture_gated(tok, pos)
        ttnn.synchronize_device(self.mesh)
        self.state = None

    def split_traces(self):
        """Per-step drafter traces + per-P verify traces: the gated greedy cycle and every sampled cycle."""
        return self.batched and self.K > 1 and (bool(self.gate) or self.sampler is not None)

    def _zero_seed(self):
        ttnn.copy_host_to_device_tensor(
            self._host(torch.zeros(self.P, self.hidden, dtype=torch.bfloat16), ttnn.bfloat16), self.buf["hsrc"]
        )

    def start(self, token, pos):
        """Stage a cycle for ``token`` (not yet through the target) at position ``pos`` with a zero
        drafter seed (the drafter attends KV rows < pos); captures the trace on first use."""
        if self.trace_id is None:
            self.capture(token, pos)
        self._write(token, 0, pos)
        self._zero_seed()
        self.state = (token, pos)

    def reset(self):
        """Drop the staged cycle (a prefill rewrote the KV): the next step starts afresh even at the same
        (token, position), never continuing another sequence's drafter seed."""
        self.state = None

    def step(self, token, pos):
        """One cycle for ``token`` at ``pos``, continuing the staged cycle (and its drafter seed) when
        it is exactly that one, otherwise restarting with a zero seed. Returns the committed tokens."""
        if self.state != (token, pos):
            self.start(token, pos)
        return self.cycle()[0]

    def cycle(self):
        """Run the staged cycle. Returns (committed tokens, accepted drafts m): d1..dm then the target
        token of row m, i.e. the greedy tokens at positions p+1..p+m+1. Stages the next cycle
        (c = last committed token at p+m+1, seed row m)."""
        _, p = self.state
        tau = self._gate_tau(p) if self.gate else None
        if tau is not None:
            K = self._draft_gated(tau)
            tid, out = self.verifies[K + 1]
        else:
            K, tid, out = self.K, self.trace_id, self.out
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        both = ttnn.to_torch(out[2]).reshape(-1).tolist()  # one readback: drafts row, then targets
        drafts, targets = both[1 : K + 1], both[K + 1 :]
        m = next((i for i in range(K) if drafts[i] != targets[i]), K)
        self.drafted += K
        committed = drafts[:m] + [targets[m]]
        c, p = committed[-1], p + m + 1
        self._write(c, m, p, row_p_written=m < K)  # row p+m+1 holds the rejected draft's KV unless all accepted
        self.state = (c, p)
        return committed, m

    def _draft_confidence(self, k):
        """Gate signal after drafter step k: the max probability of the processed draft distribution q_k (sampled
        drafts), or the drafter's raw softmax max-probability (point-mass drafts, whose q is 1)."""
        if self.point_draft:
            return ttnn.to_torch(self.steps[k - 1][1][2]).reshape(-1)[0].item()
        bits = int(self.sampler.read_out()[k - 1, 3]) & 0xFFFFFFFF
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    def step_sampled(self, token, pos, host_draw):
        """Sampled counterpart of ``step`` (the request's sampler params and history are loaded by the caller)."""
        if self.state != (token, pos):
            self.start(token, pos)
        return self.cycle_sampled(host_draw)

    def cycle_sampled(self, host_draw):
        """Run the staged cycle with lossless speculative sampling on device: drafter steps 1..k (each followed by
        its sampler add-on: x_i ~ q_i), k = K or the first step whose draft confidence falls below the context's
        gate threshold, then verify P = k + 1 rows and the rejection chain. Returns (committed tokens at positions
        p+1..p+m+1, accepted drafts m, drafts k, verify flags). A position the device flags (truncated top-k tie
        group in row m) is drawn by ``host_draw(logits_row, position, tokens_before_it)``. Stages the next cycle
        exactly as ``cycle`` (c = last committed token at p+m+1, seed row m)."""
        from models.demos.gemma4.tt.device_sampling import V_HOST

        _, p = self.state
        tau = self._gate_tau(p) if self.gate else None
        k = 0
        for k, (tid, _) in enumerate(self.steps, start=1):
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            ttnn.execute_trace(self.mesh, self.draft_addons[k - 1], cq_id=0, blocking=False)
            if k == self.K or (tau is not None and self._draft_confidence(k) < tau):
                break
        tid, out = self.verifies[k + 1]
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        ttnn.execute_trace(self.mesh, self.verify_addons[k + 1], cq_id=0, blocking=False)
        cyc = self.sampler.read_cyc()
        m, n_emit, flags = cyc[0], cyc[1], cyc[2]
        committed = cyc[4 : 4 + n_emit]
        if flags == V_HOST:
            rows = ttnn.to_torch(out[3]).reshape(-1, self.vocab)
            committed = committed + [host_draw(rows[m], p + m + 1, list(committed))]
        if len(committed) != m + 1 or not 0 <= m <= k:
            raise RuntimeError(f"sampled verify returned {cyc[:4 + k + 1]} for {k} drafts")
        self.drafted += k
        c, p = committed[-1], p + m + 1
        self._write(c, m, p, row_p_written=m < k)  # row p+m+1 holds the rejected draft's KV unless all accepted
        self.state = (c, p)
        return committed, m, k, flags

    def generate(self, token, pos, max_new, stop_ids=()):
        """Greedy tokens after ``token`` (the first generated token, at ``pos``), up to ``max_new``.
        Returns (tokens, per-cycle records)."""
        self.start(token, pos)
        out, cycles = [], []
        while len(out) < max_new:
            t0 = time.time()
            committed, m = self.cycle()
            cycles.append({"accepted": m, "s": time.time() - t0, "drafted": self.drafted})
            self.drafted = 0
            stop = False
            for t in committed:
                out.append(t)
                if t in stop_ids or len(out) >= max_new:
                    stop = True
                    break
            if stop:
                break
        ttnn.synchronize_device(self.mesh)
        return out, cycles

    def release(self):
        """Release the trace and its output buffers (a re-capture allocates them afresh)."""
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None
        if self.out is not None:
            for t in self.out:
                t.deallocate(True)
            self.out = None
        for tid, outs in (self.steps or []) + list((self.verifies or {}).values()):
            ttnn.release_trace(self.mesh, tid)
            for t in outs:
                t.deallocate(True)
        self.steps = self.verifies = None
        for tid in (self.draft_addons or []) + list((self.verify_addons or {}).values()):
            ttnn.release_trace(self.mesh, tid)
        self.draft_addons = self.verify_addons = None
