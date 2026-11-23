# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced single-user greedy decode with on-device argmax and token/position feedback.

One captured trace runs: decoder stack -> final norm -> LM head -> softcap -> argmax over the
real vocabulary -> writes the argmax token into the trace's own token input buffer and advances
both position buffers by one. Consecutive replays therefore need no host input at all: the host
enqueues ``execute_trace`` + a non-blocking 4-byte token read per step and only waits for a token
when it needs it (stop check), so the device runs back-to-back steps.

Greedy semantics are those of ``torch.argmax`` over the bf16 softcapped logits the host path reads
(first index among equal maxima): ``ttnn.argmax`` reduces the untilized logits row and breaks ties
towards the smaller index.
"""

import torch

import ttnn


class _DecodeTrace:
    """Persistent decode inputs (token [1,1] u32, RoPE position [1,32] u32, cache position [1] i32) and
    one captured trace over them. ``page_tables`` are persistent device page tables (one tensor, or
    ``{layer_type: tensor}``) owned by the caller, who updates them in place before a replay."""

    def __init__(self, model, mesh_device, kv_cache, page_tables):
        self.model = model
        self.mesh = mesh_device
        self.kv_cache = kv_cache
        self.page_tables = page_tables
        self.vocab = model.vocab_size
        self.trace_id = None
        self.inputs = None
        self.out = None

    def _host_inputs(self, token, pos):
        return self.model.prepare_decode_inputs_host(torch.tensor([token]), torch.tensor([pos]))[:3]

    def _write_inputs(self, token, pos):
        for h, d in zip(self._host_inputs(token, pos), self.inputs):
            ttnn.copy_host_to_device_tensor(h, d)

    def _forward(self):
        tok_in, pos_rope, pos_cache = self.inputs
        logits, _ = self.model.ttnn_decode_forward(
            tok_in, pos_rope, rot_mat_idxs=pos_cache, page_table=self.page_tables, kv_cache=self.kv_cache
        )
        return logits

    def prepare(self, token, pos):
        """Allocate the persistent inputs (holding ``token`` at ``pos``) if not yet allocated."""
        if self.inputs is None:
            self.inputs = [ttnn.to_device(h, device=self.mesh) for h in self._host_inputs(token, pos)]

    def _capture(self, token, pos, body):
        """Compile run (eager), restore the inputs, capture ``body``; returns its output."""
        self.prepare(token, pos)
        self._write_inputs(token, pos)
        out = body()
        out.deallocate(True)
        self._write_inputs(token, pos)
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        out = body()
        ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        return out

    def release(self):
        """Release the trace and its output buffer (a re-capture allocates them afresh)."""
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None
        if self.out is not None:
            self.out.deallocate(True)
            self.out = None


class LogitsDecoder(_DecodeTrace):
    """Traced single-user decode returning the softcapped logits row (host sampling / teacher forcing).
    Same forward as ``GreedyDecoder``; the host stages token and position every step. With a device sampler
    (``sampler``, device_sampling.DeviceSampler) a second small trace samples the logits trace's output row on
    device (``sample``): only a 16-byte record is read back."""

    sampler = None
    sample_trace = None

    def capture(self, token, pos):
        self.out = self._capture(token, pos, self._forward)
        if self.sampler is not None:
            pos_cache = self.inputs[2]
            self.sampler.sample_decode(self.out, pos_cache)  # compile run
            self.sample_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.sampler.sample_decode(self.out, pos_cache)
            ttnn.end_trace_capture(self.mesh, self.sample_trace, cq_id=0)
            ttnn.synchronize_device(self.mesh)

    def _replay(self, token, pos):
        if self.trace_id is None:
            self.capture(token, pos)
        else:
            self._write_inputs(token, pos)
        ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)

    def step(self, token, pos):
        """Logits (float32 [vocab]) of ``token`` at position ``pos``."""
        self._replay(token, pos)
        return ttnn.to_torch(self.out).reshape(-1)[: self.vocab].float()

    def sample(self, token, pos):
        """Device-sampled token at pos + 1 after ``token`` at ``pos`` with the sampler's loaded request:
        the record (token, flags, candidates kept) of row 0."""
        self._replay(token, pos)
        ttnn.execute_trace(self.mesh, self.sample_trace, cq_id=0, blocking=False)
        return self.sampler.read_out()[0]

    def release(self):
        if self.sample_trace is not None:
            ttnn.release_trace(self.mesh, self.sample_trace)
            self.sample_trace = None
        super().release()


class GreedyDecoder(_DecodeTrace):
    def _step_body(self):
        """Forward + argmax + feedback on the persistent input buffers; returns the token tensor."""
        tok_in, pos_rope, pos_cache = self.inputs
        logits = self._forward()
        row = ttnn.untilize_with_unpadding(logits, output_tensor_end=[0, 0, 0, self.vocab - 1])
        logits.deallocate(True)
        tok = ttnn.argmax(row, dim=-1, keepdim=False)  # [1, 1, 1] uint32
        row.deallocate(True)
        ttnn.copy(ttnn.reshape(tok, tok_in.shape), tok_in)
        ttnn.plus_one(pos_rope)
        ttnn.plus_one(pos_cache)
        return tok

    def capture(self, token, pos):
        self.out = self._capture(token, pos, self._step_body)

    def start(self, token, pos):
        """Stage the first decode input (last prefill token at absolute position ``pos``);
        captures the trace on first use."""
        if self.trace_id is None:
            self.capture(token, pos)
        self._write_inputs(token, pos)

    def enqueue(self):
        """Replay one step; returns (host tensor, event) for its token, readable after the event."""
        ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
        host = self.out.cpu(blocking=False)
        return host, ttnn.record_event(self.mesh, 0)

    @staticmethod
    def token(pending):
        host, event = pending
        ttnn.event_synchronize(event)
        return int(ttnn.to_torch(host).reshape(-1)[0])

    def generate(self, token, pos, max_new, stop_ids=(), lookahead=2):
        """Greedy tokens after ``token`` (at ``pos``). Returns (tokens, per-token completion times).
        Up to ``lookahead`` steps are in flight; steps enqueued past a stop token only write KV
        positions after the returned sequence."""
        import time

        self.start(token, pos)
        out, times, pending = [], [], []
        issued = 0
        while len(out) < max_new:
            while issued < max_new and len(pending) < lookahead:
                pending.append(self.enqueue())
                issued += 1
            t = self.token(pending.pop(0))
            times.append(time.time())
            out.append(t)
            if t in stop_ids:
                break
        for p in pending:
            ttnn.event_synchronize(p[1])
        return out, times
