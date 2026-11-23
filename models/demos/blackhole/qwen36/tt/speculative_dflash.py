# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native DFlash draft/verify/commit transactions with scheduler-owned inputs."""
import torch
import ttnn

from models.demos.blackhole.qwen36.tt.speculative import TargetVerifier


class _DFlashTraceCycle:
    """Shared fixed-address I/O and trace lifecycle for linear/tree transactions."""

    def _allocate_draft_io(self, paths=1):
        i32 = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        size = (2 + paths) * self.depth + 1 + (3 if paths == 2 else 0)
        self.out = ttnn.from_torch(torch.zeros((1, size), dtype=torch.int32), **i32)
        self.first_token = ttnn.from_torch(torch.zeros((1, 1), dtype=torch.int32),
                                          dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        self.first_position = ttnn.from_torch(torch.zeros((1, 1), dtype=torch.int32), **i32)
        self.first_out = ttnn.from_torch(torch.zeros((1, paths * self.depth), dtype=torch.int32), **i32)
        self.zero = ttnn.from_torch(torch.zeros((1, 1), dtype=torch.int32), **i32)
        self.initial_features = ttnn.from_torch(
            torch.zeros((1, 1, self.drafter.block_size, 5 * self.model.args.dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.trace = self.first_trace = None
        self.compiled = self.first_compiled = False

    @staticmethod
    def _next_anchor(targets, flags):
        next_token = targets[:, :1]
        for i, flag in enumerate(flags, 1):
            difference = ttnn.subtract(targets[:, i:i + 1], next_token)
            increment = ttnn.multiply(flag, difference)
            selected = ttnn.add(next_token, increment)
            if i > 1:
                ttnn.deallocate(next_token)
            ttnn.deallocate(difference)
            ttnn.deallocate(increment)
            next_token = selected
        return next_token

    def warmup_first_drafts(self, capture=True):
        if self.first_trace is not None:
            return
        if not self.first_compiled:
            self._first_drafts_step()
            ttnn.synchronize_device(self.device)
            self.first_compiled = True
        if capture:
            trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._first_drafts_step()
            ttnn.end_trace_capture(self.device, trace, cq_id=0)
            self.first_trace = trace

    def warmup(self, capture=True):
        if self.trace is not None:
            return
        if not self.compiled:
            self.step()
            ttnn.synchronize_device(self.device)
            self.compiled = True
        if capture:
            trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            self.step()
            ttnn.end_trace_capture(self.device, trace, cq_id=0)
            self.trace = trace

    def first_drafts(self, token, position):
        if not self.first_compiled:
            raise RuntimeError("Initial DFlash programs must be warmed before serving")
        for value, buffer, dtype in ((token, self.first_token, ttnn.uint32),
                                      (position, self.first_position, ttnn.int32)):
            source = ttnn.from_torch(torch.tensor([[value]], dtype=torch.int32),
                                     dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(source, buffer)
        if self.first_trace is None:
            self._first_drafts_step()
        else:
            ttnn.execute_trace(self.device, self.first_trace, cq_id=0, blocking=True)
        return [int(value) for value in ttnn.to_torch(self.first_out).reshape(-1).tolist()]

    def run(self):
        if not self.compiled:
            raise RuntimeError("DFlash programs must be warmed before replay")
        if self.trace is None:
            self.step()
        else:
            ttnn.execute_trace(self.device, self.trace, cq_id=0, blocking=True)
        return self.out

    def seed_for_warmup(self, table):
        self.seed(11, 0, table, proposals=[12] * self.depth)

    def decode_result(self, compact, position):
        accepted = sum(int(flag) for flag in compact[self.depth + 1:2 * self.depth + 1])
        return ([int(value) for value in compact[:accepted + 1]],
                [int(value) for value in compact[2 * self.depth + 1:]])

    def reset(self):
        self.drafter.reset()

    def release_traces(self):
        for name in ("trace", "first_trace", "sampled_trace", "sampled_first_trace"):
            trace = getattr(self, name, None)
            if trace is not None:
                ttnn.release_trace(self.device, trace)
                setattr(self, name, None)
        self.compiled = self.first_compiled = False
        self.sampled_compiled = self.sampled_first_compiled = False

    def _release_draft_io(self):
        for tensor in (self.out, self.first_token, self.first_position, self.first_out,
                       self.zero, self.initial_features, getattr(self, "sampled_out", None),
                       getattr(self, "verify_logits", None)):
            if tensor is not None:
                ttnn.deallocate(tensor)


class DFlashCycle(_DFlashTraceCycle, TargetVerifier):
    # The next transaction always starts at the first rejected row (anchor = target
    # correction/bonus at start + accepted + 1) and rewrites it and every later row
    # before attending, so verification K/V goes straight into the live cache.
    direct_kv = True

    def __init__(self, model, drafter, page_table_width):
        from models.demos.blackhole.qwen36.tt.verify_accept import NativeVerifyAccept

        self.drafter = drafter
        super().__init__(model, drafter.block_size, page_table_width, (5, 19, 33, 47, 61))
        self._allocate_draft_io()
        self.accept = NativeVerifyAccept(self.device, self.rows, model.args.vocab_size)

    def step(self):
        hidden, candidates = self.target_pass()
        logits = self.vocabulary(hidden)
        ttnn.deallocate(hidden)
        # Native argmax + accepted prefix; advances self.positions in place. The correction/bonus
        # anchor is target[accepted], not the last proposal; start is the old first position.
        head, accepted, next_token, advance, next_position, start = self.accept(logits, self.tokens, self.positions)
        ttnn.deallocate(logits)
        self.commit_gdn(candidates, accepted)
        self.commit_kv(candidates, accepted)
        features = ttnn.concat(list(self.features.values()), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        proposals = self.drafter.draft(
            next_token, next_position,
            ttnn.reshape(features, [1, 1, self.rows, 5 * self.model.args.dim]), start, advance)
        ttnn.deallocate(features)
        proposals_i32 = ttnn.typecast(proposals, ttnn.int32)
        compact = ttnn.concat([head, proposals_i32], dim=1)
        ttnn.copy(compact, self.out)
        # Device-only advancing replay is useful for warmup/probes. Serving still
        # reseeds every transaction from scheduler tokens, position, and real pages.
        updated = ttnn.typecast(ttnn.concat([next_token, proposals_i32], dim=1), ttnn.uint32)
        ttnn.copy(updated, self.tokens)
        for tensor in (proposals_i32, compact, updated):
            ttnn.deallocate(tensor)
        # NativeDFlash and NativeVerifyAccept own their stable outputs; do not free them here.
        return self.out

    def _first_drafts_step(self):
        proposals = self.drafter.draft(self.first_token, self.first_position,
                                       self.initial_features, self.first_position, self.zero)
        converted = ttnn.typecast(proposals, ttnn.int32)
        ttnn.copy(converted, self.first_out)
        ttnn.deallocate(converted)

    # ---- Exact speculative sampling (device_sampling.py) for random / penalized requests ----
    # Draft modes: "point" verifies the unchanged greedy selector path as a deterministic proposal (point-mass q:
    # accept x_i with probability p_i(x_i), residual p_i without x_i); "sampled" walks the selector by drawing each
    # x_i from q_i = softmax(scores_i / T) over its 16 candidates (DFlash2's reference probabilistic walk);
    # "processed" draws from those scores after the request's full vLLM processing (top-k, top-p, min_p).
    SAMPLED_MODES = ("point", "sampled", "processed")

    def enable_sampling(self, sampler, mode):
        """Allocate the sampled cycle's fixed outputs (before any trace)."""
        if mode not in self.SAMPLED_MODES:
            raise ValueError(f"DFlash draft sampling mode must be one of {self.SAMPLED_MODES}")
        if getattr(self, "sampler", None) is not None:
            return
        self.sampler, self.sampled_mode = sampler, mode
        sampler.va_buffers(self.rows)
        self.sampled_out = ttnn.from_torch(torch.zeros((1, 2 * self.rows - 1 + self.depth + 1), dtype=torch.int32),
                                           dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        self.verify_logits = None
        self.sampled_trace = self.sampled_first_trace = None
        self.sampled_compiled = self.sampled_first_compiled = False

    def _draft_kwargs(self, position):
        if self.sampled_mode == "point":
            return {}
        return dict(sampler=self.sampler, sample_position=position, processed=self.sampled_mode == "processed")

    def step_sampled(self):
        hidden, candidates = self.target_pass()
        logits = self.vocabulary(hidden)
        ttnn.deallocate(hidden)
        # Speculative rejection sampling of the proposal chain; same outputs (and in-place position advance) as
        # NativeVerifyAccept, with the emitted tokens (accepted drafts + correction/bonus) as head targets.
        head, accepted, next_token, advance, next_position, start = self.sampler.verify(
            logits, self.tokens, self.positions, self.rows, point_q=self.sampled_mode == "point", va=True)
        # Retained (fixed address in the trace): the host redraws a truncated top-k tie group from these rows.
        self.verify_logits = logits
        self.commit_gdn(candidates, accepted)
        self.commit_kv(candidates, accepted)
        features = ttnn.concat(list(self.features.values()), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        proposals = self.drafter.draft(
            next_token, next_position,
            ttnn.reshape(features, [1, 1, self.rows, 5 * self.model.args.dim]), start, advance,
            **self._draft_kwargs(next_position))
        ttnn.deallocate(features)
        proposals_i32 = ttnn.typecast(proposals, ttnn.int32)
        vflag = ttnn.slice(self.sampler.cyc, (0, 17), (1, 18))
        compact = ttnn.concat([head, proposals_i32, vflag], dim=1)
        ttnn.copy(compact, self.sampled_out)
        updated = ttnn.typecast(ttnn.concat([next_token, proposals_i32], dim=1), ttnn.uint32)
        ttnn.copy(updated, self.tokens)
        for tensor in (proposals_i32, vflag, compact, updated):
            ttnn.deallocate(tensor)
        return self.sampled_out

    def _first_drafts_step_sampled(self):
        proposals = self.drafter.draft(self.first_token, self.first_position, self.initial_features,
                                       self.first_position, self.zero, **self._draft_kwargs(self.first_position))
        converted = ttnn.typecast(proposals, ttnn.int32)
        ttnn.copy(converted, self.first_out)
        ttnn.deallocate(converted)

    def _drop_verify_logits(self):
        if self.verify_logits is not None:
            ttnn.deallocate(self.verify_logits)
            self.verify_logits = None

    def warmup_sampled(self, capture=True):
        """Run the sampled cycle (and the sampled first drafts) eagerly in both warmup phases, so every program,
        including the capture phase's prefetcher-fed variants, exists before capture at fixed addresses."""
        point = self.sampled_mode == "point"
        if capture and self.sampled_trace is not None and (point or self.sampled_first_trace is not None):
            return
        self._drop_verify_logits()
        self.step_sampled()
        ttnn.synchronize_device(self.device)
        self.sampled_compiled = True
        if not point:
            self._first_drafts_step_sampled()
            ttnn.synchronize_device(self.device)
            self.sampled_first_compiled = True
        if not capture:
            return
        if self.sampled_trace is None:
            self._drop_verify_logits()
            trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            self.step_sampled()
            ttnn.end_trace_capture(self.device, trace, cq_id=0)
            self.sampled_trace = trace
        if not point and self.sampled_first_trace is None:
            trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._first_drafts_step_sampled()
            ttnn.end_trace_capture(self.device, trace, cq_id=0)
            self.sampled_first_trace = trace

    def run_sampled(self):
        if not self.sampled_compiled:
            raise RuntimeError("Sampled DFlash programs must be warmed before replay")
        if self.sampled_trace is None:
            self._drop_verify_logits()
            self.step_sampled()
        else:
            ttnn.execute_trace(self.device, self.sampled_trace, cq_id=0, blocking=False)
        return [int(value) for value in ttnn.to_torch(self.sampled_out).reshape(-1).tolist()]

    def first_drafts_sampled(self, token, position):
        """Initial proposals for a sampled request: the greedy path (point mode) or a sampled selector walk whose
        q_i stay on device for the next verification."""
        if self.sampled_mode == "point":
            return self.first_drafts(token, position)
        if not self.sampled_first_compiled:
            raise RuntimeError("Sampled DFlash first-draft programs must be warmed before serving")
        for value, buffer, dtype in ((token, self.first_token, ttnn.uint32),
                                      (position, self.first_position, ttnn.int32)):
            source = ttnn.from_torch(torch.tensor([[value]], dtype=torch.int32),
                                     dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(source, buffer)
        if self.sampled_first_trace is None:
            self._first_drafts_step_sampled()
        else:
            ttnn.execute_trace(self.device, self.sampled_first_trace, cq_id=0, blocking=False)
        return [int(value) for value in ttnn.to_torch(self.first_out).reshape(-1).tolist()]

    def decode_sampled(self, compact):
        """(emitted tokens, next proposals, host-redraw chain index or None)."""
        rows, depth = self.rows, self.depth
        accepted = sum(int(flag) for flag in compact[rows:2 * rows - 1])
        emitted = [int(value) for value in compact[:accepted + 1]]
        proposals = [int(value) for value in compact[2 * rows - 1:2 * rows - 1 + depth]]
        vflag = int(compact[2 * rows - 1 + depth])
        return emitted, proposals, (vflag - 1 if vflag else None)

    def release(self):
        self.release_traces()
        super().release()
        self._release_draft_io()
        self.accept.release()
