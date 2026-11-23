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
            torch.zeros((1, 1, self.drafter.block_size, self.drafter.feature_width), dtype=torch.bfloat16),
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
        for name in ("trace", "first_trace"):
            trace = getattr(self, name)
            if trace is not None:
                ttnn.release_trace(self.device, trace)
                setattr(self, name, None)
        self.compiled = self.first_compiled = False

    def _release_draft_io(self):
        for tensor in (self.out, self.first_token, self.first_position, self.first_out,
                       self.zero, self.initial_features):
            ttnn.deallocate(tensor)


class DFlashCycle(_DFlashTraceCycle, TargetVerifier):
    def __init__(self, model, drafter, page_table_width):
        self.drafter = drafter
        super().__init__(model, drafter.block_size, page_table_width, drafter.feature_layers)
        self._allocate_draft_io()

    def step(self):
        hidden, candidates = self.target_pass()
        logits = self.vocabulary(hidden)
        ttnn.deallocate(hidden)
        targets, flags, accepted = self.sample_accept(logits)
        ttnn.deallocate(logits)
        self.commit_gdn(candidates, accepted)
        self.commit_kv(candidates, accepted)
        # The correction/bonus is target[accepted], not the last proposal.
        next_token = self._next_anchor(targets, flags)
        advance = ttnn.add(accepted, 1)
        start = ttnn.reshape(self.positions[:1], [1, 1])
        next_position = ttnn.add(start, advance)
        features = ttnn.concat(list(self.features.values()), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        proposals = self.drafter.draft(
            next_token, next_position,
            ttnn.reshape(features, [1, 1, self.rows, self.drafter.feature_width]), start, advance)
        ttnn.deallocate(features)
        proposals_i32 = ttnn.typecast(proposals, ttnn.int32)
        compact = ttnn.concat([targets, *flags, proposals_i32], dim=1)
        ttnn.copy(compact, self.out)
        # Device-only advancing replay is useful for warmup/probes. Serving still
        # reseeds every transaction from scheduler tokens, position, and real pages.
        updated = ttnn.typecast(ttnn.concat([next_token, proposals_i32], dim=1), ttnn.uint32)
        ttnn.copy(updated, self.tokens)
        positions = [ttnn.reshape(ttnn.add(next_position, offset), [1]) for offset in range(self.rows)]
        updated_positions = ttnn.concat(positions, dim=0)
        ttnn.copy(updated_positions, self.positions)
        for tensor in (targets, *flags, accepted, next_token, advance, next_position,
                       proposals_i32, compact, updated, updated_positions, *positions):
            ttnn.deallocate(tensor)
        # NativeDFlash owns the stable proposal output; do not free it here.
        return self.out

    def _first_drafts_step(self):
        proposals = self.drafter.draft(self.first_token, self.first_position,
                                       self.initial_features, self.first_position, self.zero)
        converted = ttnn.typecast(proposals, ttnn.int32)
        ttnn.copy(converted, self.first_out)
        ttnn.deallocate(converted)


    def release(self):
        self.release_traces()
        super().release()
        self._release_draft_io()
