# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact chained MTP transactions with one shared target pass per proposal chain."""
import torch
import ttnn
from models.demos.blackhole.qwen36.tt.speculative import TargetVerifier


class WideMTPCycle(TargetVerifier):
    """D proposals, D+1 target rows; selected state includes the current input row."""

    def __init__(self, model, depth, page_table_width):
        if not 2 <= depth <= 31:
            raise ValueError("Wider MTP depth must be in 2..31")
        model.init_mtp_cycle(page_table_width)
        super().__init__(model, depth + 1, page_table_width)
        i32 = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        self.out = ttnn.from_torch(torch.zeros((1, 3 * depth + 1), dtype=torch.int32), **i32)
        self.cache_positions = tuple(ttnn.from_torch(torch.tensor([i], dtype=torch.int32), **i32)
                                     for i in range(depth))
        self.trace = None
        self.compiled = False
        # These addresses must exist before prefill/decode traces reserve replay scratch.
        self.first_token = ttnn.from_torch(torch.zeros((1, 1), dtype=torch.int32), **i32)
        self.first_position = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), **i32)
        self.first_out = ttnn.from_torch(torch.zeros((1, depth), dtype=torch.int32), **i32)
        self.first_trace = None
        self.first_compiled = False


    def select_next(self, hidden, targets, flags, accepted_count):
        selected = hidden[:, :1, :]
        next_token = targets[:, :1]
        for i, flag in enumerate(flags, 1):
            flag_tile = ttnn.to_layout(
                ttnn.typecast(ttnn.reshape(flag, [1, 1, 1]), ttnn.bfloat16), ttnn.TILE_LAYOUT)
            previous = selected
            selected = ttnn.where(flag_tile, hidden[:, i:i + 1, :], previous)
            if i > 1:
                ttnn.deallocate(previous)
            ttnn.deallocate(flag_tile)
            previous_token = next_token
            difference = ttnn.subtract(targets[:, i:i + 1], previous_token)
            increment = ttnn.multiply(flag, difference)
            next_token = ttnn.add(previous_token, increment)
            if i > 1:
                ttnn.deallocate(previous_token)
            ttnn.deallocate(difference)
            ttnn.deallocate(increment)
        advance = ttnn.add(accepted_count, 1)
        next_position = ttnn.add(self.positions[:1], ttnn.reshape(advance, [1]))
        ttnn.deallocate(advance)
        return next_token, selected, next_position

    def draft_chain(self, first_token, target_hidden, first_position):
        proposals = []
        hidden = target_hidden
        token = first_token
        for offset, cache_position in enumerate(self.cache_positions):
            position = ttnn.add(first_position, offset)
            proposal, next_hidden = self.model.mtp.draft_device(
                token, hidden, position, cache_position=cache_position, return_hidden=True)
            ttnn.deallocate(position)
            if offset:
                ttnn.deallocate(hidden)
            hidden = next_hidden
            token = proposal
            proposals.append(proposal)
        ttnn.deallocate(hidden)
        return proposals

    def update_inputs(self, next_token, next_position, proposals, targets, flags):
        updated = ttnn.typecast(ttnn.concat([next_token, *proposals], dim=1), ttnn.uint32)
        ttnn.copy(updated, self.tokens)
        ttnn.deallocate(updated)
        positions = [ttnn.add(next_position, i) for i in range(self.rows)]
        updated_positions = ttnn.concat(positions, dim=0)
        ttnn.copy(updated_positions, self.positions)
        ttnn.deallocate(updated_positions)
        for position in positions:
            ttnn.deallocate(position)
        compact = ttnn.concat([targets, *flags, *proposals], dim=1)
        ttnn.copy(compact, self.out)
        ttnn.deallocate(compact)

    def step(self):
        hidden, candidates = self.target_pass()
        logits = self.vocabulary(hidden)
        targets, flags, accepted_count = self.sample_accept(logits)
        ttnn.deallocate(logits)
        self.commit_gdn(candidates, accepted_count)
        self.commit_kv(candidates, accepted_count)
        next_token, selected, next_position = self.select_next(hidden, targets, flags, accepted_count)
        ttnn.deallocate(hidden)
        proposals = self.draft_chain(next_token, selected, next_position)
        ttnn.deallocate(selected)
        self.update_inputs(next_token, next_position, proposals, targets, flags)
        for tensor in [targets, *flags, accepted_count, next_token, next_position, *proposals]:
            ttnn.deallocate(tensor)
        return self.out

    def seed(self, token, position, page_table_row, proposals=None):
        if proposals is None:
            token_tt = ttnn.from_torch(torch.tensor([[token]], dtype=torch.int32), dtype=ttnn.int32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            position_tt = ttnn.from_torch(torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            hidden = ttnn.reshape(self.model._last_hidden, [1, 32, self.model.args.dim])[:, :1, :]
            generated = self.draft_chain(token_tt, hidden, position_tt)
            proposals = [int(ttnn.to_torch(value).item()) for value in generated]
            for value in [token_tt, position_tt, *generated]:
                ttnn.deallocate(value)
        return super().seed(token, position, page_table_row, proposals)

    def _first_drafts_step(self):
        hidden = ttnn.reshape(self.model._last_hidden, [1, 32, self.model.args.dim])[:, :1, :]
        proposals = self.draft_chain(self.first_token, hidden, self.first_position)
        ttnn.deallocate(hidden)
        compact = ttnn.concat(proposals, dim=1)
        ttnn.copy(compact, self.first_out)
        ttnn.deallocate(compact)
        for proposal in proposals:
            ttnn.deallocate(proposal)

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
            ttnn.execute_trace(self.device, trace, cq_id=0, blocking=True)

    def first_drafts(self, token, position):
        """Generate the complete genuine chain without advancing target state."""
        if not self.first_compiled:
            raise RuntimeError("Initial wide draft programs must be warmed before serving")
        token_host = ttnn.from_torch(torch.tensor([[token]], dtype=torch.int32),
                                     dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        position_host = ttnn.from_torch(torch.tensor([position], dtype=torch.int32),
                                        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(token_host, self.first_token)
        ttnn.copy_host_to_device_tensor(position_host, self.first_position)
        if self.first_trace is None:
            self._first_drafts_step()
        else:
            ttnn.execute_trace(self.device, self.first_trace, cq_id=0, blocking=False)
        return [int(value) for value in ttnn.to_torch(self.first_out).reshape(-1).tolist()]

    def warmup(self, capture=True):
        if self.trace is not None:
            return
        if not self.compiled:
            self.step()
            ttnn.synchronize_device(self.device)
            self.compiled = True
        if capture:
            self.trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            self.step()
            ttnn.end_trace_capture(self.device, self.trace, cq_id=0)
            ttnn.execute_trace(self.device, self.trace, cq_id=0, blocking=True)

    def run(self):
        if not self.compiled:
            raise RuntimeError("Compile the wider cycle before replay")
        if self.trace is None:
            self.step()
        else:
            ttnn.execute_trace(self.device, self.trace, cq_id=0, blocking=True)
        return self.out

    def release(self):
        if self.trace is not None:
            ttnn.release_trace(self.device, self.trace)
            self.trace = None
        self.compiled = False
        if self.first_trace is not None:
            ttnn.release_trace(self.device, self.first_trace)
            self.first_trace = None
        self.first_compiled = False
