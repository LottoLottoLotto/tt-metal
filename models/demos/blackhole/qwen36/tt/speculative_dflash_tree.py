# SPDX-License-Identifier: Apache-2.0
"""Two native DFlash paths, one shared-prefix target tree, one winning commit."""
import ttnn

from models.demos.blackhole.qwen36.tt.speculative_dflash import _DFlashTraceCycle
from models.demos.blackhole.qwen36.tt.speculative_tree import TreeTargetVerifier


class DFlashTreeCycle(_DFlashTraceCycle, TreeTargetVerifier):
    def __init__(self, model, drafter, page_table_width, fork_position=3):
        if drafter.block_size != 8:
            raise ValueError("Two-branch DFlash requires block size 8")
        self.drafter = drafter
        super().__init__(model, page_table_width, fork_position)
        self._allocate_draft_io(paths=2)
        self._pending = None
        self.last_stats = None

    def _first_drafts_step(self):
        branches = self.drafter.draft_branches(
            self.first_token, self.first_position, self.initial_features,
            self.first_position, self.zero, self.fork_position,
        )
        converted = [ttnn.typecast(path, ttnn.int32) for path in branches]
        joined = ttnn.concat(converted, dim=1)
        ttnn.copy(joined, self.first_out)
        for tensor in (*converted, joined):
            ttnn.deallocate(tensor)

    def first_drafts(self, token, position):
        values = super().first_drafts(token, position)
        primary, alternate = values[:7], values[7:]
        self._pending = (int(token), int(position), tuple(primary), tuple(alternate))
        return primary

    def seed(self, token, position, page_table_row, proposals):
        primary = tuple(int(value) for value in proposals)
        if self._pending is None or self._pending[:3] != (int(token), int(position), primary):
            raise RuntimeError("DFlash branches do not belong to the scheduler's current anchor and proposals")
        TreeTargetVerifier.seed(self, token, position, page_table_row,
                                branches=(primary, self._pending[3]))

    def seed_for_warmup(self, table):
        primary = [12] * 7
        alternate = primary.copy()
        alternate[self.fork_position - 1] = 13
        TreeTargetVerifier.seed(self, 11, 0, table, branches=(primary, alternate))

    def step(self):
        hidden, candidates = self.target_pass()
        logits = self.vocabulary(hidden)
        ttnn.deallocate(hidden)
        selection = self.accept_branches(logits)
        ttnn.deallocate(logits)
        self.commit(candidates, selection)
        targets, flags = selection["targets"], selection["flags"]
        next_token = self._next_anchor(targets, flags)
        advance = ttnn.add(selection["accepted"], 1)
        start = ttnn.reshape(self.positions[:1], [1, 1])
        next_position = ttnn.add(start, advance)
        features = self.selected_features(selection["winner"])
        branches = self.drafter.draft_branches(
            next_token, next_position, features, start, advance, self.fork_position,
        )
        ttnn.deallocate(features)
        primary, alternate = [ttnn.typecast(path, ttnn.int32) for path in branches]
        compact = ttnn.concat(
            [targets, *flags, primary, alternate, selection["winner"],
             selection["accepted_a"], selection["accepted_b"]], dim=1,
        )
        ttnn.copy(compact, self.out)
        self.pack_branches(next_token, branches)
        positions = [ttnn.reshape(ttnn.add(next_position, offset), [1]) for offset in self.position_offsets]
        updated_positions = ttnn.concat(positions, dim=0)
        ttnn.copy(updated_positions, self.positions)
        for tensor in (next_token, advance, next_position, primary, alternate,
                       compact, updated_positions, *positions):
            ttnn.deallocate(tensor)
        for name, value in selection.items():
            for tensor in value if name == "flags" else (value,):
                ttnn.deallocate(tensor)
        return self.out

    def decode_result(self, compact, position):
        if len(compact) != 32:
            raise RuntimeError("Invalid two-branch DFlash result width")
        accepted = sum(int(flag) for flag in compact[8:15])
        emitted = [int(value) for value in compact[:accepted + 1]]
        primary = [int(value) for value in compact[15:22]]
        alternate = [int(value) for value in compact[22:29]]
        winner, accepted_a, accepted_b = (int(value) for value in compact[29:32])
        if accepted != max(accepted_a, accepted_b) or winner != int(accepted_b > accepted_a):
            raise RuntimeError("Invalid longest-prefix DFlash branch selection")
        self._pending = (emitted[-1], int(position) + accepted + 1, tuple(primary), tuple(alternate))
        self.last_stats = {"winner": winner, "accepted_a": accepted_a,
                           "accepted_b": accepted_b, "extra_tokens": accepted - accepted_a}
        return emitted, primary

    def reset(self):
        super().reset()
        self._pending = None
        self.last_stats = None

    def release(self):
        self.release_traces()
        super().release()
        self._release_draft_io()
        self._pending = None
