# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared exact target verification; proposal generation belongs to the caller."""
import torch
import ttnn


class TargetVerifier:
    """Verify anchor plus proposals and commit their longest greedy prefix.

    Feature buffers contain all proposed rows. Only rows through accepted_count,
    inclusive, belong to committed context; consumers must enforce that boundary.
    """

    def __init__(self, model, rows, page_table_width, feature_layers=()):
        if not 2 <= rows <= 32:
            raise ValueError("Target verification requires 2..32 rows")
        if page_table_width < 8 or page_table_width % 8:
            raise ValueError("Verification page-table width must be a positive multiple of eight")
        if model._paged_kv_caches and page_table_width > int(model._paged_kv_caches[0][0].shape[0]):
            raise ValueError("Verification page-table width exceeds allocated KV blocks")
        if len(set(feature_layers)) != len(feature_layers) or any(
            index < 0 or index >= len(model.layers) for index in feature_layers
        ):
            raise ValueError("Feature layers must be distinct valid target layer indices")
        model.init_target_verification()
        self.model, self.device = model, model.device
        self.rows, self.depth, self.width = rows, rows - 1, page_table_width
        i32 = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        self.tokens = ttnn.from_torch(torch.zeros((1, rows), dtype=torch.int32),
                                     dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        self.positions = ttnn.from_torch(torch.zeros(rows, dtype=torch.int32), **i32)
        self.page_table = ttnn.from_torch(torch.zeros((rows, page_table_width), dtype=torch.int32), **i32)
        self.features = {
            index: ttnn.from_torch(torch.zeros((1, rows, model.args.dim), dtype=torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for index in feature_layers
        }

    def target_pass(self):
        idx = ttnn.reshape(ttnn.typecast(self.positions, ttnn.uint32), [1, self.rows])
        cos = ttnn.embedding(idx, self.model._target_cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idx, self.model._target_sin_table, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(idx)
        x = self.model.embd(self.tokens)
        candidates = []
        for index, layer in enumerate(self.model.layers):
            previous = x
            x, candidate = layer.verify_many(x, cos, sin, self.positions, self.page_table,
                                             final_norm=self.model.verify_final_norm(index))
            ttnn.deallocate(previous)
            if index in self.features:
                ttnn.copy(x, self.features[index])
            candidates.append(candidate)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        hidden = self.model.verify_hidden(x)
        if hidden is not x:
            ttnn.deallocate(x)
        return hidden, candidates

    def vocabulary(self, hidden):
        return self.model._lm_head(hidden)

    def sample_accept(self, logits):
        logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        targets = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        targets_i32 = ttnn.typecast(targets, ttnn.int32)
        ttnn.deallocate(targets)
        flags = []
        accepted_count = None
        for i in range(self.depth):
            proposal = ttnn.typecast(self.tokens[:, i + 1:i + 2], ttnn.int32)
            equal = ttnn.eq(targets_i32[:, i:i + 1], proposal)
            ttnn.deallocate(proposal)
            if flags:
                chained = ttnn.multiply(flags[-1], equal)
                ttnn.deallocate(equal)
            else:
                chained = equal
            flags.append(chained)
            if accepted_count is None:
                accepted_count = ttnn.add(chained, 0)
            else:
                old = accepted_count
                accepted_count = ttnn.add(old, chained)
                ttnn.deallocate(old)
        return targets_i32, flags, accepted_count

    def commit_gdn(self, candidates, accepted_count):
        for layer, candidate in zip(self.model.layers, candidates):
            if not layer.is_full_attention:
                states, conv = candidate
                layer.attention.commit_verify_many(accepted_count, states, conv)

    def commit_kv(self, candidates, accepted_count):
        for layer, candidate in zip(self.model.layers, candidates):
            if layer.is_full_attention:
                layer.attention.commit_verify_many(candidate, accepted_count, self.positions, self.page_table)

    def set_page_table(self, row):
        row = row.to(torch.int32).reshape(1, -1)
        if row.shape[1] > self.width:
            raise ValueError("Page-table width exceeds allocated verifier")
        if row.shape[1] < self.width:
            row = torch.cat([row, torch.zeros((1, self.width - row.shape[1]), dtype=torch.int32)], dim=1)
        source = ttnn.from_torch(row.repeat(self.rows, 1), dtype=ttnn.int32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(source, self.page_table)

    def seed(self, token, position, page_table_row, proposals):
        if len(proposals) != self.depth:
            raise ValueError("A complete proposal block is required")
        self.set_page_table(page_table_row)
        tokens = ttnn.from_torch(torch.tensor([[token, *proposals]], dtype=torch.int32), dtype=ttnn.uint32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT)
        positions = ttnn.from_torch(torch.arange(position, position + self.rows, dtype=torch.int32),
                                    dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(tokens, self.tokens)
        ttnn.copy_host_to_device_tensor(positions, self.positions)
        return proposals

    def release(self):
        for tensor in (self.tokens, self.positions, self.page_table, *self.features.values()):
            ttnn.deallocate(tensor)
        self.features.clear()
