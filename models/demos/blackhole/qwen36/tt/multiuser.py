# SPDX-License-Identifier: Apache-2.0
"""Single-P150 multi-user decode for Qwen3.8-27B.

One decode step runs up to sixteen users as the rows of one tile-row pass: every matmul streams the
weights once for all users, and each row uses the wide-verification arithmetic with its own state:
- GDN layers read and update the row's own DRAM recurrent/convolution state (slot t = row t);
- attention writes the row's K/V at its own position through its own page-table row, and each
  grouped paged SDPA reads every row's own KV history.
Each user's tokens are bitwise those of the single-user decode.

Prefill is unchanged: a request is prefilled through the single-user path (live L1 state as the
scratch), then load_slot copies that state into its slot. Rows are slots, so a step of each width
uses fixed tensor addresses and is captured as an ordinary decode trace. Inactive rows read and
write the model's scratch KV block at position 0 and update their slot's state, which the next
load_slot overwrites.
"""
import torch
import ttnn

from models.demos.blackhole.qwen36.tt.attention.batched import USER_WIDTHS, release_groups, user_groups


def user_width(active):
    """Smallest step width holding `active` users."""
    for width in USER_WIDTHS:
        if active <= width:
            return width
    raise ValueError(f"at most {USER_WIDTHS[-1]} users per step")


class MultiUserDecoder:
    def __init__(self, model, slots, page_table_width, scratch_block):
        if slots not in USER_WIDTHS:
            raise ValueError(f"single-P150 multi-user serving requires max_num_seqs in {USER_WIDTHS}, got {slots}")
        if model.num_devices != 1 or not model._paged_kv_caches:
            raise RuntimeError("multi-user decode requires the single-P150 paged model")
        model.init_target_verification()
        self.model, self.device = model, model.device
        self.slots = slots
        self.widths = tuple(w for w in USER_WIDTHS if w <= slots)
        self.page_table_width = page_table_width
        self.scratch_block = scratch_block
        self.rope_deltas = [0] * slots
        dev = self.device
        # Unshifted RoPE tables (the verify tables are shifted in place per single-user request).
        self.cos_table = ttnn.from_torch(model._target_cos_host.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.sin_table = ttnn.from_torch(model._target_sin_host.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.gdn_layers = [layer for layer in model.layers if not layer.is_full_attention]
        self.recurrent = [
            [ttnn.from_torch(torch.zeros(1, 48, 128, 128, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
             for _ in range(slots)]
            for _ in self.gdn_layers
        ]
        self.conv = [
            [ttnn.from_torch(torch.zeros(1, 3, 10240, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
             for _ in range(slots)]
            for _ in self.gdn_layers
        ]

    def host_inputs(self, tokens, positions, page_table):
        """Host (tokens [1,W], positions [W], rope positions [W], page table [W,width]) for one step.

        W is the step width holding len(tokens) rows; rows with position < 0 and rows beyond the
        given ones are inactive (scratch block, position 0)."""
        tokens = torch.as_tensor(tokens).reshape(-1)
        positions = torch.as_tensor(positions).reshape(-1).to(torch.int64)
        rows = int(tokens.shape[0])
        width = user_width(rows)
        token_rows = torch.zeros((1, width), dtype=torch.int32)
        position_rows = torch.zeros(width, dtype=torch.int32)
        rope_rows = torch.zeros(width, dtype=torch.int32)
        table_rows = torch.full((width, self.page_table_width), self.scratch_block, dtype=torch.int32)
        table = None if page_table is None else torch.as_tensor(page_table).to(torch.int32)
        for row in range(rows):
            position = int(positions[row])
            if position < 0:
                continue
            token_rows[0, row] = int(tokens[row])
            position_rows[row] = position
            rope_rows[row] = position + self.rope_deltas[row]
            if table is not None:
                columns = min(self.page_table_width, int(table.shape[-1]))
                table_rows[row, :columns] = table[row, :columns]
        return (
            ttnn.from_torch(token_rows, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.from_torch(position_rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.from_torch(rope_rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.from_torch(table_rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        )

    def forward(self, tokens, positions, rope_positions, page_table):
        """Device-only decode step (trace-safe) over device inputs from host_inputs; logits [1, W, vocab]."""
        model = self.model
        width = int(positions.shape[0])
        idx = ttnn.reshape(ttnn.typecast(rope_positions, ttnn.uint32), [1, width])
        cos = ttnn.embedding(idx, self.cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idx, self.sin_table, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(idx)
        groups = user_groups(positions, page_table)
        x = model.embd(tokens)
        gdn_index = 0
        for index, layer in enumerate(model.layers):
            previous = x
            users = None
            if not layer.is_full_attention:
                users = (self.recurrent[gdn_index][:width], self.conv[gdn_index][:width])
                gdn_index += 1
            x = layer.users_forward(x, cos, sin, positions, page_table, groups, users,
                                    final_norm=model.verify_final_norm(index))
            ttnn.deallocate(previous)
        release_groups(groups)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        hidden = model.verify_hidden(x)
        if hidden is not x:
            ttnn.deallocate(x)
        logits = model._lm_head(hidden)
        ttnn.deallocate(hidden)
        return logits

    def load_slot(self, slot):
        """Copy the live single-user state (just prefilled) and its RoPE offset into `slot`."""
        model = self.model
        pool = model._deltanet_conv_pool
        pool_rows = ttnn.to_memory_config(pool, ttnn.DRAM_MEMORY_CONFIG) if pool is not None else None
        for index, layer in enumerate(self.gdn_layers):
            dn = layer.attention
            rows = ttnn.to_memory_config(dn.recurrent_state, ttnn.DRAM_MEMORY_CONFIG)
            tiles = ttnn.to_layout(ttnn.reshape(rows, [1, 48, 128, 128]), ttnn.TILE_LAYOUT,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(tiles, self.recurrent[index][slot])
            ttnn.deallocate(tiles)
            ttnn.deallocate(rows)
            if pool_rows is not None:
                conv = ttnn.slice(pool_rows, (0, 3 * index, 0), (1, 3 * index + 3, 10240))
            else:
                conv = ttnn.to_memory_config(dn.fused_conv_state, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(ttnn.reshape(conv, [1, 3, 10240]), self.conv[index][slot])
            ttnn.deallocate(conv)
        if pool_rows is not None:
            ttnn.deallocate(pool_rows)
        self.rope_deltas[slot] = int(model.rope.rope_delta)

    def remap(self, slot_remap):
        """vLLM batch condensing: slot i takes over slot_remap[i]'s request (destination <- source;
        out-of-range sources and identity rows are untouched), as Qwen36Model._remap_gdn_slots."""
        remap = torch.as_tensor(slot_remap).reshape(-1).tolist()
        moves = [(int(src), dst) for dst, src in enumerate(remap[: self.slots])
                 if 0 <= int(src) < self.slots and int(src) != dst]
        if not moves:
            return
        # Read every source before any destination is overwritten (moves may chain).
        staged = {src: ([ttnn.clone(states[src]) for states in self.recurrent],
                        [ttnn.clone(states[src]) for states in self.conv]) for src, _ in moves}
        deltas = list(self.rope_deltas)
        for src, dst in moves:
            recurrent, conv = staged[src]
            for states, value in zip(self.recurrent, recurrent):
                ttnn.copy(value, states[dst])
            for states, value in zip(self.conv, conv):
                ttnn.copy(value, states[dst])
            self.rope_deltas[dst] = deltas[src]
        for recurrent, conv in staged.values():
            for value in (*recurrent, *conv):
                ttnn.deallocate(value)

    def release(self):
        for value in (self.cos_table, self.sin_table, *(s for states in (*self.recurrent, *self.conv) for s in states)):
            ttnn.deallocate(value)
