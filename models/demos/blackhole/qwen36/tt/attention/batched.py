# SPDX-License-Identifier: Apache-2.0
"""Multi-user decode attention: row t of one tile-row pass is user t.

The arithmetic is the wide-verification grouped path (fused QKV, split_qkg_many, Q/K RMSNorm,
rope_kv_many, grouped paged SDPA, merge_group_rows, sigmoid gate, output projection), with each
row's K/V written at its own user's position through its own page-table row and each SDPA group
reading every row's own KV history (positions[group] / page_table[group]).

The per-row kernels (split_qkg_many, rope_kv_many, merge_group_rows) take at most 8 rows, so a
16-user step runs them on two 8-row halves; the QKV and output projections still run once over
all rows, so every weight is read once per step.
"""
import ttnn

from models.demos.blackhole.qwen36.tt.attention.merge_rows import merge_group_rows
from models.demos.blackhole.qwen36.tt.attention.rope_to_sdpa import rope_kv_many, verify_row_groups
from models.demos.blackhole.qwen36.tt.attention.split_qkg import split_qkg_many
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
    _get_paged_sdpa_decode_program_config,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _prefetch_linear

USER_WIDTHS = (4, 8, 16)
HALF = 8


def user_groups(positions, page_table):
    """Per-step metadata shared by every attention layer: one entry per block of at most 8 rows,
    (start, rows, positions, page table, SDPA groups of (rows, positions, page-table rows))."""
    total = int(positions.shape[0])
    blocks = []
    for start in range(0, total, HALF):
        rows = min(HALF, total - start)
        block_positions = positions[start:start + rows] if total > HALF else positions
        block_table = page_table[start:start + rows, :] if total > HALF else page_table
        groups = tuple(
            (group, block_positions[group[0]:group[-1] + 1], block_table[group[0]:group[-1] + 1, :])
            for group in verify_row_groups(rows)
        )
        blocks.append((start, rows, block_positions, block_table, groups))
    return tuple(blocks)


def release_groups(blocks):
    total = sum(rows for _, rows, _, _, _ in blocks)
    for _, _, block_positions, block_table, groups in blocks:
        for _, group_positions, group_table in groups:
            ttnn.deallocate(group_positions)
            ttnn.deallocate(group_table)
        if total > HALF:
            ttnn.deallocate(block_positions)
            ttnn.deallocate(block_table)


def _linear(attention, x, weight, program_config):
    # The optimized decode_linear of gated_attention_forward_ttnn for <= 32 rows.
    shape = list(x.shape)
    rows = 1
    for dim in shape[:-1]:
        rows *= int(dim)
    if x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    out = _prefetch_linear(
        ttnn.reshape(x, [1, 1, rows, shape[-1]]), weight,
        compute_kernel_config=attention.compute_kernel_config_decode,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=program_config, dtype=ttnn.bfloat16,
    )
    return ttnn.reshape(out, shape[:-1] + [out.shape[-1]])


def _rows(tensor, start, rows):
    """Rows [start, start+rows) of a [1,T,W] tensor (a bit copy)."""
    width = int(tensor.shape[-1])
    return ttnn.slice(tensor, (0, start, 0), (1, start + rows, width), memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _attend_block(attention, qkv, cos, sin, positions, page_table, groups):
    """split, norms, RoPE + KV rows, grouped SDPA and merge for one block of 3..8 rows."""
    weights, config = attention.weights, attention.config
    key_cache, value_cache = attention.paged_kv_cache_key, attention.paged_kv_cache_value
    query, key, gate = split_qkg_many(qkv)
    query_normed = ttnn.rms_norm(query, weight=weights.q_norm, epsilon=config.norm_eps)
    ttnn.deallocate(query)
    key_normed = ttnn.rms_norm(key, weight=weights.k_norm, epsilon=config.norm_eps)
    ttnn.deallocate(key)
    heads, head_dim, kv_heads = config.num_heads, config.head_dim, config.num_kv_heads
    group_queries = [
        ttnn.empty((1, len(rows), heads, head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                   device=attention.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for rows, _, _ in groups
    ]
    rope_kv_many(query_normed, key_normed, cos, sin, qkv, group_queries, key_cache, value_cache, positions, page_table)
    ttnn.deallocate(query_normed)
    ttnn.deallocate(key_normed)
    max_seq = key_cache.shape[0] * key_cache.shape[2]
    outputs = []
    for (rows, group_positions, group_table), q_group in zip(groups, group_queries):
        program_config = _get_paged_sdpa_decode_program_config(attention.device, max_seq)
        # The wide-verification split: 8 cores per (row, KV head); three rows on an 11x9 grid.
        reference = program_config.compute_with_storage_grid_size
        program_config.max_cores_per_head_batch = min(
            program_config.max_cores_per_head_batch, max(1, reference.x * reference.y // (2 * kv_heads))
        )
        if len(rows) == 3:
            program_config.compute_with_storage_grid_size = ttnn.CoreCoord(11, 9)
        outputs.append(ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_group, key_cache, value_cache, cur_pos_tensor=group_positions, page_table_tensor=group_table,
            is_causal=True, scale=head_dim**-0.5, program_config=program_config,
        ))
        ttnn.deallocate(q_group)
    attended = merge_group_rows(outputs)
    for out in outputs:
        ttnn.deallocate(out)
    gate_act = ttnn.sigmoid(gate)
    ttnn.deallocate(gate)
    gated = ttnn.multiply(attended, gate_act)
    ttnn.deallocate(gate_act)
    ttnn.deallocate(attended)
    return gated


def users_forward(attention, x, cos, sin, positions, page_table, blocks):
    """x [1,U,5120] (U = 4, 8 or 16 users) -> [1,U,5120]; writes each user's K/V row into the live cache."""
    users = int(x.shape[1])
    if len(x.shape) != 3 or int(x.shape[0]) != 1 or users not in USER_WIDTHS:
        raise ValueError(f"multi-user attention requires [1,U,5120] with U in {USER_WIDTHS}")
    if [(start, rows) for start, rows, _, _, _ in blocks] != [(s, min(HALF, users - s)) for s in range(0, users, HALF)]:
        raise ValueError("multi-user attention blocks must cover every row in 8-row blocks")
    for _, rows, _, _, groups in blocks:
        if [group for group, _, _ in groups] != verify_row_groups(rows):
            raise ValueError("multi-user attention groups must cover every row in VERIFY_SDPA_GROUP order")
    configs = attention.decode_program_configs
    qkv = _linear(attention, x, attention.weights.qkv_fused, configs.get("qkv"))
    if users <= HALF:
        gated = _attend_block(attention, qkv, cos, sin, positions, page_table, blocks[0][4])
    else:
        halves = []
        for start, rows, block_positions, block_table, groups in blocks:
            block_qkv, block_cos, block_sin = (_rows(t, start, rows) for t in (qkv, cos, sin))
            halves.append(_attend_block(attention, block_qkv, block_cos, block_sin, block_positions, block_table, groups))
            for tensor in (block_qkv, block_cos, block_sin):
                ttnn.deallocate(tensor)
        gated = ttnn.concat(halves, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for half in halves:
            ttnn.deallocate(half)
    ttnn.deallocate(qkv)
    output = _linear(attention, gated, attention.weights.o_proj, configs.get("o"))
    ttnn.deallocate(gated)
    return output
