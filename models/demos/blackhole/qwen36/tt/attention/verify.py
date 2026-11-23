# SPDX-License-Identifier: Apache-2.0
"""Causal multi-row attention with bitwise, accepted-prefix KV transactions."""

from dataclasses import dataclass
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("verify_kv_kernel.cpp").resolve())


@dataclass
class AttentionVerifyTransaction:
    """Single-use candidate rows; verify has already restored the live cache.

    Dropping an uncommitted transaction is safe. Commit must precede any other
    operation changing this sequence's cache or its position/page-table buffers.
    Keep metadata buffers unchanged until commit has been enqueued. All transfers
    execute on the same command queue as native cache update and SDPA.
    """

    cache_key: object
    cache_value: object
    key: object
    value: object
    positions: object
    page_table: object
    consumed: bool = False
    fork_position: int = 0


def _validate_metadata(positions, page_table, tokens):
    for name, tensor, shape in (
        ("positions", positions, (tokens,)),
        ("page_table", page_table, (tokens, int(page_table.shape[-1]))),
    ):
        if (
            tuple(tensor.shape) != shape or tensor.dtype != ttnn.int32
            or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError(f"verify_many {name} must be int32 row-major DRAM {shape}")


def _transfer_rows(cache_key, cache_value, key, value, positions, page_table, mode, accepted_count=None,
                   fork_position=0, winner=None):
    """Mode 0 saves rows, 1 restores all rows, 2 commits rows 0..device count.

    DMA only: no BF16 unpack/pack, no full-cache or full-page copies. Both metadata
    lookup and acceptance selection stay on device and are trace-replay dynamic.
    """
    tokens = int(key.shape[-2])
    heads, width = int(key.shape[1]), int(key.shape[-1])
    device = cache_key.device()
    grid = device.compute_with_storage_grid_size()
    workers = heads * (width // 32)
    if workers > grid.x * grid.y:
        raise ValueError("verify_many KV transfer needs one worker per head/dimension tile")
    core_list = [ttnn.CoreCoord(index % grid.x, index // grid.x) for index in range(workers)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    tasks = [(core, [index]) for index, core in enumerate(core_list)]
    flag = accepted_count if accepted_count is not None else positions
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    branch = winner if winner is not None else flag
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        brisc_named_compile_time_args=[
            ("tokens", tokens), ("heads", heads), ("width_tiles", width // 32),
            ("page_rows", int(cache_key.shape[2])),
            ("position_bytes", int(positions.padded_shape[-1]) * 4),
            ("table_bytes", int(page_table.padded_shape[-1]) * 4),
            ("mode", mode), ("flag_dram", int(flag.memory_config() == ttnn.DRAM_MEMORY_CONFIG)),
            ("fork_position", fork_position),
            ("winner_dram", int(branch.memory_config() == ttnn.DRAM_MEMORY_CONFIG)),
        ],
        brisc_common_runtime_args=[int(t.buffer_address()) for t in (
            cache_key, cache_value, key, value, positions, page_table, flag, branch
        )],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=False, dst_full_sync_en=False,
        ),
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=tasks, brisc_args=tasks, trisc_args=tasks,
        ),
    )
    program = ttnn.ProgramDescriptor(
        kernels=kernel.get_kernel_descriptors().kernels,
        cbs=[ttnn.CBDescriptor(
            total_size=2048, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048, tile)],
        )],
        semaphores=[],
    )
    ttnn.generic_op([cache_key, cache_value, key, value, positions, page_table, flag, branch], program)


def causal_verify_many(query, key, value, cache_key, cache_value, positions, page_table, scaling, program_config,
                      fork_position=0):
    """Native B=1 KV/SDPA per causal row; QKV and output projections stay outside.

    Profiler boundaries: _transfer_rows(mode=0) snapshot; each loop iteration's
    two paged_update_cache calls (KV), then paged_scaled_dot_product_attention_decode
    (SDPA); _transfer_rows(mode=1) rollback. Commit is mode=2. No host synchronization.
    """
    tokens = int(query.shape[-2])
    if fork_position and (not 1 <= fork_position <= 7 or tokens != 16 - fork_position):
        raise ValueError("Tree attention requires N=16-f and f in 1..7")
    if not 2 <= tokens <= 32 or int(query.shape[0]) != 1:
        raise ValueError("verify_many requires one sequence with 2..32 rows")
    _validate_metadata(positions, page_table, tokens)
    for tensor in (query, key, value, cache_key, cache_value):
        if (
            tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT
            or tensor.tile != ttnn.Tile((32, 32))
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("verify_many requires standard tiled BF16 interleaved DRAM Q/K/V and caches")
    if (
        tuple(key.shape) != tuple(value.shape) or tuple(cache_key.shape) != tuple(cache_value.shape)
        or int(key.shape[0]) != 1 or int(key.shape[-2]) != tokens
        or tuple(key.padded_shape) != (1, int(key.shape[1]), 32, int(key.shape[-1]))
        or int(key.shape[1]) != int(cache_key.shape[1])
        or int(key.shape[-1]) != int(cache_key.shape[-1])
        or int(cache_key.shape[2]) % 32 != 0 or int(key.shape[-1]) % 32 != 0
    ):
        raise ValueError("verify_many requires matching KV heads/width and tile-aligned cache pages")
    # The frozen target reference uses B=2 SDPA. Serial B=1 calls must keep
    # its cores per KV head: more cores change the softmax reduction tree
    # once the context exceeds the original per-core chunk budget.
    # The attention caller supplies a fresh program configuration each time.
    grid = program_config.compute_with_storage_grid_size
    reference_cores_per_head = max(1, grid.x * grid.y // (2 * int(key.shape[1])))
    program_config.max_cores_per_head_batch = min(
        program_config.max_cores_per_head_batch, reference_cores_per_head
    )
    saved_key = ttnn.empty(key.shape, dtype=key.dtype, layout=key.layout, device=key.device(),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    saved_value = ttnn.empty(value.shape, dtype=value.dtype, layout=value.layout, device=value.device(),
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _transfer_rows(cache_key, cache_value, saved_key, saved_value, positions, page_table, 0)
    device = query.device()
    shard = ttnn.num_cores_to_corerangeset(1, device.compute_with_storage_grid_size(), row_wise=True)
    sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(shard, [32, int(key.shape[-1])], ttnn.ShardOrientation.ROW_MAJOR),
    )
    rows = []
    try:
        # A rows first, then B's suffix. At B_f the causal frontier moves back
        # to f; stale A suffix is strictly beyond it and is overwritten before
        # becoming visible. Shared prefix KV and its SDPA are evaluated once.
        for row in range(tokens):
            position = positions[row : row + 1]
            table = page_table[row : row + 1, :]
            # Match M1 exactly: [1,H,1,D] -> [1,1,H,D], then native cache update.
            for cache, candidate in ((cache_key, key), (cache_value, value)):
                current = candidate[:, :, row : row + 1, :]
                current = ttnn.reshape(current, [1, 1, int(key.shape[1]), int(key.shape[-1])])
                current = ttnn.to_memory_config(current, sharded)
                ttnn.experimental.paged_update_cache(cache, current, update_idxs_tensor=position, page_table=table)
                ttnn.deallocate(current)
            q = query[:, :, row : row + 1, :]
            q = ttnn.reshape(q, [1, 1, int(query.shape[1]), int(query.shape[-1])])
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, cache_key, cache_value, cur_pos_tensor=position, page_table_tensor=table,
                is_causal=True, scale=scaling, program_config=program_config,
            )
            rows.append(ttnn.reshape(output, [1, int(query.shape[1]), 1, int(query.shape[-1])]))
    finally:
        # Restore every overwritten slot, even on host-side failures. Rejected rows
        # never persist and committed history outside these exact slots is untouched.
        _transfer_rows(cache_key, cache_value, saved_key, saved_value, positions, page_table, 1)
        ttnn.deallocate(saved_key)
        ttnn.deallocate(saved_value)
    output = ttnn.concat(rows, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    transaction = AttentionVerifyTransaction(
        cache_key, cache_value, key, value, positions, page_table, fork_position=fork_position
    )
    return output, transaction


def commit_verify_many(transaction, accepted_count, position_tensor, page_table, cache_key, cache_value,
                       winner=None):
    """Consume one transaction, writing exactly its accepted prefix on device."""
    if transaction.consumed:
        raise ValueError("Attention verification transaction has already been committed")
    if transaction.cache_key is not cache_key or transaction.cache_value is not cache_value:
        raise ValueError("Attention verification transaction belongs to different KV caches")
    if transaction.positions is not position_tensor or transaction.page_table is not page_table:
        raise ValueError("Attention verification commit must use its original metadata buffers")
    if (
        tuple(accepted_count.shape) != (1, 1) or accepted_count.dtype != ttnn.int32
        or accepted_count.layout != ttnn.ROW_MAJOR_LAYOUT
        or accepted_count.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
    ):
        raise ValueError("Attention accepted_count must be interleaved row-major int32 [1,1]")
    if transaction.fork_position and (
        winner is None or tuple(winner.shape) != (1, 1) or winner.dtype != ttnn.int32
        or winner.layout != ttnn.ROW_MAJOR_LAYOUT
        or winner.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
    ):
        raise ValueError("Tree attention commit requires an int32 device winner [1,1]")
    _transfer_rows(cache_key, cache_value, transaction.key, transaction.value,
                   position_tensor, page_table, 2, accepted_count,
                   fork_position=transaction.fork_position, winner=winner)
    transaction.consumed = True
