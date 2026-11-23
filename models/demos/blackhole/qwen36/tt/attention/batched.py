# SPDX-License-Identifier: Apache-2.0
"""Single-device concurrent decode; batch rows are independent requests, not MTP tokens.

The shared paged cache uses device positions (-1 skips a row) and one page-table
row per request. Both projections and SDPA run once for the complete batch.
BF16 layout kernels keep head preparation in DRAM and mask inactive outputs,
which native SDPA deliberately leaves unwritten. Existing B1/MTP paths do not
call this module; concurrent serving must call it even for its B1 bucket.
"""
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode

_KERNEL = str(Path(__file__).with_name("batched_layout_kernel.cpp").resolve())
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _layout_program(tensors, batch, heads, kv_heads, split):
    device = tensors[0].device()
    grid = device.compute_with_storage_grid_size()
    total = batch * 32 if split else ((batch + 31) // 32) * heads * 8
    workers = min(total, grid.x * grid.y)
    core_list = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    base, extra = divmod(total, workers)
    tasks = [(core, [i * base + min(i, extra), base + int(i < extra)])
             for i, core in enumerate(core_list)]
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # Assembled tile, 32 aligned row-pair staging slots, and optional positions.
    cb_sizes = [(0, 1), (1, 2)]
    if not split:
        cb_sizes.append((2, 1))
    cbs = [ttnn.CBDescriptor(
        total_size=count * 2048,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, 2048, tile)],
    ) for index, count in cb_sizes]
    addresses = [int(t.buffer_address()) for t in tensors]
    addresses += [0] * (5 - len(addresses))
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        defines=[("BATCH_SPLIT", "1" if split else "0")],
        brisc_common_runtime_args=addresses + [batch, heads, kv_heads],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=tasks, brisc_args=tasks, trisc_args=tasks,
        ),
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])


def _split_heads(qkv, batch, heads, kv_heads):
    outputs = [ttnn.empty(
        (1, batch, count, 256), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=qkv.device(), memory_config=_DRAM,
    ) for count in (heads, kv_heads, kv_heads, heads)]
    tensors = [qkv, *outputs]
    ttnn.generic_op(tensors, _layout_program(tensors, batch, heads, kv_heads, True))
    return outputs


def _concat_active_heads(gated, positions, batch, heads):
    output = ttnn.empty(
        (1, 1, batch, heads * 256), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=gated.device(), memory_config=_DRAM,
    )
    tensors = [gated, output, positions]
    ttnn.generic_op(tensors, _layout_program(tensors, batch, heads, 0, False))
    return output


def _validate(attention, x, cos, sin, positions, page_table):
    config = attention.config
    shape = tuple(int(v) for v in x.shape)
    batch = shape[-2]
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[-1] != 2048:
        raise ValueError("batched attention requires residual [1,1,B,2048]")
    if attention.device.get_num_devices() != 1:
        raise ValueError("batched attention requires a single device")
    grid = attention.device.compute_with_storage_grid_size()
    if not 1 <= batch <= grid.x * grid.y:
        raise ValueError("batched attention requires one cache-update core per batch row")
    if config.head_dim != 256 or not 1 <= config.num_kv_heads <= config.num_heads <= 32:
        raise ValueError("batched attention requires head_dim=256 and at most 32 heads")
    if not attention.use_paged_attention:
        raise ValueError("attach paged KV caches before batched attention")
    if attention.weights.qkv_fused is None:
        raise ValueError("batched attention requires QWEN36_FUSED_QKV=1 before loading weights")
    for name, tensor in (("x", x), ("cos", cos), ("sin", sin)):
        if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"batched attention {name} must be tiled BF16")
        if tensor.tile != ttnn.Tile((32, 32)):
            raise ValueError(f"batched attention {name} requires standard 32x32 tiles")
    if tuple(cos.shape) != (1, 1, batch, 64) or tuple(sin.shape) != (1, 1, batch, 64):
        raise ValueError("batched attention cos/sin must be [1,1,B,64]")
    for name, tensor in (("positions", positions), ("page_table", page_table)):
        if tensor is None or tensor.dtype != ttnn.int32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"batched attention {name} must be row-major int32")
        if tensor.memory_config() != _DRAM:
            raise ValueError(f"batched attention {name} must be interleaved DRAM")
    if tuple(positions.shape) != (batch,) or len(page_table.shape) != 2 or page_table.shape[0] != batch:
        raise ValueError("batched attention requires positions [B] and page_table [B,width]")
    return batch


def batched_decode_forward(attention, x, cos, sin, position_tensor, page_table):
    """Return BF16 DRAM [1,1,B,2048]; inactive rows are exactly zero.

    No input ownership is transferred. The external paged K/V buffers alone
    are mutated; -1 positions leave their pages untouched. Distinct live rows
    must not share a writable physical page (the serving allocator owns that
    invariant). All position/page-table reads occur on device and are traceable.
    """
    batch = _validate(attention, x, cos, sin, position_tensor, page_table)
    config, weights = attention.config, attention.weights
    heads, kv_heads = config.num_heads, config.num_kv_heads
    programs = attention.batched_program_configs.get(batch)
    if programs is None:
        from models.demos.blackhole.qwen36.tt.tp_common import create_matmul_1d_decode_progcfg
        grid_w = min(11, attention.device.compute_with_storage_grid_size().x)
        programs = {
            name: create_matmul_1d_decode_progcfg(
                batch, int(weight.shape[-2]), int(weight.shape[-1]), num_cores=cores, grid_w=grid_w,
            )
            for name, weight, cores in (("qkv", weights.qkv_fused, 64), ("o", weights.o_proj, 33))
        }
        attention.batched_program_configs[batch] = programs
    qkv = ttnn.linear(
        x, weights.qkv_fused, program_config=programs["qkv"],
        compute_kernel_config=attention.compute_kernel_config_decode,
        memory_config=_DRAM, dtype=ttnn.bfloat16,
    )
    q, k, v, gate = _split_heads(qkv, batch, heads, kv_heads)
    ttnn.deallocate(qkv)
    q_norm = ttnn.rms_norm(q, weight=weights.q_norm, epsilon=config.norm_eps, memory_config=_DRAM)
    k_norm = ttnn.rms_norm(k, weight=weights.k_norm, epsilon=config.norm_eps, memory_config=_DRAM)
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    q = apply_partial_rope_decode(q_norm, cos, sin, heads, batch, 64)
    k = apply_partial_rope_decode(k_norm, cos, sin, kv_heads, batch, 64)
    ttnn.deallocate(q_norm)
    ttnn.deallocate(k_norm)
    grid = attention.device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(batch, grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR)
    shard_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    for states, cache in ((k, attention.paged_kv_cache_key), (v, attention.paged_kv_cache_value)):
        sharded = ttnn.to_memory_config(states, shard_mc)
        ttnn.deallocate(states)
        ttnn.experimental.paged_update_cache(
            cache, sharded, update_idxs_tensor=position_tensor, page_table=page_table,
        )
        ttnn.deallocate(sharded)
    output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q, attention.paged_kv_cache_key, attention.paged_kv_cache_value,
        cur_pos_tensor=position_tensor, page_table_tensor=page_table,
        is_causal=True, scale=256 ** -0.5, memory_config=_DRAM,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False,
        ),
    )
    ttnn.deallocate(q)
    sigmoid_gate = ttnn.sigmoid(gate, memory_config=_DRAM)
    ttnn.deallocate(gate)
    gated = ttnn.multiply(output, sigmoid_gate, memory_config=_DRAM)
    ttnn.deallocate(output)
    ttnn.deallocate(sigmoid_gate)
    flat = _concat_active_heads(gated, position_tensor, batch, heads)
    ttnn.deallocate(gated)
    result = ttnn.linear(
        flat, weights.o_proj, program_config=programs["o"],
        compute_kernel_config=attention.compute_kernel_config_decode,
        memory_config=_DRAM, dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(flat)
    return result
