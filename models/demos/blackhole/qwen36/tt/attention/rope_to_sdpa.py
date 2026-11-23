# SPDX-License-Identifier: Apache-2.0
"""Exact M1 BF16 RoPE directly into SDPA query and single-core KV layouts."""

from pathlib import Path

import os

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("rope_to_sdpa_kernel.cpp").resolve())
_TILE_BYTES = 2048
_BFP8_TILE_BYTES = 1088
_WORKERS = 56


def _shape(tensor, padded=False):
    return tuple(int(value) for value in (tensor.padded_shape if padded else tensor.shape))


def _validate(q, k, cos, sin, qkv, tokens=1):
    for name, tensor, shape, padded in (
        ("q", q, (1, tokens, 24, 256), (1, tokens, 32, 256)),
        ("k", k, (1, tokens, 4, 256), (1, tokens, 32, 256)),
        ("qkv", qkv, (1, tokens, 14336), (1, 32, 14336)),
    ):
        if _shape(tensor) != shape or _shape(tensor, True) != padded:
            raise ValueError(
                f"rope_to_sdpa {name}: unsupported logical/padded shape "
                f"{_shape(tensor)}/{_shape(tensor, True)}"
            )
        if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError(f"rope_to_sdpa {name}: requires interleaved DRAM")
    for name, tensor in (("cos", cos), ("sin", sin)):
        shape = _shape(tensor)
        if shape not in ((1, tokens, 64), (1, 1, tokens, 64)):
            raise ValueError(f"rope_to_sdpa {name}: expected [1,{tokens},64] or [1,1,{tokens},64], got {shape}")
        if _shape(tensor, True) != (*shape[:-2], 32, 64):
            raise ValueError(f"rope_to_sdpa {name}: unsupported padded shape {_shape(tensor, True)}")
        if tensor.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
            raise ValueError(f"rope_to_sdpa {name}: requires interleaved DRAM or L1")
    device = q.device()
    if device is None:
        raise ValueError("rope_to_sdpa requires device tensors")
    for name, tensor in (("q", q), ("k", k), ("cos", cos), ("sin", sin), ("qkv", qkv)):
        if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"rope_to_sdpa {name}: requires tiled BF16")
        if tensor.tile != ttnn.Tile((32, 32)):
            raise ValueError(f"rope_to_sdpa {name}: requires standard non-transposed 32x32 tiles")
        if tensor.device() != device:
            raise ValueError(f"rope_to_sdpa {name}: device differs from q")
    grid = device.compute_with_storage_grid_size()
    if grid.x < 11 or grid.y < 6:
        raise ValueError("rope_to_sdpa requires at least an 11x6 logical worker grid")


def _program(q, k, cos, sin, qkv, q_out, k_out, v_out, token=0, out_token=None):
    core_list = [ttnn.CoreCoord(index % 11, index // 11) for index in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # Original compute CBs unchanged. The writer owns CB18; KV readers/writers
    # alone consume CB19, so the 48 Q workers never wait for nonexistent V work.
    # 16 tiles = 32768 B/core; no CB aliases a global output shard.
    cb_tiles = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 16: 1, 17: 3, 18: 1, 19: 4}
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * _TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, _TILE_BYTES, tile)],
        )
        for index, count in cb_tiles.items()
    ]
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for index in (0, 2, 3, 4):
        unpack[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode.extend(unpack)
    tasks = [(core, [index]) for index, core in enumerate(core_list)]
    defines = [
        (f"ROPE_{name}_DRAM", "1" if tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG else "0")
        for name, tensor in (("Q", q), ("K", k), ("COS", cos), ("SIN", sin))
    ]
    if token:
        defines.append(("ROPE_TOKEN", str(token)))
    if out_token is not None and out_token != token:
        defines.append(("ROPE_OUT_TOKEN", str(out_token)))
    # Never assume logical (0,0) is the physical NoC location of the KV shard.
    shard_core = q.device().worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        defines=defines,
        ncrisc_common_runtime_args=[int(t.buffer_address()) for t in (q, k, cos, sin, qkv)],
        brisc_common_runtime_args=[int(t.buffer_address()) for t in (q_out, k_out, v_out)]
        + [int(shard_core.x), int(shard_core.y)],
        trisc_compute_config=compute,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=tasks,
            brisc_args=tasks,
            trisc_args=tasks,
        ),
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])


def rope_to_sdpa(q, k, cos, sin, qkv):
    """Return DRAM Q and single-core height-sharded L1 K/V for B1/T1 decode.

    Q/K are normalized head-row tensors, not original [1,heads,1,256]
    tensors. QKV must remain the raw packed projection: V starts at column
    13312. The caller retains every input through the enqueued operation and
    owns/deallocates the three returned tensors after their consumers finish.
    No inputs are deallocated here and no persistent buffers are allocated.

    Empty allocations are intentional: this kernel writes every physical
    output element. Padding rows are zeroed by separate row owners without
    overlapping live-head writes; no prezero operation or scratch tensor is
    needed. All DMA source CBs remain reserved until the respective barrier.
    """
    _validate(q, k, cos, sin, qkv)
    device = q.device()
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    allocation = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    q_out = ttnn.empty((1, 1, 24, 256), memory_config=ttnn.DRAM_MEMORY_CONFIG, **allocation)
    k_out = ttnn.empty((1, 1, 4, 256), memory_config=sharded_mc, **allocation)
    v_out = ttnn.empty((1, 1, 4, 256), memory_config=sharded_mc, **allocation)
    ttnn.generic_op(
        [q, k, cos, sin, qkv, q_out, k_out, v_out],
        _program(q, k, cos, sin, qkv, q_out, k_out, v_out),
    )
    return q_out, k_out, v_out


def rope_to_sdpa_two(q, k, cos, sin, qkv, token, q_out):
    """M=2 verification RoPE for one token row of [1,2,...] Q/K/cos/sin/QKV.

    Writes token `token` into tiles [8*token, 8*token+8) of the caller's DRAM
    q_out [1,2,24,256] (both launches together fully cover it) and returns this
    token's single-core height-sharded L1 K/V [1,1,4,256] for paged_update_cache.
    Same arithmetic and rounding boundaries as the M1 launch.
    """
    if token not in (0, 1):
        raise ValueError("rope_to_sdpa_two token must be 0 or 1")
    _validate(q, k, cos, sin, qkv, tokens=2)
    if _shape(q_out) != (1, 2, 24, 256) or q_out.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("rope_to_sdpa_two requires a DRAM [1,2,24,256] q_out")
    device = q.device()
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    allocation = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_out = ttnn.empty((1, 1, 4, 256), memory_config=sharded_mc, **allocation)
    v_out = ttnn.empty((1, 1, 4, 256), memory_config=sharded_mc, **allocation)
    ttnn.generic_op(
        [q, k, cos, sin, qkv, q_out, k_out, v_out],
        _program(q, k, cos, sin, qkv, q_out, k_out, v_out, token=token),
    )
    return k_out, v_out


def rope_to_sdpa_row(q, k, cos, sin, qkv, token, q_out, out_token):
    """Wide (T=3/4) verification RoPE for token row `token` of [1,T,...] Q/K/cos/sin/QKV.

    Writes the token's query into tiles [8*out_token, 8*out_token+8) of the
    caller's DRAM q_out [1,G,24,256] (one SDPA group) and returns this token's
    single-core height-sharded L1 K/V [1,1,4,256]. Same arithmetic as M1/M2.
    """
    tokens = _shape(q)[1]
    if tokens not in (3, 4) or not 0 <= token < tokens:
        raise ValueError("rope_to_sdpa_row requires T=3/4 and a token row inside T")
    _validate(q, k, cos, sin, qkv, tokens=tokens)
    group = _shape(q_out)[1]
    if (_shape(q_out) != (1, group, 24, 256) or not 0 <= out_token < group
            or q_out.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
        raise ValueError("rope_to_sdpa_row requires a DRAM [1,G,24,256] q_out containing out_token")
    device = q.device()
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    allocation = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_out = ttnn.empty((1, 1, 4, 256), memory_config=sharded_mc, **allocation)
    v_out = ttnn.empty((1, 1, 4, 256), memory_config=sharded_mc, **allocation)
    ttnn.generic_op(
        [q, k, cos, sin, qkv, q_out, k_out, v_out],
        _program(q, k, cos, sin, qkv, q_out, k_out, v_out, token=token, out_token=out_token),
    )
    return k_out, v_out


_KV_MANY_KERNEL = str(Path(__file__).with_name("rope_kv_many_kernel.cpp").resolve())


# Rows per wide-verification SDPA launch. Every group keeps the reference split of eight cores
# per (row, KV head): 3 rows x 4 heads x 8 = 96 of 99 cores (11x9 grid), so 8 rows need 3 launches.
VERIFY_SDPA_GROUP = int(os.environ.get("QWEN36_VERIFY_SDPA_GROUP", "3"))


def verify_row_groups(tokens):
    return [tuple(range(start, min(start + VERIFY_SDPA_GROUP, tokens))) for start in range(0, tokens, VERIFY_SDPA_GROUP)]


def rope_kv_many(q, k, cos, sin, qkv, q_groups, cache_key, cache_value, positions, page_table):
    """Verification RoPE (T=2..8) for every token row in one launch.

    Token t's query goes to row t%G of q_groups[t//G] (DRAM [1,G,24,256] SDPA groups of
    consecutive rows, T>=3), or, when q_groups is one packed tensor (verify_sdpa.packed_q_shape(T)),
    to row (t%4)*6 + head%6 of tile row (head//6)*B + t//4 (B = ceil(T/4)); its K/V head rows are written directly
    into the paged caches at positions[t] through page_table row t. Same arithmetic as
    rope_to_sdpa_row, and the same cache bytes as rope_to_sdpa_row + paged_update_cache per row.
    BFP8 caches (long-context serving) receive each row through the packer's BF16->BFP8
    conversion, patched into its tile (one shared exponent per 16-element face row, so other rows
    keep their bytes).
    """
    tokens = _shape(q)[1]
    q_packed = isinstance(q_groups, ttnn.Tensor)
    if not (2 if q_packed else 3) <= tokens <= 8:
        raise ValueError("rope_kv_many requires T=3..8 (T=2..8 with packed Q)")
    _validate(q, k, cos, sin, qkv, tokens=tokens)
    if q_packed:
        from models.demos.blackhole.qwen36.tt.attention.verify_sdpa import packed_q_shape

        if (_shape(q_groups) != packed_q_shape(tokens) or q_groups.dtype != ttnn.bfloat16
                or q_groups.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError(f"rope_kv_many packed Q requires DRAM BF16 {packed_q_shape(tokens)}")
        q_groups = [q_groups]
    else:
        expected_groups = [len(rows) for rows in verify_row_groups(tokens)]
        if [(_shape(g)[1]) for g in q_groups] != expected_groups or any(
            _shape(g) != (1, _shape(g)[1], 24, 256) or g.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            for g in q_groups
        ):
            raise ValueError("rope_kv_many requires DRAM [1,G,24,256] query groups of consecutive VERIFY_SDPA_GROUP rows")
    cache_bfp8 = cache_key.dtype == ttnn.bfloat8_b
    for cache in (cache_key, cache_value):
        shape = _shape(cache)
        if (cache.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b) or cache.dtype != cache_key.dtype
                or cache.layout != ttnn.TILE_LAYOUT or len(shape) != 4
                or (shape[1], shape[3]) != (4, 256) or shape[2] % 32
                or cache.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError("rope_kv_many requires tiled BF16/BFP8 DRAM paged caches [blocks,4,block,256]")
    if (tuple(positions.shape) != (tokens,) or positions.dtype != ttnn.int32
            or positions.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or len(page_table.shape) != 2 or int(page_table.shape[0]) != tokens
            or page_table.dtype != ttnn.int32 or page_table.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
        raise ValueError("rope_kv_many requires int32 DRAM positions [T] and page_table [T,W]")
    core_list = [ttnn.CoreCoord(index % 11, index // 11) for index in range(_WORKERS)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # Two-deep row operands: the reader gathers token t+1 while compute/writer finish token t.
    cb_tiles = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 16: 2, 17: 6, 18: 1, 19: 8, 20: 1}
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * _TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, _TILE_BYTES, tile)],
        )
        for index, count in cb_tiles.items()
    ]
    if cache_bfp8:
        # 21: eight packed BFP8 row tiles per KV token (K rotated half, 3 K pass, 4 V);
        # 22: one cache tile staged for the row patch (read-modify-write).
        for index, count in ((21, 8), (22, 1)):
            cbs.append(ttnn.CBDescriptor(
                total_size=count * _BFP8_TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat8_b, _BFP8_TILE_BYTES, tile)],
            ))
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for index in (0, 2, 3, 4):
        unpack[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode.extend(unpack)
    tasks = [(core, [index]) for index, core in enumerate(core_list)]
    defines = [
        (f"ROPE_{name}_DRAM", "1" if tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG else "0")
        for name, tensor in (("Q", q), ("K", k), ("COS", cos), ("SIN", sin))
    ] + [
        ("ROPE_TOKENS", str(tokens)),
        ("ROPE_GROUP", str(VERIFY_SDPA_GROUP)),
        ("ROPE_PAGE_ROWS", str(int(cache_key.shape[2]))),
        ("ROPE_POSITION_BYTES", str(int(positions.padded_shape[-1]) * 4)),
        ("ROPE_TABLE_BYTES", str(int(page_table.padded_shape[-1]) * 4)),
    ] + ([("ROPE_CACHE_BFP8", "1")] if cache_bfp8 else []) + ([("ROPE_Q_PACKED", "1")] if q_packed else [])
    groups = list(q_groups) + [q_groups[-1]] * (4 - len(q_groups))
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KV_MANY_KERNEL,
        core_ranges=cores,
        defines=defines,
        ncrisc_common_runtime_args=[int(t.buffer_address()) for t in (q, k, cos, sin, qkv)],
        brisc_common_runtime_args=[int(t.buffer_address()) for t in (*groups, cache_key, cache_value, positions,
                                                                      page_table)],
        trisc_compute_config=compute,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=tasks,
            brisc_args=tasks,
            trisc_args=tasks,
        ),
    )
    program = ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])
    ttnn.generic_op([q, k, cos, sin, qkv, *q_groups, cache_key, cache_value, positions, page_table], program)
