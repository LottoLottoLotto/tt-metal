# SPDX-License-Identifier: Apache-2.0
"""Exact M1/M2 BF16 RoPE directly into SDPA query and single-core KV layouts."""

from pathlib import Path

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("rope_to_sdpa_kernel.cpp").resolve())
_TILE_BYTES = 2048
_HEAD_DIM = 256


def _shape(tensor, padded=False):
    return tuple(int(value) for value in (tensor.padded_shape if padded else tensor.shape))


def _workers(num_heads, num_kv_heads, tokens=1):
    """One worker per (token, head, rotary half); per token they also cover every padding row."""
    workers = 2 * (num_heads + num_kv_heads)
    if not (1 <= num_kv_heads <= num_heads <= 32) or workers < 32 - num_kv_heads or tokens * workers > 110:
        raise ValueError(f"rope_to_sdpa does not support {tokens}x{num_heads} query / {num_kv_heads} KV heads")
    return workers


def _validate(q, k, cos, sin, qkv, num_heads, num_kv_heads):
    width = (2 * num_heads + 2 * num_kv_heads) * _HEAD_DIM
    tokens = _shape(q)[0]
    if tokens not in (1, 2):
        raise ValueError(f"rope_to_sdpa supports one or two tokens, got {tokens}")
    for name, tensor, shape, padded in (
        ("q", q, (tokens, 1, num_heads, _HEAD_DIM), (tokens, 1, 32, _HEAD_DIM)),
        ("k", k, (tokens, 1, num_kv_heads, _HEAD_DIM), (tokens, 1, 32, _HEAD_DIM)),
        ("qkv", qkv, (1, tokens, width), (1, 32, width)),
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
    if grid.x < 11 or grid.x * grid.y < tokens * _workers(num_heads, num_kv_heads, tokens):
        raise ValueError("rope_to_sdpa needs an 11-wide worker grid with one core per (token, head, half)")
    return tokens


def _program(q, k, cos, sin, qkv, q_out, kv_out, num_heads, num_kv_heads, tokens):
    per_token = _workers(num_heads, num_kv_heads, tokens)
    core_list = [ttnn.CoreCoord(index % 11, index // 11) for index in range(tokens * per_token)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in core_list})
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    # Compute CBs as in the original; the writer owns CB18; KV readers/writers alone
    # consume CB19, so query workers never wait for nonexistent V work.
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
    tasks = [
        (core, [index % per_token, num_heads, num_kv_heads, index // per_token])
        for index, core in enumerate(core_list)
    ]
    defines = [
        (f"ROPE_{name}_DRAM", "1" if tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG else "0")
        for name, tensor in (("Q", q), ("K", k), ("COS", cos), ("SIN", sin))
    ]
    # Never assume logical (0,0) is the physical NoC location of the KV shard.
    shard_core = q.device().worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    k0, v0 = kv_out[0]
    k1, v1 = kv_out[-1]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        defines=defines,
        ncrisc_common_runtime_args=[int(t.buffer_address()) for t in (q, k, cos, sin, qkv)],
        brisc_common_runtime_args=[int(t.buffer_address()) for t in (q_out, k0, v0)]
        + [int(shard_core.x), int(shard_core.y)]
        + [int(t.buffer_address()) for t in (k1, v1)],
        trisc_compute_config=compute,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=tasks,
            brisc_args=tasks,
            trisc_args=tasks,
        ),
    )
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])


def rope_to_sdpa(q, k, cos, sin, qkv, num_heads, num_kv_heads):
    """Return DRAM Q and single-core height-sharded L1 K/V for B1 decode (T=1) or verify (T=2).

    Q/K are normalized head-row tensors [T,1,heads,256], not original [1,heads,T,256]
    tensors. QKV must remain the raw packed projection [1,T,W]: V starts after the Q+gate
    and K columns. The caller retains every input through the enqueued operation and
    owns/deallocates the returned tensors after their consumers finish.
    No inputs are deallocated here and no persistent buffers are allocated.

    Returns ``(q_out, k, v)`` for T=1 and ``(q_out, (k0, v0), (k1, v1))`` for T=2, where
    q_out is [T,1,H,256] (reshapes to paged SDPA's [1,T,H,256]) and each K/V is its token's
    [1,1,Hkv,256] single-core shard for ``paged_update_cache``.

    Empty allocations are intentional: this kernel writes every physical
    output element. Padding rows are zeroed by separate row owners without
    overlapping live-head writes; no prezero operation or scratch tensor is
    needed. All DMA source CBs remain reserved until the respective barrier.
    """
    tokens = _validate(q, k, cos, sin, qkv, num_heads, num_kv_heads)
    device = q.device()
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, [32, _HEAD_DIM], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    allocation = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    q_out = ttnn.empty((tokens, 1, num_heads, _HEAD_DIM), memory_config=ttnn.DRAM_MEMORY_CONFIG, **allocation)
    kv_out = [
        tuple(ttnn.empty((1, 1, num_kv_heads, _HEAD_DIM), memory_config=sharded_mc, **allocation) for _ in range(2))
        for _ in range(tokens)
    ]
    ttnn.generic_op(
        [q, k, cos, sin, qkv, q_out, *(t for pair in kv_out for t in pair)],
        _program(q, k, cos, sin, qkv, q_out, kv_out, num_heads, num_kv_heads, tokens),
    )
    if tokens == 1:
        return q_out, kv_out[0][0], kv_out[0][1]
    return q_out, kv_out[0], kv_out[1]
