# SPDX-License-Identifier: Apache-2.0
"""One-launch paged K/V write for the fused-layout B1 decode (T=1) and MTP-1 verification (T=2) attention.

Replaces, per token, the position/page-table slices and the K and V ``paged_update_cache`` calls, and (T=2) the
copies of token 1's K/V shards to DRAM (the commit candidates), with one 2*Hkv-core program (kv_write_kernel.cpp).
BF16 or BFP8_b caches (the serving image sets QWEN_SDPA_BF8=1): BF16 rows are copied, BFP8_b rows are packed in the
kernel with the block-float rounding of the stock path (validated bitwise by tools/kvwrite_ab.py).
"""
import ttnn

_KERNEL = "models/demos/blackhole/qwen36/tt/attention/kv_write_kernel.cpp"
_TILE_BYTES = 2048


def _accessor_args(tensor):
    accessor = ttnn.TensorAccessorArgs(tensor)
    if accessor.get_common_runtime_args():
        raise ValueError("kv_write accessors require compile-time metadata")
    return list(accessor.get_compile_time_args())


def supported(k_cache, v_cache, keys, positions, page_table):
    """The fused write handles interleaved BF16 TILE paged caches, single-core K/V shards and interleaved int32
    row-major positions [>=T] / page table [>=T, W]."""
    shape = list(k_cache.shape)
    tokens = len(keys)
    index_ok = (
        positions.dtype == ttnn.int32 and page_table.dtype == ttnn.int32
        and positions.layout == ttnn.ROW_MAJOR_LAYOUT and page_table.layout == ttnn.ROW_MAJOR_LAYOUT
        and not positions.is_sharded() and not page_table.is_sharded()
        and int(positions.shape[-1]) >= tokens and len(page_table.shape) == 2 and int(page_table.shape[0]) >= tokens
    )
    cache_ok = (
        k_cache.dtype in (ttnn.bfloat16, ttnn.bfloat8_b) and v_cache.dtype == k_cache.dtype
        and k_cache.layout == ttnn.TILE_LAYOUT and v_cache.layout == ttnn.TILE_LAYOUT
        and not k_cache.is_sharded() and not v_cache.is_sharded()
        and list(v_cache.shape) == shape and shape[2] % 32 == 0 and shape[3] % 32 == 0
        and shape[1] <= 32 and (shape[3] // 32) % shape[1] == 0
    )
    shards_ok = all(
        tuple(k.shape) == (1, 1, shape[1], shape[3]) and k.layout == ttnn.TILE_LAYOUT and k.dtype == ttnn.bfloat16
        and k.is_sharded() and len(ttnn.corerange_to_cores(k.memory_config().shard_spec.grid)) == 1
        for k in keys
    )
    return index_ok and cache_ok and shards_ok


class PagedKVWrite:
    def __init__(self):
        self.signature = None
        self.program = None

    def _make_program(self, k_cache, v_cache, keys, values, positions, page_table, outs):
        device = k_cache.device()
        tokens = len(keys)
        heads, block_size, dim = int(k_cache.shape[1]), int(k_cache.shape[2]), int(k_cache.shape[3])
        wt = dim // 32
        copy_last = int(outs is not None)
        k_out, v_out = outs if outs is not None else (k_cache, v_cache)
        cores = [ttnn.CoreCoord(i, 0) for i in range(2 * heads)]
        core_set = ttnn.CoreRangeSet({ttnn.CoreRange(cores[0], cores[-1])})
        shard_logical = ttnn.corerange_to_cores(keys[0].memory_config().shard_spec.grid)[0]
        for t in (*keys, *values):
            if ttnn.corerange_to_cores(t.memory_config().shard_spec.grid)[0] != shard_logical:
                raise ValueError("kv_write needs all K/V shards on one core")
        shard = device.worker_core_from_logical_core(shard_logical)
        bfp8 = int(k_cache.dtype == ttnn.bfloat8_b)
        cache_tile = 1088 if bfp8 else _TILE_BYTES
        ct = [tokens, heads, wt, block_size, copy_last, bfp8]
        for t in (positions, page_table, k_cache, v_cache, k_out, v_out):
            ct += _accessor_args(t)
        page_bytes = lambda t: int(t.padded_shape[-1]) * 4  # noqa: E731  (int32 row-major rows)
        common = [int(t.buffer_address()) for t in (positions, page_table, k_cache, v_cache, k_out, v_out)]
        common += [page_bytes(positions), page_bytes(page_table), shard.x, shard.y]
        srcs = []
        for k, v in zip(keys, values):
            srcs += [int(k.buffer_address()), int(v.buffer_address())]
        runtime = [(c, common + [i // heads, i % heads] + srcs) for i, c in enumerate(cores)]
        scratch = 256 + 2 * wt * cache_tile + tokens * wt * 64 + (wt // heads) * _TILE_BYTES * copy_last
        kernel = ttnn.KernelDescriptor(
            kernel_source=_KERNEL, core_ranges=core_set, compile_time_args=ct, runtime_args=runtime,
            config=ttnn.ReaderConfigDescriptor(),
        )
        tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
        cb = ttnn.CBDescriptor(
            total_size=(scratch + _TILE_BYTES - 1) // _TILE_BYTES * _TILE_BYTES, core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, _TILE_BYTES, tile)],
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb], semaphores=[])

    def __call__(self, k_cache, v_cache, keys, values, positions, page_table):
        """Write token t's K/V shard rows (keys[t], values[t]: [1,1,Hkv,D] TILE on one core) at positions[t] through
        page_table row t (int32 row-major). With two tokens, returns DRAM copies of the last token's (K, V)."""
        tokens = len(keys)
        if not 1 <= tokens <= 2 or len(values) != tokens or not supported(k_cache, v_cache, keys, positions,
                                                                             page_table):
            raise ValueError("kv_write: unsupported operands (see supported())")
        outs = None
        if tokens == 2:
            outs = tuple(
                ttnn.empty(list(t.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=k_cache.device(),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for t in (keys[-1], values[-1])
            )
        tensors = [positions, page_table, *keys, *values, k_cache, v_cache, *(outs or ())]
        signature = tuple((int(t.buffer_address()), tuple(t.padded_shape)) for t in tensors)
        if signature != self.signature:
            self.program = self._make_program(k_cache, v_cache, keys, values, positions, page_table, outs)
            self.signature = signature
        ttnn.generic_op(tensors, self.program)
        return outs
