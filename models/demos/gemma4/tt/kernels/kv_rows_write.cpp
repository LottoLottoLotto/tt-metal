// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Speculative-verify KV write: copies the P BF16 rows of K and V (tiled DRAM tensors, per source
// either head-major [1, Hk, P, D]: row b of head h in tile h * Dt + c, tile row b; or row-major
// [1, P, Hk, D]: tile b * Dt + c, tile row h) into their rows of a paged BF16 cache
// [blocks, Hk, block_size, D]. DMA only (the cache receives the same bits paged_update_cache
// writes). One worker per (head, column tile) owns that tile column for every row, so rows that
// share a cache tile never race. A tile row is two 32-byte face stripes (faces 0/1 or 2/3, 512 B
// apart); DRAM reads are issued as aligned 64-byte pairs, writes are the exact 32-byte stripes.
// With a nonzero ring size positions wrap modulo it before the page-table lookup (ring-buffered
// sliding caches, like paged_update_cache's cache_position_modulo).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t P = get_compile_time_arg_val(0);
    constexpr uint32_t Dt = get_compile_time_arg_val(1);
    constexpr uint32_t Hk = get_compile_time_arg_val(2);
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t pos_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t table_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t row_major_src[2] = {get_compile_time_arg_val(6), get_compile_time_arg_val(7)};
    constexpr uint32_t ring_tokens = get_compile_time_arg_val(8);  // 0: no wrap
    constexpr uint32_t tile_bytes = 2048;

    const uint32_t cache_addr[2] = {get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1)};
    const uint32_t src_addr[2] = {get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3)};
    const uint32_t pos_addr = get_arg_val<uint32_t>(4);
    const uint32_t table_addr = get_arg_val<uint32_t>(5);
    const uint32_t task = get_arg_val<uint32_t>(6);  // head * Dt + column tile
    const uint32_t head = task / Dt;
    const uint32_t column = task % Dt;

    const uint32_t scratch = get_write_ptr(0);  // 2 x 64 B row-pair staging
    const uint32_t positions = scratch + 256;   // P uint32 positions
    const uint32_t stripe = scratch + 512;      // 64 B page-table stripe

    const auto pos = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), pos_addr, pos_bytes);
    const auto table = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), table_addr, table_bytes);
    noc_async_read(get_noc_addr(0, pos), positions, ((P * 4 + 63) / 64) * 64);
    noc_async_read_barrier();

    for (uint32_t row = 0; row < P; ++row) {
        const uint32_t absolute = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(positions)[row];
        uint32_t position = absolute;
        if constexpr (ring_tokens != 0) {
            position = absolute % ring_tokens;
        }
        const uint32_t logical_page = position / block_size;
        noc_async_read(get_noc_addr(0, table) + (logical_page / 16) * 64, stripe, 64);
        noc_async_read_barrier();
        const uint32_t physical_page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stripe)[logical_page % 16];
        const uint32_t cache_tile =
            ((physical_page * Hk + head) * (block_size / 32) + (position % block_size) / 32) * Dt + column;
        const uint32_t cache_row = position % 32;
        const uint32_t cache_off = (cache_row / 16) * 1024 + (cache_row % 16) * 32;
        for (uint32_t kind = 0; kind < 2; ++kind) {
            const uint32_t src_tile = row_major_src[kind] ? row * Dt + column : task;
            const uint32_t src_row = row_major_src[kind] ? head : row;
            const uint32_t src_off = (src_row / 16) * 1024 + (src_row % 16) * 32;
            const auto src = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), src_addr[kind], tile_bytes);
            const auto cache = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), cache_addr[kind], tile_bytes);
            const uint64_t s = get_noc_addr(src_tile, src) + src_off;
            const uint64_t d = get_noc_addr(cache_tile, cache) + cache_off;
            const uint64_t s_aligned = s & ~uint64_t(63);
            const uint32_t pick = uint32_t(s) & 63;
            noc_async_read(s_aligned, scratch, 64);
            noc_async_read(s_aligned + 512, scratch + 64, 64);
            noc_async_read_barrier();
            noc_async_write(scratch + pick, d, 32);
            noc_async_write(scratch + 64 + pick, d + 512, 32);
            noc_async_write_barrier();
        }
    }
}
