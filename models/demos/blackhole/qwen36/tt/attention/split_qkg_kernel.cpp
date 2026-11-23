// SPDX-License-Identifier: Apache-2.0
// BF16 bitcopy of the packed [Q+gate per head | K | V] projection rows (T = 1 or 2 tokens,
// logical rows 0..T-1 of one tile row) into head-row Q [T,1,H,256], K [T,1,Hkv,256] and
// flat gate [1,T,H*256]. Geometry is runtime: common args 4..7 are H, Hkv, the first K
// tile (Q+gate width / 32) and T.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

namespace split_qkg_fusion {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kOutput = 0;
constexpr uint32_t kScratch = 1;
constexpr uint32_t kHalfRowBytes = 32;
constexpr uint32_t kHeadTiles = 8;  // 256 / 32 column tiles per head row

FORCE_INLINE uint32_t row_offset(uint32_t row) {
    return (row / 16) * 1024 + (row % 16) * kHalfRowBytes;
}

FORCE_INLINE void copy_half_row(uint32_t source, uint32_t destination) {
    const volatile uint32_t* src = reinterpret_cast<const volatile uint32_t*>(source);
    volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(destination);
    for (uint32_t word = 0; word < kHalfRowBytes / sizeof(uint32_t); ++word) {
        dst[word] = src[word];
    }
}
}  // namespace split_qkg_fusion
#endif

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    using namespace split_qkg_fusion;
    const uint32_t kind = get_arg_val<uint32_t>(0);  // 0 Q, 1 K, 2 gate
    const uint32_t first_tile = get_arg_val<uint32_t>(1);
    const uint32_t tile_count = get_arg_val<uint32_t>(2);
    const uint32_t q_heads = get_common_arg_val<uint32_t>(4);
    const uint32_t kv_heads = get_common_arg_val<uint32_t>(5);
    const uint32_t k_base_tile = get_common_arg_val<uint32_t>(6);
    const uint32_t tokens = get_common_arg_val<uint32_t>(7);
    const uint32_t rows = kind == 0 ? q_heads : (kind == 1 ? kv_heads : 1);
    const auto input = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto output = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(kind + 1), kTileBytes);
    const uint32_t assembled = get_write_ptr(kOutput);
    const uint32_t scratch = get_write_ptr(kScratch);

    // One producer/writer owns this buffer and every destination tile it emits.
    // Only logical rows are overwritten below, so all padding stays zero even
    // when the same local tile is reused for the worker's next tile or token.
    Noc noc;
    noc.async_write_zeros(CircularBuffer(kOutput), kTileBytes);
    noc.write_zeros_l1_barrier();

    for (uint32_t tile = first_tile; tile < first_tile + tile_count; ++tile) {
        for (uint32_t row = 0; row < rows; ++row) {
            // The source is one physical tile row of column tiles.
            // Q: head*512 + dim; K: k_base + head*256 + dim.
            // Gate: head*512 + 256 + dim, head=tile/8, dim=(tile%8)*32.
            const uint32_t source_tile = kind == 0 ? row * 16 + tile
                : (kind == 1 ? k_base_tile + row * 8 + tile : (tile / 8) * 16 + 8 + tile % 8);
            const uint64_t source = get_noc_addr(source_tile, input);
            // DRAM row-pair reads must be 64-byte aligned on both ends: one read holds
            // token rows 0 and 1 (32 bytes each) of face 0, another of face 1.
            noc_async_read(source, scratch + row * 128, 64);
            noc_async_read(source + 512, scratch + row * 128 + 64, 64);
        }
        noc_async_read_barrier();
        if (kind == 2) {
            // Gate: every token is one row of the same flat [1,T,H*256] tile.
            for (uint32_t token = 0; token < tokens; ++token) {
                const uint32_t destination = assembled + row_offset(token);
                copy_half_row(scratch + token * kHalfRowBytes, destination);
                copy_half_row(scratch + 64 + token * kHalfRowBytes, destination + 512);
            }
            noc_async_write(assembled, get_noc_addr(tile, output), kTileBytes);
            noc_async_write_barrier();
            continue;
        }
        // Q/K: token t's head rows form output tile t * 8 + tile.
        for (uint32_t token = 0; token < tokens; ++token) {
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t destination = assembled + row_offset(row);
                copy_half_row(scratch + row * 128 + token * kHalfRowBytes, destination);
                copy_half_row(scratch + row * 128 + 64 + token * kHalfRowBytes, destination + 512);
            }
            // The unique owner writes all 2048 bytes exactly once. No other core
            // zeros or updates any sub-row of this tile, so no NoC write races.
            noc_async_write(assembled, get_noc_addr(token * kHeadTiles + tile, output), kTileBytes);
            noc_async_write_barrier();
        }
    }
#endif
}
