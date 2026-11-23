// SPDX-License-Identifier: Apache-2.0
// Wide-verification SDPA row merge (BF16 bitcopy). Token t of MERGE_TOKENS rows
// lives in SDPA group t / 2 at batch t % 2 as a [24 heads, 256] tile row; output
// [1,T,6144] row t, column tile c holds head c / 8, head-column tile c % 8 --
// exactly reshape([1,B,24,256] -> [1,B,6144]) followed by a row concat.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

namespace merge_rows_fusion {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kOutput = 0;
constexpr uint32_t kScratch = 1;
constexpr uint32_t kHalfRowBytes = 32;
constexpr uint32_t kPairBytes = 64;
constexpr uint32_t kTokens = MERGE_TOKENS;

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
}  // namespace merge_rows_fusion
#endif

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    using namespace merge_rows_fusion;
    static_assert(kTokens >= 3 && kTokens <= 8, "merge_rows supports three to eight token rows");
    const uint32_t first_tile = get_arg_val<uint32_t>(0);
    const uint32_t tile_count = get_arg_val<uint32_t>(1);
    const auto output = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), kTileBytes);
    // Common args 1..4: the (padded) SDPA group addresses; token t reads group t / G.
    const uint32_t assembled = get_write_ptr(kOutput);
    const uint32_t scratch = get_write_ptr(kScratch);

    // Rows >= T stay zero; rows < T are fully overwritten for every tile.
    Noc noc;
    noc.async_write_zeros(CircularBuffer(kOutput), kTileBytes);
    noc.write_zeros_l1_barrier();

    for (uint32_t tile = first_tile; tile < first_tile + tile_count; ++tile) {
        const uint32_t head = tile / 8;
        const uint32_t column_tile = tile % 8;
        // DRAM reads need 64-byte alignment: read the aligned face-row pair.
        const uint32_t pair_offset = row_offset(head & ~uint32_t(1));
        for (uint32_t token = 0; token < kTokens; ++token) {
            const uint32_t source_tile = (token % MERGE_GROUP) * 8 + column_tile;
            const auto group = TensorAccessor(
                tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1 + token / MERGE_GROUP), kTileBytes);
            const uint64_t source = get_noc_addr(source_tile, group);
            noc_async_read(source + pair_offset, scratch + token * 2 * kPairBytes, kPairBytes);
            noc_async_read(source + pair_offset + 512, scratch + token * 2 * kPairBytes + kPairBytes, kPairBytes);
        }
        noc_async_read_barrier();
        const uint32_t pick = (head & 1) * kHalfRowBytes;
        for (uint32_t token = 0; token < kTokens; ++token) {
            const uint32_t destination = assembled + row_offset(token);
            copy_half_row(scratch + token * 2 * kPairBytes + pick, destination);
            copy_half_row(scratch + token * 2 * kPairBytes + kPairBytes + pick, destination + 512);
        }
        noc_async_write(assembled, get_noc_addr(tile, output), kTileBytes);
        noc_async_write_barrier();
    }
#endif
}
