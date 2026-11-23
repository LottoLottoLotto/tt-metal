// SPDX-License-Identifier: Apache-2.0
// M1 BF16 bitcopy: eight Q tiles, eight K tiles, 192 flattened gate tiles.
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
    const uint32_t rows = kind == 0 ? 24 : (kind == 1 ? 4 : 1);
    const auto input = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto output = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(kind + 1), kTileBytes);
    const uint32_t assembled = get_write_ptr(kOutput);
    const uint32_t scratch = get_write_ptr(kScratch);

    // One producer/writer owns this buffer and every destination tile it emits.
    // Only logical rows are overwritten below, so all padding stays zero even
    // when the same local tile is reused for the worker's next gate tile.
    Noc noc;
    noc.async_write_zeros(CircularBuffer(kOutput), kTileBytes);
    noc.write_zeros_l1_barrier();

    for (uint32_t tile = first_tile; tile < first_tile + tile_count; ++tile) {
        for (uint32_t row = 0; row < rows; ++row) {
            // The source is one physical tile row with 448 column tiles.
            // Q: head*512 + dim; K: 12288 + head*256 + dim.
            // Gate: head*512 + 256 + dim, head=tile/8, dim=(tile%8)*32.
            const uint32_t source_tile = kind == 0 ? row * 16 + tile
                : (kind == 1 ? 384 + row * 8 + tile : (tile / 8) * 16 + 8 + tile % 8);
            const uint64_t source = get_noc_addr(source_tile, input);
            // DRAM row-pair reads must be 64-byte aligned on both ends. Gather
            // only row zero's 32-byte half-face payload after DMA: directly
            // reading into odd destination rows can select the adjacent row.
            // The extra source padding row is discarded, never propagated.
            noc_async_read(source, scratch + row * 128, 64);
            noc_async_read(source + 512, scratch + row * 128 + 64, 64);
        }
        noc_async_read_barrier();
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t destination = assembled + row_offset(row);
            copy_half_row(scratch + row * 128, destination);
            copy_half_row(scratch + row * 128 + 64, destination + 512);
        }
        // The unique owner writes all 2048 bytes exactly once. No other core
        // zeros or updates any sub-row of this tile, so no NoC write races.
        noc_async_write(assembled, get_noc_addr(tile, output), kTileBytes);
        noc_async_write_barrier();
    }
#endif
}
