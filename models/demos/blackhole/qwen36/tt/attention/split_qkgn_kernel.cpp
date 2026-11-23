// SPDX-License-Identifier: Apache-2.0
// Wide-verification BF16 bitcopy for T = SPLIT_TOKENS (3..8) token rows:
// Q [1,T,24,256] and K [1,T,4,256] (token t owns head-row tiles [8t, 8t+8))
// plus gate [1,T,6144] (192 tiles carrying all token rows). One DMA pass per
// source tile serves every token row.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

namespace split_qkgn_fusion {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kOutput = 0;
constexpr uint32_t kScratch = 1;
constexpr uint32_t kHalfRowBytes = 32;
constexpr uint32_t kTokens = SPLIT_TOKENS;
// Four (T<=4) or eight face rows (128/256 B, 64-byte aligned) cover every token row per face.
constexpr uint32_t kFaceReadBytes = kTokens <= 4 ? 128 : 256;
constexpr uint32_t kRowStride = 2 * kFaceReadBytes;

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
}  // namespace split_qkgn_fusion
#endif

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    using namespace split_qkgn_fusion;
    static_assert(kTokens >= 3 && kTokens <= 8, "split_qkgn supports three to eight token rows");
    const uint32_t kind = get_arg_val<uint32_t>(0);  // 0 Q, 1 K, 2 gate
    const uint32_t first_tile = get_arg_val<uint32_t>(1);
    const uint32_t tile_count = get_arg_val<uint32_t>(2);
    // Q/K workers own the token rows [token_begin, token_end) of their column tile.
    const uint32_t token_begin = get_arg_val<uint32_t>(3);
    const uint32_t token_end = get_arg_val<uint32_t>(4);
    const uint32_t rows = kind == 0 ? 24 : (kind == 1 ? 4 : 1);
    const auto input = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto output = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(kind + 1), kTileBytes);
    const uint32_t assembled = get_write_ptr(kOutput);
    const uint32_t scratch = get_write_ptr(kScratch);

    // Only logical rows are overwritten below; padding stays zero across reuse.
    Noc noc;
    noc.async_write_zeros(CircularBuffer(kOutput), kTokens * kTileBytes);
    noc.write_zeros_l1_barrier();

    for (uint32_t tile = first_tile; tile < first_tile + tile_count; ++tile) {
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t source_tile = kind == 0 ? row * 16 + tile
                : (kind == 1 ? 384 + row * 8 + tile : (tile / 8) * 16 + 8 + tile % 8);
            const uint64_t source = get_noc_addr(source_tile, input);
            noc_async_read(source, scratch + row * kRowStride, kFaceReadBytes);
            noc_async_read(source + 512, scratch + row * kRowStride + kFaceReadBytes, kFaceReadBytes);
        }
        noc_async_read_barrier();
        if (kind < 2) {
            for (uint32_t token = token_begin; token < token_end; ++token) {
                const uint32_t tile_base = assembled + token * kTileBytes;
                for (uint32_t row = 0; row < rows; ++row) {
                    const uint32_t destination = tile_base + row_offset(row);
                    const uint32_t source = scratch + row * kRowStride + token * kHalfRowBytes;
                    copy_half_row(source, destination);
                    copy_half_row(source + kFaceReadBytes, destination + 512);
                }
                noc_async_write(tile_base, get_noc_addr(token * 8 + tile, output), kTileBytes);
            }
        } else {
            for (uint32_t token = 0; token < kTokens; ++token) {
                const uint32_t destination = assembled + row_offset(token);
                copy_half_row(scratch + token * kHalfRowBytes, destination);
                copy_half_row(scratch + kFaceReadBytes + token * kHalfRowBytes, destination + 512);
            }
            noc_async_write(assembled, get_noc_addr(tile, output), kTileBytes);
        }
        noc_async_write_barrier();
    }
#endif
}
