// SPDX-License-Identifier: Apache-2.0
// Verification (T=2..8): the original 56-worker RoPE compute of
// rope_to_sdpa_kernel.cpp, looped over every token row in one launch. Queries go
// to the SDPA group tensors; K/V head rows are written straight into their paged
// cache rows (32-byte face stripes, the same bytes paged_update_cache stores).
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_buffer.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#endif

namespace rope_fusion {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kTokens = ROPE_TOKENS;
constexpr uint32_t kInput = 0;       // UnpackToDestFp32: x's selected half
constexpr uint32_t kNegInput = 1;    // Default: standalone unary neg's input
constexpr uint32_t kCos = 2;         // UnpackToDestFp32
constexpr uint32_t kSin = 3;         // UnpackToDestFp32
constexpr uint32_t kRotated = 4;     // UnpackToDestFp32: BF16 pack of -x_hi, or raw x_lo
constexpr uint32_t kProductCos = 5;  // Default: BF16 rounded first multiply
constexpr uint32_t kProductSin = 6;  // Default: BF16 rounded second multiply
constexpr uint32_t kOutput = 16;
constexpr uint32_t kPass = 17;      // Three raw gathered rows: reader -> writer only
constexpr uint32_t kZero = 18;      // Writer-owned zero tile for query padding rows
constexpr uint32_t kValue = 19;     // Four raw V rows per KV worker: no TRISC consumer
constexpr uint32_t kScratch = 20;   // Writer-owned position / page-table metadata
#if defined(ROPE_CACHE_BFP8)
// BFP8 caches: every K/V row is converted by the packer (TRISC) and patched into its cache tile.
constexpr uint32_t kCache8 = 21;    // Eight packed BFP8 row tiles per KV token: K rot, 3 K pass, 4 V
constexpr uint32_t kStage8 = 22;    // Writer-owned staging copy of one BFP8 cache tile
constexpr uint32_t kTile8Bytes = 1088;  // 64 exponent bytes (face-major, one per face row) + 1024 mantissas
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
// A BF16 tile is four 16x16 faces. Each transfer covers one aligned face row.
constexpr uint32_t row_offset(uint32_t row) {
    return (row / 16) * 1024 + (row % 16) * 32;
}

void clear_tile(uint32_t address) {
    volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(address);
    for (uint32_t word = 0; word < kTileBytes / sizeof(uint32_t); ++word) {
        words[word] = 0;
    }
}
#endif

#if defined(COMPILE_FOR_NCRISC)
template <typename Accessor>
void gather_row(const Accessor& input, uint32_t tile, uint32_t row, uint32_t dst) {
    // DRAM DMA requires matched 64-byte alignment; select the row locally
    // after the caller's read barrier, never issue an odd-row 32-byte read.
    const uint64_t src = get_noc_addr(tile, input) + row_offset(row & ~uint32_t(1));
    noc_async_read(src, dst, 64);
    noc_async_read(src + 512, dst + 512, 64);
}

void finish_gather(uint32_t dst, uint32_t row) {
    for (uint32_t face = 0; face < 2; ++face) {
        volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(dst + face * 512);
        if (row % 2 != 0) {
            for (uint32_t word = 0; word < 8; ++word) {
                words[word] = words[8 + word];
            }
        }
        for (uint32_t word = 8; word < 16; ++word) {
            words[word] = 0;
        }
    }
}

template <typename Accessor>
void read_head(const Accessor& input, uint32_t token, uint32_t head, uint32_t half) {
    const uint32_t opposite_cb = half == 0 ? kNegInput : kRotated;
    cb_reserve_back(kInput, 1);
    cb_reserve_back(opposite_cb, 1);
    cb_reserve_back(kPass, 3);
    const uint32_t input_dst = get_write_ptr(kInput);
    const uint32_t opposite_dst = get_write_ptr(opposite_cb);
    gather_row(input, token * 8 + half, head, input_dst);
    gather_row(input, token * 8 + 1 - half, head, opposite_dst);
    const uint32_t pass_dst = get_write_ptr(kPass);
    for (uint32_t tile = 0; tile < 3; ++tile) {
        gather_row(input, token * 8 + 2 + half * 3 + tile, head, pass_dst + tile * kTileBytes);
    }
    noc_async_read_barrier();
    finish_gather(input_dst, head);
    finish_gather(opposite_dst, head);
    for (uint32_t tile = 0; tile < 3; ++tile) {
        finish_gather(pass_dst + tile * kTileBytes, head);
    }
    cb_push_back(kInput, 1);
    cb_push_back(opposite_cb, 1);
    cb_push_back(kPass, 3);
}
#elif defined(COMPILE_FOR_BRISC)
void scatter_row(uint32_t src, uint64_t tile_base, uint32_t row) {
    const uint64_t dst = tile_base + row_offset(row);
    noc_async_write(src, dst, 32);
    noc_async_write(src + 512, dst + 512, 32);
}
#if defined(ROPE_CACHE_BFP8)
// Copy BFP8 row `src_row` of an L1 tile into row `dst_row` of a DRAM cache tile: read the tile,
// replace that row's two face-row exponents and 2x16 mantissa bytes, write the tile back. A BFP8
// exponent is shared by one 16-element face row only, so every other row keeps its bytes.
void patch_row_bfp8(uint32_t src, uint32_t src_row, uint64_t tile_addr, uint32_t dst_row, uint32_t stage) {
    noc_async_read(tile_addr, stage, kTile8Bytes);
    noc_async_read_barrier();
    volatile uint8_t* s8 = reinterpret_cast<volatile uint8_t*>(src);
    volatile uint8_t* d8 = reinterpret_cast<volatile uint8_t*>(stage);
    for (uint32_t column = 0; column < 2; ++column) {
        const uint32_t src_face = (src_row / 16) * 2 + column;
        const uint32_t dst_face = (dst_row / 16) * 2 + column;
        d8[dst_face * 16 + dst_row % 16] = s8[src_face * 16 + src_row % 16];
        volatile uint32_t* sw = reinterpret_cast<volatile uint32_t*>(src + 64 + src_face * 256 + (src_row % 16) * 16);
        volatile uint32_t* dw = reinterpret_cast<volatile uint32_t*>(stage + 64 + dst_face * 256 + (dst_row % 16) * 16);
        for (uint32_t word = 0; word < 4; ++word) {
            dw[word] = sw[word];
        }
    }
    noc_async_write(stage, tile_addr, kTile8Bytes);
    noc_async_write_barrier();
}
#endif
#elif defined(COMPILE_FOR_TRISC)
void multiply_and_round(uint32_t lhs, uint32_t rhs, uint32_t output) {
    // Match binary_ng's SFPU multiply, with independent BF16 pack boundaries.
    // Do not substitute FPU mul_tiles or keep an unrounded product in DST.
    unary_op_init_common(lhs, output);
    mul_binary_tile_init();
    cb_wait_front(lhs, 1);
    cb_wait_front(rhs, 1);
    cb_reserve_back(output, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(rhs, lhs);
    copy_tile(lhs, 0, 0);
    copy_tile_to_dst_init_short_with_dt(lhs, rhs);
    copy_tile(rhs, 0, 1);
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, output);
    tile_regs_release();
    cb_push_back(output, 1);
    cb_pop_front(lhs, 1);
    cb_pop_front(rhs, 1);
}
#if defined(ROPE_CACHE_BFP8)
// BF16 row tiles -> BFP8 through the packer (the conversion paged_update_cache's tilize applies).
void pack_bfp8(uint32_t input, uint32_t count) {
    cb_wait_front(input, count);
    reconfig_data_format_srca(input);
    pack_reconfig_data_format(kCache8);
    copy_tile_to_dst_init_short(input);
    tile_regs_acquire();
    for (uint32_t tile = 0; tile < count; ++tile) {
        copy_tile(input, tile, tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t tile = 0; tile < count; ++tile) {
        pack_tile(tile, kCache8);
    }
    tile_regs_release();
    cb_pop_front(input, count);
}
#endif
#endif
}  // namespace rope_fusion

void kernel_main() {
    using namespace rope_fusion;
    // All three processors receive the original 56 task IDs.
    const uint32_t task = get_arg_val<uint32_t>(0);
    const uint32_t half = task % 2;
    const bool is_query = task < 48;
    const uint32_t head = (is_query ? task : task - 48) / 2;
#if defined(COMPILE_FOR_NCRISC)
    const auto query = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<ROPE_Q_DRAM>(), get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto key = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<ROPE_K_DRAM>(), get_common_arg_val<uint32_t>(1), kTileBytes);
    const auto cosine = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<ROPE_COS_DRAM>(), get_common_arg_val<uint32_t>(2), kTileBytes);
    const auto sine = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<ROPE_SIN_DRAM>(), get_common_arg_val<uint32_t>(3), kTileBytes);
    const auto packed = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4), kTileBytes);
    for (uint32_t token = 0; token < kTokens; ++token) {
        // Only row 0 of these elementwise operands is ever scattered out, so their
        // other rows need no clearing (every result element depends on its own inputs).
        cb_reserve_back(kCos, 1);
        cb_reserve_back(kSin, 1);
        gather_row(cosine, half, token, get_write_ptr(kCos));
        gather_row(sine, half, token, get_write_ptr(kSin));
        noc_async_read_barrier();
        finish_gather(get_write_ptr(kCos), token);
        finish_gather(get_write_ptr(kSin), token);
        cb_push_back(kCos, 1);
        cb_push_back(kSin, 1);
        if (is_query) {
            read_head(query, token, head, half);
        } else {
            read_head(key, token, head, half);
            cb_reserve_back(kValue, 4);
            const uint32_t value_dst = get_write_ptr(kValue);
            for (uint32_t tile = 0; tile < 4; ++tile) {
                // Packed Q/G occupy 12288 columns, K 1024, then V at 13312.
                gather_row(packed, 416 + head * 8 + half * 4 + tile, token, value_dst + tile * kTileBytes);
            }
            noc_async_read_barrier();
            cb_push_back(kValue, 4);
        }
    }
#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t kv_heads = 4;
    constexpr uint32_t width_tiles = 8;
    constexpr uint32_t page_rows = ROPE_PAGE_ROWS;
    // SDPA groups hold ROPE_GROUP consecutive rows: token t -> group t/G, row t%G.
    // Common args 0..3 are the (padded) group addresses.
#if defined(ROPE_CACHE_BFP8)
    constexpr uint32_t cache_page_bytes = kTile8Bytes;
#else
    constexpr uint32_t cache_page_bytes = kTileBytes;
#endif
    const auto cache_key = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4), cache_page_bytes);
    const auto cache_value = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(5), cache_page_bytes);
    const auto positions = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6), ROPE_POSITION_BYTES);
    const auto table = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(7), ROPE_TABLE_BYTES);
    const uint32_t scratch = get_write_ptr(kScratch);
    const uint32_t positions_ptr = scratch;
    const uint32_t table_ptr = scratch + 128;
    if (!is_query) {
        noc_async_read(get_noc_addr(0, positions), positions_ptr, ((kTokens * 4 + 31) / 32) * 32);
        noc_async_read_barrier();
        // Aligned 64-byte stripe of every token's page-table row, one round trip.
        for (uint32_t token = 0; token < kTokens; ++token) {
            const uint32_t position = reinterpret_cast<volatile uint32_t*>(positions_ptr)[token];
            noc_async_read(get_noc_addr(token, table) + ((position / page_rows) / 16) * 64, table_ptr + token * 64, 64);
        }
        noc_async_read_barrier();
    }
    if (task < 8) {
        cb_reserve_back(kZero, 1);
        clear_tile(get_write_ptr(kZero));
    }
#if defined(ROPE_Q_PACKED)
    // Packed verification Q (verify_sdpa.py): tile row (kv_head * B + token / 4), B = ceil(T / 4),
    // holds tokens 4b..4b+3 x the KV head's 6 Q heads in rows (token % 4) * 6 + head % 6; rows of
    // absent tokens (T % 4 != 0) are never read back as outputs; rows 24..31 zero.
    constexpr uint32_t q_blocks = (kTokens + 3) / 4;
    const auto packed_q = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), kTileBytes);
    if (task < 8) {
        const uint32_t zero = get_write_ptr(kZero);
        for (uint32_t tile = 0; tile < 4 * q_blocks * 8; ++tile) {
            scatter_row(zero, get_noc_addr(tile, packed_q), 24 + task);
        }
        // Rows of the last block's absent tokens are zeroed too (their outputs are discarded).
        for (uint32_t row = (kTokens % 4) * 6; kTokens % 4 != 0 && row < 24; ++row) {
            if (row % 8 != task) {
                continue;
            }
            for (uint32_t kv_head = 0; kv_head < 4; ++kv_head) {
                for (uint32_t tile = 0; tile < 8; ++tile) {
                    scatter_row(zero, get_noc_addr((kv_head * q_blocks + q_blocks - 1) * 8 + tile, packed_q), row);
                }
            }
        }
    }
#endif
    for (uint32_t token = 0; token < kTokens; ++token) {
#if defined(ROPE_Q_PACKED)
        const uint32_t q_tile_row = (head / 6) * q_blocks + token / 4;
        const uint32_t q_row = (token % 4) * 6 + head % 6;
        const auto& group = packed_q;
        const uint32_t out_token = q_tile_row;
#else
        const auto group = TensorAccessor(
            tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(token / ROPE_GROUP), kTileBytes);
        const uint32_t out_token = token % ROPE_GROUP;
        const uint32_t q_row = head;
        // Query padding rows 24..31 are zeroed by disjoint row owners.
        if (task < 8) {
            const uint32_t zero = get_write_ptr(kZero);
            for (uint32_t tile = 0; tile < 8; ++tile) {
                scatter_row(zero, get_noc_addr(out_token * 8 + tile, group), 24 + task);
            }
        }
#endif
        uint32_t cache_tile = 0;
        uint32_t cache_row = 0;
        if (!is_query) {
            const uint32_t position = reinterpret_cast<volatile uint32_t*>(positions_ptr)[token];
            const uint32_t logical_page = position / page_rows;
            const uint32_t physical_page =
                reinterpret_cast<volatile uint32_t*>(table_ptr + token * 64)[logical_page % 16];
            cache_tile = ((physical_page * kv_heads + head) * (page_rows / 32) + (position % page_rows) / 32) *
                         width_tiles;
            cache_row = position % 32;
        }
#if defined(ROPE_CACHE_BFP8)
        if (!is_query) {
            cb_wait_front(kCache8, 8);
            const uint32_t src = get_read_ptr(kCache8);
            const uint32_t stage = get_write_ptr(kStage8);
            // Tile order (see TRISC): K rotated half, K pass columns, then V (row token % 2).
            patch_row_bfp8(src, 0, get_noc_addr(cache_tile + half, cache_key), cache_row, stage);
            for (uint32_t tile = 0; tile < 3; ++tile) {
                patch_row_bfp8(src + (1 + tile) * kTile8Bytes, 0,
                               get_noc_addr(cache_tile + 2 + half * 3 + tile, cache_key), cache_row, stage);
            }
            for (uint32_t tile = 0; tile < 4; ++tile) {
                patch_row_bfp8(src + (4 + tile) * kTile8Bytes, token % 2,
                               get_noc_addr(cache_tile + half * 4 + tile, cache_value), cache_row, stage);
            }
            cb_pop_front(kCache8, 8);
            continue;
        }
#endif
        cb_wait_front(kPass, 3);
        const uint32_t pass_src = get_read_ptr(kPass);
        for (uint32_t tile = 0; tile < 3; ++tile) {
            const uint32_t column_tile = 2 + half * 3 + tile;
            if (is_query) {
                scatter_row(pass_src + tile * kTileBytes, get_noc_addr(out_token * 8 + column_tile, group), q_row);
            } else {
                scatter_row(pass_src + tile * kTileBytes, get_noc_addr(cache_tile + column_tile, cache_key), cache_row);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(kPass, 3);
        cb_wait_front(kOutput, 1);
        if (is_query) {
            scatter_row(get_read_ptr(kOutput), get_noc_addr(out_token * 8 + half, group), q_row);
        } else {
            scatter_row(get_read_ptr(kOutput), get_noc_addr(cache_tile + half, cache_key), cache_row);
        }
        noc_async_write_barrier();
        cb_pop_front(kOutput, 1);

        if (!is_query) {
            cb_wait_front(kValue, 4);
            const uint32_t value_src = get_read_ptr(kValue);
            // Token row t sits (t % 2) * 32 bytes into its aligned gathered row pair.
            // Only the first 32 bytes of each face stripe are written.
            for (uint32_t tile = 0; tile < 4; ++tile) {
                scatter_row(
                    value_src + tile * kTileBytes + (token % 2) * 32,
                    get_noc_addr(cache_tile + half * 4 + tile, cache_value),
                    cache_row);
            }
            noc_async_write_barrier();
            cb_pop_front(kValue, 4);
        }
    }
    if (task < 8) {
        noc_async_write_barrier();
        cb_push_back(kZero, 1);
        cb_wait_front(kZero, 1);
        cb_pop_front(kZero, 1);
    }
#elif defined(COMPILE_FOR_TRISC)
    for (uint32_t token = 0; token < kTokens; ++token) {
        if (half == 0) {
            // Pinned unary NEG is negative_tile with Default unpack, HiFi4,
            // APPROX=false, fp32_dest_acc_en=false. Pack BEFORE multiplying.
            init_sfpu(kNegInput, kRotated);
            cb_wait_front(kNegInput, 1);
            cb_reserve_back(kRotated, 1);
            tile_regs_acquire();
            copy_tile(kNegInput, 0, 0);
            negative_tile_init();
            negative_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, kRotated);
            tile_regs_release();
            cb_push_back(kRotated, 1);
            cb_pop_front(kNegInput, 1);
        }
        multiply_and_round(kInput, kCos, kProductCos);
        multiply_and_round(kRotated, kSin, kProductSin);

        // BF16 ADD is the FPU binary_ng path, NOT SFPU add_binary_tile.
        // Both operands have been packed separately and have Default unpack.
        binary_op_init_common(kProductCos, kProductSin, kOutput);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(kProductCos, kProductSin);
        cb_wait_front(kProductCos, 1);
        cb_wait_front(kProductSin, 1);
        cb_reserve_back(kOutput, 1);
        tile_regs_acquire();
        add_tiles(kProductCos, kProductSin, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kOutput);
        tile_regs_release();
        cb_push_back(kOutput, 1);
        cb_pop_front(kProductCos, 1);
        cb_pop_front(kProductSin, 1);
#if defined(ROPE_CACHE_BFP8)
        if (!is_query) {
            cb_reserve_back(kCache8, 8);
            pack_bfp8(kOutput, 1);
            pack_bfp8(kPass, 3);
            pack_bfp8(kValue, 4);
            cb_push_back(kCache8, 8);
        }
#endif
    }
#endif
}
