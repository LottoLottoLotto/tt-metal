// SPDX-License-Identifier: Apache-2.0
// Original 56-worker RoPE compute, with head-row gather/scatter. ROPE_TOKEN
// selects the token row (M1: 0; MTP M2 verification: 0 or 1 per launch).
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

// First V tile of the V source: the fused QKV projection (V at column 13312) or the rotated
// 4-bit-cache V rows (v_rotate.rotate_v_rows), which start at column 0.
#ifndef ROPE_V_TILE0
#define ROPE_V_TILE0 416
#endif
#ifndef ROPE_TOKEN
#define ROPE_TOKEN 0
#endif
// Output token tile row of Q (wide verification splits Q into SDPA groups).
#ifndef ROPE_OUT_TOKEN
#define ROPE_OUT_TOKEN ROPE_TOKEN
#endif

namespace rope_fusion {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kInput = 0;       // UnpackToDestFp32: x's selected half
constexpr uint32_t kNegInput = 1;    // Default: standalone unary neg's input
constexpr uint32_t kCos = 2;         // UnpackToDestFp32
constexpr uint32_t kSin = 3;         // UnpackToDestFp32
constexpr uint32_t kRotated = 4;     // UnpackToDestFp32: BF16 pack of -x_hi, or raw x_lo
constexpr uint32_t kProductCos = 5;  // Default: BF16 rounded first multiply
constexpr uint32_t kProductSin = 6;  // Default: BF16 rounded second multiply
constexpr uint32_t kOutput = 16;
constexpr uint32_t kPass = 17;      // Three raw gathered rows: reader -> writer only
constexpr uint32_t kZero = 18;      // Writer-owned zero tile for disjoint padding rows
constexpr uint32_t kValue = 19;     // Four raw V rows per KV worker: no TRISC consumer

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
void read_head(const Accessor& input, uint32_t head, uint32_t half) {
    const uint32_t opposite_cb = half == 0 ? kNegInput : kRotated;
    cb_reserve_back(kInput, 1);
    cb_reserve_back(opposite_cb, 1);
    cb_reserve_back(kPass, 3);
    const uint32_t input_dst = get_write_ptr(kInput);
    const uint32_t opposite_dst = get_write_ptr(opposite_cb);
    clear_tile(input_dst);
    clear_tile(opposite_dst);
    gather_row(input, ROPE_TOKEN * 8 + half, head, input_dst);
    gather_row(input, ROPE_TOKEN * 8 + 1 - half, head, opposite_dst);
    const uint32_t pass_dst = get_write_ptr(kPass);
    for (uint32_t tile = 0; tile < 3; ++tile) {
        gather_row(input, ROPE_TOKEN * 8 + 2 + half * 3 + tile, head, pass_dst + tile * kTileBytes);
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
#endif
}  // namespace rope_fusion

void kernel_main() {
    using namespace rope_fusion;
    // All three processors receive the original 56 task IDs on every launch.
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
    cb_reserve_back(kCos, 1);
    cb_reserve_back(kSin, 1);
    clear_tile(get_write_ptr(kCos));
    clear_tile(get_write_ptr(kSin));
    // Only token row zero is logical. Compute still consumes full BF16 tiles.
    gather_row(cosine, half, ROPE_TOKEN, get_write_ptr(kCos));
    gather_row(sine, half, ROPE_TOKEN, get_write_ptr(kSin));
    noc_async_read_barrier();
    finish_gather(get_write_ptr(kCos), ROPE_TOKEN);
    finish_gather(get_write_ptr(kSin), ROPE_TOKEN);
    cb_push_back(kCos, 1);
    cb_push_back(kSin, 1);
    if (is_query) {
        read_head(query, head, half);
    } else {
        read_head(key, head, half);
        cb_reserve_back(kValue, 4);
        const uint32_t value_dst = get_write_ptr(kValue);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            // Packed Q/G occupy 12288 columns, K 1024, then V at 13312 (tile 416); a rotated
            // V source (v_rotate.rotate_v_rows, [1,T,1024]) starts at tile 0.
            gather_row(packed, ROPE_V_TILE0 + head * 8 + half * 4 + tile, ROPE_TOKEN, value_dst + tile * kTileBytes);
        }
        noc_async_read_barrier();
        cb_push_back(kValue, 4);
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto output = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), kTileBytes);
    const uint32_t shard_x = get_common_arg_val<uint32_t>(3);
    const uint32_t shard_y = get_common_arg_val<uint32_t>(4);
    const uint64_t key_base = get_noc_addr(shard_x, shard_y, get_common_arg_val<uint32_t>(1));
    const uint64_t value_base = get_noc_addr(shard_x, shard_y, get_common_arg_val<uint32_t>(2));

    // Padding and live-row ownership are disjoint even across NOCs. No worker
    // clears a whole output tile while another worker writes a live head row.
    if (task < 28) {
        cb_reserve_back(kZero, 1);
        const uint32_t zero = get_write_ptr(kZero);
        clear_tile(zero);
        for (uint32_t tile = 0; tile < 8; ++tile) {
            if (task < 8) {
                scatter_row(zero, get_noc_addr(ROPE_OUT_TOKEN * 8 + tile, output), 24 + task);
            }
            scatter_row(zero, key_base + tile * kTileBytes, 4 + task);
            scatter_row(zero, value_base + tile * kTileBytes, 4 + task);
        }
        noc_async_write_barrier();
        cb_push_back(kZero, 1);
        cb_wait_front(kZero, 1);
        cb_pop_front(kZero, 1);
    }

    cb_wait_front(kPass, 3);
    const uint32_t pass_src = get_read_ptr(kPass);
    for (uint32_t tile = 0; tile < 3; ++tile) {
        const uint32_t column_tile = 2 + half * 3 + tile;
        const uint64_t dst =
            is_query ? get_noc_addr(ROPE_OUT_TOKEN * 8 + column_tile, output) : key_base + column_tile * kTileBytes;
        scatter_row(pass_src + tile * kTileBytes, dst, head);
    }
    noc_async_write_barrier();
    cb_pop_front(kPass, 3);
    cb_wait_front(kOutput, 1);
    const uint64_t rotary_dst = is_query ? get_noc_addr(ROPE_OUT_TOKEN * 8 + half, output) : key_base + half * kTileBytes;
    scatter_row(get_read_ptr(kOutput), rotary_dst, head);
    noc_async_write_barrier();
    cb_pop_front(kOutput, 1);

    if (!is_query) {
        cb_wait_front(kValue, 4);
        const uint32_t value_src = get_read_ptr(kValue);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            // Token row t sits (t % 2) * 32 bytes into its aligned gathered row pair.
            scatter_row(
                value_src + tile * kTileBytes + (ROPE_TOKEN % 2) * 32, value_base + (half * 4 + tile) * kTileBytes, head);
        }
        noc_async_write_barrier();
        cb_pop_front(kValue, 4);
    }
#elif defined(COMPILE_FOR_TRISC)
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
#endif
}
