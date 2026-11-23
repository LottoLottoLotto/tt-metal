// SPDX-License-Identifier: Apache-2.0
// 56 workers: one rotary tile plus three bitwise passthrough tiles per worker.
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
constexpr uint32_t kInput = 0;       // UnpackToDestFp32: x's selected half
constexpr uint32_t kNegInput = 1;    // Default: standalone unary neg's input
constexpr uint32_t kCos = 2;         // UnpackToDestFp32
constexpr uint32_t kSin = 3;         // UnpackToDestFp32
constexpr uint32_t kRotated = 4;     // UnpackToDestFp32: BF16 pack of -x_hi, or raw x_lo
constexpr uint32_t kProductCos = 5;  // Default: BF16 rounded first multiply
constexpr uint32_t kProductSin = 6;  // Default: BF16 rounded second multiply
constexpr uint32_t kOutput = 16;
constexpr uint32_t kPass = 17;       // Three raw tiles: reader -> writer only

#if defined(COMPILE_FOR_NCRISC)
template <typename Accessor>
void read_head(const Accessor& input, uint32_t head, uint32_t half) {
    const uint32_t opposite_cb = half == 0 ? kNegInput : kRotated;
    cb_reserve_back(kInput, 1);
    cb_reserve_back(opposite_cb, 1);
    cb_reserve_back(kPass, 3);
    noc_async_read(get_noc_addr(head * 8 + half, input), get_write_ptr(kInput), kTileBytes);
    noc_async_read(get_noc_addr(head * 8 + (1 - half), input), get_write_ptr(opposite_cb), kTileBytes);
    const uint32_t pass_dst = get_write_ptr(kPass);
    for (uint32_t tile = 0; tile < 3; ++tile) {
        noc_async_read(
            get_noc_addr(head * 8 + 2 + half * 3 + tile, input), pass_dst + tile * kTileBytes, kTileBytes);
    }
    noc_async_read_barrier();
    cb_push_back(kInput, 1);
    cb_push_back(opposite_cb, 1);
    cb_push_back(kPass, 3);
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
    // Pass this task to ALL three processors; never specialize compute by an
    // un-hashed host-side task/shape or let its args be stale on a cache hit.
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
    cb_reserve_back(kCos, 1);
    cb_reserve_back(kSin, 1);
    // Heights match logically for both M1 and M2: binary_ng selects NONE
    // subtile broadcast. Preserve each full cosine/sine tile's padded rows.
    noc_async_read(get_noc_addr(half, cosine), get_write_ptr(kCos), kTileBytes);
    noc_async_read(get_noc_addr(half, sine), get_write_ptr(kSin), kTileBytes);
    noc_async_read_barrier();
    cb_push_back(kCos, 1);
    cb_push_back(kSin, 1);
    if (is_query) {
        read_head(query, head, half);
    } else {
        read_head(key, head, half);
    }
#elif defined(COMPILE_FOR_BRISC)
    // Original final concat places both results in interleaved DRAM.
    const auto output = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(is_query ? 0 : 1), kTileBytes);
    cb_wait_front(kPass, 3);
    const uint32_t pass_src = get_read_ptr(kPass);
    for (uint32_t tile = 0; tile < 3; ++tile) {
        noc_async_write(
            pass_src + tile * kTileBytes, get_noc_addr(head * 8 + 2 + half * 3 + tile, output), kTileBytes);
    }
    noc_async_write_barrier();
    cb_pop_front(kPass, 3);
    cb_wait_front(kOutput, 1);
    noc_async_write(get_read_ptr(kOutput), get_noc_addr(head * 8 + half, output), kTileBytes);
    noc_async_write_barrier();
    cb_pop_front(kOutput, 1);
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
