// SPDX-License-Identifier: Apache-2.0
// C128 FP32 preparation. Every original binary op ends with an FP32 pack/reload.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_NCRISC)
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#endif

namespace preparation {
constexpr uint32_t bytes = 4096;
constexpr uint32_t tiles = get_named_compile_time_arg_val("tiles");
constexpr uint32_t workers = get_named_compile_time_arg_val("workers");
constexpr uint32_t stage = get_named_compile_time_arg_val("stage");
constexpr uint32_t alpha_mode = get_named_compile_time_arg_val("alpha_mode");

#if defined(COMPILE_FOR_TRISC)
enum class Op { Mul, Sub, Add };

template <Op op, bool duplicate_l = false>
FORCE_INLINE void binary(uint32_t lhs, uint32_t rhs, uint32_t output) {
    cb_wait_front(lhs, 1);
    cb_wait_front(rhs, 1);
    unary_op_init_common(lhs, output);
    if constexpr (op == Op::Mul) mul_binary_tile_init();
    else if constexpr (op == Op::Sub) sub_binary_tile_init();
    else add_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(rhs, lhs);
    copy_tile(lhs, 0, 0);
    copy_tile_to_dst_init_short_with_dt(lhs, rhs);
    copy_tile(rhs, 0, 1);
    if constexpr (op == Op::Mul) mul_binary_tile(0, 1, 0);
    else if constexpr (op == Op::Sub) sub_binary_tile(0, 1, 0);
    else add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(output, 1);
    pack_reconfig_data_format<true>(output);
    pack_tile(0, output);
    if constexpr (duplicate_l) {
        // Distinct local copy permits D_mat's reload while the writer drains L_mat.
        cb_reserve_back(16, 1);
        pack_reconfig_data_format<true>(16);
        pack_tile(0, 16);
        cb_push_back(16, 1);
    }
    tile_regs_release();
    cb_push_back(output, 1);
}
#endif
}  // namespace preparation

void kernel_main() {
    using namespace preparation;
    const uint32_t worker = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_NCRISC)
    const auto input0 = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("input0_dram") == 1>(), get_common_arg_val<uint32_t>(0), bytes);
    const auto input1 = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("input1_dram") == 1>(), get_common_arg_val<uint32_t>(1), bytes);
    const auto input2 = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("input2_dram") == 1>(), get_common_arg_val<uint32_t>(2), bytes);
    if constexpr (stage == 0) {
        if constexpr (alpha_mode != 0) {
            cb_reserve_back(3, 1);
            // TTNN tensor-scalar multiply fills an FP32 scalar tile, then uses binary SFPU.
            fill_with_val<1024, uint32_t>(get_write_ptr(3), alpha_mode == 1 ? 0x3f400000u : 0u);
            cb_push_back(3, 1);
        }
        for (uint32_t tile = worker; tile < tiles; tile += workers) {
            cb_reserve_back(0, 1);
            cb_reserve_back(1, 1);
            cb_reserve_back(2, 1);
            noc_async_read(input0.get_noc_addr(tile), get_write_ptr(0), bytes);
            noc_async_read(input1.get_noc_addr(tile), get_write_ptr(1), bytes);
            noc_async_read(input2.get_noc_addr(tile % 16), get_write_ptr(2), bytes);
            noc_async_read_barrier();
            cb_push_back(0, 1);
            cb_push_back(1, 1);
            cb_push_back(2, 1);
        }
    } else {
        const auto input3 = TensorAccessor(tensor_accessor::make_interleaved_dspec<
            get_named_compile_time_arg_val("input3_dram") == 1>(), get_common_arg_val<uint32_t>(3), bytes);
        for (uint32_t tile = worker; tile < tiles; tile += workers) {
            cb_reserve_back(0, 1);
            cb_reserve_back(1, 1);
            cb_reserve_back(2, 1);
            cb_reserve_back(3, 1);
            noc_async_read(input0.get_noc_addr(tile), get_write_ptr(0), bytes);
            noc_async_read(input1.get_noc_addr(tile), get_write_ptr(1), bytes);
            noc_async_read(input2.get_noc_addr(tile / 4), get_write_ptr(2), bytes);
            noc_async_read(input3.get_noc_addr(tile % 16), get_write_ptr(3), bytes);
            noc_async_read_barrier();
            // Use the exact TTNN Blackhole FP32 software column-broadcast helper.
            fill_tile_with_first_column(get_write_ptr(2));
            cb_push_back(0, 1);
            cb_push_back(1, 1);
            cb_push_back(2, 1);
            cb_push_back(3, 1);
        }
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto output0 = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("output0_dram") == 1>(), get_common_arg_val<uint32_t>(0), bytes);
    if constexpr (stage == 0) {
        const auto output1 = TensorAccessor(tensor_accessor::make_interleaved_dspec<
            get_named_compile_time_arg_val("output1_dram") == 1>(), get_common_arg_val<uint32_t>(1), bytes);
        for (uint32_t tile = worker; tile < tiles; tile += workers) {
            cb_wait_front(16, 1);
            cb_wait_front(17, 1);
            noc_async_write(get_read_ptr(16), output0.get_noc_addr(tile), bytes);
            noc_async_write(get_read_ptr(17), output1.get_noc_addr(tile), bytes);
            noc_async_write_barrier();
            cb_pop_front(16, 1);
            cb_pop_front(17, 1);
        }
    } else {
        for (uint32_t tile = worker; tile < tiles; tile += workers) {
            cb_wait_front(16, 1);
            noc_async_write(get_read_ptr(16), output0.get_noc_addr(tile), bytes);
            noc_async_write_barrier();
            cb_pop_front(16, 1);
        }
    }
#elif defined(COMPILE_FOR_TRISC)
    for (uint32_t tile = worker; tile < tiles; tile += workers) {
        if constexpr (stage == 0) {
            binary<Op::Mul>(0, 1, 4);  // kk_lmask = kk * L_mask
            cb_pop_front(0, 1);
            cb_pop_front(1, 1);
            binary<Op::Mul>(4, 2, 5);  // kk_diag = kk_lmask * eye
            if constexpr (alpha_mode == 0) {
                binary<Op::Sub>(4, 5, 7);
            } else {
                binary<Op::Mul>(5, 3, 6);  // kk_drop, including alpha=1 multiply by zero
                binary<Op::Sub>(4, 6, 7);
                cb_pop_front(6, 1);
            }
            cb_pop_front(4, 1);
            cb_pop_front(5, 1);
            binary<Op::Add, true>(2, 7, 8);  // L_mat = eye + kk_reg
            cb_pop_front(7, 1);
            binary<Op::Mul>(8, 2, 17);  // D_mat = L_mat * eye; not a diagonal shortcut
            cb_pop_front(8, 1);
            cb_pop_front(2, 1);
        } else {
            binary<Op::Sub>(0, 1, 4);  // L_strict = L_mat - D_mat
            cb_pop_front(0, 1);
            cb_pop_front(1, 1);
            binary<Op::Mul>(2, 4, 5);  // N = D_inv_row * L_strict, preserve operand order
            cb_pop_front(2, 1);
            cb_pop_front(4, 1);
            binary<Op::Add>(3, 5, 16);  // L_unit = eye + N
            cb_pop_front(3, 1);
            cb_pop_front(5, 1);
        }
    }
    if constexpr (stage == 0 && alpha_mode != 0) cb_pop_front(3, 1);
#endif
}
