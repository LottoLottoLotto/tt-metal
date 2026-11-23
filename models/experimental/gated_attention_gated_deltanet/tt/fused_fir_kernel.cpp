// SPDX-License-Identifier: Apache-2.0
// BF16 K=4 FIR, B=1,D=8192. Global shifted tensors are never materialized.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#endif

namespace fir {
constexpr uint32_t bytes = 2048;
constexpr uint32_t cols = 256;
constexpr uint32_t in0 = 0, wt0 = 1, inputs = 2, weights = 3, accum = 4, scratch = 6, out = 16;
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
constexpr uint32_t workers = get_named_compile_time_arg_val("workers");
constexpr uint32_t worker = get_named_compile_time_arg_val("worker");
#if defined(COMPILE_FOR_NCRISC)
FORCE_INLINE uint32_t offset(uint32_t row, uint32_t half) {
    return (row / 16) * 1024 + half * 512 + (row % 16) * 32;
}
// Replicate the weight's logical first row by integer copies. No FP conversions.
FORCE_INLINE void broadcast_weight(uint32_t address) {
    auto p = reinterpret_cast<volatile uint32_t*>(address);
    for (uint32_t half = 0; half < 2; ++half) {
        for (uint32_t row = 1; row < 32; ++row) {
            const uint32_t dst = offset(row, half) / 4;
            const uint32_t src = offset(0, half) / 4;
            for (uint32_t word = 0; word < 8; ++word) p[dst + word] = p[src + word];
        }
    }
}
// A face is contiguous only until either the source or destination crosses
// row16. Local NOC copies preserve every BF16 bit, including signed zero.
FORCE_INLINE void shift_tile(uint32_t first, uint32_t next, uint32_t dst, uint32_t shift) {
    for (uint32_t half = 0; half < 2; ++half) {
        uint32_t row = 0;
        while (row < 32) {
            const uint32_t source_row = row + shift;
            const uint32_t source_base = source_row < 32 ? first : next;
            const uint32_t source_local_row = source_row % 32;
            const uint32_t a = 16 - row % 16;
            const uint32_t b = 16 - source_local_row % 16;
            const uint32_t n = a < b ? a : b;
            noc_async_read(get_noc_addr(source_base + offset(source_local_row, half)),
                           dst + offset(row, half), n * 32);
            row += n;
        }
    }
}
#endif
#if defined(COMPILE_FOR_TRISC)
FORCE_INLINE void finish_pack(uint32_t destination) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(destination, 1);
    pack_reconfig_data_format<true>(destination);
    pack_tile(0, destination);
    tile_regs_release();
    cb_push_back(destination, 1);
}
#endif
}  // namespace fir

void kernel_main() {
    using namespace fir;
#if defined(COMPILE_FOR_NCRISC)
    const auto input = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("input_dram") == 1>(), get_common_arg_val<uint32_t>(0), bytes);
    const auto w0 = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("weight0_dram") == 1>(), get_common_arg_val<uint32_t>(1), bytes);
    const auto w1 = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("weight1_dram") == 1>(), get_common_arg_val<uint32_t>(2), bytes);
    const auto w2 = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("weight2_dram") == 1>(), get_common_arg_val<uint32_t>(3), bytes);
    const auto w3 = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("weight3_dram") == 1>(), get_common_arg_val<uint32_t>(4), bytes);
    cb_reserve_back(scratch, 1);
    const uint32_t second = get_write_ptr(scratch);
    for (uint32_t col = worker; col < cols; col += workers) {
        cb_reserve_back(wt0, 1);
        cb_reserve_back(weights, 3);
        const uint32_t first_weight = get_write_ptr(wt0);
        const uint32_t other_weights = get_write_ptr(weights);
        noc_async_read(w0.get_noc_addr(col), first_weight, bytes);
        noc_async_read(w1.get_noc_addr(col), other_weights, bytes);
        noc_async_read(w2.get_noc_addr(col), other_weights + bytes, bytes);
        noc_async_read(w3.get_noc_addr(col), other_weights + 2 * bytes, bytes);
        noc_async_read_barrier();
        broadcast_weight(first_weight);
        for (uint32_t tap = 0; tap < 3; ++tap) broadcast_weight(other_weights + tap * bytes);
        cb_push_back(wt0, 1);
        cb_push_back(weights, 3);
        for (uint32_t row = 0; row < rows; ++row) {
            cb_reserve_back(in0, 1);
            cb_reserve_back(inputs, 3);
            const uint32_t first = get_write_ptr(in0);
            const uint32_t shifted = get_write_ptr(inputs);
            noc_async_read(input.get_noc_addr(row * cols + col), first, bytes);
            noc_async_read(input.get_noc_addr((row + 1) * cols + col), second, bytes);
            noc_async_read_barrier();
            for (uint32_t tap = 1; tap < 4; ++tap)
                shift_tile(first, second, shifted + (tap - 1) * bytes, tap);
            noc_async_read_barrier();
            cb_push_back(in0, 1);
            cb_push_back(inputs, 3);
        }
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("output_dram") == 1>(), get_common_arg_val<uint32_t>(0), bytes);
    for (uint32_t col = worker; col < cols; col += workers) {
        for (uint32_t row = 0; row < rows; ++row) {
            cb_wait_front(out, 1);
            noc_async_write(get_read_ptr(out), output.get_noc_addr(row * cols + col), bytes);
            noc_async_write_barrier();
            cb_pop_front(out, 1);
        }
    }
#elif defined(COMPILE_FOR_TRISC)
    for (uint32_t col = worker; col < cols; col += workers) {
        cb_wait_front(wt0, 1);
        cb_wait_front(weights, 3);
        for (uint32_t row = 0; row < rows; ++row) {
            cb_wait_front(in0, 1);
            cb_wait_front(inputs, 3);
            unary_op_init_common(in0, accum);
            mul_binary_tile_init();
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(wt0, in0);
            copy_tile(in0, 0, 0);
            copy_tile_to_dst_init_short_with_dt(in0, wt0);
            copy_tile(wt0, 0, 1);
            mul_binary_tile(0, 1, 0);
            finish_pack(accum);
            cb_pop_front(in0, 1);
            for (uint32_t tap = 0; tap < 3; ++tap) {
                cb_wait_front(accum, 1);
                tile_regs_acquire();
                copy_tile_to_dst_init_short_with_dt(tap == 0 ? wt0 : weights, accum);
                copy_tile(accum, 0, 0);
                copy_tile_init(inputs);
                copy_tile(inputs, tap, 1);
                copy_tile_init(weights);
                copy_tile(weights, tap, 2);
                addcmul_tile_init();
                addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, 0x3f800000u);
                // Consume the old BF16 accumulator before reserving its slot.
                tile_regs_commit();
                tile_regs_wait();
                cb_pop_front(accum, 1);
                const uint32_t destination = tap == 2 ? out : accum;
                cb_reserve_back(destination, 1);
                pack_reconfig_data_format<true>(destination);
                pack_tile(0, destination);
                tile_regs_release();
                cb_push_back(destination, 1);
            }
            cb_pop_front(inputs, 3);
        }
        cb_pop_front(wt0, 1);
        cb_pop_front(weights, 3);
    }
#endif
}
