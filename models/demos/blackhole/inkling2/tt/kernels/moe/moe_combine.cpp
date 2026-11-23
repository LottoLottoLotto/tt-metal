// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling-2-Mini prefill combine: the down stage's FP32 rows (moe_grouped.cpp STAGE 1: page ((t / 32) * col_groups +
// c) * top_k + k = row-major [32 tokens, cols * 32] FP32 of pick k, column group c) -> y BF16 tiles [T, H].
// Items (token tile, column group), contiguous ranges per core: NCRISC reads the top_k pages (one contiguous read
// each); TRISC sums them as linear 1024-datum pieces in pick order (FP32 DEST, unpacked straight to DEST at FP32,
// SFPU adds), rounds once to BF16 (RNE typecast), tilizes the [32, cols * 32] block; BRISC writes the cols tiles.
// The per-element arithmetic is the stock grouped_combine's (same order, same ops).
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#endif

constexpr uint32_t cb_in = 0, cb_sum = 2, cb_out = 16;

void kernel_main() {
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t cols = get_named_compile_time_arg_val("cols");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t col_groups = hidden_tiles / cols;
    constexpr uint32_t items = tokens / 32 * col_groups;
    constexpr uint32_t page_bytes = 32 * cols * 32 * 4;
    static_assert(tokens % 32 == 0 && hidden_tiles % cols == 0);
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t workers = get_arg_val<uint32_t>(1);
    const uint32_t begin = worker * items / workers, end = (worker + 1) * items / workers;
#if defined(COMPILE_FOR_NCRISC)
    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto rows = TensorAccessor(in_args, get_common_arg_val<uint32_t>(2), page_bytes);
    for (uint32_t item = begin; item < end; ++item) {
        cb_reserve_back(cb_in, top_k * cols);
        const uint32_t d = get_write_ptr(cb_in);
        for (uint32_t k = 0; k < top_k; ++k) {
            noc_async_read(rows.get_noc_addr(item * top_k + k), d + k * page_bytes, page_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, top_k * cols);
    }
#elif defined(COMPILE_FOR_BRISC)
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(3), 2048);
    for (uint32_t item = begin; item < end; ++item) {
        const uint32_t first = (item / col_groups) * hidden_tiles + (item % col_groups) * cols;
        cb_wait_front(cb_out, cols);
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write(get_read_ptr(cb_out) + c * 2048, out.get_noc_addr(first + c), 2048);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, cols);
    }
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    unary_op_init_common(cb_in, cb_sum);
    for (uint32_t item = begin; item < end; ++item) {
        cb_wait_front(cb_in, top_k * cols);
        cb_reserve_back(cb_sum, cols);
        pack_reconfig_data_format(cb_sum);
        for (uint32_t p = 0; p < cols; ++p) {
            tile_regs_acquire();
            add_binary_tile_init();
            for (uint32_t k = 0; k < top_k; ++k) {
                copy_tile_to_dst_init_short_with_dt(cb_sum, cb_in);
                copy_tile(cb_in, k * cols + p, k == 0 ? 0 : 1);
                if (k != 0) {
                    add_binary_tile(0, 1, 0);
                }
            }
            typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),
                               static_cast<uint32_t>(DataFormat::Float16_b)>();
            typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_sum, p);
            tile_regs_release();
        }
        cb_push_back(cb_sum, cols);
        cb_pop_front(cb_in, top_k * cols);
        compute_kernel_lib::tilize<cols, cb_sum, cb_out>(1);
    }
#endif
}
