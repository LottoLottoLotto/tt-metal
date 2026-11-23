// SPDX-License-Identifier: Apache-2.0
// Compact row/column expansion -> FP32 subtract -> multiply -> clip -> exp -> multiply.
// Every original operation retains its FP32 CB pack/unpack boundary.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#endif

namespace decay {
constexpr uint32_t bytes = 4096;
constexpr uint32_t input_cb = 0, mask_cb = 1, masked_cb = 2, clipped_cb = 3, exp_cb = 4, output_cb = 16;
constexpr uint32_t diff_cb = 5, column_cb = 6, scratch_cb = 7;
constexpr uint32_t tiles = get_named_compile_time_arg_val("tiles");
constexpr uint32_t workers = get_named_compile_time_arg_val("workers");
#if defined(COMPILE_FOR_TRISC)
FORCE_INLINE void finish_pack(uint32_t destination) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(destination, 1);
    pack_tile(0, destination);
    tile_regs_release();
    cb_push_back(destination, 1);
}

// The mask remains at CB front until both original multiplies have used it.
FORCE_INLINE void multiply_mask(uint32_t source, uint32_t destination) {
    cb_wait_front(source, 1);
    cb_wait_front(mask_cb, 1);
    unary_op_init_common(source, destination);
    mul_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(mask_cb, source);
    copy_tile(source, 0, 0);
    copy_tile_to_dst_init_short_with_dt(source, mask_cb);
    copy_tile(mask_cb, 0, 1);
    mul_binary_tile(0, 1, 0);
    finish_pack(destination);
    cb_pop_front(source, 1);
}
#endif
}  // namespace decay

void kernel_main() {
    using namespace decay;
    const uint32_t worker = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_NCRISC)
    const auto input = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("input_dram") == 1>(), get_common_arg_val<uint32_t>(0), bytes);
    const auto mask = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("mask_dram") == 1>(), get_common_arg_val<uint32_t>(1), bytes);
    // The 96-worker stride preserves tile % 16: each worker reuses one mask.
    static_assert(workers % 16 == 0);
    cb_reserve_back(mask_cb, 1);
    noc_async_read(mask.get_noc_addr(worker % 16), get_write_ptr(mask_cb), bytes);
    noc_async_read_barrier();
    cb_push_back(mask_cb, 1);
    // Four contiguous 16-word row segments provide both 32-element vectors.
    // Scratch belongs solely to this reader; no compute-stage CB transaction.
    cb_reserve_back(scratch_cb, 1);
    const uint32_t scratch = get_write_ptr(scratch_cb);
    for (uint32_t tile = worker; tile < tiles; tile += workers) {
        const uint32_t batch = tile / 16;
        const uint32_t row_tile = (tile % 16) / 4;
        const uint32_t col_tile = tile % 4;
        const uint32_t batch_row = batch % 32;
        const uint32_t row_offset = ((batch_row / 16) * 512 + (batch_row % 16) * 16) * 4;
        const uint64_t row_addr = input.get_noc_addr((batch / 32) * 4 + row_tile) + row_offset;
        const uint64_t col_addr = input.get_noc_addr((batch / 32) * 4 + col_tile) + row_offset;
        cb_reserve_back(input_cb, 1);
        cb_reserve_back(column_cb, 1);
        noc_async_read(row_addr, scratch, 64);
        noc_async_read(row_addr + 1024, scratch + 64, 64);
        noc_async_read(col_addr, scratch + 128, 64);
        noc_async_read(col_addr + 1024, scratch + 192, 64);
        noc_async_read_barrier();
        auto* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        auto* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(input_cb));
        auto* cols = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(column_cb));
        // Cache each vector word in a register while filling its two faces.
        // Integer stores preserve FP32 payloads without rereading each element.
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t value = values[r];
            const uint32_t offset = (r / 16) * 512 + (r % 16) * 16;
            #pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; ++c) {
                rows[offset + c] = value;
                rows[offset + 256 + c] = value;
            }
        }
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t value = values[32 + c];
            const uint32_t offset = (c / 16) * 256 + c % 16;
            #pragma GCC unroll 16
            for (uint32_t r = 0; r < 16; ++r) {
                cols[offset + r * 16] = value;
                cols[offset + 512 + r * 16] = value;
            }
        }
        cb_push_back(input_cb, 1);
        cb_push_back(column_cb, 1);
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("output_dram") == 1>(), get_common_arg_val<uint32_t>(0), bytes);
    for (uint32_t tile = worker; tile < tiles; tile += workers) {
        cb_wait_front(output_cb, 1);
        noc_async_write(get_read_ptr(output_cb), output.get_noc_addr(tile), bytes);
        noc_async_write_barrier();
        cb_pop_front(output_cb, 1);
    }
#elif defined(COMPILE_FOR_TRISC)
    for (uint32_t tile = worker; tile < tiles; tile += workers) {
        cb_wait_front(input_cb, 1);
        cb_wait_front(column_cb, 1);
        unary_op_init_common(input_cb, diff_cb);
        sub_binary_tile_init();
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(column_cb, input_cb);
        copy_tile(input_cb, 0, 0);
        copy_tile_to_dst_init_short_with_dt(input_cb, column_cb);
        copy_tile(column_cb, 0, 1);
        sub_binary_tile(0, 1, 0);
        finish_pack(diff_cb);
        cb_pop_front(input_cb, 1);
        cb_pop_front(column_cb, 1);
        multiply_mask(diff_cb, masked_cb);

        cb_wait_front(masked_cb, 1);
        init_sfpu(masked_cb, clipped_cb);
        tile_regs_acquire();
        copy_tile(masked_cb, 0, 0);
        clamp_tile_init();
        clamp_tile(0, 0xc1a00000u, 0x00000000u);
        finish_pack(clipped_cb);
        cb_pop_front(masked_cb, 1);

        cb_wait_front(clipped_cb, 1);
        init_sfpu(clipped_cb, exp_cb);
        tile_regs_acquire();
        copy_tile(clipped_cb, 0, 0);
        // Exp initialization is required each tile: clamp modifies programmable
        // SFPU constants also used by the accurate FP32 exp reciprocal LLK.
        exp_tile_init<false>();
        exp_tile<false>(0);
        finish_pack(exp_cb);
        cb_pop_front(clipped_cb, 1);

        multiply_mask(exp_cb, output_cb);
    }
    cb_pop_front(mask_cb, 1);
#endif
}
