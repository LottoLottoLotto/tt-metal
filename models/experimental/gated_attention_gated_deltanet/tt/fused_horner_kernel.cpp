// SPDX-License-Identifier: Apache-2.0
// R = add(eye, neg_n); repeat 30 times: R = add(eye, matmul(neg_n, R)).
// The two operations always meet at a packed FP32 CB, never at live DEST.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#endif

namespace horner {
constexpr uint32_t bytes = 4096;
constexpr uint32_t neg_cb = 0, eye_cb = 1, r_cb = 2, neg_exact_cb = 3, nr_cb = 4, output_cb = 16;
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

// Both source views use UnpackToDestFp32, exactly as binary_ng's FP32 add.
// Retain eye and source at their CB fronts; the owner decides when to pop.
FORCE_INLINE void add_eye(uint32_t source, uint32_t destination) {
    cb_wait_front(eye_cb, 1);
    cb_wait_front(source, 1);
    reconfig_data_format_srca(eye_cb);
    // Match unary_op_init_common's signed-zero state without its full HW reset.
    MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
    pack_reconfig_data_format(destination);
    add_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short(eye_cb);
    copy_tile(eye_cb, 0, 0);
    copy_tile_to_dst_init_short_with_dt(eye_cb, source);
    copy_tile(source, 0, 1);
    add_binary_tile(0, 1, 0);
    finish_pack(destination);
}

FORCE_INLINE void multiply() {
    cb_wait_front(neg_cb, 1);
    cb_wait_front(r_cb, 1);
    // Restore SrcA/SrcB formats without resetting live pipeline semaphores.
    // Neither input is UnpackToDestFp32: the standard matmul uses SrcA/SrcB.
    reconfig_data_format<SrcOrder::Reverse>(neg_cb, r_cb);
    pack_reconfig_data_format(nr_cb);
    matmul_block_init(neg_cb, r_cb, false, 1, 1, 1);
    tile_regs_acquire();
    matmul_block(neg_cb, r_cb, 0, 0, 0, false, 1, 1, 1);
    finish_pack(nr_cb);
    cb_pop_front(r_cb, 1);
}
#endif
}  // namespace horner

void kernel_main() {
    using namespace horner;
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t batch = get_arg_val<uint32_t>(1);
    if (worker >= batch) {
        return;
    }
#if defined(COMPILE_FOR_NCRISC)
    const auto input = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("input_dram") == 1>(), get_common_arg_val<uint32_t>(0), bytes);
    const auto eye = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("eye_dram") == 1>(), get_common_arg_val<uint32_t>(1), bytes);
    // Read the actual supplied eye once per active core, not a synthetic identity.
    cb_reserve_back(eye_cb, 1);
    noc_async_read(eye.get_noc_addr(0), get_write_ptr(eye_cb), bytes);
    noc_async_read_barrier();
    cb_push_back(eye_cb, 1);
    for (uint32_t tile = worker; tile < batch; tile += workers) {
        // The two CB IDs alias one allocation. Advance both queues together so
        // their read/write slots agree without a duplicate read or local copy.
        cb_reserve_back(neg_cb, 1);
        cb_reserve_back(neg_exact_cb, 1);
        noc_async_read(input.get_noc_addr(tile), get_write_ptr(neg_cb), bytes);
        noc_async_read_barrier();
        cb_push_back(neg_cb, 1);
        cb_push_back(neg_exact_cb, 1);
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<
        get_named_compile_time_arg_val("output_dram") == 1>(), get_common_arg_val<uint32_t>(0), bytes);
    for (uint32_t tile = worker; tile < batch; tile += workers) {
        cb_wait_front(output_cb, 1);
        noc_async_write(get_read_ptr(output_cb), output.get_noc_addr(tile), bytes);
        noc_async_write_barrier();
        cb_pop_front(output_cb, 1);
    }
#elif defined(COMPILE_FOR_TRISC)
    // Full hardware startup is only safe once, before any tile operation.
    compute_kernel_hw_startup<SrcOrder::Reverse>(neg_cb, r_cb, nr_cb);
    for (uint32_t tile = worker; tile < batch; tile += workers) {
        add_eye(neg_exact_cb, r_cb);
        for (uint32_t iteration = 0; iteration < 30; ++iteration) {
            multiply();
            // The last add packs directly into the output queue. There is no
            // extra arithmetic, column scaling or unrounded DEST feedback.
            add_eye(nr_cb, iteration == 29 ? output_cb : r_cb);
            cb_pop_front(nr_cb, 1);
        }
        cb_pop_front(neg_cb, 1);
        cb_pop_front(neg_exact_cb, 1);
    }
    cb_pop_front(eye_cb, 1);
#endif
}
