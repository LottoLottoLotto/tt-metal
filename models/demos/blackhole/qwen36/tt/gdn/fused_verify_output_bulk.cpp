// SPDX-License-Identifier: Apache-2.0
// Verify output worker with M1's bulk state transport prologue: bulk-read the
// paired head's live row-major state, tilize it as UInt16 words and deliver the
// 16 tiles into the recurrence worker's reserved state CB. The output arithmetic
// that follows is the unchanged native output kernel.
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/blackhole/qwen36/tt/gdn/m1/native_output_entry.cpp"
#include "models/demos/blackhole/qwen36/tt/gdn/m1/layout_transport.hpp"
void kernel_main() {
    bulk_state_tilize();
    bulk_native_output_main();
}
#elif defined(COMPILE_FOR_BRISC)
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "models/demos/blackhole/qwen36/tt/gdn/m1/state_transport.hpp"
static inline uint32_t row_offset(uint32_t row, bool right) {
    return (row >= 16 ? 1024 : 0) + (right ? 512 : 0) + (row & 15) * 32;
}
void kernel_main() {
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t px = get_arg_val<uint32_t>(1), py = get_arg_val<uint32_t>(2);
    const uint32_t state_addr = get_arg_val<uint32_t>(3);
    const uint32_t inbox = get_common_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(3);
    constexpr bool gate_dram = get_named_compile_time_arg_val("gate_is_dram") == 1;
    constexpr bool gamma_dram = get_named_compile_time_arg_val("gamma_is_dram") == 1;
    constexpr bool output_dram = get_named_compile_time_arg_val("output_is_dram") == 1;
    const auto gate = TensorAccessor(tensor_accessor::make_interleaved_dspec<gate_dram>(), gate_addr, 2048);
    const auto gamma = TensorAccessor(tensor_accessor::make_interleaved_dspec<gamma_dram>(), gamma_addr, 2048);
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<output_dram>(), output_addr, 2048);

    // State transport: request (sem 3) carries the destination via mailbox sem 5.
    auto* state_request = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
    auto* mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5));
    noc_semaphore_wait(state_request, 1);
    noc_semaphore_set(state_request, 0);
    const uint32_t state_destination = *mailbox;
    noc_semaphore_set(mailbox, 0);
    cb_reserve_back(20, 16);
    bulk_state_transfer<4, false>(state_addr, get_write_ptr(20));
    noc_async_read_barrier();
    cb_push_back(20, 16);
    cb_wait_front(21, 16);
    noc_async_write(get_read_ptr(21), get_noc_addr(px, py, state_destination), 16 * 2048);
    noc_async_write_barrier();
    cb_pop_front(21, 16);
    noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(4)), 1);

    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint64_t request = get_noc_addr(px, py, get_semaphore(0));
    generate_bcast_col_scalar(CircularBuffer(4), 0x358637bdu);
    constexpr uint32_t num_tokens = get_named_compile_time_arg_val("num_tokens");
    // Every token's recurrent row lands on its own row of one set of x tiles, so the
    // row-independent RMSNorm/gate compute runs once for all rows (bitwise the per-row
    // result). The next request goes out as soon as the mailbox row has been copied,
    // letting the recurrence core stream rows without waiting on the output compute.
    cb_reserve_back(0, 4); cb_reserve_back(1, 4); cb_reserve_back(2, 4);
    const uint32_t x_dst = get_write_ptr(0), gate_dst = get_write_ptr(1), gamma_dst = get_write_ptr(2);
    noc_semaphore_inc(request, 1);
    for (uint32_t tile = 0; tile < 4; ++tile) {
        noc_async_read(get_noc_addr(head * 4 + tile, gate), gate_dst + tile * 2048, 2048);
        noc_async_read(get_noc_addr(tile, gamma), gamma_dst + tile * 2048, 2048);
    }
    for (uint32_t token = 0; token < num_tokens; ++token) {
        noc_semaphore_wait(ready, 1);
        noc_semaphore_set(ready, 0);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            noc_async_read(get_noc_addr(inbox + tile * 64), x_dst + tile * 2048 + row_offset(token, false), 32);
            noc_async_read(get_noc_addr(inbox + tile * 64 + 32), x_dst + tile * 2048 + row_offset(token, true), 32);
        }
        noc_async_read_barrier();
        // A new request acknowledges that the previous mailbox contents are no longer read.
        if (token + 1 < num_tokens) {
            noc_semaphore_inc(request, 1);
        }
    }
    cb_push_back(0, 4); cb_push_back(1, 4); cb_push_back(2, 4);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<3, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    cb_wait_front(16, 4);
    const uint32_t src = get_read_ptr(16);
    for (uint32_t tile = 0; tile < 4; ++tile) {
        const uint64_t dst = get_noc_addr(head * 4 + tile, output);
        for (uint32_t token = 0; token < num_tokens; ++token) {
            noc_async_write(src + tile * 2048 + row_offset(token, false), dst + row_offset(token, false), 32);
            noc_async_write(src + tile * 2048 + row_offset(token, true), dst + row_offset(token, true), 32);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(16, 4);
    // Drain the semaphore increments before exit (see fused_verify_writer.cpp).
    noc_async_atomic_barrier();
}
#else
void kernel_main() {}
#endif
