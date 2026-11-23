// SPDX-License-Identifier: Apache-2.0
// Integer-word layout transport; native BF16 output arithmetic is unchanged.
#if defined(COMPILE_FOR_TRISC)
// The included image snapshot differs only in its final entry symbol.
#include "native_output_entry.cpp"
#include "layout_transport.hpp"
void kernel_main() {
    bulk_state_tilize();
    bulk_native_output_main();
    bulk_state_untilize();
}
#elif defined(COMPILE_FOR_BRISC)
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "state_transport.hpp"
void kernel_main() {
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t px = get_arg_val<uint32_t>(1), py = get_arg_val<uint32_t>(2);
    const uint32_t inbox = get_common_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(3);
    constexpr bool gate_dram = get_named_compile_time_arg_val("gate_is_dram") == 1;
    // Mega-gate mode: gate tiles live at this whole-tile column offset inside
    // the shared mega projection row rather than a dedicated gate tensor.
    constexpr uint32_t gate_page_offset = get_named_compile_time_arg_val("gate_page_offset");
    constexpr bool gamma_dram = get_named_compile_time_arg_val("gamma_is_dram") == 1;
    constexpr bool output_dram = get_named_compile_time_arg_val("output_is_dram") == 1;
    const auto gate = TensorAccessor(tensor_accessor::make_interleaved_dspec<gate_dram>(), gate_addr, 2048);
    const auto gamma = TensorAccessor(tensor_accessor::make_interleaved_dspec<gamma_dram>(), gamma_addr, 2048);
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<output_dram>(), output_addr, 2048);
    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint64_t request = get_noc_addr(px, py, get_semaphore(0));
    const uint32_t state_addr = get_arg_val<uint32_t>(3);
    auto* input_request = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
    auto* final_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5));
    auto* pointer_mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6));
    noc_semaphore_wait(input_request, 1);
    noc_semaphore_set(input_request, 0);
    const uint32_t input_destination = *pointer_mailbox;
    noc_semaphore_set(pointer_mailbox, 0);

    cb_reserve_back(20, 16);
    bulk_state_transfer<4, false>(state_addr, get_write_ptr(20));
    noc_async_read_barrier();
    cb_push_back(20, 16);
    cb_wait_front(21, 16);
    noc_async_write(get_read_ptr(21), get_noc_addr(px, py, input_destination), 16 * 2048);
    noc_async_write_barrier();
    cb_pop_front(21, 16);
    noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(3)), 1);

    // Reverse CB22 aliases the empty TILE21 buffer, with independent counters.
    // Its actual pointer is published before final-state transfer starts.
    cb_reserve_back(22, 16);
    *pointer_mailbox = get_write_ptr(22);
    noc_async_write(get_semaphore(6), get_noc_addr(px, py, get_semaphore(6)), sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_set(pointer_mailbox, 0);
    noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(4)), 1);

    generate_bcast_col_scalar(CircularBuffer(4), 0x358637bdu);
    cb_reserve_back(0, 4); cb_reserve_back(1, 4); cb_reserve_back(2, 4);
    const uint32_t x_dst = get_write_ptr(0), gate_dst = get_write_ptr(1), gamma_dst = get_write_ptr(2);
    // Native M1 initializes all padding, not just the live row.
    for (uint32_t tile = 0; tile < 4; ++tile) {
        volatile uint32_t* zeros = reinterpret_cast<volatile uint32_t*>(x_dst + tile * 2048);
        for (uint32_t i = 0; i < 2048 / 4; ++i) zeros[i] = 0;
    }
    noc_semaphore_inc(request, 1);
    for (uint32_t tile = 0; tile < 4; ++tile) {
        noc_async_read(get_noc_addr(gate_page_offset + head * 4 + tile, gate), gate_dst + tile * 2048, 2048);
        noc_async_read(get_noc_addr(tile, gamma), gamma_dst + tile * 2048, 2048);
    }
    noc_semaphore_wait(ready, 1);
    noc_semaphore_set(ready, 0);
    for (uint32_t tile = 0; tile < 4; ++tile) {
        noc_async_read(get_noc_addr(inbox + tile * 64), x_dst + tile * 2048, 32);
        noc_async_read(get_noc_addr(inbox + tile * 64 + 32), x_dst + tile * 2048 + 512, 32);
    }
    noc_async_read_barrier();
    cb_push_back(0, 4); cb_push_back(1, 4); cb_push_back(2, 4);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<3, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    cb_wait_front(16, 4);
    const uint32_t src = get_read_ptr(16);
    // packed_rows=0: preserve native M1 full-tile writes, including padding.
    for (uint32_t tile = 0; tile < 4; ++tile) {
        noc_async_write(src + tile * 2048, get_noc_addr(head * 4 + tile, output), 2048);
    }
    noc_async_write_barrier();
    cb_pop_front(16, 4);

    // State TILE payload may arrive while the native output arithmetic runs,
    // but only its completion semaphore publishes it to the untilize worker.
    noc_semaphore_wait(final_ready, 1);
    noc_semaphore_set(final_ready, 0);
    cb_push_back(22, 16);
    cb_wait_front(23, 16);
    bulk_state_transfer<4, true>(state_addr, get_read_ptr(23));
    noc_async_write_barrier();
    cb_pop_front(23, 16);
    noc_async_atomic_barrier();
    // Program completion, not output-row publication, is the state commit.
}
#else
void kernel_main() {}
#endif
