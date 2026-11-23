// SPDX-License-Identifier: Apache-2.0
// Reuse the exact output compute kernel; replace only its DRAM input path.
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/blackhole/qwen36/tt/gdn/native_output_kernel.cpp"
#elif defined(COMPILE_FOR_BRISC)
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
static inline uint32_t row_offset(uint32_t row, bool right) {
    return (row >= 16 ? 1024 : 0) + (right ? 512 : 0) + (row & 15) * 32;
}
void kernel_main() {
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t px = get_arg_val<uint32_t>(1), py = get_arg_val<uint32_t>(2);
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
    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint64_t request = get_noc_addr(px, py, get_semaphore(0));
    generate_bcast_col_scalar(CircularBuffer(4), 0x358637bdu);
    constexpr uint32_t num_tokens = get_named_compile_time_arg_val("num_tokens");
    for (uint32_t token = 0; token < num_tokens; ++token) {
        cb_reserve_back(0, 4); cb_reserve_back(1, 4); cb_reserve_back(2, 4);
        const uint32_t x_dst = get_write_ptr(0), gate_dst = get_write_ptr(1), gamma_dst = get_write_ptr(2);
        // A new request acknowledges that the previous mailbox contents are no longer read.
        noc_semaphore_inc(request, 1);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            noc_async_read(get_noc_addr(head * 4 + tile, gate), gate_dst + tile * 2048, 2048);
            noc_async_read(get_noc_addr(tile, gamma), gamma_dst + tile * 2048, 2048);
        }
        noc_semaphore_wait(ready, 1);
        noc_semaphore_set(ready, 0);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            noc_async_read(get_noc_addr(inbox + tile * 64), x_dst + tile * 2048 + row_offset(token, false), 32);
            noc_async_read(get_noc_addr(inbox + tile * 64 + 32), x_dst + tile * 2048 + row_offset(token, true), 32);
        }
        noc_async_read_barrier();
        cb_push_back(0, 4); cb_push_back(1, 4); cb_push_back(2, 4);
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<3, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
        cb_wait_front(16, 4);
        const uint32_t src = get_read_ptr(16);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            const uint64_t dst = get_noc_addr(head * 4 + tile, output);
            noc_async_write(src + tile * 2048 + row_offset(token, false), dst + row_offset(token, false), 32);
            noc_async_write(src + tile * 2048 + row_offset(token, true), dst + row_offset(token, true), 32);
        }
        noc_async_write_barrier();
        cb_pop_front(16, 4);
    }
}
#else
void kernel_main() {}
#endif
