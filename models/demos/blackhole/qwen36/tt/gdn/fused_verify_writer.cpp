// SPDX-License-Identifier: Apache-2.0
// Preserve every prefix while streaming each BF16 output row to its norm worker.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(2);
    constexpr auto candidate_args = TensorAccessorArgs<3>();
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t rx = get_arg_val<uint32_t>(1 + num_tokens);
    const uint32_t ry = get_arg_val<uint32_t>(2 + num_tokens);
    const uint32_t inbox = get_arg_val<uint32_t>(3 + num_tokens);
    auto* request = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    const uint64_t ready = get_noc_addr(rx, ry, get_semaphore(1));
    Noc noc;
    auto send_output = [&]() {
        CircularBuffer cb(16);
        cb.wait_front(Vt);
        noc_semaphore_wait(request, 1);
        noc_semaphore_set(request, 0);
        const uint32_t src = cb.get_read_ptr();
        for (uint32_t tile = 0; tile < Vt; ++tile) {
            noc_async_write(src + tile * 2048, get_noc_addr(rx, ry, inbox + tile * 64), 32);
            noc_async_write(src + tile * 2048 + 512, get_noc_addr(rx, ry, inbox + tile * 64 + 32), 32);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(ready, 1);
        cb.pop_front(Vt);
    };
    auto write_state = [&](const auto& accessor) {
        CircularBuffer cb(17);
        cb.wait_front(Kt * Vt);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t tile = 0; tile < Kt * Vt; ++tile) {
            noc.async_write(src, accessor, 2048, {.offset_bytes = tile * 2048}, {.page_id = head * Kt * Vt + tile});
        }
        noc.async_write_barrier();
        cb.pop_front(Kt * Vt);
    };
    for (uint32_t token = 0; token < num_tokens; ++token) {
        const auto candidate = TensorAccessor(candidate_args, get_arg_val<uint32_t>(1 + token), 2048);
        send_output();
        write_state(candidate);
        #ifdef TREE_FORK_POSITION
        if (token == 7) {
            // The reader owns cb5's producer counters. Let it reload only after
            // final7 has vacated the shared cb5/cb17 storage.
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2)), 1);
        }
        #endif
    }
}
