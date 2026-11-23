// SPDX-License-Identifier: Apache-2.0
// Preserve every prefix while streaming each BF16 output row to its norm worker.
// GROUP_ROWS (shared verify rounds; args: head, candidate tensor [T,48,128,128], rx, ry, inbox,
// GROUP_ROWS slot state addresses, row map): a single-row group's state is final and goes straight
// to its slot (in place); every other row's state is candidate row t, committed later per group.
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
#ifdef GROUP_ROWS
    constexpr uint32_t out_arg = 2;
#else
    constexpr uint32_t out_arg = 1 + num_tokens;
#endif
    const uint32_t rx = get_arg_val<uint32_t>(out_arg);
    const uint32_t ry = get_arg_val<uint32_t>(out_arg + 1);
    const uint32_t inbox = get_arg_val<uint32_t>(out_arg + 2);
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
    auto write_state = [&](const auto& accessor, uint32_t page_base) {
        CircularBuffer cb(17);
        cb.wait_front(Kt * Vt);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t tile = 0; tile < Kt * Vt; ++tile) {
            noc.async_write(src, accessor, 2048, {.offset_bytes = tile * 2048}, {.page_id = page_base + tile});
        }
        noc.async_write_barrier();
        cb.pop_front(Kt * Vt);
    };
#ifdef GROUP_ROWS
    const uint32_t candidate_addr = get_arg_val<uint32_t>(1);
    uint32_t row_map[num_tokens];
    {
        // Writer-private scratch past the reader's row-map copy in the reader-private staging CB.
        const uint32_t map_l1 = get_write_ptr(12) + 19584;
        const auto map = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_arg_val<uint32_t>(5 + GROUP_ROWS), 128);
        noc_async_read(get_noc_addr(0, map), map_l1, 128);
        noc_async_read_barrier();
        for (uint32_t t = 0; t < num_tokens; ++t) {
            row_map[t] = reinterpret_cast<volatile uint32_t*>(map_l1)[t];
        }
    }
    for (uint32_t token = 0; token < num_tokens; ++token) {
        send_output();
        const uint32_t word = row_map[token];
        const bool active = (word >> 31) != 0;
        const bool single = active && ((word >> 8) & 0xff) == 1;
        if (!active) {
            // Inactive row: nothing to keep (multi-row groups occupy the first candidate rows).
            CircularBuffer cb(17);
            cb.wait_front(Kt * Vt);
            cb.pop_front(Kt * Vt);
        } else if (single) {
            write_state(TensorAccessor(candidate_args, get_arg_val<uint32_t>(5 + ((word >> 16) & 0xff)), 2048),
                        head * Kt * Vt);
        } else {
            write_state(TensorAccessor(candidate_args, candidate_addr, 2048), (token * 48 + head) * Kt * Vt);
        }
    }
#else
    for (uint32_t token = 0; token < num_tokens; ++token) {
        const auto candidate = TensorAccessor(candidate_args, get_arg_val<uint32_t>(1 + token), 2048);
        send_output();
        write_state(candidate, head * Kt * Vt);
        #ifdef TREE_FORK_POSITION
        if (token == 7) {
            // The reader owns cb5's producer counters. Let it reload only after
            // final7 has vacated the shared cb5/cb17 storage.
            noc_semaphore_set(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2)), 1);
        }
        #endif
    }
#endif
    // Drain the ready increments: a late acknowledgement after the next program on this core and NoC took its
    // counter baseline would leave that program's atomic barrier waiting forever.
    noc_async_atomic_barrier();
}
