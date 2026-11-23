// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_newstate = 24;

// Ordinary M1 output row, then final TILE state sent to the BF16 layout worker.
void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    const uint32_t rx = get_arg_val<uint32_t>(2);
    const uint32_t ry = get_arg_val<uint32_t>(3);
    const uint32_t inbox = get_arg_val<uint32_t>(4);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    auto* request = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    const uint64_t ready = get_noc_addr(rx, ry, get_semaphore(1));

    auto send_output = [&]() {
        CircularBuffer cb(cb_out);
        cb.wait_front(Vt);
        noc_semaphore_wait(request, 1);
        noc_semaphore_set(request, 0);
        const uint32_t src = cb.get_read_ptr();
        for (uint32_t tile = 0; tile < Vt; ++tile) {
            noc_async_write(src + tile * tile_bytes, get_noc_addr(rx, ry, inbox + tile * 64), 32);
            noc_async_write(src + tile * tile_bytes + 512, get_noc_addr(rx, ry, inbox + tile * 64 + 32), 32);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(ready, 1);
        cb.pop_front(Vt);
    };

    // Output core owns CB21 after its input transfer has been consumed. The
    // pointer becomes visible before semaphore4, and is not reused thereafter.
    auto* destination_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    auto* destination_mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6));
    noc_semaphore_wait(destination_ready, 1);
    noc_semaphore_set(destination_ready, 0);
    const uint32_t destination = *destination_mailbox;
    noc_semaphore_set(destination_mailbox, 0);

    auto write_state_from_tiles = [&]() {
        static_assert(Kt == 4 && Vt == 4, "Bulk state requires 128x128 heads");
        // Overlap candidate: the recurrence compute already published the
        // rounded final BF16 tiles in cb_newstate; the copy_tiles pass into
        // cb_final is eliminated. The output-row handshake above ordered this
        // after the compute's last cb_newstate read (the output matmul that
        // produced cb_out consumed cb_newstate first).
        CircularBuffer cb(cb_newstate);
        cb.wait_front(Kt * Vt);
        noc_async_write(cb.get_read_ptr(), get_noc_addr(rx, ry, destination), Kt * Vt * tile_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(rx, ry, get_semaphore(5)), 1);
        cb.pop_front(Kt * Vt);
    };

    send_output();
    write_state_from_tiles();
    // The next program may use this NoC on a different RISC. Its acknowledgment
    // baseline must not include a late response to either semaphore increment.
    noc_async_atomic_barrier();
}
