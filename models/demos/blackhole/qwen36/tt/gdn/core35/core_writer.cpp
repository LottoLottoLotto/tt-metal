// SPDX-License-Identifier: Apache-2.0
// Fused GDN core, recurrence worker R(h, j) writer (BRISC).
//   o:      row 0 of this worker's Vt output tiles -> the head's output worker inbox (64 B per tile and token),
//           then one semaphore increment there once every token's rows have landed;
//   T == 1: the untilized updated state rows (V-half j, 128 B each) -> the live row-major L1 state in place;
//   T >= 2: every token's tiled candidate state (Kt x Vt tiles) -> its DRAM candidate [1, HV, 128, 128].
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t cb_out = 16, cb_final = 17;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);  // local V tiles
    constexpr uint32_t num_tokens = get_compile_time_arg_val(2);
    constexpr uint32_t out_token_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t full_vt = get_compile_time_arg_val(4);  // V tiles per head (4)
    constexpr auto dst_a = TensorAccessorArgs<5>();  // live state (T == 1) or candidate 0 (T >= 2)

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t j = get_arg_val<uint32_t>(1);
    const uint32_t ox = get_arg_val<uint32_t>(2);
    const uint32_t oy = get_arg_val<uint32_t>(3);
    const uint32_t inbox = get_arg_val<uint32_t>(4);

    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;
    Noc noc;

    auto send_output = [&](uint32_t token) {
        CircularBuffer cb(cb_out);
        {
            DeviceZoneScopedN("R-wout");
            cb.wait_front(Vt);
        }
        const uint32_t src = cb.get_read_ptr();
        const uint32_t dst = inbox + token * out_token_bytes + j * Vt * 64;
        for (uint32_t t = 0; t < Vt; ++t) {
            noc_async_write(src + t * tile_bytes, get_noc_addr(ox, oy, dst + t * 64), 32);
            noc_async_write(src + t * tile_bytes + face_bytes, get_noc_addr(ox, oy, dst + t * 64 + 32), 32);
        }
        noc_async_write_barrier();
        cb.pop_front(Vt);
    };

    if constexpr (num_tokens == 1) {
        constexpr uint32_t stick_bytes = 128 * sizeof(uint16_t);
        constexpr uint32_t slice_bytes = Vt * 32 * sizeof(uint16_t);
        const auto state = TensorAccessor(dst_a, get_arg_val<uint32_t>(5), stick_bytes);
        send_output(0);
        noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(1)), 1);
        CircularBuffer cb(cb_final);
        for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
            cb.wait_front(Vt);
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t state_row = h * Kt * 32 + tile_row * 32 + row;
                noc.async_write(cb, state, slice_bytes, {.offset_bytes = row * slice_bytes},
                                {.page_id = state_row, .offset_bytes = j * slice_bytes});
            }
            noc.async_write_barrier();
            cb.pop_front(Vt);
        }
    } else {
        for (uint32_t token = 0; token < num_tokens; ++token) {
            send_output(token);
            if (token + 1 == num_tokens) {
                noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(1)), 1);
            }
            const auto cand = TensorAccessor(dst_a, get_arg_val<uint32_t>(5 + token), tile_bytes);
            CircularBuffer cb(cb_final);
            cb.wait_front(Kt * Vt);
            auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
            for (uint32_t kr = 0; kr < Kt; ++kr) {
                for (uint32_t vc = 0; vc < Vt; ++vc) {
                    const uint32_t page = (h * Kt + kr) * full_vt + j * Vt + vc;
                    noc.async_write(src, cand, tile_bytes, {.offset_bytes = (kr * Vt + vc) * tile_bytes},
                                    {.page_id = page});
                }
            }
            noc.async_write_barrier();
            cb.pop_front(Kt * Vt);
        }
    }
    noc_async_atomic_barrier();
}
