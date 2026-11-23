// SPDX-License-Identifier: Apache-2.0
// Fused residual add + RMSNorm, data-movement RISC 0: writes this core's h tiles (residual sum) as soon as
// compute packs them, multicasts the master's 1/rms column tile to the other cores, writes the n tiles.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "residual_norm_prof.hpp"

constexpr uint32_t CB_H = 16, CB_N = 17, CB_RSTD = 28;

void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t n_cores = get_compile_time_arg_val(1);
    constexpr uint32_t sem_rstd_id = get_compile_time_arg_val(2);
    constexpr uint32_t grid_x = get_compile_time_arg_val(3);
    // Multicast rectangle (NOC 0 orientation: x0, y0 = first core, x1, y1 = last core).
    constexpr uint32_t mc_x0 = get_compile_time_arg_val(4), mc_y0 = get_compile_time_arg_val(5);
    constexpr uint32_t mc_x1 = get_compile_time_arg_val(6), mc_y1 = get_compile_time_arg_val(7);
    constexpr auto h_args = TensorAccessorArgs<8>();
    constexpr auto n_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();

    const uint32_t h_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t n_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t k = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t tile0 = k * tiles;
    RN_PROF_INIT(1);  // 0: kernel start

    constexpr uint32_t tb = 2048;
#ifndef RN_NO_RESIDUAL
    const auto h = TensorAccessor(h_args, h_addr, tb);
    cb_wait_front(CB_H, tiles);
    RN_STAMP();  // 1: h packed
    const uint32_t rh = get_read_ptr(CB_H);
    for (uint32_t i = 0; i < tiles; ++i) {
        noc_async_write_tile(tile0 + i, h, rh + i * tb);
    }
#else
    RN_STAMP();
#endif
    if constexpr (n_cores > 1) {
        if (k == 0) {
            uint32_t x0 = mc_x0, y0 = mc_y0, x1 = mc_x1, y1 = mc_y1;
            if (noc_index == 1) {
                x0 = mc_x1, y0 = mc_y1, x1 = mc_x0, y1 = mc_y0;
            }
            cb_wait_front(CB_RSTD, 1);
            RN_STAMP();  // 2 (master): rstd packed
            const uint32_t rr = get_read_ptr(CB_RSTD);
            noc_async_write_multicast(rr, get_noc_multicast_addr(x0, y0, x1, y1, rr), 4096, n_cores - 1, true);
            const uint32_t sem = get_semaphore(sem_rstd_id);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem) = 1;
            noc_semaphore_set_multicast(sem, get_noc_multicast_addr(x0, y0, x1, y1, sem), n_cores - 1);
        }
    }
    const auto n = TensorAccessor(n_args, n_addr, tb);
    cb_wait_front(CB_N, tiles);
    RN_STAMP();  // 2 / 3 (master): n packed
    const uint32_t rn = get_read_ptr(CB_N);
    for (uint32_t i = 0; i < tiles; ++i) {
        noc_async_write_tile(tile0 + i, n, rn + i * tb);
    }
    noc_async_write_barrier();
    RN_STAMP();  // 3 / 4 (master): h and n written
    cb_pop_front(CB_N, tiles);
    RN_PROF_FLUSH(n_args.next_compile_time_args_offset(), 2, 2 * k + 1);
}
