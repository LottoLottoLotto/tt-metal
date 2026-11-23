// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr auto args = TensorAccessorArgs<1>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t first_tile = get_arg_val<uint32_t>(1);
    const uint32_t norm_x = get_arg_val<uint32_t>(2);
    const uint32_t norm_y = get_arg_val<uint32_t>(3);
    cb_wait_front(2, tiles);
    const uint32_t base = get_read_ptr(2);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        noc_async_write_tile(first_tile + tile, output, base + tile * 2048);
    }
    // Native RMSNorm may only read the residual after the original BF16 pack
    // and all four DRAM writes are complete. No FP32 residual is ever exposed.
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(norm_x, norm_y, get_semaphore(3)), 1);
    cb_pop_front(2, tiles);
    // Complete the semaphore transaction before another program reuses this NoC.
    noc_async_atomic_barrier();
}
