// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t group = get_compile_time_arg_val(1);
    constexpr auto args = TensorAccessorArgs<2>();
    const auto output = TensorAccessor(args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t first_tile = get_arg_val<uint32_t>(1);
    cb_wait_front(2, tiles);
    const uint32_t base = get_read_ptr(2);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        noc_async_write_tile(first_tile + tile, output, base + tile * 2048);
    }
    // Native RMSNorm may only read the residual after the original BF16 pack
    // and all its DRAM writes are complete. No FP32 residual is ever exposed.
    noc_async_write_barrier();
    for (uint32_t k = 0; k < group; ++k) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(2 + 2 * k), get_arg_val<uint32_t>(3 + 2 * k), get_semaphore(3)), 1);
    }
    cb_pop_front(2, tiles);
    // Complete the semaphore transaction before another program reuses this NoC.
    noc_async_atomic_barrier();
}
