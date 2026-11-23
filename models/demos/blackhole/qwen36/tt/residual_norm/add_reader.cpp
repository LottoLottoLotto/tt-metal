// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto residual_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), 2048);
    const auto residual = TensorAccessor(residual_args, get_arg_val<uint32_t>(1), 2048);
    const uint32_t first_tile = get_arg_val<uint32_t>(2);
    cb_reserve_back(0, tiles);
    cb_reserve_back(1, tiles);
    const uint32_t x_base = get_write_ptr(0);
    const uint32_t residual_base = get_write_ptr(1);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        noc_async_read_tile(first_tile + tile, x, x_base + tile * 2048);
        noc_async_read_tile(first_tile + tile, residual, residual_base + tile * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(0, tiles);
    cb_push_back(1, tiles);
}
