// SPDX-License-Identifier: Apache-2.0
// Prefix only: the pinned multicast/reduction communication remains unchanged.
#define kernel_main residual_norm_native_reader
#ifdef RN_SENDER
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp"
#else
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp"
#endif
#undef kernel_main

void kernel_main() {
    constexpr uint32_t tiles = get_named_compile_time_arg_val("rn_tiles");
    constexpr uint32_t rt_offset = get_named_compile_time_arg_val("rn_rt_offset");
    constexpr auto args = TensorAccessorArgs<get_named_compile_time_arg_val("rn_accessor_offset")>();
    const auto source = TensorAccessor(args, get_common_arg_val<uint32_t>(0), 2048);
    const uint32_t first_tile = get_arg_val<uint32_t>(rt_offset);
#ifdef RN_RESIDUAL
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
    noc_semaphore_wait(ready, 1);
    noc_semaphore_set(ready, 0);
#endif
    cb_reserve_back(0, tiles);
    const uint32_t base = get_write_ptr(0);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        noc_async_read_tile(first_tile + tile, source, base + tile * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(0, tiles);
    residual_norm_native_reader();
    // Native reduction receivers issue nonposted semaphore increments too.
    noc_async_atomic_barrier();
}
