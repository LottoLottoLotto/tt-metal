// SPDX-License-Identifier: Apache-2.0
#define kernel_main residual_norm_native_writer
#ifdef RN_WEIGHT_RM
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/writer_unary_sharded_ln_rm_gb.cpp"
#else
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/writer_unary_sharded_ln.cpp"
#endif
#undef kernel_main

void kernel_main() {
    residual_norm_native_writer();
    constexpr uint32_t tiles = get_named_compile_time_arg_val("rn_tiles");
    constexpr auto args = TensorAccessorArgs<get_named_compile_time_arg_val("rn_accessor_offset")>();
    const auto output = TensorAccessor(args, get_common_arg_val<uint32_t>(0), 2048);
    const uint32_t first_tile = get_arg_val<uint32_t>(5);
    cb_wait_front(16, tiles);
    const uint32_t base = get_read_ptr(16);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        noc_async_write_tile(first_tile + tile, output, base + tile * 2048);
    }
    noc_async_write_barrier();
    // The native compute performs its own wait_front after push_back. Do not pop
    // here: racing that wait would deadlock. Native sharded output is not popped
    // either, and the dispatch initializes the CB state on every invocation.
}
