// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t row_bytes = get_named_compile_time_arg_val("row_bytes");
    constexpr uint32_t pages = get_named_compile_time_arg_val("pages_per_slot");
    constexpr uint32_t copies = get_named_compile_time_arg_val("copies");
    constexpr uint32_t workers = get_named_compile_time_arg_val("workers");
    constexpr uint32_t stripes = row_bytes / 256;
    constexpr uint32_t per_slot = pages * stripes;
    const auto source = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), row_bytes);
    const auto destination = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), row_bytes);
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t i = get_arg_val<uint32_t>(0); i < copies * per_slot; i += workers) {
        const uint32_t copy = i / per_slot;
        const uint32_t local = i % per_slot;
        const uint32_t page = local / stripes;
        const uint32_t offset = (local % stripes) * 256;
        const uint32_t src_slot = get_common_arg_val<uint32_t>(2 + copy);
        const uint32_t dst_slot = get_common_arg_val<uint32_t>(2 + copies + copy);
        noc_async_read(get_noc_addr(src_slot * pages + page, source) + offset, scratch, 256);
        noc_async_read_barrier();
        noc_async_write(scratch, get_noc_addr(dst_slot * pages + page, destination) + offset, 256);
        noc_async_write_barrier();
    }
#endif
}
