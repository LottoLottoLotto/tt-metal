// SPDX-License-Identifier: Apache-2.0
// Select one speculative prefix, untilize it, and commit directly to live L1 state.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t scratch = get_write_ptr(2);
    constexpr bool flag_dram = get_named_compile_time_arg_val("flag_dram");
    constexpr bool state_dram = get_named_compile_time_arg_val("state_dram");
    constexpr bool conv_dram = get_named_compile_time_arg_val("conv_dram");
    constexpr uint32_t shard_rows = get_named_compile_time_arg_val("shard_rows");
    constexpr uint32_t state_cores = get_named_compile_time_arg_val("state_cores");
    constexpr uint32_t num_candidates = get_named_compile_time_arg_val("num_candidates");
    constexpr uint32_t live_offset = 1 + 2 * num_candidates;
    constexpr uint32_t cores_offset = live_offset + 2;
    const auto flag = TensorAccessor(tensor_accessor::make_interleaved_dspec<flag_dram>(), get_common_arg_val<uint32_t>(0), 32);
    noc_async_read(get_noc_addr(0, flag), scratch, 32);
    noc_async_read_barrier();
    const uint32_t selected = *reinterpret_cast<volatile uint32_t*>(scratch);
    // Preserve the original boolean MTP1 API; wider calls supply a prefix index.
    const uint32_t accepted = num_candidates == 2 ? (selected != 0) : selected;
    const auto candidate = TensorAccessor(tensor_accessor::make_interleaved_dspec<state_dram>(), get_common_arg_val<uint32_t>(1 + accepted), 2048);
    constexpr uint32_t conv_row_bytes = get_named_compile_time_arg_val("conv_width") * sizeof(uint16_t);
    const auto conv = TensorAccessor(tensor_accessor::make_interleaved_dspec<conv_dram>(), get_common_arg_val<uint32_t>(1 + num_candidates + accepted), conv_row_bytes);
    const uint32_t state_addr = get_common_arg_val<uint32_t>(live_offset);
    const uint32_t conv_addr = get_common_arg_val<uint32_t>(live_offset + 1);
    for (uint32_t block = 0; block < 4; ++block) {
        cb_reserve_back(0, 4);
        const uint32_t input = get_write_ptr(0);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            noc_async_read(get_noc_addr(head * 16 + block * 4 + tile, candidate), input + tile * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(0, 4);
        cb_wait_front(16, 4);
        const uint32_t output = get_read_ptr(16);
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t global_row = head * 128 + block * 32 + row;
            const uint32_t shard = global_row / shard_rows;
            const uint32_t x = get_common_arg_val<uint32_t>(cores_offset + shard);
            const uint32_t y = get_common_arg_val<uint32_t>(cores_offset + state_cores + shard);
            noc_async_write(output + row * 256, get_noc_addr(x, y, state_addr + (global_row % shard_rows) * 256), 256);
        }
        noc_async_write_barrier();
        cb_pop_front(16, 4);
    }
    // Head workers own disjoint 512-byte stripes of the conv rows (the host checks coverage).
    // All three live convolution rows share one core and one stable base address.
    constexpr uint32_t stripe_bytes = 512;
    const uint32_t stripe_offset = head * stripe_bytes;
    if (stripe_offset < conv_row_bytes) {
        const uint32_t x = get_common_arg_val<uint32_t>(cores_offset + 2 * state_cores);
        const uint32_t y = get_common_arg_val<uint32_t>(cores_offset + 1 + 2 * state_cores);
        for (uint32_t row = 0; row < 3; ++row) {
            noc_async_read(get_noc_addr(row, conv) + stripe_offset, scratch, stripe_bytes);
            noc_async_read_barrier();
            noc_async_write(
                scratch, get_noc_addr(x, y, conv_addr + row * conv_row_bytes + stripe_offset), stripe_bytes);
            noc_async_write_barrier();
        }
    }
#elif defined(COMPILE_FOR_TRISC)
    compute_kernel_hw_startup(0, 16);
    compute_kernel_lib::untilize<4, 0, 16,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(4);
#endif
}
