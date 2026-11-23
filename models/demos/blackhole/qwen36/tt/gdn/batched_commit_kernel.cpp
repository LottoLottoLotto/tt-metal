// SPDX-License-Identifier: Apache-2.0
// Multi-layer MTP commit: select one prefix candidate per layer, untilize it
// and write it into live L1 state. Identical bytes to the per-layer commit;
// layers are pipelined (NCRISC reads, TRISC untilizes, BRISC writes).
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#endif

// Common runtime args:
//   [0] flag address
//   [1 + S*L + ...] layer L (S = 2*candidates + 2): state candidates, conv
//                   candidates, live state, live conv addresses
//   [1 + S*layers + s] packed (x | y << 16) NoC coordinates of state shard s
//   [1 + S*layers + state_cores + L] packed conv core of layer L
// Per-core runtime args: [head, part]; part selects 32-row blocks {2*part, 2*part+1}.
void kernel_main() {
    constexpr uint32_t layers = get_named_compile_time_arg_val("layers");
    constexpr uint32_t shard_rows = get_named_compile_time_arg_val("shard_rows");
    constexpr uint32_t state_cores = get_named_compile_time_arg_val("state_cores");
    constexpr uint32_t blocks = 2;
    constexpr uint32_t block_tiles = 4;
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    constexpr uint32_t candidates = get_named_compile_time_arg_val("candidates");
    constexpr uint32_t stride = 2 * candidates + 2;
#endif
#if defined(COMPILE_FOR_NCRISC)
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t part = get_arg_val<uint32_t>(1);
    const uint32_t scratch = get_write_ptr(2);
    constexpr bool flag_dram = get_named_compile_time_arg_val("flag_dram");
    const auto flag = TensorAccessor(tensor_accessor::make_interleaved_dspec<flag_dram>(), get_common_arg_val<uint32_t>(0), 32);
    noc_async_read(get_noc_addr(0, flag), scratch, 32);
    noc_async_read_barrier();
    const uint32_t selected = *reinterpret_cast<volatile uint32_t*>(scratch);
    const uint32_t accepted = candidates == 2 ? (selected != 0) : selected;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const auto candidate = TensorAccessor(
            tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1 + stride * layer + accepted), 2048);
        for (uint32_t block = 0; block < blocks; ++block) {
            cb_reserve_back(0, block_tiles);
            const uint32_t input = get_write_ptr(0);
            const uint32_t first = head * 16 + (part * blocks + block) * block_tiles;
            for (uint32_t tile = 0; tile < block_tiles; ++tile) {
                noc_async_read(get_noc_addr(first + tile, candidate), input + tile * 2048, 2048);
            }
            noc_async_read_barrier();
            cb_push_back(0, block_tiles);
        }
    }
#elif defined(COMPILE_FOR_BRISC)
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t part = get_arg_val<uint32_t>(1);
    const uint32_t worker = head * 2 + part;
    const uint32_t scratch = get_write_ptr(3);
    constexpr bool flag_dram = get_named_compile_time_arg_val("flag_dram");
    const auto flag = TensorAccessor(tensor_accessor::make_interleaved_dspec<flag_dram>(), get_common_arg_val<uint32_t>(0), 32);
    noc_async_read(get_noc_addr(0, flag), scratch, 32);
    noc_async_read_barrier();
    const uint32_t selected = *reinterpret_cast<volatile uint32_t*>(scratch);
    const uint32_t accepted = candidates == 2 ? (selected != 0) : selected;
    constexpr uint32_t coords = 1 + stride * layers;
    constexpr uint32_t conv_row_bytes = 10240 * sizeof(uint16_t);
    constexpr uint32_t stripe_bytes = 512;
    const uint32_t stripe_offset = worker * stripe_bytes;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const uint32_t state_addr = get_common_arg_val<uint32_t>(1 + stride * layer + 2 * candidates);
        for (uint32_t block = 0; block < blocks; ++block) {
            cb_wait_front(16, block_tiles);
            const uint32_t output = get_read_ptr(16);
            const uint32_t block_row = head * 128 + (part * blocks + block) * 32;
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t global_row = block_row + row;
                const uint32_t shard = global_row / shard_rows;
                const uint32_t xy = get_common_arg_val<uint32_t>(coords + shard);
                noc_async_write(
                    output + row * 256,
                    get_noc_addr(xy & 0xffff, xy >> 16, state_addr + (global_row % shard_rows) * 256),
                    256);
            }
            noc_async_write_barrier();
            cb_pop_front(16, block_tiles);
        }
        // Forty workers each own one disjoint 512-byte stripe of the three conv rows.
        if (stripe_offset < conv_row_bytes) {
            const auto conv = TensorAccessor(
                tensor_accessor::make_interleaved_dspec<true>(),
                get_common_arg_val<uint32_t>(1 + stride * layer + candidates + accepted), conv_row_bytes);
            const uint32_t conv_addr = get_common_arg_val<uint32_t>(1 + stride * layer + 2 * candidates + 1);
            const uint32_t xy = get_common_arg_val<uint32_t>(coords + state_cores + layer);
            for (uint32_t row = 0; row < 3; ++row) {
                noc_async_read(get_noc_addr(row, conv) + stripe_offset, scratch + row * stripe_bytes, stripe_bytes);
            }
            noc_async_read_barrier();
            for (uint32_t row = 0; row < 3; ++row) {
                noc_async_write(
                    scratch + row * stripe_bytes,
                    get_noc_addr(xy & 0xffff, xy >> 16, conv_addr + row * conv_row_bytes + stripe_offset),
                    stripe_bytes);
            }
            noc_async_write_barrier();
        }
    }
#elif defined(COMPILE_FOR_TRISC)
    compute_kernel_hw_startup(0, 16);
    compute_kernel_lib::untilize<4, 0, 16,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(layers * blocks);
#endif
}
