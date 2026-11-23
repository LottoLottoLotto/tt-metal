// SPDX-License-Identifier: Apache-2.0
// Head split/merge for one tile row: [1,1,R,H*128] and [1,H,R,128] (R <= 32) share the same
// interleaved tile order (tile h*4+j), so the transform is a tile-for-tile copy. Rows >= R of
// every tile are written as zeros so padding never carries stale values. Destination tile of
// source tile t is (t / group) * stride + offset + t % group (identity: stride = group, offset 0),
// which also places [1,H,R,128] noise rows into the tail tile row of an [1,H,S+32,128] ring.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t pairs = RETILE_PAIRS;
    constexpr uint32_t workers = RETILE_WORKERS;
    constexpr uint32_t rows = RETILE_ROWS;
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t scratch = get_write_ptr(0);
    uint32_t first = 0;
    for (uint32_t p = 0; p < pairs; ++p) {
        const uint32_t tiles = get_common_arg_val<uint32_t>(6 * p + 2);
        const uint32_t group = get_common_arg_val<uint32_t>(6 * p + 3);
        const uint32_t stride = get_common_arg_val<uint32_t>(6 * p + 4);
        const uint32_t offset = get_common_arg_val<uint32_t>(6 * p + 5);
        const auto source = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6 * p), 2048);
        const auto destination = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6 * p + 1), 2048);
        // Global tile index g = first + t is owned by worker g % workers.
        uint32_t t = (worker + workers - first % workers) % workers;
        uint32_t slot = 0;
        for (; t < tiles; t += workers) {
            const uint32_t buffer = scratch + slot * 2048;
            noc_async_read(source.get_noc_addr(t), buffer, 2048);
            noc_async_read_barrier();
            if constexpr (rows < 32) {
                auto words = reinterpret_cast<volatile uint32_t*>(buffer);
                for (uint32_t face = 0; face < 4; ++face) {
                    for (uint32_t r = (face < 2 ? rows : (rows > 16 ? rows - 16 : 0)); r < 16; ++r) {
                        for (uint32_t w = 0; w < 8; ++w) words[face * 128 + r * 8 + w] = 0;
                    }
                }
            }
            noc_async_write(buffer, destination.get_noc_addr((t / group) * stride + offset + t % group), 2048);
            slot ^= 1;
            if (!slot) noc_async_write_barrier();
        }
        first += tiles;
    }
    noc_async_write_barrier();
#endif
}
