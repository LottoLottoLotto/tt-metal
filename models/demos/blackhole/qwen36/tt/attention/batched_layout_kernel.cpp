// SPDX-License-Identifier: Apache-2.0
// Whole-tile owners gather BF16 rows; every output padding bit is initialized.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
namespace {
constexpr uint32_t tile_bytes = 2048;
FORCE_INLINE uint32_t row_offset(uint32_t row) {
    return (row / 16) * 1024 + (row % 16) * 32;
}
FORCE_INLINE void copy_half(uint32_t src, uint32_t dst) {
    const volatile uint32_t* source = reinterpret_cast<const volatile uint32_t*>(src);
    volatile uint32_t* destination = reinterpret_cast<volatile uint32_t*>(dst);
    for (uint32_t i = 0; i < 8; ++i) destination[i] = source[i];
}
}
#endif

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t batch = get_common_arg_val<uint32_t>(5);
    const uint32_t heads = get_common_arg_val<uint32_t>(6);
    const uint32_t kv_heads = get_common_arg_val<uint32_t>(7);
    const auto input = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), tile_bytes);
    const uint32_t assembled = get_write_ptr(0);
    const uint32_t scratch = get_write_ptr(1);
    Noc noc;
#if !BATCH_SPLIT
    // Positions are one row-major DRAM stick, with an allocator-aligned transfer.
    const uint32_t position_bytes = ((batch * 4 + 63) / 64) * 64;
    const auto positions = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), position_bytes);
    noc_async_read(get_noc_addr(0, positions), get_write_ptr(2), position_bytes);
    noc_async_read_barrier();
    const volatile int32_t* position = reinterpret_cast<const volatile int32_t*>(get_write_ptr(2));
#endif
    for (uint32_t task = first; task < first + count; ++task) {
        noc.async_write_zeros(CircularBuffer(0), tile_bytes);
        noc.write_zeros_l1_barrier();
#if BATCH_SPLIT
        // Four equally sized task ranges: Q, K, V, gate. Each token owns eight
        // column tiles whose rows are heads (padded to 32).
        const uint32_t kind = task / (batch * 8);
        const uint32_t tile = task % (batch * 8);
        const uint32_t token = tile / 8;
        const uint32_t column = tile % 8;
        const uint32_t rows = (kind == 1 || kind == 2) ? kv_heads : heads;
        const uint32_t width_tiles = (2 * heads + 2 * kv_heads) * 8;
        const uint32_t pair_offset = row_offset((token % 32) & ~1u);
        const uint32_t half = (token & 1u) * 32;
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t source_column = kind == 0 ? row * 16 + column
                : kind == 3 ? row * 16 + 8 + column
                : 2 * heads * 8 + (kind - 1) * kv_heads * 8 + row * 8 + column;
            const uint64_t source = get_noc_addr((token / 32) * width_tiles + source_column, input);
            noc_async_read(source + pair_offset, scratch + row * 128, 64);
            noc_async_read(source + pair_offset + 512, scratch + row * 128 + 64, 64);
        }
        noc_async_read_barrier();
        for (uint32_t row = 0; row < rows; ++row) {
            copy_half(scratch + row * 128 + half, assembled + row_offset(row));
            copy_half(scratch + row * 128 + 64 + half, assembled + row_offset(row) + 512);
        }
        const auto output = TensorAccessor(
            tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(kind + 1), tile_bytes);
        noc_async_write(assembled, get_noc_addr(tile, output), tile_bytes);
#else
        // Flatten [1,B,H,256] into [1,1,B,H*256], never reading an inactive
        // user's unwritten SDPA result (including NaNs left in reused memory).
        const uint32_t columns = heads * 8;
        const uint32_t token_base = (task / columns) * 32;
        const uint32_t head = (task % columns) / 8;
        const uint32_t column = task % 8;
        const uint32_t rows = batch - token_base < 32 ? batch - token_base : 32;
        const uint32_t pair_offset = row_offset(head & ~1u);
        const uint32_t half = (head & 1u) * 32;
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t token = token_base + row;
            if (position[token] < 0) continue;
            const uint64_t source = get_noc_addr(token * 8 + column, input);
            noc_async_read(source + pair_offset, scratch + row * 128, 64);
            noc_async_read(source + pair_offset + 512, scratch + row * 128 + 64, 64);
        }
        noc_async_read_barrier();
        for (uint32_t row = 0; row < rows; ++row) {
            if (position[token_base + row] < 0) continue;
            copy_half(scratch + row * 128 + half, assembled + row_offset(row));
            copy_half(scratch + row * 128 + 64 + half, assembled + row_offset(row) + 512);
        }
        const auto output = TensorAccessor(
            tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), tile_bytes);
        noc_async_write(assembled, get_noc_addr(task, output), tile_bytes);
#endif
        noc_async_write_barrier();
    }
#endif
}
