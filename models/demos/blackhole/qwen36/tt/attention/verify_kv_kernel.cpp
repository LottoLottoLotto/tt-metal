// SPDX-License-Identifier: Apache-2.0
// Copy only speculative BF16 rows, without unpacking or modifying adjacent rows.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t heads = get_named_compile_time_arg_val("heads");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t page_rows = get_named_compile_time_arg_val("page_rows");
    constexpr uint32_t position_bytes = get_named_compile_time_arg_val("position_bytes");
    constexpr uint32_t table_bytes = get_named_compile_time_arg_val("table_bytes");
    constexpr uint32_t mode = get_named_compile_time_arg_val("mode");
    constexpr bool flag_dram = get_named_compile_time_arg_val("flag_dram");
    constexpr uint32_t fork_position = get_named_compile_time_arg_val("fork_position");
    constexpr bool winner_dram = get_named_compile_time_arg_val("winner_dram");
    const uint32_t task = get_arg_val<uint32_t>(0);
    const uint32_t head = task / width_tiles;
    const uint32_t column = task % width_tiles;
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t positions_ptr = scratch + 128;
    const uint32_t table_ptr = scratch + 256;
    const auto positions = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                          get_common_arg_val<uint32_t>(4), position_bytes);
    const auto table = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                      get_common_arg_val<uint32_t>(5), table_bytes);
    noc_async_read(get_noc_addr(0, positions), positions_ptr, ((tokens * 4 + 31) / 32) * 32);
    noc_async_read_barrier();
    uint32_t count = tokens;
    uint32_t winner = 0;
    if constexpr (mode == 2) {
        const auto flag = TensorAccessor(tensor_accessor::make_interleaved_dspec<flag_dram>(),
                                         get_common_arg_val<uint32_t>(6), 32);
        noc_async_read(get_noc_addr(0, flag), scratch, 32);
        noc_async_read_barrier();
        count = *reinterpret_cast<volatile uint32_t*>(scratch) + 1;
        if constexpr (fork_position != 0) {
            const auto branch = TensorAccessor(tensor_accessor::make_interleaved_dspec<winner_dram>(),
                                               get_common_arg_val<uint32_t>(7), 32);
            noc_async_read(get_noc_addr(0, branch), scratch, 32);
            noc_async_read_barrier();
            winner = *reinterpret_cast<volatile uint32_t*>(scratch);
        }
    }
    // One worker owns a head/dimension tile for every token, so even adjacent
    // positions in the same physical tile cannot race with another worker.
    for (uint32_t index = 0; index < count; ++index) {
        const uint32_t row = (mode == 2 && winner != 0 && index >= fork_position)
            ? 8 + index - fork_position : index;
        const uint32_t position = reinterpret_cast<volatile uint32_t*>(positions_ptr)[row];
        const uint32_t logical_page = position / page_rows;
        // Page-table rows are identical. Read an aligned 64-byte metadata stripe;
        // the selected logical page may occupy either half of that stripe.
        noc_async_read(get_noc_addr(0, table) + (logical_page / 16) * 64, table_ptr, 64);
        noc_async_read_barrier();
        const uint32_t physical_page = reinterpret_cast<volatile uint32_t*>(table_ptr)[logical_page % 16];
        const uint32_t cache_tile = ((physical_page * heads + head) * (page_rows / 32)
                                     + (position % page_rows) / 32) * width_tiles + column;
        const uint32_t cache_row = position % 32;
        const uint32_t cache_offset = (cache_row / 16) * 1024 + (cache_row % 16) * 32;
        const uint32_t candidate_offset = (row / 16) * 1024 + (row % 16) * 32;
        for (uint32_t kind = 0; kind < 2; ++kind) {
            const auto cache = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                              get_common_arg_val<uint32_t>(kind), tile_bytes);
            const auto candidate = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                  get_common_arg_val<uint32_t>(2 + kind), tile_bytes);
            const uint64_t cache_addr = get_noc_addr(cache_tile, cache) + cache_offset;
            const uint64_t candidate_addr = get_noc_addr(task, candidate) + candidate_offset;
            const uint64_t source = mode == 0 ? cache_addr : candidate_addr;
            const uint64_t destination = mode == 0 ? candidate_addr : cache_addr;
            // DRAM reads need aligned 64-byte row pairs. A 32-byte read into
            // a mismatched L1 half-line silently selects the neighbouring row.
            // Writes remain exact 32-byte face stripes; never write that neighbour.
            const uint64_t aligned_source = source & ~uint64_t(63);
            const uint32_t source_offset = uint32_t(source) & 63;
            noc_async_read(aligned_source, scratch, 64);
            noc_async_read(aligned_source + 512, scratch + 64, 64);
            noc_async_read_barrier();
            noc_async_write(scratch + source_offset, destination, 32);
            noc_async_write(scratch + 64 + source_offset, destination + 512, 32);
            noc_async_write_barrier();
        }
    }
#endif
}
