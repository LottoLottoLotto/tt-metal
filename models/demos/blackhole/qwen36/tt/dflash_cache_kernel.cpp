// SPDX-License-Identifier: Apache-2.0
// Committed-only ring ownership, exact BF16 row DMA, and absolute-position masks.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t mode = get_named_compile_time_arg_val("mode");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    // Batched drafting (mode 1): this ring's user owns query rows [lane*rows, (lane+1)*rows) of a shared
    // multi-user block and its noise K/V sits at the same tail rows; other rows/columns are masked.
    constexpr uint32_t lane = get_named_compile_time_arg_val("lane");
    constexpr uint32_t ring_slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t window = 2048;
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t kMaskWorkers = (window + 32) / 32;  // one per mask column tile (mode 1)
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t task = get_arg_val<uint32_t>(0);
    if constexpr (mode == 0 || mode == 1 || mode == 3) {
        // One worker owns the entire position vector, including reset. No epoch
        // overflow and no cross-core read/modify/write on ownership metadata.
        const auto owners = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                           get_common_arg_val<uint32_t>(0), window * 4);
        auto* absolute = reinterpret_cast<volatile uint32_t*>(scratch);
        if constexpr (mode == 0) {
            for (uint32_t i = 0; i < window; ++i) absolute[i] = 0xffffffff;
            noc_async_write(scratch, get_noc_addr(0, owners), window * 4);
            noc_async_write_barrier();
        } else {
            const uint32_t meta = scratch + window * 4;
            for (uint32_t i = 0; i < 4; ++i) {
                const auto source = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                   get_common_arg_val<uint32_t>(1 + i), 32);
                noc_async_read(get_noc_addr(0, source), meta + i * 64, 32);
            }
            noc_async_read(get_noc_addr(0, owners), scratch, window * 4);
            noc_async_read_barrier();
            const uint32_t position = *reinterpret_cast<volatile uint32_t*>(meta);
            const uint32_t start = *reinterpret_cast<volatile uint32_t*>(meta + 64);
            const uint32_t count = *reinterpret_cast<volatile uint32_t*>(meta + 128);
            const uint32_t anchor = *reinterpret_cast<volatile uint32_t*>(meta + 192);
            for (uint32_t i = 0; i < count; ++i) absolute[(start + i) % window] = start + i;
            // Draft preparation (mode 1) runs one worker per mask column tile. Every worker
            // applies the same idempotent ownership update locally (a concurrent re-read of
            // worker 0's write-back yields identical values); worker 0 alone writes metadata.
            if (task == 0) {
                noc_async_write(scratch, get_noc_addr(0, owners), window * 4);
                noc_async_write_barrier();
            }
            auto* indices = reinterpret_cast<volatile uint32_t*>(meta);
            const auto feature_positions = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                          get_common_arg_val<uint32_t>(5), 128);
            if (task == 0) {
                for (uint32_t i = 0; i < 32; ++i) indices[i] = i < count ? start + i : 0;
                noc_async_write(meta, get_noc_addr(0, feature_positions), 128);
                noc_async_write_barrier();
            }
            if constexpr (mode == 1) {
              if (task == 0) {
                const auto noise_positions = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                            get_common_arg_val<uint32_t>(6), 128);
                for (uint32_t i = 0; i < 32; ++i) indices[i] = i < rows ? position + i : 0;
                noc_async_write(meta, get_noc_addr(0, noise_positions), 128);
                noc_async_write_barrier();
                constexpr uint32_t token_bytes = ((rows * 4 + 31) / 32) * 32;
                const auto tokens = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                   get_common_arg_val<uint32_t>(7), token_bytes);
                for (uint32_t i = 0; i < token_bytes / 4; ++i) indices[i] = i == 0 ? anchor : 248070;
                noc_async_write(meta, get_noc_addr(0, tokens), token_bytes);
                noc_async_write_barrier();
              }
                const auto mask = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                 get_common_arg_val<uint32_t>(8), tile_bytes);
                auto* tile = reinterpret_cast<volatile uint16_t*>(meta);
                for (uint32_t column_tile = task; column_tile < (window + 32) / 32; column_tile += kMaskWorkers) {
                    for (uint32_t r = 0; r < 32; ++r) {
                        // Padding queries see the final real query's mask: no all-inf softmax rows.
                        const uint32_t local = r < lane * rows ? 0 : (r - lane * rows < rows ? r - lane * rows : rows - 1);
                        const int64_t qpos = int64_t(position) + local;
                        for (uint32_t c = 0; c < 32; ++c) {
                            const uint32_t column = column_tile * 32 + c;
                            const bool own_noise = c >= lane * rows && c < (lane + 1) * rows;
                            const uint32_t owner = column < window ? absolute[column] : position + (own_noise ? c - lane * rows : 0);
                            const int64_t delta = qpos - int64_t(owner);
                            const bool live = column < window
                                ? owner != 0xffffffff && owner < position
                                : own_noise;
                            const bool visible = live && delta < window && delta > -int64_t(window);
                            const uint32_t offset = (r / 16) * 512 + (c / 16) * 256 + (r % 16) * 16 + c % 16;
                            tile[offset] = visible ? 0 : 0xff80;  // BF16 negative infinity
                        }
                    }
                    noc_async_write(meta, get_noc_addr(column_tile, mask), tile_bytes);
                    noc_async_write_barrier();
                }
            }
        }
    } else if constexpr (mode == 2) {
        // One worker per head/dimension tile owns every row in that tile.
        const uint32_t head = task / 4;
        const uint32_t column = task % 4;
        const auto start_tensor = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                 get_common_arg_val<uint32_t>(4), 32);
        const auto count_tensor = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                 get_common_arg_val<uint32_t>(5), 32);
        noc_async_read(get_noc_addr(0, start_tensor), scratch + 128, 32);
        noc_async_read(get_noc_addr(0, count_tensor), scratch + 192, 32);
        noc_async_read_barrier();
        const uint32_t start = *reinterpret_cast<volatile uint32_t*>(scratch + 128);
        const uint32_t count = *reinterpret_cast<volatile uint32_t*>(scratch + 192);
        for (uint32_t row = 0; row < count; ++row) {
            const uint32_t slot = (start + row) % window;
            // Ring heads hold window + 32 rows: the tail tile row stages the draft's noise K/V.
            const uint32_t destination_tile = (head * (window / 32 + 1) + slot / 32) * 4 + column;
            const uint32_t source_tile = (head * ((rows + 31) / 32) + row / 32) * 4 + column;
            const uint32_t source_row = row % 32;
            const uint32_t source_offset = (source_row / 16) * 1024 + (source_row % 16) * 32;
            const uint32_t destination_offset = ((slot % 32) / 16) * 1024 + (slot % 16) * 32;
            for (uint32_t kind = 0; kind < 2; ++kind) {
                const auto destination = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                        get_common_arg_val<uint32_t>(kind), tile_bytes);
                const auto source = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                   get_common_arg_val<uint32_t>(2 + kind), tile_bytes);
                const uint64_t src = get_noc_addr(source_tile, source) + source_offset;
                const uint64_t dst = get_noc_addr(destination_tile, destination) + destination_offset;
                const uint32_t half_line = uint32_t(src) & 63;
                noc_async_read(src & ~uint64_t(63), scratch, 64);
                noc_async_read((src & ~uint64_t(63)) + 512, scratch + 64, 64);
                noc_async_read_barrier();
                noc_async_write(scratch + half_line, dst, 32);
                noc_async_write(scratch + 64 + half_line, dst + 512, 32);
                noc_async_write_barrier();
            }
        }
    } else if constexpr (mode == 6) {
        // Prefill append: source rows [offset, offset+count) of [1,8,rows,128] to ring slots
        // start.. (mod window). start and offset are tile-row aligned, so every full source tile
        // maps onto one ring tile and moves whole; a partial final tile moves row by row.
        const uint32_t head = task / 4;
        const uint32_t column = task % 4;
        for (uint32_t i = 0; i < 3; ++i) {
            const auto scalar = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                               get_common_arg_val<uint32_t>(4 + i), 32);
            noc_async_read(get_noc_addr(0, scalar), scratch + i * 64, 32);
        }
        noc_async_read_barrier();
        const uint32_t start = *reinterpret_cast<volatile uint32_t*>(scratch);
        const uint32_t count = *reinterpret_cast<volatile uint32_t*>(scratch + 64);
        const uint32_t offset = *reinterpret_cast<volatile uint32_t*>(scratch + 128);
        constexpr uint32_t source_tiles = (rows + 31) / 32;
        for (uint32_t kind = 0; kind < 2; ++kind) {
            const auto destination = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                    get_common_arg_val<uint32_t>(kind), tile_bytes);
            const auto source = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                               get_common_arg_val<uint32_t>(2 + kind), tile_bytes);
            for (uint32_t row = 0; row < count;) {
                const uint32_t source_row = offset + row;
                const uint32_t slot = (start + row) % window;
                const uint32_t source_tile = (head * source_tiles + source_row / 32) * 4 + column;
                const uint32_t destination_tile = (head * (window / 32 + 1) + slot / 32) * 4 + column;
                if (source_row % 32 == 0 && slot % 32 == 0 && count - row >= 32) {
                    noc_async_read(get_noc_addr(source_tile, source), scratch, tile_bytes);
                    noc_async_read_barrier();
                    noc_async_write(scratch, get_noc_addr(destination_tile, destination), tile_bytes);
                    noc_async_write_barrier();
                    row += 32;
                    continue;
                }
                const uint32_t in_tile = source_row % 32;
                const uint32_t source_offset = (in_tile / 16) * 1024 + (in_tile % 16) * 32;
                const uint32_t destination_offset = ((slot % 32) / 16) * 1024 + (slot % 16) * 32;
                const uint64_t src = get_noc_addr(source_tile, source) + source_offset;
                const uint64_t dst = get_noc_addr(destination_tile, destination) + destination_offset;
                const uint32_t half_line = uint32_t(src) & 63;
                noc_async_read(src & ~uint64_t(63), scratch, 64);
                noc_async_read((src & ~uint64_t(63)) + 512, scratch + 64, 64);
                noc_async_read_barrier();
                noc_async_write(scratch + half_line, dst, 32);
                noc_async_write(scratch + 64 + half_line, dst + 512, 32);
                noc_async_write_barrier();
                row += 1;
            }
        }
    } else if constexpr (mode == 7) {
        // Shared-round append: row r of key/value [1,8,rows,128] whose ring-table word (bit 31 | slot << 16) names
        // a ring slot < ring_slots goes to that ring at positions[r] % window. One worker per head/dimension tile.
        // Args: ring_slots K rings, ring_slots V rings, key, value, ring table [1,32], positions [rows].
        const uint32_t head = task / 4;
        const uint32_t column = task % 4;
        const auto table = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                          get_common_arg_val<uint32_t>(2 * ring_slots + 2), 128);
        const auto positions = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                              get_common_arg_val<uint32_t>(2 * ring_slots + 3), rows * 4);
        noc_async_read(get_noc_addr(0, table), scratch + 256, 128);
        noc_async_read(get_noc_addr(0, positions), scratch + 384, rows * 4);
        noc_async_read_barrier();
        const auto* words = reinterpret_cast<volatile uint32_t*>(scratch + 256);
        const auto* row_positions = reinterpret_cast<volatile uint32_t*>(scratch + 384);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t word = words[row];
            const uint32_t ring = (word >> 16) & 0xff;
            if ((word >> 31) == 0 || ring >= ring_slots) continue;
            const uint32_t slot = row_positions[row] % window;
            const uint32_t destination_tile = (head * (window / 32 + 1) + slot / 32) * 4 + column;
            const uint32_t source_tile = (head * ((rows + 31) / 32) + row / 32) * 4 + column;
            const uint32_t source_row = row % 32;
            const uint32_t source_offset = (source_row / 16) * 1024 + (source_row % 16) * 32;
            const uint32_t destination_offset = ((slot % 32) / 16) * 1024 + (slot % 16) * 32;
            for (uint32_t kind = 0; kind < 2; ++kind) {
                const auto destination = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                        get_common_arg_val<uint32_t>(kind * ring_slots + ring), tile_bytes);
                const auto source = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                   get_common_arg_val<uint32_t>(2 * ring_slots + kind), tile_bytes);
                const uint64_t src = get_noc_addr(source_tile, source) + source_offset;
                const uint64_t dst = get_noc_addr(destination_tile, destination) + destination_offset;
                const uint32_t half_line = uint32_t(src) & 63;
                noc_async_read(src & ~uint64_t(63), scratch, 64);
                noc_async_read((src & ~uint64_t(63)) + 512, scratch + 64, 64);
                noc_async_read_barrier();
                noc_async_write(scratch + half_line, dst, 32);
                noc_async_write(scratch + 64 + half_line, dst + 512, 32);
                noc_async_write_barrier();
            }
        }
    }
#endif
}
