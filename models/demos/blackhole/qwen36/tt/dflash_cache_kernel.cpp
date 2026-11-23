// SPDX-License-Identifier: Apache-2.0
// Committed-only ring ownership, exact BF16 row DMA, and absolute-position masks.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t mode = get_named_compile_time_arg_val("mode");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t feature_tiles = get_named_compile_time_arg_val("feature_tiles");
    constexpr uint32_t mask_token_id = get_named_compile_time_arg_val("mask_token_id");
    constexpr uint32_t window = 2048;
    constexpr uint32_t tile_bytes = 2048;
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
            noc_async_write(scratch, get_noc_addr(0, owners), window * 4);
            noc_async_write_barrier();
            auto* indices = reinterpret_cast<volatile uint32_t*>(meta);
            const auto feature_positions = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                          get_common_arg_val<uint32_t>(5), 128);
            for (uint32_t i = 0; i < 32; ++i) indices[i] = i < count ? start + i : 0;
            noc_async_write(meta, get_noc_addr(0, feature_positions), 128);
            noc_async_write_barrier();
            if constexpr (mode == 1) {
                const auto noise_positions = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                            get_common_arg_val<uint32_t>(6), 128);
                for (uint32_t i = 0; i < 32; ++i) indices[i] = i < rows ? position + i : 0;
                noc_async_write(meta, get_noc_addr(0, noise_positions), 128);
                noc_async_write_barrier();
                constexpr uint32_t token_bytes = ((rows * 4 + 31) / 32) * 32;
                const auto tokens = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                   get_common_arg_val<uint32_t>(7), token_bytes);
                for (uint32_t i = 0; i < token_bytes / 4; ++i) indices[i] = i == 0 ? anchor : mask_token_id;
                noc_async_write(meta, get_noc_addr(0, tokens), token_bytes);
                noc_async_write_barrier();
                const auto mask = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                 get_common_arg_val<uint32_t>(8), tile_bytes);
                auto* tile = reinterpret_cast<volatile uint16_t*>(meta);
                for (uint32_t column_tile = 0; column_tile < (window + 32) / 32; ++column_tile) {
                    for (uint32_t r = 0; r < 32; ++r) {
                        // Padding queries see the final real query's mask: no all-inf softmax rows.
                        const int64_t qpos = int64_t(position) + (r < rows ? r : rows - 1);
                        for (uint32_t c = 0; c < 32; ++c) {
                            const uint32_t column = column_tile * 32 + c;
                            const uint32_t owner = column < window ? absolute[column] : position + c;
                            const int64_t delta = qpos - int64_t(owner);
                            const bool live = column < window
                                ? owner != 0xffffffff && owner < position
                                : c < rows;
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
    } else if constexpr (mode == 4) {
        // Aligned 32-row prefill staging. Input bucket length and offset are not
        // compilation parameters, and physical tile padding covers the tail.
        const auto offset_tensor = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                  get_common_arg_val<uint32_t>(2), 32);
        noc_async_read(get_noc_addr(0, offset_tensor), scratch, 32);
        noc_async_read_barrier();
        const uint32_t offset = *reinterpret_cast<volatile uint32_t*>(scratch);
        const auto source = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                           get_common_arg_val<uint32_t>(0), tile_bytes);
        const auto destination = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                get_common_arg_val<uint32_t>(1), tile_bytes);
        for (uint32_t tile = task; tile < feature_tiles; tile += 64) {
            noc_async_read(get_noc_addr((offset / 32) * feature_tiles + tile, source), scratch, tile_bytes);
            noc_async_read_barrier();
            noc_async_write(scratch, get_noc_addr(tile, destination), tile_bytes);
            noc_async_write_barrier();
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
            const uint32_t destination_tile = (head * (window / 32) + slot / 32) * 4 + column;
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
    }
#endif
}
