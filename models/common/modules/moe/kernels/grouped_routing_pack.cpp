// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Gather each group's 32 token rows as whole row-major rows (one contiguous read per
// token), tilize them in compute, and write the group's tiles. Row-sized reads keep the
// DRAM transaction count ~128x below per-face-row gathers, which bound this stage.
// Workers take whole groups round-robin; empty group slots (0xffffffff) are zero rows.
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#endif

constexpr uint32_t cb_rows = 0, cb_tiles = 1, cb_count = 2, cb_map = 3, cb_weight = 4;

void kernel_main() {
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t expert_width = get_named_compile_time_arg_val("expert_width");
    constexpr uint32_t row_bytes = width_tiles * 64;
    const uint32_t worker = get_arg_val<uint32_t>(5);
    const uint32_t workers = get_arg_val<uint32_t>(6);
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto map_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto experts_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<experts_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto group_map = TensorAccessor(map_args, get_arg_val<uint32_t>(1), 256);
    const auto experts = TensorAccessor(experts_args, get_arg_val<uint32_t>(2), expert_width * 4);
#endif
#if defined(COMPILE_FOR_NCRISC)
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), row_bytes);
    cb_reserve_back(cb_count, 1);
    noc_async_read(get_noc_addr(0, experts), get_write_ptr(cb_count), 64);
    noc_async_read_barrier();
    const uint32_t count = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_count));
    cb_push_back(cb_count, 1);
    const uint32_t map_base = get_write_ptr(cb_map);
    auto* tokens = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(map_base);
    for (uint32_t g = worker; g < count; g += workers) {
        noc_async_read(get_noc_addr(g, group_map), map_base, 128);
        noc_async_read_barrier();
        cb_reserve_back(cb_rows, width_tiles);
        const uint32_t rows = get_write_ptr(cb_rows);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t token = tokens[r];
            const uint32_t dst = rows + r * row_bytes;
            if (token == 0xffffffffu) {
                for (uint32_t off = 0; off < row_bytes; off += MEM_ZEROS_SIZE) {
                    noc_async_read(get_noc_addr(MEM_ZEROS_BASE), dst + off, MEM_ZEROS_SIZE);
                }
            } else {
                noc_async_read(get_noc_addr(token, x), dst, row_bytes);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_rows, width_tiles);
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(3), 2048);
    const auto weights = TensorAccessor(weights_args, get_arg_val<uint32_t>(4), 2048);
    const uint32_t weight_map = get_write_ptr(cb_weight);
    const uint32_t weight_tile = weight_map + 128;
    noc_async_read(get_noc_addr(0, experts), weight_map, 64);
    noc_async_read_barrier();
    const uint32_t count = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weight_map);
    auto* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weight_map);
    auto* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weight_tile);
    for (uint32_t g = worker; g < count; g += workers) {
        // The group's routing weight rows: each row repeats its BF16 weight across columns.
        noc_async_read(get_noc_addr(g, group_map) + 128, weight_map, 128);
        noc_async_read_barrier();
        noc_async_writes_flushed();
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t pair = values[r] | (values[r] << 16);
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t dst = (r / 16) * 256 + h * 128 + (r % 16) * 8;
                for (uint32_t j = 0; j < 8; ++j) tile[dst + j] = pair;
            }
        }
        noc_async_write(weight_tile, get_noc_addr(g, weights), 2048);
        cb_wait_front(cb_tiles, width_tiles);
        const uint32_t src = get_read_ptr(cb_tiles);
        for (uint32_t c = 0; c < width_tiles; ++c) {
            noc_async_write(src + c * 2048, get_noc_addr(g * width_tiles + c, output), 2048);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_tiles, width_tiles);
    }
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    compute_kernel_hw_startup(cb_rows, cb_tiles);
    cb_wait_front(cb_count, 1);
    const uint32_t count = read_tile_value(cb_count, 0, 0);
    const uint32_t mine = worker < count ? (count - worker + workers - 1) / workers : 0;
    if (mine) {
        compute_kernel_lib::tilize<width_tiles, cb_rows, cb_tiles>(mine);
    }
#endif
}
