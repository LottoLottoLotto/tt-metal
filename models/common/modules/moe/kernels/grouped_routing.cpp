// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Parallel token routing with the decode router's identical fixed-point arithmetic.
// Masked prefill skips padding rows entirely; their route records are never read.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#include "models/common/modules/moe/kernels/prefill_rows.hpp"
#include "models/common/modules/moe/kernels/routing_math.hpp"

void kernel_main() {
    constexpr uint32_t num_logits = get_compile_time_arg_val(0);
    constexpr uint32_t top_k = get_compile_time_arg_val(1);
    constexpr uint32_t shared_slot = get_compile_time_arg_val(2);
    constexpr uint32_t tokens = get_compile_time_arg_val(3);
    constexpr bool masked = get_compile_time_arg_val(4);
    constexpr auto logits_args = TensorAccessorArgs<5>();
    constexpr auto routes_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr uint32_t columns = num_logits / 32 + 1;
    const auto logits = TensorAccessor(logits_args, get_arg_val<uint32_t>(0), 2048);
    const auto routes = TensorAccessor(routes_args, get_arg_val<uint32_t>(1), 128);
    const uint32_t worker = get_arg_val<uint32_t>(2);
    const uint32_t workers = get_arg_val<uint32_t>(3);
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t row_addr = scratch + columns * 2048;
    const uint32_t record_addr = row_addr + columns * 64;
    auto* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(row_addr);
    auto* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
    auto* record = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(record_addr);
    PrefillRows real;
    if constexpr (masked) {
        real.load(get_arg_val<uint32_t>(4), record_addr + 128);
    }
    for (uint32_t i = 0; i < 32; ++i) record[i] = 0;
    for (uint32_t tile_row = worker; tile_row < (tokens + 31) / 32; tile_row += workers) {
        const uint32_t rows = tokens - tile_row * 32 < 32 ? tokens - tile_row * 32 : 32;
        if constexpr (masked) {
            bool any = false;
            for (uint32_t r = 0; r < rows && !any; ++r) any = real.valid(tile_row * 32 + r);
            if (!any) continue;
        }
        for (uint32_t c = 0; c < columns; ++c) {
            noc_async_read(get_noc_addr(tile_row * columns + c, logits), scratch + c * 2048, 2048);
        }
        noc_async_read_barrier();
        for (uint32_t r = 0; r < rows; ++r) {
            if constexpr (masked) {
                if (!real.valid(tile_row * 32 + r)) continue;
            }
            for (uint32_t c = 0; c < columns; ++c) {
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t src = c * 1024 + (r / 16) * 512 + h * 256 + (r % 16) * 16;
                    for (uint32_t j = 0; j < 16; ++j) row[c * 32 + h * 16 + j] = tile[src + j];
                }
            }
            uint32_t ids[top_k];
            uint16_t weights[top_k];
            uint16_t shared_weight = 0;
            route<num_logits, top_k, true>(row_addr, ids, weights, shared_weight);
            for (uint32_t k = 0; k < top_k; ++k) record[k] = ids[k] | (uint32_t(weights[k]) << 16);
            record[top_k] = shared_slot | (uint32_t(shared_weight) << 16);
            noc_async_write(record_addr, get_noc_addr(tile_row * 32 + r, routes), 128);
            noc_async_write_barrier();
        }
    }
}
