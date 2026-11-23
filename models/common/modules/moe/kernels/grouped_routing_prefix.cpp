// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Two linear passes over route records. One 256-byte partial group per expert
// bounds L1 use independently of token count. All DRAM transfers are 64B aligned.
// Masked prefill: padding rows join no group and their inverse entries are the
// 0xffffffff sentinel, which the combine turns into zero rows.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "models/common/modules/moe/kernels/prefill_rows.hpp"

void kernel_main() {
    constexpr uint32_t experts = get_compile_time_arg_val(0);
    constexpr uint32_t top_k = get_compile_time_arg_val(1);
    constexpr uint32_t shared_slot = get_compile_time_arg_val(2);
    constexpr uint32_t tokens = get_compile_time_arg_val(3);
    constexpr uint32_t expert_width = get_compile_time_arg_val(4);
    constexpr bool masked = get_compile_time_arg_val(5);
    constexpr uint32_t slots = experts + 1;
    constexpr auto routes_args = TensorAccessorArgs<6>();
    constexpr auto map_args = TensorAccessorArgs<routes_args.next_compile_time_args_offset()>();
    constexpr auto experts_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    constexpr auto inverse_args = TensorAccessorArgs<experts_args.next_compile_time_args_offset()>();
    const auto routes = TensorAccessor(routes_args, get_arg_val<uint32_t>(0), 128);
    const auto group_map = TensorAccessor(map_args, get_arg_val<uint32_t>(1), 256);
    const auto expert_ids = TensorAccessor(experts_args, get_arg_val<uint32_t>(2), expert_width * 4);
    const auto inverse = TensorAccessor(inverse_args, get_arg_val<uint32_t>(3), 128);
    const uint32_t base = get_write_ptr(0);
    const uint32_t route_base = base + slots * 256;
    const uint32_t inverse_base = route_base + 4096;
    const uint32_t metadata_base = inverse_base + 4096;
    const uint32_t counters_base = metadata_base + 128;
    auto* groups = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto* records = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(route_base);
    auto* inverse_rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inverse_base);
    auto* metadata = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_base);
    auto* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counters_base);
    auto* starts = counts + slots;
    auto* cursors = starts + slots;
    PrefillRows real;
    if constexpr (masked) {
        real.load(get_arg_val<uint32_t>(6), (reinterpret_cast<uint32_t>(cursors + slots) + 63) & ~63u);
    }
    for (uint32_t e = 0; e < slots; ++e) {
        counts[e] = 0;
        cursors[e] = 0;
    }
    for (uint32_t i = 0; i < 1024; ++i) inverse_rows[i] = 0;
    for (uint32_t i = 0; i < 32; ++i) metadata[i] = 0;
    auto read_records = [&](uint32_t begin, uint32_t n) {
        for (uint32_t r = 0; r < n; ++r) {
            noc_async_read(get_noc_addr(begin + r, routes), route_base + r * 128, 128);
        }
        noc_async_read_barrier();
    };
    for (uint32_t begin = 0; begin < tokens; begin += 32) {
        const uint32_t n = tokens - begin < 32 ? tokens - begin : 32;
        read_records(begin, n);
        for (uint32_t r = 0; r < n; ++r) {
            if constexpr (masked) {
                if (!real.valid(begin + r)) continue;
            }
            for (uint32_t k = 0; k <= top_k; ++k) {
                const uint32_t id = records[r * 32 + k] & 0xffff;
                const uint32_t e = id == shared_slot ? experts : id;
                counts[e] = counts[e] + 1;
            }
        }
    }
    uint32_t total = 0;
    for (uint32_t e = 0; e < slots; ++e) {
        starts[e] = total;
        total += (counts[e] + 31) / 32;
    }
    // The experts tensor is ONE row/page. Stream aligned chunks within that page,
    // rather than treating each chunk as an independently bank-interleaved page.
    const uint64_t expert_dst = get_noc_addr(0, expert_ids);
    metadata[0] = total;
    uint32_t entry = 1;
    for (uint32_t e = 0; e < slots; ++e) {
        for (uint32_t g = 0; g < (counts[e] + 31) / 32; ++g) {
            metadata[entry % 32] = e == experts ? shared_slot : e;
            ++entry;
            if (entry % 32 == 0) {
                noc_async_write(metadata_base, expert_dst + (entry - 32) * 4, 128);
                noc_async_write_barrier();
            }
        }
    }
    if (entry % 32 != 0) {
        for (uint32_t i = entry % 32; i < 32; ++i) metadata[i] = 0;
        noc_async_write(metadata_base, expert_dst + (entry / 32) * 128, 128);
        noc_async_write_barrier();
    }
    for (uint32_t begin = 0; begin < tokens; begin += 32) {
        const uint32_t n = tokens - begin < 32 ? tokens - begin : 32;
        read_records(begin, n);
        for (uint32_t r = 0; r < n; ++r) {
            if constexpr (masked) {
                if (!real.valid(begin + r)) {
                    for (uint32_t k = 0; k <= top_k; ++k) inverse_rows[r * 32 + k] = 0xffffffffu;
                    continue;
                }
            }
            for (uint32_t k = 0; k <= top_k; ++k) {
                const uint32_t record = records[r * 32 + k];
                const uint32_t id = record & 0xffff;
                const uint32_t e = id == shared_slot ? experts : id;
                const uint32_t cursor = cursors[e];
                const uint32_t row = cursor % 32;
                const uint32_t group = starts[e] + cursor / 32;
                groups[e * 64 + row] = begin + r;
                groups[e * 64 + 32 + row] = record >> 16;
                inverse_rows[r * 32 + k] = group * 32 + row;
                cursors[e] = cursor + 1;
                if (row == 31) {
                    noc_async_write(base + e * 256, get_noc_addr(group, group_map), 256);
                    // The same expert can immediately begin its next group.
                    noc_async_write_barrier();
                }
            }
        }
        for (uint32_t r = 0; r < n; ++r) {
            noc_async_write(inverse_base + r * 128, get_noc_addr(begin + r, inverse), 128);
        }
        noc_async_write_barrier();
    }
    for (uint32_t e = 0; e < slots; ++e) {
        const uint32_t rows = cursors[e] % 32;
        if (rows == 0) continue;
        for (uint32_t r = rows; r < 32; ++r) {
            groups[e * 64 + r] = 0xffffffffu;
            groups[e * 64 + 32 + r] = 0;
        }
        noc_async_write(base + e * 256, get_noc_addr(starts[e] + cursors[e] / 32, group_map), 256);
    }
    noc_async_write_barrier();
}
