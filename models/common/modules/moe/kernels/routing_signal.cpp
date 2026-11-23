// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// batched_routing.cpp (identical routing and union) as the front of a fused MoE program: once the merge core's
// selection / weight rows have landed it counts itself in on every expert core (semaphore `ready_sem`), which
// wait for it before reading the metadata.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "models/common/modules/moe/kernels/routing_math.hpp"

void kernel_main() {
    constexpr uint32_t experts = get_compile_time_arg_val(0);
    constexpr uint32_t top_k = get_compile_time_arg_val(1);
    constexpr uint32_t shared = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    constexpr uint32_t width = get_compile_time_arg_val(4);
    constexpr auto logits_args = TensorAccessorArgs<5>();
    constexpr auto selection_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<selection_args.next_compile_time_args_offset()>();
    static_assert(rows <= 16 && top_k < 32 && width > rows * top_k + 1);
    static_assert(experts % 32 == 0 && rows * top_k < 256);
    constexpr uint32_t tiles = experts / 32 + 1;
    // Scratch (same layout on every core): this core's logits row and record, then the
    // merge core's record inbox, union ids, per-row weights and expert positions.
    constexpr uint32_t records_offset = tiles * 64 + 128;
    constexpr uint32_t ids_offset = records_offset + rows * 128;
    constexpr uint32_t weights_offset = ids_offset + width * 2;
    constexpr uint32_t position_offset = weights_offset + rows * width * 2;
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(weights_args.next_compile_time_args_offset() + 1);
    const uint32_t base = get_write_ptr(scratch_cb);
    const auto logits = TensorAccessor(logits_args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t row = get_arg_val<uint32_t>(3);
    const uint32_t merge_x = get_arg_val<uint32_t>(4), merge_y = get_arg_val<uint32_t>(5);
    auto* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));

    // decode_linear returns L1 logits: face-row reads need only 16-byte alignment.
    for (uint32_t t = 0; t < tiles; ++t) {
        const uint64_t src = get_noc_addr(t, logits) + row * 32;
        noc_async_read(src, base + t * 64, 32);
        noc_async_read(src + 512, base + t * 64 + 32, 32);
    }
    if (row == 0) {
        // Unselected union entries and weights stay zero (zeroed by the NOC, off the RISC).
        constexpr uint32_t clear_bytes = (rows + 1) * width * 2;
        for (uint32_t off = 0; off < clear_bytes; off += MEM_ZEROS_SIZE) {
            const uint32_t bytes = clear_bytes - off < MEM_ZEROS_SIZE ? clear_bytes - off : MEM_ZEROS_SIZE;
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), base + ids_offset + off, bytes);
        }
    }
    noc_async_read_barrier();
    uint32_t ids[top_k];
    uint16_t weights[top_k];
    uint16_t shared_weight;
    route<experts, top_k, true>(base, ids, weights, shared_weight);
    const uint32_t result = base + tiles * 64;
    auto* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result);
    for (uint32_t k = 0; k < top_k; ++k) {
        values[k] = (static_cast<uint32_t>(weights[k]) << 16) | ids[k];
    }
    values[top_k] = (static_cast<uint32_t>(shared_weight) << 16) | shared;
    noc_async_write(result, get_noc_addr(merge_x, merge_y, base + records_offset + row * 128), ((top_k + 1) * 4 + 15) & ~15u);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(merge_x, merge_y, get_semaphore(0)), 1);
    noc_async_atomic_barrier();
    if (row != 0) {
        return;
    }

    noc_semaphore_wait(arrived, rows);
    const auto selection = TensorAccessor(selection_args, get_arg_val<uint32_t>(1), width * 2);
    const auto routing = TensorAccessor(weights_args, get_arg_val<uint32_t>(2), width * 2);
    auto* records = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + records_offset);
    auto* union_ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + ids_offset);
    auto* union_weights = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + weights_offset);
    auto* position = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base + position_offset);
    uint32_t present[experts / 32] = {};
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < top_k; ++k) {
            const uint32_t id = records[r * 32 + k] & 0xffffu;
            present[id / 32] |= 1u << (id % 32);
        }
    }
    // Ascending union ids; each selected expert's union position.
    uint32_t count = 0;
    for (uint32_t word = 0; word < experts / 32; ++word) {
        for (uint32_t bits = present[word], id = word * 32; bits; bits >>= 1, ++id) {
            if (bits & 1u) {
                position[id] = static_cast<uint8_t>(count);
                union_ids[count++] = static_cast<uint16_t>(id);
            }
        }
    }
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < top_k; ++k) {
            const uint32_t record = records[r * 32 + k];
            union_weights[r * width + position[record & 0xffffu]] = static_cast<uint16_t>(record >> 16);
        }
        union_weights[r * width + count] = static_cast<uint16_t>(records[r * 32 + top_k] >> 16);
    }
    union_ids[count] = static_cast<uint16_t>(shared);
    union_ids[width - 1] = static_cast<uint16_t>(count + 1);
    noc_async_write(base + ids_offset, get_noc_addr(0, selection), width * 2);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_write(base + weights_offset + r * width * 2, get_noc_addr(r, routing), width * 2);
    }
    noc_async_write_barrier();
    constexpr uint32_t ready_sem = get_compile_time_arg_val(weights_args.next_compile_time_args_offset());
    const uint32_t experts_cores = get_arg_val<uint32_t>(6);
    for (uint32_t i = 0; i < experts_cores; ++i) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(7 + 2 * i), get_arg_val<uint32_t>(8 + 2 * i),
                                       get_semaphore(ready_sem)), 1);
    }
    noc_async_atomic_barrier();
}
