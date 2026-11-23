// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Parallel 1-2 row decode routing with batched_routing.cpp's exact (selection, weights). The top-k scan is split
// over one core per 32-logit tile and row (core (tile, row)). Each scan core keeps its tile's TopK and posts the
// survivors, ascending by id, to its row's finisher (tile 0 of the row). The finisher replays all candidates through
// the same TopK in ascending id order -- a logit outside its tile's top_k is beaten by top_k logits of that tile
// under (key desc, id asc), so the final list and its order equal the full 256-logit scan -- then runs the identical
// fixed-point softmax / shared-gate math (route_finish) and posts the token record to the union core (row 0's
// finisher), which builds batched_routing.cpp's ascending union.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
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
    constexpr uint32_t tiles = experts / 32;  // scan cores per row
    static_assert(experts % 32 == 0 && top_k <= 32 && rows <= 2 && width > rows * top_k + 1);
    // Scratch (same layout on every core): own 32 logits; own tile record; finisher: tile-record inbox, logit values
    // by id (+ shared gate at index `experts`), own token record; union core: token-record inbox, union ids, per-row
    // weights, expert positions.
    constexpr uint32_t record_bytes = ((top_k * 4 + 15) / 16) * 16;
    constexpr uint32_t staging_offset = 64;
    constexpr uint32_t inbox_offset = staging_offset + record_bytes;
    constexpr uint32_t values_offset = inbox_offset + tiles * record_bytes;
    constexpr uint32_t token_offset = values_offset + ((experts + 32) * 2 + 15) / 16 * 16;
    constexpr uint32_t token_bytes = (((top_k + 1) * 4 + 15) / 16) * 16;
    constexpr uint32_t records_offset = token_offset + token_bytes;
    constexpr uint32_t ids_offset = records_offset + rows * token_bytes;
    constexpr uint32_t weights_offset = ids_offset + width * 2;
    constexpr uint32_t position_offset = weights_offset + rows * width * 2;
    const uint32_t base = get_write_ptr(0);
    const auto logits = TensorAccessor(logits_args, get_arg_val<uint32_t>(0), 2048);
    const uint32_t tile = get_arg_val<uint32_t>(3);
    const uint32_t row = get_arg_val<uint32_t>(4);
    const uint32_t finish_x = get_arg_val<uint32_t>(5), finish_y = get_arg_val<uint32_t>(6);
    const uint32_t union_x = get_arg_val<uint32_t>(7), union_y = get_arg_val<uint32_t>(8);
    const bool finisher = tile == 0;
    const bool union_core = finisher && row == 0;
    auto* tiles_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    auto* rows_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));

    // decode_linear returns L1 logits: face-row reads need only 16-byte alignment.
    {
    DeviceZoneScopedN("R-read");
    const uint64_t src = get_noc_addr(tile, logits) + row * 32;
    noc_async_read(src, base, 32);
    noc_async_read(src + 512, base + 32, 32);
    if (finisher) {
        // The shared gate is the first logit of the extra tile.
        noc_async_read(get_noc_addr(tiles, logits) + row * 32, base + values_offset + experts * 2, 16);
    }
    if (union_core) {
        // Unselected union entries and weights stay zero (zeroed by the NOC, off the RISC).
        constexpr uint32_t clear_bytes = (rows + 1) * width * 2;
        for (uint32_t off = 0; off < clear_bytes; off += MEM_ZEROS_SIZE) {
            const uint32_t bytes = clear_bytes - off < MEM_ZEROS_SIZE ? clear_bytes - off : MEM_ZEROS_SIZE;
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), base + ids_offset + off, bytes);
        }
    }
    noc_async_read_barrier();
    }

    // This tile's top_k, scanned in ascending id order (two logits per 32-bit load).
    auto* pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    TopK<top_k> best;
    for (uint32_t e = 0; e < 32; e += 2) {
        const uint32_t pair = pairs[e / 2];
        best.consider(order_key(pair & 0xffffu), e);
        best.consider(order_key(pair >> 16), e + 1);
    }
    // Survivors in ascending local id, packed (bf16 bits << 16 | local id); unused entries 0xffffffff.
    uint32_t sorted[top_k];
    const uint32_t n = best.count;
    for (uint32_t k = 0; k < n; ++k) {
        uint32_t pos = k;
        while (pos > 0 && sorted[pos - 1] > best.best_id[k]) {
            sorted[pos] = sorted[pos - 1];
            --pos;
        }
        sorted[pos] = best.best_id[k];
    }
    auto* local = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    auto* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + staging_offset);
    for (uint32_t k = 0; k < top_k; ++k) {
        out[k] = k < n ? (static_cast<uint32_t>(local[sorted[k]]) << 16) | sorted[k] : 0xffffffffu;
    }
    noc_async_write(
        base + staging_offset, get_noc_addr(finish_x, finish_y, base + inbox_offset + tile * record_bytes), record_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(finish_x, finish_y, get_semaphore(0)), 1);
    noc_async_atomic_barrier();
    if (!finisher) {
        return;
    }

    {
    DeviceZoneScopedN("R-wait");
    noc_semaphore_wait(tiles_arrived, tiles);
    }
    auto* records = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + inbox_offset);
    auto* values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + values_offset);
    TopK<top_k> merged;
    for (uint32_t t = 0; t < tiles; ++t) {
        for (uint32_t k = 0; k < top_k; ++k) {
            const uint32_t entry = records[t * (record_bytes / 4) + k];
            if (entry == 0xffffffffu) {
                continue;
            }
            const uint32_t id = t * 32 + (entry & 0xffffu);
            const uint32_t bits = entry >> 16;
            values[id] = static_cast<uint16_t>(bits);
            merged.consider(order_key(bits), id);
        }
    }
    uint32_t ids[top_k];
    uint16_t weights[top_k];
    uint16_t shared_weight;
    {
    DeviceZoneScopedN("R-finish");
    route_finish<experts, top_k, true>(base + values_offset, merged.best_id, ids, weights, shared_weight);
    }
    auto* token = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + token_offset);
    for (uint32_t k = 0; k < top_k; ++k) {
        token[k] = (static_cast<uint32_t>(weights[k]) << 16) | ids[k];
    }
    token[top_k] = (static_cast<uint32_t>(shared_weight) << 16) | shared;
    noc_async_write(
        base + token_offset, get_noc_addr(union_x, union_y, base + records_offset + row * token_bytes), token_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(union_x, union_y, get_semaphore(1)), 1);
    noc_async_atomic_barrier();
    if (!union_core) {
        return;
    }

    // Union exactly as batched_routing.cpp's merge core.
    {
    DeviceZoneScopedN("R-rows");
    noc_semaphore_wait(rows_arrived, rows);
    }
    DeviceZoneScopedN("R-union");
    const auto selection = TensorAccessor(selection_args, get_arg_val<uint32_t>(1), width * 2);
    const auto routing = TensorAccessor(weights_args, get_arg_val<uint32_t>(2), width * 2);
    auto* tokens = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + records_offset);
    auto* union_ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + ids_offset);
    auto* union_weights = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + weights_offset);
    auto* position = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base + position_offset);
    constexpr uint32_t token_words = token_bytes / 4;
    uint32_t present[experts / 32] = {};
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < top_k; ++k) {
            const uint32_t id = tokens[r * token_words + k] & 0xffffu;
            present[id / 32] |= 1u << (id % 32);
        }
    }
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
            const uint32_t record = tokens[r * token_words + k];
            union_weights[r * width + position[record & 0xffffu]] = static_cast<uint16_t>(record >> 16);
        }
        union_weights[r * width + count] = static_cast<uint16_t>(tokens[r * token_words + top_k] >> 16);
    }
    union_ids[count] = static_cast<uint16_t>(shared);
    union_ids[width - 1] = static_cast<uint16_t>(count + 1);
    noc_async_write(base + ids_offset, get_noc_addr(0, selection), width * 2);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_write(base + weights_offset + r * width * 2, get_noc_addr(r, routing), width * 2);
    }
    noc_async_write_barrier();
}
