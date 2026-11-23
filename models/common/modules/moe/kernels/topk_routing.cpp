// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode MoE router tail on one data-movement core: top_k of the first num_logits
// router logits and the softmax over them, as compact [1, 32] rows in the same order.
//
// One row (rows == 1):
//   ids[r]     = selected slot id (UINT16; routed ascending, then the shared slot), zero padded
//   weights[r] = its routing weight (BF16), zero padded
// Two rows (rows == 2, speculative verification): the slots are the union of both rows'
// selections (routed ascending, then the shared slot) and ids[31] holds their count;
// Every expert then streams once for both tokens, and each token's expert sum only adds
// exact zeros for the other token's experts. Entries count..30 name distinct unused slots
// above the shared slot (zero weights) so fixed-grid consumers write only padding rows.
//
// With a shared expert (shared_slot set), logit column num_logits holds the shared-
// expert gate logit; the shared expert is always selected with weight sigmoid(gate).
// Replaces topk -> softmax -> untilize -> scatter (and the shared-gate multiply).
//
// rows == 2: the second data-movement RISC of the same core (built with ROUTING_HELPER)
// routes row 1 concurrently and hands its ids/weights over through cb_handoff.
//
// The data-movement cores have no FPU (soft-float ABI): the float softmax measured
// 11 us of library calls. It runs in fixed point instead (hardware 32x32->64 multiply):
// logits are exact in Q16 for |x| >= 2^-9, exp is a 32-entry 2^(-i/32) table times a
// quadratic (relative error < 2e-6), and the result is rounded once to BF16.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_scratch = 0;
constexpr uint32_t cb_handoff = 1;

// Row 1's routing as published by the helper RISC.
template <uint32_t top_k>
struct Handoff {
    uint32_t ids[top_k];
    uint16_t weights[top_k];
    uint16_t shared_weight;
};

#include "models/common/modules/moe/kernels/routing_math.hpp"

void kernel_main() {
    constexpr uint32_t num_logits = get_compile_time_arg_val(0);
    constexpr uint32_t top_k = get_compile_time_arg_val(1);
    constexpr uint32_t shared_slot = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    constexpr bool has_shared = shared_slot != 0;
    constexpr auto logits_args = TensorAccessorArgs<4>();
    constexpr auto ids_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    static_assert(rows == 1 || rows == 2);
    static_assert(num_logits % 32 == 0 && rows * top_k + (has_shared ? 1 : 0) <= 31);
    static_assert(!has_shared || shared_slot >= num_logits);
    constexpr uint32_t tiles = (num_logits + (has_shared ? 1 : 0) + 31) / 32;
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t row_bytes = 64;  // [1, 32] 16-bit row
    constexpr uint32_t logits_row_bytes = tiles * 64;

    const auto logits = TensorAccessor(logits_args, get_arg_val<uint32_t>(0), tile_bytes);

    // Row t of each logits tile: columns 0-15 in face 0, 16-31 in face 1 (32 B each).
    const uint32_t scratch = get_write_ptr(cb_scratch);
    auto read_row = [&](uint32_t r) {
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint64_t src = get_noc_addr(t, logits);
            const uint32_t dst = scratch + r * logits_row_bytes + t * 64;
            noc_async_read(src + r * 32, dst, 32);
            noc_async_read(src + 512 + r * 32, dst + 32, 32);
        }
    };

#ifdef ROUTING_HELPER
    static_assert(rows == 2);
    read_row(1);
    noc_async_read_barrier();
    cb_reserve_back(cb_handoff, 1);
    auto* handoff = reinterpret_cast<Handoff<top_k>*>(get_write_ptr(cb_handoff));
    uint32_t helper_ids[top_k];
    uint16_t helper_weights[top_k];
    uint16_t helper_shared = 0;
    route<num_logits, top_k, has_shared>(scratch + logits_row_bytes, helper_ids, helper_weights, helper_shared);
    for (uint32_t k = 0; k < top_k; ++k) {
        handoff->ids[k] = helper_ids[k];
        handoff->weights[k] = helper_weights[k];
    }
    handoff->shared_weight = helper_shared;
    cb_push_back(cb_handoff, 1);
#else
    const auto ids = TensorAccessor(ids_args, get_arg_val<uint32_t>(1), row_bytes);
    const auto weights_out = TensorAccessor(weights_args, get_arg_val<uint32_t>(2), row_bytes);
    read_row(0);
    const uint32_t ids_row = scratch + rows * logits_row_bytes;
    const uint32_t weights_row = ids_row + row_bytes;
    auto* id_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_row);
    auto* weight_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_row);
    for (uint32_t i = 0; i < row_bytes / 4; ++i) {
        id_words[i] = 0;
    }
    for (uint32_t i = 0; i < rows * row_bytes / 4; ++i) {
        weight_words[i] = 0;
    }
    noc_async_read_barrier();

    auto* id_values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ids_row);
    auto* weight_values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weights_row);
    uint32_t token_ids[rows][top_k];
    uint16_t token_weights[rows][top_k];
    uint16_t shared_weights[rows];
    route<num_logits, top_k, has_shared>(scratch, token_ids[0], token_weights[0], shared_weights[0]);
    if constexpr (rows == 2) {
        cb_wait_front(cb_handoff, 1);
        const auto* handoff = reinterpret_cast<const volatile Handoff<top_k>*>(get_read_ptr(cb_handoff));
        for (uint32_t k = 0; k < top_k; ++k) {
            token_ids[1][k] = handoff->ids[k];
            token_weights[1][k] = handoff->weights[k];
        }
        shared_weights[1] = handoff->shared_weight;
        cb_pop_front(cb_handoff, 1);
    }

    uint32_t count = 0;
    if constexpr (rows == 1) {
        for (uint32_t k = 0; k < top_k; ++k) {
            id_values[k] = static_cast<uint16_t>(token_ids[0][k]);
            weight_values[k] = token_weights[0][k];
        }
        count = top_k;
    } else {
        // Ascending union of both tokens' routed ids, each token's weight or zero.
        uint32_t i = 0, j = 0;
        while (i < top_k || j < top_k) {
            const uint32_t a = i < top_k ? token_ids[0][i] : 0xffffffffu;
            const uint32_t b = j < top_k ? token_ids[1][j] : 0xffffffffu;
            const uint32_t id = a < b ? a : b;
            id_values[count] = static_cast<uint16_t>(id);
            if (a == id) {
                weight_values[count] = token_weights[0][i++];
            }
            if (b == id) {
                weight_values[32 + count] = token_weights[1][j++];
            }
            ++count;
        }
    }
    if constexpr (has_shared) {
        // The shared slot is above every routed id, so it goes last.
        id_values[count] = static_cast<uint16_t>(shared_slot);
        for (uint32_t r = 0; r < rows; ++r) {
            weight_values[32 * r + count] = shared_weights[r];
        }
        ++count;
    }
    if constexpr (rows > 1) {
        // Union padding: distinct never-selected slots above the shared slot.
        static_assert(has_shared, "two-row routing pads with the slots above the shared slot");
        for (uint32_t k = count; k < 31; ++k) {
            id_values[k] = static_cast<uint16_t>(shared_slot + 1 + (k - count));
        }
        id_values[31] = static_cast<uint16_t>(count);
    }

    noc_async_write(ids_row, get_noc_addr(0, ids), row_bytes);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_write(weights_row + r * row_bytes, get_noc_addr(r, weights_out), row_bytes);
    }
    noc_async_write_barrier();
#endif
}
