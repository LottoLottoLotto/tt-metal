// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4, cb_state = 5;
constexpr uint32_t cb_decay_bias = 6, cb_decay_scale = 7, cb_ones = 26;

#if defined(VERIFY_TWO) || defined(VERIFY_TOKENS)
void kernel_main() {
#ifdef VERIFY_TOKENS
    constexpr uint32_t num_tokens = VERIFY_TOKENS;
    static_assert(num_tokens >= 2 && num_tokens <= 32);
#else
    constexpr uint32_t num_tokens = 2;
#endif
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr auto q_a = TensorAccessorArgs<5>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto a_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto db_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto ds_a = TensorAccessorArgs<db_a.next_compile_time_args_offset()>();
    constexpr auto ones_a = TensorAccessorArgs<ds_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t a_addr = get_arg_val<uint32_t>(4);
    const uint32_t b_addr = get_arg_val<uint32_t>(5);
    const uint32_t s_addr = get_arg_val<uint32_t>(6);
    const uint32_t db_addr = get_arg_val<uint32_t>(7);
    const uint32_t ds_addr = get_arg_val<uint32_t>(8);
    const uint32_t ones_addr = get_arg_val<uint32_t>(9);
    const uint32_t BH = get_arg_val<uint32_t>(10);

    const uint32_t tile_bytes = get_tile_size(cb_q);
    constexpr uint32_t state_stick_bytes = Vt * 32 * sizeof(uint16_t);
    const auto q_acc = TensorAccessor(q_a, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_a, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_a, v_addr, tile_bytes);
    const auto a_acc = TensorAccessor(a_a, a_addr, tile_bytes);
    const auto b_acc = TensorAccessor(b_a, b_addr, tile_bytes);
    const auto s_acc = TensorAccessor(s_a, s_addr, state_stick_bytes);
    const auto db_acc = TensorAccessor(db_a, db_addr, tile_bytes);
    const auto ds_acc = TensorAccessor(ds_a, ds_addr, tile_bytes);
    const auto ones_acc = TensorAccessor(ones_a, ones_addr, tile_bytes);
    Noc noc;

    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t count) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(count);
        for (uint32_t tile = 0; tile < count; ++tile) {
            noc.async_read(
                acc,
                cb,
                tile_bytes,
                {.page_id = base + tile},
                {.offset_bytes = tile * tile_bytes});
        }
        noc.async_read_barrier();
        cb.push_back(count);
    };

    auto read_state_into_tiles = [&]() {
        constexpr uint32_t tile_width = 32;
        constexpr uint32_t face_width = 16;
        constexpr uint32_t face_row_bytes = face_width * sizeof(uint16_t);
        constexpr uint32_t face_bytes = face_width * face_width * sizeof(uint16_t);
        CircularBuffer cb(cb_state);
        cb.reserve_back(Kt * Vt);
        for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
            for (uint32_t row = 0; row < tile_width; ++row) {
                const uint32_t state_row = h * Kt * tile_width + tile_row * tile_width + row;
                const uint32_t face_row = row % face_width;
                const uint32_t face_row_group = row / face_width;
                for (uint32_t tile_col = 0; tile_col < Vt; ++tile_col) {
                    const uint32_t tile_offset = (tile_row * Vt + tile_col) * tile_bytes;
                    for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                        const uint32_t source_offset =
                            (tile_col * tile_width + face_col * face_width) * sizeof(uint16_t);
                        const uint32_t face = face_row_group * 2 + face_col;
                        const uint32_t dest_offset =
                            tile_offset + face * face_bytes + face_row * face_row_bytes;
                        noc.async_read(
                            s_acc,
                            cb,
                            face_row_bytes,
                            {.page_id = state_row, .offset_bytes = source_offset},
                            {.offset_bytes = dest_offset});
                    }
                }
            }
        }
        noc.async_read_barrier();
        cb.push_back(Kt * Vt);
    };

    // q/k/v are [T, 1, BH, D] TILE, with ceil(BH / 32) tile rows per
    // token. a/b are [1, 1, T, BH], with the head on the tiled column.
    // Gather each head into row 0 so compute retains the T=1 arithmetic.
    constexpr uint32_t face_width = 16;
    constexpr uint32_t face_row_bytes = face_width * sizeof(uint16_t);
    constexpr uint32_t face_bytes = face_width * face_width * sizeof(uint16_t);
    auto row_offset = [](uint32_t row, uint32_t face_col) {
        return ((row / face_width) * 2 + face_col) * face_bytes + (row % face_width) * face_row_bytes;
    };
    // CB memory persists across programs, so every gathered tile is fully zeroed
    // before its row lands; compute treats these as one-row-per-tile operands.
    // Blackhole DRAM reads need 64-byte aligned source offsets; a 32-byte face
    // row at an odd row index is misaligned. Read the aligned 64-byte row pair
    // for both face columns into a scratch region of the destination tile
    // (face 2, bytes 1024..1151, otherwise zero), move the wanted row into row 0,
    // then re-zero the scratch. Issue all reads for a CB before one barrier.
    constexpr uint32_t pair_bytes = 2 * face_row_bytes;
    constexpr uint32_t scratch_bytes = 2 * face_bytes;  // face 2 start
    auto issue_row_reads = [&](const auto& acc, CircularBuffer& cb, uint32_t page, uint32_t src_row, uint32_t tile_offset) {
        for (uint32_t face_col = 0; face_col < 2; ++face_col) {
            const uint32_t src = (row_offset(src_row, face_col) / pair_bytes) * pair_bytes;
            noc.async_read(
                acc,
                cb,
                pair_bytes,
                {.page_id = page, .offset_bytes = src},
                {.offset_bytes = tile_offset + scratch_bytes + face_col * pair_bytes});
        }
    };
    auto settle_row = [&](CircularBuffer& cb, uint32_t src_row, uint32_t tile_offset) {
        auto* tile = reinterpret_cast<volatile uint64_t*>(cb.get_write_ptr() + tile_offset);
        const uint32_t pick = (src_row & 1) * (face_row_bytes / sizeof(uint64_t));
        for (uint32_t face_col = 0; face_col < 2; ++face_col) {
            const uint32_t src_w = (scratch_bytes + face_col * pair_bytes) / sizeof(uint64_t) + pick;
            const uint32_t dst_w = row_offset(0, face_col) / sizeof(uint64_t);
            tile[dst_w] = tile[src_w];
            tile[dst_w + 1] = tile[src_w + 1];
            tile[dst_w + 2] = tile[src_w + 2];
            tile[dst_w + 3] = tile[src_w + 3];
        }
        for (uint32_t i = 0; i < 2 * pair_bytes / sizeof(uint64_t); ++i) {
            tile[scratch_bytes / sizeof(uint64_t) + i] = 0;
        }
    };

    // Load the persistent state and per-head front-end constants once. Keeping
    // these resident also lets the one-tile input CBs stream t=0 then t=1
    // without the reader blocking before compute can start.
#ifdef VERIFY_BULK_STATE
    // The paired output worker bulk-reads this head's 128 contiguous state rows,
    // tilizes them as UInt16 words (bit-preserving) and writes the 16 tiles
    // straight into the reserved state CB while the operands below are gathered.
    static_assert(Kt == 4 && Vt == 4, "Bulk state requires 128x128 heads");
    const uint32_t ox = get_arg_val<uint32_t>(11);
    const uint32_t oy = get_arg_val<uint32_t>(12);
    {
        CircularBuffer cb(cb_state);
        cb.reserve_back(Kt * Vt);
        const uint32_t destination = cb.get_write_ptr();
        // Scratch word inside the reserved, unpublished state CB; the NoC write
        // completes before the output worker may overwrite it.
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination) = destination;
        noc_async_write(destination, get_noc_addr(ox, oy, get_semaphore(5)), sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(3)), 1);
    }
#else
    read_state_into_tiles();
#endif
    read_into(db_acc, cb_decay_bias, h, 1);
    read_into(ds_acc, cb_decay_scale, h, 1);
    read_into(ones_acc, cb_ones, 0, 1);

    const uint32_t head_tiles = (BH + 31) / 32;
    const uint32_t head_tile = h / 32;
    const uint32_t head_row = h % 32;
    CircularBuffer q_cb(cb_q), k_cb(cb_k), v_cb(cb_v), g_cb(cb_g), beta_cb(cb_beta);
    // Keep only element (0, col) of the settled row 0 of a one-tile operand.
    auto isolate_scalar = [&](CircularBuffer& cb, uint32_t col) {
        auto* tile = reinterpret_cast<volatile uint16_t*>(cb.get_write_ptr());
        const uint32_t face_col = col / face_width;
        const uint16_t value = tile[row_offset(0, face_col) / sizeof(uint16_t) + (col % face_width)];
        auto* tile_w = reinterpret_cast<volatile uint64_t*>(cb.get_write_ptr());
        for (uint32_t fc = 0; fc < 2; ++fc) {
            const uint32_t base = row_offset(0, fc) / sizeof(uint64_t);
            tile_w[base] = 0;
            tile_w[base + 1] = 0;
            tile_w[base + 2] = 0;
            tile_w[base + 3] = 0;
        }
        tile[0] = value;
    };
    for (uint32_t token = 0; token < num_tokens; ++token) {
        #ifdef TREE_FORK_POSITION
        if (token == 8) {
            auto* released = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
            noc_semaphore_wait(released, 1);
            noc_semaphore_set(released, 0);
            constexpr auto parent_args = TensorAccessorArgs<ones_a.next_compile_time_args_offset()>();
            const auto parent = TensorAccessor(parent_args, get_arg_val<uint32_t>(11), tile_bytes);
            read_into(parent, cb_state, h * Kt * Vt, Kt * Vt);
        }
        #endif
        // One zero fill and one DRAM round trip per token for all five operands:
        // the same bytes as five sequential read_row_into/read_scalar_into calls.
        q_cb.reserve_back(Kt);
        k_cb.reserve_back(Kt);
        v_cb.reserve_back(Vt);
        g_cb.reserve_back(1);
        beta_cb.reserve_back(1);
        noc.async_write_zeros(q_cb, Kt * tile_bytes);
        noc.async_write_zeros(k_cb, Kt * tile_bytes);
        noc.async_write_zeros(v_cb, Vt * tile_bytes);
        noc.async_write_zeros(g_cb, tile_bytes);
        noc.async_write_zeros(beta_cb, tile_bytes);
        noc.write_zeros_l1_barrier();
        const uint32_t qk_page = (token * head_tiles + head_tile) * Kt;
        const uint32_t v_page = (token * head_tiles + head_tile) * Vt;
        for (uint32_t tile = 0; tile < Kt; ++tile) {
            issue_row_reads(q_acc, q_cb, qk_page + tile, head_row, tile * tile_bytes);
            issue_row_reads(k_acc, k_cb, qk_page + tile, head_row, tile * tile_bytes);
        }
        for (uint32_t tile = 0; tile < Vt; ++tile) {
            issue_row_reads(v_acc, v_cb, v_page + tile, head_row, tile * tile_bytes);
        }
        issue_row_reads(a_acc, g_cb, head_tile, token, 0);
        issue_row_reads(b_acc, beta_cb, head_tile, token, 0);
        noc.async_read_barrier();
        for (uint32_t tile = 0; tile < Kt; ++tile) {
            settle_row(q_cb, head_row, tile * tile_bytes);
            settle_row(k_cb, head_row, tile * tile_bytes);
        }
        for (uint32_t tile = 0; tile < Vt; ++tile) {
            settle_row(v_cb, head_row, tile * tile_bytes);
        }
        settle_row(g_cb, token, 0);
        isolate_scalar(g_cb, head_row);
        settle_row(beta_cb, token, 0);
        isolate_scalar(beta_cb, head_row);
        q_cb.push_back(Kt);
        k_cb.push_back(Kt);
        v_cb.push_back(Vt);
        g_cb.push_back(1);
        beta_cb.push_back(1);
#ifdef VERIFY_BULK_STATE
        if (token == 0) {
            auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
            noc_semaphore_wait(ready, 1);
            noc_semaphore_set(ready, 0);
            CircularBuffer(cb_state).push_back(Kt * Vt);
        }
#endif
    }
}
#else
void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t FUSED_FRONTEND = get_compile_time_arg_val(2);
    constexpr auto q_a = TensorAccessorArgs<5>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto db_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto ds_a = TensorAccessorArgs<db_a.next_compile_time_args_offset()>();
    constexpr auto ones_a = TensorAccessorArgs<ds_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_arg_val<uint32_t>(4);
    const uint32_t b_addr = get_arg_val<uint32_t>(5);
    const uint32_t s_addr = get_arg_val<uint32_t>(6);
    const uint32_t db_addr = get_arg_val<uint32_t>(7);
    const uint32_t ds_addr = get_arg_val<uint32_t>(8);
    const uint32_t ones_addr = get_arg_val<uint32_t>(9);

    const uint32_t tile_bytes = get_tile_size(cb_q);
    constexpr uint32_t state_stick_bytes = Vt * 32 * sizeof(uint16_t);
    const auto q_acc = TensorAccessor(q_a, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_a, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_a, v_addr, tile_bytes);
    const auto g_acc = TensorAccessor(g_a, g_addr, tile_bytes);
    const auto b_acc = TensorAccessor(b_a, b_addr, tile_bytes);
    const auto s_acc = TensorAccessor(s_a, s_addr, state_stick_bytes);
    const auto db_acc = TensorAccessor(db_a, db_addr, tile_bytes);
    const auto ds_acc = TensorAccessor(ds_a, ds_addr, tile_bytes);
    const auto ones_acc = TensorAccessor(ones_a, ones_addr, tile_bytes);
    Noc noc;

    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t count) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(count);
        for (uint32_t t = 0; t < count; ++t) {
            noc.async_read(acc, cb, tile_bytes, {.page_id = base + t}, {.offset_bytes = t * tile_bytes});
        }
        noc.async_read_barrier();
        cb.push_back(count);
    };

    // Persistent state is compact row-major BF16, height-sharded over all
    // compute cores. Materialize one head directly into the tile-layout CB;
    // no tensor conversion, gather, or GDDR staging is involved.
    auto read_state_into_tiles = [&]() {
        constexpr uint32_t tile_width = 32;
        constexpr uint32_t face_width = 16;
        constexpr uint32_t face_row_bytes = face_width * sizeof(uint16_t);
        constexpr uint32_t face_bytes = face_width * face_width * sizeof(uint16_t);
        CircularBuffer cb(cb_state);
        cb.reserve_back(Kt * Vt);
        for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
            for (uint32_t row = 0; row < tile_width; ++row) {
                const uint32_t state_row = h * Kt * tile_width + tile_row * tile_width + row;
                const uint32_t face_row = row % face_width;
                const uint32_t face_row_group = row / face_width;
                for (uint32_t tile_col = 0; tile_col < Vt; ++tile_col) {
                    const uint32_t tile_offset = (tile_row * Vt + tile_col) * tile_bytes;
                    for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                        const uint32_t source_offset =
                            (tile_col * tile_width + face_col * face_width) * sizeof(uint16_t);
                        const uint32_t face = face_row_group * 2 + face_col;
                        const uint32_t dest_offset =
                            tile_offset + face * face_bytes + face_row * face_row_bytes;
                        noc.async_read(
                            s_acc,
                            cb,
                            face_row_bytes,
                            {.page_id = state_row, .offset_bytes = source_offset},
                            {.offset_bytes = dest_offset});
                    }
                }
            }
        }
        noc.async_read_barrier();
        cb.push_back(Kt * Vt);
    };

    read_into(q_acc, cb_q, h * Kt, Kt);
    read_into(k_acc, cb_k, h * Kt, Kt);
    read_into(v_acc, cb_v, h * Vt, Vt);
    read_into(g_acc, cb_g, h, 1);
    read_into(b_acc, cb_beta, h, 1);
    read_state_into_tiles();
    if constexpr (FUSED_FRONTEND) {
        read_into(db_acc, cb_decay_bias, h, 1);
        read_into(ds_acc, cb_decay_scale, h, 1);
        read_into(ones_acc, cb_ones, 0, 1);
    }
}
#endif
