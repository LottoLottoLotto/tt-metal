// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4, cb_state = 5;
constexpr uint32_t cb_decay_bias = 6, cb_decay_scale = 7, cb_ones = 26;
// Ordinary decode: row-major state staging, tilized by compute into cb_state.
constexpr uint32_t cb_state_rows = 8;

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

    // Persistent state is compact row-major BF16: stream this head's rows as whole
    // 256-byte sticks, one tile row (32 rows, Vt pages) at a time; compute tilizes them
    // in hardware (as in decode). Face-row scatter instead costs 1024 32-byte NoC reads.
    auto read_state_rows = [&]() {
        constexpr uint32_t tile_width = 32;
        CircularBuffer cb(cb_state_rows);
        for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
            cb.reserve_back(Vt);
            for (uint32_t row = 0; row < tile_width; ++row) {
                const uint32_t state_row = h * Kt * tile_width + tile_row * tile_width + row;
                noc.async_read(
                    s_acc, cb, state_stick_bytes, {.page_id = state_row}, {.offset_bytes = row * state_stick_bytes});
            }
            noc.async_read_barrier();
            cb.push_back(Vt);
        }
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
    auto zero_tiles = [&](uint32_t cb_id, uint32_t count) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(count);
        // All five operands can share the zero-write barrier; reads must follow it.
        noc.async_write_zeros(cb, count * tile_bytes);
    };
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
    // After the row reads have landed: move element (src_row, col) into element (0, 0)
    // of an otherwise zero tile and publish it.
    auto settle_scalar = [&](CircularBuffer& cb, uint32_t src_row, uint32_t col) {
        settle_row(cb, src_row, 0);
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
        cb.push_back(1);
    };

    // Load the persistent state and per-head front-end constants once. Keeping
    // these resident also lets the one-tile input CBs stream t=0 then t=1
    // without the reader blocking before compute can start.
    read_state_rows();
    read_into(db_acc, cb_decay_bias, h, 1);
    read_into(ds_acc, cb_decay_scale, h, 1);
    read_into(ones_acc, cb_ones, 0, 1);

    const uint32_t head_tiles = (BH + 31) / 32;
    const uint32_t head_tile = h / 32;
    const uint32_t head_row = h % 32;
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
        // One token's five operands: zero every destination, issue all row reads, then a
        // single barrier (per-operand barriers serialized five DRAM round trips).
        const uint32_t qk_page = (token * head_tiles + head_tile) * Kt;
        const uint32_t v_page = (token * head_tiles + head_tile) * Vt;
        CircularBuffer q_cb(cb_q), k_cb(cb_k), v_cb(cb_v), g_cb(cb_g), b_cb(cb_beta);
        zero_tiles(cb_q, Kt);
        zero_tiles(cb_k, Kt);
        zero_tiles(cb_v, Vt);
        zero_tiles(cb_g, 1);
        zero_tiles(cb_beta, 1);
        noc.write_zeros_l1_barrier();
        for (uint32_t tile = 0; tile < Kt; ++tile) {
            issue_row_reads(q_acc, q_cb, qk_page + tile, head_row, tile * tile_bytes);
            issue_row_reads(k_acc, k_cb, qk_page + tile, head_row, tile * tile_bytes);
        }
        for (uint32_t tile = 0; tile < Vt; ++tile) {
            issue_row_reads(v_acc, v_cb, v_page + tile, head_row, tile * tile_bytes);
        }
        issue_row_reads(a_acc, g_cb, head_tile, token, 0);
        issue_row_reads(b_acc, b_cb, head_tile, token, 0);
        noc.async_read_barrier();
        for (uint32_t tile = 0; tile < Kt; ++tile) {
            settle_row(q_cb, head_row, tile * tile_bytes);
            settle_row(k_cb, head_row, tile * tile_bytes);
        }
        q_cb.push_back(Kt);
        k_cb.push_back(Kt);
        for (uint32_t tile = 0; tile < Vt; ++tile) {
            settle_row(v_cb, head_row, tile * tile_bytes);
        }
        v_cb.push_back(Vt);
        settle_scalar(g_cb, token, head_row);
        settle_scalar(b_cb, token, head_row);
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

    // Persistent state is compact row-major BF16, height-sharded over all compute
    // cores. Stream this head's rows as whole 256-byte sticks, one tile row (32 rows,
    // Vt pages) at a time; compute tilizes them in hardware. Moving individual face
    // rows here instead costs 1024 32-byte NoC reads per head.
    auto read_state_rows = [&]() {
        constexpr uint32_t tile_width = 32;
        CircularBuffer cb(cb_state_rows);
        for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
            cb.reserve_back(Vt);
            for (uint32_t row = 0; row < tile_width; ++row) {
                const uint32_t state_row = h * Kt * tile_width + tile_row * tile_width + row;
                noc.async_read(
                    s_acc, cb, state_stick_bytes, {.page_id = state_row}, {.offset_bytes = row * state_stick_bytes});
            }
            noc.async_read_barrier();
            cb.push_back(Vt);
        }
    };

#ifdef GATHER_HEAD_ROWS
    // Inputs arrive as produced by the decode frontend, without head-split copies:
    // q/k/v [1, 1, HV, D] TILE (head h on row h % 32 of tile row h / 32) and raw a/b
    // [1, 1, HV] (head h on column h % 32 of tile h / 32, row 0). Gather this head
    // into row 0 / element (0, 0), exactly the layout the head-split path provides.
    {
        constexpr uint32_t face_width = 16;
        constexpr uint32_t face_row_bytes = face_width * sizeof(uint16_t);
        constexpr uint32_t face_bytes = face_width * face_width * sizeof(uint16_t);
        constexpr uint32_t pair_bytes = 2 * face_row_bytes;
        constexpr uint32_t scratch_bytes = 2 * face_bytes;  // face 2 start
        auto row_offset = [](uint32_t row, uint32_t face_col) {
            return ((row / face_width) * 2 + face_col) * face_bytes + (row % face_width) * face_row_bytes;
        };
        auto zero_tiles = [&](uint32_t cb_id, uint32_t count) {
            CircularBuffer cb(cb_id);
            cb.reserve_back(count);
            noc.async_write_zeros(cb, count * tile_bytes);
            noc.write_zeros_l1_barrier();
        };
        // DRAM reads need 64-byte aligned sources: read the aligned row pair of each
        // face column into face 2 of the destination tile, then move the row to row 0.
        auto issue_row_reads = [&](const auto& acc, CircularBuffer& cb, uint32_t page, uint32_t src_row,
                                   uint32_t tile_offset) {
            for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                const uint32_t src = (row_offset(src_row, face_col) / pair_bytes) * pair_bytes;
                noc.async_read(acc, cb, pair_bytes, {.page_id = page, .offset_bytes = src},
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
        auto read_row_into = [&](const auto& acc, uint32_t cb_id, uint32_t page, uint32_t count, uint32_t src_row) {
            zero_tiles(cb_id, count);
            CircularBuffer cb(cb_id);
            for (uint32_t tile = 0; tile < count; ++tile) {
                issue_row_reads(acc, cb, page + tile, src_row, tile * tile_bytes);
            }
            noc.async_read_barrier();
            for (uint32_t tile = 0; tile < count; ++tile) {
                settle_row(cb, src_row, tile * tile_bytes);
            }
            cb.push_back(count);
        };
        auto read_scalar_into = [&](const auto& acc, uint32_t cb_id, uint32_t page, uint32_t col) {
            zero_tiles(cb_id, 1);
            CircularBuffer cb(cb_id);
            issue_row_reads(acc, cb, page, 0, 0);
            noc.async_read_barrier();
            settle_row(cb, 0, 0);
            auto* tile = reinterpret_cast<volatile uint16_t*>(cb.get_write_ptr());
            const uint16_t value = tile[row_offset(0, col / face_width) / sizeof(uint16_t) + (col % face_width)];
            auto* tile_w = reinterpret_cast<volatile uint64_t*>(cb.get_write_ptr());
            for (uint32_t fc = 0; fc < 2; ++fc) {
                const uint32_t base = row_offset(0, fc) / sizeof(uint64_t);
                tile_w[base] = 0;
                tile_w[base + 1] = 0;
                tile_w[base + 2] = 0;
                tile_w[base + 3] = 0;
            }
            tile[0] = value;
            cb.push_back(1);
        };
        const uint32_t head_tile = h / 32;
        const uint32_t head_row = h % 32;
        read_row_into(q_acc, cb_q, head_tile * Kt, Kt, head_row);
        read_row_into(k_acc, cb_k, head_tile * Kt, Kt, head_row);
        read_row_into(v_acc, cb_v, head_tile * Vt, Vt, head_row);
        read_scalar_into(g_acc, cb_g, head_tile, head_row);
        read_scalar_into(b_acc, cb_beta, head_tile, head_row);
    }
#else
    read_into(q_acc, cb_q, h * Kt, Kt);
    read_into(k_acc, cb_k, h * Kt, Kt);
    read_into(v_acc, cb_v, h * Vt, Vt);
    read_into(g_acc, cb_g, h, 1);
    read_into(b_acc, cb_beta, h, 1);
#endif
    read_state_rows();
    if constexpr (FUSED_FRONTEND) {
        read_into(db_acc, cb_decay_bias, h, 1);
        read_into(ds_acc, cb_decay_scale, h, 1);
        read_into(ones_acc, cb_ones, 0, 1);
    }
}
#endif
