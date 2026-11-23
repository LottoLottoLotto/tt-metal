// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4, cb_state = 5;
constexpr uint32_t cb_decay_bias = 6, cb_decay_scale = 7, cb_ones = 26;

#ifdef VERIFY_TWO
void kernel_main() {
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

    // q/k/v arrive in the native frontend layout: [2, 1, 32, 128] TILE, one
    // tile row per token with the head on the tile row. a/b arrive as a single
    // [1, 1, 2, 32] tile with the token on the row and the head on the column.
    // Each core gathers its head row into row 0 of zeroed tiles so the compute
    // kernel sees the same one-row-per-tile operands as the T=1 path.
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
        auto* base = reinterpret_cast<volatile uint64_t*>(cb.get_write_ptr());
        const uint32_t words = count * tile_bytes / sizeof(uint64_t);
        for (uint32_t i = 0; i < words; i += 8) {
            base[i] = 0;
            base[i + 1] = 0;
            base[i + 2] = 0;
            base[i + 3] = 0;
            base[i + 4] = 0;
            base[i + 5] = 0;
            base[i + 6] = 0;
            base[i + 7] = 0;
        }
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
    // Gather row `src_row` of `count` consecutive tiles starting at `page` into row 0.
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
    // Gather element (src_row, col) of tile `page` into element (0, 0).
    auto read_scalar_into = [&](const auto& acc, uint32_t cb_id, uint32_t page, uint32_t src_row, uint32_t col) {
        zero_tiles(cb_id, 1);
        CircularBuffer cb(cb_id);
        issue_row_reads(acc, cb, page, src_row, 0);
        noc.async_read_barrier();
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
    read_state_into_tiles();
    read_into(db_acc, cb_decay_bias, h, 1);
    read_into(ds_acc, cb_decay_scale, h, 1);
    read_into(ones_acc, cb_ones, 0, 1);

    for (uint32_t token = 0; token < 2; ++token) {
        read_row_into(q_acc, cb_q, token * Kt, Kt, h);
        read_row_into(k_acc, cb_k, token * Kt, Kt, h);
        read_row_into(v_acc, cb_v, token * Vt, Vt, h);
        read_scalar_into(a_acc, cb_g, 0, token, h);
        read_scalar_into(b_acc, cb_beta, 0, token, h);
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
