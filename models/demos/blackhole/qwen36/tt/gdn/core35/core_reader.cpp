// SPDX-License-Identifier: Apache-2.0
// Fused GDN core, recurrence worker R(h, j) reader (NCRISC): value head h, V columns [64j, 64j + 64).
// Feeds the image recurrence compute kernel (Vt = 2 local V tiles) the exact operand tiles of the unfused path:
//   state:   this head's 128 rows, V-half j, row-major (128 B per row) for the hardware tilize;
//   q/k/v:   SiLU(conv) rows received in the L1 inbox from the frontend workers, as row 0 of zeroed tiles;
//   g/beta:  raw a/b of head h for every token, read in place from the mega projection row, as element (0, 0);
//   decay bias/scale (tile h) and the all-ones reduction tile, resident.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_decay_bias = 6, cb_decay_scale = 7, cb_state_rows = 8, cb_ones = 26;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);  // local V tiles
    constexpr uint32_t num_tokens = get_compile_time_arg_val(2);
    constexpr uint32_t a_tile = get_compile_time_arg_val(3);  // mega tile of the raw a segment; b is the next tile
    constexpr uint32_t inbox_token_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t producers = get_compile_time_arg_val(5);
    constexpr auto mega_a = TensorAccessorArgs<6>();
    constexpr auto s_a = TensorAccessorArgs<mega_a.next_compile_time_args_offset()>();
    constexpr auto db_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto ds_a = TensorAccessorArgs<db_a.next_compile_time_args_offset()>();
    constexpr auto ones_a = TensorAccessorArgs<ds_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t j = get_arg_val<uint32_t>(1);
    const uint32_t mega_addr = get_arg_val<uint32_t>(2);
    const uint32_t s_addr = get_arg_val<uint32_t>(3);
    const uint32_t db_addr = get_arg_val<uint32_t>(4);
    const uint32_t ds_addr = get_arg_val<uint32_t>(5);
    const uint32_t ones_addr = get_arg_val<uint32_t>(6);
    const uint32_t inbox = get_arg_val<uint32_t>(7);

    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t stick_bytes = 128 * sizeof(uint16_t);    // one full state row (V = 128)
    constexpr uint32_t slice_bytes = Vt * 32 * sizeof(uint16_t);  // this worker's V columns of a row
    const auto mega = TensorAccessor(mega_a, mega_addr, tile_bytes);
    const auto s_acc = TensorAccessor(s_a, s_addr, stick_bytes);
    const auto db_acc = TensorAccessor(db_a, db_addr, tile_bytes);
    const auto ds_acc = TensorAccessor(ds_a, ds_addr, tile_bytes);
    const auto ones_acc = TensorAccessor(ones_a, ones_addr, tile_bytes);
    Noc noc;

    // ---- independent of the frontend: state slice and per-head constants
    {
        DeviceZoneScopedN("R-state");
        // The staging CB holds all Kt tile rows, so these pushes never wait on compute (the T=1 compute
        // takes q/k/v before it tilizes the state).
        CircularBuffer cb(cb_state_rows);
        for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
            cb.reserve_back(Vt);
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t state_row = h * Kt * 32 + tile_row * 32 + row;
                noc.async_read(s_acc, cb, slice_bytes, {.page_id = state_row, .offset_bytes = j * slice_bytes},
                               {.offset_bytes = row * slice_bytes});
            }
            noc.async_read_barrier();
            cb.push_back(Vt);
        }
    }
    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t page) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        noc.async_read(acc, cb, tile_bytes, {.page_id = page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(1);
    };
    read_into(db_acc, cb_decay_bias, h);
    read_into(ds_acc, cb_decay_scale, h);
    read_into(ones_acc, cb_ones, 0);

    // ---- per-token operands
    constexpr uint32_t face_width = 16;
    constexpr uint32_t face_row_bytes = face_width * sizeof(uint16_t);
    constexpr uint32_t face_bytes = face_width * face_width * sizeof(uint16_t);
    constexpr uint32_t pair_bytes = 2 * face_row_bytes;
    constexpr uint32_t scratch_bytes = 2 * face_bytes;  // face 2 start: scratch inside a zeroed scalar tile
    auto row_offset = [](uint32_t row, uint32_t face_col) {
        return ((row / face_width) * 2 + face_col) * face_bytes + (row % face_width) * face_row_bytes;
    };
    // Local DMA copy (no RISC load/store loop): the inbox rows land in row 0 of the zeroed operand tiles.
    auto copy_words = [](uint32_t src, uint32_t dst, uint32_t bytes) { noc_async_write(src, get_noc_addr(dst), bytes); };

    auto* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    for (uint32_t token = 0; token < num_tokens; ++token) {
        CircularBuffer q_cb(cb_q), k_cb(cb_k), v_cb(cb_v), g_cb(cb_g), b_cb(cb_beta);
        q_cb.reserve_back(Kt);
        k_cb.reserve_back(Kt);
        v_cb.reserve_back(Vt);
        g_cb.reserve_back(1);
        b_cb.reserve_back(1);
        noc.async_write_zeros(q_cb, Kt * tile_bytes);
        noc.async_write_zeros(k_cb, Kt * tile_bytes);
        noc.async_write_zeros(v_cb, Vt * tile_bytes);
        noc.async_write_zeros(g_cb, tile_bytes);
        noc.async_write_zeros(b_cb, tile_bytes);
        noc.write_zeros_l1_barrier();
        // Raw a/b of head h for this token: element (token, h % 32) of mega tiles a_tile / a_tile + 1.
        for (uint32_t s = 0; s < 2; ++s) {
            CircularBuffer& cb = s ? b_cb : g_cb;
            const uint32_t src = (row_offset(token, (h % 32) / face_width) / pair_bytes) * pair_bytes;
            noc.async_read(mega, cb, pair_bytes, {.page_id = a_tile + s, .offset_bytes = src},
                           {.offset_bytes = scratch_bytes});
        }
        if (token == 0) {
            DeviceZoneScopedN("R-wait");
            noc_semaphore_wait(arrived, producers);
            noc_semaphore_set(arrived, 0);
        }
        // q (Kt rows), k (Kt rows), v (Vt rows): 64 B per tile = row 0 of face 0 and face 1.
        const uint32_t base = inbox + token * inbox_token_bytes;
        for (uint32_t t = 0; t < Kt; ++t) {
            copy_words(base + t * 64, q_cb.get_write_ptr() + t * tile_bytes, face_row_bytes);
            copy_words(base + t * 64 + 32, q_cb.get_write_ptr() + t * tile_bytes + face_bytes, face_row_bytes);
            copy_words(base + (Kt + t) * 64, k_cb.get_write_ptr() + t * tile_bytes, face_row_bytes);
            copy_words(base + (Kt + t) * 64 + 32, k_cb.get_write_ptr() + t * tile_bytes + face_bytes, face_row_bytes);
        }
        for (uint32_t t = 0; t < Vt; ++t) {
            copy_words(base + (2 * Kt + t) * 64, v_cb.get_write_ptr() + t * tile_bytes, face_row_bytes);
            copy_words(base + (2 * Kt + t) * 64 + 32, v_cb.get_write_ptr() + t * tile_bytes + face_bytes,
                       face_row_bytes);
        }
        noc.async_read_barrier();
        noc_async_write_barrier();
        for (uint32_t s = 0; s < 2; ++s) {
            CircularBuffer& cb = s ? b_cb : g_cb;
            auto* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr());
            const uint32_t col = h % 32;
            const uint32_t in_pair = (row_offset(token, col / face_width) % pair_bytes) / sizeof(uint16_t);
            const uint16_t value = tile[scratch_bytes / sizeof(uint16_t) + in_pair + (col % face_width)];
            auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr() + scratch_bytes);
            for (uint32_t i = 0; i < pair_bytes / 4; ++i) {
                words[i] = 0;
            }
            tile[0] = value;
        }
        q_cb.push_back(Kt);
        k_cb.push_back(Kt);
        v_cb.push_back(Vt);
        g_cb.push_back(1);
        b_cb.push_back(1);
    }
}
