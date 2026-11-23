// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16, cb_final = 17;

#ifdef VERIFY_TWO
void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr auto o_a = TensorAccessorArgs<2>();
    constexpr auto s1_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto s2_a = TensorAccessorArgs<s1_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t s1_addr = get_arg_val<uint32_t>(2);
    const uint32_t s2_addr = get_arg_val<uint32_t>(3);
    const uint32_t BH = get_arg_val<uint32_t>(4);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t state_stick_bytes = Vt * 32 * sizeof(uint16_t);
    const auto o_acc = TensorAccessor(o_a, o_addr, tile_bytes);
    // Candidate states are TILE [BH, K, V]: this head's Kt*Vt tiles are contiguous pages.
    const auto s1_acc = TensorAccessor(s1_a, s1_addr, tile_bytes);
    const auto s2_acc = TensorAccessor(s2_a, s2_addr, tile_bytes);
    Noc noc;

    // Output is [2, 1, 32, 128] TILE: token on the tile row, head on the row.
    // Row 0 of each computed tile is this head's result; scatter it to row h.
    constexpr uint32_t face_width = 16;
    constexpr uint32_t face_row_bytes = face_width * sizeof(uint16_t);
    constexpr uint32_t face_bytes = face_width * face_width * sizeof(uint16_t);
    auto row_offset = [](uint32_t row, uint32_t face_col) {
        return ((row / face_width) * 2 + face_col) * face_bytes + (row % face_width) * face_row_bytes;
    };
    auto write_output = [&](uint32_t token) {
        CircularBuffer cb(cb_out);
        cb.wait_front(Vt);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t tile = 0; tile < Vt; ++tile) {
            for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                noc.async_write(
                    src,
                    o_acc,
                    face_row_bytes,
                    {.offset_bytes = tile * tile_bytes + row_offset(0, face_col)},
                    {.page_id = token * Vt + tile, .offset_bytes = row_offset(h, face_col)});
            }
        }
        noc.async_write_barrier();
        cb.pop_front(Vt);
    };

    auto write_state = [&](const auto& acc) {
        CircularBuffer cb(cb_final);
        cb.wait_front(Kt * Vt);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        const uint32_t base = h * Kt * Vt;
        for (uint32_t tile = 0; tile < Kt * Vt; ++tile) {
            noc.async_write(src, acc, tile_bytes, {.offset_bytes = tile * tile_bytes}, {.page_id = base + tile});
        }
        noc.async_write_barrier();
        cb.pop_front(Kt * Vt);
    };
    write_output(0);
    write_state(s1_acc);
    write_output(1);
    write_state(s2_acc);
}
#else
void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr auto o_a = TensorAccessorArgs<2>();
    constexpr auto s_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t state_stick_bytes = Vt * 32 * sizeof(uint16_t);
    const auto o_acc = TensorAccessor(o_a, o_addr, tile_bytes);
    const auto s_acc = TensorAccessor(s_a, s_addr, state_stick_bytes);
    Noc noc;

    auto write_from = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t count) {
        CircularBuffer cb(cb_id);
        cb.wait_front(count);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t t = 0; t < count; ++t) {
            noc.async_write(src, acc, tile_bytes, {.offset_bytes = t * tile_bytes}, {.page_id = base + t});
        }
        noc.async_write_barrier();
        cb.pop_front(count);
    };

    auto write_state_from_tiles = [&]() {
        constexpr uint32_t tile_width = 32;
        constexpr uint32_t face_width = 16;
        constexpr uint32_t face_row_bytes = face_width * sizeof(uint16_t);
        constexpr uint32_t face_bytes = face_width * face_width * sizeof(uint16_t);
        CircularBuffer cb(cb_final);
        cb.wait_front(Kt * Vt);
        for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
            for (uint32_t row = 0; row < tile_width; ++row) {
                const uint32_t state_row = h * Kt * tile_width + tile_row * tile_width + row;
                const uint32_t face_row = row % face_width;
                const uint32_t face_row_group = row / face_width;
                for (uint32_t tile_col = 0; tile_col < Vt; ++tile_col) {
                    const uint32_t tile_offset = (tile_row * Vt + tile_col) * tile_bytes;
                    for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                        const uint32_t dest_offset =
                            (tile_col * tile_width + face_col * face_width) * sizeof(uint16_t);
                        const uint32_t face = face_row_group * 2 + face_col;
                        const uint32_t source_offset =
                            tile_offset + face * face_bytes + face_row * face_row_bytes;
                        noc.async_write(
                            cb,
                            s_acc,
                            face_row_bytes,
                            {.offset_bytes = source_offset},
                            {.page_id = state_row, .offset_bytes = dest_offset});
                    }
                }
            }
        }
        noc.async_write_barrier();
        cb.pop_front(Kt * Vt);
    };

    write_from(o_acc, cb_out, h * Vt, Vt);
    write_state_from_tiles();
}
#endif
