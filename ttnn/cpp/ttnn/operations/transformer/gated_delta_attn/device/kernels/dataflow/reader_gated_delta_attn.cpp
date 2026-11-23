// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Sequential scan reader. CB0 streams L_unit, q_decay, intra_attn, k_decay_t;
// CB1 streams v_beta_sc, k_bd_sc. Each full-buffer reserve waits for the
// previous phase's compute pop, including across chunk boundaries.
//
// Runtime args: head, NC, L_unit, v_beta_sc, k_bd_sc, intra_attn, q_decay,
// k_decay_t, dl_exp, L_inv, initial_state (0 = zeros).
// Compile-time args: Ct, Kt, Vt, then the nine tensor accessors in that order.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    static_assert(Ct == 4 && Kt == 4 && Vt == 4);

    const uint32_t head_idx = get_arg_val<uint32_t>(0);
    const uint32_t NC = get_arg_val<uint32_t>(1);
    const uint32_t lu_addr = get_arg_val<uint32_t>(2);
    const uint32_t vbs_addr = get_arg_val<uint32_t>(3);
    const uint32_t kbs_addr = get_arg_val<uint32_t>(4);
    const uint32_t att_addr = get_arg_val<uint32_t>(5);
    const uint32_t qdec_addr = get_arg_val<uint32_t>(6);
    const uint32_t kdt_addr = get_arg_val<uint32_t>(7);
    const uint32_t dle_addr = get_arg_val<uint32_t>(8);
    const uint32_t linv_addr = get_arg_val<uint32_t>(9);
    const uint32_t s0_addr = get_arg_val<uint32_t>(10);

    constexpr uint32_t out_tiles = Ct * Vt;
    constexpr uint32_t in_kv_tiles = Ct * Kt;
    constexpr uint32_t attn_tiles = Ct * Ct;
    constexpr uint32_t kdt_tiles = Kt * Ct;
    constexpr uint32_t state_tiles = Kt * Vt;
    constexpr uint32_t cb_matrix = tt::CBIndex::c_0;
    constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
    constexpr uint32_t cb_dl_exp = tt::CBIndex::c_6;
    constexpr uint32_t cb_S = tt::CBIndex::c_8;
    constexpr uint32_t cb_L_inv_0 = tt::CBIndex::c_14;
    constexpr uint32_t cb_L_inv_1 = tt::CBIndex::c_15;
    constexpr uint32_t cb_L_inv_2 = tt::CBIndex::c_16;
    constexpr uint32_t cb_L_inv_3 = tt::CBIndex::c_17;
    constexpr uint32_t f32_tile = get_tile_size(cb_S);

    constexpr auto lu_args = TensorAccessorArgs<3>();
    const auto lu_gen = TensorAccessor(lu_args, lu_addr, f32_tile);
    constexpr auto vbs_args = TensorAccessorArgs<lu_args.next_compile_time_args_offset()>();
    const auto vbs_gen = TensorAccessor(vbs_args, vbs_addr, f32_tile);
    constexpr auto kbs_args = TensorAccessorArgs<vbs_args.next_compile_time_args_offset()>();
    const auto kbs_gen = TensorAccessor(kbs_args, kbs_addr, f32_tile);
    constexpr auto att_args = TensorAccessorArgs<kbs_args.next_compile_time_args_offset()>();
    const auto att_gen = TensorAccessor(att_args, att_addr, f32_tile);
    constexpr auto qdec_args = TensorAccessorArgs<att_args.next_compile_time_args_offset()>();
    const auto qdec_gen = TensorAccessor(qdec_args, qdec_addr, f32_tile);
    constexpr auto kdt_args = TensorAccessorArgs<qdec_args.next_compile_time_args_offset()>();
    const auto kdt_gen = TensorAccessor(kdt_args, kdt_addr, f32_tile);
    constexpr auto dle_args = TensorAccessorArgs<kdt_args.next_compile_time_args_offset()>();
    const auto dle_gen = TensorAccessor(dle_args, dle_addr, f32_tile);
    constexpr auto linv_args = TensorAccessorArgs<dle_args.next_compile_time_args_offset()>();
    const auto linv_gen = TensorAccessor(linv_args, linv_addr, f32_tile);
    constexpr auto s0_args = TensorAccessorArgs<linv_args.next_compile_time_args_offset()>();
    const auto s0_gen = TensorAccessor(s0_args, s0_addr, f32_tile);

    Noc noc;
    auto read_tiles = [&](const auto& accessor, uint32_t cb_id, uint32_t base, uint32_t count) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(count);
        for (uint32_t tile = 0; tile < count; ++tile) {
            noc.async_read(
                accessor, cb, f32_tile, {.page_id = base + tile}, {.offset_bytes = tile * f32_tile});
        }
        noc.async_read_barrier();
        cb.push_back(count);
    };

    // State remains in the on-core FP32 CB throughout the entire scan.
    if (s0_addr != 0) {
        read_tiles(s0_gen, cb_S, head_idx * state_tiles, state_tiles);
    } else {
        CircularBuffer state(cb_S);
        state.reserve_back(state_tiles);
        noc.async_write_zeros(state, state_tiles * f32_tile);
        noc.write_zeros_l1_barrier();
        state.push_back(state_tiles);
    }

    for (uint32_t chunk = 0; chunk < NC; ++chunk) {
        const uint32_t head_chunk = head_idx * NC + chunk;
        read_tiles(lu_gen, cb_matrix, head_chunk * attn_tiles, attn_tiles);
        read_tiles(vbs_gen, cb_rhs, head_chunk * out_tiles, out_tiles);

        // These must precede k_bd_sc: its reserve waits for PA2 to consume
        // v_beta_sc, and PA2 itself needs all four diagonal inverses.
        read_tiles(linv_gen, cb_L_inv_0, head_chunk * Ct, 1);
        read_tiles(linv_gen, cb_L_inv_1, head_chunk * Ct + 1, 1);
        read_tiles(linv_gen, cb_L_inv_2, head_chunk * Ct + 2, 1);
        read_tiles(linv_gen, cb_L_inv_3, head_chunk * Ct + 3, 1);
        read_tiles(kbs_gen, cb_rhs, head_chunk * in_kv_tiles, in_kv_tiles);

        // The next matrix cannot overwrite the current one: every phase
        // consumes all 16 tiles before the following full-buffer reserve.
        read_tiles(qdec_gen, cb_matrix, head_chunk * in_kv_tiles, in_kv_tiles);
        read_tiles(att_gen, cb_matrix, head_chunk * attn_tiles, attn_tiles);
        read_tiles(kdt_gen, cb_matrix, head_chunk * kdt_tiles, kdt_tiles);
        read_tiles(dle_gen, cb_dl_exp, head_chunk, 1);
    }
}
