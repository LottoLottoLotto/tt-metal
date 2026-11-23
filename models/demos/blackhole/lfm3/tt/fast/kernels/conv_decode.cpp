// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 gated short conv, decode tail (after in_proj), one launch (conv_decode.py, ConvDecode):
//
//   y = C * conv7(s0..s5, B * x),  s_j <- s_{j+1}, s5 <- B * x      (arithmetic: conv_tail.hpp, bitwise stock)
//
// One core owns PER_CORE hidden column tiles j (B = bcx tile j, C = 64 + j, x = 128 + j; taps tensor
// [t0 | t1 | ... | t6] with each tap row replicated over the tile rows).  Only tile rows [0, ROWS) move (64 B row
// granules; KEEP: the extra state row of an odd ROWS belongs to another slot and is written back unchanged).  A
// column's state tiles are read (and consumed by TRISC) before its writer shifts the state in place.
//
// Reader (NCRISC) args: common [bcx, taps, s0..s5 (, row map: VERIFY)], per-core [first column].
// Writer (BRISC)  args: common [y, s0..s5 (, cand: VERIFY)], per-core [first column].
// Defines: ROWS, PER_CORE, KEEP (0/1), BCX_DRAM / Y_DRAM (0: L1-interleaved), VERIFY (0/1).
//
// VERIFY (speculative verification, ConvDecode(verify=cand, vmap=...)): the rows are groups of consecutive tokens,
// one per sequence (conv_tail::verify_context: row map, slot per group).  The reader stages the state rows [0, ROWS)
// (every slot of the round) in kRaw; TRISC computes every row's Bx first; the reader then builds each row's left
// context from its slot's state and its group's earlier rows before the state partial, so every row's y is bitwise
// the sequential 1-row steps'.  The state is NOT updated: the writer stores every row's Bx in the candidate tensor
// (tt/speculate.py SpecCommit commits each group's accepted rows).
#include <cstdint>
#ifndef VERIFY
#define VERIFY 0
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#endif
#include "conv_tail.hpp"

namespace conv_decode {
constexpr uint32_t kHiddenTiles = 64;  // 2048 / 32
constexpr uint32_t per = PER_CORE;
// Reader -> compute (all UnpackToDestFp32)
constexpr uint32_t kB = 0;
constexpr uint32_t kX = 1;
constexpr uint32_t kC = 2;
constexpr uint32_t kT = 3;   // t0..t5 (per * 6)
constexpr uint32_t kT6 = 4;
constexpr uint32_t kS = 5;   // s0..s5 (per * 6)
constexpr uint32_t kAcc = 6; // FP32 state partial
constexpr uint32_t kRaw = 7; // VERIFY: reader scratch, the state rows as read (per * 6 tiles) + the row map (1 tile)
// Compute -> writer
constexpr uint32_t kY = 16;
constexpr uint32_t kBx = 17;
}  // namespace conv_decode

void kernel_main() {
    using namespace conv_decode;
    using conv_tail::kState;
    using conv_tail::kTile;
#if defined(COMPILE_FOR_NCRISC)
    using RowsT = conv_tail::Rows<ROWS>;
    const auto bcx = TensorAccessor(tensor_accessor::make_interleaved_dspec<BCX_DRAM != 0>(),
                                    get_common_arg_val<uint32_t>(0), kTile);
    const auto taps = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1),
                                     kTile);
    const uint32_t col0 = get_arg_val<uint32_t>(0);
    // 1. state rows + taps t0..t5 (TRISC starts the state partial sum on them; VERIFY: the state rows go to kRaw and
    //    the row map after them, the contexts are built once the Bx rows are in)
    cb_reserve_back(kS, per * kState);
    cb_reserve_back(kT, per * kState);
    const uint32_t s_dst = VERIFY ? get_write_ptr(kRaw) : get_write_ptr(kS);
    for (uint32_t j = 0; j < kState; ++j) {
        const auto s = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                      get_common_arg_val<uint32_t>(2 + j), kTile);
        for (uint32_t i = 0; i < per; ++i) {
            RowsT::read(get_noc_addr(col0 + i, s), s_dst + (i * kState + j) * kTile);
            RowsT::read(get_noc_addr(j * kHiddenTiles + col0 + i, taps), get_write_ptr(kT) + (i * kState + j) * kTile);
        }
    }
#if VERIFY
    const uint32_t map_l1 = get_write_ptr(kRaw) + per * kState * kTile;
    noc_async_read(TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(8), 64)
                       .get_noc_addr(0),
                   map_l1, 64);
#endif
    noc_async_read_barrier();
#if !VERIFY
    cb_push_back(kS, per * kState);
#endif
    cb_push_back(kT, per * kState);
    // 2. this step's B / C / x and t6
    cb_reserve_back(kB, per);
    cb_reserve_back(kX, per);
    cb_reserve_back(kC, per);
    cb_reserve_back(kT6, per);
    for (uint32_t i = 0; i < per; ++i) {
        const uint32_t j = col0 + i;
        RowsT::read(get_noc_addr(j, bcx), get_write_ptr(kB) + i * kTile);
        RowsT::read(get_noc_addr(kHiddenTiles + j, bcx), get_write_ptr(kC) + i * kTile);
        RowsT::read(get_noc_addr(2 * kHiddenTiles + j, bcx), get_write_ptr(kX) + i * kTile);
        RowsT::read(get_noc_addr(kState * kHiddenTiles + j, taps), get_write_ptr(kT6) + i * kTile);
    }
    noc_async_read_barrier();
    cb_push_back(kB, per);
    cb_push_back(kX, per);
    cb_push_back(kC, per);
    cb_push_back(kT6, per);
#if VERIFY
    // 3. every row's left context from its slot's state rows (kRaw) and TRISC's Bx rows (kBx's read pointer on this
    //    RISC is the CB base: the writer pops it later), then the state tiles go to TRISC
    cb_wait_front(kBx, per);
    conv_tail::verify_context<ROWS>(get_write_ptr(kS), get_write_ptr(kRaw), get_read_ptr(kBx), per,
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(map_l1));
    cb_push_back(kS, per * kState);
#endif
#elif defined(COMPILE_FOR_BRISC)
    using RowsT = conv_tail::Rows<ROWS>;
    const auto y = TensorAccessor(tensor_accessor::make_interleaved_dspec<Y_DRAM != 0>(),
                                  get_common_arg_val<uint32_t>(0), kTile);
    const uint32_t col0 = get_arg_val<uint32_t>(0);
    cb_wait_front(kY, per);
    for (uint32_t i = 0; i < per; ++i) {
        RowsT::write(get_read_ptr(kY) + i * kTile, get_noc_addr(col0 + i, y));
    }
    cb_wait_front(kBx, per);
#if VERIFY
    // every row's Bx -> the candidate tensor (the state is committed later for the accepted rows)
    const auto cand = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(7),
                                     kTile);
    for (uint32_t i = 0; i < per; ++i) {
        RowsT::write(get_read_ptr(kBx) + i * kTile, get_noc_addr(col0 + i, cand));
    }
#else
    using SAcc = decltype(TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), 0u, kTile));
    SAcc s[kState] = {
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), kTile),
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), kTile),
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3), kTile),
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4), kTile),
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(5), kTile),
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6), kTile)};
    // kBx in => TRISC has consumed the state tiles (state partial first): shift in place from the read copies.
    // kS was filled once by the reader from the CB base, which is this RISC's (never advanced) read pointer.
    conv_tail::shift_state<ROWS, KEEP != 0>(s, get_read_ptr(kS), get_read_ptr(kBx), col0, per);
#endif
    noc_async_write_barrier();
    cb_pop_front(kY, per);
    cb_pop_front(kBx, per);
#elif defined(COMPILE_FOR_TRISC)
    constexpr VectorMode kVec = ROWS <= 16 ? VectorMode::R : VectorMode::RC;
    unary_op_init_common(kS, kY);
#if VERIFY
    conv_tail::bx_only<per, kVec>(kB, kX, kBx);
    conv_tail::state_partial<per, kVec>(kS, kT, kAcc);
    conv_tail::finish<per, kVec, false>(kB, kX, kC, kT6, kAcc, kY, kBx);
#else
    conv_tail::state_partial<per, kVec>(kS, kT, kAcc);
    conv_tail::finish<per, kVec>(kB, kX, kC, kT6, kAcc, kY, kBx);
#endif
#endif
}
