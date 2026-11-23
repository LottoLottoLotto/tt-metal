// SPDX-License-Identifier: Apache-2.0
// Fused residual add + RMSNorm, data-movement RISC 1 (reader + cross-core sum-of-squares gather).
//
// Every core owns `tiles` consecutive width tiles [tile0, tile0 + tiles) of the [T, W] row block.
//   1. read a, b (residual operands), the gamma tiles and (RN_POST) the post-norm gamma tiles (master: also build
//      the 1/W reduce-scaler tile and zero the CB_M entries no core writes).
//   2. once compute has packed this core's partial (CB_PART, FP32, diagonal [r][r] = sum over the core's columns
//      of x^2 for row r), write rows [0, rows) into column k of the master's CB_M tile(s) (inline 32-bit NOC
//      writes) and bump the master's gather semaphore.
//   3. master: wait for all cores, hand CB_M to compute (which produces the FP32 1/rms column in CB_RSTD; the
//      writer multicasts it). Others: wait for the rstd semaphore, then expose CB_RSTD to compute.
// RN_POST: steps 2-3 run twice (round 1: x = b, round 2: x = h); the gather semaphore counts to cores per round,
// the rstd semaphore is set to the round number. Same CB addresses each round (one-block CBs wrap).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "residual_norm_prof.hpp"

constexpr uint32_t CB_A = 0, CB_B = 1, CB_G = 2, CB_SCALER = 3, CB_PG = 4, CB_H = 16, CB_PART = 25, CB_M = 26,
                   CB_RSTD = 28;
constexpr uint32_t tiles = get_compile_time_arg_val(0);
constexpr uint32_t n_cores = get_compile_time_arg_val(1);
constexpr uint32_t rows = get_compile_time_arg_val(2);
constexpr uint32_t sem_gather_id = get_compile_time_arg_val(3);
constexpr uint32_t sem_rstd_id = get_compile_time_arg_val(4);
constexpr uint32_t grid_x = get_compile_time_arg_val(5);
constexpr uint32_t master_x = get_compile_time_arg_val(6);
constexpr uint32_t master_y = get_compile_time_arg_val(7);
constexpr uint32_t scaler_bits = get_compile_time_arg_val(8);  // bf16 1/W, packed twice
constexpr uint32_t m_tiles = (n_cores + 31) / 32;

// Local DMA zero fill from the hardware zero page (MEM_ZEROS_SIZE bytes); caller issues the read barrier.
FORCE_INLINE void zero_fill(uint32_t addr, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes) {
        const uint32_t n = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(zeros, addr, n);
        addr += n;
        bytes -= n;
    }
}

// Partial row sums (diagonal of CB_PART) -> master's CB_M column k; master: all ROUND * n_cores partials in ->
// CB_M to compute; others: rstd of this round arrived -> CB_RSTD to compute.
template <uint32_t ROUND>
FORCE_INLINE void exchange(uint32_t k, bool is_master, uint32_t m_base) {
    cb_wait_front(CB_PART, 1);
    RN_STAMP();  // partial packed
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* part = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_PART));
    const uint32_t mt = k >> 5, kc = k & 31;
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t src = (r >> 4) * 3 * 256 + (r & 15) * 17;
        const uint32_t dst = mt * 1024 + ((r >> 4) * 2 + (kc >> 4)) * 256 + (r & 15) * 16 + (kc & 15);
        noc_inline_dw_write<InlineWriteDst::L1>(get_noc_addr(master_x, master_y, m_base + dst * 4), part[src]);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(master_x, master_y, get_semaphore(sem_gather_id)), 1);
    cb_pop_front(CB_PART, 1);
    RN_STAMP();  // partial delivered

    if (is_master) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather_id));
        noc_semaphore_wait(sem, ROUND * n_cores);
        if constexpr (ROUND > 1) {
            cb_reserve_back(CB_M, m_tiles);  // compute popped round 1's block (zero entries stay in place)
        }
        cb_push_back(CB_M, m_tiles);
    } else {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rstd_id));
        noc_semaphore_wait(sem, ROUND);
        cb_reserve_back(CB_RSTD, 1);
        cb_push_back(CB_RSTD, 1);
    }
    RN_STAMP();  // master: all partials in / others: rstd arrived
}

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<9>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto pg_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    // Only common (multicast) runtime args: the buffer addresses. The core index comes from the logical coords.
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t k = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t tile0 = k * tiles;
    const bool is_master = k == 0;
    RN_PROF_INIT(0);  // 0: kernel start

    constexpr uint32_t tb = 2048;  // BF16 tile bytes
    const auto a = TensorAccessor(a_args, a_addr, tb);
    const auto g = TensorAccessor(g_args, g_addr, tb);
#ifdef RN_NO_RESIDUAL
    // Norm only: x goes straight to CB_H (the normalized input), no add.
    constexpr uint32_t cb_x = CB_H;
#else
    constexpr uint32_t cb_x = CB_A;
    const auto b = TensorAccessor(b_args, b_addr, tb);
    cb_reserve_back(CB_B, tiles);
    const uint32_t wb = get_write_ptr(CB_B);
#endif
    cb_reserve_back(cb_x, tiles);
    cb_reserve_back(CB_G, tiles);
    const uint32_t wa = get_write_ptr(cb_x);
    const uint32_t wg = get_write_ptr(CB_G);
    for (uint32_t i = 0; i < tiles; ++i) {
        noc_async_read_tile(tile0 + i, a, wa + i * tb);
#ifndef RN_NO_RESIDUAL
        noc_async_read_tile(tile0 + i, b, wb + i * tb);
#endif
    }
    for (uint32_t i = 0; i < tiles; ++i) {
        noc_async_read_tile(tile0 + i, g, wg + i * tb);
    }
#ifdef RN_POST
    const auto pg = TensorAccessor(pg_args, get_common_arg_val<uint32_t>(3), tb);
    cb_reserve_back(CB_PG, tiles);
    const uint32_t wp = get_write_ptr(CB_PG);
    for (uint32_t i = 0; i < tiles; ++i) {
        noc_async_read_tile(tile0 + i, pg, wp + i * tb);
    }
#endif
    const uint32_t m_base = get_write_ptr(CB_M);  // identical L1 address on every core of the program
    uint32_t ws = 0;
    if (is_master) {
        // 1/W reduce-scaler tile (row 0 of every face, rest 0) and the CB_M entries no core writes (rows >= rows of
        // each face; faces whose columns are all >= n_cores). Remote writes only target rows < rows and columns
        // < n_cores, so the zero fill cannot race with them. Local DMA from the zero page, overlapping the reads.
        cb_reserve_back(CB_SCALER, 1);
        ws = get_write_ptr(CB_SCALER);
        zero_fill(ws, 2048);
        cb_reserve_back(CB_M, m_tiles);
        for (uint32_t j = 0; j < m_tiles; ++j) {
            for (uint32_t f = 0; f < 4; ++f) {
                const uint32_t face = m_base + j * 4096 + f * 1024;
                const uint32_t r0 = (f >> 1) * 16, c0 = j * 32 + (f & 1) * 16;
                const uint32_t first = c0 >= n_cores ? 0 : (rows > r0 ? (rows - r0 < 16 ? rows - r0 : 16) : 0);
                if (first < 16) {
                    zero_fill(face + first * 64, (16 - first) * 64);
                }
            }
        }
    }
    noc_async_read_barrier();
#ifndef RN_NO_RESIDUAL
    cb_push_back(CB_B, tiles);
#endif
#ifdef RN_POST
    cb_push_back(CB_PG, tiles);
#endif
    cb_push_back(cb_x, tiles);
    cb_push_back(CB_G, tiles);
    if (is_master) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws);
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t c = 0; c < 8; ++c) {
                p[f * 128 + c] = scaler_bits;
            }
        }
        cb_push_back(CB_SCALER, 1);
    }
    RN_STAMP();  // 1: inputs pushed

#ifdef RN_POST
    exchange<1>(k, is_master, m_base);  // 1/rms of b
    exchange<2>(k, is_master, m_base);  // 1/rms of h
#else
    exchange<1>(k, is_master, m_base);
#endif
    noc_async_atomic_barrier();
    RN_PROF_FLUSH(pg_args.next_compile_time_args_offset(), 4, 2 * k);
}
