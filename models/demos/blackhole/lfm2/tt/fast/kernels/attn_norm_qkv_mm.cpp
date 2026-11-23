// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 attention decode: residual add + RMSNorm + qkv matmul + front end in ONE program (attn_fused.AttnNormQKVFront).
// Copy of decode_matmul_norm.cpp (NormStreamDecodeMatmul: norm_prologue.hpp + the stream matmul, bitwise) whose
// BRISC, instead of writing qkv to DRAM, sends every output column's token rows into the input tiles (CB 30, same
// L1 address on every core of the program) of the AttnDecode front workers (attn_front_fused.cpp) and increments
// their semaphore 0, as attn_qkv_mm.cpp does.
// BRISC common args: [weight, 0, 0, norm a, b, gamma, h, n, then per front worker (token-major) noc x, noc y].
#include <cstdint>
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#endif
#include "norm_prologue.hpp"

#if !defined(COMPILE_FOR_NCRISC) && !defined(COMPILE_FOR_BRISC)
namespace ckernel {
// matmul_block_init / matmul_block (api/compute/matmul.h) with an explicit math fidelity F instead of the kernel's
// MATH_FIDELITY (the prologue's norm needs HiFi4; the matmul keeps the stock decode fidelity).
template <int F>
ALWI void mm_block_init_f(uint32_t in0, uint32_t in1, uint32_t transpose, uint32_t ct, uint32_t rt, uint32_t kt_) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, transpose, ct, rt, kt_)));
    MATH((llk_math_matmul_init<static_cast<MathFidelity>(F), MM_THROTTLE>(in0, in1, transpose, ct, rt)));
    MATH((throttled_mop_status = 0));
}

template <int F>
ALWI void mm_block_f(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst, uint32_t transpose,
                     uint32_t ct, uint32_t rt, uint32_t kt_) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, ct, rt, kt_)));
#ifdef TRISC_MATH
    // matmul_block_math_dynamic_throttle (Blackhole) with fidelity F
    volatile uint32_t en = *(throttle_ptr) % 2;
    if (en) {
        if (throttled_mop_status != 1) {
            llk_math_matmul_init<static_cast<MathFidelity>(F), MM_THROTTLE_MAX>(in0, in1, transpose, ct, rt);
            throttled_mop_status = 1;
        }
        llk_math_matmul<static_cast<MathFidelity>(F), MM_THROTTLE_MAX>(idst, ct, rt);
    } else {
        if (throttled_mop_status != 0) {
            llk_math_matmul_init<static_cast<MathFidelity>(F), MM_THROTTLE>(in0, in1, transpose, ct, rt);
            throttled_mop_status = 0;
        }
        llk_math_matmul<static_cast<MathFidelity>(F), MM_THROTTLE>(idst, ct, rt);
    }
#endif
}
}  // namespace ckernel
#endif

namespace {
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");           // K tiles
constexpr uint32_t wmax = get_named_compile_time_arg_val("w");          // subblock width (tiles)
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");           // K tiles per streamed block
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");       // valid activation rows
constexpr uint32_t cb_in0 = 0;  // = np_cb_x
constexpr uint32_t cb_in1 = 1;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_front_in = 30;  // front workers' input tiles (uniform address: allocated first, on every core)
constexpr uint32_t kFrontArgs = 8;    // BRISC common-arg index of the front workers' NOC coordinates
constexpr uint32_t blk_tiles = kb * wmax;  // CB accounting unit: one ring slot, even for a narrower last subblock
static_assert(kt % kb == 0);
constexpr uint32_t ks = get_named_compile_time_arg_val("ks");  // 0: whole K in DEST; else stock chunked order
constexpr uint32_t cb_part = 24;                               // FP32 partials (ks > 0)
static_assert(ks == 0 || (kb % ks == 0 && kt % ks == 0));
constexpr bool glu = get_named_compile_time_arg_val("glu") == 1;
static_assert(!glu || (wmax % 2 == 0 && ks == 0));
}  // namespace

void kernel_main() {
    // per-core: bank, run offset (bytes, bank-relative), ncols (weight columns), first output column
    const uint32_t ncols = get_arg_val<uint32_t>(2);
    const uint32_t nsub = (ncols + wmax - 1) / wmax;
    const uint32_t rem = ncols - (nsub - 1) * wmax;
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t tile_bytes = get_named_compile_time_arg_val("in1_tile_bytes");
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t outstanding = get_named_compile_time_arg_val("outstanding");
    static_assert(outstanding >= 1 && outstanding < slots && outstanding <= 8);
    constexpr uint32_t slot_bytes = blk_tiles * tile_bytes;
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t run = get_arg_val<uint32_t>(1);
    const uint64_t src_base = get_noc_addr_from_bank_id<true>(bank, w_addr + run);
    const uint32_t ring_base = get_write_ptr(cb_in1);
    const uint32_t ring_end = ring_base + slots * slot_bytes;
    uint32_t src_off = 0;
    uint32_t pending = 0, issue_trid = 1, wait_trid = 1;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t bytes = kb * (s == nsub - 1 ? rem : wmax) * tile_bytes;
        for (uint32_t b = 0; b < kt / kb; ++b) {
            cb_reserve_back(cb_in1, blk_tiles * (pending + 1));
            uint32_t dst = get_write_ptr(cb_in1) + pending * slot_bytes;
            if (dst >= ring_end) {
                dst -= slots * slot_bytes;
            }
            noc_async_read_set_trid(issue_trid);
            noc_async_read(src_base + src_off, dst, bytes);
            src_off += bytes;
            issue_trid = issue_trid == outstanding ? 1 : issue_trid + 1;
            if (++pending == outstanding) {
                // A per-trid barrier is only safe on a block issued BEFORE the latest one: the outstanding-by-ID
                // counter can still read 0 while the command buffer is issuing the latest multi-packet read
                // (outstanding == 1 corrupted block tails). The latest block uses the full read barrier.
                if constexpr (outstanding == 1) {
                    noc_async_read_barrier();
                } else {
                    noc_async_read_barrier_with_trid(wait_trid);
                }
                cb_push_back(cb_in1, blk_tiles);
                wait_trid = wait_trid == outstanding ? 1 : wait_trid + 1;
                --pending;
            }
        }
    }
    noc_async_read_barrier();
    for (; pending > 0; --pending) {
        cb_push_back(cb_in1, blk_tiles);
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t nq = get_named_compile_time_arg_val("nq");  // q front workers per token
    constexpr uint32_t per_token = nq + 2;
    constexpr uint32_t qh = 32 / nq;
    constexpr uint32_t face_bytes = 512, row_bytes = 32, tile_bytes_bf16 = 2048;
    const uint32_t first_col = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = get_semaphore(0);
    const uint32_t front_in = get_write_ptr(cb_front_in);

    // norm prologue: n of the full row block lands in cb_in0 of every core (pushed, kt tiles)
    np_dm<0, 3>();

    auto slot_of = [](uint32_t c) -> uint32_t {
        const uint32_t h = c / 2;
        return h < 32 ? h / qh : (h < 40 ? nq : nq + 1);
    };
    auto row_of = [](uint32_t c) -> uint32_t {
        const uint32_t h = c / 2;
        return h < 32 ? h % qh : (h < 40 ? h - 32 : h - 40);
    };
    // every output column's token rows -> the front worker's input tiles (row = head - h0, tile = half)
    uint32_t col = first_col;
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        cb_wait_front(cb_out, wmax);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < wid; ++j) {
            const uint32_t c = col + j;
            const uint32_t slot = slot_of(c), i = row_of(c), half = c & 1;
            const uint32_t tile = src + j * tile_bytes_bf16;
            for (uint32_t t = 0; t < rows; ++t) {
                const uint32_t f_idx = per_token * t + slot;
                const uint32_t nx = get_common_arg_val<uint32_t>(kFrontArgs + 2 * f_idx);
                const uint32_t ny = get_common_arg_val<uint32_t>(kFrontArgs + 1 + 2 * f_idx);
                const uint32_t src_row = tile + ((t >> 4) * 2) * face_bytes + (t & 15) * row_bytes;
                const uint32_t dst_row = front_in + half * tile_bytes_bf16 + ((i >> 4) * 2) * face_bytes +
                                         (i & 15) * row_bytes;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write(src_row + f * face_bytes, get_noc_addr(nx, ny, dst_row + f * face_bytes),
                                    row_bytes);
                }
            }
        }
        col += wid;
        noc_async_write_barrier();
        cb_pop_front(cb_out, wmax);
    }
    // one increment per (token, distinct target worker); this core's columns are consecutive -> slots monotonic
    const uint32_t s_first = slot_of(first_col), s_last = slot_of(first_col + ncols - 1);
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t slot = s_first; slot <= s_last; ++slot) {
            const uint32_t f_idx = per_token * t + slot;
            const uint32_t nx = get_common_arg_val<uint32_t>(kFrontArgs + 2 * f_idx);
            const uint32_t ny = get_common_arg_val<uint32_t>(kFrontArgs + 1 + 2 * f_idx);
            noc_semaphore_inc(get_noc_addr(nx, ny, sem_addr), 1);
        }
    }
    noc_async_atomic_barrier();
#else
    constexpr int mm_fid = get_named_compile_time_arg_val("mm_fid");
    static_assert(ks == 0, "decode_matmul_norm: whole-K accumulation only");
    np_compute();
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    uint32_t cur = wmax;
    mm_block_init_f<mm_fid>(cb_in0, cb_in1, 0, wmax, 1, kt);
    cb_wait_front(cb_in0, kt);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        if (glu || wid != cur) {  // glu: the SFPU ops below re-program the math thread
            mm_block_init_f<mm_fid>(cb_in0, cb_in1, 0, wid, 1, kt);
            cur = wid;
        }
        // Whole K in FP32 DEST.
        tile_regs_acquire();
        for (uint32_t b = 0; b < kt / kb; ++b) {
            cb_wait_front(cb_in1, blk_tiles);
            for (uint32_t i = 0; i < kb; ++i) {
                mm_block_f<mm_fid>(cb_in0, cb_in1, b * kb + i, i * wid, 0, 0, wid, 1, kt);
            }
            cb_pop_front(cb_in1, blk_tiles);
        }
        constexpr uint32_t h = wmax / 2;
        if constexpr (glu) {
            // DEST tiles [0, h) gate, [h, 2h) up (glu_weight): gate <- SiLU(gate) * up, in FP32.
            silu_tile_init();
            for (uint32_t j = 0; j < h; ++j) {
                silu_tile(j);
            }
            mul_binary_tile_init();
            for (uint32_t j = 0; j < h; ++j) {
                mul_binary_tile(j, h + j, j);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, wmax);
        tile_regs_wait();
        for (uint32_t j = 0; j < (glu ? h : wid); ++j) {
            pack_tile(j, cb_out, j);
        }
        tile_regs_release();
        cb_push_back(cb_out, wmax);
    }
    cb_pop_front(cb_in0, kt);
#endif
}
