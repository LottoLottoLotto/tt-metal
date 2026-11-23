// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 attention decode, fused qkvg + front end: the qkvg matmul workers of attn_fused.AttnQKVFront (QKV_NORM 0) and
// attn_fused.AttnNormQKVFront (QKV_NORM 1: the sandwich residual + operator RMSNorm prologue of norm_prologue.hpp
// first, on the same cores, i.e. decode_matmul_norm.cpp's program).
//
// Same weight stream and math as decode_matmul_stream.cpp (StreamDecodeMatmul, preset attn_qkv, whole K in FP32
// DEST, ks = 0, no glu) / decode_matmul_norm.cpp (NormStreamDecodeMatmul): NCRISC streams the core's contiguous
// weight run from kernel start, TRISC accumulates, BRISC gathers the x rows (QKV_NORM 0) or runs the norm prologue
// (np_dm: h out, n multicast into every worker's in0).  Instead of writing qkvg to DRAM, BRISC sends every output
// column's token rows straight into the input tiles of the front workers (attn_decode.cpp with FRONT_FUSED 1): column
// c < 96 of qkvg is head c / 4 of q (0..15), k (16..19) or v (20..23), dim tile c % 4; head h of a section lands in
// row (h - h0) of the dim tile of the worker of that section's head group for token t.  Column 96 (the 16 gate logits
// + 16 zero columns) goes to the gate output tensor (rows t of its only tile).  After its writes have landed it
// increments each target worker's semaphore 0 once.  Workers with ncols = 0 (QKV_NORM: zero-padding columns) only
// take part in the norm.
//
// CBs: QKV_NORM 0: 12 x rows, 13 weight ring, 14 out, front input tiles 0; QKV_NORM 1: 0 in0 (= np_cb_x), 1 weight
// ring, 16 out, front input tiles 30 (allocated first on every core: the same L1 address everywhere), prologue CBs.
// Per-core args (all RISCs): [bank, run offset, ncols, first column].
// BRISC common args: [x (QKV_NORM 0) / weight (QKV_NORM 1) address, 0, gate address, then (QKV_NORM 1) the prologue's
// np::rt_args addresses, then per front worker (token-major, nq + nk + 1 per token) noc x, noc y].
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#ifndef QKV_NORM
#define QKV_NORM 0
#endif
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#if QKV_NORM
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/reconfig_data_format.h"
#endif
#endif
#if QKV_NORM
#include "norm_prologue.hpp"
#endif

#if QKV_NORM && !defined(COMPILE_FOR_NCRISC) && !defined(COMPILE_FOR_BRISC)
namespace ckernel {
// matmul_block_init / matmul_block with an explicit math fidelity F (decode_matmul_norm.cpp: the prologue's norm needs
// the kernel's HiFi4; the matmul keeps the stock decode fidelity)
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
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");      // K tiles
constexpr uint32_t wmax = get_named_compile_time_arg_val("w");     // subblock width (tiles)
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");      // K tiles per streamed block
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");  // valid activation rows (tokens)
#if QKV_NORM
constexpr uint32_t cb_in0 = 0;  // = np_cb_x
constexpr uint32_t cb_in1 = 1;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_front_in = 30;
#else
constexpr uint32_t cb_in0 = 12;
constexpr uint32_t cb_in1 = 13;
constexpr uint32_t cb_out = 14;
constexpr uint32_t cb_front_in = 0;
#endif
constexpr uint32_t blk_tiles = kb * wmax;
constexpr uint32_t kGateCol = 96;  // qkvg tile column of the gate logits
static_assert(kt % kb == 0);
static_assert(rows <= 16);
}  // namespace

void kernel_main() {
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
                // a per-trid barrier only on a block issued before the latest one (decode_matmul_stream.cpp)
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
    constexpr uint32_t nq = get_named_compile_time_arg_val("nq");  // q workers per token
    constexpr uint32_t nk = get_named_compile_time_arg_val("nk");  // k workers per token
    constexpr uint32_t per_token = nq + nk + 1;
    constexpr uint32_t qh = 16 / nq;  // q heads per q worker
    constexpr uint32_t kh = 4 / nk;   // k heads per k worker
    constexpr uint32_t x_tile_bytes = 2048;
    constexpr uint32_t x_face_bytes = 512;
    constexpr uint32_t x_row_bytes = 32;
    const auto gate = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2),
                                     x_tile_bytes);
#if QKV_NORM
    constexpr uint32_t kXyArg = 3 + np::rt_args;
    // norm prologue: n of the full row block lands in cb_in0 of every core (pushed, kt tiles)
    np_dm<0, 3>();
#else
    constexpr uint32_t kXyArg = 3;
    constexpr uint32_t x_top = (rows * x_row_bytes + 63) / 64 * 64;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), x_tile_bytes);
    cb_reserve_back(cb_in0, kt);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t t = 0; t < kt; ++t) {
        const uint64_t src = get_noc_addr(t, x);
        const uint32_t dst = in0 + t * x_tile_bytes;
        noc_async_read(src, dst, x_top);
        noc_async_read(src + x_face_bytes, dst + x_face_bytes, x_top);
    }
    noc_async_read_barrier();
    cb_push_back(cb_in0, kt);
#endif
    const uint32_t first_col = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = get_semaphore(0);
    const uint32_t front_in = get_write_ptr(cb_front_in);

    // column (< 96) -> (front slot, row in the worker's tiles); dim tile = column % 4
    auto slot_of = [](uint32_t col) -> uint32_t {
        const uint32_t h = col / 4;
        return h < 16 ? h / qh : (h < 20 ? nq + (h - 16) / kh : nq + nk);
    };
    auto row_of = [](uint32_t col) -> uint32_t {
        const uint32_t h = col / 4;
        return h < 16 ? h % qh : (h < 20 ? (h - 16) % kh : h - 20);
    };
    uint32_t col = first_col;
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        cb_wait_front(cb_out, wmax);
        DeviceZoneScopedN("qkv-send");
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < wid; ++j) {
            const uint32_t c = col + j;
            const uint32_t tile = src + j * x_tile_bytes;
            if (c == kGateCol) {  // gate logits: rows t of the gate output's only tile
                for (uint32_t t = 0; t < rows; ++t) {
                    const uint32_t off = (t & 15) * x_row_bytes;
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_write(tile + f * x_face_bytes + off, get_noc_addr(0, gate, f * x_face_bytes + off),
                                        x_row_bytes);
                    }
                }
                continue;
            }
            const uint32_t slot = slot_of(c), i = row_of(c), dt = c & 3;
            for (uint32_t t = 0; t < rows; ++t) {
                const uint32_t f_idx = per_token * t + slot;
                const uint32_t nx = get_common_arg_val<uint32_t>(kXyArg + 2 * f_idx);
                const uint32_t ny = get_common_arg_val<uint32_t>(kXyArg + 1 + 2 * f_idx);
                const uint32_t src_row = tile + (t & 15) * x_row_bytes;
                const uint32_t dst_row = front_in + dt * x_tile_bytes + (i & 15) * x_row_bytes;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write(src_row + f * x_face_bytes, get_noc_addr(nx, ny, dst_row + f * x_face_bytes),
                                    x_row_bytes);
                }
            }
        }
        col += wid;
        noc_async_write_barrier();
        cb_pop_front(cb_out, wmax);
    }
    // one increment per (token, distinct target worker); this core's columns are consecutive -> slots monotonic
    const uint32_t last_col = first_col + ncols - 1;
    if (ncols > 0 && first_col < kGateCol) {
        DeviceZoneScopedN("qkv-sem");
        const uint32_t s_first = slot_of(first_col);
        const uint32_t s_last = slot_of(last_col < kGateCol ? last_col : kGateCol - 1);
        for (uint32_t t = 0; t < rows; ++t) {
            for (uint32_t slot = s_first; slot <= s_last; ++slot) {
                const uint32_t f_idx = per_token * t + slot;
                const uint32_t nx = get_common_arg_val<uint32_t>(kXyArg + 2 * f_idx);
                const uint32_t ny = get_common_arg_val<uint32_t>(kXyArg + 1 + 2 * f_idx);
                noc_semaphore_inc(get_noc_addr(nx, ny, sem_addr), 1);
            }
        }
        noc_async_atomic_barrier();
    }
#else
#if QKV_NORM
    constexpr int mm_fid = get_named_compile_time_arg_val("mm_fid");
    np_compute();
#define MM_INIT mm_block_init_f<mm_fid>
#define MM_BLOCK mm_block_f<mm_fid>
#else
#define MM_INIT matmul_block_init
#define MM_BLOCK matmul_block
#endif
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    uint32_t cur = wmax;
    MM_INIT(cb_in0, cb_in1, 0, wmax, 1, kt);
    cb_wait_front(cb_in0, kt);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        if (wid != cur) {
            MM_INIT(cb_in0, cb_in1, 0, wid, 1, kt);
            cur = wid;
        }
        tile_regs_acquire();
        for (uint32_t b = 0; b < kt / kb; ++b) {
            cb_wait_front(cb_in1, blk_tiles);
            for (uint32_t i = 0; i < kb; ++i) {
                MM_BLOCK(cb_in0, cb_in1, b * kb + i, i * wid, 0, 0, wid, 1, kt);
            }
            cb_pop_front(cb_in1, blk_tiles);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, wmax);
        tile_regs_wait();
        for (uint32_t j = 0; j < wid; ++j) {
            pack_tile(j, cb_out, j);
        }
        tile_regs_release();
        cb_push_back(cb_out, wmax);
    }
    cb_pop_front(cb_in0, kt);
#endif
}
