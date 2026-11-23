// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 gated short conv, whole decode operator in one launch (conv_fused.py, ConvOpDecode):
//
//   [n   = operator_norm(h), h = a + post_norm(b)    (CF_NORM: norm_prologue.hpp, n lands in cb_x)]
//   bcx = x @ W_in                     (2048 x 6144 BFP8/BFP4, stream layout of decode_matmul.py, FP32 DEST, BF16 out)
//   y   = C * conv7(s0..s5, B * x); s_j <- s_{j+1}; s5 <- B * x    (conv_tail.hpp: bitwise stock, state in place)
//   out = y @ W_out                    (2048 x 2048 BFP8/BFP4, stream layout, FP32 DEST, BF16 out)
//
// n_workers workers (placement "rows": StreamDecodeMatmul's conv_in workers plus one hub core, together whole grid
// rows; placement (x, y): an 8 x 4 rectangle whose first worker is also the hub).  Every worker is an in_proj worker
// (its stream run of bcx columns), a tail owner (`per` hidden columns j) and an out_proj worker (its run of out
// columns).  Matmul numerics: decode_matmul_stream.cpp's matmul_block sequence (whole K in FP32 DEST, BF16 pack).
//
// NCRISC (NOC0): streams the core's W_in run (ring cb_win), then its W_out run into cb_wout (holds the whole run, so
//                the out_proj weights arrive while the tail and the y exchange run), transaction-id pipelined.
// BRISC (NOC1):  taps / state rows of its tail columns, x rows -> cb_x (or the norm prologue); every bcx tile it
//                computes -> the owner core's B/C/x slot + owner semaphore 0; once its 3*per tiles are in, the tail
//                finishes; its y rows -> the hub's cb_y slot j (+ hub semaphore 1), then the state shift (DRAM, in
//                place); the hub multicasts its whole cb_y to every core, then sets their semaphore 2; out_proj
//                runs; out rows out.
// TRISC:         [norm] -> state partial sum (s0..s5 . t0..t5, FP32; independent of this step's bcx, so it runs
//                before the in_proj matmul) -> in_proj matmul -> tail finish -> out_proj matmul.
//
// Only rows [0, rows) move (64 B granules; padding rows of y / out / bcx are undefined, as ttnn padding; keep_row:
// the extra state row of an odd row count belongs to another slot and is written back unchanged).
//
// CF_VERIFY (speculative verification, ConvOpDecode(verify=cand, vmap=...)): the rows are groups of consecutive tokens,
// one per sequence (conv_tail::verify_context: row map, slot per group).  BRISC stages the state rows [0, rows) and
// the row map in kRaw; TRISC runs in_proj first, then every row's Bx (conv_tail::bx_only); BRISC builds each row's
// left context from its slot's state and its group's earlier rows before TRISC's state partial, so every row is
// bitwise the sequential 1-row steps'.  The state is NOT updated: every row's Bx goes to the candidate tensor (BRISC
// common arg 17, the row map 18; tt/speculate.py SpecCommit commits each group's accepted rows).
#include <cstdint>
#ifndef CF_VERIFY
#define CF_VERIFY 0
#endif
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#endif
#include "conv_tail.hpp"
#if CF_NORM
// Residual add + RMSNorm prologue (norm_prologue.hpp): n lands in cb_x on every core instead of the x gather
#if !defined(COMPILE_FOR_NCRISC) && !defined(COMPILE_FOR_BRISC)
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif
#include "norm_prologue.hpp"
#endif

#if !defined(COMPILE_FOR_NCRISC) && !defined(COMPILE_FOR_BRISC)
namespace ckernel {
// matmul_block_init / matmul_block (api/compute/matmul.h) with an explicit math fidelity F instead of the kernel's
// MATH_FIDELITY (copied from decode_matmul_norm.cpp: the norm prologue needs HiFi4, the matmuls stay HiFi2).
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
#if CF_PROF
#include "tools/profiler/kernel_profiler.hpp"
#define CF_ZONE(name) DeviceZoneScopedN(name)
#else
#define CF_ZONE(name)
#endif

namespace cf {
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");        // K tiles of both matmuls (hidden / 32)
constexpr uint32_t ht = get_named_compile_time_arg_val("ht");        // hidden tiles (= kt)
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");    // valid activation rows
constexpr bool keep_row = get_named_compile_time_arg_val("keep_row") != 0;  // odd rows of a wider state
constexpr uint32_t per = get_named_compile_time_arg_val("per");      // tail columns per core
constexpr uint32_t w_in = get_named_compile_time_arg_val("w_in");    // in_proj subblock width
constexpr uint32_t kb_in = get_named_compile_time_arg_val("kb_in");  // in_proj K tiles per streamed block
constexpr uint32_t w_out = get_named_compile_time_arg_val("w_out");
constexpr uint32_t kb_out = get_named_compile_time_arg_val("kb_out");
constexpr uint32_t blk_in = kb_in * w_in;
constexpr uint32_t blk_out = kb_out * w_out;
static_assert(kt % kb_in == 0 && kt % kb_out == 0);
constexpr uint32_t kTile = conv_tail::kTile;  // BF16 tile bytes
constexpr uint32_t kState = conv_tail::kState;
// out_proj in0 (cb_y) / out (cb_out) tiles are th x 32 (th = 8 / 16 for <= 8 / 16 rows: the matmul then runs on
// 2 partial faces, 2-4x less math; the valid rows are computed identically)
constexpr uint32_t th = get_named_compile_time_arg_val("th");
constexpr uint32_t kSmall = th * 64;  // th x 32 BF16 tile bytes
static_assert(th == 8 || th == 16 || th == 32);
static_assert(rows <= th);

// Circular buffers (12-15 and 24-29 belong to the norm prologue)
constexpr uint32_t cb_x = 0;     // in_proj in0 (x rows), BRISC -> TRISC
constexpr uint32_t cb_win = 1;   // in_proj weight ring, NCRISC -> TRISC
constexpr uint32_t cb_wout = 2;  // out_proj weights (whole run), NCRISC -> TRISC
constexpr uint32_t cb_y = 3;     // out_proj in0: all y tiles, filled by the hub's multicast
constexpr uint32_t kB = 4;       // received bcx tiles (tail CBs 4-10: UnpackToDestFp32)
constexpr uint32_t kX = 5;
constexpr uint32_t kC = 6;
constexpr uint32_t kT = 7;       // taps t0..t5 (per * 6), BRISC -> TRISC
constexpr uint32_t kT6 = 8;      // tap t6
constexpr uint32_t kS = 9;       // state s0..s5 (per * 6)
constexpr uint32_t kAcc = 10;    // FP32 state partial
constexpr uint32_t kBxOut = 11;  // -> BRISC (new s5)
constexpr uint32_t kY = 16;      // -> BRISC (to the hub)
constexpr uint32_t cb_bcx = 17;  // in_proj out -> BRISC
constexpr uint32_t cb_out = 18;  // out_proj out -> BRISC
constexpr uint32_t kRaw = 19;    // CF_VERIFY: BRISC scratch, the state rows as read (per * 6 tiles) + the row map
}  // namespace cf

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace cf {
using RowsT = conv_tail::Rows<rows>;
// Row granules between a 32x32 tile (offset `off`) and a th x 32 tile (face k at k * th * 32).
constexpr uint32_t small_off(uint32_t off) { return th == 32 ? off : off / 512 * th * 32; }
}  // namespace cf
#endif

#if defined(COMPILE_FOR_TRISC)
namespace cf {
// Tail SFPU ops cover only the faces holding valid rows (faces 0/1 for <= 16 rows)
constexpr VectorMode kVec = rows <= 16 ? VectorMode::R : VectorMode::RC;
constexpr int mm_fid = get_named_compile_time_arg_val("mm_fid");  // the decode matmuls' fidelity (HiFi2)

// One column-block matmul pass, identical to decode_matmul_stream.cpp (ks == 0): per subblock of <= wmax columns
// the whole K accumulates in FP32 DEST, then the subblock is packed (BF16) into `outcb` (wmax slots per subblock).
template <uint32_t in0, uint32_t in1, uint32_t outcb, uint32_t wmax, uint32_t kb>
FORCE_INLINE void matmul_pass(uint32_t ncols) {
    constexpr uint32_t blk = kb * wmax;
    const uint32_t nsub = (ncols + wmax - 1) / wmax;
    const uint32_t rem = ncols - (nsub - 1) * wmax;
    uint32_t cur = wmax;
    mm_block_init_f<mm_fid>(in0, in1, 0, wmax, 1, kt);
    cb_wait_front(in0, kt);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        if (wid != cur) {
            mm_block_init_f<mm_fid>(in0, in1, 0, wid, 1, kt);
            cur = wid;
        }
        tile_regs_acquire();
        for (uint32_t b = 0; b < kt / kb; ++b) {
            cb_wait_front(in1, blk);
            for (uint32_t i = 0; i < kb; ++i) {
                mm_block_f<mm_fid>(in0, in1, b * kb + i, i * wid, 0, 0, wid, 1, kt);
            }
            cb_pop_front(in1, blk);
        }
        tile_regs_commit();
        cb_reserve_back(outcb, wmax);
        tile_regs_wait();
        for (uint32_t j = 0; j < wid; ++j) {
            pack_tile(j, outcb, j);
        }
        tile_regs_release();
        cb_push_back(outcb, wmax);
    }
    cb_pop_front(in0, kt);
}
}  // namespace cf
#endif

void kernel_main() {
    using namespace cf;
#if defined(COMPILE_FOR_NCRISC)
    // common: [w_in, w_out]; per core: [bank, in_run, in_ncols, out_run, out_ncols] (runs: bank-relative bytes)
    constexpr uint32_t in_bytes = get_named_compile_time_arg_val("wt_in_bytes");
    constexpr uint32_t out_bytes = get_named_compile_time_arg_val("wt_out_bytes");
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t outstanding = get_named_compile_time_arg_val("outstanding");
    static_assert(outstanding >= 1 && outstanding < slots && outstanding <= 8);
    constexpr uint32_t slot_bytes = blk_in * in_bytes;
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint64_t src_in = get_noc_addr_from_bank_id<true>(bank, get_common_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(1));
    const uint32_t in_ncols = get_arg_val<uint32_t>(2);
    const uint64_t src_out = get_noc_addr_from_bank_id<true>(bank, get_common_arg_val<uint32_t>(1) + get_arg_val<uint32_t>(3));
    const uint32_t out_ncols = get_arg_val<uint32_t>(4);
    if (in_ncols == 0) {
        return;  // hub core
    }
    const uint32_t nsub_in = (in_ncols + w_in - 1) / w_in;
    const uint32_t rem_in = in_ncols - (nsub_in - 1) * w_in;
    const uint32_t nsub_out = (out_ncols + w_out - 1) / w_out;
    const uint32_t rem_out = out_ncols - (nsub_out - 1) * w_out;
    const uint32_t n_in = nsub_in * (kt / kb_in);
    const uint32_t n_all = n_in + nsub_out * (kt / kb_out);
    const uint32_t ring_base = get_write_ptr(cb_win);
    const uint32_t ring_end = ring_base + slots * slot_bytes;
    cb_reserve_back(cb_wout, nsub_out * (kt / kb_out) * blk_out);  // the whole run fits: never waits
    const uint32_t out_base = get_write_ptr(cb_wout);
    uint32_t off_in = 0, off_out = 0;
    uint32_t pending_in = 0;  // in blocks issued, not pushed
    uint32_t queued = 0;      // blocks issued, not pushed (all in blocks precede all out blocks)
    uint32_t pushed = 0;      // blocks pushed
    uint32_t issue_trid = 1, wait_trid = 1;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    auto push_oldest = [&]() {
        if (pushed < n_in) {
            cb_push_back(cb_win, blk_in);
            --pending_in;
        } else {
            cb_push_back(cb_wout, blk_out);
        }
        ++pushed;
        --queued;
    };
    auto issue = [&](uint32_t g) {
        uint64_t src;
        uint32_t dst, bytes;
        if (g < n_in) {
            const uint32_t s = g / (kt / kb_in);
            bytes = kb_in * (s == nsub_in - 1 ? rem_in : w_in) * in_bytes;
            cb_reserve_back(cb_win, blk_in * (pending_in + 1));
            dst = get_write_ptr(cb_win) + pending_in * slot_bytes;
            if (dst >= ring_end) {
                dst -= slots * slot_bytes;
            }
            src = src_in + off_in;
            off_in += bytes;
            ++pending_in;
        } else {
            const uint32_t o = g - n_in;
            const uint32_t s = o / (kt / kb_out);
            bytes = kb_out * (s == nsub_out - 1 ? rem_out : w_out) * out_bytes;
            dst = out_base + o * blk_out * out_bytes;
            src = src_out + off_out;
            off_out += bytes;
        }
        noc_async_read_set_trid(issue_trid);
        noc_async_read(src, dst, bytes);
        issue_trid = issue_trid == outstanding ? 1 : issue_trid + 1;
        if (++queued == outstanding) {
            // per-trid barrier only on a block issued before the latest one (see decode_matmul_stream.cpp)
            if constexpr (outstanding == 1) {
                noc_async_read_barrier();
            } else {
                noc_async_read_barrier_with_trid(wait_trid);
            }
            push_oldest();
            wait_trid = wait_trid == outstanding ? 1 : wait_trid + 1;
        }
    };
    {
        CF_ZONE("n-in");
        for (uint32_t g = 0; g < n_in; ++g) {
            issue(g);
        }
    }
    {
        CF_ZONE("n-out");
        for (uint32_t g = n_in; g < n_all; ++g) {
            issue(g);
        }
        noc_async_read_barrier();
        while (queued > 0) {
            push_oldest();
        }
        noc_async_read_set_trid(0);
        noc_async_read_barrier();
    }
#elif defined(COMPILE_FOR_BRISC)
    // common: [x, out, taps, s0..s5, mcast sx, sy, ex, ey (whole rectangle, NOC1 order), mcast dests (rectangle - 1),
    //          n_workers, hub x, hub y] (+ the norm prologue's np::rt_args)
    // per core: [idx, first_in_col, in_ncols (0: hub only), first_out_col, out_ncols, hub role (0 / 1),
    //            (owner x, owner y) per in column]
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    [[maybe_unused]] const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), kTile);  // !CF_NORM
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), kTile);
    const auto taps = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2),
                                     kTile);
    using SAcc = decltype(TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), 0u, kTile));
    SAcc s[kState] = {
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3), kTile),
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4), kTile),
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(5), kTile),
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6), kTile),
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(7), kTile),
        TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(8), kTile)};
    const uint32_t n_workers = get_common_arg_val<uint32_t>(14);
    const uint32_t hub_x = get_common_arg_val<uint32_t>(15), hub_y = get_common_arg_val<uint32_t>(16);
    const uint32_t idx = get_arg_val<uint32_t>(0);
    const uint32_t first_in = get_arg_val<uint32_t>(1);
    const uint32_t in_ncols = get_arg_val<uint32_t>(2);
    const uint32_t first_out = get_arg_val<uint32_t>(3);
    const uint32_t out_ncols = get_arg_val<uint32_t>(4);
    const uint32_t sem_bcx = get_semaphore(0);     // workers: own B / C / x tiles received
    const uint32_t sem_gather = get_semaphore(1);  // hub: workers whose y rows are in
    const uint32_t sem_ready = get_semaphore(2);   // workers: the hub's y multicast has landed
    // receive slots (identical CB layout on every core; none of these CBs has been pushed yet)
    const uint32_t recv[3] = {get_write_ptr(kB), get_write_ptr(kC), get_write_ptr(kX)};
    const uint32_t y_base = get_write_ptr(cb_y);

    // hub: once every worker's y rows are in its cb_y, one multicast of the whole y buffer to the rest of the
    // rectangle (a hub inside it keeps its own copy), then their ready flags
    auto run_hub = [&]() {
        CF_ZONE("hub");
        auto* gathered = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_gather);
        noc_semaphore_wait(gathered, n_workers);
        noc_semaphore_set(gathered, 0);
        const uint32_t sx = get_common_arg_val<uint32_t>(9), sy = get_common_arg_val<uint32_t>(10);
        const uint32_t ex = get_common_arg_val<uint32_t>(11), ey = get_common_arg_val<uint32_t>(12);
        const uint32_t n_dests = get_common_arg_val<uint32_t>(13);
        noc_async_write_multicast(y_base, get_noc_multicast_addr(sx, sy, ex, ey, y_base), kt * kSmall, n_dests);
        noc_async_write_barrier();
        auto* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready);
        noc_semaphore_set(flag, 1);
        noc_semaphore_set_multicast(sem_ready, get_noc_multicast_addr(sx, sy, ex, ey, sem_ready), n_dests);
        noc_async_write_barrier();
        noc_semaphore_set(flag, 0);
    };
    if (in_ncols == 0) {
        run_hub();
        return;
    }
    const bool is_hub = get_arg_val<uint32_t>(5) != 0;

    // 1. taps / state rows of the tail columns j = idx * per + i (this core owns these state columns; their in-place
    //    writes come in step 4, after TRISC consumed them), then the in_proj in0: x rows, or (CF_NORM)
    //    n = operator_norm(h) from the norm prologue (also writes h).  The state reads are issued first and land
    //    under the prologue's / the x read barrier.
    const uint32_t s_l1 = get_write_ptr(kS);
    const uint32_t s_dst = CF_VERIFY ? get_write_ptr(kRaw) : s_l1;  // verify: the contexts are built in step 3
    {
        CF_ZONE("b1-x");
        cb_reserve_back(kT, per * kState);
        cb_reserve_back(kS, per * kState);
        cb_reserve_back(kT6, per);
        for (uint32_t i = 0; i < per; ++i) {
            const uint32_t j = idx * per + i;
            for (uint32_t k = 0; k < kState; ++k) {
                RowsT::read(get_noc_addr(k * ht + j, taps), get_write_ptr(kT) + (i * kState + k) * kTile);
                RowsT::read(get_noc_addr(j, s[k]), s_dst + (i * kState + k) * kTile);
            }
            RowsT::read(get_noc_addr(kState * ht + j, taps), get_write_ptr(kT6) + i * kTile);
        }
#if CF_VERIFY
        noc_async_read(TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                      get_common_arg_val<uint32_t>(18), 64).get_noc_addr(0),
                       s_dst + per * kState * kTile, 64);
#endif
#if CF_NORM
        np_dm<out_args.next_compile_time_args_offset(), 17 + 2 * CF_VERIFY>();
        noc_async_read_barrier();
#else
        cb_reserve_back(cb_x, kt);
        const uint32_t in0 = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < kt; ++t) {
            RowsT::read(get_noc_addr(t, x), in0 + t * kTile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, kt);
#endif
        cb_push_back(kT, per * kState);
#if !CF_VERIFY
        cb_push_back(kS, per * kState);
#endif
        cb_push_back(kT6, per);
    }

    // 2. bcx tiles -> owners (B / C / x slot of hidden column j = t % ht on core j / per)
    {
        CF_ZONE("b2-bcx");
        const uint32_t nsub = (in_ncols + w_in - 1) / w_in;
        const uint32_t rem = in_ncols - (nsub - 1) * w_in;
        uint32_t c = 0;
        for (uint32_t sb = 0; sb < nsub; ++sb) {
            const uint32_t wid = sb == nsub - 1 ? rem : w_in;
            cb_wait_front(cb_bcx, w_in);
            const uint32_t src = get_read_ptr(cb_bcx);
            for (uint32_t jj = 0; jj < wid; ++jj) {
                const uint32_t t = first_in + c + jj;
                const uint32_t j = t % ht;
                const uint32_t dst = recv[t / ht] + (j % per) * kTile;
                const uint32_t ox = get_arg_val<uint32_t>(6 + 2 * (c + jj)), oy = get_arg_val<uint32_t>(7 + 2 * (c + jj));
                RowsT::write(src + jj * kTile, get_noc_addr(ox, oy, dst));
            }
            noc_async_write_barrier();
            for (uint32_t jj = 0; jj < wid; ++jj) {
                const uint32_t ox = get_arg_val<uint32_t>(6 + 2 * (c + jj)), oy = get_arg_val<uint32_t>(7 + 2 * (c + jj));
                noc_semaphore_inc(get_noc_addr(ox, oy, sem_bcx), 1);
            }
            cb_pop_front(cb_bcx, w_in);
            c += wid;
        }
    }

    // 3. own B / C / x tiles -> tail
    {
        CF_ZONE("b3-wait");
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_bcx);
        noc_semaphore_wait(sem, 3 * per);
        noc_semaphore_set(sem, 0);
        for (uint32_t cb = kB; cb <= kC; ++cb) {
            cb_reserve_back(cb, per);
            cb_push_back(cb, per);
        }
#if CF_VERIFY
        // every row's left context from its slot's state rows (kRaw) and this step's Bx rows (TRISC: bx_only right
        // after in_proj), then the state tiles go to TRISC's state partial
        cb_wait_front(kBxOut, per);
        conv_tail::verify_context<rows>(s_l1, s_dst, get_read_ptr(kBxOut), per,
                                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_dst + per * kState * kTile));
        cb_push_back(kS, per * kState);
#endif
    }

    // 4. y rows -> the hub's cb_y slot j, then the hub's gather semaphore (the barrier sees only L1 writes: DRAM
    //    write acks are slow while the weight streams saturate DRAM); then the state shift s_j <- s_{j+1}, s5 <- Bx
    //    from the L1 copies (TRISC consumed them before its in_proj matmul)
    {
        CF_ZONE("b4-y");
        cb_wait_front(kY, per);
        for (uint32_t i = 0; i < per; ++i) {
            const uint32_t src = get_read_ptr(kY) + i * kTile;
            const uint64_t dst = get_noc_addr(hub_x, hub_y, y_base + (idx * per + i) * kSmall);
            RowsT::chunks([&](uint32_t off, uint32_t n) { noc_async_write(src + off, dst + small_off(off), n); });
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, sem_gather), 1);
        cb_wait_front(kBxOut, per);
#if CF_VERIFY
        // every row's Bx -> the candidate tensor (the state is committed later for the accepted rows)
        const auto cand = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                         get_common_arg_val<uint32_t>(17), kTile);
        for (uint32_t i = 0; i < per; ++i) {
            RowsT::write(get_read_ptr(kBxOut) + i * kTile, get_noc_addr(idx * per + i, cand));
        }
#else
        conv_tail::shift_state<rows, keep_row>(s, s_l1, get_read_ptr(kBxOut), idx * per, per);
#endif
    }

    // 5. the hub's multicast of all y tiles landed -> out_proj
    if (is_hub) {
        run_hub();
    } else {
        CF_ZONE("b5-wait");
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready);
        noc_semaphore_wait(sem, 1);
        noc_semaphore_set(sem, 0);
    }
    cb_reserve_back(cb_y, kt);
    cb_push_back(cb_y, kt);

    // 6. out rows
    {
        CF_ZONE("b6-out");
        const uint32_t nsub = (out_ncols + w_out - 1) / w_out;
        const uint32_t rem = out_ncols - (nsub - 1) * w_out;
        uint32_t col = first_out;
        for (uint32_t sb = 0; sb < nsub; ++sb) {
            const uint32_t wid = sb == nsub - 1 ? rem : w_out;
            cb_wait_front(cb_out, w_out);
            const uint32_t src = get_read_ptr(cb_out);
            for (uint32_t jj = 0; jj < wid; ++jj) {
                const uint32_t t = src + jj * kSmall;
                const uint64_t dst = get_noc_addr(col + jj, out);
                RowsT::chunks([&](uint32_t off, uint32_t n) { noc_async_write(t + small_off(off), dst + off, n); });
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, w_out);
            col += wid;
        }
    }
    noc_async_write_barrier();
    cb_pop_front(kY, per);
    cb_pop_front(kBxOut, per);
    noc_async_atomic_barrier();
#elif defined(COMPILE_FOR_TRISC)
    // per core: [in_ncols (0: hub), out_ncols]
    const uint32_t in_ncols = get_arg_val<uint32_t>(0);
    const uint32_t out_ncols = get_arg_val<uint32_t>(1);
    if (in_ncols == 0) {
        return;  // hub core
    }
#if CF_NORM
    {
        CF_ZONE("t-norm");
        np_compute();  // leaves the unpacker / packer configured for the norm: re-initialized below
    }
#endif
#if CF_VERIFY
    // in_proj, then every row's Bx (BRISC builds the rows' left contexts from it), the state partial, the tail
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_win, cb_bcx);
    {
        CF_ZONE("t-in");
        matmul_pass<cb_x, cb_win, cb_bcx, w_in, kb_in>(in_ncols);
    }
    {
        CF_ZONE("t-tail");
        conv_tail::bx_only<per, kVec>(kB, kX, kBxOut);
        conv_tail::state_partial<per, kVec>(kS, kT, kAcc);
        conv_tail::finish<per, kVec, false>(kB, kX, kC, kT6, kAcc, kY, kBxOut);
    }
#else
    {
        CF_ZONE("t-state");
        compute_kernel_hw_startup(kS, kAcc);
        conv_tail::state_partial<per, kVec>(kS, kT, kAcc);
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_win, cb_bcx);
    {
        CF_ZONE("t-in");
        matmul_pass<cb_x, cb_win, cb_bcx, w_in, kb_in>(in_ncols);
    }
    {
        CF_ZONE("t-tail");
        conv_tail::finish<per, kVec>(kB, kX, kC, kT6, kAcc, kY, kBxOut);
    }
#endif
    {
        CF_ZONE("t-out");
        reconfig_data_format_srca(cb_wout);
        reconfig_data_format_srcb<true>(cb_y);    // th x 32 in0 tiles
        pack_reconfig_data_format<true>(cb_out);  // th x 32 out tiles
        matmul_pass<cb_y, cb_wout, cb_out, w_out, kb_out>(out_ncols);
    }
#endif
}
