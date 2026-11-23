// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 gated short conv, whole decode operator in one launch (conv_fused.py, ConvOpDecode):
//
//   bcx = x @ W_in                     (2048 x 6144 BFP8, stream layout of decode_matmul.py, FP32 DEST, BF16 out)
//   y   = C * (t0*s0 + t1*s1 + t2*(B*x)); s0 <- s1; s1 <- B*x       (conv_decode.cpp's ops, state in place)
//   out = y @ W_out                    (2048 x 2048 BFP8, stream layout, FP32 DEST, BF16 out)
//
// n_workers workers (StreamDecodeMatmul's conv_in placement: the first workers row-major, cores_per_bank per DRAM
// bank) plus one hub core, together one rectangle.  Every worker is an in_proj worker (its stream run of bcx
// columns), a tail owner (`per` hidden columns j) and an out_proj worker (its run of out columns).  Numerics: same
// matmul_block sequence as decode_matmul_stream.cpp (whole K in FP32 DEST, BF16 pack) and the same SFPU arithmetic
// as the ttnn BF16 mul / addcmul (their 16-bit-DEST variants, which round every result to BF16 with RNE; stored in
// this kernel's FP32 DEST they are exact BF16 values, so the BF16 pack is exact).
//
// NCRISC (NOC0): streams the core's W_in run (ring cb_win), then its W_out run into cb_wout (holds the whole run, so
//                the out_proj weights arrive while the tail and the y exchange run), transaction-id pipelined.
// BRISC (NOC1):  x rows -> cb_x; taps / state rows of its tail columns (then s0 <- old s1 in DRAM); every bcx tile
//                it computes -> the owner core's B/C/x slot + owner semaphore 0; once its 3*per tiles are in, the
//                tail runs; its y rows -> the hub's cb_y slot j (+ hub semaphore 1), new s1 rows -> DRAM; the hub
//                multicasts its whole cb_y to every core, then sets their semaphore 2; out_proj runs; out rows out.
//                (32 concurrent multicasts, or 32 x 32 unicast row writes, measured 40 / 11 us: one hub is ~3 us.)
// TRISC:         in_proj matmul -> tail (both columns per op, grouped by unpack mode) -> out_proj matmul.
//
// Only rows [0, rows) move (64 B granules; padding rows of y / out / bcx are undefined, as ttnn padding).
//
// CF_VERIFY (speculative verification, ConvOpDecode(verify=...)): the rows are consecutive tokens of ONE sequence
// (conv-state row 0).  Row r's left context is the previous rows of this step: s1_r = Bx_{r-1}, s0_r = Bx_{r-2}
// (rows -1 / -2: the state's s1 / s0), built by BRISC from the Bx tiles once TRISC has computed them; the tail
// arithmetic per row is unchanged (bitwise the sequential 1-row steps).  The state is NOT updated: every row's Bx goes
// to the candidate tensor (tt/speculate.py SpecCommit later writes s0 <- Bx_{a-1}, s1 <- Bx_a for the accepted row a).
#include <cstdint>
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#endif
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
constexpr bool keep_row = get_named_compile_time_arg_val("keep_row") != 0;  // odd rows < 16 of a wider state
constexpr uint32_t per = get_named_compile_time_arg_val("per");      // tail columns per core
constexpr uint32_t w_in = get_named_compile_time_arg_val("w_in");    // in_proj subblock width
constexpr uint32_t kb_in = get_named_compile_time_arg_val("kb_in");  // in_proj K tiles per streamed block
constexpr uint32_t w_out = get_named_compile_time_arg_val("w_out");
constexpr uint32_t kb_out = get_named_compile_time_arg_val("kb_out");
constexpr uint32_t blk_in = kb_in * w_in;
constexpr uint32_t blk_out = kb_out * w_out;
static_assert(kt % kb_in == 0 && kt % kb_out == 0);
constexpr uint32_t kTile = 2048;  // BF16 tile bytes
// out_proj in0 (cb_y) / out (cb_out) tiles are th x 32 (th = 8 / 16 for <= 8 / 16 rows: the matmul then runs on
// 2 partial faces, 2-4x less math; the valid rows are computed identically)
constexpr uint32_t th = get_named_compile_time_arg_val("th");
constexpr uint32_t kSmall = th * 64;  // th x 32 BF16 tile bytes
static_assert(th == 8 || th == 16 || th == 32);
static_assert(rows <= th);

// Circular buffers
constexpr uint32_t cb_x = 0;     // in_proj in0 (x rows), BRISC -> TRISC
constexpr uint32_t cb_win = 1;   // in_proj weight ring, NCRISC -> TRISC
constexpr uint32_t cb_wout = 2;  // out_proj weights (whole run), NCRISC -> TRISC
constexpr uint32_t cb_y = 3;     // out_proj in0: all y tiles, filled by every owner's multicast
constexpr uint32_t kB = 4;       // received bcx tiles (UnpackToDestFp32)
constexpr uint32_t kX = 5;       // (UnpackToDestFp32)
constexpr uint32_t kC = 6;       // (UnpackToDestFp32)
constexpr uint32_t kT2 = 7;      // taps / state, BRISC -> TRISC (t2: UnpackToDestFp32)
constexpr uint32_t kS1 = 8;
constexpr uint32_t kT1 = 9;
constexpr uint32_t kS0 = 10;
constexpr uint32_t kT0 = 11;
constexpr uint32_t kBx = 16;      // (UnpackToDestFp32)
constexpr uint32_t kBxOut = 17;   // -> BRISC (new s1)
constexpr uint32_t kAcc = 18;
constexpr uint32_t kAcc2 = 19;
constexpr uint32_t kAcc3 = 20;    // (UnpackToDestFp32)
constexpr uint32_t kY = 21;       // -> BRISC (multicast)
constexpr uint32_t cb_bcx = 22;   // in_proj out -> BRISC
constexpr uint32_t cb_out = 23;   // out_proj out -> BRISC
}  // namespace cf

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace cf {
// Row granules of a BF16 tile: rows [0, top) of faces 0/1 and [0, bottom) of faces 2/3, rounded up to 64 B.
constexpr uint32_t top = rows < 16 ? rows : 16;
constexpr uint32_t bottom = rows > 16 ? rows - 16 : 0;
constexpr uint32_t c_top = (top * 32 + 63) / 64 * 64;
constexpr uint32_t c_bot = (bottom * 32 + 63) / 64 * 64;

template <typename F>
FORCE_INLINE void chunks(F&& f) {
    f(0u, c_top);
    f(512u, c_top);
    if constexpr (bottom > 0) {
        f(1024u, c_bot);
        f(1536u, c_bot);
    }
}

FORCE_INLINE void read_rows(uint64_t src, uint32_t dst) {
    chunks([&](uint32_t off, uint32_t n) { noc_async_read(src + off, dst + off, n); });
}

FORCE_INLINE void write_rows(uint32_t src, uint64_t dst) {
    chunks([&](uint32_t off, uint32_t n) { noc_async_write(src + off, dst + off, n); });
}

// Row granules between a 32x32 tile (offset `off`) and a th x 32 tile (face k at k * th * 32).
constexpr uint32_t small_off(uint32_t off) { return th == 32 ? off : off / 512 * th * 32; }
}  // namespace cf
#endif

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
namespace ckernel::sfpu {
// calculate_addcmul<..., is_fp32_dest_acc_en = false, Float16_b> (the ttnn BF16 addcmul arithmetic) for an FP32
// DEST: the SFPSTOCHRND FP16B result keeps stale low mantissa bits that a 16-bit DEST store drops; masking them
// stores exactly the value the 16-bit DEST holds (upper 16 bits), so the BF16 pack is exact.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void cf_addcmul_bf16(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1,
                            const std::uint32_t dst_index_in2, const std::uint32_t dst_index_out,
                            const std::uint32_t value) {
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    const sfpi::vFloat value_float = Converter::as_float(value);
    const std::uint32_t off_in0 = dst_index_in0 * dst_tile_size_sfpi;
    const std::uint32_t off_in1 = dst_index_in1 * dst_tile_size_sfpi;
    const std::uint32_t off_in2 = dst_index_in2 * dst_tile_size_sfpi;
    const std::uint32_t off_out = dst_index_out * dst_tile_size_sfpi;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat a_in0 = sfpi::dst_reg[off_in0];
        sfpi::vFloat a_in1 = sfpi::dst_reg[off_in1];
        sfpi::vFloat a_in2 = sfpi::dst_reg[off_in2];
        sfpi::vFloat b_in0 = sfpi::dst_reg[off_in0 + 1];
        sfpi::vFloat b_in1 = sfpi::dst_reg[off_in1 + 1];
        sfpi::vFloat b_in2 = sfpi::dst_reg[off_in2 + 1];
        sfpi::vFloat a_prod = value_float * a_in1;
        sfpi::vFloat b_prod = value_float * b_in1;
        sfpi::vFloat a_res = a_prod * a_in2 + a_in0;
        sfpi::vFloat b_res = b_prod * b_in2 + b_in0;
        sfpi::vUInt a_bits = sfpi::as<sfpi::vUInt>(sfpi::convert<sfpi::vFloat16b>(a_res, sfpi::RoundMode::Nearest));
        sfpi::vUInt b_bits = sfpi::as<sfpi::vUInt>(sfpi::convert<sfpi::vFloat16b>(b_res, sfpi::RoundMode::Nearest));
        sfpi::dst_reg[off_out] = sfpi::as<sfpi::vFloat>(a_bits & 0xFFFF0000U);
        sfpi::dst_reg[off_out + 1] = sfpi::as<sfpi::vFloat>(b_bits & 0xFFFF0000U);
        sfpi::dst_reg += 2;
    }
}
}  // namespace ckernel::sfpu
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

// ttnn.mul (binary_ng BF16 SFPU path, conv_decode.cpp) on all `per` columns at once, with the 16-bit-DEST arithmetic
// (RNE to BF16, 0*x = 0): lhs -> DEST [0, per), rhs -> DEST [per, 2 per).  The srcA unpack config must already be
// the operands' (both mul operands of every tail mul are UnpackToDestFp32 CBs).
FORCE_INLINE void mul_cols(uint32_t lhs, uint32_t rhs, uint32_t out, uint32_t out2) {
    static_assert(2 * per <= 4, "FP32 half-sync DEST holds 4 tiles");
    mul_binary_tile_init();
    cb_wait_front(lhs, per);
    cb_wait_front(rhs, per);
    cb_reserve_back(out, per);
    if (out2 != out) {
        cb_reserve_back(out2, per);
    }
    tile_regs_acquire();
    copy_tile_init(lhs);
    for (uint32_t i = 0; i < per; ++i) {
        copy_tile(lhs, i, i);
    }
    copy_tile_init(rhs);
    for (uint32_t i = 0; i < per; ++i) {
        copy_tile(rhs, i, per + i);
    }
    for (uint32_t i = 0; i < per; ++i) {
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
                               (APPROX, ckernel::BinaryOp::MUL, 8, false), i, per + i, i, kVec)));
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < per; ++i) {
        pack_tile(i, out);
    }
    if (out2 != out) {
        for (uint32_t i = 0; i < per; ++i) {
            pack_tile(i, out2);
        }
    }
    tile_regs_release();
    cb_push_back(out, per);
    if (out2 != out) {
        cb_push_back(out2, per);
    }
    cb_pop_front(lhs, per);
    cb_pop_front(rhs, per);
}

// ttnn.addcmul(a, b, c) (value 1.0, conv_decode.cpp), 16-bit-DEST arithmetic (RNE to BF16), one column per DEST
// pass (3 tiles).  srcA unpack config: the (Default-mode) operands'.
FORCE_INLINE void addcmul_cols(uint32_t a, uint32_t b, uint32_t c, uint32_t out) {
    for (uint32_t i = 0; i < per; ++i) {
        cb_wait_front(a, 1);
        cb_wait_front(b, 1);
        cb_wait_front(c, 1);
        cb_reserve_back(out, 1);
        tile_regs_acquire();
        copy_tile_init(a);
        copy_tile(a, 0, 0);
        copy_tile(b, 0, 1);
        copy_tile(c, 0, 2);
        addcmul_tile_init();
        MATH((SFPU_TERNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, cf_addcmul_bf16, (APPROX, 8), 0, 1, 2, 0,
                                kVec, 0x3f800000u)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out);
        tile_regs_release();
        cb_push_back(out, 1);
        cb_pop_front(a, 1);
        cb_pop_front(b, 1);
        cb_pop_front(c, 1);
    }
}
}  // namespace cf
#endif

void kernel_main() {
    using namespace cf;
#if defined(COMPILE_FOR_NCRISC)
    // common: [w_in, w_out]; per core: [bank, in_run, in_ncols, out_run, out_ncols] (runs: bank-relative bytes)
    constexpr uint32_t tile_bytes = get_named_compile_time_arg_val("wt_bytes");
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t outstanding = get_named_compile_time_arg_val("outstanding");
    static_assert(outstanding >= 1 && outstanding < slots && outstanding <= 8);
    constexpr uint32_t slot_bytes = blk_in * tile_bytes;
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
            bytes = kb_in * (s == nsub_in - 1 ? rem_in : w_in) * tile_bytes;
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
            bytes = kb_out * (s == nsub_out - 1 ? rem_out : w_out) * tile_bytes;
            dst = out_base + o * blk_out * tile_bytes;
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
    // common: [x, out, taps, s0, s1, mcast sx, sy, ex, ey (whole rectangle, NOC1 order), mcast dests (rectangle - 1),
    //          n_workers, hub x, hub y]
    // per core: [idx, first_in_col, in_ncols (0: hub only), first_out_col, out_ncols, hub role (0 / 1),
    //            (owner x, owner y) per in column]
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    [[maybe_unused]] const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), kTile);  // !CF_NORM
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), kTile);
    const auto taps = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2),
                                     kTile);
    const auto s0 = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3),
                                   kTile);
    const auto s1 = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4),
                                   kTile);
    const uint32_t n_workers = get_common_arg_val<uint32_t>(10);
    const uint32_t hub_x = get_common_arg_val<uint32_t>(11), hub_y = get_common_arg_val<uint32_t>(12);
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
        const uint32_t sx = get_common_arg_val<uint32_t>(5), sy = get_common_arg_val<uint32_t>(6);
        const uint32_t ex = get_common_arg_val<uint32_t>(7), ey = get_common_arg_val<uint32_t>(8);
        const uint32_t n_dests = get_common_arg_val<uint32_t>(9);
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

    // 1. in_proj in0: x rows, or (CF_NORM) n = rms_norm(a + b) * gamma from the norm prologue (also writes h)
    {
        CF_ZONE("b1-x");
#if CF_NORM
        np_dm<out_args.next_compile_time_args_offset(), 13 + CF_VERIFY>();
#else
        cb_reserve_back(cb_x, kt);
        const uint32_t in0 = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < kt; ++t) {
            read_rows(get_noc_addr(t, x), in0 + t * kTile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, kt);
#endif
    }

    // 2. taps and state rows of the tail columns j = idx * per + i (this core owns these state columns; their
    //    in-place writes come in step 5, after every read of them).  CF_VERIFY: the state rows (row 0) are pushed in
    //    step 4b, after the rows above them are filled in from this step's Bx.
    {
    CF_ZONE("b2-state");
    for (uint32_t cb = kT2; cb <= kT0; ++cb) {
        cb_reserve_back(cb, per);
    }
    for (uint32_t i = 0; i < per; ++i) {
        const uint32_t j = idx * per + i;
        const uint32_t o = i * kTile;
        read_rows(get_noc_addr(2 * ht + j, taps), get_write_ptr(kT2) + o);
        read_rows(get_noc_addr(ht + j, taps), get_write_ptr(kT1) + o);
        read_rows(get_noc_addr(j, taps), get_write_ptr(kT0) + o);
        read_rows(get_noc_addr(j, s1), get_write_ptr(kS1) + o);
        read_rows(get_noc_addr(j, s0), get_write_ptr(kS0) + o);
    }
    noc_async_read_barrier();
    for (uint32_t cb = kT2; cb <= kT0; ++cb) {
        if (!CF_VERIFY || (cb != kS1 && cb != kS0)) {
            cb_push_back(cb, per);
        }
    }
    }

    // 3. bcx tiles -> owners (B / C / x slot of hidden column j = t % ht on core j / per)
    {
        CF_ZONE("b3-bcx");
        const uint32_t nsub = (in_ncols + w_in - 1) / w_in;
        const uint32_t rem = in_ncols - (nsub - 1) * w_in;
        uint32_t c = 0;
        for (uint32_t s = 0; s < nsub; ++s) {
            const uint32_t wid = s == nsub - 1 ? rem : w_in;
            cb_wait_front(cb_bcx, w_in);
            const uint32_t src = get_read_ptr(cb_bcx);
            for (uint32_t jj = 0; jj < wid; ++jj) {
                const uint32_t t = first_in + c + jj;
                const uint32_t j = t % ht;
                const uint32_t dst = recv[t / ht] + (j % per) * kTile;
                const uint32_t ox = get_arg_val<uint32_t>(6 + 2 * (c + jj)), oy = get_arg_val<uint32_t>(7 + 2 * (c + jj));
                write_rows(src + jj * kTile, get_noc_addr(ox, oy, dst));
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

    // 4. own B / C / x tiles -> tail
    {
        CF_ZONE("b4-wait");
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_bcx);
        noc_semaphore_wait(sem, 3 * per);
        noc_semaphore_set(sem, 0);
        for (uint32_t cb = kB; cb <= kC; ++cb) {
            cb_reserve_back(cb, per);
            cb_push_back(cb, per);
        }
    }

#if CF_VERIFY
    // 4b. per-row left context from this step's Bx rows (TRISC's first tail op): s1_r = Bx_{r-1}, s0_r = Bx_{r-2};
    //     rows 0 / 1 keep the state (s0_0 = s0, s1_0 = s0_1 = s1).  Face rows r < 16: 32 B at f * 512 + r * 32.
    {
        CF_ZONE("b4-shift");
        static_assert(rows <= 16, "verify rows live in faces 0 / 1");
        cb_wait_front(kBxOut, per);
        for (uint32_t i = 0; i < per; ++i) {
            const uint32_t s1t = get_write_ptr(kS1) + i * kTile, s0t = get_write_ptr(kS0) + i * kTile;
            const uint32_t bxt = get_read_ptr(kBxOut) + i * kTile;
            for (uint32_t f = 0; f < 2; ++f) {
                auto row = [&](uint32_t tile, uint32_t r) {
                    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + f * 512 + r * 32);
                };
                for (uint32_t r = rows - 1; r >= 1; --r) {
                    auto* d1 = row(s1t, r);
                    auto* b1 = row(bxt, r - 1);
                    auto* d0 = row(s0t, r);
                    auto* b0 = r >= 2 ? row(bxt, r - 2) : row(s1t, 0);
                    for (uint32_t w = 0; w < 8; ++w) {
                        d1[w] = b1[w];
                        d0[w] = b0[w];
                    }
                }
            }
        }
        cb_push_back(kS1, per);
        cb_push_back(kS0, per);
    }
#endif

    // 5. y rows -> the hub's cb_y slot j, then the hub's gather semaphore (the barrier sees only L1 writes: DRAM
    //    write acks are slow while the weight streams saturate DRAM); then the state shift s1 -> s0, Bx -> s1
    {
        CF_ZONE("b5-y");
        cb_wait_front(kY, per);
        for (uint32_t i = 0; i < per; ++i) {
            const uint32_t src = get_read_ptr(kY) + i * kTile;
            const uint64_t dst = get_noc_addr(hub_x, hub_y, y_base + (idx * per + i) * kSmall);
            chunks([&](uint32_t off, uint32_t n) { noc_async_write(src + off, dst + small_off(off), n); });
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, sem_gather), 1);
        cb_wait_front(kBxOut, per);
#if CF_VERIFY
        // every row's Bx -> the candidate tensor (the state is committed later for the accepted row)
        const auto cand = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                         get_common_arg_val<uint32_t>(13), kTile);
        for (uint32_t i = 0; i < per; ++i) {
            write_rows(get_read_ptr(kBxOut) + i * kTile, get_noc_addr(idx * per + i, cand));
        }
#else
        if constexpr (keep_row) {
            // An odd row count < 16: state row `rows` (another slot of a wider state, not part of this step) rides
            // along in the 64 B write granules; write its old values back (kY is in: the tail read kS0 / kS1).
            for (uint32_t i = 0; i < per; ++i) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t off = i * kTile + f * 512 + rows * 32;
                    auto* bx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(kBxOut) + off);
                    auto* o1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(kS1) + off);
                    auto* o0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(kS0) + off);
                    for (uint32_t w = 0; w < 8; ++w) {
                        bx[w] = o1[w];
                        o1[w] = o0[w];
                    }
                }
            }
        }
        for (uint32_t i = 0; i < per; ++i) {
            const uint32_t j = idx * per + i;
            write_rows(get_read_ptr(kS1) + i * kTile, get_noc_addr(j, s0));  // old s1 rows (kS1 is never refilled)
            write_rows(get_read_ptr(kBxOut) + i * kTile, get_noc_addr(j, s1));
        }
#endif
    }

    // 6. the hub's multicast of all y tiles landed -> out_proj
    if (is_hub) {
        run_hub();
    } else {
        CF_ZONE("b6-wait");
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready);
        noc_semaphore_wait(sem, 1);
        noc_semaphore_set(sem, 0);
    }
    cb_reserve_back(cb_y, kt);
    cb_push_back(cb_y, kt);

    // 7. out rows
    {
        CF_ZONE("b7-out");
        const uint32_t nsub = (out_ncols + w_out - 1) / w_out;
        const uint32_t rem = out_ncols - (nsub - 1) * w_out;
        uint32_t col = first_out;
        for (uint32_t s = 0; s < nsub; ++s) {
            const uint32_t wid = s == nsub - 1 ? rem : w_out;
            cb_wait_front(cb_out, w_out);
            const uint32_t src = get_read_ptr(cb_out);
            for (uint32_t jj = 0; jj < wid; ++jj) {
                const uint32_t t = src + jj * kSmall;
                const uint64_t dst = get_noc_addr(col + jj, out);
                chunks([&](uint32_t off, uint32_t n) { noc_async_write(t + small_off(off), dst + off, n); });
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, w_out);
            col += wid;
        }
    }
    noc_async_write_barrier();
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
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_win, cb_bcx);
    {
        CF_ZONE("t-in");
        matmul_pass<cb_x, cb_win, cb_bcx, w_in, kb_in>(in_ncols);
    }
    {
        // Grouped by unpack mode (every mul operand is an UnpackToDestFp32 CB, every addcmul operand a Default
        // one): the unconditional srcA reconfigs are needed because FP32 DEST gives the two groups different
        // unpack configs.
        CF_ZONE("t-tail");
        reconfig_data_format_srca(kB);
        mul_cols(kB, kX, kBx, kBxOut);  // Bx = B * x (-> new s1)
        mul_cols(kBx, kT2, kAcc, kAcc);  // acc = Bx * t2
        reconfig_data_format_srca(kAcc);
        addcmul_cols(kAcc, kS1, kT1, kAcc2);   // acc += s1 * t1
        addcmul_cols(kAcc2, kS0, kT0, kAcc3);  // acc += s0 * t0
        reconfig_data_format_srca(kC);
        mul_cols(kC, kAcc3, kY, kY);  // y = C * acc
    }
    {
        CF_ZONE("t-out");
        reconfig_data_format_srca(cb_wout);
        reconfig_data_format_srcb<true>(cb_y);    // th x 32 in0 tiles
        pack_reconfig_data_format<true>(cb_out);  // th x 32 out tiles
        matmul_pass<cb_y, cb_wout, cb_out, w_out, kb_out>(out_ncols);
    }
#endif
}
