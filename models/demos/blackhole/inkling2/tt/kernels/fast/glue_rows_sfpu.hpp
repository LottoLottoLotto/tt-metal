// SPDX-License-Identifier: Apache-2.0
// Full-tile SFPU helpers of the tokens-as-rows glue kernel (glue_rows.cpp), FP32 DEST.  Same ops as glue_sfpu.hpp
// (the decode kernel's row-limited helpers) but over all 32 rows (VectorMode::RC, 8 iterations per face):
//   add / sub / mul (a, b, o)   o = a op b (FP32)
//   split (x, hi, lo)           hi = x with the low 16 bits cleared (exact in BF16), lo = x - hi (hi may be x)
//   rne (x)                     x = round-to-nearest-even BF16 value of x
//   affine<A, B> (x)            x = x * A + B (FP32 bit patterns)
//   exp / rsqrt / recip         the library FP32 functions
//   mul_bcast_col (x, c)        x[r][*] *= c[r][0]
#pragma once

#include "api/compute/common_globals.h"
#include "api/compute/sfpu_binary_bcast.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary.h"
#include "ckernel_sfpu_conversions.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_rsqrt.h"
#include "ckernel_sfpu_tanh.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {

constexpr uint32_t kRowsTile = 32;  // sfpi rows per DEST tile
constexpr int kRowsIt = 8;          // sfpi rows per face

template <int OP>
inline void rows_binary(const uint a, const uint b, const uint o) {
#pragma GCC unroll 8
    for (int d = 0; d < kRowsIt; d++) {
        sfpi::vFloat x = sfpi::dst_reg[a * kRowsTile];
        sfpi::vFloat y = sfpi::dst_reg[b * kRowsTile];
        sfpi::vFloat r;
        if constexpr (OP == 0) {
            r = x + y;
        } else if constexpr (OP == 1) {
            r = x - y;
        } else {
            r = x * y;
        }
        sfpi::dst_reg[o * kRowsTile] = r;
        sfpi::dst_reg++;
    }
}

inline void rows_split(const uint x, const uint hi, const uint lo) {
#pragma GCC unroll 8
    for (int d = 0; d < kRowsIt; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * kRowsTile];
        sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(v) & 0xFFFF0000U);
        sfpi::dst_reg[hi * kRowsTile] = h;
        sfpi::dst_reg[lo * kRowsTile] = v - h;
        sfpi::dst_reg++;
    }
}

inline void rows_rne(const uint x, const uint, const uint) {
#pragma GCC unroll 8
    for (int d = 0; d < kRowsIt; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * kRowsTile];
        sfpi::dst_reg[x * kRowsTile] = float32_to_bf16_rne(v);
        sfpi::dst_reg++;
    }
}

template <uint32_t A, uint32_t B>
inline void rows_affine(const uint x, const uint, const uint) {
    const sfpi::vFloat a = __builtin_bit_cast(float, A);
    const sfpi::vFloat b = __builtin_bit_cast(float, B);
#pragma GCC unroll 8
    for (int d = 0; d < kRowsIt; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * kRowsTile];
        sfpi::dst_reg[x * kRowsTile] = v * a + b;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

namespace rows {
using namespace ckernel;

ALWI void sfpu_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, ::ckernel::sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::ADD))));
}

#define ROWS_BIN(FN, A, B, O) MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::FN, (A), (B), (O), VectorMode::RC)))

ALWI void add(uint32_t a, uint32_t b, uint32_t o) { ROWS_BIN(rows_binary<0>, a, b, o); }
ALWI void sub(uint32_t a, uint32_t b, uint32_t o) { ROWS_BIN(rows_binary<1>, a, b, o); }
ALWI void mul(uint32_t a, uint32_t b, uint32_t o) { ROWS_BIN(rows_binary<2>, a, b, o); }
ALWI void split(uint32_t x, uint32_t hi, uint32_t lo) { ROWS_BIN(rows_split, x, hi, lo); }
ALWI void rne(uint32_t x) { ROWS_BIN(rows_rne, x, x, x); }
template <uint32_t A, uint32_t B>
ALWI void affine(uint32_t x) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::rows_affine<A, B>, x, x, x, VectorMode::RC)));
}

ALWI void exp(uint32_t x) {
    MATH(SFPU_UNARY_INIT_FN(exponential, ::ckernel::sfpu::exp_init, (false, 0x3F800000, false, DST_ACCUM_MODE)));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_exponential, (false, DST_ACCUM_MODE, false, 8, false),
                         x, VectorMode::RC, p_sfpu::kCONST_1_FP16B));
}
ALWI void rsqrt(uint32_t x) {
    MATH(SFPU_UNARY_INIT_FN(rsqrt, ::ckernel::sfpu::rsqrt_init, (false, false)));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (false, 8, DST_ACCUM_MODE, false, false), x,
                         VectorMode::RC));
}
ALWI void recip(uint32_t x) {
    MATH(SFPU_UNARY_INIT_FN(reciprocal, ::ckernel::sfpu::recip_init, (false, DST_ACCUM_MODE, false)));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_reciprocal, (false, DST_ACCUM_MODE, 8, false), x,
                         VectorMode::RC));
}
ALWI void tanh(uint32_t x) {
    MATH(SFPU_UNARY_INIT_FN(tanh, ::ckernel::sfpu::tanh_init, (false, DST_ACCUM_MODE)));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh, (false, DST_ACCUM_MODE, 8), x, VectorMode::RC));
}
ALWI void mul_bcast_col(uint32_t x, uint32_t c) {
    MATH((ckernel::sfpu_bcast_col_init()));
    ckernel::sfpu_mul_bcast_col(x, c);
}

}  // namespace rows
