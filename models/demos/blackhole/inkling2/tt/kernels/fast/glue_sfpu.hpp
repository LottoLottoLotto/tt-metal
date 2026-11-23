// SPDX-License-Identifier: Apache-2.0
// Custom SFPU helpers of the fused decode glue (glue_step.cpp), FP32 DEST.
//
// Row-limited ops: the glue's vectors live in tile rows 0..7 (the token row, the <= 3 query / gate rows, the
// hi | lo query halves at rows 4..6), so the elementwise ops run on rows 0..7 of faces 0 and 1 only
// (VectorMode::R, 4 iterations of 4 rows x 8 columns per face) instead of the whole tile.
//
//   gl_add / gl_sub / gl_mul (a, b, o)   o = a op b                     (FP32)
//   gl_split (x, hi, lo)                 hi = x with the low 16 bits cleared (exact in BF16), lo = x - hi (exact)
//   gl_rne (x)                           x = round-to-nearest-even BF16 value of x (packs exactly to BF16)
//   gl_rne_full (x)                      the same on the whole tile (32 rows)
//   gl_affine<A, B> (x)                  x = x * A + B (A, B: FP32 bit patterns)
//   gl_affine_to<A, B> (x, o)            o = x * A + B
//   gl_clamp<C> (x)                      x = min(max(x, -C), C) (C: FP32 bit pattern)
//   gl_split3 (x, hi, mid, lo)           x = hi + mid + lo, each exact in BF16 but lo (rounded on pack)
//   gl_clean<R> (x)                      rows >= R of the whole tile x set to 0 (also overwrites NaN / Inf)
//   gl_exp / gl_rsqrt / gl_recip / gl_tanh (x)   the library FP32 functions on rows 0..7
#pragma once

#include "api/compute/common_globals.h"
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

constexpr uint32_t kGlueTile = 32;  // sfpi rows per DEST tile
constexpr int kGlueIt = 4;          // rows 0..7 of a face

template <int OP, int IT = kGlueIt>
inline void glue_binary(const uint a, const uint b, const uint o) {
#pragma GCC unroll 8
    for (int d = 0; d < IT; d++) {
        sfpi::vFloat x = sfpi::dst_reg[a * kGlueTile];
        sfpi::vFloat y = sfpi::dst_reg[b * kGlueTile];
        sfpi::vFloat r;
        if constexpr (OP == 0) {
            r = x + y;
        } else if constexpr (OP == 1) {
            r = x - y;
        } else {
            r = x * y;
        }
        sfpi::dst_reg[o * kGlueTile] = r;
        sfpi::dst_reg++;
    }
}

template <int IT = kGlueIt>
inline void glue_split(const uint x, const uint hi, const uint lo) {
#pragma GCC unroll 8
    for (int d = 0; d < IT; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * kGlueTile];
        sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(v) & 0xFFFF0000U);
        sfpi::dst_reg[hi * kGlueTile] = h;
        sfpi::dst_reg[lo * kGlueTile] = v - h;
        sfpi::dst_reg++;
    }
}

inline void glue_split3(const uint x, const uint hi, const uint mid, const uint lo) {
#pragma GCC unroll 4
    for (int d = 0; d < kGlueIt; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * kGlueTile];
        sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(v) & 0xFFFF0000U);
        sfpi::vFloat r = v - h;
        sfpi::vFloat m = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(r) & 0xFFFF0000U);
        sfpi::dst_reg[hi * kGlueTile] = h;
        sfpi::dst_reg[mid * kGlueTile] = m;
        sfpi::dst_reg[lo * kGlueTile] = r - m;
        sfpi::dst_reg++;
    }
}

template <uint32_t C>
inline void glue_clamp(const uint x, const uint, const uint) {
    const sfpi::vFloat hi = __builtin_bit_cast(float, C);
    const sfpi::vFloat lo = -hi;
#pragma GCC unroll 4
    for (int d = 0; d < kGlueIt; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * kGlueTile];
        v_if(v > hi) { v = hi; }
        v_endif;
        v_if(v < lo) { v = lo; }
        v_endif;
        sfpi::dst_reg[x * kGlueTile] = v;
        sfpi::dst_reg++;
    }
}

template <int IT>
inline void glue_rne(const uint x, const uint, const uint) {
#pragma GCC unroll 8
    for (int d = 0; d < IT; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * kGlueTile];
        sfpi::dst_reg[x * kGlueTile] = float32_to_bf16_rne(v);
        sfpi::dst_reg++;
    }
}

template <uint32_t A, uint32_t B, int IT = kGlueIt>
inline void glue_affine(const uint x, const uint o, const uint) {
    const sfpi::vFloat a = __builtin_bit_cast(float, A);
    const sfpi::vFloat b = __builtin_bit_cast(float, B);
#pragma GCC unroll 8
    for (int d = 0; d < IT; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * kGlueTile];
        sfpi::dst_reg[o * kGlueTile] = v * a + b;
        sfpi::dst_reg++;
    }
}

// Whole tile (VectorMode::None: no per-face address increments; sfpi row f * 8 + i of a tile holds tile rows
// (f / 2) * 16 + (i / 2) * 4 .. +3, columns (f % 2) * 16 + (i % 2) * 8 .. +7; lane l = row-in-group * 8 + column,
// vConstTileId = 2 l).
template <uint32_t R>
inline void glue_clean(const uint x, const uint, const uint) {
#pragma GCC unroll 32
    for (uint32_t s = 0; s < 32; s++) {
        const uint32_t f = s >> 3, i = s & 7;
        const uint32_t row0 = (f >> 1) * 16 + (i >> 1) * 4;
        if (row0 >= R) {
            sfpi::dst_reg[x * kGlueTile + s] = 0.0f;
        } else if (row0 + 4 > R) {
            v_if(sfpi::vConstTileId >= (int)(16 * (R - row0))) { sfpi::dst_reg[x * kGlueTile + s] = 0.0f; }
            v_endif;
        }
    }
}

// Conv prologue on row-packed FP32 tiles (row j < T = column tile j; rows 4 g + j = row group g) in DEST tiles
// 0 x, 1 the 3 conv states (group s), 2 the 4 taps (group i), 3 base:  c = rne(x t3 + s0 t0 + s1 t1 + s2 t2) (every
// product and sum rounded separately, through DEST tile 7: the column-tile version's op order);
// MODE 1: 4 = h = rne(base + c); MODE 2: 4 = r = c; MODE 3: 4 = r = base + c; MODE 2 / 3: 5 / 6 = hi / lo of r.
// The outputs are written whole: rows j < T computed, every other row zero.
template <uint32_t MODE, uint32_t T>
inline void glue_conv(const uint, const uint, const uint) {
    constexpr uint32_t X = 0, S = 32, W = 64, B = 96, O = 128, HI = 160, LO = 192, SC = 224;
#pragma GCC unroll 32
    for (uint32_t s = 0; s < 32; s++) {
        if (s >= 16 || (s & 7) >= 2) {
            sfpi::dst_reg[O + s] = 0.0f;
            if constexpr (MODE != 1) {
                sfpi::dst_reg[HI + s] = 0.0f;
                sfpi::dst_reg[LO + s] = 0.0f;
            }
        }
    }
#pragma GCC unroll 4
    for (uint32_t q = 0; q < 4; q++) {
        const uint32_t b = (q >> 1) * 8 + (q & 1);  // sfpi rows 0, 1 (face 0), 8, 9 (face 1): tile rows 0..3
        {
            sfpi::vFloat x = sfpi::dst_reg[X + b], t = sfpi::dst_reg[W + b + 6];
            sfpi::dst_reg[O + b] = x * t;
        }
#pragma GCC unroll 3
        for (uint32_t st = 0; st < 3; st++) {
            sfpi::vFloat s0 = sfpi::dst_reg[S + b + 2 * st], t = sfpi::dst_reg[W + b + 2 * st];
            sfpi::dst_reg[SC + b] = s0 * t;
            sfpi::vFloat acc = sfpi::dst_reg[O + b], p = sfpi::dst_reg[SC + b];
            sfpi::dst_reg[O + b] = acc + p;
        }
        sfpi::vFloat c = float32_to_bf16_rne(sfpi::dst_reg[O + b]);
        sfpi::vFloat base = sfpi::dst_reg[B + b];
        sfpi::vFloat r;
        if constexpr (MODE == 1) {
            r = float32_to_bf16_rne(base + c);
        } else if constexpr (MODE == 2) {
            r = c;
        } else {
            r = base + c;
        }
        if constexpr (T < 4) {
            v_if(sfpi::vConstTileId >= (int)(16 * T)) { r = 0.0f; }
            v_endif;
        }
        sfpi::dst_reg[O + b] = r;
        if constexpr (MODE != 1) {
            sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(r) & 0xFFFF0000U);
            sfpi::dst_reg[HI + b] = h;
            sfpi::dst_reg[LO + b] = r - h;
        }
    }
}

// ---- row-packed (RP) tiles: row 4 g + j = row group g (an output / gate / query), column tile j < T; rows 0..15 =
// faces 0 / 1.  sfpi rows cg in {0, 1, 8, 9} hold tile rows 0..3 (one per 8-column group); cg + 2 g: rows 4 g .. +3.

// d = RP of the T column tiles s0..: row 4 g + j = row g (< 4) of tile s0 + j (sub-vector transpose); rows >= 16 zero.
// d == s0 allowed (every column group is read before it is written).
template <uint32_t T>
inline void glue_to_rp(const uint s0, const uint d, const uint) {
#pragma GCC unroll 4
    for (uint32_t q = 0; q < 4; q++) {
        const uint32_t cg = (q >> 1) * 8 + (q & 1);
        sfpi::vFloat a = sfpi::dst_reg[s0 * kGlueTile + cg];
        sfpi::vFloat b = 0.0f, c = 0.0f, e = 0.0f;
        if constexpr (T > 1) {
            b = sfpi::dst_reg[(s0 + 1) * kGlueTile + cg];
        }
        if constexpr (T > 2) {
            c = sfpi::dst_reg[(s0 + 2) * kGlueTile + cg];
        }
        if constexpr (T > 3) {
            e = sfpi::dst_reg[(s0 + 3) * kGlueTile + cg];
        }
        sfpi::subvec_transp(a, b, c, e);
        sfpi::dst_reg[d * kGlueTile + cg] = a;
        sfpi::dst_reg[d * kGlueTile + cg + 2] = b;
        sfpi::dst_reg[d * kGlueTile + cg + 4] = c;
        sfpi::dst_reg[d * kGlueTile + cg + 6] = e;
    }
#pragma GCC unroll 16
    for (uint32_t r = 16; r < 32; r++) {
        sfpi::dst_reg[d * kGlueTile + r] = 0.0f;
    }
}

// column tiles d0..d0 + T - 1 from the RP tile s: row g of tile d0 + j = row 4 g + j; every other row zero.  s == d0
// allowed.
template <uint32_t T>
inline void glue_to_cols(const uint s, const uint d0, const uint) {
#pragma GCC unroll 4
    for (uint32_t q = 0; q < 4; q++) {
        const uint32_t cg = (q >> 1) * 8 + (q & 1);
        sfpi::vFloat a = sfpi::dst_reg[s * kGlueTile + cg], b = sfpi::dst_reg[s * kGlueTile + cg + 2];
        sfpi::vFloat c = sfpi::dst_reg[s * kGlueTile + cg + 4], e = sfpi::dst_reg[s * kGlueTile + cg + 6];
        sfpi::subvec_transp(a, b, c, e);
        sfpi::dst_reg[d0 * kGlueTile + cg] = a;
        if constexpr (T > 1) {
            sfpi::dst_reg[(d0 + 1) * kGlueTile + cg] = b;
        }
        if constexpr (T > 2) {
            sfpi::dst_reg[(d0 + 2) * kGlueTile + cg] = c;
        }
        if constexpr (T > 3) {
            sfpi::dst_reg[(d0 + 3) * kGlueTile + cg] = e;
        }
    }
#pragma GCC unroll 4
    for (uint32_t j = 0; j < T; j++) {
#pragma GCC unroll 32
        for (uint32_t r = 0; r < 32; r++) {
            if (r >= 16 || (r & 7) >= 2) {
                sfpi::dst_reg[(d0 + j) * kGlueTile + r] = 0.0f;
            }
        }
    }
}

// rows 4 g + j of d = row g (< 4) of x for every j (row g broadcast over its group); rows >= 16 zero.  d == x allowed.
inline void glue_groups(const uint x, const uint d, const uint) {
#pragma GCC unroll 4
    for (uint32_t q = 0; q < 4; q++) {
        const uint32_t cg = (q >> 1) * 8 + (q & 1);
        sfpi::vFloat a = sfpi::dst_reg[x * kGlueTile + cg];
        sfpi::vFloat b = a, c = a, e = a;
        sfpi::subvec_transp(a, b, c, e);
        sfpi::dst_reg[d * kGlueTile + cg] = a;
        sfpi::dst_reg[d * kGlueTile + cg + 2] = b;
        sfpi::dst_reg[d * kGlueTile + cg + 4] = c;
        sfpi::dst_reg[d * kGlueTile + cg + 6] = e;
    }
#pragma GCC unroll 16
    for (uint32_t r = 16; r < 32; r++) {
        sfpi::dst_reg[d * kGlueTile + r] = 0.0f;
    }
}

// x = 0 (every row)
inline void glue_zero(const uint x, const uint, const uint) {
#pragma GCC unroll 32
    for (uint32_t r = 0; r < 32; r++) {
        sfpi::dst_reg[x * kGlueTile + r] = 0.0f;
    }
}

// d = x (every row)
inline void glue_copy(const uint x, const uint d, const uint) {
#pragma GCC unroll 32
    for (uint32_t r = 0; r < 32; r++) {
        sfpi::vFloat v = sfpi::dst_reg[x * kGlueTile + r];
        sfpi::dst_reg[d * kGlueTile + r] = v;
    }
}

// RP row group 0 of x copied to groups 1..NG-1
template <uint32_t NG>
inline void glue_rep(const uint x, const uint, const uint) {
#pragma GCC unroll 4
    for (uint32_t q = 0; q < 4; q++) {
        const uint32_t cg = (q >> 1) * 8 + (q & 1);
        sfpi::vFloat a = sfpi::dst_reg[x * kGlueTile + cg];
#pragma GCC unroll 3
        for (uint32_t g = 1; g < NG; g++) {
            sfpi::dst_reg[x * kGlueTile + cg + 2 * g] = a;
        }
    }
}

// RP tile x: rows 4 g + j with g >= NR or j >= T, and rows >= 16, set to 0
template <uint32_t T, uint32_t NR>
inline void glue_clean_rp(const uint x, const uint, const uint) {
#pragma GCC unroll 32
    for (uint32_t r = 0; r < 32; r++) {
        if (r >= 16 || ((r & 7) >> 1) >= NR) {
            sfpi::dst_reg[x * kGlueTile + r] = 0.0f;
        } else if constexpr (T < 4) {
            v_if(sfpi::vConstTileId >= (int)(16 * T)) { sfpi::dst_reg[x * kGlueTile + r] = 0.0f; }
            v_endif;
        }
    }
}

// d = row K (< 4) of s, every other row zero
template <uint32_t K>
inline void glue_rowkeep(const uint s, const uint d, const uint) {
#pragma GCC unroll 32
    for (uint32_t r = 0; r < 32; r++) {
        if (r >= 16 || (r & 7) >= 2) {
            sfpi::dst_reg[d * kGlueTile + r] = 0.0f;
        } else {
            sfpi::vFloat v = sfpi::dst_reg[s * kGlueTile + r];
            v_if(sfpi::vConstTileId < (int)(16 * K) || sfpi::vConstTileId >= (int)(16 * (K + 1))) { v = 0.0f; }
            v_endif;
            sfpi::dst_reg[d * kGlueTile + r] = v;
        }
    }
}

// ---- fused row-packed steps (one SFPU function each: constant DEST indices, every intermediate rounded as the
// separate-op column-tile version, through DEST 7 where the compiler could otherwise fuse a multiply-add)

// gate inputs: DEST 0 = O (row-packed, group a = O_a), 3 = s_O (row-packed, every column); group k of the gate rows
// reads group GSAME ? 0 : k.  -> 4 / 5 = hi / lo of x_k, 0 = nx_k = rne(x_k s_O); rows of groups >= NG, j >= T, >= 16: 0
template <uint32_t T, uint32_t NG, bool GSAME>
inline void glue_g1(const uint, const uint, const uint) {
#pragma GCC unroll 4
    for (uint32_t q = 0; q < 4; q++) {
        const uint32_t cg = (q >> 1) * 8 + (q & 1);
#pragma GCC unroll 4
        for (uint32_t g = 0; g < 4; g++) {
            const uint32_t r = cg + 2 * g;
            if (g >= NG) {
                sfpi::dst_reg[r] = 0.0f;
                sfpi::dst_reg[4 * kGlueTile + r] = 0.0f;
                sfpi::dst_reg[5 * kGlueTile + r] = 0.0f;
                continue;
            }
            const uint32_t src = GSAME ? cg : r;
            sfpi::vFloat o = sfpi::dst_reg[src], so = sfpi::dst_reg[3 * kGlueTile + src];
            sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(o) & 0xFFFF0000U);
            sfpi::vFloat lo = o - h;
            sfpi::vFloat nx = float32_to_bf16_rne(o * so);
            if constexpr (T < 4) {
                v_if(sfpi::vConstTileId >= (int)(16 * T)) {
                    h = 0.0f;
                    lo = 0.0f;
                    nx = 0.0f;
                }
                v_endif;
            }
            sfpi::dst_reg[4 * kGlueTile + r] = h;
            sfpi::dst_reg[5 * kGlueTile + r] = lo;
            if (!GSAME || g == NG - 1) {  // GSAME: group 0 (the source) written last
                sfpi::dst_reg[r] = nx;
            }
            if constexpr (GSAME) {
                if (g == NG - 1) {
#pragma GCC unroll 3
                    for (uint32_t g2 = 0; g2 + 1 < NG; g2++) {
                        sfpi::dst_reg[cg + 2 * g2] = nx;
                    }
                }
            }
        }
    }
#pragma GCC unroll 16
    for (uint32_t r = 16; r < 32; r++) {
        sfpi::dst_reg[r] = 0.0f;
        sfpi::dst_reg[4 * kGlueTile + r] = 0.0f;
        sfpi::dst_reg[5 * kGlueTile + r] = 0.0f;
    }
}

// gate outputs: DEST 0 = tanh(g Wf), 3 = tanh(g Wg), 1 / 2 = x hi / lo, 6 = m (row-packed rows 4 k + j):
// y = (1 - f) x + gt m (each op rounded) -> 0 / 1 = y hi / lo; rows of groups >= NG, j >= T, >= 16: 0
template <uint32_t T, uint32_t NG>
inline void glue_gate_y(const uint, const uint, const uint) {
    constexpr uint32_t SC = 7 * kGlueTile;
#pragma GCC unroll 16
    for (uint32_t r = 0; r < 16; r++) {
        if (((r & 7) >> 1) >= NG) {
            sfpi::dst_reg[r] = 0.0f;
            sfpi::dst_reg[kGlueTile + r] = 0.0f;
            continue;
        }
        sfpi::vFloat f = sfpi::dst_reg[r], hi = sfpi::dst_reg[kGlueTile + r], lo = sfpi::dst_reg[2 * kGlueTile + r];
        sfpi::dst_reg[SC + r] = hi + lo;
        sfpi::dst_reg[SC + 16 + r] = f * -1.0f + 1.0f;
        sfpi::vFloat x = sfpi::dst_reg[SC + r], a = sfpi::dst_reg[SC + 16 + r];
        sfpi::dst_reg[SC + r] = a * x;
        sfpi::vFloat gt = sfpi::dst_reg[3 * kGlueTile + r], m = sfpi::dst_reg[6 * kGlueTile + r];
        sfpi::dst_reg[SC + 16 + r] = gt * m;
        sfpi::vFloat t1 = sfpi::dst_reg[SC + r], t2 = sfpi::dst_reg[SC + 16 + r];
        sfpi::vFloat y = t1 + t2;
        if constexpr (T < 4) {
            v_if(sfpi::vConstTileId >= (int)(16 * T)) { y = 0.0f; }
            v_endif;
        }
        sfpi::vFloat yh = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(y) & 0xFFFF0000U);
        sfpi::dst_reg[r] = yh;
        sfpi::dst_reg[kGlueTile + r] = y - yh;
    }
#pragma GCC unroll 16
    for (uint32_t r = 16; r < 32; r++) {
        sfpi::dst_reg[r] = 0.0f;
        sfpi::dst_reg[kGlueTile + r] = 0.0f;
    }
}

// outputs: DEST 0 = v (HILO: hi, DEST 1 = lo), 3 = s (row-packed, every column), 2 = gamma:
// 0 = rne(v s gamma) on rows of groups < NR, j < T; every other row 0
template <uint32_t T, uint32_t NR, bool HILO>
inline void glue_out(const uint, const uint, const uint) {
#pragma GCC unroll 16
    for (uint32_t r = 0; r < 16; r++) {
        if (((r & 7) >> 1) >= NR) {
            sfpi::dst_reg[r] = 0.0f;
            continue;
        }
        sfpi::vFloat v = sfpi::dst_reg[r];
        if constexpr (HILO) {
            sfpi::vFloat lo = sfpi::dst_reg[kGlueTile + r];
            v = v + lo;
        }
        sfpi::vFloat sv = sfpi::dst_reg[3 * kGlueTile + r];
        v = v * sv;
        sfpi::vFloat gm = sfpi::dst_reg[2 * kGlueTile + r];
        v = float32_to_bf16_rne(v * gm);
        if constexpr (T < 4) {
            v_if(sfpi::vConstTileId >= (int)(16 * T)) { v = 0.0f; }
            v_endif;
        }
        sfpi::dst_reg[r] = v;
    }
#pragma GCC unroll 16
    for (uint32_t r = 16; r < 32; r++) {
        sfpi::dst_reg[r] = 0.0f;
    }
}

// per-gate g rows: DEST 2 k + c2 = row k of DEST 6 + c2 (k < NG), every other row zero
template <uint32_t NG>
inline void glue_gk(const uint, const uint, const uint) {
#pragma GCC unroll 3
    for (uint32_t k = 0; k < NG; k++) {
#pragma GCC unroll 2
        for (uint32_t c2 = 0; c2 < 2; c2++) {
            const uint32_t d = (2 * k + c2) * kGlueTile, s = (6 + c2) * kGlueTile;
#pragma GCC unroll 32
            for (uint32_t r = 0; r < 32; r++) {
                if (r >= 16 || (r & 7) >= 2) {
                    sfpi::dst_reg[d + r] = 0.0f;
                } else {
                    sfpi::vFloat v = sfpi::dst_reg[s + r];
                    v_if(sfpi::vConstTileId < (int)(16 * k) || sfpi::vConstTileId >= (int)(16 * (k + 1))) { v = 0.0f; }
                    v_endif;
                    sfpi::dst_reg[d + r] = v;
                }
            }
        }
    }
}

// rows 0..3 of x: column 0 copied to every column (the 8 lanes of a row: masked, then rotate-and-add)
inline void glue_col0(const uint x, const uint, const uint) {
    sfpi::vFloat v = sfpi::dst_reg[x * kGlueTile];
    v_if((sfpi::vConstTileId & 15) != 0) { v = 0.0f; }
    v_endif;
    sfpi::vFloat t = sfpi::subvec_shflror1(v);
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    v = v + t;
    t = sfpi::subvec_shflror1(v);
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    t = sfpi::subvec_shflror1(t);
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    v = v + t;
    t = sfpi::subvec_shflror1(v);
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    t = sfpi::subvec_shflror1(t);
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    t = sfpi::subvec_shflror1(t);
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    t = sfpi::subvec_shflror1(t);
    __builtin_rvtt_sfpnop();
    __builtin_rvtt_sfpnop();
    v = v + t;
    sfpi::dst_reg[x * kGlueTile] = v;
    sfpi::dst_reg[x * kGlueTile + 1] = v;
    sfpi::dst_reg[x * kGlueTile + 8] = v;
    sfpi::dst_reg[x * kGlueTile + 9] = v;
}

}  // namespace ckernel::sfpu
#endif

namespace glue {
using namespace ckernel;

// the math unit waits until the SFPU is idle (an FPU copy into a DEST tile the SFPU still reads)
ALWI void sfpu_fence() { MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)); }

ALWI void sfpu_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, ::ckernel::sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::ADD))));
}

#define GLUE_BIN(FN, A, B, O, VM) \
    MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::FN, (A), (B), (O), (VM))))

ALWI void add(uint32_t a, uint32_t b, uint32_t o) { GLUE_BIN(glue_binary<0>, a, b, o, VectorMode::R); }
ALWI void sub(uint32_t a, uint32_t b, uint32_t o) { GLUE_BIN(glue_binary<1>, a, b, o, VectorMode::R); }
ALWI void mul(uint32_t a, uint32_t b, uint32_t o) { GLUE_BIN(glue_binary<2>, a, b, o, VectorMode::R); }
ALWI void split(uint32_t x, uint32_t hi, uint32_t lo) { GLUE_BIN(glue_split<>, x, hi, lo, VectorMode::R); }
// rows 0..15 (row-packed tiles)
ALWI void add16(uint32_t a, uint32_t b, uint32_t o) { MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::glue_binary<0, 8>, a, b, o, VectorMode::R))); }
ALWI void mul16(uint32_t a, uint32_t b, uint32_t o) { MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::glue_binary<2, 8>, a, b, o, VectorMode::R))); }
ALWI void split16(uint32_t x, uint32_t hi, uint32_t lo) { GLUE_BIN(glue_split<8>, x, hi, lo, VectorMode::R); }
ALWI void rne16(uint32_t x) { GLUE_BIN(glue_rne<8>, x, x, x, VectorMode::R); }
template <uint32_t A, uint32_t B>
ALWI void affine16(uint32_t x) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::glue_affine<A, B, 8>, x, x, x, VectorMode::R)));
}
#define GLUE_WHOLE(A, B, ...) \
    MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::__VA_ARGS__, (A), (B), 0, VectorMode::None)))
template <uint32_t T>
ALWI void to_rp(uint32_t s0, uint32_t d) { GLUE_WHOLE(s0, d, glue_to_rp<T>); }
template <uint32_t T>
ALWI void to_cols(uint32_t s, uint32_t d0) { GLUE_WHOLE(s, d0, glue_to_cols<T>); }
ALWI void groups(uint32_t x, uint32_t d) { GLUE_WHOLE(x, d, glue_groups); }
ALWI void copy16(uint32_t x, uint32_t d) { GLUE_WHOLE(x, d, glue_copy); }
ALWI void zero(uint32_t x) { GLUE_WHOLE(x, x, glue_zero); }
template <uint32_t T, uint32_t NG, bool GSAME>
ALWI void g1() { GLUE_WHOLE(0, 0, glue_g1<T, NG, GSAME>); }
template <uint32_t T, uint32_t NG>
ALWI void gate_y() { GLUE_WHOLE(0, 0, glue_gate_y<T, NG>); }
template <uint32_t T, uint32_t NR, bool HILO>
ALWI void out() { GLUE_WHOLE(0, 0, glue_out<T, NR, HILO>); }
ALWI void col0(uint32_t x) { GLUE_WHOLE(x, x, glue_col0); }
template <uint32_t NG>
ALWI void gk() { GLUE_WHOLE(0, 0, glue_gk<NG>); }
template <uint32_t NG>
ALWI void rep(uint32_t x) { GLUE_WHOLE(x, x, glue_rep<NG>); }
template <uint32_t T, uint32_t NR>
ALWI void clean_rp(uint32_t x) { GLUE_WHOLE(x, x, glue_clean_rp<T, NR>); }
template <uint32_t K>
ALWI void rowkeep(uint32_t s, uint32_t d) { GLUE_WHOLE(s, d, glue_rowkeep<K>); }
// x = hi + mid + lo, each exact in BF16 (lo rounded on pack: ~2^-25 of x) -- the cross-core transport of partials
ALWI void split3(uint32_t x, uint32_t hi, uint32_t mid, uint32_t lo) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::glue_split3, x, hi, mid, VectorMode::R, lo)));
}
ALWI void rne(uint32_t x) { GLUE_BIN(glue_rne<4>, x, x, x, VectorMode::R); }
ALWI void rne_full(uint32_t x) { GLUE_BIN(glue_rne<8>, x, x, x, VectorMode::RC); }
template <uint32_t A, uint32_t B>
ALWI void affine(uint32_t x) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::glue_affine<A, B>, x, x, x, VectorMode::R)));
}
template <uint32_t A, uint32_t B>
ALWI void affine_to(uint32_t x, uint32_t o) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::glue_affine<A, B>, x, o, x, VectorMode::R)));
}
template <uint32_t C>
ALWI void clamp(uint32_t x) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::glue_clamp<C>, x, x, x, VectorMode::R)));
}
template <uint32_t R>
ALWI void clean(uint32_t x) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::glue_clean<R>, x, x, x, VectorMode::None)));
}

// the conv prologue on row-packed tiles (DEST tiles 0..3 -> 4..6, scratch 7; ckernel::sfpu::glue_conv)
template <uint32_t MODE, uint32_t T>
ALWI void conv() {
    MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::glue_conv<MODE, T>, 0, 0, 0, VectorMode::None)));
}

ALWI void exp(uint32_t x) {
    MATH(SFPU_UNARY_INIT_FN(exponential, ::ckernel::sfpu::exp_init, (false, 0x3F800000, false, DST_ACCUM_MODE)));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_exponential, (false, DST_ACCUM_MODE, false, 4, false),
                         x, VectorMode::R, p_sfpu::kCONST_1_FP16B));
}
ALWI void rsqrt(uint32_t x) {
    MATH(SFPU_UNARY_INIT_FN(rsqrt, ::ckernel::sfpu::rsqrt_init, (false, false)));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (false, 4, DST_ACCUM_MODE, false, false), x,
                         VectorMode::R));
}
ALWI void recip(uint32_t x) {
    MATH(SFPU_UNARY_INIT_FN(reciprocal, ::ckernel::sfpu::recip_init, (false, DST_ACCUM_MODE, false)));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_reciprocal, (false, DST_ACCUM_MODE, 4, false), x,
                         VectorMode::R));
}
ALWI void tanh16(uint32_t x) {
    MATH(SFPU_UNARY_INIT_FN(tanh, ::ckernel::sfpu::tanh_init, (false, DST_ACCUM_MODE)));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh, (false, DST_ACCUM_MODE, 8), x, VectorMode::R));
}
ALWI void tanh(uint32_t x) {
    MATH(SFPU_UNARY_INIT_FN(tanh, ::ckernel::sfpu::tanh_init, (false, DST_ACCUM_MODE)));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh, (false, DST_ACCUM_MODE, 4), x, VectorMode::R));
}

}  // namespace glue
