// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling-2-Mini exact router on the SFPU (TRISC math thread): logits = x . W_r in IEEE FP32 multiply-adds.
//
// The FPU matmul accumulates at ~11-bit relative precision; the SFPU MAD is FP32 and the BF16 x BF16 products are
// exact, so the logits match an FP32 reference up to summation-order rounding (~1e-7 relative).
//
// Router weight layout (moe.py `router_layout`): W_r^T [H, E] row-major BF16 cut into 1024-element "tiles" of 8 k
// rows x 128 experts.  Unpacked into DEST, face f of such a tile holds k = 8g + 2f in rows 0-7 and k = 8g + 2f + 1 in
// rows 8-15, expert e at (row e / 16, column e % 16).  An SFPU slot (dst_reg[n], 32 lanes) covers a quarter of
// a face's rows 0-7 or 8-15, so slots 8f + 4h + q (q = 0..3) are the 128 experts of k = 8g + 2f + h, and slot
// 8f + 4h + q and slot q of another tile hold the same experts lane for lane.  The mapping of lanes to experts never
// has to be known: accumulators live in slots q of tile 0 face 0, which pack back to L1 as a [., 128] row in expert
// order (rows 0-7 of face 0 = 512 B), the same order as the weights.
//
// DEST (FP32, tile = 32 slots): accumulators at slot base ACC (decode: tile 0 = the core's partial; prefill: tile 3 =
// the current k block, flushed into tile 0), row r of the block in slots ACC + 4r .. + 3; tile 1 = the current weight
// tile.  Summation order (decode == prefill, bitwise): partial_b = sum over k in block b, sequential in k; logits =
// sum over blocks b in order (blocks of 8 * router_tiles k, the decode router cores' slices).
#pragma once

#include "ckernel_sfpu_silu.h"  // _sfpu_sigmoid_ (FP32: accurate exp + 2 Newton reciprocal steps)

namespace ckernel::sfpu {

// acc[ACC + 4r + q] = 0 for the block's ROWS rows.
template <uint32_t ROWS, uint32_t ACC>
inline void ink_router_zero() {
#pragma GCC unroll 32
    for (uint32_t s = 0; s < 4 * ROWS; ++s) {
        sfpi::dst_reg[ACC + s] = 0.0f;
    }
}

// Prefill: logits (tile 0) += block partial (tile 3); block partial = 0.
template <uint32_t ROWS>
inline void ink_router_flush() {
#pragma GCC unroll 32
    for (uint32_t s = 0; s < 4 * ROWS; ++s) {
        const sfpi::vFloat a = sfpi::dst_reg[s];
        sfpi::dst_reg[s] = a + sfpi::dst_reg[96 + s];
        sfpi::dst_reg[96 + s] = 0.0f;
    }
}

// acc(row r) += sum over the 8 k of weight tile 1 of W[k] * x_r[k]; x_r = 8 BF16 values at x_addr + r * row_bytes.
// The BF16 x values become SFPU scalars through SFPLOADI immediates (exact).
template <uint32_t ROWS, uint32_t ACC>
inline void ink_router_mac(const uint32_t x_addr, const uint32_t row_bytes) {
#pragma GCC unroll 8
    for (uint32_t r = 0; r < ROWS; ++r) {
        const volatile uint32_t* xp = reinterpret_cast<volatile uint32_t*>(x_addr + r * row_bytes);
        sfpi::vFloat a0 = sfpi::dst_reg[ACC + 4 * r + 0];
        sfpi::vFloat a1 = sfpi::dst_reg[ACC + 4 * r + 1];
        sfpi::vFloat a2 = sfpi::dst_reg[ACC + 4 * r + 2];
        sfpi::vFloat a3 = sfpi::dst_reg[ACC + 4 * r + 3];
#pragma GCC unroll 4
        for (uint32_t p = 0; p < 4; ++p) {
            const uint32_t two = xp[p];  // x[2p] (low half), x[2p + 1] (high half)
            {
                const sfpi::vFloat xs = sfpi::vFloat(sfpi::sFloat16b(two & 0xffffu));
                a0 += sfpi::dst_reg[32 + 8 * p + 0] * xs;
                a1 += sfpi::dst_reg[32 + 8 * p + 1] * xs;
                a2 += sfpi::dst_reg[32 + 8 * p + 2] * xs;
                a3 += sfpi::dst_reg[32 + 8 * p + 3] * xs;
            }
            {
                const sfpi::vFloat xs = sfpi::vFloat(sfpi::sFloat16b(two >> 16));
                a0 += sfpi::dst_reg[32 + 8 * p + 4] * xs;
                a1 += sfpi::dst_reg[32 + 8 * p + 5] * xs;
                a2 += sfpi::dst_reg[32 + 8 * p + 6] * xs;
                a3 += sfpi::dst_reg[32 + 8 * p + 7] * xs;
            }
        }
        sfpi::dst_reg[ACC + 4 * r + 0] = a0;
        sfpi::dst_reg[ACC + 4 * r + 1] = a1;
        sfpi::dst_reg[ACC + 4 * r + 2] = a2;
        sfpi::dst_reg[ACC + 4 * r + 3] = a3;
    }
}

// Prefill tail: tile 0 slots s < 4 ROWS (logits) -> scores = sigmoid (in place), tile 1 slot s = scores + bias with
// the FP32 bias replicated over tile 2 (every 512 B of it = bias[128], so slot s of tile 2 is the bias of slot s).
template <uint32_t ROWS>
inline void ink_router_finish_rows() {
#pragma GCC unroll 1
    for (uint32_t s = 0; s < 4 * ROWS; ++s) {
        const sfpi::vFloat sc = _sfpu_sigmoid_<true>(sfpi::dst_reg[0]);
        sfpi::dst_reg[0] = sc;
        sfpi::dst_reg[32] = sc + sfpi::dst_reg[64];
        sfpi::dst_reg++;
    }
}

// Decode tail: gathered partials p (p < P) at 512 B each from tile 0 (slot 32 (p / 8) + 8 ((p % 8) / 2) + 4 (p % 2)
// + q), the FP32 bias at partial index P.  logits = sum_p partial_p (fixed order), then tile 0 slots 0-3 = sel =
// sigmoid(logits) + bias (face 0 rows 0-7), slots 4-7 = scores (rows 8-15): the packed tile starts with sel[E],
// scores[E].
template <uint32_t P>
inline void ink_router_finish_partials() {
    constexpr uint32_t bias_slot = 32 * (P / 8) + 8 * ((P % 8) / 2) + 4 * (P % 2);
#pragma GCC unroll 4
    for (uint32_t q = 0; q < 4; ++q) {
        sfpi::vFloat s = sfpi::dst_reg[q];
#pragma GCC unroll 32
        for (uint32_t p = 1; p < P; ++p) {
            s += sfpi::dst_reg[32 * (p / 8) + 8 * ((p % 8) / 2) + 4 * (p % 2) + q];
        }
        sfpi::dst_reg[q] = s;
    }
#pragma GCC unroll 1
    for (uint32_t q = 0; q < 4; ++q) {
        const sfpi::vFloat sc = _sfpu_sigmoid_<true>(sfpi::dst_reg[0]);
        sfpi::dst_reg[4] = sc;
        sfpi::dst_reg[0] = sc + sfpi::dst_reg[bias_slot];
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
