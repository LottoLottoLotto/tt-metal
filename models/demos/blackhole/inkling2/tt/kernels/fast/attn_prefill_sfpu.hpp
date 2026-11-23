// SPDX-License-Identifier: Apache-2.0
// SFPU helpers of the attention prefill compute kernels (attn_prefill_compute.cpp, attn_prefill_kv_compute.cpp),
// FP32 DEST, full tiles.  They reproduce attention.py's former ttnn ops bit for bit:
//   pa::sigmoid16(x)   x = sigmoid(x) as ttnn computes it with a 16-bit DEST (the BF16 gate's LHS activation of
//                      ttnn.mul: exp_21f + one reciprocal iteration, rounded to BF16, RNE); init: sigmoid_tile_init()
//   pa::rne16(x)       x = BF16 round-to-nearest-even of x: every FP32 -> BF16 result of the former ttnn ops was RNE
//                      (exact ties included), the packer rounds ties away; init: pa::rne16_init()
//   pa::mul_col(x, c)  x[r][*] *= c[r][0] (the SFPU column broadcast multiply: a column [.., 1] FP32 factor times a
//                      tile, as ttnn's broadcast multiply)
#pragma once

#include "api/compute/common_globals.h"
#include "api/compute/sfpu_binary_bcast.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void pa_sigmoid16() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r = _sfpu_sigmoid_<false>(v);
        sfpi::dst_reg[0] = float32_to_bf16_rne(r);  // the BF16-DEST path's convert<vFloat16b>(Nearest)
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS = 8>
inline void pa_rne16() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = float32_to_bf16_rne(v);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

namespace pa {
using namespace ckernel;

ALWI void sigmoid16(uint32_t x) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, pa_sigmoid16, (8), x, VectorMode::RC));
}
ALWI void rne16_init() { sigmoid_tile_init(); }  // the generic unary SFPU setup (no state used by rne16)
ALWI void rne16(uint32_t x) { MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, pa_rne16, (8), x, VectorMode::RC)); }
ALWI void mul_col(uint32_t x, uint32_t c) {
    MATH((ckernel::sfpu_bcast_col_init()));
    ckernel::sfpu_mul_bcast_col(x, c);
}

}  // namespace pa
