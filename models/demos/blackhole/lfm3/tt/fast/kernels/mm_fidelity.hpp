// SPDX-License-Identifier: Apache-2.0
// matmul_block_init / matmul_block (api/compute/matmul.h) with an explicit math fidelity F instead of the kernel's
// MATH_FIDELITY: kernels whose compute config is HiFi4 for a norm prologue keep the stock decode matmul fidelity.
// Compute (TRISC) only; include after api/compute/matmul.h.
#pragma once

#if !defined(COMPILE_FOR_NCRISC) && !defined(COMPILE_FOR_BRISC)
namespace ckernel {
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
