// SPDX-License-Identifier: Apache-2.0
// Fused residual add + RMSNorm compute (HiFi4, FP32 dest). Per core, `tiles` width tiles:
//   h = a + b                                       -> CB_H (BF16; also the residual output)
//   P = sum_i h_i h_i^T (matmul, FP32 acc)          -> CB_PART (FP32; diagonal P[r][r] = this core's sum of h^2
//                                                      over its columns for row r, products exact)
//   master: mean = reduce_row(CB_M) with scaler 1/W; rstd = rsqrt(mean + eps) (SFPU, column faces only)
//           -> CB_RSTD (FP32), multicast to the other cores by the writer
//   n = (h * rstd[col bcast]) * gamma               -> CB_N (BF16)
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

constexpr uint32_t CB_A = 0, CB_B = 1, CB_G = 2, CB_SCALER = 3, CB_H = 16, CB_N = 17, CB_PART = 25, CB_M = 26,
                   CB_RSTD = 28;

void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    const bool is_master = get_absolute_logical_x() == 0 && get_absolute_logical_y() == 0;

#ifndef RN_NO_RESIDUAL
    binary_op_init_common(CB_A, CB_B, CB_H);
    add_tiles_init(CB_A, CB_B);
    cb_wait_front(CB_A, tiles);
    cb_wait_front(CB_B, tiles);
    cb_reserve_back(CB_H, tiles);
    tile_regs_acquire();
    for (uint32_t i = 0; i < tiles; ++i) {
        add_tiles(CB_A, CB_B, i, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < tiles; ++i) {
        pack_tile(i, CB_H);
    }
    tile_regs_release();
    cb_push_back(CB_H, tiles);
#else
    binary_op_init_common(CB_H, CB_H, CB_PART);
#endif

    // Row sums of h^2 over this core's columns: diagonal of sum_i h_i h_i^T.
    pack_reconfig_data_format(CB_PART);
    matmul_init(CB_H, CB_H, 1);
    cb_wait_front(CB_H, tiles);
    tile_regs_acquire();
    for (uint32_t i = 0; i < tiles; ++i) {
        matmul_tiles(CB_H, CB_H, i, i, 0);
    }
    tile_regs_commit();
    cb_reserve_back(CB_PART, 1);
    tile_regs_wait();
    pack_tile(0, CB_PART);
    tile_regs_release();
    cb_push_back(CB_PART, 1);

    if (is_master) {
        // CB_M column k = core k's partials; reduce_row with the 1/W scaler -> mean (column 0).
        reconfig_data_format(CB_SCALER, CB_M);
        pack_reconfig_data_format(CB_RSTD);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_M, CB_SCALER, CB_RSTD);
        cb_wait_front(CB_SCALER, 1);
        cb_wait_front(CB_M, m_tiles);
        tile_regs_acquire();
        for (uint32_t j = 0; j < m_tiles; ++j) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_M, CB_SCALER, j, 0, 0);
        }
        reduce_uninit();
        // + eps, rsqrt on the column faces only (face 0 holds rows 0..15, face 2 rows 16..31).
        constexpr VectorMode vm = rows <= 16 ? VectorMode::RC_custom : VectorMode::C;
        binop_with_scalar_tile_init();
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar, (APPROX, 0 /*ADD*/, 8), 0,
                             vm, eps_bits));
        rsqrt_tile_init<false>();
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt,
                             (APPROX, 8, DST_ACCUM_MODE, false, false), 0, vm));
        tile_regs_commit();
        cb_reserve_back(CB_RSTD, 1);
        tile_regs_wait();
        pack_tile(0, CB_RSTD);
        tile_regs_release();
        cb_push_back(CB_RSTD, 1);
    }

    // n = (h * rstd) * gamma. Gamma tiles hold w in every row, so the gamma multiply runs in place on the DEST
    // tiles (DEST -> srcA, configured FP32 like CB_RSTD; this matches the native kernel's FP32 x*rstd pack +
    // unpack bit for bit in every test while saving that L1 round trip).
    cb_wait_front(CB_RSTD, 1);
    cb_wait_front(CB_G, tiles);
    cb_reserve_back(CB_N, tiles);
    reconfig_data_format(CB_H, CB_RSTD);
    pack_reconfig_data_format(CB_N);
    mul_bcast_cols_init_short(CB_H, CB_RSTD);
    tile_regs_acquire();
    for (uint32_t i = 0; i < tiles; ++i) {
        mul_tiles_bcast_cols(CB_H, CB_RSTD, i, 0, i);
    }
    reconfig_data_format(CB_RSTD, CB_G);
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_G);
    for (uint32_t i = 0; i < tiles; ++i) {
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_G, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < tiles; ++i) {
        pack_tile(i, CB_N);
    }
    tile_regs_release();
    cb_push_back(CB_N, tiles);
}
