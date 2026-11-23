// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4, cb_state = 5;
constexpr uint32_t cb_decay_bias = 6, cb_decay_scale = 7, cb_state_rows = 8;
constexpr uint32_t cb_out = 16, cb_final = 17, cb_decayed = 18, cb_vread = 19;
constexpr uint32_t cb_delta = 20, cb_delta_beta = 21, cb_kcol = 22, cb_outer = 23;
constexpr uint32_t cb_newstate = 24, cb_decay = 25, cb_ones = 26, cb_sq = 27;
constexpr uint32_t cb_sum = 28, cb_inv = 29, cb_qnorm = 30, cb_knorm = 31;

inline void wait(uint32_t cb, uint32_t count) { CircularBuffer(cb).wait_front(count); }
inline void pop(uint32_t cb, uint32_t count) { CircularBuffer(cb).pop_front(count); }

// FP32 DEST in half-sync mode holds four tiles per half. Each helper fills up to that
// many DEST tiles per acquire/commit/pack/release cycle: the per-tile arithmetic and
// the single BF16 pack of every result are unchanged, only the sync overhead is shared.
constexpr uint32_t kDstTiles = 4;

// The per-head scalars (inverse norms, sigmoid(beta), softplus/exp decay) are consumed
// only as tile element [0,0] (scalar broadcast). Their SFPU ops therefore run on the
// first row block of faces 0-1 (VectorMode::R, one iteration) instead of 4 faces x 8.
constexpr int kScalarIterations = 1;

inline uint32_t batch(uint32_t first, uint32_t count) {
    return count - first < kDstTiles ? count - first : kDstTiles;
}

inline void pack_batch(uint32_t out, uint32_t first, uint32_t n) {
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < n; ++j) {
        pack_tile(j, out, first + j);
    }
    tile_regs_release();
}

void mm(uint32_t a, uint32_t b, uint32_t out, uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    cb_reserve_back(out, Mt * Nt);
    pack_reconfig_data_format(out);
    reconfig_data_format(b, a);
    matmul_init(a, b, 0);
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t n0 = 0; n0 < Nt; n0 += kDstTiles) {
            const uint32_t n = batch(n0, Nt);
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t k = 0; k < Kt; ++k) {
                    matmul_tiles(a, b, m * Kt + k, k * Nt + n0 + j, j);
                }
            }
            pack_batch(out, m * Nt + n0, n);
        }
    }
    cb_push_back(out, Mt * Nt);
}

void ew(uint32_t a, uint32_t b, uint32_t out, uint32_t count, bool subtract) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    if (subtract) {
        sub_tiles_init(a, b);
    } else {
        add_tiles_init(a, b);
    }
    for (uint32_t i0 = 0; i0 < count; i0 += kDstTiles) {
        const uint32_t n = batch(i0, count);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            if (subtract) {
                sub_tiles(a, b, i0 + j, i0 + j, j);
            } else {
                add_tiles(a, b, i0 + j, i0 + j, j);
            }
        }
        pack_batch(out, i0, n);
    }
    cb_push_back(out, count);
}

void scalar_mul(uint32_t a, uint32_t scalar, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, scalar);
    mul_tiles_bcast_scalar_init_short(a, scalar);
    for (uint32_t i0 = 0; i0 < count; i0 += kDstTiles) {
        const uint32_t n = batch(i0, count);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            mul_tiles_bcast_scalar(a, scalar, i0 + j, 0, j);
        }
        pack_batch(out, i0, n);
    }
    cb_push_back(out, count);
}

void exp_one(uint32_t in, uint32_t out) {
    cb_reserve_back(out, 1);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    exp_tile_init();
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_exponential,
        (false, DST_ACCUM_MODE, false, kScalarIterations, true), 0, VectorMode::R, p_sfpu::kCONST_1_FP16B));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

void copy_tiles(uint32_t in, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    for (uint32_t i0 = 0; i0 < count; i0 += kDstTiles) {
        const uint32_t n = batch(i0, count);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            copy_tile(in, i0 + j, j);
        }
        pack_batch(out, i0, n);
    }
    cb_push_back(out, count);
}
void multiply_tiles(uint32_t a, uint32_t b, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    mul_tiles_init(a, b);
    for (uint32_t i0 = 0; i0 < count; i0 += kDstTiles) {
        const uint32_t n = batch(i0, count);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            mul_tiles(a, b, i0 + j, i0 + j, j);
        }
        pack_batch(out, i0, n);
    }
    cb_push_back(out, count);
}

void rowsum_k(uint32_t in, uint32_t out, uint32_t Kt) {
    cb_reserve_back(out, 1);
    pack_reconfig_data_format(out);
    reconfig_data_format(cb_ones, in);
    matmul_init(in, cb_ones, 0);
    tile_regs_acquire();
    for (uint32_t k = 0; k < Kt; ++k) {
        matmul_tiles(in, cb_ones, k, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

void inverse_norm(
    uint32_t in, uint32_t out, uint32_t eps_bits, uint32_t scale_bits, bool apply_scale) {
    cb_reserve_back(out, 1);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    binop_with_scalar_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, kScalarIterations), 0, VectorMode::R, eps_bits));
    rsqrt_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt,
        (APPROX, kScalarIterations, DST_ACCUM_MODE, false, false), 0, VectorMode::R));
    if (apply_scale) {
        binop_with_scalar_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
            (APPROX, MUL_UNARY, kScalarIterations), 0, VectorMode::R, scale_bits));
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

void normalize(
    uint32_t in, uint32_t out, uint32_t Kt, uint32_t eps_bits, uint32_t scale_bits, bool apply_scale) {
    multiply_tiles(in, in, cb_sq, Kt);
    wait(cb_sq, Kt);
    rowsum_k(cb_sq, cb_sum, Kt);
    wait(cb_sum, 1);
    pop(cb_sq, Kt);
    inverse_norm(cb_sum, cb_inv, eps_bits, scale_bits, apply_scale);
    wait(cb_inv, 1);
    pop(cb_sum, 1);
    scalar_mul(in, cb_inv, out, Kt);
    wait(out, Kt);
    pop(cb_inv, 1);
}

void sigmoid_one(uint32_t in, uint32_t out) {
    cb_reserve_back(out, 1);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    sigmoid_tile_init();
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sigmoid,
        (false, DST_ACCUM_MODE, kScalarIterations), 0, VectorMode::R));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

void softplus_one(uint32_t in, uint32_t out) {
    cb_reserve_back(out, 1);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    softplus_tile_init();
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_softplus,
        (APPROX, DST_ACCUM_MODE, kScalarIterations), 0, VectorMode::R, 0x3F800000u, 0x3F800000u, 0x41A00000u));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

}  // namespace

#if defined(VERIFY_TWO) || defined(VERIFY_TOKENS)
void kernel_main() {
#ifdef VERIFY_TOKENS
    constexpr uint32_t num_tokens = VERIFY_TOKENS;
    static_assert(num_tokens >= 2 && num_tokens <= 32);
#else
    constexpr uint32_t num_tokens = 2;
#endif
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(4);
    constexpr uint32_t kv = Kt * Vt;
    compute_kernel_hw_startup(cb_q, cb_k, cb_out);

    // The reader streams the row-major persistent state one tile row (32 rows) at a
    // time; tilize it in hardware into the tiled working state, once for all tokens.
    // Per-head normalization and decay constants stay resident across the token steps.
    cb_reserve_back(cb_state, kv);
    tilize_init(cb_state_rows, Vt, cb_state);
    for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
        cb_wait_front(cb_state_rows, Vt);
        tilize_block(cb_state_rows, Vt, cb_state, 0, tile_row * Vt);
        cb_pop_front(cb_state_rows, Vt);
    }
    tilize_uninit(cb_state_rows, cb_state);
    cb_push_back(cb_state, kv);
    wait(cb_state, kv);
    wait(cb_decay_bias, 1);
    wait(cb_decay_scale, 1);
    wait(cb_ones, 1);

    uint32_t current_state = cb_state;
    for (uint32_t token = 0; token < num_tokens; ++token) {
        #ifdef TREE_FORK_POSITION
        if (token == 8) {
            pop(current_state, kv);
            wait(cb_state, kv);
            current_state = cb_state;
        }
        #endif
        wait(cb_q, Kt);
        wait(cb_k, Kt);
        wait(cb_v, Vt);
        wait(cb_g, 1);
        wait(cb_beta, 1);

        normalize(cb_q, cb_qnorm, Kt, EPS_BITS, SCALE_BITS, true);
        normalize(cb_k, cb_knorm, Kt, EPS_BITS, SCALE_BITS, false);

        sigmoid_one(cb_beta, cb_sum);
        wait(cb_sum, 1);

        ew(cb_g, cb_decay_bias, cb_sq, 1, false);
        wait(cb_sq, 1);
        softplus_one(cb_sq, cb_inv);
        wait(cb_inv, 1);
        pop(cb_sq, 1);
        scalar_mul(cb_inv, cb_decay_scale, cb_sq, 1);
        wait(cb_sq, 1);
        pop(cb_inv, 1);
        exp_one(cb_sq, cb_decay);
        wait(cb_decay, 1);
        pop(cb_sq, 1);

        scalar_mul(current_state, cb_decay, cb_decayed, kv);
        wait(cb_decayed, kv);
        // The decayed copy is now the only remaining consumer of this state.
        // On t=1 this releases cb_newstate before it is reused as the destination.
        pop(current_state, kv);

        mm(cb_knorm, cb_decayed, cb_vread, 1, Kt, Vt);
        wait(cb_vread, Vt);
        ew(cb_v, cb_vread, cb_delta, Vt, true);
        wait(cb_delta, Vt);
        scalar_mul(cb_delta, cb_sum, cb_delta_beta, Vt);
        wait(cb_delta_beta, Vt);

        cb_reserve_back(cb_kcol, Kt);
        pack_reconfig_data_format(cb_kcol);
        reconfig_data_format_srca(cb_knorm);
        transpose_init(cb_knorm);
        for (uint32_t k0 = 0; k0 < Kt; k0 += kDstTiles) {
            const uint32_t n = batch(k0, Kt);
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                transpose_tile(cb_knorm, k0 + j, j);
            }
            pack_batch(cb_kcol, k0, n);
        }
        cb_push_back(cb_kcol, Kt);
        wait(cb_kcol, Kt);

        mm(cb_kcol, cb_delta_beta, cb_outer, Kt, 1, Vt);
        wait(cb_outer, kv);
        ew(cb_decayed, cb_outer, cb_newstate, kv, false);
        wait(cb_newstate, kv);
        mm(cb_qnorm, cb_newstate, cb_out, 1, Kt, Vt);
        copy_tiles(cb_newstate, cb_final, kv);

        pop(cb_qnorm, Kt);
        pop(cb_knorm, Kt);
        pop(cb_sum, 1);
        pop(cb_q, Kt);
        pop(cb_k, Kt);
        pop(cb_v, Vt);
        pop(cb_g, 1);
        pop(cb_beta, 1);
        pop(cb_decay, 1);
        pop(cb_decayed, kv);
        pop(cb_vread, Vt);
        pop(cb_delta, Vt);
        pop(cb_delta_beta, Vt);
        pop(cb_kcol, Kt);
        pop(cb_outer, kv);

        if (token + 1 < num_tokens) {
            // Keep the rounded state as the next row's input; cb_final has
            // already received the independent candidate consumed by the writer.
            current_state = cb_newstate;
        } else {
            pop(cb_newstate, kv);
        }
    }

    pop(cb_decay_bias, 1);
    pop(cb_decay_scale, 1);
    pop(cb_ones, 1);
}
#else
void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t FUSED_FRONTEND = get_compile_time_arg_val(2);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(4);
    constexpr uint32_t kv = Kt * Vt;
    compute_kernel_hw_startup(cb_q, cb_k, cb_out);

    wait(cb_q, Kt);
    wait(cb_k, Kt);
    wait(cb_v, Vt);
    wait(cb_g, 1);
    wait(cb_beta, 1);
    // The reader streams the row-major state one tile row (32 rows) at a time;
    // tilize it in hardware into the tiled working state.
    cb_reserve_back(cb_state, kv);
    tilize_init(cb_state_rows, Vt, cb_state);
    for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
        cb_wait_front(cb_state_rows, Vt);
        tilize_block(cb_state_rows, Vt, cb_state, 0, tile_row * Vt);
        cb_pop_front(cb_state_rows, Vt);
    }
    tilize_uninit(cb_state_rows, cb_state);
    cb_push_back(cb_state, kv);
    wait(cb_state, kv);

    uint32_t Q = cb_q;
    uint32_t K = cb_k;
    uint32_t Beta = cb_beta;
    if constexpr (FUSED_FRONTEND) {
        wait(cb_decay_bias, 1);
        wait(cb_decay_scale, 1);
        wait(cb_ones, 1);

        normalize(cb_q, cb_qnorm, Kt, EPS_BITS, SCALE_BITS, true);
        normalize(cb_k, cb_knorm, Kt, EPS_BITS, SCALE_BITS, false);
        pop(cb_ones, 1);
        Q = cb_qnorm;
        K = cb_knorm;

        sigmoid_one(cb_beta, cb_sum);
        wait(cb_sum, 1);
        Beta = cb_sum;

        ew(cb_g, cb_decay_bias, cb_sq, 1, false);
        wait(cb_sq, 1);
        softplus_one(cb_sq, cb_inv);
        wait(cb_inv, 1);
        pop(cb_sq, 1);
        scalar_mul(cb_inv, cb_decay_scale, cb_sq, 1);
        wait(cb_sq, 1);
        pop(cb_inv, 1);
        exp_one(cb_sq, cb_decay);
        wait(cb_decay, 1);
        pop(cb_sq, 1);
        pop(cb_decay_bias, 1);
        pop(cb_decay_scale, 1);
    } else {
        exp_one(cb_g, cb_decay);
        wait(cb_decay, 1);
    }

    scalar_mul(cb_state, cb_decay, cb_decayed, kv);
    wait(cb_decayed, kv);

    mm(K, cb_decayed, cb_vread, 1, Kt, Vt);
    wait(cb_vread, Vt);
    ew(cb_v, cb_vread, cb_delta, Vt, true);
    wait(cb_delta, Vt);
    scalar_mul(cb_delta, Beta, cb_delta_beta, Vt);
    wait(cb_delta_beta, Vt);

    cb_reserve_back(cb_kcol, Kt);
    pack_reconfig_data_format(cb_kcol);
    reconfig_data_format_srca(K);
    transpose_init(K);
    for (uint32_t k0 = 0; k0 < Kt; k0 += kDstTiles) {
        const uint32_t n = batch(k0, Kt);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            transpose_tile(K, k0 + j, j);
        }
        pack_batch(cb_kcol, k0, n);
    }
    cb_push_back(cb_kcol, Kt);
    wait(cb_kcol, Kt);

    mm(cb_kcol, cb_delta_beta, cb_outer, Kt, 1, Vt);
    wait(cb_outer, kv);
    ew(cb_decayed, cb_outer, cb_newstate, kv, false);
    wait(cb_newstate, kv);
    mm(Q, cb_newstate, cb_out, 1, Kt, Vt);
    // Untilize the final state into row-major sticks, one tile row at a time, for the
    // writer's whole-row stores into the persistent state.
    pack_untilize_init<Vt, Vt>(cb_newstate, cb_final);
    for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
        cb_reserve_back(cb_final, Vt);
        pack_untilize_block<Vt, Vt>(cb_newstate, 1, cb_final);
        cb_push_back(cb_final, Vt);
        cb_pop_front(cb_newstate, Vt);
    }
    pack_untilize_uninit(cb_final);

    if constexpr (FUSED_FRONTEND) {
        pop(cb_qnorm, Kt);
        pop(cb_knorm, Kt);
        pop(cb_sum, 1);
    }
    pop(cb_q, Kt);
    pop(cb_k, Kt);
    pop(cb_v, Vt);
    pop(cb_g, 1);
    pop(cb_beta, 1);
    pop(cb_state, kv);
    pop(cb_decay, 1);
    pop(cb_decayed, kv);
    pop(cb_vread, Vt);
    pop(cb_delta, Vt);
    pop(cb_delta_beta, Vt);
    pop(cb_kcol, Kt);
    pop(cb_outer, kv);
}
#endif
