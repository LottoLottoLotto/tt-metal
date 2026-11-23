// SPDX-License-Identifier: Apache-2.0
// Verify-only copy of chunk_gated_delta_rule recurrent compute (VERIFY_TWO/VERIFY_TOKENS).

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
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4, cb_state = 5;
constexpr uint32_t cb_decay_bias = 6, cb_decay_scale = 7;
constexpr uint32_t cb_out = 16, cb_final = 17, cb_decayed = 18, cb_vread = 19;
constexpr uint32_t cb_delta = 20, cb_delta_beta = 21, cb_kcol = 22, cb_outer = 23;
constexpr uint32_t cb_newstate = 24, cb_decay = 25, cb_ones = 26, cb_sq = 27;
constexpr uint32_t cb_sum = 28, cb_inv = 29;

inline void wait(uint32_t cb, uint32_t count) { CircularBuffer(cb).wait_front(count); }
inline void pop(uint32_t cb, uint32_t count) { CircularBuffer(cb).pop_front(count); }

void mm(uint32_t a, uint32_t b, uint32_t out, uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    cb_reserve_back(out, Mt * Nt);
    pack_reconfig_data_format(out);
    reconfig_data_format(b, a);
    matmul_init(a, b, 0);
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t n = 0; n < Nt; ++n) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; ++k) {
                matmul_tiles(a, b, m * Kt + k, k * Nt + n, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out, m * Nt + n);
            tile_regs_release();
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
    for (uint32_t i = 0; i < count; ++i) {
        tile_regs_acquire();
        if (subtract) {
            sub_tiles(a, b, i, i, 0);
        } else {
            add_tiles(a, b, i, i, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    cb_push_back(out, count);
}

void scalar_mul(uint32_t a, uint32_t scalar, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, scalar);
    mul_tiles_bcast_scalar_init_short(a, scalar);
    for (uint32_t i = 0; i < count; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scalar, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
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
    exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

void multiply_tiles(uint32_t a, uint32_t b, uint32_t out, uint32_t count) {
    cb_reserve_back(out, count);
    pack_reconfig_data_format(out);
    reconfig_data_format(a, b);
    mul_tiles_init(a, b);
    for (uint32_t i = 0; i < count; ++i) {
        tile_regs_acquire();
        mul_tiles(a, b, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
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
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    if (apply_scale) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_bits);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

void sigmoid_one(uint32_t in, uint32_t out) {
    cb_reserve_back(out, 1);
    pack_reconfig_data_format(out);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    sigmoid_tile_init();
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    sigmoid_tile(0);
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
    softplus_tile(0, 0x3F800000u, 0x3F800000u, 0x41A00000u);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    cb_push_back(out, 1);
}

// Row-wise L2 normalization of up to 32 rows at once: identical per-row arithmetic to
// normalize() (row sums by the ones matmul fill every column; each row is scaled by
// its own column-0 inverse instead of element (0, 0)).
void normalize_rows(
    uint32_t in, uint32_t out, uint32_t Kt, uint32_t eps_bits, uint32_t scale_bits, bool apply_scale) {
    multiply_tiles(in, in, cb_sq, Kt);
    wait(cb_sq, Kt);
    rowsum_k(cb_sq, cb_sum, Kt);
    wait(cb_sum, 1);
    pop(cb_sq, Kt);
    inverse_norm(cb_sum, cb_inv, eps_bits, scale_bits, apply_scale);
    wait(cb_inv, 1);
    pop(cb_sum, 1);
    cb_reserve_back(out, Kt);
    pack_reconfig_data_format(out);
    reconfig_data_format(in, cb_inv);
    mul_bcast_cols_init_short(in, cb_inv);
    for (uint32_t i = 0; i < Kt; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(in, cb_inv, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, i);
        tile_regs_release();
    }
    cb_push_back(out, Kt);
    pop(cb_inv, 1);
    pop(in, Kt);
}

}  // namespace

constexpr uint32_t cb_qall = 8, cb_kall = 9, cb_qnall = 10, cb_knall = 11;
constexpr uint32_t cb_gall = 13, cb_ball = 14, cb_biasall = 15, cb_dall = 30, cb_sall = 31;

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

    // All rows' q/k are normalized once; the reader hands row t back in row 0 of
    // cb_q/cb_k for token t. This runs before the state wait: the reader publishes
    // the state only after it has received these normalized rows.
    wait(cb_ones, 1);
    wait(cb_qall, Kt);
    normalize_rows(cb_qall, cb_qnall, Kt, EPS_BITS, SCALE_BITS, true);
    wait(cb_kall, Kt);
    normalize_rows(cb_kall, cb_knall, Kt, EPS_BITS, SCALE_BITS, false);
    // Decay exp(softplus(g + bias) * scale) and sigmoid(beta) for every row at once
    // (row t, column 0 per token; elementwise ops, the same per-element arithmetic).
    // The reader hands row t's values back at (0, 0) of cb_g / cb_beta.
    wait(cb_decay_bias, 1);
    wait(cb_decay_scale, 1);
    wait(cb_gall, 1);
    wait(cb_biasall, 1);
    ew(cb_gall, cb_biasall, cb_sq, 1, false);
    wait(cb_sq, 1);
    pop(cb_gall, 1);
    pop(cb_biasall, 1);
    softplus_one(cb_sq, cb_inv);
    wait(cb_inv, 1);
    pop(cb_sq, 1);
    scalar_mul(cb_inv, cb_decay_scale, cb_sq, 1);
    wait(cb_sq, 1);
    pop(cb_inv, 1);
    exp_one(cb_sq, cb_dall);
    pop(cb_sq, 1);
    wait(cb_ball, 1);
    sigmoid_one(cb_ball, cb_sall);
    pop(cb_ball, 1);
#if !defined(BATCHED_USERS) && !defined(GROUP_ROWS)
    wait(cb_state, kv);
#endif

    uint32_t current_state = cb_state;
    for (uint32_t token = 0; token < num_tokens; ++token) {
        #ifdef TREE_FORK_POSITION
        if (token == 8) {
            pop(current_state, kv);
            wait(cb_state, kv);
            current_state = cb_state;
        }
        #endif
        #ifdef BATCHED_USERS
        // Every row is a different user: the reader has loaded this row's own state.
        wait(cb_state, kv);
        current_state = cb_state;
        #endif
        wait(cb_q, Kt);
        wait(cb_k, Kt);
        wait(cb_v, Vt);
        wait(cb_g, 1);
        wait(cb_beta, 1);
        #ifdef GROUP_ROWS
        // Reader flags (word 8 of this row's cb_g tile): bit 0 the row starts a group from the
        // state the reader loaded, bit 1 the next row continues this row's group.
        const uint32_t row_flags = read_tile_value(cb_g, 0, 8);
        if (row_flags & 1) {
            wait(cb_state, kv);
            current_state = cb_state;
        }
        #endif
        // cb_g carries this row's decay, cb_beta its sigmoid(beta), both at (0, 0).
        scalar_mul(current_state, cb_g, cb_decayed, kv);
        wait(cb_decayed, kv);
        // The decayed copy is now the only remaining consumer of this state.
        // On t=1 this releases cb_newstate before it is reused as the destination.
        pop(current_state, kv);

        mm(cb_k, cb_decayed, cb_vread, 1, Kt, Vt);
        wait(cb_vread, Vt);
        ew(cb_v, cb_vread, cb_delta, Vt, true);
        wait(cb_delta, Vt);
        scalar_mul(cb_delta, cb_beta, cb_delta_beta, Vt);
        wait(cb_delta_beta, Vt);

        cb_reserve_back(cb_kcol, Kt);
        pack_reconfig_data_format(cb_kcol);
        reconfig_data_format_srca(cb_k);
        transpose_init(cb_k);
        for (uint32_t k = 0; k < Kt; ++k) {
            tile_regs_acquire();
            transpose_tile(cb_k, k, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_kcol, k);
            tile_regs_release();
        }
        cb_push_back(cb_kcol, Kt);
        wait(cb_kcol, Kt);

        mm(cb_kcol, cb_delta_beta, cb_outer, Kt, 1, Vt);
        wait(cb_outer, kv);
        // newstate = decayed + outer, packed twice from the same FP32 dest: the
        // persistent next-row input and the candidate the writer streams out.
        // Identical BF16 rounding to the former unpack/copy/repack of newstate.
        cb_reserve_back(cb_newstate, kv);
        cb_reserve_back(cb_final, kv);
        pack_reconfig_data_format(cb_newstate);
        reconfig_data_format(cb_decayed, cb_outer);
        add_tiles_init(cb_decayed, cb_outer);
        for (uint32_t i = 0; i < kv; ++i) {
            tile_regs_acquire();
            add_tiles(cb_decayed, cb_outer, i, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_newstate, i);
            pack_tile(0, cb_final, i);
            tile_regs_release();
        }
        cb_push_back(cb_newstate, kv);
        cb_push_back(cb_final, kv);
        wait(cb_newstate, kv);
        mm(cb_q, cb_newstate, cb_out, 1, Kt, Vt);

        pop(cb_q, Kt);
        pop(cb_k, Kt);
        pop(cb_v, Vt);
        pop(cb_g, 1);
        pop(cb_beta, 1);
        pop(cb_decayed, kv);
        pop(cb_vread, Vt);
        pop(cb_delta, Vt);
        pop(cb_delta_beta, Vt);
        pop(cb_kcol, Kt);
        pop(cb_outer, kv);

#ifdef BATCHED_USERS
        // The writer stores cb_final to this row's user; nothing carries to the next row.
        pop(cb_newstate, kv);
#elif defined(GROUP_ROWS)
        if (row_flags & 2) {
            current_state = cb_newstate;
        } else {
            pop(cb_newstate, kv);
        }
#else
        if (token + 1 < num_tokens) {
            // Keep the rounded state as the next row's input; cb_final has
            // already received the independent candidate consumed by the writer.
            current_state = cb_newstate;
        } else {
            pop(cb_newstate, kv);
        }
#endif
    }

    pop(cb_decay_bias, 1);
    pop(cb_decay_scale, 1);
    pop(cb_ones, 1);
}
