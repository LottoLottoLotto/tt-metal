// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 attention prefill (fast/attn_prefill.py FlashPrefill), compute: per work item (q head, q chunk) the q / r
// finalization of the projection rows, flash attention over the item's key tiles with the relative-position bias,
// causal / window masking and the attention sink computed in the kernel, and the sigmoid output gate.
//
// Finalization (attention.py semantics, per row: one factor from the row's sum of squares), op for op as the former
// ttnn path (bitwise):
//   r_aug = [L2clip(r, 8) [* tau] | 1 | sink | 0 ..],  q = L2clip(q, 22.63) / sqrt(128) [* tau]   (BF16)
//   ss = ttnn.sum(ttnn.mul(x, x, fp32), -1): squares on the SFPU (exact) -> FP32 CB -> FPU row reduce (the scaler
//   matmul; source registers keep TF32, as in ttnn's reduce) -> column 0; f = min(c rsqrt(ss), cap) [* tau] on the
//   SFPU; x * f by the SFPU column broadcast multiply.  r_aug goes back to the reader, which builds Q~ from it.
// Virtual key tiles (the reader's order): vt = 0 the sink tile (K = V = 0; score = its bias tile: column 0 = the
// sink logit, NEG elsewhere), vt >= 1 key position tile kt = w0 + vt - 1 (vt < nvt), beyond: padding (NEG).
// Bias: q tile r holds positions 32 (p0 + r) + o + i (o = start % 32); key tile kt, t = p0 + r - kt:
//   tmin <= t <= 32: bias tile B[r][t] = sum_k Q~[r][k] @ W_t,k  (Q~[r][k] = diag(r_aug[:, k]), W_t,k[m, j] =
//                 proj[k, 32 t + o + m - j] (0 outside [0, 1024)) for k < 16; k = 16: NEG where the key is masked
//                 (future, or past the 1024 window on sliding layers)) = r . proj[:, d] (+ NEG), rounded to BF16
//                 (the reader streams the o-shifted table of the pre-pass when o != 0; tmin = -1 then, else 0);
//   t < tmin:     NEG (future); t > 32: NEG on sliding layers, no bias (far key) on full layers.
// The B tiles of the item (Sq x 34, the NEG tile, Sq sink tiles) are computed once (prologue, streaming the table
// slots) and added to the scores of each key chunk (in place, only where needed).  The loop body is the stock ttnn
// SDPA (sdpa_inner_loop, STANDARD): online softmax, fp32 DEST, BF16 running output.  Output: BF16(out) *
// sigmoid(gate) as ttnn's BF16 multiply with the sigmoid LHS activation (16-bit DEST numerics, see
// attn_prefill_sfpu.hpp) -> BF16.
#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reduce.h"
#include "attn_prefill_sfpu.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

constexpr uint32_t Sq = get_compile_time_arg_val(0);
constexpr uint32_t Sk = get_compile_time_arg_val(1);
constexpr uint32_t DHt = get_compile_time_arg_val(2);
constexpr uint32_t qn = get_compile_time_arg_val(3);
constexpr bool SLIDING = get_compile_time_arg_val(4) != 0;
constexpr uint32_t scale_fp32 = get_compile_time_arg_val(5);
constexpr uint32_t cb_qr = get_compile_time_arg_val(6);
constexpr uint32_t cb_rr = get_compile_time_arg_val(7);
constexpr uint32_t cb_g = get_compile_time_arg_val(8);
constexpr uint32_t cb_tau = get_compile_time_arg_val(9);
constexpr uint32_t cb_ra = get_compile_time_arg_val(10);
constexpr uint32_t cb_q = get_compile_time_arg_val(11);
constexpr uint32_t cb_k = get_compile_time_arg_val(12);
constexpr uint32_t cb_v = get_compile_time_arg_val(13);
constexpr uint32_t cb_qt = get_compile_time_arg_val(14);
constexpr uint32_t cb_tbl = get_compile_time_arg_val(15);
constexpr uint32_t cb_bias = get_compile_time_arg_val(16);
constexpr uint32_t cb_scale = get_compile_time_arg_val(17);
constexpr uint32_t cb_col = get_compile_time_arg_val(18);
constexpr uint32_t cb_qk = get_compile_time_arg_val(19);
constexpr uint32_t cb_oA = get_compile_time_arg_val(20);
constexpr uint32_t cb_oB = get_compile_time_arg_val(21);
constexpr uint32_t cb_mA = get_compile_time_arg_val(22);
constexpr uint32_t cb_mB = get_compile_time_arg_val(23);
constexpr uint32_t cb_sA = get_compile_time_arg_val(24);
constexpr uint32_t cb_sB = get_compile_time_arg_val(25);
constexpr uint32_t cb_emd = get_compile_time_arg_val(26);
constexpr uint32_t cb_out = get_compile_time_arg_val(27);
constexpr uint32_t cb_fin = get_compile_time_arg_val(28);
constexpr uint32_t cb_c = get_compile_time_arg_val(29);   // constants: M16, E17, E16 (BF16)
constexpr uint32_t cb_p = get_compile_time_arg_val(30);   // params: word 0 = chunk start / 32, word 1 = start % 32
constexpr uint32_t cb_x = get_compile_time_arg_val(31);   // scratch BF16 (2 tiles)
constexpr uint32_t cb_sq = get_compile_time_arg_val(32);  // scratch FP32: squares (DHt tiles)
constexpr uint32_t cb_fr = get_compile_time_arg_val(33);  // scratch FP32 (unpacked to DEST): factor column
constexpr uint32_t cb_one = get_compile_time_arg_val(34); // FP32 SUM reduce scaler (writer)
constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(35);
constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(36);
constexpr uint32_t out_subblock_w = get_compile_time_arg_val(37);
constexpr uint32_t out_subblock_h = get_compile_time_arg_val(38);
constexpr uint32_t Q_MUL = get_compile_time_arg_val(39);  // fp32 bits: 22.63 / sqrt(128)
constexpr uint32_t Q_CAP = get_compile_time_arg_val(40);  // 1 / sqrt(128)
constexpr uint32_t R_MUL = get_compile_time_arg_val(41);  // 8
constexpr uint32_t R_CAP = get_compile_time_arg_val(42);  // 1
constexpr uint32_t kTiny = 0x0DA24260u;                   // 1e-30: rsqrt stays finite off column 0
constexpr uint32_t kM16 = 0, kE17 = 1, kE16 = 2;          // constant tiles

constexpr uint32_t kSlot = 18;
constexpr uint32_t kBand = 32;
constexpr uint32_t kRow = kBand + 2;          // band bias tiles per q tile row (t = -1..32 at t + 1)
constexpr uint32_t kNegIdx = Sq * kRow;       // the NEG tile in cb_bias
constexpr uint32_t kSinkIdx = kNegIdx + 1;    // sink bias tile of q tile row r: kSinkIdx + r
constexpr uint32_t kBiasTiles = kSinkIdx + Sq;
constexpr uint32_t kNone = 0xFFFFFFFF;
constexpr uint32_t q_chunk_tiles = Sq * DHt;
constexpr uint32_t qk_chunk_tiles = Sq * Sk;

// bias tile index of score tile (q row r, virtual key tile vt) (kNone: no bias)
FORCE_INLINE uint32_t bias_index(uint32_t r, uint32_t vt, uint32_t nvt, uint32_t p0, uint32_t w0, int32_t tmin) {
    if (vt == 0) {
        return kSinkIdx + r;
    }
    if (vt >= nvt) {
        return kNegIdx;
    }
    const int32_t t = static_cast<int32_t>(p0 + r) - static_cast<int32_t>(w0 + vt - 1);
    if (t < tmin) {
        return kNegIdx;
    }
    if (t > static_cast<int32_t>(kBand)) {
        return SLIDING ? kNegIdx : kNone;
    }
    return r * kRow + static_cast<uint32_t>(t + 1);
}

// cb_fr <- column 0: f = min(c rsqrt(ss), cap) [* tau], ss = row sums of squares of tiles x0 .. x0 + nd - 1 of
// cb_in, exactly as attention.py's former _clip_factor (ttnn.mul(x, x, fp32) -> ttnn.sum(-1) -> rsqrt -> * c ->
// minimum(cap) [-> * tau])
void row_factor(uint32_t cb_in, uint32_t x0, uint32_t nd, uint32_t c, uint32_t cap, uint32_t tau_idx) {
    // squares (SFPU, exact) -> cb_sq (FP32)
    reconfig_data_format_srca(cb_in);
    copy_tile_to_dst_init_short(cb_in);
    tile_regs_acquire();
    for (uint32_t d = 0; d < nd; ++d) {
        copy_tile(cb_in, x0 + d, d);
    }
    mul_binary_tile_init();
    for (uint32_t d = 0; d < nd; ++d) {
        mul_binary_tile(d, d, d);
    }
    tile_regs_commit();
    // always the whole CB (DHt tiles): the CB pointers stay at its start, so the indexed accesses never wrap
    cb_reserve_back(cb_sq, DHt);
    pack_reconfig_data_format(cb_sq);
    tile_regs_wait();
    for (uint32_t d = 0; d < nd; ++d) {
        pack_tile<true>(d, cb_sq, d);
    }
    tile_regs_release();
    cb_push_back(cb_sq, DHt);
    // row reduce (ttnn's REDUCE_ROW SUM sequence) -> DEST 0 column 0, then the factor chain
    cb_wait_front(cb_sq, DHt);
    reconfig_data_format(cb_one, cb_sq);
    pack_reconfig_data_format(cb_fr);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_one, cb_fr);
    tile_regs_acquire();
    for (uint32_t d = 0; d < nd; ++d) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_one, d, 0, 0);
    }
    reduce_uninit();
    unary_max_tile_init();
    unary_max_tile(0, kTiny);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, c);
    unary_min_tile_init();
    unary_min_tile(0, cap);
    if constexpr (!SLIDING) {
        reconfig_data_format_srca(cb_tau);
        copy_tile_to_dst_init_short(cb_tau);
        copy_tile(cb_tau, tau_idx, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    cb_pop_front(cb_sq, DHt);
    cb_reserve_back(cb_fr, 1);
    tile_regs_wait();
    pack_tile(0, cb_fr);
    tile_regs_release();
    cb_push_back(cb_fr, 1);
    cb_wait_front(cb_fr, 1);
}

// r_aug rows (-> cb_ra, for the reader's Q~) and finished q (-> cb_q) of the item
void finalize_qr() {
    cb_wait_front(cb_qr, q_chunk_tiles);
    cb_wait_front(cb_rr, Sq);
    if constexpr (!SLIDING) {
        cb_wait_front(cb_tau, Sq);
    }
    cb_reserve_back(cb_ra, Sq);
    for (uint32_t r = 0; r < Sq; ++r) {
        // Rm = R (.) M16 (r columns), Rx = R (.) E17 (the sink column)   (exact: 0 / 1 masks)
        reconfig_data_format(cb_rr, cb_c);
        mul_tiles_init(cb_rr, cb_c);
        cb_reserve_back(cb_x, 2);
        pack_reconfig_data_format(cb_x);
        for (uint32_t i = 0; i < 2; ++i) {
            tile_regs_acquire();
            mul_tiles(cb_rr, cb_c, r, i == 0 ? kM16 : kE17, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_x);
            tile_regs_release();
        }
        cb_push_back(cb_x, 2);
        cb_wait_front(cb_x, 2);
        row_factor(cb_x, 0, 1, R_MUL, R_CAP, r);
        // r_aug = Rm * f (FP32, one BF16 rounding at the pack) + Rx + E16 (other columns: exact)
        tile_regs_acquire();
        reconfig_data_format_srca(cb_x);
        copy_tile_to_dst_init_short(cb_x);
        copy_tile(cb_x, 0, 0);
        copy_tile(cb_x, 1, 2);
        reconfig_data_format_srca(cb_fr);
        copy_tile_to_dst_init_short(cb_fr);
        copy_tile(cb_fr, 0, 1);
        reconfig_data_format_srca(cb_c);
        copy_tile_to_dst_init_short(cb_c);
        copy_tile(cb_c, kE16, 3);
        pa::mul_col(0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 2, 0);
        add_binary_tile(0, 3, 0);
        pa::rne16_init();
        pa::rne16(0);
        tile_regs_commit();
        pack_reconfig_data_format(cb_ra);
        tile_regs_wait();
        pack_tile<true>(0, cb_ra, r);
        tile_regs_release();
        cb_pop_front(cb_fr, 1);
        cb_pop_front(cb_x, 2);
    }
    cb_push_back(cb_ra, Sq);
    cb_pop_front(cb_rr, Sq);

    cb_reserve_back(cb_q, q_chunk_tiles);
    for (uint32_t r = 0; r < Sq; ++r) {
        row_factor(cb_qr, r * DHt, DHt, Q_MUL, Q_CAP, r);
        for (uint32_t d0 = 0; d0 < DHt; d0 += 2) {
            const uint32_t nd = DHt - d0 < 2 ? DHt - d0 : 2;
            tile_regs_acquire();
            reconfig_data_format_srca(cb_qr);
            copy_tile_to_dst_init_short(cb_qr);
            for (uint32_t d = 0; d < nd; ++d) {
                copy_tile(cb_qr, r * DHt + d0 + d, d);
            }
            reconfig_data_format_srca(cb_fr);
            copy_tile_to_dst_init_short(cb_fr);
            copy_tile(cb_fr, 0, 2);
            for (uint32_t d = 0; d < nd; ++d) {
                pa::mul_col(d, 2);
            }
            pa::rne16_init();
            for (uint32_t d = 0; d < nd; ++d) {
                pa::rne16(d);
            }
            tile_regs_commit();
            pack_reconfig_data_format(cb_q);
            tile_regs_wait();
            for (uint32_t d = 0; d < nd; ++d) {
                pack_tile<true>(d, cb_q, r * DHt + d0 + d);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_fr, 1);
    }
    cb_push_back(cb_q, q_chunk_tiles);
    cb_pop_front(cb_qr, q_chunk_tiles);
    if constexpr (!SLIDING) {
        cb_pop_front(cb_tau, Sq);
    }
}

// cb_fin = cb_out * sigmoid(gate) as ttnn.mul(gate, out, input_tensor_a_activations=[SIGMOID]) on BF16 tensors:
// sigmoid with the 16-bit-DEST numerics rounded to BF16 (-> cb_x), then the FP32 product rounded to BF16 (RNE)
void gate_out() {
    cb_wait_front(cb_out, q_chunk_tiles);
    cb_wait_front(cb_g, q_chunk_tiles);
    cb_reserve_back(cb_fin, q_chunk_tiles);
    for (uint32_t i = 0; i < q_chunk_tiles; ++i) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_g);
        copy_tile_to_dst_init_short(cb_g);
        copy_tile(cb_g, i, 0);
        sigmoid_tile_init();
        pa::sigmoid16(0);
        tile_regs_commit();
        cb_reserve_back(cb_x, 1);
        pack_reconfig_data_format(cb_x);
        tile_regs_wait();
        pack_tile(0, cb_x);
        tile_regs_release();
        cb_push_back(cb_x, 1);
        cb_wait_front(cb_x, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_x);
        copy_tile_to_dst_init_short(cb_x);
        copy_tile(cb_x, 0, 0);
        reconfig_data_format_srca(cb_out);
        copy_tile_to_dst_init_short(cb_out);
        copy_tile(cb_out, i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        pa::rne16_init();
        pa::rne16(0);
        tile_regs_commit();
        pack_reconfig_data_format(cb_fin);
        tile_regs_wait();
        pack_tile(0, cb_fin);
        tile_regs_release();
        cb_pop_front(cb_x, 1);
    }
    cb_push_back(cb_fin, q_chunk_tiles);
    cb_pop_front(cb_out, q_chunk_tiles);
    cb_pop_front(cb_g, q_chunk_tiles);
}

// in0 = exp(in0 - rowmax) in place, row partial sums -> reduce_cb (L1 accumulation): the stock
// sub_exp_block_bcast_cols_inplace with the EXACT exponential (the stock helper hard-codes the approximate one; the
// ttnn SDPA the old sliding-layer prefill used runs with exp_approx_mode = False)
void sub_exp_rows_inplace(uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols) {
    constexpr uint32_t rows = Sq;
    sub_bcast_cols_init_short(cb_qk, in1_cb);
    reconfig_data_format(cb_qk, in1_cb);
    exp_tile_init<false>();
    cb_wait_front(cb_qk, rows * cols);
    cb_wait_front(in1_cb, rows);
    cb_reserve_back(reduce_cb, rows);
#ifdef SUB_EXP_GRANULARITY
    const uint32_t dst_tiles = (cols < SUB_EXP_GRANULARITY) ? cols : SUB_EXP_GRANULARITY;
    const uint32_t granularity = (cols >= SUB_EXP_GRANULARITY) ? (cols / SUB_EXP_GRANULARITY) : 1;
#else
    const uint32_t dst_tiles = cols;
    const uint32_t granularity = 1;
#endif
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(cb_qk, in1_cb, j, i, j);
                exp_tile<false>(j);
            }
            tile_regs_commit();
            cb_pop_front(cb_qk, dst_tiles);
            cb_reserve_back(cb_qk, dst_tiles);
            tile_regs_wait();
            pack_reconfig_data_format(cb_qk);
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, cb_qk);
            }
            cb_push_back(cb_qk, dst_tiles);
            pack_reconfig_data_format(reduce_cb);
            if (u > 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile<true>(j, reduce_cb, i);
                if (u == 0 && j == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }
            tile_regs_release();
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
    }
    cb_push_back(reduce_cb, rows);
}

// scores of the chunk += bias tiles (in place; only the tiles with a bias are rewritten)
void add_bias(uint32_t ch, uint32_t nvt, uint32_t p0, uint32_t w0, int32_t tmin) {
    bool any = false;
    for (uint32_t r = 0; r < Sq && !any; ++r) {
        for (uint32_t c = 0; c < Sk; ++c) {
            if (bias_index(r, ch * Sk + c, nvt, p0, w0, tmin) != kNone) {
                any = true;
                break;
            }
        }
    }
    if (!any) {
        return;
    }
    reconfig_data_format(cb_qk, cb_bias);
    pack_reconfig_data_format(cb_qk);
    add_tiles_init(cb_qk, cb_bias);
    cb_wait_front(cb_qk, qk_chunk_tiles);
    for (uint32_t r = 0; r < Sq; ++r) {
        for (uint32_t c = 0; c < Sk; ++c) {
            const uint32_t b = bias_index(r, ch * Sk + c, nvt, p0, w0, tmin);
            if (b == kNone) {
                continue;
            }
            const uint32_t i = r * Sk + c;
            tile_regs_acquire();
            add_tiles(cb_qk, cb_bias, i, b, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_qk, i);
            tile_regs_release();
        }
    }
    cb_pop_front(cb_qk, qk_chunk_tiles);
    cb_reserve_back(cb_qk, qk_chunk_tiles);
    cb_push_back(cb_qk, qk_chunk_tiles);
}

// B tiles of the item: table slot s = t + 1 -> cb_bias[r * 34 + s] (smin <= s <= tmax + 1), the sink slot ->
// cb_bias[kSinkIdx + r], the NEG slot -> cb_bias[kNegIdx]; the k ranges match the reader's slot_range
void prologue(uint32_t smin, uint32_t tmax) {
    cb_wait_front(cb_qt, Sq * kSlot);
    cb_reserve_back(cb_bias, kBiasTiles);
    reconfig_data_format(cb_tbl, cb_qt);
    pack_reconfig_data_format(cb_bias);
    matmul_block_init(cb_qt, cb_tbl, 0, 1, Sq, kSlot);
    for (uint32_t s = smin; s <= tmax + 2; ++s) {
        const bool sink = s == tmax + 2;
        const uint32_t k0 = sink ? 16 : 0;
        const uint32_t k1 = sink ? 18 : ((s <= 1 || s >= kBand) ? 17 : 16);
        cb_wait_front(cb_tbl, kSlot);
        tile_regs_acquire();
        for (uint32_t k = k0; k < k1; ++k) {
            matmul_block(cb_qt, cb_tbl, k, k, 0, 0, 1, Sq, kSlot);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < Sq; ++r) {
            pack_tile<true>(r, cb_bias, sink ? kSinkIdx + r : r * kRow + s);
        }
        tile_regs_release();
        cb_pop_front(cb_tbl, kSlot);
    }
    // NEG tile
    cb_wait_front(cb_tbl, kSlot);
    reconfig_data_format_srca(cb_tbl);
    copy_tile_to_dst_init_short(cb_tbl);
    tile_regs_acquire();
    copy_tile(cb_tbl, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb_bias, kNegIdx);
    tile_regs_release();
    cb_pop_front(cb_tbl, kSlot);
    cb_push_back(cb_bias, kBiasTiles);
}

void kernel_main() {
    const uint32_t gq_start = get_arg_val<uint32_t>(0);
    const uint32_t gq_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t qk_in0_num_subblocks = Sq / qk_subblock_h;
    constexpr uint32_t qk_in1_num_subblocks = Sk / qk_subblock_w;
    constexpr uint32_t out_in0_num_subblocks = Sq / out_subblock_h;
    constexpr uint32_t out_in1_num_subblocks = DHt / out_subblock_w;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_out);
    matmul_init(cb_q, cb_k);

    cb_wait_front(cb_p, 1);
    const uint32_t s_t = reinterpret_cast<const volatile uint32_t*>(get_tile_address(cb_p, 0))[0];
    const uint32_t o = reinterpret_cast<const volatile uint32_t*>(get_tile_address(cb_p, 0))[1];
    const int32_t tmin = o != 0 ? -1 : 0;
    cb_wait_front(cb_c, 3);
    cb_wait_front(cb_one, 1);

    for (uint32_t it = 0; it < gq_count; ++it) {
        const uint32_t gq = gq_start + it;
        const uint32_t qc = gq % qn;
        const uint32_t p0 = s_t + qc * Sq;
        const uint32_t w0 = SLIDING && p0 > kBand ? p0 - kBand : 0;
        const uint32_t nkeys = p0 + Sq - w0 + (o != 0 ? 1 : 0);
        const uint32_t nvt = 1 + nkeys;
        const uint32_t nch = (nvt + Sk - 1) / Sk;
        const uint32_t tmax = p0 + Sq - 1 - w0 < kBand ? p0 + Sq - 1 - w0 : kBand;

        finalize_qr();
        prologue(o != 0 ? 0 : 1, tmax);
        cb_wait_front(cb_bias, kBiasTiles);

        uint32_t prev_sum = cb_sA, cur_sum = cb_sB;
        uint32_t prev_max = cb_mA, cur_max = cb_mB;
        uint32_t prev_out = cb_oA, cur_out = cb_oB;
        for (uint32_t ch = 0; ch < nch; ++ch) {
            /* QK = Q @ K^T */
            reconfig_data_format(cb_k, cb_q);
            pack_reconfig_data_format(cb_qk);
            matmul_blocks(cb_q, cb_k, cb_qk, Sq, Sk, DHt, 1, qk_in0_num_subblocks, qk_in1_num_subblocks, DHt,
                          qk_subblock_h, qk_subblock_w, true);
            /* QK += bias */
            add_bias(ch, nvt, p0, w0, tmin);
            reconfig_data_format(cb_qk, cb_scale);
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk, cb_scale, Sq>(cur_max, prev_max, Sk, ch > 0);
#ifdef EXACT_EXP  // sliding: as the old ttnn SDPA path (1.2x the time); full layers on request (2.2x)
            sub_exp_rows_inplace(cur_max, cur_sum, Sk);
#else  // approximate exp as the old full-layer flash prefill
            sub_exp_block_bcast_cols_inplace<cb_qk, Sq, scale_fp32, true>(cur_max, cur_sum, Sk);
#endif
            /* OUT_IM = P @ V */
            reconfig_data_format(cb_v, cb_qk);
            pack_reconfig_data_format(cur_out);
            matmul_blocks(cb_qk, cb_v, cur_out, Sq, DHt, Sk, 1, out_in0_num_subblocks, out_in1_num_subblocks, Sk,
                          out_subblock_h, out_subblock_w, false);
            cb_pop_front(cb_qk, qk_chunk_tiles);
            reconfig_data_format(prev_max, cur_max);
            if (ch > 0) {
                sub_exp_block<scale_fp32>(prev_max, cur_max, cb_emd, Sq);
                cb_pop_front(prev_max, Sq);
                mul_tiles_bcast_cols_inplace(prev_sum, cb_emd, Sq);
                add_block_inplace(cur_sum, prev_sum, Sq);
                mul_block_bcast_cols<Sq, DHt, false, true>(prev_out, cb_emd, cur_out);
            }
            std::swap(prev_sum, cur_sum);
            std::swap(prev_out, cur_out);
            std::swap(prev_max, cur_max);
        }
        matmul_reduce<Sq>(cb_col, prev_sum);

        /* out = OUT_ACC / sum, then * sigmoid(gate) */
        recip_block_inplace(prev_sum, Sq);
        pack_reconfig_data_format(cb_out);
        mul_block_bcast_cols<Sq, DHt, false, false>(prev_out, prev_sum, cb_out);
        cb_pop_front(prev_max, Sq);
        gate_out();
        cb_pop_front(cb_q, q_chunk_tiles);
        cb_pop_front(cb_bias, kBiasTiles);
        cb_pop_front(cb_qt, Sq * kSlot);
    }
}
