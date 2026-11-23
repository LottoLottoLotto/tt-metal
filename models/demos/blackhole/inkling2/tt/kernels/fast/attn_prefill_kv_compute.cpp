// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 attention prefill K / V front (fast/attn_prefill.py KVFront), compute.  Per work unit (kv head g, kind kv,
// aligned-frame tile row i'; see the reader) and column tile c of the head, attention.py's former ttnn chunk path op
// for op (the K / V cache rows come out bitwise):
//   x_lag = rows of the linear output shifted by o + lag (exact 0/1 shift matmuls: S_up(u') @ X_hi + S_dn(u') @ X_lo)
//   sliding: conv = sum_lag x_lag * taps_lag (fp32 taps incl. residual and 1/sqrt(5)), lag order 3, 2, 1, 0
//   full:    conv = (sum_lag x_lag * (static_lag [+] dyn_lag) [+] x_0) / 3, dyn_lag = the row's tap of head g (dyn
//            projection, shifted by o, column-broadcast by a 0/1 selector matmul); [+] = FPU eltwise adds as ttnn's
//            adds of those tensors ran (BF16 + BF16 -> FP32 taps; the FP32 sum, packed and read through the TF32
//            source register, + the BF16 row), each in its own DEST section
//   (fp32 on the SFPU, one BF16 rounding), then per row over the head's 128 channels, ss = ttnn.sum(ttnn.mul(y, y,
//   fp32)) (SFPU squares, the FPU row reduce of ttnn's REDUCE_ROW SUM):
//   k: * min(22.63 rsqrt(ss), 1);  v: * rsqrt(ss / 128 + eps)   (SFPU, column broadcast multiply)
// rounded to BF16 (RNE, as every FP32 -> BF16 result of those ops), then (K_RT / V_RT: block-float caches) to the cache
// format, as ttnn's typecast of the BF16 rows.
// Tile 0 rows < o keep the cache's existing rows (factor rows < o zeroed, + E_lt @ old).
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "attn_prefill_sfpu.hpp"

constexpr uint32_t Tpt = get_compile_time_arg_val(0);
constexpr bool SLIDING = get_compile_time_arg_val(1) != 0;
constexpr uint32_t cb_x = get_compile_time_arg_val(2);
constexpr uint32_t cb_tap = get_compile_time_arg_val(3);
constexpr uint32_t cb_old = get_compile_time_arg_val(4);
constexpr uint32_t cb_oldv = get_compile_time_arg_val(5);
constexpr uint32_t cb_dyn = get_compile_time_arg_val(6);
constexpr uint32_t cb_m = get_compile_time_arg_val(7);
constexpr uint32_t cb_p = get_compile_time_arg_val(8);
constexpr uint32_t cb_e0 = get_compile_time_arg_val(9);   // shifted dyn rows (BF16)
constexpr uint32_t cb_e = get_compile_time_arg_val(10);   // expanded dyn taps, 4 lags (BF16)
constexpr uint32_t cb_y = get_compile_time_arg_val(11);   // conv output (BF16, 4)
constexpr uint32_t cb_sq = get_compile_time_arg_val(12);  // squares (FP32, 4)
constexpr uint32_t cb_fr = get_compile_time_arg_val(13);  // factor column (FP32, unpacked to DEST)
constexpr uint32_t cb_r16 = get_compile_time_arg_val(14); // BF16 rounding scratch (block-float outputs)
constexpr uint32_t cb_one = get_compile_time_arg_val(15); // reduce scaler (1.0)
constexpr uint32_t cb_ok = get_compile_time_arg_val(16);  // output, K format
constexpr uint32_t cb_ov = get_compile_time_arg_val(17);  // output, V format
constexpr uint32_t K_CLIP = get_compile_time_arg_val(18); // fp32 bits 22.63
constexpr uint32_t V_INV = get_compile_time_arg_val(19);  // 1 / 128
constexpr uint32_t V_EPS = get_compile_time_arg_val(20);
constexpr uint32_t C_SCALE = get_compile_time_arg_val(21);  // full conv scale 1 / 3
constexpr bool K_RT = get_compile_time_arg_val(22) != 0;    // K format is not BF16: round to BF16 first
constexpr bool V_RT = get_compile_time_arg_val(23) != 0;
constexpr uint32_t cb_w = get_compile_time_arg_val(24);    // full: taps static + dyn (FP32, unpacked to DEST)
constexpr uint32_t cb_acc = get_compile_time_arg_val(25);  // full: the lag sum (FP32, FPU source)
constexpr uint32_t cb_x0 = get_compile_time_arg_val(26);   // full: the lag-0 rows (BF16)
constexpr uint32_t kTiny = 0x0DA24260u;                     // 1e-30
constexpr uint32_t kOneF = 0x3F800000u;
// matrices in cb_m (reader-built BF16 0/1 tiles)
constexpr uint32_t kSlo = 4, kRo = 8, kElt = 9, kSupO = 10, kSdnO = 11, kJ = 12;
constexpr uint32_t kMats = SLIDING ? 12 : 44;

void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_m, cb_x, cb_y);
    cb_wait_front(cb_p, 1);
    const volatile uint32_t* pw = reinterpret_cast<const volatile uint32_t*>(get_tile_address(cb_p, 0));
    const uint32_t o = pw[1];
    cb_wait_front(cb_m, kMats);
    cb_wait_front(cb_one, 1);

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t gk = u / (Tpt + 1);
        const uint32_t ip = u % (Tpt + 1);
        const uint32_t kv = gk >> 3, g = gk & 7;
        if (ip == Tpt && o == 0) {
            continue;
        }
        if constexpr (!SLIDING) {
            // dyn rows shifted by o (row j' <- chunk row 32 i' + j' - o), then the 4 lag selectors of head g
            cb_wait_front(cb_dyn, 2);
            reconfig_data_format(cb_dyn, cb_m);
            matmul_init(cb_m, cb_dyn, 0);
            tile_regs_acquire();
            matmul_tiles(cb_m, cb_dyn, kSupO, 1, 0);
            if (o != 0) {
                matmul_tiles(cb_m, cb_dyn, kSdnO, 0, 0);
            }
            tile_regs_commit();
            cb_reserve_back(cb_e0, 1);
            pack_reconfig_data_format(cb_e0);
            tile_regs_wait();
            pack_tile(0, cb_e0);
            tile_regs_release();
            cb_push_back(cb_e0, 1);
            cb_pop_front(cb_dyn, 2);
            cb_wait_front(cb_e0, 1);
            reconfig_data_format(cb_m, cb_e0);
            matmul_init(cb_e0, cb_m, 0);
            tile_regs_acquire();
            for (uint32_t l = 0; l < 4; ++l) {
                matmul_tiles(cb_e0, cb_m, 0, kJ + g * 4 + l, l);
            }
            tile_regs_commit();
            cb_reserve_back(cb_e, 4);
            pack_reconfig_data_format(cb_e);
            tile_regs_wait();
            for (uint32_t l = 0; l < 4; ++l) {
                pack_tile(l, cb_e);
            }
            tile_regs_release();
            cb_push_back(cb_e, 4);
            cb_pop_front(cb_e0, 1);
            cb_wait_front(cb_e, 4);
        }
        // ---- conv per column tile -> cb_y (BF16)
        cb_reserve_back(cb_y, 4);
        for (uint32_t c = 0; c < 4; ++c) {
            cb_wait_front(cb_x, 3);
            cb_wait_front(cb_tap, 4);
            if constexpr (!SLIDING) {  // w_lag = static_lag [+] dyn_lag -> cb_w (FP32)
                tile_regs_acquire();
                reconfig_data_format(cb_tap, cb_e);
                add_tiles_init(cb_tap, cb_e);
                for (uint32_t l = 0; l < 4; ++l) {
                    add_tiles(cb_tap, cb_e, l, l, l);
                }
                tile_regs_commit();
                cb_reserve_back(cb_w, 4);
                pack_reconfig_data_format(cb_w);
                tile_regs_wait();
                for (uint32_t l = 0; l < 4; ++l) {
                    pack_tile(l, cb_w);
                }
                tile_regs_release();
                cb_push_back(cb_w, 4);
                cb_wait_front(cb_w, 4);
            }
            tile_regs_acquire();
            reconfig_data_format(cb_x, cb_m);
            matmul_init(cb_m, cb_x, 0);
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t uu = o + l;
                const uint32_t q = uu >> 5;
                matmul_tiles(cb_m, cb_x, l, 2 - q, l);
                if ((uu & 31) != 0) {
                    matmul_tiles(cb_m, cb_x, kSlo + l, 1 - q, l);
                }
            }
            {
                constexpr uint32_t cbt = SLIDING ? cb_tap : cb_w;  // FP32, unpacked to DEST (exact)
                reconfig_data_format_srca(cbt);
                copy_tile_to_dst_init_short(cbt);
                for (uint32_t l = 0; l < 4; ++l) {
                    copy_tile(cbt, l, 4 + l);
                }
            }
            mul_binary_tile_init();
            mul_binary_tile(3, 7, 3);
            mul_binary_tile(2, 6, 2);
            add_binary_tile_init();
            add_binary_tile(3, 2, 3);
            mul_binary_tile_init();
            mul_binary_tile(1, 5, 1);
            add_binary_tile_init();
            add_binary_tile(3, 1, 3);
            mul_binary_tile_init();
            if constexpr (SLIDING) {
                mul_binary_tile(0, 4, 0);
                add_binary_tile_init();
                add_binary_tile(3, 0, 3);
                pa::rne16_init();
                pa::rne16(3);
                tile_regs_commit();
                pack_reconfig_data_format(cb_y);
                tile_regs_wait();
                pack_tile<true>(3, cb_y, c);
                tile_regs_release();
            } else {
                mul_binary_tile(0, 4, 4);
                add_binary_tile_init();
                add_binary_tile(3, 4, 3);
                tile_regs_commit();
                cb_pop_front(cb_w, 4);
                cb_reserve_back(cb_acc, 1);
                cb_reserve_back(cb_x0, 1);
                tile_regs_wait();
                pack_reconfig_data_format(cb_acc);
                pack_tile(3, cb_acc);
                pack_reconfig_data_format(cb_x0);
                pack_tile(0, cb_x0);
                tile_regs_release();
                cb_push_back(cb_acc, 1);
                cb_push_back(cb_x0, 1);
                cb_wait_front(cb_acc, 1);
                cb_wait_front(cb_x0, 1);
                tile_regs_acquire();
                reconfig_data_format(cb_acc, cb_x0);
                add_tiles_init(cb_acc, cb_x0);
                add_tiles(cb_acc, cb_x0, 0, 0, 0);
                binop_with_scalar_tile_init();
                mul_unary_tile(0, C_SCALE);
                pa::rne16_init();
                pa::rne16(0);
                tile_regs_commit();
                pack_reconfig_data_format(cb_y);
                tile_regs_wait();
                pack_tile<true>(0, cb_y, c);
                tile_regs_release();
                cb_pop_front(cb_acc, 1);
                cb_pop_front(cb_x0, 1);
            }
            cb_pop_front(cb_x, 3);
            cb_pop_front(cb_tap, 4);
        }
        cb_push_back(cb_y, 4);
        if constexpr (!SLIDING) {
            cb_pop_front(cb_e, 4);
        }
        // ---- squares (SFPU, exact) -> cb_sq
        cb_wait_front(cb_y, 4);
        reconfig_data_format_srca(cb_y);
        copy_tile_to_dst_init_short(cb_y);
        tile_regs_acquire();
        for (uint32_t c = 0; c < 4; ++c) {
            copy_tile(cb_y, c, c);
        }
        mul_binary_tile_init();
        for (uint32_t c = 0; c < 4; ++c) {
            mul_binary_tile(c, c, c);
        }
        tile_regs_commit();
        cb_reserve_back(cb_sq, 4);
        pack_reconfig_data_format(cb_sq);
        tile_regs_wait();
        for (uint32_t c = 0; c < 4; ++c) {
            pack_tile(c, cb_sq);
        }
        tile_regs_release();
        cb_push_back(cb_sq, 4);
        // ---- row reduce (ttnn REDUCE_ROW SUM) + factor -> cb_fr (column 0)
        cb_wait_front(cb_sq, 4);
        reconfig_data_format(cb_one, cb_sq);
        pack_reconfig_data_format(cb_fr);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_one, cb_fr);
        tile_regs_acquire();
        for (uint32_t c = 0; c < 4; ++c) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_one, c, 0, 0);
        }
        reduce_uninit();
        if (kv == 0) {
            unary_max_tile_init();
            unary_max_tile(0, kTiny);
            rsqrt_tile_init();
            rsqrt_tile(0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, K_CLIP);
            unary_min_tile_init();
            unary_min_tile(0, kOneF);
        } else {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, V_INV);
            add_unary_tile(0, V_EPS);
            rsqrt_tile_init();
            rsqrt_tile(0);
        }
        const bool merge = ip == 0 && o != 0;
        if (merge) {  // rows < o: factor 0 (the old rows come from E_lt @ old)
            reconfig_data_format_srca(cb_m);
            copy_tile_to_dst_init_short(cb_m);
            copy_tile(cb_m, kRo, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        cb_pop_front(cb_sq, 4);
        cb_reserve_back(cb_fr, 1);
        tile_regs_wait();
        pack_tile(0, cb_fr);
        tile_regs_release();
        cb_push_back(cb_fr, 1);
        cb_wait_front(cb_fr, 1);
        // ---- out_c = Y_c * f (+ E_lt @ old_c), BF16 (then the cache format)
        const uint32_t cbo = kv == 0 ? cb_ok : cb_ov;
        const uint32_t cbold = kv == 0 ? cb_old : cb_oldv;
        const bool rt = kv == 0 ? K_RT : V_RT;
        cb_reserve_back(cbo, 4);
        for (uint32_t c = 0; c < 4; ++c) {
            tile_regs_acquire();
            if (merge) {
                cb_wait_front(cbold, 1);
                reconfig_data_format(cbold, cb_m);
                matmul_init(cb_m, cbold, 0);
                matmul_tiles(cb_m, cbold, kElt, 0, 2);
            }
            reconfig_data_format_srca(cb_y);
            copy_tile_to_dst_init_short(cb_y);
            copy_tile(cb_y, c, 0);
            reconfig_data_format_srca(cb_fr);
            copy_tile_to_dst_init_short(cb_fr);
            copy_tile(cb_fr, 0, 1);
            pa::mul_col(0, 1);
            if (merge) {
                add_binary_tile_init();
                add_binary_tile(0, 2, 0);
            }
            pa::rne16_init();
            pa::rne16(0);
            tile_regs_commit();
            if (rt) {
                cb_reserve_back(cb_r16, 1);
                pack_reconfig_data_format(cb_r16);
                tile_regs_wait();
                pack_tile(0, cb_r16);
                tile_regs_release();
                cb_push_back(cb_r16, 1);
                cb_wait_front(cb_r16, 1);
                tile_regs_acquire();
                reconfig_data_format_srca(cb_r16);
                copy_tile_to_dst_init_short(cb_r16);
                copy_tile(cb_r16, 0, 0);
                tile_regs_commit();
                cb_pop_front(cb_r16, 1);
            }
            pack_reconfig_data_format(cbo);
            tile_regs_wait();
            pack_tile<true>(0, cbo, c);
            tile_regs_release();
            if (merge) {
                cb_pop_front(cbold, 1);
            }
        }
        cb_push_back(cbo, 4);
        cb_pop_front(cb_fr, 1);
        cb_pop_front(cb_y, 4);
    }
}
