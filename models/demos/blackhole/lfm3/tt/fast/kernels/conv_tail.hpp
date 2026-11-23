// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 gated short conv (7 taps, 6 state rows), decode tail shared by conv_decode.cpp and conv_fused.cpp.
//
// Stock arithmetic (layers.ShortConv.decode / _taps_fp32), reproduced bit for bit in an FP32-DEST kernel:
//
//   Bx   = rne(B * x)                    ttnn.mul BF16: SFPU mul, 16-bit DEST variant (RNE to BF16, 0 * v = +0)
//   acc  = s0 * t0                       ttnn.mul(dtype=float32): SFPU mul, FP32 DEST (exact BF16 x BF16 product)
//   acc  = acc + s_j * t_j, j = 1..5     ttnn.mul(float32) + ttnn.add(float32): SFPU mul, SFPU add (FP32)
//   acc  = acc + Bx * t6
//   conv = rne(acc)                      ttnn.typecast float32 -> bfloat16 (RNE + mask, typecast_tile<0, 5>)
//   y    = rne(C * conv)                 ttnn.mul BF16 (16-bit DEST variant)
//   s_j <- s_{j+1} (j < 5), s5 <- Bx     state shift (in place, DRAM)
//
// Every op is the same SFPU function the ttnn op runs (calculate_sfpu_binary_mul with its 16-bit / 32-bit DEST
// template argument, calculate_sfpu_binary<ADD>, the typecast's RNE), one op per SFPU pass, so the FP32 adds see
// exactly the stock operands; the BF16 results are stored in FP32 DEST as exact BF16 values (low 16 bits zero), so
// the BF16 pack is exact.  The state part (j = 0..5) does not depend on this step's in_proj output: it is computed
// first (`state_partial`, FP32 stash tile per column), the rest (`finish`) once B / C / x are in.
//
// CB layouts (one core, `per` hidden column tiles i): taps t0..t5 at i * 6 + j, state s0..s5 at i * 6 + j, t6 / B /
// C / x / acc / y / Bx at i.  Only tile rows [0, R) are meaningful (R = the decode rows; the DM side moves 64 B
// row granules, i.e. R rounded up to even within each face pair).
#pragma once

#include <cstdint>

namespace conv_tail {
constexpr uint32_t kTaps = 7;          // conv_L_cache
constexpr uint32_t kState = kTaps - 1;  // state rows per slot
constexpr uint32_t kTile = 2048;        // BF16 32x32 tile bytes
}  // namespace conv_tail

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace conv_tail {
// Row granules of a 32x32 BF16 tile holding rows [0, R): rows [0, min(R, 16)) of faces 0/1, [0, R - 16) of faces
// 2/3, each rounded up to 64 B (2 rows).
template <uint32_t R>
struct Rows {
    static_assert(R >= 1 && R <= 32);
    static constexpr uint32_t top = R < 16 ? R : 16;
    static constexpr uint32_t bottom = R > 16 ? R - 16 : 0;
    static constexpr uint32_t c_top = (top * 32 + 63) / 64 * 64;
    static constexpr uint32_t c_bot = (bottom * 32 + 63) / 64 * 64;

    template <typename F>
    static FORCE_INLINE void chunks(F&& f) {
        f(0u, c_top);
        f(512u, c_top);
        if constexpr (bottom > 0) {
            f(1024u, c_bot);
            f(1536u, c_bot);
        }
    }
    static FORCE_INLINE void read(uint64_t src, uint32_t dst) {
        chunks([&](uint32_t off, uint32_t n) { noc_async_read(src + off, dst + off, n); });
    }
    static FORCE_INLINE void write(uint32_t src, uint64_t dst) {
        chunks([&](uint32_t off, uint32_t n) { noc_async_write(src + off, dst + off, n); });
    }
};

// Byte offset of row r in face 0 (r < 16) or face 2 (r >= 16); the row's second half is at + 512.
constexpr uint32_t row_off(uint32_t r) { return (r / 16) * 1024 + (r % 16) * 32; }

FORCE_INLINE void copy_row(uint32_t dst_tile, uint32_t src_tile, uint32_t r) {
    for (uint32_t f = 0; f < 2; ++f) {
        auto* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + row_off(r) + f * 512);
        auto* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + row_off(r) + f * 512);
        for (uint32_t w = 0; w < 8; ++w) {
            d[w] = s[w];
        }
    }
}

// State shift of `per` columns: s_j <- old s_{j+1} (j < 5), s5 <- Bx, rows [0, R) per granule.  s_l1: the 6 * per
// state tiles as read (column-major: i * 6 + j); bx_l1: the per Bx tiles.  Keep (an odd R < the state's rows):
// state row R rides along in the last 64 B granule but belongs to a slot outside this step; its old values are put
// back first (descending, so every source row is still the old one).  The caller follows with a write barrier
// before reusing the L1 tiles.
template <uint32_t R, bool Keep, typename Acc>
FORCE_INLINE void shift_state(const Acc* s, uint32_t s_l1, uint32_t bx_l1, uint32_t col0, uint32_t per) {
    for (uint32_t i = 0; i < per; ++i) {
        const uint32_t base = s_l1 + i * kState * kTile;
        const uint32_t bx = bx_l1 + i * kTile;
        if constexpr (Keep) {
            copy_row(bx, base + (kState - 1) * kTile, R);
            for (uint32_t j = kState - 1; j >= 1; --j) {
                copy_row(base + j * kTile, base + (j - 1) * kTile, R);
            }
        }
        for (uint32_t j = 0; j + 1 < kState; ++j) {
            Rows<R>::write(base + (j + 1) * kTile, get_noc_addr(col0 + i, s[j]));
        }
        Rows<R>::write(bx, get_noc_addr(col0 + i, s[kState - 1]));
    }
}

// Speculative verification (conv_decode.cpp / conv_fused.cpp, verify mode): the R rows are groups of consecutive
// tokens, one group per sequence (a shared round: every stream's pending token + its drafts).  The row map
// (tt/speculate.py map_word, one u32 per row) gives row r its group offset k (bits 0-7), group length (8-15) and the
// sequence's conv-state slot sigma (16-23); bit 31 marks an active row.  Row r's left context is c[k + j] of
// c = [s0[sigma], ..., s5[sigma], Bx[r - k], Bx[r - k + 1], ...] (the state and this step's earlier rows of the same
// group), so state tile j's row r becomes raw tile (k + j) row sigma when k + j < 6, else Bx row r + j - 6; the tail
// arithmetic per row is then exactly the sequential 1-row step's.  raw_l1: the 6 * per state tiles as read (rows
// [0, R): every slot of the round, sigma < R); s_l1: the tiles TRISC consumes (built here; raw and Bx are only read).
// An inactive row copies its raw rows (finite values; its outputs are ignored).
constexpr uint32_t kMapActive = 1u << 31;

FORCE_INLINE void copy_row_from(uint32_t dst_tile, uint32_t dr, uint32_t src_tile, uint32_t sr) {
    for (uint32_t f = 0; f < 2; ++f) {
        auto* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + row_off(dr) + f * 512);
        auto* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + row_off(sr) + f * 512);
        for (uint32_t w = 0; w < 8; ++w) {
            d[w] = s[w];
        }
    }
}

template <uint32_t R>
FORCE_INLINE void verify_context(uint32_t s_l1, uint32_t raw_l1, uint32_t bx_l1, uint32_t per,
                                 const volatile tt_l1_ptr uint32_t* map) {
    static_assert(R >= 1 && R <= 16, "verify rows live in faces 0 / 1");
    for (uint32_t i = 0; i < per; ++i) {
        const uint32_t base = s_l1 + i * kState * kTile;
        const uint32_t raw = raw_l1 + i * kState * kTile;
        const uint32_t bx = bx_l1 + i * kTile;
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t m = map[r];
            const bool active = (m & kMapActive) != 0;
            const uint32_t k = m & 0xFFu, slot = (m >> 16) & 0xFFu;
            for (uint32_t j = 0; j < kState; ++j) {
                const uint32_t c = k + j;
                if (!active) {
                    copy_row_from(base + j * kTile, r, raw + j * kTile, r);
                } else if (c < kState) {
                    copy_row_from(base + j * kTile, r, raw + c * kTile, slot);
                } else {
                    copy_row_from(base + j * kTile, r, bx, r + j - kState);
                }
            }
        }
    }
}
}  // namespace conv_tail
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

#if defined(TRISC_MATH)
namespace ckernel::sfpu {
// ttnn.typecast float32 -> bfloat16 (calculate_typecast_fp32_to_fp16b: the same integer RNE + mask), in place.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void ct_rne_bf16(const uint dst_in0, const uint /*unused*/, const uint dst_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_in0 * dst_tile_size_sfpi];
        sfpi::dst_reg[dst_out * dst_tile_size_sfpi] = float32_to_bf16_rne(v);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace conv_tail {
// SFPU tail ops on DEST tiles (FP32 DEST, half sync: tiles 0..3).  V: the faces with valid rows.
template <VectorMode V>
FORCE_INLINE void mul_bf16(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
                           (APPROX, ckernel::BinaryOp::MUL, 8, false), a, b, o, V)));
}
template <VectorMode V>
FORCE_INLINE void mul_f32(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
                           (APPROX, ckernel::BinaryOp::MUL, 8, true), a, b, o, V)));
}
template <VectorMode V>
FORCE_INLINE void add_f32(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary, (APPROX, ckernel::BinaryOp::ADD, 8),
                           a, b, o, V)));
}
template <VectorMode V>
FORCE_INLINE void rne(uint32_t a) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, ct_rne_bf16, (APPROX, 8), a, a, a, V)));
}

// srcA unpack + datacopy for `cb` (unconditional: the UnpackToDestFp32 tail CBs and the Default-mode matmul /
// norm CBs share formats but not unpack configs).
FORCE_INLINE void copy_from(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
}

// acc_i = sum_{j<6} s_j * t_j (FP32, stock order) for the `per` columns -> cb_acc (FP32 tiles).
template <uint32_t per, VectorMode V>
FORCE_INLINE void state_partial(uint32_t cb_s, uint32_t cb_t, uint32_t cb_acc) {
    cb_wait_front(cb_s, per * kState);
    cb_wait_front(cb_t, per * kState);
    cb_reserve_back(cb_acc, per);
    copy_from(cb_s);
    pack_reconfig_data_format(cb_acc);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < per; ++i) {
        tile_regs_acquire();
        copy_tile(cb_s, i * kState, 0);
        copy_tile(cb_t, i * kState, 1);
        mul_f32<V>(0, 1, 0);
        for (uint32_t j = 1; j < kState; ++j) {
            copy_tile(cb_s, i * kState + j, 1);
            copy_tile(cb_t, i * kState + j, 2);
            mul_f32<V>(1, 2, 1);
            add_f32<V>(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_acc, i);
        tile_regs_release();
    }
    cb_push_back(cb_acc, per);
    cb_pop_front(cb_s, per * kState);
    cb_pop_front(cb_t, per * kState);
}

// Bx = rne(B * x) -> cb_bx; acc += Bx * t6; y = rne(C * rne(acc)) -> cb_y, for the `per` columns (cb_y / cb_bx
// BF16).  PackBx false (verify mode): Bx was already sent by bx_only; it is recomputed here (same op, same value)
// but not packed.
template <uint32_t per, VectorMode V, bool PackBx = true>
FORCE_INLINE void finish(uint32_t cb_b, uint32_t cb_x, uint32_t cb_c, uint32_t cb_t6,
                         uint32_t cb_acc, uint32_t cb_y, uint32_t cb_bx) {
    cb_wait_front(cb_b, per);
    cb_wait_front(cb_x, per);
    cb_wait_front(cb_c, per);
    cb_wait_front(cb_t6, per);
    cb_wait_front(cb_acc, per);
    cb_reserve_back(cb_y, per);
    if constexpr (PackBx) {
        cb_reserve_back(cb_bx, per);
    }
    pack_reconfig_data_format(cb_y);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < per; ++i) {
        tile_regs_acquire();
        copy_from(cb_b);
        copy_tile(cb_b, i, 0);
        copy_tile(cb_x, i, 1);
        mul_bf16<V>(0, 1, 0);  // Bx
        copy_tile(cb_t6, i, 1);
        mul_f32<V>(0, 1, 1);  // Bx * t6
        copy_from(cb_acc);
        copy_tile(cb_acc, i, 2);
        add_f32<V>(2, 1, 2);  // acc
        rne<V>(2);            // conv
        copy_from(cb_c);
        copy_tile(cb_c, i, 1);
        mul_bf16<V>(1, 2, 1);  // y = C * conv
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(1, cb_y, i);
        if constexpr (PackBx) {
            pack_tile<true>(0, cb_bx, i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_y, per);
    if constexpr (PackBx) {
        cb_push_back(cb_bx, per);
    }
    cb_pop_front(cb_b, per);
    cb_pop_front(cb_x, per);
    cb_pop_front(cb_c, per);
    cb_pop_front(cb_t6, per);
    cb_pop_front(cb_acc, per);
}

// Verify mode, before the state partial: Bx = rne(B * x) -> cb_bx (the DM side builds every row's left context from
// it); B / x stay in their CBs for finish.
template <uint32_t per, VectorMode V>
FORCE_INLINE void bx_only(uint32_t cb_b, uint32_t cb_x, uint32_t cb_bx) {
    cb_wait_front(cb_b, per);
    cb_wait_front(cb_x, per);
    cb_reserve_back(cb_bx, per);
    pack_reconfig_data_format(cb_bx);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < per; ++i) {
        tile_regs_acquire();
        copy_from(cb_b);
        copy_tile(cb_b, i, 0);
        copy_tile(cb_x, i, 1);
        mul_bf16<V>(0, 1, 0);  // Bx
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_bx, i);
        tile_regs_release();
    }
    cb_push_back(cb_bx, per);
}
}  // namespace conv_tail
#endif
