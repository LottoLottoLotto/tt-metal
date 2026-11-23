// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 gated short conv, prefill middle (between in_proj and out_proj), one launch per chunk:
//
//   Bx[t] = B[t] * x[t];  y[t] = C[t] * (t2 * Bx[t] + t1 * Bx[t-1] + t0 * Bx[t-2])
//
// with Bx[-1], Bx[-2] = the slot's conv state rows s1, s0 (zeros for a fresh sequence), and the state rows replaced by
// Bx[valid-1], Bx[valid-2] of the extended sequence [s0, s1, Bx...] (bitwise ShortConv.prefill: the same LLK ops,
// operand order, unpack modes and BF16 pack points as tt/fast/kernels/conv_decode.cpp, which is bitwise the ttnn op
// chain; the shifted operands Bx[t-1], Bx[t-2] are exact row copies).
//
// Core = hidden column tiles [col0, col0 + per); per column the row tiles r = 0 .. Tt-1 run in order.
// Reader (NCRISC): taps (rows replicated) once per column, then B, x, C tiles of every row tile.
// Compute: Bx -> (compute, writer); acc = Bx*t2; acc += P1*t1; acc += P2*t0; y = C*acc.
// Writer (BRISC): builds P1 = rows shifted down by one, P2 = by two, from the previous Bx tile (initially a virtual
// tile whose rows 30, 31 hold s0, s1) and the current one (face-row copies within L1), writes y, and at the end of the
// column read-modify-writes the slot's row of the s0 / s1 tiles.
#include <cstdint>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace conv_prefill {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kHiddenTiles = 64;  // 2048 / 32
constexpr uint32_t kB = 0, kX = 1, kT2 = 2, kP1 = 3, kT1 = 4, kP2 = 5, kT0 = 6, kC = 7;
constexpr uint32_t kBx = 16, kBxW = 17, kAcc = 18, kAcc2 = 19, kAcc3 = 20, kY = 21;
constexpr uint32_t kScratch = 8;  // writer: prev tile, s0 tile, s1 tile
}  // namespace conv_prefill

#if defined(COMPILE_FOR_TRISC)
namespace conv_prefill {
FORCE_INLINE void sfpu_mul(uint32_t lhs, uint32_t rhs, uint32_t out, uint32_t out2, bool pop_rhs) {
    mul_binary_tile_init();
    cb_wait_front(lhs, 1);
    cb_wait_front(rhs, 1);
    cb_reserve_back(out, 1);
    if (out2 != out) {
        cb_reserve_back(out2, 1);
    }
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(rhs, lhs);
    copy_tile(lhs, 0, 0);
    copy_tile_to_dst_init_short_with_dt(lhs, rhs);
    copy_tile(rhs, 0, 1);
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    if (out2 != out) {
        pack_tile(0, out2);
    }
    tile_regs_release();
    cb_push_back(out, 1);
    if (out2 != out) {
        cb_push_back(out2, 1);
    }
    cb_pop_front(lhs, 1);
    if (pop_rhs) {
        cb_pop_front(rhs, 1);
    }
}

FORCE_INLINE void addcmul(uint32_t a, uint32_t b, uint32_t c, uint32_t out) {
    cb_wait_front(a, 1);
    cb_wait_front(b, 1);
    cb_wait_front(c, 1);
    cb_reserve_back(out, 1);
    tile_regs_acquire();
    copy_tile_init(a);
    copy_tile(a, 0, 0);
    copy_tile_init(b);
    copy_tile(b, 0, 1);
    copy_tile_init(c);
    copy_tile(c, 0, 2);
    addcmul_tile_init();
    addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, 0x3f800000u);  // value = 1.0f
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    cb_pop_front(a, 1);
    cb_pop_front(b, 1);  // c (a tap) stays for the column
}
}  // namespace conv_prefill
#endif

void kernel_main() {
    using namespace conv_prefill;
    constexpr uint32_t per = PER_CORE;
    constexpr uint32_t row_tiles = ROW_TILES;
    const uint32_t col0 = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_NCRISC)
    const auto bcx = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0),
                                    kTileBytes);
    const auto taps = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                     get_common_arg_val<uint32_t>(1), kTileBytes);
    for (uint32_t i = 0; i < per; ++i) {
        const uint32_t j = col0 + i;
        cb_reserve_back(kT0, 1);
        cb_reserve_back(kT1, 1);
        cb_reserve_back(kT2, 1);
        noc_async_read(get_noc_addr(j, taps), get_write_ptr(kT0), kTileBytes);
        noc_async_read(get_noc_addr(kHiddenTiles + j, taps), get_write_ptr(kT1), kTileBytes);
        noc_async_read(get_noc_addr(2 * kHiddenTiles + j, taps), get_write_ptr(kT2), kTileBytes);
        noc_async_read_barrier();
        cb_push_back(kT0, 1);
        cb_push_back(kT1, 1);
        cb_push_back(kT2, 1);
        for (uint32_t r = 0; r < row_tiles; ++r) {
            const uint32_t row = r * 3 * kHiddenTiles;
            cb_reserve_back(kB, 1);
            cb_reserve_back(kX, 1);
            cb_reserve_back(kC, 1);
            noc_async_read(get_noc_addr(row + j, bcx), get_write_ptr(kB), kTileBytes);
            noc_async_read(get_noc_addr(row + 2 * kHiddenTiles + j, bcx), get_write_ptr(kX), kTileBytes);
            noc_async_read(get_noc_addr(row + kHiddenTiles + j, bcx), get_write_ptr(kC), kTileBytes);
            noc_async_read_barrier();
            cb_push_back(kB, 1);
            cb_push_back(kX, 1);
            cb_push_back(kC, 1);
        }
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto s0 = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0),
                                   kTileBytes);
    const auto s1 = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1),
                                   kTileBytes);
    const auto y = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2),
                                  kTileBytes);
    const uint32_t valid = get_common_arg_val<uint32_t>(3);
    const uint32_t slot = get_common_arg_val<uint32_t>(4);
    const bool fresh = get_common_arg_val<uint32_t>(5) != 0;
    // Face-row geometry: tile = 4 faces of 16 x 16 BF16 (512 B), row = 32 B; tile row t lives in faces
    // (t / 16) * 2 + {0, 1}, face row t % 16.
    auto row_off = [](uint32_t t, uint32_t half) { return ((t / 16) * 2 + half) * 512 + (t % 16) * 32; };
    const uint32_t prev = get_write_ptr(kScratch);
    const uint32_t st0 = prev + kTileBytes, st1 = st0 + kTileBytes;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    auto local = [&](uint32_t addr) { return get_noc_addr(my_x[noc_index], my_y[noc_index], addr); };
    auto copy_row = [&](uint32_t src_tile, uint32_t src_row, uint32_t dst_tile, uint32_t dst_row) {
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_read(local(src_tile + row_off(src_row, h)), dst_tile + row_off(dst_row, h), 32);
        }
    };
    // dst rows [d, d + n) <- src rows [s, s + n) within one face row band (no 16-row crossing)
    auto copy_rows = [&](uint32_t src_tile, uint32_t s, uint32_t dst_tile, uint32_t d, uint32_t n) {
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_read(local(src_tile + row_off(s, h)), dst_tile + row_off(d, h), n * 32);
        }
    };
    const uint32_t last_tile = (valid - 1) / 32;
    for (uint32_t i = 0; i < per; ++i) {
        const uint32_t j = col0 + i;
        // virtual previous tile: rows 30, 31 = s0[slot], s1[slot] (other rows unused)
        noc_async_read(get_noc_addr(j, s0), st0, kTileBytes);
        noc_async_read(get_noc_addr(j, s1), st1, kTileBytes);
        noc_async_read_barrier();
        if (fresh) {
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_read(zeros, prev + row_off(30, h), 32);
                noc_async_read(zeros, prev + row_off(31, h), 32);
            }
        } else {
            copy_row(st0, slot, prev, 30);
            copy_row(st1, slot, prev, 31);
        }
        noc_async_read_barrier();
        for (uint32_t r = 0; r < row_tiles; ++r) {
            cb_wait_front(kBxW, 1);
            const uint32_t cur = get_read_ptr(kBxW);
            cb_reserve_back(kP1, 1);
            cb_reserve_back(kP2, 1);
            const uint32_t p1 = get_write_ptr(kP1), p2 = get_write_ptr(kP2);
            // P1: row t <- Bx[t - 1]
            copy_row(prev, 31, p1, 0);
            copy_rows(cur, 0, p1, 1, 15);
            copy_row(cur, 15, p1, 16);
            copy_rows(cur, 16, p1, 17, 15);
            // P2: row t <- Bx[t - 2]
            copy_rows(prev, 30, p2, 0, 2);
            copy_rows(cur, 0, p2, 2, 14);
            copy_rows(cur, 14, p2, 16, 2);
            copy_rows(cur, 16, p2, 18, 14);
            if (r == last_tile) {
                // new state: s1 <- Bx[valid - 1], s0 <- Bx[valid - 2] (Bx[-1] = prev row 31 = old s1 when valid == 1)
                const uint32_t t1 = (valid - 1) % 32;
                copy_row(cur, t1, st1, slot);
                if (t1 >= 1) {
                    copy_row(cur, t1 - 1, st0, slot);
                } else {
                    copy_row(prev, 31, st0, slot);
                }
            }
            noc_async_read_barrier();
            cb_push_back(kP1, 1);
            cb_push_back(kP2, 1);
            noc_async_read(local(cur), prev, kTileBytes);
            noc_async_read_barrier();
            cb_pop_front(kBxW, 1);
            cb_wait_front(kY, 1);
            noc_async_write(get_read_ptr(kY), get_noc_addr(r * kHiddenTiles + j, y), kTileBytes);
            noc_async_writes_flushed();
            cb_pop_front(kY, 1);
        }
        noc_async_write(st0, get_noc_addr(j, s0), kTileBytes);
        noc_async_write(st1, get_noc_addr(j, s1), kTileBytes);
        noc_async_write_barrier();
    }
#elif defined(COMPILE_FOR_TRISC)
    unary_op_init_common(kB, kY);
    for (uint32_t i = 0; i < per; ++i) {
        for (uint32_t r = 0; r < row_tiles; ++r) {
            sfpu_mul(kB, kX, kBx, kBxW, true);
            sfpu_mul(kBx, kT2, kAcc, kAcc, false);
            addcmul(kAcc, kP1, kT1, kAcc2);
            addcmul(kAcc2, kP2, kT0, kAcc3);
            sfpu_mul(kC, kAcc3, kY, kY, true);
        }
        cb_pop_front(kT0, 1);
        cb_pop_front(kT1, 1);
        cb_pop_front(kT2, 1);
    }
#endif
}
