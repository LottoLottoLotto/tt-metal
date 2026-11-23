// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 gated short conv, decode tail (after in_proj), one launch:
//
//   Bx  = B * x                                  (ttnn.mul:     SFPU mul_binary_tile, BF16 pack)
//   acc = Bx * t2                                (ttnn.mul)
//   acc = acc + s1 * t1                          (ttnn.addcmul: SFPU addcmul_tile, BF16 pack)
//   acc = acc + s0 * t0                          (ttnn.addcmul)
//   y   = C * acc                                (ttnn.mul)
//   s0 <- s1, s1 <- Bx                           (conv state shift, in place)
//
// Same LLK calls, operand order, unpack modes and BF16 pack points as the ttnn op sequence in ShortConv.decode, so
// the result is bitwise identical (every op is an SFPU op on DST loaded by copy_tile, all formats BF16, so one
// unary_op_init_common serves the whole kernel).  One core owns `per` hidden column tiles j (B = bcx tile j,
// C = 64 + j, x = 128 + j; taps tensor [t0 | t1 | t2] with each tap row replicated over the 32 tile rows); a column's
// state tiles are read (one barrier) before that column's Bx exists, so the writer's in-place state updates never
// race a read.
//
// ROWS < 32: only tile rows [0, ROWS) are moved (ROWS even, <= 16: face 0/1 rows, 64 B aligned DRAM transfers); the
// rest of every input CB slot stays zero (cleared once from the core's zero page), so y's padding rows are exact
// zeros and the state's padding rows are never written.
//
// Reader (NCRISC) args: common [bcx, taps, s0, s1], per-core [first column].
// Writer (BRISC)  args: common [s0, s1, y],          per-core [first column].
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

namespace conv_decode {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kHiddenTiles = 64;  // 2048 / 32
// Reader -> compute (UnpackToDestFp32 for the SFPU multiplies' operands, Default for the FPU addcmuls').
constexpr uint32_t kB = 0;
constexpr uint32_t kX = 1;
constexpr uint32_t kT2 = 2;
constexpr uint32_t kS1 = 3;
constexpr uint32_t kT1 = 4;
constexpr uint32_t kS0 = 5;
constexpr uint32_t kT0 = 6;
constexpr uint32_t kC = 7;
constexpr uint32_t kPass = 8;  // reader -> writer: old s1 (the new s0)
// Compute intermediates / outputs.
constexpr uint32_t kBx = 16;    // Fp32 unpack (SFPU mul operand)
constexpr uint32_t kBxOut = 17; // -> writer (new s1)
constexpr uint32_t kAcc = 18;   // Default (addcmul addend)
constexpr uint32_t kAcc2 = 19;  // Default
constexpr uint32_t kAcc3 = 20;  // Fp32 unpack
constexpr uint32_t kY = 21;     // -> writer
}  // namespace conv_decode

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace conv_decode {
constexpr uint32_t kRows = ROWS;
constexpr bool kFull = kRows >= 32;
constexpr uint32_t kFaceBytes = kRows * 32;  // rows [0, ROWS) of face 0 (and of face 1 at +512)

template <typename Accessor>
FORCE_INLINE void read_rows(const Accessor& acc, uint32_t tile, uint32_t dst) {
    const uint64_t src = get_noc_addr(tile, acc);
    if constexpr (kFull) {
        noc_async_read(src, dst, kTileBytes);
    } else {
        noc_async_read(src, dst, kFaceBytes);
        noc_async_read(src + 512, dst + 512, kFaceBytes);
    }
}

template <typename Accessor>
FORCE_INLINE void write_rows(uint32_t src, const Accessor& acc, uint32_t tile) {
    const uint64_t dst = get_noc_addr(tile, acc);
    if constexpr (kFull) {
        noc_async_write(src, dst, kTileBytes);
    } else {
        noc_async_write(src, dst, kFaceBytes);
        noc_async_write(src + 512, dst + 512, kFaceBytes);
    }
}

FORCE_INLINE void clear_cb(uint32_t cb, uint32_t tiles) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    const uint32_t dst = get_write_ptr(cb);
    for (uint32_t off = 0; off < tiles * kTileBytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + off, MEM_ZEROS_SIZE);
    }
}
}  // namespace conv_decode
#endif

#if defined(COMPILE_FOR_TRISC)
namespace conv_decode {
// ttnn.mul (binary_ng SFPU path for BF16): lhs -> DST0, rhs -> DST1, mul_binary_tile, BF16 pack.
FORCE_INLINE void sfpu_mul(uint32_t lhs, uint32_t rhs, uint32_t out, uint32_t out2) {
    // (unary_op_init_common once in kernel_main: every op unpacks BF16 A -> DST and packs BF16)
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
    cb_pop_front(rhs, 1);
}

// ttnn.addcmul(a, b, c) for BF16 tiles (ternary_addc_ops_sfpu.cpp; the B > 1 row-broadcast variant only adds exact
// broadcast copies): a, b, c -> DST 0..2 (Default unpack), SFPU addcmul_tile with value 1.0f, BF16 pack.
FORCE_INLINE void addcmul(uint32_t a, uint32_t b, uint32_t c, uint32_t out) {
    // (unary_op_init_common once in kernel_main)
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
    cb_pop_front(b, 1);
    cb_pop_front(c, 1);
}
}  // namespace conv_decode
#endif

void kernel_main() {
    using namespace conv_decode;
    constexpr uint32_t per = PER_CORE;  // hidden column tiles per core
#if defined(COMPILE_FOR_NCRISC)
    const auto bcx = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0),
                                    kTileBytes);
    const auto taps = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                     get_common_arg_val<uint32_t>(1), kTileBytes);
    const auto s0 = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2),
                                   kTileBytes);
    const auto s1 = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3),
                                   kTileBytes);
    const uint32_t col0 = get_arg_val<uint32_t>(0);
    if constexpr (!kFull) {
        for (uint32_t cb = kB; cb <= kPass; ++cb) {
            clear_cb(cb, CB_TILES);
        }
        noc_async_read_barrier();  // before the row reads land in the same slots
    }
    for (uint32_t i = 0; i < per; ++i) {
        const uint32_t j = col0 + i;
        for (uint32_t cb = kB; cb <= kPass; ++cb) {
            cb_reserve_back(cb, 1);
        }
        read_rows(bcx, j, get_write_ptr(kB));
        read_rows(bcx, 2 * kHiddenTiles + j, get_write_ptr(kX));
        read_rows(bcx, kHiddenTiles + j, get_write_ptr(kC));
        read_rows(taps, 2 * kHiddenTiles + j, get_write_ptr(kT2));
        read_rows(taps, kHiddenTiles + j, get_write_ptr(kT1));
        read_rows(taps, j, get_write_ptr(kT0));
        read_rows(s1, j, get_write_ptr(kS1));
        read_rows(s1, j, get_write_ptr(kPass));
        read_rows(s0, j, get_write_ptr(kS0));
        noc_async_read_barrier();
#ifdef KEEP_ROW
        // An odd row count is rounded up to ROWS: state row KEEP_ROW (another slot of a wider state, not part of
        // this step) rides along in the 64 B write granules.  Keep it: kPass row KEEP_ROW + 1 (never written out)
        // <- old s1[KEEP_ROW] (the writer puts it back into the new s1), kPass row KEEP_ROW <- old s0[KEEP_ROW].
        for (uint32_t f = 0; f < 2; ++f) {
            auto* pass = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kPass) + f * 512 + KEEP_ROW * 32);
            auto* old0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kS0) + f * 512 + KEEP_ROW * 32);
            for (uint32_t w = 0; w < 8; ++w) {
                pass[8 + w] = pass[w];
                pass[w] = old0[w];
            }
        }
#endif
        for (uint32_t cb = kB; cb <= kPass; ++cb) {
            cb_push_back(cb, 1);
        }
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto s0 = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0),
                                   kTileBytes);
    const auto s1 = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1),
                                   kTileBytes);
    const auto y = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2),
                                  kTileBytes);
    const uint32_t col0 = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < per; ++i) {
        const uint32_t j = col0 + i;
        // y last: it depends on every state read of this column, which the state writes must follow.
        cb_wait_front(kY, 1);
        noc_async_write(get_read_ptr(kY), get_noc_addr(j, y), kTileBytes);
        cb_wait_front(kBxOut, 1);
        cb_wait_front(kPass, 1);
#ifdef KEEP_ROW
        for (uint32_t f = 0; f < 2; ++f) {
            auto* bx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(kBxOut) + f * 512 + KEEP_ROW * 32);
            auto* pass = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(kPass) + f * 512 + KEEP_ROW * 32);
            for (uint32_t w = 0; w < 8; ++w) {
                bx[w] = pass[8 + w];
            }
        }
#endif
        write_rows(get_read_ptr(kBxOut), s1, j);
        write_rows(get_read_ptr(kPass), s0, j);
        noc_async_write_barrier();
        cb_pop_front(kY, 1);
        cb_pop_front(kBxOut, 1);
        cb_pop_front(kPass, 1);
    }
#elif defined(COMPILE_FOR_TRISC)
    unary_op_init_common(kB, kY);
    for (uint32_t i = 0; i < per; ++i) {
        sfpu_mul(kB, kX, kBx, kBxOut);
        sfpu_mul(kBx, kT2, kAcc, kAcc);
        addcmul(kAcc, kS1, kT1, kAcc2);
        addcmul(kAcc2, kS0, kT0, kAcc3);
        sfpu_mul(kC, kAcc3, kY, kY);
    }
#endif
}
