// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 gated short conv, prefill middle (between in_proj and out_proj), one launch per chunk:
//
//   Bx[t] = B[t] * x[t];  conv[t] = sum_{j=0..6} taps[j] * Bx[t-6+j];  y[t] = C[t] * conv[t]
//
// with Bx[-6 .. -1] = the slot's conv state rows s0 .. s5 (zeros for a fresh sequence), and the state rows replaced
// by Bx[valid-6 .. valid-1] of the extended sequence [s0 .. s5, Bx...].  Bitwise ShortConv's ttnn op chain:
//   Bx = mul(B, x) (BF16 SFPU mul, 16-bit DEST: FP32 product rounded to BF16 (RNE), 0 * x = +0)
//   conv = typecast(sum_j mul(row_j, tap_j, fp32), bf16) (exact BF16xBF16 products, FP32 SFPU adds in tap order,
//          one RNE rounding to BF16 by the SFPU typecast)
//   y = mul(C, conv) (BF16 SFPU mul)
// computed here in a 32-bit DEST with the same SFPU functions (the BF16 muls call the 16-bit-DEST variant of the SFPU
// mul).  All CB operands unpack to DEST as FP32 (exact for BF16).  The shifted operands Bx[t-k] are exact row copies.
//
// Work = the 64 x Tt (hidden column tile, row tile) units in column-major order, cut into one contiguous range per core
// (all cores of the grid); a range is at most two segments (column j, row tiles [r0, r1)), run in order.  A segment
// with r0 > 0 first recomputes Bx of row tile r0 - 1 (its left context); one with r0 = 0 takes the slot's state.
// Reader (NCRISC): per segment the 7 taps (rows replicated), [B, x of row tile r0 - 1], then B, x, C of its row tiles.
// Compute: Bx(r+1) runs before conv(r) so the writer builds the shifted tiles of r+1 while compute does conv(r).
// Writer (BRISC): builds P_j (row t <- Bx[t-6+j], j = 0..5) from the previous Bx tile (for r0 = 0 a virtual tile whose
// rows 26..31 hold s0..s5) and the current one (face-row copies within L1), writes y, and the segment holding row
// valid - 1 read-modify-writes the slot's row of the six state tiles.  State reads (left context of r0 = 0 segments)
// all happen first, then a barrier (semaphore 0 on core 0, which then bumps semaphore 1 on every core): state writes
// wait for it, so no core reads a state row another core already replaced.
//
// Per-core args: [segments, (column, r0, r1) x 2, core index].  Writer common args: [s0..s5, y, valid, slot, fresh,
// cores, then (noc x, noc y) of core 0 .. cores - 1].
#include <cstdint>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace conv_prefill {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kHiddenTiles = 64;  // 2048 / 32
constexpr uint32_t kTaps = 7;
constexpr uint32_t kState = kTaps - 1;
constexpr uint32_t kB = 0, kX = 1, kC = 2, kT = 3, kP = 4;
constexpr uint32_t kBx = 16, kBxW = 17, kY = 18;
constexpr uint32_t kScratch = 8;  // writer: prev tile, virtual (state) tile, then the six state tiles
}  // namespace conv_prefill

#if defined(COMPILE_FOR_TRISC)
namespace conv_prefill {
constexpr uint32_t kF32 = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t kBf16 = static_cast<uint32_t>(DataFormat::Float16_b);

FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    copy_tile_init(cb);
    copy_tile(cb, tile, dst);
}

FORCE_INLINE void to_bf16(uint32_t dst) {
    typecast_tile_init<kF32, kBf16>();
    typecast_tile<kF32, kBf16>(dst);
}

// BF16 SFPU mul (binary_ng's 16-bit-DEST variant: RNE to BF16, 0 * x = +0) inside this 32-bit-DEST kernel
FORCE_INLINE void mul_bf16(uint32_t a, uint32_t b, uint32_t out) {
    mul_binary_tile_init();
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
                           (APPROX, ckernel::BinaryOp::MUL, 8, false), a, b, out, VectorMode::RC)));
}

// Bx = bf16(B * x) -> kBxW (writer's shift source) and, unless `pre` (the left-context tile of a segment), kBx
// (compute's tap-6 operand)
FORCE_INLINE void bx_tile(bool pre = false) {
    cb_wait_front(kB, 1);
    cb_wait_front(kX, 1);
    if (!pre) {
        cb_reserve_back(kBx, 1);
    }
    cb_reserve_back(kBxW, 1);
    tile_regs_acquire();
    load(kB, 0, 0);
    load(kX, 0, 1);
    mul_bf16(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    if (!pre) {
        pack_tile(0, kBx);
    }
    pack_tile(0, kBxW);
    tile_regs_release();
    if (!pre) {
        cb_push_back(kBx, 1);
    }
    cb_push_back(kBxW, 1);
    cb_pop_front(kB, 1);
    cb_pop_front(kX, 1);
}

// y = bf16(C * bf16(sum_j P_j * t_j + Bx * t_6))
FORCE_INLINE void conv_tile() {
    cb_wait_front(kP, kState);
    cb_wait_front(kBx, 1);
    cb_wait_front(kC, 1);
    cb_reserve_back(kY, 1);
    tile_regs_acquire();
    load(kP, 0, 0);
    load(kT, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    for (uint32_t j = 1; j < kTaps; ++j) {
        if (j < kState) {
            load(kP, j, 1);
        } else {
            load(kBx, 0, 1);
        }
        load(kT, j, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    to_bf16(0);
    load(kC, 0, 1);
    mul_bf16(1, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, kY);
    tile_regs_release();
    cb_push_back(kY, 1);
    cb_pop_front(kP, kState);
    cb_pop_front(kBx, 1);
    cb_pop_front(kC, 1);
}
}  // namespace conv_prefill
#endif

void kernel_main() {
    using namespace conv_prefill;
    const uint32_t nseg = get_arg_val<uint32_t>(0);
    auto seg_col = [](uint32_t s) { return get_arg_val<uint32_t>(1 + 3 * s); };
    auto seg_r0 = [](uint32_t s) { return get_arg_val<uint32_t>(2 + 3 * s); };
    auto seg_r1 = [](uint32_t s) { return get_arg_val<uint32_t>(3 + 3 * s); };
#if defined(COMPILE_FOR_NCRISC)
    const auto bcx = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0),
                                    kTileBytes);
    const auto taps = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                     get_common_arg_val<uint32_t>(1), kTileBytes);
    for (uint32_t sg = 0; sg < nseg; ++sg) {
        const uint32_t j = seg_col(sg), r0 = seg_r0(sg), r1 = seg_r1(sg);
        cb_reserve_back(kT, kTaps);
        const uint32_t t = get_write_ptr(kT);
        for (uint32_t k = 0; k < kTaps; ++k) {
            noc_async_read(get_noc_addr(k * kHiddenTiles + j, taps), t + k * kTileBytes, kTileBytes);
        }
        noc_async_read_barrier();
        cb_push_back(kT, kTaps);
        for (uint32_t r = r0 > 0 ? r0 - 1 : 0; r < r1; ++r) {
            const uint32_t row = r * 3 * kHiddenTiles;
            cb_reserve_back(kB, 1);
            cb_reserve_back(kX, 1);
            noc_async_read(get_noc_addr(row + j, bcx), get_write_ptr(kB), kTileBytes);
            noc_async_read(get_noc_addr(row + 2 * kHiddenTiles + j, bcx), get_write_ptr(kX), kTileBytes);
            noc_async_read_barrier();
            cb_push_back(kB, 1);
            cb_push_back(kX, 1);
            if (r < r0) {
                continue;  // left-context tile: B, x only
            }
            cb_reserve_back(kC, 1);
            noc_async_read(get_noc_addr(row + kHiddenTiles + j, bcx), get_write_ptr(kC), kTileBytes);
            noc_async_read_barrier();
            cb_push_back(kC, 1);
        }
    }
#elif defined(COMPILE_FOR_BRISC)
    auto dram = [](uint32_t addr) {
        return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), addr, kTileBytes);
    };
    using Acc = decltype(dram(0));
    const Acc st_acc[kState] = {dram(get_common_arg_val<uint32_t>(0)), dram(get_common_arg_val<uint32_t>(1)),
                                dram(get_common_arg_val<uint32_t>(2)), dram(get_common_arg_val<uint32_t>(3)),
                                dram(get_common_arg_val<uint32_t>(4)), dram(get_common_arg_val<uint32_t>(5))};
    uint32_t a = kState;
    const auto y = dram(get_common_arg_val<uint32_t>(a++));
    const uint32_t valid = get_common_arg_val<uint32_t>(a++);
    const uint32_t slot = get_common_arg_val<uint32_t>(a++);
    const bool fresh = get_common_arg_val<uint32_t>(a++) != 0;
    const uint32_t ncores = get_common_arg_val<uint32_t>(a++);
    const uint32_t table = a;  // (noc x, noc y) of core c at table + 2 c
    const uint32_t me = get_arg_val<uint32_t>(7);
    // Face-row geometry: tile = 4 faces of 16 x 16 BF16 (512 B), row = 32 B; tile row t lives in faces
    // (t / 16) * 2 + {0, 1}, face row t % 16.
    auto row_off = [](uint32_t t, uint32_t half) { return ((t / 16) * 2 + half) * 512 + (t % 16) * 32; };
    const uint32_t prev = get_write_ptr(kScratch);
    const uint32_t virt = prev + kTileBytes;
    uint32_t st[kState];
    for (uint32_t k = 0; k < kState; ++k) {
        st[k] = prev + (k + 2) * kTileBytes;
    }
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    auto local = [&](uint32_t addr) { return get_noc_addr(my_x[noc_index], my_y[noc_index], addr); };
    // dst rows [d, d + n) <- src rows [s, s + n), both runs within one 16-row band
    auto copy_rows = [&](uint32_t src_tile, uint32_t s, uint32_t dst_tile, uint32_t d, uint32_t n) {
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_read(local(src_tile + row_off(s, h)), dst_tile + row_off(d, h), n * 32);
        }
    };
    auto read_state = [&](uint32_t j) {
        for (uint32_t k = 0; k < kState; ++k) {
            noc_async_read(get_noc_addr(j, st_acc[k]), st[k], kTileBytes);
        }
        noc_async_read_barrier();
    };
    const uint32_t last_tile = (valid - 1) / 32;
    const uint32_t t_last = (valid - 1) % 32;
    // phase 0: the left context of a segment starting at row tile 0 (at most one per core): virtual previous tile,
    // rows 26..31 = s0[slot] .. s5[slot] (other rows unused); then the state-read barrier
    int32_t st_col = -1;  // column whose state tiles st[] hold
    for (uint32_t sg = 0; sg < nseg; ++sg) {
        if (seg_r0(sg) != 0) {
            continue;
        }
        const uint32_t j = seg_col(sg);
        read_state(j);
        st_col = static_cast<int32_t>(j);
        for (uint32_t k = 0; k < kState; ++k) {
            if (fresh) {
                for (uint32_t h = 0; h < 2; ++h) {
                    noc_async_read(zeros, virt + row_off(26 + k, h), 32);
                }
            } else {
                copy_rows(st[k], slot, virt, 26 + k, 1);
            }
        }
        noc_async_read_barrier();
    }
    const uint32_t sem_in = get_semaphore(0), sem_go = get_semaphore(1);
    auto core_noc = [&](uint32_t c, uint32_t addr) {
        return get_noc_addr(get_common_arg_val<uint32_t>(table + 2 * c), get_common_arg_val<uint32_t>(table + 2 * c + 1),
                            addr);
    };
    noc_semaphore_inc(core_noc(0, sem_in), 1);
    if (me == 0) {
        volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_in);
        noc_semaphore_wait(in, ncores);
        noc_semaphore_set(in, 0);
        for (uint32_t c = 0; c < ncores; ++c) {
            noc_semaphore_inc(core_noc(c, sem_go), 1);
        }
    }
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go);
    for (uint32_t sg = 0; sg < nseg; ++sg) {
        const uint32_t j = seg_col(sg), r0 = seg_r0(sg), r1 = seg_r1(sg);
        if (r0 == 0) {
            noc_async_read(local(virt), prev, kTileBytes);
            noc_async_read_barrier();
        } else {  // Bx of row tile r0 - 1 (compute's pre tile)
            cb_wait_front(kBxW, 1);
            noc_async_read(local(get_read_ptr(kBxW)), prev, kTileBytes);
            noc_async_read_barrier();
            cb_pop_front(kBxW, 1);
        }
        const bool has_state = last_tile >= r0 && last_tile < r1;
        if (has_state && st_col != static_cast<int32_t>(j)) {
            read_state(j);  // read-modify-write: only row `slot` changes, no other core touches these tiles now
            st_col = static_cast<int32_t>(j);
        }
        for (uint32_t r = r0; r < r1; ++r) {
            cb_wait_front(kBxW, 1);
            const uint32_t cur = get_read_ptr(kBxW);
            cb_reserve_back(kP, kState);
            const uint32_t p = get_write_ptr(kP);
            // P_j (shift s = 6 - j): row t <- Bx[t - s]
            for (uint32_t jj = 0; jj < kState; ++jj) {
                const uint32_t s = kState - jj;
                const uint32_t dst = p + jj * kTileBytes;
                copy_rows(prev, 32 - s, dst, 0, s);
                copy_rows(cur, 0, dst, s, 16 - s);
                copy_rows(cur, 16 - s, dst, 16, s);
                copy_rows(cur, 16, dst, 16 + s, 16 - s);
            }
            if (r == last_tile) {
                // new state: s_k <- Bx[valid - 6 + k] (rows before this tile come from prev: the previous Bx tile or
                // the virtual tile of the old state)
                for (uint32_t k = 0; k < kState; ++k) {
                    const int32_t idx = static_cast<int32_t>(t_last) - static_cast<int32_t>(kState - 1 - k);
                    if (idx >= 0) {
                        copy_rows(cur, idx, st[k], slot, 1);
                    } else {
                        copy_rows(prev, 32 + idx, st[k], slot, 1);
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(kP, kState);
            noc_async_read(local(cur), prev, kTileBytes);
            noc_async_read_barrier();
            cb_pop_front(kBxW, 1);
            if (r > r0) {
                cb_wait_front(kY, 1);
                noc_async_write(get_read_ptr(kY), get_noc_addr((r - 1) * kHiddenTiles + j, y), kTileBytes);
                noc_async_writes_flushed();
                cb_pop_front(kY, 1);
            }
        }
        cb_wait_front(kY, 1);
        noc_async_write(get_read_ptr(kY), get_noc_addr((r1 - 1) * kHiddenTiles + j, y), kTileBytes);
        noc_async_writes_flushed();
        cb_pop_front(kY, 1);
        if (has_state) {
            noc_semaphore_wait(go, 1);  // every core has read its left-context state rows
            for (uint32_t k = 0; k < kState; ++k) {
                noc_async_write(st[k], get_noc_addr(j, st_acc[k]), kTileBytes);
            }
        }
        noc_async_write_barrier();
    }
    noc_semaphore_wait(go, 1);
    noc_semaphore_set(go, 0);
#elif defined(COMPILE_FOR_TRISC)
    unary_op_init_common(kB, kY);
    for (uint32_t sg = 0; sg < nseg; ++sg) {
        const uint32_t r0 = seg_r0(sg), r1 = seg_r1(sg);
        cb_wait_front(kT, kTaps);
        if (r0 > 0) {
            bx_tile(true);
        }
        bx_tile();
        for (uint32_t r = r0; r < r1; ++r) {
            if (r + 1 < r1) {
                bx_tile();
            }
            conv_tile();
        }
        cb_pop_front(kT, kTaps);
    }
#endif
}
