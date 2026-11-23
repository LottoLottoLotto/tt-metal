// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 attention prefill front end (head_dim 128, 16 q heads, 4 kv heads), one launch per chunk of Tp rows:
//
//   q, k, v  = heads of y[:, :3072]                     (slice + nlp_create_qkv_heads: head h, dim tile c, row tile r
//                                                        = y tile (r, base + 4h + c), base q 0 / k 64 / v 80)
//   qn, kn   = rms_norm(q, q_w), rms_norm(k, k_w)       (ttnn.rms_norm, HiFi4, FP32 DEST, eps)
//   q', k'   = x * cos + rotate_half(x) * sin           (Rope.apply: matmul(x, R) + ttnn.mul + ttnn.addcmul; cos/sin =
//                                                        rope table rows start .. start + Tp - 1)
//   q_out    = q' [1, 16, Tp, 128] BF16
//   K/V caches: rows of block cpt[t / 64] <- BFP8(k'), BFP8(v)    (typecast + paged_fill_cache)
//
// Arithmetic: the decode front's (fast/kernels/attn_decode.cpp, bitwise the same reference ops), on the native
// [rows = tokens] tile layout (every op of the reference computes per-row values, so tokens on tile rows give the
// same bits as heads on tile rows):
//   rms_norm: the RMSNORM path of ttnn's layernorm.cpp on the four head tiles of a row tile (x^2 FPU, FP32 row reduce
//             with the 1.0 scaler, * 1/128, + eps, rsqrt, * bcast col, * gamma bcast row), same CB formats (FP32
//             intermediates), block size 4 (fp32 DEST), same LLK calls.
//   RoPE:     one SFPU pass per rotate_half tile pair (front_rope_pair): the SFPU mul's fp32 product + software BF16
//             RNE, then the SFPU addcmul's fp32 MAD + BF16 RNE.  rotate_half(x) * sin == x[(h + 2) % 4] * sin_signed[h],
//             sin_signed = [-sin[:64], sin[64:]]: the sign flip is an exact SFPU negation of the BF16 table value.
//   K/V:      the packer converts k' (BF16-exact values in FP32 DEST) and v to BFP8 in precise BFP mode (compute config
//             bfp8_pack_precise, as ttnn.typecast: round half away from zero); whole cache tiles are written (chunk
//             starts are block aligned).  After each BFP8 / output pack the PACK thread drains (tensix_sync) before
//             the next unit's init reprograms the packer.
//
// Work units u = r * 24 + j (row tile r, j: 0..15 q head j, 16..19 k head j - 16, 20..23 v head j - 20); each core
// runs the contiguous unit range [u0, u1) (per-core args).  The chunk start comes from a [1] int32 device tensor (one
// program for every start): rope table row tile = min(start / 32 + r, table_tiles - 1) (the clamp only guards padded
// rows past the table's end); cache tile of (r, kv head h, dim tile c) = ((cpt[r / 2] * 4 + h) * 2 + r % 2) * 4 + c.
//
// Reader common: [y, gamma (row 0 = q_w | k_w, 8 tiles), cos table, sin table (tiled [max_pos, 128] BF16), start,
//                 table row tiles, eps bits].  Writer common: [q_out, k cache, v cache, chunk page table].
// Defines: Y_WT (y width tiles), ROW_TILES (Tp / 32), CPT_READ (chunk page-table bytes read, 64 B multiple).
#include <cstdint>

#if defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#elif defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#endif

namespace attn_front {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kBfp8TileBytes = 1088;
constexpr uint32_t kUnitsPerRow = 24;  // 16 q, 4 k, 4 v heads
constexpr uint32_t kQHeads = 16, kNormHeads = 20;
// layernorm.cpp's CB indices and formats
constexpr uint32_t kIn = 0;        // BF16, 8 tiles: one head's 4 tiles of a row tile (double buffered)
constexpr uint32_t kScaler = 2;    // BF16
constexpr uint32_t kEps = 3;       // BF16
constexpr uint32_t kGamma = 5;     // BF16, 8 tiles (row 0): q_w tiles 0..3, k_w tiles 4..7
constexpr uint32_t kNormOut = 16;  // BF16, 4 tiles
constexpr uint32_t kEx2 = 19;      // FP32
constexpr uint32_t kXmm2 = 20;     // FP32, 4 tiles
constexpr uint32_t kEx2pe = 21;    // FP32
constexpr uint32_t kFusion = 22;   // FP32, 4 tiles
// RoPE / outputs / scratch
constexpr uint32_t kCos = 6;       // BF16, 8 tiles: the 4 cos tiles of a row tile (double buffered)
constexpr uint32_t kSin = 7;       // BF16, 8 tiles: sin (unsigned table values)
constexpr uint32_t kScratch = 11;  // reader: start; writer: chunk page table
constexpr uint32_t kQOut = 24;     // BF16, 8 tiles
constexpr uint32_t kKVOut = 25;    // BFP8, 8 tiles
constexpr uint32_t kKvHeads = 4;
constexpr uint32_t kHeadTiles = 4;  // head_dim 128
constexpr uint32_t kHeadDim = 128;
}  // namespace attn_front

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
namespace ckernel::sfpu {
// RoPE of one rotate_half tile pair (h, h + 2), h in {0, 1}, in ONE SFPU pass, per element the reference's two
// packed ops:
//   t   = ttnn.mul(x, cos)            calculate_sfpu_binary_mul<is_fp32_dest_acc_en = false>: fp32 product, software
//                                     RNE to BF16, exact 0 when an operand is 0
//   out = ttnn.addcmul(t, xo, sin)    calculate_addcmul<is_fp32_dest_acc_en = false, Float16_b>: fp32 MAD, BF16 RNE
// DEST tiles (relative): 0 x[h], 1 x[h+2], 2 cos[h], 3 cos[h+2], 4 sin[h], 5 sin[h+2] ->
// 6 out[h] = t(x[h], cos[h]) + x[h+2] * (-sin[h]), 7 out[h+2] = t(x[h+2], cos[h+2]) + x[h] * sin[h+2]
// (BF16 values held as FP32).
inline void front_rope_pair() {
    constexpr uint32_t ts = 32;
#pragma GCC unroll 2
    for (int d = 0; d < 8; ++d) {
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat c0 = sfpi::dst_reg[2 * ts];
        sfpi::vFloat t0 = float32_to_bf16_rne(x0 * c0);
        v_if(x0 == 0 || c0 == 0) { t0 = 0.0f; }
        v_endif;
        sfpi::vFloat x1 = sfpi::dst_reg[1 * ts];
        const sfpi::vFloat s0p = sfpi::dst_reg[4 * ts];
        sfpi::vFloat s0 = -s0p;
        sfpi::vFloat o0 = x1 * s0 + t0;
        sfpi::vFloat r0 = sfpi::convert<sfpi::vFloat16b>(o0, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[6 * ts] = r0;
        sfpi::vFloat c1 = sfpi::dst_reg[3 * ts];
        sfpi::vFloat t1 = float32_to_bf16_rne(x1 * c1);
        v_if(x1 == 0 || c1 == 0) { t1 = 0.0f; }
        v_endif;
        sfpi::vFloat x0b = sfpi::dst_reg[0];
        sfpi::vFloat s1 = sfpi::dst_reg[5 * ts];
        sfpi::vFloat o1 = x0b * s1 + t1;
        sfpi::vFloat r1 = sfpi::convert<sfpi::vFloat16b>(o1, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[7 * ts] = r1;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

#if defined(COMPILE_FOR_TRISC)
namespace attn_front {
namespace numeric = norm::kernel_util::compute::numeric;
namespace policies = norm::kernel_util::compute::policies;

// ttnn.rms_norm (layernorm.cpp, RMSNORM, gamma, no beta, Wt = 4, W = 128, block 4) on the head tiles in kIn ->
// kNormOut, gamma tiles g0 .. g0 + 3 of kGamma.
FORCE_INLINE void rms_norm(uint32_t g0) {
    DataflowBuffer in(kIn), xmm2(kXmm2), scaler(kScaler), ex2(kEx2), eps(kEps), ex2pe(kEx2pe), fusion(kFusion),
        out(kNormOut);
    binary_op_init_common(kIn, kIn, kXmm2);
    eps.wait_front(1);
    reconfig_data_format(kIn, kIn);
    pack_reconfig_data_format(kXmm2);
    // x^2
    mul_tiles_init(kIn, kIn);
    in.wait_front(kHeadTiles);
    tile_regs_acquire();
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        mul_tiles(kIn, kIn, c, c, c);
    }
    tile_regs_commit();
    xmm2.reserve_back(kHeadTiles);
    tile_regs_wait();
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        pack_tile(c, kXmm2);
    }
    tile_regs_release();
    xmm2.push_back(kHeadTiles);
    reconfig_data_format(kIn, kXmm2, kIn, kScaler);
    // E[x^2]
    numeric::row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, true, policies::FullBlockWithPopPolicy>(
        xmm2, scaler, ex2, kHeadDim, kHeadTiles, kHeadTiles, 32);
    // 1/sqrt(E[x^2] + eps)
    ex2.wait_front(1);
    reconfig_data_format(kEx2, kEps);
    tile_regs_acquire();
    add_tiles_init(kEx2, kEps);
    add_tiles(kEx2, kEps, 0, 0, 0);
    rsqrt_tile_init<false>();
    rsqrt_tile<false>(0);
    tile_regs_commit();
    ex2.pop_front(1);
    ex2pe.reserve_back(1);
    pack_reconfig_data_format(kEx2pe);
    tile_regs_wait();
    pack_tile(0, kEx2pe);
    tile_regs_release();
    ex2pe.push_back(1);
    // x * rstd
    ex2pe.wait_front(1);
    reconfig_data_format(kIn, kEx2pe);
    pack_reconfig_data_format(kFusion);
    fusion.reserve_back(kHeadTiles);
    reconfig_data_format_srca(kFusion, kIn);
    tile_regs_acquire();
    mul_bcast_cols_init_short(kIn, kEx2pe);
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        mul_tiles_bcast_cols(kIn, kEx2pe, c, 0, c);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        pack_tile(c, kFusion);
    }
    tile_regs_release();
    fusion.push_back(kHeadTiles);
    reconfig_data_format_srca(kIn, kFusion);
    // * gamma
    pack_reconfig_data_format(kNormOut);
    reconfig_data_format_srcb(kEx2pe, kGamma);
    cb_wait_front(kGamma, 2 * kHeadTiles);
    fusion.wait_front(kHeadTiles);
    tile_regs_acquire();
    mul_bcast_rows_init_short(kFusion, kGamma);
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        mul_tiles_bcast_rows(kFusion, kGamma, c, g0 + c, c);
    }
    tile_regs_commit();
    fusion.pop_front(kHeadTiles);
    out.reserve_back(kHeadTiles);
    tile_regs_wait();
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        pack_tile(c, kNormOut);
    }
    tile_regs_release();
    out.push_back(kHeadTiles);
    ex2pe.pop_front(1);
    in.pop_front(kHeadTiles);
}

// out[h] = x[h] * cos[h] + x[(h + 2) % 4] * sin_signed[h] for the four head tiles, BF16 RNE after each op (as the
// reference's separately packed ttnn.mul and ttnn.addcmul), packed into out_cb (BF16 or BFP8).  The two tiles of a
// rotate_half pair (h, h + 2) share one DEST block (full-sync DEST, 8 FP32 tiles).  cos / sin stay in their CBs.
FORCE_INLINE void rope(uint32_t out_cb) {
    unary_op_init_common(kNormOut, out_cb);
    cb_wait_front(kNormOut, kHeadTiles);
    cb_wait_front(kCos, kHeadTiles);
    cb_wait_front(kSin, kHeadTiles);
    cb_reserve_back(out_cb, kHeadTiles);
    for (uint32_t h = 0; h < kHeadTiles / 2; ++h) {
        const uint32_t h2 = h + kHeadTiles / 2;
        tile_regs_acquire();
        copy_tile_init(kNormOut);
        copy_tile(kNormOut, h, 0);
        copy_tile(kNormOut, h2, 1);
        copy_tile_init(kCos);
        copy_tile(kCos, h, 2);
        copy_tile(kCos, h2, 3);
        copy_tile_init(kSin);
        copy_tile(kSin, h, 4);
        copy_tile(kSin, h2, 5);
        MATH((SFPU_UNARY_INIT(unused)));
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, front_rope_pair, 0, VectorMode::RC)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(6, out_cb, h);
        pack_tile<true>(7, out_cb, h2);
        tile_regs_release();
    }
    cb_push_back(out_cb, kHeadTiles);
    cb_pop_front(kNormOut, kHeadTiles);
    // The next unit's init reprograms the packer from the RISC: let this pack drain first (without it the last
    // tile's BFP8 exponents were sometimes corrupted).
    PACK((ckernel::tensix_sync()));
}

// in_cb tiles -> out_cb (v -> BFP8: ttnn.typecast's conversion, the packer in precise BFP8 mode).
FORCE_INLINE void copy_tiles(uint32_t in_cb, uint32_t out_cb) {
    unary_op_init_common(in_cb, out_cb);
    cb_wait_front(in_cb, kHeadTiles);
    cb_reserve_back(out_cb, kHeadTiles);
    tile_regs_acquire();
    copy_tile_init(in_cb);
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        copy_tile(in_cb, c, c);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        pack_tile(c, out_cb);
    }
    tile_regs_release();
    cb_push_back(out_cb, kHeadTiles);
    cb_pop_front(in_cb, kHeadTiles);
    PACK((ckernel::tensix_sync()));
}
}  // namespace attn_front
#endif

void kernel_main() {
    using namespace attn_front;
    constexpr uint32_t row_tiles = ROW_TILES;
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t u1 = get_arg_val<uint32_t>(1);
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t y_wt = Y_WT;
    auto tiles_at = [](uint32_t addr) {
        return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), addr, kTileBytes);
    };
    const auto y = tiles_at(get_common_arg_val<uint32_t>(0));
    const auto gamma = tiles_at(get_common_arg_val<uint32_t>(1));
    const auto cos_t = tiles_at(get_common_arg_val<uint32_t>(2));
    const auto sin_t = tiles_at(get_common_arg_val<uint32_t>(3));
    const auto start_t = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(4), 4);
    const uint32_t table_tiles = get_common_arg_val<uint32_t>(5);
    const uint32_t eps_bits = get_common_arg_val<uint32_t>(6);
    const uint32_t scratch = get_write_ptr(kScratch);
    noc_async_read(get_noc_addr(0, start_t), scratch, 64);
    cb_reserve_back(kGamma, 2 * kHeadTiles);
    for (uint32_t c = 0; c < 2 * kHeadTiles; ++c) {
        noc_async_read_tile(c, gamma, get_write_ptr(kGamma) + c * kTileBytes);
    }
    // constant tiles while the reads are in flight
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<kScaler, ckernel::PoolType::SUM,
                                                             ckernel::ReduceDim::REDUCE_ROW,
                                                             dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    // generate_bcast_col_scalar: column 0 of faces 0 and 2
    cb_reserve_back(kEps, 1);
    volatile uint16_t* e = reinterpret_cast<volatile uint16_t*>(get_write_ptr(kEps));
    for (uint32_t k = 0; k < 4; k += 2) {
        for (uint32_t j = 0; j < 256; j += 16) {
            e[(k << 8) + j] = static_cast<uint16_t>(eps_bits >> 16);
        }
    }
    cb_push_back(kEps, 1);
    noc_async_read_barrier();
    invalidate_l1_cache();  // the RISC data cache may hold this L1 line from an earlier kernel
    cb_push_back(kGamma, 2 * kHeadTiles);
    const uint32_t tile0 = static_cast<uint32_t>(*reinterpret_cast<volatile int32_t*>(scratch)) / 32;
    uint32_t cur_r = 0xFFFFFFFFu;
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t r = u / kUnitsPerRow, j = u % kUnitsPerRow;
        const bool rope_rows = j < kNormHeads && r != cur_r;
        if (rope_rows) {
            cur_r = r;
            const uint32_t pt = tile0 + r < table_tiles ? tile0 + r : table_tiles - 1;
            cb_reserve_back(kCos, kHeadTiles);
            cb_reserve_back(kSin, kHeadTiles);
            const uint32_t cs = get_write_ptr(kCos), sn = get_write_ptr(kSin);
            for (uint32_t c = 0; c < kHeadTiles; ++c) {
                noc_async_read_tile(pt * kHeadTiles + c, cos_t, cs + c * kTileBytes);
                noc_async_read_tile(pt * kHeadTiles + c, sin_t, sn + c * kTileBytes);
            }
        }
        const uint32_t base = j < kQHeads ? kHeadTiles * j : 64 + kHeadTiles * (j - kQHeads);  // k 64.., v 80..
        cb_reserve_back(kIn, kHeadTiles);
        const uint32_t in = get_write_ptr(kIn);
        for (uint32_t c = 0; c < kHeadTiles; ++c) {
            noc_async_read_tile(r * y_wt + base + c, y, in + c * kTileBytes);
        }
        noc_async_read_barrier();
        if (rope_rows) {
            cb_push_back(kCos, kHeadTiles);
            cb_push_back(kSin, kHeadTiles);
        }
        cb_push_back(kIn, kHeadTiles);
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto q_out = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0),
                                      kTileBytes);
    const auto k_cache = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(1), kBfp8TileBytes);
    const auto v_cache = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(2), kBfp8TileBytes);
    const auto cpt_t = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                      get_common_arg_val<uint32_t>(3), CPT_READ);
    const uint32_t scratch = get_write_ptr(kScratch) + 1024;  // reader uses [0, 64)
    noc_async_read(get_noc_addr(0, cpt_t), scratch, CPT_READ);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* cpt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t r = u / kUnitsPerRow, j = u % kUnitsPerRow;
        if (j < kQHeads) {
            cb_wait_front(kQOut, kHeadTiles);
            const uint32_t src = get_read_ptr(kQOut);
            for (uint32_t c = 0; c < kHeadTiles; ++c) {
                noc_async_write_tile((j * row_tiles + r) * kHeadTiles + c, q_out, src + c * kTileBytes);
            }
            noc_async_writes_flushed();
            cb_pop_front(kQOut, kHeadTiles);
        } else {
            const uint32_t h = (j - kQHeads) % kKvHeads;
            const uint32_t id0 = ((cpt[r / 2] * kKvHeads + h) * 2 + r % 2) * kHeadTiles;
            cb_wait_front(kKVOut, kHeadTiles);
            const uint32_t src = get_read_ptr(kKVOut);
            for (uint32_t c = 0; c < kHeadTiles; ++c) {
                noc_async_write_tile(id0 + c, j < kNormHeads ? k_cache : v_cache, src + c * kBfp8TileBytes);
            }
            noc_async_writes_flushed();
            cb_pop_front(kKVOut, kHeadTiles);
        }
    }
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    uint32_t cur_r = 0xFFFFFFFFu;
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t r = u / kUnitsPerRow, j = u % kUnitsPerRow;
        if (j >= kNormHeads) {
            copy_tiles(kIn, kKVOut);
            continue;
        }
        rms_norm(j < kQHeads ? 0 : kHeadTiles);
        if (r != cur_r) {  // the previous row tile's cos / sin: waited by its rope() (every q / k unit runs one)
            if (cur_r != 0xFFFFFFFFu) {
                cb_pop_front(kCos, kHeadTiles);
                cb_pop_front(kSin, kHeadTiles);
            }
            cur_r = r;
        }
        rope(j < kQHeads ? kQOut : kKVOut);
    }
#endif
}
