// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 attention decode front end (head_dim 128, 16 q heads, 4 kv heads), one launch per call:
//
//   q, k, v  = heads of the qkv(g) row t                (nlp_create_qkv_heads on qkv[:, :3072])
//   qn, kn   = rms_norm(q, q_w), rms_norm(k, k_w)       (ttnn.rms_norm, HiFi4, FP32 DEST, eps)
//   q', k'   = x * cos + rotate_half(x) * sin           (Rope.apply: matmul(x, R) + ttnn.mul + ttnn.addcmul)
//   q_out[t] = q' as the [1, B, 16, 128] SDPA query     (permute)
//   K/V caches: row (pos % 64) of block page_table[t][pos / 64] <- BFP8(k'), BFP8(v)   (paged_update_cache)
//   gate[t]  = the 32 gate-logit columns of row t       (raw gate_proj output; sigmoid is applied by the consumer)
//
// Two input modes (FRONT_FUSED):
//   0  (attn_decode.AttnDecode)     the reader gathers the head rows from the qkv(g) tensor in DRAM; the value worker
//                                   copies row t of the gate source tile (qkvg's tile column 96 or a separate
//                                   [1, 1, B, 32] gate tensor) into the gate output.
//   1  (attn_fused.AttnQKVFront)    the head rows are written into this worker's input tiles (CB kIn) by the qkv
//                                   matmul workers of the same program (kernels/attn_qkv_mm.cpp), which increment
//                                   semaphore 0 once each; the reader waits for `expected` increments.  The matmul
//                                   worker of the gate column writes the gate output itself.
//
// Heads are moved to tile rows (row i = head h0 + i: a row-wise layout in which every op of the reference sequence
// computes the same per-row values).  Workers per token t (attn_decode.front_layout): NQ query workers (16 / NQ heads
// each), NK key workers (4 / NK heads each) and one value worker (4 heads).  A head is 4 tiles wide (dims 0-31,
// 32-63, 64-95, 96-127).
//   rms_norm: the RMSNORM path of ttnn's layernorm.cpp on the four head tiles (x^2 FPU, FP32 row reduce with the
//             1.0 scaler, * 1/128, + eps, rsqrt, * bcast col, * gamma bcast row), same CB formats (FP32
//             intermediates), block size 4 (fp32 DEST) and the same LLK calls.
//   RoPE:     one SFPU pass per rotate_half tile pair (attn_rope_pair): the SFPU mul's fp32 product + software BF16
//             RNE and the SFPU addcmul's fp32 MAD + BF16 RNE, i.e. binary_ng's / ternary's explicit BF16 rounding, so
//             the FP32 DEST holds exactly the BF16 values the reference packs.  rotate_half(x) * sin ==
//             x[(h + 2) % 4] * sin_signed[h]
//             with sin_signed = [-sin[:64], sin[64:]] (the sign flip is applied to the BF16 table row here: exact).
//             The layer type's tables (Rope.cos_rm / sin_rm, [max_pos, 128] BF16 row-major) are read at cur_pos.
//   KV:       the packer converts the k'/v head rows to BFP8 (shared exponent per 16-element face row, so a row's
//             encoding is independent of the other rows); the writer patches that row's 2 exponent bytes and 2x16
//             mantissa bytes into the cache tile (read whole, patched in L1, written whole).
//
// Per-core args (all RISCs): [role (0 q, 1 k, 2 v), first head h0, heads nh, token t, expected matmul writers].
// Reader common: [qkv (0 when fused), gamma, cos table, sin table, cur_pos, page table, k cache, v cache, eps bits,
//                 gate source (0 when fused), gate out].
// Writer common: [q_out, k cache, v cache].
//
// FRONT_VERIFY (speculative verification, AttnNormQKVFront(verify=True); fused mode): the rows are groups of
// consecutive tokens, one group per sequence (positions pos_g, pos_g + 1, ...; inactive rows -1), so several rows of a
// group patch the same 32-row cache tile.  The row map (reader common arg 11, tt/speculate.py map_word: offset k in
// bits 0-7, group length in 8-15) bounds each group: per cache tile of a group one leader (its first row of this
// step) reads, patches and writes the tile; the group's other rows in that tile send their packed BFP8 head rows into
// the leader's receive slots (CB kRecv) and increment its semaphore 1.  Rows of different groups write different
// sequences' blocks and never share a tile.  Extra per-core args: the NOC (x, y) of the same-slot worker of every
// token.
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#ifndef FRONT_IN_CB
#define FRONT_IN_CB 0
#endif
#ifndef FRONT_FUSED
#define FRONT_FUSED 0
#endif
#ifndef FRONT_VERIFY
#define FRONT_VERIFY 0
#endif

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

namespace attn_decode {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kBfp8TileBytes = 1088;  // 64 B exponents (face-major, 16 rows each) + 4 x 256 B mantissas
constexpr uint32_t kRoleQ = 0, kRoleK = 1, kRoleV = 2;
// layernorm.cpp's CB indices and formats
constexpr uint32_t kIn = FRONT_IN_CB;  // BF16, 4 tiles: head rows (gathered, or written remotely by matmul workers)
constexpr uint32_t kScaler = 2;   // BF16
constexpr uint32_t kEps = 3;      // BF16
constexpr uint32_t kGamma = 5;    // BF16, 4 tiles (row 0)
constexpr uint32_t kNormOut = 16; // BF16, 4 tiles
constexpr uint32_t kEx2 = 19;     // FP32
constexpr uint32_t kXmm2 = 20;    // FP32, 4 tiles
constexpr uint32_t kEx2pe = 21;   // FP32
constexpr uint32_t kFusion = 22;  // FP32, 4 tiles
// RoPE / outputs / scratch
constexpr uint32_t kCos = 6;      // BF16, 4 tiles: cos row replicated over the head rows
constexpr uint32_t kSin = 7;      // BF16, 4 tiles: sin_signed
constexpr uint32_t kGather = 8;   // reader scratch (unfused): 64 B face-row pairs, 2 tiles
constexpr uint32_t kPosBox = 9;   // reader -> writer: {pos, block}
constexpr uint32_t kCache = 10;   // BFP8, 4 * nh pages: cache tiles to patch (reader -> writer)
constexpr uint32_t kScratch = 11; // reader scratch: [0, 64) cur_pos, [64, 128) page table, [256, 512) cos row,
                                  // [512, 768) sin row, [1024, 1152) gate row pair
constexpr uint32_t kQOut = 24;    // BF16, 4 tiles
constexpr uint32_t kKVOut = 25;   // BFP8, 4 tiles
constexpr uint32_t kRecv = 26;    // FRONT_VERIFY: BFP8, kHeadTiles pages per follower (leaders' receive slots)
constexpr uint32_t kBlock = 64;   // KV page (tokens)
constexpr uint32_t kKvHeads = 4;
constexpr uint32_t kHeadTiles = 4;  // head_dim 128
constexpr uint32_t kHeadDim = 128;
}  // namespace attn_decode

#if defined(COMPILE_FOR_NCRISC)
namespace attn_decode {
// 32 B copy (one face row of a BF16 tile); flip = 0x80008000 negates the 16 BF16 values (sign bits)
FORCE_INLINE void copy32(uint32_t dst, uint32_t src, uint32_t flip = 0) {
    volatile uint32_t* d = reinterpret_cast<volatile uint32_t*>(dst);
    volatile uint32_t* s = reinterpret_cast<volatile uint32_t*>(src);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w] ^ flip;
    }
}
// Byte offset of (row, face column f) in a BF16 tile.
FORCE_INLINE uint32_t row_face(uint32_t row, uint32_t f) { return ((row >> 4) * 2 + f) * 512 + (row & 15) * 32; }
}  // namespace attn_decode
#endif

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
namespace ckernel::sfpu {
// RoPE of one rotate_half tile pair in ONE SFPU pass, per element the reference's two packed ops:
//   t   = ttnn.mul(x, cos)            calculate_sfpu_binary_mul<is_fp32_dest_acc_en = false>: fp32 product, software
//                                     RNE to BF16, exact 0 when an operand is 0
//   out = ttnn.addcmul(t, xo, sin)    calculate_addcmul<is_fp32_dest_acc_en = false, Float16_b>: fp32 MAD, BF16 RNE
// DEST tiles (relative): 0 x[h], 1 x[h+2], 2 cos[h], 3 cos[h+2], 4 sin_signed[h], 5 sin_signed[h+2] ->
// 6 out[h] = t(x[h], cos[h]) + x[h+2] sin_signed[h], 7 out[h+2] = t(x[h+2], cos[h+2]) + x[h] sin_signed[h+2]
// (BF16 values held as FP32).
inline void attn_rope_pair() {
    constexpr uint32_t ts = 32;
#pragma GCC unroll 2
    for (int d = 0; d < 8; ++d) {
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat c0 = sfpi::dst_reg[2 * ts];
        sfpi::vFloat t0 = float32_to_bf16_rne(x0 * c0);
        v_if(x0 == 0 || c0 == 0) { t0 = 0.0f; }
        v_endif;
        sfpi::vFloat x1 = sfpi::dst_reg[1 * ts];
        sfpi::vFloat s0 = sfpi::dst_reg[4 * ts];
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
namespace attn_decode {
namespace numeric = norm::kernel_util::compute::numeric;
namespace policies = norm::kernel_util::compute::policies;

// ttnn.rms_norm (layernorm.cpp, RMSNORM, gamma, no beta, Wt = 4, W = 128, block 4) on the head tiles in kIn ->
// kNormOut.
FORCE_INLINE void rms_norm() {
    DataflowBuffer in(kIn), xmm2(kXmm2), scaler(kScaler), ex2(kEx2), eps(kEps), ex2pe(kEx2pe), fusion(kFusion),
        gamma(kGamma), out(kNormOut);
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
    gamma.wait_front(kHeadTiles);
    fusion.wait_front(kHeadTiles);
    tile_regs_acquire();
    mul_bcast_rows_init_short(kFusion, kGamma);
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        mul_tiles_bcast_rows(kFusion, kGamma, c, c, c);
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
// rotate_half pair (h, h + 2) share one DEST block (full-sync DEST, 8 FP32 tiles): x[h], x[h+2], cos, sin of both.
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
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, attn_rope_pair, 0, VectorMode::RC)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(6, out_cb, h);
        pack_tile<true>(7, out_cb, h2);
        tile_regs_release();
    }
    cb_push_back(out_cb, kHeadTiles);
    cb_pop_front(kNormOut, kHeadTiles);
    cb_pop_front(kCos, kHeadTiles);
    cb_pop_front(kSin, kHeadTiles);
}

// v rows -> BFP8 (paged_update_cache's conversion: the packer).
FORCE_INLINE void pack_value() {
    unary_op_init_common(kIn, kKVOut);
    cb_wait_front(kIn, kHeadTiles);
    cb_reserve_back(kKVOut, kHeadTiles);
    tile_regs_acquire();
    copy_tile_init(kIn);
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        copy_tile(kIn, c, c);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t c = 0; c < kHeadTiles; ++c) {
        pack_tile(c, kKVOut);
    }
    tile_regs_release();
    cb_push_back(kKVOut, kHeadTiles);
    cb_pop_front(kIn, kHeadTiles);
}
}  // namespace attn_decode
#endif

void kernel_main() {
    using namespace attn_decode;
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t h0 = get_arg_val<uint32_t>(1);
    const uint32_t nh = get_arg_val<uint32_t>(2);
    const uint32_t t = get_arg_val<uint32_t>(3);
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t table_page = kHeadDim * 2;  // rope table row bytes (BF16)
    constexpr uint32_t pos_page = POS_PAGE;        // cur_pos row bytes (B int32)
    constexpr uint32_t pt_page = PT_PAGE;          // page table row bytes (blocks int32)
    const auto gamma = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                      get_common_arg_val<uint32_t>(1), kTileBytes);
    const auto cos_t = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                      get_common_arg_val<uint32_t>(2), table_page);
    const auto sin_t = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                      get_common_arg_val<uint32_t>(3), table_page);
    const auto cur_pos = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(4), pos_page);
    const auto page_table = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                           get_common_arg_val<uint32_t>(5), pt_page);
    const uint32_t cache_addr = get_common_arg_val<uint32_t>(role == kRoleV ? 7 : 6);
    const auto cache = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), cache_addr, kBfp8TileBytes);
    const uint32_t eps_bits = get_common_arg_val<uint32_t>(8);

    // 1. position-independent reads: this token's position, the head rows (unfused), gamma, the gate row (64 B
    //    aligned DRAM reads into 64 B aligned scratch).  gather: head h, dim tile c of token t = row t of qkv column
    //    tile base + 4h + c.
    const uint32_t scratch = get_write_ptr(kScratch);
    noc_async_read(get_noc_addr(0, cur_pos, (t * 4) & ~63u), scratch, 64);
#if FRONT_VERIFY
    // the verify row map (one 64 B page: u32 per row) -> scratch [128, 192)
    noc_async_read(TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(11), 64)
                       .get_noc_addr(0),
                   scratch + 128, 64);
#endif
    const uint32_t pair_off = ((t >> 4) * 2) * 512 + ((t & 15) & ~1u) * 32;  // rows t & ~1, t | 1 of face 0
    const uint32_t sel = (t & 1) * 32;
#if !FRONT_FUSED
    const auto qkv = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0),
                                    kTileBytes);
    const uint32_t base = role == kRoleQ ? 0 : (role == kRoleK ? 64 : 80);
    const uint32_t gather = get_write_ptr(kGather);
    for (uint32_t i = 0; i < nh; ++i) {
        for (uint32_t c = 0; c < kHeadTiles; ++c) {
            const uint64_t src = get_noc_addr(base + kHeadTiles * (h0 + i) + c, qkv, pair_off);
            const uint32_t dst = gather + (i * kHeadTiles + c) * 128;
            noc_async_read(src, dst, 64);
            noc_async_read(src + 512, dst + 64, 64);
        }
    }
    if (role == kRoleV) {
        const auto gsrc = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                         get_common_arg_val<uint32_t>(9), kTileBytes);
        const uint64_t src = get_noc_addr(GATE_TILE, gsrc, pair_off);
        noc_async_read(src, scratch + 1024, 64);
        noc_async_read(src + 512, scratch + 1088, 64);
    }
#endif
    if (role != kRoleV) {
        cb_reserve_back(kGamma, kHeadTiles);
        const uint32_t g0 = role == kRoleQ ? 0 : kHeadTiles;
        for (uint32_t c = 0; c < kHeadTiles; ++c) {
            noc_async_read(get_noc_addr(g0 + c, gamma), get_write_ptr(kGamma) + c * kTileBytes, kTileBytes);
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
    }
    noc_async_read_barrier();
    const int32_t pos = *reinterpret_cast<volatile int32_t*>(scratch + ((t * 4) & 63u));
    const uint32_t rope_pos = pos < 0 ? 0 : static_cast<uint32_t>(pos);
    const bool kv = role != kRoleQ;
    const bool live = pos >= 0;
    // FRONT_VERIFY: the leader of this row's cache tile (the group's first row of this step in that tile) and, for a
    // leader, the number of the group's rows after it in the same tile (positions are consecutive within a group;
    // every row's position and map word are in the first 64 B read)
    uint32_t t0 = t, followers = 0;
#if FRONT_VERIFY
    if (kv && live) {
        constexpr uint32_t nrows = POS_PAGE / 4;
        const volatile int32_t* p = reinterpret_cast<volatile int32_t*>(scratch);
        const uint32_t m = reinterpret_cast<volatile uint32_t*>(scratch + 128)[t];
        const uint32_t g0 = t - (m & 0xFFu), g1 = g0 + ((m >> 8) & 0xFFu);
        const uint32_t tile = static_cast<uint32_t>(pos) / 32;
        while (t0 > g0 && p[t0 - 1] >= 0 && static_cast<uint32_t>(p[t0 - 1]) / 32 == tile) {
            --t0;
        }
        if (t0 == t) {
            for (uint32_t u = t + 1; u < g1 && u < nrows && p[u] >= 0 && static_cast<uint32_t>(p[u]) / 32 == tile;
                 ++u) {
                ++followers;
            }
        }
    }
#endif
    const bool patcher = kv && live && t0 == t;  // reads, patches and writes this row's cache tiles
    // 2. position-dependent reads (rope rows, page-table entry) overlap the head-row placement and the norm
    if (role != kRoleV) {
        noc_async_read(get_noc_addr(rope_pos, cos_t), scratch + 256, table_page);
        noc_async_read(get_noc_addr(rope_pos, sin_t), scratch + 512, table_page);
        cb_push_back(kGamma, kHeadTiles);
    }
    const uint32_t entry = (rope_pos / kBlock) * 4;
    if (kv && live) {
        noc_async_read(get_noc_addr(t, page_table, entry & ~63u), scratch + 64, 64);
    }
#if !FRONT_FUSED
    cb_reserve_back(kIn, kHeadTiles);
    const uint32_t in = get_write_ptr(kIn);
    for (uint32_t i = 0; i < nh; ++i) {
        for (uint32_t c = 0; c < kHeadTiles; ++c) {
            for (uint32_t f = 0; f < 2; ++f) {
                copy32(in + c * kTileBytes + row_face(i, f), gather + (i * kHeadTiles + c) * 128 + f * 64 + sel);
            }
        }
    }
    cb_push_back(kIn, kHeadTiles);
    if (role == kRoleV) {  // gate logits of row t -> the gate output (row t of its only tile)
        const auto gout = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                         get_common_arg_val<uint32_t>(10), kTileBytes);
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_write(scratch + 1024 + f * 64 + sel, get_noc_addr(0, gout, row_face(t, f)), 32);
        }
    }
#endif
    noc_async_read_barrier();
    // 3. cache tiles to patch (K/V), rope tiles (Q/K).  FRONT_FUSED: the cache tiles are read after the head rows have
    //    arrived (they are needed only for the final patch; read during the weight stream, 16 rows' scattered tile
    //    reads slow the stream down)
    uint32_t block = 0;
    if (kv) {
        block = live ? *reinterpret_cast<volatile uint32_t*>(scratch + 64 + (entry & 63u)) : 0;
        cb_reserve_back(kPosBox, 1);
        volatile uint32_t* box = reinterpret_cast<volatile uint32_t*>(get_write_ptr(kPosBox));
        box[0] = static_cast<uint32_t>(pos);
        box[1] = block;
        box[2] = t0;
        box[3] = followers;
        cb_push_back(kPosBox, 1);
    }
    auto read_cache = [&]() {
        const uint32_t rt = (rope_pos % kBlock) / 32;
        cb_reserve_back(kCache, kHeadTiles * nh);
        const uint32_t dst = get_write_ptr(kCache);
        for (uint32_t i = 0; i < nh; ++i) {
            for (uint32_t ct = 0; ct < kHeadTiles; ++ct) {
                const uint32_t id = ((block * kKvHeads + h0 + i) * 2 + rt) * kHeadTiles + ct;
                noc_async_read(get_noc_addr(id, cache), dst + (i * kHeadTiles + ct) * kBfp8TileBytes,
                               kBfp8TileBytes);
            }
        }
    };
#if !FRONT_FUSED
    if (patcher) {
        read_cache();
    }
#endif
    if (role != kRoleV) {
        cb_reserve_back(kCos, kHeadTiles);
        cb_reserve_back(kSin, kHeadTiles);
        const uint32_t cs = get_write_ptr(kCos), sn = get_write_ptr(kSin);
        for (uint32_t i = 0; i < nh; ++i) {
            for (uint32_t c = 0; c < kHeadTiles; ++c) {
                const uint32_t flip = c < kHeadTiles / 2 ? 0x80008000u : 0u;  // sin_signed = [-sin[:64], sin[64:]]
                for (uint32_t f = 0; f < 2; ++f) {
                    copy32(cs + c * kTileBytes + row_face(i, f), scratch + 256 + c * 64 + f * 32);
                    copy32(sn + c * kTileBytes + row_face(i, f), scratch + 512 + c * 64 + f * 32, flip);
                }
            }
        }
        cb_push_back(kCos, kHeadTiles);
        cb_push_back(kSin, kHeadTiles);
    }
#if FRONT_FUSED
    // 4. the head rows: written into the (untouched, so write pointer = base) input tiles by the matmul workers
    {
        const uint32_t expected = get_arg_val<uint32_t>(4);
        cb_reserve_back(kIn, kHeadTiles);
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        DeviceZoneScopedN("front-wait");
        noc_semaphore_wait(sem, expected);
        noc_semaphore_set(sem, 0);
        cb_push_back(kIn, kHeadTiles);
    }
    if (patcher) {
        read_cache();
    }
#endif
    if (patcher) {
        noc_async_read_barrier();
        cb_push_back(kCache, kHeadTiles * nh);
    }
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_BRISC)
    if (role == kRoleQ) {
        const auto q_out = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                          get_common_arg_val<uint32_t>(0), kTileBytes);
        cb_wait_front(kQOut, kHeadTiles);
        DeviceZoneScopedN("front-qwrite");
        const uint32_t src = get_read_ptr(kQOut);
        for (uint32_t c = 0; c < kHeadTiles; ++c) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint64_t dst = get_noc_addr(t * kHeadTiles + c, q_out, ((h0 >> 4) * 2 + f) * 512 + (h0 & 15) * 32);
                noc_async_write(src + c * kTileBytes + f * 512, dst, nh * 32);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(kQOut, kHeadTiles);
    } else {
        const auto cache = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                          get_common_arg_val<uint32_t>(role == kRoleV ? 2 : 1), kBfp8TileBytes);
        cb_wait_front(kPosBox, 1);
        volatile uint32_t* box = reinterpret_cast<volatile uint32_t*>(get_read_ptr(kPosBox));
        const int32_t pos = static_cast<int32_t>(box[0]);
        const uint32_t block = box[1];
        const uint32_t t0 = box[2], followers = box[3];
        cb_wait_front(kKVOut, kHeadTiles);
        DeviceZoneScopedN("front-kvwrite");
        if (pos >= 0 && t0 != t) {
            // FRONT_VERIFY follower: the packed head rows -> the leader's receive slot t - t0 - 1, then its semaphore 1
            const uint32_t lx = get_arg_val<uint32_t>(5 + 2 * t0), ly = get_arg_val<uint32_t>(6 + 2 * t0);
            constexpr uint32_t slot_bytes = kHeadTiles * kBfp8TileBytes;
            const uint32_t slot = get_write_ptr(kRecv) + (t - t0 - 1) * slot_bytes;
            noc_async_write(get_read_ptr(kKVOut), get_noc_addr(lx, ly, slot), slot_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(lx, ly, get_semaphore(1)), 1);
            noc_async_atomic_barrier();
        } else if (pos >= 0) {
            const uint32_t r = static_cast<uint32_t>(pos) % kBlock;
            const uint32_t rt = r / 32, rr = r % 32;
            cb_wait_front(kCache, kHeadTiles * nh);
            if (followers) {
                volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
                noc_semaphore_wait(in, followers);
                noc_semaphore_set(in, 0);
            }
            const uint32_t tiles = get_read_ptr(kCache), src = get_read_ptr(kKVOut);
            // head row i of the packed tile s -> cache row row of tile dst (2 exponent bytes, 2 x 16 mantissa bytes)
            auto patch = [&](uint32_t dst, uint32_t s, uint32_t row, uint32_t i) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t face = (row >> 4) * 2 + f;
                    reinterpret_cast<volatile uint8_t*>(dst)[face * 16 + (row & 15)] =
                        reinterpret_cast<volatile uint8_t*>(s)[f * 16 + i];
                    volatile uint32_t* dm = reinterpret_cast<volatile uint32_t*>(dst + 64 + face * 256 + (row & 15) * 16);
                    volatile uint32_t* sm = reinterpret_cast<volatile uint32_t*>(s + 64 + f * 256 + i * 16);
                    for (uint32_t w = 0; w < 4; ++w) {
                        dm[w] = sm[w];
                    }
                }
            };
            [[maybe_unused]] const uint32_t recv = FRONT_VERIFY ? get_write_ptr(kRecv) : 0;
            for (uint32_t i = 0; i < nh; ++i) {
                for (uint32_t ct = 0; ct < kHeadTiles; ++ct) {
                    const uint32_t dst = tiles + (i * kHeadTiles + ct) * kBfp8TileBytes;
                    patch(dst, src + ct * kBfp8TileBytes, rr, i);
                    for (uint32_t k = 0; k < followers; ++k) {  // FRONT_VERIFY: the rows after this one in the tile
                        patch(dst, recv + (k * kHeadTiles + ct) * kBfp8TileBytes, rr + k + 1, i);
                    }
                    const uint32_t id = ((block * kKvHeads + h0 + i) * 2 + rt) * kHeadTiles + ct;
                    noc_async_write(dst, get_noc_addr(id, cache), kBfp8TileBytes);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(kCache, kHeadTiles * nh);
        }
        cb_pop_front(kKVOut, kHeadTiles);
        cb_pop_front(kPosBox, 1);
    }
#elif defined(COMPILE_FOR_TRISC)
    if (role == kRoleV) {
        pack_value();
    } else {
        {
            cb_wait_front(kIn, kHeadTiles);
            DeviceZoneScopedN("front-norm");
            rms_norm();
        }
        DeviceZoneScopedN("front-rope");
        rope(role == kRoleQ ? kQOut : kKVOut);
    }
#endif
}
