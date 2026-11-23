// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 attention decode front end after the qkv matmul, one launch (head_dim 64, 32 q heads, 8 kv heads):
//
//   q, k, v  = heads of qkv row t                       (nlp_create_qkv_heads)
//   qn, kn   = rms_norm(q, q_w), rms_norm(k, k_w)       (ttnn.rms_norm, HiFi4, FP32 DEST, eps)
//   q', k'   = x * cos + rotate_half(x) * sin           (Rope.apply: matmul(x, R) + ttnn.mul + ttnn.addcmul)
//   q_out[t] = q' as the [1, B, 32, 64] SDPA query      (permute)
//   K/V caches: row (pos % 64) of block page_table[t][pos / 64] <- BFP8(k'), BFP8(v)   (paged_update_cache)
//
// Heads are moved to tile rows (row i = head h0 + i: a row-wise layout in which every op of the reference sequence
// computes the same per-row values).  Workers per token t: NQ query workers (QH heads each), one key worker and one
// value worker (8 heads each).
//   rms_norm: the RMSNORM path of ttnn's layernorm.cpp on the two head tiles (x^2 FPU, FP32 row reduce with the
//             1.0 scaler, * 1/64, + eps, rsqrt, * bcast col, * gamma bcast row), same CB formats (FP32
//             intermediates) and the same LLK calls.
//   RoPE:     SFPU mul instantiated with is_fp32_dest_acc_en = false and the SFPU addcmul's MAD + BF16 RNE
//             (attn_addcmul_bf16), i.e. binary_ng's / ternary's explicit BF16 rounding, so the FP32 DEST holds
//             exactly the BF16 values the reference packs.  rotate_half(x) * sin == x_other * sin_signed with sin_signed = [-sin[:32], sin[32:]].
//   KV:       the packer converts the k'/v head rows to BFP8 (shared exponent per 16-element face row, so a row's
//             encoding is independent of the other rows); the writer patches that row's 2 exponent bytes and 2x16
//             mantissa bytes into the cache tile (read whole, patched in L1, written whole).
//
// Per-core args (all RISCs): [role (0 q, 1 k, 2 v), first head h0, heads nh, token t].
// Reader common: [qkv, gamma, rope table, cur_pos, page table, k cache, v cache, eps bits].
// Writer common: [q_out, k cache, v cache].
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
#include "api/compute/eltwise_unary/addcmul.h"
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
constexpr uint32_t kIn = 0;       // BF16, 2 tiles: gathered head rows, dims 0-31 / 32-63
constexpr uint32_t kScaler = 2;   // BF16
constexpr uint32_t kEps = 3;      // BF16
constexpr uint32_t kGamma = 5;    // BF16, 2 tiles (row 0)
constexpr uint32_t kNormOut = 16; // BF16, 2 tiles
constexpr uint32_t kEx2 = 19;     // FP32
constexpr uint32_t kXmm2 = 20;    // FP32, 2 tiles
constexpr uint32_t kEx2pe = 21;   // FP32
constexpr uint32_t kFusion = 22;  // FP32, 2 tiles
// RoPE / outputs / scratch
constexpr uint32_t kCos = 6;      // BF16, 2 tiles: cos row replicated over the head rows
constexpr uint32_t kSin = 7;      // BF16, 2 tiles: sin_signed
constexpr uint32_t kGather = 8;   // reader scratch: 64 B face-row pairs
constexpr uint32_t kPosBox = 9;   // reader -> writer: {pos, block}
constexpr uint32_t kCache = 10;   // BFP8, 2 * nh pages: cache tiles to patch (reader -> writer)
constexpr uint32_t kScratch = 11; // reader scratch: cur_pos / page table / rope row
constexpr uint32_t kQOut = 24;    // BF16, 2 tiles
constexpr uint32_t kKVOut = 25;   // BFP8, 2 tiles
constexpr uint32_t kBlock = 64;   // KV page (tokens)
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kHeadTiles = 2;  // head_dim 64
}  // namespace attn_decode

#if defined(COMPILE_FOR_NCRISC)
namespace attn_decode {
FORCE_INLINE void copy32(uint32_t dst, uint32_t src) {
    volatile uint32_t* d = reinterpret_cast<volatile uint32_t*>(dst);
    volatile uint32_t* s = reinterpret_cast<volatile uint32_t*>(src);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w];
    }
}
// Byte offset of (row, face column f) in a BF16 tile.
FORCE_INLINE uint32_t row_face(uint32_t row, uint32_t f) { return ((row >> 4) * 2 + f) * 512 + (row & 15) * 32; }
}  // namespace attn_decode
#endif

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
namespace ckernel::sfpu {
// calculate_addcmul<APPROX, is_fp32_dest_acc_en = false, Float16_b> (value 1.0f) with its BF16 RNE result stored
// as FP32: the library's 16-bit-DEST variant stores a vFloat16b, which a 32-bit DEST does not hold.
inline void attn_addcmul_bf16(const uint32_t in0, const uint32_t in1, const uint32_t in2, const uint32_t out) {
    constexpr uint32_t ts = 32;
#pragma GCC unroll 8
    for (int d = 0; d < 8; ++d) {
        sfpi::vFloat a = sfpi::dst_reg[in0 * ts];
        sfpi::vFloat b = sfpi::dst_reg[in1 * ts];
        sfpi::vFloat c = sfpi::dst_reg[in2 * ts];
        sfpi::vFloat r = b * c + a;
        sfpi::vFloat rounded = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[out * ts] = rounded;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

#if defined(COMPILE_FOR_TRISC)
namespace attn_decode {
namespace numeric = norm::kernel_util::compute::numeric;
namespace policies = norm::kernel_util::compute::policies;

// ttnn.rms_norm (layernorm.cpp, RMSNORM, gamma, no beta, Wt = 2, W = 64) on the two head tiles in kIn -> kNormOut.
FORCE_INLINE void rms_norm() {
    DataflowBuffer in(kIn), xmm2(kXmm2), scaler(kScaler), ex2(kEx2), eps(kEps), ex2pe(kEx2pe), fusion(kFusion),
        gamma(kGamma), out(kNormOut);
    binary_op_init_common(kIn, kIn, kXmm2);
    eps.wait_front(1);
    reconfig_data_format(kIn, kIn);
    pack_reconfig_data_format(kXmm2);
    // x^2
    mul_tiles_init(kIn, kIn);
    in.wait_front(2);
    tile_regs_acquire();
    mul_tiles(kIn, kIn, 0, 0, 0);
    mul_tiles(kIn, kIn, 1, 1, 1);
    tile_regs_commit();
    xmm2.reserve_back(2);
    tile_regs_wait();
    pack_tile(0, kXmm2);
    pack_tile(1, kXmm2);
    tile_regs_release();
    xmm2.push_back(2);
    reconfig_data_format(kIn, kXmm2, kIn, kScaler);
    // E[x^2]
    numeric::row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, true, policies::FullBlockWithPopPolicy>(
        xmm2, scaler, ex2, 64, 2, 2, 32);
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
    fusion.reserve_back(2);
    reconfig_data_format_srca(kFusion, kIn);
    tile_regs_acquire();
    mul_bcast_cols_init_short(kIn, kEx2pe);
    mul_tiles_bcast_cols(kIn, kEx2pe, 0, 0, 0);
    mul_tiles_bcast_cols(kIn, kEx2pe, 1, 0, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, kFusion);
    pack_tile(1, kFusion);
    tile_regs_release();
    fusion.push_back(2);
    reconfig_data_format_srca(kIn, kFusion);
    // * gamma
    pack_reconfig_data_format(kNormOut);
    reconfig_data_format_srcb(kEx2pe, kGamma);
    gamma.wait_front(2);
    fusion.wait_front(2);
    tile_regs_acquire();
    mul_bcast_rows_init_short(kFusion, kGamma);
    mul_tiles_bcast_rows(kFusion, kGamma, 0, 0, 0);
    mul_tiles_bcast_rows(kFusion, kGamma, 1, 1, 1);
    tile_regs_commit();
    fusion.pop_front(2);
    out.reserve_back(2);
    tile_regs_wait();
    pack_tile(0, kNormOut);
    pack_tile(1, kNormOut);
    tile_regs_release();
    out.push_back(2);
    ex2pe.pop_front(1);
    in.pop_front(2);
}

// out[h] = x[h] * cos[h] + x[1 - h] * sin_signed[h] for the two head tiles, BF16 RNE after each op (as the
// reference's separately packed ttnn.mul and ttnn.addcmul), packed into out_cb (BF16 or BFP8).
FORCE_INLINE void rope(uint32_t out_cb) {
    unary_op_init_common(kNormOut, out_cb);
    cb_wait_front(kNormOut, 2);
    cb_wait_front(kCos, 2);
    cb_wait_front(kSin, 2);
    cb_reserve_back(out_cb, 2);
    for (uint32_t h = 0; h < 2; ++h) {
        tile_regs_acquire();
        copy_tile_init(kNormOut);
        copy_tile(kNormOut, h, 0);
        copy_tile_init(kCos);
        copy_tile(kCos, h, 1);
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
                               (APPROX, ckernel::BinaryOp::MUL, 8, false), 0, 1, 0, VectorMode::RC)));
        copy_tile_init(kNormOut);
        copy_tile(kNormOut, 1 - h, 1);
        copy_tile_init(kSin);
        copy_tile(kSin, h, 2);
        addcmul_tile_init();
        MATH((SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, attn_addcmul_bf16, 0, 1, 2, 0,
                                                 VectorMode::RC)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_push_back(out_cb, 2);
    cb_pop_front(kNormOut, 2);
    cb_pop_front(kCos, 2);
    cb_pop_front(kSin, 2);
}

// v rows -> BFP8 (paged_update_cache's conversion: the packer).
FORCE_INLINE void pack_value() {
    constexpr uint32_t in_cb = kIn, out_cb = kKVOut;
    unary_op_init_common(in_cb, out_cb);
    cb_wait_front(in_cb, 2);
    cb_reserve_back(out_cb, 2);
    tile_regs_acquire();
    copy_tile_init(in_cb);
    copy_tile(in_cb, 0, 0);
    copy_tile(in_cb, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    pack_tile(1, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 2);
    cb_pop_front(in_cb, 2);
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
    constexpr uint32_t table_page = TABLE_PAGE;  // rope table row bytes: [cos (64) | sin_signed (64)] BF16
    constexpr uint32_t pos_page = POS_PAGE;      // cur_pos row bytes (B int32)
    constexpr uint32_t pt_page = PT_PAGE;        // page table row bytes (blocks int32)
    const auto qkv = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0),
                                    kTileBytes);
    const auto gamma = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                      get_common_arg_val<uint32_t>(1), kTileBytes);
    const auto table = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                      get_common_arg_val<uint32_t>(2), table_page);
    const auto cur_pos = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(3), pos_page);
    const auto page_table = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                           get_common_arg_val<uint32_t>(4), pt_page);
    const uint32_t cache_addr = get_common_arg_val<uint32_t>(role == kRoleV ? 6 : 5);
    const auto cache = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), cache_addr, kBfp8TileBytes);
    const uint32_t eps_bits = get_common_arg_val<uint32_t>(7);

    // 1. position-independent reads: this token's position, the head rows, gamma (64 B aligned DRAM reads into 64 B
    //    aligned scratch).  gather: head h, half c of token t = row t of qkv column tile base + 2h + c.
    const uint32_t scratch = get_write_ptr(kScratch);
    noc_async_read(get_noc_addr(0, cur_pos, (t * 4) & ~63u), scratch, 64);
    const uint32_t base = role == kRoleQ ? 0 : (role == kRoleK ? 64 : 80);
    const uint32_t gather = get_write_ptr(kGather);
    const uint32_t src_off = ((t >> 4) * 2) * 512 + ((t & 15) & ~1u) * 32;
    for (uint32_t i = 0; i < nh; ++i) {
        for (uint32_t c = 0; c < kHeadTiles; ++c) {
            const uint64_t src = get_noc_addr(base + 2 * (h0 + i) + c, qkv, src_off);
            const uint32_t dst = gather + (i * 2 + c) * 128;
            noc_async_read(src, dst, 64);
            noc_async_read(src + 512, dst + 64, 64);
        }
    }
    if (role != kRoleV) {
        cb_reserve_back(kGamma, 2);
        const uint32_t g0 = role == kRoleQ ? 0 : 2;
        noc_async_read(get_noc_addr(g0, gamma), get_write_ptr(kGamma), kTileBytes);
        noc_async_read(get_noc_addr(g0 + 1, gamma), get_write_ptr(kGamma) + kTileBytes, kTileBytes);
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
    // 2. position-dependent reads (rope row, page-table entry) overlap the head-row placement and the norm
    if (role != kRoleV) {
        noc_async_read(get_noc_addr(rope_pos, table), scratch + 256, table_page);
        cb_push_back(kGamma, 2);
    }
    const uint32_t entry = (rope_pos / kBlock) * 4;
    if (kv && live) {
        noc_async_read(get_noc_addr(t, page_table, entry & ~63u), scratch + 64, 64);
    }
    cb_reserve_back(kIn, 2);
    const uint32_t in = get_write_ptr(kIn);
    const uint32_t sel = (t & 1) * 32;
    for (uint32_t i = 0; i < nh; ++i) {
        for (uint32_t c = 0; c < kHeadTiles; ++c) {
            for (uint32_t f = 0; f < 2; ++f) {
                copy32(in + c * kTileBytes + row_face(i, f), gather + (i * 2 + c) * 128 + f * 64 + sel);
            }
        }
    }
    cb_push_back(kIn, 2);
    noc_async_read_barrier();
    // 3. cache tiles to patch (K/V), rope tiles (Q/K)
    if (kv) {
        const uint32_t block = live ? *reinterpret_cast<volatile uint32_t*>(scratch + 64 + (entry & 63u)) : 0;
        cb_reserve_back(kPosBox, 1);
        volatile uint32_t* box = reinterpret_cast<volatile uint32_t*>(get_write_ptr(kPosBox));
        box[0] = static_cast<uint32_t>(pos);
        box[1] = block;
        cb_push_back(kPosBox, 1);
        if (live) {
            const uint32_t rt = (rope_pos % kBlock) / 32;
            cb_reserve_back(kCache, 2 * nh);
            const uint32_t dst = get_write_ptr(kCache);
            for (uint32_t i = 0; i < nh; ++i) {
                for (uint32_t ct = 0; ct < kHeadTiles; ++ct) {
                    const uint32_t id = ((block * kKvHeads + h0 + i) * 2 + rt) * kHeadTiles + ct;
                    noc_async_read(get_noc_addr(id, cache), dst + (i * 2 + ct) * kBfp8TileBytes, kBfp8TileBytes);
                }
            }
        }
    }
    if (role != kRoleV) {
        cb_reserve_back(kCos, 2);
        cb_reserve_back(kSin, 2);
        const uint32_t cs = get_write_ptr(kCos), sn = get_write_ptr(kSin), row = scratch + 256;
        for (uint32_t i = 0; i < nh; ++i) {
            for (uint32_t c = 0; c < kHeadTiles; ++c) {
                for (uint32_t f = 0; f < 2; ++f) {
                    copy32(cs + c * kTileBytes + row_face(i, f), row + c * 64 + f * 32);
                    copy32(sn + c * kTileBytes + row_face(i, f), row + 128 + c * 64 + f * 32);
                }
            }
        }
        cb_push_back(kCos, 2);
        cb_push_back(kSin, 2);
    }
    if (kv && live) {
        noc_async_read_barrier();
        cb_push_back(kCache, 2 * nh);
    }
#elif defined(COMPILE_FOR_BRISC)
    if (role == kRoleQ) {
        const auto q_out = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                          get_common_arg_val<uint32_t>(0), kTileBytes);
        cb_wait_front(kQOut, 2);
        const uint32_t src = get_read_ptr(kQOut);
        for (uint32_t c = 0; c < kHeadTiles; ++c) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint64_t dst = get_noc_addr(t * kHeadTiles + c, q_out, ((h0 >> 4) * 2 + f) * 512 + (h0 & 15) * 32);
                noc_async_write(src + c * kTileBytes + f * 512, dst, nh * 32);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(kQOut, 2);
    } else {
        const auto cache = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                          get_common_arg_val<uint32_t>(role == kRoleV ? 2 : 1), kBfp8TileBytes);
        cb_wait_front(kPosBox, 1);
        volatile uint32_t* box = reinterpret_cast<volatile uint32_t*>(get_read_ptr(kPosBox));
        const int32_t pos = static_cast<int32_t>(box[0]);
        const uint32_t block = box[1];
        cb_wait_front(kKVOut, 2);
        if (pos >= 0) {
            const uint32_t r = static_cast<uint32_t>(pos) % kBlock;
            const uint32_t rt = r / 32, rr = r % 32;
            cb_wait_front(kCache, 2 * nh);
            const uint32_t tiles = get_read_ptr(kCache), src = get_read_ptr(kKVOut);
            for (uint32_t i = 0; i < nh; ++i) {
                for (uint32_t ct = 0; ct < kHeadTiles; ++ct) {
                    const uint32_t dst = tiles + (i * 2 + ct) * kBfp8TileBytes;
                    const uint32_t s = src + ct * kBfp8TileBytes;
                    for (uint32_t f = 0; f < 2; ++f) {
                        const uint32_t face = (rr >> 4) * 2 + f;
                        reinterpret_cast<volatile uint8_t*>(dst)[face * 16 + (rr & 15)] =
                            reinterpret_cast<volatile uint8_t*>(s)[f * 16 + i];
                        volatile uint32_t* dm = reinterpret_cast<volatile uint32_t*>(dst + 64 + face * 256 + (rr & 15) * 16);
                        volatile uint32_t* sm = reinterpret_cast<volatile uint32_t*>(s + 64 + f * 256 + i * 16);
                        for (uint32_t w = 0; w < 4; ++w) {
                            dm[w] = sm[w];
                        }
                    }
                    const uint32_t id = ((block * kKvHeads + h0 + i) * 2 + rt) * kHeadTiles + ct;
                    noc_async_write(dst, get_noc_addr(id, cache), kBfp8TileBytes);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(kCache, 2 * nh);
        }
        cb_pop_front(kKVOut, 2);
        cb_pop_front(kPosBox, 1);
    }
#elif defined(COMPILE_FOR_TRISC)
    if (role == kRoleV) {
        pack_value();
    } else {
        rms_norm();
        rope(role == kRoleQ ? kQOut : kKVOut);
    }
#endif
}
