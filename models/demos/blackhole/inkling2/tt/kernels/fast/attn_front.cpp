// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 attention decode / verify front (fast/attn_decode.py), fused behind the projection: one core per kv head
// group g (8 cores) in the projection's program, ROWS tokens at consecutive positions p .. p + ROWS - 1 (decode: 1;
// verify: 1..8).  The projection workers (decode_matmul_stream.cpp, route mode) write every token's rows straight into
// this core's input tiles (token r: tile r of Qin / Rin / cIn, tile 3 + r of X) and bump semaphore 0 once per route:
// q of heads 3g..3g+2 (Qin row 4h + c = chunk c), r (Rin rows 0 / 4 / 8, cols 0-15; cols 16-31 zero rows from the
// sender), the k | v linear outputs (X rows 0-7), the sinks (cIn [0, 32): small tile 2g + 1 cols 16..18) and, DYNAMIC,
// the dynamic tap rows (cIn [64, 96) / [128, 160): small tiles 16 / 17, face g >= 4).  Meanwhile the reader fetches
// the conv inputs of positions p-3, p-2, p-1 from the position-indexed conv ring ([1, 1, 32, 2048] BF16 TILE, row
// pos % 32 = x[pos]) into X tiles 0..2, the per-layer fp32 taps (tile 4 g + lag: rows 0-3 = k chunks, rows 4-7 = v
// chunks of head g, 32 channels per row) and, for block-float caches, the (<= 2) cache tiles to patch.  X tiles
// 0 .. 3 + ROWS - 1 = x[p-3] .. x[p+ROWS-1]: token r's lag l input is X tile 3 + r - l (causal conv across tokens).
//
// Same math as attention.py Inkling2Attention.decode (fp32 SFPU, one BF16 rounding per stored value), per token:
//   conv  sliding: c = sum_lag x_lag * tap'_lag (residual + 1/sqrt(5) folded into tap')
//         DYNAMIC: c = (sum_lag x_lag * (tap_lag + dyn_lag) + x_0) * 1/3            -> BF16
//   k = c_k * min(22.63 / ||c_k||, 1); v = c_v * rsqrt(||c_v||^2 / 128 + eps)      -> BF16 -> cache row
//   q = q * min(22.63 / sqrt(128) / ||q||, 1 / sqrt(128)) [* tau]                   -> BF16
//   r = r * min(8 / ||r||, 1) [* tau]                                                -> BF16
// Sums of squares: FPU matmul X X^T of BF16 rows (exact products, FP32 DEST), the diagonal read back and summed
// in fp32 on the SFPU.  Vectors live in "chunk rows" (32 values per tile row): q head h chunk c = row 4h + c,
// r head h = row 4h, conv k chunk c = row c, v chunk c = row 4 + c, so a 4-row group maps to SFPU slots
// (2s, 2s + 1) of faces 0 / 1 and per-group scalars apply per slot.  Every token runs the single-token sequence
// (bitwise the same values as ROWS single-token steps).
//
// Outputs per token r at position pos = p + r: SDPA query q_out tiles (8 r + g) * 5 .. + 4 (rows 0-2 = heads
// 3g..3g+2, row 3+ zero; tile 4 = [r (16) | sink | 0..]), the K / V cache row (ring slot pos % 1088 or full row
// pos), the conv ring row pos % 32 = x[pos].  Nothing else is touched: a rejected verify token only wrote rows of its
// own position (rewritten before they are read again).
//
// Per-core args: [g, routes expected].  Common args (both data-movement RISCs): [P, S, conv ring, taps, cur_pos
// (int32 word r = position of token r), tau (fp32 tile, element (0, r) = tau of token r), k cache, v cache, q_out]
// (BATCH: then slot (int32 word r = physical slot of row r), page table (int32 [32, W] row-major, page r = row r's
// pool blocks); SDPA_SIGNAL: then the SDPA core count and their noc coordinates, see the end of the writer).
// Defines ROWS, DYNAMIC, RING, K_FMT / V_FMT: cache formats (0 BF16, 1 BFP8, 2 BFP4), constants.
//
// BATCH (batched serving, fast/attn_decode.py AttnDecodeFast.batch; ROWS <= 16): every row is its own sequence: row r
// at position cur_pos[r] in slot slot[r]: its conv-ring lags x[p-3..p-1] come from slot r's conv ring (pages
// slot * 64 ..; lag tiles of row r in cLag tiles 3 r + 3 - l), its K / V row goes to slot r's ring (pages slot * 1152
// ..) or, full layers, to pool block page_table[r][p / 64] (PT_PAGE = the page table's aligned page bytes), its
// conv-ring row to slot r's ring; the routed rows stay in X tiles 3 + r (tiles 0..2 unused).  Per token the math is
// the single-token sequence above (bitwise a one-row decode of that sequence).  Two rows may be replicas (pad rows:
// same slot, position and inputs): they write identical bytes to the same places (block-float tiles: every row
// patches its own copy of the tile read before any write).  The writer publishes "rows done" (semaphore ROWS_SEM of
// this core = r + 1 once row r's query tiles and K / V row are in DRAM) instead of the READY_SEM signal: each SDPA
// row starts as soon as its row is written (attn_sdpa.cpp polls it), overlapping the fronts' serial rows.  Block-float
// caches: a row is published only once no later row of the launch still writes back a tile of its 64-key block (the
// SDPA reads whole blocks; an SDPA read of a tile during its write-back returned torn rows, tt/NOTES-spec.md).
#ifndef BATCH
#define BATCH 0
#endif
#ifndef PT_PAGE
#define PT_PAGE 64
#endif
#ifndef ROWS_SEM
#define ROWS_SEM 6
#endif
#include <cstdint>
#include "read_vc.hpp"
#include "tools/profiler/kernel_profiler.hpp"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace front {
constexpr bool kDyn = DYNAMIC == 1;       // full layer: dynamic conv taps, tau
constexpr bool kRing = RING == 1;         // sliding ring cache
constexpr uint32_t kRows = ROWS;          // tokens (consecutive positions; BATCH: rows of independent sequences)
constexpr bool kBatch = BATCH == 1;
static_assert(kRows >= 1 && kRows <= (kBatch ? 16 : 8));
constexpr uint32_t kBf16 = 2048, kFp32 = 4096;
// cache formats (K_FMT / V_FMT): 0 BF16, 1 BFP8, 2 BFP4
constexpr uint32_t tile_bytes(uint32_t fmt) { return fmt == 0 ? 2048 : (fmt == 1 ? 1088 : 576); }
constexpr uint32_t mant_row(uint32_t fmt) { return fmt == 1 ? 16 : 8; }  // block-float mantissa bytes per face row
constexpr uint32_t kFmt[2] = {K_FMT, V_FMT};
constexpr bool kAnyBlock = K_FMT != 0 || V_FMT != 0;
constexpr uint32_t kSlot = 1088;          // cache tile slots in cCache (8 per tile group: K chunks 0-3, V chunks 0-3)
constexpr uint32_t kRingSlots = 1088;     // ring modulus (attention.RING_SLOTS)
constexpr uint32_t kRingTiles = 36;       // ring cache rows / 32 (1088 + 64 zero rows)
constexpr uint32_t kConvRing = 32;        // conv ring rows (attention.CONV_RING)
// CBs
// cQin / cRin / cX / cIn are written remotely by the projection workers (declared first on every core of the program:
// the same L1 address everywhere)
constexpr uint32_t cQin = 26, cRin = 27, cX = 28, cIn = 29, cT = 3, cSc = 4, cScW = 5, cCache = 6, cQasm = 7,
                   cStage = 8, cLag = 9;
constexpr uint32_t cConv = 16, cG = 17, cF = 18, cQo = 19, cRo = 20, cKo = 21, cVo = 22;
// scalar page words (token r: word 16 r + w); wTile (BATCH, writer page): the row's first cache tile of group g;
// wGrp (BATCH, compute page): lags taken from the routed rows | base row << 8; wOwn / wFlr (BATCH, writer page): the
// row whose cache-tile copy this row patches (| 0x100: a replica, writes no block-float tile) / the row that writes
// that copy back; wBlk (BATCH, writer page, block-float): the last row of the launch writing into the row's 64-key
// block (verify groups below)
constexpr uint32_t wTau = 0, wBlk = 1, wDk = 1, wDv = 5, wPos = 9, wSink = 10, wTile = 13, wGrp = 14, wOwn = 14,
                   wFlr = 15, kScStride = 16;
// BATCH: per-slot page strides of the conv ring (32 x 2048 BF16) and the sliding ring (8 heads x 36 x 4 tiles)
constexpr uint32_t kConvSlotPages = 64, kRingSlotPages = 8 * kRingTiles * 4;
// cStage layout: [0, 64) cur_pos, [64, 128) tau, BATCH: [128, 192) slot words, [192, 1216) page-table chunks (64 B per
// row), [1216, ..) conv-ring tile images (row r, chunk c: 2 KB at kLagStage + (8 r + c) * 2048, only the half / halves
// holding rows p-3..p-1 read); else [128, ..) the 576 B conv-ring reads
constexpr uint32_t kSlotStage = 128, kPtStage = 192, kLagStage = 1216;
// cache slot (ring) / row (full) of position pos, and the first cache tile (4 dim tiles) of that row of group g
FORCE_INLINE uint32_t cache_slot(uint32_t pos) { return kRing ? pos % kRingSlots : pos; }
FORCE_INLINE uint32_t row_tile(uint32_t slot, uint32_t g) {
    return kRing ? (g * kRingTiles + slot / 32) * 4 : (((slot / 64) * 8 + g) * 2 + (slot % 64) / 32) * 4;
}
// BATCH: the first cache tile (4 dim tiles) of group g of the row at position pos: sliding in slot `slot`'s ring,
// full in pool block `blk` (the page-table entry of pos / 64)
FORCE_INLINE uint32_t row_tile_b(uint32_t pos, uint32_t g, uint32_t slot, uint32_t blk) {
    return kRing ? slot * kRingSlotPages + row_tile(pos % kRingSlots, g)
                 : ((blk * 8 + g) * 2 + (pos % 64) / 32) * 4;
}
}  // namespace front

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace front {
FORCE_INLINE void copy32(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w];
    }
}
FORCE_INLINE void zero_l1(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + off, bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE);
    }
}
// byte offset of (row, face column f) in a BF16 tile
FORCE_INLINE uint32_t rf(uint32_t row, uint32_t f) { return ((row >> 4) * 2 + f) * 512 + (row & 15) * 32; }
FORCE_INLINE auto dram(uint32_t arg, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(arg), page);
}
#if SDPA_SIGNAL
// fused with the SDPA: +1 on semaphore READY_SEM + g of every SDPA core (common args kSig: count, then (noc x, noc y)
// per SDPA core; kSig = 9, BATCH 11)
constexpr uint32_t kSig = kBatch ? 11 : 9;
FORCE_INLINE void signal_sdpa(uint32_t id) {
    const uint32_t n = get_common_arg_val<uint32_t>(kSig);
    const uint32_t sem = get_semaphore(id);
    for (uint32_t i = 0; i < n; ++i) {
        noc_semaphore_inc(get_noc_addr(get_common_arg_val<uint32_t>(kSig + 1 + 2 * i),
                                       get_common_arg_val<uint32_t>(kSig + 2 + 2 * i), sem),
                          1);
    }
    noc_async_atomic_barrier();
}
#endif
}  // namespace front
#endif

#if defined(COMPILE_FOR_TRISC)
#ifdef TRISC_MATH
namespace ckernel::sfpu {
// slot 0 of the tile at the call's dst index (element (0, 0) lives there): ((a + b) + c) + d
inline void ink_sum4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    sfpi::vFloat v = Converter::as_float(a);
    v = v + sfpi::vFloat(Converter::as_float(b));
    v = v + sfpi::vFloat(Converter::as_float(c));
    v = v + sfpi::vFloat(Converter::as_float(d));
    sfpi::dst_reg[0] = v;
}
inline void ink_mul0(uint32_t m) {
    const sfpi::vFloat v = sfpi::dst_reg[0];
    sfpi::dst_reg[0] = v * sfpi::vFloat(Converter::as_float(m));
}
inline void ink_add0(uint32_t m) {
    const sfpi::vFloat v = sfpi::dst_reg[0];
    sfpi::dst_reg[0] = v + sfpi::vFloat(Converter::as_float(m));
}
inline void ink_min0(uint32_t m) {
    sfpi::vFloat v = sfpi::dst_reg[0];
    const sfpi::vFloat c = Converter::as_float(m);
    v_if(v > c) { v = c; }
    v_endif;
    sfpi::dst_reg[0] = v;
}
// The 8 factors of a token in ONE SFPU call, slot t of the call's tile (face 0 rows 4 (t / 2) .. + 3, columns of
// parity t % 2) = factor t: per factor the ops of the per-tile sequence (ink_sum4 / ink_mul0 / ink_add0, the rsqrt of
// calculate_rsqrt (APPROX false, reciprocal, its sqrt_init constants loaded by rsqrt_tile_init), ink_mul0 / ink_min0)
// in the same order, in FP32 (DEST is FP32: the stores in between were exact): bitwise the same values.
// dg: Gram diagonals [conv rows 0-11 | q rows 0-11 | r rows 0-11]; DYN: q / r * tau.
sfpi_inline sfpi::vFloat ink_s4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    sfpi::vFloat v = Converter::as_float(a);
    v = v + sfpi::vFloat(Converter::as_float(b));
    v = v + sfpi::vFloat(Converter::as_float(c));
    v = v + sfpi::vFloat(Converter::as_float(d));
    return v;
}
sfpi_inline sfpi::vFloat ink_mul(sfpi::vFloat v, uint32_t m) { return v * sfpi::vFloat(Converter::as_float(m)); }
sfpi_inline sfpi::vFloat ink_min(sfpi::vFloat v, uint32_t m) {
    const sfpi::vFloat c = Converter::as_float(m);
    v_if(v > c) { v = c; }
    v_endif;
    return v;
}
sfpi_inline sfpi::vFloat ink_rsqrt(sfpi::vFloat v) { return _calculate_sqrt_body_<APPROX, true, false>(v); }
template <uint32_t H, bool DYN>  // q head H (slot 2 + H), r head H (slot 5 + H)
sfpi_inline void ink_qr(const uint32_t* dg, uint32_t tau) {
    sfpi::vFloat q = ink_rsqrt(ink_s4(dg[12 + 4 * H], dg[13 + 4 * H], dg[14 + 4 * H], dg[15 + 4 * H]));
    q = ink_min(ink_mul(q, Q_MUL_BITS), Q_CAP_BITS);
    sfpi::vFloat r = ink_rsqrt(ink_s4(dg[24 + 4 * H], 0u, 0u, 0u));
    r = ink_min(ink_mul(r, R_CLIP_BITS), 0x3F800000u);
    if constexpr (DYN) {
        q = ink_mul(q, tau);
        r = ink_mul(r, tau);
    }
    sfpi::dst_reg[2 + H] = q;
    sfpi::dst_reg[5 + H] = r;
}
template <bool DYN>
inline void ink_factors(const uint32_t* dg, uint32_t tau) {
    sfpi::vFloat k = ink_rsqrt(ink_s4(dg[0], dg[1], dg[2], dg[3]));
    sfpi::dst_reg[0] = ink_min(ink_mul(k, QK_CLIP_BITS), 0x3F800000u);  // k: min(22.63 rsqrt, 1)
    sfpi::vFloat v = ink_mul(ink_s4(dg[4], dg[5], dg[6], dg[7]), 0x3C000000u);  // v: rsqrt(ss / 128 + eps)
    v = v + sfpi::vFloat(Converter::as_float(EPS_BITS));
    sfpi::dst_reg[1] = ink_rsqrt(v);
    ink_qr<0, DYN>(dg, tau);
    ink_qr<1, DYN>(dg, tau);
    ink_qr<2, DYN>(dg, tau);
}
// rows 4s..4s+3 (slots 2s, 2s+1 of faces 0 and 1) *= f_s
inline void ink_scale3(uint32_t f0, uint32_t f1, uint32_t f2) {
    const uint32_t f[3] = {f0, f1, f2};
    for (uint32_t s = 0; s < 3; ++s) {
        const sfpi::vFloat m = Converter::as_float(f[s]);
        for (uint32_t face = 0; face < 2; ++face) {
            for (uint32_t i = 0; i < 2; ++i) {
                const uint32_t o = face * 8 + 2 * s + i;
                sfpi::dst_reg[o] = sfpi::dst_reg[o] * m;
            }
        }
    }
}
// conv over rows 0-7 (slots 0..3 of faces 0, 1): DEST tiles 0..3 = x_lag, 4..7 = taps_lag; result into tile 0
inline void ink_conv_static() {
    for (uint32_t face = 0; face < 2; ++face) {
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t o = face * 8 + i;
            sfpi::vFloat acc = sfpi::dst_reg[3 * 32 + o] * sfpi::dst_reg[7 * 32 + o];
            sfpi::vFloat t = sfpi::dst_reg[2 * 32 + o] * sfpi::dst_reg[6 * 32 + o];
            acc = acc + t;
            t = sfpi::dst_reg[1 * 32 + o] * sfpi::dst_reg[5 * 32 + o];
            acc = acc + t;
            t = sfpi::dst_reg[o] * sfpi::dst_reg[4 * 32 + o];
            acc = acc + t;
            sfpi::dst_reg[o] = acc;
        }
    }
}
inline void ink_conv_dyn(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, uint32_t v0, uint32_t v1, uint32_t v2,
                         uint32_t v3, uint32_t scale) {
    const uint32_t kk[4] = {k0, k1, k2, k3}, vv[4] = {v0, v1, v2, v3};
    const sfpi::vFloat sc = Converter::as_float(scale);
    for (uint32_t face = 0; face < 2; ++face) {
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t o = face * 8 + i;
            const uint32_t* d = i < 2 ? kk : vv;  // rows 0-3 k, rows 4-7 v
            sfpi::vFloat w = sfpi::dst_reg[7 * 32 + o];
            w = w + sfpi::vFloat(Converter::as_float(d[3]));
            sfpi::vFloat acc = sfpi::dst_reg[3 * 32 + o] * w;
            for (int lag = 2; lag >= 0; --lag) {
                w = sfpi::dst_reg[(4 + lag) * 32 + o];
                w = w + sfpi::vFloat(Converter::as_float(d[lag]));
                sfpi::vFloat t = sfpi::dst_reg[lag * 32 + o] * w;
                acc = acc + t;
            }
            const sfpi::vFloat x0 = sfpi::dst_reg[o];
            acc = acc + x0;
            sfpi::dst_reg[o] = acc * sc;
        }
    }
}
}  // namespace ckernel::sfpu
#endif

namespace front {
#define INK_SFPU(fn, idst, ...) \
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, fn, idst, VectorMode::RC_custom, ##__VA_ARGS__)))
#define INK_RSQRT0(idst)                                                                                         \
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, 1, DST_ACCUM_MODE, false, false), \
                          idst, VectorMode::RC_custom)))
// L1 words of a CB's front tile on every TRISC: UNPACK resolves the address (after its cb_wait_front) and hands it to
// MATH / PACK through the mailbox ONCE; each thread then reads L1 itself (bursts of read_tile_value overrun the
// mailbox and hang).
FORCE_INLINE const volatile uint32_t* l1_words(uint32_t cb, uint32_t tile) {
    return reinterpret_cast<const volatile uint32_t*>(get_tile_address(cb, tile));
}

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile(dst + i, cb, i);
    }
    cb_push_back(cb, n);
}

// BATCH: per token the decode's sequence (conv over the row's own lag tiles); the 8 factors in ONE SFPU call
// (ink_factors: slots 0..7 of one DEST tile, one FP32 tile packed; the same per-lane ops in the same order).  (Tokens
// in pairs through the Gram / factor / scale phases measured no faster: the phases are work-bound.)
FORCE_INLINE void compute_batch() {
    cb_wait_front(cSc, 1);
    cb_wait_front(cX, 3 + kRows);
    cb_wait_front(cLag, 3 * kRows);
    cb_wait_front(cT, 4);
    cb_wait_front(cQin, kRows);
    cb_wait_front(cRin, kRows);
    const volatile uint32_t* sc = l1_words(cSc, 0);
    for (uint32_t tok = 0; tok < kRows; ++tok) {
        const uint32_t sb = tok * kScStride;
        {
            DeviceZoneScopedN("f-conv");
            // ---- 1. conv: DEST 0 = x (X tile 3 + tok), 1..3 = lags 1..3: the row's own conv-ring rows (cLag tile
            // 3 tok + 3 - l) or, verify groups, the routed rows of its group (X tile 3 + base - l: not in the ring
            // yet), 4..7 = taps_lag (FP32, UnpackToDestFp32)
            const uint32_t gw = sc[sb + wGrp], nx = gw & 0xFFu, br = gw >> 8;
            tile_regs_acquire();
            reconfig_data_format_srca(cX);
            copy_tile_to_dst_init_short(cX);
            copy_tile(cX, 3 + tok, 0);
            for (uint32_t l = 1; l < 4; ++l) {
                if (l <= nx) {
                    copy_tile(cX, 3 + br - l, l);
                } else {
                    copy_tile(cLag, 3 * tok + 3 - l, l);
                }
            }
            copy_tile_to_dst_init_short_with_dt(cX, cT);
            for (uint32_t l = 0; l < 4; ++l) {
                copy_tile(cT, l, 4 + l);
            }
            binop_with_scalar_tile_init();
            if constexpr (kDyn) {
                uint32_t d[8];
                for (uint32_t i = 0; i < 8; ++i) {
                    d[i] = sc[sb + wDk + i];
                }
                INK_SFPU(ink_conv_dyn, 0, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], CONV_SCALE);
            } else {
                INK_SFPU(ink_conv_static, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cConv, 1);
            tile_regs_release();
        }
        DeviceZoneScopedN("f-norms");
        // ---- 2. Gram tiles G = X X^T of conv (DEST 0), q (1), r (2)
        cb_wait_front(cConv, 1);
        tile_regs_acquire();
        const uint32_t src[3] = {cConv, cQin, cRin};
        for (uint32_t i = 0; i < 3; ++i) {
            const uint32_t t = i == 0 ? 0 : tok;
            reconfig_data_format(src[i], src[i]);
            matmul_block_init(src[i], src[i], 1, 1, 1, 1);
            matmul_block(src[i], src[i], t, t, i, 1, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cG, 3);
        tile_regs_release();
        // ---- 3. factors: slot t of DEST tile 0 (face 0 rows 4 (t / 2) .. + 3, columns of parity t % 2) = factor t:
        // 0 k, 1 v, 2..4 q heads, 5..7 r heads
        {
            DeviceZoneScopedN("f-fac");
            cb_wait_front(cG, 3);
            uint32_t dg[3][12];
            {
                const volatile uint32_t* gt = l1_words(cG, 0);
                for (uint32_t t = 0; t < 3; ++t) {
                    for (uint32_t i = 0; i < 12; ++i) {
                        dg[t][i] = gt[t * 1024 + i * 16 + i];
                    }
                }
            }
            tile_regs_acquire();
            rsqrt_tile_init<false>();  // the sqrt_init constants of the rsqrt (the SFPU init of the call)
            INK_SFPU(ink_factors<kDyn>, 0, &dg[0][0], sc[sb + wTau]);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cF, 1);
            tile_regs_release();
            cb_pop_front(cG, 3);
        }
        // ---- 4. scale: conv rows (k: rows 0-3, v: rows 4-7), q heads, r heads; factor t = FP32 word 64 (t / 2) + t % 2
        // of the factor tile (the first element of slot t)
        cb_wait_front(cF, 1);
        uint32_t f[8];
        {
            const volatile uint32_t* ft = l1_words(cF, 0);
            for (uint32_t t = 0; t < 8; ++t) {
                f[t] = ft[(t >> 1) * 64 + (t & 1)];
            }
        }
        tile_regs_acquire();
        reconfig_data_format_srca(cConv);
        copy_tile_to_dst_init_short(cConv);
        copy_tile(cConv, 0, 0);
        copy_tile(cQin, tok, 1);
        copy_tile(cRin, tok, 2);
        binop_with_scalar_tile_init();
        INK_SFPU(ink_scale3, 0, f[0], f[1], 0x3F800000u);
        INK_SFPU(ink_scale3, 1, f[2], f[3], f[4]);
        INK_SFPU(ink_scale3, 2, f[5], f[6], f[7]);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cKo, 1);  // the conv tile in the K and the V cache format
        pack_to(0, cVo, 1);
        pack_to(1, cQo, 1);
        pack_to(2, cRo, 1);
        tile_regs_release();
        cb_pop_front(cF, 1);
        cb_pop_front(cConv, 1);
    }
    cb_pop_front(cQin, kRows);
    cb_pop_front(cRin, kRows);
    cb_pop_front(cX, 3 + kRows);
    cb_pop_front(cLag, 3 * kRows);
    cb_pop_front(cT, 4);
    cb_pop_front(cSc, 1);
}

// decode / verify: one token at a time
FORCE_INLINE void compute() {
    cb_wait_front(cSc, 1);
    cb_wait_front(cX, 3 + kRows);
    cb_wait_front(cT, 4);
    cb_wait_front(cQin, kRows);
    cb_wait_front(cRin, kRows);
    // scalars read from L1 by every TRISC (l1_words: one mailbox transfer of the address; never inside MATH(...))
    const volatile uint32_t* sc = l1_words(cSc, 0);
    for (uint32_t tok = 0; tok < kRows; ++tok) {
        const uint32_t sb = tok * kScStride;
        {
        DeviceZoneScopedN("f-conv");
        // ---- 1. conv: DEST 0..3 = x_lag = X tile 3 + tok - lag (BF16), 4..7 = taps_lag (FP32, UnpackToDestFp32)
        tile_regs_acquire();
        reconfig_data_format_srca(cX);
        copy_tile_to_dst_init_short(cX);
        for (uint32_t l = 0; l < 4; ++l) {
            copy_tile(cX, 3 + tok - l, l);
        }
        copy_tile_to_dst_init_short_with_dt(cX, cT);
        for (uint32_t l = 0; l < 4; ++l) {
            copy_tile(cT, l, 4 + l);
        }
        binop_with_scalar_tile_init();
        if constexpr (kDyn) {
            uint32_t d[8];
            for (uint32_t i = 0; i < 8; ++i) {
                d[i] = sc[sb + wDk + i];
            }
            INK_SFPU(ink_conv_dyn, 0, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], CONV_SCALE);
        } else {
            INK_SFPU(ink_conv_static, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cConv, 1);
        tile_regs_release();
        }
        DeviceZoneScopedN("f-norms");
        // ---- 2. Gram tiles G = X X^T of conv (DEST 0), q (1), r (2)
        cb_wait_front(cConv, 1);
        tile_regs_acquire();
        const uint32_t src[3] = {cConv, cQin, cRin};
        for (uint32_t i = 0; i < 3; ++i) {
            const uint32_t t = i == 0 ? 0 : tok;
            reconfig_data_format(src[i], src[i]);
            matmul_block_init(src[i], src[i], 1, 1, 1, 1);
            matmul_block(src[i], src[i], t, t, i, 1, 1, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cG, 3);
        tile_regs_release();
        // ---- 3. factors: DEST 0 k, 1 v, 2..4 q heads, 5..7 r heads (element (0, 0) of each)
        {
        DeviceZoneScopedN("f-fac");
        cb_wait_front(cG, 3);
        uint32_t dg[3][12];
        {
            const volatile uint32_t* gt = l1_words(cG, 0);
            for (uint32_t t = 0; t < 3; ++t) {
                for (uint32_t i = 0; i < 12; ++i) {
                    dg[t][i] = gt[t * 1024 + i * 16 + i];
                }
            }
        }
        const uint32_t tau = sc[sb + wTau];
        tile_regs_acquire();
        binop_with_scalar_tile_init();
        INK_SFPU(ink_sum4, 0, dg[0][0], dg[0][1], dg[0][2], dg[0][3]);
        INK_SFPU(ink_sum4, 1, dg[0][4], dg[0][5], dg[0][6], dg[0][7]);
        for (uint32_t h = 0; h < 3; ++h) {
            INK_SFPU(ink_sum4, 2 + h, dg[1][4 * h], dg[1][4 * h + 1], dg[1][4 * h + 2], dg[1][4 * h + 3]);
            INK_SFPU(ink_sum4, 5 + h, dg[2][4 * h], 0u, 0u, 0u);
        }
        // v: ss / 128 + eps (before the rsqrt)
        INK_SFPU(ink_mul0, 1, 0x3C000000u);
        INK_SFPU(ink_add0, 1, EPS_BITS);
        rsqrt_tile_init<false>();
        for (uint32_t t = 0; t < 8; ++t) {
            INK_RSQRT0(t);
        }
        binop_with_scalar_tile_init();
        INK_SFPU(ink_mul0, 0, QK_CLIP_BITS);  // k: min(22.63 rsqrt, 1)
        INK_SFPU(ink_min0, 0, 0x3F800000u);
        for (uint32_t h = 0; h < 3; ++h) {
            INK_SFPU(ink_mul0, 2 + h, Q_MUL_BITS);  // q: min(22.63 / sqrt(128) rsqrt, 1 / sqrt(128)) [* tau]
            INK_SFPU(ink_min0, 2 + h, Q_CAP_BITS);
            INK_SFPU(ink_mul0, 5 + h, R_CLIP_BITS);  // r: min(8 rsqrt, 1) [* tau]
            INK_SFPU(ink_min0, 5 + h, 0x3F800000u);
            if constexpr (kDyn) {
                INK_SFPU(ink_mul0, 2 + h, tau);
                INK_SFPU(ink_mul0, 5 + h, tau);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cF, 8);
        tile_regs_release();
        cb_pop_front(cG, 3);
        }
        // ---- 4. scale: conv rows (k: rows 0-3, v: rows 4-7), q heads, r heads
        cb_wait_front(cF, 8);
        uint32_t f[8];
        {
            const volatile uint32_t* ft = l1_words(cF, 0);
            for (uint32_t t = 0; t < 8; ++t) {
                f[t] = ft[t * 1024];
            }
        }
        tile_regs_acquire();
        reconfig_data_format_srca(cConv);
        copy_tile_to_dst_init_short(cConv);
        copy_tile(cConv, 0, 0);
        copy_tile(cQin, tok, 1);
        copy_tile(cRin, tok, 2);
        binop_with_scalar_tile_init();
        INK_SFPU(ink_scale3, 0, f[0], f[1], 0x3F800000u);
        INK_SFPU(ink_scale3, 1, f[2], f[3], f[4]);
        INK_SFPU(ink_scale3, 2, f[5], f[6], f[7]);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cKo, 1);  // the conv tile (k rows 0-3, v rows 4-7) in the K and the V cache format
        pack_to(0, cVo, 1);
        pack_to(1, cQo, 1);
        pack_to(2, cRo, 1);
        tile_regs_release();
        cb_pop_front(cF, 8);
        cb_pop_front(cConv, 1);
    }
    cb_pop_front(cQin, kRows);
    cb_pop_front(cRin, kRows);
    cb_pop_front(cX, 3 + kRows);
    cb_pop_front(cT, 4);
    cb_pop_front(cSc, 1);
}
}  // namespace front
#endif

#if defined(COMPILE_FOR_NCRISC)
namespace front {
// reader: positions, tau, conv-ring rows of p-3 .. p-1, taps, block-float cache tiles (all while the projection runs);
// then the routed projection rows (semaphore 0), scalar pages, the tokens' conv-ring rows
FORCE_INLINE void reader(uint32_t g) {
    const uint32_t expected = get_arg_val<uint32_t>(1);  // routes (+ zero routes) the projection workers send
    const auto ring = dram(2, kBf16);
    const auto taps = dram(3, kFp32);
    const auto cur_pos = dram(4, 4);
    const auto tau = dram(5, kFp32);
    const uint32_t stage = get_write_ptr(cStage);
    noc_async_read(get_noc_addr(0, cur_pos, 0), stage, 64);
    noc_async_read(get_noc_addr(0, tau, 0), stage + 64, 64);
    if constexpr (kBatch) {
        noc_async_read(get_noc_addr(0, dram(9, 4), 0), stage + kSlotStage, 64);
    }
    cb_reserve_back(cQin, kRows);
    cb_reserve_back(cRin, kRows);
    cb_reserve_back(cX, 3 + kRows);
    const uint32_t xt = get_write_ptr(cX);
    cb_reserve_back(cT, 4);
    for (uint32_t l = 0; l < 4; ++l) {
        noc_async_read(get_noc_addr(g * 4 + l, taps), get_write_ptr(cT) + l * kFp32, kFp32);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* posw = reinterpret_cast<volatile tt_l1_ptr int32_t*>(stage);
    [[maybe_unused]] const uint32_t p0 = static_cast<uint32_t>(posw[0]);
    // BATCH: per row its slot word and its cache tile (group g)
    [[maybe_unused]] volatile tt_l1_ptr int32_t* slotw =
        reinterpret_cast<volatile tt_l1_ptr int32_t*>(stage + kSlotStage);
    [[maybe_unused]] uint32_t rtile[kBatch ? kRows : 1], grpw[kBatch ? kRows : 1], ownw[kBatch ? kRows : 1],
        flrw[kBatch ? kRows : 1], blkw[kBatch ? kRows : 1];
    if constexpr (kBatch) {
        cb_reserve_back(cLag, 3 * kRows);
        // full layers: the 64 B page-table chunk holding entry p / 64 of row r
        if constexpr (!kRing) {
            const auto pt = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(10),
                                           PT_PAGE);
            for (uint32_t r = 0; r < kRows; ++r) {
                const uint32_t blk = static_cast<uint32_t>(posw[r]) / 64;
                noc_async_read(get_noc_addr(r, pt, (blk * 4) & ~63u), stage + kPtStage + 64 * r, 64);
            }
        }
        // x[p - l], l = 1..3, of every row from its slot's conv ring: per (row, chunk tile) ONE read of the tile half
        // holding the 3 rows (faces 2h, 2h + 1: 1 KB contiguous), or of the whole tile when they straddle the halves
        // (p % 32 in {1, 2, 17, 18}): 8 reads per row instead of 48 small ones
        {
        DeviceZoneScopedN("f-lagrd");
        for (uint32_t r = 0; r < kRows; ++r) {
            const uint32_t p = static_cast<uint32_t>(posw[r]), sbase = static_cast<uint32_t>(slotw[r]) * kConvSlotPages;
            const uint32_t a = (p + kConvRing - 3) % kConvRing, b = (p + kConvRing - 1) % kConvRing;
            const bool one = a <= b && (a >> 4) == (b >> 4);
            const uint32_t off = one ? (a >> 4) * 1024 : 0, len = one ? 1024 : 2048;
            for (uint32_t c = 0; c < 8; ++c) {
                const uint32_t tile = sbase + (c < 4 ? 4 * g + c : 32 + 4 * g + c - 4);
                noc_async_read(get_noc_addr(tile, ring, off), stage + kLagStage + (8 * r + c) * kBf16 + off, len);
            }
        }
        noc_async_read_barrier();
        }
        for (uint32_t r = 0; r < kRows; ++r) {
            const uint32_t p = static_cast<uint32_t>(posw[r]);
            const uint32_t blk = kRing ? 0u
                                       : static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr int32_t*>(
                                             stage + kPtStage + 64 * r)[(p / 64) % 16]);
            rtile[r] = row_tile_b(p, g, static_cast<uint32_t>(slotw[r]), blk);
        }
        // verify groups (batched speculation, fast/attn_decode.py AttnDecodeFast.batch): rows of one sequence at
        // consecutive positions in consecutive rows.  Row r replicates base row br (the first row with its slot and
        // position: pad rows); br's lags at positions of earlier rows of its group come from those rows' routed X
        // tiles (the conv ring gets them only at the end of this launch).  Block-float caches: the rows writing the
        // same cache tile patch ONE copy (the first such row's) and only the last of them writes it back; replicas
        // write nothing (their rows equal the base row's).
        for (uint32_t r = 0; r < kRows; ++r) {
            uint32_t br = r;
            for (uint32_t b = 0; b < r; ++b) {
                if (slotw[b] == slotw[r] && posw[b] == posw[r]) {
                    br = b;
                    break;
                }
            }
            uint32_t nx = 0;
            while (nx < 3 && br > nx && slotw[br - nx - 1] == slotw[br] && posw[br - nx - 1] + 1 + (int32_t)nx == posw[br]) {
                ++nx;
            }
            grpw[r] = nx | br << 8;
            uint32_t own = r;
            flrw[r] = r;
            if (br == r) {
                for (uint32_t b = 0; b < r; ++b) {
                    if ((ownw[b] >> 8) == 0 && rtile[b] == rtile[r]) {
                        own = ownw[b];
                        break;
                    }
                }
                flrw[own] = r;  // the copy's last writer so far
            }
            ownw[r] = own | (br != r ? 0x100u : 0u);
        }
        // block-float caches: the last row writing into each row's 64-key block (ring: the slot's ring block; full:
        // the pool block), i.e. the block's last tile write-back: the row is done (its SDPA may read the block) only
        // after it.  A verify group straddling the two tiles of a block (p % 64 == 31 -> 32) used to publish its
        // first-tile rows while a later row rewrote the second tile; SDPA reads of that tile during the write-back
        // intermittently saw the new row with huge K / V values (in exactly its nonzero dims): a masked key that won
        // the softmax (tt/NOTES-spec.md, BSpec3)
        if constexpr (kAnyBlock) {
            const auto blk = [&](uint32_t t) {
                return kRing ? (static_cast<uint32_t>(slotw[t]) << 16) | (static_cast<uint32_t>(posw[t]) % kRingSlots) / 64
                             : rtile[t] >> 6;
            };
            for (uint32_t r = 0; r < kRows; ++r) {
                blkw[r] = r;
                for (uint32_t t = r + 1; t < kRows; ++t) {
                    if ((ownw[t] >> 8) == 0 && blk(t) == blk(r)) {
                        blkw[r] = t;
                    }
                }
            }
        }
        // block-float caches: every row's 4 + 4 cache tiles (group r), patched and rewritten by the writer
        if constexpr (kAnyBlock) {
            cb_reserve_back(cCache, 1);
            const uint32_t ca = get_write_ptr(cCache);
            for (uint32_t r = 0; r < kRows; ++r) {
                for (uint32_t w = 0; w < 2; ++w) {
                    if (kFmt[w] == 0) {
                        continue;
                    }
                    const auto cache = dram(6 + w, tile_bytes(kFmt[w]));
                    for (uint32_t c = 0; c < 4; ++c) {
                        noc_async_read(get_noc_addr(rtile[r] + c, cache, 0), ca + (r * 8 + w * 4 + c) * kSlot,
                                       tile_bytes(kFmt[w]));
                    }
                }
            }
        }
        // lag tile 3 r + 3 - l, row c (faces 0 / 1) <- conv-ring row (p - l) % 32 of chunk c: 32 B local NOC copies
        // (one-packet reads with the source core / size state set once)
        {
        DeviceZoneScopedN("f-lagcp");
        const uint32_t lt = get_write_ptr(cLag);
        noc_async_read_one_packet_set_state(get_noc_addr(0), 32);
        for (uint32_t r = 0; r < kRows; ++r) {
            const uint32_t p = static_cast<uint32_t>(posw[r]);
            for (uint32_t l = 1; l < 4; ++l) {
                const uint32_t q = (p + kConvRing - l) % kConvRing;
                for (uint32_t c = 0; c < 8; ++c) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_read_one_packet_with_state(stage + kLagStage + (8 * r + c) * kBf16 + rf(q, f),
                                                             lt + (3 * r + 3 - l) * kBf16 + rf(c, f));
                    }
                }
            }
        }
        }
        {
        DeviceZoneScopedN("f-cachewait");
        noc_async_read_barrier();
        }
    } else {
        // x[p - l], l = 1..3, from conv-ring row (p - l) % 32 -> X tile 3 - l rows 0-7: one 576 B read per ring tile
        // from the 64 B-aligned start covers the row in faces 0 and 1 (offsets o and o + 512, o - start in {0, 32})
        uint32_t n = 0;
        for (uint32_t l = 1; l < 4; ++l) {
            const uint32_t o = rf((p0 + kConvRing - l) % kConvRing, 0);
            for (uint32_t c = 0; c < 8; ++c) {
                const uint32_t tile = c < 4 ? 4 * g + c : 32 + 4 * g + c - 4;
                noc_async_read(get_noc_addr(tile, ring, o & ~63u), stage + 128 + 576 * n, 576);
                ++n;
            }
        }
        // block-float caches: the 4 + 4 cache tiles of each tile group (<= 2: consecutive slots span <= 2 tiles);
        // group 0 = token 0's tile, group 1 = the last token's (if different); the writer patches and rewrites them
        const uint32_t slot0 = cache_slot(p0), slotn = cache_slot(p0 + kRows - 1);
        const uint32_t tile0 = row_tile(slot0, g), tilen = row_tile(slotn, g);
        if constexpr (kAnyBlock) {
            cb_reserve_back(cCache, 1);
            const uint32_t ca = get_write_ptr(cCache);
            for (uint32_t grp = 0; grp < (tilen != tile0 ? 2u : 1u); ++grp) {
                const uint32_t rt = grp ? tilen : tile0;
                for (uint32_t w = 0; w < 2; ++w) {
                    if (kFmt[w] == 0) {
                        continue;
                    }
                    const auto cache = dram(6 + w, tile_bytes(kFmt[w]));
                    for (uint32_t c = 0; c < 4; ++c) {
                        noc_async_read(get_noc_addr(rt + c, cache, 0), ca + (grp * 8 + w * 4 + c) * kSlot,
                                       tile_bytes(kFmt[w]));
                    }
                }
            }
        }
        noc_async_read_barrier();
        n = 0;
        for (uint32_t l = 1; l < 4; ++l) {
            const uint32_t o = rf((p0 + kConvRing - l) % kConvRing, 0);
            for (uint32_t c = 0; c < 8; ++c) {
                for (uint32_t f = 0; f < 2; ++f) {
                    copy32(xt + (3 - l) * kBf16 + rf(c, f), stage + 128 + 576 * n + (o & 63u) + f * 512);
                }
                ++n;
            }
        }
    }
    // the projection rows: q (Qin rows 4h + c), r (Rin rows 0 / 4 / 8, cols 16.. zeroed by the sender), k | v
    // (X rows 0-7), sinks / dynamic taps (cIn: [0, 32) sinks row, [64, 96) k taps row, [128, 160) v taps row)
    {
        DeviceZoneScopedN("f-wait");
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait(sem, expected);
        noc_semaphore_set(sem, 0);
    }

    const uint32_t in = get_write_ptr(cIn);
    cb_reserve_back(cSc, 1);
    cb_reserve_back(cScW, 1);
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cSc));
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cScW));
    volatile tt_l1_ptr uint32_t* tauw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 64);
    for (uint32_t tok = 0; tok < kRows; ++tok) {
        const uint32_t sb = tok * kScStride, ti = in + tok * kBf16;
        sc[sb + wTau] = tauw[tok];
        sw[sb + wPos] = static_cast<uint32_t>(posw[tok]);
        if constexpr (kBatch) {
            sw[sb + wTile] = rtile[tok];
            sw[sb + wOwn] = ownw[tok];
            sw[sb + wFlr] = flrw[ownw[tok] & 0xFFu];
            if constexpr (kAnyBlock) {
                sw[sb + wBlk] = blkw[tok];
            }
            sc[sb + wGrp] = grpw[tok];
        }
        for (uint32_t h = 0; h < 3; ++h) {
            sw[sb + wSink + h] = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ti)[h];  // small tile 2g + 1, cols 16 + h
        }
        if constexpr (kDyn) {
            const uint32_t e = (4 * g) & 15;
            for (uint32_t l = 0; l < 4; ++l) {
                sc[sb + wDk + l] = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ti + 64)[e + l])
                                   << 16;
                sc[sb + wDv + l] = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ti + 128)[e + l])
                                   << 16;
            }
        }
    }
    cb_push_back(cQin, kRows);
    cb_push_back(cRin, kRows);
    cb_push_back(cX, 3 + kRows);
    if constexpr (kBatch) {
        cb_push_back(cLag, 3 * kRows);
    }
    cb_push_back(cT, 4);
    cb_push_back(cSc, 1);
    if constexpr (kAnyBlock) {
        cb_push_back(cCache, 1);
    }
    cb_push_back(cScW, 1);
    // conv ring: row (p + tok) % 32 <- x[p + tok] (X tile 3 + tok rows 0-7; BATCH: row p_tok % 32 of slot tok's ring)
    for (uint32_t tok = 0; tok < kRows; ++tok) {
        const uint32_t row = (kBatch ? static_cast<uint32_t>(posw[tok]) : p0 + tok) % kConvRing;
        const uint32_t sbase = kBatch ? static_cast<uint32_t>(slotw[tok]) * kConvSlotPages : 0u;
        for (uint32_t c = 0; c < 8; ++c) {
            const uint32_t tile = sbase + (c < 4 ? 4 * g + c : 32 + 4 * g + c - 4);
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write(xt + (3 + tok) * kBf16 + rf(c, f), get_noc_addr(tile, ring, rf(row, f)), 32);
            }
        }
    }
    noc_async_write_barrier();
#if SDPA_SIGNAL
    if (g == 0) {  // relay: every big projection worker's writes (incl. the gate rows in P) landed
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(DONE_SEM)), N_BIG);
        signal_sdpa(GATES_SEM);
    }
#endif
}
}  // namespace front
#endif

#if defined(COMPILE_FOR_BRISC)
namespace front {
// writer: per token, the SDPA query tiles and the K / V cache rows
FORCE_INLINE void writer(uint32_t g) {
    const auto q_out = dram(8, kBf16);
    cb_reserve_back(cQasm, 5);
    const uint32_t qa = get_write_ptr(cQasm);
    zero_l1(qa, 5 * kBf16);
    cb_wait_front(cScW, 1);
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cScW));
    const uint32_t slot0 = cache_slot(sw[wPos]);
    const uint32_t tile0 = row_tile(slot0, g);
    noc_async_read_barrier();
    if constexpr (kAnyBlock) {
        cb_wait_front(cCache, 1);
    }
    for (uint32_t tok = 0; tok < kRows; ++tok) {
        const uint32_t sb = tok * kScStride;
        const uint32_t slot = cache_slot(sw[sb + wPos]);
        // q: chunk row 4h + c of cQo -> row h of tile c; r: row 4h of cRo -> row h of tile 4 (cols 0-15); sink -> col 16
        cb_wait_front(cQo, 1);
        cb_wait_front(cRo, 1);
        if (tok > 0) {
            noc_async_writes_flushed();  // the previous token's query tiles have left qa
        }
        const uint32_t qo = get_read_ptr(cQo), ro = get_read_ptr(cRo);
        for (uint32_t h = 0; h < 3; ++h) {
            for (uint32_t c = 0; c < 4; ++c) {
                for (uint32_t f = 0; f < 2; ++f) {
                    copy32(qa + c * kBf16 + rf(h, f), qo + rf(4 * h + c, f));
                }
            }
            copy32(qa + 4 * kBf16 + rf(h, 0), ro + rf(4 * h, 0));
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(qa + 4 * kBf16 + rf(h, 1))[0] =
                static_cast<uint16_t>(sw[sb + wSink + h]);
        }
        for (uint32_t c = 0; c < 5; ++c) {
            noc_async_write(qa + c * kBf16, get_noc_addr((tok * 8 + g) * 5 + c, q_out), kBf16);
        }
        // K / V row
        const uint32_t r = slot % 32;
        // BATCH: every row its own tile (the reader's wTile); block-float: the copy of the tile's first writing row,
        // patched by every row of the tile, written back by the last (wFlr); replicas write nothing
        const uint32_t rt = kBatch ? sw[sb + wTile] : row_tile(slot, g);
        const uint32_t grp = kBatch ? sw[sb + wOwn] & 0xFFu : static_cast<uint32_t>(rt != tile0);
        [[maybe_unused]] const bool replica = kBatch && (sw[sb + wOwn] >> 8) != 0;
        // block-float: rewrite the patched tile group after its last token
        const bool flush = kBatch ? tok == sw[sb + wFlr] : (tok + 1 == kRows || row_tile(cache_slot(sw[sb + kScStride + wPos]), g) != rt);
        DeviceZoneScopedN("f-kvwrite");
        for (uint32_t w = 0; w < 2; ++w) {  // K then V: chunk c of the conv tile = row 4 w + c
            const uint32_t cb = w == 0 ? cKo : cVo;
            cb_wait_front(cb, 1);
            const uint32_t src = get_read_ptr(cb);
            const uint32_t fmt = kFmt[w];
            const auto cache = dram(6 + w, tile_bytes(fmt));
            for (uint32_t c = 0; c < 4; ++c) {
                const uint32_t sr = 4 * w + c;
                if (fmt == 0) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_write(src + rf(sr, f), get_noc_addr(rt + c, cache, rf(r, f)), 32);
                    }
                    continue;
                }
                if (replica) {
                    continue;
                }
                // packed block-float tile: 64 exponent bytes (face-major, one per face row), then the mantissas
                // (mant_row bytes per face row); patch row r of the cached tile, rewrite the whole tile
                const uint32_t mr = mant_row(fmt);
                const uint32_t dst = get_read_ptr(cCache) + (grp * 8 + w * 4 + c) * kSlot;
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t face = (r >> 4) * 2 + f;
                    reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst)[face * 16 + (r & 15)] =
                        reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src)[f * 16 + sr];
                    volatile tt_l1_ptr uint32_t* dm =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + 64 + face * 16 * mr + (r & 15) * mr);
                    volatile tt_l1_ptr uint32_t* sm =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + 64 + f * 16 * mr + sr * mr);
                    for (uint32_t k = 0; k < mr / 4; ++k) {
                        dm[k] = sm[k];
                    }
                }
                if (flush) {
                    noc_async_write(dst, get_noc_addr(rt + c, cache, 0), tile_bytes(fmt));
                }
            }
        }
        noc_async_writes_flushed();  // cKo / cVo / cQo / cRo pages are read by the writes above
        cb_pop_front(cQo, 1);
        cb_pop_front(cRo, 1);
        cb_pop_front(cKo, 1);
        cb_pop_front(cVo, 1);
#if SDPA_SIGNAL
        if constexpr (kBatch) {  // rows done (polled by the SDPA): the leading rows whose query tiles and K / V rows are
            // in DRAM (block-float: and every tile of their 64-key block written back, wBlk)
            noc_async_write_barrier();
            uint32_t d = tok + 1;
            if constexpr (kAnyBlock) {
                for (uint32_t q = 0; q <= tok; ++q) {
                    if ((sw[q * kScStride + wOwn] >> 8) == 0 && sw[q * kScStride + wBlk] > tok) {
                        d = q;
                        break;
                    }
                }
            }
            volatile tt_l1_ptr uint32_t* rs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ROWS_SEM));
            if (d > *rs) {
                *rs = d;
            }
        }
#endif
    }
    noc_async_write_barrier();
    cb_pop_front(cScW, 1);
    if constexpr (kAnyBlock) {
        cb_pop_front(cCache, 1);
    }
#if SDPA_SIGNAL
    if constexpr (!kBatch) {
        signal_sdpa(READY_SEM + g);  // this group's query tiles and K / V rows are in DRAM
    }
#endif
}
}  // namespace front
#endif

void kernel_main() {
    ink_set_read_vc();
    const uint32_t g = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_NCRISC)
    front::reader(g);
#elif defined(COMPILE_FOR_BRISC)
    front::writer(g);
#else
    compute_kernel_hw_startup(front::cX, front::cT, front::cConv);
    if constexpr (front::kBatch) {
        front::compute_batch();
    } else {
        front::compute();
    }
#endif
}
