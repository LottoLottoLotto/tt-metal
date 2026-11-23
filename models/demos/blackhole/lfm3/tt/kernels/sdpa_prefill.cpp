// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 prefill attention (tt/prefill/sdpa.py PrefillSDPA): causal (or sliding-window) GQA attention of one prefill
// chunk over the paged BFP8 KV cache, 16 q heads / 4 kv heads, head_dim 128, with the per-head sigmoid output gate in
// the epilogue.  Output: the gated, head-concatenated attention [1, 1, Tp, 2048] (column 128 h + e = head h) that
// o_proj takes.
//
// Work unit u = (kv head g = u % 4, query block b = u / 4): RT query row tiles (32 rows each) of the 4 q heads of
// group g, i.e. QT = 4 RT query tiles that share every K/V read.  Keys of unit u: key tiles [kt0, qt_last] (full:
// kt0 = 0; sliding: kt0 = first query tile - WT); key tile kt = cache page pt[kt / 2], half kt % 2.
//
// Transposed formulation (every matmul product exact at HiFi2: in0 -> srcB keeps 7 significant bits = a BFP8
// mantissa, in1 -> srcA keeps 10 >= the 8 of BF16):
//   S^T(kt, j) = K(kt) Q_j^T          in0 = K tile row (BFP8), in1 = Q^T tiles (BF16, transposed once per segment)
//   P^T = 2^(S^T * scale * log2 e - c log2 e)      FP32 SFPU exp on DEST (no running max: |q.k| <= qk_bound (host
//                                                   check) keeps every P in FP32 range for the fixed offset c)
//   O^T(:, j) += V^T(kt) P^T(kt, j)   in0 = V^T (BF16 copy of the BFP8 V, transposed once per chunk), in1 = P^T
//   l(j)      += 1 P^T(kt, j)         in0 = ones tile
// O^T and l accumulate in FP32 (DEST within a chunk, packer L1 accumulation across chunks); the epilogue forms
// out = transpose(O^T * sigmoid(gate) / l).  The exp (and every other SFPU op) runs on the PACK thread alongside the
// math thread's matmuls; SFPU and matmul instructions contend (each SFPU instruction slows the matmuls, measured with
// the SP_PROF timers), so the unmasked exp is an SFPLOADMACRO pipeline of 5 issued instructions per row (sp_exp_lm).
//
// Schedule (start-dependent; every RISC derives it from the start tensor): the units' key tiles, in unit order, are
// cut into NCE contiguous equal shares (NCE = min(NC, W / MINSEG)); core c processes the segments of its share.  A
// unit split over several cores: its first core (whose share ends with it) finalizes it, the later cores (whose
// shares start with it) write their partial [O^T | l] to a DRAM scratch slot (one per core) and signal the
// finalizer's semaphore; the finalizer adds the partials (packer L1 accumulation) after its own segments.
//
// Common args (reader / writer): q, k cache, v cache, page table, start, consts, out, gate, scratch, then the
// (noc x, noc y) of core 0 .. NC - 1.  Per-core args (all RISCs): core index c.
#include <cstdint>
#ifdef SP_DEBUG
#include "api/debug/dprint.h"
#define SPDBG(...) DPRINT(__VA_ARGS__)
#else
#define SPDBG(...)
#endif
#ifdef SP_PROF
#include "tools/profiler/kernel_profiler.hpp"
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#endif

namespace sp {
constexpr uint32_t TPT = SP_TPT;        // Tp / 32
constexpr uint32_t RT = SP_RT;          // query row tiles per head per unit
constexpr uint32_t QT = 4 * RT;         // query tiles per unit (4 heads)
constexpr uint32_t KC = SP_KC;          // key tiles per chunk (L1 residency)
constexpr uint32_t WT = SP_WT;          // sliding window / 32 (0: full causal)
constexpr uint32_t NC = SP_NC;          // cores
constexpr uint32_t MINSEG = SP_MINSEG;  // min key tiles per core share
constexpr uint32_t NB = TPT / RT;
constexpr uint32_t NU = 4 * NB;  // units
constexpr bool HAS_GATE = SP_GATE != 0;
constexpr uint32_t GW = SP_GATE_W;    // gate tensor width (tiles)
constexpr uint32_t GC = SP_GATE_COL;  // gate logits tile column (head h at column h of that tile)
constexpr uint32_t PTW = SP_PT_W;     // page-table row entries
constexpr uint32_t kBf16 = 2048, kFp32 = 4096, kBfp8 = 1088;
constexpr uint32_t kPart = 5 * QT;  // scratch slot tiles: O^T (4 QT, tile j * 4 + d) | l (QT)
constexpr uint32_t kOutW = 64;      // output width in tiles (2048)
// CBs
constexpr uint32_t cQ = 0, cK = 1, cV = 2, cVT = 3, cP = 4, cOnes = 5, cZero = 6, cMask = 7, cE = 8, cGr = 9,
                   cPar = 10, cOs = 11, cOut = 12, cScr = 13, cScrW = 14, cSend = 20, cAck = 21, cQT = 22;
constexpr uint32_t cO = 16, cL = 17, cS = 18, cPart = 19;  // FP32, unpacked straight to DEST
// consts tensor tiles: zero, ones, diag mask, window-edge mask, E_0 .. E_15 (E_c: column c all ones)
constexpr uint32_t kZeroT = 0, kOnesT = 1, kDiagT = 2, kEdgeT = 3, kE0T = 4;
// plan (params page words): [0] segments, [1] start / 32, then 4 words per segment:
//   unit | flags << 16, lo, hi (key tiles relative to the unit's first), peer (sender: finalizer core;
//   finalizer: number of contributing cores)
constexpr uint32_t kHdr = 2, kSegW = 4;
constexpr uint32_t kFirst = 1, kLast = 2;
// mask classes of a (key tile, query row tile) round
constexpr uint32_t kNone = 0, kDiag = 1, kEdge = 2;

// unit u: first key tile kt0 and key tiles n
FORCE_INLINE void unit_range(uint32_t st, uint32_t u, uint32_t& kt0, uint32_t& n) {
    const uint32_t qf = st + (u >> 2) * RT;
    kt0 = (WT != 0 && qf > WT) ? qf - WT : 0;
    n = qf + RT - kt0;
}
FORCE_INLINE uint32_t mask_class(uint32_t kt, uint32_t qt) {
    return kt == qt ? kDiag : ((WT != 0 && kt + WT == qt) ? kEdge : kNone);
}
// key tiles [kt_lo, kt_hi) of query row tile qt (absolute), clipped to [a, b)
FORCE_INLINE void row_keys(uint32_t qt, uint32_t a, uint32_t b, uint32_t& lo, uint32_t& hi) {
    lo = (WT != 0 && qt > WT) ? qt - WT : 0;
    hi = qt + 1;
    lo = lo > a ? lo : a;
    hi = hi < b ? hi : b;
}
}  // namespace sp

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace sp {
// This core's segments (identical arithmetic on every RISC); returns the segment count.
FORCE_INLINE uint32_t make_plan(uint32_t st, uint32_t c, volatile tt_l1_ptr uint32_t* plan) {
    uint32_t W = 0;
    for (uint32_t u = 0; u < NU; ++u) {
        uint32_t kt0, n;
        unit_range(st, u, kt0, n);
        W += n;
    }
    uint32_t nce = W / MINSEG;
    nce = nce < 1 ? 1 : (nce > NC ? NC : nce);
    plan[1] = st;
    plan[0] = 0;
    if (c >= nce) {
        return 0;
    }
    const uint32_t lo = c * W / nce, hi = (c + 1) * W / nce;
    auto core_of = [&](uint32_t x) {
        uint32_t k = x * nce / W;
        while (k + 1 < nce && (k + 1) * W / nce <= x) {
            ++k;
        }
        while (k > 0 && k * W / nce > x) {
            --k;
        }
        return k;
    };
    uint32_t ns = 0, base = 0;
    for (uint32_t u = 0; u < NU && base < hi; ++u) {
        uint32_t kt0, n;
        unit_range(st, u, kt0, n);
        if (base + n > lo) {
            const uint32_t s_lo = (lo > base ? lo : base) - base;
            const uint32_t s_hi = (hi < base + n ? hi : base + n) - base;
            const uint32_t fl = (s_lo == 0 ? kFirst : 0u) | (s_hi == n ? kLast : 0u);
            uint32_t peer = 0;
            if (!(fl & kFirst)) {
                peer = core_of(base);
            } else if (!(fl & kLast)) {
                peer = core_of(base + n - 1) - c;
            }
            volatile tt_l1_ptr uint32_t* s = plan + kHdr + kSegW * ns;
            s[0] = u | (fl << 16);
            s[1] = s_lo;
            s[2] = s_hi;
            s[3] = peer;
            ++ns;
        }
        base += n;
    }
    plan[0] = ns;
    return ns;
}

FORCE_INLINE uint32_t read_start(uint32_t l1) {
    const auto start = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4),
                                      64);
    noc_async_read(get_noc_addr(0, start, 0), l1, 64);
    noc_async_read_barrier();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
}

FORCE_INLINE void reader(uint32_t c) {
    const auto q = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0),
                                  kBf16);
    const auto kc = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1),
                                   kBfp8);
    const auto vc = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2),
                                   kBfp8);
    const auto pt = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3),
                                   PTW * 4);
    const auto consts = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                       get_common_arg_val<uint32_t>(5), kBf16);
    const auto gate = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(7),
                                     kBf16);
    const auto scratch = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(8), kFp32);
    const uint32_t scr = get_write_ptr(cScr);  // [0, 64) start; [64, ...) page-table slice
    const uint32_t st = read_start(scr) / 32;
    cb_reserve_back(cPar, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cPar));
    const uint32_t ns = make_plan(st, c, plan);
    cb_push_back(cPar, 1);
    SPDBG("R c={} st={} ns={}\n", c, st, ns);
    if (ns == 0) {
        return;
    }
    // constant tiles (resident for the whole program)
    cb_reserve_back(cZero, 1);
    cb_reserve_back(cOnes, 1);
    cb_reserve_back(cMask, 2);
    cb_reserve_back(cE, 16);
    noc_async_read(get_noc_addr(kZeroT, consts), get_write_ptr(cZero), kBf16);
    noc_async_read(get_noc_addr(kOnesT, consts), get_write_ptr(cOnes), kBf16);
    noc_async_read(get_noc_addr(kDiagT, consts), get_write_ptr(cMask), kBf16);
    noc_async_read(get_noc_addr(kEdgeT, consts), get_write_ptr(cMask) + kBf16, kBf16);
    for (uint32_t i = 0; i < 16; ++i) {
        noc_async_read(get_noc_addr(kE0T + i, consts), get_write_ptr(cE) + i * kBf16, kBf16);
    }
    noc_async_read_barrier();
    cb_push_back(cZero, 1);
    cb_push_back(cOnes, 1);
    cb_push_back(cMask, 2);
    cb_push_back(cE, 16);
    uint32_t fin_m = 0;
    for (uint32_t i = 0; i < ns; ++i) {
        volatile tt_l1_ptr uint32_t* s = plan + kHdr + kSegW * i;
        const uint32_t u = s[0] & 0xFFFF, fl = s[0] >> 16, lo = s[1], hi = s[2];
        const uint32_t b = u >> 2, g = u & 3;
        uint32_t kt0, n;
        unit_range(st, u, kt0, n);
        if ((fl & kFirst) && !(fl & kLast)) {
            fin_m = s[3];
        }
        // Q tiles of the unit, one row tile r at a time (16 tiles: slot d * 4 + hh <- head 4 g + hh, row tile
        // b RT + r, dim tile d); compute transposes them into cQT
        for (uint32_t r = 0; r < RT; ++r) {
            cb_reserve_back(cQ, 16);
            const uint32_t qdst = get_write_ptr(cQ);
            for (uint32_t d = 0; d < 4; ++d) {
                for (uint32_t hh = 0; hh < 4; ++hh) {
                    const uint32_t tile = ((4 * g + hh) * TPT + b * RT + r) * 4 + d;
                    noc_async_read(get_noc_addr(tile, q), qdst + (d * 4 + hh) * kBf16, kBf16);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cQ, 16);
        }
        const bool epilogue = HAS_GATE && (fl & kFirst);
        if (epilogue) {
            cb_reserve_back(cGr, RT);
            for (uint32_t r = 0; r < RT; ++r) {
                noc_async_read(get_noc_addr((b * RT + r) * GW + GC, gate), get_write_ptr(cGr) + r * kBf16, kBf16);
            }
        }
        // page-table entries of key tiles [kt0 + lo, kt0 + hi)
        const uint32_t e_first = (kt0 + lo) >> 1, e_last = (kt0 + hi - 1) >> 1;
        const uint32_t off = (e_first * 4) & ~63u;
        const uint32_t bytes = ((e_last * 4 + 4 + 63) & ~63u) - off;
        noc_async_read(get_noc_addr(0, pt, off), scr + 64, bytes);
        noc_async_read_barrier();
        if (epilogue) {
            cb_push_back(cGr, RT);
        }
        volatile tt_l1_ptr uint32_t* pte = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 64);
        const uint32_t e_base = off / 4;
        for (uint32_t cs = kt0 + lo; cs < kt0 + hi; cs += KC) {
            SPDBG("R chunk {} u={}\n", cs, u);
            const uint32_t nk = (kt0 + hi - cs) < KC ? (kt0 + hi - cs) : KC;
            cb_reserve_back(cK, KC * 4);
            cb_reserve_back(cV, KC * 4);
            const uint32_t kdst = get_write_ptr(cK), vdst = get_write_ptr(cV);
            for (uint32_t k = 0; k < nk; ++k) {
                const uint32_t kt = cs + k;
                const uint32_t page = pte[(kt >> 1) - e_base];
                const uint32_t tid = ((page * 4 + g) * 2 + (kt & 1)) * 4;
                for (uint32_t d = 0; d < 4; ++d) {
                    noc_async_read(get_noc_addr(tid + d, kc), kdst + (k * 4 + d) * kBfp8, kBfp8);
                    noc_async_read(get_noc_addr(tid + d, vc), vdst + (k * 4 + d) * kBfp8, kBfp8);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cK, KC * 4);
            cb_push_back(cV, KC * 4);
        }
    }
    // finalizer of a split unit: the partials of the m following cores
    if (fin_m) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait(sem, fin_m);
        SPDBG("R fin m={} got\n", fin_m);
        noc_semaphore_set(sem, 0);
        for (uint32_t cc = c + 1; cc <= c + fin_m; ++cc) {
            for (uint32_t t = 0; t < kPart; t += 4) {
                cb_reserve_back(cPart, 4);
                const uint32_t dst = get_write_ptr(cPart);
                for (uint32_t i = 0; i < 4; ++i) {
                    noc_async_read(get_noc_addr(cc * kPart + t + i, scratch), dst + i * kFp32, kFp32);
                }
                noc_async_read_barrier();
                cb_push_back(cPart, 4);
            }
        }
    }
}

FORCE_INLINE void writer(uint32_t c) {
    const auto out = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6),
                                    kBf16);
    const auto scratch = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(8), kFp32);
    const uint32_t scr = get_write_ptr(cScrW);
    const uint32_t st = read_start(scr) / 32;
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 64);
    const uint32_t ns = make_plan(st, c, plan);
    for (uint32_t i = 0; i < ns; ++i) {
        SPDBG("W seg {}\n", i);
        volatile tt_l1_ptr uint32_t* s = plan + kHdr + kSegW * i;
        const uint32_t u = s[0] & 0xFFFF, fl = s[0] >> 16;
        const uint32_t b = u >> 2, g = u & 3;
        if (!(fl & kFirst)) {
            // partial [O^T | l] of a split unit -> scratch slot c, then the finalizer's semaphore.  cO / cL have one
            // consumer (compute's unpack thread: a CB's ack count is per consumer); the writer reads them in place
            // (always at the CB base: full-capacity reservations) between cSend (compute: packed) and cAck (writer:
            // copied, compute may pop and reuse them).
            const uint32_t f = s[3];
            cb_wait_front(cSend, 1);
            const uint32_t o = get_local_cb_interface(cO).fifo_limit - get_local_cb_interface(cO).fifo_size;
            const uint32_t l = get_local_cb_interface(cL).fifo_limit - get_local_cb_interface(cL).fifo_size;
            for (uint32_t t = 0; t < QT * 4; ++t) {
                noc_async_write(o + t * kFp32, get_noc_addr(c * kPart + t, scratch), kFp32);
            }
            for (uint32_t t = 0; t < QT; ++t) {
                noc_async_write(l + t * kFp32, get_noc_addr(c * kPart + QT * 4 + t, scratch), kFp32);
            }
            noc_async_write_barrier();
            const uint32_t fx = get_common_arg_val<uint32_t>(9 + 2 * f), fy = get_common_arg_val<uint32_t>(10 + 2 * f);
            noc_semaphore_inc(get_noc_addr(fx, fy, get_semaphore(0)), 1);
            noc_async_atomic_barrier();
            cb_pop_front(cSend, 1);
            cb_reserve_back(cAck, 1);
            cb_push_back(cAck, 1);
        } else {
            // this core's output columns: query tile j = (r, hh) -> rows of tile b RT + r, columns of head 4 g + hh
            for (uint32_t r = 0; r < RT; ++r) {
                for (uint32_t hh = 0; hh < 4; ++hh) {
                    cb_wait_front(cOut, 4);
                    const uint32_t src = get_read_ptr(cOut);
                    for (uint32_t d = 0; d < 4; ++d) {
                        const uint32_t tile = (b * RT + r) * kOutW + (4 * g + hh) * 4 + d;
                        noc_async_write(src + d * kBf16, get_noc_addr(tile, out), kBf16);
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cOut, 4);
                    SPDBG("W out r={} hh={}\n", r, hh);
                }
            }
        }
    }
}
}  // namespace sp
#endif

#if defined(COMPILE_FOR_TRISC) && (defined(TRISC_MATH) || defined(TRISC_PACK))
namespace ckernel::sfpu {
// Masked key tiles (diagonal / window edge: -1e30 logits -> 0): P = 2^z, z = S * EXP_A + EXP_B - 127 (EXP_A = scale
// log2 e, EXP_B = 127 - c log2 e) on one face (8 rows of 32), 16 instructions per row: x = z + 127; the exp_21f shift
// splits x into floor(x) (exponent field of x * 2^23 as an integer) and the 23-bit fraction F; 2^frac = 1 + F (k1 + F
// (k2 + F k3)) in [1, 2) (cubic through 1 and 2 - 2^-20, rel. error 1.0e-4; k scaled to the integer fraction); result
// = setexp(2^frac, floor(x)), with the integer part zeroed where x <= 0.  Constants: LREG12 = EXP_A, LREG13 = k3,
// LREG14 = k2 (sp_exp_masked_init), LREG5 = EXP_B, LREG6 = k1; ADDR_MOD_6 advances DEST by one SFPU row on the store.
// Clobbers the unmasked exp's constants (the caller reloads them: sp_exp_init).
inline void sp_exp_masked_init() {
    sfpi::vConstFloatPrgm0 = EXP_A;
    sfpi::vConstFloatPrgm1 = EXP_K3;
    sfpi::vConstFloatPrgm2 = EXP_K2;
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}
inline void sp_exp_masked() {
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, hi16(EXP_B));
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, lo16(EXP_B));
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, hi16(EXP_K1));
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, lo16(EXP_K1));
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG5, p_sfpu::LREG3, 0);   // x
        TTI_SFPNOP;
        TTI_SFPEXEXP(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);                             // e = exponent(x)
        TTI_SFPEXMAN(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);                             // 1.mantissa(x)
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);                              // x * 2^23
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);      // F bits
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);                                 // F
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG14, p_sfpu::LREG2, 0);  // k3 F + k2
        TTI_SFPGT(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 8);                             // mask x > 0
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG2, 0);    // (..) F + k1
        TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG0, 1);                   // floor(x) & mask
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0); // 2^frac
        TTI_SFPNOP;  // MAD latency (removing either NOP changed results: no interlock)
        TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, 2);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
}

// Unmasked key tiles (all but the diagonal / window-edge ones): every logit is within the bound, so
//   t  = S * EXP_A + EXP_M           (EXP_M = 2175 - c log2 e: t = z + 2175 in [2048, 4096), so t's mantissa holds
//                                     z + 127 in fixed point with 12 fraction bits; z = log2 P in [-104, 87])
//   y0 = t << 11                     (bit pattern: sign 0, exponent floor(z) + 127, mantissa the fraction f)
//   v  = setexp(y0, 127) = 1 + f
//   p  = ((Q3 v + Q2) v + Q1) v + Q0 ~ 2^f in [1, 2) (cubic through (1, 1), minimax rel. error 8.6e-5)
//   P  = setexp(p, exponent of y0) = 2^floor(z) p          (end to end rel. error <= 2.3e-4, std 7.7e-5)
// evaluated with SFPLOADMACRO (sp_exp_lm): each LOADMACRO loads a DEST row and schedules up to one simple / MAD /
// round / store instruction for later cycles, so t, the shift, v, the y0 round trip through DEST, P and both stores
// cost no issue slots: 5 issued instructions per row instead of 9 (16 for the masked variant).  The issue slots are
// the cost that matters: the SFPU instructions (pack thread) and the matmuls (math thread) contend, so every SFPU
// instruction removed also speeds up the matmuls.  Constants: LREG6 = EXP_A, LREG7 = EXP_M, LREG8 (fixed 0.8373,
// 0x3F56594B) is the shift amount 11, LREG11 = Q0, LREG12 = Q3, LREG13 = Q2, LREG14 = Q1.
FORCE_INLINE void sp_cfg(uint32_t dest, uint32_t w) {  // SFPCONFIG dest <- 32-bit word w (via LREG0)
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, w & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, w >> 16);
    TT_SFPCONFIG(0, dest, 0);
    TTI_SFPNOP;
}
inline void sp_exp_init() {
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, hi16(EXP_A));
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, lo16(EXP_A));
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, hi16(EXP_M));
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, lo16(EXP_M));
    // instruction templates (VD and one source are overridden by the macro's VD):
    //   0: t = A * s + M (MAD; s = VB)    1: y0 = t << LREG8 (SHFT2; LREG8 = 0.8373 = 0x3F56594B: shift 11)
    //   2: v = setexp(y0, 127)            3: P = {LREG5 sign/mantissa, VD exponent} (SETEXP, exponent from VB = VD)
    sp_cfg(0, TT_OP_SFPMAD(p_sfpu::LREG6, 0, p_sfpu::LREG7, 0, 0));
    sp_cfg(1, TT_OP_SFPSHFT2(0, 8, 0, 5));
    sp_cfg(2, TT_OP_SFPSETEXP(127, 0, 0, 1));
    sp_cfg(3, TT_OP_SFPSETEXP(0, p_sfpu::LREG5, 0, 2));
    // sequences (bytes: simple | MAD | round | store; bit 7 VB = VD, bits 5:3 delay, bits 2:0 template 4 + i / 3 store)
    //   0 (LMb, loads S): MAD t @1, SHFT2 y0 @3, store y0 @4, SETEXP v @5
    //   1 (LMc, loads y0): SETEXP P @2, store P @3
    sp_cfg(4, 0x239D8C2Eu);
    sp_cfg(5, 0x1B000097u);
    TTI_SFPCONFIG(0xF00, 8, 1);  // store format: DEST default; delays count issued SFPU instructions
    TTI_SFPNOP;
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, hi16(EXP_Q0));
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, lo16(EXP_Q0));
    TTI_SFPCONFIG(0, 11, 0);
    TTI_SFPNOP;
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, hi16(EXP_Q3));
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, lo16(EXP_Q3));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPNOP;
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, hi16(EXP_Q2));
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, lo16(EXP_Q2));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPNOP;
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, hi16(EXP_Q1));
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, lo16(EXP_Q1));
    TTI_SFPCONFIG(0, 14, 0);
    TTI_SFPNOP;
}
// LOADMACRO exp of the 128 SFPU rows (4 tiles) of the current DEST half: 5 issued instructions per row.  Period j
// (5 slots): MAD2(row j - 2), LMb(row j), MAD3(row j - 2), LMc(row j - 2), MAD4(row j - 2); scheduled units per slot
// (mod 5): MAD 3 (t), round 0 (SHFT2), simple 2 (v) / 1 (P), store 1 (y0) / 2 (P): no collision with each other or
// with the issued MADs (slots 0, 2, 4).  Row j's S / t / y0 / v live in LREG (j % 3) (LMb VD), y0 goes through DEST
// (LMb stores it, LMc reloads it into LREG3), Horner in LREG4, p in LREG5.  The DEST RWC follows row j - 2 (LMc
// advances it by one row); LMb reads 2 rows ahead.
#define SP_LMB(vd, imm) TTI_SFPLOADMACRO((0 << 2) | (vd), 0, ADDR_MOD_7, (imm))
#define SP_LMC() TTI_SFPLOADMACRO((1 << 2) | 3, 0, ADDR_MOD_6, 0)
#define SP_MAD2(r) TTI_SFPMAD((r), p_sfpu::LREG12, p_sfpu::LREG13, p_sfpu::LREG4, 0)
#define SP_MAD3(r) TTI_SFPMAD(p_sfpu::LREG4, (r), p_sfpu::LREG14, p_sfpu::LREG4, 0)
#define SP_MAD4(r) TTI_SFPMAD(p_sfpu::LREG4, (r), p_sfpu::LREG11, p_sfpu::LREG5, 0)
#define SP_PERIOD(r, rm2) \
    SP_MAD2(rm2);         \
    SP_LMB(r, 4);         \
    SP_MAD3(rm2);         \
    SP_LMC();             \
    SP_MAD4(rm2)
inline void sp_exp_lm() {
    _llk_math_eltwise_sfpu_start_(0);
    TTI_SFPNOP;
    SP_LMB(0, 0);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    SP_LMB(1, 2);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#pragma GCC unroll 0
    for (int i = 0; i < 42; ++i) {  // periods 2 .. 127
        SP_PERIOD(2, 0);
        SP_PERIOD(0, 1);
        SP_PERIOD(1, 2);
    }
    SP_MAD2(0);  // periods 128, 129: rows 126, 127
    TTI_SFPNOP;
    SP_MAD3(0);
    SP_LMC();
    SP_MAD4(0);
    SP_MAD2(1);
    TTI_SFPNOP;
    SP_MAD3(1);
    SP_LMC();
    SP_MAD4(1);
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        TTI_SFPNOP;
    }
    _llk_math_eltwise_sfpu_done_();
}
inline void sp_neg1_restore() {
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0xBF80);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPCONFIG(0, 11, 0);
    TTI_SFPNOP;
}
// kernel end: LREG11 = -1 and the default SFPU config (LOADMACRO control), as later programs on this core expect
inline void sp_sfpu_restore() {
    sp_neg1_restore();
    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPNOP;
}
// epilogue, DEST tiles (relative): 0 l, 1 gate logits g (rows identical), 2, 3 O^T tiles: s = sigmoid(g) / l =
// 1 / ((1 + exp(-g)) l) (FP32 exp, 2 Newton steps), tiles 2, 3 *= s, tile 0 = s
inline void sp_epi_first() {
#pragma GCC unroll 2
    for (int i = 0; i < 8; ++i) {
        sfpi::vFloat l = sfpi::dst_reg[0];
#if SP_GATE
        sfpi::vFloat g = sfpi::dst_reg[32];
        sfpi::vFloat e = _sfpu_exp_accurate_<true>(-g);
        sfpi::vFloat s = sfpu_reciprocal_iter<2>((e + 1.0f) * l);
#else
        sfpi::vFloat s = sfpu_reciprocal_iter<2>(l);
#endif
        sfpi::dst_reg[64] = sfpi::dst_reg[64] * s;
        sfpi::dst_reg[96] = sfpi::dst_reg[96] * s;
        sfpi::dst_reg[0] = s;
        sfpi::dst_reg++;
    }
}
// DEST tile 0 = s; tiles 1, 2 *= s
inline void sp_epi_second() {
#pragma GCC unroll 4
    for (int i = 0; i < 8; ++i) {
        sfpi::vFloat s = sfpi::dst_reg[0];
        sfpi::dst_reg[32] = sfpi::dst_reg[32] * s;
        sfpi::dst_reg[64] = sfpi::dst_reg[64] * s;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

#if defined(COMPILE_FOR_TRISC)
namespace sp {
#define SP_SFPU(fn, idst) \
    PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, fn, idst, VectorMode::RC)))

// SP_PROF: per-thread phase timers (profiling only).  SPT(ph) waits for this thread's issued Tensix instructions to
// complete (tensix_sync), then charges the wall-clock time since the previous mark to the previous phase; compute()
// emits the totals (cycles) as timestamped data SPT0 .. SPT11 per TRISC.  Phases: 0 setup / Q^T / zero fill, 1 round
// init (reconfig + DEST acquire), 2 S^T matmuls + mask, 3 pack thread waiting for math, 4 exp, 5 P packs + release,
// 6 B pieces (O^T += V^T P^T), 7 C pieces (l), 8 V^T pieces, 9 partials, 10 K / V chunk wait, 11 epilogue.
#ifdef SP_PROF
uint32_t prof_acc[12];
uint32_t prof_last, prof_cur;
FORCE_INLINE void prof_mark(uint32_t ph) {
    ckernel::tensix_sync();
    const uint32_t t = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    prof_acc[prof_cur] += t - prof_last;
    prof_last = t;
    prof_cur = ph;
}
FORCE_INLINE void prof_emit() {
    prof_mark(0);
    DeviceTimestampedData("SPT0", prof_acc[0]);
    DeviceTimestampedData("SPT1", prof_acc[1]);
    DeviceTimestampedData("SPT2", prof_acc[2]);
    DeviceTimestampedData("SPT3", prof_acc[3]);
    DeviceTimestampedData("SPT4", prof_acc[4]);
    DeviceTimestampedData("SPT5", prof_acc[5]);
    DeviceTimestampedData("SPT6", prof_acc[6]);
    DeviceTimestampedData("SPT7", prof_acc[7]);
    DeviceTimestampedData("SPT8", prof_acc[8]);
    DeviceTimestampedData("SPT9", prof_acc[9]);
    DeviceTimestampedData("SPT10", prof_acc[10]);
    DeviceTimestampedData("SPT11", prof_acc[11]);
}
#define SPT(ph) prof_mark(ph)
#else
#define SPT(ph)
#endif

// the SFPU runs on the PACK thread only (the math thread never issues SFPU work): its common init once at the start
// (sfpu_init_once), then only the programmable constants per use
FORCE_INLINE void sfpu_init_once() { PACK((llk_math_eltwise_unary_sfpu_init<SfpuType::unused>())); }
FORCE_INLINE void sfpu_init_exp() { PACK((ckernel::sfpu::sp_exp_init())); }
// the epilogue's sfpi code expects LREG11 = -1 (vConstNeg1; the exp uses it as Q0)
FORCE_INLINE void sfpu_init_epi() {
    PACK((ckernel::sfpu::sfpu_reciprocal_init<false>()));
    PACK((ckernel::sfpu::sp_neg1_restore()));
}
FORCE_INLINE void init_phase_a() {
    reconfig_data_format(cQT, cK);
    matmul_block_init(cK, cQT, 0, 4, 1, 4);
}

// Segment schedule.  The SFPU exp (pack thread) runs alongside the math thread's matmuls, so the math thread's other
// work is interleaved between the exp rounds: a column group t = (chunk, query row tile r) is
//   A(t):  rounds kt: S^T(kt, 4 heads) = K(kt) Q^T (math) -> P^T = exp (pack thread SFPU) -> cP;
// and the "pieces" B / C of the previous group (O^T(:, hh) += V^T P^T(:, hh) for hh = 0..3, l += 1 P^T) and V^T of
// the current chunk (one key tile per piece) run on the math thread after each A round, while the pack thread
// exponentiates.  Order constraints: B / C of a group after its chunk's V^T and its own A; cP double-buffered (A(t)
// writes while B(t - 1) reads), cVT single (V^T of chunk c + 1 starts after the last B / C piece of chunk c).
struct Group {
    uint32_t cs, r, klo, khi;
    bool last;  // last group of its chunk (its C piece pops the chunk's V^T)
};

struct Pipe {
    bool written[RT];  // O^T / l of row tile r hold this segment's data (else packed without L1 accumulation)
    Group bc;          // group whose B / C pieces are pending
    uint32_t bc_next;  // 0..3: B hh, 4: C, 5: none pending
    uint32_t vt_nk, vt_next;  // V^T pieces of the current chunk: key tiles vt_next .. vt_nk - 1 pending
    // a piece / mask since the last S^T round changed the unpack + math configuration (mm) or the packer's output
    // format / L1 accumulation (pk): the next round re-initializes only what changed
    bool mm_dirty, pk_dirty;
};

FORCE_INLINE void piece_vt(Pipe& p) {
    const uint32_t k = p.vt_next;
    p.mm_dirty = true;
    if (k == 0) {
        cb_reserve_back(cVT, KC * 4);
    }
    reconfig_data_format_srca(cV);
    transpose_init(cV);
    pack_reconfig_data_format(cVT);
    PACK((llk_pack_reconfig_l1_acc(0)));
    tile_regs_acquire();
    for (uint32_t d = 0; d < 4; ++d) {
        transpose_tile(cV, k * 4 + d, d);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t d = 0; d < 4; ++d) {
        pack_tile<true>(d, cVT, d * KC + k);
    }
    tile_regs_release();
    if (++p.vt_next == p.vt_nk) {
        cb_push_back(cVT, KC * 4);
        cb_pop_front(cV, KC * 4);
    }
}

FORCE_INLINE void piece_bc(Pipe& p) {
    const Group& t = p.bc;
    p.mm_dirty = p.pk_dirty = true;
    const uint32_t r = t.r;
    if (p.bc_next == 0) {
        cb_wait_front(cVT, KC * 4);
        cb_wait_front(cP, KC * 4);
    }
    // O^T / l accumulate across chunks in L1 (packer accumulation; measured faster than reloading them into DEST)
    PACK((llk_pack_reconfig_l1_acc(p.written[r] ? 1 : 0)));
    if (p.bc_next < 4) {
        // O^T(d, j) += sum_k V^T(d, k) P^T(k, hh) for the 4 heads hh (j = r * 4 + hh) of one dim tile d: srcB holds
        // V^T(d, k), the 4 P^T tiles stream through srcA (the efficient ct >= rt reuse)
        SPT(6);
        const uint32_t d = p.bc_next;
        pack_reconfig_data_format(cO);
        tile_regs_acquire();
        reconfig_data_format(cP, cVT);
        matmul_block_init(cVT, cP, 0, 4, 1, 1);
        for (uint32_t kt = t.klo; kt < t.khi; ++kt) {
            const uint32_t k = kt - t.cs;
            matmul_block(cVT, cP, d * KC + k, k * 4, 0, 0, 4, 1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t hh = 0; hh < 4; ++hh) {
            pack_tile<true>(hh, cO, (r * 4 + hh) * 4 + d);
        }
        tile_regs_release();
        ++p.bc_next;
        return;
    }
    // l(j) += sum_k 1 P^T(k, hh)
    SPT(7);
    pack_reconfig_data_format(cL);
    tile_regs_acquire();
    reconfig_data_format(cP, cOnes);
    matmul_block_init(cOnes, cP, 0, 4, 1, 1);
    for (uint32_t kt = t.klo; kt < t.khi; ++kt) {
        matmul_block(cOnes, cP, 0, (kt - t.cs) * 4, 0, 0, 4, 1, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t hh = 0; hh < 4; ++hh) {
        pack_tile<true>(hh, cL, r * 4 + hh);
    }
    tile_regs_release();
    p.written[r] = true;
    cb_pop_front(cP, KC * 4);
    if (t.last) {
        cb_pop_front(cVT, KC * 4);
    }
    p.bc_next = 5;
}

// one pending math piece (B / C of the previous group first, then V^T); false if none
FORCE_INLINE bool piece(Pipe& p) {
    if (p.bc_next < 5) {
        piece_bc(p);
        return true;
    }
    if (p.vt_next < p.vt_nk) {
        SPT(8);
        piece_vt(p);
        return true;
    }
    return false;
}

// one segment's keys [a0, a1) of unit row block b, accumulated into the reserved cO / cL
FORCE_INLINE void segment_body(uint32_t st, uint32_t b, uint32_t a0, uint32_t a1) {
    Pipe p;
    for (uint32_t r = 0; r < RT; ++r) {
        p.written[r] = false;
    }
    p.bc_next = 5;
    p.vt_nk = p.vt_next = 0;
    p.mm_dirty = p.pk_dirty = true;
    sfpu_init_exp();
    // P^T / V^T are packed as BF16 (half the packer bytes of FP32: the pack thread, which also runs the exp, is the
    // bottleneck), O^T / l as FP32 with L1 accumulation after their first chunk
    pack_reconfig_data_format(cP);
    PACK((llk_pack_reconfig_l1_acc(0)));
    for (uint32_t cs = a0; cs < a1; cs += KC) {
        const uint32_t nk = (a1 - cs) < KC ? (a1 - cs) : KC;
        SPT(10);
        cb_wait_front(cK, KC * 4);
        cb_wait_front(cV, KC * 4);
        SPT(0);
        // V^T of this chunk becomes pending (behind the previous group's B / C)
        p.vt_nk = nk;
        p.vt_next = 0;
        uint32_t r_last = 0;
        for (uint32_t r = 0; r < RT; ++r) {
            uint32_t klo, khi;
            row_keys(st + b * RT + r, cs, cs + nk, klo, khi);
            if (klo < khi) {
                r_last = r;
            }
        }
        for (uint32_t r = 0; r < RT; ++r) {
            const uint32_t qt = st + b * RT + r;
            uint32_t klo, khi;
            row_keys(qt, cs, cs + nk, klo, khi);
            if (klo >= khi) {
                continue;
            }
            {
                SPT(0);
                cb_reserve_back(cP, KC * 4);
                for (uint32_t kt = klo; kt < khi; ++kt) {
                    const uint32_t k = kt - cs;
                    const uint32_t cls = mask_class(kt, qt);
                    SPT(1);
                    if (p.mm_dirty) {
                        init_phase_a();
                        p.mm_dirty = false;
                    }
                    if (p.pk_dirty) {
                        pack_reconfig_data_format(cP);
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        p.pk_dirty = false;
                    }
                    tile_regs_acquire();
                    SPT(2);
                    for (uint32_t d = 0; d < 4; ++d) {
                        matmul_block(cK, cQT, k * 4 + d, (r * 4 + d) * 4, 0, 0, 4, 1, 4);
                    }
                    if (cls != kNone) {
                        reconfig_data_format(cZero, cMask);
                        p.mm_dirty = true;
                        add_tiles_init(cZero, cMask, true);
                        for (uint32_t i = 0; i < 4; ++i) {
                            add_tiles(cZero, cMask, 0, cls - 1, i);
                        }
                    }
                    tile_regs_commit();
                    SPT(3);
                    tile_regs_wait();
                    SPT(4);
                    if (cls != kNone) {
                        PACK((ckernel::sfpu::sp_exp_masked_init()));
                        for (uint32_t i = 0; i < 4; ++i) {
                            SP_SFPU(sp_exp_masked, i);
                        }
                        sfpu_init_exp();
                    } else {
                        PACK((ckernel::sfpu::sp_exp_lm()));
                    }
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    SPT(5);
                    for (uint32_t i = 0; i < 4; ++i) {
                        pack_tile<true>(i, cP, k * 4 + i);
                    }
                    tile_regs_release();
                    SPDBG("C kt {} cls {}\n", kt, cls);
                    piece(p);  // math work for the other DEST half while the pack thread exponentiates
                }
                cb_push_back(cP, KC * 4);
            }
            // this group's B / C go pending once the previous group's are done and this chunk's V^T is complete
            while (p.bc_next < 5 || p.vt_next < p.vt_nk) {
                piece(p);
            }
            p.bc = Group{cs, r, klo, khi, r == r_last};
            p.bc_next = 0;
        }
        cb_pop_front(cK, KC * 4);
    }
    while (piece(p)) {
    }
    SPT(0);
    // row tiles without keys in this segment contribute zero
    bool any = false;
    for (uint32_t r = 0; r < RT; ++r) {
        any |= !p.written[r];
    }
    if (any) {
        reconfig_data_format_srca(cZero);
        copy_tile_to_dst_init_short(cZero);
        pack_reconfig_data_format(cO);
        PACK((llk_pack_reconfig_l1_acc(0)));
        for (uint32_t r = 0; r < RT; ++r) {
            if (p.written[r]) {
                continue;
            }
            for (uint32_t t = 0; t < 5; ++t) {  // 4 rounds of O^T tiles, then the l tiles
                tile_regs_acquire();
                for (uint32_t i = 0; i < 4; ++i) {
                    copy_tile(cZero, 0, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < 4; ++i) {
                    if (t < 4) {
                        pack_tile<true>(i, cO, (r * 4 + t) * 4 + i);
                    } else {
                        pack_tile<true>(i, cL, r * 4 + i);
                    }
                }
                tile_regs_release();
            }
        }
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
}

// add the m partials of the following cores (scratch order: O^T tiles 0 .. 4 QT - 1, then l tiles 0 .. QT - 1)
FORCE_INLINE void add_partials(uint32_t m) {
    SPT(9);
    reconfig_data_format_srca(cPart);
    copy_tile_to_dst_init_short(cPart);
    pack_reconfig_data_format(cO);
    PACK((llk_pack_reconfig_l1_acc(1)));
    for (uint32_t p = 0; p < m; ++p) {
        for (uint32_t t = 0; t < kPart; t += 4) {
            cb_wait_front(cPart, 4);
            tile_regs_acquire();
            for (uint32_t i = 0; i < 4; ++i) {
                copy_tile(cPart, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < 4; ++i) {
                if (t < QT * 4) {
                    pack_tile<true>(i, cO, t + i);
                } else {
                    pack_tile<true>(i, cL, t - QT * 4 + i);
                }
            }
            tile_regs_release();
            cb_pop_front(cPart, 4);
        }
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
}

// out tiles of query tile j = (r, hh): transpose(O^T(d, j) * s), s = sigmoid(g) / l
FORCE_INLINE void epilogue(uint32_t g) {
    SPT(11);
    cb_wait_front(cO, QT * 4);
    cb_wait_front(cL, QT);
    if constexpr (HAS_GATE) {
        cb_wait_front(cGr, RT);
    }
    sfpu_init_epi();
    for (uint32_t r = 0; r < RT; ++r) {
        for (uint32_t hh = 0; hh < 4; ++hh) {
            const uint32_t j = r * 4 + hh;
            // l, O^T(0), O^T(1) (FP32 straight to DEST), gate logits of head 4 g + hh broadcast over rows
            // (E_c x gate tile^T: row u = column c of the gate tile)
            reconfig_data_format_srca(cL);
            copy_tile_to_dst_init_short(cL);
            tile_regs_acquire();
            copy_tile(cL, j, 0);
            copy_tile(cO, j * 4, 2);
            copy_tile(cO, j * 4 + 1, 3);
            if constexpr (HAS_GATE) {
                reconfig_data_format(cGr, cE);
                matmul_block_init(cE, cGr, 1, 1, 1, 1);
                matmul_block(cE, cGr, 4 * g + hh, r, 1, 1, 1, 1, 1);
            }
            tile_regs_commit();
            cb_reserve_back(cOs, 4);
            cb_reserve_back(cS, 1);
            tile_regs_wait();
            SP_SFPU(sp_epi_first, 0);
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_reconfig_data_format(cOs);
            pack_tile<true>(2, cOs, 0);
            pack_tile<true>(3, cOs, 1);
            pack_reconfig_data_format(cS);
            pack_tile(0, cS);
            tile_regs_release();
            cb_push_back(cS, 1);
            // s, O^T(2), O^T(3)
            cb_wait_front(cS, 1);
            reconfig_data_format_srca(cS);
            copy_tile_to_dst_init_short(cS);
            tile_regs_acquire();
            copy_tile(cS, 0, 0);
            copy_tile(cO, j * 4 + 2, 1);
            copy_tile(cO, j * 4 + 3, 2);
            tile_regs_commit();
            tile_regs_wait();
            SP_SFPU(sp_epi_second, 0);
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_reconfig_data_format(cOs);
            pack_tile<true>(1, cOs, 2);
            pack_tile<true>(2, cOs, 3);
            tile_regs_release();
            cb_push_back(cOs, 4);
            SPDBG("C epi j={}\n", j);
            cb_pop_front(cS, 1);
            // transpose -> output tiles (rows = queries, columns = head dims)
            cb_wait_front(cOs, 4);
            reconfig_data_format_srca(cOs);
            transpose_init(cOs);
            pack_reconfig_data_format(cOut);
            tile_regs_acquire();
            for (uint32_t d = 0; d < 4; ++d) {
                transpose_tile(cOs, d, d);
            }
            tile_regs_commit();
            cb_reserve_back(cOut, 4);
            tile_regs_wait();
            for (uint32_t d = 0; d < 4; ++d) {
                pack_tile(d, cOut);
            }
            tile_regs_release();
            cb_push_back(cOut, 4);
            cb_pop_front(cOs, 4);
        }
    }
    cb_pop_front(cO, QT * 4);
    cb_pop_front(cL, QT);
    if constexpr (HAS_GATE) {
        cb_pop_front(cGr, RT);
    }
}

FORCE_INLINE void compute() {
#ifdef SP_PROF
    for (uint32_t i = 0; i < 12; ++i) {
        prof_acc[i] = 0;
    }
    prof_cur = 0;
    prof_last = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
#endif
    cb_wait_front(cPar, 1);
    const uint32_t pa = get_tile_address(cPar, 0);
    volatile uint32_t* plan = reinterpret_cast<volatile uint32_t*>(pa);
    const uint32_t ns = plan[0], st = plan[1];
    if (ns == 0) {
        cb_pop_front(cPar, 1);
        return;
    }
    cb_wait_front(cZero, 1);
    cb_wait_front(cOnes, 1);
    cb_wait_front(cMask, 2);
    cb_wait_front(cE, 16);
    sfpu_init_once();
    for (uint32_t i = 0; i < ns; ++i) {
        SPT(0);
        const uint32_t w0 = plan[kHdr + kSegW * i];
        const uint32_t u = w0 & 0xFFFF, fl = w0 >> 16;
        const uint32_t lo = plan[kHdr + kSegW * i + 1], hi = plan[kHdr + kSegW * i + 2];
        const uint32_t peer = plan[kHdr + kSegW * i + 3];
        const uint32_t b = u >> 2, g = u & 3;
        uint32_t kt0, n;
        unit_range(st, u, kt0, n);
        // Q^T: every Q tile transposed once per segment (slot kept), so the QK matmul needs no unpack transpose; the
        // reader streams the unit's Q one row tile (16 tiles) at a time
        reconfig_data_format_srca(cQ);
        transpose_init(cQ);
        pack_reconfig_data_format(cQT);
        PACK((llk_pack_reconfig_l1_acc(0)));
        cb_reserve_back(cQT, QT * 4);
        for (uint32_t r = 0; r < RT; ++r) {
            cb_wait_front(cQ, 16);
            for (uint32_t t = 0; t < 16; t += 4) {
                tile_regs_acquire();
                for (uint32_t d = 0; d < 4; ++d) {
                    transpose_tile(cQ, t + d, d);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t d = 0; d < 4; ++d) {
                    pack_tile<true>(d, cQT, r * 16 + t + d);
                }
                tile_regs_release();
            }
            cb_pop_front(cQ, 16);
        }
        cb_push_back(cQT, QT * 4);
        cb_wait_front(cQT, QT * 4);
        cb_reserve_back(cO, QT * 4);
        cb_reserve_back(cL, QT);
        segment_body(st, b, kt0 + lo, kt0 + hi);
        SPDBG("C seg {} done\n", i);
        cb_pop_front(cQT, QT * 4);
        if ((fl & kFirst) && !(fl & kLast)) {
            add_partials(peer);
        }
        cb_push_back(cO, QT * 4);
        cb_push_back(cL, QT);
        if (fl & kFirst) {
            epilogue(g);
        } else {
            // partial of a split unit: the writer copies cO / cL out in place (cSend), then acks (cAck)
            cb_reserve_back(cSend, 1);
            cb_push_back(cSend, 1);
            cb_wait_front(cAck, 1);
            cb_pop_front(cAck, 1);
            cb_wait_front(cO, QT * 4);
            cb_wait_front(cL, QT);
            cb_pop_front(cO, QT * 4);
            cb_pop_front(cL, QT);
        }
    }
    cb_pop_front(cPar, 1);
    SPDBG("C end\n");
    // leave the SFPU as the firmware / stock LLK inits expect it for later programs on this core: LREG11 = -1
    // (vConstNeg1; the exp uses it as Q0 and only finalizer cores run the epilogue's restore) and the default SFPU
    // config (LOADMACRO control).  Without this the fast decode step after a split-unit prefill chunk diverged.
    PACK((ckernel::sfpu::sp_sfpu_restore()));
#ifdef SP_PROF
    prof_emit();
#endif
}
}  // namespace sp
#endif

void kernel_main() {
    const uint32_t c = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_NCRISC)
    sp::reader(c);
#elif defined(COMPILE_FOR_BRISC)
    sp::writer(c);
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(sp::cK, sp::cQ, sp::cP);
    sp::compute();
#endif
}
