// SPDX-License-Identifier: Apache-2.0
// Inkling-2-Mini fused glue, tokens as tile rows (prefill chunks / speculative-verify rows).  Python side and the
// semantics: tt/fast/glue_rows.py.  One program = one glue segment of a layer application over NR row tiles
// (Tp = 32 NR tokens, token t = row t % 32 of row tile t / 32):
//   prologue  PRO 0: v = h (given, BF16)
//             PRO 1: c = bf16(conv(x)), v = h = bf16(r_prev + c)        (post-MLP conv + residual, h -> OH)
//             PRO 2: c = bf16(conv(x)), v = r = h + c (BOUND: r = c)     (post-attention conv + residual, FP32 r -> OH)
//             conv: y_t = t3 x_t + t0 x_{t-3} + t1 x_{t-2} + t2 x_{t-1} (taps folded with the residual / sqrt(5));
//             x_{t-s} = S_s x_n + S'_s x_{n-1} (one-hot shift matmuls, exact), x_{-1} = the conv ring's rows at
//             positions start-3 .. start-1; the inputs of the last WR valid rows go into the ring (row pos % 32)
//   STOP      done after the prologue (end of a loop pass)
//   round A   per token: q . X_i (block states, raw BF16, i < NS) and q . v for NQ queries, sum v^2 (Gram diagonal)
//             -> master: s_v = 1/rms(v), logits (s_i: per-token state scales from SC, s_v), softmax over the states,
//             coefficients C = W s (row form [4 i + q][t]); APPEND: s_v -> SC row NS (v becomes state NS)
//   NQ == 0   out = bf16(v s_v gamma)                                                  (layer 0)
//   mix       O_q = sum_i diag(C_iq) X_i + diag(C_vq) v                                (FP32 via hi / lo matmuls)
//   round B   1 / rms(O_q) -> column form
//   OUTM 0    out_q = bf16(O_q / rms(O_q) gamma_q);  OUTM 1: O_q (FP32) -> OO_q, nx_q = bf16(O_q / rms(O_q)) -> out_q
//   OUTM 2    (message gates follow) OUTM 1 with nx_q -> block q of CAT, plus the NG gates' messages
//             m_k = s1 + w_k (s0 - s1) (w_k = column c_k of the per-token message weights W) -> M_k (FP32) and
//             nm_k = bf16(m_k / rms(m_k)) -> block NQ + k of CAT (|m_k|^2 rides along in round A); the caller's
//             matmul CAT x [down weights] gives the gates' g = bf16(down(cat(norm x, norm m)))
//   GATE      y_k = O_src(k) - tanh(g_k Wf_k) O_src(k) + tanh(g_k Wg_k) m_k -> round B / outputs as O_k (NQ = NG)
//
// Layout: NGRP groups of CG cores (rectangles, master = the group's first core); group g handles row tiles
// g, g + NGRP, ...; core c of a group owns hidden column tiles [c TC, c TC + TC).  Cross-core sums go to the
// group's master as BF16 hi / lo rows of gather tiles (PF quantities per gather tile, rows m CG + c), reduced with
// exact BF16 matmuls (selector tiles E), results multicast to the group.  Per-token scalars of the cores' partials are
// in row form ([quantity][token], from Q X^T matmuls / Gram diagonals); the master returns scales in column form
// (column 0 = the token's value, from an identity matmul with transposed B) and mixing coefficients in row form
// (cores expand them to diagonal tiles).  Products of BF16 values are exact and accumulate in FP32 DEST.
#include <cstdint>
#include "read_vc.hpp"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "glue_rows_sfpu.hpp"
#endif

namespace gr {
#define NCT(name) get_named_compile_time_arg_val(name)
constexpr uint32_t TC = NCT("TC");      // column tiles per core
constexpr uint32_t CG = NCT("CG");      // cores per group
constexpr uint32_t NGRP = NCT("NGRP");  // groups
constexpr uint32_t PRO = NCT("PRO");
constexpr uint32_t BOUND = NCT("BOUND");
constexpr uint32_t STOP = NCT("STOP");
constexpr uint32_t NQ = NCT("NQ");      // block-residual queries (0: layer 0 / END)
constexpr uint32_t NS = NCT("NS");      // stored block states
constexpr uint32_t APPEND = NCT("APPEND");
constexpr uint32_t OUTM = NCT("OUTM");
constexpr uint32_t NOUT = NCT("NOUT");  // output tensors (OUTM 0: gamma rows)
constexpr uint32_t PF = NCT("PF");      // gather rows per tile: floor(32 / CG)
constexpr uint32_t WR = NCT("WR");      // ring rows written (the last WR valid rows)
constexpr uint32_t INVH = NCT("INVH");
constexpr uint32_t EPS_BITS = NCT("EPS");
constexpr uint32_t BASE_F32 = NCT("BASE_F32");
constexpr uint32_t NG = NCT("NG");        // message gates (OUTM 2: messages computed; GATE: gates applied)
constexpr uint32_t GATE = NCT("GATE");
constexpr uint32_t GSAME = NCT("GSAME");  // GATE: every gate reads O_0 (sliding layers)

constexpr uint32_t HT = 72;
constexpr bool HAS_LO = PRO == 2;  // v = r (FP32 hi / lo); else BF16 h
constexpr bool MIX = NQ > 0;
constexpr bool MSG = OUTM == 2;
constexpr bool ROUND_A = !STOP && !GATE;
constexpr bool NEED_BASE = !GATE && (PRO == 0 || PRO == 1 || (PRO == 2 && !BOUND));
constexpr uint32_t NLT = MIX && !GATE ? NS + 1 : 0;  // logit partial tiles (states + v)
constexpr uint32_t NMG = MSG ? NG : 0;                 // message Gram partial tiles
constexpr uint32_t NPA = NLT + 1 + NMG;                // round-A partial tiles (+ Gram of v)
constexpr uint32_t NSEL = NLT * NQ;
constexpr uint32_t NXO = GSAME ? 1 : NG;               // GATE: O sources
constexpr uint32_t CATW = (NQ + NG) * 72;              // OUTM 2: CAT width in tiles
// gather entries: round A: logits (u = i NQ + q, hl) -> entry 2u + hl, slots [0, SA_L); sum v^2 (hl) -> slots
// [SA_L, SA_M); |m_k|^2 -> slots SA_M + k SPQ ..; round B: q -> slots NSLOT_A + q SPQ + (hl / PF)
constexpr uint32_t SPQ = (2 + PF - 1) / PF;
constexpr uint32_t SA_L = (2 * NSEL + PF - 1) / PF;
constexpr uint32_t SA_M = SA_L + SPQ;
constexpr uint32_t NSLOT_A = ROUND_A ? SA_M + NMG * SPQ : 0;
constexpr uint32_t NSLOT_B = MIX ? NQ * SPQ : 0;
constexpr uint32_t NSLOT = NSLOT_A + NSLOT_B;
// slot s's gather entries (member rows m CG + c, m < members): the rows >= members CG stay zero (master zero-fills
// them at start; disjoint from the remote writes)
constexpr uint32_t slot_members(uint32_t s) {
    if (s < SA_L) {
        return 2 * NSEL - s * PF < PF ? 2 * NSEL - s * PF : PF;
    }
    const uint32_t e = ((s < NSLOT_A ? (s - SA_L) % SPQ : (s - NSLOT_A) % SPQ)) * PF;
    return 2 - e < PF ? 2 - e : PF;
}

// constants (CB_CONST): cores read [0, K_CORE); the master all
constexpr uint32_t K_I = 0, K_S = 1, K_SP = 4, K_SEL = 7, K_CORE = K_SEL + NSEL;
constexpr uint32_t K_REPX = K_CORE, K_REPV = K_CORE + 1, K_SUMQ = K_CORE + 2, K_MASK = K_CORE + 3,
                   K_MINV = K_CORE + 4, K_EA = K_CORE + 5, K_EB = K_EA + NSLOT_A, NCONST = K_EB + NSLOT_B;

// CB indices
constexpr uint32_t CB_IN = 0, CB_PREV = 1, CB_BASE = 2, CB_TAP = 3, CB_X = 4, CB_Q = 5, CB_GAM = 6, CB_CONST = 7,
                   CB_G = 8, CB_HV = 9, CB_RF = 10, CB_VH = 11, CB_VL = 12, CB_PF = 13, CB_PS = 14, CB_RAC = 15,
                   CB_RAS = 16, CB_RB = 17, CB_DG = 18, CB_OH = 19, CB_OL = 20, CB_OUT = 21, CB_OO = 22, CB_SC = 23,
                   CB_MF = 24, CB_MS = 25, CB_SV = 26, CB_STG = 27, CB_RSCR = 28, CB_ACK = 29, CB_W = 30, CB_WS = 31,
                   CB_CSEL = 32, CB_WC = 33, CB_S1 = 34, CB_DD = 35, CB_MH = 36, CB_ML = 37, CB_MO = 38, CB_RAM = 39,
                   CB_NM = 40, CB_GG = 41, CB_WF = 42, CB_WG = 43, CB_XO = 44, CB_MI = 45, CB_MF2 = 46, CB_MS2 = 47;

// common runtime args
constexpr uint32_t A_IN = 0, A_BASE = 1, A_RING = 2, A_TAP = 3, A_X = 7, A_SC = 11, A_Q = 12, A_GAM = 13, A_OUT = 16,
                   A_OH = 19, A_OO = 20, A_POS = 23, A_CONST = 24, A_NR = 25, A_S1 = 26, A_D = 27, A_W = 28,
                   A_CSEL = 29, A_GC = 30, A_CAT = 33, A_M = 34, A_G = 37, A_WF = 38, A_WG = 41;
// per-core runtime args
constexpr uint32_t P_G = 0, P_C = 1, P_MX = 2, P_MY = 3, P_X0 = 4, P_Y0 = 5, P_X1 = 6, P_Y1 = 7;

constexpr uint32_t BF = 2048, FP = 4096;
}  // namespace gr

// ====================================================================================================== RISCs
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace gr {
FORCE_INLINE uint32_t arg(uint32_t i) { return get_common_arg_val<uint32_t>(i); }
FORCE_INLINE uint32_t parg(uint32_t i) { return get_arg_val<uint32_t>(i); }

template <bool DRAM_>
FORCE_INLINE auto accessor(uint32_t addr, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<DRAM_>(), addr, page);
}

FORCE_INLINE void zero_fill(uint32_t addr, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes) {
        const uint32_t n = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(zeros, addr, n);
        addr += n;
        bytes -= n;
    }
}

// byte offset of (row r, columns 16 f ..) in a 32 x 32 tile of `esz`-byte elements
FORCE_INLINE uint32_t row_off(uint32_t r, uint32_t f, uint32_t esz) {
    return ((r >> 4) * 2 + f) * 256 * esz + (r & 15) * 16 * esz;
}

// start position / valid rows (int32 words 0 / 1 of the pos tensor); GR_SLOTS (batched model, slot-major rings):
// word 2 = the sequence's slot -> sb = its first ring page (slot * HT; 0 without GR_SLOTS)
FORCE_INLINE void read_pos(uint32_t scratch, uint32_t& start, uint32_t& valid, uint32_t& sb) {
    const auto p = accessor<true>(arg(A_POS), 128);
    noc_async_read(get_noc_addr(0, p), scratch, 64);
    noc_async_read_barrier();
    auto* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    start = w[0];
    valid = w[1];
#ifdef GR_SLOTS
    sb = w[2] * HT;
#endif
}
}  // namespace gr
#endif

#if defined(COMPILE_FOR_NCRISC)
// ------------------------------------------------------------------------------------------------ reader
void kernel_main() {
    ink_set_read_vc_rows();
    using namespace gr;
    const uint32_t g = parg(P_G), c = parg(P_C);
    const bool master = c == 0;
    const uint32_t j0 = c * TC;
    const uint32_t NR = arg(A_NR);
    const uint32_t scr = get_write_ptr(CB_RSCR);
    uint32_t start = 0, valid = 0, sb = 0;
    if constexpr (PRO > 0) {
        read_pos(scr, start, valid, sb);
    }

    // ---- once: constants, taps, queries, gamma; master: zero the gather tiles
    {
        const auto kt = accessor<true>(arg(A_CONST), BF);
        cb_reserve_back(CB_CONST, NCONST);
        const uint32_t kc = get_write_ptr(CB_CONST);
        const uint32_t nk = master ? NCONST : K_CORE;
        for (uint32_t i = 0; i < nk; ++i) {
            noc_async_read_tile(i, kt, kc + i * BF);
        }
        if constexpr (PRO > 0) {
            cb_reserve_back(CB_TAP, 4 * TC);
            const uint32_t w = get_write_ptr(CB_TAP);
            for (uint32_t i = 0; i < 4; ++i) {
                const auto tp = accessor<true>(arg(A_TAP + i), FP);
                for (uint32_t j = 0; j < TC; ++j) {
                    noc_async_read_tile(j0 + j, tp, w + (4 * j + i) * FP);
                }
            }
        }
        if constexpr (MIX && !GATE) {
            const auto q = accessor<true>(arg(A_Q), BF);
            cb_reserve_back(CB_Q, 2 * TC);
            const uint32_t w = get_write_ptr(CB_Q);
            for (uint32_t j = 0; j < TC; ++j) {
                noc_async_read_tile(j0 + j, q, w + j * BF);
                noc_async_read_tile(HT + j0 + j, q, w + (TC + j) * BF);
            }
        }
        if constexpr (MSG) {  // column selectors E_(c_k % 32) of the message-weight columns
            const auto cs = accessor<true>(arg(A_CSEL), BF);
            cb_reserve_back(CB_CSEL, NG);
            const uint32_t w = get_write_ptr(CB_CSEL);
            for (uint32_t k = 0; k < NG; ++k) {
                noc_async_read_tile(arg(A_GC + k) & 31, cs, w + k * BF);
            }
        }
        if constexpr (GATE) {  // forget / gain weights [64, H]: tiles (c2, column)
            cb_reserve_back(CB_WF, 2 * NG * TC);
            cb_reserve_back(CB_WG, 2 * NG * TC);
            const uint32_t wf = get_write_ptr(CB_WF), wg = get_write_ptr(CB_WG);
            for (uint32_t k = 0; k < NG; ++k) {
                const auto f = accessor<true>(arg(A_WF + k), BF);
                const auto gg = accessor<true>(arg(A_WG + k), BF);
                for (uint32_t c2 = 0; c2 < 2; ++c2) {
                    for (uint32_t j = 0; j < TC; ++j) {
                        noc_async_read_tile(c2 * HT + j0 + j, f, wf + ((2 * k + c2) * TC + j) * BF);
                        noc_async_read_tile(c2 * HT + j0 + j, gg, wg + ((2 * k + c2) * TC + j) * BF);
                    }
                }
            }
        }
        if constexpr (!STOP && OUTM == 0) {
            cb_reserve_back(CB_GAM, NOUT * TC);
            const uint32_t w = get_write_ptr(CB_GAM);
            for (uint32_t o = 0; o < NOUT; ++o) {
                const auto gm = accessor<true>(arg(A_GAM + o), BF);
                for (uint32_t j = 0; j < TC; ++j) {
                    noc_async_read_tile(j0 + j, gm, w + (o * TC + j) * BF);
                }
            }
        }
        if (master && !STOP) {
            const uint32_t gb = get_write_ptr(CB_G);
            for (uint32_t s = 0; s < NSLOT; ++s) {
                const uint32_t r0 = slot_members(s) * CG;
                for (uint32_t f = 0; f < 4; ++f) {
                    const uint32_t fr = (f >> 1) * 16;
                    const uint32_t a = r0 > fr ? (r0 - fr < 16 ? r0 - fr : 16) : 0;
                    if (a < 16) {
                        zero_fill(gb + s * BF + f * 512 + a * 32, (16 - a) * 32);
                    }
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_CONST, NCONST);
        if constexpr (PRO > 0) {
            cb_push_back(CB_TAP, 4 * TC);
        }
        if constexpr (MIX && !GATE) {
            cb_push_back(CB_Q, 2 * TC);
        }
        if constexpr (MSG) {
            cb_push_back(CB_CSEL, NG);
        }
        if constexpr (GATE) {
            cb_push_back(CB_WF, 2 * NG * TC);
            cb_push_back(CB_WG, 2 * NG * TC);
        }
        if constexpr (!STOP && OUTM == 0) {
            cb_push_back(CB_GAM, NOUT * TC);
        }
    }

    // ---- per row tile
    for (uint32_t n = g; n < NR; n += NGRP) {
        if constexpr (PRO > 0) {
            const auto in = accessor<true>(arg(A_IN), BF);
            cb_reserve_back(CB_IN, TC);
            cb_reserve_back(CB_PREV, TC);
            const uint32_t wi = get_write_ptr(CB_IN), wp = get_write_ptr(CB_PREV);
            for (uint32_t j = 0; j < TC; ++j) {
                noc_async_read_tile(n * HT + j0 + j, in, wi + j * BF);
            }
            if (n > 0) {
                for (uint32_t j = 0; j < TC; ++j) {
                    noc_async_read_tile((n - 1) * HT + j0 + j, in, wp + j * BF);
                }
            } else {
                // context tile: rows 29..31 = ring rows of positions start-3 .. start-1 (FP32 -> BF16, exact)
                zero_fill(wp, TC * BF);
                const auto rg = accessor<true>(arg(A_RING), FP);
                for (uint32_t j = 0; j < TC; ++j) {
                    const uint64_t a = get_noc_addr(sb + j0 + j, rg);
                    for (uint32_t s = 0; s < 3; ++s) {
                        const uint32_t rr = (start + 29 + s) & 31;
                        for (uint32_t f = 0; f < 2; ++f) {
                            noc_async_read(a + row_off(rr, f, 4), scr + 64 + ((j * 3 + s) * 2 + f) * 64, 64);
                        }
                    }
                }
                noc_async_read_barrier();
                for (uint32_t j = 0; j < TC; ++j) {
                    for (uint32_t s = 0; s < 3; ++s) {
                        for (uint32_t f = 0; f < 2; ++f) {
                            auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 64 + ((j * 3 + s) * 2 + f) * 64);
                            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wp + j * BF + row_off(29 + s, f, 2));
                            for (uint32_t e = 0; e < 16; ++e) {
                                dst[e] = static_cast<uint16_t>(src[e] >> 16);
                            }
                        }
                    }
                }
            }
        }
        if constexpr (NEED_BASE) {
            constexpr uint32_t bp = BASE_F32 ? FP : BF;
            const auto b = accessor<true>(arg(A_BASE), bp);
            cb_reserve_back(CB_BASE, TC);
            const uint32_t w = get_write_ptr(CB_BASE);
            for (uint32_t j = 0; j < TC; ++j) {
                noc_async_read_tile(n * HT + j0 + j, b, w + j * bp);
            }
        }
        if constexpr (MIX && NS > 0) {
            cb_reserve_back(CB_X, NS * TC);
            const uint32_t w = get_write_ptr(CB_X);
            for (uint32_t i = 0; i < NS; ++i) {
                const auto x = accessor<true>(arg(A_X + i), BF);
                for (uint32_t j = 0; j < TC; ++j) {
                    noc_async_read_tile(n * HT + j0 + j, x, w + (i * TC + j) * BF);
                }
            }
            if (master) {
                const auto sc = accessor<true>(arg(A_SC), FP);
                cb_reserve_back(CB_SC, 1);
                // rows < NS only (the tensor's other rows may be uninitialized): zero tile + row reads
                const uint32_t ws = get_write_ptr(CB_SC);
                zero_fill(ws, FP);
                noc_async_read_barrier();
                const uint64_t a = get_noc_addr(n, sc);
                for (uint32_t i = 0; i < NS; ++i) {
                    noc_async_read(a + row_off(i, 0, 4), ws + row_off(i, 0, 4), 64);
                    noc_async_read(a + row_off(i, 1, 4), ws + row_off(i, 1, 4), 64);
                }
            }
        }
        if constexpr (MSG) {  // message weights W (tiles (n, 0..1) of [Tp, 64]), s1 and d = s0 - s1 (FP32)
            const auto wa = accessor<true>(arg(A_W), FP);
            const auto s1 = accessor<true>(arg(A_S1), FP);
            const auto dd = accessor<true>(arg(A_D), FP);
            cb_reserve_back(CB_W, 2);
            cb_reserve_back(CB_S1, TC);
            cb_reserve_back(CB_DD, TC);
            const uint32_t ww = get_write_ptr(CB_W), w1 = get_write_ptr(CB_S1), wd = get_write_ptr(CB_DD);
            noc_async_read_tile(2 * n, wa, ww);
            noc_async_read_tile(2 * n + 1, wa, ww + FP);
            for (uint32_t j = 0; j < TC; ++j) {
                noc_async_read_tile(n * HT + j0 + j, s1, w1 + j * FP);
                noc_async_read_tile(n * HT + j0 + j, dd, wd + j * FP);
            }
        }
        if constexpr (GATE) {  // O sources, messages (FP32), g (BF16 [Tp, 64 NG]: tiles (n, 2k + c2))
            cb_reserve_back(CB_XO, NXO * TC);
            cb_reserve_back(CB_MI, NG * TC);
            cb_reserve_back(CB_GG, 2 * NG);
            const uint32_t wx = get_write_ptr(CB_XO), wm = get_write_ptr(CB_MI), wg = get_write_ptr(CB_GG);
            for (uint32_t k = 0; k < NXO; ++k) {
                const auto o = accessor<true>(arg(A_OO + k), FP);
                for (uint32_t j = 0; j < TC; ++j) {
                    noc_async_read_tile(n * HT + j0 + j, o, wx + (k * TC + j) * FP);
                }
            }
            for (uint32_t k = 0; k < NG; ++k) {
                const auto mm = accessor<true>(arg(A_M + k), FP);
                for (uint32_t j = 0; j < TC; ++j) {
                    noc_async_read_tile(n * HT + j0 + j, mm, wm + (k * TC + j) * FP);
                }
            }
            const auto ga = accessor<true>(arg(A_G), BF);
            for (uint32_t i = 0; i < 2 * NG; ++i) {
                noc_async_read_tile(n * 2 * NG + i, ga, wg + i * BF);
            }
        }
        noc_async_read_barrier();
        if constexpr (PRO > 0) {
            cb_push_back(CB_IN, TC);
            cb_push_back(CB_PREV, TC);
        }
        if constexpr (NEED_BASE) {
            cb_push_back(CB_BASE, TC);
        }
        if constexpr (MSG) {
            cb_push_back(CB_W, 2);
            cb_push_back(CB_S1, TC);
            cb_push_back(CB_DD, TC);
        }
        if constexpr (GATE) {
            cb_push_back(CB_XO, NXO * TC);
            cb_push_back(CB_MI, NG * TC);
            cb_push_back(CB_GG, 2 * NG);
        }
        if constexpr (MIX && NS > 0) {
            cb_push_back(CB_X, NS * TC);
            if (master) {
                cb_push_back(CB_SC, 1);
            }
        }
        if constexpr (PRO > 0) {
            // conv ring: inputs of the last WR valid rows -> ring row (start + t) % 32 (after the context read above,
            // which only the cores of row tile 0 do; the same cores write, so no other core reads these rows)
            if (n == 0) {
                const auto in = accessor<true>(arg(A_IN), BF);
                const auto rg = accessor<true>(arg(A_RING), FP);
                const uint32_t t0 = valid > WR ? valid - WR : 0;
                const uint32_t stg = scr + 64 + 6 * TC * 64;
                for (uint32_t t = t0; t < valid; ++t) {
                    const uint32_t r = t & 31, k = t - t0;
                    for (uint32_t j = 0; j < TC; ++j) {
                        const uint64_t a = get_noc_addr((t >> 5) * HT + j0 + j, in);
                        for (uint32_t f = 0; f < 2; ++f) {  // 64 B-aligned row pair containing row r
                            noc_async_read(a + row_off(r & ~1u, f, 2), stg + ((k * TC + j) * 2 + f) * 64, 64);
                        }
                    }
                }
                noc_async_read_barrier();
                const uint32_t cvt = stg + WR * TC * 128;
                for (uint32_t t = t0; t < valid; ++t) {
                    const uint32_t r = t & 31, k = t - t0, rr = (start + t) & 31;
                    for (uint32_t j = 0; j < TC; ++j) {
                        const uint64_t a = get_noc_addr(sb + j0 + j, rg);
                        for (uint32_t f = 0; f < 2; ++f) {
                            auto* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stg + ((k * TC + j) * 2 + f) * 64 +
                                                                                     (r & 1) * 32);
                            const uint32_t d = cvt + ((k * TC + j) * 2 + f) * 64;
                            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d);
                            for (uint32_t e = 0; e < 16; ++e) {
                                dst[e] = static_cast<uint32_t>(src[e]) << 16;
                            }
                            noc_async_write(d, a + row_off(rr, f, 4), 64);
                        }
                    }
                }
                noc_async_write_barrier();
            }
        }
    }
    noc_async_write_barrier();
}

#elif defined(COMPILE_FOR_BRISC)
// ------------------------------------------------------------------------------------------------ writer / rounds
namespace gr {
uint32_t g_l1 = 0;
uint32_t mx = 0, my = 0, mcx0 = 0, mcy0 = 0, mcx1 = 0, mcy1 = 0;

FORCE_INLINE uint64_t mcast_addr(uint32_t addr) {
    if (noc_index == 1) {
        return get_noc_multicast_addr(mcx1, mcy1, mcx0, mcy0, addr);
    }
    return get_noc_multicast_addr(mcx0, mcy0, mcx1, mcy1, addr);
}

// (row, half) of gather slot s on the master
FORCE_INLINE uint64_t g_row(uint32_t s, uint32_t row, uint32_t half) {
    return get_noc_addr(mx, my, g_l1 + s * BF + row_off(row, half, 2));
}

// gather entry (slot base sb, entry e) of core c
FORCE_INLINE void send_row(uint32_t src, uint32_t sb, uint32_t e, uint32_t c) {
    const uint32_t s = sb + e / PF, row = (e % PF) * CG + c;
    noc_async_write(src, g_row(s, row, 0), 32);
    noc_async_write(src + 512, g_row(s, row, 1), 32);
}

// row q of a BF16 tile (two 32 B halves 512 B apart for q < 16)
FORCE_INLINE uint32_t tile_row(uint32_t tile, uint32_t q) { return tile + row_off(q, 0, 2); }

// diagonal of a BF16 tile -> a staging row (halves at +0 / +512 like a tile row)
FORCE_INLINE void diag_row(uint32_t tile, uint32_t stg) {
    auto* d0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stg);
    auto* d1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stg + 512);
    for (uint32_t t = 0; t < 16; ++t) {
        d0[t] = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile + row_off(t, 0, 2) + t * 2);
        d1[t] = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile + row_off(16 + t, 1, 2) + t * 2);
    }
}

// semaphore round: after this core's gather writes -> master: wait all, push the gather slots, wait the result CB(s)
// and multicast them; others: wait the round counter, push the result CB(s).  res2_n = 0: no second result CB.
FORCE_INLINE void finish_round(bool master, uint32_t ord, uint32_t nslots, uint32_t res_cb, uint32_t res_n,
                               uint32_t res_page, uint32_t res2_cb = 0, uint32_t res2_n = 0, uint32_t res2_page = 0) {
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(mx, my, get_semaphore(0)), 1);
    if (master) {
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait_min(sem, CG * (ord + 1));
        cb_reserve_back(CB_G, nslots);
        cb_push_back(CB_G, nslots);
        cb_wait_front(res_cb, res_n);  // the master's compute pops them only after CB_ACK
        if (res2_n) {
            cb_wait_front(res2_cb, res2_n);
        }
        if constexpr (CG > 1) {
            const uint32_t src = get_read_ptr(res_cb);
            noc_async_write_multicast(src, mcast_addr(src), res_n * res_page, CG - 1, true);
            if (res2_n) {
                const uint32_t src2 = get_read_ptr(res2_cb);
                noc_async_write_multicast(src2, mcast_addr(src2), res2_n * res2_page, CG - 1, true);
            }
            const uint32_t rs = get_semaphore(1);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs) = ord + 1;
            noc_semaphore_set_multicast(rs, mcast_addr(rs), CG - 1);
            noc_async_write_barrier();
        }
        cb_reserve_back(CB_ACK, 1);
        cb_push_back(CB_ACK, 1);
    } else {
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        noc_semaphore_wait_min(sem, ord + 1);
        cb_reserve_back(res_cb, res_n);
        cb_push_back(res_cb, res_n);
        if (res2_n) {
            cb_reserve_back(res2_cb, res2_n);
            cb_push_back(res2_cb, res2_n);
        }
    }
}

// TC tiles at the front of cb -> tensor tiles base + j (page bytes), then pop
FORCE_INLINE void drain(uint32_t cb, uint32_t addr, uint32_t page, uint32_t base) {
    const auto o = accessor<true>(addr, page);
    cb_wait_front(cb, TC);
    const uint32_t r = get_read_ptr(cb);
    for (uint32_t j = 0; j < TC; ++j) {
        noc_async_write_tile(base + j, o, r + j * page);
    }
    noc_async_write_barrier();
    cb_pop_front(cb, TC);
}
}  // namespace gr

void kernel_main() {
    ink_set_read_vc_rows();
    using namespace gr;
    const uint32_t g = parg(P_G), c = parg(P_C);
    mx = parg(P_MX);
    my = parg(P_MY);
    mcx0 = parg(P_X0);
    mcy0 = parg(P_Y0);
    mcx1 = parg(P_X1);
    mcy1 = parg(P_Y1);
    const bool master = c == 0;
    const uint32_t j0 = c * TC;
    const uint32_t NR = arg(A_NR);
    g_l1 = get_write_ptr(CB_G);
    const uint32_t stg = get_write_ptr(CB_STG);
    uint32_t ord = 0;
    for (uint32_t n = g; n < NR; n += NGRP) {
        // prologue output: h (BF16) / r (FP32)
        if constexpr (PRO == 1) {
            const auto o = accessor<true>(arg(A_OH), BF);
            cb_wait_front(CB_HV, TC);
            const uint32_t r = get_read_ptr(CB_HV);
            for (uint32_t j = 0; j < TC; ++j) {
                noc_async_write_tile(n * HT + j0 + j, o, r + j * BF);
            }
            noc_async_write_barrier();
            if constexpr (STOP) {
                cb_pop_front(CB_HV, TC);
            }
        } else if constexpr (PRO == 2) {
            const auto o = accessor<true>(arg(A_OH), FP);
            cb_wait_front(CB_RF, TC);
            const uint32_t r = get_read_ptr(CB_RF);
            for (uint32_t j = 0; j < TC; ++j) {
                noc_async_write_tile(n * HT + j0 + j, o, r + j * FP);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_RF, TC);
        }
        if constexpr (STOP) {
            continue;
        }
        if constexpr (MSG) {  // the gates' messages m_k (FP32) for the gate program
            for (uint32_t k = 0; k < NG; ++k) {
                drain(CB_MO, arg(A_M + k), FP, n * HT + j0);
            }
        }

        // ---- round A: logits rows q of the logit tiles, Gram diagonals (v, messages)
        if constexpr (ROUND_A) {
            cb_wait_front(CB_PS, 2 * NPA);
            const uint32_t ps = get_read_ptr(CB_PS);
            for (uint32_t i = 0; i < NLT; ++i) {
                for (uint32_t q = 0; q < NQ; ++q) {
                    for (uint32_t hl = 0; hl < 2; ++hl) {
                        send_row(tile_row(ps + (2 * i + hl) * BF, q), 0, 2 * (i * NQ + q) + hl, c);
                    }
                }
            }
            for (uint32_t hl = 0; hl < 2; ++hl) {
                diag_row(ps + (2 * NLT + hl) * BF, stg + hl * 32);
                send_row(stg + hl * 32, SA_L, hl, c);
            }
            for (uint32_t k = 0; k < NMG; ++k) {
                for (uint32_t hl = 0; hl < 2; ++hl) {
                    const uint32_t st = stg + 256 + (2 * k + hl) * 32;
                    diag_row(ps + (2 * (NLT + 1 + k) + hl) * BF, st);
                    send_row(st, SA_M + k * SPQ, hl, c);
                }
            }
            if constexpr (MSG) {
                finish_round(master, ord, NSLOT_A, CB_RAC, 2, BF, CB_RAM, NG, FP);
            } else if constexpr (MIX) {
                finish_round(master, ord, NSLOT_A, CB_RAC, 2, BF);
            } else {
                finish_round(master, ord, NSLOT_A, CB_RAS, 1, FP);
            }
            ++ord;
            cb_pop_front(CB_PS, 2 * NPA);
        }
        if constexpr (MSG) {  // nm_k -> CAT block NQ + k
            for (uint32_t k = 0; k < NG; ++k) {
                drain(CB_NM, arg(A_CAT), BF, n * CATW + (NQ + k) * HT + j0);
            }
        }
        if constexpr (APPEND) {
            if (master) {  // s_v -> SC row NS of tile n
                const auto sc = accessor<true>(arg(A_SC), FP);
                cb_wait_front(CB_SV, 1);
                const uint32_t r = get_read_ptr(CB_SV);
                const uint64_t a = get_noc_addr(n, sc);
                noc_async_write(r, a + row_off(NS, 0, 4), 64);
                noc_async_write(r + 1024, a + row_off(NS, 1, 4), 64);
                noc_async_write_barrier();
                cb_pop_front(CB_SV, 1);
            }
        }

        // ---- round B: Gram diagonals of O_q
        if constexpr (MIX) {
            cb_wait_front(CB_PS, 2 * NQ);
            const uint32_t pb = get_read_ptr(CB_PS);
            for (uint32_t q = 0; q < NQ; ++q) {
                for (uint32_t hl = 0; hl < 2; ++hl) {
                    const uint32_t st = stg + 64 + (2 * q + hl) * 32;
                    diag_row(pb + (2 * q + hl) * BF, st);
                    send_row(st, NSLOT_A + q * SPQ, hl, c);
                }
            }
            finish_round(master, ord, NSLOT_B, CB_RB, NQ, FP);
            ++ord;
            cb_pop_front(CB_PS, 2 * NQ);
        }

        // ---- outputs
        constexpr uint32_t NO = MIX ? (OUTM == 0 ? NOUT : NQ) : 1;
        for (uint32_t o = 0; o < NO; ++o) {
            if constexpr (OUTM >= 1) {
                drain(CB_OO, arg(A_OO + o), FP, n * HT + j0);
            }
            if constexpr (OUTM == 2) {
                drain(CB_OUT, arg(A_CAT), BF, n * CATW + o * HT + j0);
            } else {
                drain(CB_OUT, arg(A_OUT + o), BF, n * HT + j0);
            }
        }
    }
    noc_async_write_barrier();
}

#else
// ====================================================================================================== compute
namespace gr {
ALWI void cp(uint32_t cb, uint32_t t, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, t, d);
}
ALWI void mm_begin(uint32_t a, uint32_t b, uint32_t tr) {
    reconfig_data_format(b, a);
    matmul_init(a, b, tr);
}
ALWI void pk(uint32_t d, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
}
ALWI void sec_begin() { tile_regs_acquire(); }
ALWI void sec_pack() {
    tile_regs_commit();
    tile_regs_wait();
}
ALWI void sec_end() { tile_regs_release(); }

// FP32 tiles [0, n) of `src` (front) -> hi / lo BF16 pairs pushed to `dst`; pops src
ALWI void split_tiles(uint32_t src, uint32_t n, uint32_t dst) {
    cb_wait_front(src, n);
    cb_reserve_back(dst, 2 * n);
    for (uint32_t u = 0; u < n; u += 2) {
        const uint32_t m = n - u < 2 ? n - u : 2;
        sec_begin();
        for (uint32_t k = 0; k < m; ++k) {
            cp(src, u + k, 3 * k);
        }
        rows::sfpu_init();
        for (uint32_t k = 0; k < m; ++k) {
            rows::split(3 * k, 3 * k + 1, 3 * k + 2);
        }
        sec_pack();
        for (uint32_t k = 0; k < m; ++k) {
            pk(3 * k + 1, dst);
            pk(3 * k + 2, dst);
        }
        sec_end();
    }
    cb_push_back(dst, 2 * n);
    cb_pop_front(src, n);
}

// master: D[dst] += E_(kbase + s) x G_s for the n gather slots at the CB_G front
ALWI void reduce(uint32_t kbase, uint32_t s0, uint32_t n, uint32_t dst) {  // E tile kbase + s for slot s
    for (uint32_t s = s0; s < s0 + n; ++s) {
        matmul_tiles(CB_CONST, CB_G, kbase + s, s, dst);
    }
}

// D[d] = column form of row 0 of the hi / lo tiles (t, t + 1) of `cb`:  I x B^T
ALWI void col_form(uint32_t cb, uint32_t t, uint32_t d) {
    mm_begin(CB_CONST, cb, 1);
    matmul_tiles(CB_CONST, cb, K_I, t, d);
    matmul_tiles(CB_CONST, cb, K_I, t + 1, d);
}

// master: finish round B: s_q = 1/rms(O_q) -> NQ column-form FP32 tiles in CB_RB
ALWI void finish_b() {
    cb_wait_front(CB_G, NSLOT_B);
    sec_begin();
    mm_begin(CB_CONST, CB_G, 0);
    for (uint32_t q = 0; q < NQ; ++q) {
        reduce(K_EB, q * SPQ, SPQ, q);
    }
    rows::sfpu_init();
    for (uint32_t q = 0; q < NQ; ++q) {
        rows::affine<INVH, EPS_BITS>(q);
        rows::rsqrt(q);
        rows::sfpu_init();
    }
    sec_pack();
    cb_reserve_back(CB_MF, NQ);
    for (uint32_t q = 0; q < NQ; ++q) {
        pk(q, CB_MF);
    }
    cb_push_back(CB_MF, NQ);
    sec_end();
    cb_pop_front(CB_G, NSLOT_B);
    split_tiles(CB_MF, NQ, CB_MS);
    cb_wait_front(CB_MS, 2 * NQ);
    sec_begin();
    for (uint32_t q = 0; q < NQ; ++q) {
        col_form(CB_MS, 2 * q, q);
    }
    sec_pack();
    cb_reserve_back(CB_RB, NQ);
    for (uint32_t q = 0; q < NQ; ++q) {
        pk(q, CB_RB);
    }
    cb_push_back(CB_RB, NQ);
    sec_end();
    cb_pop_front(CB_MS, 2 * NQ);
}

// master: finish round A
ALWI void finish_a() {
    cb_wait_front(CB_G, NSLOT_A);
    sec_begin();
    mm_begin(CB_CONST, CB_G, 0);
    if constexpr (MIX) {
        reduce(K_EA, 0, SA_L, 0);
    }
    reduce(K_EA, SA_L, SPQ, 1);  // sum v^2 -> D1 row 0
    for (uint32_t k = 0; k < NMG; ++k) {
        reduce(K_EA, SA_M + k * SPQ, SPQ, 2 + k);  // |m_k|^2 -> D(2 + k) row 0
    }
    rows::sfpu_init();
    for (uint32_t d = 1; d < 2 + NMG; ++d) {
        rows::affine<INVH, EPS_BITS>(d);
        rows::rsqrt(d);
        rows::sfpu_init();
    }
    sec_pack();
    cb_reserve_back(CB_MF, 2);
    pk(0, CB_MF);
    pk(1, CB_MF);
    cb_push_back(CB_MF, 2);
    if constexpr (APPEND) {
        cb_reserve_back(CB_SV, 1);
        pk(1, CB_SV);
        cb_push_back(CB_SV, 1);
    }
    if constexpr (MSG) {
        cb_reserve_back(CB_MF2, NG);
        for (uint32_t k = 0; k < NG; ++k) {
            pk(2 + k, CB_MF2);
        }
        cb_push_back(CB_MF2, NG);
    }
    sec_end();
    cb_pop_front(CB_G, NSLOT_A);
    if constexpr (MSG) {  // 1 / rms(m_k) in column form -> CB_RAM
        split_tiles(CB_MF2, NG, CB_MS2);
        cb_wait_front(CB_MS2, 2 * NG);
        sec_begin();
        for (uint32_t k = 0; k < NG; ++k) {
            col_form(CB_MS2, 2 * k, k);
        }
        sec_pack();
        cb_reserve_back(CB_RAM, NG);
        for (uint32_t k = 0; k < NG; ++k) {
            pk(k, CB_RAM);
        }
        cb_push_back(CB_RAM, NG);
        sec_end();
        cb_pop_front(CB_MS2, 2 * NG);
    }
    cb_wait_front(CB_MF, 2);
    if constexpr (!MIX) {  // s_v in column form
        sec_begin();
        cp(CB_MF, 1, 0);
        rows::sfpu_init();
        rows::split(0, 1, 2);
        sec_pack();
        cb_reserve_back(CB_MS, 2);
        pk(1, CB_MS);
        pk(2, CB_MS);
        cb_push_back(CB_MS, 2);
        sec_end();
        cb_wait_front(CB_MS, 2);
        sec_begin();
        col_form(CB_MS, 0, 0);
        sec_pack();
        cb_reserve_back(CB_RAS, 1);
        pk(0, CB_RAS);
        cb_push_back(CB_RAS, 1);
        sec_end();
        cb_pop_front(CB_MS, 2);
        cb_pop_front(CB_MF, 2);
        return;
    } else {
        // MS 0..3 = SC hi / lo, s_v hi / lo
        sec_begin();
        if constexpr (NS > 0) {
            cb_wait_front(CB_SC, 1);
            cp(CB_SC, 0, 0);
        }
        cp(CB_MF, 1, 3);
        rows::sfpu_init();
        rows::split(0, 1, 2);
        rows::split(3, 4, 5);
        sec_pack();
        cb_reserve_back(CB_MS, 4);
        pk(1, CB_MS);
        pk(2, CB_MS);
        pk(4, CB_MS);
        pk(5, CB_MS);
        cb_push_back(CB_MS, 4);
        sec_end();
        if constexpr (NS > 0) {
            cb_pop_front(CB_SC, 1);
        }
        cb_wait_front(CB_MS, 4);
        // S_rep (rows 4 i + q = s_i), E = exp(logits * S_rep) masked
        sec_begin();
        mm_begin(CB_CONST, CB_MS, 0);
        if constexpr (NS > 0) {
            matmul_tiles(CB_CONST, CB_MS, K_REPX, 0, 0);
            matmul_tiles(CB_CONST, CB_MS, K_REPX, 1, 0);
        }
        matmul_tiles(CB_CONST, CB_MS, K_REPV, 2, 0);
        matmul_tiles(CB_CONST, CB_MS, K_REPV, 3, 0);
        cp(CB_MF, 0, 1);
        rows::sfpu_init();
        rows::mul(1, 0, 1);
        rows::exp(1);
        cp(CB_CONST, K_MASK, 2);
        rows::sfpu_init();
        rows::mul(1, 2, 1);
        rows::split(1, 2, 3);
        sec_pack();
        cb_reserve_back(CB_MF, 2);
        pk(0, CB_MF);  // MF 2 = S_rep
        pk(1, CB_MF);  // MF 3 = E
        cb_push_back(CB_MF, 2);
        cb_reserve_back(CB_MS, 2);
        pk(2, CB_MS);  // MS 4, 5 = E hi / lo
        pk(3, CB_MS);
        cb_push_back(CB_MS, 2);
        sec_end();
        cb_wait_front(CB_MF, 4);
        cb_wait_front(CB_MS, 6);
        // Z = per-query sums over the states; C = E / Z * S_rep
        sec_begin();
        mm_begin(CB_CONST, CB_MS, 0);
        matmul_tiles(CB_CONST, CB_MS, K_SUMQ, 4, 0);
        matmul_tiles(CB_CONST, CB_MS, K_SUMQ, 5, 0);
        cp(CB_CONST, K_MINV, 1);
        rows::sfpu_init();
        rows::add(0, 1, 0);
        rows::recip(0);
        cp(CB_MF, 3, 1);
        cp(CB_MF, 2, 2);
        rows::sfpu_init();
        rows::mul(1, 0, 1);
        rows::mul(1, 2, 1);
        rows::split(1, 2, 3);
        sec_pack();
        cb_reserve_back(CB_RAC, 2);
        pk(2, CB_RAC);
        pk(3, CB_RAC);
        cb_push_back(CB_RAC, 2);
        sec_end();
        cb_pop_front(CB_MF, 4);
        cb_pop_front(CB_MS, 6);
    }
}
// round B partials (Gram diagonals of the NQ rows-of-tiles in CB_OH / CB_OL), master finish, outputs
ALWI void round_b_outputs(bool master) {
    cb_wait_front(CB_OH, NQ * TC);
    cb_wait_front(CB_OL, NQ * TC);
    sec_begin();
    mm_begin(CB_OH, CB_OH, 1);
    for (uint32_t q = 0; q < NQ; ++q) {
        for (uint32_t j = 0; j < TC; ++j) {
            matmul_tiles(CB_OH, CB_OH, q * TC + j, q * TC + j, q);
        }
    }
    mm_begin(CB_OH, CB_OL, 1);
    for (uint32_t q = 0; q < NQ; ++q) {
        for (uint32_t j = 0; j < TC; ++j) {
            matmul_tiles(CB_OH, CB_OL, q * TC + j, q * TC + j, q);
        }
    }
    mm_begin(CB_OL, CB_OH, 1);
    for (uint32_t q = 0; q < NQ; ++q) {
        for (uint32_t j = 0; j < TC; ++j) {
            matmul_tiles(CB_OL, CB_OH, q * TC + j, q * TC + j, q);
        }
    }
    sec_pack();
    cb_reserve_back(CB_PF, NQ);
    for (uint32_t q = 0; q < NQ; ++q) {
        pk(q, CB_PF);
    }
    cb_push_back(CB_PF, NQ);
    sec_end();
    split_tiles(CB_PF, NQ, CB_PS);
    if (master) {
        finish_b();
        cb_wait_front(CB_ACK, 1);
        cb_pop_front(CB_ACK, 1);
    }

    cb_wait_front(CB_RB, NQ);
    constexpr uint32_t NO = OUTM == 0 ? NOUT : NQ;
    for (uint32_t q = 0; q < NO; ++q) {
        cb_reserve_back(CB_OUT, TC);
        if constexpr (OUTM >= 1) {
            cb_reserve_back(CB_OO, TC);
        }
        for (uint32_t j = 0; j < TC; ++j) {
            sec_begin();
            cp(CB_OH, q * TC + j, 0);
            cp(CB_OL, q * TC + j, 1);
            rows::sfpu_init();
            rows::add(0, 1, 0);
            if constexpr (OUTM >= 1) {
                cp(CB_OH, q * TC + j, 3);
                cp(CB_OL, q * TC + j, 4);
                rows::sfpu_init();
                rows::add(3, 4, 3);
            }
            cp(CB_RB, q, 2);
            rows::mul_bcast_col(0, 2);
            if constexpr (OUTM == 0) {
                cp(CB_GAM, q * TC + j, 1);
                rows::sfpu_init();
                rows::mul(0, 1, 0);
            }
            rows::sfpu_init();
            rows::rne(0);
            sec_pack();
            pk(0, CB_OUT);
            if constexpr (OUTM >= 1) {
                pk(3, CB_OO);
            }
            sec_end();
        }
        cb_push_back(CB_OUT, TC);
        if constexpr (OUTM >= 1) {
            cb_push_back(CB_OO, TC);
        }
    }
    cb_pop_front(CB_RB, NQ);
    cb_pop_front(CB_OH, NQ * TC);
    cb_pop_front(CB_OL, NQ * TC);
}

// GATE: y_k = x - tanh(g Wf) x + tanh(g Wg) m -> CB_OH / CB_OL (hi / lo), k = 0 .. NG-1
ALWI void gate_rows() {
    cb_wait_front(CB_XO, NXO * TC);
    cb_wait_front(CB_MI, NG * TC);
    cb_wait_front(CB_GG, 2 * NG);
    cb_reserve_back(CB_OH, NG * TC);
    cb_reserve_back(CB_OL, NG * TC);
    for (uint32_t k = 0; k < NG; ++k) {
        const uint32_t xs = GSAME ? 0 : k;
        for (uint32_t j = 0; j < TC; ++j) {
            sec_begin();
            mm_begin(CB_GG, CB_WF, 0);
            matmul_tiles(CB_GG, CB_WF, 2 * k, (2 * k) * TC + j, 0);
            matmul_tiles(CB_GG, CB_WF, 2 * k + 1, (2 * k + 1) * TC + j, 0);
            mm_begin(CB_GG, CB_WG, 0);
            matmul_tiles(CB_GG, CB_WG, 2 * k, (2 * k) * TC + j, 1);
            matmul_tiles(CB_GG, CB_WG, 2 * k + 1, (2 * k + 1) * TC + j, 1);
            rows::tanh(0);
            rows::tanh(1);
            cp(CB_XO, xs * TC + j, 2);
            cp(CB_MI, k * TC + j, 3);
            rows::sfpu_init();
            rows::mul(0, 2, 0);
            rows::sub(2, 0, 2);
            rows::mul(1, 3, 1);
            rows::add(2, 1, 2);
            rows::split(2, 0, 1);
            sec_pack();
            pk(0, CB_OH);
            pk(1, CB_OL);
            sec_end();
        }
    }
    cb_push_back(CB_OH, NG * TC);
    cb_push_back(CB_OL, NG * TC);
    cb_pop_front(CB_XO, NXO * TC);
    cb_pop_front(CB_MI, NG * TC);
    cb_pop_front(CB_GG, 2 * NG);
}

// OUTM 2: w_k = column c_k of W (column form), m_k = s1 + w_k d -> CB_MO (FP32, the writer drains it per gate),
// CB_MH / CB_ML (hi / lo)
ALWI void messages() {
    cb_wait_front(CB_W, 2);
    cb_reserve_back(CB_WC, NG);
    for (uint32_t k = 0; k < NG; ++k) {
        const uint32_t ck = get_common_arg_val<uint32_t>(A_GC + k);
        sec_begin();
        cp(CB_W, ck >> 5, 0);
        rows::sfpu_init();
        rows::split(0, 1, 2);
        sec_pack();
        cb_reserve_back(CB_WS, 2);
        pk(1, CB_WS);
        pk(2, CB_WS);
        cb_push_back(CB_WS, 2);
        sec_end();
        cb_wait_front(CB_WS, 2);
        sec_begin();
        mm_begin(CB_WS, CB_CSEL, 0);
        matmul_tiles(CB_WS, CB_CSEL, 0, k, 0);
        matmul_tiles(CB_WS, CB_CSEL, 1, k, 0);
        sec_pack();
        pk(0, CB_WC);
        sec_end();
        cb_pop_front(CB_WS, 2);
    }
    cb_push_back(CB_WC, NG);
    cb_pop_front(CB_W, 2);
    cb_wait_front(CB_WC, NG);
    cb_wait_front(CB_S1, TC);
    cb_wait_front(CB_DD, TC);
    cb_reserve_back(CB_MH, NG * TC);
    cb_reserve_back(CB_ML, NG * TC);
    for (uint32_t k = 0; k < NG; ++k) {
        cb_reserve_back(CB_MO, TC);
        for (uint32_t j = 0; j < TC; ++j) {
            sec_begin();
            cp(CB_DD, j, 0);
            cp(CB_WC, k, 1);
            rows::mul_bcast_col(0, 1);
            cp(CB_S1, j, 1);
            rows::sfpu_init();
            rows::add(0, 1, 0);
            rows::split(0, 1, 2);
            sec_pack();
            pk(0, CB_MO);
            pk(1, CB_MH);
            pk(2, CB_ML);
            sec_end();
        }
        cb_push_back(CB_MO, TC);
    }
    cb_push_back(CB_MH, NG * TC);
    cb_push_back(CB_ML, NG * TC);
    cb_pop_front(CB_WC, NG);
    cb_pop_front(CB_S1, TC);
    cb_pop_front(CB_DD, TC);
    cb_wait_front(CB_MH, NG * TC);
    cb_wait_front(CB_ML, NG * TC);
}

// OUTM 2 after round A: nm_k = bf16(m_k / rms(m_k)) -> CB_NM (per gate)
ALWI void message_norms() {
    cb_wait_front(CB_RAM, NG);
    for (uint32_t k = 0; k < NG; ++k) {
        cb_reserve_back(CB_NM, TC);
        for (uint32_t j = 0; j < TC; ++j) {
            sec_begin();
            cp(CB_MH, k * TC + j, 0);
            cp(CB_ML, k * TC + j, 1);
            rows::sfpu_init();
            rows::add(0, 1, 0);
            cp(CB_RAM, k, 2);
            rows::mul_bcast_col(0, 2);
            rows::sfpu_init();
            rows::rne(0);
            sec_pack();
            pk(0, CB_NM);
            sec_end();
        }
        cb_push_back(CB_NM, TC);
    }
    cb_pop_front(CB_RAM, NG);
    cb_pop_front(CB_MH, NG * TC);
    cb_pop_front(CB_ML, NG * TC);
}
}  // namespace gr

void kernel_main() {
    ink_set_read_vc_rows();
    using namespace gr;
    const uint32_t g = get_arg_val<uint32_t>(P_G), c = get_arg_val<uint32_t>(P_C);
    const bool master = c == 0;
    const uint32_t NR = get_common_arg_val<uint32_t>(A_NR);
    compute_kernel_hw_startup(CB_CONST, CB_CONST, CB_OUT);
    cb_wait_front(CB_CONST, NCONST);
    if constexpr (PRO > 0) {
        cb_wait_front(CB_TAP, 4 * TC);
    }
    if constexpr (NLT > 0) {
        cb_wait_front(CB_Q, 2 * TC);
    }
    if constexpr (!STOP && OUTM == 0) {
        cb_wait_front(CB_GAM, NOUT * TC);
    }
    if constexpr (MSG) {
        cb_wait_front(CB_CSEL, NG);
    }
    if constexpr (GATE) {
        cb_wait_front(CB_WF, 2 * NG * TC);
        cb_wait_front(CB_WG, 2 * NG * TC);
    }

    for (uint32_t n = g; n < NR; n += NGRP) {
        if constexpr (GATE) {
            gate_rows();
            round_b_outputs(master);
            continue;
        }
        // ------------------------------------------------------------ prologue
        uint32_t vh = CB_BASE;
        if constexpr (PRO > 0) {
            cb_wait_front(CB_IN, TC);
            cb_wait_front(CB_PREV, TC);
            if constexpr (NEED_BASE) {
                cb_wait_front(CB_BASE, TC);
            }
            if constexpr (PRO == 1) {
                cb_reserve_back(CB_HV, TC);
            } else {
                cb_reserve_back(CB_RF, TC);
                cb_reserve_back(CB_VH, TC);
                cb_reserve_back(CB_VL, TC);
            }
            for (uint32_t j = 0; j < TC; ++j) {
                sec_begin();
                mm_begin(CB_CONST, CB_IN, 0);
                for (uint32_t s = 0; s < 3; ++s) {
                    matmul_tiles(CB_CONST, CB_IN, K_S + s, j, 1 + s);  // D(s+1) = x_{t-s-1} (in-tile part)
                }
                mm_begin(CB_CONST, CB_PREV, 0);
                for (uint32_t s = 0; s < 3; ++s) {
                    matmul_tiles(CB_CONST, CB_PREV, K_SP + s, j, 1 + s);  // + previous tile's last rows
                }
                cp(CB_IN, j, 0);
                cp(CB_TAP, 4 * j + 3, 4);
                rows::sfpu_init();
                rows::mul(0, 4, 0);
                for (uint32_t s = 3; s >= 1; --s) {  // tap (3 - s) x x_{t-s}, oldest first (as the decode kernel)
                    cp(CB_TAP, 4 * j + 3 - s, 4);
                    rows::sfpu_init();
                    rows::mul(s, 4, s);
                    rows::add(0, s, 0);
                }
                rows::rne(0);
                if constexpr (PRO == 1) {
                    cp(CB_BASE, j, 1);
                    rows::sfpu_init();
                    rows::add(1, 0, 0);
                    rows::rne(0);
                    sec_pack();
                    pk(0, CB_HV);
                } else {
                    if constexpr (!BOUND) {
                        cp(CB_BASE, j, 1);
                        rows::sfpu_init();
                        rows::add(1, 0, 0);
                    }
                    rows::split(0, 1, 2);
                    sec_pack();
                    pk(0, CB_RF);
                    pk(1, CB_VH);
                    pk(2, CB_VL);
                }
                sec_end();
            }
            if constexpr (PRO == 1) {
                cb_push_back(CB_HV, TC);
                vh = CB_HV;
            } else {
                cb_push_back(CB_RF, TC);
                cb_push_back(CB_VH, TC);
                cb_push_back(CB_VL, TC);
                vh = CB_VH;
            }
            cb_pop_front(CB_IN, TC);
            cb_pop_front(CB_PREV, TC);
            if constexpr (NEED_BASE) {
                cb_pop_front(CB_BASE, TC);
            }
        } else {
            cb_wait_front(CB_BASE, TC);
        }
        if constexpr (STOP) {
            continue;  // the writer pops CB_HV
        }
        cb_wait_front(vh, TC);
        if constexpr (HAS_LO) {
            cb_wait_front(CB_VL, TC);
        }
        if constexpr (MSG) {
            messages();
        }

        // ------------------------------------------------------------ round A partials
        if constexpr (NLT > 0 && NS > 0) {
            cb_wait_front(CB_X, NS * TC);
        }
        sec_begin();
        constexpr uint32_t DG_ = NLT;  // Gram accumulator of v; messages: NLT + 1 + k
        if constexpr (NLT > 0) {
            if constexpr (NS > 0) {
                mm_begin(CB_Q, CB_X, 1);
                for (uint32_t i = 0; i < NS; ++i) {
                    for (uint32_t j = 0; j < TC; ++j) {
                        matmul_tiles(CB_Q, CB_X, j, i * TC + j, i);
                        matmul_tiles(CB_Q, CB_X, TC + j, i * TC + j, i);
                    }
                }
            }
            mm_begin(CB_Q, vh, 1);
            for (uint32_t j = 0; j < TC; ++j) {
                matmul_tiles(CB_Q, vh, j, j, NS);
                matmul_tiles(CB_Q, vh, TC + j, j, NS);
            }
            if constexpr (HAS_LO) {
                mm_begin(CB_Q, CB_VL, 1);
                for (uint32_t j = 0; j < TC; ++j) {
                    matmul_tiles(CB_Q, CB_VL, j, j, NS);
                    matmul_tiles(CB_Q, CB_VL, TC + j, j, NS);
                }
            }
        }
        mm_begin(vh, vh, 1);
        for (uint32_t j = 0; j < TC; ++j) {
            matmul_tiles(vh, vh, j, j, DG_);
        }
        if constexpr (HAS_LO) {
            mm_begin(vh, CB_VL, 1);
            for (uint32_t j = 0; j < TC; ++j) {
                matmul_tiles(vh, CB_VL, j, j, DG_);
            }
            mm_begin(CB_VL, vh, 1);
            for (uint32_t j = 0; j < TC; ++j) {
                matmul_tiles(CB_VL, vh, j, j, DG_);
            }
        }
        if constexpr (MSG) {
            mm_begin(CB_MH, CB_MH, 1);
            for (uint32_t k = 0; k < NG; ++k) {
                for (uint32_t j = 0; j < TC; ++j) {
                    matmul_tiles(CB_MH, CB_MH, k * TC + j, k * TC + j, DG_ + 1 + k);
                }
            }
            mm_begin(CB_MH, CB_ML, 1);
            for (uint32_t k = 0; k < NG; ++k) {
                for (uint32_t j = 0; j < TC; ++j) {
                    matmul_tiles(CB_MH, CB_ML, k * TC + j, k * TC + j, DG_ + 1 + k);
                }
            }
            mm_begin(CB_ML, CB_MH, 1);
            for (uint32_t k = 0; k < NG; ++k) {
                for (uint32_t j = 0; j < TC; ++j) {
                    matmul_tiles(CB_ML, CB_MH, k * TC + j, k * TC + j, DG_ + 1 + k);
                }
            }
        }
        sec_pack();
        cb_reserve_back(CB_PF, NPA);
        for (uint32_t u = 0; u < NPA; ++u) {
            pk(u, CB_PF);
        }
        cb_push_back(CB_PF, NPA);
        sec_end();
        split_tiles(CB_PF, NPA, CB_PS);
        if (master) {
            finish_a();
            cb_wait_front(CB_ACK, 1);  // the writer has multicast the result: it may be popped now
            cb_pop_front(CB_ACK, 1);
        }
        if constexpr (MSG) {
            message_norms();
        }

        if constexpr (!MIX) {  // layer 0: out = bf16(v s_v gamma)
            cb_wait_front(CB_RAS, 1);
            cb_reserve_back(CB_OUT, TC);
            for (uint32_t j = 0; j < TC; ++j) {
                sec_begin();
                cp(vh, j, 0);
                cp(CB_RAS, 0, 2);
                rows::mul_bcast_col(0, 2);
                cp(CB_GAM, j, 1);
                rows::sfpu_init();
                rows::mul(0, 1, 0);
                rows::rne(0);
                sec_pack();
                pk(0, CB_OUT);
                sec_end();
            }
            cb_push_back(CB_OUT, TC);
            cb_pop_front(CB_RAS, 1);
            cb_pop_front(vh, TC);
            continue;
        } else {
            // ------------------------------------------------------------ diagonal coefficient tiles
            cb_wait_front(CB_RAC, 2);
            cb_reserve_back(CB_DG, 2 * NSEL);
            for (uint32_t u = 0; u < NSEL; ++u) {
                sec_begin();
                mm_begin(CB_CONST, CB_RAC, 0);
                matmul_tiles(CB_CONST, CB_RAC, K_SEL + u, 0, 0);
                matmul_tiles(CB_CONST, CB_RAC, K_SEL + u, 1, 0);
                cp(CB_CONST, K_I, 1);
                rows::sfpu_init();
                rows::mul(0, 1, 0);
                rows::split(0, 0, 1);
                sec_pack();
                pk(0, CB_DG);
                pk(1, CB_DG);
                sec_end();
            }
            cb_push_back(CB_DG, 2 * NSEL);
            cb_pop_front(CB_RAC, 2);
            cb_wait_front(CB_DG, 2 * NSEL);

            // ------------------------------------------------------------ O_q = sum_i diag(C_iq) X_i + diag(C_vq) v
            cb_reserve_back(CB_OH, NQ * TC);
            cb_reserve_back(CB_OL, NQ * TC);
            for (uint32_t q = 0; q < NQ; ++q) {
                for (uint32_t j = 0; j < TC; ++j) {
                    sec_begin();
                    if constexpr (NS > 0) {
                        mm_begin(CB_DG, CB_X, 0);
                        for (uint32_t i = 0; i < NS; ++i) {
                            const uint32_t u = i * NQ + q;
                            matmul_tiles(CB_DG, CB_X, 2 * u, i * TC + j, 0);
                            matmul_tiles(CB_DG, CB_X, 2 * u + 1, i * TC + j, 0);
                        }
                    }
                    const uint32_t uv = NS * NQ + q;
                    mm_begin(CB_DG, vh, 0);
                    matmul_tiles(CB_DG, vh, 2 * uv, j, 0);
                    matmul_tiles(CB_DG, vh, 2 * uv + 1, j, 0);
                    if constexpr (HAS_LO) {
                        mm_begin(CB_DG, CB_VL, 0);
                        matmul_tiles(CB_DG, CB_VL, 2 * uv, j, 0);
                        matmul_tiles(CB_DG, CB_VL, 2 * uv + 1, j, 0);
                    }
                    rows::sfpu_init();
                    rows::split(0, 0, 1);
                    sec_pack();
                    pk(0, CB_OH);
                    pk(1, CB_OL);
                    sec_end();
                }
            }
            cb_push_back(CB_OH, NQ * TC);
            cb_push_back(CB_OL, NQ * TC);
            cb_pop_front(CB_DG, 2 * NSEL);
            if constexpr (NS > 0) {
                cb_pop_front(CB_X, NS * TC);
            }
            cb_pop_front(vh, TC);
            if constexpr (HAS_LO) {
                cb_pop_front(CB_VL, TC);
            }
            round_b_outputs(master);
        }
    }
}
#endif
