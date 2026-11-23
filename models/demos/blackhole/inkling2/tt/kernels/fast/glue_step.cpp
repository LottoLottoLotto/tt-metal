// SPDX-License-Identifier: Apache-2.0
// Inkling-2-Mini fused decode glue step (one generic_op; Python side and semantics: tt/fast/glue.py).
//
// One program = one of the glue segments of a layer application (row 0 of every [1, 1, 32, 2304] tile tensor is
// the token):
//   prologue  PRO 0: v = h (given, BF16)
//             PRO 1: d = bf16(conv(x) with state), v = h = bf16(r_prev + d)          (post-MLP conv + residual)
//             PRO 2: c = bf16(conv(x) with state), v = r = h + c (BOUND: r = c), FP32 (post-attention conv + residual)
//             PRO 3: e = the raw embedding row; round E: 1/rms(e); v = h = bf16(e / rms(e) * gamma_embed)  (layer 0)
//             (conv: sum_j taps_j * x[t-3+j]; the states are the conv inputs at positions p-3..p-1 = rows
//             (p-3..p-1) % 32 of the position ring (FP32 tile), x is written to ring row p % 32; p = cur_pos[0])
//   NQ == 0 (layer 0): round V: 1/rms(v); X row 0 = bf16(v / rms(v)); out = bf16(v / rms(v) * gamma)
//   NQ > 0  round MX (ONE round): every core sends its partial of P = [q_a . S_i ; S_i . S_j] over the states
//           S = [v (raw), X rows < NS]; the master finishes s_v = 1/rms(v) (from S_0 . S_0), the logits (the partial's
//           column scaled by s_v: the FP32-normed partial), softmax W, the coefficients C (W with s_v folded into
//           the partial's column) and |O_a|^2 = c_a^T G c_a (Gram) -> s_O; it multicasts [s_v, s_O] and C.
//           O = C S (FP32); APPEND: X row NS = bf16(v s_v)
//     NG == 0:  out_a = bf16(O_a s_O_a * gamma_a)
//     NG > 0:   nx = bf16(O s_O); round DN: g_k = bf16(nx_src(k) . Wd_k + nM_k . Wm_k) (nM: the BF16 normed message,
//               Wm: the message half of the down projection); y_k = (1 - tanh(g_k Wf_k)) x_src(k) + tanh(g_k Wg_k) m_k;
//               round SG: 1/rms(y_k); out_k = bf16(y_k / rms(y_k) * gamma_k)
//   MSG     (pass restart, PRO 1 = the previous pass's END: h = bf16(r + conv(x))): messages of the 48 queries over
//           [h, X row NS] (NS = the previous pass's layer-12 block state): round MG (logits, Gram), the master finishes
//           s_h, w = softmax, a_h = w_h s_h, a_x = w_x, s_M = 1/rms(M) (Gram), multicasts them; every core writes
//           M = a_h h + a_x X_NS (FP32, rows 0..47 of a [64, H] tensor) and nM = bf16(M s_M).
//
// Layout: C cores per group (a rectangle; per-core arg k = core index), core k owns the T hidden column tiles
// [k T, k T + T); core 0 is the group's master.  Decode: one group; verify rows: one group per row (instance), each
// an independent copy of the decode program on its own token row; batch rows (MULTI 2): likewise per row of a wave,
// each row its own sequence (position / slot from the batch tables; see the reader / writer for the row handling).
// Cross-core sums ("rounds"): every core packs its partial tile(s) as 3 BF16 parts (x = hi + mid + lo, hi = x with
// the low 16 bits cleared, mid likewise of x - hi: ~2^-25 relative), the writer RISC copies the needed rows into
// row k of the master's gather tiles (one tile per slot; unwritten rows / columns zero-filled) and bumps the
// master's semaphore; the master reduces with exact BF16 matmuls (selector E_r: ones in row r -> result row r;
// SEL7: rows 0..7 -> broadcast),
// finishes the round and multicasts the result tile(s) + a round counter.  Products of BF16 values are exact and
// accumulate in FP32 DEST, so every dot product / norm / mix is FP32-accurate (hi / lo operands: ~2^-17 relative).
//
// Garbage rule: only some rows of the input tiles are read (rows 0-1, row 0); every value derived from them goes
// through row-local ops and is cleaned (rows >= R zeroed) before it can enter a matmul that contracts over rows.
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
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "glue_sfpu.hpp"
#endif
#include "tools/profiler/kernel_profiler.hpp"  // DeviceZoneScopedN (no-op unless the device profiler is on)

namespace gs {
#define NCT(name) get_named_compile_time_arg_val(name)
constexpr uint32_t T = NCT("T");          // column tiles per core
constexpr uint32_t C = NCT("C");          // cores
constexpr uint32_t PRO = NCT("PRO");
constexpr uint32_t BOUND = NCT("BOUND");
constexpr uint32_t NQ = NCT("NQ");
constexpr uint32_t NG = NCT("NG");
constexpr uint32_t GSAME = NCT("GSAME");  // every gate reads O row 0 (sliding) / gate k reads O row k (full)
constexpr uint32_t NOUT = NCT("NOUT");
constexpr uint32_t MSG = NCT("MSG");      // messages program (pass restart)
constexpr uint32_t INVH = NCT("INVH");    // FP32 bits of 1 / hidden
constexpr uint32_t EPS_BITS = NCT("EPS");  // FP32 bits of eps
constexpr uint32_t IN_DRAM = NCT("IN_DRAM");
constexpr uint32_t BASE_F32 = NCT("BASE_F32");
// MULTI: 1 = verify instances (one core group per token row; instance code compiled out of the decode binaries),
// 2 = batch instances (FastGlue.batch: one core group per batch row of a wave, each row its own sequence: position
// cur_pos[row], conv rings = slot[row]'s pages; see the reader), 3 = multi-row batch instances (MR, R > 4: one launch,
// group g runs the rows g, g + G, g + 2 G, .. (G groups, n = nrw() <= MRN rows) PIPELINED: every RISC runs the
// program's segments (between two rounds) for all n rows before the next round, and row q's rounds have their own
// master, core q of the group (gather tiles, finish, result multicast, result semaphore 4 + q), so the n rows' rounds
// overlap; each row's arithmetic is exactly the one-row program's (same sections on the same tile indices: every
// per-row CB holds MRN rows' pages, row q's region is selected per segment; see "MR" below).  The shared inputs
// (constants, queries, gamma, taps, gate weights) are read once.
// BSH: the base is a shared rows tensor (row = the instance's row) instead of a per-instance tensor (row 0)
constexpr bool MULTI = NCT("MULTI") >= 1;
constexpr bool BATCH = NCT("MULTI") >= 2;
constexpr bool MR = NCT("MULTI") == 3;
constexpr uint32_t MRN = NCT("MRN");  // MR: most rows per group (per-row CB capacity); else 1
constexpr bool BSH = NCT("BSH") == 1;
constexpr uint32_t GX = NCT("GX");  // decode: rectangle width (core k = y GX + x)

constexpr uint32_t HT = 72;  // hidden tiles (2304 / 32)
constexpr bool CONV = PRO == 1 || PRO == 2;
constexpr bool MIX = NQ > 0;
constexpr bool HAS_LO = PRO == 2;  // v = r is FP32 (hi / lo); else v is BF16
constexpr bool HAS_E = PRO == 3;
constexpr bool HAS_V = !MIX && !MSG;
constexpr uint32_t NMQ = 48;       // message queries (12 loop layers x q / k / v / mlp)

// CB indices
constexpr uint32_t CB_IN = 0, CB_BASE = 1, CB_NH = 2, CB_ST = 3, CB_TAP = 4, CB_X = 5, CB_Q = 6, CB_GAM = 7,
                   CB_MSG = 8, CB_NM = 9, CB_WD = 10, CB_WF = 11, CB_WG = 12, CB_CONST = 13, CB_G = 14,
                   CB_RESF = 15, CB_RESB = 16, CB_D = 17, CB_HV = 18, CB_RF = 19, CB_VH = 20, CB_VL = 21,
                   CB_PART = 22, CB_XF = 23, CB_OH = 24, CB_OL = 25, CB_NX = 26, CB_GK = 27, CB_OUT = 28,
                   CB_SCR = 29, CB_PH = 30, CB_PL = 31, CB_STG = 32, CB_RSCR = 33, CB_WM = 34, CB_GE = 35,
                   CB_ZQ = 36, CB_ZQL = 37, CB_MF = 38;
constexpr uint32_t CB_ORP = CB_OH, CB_YR = CB_OL;  // row-packed gate inputs x / gate outputs y (hi, lo; BF16)
// CB_D: v lo as column tiles (row 0; CONV programs with an FP32 v, built from the row-packed CB_VL in the MX round)
constexpr uint32_t CB_VLC = CB_D;
// MR: a round's master finishes into its own result CBs (one row: the one-row page indices), its writer multicasts
// them into every core's RESF / RESB pages of that row; else the master's compute packs RESF / RESB directly
constexpr uint32_t CB_MRESF = MR ? 39 : CB_RESF, CB_MRESB = MR ? 40 : CB_RESB;
// constant tiles in CB_CONST (host-built per (NS, NQ), glue.GlueGrid.consts); the master's writer reads all and
// multicasts tiles >= K_CORE (the ones the other cores use) to its group
constexpr uint32_t K_E = 0,      // + r (r < 5): ones in row r -> reduce result row r
    K_J = 5,                     // all ones
    K_SEL7 = 6,                  // rows 0..7 all ones (reduce result broadcast over rows 0..7)
    K_MASK = 7,                  // softmax mask: rows < NQ, columns <= NS (MSG: all rows, columns < 16)
    K_COL0 = 8,                  // rows 0..7, column 0
    K_COLR = 9,                  // rows 0..7, columns 1..NS
    K_CORE = 10,
    K_E00 = 10,                  // [0][0]: row 0 of v
    K_SX = 11,                   // [1 + i][i], i < NS: X rows -> rows 1..NS (XF)
    K_Q = 12,                    // [a][a], a < NQ: the hi query rows
    K_QL = 13,                   // [a][4 + a]: the lo query rows -> rows a
    K_SX3 = 14,                  // [4 + i][i], i < NS: X rows -> rows 4..
    K_E30 = 15,                  // [3][0]: v -> row 3
    K_RM = 16,                   // + k (k < 3): row k ones
    K_ID = 19,                   // identity
    K_MX = 20,                   // [1][NS]: X row NS -> row 1 (MSG)
    K_E0J = 21,                  // + j (j < 4): [0][j]: row j of a row-packed tile -> row 0
    K_E3J = 25,                  // + j (j < 4): [3][j]: row j of a row-packed tile -> row 3
    NCONST = 29;
// gather tiles per slot: core k -> tile k / 32, row k % 32 (rows >= CL of the last tile stay zero)
constexpr uint32_t NGT = (C + 31) / 32, CL = C - 32 * (NGT - 1);

// rounds (in this order when present): ordinals, gather slot bases / counts, partial tile bases, result tiles
constexpr uint32_t O_E = 0, O_V = O_E + HAS_E, O_MX = O_V + HAS_V, O_MG = O_MX + MIX, O_DN = O_MG + MSG,
                   O_SG = O_DN + (NG > 0);
// every partial value travels as NPT BF16 parts (2: hi + lo, ~2^-17 relative; 3: hi + mid + lo, ~2^-25 -- measured
// no better: the outputs' exactness is not limited by the transport); one gather slot / partial tile each
// (compile-time arg, glue.NPT)
constexpr uint32_t NPT = NCT("NPT");
constexpr uint32_t MXR = NQ + 5;  // MX slot rows: NQ logit rows + up to 5 Gram rows (NS <= 4)
constexpr uint32_t G_E = 0, N_E = HAS_E ? NPT : 0;
constexpr uint32_t G_V = G_E + N_E, N_V = HAS_V ? NPT : 0;
constexpr uint32_t G_MX = G_V + N_V, N_MX = MIX ? NPT * MXR : 0;
constexpr uint32_t G_MG = G_MX + N_MX, N_MG = MSG ? 7 * NPT : 0;
constexpr uint32_t G_DN = G_MG + N_MG, N_DN = 2 * NPT * NG;
constexpr uint32_t G_SG = G_DN + N_DN, N_SG = NPT * NG;
constexpr uint32_t N_GSLOT = G_SG + N_SG;
constexpr uint32_t P_E = 0, P_V = P_E + (HAS_E ? NPT : 0), P_MX = P_V + (HAS_V ? NPT : 0),
                   P_MG = P_MX + (MIX ? NPT : 0), P_DN = P_MG + (MSG ? 3 * NPT : 0), P_SG = P_DN + 2 * NPT * NG;
// RESF (FP32): E 0, V 0 | 1 (after E); MX: [s_v | 1] 0, s_O 1; SG 2.  RESB (BF16): MX C hi / lo 0..1, DN 2..3;
// MSG: per query tile t: Bc hi / lo, Bs hi / lo at 4 t..
constexpr uint32_t R_E = 0, R_V = HAS_E ? 1 : 0, R_M = 0, R_SO = 1, R_SG = 2, R_C = 0, R_DN = 2;
// rounds per row
constexpr uint32_t NRND = O_SG + (NG > 0);
// round kinds and, per kind, the one-row result pages (first, count) in RESF / RESB and partial pages in CB_PART;
// MR: every round-indexed CB (RESF / RESB results, CB_PART) holds the round's block of all n rows round-major: row q's
// copy of one-row page f + i of a round with block (f, c) is page n f + q c + i
enum : uint32_t { RK_E, RK_V, RK_MX, RK_MG, RK_DN, RK_SG };
constexpr uint32_t round_kind(uint32_t o) {
    return (HAS_E && o == O_E)    ? RK_E
           : (HAS_V && o == O_V)  ? RK_V
           : (MIX && o == O_MX)   ? RK_MX
           : (MSG && o == O_MG)   ? RK_MG
           : (NG > 0 && o == O_DN) ? RK_DN
                                   : RK_SG;
}
struct Blk {
    uint32_t f, c;
};
constexpr Blk resf_blk(uint32_t kind) {
    return kind == RK_E ? Blk{R_E, 1} : kind == RK_V ? Blk{R_V, 1} : kind == RK_MX ? Blk{R_M, 2}
           : kind == RK_SG ? Blk{R_SG, 1} : Blk{0, 0};
}
constexpr Blk resb_blk(uint32_t kind) {
    return kind == RK_MX ? Blk{R_C, 2} : kind == RK_MG ? Blk{0, 8} : kind == RK_DN ? Blk{R_DN, 2} : Blk{0, 0};
}
constexpr Blk part_blk(uint32_t kind) {
    return kind == RK_E ? Blk{P_E, NPT} : kind == RK_V ? Blk{P_V, NPT} : kind == RK_MX ? Blk{P_MX, NPT}
           : kind == RK_MG ? Blk{P_MG, 3 * NPT} : kind == RK_DN ? Blk{P_DN, 2 * NPT * NG} : Blk{P_SG, NPT};
}
// MR: pages one row pushes into the per-row CBs (row q's region starts at page q P; glue.py sizes them MRN P)
constexpr bool NEED_BASE = PRO == 0 || PRO == 1 || PRO == 3 || (PRO == 2 && !BOUND);
constexpr uint32_t PR_IN = CONV ? 1 : 0, PR_BASE = NEED_BASE ? (CONV ? 1 : T) : 0, PR_X = (MIX || MSG) ? T : 0,
                   PR_MSG = NG > 0 ? 1 : 0, PR_NM = NG * T, PR_HV = HAS_E ? T : (PRO == 1 ? 1 : 0),
                   PR_V = PRO == 2 ? 1 : 0,  // VH, VL, RF
    PR_XF = (MIX || MSG) ? T : 0, PR_Z = MIX ? T : 0, PR_VLC = (MIX && HAS_LO) ? T : 0, PR_NX = NG > 0 ? T : 0,
                   PR_GY = NG > 0 ? 2 : 0,  // ORP, YR
    PR_GK = 2 * NG, PR_SCR = MSG ? 4 : 0, PR_RESF = MSG ? 2 : 0,  // MSG: compute-made RESF / SCR pages
    PR_NH = CONV ? 1 : T, PR_OUT = MSG ? 2 * T : (MIX ? 1 : T), PR_MF = MSG ? 2 * T : 0;

// columns written per gather slot (the rest is zero-filled)
constexpr uint32_t slot_cols(uint32_t s) {
    if (s >= G_DN && s < G_SG) {
        return 32;
    }
    if (s >= G_MG && s < G_MG + N_MG) {
        const uint32_t i = (s - G_MG) / NPT;
        return (i == 2 || i == 3) ? 16 : 32;
    }
    return 16;
}

// common runtime args (shared by every core); TAPO: != 0: h (PRO 1) is also written to row `inst` of this shared rows
// tensor; PTAP: the conv taps packed per core (MULTI); BATCH only: ROW0 = the wave's first batch row (instance r =
// row ROW0 + r), SLOT = the batch tables' slot words (int32 [1, 32]: word b = physical slot of row b; A_POS is then
// the cur_pos words: word b = row b's position)
constexpr uint32_t A_IN = 0, A_BASE = 1, A_RING = 2, A_POS = 3, A_TAP = 5, A_X = 9, A_Q = 10, A_GAM = 11, A_M = 12,
                   A_NM = 13, A_WD = 14, A_WF = 17, A_WG = 20, A_OH = 23, A_OUT = 24, A_NS = 27, A_APPEND = 28,
                   A_MC = 29, A_CONST = 32, A_WM = 33, A_GE = 36, A_MO = 37, A_NMO = 38, A_TAPO = 39, A_PTAP = 40,
                   A_ROW0 = 41, A_SLOT = 42;
// MULTI per-core runtime args: core index k within its group, instance (token row), the group's master / multicast
// rectangle (NOC), the per-instance tensor addresses (override the common args BASE, X, M, NM, OH, MO, NMO), then the
// siblings (core k of the other instances' groups): count, NOC x / y pairs; MR: then the group's row count NRW, the
// per-instance addresses of its rows q = 1 .. MRN - 1 (7 each; row q = instance + G q; padded), and the NOC x / y of
// row q's master (core q of the group), q < MRN.  Decode: no per-core args (k from the logical coordinates, the
// master / rectangle compile-time)
constexpr uint32_t PC_K = 0, PC_INST = 1, PC_MX = 2, PC_MY = 3, PC_MC = 4, PC_OVR = 8, PC_SIB = 15, N_OVR = 7;

constexpr uint32_t BF = 2048, FP = 4096;  // tile bytes

FORCE_INLINE uint32_t core_index() {
    if constexpr (MULTI) {
        return get_arg_val<uint32_t>(PC_K);
    } else {
        return get_absolute_logical_y() * GX + get_absolute_logical_x();
    }
}
FORCE_INLINE uint32_t instance() {
    if constexpr (MULTI) {
        return get_arg_val<uint32_t>(PC_INST);
    } else {
        return 0;
    }
}
FORCE_INLINE uint32_t nsib() {
    if constexpr (MULTI) {
        return get_arg_val<uint32_t>(PC_SIB);
    } else {
        return 0;
    }
}
// rows of this core's group (MR: per-core arg after the siblings; else 1)
FORCE_INLINE uint32_t nrw() {
    if constexpr (MR) {
        return get_arg_val<uint32_t>(PC_SIB + 1 + 2 * nsib());
    } else {
        return 1;
    }
}
}  // namespace gs

// ====================================================================================================== RISCs
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace gs {
// master core and multicast rectangle of this core's group (NOC coordinates)
FORCE_INLINE uint32_t MXc() {
    if constexpr (MULTI) {
        return get_arg_val<uint32_t>(PC_MX);
    } else {
        return NCT("MX");
    }
}
FORCE_INLINE uint32_t MYc() {
    if constexpr (MULTI) {
        return get_arg_val<uint32_t>(PC_MY);
    } else {
        return NCT("MY");
    }
}
FORCE_INLINE uint32_t MC(uint32_t i) {
    if constexpr (MULTI) {
        return get_arg_val<uint32_t>(PC_MC + i);
    } else {
        return i == 0 ? NCT("MCX0") : i == 1 ? NCT("MCY0") : i == 2 ? NCT("MCX1") : NCT("MCY1");
    }
}

// MR: the row of its group this RISC is working on (q = 0 .. nrw() - 1)
uint32_t g_q = 0;
// first per-core arg of the current row's per-instance addresses
FORCE_INLINE uint32_t ovr0() {
    if constexpr (MR) {
        if (g_q) {
            return PC_SIB + 2 + 2 * nsib() + N_OVR * (g_q - 1);
        }
    }
    return PC_OVR;
}

FORCE_INLINE uint32_t arg(uint32_t i) {
    if constexpr (MULTI) {
        switch (i) {
            case A_BASE: return get_arg_val<uint32_t>(ovr0() + 0);
            case A_X: return get_arg_val<uint32_t>(ovr0() + 1);
            case A_M: return get_arg_val<uint32_t>(ovr0() + 2);
            case A_NM: return get_arg_val<uint32_t>(ovr0() + 3);
            case A_OH: return get_arg_val<uint32_t>(ovr0() + 4);
            case A_MO: return get_arg_val<uint32_t>(ovr0() + 5);
            case A_NMO: return get_arg_val<uint32_t>(ovr0() + 6);
            default: break;
        }
    }
    return get_common_arg_val<uint32_t>(i);
}
// the instance's row of the shared rows tensors (decode 0, verify instance r: row r, batch: ROW0 + r, MR: row q of
// group r = r + G q; < 16)
FORCE_INLINE uint32_t row_of() {
    if constexpr (MR) {
        return get_arg_val<uint32_t>(PC_INST) + get_common_arg_val<uint32_t>(A_ROW0) + (nsib() + 1) * g_q;
    } else if constexpr (BATCH) {
        return get_arg_val<uint32_t>(PC_INST) + get_common_arg_val<uint32_t>(A_ROW0);
    } else {
        return instance();
    }
}

// MR: a CB's first page (L1 bytes; the same on every core of the program); nothing is ever popped, so CB page i
// (counted from the program start) is at cb_base + i * page
FORCE_INLINE uint32_t cb_base(uint32_t cb) {
    auto& c = get_local_cb_interface(cb);
    return c.fifo_limit - c.fifo_size;
}
// the current row's P pages of a per-row CB (consumer side): wait for them, their L1 address
FORCE_INLINE uint32_t front(uint32_t cb, uint32_t P, uint32_t page) {
    if constexpr (MR) {
        cb_wait_front(cb, (g_q + 1) * P);
        return cb_base(cb) + g_q * P * page;
    } else {
        cb_wait_front(cb, P);
        return get_read_ptr(cb);
    }
}
// NOC x / y of the master of the current row's rounds (MR: core g_q of the group)
FORCE_INLINE uint32_t pc_masters() { return PC_SIB + 2 + 2 * nsib() + N_OVR * (MRN - 1); }
FORCE_INLINE uint32_t RMX() {
    if constexpr (MR) {
        return get_arg_val<uint32_t>(pc_masters() + 2 * g_q);
    } else {
        return MXc();
    }
}
FORCE_INLINE uint32_t RMY() {
    if constexpr (MR) {
        return get_arg_val<uint32_t>(pc_masters() + 2 * g_q + 1);
    } else {
        return MYc();
    }
}

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

FORCE_INLINE void copy_words(uint32_t dst, uint32_t src, uint32_t words) {
    auto* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    auto* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < words; ++i) {
        d[i] = s[i];
    }
}

// byte offset of row r, column half f of an FP32 tile
FORCE_INLINE uint32_t f32_row(uint32_t r, uint32_t f) { return ((r >> 4) * 2 + f) * 1024 + (r & 15) * 64; }

// mcast rectangle (this core's group) in this RISC's NOC orientation
FORCE_INLINE uint64_t mcast_addr(uint32_t addr) {
    if (noc_index == 1) {
        return get_noc_multicast_addr(MC(2), MC(3), MC(0), MC(1), addr);
    }
    return get_noc_multicast_addr(MC(0), MC(1), MC(2), MC(3), addr);
}
}  // namespace gs
#endif

#if defined(COMPILE_FOR_NCRISC)
// ------------------------------------------------------------------------------------------------ reader
namespace gs {
// CB_RSCR: the cur_pos words at RS_POS, BATCH: the slot words at RS_SLOT; MR: the input rows' faces (the group's rows
// are < 16: faces 0 / 1 of every column tile; (j, f) at RS_IN + (2 j + f) 512) | per-row staging blocks at RS_Q (MR:
// row q at RS_Q + q RS_QSZ): x row pairs (j: 128 B) | base row pairs (j: 128 B) | lag rows (verify, input-row pairs:
// (j, s) 128 B; BATCH, ring rows: (j, f, s) 64 B) | BATCH message rows (consecutive gate rows: (j, f) 192 B) |
// normed message row pairs ((kk, j, f) 64 B; consecutive gate rows: (j, f) 128 B)
constexpr uint32_t RS_POS = 0, RS_SLOT = 64, RS_IN = 128, RS_Q = RS_IN + (BATCH ? 1024 * T : 0), RS_QSZ = 1408 * T;
constexpr uint32_t RQ_X = 0, RQ_B = 128 * T, RQ_L = 256 * T, RQ_M = 640 * T, RQ_N = 1024 * T;
// BATCH: combined reads (a row's 3 conv lags of a tile half in one 192 B read when they share a face; the 3 gates'
// consecutive message rows in one read): fewer DRAM transactions, which bound the reads of 4 groups x rows
constexpr bool COMB = BATCH;
constexpr bool NEED_BASE_RD = NEED_BASE;
constexpr uint32_t BASE_PAGE = BASE_F32 ? FP : BF;
// the current row's staging block / write address of its pages of a per-row CB (nothing is popped: page g_q P)
FORCE_INLINE uint32_t stq() { return get_write_ptr(CB_RSCR) + RS_Q + (MR ? g_q * RS_QSZ : 0); }
FORCE_INLINE uint32_t cbw(uint32_t cb, uint32_t P, uint32_t page) { return cb_base(cb) + (MR ? g_q * P * page : 0); }
FORCE_INLINE void push(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    cb_push_back(cb, n);
}
// the 3 gates read consecutive message rows c0 .. c0 + 2 (c0 % 4 == 0: one face, 128 B-aligned BF16 row quad)
FORCE_INLINE bool msg_run() {
    if constexpr (COMB && NG == 3) {
        const uint32_t c0 = arg(A_MC);
        return (c0 & 3) == 0 && arg(A_MC + 1) == c0 + 1 && arg(A_MC + 2) == c0 + 2;
    } else {
        return false;
    }
}
// BF16 row (16 values at src) -> FP32 (16 words at dst)
FORCE_INLINE void widen(uint32_t dst, uint32_t src) {
    auto* x16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
    auto* x32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t c = 0; c < 16; ++c) {
        x32[c] = static_cast<uint32_t>(x16[c]) << 16;
    }
}
// position / ring slot page of the current row (CONV; the words are in RS_POS / RS_SLOT)
FORCE_INLINE void row_pos(uint32_t& p, uint32_t& sb) {
    const uint32_t rscr = get_write_ptr(CB_RSCR);
    if constexpr (BATCH) {
        const uint32_t row = row_of();
        p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rscr + RS_POS)[row];
        sb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rscr + RS_SLOT)[row] * HT;
    } else {
        p = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rscr + RS_POS) + instance();
        sb = 0;
    }
}
// BATCH verify groups (batched speculation): rows of one sequence at consecutive positions in consecutive rows.  The
// row is a replica of base row br (the first row with its slot and position: pad rows replicate one); br's lags at
// positions of the earlier rows of its group (returned count, <= 3) are rows br-1, br-2, .. of the input rows tensor
// (the ring gets them in this launch, concurrently): BF16 -> FP32, exactly what those rows write into the ring.
FORCE_INLINE uint32_t group_lag(uint32_t& br) {
    const uint32_t rscr = get_write_ptr(CB_RSCR), row = row_of();
    volatile tt_l1_ptr int32_t* P = reinterpret_cast<volatile tt_l1_ptr int32_t*>(rscr + RS_POS);
    volatile tt_l1_ptr int32_t* S = reinterpret_cast<volatile tt_l1_ptr int32_t*>(rscr + RS_SLOT);
    br = row;
    for (uint32_t b = 0; b < row; ++b) {
        if (S[b] == S[row] && P[b] == P[row]) {
            br = b;
            break;
        }
    }
    uint32_t n = 0;
    while (n < 3 && br > n && S[br - n - 1] == S[br] && P[br - n - 1] + 1 + static_cast<int32_t>(n) == P[br]) {
        ++n;
    }
    return n;
}

// The current row's reads, in three groups without barriers: first = the group's first row, which also reads the
// shared inputs (taps, embed gamma, queries, gamma).  Instance r (verify row r): the token is row r (< 16) of the
// shared input / output rows tensors at p = cur_pos[0] + r; batch row (BATCH): its own sequence, p = cur_pos[row].
// X: the prologue inputs that do not depend on the position: the x / base rows (CONV), the base, the embed gamma
FORCE_INLINE void reads_x(const bool first) {
    DeviceZoneScopedN("n-rx");
    const uint32_t k = core_index(), j0 = k * T, st = stq(), row = row_of();
    const uint32_t rpair = (row & ~1u) * 32;  // byte offset of the 64 B row pair containing row `row` in a face
    if constexpr (CONV) {
        const auto in = accessor<IN_DRAM == 1>(arg(A_IN), BF);
        if (first) {
            noc_async_read_tile(k, accessor<true>(arg(A_PTAP), FP), cb_base(CB_TAP));
        }
        for (uint32_t j = 0; j < T; ++j) {
            const uint64_t src = get_noc_addr(j0 + j, in);
            if constexpr (MR) {  // faces 0 / 1 (rows 0..15: every row of the group), once
                if (first) {
                    const uint32_t w = get_write_ptr(CB_RSCR) + RS_IN + j * 1024;
                    noc_async_read(src, w, 512);
                    noc_async_read(src + 512, w + 512, 512);
                }
            } else {
                noc_async_read(src + rpair, st + RQ_X + j * 128, 64);
                noc_async_read(src + 512 + rpair, st + RQ_X + j * 128 + 64, 64);
            }
        }
    }
    if constexpr (NEED_BASE_RD) {
        const auto b = accessor<true>(arg(A_BASE), BASE_PAGE);
        const uint32_t w = cbw(CB_BASE, PR_BASE, BASE_PAGE);
        if constexpr (CONV && BASE_F32) {  // r: FP32 row 0 -> row j
            for (uint32_t j = 0; j < T; ++j) {
                const uint64_t a = get_noc_addr(j0 + j, b);
                noc_async_read(a, w + f32_row(j, 0), 64);
                noc_async_read(a + 1024, w + f32_row(j, 1), 64);
            }
        } else if (CONV || BSH) {  // BF16 row 0 (shared rows tensor: row `row`): row pairs staged
            if (!CONV) {
                zero_fill(w, T * BASE_PAGE);
            }
            const uint32_t rp = BSH ? rpair : 0;
            for (uint32_t j = 0; j < T; ++j) {
                const uint64_t a = get_noc_addr(j0 + j, b);
                noc_async_read(a + rp, st + RQ_B + j * 128, 64);
                noc_async_read(a + 512 + rp, st + RQ_B + j * 128 + 64, 64);
            }
        } else {
            for (uint32_t j = 0; j < T; ++j) {
                noc_async_read_tile(j0 + j, b, w + j * BASE_PAGE);
            }
        }
    }
    if (HAS_E && first) {
        const auto g = accessor<true>(arg(A_GE), BF);
        for (uint32_t j = 0; j < T; ++j) {
            noc_async_read_tile(j0 + j, g, cb_base(CB_GE) + j * BF);
        }
    }
}

// M: the block states, queries, gamma, messages (FP32 rows) and normed messages (BF16 rows)
FORCE_INLINE void reads_m(const bool first) {
    DeviceZoneScopedN("n-rm");
    const uint32_t k = core_index(), j0 = k * T, st = stq();
    constexpr uint32_t NQT = MSG ? 4 : 1;  // query tiles per column tile (MSG: hi 0..31, hi 32..47, lo, lo)
    if constexpr (MIX || MSG) {
        const auto x = accessor<true>(arg(A_X), BF);
        const uint32_t wx = cbw(CB_X, T, BF);
        for (uint32_t j = 0; j < T; ++j) {
            noc_async_read_tile(j0 + j, x, wx + j * BF);
        }
        if (first) {
            const auto q = accessor<true>(arg(A_Q), BF);
            for (uint32_t j = 0; j < T; ++j) {
                for (uint32_t tr = 0; tr < NQT; ++tr) {
                    noc_async_read_tile(tr * HT + j0 + j, q, cb_base(CB_Q) + (tr * T + j) * BF);
                }
            }
        }
    }
    if (NOUT > 0 && first) {  // MIX: the core's row-packed gamma tile (rows 4 a + j); layer 0: column tiles
        const auto g = accessor<true>(arg(A_GAM), BF);
        if constexpr (MIX) {
            noc_async_read_tile(k, g, cb_base(CB_GAM));
        } else {
            for (uint32_t j = 0; j < T; ++j) {
                noc_async_read_tile(j0 + j, g, cb_base(CB_GAM) + j * BF);
            }
        }
    }
    if constexpr (NG > 0) {
        // message m_k (FP32) -> CB_MSG (row-packed) row 4 k + j; normed message nM_k (BF16, via 64 B-aligned row
        // pairs) -> row src(k) of CB_NM tile k T + j (consecutive gate rows: one read per tile half each, staged)
        const auto m = accessor<true>(arg(A_M), FP);
        const auto nm = accessor<true>(arg(A_NM), BF);
        const uint32_t wm = cbw(CB_MSG, 1, FP);
        if (msg_run()) {
            const uint32_t c = arg(A_MC), rr = c & 31;
            for (uint32_t j = 0; j < T; ++j) {
                const uint64_t a = get_noc_addr((c >> 5) * HT + j0 + j, m);
                const uint64_t b = get_noc_addr((c >> 5) * HT + j0 + j, nm);
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(a + f32_row(rr, f), st + RQ_M + (j * 2 + f) * 192, 192);
                    noc_async_read(b + ((rr >> 4) * 2 + f) * 512 + (rr & 15) * 32, st + RQ_N + (j * 2 + f) * 128, 128);
                }
            }
        } else {
            for (uint32_t kk = 0; kk < NG; ++kk) {
                const uint32_t c = arg(A_MC + kk), rr = c & 31;
                for (uint32_t j = 0; j < T; ++j) {
                    const uint64_t a = get_noc_addr((c >> 5) * HT + j0 + j, m);
                    const uint64_t b = get_noc_addr((c >> 5) * HT + j0 + j, nm);
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_read(a + f32_row(rr, f), wm + f32_row(4 * kk + j, f), 64);
                        const uint32_t off = ((rr >> 4) * 2 + f) * 512 + ((rr & 15) & ~1u) * 32;
                        noc_async_read(b + off, st + RQ_N + ((kk * T + j) * 2 + f) * 64, 64);
                    }
                }
            }
        }
    }
}

// S: the conv states (CONV; lags: the inputs at positions p-3..p-1: ring rows (p-3..p-1) % 32 of slot sb's pages, or
// (positions >= cur_pos[0], verify instances) row inst-3+s of the input rows tensor (BF16 -> FP32: exactly the value
// its own instance writes into the ring); batch rows: every lag from the ring, verify groups (group_lag) then
// overwrite the in-group lags from the input rows' faces at RS_IN (MR: read by the group's first row; else here)
FORCE_INLINE void reads_s(const uint32_t p, const uint32_t sb) {
    DeviceZoneScopedN("n-rs");
    const uint32_t j0 = core_index() * T, st = stq();
    const uint32_t lag = BATCH ? 0 : instance();  // lags taken from the input rows (verify: positions >= cur_pos[0])
    if constexpr (CONV && BATCH && !MR) {
        uint32_t br;
        if (group_lag(br) > 0) {
            const auto in = accessor<IN_DRAM == 1>(arg(A_IN), BF);
            const uint32_t w = get_write_ptr(CB_RSCR) + RS_IN;
            for (uint32_t j = 0; j < T; ++j) {
                const uint64_t src = get_noc_addr(j0 + j, in);
                noc_async_read(src, w + j * 1024, 512);
                noc_async_read(src + 512, w + j * 1024 + 512, 512);
            }
        }
    }
    if constexpr (CONV) {
        const auto in = accessor<IN_DRAM == 1>(arg(A_IN), BF);
        const auto ring = accessor<true>(arg(A_RING), FP);
        const uint32_t w_st = cbw(CB_ST, 1, FP), r0 = (p + 29) & 31;
        for (uint32_t j = 0; j < T; ++j) {
            const uint64_t a = get_noc_addr(sb + j0 + j, ring);
            if (COMB && ((r0 & 15) <= 13)) {  // the 3 lag rows of a half: one face, consecutive
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(a + f32_row(r0, f), st + RQ_L + (j * 2 + f) * 192, 192);
                }
                continue;
            }
            for (uint32_t s = 0; s < 3; ++s) {
                if (lag + s >= 3) {
                    const uint32_t q = lag + s - 3;
                    const uint64_t src = get_noc_addr(j0 + j, in) + (q & ~1u) * 32;
                    noc_async_read(src, st + RQ_L + (j * 3 + s) * 128, 64);
                    noc_async_read(src + 512, st + RQ_L + (j * 3 + s) * 128 + 64, 64);
                    continue;
                }
                const uint32_t rr = (p + 29 + s) & 31;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(a + f32_row(rr, f),
                                   COMB ? st + RQ_L + ((j * 2 + f) * 3 + s) * 64 : w_st + f32_row(4 * s + j, f), 64);
                }
            }
        }
    }
}

// After the row's reads landed: staged rows -> CB pages (BF16 copies, FP32 widening, local NOC copies of the
// combined reads), pushes, x -> its ring row p % 32.  X: the prologue inputs (x, conv states, base, taps, embed gamma)
FORCE_INLINE void finish_x(const bool first, const uint32_t p, const uint32_t sb) {
    DeviceZoneScopedN("n-fx");
    const uint32_t j0 = core_index() * T, st = stq(), row = row_of();
    const uint32_t lag = BATCH ? 0 : instance();
    const uint32_t w_in = cbw(CB_IN, 1, FP), w_st = cbw(CB_ST, 1, FP);
    if constexpr (CONV && COMB) {  // ring lag rows -> rows 4 s + j (local copies: the NOC moves them meanwhile)
        const uint32_t r0 = (p + 29) & 31;
        for (uint32_t j = 0; j < T; ++j) {
            for (uint32_t f = 0; f < 2; ++f) {
                for (uint32_t s = 0; s < 3; ++s) {
                    const uint32_t src = (r0 & 15) <= 13 ? st + RQ_L + (j * 2 + f) * 192 + s * 64
                                                         : st + RQ_L + ((j * 2 + f) * 3 + s) * 64;
                    noc_async_read(get_noc_addr(src), w_st + f32_row(4 * s + j, f), 64);
                }
            }
        }
    }
    if constexpr (NEED_BASE_RD) {
        const uint32_t w = cbw(CB_BASE, PR_BASE, BASE_PAGE);
        if constexpr (CONV && !BASE_F32) {  // BF16 row (row & 1 of the pair) -> row j
            for (uint32_t j = 0; j < T; ++j) {
                for (uint32_t f = 0; f < 2; ++f) {
                    copy_words(w + f * 512 + j * 32, st + RQ_B + j * 128 + f * 64 + (BSH ? (row & 1) * 32 : 0), 8);
                }
            }
        } else if (!CONV && BSH) {  // row `row` -> row 0
            for (uint32_t j = 0; j < T; ++j) {
                for (uint32_t f = 0; f < 2; ++f) {
                    copy_words(w + j * BF + f * 512, st + RQ_B + j * 128 + f * 64 + (row & 1) * 32, 8);
                }
            }
        }
        push(CB_BASE, PR_BASE);
    }
    if (HAS_E && first) {
        push(CB_GE, T);
    }
    if constexpr (CONV) {
        // x: BF16 row -> FP32 row j; verify lag rows likewise -> rows 4 s + j
        for (uint32_t j = 0; j < T; ++j) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t src = MR ? get_write_ptr(CB_RSCR) + RS_IN + j * 1024 + f * 512 + row * 32
                                        : st + RQ_X + j * 128 + f * 64 + (row & 1) * 32;
                widen(w_in + f32_row(j, f), src);
            }
        }
        push(CB_IN, 1);
        for (uint32_t s = 3 - (lag < 3 ? lag : 3); s < 3; ++s) {
            const uint32_t q = lag + s - 3;
            for (uint32_t j = 0; j < T; ++j) {
                for (uint32_t f = 0; f < 2; ++f) {
                    widen(w_st + f32_row(4 * s + j, f), st + RQ_L + (j * 3 + s) * 128 + f * 64 + (q & 1) * 32);
                }
            }
        }
        if (first) {
            push(CB_TAP, 1);
        }
        if constexpr (COMB) {
            noc_async_read_barrier();  // the local lag copies
        }
        if constexpr (BATCH) {  // verify groups: the in-group lags from the input rows (over the ring's rows)
            uint32_t br;
            const uint32_t gl = group_lag(br);
            for (uint32_t s = 3 - gl; s < 3; ++s) {
                const uint32_t q = br + s - 3;
                for (uint32_t j = 0; j < T; ++j) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        widen(w_st + f32_row(4 * s + j, f), get_write_ptr(CB_RSCR) + RS_IN + j * 1024 + f * 512 + q * 32);
                    }
                }
            }
        }
        push(CB_ST, 1);
        // x (FP32) -> ring row p % 32
        const auto ring = accessor<true>(arg(A_RING), FP);
        const uint32_t rr = p & 31;
        for (uint32_t j = 0; j < T; ++j) {
            const uint64_t a = get_noc_addr(sb + j0 + j, ring);
            noc_async_write(w_in + f32_row(j, 0), a + f32_row(rr, 0), 64);
            noc_async_write(w_in + f32_row(j, 1), a + f32_row(rr, 1), 64);
        }
    }
}

// M: the block states, queries, gamma, messages, normed messages
FORCE_INLINE void finish_m(const bool first) {
    DeviceZoneScopedN("n-fm");
    const uint32_t st = stq();
    if constexpr (MIX || MSG) {
        push(CB_X, T);
        if (first) {
            push(CB_Q, (MSG ? 4 : 1) * T);
        }
    }
    if (NOUT > 0 && first) {
        push(CB_GAM, MIX ? 1 : T);
    }
    if constexpr (NG > 0) {
        const uint32_t wm = cbw(CB_MSG, 1, FP), wnm = cbw(CB_NM, PR_NM, BF);
        const bool run = msg_run();
        if (run) {  // message rows c0 + kk -> rows 4 kk + j (local copies)
            for (uint32_t j = 0; j < T; ++j) {
                for (uint32_t f = 0; f < 2; ++f) {
                    for (uint32_t kk = 0; kk < NG; ++kk) {
                        noc_async_read(get_noc_addr(st + RQ_M + (j * 2 + f) * 192 + kk * 64),
                                       wm + f32_row(4 * kk + j, f), 64);
                    }
                }
            }
        }
        for (uint32_t kk = 0; kk < NG; ++kk) {
            const uint32_t rr = arg(A_MC + kk) & 31, src = GSAME ? 0 : kk;
            for (uint32_t j = 0; j < T; ++j) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t s = run ? st + RQ_N + (j * 2 + f) * 128 + kk * 32
                                           : st + RQ_N + ((kk * T + j) * 2 + f) * 64 + (rr & 1) * 32;
                    copy_words(wnm + (kk * T + j) * BF + f * 512 + src * 32, s, 8);
                }
            }
        }
        if (run) {
            noc_async_read_barrier();
        }
        push(CB_MSG, 1);
        push(CB_NM, NG * T);
    }
}
}  // namespace gs

// ---- batch 3: gate weights (full tiles; read once, after the first row's inputs: MR after every row's, which the
// rows' first segments need earlier).  Verify instances (R > 1): the R cores k of the groups share the reads:
// instance r reads tiles t % R == r and writes them to the same L1 address on its siblings (identical CB layout),
// then bumps their semaphore 3; each core waits for R - 1 bumps
FORCE_INLINE void gate_weights() {
    using namespace gs;
    const uint32_t j0 = core_index() * T, inst = instance();
    {
        DeviceZoneScopedN("n-b3");
        cb_reserve_back(CB_WD, NG * T * 2);
        cb_reserve_back(CB_WM, NG * T * 2);
        cb_reserve_back(CB_WF, NG * 2 * T);
        cb_reserve_back(CB_WG, NG * 2 * T);
        const uint32_t wd = get_write_ptr(CB_WD), wdm = get_write_ptr(CB_WM), wf = get_write_ptr(CB_WF),
                       wgg = get_write_ptr(CB_WG);
        const uint32_t nsib = gs::nsib(), R = nsib + 1;
        uint32_t t = 0;
        auto fetch = [&](uint32_t page, const auto& acc, uint32_t dst) {
            if (t++ % R == inst) {
                noc_async_read_tile(page, acc, dst);
            }
        };
        for (uint32_t kk = 0; kk < NG; ++kk) {
            const auto d = accessor<true>(arg(A_WD + kk), BF);
            const auto dm = accessor<true>(arg(A_WM + kk), BF);
            const auto f = accessor<true>(arg(A_WF + kk), BF);
            const auto g = accessor<true>(arg(A_WG + kk), BF);
            for (uint32_t j = 0; j < T; ++j) {
                for (uint32_t c2 = 0; c2 < 2; ++c2) {
                    fetch((j0 + j) * 2 + c2, d, wd + ((kk * T + j) * 2 + c2) * BF);
                    fetch((j0 + j) * 2 + c2, dm, wdm + ((kk * T + j) * 2 + c2) * BF);
                    fetch(c2 * HT + j0 + j, f, wf + ((kk * 2 + c2) * T + j) * BF);
                    fetch(c2 * HT + j0 + j, g, wgg + ((kk * 2 + c2) * T + j) * BF);
                }
            }
        }
        noc_async_read_barrier();
        if (nsib) {
            // forward this core's share (the same tile order as above) to every sibling
            t = 0;
            for (uint32_t kk = 0; kk < NG; ++kk) {
                for (uint32_t j = 0; j < T; ++j) {
                    for (uint32_t c2 = 0; c2 < 2; ++c2) {
                        const uint32_t a[4] = {wd + ((kk * T + j) * 2 + c2) * BF, wdm + ((kk * T + j) * 2 + c2) * BF,
                                               wf + ((kk * 2 + c2) * T + j) * BF, wgg + ((kk * 2 + c2) * T + j) * BF};
                        for (uint32_t q = 0; q < 4; ++q) {
                            if (t++ % R == inst) {
                                for (uint32_t s = 0; s < nsib; ++s) {
                                    noc_async_write(a[q], get_noc_addr(get_arg_val<uint32_t>(PC_SIB + 1 + 2 * s),
                                                                       get_arg_val<uint32_t>(PC_SIB + 2 + 2 * s), a[q]),
                                                    BF);
                                }
                            }
                        }
                    }
                }
            }
            noc_async_write_barrier();
            const uint32_t sem = get_semaphore(3);
            for (uint32_t s = 0; s < nsib; ++s) {
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(PC_SIB + 1 + 2 * s),
                                               get_arg_val<uint32_t>(PC_SIB + 2 + 2 * s), sem),
                                  1);
            }
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), nsib);
        }
        cb_push_back(CB_WD, NG * T * 2);
        cb_push_back(CB_WM, NG * T * 2);
        cb_push_back(CB_WF, NG * 2 * T);
        cb_push_back(CB_WG, NG * 2 * T);
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}

// Rows in order.  The first row: X with the position / slot words (CONV), a barrier, S (its ring rows depend on them),
// a barrier, X's pages (the compute starts the prologue), M (MR: with S), a barrier, M's pages.  Later rows (MR): every
// read at once, one barrier, the pages.  Then the gate weights (needed after the first round).
void kernel_main() {
    ink_set_read_vc();
    using namespace gs;
    const uint32_t nr = nrw(), rscr = get_write_ptr(CB_RSCR);
    for (uint32_t q = 0; q < nr; ++q) {
        if constexpr (MR) {
            g_q = q;
        }
        const bool first = q == 0;
        if (first && CONV) {
            noc_async_read(get_noc_addr(0, accessor<true>(arg(A_POS), 4), 0), rscr + RS_POS, 64);
            if constexpr (BATCH) {
                noc_async_read(get_noc_addr(0, accessor<true>(arg(A_SLOT), 4), 0), rscr + RS_SLOT, 64);
            }
            reads_x(true);
            noc_async_read_barrier();
        } else {
            reads_x(first);
        }
        uint32_t p = 0, sb = 0;  // position, first ring page of the slot (batch: slot[row] * HT)
        if constexpr (CONV) {
            row_pos(p, sb);
            reads_s(p, sb);
        }
        if (MR || !first) {
            reads_m(first);
        }
        noc_async_read_barrier();
        finish_x(first, p, sb);
        if (!MR && first) {
            reads_m(true);
            noc_async_read_barrier();
        }
        finish_m(first);
    }
    if constexpr (NG > 0) {
        gate_weights();
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}

#elif defined(COMPILE_FOR_BRISC)
// ------------------------------------------------------------------------------------------------ writer / rounds
namespace gs {
// L1 address of the gather tiles (identical on every core), captured before the master's CB_G pushes move its
// write pointer
uint32_t g_l1 = 0;

// row k of slot g's gather tiles (columns 0..15, +1: 16..31) on the current row's master
FORCE_INLINE uint64_t g_row(uint32_t g, uint32_t k, uint32_t half) {
    const uint32_t r = k & 31;
    return get_noc_addr(RMX(), RMY(), g_l1 + (g * NGT + (k >> 5)) * BF + ((r >> 4) * 2 + half) * 512 + (r & 15) * 32);
}

// element (r, c) of a BF16 tile
FORCE_INLINE uint32_t el(uint32_t tile, uint32_t r, uint32_t c) {
    return tile + ((r >> 4) * 2 + (c >> 4)) * 512 + (r & 15) * 32 + (c & 15) * 2;
}

// the current row's partial tiles of a round whose one-row partial pages are [f, f + c): the returned address + p BF
// is one-row page p (MR: the round-major block of row g_q)
FORCE_INLINE uint32_t part_tiles(uint32_t f, uint32_t c) {
    DeviceZoneScopedN("b-part");
    if constexpr (MR) {
        const uint32_t off = (nrw() - 1) * f + g_q * c;
        cb_wait_front(CB_PART, off + f + c);
        return cb_base(CB_PART) + off * BF;
    } else {
        cb_wait_front(CB_PART, f + c);
        return get_read_ptr(CB_PART);
    }
}

// 512 B of staging rows for round ord (MR: per row too: the previous row's writes may still be in flight)
FORCE_INLINE uint32_t stg_of(uint32_t ord) {
    if constexpr (MR) {
        ord += g_q * NRND;
    }
    return get_write_ptr(CB_STG) + ord * 512;
}

// 32 B staging row at stg holding `copies` copies of the BF16 value at src (the rest zero)
FORCE_INLINE void stage(uint32_t stg, uint32_t src, uint32_t copies) {
    const uint32_t v = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
    auto* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg);
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t lo = 2 * i < copies ? v : 0, hi = 2 * i + 1 < copies ? v : 0;
        w[i] = lo | (hi << 16);
    }
}

// sum of squares (partial tiles pb + p: diagonal [r][r]) of R rows -> slots gb + NPT r + p, column 0
FORCE_INLINE void send_sumsq(uint32_t k, uint32_t R, uint32_t pb, uint32_t gb, uint32_t ord) {
    const uint32_t part = part_tiles(pb, NPT);
    DeviceZoneScopedN("b-xchg");
    const uint32_t stg = stg_of(ord);
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t p = 0; p < NPT; ++p) {
            const uint32_t s = stg + (NPT * r + p) * 32;
            stage(s, el(part + (pb + p) * BF, r, r), 1);
            noc_async_write(s, g_row(gb + NPT * r + p, k, 0), 32);
        }
    }
}

// round MX: rows a < NQ (logits) and 3 + i, i <= NS (Gram) of the partial (columns 0..15) -> slots G_MX + NPT idx + p
FORCE_INLINE void send_mix(uint32_t k, uint32_t NS) {
    const uint32_t part = part_tiles(P_MX, NPT);
    DeviceZoneScopedN("b-xchg");
    for (uint32_t idx = 0; idx < NQ + NS + 1; ++idx) {
        const uint32_t row = idx < NQ ? idx : 3 + idx - NQ;
        for (uint32_t p = 0; p < NPT; ++p) {
            noc_async_write(el(part + (P_MX + p) * BF, row, 0), g_row(G_MX + NPT * idx + p, k, 0), 32);
        }
    }
}

// round MG (slot G_MG + NPT idx + p): idx 0 / 1: P0 rows 0 / 1 (32 columns), 2 / 3: P1 rows 0 / 1 (16 columns),
// 4..6: the Gram values h.h, h.x, x.x replicated over 32 columns
FORCE_INLINE void send_msg(uint32_t k, uint32_t ord) {
    const uint32_t part = part_tiles(P_MG, 3 * NPT);
    DeviceZoneScopedN("b-xchg");
    const uint32_t stg = stg_of(ord);
    for (uint32_t p = 0; p < NPT; ++p) {
        const uint32_t t0 = part + (P_MG + p) * BF, t1 = part + (P_MG + NPT + p) * BF,
                       tg = part + (P_MG + 2 * NPT + p) * BF;
        for (uint32_t r = 0; r < 2; ++r) {
            noc_async_write(el(t0, r, 0), g_row(G_MG + NPT * r + p, k, 0), 32);
            noc_async_write(el(t0, r, 16), g_row(G_MG + NPT * r + p, k, 1), 32);
            noc_async_write(el(t1, r, 0), g_row(G_MG + NPT * (2 + r) + p, k, 0), 32);
        }
        for (uint32_t q = 0; q < 3; ++q) {
            const uint32_t s = stg + (NPT * q + p) * 32;
            stage(s, el(tg, q == 2 ? 1 : 0, q == 0 ? 0 : 1), 16);
            noc_async_write(s, g_row(G_MG + NPT * (4 + q) + p, k, 0), 32);
            noc_async_write(s, g_row(G_MG + NPT * (4 + q) + p, k, 1), 32);
        }
    }
}

// round DN: row src(kk) (columns 0..31) of part tiles P_DN + NPT (2 kk + c2) + p -> the slot of the same index
FORCE_INLINE void send_down(uint32_t k) {
    const uint32_t part = part_tiles(P_DN, 2 * NPT * NG);
    DeviceZoneScopedN("b-xchg");
    for (uint32_t kk = 0; kk < NG; ++kk) {
        const uint32_t src = GSAME ? 0 : kk;
        for (uint32_t i = 2 * NPT * kk; i < 2 * NPT * (kk + 1); ++i) {
            const uint32_t t = part + (P_DN + i) * BF;
            noc_async_write(el(t, src, 0), g_row(G_DN + i, k, 0), 32);
            noc_async_write(el(t, src, 16), g_row(G_DN + i, k, 1), 32);
        }
    }
}

// gather slots of a round kind
constexpr uint32_t nslots_of(uint32_t kind) {
    return kind == RK_E ? N_E : kind == RK_V ? N_V : kind == RK_MX ? N_MX : kind == RK_MG ? N_MG
           : kind == RK_DN ? N_DN : N_SG;
}

// a receiver's result pages of a round (RESF block fa, RESB block fb) arrived: push them
FORCE_INLINE void push_res(const Blk fa, const Blk fb) {
    if (fa.c) {
        cb_reserve_back(CB_RESF, fa.c);
        cb_push_back(CB_RESF, fa.c);
    }
    if (fb.c) {
        cb_reserve_back(CB_RESB, fb.c);
        cb_push_back(CB_RESB, fb.c);
    }
}

// End of round ord (one row): signal the master; master: wait for all cores, hand the gather tiles to its compute,
// multicast the result tiles (its RESF / RESB block of the round) + the round counter; others: wait, push the results.
FORCE_INLINE void sync(uint32_t k, uint32_t ord) {
    const uint32_t kind = round_kind(ord), ns = nslots_of(kind);
    const Blk fa = resf_blk(kind), fb = resb_blk(kind);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(MXc(), MYc(), get_semaphore(0)), 1);
    if (k == 0) {
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        {
            DeviceZoneScopedN("b-gather");
            noc_semaphore_wait_min(sem, C * (ord + 1));
        }
        cb_reserve_back(CB_G, ns * NGT);
        cb_push_back(CB_G, ns * NGT);
        {
            DeviceZoneScopedN("b-master");
            if (fa.c) {
                cb_wait_front(CB_RESF, fa.f + fa.c);
            }
            if (fb.c) {
                cb_wait_front(CB_RESB, fb.f + fb.c);
            }
        }
        if constexpr (C > 1) {
            if (fa.c) {
                const uint32_t sa = get_read_ptr(CB_RESF) + fa.f * FP;
                noc_async_write_multicast(sa, mcast_addr(sa), fa.c * FP, C - 1, true);
            }
            if (fb.c) {
                const uint32_t sb = get_read_ptr(CB_RESB) + fb.f * BF;
                noc_async_write_multicast(sb, mcast_addr(sb), fb.c * BF, C - 1, true);
            }
            const uint32_t rs = get_semaphore(1);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs) = ord + 1;
            noc_semaphore_set_multicast(rs, mcast_addr(rs), C - 1);
        }
    } else {
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        DeviceZoneScopedN("b-wres");
        noc_semaphore_wait_min(sem, ord + 1);
        push_res(fa, fb);
    }
}

// MR: this core's partial of the current row's round is out: signal that row's master
FORCE_INLINE void signal() {
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(RMX(), RMY(), get_semaphore(0)), 1);
}

// MR: round ord of row k, on its master (core k): wait for all cores, hand the gather tiles to the compute; its result
// block (CB_MRESF / CB_MRESB, one-row pages) goes to row k's RESF / RESB pages of core 0 (core 0: a local copy;
// others: a write, then core 0's semaphore 4 + 1).  Core 0, once every row's block is there, multicasts the round's
// blocks of all n rows to the group and sets semaphore 5 = ord + 1 everywhere (one multicast per round: concurrent
// linked multicasts of several masters to the same rectangle stall each other).
FORCE_INLINE void master_mr(uint32_t k, uint32_t n, uint32_t ord) {
    const uint32_t kind = round_kind(ord), ns = nslots_of(kind);
    const Blk fa = resf_blk(kind), fb = resb_blk(kind);
    {
        DeviceZoneScopedN("b-gather");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), C * (ord + 1));
    }
    cb_reserve_back(CB_G, ns * NGT);
    cb_push_back(CB_G, ns * NGT);
    {
        DeviceZoneScopedN("b-master");
        if (fa.c) {
            cb_wait_front(CB_MRESF, fa.f + fa.c);
        }
        if (fb.c) {
            cb_wait_front(CB_MRESB, fb.f + fb.c);
        }
    }
    const uint32_t sa = get_read_ptr(CB_MRESF) + fa.f * FP, da = cb_base(CB_RESF) + (n * fa.f + k * fa.c) * FP;
    const uint32_t sb = get_read_ptr(CB_MRESB) + fb.f * BF, db = cb_base(CB_RESB) + (n * fb.f + k * fb.c) * BF;
    auto at0 = [k](uint32_t addr) { return k ? get_noc_addr(MXc(), MYc(), addr) : get_noc_addr(addr); };
    if (fa.c) {
        noc_async_write(sa, at0(da), fa.c * FP);
    }
    if (fb.c) {
        noc_async_write(sb, at0(db), fb.c * BF);
    }
    noc_async_write_barrier();
    if (k) {
        noc_semaphore_inc(at0(get_semaphore(4)), 1);
        return;
    }
    {
        DeviceZoneScopedN("b-collect");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4)), (n - 1) * (ord + 1));
    }
    if constexpr (C > 1) {
        if (fa.c) {
            const uint32_t a = cb_base(CB_RESF) + n * fa.f * FP;
            noc_async_write_multicast(a, mcast_addr(a), n * fa.c * FP, C - 1, true);
        }
        if (fb.c) {
            const uint32_t b = cb_base(CB_RESB) + n * fb.f * BF;
            noc_async_write_multicast(b, mcast_addr(b), n * fb.c * BF, C - 1, true);
        }
        const uint32_t rs = get_semaphore(5);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs) = ord + 1;
        noc_semaphore_set_multicast(rs, mcast_addr(rs), C - 1);
    }
}

// MR: wait for the results of round ord (every row's), push them (RESF / RESB: round-major blocks, rows in order)
FORCE_INLINE void recv_mr(uint32_t n, uint32_t ord) {
    const uint32_t kind = round_kind(ord);
    const Blk fa = resf_blk(kind), fb = resb_blk(kind);
    if (core_index()) {
        DeviceZoneScopedN("b-wres");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5)), ord + 1);
    }
    push_res(Blk{fa.f, n * fa.c}, Blk{fb.f, n * fb.c});
}

// row 0 of the T tiles of cb (BF16; RP: rows j of one row-packed tile) -> row `row` of the T column tiles of tensor
// addr
template <bool RP = false>
FORCE_INLINE void write_row(uint32_t cb, uint32_t addr, uint32_t j0, uint32_t row, uint32_t src_row = 0) {
    const auto o = accessor<true>(addr, BF);
    const uint32_t r = front(cb, RP ? 1 : T, BF);
    for (uint32_t j = 0; j < T; ++j) {
        const uint64_t a = get_noc_addr(j0 + j, o) + row * 32;
        const uint32_t s = RP ? r + j * 32 : r + j * BF + src_row * 32;
        noc_async_write(s, a, 32);
        noc_async_write(s + 512, a + 512, 32);
    }
}

// constants: core 0 reads every tile and multicasts tiles [K_CORE, NCONST) (MR: all, the cores of rows q < n are
// masters too) to its group (semaphore 2 = arrived); then every master zero-fills its gather tiles' unwritten rows /
// columns (disjoint from the remote writes; off the constants' path: the compute reads them only after this RISC's
// first gather push)
FORCE_INLINE void constants(uint32_t k, uint32_t n) {
    DeviceZoneScopedN("b-const");
    constexpr uint32_t K0 = MR ? 0 : K_CORE;
    cb_reserve_back(CB_CONST, NCONST);
    const uint32_t kc = get_write_ptr(CB_CONST), ks = get_semaphore(2);
    if (k == 0) {
        const auto kt = accessor<true>(arg(A_CONST), BF);
        for (uint32_t i = 0; i < NCONST; ++i) {
            noc_async_read_tile(i, kt, kc + i * BF);
        }
        noc_async_read_barrier();
        if constexpr (C > 1) {
            noc_async_write_multicast(kc + K0 * BF, mcast_addr(kc + K0 * BF), (NCONST - K0) * BF, C - 1, true);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ks) = 1;
            noc_semaphore_set_multicast(ks, mcast_addr(ks), C - 1);
        }
    } else {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ks), 1);
    }
    cb_push_back(CB_CONST, NCONST);
    if (k < n) {
        for (uint32_t s = 0; s < N_GSLOT; ++s) {
            const uint32_t ncols = slot_cols(s);
            for (uint32_t gt = 0; gt < NGT; ++gt) {
                const uint32_t rows = gt + 1 == NGT ? CL : 32;
                for (uint32_t f = 0; f < 4; ++f) {
                    const uint32_t row0 = (f >> 1) * 16, col0 = (f & 1) * 16;
                    uint32_t written = 0;
                    if (col0 < ncols && rows > row0) {
                        written = rows - row0 < 16 ? rows - row0 : 16;
                    }
                    if (written < 16) {
                        zero_fill(g_l1 + (s * NGT + gt) * BF + f * 512 + written * 32, (16 - written) * 32);
                    }
                }
            }
        }
        noc_async_read_barrier();
    }
}

// outputs: MIX: row 4 o + j of the row-packed CB_OUT tile (layer 0: row o of column tile j) -> the instance's row of
// output tensor o, column tile j0 + j (a shared rows tensor; decode: row 0)
FORCE_INLINE void outputs(uint32_t j0) {
    const uint32_t r = front(CB_OUT, PR_OUT, BF);
    const uint32_t orow = row_of() * 32;
    for (uint32_t o = 0; o < NOUT; ++o) {
        const auto out = accessor<true>(arg(A_OUT + o), BF);
        for (uint32_t j = 0; j < T; ++j) {
            const uint64_t a = get_noc_addr(j0 + j, out) + orow;
            const uint32_t s = MIX ? r + (4 * o + j) * 32 : r + j * BF + o * 32;
            noc_async_write(s, a, 32);
            noc_async_write(s + 512, a + 512, 32);
        }
    }
}
}  // namespace gs

// segment s of the current row (see the compute): the writes of what the segment made, then (s < NRND) this core's
// partial of round s to the row's master
FORCE_INLINE void wseg(const uint32_t s, const uint32_t k, const uint32_t j0, const uint32_t NS) {
    using namespace gs;
    if (s == 0) {
        if constexpr (PRO == 1) {  // h -> out (row 0); TAPO: also the instance's row of the shared tap rows
            write_row<true>(CB_HV, arg(A_OH), j0, 0);
            if (arg(A_TAPO)) {
                write_row<true>(CB_HV, arg(A_TAPO), j0, row_of());
            }
        } else if constexpr (PRO == 2) {  // r (row-packed FP32) -> out (FP32 row 0)
            const auto o = accessor<true>(arg(A_OH), FP);
            const uint32_t r = front(CB_RF, 1, FP);
            for (uint32_t j = 0; j < T; ++j) {
                const uint64_t a = get_noc_addr(j0 + j, o);
                noc_async_write(r + j * 64, a, 64);
                noc_async_write(r + 1024 + j * 64, a + 1024, 64);
            }
        }
    } else {
        const uint32_t kind = round_kind(s - 1);
        if (kind == RK_E) {  // embed_norm: h -> out (row 0)
            write_row(CB_HV, arg(A_OH), j0, 0);
        } else if (kind == RK_MG) {  // M (FP32) and nM (BF16) tiles: query tile t (rows 32 t ..) of the [64, H] outputs
            const auto mo = accessor<true>(arg(A_MO), FP);
            const auto nmo = accessor<true>(arg(A_NMO), BF);
            const uint32_t rm = front(CB_MF, PR_MF, FP), rn = front(CB_OUT, PR_OUT, BF);
            for (uint32_t t = 0; t < 2; ++t) {
                for (uint32_t j = 0; j < T; ++j) {
                    noc_async_write_page(t * HT + j0 + j, mo, rm + (t * T + j) * FP);
                    noc_async_write_page(t * HT + j0 + j, nmo, rn + (t * T + j) * BF);
                }
            }
        } else if (kind == RK_V || kind == RK_MX) {
            if (arg(A_APPEND)) {  // bf16(v s_v) -> X row NS
                write_row<CONV>(CB_NH, arg(A_X), j0, NS);
            }
            if (kind == RK_V || NG == 0) {
                outputs(j0);
            }
        } else if (kind == RK_SG) {
            outputs(j0);
        }
    }
    if (s < NRND) {
        switch (round_kind(s)) {
            case RK_E: send_sumsq(k, 1, P_E, G_E, O_E); break;
            case RK_V: send_sumsq(k, 1, P_V, G_V, O_V); break;
            case RK_MX: send_mix(k, NS); break;
            case RK_MG: send_msg(k, O_MG); break;
            case RK_DN: send_down(k); break;
            default: send_sumsq(k, NG, P_SG, G_SG, O_SG); break;
        }
    }
}

void kernel_main() {
    ink_set_read_vc();
    using namespace gs;
    const uint32_t k = core_index(), j0 = k * T, NS = arg(A_NS), n = nrw();
    g_l1 = get_write_ptr(CB_G);
    constants(k, n);
    for (uint32_t s = 0; s <= NRND; ++s) {
        for (uint32_t q = 0; q < n; ++q) {
            if constexpr (MR) {
                g_q = q;
                wseg(s, k, j0, NS);
                if (s < NRND) {
                    signal();
                }
            } else {
                wseg(s, k, j0, NS);
            }
        }
        if (s < NRND) {
            if constexpr (MR) {
                if (k < n) {
                    master_mr(k, n, s);
                }
                recv_mr(n, s);
            } else {
                sync(k, s);
            }
        }
    }
    noc_async_write_barrier();
}

#else
// ====================================================================================================== compute
namespace gs {
// MR (see the RISCs): nothing is popped; before a segment of row q the unpacker puts page q P of every per-row CB at
// its front (read pointer + its own tiles-acked count, which only its waits use: the producers never see it), and
// the results of the round just finished (RESF / RESB, round-major) likewise, so the one-row code runs unchanged on
// row q's pages
ALWI void at(uint32_t cb, uint32_t off) {
#ifdef TRISC_UNPACK
    auto& c = get_local_cb_interface(cb);
    c.tiles_acked = off;
    c.fifo_rd_ptr = c.fifo_limit - c.fifo_size + off * c.fifo_page_size;
#endif
}
ALWI void use_row(uint32_t q, uint32_t s, uint32_t n) {
    if constexpr (MR) {
        constexpr uint32_t cbs[] = {CB_IN, CB_ST,  CB_BASE, CB_X,   CB_MSG, CB_NM, CB_HV, CB_VH, CB_VL, CB_XF,
                                    CB_ZQ, CB_ZQL, CB_VLC,  CB_NX,  CB_ORP, CB_YR, CB_GK, CB_PH, CB_PL, CB_SCR};
        constexpr uint32_t pages[] = {PR_IN, PR_IN, PR_BASE, PR_X,  PR_MSG, PR_NM, PR_HV, PR_V,  PR_V,  PR_XF,
                                      PR_Z,  PR_Z,  PR_VLC,  PR_NX, PR_GY,  PR_GY, PR_GK, PR_NX, PR_NX, PR_SCR};
        for (uint32_t i = 0; i < sizeof(cbs) / sizeof(cbs[0]); ++i) {
            if (pages[i]) {
                at(cbs[i], q * pages[i]);
            }
        }
        if constexpr (MSG) {  // compute-made per-row pages
            at(CB_RESF, q * PR_RESF);
        }
        if (s) {
            const uint32_t kind = round_kind(s - 1);
            const Blk fa = resf_blk(kind), fb = resb_blk(kind);
            if (fa.c) {
                at(CB_RESF, (n - 1) * fa.f + q * fa.c);
            }
            if (fb.c) {
                at(CB_RESB, (n - 1) * fb.f + q * fb.c);
            }
        }
    }
}
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
ALWI void pack_to(uint32_t cb, uint32_t d) {
    cb_reserve_back(cb, 1);
    pk(d, cb);
    cb_push_back(cb, 1);
}

// D[dst] += sel x (gather tiles of slot s)   (after mm_begin(CB_CONST, CB_G, 0))
ALWI void red(uint32_t sel, uint32_t s, uint32_t dst) {
    for (uint32_t gt = 0; gt < NGT; ++gt) {
        matmul_tiles(CB_CONST, CB_G, sel, s * NGT + gt, dst);
    }
}

// master: 1 / rms of the R rows of a sum-of-squares round -> RESF tile
ALWI void finish_sumsq(uint32_t R, uint32_t gb) {
    DeviceZoneScopedN("t-fin");
    cb_wait_front(CB_G, (gb + NPT * R) * NGT);
    sec_begin();
    mm_begin(CB_CONST, CB_G, 0);
    for (uint32_t s = 0; s < NPT * R; ++s) {
        red(K_E + s / NPT, gb + s, 0);
    }
    glue::sfpu_init();
    glue::affine<INVH, EPS_BITS>(0);
    glue::rsqrt(0);
    if (gb == G_SG) {  // gate k's scale (column 0) -> every column of rows 4 k + j (row-packed outputs)
        glue::sfpu_init();
        glue::col0(0);
        glue::groups(0, 0);
    }
    sec_pack();
    pack_to(CB_MRESF, 0);
    sec_end();
}

// D[x] -> its NPT transport parts in D[d], D[d + 1] (, D[d + 2])
ALWI void parts_of(uint32_t x, uint32_t d) {
    if constexpr (NPT == 3) {
        glue::split3(x, d, d + 1, d + 2);
    } else {
        glue::split(x, d, d + 1);
    }
}

// D[d .. d + n) -> CB_PART (after sec_pack)
ALWI void pack_parts(uint32_t d, uint32_t n) {
    cb_reserve_back(CB_PART, n);
    for (uint32_t i = 0; i < n; ++i) {
        pk(d + i, CB_PART);
    }
    cb_push_back(CB_PART, n);
}

// partial sum of squares of rows (hi + lo) of the T tiles -> CB_PART (NPT parts)
ALWI void partial_sumsq(uint32_t vh, uint32_t vl, bool lo) {
    DeviceZoneScopedN("t-part");
    sec_begin();
    mm_begin(vh, vh, 1);
    for (uint32_t j = 0; j < T; ++j) {
        matmul_tiles(vh, vh, j, j, 0);
    }
    if (lo) {
        mm_begin(vh, vl, 1);
        for (uint32_t j = 0; j < T; ++j) {
            matmul_tiles(vh, vl, j, j, 1);
        }
        glue::sfpu_init();
        glue::add(1, 1, 1);
        glue::add(0, 1, 0);
    }
    glue::sfpu_init();
    parts_of(0, 1);
    sec_pack();
    pack_parts(1, NPT);
    sec_end();
}

// D0 = (hi + lo) of tile j, scaled by column 0 of the RESF tile rt (per row)
ALWI void load_scaled(uint32_t vh, uint32_t vl, bool lo, uint32_t j, uint32_t rt) {
    cp(vh, j, 0);
    if (lo) {
        cp(vl, j, 1);
        glue::sfpu_init();
        glue::add(0, 1, 0);
    }
    cp(CB_RESF, rt, 2);
    MATH((ckernel::sfpu_bcast_col_init()));
    ckernel::sfpu_mul_bcast_col(0, 2);
}

// out_j = bf16((hi + lo) / rms * gamma) for all T tiles -> CB_OUT
template <uint32_t R>
ALWI void outputs(uint32_t vh, uint32_t vl, bool lo, uint32_t rt) {
    DeviceZoneScopedN("t-out");
    cb_wait_front(CB_GAM, T);
    cb_reserve_back(CB_OUT, T);
    for (uint32_t j = 0; j < T; ++j) {
        sec_begin();
        load_scaled(vh, vl, lo, j, rt);
        cp(CB_GAM, j, 1);
        glue::sfpu_init();
        glue::mul(0, 1, 0);
        glue::rne(0);
        glue::clean<R>(0);
        sec_pack();
        pk(0, CB_OUT);
        sec_end();
    }
    cb_push_back(CB_OUT, T);
}

// nv = bf16(v s_v) (row 0; CONV: rows j of one row-packed tile) -> CB_NH (the appended block state)
ALWI void normed_partial(uint32_t vh, uint32_t rt) {
    constexpr uint32_t n = CONV ? 1 : T;
    cb_reserve_back(CB_NH, n);
    for (uint32_t j = 0; j < n; ++j) {
        sec_begin();
        load_scaled(vh, CB_VL, HAS_LO, j, rt);
        glue::sfpu_init();
        glue::rne(0);
        glue::clean<CONV ? T : 1>(0);
        sec_pack();
        pk(0, CB_NH);
        sec_end();
    }
    cb_push_back(CB_NH, n);
}

// D_j = O_j = C S_j for every column tile j (FP32; S = XF (+ the lo row of v: column tiles CB_VLC))
ALWI void mix_all() {
    mm_begin(CB_RESB, CB_XF, 0);
    for (uint32_t j = 0; j < T; ++j) {
        matmul_tiles(CB_RESB, CB_XF, R_C, j, j);
        matmul_tiles(CB_RESB, CB_XF, R_C + 1, j, j);
    }
    if constexpr (HAS_LO) {
        mm_begin(CB_RESB, CB_VLC, 0);
        for (uint32_t j = 0; j < T; ++j) {
            matmul_tiles(CB_RESB, CB_VLC, R_C, j, j);
        }
    }
}


// ---------------------------------------------------------------- round MX
// per core: XF = [v; X rows < NS], ZQ = [q hi (rows a); v (row 3); X rows (4..)], ZQL = [q lo (rows a); v lo (row 3)];
// partial P = ZQ XF^T + ZQL XF^T (+ ZQ VLC^T): rows a: q_a . S_i, rows 3 + i: S_i . S_j (FP32), as hi / lo.
// CONV: v hi / lo are row-packed tiles (row j -> rows 0 / 3 by the selectors E0J_j / E3J_j); VLC_j = v lo of column
// tile j in row 0
ALWI void mix_partial(uint32_t vh) {
    DeviceZoneScopedN("t-mxp");
    cb_wait_front(CB_X, T);
    cb_wait_front(CB_Q, T);
    cb_reserve_back(CB_XF, T);
    cb_reserve_back(CB_ZQ, T);
    cb_reserve_back(CB_ZQL, T);
    if constexpr (HAS_LO) {
        cb_reserve_back(CB_VLC, T);
    }
    for (uint32_t j = 0; j < T; ++j) {
        sec_begin();
        mm_begin(CB_CONST, CB_X, 0);
        matmul_tiles(CB_CONST, CB_X, K_SX, j, 0);
        matmul_tiles(CB_CONST, CB_X, K_SX3, j, 1);
        mm_begin(CB_CONST, vh, 0);
        matmul_tiles(CB_CONST, vh, CONV ? K_E0J + j : K_E00, CONV ? 0 : j, 0);
        matmul_tiles(CB_CONST, vh, CONV ? K_E3J + j : K_E30, CONV ? 0 : j, 1);
        mm_begin(CB_CONST, CB_Q, 0);
        matmul_tiles(CB_CONST, CB_Q, K_Q, j, 1);
        matmul_tiles(CB_CONST, CB_Q, K_QL, j, 2);
        if constexpr (HAS_LO) {
            mm_begin(CB_CONST, CB_VL, 0);
            matmul_tiles(CB_CONST, CB_VL, K_E3J + j, 0, 2);
            matmul_tiles(CB_CONST, CB_VL, K_E0J + j, 0, 3);
        }
        sec_pack();
        pk(0, CB_XF);
        pk(1, CB_ZQ);
        pk(2, CB_ZQL);
        if constexpr (HAS_LO) {
            pk(3, CB_VLC);
        }
        sec_end();
    }
    cb_push_back(CB_XF, T);
    cb_push_back(CB_ZQ, T);
    cb_push_back(CB_ZQL, T);
    if constexpr (HAS_LO) {
        cb_push_back(CB_VLC, T);
        cb_wait_front(CB_VLC, T);
    }
    cb_wait_front(CB_XF, T);
    cb_wait_front(CB_ZQ, T);
    cb_wait_front(CB_ZQL, T);
    sec_begin();
    mm_begin(CB_ZQ, CB_XF, 1);
    for (uint32_t j = 0; j < T; ++j) {
        matmul_tiles(CB_ZQ, CB_XF, j, j, 0);
    }
    mm_begin(CB_ZQL, CB_XF, 1);
    for (uint32_t j = 0; j < T; ++j) {
        matmul_tiles(CB_ZQL, CB_XF, j, j, 0);
    }
    if constexpr (HAS_LO) {
        mm_begin(CB_ZQ, CB_VLC, 1);
        for (uint32_t j = 0; j < T; ++j) {
            matmul_tiles(CB_ZQ, CB_VLC, j, j, 0);
        }
    }
    glue::sfpu_init();
    parts_of(0, 1);
    sec_pack();
    pack_parts(1, NPT);
    sec_end();
}

// master: R = the reduced partial (logits rows a, Gram rows i), states i = 0 (v) .. NS
//   M = [s_v, 1, .., 1] (columns), E = exp(R M) (masked), G = Gram; Z = rowsum E, C = E M / Z (-> RESB),
//   n = rowsum(C o (C G)) = |O|^2, s_O = rsqrt(n / H + eps) (-> RESF)
ALWI void mix_finish(uint32_t NS) {
    cb_wait_front(CB_G, (G_MX + N_MX) * NGT);
    { DeviceZoneScopedN("t-mxf"); }
    {
    DeviceZoneScopedN("t-mf1");
    sec_begin();
    mm_begin(CB_CONST, CB_G, 0);
    for (uint32_t idx = 0; idx < NQ + NS + 1; ++idx) {
        for (uint32_t p = 0; p < NPT; ++p) {
            const uint32_t s = G_MX + NPT * idx + p;
            if (idx < NQ) {
                red(K_E + idx, s, 0);  // logits
            } else {
                red(K_E + idx - NQ, s, 2);  // Gram row i
                if (idx == NQ) {
                    red(K_SEL7, s, 1);  // v . S_j over rows 0..7: column 0 = |v|^2
                }
            }
        }
    }
    cp(CB_CONST, K_COL0, 3);
    cp(CB_CONST, K_COLR, 4);
    glue::sfpu_init();
    glue::mul(1, 3, 1);
    glue::affine<INVH, EPS_BITS>(1);
    glue::rsqrt(1);
    glue::sfpu_init();
    glue::mul(1, 3, 1);
    glue::add(1, 4, 1);  // M
    glue::mul(0, 1, 0);  // logits of the normed states
    glue::exp(0);
    cp(CB_CONST, K_MASK, 3);
    glue::sfpu_init();
    glue::mul(0, 3, 0);    // E
    glue::split(0, 3, 4);  // E hi / lo
    glue::split(2, 5, 6);  // G hi / lo
    sec_pack();
    cb_reserve_back(CB_SCR, 4);
    for (uint32_t d = 3; d < 7; ++d) {
        pk(d, CB_SCR);
    }
    cb_push_back(CB_SCR, 4);
    pack_to(CB_MRESF, 1);
    sec_end();
    }
    { DeviceZoneScopedN("t-mf2"); }
    cb_wait_front(CB_SCR, 4);
    cb_wait_front(CB_MRESF, R_M + 1);
    sec_begin();
    mm_begin(CB_SCR, CB_CONST, 0);
    matmul_tiles(CB_SCR, CB_CONST, 0, K_J, 0);
    matmul_tiles(CB_SCR, CB_CONST, 1, K_J, 0);  // Z
    cp(CB_SCR, 0, 1);
    cp(CB_SCR, 1, 2);
    glue::sfpu_init();
    glue::add(1, 2, 1);  // E
    cp(CB_MRESF, R_M, 2);
    glue::recip(0);
    glue::sfpu_init();
    glue::mul(1, 2, 1);
    glue::mul(1, 0, 1);  // C
    glue::clean<NQ>(1);
    glue::split(1, 2, 3);
    sec_pack();
    cb_reserve_back(CB_MRESB, 2);
    pk(2, CB_MRESB);
    pk(3, CB_MRESB);
    cb_push_back(CB_MRESB, 2);
    sec_end();

    cb_wait_front(CB_MRESB, R_C + 2);
    sec_begin();
    mm_begin(CB_MRESB, CB_SCR, 0);
    matmul_tiles(CB_MRESB, CB_SCR, R_C, 2, 0);
    matmul_tiles(CB_MRESB, CB_SCR, R_C, 3, 0);
    matmul_tiles(CB_MRESB, CB_SCR, R_C + 1, 2, 0);  // Y = C G
    cp(CB_MRESB, R_C, 1);
    cp(CB_MRESB, R_C + 1, 2);
    glue::sfpu_init();
    glue::add(1, 2, 1);  // C
    glue::mul(0, 1, 0);  // C o Y
    glue::split(0, 1, 2);
    sec_pack();
    cb_reserve_back(CB_SCR, 2);
    pk(1, CB_SCR);
    pk(2, CB_SCR);
    cb_push_back(CB_SCR, 2);
    sec_end();

    cb_wait_front(CB_SCR, 6);
    sec_begin();
    mm_begin(CB_SCR, CB_CONST, 0);
    matmul_tiles(CB_SCR, CB_CONST, 4, K_J, 0);
    matmul_tiles(CB_SCR, CB_CONST, 5, K_J, 0);  // n = |O|^2
    glue::sfpu_init();
    glue::affine<INVH, EPS_BITS>(0);
    glue::rsqrt(0);
    glue::clean<NQ>(0);
    glue::sfpu_init();
    glue::groups(0, 0);  // s_O_a -> rows 4 a + j (row-packed outputs)
    sec_pack();
    pack_to(CB_MRESF, 0);
    sec_end();
}

// ---------------------------------------------------------------- messages
// master, query tile t (columns = queries 32 t ..): from the broadcast sums Lh = q . h, Lx = q . x, g_hh, g_hx, g_xx:
// s_h = rsqrt(g_hh / H + eps), w_h = 1 / (1 + exp(Lx - s_h Lh)), a_h = w_h s_h, a_x = 1 - w_h = exp(..) w_h,
// s_M = rsqrt((a_h^2 g_hh + 2 a_h a_x g_hx + a_x^2 g_xx) / H + eps) -> Bc = [a_h; a_x] (rows 0, 1), Bs = [s_M]
ALWI void msg_finish(uint32_t t) {
    DeviceZoneScopedN("t-mgf");
    cb_wait_front(CB_G, (G_MG + N_MG) * NGT);
    sec_begin();
    mm_begin(CB_CONST, CB_G, 0);
    for (uint32_t p = 0; p < NPT; ++p) {
        red(K_SEL7, G_MG + NPT * (2 * t) + p, 0);
        red(K_SEL7, G_MG + NPT * (2 * t + 1) + p, 1);
        red(K_SEL7, G_MG + NPT * 4 + p, 2);
        red(K_SEL7, G_MG + NPT * 5 + p, 3);
        red(K_SEL7, G_MG + NPT * 6 + p, 4);
    }
    glue::sfpu_init();
    glue::affine_to<INVH, EPS_BITS>(2, 5);
    glue::rsqrt(5);  // s_h
    glue::sfpu_init();
    glue::mul(0, 5, 0);
    glue::sub(1, 0, 1);
    glue::clamp<0x42A00000u>(1);  // |d| <= 80: exp(d) and 1 + exp(d) stay finite (message queries reach |q| ~ 68)
    glue::exp(1);                 // e = exp(Lx - s_h Lh)
    glue::sfpu_init();
    glue::affine_to<0x3F800000u, 0x3F800000u>(1, 0);
    glue::recip(0);  // w_h
    glue::sfpu_init();
    glue::mul(1, 0, 1);  // a_x
    glue::mul(0, 5, 0);  // a_h
    glue::mul(0, 0, 6);
    glue::mul(6, 2, 6);
    glue::mul(0, 1, 7);
    glue::mul(7, 3, 7);
    glue::add(7, 7, 7);
    glue::add(6, 7, 6);
    glue::mul(1, 1, 7);
    glue::mul(7, 4, 7);
    glue::add(6, 7, 6);
    glue::affine<INVH, EPS_BITS>(6);
    glue::rsqrt(6);  // s_M
    cp(CB_CONST, K_RM, 2);
    cp(CB_CONST, K_RM + 1, 3);
    glue::sfpu_init();
    glue::mul(0, 2, 0);
    glue::mul(1, 3, 1);
    glue::add(0, 1, 0);  // Bc
    glue::mul(6, 2, 6);  // Bs
    if (t == 1) {        // queries 32..47 only
        cp(CB_CONST, K_MASK, 4);
        glue::sfpu_init();
        glue::mul(0, 4, 0);
        glue::mul(6, 4, 6);
    }
    glue::clean<2>(0);
    glue::clean<1>(6);
    glue::split(0, 1, 3);
    glue::split(6, 4, 5);
    glue::clean<2>(1);
    glue::clean<2>(3);
    glue::clean<1>(4);
    glue::clean<1>(5);
    sec_pack();
    cb_reserve_back(CB_MRESB, 4);
    pk(1, CB_MRESB);
    pk(3, CB_MRESB);
    pk(4, CB_MRESB);
    pk(5, CB_MRESB);
    cb_push_back(CB_MRESB, 4);
    sec_end();
}

// messages, before round MG: XF and the partials
ALWI void msg_partials(uint32_t vh) {
    // XF = [h; X row NS]
    cb_wait_front(CB_X, T);
    cb_wait_front(CB_Q, 4 * T);
    cb_reserve_back(CB_XF, T);
    for (uint32_t j = 0; j < T; ++j) {
        sec_begin();
        mm_begin(CB_CONST, CB_X, 0);
        matmul_tiles(CB_CONST, CB_X, K_MX, j, 0);
        mm_begin(CB_CONST, vh, 0);
        matmul_tiles(CB_CONST, vh, CONV ? K_E0J + j : K_E00, CONV ? 0 : j, 0);
        sec_pack();
        pk(0, CB_XF);
        sec_end();
    }
    cb_push_back(CB_XF, T);
    cb_wait_front(CB_XF, T);
    // partial: P0 = XF Q0^T, P1 = XF Q1^T (states x queries, q hi + lo), Gram XF XF^T
    {
        DeviceZoneScopedN("t-mgp");
        sec_begin();
        mm_begin(CB_XF, CB_Q, 1);
        for (uint32_t j = 0; j < T; ++j) {
            matmul_tiles(CB_XF, CB_Q, j, j, 0);
            matmul_tiles(CB_XF, CB_Q, j, 2 * T + j, 0);
            matmul_tiles(CB_XF, CB_Q, j, T + j, 1);
            matmul_tiles(CB_XF, CB_Q, j, 3 * T + j, 1);
        }
        glue::sfpu_init();
        parts_of(0, 2);
        parts_of(1, 2 + NPT);
        sec_pack();
        pack_parts(2, 2 * NPT);
        sec_end();
        sec_begin();
        mm_begin(CB_XF, CB_XF, 1);
        for (uint32_t j = 0; j < T; ++j) {
            matmul_tiles(CB_XF, CB_XF, j, j, 0);
        }
        glue::sfpu_init();
        parts_of(0, 1);
        sec_pack();
        pack_parts(1, NPT);
        sec_end();
    }
}

// messages, after round MG: the coefficients and M / nM
ALWI void msg_out() {
    {
        DeviceZoneScopedN("t-wait");
        cb_wait_front(CB_RESB, 8);
    }
    // per query tile t: CF = Bc^T (rows = queries: a_h, a_x), S = Bs^T (column 0 = s_M), FP32
    for (uint32_t t = 0; t < 2; ++t) {
        sec_begin();
        mm_begin(CB_CONST, CB_RESB, 1);
        matmul_tiles(CB_CONST, CB_RESB, K_ID, 4 * t, 0);
        matmul_tiles(CB_CONST, CB_RESB, K_ID, 4 * t + 1, 1);
        matmul_tiles(CB_CONST, CB_RESB, K_ID, 4 * t + 2, 2);
        matmul_tiles(CB_CONST, CB_RESB, K_ID, 4 * t + 3, 2);
        sec_pack();
        cb_reserve_back(CB_SCR, 2);
        pk(0, CB_SCR);
        pk(1, CB_SCR);
        cb_push_back(CB_SCR, 2);
        pack_to(CB_RESF, 2);
        sec_end();
    }
    cb_wait_front(CB_SCR, 4);
    cb_wait_front(CB_RESF, 2);
    // M = CF XF (FP32) -> CB_MF; nM = bf16(M s_M) -> CB_OUT
    DeviceZoneScopedN("t-mgo");
    cb_reserve_back(CB_MF, 2 * T);
    cb_reserve_back(CB_OUT, 2 * T);
    for (uint32_t t = 0; t < 2; ++t) {
        for (uint32_t j = 0; j < T; ++j) {
            sec_begin();
            mm_begin(CB_SCR, CB_XF, 0);
            matmul_tiles(CB_SCR, CB_XF, 2 * t, j, 0);
            matmul_tiles(CB_SCR, CB_XF, 2 * t + 1, j, 0);
            matmul_tiles(CB_SCR, CB_XF, 2 * t, j, 1);
            matmul_tiles(CB_SCR, CB_XF, 2 * t + 1, j, 1);
            cp(CB_RESF, t, 2);
            MATH((ckernel::sfpu_bcast_col_init()));
            ckernel::sfpu_mul_bcast_col(1, 2);
            glue::sfpu_init();
            glue::rne_full(1);
            sec_pack();
            pk(0, CB_MF);
            pk(1, CB_OUT);
            sec_end();
        }
    }
    cb_push_back(CB_MF, 2 * T);
    cb_push_back(CB_OUT, 2 * T);
}

// the CB holding v (hi) after the prologue
constexpr uint32_t VHC = (HAS_E || PRO == 1) ? CB_HV : (CONV ? CB_VH : CB_BASE);

ALWI void wait_v() {
    cb_wait_front(VHC, CONV ? 1 : T);
    if constexpr (HAS_LO) {
        cb_wait_front(CB_VL, 1);
    }
}

// v is there: the partial of the first round over it
ALWI void first_partial() {
    wait_v();
    if constexpr (MSG) {
        msg_partials(VHC);
    } else if constexpr (HAS_V) {  // layer 0: round V
        partial_sumsq(VHC, CB_VL, HAS_LO);
    } else {
        mix_partial(VHC);
    }
}

// segment 0: the prologue, then the first round's partial
// CONV: one section on row-packed tiles (row j = column tile j): x, states, taps, base -> h (PRO 1) or r, hi, lo
ALWI void seg_first() {
    if constexpr (CONV) {
        DeviceZoneScopedN("t-pro");
        constexpr uint32_t MODE = PRO == 1 ? 1 : (BOUND ? 2 : 3);
        {
            DeviceZoneScopedN("t-prow");
            cb_wait_front(CB_IN, 1);
            cb_wait_front(CB_ST, 1);
            cb_wait_front(CB_TAP, 1);
            if constexpr (MODE != 2) {
                cb_wait_front(CB_BASE, 1);
            }
        }
        sec_begin();
        cp(CB_IN, 0, 0);
        cp(CB_ST, 0, 1);
        cp(CB_TAP, 0, 2);
        if constexpr (MODE != 2) {
            cp(CB_BASE, 0, 3);
        }
        glue::sfpu_init();
        glue::conv<MODE, T>();
        sec_pack();
        if constexpr (PRO == 1) {
            pack_to(CB_HV, 4);
        } else {
            cb_reserve_back(CB_RF, 1);
            cb_reserve_back(CB_VH, 1);
            cb_reserve_back(CB_VL, 1);
            pk(4, CB_RF);
            pk(5, CB_VH);
            pk(6, CB_VL);
            cb_push_back(CB_RF, 1);
            cb_push_back(CB_VH, 1);
            cb_push_back(CB_VL, 1);
        }
        sec_end();
    } else {
        cb_wait_front(CB_BASE, T);
    }
    cb_wait_front(CB_CONST, NCONST);
    if constexpr (HAS_E) {  // embed_norm: round E
        partial_sumsq(CB_BASE, CB_VL, false);
        return;
    }
    first_partial();
}

// after round E: h = bf16(e / rms(e) * gamma_e), then round V's partial
ALWI void seg_e() {
    {
        DeviceZoneScopedN("t-wait");
        cb_wait_front(CB_RESF, R_E + 1);
    }
    cb_wait_front(CB_GE, T);
    cb_reserve_back(CB_HV, T);
    for (uint32_t j = 0; j < T; ++j) {
        sec_begin();
        load_scaled(CB_BASE, CB_VL, false, j, R_E);
        cp(CB_GE, j, 1);
        glue::sfpu_init();
        glue::mul(0, 1, 0);
        glue::rne(0);
        glue::clean<1>(0);
        sec_pack();
        pk(0, CB_HV);
        sec_end();
    }
    cb_push_back(CB_HV, T);
    first_partial();
}

// after round V (layer 0): X row 0 = bf16(v / rms(v)), out = in_norm(v) (the same 1 / rms)
ALWI void seg_v() {
    {
        DeviceZoneScopedN("t-wait");
        cb_wait_front(CB_RESF, R_V + 1);
    }
    normed_partial(VHC, R_V);
    outputs<1>(VHC, CB_VL, HAS_LO, R_V);
}

// after round MX: the appended block state; outputs (NG == 0) or the gates' x / nx and round DN's partials
ALWI void seg_mx() {
    {
        DeviceZoneScopedN("t-wait");
        cb_wait_front(CB_RESF, R_SO + 1);
        cb_wait_front(CB_RESB, R_C + 2);
    }
    if (get_common_arg_val<uint32_t>(A_APPEND)) {
        normed_partial(VHC, R_M);
    }
    // ---------------------------------------------------------------- outputs / gates (row-packed: row 4 a + j)
    // O_j = C S_j for every column tile (DEST j), moved to row-packed form (exact): row 4 a + j = O_a of tile j.
    // DEST rules here: an FP32 CB copied with unpack-to-dest (RESF, MSG) goes to a slot no earlier op of the section
    // (nor the previous section's pack) touches: the unpacker writes it whenever it gets there; no pack from DEST 7.
    if constexpr (NG == 0) {  // out_a = bf16(O_a s_O_a gamma_a)
        DeviceZoneScopedN("t-out");
        cb_wait_front(CB_GAM, 1);
        sec_begin();
        mix_all();
        cp(CB_RESF, R_SO, 3);
        glue::sfpu_init();
        glue::to_rp<T>(0, 0);
        glue::sfpu_fence();  // DEST 2 was read by the move
        cp(CB_GAM, 0, 2);
        glue::sfpu_init();
        glue::out<T, NOUT, false>();
        sec_pack();
        pack_to(CB_OUT, 0);
        sec_end();
        return;
    }
    // gate k reads x_k = O_src(k) (GSAME: every gate O_0): row group k of the row-packed O (hi / lo -> CB_ORP); the
    // DN round's nx = bf16(x s_O) as column tiles (row k) -> CB_NX
    {
        DeviceZoneScopedN("t-goh");
        cb_reserve_back(CB_ORP, 2);
        cb_reserve_back(CB_NX, T);
        sec_begin();
        mix_all();
        cp(CB_RESF, R_SO, 3);
        glue::sfpu_init();
        glue::to_rp<T>(0, 0);
        glue::g1<T, NG, GSAME>();
        glue::to_cols<T>(0, 0);
        sec_pack();
        for (uint32_t j = 0; j < T; ++j) {
            pk(j, CB_NX);
        }
        pk(4, CB_ORP);
        pk(5, CB_ORP);
        sec_end();
        cb_push_back(CB_ORP, 2);
        cb_push_back(CB_NX, T);
    }
    cb_wait_front(CB_NX, T);
    cb_wait_front(CB_WD, NG * T * 2);
    cb_wait_front(CB_WM, NG * T * 2);
    cb_wait_front(CB_NM, NG * T);
    for (uint32_t kk = 0; kk < NG; ++kk) {  // round DN partials: nx . Wd_k + nM_k . Wm_k (rows src(k) valid)
        sec_begin();
        mm_begin(CB_NX, CB_WD, 0);
        for (uint32_t j = 0; j < T; ++j) {
            matmul_tiles(CB_NX, CB_WD, j, (kk * T + j) * 2, 0);
            matmul_tiles(CB_NX, CB_WD, j, (kk * T + j) * 2 + 1, 1);
        }
        mm_begin(CB_NM, CB_WM, 0);
        for (uint32_t j = 0; j < T; ++j) {
            matmul_tiles(CB_NM, CB_WM, kk * T + j, (kk * T + j) * 2, 0);
            matmul_tiles(CB_NM, CB_WM, kk * T + j, (kk * T + j) * 2 + 1, 1);
        }
        glue::sfpu_init();
        parts_of(0, 2);
        parts_of(1, 2 + NPT);
        sec_pack();
        pack_parts(2, 2 * NPT);
        sec_end();
    }
}

// master of round DN: g = bf16(down)
ALWI void dn_finish() {
    cb_wait_front(CB_G, (G_DN + N_DN) * NGT);
    sec_begin();
    mm_begin(CB_CONST, CB_G, 0);
    for (uint32_t s = 0; s < N_DN; ++s) {
        red(K_E + s / (2 * NPT), G_DN + s, (s / NPT) & 1);
    }
    glue::sfpu_init();
    glue::rne(0);
    glue::rne(1);
    glue::clean<NG>(0);
    glue::clean<NG>(1);
    sec_pack();
    cb_reserve_back(CB_MRESB, 2);
    pk(0, CB_MRESB);
    pk(1, CB_MRESB);
    cb_push_back(CB_MRESB, 2);
    sec_end();
}

// after round DN: the gate outputs y, round SG's partial
ALWI void seg_dn() {
    {
        DeviceZoneScopedN("t-wait");
        cb_wait_front(CB_RESB, R_DN + 2);
    }
    // g rows per gate: GK tiles (k, c2) = row k of g (rank columns 32 c2 ..), every other row zero
    {
        DeviceZoneScopedN("t-gy");
        cb_reserve_back(CB_GK, 2 * NG);
        sec_begin();
        cp(CB_RESB, R_DN, 6);
        cp(CB_RESB, R_DN + 1, 7);
        glue::sfpu_init();
        glue::gk<NG>();
        sec_pack();
        for (uint32_t i = 0; i < 2 * NG; ++i) {
            pk(i, CB_GK);
        }
        sec_end();
        cb_push_back(CB_GK, 2 * NG);
        // y_k = (1 - tanh(g_k Wf_k)) x_k + tanh(g_k Wg_k) m_k on the row-packed rows 4 k + j (f / g column tiles of
        // every j in DEST 0.. / 3.., then moved; g at 3 for every T: x hi / lo go to 1 / 2) -> y hi / lo row-packed
        // (CB_YR)
        cb_wait_front(CB_GK, 2 * NG);
        cb_wait_front(CB_WF, NG * 2 * T);
        cb_wait_front(CB_WG, NG * 2 * T);
        cb_wait_front(CB_MSG, 1);
        cb_wait_front(CB_ORP, 2);
        cb_reserve_back(CB_YR, 2);
        sec_begin();
        mm_begin(CB_GK, CB_WF, 0);
        for (uint32_t j = 0; j < T; ++j) {
            for (uint32_t i = 0; i < 2 * NG; ++i) {
                matmul_tiles(CB_GK, CB_WF, i, i * T + j, j);
            }
        }
        mm_begin(CB_GK, CB_WG, 0);
        for (uint32_t j = 0; j < T; ++j) {
            for (uint32_t i = 0; i < 2 * NG; ++i) {
                matmul_tiles(CB_GK, CB_WG, i, i * T + j, 3 + j);
            }
        }
        cp(CB_MSG, 0, 6);
        glue::sfpu_init();
        glue::to_rp<T>(0, 0);
        glue::to_rp<T>(3, 3);
        glue::tanh16(0);
        glue::tanh16(3);
        glue::sfpu_fence();  // DEST 1 / 2 were read by the move
        cp(CB_ORP, 0, 1);
        cp(CB_ORP, 1, 2);
        glue::sfpu_init();
        glue::gate_y<T, NG>();
        sec_pack();
        pk(0, CB_YR);
        pk(1, CB_YR);
        sec_end();
        cb_push_back(CB_YR, 2);
    }
    // round SG over the column tiles of y hi / lo (the column-tile partial: bitwise the earlier kernel's)
    cb_wait_front(CB_YR, 2);
    for (uint32_t h = 0; h < 2; ++h) {
        const uint32_t cb = h ? CB_PL : CB_PH;
        cb_reserve_back(cb, T);
        sec_begin();
        cp(CB_YR, h, 0);
        glue::sfpu_init();
        glue::to_cols<T>(0, 1);
        sec_pack();
        for (uint32_t j = 0; j < T; ++j) {
            pk(1 + j, cb);
        }
        sec_end();
        cb_push_back(cb, T);
    }
    cb_wait_front(CB_PH, T);
    cb_wait_front(CB_PL, T);
    partial_sumsq(CB_PH, CB_PL, true);
}

// after round SG: out_k = bf16(y_k s_SG_k gamma_k) (row-packed)
ALWI void seg_sg() {
    {
        DeviceZoneScopedN("t-wait");
        cb_wait_front(CB_RESF, R_SG + 1);
    }
    DeviceZoneScopedN("t-out");
    cb_wait_front(CB_GAM, 1);
    sec_begin();
    cp(CB_YR, 0, 0);
    cp(CB_YR, 1, 1);
    cp(CB_RESF, R_SG, 3);
    cp(CB_GAM, 0, 2);
    glue::sfpu_init();
    glue::out<T, NOUT, true>();
    sec_pack();
    pack_to(CB_OUT, 0);
    sec_end();
}

// segment s of a row: s = 0 the prologue + round 0's partial; s = o + 1: what follows round o (its results), up to
// the next round's partial (s = NRND: the outputs)
ALWI void seg(const uint32_t s) {
    if (s == 0) {
        seg_first();
        return;
    }
    switch (round_kind(s - 1)) {
        case RK_E: seg_e(); break;
        case RK_V: seg_v(); break;
        case RK_MX: seg_mx(); break;
        case RK_MG: msg_out(); break;
        case RK_DN: seg_dn(); break;
        default: seg_sg(); break;
    }
}

// the master's part of round o (its result pages -> CB_MRESF / CB_MRESB; = RESF / RESB except MR)
ALWI void finish(const uint32_t o, const uint32_t NS) {
    switch (round_kind(o)) {
        case RK_E: finish_sumsq(1, G_E); break;
        case RK_V: finish_sumsq(1, G_V); break;
        case RK_MX: mix_finish(NS); break;
        case RK_MG:
            msg_finish(0);
            msg_finish(1);
            break;
        case RK_DN: dn_finish(); break;
        default: finish_sumsq(NG, G_SG); break;
    }
}
}  // namespace gs

// segments in order, each for every row of the group (MR: row q's CB pages selected first); between segment s and
// s + 1 the master of round s finishes it (MR: core q is row q's master; else core 0)
void kernel_main() {
    ink_set_read_vc();
    using namespace gs;
    const uint32_t k = core_index(), n = nrw();
    const uint32_t NS = get_common_arg_val<uint32_t>(A_NS);
    compute_kernel_hw_startup(CB_CONST, CB_CONST, CB_OUT);
    for (uint32_t s = 0; s <= NRND; ++s) {
        for (uint32_t q = 0; q < n; ++q) {
            use_row(q, s, n);
            seg(s);
        }
        if (s < NRND && k < n) {
            finish(s, NS);
        }
    }
}
#endif
