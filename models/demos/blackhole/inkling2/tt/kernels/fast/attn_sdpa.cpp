// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 decode SDPA (fast/attn_decode.py SDPADecodeInk), adapted from the LFM2.5 kernel (lfm2/tt/fast/kernels/
// attn_sdpa.cpp): 24 q heads / 8 kv heads (GQA 3), head_dim 128, BF16 or BFP8 K/V, one sequence, ROWS query rows
// (named arg "rows": 1 = decode, up to 8 = speculative verify; row b at its own position cur_pos[b], so consecutive
// positions give causal attention across the rows), with in-kernel relative-position bias, attention sink and the
// sigmoid output gate.
//
// Caches (attention.py layouts, no page table):
//   RING (sliding): [1, 8, 1152, 128]; slot s holds the latest position == s mod 1088; valid keys = the nvalid =
//                   min(pos+1, 1024) slots ending at pos % 1088 (cyclic; the 64 slots after it are dead), rows
//                   1088.. zero (stock decode's sink, never read here).
//   full:           [nblocks + 1, 8, 64, 128] identity block order; valid keys [0, pos].
// The live keys are [off, off + nvalid) of a 64-key block sequence; ring: virtual block v = physical block (fb + v) %
// 17 (fb / off: block / offset of the oldest valid slot), so the first block may need a suffix mask (keys >= off) and
// the last a prefix mask; full: fb = off = 0.  Key tile (32 keys) tt of kv head g, dim tile d at page ring
// (g * 36 + tt) * 4 + d, full ((tt / 2 * 8 + g) * 2 + tt % 2) * 4 + d.
//
// Scores S = Q_aug K_aug^T: Q_aug = the front's 5 query tiles of the group (rows 0-2 = q heads 3g..3g+2, already
// clipped / scaled by 1/sqrt(128) [* tau]; tile 4 = [r (16) | sink | 0..]), K_aug = [K (4 dim tiles) | Pg]: Pg row
// = proj[:, d] of the key's distance d (< 1024, else 0), cols 16..31 zero; so S = q.k + r.proj[:, d] (the relative
// bias, FP32 in DEST).  Pg tiles are read from the per-layer table TBL (2 copies, the second shifted by one row, so
// every 16-row run starts 64 B aligned; each replicated on the 8 DRAM banks, page = shift * 8 + bank): row i =
// proj[:, 2047 - i] for i in [1024, 2048) and proj[:, 3135 - i] for [2112, 3136) (ring wrap: distance 1088 + b - j),
// zero for i < 1024 / [2048, 2112) (distance >= 1024) and the tail; key j (physical slot) of a row at bias position b
// (ring: pos % 1088, full: pos) uses row 2047 - b + j.  Blocks whose keys all have distance >= 1024 (full layers)
// skip the bias.
// The sink is one more "partition" of the reducer: m = sink logit (bf16 from Q_aug), l = 1, O = 0.
//
// Partitions / two-pass softmax / reduction: as the LFM kernel (8 groups per row; row b's group g is reducer group
// 8 b + g, query tiles (8 b + g) * 5 + 0..4), with O = 4 dim tiles + l (5 gathered FP32 tiles per row); schedule:
// every row split over the cores as a one-row decode splits it (vcore below: verify rows == decode steps, bitwise).
// The reducer also multiplies by sigmoid(gate) (gate row b of the group's heads from the projection output P,
// tiles 96 + 4 h + c) and writes its 3 heads into ROWS == 1: the row-major attention row [1, 1, 1, 3072] BF16 (one
// DRAM page: the out projection reads it with one transfer); else row b of the TILE [1, 1, 32, 3072] output.
//
// Per-core args (all RISCs): [core index c, swap].  Common args: [q, k cache, v cache, cur_pos, table, consts, out,
// cur_pos, P, then (noc x, noc y) of core 0 .. NC - 1] (BATCH: then slot, page table, (noc x, noc y) of front 0..7).
//
// BATCH (batched serving, fast/attn_decode.py AttnDecodeFast.batch; ROWS <= 16): row b is its own sequence at
// cur_pos[b]: sliding keys from slot slot[b]'s ring (pages slot * 1152 ..), full keys through page_table row b (int32
// [32, W] row-major, PT_PAGE = its aligned page bytes; entry j = pool block of positions 64 j ..).  Positions, bias,
// masks and the schedule are the row's own (vcore), so row b == a one-row decode of that sequence, bitwise.  Full
// layers: each partition's page-table entries (<= 16, 2 x 64 B) are read into a double buffer one partition ahead.
// Readiness per row: front g publishes its rows done in its semaphore ROWS_SEM (attn_front.cpp); a segment of row b,
// group g (and the reducer's sink read) starts once that count exceeds b (polled over the NOC), so the SDPA streams
// row b's keys while the fronts still compute later rows.
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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#else
#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace sdpa {
constexpr uint32_t W = get_named_compile_time_arg_val("workers");
constexpr uint32_t NKT = get_named_compile_time_arg_val("gather_kt");    // ceil(4 * (PMAX + 1) / 32)
constexpr uint32_t C = get_named_compile_time_arg_val("chunk_blocks");
constexpr uint32_t PMAX = get_named_compile_time_arg_val("partitions");
constexpr uint32_t NC = get_named_compile_time_arg_val("cores");
constexpr uint32_t RMAX = get_named_compile_time_arg_val("slots");
constexpr uint32_t K0 = get_named_compile_time_arg_val("k0");
constexpr uint32_t MAXPART = get_named_compile_time_arg_val("max_parts");
constexpr bool kRing = get_named_compile_time_arg_val("ring") == 1;
// fused (one program with the projection + front, fast/attn_decode.py): the front core of kv group g raises semaphore
// kReady + g on every SDPA core once its query tiles and K / V cache rows are in DRAM; front 0 raises kGates once
// every projection worker's gate rows are in P.  (Reading older K / V blocks before the front signals was measured
// slower: those reads slow the projection's weight stream, NOTES-attn.md.)
constexpr bool kFused = get_named_compile_time_arg_val("fused") == 1;
constexpr uint32_t kSem0 = get_named_compile_time_arg_val("sem0");    // gather semaphores kSem0 .. kSem0 + RMAX - 1
constexpr uint32_t kReady = get_named_compile_time_arg_val("ready");  // front-ready semaphores kReady .. kReady + 7
constexpr uint32_t kGates = get_named_compile_time_arg_val("gates");  // gate rows in P (front 0 relays the workers')
// gate column 96 + m of the projection sits at physical column kGateStride (m / kGateDiv) + kGateOff + m % kGateDiv of P
// (the projection's column permutation, fast/attn_decode.py: BIG_PERM 8 / 3 / 5, QG_PERM 12 / 6 / 6)
constexpr uint32_t kGateStride = get_named_compile_time_arg_val("gate_stride");
constexpr uint32_t kGateDiv = get_named_compile_time_arg_val("gate_div");
constexpr uint32_t kGateOff = get_named_compile_time_arg_val("gate_off");

constexpr uint32_t kParHdr = 8;
constexpr uint32_t kMailSlot = (16 * (PMAX + 1) + 63) / 64 * 64;
constexpr uint32_t kTableArg = 9;
constexpr uint32_t kBlock = 64;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t ROWS = get_named_compile_time_arg_val("rows");
static_assert(ROWS >= 1 && ROWS <= 16);
constexpr uint32_t kDt = 4;   // head_dim tiles
constexpr uint32_t kNo = 5;   // gathered tiles per partition row: O (4) | l
constexpr uint32_t kBf16 = 2048;
constexpr uint32_t kFp32 = 4096;
constexpr uint32_t kKb = get_named_compile_time_arg_val("k_bytes");  // K / V cache tile bytes (BF16 / BFP8 / BFP4)
constexpr uint32_t kVb = get_named_compile_time_arg_val("v_bytes");
constexpr uint32_t kTblPage = 3136 * 32;  // one table copy (bytes)
constexpr uint32_t kSlotG = kNo * NKT * kFp32;
constexpr uint32_t kOutstanding = get_named_compile_time_arg_val("outstanding");
constexpr uint32_t kRingTiles = 36;    // physical ring rows / 32 (1088 + 64)
constexpr uint32_t kRingSlots = 1088;  // ring modulus
constexpr uint32_t kRingBlocks = kRingSlots / 64;
// consts pages: prefix masks (v live keys) at kMaskPage + 2 (v - 1), suffix masks (o dead keys) at kSufPage + 2 (o - 1)
constexpr uint32_t kOnesPage = 1, kAPage = 2, kMaskPage = 3, kSufPage = kMaskPage + 126, kConstReps = 8;
// CBs
constexpr uint32_t cQ = 0, cK = 1, cV = 2, cMask = 3, cZero = 4, cScaler = 5, cOnes = 6, cParC = 7, cKa = 8,
                   cScratch = 9, cMail = 10, cA = 11, cGate = 12;
constexpr uint32_t cS = 16, cMloc = 17, cM = 18, cP = 19, cO = 20, cG = 21, cOut = 22, cMsend = 23, cMraw = 24,
                   cAf = 25, cMacc = 26, cAlpha = 27, cOacc = 28;
constexpr bool kBatch = BATCH == 1;
constexpr uint32_t kRingSlotPages = kKvHeads * kRingTiles * kDt;  // BATCH: K / V ring pages per slot
// BATCH: reducer slot r of core c reduces group c + r NC, i.e. ONE (row, group); the compute and the writer both run it
// right before the core's first partition of a row > that row + kLag (the fronts publish rows one by one: the earlier
// rows' reductions overlap the later rows instead of all queueing after the last), else at the end
constexpr uint32_t kLag = 1;
FORCE_INLINE uint32_t slot_row(uint32_t c, uint32_t r) {
    const uint32_t grp = c + r * NC;
    return grp < kKvHeads * ROWS ? grp / kKvHeads : ROWS;
}
}  // namespace sdpa

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace sdpa {
struct Row {
    int32_t pos;
    uint32_t n, parts, step, rem, weight, nvalid, bpos, near0;  // near0: first block with a key at distance < 1024
    uint32_t fb, off;  // ring: physical block / in-block offset of the oldest valid key (virtual block 0)
#if BATCH
    uint32_t base;  // sliding: the row's slot's first K / V ring page
#endif
};
FORCE_INLINE void row_work(int32_t pos, Row& r) {
    r.pos = pos;
    r.n = r.parts = r.step = r.rem = r.weight = r.nvalid = r.bpos = r.near0 = r.fb = r.off = 0;
    if (pos < 0) {
        return;
    }
    const uint32_t p = static_cast<uint32_t>(pos);
    if constexpr (kRing) {
        r.nvalid = p + 1 < 1024 ? p + 1 : 1024;
        r.bpos = p % kRingSlots;
        const uint32_t v0 = (r.bpos + kRingSlots + 1 - r.nvalid) % kRingSlots;  // oldest valid slot
        r.fb = v0 / kBlock;
        r.off = v0 % kBlock;
        r.near0 = 0;
    } else {
        r.nvalid = p + 1;
        r.bpos = p;
        r.near0 = p < 1087 ? 0 : (p - 1087) / kBlock + 1;
    }
    r.n = (r.off + r.nvalid + kBlock - 1) / kBlock;
    const uint32_t act = r.n < W ? r.n : W;
    const uint32_t nch = (r.n + act * C - 1) / (act * C);
    r.parts = act * nch;
    r.step = r.n / r.parts;
    r.rem = r.n - r.step * r.parts;
    r.weight = r.n + K0 * r.parts;
}
// Schedule: every row is split over the NC cores exactly as a one-row decode splits it (row b's 8 groups x weight
// range cut into NC equal pieces), core c playing decode core vcore(b, c) for row b (rows rotated over the cores for
// balance).  So each (row, group) has the decode's partitions, segments and segment order: verify rows are bitwise
// equal to sequential decode steps (one shared split of all rows' weight merged partitions differently).
FORCE_INLINE uint32_t vcore(uint32_t b, uint32_t c) { return (c + b * NC / ROWS) % NC; }
FORCE_INLINE void plan(uint32_t pos_l1, Row* rw) {
    for (uint32_t b = 0; b < ROWS; ++b) {
        row_work(reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_l1)[b], rw[b]);
    }
}
template <typename F>
FORCE_INLINE void for_each_part(const Row* rw, uint32_t c, F&& f) {
    for (uint32_t b = 0; b < ROWS; ++b) {
        const Row& r = rw[b];
        if (r.parts == 0) {
            continue;
        }
        const uint32_t total = kKvHeads * r.weight, v = vcore(b, c);
        const uint32_t lo = v * total / NC, hi = (v + 1) * total / NC;
        for (uint32_t g = 0; g < kKvHeads; ++g) {
            const uint32_t gb = g * r.weight;
            if (gb >= hi) {
                break;
            }
            if (gb + r.weight <= lo) {
                continue;
            }
            uint32_t blo = 0, acc = 0;
            bool first = true;
            for (uint32_t p = 0; p < r.parts; ++p) {
                uint32_t bhi = blo + r.step, acc2 = acc + r.rem;
                if (acc2 >= r.parts) {
                    acc2 -= r.parts;
                    ++bhi;
                }
                const uint32_t s = gb + blo + K0 * p;
                if (s >= hi) {
                    break;
                }
                if (s >= lo) {
                    const bool last = p + 1 == r.parts || gb + bhi + K0 * (p + 1) >= hi;
                    f(b, g, p, blo, bhi, first, last);
                    first = false;
                }
                blo = bhi;
                acc = acc2;
            }
        }
    }
}
// 2-level reduction: a core merges its consecutive partitions of one group (a "segment") in compute and sends ONE
// partial; the group's reducer gathers one partial per segment.  Segment index of core c for a (row, group) = the
// number of (virtual, vcore) cores before it with a partition start in the group (increasing weight offsets).
FORCE_INLINE uint32_t core_lo(uint32_t c, uint32_t total) { return c * total / NC; }
FORCE_INLINE uint32_t part_start(const Row& r, uint32_t p) { return p * (r.step + K0) + p * r.rem / r.parts; }
FORCE_INLINE uint32_t core_of(uint32_t x, uint32_t total) {
    uint32_t c = x * NC / total;
    while (c + 1 < NC && core_lo(c + 1, total) <= x) {
        ++c;
    }
    while (c > 0 && core_lo(c, total) > x) {
        --c;
    }
    return c;
}
FORCE_INLINE bool has_part(uint32_t c, uint32_t gb, const Row& r, uint32_t total) {
    const uint32_t lo = core_lo(c, total), hi = core_lo(c + 1, total);
    if (hi <= gb) {
        return false;
    }
    uint32_t a = 0, b = r.parts;  // first p with gb + start(p) >= lo
    while (a < b) {
        const uint32_t m = (a + b) / 2;
        if (gb + part_start(r, m) >= lo) {
            b = m;
        } else {
            a = m + 1;
        }
    }
    return a < r.parts && gb + part_start(r, a) < hi;
}
struct GroupSegs {
    uint32_t index, count;  // segment index of core c (if it has one), segments of the group
};
FORCE_INLINE GroupSegs group_segs(const Row* rw, uint32_t b, uint32_t g, uint32_t c) {
    const Row& r = rw[b];
    const uint32_t total = kKvHeads * r.weight, gb = g * r.weight, v = vcore(b, c);
    GroupSegs s{0, 0};
    const uint32_t c1 = core_of(gb + r.weight - 1, total);
    for (uint32_t cc = core_of(gb, total); cc <= c1; ++cc) {
        if (has_part(cc, gb, r, total)) {
            if (cc < v) {
                ++s.index;
            }
            ++s.count;
        }
    }
    return s;
}
FORCE_INLINE void zero_l1(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + off, bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE);
    }
}
FORCE_INLINE uint32_t fkey(uint32_t u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
FORCE_INLINE auto dram(uint32_t arg, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(arg), page);
}
// physical block of virtual block v of row r
FORCE_INLINE uint32_t phys_block(const Row& r, uint32_t v) {
    if constexpr (kRing) {
        const uint32_t b = r.fb + v;
        return b >= kRingBlocks ? b - kRingBlocks : b;
    } else {
        return v;
    }
}
FORCE_INLINE uint32_t kv_page(uint32_t g, uint32_t tt, uint32_t d) {
    if constexpr (kRing) {
        return (g * kRingTiles + tt) * kDt + d;
    } else {
        return (((tt >> 1) * kKvHeads + g) * 2 + (tt & 1)) * kDt + d;
    }
}
FORCE_INLINE void copy32(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w];
    }
}
}  // namespace sdpa
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace sdpa {
// scratch layout: [0, 64) cur_pos; [64, ...) Row[ROWS]; then per partition info; then 1.5 KB staging (gate rows;
// BATCH sliding: first the slot words); BATCH: then the page-table double buffer (2 x 128 B), the rows-done poll word
constexpr uint32_t kRows = 64;
constexpr uint32_t kInfo = kRows + (ROWS * sizeof(Row) + 63) / 64 * 64;
constexpr uint32_t kStage = kInfo + (4 * MAXPART + 63) / 64 * 64;
constexpr uint32_t kPtBuf = kStage + 3 * kDt * 2 * 64;
constexpr uint32_t kPoll = kPtBuf + 256;
constexpr uint32_t kBatchArg = kTableArg + 2 * NC;  // BATCH common args: slot, page table, front noc coordinates

// BATCH: rows done by front g (its semaphore ROWS_SEM): poll (a 4 B NOC read into l1) until it exceeds b
FORCE_INLINE uint32_t poll_rows(uint32_t g, uint32_t b, uint32_t l1) {
    const uint64_t src = get_noc_addr(get_common_arg_val<uint32_t>(kBatchArg + 2 + 2 * g),
                                      get_common_arg_val<uint32_t>(kBatchArg + 3 + 2 * g), get_semaphore(ROWS_SEM));
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    while (true) {
        noc_async_read(src, l1, 4);
        noc_async_read_barrier();
        const uint32_t n = w[0];
        if (n > b) {
            return n;
        }
    }
}

// gate rows of core c's reducer groups (one row: group = c + r NC) into cGate (slot r: 4 tiles, row h of tile cc =
// P tile 96 + 4 (3g + h) + cc of the group's row); st: 1.5 KB staging.  Fused: once every projection worker's gate
// rows are in P (front 0 relays them: semaphore kGates).
FORCE_INLINE void read_gates(uint32_t c, const Row* rw, uint32_t st) {
    const auto P = dram(8, kBf16);
    if constexpr (kFused) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(kGates)), 1);
    }
    for (uint32_t r = 0; r < RMAX; ++r) {
        const uint32_t grp = c + r * NC;
        if (grp >= kKvHeads * ROWS || rw[grp / kKvHeads].parts == 0) {
            continue;
        }
        const uint32_t g = grp % kKvHeads, b = grp / kKvHeads;
        cb_reserve_back(cGate, kDt * (r + 1));
        const uint32_t gt = get_write_ptr(cGate) + r * kDt * kBf16;
        zero_l1(gt, kDt * kBf16);
        for (uint32_t h = 0; h < 3; ++h) {
            for (uint32_t cc = 0; cc < kDt; ++cc) {
                // gate column 96 + m of the projection at physical column kGateStride (m / kGateDiv) + kGateOff +
                // m % kGateDiv
                const uint32_t m = 4 * (3 * g + h) + cc;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(get_noc_addr(kGateStride * (m / kGateDiv) + kGateOff + m % kGateDiv, P,
                                                f * 512 + (b & ~1u) * 32),
                                   st + ((h * kDt + cc) * 2 + f) * 64, 64);
                }
            }
        }
        noc_async_read_barrier();
        for (uint32_t h = 0; h < 3; ++h) {
            for (uint32_t cc = 0; cc < kDt; ++cc) {
                for (uint32_t f = 0; f < 2; ++f) {
                    copy32(gt + cc * kBf16 + f * 512 + h * 32, st + ((h * kDt + cc) * 2 + f) * 64 + (b & 1u) * 32);
                }
            }
        }
    }
    cb_push_back(cGate, kDt * RMAX);
}

FORCE_INLINE void reader(uint32_t c) {
    const auto q = dram(0, kBf16);
    const auto kc = dram(1, kKb);
    const auto vc = dram(2, kVb);
    const auto cur_pos = dram(3, 4);
    const auto tbl = dram(4, kTblPage);
    const auto consts = dram(5, kBf16);
    const auto P = dram(8, kBf16);
    const uint32_t rep = c % kConstReps;
    const uint32_t scratch = get_write_ptr(cScratch);
    noc_async_read(get_noc_addr(0, cur_pos, 0), scratch, 64);
    if constexpr (kBatch && kRing) {
        noc_async_read(get_noc_addr(0, dram(kBatchArg, 4), 0), scratch + kStage, 64);
    }
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cScaler, ckernel::PoolType::MAX,
                                                             ckernel::ReduceDim::REDUCE_ROW>();
    noc_async_read_barrier();
    Row* rw = reinterpret_cast<Row*>(scratch + kRows);
    plan(scratch, rw);
#if BATCH
    if constexpr (kRing) {
        for (uint32_t b = 0; b < ROWS; ++b) {
            rw[b].base = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr int32_t*>(scratch + kStage)[b]) *
                         kRingSlotPages;
        }
    }
#endif
    // params for compute: [0] partitions, [1 + r] live gather tiles of reducer slot r (0: none), [8 + k] partition k:
    // blocks | masked (last block: prefix mask) << 16 | first << 17 | last << 18 | masked first block (suffix mask)
    // << 19 | far blocks (no bias) << 20
    cb_reserve_back(cParC, 1);
    volatile tt_l1_ptr uint32_t* pc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cParC));
    volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + kInfo);
    uint32_t np = 0;
    for_each_part(rw, c, [&](uint32_t b, uint32_t g, uint32_t, uint32_t blo, uint32_t bhi, bool first,
                             bool last) {
        const Row& r = rw[b];
        const bool masked = bhi == r.n && (r.off + r.nvalid) % kBlock != 0;
        const bool masked_first = blo == 0 && r.off != 0;
        const uint32_t far = r.near0 > blo ? (r.near0 < bhi ? r.near0 - blo : bhi - blo) : 0;
        pc[kParHdr + np] = (bhi - blo) | (masked ? 1u << 16 : 0u) | (first ? 1u << 17 : 0u) |
                           (last ? 1u << 18 : 0u) | (masked_first ? 1u << 19 : 0u) | (far << 20);
        if constexpr (kBatch) {  // the partition's row: compute interleaves the reductions (kLag)
            pc[kParHdr + MAXPART + np] = b;
        }
        info[np] = blo | (g << 16) | (b << 20);
        ++np;
    });
    pc[0] = np;
    bool reducer = false;
    for (uint32_t r = 0; r < RMAX; ++r) {
        const uint32_t grp = c + r * NC;
        const uint32_t parts = grp < kKvHeads * ROWS ? rw[grp / kKvHeads].parts : 0;
        const uint32_t segs = parts ? group_segs(rw, grp / kKvHeads, grp % kKvHeads, c).count : 0;
        pc[1 + r] = segs ? (4 * (segs + 1) + 31) / 32 : 0;
        reducer = reducer || parts > 0;
    }
    cb_push_back(cParC, 1);
    if (np > 0) {
        cb_reserve_back(cZero, 1);
        zero_l1(get_write_ptr(cZero), kBf16);
        cb_reserve_back(cOnes, 1);
        noc_async_read(get_noc_addr(kOnesPage * kConstReps + rep, consts), get_write_ptr(cOnes), kBf16);
        // Pg tiles: faces 1 / 3 stay zero (only faces 0 / 2 are ever written)
        cb_reserve_back(cKa, 2 * (kOutstanding + 1));
        zero_l1(get_write_ptr(cKa), 2 * (kOutstanding + 1) * kBf16);
    }
    if (reducer) {
        cb_reserve_back(cA, NKT);
        for (uint32_t kt = 0; kt < NKT; ++kt) {
            noc_async_read(get_noc_addr(kAPage * kConstReps + rep, consts), get_write_ptr(cA) + kt * kBf16, kBf16);
        }
    }
    // gate rows of the reducer's groups (read_gates): only the reduction needs them: read after the K / V stream (off
    // the start-up path; BATCH: the writer reads them before its first reduction)
    // fused: wait until front g's outputs (query tiles, this step's K / V rows) are in DRAM
    uint32_t ready = 0;
    [[maybe_unused]] uint32_t seen[kKvHeads] = {0, 0, 0, 0, 0, 0, 0, 0};  // BATCH: rows done per front (polled)
    auto wait_ready = [&](uint32_t g, uint32_t b) {  // front g: query tiles / K / V rows (BATCH: of row b) written
        if constexpr (kFused) {
            if constexpr (kBatch) {
                if (seen[g] <= b) {
                    seen[g] = poll_rows(g, b, scratch + kPoll);
                }
            } else if (!((ready >> g) & 1)) {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(kReady + g)), 1);
                ready |= 1u << g;
            }
        }
    };
    auto gates = [&]() {
        if constexpr (!kBatch) {
            read_gates(c, rw, scratch + kStage);
        }
    };
    if (np == 0) {
        noc_async_read_barrier();
        if (reducer) {
            cb_push_back(cA, NKT);
            gates();
        }
        return;
    }
    DeviceZoneScopedN("s-kv");
    // BATCH full layers: partition k's page-table entries [lo, lo + nb) of its row (<= 16 entries: <= 2 x 64 B from
    // the 64 B-aligned entry lo & ~15) into buffer k & 1; partition k + 1's are fetched at the end of partition k
    [[maybe_unused]] auto pt_fetch = [&](uint32_t k) {
        const uint32_t lo = info[k] & 0xFFFF, nb = pc[kParHdr + k] & 0xFFFF;
        const uint32_t a0 = (lo * 4) & ~63u;
        const auto pt = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                       get_common_arg_val<uint32_t>(kBatchArg + 1), PT_PAGE);
        noc_async_read(get_noc_addr(info[k] >> 20, pt, a0), scratch + kPtBuf + (k & 1) * 128,
                       ((lo + nb) * 4 - a0 + 63) & ~63u);
    };
    if constexpr (kBatch && !kRing) {
        pt_fetch(0);  // the first partition starts a segment: its query-tile barrier covers this read
    }
    uint32_t g = 0;
    for (uint32_t k = 0; k < np; ++k) {
        const uint32_t word = pc[kParHdr + k];
        const uint32_t nb = word & 0xFFFF;
        const uint32_t lo_blk = info[k] & 0xFFFF;
        const bool masked = (word >> 16) & 1, first = (word >> 17) & 1, masked_first = (word >> 19) & 1;
        const uint32_t far = word >> 20;
        const Row& row = rw[info[k] >> 20];
        if (first) {  // a new segment: the group's 5 query tiles (row b, group g), once its front is done
            g = (info[k] >> 16) & 0xF;
            wait_ready(g, info[k] >> 20);
            const uint32_t qt = ((info[k] >> 20) * kKvHeads + g) * 5;
            cb_reserve_back(cQ, 5);
            for (uint32_t t = 0; t < 5; ++t) {
                noc_async_read(get_noc_addr(qt + t, q), get_write_ptr(cQ) + t * kBf16, kBf16);
            }
        }
        // mask pairs in consumption order: suffix (first block), prefix (last block)
        const uint32_t nmask = (masked_first ? 2u : 0u) + (masked ? 2u : 0u);
        if (nmask) {
            cb_reserve_back(cMask, nmask);
            const uint32_t m_end = get_local_cb_interface(cMask).fifo_limit;
            const uint32_t m_size = get_local_cb_interface(cMask).fifo_size;
            uint32_t mw = get_write_ptr(cMask);
            for (uint32_t which = 0; which < 2; ++which) {
                if (which == 0 ? !masked_first : !masked) {
                    continue;
                }
                const uint32_t base = which == 0 ? kSufPage + 2 * (row.off - 1)
                                                 : kMaskPage + 2 * ((row.off + row.nvalid - 1) % kBlock);
                for (uint32_t t = 0; t < 2; ++t) {
                    noc_async_read(get_noc_addr((base + t) * kConstReps + rep, consts), mw, kBf16);
                    mw += kBf16;
                    if (mw >= m_end) {
                        mw -= m_size;
                    }
                }
            }
        }
        if (first || nmask) {
            noc_async_read_barrier();
            if (k == 0) {
                cb_push_back(cZero, 1);
                cb_push_back(cOnes, 1);
                if (reducer) {
                    cb_push_back(cA, NKT);
                }
            }
            if (first) {
                cb_push_back(cQ, 5);
            }
            if (nmask) {
                cb_push_back(cMask, nmask);
            }
        }
        // K (+ Pg) then V blocks, kOutstanding blocks in flight.  K tiles to slot d * 2 + t, V tiles to t * 4 + d.
        for (uint32_t which = 0; which < 2; ++which) {
            const uint32_t cb = which == 0 ? cK : cV;
            const auto& acc = which == 0 ? kc : vc;
            const uint32_t ring_end = get_local_cb_interface(cb).fifo_limit;
            const uint32_t ring_size = get_local_cb_interface(cb).fifo_size;
            const uint32_t ka_end = get_local_cb_interface(cKa).fifo_limit;
            const uint32_t ka_size = get_local_cb_interface(cKa).fifo_size;
            uint32_t pending = 0, pend_near = 0, issue_trid = 1, wait_trid = 1;
            reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
            for (uint32_t blk = lo_blk; blk < lo_blk + nb; ++blk) {
                const bool near = which == 0 && blk - lo_blk >= far;
                const uint32_t pb = phys_block(row, blk);  // ring block / logical block (the bias row)
                // K / V pages: sliding BATCH: in the row's slot; full BATCH: the pool block of logical block blk
                uint32_t kvb = pb, base = 0;
#if BATCH
                if constexpr (kRing) {
                    base = row.base;
                } else {
                    kvb = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr int32_t*>(
                        scratch + kPtBuf + (k & 1) * 128)[blk - (lo_blk & ~15u)]);
                }
#endif
                cb_reserve_back(cb, 2 * kDt * (pending + 1));
                const uint32_t tb = which == 0 ? kKb : kVb;
                uint32_t dst = get_write_ptr(cb) + pending * 2 * kDt * tb;
                if (dst >= ring_end) {
                    dst -= ring_size;
                }
                noc_async_read_set_trid(issue_trid);
                for (uint32_t t = 0; t < 2; ++t) {
                    for (uint32_t d = 0; d < kDt; ++d) {
                        const uint32_t slot = which == 0 ? d * 2 + t : t * kDt + d;
                        noc_async_read(get_noc_addr(base + kv_page(g, 2 * kvb + t, d), acc), dst + slot * tb, tb);
                    }
                }
                if (near) {
                    cb_reserve_back(cKa, 2 * (pend_near + 1));
                    uint32_t ka = get_write_ptr(cKa) + pend_near * 2 * kBf16;
                    if (ka >= ka_end) {
                        ka -= ka_size;
                    }
                    for (uint32_t t = 0; t < 2; ++t) {
                        const uint32_t i0 = 2047 - row.bpos + kBlock * pb + 32 * t;
                        const uint32_t page = (i0 & 1) * 8 + rep;  // shift copy (i0 & 1), DRAM-bank replica rep
                        const uint32_t off = (i0 & ~1u) * 32;
                        noc_async_read(get_noc_addr(page, tbl, off), ka + t * kBf16, 512);
                        noc_async_read(get_noc_addr(page, tbl, off + 512), ka + t * kBf16 + 1024, 512);
                    }
                    ++pend_near;
                }
                issue_trid = issue_trid == kOutstanding ? 1 : issue_trid + 1;
                if (++pending == kOutstanding) {
                    noc_async_read_barrier_with_trid(wait_trid);
                    cb_push_back(cb, 2 * kDt);
                    const uint32_t done = blk + 1 - kOutstanding;  // the block just completed
                    if (which == 0 && done - lo_blk >= far) {
                        cb_push_back(cKa, 2);
                        --pend_near;
                    }
                    wait_trid = wait_trid == kOutstanding ? 1 : wait_trid + 1;
                    --pending;
                }
            }
            noc_async_read_set_trid(0);
            if constexpr (kBatch && !kRing) {
                if (which == 1 && k + 1 < np) {
                    pt_fetch(k + 1);  // completes with this partition's last V reads (the barrier below)
                }
            }
            noc_async_read_barrier();
            for (; pending > 0; --pending) {
                cb_push_back(cb, 2 * kDt);
            }
            for (; pend_near > 0; --pend_near) {
                cb_push_back(cKa, 2);
            }
        }
    }
    if (reducer) {
        gates();
    }
}

FORCE_INLINE void writer_zero(uint32_t gather, uint32_t parts, uint32_t nkt) {
    const uint32_t mt = get_write_ptr(cM);
    const uint32_t mraw = get_write_ptr(cMraw);
    cb_reserve_back(cG, kNo * NKT);
    for (uint32_t k0 = 0; k0 < 32 * nkt; k0 += 16) {  // rows >= 4 parts of the live gather tiles
        const uint32_t first = 4 * parts > k0 ? 4 * parts - k0 : 0;
        if (first >= 16) {
            continue;
        }
        const uint32_t kt = k0 / 32, half = (k0 % 32) / 16;
        for (uint32_t cc = 0; cc < kNo; ++cc) {
            for (uint32_t f = 0; f < 2; ++f) {
                zero_l1(gather + (kt * kNo + cc) * kFp32 + (half * 2 + f) * 1024 + first * 64, (16 - first) * 64);
            }
        }
    }
    cb_reserve_back(cMraw, NKT);
    zero_l1(mraw, nkt * kFp32);
    cb_reserve_back(cM, 1);
    zero_l1(mt, kFp32);
    noc_async_read_barrier();
    for (uint32_t kt = 0; kt < nkt; ++kt) {
        for (uint32_t f = 0; f < 2; ++f) {
            volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mraw + kt * kFp32 + f * 1024);
            for (uint32_t w = 0; w < 64; ++w) {
                row[w] = 0xF149F2CAu;
            }
        }
    }
}

template <typename Out>
FORCE_INLINE void writer_reduce(uint32_t r, uint32_t b, uint32_t g, uint32_t parts, uint32_t mail, uint32_t gather,
                                const Out& out, uint32_t sinks) {
    const uint32_t mt = get_write_ptr(cM);
    const uint32_t mraw = get_write_ptr(cMraw);
    const uint32_t nkt = (4 * (parts + 1) + 31) / 32;
    {
        DeviceZoneScopedN("s-wz");
        writer_zero(gather, parts, nkt);
    }
    // the sink partition (index parts): m = sink logit (rows 0-2; row 3 never wins), O = 0, l = 1
    {
        const uint32_t k = 4 * parts, kt = k / 32, kk = k % 32;
        for (uint32_t rr = 0; rr < 4; ++rr) {
            const uint32_t row = kk + rr;
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* l = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    gather + (kt * kNo + 4) * kFp32 + ((row >> 4) * 2 + f) * 1024 + (row & 15) * 64);
                for (uint32_t w = 0; w < 16; ++w) {
                    l[w] = 0x3F800000u;
                }
            }
        }
    }
    {
        DeviceZoneScopedN("s-gather-wait");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(kSem0 + r));
        noc_semaphore_wait(s, parts);
        noc_semaphore_set(s, 0);
    }
    volatile tt_l1_ptr uint32_t* mail_v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mail);
    for (uint32_t rr = 0; rr < 4; ++rr) {
        const uint32_t sink = rr < 3 ? static_cast<uint32_t>(
                                           reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sinks + rr * 32)[0]) << 16
                                     : 0xF149F2CAu;
        mail_v[4 * parts + rr] = sink;
        uint32_t best = mail_v[rr], bk = fkey(best);
        for (uint32_t p = 0; p <= parts; ++p) {
            const uint32_t v = mail_v[4 * p + rr], k = fkey(v);
            if (k > bk) {
                best = v;
                bk = k;
            }
            const uint32_t col = 4 * p + rr, kt = col / 32, kk = col % 32;
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mraw + kt * kFp32 + (kk >> 4) * 1024 + rr * 64)[kk & 15] = v;
        }
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mt + rr * 64)[0] = best;
    }
    cb_push_back(cM, 1);
    cb_push_back(cMraw, NKT);
    cb_push_back(cG, kNo * NKT);
    // out rows 0-2 (q heads 3g + rr): ROWS == 1: the ROW-MAJOR attention row [3072] (head h at [128 h, 128 h + 128)),
    // the o_proj's x_rm input; else row b of the TILE output (tile (3 g + rr) * 4 + cc): 2 x 32 B per (head, dim tile)
    DeviceZoneScopedN("s-wout");
    cb_wait_front(cOut, kDt);
    const uint32_t src = get_read_ptr(cOut);
    for (uint32_t rr = 0; rr < 3; ++rr) {
        for (uint32_t cc = 0; cc < kDt; ++cc) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint64_t dst = ROWS == 1 ? get_noc_addr(0, out, 2 * (128 * (3 * g + rr) + 32 * cc + 16 * f))
                                               : get_noc_addr((3 * g + rr) * kDt + cc, out, f * 512 + b * 32);
                noc_async_write(src + cc * kBf16 + f * 512 + rr * 32, dst, 32);
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cOut, kDt);
}

FORCE_INLINE void writer(uint32_t c) {
    const auto cur_pos = dram(7, 4);
    const auto q = dram(0, kBf16);
    constexpr auto out_args = TensorAccessorArgs<0>();
    // ROWS == 1: one row-major page; else BF16 tiles
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(6), ROWS == 1 ? 3072 * 2 : kBf16);
    // mail: RMAX mailbox slots, then [pos_box 64, stage 64, Row[ROWS], sinks 128 per slot, BATCH: gate staging 1.5 KB]
    const uint32_t mail = get_write_ptr(cMail);
    const uint32_t pos_box = mail + RMAX * kMailSlot;
    const uint32_t stage = pos_box + 64;
    const uint32_t sinks = stage + 64 + (ROWS * sizeof(Row) + 63) / 64 * 64;
    const uint32_t gather = get_write_ptr(cG);
    noc_async_read(get_noc_addr(0, cur_pos, 0), pos_box, 64);
    noc_async_read_barrier();
    Row* rw = reinterpret_cast<Row*>(stage + 64);
    plan(pos_box, rw);
    // sink logits of this core's reducer groups: rows 0-3 of face 1 of query tile 4 (128 B; fused: once the group's
    // front has written its query tiles; BATCH: right before the slot's reduction, so waiting for a late row's front
    // never holds back the earlier rows' partials)
    auto sink_read = [&](uint32_t r) {
        const uint32_t grp = c + r * NC;
        if (grp < kKvHeads * ROWS && rw[grp / kKvHeads].parts > 0) {
            if constexpr (kBatch) {
                poll_rows(grp % kKvHeads, grp / kKvHeads, stage);
            } else if constexpr (kFused) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(kReady + grp % kKvHeads)), 1);
            }
            noc_async_read(get_noc_addr(grp * 5 + 4, q, 512), sinks + r * 128, 128);
        }
    };
    if constexpr (!kBatch) {
        for (uint32_t r = 0; r < RMAX; ++r) {
            sink_read(r);
        }
    }
    // reduction of slot r (BATCH: the gate rows before the first one: read while the writer waits for a partial once
    // the projection has written them, else right there; its sink logits right before it)
    [[maybe_unused]] bool gated = false;
    [[maybe_unused]] auto gates_now = [&]() {
        read_gates(c, rw, sinks + RMAX * 128);
        gated = true;
    };
    auto reduce_slot = [&](uint32_t r) {
        const uint32_t grp = c + r * NC;
        const uint32_t b = grp / kKvHeads;
        if (grp < kKvHeads * ROWS && rw[b].parts > 0) {
            if constexpr (kBatch) {
                if (!gated) {
                    gates_now();
                }
                sink_read(r);
            }
            noc_async_read_barrier();
            writer_reduce(r, b, grp % kKvHeads, group_segs(rw, b, grp % kKvHeads, c).count, mail + r * kMailSlot,
                          gather + r * kSlotG, out, sinks + r * 128);
        } else {
            cb_reserve_back(cG, kNo * NKT);
            cb_push_back(cG, kNo * NKT);
        }
    };
    uint32_t next_r = 0;
    for_each_part(rw, c, [&](uint32_t b, uint32_t g, uint32_t, uint32_t, uint32_t, bool, bool last) {
        if constexpr (kBatch) {  // the same points as the compute (kLag)
            while (next_r < RMAX && b > slot_row(c, next_r) + kLag) {
                reduce_slot(next_r++);
            }
        }
        if (!last) {  // merged into the segment's partial in compute
            return;
        }
        const uint32_t p = group_segs(rw, b, g, c).index;
        const uint32_t grp = b * kKvHeads + g;
        const uint32_t red = grp % NC, r = grp / NC;
        const uint32_t red_x = get_common_arg_val<uint32_t>(kTableArg + 2 * red);
        const uint32_t red_y = get_common_arg_val<uint32_t>(kTableArg + 2 * red + 1);
        if constexpr (kBatch) {
            volatile tt_l1_ptr uint32_t* gs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(kGates));
            while (!gated && !cb_pages_available_at_front(cMsend, 1)) {
                if (*gs >= 1) {
                    gates_now();
                }
            }
        }
        cb_wait_front(cMsend, 1);
        {
            const uint32_t m = get_read_ptr(cMsend);
            volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
            for (uint32_t rr = 0; rr < 4; ++rr) {
                st[rr] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m + rr * 64)[0];
            }
            cb_pop_front(cMsend, 1);
        }
        cb_wait_front(cO, kNo);
        DeviceZoneScopedN("s-send");
        noc_async_write(stage, get_noc_addr(red_x, red_y, mail + r * kMailSlot + 16 * p), 16);
        const uint32_t o = get_read_ptr(cO);
        const uint32_t k = 4 * p, kt = k / 32, kk = k % 32;
        const uint32_t slot_g = gather + r * kSlotG;
        for (uint32_t cc = 0; cc < kNo; ++cc) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t dst = slot_g + (kt * kNo + cc) * kFp32 + ((kk >> 4) * 2 + f) * 1024 + (kk & 15) * 64;
                noc_async_write(o + cc * kFp32 + f * 1024, get_noc_addr(red_x, red_y, dst), 256);
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(red_x, red_y, get_semaphore(kSem0 + r)), 1);
        noc_async_atomic_barrier();
        cb_pop_front(cO, kNo);
    });
    while (next_r < RMAX) {
        reduce_slot(next_r++);
    }
}
}  // namespace sdpa
#endif

#if defined(COMPILE_FOR_TRISC)
// rows 0-3 of faces 0 / 1 only (VectorMode::R, 2 iterations per face): the reducer's live rows (3 heads + pad)
#define SD_ROWS_RECIP(idst)                                                                                           \
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_reciprocal, (APPROX, DST_ACCUM_MODE, 2, true), idst, \
                         VectorMode::R))
#define SD_ROWS_SIGMOID(idst) \
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sigmoid, (false, DST_ACCUM_MODE, 2), idst, VectorMode::R))
#define SD_ROWS_MUL(a, b, o)                                                                                    \
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,                             \
                           (APPROX, ckernel::BinaryOp::MUL, 2, DST_ACCUM_MODE), a, b, o, VectorMode::R)))
namespace sdpa {
FORCE_INLINE void compute_sdpa() {
    cb_wait_front(cParC, 1);
    // the params page read from L1 by every TRISC (one mailbox transfer of its address)
    const volatile uint32_t* pcw = reinterpret_cast<const volatile uint32_t*>(get_tile_address(cParC, 0));
    const uint32_t np = pcw[0];
    uint32_t nkts[RMAX];
    for (uint32_t r = 0; r < RMAX; ++r) {
        nkts[r] = pcw[1 + r];
    }
    // reducer slots: A[r, 4p + r] = exp(m_p[r] - M[r]), [O | l] = A G, out = O / l * sigmoid(gate)
    bool have_a = false;
    auto reduce = [&](uint32_t r) {
        const uint32_t nkt = nkts[r];
        if (nkt == 0) {
            cb_wait_front(cG, kNo * NKT);
            cb_pop_front(cG, kNo * NKT);
            return;
        }
        DeviceZoneScopedN("s-reduce");
        if (!have_a) {
            cb_wait_front(cA, NKT);
            cb_wait_front(cGate, kDt * RMAX);
            have_a = true;
        }
        cb_wait_front(cMraw, NKT);
        cb_wait_front(cM, 1);
        for (uint32_t kt = 0; kt < nkt; ++kt) {
            reconfig_data_format(cMraw, cM);
            pack_reconfig_data_format(cAf);
            sub_bcast_cols_init_short(cMraw, cM);
            tile_regs_acquire();
            sub_tiles_bcast_cols(cMraw, cM, kt, 0, 0);
            exp_tile_init<false>();
            exp_tile<false, false, InputClamping::ClampToNegative, 2>(0, VectorMode::R);
            reconfig_data_format_srca(cMraw, cA);
            copy_tile_init(cA);
            copy_tile(cA, kt, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            cb_reserve_back(cAf, 1);
            tile_regs_wait();
            pack_tile(0, cAf, 0);
            tile_regs_release();
            cb_push_back(cAf, 1);
        }
        cb_wait_front(cAf, nkt);
        cb_wait_front(cG, kNo * NKT);
        reconfig_data_format(cG, cAf);
        pack_reconfig_data_format(cOut);
        matmul_block_init(cAf, cG, 0, kNo, 1, NKT);
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < nkt; ++kt) {
            matmul_block(cAf, cG, kt, kt * kNo, 0, 0, kNo, 1, NKT);
        }
        // only rows 0-3 (the group's 3 heads + the pad row) are ever written out: BATCH runs 1 / l and * sigmoid(gate)
        // on those rows only (ITERATIONS 2 of faces 0 / 1: the same per-element ops as the whole-tile calls)
        if constexpr (kBatch) {
            recip_tile_init();
            SD_ROWS_RECIP(kDt);
            mul_binary_tile_init();
            for (uint32_t cc = 0; cc < kDt; ++cc) {
                SD_ROWS_MUL(cc, kDt, cc);
            }
            reconfig_data_format_srca(cG, cGate);
            for (uint32_t cc = 0; cc < kDt; ++cc) {
                copy_tile_init(cGate);
                copy_tile(cGate, r * kDt + cc, 5);
                sigmoid_tile_init();
                SD_ROWS_SIGMOID(5);
                mul_binary_tile_init();
                SD_ROWS_MUL(cc, 5, cc);
            }
        } else {
        recip_tile_init();
        recip_tile(kDt);
        mul_binary_tile_init();
        for (uint32_t cc = 0; cc < kDt; ++cc) {
            mul_binary_tile(cc, kDt, cc);
        }
        // * sigmoid(gate)
        reconfig_data_format_srca(cG, cGate);
        for (uint32_t cc = 0; cc < kDt; ++cc) {
            copy_tile_init(cGate);
            copy_tile(cGate, r * kDt + cc, 5);
            sigmoid_tile_init();
            sigmoid_tile<VectorMode::R>(5);
            mul_binary_tile_init();
            mul_binary_tile(cc, 5, cc);
        }
        }
        tile_regs_commit();
        cb_reserve_back(cOut, kDt);
        pack_reconfig_data_format(cOut);
        tile_regs_wait();
        for (uint32_t cc = 0; cc < kDt; ++cc) {
            pack_tile(cc, cOut, cc);
        }
        tile_regs_release();
        cb_push_back(cOut, kDt);
        cb_pop_front(cAf, nkt);
        if (nkt < NKT) {
            cb_reserve_back(cAf, NKT - nkt);
            cb_push_back(cAf, NKT - nkt);
            cb_wait_front(cAf, NKT - nkt);
            cb_pop_front(cAf, NKT - nkt);
        }
        cb_pop_front(cMraw, NKT);
        cb_pop_front(cM, 1);
        cb_pop_front(cG, kNo * NKT);
    };
    [[maybe_unused]] const uint32_t c = get_arg_val<uint32_t>(0);
    uint32_t next_r = 0;
    for (uint32_t k = 0; k < np; ++k) {
        if constexpr (kBatch) {  // the same points as the writer (kLag)
            const uint32_t b = pcw[kParHdr + MAXPART + k];
            while (next_r < RMAX && b > slot_row(c, next_r) + kLag) {
                reduce(next_r++);
            }
        }
        const uint32_t word = pcw[kParHdr + k];
        const uint32_t nb = word & 0xFFFF;
        const bool mask_last = (word >> 16) & 1, mask_first = (word >> 19) & 1;
        const bool seg_first = (word >> 17) & 1, seg_last = (word >> 18) & 1;
        const uint32_t far = word >> 20;
        if ((word >> 17) & 1) {
            cb_wait_front(cQ, 5);
        }
        const uint32_t nt = 2 * nb;
        // pass 1: S = Q_aug K_aug^T
        {
            DeviceZoneScopedN("s-qk");
            reconfig_data_format(cK, cQ);
            pack_reconfig_data_format(cS);
            matmul_block_init(cQ, cK, 1, 2, 1, 5);
            for (uint32_t blk = 0; blk < nb; ++blk) {
                cb_wait_front(cK, 2 * kDt);
                tile_regs_acquire();
                for (uint32_t d = 0; d < kDt; ++d) {
                    matmul_block(cQ, cK, d, d * 2, 0, 1, 2, 1, 5);
                }
                const bool near = blk >= far;
                if (near) {
                    cb_wait_front(cKa, 2);
                    reconfig_data_format(cKa, cQ);
                    matmul_block_init(cQ, cKa, 1, 2, 1, 5);
                    matmul_block(cQ, cKa, 4, 0, 0, 1, 2, 1, 5);
                }
                // ring wrap: suffix mask on the first block, prefix mask on the last (pairs in this order)
                const bool mfirst = mask_first && blk == 0, mlast = mask_last && blk == nb - 1;
                if (mfirst || mlast) {
                    cb_wait_front(cZero, 1);
                    reconfig_data_format(cZero, cMask);
                    add_tiles_init(cZero, cMask, true);
                    for (uint32_t m = 0; m < (mfirst && mlast ? 2u : 1u); ++m) {
                        cb_wait_front(cMask, 2);
                        add_tiles(cZero, cMask, 0, 0, 0);
                        add_tiles(cZero, cMask, 0, 1, 1);
                        cb_pop_front(cMask, 2);
                    }
                }
                if (near || mfirst || mlast) {
                    reconfig_data_format(cK, cQ);
                    matmul_block_init(cQ, cK, 1, 2, 1, 5);
                }
                tile_regs_commit();
                cb_reserve_back(cS, 2);
                tile_regs_wait();
                pack_tile(0, cS, 0);
                pack_tile(1, cS, 1);
                tile_regs_release();
                cb_push_back(cS, 2);
                cb_pop_front(cK, 2 * kDt);
                if (near) {
                    cb_pop_front(cKa, 2);
                }
            }
        }
        // local row max m, merged with the segment's running max: M' = max(M, m), alpha = exp(M - M')
        {
            cb_wait_front(cS, nt);
            cb_wait_front(cScaler, 1);
            reconfig_data_format(cS, cScaler);
            pack_reconfig_data_format(cMloc);
            tile_regs_acquire();
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cS, cScaler, cMloc);
            for (uint32_t j = 0; j < nt; ++j) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cS, cScaler, j, 0, 0);
            }
            reduce_uninit();
            if (!seg_first) {
                cb_wait_front(cMacc, 1);
                reconfig_data_format_srca(cS, cMacc);
                copy_tile_init(cMacc);
                copy_tile(cMacc, 0, 1);
                copy_tile(cMacc, 0, 2);
                binary_max_tile_init();
                binary_max_tile(0, 1, 0);
                sub_binary_tile_init();
                sub_binary_tile(2, 0, 2);
                exp_tile_init<false>();
                exp_tile<false, false, InputClamping::ClampToNegative, EXP_ITERS>(2, VectorMode::R);
            }
            tile_regs_commit();
            if (!seg_first) {
                cb_pop_front(cMacc, 1);
                cb_reserve_back(cAlpha, 1);
            }
            const uint32_t mcb = seg_last ? cMsend : cMacc;
            cb_reserve_back(cMloc, 1);
            cb_reserve_back(mcb, 1);
            tile_regs_wait();
            pack_tile(0, cMloc, 0);
            pack_tile(0, mcb, 0);
            if (!seg_first) {
                pack_tile(2, cAlpha, 0);
            }
            tile_regs_release();
            cb_push_back(cMloc, 1);
            cb_push_back(mcb, 1);
            if (!seg_first) {
                cb_push_back(cAlpha, 1);
            }
        }
        // pass 2: P = exp(S - m)
        {
            cb_wait_front(cMloc, 1);
            reconfig_data_format(cS, cMloc);
            pack_reconfig_data_format(cP);
            sub_bcast_cols_init_short(cS, cMloc);
            exp_tile_init<false>();
            PACK((llk_pack_relu_config(ReluConfig::zero())));
            for (uint32_t j = 0; j < nt; j += 4) {
                const uint32_t cnt = nt - j < 4 ? nt - j : 4;
                tile_regs_acquire();
                for (uint32_t jj = 0; jj < cnt; ++jj) {
                    sub_tiles_bcast_cols(cS, cMloc, j + jj, 0, jj);
                    exp_tile<false, false, InputClamping::ClampToNegative, EXP_ITERS>(jj, VectorMode::R);  // rows 0-3
                }
                tile_regs_commit();
                cb_reserve_back(cP, cnt);
                tile_regs_wait();
                for (uint32_t jj = 0; jj < cnt; ++jj) {
                    pack_tile(jj, cP, jj);
                }
                tile_regs_release();
                cb_push_back(cP, cnt);
            }
            PACK((llk_pack_relu_config(ReluConfig::none())));
            cb_pop_front(cS, nt);
            cb_pop_front(cMloc, 1);
        }
        // O = alpha O_acc + P V (DEST 0..3), l = alpha l_acc + P 1 (DEST 4)
        {
            DeviceZoneScopedN("s-pv");
            cb_wait_front(cP, nt);
            tile_regs_acquire();
            if (!seg_first) {
                cb_wait_front(cOacc, kNo);
                cb_wait_front(cAlpha, 1);
                reconfig_data_format(cOacc, cAlpha);
                mul_bcast_cols_init_short(cOacc, cAlpha);
                for (uint32_t cc = 0; cc < kNo; ++cc) {
                    mul_tiles_bcast_cols(cOacc, cAlpha, cc, 0, cc);
                }
            }
            reconfig_data_format(cV, cP);
            matmul_block_init(cP, cV, 0, kDt, 1, nt);
            for (uint32_t blk = 0; blk < nb; ++blk) {
                cb_wait_front(cV, 2 * kDt);
                for (uint32_t t = 0; t < 2; ++t) {
                    matmul_block(cP, cV, 2 * blk + t, t * kDt, 0, 0, kDt, 1, nt);
                }
                cb_pop_front(cV, 2 * kDt);
            }
            cb_wait_front(cOnes, 1);
            reconfig_data_format(cOnes, cP);
            matmul_block_init(cP, cOnes, 0, 1, 1, nt);
            for (uint32_t j = 0; j < nt; ++j) {
                matmul_block(cP, cOnes, j, 0, kDt, 0, 1, 1, nt);
            }
            tile_regs_commit();
            if (!seg_first) {
                cb_pop_front(cOacc, kNo);
                cb_pop_front(cAlpha, 1);
            }
            const uint32_t ocb = seg_last ? cO : cOacc;
            cb_reserve_back(ocb, kNo);
            pack_reconfig_data_format(ocb);
            tile_regs_wait();
            for (uint32_t cc = 0; cc < kNo; ++cc) {
                pack_tile(cc, ocb, cc);
            }
            tile_regs_release();
            cb_push_back(ocb, kNo);
            cb_pop_front(cP, nt);
        }
        if (k + 1 < np && nt < 2 * C) {
            for (const uint32_t cb : {cS, cP}) {
                cb_reserve_back(cb, 2 * C - nt);
                cb_push_back(cb, 2 * C - nt);
                cb_wait_front(cb, 2 * C - nt);
                cb_pop_front(cb, 2 * C - nt);
            }
        }
        if ((word >> 18) & 1) {
            cb_pop_front(cQ, 5);
        }
    }
    cb_pop_front(cParC, 1);
    while (next_r < RMAX) {
        reduce(next_r++);
    }
    if (have_a) {
        cb_pop_front(cA, NKT);
        cb_pop_front(cGate, kDt * RMAX);
    }
}
}  // namespace sdpa
#endif

void kernel_main() {
    ink_set_read_vc();
    using namespace sdpa;
    const uint32_t c = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    const bool swap = get_arg_val<uint32_t>(1) != 0;
#if defined(COMPILE_FOR_NCRISC)
    if (swap) {
        writer(c);
    } else {
        reader(c);
    }
#else
    if (swap) {
        reader(c);
    } else {
        writer(c);
    }
#endif
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cQ, cK, cS);
    compute_sdpa();
#endif
}
