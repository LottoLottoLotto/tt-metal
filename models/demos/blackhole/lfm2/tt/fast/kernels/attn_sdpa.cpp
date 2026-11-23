// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 paged SDPA decode for head_dim 64, 32 q heads / 8 kv heads (GQA 4), BFP8 paged K/V (block 64), causal at
// cur_pos (attn_fused.SDPADecode), for 1..16 batch rows on every core of the grid.
//
// Partitions (per group = (batch row b, kv head g); depend only on the row's own position, so a row's result does not
// depend on the batch it runs in): the n = cur_pos/64 + 1 live blocks are split into parts = a * nch contiguous
// partitions (a = min(W, n), nch = ceil(n / (a * C)), C = `chunk_blocks`: what one core holds in L1); partition p =
// blocks [p*n/parts, (p+1)*n/parts).  Rows with cur_pos < 0 have no partitions.
//
// Schedule: the partitions of all groups, in (b, g, p) order, weighted by blocks + K0 (per-partition overhead), are cut
// into NC contiguous ranges of equal weight (core c: the partitions whose weighted start lies in [c T / NC,
// (c + 1) T / NC)), so idle rows cost nothing and short and long rows share the cores.  Every core derives the
// whole schedule from cur_pos (identical arithmetic on every core).  The reducer of group G is core G % NC (slot
// G / NC: at most RMAX groups per core).
//
// Per partition, an exact two-pass softmax over its blocks (no online rescaling, no inter-core barrier):
//   pass 1  S = Q K^T per block (Q: the group's 4 q heads in rows 0-3 of two [32, 32] tiles, zero rows 4-31; FP32 S
//           kept in L1), the last live block's tokens > cur_pos masked (-1e9); m_p = rowmax(S) (FP32).
//   pass 2  P = exp((S - m_p) * scale) (rows 0-3 only, BF16), O_p = P V and l_p = P 1 (FP32 DEST over the
//           partition's blocks).
//   reduce  every partition's m_p (16 B, the reducer's mailbox slot p) and rows 0-3 of [O_p | l_p] (rows 4p..4p+3 of
//           the reducer's gather tiles G of that group); the reducer takes M = max_p m_p (exact), forms
//           A[r, 4p + r] = exp((m_p[r] - M[r]) * scale) (accurate SFPU exp, FP32), [O | l] = A G (a fixed-order sum
//           done by the matmul unit) and out = O / l, written as rows (4g + r) of the [1, 1, B, 2048] output
//           (head-major: the flattened SDPA output the out projection takes).
// Per-core roles: the KV reader runs on NCRISC / NOC0, or (swap) on BRISC / NOC1 for the grid rows NOC0 serves
// poorly; the writer (partition results, reductions, output) on the other one.
//
// Per-core args (all RISCs): [core index c, swap].
// Common args (both data-movement RISCs): [q, k cache, v cache, cur_pos, page table, consts, out, cur_pos, then
// (noc x, noc y) of core 0 .. NC - 1].
#include <cstdint>
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
constexpr uint32_t W = get_named_compile_time_arg_val("workers");   // max partitions per group before chunking
constexpr uint32_t NKT = get_named_compile_time_arg_val("gather_kt");  // gather K tiles = ceil(4 * PMAX / 32)
constexpr uint32_t ROWS = get_named_compile_time_arg_val("rows");    // batch rows B (<= 16)
constexpr uint32_t C = get_named_compile_time_arg_val("chunk_blocks");  // max blocks per partition (L1 budget)
constexpr uint32_t PMAX = get_named_compile_time_arg_val("partitions");  // max partitions per group (mailbox)
constexpr uint32_t NC = get_named_compile_time_arg_val("cores");     // cores of the schedule
constexpr uint32_t RMAX = get_named_compile_time_arg_val("slots");   // reducer slots per core = ceil(8 ROWS / NC)
constexpr uint32_t K0 = get_named_compile_time_arg_val("k0");        // per-partition overhead in block units
constexpr uint32_t MAXPART = get_named_compile_time_arg_val("max_parts");  // partitions per core (params page)
constexpr uint32_t kParHdr = 8;  // params page: [0] partitions, [1 + r] live gather tiles of slot r, [8 + k] partition k
constexpr uint32_t kMailSlot = (16 * PMAX + 63) / 64 * 64;  // one reducer slot's mailbox (m_p of every partition)
constexpr uint32_t kTableArg = 8;  // common args: noc (x, y) of core c at kTableArg + 2 c
constexpr uint32_t kBlock = 64;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kBfp8 = 1088;
constexpr uint32_t kBf16 = 2048;
constexpr uint32_t kFp32 = 4096;
constexpr uint32_t kSlotG = 3 * NKT * kFp32;  // one reducer slot's gather tiles
constexpr uint32_t kOutstanding = get_named_compile_time_arg_val("outstanding");  // KV blocks in flight per reader
constexpr uint32_t kRingK = 4 * (kOutstanding + 1);                        // K ring tiles
constexpr uint32_t kRingV = get_named_compile_time_arg_val("v_tiles");     // V CB tiles (one partition's blocks)
// host-built consts tensor pages (BF16 tiles): (zeros, unused), ones, A (A[r, k] = (r < 4 && k % 4 == r)), then the
// mask pair for v = 1..63 live tokens of the last block (token column j of tile t: 0 if 32 t + j < v else -1e9)
// (each page replicated kConstReps times: page p, replica r at p * kConstReps + r)
constexpr uint32_t kOnesPage = 1, kAPage = 2, kMaskPage = 3, kConstReps = 8;
// CBs
constexpr uint32_t cQ = 0, cK = 1, cV = 2, cMask = 3, cZero = 4, cScaler = 5, cOnes = 6, cParC = 7, cScratch = 9,
                   cMail = 10, cA = 11;
constexpr uint32_t cS = 16, cMloc = 17, cM = 18, cP = 19, cO = 20, cG = 21, cOut = 22, cMsend = 23, cMraw = 24,
                   cAf = 25;
}  // namespace sdpa

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace sdpa {
// one batch row's partitioning (the same for its 8 kv-head groups)
struct Row {
    int32_t pos;
    uint32_t n, parts, step, rem, weight;  // live blocks, partitions, n = step * parts + rem, weight n + K0 parts
};
FORCE_INLINE void row_work(int32_t pos, Row& r) {
    r.pos = pos;
    r.n = r.parts = r.step = r.rem = r.weight = 0;
    if (pos < 0) {
        return;
    }
    r.n = static_cast<uint32_t>(pos) / kBlock + 1;
    const uint32_t act = r.n < W ? r.n : W;
    const uint32_t nch = (r.n + act * C - 1) / (act * C);
    r.parts = act * nch;
    r.step = r.n / r.parts;
    r.rem = r.n - r.step * r.parts;
    r.weight = r.n + K0 * r.parts;
}
// every row's partitioning (cur_pos words at pos_l1) and this core's share [lo, hi) of the total weight
FORCE_INLINE void plan(uint32_t pos_l1, uint32_t c, Row* rw, uint32_t& lo, uint32_t& hi) {
    uint32_t total = 0;
    for (uint32_t b = 0; b < ROWS; ++b) {
        row_work(reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_l1)[b], rw[b]);
        total += kKvHeads * rw[b].weight;
    }
    lo = c * total / NC;
    hi = (c + 1) * total / NC;
}
// f(b, g, p, first block, end block, first of the group on this core, last of the group on this core) for every
// partition whose weighted start lies in [lo, hi), in (b, g, p) order.  Partition bounds p * n / parts are
// stepped incrementally (no division per partition).
template <typename F>
FORCE_INLINE void for_each_part(const Row* rw, uint32_t lo, uint32_t hi, F&& f) {
    uint32_t base = 0;
    for (uint32_t b = 0; b < ROWS && base < hi; ++b) {
        const Row& r = rw[b];
        const uint32_t wrow = kKvHeads * r.weight;
        if (r.parts == 0 || base + wrow <= lo) {
            base += wrow;
            continue;
        }
        for (uint32_t g = 0; g < kKvHeads; ++g) {
            const uint32_t gb = base + g * r.weight;
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
        base += wrow;
    }
}
// zero L1 [dst, dst + bytes) by local NOC reads of the hardware zero region (bytes % 512 == 0 or < 512)
FORCE_INLINE void zero_l1(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + off, bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE);
    }
}
// order-preserving key of an FP32 bit pattern (unsigned compare == float compare, -0 < +0)
FORCE_INLINE uint32_t fkey(uint32_t u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
}  // namespace sdpa
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace sdpa {
// KV reader: schedule, params for compute, constants; per segment (this core's partitions of one group) Q and the
// page-table slice; per partition its mask and the K / V block stream (the core's NOC: NOC0 on NCRISC, NOC1 on BRISC)
FORCE_INLINE void reader(uint32_t c) {
    constexpr uint32_t pt_page = get_named_compile_time_arg_val("pt_page");
    constexpr uint32_t pos_page = get_named_compile_time_arg_val("pos_page");
    // scratch: [0, 64) cur_pos; [kPt, kPt + pt_page + 64) page-table slice; [kRows, ...) Row[ROWS]; [kInfo, ...)
    // per partition (first block | g << 16 | b << 20)
    constexpr uint32_t kPt = 64;
    constexpr uint32_t kRows = kPt + (pt_page + 127) / 64 * 64;
    constexpr uint32_t kInfo = kRows + (ROWS * sizeof(Row) + 63) / 64 * 64;
    const auto q = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0),
                                  kBf16);
    const auto kc = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1),
                                   kBfp8);
    const auto vc = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2),
                                   kBfp8);
    const auto cur_pos = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(3), pos_page);
    const auto page_table = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                           get_common_arg_val<uint32_t>(4), pt_page);
    const auto consts = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                       get_common_arg_val<uint32_t>(5), kBf16);
    const uint32_t rep = c % kConstReps;  // consts replica (spreads the DRAM banks)
    const uint32_t scratch = get_write_ptr(cScratch);
    noc_async_read(get_noc_addr(0, cur_pos, 0), scratch, 64);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cScaler, ckernel::PoolType::MAX,
                                                             ckernel::ReduceDim::REDUCE_ROW>();
    noc_async_read_barrier();
    Row* rw = reinterpret_cast<Row*>(scratch + kRows);
    uint32_t lo, hi;
    plan(scratch, c, rw, lo, hi);
    // params for compute: [0] partitions, [1 + r] live gather tiles of reducer slot r (0: none), [8 + k] partition k:
    // blocks | masked << 16 | first of its group here << 17 | last of its group here << 18
    cb_reserve_back(cParC, 1);
    volatile tt_l1_ptr uint32_t* pc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cParC));
    volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + kInfo);
    uint32_t np = 0;
    for_each_part(rw, lo, hi, [&](uint32_t b, uint32_t g, uint32_t, uint32_t blo, uint32_t bhi, bool first,
                                  bool last) {
        const bool masked = bhi == rw[b].n && static_cast<uint32_t>(rw[b].pos) % kBlock != kBlock - 1;
        pc[kParHdr + np] = (bhi - blo) | (masked ? 1u << 16 : 0u) | (first ? 1u << 17 : 0u) | (last ? 1u << 18 : 0u);
        info[np] = blo | (g << 16) | (b << 20);
        ++np;  // < MAXPART (host bound)
    });
    pc[0] = np;
    bool reducer = false;
    for (uint32_t r = 0; r < RMAX; ++r) {
        const uint32_t grp = c + r * NC;
        const uint32_t parts = grp < kKvHeads * ROWS ? rw[grp / kKvHeads].parts : 0;
        pc[1 + r] = (4 * parts + 31) / 32;
        reducer = reducer || parts > 0;
    }
    cb_push_back(cParC, 1);
    // constant tiles from the host-built consts tensor: zero, ones (partition work), A (reductions); they land with
    // the first segment's reads (one barrier)
    if (np > 0) {
        cb_reserve_back(cZero, 1);
        zero_l1(get_write_ptr(cZero), kBf16);
        cb_reserve_back(cOnes, 1);
        noc_async_read(get_noc_addr(kOnesPage * kConstReps + rep, consts), get_write_ptr(cOnes), kBf16);
    }
    if (reducer) {
        cb_reserve_back(cA, NKT);
        for (uint32_t kt = 0; kt < NKT; ++kt) {
            noc_async_read(get_noc_addr(kAPage * kConstReps + rep, consts), get_write_ptr(cA) + kt * kBf16, kBf16);
        }
    }
    if (np == 0) {
        noc_async_read_barrier();
        if (reducer) {
            cb_push_back(cA, NKT);
        }
        return;
    }
    DeviceZoneScopedN("sdpa-kv");
    uint32_t pt_row = 0;  // page-table entry e of the current segment's row at pt_row + 4 e
    uint32_t g = 0;
    for (uint32_t k = 0; k < np; ++k) {
        const uint32_t word = pc[kParHdr + k];
        const uint32_t nb = word & 0xFFFF;
        const uint32_t lo_blk = info[k] & 0xFFFF;
        const bool masked = (word >> 16) & 1, first = (word >> 17) & 1;
        uint32_t pt_first = 0;
        if (first) {  // a new segment: Q rows 4g..4g+3 of batch b, the segment's page-table entries
            const uint32_t b = info[k] >> 20;
            g = (info[k] >> 16) & 0xF;
            uint32_t end = lo_blk + nb;
            for (uint32_t j = k; !((pc[kParHdr + j] >> 18) & 1);) {
                ++j;
                end = (info[j] & 0xFFFF) + (pc[kParHdr + j] & 0xFFFF);
            }
            cb_reserve_back(cQ, 2);
            const uint32_t qdst = get_write_ptr(cQ);
            pt_first = (lo_blk * 4) & ~63u;
            const uint32_t pt_bytes = ((end * 4 + 63) & ~63u) - pt_first;
            noc_async_read(get_noc_addr(b, page_table, pt_first), scratch + kPt, pt_bytes);
            // rows 0-3 of faces 0/1 from q, every other row zero (disjoint byte ranges: no ordering needed)
            const uint32_t r0 = 4 * g;
            for (uint32_t t = 0; t < 2; ++t) {
                const uint32_t tile = qdst + t * kBf16;
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t off = (((r0 >> 4) * 2 + f) * 512) + (r0 & 15) * 32;
                    noc_async_read(get_noc_addr(b * 2 + t, q, off), tile + f * 512, 128);
                    zero_l1(tile + f * 512 + 128, 384);
                }
                zero_l1(tile + 1024, 1024);
            }
        }
        if (masked) {  // the group's last block: mask pair for its live tokens
            const uint32_t valid = static_cast<uint32_t>(rw[info[k] >> 20].pos) % kBlock + 1;
            cb_reserve_back(cMask, 2);
            for (uint32_t t = 0; t < 2; ++t) {
                const uint32_t page = (kMaskPage + 2 * (valid - 1) + t) * kConstReps + rep;
                noc_async_read(get_noc_addr(page, consts), get_write_ptr(cMask) + t * kBf16, kBf16);
            }
        }
        if (first || masked) {
            noc_async_read_barrier();
            if (k == 0) {
                cb_push_back(cZero, 1);
                cb_push_back(cOnes, 1);
                if (reducer) {
                    cb_push_back(cA, NKT);
                }
            }
            if (first) {
                cb_push_back(cQ, 2);
                pt_row = scratch + kPt - pt_first;
            }
            if (masked) {
                cb_push_back(cMask, 2);
            }
        }
        // K then V blocks (transaction-id pipelined, kOutstanding blocks in flight).  K tiles go to slot d*2 + t (the
        // [dims, tokens] order the transposed QK matmul takes), V tiles to slot t*2 + d.
        for (uint32_t which = 0; which < 2; ++which) {
            const uint32_t cb = which == 0 ? cK : cV;
            const auto& acc = which == 0 ? kc : vc;
            // the CB wraps at its own bounds [fifo_limit - fifo_size, fifo_limit) (the write pointer is anywhere in
            // the ring at a partition's start)
            const uint32_t ring_end = get_local_cb_interface(cb).fifo_limit;
            const uint32_t ring_size = get_local_cb_interface(cb).fifo_size;
            uint32_t pending = 0, issue_trid = 1, wait_trid = 1;
            reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
            for (uint32_t blk = lo_blk; blk < lo_blk + nb; ++blk) {
                const uint32_t page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt_row)[blk];
                cb_reserve_back(cb, 4 * (pending + 1));
                uint32_t dst = get_write_ptr(cb) + pending * 4 * kBfp8;
                if (dst >= ring_end) {
                    dst -= ring_size;
                }
                noc_async_read_set_trid(issue_trid);
                const uint32_t base_id = (page * kKvHeads + g) * 4;
                for (uint32_t t = 0; t < 2; ++t) {
                    for (uint32_t d = 0; d < 2; ++d) {
                        const uint32_t slot = which == 0 ? d * 2 + t : t * 2 + d;
                        noc_async_read(get_noc_addr(base_id + t * 2 + d, acc), dst + slot * kBfp8, kBfp8);
                    }
                }
                issue_trid = issue_trid == kOutstanding ? 1 : issue_trid + 1;
                if (++pending == kOutstanding) {
                    noc_async_read_barrier_with_trid(wait_trid);
                    cb_push_back(cb, 4);
                    wait_trid = wait_trid == kOutstanding ? 1 : wait_trid + 1;
                    --pending;
                }
            }
            noc_async_read_set_trid(0);
            noc_async_read_barrier();
            for (; pending > 0; --pending) {
                cb_push_back(cb, 4);
            }
        }
    }
}

// reducer slot r: zero the gather rows no partition writes (rows >= 4 parts), the m scatter tiles, the M column
FORCE_INLINE void writer_zero(uint32_t gather, uint32_t parts) {
    const uint32_t mt = get_write_ptr(cM);
    const uint32_t mraw = get_write_ptr(cMraw);
    const uint32_t nkt = (4 * parts + 31) / 32;  // live gather tiles (compute reads only these)
    cb_reserve_back(cG, 3 * NKT);
    for (uint32_t k0 = 0; k0 < 32 * nkt; k0 += 16) {  // 16-row face halves of the live gather tiles
        const uint32_t first = 4 * parts > k0 ? 4 * parts - k0 : 0;
        if (first >= 16) {
            continue;
        }
        const uint32_t kt = k0 / 32, half = (k0 % 32) / 16;
        for (uint32_t c = 0; c < 3; ++c) {
            for (uint32_t f = 0; f < 2; ++f) {
                zero_l1(gather + (kt * 3 + c) * kFp32 + (half * 2 + f) * 1024 + first * 64, (16 - first) * 64);
            }
        }
    }
    cb_reserve_back(cMraw, NKT);
    zero_l1(mraw, nkt * kFp32);
    cb_reserve_back(cM, 1);
    zero_l1(mt, kFp32);
    noc_async_read_barrier();
    // rows 0-3 (the exponentiated ones) of the m scatter tiles: -1e30 where no partition max lands -> weight 0
    for (uint32_t kt = 0; kt < nkt; ++kt) {
        for (uint32_t f = 0; f < 2; ++f) {
            volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mraw + kt * kFp32 + f * 1024);
            for (uint32_t w = 0; w < 64; ++w) {
                row[w] = 0xF149F2CAu;
            }
        }
    }
}

// reducer slot r of group (b, g): wait for its parts partitions; M[r] = max_p m_p[r] (exact) into column 0 of the M
// tile; m_p[r] into element (r, 4p + r) of the m scatter tiles (compute forms A = (k % 4 == r) exp((m - M) scale));
// then the output rows
template <typename Out>
FORCE_INLINE void writer_reduce(uint32_t r, uint32_t b, uint32_t g, uint32_t parts, uint32_t mail, uint32_t gather,
                                const Out& out) {
    const uint32_t mt = get_write_ptr(cM);
    const uint32_t mraw = get_write_ptr(cMraw);
    writer_zero(gather, parts);
    {
        DeviceZoneScopedN("sdpa-gather-wait");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(r));
        noc_semaphore_wait(s, parts);
        noc_semaphore_set(s, 0);
    }
    volatile tt_l1_ptr uint32_t* mail_v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mail);
    for (uint32_t rr = 0; rr < 4; ++rr) {
        uint32_t best = mail_v[rr], bk = fkey(best);
        for (uint32_t p = 0; p < parts; ++p) {
            const uint32_t v = mail_v[4 * p + rr], k = fkey(v);
            if (k > bk) {
                best = v;
                bk = k;
            }
            const uint32_t col = 4 * p + rr, kt = col / 32, kk = col % 32;
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mraw + kt * kFp32 + (kk >> 4) * 1024 + rr * 64)[kk & 15] = v;
        }
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mt + rr * 64)[0] = best;  // column 0 of row rr
    }
    cb_push_back(cM, 1);
    cb_push_back(cMraw, NKT);
    cb_push_back(cG, 3 * NKT);
    // out rows 0-3 (q heads 4g + rr), dims [0, 32) / [32, 64) -> row b of the flattened attention output's tile
    // columns 2 (4g + rr) + c of the output tensor
    cb_wait_front(cOut, 2);
    const uint32_t src = get_read_ptr(cOut);
    for (uint32_t rr = 0; rr < 4; ++rr) {
        for (uint32_t c = 0; c < 2; ++c) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t off = (((b >> 4) * 2 + f) * 512) + (b & 15) * 32;
                const uint64_t dst = get_noc_addr((4 * g + rr) * 2 + c, out, off);
                noc_async_write(src + c * kBf16 + f * 512 + rr * 32, dst, 32);
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cOut, 2);
}

// writer role: partition results to their reducers, this core's reductions, output
FORCE_INLINE void writer(uint32_t c) {
    // the reader's schedule is not shared: recompute it from cur_pos (cheap, same result)
    constexpr uint32_t pos_page = get_named_compile_time_arg_val("pos_page");
    const auto cur_pos = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(7), pos_page);
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(6), kBf16);
    // mail: RMAX mailbox slots, then [pos_box 64 B, stage 64 B, Row[ROWS]]
    const uint32_t mail = get_write_ptr(cMail);
    const uint32_t pos_box = mail + RMAX * kMailSlot;
    const uint32_t stage = pos_box + 64;
    const uint32_t gather = get_write_ptr(cG);  // reducer slot r's gather tiles at gather + r * kSlotG
    noc_async_read(get_noc_addr(0, cur_pos, 0), pos_box, 64);
    noc_async_read_barrier();
    Row* rw = reinterpret_cast<Row*>(stage + 64);
    uint32_t lo, hi;
    plan(pos_box, c, rw, lo, hi);
    // 1. per partition p of group G: its 4 row maxima (column 0, rows 0-3 of the FP32 m tile) -> mailbox slot p of
    //    reducer slot G / NC on core G % NC; rows 0-3 of [O_lo | O_hi | l] (FP32, relative to its max) -> rows
    //    4p..4p+3 of that slot's gather tiles; one semaphore increment per partition
    for_each_part(rw, lo, hi, [&](uint32_t b, uint32_t g, uint32_t p, uint32_t, uint32_t, bool, bool) {
        const uint32_t grp = b * kKvHeads + g;
        const uint32_t red = grp % NC, r = grp / NC;
        const uint32_t red_x = get_common_arg_val<uint32_t>(kTableArg + 2 * red);
        const uint32_t red_y = get_common_arg_val<uint32_t>(kTableArg + 2 * red + 1);
        cb_wait_front(cMsend, 1);
        {
            const uint32_t m = get_read_ptr(cMsend);
            volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
            for (uint32_t rr = 0; rr < 4; ++rr) {
                st[rr] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m + rr * 64)[0];
            }
            cb_pop_front(cMsend, 1);
        }
        cb_wait_front(cO, 3);
        DeviceZoneScopedN("sdpa-send");
        noc_async_write(stage, get_noc_addr(red_x, red_y, mail + r * kMailSlot + 16 * p), 16);
        const uint32_t o = get_read_ptr(cO);
        const uint32_t k = 4 * p, kt = k / 32, kk = k % 32;
        const uint32_t slot_g = gather + r * kSlotG;
        for (uint32_t cc = 0; cc < 3; ++cc) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t dst = slot_g + (kt * 3 + cc) * kFp32 + ((kk >> 4) * 2 + f) * 1024 + (kk & 15) * 64;
                noc_async_write(o + cc * kFp32 + f * 1024, get_noc_addr(red_x, red_y, dst), 256);
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(red_x, red_y, get_semaphore(r)), 1);
        noc_async_atomic_barrier();
        cb_pop_front(cO, 3);
    });
    // 2. this core's reducer slots: groups c + r NC.  The partitions land at fixed slot addresses (gather + r kSlotG)
    //    and compute reads the gather tiles from the CB front: an idle slot passes its region through (pushed
    //    unfilled, popped unread) to keep the two in step.
    for (uint32_t r = 0; r < RMAX; ++r) {
        const uint32_t grp = c + r * NC;
        const uint32_t b = grp / kKvHeads;
        if (grp < kKvHeads * ROWS && rw[b].parts > 0) {
            writer_reduce(r, b, grp % kKvHeads, rw[b].parts, mail + r * kMailSlot, gather + r * kSlotG, out);
        } else {
            cb_reserve_back(cG, 3 * NKT);
            cb_push_back(cG, 3 * NKT);
        }
    }
}
}  // namespace sdpa
#endif

#if defined(COMPILE_FOR_TRISC)
namespace sdpa {
FORCE_INLINE void compute_sdpa() {
    cb_wait_front(cParC, 1);
    const uint32_t np = read_tile_value(cParC, 0, 0);
    uint32_t nkts[RMAX];
    for (uint32_t r = 0; r < RMAX; ++r) {
        nkts[r] = read_tile_value(cParC, 0, 1 + r);  // reducer slot r: live gather tiles ceil(4 parts / 32), 0: none
    }
    // one exact two-pass softmax per partition (the params page stays at the front until the end)
    for (uint32_t k = 0; k < np; ++k) {
    const uint32_t word = read_tile_value(cParC, 0, kParHdr + k);
    const uint32_t nb = word & 0xFFFF;
    const bool mask_last = (word >> 16) & 1;
    if ((word >> 17) & 1) {
        cb_wait_front(cQ, 2);  // a new group (segment): its Q
    }
    const uint32_t nt = 2 * nb;  // token tiles
    // pass 1: S = Q K^T
    {
        DeviceZoneScopedN("sdpa-qk");
        reconfig_data_format(cK, cQ);
        pack_reconfig_data_format(cS);
        matmul_block_init(cQ, cK, 1, 2, 1, 2);
        for (uint32_t blk = 0; blk < nb; ++blk) {
            cb_wait_front(cK, 4);
            tile_regs_acquire();
            for (uint32_t d = 0; d < 2; ++d) {
                matmul_block(cQ, cK, d, d * 2, 0, 1, 2, 1, 2);
            }
            if (mask_last && blk == nb - 1) {
                cb_wait_front(cMask, 2);
                cb_wait_front(cZero, 1);
                reconfig_data_format(cZero, cMask);
                add_tiles_init(cZero, cMask, true);
                add_tiles(cZero, cMask, 0, 0, 0);
                add_tiles(cZero, cMask, 0, 1, 1);
                reconfig_data_format(cK, cQ);
                matmul_block_init(cQ, cK, 1, 2, 1, 2);
            }
            tile_regs_commit();
            cb_reserve_back(cS, 2);
            tile_regs_wait();
            pack_tile(0, cS, 0);
            pack_tile(1, cS, 1);
            tile_regs_release();
            cb_push_back(cS, 2);
            cb_pop_front(cK, 4);
        }
        if (mask_last) {
            cb_pop_front(cMask, 2);
        }
    }
    // local row max
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
        tile_regs_commit();
        cb_reserve_back(cMloc, 1);
        cb_reserve_back(cMsend, 1);
        tile_regs_wait();
        pack_tile(0, cMloc, 0);
        pack_tile(0, cMsend, 0);
        tile_regs_release();
        cb_push_back(cMloc, 1);
        cb_push_back(cMsend, 1);
    }
    // pass 2: P = exp((S - M) * scale)
    {
        cb_wait_front(cMloc, 1);
        DeviceZoneScopedN("sdpa-exp");
        reconfig_data_format(cS, cMloc);
        pack_reconfig_data_format(cP);
        sub_bcast_cols_init_short(cS, cMloc);
#if EXP_APPROX
        exp_tile_init<true, SCALE_FP32, InputClamping::None>();
#else
        exp_tile_init<false>();
#endif
        PACK((llk_pack_relu_config(ReluConfig::zero())));
        for (uint32_t j = 0; j < nt; j += 4) {
            const uint32_t cnt = nt - j < 4 ? nt - j : 4;
            tile_regs_acquire();
            for (uint32_t jj = 0; jj < cnt; ++jj) {
                sub_tiles_bcast_cols(cS, cMloc, j + jj, 0, jj);
#if EXP_APPROX
                exp_tile<true, false, InputClamping::None, 8>(jj, VectorMode::R);  // approx: 8 or 32 only
#else
                exp_tile<false, true, InputClamping::ClampToNegative, EXP_ITERS>(jj, VectorMode::R, SCALE_BF16);
#endif
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
    // O = P V (dst 0, 1), l = P 1 (dst 2), FP32 in DEST over all blocks
    {
        DeviceZoneScopedN("sdpa-pv");
        cb_wait_front(cP, nt);
        reconfig_data_format(cV, cP);
        matmul_block_init(cP, cV, 0, 2, 1, nt);
        tile_regs_acquire();
        for (uint32_t blk = 0; blk < nb; ++blk) {
            cb_wait_front(cV, 4);
            for (uint32_t t = 0; t < 2; ++t) {
                matmul_block(cP, cV, 2 * blk + t, t * 2, 0, 0, 2, 1, nt);
            }
            cb_pop_front(cV, 4);
        }
        cb_wait_front(cOnes, 1);
        reconfig_data_format(cOnes, cP);
        matmul_block_init(cP, cOnes, 0, 1, 1, nt);
        for (uint32_t j = 0; j < nt; ++j) {
            matmul_block(cP, cOnes, j, 0, 2, 0, 1, 1, nt);
        }
        tile_regs_commit();
        cb_reserve_back(cO, 3);
        pack_reconfig_data_format(cO);
        tile_regs_wait();
        for (uint32_t cc = 0; cc < 3; ++cc) {
            pack_tile(cc, cO, cc);
        }
        tile_regs_release();
        cb_push_back(cO, 3);
        cb_pop_front(cP, nt);
    }
// S and P are indexed from their CB fronts without wrap: rewind both to the CB base (they hold 2 C tiles) before
    // the next partition by passing the unused rest through
    if (k + 1 < np && nt < 2 * C) {
        for (const uint32_t cb : {cS, cP}) {
            cb_reserve_back(cb, 2 * C - nt);
            cb_push_back(cb, 2 * C - nt);
            cb_wait_front(cb, 2 * C - nt);
            cb_pop_front(cb, 2 * C - nt);
        }
    }
    if ((word >> 18) & 1) {
        cb_pop_front(cQ, 2);  // the group's last partition on this core
    }
    }  // partitions
    cb_pop_front(cParC, 1);
    // reducer slots: A[r, 4p + r] = exp((m_p[r] - M[r]) * scale) (the partition weights), [O | l] = A G (fixed-order
    // sum over the partitions by the matmul unit), out = O / l
    bool have_a = false;
    for (uint32_t r = 0; r < RMAX; ++r) {
        const uint32_t nkt = nkts[r];
        if (nkt == 0) {  // idle slot: its gather region passes through (the writer pushed it unfilled)
            cb_wait_front(cG, 3 * NKT);
            cb_pop_front(cG, 3 * NKT);
            continue;
        }
        DeviceZoneScopedN("sdpa-reduce");
        if (!have_a) {
            cb_wait_front(cA, NKT);
            have_a = true;
        }
        cb_wait_front(cMraw, NKT);
        cb_wait_front(cM, 1);
        for (uint32_t kt = 0; kt < nkt; ++kt) {  // live gather tiles only (the rest are all-zero rows)
            reconfig_data_format(cMraw, cM);
            pack_reconfig_data_format(cAf);
            sub_bcast_cols_init_short(cMraw, cM);
            tile_regs_acquire();
            sub_tiles_bcast_cols(cMraw, cM, kt, 0, 0);
            exp_tile_init<false>();  // accurate exp for the worker weights (also with EXP_APPROX)
            exp_tile<false, true, InputClamping::ClampToNegative, 2>(0, VectorMode::R, SCALE_BF16);
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
        cb_wait_front(cG, 3 * NKT);
        reconfig_data_format(cG, cAf);
        pack_reconfig_data_format(cOut);
        matmul_block_init(cAf, cG, 0, 3, 1, NKT);
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < nkt; ++kt) {
            matmul_block(cAf, cG, kt, kt * 3, 0, 0, 3, 1, NKT);
        }
        recip_tile_init();
        recip_tile(2);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);
        mul_binary_tile(1, 2, 1);
        tile_regs_commit();
        cb_reserve_back(cOut, 2);
        pack_reconfig_data_format(cOut);
        tile_regs_wait();
        pack_tile(0, cOut, 0);
        pack_tile(1, cOut, 1);
        tile_regs_release();
        cb_push_back(cOut, 2);
        cb_pop_front(cAf, nkt);
        if (nkt < NKT) {  // the next slot indexes cAf from its front without wrap: pass the rest through
            cb_reserve_back(cAf, NKT - nkt);
            cb_push_back(cAf, NKT - nkt);
            cb_wait_front(cAf, NKT - nkt);
            cb_pop_front(cAf, NKT - nkt);
        }
        cb_pop_front(cMraw, NKT);
        cb_pop_front(cM, 1);
        cb_pop_front(cG, 3 * NKT);
    }
    if (have_a) {
        cb_pop_front(cA, NKT);
    }
}

}  // namespace sdpa
#endif

void kernel_main() {
    using namespace sdpa;
    const uint32_t c = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    // per-core role swap: the reader runs on BRISC (NOC1) where NOC0 serves this core poorly
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
    // ------------------------------------------------------------------------------------------------ compute
    compute_kernel_hw_startup<SrcOrder::Reverse>(cQ, cK, cS);
    compute_sdpa();
#endif
}
