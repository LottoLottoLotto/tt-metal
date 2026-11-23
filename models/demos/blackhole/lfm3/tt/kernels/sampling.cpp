// SPDX-License-Identifier: Apache-2.0
// Exact on-device vLLM sampling for LFM3 (copied from the LFM2.5 runtime; tt/sampling.py; forked from Qwen36's device_sampling_kernel.cpp without
// the speculative modes, with a cheaper per-worker selection and merge, plus the full-vocabulary mass of
// top_k-disabled rows).
//
// SAMPLE_MODE 0 (scan, BRISC + NCRISC workers): each worker visits a disjoint stripe of vocabulary tiles, applies
//   vLLM's penalties to every logit of every active row (from the per-slot output-count/prompt-flag history), and
//   keeps its exact top-keff by (penalized float32 value desc, token id asc): an append buffer of 2 keff entries,
//   compacted to the best keff by partition selection whenever it fills (the keff-th key is the new threshold).
//   Its list goes out as 8 sorted heads followed by the rest unsorted. F_FULL rows also accumulate
//   sum_i 2^(-(m - l_i) log2(e) / T) over the worker's tokens relative to a reference m, per integer exponent bucket
//   (Q1.31 weights from a 512-entry table), tokens below 2^-46 of m skipped. m: the row's global maximum for
//   unpenalized rows with SAMPLE_GLOBAL_MAX (from mode 2), else the worker's running maximum (exact either way; the
//   global maximum skips far more tokens). Only the needed face rows are read.
// SAMPLE_MODE 2 (row maxima, BRISC + NCRISC workers, the scan's stripes): each worker's stripe maximum of every
//   unpenalized F_FULL row -> rmax, read by the SAMPLE_GLOBAL_MAX scan that follows.
// SAMPLE_MODE 1 (row, one BRISC per row): exact merge into the global top-keff (the heads of every worker give a
//   lower bound of the keff-th entry; only workers whose 8th head beats it are read in full), the row's
//   full-vocabulary mass from the worker sums (F_FULL), then the vLLM sampler (temperature, min_p, top-k, top-p,
//   softmax) and, per SAMPLE_ROW_OP:
//     0 SAMPLE: counter-based draw, token + flags to out[row], history += token (unless the host redraws);
//     2 PREP:   store the processed distribution (ids, probabilities, kept tail) in rowbuf[row] (tests).
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "sampling_math.hpp"

namespace {
using namespace dsm;
constexpr uint32_t R = SAMPLE_ROWS;
constexpr uint32_t K = SAMPLE_K;
constexpr uint32_t NW = SAMPLE_WORKERS;
constexpr uint32_t TILES = SAMPLE_TILES;
constexpr uint32_t RB = ((R * 32 + 63) / 64) * 64;  // bytes of R face rows, DRAM-read aligned
// A staged vocabulary tile: its first 512 + RB bytes (face 0, then face 1's first R rows) in one read: rows [0, R) of
// both faces (columns 0-15 and 16-31). One request per tile (small DRAM reads are request-bound).
constexpr uint32_t TILE_BYTES = 512 + RB;
// L1 offset of row r's 16 logits of half h (columns 16h..16h+15) inside one staged tile.
constexpr uint32_t half_offset(uint32_t r, uint32_t h) { return h * 512 + r * 32; }
constexpr uint32_t STRIPE = (TILES + NW - 1) / NW;   // vocabulary tiles per worker (at most)
constexpr uint32_t HEADS = 8;                         // sorted leading entries of a worker list (the heads page)
constexpr uint32_t RMAX_COPIES = 8;                   // copies of the row-maximum page (tt/sampling.py _RMAX_COPIES)
// Per-worker list: a worker keeps its best min(keff, KW). With keff <= KW the merge is the exact global top-keff;
// beyond, a worker that may have dropped entries bounds the exact prefix of the merged list (mode 1).
constexpr uint32_t KW = 24;
constexpr uint32_t ENTRY_PAGE = KW * 8;               // (key, id) pairs per worker list
constexpr uint32_t CAP = 2 * KW;                      // selection buffer per row
inline uint32_t worker_list(uint32_t keff) { return keff < KW ? keff : KW; }
constexpr uint32_t MASS_SLOT = 64;                    // per (row, worker) in the mass tensor: mq (int64 Q.24), s (Q.46)
constexpr uint32_t ROW_PAGE = 1024;
constexpr uint32_t FULL_BATCH = 32;                   // worker lists read per round of the merge
// rowbuf words (PREP).
constexpr uint32_t W_COUNT = 0, W_FLAGS = 1, W_TOKEN = 2, W_IDS = 4, W_PROBS = 4 + 96, W_TAIL = 248, W_ZFULL = 250;
constexpr int64_t NO_MAX = (int64_t)0x8000000000000000ull;

struct Params {
    uint32_t w[32];
};

inline RowParams row_params(const Params& p) {
    return RowParams{p.w[0], bitsf(p.w[1]), p.w[2], bitsf(p.w[4]), bitsf(p.w[13]), p.w[18], p.w[19]};
}

inline void barrier_compiler() { asm volatile("" ::: "memory"); }

template <typename A>
inline void read_page(const A& acc, uint32_t page, uint32_t l1, uint32_t bytes, uint32_t offset = 0) {
    noc_async_read(acc.get_noc_addr(page, offset), l1, bytes);
}

// Common runtime args: 0 logits, 1 heads, 2 params, 3 partial, 4 rowbuf, 5 out, 6 mass, 7 pos, 8 rmax (the workers'
// stripe maxima, mode 2), 9 aux (u32 tokens: a row with params word W_EXTRA = n also counts aux[b .. b + n) in its
// output history, b = params word W_EXTRA_BASE), then the per-slot history tensors' base addresses.
inline auto hist_acc(uint32_t slot) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(10 + slot), 128);
}

// history[slot][token] += 1 (read-modify-write of the 128-byte page; DRAM writes are 16-byte aligned).
inline void commit(uint32_t slot, uint32_t token, uint32_t l1) {
    const auto hist = hist_acc(slot);
    const uint32_t page = token / 32;
    read_page(hist, page, l1, 128);
    noc_async_read_barrier();
    auto h = reinterpret_cast<volatile uint32_t*>(l1);
    h[token % 32] = h[token % 32] + 1;
    noc_async_write(l1, hist.get_noc_addr(page), 128);
    noc_async_write_barrier();
}

// (key desc, id asc): the device's total order of candidates.
inline bool better(uint32_t ka, uint32_t ia, uint32_t kb, uint32_t ib) { return ka > kb || (ka == kb && ia < ib); }

inline void swap_entry(uint32_t* k, uint32_t* id, uint32_t a, uint32_t b) {
    const uint32_t tk = k[a], ti = id[a];
    k[a] = k[b];
    id[a] = id[b];
    k[b] = tk;
    id[b] = ti;
}

// Partition selection: afterwards [0, nth] hold the best nth + 1 entries of [0, n), entry nth the (nth + 1)-th best.
inline void select_best(uint32_t* k, uint32_t* id, uint32_t n, uint32_t nth) {
    uint32_t lo = 0, hi = n - 1;
    while (lo < hi) {
        swap_entry(k, id, lo + (hi - lo) / 2, hi);
        const uint32_t pk = k[hi], pi = id[hi];
        uint32_t store = lo;
        for (uint32_t i = lo; i < hi; ++i) {
            if (better(k[i], id[i], pk, pi)) swap_entry(k, id, i, store++);
        }
        swap_entry(k, id, store, hi);
        if (store == nth) return;
        if (store < nth) {
            lo = store + 1;
        } else {
            hi = store - 1;
        }
    }
}

// Insertion sort of [0, n) best-first (n <= HEADS).
inline void sort_best(uint32_t* k, uint32_t* id, uint32_t n) {
    for (uint32_t i = 1; i < n; ++i) {
        const uint32_t kk = k[i], ii = id[i];
        uint32_t j = i;
        while (j && better(kk, ii, k[j - 1], id[j - 1])) {
            k[j] = k[j - 1];
            id[j] = id[j - 1];
            --j;
        }
        k[j] = kk;
        id[j] = ii;
    }
}

// Running full-vocabulary mass of one row on one worker: s (Q.46) + bucket[n] (sums of Q1.31 weights 2^-frac whose
// integer exponent is n), all relative to the reference mq.
struct Mass {
    int64_t mq;    // reference (running maximum) in Q.24; NO_MAX before the first token
    uint64_t s;    // collapsed part, Q.46
    uint32_t th;   // key at or below mq - window: such tokens add nothing
    uint32_t pad;
    uint64_t bucket[MASS_BITS];
};

inline uint64_t mass_collapse(Mass& ms) {
    uint64_t s = ms.s;
    for (uint32_t n = 0; n < MASS_BITS; ++n) {
        s += (ms.bucket[n] << 15) >> n;  // bucket < 2^41 (<= 2^10 tokens of weight <= 2^31 per worker and row)
        ms.bucket[n] = 0;
    }
    return s;
}

inline void mass_add(Mass& ms, uint32_t fb, const uint32_t* p) {
    const int64_t lq = f2q24(fb);
    if (lq > ms.mq) {
        // New running maximum: rescale what was summed so far to it.
        if (ms.mq != NO_MAX) ms.s = rescale_mass(mass_collapse(ms), scale_q24((uint64_t)(lq - ms.mq), p[18], p[19]));
        ms.mq = lq;
        const int64_t win = (int64_t)(((uint64_t)p[21] << 32) | p[20]);
        ms.th = q24_floor_key(lq - win);
    }
    const uint64_t y = scale_q24((uint64_t)(ms.mq - lq), p[18], p[19]);
    if (y >= ((uint64_t)MASS_BITS << 24)) return;
    ms.bucket[(uint32_t)y >> 24] += mass_w31((uint32_t)y & 0xffffffu);
}

// Largest raw key of row r in this worker's staged stripe.
inline uint32_t stripe_max(uint32_t tile_l1, uint32_t ntiles, uint32_t r) {
    uint32_t mk = 0;
    for (uint32_t i = 0; i < ntiles; ++i) {
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t* f = reinterpret_cast<const uint32_t*>(tile_l1 + i * TILE_BYTES + half_offset(r, half));
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t k0 = fkey(f[j] << 16), k1 = fkey(f[j] & 0xffff0000u);
                mk = k0 > mk ? k0 : mk;
                mk = k1 > mk ? k1 : mk;
            }
        }
    }
    return mk;
}
}  // namespace
#endif

void kernel_main() {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#if SAMPLE_MODE == 0
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t base = get_write_ptr(0) + (is_ncrisc ? SAMPLE_SCRATCH_HALF : 0);
    const auto logits = TensorAccessor(tensor_accessor::make_interleaved_dspec<SAMPLE_LOGITS_DRAM>(), get_common_arg_val<uint32_t>(0), 2048);
    const auto params = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 128);
    const auto partial = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3), ENTRY_PAGE);
    const auto heads = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), NW * 64);
    const auto mass = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6), NW * MASS_SLOT);
    // Scratch: params R*128 | logits STRIPE*TILE_BYTES | hist R*STRIPE*128 | stage K*8+64 (its last 64 bytes: aux) |
    // (SAMPLE_GLOBAL_MAX) every worker's stripe maxima NW*64, per RISC half.
    // Every region is a multiple of 64 bytes (DRAM reads need 64-byte aligned L1 destinations). The per-row selection
    // buffer and mass state live in the RISC's local memory (single-cycle), one row at a time.
    const uint32_t p_l1 = base;
    const uint32_t tile_l1 = p_l1 + R * 128;
    const uint32_t hist_l1 = tile_l1 + STRIPE * TILE_BYTES;
    const uint32_t stage_l1 = hist_l1 + R * STRIPE * 128;
#if SAMPLE_PROFILE
    const uint32_t t_begin = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
#endif
    for (uint32_t r = 0; r < R; ++r) read_page(params, r, p_l1 + r * 128, 128);
    noc_async_read_barrier();
    barrier_compiler();
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(p_l1);
    // This worker's stripe: tiles worker, worker + NW, ... (and the penalized rows' history pages), read once.
    const uint32_t ntiles = worker < TILES ? (TILES - worker + NW - 1) / NW : 0;
    for (uint32_t i = 0; i < ntiles; ++i) {
        const uint32_t t = worker + i * NW;
        const uint32_t dst = tile_l1 + i * TILE_BYTES;
        read_page(logits, t, dst, TILE_BYTES, 0);
        for (uint32_t r = 0; r < R; ++r) {
            if ((pw[r * 32] & (F_ACTIVE | F_PENALTY)) == (F_ACTIVE | F_PENALTY)) {
                read_page(hist_acc(pw[r * 32 + 11]), t, hist_l1 + (r * STRIPE + i) * 128, 128);
            }
        }
    }
    // Verify rows: the group's drafts before the row count as outputs (aux tokens [W_EXTRA_BASE, + W_EXTRA)).
    const uint32_t aux_l1 = stage_l1 + K * 8;
    bool any_extra = false;
    for (uint32_t r = 0; r < R; ++r) {
        any_extra |= (pw[r * 32] & (F_ACTIVE | F_PENALTY)) == (F_ACTIVE | F_PENALTY) && pw[r * 32 + W_EXTRA] != 0;
    }
    if (any_extra) {
        read_page(TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(9), 64),
                  0, aux_l1, 64);
    }
#if SAMPLE_GLOBAL_MAX
    // The global maximum of every unpenalized F_FULL row from all workers' stripe maxima (mode 2, launched just before;
    // one of its RMAX_COPIES copies, spread over the DRAM banks: every worker reads a whole copy).
    const uint32_t rmax_l1 = stage_l1 + K * 8 + 64;
    read_page(TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(8), NW * 64),
              worker % RMAX_COPIES, rmax_l1, NW * 64);
#endif
    noc_async_read_barrier();
    barrier_compiler();
    uint32_t* stage = reinterpret_cast<uint32_t*>(stage_l1);
#if SAMPLE_GLOBAL_MAX
    uint32_t gmax[R];
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t* slots = reinterpret_cast<const uint32_t*>(rmax_l1);
        uint32_t mk = 0;
        if ((pw[r * 32] & (F_ACTIVE | F_FULL | F_PENALTY)) == (F_ACTIVE | F_FULL)) {
            for (uint32_t w = 0; w < NW; ++w) mk = slots[w * 16 + r] > mk ? slots[w * 16 + r] : mk;
        }
        gmax[r] = mk;
    }
#endif
#if SAMPLE_PROFILE
    const uint32_t t_dma = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - t_begin;
    uint32_t t_cmp = 0, t_out = 0, n_mass = 0, n_app = 0;
#endif
    uint32_t* mstage = stage + ENTRY_PAGE / 4;
    uint32_t rk[CAP], ri[CAP];
    Mass ms;
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t* p = pw + r * 32;
        const uint32_t fl = p[0];
        if (!(fl & F_ACTIVE)) continue;
#if SAMPLE_PROFILE
        const uint32_t ta = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
#endif
        const uint32_t kw = worker_list(p[14]);
        const uint32_t cap = 2 * kw;
        const bool pen = (fl & F_PENALTY) != 0;
        const bool full = (fl & F_FULL) != 0;
        // Monotone penalties (presence, frequency >= 0, repetition >= 1) never raise a logit: a raw value at or below
        // the threshold needs no history lookup.
        const bool mono = (fl & F_MONO) != 0;
        const bool rep_on = (fl & F_REP) != 0;
        const uint32_t n_extra = pen ? p[W_EXTRA] : 0;
        const uint32_t* extra = reinterpret_cast<const uint32_t*>(aux_l1) + p[W_EXTRA_BASE];
        const float rep = bitsf(p[7]), rrep = bitsf(p[8]), freq = bitsf(p[6]), pres = bitsf(p[5]);
        // The mass reference: an unpenalized row's global maximum (exact: no token exceeds it, and the window cuts
        // relative to it); a penalized row's running maximum (penalties may lower the top token).
        ms.mq = NO_MAX;
        ms.s = 0;
        ms.th = 0;
        for (uint32_t n = 0; n < MASS_BITS; ++n) ms.bucket[n] = 0;
#if SAMPLE_GLOBAL_MAX
        if (full && !pen) {
            ms.mq = f2q24(keyf(gmax[r]));
            ms.th = q24_floor_key(ms.mq - (int64_t)(((uint64_t)p[21] << 32) | p[20]));
        }
#endif
        uint32_t cnt = 0;
        uint32_t th = 0;   // selection threshold key (0: accept everything)
        uint32_t lim = 0;  // skip limit: th, lowered to the mass window for F_FULL rows
        // With a non-negative limit and penalties that can only lower logits, a bf16 pair is rejected by two signed
        // compares of its raw bits: negative values are negative integers, positive ones order as integers.
        const bool fast_ok = !pen || mono;
        for (uint32_t i = 0; i < ntiles; ++i) {
            const uint32_t id0 = (worker + i * NW) * 32;
            const uint32_t* h = reinterpret_cast<const uint32_t*>(hist_l1 + (r * STRIPE + i) * 128);
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t* f = reinterpret_cast<const uint32_t*>(tile_l1 + i * TILE_BYTES + half_offset(r, half));
                for (uint32_t j = 0; j < 8; ++j) {
                    const uint32_t word = f[j];
                    if (fast_ok && lim >= 0x80000000u) {
                        const int32_t tb = (int32_t)(lim & 0x7fffffffu);
                        if ((int32_t)(word << 16) <= tb && (int32_t)(word & 0xffff0000u) <= tb) continue;
                    }
                    for (uint32_t b = 0; b < 2; ++b) {
                        const uint32_t c = 16 * half + 2 * j + b;
                        uint32_t fb = b ? (word & 0xffff0000u) : (word << 16);
                        uint32_t key = fkey(fb);
                        if (fast_ok && key <= lim) continue;
                        if (pen) {
                            const uint32_t e = h[c];
                            uint32_t n_out = e & 0x7fffffffu;
                            for (uint32_t x = 0; x < n_extra; ++x) n_out += extra[x] == id0 + c ? 1u : 0u;
                            const bool in_prompt = (e >> 31) != 0;
                            if (n_out || (rep_on && in_prompt)) {
                                fb = penalize(fb, n_out, in_prompt, rep_on, rep, rrep, freq, pres);
                                key = fkey(fb);
                            }
                        }
                        if (full && key > ms.th) {
                            mass_add(ms, fb, p);
#if SAMPLE_PROFILE
                            ++n_mass;
#endif
                            lim = ms.th < th ? ms.th : th;
                        }
                        // Scan order is ascending id: an entry equal to the threshold ranks after it.
                        if (key <= th) continue;
#if SAMPLE_PROFILE
                        ++n_app;
#endif
                        rk[cnt] = key;
                        ri[cnt] = id0 + c;
                        if (++cnt == cap) {
                            select_best(rk, ri, cnt, kw - 1);
                            cnt = kw;
                            th = rk[kw - 1];
                            lim = full && ms.th < th ? ms.th : th;
                        }
                    }
                }
            }
        }
#if SAMPLE_PROFILE
        const uint32_t tb = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        t_cmp += tb - ta;
#endif
        // The worker's best kw, 8 sorted heads first, as interleaved (key, id) pages, one per (row, worker); the heads
        // also go to the row's contiguous heads page; F_FULL rows write the worker's mass slot.
        if (cnt > kw) {
            select_best(rk, ri, cnt, kw - 1);
            cnt = kw;
        }
        if (cnt > HEADS) select_best(rk, ri, cnt, HEADS - 1);
        sort_best(rk, ri, cnt < HEADS ? cnt : HEADS);
        noc_async_write_barrier();  // the previous row's stage
        for (uint32_t i = 0; i < KW; ++i) {
            stage[2 * i] = i < cnt ? rk[i] : 0;
            stage[2 * i + 1] = i < cnt ? ri[i] : NO_ID;
        }
        noc_async_write(stage_l1, partial.get_noc_addr(r * NW + worker), ENTRY_PAGE);
        noc_async_write(stage_l1, heads.get_noc_addr(r, worker * 64), 64);
        if (full) {
            const uint64_t s = mass_collapse(ms);
            mstage[0] = (uint32_t)(uint64_t)ms.mq;
            mstage[1] = (uint32_t)((uint64_t)ms.mq >> 32);
            mstage[2] = (uint32_t)s;
            mstage[3] = (uint32_t)(s >> 32);
            noc_async_write((uint32_t)mstage, mass.get_noc_addr(r, worker * MASS_SLOT), 16);
        }
#if SAMPLE_PROFILE
        t_out += reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - tb;
#endif
    }
    noc_async_write_barrier();
#if SAMPLE_PROFILE
    // Profile (wall-clock cycles): row 0's mass slot words 4..7 = dma wait, scan compute, output, total.
    mstage[4] = t_dma;
    mstage[5] = t_cmp;
    mstage[6] = t_out;
    mstage[7] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - t_begin;
    mstage[8] = n_mass;
    mstage[9] = n_app;
    mstage[10] = ms.th;
    mstage[11] = (uint32_t)(ms.mq >> 24);
    noc_async_write((uint32_t)(mstage + 4), mass.get_noc_addr(0, worker * MASS_SLOT + 16), 32);
    noc_async_write_barrier();
#endif
#elif SAMPLE_MODE == 2
    // Row maxima (before a SAMPLE_GLOBAL_MAX scan; same workers and stripes): the worker's stripe maximum key of every
    // unpenalized F_FULL row -> word r of its 64-byte slot in every copy of rmax (other rows 0).
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t base = get_write_ptr(0) + (is_ncrisc ? SAMPLE_SCRATCH_HALF : 0);
    const auto logits = TensorAccessor(tensor_accessor::make_interleaved_dspec<SAMPLE_LOGITS_DRAM>(), get_common_arg_val<uint32_t>(0), 2048);
    const auto params = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 128);
    const auto rmax = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(8), NW * 64);
    // Scratch per RISC half: params R*128 | logits STRIPE*TILE_BYTES | stage 64.
    const uint32_t p_l1 = base;
    const uint32_t tile_l1 = p_l1 + R * 128;
    const uint32_t stage_l1 = tile_l1 + STRIPE * TILE_BYTES;
    for (uint32_t r = 0; r < R; ++r) read_page(params, r, p_l1 + r * 128, 128);
    const uint32_t ntiles = worker < TILES ? (TILES - worker + NW - 1) / NW : 0;
    for (uint32_t i = 0; i < ntiles; ++i) {
        const uint32_t t = worker + i * NW;
        read_page(logits, t, tile_l1 + i * TILE_BYTES, TILE_BYTES, 0);
    }
    noc_async_read_barrier();
    barrier_compiler();
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(p_l1);
    uint32_t* stage = reinterpret_cast<uint32_t*>(stage_l1);
    for (uint32_t r = 0; r < 16; ++r) {
        const bool need = r < R && (pw[r * 32] & (F_ACTIVE | F_FULL | F_PENALTY)) == (F_ACTIVE | F_FULL);
        stage[r] = need ? stripe_max(tile_l1, ntiles, r) : 0;
    }
    for (uint32_t c = 0; c < RMAX_COPIES; ++c) noc_async_write(stage_l1, rmax.get_noc_addr(c, worker * 64), 64);
    noc_async_write_barrier();
#elif SAMPLE_MODE == 1
#if defined(COMPILE_FOR_BRISC)
    const uint32_t row = get_arg_val<uint32_t>(0);
    const uint32_t base = get_write_ptr(0);
    const auto params = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 128);
    const auto partial = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3), ENTRY_PAGE);
    [[maybe_unused]] const auto rowbuf = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4), ROW_PAGE);
    [[maybe_unused]] const auto out = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(5), 16);
    const auto mass = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6), NW * MASS_SLOT);
    const auto pos = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(7), 64);
    const auto heads = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), NW * 64);
    // Scratch: params 128 | pos 64 | out 64 | hist 128 | heads NW*64 | mass NW*64 | lists FULL_BATCH*ENTRY_PAGE |
    // result ROW_PAGE | sampler
    const uint32_t p_l1 = base;
    const uint32_t pos_l1 = p_l1 + 128;
    const uint32_t out_l1 = pos_l1 + 64;
    const uint32_t h_l1 = out_l1 + 64;
    const uint32_t heads_l1 = h_l1 + 128;
    const uint32_t mass_l1 = heads_l1 + NW * 64;
    const uint32_t lists_l1 = mass_l1 + NW * MASS_SLOT;
    const uint32_t res_l1 = lists_l1 + FULL_BATCH * ENTRY_PAGE;
    auto mkeys = reinterpret_cast<uint32_t*>(res_l1 + ROW_PAGE);
    auto mids = mkeys + K;
    auto weights = reinterpret_cast<float*>(mids + K);
    read_page(params, row, p_l1, 128);
    read_page(pos, 0, pos_l1, 64);
    read_page(heads, row, heads_l1, NW * 64);
    noc_async_read_barrier();
    barrier_compiler();
    Params prm;
    for (uint32_t i = 0; i < 32; ++i) prm.w[i] = reinterpret_cast<const uint32_t*>(p_l1)[i];
    auto o = reinterpret_cast<volatile uint32_t*>(out_l1);
    auto res = reinterpret_cast<volatile uint32_t*>(res_l1);
    if (!(prm.w[0] & F_ACTIVE)) {
#if SAMPLE_ROW_OP == 0
        o[0] = 0;
        o[1] = 0;
        o[2] = 0;
        o[3] = 0;
        noc_async_write(out_l1, out.get_noc_addr(row), 16);
        noc_async_write_barrier();
#endif
        return;
    }
    const bool full = (prm.w[0] & F_FULL) != 0;
    if (full) {
        read_page(mass, row, mass_l1, NW * MASS_SLOT);
    }
    const uint32_t keff = prm.w[14];
    // Exact merge into the global best keff, kept in a local-memory min-heap (worst at the root).
    uint32_t hk[K], hi[K];
    uint32_t size = 0;
    auto offer = [&](uint32_t key, uint32_t id) -> bool {
        if (size < keff) {
            uint32_t i = size++;
            while (i) {
                const uint32_t parent = (i - 1) >> 1;
                if (!better(hk[parent], hi[parent], key, id)) break;
                hk[i] = hk[parent];
                hi[i] = hi[parent];
                i = parent;
            }
            hk[i] = key;
            hi[i] = id;
            return true;
        }
        if (!better(key, id, hk[0], hi[0])) return false;
        uint32_t i = 0;
        for (;;) {
            uint32_t l = 2 * i + 1;
            if (l >= keff) break;
            uint32_t c = l;
            if (l + 1 < keff && better(hk[l], hi[l], hk[l + 1], hi[l + 1])) c = l + 1;
            if (!better(key, id, hk[c], hi[c])) break;
            hk[i] = hk[c];
            hi[i] = hi[c];
            i = c;
        }
        hk[i] = key;
        hi[i] = id;
        return true;
    };
    // 1) Every worker's sorted heads (a list is abandoned at its first entry that cannot enter).
    const uint32_t* hd = reinterpret_cast<const uint32_t*>(heads_l1);
    for (uint32_t w = 0; w < NW; ++w) {
        for (uint32_t j = 0; j < HEADS; ++j) {
            const uint32_t key = hd[w * 16 + 2 * j], id = hd[w * 16 + 2 * j + 1];
            if (id == NO_ID || !offer(key, id)) break;
        }
    }
    // 2) The heap root now bounds the global keff-th entry from below: a worker whose 8th head does not beat it has
    //    no further entry in the global best; the others' remaining (unsorted) entries are read and offered.
    //    A worker that kept fewer entries than keff (kw = KW < keff) may have dropped some: its worst kept entry
    //    bounds them, so the merged list is exact down to the best such bound (bk, bi).
    const uint32_t kw = worker_list(keff);
    const bool truncating = kw < keff;
    bool bounded = false;
    uint32_t bk = 0, bi = NO_ID;
    if (kw > HEADS) {
        uint32_t pending[FULL_BATCH];
        uint32_t np = 0;
        for (uint32_t w = 0; w <= NW; ++w) {
            if (w < NW) {
                const uint32_t key = hd[w * 16 + 2 * (HEADS - 1)], id = hd[w * 16 + 2 * (HEADS - 1) + 1];
                if (id != NO_ID && (size < keff || better(key, id, hk[0], hi[0]))) pending[np++] = w;
            }
            if (np == FULL_BATCH || (w == NW && np)) {
                for (uint32_t q = 0; q < np; ++q) {
                    read_page(partial, row * NW + pending[q], lists_l1 + q * ENTRY_PAGE, ENTRY_PAGE - HEADS * 8,
                              HEADS * 8);
                }
                noc_async_read_barrier();
                barrier_compiler();
                for (uint32_t q = 0; q < np; ++q) {
                    const uint32_t* e = reinterpret_cast<const uint32_t*>(lists_l1 + q * ENTRY_PAGE);
                    const uint32_t* h8 = hd + pending[q] * 16 + 2 * (HEADS - 1);
                    uint32_t wk = h8[0], wi = h8[1];  // the worker's worst kept entry
                    uint32_t j = 0;
                    for (; j < kw - HEADS; ++j) {
                        if (e[2 * j + 1] == NO_ID) break;
                        offer(e[2 * j], e[2 * j + 1]);
                        if (better(wk, wi, e[2 * j], e[2 * j + 1])) {
                            wk = e[2 * j];
                            wi = e[2 * j + 1];
                        }
                    }
                    if (truncating && j == kw - HEADS && (!bounded || better(wk, wi, bk, bi))) {
                        bounded = true;
                        bk = wk;
                        bi = wi;
                    }
                }
                np = 0;
            }
        }
    }
    // Heap-sort best-first; the exact prefix ends at the bound.
    uint32_t merged = size;
    for (uint32_t n = size; n > 0; --n) {
        mkeys[n - 1] = hk[0];
        mids[n - 1] = hi[0];
        const uint32_t key = hk[n - 1], id = hi[n - 1];
        uint32_t i = 0;
        for (;;) {
            uint32_t l = 2 * i + 1;
            if (l >= n - 1) break;
            uint32_t c = l;
            if (l + 1 < n - 1 && better(hk[l], hi[l], hk[l + 1], hi[l + 1])) c = l + 1;
            if (!better(key, id, hk[c], hi[c])) break;
            hk[i] = hk[c];
            hi[i] = hi[c];
            i = c;
        }
        hk[i] = key;
        hi[i] = id;
    }
    if (bounded) {
        uint32_t v = 0;
        while (v < merged && !better(bk, bi, mkeys[v], mids[v])) ++v;
        merged = v;
    }
    // Full-vocabulary mass relative to the global maximum l_0 (the reference of the candidate weights).
    double zfull = bitsd(0xbff0000000000000ull);  // -1: not a full row
    if (full) {
        noc_async_read_barrier();
        barrier_compiler();
        const int64_t m0 = f2q24(keyf(mkeys[0]));
        uint64_t z = 0;
        const uint32_t* ms = reinterpret_cast<const uint32_t*>(mass_l1);
        for (uint32_t w = 0; w < NW; ++w) {
            const uint32_t* e = ms + w * (MASS_SLOT / 4);
            const int64_t mq = (int64_t)(((uint64_t)e[1] << 32) | e[0]);
            if (mq == NO_MAX) continue;
            const uint64_t s = ((uint64_t)e[3] << 32) | e[2];
            z += mq >= m0 ? s : rescale_mass(s, scale_q24((uint64_t)(m0 - mq), prm.w[18], prm.w[19]));
        }
        zfull = d_mul(d_u642d(z), bitsd(0x3d10000000000000ull));  // * 2^-46
    }
    double total, tail;
    uint32_t host;
    const RowParams rp = row_params(prm);
    uint32_t n = process(mkeys, merged, rp, zfull, weights, &total, &tail, &host);
    uint32_t token = mids[0];
    uint32_t rflags = host ? (R_HOST | host) : 0;
    if (!host && !(rp.flags & F_GREEDY)) {
        double u = uniform(prm.w[9], prm.w[10], reinterpret_cast<volatile uint32_t*>(pos_l1)[row] + 1, P_SAMPLE);
        const uint32_t i = draw(weights, n, total, tail, u);
        if (i == n) {
            rflags = R_HOST | R_TAIL;  // the draw landed beyond the candidates: the host redraws with the same u
        } else {
            token = mids[i];
        }
    }
#if SAMPLE_ROW_OP == 0
    o[0] = token;
    o[1] = rflags;
    o[2] = n;
    o[3] = 0;
    noc_async_write(out_l1, out.get_noc_addr(row), 16);
    noc_async_write_barrier();
    if (!rflags && !(prm.w[0] & F_NOCOMMIT)) commit(prm.w[11], token, h_l1);
#else
    res[W_COUNT] = n;
    res[W_FLAGS] = rflags;
    res[W_TOKEN] = token;
    res[3] = 0;
    for (uint32_t i = 0; i < n; ++i) {
        res[W_IDS + i] = mids[i];
        uint64_t b = dbits(d_div(d_f2d(weights[i]), total));
        res[W_PROBS + 2 * i] = (uint32_t)b;
        res[W_PROBS + 2 * i + 1] = (uint32_t)(b >> 32);
    }
    const uint64_t tb = dbits(d_div(tail, total)), zb = dbits(zfull);
    res[W_TAIL] = (uint32_t)tb;
    res[W_TAIL + 1] = (uint32_t)(tb >> 32);
    res[W_ZFULL] = (uint32_t)zb;
    res[W_ZFULL + 1] = (uint32_t)(zb >> 32);
    noc_async_write(res_l1, rowbuf.get_noc_addr(row), ROW_PAGE);
    noc_async_write_barrier();
#endif
#endif
#endif
#endif
}
