// SPDX-License-Identifier: Apache-2.0
// Exact on-device vLLM sampling of Inkling's FP32 softcapped logits (fast/sampling.py).  Ported from qwen36
// device_sampling_kernel.cpp (BF16 logits, + MTP draft/verify) to FP32 TILE logits; + the full-vocabulary path
// (F_FULL rows: top_k unset or > 64) and the speculative accept / commit step.
//
// SAMPLE_MODE 0 (scan, BRISC + NCRISC workers): each worker visits a disjoint stripe of vocabulary tiles
//   (t = worker + j * NW), applies vLLM's penalties to the logits (< SAMPLE_VALID) of every active row (per-slot
//   output-count / prompt-flag history + the row's extra tokens: the verified drafts before it), and keeps its exact
//   top-K by (penalized float32 value desc, token id asc).  Only rows 0..R-1 of faces 0 / 1 of each tile are read
//   (FP32: row r at r * 64 / 1024 + r * 64).  F_FULL rows also write their penalized row to plog (page r * TILES + t).
// SAMPLE_MODE 1 (row, one BRISC per row): exact heap merge of the worker lists into the global top-K.  Top-K rows: the
//   vLLM sampler (temperature, min_p, top-k, top-p, softmax) on the candidates, a counter-based draw, emit (unless the
//   top-k tie group was truncated: R_INEXACT, the host redraws).  F_FULL rows: the sorted candidates + the row
//   maximum go to rowstate (page row) for modes 2 / 3.
// SAMPLE_MODE 2 (weights, workers; F_FULL rows): integer weights w = exp((l - max) / T) as Q18.46 (wfix46; min_p:
//   w < tmin -> 0) of the stripe's tokens -> per-tile sums (tsum page row * NW + worker) and the stripe sum (wsum page
//   row, 64-byte slot per worker).  Integer sums: exact and independent of the summation order.
// SAMPLE_MODE 3 (draw, one BRISC per F_FULL row): Z = sum of the stripe sums; target = floor(u * Z) walks stripes
//   (worker order) -> tiles -> tokens (inverse CDF in that fixed order).  top-p: kept = the shortest candidate
//   prefix whose cumulative weight reaches Z - floor((1 - p) * Z) (vLLM's ascending-cumulative mask) when the
//   candidates reach it (exact), then the draw walks those candidates; otherwise (and for top-k > 64) rejection
//   sampling: up to 4 independent draws from the full distribution, the first one inside the nucleus / top-k set is
//   the token (a candidate is decided here, others by modes 4 / 5).  All 4 rejected: R_INEXACT (host redraw).
// SAMPLE_MODE 4 (tail, workers; pending F_FULL rows): per pending draw x: the stripe's weight ranked at or after x
//   (key < key_x, or equal key and id >= x) and the count of keys > key_x -> tailpart slot.
// SAMPLE_MODE 5 (final, one BRISC per pending row): the first accepted draw -> emit, else R_INEXACT.
// SAMPLE_MODE 6 (accept, one BRISC; speculative verify of R rows): accepted k = longest prefix with
//   sampled(row i) == draft i + 1 (rows without R_INEXACT); history += drafts 1..k and the token of row k (unless
//   inexact); the R sampled tokens -> words 0..R-1 of tok (the verify ids row).
// Emit: [token, flags, n, 0] -> page `row` of sel (SAMPLE_SEL_PAGE bytes per page), token -> word 0 of tok (row 0,
//   SAMPLE_SPEC 0: the decode trace's embedding index row), history += token (unless F_NOCOMMIT / inexact).
// Common runtime args: logits, heads, params, partial, sel, tok, pos, aux (draft log: extra tokens), plog, rowstate,
//   wsum, tsum, tailpart, then the per-slot history addresses.
#include <cstdint>
#include "read_vc.hpp"
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "sampling_math.hpp"

namespace {
using namespace dsm;
constexpr uint32_t R = SAMPLE_ROWS;
constexpr uint32_t K = SAMPLE_K;
constexpr uint32_t NW = SAMPLE_WORKERS;
constexpr uint32_t TILES = SAMPLE_TILES;
constexpr uint32_t NVALID = SAMPLE_VALID;  // sampled columns (the unembedding padding is skipped)
constexpr uint32_t RB = R * 64;                // bytes of R FP32 face rows
constexpr uint32_t TILE_BYTES = 2 * RB;        // staged bytes per vocabulary tile (faces 0 and 1)
constexpr uint32_t half_offset(uint32_t r, uint32_t h) { return h * RB + r * 64; }
constexpr uint32_t BATCH = SAMPLE_BATCH;
constexpr uint32_t ENTRY_PAGE = K * 8;  // (key, id) pairs per worker list
constexpr uint32_t STRIPE = (TILES + NW - 1) / NW;  // tiles per worker (at most)
constexpr uint32_t TSUM_PAGE = ((STRIPE * 8 + 63) / 64) * 64;
constexpr uint32_t RS_PAGE = 1024;  // rowstate: 64 state words | K (key, id) candidate pairs
constexpr uint32_t PLOG_PAGE = 128;
constexpr uint32_t HIST_ARG0 = 13;
constexpr uint32_t ND = 4;  // rejection draws
// rowstate words
constexpr uint32_t RS_MAX = 0, RS_STATUS = 1, RS_NCAND = 2, RS_Z = 4, RS_LZ = 6, RS_DRAWS = 8;  // draws: [id, v, state]
constexpr uint32_t ST_NONE = 0, ST_PENDING = 2;

struct Params {
    uint32_t w[32];
};

inline RowParams row_params(const Params& p) {
    return RowParams{p.w[0], bitsf(p.w[1]), p.w[2], bitsf(p.w[4]), bitsf(p.w[13]), p.w[18], p.w[19]};
}

inline void barrier_compiler() { asm volatile("" ::: "memory"); }

inline auto acc(uint32_t arg, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(arg), page);
}

template <typename A>
inline void read_page(const A& a, uint32_t page, uint32_t l1, uint32_t bytes, uint32_t offset = 0) {
    noc_async_read(a.get_noc_addr(page, offset), l1, bytes);
}

// Per-slot history tensors: common runtime args HIST_ARG0.. hold their base addresses.
inline auto hist_acc(uint32_t slot) { return acc(HIST_ARG0 + slot, 128); }

// history[slot][token] += 1 (read-modify-write of the 128-byte page).
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

inline uint64_t u64(const volatile uint32_t* w) { return (uint64_t)w[0] | ((uint64_t)w[1] << 32); }
inline void put64(volatile uint32_t* w, uint64_t v) {
    w[0] = (uint32_t)v;
    w[1] = (uint32_t)(v >> 32);
}

// Row params -> the weight of a penalized logit (bits v) below the row maximum mq (min_p mask applied).
struct Weigher {
    int64_t mq;
    uint32_t c_lo, c_hi;
    uint64_t tmin;
    inline uint64_t operator()(uint32_t v) const {
        const uint64_t w = wfix46(mq, fixq24(v), c_lo, c_hi);
        return w < tmin ? 0 : w;
    }
};
inline Weigher weigher(const uint32_t* p, uint32_t max_bits) {
    return Weigher{fixq24(max_bits), p[18], p[19], (uint64_t)p[21] | ((uint64_t)p[22] << 32)};
}

// [token, flags, n, 0] -> sel page `row`; token -> tok word 0 (row 0 of a decode step); history += token.
inline void emit(const uint32_t* p, uint32_t row, uint32_t token, uint32_t flags, uint32_t n, uint32_t l1) {
    const auto sel = acc(4, SAMPLE_SEL_PAGE);
    auto o = reinterpret_cast<volatile uint32_t*>(l1);
    o[0] = token;
    o[1] = flags;
    o[2] = n;
    o[3] = 0;
    noc_async_write(l1, sel.get_noc_addr(row), 16);
#if SAMPLE_SPEC == 0
    if (row == 0) {
        const auto tok = acc(5, SAMPLE_TOK_PAGE);
        auto tk = reinterpret_cast<volatile uint32_t*>(l1 + 64);
        tk[0] = token;
        noc_async_write(l1 + 64, tok.get_noc_addr(0), 4);
    }
#endif
    noc_async_write_barrier();
    if (!(flags & R_INEXACT) && !(p[0] & F_NOCOMMIT)) commit(p[11], token, l1 + 128);
}

// Drawn token's absolute position = the row's input position (pos word `row`) + 1.
inline uint32_t row_position(uint32_t pos_l1, uint32_t row) {
    return reinterpret_cast<volatile uint32_t*>(pos_l1)[row] + 1;
}

// Inverse-CDF walk of the full distribution of F_FULL row `row`: stripes (worker order, sums in wsum_l1 64-byte
// slots) -> tiles (tsum page) -> tokens (plog page) for a target in [0, Z) -> (id, penalized value bits).
inline void walk(uint32_t row, uint64_t target, const Weigher& wt, uint32_t wsum_l1, uint32_t tsum_l1,
                 uint32_t plog_l1, uint32_t* id_out, uint32_t* v_out) {
    const auto tsum = acc(11, TSUM_PAGE);
    const auto plog = acc(8, PLOG_PAGE);
    uint64_t cum = 0;
    uint32_t w = 0;
    for (; w + 1 < NW; ++w) {
        const uint64_t s = u64(reinterpret_cast<volatile uint32_t*>(wsum_l1 + w * 64));
        if (cum + s > target) break;
        cum += s;
    }
    read_page(tsum, row * NW + w, tsum_l1, TSUM_PAGE);
    noc_async_read_barrier();
    barrier_compiler();
    const auto ts = reinterpret_cast<volatile uint32_t*>(tsum_l1);
    uint32_t j = 0;
    uint32_t last = 0;
    for (uint32_t t = w; t < TILES; t += NW, ++j) {
        last = j;
        const uint64_t s = u64(ts + 2 * j);
        if (s == 0) continue;
        if (cum + s > target) break;
        cum += s;
    }
    const uint32_t t = w + last * NW;
    read_page(plog, row * TILES + t, plog_l1, PLOG_PAGE);
    noc_async_read_barrier();
    barrier_compiler();
    const auto v = reinterpret_cast<volatile uint32_t*>(plog_l1);
    uint32_t pick = NO_ID, pv = 0;
    for (uint32_t c = 0; c < 32; ++c) {
        const uint32_t id = t * 32 + c;
        if (id >= NVALID) break;
        const uint64_t x = wt(v[c]);
        if (x == 0) continue;
        pick = id;
        pv = v[c];
        if (cum + x > target) break;
        cum += x;
    }
    *id_out = pick;
    *v_out = pv;
}
}  // namespace
#endif

void kernel_main() {
    ink_set_read_vc();
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#if SAMPLE_MODE == 0
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t base = get_write_ptr(0) + (is_ncrisc ? SAMPLE_SCRATCH_HALF : 0);
    const auto logits = acc(0, 4096);
    const auto params = acc(2, 128);
    const auto partial = acc(3, ENTRY_PAGE);
    const auto heads = acc(1, NW * 64);
    const auto aux = acc(7, 128);
    const auto plog = acc(8, PLOG_PAGE);
    // Scratch: params R*128 | heap keys R*K*4 | heap ids R*K*4 | logits BATCH*TILE_BYTES | hist BATCH*R*128 | aux 7*128
    const uint32_t p_l1 = base;
    const uint32_t keys_l1 = p_l1 + R * 128;
    const uint32_t ids_l1 = keys_l1 + R * K * 4;
    const uint32_t tile_l1 = ids_l1 + R * K * 4;
    const uint32_t hist_l1 = tile_l1 + BATCH * TILE_BYTES;
    const uint32_t aux_l1 = hist_l1 + BATCH * R * 128;
    for (uint32_t r = 0; r < R; ++r) read_page(params, r, p_l1 + r * 128, 128);
    noc_async_read_barrier();
    barrier_compiler();
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(p_l1);
    bool any_pen = false, any_full = false;
    uint32_t n_aux = 0;
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t fl = pw[r * 32];
        any_pen |= (fl & (F_ACTIVE | F_PENALTY)) == (F_ACTIVE | F_PENALTY);
        any_full |= (fl & (F_ACTIVE | F_FULL)) == (F_ACTIVE | F_FULL);
        if ((fl & (F_ACTIVE | F_PENALTY | F_EXTRA)) == (F_ACTIVE | F_PENALTY | F_EXTRA) && pw[r * 32 + 20] > n_aux) {
            n_aux = pw[r * 32 + 20];
        }
    }
    // extra tokens of row r: words 0 of aux pages 0 .. p[20] - 1 (the drafts verified before the row)
    for (uint32_t i = 0; i < n_aux; ++i) read_page(aux, i, aux_l1 + i * 128, 64);
    noc_async_read_barrier();
    barrier_compiler();
    uint32_t extra[8];
    for (uint32_t i = 0; i < n_aux && i < 8; ++i) extra[i] = reinterpret_cast<const uint32_t*>(aux_l1 + i * 128)[0];
    // Per row: a bounded min-heap (worst entry at the root) of the best keff entries by (key desc, id asc), in the
    // RISC's local memory for R <= 2, else in L1.
#if SAMPLE_ROWS <= 2
    uint32_t hk[R * K], hi[R * K];
#else
    uint32_t* hk = reinterpret_cast<uint32_t*>(keys_l1);
    uint32_t* hi = reinterpret_cast<uint32_t*>(ids_l1);
#endif
    for (uint32_t first = worker; first < TILES; first += NW * BATCH) {
        uint32_t count = 0;
        for (uint32_t t = first; t < TILES && count < BATCH; t += NW, ++count) {
            uint32_t dst = tile_l1 + count * TILE_BYTES;
            read_page(logits, t, dst, RB, 0);
            read_page(logits, t, dst + RB, RB, 1024);
            if (any_pen) {
                for (uint32_t r = 0; r < R; ++r) {
                    if ((pw[r * 32] & (F_ACTIVE | F_PENALTY)) == (F_ACTIVE | F_PENALTY)) {
                        read_page(hist_acc(pw[r * 32 + 11]), t, hist_l1 + (count * R + r) * 128, 128);
                    }
                }
            }
        }
        noc_async_read_barrier();
        barrier_compiler();
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t* p = pw + r * 32;
            const uint32_t fl = p[0];
            if (!(fl & F_ACTIVE)) continue;
            const uint32_t keff = p[14];
            uint32_t* rk = hk + r * K;
            uint32_t* ri = hi + r * K;
            const bool pen = (fl & F_PENALTY) != 0;
            const bool full = (fl & F_FULL) != 0;
            // Monotone penalties (presence, frequency >= 0, repetition >= 1) never raise a logit: a raw value at or
            // below the threshold needs no history lookup (top-K rows; F_FULL rows penalize every logit).
            const bool mono = (fl & F_MONO) != 0;
            const bool rep_on = (fl & F_REP) != 0;
            const uint32_t nex = (pen && (fl & F_EXTRA)) ? p[20] : 0;
            const float rep = bitsf(p[7]), rrep = bitsf(p[8]), freq = bitsf(p[6]), pres = bitsf(p[5]);
            uint32_t size = p[15] ? p[16] : 0;  // persisted across batches in words 15/16 (scratch copy)
            uint32_t th = size < keff ? 0u : rk[0];
            const bool fast_ok = !full && (!pen || mono);
            uint32_t c_t = 0;
            for (uint32_t t = first; t < TILES && c_t < BATCH; t += NW, ++c_t) {
                const uint32_t id0 = t * 32;
                const uint32_t* h = reinterpret_cast<const uint32_t*>(hist_l1 + (c_t * R + r) * 128);
                for (uint32_t half = 0; half < 2; ++half) {
                    uint32_t* f = reinterpret_cast<uint32_t*>(tile_l1 + c_t * TILE_BYTES + half_offset(r, half));
                    for (uint32_t j = 0; j < 16; ++j) {
                        const uint32_t c = 16 * half + j;
                        const uint32_t id = id0 + c;
                        if (id >= NVALID) break;
                        uint32_t fb = f[j];
                        uint32_t key = fkey(fb);
                        if (key <= th && fast_ok) continue;
                        if (pen) {
                            const uint32_t e = h[c];
                            uint32_t cnt = e & 0x7fffffffu;
                            for (uint32_t i = 0; i < nex; ++i) cnt += extra[i] == id ? 1u : 0u;
                            const bool in_prompt = (e >> 31) != 0;
                            if (cnt || (rep_on && in_prompt)) {
                                fb = penalize(fb, cnt, in_prompt, rep_on, rep, rrep, freq, pres);
                                key = fkey(fb);
                                if (full) f[j] = fb;
                            }
                        }
                        if (key <= th) continue;
                        if (size < keff) {
                            // Push: sift up toward the root while the new entry is worse than its parent.
                            uint32_t i = size++;
                            while (i) {
                                const uint32_t parent = (i - 1) >> 1;
                                if (!(key < rk[parent] || (key == rk[parent] && id > ri[parent]))) break;
                                rk[i] = rk[parent];
                                ri[i] = ri[parent];
                                i = parent;
                            }
                            rk[i] = key;
                            ri[i] = id;
                            if (size == keff) th = rk[0];
                            continue;
                        }
                        // Replace the worst (scan order is ascending id, so an equal key never wins), sift down.
                        uint32_t i = 0;
                        for (;;) {
                            uint32_t l = 2 * i + 1;
                            if (l >= keff) break;
                            uint32_t w = l;
                            if (l + 1 < keff && (rk[l + 1] < rk[l] || (rk[l + 1] == rk[l] && ri[l + 1] > ri[l]))) {
                                w = l + 1;
                            }
                            if (!(rk[w] < key || (rk[w] == key && ri[w] > id))) break;
                            rk[i] = rk[w];
                            ri[i] = ri[w];
                            i = w;
                        }
                        rk[i] = key;
                        ri[i] = id;
                        th = rk[0];
                    }
                }
                if (full) {  // the penalized row of this tile (columns >= NVALID keep their raw values; never read)
                    const uint32_t src = tile_l1 + c_t * TILE_BYTES;
                    noc_async_write(src + half_offset(r, 0), plog.get_noc_addr(r * TILES + t, 0), 64);
                    noc_async_write(src + half_offset(r, 1), plog.get_noc_addr(r * TILES + t, 64), 64);
                }
            }
            // Persist the heap size for the next batch (params scratch copy, words 15/16).
            const_cast<uint32_t*>(p)[15] = 1;
            const_cast<uint32_t*>(p)[16] = size;
        }
        if (any_full) noc_async_write_barrier();  // the staged tiles are overwritten by the next batch
    }
    // Heap-sort each row best-first and write interleaved (key, id) pages, one per (row, worker).
    uint32_t* stage = reinterpret_cast<uint32_t*>(tile_l1);
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t* p = pw + r * 32;
        uint32_t* rk = hk + r * K;
        uint32_t* ri = hi + r * K;
        uint32_t size = (p[0] & F_ACTIVE) && p[15] ? p[16] : 0;
        for (uint32_t i = 0; i < K; ++i) {
            stage[2 * i] = 0;
            stage[2 * i + 1] = NO_ID;
        }
        for (uint32_t n = size; n > 0; --n) {
            // Pop the worst entry into slot n-1.
            stage[2 * (n - 1)] = rk[0];
            stage[2 * (n - 1) + 1] = ri[0];
            const uint32_t key = rk[n - 1], id = ri[n - 1];
            uint32_t i = 0;
            for (;;) {
                uint32_t l = 2 * i + 1;
                if (l >= n - 1) break;
                uint32_t w = l;
                if (l + 1 < n - 1 && (rk[l + 1] < rk[l] || (rk[l + 1] == rk[l] && ri[l + 1] > ri[l]))) w = l + 1;
                if (!(rk[w] < key || (rk[w] == key && ri[w] > id))) break;
                rk[i] = rk[w];
                ri[i] = ri[w];
                i = w;
            }
            rk[i] = key;
            ri[i] = id;
        }
        noc_async_write(tile_l1, partial.get_noc_addr(r * NW + worker), ENTRY_PAGE);
        // The first 8 entries also go to the row's contiguous head page (one read for the merge).
        noc_async_write(tile_l1, heads.get_noc_addr(r, worker * 64), 64);
        noc_async_write_barrier();
    }
#elif SAMPLE_MODE == 1
#if defined(COMPILE_FOR_BRISC)
    const uint32_t row = get_arg_val<uint32_t>(0);
    const uint32_t base = get_write_ptr(0);
    const auto params = acc(2, 128);
    const auto partial = acc(3, ENTRY_PAGE);
    const auto pos = acc(6, 64);
    const auto heads = acc(1, NW * 64);
    const auto rowstate = acc(9, RS_PAGE);
    // Scratch: params 128 | pos 64 | out 256 | heads NW*64 | chunk 64 | rowstate RS_PAGE | sampler scratch
    const uint32_t p_l1 = base;
    const uint32_t pos_l1 = p_l1 + 128;
    const uint32_t out_l1 = pos_l1 + 64;
    const uint32_t heads_l1 = out_l1 + 256;
    const uint32_t chunk_l1 = heads_l1 + NW * 64;
    const uint32_t rs_l1 = chunk_l1 + 64;
    const uint32_t res_l1 = rs_l1 + RS_PAGE;
    auto mkeys = reinterpret_cast<uint32_t*>(res_l1);
    auto mids = mkeys + K;
    auto weights = reinterpret_cast<float*>(mids + K);
    auto xs = weights + K;
    read_page(params, row, p_l1, 128);
    read_page(pos, 0, pos_l1, 64);
    read_page(heads, row, heads_l1, NW * 64);
    noc_async_read_barrier();
    barrier_compiler();
    Params prm;
    for (uint32_t i = 0; i < 32; ++i) prm.w[i] = reinterpret_cast<const uint32_t*>(p_l1)[i];
    if (!(prm.w[0] & F_ACTIVE)) return;
    const uint32_t keff = prm.w[14];
    // Exact merge of the worker lists (each sorted by key desc, id asc) into the global best keff, kept in a
    // local-memory min-heap (worst at the root). A list is abandoned at its first entry that cannot enter.
    uint32_t hk[K], hi[K];
    uint32_t size = 0;
    const uint32_t* hd = reinterpret_cast<const uint32_t*>(heads_l1);
    const uint32_t* ck = reinterpret_cast<const uint32_t*>(chunk_l1);
    for (uint32_t w = 0; w < NW; ++w) {
        const uint32_t* entries = hd + w * 16;
        for (uint32_t j = 0; j < keff; ++j) {
            if (j && j % 8 == 0) {
                read_page(partial, row * NW + w, chunk_l1, 64, j * 8);
                noc_async_read_barrier();
                barrier_compiler();
                entries = ck;
            }
            const uint32_t key = entries[2 * (j % 8)], id = entries[2 * (j % 8) + 1];
            if (id == NO_ID) break;
            if (size < keff) {
                uint32_t i = size++;
                while (i) {
                    const uint32_t parent = (i - 1) >> 1;
                    if (!(key < hk[parent] || (key == hk[parent] && id > hi[parent]))) break;
                    hk[i] = hk[parent];
                    hi[i] = hi[parent];
                    i = parent;
                }
                hk[i] = key;
                hi[i] = id;
                continue;
            }
            if (!(key > hk[0] || (key == hk[0] && id < hi[0]))) break;
            uint32_t i = 0;
            for (;;) {
                uint32_t l = 2 * i + 1;
                if (l >= keff) break;
                uint32_t c = l;
                if (l + 1 < keff && (hk[l + 1] < hk[l] || (hk[l + 1] == hk[l] && hi[l + 1] > hi[l]))) c = l + 1;
                if (!(hk[c] < key || (hk[c] == key && hi[c] > id))) break;
                hk[i] = hk[c];
                hi[i] = hi[c];
                i = c;
            }
            hk[i] = key;
            hi[i] = id;
        }
    }
    // Heap-sort best-first.
    const uint32_t merged = size;
    for (uint32_t n = size; n > 0; --n) {
        mkeys[n - 1] = hk[0];
        mids[n - 1] = hi[0];
        const uint32_t key = hk[n - 1], id = hi[n - 1];
        uint32_t i = 0;
        for (;;) {
            uint32_t l = 2 * i + 1;
            if (l >= n - 1) break;
            uint32_t c = l;
            if (l + 1 < n - 1 && (hk[l + 1] < hk[l] || (hk[l + 1] == hk[l] && hi[l + 1] > hi[l]))) c = l + 1;
            if (!(hk[c] < key || (hk[c] == key && hi[c] > id))) break;
            hk[i] = hk[c];
            hi[i] = hi[c];
            i = c;
        }
        hk[i] = key;
        hi[i] = id;
    }
    if (prm.w[0] & F_FULL) {
        // the row maximum + the sorted candidates (penalized value bits, id) for modes 2 / 3
        auto rs = reinterpret_cast<volatile uint32_t*>(rs_l1);
        for (uint32_t i = 0; i < 64; ++i) rs[i] = 0;
        rs[RS_MAX] = keyf(mkeys[0]);
        rs[RS_STATUS] = ST_NONE;
        rs[RS_NCAND] = merged;
        for (uint32_t i = 0; i < K; ++i) {
            rs[64 + 2 * i] = i < merged ? keyf(mkeys[i]) : 0;
            rs[64 + 2 * i + 1] = i < merged ? mids[i] : NO_ID;
        }
        noc_async_write(rs_l1, rowstate.get_noc_addr(row), RS_PAGE);
        noc_async_write_barrier();
        return;
    }
    double total;
    bool inexact;
    const RowParams rp = row_params(prm);
    uint32_t n = process(mkeys, keff, rp, xs, weights, &total, &inexact);
    uint32_t token = mids[0];
    if (!(rp.flags & F_GREEDY)) {
        double u = uniform(prm.w[9], prm.w[10], row_position(pos_l1, row), P_SAMPLE);
        token = mids[draw(weights, n, total, u)];
    }
    emit(prm.w, row, token, inexact ? R_INEXACT : 0, n, out_l1);
#endif
#elif SAMPLE_MODE == 2 || SAMPLE_MODE == 4
    // weights (2) / tail (4): workers over their stripes of every active F_FULL row (mode 4: pending rows only)
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t base = get_write_ptr(0) + (is_ncrisc ? SAMPLE_SCRATCH_HALF : 0);
    const auto params = acc(2, 128);
    const auto plog = acc(8, PLOG_PAGE);
    const auto rowstate = acc(9, RS_PAGE);
    // Scratch: params 128 | rowstate 256 | slot 64 | tsum TSUM_PAGE | plog BATCH * 128
    const uint32_t p_l1 = base;
    const uint32_t rs_l1 = p_l1 + 128;
    const uint32_t slot_l1 = rs_l1 + 256;
    const uint32_t ts_l1 = slot_l1 + 64;
    const uint32_t pl_l1 = ts_l1 + TSUM_PAGE;
    for (uint32_t row = 0; row < R; ++row) {
        read_page(params, row, p_l1, 128);
        read_page(rowstate, row, rs_l1, 256);
        noc_async_read_barrier();
        barrier_compiler();
        const uint32_t* p = reinterpret_cast<const uint32_t*>(p_l1);
        const volatile uint32_t* rs = reinterpret_cast<volatile uint32_t*>(rs_l1);
        if ((p[0] & (F_ACTIVE | F_FULL)) != (F_ACTIVE | F_FULL)) continue;
#if SAMPLE_MODE == 4
        if (rs[RS_STATUS] != ST_PENDING) continue;
        uint32_t dk[ND], did[ND];
        bool live[ND];
        for (uint32_t i = 0; i < ND; ++i) {
            did[i] = rs[RS_DRAWS + 4 * i];
            dk[i] = fkey(rs[RS_DRAWS + 4 * i + 1]);
            live[i] = rs[RS_DRAWS + 4 * i + 2] == 0 && did[i] != NO_ID;
        }
        uint64_t tail[ND] = {0, 0, 0, 0};
        uint32_t cnt[ND] = {0, 0, 0, 0};
        const bool topp = (p[0] & F_TOPP) != 0;
#endif
        const Weigher wt = weigher(p, rs[RS_MAX]);
        uint64_t stripe = 0;
        auto ts = reinterpret_cast<volatile uint32_t*>(ts_l1);
        uint32_t j = 0;
        for (uint32_t first = worker; first < TILES; first += NW * BATCH) {
            uint32_t count = 0;
            for (uint32_t t = first; t < TILES && count < BATCH; t += NW, ++count) {
                read_page(plog, row * TILES + t, pl_l1 + count * PLOG_PAGE, PLOG_PAGE);
            }
            noc_async_read_barrier();
            barrier_compiler();
            count = 0;
            for (uint32_t t = first; t < TILES && count < BATCH; t += NW, ++count, ++j) {
                const uint32_t* v = reinterpret_cast<const uint32_t*>(pl_l1 + count * PLOG_PAGE);
                uint64_t s = 0;
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t id = t * 32 + c;
                    if (id >= NVALID) break;
#if SAMPLE_MODE == 2
                    s += wt(v[c]);
#else
                    const uint32_t key = fkey(v[c]);
                    const uint64_t x = topp ? wt(v[c]) : 0;
                    for (uint32_t i = 0; i < ND; ++i) {
                        if (!live[i]) continue;
                        if (key < dk[i] || (key == dk[i] && id >= did[i])) tail[i] += x;
                        if (key > dk[i]) ++cnt[i];
                    }
#endif
                }
#if SAMPLE_MODE == 2
                put64(ts + 2 * j, s);
                stripe += s;
#endif
            }
        }
        auto slot = reinterpret_cast<volatile uint32_t*>(slot_l1);
        for (uint32_t i = 0; i < 16; ++i) slot[i] = 0;
#if SAMPLE_MODE == 2
        for (uint32_t i = 2 * j; i < TSUM_PAGE / 4; ++i) ts[i] = 0;
        put64(slot, stripe);
        noc_async_write(ts_l1, acc(11, TSUM_PAGE).get_noc_addr(row * NW + worker), TSUM_PAGE);
        noc_async_write(slot_l1, acc(10, NW * 64).get_noc_addr(row, worker * 64), 64);
#else
        for (uint32_t i = 0; i < ND; ++i) {
            put64(slot + 4 * i, tail[i]);
            slot[4 * i + 2] = cnt[i];
        }
        noc_async_write(slot_l1, acc(12, NW * 64).get_noc_addr(row, worker * 64), 64);
#endif
        noc_async_write_barrier();
    }
#elif SAMPLE_MODE == 3 || SAMPLE_MODE == 5
#if defined(COMPILE_FOR_BRISC)
    // draw (3) / final (5): one core per row
    const uint32_t row = get_arg_val<uint32_t>(0);
    const uint32_t base = get_write_ptr(0);
    const auto params = acc(2, 128);
    const auto pos = acc(6, 64);
    const auto rowstate = acc(9, RS_PAGE);
    // Scratch: params 128 | pos 64 | out 256 | rowstate RS_PAGE | tsum TSUM_PAGE | plog 128 | sums NW*64
    const uint32_t p_l1 = base;
    const uint32_t pos_l1 = p_l1 + 128;
    const uint32_t out_l1 = pos_l1 + 64;
    const uint32_t rs_l1 = out_l1 + 256;
    const uint32_t ts_l1 = rs_l1 + RS_PAGE;
    const uint32_t pl_l1 = ts_l1 + TSUM_PAGE;
    const uint32_t sums_l1 = pl_l1 + 128;
    read_page(params, row, p_l1, 128);
    read_page(pos, 0, pos_l1, 64);
    read_page(rowstate, row, rs_l1, RS_PAGE);
    noc_async_read_barrier();
    barrier_compiler();
    const uint32_t* p = reinterpret_cast<const uint32_t*>(p_l1);
    auto rs = reinterpret_cast<volatile uint32_t*>(rs_l1);
    if ((p[0] & (F_ACTIVE | F_FULL)) != (F_ACTIVE | F_FULL)) return;
    const bool topp = (p[0] & F_TOPP) != 0, topk = (p[0] & F_TOPK) != 0;
    const uint32_t k = p[2];
#if SAMPLE_MODE == 3
    read_page(acc(10, NW * 64), row, sums_l1, NW * 64);
    noc_async_read_barrier();
    barrier_compiler();
    const Weigher wt = weigher(p, rs[RS_MAX]);
    uint64_t Z = 0;
    for (uint32_t w = 0; w < NW; ++w) Z += u64(reinterpret_cast<volatile uint32_t*>(sums_l1 + w * 64));
    const uint32_t position = row_position(pos_l1, row);
    const uint32_t ncand = rs[RS_NCAND];
    const volatile uint32_t* cand = rs + 64;
    uint64_t Lz = 0;
    if (topp) {
        Lz = mulshr(Z, p[23], p[24]);  // floor((1 - p) * Z), (1 - p) = p[23] / 2^p[24] exactly
        // vLLM masks rank r when the ascending cumulative (Z - C_{r-1}) <= (1 - p) Z: kept = the shortest candidate
        // prefix with C_j >= Z - Lz.
        uint64_t C = 0;
        uint32_t kept = 0;
        for (uint32_t i = 0; i < ncand; ++i) {
            C += wt(cand[2 * i]);
            if (C >= Z - Lz) {
                kept = i + 1;
                break;
            }
        }
        if (kept) {
            const uint64_t target = mulshr(C, uniform_bits(p[9], p[10], position, P_SAMPLE), 52);
            uint64_t cum = 0;
            uint32_t pick = 0;
            for (uint32_t i = 0; i < kept; ++i) {
                const uint64_t x = wt(cand[2 * i]);
                if (x == 0) continue;
                pick = i;
                if (cum + x > target) break;
                cum += x;
            }
            emit(p, row, cand[2 * pick + 1], 0, kept, out_l1);
            return;
        }
    }
    // full-distribution draws: one (no top-p / top-k > 64), else rejection sampling
    const uint32_t nd = (topp || topk) ? ND : 1;
    bool pending = false;
    for (uint32_t i = 0; i < ND; ++i) {
        rs[RS_DRAWS + 4 * i] = NO_ID;
        rs[RS_DRAWS + 4 * i + 2] = 2;
    }
    for (uint32_t i = 0; i < nd; ++i) {
        const uint32_t purpose = i == 0 ? P_SAMPLE : P_REJ + i - 1;
        const uint64_t target = mulshr(Z, uniform_bits(p[9], p[10], position, purpose), 52);
        uint32_t id, v;
        walk(row, target, wt, sums_l1, ts_l1, pl_l1, &id, &v);
        if (nd == 1) {
            emit(p, row, id, 0, 0, out_l1);
            return;
        }
        // a candidate is decided here: every larger key is a candidate (rank = its index's tie-group start)
        uint32_t state = 0;
        for (uint32_t c = 0; c < ncand; ++c) {
            if (cand[2 * c + 1] != id) continue;
            uint32_t greater = c;
            while (greater && cand[2 * (greater - 1)] == cand[2 * c]) --greater;
            state = (!topk || greater < k) ? 1 : 2;  // top-p unresolved: every candidate is in the nucleus
            break;
        }
        rs[RS_DRAWS + 4 * i] = id;
        rs[RS_DRAWS + 4 * i + 1] = v;
        rs[RS_DRAWS + 4 * i + 2] = state;
        if (state == 1) {
            if (!pending) {
                emit(p, row, id, 0, i + 1, out_l1);
                return;
            }
            break;
        }
        if (state == 0) pending = true;
    }
    if (!pending) {  // every draw was a rejected candidate: the host redraws
        emit(p, row, rs[RS_DRAWS], R_INEXACT, 0, out_l1);
        return;
    }
    rs[RS_STATUS] = ST_PENDING;
    put64(rs + RS_Z, Z);
    put64(rs + RS_LZ, Lz);
    noc_async_write(rs_l1, rowstate.get_noc_addr(row), 256);
    noc_async_write_barrier();
#else
    if (rs[RS_STATUS] != ST_PENDING) return;
    read_page(acc(12, NW * 64), row, sums_l1, NW * 64);
    noc_async_read_barrier();
    barrier_compiler();
    const uint64_t Lz = u64(rs + RS_LZ);
    for (uint32_t i = 0; i < ND; ++i) {
        const uint32_t id = rs[RS_DRAWS + 4 * i], state = rs[RS_DRAWS + 4 * i + 2];
        if (id == NO_ID) break;
        bool ok = state == 1;
        if (state == 0) {
            uint64_t tail = 0;
            uint32_t greater = 0;
            for (uint32_t w = 0; w < NW; ++w) {
                const volatile uint32_t* s = reinterpret_cast<volatile uint32_t*>(sums_l1 + w * 64) + 4 * i;
                tail += u64(s);
                greater += s[2];
            }
            ok = (!topp || tail > Lz) && (!topk || greater < k);
        }
        if (ok) {
            emit(p, row, id, 0, i + 1, out_l1);
            return;
        }
    }
    emit(p, row, rs[RS_DRAWS], R_INEXACT, 0, out_l1);
#endif
#endif
#elif SAMPLE_MODE == 6
#if defined(COMPILE_FOR_BRISC)
    // accept + commit of a speculative verify of SAMPLE_ROWS rows (drafts: words 0 of aux pages 0 .. R - 2)
    const uint32_t base = get_write_ptr(0);
    const auto params = acc(2, 128);
    const auto sel = acc(4, SAMPLE_SEL_PAGE);
    const auto tok = acc(5, SAMPLE_TOK_PAGE);
    const auto aux = acc(7, 128);
    const uint32_t p_l1 = base;
    const uint32_t sel_l1 = p_l1 + 128;           // R pages of 64 bytes
    const uint32_t aux_l1 = sel_l1 + R * 64;      // R - 1 pages of 128 bytes
    const uint32_t ids_l1 = aux_l1 + R * 128;
    const uint32_t h_l1 = ids_l1 + 64;
    read_page(params, 0, p_l1, 128);
    for (uint32_t r = 0; r < R; ++r) read_page(sel, r, sel_l1 + r * 64, 16);
    for (uint32_t r = 0; r + 1 < R; ++r) read_page(aux, r, aux_l1 + r * 128, 64);
    noc_async_read_barrier();
    barrier_compiler();
    const uint32_t* p = reinterpret_cast<const uint32_t*>(p_l1);
    auto ids = reinterpret_cast<volatile uint32_t*>(ids_l1);
    for (uint32_t r = 0; r < R; ++r) ids[r] = reinterpret_cast<volatile uint32_t*>(sel_l1 + r * 64)[0];
    noc_async_write(ids_l1, tok.get_noc_addr(0), R * 4);
    noc_async_write_barrier();
    if (!(p[0] & F_ACTIVE)) return;
    uint32_t k = 0;
    while (k + 1 < R) {
        const volatile uint32_t* s = reinterpret_cast<volatile uint32_t*>(sel_l1 + k * 64);
        if ((s[1] & R_INEXACT) || s[0] != reinterpret_cast<volatile uint32_t*>(aux_l1 + k * 128)[0]) break;
        ++k;
    }
    for (uint32_t i = 0; i < k; ++i) commit(p[11], ids[i], h_l1);
    const volatile uint32_t* sk = reinterpret_cast<volatile uint32_t*>(sel_l1 + k * 64);
    if (!(sk[1] & R_INEXACT)) commit(p[11], sk[0], h_l1);
#endif
#endif
#endif
}
