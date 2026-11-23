// SPDX-License-Identifier: Apache-2.0
// Exact on-device vLLM sampling and lossless speculative rejection sampling (device_sampling.py).
//
// SAMPLE_MODE 0 (scan, BRISC + NCRISC workers): each worker visits a disjoint stripe of vocabulary tiles, applies
//   vLLM's penalties to every logit of every active row (from the per-slot output-count/prompt-flag history plus the
//   row's speculative extra tokens), and keeps its exact top-K by (penalized float32 value desc, token id asc).
// SAMPLE_MODE 1 (row, one BRISC per row): exact heap merge of the worker lists into the global top-K, then the
//   vLLM sampler (temperature, min_p, top-k, top-p, softmax) and, per SAMPLE_ROW_OP:
//     0 SAMPLE: counter-based draw, token + flags to out[row], history += token;
//     1 DRAFT:  draw (or argmax point mass) from the processed draft distribution q, store q in qbuf page
//               SAMPLE_Q_PAGE and the token in word SAMPLE_OUT_WORD of out page 0;
//     2 PREP:   store the processed target distribution in rowbuf page `row` for SAMPLE_MODE 2.
// SAMPLE_MODE 2 (chain verify, one BRISC): sequential speculative rejection sampling of the D = SAMPLE_ROWS - 1
//   drafts tokens[1..D] against rowbuf rows 0..D: accept x_i with probability min(1, p_i(x_i) / q_i(x_i)); the
//   first rejection draws from normalize(max(0, p_i - q_i)); full acceptance draws the bonus from p_D. q_i is the
//   stored draft distribution (qbuf page i) or, with SAMPLE_POINT_Q, the point mass on x_i (any deterministic
//   proposer). A stale q (its token is not the draft) emits a direct draw from p_i. History += emitted tokens.
// SAMPLE_MODE 3 (selector draw, one BRISC): DFlash2 sampled candidate walk for draft position SAMPLE_SEL_TASK: draw
//   x from q = softmax(scores / T) over the 16 selector candidates (SAMPLE_SEL_PROC 1: the request's full vLLM
//   processing of those scores), store q in qbuf page SAMPLE_SEL_TASK and the token as the next predecessor/path.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "device_sampling_math.hpp"

namespace {
using namespace dsm;
constexpr uint32_t R = SAMPLE_ROWS;
constexpr uint32_t K = SAMPLE_K;
constexpr uint32_t NW = SAMPLE_WORKERS;
constexpr uint32_t TILES = SAMPLE_TILES;
constexpr uint32_t SPLIT = SAMPLE_SPLIT_TILE;
constexpr uint32_t ID_OFFSET = SAMPLE_ID_OFFSET;
constexpr uint32_t VOCAB_TILES = SAMPLE_VOCAB_TILES;
constexpr uint32_t RB = ((R * 32 + 63) / 64) * 64;  // bytes of R face rows, DRAM-read aligned
// Bytes staged per vocabulary tile: two face-row blocks (TILE logits) or one 64-byte segment per row (ROW_MAJOR).
constexpr uint32_t TILE_BYTES = SAMPLE_LOGITS_RM ? R * 64 : 2 * RB;
// L1 offset of row r's 16 logits of half h (columns 16h..16h+15) inside one staged tile.
constexpr uint32_t half_offset(uint32_t r, uint32_t h) { return SAMPLE_LOGITS_RM ? r * 64 + h * 32 : h * RB + r * 32; }
constexpr uint32_t BATCH = SAMPLE_BATCH;
constexpr uint32_t ENTRY_PAGE = K * 8;                // (key, id) pairs per worker list
constexpr uint32_t ROW_PAGE = 1024;
// Row/q buffer words.
constexpr uint32_t W_COUNT = 0, W_FLAGS = 1, W_TOKEN = 2, W_IDS = 4, W_PROBS = 4 + 96;

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

// Absolute position of the token drawn for `row`: 0 params word 12; 1 pos[WORD] + ADD (single row);
// 2 pos[row] + ADD (decode: one input position per row).
inline uint32_t row_position(const Params& p, uint32_t pos_l1, uint32_t row) {
#if SAMPLE_POS_MODE == 0
    return p.w[12];
#elif SAMPLE_POS_MODE == 1
    return reinterpret_cast<volatile uint32_t*>(pos_l1)[SAMPLE_POS_WORD] + SAMPLE_POS_ADD;
#else
    return reinterpret_cast<volatile uint32_t*>(pos_l1)[row] + SAMPLE_POS_ADD;
#endif
}

// Per-slot history tensors: common runtime args 10.. hold their base addresses.
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

// Mode-specific extra buffer addresses follow the history addresses in the common runtime args.
inline uint32_t extra_arg(uint32_t j) { return get_common_arg_val<uint32_t>(10 + SAMPLE_NSLOTS + j); }
}  // namespace
#endif

void kernel_main() {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#if SAMPLE_MODE == 0
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t base = get_write_ptr(0) + (is_ncrisc ? SAMPLE_SCRATCH_HALF : 0);
    const auto logits = TensorAccessor(tensor_accessor::make_interleaved_dspec<SAMPLE_LOGITS_DRAM>(), get_common_arg_val<uint32_t>(0),
                                       SAMPLE_LOGITS_RM ? TILES * 64 : 2048);
    const auto params = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 128);
    const auto partial = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3), ENTRY_PAGE);
    const auto aux = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(7), 128);
    const auto heads = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), NW * 64);
    // Scratch: params R*128 | aux 128 | heap keys R*K*4 | heap ids R*K*4 | logits BATCH*TILE_BYTES | hist BATCH*R*128
    const uint32_t p_l1 = base;
    const uint32_t aux_l1 = p_l1 + R * 128;
    const uint32_t keys_l1 = aux_l1 + 128;
    const uint32_t ids_l1 = keys_l1 + R * K * 4;
    const uint32_t tile_l1 = ids_l1 + R * K * 4;
    const uint32_t hist_l1 = tile_l1 + BATCH * TILE_BYTES;
    for (uint32_t r = 0; r < R; ++r) read_page(params, SAMPLE_PARAM_BASE + r, p_l1 + r * 128, 128);
    noc_async_read_barrier();
    barrier_compiler();
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(p_l1);
    // Speculative extra tokens: row r also counts aux words [b_r, b_r + n_r) as outputs, b_r = SAMPLE_EXTRA_WORD +
    // params word 21 (0 except in shared rounds, where a group's row j counts the group's drafts 1..j) and n_r = params
    // word 20 (e.g. verify row r counts the drafts x_1..x_r, draft step i the chain's earlier drafts).
    bool any_extra = false;
    for (uint32_t r = 0; r < R; ++r) any_extra |= (pw[r * 32] & F_EXTRA) != 0 && pw[r * 32 + 20] != 0;
    if (any_extra) {
        read_page(aux, 0, aux_l1, 128);
        noc_async_read_barrier();
        barrier_compiler();
    }
    const uint32_t* aux_words = reinterpret_cast<const uint32_t*>(aux_l1);
    bool any_pen = false;
    for (uint32_t r = 0; r < R; ++r) {
        any_pen |= (pw[r * 32] & (F_ACTIVE | F_PENALTY)) == (F_ACTIVE | F_PENALTY);
    }
    // Per row: a bounded min-heap (worst entry at the root) of the best keff entries by (key desc, id asc). For the
    // single-request cases (R <= 2) it lives in the RISC's single-cycle local memory (the stack), else in L1.
#if SAMPLE_ROWS <= 2
    uint32_t hk[R * K], hi[R * K];
#else
    uint32_t* hk = reinterpret_cast<uint32_t*>(keys_l1);
    uint32_t* hi = reinterpret_cast<uint32_t*>(ids_l1);
#endif
#if SAMPLE_PROFILE
    uint32_t t_begin = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L), t_dma = 0, t_cmp = 0;
#endif
    for (uint32_t first = worker; first < TILES; first += NW * BATCH) {
#if SAMPLE_PROFILE
        uint32_t ta = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
#endif
        uint32_t count = 0;
        for (uint32_t t = first; t < TILES && count < BATCH; t += NW, ++count) {
            uint32_t dst = tile_l1 + count * TILE_BYTES;
#if SAMPLE_LOGITS_RM
            for (uint32_t r = 0; r < R; ++r) {
                if (pw[r * 32] & F_ACTIVE) read_page(logits, r, dst + r * 64, 64, t * 64);
            }
#else
            read_page(logits, t, dst, RB, 0);
            read_page(logits, t, dst + RB, RB, 512);
#endif
            if (any_pen) {
                uint32_t vt = t < SPLIT ? t : t + ID_OFFSET / 32;
                for (uint32_t r = 0; r < R; ++r) {
                    if ((pw[r * 32] & (F_ACTIVE | F_PENALTY)) == (F_ACTIVE | F_PENALTY)) {
                        read_page(hist_acc(pw[r * 32 + 11]), vt, hist_l1 + (count * R + r) * 128, 128);
                    }
                }
            }
        }
        noc_async_read_barrier();
        barrier_compiler();
#if SAMPLE_PROFILE
        uint32_t tb = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        t_dma += tb - ta;
#endif
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t* p = pw + r * 32;
            const uint32_t fl = p[0];
            if (!(fl & F_ACTIVE)) continue;
            const uint32_t keff = p[14];
            uint32_t* rk = hk + r * K;
            uint32_t* ri = hi + r * K;
            const bool pen = (fl & F_PENALTY) != 0;
            // Monotone penalties (presence, frequency >= 0, repetition >= 1) never raise a logit: a raw value at or
            // below the threshold needs no history lookup.
            const bool mono = (fl & F_MONO) != 0;
            const bool rep_on = (fl & F_REP) != 0;
            static_assert(SAMPLE_EXTRA_WORD < 32, "extra tokens live in the 32-word aux page");
            const uint32_t extra_base = SAMPLE_EXTRA_WORD + p[21] < 32 ? SAMPLE_EXTRA_WORD + p[21] : 32;
            const uint32_t* extras = aux_words + extra_base;
            const uint32_t n_extra = (fl & F_EXTRA) ? (p[20] < 32 - extra_base ? p[20] : 32 - extra_base) : 0;
            const float rep = bitsf(p[7]), rrep = bitsf(p[8]), freq = bitsf(p[6]), pres = bitsf(p[5]);
            uint32_t size = p[15] ? p[16] : 0;  // persisted across batches in words 15/16/17 (scratch copy)
            uint32_t th = size < keff ? 0u : rk[0];
            // With a non-negative threshold and penalties that can only lower logits, a bf16 pair is rejected by two
            // signed compares of its raw bits: negative values are negative integers, positive ones order as integers.
            const bool fast_ok = !pen || mono;
            uint32_t c_t = 0;
            for (uint32_t t = first; t < TILES && c_t < BATCH; t += NW, ++c_t) {
                const uint32_t id0 = t < SPLIT ? t * 32 : t * 32 + ID_OFFSET;
                const uint32_t* h = reinterpret_cast<const uint32_t*>(hist_l1 + (c_t * R + r) * 128);
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t* f = reinterpret_cast<const uint32_t*>(tile_l1 + c_t * TILE_BYTES + half_offset(r, half));
                    for (uint32_t j = 0; j < 8; ++j) {
                        const uint32_t word = f[j];
                        if (fast_ok && th >= 0x80000000u) {
                            const int32_t tb = (int32_t)(th & 0x7fffffffu);
                            if ((int32_t)(word << 16) <= tb && (int32_t)(word & 0xffff0000u) <= tb) continue;
                        }
                        for (uint32_t b = 0; b < 2; ++b) {
                            const uint32_t c = 16 * half + 2 * j + b;
                            uint32_t fb = b ? (word & 0xffff0000u) : (word << 16);
                            uint32_t key = fkey(fb);
                            if (key <= th && fast_ok) continue;
                            const uint32_t id = id0 + c;
                            if (pen) {
                                const uint32_t e = h[c];
                                uint32_t cnt = e & 0x7fffffffu;
                                for (uint32_t x = 0; x < n_extra; ++x) cnt += extras[x] == id ? 1u : 0u;
                                const bool in_prompt = (e >> 31) != 0;
                                if (cnt || (rep_on && in_prompt)) {
                                    fb = penalize(fb, cnt, in_prompt, rep_on, rep, rrep, freq, pres);
                                    key = fkey(fb);
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
                }
            }
            // Persist the heap size for the next batch (params scratch copy, words 15/16).
            const_cast<uint32_t*>(p)[15] = 1;
            const_cast<uint32_t*>(p)[16] = size;
        }
#if SAMPLE_PROFILE
        t_cmp += reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - tb;
#endif
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
#if SAMPLE_PROFILE
        stage[2 * K - 4] = t_dma;
        stage[2 * K - 3] = t_cmp;
        stage[2 * K - 2] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - t_begin;
        stage[2 * K - 1] = t_begin;
#endif
        noc_async_write(tile_l1, partial.get_noc_addr(r * NW + worker), ENTRY_PAGE);
        // The first 8 entries also go to the row's contiguous head page (one read for the merge).
        noc_async_write(tile_l1, heads.get_noc_addr(r, worker * 64), 64);
        noc_async_write_barrier();
    }
#elif SAMPLE_MODE == 1
#if defined(COMPILE_FOR_BRISC)
    const uint32_t row = get_arg_val<uint32_t>(0);
    const uint32_t base = get_write_ptr(0);
    const auto params = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 128);
    const auto partial = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3), ENTRY_PAGE);
    const auto rowbuf = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4), ROW_PAGE);
    const auto qbuf = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(5), ROW_PAGE);
    const auto out = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6), SAMPLE_OUT_PAGE);
    const auto pos = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(8), 64);
    const auto heads = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), NW * 64);
    // Scratch: params 128 | pos 64 | out 64 | hist 128 | chunk 64 | heads NW*64 | result ROW_PAGE | sampler scratch
    const uint32_t p_l1 = base;
    const uint32_t pos_l1 = p_l1 + 128;
    const uint32_t out_l1 = pos_l1 + 64;
    const uint32_t h_l1 = out_l1 + 64;
    const uint32_t chunk_l1 = h_l1 + 128;
    const uint32_t heads_l1 = chunk_l1 + 64;
    const uint32_t res_l1 = ((heads_l1 + NW * 64 + 63) / 64) * 64;
    auto mkeys = reinterpret_cast<uint32_t*>(res_l1 + ROW_PAGE);
    auto mids = mkeys + K;
    auto weights = reinterpret_cast<float*>(mids + K);
    auto xs = weights + K;
    read_page(params, SAMPLE_PARAM_BASE + row, p_l1, 128);
#if SAMPLE_POS_MODE != 0
    read_page(pos, 0, pos_l1, 64);
#endif
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
    const uint32_t keff = prm.w[14];
#if SAMPLE_PROFILE
    const uint32_t m0 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t m1 = m0;
#endif
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
#if SAMPLE_PROFILE
    const uint32_t m2 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
#endif
    double total;
    bool inexact;
    const RowParams rp = row_params(prm);
    uint32_t n = process(mkeys, keff, rp, xs, weights, &total, &inexact);
    uint32_t rflags = inexact ? R_INEXACT : 0;
#if SAMPLE_PROFILE
    const uint32_t m3 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    // Profile: report (reads, merge, process) wall-clock deltas in the result flags' upper bits area.
    res[200] = m1 - m0;
    res[201] = m2 - m1;
    res[202] = m3 - m2;
#endif
#if SAMPLE_ROW_OP == 0
    uint32_t token = mids[0];
    if (!(rp.flags & F_GREEDY)) {
        double u = uniform(prm.w[9], prm.w[10], row_position(prm, pos_l1, row), P_SAMPLE);
        token = mids[draw(weights, n, total, u)];
    }
    o[0] = token;
    o[1] = rflags;
    o[2] = n;
    o[3] = 0;
    noc_async_write(out_l1, out.get_noc_addr(row), 16);
    noc_async_write_barrier();
    if (!inexact && !(prm.w[0] & F_NOCOMMIT)) commit(prm.w[11], token, h_l1);
#else
    uint32_t token = mids[0];
#if SAMPLE_ROW_OP == 1
#if SAMPLE_POINT_DRAFT
    n = 1;
    weights[0] = bitsf(0x3f800000u);
    total = bitsd(0x3ff0000000000000ull);
#else
    if (!(rp.flags & F_GREEDY)) {
        double u = uniform(prm.w[9], prm.w[10], row_position(prm, pos_l1, 0), P_DRAFT);
        token = mids[draw(weights, n, total, u)];
    }
#endif
#endif
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
#if SAMPLE_ROW_OP == 1
    noc_async_write(res_l1, qbuf.get_noc_addr(SAMPLE_Q_PAGE), ROW_PAGE);
    // The drafted token goes to word SAMPLE_OUT_WORD of the 128-byte out page (read-modify-write).
    read_page(out, 0, h_l1, 128);
    noc_async_read_barrier();
    reinterpret_cast<volatile uint32_t*>(h_l1)[SAMPLE_OUT_WORD] = token;
    noc_async_write(h_l1, out.get_noc_addr(0), 128);
#else
    noc_async_write(res_l1, rowbuf.get_noc_addr(row), ROW_PAGE);
#endif
    noc_async_write_barrier();
#endif
#endif
#elif SAMPLE_MODE == 2
#if defined(COMPILE_FOR_BRISC)
    // Chain verify. Common args: params, rowbuf, qbuf, out (cyc [1,32] int32), aux = tokens [1,R] (anchor, x_1..x_D),
    // pos = positions (word 0: anchor position P). With SAMPLE_VA also NativeVerifyAccept's outputs (extra args 0..5:
    // head [1,2R-1], accepted, next_token, advance, next_position, start) and positions advanced in place.
    constexpr uint32_t D = R - 1;
    const uint32_t base = get_write_ptr(0);
    const auto params = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 128);
    const auto rowbuf = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4), ROW_PAGE);
    const auto qbuf = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(5), ROW_PAGE);
    const auto out = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6), 128);
    const auto aux = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(7), 64);
    const auto pos = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(8), 64);
    // Scratch: params 128 | pos 64 | aux 64 | out 128 | hist 128 | VA 512 | residual K*8 | rows R*1024 | q D*1024
    const uint32_t p_l1 = base, pos_l1 = base + 128, aux_l1 = base + 192, out_l1 = base + 256, h_l1 = base + 384;
    const uint32_t va_l1 = base + 512;
    const uint32_t res_l1 = base + 1024;
    const uint32_t rows_l1 = res_l1 + ((K * 8 + 1023) / 1024) * 1024;
    const uint32_t q_l1 = rows_l1 + R * ROW_PAGE;
    read_page(params, 0, p_l1, 128);
    read_page(pos, 0, pos_l1, 64);
    read_page(aux, 0, aux_l1, 64);
    for (uint32_t r = 0; r < R; ++r) read_page(rowbuf, r, rows_l1 + r * ROW_PAGE, ROW_PAGE);
#if !SAMPLE_POINT_Q
    for (uint32_t i = 0; i < D; ++i) read_page(qbuf, i, q_l1 + i * ROW_PAGE, ROW_PAGE);
#endif
    noc_async_read_barrier();
    barrier_compiler();
    auto prm = reinterpret_cast<volatile uint32_t*>(p_l1);
    auto tokens = reinterpret_cast<volatile uint32_t*>(aux_l1);
    auto row = [&](uint32_t r) { return reinterpret_cast<volatile uint32_t*>(rows_l1 + r * ROW_PAGE); };
    auto qrow = [&](uint32_t i) { return reinterpret_cast<volatile uint32_t*>(q_l1 + i * ROW_PAGE); };
    auto prob = [](volatile uint32_t* b, uint32_t i) -> double {
        return bitsd((uint64_t)b[W_PROBS + 2 * i] | ((uint64_t)b[W_PROBS + 2 * i + 1] << 32));
    };
    auto count = [](volatile uint32_t* b) -> uint32_t { return b[W_COUNT] < K ? b[W_COUNT] : K; };
    auto lookup = [&](volatile uint32_t* b, uint32_t token) -> double {
        uint32_t n = count(b);
        for (uint32_t i = 0; i < n; ++i) {
            if (b[W_IDS + i] == token) return prob(b, i);
        }
        return bitsd(0);
    };
    const uint32_t seed_lo = prm[9], seed_hi = prm[10], slot = prm[11];
    const bool active = (prm[0] & F_ACTIVE) != 0;
    const uint32_t P = reinterpret_cast<volatile uint32_t*>(pos_l1)[0];
    auto sample_list = [&](volatile uint32_t* b, uint32_t position, uint32_t purpose) -> uint32_t {
        uint32_t n = count(b);
        double target = uniform(seed_lo, seed_hi, position, purpose);
        double cum = bitsd(0);
        for (uint32_t i = 0; i + 1 < n; ++i) {
            cum = d_add(cum, prob(b, i));
            if (d_gt(cum, target)) return b[W_IDS + i];
        }
        return b[W_IDS + n - 1];
    };
    uint32_t emitted[16];
    for (uint32_t i = 0; i < 16; ++i) emitted[i] = 0;
    uint32_t acc = 0, vflags = 0;
    bool decided = false;
    if (active) {
        for (uint32_t i = 0; i < D && !decided; ++i) {
            volatile uint32_t* p = row(i);
            const uint32_t x = tokens[i + 1];
            const uint32_t position = P + i + 1;
            if (p[W_FLAGS] & R_INEXACT) {
                // The host redraws this position exactly from the retained logits.
                emitted[i] = p[W_IDS];
                vflags = 1 + i;
                decided = true;
                break;
            }
#if !SAMPLE_POINT_Q
            volatile uint32_t* q = qrow(i);
            if (q[W_TOKEN] != x) {
                // The stored q does not belong to this draft: discard the draft, draw directly from p_i (exact).
                emitted[i] = sample_list(p, position, P_SAMPLE);
                decided = true;
                break;
            }
            const double qx = lookup(q, x);
#else
            const double qx = bitsd(0x3ff0000000000000ull);
#endif
            const double px = lookup(p, x);
            const double u = uniform(seed_lo, seed_hi, position, P_ACCEPT);
            if (d_lt(d_mul(u, qx), px)) {
                emitted[i] = x;
                ++acc;
                continue;
            }
            // Residual normalize(max(0, p_i - q_i)) over the support of p_i.
            const uint32_t n = count(p);
            auto res = reinterpret_cast<double*>(res_l1);
            double sum = bitsd(0);
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t id = p[W_IDS + k];
#if SAMPLE_POINT_Q
                const double qk = id == x ? bitsd(0x3ff0000000000000ull) : bitsd(0);
#else
                const double qk = lookup(q, id);
#endif
                const double d = d_sub(prob(p, k), qk);
                res[k] = d_gt(d, bitsd(0)) ? d : bitsd(0);
                sum = d_add(sum, res[k]);
            }
            if (!d_gt(sum, bitsd(0))) {
                for (uint32_t k = 0; k < n; ++k) res[k] = prob(p, k);
                sum = bitsd(0x3ff0000000000000ull);
            }
            const double target = d_mul(uniform(seed_lo, seed_hi, position, P_RESIDUAL), sum);
            double cum = bitsd(0);
            uint32_t t = p[W_IDS + n - 1];
            for (uint32_t k = 0; k < n; ++k) {
                if (!d_gt(res[k], bitsd(0))) continue;
                cum = d_add(cum, res[k]);
                t = p[W_IDS + k];
                if (d_gt(cum, target)) break;
            }
            emitted[i] = t;
            decided = true;
        }
        if (!decided) {
            // Every draft accepted: the bonus token from p_D.
            volatile uint32_t* p = row(D);
            if (p[W_FLAGS] & R_INEXACT) {
                emitted[D] = p[W_IDS];
                vflags = 1 + D;
            } else {
                emitted[D] = sample_list(p, P + D + 1, P_SAMPLE);
            }
        }
    }
    auto o = reinterpret_cast<volatile uint32_t*>(out_l1);
    for (uint32_t i = 0; i < 32; ++i) o[i] = 0;
    for (uint32_t i = 0; i < R; ++i) o[i] = emitted[i];
    o[16] = acc;
    o[17] = vflags;
    o[18] = P;
    o[19] = active ? 1 : 0;
    noc_async_write(out_l1, out.get_noc_addr(0), 128);
#if SAMPLE_VA
    {
        constexpr uint32_t head_bytes = ((2 * R - 1) * 4 + 31) / 32 * 32;
        constexpr uint32_t pos_bytes = (R * 4 + 31) / 32 * 32;
        const auto head = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), extra_arg(0), head_bytes);
        auto h = reinterpret_cast<volatile int32_t*>(va_l1);
        for (uint32_t i = 0; i < 32; ++i) h[i] = 0;
        for (uint32_t i = 0; i < R; ++i) h[i] = (int32_t)emitted[i];
        for (uint32_t i = 0; i < D; ++i) h[R + i] = i < acc ? 1 : 0;
        noc_async_write(va_l1, head.get_noc_addr(0), head_bytes);
        auto sc = reinterpret_cast<volatile int32_t*>(va_l1 + 128);
        const int32_t values[5] = {(int32_t)acc, (int32_t)emitted[acc], (int32_t)acc + 1, (int32_t)(P + acc + 1), (int32_t)P};
        for (uint32_t k = 0; k < 5; ++k) {
            for (uint32_t i = 0; i < 8; ++i) sc[8 * k + i] = i ? 0 : values[k];
            const auto scalar = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), extra_arg(1 + k), 32);
            noc_async_write(va_l1 + 128 + 32 * k, scalar.get_noc_addr(0), 32);
        }
        auto np = reinterpret_cast<volatile int32_t*>(va_l1 + 320);
        for (uint32_t i = 0; i < 8; ++i) np[i] = 0;
        for (uint32_t r = 0; r < R; ++r) np[r] = (int32_t)(P + acc + 1 + r);
        noc_async_write(va_l1 + 320, pos.get_noc_addr(0), pos_bytes);
    }
#endif
    noc_async_write_barrier();
    if (active && !(prm[0] & F_NOCOMMIT)) {
        const uint32_t last = vflags ? acc : acc + 1;  // a host-redrawn token is committed by the host
        for (uint32_t i = 0; i < last; ++i) commit(slot, emitted[i], h_l1);
    }
#endif
#elif SAMPLE_MODE == 3
#if defined(COMPILE_FOR_BRISC)
    // DFlash2 sampled selector step SAMPLE_SEL_TASK. Extra args: scores (BF16 tile, row 0 = the 16 candidate scores),
    // candidates (uint32 [1,16]), next (uint32 [1,1]), path (uint32 [1,rows]). pos word 0 = anchor position.
    // SAMPLE_SEL_LANE (shared rounds' batched drafter, dflash_slots.py): runtime args 1..3 give the path row (TASK),
    // the parameter row of the lane's request and the draft's offset from the anchor position; the draw uses purpose
    // SAMPLE_SEL_PURPOSE and q is not stored (shared rounds verify by matching the target's own draw).
    constexpr uint32_t NC = 16;
#if SAMPLE_SEL_LANE
    const uint32_t TASK = get_arg_val<uint32_t>(1);
    const uint32_t param_row = get_arg_val<uint32_t>(2);
    const uint32_t pos_add = get_arg_val<uint32_t>(3);
#else
    constexpr uint32_t TASK = SAMPLE_SEL_TASK;
    constexpr uint32_t param_row = 0;
    constexpr uint32_t pos_add = 1 + TASK;
#endif
    constexpr uint32_t path_bytes = ((SAMPLE_SEL_ROWS * 4 + 31) / 32) * 32;
    const uint32_t base = get_write_ptr(0);
    const auto params = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 128);
    [[maybe_unused]] const auto qbuf = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(5), ROW_PAGE);
    const auto pos = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(8), 64);
    const auto scores = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), extra_arg(0), 2048);
    const auto candidates = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), extra_arg(1), 64);
    const auto next = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), extra_arg(2), 32);
    const auto path = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), extra_arg(3), path_bytes);
    // Scratch: scores 2048 | params 128 | pos 64 | ids 64 | path 128 | next 64 | q ROW_PAGE
    const uint32_t s_l1 = base, p_l1 = base + 2048, pos_l1 = base + 2176, ids_l1 = base + 2240, path_l1 = base + 2304;
    const uint32_t next_l1 = base + 2432;
    [[maybe_unused]] const uint32_t q_l1 = base + 2560;
    read_page(scores, 0, s_l1, 2048);
    read_page(candidates, 0, ids_l1, 64);
    read_page(params, param_row, p_l1, 128);
    read_page(pos, 0, pos_l1, 64);
    if (TASK) read_page(path, 0, path_l1, path_bytes);
    noc_async_read_barrier();
    barrier_compiler();
    Params prm;
    for (uint32_t i = 0; i < 32; ++i) prm.w[i] = reinterpret_cast<const uint32_t*>(p_l1)[i];
    auto sv = reinterpret_cast<volatile uint16_t*>(s_l1);
    auto cid = reinterpret_cast<volatile uint32_t*>(ids_l1);
    // Candidates by (score desc, id asc): the original selector's greedy order (lower id wins ties).
    uint32_t keys[NC], ids[NC];
    for (uint32_t i = 0; i < NC; ++i) {
        uint32_t key = fkey((uint32_t)sv[i] << 16), id = cid[i];
        uint32_t j = i;
        while (j && (keys[j - 1] < key || (keys[j - 1] == key && ids[j - 1] > id))) {
            keys[j] = keys[j - 1];
            ids[j] = ids[j - 1];
            --j;
        }
        keys[j] = key;
        ids[j] = id;
    }
    float weights[NC];
    double total = bitsd(0x3ff0000000000000ull);
    uint32_t n = 1;
    weights[0] = bitsf(0x3f800000u);
    uint32_t token = ids[0];
    const RowParams rp = row_params(prm);
    if ((prm.w[0] & F_ACTIVE) && !(rp.flags & F_GREEDY)) {
#if SAMPLE_SEL_PROC
        float xs[NC];
        bool inexact;
        RowParams qp = rp;
        if (qp.top_k > NC) qp.top_k = NC;
        n = process(keys, NC, qp, xs, weights, &total, &inexact);
#else
        const float l0 = bitsf(keyf(keys[0]));
        total = bitsd(0);
        n = NC;
        for (uint32_t i = 0; i < NC; ++i) {
            weights[i] = exp_weight(s_sub(l0, bitsf(keyf(keys[i]))), rp.c_lo, rp.c_hi, rp.temperature);
            total = d_add(total, d_f2d(weights[i]));
        }
#endif
        const uint32_t position = reinterpret_cast<volatile uint32_t*>(pos_l1)[0] + pos_add;
        token = ids[draw(weights, n, total, uniform(prm.w[9], prm.w[10], position, SAMPLE_SEL_PURPOSE))];
    }
#if !SAMPLE_SEL_LANE
    auto q = reinterpret_cast<volatile uint32_t*>(q_l1);
    q[W_COUNT] = n;
    q[W_FLAGS] = 0;
    q[W_TOKEN] = token;
    q[3] = 0;
    for (uint32_t i = 0; i < n; ++i) {
        q[W_IDS + i] = ids[i];
        uint64_t b = dbits(d_div(d_f2d(weights[i]), total));
        q[W_PROBS + 2 * i] = (uint32_t)b;
        q[W_PROBS + 2 * i + 1] = (uint32_t)(b >> 32);
    }
    noc_async_write(q_l1, qbuf.get_noc_addr(TASK), ROW_PAGE);
#endif
    auto pa = reinterpret_cast<volatile uint32_t*>(path_l1);
    if (!TASK) for (uint32_t i = 0; i < path_bytes / 4; ++i) pa[i] = 0;
    pa[TASK] = token;
    auto nx = reinterpret_cast<volatile uint32_t*>(next_l1);
    for (uint32_t i = 0; i < 8; ++i) nx[i] = 0;
    nx[0] = token;
    noc_async_write(path_l1, path.get_noc_addr(0), path_bytes);
    noc_async_write(next_l1, next.get_noc_addr(0), 32);
    noc_async_write_barrier();
#endif
#endif
#endif
}
