// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 decode MoE v2 routing on the router cores (data-movement RISCs, integer only).
//
// The selection is lfm_route's (sigmoid_math.hpp, flags 3: BF16-rounded logits, scores from the exact host table):
// the top_k experts by (key = Q24 score + Q24 bias) descending, equal keys -> lower id first; ids ascending; BF16
// weights score / (sum + 1e-6).  That order is a total order, so the top_k of a union of candidate lists, each the
// top_k of its own experts, is the global top_k: every router RISC scores 16 experts (`score_experts`), the RISCs
// merge their lists, the router cores swap their column lists, and `finish_route` forms the weights with the same
// integer arithmetic (moe_decode2_math.hpp: exact 32-bit-only division) -> bitwise lfm_route.
#pragma once
#include <cstdint>

#include "models/demos/blackhole/lfm2/tt/fast/kernels/sigmoid_math.hpp"
#include "models/demos/blackhole/lfm2/tt/fast/kernels/moe_decode2_math.hpp"

namespace moe2 {

// Candidate list as exchanged in L1 (one 64 B record): ids, keys, scores of the top_k, best first.
template <uint32_t top_k>
struct Cands {
    uint32_t id[top_k];
    int32_t key[top_k];
    uint32_t score[top_k];
    uint32_t count = 0;

    // Insert in (key desc, id asc) order, keeping the best top_k.
    void consider(int32_t k, uint32_t e, uint32_t s) {
        if (count == top_k && (k < key[top_k - 1] || (k == key[top_k - 1] && e > id[top_k - 1]))) {
            return;
        }
        uint32_t pos = count < top_k ? count++ : top_k - 1;
        while (pos > 0 && (key[pos - 1] < k || (key[pos - 1] == k && id[pos - 1] > e))) {
            key[pos] = key[pos - 1];
            id[pos] = id[pos - 1];
            score[pos] = score[pos - 1];
            --pos;
        }
        key[pos] = k;
        id[pos] = e;
        score[pos] = s;
    }

    void merge(const volatile tt_l1_ptr uint32_t* rec) {
        for (uint32_t i = 0; i < top_k; ++i) {
            consider(static_cast<int32_t>(rec[top_k + i]), rec[i], rec[2 * top_k + i]);
        }
    }

    void store(volatile tt_l1_ptr uint32_t* rec) const {
        for (uint32_t i = 0; i < top_k; ++i) {
            rec[i] = id[i];
            rec[top_k + i] = static_cast<uint32_t>(key[i]);
            rec[2 * top_k + i] = score[i];
        }
    }
};

// Scores of the n consecutive experts first.. from their FP32 logits (lfm_route's flags-3 path).  The caller has
// waited for the data and invalidated the L1 data cache: plain loads.
template <uint32_t top_k, uint32_t n, uint32_t e_lo, uint32_t e_hi>
inline void score_experts(Cands<top_k>& best, const uint32_t* logits, uint32_t first, const int32_t* bias,
                          const uint16_t* table) {
    constexpr uint32_t n_exp = e_hi - e_lo + 1;
    uint32_t s[n];
    int32_t key[n];
#pragma GCC unroll 4
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t b = f32_round_bf16(logits[i]) >> 16;
        const uint32_t mag = b & 0x7fffu, ex = mag >> 7;
        uint16_t s16;
        if (ex < e_lo) {
            s16 = 0x3f00;  // 0.5
        } else if (ex > e_hi) {
            s16 = (b & 0x8000u) ? 0 : 0x3f80;
        } else {
            s16 = table[((b >> 15) * n_exp + (ex - e_lo)) * 128 + (mag & 0x7fu)];
        }
        s[i] = bf16_to_q24(s16);
        key[i] = static_cast<int32_t>(s[i]) + bias[first + i];
    }
    for (uint32_t i = 0; i < n; ++i) {
        best.consider(key[i], first + i, s[i]);
    }
}

// ids ascending + BF16 weights score / (sum + 1e-6), exactly as lfm_route's tail.
template <uint32_t top_k>
inline void finish_route(const Cands<top_k>& best, uint32_t* ids, uint16_t* weights) {
    constexpr uint32_t eps_q24 = 17;
    uint32_t id[top_k], score[top_k];
    for (uint32_t k = 0; k < top_k; ++k) {
        uint32_t pos = k;
        while (pos > 0 && id[pos - 1] > best.id[k]) {
            id[pos] = id[pos - 1];
            score[pos] = score[pos - 1];
            --pos;
        }
        id[pos] = best.id[k];
        score[pos] = best.score[k];
    }
    uint32_t total = eps_q24;  // top_k <= 4 scores <= 2^24: < 2^27
    for (uint32_t k = 0; k < top_k; ++k) {
        total += score[k];
    }
    const uint64_t reciprocal = recip62(total);
    for (uint32_t k = 0; k < top_k; ++k) {
        ids[k] = id[k];
        weights[k] = q62_to_bf16_clz(static_cast<uint64_t>(score[k]) * reciprocal);
    }
}

}  // namespace moe2
