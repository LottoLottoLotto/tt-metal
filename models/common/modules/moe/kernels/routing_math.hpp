// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
// BF16 -> signed Q16 fixed point; |x| is clamped below 2^14 (router logits are far smaller).
inline int32_t bf16_to_q16(uint16_t bits) {
    const int32_t exponent = (bits >> 7) & 0xff;
    if (exponent == 0) {
        return 0;
    }
    const int32_t mantissa = 0x80 | (bits & 0x7f);  // value = mantissa * 2^(exponent - 134)
    const int32_t shift = exponent - 118;           // Q16: mantissa * 2^(exponent - 118)
    int32_t magnitude;
    if (shift >= 0) {
        magnitude = mantissa << (shift > 22 ? 22 : shift);
    } else {
        magnitude = shift <= -8 ? 0 : mantissa >> -shift;
    }
    return (bits & 0x8000) ? -magnitude : magnitude;
}

// exp(-d) in Q31 for d >= 0 given in Q16.
inline uint32_t exp_neg_q31(uint32_t d) {
    // 2^(-i/32) in Q31, i = 0..32.
    static constexpr uint32_t pow2_neg[33] = {
        2147483648u, 2101467502u, 2056437387u, 2012372174u, 1969251188u, 1927054196u, 1885761398u,
        1845353420u, 1805811301u, 1767116489u, 1729250827u, 1692196547u, 1655936265u, 1620452965u,
        1585730000u, 1551751076u, 1518500250u, 1485961921u, 1454120821u, 1422962010u, 1392470869u,
        1362633090u, 1333434672u, 1304861917u, 1276901417u, 1249540052u, 1222764986u, 1196563654u,
        1170923762u, 1145833280u, 1121280436u, 1097253708u, 1073741824u};
    // t = d * log2(e) in Q16 (log2 e in Q30).
    const uint64_t t = (static_cast<uint64_t>(d) * 1549082005u) >> 30;
    const uint64_t whole = t >> 16;
    if (whole >= 31) {
        return 0;
    }
    const uint32_t frac = static_cast<uint32_t>(t) & 0xffff;
    // 2^-(i/32 + r/65536) = table[i] * e^(-a), a = r * ln 2 / 65536 < 0.0217 (Q31).
    const uint32_t a = static_cast<uint32_t>((static_cast<uint64_t>(frac & 0x7ff) * 1488522236u) >> 16);
    const uint32_t a2_half = static_cast<uint32_t>((static_cast<uint64_t>(a) * a) >> 32);
    const uint32_t poly = 0x80000000u - a + a2_half;
    const uint32_t value = static_cast<uint32_t>((static_cast<uint64_t>(pow2_neg[frac >> 11]) * poly) >> 31);
    return value >> whole;
}

// Unsigned Q62 fixed point (value < 2) -> BF16, round to nearest even.
inline uint16_t q62_to_bf16(uint64_t q) {
    if (q == 0) {
        return 0;
    }
    uint32_t top = 63;
    while (!((q >> top) & 1u)) {
        --top;
    }
    uint32_t exponent = top + 127 - 62;
    uint32_t mantissa;
    if (top >= 8) {
        const uint32_t drop = top - 7;
        mantissa = static_cast<uint32_t>(q >> drop);  // 8 bits incl. the leading one
        const uint64_t rest = q & ((uint64_t{1} << drop) - 1);
        const uint64_t half = uint64_t{1} << (drop - 1);
        if (rest > half || (rest == half && (mantissa & 1u))) {
            if (++mantissa == 0x100) {
                mantissa = 0x80;
                ++exponent;
            }
        }
    } else {
        mantissa = static_cast<uint32_t>(q << (7 - top));
    }
    return static_cast<uint16_t>((exponent << 7) | (mantissa & 0x7f));
}

// Monotonic unsigned key for BF16 ordering (negative values reversed).
inline uint32_t order_key(uint32_t bits) { return (bits & 0x8000u) ? (~bits & 0xffffu) : (bits | 0x8000u); }

// Running top_k of (order key, expert id) candidates, descending by key; ties keep the
// lower expert id first when candidates are offered in ascending id order.
template <uint32_t top_k>
struct TopK {
    uint32_t best_key[top_k];
    uint32_t best_id[top_k];
    uint32_t count = 0;
    uint32_t threshold = 0;  // best_key[top_k - 1] once the list is full

    inline void consider(uint32_t key, uint32_t e) {
        if (count == top_k && key <= threshold) {
            return;
        }
        uint32_t pos = count < top_k ? count++ : top_k - 1;
        while (pos > 0 && best_key[pos - 1] < key) {
            best_key[pos] = best_key[pos - 1];
            best_id[pos] = best_id[pos - 1];
            --pos;
        }
        best_key[pos] = key;
        best_id[pos] = e;
        threshold = best_key[top_k - 1];
    }
};

// Routing tail for one token given its descending top_k ids (best_id[0] = max logit):
// top_k ids ascending with their softmax weights, plus the shared expert's sigmoid gate
// weight (logit at index num_logits) when present. Reads the logits by id from logits_row.
template <uint32_t num_logits, uint32_t top_k, bool has_shared>
void route_finish(
    uint32_t logits_row, const uint32_t* best_id, uint32_t* ids, uint16_t* weights, uint16_t& shared_weight) {
    auto* values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(logits_row);
    // Softmax over the selected logits: w_k = exp(x_k - max) in Q31, normalized with one
    // 64-bit reciprocal (total in [2^31, 2^36)) to Q62.
    const int32_t max_logit = bf16_to_q16(values[best_id[0]]);
    uint32_t raw[top_k];
    uint64_t total = 0;
    for (uint32_t k = 0; k < top_k; ++k) {
        raw[k] = exp_neg_q31(static_cast<uint32_t>(max_logit - bf16_to_q16(values[best_id[k]])));
        total += raw[k];
    }
    const uint64_t reciprocal = (uint64_t{1} << 62) / total;  // Q62 / Q31 -> Q31

    // Ascending ids (insertion sort, top_k <= 32), carrying each id's weight.
    for (uint32_t k = 0; k < top_k; ++k) {
        const uint16_t weight = q62_to_bf16(static_cast<uint64_t>(raw[k]) * reciprocal);
        uint32_t pos = k;
        while (pos > 0 && ids[pos - 1] > best_id[k]) {
            ids[pos] = ids[pos - 1];
            weights[pos] = weights[pos - 1];
            --pos;
        }
        ids[pos] = best_id[k];
        weights[pos] = weight;
    }
    if constexpr (has_shared) {
        // sigmoid(g) = 1 / (1 + e^-g): with e = exp(-|g|), g >= 0 -> 1/(1+e), else e/(1+e).
        const int32_t gate = bf16_to_q16(values[num_logits]);
        const uint32_t e = exp_neg_q31(static_cast<uint32_t>(gate < 0 ? -gate : gate));
        const uint64_t inverse = (uint64_t{1} << 62) / (uint64_t{0x80000000u} + e);
        const uint64_t numerator = gate >= 0 ? uint64_t{0x80000000u} : e;
        shared_weight = q62_to_bf16(numerator * inverse);
    }
}

// One token's routing: top_k ids ascending with their softmax weights, plus the shared
// expert's sigmoid gate weight when present.
template <uint32_t num_logits, uint32_t top_k, bool has_shared>
void route(uint32_t logits_row, uint32_t* ids, uint16_t* weights, uint16_t& shared_weight) {
    auto* pairs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(logits_row);
    // Two logits per 32-bit load (a 16-bit volatile load per logit measured 5.7 us).
    TopK<top_k> best;
    for (uint32_t e = 0; e < num_logits; e += 2) {
        const uint32_t pair = pairs[e / 2];
        best.consider(order_key(pair & 0xffffu), e);
        best.consider(order_key(pair >> 16), e + 1);
    }
    route_finish<num_logits, top_k, has_shared>(logits_row, best.best_id, ids, weights, shared_weight);
}
}  // namespace
