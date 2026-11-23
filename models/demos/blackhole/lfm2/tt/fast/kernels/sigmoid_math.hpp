// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 router math for the decode MoE (data-movement RISCs: integer only).
//
// The scores / selection / weights are bitwise those of tt/kernels/sigmoid_routing.cpp (the prefill and the current
// decode routing): the helpers below are copied from it, except that the per-expert 64-bit division
// (1 << 62) / (2^31 + e) of sigmoid_q62, a libgcc call that measured ~15 us for 64 experts, is replaced by an exact
// floor division from a Newton reciprocal plus a remainder correction (same result for every input).
#pragma once
#include <cstdint>

#include "models/common/modules/moe/kernels/routing_math.hpp"

namespace {
// |x| of an FP32 value in Q16, rounded to nearest; clamped below 2^14 (router logits are far smaller).
inline uint32_t f32_abs_q16(uint32_t bits) {
    const int32_t exponent = (bits >> 23) & 0xff;
    if (exponent == 0) {
        return 0;
    }
    const uint32_t mantissa = 0x800000u | (bits & 0x7fffffu);  // value = mantissa * 2^(exponent - 150)
    const int32_t shift = exponent - 134;                       // Q16: mantissa * 2^(exponent - 134)
    if (shift >= 0) {
        return shift > 6 ? 0x3fffffffu : mantissa << shift;
    }
    if (shift <= -25) {
        return 0;
    }
    return (mantissa + (1u << (-shift - 1))) >> -shift;
}

inline uint32_t f32_round_bf16(uint32_t bits) { return (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u; }

// BF16 bits (positive) -> Q24, exact for values >= 2^-17.
inline uint32_t bf16_to_q24(uint16_t bits) {
    const int32_t exponent = (bits >> 7) & 0xff;
    if (exponent == 0) {
        return 0;
    }
    const uint32_t mantissa = 0x80u | (bits & 0x7fu);  // value = mantissa * 2^(exponent - 134)
    const int32_t shift = exponent - 110;              // Q24
    return shift >= 0 ? mantissa << shift : (shift <= -8 ? 0 : mantissa >> -shift);
}

inline uint32_t mulhi(uint32_t a, uint32_t b) { return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32); }

// floor(2^62 / (2^31 + e)) for e in [0, 2^31] (Q31 exp), exactly.
inline uint64_t recip_q62(uint32_t e) {
    if (e >= 0x80000000u) {
        return uint64_t{1} << 30;  // d = 2^32
    }
    const uint32_t d = 0x80000000u + e;  // y = d / 2^32 in [0.5, 1)
    // r = 1 / y in Q30, r0 = 48/17 - 32/17 y (relative error <= 1/17), three Newton steps r <- r + r (1 - y r).
    uint32_t r = 3031741621u - mulhi(d, 2021161080u);
    for (int i = 0; i < 3; ++i) {
        const int32_t err = static_cast<int32_t>(0x40000000u - mulhi(d, r));  // 1 - y r in Q30
        r += static_cast<int32_t>((static_cast<int64_t>(r) * err) >> 30);
    }
    // r ~ 2^62 / d within a few units: correct to the exact floor.
    const uint64_t target = uint64_t{1} << 62;
    uint64_t q = r, prod = q * d;
    while (prod > target) {
        --q;
        prod -= d;
    }
    while (prod + d <= target) {
        ++q;
        prod += d;
    }
    return q;
}

// sigmoid(x) in Q62 for an FP32 logit (sigmoid_routing.cpp's sigmoid_q62, bitwise).
inline uint64_t sigmoid_q62(uint32_t bits) {
    const uint32_t e = exp_neg_q31(f32_abs_q16(bits));  // exp(-|x|) in Q31
    const uint64_t inverse = recip_q62(e);
    const uint64_t numerator = (bits & 0x80000000u) ? e : uint64_t{0x80000000u};
    return numerator * inverse;
}

// sigmoid(x) in Q24 to within 1024 (piecewise linear in |x| on a 1/16 grid, error <= 788): the cheap first pass.
inline uint32_t sigmoid_q24_approx(uint32_t bits) {
    static constexpr uint32_t table[257] = {
        8388608u, 8650667u, 8912214u, 9172744u, 9431757u, 9688764u, 9943296u, 10194896u,
        10443135u, 10687603u, 10927921u, 11163737u, 11394728u, 11620604u, 11841108u, 12056013u,
        12265128u, 12468290u, 12665372u, 12856274u, 13040928u, 13219292u, 13391352u, 13557118u,
        13716624u, 13869922u, 14017088u, 14158210u, 14293396u, 14422764u, 14546446u, 14664583u,
        14777323u, 14884823u, 14987245u, 15084754u, 15177517u, 15265706u, 15349491u, 15429041u,
        15504527u, 15576115u, 15643971u, 15708256u, 15769129u, 15826746u, 15881257u, 15932808u,
        15981542u, 16027595u, 16071101u, 16112187u, 16150975u, 16187583u, 16222125u, 16254709u,
        16285438u, 16314412u, 16341724u, 16367465u, 16391720u, 16414571u, 16436096u, 16456368u,
        16475457u, 16493431u, 16510351u, 16526278u, 16541267u, 16555374u, 16568648u, 16581136u,
        16592886u, 16603938u, 16614335u, 16624113u, 16633310u, 16641958u, 16650091u, 16657738u,
        16664929u, 16671689u, 16678045u, 16684020u, 16689637u, 16694917u, 16699880u, 16704545u,
        16708930u, 16713052u, 16716925u, 16720566u, 16723987u, 16727203u, 16730224u, 16733064u,
        16735732u, 16738240u, 16740596u, 16742810u, 16744891u, 16746846u, 16748683u, 16750409u,
        16752030u, 16753554u, 16754986u, 16756331u, 16757595u, 16758782u, 16759898u, 16760946u,
        16761931u, 16762856u, 16763726u, 16764542u, 16765310u, 16766031u, 16766708u, 16767344u,
        16767942u, 16768504u, 16769031u, 16769527u, 16769992u, 16770430u, 16770841u, 16771227u,
        16771590u, 16771931u, 16772251u, 16772551u, 16772834u, 16773099u, 16773349u, 16773583u,
        16773803u, 16774010u, 16774204u, 16774386u, 16774558u, 16774719u, 16774870u, 16775012u,
        16775146u, 16775271u, 16775389u, 16775500u, 16775604u, 16775701u, 16775793u, 16775879u,
        16775960u, 16776036u, 16776108u, 16776175u, 16776238u, 16776297u, 16776353u, 16776405u,
        16776454u, 16776500u, 16776544u, 16776585u, 16776623u, 16776659u, 16776693u, 16776724u,
        16776754u, 16776782u, 16776808u, 16776833u, 16776856u, 16776878u, 16776898u, 16776918u,
        16776936u, 16776953u, 16776969u, 16776984u, 16776998u, 16777011u, 16777023u, 16777035u,
        16777046u, 16777056u, 16777066u, 16777075u, 16777084u, 16777092u, 16777099u, 16777106u,
        16777113u, 16777119u, 16777125u, 16777131u, 16777136u, 16777141u, 16777145u, 16777149u,
        16777153u, 16777157u, 16777161u, 16777164u, 16777167u, 16777170u, 16777173u, 16777176u,
        16777178u, 16777180u, 16777183u, 16777185u, 16777186u, 16777188u, 16777190u, 16777192u,
        16777193u, 16777194u, 16777196u, 16777197u, 16777198u, 16777199u, 16777200u, 16777201u,
        16777202u, 16777203u, 16777204u, 16777204u, 16777205u, 16777206u, 16777206u, 16777207u,
        16777208u, 16777208u, 16777209u, 16777209u, 16777209u, 16777210u, 16777210u, 16777211u,
        16777211u, 16777211u, 16777211u, 16777212u, 16777212u, 16777212u, 16777212u, 16777213u,
        16777213u, 16777213u, 16777213u, 16777213u, 16777214u, 16777214u, 16777214u, 16777214u,
        16777214u,
    };
    const uint32_t a = f32_abs_q16(bits);
    uint32_t s = 1u << 24;
    if (a < (16u << 16)) {
        const uint32_t i = a >> 12, frac = a & 0xfffu;
        s = table[i] + (((table[i + 1] - table[i]) * frac) >> 12);
    }
    return (bits & 0x80000000u) ? (1u << 24) - s : s;
}

// Descending top_k insertion; equal keys keep the earlier (lower) id first.
template <uint32_t top_k>
struct TopK {
    uint32_t id[top_k];
    int32_t key[top_k];
    uint32_t count = 0;
    void consider(int32_t k, uint32_t e) {
        if (count == top_k && k <= key[top_k - 1]) {
            return;
        }
        uint32_t pos = count < top_k ? count++ : top_k - 1;
        while (pos > 0 && key[pos - 1] < k) {
            key[pos] = key[pos - 1];
            id[pos] = id[pos - 1];
            --pos;
        }
        key[pos] = k;
        id[pos] = e;
    }
};

// One token's routing.  logits: FP32 bits in expert order; bias: Q24.  ids ascending, BF16 weights
// score / (sum + 1e-6): bitwise sigmoid_routing.cpp.
// flags 3 (BF16 logits, BF16 scores: the default) -> a score depends only on the logit's BF16 bits and is read from
// `table` (sigmoid_table.py: the same integer arithmetic tabulated on the host for BF16 exponents [e_lo, e_hi];
// constant 0.5 below, 1 / 0 above): one cheap pass.
// Other flags -> two passes: approximate keys (|error| <= delta) select the candidates within 2 delta of the
// approximate top_k-th key, which provably contain the exact top_k; only those get the exact (expensive) score, in
// ascending id order, so ties resolve exactly as in the one-pass original.
template <uint32_t num_experts, uint32_t top_k, uint32_t flags, uint32_t e_lo, uint32_t e_hi>
inline void lfm_route(const volatile tt_l1_ptr uint32_t* logits, const volatile tt_l1_ptr int32_t* bias,
                      const volatile tt_l1_ptr uint16_t* table, uint32_t* ids, uint16_t* weights) {
    constexpr bool round_logits = flags & 1u;
    constexpr bool round_scores = flags & 2u;
    constexpr uint32_t eps_q24 = 17;  // Q24 of the 1e-6 in the normalization denominator
    uint32_t scores[num_experts];
    TopK<top_k> best;
    if constexpr (round_logits && round_scores) {
        constexpr uint32_t n_exp = e_hi - e_lo + 1;
        for (uint32_t e = 0; e < num_experts; ++e) {
            const uint32_t b = f32_round_bf16(logits[e]) >> 16;
            const uint32_t mag = b & 0x7fffu, ex = mag >> 7;
            uint16_t s16;
            if (ex < e_lo) {
                s16 = 0x3f00;  // 0.5
            } else if (ex > e_hi) {
                s16 = (b & 0x8000u) ? 0 : 0x3f80;
            } else {
                s16 = table[((b >> 15) * n_exp + (ex - e_lo)) * 128 + (mag & 0x7fu)];
            }
            const uint32_t s = bf16_to_q24(s16);
            scores[e] = s;
            best.consider(static_cast<int32_t>(s) + bias[e], e);
        }
    } else {
        // approximation 1024 (table 788 + Q16 input rounding + the exact path's own 2e-6 exp error), plus half a
        // BF16 ulp of a score < 1 when the exact score is rounded to BF16
        constexpr int32_t delta = 1024 + (round_scores ? (1 << 15) : 1);
        uint32_t bits[num_experts];
        int32_t approx[num_experts];
        TopK<top_k> rough;
        for (uint32_t e = 0; e < num_experts; ++e) {
            uint32_t b = logits[e];
            if constexpr (round_logits) {
                b = f32_round_bf16(b);
            }
            bits[e] = b;
            approx[e] = static_cast<int32_t>(sigmoid_q24_approx(b)) + bias[e];
            rough.consider(approx[e], e);
        }
        const int32_t floor_key = rough.key[top_k - 1] - 2 * delta;
        for (uint32_t e = 0; e < num_experts; ++e) {
            if (approx[e] < floor_key) {
                continue;
            }
            const uint64_t q62 = sigmoid_q62(bits[e]);
            uint32_t s;
            if constexpr (round_scores) {
                s = bf16_to_q24(q62_to_bf16(q62));
            } else {
                s = static_cast<uint32_t>((q62 + (uint64_t{1} << 37)) >> 38);
            }
            scores[e] = s;
            best.consider(static_cast<int32_t>(s) + bias[e], e);
        }
    }
    uint32_t* best_id = best.id;
    for (uint32_t k = 1; k < top_k; ++k) {  // ascending ids
        const uint32_t id = best_id[k];
        uint32_t pos = k;
        while (pos > 0 && best_id[pos - 1] > id) {
            best_id[pos] = best_id[pos - 1];
            --pos;
        }
        best_id[pos] = id;
    }
    uint64_t total = eps_q24;
    for (uint32_t k = 0; k < top_k; ++k) {
        total += scores[best_id[k]];
    }
    const uint64_t reciprocal = (uint64_t{1} << 62) / total;
    for (uint32_t k = 0; k < top_k; ++k) {
        ids[k] = best_id[k];
        weights[k] = q62_to_bf16(static_cast<uint64_t>(scores[best_id[k]]) * reciprocal);
    }
}
}  // namespace
