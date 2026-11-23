// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling-2-Mini router tail on a data-movement RISC: top-k selection on FP32 bit patterns (exact, ties to the lower
// expert id) and the renormalised, scaled routing weights in IEEE single precision (libgcc soft float).
#pragma once

#include <cstdint>

namespace ink_moe {

// Monotone map FP32 bits -> unsigned order (larger float -> larger key; -0 < +0, NaNs at the ends).
inline uint32_t order_key(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }

// Top 8 of sel[e0, e1) in registers: the sorted list is 16 scalars, an insertion a chain of register compare-exchanges
// that stops at the first non-swap (the generic version's arrays cost ~70 cycles per insertion in memory).  Strict
// compares: equal keys keep the earlier (lower) id first.
#define INK_TOP8_CX(a, b)            \
    if (k##b > k##a) {               \
        uint32_t t_ = k##a;          \
        k##a = k##b;                 \
        k##b = t_;                   \
        t_ = i##a;                   \
        i##a = i##b;                 \
        i##b = t_;                   \
    } else {                         \
        break;                       \
    }
#define INK_TOP8_TRY(kk, e)                                                                                    \
    if ((kk) > k7) {                                                                                           \
        k7 = (kk);                                                                                             \
        i7 = (e);                                                                                              \
        do {                                                                                                   \
            INK_TOP8_CX(6, 7) INK_TOP8_CX(5, 6) INK_TOP8_CX(4, 5) INK_TOP8_CX(3, 4) INK_TOP8_CX(2, 3)           \
            INK_TOP8_CX(1, 2) INK_TOP8_CX(0, 1)                                                                \
        } while (0);                                                                                           \
    }

__attribute__((noinline, optimize("O2"))) inline void select_top8_keys(const volatile uint32_t* sel, uint32_t e0,
                                                                        uint32_t e1, uint32_t* keys, uint32_t* ids) {
    uint32_t k0 = 0, k1 = 0, k2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0, k7 = 0;
    uint32_t i0 = e0, i1 = e0 + 1, i2 = e0 + 2, i3 = e0 + 3, i4 = e0 + 4, i5 = e0 + 5, i6 = e0 + 6, i7 = e0 + 7;
    auto key = [](uint32_t b) {
        return b ^ (static_cast<uint32_t>(static_cast<int32_t>(b) >> 31) | 0x80000000u);  // == order_key(b)
    };
    for (uint32_t e = e0; e < e1; e += 4) {
        const uint32_t b0 = sel[e], b1 = sel[e + 1], b2 = sel[e + 2], b3 = sel[e + 3];
        const uint32_t q0 = key(b0), q1 = key(b1), q2 = key(b2), q3 = key(b3);
        INK_TOP8_TRY(q0, e)
        INK_TOP8_TRY(q1, e + 1)
        INK_TOP8_TRY(q2, e + 2)
        INK_TOP8_TRY(q3, e + 3)
    }
    keys[0] = k0, keys[1] = k1, keys[2] = k2, keys[3] = k3, keys[4] = k4, keys[5] = k5, keys[6] = k6, keys[7] = k7;
    ids[0] = i0, ids[1] = i1, ids[2] = i2, ids[3] = i3, ids[4] = i4, ids[5] = i5, ids[6] = i6, ids[7] = i7;
}
#undef INK_TOP8_TRY
#undef INK_TOP8_CX

// sel[e] FP32 bits of scores + bias for e in [e0, e1) -> keys / ids[0..K) by descending value (ties: lower id
// first).  Loads are issued 4 at a time (L1 load latency dominates a scalar scan); the running K-th key rejects most
// experts.  K = 8: select_top8_keys.
template <uint32_t K>
inline void select_topk_keys(const volatile uint32_t* sel, uint32_t e0, uint32_t e1, uint32_t* keys, uint32_t* ids) {
    static_assert(K >= 1);
    if constexpr (K == 8) {
        select_top8_keys(sel, e0, e1, keys, ids);
        return;
    }
    for (uint32_t k = 0; k < K; ++k) {
        keys[k] = 0;  // below every order key except the all-ones NaN pattern
        ids[k] = e0 + k;
    }
    uint32_t thr = 0;
    auto insert = [&](uint32_t key, uint32_t e) {
        uint32_t p = K - 1;
        while (p > 0 && keys[p - 1] < key) {
            keys[p] = keys[p - 1];
            ids[p] = ids[p - 1];
            --p;
        }
        keys[p] = key;
        ids[p] = e;
        thr = keys[K - 1];
    };
    for (uint32_t e = e0; e < e1; e += 4) {
        const uint32_t k0 = order_key(sel[e]), k1 = order_key(sel[e + 1]), k2 = order_key(sel[e + 2]),
                       k3 = order_key(sel[e + 3]);
        if (k0 > thr) {
            insert(k0, e);
        }
        if (k1 > thr) {
            insert(k1, e + 1);
        }
        if (k2 > thr) {
            insert(k2, e + 2);
        }
        if (k3 > thr) {
            insert(k3, e + 3);
        }
    }
}

template <uint32_t E, uint32_t K>
inline void select_topk(const volatile uint32_t* sel, uint32_t* ids) {
    static_assert(E % 4 == 0);
    uint32_t keys[K];
    select_topk_keys<K>(sel, 0, E, keys, ids);
}

// Top K of two descending (key, id) lists whose ids are all lower in `a` than in `b` -> ids (ties: a first = the lower
// id, the order of select_topk over the whole range).
template <uint32_t K>
inline void merge_topk(const uint32_t* ka, const uint32_t* ia, const volatile uint32_t* kb, const volatile uint32_t* ib,
                       uint32_t* ids) {
    uint32_t i = 0, j = 0;
    for (uint32_t k = 0; k < K; ++k) {
        if (ka[i] >= kb[j]) {
            ids[k] = ia[i++];
        } else {
            ids[k] = ib[j++];
        }
    }
}

inline float as_float(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } c{bits};
    return c.f;
}

inline uint32_t as_bits(float f) {
    union {
        float f;
        uint32_t u;
    } c{f};
    return c.u;
}

// Reference: w = s[ids]; w / (sum(w) + 1e-20) * scale  (FP32; the sum in selection order).
template <uint32_t K>
inline void route_weights(const volatile uint32_t* scores, const uint32_t* ids, float scale, float* w) {
    float s[K];
    float sum = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        s[k] = as_float(scores[ids[k]]);
        sum += s[k];
    }
    const float denom = sum + 1e-20f;
    for (uint32_t k = 0; k < K; ++k) {
        w[k] = (s[k] / denom) * scale;
    }
}

// Gate/up pair placement (moe_layout.py gu_pairs / gu_start / gu_before).
template <uint32_t BASE, uint32_t ODD>
struct PairPlacement {
    static uint32_t pairs(uint32_t e, uint32_t b) { return BASE + (ODD && ((e + b) % 2 == 0) ? 1 : 0); }
    static uint32_t start(uint32_t e, uint32_t b) { return b * BASE + ODD * ((b + (e % 2 == 0 ? 1 : 0)) / 2); }
    static uint32_t before(uint32_t e, uint32_t b) { return e * BASE + ODD * ((e + (b % 2 == 0 ? 1 : 0)) / 2); }
};

// Bank-balanced gate/up units (BAL; needs ODD).  A bank b with pairs(e, b) = BASE + 1 holds expert e's "floating" pair
// (its last, l = BASE); the same pair is also stored in bank b + 1 (moe_layout.gate_up_dup_layout, unit e / 2 of that
// bank's dup run).  The static layout gives the banks of parity p BASE U + #slots of parity p units; balance_floating
// moves the floating pairs of the last (n_major - n_minor) / 2 slots of the majority parity to the next bank, so
// every bank streams floor / ceil(U (2 BASE + 1) / 2) units and no pair moves when the parities are balanced.
// moved[u] bit b: slot u's floating pair of bank b is served by bank b + 1.  The pair's data, and so every result
// bit, is the same whichever bank serves it.
template <uint32_t BANKS, uint32_t BASE, typename S>
inline void balance_floating(const S* slot_e, uint32_t U, volatile uint8_t* moved) {
    static_assert(BANKS == 8);
    uint32_t n_odd = 0;
    for (uint32_t u = 0; u < U; ++u) {
        n_odd += slot_e[u] & 1;
    }
    const uint32_t n_even = U - n_odd;
    const uint32_t major = n_even >= n_odd ? 0 : 1;
    uint32_t m = (major ? n_odd - n_even : n_even - n_odd) / 2;
    for (uint32_t u = U; u-- > 0;) {
        uint8_t mask = 0;
        if (m && (slot_e[u] & 1) == major) {
            mask = major ? 0xaa : 0x55;  // the banks b with (e + b) even
            --m;
        }
        moved[u] = mask;
    }
}

// Bank `bank`'s gate/up units in slot order: f(u, j, unit, dup) with j the pair (intermediate tile) index and unit the
// unit index in the bank's main run (dup = false) or dup run (dup = true).  BAL = false: the static layout.
template <uint32_t BANKS, uint32_t BASE, uint32_t ODD, bool BAL, typename S, typename F>
inline void bank_units(uint32_t bank, const S* slot_e, const volatile uint8_t* moved, uint32_t U, F&& f) {
    using P = PairPlacement<BASE, ODD>;
    static_assert(!BAL || ODD);
    const uint32_t prev = bank == 0 ? BANKS - 1 : bank - 1;
    for (uint32_t u = 0; u < U; ++u) {
        const uint32_t e = slot_e[u];
        const uint32_t np = P::pairs(e, bank);
        const uint32_t n_main = (BAL && np == BASE + 1 && ((moved[u] >> bank) & 1)) ? BASE : np;
        const uint32_t j0 = P::start(e, bank), i0 = P::before(e, bank);
        for (uint32_t l = 0; l < n_main; ++l) {
            f(u, j0 + l, i0 + l, false);
        }
        if (BAL && np == BASE && ((moved[u] >> prev) & 1)) {
            f(u, P::start(e, prev) + BASE, e / 2, true);
        }
    }
}

// Union of n expert ids (< E) in ascending id order -> slot_e[0..U), returns U.
template <uint32_t E>
inline uint32_t union_ascending(const volatile uint32_t* ids, uint32_t n, uint32_t* slot_e) {
    static_assert(E % 32 == 0);
    uint32_t mask[E / 32];
    for (uint32_t w = 0; w < E / 32; ++w) {
        mask[w] = 0;
    }
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t e = ids[i];
        mask[e >> 5] |= 1u << (e & 31);
    }
    uint32_t u = 0;
    for (uint32_t w = 0; w < E / 32; ++w) {
        for (uint32_t m = mask[w]; m; m &= m - 1) {
            slot_e[u++] = 32 * w + __builtin_ctz(m);
        }
    }
    return u;
}

}  // namespace ink_moe
