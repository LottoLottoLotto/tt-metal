// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// Integer helpers of the v2 decode MoE routing (no device includes: also compiled by the host check
// tests/fast/check_moe2_math.cpp, which compares them with the plain 64-bit expressions exhaustively).
#pragma once
#include <cstdint>

namespace moe2 {

// (u1 * 2^32 + u0) / v for u1 < v (quotient < 2^32), with 32-bit divisions only (Hacker's Delight divlu).
inline uint32_t divlu(uint32_t u1, uint32_t u0, uint32_t v) {
    constexpr uint32_t b = 65536;
    const int s = __builtin_clz(v);
    v <<= s;
    const uint32_t vn1 = v >> 16, vn0 = v & 0xffff;
    const uint32_t un32 = (u1 << s) | (s ? u0 >> (32 - s) : 0);
    const uint32_t un10 = u0 << s;
    const uint32_t un1 = un10 >> 16, un0 = un10 & 0xffff;
    uint32_t q1 = un32 / vn1, rhat = un32 - q1 * vn1;
    while (q1 >= b || q1 * vn0 > b * rhat + un1) {
        --q1;
        rhat += vn1;
        if (rhat >= b) {
            break;
        }
    }
    const uint32_t un21 = un32 * b + un1 - q1 * v;
    uint32_t q0 = un21 / vn1;
    rhat = un21 - q0 * vn1;
    while (q0 >= b || q0 * vn0 > b * rhat + un0) {
        --q0;
        rhat += vn1;
        if (rhat >= b) {
            break;
        }
    }
    return q1 * b + q0;
}

// floor(2^62 / t) for 1 <= t < 2^32: the routing's reciprocal without the 64-bit libgcc division.
inline uint64_t recip62(uint32_t t) {
    const uint32_t hi = (1u << 30) / t;
    const uint32_t rem = (1u << 30) - hi * t;
    return (static_cast<uint64_t>(hi) << 32) | divlu(rem, 0, t);
}

// routing_math.hpp's q62_to_bf16 (unsigned Q62 -> BF16, round to nearest even) with a count-leading-zeros top bit.
inline uint16_t q62_to_bf16_clz(uint64_t q) {
    if (q == 0) {
        return 0;
    }
    const uint32_t hi = static_cast<uint32_t>(q >> 32);
    const uint32_t top = hi ? 63 - __builtin_clz(hi) : 31 - __builtin_clz(static_cast<uint32_t>(q));
    uint32_t exponent = top + 127 - 62;
    uint32_t mantissa;
    if (top >= 8) {
        const uint32_t drop = top - 7;
        mantissa = static_cast<uint32_t>(q >> drop);
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

}  // namespace moe2
