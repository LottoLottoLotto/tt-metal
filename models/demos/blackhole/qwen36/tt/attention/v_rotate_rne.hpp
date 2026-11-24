// SPDX-License-Identifier: Apache-2.0
// Round-to-nearest-even BFP4 (bfloat4_b) quantization of one 16-value BF16 block (one face row of
// a 32x32 tile), in integer arithmetic. Semantics are ttquant.bfp.rtn / tt-metal's host
// convert_u32_to_bfp<Bfp4_b, round-to-nearest>: shared exponent E = max biased exponent of the
// block (zero/denormal inputs contribute 0), q = RNE(|x| * 2^(2 - (E - 127))) saturated at 7 (no
// exponent bump), sign dropped when q == 0. The device packer TRUNCATES when it converts to BFP4,
// so the cache writes go through these values: on the grid the packer (and paged_update_cache's
// tilize) stores them exactly. Plain C++ so the host check compiles the same code.
#pragma once
#include <stdint.h>

namespace v_rotate_rne {

// BF16 encoding of q * 2^(E - 129): biased exponent E + kExpOffset[q] and mantissa kMantissa[q].
constexpr int32_t kExpOffset[8] = {0, -2, -1, -1, 0, 0, 0, 0};
constexpr uint32_t kMantissa[8] = {0, 0x00, 0x00, 0x40, 0x00, 0x20, 0x40, 0x60};

inline uint32_t block_exponent(const volatile uint32_t* words) {
    uint32_t e = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t w = words[i];
        const uint32_t lo = (w >> 7) & 0xFF;
        const uint32_t hi = (w >> 23) & 0xFF;
        e = lo > e ? lo : e;
        e = hi > e ? hi : e;
    }
    return e;
}

// 4-bit code (sign << 3 | q) of BF16 value h in a block with shared exponent E.
inline uint32_t code(uint32_t h, uint32_t E) {
    const uint32_t e = (h >> 7) & 0xFF;
    if (e == 0) {
        return 0;
    }
    const uint32_t shift = E - e + 5;  // >= 5: |x| = S * 2^(e-134), y = S * 2^(e-E-5)
    if (shift > 8) {
        return 0;  // S < 2^8: y < 0.5
    }
    const uint32_t s = 0x80 | (h & 0x7F);
    uint32_t q = s >> shift;
    const uint32_t rem = s & ((1u << shift) - 1);
    const uint32_t half = 1u << (shift - 1);
    q += (rem > half || (rem == half && (q & 1))) ? 1 : 0;
    q = q > 7 ? 7 : q;
    return q == 0 ? 0 : (((h >> 15) & 1) << 3) | q;
}

// BF16 bits of the value a 4-bit code denotes in a block with shared exponent E (values whose
// exponent would underflow BF16's normal range flush to +0; never hit by real activations).
inline uint32_t decode(uint32_t c, uint32_t E) {
    const uint32_t q = c & 7;
    if (q == 0) {
        return 0;
    }
    const int32_t exponent = static_cast<int32_t>(E) + kExpOffset[q];
    if (exponent <= 0) {
        return 0;
    }
    return ((c >> 3) << 15) | (static_cast<uint32_t>(exponent) << 7) | kMantissa[q];
}

// In-place: 16 BF16 values (8 words) -> their BFP4 grid values (still BF16).
inline void round_row_bf16(volatile uint32_t* words) {
    const uint32_t E = block_exponent(words);
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t w = words[i];
        const uint32_t lo = decode(code(w & 0xFFFF, E), E);
        const uint32_t hi = decode(code(w >> 16, E), E);
        words[i] = lo | (hi << 16);
    }
}

// 16 BF16 values (8 words) -> shared exponent (returned) and two BFP4 data words (value j of the
// row in bits 4j..4j+3 of word j / 8, as tt-metal's pack_as_bfp_tiles lays out a face row).
inline uint32_t encode_row_bfp4(const volatile uint32_t* words, volatile uint32_t* data) {
    const uint32_t E = block_exponent(words);
    for (uint32_t half = 0; half < 2; ++half) {
        uint32_t packed = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t w = words[half * 4 + i];
            packed |= code(w & 0xFFFF, E) << (8 * i);
            packed |= code(w >> 16, E) << (8 * i + 4);
        }
        data[half] = packed;
    }
    return E;
}

}  // namespace v_rotate_rne
