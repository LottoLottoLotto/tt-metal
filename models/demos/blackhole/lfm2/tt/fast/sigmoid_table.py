# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Host emulation of the integer router sigmoid (tt/kernels/sigmoid_routing.cpp with flags 3: BF16-rounded logits,
BF16-rounded scores) and the score table the decode router tail reads instead of evaluating it.

With flags 3 a score depends only on the logit's BF16 bits, so it is tabulated exactly: score_bf16[sign][e][m] for
BF16 exponents e in [E_LO, E_HI] (mantissa m); below E_LO the score is 0.5 for both signs, above E_HI it is 1 (positive)
or 0 (negative).  `build()` checks those constant ranges against the emulation for every BF16 magnitude.
"""
from __future__ import annotations

import torch

POW2_NEG = [2147483648, 2101467502, 2056437387, 2012372174, 1969251188, 1927054196, 1885761398, 1845353420,
            1805811301, 1767116489, 1729250827, 1692196547, 1655936265, 1620452965, 1585730000, 1551751076,
            1518500250, 1485961921, 1454120821, 1422962010, 1392470869, 1362633090, 1333434672, 1304861917,
            1276901417, 1249540052, 1222764986, 1196563654, 1170923762, 1145833280, 1121280436, 1097253708,
            1073741824]
M32, M64 = (1 << 32) - 1, (1 << 64) - 1


def f32_abs_q16(bits: int) -> int:
    exponent = (bits >> 23) & 0xFF
    if exponent == 0:
        return 0
    mantissa = 0x800000 | (bits & 0x7FFFFF)
    shift = exponent - 134
    if shift >= 0:
        return 0x3FFFFFFF if shift > 6 else (mantissa << shift) & M32
    if shift <= -25:
        return 0
    return (mantissa + (1 << (-shift - 1))) >> -shift


def exp_neg_q31(d: int) -> int:
    t = ((d * 1549082005) & M64) >> 30
    whole = t >> 16
    if whole >= 31:
        return 0
    frac = t & 0xFFFF
    a = (((frac & 0x7FF) * 1488522236) & M64) >> 16
    a2_half = ((a * a) & M64) >> 32
    poly = (0x80000000 - a + a2_half) & M32
    value = ((POW2_NEG[frac >> 11] * poly) & M64) >> 31
    return (value & M32) >> whole


def q62_to_bf16(q: int) -> int:
    if q == 0:
        return 0
    top = q.bit_length() - 1
    exponent = top + 127 - 62
    if top >= 8:
        drop = top - 7
        mantissa = q >> drop
        rest = q & ((1 << drop) - 1)
        half = 1 << (drop - 1)
        if rest > half or (rest == half and (mantissa & 1)):
            mantissa += 1
            if mantissa == 0x100:
                mantissa = 0x80
                exponent += 1
    else:
        mantissa = q << (7 - top)
    return ((exponent << 7) | (mantissa & 0x7F)) & 0xFFFF


def score_bf16(bf16_bits: int) -> int:
    """BF16 bits of the router score for a BF16 logit (sigmoid_routing.cpp with flags 3)."""
    bits = bf16_bits << 16
    e = exp_neg_q31(f32_abs_q16(bits))
    inverse = (1 << 62) // (0x80000000 + e)
    numerator = e if bits & 0x80000000 else 0x80000000
    return q62_to_bf16(numerator * inverse)


E_LO, E_HI = 113, 131  # tabulated BF16 exponent range (checked in build())
HALF, ONE = 0x3F00, 0x3F80


def build() -> torch.Tensor:
    """[2 * (E_HI - E_LO + 1) * 128] uint16 BF16 score bits: index ((sign * n_exp) + (e - E_LO)) * 128 + mantissa."""
    for mag in range(0x0000, 0x7F80):  # every finite magnitude
        e = mag >> 7
        if e < E_LO:
            assert score_bf16(mag) == HALF and score_bf16(mag | 0x8000) == HALF, hex(mag)
        elif e > E_HI:
            assert score_bf16(mag) == ONE and score_bf16(mag | 0x8000) == 0, hex(mag)
    n_exp = E_HI - E_LO + 1
    table = [score_bf16((sign << 15) | ((E_LO + i) << 7) | m)
             for sign in (0, 1) for i in range(n_exp) for m in range(128)]
    return torch.tensor(table, dtype=torch.int32).to(torch.int16)
