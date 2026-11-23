// SPDX-License-Identifier: Apache-2.0
// Arithmetic for exact on-device vLLM sampling (tt/sampling.py; forked from Qwen36's device_sampling_math.hpp and
// extended with the full-vocabulary probability mass for top_k-disabled rows). Shared by the data-movement kernel
// and the host unit test. On device every float/double operation is an explicit libgcc IEEE soft-float call: the JIT
// builds with -ffast-math, which would otherwise rewrite x / T as x * (1 / T), fuse multiply-adds, and assume
// finite math. The host test build maps the same names to native IEEE operations (-fno-fast-math).
#pragma once
#include <cstdint>

#if defined(DEVICE_SAMPLING_HOST_TEST)
static inline float s_add(float a, float b) { return a + b; }
static inline float s_sub(float a, float b) { return a - b; }
static inline float s_mul(float a, float b) { return a * b; }
static inline float s_div(float a, float b) { return a / b; }
static inline float s_u2f(uint32_t a) { return (float)a; }
static inline float s_i2f(int32_t a) { return (float)a; }
static inline int32_t s_f2i(float a) { return (int32_t)a; }
static inline double d_add(double a, double b) { return a + b; }
static inline double d_sub(double a, double b) { return a - b; }
static inline double d_mul(double a, double b) { return a * b; }
static inline double d_div(double a, double b) { return a / b; }
static inline double d_f2d(float a) { return (double)a; }
static inline float d_d2f(double a) { return (float)a; }
static inline double d_u642d(uint64_t a) { return (double)a; }
static inline float s_i642f(int64_t a) { return (float)a; }
#else
extern "C" {
float __addsf3(float, float);
float __subsf3(float, float);
float __mulsf3(float, float);
float __divsf3(float, float);
float __floatunsisf(uint32_t);
float __floatsisf(int32_t);
int32_t __fixsfsi(float);
double __adddf3(double, double);
double __subdf3(double, double);
double __muldf3(double, double);
double __divdf3(double, double);
double __extendsfdf2(float);
float __truncdfsf2(double);
double __floatundidf(uint64_t);
float __floatdisf(int64_t);
}
static inline float s_add(float a, float b) { return __addsf3(a, b); }
static inline float s_sub(float a, float b) { return __subsf3(a, b); }
static inline float s_mul(float a, float b) { return __mulsf3(a, b); }
static inline float s_div(float a, float b) { return __divsf3(a, b); }
static inline float s_u2f(uint32_t a) { return __floatunsisf(a); }
static inline float s_i2f(int32_t a) { return __floatsisf(a); }
static inline int32_t s_f2i(float a) { return __fixsfsi(a); }
static inline double d_add(double a, double b) { return __adddf3(a, b); }
static inline double d_sub(double a, double b) { return __subdf3(a, b); }
static inline double d_mul(double a, double b) { return __muldf3(a, b); }
static inline double d_div(double a, double b) { return __divdf3(a, b); }
static inline double d_f2d(float a) { return __extendsfdf2(a); }
static inline float d_d2f(double a) { return __truncdfsf2(a); }
static inline double d_u642d(uint64_t a) { return __floatundidf(a); }
static inline float s_i642f(int64_t a) { return __floatdisf(a); }
#endif

namespace dsm {

constexpr uint32_t NO_ID = 0xffffffffu;
// Row flags (params word 0). F_FULL: top_k disabled or above the candidate count; the scan also accumulates the
// row's full-vocabulary probability mass.  F_NOCOMMIT: the row op does not append its token to the history
// (speculative verify rows: tt/speculate.py SpecCommit appends the emitted tokens).  Params word 12 (W_EXTRA): the
// row's history also counts aux tokens [b, b + n), b = params word 15 (W_EXTRA_BASE) (a shared verify round: the
// drafts of the row's group before this row).
constexpr uint32_t F_ACTIVE = 1, F_GREEDY = 2, F_PENALTY = 4, F_REP = 8, F_NOCOMMIT = 32, F_MONO = 64, F_FULL = 128;
constexpr uint32_t W_EXTRA = 12, W_EXTRA_BASE = 15;
// Result flags: R_HOST = the host redraws this row from the logits (bit 0); the reason in the upper bits.
constexpr uint32_t R_HOST = 1, R_TIE = 2, R_TAIL = 4, R_CUT = 8;
// Mass weights below 2^-MASS_BITS of the maximum are dropped (<= 128000 * 2^-46 = 1.8e-9 of the mass).
constexpr uint32_t MASS_BITS = 46;
// RNG purpose (Philox counter word 1).
constexpr uint32_t P_SAMPLE = 1;

static inline uint32_t fbits(float f) { return __builtin_bit_cast(uint32_t, f); }
static inline float bitsf(uint32_t u) { return __builtin_bit_cast(float, u); }
static inline uint64_t dbits(double d) { return __builtin_bit_cast(uint64_t, d); }
static inline double bitsd(uint64_t u) { return __builtin_bit_cast(double, u); }

// Monotonic unsigned key of an IEEE float (NaNs excluded). -0 is canonicalized to +0 (torch compares them equal).
static inline uint32_t fkey(uint32_t f) {
    if (f == 0x80000000u) f = 0;
    return f ^ ((uint32_t)((int32_t)f >> 31) | 0x80000000u);
}
static inline uint32_t keyf(uint32_t k) { return k & 0x80000000u ? k ^ 0x80000000u : ~k; }
static inline uint64_t dkey(double d) {
    uint64_t u = dbits(d);
    if (u == 0x8000000000000000ull) u = 0;
    return u ^ ((uint64_t)((int64_t)u >> 63) | 0x8000000000000000ull);
}
static inline bool f_gt(float a, float b) { return fkey(fbits(a)) > fkey(fbits(b)); }
static inline bool f_ge(float a, float b) { return fkey(fbits(a)) >= fkey(fbits(b)); }
static inline bool f_lt(float a, float b) { return fkey(fbits(a)) < fkey(fbits(b)); }
static inline bool d_lt(double a, double b) { return dkey(a) < dkey(b); }
static inline bool d_le(double a, double b) { return dkey(a) <= dkey(b); }
static inline bool d_gt(double a, double b) { return dkey(a) > dkey(b); }

// vLLM apply_all_penalties on one float32 logit (CPU torch semantics):
// repetition: logits *= (logits > 0 ? fl(1/rep) : rep) where the token occurs in prompt or output;
// frequency: logits -= fl(freq * count); presence: logits -= presence * (count > 0).
static inline uint32_t penalize(uint32_t fb, uint32_t count, bool in_prompt, bool rep_on, float rep, float recip_rep,
                                float freq, float pres) {
    float l = bitsf(fb);
    if (rep_on && (in_prompt || count)) {
        l = s_mul(l, f_gt(l, bitsf(0)) ? recip_rep : rep);
    }
    l = s_sub(l, s_mul(freq, s_u2f(count)));
    if (count) l = s_sub(l, pres);
    return fbits(l);
}

// exp(x) for finite x <= 0 in float32 (Cephes expf: Cody-Waite reduction, degree-6 polynomial, <= 1 ulp).
static inline float exp_nonpos(float x) {
    if (f_lt(x, bitsf(0xc2d00000u))) return bitsf(0);  // x < -104: below the smallest denormal
    const float log2e = bitsf(0x3fb8aa3bu), half = bitsf(0x3f000000u);
    float t = s_mul(x, log2e);
    int32_t n = s_f2i(s_sub(t, half));  // round half down; t <= 0
    float fn = s_i2f(n);
    // If t - 0.5 was an exact integer, fixsfsi truncated toward zero; either neighbour keeps |r| <= ln2.
    float r = s_sub(s_sub(x, s_mul(fn, bitsf(0x3f317200u))), s_mul(fn, bitsf(0x35bfbe8eu)));
    float z = s_mul(r, r);
    float p = bitsf(0x39506967u);                                // 1.9875691500E-4
    p = s_add(s_mul(p, r), bitsf(0x3ab743ceu));                  // 1.3981999507E-3
    p = s_add(s_mul(p, r), bitsf(0x3c088908u));                  // 8.3334519073E-3
    p = s_add(s_mul(p, r), bitsf(0x3d2aa9c1u));                  // 4.1665795894E-2
    p = s_add(s_mul(p, r), bitsf(0x3e2aaaaau));                  // 1.6666665459E-1
    p = s_add(s_mul(p, r), bitsf(0x3f000000u));                  // 5.0000001201E-1
    p = s_add(s_add(s_mul(p, z), r), bitsf(0x3f800000u));
    // Scale by 2^n. p is in [0.5, 2); n >= -151 here.
    if (n >= -125) return bitsf(fbits(p) + ((uint32_t)n << 23));
    float scaled = bitsf(fbits(p) + ((uint32_t)(n + 100) << 23));  // normal intermediate
    return s_mul(scaled, bitsf(0x0d800000u));                       // * 2^-100, correctly rounded denormal
}

// 2^(-k/64) in Q1.31, k = 0..63.
static constexpr uint32_t EXP2_NEG_TABLE[64] = {
    0x80000000u, 0x7e9f0606u, 0x7d41d96eu, 0x7be86fbau, 0x7a92be8bu, 0x7940bb9eu, 0x77f25cceu, 0x76a7980fu,
    0x75606374u, 0x741cb528u, 0x72dc8374u, 0x719fc4b9u, 0x70666f76u, 0x6f307a41u, 0x6dfddbccu, 0x6cce8ae1u,
    0x6ba27e65u, 0x6a79ad56u, 0x69540ec9u, 0x683199edu, 0x6712460bu, 0x65f60a7fu, 0x64dcdec3u, 0x63c6ba64u,
    0x62b39509u, 0x61a3666du, 0x60962665u, 0x5f8bccdbu, 0x5e8451d0u, 0x5d7fad59u, 0x5c7dd7a4u, 0x5b7ec8f2u,
    0x5a82799au, 0x5988e209u, 0x5891fac1u, 0x579dbc57u, 0x56ac1f75u, 0x55bd1cdbu, 0x54d0ad5au, 0x53e6c9dau,
    0x52ff6b55u, 0x521a8ad7u, 0x51382182u, 0x50582888u, 0x4f7a9930u, 0x4e9f6cd4u, 0x4dc69cddu, 0x4cf022cau,
    0x4c1bf829u, 0x4b4a169cu, 0x4a7a77d4u, 0x49ad1598u, 0x48e1e9bau, 0x4818ee22u, 0x47521cc6u, 0x468d6faeu,
    0x45cae0f2u, 0x450a6abbu, 0x444c0740u, 0x438fb0cbu, 0x42d561b4u, 0x421d1462u, 0x4166c34cu, 0x40b268fau,
};

// 2^(-frac / 2^24) for frac in [0, 2^24) in Q1.31, in (2^30, 2^31] (relative error ~1e-9).
static inline uint32_t exp2_frac_q31(uint32_t frac) {
    const uint32_t k = frac >> 18;
    const uint32_t rq = (frac & 0x3ffffu) << 6;                              // r in Q0.30, r < 1/64
    const uint32_t a = (uint32_t)(((uint64_t)rq * 0x58B90BFCu) >> 30);      // a = r ln2 in Q0.31
    const uint32_t a2 = (uint32_t)(((uint64_t)a * a) >> 31);
    const uint32_t a3 = (uint32_t)(((uint64_t)a2 * a) >> 31);
    const uint32_t a4 = (uint32_t)(((uint64_t)a3 * a) >> 31);
    const uint32_t poly = 0x80000000u - a + (a2 >> 1) - a3 / 6 + a4 / 24;    // e^-a in Q1.31
    return (uint32_t)(((uint64_t)EXP2_NEG_TABLE[k] * poly) >> 31);
}

// 2^(-y) for y = yq / 2^24 in [0, 150), integer arithmetic only (relative error ~1e-9 before the final rounding to
// float32, i.e. at most one ulp off the correctly rounded value).
static inline float exp2_neg_q24(uint32_t yq) {
    const uint32_t n = yq >> 24;
    const uint32_t mant = exp2_frac_q31(yq & 0xffffffu);                     // 2^-frac in Q1.31, (2^30, 2^31]
    const uint32_t lz = (uint32_t)__builtin_clz(mant);
    const int32_t exponent = 127 - (int32_t)lz - (int32_t)n;                 // value = (mant << lz) * 2^(exponent-158)
    uint32_t norm = mant << lz;                                              // bit 31 set
    uint32_t bits;
    if (exponent >= 1) {
        uint32_t q = norm >> 8, rem = norm & 0xffu;
        q += (rem > 0x80u || (rem == 0x80u && (q & 1u))) ? 1u : 0u;           // round to nearest even
        bits = ((uint32_t)exponent << 23) + (q - 0x800000u);                   // carry into the exponent is correct
    } else {
        const uint32_t shift = (uint32_t)(9 - exponent);                       // denormal: value / 2^-149
        if (shift >= 33) return bitsf(0);
        uint32_t q = shift == 32 ? 0u : norm >> shift;
        const uint64_t rem = ((uint64_t)norm << (32 - shift)) & 0xffffffffull;
        q += (rem > 0x80000000ull || (rem == 0x80000000ull && (q & 1u))) ? 1u : 0u;
        bits = q;
    }
    return bitsf(bits);
}

// ---- Full-vocabulary mass (F_FULL rows): sum_i 2^(-(m - l_i) c) in Q.46 fixed point, c = log2(e) / T, relative to
// a reference m (the worker's running maximum, then the global maximum). Logits as signed Q.24 fixed point.

// Signed Q.24 of a finite float32 (|v| < 2^39; magnitudes below 2^-24 read as 0).
static inline int64_t f2q24(uint32_t b) {
    const uint32_t e = (b >> 23) & 0xffu;
    if (e < 103) return 0;
    const uint64_t m = (b & 0x7fffffu) | 0x800000u;
    const int32_t sh = (int32_t)e - 126;  // v = m * 2^(e - 150) -> Q.24 = m * 2^(e - 126)
    const uint64_t q = sh >= 0 ? (m << (sh > 39 ? 39 : sh)) : (m >> -sh);
    return (b >> 31) ? -(int64_t)q : (int64_t)q;
}

// y = d * c in Q.24 for d >= 0 in Q.24 and c in unsigned Q32.32; saturates at 2^62 (far beyond any cutoff).
static inline uint64_t scale_q24(uint64_t d, uint32_t c_lo, uint32_t c_hi) {
    if ((d >> 32) == 0) {
        const uint32_t d32 = (uint32_t)d;
        return (uint64_t)d32 * c_hi + (((uint64_t)d32 * c_lo) >> 32);
    }
    if ((d >> 40) != 0) return 1ull << 62;
    const uint32_t dh = (uint32_t)(d >> 8);  // >= 256 logits below the reference: Q.16 is plenty
    const uint64_t y = (uint64_t)dh * c_hi + (((uint64_t)dh * c_lo) >> 32);
    return (y >> 54) ? (1ull << 62) : (y << 8);
}

// 2^(-k/512) in Q1.31, k = 0..511 (the per-token mass weights).
static constexpr uint32_t EXP2_512[512] = {
    0x80000000u, 0x7fd3ab29u, 0x7fa765adu, 0x7f7b2f85u, 0x7f4f08aeu, 0x7f22f122u, 0x7ef6e8dau, 0x7ecaefd3u,
    0x7e9f0606u, 0x7e732b6fu, 0x7e476009u, 0x7e1ba3cdu, 0x7deff6b6u, 0x7dc458c1u, 0x7d98c9e6u, 0x7d6d4a22u,
    0x7d41d96eu, 0x7d1677c5u, 0x7ceb2523u, 0x7cbfe183u, 0x7c94acdeu, 0x7c69872fu, 0x7c3e7073u, 0x7c1368a3u,
    0x7be86fbau, 0x7bbd85b3u, 0x7b92aa88u, 0x7b67de36u, 0x7b3d20b6u, 0x7b127204u, 0x7ae7d21au, 0x7abd40f3u,
    0x7a92be8bu, 0x7a684adbu, 0x7a3de5dfu, 0x7a138f92u, 0x79e947efu, 0x79bf0ef1u, 0x7994e492u, 0x796ac8cdu,
    0x7940bb9eu, 0x7916bd00u, 0x78ecccecu, 0x78c2eb5fu, 0x78991854u, 0x786f53c4u, 0x78459dacu, 0x781bf607u,
    0x77f25cceu, 0x77c8d1feu, 0x779f5590u, 0x7775e782u, 0x774c87ccu, 0x7723366bu, 0x76f9f359u, 0x76d0be91u,
    0x76a7980fu, 0x767e7fceu, 0x765575c8u, 0x762c79f9u, 0x76038c5bu, 0x75daaceau, 0x75b1dba2u, 0x7589187cu,
    0x75606374u, 0x7537bc85u, 0x750f23abu, 0x74e698e0u, 0x74be1c20u, 0x7495ad66u, 0x746d4cacu, 0x7444f9efu,
    0x741cb528u, 0x73f47e54u, 0x73cc556du, 0x73a43a70u, 0x737c2d55u, 0x73542e1au, 0x732c3cbau, 0x7304592eu,
    0x72dc8374u, 0x72b4bb85u, 0x728d015du, 0x726554f8u, 0x723db650u, 0x72162561u, 0x71eea226u, 0x71c72c9au,
    0x719fc4b9u, 0x71786a7eu, 0x71511de4u, 0x7129dee6u, 0x7102ad80u, 0x70db89adu, 0x70b47368u, 0x708d6aadu,
    0x70666f76u, 0x703f81c0u, 0x7018a185u, 0x6ff1cec1u, 0x6fcb096fu, 0x6fa4518bu, 0x6f7da710u, 0x6f5709f9u,
    0x6f307a41u, 0x6f09f7e5u, 0x6ee382deu, 0x6ebd1b2au, 0x6e96c0c3u, 0x6e7073a4u, 0x6e4a33c9u, 0x6e24012du,
    0x6dfddbccu, 0x6dd7c3a1u, 0x6db1b8a8u, 0x6d8bbadbu, 0x6d65ca38u, 0x6d3fe6b8u, 0x6d1a1057u, 0x6cf44711u,
    0x6cce8ae1u, 0x6ca8dbc3u, 0x6c8339b2u, 0x6c5da4aau, 0x6c381ca6u, 0x6c12a1a2u, 0x6bed3399u, 0x6bc7d286u,
    0x6ba27e65u, 0x6b7d3733u, 0x6b57fce9u, 0x6b32cf84u, 0x6b0daeffu, 0x6ae89b56u, 0x6ac39485u, 0x6a9e9a86u,
    0x6a79ad56u, 0x6a54ccf0u, 0x6a2ff94fu, 0x6a0b3270u, 0x69e6784du, 0x69c1cae2u, 0x699d2a2cu, 0x69789625u,
    0x69540ec9u, 0x692f9414u, 0x690b2601u, 0x68e6c48cu, 0x68c26fb1u, 0x689e276bu, 0x6879ebb6u, 0x6855bc8eu,
    0x683199edu, 0x680d83d1u, 0x67e97a34u, 0x67c57d12u, 0x67a18c68u, 0x677da82fu, 0x6759d065u, 0x67360505u,
    0x6712460bu, 0x66ee9371u, 0x66caed35u, 0x66a75352u, 0x6683c5c3u, 0x66604485u, 0x663ccf92u, 0x661966e7u,
    0x65f60a7fu, 0x65d2ba57u, 0x65af766au, 0x658c3eb3u, 0x6569132fu, 0x6545f3d9u, 0x6522e0adu, 0x64ffd9a8u,
    0x64dcdec3u, 0x64b9effcu, 0x64970d4fu, 0x647436b6u, 0x64516c2eu, 0x642eadb3u, 0x640bfb41u, 0x63e954d2u,
    0x63c6ba64u, 0x63a42bf2u, 0x6381a978u, 0x635f32f2u, 0x633cc85bu, 0x631a69afu, 0x62f816ebu, 0x62d5d00au,
    0x62b39509u, 0x629165e2u, 0x626f4292u, 0x624d2b15u, 0x622b1f66u, 0x62091f82u, 0x61e72b65u, 0x61c5430au,
    0x61a3666du, 0x6181958bu, 0x615fd05eu, 0x613e16e4u, 0x611c6919u, 0x60fac6f7u, 0x60d9307bu, 0x60b7a5a1u,
    0x60962665u, 0x6074b2c3u, 0x60534ab7u, 0x6031ee3du, 0x60109d51u, 0x5fef57efu, 0x5fce1e12u, 0x5facefb8u,
    0x5f8bccdbu, 0x5f6ab579u, 0x5f49a98cu, 0x5f28a911u, 0x5f07b405u, 0x5ee6ca62u, 0x5ec5ec26u, 0x5ea5194cu,
    0x5e8451d0u, 0x5e6395aeu, 0x5e42e4e3u, 0x5e223f6au, 0x5e01a53fu, 0x5de11660u, 0x5dc092c7u, 0x5da01a70u,
    0x5d7fad59u, 0x5d5f4b7du, 0x5d3ef4d7u, 0x5d1ea965u, 0x5cfe6923u, 0x5cde340bu, 0x5cbe0a1cu, 0x5c9deb50u,
    0x5c7dd7a4u, 0x5c5dcf14u, 0x5c3dd19cu, 0x5c1ddf38u, 0x5bfdf7e5u, 0x5bde1b9fu, 0x5bbe4a61u, 0x5b9e8429u,
    0x5b7ec8f2u, 0x5b5f18b8u, 0x5b3f7377u, 0x5b1fd92du, 0x5b0049d4u, 0x5ae0c56au, 0x5ac14beau, 0x5aa1dd50u,
    0x5a82799au, 0x5a6320c3u, 0x5a43d2c6u, 0x5a248fa2u, 0x5a055751u, 0x59e629d1u, 0x59c7071cu, 0x59a7ef30u,
    0x5988e209u, 0x5969dfa3u, 0x594ae7fbu, 0x592bfb0cu, 0x590d18d3u, 0x58ee414du, 0x58cf7474u, 0x58b0b247u,
    0x5891fac1u, 0x58734ddeu, 0x5854ab9bu, 0x583613f4u, 0x581786e6u, 0x57f9046cu, 0x57da8c83u, 0x57bc1f28u,
    0x579dbc57u, 0x577f640bu, 0x57611642u, 0x5742d2f8u, 0x57249a29u, 0x57066bd2u, 0x56e847efu, 0x56ca2e7cu,
    0x56ac1f75u, 0x568e1ad8u, 0x567020a0u, 0x565230cau, 0x56344b52u, 0x56167035u, 0x55f89f70u, 0x55dad8fdu,
    0x55bd1cdbu, 0x559f6b05u, 0x5581c378u, 0x55642630u, 0x55469329u, 0x55290a61u, 0x550b8bd4u, 0x54ee177du,
    0x54d0ad5au, 0x54b34d67u, 0x5495f7a1u, 0x5478ac03u, 0x545b6a8bu, 0x543e3335u, 0x542105fdu, 0x5403e2e0u,
    0x53e6c9dau, 0x53c9bae9u, 0x53acb607u, 0x538fbb33u, 0x5372ca68u, 0x5355e3a3u, 0x533906e0u, 0x531c341du,
    0x52ff6b55u, 0x52e2ac85u, 0x52c5f7aau, 0x52a94cc0u, 0x528cabc3u, 0x527014b1u, 0x52538786u, 0x5237043fu,
    0x521a8ad7u, 0x51fe1b4cu, 0x51e1b59au, 0x51c559beu, 0x51a907b4u, 0x518cbf7au, 0x5170810bu, 0x51544c64u,
    0x51382182u, 0x511c0061u, 0x50ffe8feu, 0x50e3db56u, 0x50c7d765u, 0x50abdd28u, 0x508fec9cu, 0x507405bdu,
    0x50582888u, 0x503c54f9u, 0x50208b0eu, 0x5004cac2u, 0x4fe91413u, 0x4fcd66fdu, 0x4fb1c37cu, 0x4f96298fu,
    0x4f7a9930u, 0x4f5f125eu, 0x4f439514u, 0x4f28214fu, 0x4f0cb70cu, 0x4ef15648u, 0x4ed5ff00u, 0x4ebab12fu,
    0x4e9f6cd4u, 0x4e8431eau, 0x4e69006eu, 0x4e4dd85du, 0x4e32b9b4u, 0x4e17a470u, 0x4dfc988cu, 0x4de19607u,
    0x4dc69cddu, 0x4dabad0au, 0x4d90c68bu, 0x4d75e95eu, 0x4d5b157eu, 0x4d404ae9u, 0x4d25899cu, 0x4d0ad192u,
    0x4cf022cau, 0x4cd57d3fu, 0x4cbae0efu, 0x4ca04dd6u, 0x4c85c3f1u, 0x4c6b433eu, 0x4c50cbb8u, 0x4c365d5du,
    0x4c1bf829u, 0x4c019c19u, 0x4be7492bu, 0x4bccff5bu, 0x4bb2bea5u, 0x4b988707u, 0x4b7e587eu, 0x4b643305u,
    0x4b4a169cu, 0x4b30033du, 0x4b15f8e6u, 0x4afbf794u, 0x4ae1ff43u, 0x4ac80ff2u, 0x4aae299bu, 0x4a944c3du,
    0x4a7a77d4u, 0x4a60ac5eu, 0x4a46e9d6u, 0x4a2d303bu, 0x4a137f88u, 0x49f9d7bbu, 0x49e038d0u, 0x49c6a2c6u,
    0x49ad1598u, 0x49939143u, 0x497a15c4u, 0x4960a319u, 0x4947393fu, 0x492dd831u, 0x49147feeu, 0x48fb3072u,
    0x48e1e9bau, 0x48c8abc3u, 0x48af768au, 0x48964a0cu, 0x487d2646u, 0x48640b35u, 0x484af8d6u, 0x4831ef26u,
    0x4818ee22u, 0x47fff5c6u, 0x47e70611u, 0x47ce1efeu, 0x47b5408cu, 0x479c6ab6u, 0x47839d7bu, 0x476ad8d6u,
    0x47521cc6u, 0x47396946u, 0x4720be55u, 0x47081beeu, 0x46ef8210u, 0x46d6f0b7u, 0x46be67e0u, 0x46a5e789u,
    0x468d6faeu, 0x4675004cu, 0x465c9961u, 0x46443ae9u, 0x462be4e2u, 0x46139749u, 0x45fb521au, 0x45e31554u,
    0x45cae0f2u, 0x45b2b4f2u, 0x459a9152u, 0x4582760eu, 0x456a6323u, 0x4552588eu, 0x453a564du, 0x45225c5du,
    0x450a6abbu, 0x44f28163u, 0x44daa054u, 0x44c2c78au, 0x44aaf702u, 0x44932eb9u, 0x447b6eadu, 0x4463b6dbu,
    0x444c0740u, 0x44345fd9u, 0x441cc0a3u, 0x4405299cu, 0x43ed9ac0u, 0x43d6140du, 0x43be957fu, 0x43a71f15u,
    0x438fb0cbu, 0x43784a9fu, 0x4360ec8du, 0x43499693u, 0x433248aeu, 0x431b02dbu, 0x4303c518u, 0x42ec8f61u,
    0x42d561b4u, 0x42be3c0eu, 0x42a71e6cu, 0x429008ccu, 0x4278fb2bu, 0x4261f586u, 0x424af7dau, 0x42340224u,
    0x421d1462u, 0x42062e91u, 0x41ef50aeu, 0x41d87ab6u, 0x41c1aca7u, 0x41aae67fu, 0x41942839u, 0x417d71d4u,
    0x4166c34cu, 0x41501ca0u, 0x41397dccu, 0x4122e6cdu, 0x410c57a2u, 0x40f5d046u, 0x40df50b8u, 0x40c8d8f5u,
    0x40b268fau, 0x409c00c4u, 0x4085a051u, 0x406f479eu, 0x4058f6a8u, 0x4042ad6du, 0x402c6be9u, 0x4016321bu,
};

// 2^(-frac / 2^24) for frac in [0, 2^24) in Q1.31: table at 9 fraction bits, then e^-a ~ 1 - a + a^2/2 for the
// remainder a = r ln2 < 0.00136 (relative error < a^3/6 = 4.2e-10, plus Q1.31 rounding).
static inline uint32_t mass_w31(uint32_t frac) {
    const uint32_t r = frac & 0x7fffu;                                         // units of 2^-24
    const uint32_t a = (uint32_t)(((uint64_t)r * 0xB17217F8u) >> 24);          // r ln2 in Q0.32 (< 2^23)
    const uint32_t a2 = (uint32_t)(((uint64_t)a * a) >> 32);                   // a^2 in Q0.32
    const uint32_t poly = 0x80000000u - (a >> 1) + (a2 >> 2);                  // e^-a in Q1.31
    return (uint32_t)(((uint64_t)EXP2_512[frac >> 15] * poly) >> 31);
}

// s * 2^(-y) (y in Q.24) for s < 2^63: rescales a Q.46 mass to a higher reference.
static inline uint64_t rescale_mass(uint64_t s, uint64_t yq) {
    if (yq >= (63ull << 24)) return 0;
    const uint32_t f = exp2_frac_q31((uint32_t)yq & 0xffffffu);  // Q1.31
    const uint64_t hi = (s >> 32) * f, lo = ((s & 0xffffffffull) * f) >> 31;
    return ((hi << 1) + lo) >> (uint32_t)(yq >> 24);
}

// Key of a float at or below q / 2^24 (the skip threshold of the mass window; a few ulps low is harmless).
static inline uint32_t q24_floor_key(int64_t q) {
    const float v = s_mul(s_i642f(q), bitsf(0x33800000u));  // * 2^-24, exact
    const uint32_t k = fkey(fbits(v));
    return k > 4 ? k - 4 : 0;
}

// exp(-d / T) for float d >= 0 given c = log2(e) / T as unsigned Q32.32 (c_hi.c_lo): weight of a candidate d below
// the maximum logit. d >= 256 (very high temperature only) takes the float path.
static inline float exp_weight(float d, uint32_t c_lo, uint32_t c_hi, float temperature) {
    const uint32_t b = fbits(d);
    if (b == 0) return bitsf(0x3f800000u);
    const uint32_t e = b >> 23;
    if (e >= 127 + 8) return exp_nonpos(s_div(bitsf(b ^ 0x80000000u), temperature));
    const uint32_t m = (b & 0x7fffffu) | 0x800000u;
    const int32_t sh = (int32_t)e - 126;
    const uint32_t dq = sh >= 0 ? (m << sh) : (sh > -32 ? (m >> -sh) : 0u);  // d in Q8.24
    const uint64_t y = (uint64_t)dq * c_hi + (((uint64_t)dq * c_lo) >> 32);
    if (y >= ((uint64_t)150 << 24)) return bitsf(0);
    return exp2_neg_q24((uint32_t)y);
}

// Philox4x32-10 (Random123): counter (position, purpose, 0, 0), key (seed_lo, seed_hi).
static inline void philox(uint32_t c[4], uint32_t k0, uint32_t k1) {
    for (int i = 0; i < 10; ++i) {
        uint64_t p0 = (uint64_t)0xD2511F53u * c[0];
        uint64_t p1 = (uint64_t)0xCD9E8D57u * c[2];
        uint32_t n0 = (uint32_t)(p1 >> 32) ^ c[1] ^ k0;
        uint32_t n2 = (uint32_t)(p0 >> 32) ^ c[3] ^ k1;
        c[0] = n0;
        c[1] = (uint32_t)p1;
        c[2] = n2;
        c[3] = (uint32_t)p0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
}
// Uniform double in [0, 1) with 52 random bits.
static inline double uniform(uint32_t seed_lo, uint32_t seed_hi, uint32_t position, uint32_t purpose) {
    uint32_t c[4] = {position, purpose, 0, 0};
    philox(c, seed_lo, seed_hi);
    uint64_t mant = ((uint64_t)c[0] << 20) | (c[1] >> 12);
    return d_sub(bitsd(0x3ff0000000000000ull | mant), bitsd(0x3ff0000000000000ull));
}

struct RowParams {
    uint32_t flags;
    float temperature;
    uint32_t top_k;  // 0: disabled (F_FULL rows only)
    float min_p;
    float one_minus_top_p;
    uint32_t c_lo, c_hi;  // log2(e) / T in Q32.32
};

// vLLM Sampler on the global top-K list (keys descending, ties by ascending id): temperature, min_p, top-k (ties
// kept), top-p (ascending cumulative probability <= 1 - p masked; the largest is always kept), then the final
// softmax. Fills weights[i] = exp(x_i - x_0) for the kept prefix and returns its length.
//
// zfull < 0: a top_k <= 64 row (the candidates hold the top-k plus a tie margin). Otherwise (F_FULL) zfull is the
// row's full-vocabulary mass sum_i exp((l_i - l_0) / T) and the tail = zfull - candidates may be kept: *tail is set
// to the kept tail mass (included in *total; the draw lands in it -> host) or 0 when the tail is masked.
// *host != 0 (R_TIE, R_CUT) when the kept set cannot be resolved from the candidates (the host redraws the token).
static inline uint32_t process(const uint32_t* keys, uint32_t n, const RowParams& prm, double zfull, float* weights,
                               double* total, double* tail, uint32_t* host) {
    *host = 0;
    *tail = bitsd(0);
    if (prm.flags & F_GREEDY) {
        weights[0] = bitsf(0x3f800000u);
        *total = bitsd(0x3ff0000000000000ull);
        return 1;
    }
    const bool full = d_le(bitsd(0), zfull);
    // Tail state for full rows: open = tokens beyond the candidates are still kept; known = their mass is zfull - sum.
    bool open = full, known = full;
    uint32_t m = n;
    const float l0 = bitsf(keyf(keys[0]));
    if (!full || (prm.top_k != 0 && prm.top_k < n)) {
        // vLLM applies top-k to x = l / T (logits.div_(temp)). Division by T > 0 is monotone, so only the tie group
        // at the k-th value needs x: keep every later candidate whose x equals x_{k-1}.
        const uint32_t k = prm.top_k < n ? prm.top_k : n;
        const float theta = s_div(bitsf(keyf(keys[k - 1])), prm.temperature);
        m = k;
        while (m < n && f_ge(s_div(bitsf(keyf(keys[m])), prm.temperature), theta)) ++m;
        if (m == n) *host = R_TIE;  // the tie group may continue beyond the K candidates
        open = known = false;
    } else if (prm.top_k != 0) {
        known = false;  // top_k beyond the candidates: part of the tail is kept, its mass is unknown
    }
    // Unnormalized probabilities exp((l_i - l_0) / T).
    for (uint32_t i = 0; i < m; ++i) {
        weights[i] = exp_weight(s_sub(l0, bitsf(keyf(keys[i]))), prm.c_lo, prm.c_hi, prm.temperature);
    }
    double cand = bitsd(0);
    for (uint32_t i = 0; i < m; ++i) cand = d_add(cand, d_f2d(weights[i]));
    // min_p: probs < max_prob * min_p masked  <=>  weight < min_p.
    if (fbits(prm.min_p) != 0) {
        uint32_t kept = 1;
        while (kept < m && !f_lt(weights[kept], prm.min_p)) ++kept;
        if (kept < m) {
            // Equal logits have equal weights: a cut before the end masks the whole truncated tie group / the tail.
            *host = 0;
            open = known = false;
            m = kept;
            cand = bitsd(0);
            for (uint32_t i = 0; i < m; ++i) cand = d_add(cand, d_f2d(weights[i]));
        } else if (open) {
            known = false;  // tail tokens at or above min_p are kept
        }
    }
    double tmass = bitsd(0);
    if (open && known) {
        tmass = d_sub(zfull, cand);
        if (!d_gt(tmass, bitsd(0))) tmass = bitsd(0);
    }
    // top-p: ascending cumulative probability <= 1 - p masked (largest always kept); cum/z <= limit <=> cum <= limit*z.
    // A kept tail is the bottom of the ascending order: it is wholly masked iff its mass <= limit * z.
    if (fbits(prm.one_minus_top_p) != 0) {
        const double lim = d_f2d(prm.one_minus_top_p);
        // Kept prefix for a kept-tail mass t; 0 when the tail is not wholly masked (the nucleus reaches beyond).
        auto cut = [&](double t) -> uint32_t {
            const double limz = d_mul(lim, d_add(cand, t));
            if (!d_le(t, limz)) return 0;
            double cum = t;
            uint32_t kept = m;
            while (kept > 1) {
                cum = d_add(cum, d_f2d(weights[kept - 1]));
                if (!d_le(cum, limz)) break;
                --kept;
            }
            return kept;
        };
        uint32_t kept;
        if (open && !known) {
            // The kept tail mass (top_k beyond the candidates, or min_p reaching into the tail) lies in [0, zfull -
            // cand] and more of it masks less: the cut is resolved when both bounds mask the tail at the same index.
            double hi = d_sub(zfull, cand);
            if (!d_gt(hi, bitsd(0))) hi = bitsd(0);
            kept = cut(hi);
            if (kept != 0 && kept != cut(bitsd(0))) kept = 0;
        } else {
            kept = cut(tmass);
        }
        if (kept == 0) {
            *host = R_CUT;  // the nucleus reaches beyond the candidates
            return m;
        }
        open = false;
        tmass = bitsd(0);
        if (kept < m) {
            cand = bitsd(0);
            for (uint32_t i = 0; i < kept; ++i) cand = d_add(cand, d_f2d(weights[i]));
        }
        m = kept;
    } else if (open && !known) {
        *host = R_CUT;
        return m;
    }
    *tail = tmass;
    *total = d_add(cand, tmass);
    return m;
}

// Inverse-CDF draw over weights[0..n) (descending-probability order) then the kept tail, with u in [0, 1). Returns n
// when the draw lands in the tail (only possible with tail > 0).
static inline uint32_t draw(const float* weights, uint32_t n, double total, double tail, double u) {
    const double target = d_mul(u, total);
    double cum = bitsd(0);
    const bool has_tail = d_gt(tail, bitsd(0));
    for (uint32_t i = 0; i + 1 < n || (has_tail && i < n); ++i) {
        cum = d_add(cum, d_f2d(weights[i]));
        if (d_gt(cum, target)) return i;
    }
    return has_tail ? n : n - 1;
}

// Inverse-CDF draw over weights[0..n) (descending-probability order) with u in [0, 1).
static inline uint32_t draw(const float* weights, uint32_t n, double total, double u) {
    double target = d_mul(u, total);
    double cum = bitsd(0);
    for (uint32_t i = 0; i + 1 < n; ++i) {
        cum = d_add(cum, d_f2d(weights[i]));
        if (d_gt(cum, target)) return i;
    }
    return n - 1;
}

}  // namespace dsm
