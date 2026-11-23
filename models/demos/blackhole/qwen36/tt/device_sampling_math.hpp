// SPDX-License-Identifier: Apache-2.0
// Arithmetic for exact on-device vLLM sampling (device_sampling.py). Shared by the data-movement kernel and the
// host unit test. On device every float/double operation is an explicit libgcc IEEE soft-float call: the JIT
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
#endif

namespace dsm {

constexpr uint32_t NO_ID = 0xffffffffu;
// Row flags (params word 0).
constexpr uint32_t F_ACTIVE = 1, F_GREEDY = 2, F_PENALTY = 4, F_REP = 8, F_EXTRA = 16, F_NOCOMMIT = 32, F_MONO = 64;
// Result flags.
constexpr uint32_t R_INEXACT = 1;
// RNG purposes (Philox counter word 1).
constexpr uint32_t P_SAMPLE = 1, P_ACCEPT = 2, P_RESIDUAL = 3, P_DRAFT = 4;

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

// 2^(-y) for y = yq / 2^24 in [0, 150), integer arithmetic only (relative error ~1e-9 before the final rounding to
// float32, i.e. at most one ulp off the correctly rounded value).
static inline float exp2_neg_q24(uint32_t yq) {
    const uint32_t n = yq >> 24;
    const uint32_t frac = yq & 0xffffffu;
    const uint32_t k = frac >> 18;
    const uint32_t rq = (frac & 0x3ffffu) << 6;                              // r in Q0.30, r < 1/64
    const uint32_t a = (uint32_t)(((uint64_t)rq * 0x58B90BFCu) >> 30);      // a = r ln2 in Q0.31
    const uint32_t a2 = (uint32_t)(((uint64_t)a * a) >> 31);
    const uint32_t a3 = (uint32_t)(((uint64_t)a2 * a) >> 31);
    const uint32_t a4 = (uint32_t)(((uint64_t)a3 * a) >> 31);
    const uint32_t poly = 0x80000000u - a + (a2 >> 1) - a3 / 6 + a4 / 24;    // e^-a in Q1.31
    const uint32_t mant = (uint32_t)(((uint64_t)EXP2_NEG_TABLE[k] * poly) >> 31);  // 2^-frac in Q1.31, (2^30, 2^31]
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
    uint32_t top_k;
    float min_p;
    float one_minus_top_p;
    uint32_t c_lo, c_hi;  // log2(e) / T in Q32.32
};

// vLLM Sampler on the global top-K list (keys descending, ties by ascending id): temperature, min_p, top-k (ties
// kept), top-p (ascending cumulative probability <= 1 - p masked; the largest is always kept), then the final
// softmax. Fills weights[i] = exp(x_i - x_0) for the kept prefix and returns its length; sets *inexact when the
// top-k tie group may extend beyond the K candidates (the device cannot see the rest of the vocabulary).
static inline uint32_t process(const uint32_t* keys, uint32_t n, const RowParams& prm, float* x, float* weights,
                               double* total, bool* inexact) {
    (void)x;
    *inexact = false;
    if (prm.flags & F_GREEDY) {
        weights[0] = bitsf(0x3f800000u);
        *total = bitsd(0x3ff0000000000000ull);
        return 1;
    }
    const uint32_t k = prm.top_k < n ? prm.top_k : n;
    // vLLM applies top-k to x = l / T (logits.div_(temp)). Division by T > 0 is monotone, so only the tie group at
    // the k-th value needs x: keep every later candidate whose x equals x_{k-1}.
    const float theta = s_div(bitsf(keyf(keys[k - 1])), prm.temperature);
    uint32_t m = k;
    while (m < n && f_ge(s_div(bitsf(keyf(keys[m])), prm.temperature), theta)) ++m;
    if (m == n) *inexact = true;  // the tie group may continue beyond the K candidates
    // Unnormalized probabilities exp((l_i - l_0) / T).
    const float l0 = bitsf(keyf(keys[0]));
    for (uint32_t i = 0; i < m; ++i) {
        weights[i] = exp_weight(s_sub(l0, bitsf(keyf(keys[i]))), prm.c_lo, prm.c_hi, prm.temperature);
    }
    // min_p: probs < max_prob * min_p masked  <=>  weight < min_p.
    if (fbits(prm.min_p) != 0) {
        uint32_t kept = 1;
        while (kept < m && !f_lt(weights[kept], prm.min_p)) ++kept;
        // Equal logits have equal weights: a cut before the end masks the whole truncated tie group.
        if (kept < m) *inexact = false;
        m = kept;
    }
    // top-p: ascending cumulative probability <= 1 - p masked (largest always kept); cum/z <= limit <=> cum <= limit*z.
    double z = bitsd(0);
    for (uint32_t i = 0; i < m; ++i) z = d_add(z, d_f2d(weights[i]));
    const double limz = d_mul(d_f2d(prm.one_minus_top_p), z);
    double cum = bitsd(0);
    uint32_t kept = m;
    while (kept > 1) {
        cum = d_add(cum, d_f2d(weights[kept - 1]));
        if (!d_le(cum, limz)) break;
        --kept;
    }
    double w = z;
    if (kept < m) {
        w = bitsd(0);
        for (uint32_t i = 0; i < kept; ++i) w = d_add(w, d_f2d(weights[i]));
    }
    *total = w;
    return kept;
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
