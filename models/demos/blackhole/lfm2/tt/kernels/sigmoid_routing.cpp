// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 MoE routing on data-movement cores: sigmoid scores, expert-bias top-k selection, normalized weights.
//
//   score_e  = sigmoid(logit_e)                       (fixed point; optionally rounded to BF16 like HF)
//   select   = top_k of score_e + expert_bias_e      (ties: lower expert id)
//   weight_k = score_k / (sum_k score_k + 1e-6)       (norm_topk_prob; rounded once to BF16)
//
// Input logits: FP32 tiles [1, 1, T, E] (E % 32 == 0), optionally rounded to BF16 first (flag bit 0) to mirror the
// HF BF16 router Linear.  Expert bias: one row-major page of E int32 values in Q24 (host-converted from the FP32
// buffer).  Output: one 128 B route record per token, word k = expert_id | (BF16 weight << 16) for k < top_k (ids
// ascending), the format grouped_routing_prefix.cpp consumes (called there with top_k - 1 and a shared slot id that
// never occurs, so all top_k records are routed experts).
//
// The data-movement RISCs have no FPU: everything is integer.  exp() is routing_math.hpp's Q31 table/quadratic
// (relative error < 2e-6), so the scores carry ~1e-6 absolute error, far below BF16 resolution.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

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

// sigmoid(x) in Q62 for an FP32 logit.
inline uint64_t sigmoid_q62(uint32_t bits) {
    const uint32_t e = exp_neg_q31(f32_abs_q16(bits));  // exp(-|x|) in Q31
    const uint64_t inverse = (uint64_t{1} << 62) / (uint64_t{0x80000000u} + e);
    const uint64_t numerator = (bits & 0x80000000u) ? e : uint64_t{0x80000000u};
    return numerator * inverse;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t num_experts = get_compile_time_arg_val(0);
    constexpr uint32_t top_k = get_compile_time_arg_val(1);
    constexpr uint32_t tokens = get_compile_time_arg_val(2);
    constexpr uint32_t flags = get_compile_time_arg_val(3);
    constexpr bool round_logits = flags & 1u;
    constexpr bool round_scores = flags & 2u;
    constexpr auto logits_args = TensorAccessorArgs<4>();
    constexpr auto bias_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto routes_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    static_assert(num_experts % 32 == 0 && top_k >= 1 && top_k <= 31);
    constexpr uint32_t columns = num_experts / 32;
    constexpr uint32_t tile_bytes = 4096;  // FP32 32x32
    // Q24 of the 1e-6 in the normalization denominator.
    constexpr uint32_t eps_q24 = 17;

    const auto logits = TensorAccessor(logits_args, get_arg_val<uint32_t>(0), tile_bytes);
    const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(1), num_experts * 4);
    const auto routes = TensorAccessor(routes_args, get_arg_val<uint32_t>(2), 128);
    const uint32_t worker = get_arg_val<uint32_t>(3);
    const uint32_t workers = get_arg_val<uint32_t>(4);

    const uint32_t scratch = get_write_ptr(0);
    const uint32_t bias_addr = scratch + columns * tile_bytes;
    const uint32_t record_addr = bias_addr + num_experts * 4;
    auto* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    auto* bias_q24 = reinterpret_cast<volatile tt_l1_ptr int32_t*>(bias_addr);
    auto* record = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(record_addr);

    noc_async_read(get_noc_addr(0, bias), bias_addr, num_experts * 4);
    noc_async_read_barrier();
    for (uint32_t i = 0; i < 32; ++i) {
        record[i] = 0;
    }

    uint32_t scores[num_experts];
    for (uint32_t tile_row = worker; tile_row < (tokens + 31) / 32; tile_row += workers) {
        const uint32_t rows = tokens - tile_row * 32 < 32 ? tokens - tile_row * 32 : 32;
        for (uint32_t c = 0; c < columns; ++c) {
            noc_async_read(get_noc_addr(tile_row * columns + c, logits), scratch + c * tile_bytes, tile_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t r = 0; r < rows; ++r) {
            // Row r of each column tile: faces (r / 16) * 2 + h, row r % 16 (u32 units: 1024 per tile, 256 per face).
            uint32_t best_id[top_k];
            int32_t best_key[top_k];
            uint32_t count = 0;
            for (uint32_t c = 0; c < columns; ++c) {
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t src = c * 1024 + (r / 16) * 512 + h * 256 + (r % 16) * 16;
                    for (uint32_t j = 0; j < 16; ++j) {
                        const uint32_t e = c * 32 + h * 16 + j;
                        uint32_t bits = tile[src + j];
                        if constexpr (round_logits) {
                            bits = f32_round_bf16(bits);
                        }
                        const uint64_t q62 = sigmoid_q62(bits);
                        uint32_t s;
                        if constexpr (round_scores) {
                            s = bf16_to_q24(q62_to_bf16(q62));
                        } else {
                            s = static_cast<uint32_t>((q62 + (uint64_t{1} << 37)) >> 38);
                        }
                        scores[e] = s;
                        const int32_t key = static_cast<int32_t>(s) + bias_q24[e];
                        // Descending insertion; equal keys keep the lower id first.
                        if (count == top_k && key <= best_key[top_k - 1]) {
                            continue;
                        }
                        uint32_t pos = count < top_k ? count++ : top_k - 1;
                        while (pos > 0 && best_key[pos - 1] < key) {
                            best_key[pos] = best_key[pos - 1];
                            best_id[pos] = best_id[pos - 1];
                            --pos;
                        }
                        best_key[pos] = key;
                        best_id[pos] = e;
                    }
                }
            }
            // Ascending ids.
            for (uint32_t k = 1; k < top_k; ++k) {
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
                const uint16_t w = q62_to_bf16(static_cast<uint64_t>(scores[best_id[k]]) * reciprocal);
                record[k] = best_id[k] | (static_cast<uint32_t>(w) << 16);
            }
            noc_async_write(record_addr, get_noc_addr(tile_row * 32 + r, routes), 128);
            noc_async_write_barrier();
        }
    }
}
