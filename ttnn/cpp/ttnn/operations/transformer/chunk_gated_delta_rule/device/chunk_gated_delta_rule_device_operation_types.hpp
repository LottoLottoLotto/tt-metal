// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Types for the standalone chunk_gated_delta_rule ttnn op.
// Algorithm derived solely from flash-linear-attention `naive_chunk_gated_delta_rule`.

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

struct ChunkGatedDeltaRuleParams {
    uint32_t BH;          // B * HV   (one Tensix core per head)
    uint32_t num_chunks;  // NC = ceil(T_padded / C)
    uint32_t chunk_size;  // C (multiple of 32)
    uint32_t key_dim;     // K (multiple of 32)
    uint32_t val_dim;     // V (multiple of 32)
    bool has_initial_state;
    bool output_final_state;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

// All device-op input tensors are fp32, TILE layout, DRAM interleaved.
// Head-split + scale + padding are done host-side; the kernel does all the math.
struct ChunkGatedDeltaRuleInputs {
    Tensor q;                             // [BH, NC, C, K]  (already scaled by `scale`)
    Tensor k;                             // [BH, NC, C, K]
    Tensor v;                             // [BH, NC, C, V]
    Tensor g;                             // [BH, NC, C, 1]  (log-space decay, column form)
    Tensor beta;                          // [BH, NC, C, 1]  (column form)
    Tensor eye_c;                         // [1, 1, C, C]    identity (constant)
    Tensor tril_c;                        // [1, 1, C, C]    lower-tri ones incl diag (constant)
    Tensor ones_c;                        // [1, 1, C, C]    all ones (constant, broadcasts)
    std::optional<Tensor> initial_state;  // [BH, K, V] or absent (zeros)
};

struct RecurrentGatedDeltaRuleParams {
    uint32_t BH;
    uint32_t key_dim;
    uint32_t val_dim;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    bool fuse_decode_frontend;
    float q_scale;
};

struct RecurrentGatedDeltaRuleInputs {
    Tensor q;      // [BH, 1, K], normalized and scaled
    Tensor k;      // [BH, 1, K], normalized
    Tensor v;      // [BH, 1, V]
    Tensor g;      // [BH, 1], log-space decay
    Tensor beta;   // [BH, 1]
    std::optional<Tensor> decay_bias;   // [BH, 1, 1], required by fused front end
    std::optional<Tensor> decay_scale;  // [BH, 1, 1], -exp(A_log), required by fused front end
    std::optional<Tensor> norm_ones;     // [1, 32, 32], required by fused front end
    Tensor state;  // [BH, K, V]
};

struct VerifyTwoGatedDeltaRuleParams {
    uint32_t BH;       // Exactly 48 heads, one per Tensix core
    uint32_t key_dim;  // K
    uint32_t val_dim;  // V
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    float q_scale;
};

struct VerifyTwoGatedDeltaRuleInputs {
    Tensor q;            // [2, 1, BH, K], raw, head dimension padded to a tile
    Tensor k;            // [2, 1, BH, K], raw
    Tensor v;            // [2, 1, BH, V]
    Tensor a;            // [1, 1, 2, BH], raw decay gate
    Tensor b;            // [1, 1, 2, BH], raw update gate
    Tensor decay_bias;   // [BH, 1, 1]
    Tensor decay_scale;  // [BH, 1, 1], precomputed -exp(A_log)
    Tensor norm_ones;    // one BF16 all-ones tile
    Tensor state;        // [BH,K,V] or [1,BH,K,V], row-major L1 height-sharded
};

}  // namespace ttnn::prim
