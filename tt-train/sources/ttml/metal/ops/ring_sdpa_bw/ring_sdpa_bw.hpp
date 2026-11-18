// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_sdpa_bw_kv_device_operation.hpp"
#include "device/ring_sdpa_bw_q_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

using RingDirection = ttml::metal::ops::ring_sdpa_bw::RingDirection;

// Second-chunk tensors for RingChunkSelection::ZigZag (see metal/ops/common/ring_sdpa_utils.hpp).
// Devices assigned chunk 1 at a step read the chunk-1 inputs and write grad_query, grad_key and
// grad_value in place and leave chunk 0's untouched, so the chunk-0 preallocated_grad_query/key/
// value are required; every chunk-1 tensor must have its chunk-0 counterpart's spec. The returned
// gradients always refer to chunk 0.
struct RingSDPABwChunk {
    ttnn::Tensor grad_output;
    ttnn::Tensor attn_output;
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
    ttnn::Tensor intermediates;
    ttnn::Tensor grad_query;
    ttnn::Tensor grad_key;
    ttnn::Tensor grad_value;
};

// Returns [grad_Q, grad_K, grad_V] of chunk 0.
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_sdpa_bw(
    const ttnn::Tensor& grad_output,    // Gradient w.r.t. output
    const ttnn::Tensor& attn_output,    // sdpa forward output (needed for gradients)
    const ttnn::Tensor& query,          // input Q (needed for gradients)
    const ttnn::Tensor& key,            // input K (needed for gradients)
    const ttnn::Tensor& value,          // input V (needed for gradients)
    const ttnn::Tensor& intermediates,  // From forward pass (FP32 logsumexp per row)
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type = AttentionMaskType::None,
    RingDirection ring_direction = RingDirection::Backward,
    const std::optional<ttnn::Tensor>& preallocated_grad_query = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_key = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_value = std::nullopt,
    ops::RingChunkSelection chunk_selection = ops::RingChunkSelection::None,
    const std::optional<RingSDPABwChunk>& chunk1 = std::nullopt);

}  // namespace ttml::metal
