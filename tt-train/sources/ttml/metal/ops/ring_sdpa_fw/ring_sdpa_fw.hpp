// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_sdpa_fw_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Returns [output, intermediates] of chunk 0. With RingLoadBalancing::ZigZag the devices
// assigned chunk 1 at this step write into chunk1->output / chunk1->intermediates instead
// and leave chunk 0's untouched, so preallocated_output and preallocated_intermediates are
// required and every chunk-1 tensor must have its chunk-0 counterpart's spec
// (see RingLoadBalancing in metal/ops/common/ring_sdpa_utils.hpp).
std::tuple<ttnn::Tensor, ttnn::Tensor> ring_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type = AttentionMaskType::None,
    ops::ring_sdpa_fw::RingDirection ring_direction = ops::ring_sdpa_fw::RingDirection::Backward,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_intermediates = std::nullopt,
    ops::RingLoadBalancing load_balancing = ops::RingLoadBalancing::None,
    const std::optional<ops::ring_sdpa_fw::RingSDPAFwChunk>& chunk1 = std::nullopt);

}  // namespace ttml::metal
