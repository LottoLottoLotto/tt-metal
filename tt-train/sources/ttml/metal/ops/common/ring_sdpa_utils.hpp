// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>
#include <utility>

#include "metal/common/const_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops {

// Causal load balancing of the ring ops: which tensors each device computes on at a ring step.
// - None:   every executing device uses the single set of Q/K/V/output tensors.
// - ZigZag: load-balanced causal layout. The sequence is cut into 2N global chunks and
//           device d holds global chunks [d | 2N-1-d], so its local Q/K/V are two chunks:
//           local chunk 0 (global chunk d for Q, global chunk src for the K/V it holds) and
//           local chunk 1 (global chunk 2N-1-d for Q, 2N-1-src for K/V). The ring op computes
//           the one same-chunk block that is visible at this step: (Q0, K0, V0) when src < d,
//           (Q1, K1, V1) when src > d, both unmasked. The cross block (Q1, K0) is always
//           fully visible and is computed by a separate mask-None call; (Q0, K1) is never
//           visible; the src == d step is a plain causal call over the full local sequence
//           (local order is monotone).
enum class RingLoadBalancing : uint8_t { None = 0, ZigZag = 1 };

// Source device of the K/V a device holds at a ring step:
// - Backward ring direction: src = (d + s) mod ring_size
// - Forward ring direction:  src = (d - s) mod ring_size
inline uint32_t get_ring_source_device(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    ttnn_fixed::distributed::RingShiftDirection ring_direction) {
    if (ring_direction == ttnn_fixed::distributed::RingShiftDirection::Backward) {
        return (device_ring_id + step) % ring_size;
    }
    // Reduce step mod ring_size before subtracting so the uint32 expression cannot
    // underflow: plain (d - s + ring_size) wraps mod 2^32 when s > d, which is only
    // correct for power-of-2 ring sizes.
    return (device_ring_id + ring_size - (step % ring_size)) % ring_size;
}

// Determine if a device should execute at this ring step and which mask type to use.
// Shared by the ring_sdpa_fw and ring_sdpa_bw program factories, which must agree on
// the schedule: the backward relies on skipping exactly the (device, step) pairs the
// forward skipped.
// Returns: (should_execute, mask_type_to_use)
//
// Causal logic on the source chunk (contiguous sharding):
// - src == d: diagonal chunk, apply causal mask
// - src < d:  earlier chunk, full attention (no mask)
// - src > d:  later chunk, fully masked, skip
inline std::pair<bool, AttentionMaskType> get_device_execution_info(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    AttentionMaskType mask_type,
    ttnn_fixed::distributed::RingShiftDirection ring_direction) {
    if (mask_type != AttentionMaskType::Causal) {
        // Non-causal: all devices execute with full attention
        return {true, AttentionMaskType::None};
    }

    const uint32_t src_device = get_ring_source_device(device_ring_id, step, ring_size, ring_direction);

    if (src_device == device_ring_id) {
        return {true, AttentionMaskType::Causal};  // Diagonal: use causal mask
    } else if (src_device < device_ring_id) {
        return {true, AttentionMaskType::None};  // Earlier: full attention (no mask)
    } else {
        return {false, AttentionMaskType::None};  // Later: skip
    }
}

// Work of one device at one ring step: whether it runs, with which mask, on which chunk.
struct RingStepWork {
    bool execute = false;
    AttentionMaskType mask_type = AttentionMaskType::None;
    uint32_t chunk = 0;  // 0: primary tensors, 1: the op's second-chunk tensors

    bool operator==(const RingStepWork&) const = default;
};

// Schedule used by the ring program factories. RingLoadBalancing::None is exactly
// get_device_execution_info on chunk 0. RingLoadBalancing::ZigZag (causal only) returns
// the unmasked same-chunk block described at RingLoadBalancing: chunk 0 when src < d,
// chunk 1 when src > d, and no work when src == d.
inline RingStepWork get_ring_step_work(
    uint32_t device_ring_id,
    uint32_t step,
    uint32_t ring_size,
    AttentionMaskType mask_type,
    ttnn_fixed::distributed::RingShiftDirection ring_direction,
    RingLoadBalancing load_balancing) {
    if (load_balancing == RingLoadBalancing::None) {
        const auto [execute, effective_mask] =
            get_device_execution_info(device_ring_id, step, ring_size, mask_type, ring_direction);
        return {.execute = execute, .mask_type = effective_mask, .chunk = 0U};
    }

    TT_FATAL(
        mask_type == AttentionMaskType::Causal,
        "Zig-zag load balancing is only defined for causal attention, got mask type {}",
        static_cast<int>(mask_type));
    const uint32_t src_device = get_ring_source_device(device_ring_id, step, ring_size, ring_direction);
    if (src_device == device_ring_id) {
        return {.execute = false, .mask_type = AttentionMaskType::None, .chunk = 0U};
    }
    return {
        .execute = true,
        .mask_type = AttentionMaskType::None,
        .chunk = src_device < device_ring_id ? 0U : 1U,
    };
}

// Common attribute validation for the ring SDPA device operations. The ring factories
// dispatch straight to the SDPA program factories, so the SDPA device operations'
// validation never runs for the ring path — these checks are its replacement.
template <typename RingAttrs>
inline void validate_ring_attributes(const RingAttrs& attrs, const ttnn::Tensor& query) {
    TT_FATAL(query.device() != nullptr, "Query tensor must be on a mesh device");
    TT_FATAL(attrs.ring_size > 0, "Ring size must be > 0");
    TT_FATAL(attrs.step < attrs.ring_size, "Step {} must be < ring_size {}", attrs.step, attrs.ring_size);
    // The ring path has no way to feed a mask tensor to the underlying SDPA kernels, so
    // Arbitrary would silently degrade to full attention.
    TT_FATAL(
        attrs.mask_type != AttentionMaskType::Arbitrary,
        "Ring SDPA does not support Arbitrary attention masks, use None or Causal");

    const auto mesh_shape = query.device()->shape();
    TT_FATAL(
        attrs.ring_axis < mesh_shape.dims(),
        "Ring axis {} must be < mesh dimensions {}",
        attrs.ring_axis,
        mesh_shape.dims());
    TT_FATAL(
        attrs.ring_size == mesh_shape[attrs.ring_axis],
        "Ring size {} must match mesh extent {} along ring axis {}",
        attrs.ring_size,
        mesh_shape[attrs.ring_axis],
        attrs.ring_axis);
}

// Load-balancing contract shared by the ring ops: the second chunk is supplied exactly
// when zig-zag load balancing is requested, and zig-zag is only defined for causal
// steps other than the diagonal step (step 0), which the caller runs as one full causal
// call over the whole local sequence.
template <typename RingAttrs>
inline void validate_ring_load_balancing(const RingAttrs& attrs, bool has_second_chunk) {
    if (attrs.load_balancing == RingLoadBalancing::None) {
        TT_FATAL(!has_second_chunk, "Second-chunk tensors were given but load balancing is None");
        return;
    }
    TT_FATAL(has_second_chunk, "Zig-zag load balancing requires the second-chunk tensors");
    TT_FATAL(
        attrs.mask_type == AttentionMaskType::Causal,
        "Zig-zag load balancing requires a causal mask, got mask type {}",
        static_cast<int>(attrs.mask_type));
    TT_FATAL(
        attrs.step != 0U,
        "Zig-zag load balancing is not defined for step 0 (every device holds its own K/V); run step 0 as one "
        "causal call over the full local sequence");
}

// With zig-zag load balancing the devices that compute chunk 1 at a step never write the
// chunk-0 outputs, so a chunk-0 output the op allocated itself would hold uninitialized memory
// on those devices. The caller must supply (and pre-fill) every chunk-0 output, exactly as it
// supplies the chunk-1 outputs.
template <typename RingAttrs>
inline void validate_chunk0_output_given(
    const RingAttrs& attrs, const std::optional<ttnn::Tensor>& preallocated, std::string_view name) {
    TT_FATAL(
        attrs.load_balancing == RingLoadBalancing::None || preallocated.has_value(),
        "Zig-zag load balancing requires a preallocated chunk-0 '{}': devices that compute chunk 1 at this step "
        "never write it",
        name);
}

// Every second-chunk tensor must have exactly the tensor spec (logical and padded shape, dtype,
// layout, memory config) and device of its first-chunk counterpart: the two chunks are halves of
// one local sequence, each device's program is built from whichever chunk it computes, and
// buffers sized from chunk 0 (the backward u_scaler) serve both chunks.
inline void validate_same_spec(const ttnn::Tensor& chunk1, const ttnn::Tensor& chunk0, std::string_view name) {
    TT_FATAL(chunk1.device() == chunk0.device(), "Second-chunk tensor '{}' must be on the same mesh device", name);
    TT_FATAL(
        chunk1.tensor_spec() == chunk0.tensor_spec(),
        "Second-chunk tensor '{}' spec {} must match its first-chunk counterpart's spec {}",
        name,
        chunk1.tensor_spec(),
        chunk0.tensor_spec());
}

// Per-tensor device contract of the reused SDPA kernels: on device, allocated,
// co-located with the query, rank 4, TILE layout, expected dtype, interleaved.
inline void validate_sdpa_tensor(
    const ttnn::Tensor& tensor,
    std::string_view name,
    const ttnn::Tensor& query,
    ttnn::DataType required_dtype = ttnn::DataType::BFLOAT16) {
    check_device_tensor(tensor, "RingSDPA", name, {.dtypes = {required_dtype}});
    TT_FATAL(tensor.device() == query.device(), "Tensor '{}' must be on the same mesh device as the query", name);
    TT_FATAL(
        tensor.logical_shape().rank() == 4U,
        "Tensor '{}' must have rank 4, got rank {}",
        name,
        tensor.logical_shape().rank());
}

// Q/K/V contract shared by the ring forward and backward operations: the per-tensor
// device contract plus the shape consistency the reused SDPA kernels assume.
inline void validate_ring_qkv(const ttnn::Tensor& query, const ttnn::Tensor& key, const ttnn::Tensor& value) {
    validate_sdpa_tensor(query, "Query", query);
    validate_sdpa_tensor(key, "Key", query);
    validate_sdpa_tensor(value, "Value", query);

    const auto [qB, qH, qS, qE] = query.logical_shape().to_array_4D();
    const auto [kB, kH, kS, kE] = key.logical_shape().to_array_4D();
    const auto [vB, vH, vS, vE] = value.logical_shape().to_array_4D();

    TT_FATAL(qH > 0 && kH > 0, "Number of heads must be greater than zero. Got q_heads={}, kv_heads={}", qH, kH);
    TT_FATAL(
        qH % kH == 0,
        "Number of query heads ({}) must be divisible by number of key/value heads ({}) for grouped attention",
        qH,
        kH);
    TT_FATAL(kH == vH, "Key and Value must have the same number of heads. Got key_heads={}, value_heads={}", kH, vH);
    TT_FATAL(
        qB == kB && qB == vB && qS == kS && qS == vS && qE == kE,
        "Query/Key/Value must agree on batch, sequence length, and Q/K inner dim. Got Query={}, Key={}, Value={}",
        query.logical_shape(),
        key.logical_shape(),
        value.logical_shape());
}

// Contract for tensors shaped like the attention output — grad_output, attn_output,
// and preallocated forward output: query B/H/S with the value's inner dim.
inline void validate_output_like_tensor(
    const ttnn::Tensor& tensor, std::string_view name, const ttnn::Tensor& query, const ttnn::Tensor& value) {
    validate_sdpa_tensor(tensor, name, query);
    const auto query_dims = query.logical_shape().to_array_4D();
    const auto value_inner_dim = value.logical_shape().to_array_4D()[3];
    const ttnn::Shape expected_shape{query_dims[0], query_dims[1], query_dims[2], value_inner_dim};
    TT_FATAL(
        tensor.logical_shape() == expected_shape,
        "Tensor '{}' shape {} must be {} (query B/H/S with value inner dim)",
        name,
        tensor.logical_shape(),
        expected_shape);
}

// Contract for preallocated gradient tensors: same device contract and shape as the
// tensor they are the gradient of.
inline void validate_grad_like_tensor(
    const ttnn::Tensor& tensor, std::string_view name, const ttnn::Tensor& reference, const ttnn::Tensor& query) {
    validate_sdpa_tensor(tensor, name, query);
    TT_FATAL(
        tensor.logical_shape() == reference.logical_shape(),
        "Tensor '{}' shape {} must match the shape {} of the tensor it is the gradient of",
        name,
        tensor.logical_shape(),
        reference.logical_shape());
}

// The SDPA kernels read the logsumexp from column 0 of a (B, H, S, TILE_WIDTH) FP32
// tile-layout tensor; anything else is silently misinterpreted on device.
inline void validate_intermediates_tensor(const ttnn::Tensor& intermediates, const ttnn::Tensor& query) {
    TT_FATAL(
        intermediates.dtype() == ttnn::DataType::FLOAT32,
        "Intermediates must be FLOAT32, got {}",
        intermediates.dtype());
    TT_FATAL(
        intermediates.layout() == ttnn::Layout::TILE,
        "Intermediates must be TILE layout, got {}",
        intermediates.layout());
    const auto [batch, heads, seq_len, dim] = query.logical_shape().to_array_4D();
    const ttnn::Shape expected_shape{batch, heads, seq_len, tt::constants::TILE_WIDTH};
    TT_FATAL(
        intermediates.logical_shape() == expected_shape,
        "Intermediates shape {} must be {} (one logsumexp tile column per query row)",
        intermediates.logical_shape(),
        expected_shape);
}

}  // namespace ttml::metal::ops
