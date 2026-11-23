// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_device_operation.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

ChunkGatedDeltaRuleDeviceOperation::program_factory_t ChunkGatedDeltaRuleDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChunkGatedDeltaRuleProgramFactory{};
}

void ChunkGatedDeltaRuleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    auto check = [](const Tensor& t, const char* name, DataType dt) {
        TT_FATAL(t.layout() == Layout::TILE, "chunk_gated_delta_rule: {} must be TILE layout", name);
        TT_FATAL(t.dtype() == dt, "chunk_gated_delta_rule: {} has wrong dtype", name);
        TT_FATAL(t.buffer() != nullptr, "chunk_gated_delta_rule: {} must be on device", name);
    };
    // GPU-style mixed precision: q/k/v are bf16; gate/decay, state and mask constants stay fp32.
    check(in.q, "q", DataType::BFLOAT16);
    check(in.k, "k", DataType::BFLOAT16);
    check(in.v, "v", DataType::BFLOAT16);
    check(in.g, "g", DataType::FLOAT32);
    check(in.beta, "beta", DataType::FLOAT32);
    check(in.eye_c, "eye_c", DataType::FLOAT32);
    check(in.tril_c, "tril_c", DataType::FLOAT32);
    check(in.ones_c, "ones_c", DataType::FLOAT32);
    if (in.initial_state.has_value()) {
        check(*in.initial_state, "initial_state", DataType::FLOAT32);
    }
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
}

ChunkGatedDeltaRuleDeviceOperation::spec_return_value_t ChunkGatedDeltaRuleDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    // o is bf16 (matches cb_out); the recurrent final state stays fp32.
    const auto o_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto s_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    ttnn::Shape o_shape({attrs.BH, attrs.num_chunks, attrs.chunk_size, attrs.val_dim});
    ttnn::Shape s_shape({attrs.BH, attrs.key_dim, attrs.val_dim});
    return {tt::tt_metal::TensorSpec(o_shape, o_layout), tt::tt_metal::TensorSpec(s_shape, s_layout)};
}

ChunkGatedDeltaRuleDeviceOperation::tensor_return_value_t ChunkGatedDeltaRuleDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.q.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

std::vector<Tensor> chunk_gated_delta_rule(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye_c,
    const Tensor& tril_c,
    const Tensor& ones_c,
    const std::optional<Tensor>& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ChunkGatedDeltaRuleDeviceOperation;

    const auto& q_shape = q.logical_shape();  // [BH, NC, C, K]
    const auto& v_shape = v.logical_shape();  // [BH, NC, C, V]

    auto attrs = OperationType::operation_attributes_t{
        .BH = q_shape[0],
        .num_chunks = q_shape[1],
        .chunk_size = chunk_size,
        .key_dim = q_shape[3],
        .val_dim = v_shape[3],
        .has_initial_state = initial_state.has_value(),
        .output_final_state = output_final_state,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .eye_c = eye_c,
        .tril_c = tril_c,
        .ones_c = ones_c,
        .initial_state = initial_state,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

RecurrentGatedDeltaRuleDeviceOperation::program_factory_t
RecurrentGatedDeltaRuleDeviceOperation::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return RecurrentGatedDeltaRuleProgramFactory{};
}

void RecurrentGatedDeltaRuleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto check = [](const Tensor& t, const char* name) {
        TT_FATAL(t.layout() == Layout::TILE, "recurrent_gated_delta_rule: {} must use TILE layout", name);
        TT_FATAL(t.dtype() == DataType::BFLOAT16, "recurrent_gated_delta_rule: {} must use bfloat16", name);
        TT_FATAL(t.buffer() != nullptr, "recurrent_gated_delta_rule: {} must be on device", name);
        TT_FATAL(!t.is_sharded(), "recurrent_gated_delta_rule: {} must be interleaved", name);
    };
    check(in.q, "q");
    check(in.k, "k");
    check(in.v, "v");
    check(in.g, "g");
    check(in.beta, "beta");
    auto check_head_shape = [&](const Tensor& tensor, uint32_t dim, const char* name) {
        TT_FATAL(
            tensor.logical_shape() == ttnn::Shape({attrs.BH, 1, dim}) &&
                tensor.padded_shape() == ttnn::Shape({attrs.BH, TILE_HEIGHT, ((dim + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH}),
            "recurrent_gated_delta_rule: {} must have one tiled row per head [BH,1,D]",
            name);
        TT_FATAL(tensor.device() == in.q.device(), "recurrent_gated_delta_rule: {} is on a different device", name);
    };
    if (attrs.gather_head_rows) {
        // Frontend layout: heads fill tile rows of q/k/v and tile columns of raw a/b.
        TT_FATAL(attrs.fuse_decode_frontend, "gathered head rows require the fused decode front end (raw a/b)");
        auto check_rows = [&](const Tensor& tensor, uint32_t dim, const char* name) {
            TT_FATAL(
                tensor.logical_shape() == ttnn::Shape({1, 1, attrs.BH, dim}),
                "recurrent_gated_delta_rule: gathered {} must be [1,1,BH,D]",
                name);
            TT_FATAL(tensor.device() == in.q.device(), "recurrent_gated_delta_rule: {} is on a different device", name);
        };
        auto check_scalars = [&](const Tensor& tensor, const char* name) {
            TT_FATAL(
                tensor.logical_shape() == ttnn::Shape({1, 1, attrs.BH}),
                "recurrent_gated_delta_rule: gathered {} must be [1,1,BH]",
                name);
            TT_FATAL(tensor.device() == in.q.device(), "recurrent_gated_delta_rule: {} is on a different device", name);
        };
        check_rows(in.q, attrs.key_dim, "q");
        check_rows(in.k, attrs.key_dim, "k");
        check_rows(in.v, attrs.val_dim, "v");
        check_scalars(in.g, "a");
        check_scalars(in.beta, "b");
    } else {
        check_head_shape(in.q, attrs.key_dim, "q");
        check_head_shape(in.k, attrs.key_dim, "k");
        check_head_shape(in.v, attrs.val_dim, "v");
        check_head_shape(in.g, 1, "g");
        check_head_shape(in.beta, 1, "beta");
    }
    TT_FATAL(
        in.state.layout() == Layout::ROW_MAJOR,
        "recurrent_gated_delta_rule: persistent state must use ROW_MAJOR layout");
    TT_FATAL(
        in.state.dtype() == DataType::BFLOAT16,
        "recurrent_gated_delta_rule: persistent state must use bfloat16");
    TT_FATAL(in.state.buffer() != nullptr, "recurrent_gated_delta_rule: state must be on device");
    TT_FATAL(
        in.state.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "recurrent_gated_delta_rule: persistent state must be height-sharded");
    TT_FATAL(
        in.state.memory_config().buffer_type() == BufferType::L1,
        "recurrent_gated_delta_rule: persistent state must reside in L1");
    if (attrs.fuse_decode_frontend) {
        TT_FATAL(
            in.decay_bias.has_value() && in.decay_scale.has_value() && in.norm_ones.has_value(),
            "fused recurrent front end requires decay tensors and a normalization ones tile");
        check(*in.decay_bias, "decay_bias");
        check(*in.decay_scale, "decay_scale");
        check(*in.norm_ones, "norm_ones");
        check_head_shape(*in.decay_bias, 1, "decay_bias");
        check_head_shape(*in.decay_scale, 1, "decay_scale");
        TT_FATAL(
            in.norm_ones->logical_volume() == TILE_HEIGHT * TILE_WIDTH &&
                in.norm_ones->physical_volume() == TILE_HEIGHT * TILE_WIDTH &&
                in.norm_ones->device() == in.q.device(),
            "recurrent_gated_delta_rule: norm_ones must be one shared normalization tile on the same device");
    }
    TT_FATAL(attrs.key_dim > 0 && attrs.key_dim % TILE_WIDTH == 0, "key_dim must be positive and tile aligned");
    TT_FATAL(attrs.val_dim > 0 && attrs.val_dim % TILE_WIDTH == 0, "val_dim must be positive and tile aligned");
    TT_FATAL(
        in.state.logical_shape() == ttnn::Shape({attrs.BH, attrs.key_dim, attrs.val_dim}) &&
            in.state.device() == in.q.device(),
        "recurrent_gated_delta_rule: state must be [BH,K,V] on the same device");
    const auto& shard = in.state.shard_spec().value();
    TT_FATAL(
        shard.orientation == ShardOrientation::ROW_MAJOR && shard.shape[1] == attrs.val_dim &&
            shard.shape[0] * shard.grid.num_cores() >= attrs.BH * attrs.key_dim,
        "recurrent_gated_delta_rule: state shards must cover all rows in row-major order");
}

RecurrentGatedDeltaRuleDeviceOperation::spec_return_value_t
RecurrentGatedDeltaRuleDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    return {
        tt::tt_metal::TensorSpec(ttnn::Shape({attrs.BH, 1, attrs.val_dim}), layout),
        in.state.tensor_spec(),
    };
}

RecurrentGatedDeltaRuleDeviceOperation::tensor_return_value_t
RecurrentGatedDeltaRuleDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {create_device_tensor(specs[0], in.q.device()), in.state};
}

std::vector<Tensor> recurrent_gated_delta_rule(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    bool fuse_decode_frontend,
    const std::optional<Tensor>& decay_bias,
    const std::optional<Tensor>& decay_scale,
    const std::optional<Tensor>& norm_ones,
    const Tensor& state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool gather_head_rows) {
    using OperationType = RecurrentGatedDeltaRuleDeviceOperation;
    const auto& q_shape = q.logical_shape();
    const auto& v_shape = v.logical_shape();
    const uint32_t rank = gather_head_rows ? 4 : 3;
    TT_FATAL(
        q_shape.rank() == rank && k.logical_shape().rank() == rank && v_shape.rank() == rank,
        "recurrent_gated_delta_rule: q/k/v must be head-major [BH,1,D] or gathered [1,1,BH,D]");
    auto attrs = OperationType::operation_attributes_t{
        .BH = gather_head_rows ? q_shape[2] : q_shape[0],
        .key_dim = q_shape[-1],
        .val_dim = v_shape[-1],
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
        .fuse_decode_frontend = fuse_decode_frontend,
        .q_scale = 1.0f / std::sqrt(static_cast<float>(q_shape[-1])),
        .gather_head_rows = gather_head_rows,
    };
    auto inputs = OperationType::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .decay_bias = decay_bias,
        .decay_scale = decay_scale,
        .norm_ones = norm_ones,
        .state = state,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, inputs);
}

VerifyTwoGatedDeltaRuleDeviceOperation::program_factory_t
VerifyTwoGatedDeltaRuleDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return VerifyTwoGatedDeltaRuleProgramFactory{};
}

void VerifyTwoGatedDeltaRuleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto check_bf16_tile = [](const Tensor& tensor, const char* name) {
        TT_FATAL(
            tensor.layout() == Layout::TILE,
            "verify_two_gated_delta_rule: {} must use TILE layout",
            name);
        TT_FATAL(
            tensor.dtype() == DataType::BFLOAT16,
            "verify_two_gated_delta_rule: {} must use bfloat16",
            name);
        TT_FATAL(
            tensor.buffer() != nullptr,
            "verify_two_gated_delta_rule: {} must be on device",
            name);
        TT_FATAL(!tensor.is_sharded(), "verify_two_gated_delta_rule: {} must be interleaved", name);
    };
    check_bf16_tile(in.q, "q");
    check_bf16_tile(in.k, "k");
    check_bf16_tile(in.v, "v");
    check_bf16_tile(in.a, "a");
    check_bf16_tile(in.b, "b");
    check_bf16_tile(in.decay_bias, "decay_bias");
    check_bf16_tile(in.decay_scale, "decay_scale");
    check_bf16_tile(in.norm_ones, "norm_ones");

    TT_FATAL(attrs.BH >= 1 && attrs.BH <= 110, "verify_two_gated_delta_rule: expected 1..110 heads, got {}", attrs.BH);
    // One tile row per 32 heads for q/k/v; one tile column per 32 heads for a/b.
    const uint32_t head_rows_padded = ((attrs.BH + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    TT_FATAL(attrs.key_dim == 128 && attrs.val_dim == 128, "verify_two_gated_delta_rule: expected K=V=128");

    const auto& q_shape = in.q.logical_shape();
    const auto& k_shape = in.k.logical_shape();
    const auto& v_shape = in.v.logical_shape();
    const auto& a_shape = in.a.logical_shape();
    const auto& b_shape = in.b.logical_shape();
    const auto& db_shape = in.decay_bias.logical_shape();
    const auto& ds_shape = in.decay_scale.logical_shape();
    const auto& state_shape = in.state.logical_shape();
    // q/k/v arrive in the native frontend layout [2, 1, BH, D]; a/b as [1, 1, 2, BH].
    // The reader gathers each head's row on device, so no host relayout is needed.
    auto check_qkv = [&](const ttnn::Shape& shape, uint32_t dim, const char* name) {
        TT_FATAL(
            shape.rank() == 4 && shape[0] == 2 && shape[1] == 1 && shape[2] == attrs.BH && shape[3] == dim,
            "verify_two_gated_delta_rule: {} must have shape [2,1,BH,D], got {}",
            name,
            shape);
    };
    check_qkv(q_shape, attrs.key_dim, "q");
    check_qkv(k_shape, attrs.key_dim, "k");
    check_qkv(v_shape, attrs.val_dim, "v");
    for (const auto* tensor : {&in.q, &in.k, &in.v}) {
        TT_FATAL(
            tensor->padded_shape() == ttnn::Shape({2, 1, head_rows_padded, 128}),
            "verify_two_gated_delta_rule: q/k/v must have padded shape [2,1,ceil32(BH),128]");
    }
    auto check_ab = [&](const ttnn::Shape& shape, const char* name) {
        TT_FATAL(
            shape.rank() == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] == 2 && shape[3] == attrs.BH,
            "verify_two_gated_delta_rule: {} must have shape [1,1,2,BH], got {}",
            name,
            shape);
    };
    check_ab(a_shape, "a");
    check_ab(b_shape, "b");
    for (const auto* tensor : {&in.a, &in.b}) {
        TT_FATAL(
            tensor->padded_shape() == ttnn::Shape({1, 1, 32, head_rows_padded}),
            "verify_two_gated_delta_rule: a/b must have padded shape [1,1,32,ceil32(BH)]");
    }
    TT_FATAL(
        db_shape.rank() == 3 && db_shape[0] == attrs.BH && db_shape[1] == 1 && db_shape[2] == 1,
        "verify_two_gated_delta_rule: decay_bias must have shape [BH,1,1]");
    TT_FATAL(
        ds_shape.rank() == 3 && ds_shape[0] == attrs.BH && ds_shape[1] == 1 && ds_shape[2] == 1,
        "verify_two_gated_delta_rule: decay_scale must have shape [BH,1,1]");
    TT_FATAL(
        in.decay_bias.padded_shape() == ttnn::Shape({attrs.BH, 32, 32}) &&
            in.decay_scale.padded_shape() == ttnn::Shape({attrs.BH, 32, 32}),
        "verify_two_gated_delta_rule: decay constants must contain one tile per head");
    TT_FATAL(
        in.norm_ones.logical_volume() == TILE_HEIGHT * TILE_WIDTH &&
            in.norm_ones.physical_volume() == TILE_HEIGHT * TILE_WIDTH,
        "verify_two_gated_delta_rule: norm_ones must be one shared normalization tile");
    for (const auto* tensor :
         {&in.k, &in.v, &in.a, &in.b, &in.decay_bias, &in.decay_scale, &in.norm_ones, &in.state}) {
        TT_FATAL(tensor->device() == in.q.device(), "verify_two_gated_delta_rule: tensors must share one device");
    }

    TT_FATAL(
        in.state.layout() == Layout::ROW_MAJOR,
        "verify_two_gated_delta_rule: state must use ROW_MAJOR layout");
    TT_FATAL(
        in.state.dtype() == DataType::BFLOAT16,
        "verify_two_gated_delta_rule: state must use bfloat16");
    TT_FATAL(in.state.buffer() != nullptr, "verify_two_gated_delta_rule: state must be on device");
    TT_FATAL(
        in.state.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "verify_two_gated_delta_rule: state must be height-sharded");
    TT_FATAL(
        in.state.memory_config().buffer_type() == BufferType::L1,
        "verify_two_gated_delta_rule: state must reside in L1");
    TT_FATAL(
        state_shape == ttnn::Shape({attrs.BH, attrs.key_dim, attrs.val_dim}) ||
            state_shape == ttnn::Shape({1, attrs.BH, attrs.key_dim, attrs.val_dim}),
        "verify_two_gated_delta_rule: state must have shape [BH,K,V] or [1,BH,K,V]");
    const auto& shard = in.state.shard_spec().value();
    TT_FATAL(
        shard.orientation == ShardOrientation::ROW_MAJOR && shard.shape[1] == attrs.val_dim &&
            shard.shape[0] * shard.grid.num_cores() >= attrs.BH * attrs.key_dim,
        "verify_two_gated_delta_rule: state requires row-major [rows,V] shards covering BH*K rows");
}

VerifyTwoGatedDeltaRuleDeviceOperation::spec_return_value_t
VerifyTwoGatedDeltaRuleDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto output_layout =
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    // Candidate states are emitted in TILE layout: the writer streams 16 whole
    // tiles per candidate instead of 4096 face-row scatters. The commit step
    // converts only the selected candidate into the row-major live state.
    const auto candidate_layout =
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG);
    const ttnn::Shape candidate_shape({1, attrs.BH, attrs.key_dim, attrs.val_dim});
    return {
        tt::tt_metal::TensorSpec(
            ttnn::Shape({2, 1, attrs.BH, attrs.val_dim}),
            output_layout),
        tt::tt_metal::TensorSpec(candidate_shape, candidate_layout),
        tt::tt_metal::TensorSpec(candidate_shape, candidate_layout),
    };
}

VerifyTwoGatedDeltaRuleDeviceOperation::tensor_return_value_t
VerifyTwoGatedDeltaRuleDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {
        create_device_tensor(specs[0], in.q.device()),
        create_device_tensor(specs[1], in.q.device()),
        create_device_tensor(specs[2], in.q.device()),
    };
}

std::vector<Tensor> verify_two_gated_delta_rule(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& a,
    const Tensor& b,
    const Tensor& decay_bias,
    const Tensor& decay_scale,
    const Tensor& norm_ones,
    const Tensor& state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = VerifyTwoGatedDeltaRuleDeviceOperation;
    const auto& q_shape = q.logical_shape();
    const auto& v_shape = v.logical_shape();
    TT_FATAL(
        q_shape.rank() == 4 && k.logical_shape().rank() == 4 && v_shape.rank() == 4,
        "verify_two_gated_delta_rule: q/k/v must have shape [2,1,BH,D]");
    auto attrs = OperationType::operation_attributes_t{
        .BH = q_shape[2],
        .key_dim = q_shape[3],
        .val_dim = v_shape[3],
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
        .q_scale = 1.0f / std::sqrt(static_cast<float>(q_shape[3])),
    };
    auto inputs = OperationType::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .a = a,
        .b = b,
        .decay_bias = decay_bias,
        .decay_scale = decay_scale,
        .norm_ones = norm_ones,
        .state = state,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, inputs);
}

}  // namespace ttnn::prim
