// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the standalone chunk_gated_delta_rule op.
// Parallelism: one Tensix core per (B*HV) head; each core loops over NC chunks
// holding the recurrent state S [K,V] on-core. All math is in the compute kernel,
// derived from flash-linear-attention `naive_chunk_gated_delta_rule`.

#include "chunk_gated_delta_rule_program_factory.hpp"

#include <bit>
#include <algorithm>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

// CB index plan (all fp32). Kept in sync with the compute/reader/writer kernels.
namespace cb {
constexpr uint32_t q = tt::CBIndex::c_0;           // [C,K]
constexpr uint32_t k = tt::CBIndex::c_1;           // [C,K]
constexpr uint32_t v = tt::CBIndex::c_2;           // [C,V]
constexpr uint32_t g = tt::CBIndex::c_3;           // [C,1] column
constexpr uint32_t beta = tt::CBIndex::c_4;        // [C,1] column
constexpr uint32_t eye = tt::CBIndex::c_5;         // [C,C] constant
constexpr uint32_t tril = tt::CBIndex::c_6;        // [C,C] constant
constexpr uint32_t ones = tt::CBIndex::c_7;        // [C,C] all-ones constant
constexpr uint32_t S = tt::CBIndex::c_8;           // [K,V] persistent state
constexpr uint32_t decay = tt::CBIndex::c_9;       // [C,1]
constexpr uint32_t decay_exp = tt::CBIndex::c_10;  // [C,1]
constexpr uint32_t decay_row = tt::CBIndex::c_11;  // [1,C] transpose(decay)
constexpr uint32_t lmask = tt::CBIndex::c_12;      // [C,C]
constexpr uint32_t mmat = tt::CBIndex::c_13;       // [C,C]  M = I + strictly_lower(kk*Lmask)
constexpr uint32_t vbeta = tt::CBIndex::c_14;      // [C,V]
constexpr uint32_t kbeta = tt::CBIndex::c_15;      // [C,K]
constexpr uint32_t out = tt::CBIndex::c_16;        // [C,V] output o
constexpr uint32_t u = tt::CBIndex::c_17;          // [C,V]
constexpr uint32_t w = tt::CBIndex::c_18;          // [C,K]
constexpr uint32_t qdecay = tt::CBIndex::c_19;     // [C,K]
constexpr uint32_t intra = tt::CBIndex::c_20;      // [C,C]
constexpr uint32_t s2 = tt::CBIndex::c_21;         // [K,V] ping-pong state buffer
constexpr uint32_t vnew = tt::CBIndex::c_22;       // [C,V]
constexpr uint32_t ointer = tt::CBIndex::c_23;     // [C,V]
constexpr uint32_t kdec_t = tt::CBIndex::c_24;     // [K,C]
constexpr uint32_t supd = tt::CBIndex::c_25;       // [K,V]
constexpr uint32_t stmp = tt::CBIndex::c_26;       // [K,V]
constexpr uint32_t final_s = tt::CBIndex::c_27;    // [K,V]
constexpr uint32_t scr1 = tt::CBIndex::c_28;       // scratch [C,C]
constexpr uint32_t scr2 = tt::CBIndex::c_29;       // scratch [C,C]
constexpr uint32_t scr3 = tt::CBIndex::c_30;       // scratch [C,C]
constexpr uint32_t s3 = tt::CBIndex::c_31;         // [K,V] ping-pong state buffer 3
}  // namespace cb

tt::tt_metal::ProgramDescriptor ChunkGatedDeltaRuleProgramFactory::create_descriptor(
    const ChunkGatedDeltaRuleParams& attrs, const ChunkGatedDeltaRuleInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t Ct = attrs.chunk_size / TILE_HEIGHT;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;
    const uint32_t has_s0 = attrs.has_initial_state ? 1u : 0u;

    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;
    uint32_t scr = cc;
    scr = std::max(scr, ck);
    scr = std::max(scr, cv);
    scr = std::max(scr, kv);
    scr = std::max(scr, kc);

    // GPU-style mixed precision: q/k/v and the output o are bf16; the recurrent state, gate/decay,
    // masks, WY-inverse and all scratch stay fp32 (numerically sensitive). Matches FLA's Triton dtypes.
    const tt::DataFormat df_io = tt::DataFormat::Float16_b;  // bf16 for q/k/v/out

    auto* device = in.q.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t grid_y = grid.y;
    TT_FATAL(BH <= grid.x * grid_y, "num_heads {} exceeds compute cores {}", BH, grid.x * grid_y);

    std::vector<CoreCoord> head_cores(BH);
    std::set<CoreRange> core_set;
    for (uint32_t h = 0; h < BH; h++) {
        head_cores[h] = CoreCoord{h / grid_y, h % grid_y};
        core_set.insert(CoreRange{head_cores[h], head_cores[h]});
    }
    CoreRangeSet cores{core_set};

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t n_tiles, uint32_t nbuf = 1, tt::DataFormat fmt = tt::DataFormat::Float32) {
        const uint32_t ts = tt::tile_size(fmt);
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * nbuf * ts,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = ts}}}});
    };

    add_cb(cb::q, ck, 1, df_io);
    add_cb(cb::k, ck, 1, df_io);
    add_cb(cb::v, cv, 1, df_io);
    add_cb(cb::g, Ct);
    add_cb(cb::beta, Ct);
    add_cb(cb::eye, cc);
    add_cb(cb::tril, cc);
    add_cb(cb::ones, cc);
    add_cb(cb::S, kv, 2);
    add_cb(cb::decay, Ct);
    add_cb(cb::decay_exp, Ct);
    add_cb(cb::decay_row, Ct);
    add_cb(cb::lmask, cc);
    add_cb(cb::mmat, cc);
    add_cb(cb::vbeta, cv);
    add_cb(cb::kbeta, ck);
    add_cb(cb::out, cv, 2, df_io);
    add_cb(cb::u, cv);
    add_cb(cb::w, ck);
    add_cb(cb::qdecay, ck);
    add_cb(cb::intra, cc);
    add_cb(cb::s2, kv, 2);
    add_cb(cb::vnew, cv);
    add_cb(cb::ointer, cv);
    add_cb(cb::kdec_t, kc);
    add_cb(cb::supd, kv);
    add_cb(cb::stmp, kv);
    add_cb(cb::final_s, kv);
    add_cb(cb::scr1, scr);
    add_cb(cb::scr2, scr);
    add_cb(cb::scr3, scr);
    add_cb(cb::s3, kv, 2);

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";
    const std::vector<uint32_t> ct_args = {Ct, Kt, Vt, has_s0};

    // Reader compile args: {Ct,Kt,Vt,has_s0} + TensorAccessorArgs for each input (in order).
    std::vector<uint32_t> reader_ct = ct_args;
    TensorAccessorArgs(*in.q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.v.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.g.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.beta.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.eye_c.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tril_c.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.ones_c.buffer()).append_to(reader_ct);
    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = ct_args;
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_chunk_gated_delta_rule.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(BH);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_chunk_gated_delta_rule.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(BH);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/chunk_gated_delta_rule.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = ct_args;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
    compute.runtime_args.reserve(BH);

    auto* q_buf = in.q.buffer();
    auto* k_buf = in.k.buffer();
    auto* v_buf = in.v.buffer();
    auto* g_buf = in.g.buffer();
    auto* beta_buf = in.beta.buffer();
    auto* eye_buf = in.eye_c.buffer();
    auto* tril_buf = in.tril_c.buffer();
    auto* ones_buf = in.ones_c.buffer();
    auto* s0_buf = in.initial_state.has_value() ? in.initial_state->buffer() : nullptr;
    auto* o_buf = outputs[0].buffer();
    auto* fs_buf = outputs[1].buffer();

    for (uint32_t h = 0; h < BH; h++) {
        const auto& core = head_cores[h];
        reader.emplace_runtime_args(
            core, {h, NC, q_buf, k_buf, v_buf, g_buf, beta_buf, eye_buf, tril_buf, ones_buf, s0_buf});
        writer.emplace_runtime_args(core, {h, NC, o_buf, fs_buf});
        compute.emplace_runtime_args(core, {NC});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

tt::tt_metal::ProgramDescriptor RecurrentGatedDeltaRuleProgramFactory::create_descriptor(
    const RecurrentGatedDeltaRuleParams& attrs,
    const RecurrentGatedDeltaRuleInputs& in,
    std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;
    const uint32_t kv = Kt * Vt;
    const tt::DataFormat df = tt::DataFormat::Float16_b;

    auto* device = in.q.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    TT_FATAL(BH <= grid.x * grid.y, "num_heads {} exceeds compute cores {}", BH, grid.x * grid.y);

    std::vector<CoreCoord> head_cores(BH);
    std::set<CoreRange> core_set;
    for (uint32_t h = 0; h < BH; h++) {
        head_cores[h] = CoreCoord{h / grid.y, h % grid.y};
        core_set.insert(CoreRange{head_cores[h], head_cores[h]});
    }
    CoreRangeSet cores{core_set};

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t tiles) {
        const uint32_t ts = tt::tile_size(df);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * ts,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = ts}}}});
    };
    auto add_aliased_cbs = [&](std::initializer_list<uint32_t> indices, uint32_t tiles) {
        const uint32_t ts = tt::tile_size(df);
        CBDescriptor::FormatDescriptors formats;
        formats.reserve(indices.size());
        for (const uint32_t idx : indices) {
            formats.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = ts});
        }
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * ts, .core_ranges = cores, .format_descriptors = std::move(formats)});
    };

    add_cb(0, Kt);   // q
    add_cb(1, Kt);   // k
    add_cb(2, Vt);   // v
    add_cb(3, 1);    // g or raw a
    add_cb(4, 1);    // beta or raw b
    // The input state is consumed before the final state is produced. Give both
    // logical CBs one physical region, saving kv BF16 tiles (32 KiB at K=V=128).
    add_aliased_cbs({5, 17}, kv);  // input state / final state
    add_cb(16, Vt);                // output
    add_cb(18, kv);  // decayed state
    add_cb(19, Vt);  // state read
    add_cb(20, Vt);  // delta
    add_cb(21, Vt);  // beta * delta
    add_cb(22, Kt);  // transposed key
    add_cb(23, kv);  // outer product
    add_cb(24, kv);  // updated state before write
    add_cb(25, 1);   // exp(g)
    if (attrs.fuse_decode_frontend) {
        add_cb(6, 1);    // decay bias
        add_cb(7, 1);    // decay scale
        add_cb(26, 1);   // all-ones reduction tile / gate product
        add_cb(27, Kt);  // squared vector / gate sum
        add_cb(28, 1);   // row sum / sigmoid beta
        add_cb(29, 1);   // inverse norm / softplus
        add_cb(30, Kt);  // normalized and scaled q
        add_cb(31, Kt);  // normalized k
    }
    const std::vector<uint32_t> ct_args = {
        Kt,
        Vt,
        static_cast<uint32_t>(attrs.fuse_decode_frontend),
        std::bit_cast<uint32_t>(attrs.q_scale),
        std::bit_cast<uint32_t>(1.0e-6f)};

    std::vector<uint32_t> reader_ct = ct_args;
    TensorAccessorArgs(*in.q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.v.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.g.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.beta.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.state.buffer()).append_to(reader_ct);
    auto* decay_bias_buf =
        attrs.fuse_decode_frontend ? in.decay_bias->buffer() : in.q.buffer();
    auto* decay_scale_buf =
        attrs.fuse_decode_frontend ? in.decay_scale->buffer() : in.q.buffer();
    TensorAccessorArgs(*decay_bias_buf).append_to(reader_ct);
    TensorAccessorArgs(*decay_scale_buf).append_to(reader_ct);
    auto* norm_ones_buf =
        attrs.fuse_decode_frontend ? in.norm_ones->buffer() : in.q.buffer();
    TensorAccessorArgs(*norm_ones_buf).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {Kt, Vt};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);

    const std::string kdir =
        "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_recurrent_gated_delta_rule.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(BH);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_recurrent_gated_delta_rule.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(BH);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/recurrent_gated_delta_rule.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = ct_args;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi2, .fp32_dest_acc_en = true, .math_approx_mode = false};

    auto* q_buf = in.q.buffer();
    auto* k_buf = in.k.buffer();
    auto* v_buf = in.v.buffer();
    auto* g_buf = in.g.buffer();
    auto* beta_buf = in.beta.buffer();
    auto* state_buf = in.state.buffer();
    auto* out_buf = outputs[0].buffer();
    auto* final_buf = outputs[1].buffer();

    for (uint32_t h = 0; h < BH; h++) {
        const auto& core = head_cores[h];
        reader.emplace_runtime_args(
            core,
            {h, q_buf, k_buf, v_buf, g_buf, beta_buf, state_buf, decay_bias_buf, decay_scale_buf, norm_ones_buf});
        writer.emplace_runtime_args(core, {h, out_buf, final_buf});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

tt::tt_metal::ProgramDescriptor VerifyTwoGatedDeltaRuleProgramFactory::create_descriptor(
    const VerifyTwoGatedDeltaRuleParams& attrs,
    const VerifyTwoGatedDeltaRuleInputs& in,
    std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;
    const uint32_t kv = Kt * Vt;
    const tt::DataFormat df = tt::DataFormat::Float16_b;

    auto* device = in.q.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    TT_FATAL(BH == 32, "verify-two requires exactly 32 heads");
    TT_FATAL(BH <= grid.x * grid.y, "num_heads {} exceeds compute cores {}", BH, grid.x * grid.y);

    std::vector<CoreCoord> head_cores(BH);
    std::set<CoreRange> core_set;
    for (uint32_t h = 0; h < BH; ++h) {
        head_cores[h] = CoreCoord{h / grid.y, h % grid.y};
        core_set.insert(CoreRange{head_cores[h], head_cores[h]});
    }
    CoreRangeSet cores{core_set};

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t tiles) {
        const uint32_t ts = tt::tile_size(df);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * ts,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = ts}}}});
    };
    auto add_aliased_cbs = [&](std::initializer_list<uint32_t> indices, uint32_t tiles) {
        const uint32_t ts = tt::tile_size(df);
        CBDescriptor::FormatDescriptors formats;
        formats.reserve(indices.size());
        for (const uint32_t idx : indices) {
            formats.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = ts});
        }
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * ts, .core_ranges = cores, .format_descriptors = std::move(formats)});
    };

    add_cb(0, Kt);                 // q stream, one token at a time
    add_cb(1, Kt);                 // k stream
    add_cb(2, Vt);                 // v stream
    add_cb(3, 1);                  // raw a stream
    add_cb(4, 1);                  // raw b stream
    add_aliased_cbs({5, 17}, kv);  // input state, then candidate-state stream
    add_cb(16, Vt);                // output stream
    add_cb(18, kv);                // decayed state
    add_cb(19, Vt);                // state read
    add_cb(20, Vt);                // delta
    add_cb(21, Vt);                // beta * delta
    add_cb(22, Kt);                // transposed key
    add_cb(23, kv);                // outer product
    add_cb(24, kv);                // current updated state
    add_cb(25, 1);                 // exp(g)
    add_cb(6, 1);                  // decay bias, resident across both tokens
    add_cb(7, 1);                  // decay scale, resident across both tokens
    add_cb(26, 1);                 // all-ones reduction tile, resident across both tokens
    add_cb(27, Kt);                // squared vector
    add_cb(28, 1);                 // row sum / sigmoid beta
    add_cb(29, 1);                 // inverse norm / softplus
    add_cb(30, Kt);                // normalized and scaled q
    add_cb(31, Kt);                // normalized k

    const std::vector<uint32_t> ct_args = {
        Kt,
        Vt,
        1,
        std::bit_cast<uint32_t>(attrs.q_scale),
        std::bit_cast<uint32_t>(1.0e-6f)};

    std::vector<uint32_t> reader_ct = ct_args;
    TensorAccessorArgs(*in.q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.v.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.a.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.b.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.state.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.decay_bias.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.decay_scale.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.norm_ones.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {Kt, Vt};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[2].buffer()).append_to(writer_ct);

    const std::string kdir =
        "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_recurrent_gated_delta_rule.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.defines = {{"VERIFY_TWO", "1"}};
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(BH);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_recurrent_gated_delta_rule.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.defines = {{"VERIFY_TWO", "1"}};
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(BH);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/recurrent_gated_delta_rule.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = ct_args;
    compute.defines = {{"VERIFY_TWO", "1"}};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi2, .fp32_dest_acc_en = true, .math_approx_mode = false};

    auto* q_buf = in.q.buffer();
    auto* k_buf = in.k.buffer();
    auto* v_buf = in.v.buffer();
    auto* a_buf = in.a.buffer();
    auto* b_buf = in.b.buffer();
    auto* state_buf = in.state.buffer();
    auto* decay_bias_buf = in.decay_bias.buffer();
    auto* decay_scale_buf = in.decay_scale.buffer();
    auto* norm_ones_buf = in.norm_ones.buffer();
    auto* out_buf = outputs[0].buffer();
    auto* state1_buf = outputs[1].buffer();
    auto* state2_buf = outputs[2].buffer();

    for (uint32_t h = 0; h < BH; ++h) {
        const auto& core = head_cores[h];
        reader.emplace_runtime_args(
            core,
            {h,
             q_buf,
             k_buf,
             v_buf,
             a_buf,
             b_buf,
             state_buf,
             decay_bias_buf,
             decay_scale_buf,
             norm_ones_buf,
             BH});
        writer.emplace_runtime_args(core, {h, out_buf, state1_buf, state2_buf, BH});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
