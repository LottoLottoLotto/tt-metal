// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <core/xtensor_utils.hpp>
#include <limits>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ops/ring_sdpa_bw/ring_sdpa_bw.hpp"
#include "metal/ops/ring_sdpa_fw/ring_sdpa_fw.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using ttml::metal::AttentionMaskType;
using ttml::metal::ops::get_ring_step_work;
using ttml::metal::ops::RingLoadBalancing;
using ttml::ttnn_fixed::distributed::RingShiftDirection;

constexpr float kSentinel = 7.0F;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr size_t kB = 1, kH = 4, kG = 2, kC = 128, kD = 64;
constexpr float kQScale = 3.0F;
constexpr double kForwardRelTol = 2e-2;
constexpr double kBackwardRelTol = 5e-2;

tt::tt_metal::distributed::MeshShape system_mesh_shape() {
    return tt::tt_metal::distributed::SystemMesh::instance().shape();
}

uint32_t ring_axis_of(const tt::tt_metal::distributed::MeshShape& shape) {
    uint32_t axis = 0;
    for (uint32_t i = 1; i < shape.dims(); ++i) {
        if (shape[i] > shape[axis]) {
            axis = i;
        }
    }
    return axis;
}

class RingSDPAChunkTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (system_mesh_shape().mesh_size() > 1U) {
            ttml::ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(system_mesh_shape().mesh_size()));
            ttml::autograd::ctx().open_device(system_mesh_shape());
        }
    }
    static void TearDownTestSuite() {
        if (system_mesh_shape().mesh_size() > 1U) {
            ttml::autograd::ctx().close_device();
        }
    }
    void SetUp() override {
        if (system_mesh_shape().mesh_size() <= 1U) {
            GTEST_SKIP() << "Ring SDPA chunk tests need a multi-device mesh";
        }
        ttml::autograd::ctx().get_device().clear_program_cache();
    }
};

xt::xarray<float> to_bf16(const xt::xarray<float>& x) {
    xt::xarray<float> rounded = x;
    for (auto& value : rounded) {
        value = static_cast<float>(bfloat16(value));
    }
    return rounded;
}

xt::xarray<float> uniform(const std::array<size_t, 4>& shape, float bound, uint32_t seed) {
    return to_bf16(ttml::test_utils::make_uniform_xarray<float>(shape, -bound, bound, seed));
}

struct ForwardReference {
    xt::xarray<float> output;
    xt::xarray<float> lse;
    xt::xarray<float> probabilities;
};

ForwardReference reference_forward(const xt::xarray<float>& q, const xt::xarray<float>& k, const xt::xarray<float>& v) {
    const size_t B = q.shape()[0], H = q.shape()[1], S = q.shape()[2], D = q.shape()[3];
    const size_t group = H / k.shape()[1];
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    ForwardReference ref{xt::zeros<float>({B, H, S, D}), xt::zeros<float>({B, H, S}), xt::zeros<float>({B, H, S, S})};
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t g = h / group;
            for (size_t i = 0; i < S; ++i) {
                std::vector<double> scores(S);
                double max_score = -std::numeric_limits<double>::infinity();
                for (size_t j = 0; j < S; ++j) {
                    double dot = 0.0;
                    for (size_t d = 0; d < D; ++d) {
                        dot += static_cast<double>(q(b, h, i, d)) * k(b, g, j, d);
                    }
                    scores[j] = dot * scale;
                    max_score = std::max(max_score, scores[j]);
                }
                double sum = 0.0;
                for (size_t j = 0; j < S; ++j) {
                    sum += std::exp(scores[j] - max_score);
                }
                ref.lse(b, h, i) = static_cast<float>(max_score + std::log(sum));
                for (size_t j = 0; j < S; ++j) {
                    const double p = std::exp(scores[j] - max_score) / sum;
                    ref.probabilities(b, h, i, j) = static_cast<float>(p);
                    for (size_t d = 0; d < D; ++d) {
                        ref.output(b, h, i, d) += static_cast<float>(p * v(b, g, j, d));
                    }
                }
            }
        }
    }
    return ref;
}

struct Gradients {
    xt::xarray<float> dq;
    xt::xarray<float> dk;
    xt::xarray<float> dv;
};

Gradients reference_backward(
    const xt::xarray<float>& q,
    const xt::xarray<float>& k,
    const xt::xarray<float>& v,
    const xt::xarray<float>& o,
    const xt::xarray<float>& probabilities,
    const xt::xarray<float>& d_o) {
    const size_t B = q.shape()[0], H = q.shape()[1], S = q.shape()[2], D = q.shape()[3];
    const size_t group = H / k.shape()[1];
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    Gradients grads{xt::zeros<float>(q.shape()), xt::zeros<float>(k.shape()), xt::zeros<float>(v.shape())};
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t g = h / group;
            for (size_t i = 0; i < S; ++i) {
                double u = 0.0;
                for (size_t d = 0; d < D; ++d) {
                    u += static_cast<double>(d_o(b, h, i, d)) * o(b, h, i, d);
                }
                for (size_t j = 0; j < S; ++j) {
                    const double p = probabilities(b, h, i, j);
                    double dp = 0.0;
                    for (size_t d = 0; d < D; ++d) {
                        dp += static_cast<double>(d_o(b, h, i, d)) * v(b, g, j, d);
                        grads.dv(b, g, j, d) += static_cast<float>(p * d_o(b, h, i, d));
                    }
                    const double ds = p * (dp - u) * scale;
                    for (size_t d = 0; d < D; ++d) {
                        grads.dq(b, h, i, d) += static_cast<float>(ds * k(b, g, j, d));
                        grads.dk(b, g, j, d) += static_cast<float>(ds * q(b, h, i, d));
                    }
                }
            }
        }
    }
    return grads;
}

double rel_l2(const xt::xarray<float>& actual, const xt::xarray<float>& expected) {
    double diff = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double e = expected.flat(i);
        const double a = actual.flat(i);
        diff += (a - e) * (a - e);
        norm += e * e;
    }
    return std::sqrt(diff / std::max(norm, 1e-30));
}

struct Chunk {
    xt::xarray<float> q, k, v;
    xt::xarray<float> o;
    xt::xarray<float> d_o;
    ForwardReference fw;
    Gradients bw;
};

std::array<Chunk, 2> make_chunks(uint32_t seed) {
    std::array<Chunk, 2> chunks;
    for (uint32_t i = 0; i < 2; ++i) {
        auto& chunk = chunks[i];
        const uint32_t s = seed + 100U * i;
        chunk.q = uniform({kB, kH, kC, kD}, kQScale, s);
        chunk.k = uniform({kB, kG, kC, kD}, 1.0F, s + 1);
        chunk.v = uniform({kB, kG, kC, kD}, 1.0F, s + 2);
        chunk.o = uniform({kB, kH, kC, kD}, 1.0F, s + 3);
        chunk.d_o = uniform({kB, kH, kC, kD}, 1.0F, s + 4);
        chunk.fw = reference_forward(chunk.q, chunk.k, chunk.v);
        chunk.bw = reference_backward(chunk.q, chunk.k, chunk.v, chunk.o, chunk.fw.probabilities, chunk.d_o);
    }
    return chunks;
}

xt::xarray<float> lse_to_intermediates(const xt::xarray<float>& lse) {
    const size_t B = lse.shape()[0], H = lse.shape()[1], S = lse.shape()[2];
    xt::xarray<float> out = xt::zeros<float>({B, H, S, size_t{32}});
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < S; ++s) {
                out(b, h, s, 0) = lse(b, h, s);
            }
        }
    }
    return out;
}

xt::xarray<float> lse_column(const xt::xarray<float>& intermediates) {
    return xt::view(intermediates, xt::all(), xt::all(), xt::all(), 0);
}

ttnn::Tensor to_device_bf16(const xt::xarray<float>& x) {
    return ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
        x, &ttml::autograd::ctx().get_device(), ttnn::Layout::TILE);
}

ttnn::Tensor to_device_fp32(const xt::xarray<float>& x) {
    return ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        x, &ttml::autograd::ctx().get_device(), ttnn::Layout::TILE);
}

ttnn::Tensor filled(
    const ttnn::Shape& shape,
    float value,
    ttnn::DataType dtype,
    const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
    return ttnn::full(
        shape, value, dtype, ttnn::Layout::TILE, std::ref(ttml::autograd::ctx().get_device()), memory_config);
}

std::vector<xt::xarray<float>> per_device(const ttnn::Tensor& tensor) {
    return ttml::core::to_xtensor<float>(tensor, ttml::core::IdentityComposer{});
}

uint32_t ring_id_of(uint32_t device_index, const tt::tt_metal::distributed::MeshShape& shape, uint32_t ring_axis) {
    uint32_t stride = 1;
    for (uint32_t i = shape.dims(); i-- > ring_axis + 1;) {
        stride *= shape[i];
    }
    return (device_index / stride) % shape[ring_axis];
}

bool all_equal(const xt::xarray<float>& x, float value) {
    return std::all_of(x.begin(), x.end(), [value](float v) { return v == value; });
}

struct RingSetup {
    tt::tt_metal::distributed::MeshShape mesh_shape;
    uint32_t ring_axis;
    uint32_t ring_size;
    uint32_t num_devices;
};

RingSetup ring_setup() {
    const auto mesh_shape = ttml::autograd::ctx().get_device().shape();
    const uint32_t ring_axis = ring_axis_of(mesh_shape);
    return RingSetup{mesh_shape, ring_axis, mesh_shape[ring_axis], static_cast<uint32_t>(mesh_shape.mesh_size())};
}

uint32_t active_chunk(uint32_t ring_id, uint32_t step, const RingSetup& ring, RingShiftDirection direction) {
    const auto work = get_ring_step_work(
        ring_id, step, ring.ring_size, AttentionMaskType::Causal, direction, RingLoadBalancing::ZigZag);
    EXPECT_TRUE(work.execute);
    EXPECT_EQ(work.mask_type, AttentionMaskType::None);
    return work.chunk;
}

// keep_alive holds the first pass's tensors so the cache-hit pass binds new buffer addresses.
template <typename Launch>
void for_each_direction_step_and_pass(const RingSetup& ring, Launch&& launch) {
    auto& device = ttml::autograd::ctx().get_device();
    for (auto direction : {RingShiftDirection::Backward, RingShiftDirection::Forward}) {
        for (uint32_t step = 1; step < ring.ring_size; ++step) {
            std::vector<ttnn::Tensor> keep_alive;
            for (uint32_t pass = 0; pass < 2; ++pass) {
                SCOPED_TRACE(fmt::format(
                    "direction {} step {} pass {}",
                    direction == RingShiftDirection::Backward ? "Backward" : "Forward",
                    step,
                    pass));
                const auto entries_before = device.num_program_cache_entries();
                launch(
                    direction,
                    step,
                    /*seed=*/1000U * step + 10U * pass + (direction == RingShiftDirection::Forward ? 5U : 0U),
                    keep_alive);
                if (pass == 0) {
                    EXPECT_GT(device.num_program_cache_entries(), entries_before) << "first launch must build programs";
                } else {
                    EXPECT_EQ(device.num_program_cache_entries(), entries_before)
                        << "second launch must hit the program cache (override_runtime_arguments path)";
                }
            }
        }
    }
}

void run_zigzag_forward_and_check(
    const RingSetup& ring,
    RingShiftDirection direction,
    uint32_t step,
    uint32_t seed,
    std::vector<ttnn::Tensor>& keep_alive,
    const ttnn::MemoryConfig& output_memory = ttnn::DRAM_MEMORY_CONFIG) {
    const ttnn::Shape out_shape{kB, kH, kC, kD};
    const ttnn::Shape inter_shape{kB, kH, kC, 32};
    const auto chunks = make_chunks(seed);
    std::array<ttnn::Tensor, 2> q, k, v, out, inter;
    for (uint32_t chunk = 0; chunk < 2; ++chunk) {
        q[chunk] = to_device_bf16(chunks[chunk].q);
        k[chunk] = to_device_bf16(chunks[chunk].k);
        v[chunk] = to_device_bf16(chunks[chunk].v);
        out[chunk] = filled(out_shape, kSentinel, ttnn::DataType::BFLOAT16, output_memory);
        inter[chunk] = filled(inter_shape, kNegInf, ttnn::DataType::FLOAT32, output_memory);
        keep_alive.insert(keep_alive.end(), {q[chunk], k[chunk], v[chunk], out[chunk], inter[chunk]});
    }

    ttml::metal::ring_sdpa_fw(
        q[0],
        k[0],
        v[0],
        ring.ring_size,
        ring.ring_axis,
        step,
        AttentionMaskType::Causal,
        direction,
        out[0],
        inter[0],
        RingLoadBalancing::ZigZag,
        ttml::metal::ops::ring_sdpa_fw::RingSDPAFwChunk{
            .query = q[1], .key = k[1], .value = v[1], .output = out[1], .intermediates = inter[1]});

    const std::array<std::vector<xt::xarray<float>>, 2> out_host{per_device(out[0]), per_device(out[1])};
    const std::array<std::vector<xt::xarray<float>>, 2> inter_host{per_device(inter[0]), per_device(inter[1])};

    for (uint32_t dev = 0; dev < ring.num_devices; ++dev) {
        const uint32_t ring_id = ring_id_of(dev, ring.mesh_shape, ring.ring_axis);
        const uint32_t active = active_chunk(ring_id, step, ring, direction);
        const uint32_t idle = 1U - active;
        SCOPED_TRACE(fmt::format("device {} (ring id {}) active chunk {}", dev, ring_id, active));

        EXPECT_LT(rel_l2(out_host[active][dev], chunks[active].fw.output), kForwardRelTol);
        EXPECT_LT(rel_l2(lse_column(inter_host[active][dev]), chunks[active].fw.lse), kForwardRelTol);
        EXPECT_GT(rel_l2(out_host[active][dev], chunks[idle].fw.output), 10 * kForwardRelTol);

        EXPECT_TRUE(all_equal(out_host[idle][dev], kSentinel)) << "wrote into the idle chunk's output";
        EXPECT_TRUE(all_equal(lse_column(inter_host[idle][dev]), kNegInf))
            << "wrote into the idle chunk's intermediates";
    }
}

}  // namespace

TEST_F(RingSDPAChunkTest, ZigZagForwardBindsAssignedChunkPerDevice) {
    const auto ring = ring_setup();
    for_each_direction_step_and_pass(
        ring, [&](RingShiftDirection direction, uint32_t step, uint32_t seed, std::vector<ttnn::Tensor>& keep_alive) {
            run_zigzag_forward_and_check(ring, direction, step, seed, keep_alive);
        });
}

TEST_F(RingSDPAChunkTest, ZigZagForwardOutputMemoryConfigIsPartOfProgramCacheKey) {
    const auto ring = ring_setup();
    auto& device = ttml::autograd::ctx().get_device();
    std::vector<ttnn::Tensor> keep_alive;

    run_zigzag_forward_and_check(ring, RingShiftDirection::Backward, /*step=*/1U, /*seed=*/11U, keep_alive);
    const auto entries_after_dram = device.num_program_cache_entries();
    run_zigzag_forward_and_check(
        ring, RingShiftDirection::Backward, /*step=*/1U, /*seed=*/12U, keep_alive, ttnn::L1_MEMORY_CONFIG);
    EXPECT_GT(device.num_program_cache_entries(), entries_after_dram)
        << "L1 outputs must not reuse the programs built for DRAM outputs";
}

TEST_F(RingSDPAChunkTest, ZigZagBackwardBindsAssignedChunkPerDevice) {
    const auto ring = ring_setup();

    for_each_direction_step_and_pass(
        ring, [&](RingShiftDirection direction, uint32_t step, uint32_t seed, std::vector<ttnn::Tensor>& keep_alive) {
            const auto chunks = make_chunks(seed);
            for (uint32_t chunk = 0; chunk < 2; ++chunk) {
                const auto& l = chunks[chunk];
                const auto wrong_u = reference_backward(l.q, l.k, l.v, chunks[1U - chunk].o, l.fw.probabilities, l.d_o);
                ASSERT_GT(rel_l2(wrong_u.dq, l.bw.dq), 4 * kBackwardRelTol);
                ASSERT_GT(rel_l2(wrong_u.dk, l.bw.dk), 4 * kBackwardRelTol);
            }

            std::array<ttnn::Tensor, 2> q, k, v, o, d_o, inter, dq, dk, dv;
            for (uint32_t chunk = 0; chunk < 2; ++chunk) {
                q[chunk] = to_device_bf16(chunks[chunk].q);
                k[chunk] = to_device_bf16(chunks[chunk].k);
                v[chunk] = to_device_bf16(chunks[chunk].v);
                o[chunk] = to_device_bf16(chunks[chunk].o);
                d_o[chunk] = to_device_bf16(chunks[chunk].d_o);
                inter[chunk] = to_device_fp32(lse_to_intermediates(chunks[chunk].fw.lse));
                dq[chunk] = filled(q[chunk].logical_shape(), kSentinel, ttnn::DataType::BFLOAT16);
                dk[chunk] = filled(k[chunk].logical_shape(), kSentinel, ttnn::DataType::BFLOAT16);
                dv[chunk] = filled(v[chunk].logical_shape(), kSentinel, ttnn::DataType::BFLOAT16);
                keep_alive.insert(keep_alive.end(), {q[chunk], k[chunk], v[chunk], o[chunk], d_o[chunk]});
                keep_alive.insert(keep_alive.end(), {inter[chunk], dq[chunk], dk[chunk], dv[chunk]});
            }

            ttml::metal::ring_sdpa_bw(
                d_o[0],
                o[0],
                q[0],
                k[0],
                v[0],
                inter[0],
                ring.ring_size,
                ring.ring_axis,
                step,
                AttentionMaskType::Causal,
                direction,
                dq[0],
                dk[0],
                dv[0],
                RingLoadBalancing::ZigZag,
                ttml::metal::RingSDPABwChunk{
                    .grad_output = d_o[1],
                    .attn_output = o[1],
                    .query = q[1],
                    .key = k[1],
                    .value = v[1],
                    .intermediates = inter[1],
                    .grad_query = dq[1],
                    .grad_key = dk[1],
                    .grad_value = dv[1]});

            const std::array<std::vector<xt::xarray<float>>, 2> dq_host{per_device(dq[0]), per_device(dq[1])};
            const std::array<std::vector<xt::xarray<float>>, 2> dk_host{per_device(dk[0]), per_device(dk[1])};
            const std::array<std::vector<xt::xarray<float>>, 2> dv_host{per_device(dv[0]), per_device(dv[1])};

            for (uint32_t dev = 0; dev < ring.num_devices; ++dev) {
                const uint32_t ring_id = ring_id_of(dev, ring.mesh_shape, ring.ring_axis);
                const uint32_t active = active_chunk(ring_id, step, ring, direction);
                const uint32_t idle = 1U - active;
                SCOPED_TRACE(fmt::format("device {} (ring id {}) active chunk {}", dev, ring_id, active));
                const auto& ref = chunks[active].bw;

                EXPECT_LT(rel_l2(dq_host[active][dev], ref.dq), kBackwardRelTol);
                EXPECT_LT(rel_l2(dk_host[active][dev], ref.dk), kBackwardRelTol);
                EXPECT_LT(rel_l2(dv_host[active][dev], ref.dv), kBackwardRelTol);

                EXPECT_TRUE(all_equal(dq_host[idle][dev], kSentinel)) << "wrote into the idle chunk's dQ";
                EXPECT_TRUE(all_equal(dk_host[idle][dev], kSentinel)) << "wrote into the idle chunk's dK";
                EXPECT_TRUE(all_equal(dv_host[idle][dev], kSentinel)) << "wrote into the idle chunk's dV";
            }
        });
}

TEST_F(RingSDPAChunkTest, ZigZagValidationRejectsInconsistentChunkArguments) {
    const auto ring = ring_setup();
    const auto chunks = make_chunks(4242U);
    const auto q = to_device_bf16(chunks[0].q);
    const auto k = to_device_bf16(chunks[0].k);
    const auto v = to_device_bf16(chunks[0].v);
    const auto o = to_device_bf16(chunks[0].o);
    const auto d_o = to_device_bf16(chunks[0].d_o);
    const auto inter = to_device_fp32(lse_to_intermediates(chunks[0].fw.lse));
    const ttnn::Shape out_shape{kB, kH, kC, kD};
    const ttnn::Shape inter_shape{kB, kH, kC, 32};
    const ttml::metal::ops::ring_sdpa_fw::RingSDPAFwChunk fw_chunk{
        .query = q,
        .key = k,
        .value = v,
        .output = filled(out_shape, 0.0F, ttnn::DataType::BFLOAT16),
        .intermediates = filled(inter_shape, 0.0F, ttnn::DataType::FLOAT32)};
    const ttml::metal::RingSDPABwChunk bw_chunk{
        .grad_output = d_o,
        .attn_output = o,
        .query = q,
        .key = k,
        .value = v,
        .intermediates = inter,
        .grad_query = filled(q.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16),
        .grad_key = filled(k.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16),
        .grad_value = filled(v.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16)};

    const auto short_chunks = [] {
        auto q_short = uniform({kB, kH, kC / 2, kD}, kQScale, 7U);
        auto kv_short = uniform({kB, kG, kC / 2, kD}, 1.0F, 8U);
        return std::pair{to_device_bf16(q_short), to_device_bf16(kv_short)};
    }();
    const ttml::metal::ops::ring_sdpa_fw::RingSDPAFwChunk fw_short_chunk{
        .query = short_chunks.first,
        .key = short_chunks.second,
        .value = short_chunks.second,
        .output = filled(short_chunks.first.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16),
        .intermediates = filled(ttnn::Shape{kB, kH, kC / 2, 32}, 0.0F, ttnn::DataType::FLOAT32)};

    const auto out0 = filled(out_shape, 0.0F, ttnn::DataType::BFLOAT16);
    const auto inter0 = filled(inter_shape, 0.0F, ttnn::DataType::FLOAT32);
    const auto dq0 = filled(q.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16);
    const auto dk0 = filled(k.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16);
    const auto dv0 = filled(v.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16);

    auto fw_l1_output_chunk = fw_chunk;
    fw_l1_output_chunk.output = filled(out_shape, 0.0F, ttnn::DataType::BFLOAT16, ttnn::L1_MEMORY_CONFIG);
    auto bw_l1_attn_output_chunk = bw_chunk;
    bw_l1_attn_output_chunk.attn_output = ttnn::to_memory_config(o, ttnn::L1_MEMORY_CONFIG);

    using FwChunkOpt = std::optional<ttml::metal::ops::ring_sdpa_fw::RingSDPAFwChunk>;
    using BwChunkOpt = std::optional<ttml::metal::RingSDPABwChunk>;
    auto fw = [&](uint32_t step, RingLoadBalancing selection, const FwChunkOpt& chunk1, bool give_chunk0 = true) {
        return [&, step, selection, chunk1, give_chunk0] {
            ttml::metal::ring_sdpa_fw(
                q,
                k,
                v,
                ring.ring_size,
                ring.ring_axis,
                step,
                AttentionMaskType::Causal,
                RingShiftDirection::Backward,
                give_chunk0 ? std::optional{out0} : std::nullopt,
                give_chunk0 ? std::optional{inter0} : std::nullopt,
                selection,
                chunk1);
        };
    };
    auto bw = [&](uint32_t step, RingLoadBalancing selection, const BwChunkOpt& chunk1, bool give_chunk0 = true) {
        return [&, step, selection, chunk1, give_chunk0] {
            ttml::metal::ring_sdpa_bw(
                d_o,
                o,
                q,
                k,
                v,
                inter,
                ring.ring_size,
                ring.ring_axis,
                step,
                AttentionMaskType::Causal,
                RingShiftDirection::Backward,
                give_chunk0 ? std::optional{dq0} : std::nullopt,
                give_chunk0 ? std::optional{dk0} : std::nullopt,
                give_chunk0 ? std::optional{dv0} : std::nullopt,
                selection,
                chunk1);
        };
    };

    EXPECT_THAT(
        fw(1U, RingLoadBalancing::ZigZag, FwChunkOpt{}),
        ThrowsMessage<std::runtime_error>(HasSubstr("requires the second-chunk tensors")));
    EXPECT_THAT(
        fw(0U, RingLoadBalancing::ZigZag, FwChunkOpt{fw_chunk}),
        ThrowsMessage<std::runtime_error>(HasSubstr("not defined for step 0")));
    EXPECT_THAT(
        fw(1U, RingLoadBalancing::None, FwChunkOpt{fw_chunk}),
        ThrowsMessage<std::runtime_error>(HasSubstr("load balancing is None")));
    EXPECT_THAT(
        fw(1U, RingLoadBalancing::ZigZag, FwChunkOpt{fw_chunk}, /*give_chunk0=*/false),
        ThrowsMessage<std::runtime_error>(HasSubstr("requires a preallocated chunk-0 'output'")));
    EXPECT_THAT(
        fw(1U, RingLoadBalancing::ZigZag, FwChunkOpt{fw_short_chunk}),
        ThrowsMessage<std::runtime_error>(HasSubstr("Second-chunk tensor 'query'")));
    EXPECT_THAT(
        fw(1U, RingLoadBalancing::ZigZag, FwChunkOpt{fw_l1_output_chunk}),
        ThrowsMessage<std::runtime_error>(HasSubstr("Second-chunk tensor 'output'")));

    EXPECT_THAT(
        bw(1U, RingLoadBalancing::ZigZag, BwChunkOpt{}),
        ThrowsMessage<std::runtime_error>(HasSubstr("requires the second-chunk tensors")));
    EXPECT_THAT(
        bw(0U, RingLoadBalancing::ZigZag, BwChunkOpt{bw_chunk}),
        ThrowsMessage<std::runtime_error>(HasSubstr("not defined for step 0")));
    EXPECT_THAT(
        bw(1U, RingLoadBalancing::None, BwChunkOpt{bw_chunk}),
        ThrowsMessage<std::runtime_error>(HasSubstr("load balancing is None")));
    EXPECT_THAT(
        bw(1U, RingLoadBalancing::ZigZag, BwChunkOpt{bw_chunk}, /*give_chunk0=*/false),
        ThrowsMessage<std::runtime_error>(HasSubstr("requires a preallocated chunk-0 'grad query'")));
    EXPECT_THAT(
        bw(1U, RingLoadBalancing::ZigZag, BwChunkOpt{bw_l1_attn_output_chunk}),
        ThrowsMessage<std::runtime_error>(HasSubstr("Second-chunk tensor 'attention output'")));
}
