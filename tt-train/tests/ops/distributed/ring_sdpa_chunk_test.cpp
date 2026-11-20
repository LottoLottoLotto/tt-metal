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
#include "ttnn/operations/data_movement/copy/copy.hpp"
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

ttml::metal::RingSDPAFwChunk upload_fw_chunk(
    const Chunk& chunk, const ttnn::MemoryConfig& output_memory = ttnn::DRAM_MEMORY_CONFIG) {
    return {
        .query = to_device_bf16(chunk.q),
        .key = to_device_bf16(chunk.k),
        .value = to_device_bf16(chunk.v),
        .output = filled(ttnn::Shape{kB, kH, kC, kD}, kSentinel, ttnn::DataType::BFLOAT16, output_memory),
        .intermediates = filled(ttnn::Shape{kB, kH, kC, 32}, kNegInf, ttnn::DataType::FLOAT32, output_memory)};
}

ttml::metal::RingSDPABwChunk upload_bw_chunk(const Chunk& chunk) {
    return {
        .grad_output = to_device_bf16(chunk.d_o),
        .attn_output = to_device_bf16(chunk.o),
        .query = to_device_bf16(chunk.q),
        .key = to_device_bf16(chunk.k),
        .value = to_device_bf16(chunk.v),
        .intermediates = to_device_fp32(lse_to_intermediates(chunk.fw.lse)),
        .grad_query = filled(ttnn::Shape{kB, kH, kC, kD}, kSentinel, ttnn::DataType::BFLOAT16),
        .grad_key = filled(ttnn::Shape{kB, kG, kC, kD}, kSentinel, ttnn::DataType::BFLOAT16),
        .grad_value = filled(ttnn::Shape{kB, kG, kC, kD}, kSentinel, ttnn::DataType::BFLOAT16)};
}

void keep(std::vector<ttnn::Tensor>& keep_alive, const ttml::metal::RingSDPAFwChunk& chunk) {
    keep_alive.insert(keep_alive.end(), {chunk.query, chunk.key, chunk.value, chunk.output, chunk.intermediates});
}

void keep(std::vector<ttnn::Tensor>& keep_alive, const ttml::metal::RingSDPABwChunk& chunk) {
    keep_alive.insert(keep_alive.end(), {chunk.grad_output, chunk.attn_output, chunk.query, chunk.key, chunk.value});
    keep_alive.insert(keep_alive.end(), {chunk.intermediates, chunk.grad_query, chunk.grad_key, chunk.grad_value});
}

void reset_outputs(const ttml::metal::RingSDPAFwChunk& chunk) {
    ttnn::copy(filled(chunk.output.logical_shape(), kSentinel, ttnn::DataType::BFLOAT16), chunk.output);
    ttnn::copy(filled(chunk.intermediates.logical_shape(), kNegInf, ttnn::DataType::FLOAT32), chunk.intermediates);
}

void reset_grads(const ttml::metal::RingSDPABwChunk& chunk) {
    ttnn::copy(filled(chunk.grad_query.logical_shape(), kSentinel, ttnn::DataType::BFLOAT16), chunk.grad_query);
    ttnn::copy(filled(chunk.grad_key.logical_shape(), kSentinel, ttnn::DataType::BFLOAT16), chunk.grad_key);
    ttnn::copy(filled(chunk.grad_value.logical_shape(), kSentinel, ttnn::DataType::BFLOAT16), chunk.grad_value);
}

void expect_zigzag_forward_wrote_assigned_chunks(
    const RingSetup& ring,
    RingShiftDirection direction,
    uint32_t step,
    const std::array<Chunk, 2>& data,
    const std::vector<ttml::metal::RingSDPAFwChunk>& fw_chunks) {
    const std::array<std::vector<xt::xarray<float>>, 2> out_host{
        per_device(fw_chunks[0].output), per_device(fw_chunks[1].output)};
    const std::array<std::vector<xt::xarray<float>>, 2> inter_host{
        per_device(fw_chunks[0].intermediates), per_device(fw_chunks[1].intermediates)};

    for (uint32_t dev = 0; dev < ring.num_devices; ++dev) {
        const uint32_t ring_id = ring_id_of(dev, ring.mesh_shape, ring.ring_axis);
        const uint32_t active = active_chunk(ring_id, step, ring, direction);
        const uint32_t idle = 1U - active;
        SCOPED_TRACE(fmt::format("device {} (ring id {}) active chunk {}", dev, ring_id, active));

        EXPECT_LT(rel_l2(out_host[active][dev], data[active].fw.output), kForwardRelTol);
        EXPECT_LT(rel_l2(lse_column(inter_host[active][dev]), data[active].fw.lse), kForwardRelTol);
        EXPECT_GT(rel_l2(out_host[active][dev], data[idle].fw.output), 10 * kForwardRelTol);

        EXPECT_TRUE(all_equal(out_host[idle][dev], kSentinel)) << "wrote into the idle chunk's output";
        EXPECT_TRUE(all_equal(lse_column(inter_host[idle][dev]), kNegInf))
            << "wrote into the idle chunk's intermediates";
    }
}

void expect_zigzag_backward_wrote_assigned_chunks(
    const RingSetup& ring,
    RingShiftDirection direction,
    uint32_t step,
    const std::array<Chunk, 2>& data,
    const std::vector<ttml::metal::RingSDPABwChunk>& bw_chunks) {
    const std::array<std::vector<xt::xarray<float>>, 2> dq_host{
        per_device(bw_chunks[0].grad_query), per_device(bw_chunks[1].grad_query)};
    const std::array<std::vector<xt::xarray<float>>, 2> dk_host{
        per_device(bw_chunks[0].grad_key), per_device(bw_chunks[1].grad_key)};
    const std::array<std::vector<xt::xarray<float>>, 2> dv_host{
        per_device(bw_chunks[0].grad_value), per_device(bw_chunks[1].grad_value)};

    for (uint32_t dev = 0; dev < ring.num_devices; ++dev) {
        const uint32_t ring_id = ring_id_of(dev, ring.mesh_shape, ring.ring_axis);
        const uint32_t active = active_chunk(ring_id, step, ring, direction);
        const uint32_t idle = 1U - active;
        SCOPED_TRACE(fmt::format("device {} (ring id {}) active chunk {}", dev, ring_id, active));
        const auto& ref = data[active].bw;

        EXPECT_LT(rel_l2(dq_host[active][dev], ref.dq), kBackwardRelTol);
        EXPECT_LT(rel_l2(dk_host[active][dev], ref.dk), kBackwardRelTol);
        EXPECT_LT(rel_l2(dv_host[active][dev], ref.dv), kBackwardRelTol);

        EXPECT_TRUE(all_equal(dq_host[idle][dev], kSentinel)) << "wrote into the idle chunk's dQ";
        EXPECT_TRUE(all_equal(dk_host[idle][dev], kSentinel)) << "wrote into the idle chunk's dK";
        EXPECT_TRUE(all_equal(dv_host[idle][dev], kSentinel)) << "wrote into the idle chunk's dV";
    }
}

void run_zigzag_forward_and_check(
    const RingSetup& ring,
    RingShiftDirection direction,
    uint32_t step,
    uint32_t seed,
    std::vector<ttnn::Tensor>& keep_alive,
    const ttnn::MemoryConfig& output_memory = ttnn::DRAM_MEMORY_CONFIG) {
    const auto data = make_chunks(seed);
    const std::vector<ttml::metal::RingSDPAFwChunk> fw_chunks{
        upload_fw_chunk(data[0], output_memory), upload_fw_chunk(data[1], output_memory)};
    for (const auto& chunk : fw_chunks) {
        keep(keep_alive, chunk);
    }

    ttml::metal::ring_sdpa_fw(
        fw_chunks,
        ring.ring_size,
        ring.ring_axis,
        step,
        AttentionMaskType::Causal,
        direction,
        RingLoadBalancing::ZigZag);

    expect_zigzag_forward_wrote_assigned_chunks(ring, direction, step, data, fw_chunks);
}

auto launch_fw(
    const RingSetup& ring,
    std::vector<ttml::metal::RingSDPAFwChunk> chunks,
    uint32_t step,
    AttentionMaskType mask_type,
    RingLoadBalancing load_balancing) {
    return [ring, chunks = std::move(chunks), step, mask_type, load_balancing] {
        ttml::metal::ring_sdpa_fw(
            chunks, ring.ring_size, ring.ring_axis, step, mask_type, RingShiftDirection::Backward, load_balancing);
    };
}

auto launch_bw(
    const RingSetup& ring,
    std::vector<ttml::metal::RingSDPABwChunk> chunks,
    uint32_t step,
    AttentionMaskType mask_type,
    RingLoadBalancing load_balancing) {
    return [ring, chunks = std::move(chunks), step, mask_type, load_balancing] {
        ttml::metal::ring_sdpa_bw(
            chunks, ring.ring_size, ring.ring_axis, step, mask_type, RingShiftDirection::Backward, load_balancing);
    };
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
            const auto data = make_chunks(seed);
            for (uint32_t chunk = 0; chunk < 2; ++chunk) {
                const auto& l = data[chunk];
                const auto wrong_u = reference_backward(l.q, l.k, l.v, data[1U - chunk].o, l.fw.probabilities, l.d_o);
                ASSERT_GT(rel_l2(wrong_u.dq, l.bw.dq), 4 * kBackwardRelTol);
                ASSERT_GT(rel_l2(wrong_u.dk, l.bw.dk), 4 * kBackwardRelTol);
            }

            const std::vector<ttml::metal::RingSDPABwChunk> bw_chunks{
                upload_bw_chunk(data[0]), upload_bw_chunk(data[1])};
            for (const auto& chunk : bw_chunks) {
                keep(keep_alive, chunk);
            }

            ttml::metal::ring_sdpa_bw(
                bw_chunks,
                ring.ring_size,
                ring.ring_axis,
                step,
                AttentionMaskType::Causal,
                direction,
                RingLoadBalancing::ZigZag);

            expect_zigzag_backward_wrote_assigned_chunks(ring, direction, step, data, bw_chunks);
        });
}

TEST_F(RingSDPAChunkTest, ZigZagCacheHitBindsChunksByListPositionNotByCachedAddress) {
    const auto ring = ring_setup();
    auto& device = ttml::autograd::ctx().get_device();
    const auto direction = RingShiftDirection::Backward;
    const uint32_t step = 1U;
    const auto data = make_chunks(31U);
    const std::array<Chunk, 2> swapped_data{data[1], data[0]};

    std::vector<ttml::metal::RingSDPAFwChunk> fw_chunks{upload_fw_chunk(data[0]), upload_fw_chunk(data[1])};
    ttml::metal::ring_sdpa_fw(
        fw_chunks,
        ring.ring_size,
        ring.ring_axis,
        step,
        AttentionMaskType::Causal,
        direction,
        RingLoadBalancing::ZigZag);
    expect_zigzag_forward_wrote_assigned_chunks(ring, direction, step, data, fw_chunks);

    std::vector<ttml::metal::RingSDPABwChunk> bw_chunks{upload_bw_chunk(data[0]), upload_bw_chunk(data[1])};
    ttml::metal::ring_sdpa_bw(
        bw_chunks,
        ring.ring_size,
        ring.ring_axis,
        step,
        AttentionMaskType::Causal,
        direction,
        RingLoadBalancing::ZigZag);
    expect_zigzag_backward_wrote_assigned_chunks(ring, direction, step, data, bw_chunks);

    for (const auto& chunk : fw_chunks) {
        reset_outputs(chunk);
    }
    for (const auto& chunk : bw_chunks) {
        reset_grads(chunk);
    }
    std::swap(fw_chunks[0], fw_chunks[1]);
    std::swap(bw_chunks[0], bw_chunks[1]);

    const auto entries_before_hit = device.num_program_cache_entries();
    ttml::metal::ring_sdpa_fw(
        fw_chunks,
        ring.ring_size,
        ring.ring_axis,
        step,
        AttentionMaskType::Causal,
        direction,
        RingLoadBalancing::ZigZag);
    ttml::metal::ring_sdpa_bw(
        bw_chunks,
        ring.ring_size,
        ring.ring_axis,
        step,
        AttentionMaskType::Causal,
        direction,
        RingLoadBalancing::ZigZag);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "swapped chunks must hit the program cache";

    expect_zigzag_forward_wrote_assigned_chunks(ring, direction, step, swapped_data, fw_chunks);
    expect_zigzag_backward_wrote_assigned_chunks(ring, direction, step, swapped_data, bw_chunks);
}

TEST_F(RingSDPAChunkTest, ContiguousCausalSkippedDevicesKeepCallerOutputsAndGrads) {
    const auto ring = ring_setup();
    for (auto direction : {RingShiftDirection::Backward, RingShiftDirection::Forward}) {
        for (uint32_t step = 1; step < ring.ring_size; ++step) {
            SCOPED_TRACE(fmt::format(
                "direction {} step {}", direction == RingShiftDirection::Backward ? "Backward" : "Forward", step));
            const auto data = make_chunks(2000U + step)[0];
            const auto fw_chunk = upload_fw_chunk(data);
            const auto bw_chunk = upload_bw_chunk(data);

            ttml::metal::ring_sdpa_fw(
                {fw_chunk}, ring.ring_size, ring.ring_axis, step, AttentionMaskType::Causal, direction);
            ttml::metal::ring_sdpa_bw(
                {bw_chunk}, ring.ring_size, ring.ring_axis, step, AttentionMaskType::Causal, direction);

            const auto out_host = per_device(fw_chunk.output);
            const auto inter_host = per_device(fw_chunk.intermediates);
            const auto dq_host = per_device(bw_chunk.grad_query);
            const auto dk_host = per_device(bw_chunk.grad_key);
            const auto dv_host = per_device(bw_chunk.grad_value);

            for (uint32_t dev = 0; dev < ring.num_devices; ++dev) {
                const uint32_t ring_id = ring_id_of(dev, ring.mesh_shape, ring.ring_axis);
                const auto work = get_ring_step_work(
                    ring_id, step, ring.ring_size, AttentionMaskType::Causal, direction, RingLoadBalancing::None);
                SCOPED_TRACE(fmt::format("device {} (ring id {}) executes {}", dev, ring_id, work.execute));

                if (work.execute) {
                    EXPECT_EQ(work.mask_type, AttentionMaskType::None);
                    EXPECT_LT(rel_l2(out_host[dev], data.fw.output), kForwardRelTol);
                    EXPECT_LT(rel_l2(lse_column(inter_host[dev]), data.fw.lse), kForwardRelTol);
                    EXPECT_LT(rel_l2(dq_host[dev], data.bw.dq), kBackwardRelTol);
                    EXPECT_LT(rel_l2(dk_host[dev], data.bw.dk), kBackwardRelTol);
                    EXPECT_LT(rel_l2(dv_host[dev], data.bw.dv), kBackwardRelTol);
                } else {
                    EXPECT_TRUE(all_equal(out_host[dev], kSentinel)) << "skipped device wrote its output";
                    EXPECT_TRUE(all_equal(inter_host[dev], kNegInf)) << "skipped device wrote its intermediates";
                    EXPECT_TRUE(all_equal(dq_host[dev], kSentinel)) << "skipped device wrote its dQ";
                    EXPECT_TRUE(all_equal(dk_host[dev], kSentinel)) << "skipped device wrote its dK";
                    EXPECT_TRUE(all_equal(dv_host[dev], kSentinel)) << "skipped device wrote its dV";
                }
            }
        }
    }
}

TEST_F(RingSDPAChunkTest, ValidationRejectsChunkCountThatDoesNotMatchLoadBalancing) {
    const auto ring = ring_setup();
    const auto data = make_chunks(4242U);
    const auto fw_chunk = upload_fw_chunk(data[0]);
    const auto bw_chunk = upload_bw_chunk(data[0]);

    EXPECT_THAT(
        launch_fw(ring, {fw_chunk, fw_chunk}, 1U, AttentionMaskType::Causal, RingLoadBalancing::None),
        ThrowsMessage<std::runtime_error>(HasSubstr("expects 1 chunks, got 2")));
    EXPECT_THAT(
        launch_fw(ring, {fw_chunk}, 1U, AttentionMaskType::Causal, RingLoadBalancing::ZigZag),
        ThrowsMessage<std::runtime_error>(HasSubstr("expects 2 chunks, got 1")));
    EXPECT_THAT(
        launch_bw(ring, {bw_chunk, bw_chunk}, 1U, AttentionMaskType::Causal, RingLoadBalancing::None),
        ThrowsMessage<std::runtime_error>(HasSubstr("expects 1 chunks, got 2")));
    EXPECT_THAT(
        launch_bw(ring, {bw_chunk}, 1U, AttentionMaskType::Causal, RingLoadBalancing::ZigZag),
        ThrowsMessage<std::runtime_error>(HasSubstr("expects 2 chunks, got 1")));
}

TEST_F(RingSDPAChunkTest, ValidationRejectsZigZagAtStepZeroAndWithoutCausalMask) {
    const auto ring = ring_setup();
    const auto data = make_chunks(4243U);
    const std::vector<ttml::metal::RingSDPAFwChunk> fw_chunks{upload_fw_chunk(data[0]), upload_fw_chunk(data[1])};
    const std::vector<ttml::metal::RingSDPABwChunk> bw_chunks{upload_bw_chunk(data[0]), upload_bw_chunk(data[1])};

    EXPECT_THAT(
        launch_fw(ring, fw_chunks, 0U, AttentionMaskType::Causal, RingLoadBalancing::ZigZag),
        ThrowsMessage<std::runtime_error>(HasSubstr("not defined for step 0")));
    EXPECT_THAT(
        launch_fw(ring, fw_chunks, 1U, AttentionMaskType::None, RingLoadBalancing::ZigZag),
        ThrowsMessage<std::runtime_error>(HasSubstr("requires a causal mask")));
    EXPECT_THAT(
        launch_bw(ring, bw_chunks, 0U, AttentionMaskType::Causal, RingLoadBalancing::ZigZag),
        ThrowsMessage<std::runtime_error>(HasSubstr("not defined for step 0")));
    EXPECT_THAT(
        launch_bw(ring, bw_chunks, 1U, AttentionMaskType::None, RingLoadBalancing::ZigZag),
        ThrowsMessage<std::runtime_error>(HasSubstr("requires a causal mask")));
}

TEST_F(RingSDPAChunkTest, ValidationRejectsChunksWhoseSpecsDifferFromChunkZero) {
    const auto ring = ring_setup();
    const auto data = make_chunks(4244U);
    const auto fw_chunk = upload_fw_chunk(data[0]);
    const auto bw_chunk = upload_bw_chunk(data[0]);

    const auto q_short = to_device_bf16(uniform({kB, kH, kC / 2, kD}, kQScale, 7U));
    const auto kv_short = to_device_bf16(uniform({kB, kG, kC / 2, kD}, 1.0F, 8U));
    const ttml::metal::RingSDPAFwChunk fw_short_chunk{
        .query = q_short,
        .key = kv_short,
        .value = kv_short,
        .output = filled(q_short.logical_shape(), 0.0F, ttnn::DataType::BFLOAT16),
        .intermediates = filled(ttnn::Shape{kB, kH, kC / 2, 32}, 0.0F, ttnn::DataType::FLOAT32)};
    const auto fw_l1_output_chunk = upload_fw_chunk(data[1], ttnn::L1_MEMORY_CONFIG);
    auto bw_l1_attn_output_chunk = upload_bw_chunk(data[1]);
    bw_l1_attn_output_chunk.attn_output =
        ttnn::to_memory_config(bw_l1_attn_output_chunk.attn_output, ttnn::L1_MEMORY_CONFIG);

    EXPECT_THAT(
        launch_fw(ring, {fw_chunk, fw_short_chunk}, 1U, AttentionMaskType::Causal, RingLoadBalancing::ZigZag),
        ThrowsMessage<std::runtime_error>(HasSubstr("Chunk 1 'query'")));
    EXPECT_THAT(
        launch_fw(ring, {fw_chunk, fw_l1_output_chunk}, 1U, AttentionMaskType::Causal, RingLoadBalancing::ZigZag),
        ThrowsMessage<std::runtime_error>(HasSubstr("Chunk 1 'output'")));
    EXPECT_THAT(
        launch_bw(
            ring, {bw_chunk, bw_l1_attn_output_chunk}, 1U, AttentionMaskType::Causal, RingLoadBalancing::ZigZag),
        ThrowsMessage<std::runtime_error>(HasSubstr("Chunk 1 'attn_output'")));
}

TEST_F(RingSDPAChunkTest, ValidationRejectsEmptyChunkListBeforeLaunch) {
    const auto ring = ring_setup();
    const auto u_scaler = filled(ttnn::Shape{1, 1, kB * kH * kC, 32}, 0.0F, ttnn::DataType::FLOAT32);

    EXPECT_THAT(
        launch_fw(ring, {}, 1U, AttentionMaskType::Causal, RingLoadBalancing::None),
        ThrowsMessage<std::runtime_error>(HasSubstr("needs at least one chunk")));
    EXPECT_THAT(
        [&] { ttnn::prim::ttml_ring_sdpa_bw_q({}, ring.ring_size, ring.ring_axis, 1U, AttentionMaskType::Causal); },
        ThrowsMessage<std::runtime_error>(HasSubstr("needs at least one chunk")));
    EXPECT_THAT(
        [&] {
            ttnn::prim::ttml_ring_sdpa_bw_kv(
                {}, u_scaler, ring.ring_size, ring.ring_axis, 1U, AttentionMaskType::Causal);
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("needs at least one chunk")));
}

TEST_F(RingSDPAChunkTest, ValidationRejectsDefaultConstructedChunkTensorBeforeLaunch) {
    const auto ring = ring_setup();
    const auto data = make_chunks(4245U);
    auto fw_chunk = upload_fw_chunk(data[0]);
    fw_chunk.output = ttnn::Tensor();
    auto bw_chunk = upload_bw_chunk(data[0]);
    bw_chunk.grad_query = ttnn::Tensor();

    EXPECT_THAT(
        launch_fw(ring, {fw_chunk}, 1U, AttentionMaskType::Causal, RingLoadBalancing::None),
        ThrowsMessage<std::runtime_error>(HasSubstr("Chunk 0 'output' is a default-constructed tensor")));
    EXPECT_THAT(
        launch_bw(ring, {bw_chunk}, 1U, AttentionMaskType::Causal, RingLoadBalancing::None),
        ThrowsMessage<std::runtime_error>(HasSubstr("Chunk 0 'grad_query' is a default-constructed tensor")));
}
