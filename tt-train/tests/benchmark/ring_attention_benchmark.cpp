// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <core/xtensor_utils.hpp>
#include <string>
#include <tools/profiler/op_profiler.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/ring_attention_sdpa.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/create_socket.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace {

using ttml::ops::distributed::RingAttentionLayout;

struct Config {
    uint32_t batch = 1U;
    std::vector<uint32_t> seqs{32768U};
    uint32_t heads = 8U;
    uint32_t kv_heads = 2U;
    uint32_t head_dim = 128U;
    uint32_t iters = 5U;
    bool check = false;
    bool profile_read = false;
    std::vector<std::string> layouts{"contiguous", "zigzag"};
};

xt::xarray<float> zigzag_permute(const xt::xarray<float>& x, uint32_t cp_size, bool to_device) {
    const size_t B = x.shape()[0], H = x.shape()[1], S = x.shape()[2], D = x.shape()[3];
    xt::xarray<float> out = xt::empty<float>(x.shape());
    for (size_t bh = 0; bh < B * H; ++bh) {
        for (size_t pos = 0; pos < S; ++pos) {
            const size_t global = ttml::ops::distributed::zigzag_global_position(pos, S, cp_size);
            const size_t from = to_device ? global : pos;
            const size_t to = to_device ? pos : global;
            std::copy_n(x.data() + (bh * S + from) * D, D, out.data() + (bh * S + to) * D);
        }
    }
    return out;
}

xt::xarray<float> gather_ring(const ttnn::Tensor& tensor, uint32_t cp_axis) {
    const auto shards = ttml::core::to_xtensor<float>(tensor, ttml::core::IdentityComposer{});
    const auto mesh_shape = ttml::autograd::ctx().get_device().shape();
    const uint32_t ring_size = mesh_shape[cp_axis];
    uint32_t stride = 1U;
    for (uint32_t i = mesh_shape.dims(); i-- > cp_axis + 1U;) {
        stride *= mesh_shape[i];
    }
    const auto& first = shards[0];
    const size_t B = first.shape()[0], H = first.shape()[1], S_local = first.shape()[2], D = first.shape()[3];
    xt::xarray<float> out = xt::empty<float>({B, H, S_local * ring_size, D});
    for (uint32_t r = 0; r < ring_size; ++r) {
        const auto& shard = shards[r * stride];
        for (size_t bh = 0; bh < B * H; ++bh) {
            std::copy_n(
                shard.data() + bh * S_local * D, S_local * D, out.data() + (bh * S_local * ring_size + r * S_local) * D);
        }
    }
    return out;
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

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2U;
    return values.size() % 2U == 1U ? values[mid] : 0.5 * (values[mid - 1U] + values[mid]);
}

struct RunResult {
    std::vector<double> fw_ms;
    std::vector<double> bw_ms;
    xt::xarray<float> out, dq, dk, dv;
};

RunResult run_layout(
    RingAttentionLayout layout,
    const std::string& name,
    uint32_t seq,
    const Config& cfg,
    const xt::xarray<float>& q_host,
    const xt::xarray<float>& k_host,
    const xt::xarray<float>& v_host,
    const xt::xarray<float>& do_host) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis = pctx.get_cp_axis().value();
    const uint32_t ring_size = pctx.get_cp_size();
    const bool zigzag = layout == RingAttentionLayout::ZigZag;
    const auto to_device_order = [&](const xt::xarray<float>& x) {
        return zigzag ? zigzag_permute(x, ring_size, /*to_device=*/true) : x;
    };
    const auto to_natural_order = [&](const xt::xarray<float>& x) {
        return zigzag ? zigzag_permute(x, ring_size, /*to_device=*/false) : x;
    };

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, /*dim=*/2, cp_axis);
    const auto upload = [&](const xt::xarray<float>& x) {
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            to_device_order(x), device, ttnn::Layout::TILE, mapper.get());
    };
    const ttnn::Tensor q_tt = upload(q_host);
    const ttnn::Tensor k_tt = upload(k_host);
    const ttnn::Tensor v_tt = upload(v_host);
    const ttnn::Tensor do_tt = upload(do_host);

    RunResult result;
    for (uint32_t iter = 0; iter <= cfg.iters; ++iter) {
        auto q = autograd::create_tensor(q_tt, /*requires_grad=*/true);
        auto k = autograd::create_tensor(k_tt, /*requires_grad=*/true);
        auto v = autograd::create_tensor(v_tt, /*requires_grad=*/true);

        tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
        tt::tt_metal::op_profiler::tracy_message(fmt::format("`TT_SIGNPOST: {}_s{}_fw_it{}`", name, seq, iter));
        const auto t0 = std::chrono::steady_clock::now();
        auto out = ops::distributed::ring_attention_sdpa(
            q,
            k,
            v,
            /*mask=*/std::nullopt,
            ttml::metal::AttentionMaskType::Causal,
            /*reuse_forward_kv=*/true,
            layout);
        tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
        const auto t1 = std::chrono::steady_clock::now();
        tt::tt_metal::op_profiler::tracy_message(fmt::format("`TT_SIGNPOST: {}_s{}_bw_it{}`", name, seq, iter));
        out->set_grad(do_tt);
        out->backward();
        tt::tt_metal::distributed::Synchronize(*device, std::nullopt);
        const auto t2 = std::chrono::steady_clock::now();

        if (iter > 0U) {
            result.fw_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            result.bw_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        if (cfg.check && iter == 0U) {
            result.out = to_natural_order(gather_ring(out->get_value(), cp_axis));
            result.dq = to_natural_order(gather_ring(q->get_grad(), cp_axis));
            result.dk = to_natural_order(gather_ring(k->get_grad(), cp_axis));
            result.dv = to_natural_order(gather_ring(v->get_grad(), cp_axis));
        }
        autograd::ctx().reset_graph();
    }
    if (cfg.profile_read) {
        tt::tt_metal::ReadMeshDeviceProfilerResults(*device);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    CLI::App app{"Ring attention benchmark: one attention layer fw+bw per CP layout"};
    app.add_option("--batch", cfg.batch, "Batch size");
    app.add_option("--seq", cfg.seqs, "Global sequence length(s), comma-separated to sweep")->delimiter(',');
    app.add_option("--heads", cfg.heads, "Query heads per device");
    app.add_option("--kv-heads", cfg.kv_heads, "Key/value heads per device");
    app.add_option("--head-dim", cfg.head_dim, "Head dimension");
    app.add_option("--iters", cfg.iters, "Timed iterations per layout (after one untimed warm-up)");
    app.add_option("--layouts", cfg.layouts, "Layouts to run: contiguous, zigzag")->delimiter(',');
    app.add_flag("--check", cfg.check, "Compare zig-zag against contiguous (outputs and gradients)");
    app.add_flag("--profile-read", cfg.profile_read, "Read device profiler results after every (S, layout)");
    CLI11_PARSE(app, argc, argv);

    auto& ctx = ttml::autograd::ctx();
    ctx.initialize_distributed_context(argc, argv);
    const auto mesh_shape = tt::tt_metal::distributed::SystemMesh::instance().shape();
    ttml::ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(mesh_shape.mesh_size()));
    ctx.open_device(mesh_shape);
    ctx.initialize_socket_manager(ttnn::distributed::SocketType::FABRIC);
    uint32_t non_trivial_axes = 0U;
    for (uint32_t i = 0; i < mesh_shape.dims(); ++i) {
        non_trivial_axes += mesh_shape[i] > 1U ? 1U : 0U;
    }
    ctx.initialize_parallelism_context({.enable_ddp = false, .enable_tp = non_trivial_axes > 1U, .enable_cp = true});
    const auto& pctx = ctx.get_parallelism_context();
    const uint32_t ring_size = pctx.get_cp_size();

    for (const uint32_t seq : cfg.seqs) {
        TT_FATAL(seq % (2U * ring_size * 32U) == 0U, "--seq {} must be a multiple of 2 * ring_size * 32", seq);
        fmt::print(
            "mesh {} cp_axis {} ring {} | B {} S {} (S_local {}) Hq {} Hkv {} D {} | iters {}\n",
            mesh_shape,
            pctx.get_cp_axis().value(),
            ring_size,
            cfg.batch,
            seq,
            seq / ring_size,
            cfg.heads,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.iters);

        const std::array<size_t, 4> q_shape{cfg.batch, cfg.heads, seq, cfg.head_dim};
        const std::array<size_t, 4> kv_shape{cfg.batch, cfg.kv_heads, seq, cfg.head_dim};
        const auto q_host = ttml::test_utils::make_uniform_xarray<float>(q_shape, -1.0F, 1.0F, 1U);
        const auto k_host = ttml::test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, 2U);
        const auto v_host = ttml::test_utils::make_uniform_xarray<float>(kv_shape, -1.0F, 1.0F, 3U);
        const auto do_host = ttml::test_utils::make_uniform_xarray<float>(q_shape, -1.0F, 1.0F, 4U);

        std::vector<std::pair<std::string, RunResult>> results;
        for (const std::string& name : cfg.layouts) {
            TT_FATAL(name == "contiguous" || name == "zigzag", "Unknown layout '{}'", name);
            const auto layout = name == "zigzag" ? RingAttentionLayout::ZigZag : RingAttentionLayout::Contiguous;
            auto result = run_layout(layout, name, seq, cfg, q_host, k_host, v_host, do_host);
            fmt::print(
                "[RING_ATTN_BENCH] layout={} S={} S_local={} fw_ms={:.2f} bw_ms={:.2f} fw_all={{{:.2f}}} "
                "bw_all={{{:.2f}}}\n",
                name,
                seq,
                seq / ring_size,
                median(result.fw_ms),
                median(result.bw_ms),
                fmt::join(result.fw_ms, ","),
                fmt::join(result.bw_ms, ","));
            results.emplace_back(name, std::move(result));
        }

        if (cfg.check && results.size() == 2U) {
            const auto& base = results[0].second;
            const auto& other = results[1].second;
            fmt::print(
                "[RING_ATTN_BENCH] check rel_l2({}, {}): out {:.2e} dQ {:.2e} dK {:.2e} dV {:.2e}\n",
                results[1].first,
                results[0].first,
                rel_l2(other.out, base.out),
                rel_l2(other.dq, base.dq),
                rel_l2(other.dk, base.dk),
                rel_l2(other.dv, base.dv));
        }
    }
    ctx.close_device();
    return 0;
}
