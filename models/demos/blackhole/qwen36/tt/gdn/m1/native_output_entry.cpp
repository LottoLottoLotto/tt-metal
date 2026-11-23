// SPDX-License-Identifier: Apache-2.0
// Fixed P150 batch-one GDN RMSNorm, gate, and projection-layout preparation.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#endif

namespace native_output {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kHalfRowBytes = 32;
constexpr uint32_t kWidthTiles = 4;
constexpr uint32_t kX = 0;
constexpr uint32_t kGate = 1;
constexpr uint32_t kGamma = 2;
constexpr uint32_t kScaler = 3;
constexpr uint32_t kEps = 4;
constexpr uint32_t kGateClipped = 12;
constexpr uint32_t kRmsOutput = 8;
constexpr uint32_t kOutput = 16;

FORCE_INLINE uint32_t tile_row_offset(uint32_t row, bool right) {
    const uint32_t face_row = row & 15;
    const uint32_t bottom = row >= 16 ? 1024 : 0;
    return bottom + (right ? 512 : 0) + face_row * kHalfRowBytes;
}
}  // namespace native_output
struct Core {
    static constexpr uint32_t num_tokens = get_named_compile_time_arg_val("num_tokens");
    // packed_rows: gate/output hold every token as one row of a single tile row
    // ([1, T, 6144] in tile layout) instead of one tile row per token ([T, 1, 6144]).
    static constexpr bool packed_rows = get_named_compile_time_arg_val("packed_rows") == 1;
};


#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_unary/clamp.h"
#define kernel_main rmsnorm_kernel_main
#include "ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp"
#undef kernel_main
#endif

void bulk_native_output_main() {
    using namespace native_output;
#if defined(COMPILE_FOR_BRISC)
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t recurrent_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(3);
    constexpr bool recurrent_is_dram = get_named_compile_time_arg_val("recurrent_is_dram") == 1;
    constexpr bool gate_is_dram = get_named_compile_time_arg_val("gate_is_dram") == 1;
    constexpr bool gamma_is_dram = get_named_compile_time_arg_val("gamma_is_dram") == 1;
    constexpr bool output_is_dram = get_named_compile_time_arg_val("output_is_dram") == 1;

    const auto recurrent = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<recurrent_is_dram>(), recurrent_addr, kTileBytes);
    const auto gate = TensorAccessor(tensor_accessor::make_interleaved_dspec<gate_is_dram>(), gate_addr, kTileBytes);
    const auto gamma = TensorAccessor(tensor_accessor::make_interleaved_dspec<gamma_is_dram>(), gamma_addr, kTileBytes);
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<output_is_dram>(), output_addr, kTileBytes);

    generate_bcast_col_scalar(CircularBuffer(kEps), 0x358637bdu);  // 1e-6f

    for (uint32_t token = 0; token < Core::num_tokens; ++token) {
    cb_reserve_back(kX, kWidthTiles);
    cb_reserve_back(kGate, kWidthTiles);
    cb_reserve_back(kGamma, kWidthTiles);
    const uint32_t x_dst = get_write_ptr(kX);
    const uint32_t gate_dst = get_write_ptr(kGate);
    const uint32_t gamma_dst = get_write_ptr(kGamma);
    const uint32_t gate_tile_base = Core::packed_rows ? head * kWidthTiles : token * 192 + head * kWidthTiles;
    // Row-wise norm and gate multiply: the recurrent row must land on the same
    // tile row as its gate row (row `token` when packed, row 0 otherwise).
    const uint32_t x_row = Core::packed_rows ? token : 0;
    for (uint32_t tile = 0; tile < kWidthTiles; ++tile) {
        const uint32_t local = get_write_ptr(11);
        const uint32_t tile_dst = x_dst + tile * kTileBytes;
        volatile uint32_t* zeros = reinterpret_cast<volatile uint32_t*>(tile_dst);
        for (uint32_t i = 0; i < kTileBytes / 4; ++i) zeros[i] = 0;
        const uint64_t recurrent_src = get_noc_addr(token * 8 + (head / 32) * kWidthTiles + tile, recurrent);
        const uint32_t source_row = (head % 32) & ~1u;
        noc_async_read(recurrent_src + tile_row_offset(source_row, false), local, 64);
        noc_async_read(recurrent_src + tile_row_offset(source_row, true), local + 64, 64);
        noc_async_read_barrier();
        for (uint32_t half = 0; half < 2; ++half) {
            const volatile uint32_t* src = reinterpret_cast<const volatile uint32_t*>(local + half * 64 + (head & 1) * 32);
            volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(tile_dst + tile_row_offset(x_row, half != 0));
            for (uint32_t i = 0; i < kHalfRowBytes / 4; ++i) dst[i] = src[i];
        }
        noc_async_read(get_noc_addr(gate_tile_base + tile, gate), gate_dst + tile * kTileBytes, kTileBytes);
        noc_async_read(get_noc_addr(tile, gamma), gamma_dst + tile * kTileBytes, kTileBytes);
    }
    noc_async_read_barrier();
    cb_push_back(kX, kWidthTiles);
    cb_push_back(kGate, kWidthTiles);
    cb_push_back(kGamma, kWidthTiles);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        kScaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();

    cb_wait_front(kOutput, kWidthTiles);
    const uint32_t out_src = get_read_ptr(kOutput);
    for (uint32_t tile = 0; tile < kWidthTiles; ++tile) {
        if constexpr (Core::packed_rows) {
            // Only this token's row of the computed tile belongs to it; the other
            // rows of the packed destination tile are owned by other tokens.
            const uint64_t dst = get_noc_addr(head * kWidthTiles + tile, output);
            noc_async_write(out_src + tile * kTileBytes + tile_row_offset(token, false),
                            dst + tile_row_offset(token, false), kHalfRowBytes);
            noc_async_write(out_src + tile * kTileBytes + tile_row_offset(token, true),
                            dst + tile_row_offset(token, true), kHalfRowBytes);
        } else {
            noc_async_write(out_src + tile * kTileBytes,
                            get_noc_addr(token * 192 + head * kWidthTiles + tile, output), kTileBytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(kOutput, kWidthTiles);
    }
#elif defined(COMPILE_FOR_TRISC)
    for (uint32_t token = 0; token < Core::num_tokens; ++token) {
#ifdef OUTPUT_BYPASS
    cb_wait_front(kGate, kWidthTiles);
    cb_reserve_back(kOutput, kWidthTiles);
    for (uint32_t tile = 0; tile < kWidthTiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(kGate);
        copy_tile(kGate, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format<true>(kOutput);
        pack_tile(0, kOutput);
        tile_regs_release();
    }
    cb_push_back(kOutput, kWidthTiles);
    cb_pop_front(kGate, kWidthTiles);
#else
    rmsnorm_kernel_main();
    cb_pop_front(kGamma, kWidthTiles);

    cb_wait_front(kGate, kWidthTiles);
    cb_reserve_back(kGateClipped, kWidthTiles);
    for (uint32_t tile = 0; tile < kWidthTiles; ++tile) {
        tile_regs_acquire();
        copy_tile_init(kGate);
        copy_tile(kGate, tile, 0);
        silu_tile_init();
        silu_tile(0);
        clamp_tile_init();
        clamp_tile(0, 0xc61c4000u, 0x461c4000u);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format<true>(kGateClipped);
        pack_tile(0, kGateClipped);
        tile_regs_release();
    }
    cb_push_back(kGateClipped, kWidthTiles);

    cb_wait_front(kRmsOutput, kWidthTiles);
    cb_wait_front(kGateClipped, kWidthTiles);
    cb_reserve_back(kOutput, kWidthTiles);
    for (uint32_t tile = 0; tile < kWidthTiles; ++tile) {
        mul_binary_tile_init();
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(kRmsOutput, kRmsOutput);
        copy_tile(kRmsOutput, tile, 0);
        copy_tile_to_dst_init_short_with_dt(kRmsOutput, kGateClipped);
        copy_tile(kGateClipped, tile, 1);
        mul_binary_tile(0, 1, 0);
        clamp_tile_init();
        clamp_tile(0, 0xc61c4000u, 0x461c4000u);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format<true>(kOutput);
        pack_tile(0, kOutput);
        tile_regs_release();
    }
    cb_push_back(kOutput, kWidthTiles);
    cb_pop_front(kRmsOutput, kWidthTiles);
    cb_pop_front(kGateClipped, kWidthTiles);
    cb_pop_front(kGate, kWidthTiles);
#endif
    }
    cb_pop_front(kEps, 1);
#endif
}
