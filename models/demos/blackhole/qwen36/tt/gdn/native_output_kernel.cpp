// SPDX-License-Identifier: Apache-2.0
// P150 batch-one GDN RMSNorm, gate, and projection-layout preparation.
// One worker per value head; value_heads is compile-time (27B: 48).
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
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
    static constexpr uint32_t value_heads = get_named_compile_time_arg_val("value_heads");
    // head_major: recurrent is [HV, 1, 128] (one tile row per head, T == 1) as emitted
    // by the recurrence; otherwise [T, 1, HV, 128] with heads filling tile rows.
    static constexpr bool head_major = get_named_compile_time_arg_val("head_major") == 1;
    static constexpr uint32_t recurrent_token_tiles = ((value_heads + 31) / 32) * 4;
    // Gate/output rows are HV*128 wide: 4 tiles per head.
    static constexpr uint32_t gate_token_tiles = value_heads * 4;
    // gate_tile_offset: the gate is read in place from a wider row (e.g. the mega
    // projection), starting this many tiles in; gate_row_tiles is that row's width.
    static constexpr uint32_t gate_tile_offset = get_named_compile_time_arg_val("gate_tile_offset");
    static constexpr uint32_t gate_row_tiles = get_named_compile_time_arg_val("gate_row_tiles");
    // Packed rows share every tile: the row-wise norm, gate and write run once for all
    // tokens (each token on its own tile row). Otherwise one pass per token.
    static constexpr uint32_t passes = packed_rows ? 1 : num_tokens;
    static constexpr uint32_t pass_rows = packed_rows ? num_tokens : 1;
};


#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_unary/clamp.h"
#define kernel_main rmsnorm_kernel_main
#include "ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp"
#undef kernel_main
#endif

void kernel_main() {
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
    // slot_rows: recurrent already has the gate/output layout, so its tiles are read whole.
    constexpr bool slot_rows = get_named_compile_time_arg_val("slot_rows") == 1;

    const auto recurrent = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<recurrent_is_dram>(), recurrent_addr, kTileBytes);
    const auto gate = TensorAccessor(tensor_accessor::make_interleaved_dspec<gate_is_dram>(), gate_addr, kTileBytes);
    const auto gamma = TensorAccessor(tensor_accessor::make_interleaved_dspec<gamma_is_dram>(), gamma_addr, kTileBytes);
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<output_is_dram>(), output_addr, kTileBytes);

    generate_bcast_col_scalar(CircularBuffer(kEps), 0x358637bdu);  // 1e-6f

    for (uint32_t pass = 0; pass < Core::passes; ++pass) {
    cb_reserve_back(kX, kWidthTiles);
    cb_reserve_back(kGate, kWidthTiles);
    cb_reserve_back(kGamma, kWidthTiles);
    const uint32_t x_dst = get_write_ptr(kX);
    const uint32_t gate_dst = get_write_ptr(kGate);
    const uint32_t gamma_dst = get_write_ptr(kGamma);
    Noc noc;
    if constexpr (!slot_rows) {
        noc.async_write_zeros(CircularBuffer(kX), kWidthTiles * kTileBytes);
        // The gathered rows are written by BRISC after the entire tile padding is zero.
        noc.write_zeros_l1_barrier();
    }
    const uint32_t gate_tile_base = Core::gate_tile_offset + head * kWidthTiles
        + (Core::packed_rows ? 0 : pass * Core::gate_row_tiles);
    for (uint32_t tile = 0; tile < kWidthTiles; ++tile) {
        const uint32_t local = get_write_ptr(11);
        const uint32_t tile_dst = x_dst + tile * kTileBytes;
        if constexpr (slot_rows) {
            noc_async_read(get_noc_addr(head * kWidthTiles + tile, recurrent), tile_dst, kTileBytes);
        }
        // Row-wise norm and gate multiply: each recurrent row must land on the same
        // tile row as its gate row (row `token` when packed, row 0 otherwise).
        for (uint32_t r = 0; r < (slot_rows ? 0 : Core::pass_rows); ++r) {
            const uint32_t token = pass + r;
            const uint32_t x_row = Core::packed_rows ? token : 0;
            const uint64_t recurrent_src = Core::head_major
                ? get_noc_addr(head * kWidthTiles + tile, recurrent)
                : get_noc_addr(token * Core::recurrent_token_tiles + (head / 32) * kWidthTiles + tile, recurrent);
            const uint32_t recurrent_row = Core::head_major ? 0 : head % 32;
            const uint32_t source_row = recurrent_row & ~1u;
            noc_async_read(recurrent_src + tile_row_offset(source_row, false), local, 64);
            noc_async_read(recurrent_src + tile_row_offset(source_row, true), local + 64, 64);
            noc_async_read_barrier();
            for (uint32_t half = 0; half < 2; ++half) {
                const volatile uint32_t* src = reinterpret_cast<const volatile uint32_t*>(local + half * 64 + (recurrent_row & 1) * 32);
                volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(tile_dst + tile_row_offset(x_row, half != 0));
                for (uint32_t i = 0; i < kHalfRowBytes / 4; ++i) dst[i] = src[i];
            }
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
            // Only the tokens' rows of the computed tile belong to this op; the other
            // rows of the destination tile are padding.
            const uint64_t dst = get_noc_addr(head * kWidthTiles + tile, output);
            for (uint32_t token = 0; token < Core::num_tokens; ++token) {
                noc_async_write(out_src + tile * kTileBytes + tile_row_offset(token, false),
                                dst + tile_row_offset(token, false), kHalfRowBytes);
                noc_async_write(out_src + tile * kTileBytes + tile_row_offset(token, true),
                                dst + tile_row_offset(token, true), kHalfRowBytes);
            }
        } else {
            noc_async_write(out_src + tile * kTileBytes,
                            get_noc_addr(pass * Core::gate_token_tiles + head * kWidthTiles + tile, output), kTileBytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(kOutput, kWidthTiles);
    }
#elif defined(COMPILE_FOR_TRISC)
    for (uint32_t pass = 0; pass < Core::passes; ++pass) {
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
