// SPDX-License-Identifier: Apache-2.0
// Fused GDN core, output worker O(head) (unified BRISC / NCRISC / TRISC).
//
// Phase F (frontend): this core owns two of the 2*H + HV source heads (128 channels each) and computes their
// four-tap BF16 FIR + SiLU exactly as native_frontend_kernel.cpp (same elementwise tile ops; the two sources and T
// tokens share one pass: slot = source * T + token on tile rows 8 * slot + 2 * dim), advances their convolution
// history (T == 1: live L1 history; T >= 2: the DRAM prefix candidates), and sends each SiLU(conv) row straight into
// the L1 inbox of every recurrence worker that consumes it (q/k: the GQA value heads' two V-halves; v: its own
// V-half), then increments their arrival semaphore.
// Phase O (output): RMSNorm(o) * clamp(SiLU(gate)), clamped, exactly as native_output_kernel.cpp, where o arrives
// as the two V-halves' rows from the recurrence workers (semaphore 1) instead of a DRAM round trip. NCRISC stages
// gate/gamma/eps/scaler while phase F runs.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#endif

#if defined(COMPILE_FOR_TRISC)
// The stock layernorm compute first (as native_output_kernel.cpp includes it), then the frontend's compute APIs.
#include "api/compute/eltwise_unary/clamp.h"
#define kernel_main rmsnorm_kernel_main
#include "ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp"
#undef kernel_main
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#endif

namespace core35 {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kHalfRowBytes = 32;
constexpr uint32_t kWidthTiles = 4;
constexpr uint32_t kNumTokens = get_named_compile_time_arg_val("num_tokens");
constexpr uint32_t kQkvWidth = get_named_compile_time_arg_val("qkv_width");
constexpr uint32_t kHistoryPageBytes = kQkvWidth * sizeof(uint16_t);
constexpr uint32_t kKeyHeads = get_named_compile_time_arg_val("key_heads");
constexpr uint32_t kGqa = get_named_compile_time_arg_val("gqa");
constexpr uint32_t kValueHeads = get_named_compile_time_arg_val("value_heads");
constexpr uint32_t kSourcesPerCore = 2;
static_assert(kQkvWidth == (2 * kKeyHeads + kValueHeads) * 128);
static_assert(kKeyHeads * kGqa == kValueHeads);
static_assert(kNumTokens >= 1 && kNumTokens * kSourcesPerCore <= 4, "one frontend pass: <= 4 (source, token) slots");
// Recurrence inbox per token: q (4 dim rows), k (4), v (the V-half's 2), 64 B each; output inbox per token: the
// two V-halves' 2 rows each.
constexpr uint32_t kRecInboxTokenBytes = get_named_compile_time_arg_val("rec_inbox_token_bytes");
constexpr uint32_t kOutInboxTokenBytes = get_named_compile_time_arg_val("out_inbox_token_bytes");
// Frontend CBs (renumbered from native_frontend_kernel.cpp so they coexist with the output CBs).
constexpr uint32_t kInputCb = 17;
constexpr uint32_t kWeightCb = 18;
constexpr uint32_t kScratchCb = 19;
constexpr uint32_t kHistoryCb = 20;
constexpr uint32_t kFrontOutCb = 21;
constexpr uint32_t kTempCb = 22;
// Output CBs (native_output_kernel.cpp numbering).
constexpr uint32_t kX = 0;
constexpr uint32_t kGate = 1;
constexpr uint32_t kGamma = 2;
constexpr uint32_t kScaler = 3;
constexpr uint32_t kEps = 4;
constexpr uint32_t kGateClipped = 12;
constexpr uint32_t kRmsOutput = 8;
constexpr uint32_t kOutput = 16;

FORCE_INLINE uint32_t tile_row_offset(uint32_t row, bool right) {
    return (row >= 16 ? 1024 : 0) + (right ? 512 : 0) + (row & 15) * kHalfRowBytes;
}

}  // namespace core35

void kernel_main() {
    using namespace core35;
#if defined(COMPILE_FOR_BRISC)
    constexpr bool output_is_dram = get_named_compile_time_arg_val("output_is_dram") == 1;
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t mega_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t weights_addr = get_common_arg_val<uint32_t>(1);  // host-arranged FIR weight tiles, 5 per core
    const uint32_t state_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t state_x = get_common_arg_val<uint32_t>(7);
    const uint32_t state_y = get_common_arg_val<uint32_t>(8);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(9);
    const uint32_t inbox = get_common_arg_val<uint32_t>(10);
    const auto mega = TensorAccessor(tensor_accessor::make_interleaved_dspec<false>(), mega_addr, kTileBytes);
    const auto weights = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), weights_addr, kTileBytes);
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<output_is_dram>(), output_addr, kTileBytes);
    Noc noc;

    // ------------------------------------------------------------------ phase F
    constexpr uint32_t T = kNumTokens;
    constexpr uint32_t kRowBytes = 128 * sizeof(uint16_t);
    constexpr uint32_t kSources = 3 + T;  // per source head: three history rows, then one row per token
    const uint32_t rows_base = get_write_ptr(kHistoryCb);
    const uint32_t fcore = get_arg_val<uint32_t>(11);  // this core's slot in the arranged weight tensor

    uint32_t worker[kSourcesPerCore];
    uint32_t dests[kSourcesPerCore][4];
    for (uint32_t s = 0; s < kSourcesPerCore; ++s) {
        worker[s] = get_arg_val<uint32_t>(1 + s);
        for (uint32_t d = 0; d < 4; ++d) {
            dests[s][d] = get_arg_val<uint32_t>(3 + s * 4 + d);
        }
    }
    auto src_row = [&](uint32_t s, uint32_t index) { return rows_base + (s * kSources + index) * kRowBytes; };
    // FIR source row of (source s, token, tap): live history rows 0..2, then the token rows.
    auto source = [&](uint32_t s, uint32_t token, uint32_t tap) { return src_row(s, token + tap); };

    // The weight tiles arrive pre-arranged (taps 0..3, bias: row 8 * slot + 2 * dim holds that source's channels
    // 32 dim .. 32 dim + 31, zeros elsewhere) exactly as native_frontend_kernel.cpp assembles them; the input
    // tiles are zero-filled and their rows land by DMA.
    cb_reserve_back(kInputCb, 4);
    cb_reserve_back(kWeightCb, 5);
    const uint32_t input_dst = get_write_ptr(kInputCb);
    const uint32_t weight_dst = get_write_ptr(kWeightCb);
    noc.async_write_zeros(CircularBuffer(kInputCb), 4 * kTileBytes);
    {
    DeviceZoneScopedN("F-read");
    for (uint32_t p = 0; p < 5; ++p) {
        noc_async_read(get_noc_addr(fcore * 5 + p, weights), weight_dst + p * kTileBytes, kTileBytes);
    }
    for (uint32_t s = 0; s < kSourcesPerCore; ++s) {
        const uint32_t flat_head = worker[s] * 128;
        for (uint32_t row = 0; row < 3; ++row) {
            noc_async_read(get_noc_addr(state_x, state_y,
                state_addr + row * kHistoryPageBytes + flat_head * sizeof(uint16_t)), src_row(s, row), kRowBytes);
        }
        // Token rows straight from the mega row tiles (row `token` of face 0 / face 1 of each dim tile).
        for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
            const uint64_t token_src = get_noc_addr(flat_head / 32 + dim_tile, mega);
            for (uint32_t token = 0; token < T; ++token) {
                const uint32_t row = src_row(s, 3 + token) + dim_tile * 64;
                noc_async_read(token_src + token * kHalfRowBytes, row, kHalfRowBytes);
                noc_async_read(token_src + 512 + token * kHalfRowBytes, row + 32, kHalfRowBytes);
            }
        }
    }
    noc_async_read_barrier();
    noc.write_zeros_l1_barrier();
    }
    // Input tile rows by local DMA: tap t, slot (s, token), dim d -> row 8 * slot + 2 * d.
    for (uint32_t tap = 0; tap < 4; ++tap) {
        for (uint32_t s = 0; s < kSourcesPerCore; ++s) {
            for (uint32_t token = 0; token < T; ++token) {
                const uint32_t slot = s * T + token;
                const uint32_t input = source(s, token, tap);
                const uint32_t tile = input_dst + tap * kTileBytes;
                for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
                    const uint32_t row = 8 * slot + 2 * dim_tile;
                    noc_async_write(input + dim_tile * 64, get_noc_addr(tile + tile_row_offset(row, false)),
                                    kHalfRowBytes);
                    noc_async_write(input + dim_tile * 64 + 32, get_noc_addr(tile + tile_row_offset(row, true)),
                                    kHalfRowBytes);
                }
            }
        }
    }
    noc_async_write_barrier();
    cb_push_back(kInputCb, 4);
    cb_push_back(kWeightCb, 5);
    // History transition: rows 1..3 of each token's window become its new rows 0..2 (completed by the
    // scatter's write barrier below).
    for (uint32_t s = 0; s < kSourcesPerCore; ++s) {
        const uint32_t flat_head = worker[s] * 128;
        for (uint32_t token = 0; token < T; ++token) {
            for (uint32_t tap = 1; tap < 4; ++tap) {
                if constexpr (T == 1) {
                    noc_async_write(source(s, token, tap), get_noc_addr(state_x, state_y,
                        state_addr + (tap - 1) * kHistoryPageBytes + flat_head * sizeof(uint16_t)), kRowBytes);
                } else {
                    // Verification never mutates live history; retain every prefix.
                    const auto candidate = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                          get_common_arg_val<uint32_t>(11 + token), kHistoryPageBytes);
                    noc_async_write(source(s, token, tap),
                                    get_noc_addr(tap - 1, candidate) + flat_head * sizeof(uint16_t), kRowBytes);
                }
            }
        }
    }
    // Scatter SiLU(conv) rows into the consumers' inboxes.
    {
        DeviceZoneScopedN("F-wait");
        cb_wait_front(kFrontOutCb, 1);
    }
    const uint32_t front_src = get_read_ptr(kFrontOutCb);
    for (uint32_t s = 0; s < kSourcesPerCore; ++s) {
        const uint32_t w = worker[s];
        const uint32_t kind = w < kKeyHeads ? 0 : (w < 2 * kKeyHeads ? 1 : 2);
        const uint32_t ndest = kind < 2 ? 4 : 2;
        for (uint32_t token = 0; token < T; ++token) {
            const uint32_t slot = s * T + token;
            for (uint32_t dest = 0; dest < ndest; ++dest) {
                const uint32_t xy = dests[s][dest];
                // q/k: all four dim rows; v: the destination V-half's two rows (dest index = half).
                const uint32_t first_dim = kind < 2 ? 0 : 2 * dest;
                const uint32_t dims = kind < 2 ? 4 : 2;
                const uint32_t base = inbox + token * kRecInboxTokenBytes + kind * 4 * 64;
                for (uint32_t d = 0; d < dims; ++d) {
                    const uint32_t row = 8 * slot + 2 * (first_dim + d);
                    const uint64_t dst = get_noc_addr(xy & 0xffff, xy >> 16, base + d * 64);
                    noc_async_write(front_src + tile_row_offset(row, false), dst, kHalfRowBytes);
                    noc_async_write(front_src + tile_row_offset(row, true), dst + 32, kHalfRowBytes);
                }
            }
        }
    }
    noc_async_write_barrier();
    for (uint32_t s = 0; s < kSourcesPerCore; ++s) {
        const uint32_t w = worker[s];
        const uint32_t ndest = w < 2 * kKeyHeads ? 4 : 2;
        for (uint32_t dest = 0; dest < ndest; ++dest) {
            const uint32_t xy = dests[s][dest];
            noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, get_semaphore(0)), 1);
        }
    }
    cb_pop_front(kFrontOutCb, 1);

    // ------------------------------------------------------------------ phase O
    auto* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    {
        DeviceZoneScopedN("O-wait");
        noc_semaphore_wait(arrived, 2);
    }
    noc_semaphore_set(arrived, 0);
    cb_reserve_back(kX, kWidthTiles);
    const uint32_t x_dst = get_write_ptr(kX);
    noc.async_write_zeros(CircularBuffer(kX), kWidthTiles * kTileBytes);
    noc.write_zeros_l1_barrier();
    for (uint32_t tile = 0; tile < kWidthTiles; ++tile) {
        const uint32_t half = tile / 2;
        const uint32_t local = tile % 2;
        for (uint32_t token = 0; token < T; ++token) {
            const uint32_t src = inbox + token * kOutInboxTokenBytes + half * 128 + local * 64;
            noc_async_write(src, get_noc_addr(x_dst + tile * kTileBytes + tile_row_offset(token, false)), kHalfRowBytes);
            noc_async_write(src + 32, get_noc_addr(x_dst + tile * kTileBytes + tile_row_offset(token, true)),
                            kHalfRowBytes);
        }
    }
    noc_async_write_barrier();
    cb_push_back(kX, kWidthTiles);

    {
        DeviceZoneScopedN("O-wout");
        cb_wait_front(kOutput, kWidthTiles);
    }
    const uint32_t out_src = get_read_ptr(kOutput);
    for (uint32_t tile = 0; tile < kWidthTiles; ++tile) {
        const uint64_t dst = get_noc_addr(head * kWidthTiles + tile, output);
        if constexpr (T > 1) {
            // Packed token rows: only the tokens' rows of the computed tile belong to this op.
            for (uint32_t token = 0; token < T; ++token) {
                noc_async_write(out_src + tile * kTileBytes + tile_row_offset(token, false),
                                dst + tile_row_offset(token, false), kHalfRowBytes);
                noc_async_write(out_src + tile * kTileBytes + tile_row_offset(token, true),
                                dst + tile_row_offset(token, true), kHalfRowBytes);
            }
        } else {
            noc_async_write(out_src + tile * kTileBytes, dst, kTileBytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(kOutput, kWidthTiles);
    noc_async_atomic_barrier();

#elif defined(COMPILE_FOR_NCRISC)
    // Phase O operands that do not depend on the recurrence: eps, gate (mega row in place), gamma, reduce scaler.
    constexpr bool gamma_is_dram = get_named_compile_time_arg_val("gamma_is_dram") == 1;
    constexpr uint32_t gate_tile_offset = get_named_compile_time_arg_val("gate_tile_offset");
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(1);
    const auto gate = TensorAccessor(tensor_accessor::make_interleaved_dspec<false>(), gate_addr, kTileBytes);
    const auto gamma = TensorAccessor(tensor_accessor::make_interleaved_dspec<gamma_is_dram>(), gamma_addr, kTileBytes);

    generate_bcast_col_scalar(CircularBuffer(kEps), 0x358637bdu);  // 1e-6f
    cb_reserve_back(kGate, kWidthTiles);
    cb_reserve_back(kGamma, kWidthTiles);
    const uint32_t gate_dst = get_write_ptr(kGate);
    const uint32_t gamma_dst = get_write_ptr(kGamma);
    for (uint32_t tile = 0; tile < kWidthTiles; ++tile) {
        noc_async_read(get_noc_addr(gate_tile_offset + head * kWidthTiles + tile, gate), gate_dst + tile * kTileBytes,
                       kTileBytes);
        noc_async_read(get_noc_addr(tile, gamma), gamma_dst + tile * kTileBytes, kTileBytes);
    }
    noc_async_read_barrier();
    cb_push_back(kGate, kWidthTiles);
    cb_push_back(kGamma, kWidthTiles);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        kScaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();

#elif defined(COMPILE_FOR_TRISC)
    // ------------------------------------------------------------------ phase F (native_frontend_kernel.cpp)
    // ttnn.multiply uses the precise SFPU binary path for BF16. Preserve its per-op BF16 boundaries and leave the
    // baseline BF16 adds on the FPU. silu_tile/mul_binary_tile arithmetic and inits do not depend on APPROX.
    unary_op_init_common(kInputCb, kScratchCb);
    cb_wait_front(kInputCb, 4);
    cb_wait_front(kWeightCb, 5);

    mul_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(kWeightCb, kInputCb);
    copy_tile(kInputCb, 3, 0);
    copy_tile_to_dst_init_short_with_dt(kInputCb, kWeightCb);
    copy_tile(kWeightCb, 3, 1);
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    cb_reserve_back(kScratchCb, 1);
    tile_regs_wait();
    pack_reconfig_data_format<true>(kScratchCb);
    pack_tile(0, kScratchCb);
    tile_regs_release();
    cb_push_back(kScratchCb, 1);

    for (uint32_t tap = 0; tap < 3; ++tap) {
        mul_binary_tile_init();
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(kWeightCb, kInputCb);
        copy_tile(kInputCb, tap, 0);
        copy_tile_to_dst_init_short_with_dt(kInputCb, kWeightCb);
        copy_tile(kWeightCb, tap, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(kTempCb, 1);
        tile_regs_wait();
        pack_reconfig_data_format<true>(kTempCb);
        pack_tile(0, kTempCb);
        tile_regs_release();
        cb_push_back(kTempCb, 1);

        cb_wait_front(kScratchCb, 1);
        cb_wait_front(kTempCb, 1);
        tile_regs_acquire();
        add_tiles_init(kScratchCb, kTempCb);
        add_tiles(kScratchCb, kTempCb, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(kScratchCb, 1);
        cb_pop_front(kTempCb, 1);
        cb_reserve_back(kScratchCb, 1);
        pack_reconfig_data_format<true>(kScratchCb);
        pack_tile(0, kScratchCb);
        tile_regs_release();
        cb_push_back(kScratchCb, 1);
    }

    cb_wait_front(kScratchCb, 1);
    tile_regs_acquire();
    add_tiles_init(kScratchCb, kWeightCb);
    add_tiles(kScratchCb, kWeightCb, 0, 4, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_pop_front(kScratchCb, 1);
    cb_reserve_back(kTempCb, 1);
    pack_reconfig_data_format<true>(kTempCb);
    pack_tile(0, kTempCb);
    tile_regs_release();
    cb_push_back(kTempCb, 1);

    cb_wait_front(kTempCb, 1);
    cb_reserve_back(kFrontOutCb, 1);
    tile_regs_acquire();
    copy_tile_init(kTempCb);
    copy_tile(kTempCb, 0, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format<true>(kFrontOutCb);
    pack_tile(0, kFrontOutCb);
    tile_regs_release();
    cb_pop_front(kTempCb, 1);
    cb_push_back(kFrontOutCb, 1);

    cb_pop_front(kInputCb, 4);
    cb_pop_front(kWeightCb, 5);

    // ------------------------------------------------------------------ phase O (native_output_kernel.cpp)
    // The gate does not depend on the recurrence: SiLU + clamp it while the recurrence runs (elementwise,
    // every SFPU op re-initialized before use, so the order relative to the RMSNorm does not change any value).
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

    {
    DeviceZoneScopedN("O-norm");
    rmsnorm_kernel_main();
    }
    cb_pop_front(kGamma, kWidthTiles);


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
    cb_pop_front(kEps, 1);
#endif
}
