// SPDX-License-Identifier: Apache-2.0
// Fixed P150 batch-one Qwen3.5 GDN convolution, head expansion, and state shift.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#endif

namespace native_gdn {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kHalfRowBytes = 32;
constexpr uint32_t kMegaQBase = 0;
constexpr uint32_t kMegaKBase = 2048;
constexpr uint32_t kMegaVBase = 4096;
constexpr uint32_t kMegaATile = 256;
constexpr uint32_t kMegaBTile = 257;
constexpr uint32_t kInputCb = 0;
constexpr uint32_t kWeightCb = 1;
constexpr uint32_t kScratchCb = 2;
constexpr uint32_t kOutputCb = 16;
constexpr uint32_t kTempCb = 17;

FORCE_INLINE uint32_t tile_row_offset(uint32_t row, bool right) {
    const uint32_t face_row = row & 15;
    const uint32_t bottom = row >= 16 ? 1024 : 0;
    return bottom + (right ? 512 : 0) + face_row * kHalfRowBytes;
}

FORCE_INLINE uint32_t source_flat(uint32_t kind, uint32_t dim_tile, uint32_t row) {
    if (kind == 0) return kMegaQBase + (row >> 1) * 128 + dim_tile * 32;
    if (kind == 1) return kMegaKBase + (row >> 1) * 128 + dim_tile * 32;
    return kMegaVBase + row * 128 + dim_tile * 32;
}

FORCE_INLINE bool unique_state_row(uint32_t kind, uint32_t row) {
    return kind == 2 || (row & 1) == 0;
}
}  // namespace native_gdn

struct Core {
    static constexpr uint32_t output_kind = get_named_compile_time_arg_val("output_kind");
    static constexpr uint32_t dim_tile = get_named_compile_time_arg_val("dim_tile");
    static constexpr uint32_t num_tokens = get_named_compile_time_arg_val("num_tokens");
    static constexpr uint32_t token_index = get_named_compile_time_arg_val("token_index");
};

void kernel_main() {
    using namespace native_gdn;
#if defined(COMPILE_FOR_BRISC)
    const uint32_t mega_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tap_addrs[4] = {
        get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4)};
    const uint32_t bias_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t state_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t state_x[3] = {
        get_common_arg_val<uint32_t>(7), get_common_arg_val<uint32_t>(8), get_common_arg_val<uint32_t>(9)};
    const uint32_t state_y[3] = {
        get_common_arg_val<uint32_t>(10), get_common_arg_val<uint32_t>(11), get_common_arg_val<uint32_t>(12)};
    const uint32_t a_addr = get_common_arg_val<uint32_t>(13);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(14);
    const uint32_t output_addrs[3] = {
        get_common_arg_val<uint32_t>(15), get_common_arg_val<uint32_t>(16), get_common_arg_val<uint32_t>(17)};
    const uint32_t candidate_state_addrs[2] = {
        get_common_arg_val<uint32_t>(18), get_common_arg_val<uint32_t>(19)};

    constexpr bool mega_is_dram = get_named_compile_time_arg_val("mega_is_dram") == 1;
    constexpr bool tap_is_dram = get_named_compile_time_arg_val("tap_is_dram") == 1;
    constexpr bool bias_is_dram = get_named_compile_time_arg_val("bias_is_dram") == 1;
    constexpr bool output_is_dram = get_named_compile_time_arg_val("output_is_dram") == 1;
    constexpr bool candidate_is_dram = get_named_compile_time_arg_val("candidate_is_dram") == 1;
    const auto candidate_s1 = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<candidate_is_dram>(), candidate_state_addrs[0], 16384);
    const auto candidate_s2 = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<candidate_is_dram>(), candidate_state_addrs[1], 16384);
    const auto mega = TensorAccessor(tensor_accessor::make_interleaved_dspec<mega_is_dram>(), mega_addr, kTileBytes);
    const auto bias = TensorAccessor(tensor_accessor::make_interleaved_dspec<bias_is_dram>(), bias_addr, kTileBytes);
    const auto output = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<output_is_dram>(), output_addrs[Core::output_kind], kTileBytes);

    constexpr uint32_t token = Core::token_index;
    constexpr uint32_t packed_tile = Core::output_kind * 4 + Core::dim_tile;
    for (uint32_t tap = 0; tap < 4; ++tap) {
        const auto tap_accessor =
            TensorAccessor(tensor_accessor::make_interleaved_dspec<tap_is_dram>(), tap_addrs[tap], kTileBytes);
        cb_reserve_back(kInputCb, 1);
        cb_reserve_back(kWeightCb, 1);
        const uint32_t input_dst = get_write_ptr(kInputCb);
        const uint32_t weight_dst = get_write_ptr(kWeightCb);
        noc_async_read(get_noc_addr(packed_tile, tap_accessor), weight_dst, kTileBytes);
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t flat = source_flat(Core::output_kind, Core::dim_tile, row);
            uint64_t input_src;
            // Verification reads the immutable history followed by projected tokens.
            // Ordinary decode retains its in-place history shift.
            const uint32_t history_index = tap + token;
            if (history_index < 3) {
                input_src = get_noc_addr(state_x[history_index], state_y[history_index], state_addr + flat * sizeof(uint16_t));
                noc_async_read(input_src, input_dst + tile_row_offset(row, false), kHalfRowBytes);
                noc_async_read(input_src + kHalfRowBytes, input_dst + tile_row_offset(row, true), kHalfRowBytes);
            } else {
                const uint32_t tile = flat / 32;
                input_src = get_noc_addr(tile, mega);
                const uint32_t source_offset = tile_row_offset(history_index - 3, false);
                noc_async_read(input_src + source_offset, input_dst + tile_row_offset(row, false), kHalfRowBytes);
                noc_async_read(input_src + source_offset + 512, input_dst + tile_row_offset(row, true), kHalfRowBytes);
            }
        }
        noc_async_read_barrier();
        cb_push_back(kInputCb, 1);
        cb_push_back(kWeightCb, 1);

        // Verification writes prefix candidates only; decode also shifts live history.
        if (tap > 0) {
            for (uint32_t row = 0; row < 32; ++row) {
                if (!unique_state_row(Core::output_kind, row)) continue;
                const uint32_t flat = source_flat(Core::output_kind, Core::dim_tile, row);
                if constexpr (Core::num_tokens == 1) {
                    const uint64_t dst = get_noc_addr(state_x[tap - 1], state_y[tap - 1], state_addr + flat * sizeof(uint16_t));
                    noc_async_write(input_dst + tile_row_offset(row, false), dst, kHalfRowBytes);
                    noc_async_write(input_dst + tile_row_offset(row, true), dst + kHalfRowBytes, kHalfRowBytes);
                }
                if (candidate_state_addrs[token] != 0) {
                    const uint32_t candidate_row = tap - 1;
                    const uint32_t candidate_col = flat;
                    const uint64_t candidate_dst = token == 0
                        ? get_noc_addr(candidate_row, candidate_s1) + candidate_col * sizeof(uint16_t)
                        : get_noc_addr(candidate_row, candidate_s2) + candidate_col * sizeof(uint16_t);
                    noc_async_write(input_dst + tile_row_offset(row, false), candidate_dst, kHalfRowBytes);
                    noc_async_write(input_dst + tile_row_offset(row, true), candidate_dst + kHalfRowBytes, kHalfRowBytes);
                }
            }
            noc_async_write_barrier();
        }
    }

    // Bias is pre-expanded into one complete tile for every output core.
    cb_reserve_back(kWeightCb, 1);
    const uint32_t bias_dst = get_write_ptr(kWeightCb);
    noc_async_read(get_noc_addr(packed_tile, bias), bias_dst, kTileBytes);
    noc_async_read_barrier();
    cb_push_back(kWeightCb, 1);

    cb_wait_front(kOutputCb, 1);
    const uint32_t output_src = get_read_ptr(kOutputCb);
    const uint64_t output_dst = get_noc_addr(token * 4 + Core::dim_tile, output);
    noc_async_write(output_src, output_dst, kTileBytes);
    noc_async_write_barrier();
    cb_pop_front(kOutputCb, 1);

    // One worker extracts each token's a and b rows directly from mega output.
    if constexpr (Core::output_kind == 0 && Core::dim_tile == 0) {
        const uint32_t local = get_write_ptr(kInputCb);
        const uint32_t out_addrs[2] = {a_addr, b_addr};
        const uint32_t source_tiles[2] = {kMegaATile, kMegaBTile};
        for (uint32_t i = 0; i < 2; ++i) {
            const uint64_t source = get_noc_addr(source_tiles[i], mega);
            noc_async_read(source + tile_row_offset(token, false), local, kHalfRowBytes);
            noc_async_read(source + tile_row_offset(token, true), local + kHalfRowBytes, kHalfRowBytes);
            noc_async_read_barrier();
            const auto scalar_out = TensorAccessor(
                tensor_accessor::make_interleaved_dspec<output_is_dram>(), out_addrs[i], kTileBytes);
            const uint64_t destination = get_noc_addr(0, scalar_out);
            noc_async_write(local, destination + tile_row_offset(token, false), kHalfRowBytes);
            noc_async_write(local + kHalfRowBytes, destination + tile_row_offset(token, true), kHalfRowBytes);
            noc_async_write_barrier();
        }
    }
#elif defined(COMPILE_FOR_TRISC)
    unary_op_init_common(kInputCb, kScratchCb);
    cb_wait_front(kInputCb, 4);
    cb_wait_front(kWeightCb, 5);

    // ttnn.multiply uses the precise SFPU binary path for BF16. Preserve its
    // per-op BF16 boundaries and leave the baseline BF16 adds on the FPU.
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
    cb_reserve_back(kOutputCb, 1);
    tile_regs_acquire();
    copy_tile_init(kTempCb);
    copy_tile(kTempCb, 0, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format<true>(kOutputCb);
    pack_tile(0, kOutputCb);
    tile_regs_release();
    cb_pop_front(kTempCb, 1);
    cb_push_back(kOutputCb, 1);

    cb_pop_front(kInputCb, 4);
    cb_pop_front(kWeightCb, 5);
#endif
}
