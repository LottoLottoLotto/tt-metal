// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Inverse gather of the FP32 routed outputs with one final BF16 rounding.
// Work items are (32-token tile row, `chunk`-tile column block). For every selected slot
// the reader gathers each token's contiguous row segment from the row-major projections,
// forming a [32, chunk*32] row-major FP32 block. Summation is elementwise, so compute adds
// the slots' blocks as linear 1024-datum pieces directly (FP32 DEST, slot order), rounds
// once to BF16, and tilizes the BF16 block. Row segments keep the DRAM transaction count
// far below per-face-row gathers, which bounded this stage.
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#endif

constexpr uint32_t cb_in = 0, cb_inverse = 1, cb_sum = 2, cb_out = 16;

void kernel_main() {
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t selected = get_named_compile_time_arg_val("selected");
    constexpr uint32_t chunk = get_named_compile_time_arg_val("chunk");
    constexpr uint32_t chunks = width_tiles / chunk;
    constexpr uint32_t items = ((tokens + 31) / 32) * chunks;
    static_assert(width_tiles % chunk == 0);
    const uint32_t worker = get_arg_val<uint32_t>(3);
    const uint32_t workers = get_arg_val<uint32_t>(4);
    // Contiguous item ranges: a worker mostly stays within one tile row.
    const uint32_t begin = worker * items / workers, end = (worker + 1) * items / workers;
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    constexpr auto packed_args = TensorAccessorArgs<0>();
    constexpr auto inverse_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<inverse_args.next_compile_time_args_offset()>();
#endif
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t segment = chunk * 32 * 4;  // one token's FP32 columns in the block
    const auto packed = TensorAccessor(packed_args, get_arg_val<uint32_t>(0), width_tiles * 32 * 4);
    const auto inverse = TensorAccessor(inverse_args, get_arg_val<uint32_t>(1), 128);
    const uint32_t inverse_base = get_write_ptr(cb_inverse);
    auto* indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inverse_base);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t last_tile_row = 0xffffffffu;
    for (uint32_t item = begin; item < end; ++item) {
        const uint32_t tile_row = item / chunks;
        const uint32_t column = (item % chunks) * segment;
        const uint32_t valid = tokens - tile_row * 32 < 32 ? tokens - tile_row * 32 : 32;
        if (tile_row != last_tile_row) {
            for (uint32_t r = 0; r < valid; ++r) {
                noc_async_read(get_noc_addr(tile_row * 32 + r, inverse), inverse_base + r * 128, 128);
            }
            noc_async_read_barrier();
            last_tile_row = tile_row;
        }
        cb_reserve_back(cb_in, selected * chunk);
        const uint32_t base = get_write_ptr(cb_in);
        for (uint32_t s = 0; s < selected; ++s) {
            const uint32_t block = base + s * chunk * 4096;
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t index = r < valid ? indices[r * 32 + s] : 0xffffffffu;
                const uint32_t dst = block + r * segment;
                if (index == 0xffffffffu) {
                    // Rows past the tokens, and masked-prefill padding rows, sum to zero.
                    for (uint32_t off = 0; off < segment; off += MEM_ZEROS_SIZE) {
                        noc_async_read(zeros, dst + off, MEM_ZEROS_SIZE);
                    }
                } else {
                    noc_async_read(get_noc_addr(index, packed) + column, dst, segment);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, selected * chunk);
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(2), 2048);
    for (uint32_t item = begin; item < end; ++item) {
        const uint32_t first = (item / chunks) * width_tiles + (item % chunks) * chunk;
        cb_wait_front(cb_out, chunk);
        for (uint32_t c = 0; c < chunk; ++c) {
            noc_async_write(get_read_ptr(cb_out) + c * 2048, get_noc_addr(first + c, out), 2048);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, chunk);
    }
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    unary_op_init_common(cb_in, cb_sum);
    for (uint32_t item = begin; item < end; ++item) {
        cb_wait_front(cb_in, selected * chunk);
        cb_reserve_back(cb_sum, chunk);
        pack_reconfig_data_format(cb_sum);
        for (uint32_t p = 0; p < chunk; ++p) {
            tile_regs_acquire();
            add_binary_tile_init();
            for (uint32_t s = 0; s < selected; ++s) {
                // CB0 unpacks directly to DST at full FP32 precision (host unpack mode).
                copy_tile_to_dst_init_short_with_dt(cb_sum, cb_in);
                copy_tile(cb_in, s * chunk + p, s == 0 ? 0 : 1);
                if (s != 0) add_binary_tile(0, 1, 0);
            }
            // Match the explicit RNE primitive already used by the native Qwen kernels.
            typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
            typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_sum, p);
            tile_regs_release();
        }
        cb_push_back(cb_sum, chunk);
        cb_pop_front(cb_in, selected * chunk);
        // The summed block is row-major BF16 [32, chunk*32]; tilize it into output tiles.
        compute_kernel_lib::tilize<chunk, cb_sum, cb_out>(1);
    }
#endif
}
