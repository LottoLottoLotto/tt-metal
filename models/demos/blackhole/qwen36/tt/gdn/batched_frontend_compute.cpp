// SPDX-License-Identifier: Apache-2.0
// Batched GDN frontend compute: per QKV column tile (32 channels x every slot row),
// the B1 native frontend's four-tap FIR, bias, and SiLU with its exact op sequence and
// BF16 boundaries. Only the element placement differs: slots fill the tile rows, so the
// taps and bias (row 0) are row-broadcast into full tiles first. With at most 16 slots
// the SFPU work (products, SiLU) covers only the upper faces; lower rows are unused.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"

namespace {
constexpr uint32_t kRows = 0, kX = 1, kParams = 2, kWeights = 3, kHistory = 4;
constexpr uint32_t kScratch = 5, kTemp = 6, kOutput = 16;
constexpr VectorMode kSlotRows = get_compile_time_arg_val(0) <= 16 ? VectorMode::R : VectorMode::RC;

void multiply(uint32_t input_cb, uint32_t input_tile, uint32_t weight_tile, uint32_t out) {
    mul_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(kWeights, input_cb);
    copy_tile(input_cb, input_tile, 0);
    copy_tile_to_dst_init_short_with_dt(input_cb, kWeights);
    copy_tile(kWeights, weight_tile, 1);
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE), 0, 1, 0, kSlotRows)));
    tile_regs_commit();
    cb_reserve_back(out, 1);
    tile_regs_wait();
    pack_reconfig_data_format<true>(out);
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
}

void add(uint32_t a, uint32_t b, uint32_t b_tile, uint32_t out) {
    cb_wait_front(a, 1);
    tile_regs_acquire();
    add_tiles_init(a, b);
    add_tiles(a, b, 0, b_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_pop_front(a, 1);
    if (b != kWeights) cb_pop_front(b, 1);
    cb_reserve_back(out, 1);
    pack_reconfig_data_format<true>(out);
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
}
}  // namespace

void kernel_main() {
    const uint32_t tiles = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(kParams, kWeights);
    // Weight tile p * tiles + i: tap p (4 = bias) of column tile i, on every row. The
    // writer fetches these in parallel with the reader's history rows.
    cb_wait_front(kParams, 5 * tiles);
    cb_reserve_back(kWeights, 5 * tiles);
    unary_bcast_init<BroadcastType::ROW>(kParams, kWeights);
    for (uint32_t tile = 0; tile < 5 * tiles; ++tile) {
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(kParams, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kWeights);
        tile_regs_release();
    }
    unary_bcast_uninit<BroadcastType::ROW>(kParams);
    cb_push_back(kWeights, 5 * tiles);
    cb_pop_front(kParams, 5 * tiles);

    tilize_init(kRows, tiles, kHistory);
    for (uint32_t tap = 0; tap < 3; ++tap) {
        cb_wait_front(kRows, tiles);
        cb_reserve_back(kHistory, tiles);
        tilize_block(kRows, tiles, kHistory);
        cb_push_back(kHistory, tiles);
        cb_pop_front(kRows, tiles);
    }
    tilize_uninit(kRows, kHistory);

    cb_wait_front(kHistory, 3 * tiles);
    cb_wait_front(kX, tiles);
    cb_wait_front(kWeights, 5 * tiles);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        // Tap 3 is the current token (mega row), taps 0..2 the tilized history rows.
        unary_op_init_common(kX, kScratch);
        multiply(kX, tile, 3 * tiles + tile, kScratch);
        for (uint32_t tap = 0; tap < 3; ++tap) {
            multiply(kHistory, tap * tiles + tile, tap * tiles + tile, kTemp);
            cb_wait_front(kTemp, 1);
            add(kScratch, kTemp, 0, kScratch);
        }
        add(kScratch, kWeights, 4 * tiles + tile, kTemp);

        cb_wait_front(kTemp, 1);
        cb_reserve_back(kOutput, 1);
        tile_regs_acquire();
        copy_tile_init(kTemp);
        copy_tile(kTemp, 0, 0);
        silu_tile_init();
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, 8), 0, kSlotRows));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format<true>(kOutput);
        pack_tile(0, kOutput);
        tile_regs_release();
        cb_pop_front(kTemp, 1);
        cb_push_back(kOutput, 1);
    }
    cb_pop_front(kHistory, 3 * tiles);
    cb_pop_front(kX, tiles);
    cb_pop_front(kWeights, 5 * tiles);
}
