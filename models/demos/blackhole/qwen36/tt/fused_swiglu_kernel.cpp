// SPDX-License-Identifier: Apache-2.0
// One padded tile row: 768 input tiles [up|gate], 384 output tiles, 96 workers.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_buffer.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#endif

namespace fused_swiglu {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kHalfTiles = 384;
constexpr uint32_t kTiles = 4;
constexpr uint32_t kGate = 0;
constexpr uint32_t kUp = 1;
constexpr uint32_t kSilu = 2;
constexpr uint32_t kOutput = 16;
}  // namespace fused_swiglu

void kernel_main() {
    using namespace fused_swiglu;
#if defined(COMPILE_FOR_NCRISC)
    const uint32_t first = get_arg_val<uint32_t>(0);
    const auto input = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<false>(), get_common_arg_val<uint32_t>(0), kTileBytes);
    cb_reserve_back(kGate, kTiles);
    cb_reserve_back(kUp, kTiles);
    const uint32_t gate_dst = get_write_ptr(kGate);
    const uint32_t up_dst = get_write_ptr(kUp);
    for (uint32_t tile = 0; tile < kTiles; ++tile) {
        noc_async_read(get_noc_addr(first + tile + kHalfTiles, input), gate_dst + tile * kTileBytes, kTileBytes);
        noc_async_read(get_noc_addr(first + tile, input), up_dst + tile * kTileBytes, kTileBytes);
    }
    noc_async_read_barrier();
    cb_push_back(kGate, kTiles);
    cb_push_back(kUp, kTiles);
#elif defined(COMPILE_FOR_BRISC)
    const uint32_t first = get_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<false>(), get_common_arg_val<uint32_t>(0), kTileBytes);
    cb_wait_front(kOutput, kTiles);
    const uint32_t src = get_read_ptr(kOutput);
    for (uint32_t tile = 0; tile < kTiles; ++tile) {
        noc_async_write(src + tile * kTileBytes, get_noc_addr(first + tile, output), kTileBytes);
    }
    noc_async_write_barrier();
    cb_pop_front(kOutput, kTiles);
#elif defined(COMPILE_FOR_TRISC)
    // Exactly the standalone unary SFPU path. Do NOT retain the unrounded SiLU
    // result in DST for multiplication: the intervening BF16 pack is observable.
    init_sfpu(kGate, kSilu);
    cb_wait_front(kGate, kTiles);
    cb_reserve_back(kSilu, kTiles);
    tile_regs_acquire();
    for (uint32_t tile = 0; tile < kTiles; ++tile) {
        copy_tile(kGate, tile, tile);
        silu_tile_init();
        silu_tile(tile);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t tile = 0; tile < kTiles; ++tile) {
        pack_tile(tile, kSilu);
    }
    tile_regs_release();
    cb_push_back(kSilu, kTiles);
    cb_pop_front(kGate, kTiles);

    // Default ttnn.mul is SFPU binary_ng. kSilu/kUp use UnpackToDestFp32,
    // even though fp32_dest_acc_en is false, matching its BF16 configuration.
    unary_op_init_common(kSilu, kOutput);
    mul_binary_tile_init();
    cb_wait_front(kSilu, kTiles);
    cb_wait_front(kUp, kTiles);
    cb_reserve_back(kOutput, kTiles);
    // The interleaved binary_ng path acquires DST for one pair at a time;
    // its factory warns that four SFPU pairs per cycle do not work.
    for (uint32_t tile = 0; tile < kTiles; ++tile) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(kUp, kSilu);
        copy_tile(kSilu, tile, 0);
        copy_tile_to_dst_init_short_with_dt(kSilu, kUp);
        copy_tile(kUp, tile, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kOutput);
        tile_regs_release();
    }
    cb_push_back(kOutput, kTiles);
    cb_pop_front(kSilu, kTiles);
    cb_pop_front(kUp, kTiles);
#endif
}
