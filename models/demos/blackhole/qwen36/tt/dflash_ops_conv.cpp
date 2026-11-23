// SPDX-License-Identifier: Apache-2.0
// Two causal taps, 16 channels/group, with three BF16 rounding boundaries/tap.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/typecast.h"
#endif

namespace {
constexpr uint32_t bytes = 2048;
constexpr uint32_t workers = 64;
constexpr uint32_t width_tiles = DFLASH_HIDDEN / 32;
constexpr uint32_t groups = DFLASH_HIDDEN / 16;
constexpr uint32_t rows = DFLASH_ROWS;
constexpr uint32_t branch = DFLASH_BRANCH;
constexpr uint32_t offset(uint32_t r, uint32_t c) {
    return (r / 16) * 512 + (c / 16) * 256 + (r % 16) * 16 + c % 16;
}
#if defined(COMPILE_FOR_TRISC)
void load(uint32_t cb, uint32_t dst) {
    copy_tile_to_dst_init_short_with_dt(0, cb);
    copy_tile(cb, 0, dst);
}
void publish(uint32_t cb) {
    // The raw packer/SFPU convert rounds BF16 ties differently from PyTorch.
    // Use the existing explicit round-to-nearest-even FP32->BF16 primitive.
    typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
    typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}
#endif
}

void kernel_main() {
    const uint32_t worker = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_NCRISC)
    const auto hidden = TensorAccessor(tensor_accessor::make_interleaved_dspec<DFLASH_HIDDEN_DRAM>(), get_common_arg_val<uint32_t>(0), bytes);
    const auto base = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), bytes);
    const auto dynamic = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), bytes);
    const uint32_t scratch = get_write_ptr(3);
    auto h = reinterpret_cast<volatile uint16_t*>(scratch);
    auto b = reinterpret_cast<volatile uint16_t*>(scratch + bytes);
    auto d = reinterpret_cast<volatile uint16_t*>(scratch + 2 * bytes);
    for (uint32_t tile = worker; tile < width_tiles; tile += workers) {
        noc_async_read(get_noc_addr(tile, hidden), scratch, bytes);
        noc_async_read(get_noc_addr(tile, base), scratch + bytes, bytes);
        noc_async_read_barrier();
        cb_reserve_back(7, 1);
        auto zero = reinterpret_cast<volatile uint32_t*>(get_write_ptr(7));
        for (uint32_t i = 0; i < bytes / 4; ++i) zero[i] = 0;
        cb_push_back(7, 1);
        for (uint32_t tap = 0; tap < 2; ++tap) {
            const uint32_t group = tile * 2;
            const uint32_t column = (branch * 2 + tap) * groups + group;
            noc_async_read(get_noc_addr(column / 32, dynamic), scratch + 2 * bytes, bytes);
            noc_async_read_barrier();
            cb_reserve_back(0, 1);
            cb_reserve_back(1, 1);
            cb_reserve_back(2, 1);
            auto values = reinterpret_cast<volatile uint16_t*>(get_write_ptr(0));
            auto kernels = reinterpret_cast<volatile uint16_t*>(get_write_ptr(1));
            auto coefficients = reinterpret_cast<volatile uint16_t*>(get_write_ptr(2));
            for (uint32_t r = 0; r < 32; ++r) {
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t i = offset(r, c);
                    values[i] = r < rows && r >= tap ? h[offset(r - tap, c)] : 0;
                    kernels[i] = r < rows ? b[offset(branch * 2 + tap, c)] : 0;
                    coefficients[i] = r < rows ? d[offset(r, column % 32 + c / 16)] : 0;
                }
            }
            cb_push_back(0, 1);
            cb_push_back(1, 1);
            cb_push_back(2, 1);
        }
    }
#elif defined(COMPILE_FOR_BRISC)
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), bytes);
    for (uint32_t tile = worker; tile < width_tiles; tile += workers) {
        cb_wait_front(16, 1);
        noc_async_write(get_read_ptr(16), get_noc_addr(tile, output), bytes);
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
#elif defined(COMPILE_FOR_TRISC)
    for (uint32_t tile = worker; tile < width_tiles; tile += workers) {
        for (uint32_t tap = 0; tap < 2; ++tap) {
            cb_wait_front(0, 1);
            cb_wait_front(1, 1);
            cb_reserve_back(4, 1);
            unary_op_init_common(0, 4);
            mul_binary_tile_init();
            tile_regs_acquire();
            load(0, 0);
            load(1, 1);
            mul_binary_tile(0, 1, 0);
            publish(4); // BF16(base * value), not folded into the FMA.
            cb_pop_front(1, 1);

            const uint32_t previous = tap == 0 ? 7 : 6;
            cb_wait_front(previous, 1);
            cb_wait_front(4, 1);
            cb_reserve_back(5, 1);
            unary_op_init_common(previous, 5);
            add_binary_tile_init();
            tile_regs_acquire();
            load(previous, 0);
            load(4, 1);
            add_binary_tile(0, 1, 0);
            publish(5); // BF16(output + rounded base product).
            cb_pop_front(previous, 1);
            cb_pop_front(4, 1);

            const uint32_t destination = tap == 0 ? 6 : 16;
            cb_wait_front(5, 1);
            cb_wait_front(2, 1);
            cb_reserve_back(destination, 1);
            unary_op_init_common(5, destination);
            addcmul_tile_init();
            tile_regs_acquire();
            load(5, 0);
            load(2, 1);
            load(0, 2);
            addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, 0x3f800000);
            publish(destination); // Native SFPU MAD then round-to-nearest BF16.
            cb_pop_front(5, 1);
            cb_pop_front(2, 1);
            cb_pop_front(0, 1);
        }
    }
#endif
}
