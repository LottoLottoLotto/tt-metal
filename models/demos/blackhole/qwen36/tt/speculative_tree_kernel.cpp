// SPDX-License-Identifier: Apache-2.0
// Fixed B8 tree metadata selection and bitwise selected-feature gather.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t mode = get_named_compile_time_arg_val("mode");
    constexpr uint32_t fork = get_named_compile_time_arg_val("fork_position");
    constexpr uint32_t nodes = 16 - fork;
    constexpr bool flag_dram = get_named_compile_time_arg_val("flag_dram");
    const uint32_t scratch = get_write_ptr(0);
    auto dram = [](uint32_t address, uint32_t bytes) {
        return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), address, bytes);
    };
    if constexpr (mode == 0) {
        // Inputs are authoritative anchor, A[7], B[7]; no host token readback.
        const auto anchor = TensorAccessor(tensor_accessor::make_interleaved_dspec<flag_dram>(),
                                           get_common_arg_val<uint32_t>(0), 32);
        noc_async_read(get_noc_addr(0, anchor), scratch, 32);
        noc_async_read(get_noc_addr(0, dram(get_common_arg_val<uint32_t>(1), 32)), scratch + 64, 32);
        noc_async_read(get_noc_addr(0, dram(get_common_arg_val<uint32_t>(2), 32)), scratch + 128, 32);
        noc_async_read_barrier();
        auto* out = reinterpret_cast<volatile uint32_t*>(scratch + 192);
        const auto* a = reinterpret_cast<volatile uint32_t*>(scratch + 64);
        const auto* b = reinterpret_cast<volatile uint32_t*>(scratch + 128);
        for (uint32_t i = 0; i < 16; ++i) out[i] = 0;
        out[0] = *reinterpret_cast<volatile uint32_t*>(scratch);
        for (uint32_t i = 0; i < 7; ++i) out[i + 1] = a[i];
        for (uint32_t i = fork; i < 8; ++i) out[8 + i - fork] = b[i - 1];
        const auto dest = dram(get_common_arg_val<uint32_t>(3),
                              get_named_compile_time_arg_val("output_bytes"));
        noc_async_write(scratch + 192, get_noc_addr(0, dest), 64);
        noc_async_write_barrier();
    } else if constexpr (mode == 1) {
        const auto targets = dram(get_common_arg_val<uint32_t>(0),
                                  get_named_compile_time_arg_val("input_bytes"));
        const auto tokens = dram(get_common_arg_val<uint32_t>(1),
                                 get_named_compile_time_arg_val("output_bytes"));
        noc_async_read(get_noc_addr(0, targets), scratch, 64);
        noc_async_read(get_noc_addr(0, tokens), scratch + 64, 64);
        noc_async_read_barrier();
        const auto* argmax = reinterpret_cast<volatile uint32_t*>(scratch);
        const auto* ids = reinterpret_cast<volatile uint32_t*>(scratch + 64);
        auto node = [](uint32_t branch, uint32_t i) {
            return branch && i >= fork ? 8 + i - fork : i;
        };
        uint32_t counts[2] = {0, 0};
        for (uint32_t branch = 0; branch < 2; ++branch) {
            for (uint32_t i = 1; i < 8; ++i) {
                if (argmax[node(branch, i - 1)] != ids[node(branch, i)]) break;
                ++counts[branch];
            }
        }
        const uint32_t winner = counts[1] > counts[0];
        const uint32_t accepted = counts[winner];
        auto* out = reinterpret_cast<volatile uint32_t*>(scratch + 128);
        for (uint32_t i = 0; i < 8; ++i) out[i] = argmax[node(winner, i)];
        noc_async_write(scratch + 128, get_noc_addr(0, dram(get_common_arg_val<uint32_t>(2), 32)), 32);
        noc_async_write_barrier();
        // Output addresses: targets, seven flags, accepted, winner, A, B, node.
        for (uint32_t i = 1; i < 8; ++i) out[i] = 0;
        for (uint32_t i = 0; i < 12; ++i) {
            out[0] = i < 7 ? uint32_t(i < accepted)
                : i == 7 ? accepted : i == 8 ? winner : i == 9 ? counts[0]
                : i == 10 ? counts[1] : node(winner, accepted);
            noc_async_write(scratch + 128, get_noc_addr(0, dram(get_common_arg_val<uint32_t>(3 + i), 32)), 32);
            noc_async_write_barrier();
        }
    } else {
        constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
        constexpr uint32_t layers = get_named_compile_time_arg_val("layers");
        constexpr uint32_t workers = get_named_compile_time_arg_val("workers");
        const uint32_t worker = get_arg_val<uint32_t>(0);
        const auto flag = TensorAccessor(tensor_accessor::make_interleaved_dspec<flag_dram>(),
                                         get_common_arg_val<uint32_t>(0), 32);
        noc_async_read(get_noc_addr(0, flag), scratch, 32);
        noc_async_read_barrier();
        const bool winner = *reinterpret_cast<volatile uint32_t*>(scratch) != 0;
        const auto output = dram(get_common_arg_val<uint32_t>(1), 2048);
        for (uint32_t column = worker; column < layers * width_tiles; column += workers) {
            const auto input = dram(get_common_arg_val<uint32_t>(2 + column / width_tiles), 2048);
            const uint64_t src = get_noc_addr(column % width_tiles, input);
            const uint64_t dst = get_noc_addr(column, output);
            for (uint32_t i = 0; i < 8; ++i) {
                const uint32_t row = winner && i >= fork ? 8 + i - fork : i;
                const uint32_t offset = (row / 16) * 1024 + (row % 16) * 32;
                noc_async_read(src + (offset & ~63u), scratch, 64);
                noc_async_read(src + (offset & ~63u) + 512, scratch + 64, 64);
                noc_async_read_barrier();
                noc_async_write(scratch + (offset & 63), dst + i * 32, 32);
                noc_async_write(scratch + 64 + (offset & 63), dst + 512 + i * 32, 32);
                noc_async_write_barrier();
            }
        }
    }
#endif
}
