// SPDX-License-Identifier: Apache-2.0
// Shared verify rounds: for every group of two or more rows, copy its accepted row's GDN candidates
// (the TILE recurrent state [48,128,128] and the three convolution history rows) into the group's
// slot. Pure DMA; the bytes are those the single-user commit selects.
//
// Common args: [0] commit table ([1,32] uint32: count, then (slot, row) pairs), then per layer:
// candidate states [T,48,128,128], candidate histories [1,3T,10240], GC_SLOTS slot states, GC_SLOTS
// slot histories. Per-core arg: core index (BRISC worker 2i, NCRISC worker 2i + 1).
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t layers = GC_LAYERS;
    constexpr uint32_t slots = GC_SLOTS;
    constexpr uint32_t workers = GC_WORKERS;
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t state_tiles = 48 * 16;
    constexpr uint32_t history_bytes = 10240 * 2;
    constexpr uint32_t chunk_bytes = 1024;
    constexpr uint32_t conv_chunks = 3 * history_bytes / chunk_bytes;
    constexpr uint32_t stride = 2 + 2 * slots;
    constexpr uint32_t batch = 6;
#if defined(COMPILE_FOR_BRISC)
    const uint32_t worker = 2 * get_arg_val<uint32_t>(0);
    const uint32_t scratch = get_write_ptr(0);
#else
    const uint32_t worker = 2 * get_arg_val<uint32_t>(0) + 1;
    const uint32_t scratch = get_write_ptr(1);
#endif
    const auto table = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), 128);
    noc_async_read(get_noc_addr(0, table), scratch, 128);
    noc_async_read_barrier();
    const auto* words = reinterpret_cast<volatile uint32_t*>(scratch);
    const uint32_t count = words[0] < 8 ? words[0] : 8;
    uint32_t group_slot[8], group_row[8];
    for (uint32_t g = 0; g < count; ++g) {
        group_slot[g] = words[1 + 2 * g];
        group_row[g] = words[2 + 2 * g];
    }
    const uint32_t buffer = scratch + 128;
    const uint32_t first = worker * state_tiles / workers, last = (worker + 1) * state_tiles / workers;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const uint32_t base = 1 + layer * stride;
        const auto candidates = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(base), tile_bytes);
        const auto histories = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(base + 1), history_bytes);
        for (uint32_t g = 0; g < count; ++g) {
            const auto state = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                              get_common_arg_val<uint32_t>(base + 2 + group_slot[g]), tile_bytes);
            const auto history = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                get_common_arg_val<uint32_t>(base + 2 + slots + group_slot[g]), history_bytes);
            const uint32_t source = group_row[g] * state_tiles;
            for (uint32_t tile = first; tile < last; tile += batch) {
                const uint32_t n = last - tile < batch ? last - tile : batch;
                for (uint32_t i = 0; i < n; ++i) {
                    noc_async_read(get_noc_addr(source + tile + i, candidates), buffer + i * tile_bytes, tile_bytes);
                }
                noc_async_read_barrier();
                for (uint32_t i = 0; i < n; ++i) {
                    noc_async_write(buffer + i * tile_bytes, get_noc_addr(tile + i, state), tile_bytes);
                }
                noc_async_write_barrier();
            }
            for (uint32_t chunk = worker; chunk < conv_chunks; chunk += workers) {
                const uint32_t row = chunk / (history_bytes / chunk_bytes);
                const uint32_t offset = (chunk % (history_bytes / chunk_bytes)) * chunk_bytes;
                noc_async_read(get_noc_addr(3 * group_row[g] + row, histories) + offset, buffer, chunk_bytes);
                noc_async_read_barrier();
                noc_async_write(buffer, get_noc_addr(row, history) + offset, chunk_bytes);
                noc_async_write_barrier();
            }
        }
    }
#endif
}
