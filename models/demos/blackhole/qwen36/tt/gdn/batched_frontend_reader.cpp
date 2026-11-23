// SPDX-License-Identifier: Apache-2.0
// Batched GDN frontend reader. A worker owns a contiguous run of QKV column tiles
// (32 channels each) for every slot row at once: the mega tile already holds the slot
// rows and the history rows arrive row-major for compute to tilize (the writer fetches
// the taps/bias). After the reads land, active slots' history windows advance in place;
// masked slots are never written.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t kTileBytes = 2048;
    constexpr uint32_t kHistoryPageBytes = 8192 * sizeof(uint16_t);
    constexpr uint32_t kRows = 0, kX = 1, kPositions = 7;
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr bool mega_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool positions_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t positions_page_bytes = get_compile_time_arg_val(3);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    const auto mega = TensorAccessor(tensor_accessor::make_interleaved_dspec<mega_dram>(),
                                     get_arg_val<uint32_t>(2), kTileBytes);
    const auto history = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_arg_val<uint32_t>(3), kHistoryPageBytes);
    const auto positions = TensorAccessor(tensor_accessor::make_interleaved_dspec<positions_dram>(),
                                          get_arg_val<uint32_t>(4), positions_page_bytes);
    Noc noc;
    // Row-major history: tap r's block is 32 rows of `tiles` column tiles; rows >= B stay zero.
    const uint32_t row_bytes = tiles * 64;
    cb_reserve_back(kRows, 3 * tiles);
    cb_reserve_back(kX, tiles);
    cb_reserve_back(kPositions, 1);
    noc.async_write_zeros(CircularBuffer(kRows), 3 * tiles * kTileBytes);
    noc.write_zeros_l1_barrier();
    const uint32_t rows = get_write_ptr(kRows);
    const uint32_t x = get_write_ptr(kX);
    const uint32_t scratch = get_write_ptr(kPositions);
    noc_async_read(get_noc_addr(0, positions), scratch, 64);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        noc_async_read(get_noc_addr(first + tile, mega), x + tile * kTileBytes, kTileBytes);
    }
    for (uint32_t tap = 0; tap < 3; ++tap) {
        for (uint32_t slot = 0; slot < batch; ++slot) {
            noc_async_read(get_noc_addr(slot * 3 + tap, history) + first * 64,
                           rows + tap * tiles * kTileBytes + slot * row_bytes, row_bytes);
        }
    }
    noc_async_read_barrier();
    cb_push_back(kRows, 3 * tiles);
    cb_push_back(kX, tiles);
    // History transition: taps 1..3 of the window become rows 0..2. The staged bytes stay
    // untouched: compute only reads these buffers and nothing refills them.
    const auto* position = reinterpret_cast<volatile int32_t*>(scratch);
    for (uint32_t slot = 0; slot < batch; ++slot) {
        if (position[slot] < 0) continue;
        for (uint32_t tap = 1; tap < 3; ++tap) {
            noc_async_write(rows + tap * tiles * kTileBytes + slot * row_bytes,
                            get_noc_addr(slot * 3 + tap - 1, history) + first * 64, row_bytes);
        }
        const uint64_t newest = get_noc_addr(slot * 3 + 2, history) + first * 64;
        const uint32_t face_row = (slot / 16) * 1024 + (slot % 16) * 32;
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            noc_async_write(x + tile * kTileBytes + face_row, newest + tile * 64, 32);
            noc_async_write(x + tile * kTileBytes + face_row + 512, newest + tile * 64 + 32, 32);
        }
    }
    noc_async_write_barrier();
}
