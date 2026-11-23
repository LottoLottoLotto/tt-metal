// SPDX-License-Identifier: Apache-2.0
// Batched GDN frontend writer: fetches the taps/bias rows (in parallel with the reader's
// activations), then stores whole activated QKV column tiles, slot rows included.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t kTileBytes = 2048, kParams = 2, kOutput = 16;
    constexpr bool qkv_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool tap_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool bias_dram = get_compile_time_arg_val(2) == 1;
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    const auto qkv = TensorAccessor(tensor_accessor::make_interleaved_dspec<qkv_dram>(), get_arg_val<uint32_t>(2), kTileBytes);
    Noc noc;
    // Params page p * tiles + i: tap p (4 = bias) of column tile i, zero outside row 0.
    cb_reserve_back(kParams, 5 * tiles);
    noc.async_write_zeros(CircularBuffer(kParams), 5 * tiles * kTileBytes);
    noc.write_zeros_l1_barrier();
    const uint32_t params = get_write_ptr(kParams);
    for (uint32_t param = 0; param < 5; ++param) {
        const uint32_t address = get_arg_val<uint32_t>(3 + param);
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            const uint64_t src = param < 4
                ? get_noc_addr(first + tile, TensorAccessor(tensor_accessor::make_interleaved_dspec<tap_dram>(), address, kTileBytes))
                : get_noc_addr(first + tile, TensorAccessor(tensor_accessor::make_interleaved_dspec<bias_dram>(), address, kTileBytes));
            // Rows 0-1 of both upper faces: the parameter row (row 1 is zero padding).
            const uint32_t dst = params + (param * tiles + tile) * kTileBytes;
            noc_async_read(src, dst, 64);
            noc_async_read(src + 512, dst + 512, 64);
        }
    }
    noc_async_read_barrier();
    cb_push_back(kParams, 5 * tiles);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        cb_wait_front(kOutput, 1);
        noc_async_write(get_read_ptr(kOutput), get_noc_addr(first + tile, qkv), kTileBytes);
        noc_async_writes_flushed();
        cb_pop_front(kOutput, 1);
    }
    noc_async_write_barrier();
}
