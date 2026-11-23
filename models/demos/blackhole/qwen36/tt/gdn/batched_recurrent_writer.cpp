// SPDX-License-Identifier: Apache-2.0
// Per-job operands, output rows, and state stores for (slot, value-head) jobs; the
// reader core-mate streams the state in. Each job is independent, never a time step of
// its neighbour.
//
// Operands for the next job are gathered before waiting on the current one, into
// double-buffered CBs whose pages stay zero outside row 0 (tile element 0 for the
// scalars). Each is a 32-byte L1 row (activations) or a staged per-head constant, so
// none of these small reads waits in a DRAM queue behind the state stream. Working state row m is key row (m / 16) + 8 * (m % 16) (see the reader),
// so q and k are permuted identically here and every product and sum over the key
// dimension sees the same set of terms; v and the output are value-indexed.
// Output rows land in o_proj input layout ([1,B,HV*128], slot rows); updated state goes
// back as 4-KiB bank runs. Writes are only flushed, never acknowledged, before
// releasing a CB page; the final barrier covers all.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t kRun = 4096;
    constexpr uint32_t kStage = 10, kScratch = 9;
    constexpr bool mega_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool positions_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t positions_page_bytes = get_compile_time_arg_val(2);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t stride = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    // qkv: L1 [1,1,32,8192] activated q|k|v, slot rows. mega: a/b columns at tiles 256/257.
    const auto qkv = TensorAccessor(tensor_accessor::make_interleaved_dspec<false>(), get_arg_val<uint32_t>(3), tile_bytes);
    const auto mega = TensorAccessor(tensor_accessor::make_interleaved_dspec<mega_dram>(), get_arg_val<uint32_t>(4), tile_bytes);
    const auto bias = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_arg_val<uint32_t>(5), tile_bytes);
    const auto scale = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_arg_val<uint32_t>(6), tile_bytes);
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_arg_val<uint32_t>(7), tile_bytes);
    const auto state = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_arg_val<uint32_t>(8), 256);
    const auto positions = TensorAccessor(tensor_accessor::make_interleaved_dspec<positions_dram>(),
                                          get_arg_val<uint32_t>(9), positions_page_bytes);
    const uint32_t scratch = get_write_ptr(kScratch);
    const uint32_t stage = get_write_ptr(kStage);
    noc_async_read(get_noc_addr(0, positions), scratch, 64);
    Noc noc;
    constexpr uint32_t operands[] = {0, 1, 2, 3, 4, 6, 7};
    constexpr uint32_t pages[] = {8, 8, 8, 2, 2, 2, 2};
    for (uint32_t i = 0; i < 7; ++i) {
        cb_reserve_back(operands[i], pages[i]);
        noc.async_write_zeros(CircularBuffer(operands[i]), pages[i] * tile_bytes);
    }
    noc.write_zeros_l1_barrier();
    // CB26 has one physical tile: compute consumes it but never modifies its bytes.
    auto* ones = reinterpret_cast<volatile uint32_t*>(get_write_ptr(26));
    for (uint32_t i = 0; i < tile_bytes / 4; ++i) ones[i] = 0x3f803f80;
    auto* zero = reinterpret_cast<volatile uint32_t*>(scratch + 64);
    for (uint32_t i = 0; i < 8; ++i) zero[i] = 0;

    // Per-head decay constants sit at element 0 of their tiles: stage all heads once.
    const uint32_t decay = stage + 1024;
    for (uint32_t head = 0; head < 32; ++head) {
        noc_async_read(get_noc_addr(head, bias), decay + head * 64, 64);
        noc_async_read(get_noc_addr(head, scale), decay + 2048 + head * 64, 64);
    }
    auto gather = [&](uint32_t index) {
        const uint32_t slot = index / 32, head = index % 32, key = head / 2;
        const uint32_t face_row = (slot / 16) * 1024 + (slot % 16) * 32;
        cb_reserve_back(0, 4); cb_reserve_back(1, 4); cb_reserve_back(2, 4);
        cb_reserve_back(3, 1); cb_reserve_back(4, 1); cb_reserve_back(6, 1); cb_reserve_back(7, 1);
        // L1 rows: q and k to staging in natural key order, v straight into place.
        const uint32_t sources[3] = {key * 4, 64 + key * 4, 128 + head * 4};
        for (uint32_t tile = 0; tile < 4; ++tile) {
            for (uint32_t face = 0; face < 2; ++face) {
                for (uint32_t operand = 0; operand < 2; ++operand) {
                    noc_async_read(get_noc_addr(sources[operand] + tile, qkv) + face_row + face * 512,
                                   stage + operand * 256 + (tile * 2 + face) * 32, 32);
                }
                noc_async_read(get_noc_addr(sources[2] + tile, qkv) + face_row + face * 512,
                               get_write_ptr(2) + tile * tile_bytes + face * 512, 32);
            }
        }
        const uint32_t scalar = face_row + (head / 16) * 512;
        noc_async_read(get_noc_addr(256, mega) + scalar, stage + 512, 32);
        noc_async_read(get_noc_addr(257, mega) + scalar, stage + 544, 32);
        noc_async_read_barrier();
        // q, k: working column 16b + i <- key b + 8i, so word w of working row b pairs
        // keys b + 16w (low half) and b + 16w + 8 (high half).
        for (uint32_t operand = 0; operand < 2; ++operand) {
            const uint32_t dst = get_write_ptr(operand);
            for (uint32_t w = 0; w < 8; ++w) {
                const auto* src = reinterpret_cast<volatile uint32_t*>(stage + operand * 256 + w * 32);
                for (uint32_t k = 0; k < 4; ++k) {
                    const uint32_t lo = src[k], hi = src[k + 4];
                    reinterpret_cast<volatile uint32_t*>(dst + k * tile_bytes)[w] = (lo & 0xffffu) | (hi << 16);
                    reinterpret_cast<volatile uint32_t*>(dst + k * tile_bytes + 512)[w] = (lo >> 16) | (hi & 0xffff0000u);
                }
            }
        }
        const auto* scalars = reinterpret_cast<volatile uint16_t*>(stage + 512);
        const auto* constants = reinterpret_cast<volatile uint16_t*>(decay);
        reinterpret_cast<volatile uint16_t*>(get_write_ptr(3))[0] = scalars[head % 16];
        reinterpret_cast<volatile uint16_t*>(get_write_ptr(4))[0] = scalars[16 + head % 16];
        reinterpret_cast<volatile uint16_t*>(get_write_ptr(6))[0] = constants[head * 32];
        reinterpret_cast<volatile uint16_t*>(get_write_ptr(7))[0] = constants[1024 + head * 32];
        cb_push_back(0, 4); cb_push_back(1, 4); cb_push_back(2, 4);
        cb_push_back(3, 1); cb_push_back(4, 1); cb_push_back(6, 1); cb_push_back(7, 1);
        cb_reserve_back(26, 1);
        cb_push_back(26, 1);
    };

    if (count > 0) gather(first);  // Its barrier also covers the positions and constants.
    for (uint32_t job = 0; job < count; ++job) {
        const uint32_t index = first + job * stride;
        if (job + 1 < count) gather(index + stride);
        const uint32_t slot = index / 32, head = index % 32;
        const bool active = reinterpret_cast<volatile int32_t*>(scratch)[slot] >= 0;
        const uint32_t face_row = (slot / 16) * 1024 + (slot % 16) * 32;
        cb_wait_front(16, 4);
        const uint32_t src = get_read_ptr(16);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            const uint64_t dst = get_noc_addr(head * 4 + tile, output) + face_row;
            for (uint32_t face = 0; face < 2; ++face) {
                noc_async_write(active ? src + tile * 2048 + face * 512 : scratch + 64, dst + face * 512, 32);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(16, 4);
        // Inactive jobs drain the compute stream without touching persistent bytes.
        for (uint32_t tile_row = 0; tile_row < 4; ++tile_row) {
            cb_wait_front(17, 4);
            if (active) {
                const uint32_t rows = get_read_ptr(17);
                for (uint32_t run = 0; run < 2; ++run) {
                    noc_async_write(rows + run * kRun, get_noc_addr(index * 128 + tile_row * 2 + run, state), kRun);
                }
                noc_async_writes_flushed();
            }
            cb_pop_front(17, 4);
        }
    }
    noc_async_write_barrier();
}
