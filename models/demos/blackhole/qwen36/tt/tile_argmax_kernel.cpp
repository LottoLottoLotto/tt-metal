// SPDX-License-Identifier: Apache-2.0
// Row argmax of BF16 TILE logits (rows <= 16 of the first tile row) without untilizing.
// Workers (core, lane): BRISC is lane 0, NCRISC lane 1; worker w = core * 2 + lane scans the contiguous tile range
// [w * tiles / W, (w + 1) * tiles / W), reading only face rows 0..rows-1 of faces 0 and 1 (face_bytes each, 64 B
// aligned). The first maximum (lowest column) wins; workers own increasing column ranges, so the gather keeps the
// earliest strict maximum. Each worker posts (key, index) per row to the gather core (core 0); its BRISC reduces in
// worker order and writes the uint32 indices to the output page.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t tiles = get_named_compile_time_arg_val("tiles");
    constexpr uint32_t workers = get_named_compile_time_arg_val("workers");
    constexpr uint32_t gather_x = get_named_compile_time_arg_val("gather_x");
    constexpr uint32_t gather_y = get_named_compile_time_arg_val("gather_y");
    constexpr uint32_t face_bytes = get_named_compile_time_arg_val("face_bytes");
    constexpr uint32_t slot_bytes = get_named_compile_time_arg_val("slot_bytes");
    static_assert(rows >= 1 && rows <= 16 && face_bytes >= rows * 32 && face_bytes % 64 == 0);
    static_assert(slot_bytes >= rows * 8 && slot_bytes % 16 == 0);
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t lane = 0;
#else
    constexpr uint32_t lane = 1;
#endif
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t worker = core * 2 + lane;
    const uint32_t first = worker * tiles / workers;
    const uint32_t last = (worker + 1) * tiles / workers;
    const uint32_t buf = get_write_ptr(lane);  // CB 0 (BRISC) / CB 1 (NCRISC): 2 * face_bytes per tile
    const uint32_t slots = get_write_ptr(2);   // gather slots, same address on every core
    const auto logits = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), 2048);
    for (uint32_t t = first; t < last; ++t) {
        const uint64_t src = get_noc_addr(t, logits);
        noc_async_read(src, buf + (t - first) * 2 * face_bytes, face_bytes);
        noc_async_read(src + 512, buf + (t - first) * 2 * face_bytes + face_bytes, face_bytes);
    }
    noc_async_read_barrier();
    uint32_t best_key[rows];
    uint32_t best_idx[rows];
    for (uint32_t r = 0; r < rows; ++r) {
        best_key[r] = 0;
        best_idx[r] = first * 32;
    }
    for (uint32_t t = first; t < last; ++t) {
        volatile tt_l1_ptr uint16_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf + (t - first) * 2 * face_bytes);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint16_t* q = p + f * (face_bytes / 2) + r * 16;
                for (uint32_t c = 0; c < 16; ++c) {
                    // -0 compares equal to +0 (as in ttnn.argmax), so both map to the key of +0.
                    const uint32_t u = q[c] == 0x8000u ? 0u : q[c];
                    // Monotonic map of BF16 bits to unsigned order (+1 so any value beats the initial 0).
                    const uint32_t key = (u & 0x8000u) ? ((~u) & 0xFFFFu) + 1 : (u | 0x8000u) + 1;
                    if (key > best_key[r]) {
                        best_key[r] = key;
                        best_idx[r] = t * 32 + f * 16 + c;
                    }
                }
            }
        }
    }
    volatile tt_l1_ptr uint32_t* mine = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    for (uint32_t r = 0; r < rows; ++r) {
        mine[2 * r] = best_key[r];
        mine[2 * r + 1] = best_idx[r];
    }
    const uint32_t sem = get_semaphore(0);
    noc_async_write(buf, get_noc_addr(gather_x, gather_y, slots + worker * slot_bytes), slot_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(gather_x, gather_y, sem), 1);
    noc_async_atomic_barrier();
#if defined(COMPILE_FOR_BRISC)
    if (core == 0) {
        volatile tt_l1_ptr uint32_t* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
        noc_semaphore_wait(count, workers);
        noc_semaphore_set(count, 0);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots);
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t key = 0, idx = 0;
            for (uint32_t w = 0; w < workers; ++w) {
                const uint32_t k = s[w * (slot_bytes / 4) + 2 * r];
                if (k > key) {
                    key = k;
                    idx = s[w * (slot_bytes / 4) + 2 * r + 1];
                }
            }
            out[r] = idx;
        }
        const auto output = TensorAccessor(
            tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), rows * 4);
        noc_async_write(buf, get_noc_addr(0, output), rows * 4);
        noc_async_write_barrier();
    }
#endif
#endif
}
