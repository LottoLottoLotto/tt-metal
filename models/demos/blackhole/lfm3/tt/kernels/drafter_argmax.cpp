// SPDX-License-Identifier: Apache-2.0
// Row top-2 of BF16 TILE logits, rows <= 16 of the first tile row, without untilizing (tt/drafter.py: the draft rows'
// argmax over the draft vocabulary and the top-1 / top-2 logits, the drafter's confidence feature).  spec_argmax.cpp's
// work split (worker w = core * 2 + lane scans tiles [w * tiles / W, (w + 1) * tiles / W), face rows [0, rows) of faces
// 0 and 1; the first maximum wins: ties -> the lowest id) with a word-wise inner loop and the runner-up kept.
// Output page (uint32): [idx_r | key1_r | key2_r for r < rows] at 3 r, 3 r + 1, 3 r + 2; key = the monotonic unsigned
// map of the BF16 bits (+1): key -> value on the host.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

namespace {
FORCE_INLINE uint32_t key_of(uint32_t u) {
    u = (u == 0x8000u) ? 0u : u;  // -0 == +0
    return (u & 0x8000u) ? ((~u) & 0xFFFFu) + 1 : (u | 0x8000u) + 1;
}
}  // namespace

void kernel_main() {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t tiles = get_named_compile_time_arg_val("tiles");
    constexpr uint32_t workers = get_named_compile_time_arg_val("workers");
    constexpr uint32_t gather_x = get_named_compile_time_arg_val("gather_x");
    constexpr uint32_t gather_y = get_named_compile_time_arg_val("gather_y");
    static_assert(rows >= 1 && rows <= 16);
    constexpr uint32_t rb = (rows * 32 + 63) / 64 * 64;  // face rows [0, rows) of one face, DRAM-read aligned
    constexpr uint32_t tile_bytes = 2 * rb;
    constexpr uint32_t slot_bytes = (rows * 12 + 15) / 16 * 16;
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t lane = 0;
#else
    constexpr uint32_t lane = 1;
#endif
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t worker = core * 2 + lane;
    const uint32_t first = worker * tiles / workers;
    const uint32_t last = (worker + 1) * tiles / workers;
    const uint32_t buf = get_write_ptr(lane);  // CB 0 (BRISC) / CB 1 (NCRISC): tile_bytes per tile
    const uint32_t slots = get_write_ptr(2);   // gather slots, same address on every core
    const auto logits = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), 2048);
    for (uint32_t t = first; t < last; ++t) {
        const uint64_t src = get_noc_addr(t, logits);
        noc_async_read(src, buf + (t - first) * tile_bytes, rb);
        noc_async_read(src + 512, buf + (t - first) * tile_bytes + rb, rb);
    }
    noc_async_read_barrier();
    uint32_t k1[rows], k2[rows], i1[rows];
    for (uint32_t r = 0; r < rows; ++r) {
        k1[r] = 0;
        k2[r] = 0;
        i1[r] = first * 32;
    }
    for (uint32_t t = first; t < last; ++t) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(buf + (t - first) * tile_bytes);
        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t a = k1[r], b = k2[r], ia = i1[r];
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t* q = p + f * (rb / 4) + r * 8;  // 16 BF16 = 8 words
                const uint32_t base = t * 32 + f * 16;
#pragma GCC unroll 8
                for (uint32_t c = 0; c < 8; ++c) {
                    const uint32_t w = q[c];
                    const uint32_t lo = key_of(w & 0xFFFFu), hi = key_of(w >> 16);
                    if (lo > a) {
                        b = a;
                        a = lo;
                        ia = base + 2 * c;
                    } else if (lo > b) {
                        b = lo;
                    }
                    if (hi > a) {
                        b = a;
                        a = hi;
                        ia = base + 2 * c + 1;
                    } else if (hi > b) {
                        b = hi;
                    }
                }
            }
            k1[r] = a;
            k2[r] = b;
            i1[r] = ia;
        }
    }
    volatile tt_l1_ptr uint32_t* mine = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    for (uint32_t r = 0; r < rows; ++r) {
        mine[3 * r] = k1[r];
        mine[3 * r + 1] = k2[r];
        mine[3 * r + 2] = i1[r];
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
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t a = 0, b = 0, ia = 0;
            for (uint32_t w = 0; w < workers; ++w) {  // worker order = column order: strict > keeps the lowest id
                const uint32_t* s = reinterpret_cast<const uint32_t*>(slots + w * slot_bytes) + 3 * r;
                const uint32_t x1 = s[0], x2 = s[1];
                if (x1 > a) {
                    b = a > x2 ? a : x2;
                    a = x1;
                    ia = s[2];
                } else {
                    const uint32_t m = x1 > x2 ? x1 : x2;  // x1 <= a: the best runner-up is this worker's x1
                    b = m > b ? m : b;
                }
            }
            out[3 * r] = ia;
            out[3 * r + 1] = a;
            out[3 * r + 2] = b;
        }
        const auto output = TensorAccessor(
            tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), rows * 12);
        noc_async_write(buf, get_noc_addr(0, output), rows * 12);
        noc_async_write_barrier();
    }
#endif
#endif
}
