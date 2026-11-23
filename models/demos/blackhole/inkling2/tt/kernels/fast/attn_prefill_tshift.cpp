// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 attention prefill (fast/attn_prefill.py FlashPrefill._shift): the layer's bias table shifted by
// o = start % 32 rows (start from the device tensor pos) into the scratch table of the same layout, for the band
// slots s = t + 1 (distance tile t = -1 .. 32) and tiles k = 0 .. 16:
//   row m of (s, k) = row m + o of table (s, k) (m + o < 32), else row m + o - 32 of table (s + 1, k)
// (query rows of an unaligned chunk are positions 32 p + o + m: the distances of tile t are 32 t + o + m - j).
// o == 0: nothing (the flash reader reads the layer's table).  Work unit u = s * 17 + k.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t kSlot = get_compile_time_arg_val(0);
constexpr uint32_t cb_buf = get_compile_time_arg_val(1);
constexpr uint32_t kBf16 = 2048;
constexpr uint32_t kTiles = 17;

FORCE_INLINE auto dram(uint32_t addr, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), addr, page);
}

// row drow of dst = row srow of src (BF16 tiles: 2 faces x 16 elements per row)
FORCE_INLINE void copy_row(uint32_t dst, uint32_t drow, uint32_t src, uint32_t srow) {
    for (uint32_t h = 0; h < 2; ++h) {
        volatile tt_l1_ptr uint32_t* dp =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + ((drow >> 4) * 2 + h) * 512 + (drow & 15) * 32);
        const volatile tt_l1_ptr uint32_t* sp =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(src + ((srow >> 4) * 2 + h) * 512 + (srow & 15) * 32);
        for (uint32_t w = 0; w < 8; ++w) {
            dp[w] = sp[w];
        }
    }
}

void kernel_main() {
    const uint32_t table_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2);
    const uint32_t u0 = get_arg_val<uint32_t>(3);
    const uint32_t nu = get_arg_val<uint32_t>(4);
    if (nu == 0) {
        return;
    }
    const auto tb = dram(table_addr, kBf16);
    const auto ot = dram(out_addr, kBf16);
    const auto ps = dram(pos_addr, 128);
    const uint32_t a = get_write_ptr(cb_buf), b = a + kBf16, out = a + 2 * kBf16, pp = a + 3 * kBf16;
    noc_async_read(get_noc_addr(0, ps), pp, 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp)[0] & 31;
    if (o == 0) {
        return;
    }
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t s = u / kTiles, k = u % kTiles;
        noc_async_read(get_noc_addr(s * kSlot + k, tb), a, kBf16);
        noc_async_read(get_noc_addr((s + 1) * kSlot + k, tb), b, kBf16);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t m = 0; m < 32; ++m) {
            const uint32_t r = m + o;
            if (r < 32) {
                copy_row(out, m, a, r);
            } else {
                copy_row(out, m, b, r - 32);
            }
        }
        noc_async_write(out, get_noc_addr(s * kSlot + k, ot), kBf16);
        noc_async_write_barrier();
    }
}
