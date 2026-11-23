// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 attention prefill (fast/attn_prefill.py FlashPrefill.conv_update): the K | V short-conv inputs of a chunk
// into the slot's position-indexed conv-input ring ([1, 1, 32, 2048] BF16, row pos % 32 = x[pos]; the fused decode
// reads rows pos - 3 .. pos - 1) and the stock decode's shift states (state j = x[end - 3 + j], row 0 of [1, 1, 32,
// 2048]), end = start + valid (start / valid from the device tensor pos).  Positions max(start, end - 32) .. end - 1
// are written from the chunk's k / v linear outputs (channels 0..1023 / 1024..2047); older ring rows are kept.
// Work unit = one column tile c of the 2048 channels (64 units).  SLOTTED (batched serving): the ring of sequence slot
// s = pos word 2 (pages s * 64 ..); the shift states stay one shared scratch (only the stock decode reads them).
#ifndef SLOTTED
#define SLOTTED 0
#endif
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t Tpt = get_compile_time_arg_val(0);
constexpr uint32_t cb_buf = get_compile_time_arg_val(1);
constexpr uint32_t kBf16 = 2048;

FORCE_INLINE auto dram(uint32_t addr, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), addr, page);
}

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

FORCE_INLINE void zero_l1(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + off, bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE);
    }
}

void kernel_main() {
    uint32_t a = 0;
    const uint32_t xk_addr = get_arg_val<uint32_t>(a++);  // k / v linear outputs: tensor, first column tile, width
    const uint32_t xk_c0 = get_arg_val<uint32_t>(a++);
    const uint32_t xk_w = get_arg_val<uint32_t>(a++);
    const uint32_t xv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t xv_c0 = get_arg_val<uint32_t>(a++);
    const uint32_t xv_w = get_arg_val<uint32_t>(a++);
    const uint32_t ring_addr = get_arg_val<uint32_t>(a++);
    const uint32_t s0_addr = get_arg_val<uint32_t>(a++);
    const uint32_t s1_addr = get_arg_val<uint32_t>(a++);
    const uint32_t s2_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(a++);
    const uint32_t c0 = get_arg_val<uint32_t>(a++);
    const uint32_t nc = get_arg_val<uint32_t>(a++);
    if (nc == 0) {
        return;
    }
    const auto xk = dram(xk_addr, kBf16);
    const auto xv = dram(xv_addr, kBf16);
    const auto rg = dram(ring_addr, kBf16);
    const auto ps = dram(pos_addr, 128);
    const uint32_t st_addr[3] = {s0_addr, s1_addr, s2_addr};
    // L1: [ring tile | x tile A | x tile B | state tile | pos page]
    const uint32_t base = get_write_ptr(cb_buf);
    const uint32_t ring = base, xa = base + kBf16, xb = base + 2 * kBf16, st = base + 3 * kBf16;
    const uint32_t pp = base + 4 * kBf16;
    noc_async_read(get_noc_addr(0, ps), pp, 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp)[0];
    const uint32_t valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp)[1];
    const uint32_t end = start + valid;
    const uint32_t r0 = valid > 32 ? valid - 32 : 0;  // first chunk row written
    const uint32_t ta = r0 >> 5, tb = (valid - 1) >> 5;
    const uint32_t ring_base = SLOTTED ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp)[2] * 64 : 0;
    for (uint32_t c = c0; c < c0 + nc; ++c) {
        noc_async_read(get_noc_addr(ring_base + c, rg), ring, kBf16);
        const bool isk = c < 32;
        const uint32_t xw = isk ? xk_w : xv_w;
        const uint32_t xc = (isk ? xk_c0 + c : xv_c0 + c - 32);
        noc_async_read(get_noc_addr(ta * xw + xc, isk ? xk : xv), xa, kBf16);
        if (tb != ta && tb < Tpt) {
            noc_async_read(get_noc_addr(tb * xw + xc, isk ? xk : xv), xb, kBf16);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t r = r0; r < valid; ++r) {
            copy_row(ring, (start + r) & 31, (r >> 5) == ta ? xa : xb, r & 31);
        }
        noc_async_write(ring, get_noc_addr(ring_base + c, rg), kBf16);
        for (uint32_t j = 0; j < 3; ++j) {
            zero_l1(st, kBf16);
            noc_async_read_barrier();
            if (end + j >= 3) {
                copy_row(st, 0, ring, (end + j - 3) & 31);
            }
            const auto sd = dram(st_addr[j], kBf16);
            noc_async_write(st, get_noc_addr(c, sd), kBf16);
            noc_async_write_barrier();
        }
    }
}
