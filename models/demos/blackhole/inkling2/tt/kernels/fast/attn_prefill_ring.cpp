// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 sliding-layer prefill (fast/attn_prefill.py FlashPrefill.ring_write): the aligned-frame chunk K / V tile
// rows [1, 8, Tq 32, 128] (the ring's format; row tile i = position tile start / 32 + i, start from the device
// tensor pos) into the ring [1, 8, 1152, 128] at slot tile (start / 32 + i) % 34, for the last n row tiles.  For an
// aligned start (o = 0) KVFront does not compute the frame's extra last tile (row tile Tq - 1): it is never copied
// (kn / vn are uninitialized there; the ring slot keeps its finite old rows, masked as dead / stale).
// Work unit u = (kv head g, tile first + u % n): 4 K + 4 V tiles, DRAM -> L1 -> DRAM.  SLOTTED (batched serving): the
// ring of sequence slot s = pos word 2 ([S, 8, 1152, 128]: pages s * 1152 ..).
#ifndef SLOTTED
#define SLOTTED 0
#endif
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t NKH = get_compile_time_arg_val(0);
constexpr uint32_t DHt = get_compile_time_arg_val(1);
constexpr uint32_t Tq = get_compile_time_arg_val(2);  // kn row tiles
constexpr uint32_t ntile = get_compile_time_arg_val(3);
constexpr uint32_t k_bytes = get_compile_time_arg_val(4);
constexpr uint32_t v_bytes = get_compile_time_arg_val(5);
constexpr uint32_t cb_buf = get_compile_time_arg_val(6);
constexpr uint32_t kRingTiles = 36, kRingMod = 34;

FORCE_INLINE auto dram(uint32_t addr, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), addr, page);
}

void kernel_main() {
    const uint32_t kn_addr = get_arg_val<uint32_t>(0);
    const uint32_t vn_addr = get_arg_val<uint32_t>(1);
    const uint32_t kr_addr = get_arg_val<uint32_t>(2);
    const uint32_t vr_addr = get_arg_val<uint32_t>(3);
    const uint32_t pos_addr = get_arg_val<uint32_t>(4);
    const uint32_t first = get_arg_val<uint32_t>(5);
    const uint32_t u0 = get_arg_val<uint32_t>(6);
    const uint32_t nu = get_arg_val<uint32_t>(7);
    if (nu == 0) {
        return;
    }
    const auto kn = dram(kn_addr, k_bytes);
    const auto vn = dram(vn_addr, v_bytes);
    const auto kr = dram(kr_addr, k_bytes);
    const auto vr = dram(vr_addr, v_bytes);
    const auto ps = dram(pos_addr, 128);
    const uint32_t buf = get_write_ptr(cb_buf);
    const uint32_t vbuf = buf + DHt * 2048;
    const uint32_t pbuf = vbuf + DHt * 2048;
    noc_async_read(get_noc_addr(0, ps), pbuf, 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pbuf)[0];
    const uint32_t s_t = start >> 5;
    const uint32_t ring_base =
        SLOTTED ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pbuf)[2] * NKH * kRingTiles * DHt : 0;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t g = u / ntile;
        const uint32_t i = first + u % ntile;
        if (i == Tq - 1 && (start & 31) == 0) {
            continue;
        }
        const uint32_t src = (g * Tq + i) * DHt;
        const uint32_t dst = ring_base + (g * kRingTiles + (s_t + i) % kRingMod) * DHt;
        for (uint32_t d = 0; d < DHt; ++d) {
            noc_async_read(get_noc_addr(src + d, kn), buf + d * k_bytes, k_bytes);
            noc_async_read(get_noc_addr(src + d, vn), vbuf + d * v_bytes, v_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t d = 0; d < DHt; ++d) {
            noc_async_write(buf + d * k_bytes, get_noc_addr(dst + d, kr), k_bytes);
            noc_async_write(vbuf + d * v_bytes, get_noc_addr(dst + d, vr), v_bytes);
        }
        noc_async_write_barrier();
    }
}
