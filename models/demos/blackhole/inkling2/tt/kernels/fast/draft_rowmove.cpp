// SPDX-License-Identifier: Apache-2.0
// Row-segment mover of the drafter (fast/draft_ops.py RowMove): copies 32-value tile-row segments between BF16
// TILE interleaved DRAM tensors without any compute, in one launch: rows -> heads ([1, 1, U, n*128] -> [1, U, 32,
// 128]), heads -> rows, and rows -> paged K / V cache rows at device-side positions (no read-modify-write of cache
// tiles: each 32-value face row is its own 32 B write, so rows of the same tile written by different workers never
// race, unlike paged_update_cache).
//
// A tensor is viewed as a 2-D matrix of padded rows ("grow": every leading dim folded into rows, each 32-row tile
// row is one page row of Wt column tiles): page(grow, ct) = (grow >> 5) * Wt + ct, row grow & 31 inside the tile.
// Move (4 per-worker runtime words): w0 = src | dst << 4 | mode << 8 | ntiles << 16; w1 = src grow; w2 = src ct |
// dst ct << 16; w3 = mode 0: dst grow; mode 1 (paged cache, identity page table, block 64): user | head << 8 |
// nkv << 16 -> pos = positions[user], dst grow = ((pos >> 6) * nkv + head) * 64 + (pos & 63).
// Common args: [positions address (int32 row-major, page 0), (address, Wt) per tensor].  Per-worker args: [n moves,
// moves...].  Workers: BRISC and NCRISC of every core (separate argument lists).
// DRAM reads are 64 B aligned (the pair of face rows containing the row), writes are 32 B (one face row).
#include <cstdint>
#include "read_vc.hpp"
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

void kernel_main() {
    ink_set_read_vc();
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t buf_cb = 0;
#else
    constexpr uint32_t buf_cb = 1;
#endif
    const uint32_t n = get_arg_val<uint32_t>(0);
    if (n == 0) {
        return;
    }
    const uint32_t buf = (get_write_ptr(buf_cb) + 63) & ~63u;
    // positions: first 32 int32 words of the positions tensor (only when a move targets a cache)
    volatile tt_l1_ptr int32_t* posv = reinterpret_cast<volatile tt_l1_ptr int32_t*>(buf);
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(0);
    bool need_pos = false;
    for (uint32_t m = 0; m < n; ++m) {
        need_pos |= ((get_arg_val<uint32_t>(1 + 4 * m) >> 8) & 1) != 0;
    }
    if (need_pos) {
        const auto pt = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), pos_addr, 128);
        noc_async_read(pt.get_noc_addr(0), buf, 128);
        noc_async_read_barrier();
    }
    const uint32_t data = buf + 128;
    // phase 1: read the 64 B pair of every source face row
    uint32_t k = 0;
    for (uint32_t m = 0; m < n; ++m) {
        const uint32_t w0 = get_arg_val<uint32_t>(1 + 4 * m);
        const uint32_t sg = get_arg_val<uint32_t>(2 + 4 * m);
        const uint32_t sct = get_arg_val<uint32_t>(3 + 4 * m) & 0xFFFF;
        const uint32_t s = w0 & 0xF;
        const uint32_t nt = (w0 >> 16) & 0xFF;
        const uint32_t wt = get_common_arg_val<uint32_t>(2 + 2 * s);
        const auto src = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(1 + 2 * s), 2048);
        const uint32_t r = sg & 31;
        for (uint32_t t = 0; t < nt; ++t) {
            const uint64_t page = src.get_noc_addr((sg >> 5) * wt + sct + t);
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t off = ((r >> 4) * 2 + f) * 512 + (r & 15) * 32;
                noc_async_read(page + (off & ~63u), data + k * 64, 64);
                ++k;
            }
        }
    }
    noc_async_read_barrier();
    // phase 2: one 32 B write per face row
    k = 0;
    for (uint32_t m = 0; m < n; ++m) {
        const uint32_t w0 = get_arg_val<uint32_t>(1 + 4 * m);
        const uint32_t sg = get_arg_val<uint32_t>(2 + 4 * m);
        const uint32_t dct = get_arg_val<uint32_t>(3 + 4 * m) >> 16;
        const uint32_t w3 = get_arg_val<uint32_t>(4 + 4 * m);
        const uint32_t d = (w0 >> 4) & 0xF;
        const uint32_t nt = (w0 >> 16) & 0xFF;
        uint32_t dg = w3;
        if ((w0 >> 8) & 1) {
            const uint32_t pos = static_cast<uint32_t>(posv[w3 & 0xFF]);
            const uint32_t head = (w3 >> 8) & 0xFF;
            const uint32_t nkv = (w3 >> 16) & 0xFF;
            dg = ((pos >> 6) * nkv + head) * 64 + (pos & 63);
        }
        const uint32_t wt = get_common_arg_val<uint32_t>(2 + 2 * d);
        const auto dst = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(1 + 2 * d), 2048);
        const uint32_t r = sg & 31;
        const uint32_t dr = dg & 31;
        for (uint32_t t = 0; t < nt; ++t) {
            const uint64_t page = dst.get_noc_addr((dg >> 5) * wt + dct + t);
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t off = ((r >> 4) * 2 + f) * 512 + (r & 15) * 32;
                const uint32_t doff = ((dr >> 4) * 2 + f) * 512 + (dr & 15) * 32;
                noc_async_write(data + k * 64 + (off & 63u), page + doff, 32);
                ++k;
            }
        }
    }
    noc_async_write_barrier();
#endif
}
