// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 attention decode, fused qkv + front end (attn_fused.AttnQKVFront): the qkv matmul workers.
//
// Same weight stream and math as decode_matmul_stream.cpp (StreamDecodeMatmul, preset attn_qkv, whole K in FP32
// DEST, ks = 0, no glu): NCRISC streams the core's contiguous weight run, TRISC accumulates, BRISC gathers the x
// rows.  Instead of writing qkv to DRAM, BRISC sends every output column's token rows straight into the input tiles
// of the front-end workers (attn_front_fused.cpp, the AttnDecode program): column c of qkv is head c / 2 (q 0..31,
// k 32..39, v 40..47), half c % 2; head h of section s lands in row (h - h0) of the half-c input tile of that
// section's worker for token t.  After its writes have landed it increments each target worker's semaphore once.
//
// CBs: 12 x rows (BF16), 13 weight ring, 14 out (BF16); front CB 0 (2 BF16 tiles) is the remote target.
// Per-core args (all RISCs): [bank, run offset, ncols, first column].
// BRISC common args: [x address, then per front worker (token-major, NQ + 2 per token) noc x, noc y].
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace {
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");      // K tiles
constexpr uint32_t wmax = get_named_compile_time_arg_val("w");     // subblock width (tiles)
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");      // K tiles per streamed block
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");  // valid activation rows (tokens)
constexpr uint32_t cb_in0 = 12;
constexpr uint32_t cb_in1 = 13;
constexpr uint32_t cb_out = 14;
constexpr uint32_t cb_front_in = 0;  // front workers' input tiles (same L1 address on every core)
constexpr uint32_t blk_tiles = kb * wmax;
static_assert(kt % kb == 0);
}  // namespace

void kernel_main() {
    const uint32_t ncols = get_arg_val<uint32_t>(2);
    const uint32_t nsub = (ncols + wmax - 1) / wmax;
    const uint32_t rem = ncols - (nsub - 1) * wmax;
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t tile_bytes = get_named_compile_time_arg_val("in1_tile_bytes");
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t outstanding = get_named_compile_time_arg_val("outstanding");
    static_assert(outstanding >= 1 && outstanding < slots && outstanding <= 8);
    constexpr uint32_t slot_bytes = blk_tiles * tile_bytes;
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t run = get_arg_val<uint32_t>(1);
    const uint64_t src_base = get_noc_addr_from_bank_id<true>(bank, w_addr + run);
    const uint32_t ring_base = get_write_ptr(cb_in1);
    const uint32_t ring_end = ring_base + slots * slot_bytes;
    uint32_t src_off = 0;
    uint32_t pending = 0, issue_trid = 1, wait_trid = 1;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t bytes = kb * (s == nsub - 1 ? rem : wmax) * tile_bytes;
        for (uint32_t b = 0; b < kt / kb; ++b) {
            cb_reserve_back(cb_in1, blk_tiles * (pending + 1));
            uint32_t dst = get_write_ptr(cb_in1) + pending * slot_bytes;
            if (dst >= ring_end) {
                dst -= slots * slot_bytes;
            }
            noc_async_read_set_trid(issue_trid);
            noc_async_read(src_base + src_off, dst, bytes);
            src_off += bytes;
            issue_trid = issue_trid == outstanding ? 1 : issue_trid + 1;
            if (++pending == outstanding) {
                if constexpr (outstanding == 1) {
                    noc_async_read_barrier();
                } else {
                    noc_async_read_barrier_with_trid(wait_trid);
                }
                cb_push_back(cb_in1, blk_tiles);
                wait_trid = wait_trid == outstanding ? 1 : wait_trid + 1;
                --pending;
            }
        }
    }
    noc_async_read_barrier();
    for (; pending > 0; --pending) {
        cb_push_back(cb_in1, blk_tiles);
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t nq = get_named_compile_time_arg_val("nq");  // q workers per token
    constexpr uint32_t per_token = nq + 2;
    constexpr uint32_t qh = 32 / nq;  // q heads per q worker
    constexpr uint32_t x_tile_bytes = 2048;
    constexpr uint32_t x_face_bytes = 512;
    constexpr uint32_t x_row_bytes = 32;
    constexpr uint32_t top = rows < 16 ? rows : 16;
    constexpr uint32_t bottom = rows > 16 ? rows - 16 : 0;
    constexpr uint32_t x_top = (top * x_row_bytes + 63) / 64 * 64;
    constexpr uint32_t x_bottom = (bottom * x_row_bytes + 63) / 64 * 64;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), x_tile_bytes);
    const uint32_t first_col = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = get_semaphore(0);
    const uint32_t front_in = get_write_ptr(cb_front_in);

    cb_reserve_back(cb_in0, kt);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t t = 0; t < kt; ++t) {
        const uint64_t src = get_noc_addr(t, x);
        const uint32_t dst = in0 + t * x_tile_bytes;
        noc_async_read(src, dst, x_top);
        noc_async_read(src + x_face_bytes, dst + x_face_bytes, x_top);
        if constexpr (bottom > 0) {
            noc_async_read(src + 2 * x_face_bytes, dst + 2 * x_face_bytes, x_bottom);
            noc_async_read(src + 3 * x_face_bytes, dst + 3 * x_face_bytes, x_bottom);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_in0, kt);

    // column -> (front slot, row in the worker's tiles, half)
    auto slot_of = [](uint32_t col) -> uint32_t {
        const uint32_t h = col / 2;
        return h < 32 ? h / qh : (h < 40 ? nq : nq + 1);
    };
    auto row_of = [](uint32_t col) -> uint32_t {
        const uint32_t h = col / 2;
        return h < 32 ? h % qh : (h < 40 ? h - 32 : h - 40);
    };
    uint32_t col = first_col;
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        cb_wait_front(cb_out, wmax);
        DeviceZoneScopedN("qkv-send");
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < wid; ++j) {
            const uint32_t c = col + j;
            const uint32_t slot = slot_of(c), i = row_of(c), half = c & 1;
            const uint32_t tile = src + j * x_tile_bytes;
            for (uint32_t t = 0; t < rows; ++t) {
                const uint32_t f_idx = per_token * t + slot;
                const uint32_t nx = get_common_arg_val<uint32_t>(1 + 2 * f_idx);
                const uint32_t ny = get_common_arg_val<uint32_t>(2 + 2 * f_idx);
                const uint32_t src_row = tile + ((t >> 4) * 2) * x_face_bytes + (t & 15) * x_row_bytes;
                const uint32_t dst_row = front_in + half * x_tile_bytes + ((i >> 4) * 2) * x_face_bytes +
                                         (i & 15) * x_row_bytes;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write(src_row + f * x_face_bytes, get_noc_addr(nx, ny, dst_row + f * x_face_bytes),
                                    x_row_bytes);
                }
            }
        }
        col += wid;
        noc_async_write_barrier();
        cb_pop_front(cb_out, wmax);
    }
    // one increment per (token, distinct target worker); this core's columns are consecutive -> slots monotonic
    DeviceZoneScopedN("qkv-sem");
    const uint32_t s_first = slot_of(first_col), s_last = slot_of(first_col + ncols - 1);
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t slot = s_first; slot <= s_last; ++slot) {
            const uint32_t f_idx = per_token * t + slot;
            const uint32_t nx = get_common_arg_val<uint32_t>(1 + 2 * f_idx);
            const uint32_t ny = get_common_arg_val<uint32_t>(2 + 2 * f_idx);
            noc_semaphore_inc(get_noc_addr(nx, ny, sem_addr), 1);
        }
    }
    noc_async_atomic_barrier();
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    uint32_t cur = wmax;
    matmul_block_init(cb_in0, cb_in1, 0, wmax, 1, kt);
    cb_wait_front(cb_in0, kt);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        if (wid != cur) {
            matmul_block_init(cb_in0, cb_in1, 0, wid, 1, kt);
            cur = wid;
        }
        tile_regs_acquire();
        for (uint32_t b = 0; b < kt / kb; ++b) {
            cb_wait_front(cb_in1, blk_tiles);
            for (uint32_t i = 0; i < kb; ++i) {
                matmul_block(cb_in0, cb_in1, b * kb + i, i * wid, 0, 0, wid, 1, kt);
            }
            cb_pop_front(cb_in1, blk_tiles);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, wmax);
        tile_regs_wait();
        for (uint32_t j = 0; j < wid; ++j) {
            pack_tile(j, cb_out, j);
        }
        tile_regs_release();
        cb_push_back(cb_out, wmax);
    }
    cb_pop_front(cb_in0, kt);
#endif
}
