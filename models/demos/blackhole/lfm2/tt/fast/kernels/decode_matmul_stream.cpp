// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2.5 decode matmul streamed from DRAM: out[rows, N] = x[rows, K] @ W[K, N], rows <= 32 (one tile row).
//
// W is BFP8/BFP4/BF16 [K, N] in the "stream" layout of decode_matmul.py (stream_layout): every worker core owns
// `ncols` consecutive weight tile columns and its weight tiles are ONE contiguous run in one DRAM bank, ordered
// by subblock (w columns), then K, then column (the K-major [kb, w] blocks the matmul consumes).
//
// NCRISC: streams the core's run in K blocks of kb x w tiles into the in1 ring (transaction-id pipelined).
// BRISC:  gathers rows 0..rows-1 of the tiled input x (DRAM or L1 interleaved) into the in0 tiles (rows >= `rows`
//         of in0 are never read and never written to the output), then writes rows 0..rows-1 of every output
//         tile (padding rows of the output tensor are left untouched: ttnn padding).
// TRISC:  per subblock, K accumulates in FP32 DEST with matmul_block (stock fidelity): the whole K (ks = 0) or in
//         the stock 1D matmul's in0_block_w = ks order (bitwise ttnn.linear); glu: each subblock is w/2 gate then
//         w/2 up columns, SiLU(gate) * up is formed in DEST and only the w/2 products are packed.
#include <cstdint>
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace {
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");           // K tiles
constexpr uint32_t wmax = get_named_compile_time_arg_val("w");          // subblock width (tiles)
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");           // K tiles per streamed block
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");       // valid activation rows
constexpr uint32_t cb_in0 = 0;
constexpr uint32_t cb_in1 = 1;
constexpr uint32_t cb_out = 16;
constexpr uint32_t blk_tiles = kb * wmax;  // CB accounting unit: one ring slot, even for a narrower last subblock
static_assert(kt % kb == 0);
constexpr uint32_t ks = get_named_compile_time_arg_val("ks");  // 0: whole K in DEST; else stock chunked order
constexpr uint32_t cb_part = 24;                               // FP32 partials (ks > 0)
static_assert(ks == 0 || (kb % ks == 0 && kt % ks == 0));
constexpr bool glu = get_named_compile_time_arg_val("glu") == 1;
static_assert(!glu || (wmax % 2 == 0 && ks == 0));
}  // namespace

void kernel_main() {
    // per-core: bank, run offset (bytes, bank-relative), ncols (weight columns), first output column
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
                // A per-trid barrier is only safe on a block issued BEFORE the latest one: the outstanding-by-ID
                // counter can still read 0 while the command buffer is issuing the latest multi-packet read
                // (outstanding == 1 corrupted block tails). The latest block uses the full read barrier.
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
    constexpr uint32_t out_tile_bytes = get_named_compile_time_arg_val("out_tile_bytes");
    constexpr uint32_t out_face_bytes = out_tile_bytes / 4;
    constexpr uint32_t out_row_bytes = out_face_bytes / 16;
    constexpr uint32_t x_tile_bytes = 2048;  // BF16
    constexpr uint32_t x_face_bytes = 512;
    constexpr uint32_t x_row_bytes = 32;
    // Transfers are whole 64 B (DRAM alignment): an odd row count also moves the next (padding) row.
    constexpr uint32_t top = rows < 16 ? rows : 16;
    constexpr uint32_t bottom = rows > 16 ? rows - 16 : 0;
    constexpr uint32_t x_top = (top * x_row_bytes + 63) / 64 * 64;
    constexpr uint32_t x_bottom = (bottom * x_row_bytes + 63) / 64 * 64;
    constexpr uint32_t o_top = (top * out_row_bytes + 63) / 64 * 64;
    constexpr uint32_t o_bottom = (bottom * out_row_bytes + 63) / 64 * 64;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(1), x_tile_bytes);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2), out_tile_bytes);
    const uint32_t first_col = get_arg_val<uint32_t>(3);

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

    uint32_t col = first_col;
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = glu ? wmax / 2 : (s == nsub - 1 ? rem : wmax);
        cb_wait_front(cb_out, wmax);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < wid; ++j) {
            const uint64_t dst = get_noc_addr(col + j, out);
            const uint32_t t = src + j * out_tile_bytes;
            noc_async_write(t, dst, o_top);
            noc_async_write(t + out_face_bytes, dst + out_face_bytes, o_top);
            if constexpr (bottom > 0) {
                noc_async_write(t + 2 * out_face_bytes, dst + 2 * out_face_bytes, o_bottom);
                noc_async_write(t + 3 * out_face_bytes, dst + 3 * out_face_bytes, o_bottom);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, wmax);
        col += wid;
    }
    noc_async_write_barrier();
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    uint32_t cur = wmax;
    matmul_block_init(cb_in0, cb_in1, 0, wmax, 1, kt);
    cb_wait_front(cb_in0, kt);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        if (glu || wid != cur) {  // glu: the SFPU ops below re-program the math thread
            matmul_block_init(cb_in0, cb_in1, 0, wid, 1, kt);
            cur = wid;
        }
        if constexpr (ks == 0) {
            // Whole K in FP32 DEST.
            tile_regs_acquire();
            for (uint32_t b = 0; b < kt / kb; ++b) {
                cb_wait_front(cb_in1, blk_tiles);
                for (uint32_t i = 0; i < kb; ++i) {
                    matmul_block(cb_in0, cb_in1, b * kb + i, i * wid, 0, 0, wid, 1, kt);
                }
                cb_pop_front(cb_in1, blk_tiles);
            }
            constexpr uint32_t h = wmax / 2;
            if constexpr (glu) {
                // DEST tiles [0, h) gate, [h, 2h) up (glu_weight): gate <- SiLU(gate) * up, in FP32.
                silu_tile_init();
                for (uint32_t j = 0; j < h; ++j) {
                    silu_tile(j);
                }
                mul_binary_tile_init();
                for (uint32_t j = 0; j < h; ++j) {
                    mul_binary_tile(j, h + j, j);
                }
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, wmax);
            tile_regs_wait();
            for (uint32_t j = 0; j < (glu ? h : wid); ++j) {
                pack_tile(j, cb_out, j);
            }
            tile_regs_release();
            cb_push_back(cb_out, wmax);
        } else {
            // The stock 1D matmul's order (in0_block_w = ks, packer_l1_acc): each ks-tile K chunk sums in DEST, is
            // packed with FP32 L1 accumulation into the partials, and the last chunk starts from the partials
            // reloaded into DEST (UnpackToDestFp32) -> bitwise ttnn.linear with that in0_block_w.
            constexpr uint32_t nch = kt / (ks ? ks : 1);  // (if constexpr does not discard outside templates)
            for (uint32_t c = 0; c < nch; ++c) {
                const uint32_t k0 = c * ks;
                const uint32_t off = k0 % kb;
                if (off == 0) {
                    cb_wait_front(cb_in1, blk_tiles);
                }
                tile_regs_acquire();
                if (nch > 1 && c == nch - 1) {
                    copy_tile_to_dst_init_short_with_dt(cb_in1, cb_part);
                    cb_wait_front(cb_part, wmax);
                    for (uint32_t j = 0; j < wid; ++j) {
                        copy_tile(cb_part, j, j);
                    }
                    cb_pop_front(cb_part, wmax);
                    reconfig_data_format_srca(cb_part, cb_in1);
                    matmul_block_init(cb_in0, cb_in1, 0, wid, 1, kt);
                }
                for (uint32_t i = 0; i < ks; ++i) {
                    matmul_block(cb_in0, cb_in1, k0 + i, (off + i) * wid, 0, 0, wid, 1, kt);
                }
                if (off + ks == kb) {
                    cb_pop_front(cb_in1, blk_tiles);
                }
                tile_regs_commit();
                const bool last = c == nch - 1;
                const uint32_t dst_cb = last ? cb_out : cb_part;
                cb_reserve_back(dst_cb, wmax);
                tile_regs_wait();
                PACK((pack_reconfig_data_format(dst_cb)));
                PACK((llk_pack_reconfig_l1_acc(!last && c > 0 ? 1 : 0)));
                for (uint32_t j = 0; j < wid; ++j) {
                    pack_tile(j, dst_cb, j);
                }
                tile_regs_release();
                cb_push_back(dst_cb, wmax);
                if (!last && c + 2 < nch) {
                    // consumed in place: the next chunk accumulates onto the same L1 tiles
                    cb_wait_front(cb_part, wmax);
                    cb_pop_front(cb_part, wmax);
                }
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
    }
    cb_pop_front(cb_in0, kt);
#endif
}
