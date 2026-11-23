// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Fused decode LM head: logits = 30 tanh((x @ W) / 30) streamed from DRAM, + the per-row argmax (greedy token).
// Bitwise the stock head (layers.LMHead: ttnn.linear with the pad bias, in0_block_w = 2, packer L1 accumulation,
// then ttnn mul / tanh / mul on FP32 tensors) on the valid columns; padding columns read -30 as there.
//
// W: BFP8/BFP4 [K, N] in decode_matmul.py's stream layout (every worker = `ncols` consecutive weight tile columns,
// ONE contiguous run in one DRAM bank).  Workers ordered by their first column (worker index = per-core arg 4).
//
// Activation / partial / output tiles in L1 are [th, 32] (th = 16 for rows <= 16, else 32): the FPU and the packer
// work on th rows instead of 32 (each row's values are unchanged).
// NCRISC: streams the core's run in kb x w blocks into the in1 ring (decode_matmul_stream.cpp's reader; NOC 0,
//         workers placed by moe.stream_workers: link-disjoint DRAM response paths).
// TRISC:  per subblock of w columns, K in chunks of ks = 2 tiles: each chunk's products sum in FP32 DEST and are
//         packed with FP32 L1 accumulation into the partials (chunk 0 overwrites) = the stock FUSE_BIAS order; then
//         the partials are unpacked through srcA (TF32, as the stock bias add) + a BF16 zero row-broadcast bias tile
//         (add_tiles_bcast_rows), and the softcap runs on DEST: SFPU mul 1/30, tanh (accurate), mul 30 (FP32).
// BRISC:  zero bias tile, gathers rows 0..rows-1 of x into the in0 tiles, then per output tile: padding columns
//         (>= valid) of rows < rows set to -30, rows < rows written to DRAM, and (argmax) the row maxima scanned on
//         the RISC (FP32 bits mapped to a monotonic unsigned key, -0 == +0; ties -> lowest column).  Every worker
//         posts (key, column) per row to worker 0, which reduces in worker (= column) order and writes the uint32
//         column of row r to word r of page 0 of `ids`.
// Common runtime args: [w, out, x, ids].  Per-core: [bank, run offset, ncols, first col, worker index].
#include <cstdint>
#include "read_vc.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#endif

namespace {
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");      // K tiles
constexpr uint32_t wmax = get_named_compile_time_arg_val("w");     // subblock width (tiles)
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");      // K tiles per streamed block
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");  // valid activation rows (<= 32)
constexpr uint32_t ks = 2;                                         // stock in0_block_w of this shape
constexpr uint32_t cb_in0 = 0;
constexpr uint32_t cb_in1 = 1;
constexpr uint32_t cb_zero = 3;   // BF16 zero bias tile
constexpr uint32_t cb_slots = 4;  // argmax gather slots (worker 0)
constexpr uint32_t cb_mine = 5;   // this worker's (key, column) rows
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_part = 24;  // FP32 partials
constexpr uint32_t blk_tiles = kb * wmax;
static_assert(kt % kb == 0 && kb % ks == 0 && rows >= 1 && rows <= 32);
}  // namespace

void kernel_main() {
    ink_set_read_vc();
    const uint32_t ncols = get_arg_val<uint32_t>(2);
    const uint32_t nsub = (ncols + wmax - 1) / wmax;
    const uint32_t rem = ncols - (nsub - 1) * wmax;
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t tile_bytes = get_named_compile_time_arg_val("in1_tile_bytes");
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t outstanding = get_named_compile_time_arg_val("outstanding");
    static_assert(outstanding >= 1 && outstanding < slots && outstanding <= 8);
    constexpr uint32_t slot_bytes = blk_tiles * tile_bytes;
    const uint64_t src_base =
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(0), get_common_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(1));
    const uint32_t ring_end = get_write_ptr(cb_in1) + slots * slot_bytes;
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
                // per-trid barrier only on a block issued before the latest one (see decode_matmul_stream.cpp)
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
    constexpr uint32_t valid = get_named_compile_time_arg_val("valid");  // logit columns (the rest is padding)
    constexpr bool argmax = get_named_compile_time_arg_val("argmax") == 1;
    constexpr uint32_t workers = get_named_compile_time_arg_val("workers");
    constexpr uint32_t gather_x = get_named_compile_time_arg_val("gather_x");
    constexpr uint32_t gather_y = get_named_compile_time_arg_val("gather_y");
    constexpr uint32_t ids_page = get_named_compile_time_arg_val("ids_page");
    static_assert(!argmax || rows <= 16);
    // in0 / out CB tiles are [th, 32]: 2 faces of [16, 16] (+ 2 more for th = 32).  DRAM tiles (x in, logits
    // out) are 32 x 32.  (Partial-face tiles, th < 16, put face 1 of the matmul output elsewhere: not used.)
    constexpr uint32_t th = get_named_compile_time_arg_val("th");
    static_assert(th >= rows && (th == 16 || th == 32));
    constexpr uint32_t fr = 16;                          // face rows
    constexpr uint32_t out_row_bytes = 64;               // FP32
    constexpr uint32_t out_face_bytes = fr * out_row_bytes;
    constexpr uint32_t out_tile_bytes = th * 32 * 4;
    constexpr uint32_t dram_out_face_bytes = 1024;
    constexpr uint32_t x_row_bytes = 32;                 // BF16
    constexpr uint32_t x_face_bytes = fr * x_row_bytes;
    constexpr uint32_t x_tile_bytes = th * 32 * 2;
    constexpr uint32_t dram_x_face_bytes = 512;
    constexpr uint32_t top = rows < 16 ? rows : 16;
    constexpr uint32_t bottom = rows > 16 ? rows - 16 : 0;
    // whole 64 B transfers (DRAM alignment): an odd row count also moves the next (padding) x row (fits in the tile)
    constexpr uint32_t x_top = (top * x_row_bytes + 63) / 64 * 64;
    constexpr uint32_t x_bottom = (bottom * x_row_bytes + 63) / 64 * 64;
    constexpr uint32_t o_top = top * out_row_bytes;
    constexpr uint32_t o_bottom = bottom * out_row_bytes;
    constexpr uint32_t slot_bytes = ((rows * 8 + 15) / 16) * 16;  // (key, column) per row
    constexpr uint32_t slot_words = slot_bytes / 4;
    const auto x = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2),
                                  2048);
    const auto out = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1),
                                    4096);
    const uint32_t first_col = get_arg_val<uint32_t>(3);
    const uint32_t widx = get_arg_val<uint32_t>(4);

    // zero bias tile (BF16)
    cb_reserve_back(cb_zero, 1);
    {
        const uint32_t z = get_write_ptr(cb_zero);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < x_tile_bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, z + off, x_tile_bytes - off < MEM_ZEROS_SIZE ? x_tile_bytes - off : MEM_ZEROS_SIZE);
        }
    }
    // x rows
    cb_reserve_back(cb_in0, kt);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t t = 0; t < kt; ++t) {
        const uint64_t src = x.get_noc_addr(t);
        const uint32_t dst = in0 + t * x_tile_bytes;
        noc_async_read(src, dst, x_top);
        noc_async_read(src + dram_x_face_bytes, dst + x_face_bytes, x_top);
        if constexpr (bottom > 0) {
            noc_async_read(src + 2 * dram_x_face_bytes, dst + 2 * x_face_bytes, x_bottom);
            noc_async_read(src + 3 * dram_x_face_bytes, dst + 3 * x_face_bytes, x_bottom);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_zero, 1);
    cb_push_back(cb_in0, kt);

    uint32_t best_key[argmax ? rows : 1];
    uint32_t best_idx[argmax ? rows : 1];
    if constexpr (argmax) {
        for (uint32_t r = 0; r < rows; ++r) {
            best_key[r] = 0;
            best_idx[r] = first_col * 32;
        }
    }
    uint32_t col = first_col;
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        cb_wait_front(cb_out, wmax);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < wid; ++j) {
            const uint32_t t = src + j * out_tile_bytes;
            const uint32_t c0 = (col + j) * 32;
            if (c0 + 32 > valid) {  // padding columns read -30 (the stock pad bias -1e4 through the softcap)
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint32_t rb = t + (r >= 16 ? 2 * out_face_bytes + (r - 16) * out_row_bytes : r * out_row_bytes);
                    for (uint32_t c = (valid > c0 ? valid - c0 : 0); c < 32; ++c) {
                        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rb + (c >= 16 ? out_face_bytes : 0) + (c & 15) * 4) =
                            0xC1F00000u;  // -30.0f
                    }
                }
            }
            if constexpr (argmax) {
                const uint32_t n = c0 + 32 > valid ? (valid > c0 ? valid - c0 : 0) : 32;
                for (uint32_t r = 0; r < rows; ++r) {
                    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t + r * out_row_bytes);
                    uint32_t bk = best_key[r], bi = best_idx[r];
                    for (uint32_t c = 0; c < n; ++c) {
                        const uint32_t v = p[(c >> 4) * (out_face_bytes / 4) + (c & 15)];
                        const uint32_t u = v == 0x80000000u ? 0u : v;
                        const uint32_t key = u ^ ((uint32_t)((int32_t)u >> 31) | 0x80000000u);
                        if (key > bk) {
                            bk = key;
                            bi = c0 + c;
                        }
                    }
                    best_key[r] = bk;
                    best_idx[r] = bi;
                }
            }
            const uint64_t dst = out.get_noc_addr(col + j);
            noc_async_write(t, dst, o_top);
            noc_async_write(t + out_face_bytes, dst + dram_out_face_bytes, o_top);
            if constexpr (bottom > 0) {
                noc_async_write(t + 2 * out_face_bytes, dst + 2 * dram_out_face_bytes, o_bottom);
                noc_async_write(t + 3 * out_face_bytes, dst + 3 * dram_out_face_bytes, o_bottom);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, wmax);
        col += wid;
    }
    noc_async_write_barrier();
    if constexpr (argmax) {
        const uint32_t mine = get_write_ptr(cb_mine);
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mine);
        for (uint32_t r = 0; r < rows; ++r) {
            m[2 * r] = best_key[r];
            m[2 * r + 1] = best_idx[r];
        }
        const uint32_t slots_base = get_write_ptr(cb_slots);
        const uint32_t sem = get_semaphore(0);
        noc_async_write(mine, get_noc_addr(gather_x, gather_y, slots_base + widx * slot_bytes), slot_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(gather_x, gather_y, sem), 1);
        noc_async_atomic_barrier();
        if (widx == 0) {
            volatile tt_l1_ptr uint32_t* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
            noc_semaphore_wait(count, workers);
            noc_semaphore_set(count, 0);
            volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots_base);
            for (uint32_t r = 0; r < rows; ++r) {
                uint32_t key = 0, idx = 0;
                for (uint32_t w = 0; w < workers; ++w) {
                    const uint32_t k = sl[w * slot_words + 2 * r];
                    if (k > key) {
                        key = k;
                        idx = sl[w * slot_words + 2 * r + 1];
                    }
                }
                m[r] = idx;
            }
            const auto ids = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                            get_common_arg_val<uint32_t>(3), ids_page);
            noc_async_write(mine, ids.get_noc_addr(0), rows * 4);
            noc_async_write_barrier();
        }
    }
#else
    constexpr uint32_t inv_cap = get_named_compile_time_arg_val("inv_cap");  // FP32 bits of 1 / softcap
    constexpr uint32_t cap = get_named_compile_time_arg_val("cap");          // FP32 bits of softcap
    constexpr uint32_t nch = kt / ks;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    cb_wait_front(cb_in0, kt);
    cb_wait_front(cb_zero, 1);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        matmul_block_init(cb_in0, cb_in1, 0, wid, 1, kt);
        PACK((pack_reconfig_data_format(cb_part)));
        for (uint32_t c = 0; c < nch; ++c) {
            const uint32_t off = (c * ks) % kb;
            if (off == 0) {
                cb_wait_front(cb_in1, blk_tiles);
            }
            tile_regs_acquire();
            for (uint32_t i = 0; i < ks; ++i) {
                matmul_block(cb_in0, cb_in1, c * ks + i, (off + i) * wid, 0, 0, wid, 1, kt);
            }
            if (off + ks == kb) {
                cb_pop_front(cb_in1, blk_tiles);
            }
            tile_regs_commit();
            cb_reserve_back(cb_part, wmax);
            tile_regs_wait();
            PACK((llk_pack_reconfig_l1_acc(c > 0 ? 1 : 0)));
            for (uint32_t j = 0; j < wid; ++j) {
                pack_tile(j, cb_part, j);
            }
            tile_regs_release();
            cb_push_back(cb_part, wmax);
            if (c + 1 < nch) {  // accumulated in place by the next chunk
                cb_wait_front(cb_part, wmax);
                cb_pop_front(cb_part, wmax);
            }
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
        PACK((pack_reconfig_data_format(cb_out)));
        // the stock bias stage: partials (srcA) + bias row (srcB), then the softcap on DEST
        reconfig_data_format(cb_in1, cb_part, cb_in0, cb_zero);
        cb_wait_front(cb_part, wmax);
        cb_reserve_back(cb_out, wmax);
        add_bcast_rows_init_short(cb_part, cb_zero);
        tile_regs_acquire();
        for (uint32_t j = 0; j < wid; ++j) {
            add_tiles_bcast_rows(cb_part, cb_zero, j, 0, j);
        }
        // SFPU mul by the scalar parameter == ttnn.mul(FP32 tensor, scalar) bitwise (tests/test_fast_head.py)
        binop_with_scalar_tile_init();
        for (uint32_t j = 0; j < wid; ++j) {
            mul_unary_tile(j, inv_cap);
        }
        tanh_tile_init();
        for (uint32_t j = 0; j < wid; ++j) {
            tanh_tile(j);
        }
        binop_with_scalar_tile_init();
        for (uint32_t j = 0; j < wid; ++j) {
            mul_unary_tile(j, cap);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < wid; ++j) {
            pack_tile(j, cb_out, j);
        }
        tile_regs_release();
        cb_push_back(cb_out, wmax);
        cb_pop_front(cb_part, wmax);
        reconfig_data_format(cb_part, cb_in1, cb_zero, cb_in0);
    }
    cb_pop_front(cb_in0, kt);
#endif
}
