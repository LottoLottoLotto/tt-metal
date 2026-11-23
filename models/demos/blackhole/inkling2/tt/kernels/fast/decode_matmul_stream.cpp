// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Decode matmul streamed from DRAM (ported from LFM2.5): out[rows, N] = x[rows, K] @ W[K, N], rows <= 32.
// Inkling: every worker reads its own input (per-core arg 4 = index into the common x address list), so q|g, k and
// v projections of different inputs run as one program.  Common args: [w, out, x_0, x_1, ...].  seg_mode 1: arg 4
// packs one 2-bit input index per subblock (subblock s: bits 2s..2s+1, <= 16 subblocks); every referenced input is
// read into in0 at tile offset index * kt.
//
// W is BFP8/BFP4/BF16 [K, N] in the "stream" layout of decode_matmul.py (stream_layout): every worker core owns
// `ncols` consecutive weight tile columns and its weight tiles are ONE contiguous run in one DRAM bank, ordered
// by subblock (w columns), then K, then column (the K-major [kb, w] blocks the matmul consumes).
//
// NCRISC: streams the core's run in K blocks of kb x w tiles into the in1 ring (transaction-id pipelined).  pf: the
//         blocks set in per-core arg 5 (bit i = the i-th block of the run, consumption order) were prefetched into
//         L1 (fast/attn_decode.py AttnPrefetch, kernels/fast/attn_prefetch.cpp) and are read from there: per set bit,
//         two more args (noc x << 16 | noc y, L1 address).  NCRISC's per-core args are its own list (BRISC: routes).
// BRISC:  gathers rows 0..rows-1 of the tiled input x (DRAM or L1 interleaved) into the in0 tiles (rows >= `rows`
//         of in0 are never read and never written to the output), then writes rows 0..rows-1 of every output
//         tile (padding rows of the output tensor are left untouched: ttnn padding).
// TRISC:  per subblock, K accumulates in FP32 DEST with matmul_block (stock fidelity): the whole K (ks = 0) or in
//         the stock 1D matmul's in0_block_w = ks order (bitwise ttnn.linear); glu: each subblock is w/2 gate then
//         w/2 up columns, SiLU(gate) * up is formed in DEST and only the w/2 products are packed.
#include <cstdint>
#include "read_vc.hpp"
#include "tools/profiler/kernel_profiler.hpp"
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
constexpr bool seg_mode = get_named_compile_time_arg_val("seg_mode") == 1;
// input index of subblock s (per-core arg 4)
FORCE_INLINE uint32_t sub_seg(uint32_t word, uint32_t s) { return seg_mode ? (word >> (2 * s)) & 3 : word; }
// x_first (decode_matmul.py; DRAM inputs of 2 KB BF16 tile pages on the 8 banks): the input rows are gathered by BOTH
// data-movement RISCs (BRISC on its NOC the even, NCRISC on its NOC the odd tiles of the worker's rotated order) before
// any weight read: with many rows the gather is bound by the NOC paths out of the DRAM banks, not by DRAM
constexpr bool x_first = get_named_compile_time_arg_val("x_first") == 1;
constexpr uint32_t cb_xdone = 3;  // x_first: NCRISC's half landed (one page pushed to BRISC)
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
// tiles i = part, part + 2, .. of the rotated order t = (i + rot) % kt of input x (bank address xaddr) -> in0
FORCE_INLINE void x_gather(uint32_t xaddr, uint32_t in0, uint32_t rot, uint32_t part) {
    constexpr uint32_t top = rows < 16 ? rows : 16, bottom = rows > 16 ? rows - 16 : 0;
    constexpr uint32_t x_top = (top * 32 + 63) / 64 * 64, x_bottom = (bottom * 32 + 63) / 64 * 64;
    for (uint32_t i = part; i < kt; i += 2) {
        const uint32_t t = (i + rot) % kt;
        const uint64_t src = get_noc_addr_from_bank_id<true>(t % 8, xaddr + (t / 8) * 2048);
        const uint32_t dst = in0 + t * 2048;
        if constexpr (bottom == 0 && top >= 8) {  // faces 0 and 1 (adjacent): one read; rows >= `rows` unused
            noc_async_read(src, dst, 1024);
        } else if constexpr (bottom >= 8) {  // the whole tile: one read (rows >= `rows` unused)
            noc_async_read(src, dst, 2048);
        } else {
            noc_async_read(src, dst, x_top);
            noc_async_read(src + 512, dst + 512, x_top);
            if constexpr (bottom > 0) {
                noc_async_read(src + 1024, dst + 1024, x_bottom);
                noc_async_read(src + 1536, dst + 1536, x_bottom);
            }
        }
    }
}
// inputs a worker reads: bit i = x_i (seg_mode: every subblock's index; else the worker's one input, at in0 offset 0)
FORCE_INLINE uint32_t x_need(uint32_t seg_word, uint32_t nsub) {
    uint32_t need = 0;
    if constexpr (seg_mode) {
        for (uint32_t s = 0; s < nsub; ++s) {
            need |= 1u << sub_seg(seg_word, s);
        }
    } else {
        need = 1;
    }
    return need;
}
#endif
}  // namespace

void kernel_main() {
    ink_set_read_vc();
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
    DeviceZoneScopedN("m-stream");
    // x_first: this RISC gathers the odd tiles of the input rows (x_gather), then no weight read before in0 is
    // complete (with many rows the gather would otherwise queue behind every worker's block reads while the compute,
    // waiting for in0, stalls the stream; batched attention)
    if constexpr (x_first) {
        constexpr uint32_t nin = get_named_compile_time_arg_val("n_in");
        const uint32_t seg_word = get_arg_val<uint32_t>(4);
        const uint32_t need = x_need(seg_word, nsub);
        cb_reserve_back(cb_xdone, 1);
        for (uint32_t xi = 0; xi < nin; ++xi) {
            if ((need >> xi) & 1) {
                x_gather(get_common_arg_val<uint32_t>(2 + (seg_mode ? xi : seg_word)),
                         get_write_ptr(cb_in0) + xi * kt * 2048, bank, 1);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_xdone, 1);
        cb_wait_front(cb_in0, kt * nin);
    }
    uint32_t src_off = 0;
    constexpr bool pf = get_named_compile_time_arg_val("pf") == 1;
    const uint32_t pf_mask = pf ? get_arg_val<uint32_t>(5) : 0;
    uint32_t pf_arg = 6, blk = 0;
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
            uint64_t src = src_base + src_off;
            if (pf && ((pf_mask >> blk) & 1)) {
                const uint32_t xy = get_arg_val<uint32_t>(pf_arg);
                src = get_noc_addr(xy >> 16, xy & 0xFFFF, get_arg_val<uint32_t>(pf_arg + 1));
                pf_arg += 2;
            }
            noc_async_read(src, dst, bytes);
            src_off += bytes;
            ++blk;
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
    // pad_zero (rows <= 16, BF16 out): the output tiles get rows 0..rows-1 and zeros elsewhere (finite padding rows)
    constexpr bool pad_zero = get_named_compile_time_arg_val("pad_zero") == 1;
    static_assert(!pad_zero || (rows <= 16 && out_tile_bytes == 2048 && MEM_ZEROS_SIZE >= 512));
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
    const uint32_t seg_word = get_arg_val<uint32_t>(4);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), out_tile_bytes);
    const uint32_t first_col = get_arg_val<uint32_t>(3);
    const uint32_t need = x_need(seg_word, nsub);
    constexpr uint32_t nin = get_named_compile_time_arg_val("n_in");  // in0 slots (inputs), seg_mode; else 1

    cb_reserve_back(cb_in0, kt * nin);
    DeviceZoneScopedN("m-x");
    {
    DeviceZoneScopedN("m-xin");
    for (uint32_t xi = 0; xi < nin; ++xi) {
    if (!((need >> xi) & 1)) {
        continue;
    }
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(2 + (seg_mode ? xi : seg_word)),
                                  get_named_compile_time_arg_val("x_rm") == 1 ? kt * 64 : x_tile_bytes);
    const uint32_t in0 = get_write_ptr(cb_in0) + xi * kt * x_tile_bytes;
    // x_rm (rows == 1): x is a ROW-MAJOR BF16 row (one DRAM page of K * 64 B = 32 values per K tile, i.e. faces 0 and
    // 1 of row 0): one read into the row buffer CB x_rowbuf, then local NOC reads into row 0 of the in0 tiles
    constexpr bool x_rm = get_named_compile_time_arg_val("x_rm") == 1;
    static_assert(!x_rm || rows == 1);
    if constexpr (x_rm) {
        const uint32_t rb = get_write_ptr(get_named_compile_time_arg_val("x_rowbuf"));
        noc_async_read(get_noc_addr(0, x), rb, kt * 64);
        noc_async_read_barrier();
        for (uint32_t t = 0; t < kt; ++t) {
            noc_async_read(get_noc_addr(rb + 64 * t), in0 + t * x_tile_bytes, 32);
            noc_async_read(get_noc_addr(rb + 64 * t + 32), in0 + t * x_tile_bytes + x_face_bytes, 32);
        }
    } else if constexpr (x_first) {  // the even tiles (NCRISC: the odd ones)
        x_gather(get_common_arg_val<uint32_t>(2 + (seg_mode ? xi : seg_word)), in0, get_arg_val<uint32_t>(0), 0);
    } else {
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
    }
    }
    noc_async_read_barrier();
    if constexpr (x_first) {
        cb_wait_front(cb_xdone, 1);
        cb_pop_front(cb_xdone, 1);
    }
    }
    cb_push_back(cb_in0, kt * nin);

    // route (fused consumers): per-core args from index 5 = [dram mask (bit j: local column j is also written to
    // DRAM), n, then n x (j | face << 8 | zero << 9 | target << 12, noc x, noc y, byte offset)]: row r (< rows <= 16)
    // of face `face` of local column j (or 32 zero bytes) -> L1 of core (x, y) at get_write_ptr(route_cb<target>) +
    // offset + r * route_stride (CBs declared first on every core of the program: uniform addresses); once the
    // subblock holding column j is written, one semaphore-0 increment on the destination per route (not per row), so
    // the consumers of a worker's first subblocks start before its last ones.  done_sem (< 16): at the very end one
    // increment of semaphore done_sem on core (done_x, done_y) (all DRAM writes landed).
    constexpr uint32_t route_stride = get_named_compile_time_arg_val("route_stride");
    constexpr bool route = get_named_compile_time_arg_val("route") == 1;
    static_assert(!route || rows <= 16);
    const uint32_t dram_mask = route ? get_arg_val<uint32_t>(5) : 0xFFFFFFFFu;
    const uint32_t n_routes = route ? get_arg_val<uint32_t>(6) : 0;
    uint32_t route_base[4] = {0, 0, 0, 0};
    if constexpr (route) {
        route_base[0] = get_write_ptr(get_named_compile_time_arg_val("route_cb0"));
        route_base[1] = get_write_ptr(get_named_compile_time_arg_val("route_cb1"));
        route_base[2] = get_write_ptr(get_named_compile_time_arg_val("route_cb2"));
        route_base[3] = get_write_ptr(get_named_compile_time_arg_val("route_cb3"));
    }

    uint32_t col = first_col;
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = glu ? wmax / 2 : (s == nsub - 1 ? rem : wmax);
        cb_wait_front(cb_out, wmax);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < wid; ++j) {
            const uint32_t t = src + j * out_tile_bytes;
            if constexpr (route) {
                const uint32_t jl = col + j - first_col;
                for (uint32_t k = 0; k < n_routes; ++k) {
                    const uint32_t w = get_arg_val<uint32_t>(7 + 4 * k);
                    if ((w & 0xFF) != jl) {
                        continue;
                    }
                    const uint64_t dst = get_noc_addr(get_arg_val<uint32_t>(8 + 4 * k), get_arg_val<uint32_t>(9 + 4 * k),
                                                      route_base[(w >> 12) & 3] + get_arg_val<uint32_t>(10 + 4 * k));
                    const uint32_t rsrc = (w >> 9) & 1 ? MEM_ZEROS_BASE : t + ((w >> 8) & 1) * out_face_bytes;
                    for (uint32_t r = 0; r < rows; ++r) {
                        noc_async_write((w >> 9) & 1 ? rsrc : rsrc + r * out_row_bytes, dst + r * route_stride, 32);
                    }
                }
                if (!((dram_mask >> jl) & 1)) {
                    continue;
                }
            }
            const uint64_t dst = get_noc_addr(col + j, out);
            if constexpr (pad_zero) {
                // rows 0..rows-1 (BF16 out), every other row of the tile zero (from the L1 zero region)
                constexpr uint32_t zeros = MEM_ZEROS_BASE;
                constexpr uint32_t live = rows * 32;
                noc_async_write(t, dst, live);
                noc_async_write(t + out_face_bytes, dst + out_face_bytes, live);
                if constexpr (live < out_face_bytes) {
                    noc_async_write(zeros, dst + live, out_face_bytes - live);
                    noc_async_write(zeros, dst + out_face_bytes + live, out_face_bytes - live);
                }
                noc_async_write(zeros, dst + 2 * out_face_bytes, out_face_bytes);
                noc_async_write(zeros, dst + 3 * out_face_bytes, out_face_bytes);
                continue;
            }
            noc_async_write(t, dst, o_top);
            noc_async_write(t + out_face_bytes, dst + out_face_bytes, o_top);
            if constexpr (bottom > 0) {
                noc_async_write(t + 2 * out_face_bytes, dst + 2 * out_face_bytes, o_bottom);
                noc_async_write(t + 3 * out_face_bytes, dst + 3 * out_face_bytes, o_bottom);
            }
        }
        if constexpr (route) {
            bool any = false;
            for (uint32_t k = 0; k < n_routes; ++k) {
                const uint32_t jl = get_arg_val<uint32_t>(7 + 4 * k) & 0xFF;
                if (jl < col - first_col || jl >= col - first_col + wid) {
                    continue;
                }
                if (!any) {
                    noc_async_write_barrier();
                    any = true;
                }
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(8 + 4 * k), get_arg_val<uint32_t>(9 + 4 * k),
                                               get_semaphore(0)),
                                  1);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, wmax);
        col += wid;
    }
    noc_async_write_barrier();
    constexpr uint32_t done_sem = get_named_compile_time_arg_val("done_sem");
    if constexpr (done_sem < 16) {
        noc_semaphore_inc(get_noc_addr(get_named_compile_time_arg_val("done_x"), get_named_compile_time_arg_val("done_y"),
                                       get_semaphore(done_sem)),
                          1);
    }
    noc_async_atomic_barrier();
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    uint32_t cur = wmax;
    matmul_block_init(cb_in0, cb_in1, 0, wmax, 1, kt);
    constexpr uint32_t nin = get_named_compile_time_arg_val("n_in");
    const uint32_t seg_word = get_arg_val<uint32_t>(4);
    cb_wait_front(cb_in0, kt * nin);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        const uint32_t x0 = seg_mode ? sub_seg(seg_word, s) * kt : 0;  // in0 tile offset of this subblock's input
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
                    matmul_block(cb_in0, cb_in1, x0 + b * kb + i, i * wid, 0, 0, wid, 1, kt);
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
                    matmul_block(cb_in0, cb_in1, x0 + k0 + i, (off + i) * wid, 0, 0, wid, 1, kt);
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
    cb_pop_front(cb_in0, kt * nin);
#endif
}
