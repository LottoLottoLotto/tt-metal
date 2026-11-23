// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 dense MLP decode in ONE program (FusedMLPDecode, mlp_fused.py):
//     h = a + rms_norm(b) * post (or a + b, or a);  n = rms_norm(h) * gamma        (norm_prologue.hpp)
//     g = SiLU(n @ W1) * (n @ W3)                                                  (w13, glu stream layout)
//     f = g @ W2                                                                   (w2, stream layout)
// = NormStreamDecodeMatmul(w13) followed by StreamDecodeMatmul(w2), bitwise (same accumulation order, same packs),
// on one gx x gy rectangle; every core is a w13 worker and a w2 worker of the same DRAM bank (both stream layouts
// share cores_per_bank = gy).
//
// NCRISC: streams the core's w13 run into ring CB 1 from kernel start (i.e. during the norm), then its w2 run into
//         ring CB 2 (sized to hold most or all of it): the bank streams back to back while g is assembled.
// BRISC:  np_dm() (n -> CB 0 on every core); per w13 subblock, the w/2 packed g tiles (rows < np_rows) go to the
//         master's CB 3 at their column (unicast, th x 32 tiles); then one gather increment; the master multicasts
//         the whole g block (one transfer) to the rectangle + "g ready"; every core pushes CB 3 (w2's in0); then
//         writes its f rows.
// TRISC:  np_compute(), the w13 glu matmul (as decode_matmul_norm.cpp), then the w2 matmul over CB 3 (whole K in
//         FP32 DEST, as decode_matmul_stream.cpp with k_split 0) at fidelity mm_fid.
#include <cstdint>
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#endif
#include "norm_prologue.hpp"
#include "mm_fidelity.hpp"
#if MF_PROF
// Device-profiler zones, all opened at kernel start (nested): each zone's duration is the time of that milestone.
#include "tools/profiler/kernel_profiler.hpp"
#define MF_ZONE(name) DeviceZoneScopedN(name)
#else
#define MF_ZONE(name)
#endif

namespace {
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");    // w13 K tiles (hidden)
constexpr uint32_t wmax = get_named_compile_time_arg_val("w");   // w13 subblock width (gate + up tiles)
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");    // w13 K tiles per streamed block
constexpr uint32_t kt2 = get_named_compile_time_arg_val("kt2");  // w2 K tiles (intermediate) = g tiles
constexpr uint32_t w2max = get_named_compile_time_arg_val("w2");
constexpr uint32_t kb2 = get_named_compile_time_arg_val("kb2");
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
constexpr uint32_t cb_in0 = 0;  // = np_cb_x: n
constexpr uint32_t cb_in1 = 1;  // w13 ring
constexpr uint32_t cb_in2 = 2;  // w2 ring
constexpr uint32_t cb_g = 3;    // g block (w2 in0), identical address on every core
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_out2 = 17;
constexpr uint32_t gh = wmax / 2;  // g tiles per w13 subblock
// g (w2 in0) and f (w2 out) tiles are th x 32 (th = 8 / 16 / 32: the smallest that holds the rows; the matmul then
// runs on 2 partial faces and the g block multicast moves th / 32 of a full tile per g tile; valid rows identical)
constexpr uint32_t th = get_named_compile_time_arg_val("th");
constexpr uint32_t kSmall = th * 64;  // th x 32 BF16 tile bytes
static_assert((th == 8 || th == 16 || th == 32) && rows <= th);
static_assert(wmax % 2 == 0 && kt % kb == 0 && kt2 % kb2 == 0);
}  // namespace

#if defined(COMPILE_FOR_NCRISC)
// Streams `ncols` weight columns (a contiguous run in one bank, stream layout) in kb x w blocks into ring `cb`.
template <uint32_t cb, uint32_t tile_bytes, uint32_t w, uint32_t kbk, uint32_t ktk, uint32_t slots, uint32_t outstanding>
FORCE_INLINE void stream_run(uint64_t src_base, uint32_t ncols) {
    static_assert(outstanding >= 1 && outstanding < slots && outstanding <= 8);
    constexpr uint32_t blk_tiles = kbk * w;
    constexpr uint32_t slot_bytes = blk_tiles * tile_bytes;
    const uint32_t nsub = (ncols + w - 1) / w;
    const uint32_t rem = ncols - (nsub - 1) * w;
    const uint32_t ring_end = get_write_ptr(cb) + slots * slot_bytes;
    uint32_t src_off = 0;
    uint32_t pending = 0, issue_trid = 1, wait_trid = 1;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t bytes = kbk * (s == nsub - 1 ? rem : w) * tile_bytes;
        for (uint32_t b = 0; b < ktk / kbk; ++b) {
            cb_reserve_back(cb, blk_tiles * (pending + 1));
            uint32_t dst = get_write_ptr(cb) + pending * slot_bytes;
            if (dst >= ring_end) {
                dst -= slots * slot_bytes;
            }
            noc_async_read_set_trid(issue_trid);
            noc_async_read(src_base + src_off, dst, bytes);
            src_off += bytes;
            issue_trid = issue_trid == outstanding ? 1 : issue_trid + 1;
            if (++pending == outstanding) {
                // per-trid barrier only on a block issued before the latest one (decode_matmul_stream.cpp)
                if constexpr (outstanding == 1) {
                    noc_async_read_barrier();
                } else {
                    noc_async_read_barrier_with_trid(wait_trid);
                }
                cb_push_back(cb, blk_tiles);
                wait_trid = wait_trid == outstanding ? 1 : wait_trid + 1;
                --pending;
            }
        }
    }
    noc_async_read_barrier();
    for (; pending > 0; --pending) {
        cb_push_back(cb, blk_tiles);
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
}
#endif

void kernel_main() {
    // per-core: bank, w13 run offset (bytes), w13 ncols, first g column, w2 run offset, w2 ncols, first f column
    const uint32_t ncols = get_arg_val<uint32_t>(2);
    const uint32_t nsub = (ncols + wmax - 1) / wmax;
    const uint32_t rem = ncols - (nsub - 1) * wmax;
    const uint32_t ncols2 = get_arg_val<uint32_t>(5);
    const uint32_t nsub2 = (ncols2 + w2max - 1) / w2max;
    const uint32_t rem2 = ncols2 - (nsub2 - 1) * w2max;
#if defined(COMPILE_FOR_NCRISC)
    const uint32_t bank = get_arg_val<uint32_t>(0);
    MF_ZONE("n-w2-read");
    {
    MF_ZONE("n-w13-read");
    stream_run<cb_in1, get_named_compile_time_arg_val("in1_tile_bytes"), wmax, kb, kt,
               get_named_compile_time_arg_val("slots"), get_named_compile_time_arg_val("outstanding")>(
        get_noc_addr_from_bank_id<true>(bank, get_common_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(1)), ncols);
    }
    if constexpr (get_named_compile_time_arg_val("w2_gate") == 1) {
        // w2 reads start only once EVERY core's w13 run is in L1: a core whose w13 share is done early would
        // otherwise stream its w2 run at full speed and starve the w13 stragglers of its bank, while w2 cannot be
        // computed before the last g tile anyway (profiled: last w13 core 42 -> 84 us).
        constexpr uint32_t n_cores = get_named_compile_time_arg_val("np_cores");
        constexpr uint32_t master_x = get_named_compile_time_arg_val("np_master_x");
        constexpr uint32_t master_y = get_named_compile_time_arg_val("np_master_y");
        const uint32_t sem_done = get_semaphore(get_named_compile_time_arg_val("sem_w13_done"));
        const uint32_t sem_go = get_semaphore(get_named_compile_time_arg_val("sem_w2_go"));
        noc_semaphore_inc(get_noc_addr(master_x, master_y, sem_done), 1);
        if (np::core_index() == 0) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done), n_cores);
            uint32_t mx0 = get_named_compile_time_arg_val("np_mc_x0"), my0 = get_named_compile_time_arg_val("np_mc_y0");
            uint32_t mx1 = get_named_compile_time_arg_val("np_mc_x1"), my1 = get_named_compile_time_arg_val("np_mc_y1");
            if (noc_index == 1) {
                const uint32_t tx = mx0, ty = my0;
                mx0 = mx1, my0 = my1, mx1 = tx, my1 = ty;
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go) = 1;
            if constexpr (n_cores > 1) {
                noc_semaphore_set_multicast(sem_go, get_noc_multicast_addr(mx0, my0, mx1, my1, sem_go), n_cores - 1);
            }
        } else {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go), 1);
        }
        noc_async_atomic_barrier();
    }
    stream_run<cb_in2, get_named_compile_time_arg_val("in2_tile_bytes"), w2max, kb2, kt2,
               get_named_compile_time_arg_val("slots2"), get_named_compile_time_arg_val("outstanding2")>(
        get_noc_addr_from_bank_id<true>(bank, get_common_arg_val<uint32_t>(1) + get_arg_val<uint32_t>(4)), ncols2);
#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t tb = 2048;  // full BF16 tile (w13 out, f tensor)
    constexpr uint32_t face = 512;
    constexpr uint32_t row_bytes = 32;
    constexpr uint32_t master_x = get_named_compile_time_arg_val("np_master_x");
    constexpr uint32_t master_y = get_named_compile_time_arg_val("np_master_y");
    constexpr uint32_t n_cores = get_named_compile_time_arg_val("np_cores");
    constexpr uint32_t sem_gg_id = get_named_compile_time_arg_val("sem_g_gather");
    constexpr uint32_t sem_gr_id = get_named_compile_time_arg_val("sem_g_ready");
    // Transfers are whole 64 B (DRAM alignment): an odd row count also moves the next (padding) row.
    constexpr uint32_t top = rows < 16 ? rows : 16;
    constexpr uint32_t bottom = rows > 16 ? rows - 16 : 0;
    constexpr uint32_t o_top = (top * row_bytes + 63) / 64 * 64;
    constexpr uint32_t o_bottom = (bottom * row_bytes + 63) / 64 * 64;
    constexpr auto f_args = TensorAccessorArgs<0>();
    const auto f = TensorAccessor(f_args, get_common_arg_val<uint32_t>(0), tb);

    const uint32_t gb = get_write_ptr(cb_g);  // identical on every core
    uint32_t col = get_arg_val<uint32_t>(3);
    MF_ZONE("b-f-written");
    {
    MF_ZONE("b-g-ready");
    {
    MF_ZONE("b-g-sent");
    {
    MF_ZONE("b-n-ready");
    // norm prologue: n of the full row block lands in cb_in0 of every core (pushed, kt tiles)
    np_dm<f_args.next_compile_time_args_offset(), 1>();
    }
    // g tiles -> the master's g block (th x 32 tiles)
    for (uint32_t s = 0; s < nsub; ++s) {
        cb_wait_front(cb_out, wmax);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < gh; ++j) {
            const uint32_t t = src + j * tb, dst = gb + (col + j) * kSmall;
            if constexpr (th < 32) {  // rows of faces 0 / 1 -> the th x 32 tile's two faces
                for (uint32_t fc = 0; fc < 2; ++fc) {
                    noc_async_write(t + fc * face, get_noc_addr(master_x, master_y, dst + fc * th * row_bytes),
                                    rows * row_bytes);
                }
            } else {
                noc_async_write(t, get_noc_addr(master_x, master_y, dst), tb);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, wmax);
        col += gh;
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(master_x, master_y, get_semaphore(sem_gg_id)), 1);
    }
    cb_reserve_back(cb_g, kt2);
    if (np::core_index() == 0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gg_id)), n_cores);
        if constexpr (n_cores > 1) {
            uint32_t mx0 = get_named_compile_time_arg_val("np_mc_x0"), my0 = get_named_compile_time_arg_val("np_mc_y0");
            uint32_t mx1 = get_named_compile_time_arg_val("np_mc_x1"), my1 = get_named_compile_time_arg_val("np_mc_y1");
            if (noc_index == 1) {
                const uint32_t tx = mx0, ty = my0;
                mx0 = mx1, my0 = my1, mx1 = tx, my1 = ty;
            }
            noc_async_write_multicast(gb, get_noc_multicast_addr(mx0, my0, mx1, my1, gb), kt2 * kSmall, n_cores - 1,
                                      true);
            const uint32_t sem = get_semaphore(sem_gr_id);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem) = 1;
            noc_semaphore_set_multicast(sem, get_noc_multicast_addr(mx0, my0, mx1, my1, sem), n_cores - 1);
        }
    } else {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gr_id)), 1);
    }
    cb_push_back(cb_g, kt2);
    }
    // f rows
    col = get_arg_val<uint32_t>(6);
    for (uint32_t s = 0; s < nsub2; ++s) {
        const uint32_t wid = s == nsub2 - 1 ? rem2 : w2max;
        cb_wait_front(cb_out2, w2max);
        const uint32_t src = get_read_ptr(cb_out2);
        for (uint32_t j = 0; j < wid; ++j) {
            const uint64_t dst = get_noc_addr(col + j, f);
            const uint32_t t = src + j * kSmall;
            noc_async_write(t, dst, o_top);
            noc_async_write(t + (th < 32 ? th * row_bytes : face), dst + face, o_top);
            if constexpr (bottom > 0) {
                noc_async_write(t + 2 * face, dst + 2 * face, o_bottom);
                noc_async_write(t + 3 * face, dst + 3 * face, o_bottom);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out2, w2max);
        col += wid;
    }
    noc_async_write_barrier();
#else
    constexpr int mm_fid = get_named_compile_time_arg_val("mm_fid");
    MF_ZONE("t-w2-done");
    {
    MF_ZONE("t-g-in");
    {
    MF_ZONE("t-w13-done");
    {
    MF_ZONE("t-norm-done");
    np_compute();
    }
    // ---- w13 + SwiGLU (decode_matmul_norm.cpp's glu path)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    constexpr uint32_t blk = kb * wmax;
    mm_block_init_f<mm_fid>(cb_in0, cb_in1, 0, wmax, 1, kt);
    cb_wait_front(cb_in0, kt);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        mm_block_init_f<mm_fid>(cb_in0, cb_in1, 0, wid, 1, kt);  // the SFPU ops below re-program the math thread
        tile_regs_acquire();
        for (uint32_t b = 0; b < kt / kb; ++b) {
            cb_wait_front(cb_in1, blk);
            for (uint32_t i = 0; i < kb; ++i) {
                mm_block_f<mm_fid>(cb_in0, cb_in1, b * kb + i, i * wid, 0, 0, wid, 1, kt);
            }
            cb_pop_front(cb_in1, blk);
        }
        // DEST tiles [0, gh) gate, [gh, 2 gh) up (glu_weight): gate <- SiLU(gate) * up, in FP32.
        silu_tile_init();
        for (uint32_t j = 0; j < gh; ++j) {
            silu_tile(j);
        }
        mul_binary_tile_init();
        for (uint32_t j = 0; j < gh; ++j) {
            mul_binary_tile(j, gh + j, j);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, wmax);
        tile_regs_wait();
        for (uint32_t j = 0; j < gh; ++j) {
            pack_tile(j, cb_out, j);
        }
        tile_regs_release();
        cb_push_back(cb_out, wmax);
    }
    cb_pop_front(cb_in0, kt);
    }
    cb_wait_front(cb_g, kt2);
    }
    // ---- w2 over the g block (decode_matmul_stream.cpp, whole K in DEST; th x 32 in0 / out tiles)
    reconfig_data_format_srca(cb_in2);
    reconfig_data_format_srcb<true>(cb_g);
    pack_reconfig_data_format<true>(cb_out2);
    constexpr uint32_t blk2 = kb2 * w2max;
    uint32_t cur = w2max;
    mm_block_init_f<mm_fid>(cb_g, cb_in2, 0, w2max, 1, kt2);
    cb_wait_front(cb_g, kt2);
    for (uint32_t s = 0; s < nsub2; ++s) {
        const uint32_t wid = s == nsub2 - 1 ? rem2 : w2max;
        if (wid != cur) {
            mm_block_init_f<mm_fid>(cb_g, cb_in2, 0, wid, 1, kt2);
            cur = wid;
        }
        tile_regs_acquire();
        for (uint32_t b = 0; b < kt2 / kb2; ++b) {
            cb_wait_front(cb_in2, blk2);
            for (uint32_t i = 0; i < kb2; ++i) {
                mm_block_f<mm_fid>(cb_g, cb_in2, b * kb2 + i, i * wid, 0, 0, wid, 1, kt2);
            }
            cb_pop_front(cb_in2, blk2);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out2, w2max);
        tile_regs_wait();
        for (uint32_t j = 0; j < wid; ++j) {
            pack_tile(j, cb_out2, j);
        }
        tile_regs_release();
        cb_push_back(cb_out2, w2max);
    }
    cb_pop_front(cb_g, kt2);
#endif
}
