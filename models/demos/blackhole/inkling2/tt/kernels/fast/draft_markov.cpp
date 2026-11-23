// SPDX-License-Identifier: Apache-2.0
// One Markov step of the drafter's head, fused (fast/draft_ops.py MarkovArgmax):
//     token = argmax_v bf16(base[row, v] + fold_b[v] + latent . fold[:, v]),  v < valid (ties -> lowest v)
// = the decode_matmul_stream.cpp weight stream (fold [K = rank, N = vocab] in the stream layout, one contiguous run per
// worker) with the base row and fold_b added in FP32 DEST before the K accumulation, and an argmax epilogue instead
// of the output write: no FP32 [32, V] tensors.
//
// NCRISC: streams the worker's fold run (kb x w blocks, transaction-id pipelined; as decode_matmul_stream.cpp).
// BRISC:  in0 = row 0 of the latent ([1, 1, 32, rank] BF16 TILE: face rows 0-1 pairs); per subblock the base tiles
//         (row `brow` of the base logits [1, 1, rows, Vp] BF16 TILE moved to row 0) and fold_b tiles (row 0 of a
//         [1, 1, 32, Vp] BF16 TILE); then scans row 0 of the packed FP32 results (RNE to BF16, monotonic integer key,
//         first strict maximum) and posts (key, index) to the gather core (worker 0), which reduces in worker order
//         (= increasing columns) and writes the id to word 0 of the uint32 output row and to word 0 of page brow - 1
//         of the draft log (uint32 row-major [>= 7, 32]: one host read returns every draft of the block).
//         users = U > 1 (batched drafter): user u = row r_u = brow * U + u of every tile (latent row u moved there
//         once; base row r_u and fold_b row r_u read in place: fold_b must hold the bias in every row), U argmaxes
//         in one fold stream; user u's id -> word u of the output row and of log page brow - 1.
// TRISC:  DEST[j] = base + fold_b (BF16 -> FP32 copies, SFPU add), then matmul_block accumulates latent @ fold.
// Common args: [w, latent, base, fold_b, out, log].  Per-core: [bank, run offset, ncols, first col, worker].
#include <cstdint>
#include "read_vc.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace {
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
constexpr uint32_t wmax = get_named_compile_time_arg_val("w");
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");
constexpr uint32_t cb_in0 = 0;
constexpr uint32_t cb_in1 = 1;
constexpr uint32_t cb_base = 3;
constexpr uint32_t cb_fb = 4;
constexpr uint32_t cb_stage = 5;
constexpr uint32_t cb_slots = 6;
constexpr uint32_t cb_out = 16;
constexpr uint32_t blk_tiles = kb * wmax;
static_assert(kt % kb == 0);
static_assert(2 * wmax <= 4, "DEST (FP32, half) holds 2 w tiles");
// FP32 / BF16 bits -> unsigned key with the float order (-0 just below +0)
FORCE_INLINE uint32_t mono(uint32_t v) { return v ^ ((uint32_t)((int32_t)v >> 31) | 0x80000000u); }
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
    constexpr uint32_t brow = get_named_compile_time_arg_val("brow");  // base row of this step (user 0)
    constexpr uint32_t users = get_named_compile_time_arg_val("users");  // user u: latent row u, base row brow*users+u
    constexpr uint32_t valid = get_named_compile_time_arg_val("valid");
    constexpr uint32_t workers = get_named_compile_time_arg_val("workers");
    constexpr uint32_t gather_x = get_named_compile_time_arg_val("gather_x");
    constexpr uint32_t gather_y = get_named_compile_time_arg_val("gather_y");
    static_assert(brow < 16 && users >= 1 && users <= 4 && brow * users + users - 1 < 32);
    constexpr uint32_t slot_bytes = users <= 2 ? 16 : 32;  // per worker: (key, idx) per user
    const auto x = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), 2048);
    const auto base = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 2048);
    const auto fb = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3), 2048);
    const uint32_t first_col = get_arg_val<uint32_t>(3);
    const uint32_t worker = get_arg_val<uint32_t>(4);

    // users = 1: latent row 0, base row brow moved to row 0, fold_b row 0.  users > 1 (in place, no per-tile copies;
    // the batched block interleaves its rows: block row r of user u = tile row r * users + u): user u lives in row
    // r_u = brow * users + u of every tile; the users' base and fold_b rows (fold_b replicated in all 32 rows) are
    // one contiguous run of row pairs per face half, read where they are; latent row u is moved to row r_u once.
    constexpr bool inplace = users > 1;
    constexpr uint32_t r_first = brow * users, r_last = r_first + users - 1;
    // per face half h (rows 16h .. 16h+15): the 64 B-aligned row-pair run covering r_first .. r_last (bytes 0: none)
    constexpr uint32_t lo0 = r_first, hi0 = r_last < 15 ? r_last : 15;
    constexpr uint32_t lo1 = r_first > 16 ? r_first : 16, hi1 = r_last;
    constexpr uint32_t off0 = (lo0 & ~1u) * 32, bytes0 = lo0 <= hi0 ? ((hi0 | 1u) - (lo0 & ~1u) + 1) * 32 : 0;
    constexpr uint32_t off1 = 1024 + ((lo1 & 15u) & ~1u) * 32,
                       bytes1 = lo1 <= hi1 ? ((hi1 | 1u) - (lo1 & ~1u) + 1) * 32 : 0;
    constexpr uint32_t in0_bytes = (users + 1) / 2 * 64;  // rows 0..users-1 (64 B row pairs) of faces 0 / 1
    cb_reserve_back(cb_in0, kt);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t t = 0; t < kt; ++t) {
        const uint64_t src = x.get_noc_addr(t);
        noc_async_read(src, in0 + t * 2048, in0_bytes);
        noc_async_read(src + 512, in0 + t * 2048 + 512, in0_bytes);
    }
    noc_async_read_barrier();
    if constexpr (inplace) {
        for (uint32_t t = 0; t < kt; ++t) {
            for (uint32_t face = 0; face < 2; ++face) {
                for (uint32_t u = users; u-- > 0;) {  // r_u >= users for brow >= 1: never a source row
                    const uint32_t r = r_first + u;
                    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        in0 + t * 2048 + (r >> 4) * 1024 + face * 512 + (r & 15u) * 32);
                    volatile tt_l1_ptr uint32_t* sp =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0 + t * 2048 + face * 512 + u * 32);
                    for (uint32_t w = 0; w < 8; ++w) {
                        d[w] = sp[w];
                    }
                }
            }
        }
    }
    cb_push_back(cb_in0, kt);

    const uint32_t stage = get_write_ptr(cb_stage);
    constexpr uint32_t pair = (brow & ~1u) * 32;  // users = 1: byte offset of the 64 B row pair holding brow in a face
    constexpr uint32_t pick = (brow & 1u) * 32;
    // base / fold_b tiles of subblock s: issue(s) reserves + starts the reads, finish(s) waits + pushes.  users = 1:
    // back to back.  users > 1: the reads of s + 2 are in flight while s is scanned (cb_base / cb_fb hold 3 subblocks).
    auto issue = [&](uint32_t s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        const uint32_t col0 = first_col + s * wmax;
        cb_reserve_back(cb_base, wmax);
        cb_reserve_back(cb_fb, wmax);
        const uint32_t bt = get_write_ptr(cb_base);
        const uint32_t ft = get_write_ptr(cb_fb);
        for (uint32_t j = 0; j < wid; ++j) {
            const uint64_t b = base.get_noc_addr(col0 + j);
            const uint64_t f = fb.get_noc_addr(col0 + j);
            for (uint32_t face = 0; face < 2; ++face) {
                if constexpr (inplace) {
                    if constexpr (bytes0 > 0) {  // rows r_first .. r_last in the upper face half
                        noc_async_read(b + off0 + face * 512, bt + j * 2048 + off0 + face * 512, bytes0);
                        noc_async_read(f + off0 + face * 512, ft + j * 2048 + off0 + face * 512, bytes0);
                    }
                    if constexpr (bytes1 > 0) {  // ... in the lower face half
                        noc_async_read(b + off1 + face * 512, bt + j * 2048 + off1 + face * 512, bytes1);
                        noc_async_read(f + off1 + face * 512, ft + j * 2048 + off1 + face * 512, bytes1);
                    }
                } else {
                    noc_async_read(b + face * 512 + pair, stage + (2 * j + face) * 64, 64);
                    noc_async_read(f + face * 512, ft + j * 2048 + face * 512, 64);
                }
            }
        }
    };
    auto finish = [&](uint32_t s) {
        noc_async_read_barrier();
        if constexpr (!inplace) {
            const uint32_t wid = s == nsub - 1 ? rem : wmax;
            const uint32_t bt = get_write_ptr(cb_base);
            for (uint32_t j = 0; j < wid; ++j) {
                for (uint32_t face = 0; face < 2; ++face) {
                    volatile tt_l1_ptr uint32_t* d =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bt + j * 2048 + face * 512);
                    volatile tt_l1_ptr uint32_t* sp =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + (2 * j + face) * 64 + pick);
                    for (uint32_t w = 0; w < 8; ++w) {
                        d[w] = sp[w];
                    }
                }
            }
        }
        cb_push_back(cb_base, wmax);
        cb_push_back(cb_fb, wmax);
    };

    // Scan: a value v wins iff key(bf16_rne(v)) > best key (first strict maximum).  RNE rounding is monotone, so that
    // is mono(v) > thr with thr = the largest raw key that rounds to <= the best (recomputed only on a new best).
    // Fast path (best >= +0, the usual case after the first values): only a positive v can win, mono(v) > thr <=>
    // (int32) v > (int32) (thr ^ sign), checked branch-free for a whole 16-value face row; a hit rescans that row.
    uint32_t best_key[users], best_idx[users], thr[users];
    for (uint32_t u = 0; u < users; ++u) {
        best_key[u] = 0;
        best_idx[u] = first_col * 32;
        thr[u] = 0;
    }
    auto take = [&](uint32_t u, uint32_t v, uint32_t idx) -> uint32_t {
        uint32_t r = (v + 0x7FFFu + ((v >> 16) & 1u)) & 0xFFFF0000u;  // RNE to BF16
        r = r == 0x80000000u ? 0u : r;
        best_key[u] = mono(r);
        best_idx[u] = idx;
        const uint32_t lsb = (r >> 16) & 1u;  // largest raw value rounding to <= r (ties to even)
        return (int32_t)r >= 0 ? mono(r | (lsb ? 0x7FFFu : 0x8000u)) : mono(r - (lsb ? 0x7FFFu : 0x8000u));
    };
    issue(0);
    finish(0);
    if (inplace && nsub > 1) {
        issue(1);
    }
    uint32_t col = first_col;
    for (uint32_t s = 0; s < nsub; ++s) {
        if (s + 1 < nsub) {
            if constexpr (inplace) {
                finish(s + 1);  // read during the scan of s - 1
                if (s + 2 < nsub) {
                    issue(s + 2);
                }
            } else {
                issue(s + 1);
                finish(s + 1);
            }
        }
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        cb_wait_front(cb_out, wmax);
        asm volatile("" ::: "memory");  // the packed tiles are read through plain loads below
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < wid; ++j) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t idx0 = (col + j) * 32 + face * 16;
                const uint32_t n = idx0 >= valid ? 0 : (valid - idx0 < 16 ? valid - idx0 : 16);
                for (uint32_t u = 0; u < users; ++u) {  // user u's row of the FP32 result (64 B face rows)
                    const uint32_t r = inplace ? r_first + u : 0;
                    const uint32_t* q = reinterpret_cast<const uint32_t*>(
                        src + j * 4096 + (r >> 4) * 2048 + face * 1024 + (r & 15u) * 64);
                    uint32_t t = thr[u];
                    if (n == 16 && t >= 0x80000000u) {
                        const int32_t ts = (int32_t)(t ^ 0x80000000u);
                        const int32_t* qs = reinterpret_cast<const int32_t*>(q);
                        uint32_t hit = 0;
#pragma GCC unroll 16
                        for (uint32_t k = 0; k < 16; ++k) {
                            hit |= (uint32_t)(qs[k] > ts);
                        }
                        if (!hit) {
                            continue;
                        }
                    }
                    for (uint32_t c = 0; c < n; ++c) {
                        const uint32_t v = q[c];
                        if (mono(v) > t) {
                            t = take(u, v, idx0 + c);
                        }
                    }
                    thr[u] = t;
                }
            }
        }
        cb_pop_front(cb_out, wmax);
        col += wid;
    }
    // post (key, idx) per user to the gather core; worker 0 reduces
    const uint32_t slots = get_write_ptr(cb_slots);
    volatile tt_l1_ptr uint32_t* mine = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    for (uint32_t u = 0; u < users; ++u) {
        mine[2 * u] = best_key[u];
        mine[2 * u + 1] = best_idx[u];
    }
    const uint32_t sem = get_semaphore(0);
    noc_async_write(stage, get_noc_addr(gather_x, gather_y, slots + worker * slot_bytes), slot_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(gather_x, gather_y, sem), 1);
    noc_async_atomic_barrier();
    if (worker == 0) {
        volatile tt_l1_ptr uint32_t* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
        noc_semaphore_wait(count, workers);
        noc_semaphore_set(count, 0);
        volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots);
        for (uint32_t u = 0; u < users; ++u) {
            uint32_t key = 0, idx = 0;
            for (uint32_t w = 0; w < workers; ++w) {
                if (sl[w * (slot_bytes / 4) + 2 * u] > key) {
                    key = sl[w * (slot_bytes / 4) + 2 * u];
                    idx = sl[w * (slot_bytes / 4) + 2 * u + 1];
                }
            }
            mine[u] = idx;
        }
        const auto out = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4), 128);
        noc_async_write(stage, out.get_noc_addr(0), 4 * users);
        // the draft log: words 0..users-1 of page brow - 1 (draft brow of the block; one read returns every draft)
        if constexpr (brow > 0) {
            const auto log = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(5), 128);
            noc_async_write(stage, log.get_noc_addr(brow - 1), 4 * users);
        }
        noc_async_write_barrier();
    }
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    cb_wait_front(cb_in0, kt);
    for (uint32_t s = 0; s < nsub; ++s) {
        const uint32_t wid = s == nsub - 1 ? rem : wmax;
        cb_wait_front(cb_base, wmax);
        cb_wait_front(cb_fb, wmax);
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(cb_in1, cb_base);
        for (uint32_t j = 0; j < wid; ++j) {
            copy_tile(cb_base, j, j);
        }
        copy_tile_to_dst_init_short_with_dt(cb_base, cb_fb);
        for (uint32_t j = 0; j < wid; ++j) {
            copy_tile(cb_fb, j, wid + j);
        }
        add_binary_tile_init();
        for (uint32_t j = 0; j < wid; ++j) {
            add_binary_tile(j, wid + j, j);
        }
        cb_pop_front(cb_base, wmax);
        cb_pop_front(cb_fb, wmax);
        reconfig_data_format_srca(cb_fb, cb_in1);
        matmul_block_init(cb_in0, cb_in1, 0, wid, 1, kt);
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
