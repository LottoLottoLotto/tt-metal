// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling-2-Mini prefill routing, one program: x rows (row-major BF16 [T, H]) -> exact FP32 router (moe_router.hpp:
// SFPU FP32 multiply-adds, sigmoid, + bias) -> top-8 + weights per token (moe_route.hpp: exact FP32 order, ties to
// the lower id; IEEE soft float weights) -> one 32 B record per token (ids, bf16(w_k * f2[id_k])) -> expert groups.
// Work items: blocks of `rows` consecutive tokens over the whole K, dealt round-robin over the cores; per item NCRISC
// streams the x rows and all router weight tiles, TRISC packs [sel tile | scores tile] (row r at 512 r B of each),
// BRISC selects and writes the records (and, with `dump`, the FP32 sel / scores rows to a [T, 2E] tensor).
// Table (prefix + pack metadata, replaces the stock single-core prefix): after its items every core counts itself on
// the `n_table` table cores (semaphore 0); table core i then reads every record, counts every expert (groups =
// ceil(count / 32), ascending expert id) and fills the 32-slot group pages of experts [i E / n_table, ...): slot =
// [token | k << 16, bf16 weight] in token order, pads [0xffffffff, 0]; table core 0 writes the meta page [counts E,
// group starts E + 1].
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#if defined(TRISC_MATH)
#include "models/demos/blackhole/inkling2/tt/kernels/moe/moe_router.hpp"
#endif
#else
#include "api/dataflow/dataflow_api.h"
#include "models/demos/blackhole/inkling2/tt/kernels/moe/moe_route.hpp"
#endif

constexpr uint32_t cb_x = 0, cb_w = 1, cb_scratch = 2, cb_out = 4, cb_bias = 7, cb_recs = 8, cb_meta = 9,
                   cb_groups = 10;
constexpr uint32_t META_BYTES = 2048;  // meta page: counts [E], group starts [E + 1] (uint32)

void kernel_main() {
    constexpr uint32_t E = get_named_compile_time_arg_val("experts");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t H = get_named_compile_time_arg_val("hidden");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");  // = decode router_tiles
    constexpr uint32_t w_tiles = H * E / 1024;
    constexpr uint32_t row_bytes = H * 2;
    constexpr uint32_t items = tokens / rows;
    static_assert(E == 128 && tokens % rows == 0 && rows <= 8 && w_tiles % block_tiles == 0);
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t workers = get_arg_val<uint32_t>(1);

#if defined(COMPILE_FOR_NCRISC)
    // common args: -, -, x rows, router weight tiles, multicast rectangle (NOC x0, y0, x1, y1), the senders' NOC x, y.
    // The router weights stay resident in cb_w (all w_tiles): sender s (worker s < n_send) reads tiles
    // [s w_tiles / n_send, ...) from DRAM and multicasts them to every core once all cores have started (semaphore 2
    // = ready count on the senders); semaphore 1 counts the senders whose tiles have landed everywhere.
    constexpr uint32_t n_send = get_named_compile_time_arg_val("n_send");
    constexpr uint32_t per_send = w_tiles / n_send;
    constexpr uint32_t mc_tiles = 12;
    static_assert(w_tiles % n_send == 0 && per_send % mc_tiles == 0);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(2), row_bytes);
    const auto wr = TensorAccessor(w_args, get_common_arg_val<uint32_t>(3), 2048);
    const uint32_t mx0 = get_common_arg_val<uint32_t>(4), my0 = get_common_arg_val<uint32_t>(5);
    const uint32_t mx1 = get_common_arg_val<uint32_t>(6), my1 = get_common_arg_val<uint32_t>(7);
    const uint32_t sem_w = get_semaphore(1), sem_ready = get_semaphore(2);
    cb_reserve_back(cb_w, w_tiles);
    const uint32_t wbase = get_write_ptr(cb_w);
    for (uint32_t s = 0; s < n_send; ++s) {
        noc_semaphore_inc(get_noc_addr(get_common_arg_val<uint32_t>(8 + 2 * s), get_common_arg_val<uint32_t>(9 + 2 * s),
                                       sem_ready), 1);
    }
    if (worker < n_send) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready), workers);
        for (uint32_t b = worker * per_send; b < (worker + 1) * per_send; b += mc_tiles) {
            const uint32_t dst = wbase + b * 2048;
            for (uint32_t i = 0; i < mc_tiles; ++i) {
                noc_async_read(wr.get_noc_addr(b + i), dst + i * 2048, 2048);
            }
            noc_async_read_barrier();
            noc_async_write_multicast_loopback_src(dst, get_noc_multicast_addr(mx0, my0, mx1, my1, dst),
                                                   mc_tiles * 2048, workers);
        }
        noc_async_write_barrier();
        noc_semaphore_inc_multicast(get_noc_multicast_addr(mx0, my0, mx1, my1, sem_w), 1, workers - 1);
        noc_semaphore_inc(get_noc_addr(sem_w), 1);
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_w), n_send);
    cb_push_back(cb_w, w_tiles);
    for (uint32_t item = worker; item < items; item += workers) {
        cb_reserve_back(cb_x, 1);
        const uint32_t xd = get_write_ptr(cb_x);
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_read(x.get_noc_addr(item * rows + r), xd + r * row_bytes, row_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, 1);
    }
    noc_async_atomic_barrier();
#elif defined(COMPILE_FOR_BRISC)
    // args: worker, workers, bias, factors, records, dump, map, meta; common args: table cores' NOC x, y
    constexpr uint32_t scale_x1000 = get_named_compile_time_arg_val("route_scale_x1000");
    constexpr bool dump_rows = get_named_compile_time_arg_val("dump") != 0;
    constexpr uint32_t n_table = get_named_compile_time_arg_val("n_table");
    constexpr uint32_t ept = E / n_table;  // experts per table core
    static_assert(top_k == 8 && E % n_table == 0 && tokens % 8 == 0 && tokens < 65536);
    constexpr auto b_args = TensorAccessorArgs<0>();
    constexpr auto f_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto bias = TensorAccessor(b_args, get_common_arg_val<uint32_t>(2), E * 4);
    const auto factors = TensorAccessor(f_args, get_common_arg_val<uint32_t>(3), 2 * E * 4);
    const auto records = TensorAccessor(r_args, get_common_arg_val<uint32_t>(4), 256);
    const auto dump = TensorAccessor(d_args, get_common_arg_val<uint32_t>(5), 2 * E * 4);
    const auto map = TensorAccessor(m_args, get_common_arg_val<uint32_t>(6), 256);
    const auto meta = TensorAccessor(t_args, get_common_arg_val<uint32_t>(7), META_BYTES);
    // scratch: factors [2E] words, then one 32 B record per row of the item.
    const uint32_t fac_addr = get_write_ptr(cb_scratch);
    const uint32_t rec_addr = fac_addr + 2 * E * 4;
    noc_async_read(factors.get_noc_addr(0), fac_addr, 2 * E * 4);
    // Bias tile: bias[E] FP32 replicated over the whole 4 KB tile (every 512 B), for the SFPU's slot-wise add.
    cb_reserve_back(cb_bias, 1);
    const uint32_t bias_addr = get_write_ptr(cb_bias);
    for (uint32_t c = 0; c < 4096 / (E * 4); ++c) {
        noc_async_read(bias.get_noc_addr(0), bias_addr + c * E * 4, E * 4);
    }
    noc_async_read_barrier();
    cb_push_back(cb_bias, 1);
    auto* fac = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fac_addr);
    const float scale = scale_x1000 / 1000.0f;
    for (uint32_t item = worker; item < items; item += workers) {
        cb_wait_front(cb_out, 2);
        DeviceZoneScopedN("RT-SEL");
        const uint32_t sel = get_read_ptr(cb_out), scores = sel + 4096;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t t = item * rows + r;
            uint32_t ids[top_k];
            float w[top_k];
            ink_moe::select_topk<E, top_k>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel + r * 512), ids);
            ink_moe::route_weights<top_k>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scores + r * 512), ids,
                                          scale, w);
            // record: words 0-1 = the 8 ids (bytes, selection order), words 2-5 = bf16(w_k * f2[id_k]) pairs
            auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rec_addr + r * 32);
            uint32_t bf[top_k];
            for (uint32_t k = 0; k < top_k; ++k) {
                const uint32_t b = ink_moe::as_bits(w[k] * ink_moe::as_float(fac[E + ids[k]]));
                bf[k] = (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;  // round to nearest even
            }
            rec[0] = ids[0] | ids[1] << 8 | ids[2] << 16 | ids[3] << 24;
            rec[1] = ids[4] | ids[5] << 8 | ids[6] << 16 | ids[7] << 24;
            for (uint32_t k = 0; k < top_k; k += 2) {
                rec[2 + k / 2] = bf[k] | bf[k + 1] << 16;
            }
            rec[6] = rec[7] = 0;
            if constexpr (dump_rows) {
                noc_async_write(sel + r * 512, dump.get_noc_addr(t), E * 4);
                noc_async_write(scores + r * 512, dump.get_noc_addr(t) + E * 4, E * 4);
            }
        }
        const uint32_t t0 = item * rows;
        noc_async_write(rec_addr, records.get_noc_addr(t0 / 8) + (t0 % 8) * 32, rows * 32);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 2);
    }
    // Every record of this core has landed: count it on every table core.
    const uint32_t sem = get_semaphore(0);
    for (uint32_t i = 0; i < n_table; ++i) {
        noc_semaphore_inc(get_noc_addr(get_common_arg_val<uint32_t>(8 + 2 * i), get_common_arg_val<uint32_t>(9 + 2 * i),
                                       sem), 1);
    }
    if (worker < n_table) {
        // Table core: experts [e0, e0 + ept).  Groups = 32-slot runs per expert, ascending expert id, the rows of an
        // expert in token order; slot entry = [token | k << 16, bf16 weight] (pad: [0xffffffff, 0]).  Counters and
        // cursors live in the RISC's local memory (stack), the records are read through a plain pointer.
        DeviceZoneScopedN("RT-TABLE");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), workers);
        const uint32_t rb = get_write_ptr(cb_recs);
        for (uint32_t p = 0; p < tokens / 8; ++p) {
            noc_async_read(records.get_noc_addr(p), rb + p * 256, 256);
        }
        uint16_t cnt[E];
        for (uint32_t e = 0; e < E; ++e) {
            cnt[e] = 0;
        }
        noc_async_read_barrier();
        const uint32_t* recs = reinterpret_cast<const uint32_t*>(rb);
        for (uint32_t t = 0; t < tokens; ++t) {
            const uint32_t a = recs[8 * t], b = recs[8 * t + 1];
            cnt[a & 0xff]++;
            cnt[(a >> 8) & 0xff]++;
            cnt[(a >> 16) & 0xff]++;
            cnt[a >> 24]++;
            cnt[b & 0xff]++;
            cnt[(b >> 8) & 0xff]++;
            cnt[(b >> 16) & 0xff]++;
            cnt[b >> 24]++;
        }
        // meta staging page: counts [E], group starts [E + 1]
        const uint32_t mb = get_write_ptr(cb_meta);
        auto* mcnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mb);
        auto* start = mcnt + E;
        uint32_t total = 0;
        for (uint32_t e = 0; e < E; ++e) {
            mcnt[e] = cnt[e];
            start[e] = total;
            total += (cnt[e] + 31u) / 32u;
        }
        start[E] = total;
        const uint32_t e0 = worker * ept, e1 = e0 + ept;
        const uint32_t g0 = start[e0];
        uint32_t cur[ept];
        for (uint32_t e = e0; e < e1; ++e) {
            cur[e - e0] = (start[e] - g0) * 32;
        }
        uint32_t* slots = reinterpret_cast<uint32_t*>(get_write_ptr(cb_groups));
        for (uint32_t t = 0; t < tokens; ++t) {
            const uint32_t ab[2] = {recs[8 * t], recs[8 * t + 1]};
#pragma GCC unroll 8
            for (uint32_t k = 0; k < top_k; ++k) {
                const uint32_t i = ((ab[k / 4] >> (8 * (k % 4))) & 0xff) - e0;
                if (i < ept) {
                    const uint32_t s = cur[i]++;
                    slots[2 * s] = t | k << 16;
                    slots[2 * s + 1] = (recs[8 * t + 2 + k / 2] >> (16 * (k % 2))) & 0xffffu;
                }
            }
        }
        for (uint32_t e = e0; e < e1; ++e) {
            for (uint32_t s = cur[e - e0]; s < (start[e + 1] - g0) * 32; ++s) {
                slots[2 * s] = 0xffffffffu;
                slots[2 * s + 1] = 0;
            }
        }
        const uint32_t sb = get_write_ptr(cb_groups);
        for (uint32_t g = g0; g < start[e1]; ++g) {
            noc_async_write(sb + (g - g0) * 256, map.get_noc_addr(g), 256);
        }
        if (worker == 0) {  // meta page: counts [E], starts [E + 1] (groups; starts[E] = total)
            noc_async_write(mb, meta.get_noc_addr(0), META_BYTES);
        }
        noc_async_write_barrier();
    }
    noc_async_atomic_barrier();
#elif defined(COMPILE_FOR_TRISC)
    static_assert(DST_ACCUM_MODE, "FP32 destination accumulation required");
    // DEST: tile 0 logits / scores, tile 1 weight tile / sel, tile 2 bias, tile 3 k-block partial (summation order
    // of the decode router: moe_router.hpp).
    unary_op_init_common(cb_w, cb_out);
    copy_tile_to_dst_init_short(cb_w);
    MATH(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false)));
    pack_reconfig_data_format(cb_out);
    cb_wait_front(cb_bias, 1);
    cb_wait_front(cb_w, w_tiles);  // resident router weights
    for (uint32_t item = worker; item < items; item += workers) {
        cb_wait_front(cb_x, 1);
        DeviceZoneScopedN("RT-MAC");
        const uint32_t x_addr = get_tile_address(cb_x, 0);
        tile_regs_acquire();
        MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ink_router_zero<rows, 0>, 0, VectorMode::None)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ink_router_zero<rows, 96>, 0, VectorMode::None)));
        for (uint32_t b = 0; b < w_tiles; b += block_tiles) {
            for (uint32_t i = b; i < b + block_tiles; ++i) {
                MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));  // SFPU done with DEST tile 1
                copy_tile(cb_w, i, 1);
                MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ink_router_mac<rows, 96>, 0,
                                                           VectorMode::None, x_addr + i * 16, row_bytes)));
            }
            MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ink_router_flush<rows>, 0, VectorMode::None)));
        }
        copy_tile_to_dst_init_short_with_dt(cb_w, cb_bias);
        MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));
        copy_tile(cb_bias, 0, 2);
        MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ink_router_finish_rows<rows>, 0,
                                                   VectorMode::None)));
        copy_tile_to_dst_init_short_with_dt(cb_bias, cb_w);
        tile_regs_commit();
        cb_pop_front(cb_x, 1);
        cb_reserve_back(cb_out, 2);
        tile_regs_wait();
        pack_tile(1, cb_out);  // sel
        pack_tile(0, cb_out);  // scores
        tile_regs_release();
        cb_push_back(cb_out, 2);
    }
    cb_pop_front(cb_bias, 1);
    cb_pop_front(cb_w, w_tiles);
#endif
}
