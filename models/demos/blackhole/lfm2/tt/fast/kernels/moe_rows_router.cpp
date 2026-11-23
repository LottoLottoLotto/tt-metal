// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 multi-row decode MoE (moe_rows.py), router role: 2 cores inside the gate/up program.  Per core one 32-expert
// column of the FP32 router logits for up to 16 token rows, then the per-row routing and its hand-off to every
// gate/up worker.
//
// Logits: moe_decode2_router.cpp's matmul unchanged (bitwise StreamDecodeMatmul's router preset = MoE.route's
// ttnn.linear; matmul rows are independent), with in0 = the upper half of each x tile ([16, 32]: rows 0-15, one
// 1024 B read) instead of rows 0-1.
// Routing: BRISC sends this core's logits tile ([16, 32] FP32, 2 KB) to the other router core (semaphore `cross`),
// so both cores hold all 64 logits of every row.  The 4 router RISCs (q = 2 * col + risc) take contiguous row ranges
// of `rows_per_risc` rows: per row moe2::score_experts over the 64 experts + moe2::finish_route (bitwise lfm_route,
// flags 3), one 32 B record (ids[4], BF16 weights[4] as words) per row.  The RISC writes its records into every
// gate/up worker's landing buffer at row offset, waits for the acks, then increments each worker's `route_sem`;
// the workers wait for `senders` increments.
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "models/demos/blackhole/lfm2/tt/fast/kernels/moe_decode2_route.hpp"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace {
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");        // K tiles
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");        // K tiles per streamed weight block
constexpr uint32_t ks = get_named_compile_time_arg_val("ks");        // stock in0_block_w
constexpr uint32_t group = get_named_compile_time_arg_val("group");  // chunks per DEST acquire
constexpr uint32_t xg = 8;                                           // x tiles per push
constexpr uint32_t nch = kt / ks;
constexpr uint32_t half_kt = kt / 2;
static_assert(kt % kb == 0 && kb % ks == 0 && kt % ks == 0 && nch >= 2 && group >= 1 && group <= 4);
static_assert(half_kt % xg == 0 && half_kt / xg <= 4 && half_kt % ks == 0);
// cb_in0: x tiles [0, kt/2) (BRISC), cb_in0b: [kt/2, kt) (NCRISC)
constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_in0b = 2, cb_scratch = 3, cb_land = 7, cb_out = 16, cb_part = 24;
}  // namespace

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace {
constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
constexpr uint32_t e_lo = get_named_compile_time_arg_val("score_e_lo");
constexpr uint32_t e_hi = get_named_compile_time_arg_val("score_e_hi");
constexpr uint32_t table_bytes = 2 * (e_hi - e_lo + 1) * 128 * 2;
constexpr uint32_t out_face_bytes = get_named_compile_time_arg_val("out_face_bytes");
constexpr uint32_t in0_tile_bytes = get_named_compile_time_arg_val("in0_tile_bytes");
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");                    // token rows (<= 16)
constexpr uint32_t rows_per_risc = get_named_compile_time_arg_val("rows_per_risc");  // contiguous rows per RISC
constexpr uint32_t route_sem = get_named_compile_time_arg_val("route_sem");          // workers: + 1 per sender
constexpr uint32_t cross_sem = get_named_compile_time_arg_val("cross_sem");          // other router's logits in
constexpr uint32_t rec_bytes = 32;                                                   // per row: ids, weights
constexpr uint32_t logits_bytes = 2 * out_face_bytes;                                // [16, 32] FP32 tile
static_assert(experts == 64 && top_k <= 4 && rows <= 16 && in0_tile_bytes == 1024 && out_face_bytes == 1024);
// scratch: per RISC r: [table | bias] at r * area; the other column's logits tile; per RISC its row records
constexpr uint32_t area = (table_bytes + experts * 4 + 63) / 64 * 64;
constexpr uint32_t other_off = 2 * area;
constexpr uint32_t recs_off = other_off + logits_bytes;
constexpr uint32_t pos_off = recs_off + 2 * rows_per_risc * rec_bytes;  // per RISC 64 B: cur_pos (has_pos)
constexpr bool has_pos = get_named_compile_time_arg_val("has_pos") == 1;

// x tiles [first, first + half_kt) -> cb (rows 0-15: the first 1024 B of each tile), xg tiles per transaction id
// (trid_base + g), pushed in order.
template <uint32_t cb>
void gather_x_issue(uint32_t first, uint32_t trid_base) {
    constexpr auto x_args = TensorAccessorArgs<get_named_compile_time_arg_val("x_cta")>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(1), 2048);
    cb_reserve_back(cb, half_kt);
    const uint32_t dst = get_write_ptr(cb);
    for (uint32_t g = 0; g < half_kt / xg; ++g) {
        noc_async_read_set_trid(trid_base + g);
        for (uint32_t t = 0; t < xg; ++t) {
            noc_async_read(get_noc_addr(first + g * xg + t, x), dst + (g * xg + t) * in0_tile_bytes, in0_tile_bytes);
        }
    }
}

template <uint32_t cb>
void gather_x_push(uint32_t trid_base, bool latest) {
    for (uint32_t g = 0; g < half_kt / xg; ++g) {
        if (latest && g + 1 == half_kt / xg) {
            noc_async_read_barrier();
        } else {
            noc_async_read_barrier_with_trid(trid_base + g);
        }
        cb_push_back(cb, xg);
    }
}

template <uint32_t risc>
void read_table(uint32_t table_addr, uint32_t bias_addr) {
    constexpr auto table_args = TensorAccessorArgs<get_named_compile_time_arg_val("table_cta")>();
    constexpr auto bias_args = TensorAccessorArgs<get_named_compile_time_arg_val("bias_cta")>();
    const auto table = TensorAccessor(table_args, table_addr, table_bytes);
    const auto bias = TensorAccessor(bias_args, bias_addr, experts * 4);
    const uint32_t dst = get_write_ptr(cb_scratch) + risc * area;
    noc_async_read(get_noc_addr(0, table), dst, table_bytes);
    noc_async_read(get_noc_addr(0, bias), dst + table_bytes, experts * 4);
    noc_async_read_barrier();
}

// Router RISC `risc` of column `col`.  args from first_arg: the other router core's NoC x, y, the gate/up worker
// count and their NoC (x, y) pairs.
template <uint32_t risc>
void route_rows(uint32_t col, uint32_t first_arg) {
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint32_t table = scratch + risc * area;
    const uint32_t bias = table + table_bytes;
    const uint32_t other = scratch + other_off;
    const uint32_t recs = scratch + recs_off + risc * rows_per_risc * rec_bytes;
    const uint32_t other_x = get_arg_val<uint32_t>(first_arg), other_y = get_arg_val<uint32_t>(first_arg + 1);
    cb_wait_front(cb_out, 1);
    const uint32_t own = get_read_ptr(cb_out);
    if constexpr (risc == 0) {
        DeviceZoneScopedN("R-CROSS");
        noc_async_write(own, get_noc_addr(other_x, other_y, other), logits_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(other_x, other_y, get_semaphore(cross_sem)), 1);
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(cross_sem)), 1);
    invalidate_l1_cache();
    const uint32_t q = 2 * col + risc;
    const uint32_t r0 = q * rows_per_risc;
    const uint32_t r1 = r0 + rows_per_risc < rows ? r0 + rows_per_risc : rows;
    if (r0 >= r1) {
        return;
    }
    // rows with cur_pos < 0 (inactive slots) select no expert: id 0xff (ignored by the workers), weight 0
    const volatile tt_l1_ptr int32_t* pos = nullptr;
    if constexpr (has_pos) {
        constexpr auto pos_args = TensorAccessorArgs<get_named_compile_time_arg_val("pos_cta")>();
        const auto pos_t = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(5), rows * 4);
        const uint32_t dst = get_write_ptr(cb_scratch) + pos_off + risc * 64;
        noc_async_read(get_noc_addr(0, pos_t), dst, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(dst);
    }
    {
        DeviceZoneScopedN("R-ROUTE");
        for (uint32_t r = r0; r < r1; ++r) {
            auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recs + (r - r0) * rec_bytes);
            if (has_pos && pos[r] < 0) {
                for (uint32_t k = 0; k < 4; ++k) {
                    rec[k] = 0xffu;
                    rec[4 + k] = 0u;
                }
                continue;
            }
            moe2::Cands<top_k> best;
            for (uint32_t c = 0; c < 2; ++c) {
                const uint32_t tile = c == col ? own : other;
                for (uint32_t f = 0; f < 2; ++f) {
                    moe2::score_experts<top_k, 16, e_lo, e_hi>(
                        best, reinterpret_cast<const uint32_t*>(tile + f * out_face_bytes + r * 64), c * 32 + f * 16,
                        reinterpret_cast<const int32_t*>(bias), reinterpret_cast<const uint16_t*>(table));
                }
            }
            uint32_t ids[top_k];
            uint16_t wts[top_k];
            moe2::finish_route<top_k>(best, ids, wts);
            for (uint32_t k = 0; k < 4; ++k) {
                rec[k] = k < top_k ? ids[k] : 0u;
                rec[4 + k] = k < top_k ? static_cast<uint32_t>(wts[k]) : 0u;
            }
        }
    }
    DeviceZoneScopedN("R-SEND");
    const uint32_t n = get_arg_val<uint32_t>(first_arg + 2);
    const uint32_t land = get_write_ptr(cb_land) + r0 * rec_bytes;
    const uint32_t bytes = (r1 - r0) * rec_bytes;
    for (uint32_t w = 0; w < n; ++w) {
        const uint32_t a = first_arg + 3 + 2 * w;
        noc_async_write(recs, get_noc_addr(get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1), land), bytes);
    }
    noc_async_write_barrier();  // acked: the records are in every worker's L1 before its flag moves
    const uint32_t sem = get_semaphore(route_sem);
    for (uint32_t w = 0; w < n; ++w) {
        const uint32_t a = first_arg + 3 + 2 * w;
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1), sem), 1);
    }
    noc_async_atomic_barrier();
}
}  // namespace
#endif

void kernel_main() {
    // common args: weight, x, logits, table, bias addresses; per core: col (= weight bank), then for BRISC and then
    // for NCRISC: the other router core's NoC x, y, the worker count and their NoC (x, y)
    const uint32_t col = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t tile_bytes = 2048;  // BF16 weight tiles
    constexpr uint32_t blocks = kt / kb;
    constexpr uint32_t block_bytes = kb * tile_bytes;
    constexpr uint32_t w_trids = 11;  // weight blocks use transaction ids 1..11, x groups 12..15
    const uint64_t src = get_noc_addr_from_bank_id<true>(col, get_common_arg_val<uint32_t>(0));
    {
        DeviceZoneScopedN("R-WREAD");
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
        cb_reserve_back(cb_in1, kt);
        const uint32_t dst = get_write_ptr(cb_in1);
        for (uint32_t b = 0; b < blocks; ++b) {
            noc_async_read_set_trid(b % w_trids + 1);
            noc_async_read(src + b * block_bytes, dst + b * block_bytes, block_bytes);
        }
        gather_x_issue<cb_in0b>(half_kt, w_trids + 1);
        for (uint32_t b = 0; b < blocks; ++b) {
            noc_async_read_barrier_with_trid(b % w_trids + 1);
            cb_push_back(cb_in1, kb);
        }
        gather_x_push<cb_in0b>(w_trids + 1, true);
        noc_async_read_set_trid(0);
    }
    read_table<1>(get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4));
    route_rows<1>(col, 4 + 2 * get_arg_val<uint32_t>(3));
#elif defined(COMPILE_FOR_BRISC)
    constexpr auto logits_args = TensorAccessorArgs<get_named_compile_time_arg_val("logits_cta")>();
    const auto logits = TensorAccessor(logits_args, get_common_arg_val<uint32_t>(2), 4096);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    gather_x_issue<cb_in0>(0, 1);
    gather_x_push<cb_in0>(1, true);
    noc_async_read_set_trid(0);
    read_table<0>(get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4));
    route_rows<0>(col, 1);
    // logits rows 0-15 of this column -> FP32 tile `col` of the logits tensor (faces 0 and 1; tests)
    noc_async_write(get_read_ptr(cb_out), get_noc_addr(col, logits), logits_bytes);
    noc_async_write_barrier();
#else
    DeviceZoneScopedN("R-MM");
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_block_init(cb_in0, cb_in1, 0, 1, 1, kt);
    // Chunks 0..nch-2: partial tile in place (reserved once, pushed once).
    cb_reserve_back(cb_part, 1);
    PACK((pack_reconfig_data_format(cb_part)));
    uint32_t waited = 0, x_waited = 0;
    auto mm = [&](uint32_t k, uint32_t dst) {
        if (k >= waited) {
            waited += kb;
            cb_wait_front(cb_in1, waited);
        }
        if (k >= x_waited) {
            x_waited += xg;
            if (x_waited <= half_kt) {
                cb_wait_front(cb_in0, x_waited);
            } else {
                cb_wait_front(cb_in0b, x_waited - half_kt);
            }
        }
        if (k < half_kt) {
            matmul_block(cb_in0, cb_in1, k, k, dst, 0, 1, 1, kt);
        } else {
            matmul_block(cb_in0b, cb_in1, k - half_kt, k, dst, 0, 1, 1, kt);
        }
    };
    for (uint32_t c = 0; c + 1 < nch; c += group) {
        const uint32_t n = c + group < nch - 1 ? group : nch - 1 - c;
        tile_regs_acquire();
        for (uint32_t g = 0; g < n; ++g) {
            for (uint32_t i = 0; i < ks; ++i) {
                mm((c + g) * ks + i, g);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t g = 0; g < n; ++g) {
            if (c + g == 1) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
            pack_tile<true>(g, cb_part, 0);
        }
        tile_regs_release();
    }
    cb_push_back(cb_part, 1);
    PACK((llk_pack_reconfig_l1_acc(0)));
    // Last chunk: DEST <- partial (exact FP32 reload), then its K tiles accumulate on top.
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_in1, cb_part);
    cb_wait_front(cb_part, 1);
    copy_tile(cb_part, 0, 0);
    cb_pop_front(cb_part, 1);
    reconfig_data_format_srca(cb_part, cb_in1);
    matmul_block_init(cb_in0, cb_in1, 0, 1, 1, kt);
    for (uint32_t i = 0; i < ks; ++i) {
        mm((nch - 1) * ks + i, 0);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    PACK((pack_reconfig_data_format(cb_out)));
    pack_tile(0, cb_out, 0);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in1, kt);
    cb_pop_front(cb_in0, half_kt);
    cb_pop_front(cb_in0b, half_kt);
#endif
}
