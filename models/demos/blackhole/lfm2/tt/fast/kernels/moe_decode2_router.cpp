// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 decode MoE v2, router role (2 cores inside the experts program): FP32 router logits of one 32-expert column,
// the column's top_k candidates, and the hand-off to every expert worker.
//
// Logits: bitwise StreamDecodeMatmul's router preset (ttnn.linear's in0_block_w = 2 order, HiFi4, FP32 DEST): every
// 2-tile K chunk sums in DEST from zero, chunks 0..nch-2 are packed with FP32 L1 accumulation onto one partial tile
// (chunk 0 without), and the last chunk starts from the partial reloaded into DEST (UnpackToDestFp32).  Unlike the
// stock loop, the partial stays in place (no CB round trip through the unpacker per chunk) and `group` chunks share
// one DEST acquire (the packer still adds them to L1 one by one, in chunk order).  in0 is a [16, 32] or [32, 32] tile
// per K tile (row 0 = the token; rows 1.. are don't-care: matmul rows are independent).
// NCRISC: the core's weight column (one contiguous run in bank `col`) into L1, one transaction id per block, and the
//         second half of the x tiles; then the score table + bias, scores of experts 16..31 of the column.
// BRISC:  the first half of the x tiles; the table + bias; after the logits tile: the logits row to `logits`
//         (tests / debug), scores of experts 0..15 of the column.
// x tiles: one 576 B read per tile (face 0 + rows 0-1 of face 1), progressive pushes.
// Both RISCs swap their lists through L1 (semaphores `xchg`), merge them into the column's top_k record and multicast
// it to slot `col` of the workers' landing buffers, rectangle by rectangle (BRISC the first rectangles, NCRISC the
// rest), each followed by a multicast set of the workers' semaphore `route_sem + col` (same NoC: ordered after it).
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
constexpr uint32_t route_sem = get_named_compile_time_arg_val("route_sem");  // workers: routing delivered
constexpr uint32_t xchg_sem = get_named_compile_time_arg_val("xchg_sem");    // + risc: RISC list posted
constexpr uint32_t cross_sem = get_named_compile_time_arg_val("cross_sem");  // other router's column list in
constexpr uint32_t rec_bytes = 64;
constexpr uint32_t x_read_bytes = 576;  // face 0 + rows 0-1 of face 1 (64 B multiple, DRAM-aligned)
static_assert(3 * top_k * 4 <= rec_bytes && top_k <= 4 && in0_tile_bytes >= 1024);
// scratch: per RISC r: [table | bias] at r * area; then 64 B records: RISC lists (2), column lists by column (2),
// payloads per RISC (2), the constant 1 for remote semaphore sets
constexpr uint32_t area = (table_bytes + experts * 4 + 63) / 64 * 64;

// x tiles [first, first + half_kt) -> cb, xg tiles per transaction id (trid_base + g), pushed in order.
template <uint32_t cb>
void gather_x_issue(uint32_t first, uint32_t trid_base) {
    constexpr auto x_args = TensorAccessorArgs<get_named_compile_time_arg_val("x_cta")>();
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(1), 2048);
    cb_reserve_back(cb, half_kt);
    const uint32_t dst = get_write_ptr(cb);
    for (uint32_t g = 0; g < half_kt / xg; ++g) {
        noc_async_read_set_trid(trid_base + g);
        for (uint32_t t = 0; t < xg; ++t) {
            noc_async_read(get_noc_addr(first + g * xg + t, x), dst + (g * xg + t) * in0_tile_bytes, x_read_bytes);
        }
    }
}

// Pushes the gathered x groups (the caller issued nothing after them except as noted: the last group of the latest
// issued reads waits on the full barrier, decode_matmul_stream.cpp).
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

// Router RISC `risc` (0 BRISC: experts 0..15 of the column, 1 NCRISC: 16..31).  args from first_arg: the other
// router core's NoC x, y, then this RISC's workers: count, NoC (x, y) pairs.
template <uint32_t risc>
void route_and_send(uint32_t col, uint32_t first_arg) {
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint32_t table = scratch + risc * area;
    const uint32_t bias = table + table_bytes;
    const uint32_t lists = scratch + 2 * area;       // 2 RISC lists
    const uint32_t columns = lists + 2 * rec_bytes;  // column lists, slot = column
    const uint32_t payload = columns + 2 * rec_bytes + risc * rec_bytes;
    const uint32_t one = columns + 4 * rec_bytes;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one) = 1;  // (both RISCs store the same value)
    cb_wait_front(cb_out, 1);
    invalidate_l1_cache();
    moe2::Cands<top_k> best;
    {
        DeviceZoneScopedN("MOE2-SCORE");
        moe2::score_experts<top_k, 16, e_lo, e_hi>(
            best, reinterpret_cast<const uint32_t*>(get_read_ptr(cb_out) + risc * out_face_bytes),
            col * 32 + risc * 16, reinterpret_cast<const int32_t*>(bias), reinterpret_cast<const uint16_t*>(table));
    }
    // the two RISCs' lists -> the column's list (both RISCs)
    best.store(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lists + risc * rec_bytes));
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(xchg_sem + risc)) = 1;
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(xchg_sem + 1 - risc)), 1);
    best.merge(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lists + (1 - risc) * rec_bytes));
    // column lists between the router cores (BRISC sends; data, flushed, then the flag: same source, destination
    // and VC, so the flag lands after the data)
    const uint32_t other_x = get_arg_val<uint32_t>(first_arg), other_y = get_arg_val<uint32_t>(first_arg + 1);
    if constexpr (risc == 0) {
        DeviceZoneScopedN("R-CROSS");
        best.store(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(columns + col * rec_bytes));
        noc_async_write(columns + col * rec_bytes, get_noc_addr(other_x, other_y, columns + col * rec_bytes),
                        rec_bytes);
        noc_async_writes_flushed();
        noc_semaphore_set_remote(one, get_noc_addr(other_x, other_y, get_semaphore(cross_sem)));
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(cross_sem)), 1);
    best.merge(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(columns + (1 - col) * rec_bytes));
    // final routing -> payload: ids (top_k words), BF16 weights (top_k words)
    uint32_t ids[top_k];
    uint16_t wts[top_k];
    {
        DeviceZoneScopedN("R-FINISH");
        moe2::finish_route<top_k>(best, ids, wts);
    }
    auto* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload);
    for (uint32_t k = 0; k < top_k; ++k) {
        out[k] = ids[k];
        out[top_k + k] = wts[k];
    }
    DeviceZoneScopedN("R-SEND");
    // every worker: payload; once the payloads left this core, the route flags
    const uint32_t n = get_arg_val<uint32_t>(first_arg + 2);
    const uint32_t land = get_write_ptr(cb_land);
    for (uint32_t w = 0; w < n; ++w) {
        const uint32_t a = first_arg + 3 + 2 * w;
        noc_async_write(payload, get_noc_addr(get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1), land),
                        rec_bytes);
    }
    noc_async_writes_flushed();
    const uint32_t sem = get_semaphore(route_sem);
    for (uint32_t w = 0; w < n; ++w) {
        const uint32_t a = first_arg + 3 + 2 * w;
        noc_semaphore_set_remote(one, get_noc_addr(get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1), sem));
    }
    noc_async_write_barrier();
}
}  // namespace
#endif

void kernel_main() {
    // common args: weight, x, logits, table, bias addresses; per core: col (= weight bank), then for BRISC and then
    // for NCRISC: the other router core's NoC x, y, the RISC's worker count and their NoC (x, y)
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
        // every weight block was issued before the x reads: per-trid barriers are safe (a reused id only delays)
        for (uint32_t b = 0; b < blocks; ++b) {
            noc_async_read_barrier_with_trid(b % w_trids + 1);
            cb_push_back(cb_in1, kb);
        }
        gather_x_push<cb_in0b>(w_trids + 1, true);
        noc_async_read_set_trid(0);
    }
    read_table<1>(get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4));
    route_and_send<1>(col, 4 + 2 * get_arg_val<uint32_t>(3));
#elif defined(COMPILE_FOR_BRISC)
    constexpr auto logits_args = TensorAccessorArgs<get_named_compile_time_arg_val("logits_cta")>();
    const auto logits = TensorAccessor(logits_args, get_common_arg_val<uint32_t>(2), 4096);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    gather_x_issue<cb_in0>(0, 1);
    gather_x_push<cb_in0>(1, true);
    noc_async_read_set_trid(0);
    read_table<0>(get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4));
    route_and_send<0>(col, 1);
    // logits row of this column -> FP32 tile `col` of the logits tensor (face 0 row 0, face 1 row 0; tests)
    const uint32_t res = get_read_ptr(cb_out);
    const uint64_t lg = get_noc_addr(col, logits);
    noc_async_write(res, lg, 64);
    noc_async_write(res + out_face_bytes, lg + 1024, 64);
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
