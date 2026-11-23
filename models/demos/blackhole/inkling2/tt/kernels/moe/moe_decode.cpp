// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling-2-Mini decode MoE for R = 1..16 token rows (R > 1: speculative verify or the rows of independent batched
// sequences), one program with three roles, weights streamed from the resident bank layout of moe_layout.py.  Every
// row is routed by the exact FP32 router (bitwise the prefill router); the UNION of the rows' selected experts is
// streamed once, in ascending expert id ("slots"; duplicate rows add no slot), and each row's activation is scaled by
// its own routing weight (0 for a slot the row did not select).  Rows are independent in every FPU / SFPU op and the
// down sum runs over the slots in id order, so row r of an R-row call is bitwise the R = 1 call on that row.  Rows
// live in faces 0 / 1 of every row tile (rows 0..15): x reads, activation / output tiles ([AR, 32], AR <= 16).
//
// Dispatch shape (MoE7): one kernel group per role (MOE_ROLE 0 gate/up, 1 down, 2 router: the binary holds only that
// role's code; a single all-role binary measured slower in the model), every CB and semaphore on ONE bounding
// rectangle, no per-core runtime args.  A core finds its index in the common args: byte (y * grid_x + x) of the core
// table (absolute logical coordinates) = 1 + its index in [gate/up cores | down cores | router cores].  gate/up
// index i = bank * gate_splits + part, down likewise, router index = group * routers + r (both router row groups in
// one kernel group).  Per-role compile-time constants are named args gate_* / down_*.  NOC choice without a kernel
// group: the "stream" side (weight stream + its routing share) runs on NCRISC (NOC0) and the "write" side on BRISC
// (NOC1), swapped at run time on the stream cores of the banks in `noc1_mask` (BRISC streams on NOC1, NCRISC writes
// on NOC0).  CB indices are per role where tile shapes / formats differ (a kernel's CB formats are compile-time per
// index); gate/up and down share the weight CB (each role's ring wraps where the CB does).
//
// ROUTER role: `routers` cores per row group, core r owns k in [r ks, (r + 1) ks) (ks = 8 router_tiles).  NCRISC
//   streams its router weight tiles (moe_router.hpp layout); BRISC reads rows QR0..QR0+QRN-1 of the covering x tiles
//   (face rows as stored) and the FP32 bias; TRISC accumulates the partial logits with SFPU FP32 multiply-adds and
//   packs them (row q at 512 q B).  BRISC all-gathers the partials (512 B per row to every router core of its group +
//   semaphore 0), then TRISC sums each row's partials in a fixed order, scores = sigmoid, sel = scores + bias -> 1 KB
//   [sel | scores] per row, which BRISC writes into the `rows` buffer (CB 6) of its share of the workers (worker w ->
//   router w % routers), followed by one flag write.  R > 1: group 0 routes rows [0, RN), group 1 rows [RN, R).
// Routing (every worker): the stream side selects the top-k of rows [0, RN), the write side of rows [RN, R) (exact
//   FP32 order, ties to the lower id) into an id table in L1 (R = 1: the stream side the lower, the write side the
//   upper expert half, the stream side merges); the stream side forms the union (ascending ids = slots) and hands the
//   write side the plan.  Gate/up stream side computes the renormalised weights w = s / (sum + 1e-20) * route_scale
//   (IEEE soft float) while its first unit streams, and the per-slot TRISC table [f13, w_0 f2, .., w_{R-1} f2] (0
//   where the row did not select).
// RTK (R > 8, batched rows; R <= 8 is unchanged): `router_groups` (3) groups of `routers` cores split the rows as
//   evenly as possible (group g: rows [QR0, QR0 + QRN)), and the top-k moves to the router: core r < QRN of a group
//   receives only row QR0 + r's partials (semaphore 0), TRISC finishes that row alone, BRISC dumps it (route row words
//   32..), selects its top-k, computes its weights (the same soft-float code as the workers', so the same bits) and
//   writes a 128 B record [ids (8), 0 (8), w (8), w f2 (8)] into the `rows` buffer of every worker, then increments
//   semaphore 2 there.  Workers wait for semaphore 2 == R, copy the ids, form the union; the gate/up TRISC table takes
//   w f2 from the records (no per-row work on the workers).  Every summation / selection / weight is the R = 1 one.
//
// GATE_UP role: `gate_splits` cores per DRAM bank.  Bank b's units are (slot u, local pair l < pairs(e_u, b)) in
//   slot-major order (moe_route.hpp bank_units), dealt round-robin to the bank's cores; with BAL (balance) the
//   floating pairs chosen by balance_floating are streamed by the next bank from its gate_up_dup run instead, so
//   every bank streams the same number of units (same data, same results).  A unit streams the pair's [K, {gate,
//   up}] block (2 kt contiguous tiles, in `blk_k`-row blocks), accumulates gate and up in FP32 DEST (matmul_block,
//   ct = 2, in1 tiles transposed when K_ALIGNED; in0 rows 0..R-1 = the tokens), and the pack thread applies g, u *=
//   f13[e]; act = 4 tanh(g/4) sigmoid(g) 25 tanh(u/25) * w_row on rows 0..R-1 (per-lane row weights), packs one BF16
//   tile; the write side copies rows 0..R-1 of faces 0 / 1 into the [AR, 32] activation tile (u, j) (j = start(e_u,
//   b) + l), then increments counter u on every down core.
// DOWN role: `down_splits` cores per bank, `down_cols` output tiles each.  The write side zeroes its counters
//   (cb_rows) and tells every gate/up core (semaphore 1) before any can count; the stream side prefetches the core's
//   [K, cols] block of every slot; the write side waits for counter u (I/32 units), reads the slot's [AR, 32]
//   activation tiles (one read per page) as its in0 tiles; TRISC accumulates every slot in FP32 DEST; the write side
//   writes rows 0..R-1 of the cols output tiles into zeroed staging tiles and stores them (rows R..31 of the output
//   are zero).
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "models/demos/blackhole/inkling2/tt/kernels/fast/read_vc.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#endif
#include "models/common/modules/moe/kernels/streaming/experts_matmul.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "models/demos/blackhole/inkling2/tt/kernels/moe/moe_route.hpp"
#endif

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_PACK)
#include "sfpu/ckernel_sfpu_converter.h"
namespace ckernel::sfpu {
// tanh with ~2e-6 relative accuracy from the accurate sigmoid; Taylor below |y| < 1/16 (no cancellation).
sfpi_inline sfpi::vFloat ink_tanh(sfpi::vFloat y) {
    sfpi::vFloat t = _sfpu_sigmoid_<true>(y + y) * 2.0f - 1.0f;
    const sfpi::vFloat y2 = y * y;
    v_if(sfpi::abs(y) < 0.0625f) { t = y * (1.0f - y2 * (0.333333343f - y2 * 0.133333340f)); }
    v_endif;
    return t;
}

// act = 4 tanh(g/4) * sigmoid(g) * 25 tanh(u/25) * w_row with g, u scaled by f first.  ITERS = SFPU iterations per
// face (2 = rows 0-3, 4 = rows 0-7, 8 = rows 0-15; VectorMode::R = faces 0 and 1).  DEST tiles dst_gate + 2 / + 3 are
// scratch; the steps keep few values live (the SFPU has 8 LREGs and no spill).  R = 1: w = the weight's FP32 bits.
// R > 1: w = L1 address of the slot's row weights (zero-padded to 4 / 8 / 16 rows); an SFPLOAD covers 4 rows x 8
// columns, lane 8i + c holding row 4 (d / 2) + i (vConstTileId = 2 lane), so each iteration first builds its per-lane
// weights in tile + 3.
template <int ITERS, uint32_t R>
inline void calculate_ink_swiglu(const uint dst_gate, const uint dst_up, const uint dst_out, const uint w,
                                 const uint f_bits) {
    constexpr uint tile_stride = 32;
    const uint dst_tmp = dst_gate + 2;
    const uint dst_w = dst_gate + 3;
#pragma GCC unroll 1
    for (int d = 0; d < ITERS; d++) {
        if constexpr (R > 1) {
            const uint32_t* wr = reinterpret_cast<const uint32_t*>(w) + 4 * (d / 2);
            const sfpi::vFloat w1 = Converter::as_float(wr[1]);
            const sfpi::vFloat w2 = Converter::as_float(wr[2]);
            const sfpi::vFloat w3 = Converter::as_float(wr[3]);
            sfpi::vFloat v = Converter::as_float(wr[0]);
            v_if(sfpi::vConstTileId >= 16) { v = w1; }
            v_endif;
            v_if(sfpi::vConstTileId >= 32) { v = w2; }
            v_endif;
            v_if(sfpi::vConstTileId >= 48) { v = w3; }
            v_endif;
            sfpi::dst_reg[dst_w * tile_stride] = v;
        }
        {
            const sfpi::vFloat u = sfpi::dst_reg[dst_up * tile_stride] * Converter::as_float(f_bits);
            sfpi::dst_reg[dst_up * tile_stride] = ink_tanh(u * 0.04f) * 25.0f;
        }
        {
            const sfpi::vFloat g = sfpi::dst_reg[dst_gate * tile_stride] * Converter::as_float(f_bits);
            sfpi::dst_reg[dst_tmp * tile_stride] = ink_tanh(g * 0.25f) * 4.0f;
        }
        {
            const sfpi::vFloat g = sfpi::dst_reg[dst_gate * tile_stride] * Converter::as_float(f_bits);
            sfpi::vFloat r = _sfpu_sigmoid_<true>(g) * sfpi::dst_reg[dst_tmp * tile_stride];
            r = r * sfpi::dst_reg[dst_up * tile_stride];
            if constexpr (R == 1) {
                sfpi::dst_reg[dst_out * tile_stride] = r * Converter::as_float(w);
            } else {
                sfpi::dst_reg[dst_out * tile_stride] = r * sfpi::dst_reg[dst_w * tile_stride];
            }
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu

// Unit u's SwiGLU: its slot from the plan page, [f13, row weights] from the slot table page.
template <int ITERS, uint32_t R, uint32_t WR>
inline void ink_unit_swiglu(const uint32_t plan, const uint32_t table, const uint32_t u) {
    const uint32_t slot = reinterpret_cast<volatile uint32_t*>(plan)[1 + u];
    const uint32_t entry = table + slot * (1 + WR) * 4;
    const uint32_t f_bits = reinterpret_cast<volatile uint32_t*>(entry)[0];
    const uint32_t w = R == 1 ? reinterpret_cast<volatile uint32_t*>(entry)[1] : entry + 4;
    _llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::calculate_ink_swiglu<ITERS, R>, 0, 1, 0, VectorMode::R, w,
                                          f_bits);
}
#endif
#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
#include "models/demos/blackhole/inkling2/tt/kernels/moe/moe_router.hpp"
#endif

#define INK_CTA(name) get_named_compile_time_arg_val(name)

// CB indices.  Every CB is declared on the program's one rectangle (same address on every core).  Workers: cb_w
// (weight blocks, both roles), cb_scratch (every role), cb_route (TRISC plan / slot table), cb_stage (down output
// staging), cb_rows (R x [sel E | scores E] FP32, the stream / write side flag words (64 B), then the down cores'
// counters, one word per 16 B per slot).  Gate/up: cb_gx (x tiles), cb_gout (activation tile).  Down ([AR, 32]
// tiles): cb_dx (act tiles), cb_dout.  Router: cb_rx (x face rows), cb_rw (weight tiles), cb_part (own partials,
// FP32), cb_rout ([sel | scores], FP32), cb_gather (per row: partials + bias, FP32, unpacked straight to DEST).
constexpr uint32_t cb_gx = 0, cb_w = 1, cb_scratch = 2, cb_route = 3, cb_gout = 4, cb_stage = 5, cb_rows = 6,
                   cb_gather = 7, cb_dx = 8, cb_dout = 9, cb_rx = 10, cb_rw = 11, cb_part = 12, cb_rout = 13;
// Flag values (magic numbers rather than 1: flag words start as whatever L1 held before; the reader clears them).
constexpr uint32_t ROUTER_FLAG = 0x9e3779b1u;  // router -> worker: rows landed
constexpr uint32_t TOPK_FLAG = 0x85ebca6bu;    // worker write side -> stream side: its rows' ids are in the id table
constexpr uint32_t PLAN_FLAG = 0xc2b2ae35u;    // worker stream side -> write side: the slot plan is ready

constexpr uint32_t E = INK_CTA("experts");
constexpr uint32_t top_k = INK_CTA("top_k");
constexpr uint32_t banks = INK_CTA("banks");
constexpr uint32_t inter_tiles = INK_CTA("inter_tiles");
constexpr uint32_t pair_base = INK_CTA("pair_base");
constexpr uint32_t pair_odd = INK_CTA("pair_odd");
constexpr uint32_t routers = INK_CTA("routers");      // router cores per row group (K slices)
constexpr uint32_t R = INK_CTA("rows");               // token rows
constexpr uint32_t RP = INK_CTA("row_pad");           // R rounded up to even (64 B reads)
constexpr uint32_t WR = INK_CTA("w_rows");            // row weights per slot entry (4 / 8 / 16)
constexpr uint32_t max_slots = INK_CTA("slots");      // min(E, R top_k)
constexpr uint32_t group = INK_CTA("act_group");      // intermediate tiles per act page
constexpr uint32_t AR = INK_CTA("act_rows");          // rows of an act tile ([AR, 32], >= R)
constexpr bool BAL = INK_CTA("balance") != 0;         // bank-balanced gate/up (dup pairs)
constexpr uint32_t RGROUPS = INK_CTA("router_groups");
constexpr uint32_t n_gate = INK_CTA("n_gate");        // gate/up cores (banks x gate_splits)
constexpr uint32_t n_down = INK_CTA("n_down");        // down cores
constexpr uint32_t grid_x = INK_CTA("grid_x");        // core table stride (logical grid width)
constexpr uint32_t table_words = INK_CTA("table_words");
constexpr uint32_t noc1_mask = INK_CTA("noc1_mask");  // banks whose stream cores swap NOCs (stream on NOC1)
constexpr uint32_t tile_bytes = INK_CTA("w_tile_bytes");
constexpr uint32_t RN = (R + 1) / 2;                  // rows selected on the stream side / routed by group 0
constexpr uint32_t row_bytes = 2 * E * 4;             // [sel | scores] FP32
constexpr uint32_t act_pages = inter_tiles / group;   // per slot
constexpr bool RTK = R > 8;                           // router-side top-k + weights (records), see the header
constexpr uint32_t rec_bytes = 128;                   // RTK record: [ids | 0 | w | w f2], 8 words each
constexpr uint32_t rows_area = RTK ? R * rec_bytes : R * row_bytes;  // cb_rows: rows / records, then flags, counters
static_assert(top_k <= 16 && R >= 1 && R <= 16 && RP % 2 == 0 && RP >= R && WR >= R);
static_assert(inter_tiles % group == 0);
static_assert(RGROUPS == 1 || (RGROUPS == 2 && R > 1) || (RTK && RGROUPS == 3));
static_assert(!RTK || (top_k == 8 && (R + RGROUPS - 1) / RGROUPS <= routers));

// Worker scratch (cb_scratch, core-local) byte offsets: factors [f13 | f2], route headers (writer core, 128 B per row),
// id table [R][top_k], local flags, plan [U, slot ids], R = 1: the write side's top-k of the upper expert half (keys,
// ids), gate/up: the moved-floating-pair masks per slot (balance, one byte each).  R <= 8: the 8-row layout; R > 8
// moves everything after the headers up.
constexpr bool SCR_WIDE = RTK;
constexpr uint32_t SCR_FAC = 0, SCR_HDR = 1024, SCR_IDS = SCR_WIDE ? 3072 : 2048,
                   SCR_FLAG_TOPK = SCR_WIDE ? 3584 : 2304, SCR_FLAG_PLAN = SCR_FLAG_TOPK + 16,
                   SCR_PLAN = SCR_WIDE ? 3648 : 2368, SCR_HALF = SCR_WIDE ? 4224 : 3072,
                   SCR_MOVED = SCR_WIDE ? 4352 : 3136;
static_assert(SCR_HDR + 128 * R <= SCR_IDS && SCR_IDS + 4 * top_k * R <= SCR_FLAG_TOPK &&
              SCR_FLAG_PLAN + 16 <= SCR_PLAN && SCR_PLAN + 4 * (1 + max_slots) <= SCR_HALF &&
              SCR_HALF + 8 * top_k <= SCR_MOVED && SCR_MOVED + max_slots <= 16384);

// Common runtime args (every RISC): tensor addresses, the core table (4 bytes per word), then (noc x, y) of the gate/up
// cores, the down cores (contiguous: worker w at CA_GATE_NOC + 2 w) and the router cores (group-major).
constexpr uint32_t CA_X = 0, CA_ACT = 1, CA_OUT = 2, CA_ROUTE = 3, CA_FAC = 4, CA_GATE_UP = 5, CA_DOWN = 6, CA_DUP = 7,
                   CA_ROUTER_W = 8, CA_BIAS = 9, CA_TABLE = 10;
constexpr uint32_t CA_GATE_NOC = CA_TABLE + table_words;
constexpr uint32_t CA_DOWN_NOC = CA_GATE_NOC + 2 * n_gate;
constexpr uint32_t CA_ROUTER_NOC = CA_DOWN_NOC + 2 * n_down;

// Per-role stream constants (RL 0 gate/up, 1 down).
template <uint32_t RL>
struct Role;
template <>
struct Role<0> {
    static constexpr uint32_t kt = INK_CTA("gate_k_tiles"), cols = INK_CTA("gate_cols"), blk_k = INK_CTA("gate_blk_k"),
                              splits = INK_CTA("gate_splits"), pages = INK_CTA("gate_pages"),
                              page_bytes = INK_CTA("gate_page_bytes"), buffers = INK_CTA("gate_buffers"),
                              ahead = INK_CTA("gate_ahead"), cb_x = cb_gx, cb_out = cb_gout, w_arg = CA_GATE_UP;
};
template <>
struct Role<1> {
    static constexpr uint32_t kt = INK_CTA("down_k_tiles"), cols = INK_CTA("down_cols"), blk_k = INK_CTA("down_blk_k"),
                              splits = INK_CTA("down_splits"), pages = INK_CTA("down_pages"),
                              page_bytes = INK_CTA("down_page_bytes"), buffers = INK_CTA("down_buffers"),
                              ahead = INK_CTA("down_ahead"), cb_x = cb_dx, cb_out = cb_dout, w_arg = CA_DOWN;
};
static_assert(Role<0>::cols == 2 && Role<0>::kt % Role<0>::blk_k == 0 && Role<1>::kt % Role<1>::blk_k == 0);

// Router row group RGI: rows [QR0, QR0 + QRN) (the rows split as evenly as possible, the first groups one more; two
// groups: [0, RN), [RN, R)); its x face rows are read from the even row QA (64 B-aligned DRAM reads): QXP rows per
// face, row QR0 + q at QD + q.
template <uint32_t RGI>
struct RowGroup {
    static_assert(RGI < RGROUPS);
    static constexpr uint32_t base = R / RGROUPS, extra = R % RGROUPS;
    static constexpr uint32_t QR0 = RGI * base + (RGI < extra ? RGI : extra);
    static constexpr uint32_t QRN = base + (RGI < extra ? 1 : 0);
    static constexpr uint32_t QA = QR0 & ~1u, QD = QR0 - QA;
    static constexpr uint32_t QXP = (QD + QRN + 1) & ~1u;
};

// 1 + this core's index in [gate/up | down | router] cores, 0 = not a MoE core.
inline uint32_t core_slot() {
    const uint32_t i = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    return (get_common_arg_val<uint32_t>(CA_TABLE + i / 4) >> (8 * (i % 4))) & 0xffu;
}

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
// Positional compile-time args (both data-movement RISCs): accessors of x, act, out, route, factors, router_w, bias.
constexpr auto x_args = TensorAccessorArgs<0>();
constexpr auto act_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
constexpr auto route_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
constexpr auto fac_args = TensorAccessorArgs<route_args.next_compile_time_args_offset()>();
constexpr auto rw_args = TensorAccessorArgs<fac_args.next_compile_time_args_offset()>();
constexpr auto bias_args = TensorAccessorArgs<rw_args.next_compile_time_args_offset()>();

inline uint32_t carg(uint32_t i) { return get_common_arg_val<uint32_t>(i); }

// Top-k of rows [q0, q1) of the rows buffer into the id table.
inline void topk_rows(uint32_t rows, volatile tt_l1_ptr uint32_t* table, uint32_t q0, uint32_t q1) {
    for (uint32_t q = q0; q < q1; ++q) {
        uint32_t ids[top_k];
        ink_moe::select_topk<E, top_k>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows + q * row_bytes), ids);
        for (uint32_t k = 0; k < top_k; ++k) {
            table[q * top_k + k] = ids[k];
        }
    }
}

inline void wait_flag(uint32_t addr, uint32_t value) {
    auto* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    noc_semaphore_wait(f, value);
    noc_semaphore_set(f, 0);
}

// Transaction-id ring feeding `cb` with fixed-size blocks: up to `ahead` blocks in flight, each pushed once landed.
// `buffers` = the CB's size in blocks (the ring wraps where the CB does: cb_w is shared by both stream roles, sized
// for both block sizes); transaction ids cycle over min(buffers, 15) (> ahead + 1: never two blocks in flight on one).
template <uint32_t cb, uint32_t buffers, uint32_t ahead, uint32_t block_tiles, uint32_t block_bytes>
struct TridRing {
    static constexpr uint32_t trids = buffers < 15 ? buffers : 15;
    static_assert(ahead >= 1 && ahead + 1 < trids);
    uint32_t begin, end, write, free = buffers, trid = 1, wait = 1, pending = 0;

    void start() {
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
        cb_reserve_back(cb, block_tiles * (ahead + 1));
        begin = write = get_write_ptr(cb);
        end = begin + buffers * block_bytes;
    }
    void push_oldest() {
        noc_async_read_barrier_with_trid(wait);
        cb_push_back(cb, block_tiles);
        wait = wait == trids ? 1 : wait + 1;
        --pending;
    }
    void next() {
        ++pending;
        write += block_bytes;
        if (write >= end) {
            write = begin;
        }
        if (free == buffers - ahead) {
            push_oldest();
            cb_reserve_back(cb, block_tiles * (ahead + 1));
        } else {
            --free;
        }
        trid = trid == trids ? 1 : trid + 1;
    }
    void finish() {
        while (pending) {
            push_oldest();
        }
    }
};

// Worker stream side (NCRISC; BRISC on the cores of noc1_mask banks): routing of rows [0, RN) + the slot plan, then
// the weight stream (gate/up: also the routing weights / TRISC slot table and, on the writer core, the route rows).
template <uint32_t RL>
inline void stream_side(const uint32_t bank, const uint32_t part) {
    using C = Role<RL>;
    constexpr bool gate_up = RL == 0;
    constexpr uint32_t kt = C::kt, cols = C::cols, splits = C::splits, pages = C::pages, page_bytes = C::page_bytes;
    constexpr uint32_t blocks = kt / C::blk_k;  // per unit (gate/up) / per slot (down)
    constexpr uint32_t block_tiles = C::blk_k * cols, block_bytes = block_tiles * tile_bytes;
    constexpr uint32_t route_page = INK_CTA("route_page");
    static_assert(pages * page_bytes == block_bytes);
    const uint32_t vc = (bank + part) & 3;
    const auto factors = TensorAccessor(fac_args, carg(CA_FAC), 2 * E * 4);
    const auto route_out = TensorAccessor(route_args, carg(CA_ROUTE), 4 * (32 + 2 * E));
    const bool writer = gate_up && bank == 0 && part == 0;
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint32_t rows = get_write_ptr(cb_rows);
    auto* fac = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_FAC);
    auto* id_tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_IDS);
    auto* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_PLAN);
    if constexpr (gate_up) {
        noc_async_read(factors.get_noc_addr(0), scratch + SCR_FAC, 2 * E * 4);
    }
    {
        DeviceZoneScopedN("W-FLAG");
        if constexpr (RTK) {
            wait_flag(get_semaphore(2), R);  // every row's record landed
        } else {
            wait_flag(rows + R * row_bytes, ROUTER_FLAG);  // stream side flag word
        }
    }
    noc_async_read_barrier();
    uint32_t slot_e[max_slots];
    uint32_t U;
    {
        DeviceZoneScopedN("INK-ROUTE");
        if constexpr (RTK) {  // ids from the router's records
            for (uint32_t q = 0; q < R; ++q) {
                auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows + q * rec_bytes);
                for (uint32_t k = 0; k < top_k; ++k) {
                    id_tab[q * top_k + k] = rec[k];
                }
            }
        } else if constexpr (R == 1) {  // lower expert half here, upper half on the write side (route_plan), then
                                        // the merge
            uint32_t ka[top_k], ia[top_k], ids[top_k];
            ink_moe::select_topk_keys<top_k>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows), 0, E / 2, ka, ia);
            wait_flag(scratch + SCR_FLAG_TOPK, TOPK_FLAG);
            auto* half = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_HALF);
            ink_moe::merge_topk<top_k>(ka, ia, half, half + top_k, ids);
            for (uint32_t k = 0; k < top_k; ++k) {
                id_tab[k] = ids[k];
            }
        } else {
            topk_rows(rows, id_tab, 0, RN);
        }
        if constexpr (R > RN && !RTK) {
            wait_flag(scratch + SCR_FLAG_TOPK, TOPK_FLAG);
        }
        U = ink_moe::union_ascending<E>(id_tab, R * top_k, slot_e);
        for (uint32_t u = 0; u < U; ++u) {
            plan[1 + u] = slot_e[u];
        }
        plan[0] = U;
        if constexpr (gate_up && BAL) {
            ink_moe::balance_floating<banks, pair_base>(slot_e, U,
                                                        reinterpret_cast<volatile uint8_t*>(scratch + SCR_MOVED));
        }
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_FLAG_PLAN) = PLAN_FLAG;
    }
    // Weight stream first (it only needs the slots); the gate/up cores compute the routing weights while their first
    // unit's blocks are in flight, the down cores never need them.
    const uint64_t base = get_noc_addr_from_bank_id<true>(bank, carg(C::w_arg));
    TridRing<cb_w, C::buffers, C::ahead, block_tiles, block_bytes> ring;
    ring.start();
    noc_async_read_one_packet_set_state<true>(base, page_bytes, vc);
    // one unit / slot slice: `blocks` blocks from offset src of the run at base_lo (the same bank, so the same NoC
    // endpoint as the state)
    auto stream = [&](uint32_t base_lo, uint32_t src) {
        for (uint32_t b = 0; b < blocks; ++b) {
            uint32_t dst = ring.write;
            noc_async_read_set_trid(ring.trid);
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read_one_packet_with_state_with_trid(base_lo, src, dst, ring.trid);
                src += page_bytes;
                dst += page_bytes;
            }
            ring.next();
        }
    };
    if constexpr (gate_up) {
        // TRISC page 0: [n, slot of each of this core's units]; page 1: per slot [f13, w_0 f2 .. w_{WR-1} f2].
        constexpr uint32_t entry = 1 + WR;
        constexpr uint32_t unit_bytes = 2 * kt * tile_bytes;
        const uint32_t main_lo = static_cast<uint32_t>(base);
        const uint32_t dup_lo = BAL ? static_cast<uint32_t>(get_noc_addr_from_bank_id<true>(bank, carg(CA_DUP))) : 0;
        const auto* moved = reinterpret_cast<const volatile uint8_t*>(scratch + SCR_MOVED);
        cb_reserve_back(cb_route, 2);
        auto* p0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route));
        auto* p1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route) + route_page);
        // This core's units (bank_units order; unit q of the bank goes to core q % splits).
        uint32_t n = 0, first_lo = 0, first_src = 0, q = 0;
        ink_moe::bank_units<banks, pair_base, pair_odd, BAL>(
            bank, slot_e, moved, U, [&](uint32_t u, uint32_t, uint32_t unit, bool dup) {
                if (q++ % splits == part) {
                    if (n == 0) {
                        first_lo = dup ? dup_lo : main_lo;
                        first_src = unit * unit_bytes;
                    }
                    p0[1 + n++] = u;
                }
            });
        p0[0] = n;
        cb_push_back(cb_route, 1);
        if (n) {
            stream(first_lo, first_src);
        }
        {
            uint8_t slot_of[E];
            for (uint32_t u = 0; u < U; ++u) {
                slot_of[slot_e[u]] = u;
                p1[u * entry] = fac[slot_e[u]];
                for (uint32_t r = 0; r < WR; ++r) {
                    p1[u * entry + 1 + r] = 0;
                }
            }
            [[maybe_unused]] const float scale = INK_CTA("route_scale_x1000") / 1000.0f;
            for (uint32_t q = 0; q < R; ++q) {
                if constexpr (RTK) {  // the router's record: ids, w (words 16..), w f2 (words 24..)
                    auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows + q * rec_bytes);
                    for (uint32_t k = 0; k < top_k; ++k) {
                        p1[slot_of[rec[k]] * entry + 1 + q] = rec[24 + k];
                    }
                    if (writer) {  // route row q words 0-31: ids, weights at 16 (tests)
                        auto* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_HDR + q * 128);
                        for (uint32_t k = 0; k < 32; ++k) {
                            o[k] = k < 24 ? rec[k] : 0u;
                        }
                        noc_async_write(scratch + SCR_HDR + q * 128, route_out.get_noc_addr(q), 128);
                    }
                    continue;
                }
                uint32_t ids[top_k];
                for (uint32_t k = 0; k < top_k; ++k) {
                    ids[k] = id_tab[q * top_k + k];
                }
                float w[top_k];
                ink_moe::route_weights<top_k>(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows + q * row_bytes + E * 4), ids, scale, w);
                for (uint32_t k = 0; k < top_k; ++k) {
                    p1[slot_of[ids[k]] * entry + 1 + q] = ink_moe::as_bits(w[k] * ink_moe::as_float(fac[E + ids[k]]));
                }
                if (writer) {  // route row q words 0-31: ids, weights at 16 (tests)
                    auto* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_HDR + q * 128);
                    for (uint32_t k = 0; k < 32; ++k) {
                        o[k] = k < top_k ? ids[k] : (k >= 16 && k < 16 + top_k ? ink_moe::as_bits(w[k - 16]) : 0u);
                    }
                    noc_async_write(scratch + SCR_HDR + q * 128, route_out.get_noc_addr(q), 128);
                }
            }
            cb_push_back(cb_route, 1);
        }
        q = 0;
        uint32_t seen = 0;
        ink_moe::bank_units<banks, pair_base, pair_odd, BAL>(
            bank, slot_e, moved, U, [&](uint32_t, uint32_t, uint32_t unit, bool dup) {
                if (q++ % splits == part && seen++) {
                    stream(dup ? dup_lo : main_lo, unit * unit_bytes);
                }
            });
    } else {
        constexpr uint32_t group_bytes = kt * cols * tile_bytes;
        constexpr uint32_t expert_bytes = splits * group_bytes;
        cb_reserve_back(cb_route, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route))[0] = U;
        cb_push_back(cb_route, 1);
        for (uint32_t u = 0; u < U; ++u) {
            stream(static_cast<uint32_t>(base), slot_e[u] * expert_bytes + part * group_bytes);
        }
    }
    ring.finish();
    noc_async_read_set_trid(0);
    noc_async_write_barrier();
}

// Worker write side (BRISC; NCRISC on the cores of noc1_mask banks): routing share of rows [RN, R), then gate/up:
// x tiles in, activation rows out + counters; down: act tiles in, output tiles out.
template <uint32_t RL>
inline void write_side(const uint32_t bank, const uint32_t part) {
    using C = Role<RL>;
    constexpr uint32_t kt = C::kt, cols = C::cols, splits = C::splits, cb_x = C::cb_x, cb_out = C::cb_out;
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint32_t rows = get_write_ptr(cb_rows);
    auto* id_tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_IDS);
    auto* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_PLAN);
    // Routing share: top-k of rows [RN, R) (RTK: none, the router selected), then the plan from the stream side.
    auto route_plan = [&]() -> uint32_t {
        if constexpr (!RTK) {
            wait_flag(rows + R * row_bytes + 16, ROUTER_FLAG);  // write side flag word
            if constexpr (R == 1) {  // upper expert half (keys, ids) for the stream side's merge
                uint32_t kb[top_k], ib[top_k];
                ink_moe::select_topk_keys<top_k>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows), E / 2, E, kb,
                                                 ib);
                auto* half = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_HALF);
                for (uint32_t k = 0; k < top_k; ++k) {
                    half[k] = kb[k];
                    half[top_k + k] = ib[k];
                }
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_FLAG_TOPK) = TOPK_FLAG;
            } else if constexpr (R > RN) {
                topk_rows(rows, id_tab, RN, R);
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SCR_FLAG_TOPK) = TOPK_FLAG;
            }
        }
        wait_flag(scratch + SCR_FLAG_PLAN, PLAN_FLAG);
        return plan[0];
    };
    const auto act = TensorAccessor(act_args, carg(CA_ACT), group * AR * 64);
    const uint32_t counters = rows + rows_area + 64;  // cb_rows: the same address on every down core
    if constexpr (RL == 0) {
        const auto x = TensorAccessor(x_args, carg(CA_X), 2048);
        // x rows 0..RP-1 of faces 0 / 1 straight into the in0 tiles (rows >= R are don't-care: matmul rows are
        // independent).
        cb_reserve_back(cb_x, kt);
        {
            const uint32_t d = get_write_ptr(cb_x);
            for (uint32_t t = 0; t < kt; ++t) {
                const uint64_t src = x.get_noc_addr(t);
                noc_async_read(src, d + t * 2048, RP * 32);
                noc_async_read(src + 512, d + t * 2048 + 512, RP * 32);
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_x, kt);
        const uint32_t U = route_plan();
        bool ready = false;
        uint32_t q = 0;
        ink_moe::bank_units<banks, pair_base, pair_odd, BAL>(
            bank, plan + 1, reinterpret_cast<const volatile uint8_t*>(scratch + SCR_MOVED), U,
            [&](uint32_t u, uint32_t j, uint32_t, bool) {
                if (q++ % splits != part) {
                    return;
                }
                cb_wait_front(cb_out, 1);
                const uint32_t t = get_read_ptr(cb_out);
                // act tile (u, j) = [AR, 32] tile layout: face 0 rows, then face 1 rows (rows R..AR-1 don't-care)
                const uint64_t dst = act.get_noc_addr(u * act_pages + j / group) + (j % group) * AR * 64;
                noc_async_write(t, dst, R * 32);                  // face 0 rows 0..R-1
                noc_async_write(t + 512, dst + AR * 32, R * 32);  // face 1 rows 0..R-1
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
                if (!ready) {  // every down core has zeroed its counters
                    wait_flag(get_semaphore(1), n_down);
                    ready = true;
                }
                const uint32_t c = counters + u * 16;
                for (uint32_t d = 0; d < n_down; ++d) {
                    noc_semaphore_inc(get_noc_addr(carg(CA_DOWN_NOC + 2 * d), carg(CA_DOWN_NOC + 2 * d + 1), c), 1);
                }
            });
        if (!ready) {
            wait_flag(get_semaphore(1), n_down);
        }
        noc_async_atomic_barrier();
    } else {
        const auto out = TensorAccessor(out_args, carg(CA_OUT), 2048);
        // Zeroed counters (then every gate/up core may count into them) and output staging tiles.
        constexpr uint32_t counter_bytes = max_slots * 16;
        const uint32_t stage = get_write_ptr(cb_stage);
        for (uint32_t off = 0; off < counter_bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), counters + off,
                           counter_bytes - off < MEM_ZEROS_SIZE ? counter_bytes - off : MEM_ZEROS_SIZE);
        }
        for (uint32_t off = 0; off < cols * 2048; off += MEM_ZEROS_SIZE) {
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), stage + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        {
            const uint32_t sem = get_semaphore(1);
            for (uint32_t g = 0; g < n_gate; ++g) {
                noc_semaphore_inc(get_noc_addr(carg(CA_GATE_NOC + 2 * g), carg(CA_GATE_NOC + 2 * g + 1), sem), 1);
            }
        }
        const uint32_t U = route_plan();
        for (uint32_t u = 0; u < U; ++u) {
            cb_reserve_back(cb_x, kt);
            const uint32_t d = get_write_ptr(cb_x);
            {
                DeviceZoneScopedN("D-ACT");
                wait_flag(counters + u * 16, inter_tiles);
            }
            // in0 tiles are [AR, 32] (AR * 64 B, the act page layout): one read per page
            for (uint32_t pg = 0; pg < act_pages; ++pg) {
                noc_async_read(act.get_noc_addr(u * act_pages + pg), d + pg * group * AR * 64, group * AR * 64);
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, kt);
        }
        cb_wait_front(cb_out, cols);
        {
            auto* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out));
            auto* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
            constexpr uint32_t o_tile = AR * 16, o_face1 = AR * 8;  // out tile c: [AR, 32], faces 0 / 1 of AR rows
            for (uint32_t c = 0; c < cols; ++c) {
                for (uint32_t i = 0; i < R * 8; ++i) {
                    st[c * 512 + i] = o[c * o_tile + i];                  // face 0 rows 0..R-1
                    st[c * 512 + 128 + i] = o[c * o_tile + o_face1 + i];  // face 1 rows 0..R-1
                }
            }
        }
        constexpr uint32_t bank_cols = cols * splits;
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write(stage + c * 2048, out.get_noc_addr(bank * bank_cols + part * cols + c), 2048);
        }
        cb_pop_front(cb_out, cols);
    }
}

#if defined(COMPILE_FOR_NCRISC)
// Router NCRISC: this core's router weight tiles [r rt, (r + 1) rt) (2 KB pages), all reads issued at once (one
// transaction id per batch of rbatch pages), each batch pushed as it lands.
inline void router_weights(const uint32_t r) {
    constexpr uint32_t rt = INK_CTA("router_tiles");
    constexpr uint32_t rbatch = INK_CTA("router_batch");
    constexpr uint32_t batches = rt / rbatch;
    static_assert(rt % rbatch == 0 && batches <= 15);
    const auto wr = TensorAccessor(rw_args, carg(CA_ROUTER_W), 2048);
    const uint32_t first = r * rt;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    cb_reserve_back(cb_rw, rt);
    const uint32_t dst = get_write_ptr(cb_rw);
    for (uint32_t b = 0; b < batches; ++b) {
        noc_async_read_set_trid(b + 1);
        for (uint32_t i = 0; i < rbatch; ++i) {
            const uint32_t p = b * rbatch + i;
            noc_async_read(wr.get_noc_addr(first + p), dst + p * 2048, 2048);
        }
    }
    for (uint32_t b = 0; b < batches; ++b) {
        noc_async_read_barrier_with_trid(b + 1);
        cb_push_back(cb_rw, rbatch);
    }
    noc_async_read_set_trid(0);
}
#else
// Router BRISC (row group RGI): x face rows + bias in, partial all-gather over the group's router cores, then
// [sel | scores] of each row out to this core's workers (w % routers == r) and their flag word.
template <uint32_t RGI>
inline void router_rows(const uint32_t r) {
    using G = RowGroup<RGI>;
    constexpr uint32_t QR0 = G::QR0, QRN = G::QRN, QA = G::QA, QXP = G::QXP;
    constexpr uint32_t ks = 8 * INK_CTA("router_tiles");  // k per router core
    constexpr uint32_t gather_tiles = (routers + 8) / 8;  // partials + bias, 512 B each
    constexpr uint32_t region = gather_tiles * 4096;      // per row
    constexpr uint32_t n_workers = n_gate + n_down;
    constexpr uint32_t rnoc = CA_ROUTER_NOC + 2 * RGI * routers;  // this group's router cores
    const auto x = TensorAccessor(x_args, carg(CA_X), 2048);
    const auto bias = TensorAccessor(bias_args, carg(CA_BIAS), E * 4);
    const auto dump = TensorAccessor(route_args, carg(CA_ROUTE), 4 * (32 + 2 * E));
    const uint32_t scratch = get_write_ptr(cb_scratch);
    // x rows of faces 0 / 1 of the tiles covering [k0, k0 + ks): tile t at 64 QXP t B, face f at + 32 QXP f (TRISC
    // addresses them in place), and the bias into every row's region (partial index `routers`).
    const uint32_t k0 = r * ks, t0 = k0 / 32, nt = (k0 + ks - 1) / 32 - t0 + 1;
    cb_reserve_back(cb_rx, 1);
    const uint32_t xb = get_write_ptr(cb_rx);
    for (uint32_t t = 0; t < nt; ++t) {
        const uint64_t src = x.get_noc_addr(t0 + t);
        noc_async_read(src + QA * 32, xb + t * 64 * QXP, 32 * QXP);
        noc_async_read(src + 512 + QA * 32, xb + t * 64 * QXP + 32 * QXP, 32 * QXP);
    }
    cb_reserve_back(cb_gather, QRN * gather_tiles);
    const uint32_t gather = get_write_ptr(cb_gather);
    for (uint32_t q = 0; q < QRN; ++q) {
        noc_async_read(bias.get_noc_addr(0), gather + q * region + routers * 512, E * 4);
    }
    noc_async_read_barrier();
    cb_push_back(cb_rx, 1);
    // All-gather: this core's partial of row q (512 B at 512 q of the packed FP32 tile) into slot r of row q's
    // region on every router core of the group, then semaphore 0 += 1 there; all `routers` partials landed -> TRISC.
    {
        DeviceZoneScopedN("RT-PART");
        cb_wait_front(cb_part, 1);
    }
    const uint32_t part = get_read_ptr(cb_part);
    for (uint32_t j = 0; j < routers; ++j) {
        const uint64_t dst = get_noc_addr(carg(rnoc + 2 * j), carg(rnoc + 2 * j + 1), gather);
        for (uint32_t q = 0; q < QRN; ++q) {
            noc_async_write(part + q * 512, dst + q * region + r * 512, 512);
        }
    }
    noc_async_write_barrier();
    const uint32_t sem = get_semaphore(0);
    for (uint32_t j = 0; j < routers; ++j) {
        noc_semaphore_inc(get_noc_addr(carg(rnoc + 2 * j), carg(rnoc + 2 * j + 1), sem), 1);
    }
    cb_pop_front(cb_part, 1);
    {
        DeviceZoneScopedN("RT-GATHER");
        wait_flag(sem, routers);
    }
    cb_push_back(cb_gather, QRN * gather_tiles);
    // Fan-out: row q's [sel | scores] (1 KB) into the rows buffer of this core's workers, then their flag words.
    const uint32_t rows = get_write_ptr(cb_rows);
    for (uint32_t q = 0; q < QRN; ++q) {
        {
            DeviceZoneScopedN("RT-ROWS");
            cb_wait_front(cb_rout, 1);
        }
        const uint32_t res = get_read_ptr(cb_rout);
        for (uint32_t w = r; w < n_workers; w += routers) {
            noc_async_write(res, get_noc_addr(carg(CA_GATE_NOC + 2 * w), carg(CA_GATE_NOC + 2 * w + 1),
                                              rows + (QR0 + q) * row_bytes),
                            row_bytes);
        }
        if (r == 0) {
            noc_async_write(res, dump.get_noc_addr(QR0 + q) + 128, row_bytes);  // route row words 32.. (tests)
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_rout, 1);
    }
    noc_async_write_barrier();
    auto* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    // flag words: stream side at +0 (rows [0, RN): group 0), write side at +16 (group 1; one group: both)
    for (uint32_t i = 0; i < 8; ++i) {
        flag[i] = i % 4 == 0 ? ROUTER_FLAG : 0;
    }
    constexpr uint32_t flag_off = RGROUPS == 1 ? 0 : 16 * RGI, flag_bytes = RGROUPS == 1 ? 32 : 16;
    for (uint32_t w = r; w < n_workers; w += routers) {
        noc_async_write(scratch, get_noc_addr(carg(CA_GATE_NOC + 2 * w), carg(CA_GATE_NOC + 2 * w + 1),
                                              rows + R * row_bytes + flag_off),
                        flag_bytes);
    }
}

// Router BRISC, RTK (row group RGI): x face rows in; this core's partial of row q to router core q of the group (only
// cores q < QRN finish rows), semaphore 0 there.  Core r < QRN (owner of row QR0 + r): bias and factors in, its row's
// gathered partials -> TRISC -> [sel | scores] -> route row dump (tests), top-k + weights -> a record [ids | 0 | w |
// w f2] in the `rows` buffer of every worker, then semaphore 2 += 1 there.
template <uint32_t RGI>
inline void router_records(const uint32_t r) {
    using G = RowGroup<RGI>;
    constexpr uint32_t QR0 = G::QR0, QRN = G::QRN, QA = G::QA, QXP = G::QXP;
    constexpr uint32_t ks = 8 * INK_CTA("router_tiles");  // k per router core
    constexpr uint32_t n_workers = n_gate + n_down;
    constexpr uint32_t rnoc = CA_ROUTER_NOC + 2 * RGI * routers;  // this group's router cores
    constexpr uint32_t RT_REC = 0, RT_FAC = 1024;                 // router scratch: the record, factors [f13 | f2]
    const auto x = TensorAccessor(x_args, carg(CA_X), 2048);
    const auto bias = TensorAccessor(bias_args, carg(CA_BIAS), E * 4);
    const auto factors = TensorAccessor(fac_args, carg(CA_FAC), 2 * E * 4);
    const auto dump = TensorAccessor(route_args, carg(CA_ROUTE), 4 * (32 + 2 * E));
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const bool owner = r < QRN;
    const uint32_t k0 = r * ks, t0 = k0 / 32, nt = (k0 + ks - 1) / 32 - t0 + 1;
    cb_reserve_back(cb_rx, 1);
    const uint32_t xb = get_write_ptr(cb_rx);
    for (uint32_t t = 0; t < nt; ++t) {
        const uint64_t src = x.get_noc_addr(t0 + t);
        noc_async_read(src + QA * 32, xb + t * 64 * QXP, 32 * QXP);
        noc_async_read(src + 512 + QA * 32, xb + t * 64 * QXP + 32 * QXP, 32 * QXP);
    }
    const uint32_t gather = get_write_ptr(cb_gather);  // one row region: partials 0..routers-1, then the bias
    if (owner) {
        cb_reserve_back(cb_gather, (routers + 8) / 8);
        noc_async_read(bias.get_noc_addr(0), gather + routers * 512, E * 4);
        noc_async_read(factors.get_noc_addr(0), scratch + RT_FAC, 2 * E * 4);
    }
    noc_async_read_barrier();
    cb_push_back(cb_rx, 1);
    {
        DeviceZoneScopedN("RT-PART");
        cb_wait_front(cb_part, 1);
    }
    const uint32_t part = get_read_ptr(cb_part);
    for (uint32_t q = 0; q < QRN; ++q) {
        noc_async_write(part + q * 512, get_noc_addr(carg(rnoc + 2 * q), carg(rnoc + 2 * q + 1), gather + r * 512),
                        512);
    }
    noc_async_write_barrier();
    const uint32_t sem = get_semaphore(0);
    for (uint32_t q = 0; q < QRN; ++q) {
        noc_semaphore_inc(get_noc_addr(carg(rnoc + 2 * q), carg(rnoc + 2 * q + 1), sem), 1);
    }
    cb_pop_front(cb_part, 1);
    if (!owner) {
        return;
    }
    {
        DeviceZoneScopedN("RT-GATHER");
        wait_flag(sem, routers);
    }
    cb_push_back(cb_gather, (routers + 8) / 8);
    {
        DeviceZoneScopedN("RT-ROWS");
        cb_wait_front(cb_rout, 1);
    }
    const uint32_t res = get_read_ptr(cb_rout);
    noc_async_write(res, dump.get_noc_addr(QR0 + r) + 128, row_bytes);  // route row words 32.. (tests)
    uint32_t ids[top_k];
    ink_moe::select_topk<E, top_k>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res), ids);
    float w[top_k];
    ink_moe::route_weights<top_k>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res + E * 4), ids,
                                  INK_CTA("route_scale_x1000") / 1000.0f, w);
    auto* fac = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + RT_FAC);
    auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + RT_REC);
    for (uint32_t k = 0; k < top_k; ++k) {
        rec[k] = ids[k];
        rec[8 + k] = 0;
        rec[16 + k] = ink_moe::as_bits(w[k]);
        rec[24 + k] = ink_moe::as_bits(w[k] * ink_moe::as_float(fac[E + ids[k]]));
    }
    noc_async_write_barrier();
    cb_pop_front(cb_rout, 1);
    const uint32_t rows = get_write_ptr(cb_rows);
    for (uint32_t i = 0; i < n_workers; ++i) {
        noc_async_write(scratch + RT_REC, get_noc_addr(carg(CA_GATE_NOC + 2 * i), carg(CA_GATE_NOC + 2 * i + 1),
                                                       rows + (QR0 + r) * rec_bytes),
                        rec_bytes);
    }
    noc_async_write_barrier();
    const uint32_t sem2 = get_semaphore(2);
    for (uint32_t i = 0; i < n_workers; ++i) {
        noc_semaphore_inc(get_noc_addr(carg(CA_GATE_NOC + 2 * i), carg(CA_GATE_NOC + 2 * i + 1), sem2), 1);
    }
}
#endif
#endif  // COMPILE_FOR_NCRISC || COMPILE_FOR_BRISC

#if defined(COMPILE_FOR_TRISC)
// Router TRISC (row group RGI).  Phase 1: partial logits of this core's k slice for its rows (row q = tile 0 slots
// 4q..4q+3; weight tile = DEST tile 1).  Stock init (no DEST remap/swizzle): the SFPU slot layout of moe_router.hpp
// is the default one.
template <uint32_t RGI>
inline void router_trisc(const uint32_t r) {
    using G = RowGroup<RGI>;
    constexpr uint32_t QRN = G::QRN, QXP = G::QXP, QD = G::QD;
    constexpr uint32_t rt = INK_CTA("router_tiles");
    constexpr uint32_t rbatch = INK_CTA("router_batch");
    constexpr uint32_t gather_tiles = (routers + 8) / 8;
    unary_op_init_common(cb_rw, cb_part);
    copy_tile_to_dst_init_short(cb_rw);
    MATH(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false)));
    pack_reconfig_data_format(cb_part);
    cb_wait_front(cb_rx, 1);
    const uint32_t xb = get_tile_address(cb_rx, 0);
    const uint32_t k0 = r * rt * 8, t0 = k0 / 32;
    tile_regs_acquire();
    MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ink_router_zero<QRN, 0>, 0, VectorMode::None)));
    for (uint32_t b = 0; b < rt; b += rbatch) {
        cb_wait_front(cb_rw, rbatch);
        for (uint32_t i = 0; i < rbatch; ++i) {
            const uint32_t k = k0 + 8 * (b + i);  // x[q][k..k+7]: face (k / 16) % 2, row q, columns k % 16..
            const uint32_t xa = xb + (k / 32 - t0) * 64 * QXP + ((k / 16) % 2) * 32 * QXP + QD * 32 + (k % 16) * 2;
            MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));  // SFPU done with DEST tile 1
            copy_tile(cb_rw, i, 1);
            MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ink_router_mac<QRN, 0>, 0, VectorMode::None, xa,
                                                       32u)));
        }
        cb_pop_front(cb_rw, rbatch);
    }
    tile_regs_commit();
    cb_pop_front(cb_rx, 1);
    cb_reserve_back(cb_part, 1);
    tile_regs_wait();
    pack_tile(0, cb_part);
    tile_regs_release();
    cb_push_back(cb_part, 1);
    // Phase 2, per row: gathered partials + bias (FP32 straight to DEST) -> [sel | scores].  RTK: core r < QRN
    // finishes its row QR0 + r only, the other cores are done.
    constexpr uint32_t QF = RTK ? 1 : QRN;
    if (RTK && r >= QRN) {
        return;
    }
    cb_wait_front(cb_gather, QF * gather_tiles);
    copy_tile_to_dst_init_short_with_dt(cb_rw, cb_gather);
    for (uint32_t q = 0; q < QF; ++q) {
        tile_regs_acquire();
        for (uint32_t t = 0; t < gather_tiles; ++t) {
            copy_tile(cb_gather, q * gather_tiles + t, t);
        }
        MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::ink_router_finish_partials<routers>, 0,
                                                   VectorMode::None)));
        tile_regs_commit();
        cb_reserve_back(cb_rout, 1);
        tile_regs_wait();
        pack_tile(0, cb_rout);
        tile_regs_release();
        cb_push_back(cb_rout, 1);
    }
    cb_pop_front(cb_gather, QF * gather_tiles);
}

// Worker TRISC: gate/up units (matmul + softcapped SwiGLU) / down slots (FP32 DEST accumulation over the slots).
template <uint32_t RL>
inline void worker_trisc() {
    using C = Role<RL>;
    constexpr uint32_t kt = C::kt, cols = C::cols, blk_k = C::blk_k, cb_x = C::cb_x, cb_out = C::cb_out;
    constexpr uint32_t blocks = kt / blk_k, block_tiles = blk_k * cols;
    constexpr uint32_t transpose = K_ALIGNED;
    deepseek_compute_kernel_init();
    reconfig_data_format<SrcOrder::Reverse, true>(cb_x, cb_w);
    pack_reconfig_data_format<true>(cb_out);
    matmul_block_init(cb_x, cb_w, transpose, cols, 1, kt);
    if constexpr (RL == 0) {
        constexpr int iters = R <= 4 ? 2 : (R <= 8 ? 4 : 8);  // SFPU iterations per face: rows 0-3 / 0-7 / 0-15
        PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false)));
        cb_wait_front(cb_route, 1);
        const uint32_t units = read_tile_value(cb_route, 0, 0);
        const uint32_t plan = get_tile_address(cb_route, 0);
        uint32_t table = 0;
        cb_wait_front(cb_x, kt);
        for (uint32_t u = 0; u < units; ++u) {
            tile_regs_acquire();
            for (uint32_t b = 0; b < blocks; ++b) {
                cb_wait_front(cb_w, block_tiles);
                for (uint32_t k = 0; k < blk_k; ++k) {
                    matmul_block(cb_x, cb_w, b * blk_k + k, k * cols, 0, transpose, cols, 1, kt);
                }
                cb_pop_front(cb_w, block_tiles);
            }
            tile_regs_commit();
            if (u == 0) {
                cb_wait_front(cb_route, 2);
                table = get_tile_address(cb_route, 1);
            }
            cb_reserve_back(cb_out, 1);
            PACK(TTI_SEMWAIT(p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK),
                             p_stall::STALL_ON_ZERO));
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            PACK((ink_unit_swiglu<iters, R, WR>(plan, table, u)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_tile(0, cb_out, 0);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        if (units == 0) {
            cb_wait_front(cb_route, 2);
        }
        cb_pop_front(cb_x, kt);
        cb_pop_front(cb_route, 2);
    } else {
        // Slots in id order (row r: its experts in id order, zero rows elsewhere = bitwise the R = 1 sum).
        cb_wait_front(cb_route, 1);
        const uint32_t slots = read_tile_value(cb_route, 0, 0);
        tile_regs_acquire();
        for (uint32_t s = 0; s < slots; ++s) {
            cb_wait_front(cb_x, kt);
            for (uint32_t b = 0; b < blocks; ++b) {
                cb_wait_front(cb_w, block_tiles);
                for (uint32_t k = 0; k < blk_k; ++k) {
                    matmul_block(cb_x, cb_w, b * blk_k + k, k * cols, 0, transpose, cols, 1, kt);
                }
                cb_pop_front(cb_w, block_tiles);
            }
            cb_pop_front(cb_x, kt);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, cols);
        for (uint32_t c = 0; c < cols; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, cols);
        cb_pop_front(cb_route, 1);
    }
}
#endif  // COMPILE_FOR_TRISC

// Role RL's work on its core i (index within the role).
template <uint32_t RL>
inline void run_role(const uint32_t i) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#if defined(COMPILE_FOR_NCRISC)
    constexpr bool ncrisc = true;
#else
    constexpr bool ncrisc = false;
#endif
    if constexpr (RL < 2) {
        const uint32_t bank = i / Role<RL>::splits, part = i % Role<RL>::splits;
        if (ncrisc != (((noc1_mask >> bank) & 1u) != 0)) {
            stream_side<RL>(bank, part);
        } else {
            write_side<RL>(bank, part);
        }
    } else {
#if defined(COMPILE_FOR_NCRISC)
        router_weights(i % routers);
#else
        if constexpr (RTK) {
            if (i < routers) {
                router_records<0>(i);
            } else if (RGROUPS == 2 || i < 2 * routers) {
                router_records<1>(i - routers);
            } else if constexpr (RGROUPS == 3) {
                router_records<2>(i - 2 * routers);
            }
        } else if (i < routers) {
            router_rows<0>(i);
        } else if constexpr (RGROUPS > 1) {
            router_rows<1>(i - routers);
        }
#endif
    }
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    static_assert(DST_ACCUM_MODE, "FP32 destination accumulation required");
    if constexpr (RL < 2) {
        worker_trisc<RL>();
    } else if (i < routers) {
        router_trisc<0>(i);
    } else if constexpr (RGROUPS == 3) {
        if (i < 2 * routers) {
            router_trisc<1>(i - routers);
        } else {
            router_trisc<2>(i - 2 * routers);
        }
    } else if constexpr (RGROUPS > 1) {
        router_trisc<1>(i - routers);
    }
#endif
}

void kernel_main() {
    ink_set_read_vc();
#ifdef INK_MOE_NOOP
    return;  // dispatch-cost probe (tests/prof_moe_dispatch.py): the launch without the work
#endif
    constexpr uint32_t role = MOE_ROLE;  // this binary's role: 0 gate/up, 1 down, 2 router
    static_assert(role <= 2);
    constexpr uint32_t first = role == 0 ? 0 : (role == 1 ? n_gate : n_gate + n_down);
    run_role<role>(core_slot() - 1 - first);
}
