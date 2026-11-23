// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 decode MoE experts (one token row), streamed from the resident BFP4 bank layout (streaming_experts.py:
// interleave_gate_up_for_banks + shuffle_for_banks, the layout GroupedExperts uses for prefill).
//
// stage 0 (router tail + gate/up + routed SwiGLU):
//   NCRISC of every core reads the FP32 router logits row and the Q24 expert bias and runs the integer routing of
//   tt/kernels/sigmoid_routing.cpp (sigmoid_math.hpp: bitwise the same ids and BF16 weights), hands the weights to
//   TRISC, and one core (writer) stores the ids row (UINT16 [1,32]) and the weight row (BF16 [1,32]) for stage 1.
//   Bank b holds, per expert, 2h columns (h = I/32/banks gate tiles = intermediate tiles [b*h, b*h+h), then the
//   matching up tiles), each column K tiles contiguous.  LFM2 (I = 1536) has h = 6: 12 FP32 accumulators per bank and
//   expert, more than one half-sync DEST (4).  The work is cut into units (expert s, chunk c) of P gate/up pairs: the
//   unit's reads are gate columns [cP, cP+P) and up columns [h+cP, h+cP+P) (two contiguous runs) landing column-major
//   as one [K, 2P] block, so one transposed custom_mm leaves gate in DEST 0..P-1 and up in DEST P..2P-1; the pack
//   thread applies SiLU(g) * u * w_s in FP32 and packs P [1,32] tiles.  `splits` cores per bank take the units
//   round-robin.  Output: page (s * banks + b) of the compact activation (row-major BF16, h [1,32] tiles per page),
//   chunk c at byte offset c * P * 64.
// stage 1 (down + expert sum): bank b's `bank_cols` output columns are split over `splits` cores (cols each); each
//   core streams its column slice of every selected expert and accumulates the top_k experts in FP32 DEST (selection
//   order), then writes its cols output tiles (row 0 real, rows 1-31 zero) into the [1,1,32,H] BF16 tile output.
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#endif
#include "models/common/modules/moe/kernels/streaming/experts_matmul.hpp"
#if defined(COMPILE_FOR_NCRISC)
#include "models/demos/blackhole/lfm2/tt/fast/kernels/sigmoid_math.hpp"
#endif
#if defined(COMPILE_FOR_TRISC) && defined(TRISC_PACK)
#include "sfpu/ckernel_sfpu_converter.h"
namespace ckernel::sfpu {
// out = SiLU(gate) * up * weight on token row 0 (VectorMode::R: two 4-row x 8-column slots per upper face).
inline void calculate_lfm_swiglu(const uint dst_gate, const uint dst_up, const uint dst_out, const uint weight_bits) {
    constexpr uint tile_stride = 32;
    const sfpi::vFloat weight = Converter::as_float(weight_bits);
#pragma GCC unroll 2
    for (int d = 0; d < 2; d++) {
        const sfpi::vFloat gate = sfpi::dst_reg[dst_gate * tile_stride];
        const sfpi::vFloat up = sfpi::dst_reg[dst_up * tile_stride];
        sfpi::dst_reg[dst_out * tile_stride] = gate * _sfpu_sigmoid_<true>(gate) * up * weight;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_x = 0, cb_w = 1, cb_index = 2, cb_scratch = 3, cb_out = 4, cb_route = 5;

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
// Transaction-id ring feeding `cb` with fixed-size blocks (batched_experts_kernel.cpp): the caller issues one block's
// reads at `write` tagged `trid`, then calls next(); up to `ahead` blocks stay in flight, each pushed once it lands.
template <uint32_t cb, uint32_t buffers, uint32_t ahead, uint32_t block_tiles, uint32_t block_bytes>
struct TridRing {
    static_assert(ahead >= 1 && ahead + 1 < buffers && buffers <= 15);
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
        wait = wait == buffers ? 1 : wait + 1;
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
        trid = trid == buffers ? 1 : trid + 1;
    }
    void finish() {
        while (pending) {
            push_oldest();
        }
    }
};
#endif

void kernel_main() {
    constexpr uint32_t stage = MOE_STAGE;  // define: the stages use different accessor argument lists
    constexpr uint32_t kt = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t bank_cols = get_named_compile_time_arg_val("bank_cols");  // columns per bank and expert
    constexpr uint32_t cols = get_named_compile_time_arg_val("core_cols");       // DEST tiles per unit / core
    constexpr uint32_t splits = get_named_compile_time_arg_val("splits");        // cores per bank
    constexpr uint32_t banks = get_named_compile_time_arg_val("banks");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t half = get_named_compile_time_arg_val("inter_tiles") / banks;  // activated tiles per bank
    constexpr uint32_t act_page_bytes = half * 64;
    constexpr bool fused = stage == 0;
    constexpr uint32_t pairs = cols / 2;  // stage 0: gate/up pairs per unit
    constexpr uint32_t chunks = fused ? half / pairs : 1;
    constexpr uint32_t units = fused ? top_k * chunks : 0;
    static_assert(cols >= 1 && cols <= 4, "accumulators must fit four FP32 DEST slots");
    static_assert(!fused || (cols % 2 == 0 && bank_cols == 2 * half && half % pairs == 0));
    static_assert(fused || cols * splits == bank_cols);
    static_assert(top_k <= 31);
    constexpr bool transposed = cols > 1;  // column-major weight blocks: one SrcB unpack serves all columns

#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t weight_addr = get_named_compile_time_arg_val("weight_addr");
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("weight_page_bytes");
    constexpr uint32_t pages = get_named_compile_time_arg_val("weight_pages");  // per contiguous run
    constexpr uint32_t run_bytes = get_named_compile_time_arg_val("weight_run_bytes");
    constexpr uint32_t expert_bytes = get_named_compile_time_arg_val("weight_expert_bytes");
    constexpr uint32_t column_bytes = run_bytes / (fused ? pairs : cols);
    constexpr uint32_t buffers = get_named_compile_time_arg_val("weight_buffers");
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t part = get_arg_val<uint32_t>(2);
    const uint64_t base = get_noc_addr_from_bank_id<true>(bank, weight_addr);
    uint32_t expert_offset[top_k];
    {
        // Both roles route the token themselves from the logits row (no ids hand-off between the two roles).
        constexpr uint32_t num_experts = get_named_compile_time_arg_val("experts");
        constexpr uint32_t flags = get_named_compile_time_arg_val("route_flags");
        constexpr uint32_t columns = num_experts / 32;
        constexpr uint32_t e_lo = get_named_compile_time_arg_val("score_e_lo");
        constexpr uint32_t e_hi = get_named_compile_time_arg_val("score_e_hi");
        constexpr uint32_t table_bytes = 2 * (e_hi - e_lo + 1) * 128 * 2;
        constexpr auto logits_args = TensorAccessorArgs<0>();
        constexpr auto bias_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
        constexpr auto ids_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
        constexpr auto route_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
        constexpr auto table_args = TensorAccessorArgs<route_args.next_compile_time_args_offset()>();
        static_assert(num_experts % 32 == 0);
        const auto logits = TensorAccessor(logits_args, get_arg_val<uint32_t>(3), 4096);  // FP32 tiles
        const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(4), num_experts * 4);
        const auto ids_out = TensorAccessor(ids_args, get_arg_val<uint32_t>(5), 64);
        const auto route_out = TensorAccessor(route_args, get_arg_val<uint32_t>(6), 64);
        const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(7), table_bytes);
        const bool writer = get_arg_val<uint32_t>(8) != 0;
        // Scratch: logits row 0 in expert order (column c: face 0 row 0, face 1 row 0), bias, ids row, score table.
        const uint32_t scratch = get_write_ptr(cb_index);
        const uint32_t bias_addr = scratch + columns * 128;
        const uint32_t ids_addr = bias_addr + num_experts * 4;
        const uint32_t table_addr = ids_addr + 64;
        noc_async_read(get_noc_addr(0, table), table_addr, table_bytes);
        noc_async_read(get_noc_addr(0, bias), bias_addr, num_experts * 4);
        for (uint32_t c = 0; c < columns; ++c) {
            const uint64_t src = get_noc_addr(c, logits);
            noc_async_read(src, scratch + c * 128, 64);
            noc_async_read(src + 1024, scratch + c * 128 + 64, 64);
        }
        noc_async_read_barrier();
        uint32_t sel[top_k];
        uint16_t wts[top_k];
        {
            DeviceZoneScopedN("MOE-ROUTE");
            lfm_route<num_experts, top_k, flags, e_lo, e_hi>(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch),
                reinterpret_cast<volatile tt_l1_ptr int32_t*>(bias_addr),
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(table_addr), sel, wts);
        }
        for (uint32_t k = 0; k < top_k; ++k) {
            expert_offset[k] = sel[k] * expert_bytes;
        }
        // Weight row for TRISC (gate/up role), ids + weight rows stored by the writer core (tests / debugging).
        cb_reserve_back(cb_route, 1);
        const uint32_t route_addr = get_write_ptr(cb_route);
        auto* route_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(route_addr);
        auto* id_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_addr);
        for (uint32_t i = 0; i < 16; ++i) {
            const uint32_t a = 2 * i, b = 2 * i + 1;
            route_words[i] = (a < top_k ? wts[a] : 0u) | ((b < top_k ? static_cast<uint32_t>(wts[b]) : 0u) << 16);
            id_words[i] = (a < top_k ? sel[a] : 0u) | ((b < top_k ? sel[b] : 0u) << 16);
        }
        cb_push_back(cb_route, 1);
        if (writer) {
            noc_async_write(ids_addr, get_noc_addr(0, ids_out), 64);
            noc_async_write(route_addr, get_noc_addr(0, route_out), 64);
        }
    }
    TridRing<cb_w, buffers, buffers - 2, kt * cols, (fused ? 2 : 1) * run_bytes> ring;
    ring.start();
    noc_async_read_one_packet_set_state<true>(base, page_bytes, vc);
    if constexpr (fused) {
        for (uint32_t u = part; u < units; u += splits) {
            const uint32_t s = u / chunks, c = u % chunks;
            uint32_t dst = ring.write;
            noc_async_read_set_trid(ring.trid);
            for (uint32_t run = 0; run < 2; ++run) {  // gate pair, then up pair
                uint32_t src = expert_offset[s] + (run * half + c * pairs) * column_bytes;
                for (uint32_t p = 0; p < pages; ++p) {
                    noc_async_read_one_packet_with_state_with_trid(static_cast<uint32_t>(base), src, dst, ring.trid);
                    src += page_bytes;
                    dst += page_bytes;
                }
            }
            ring.next();
        }
    } else {
        for (uint32_t s = 0; s < top_k; ++s) {
            uint32_t src = expert_offset[s] + part * run_bytes;
            uint32_t dst = ring.write;
            noc_async_read_set_trid(ring.trid);
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read_one_packet_with_state_with_trid(static_cast<uint32_t>(base), src, dst, ring.trid);
                src += page_bytes;
                dst += page_bytes;
            }
            ring.next();
        }
    }
    ring.finish();
    noc_async_read_set_trid(0);
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_BRISC)
    const uint32_t scratch = get_write_ptr(cb_scratch);
#if MOE_STAGE == 0
        // args: x, act out, bank, part, number of down cores, their NoC (x, y) pairs
        constexpr auto x_args = TensorAccessorArgs<0>();
        constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
        const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), 2048);
        const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(1), act_page_bytes);
        const uint32_t bank = get_arg_val<uint32_t>(2);
        const uint32_t part = get_arg_val<uint32_t>(3);
        const uint32_t down_cores = get_arg_val<uint32_t>(4);
        // Row 0 of each x tile -> [1,32] input tile.  Face-row pairs land in 64 B aligned scratch first (DRAM reads
        // keep source/destination alignment), then local copies place the two 32 B halves.
        for (uint32_t t = 0; t < kt; ++t) {
            const uint64_t src = get_noc_addr(t, x);
            noc_async_read(src, scratch + t * 128, 64);
            noc_async_read(src + 512, scratch + t * 128 + 64, 64);
        }
        noc_async_read_barrier();
        cb_reserve_back(cb_x, kt);
        const uint32_t input = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < kt; ++t) {
            const uint64_t dst = get_noc_addr(input + t * 64);
            noc_async_write(scratch + t * 128, dst, 32);
            noc_async_write(scratch + t * 128 + 64, dst + 32, 32);
        }
        noc_async_write_barrier();
        cb_push_back(cb_x, kt);
        // Each finished unit: its activation tiles land in the expert's page, then every down core's semaphore for
        // that expert is incremented (the down cores wait for banks * chunks units per expert).
        for (uint32_t u = part; u < units; u += splits) {
            const uint32_t s = u / chunks, c = u % chunks;
            cb_wait_front(cb_out, pairs);
            noc_async_write(get_read_ptr(cb_out), get_noc_addr(s * banks + bank, out) + c * pairs * 64, pairs * 64);
            noc_async_write_barrier();
            cb_pop_front(cb_out, pairs);
            const uint32_t sem = get_semaphore(s);
            for (uint32_t d = 0; d < down_cores; ++d) {
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(5 + 2 * d), get_arg_val<uint32_t>(6 + 2 * d), sem), 1);
            }
        }
#else
        // args: act, out, bank, part
        constexpr uint32_t units_per_expert = get_named_compile_time_arg_val("units_per_expert");
        constexpr auto x_args = TensorAccessorArgs<0>();
        constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
        const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), act_page_bytes);
        const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(1), 2048);
        const uint32_t bank = get_arg_val<uint32_t>(2);
        const uint32_t part = get_arg_val<uint32_t>(3);
        // Zeroed output staging tiles: rows 1-31 of the output stay zero.
        for (uint32_t off = 0; off < cols * 2048; off += MEM_ZEROS_SIZE) {
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), scratch + off, MEM_ZEROS_SIZE);
        }
        // Every expert's full intermediate row, gathered from its per-bank pages once all its units are written.
        for (uint32_t s = 0; s < top_k; ++s) {
            auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(s));
            noc_semaphore_wait(sem, units_per_expert);
            noc_semaphore_set(sem, 0);
            cb_reserve_back(cb_x, kt);
            const uint32_t input = get_write_ptr(cb_x);
            for (uint32_t b = 0; b < banks; ++b) {
                noc_async_read(get_noc_addr(s * banks + b, x), input + b * act_page_bytes, act_page_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, kt);
        }
        cb_wait_front(cb_out, cols);
        const uint32_t results = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < cols; ++c) {
            const uint64_t dst = get_noc_addr(scratch + c * 2048);
            noc_async_write(results + c * 64, dst, 32);             // face 0 row 0
            noc_async_write(results + c * 64 + 32, dst + 512, 32);  // face 1 row 0
        }
        noc_async_write_barrier();
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write(scratch + c * 2048, get_noc_addr(bank * bank_cols + part * cols + c, out), 2048);
        }
        cb_pop_front(cb_out, cols);
#endif
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    static_assert(DST_ACCUM_MODE, "FP32 destination accumulation required");
    const uint32_t part = get_arg_val<uint32_t>(0);
    deepseek_compute_kernel_init();
    reconfig_data_format<SrcOrder::Reverse, true>(cb_x, cb_w);
    pack_reconfig_data_format<true>(cb_out);
    custom_mm_block_init_short<false, true, false>(cb_x, cb_w, cb_out, cols);
    if constexpr (fused) {
        PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false)));
        cb_wait_front(cb_x, kt);
        cb_wait_front(cb_route, 1);
        for (uint32_t u = part; u < units; u += splits) {
            const uint32_t s = u / chunks;
            const uint32_t pair = read_tile_value(cb_route, 0, s / 2);
            const uint32_t weight_bits = s % 2 ? pair & 0xffff0000u : pair << 16;
            cb_wait_front(cb_w, kt * cols);
            cb_reserve_back(cb_out, pairs);
            tile_regs_acquire();
            custom_mm_block<true, transposed>(cb_x, cb_w, 0, 0, 0, kt, cols);
            tile_regs_commit();
            PACK(TTI_SEMWAIT(p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK),
                             p_stall::STALL_ON_ZERO));
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            for (uint32_t j = 0; j < pairs; ++j) {
                PACK((_llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::calculate_lfm_swiglu, j, pairs + j, j,
                                                            VectorMode::R, weight_bits)));
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            for (uint32_t j = 0; j < pairs; ++j) {
                pack_tile(j, cb_out, j);
            }
            tile_regs_release();
            cb_push_back(cb_out, pairs);
            cb_pop_front(cb_w, kt * cols);
        }
        cb_pop_front(cb_x, kt);
    } else {
        // DEST slot c accumulates output column c over the top_k experts (selection order).
        tile_regs_acquire();
        for (uint32_t s = 0; s < top_k; ++s) {
            cb_wait_front(cb_x, kt);
            cb_wait_front(cb_w, kt * cols);
            if (s + 1 == top_k) {
                custom_mm_block<true, transposed>(cb_x, cb_w, 0, 0, 0, kt, cols);
            } else {
                custom_mm_block<false, transposed>(cb_x, cb_w, 0, 0, 0, kt, cols);
            }
            cb_pop_front(cb_x, kt);
            cb_pop_front(cb_w, kt * cols);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, cols);
        for (uint32_t c = 0; c < cols; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, cols);
    }
    custom_mm_block_uninit<false>();
#endif
}
