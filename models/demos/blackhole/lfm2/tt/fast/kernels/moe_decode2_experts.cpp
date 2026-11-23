// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 decode MoE v2 experts (one token row): moe_decode_experts.cpp with the routing delivered by the router role
// (moe_decode2_router.cpp, same program) instead of being evaluated on every worker.
//
// both stages: NCRISC waits for the route semaphore (set by a router core after it wrote the final ids and BF16
// weights into this core's landing buffer).
// stage 0 (gate/up + routed SwiGLU):
//   NCRISC hands the weights to TRISC, and one core (writer) stores the ids row (UINT16 [1,32]) and the weight row.
//   Bank b holds, per expert, 2h columns (h = I/32/banks gate tiles = intermediate tiles [b*h, b*h+h), then the
//   matching up tiles), each column K tiles contiguous.  LFM2 (I = 1536) has h = 6: 12 FP32 accumulators per bank and
//   expert, more than one half-sync DEST (4).  The work is cut into units (expert s, chunk c) of P gate/up pairs: the
//   unit's reads are gate columns [cP, cP+P) and up columns [h+cP, h+cP+P) (two contiguous runs) landing column-major
//   as one [K, 2P] block, so one transposed custom_mm leaves gate in DEST 0..P-1 and up in DEST P..2P-1; the pack
//   thread applies SiLU(g) * u * w_s in FP32 and packs P [1,32] tiles.  `splits` cores per bank take the units
//   round-robin.  Each unit's P tiles are written straight into every down worker's cb_act at
//   [(s * banks + b) * h + c * P] (no intermediate tensor), then the down worker's semaphore of bank b gets
//   1 << (8 s) (byte lane s = expert s's finished chunks of bank b).
// stage 1 (down + expert sum): bank b's `bank_cols` output columns are split over `splits` cores (cols each); each
//   core streams its column slice of every selected expert (in bank-piece order, see the NCRISC loop) and
//   accumulates in FP32 DEST over the experts (selection order) and, within an expert, over its K tiles one bank
//   piece at a time as soon as that piece's activations are in (bitwise one call per expert).  Its output tiles are
//   zero-filled during the routing; at the end only row 0 is written.
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#endif
#include "models/common/modules/moe/kernels/streaming/experts_matmul.hpp"
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

// cb_act: the down workers' activation rows [expert][bank][half tiles] ([1,32] BF16 tiles), written directly by the
// gate/up workers (one CB over all workers, so the same L1 address everywhere).
constexpr uint32_t cb_x = 0, cb_w = 1, cb_index = 2, cb_scratch = 3, cb_out = 4, cb_route = 5, cb_act = 6,
                   cb_land = 7;

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
    constexpr bool fused = stage == 0;
    constexpr uint32_t pairs = cols / 2;  // stage 0: gate/up pairs per unit
    constexpr uint32_t chunks = fused ? half / pairs : 1;
    constexpr uint32_t units = fused ? top_k * chunks : 0;
    constexpr uint32_t pieces = get_named_compile_time_arg_val("down_pieces");  // stage 1: bank pieces per block
    static_assert(banks % pieces == 0);
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
    constexpr uint32_t tile_bytes = column_bytes / kt;
    constexpr uint32_t buffers = get_named_compile_time_arg_val("weight_buffers");
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t part = get_arg_val<uint32_t>(2);
    const uint64_t base = get_noc_addr_from_bank_id<true>(bank, weight_addr);
    uint32_t expert_offset[top_k];
    {
        // The router cores deliver the final routing (ids, BF16 weights) into the landing buffer, then the flag.
        constexpr uint32_t route_sem = get_named_compile_time_arg_val("route_sem");
        constexpr auto ids_args = TensorAccessorArgs<0>();
        constexpr auto route_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
        const auto ids_out = TensorAccessor(ids_args, get_arg_val<uint32_t>(3), 64);
        const auto route_out = TensorAccessor(route_args, get_arg_val<uint32_t>(4), 64);
        const bool writer = get_arg_val<uint32_t>(5) != 0;
        const uint32_t ids_addr = get_write_ptr(cb_index);
        uint32_t sel[top_k];
        uint16_t wts[top_k];
        {
            DeviceZoneScopedN("MOE-WAIT");
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(route_sem)), 1);
        }
        auto* land = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_land));
        for (uint32_t k = 0; k < top_k; ++k) {
            sel[k] = land[k];
            wts[k] = static_cast<uint16_t>(land[top_k + k]);
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
    // ring blocks: a gate/up unit, or `pieces` bank pieces of a down expert
    constexpr uint32_t block_tiles = fused ? kt * cols : pieces * half * cols;
    TridRing<cb_w, buffers, buffers - 2, block_tiles, fused ? 2 * run_bytes : block_tiles * tile_bytes> ring;
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
        // L1 blocks ordered [bank piece b][column c][half K tiles] (the K slice of bank b's activations), `pieces`
        // pieces per block: TRISC consumes them piece by piece as the pieces' activations arrive.
        static_assert(fused || page_bytes == half * tile_bytes);  // (not a template: checked in both stages)
        for (uint32_t s = 0; s < top_k; ++s) {
            const uint32_t src = expert_offset[s] + part * run_bytes;
            for (uint32_t b0 = 0; b0 < banks; b0 += pieces) {
                const uint32_t dst = ring.write;
                noc_async_read_set_trid(ring.trid);
                for (uint32_t b = 0; b < pieces; ++b) {
                    for (uint32_t c = 0; c < cols; ++c) {
                        noc_async_read_one_packet_with_state_with_trid(
                            static_cast<uint32_t>(base), src + (c * kt + (b0 + b) * half) * tile_bytes,
                            dst + (b * cols + c) * half * tile_bytes, ring.trid);
                    }
                }
                ring.next();
            }
        }
    }
    ring.finish();
    noc_async_read_set_trid(0);
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_BRISC)
    const uint32_t scratch = get_write_ptr(cb_scratch);
#if MOE_STAGE == 0
        // args: x, bank, part, number of down cores, their NoC (x, y) pairs
        constexpr uint32_t bank_sem = get_named_compile_time_arg_val("bank_sem");
        constexpr auto x_args = TensorAccessorArgs<0>();
        const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), 2048);
        const uint32_t bank = get_arg_val<uint32_t>(1);
        const uint32_t part = get_arg_val<uint32_t>(2);
        const uint32_t down_cores = get_arg_val<uint32_t>(3);
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
        // Each finished unit: its activation tiles go straight into every down worker's cb_act, then (after the
        // writes left this core: same destination and VC, so they land first) each down worker's semaphore of this
        // bank gets 1 << (8 s): byte lane s counts expert s's finished chunks of the bank.
        const uint32_t act = get_write_ptr(cb_act);
        const uint32_t sem = get_semaphore(bank_sem + bank);
        for (uint32_t u = part; u < units; u += splits) {
            const uint32_t s = u / chunks, c = u % chunks;
            cb_wait_front(cb_out, pairs);
            const uint32_t slot = act + ((s * banks + bank) * half + c * pairs) * 64;
            for (uint32_t d = 0; d < down_cores; ++d) {
                const uint32_t dx = get_arg_val<uint32_t>(4 + 2 * d), dy = get_arg_val<uint32_t>(5 + 2 * d);
                noc_async_write(get_read_ptr(cb_out), get_noc_addr(dx, dy, slot), pairs * 64);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, pairs);
            for (uint32_t d = 0; d < down_cores; ++d) {
                const uint32_t dx = get_arg_val<uint32_t>(4 + 2 * d), dy = get_arg_val<uint32_t>(5 + 2 * d);
                noc_semaphore_inc(get_noc_addr(dx, dy, sem), 1u << (8 * s));
            }
        }
#else
        // args: out, bank, part
        constexpr uint32_t bank_sem = get_named_compile_time_arg_val("bank_sem");
        constexpr uint32_t gate_chunks = get_named_compile_time_arg_val("gate_chunks");  // units per (expert, bank)
        static_assert(gate_chunks < 256 && top_k <= 4);
        constexpr auto out_args = TensorAccessorArgs<0>();
        const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0), 2048);
        const uint32_t bank = get_arg_val<uint32_t>(1);
        const uint32_t part = get_arg_val<uint32_t>(2);
        // Output tiles zero-filled while the experts stream (rows 1-31 stay zero); at the end only row 0 is written.
        for (uint32_t off = 0; off < cols * 2048; off += MEM_ZEROS_SIZE) {
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), scratch + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write(scratch + c * 2048, get_noc_addr(bank * bank_cols + part * cols + c, out), 2048);
        }
        noc_async_write_barrier();  // before the staging rows are overwritten (long before: during the routing)
        // The activation pieces arrive in cb_act from the gate/up workers; publish them to TRISC in K order (expert,
        // then bank) as each bank's chunks of the expert are all in.
        cb_reserve_back(cb_act, top_k * kt);
        for (uint32_t s = 0; s < top_k; ++s) {
            for (uint32_t b = 0; b < banks; ++b) {
                auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(bank_sem + b));
                do {
                    invalidate_l1_cache();
                } while (((*sem >> (8 * s)) & 0xff) < gate_chunks);
                cb_push_back(cb_act, half);
            }
        }
        {
            DeviceZoneScopedN("D-OUT");
            constexpr bool out_l1 = get_named_compile_time_arg_val("out_l1") == 1;
            cb_wait_front(cb_out, cols);
            const uint32_t results = get_read_ptr(cb_out);  // [1,32] tiles: face 0 row 0, face 1 row 0 (32 B each)
            if constexpr (out_l1) {
                // L1 output: the 32 B rows straight into the zero-filled tiles
                for (uint32_t c = 0; c < cols; ++c) {
                    const uint64_t dst = get_noc_addr(bank * bank_cols + part * cols + c, out);
                    noc_async_write(results + c * 64, dst, 32);
                    noc_async_write(results + c * 64 + 32, dst + 512, 32);
                }
            } else {
                // DRAM output: rows into the zeroed staging tiles, then 64 B (rows 0-1) per face
                for (uint32_t c = 0; c < cols; ++c) {
                    const uint64_t dst = get_noc_addr(scratch + c * 2048);
                    noc_async_write(results + c * 64, dst, 32);
                    noc_async_write(results + c * 64 + 32, dst + 512, 32);
                }
                noc_async_write_barrier();
                for (uint32_t c = 0; c < cols; ++c) {
                    const uint64_t dst = get_noc_addr(bank * bank_cols + part * cols + c, out);
                    noc_async_write(scratch + c * 2048, dst, 64);
                    noc_async_write(scratch + c * 2048 + 512, dst + 512, 64);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, cols);
        }
#endif
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    static_assert(DST_ACCUM_MODE, "FP32 destination accumulation required");
    const uint32_t part = get_arg_val<uint32_t>(0);
    deepseek_compute_kernel_init();
    constexpr uint32_t cb_in = fused ? cb_x : cb_act;
    reconfig_data_format<SrcOrder::Reverse, true>(cb_in, cb_w);
    pack_reconfig_data_format<true>(cb_out);
    custom_mm_block_init_short<false, true, false>(cb_in, cb_w, cb_out, cols);
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
        // DEST slot c accumulates output column c over the top_k experts (selection order) and, within an expert,
        // over its K tiles in order, one bank piece (half tiles) at a time: custom_mm splits accumulation only within
        // a tile, so the chained calls (finalize on the very last) sum exactly as one call per expert.
        static_assert(half % 2 == 0, "custom_mm kt_dim must be even");
        tile_regs_acquire();
        for (uint32_t s = 0; s < top_k; ++s) {
            for (uint32_t b0 = 0; b0 < banks; b0 += pieces) {
                cb_wait_front(cb_w, pieces * half * cols);
                for (uint32_t b = 0; b < pieces; ++b) {
                    cb_wait_front(cb_act, half);
                    if (s + 1 == top_k && b0 + b + 1 == banks) {
                        custom_mm_block<true, transposed>(cb_act, cb_w, 0, b * half * cols, 0, half, cols);
                    } else {
                        custom_mm_block<false, transposed>(cb_act, cb_w, 0, b * half * cols, 0, half, cols);
                    }
                    cb_pop_front(cb_act, half);
                }
                cb_pop_front(cb_w, pieces * half * cols);
            }
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
