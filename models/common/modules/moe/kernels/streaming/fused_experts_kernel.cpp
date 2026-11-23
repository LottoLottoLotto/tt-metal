// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Fused union-expert decode (1-8 rows, one row group): batched_experts_kernel.cpp's gate/up stage and down stage in ONE program.
// Gate/up runs on every core (the bank's `g_splits` cores take the union's experts round-robin); each core then
// signals every down core once its activation pages are written (semaphore 0). Down cores (a subset: the bank's
// first `d_splits` cores) stream their down-weight column slices right after their own gate/up weights -- the
// prefetch overlaps the gate/up tail -- and gather the activations once all gate/up cores have signalled. Both
// stages keep batched_experts_kernel.cpp's tile math and accumulation order, so outputs are bitwise those of the
// two-program path.
// FUSED_ROUTING: routing_signal.cpp runs in the same program on separate cores and counts itself in on every core
// here (semaphore 1) once the selection / weight rows have landed. Each bank's part-0 core then takes the shared
// expert (always selected, union position count - 1) FIRST: its gate/up weights stream and its matmul runs while
// routing is still in flight; routed positions s go to part (s + 1) % g_splits. Which core computes an expert does
// not change its math, so outputs stay bitwise equal.
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#endif
#include "models/common/modules/moe/kernels/streaming/experts_matmul.hpp"
#if defined(COMPILE_FOR_TRISC) && defined(TRISC_PACK)
#include "sfpu/ckernel_sfpu_converter.h"
namespace ckernel::sfpu {
template <uint32_t tile_rows>
inline void calculate_batched_swiglu(
    const uint dst_gate, const uint dst_up, const uint dst_out, const uint32_t* weights) {
    constexpr uint tile_stride = 32;
    // One SFPU slot covers four rows x eight columns. VectorMode::R invokes this
    // once per upper face; the two slots per four rows cover all sixteen columns.
    for (uint32_t block = 0; block < (tile_rows + 3) / 4; ++block) {
        const sfpi::vFloat w0 = Converter::as_float(weights[block * 4]);
        const sfpi::vFloat w1 = Converter::as_float(weights[block * 4 + 1]);
        const sfpi::vFloat w2 = Converter::as_float(weights[block * 4 + 2]);
        sfpi::vFloat weight = Converter::as_float(weights[block * 4 + 3]);
        v_if(sfpi::vConstTileId < 16) { weight = w0; }
        v_elseif(sfpi::vConstTileId < 32) { weight = w1; }
        v_elseif(sfpi::vConstTileId < 48) { weight = w2; }
        v_endif;
#pragma GCC unroll 2
        for (uint32_t d = 0; d < 2; ++d) {
            const sfpi::vFloat gate = sfpi::dst_reg[dst_gate * tile_stride];
            const sfpi::vFloat up = sfpi::dst_reg[dst_up * tile_stride];
            sfpi::dst_reg[dst_out * tile_stride] = gate * _sfpu_sigmoid_<true>(gate) * up * weight;
            sfpi::dst_reg++;
        }
    }
}
}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_x = 0, cb_w = 1, cb_index = 2, cb_scratch = 3, cb_out = 4, cb_route = 5;
constexpr uint32_t cb_dx = 6, cb_dw = 7, cb_dscratch = 8, cb_dout = 9;
constexpr uint32_t no_part = 0xffffffffu;
#ifdef FUSED_ROUTING
constexpr bool shared_first = true;
#else
constexpr bool shared_first = false;
#endif
// Union position of a core's i-th gate/up slot index i (i = part, part + splits, ...).
inline uint32_t gate_position(uint32_t i, uint32_t count) { return shared_first ? (i == 0 ? count - 1 : i - 1) : i; }

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
// Transaction-id ring feeding `cb` with fixed-size blocks. The caller issues one block's
// reads at `write` tagged `trid`, then calls next(); up to `ahead` blocks stay in flight
// and each is pushed once its reads land.
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
    constexpr uint32_t gkt = get_named_compile_time_arg_val("g_k_tiles");
    constexpr uint32_t g_n = get_named_compile_time_arg_val("g_per_core_n");  // gate/up bank columns
    constexpr uint32_t g_splits = get_named_compile_time_arg_val("g_splits");
    constexpr uint32_t dkt = get_named_compile_time_arg_val("d_k_tiles");
    constexpr uint32_t d_n = get_named_compile_time_arg_val("d_per_core_n");  // down bank columns
    constexpr uint32_t d_cols = get_named_compile_time_arg_val("d_cols");     // down columns per core
    constexpr uint32_t d_splits = get_named_compile_time_arg_val("d_splits");
    constexpr uint32_t g_cores = get_named_compile_time_arg_val("g_cores");
    constexpr uint32_t banks = get_named_compile_time_arg_val("banks");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t tile_rows = get_named_compile_time_arg_val("tile_rows");
    constexpr uint32_t groups = get_named_compile_time_arg_val("row_groups");
    constexpr uint32_t selected = get_named_compile_time_arg_val("selected");
    constexpr uint32_t width = get_named_compile_time_arg_val("metadata_width");
    constexpr uint32_t row_bytes = tile_rows * 64;
    constexpr uint32_t half = get_named_compile_time_arg_val("inter_tiles") / banks;
    constexpr uint32_t act_page_bytes = half * row_bytes;
    static_assert(tile_rows == 1 || tile_rows == 2 || tile_rows == 4 || tile_rows == 8);
    static_assert(rows == groups * tile_rows && groups <= 2);
    static_assert(width > selected && width % 32 == 0);
    static_assert(g_n == 2 * half && g_n <= 4 && d_cols * d_splits == d_n && d_cols * groups <= 4);
    constexpr bool g_transposed = g_n > 1;
    constexpr bool d_transposed = d_cols > 1;

#if defined(COMPILE_FOR_NCRISC)
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t g_vc = get_arg_val<uint32_t>(1);
    const uint32_t g_part = get_arg_val<uint32_t>(2);
    const uint32_t d_vc = get_arg_val<uint32_t>(3);
    const uint32_t d_part = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint16_t* ids = nullptr;
    uint32_t count = 0;
    {
        // Gate/up: this core's experts, one whole bank block each.
        constexpr uint32_t page_bytes = get_named_compile_time_arg_val("g_page_bytes");
        constexpr uint32_t pages = get_named_compile_time_arg_val("g_pages");
        constexpr uint32_t block_bytes = get_named_compile_time_arg_val("g_block_bytes");
        constexpr uint32_t expert_bytes = get_named_compile_time_arg_val("g_expert_bytes");
        const uint64_t base = get_noc_addr_from_bank_id<true>(bank, get_named_compile_time_arg_val("g_weight_addr"));
        TridRing<cb_w, 3, 1, gkt * g_n, block_bytes> ring;
        ring.start();
        noc_async_read_one_packet_set_state<true>(base, page_bytes, g_vc);
        auto stream = [&](uint32_t expert) {
            uint32_t src = expert * expert_bytes;
            uint32_t dst = ring.write;
            noc_async_read_set_trid(ring.trid);
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read_one_packet_with_state_with_trid(static_cast<uint32_t>(base), src, dst, ring.trid);
                src += page_bytes;
                dst += page_bytes;
            }
            ring.next();
        };
        uint32_t first = g_part;
        if constexpr (shared_first) {
            if (g_part == 0) {
                stream(get_named_compile_time_arg_val("shared_slot"));
                first = g_splits;
            }
        }
        cb_wait_front(cb_index, 1);
        ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_index));
        count = ids[width - 1];
        for (uint32_t i = first; i < count; i += g_splits) {
            stream(static_cast<uint32_t>(ids[gate_position(i, count)]));
        }
        ring.finish();
        noc_async_read_set_trid(0);
    }
    if (d_part != no_part) {
        // Down: this core's column slice of every expert, prefetched while gate/up finishes.
        constexpr uint32_t page_bytes = get_named_compile_time_arg_val("d_page_bytes");
        constexpr uint32_t pages = get_named_compile_time_arg_val("d_pages");
        constexpr uint32_t block_bytes = get_named_compile_time_arg_val("d_block_bytes");
        constexpr uint32_t expert_bytes = get_named_compile_time_arg_val("d_expert_bytes");
        const uint64_t base = get_noc_addr_from_bank_id<true>(bank, get_named_compile_time_arg_val("d_weight_addr"));
        TridRing<cb_dw, 8, 6, dkt * d_cols, block_bytes> ring;
        ring.start();
        noc_async_read_one_packet_set_state<true>(base, page_bytes, d_vc);
        const uint32_t column_offset = d_part * block_bytes;
        for (uint32_t s = 0; s < count; ++s) {
            uint32_t src = static_cast<uint32_t>(ids[s]) * expert_bytes + column_offset;
            uint32_t dst = ring.write;
            noc_async_read_set_trid(ring.trid);
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read_one_packet_with_state_with_trid(static_cast<uint32_t>(base), src, dst, ring.trid);
                src += page_bytes;
                dst += page_bytes;
            }
            ring.next();
        }
        ring.finish();
        noc_async_read_set_trid(0);
    }
#elif defined(COMPILE_FOR_BRISC)
    const uint32_t bank = get_arg_val<uint32_t>(5);
    const uint32_t g_part = get_arg_val<uint32_t>(6);
    const uint32_t d_part = get_arg_val<uint32_t>(7);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto ids_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto route_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto act_args = TensorAccessorArgs<route_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), 2048);
    const auto ids = TensorAccessor(ids_args, get_arg_val<uint32_t>(1), width * 2);
    const auto route = TensorAccessor(route_args, get_arg_val<uint32_t>(2), width * 2);
    const auto act = TensorAccessor(act_args, get_arg_val<uint32_t>(3), act_page_bytes);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(4), 2048);
    // Selection first (the weight streams wait for it); with FUSED_ROUTING only once routing has signalled, after
    // the token rows (independent of routing) are staged.
    auto read_selection = [&]() {
        cb_reserve_back(cb_index, 1);
        const uint32_t meta = get_write_ptr(cb_index);
        noc_async_read(get_noc_addr(0, ids), meta, width * 2);
        noc_async_read_barrier();
        const uint32_t n = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta)[width - 1];
        cb_push_back(cb_index, 1);
        return n;
    };
#ifndef FUSED_ROUTING
    const uint32_t count = read_selection();
#endif
    const uint32_t scratch = get_write_ptr(cb_scratch);
    constexpr uint32_t read_face_bytes = (rows < 2 ? 2 : rows) * 32;
    for (uint32_t t = 0; t < gkt; ++t) {
        const uint64_t src = get_noc_addr(t, x);
        noc_async_read(src, scratch + t * 2 * read_face_bytes, read_face_bytes);
        noc_async_read(src + 512, scratch + (t * 2 + 1) * read_face_bytes, read_face_bytes);
    }
    noc_async_read_barrier();
    cb_reserve_back(cb_x, groups * gkt);
    {
        const uint32_t input = get_write_ptr(cb_x);
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t t = 0; t < gkt; ++t) {
                const uint32_t src = scratch + t * 2 * read_face_bytes + g * tile_rows * 32;
                const uint64_t dst = get_noc_addr(input + (g * gkt + t) * row_bytes);
                noc_async_write(src, dst, row_bytes / 2);
                noc_async_write(src + read_face_bytes, dst + row_bytes / 2, row_bytes / 2);
            }
        }
        noc_async_write_barrier();
    }
    cb_push_back(cb_x, groups * gkt);
#ifdef FUSED_ROUTING
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 1);
    const uint32_t count = read_selection();
#endif
    cb_reserve_back(cb_route, 1);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(get_noc_addr(r, route), get_write_ptr(cb_route) + r * width * 2, width * 2);
    }
    noc_async_read_barrier();
    cb_push_back(cb_route, 1);
    {
        for (uint32_t i = g_part; i < count; i += g_splits) {
            const uint32_t s = gate_position(i, count);
            for (uint32_t g = 0; g < groups; ++g) {
                cb_wait_front(cb_out, half);
                noc_async_write(get_read_ptr(cb_out), get_noc_addr((s * groups + g) * banks + bank, act), act_page_bytes);
                noc_async_writes_flushed();
                cb_pop_front(cb_out, half);
            }
        }
    }
    // Every activation page of this core has landed: count this core in on every down core.
    noc_async_write_barrier();
    const uint32_t down_cores = get_arg_val<uint32_t>(8);
    for (uint32_t i = 0; i < down_cores; ++i) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(9 + 2 * i), get_arg_val<uint32_t>(10 + 2 * i),
                                       get_semaphore(0)), 1);
    }
    noc_async_atomic_barrier();
    if (d_part != no_part) {
        const uint32_t scratch = get_write_ptr(cb_dscratch);
        // Zeroed staging tiles: output rows beyond the bucket stay zero.
        for (uint32_t off = 0; off < d_cols * 2048; off += MEM_ZEROS_SIZE) {
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), scratch + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), g_cores);
        // Every expert's full intermediate input, gathered from its per-bank pages.
        TridRing<cb_dx, 4, 2, groups * dkt, groups * dkt * row_bytes> ring;
        ring.start();
        for (uint32_t s = 0; s < count; ++s) {
            noc_async_read_set_trid(ring.trid);
            for (uint32_t g = 0; g < groups; ++g) {
                for (uint32_t b = 0; b < banks; ++b) {
                    noc_async_read(get_noc_addr((s * groups + g) * banks + b, act),
                        ring.write + (g * dkt + b * half) * row_bytes, act_page_bytes);
                }
            }
            ring.next();
        }
        ring.finish();
        noc_async_read_set_trid(0);
        cb_wait_front(cb_dout, d_cols * groups);
        const uint32_t results = get_read_ptr(cb_dout);
        for (uint32_t c = 0; c < d_cols; ++c) {
            for (uint32_t g = 0; g < groups; ++g) {
                const uint32_t src = results + (c * groups + g) * row_bytes;
                const uint64_t dst = get_noc_addr(scratch + c * 2048 + g * tile_rows * 32);
                noc_async_write(src, dst, row_bytes / 2);
                noc_async_write(src + row_bytes / 2, dst + 512, row_bytes / 2);
            }
        }
        noc_async_write_barrier();
        for (uint32_t c = 0; c < d_cols; ++c) {
            noc_async_write(scratch + c * 2048, get_noc_addr(bank * d_n + d_part * d_cols + c, out), 2048);
        }
        cb_pop_front(cb_dout, d_cols * groups);
    }
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    static_assert(DST_ACCUM_MODE, "batched experts require FP32 destination accumulation");
    const uint32_t g_part = get_arg_val<uint32_t>(0);
    const uint32_t d_part = get_arg_val<uint32_t>(1);
    deepseek_compute_kernel_init();
    reconfig_data_format<SrcOrder::Reverse, true>(cb_x, cb_w);
    pack_reconfig_data_format<true>(cb_out);
    custom_mm_block_init_short<false, true, false>(cb_x, cb_w, cb_out, g_n);
    PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false)));
    uint32_t count = 0;
    auto read_metadata = [&]() {
        cb_wait_front(cb_index, 1);
        count = read_tile_value(cb_index, 0, (width - 1) / 2) >> 16;
        cb_wait_front(cb_route, 1);
    };
    // One expert block for row group g: matmul into DEST, then SwiGLU x routing weight in the packer's SFPU.
    // The routing metadata is read after the matmul (the shared expert's matmul may precede routing).
    auto expert_group = [&](uint32_t i, uint32_t g) {
        cb_reserve_back(cb_out, half);
        tile_regs_acquire();
        custom_mm_block<true, g_transposed>(cb_x, cb_w, g * gkt, 0, 0, gkt, g_n);
        tile_regs_commit();
        if (count == 0) {
            read_metadata();
        }
        const uint32_t s = gate_position(i, count);
        uint32_t weight_bits[tile_rows < 4 ? 4 : tile_rows] = {};
        for (uint32_t r = 0; r < tile_rows; ++r) {
            const uint32_t pair = read_tile_value(cb_route, 0, (g * tile_rows + r) * width / 2 + s / 2);
            weight_bits[r] = s % 2 ? pair & 0xffff0000u : pair << 16;
        }
        PACK(TTI_SEMWAIT(p_stall::STALL_TDMA | p_stall::STALL_CFG,
            semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t j = 0; j < half; ++j) {
            PACK((_llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::calculate_batched_swiglu<tile_rows>,
                j, half + j, j, VectorMode::R, weight_bits)));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        for (uint32_t j = 0; j < half; ++j) {
            pack_tile(j, cb_out, j);
        }
        tile_regs_release();
        cb_push_back(cb_out, half);
    };
    cb_wait_front(cb_x, groups * gkt);
    uint32_t first = g_part;
    if (shared_first && g_part == 0) {
        cb_wait_front(cb_w, gkt * g_n);
        for (uint32_t g = 0; g < groups; ++g) {
            expert_group(0, g);
        }
        cb_pop_front(cb_w, gkt * g_n);
        first = g_splits;
    } else {
        read_metadata();
    }
    for (uint32_t i = first; i < count; i += g_splits) {
        cb_wait_front(cb_w, gkt * g_n);
        for (uint32_t g = 0; g < groups; ++g) {
            expert_group(i, g);
        }
        cb_pop_front(cb_w, gkt * g_n);
    }
    cb_pop_front(cb_x, groups * gkt);
    custom_mm_block_uninit<false>();
    if (d_part != no_part) {
        reconfig_data_format<SrcOrder::Reverse, true>(cb_dx, cb_dw);
        pack_reconfig_data_format<true>(cb_dout);
        custom_mm_block_init_short<false, true, false>(cb_dx, cb_dw, cb_dout, d_cols);
        // DEST slot g * cols + c accumulates column c of row group g over every expert,
        // in union order; the other rows' experts contribute exact zeros.
        tile_regs_acquire();
        for (uint32_t s = 0; s < count; ++s) {
            cb_wait_front(cb_dx, groups * dkt);
            cb_wait_front(cb_dw, dkt * d_cols);
            for (uint32_t g = 0; g < groups; ++g) {
                if (s + 1 == count) {
                    custom_mm_block<true, d_transposed>(cb_dx, cb_dw, g * dkt, 0, g * d_cols, dkt, d_cols);
                } else {
                    custom_mm_block<false, d_transposed>(cb_dx, cb_dw, g * dkt, 0, g * d_cols, dkt, d_cols);
                }
            }
            cb_pop_front(cb_dx, groups * dkt);
            cb_pop_front(cb_dw, dkt * d_cols);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_dout, d_cols * groups);
        for (uint32_t c = 0; c < d_cols; ++c) {
            for (uint32_t g = 0; g < groups; ++g) {
                pack_tile(g * d_cols + c, cb_dout);
            }
        }
        tile_regs_release();
        cb_push_back(cb_dout, d_cols * groups);
        custom_mm_block_uninit<false>();
    }
#endif
}
