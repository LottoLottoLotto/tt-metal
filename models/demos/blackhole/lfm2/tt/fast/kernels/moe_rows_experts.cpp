// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 multi-row decode MoE (moe_rows.py): the union of the B rows' selected experts streamed once per layer.
//
// stage 0 (gate/up + routed SwiGLU, same program as the router role moe_rows_router.cpp):
//   Both data-movement RISCs wait for the routing (`senders` increments of `route_sem`; the per-row records ids[4],
//   BF16 weights[4] are in the landing buffer) and form the union of the selected experts in ascending id order.
//   NCRISC streams the units of this core: unit (union slot s, chunk c) = gate column c and up column h + c of
//   expert s in this core's bank (h = I/32/banks = 6; two contiguous K runs landing as one column-major [K, 2]
//   block), units taken round-robin by the bank's `splits` cores (moe_decode2_experts.cpp's gate/up role with
//   pairs = 1).  BRISC builds the per-row weight table [rows][width] (BF16, zero where a row did not select the
//   union expert; batched_experts_kernel.cpp's layout) and the in0 tiles: `groups` row groups of [tile_rows, 32]
//   tiles (rows >= B zero).  TRISC: per unit and row group one transposed custom_mm (gate -> DEST 2g, up -> 2g + 1),
//   SiLU(g) * u * w_row on the pack thread (per-row weights by SFPU lane), one [tile_rows, 32] tile packed.  BRISC
//   writes it into the L1-interleaved activation tensor: page (s * groups + g) * banks + bank, tile c.
//   The writer core (bank 0, part 0) also stores the union (UINT16 ids, count at width - 1) and the weight table.
// stage 1 (down + expert sum, its own program): batched_experts_kernel.cpp's down role: bank b's `bank_cols` output
//   columns split over `splits` cores; each streams its column slice of every union expert (ascending ids) and
//   accumulates them in FP32 DEST per row group; a row's unselected experts add exact zeros, so each row sums its
//   own experts in ascending id order exactly as the one-row path (moe_decode2) does.
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#endif
#include "models/common/modules/moe/kernels/streaming/experts_matmul.hpp"
#if defined(COMPILE_FOR_TRISC) && defined(TRISC_PACK)
#include "sfpu/ckernel_sfpu_converter.h"
namespace ckernel::sfpu {
// out = SiLU(gate) * up * weight[row] (batched_experts_kernel.cpp): one SFPU slot covers four rows x eight
// columns; VectorMode::R calls this once per upper face.
template <uint32_t tile_rows>
inline void calculate_rows_swiglu(const uint dst_gate, const uint dst_up, const uint dst_out,
                                  const uint32_t* weights) {
    constexpr uint tile_stride = 32;
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

constexpr uint32_t cb_x = 0, cb_w = 1, cb_index = 2, cb_scratch = 3, cb_out = 4, cb_route = 5, cb_land = 7;

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
// Transaction-id ring feeding `cb` with fixed-size blocks (batched_experts_kernel.cpp).
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

// Stage 0: wait for the routing, union of the selected experts (ascending ids) -> ids[], returns the count.
template <uint32_t rows, uint32_t top_k, uint32_t senders>
inline uint32_t routed_union(uint8_t* ids) {
    constexpr uint32_t route_sem = get_named_compile_time_arg_val("route_sem");
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(route_sem)), senders);
    invalidate_l1_cache();
    const auto* land = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_land));
    uint32_t lo = 0, hi = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < top_k; ++k) {
            const uint32_t e = land[r * 8 + k];
            if (e < 32) {
                lo |= 1u << e;
            } else if (e < 64) {  // 0xff: a masked (inactive) row
                hi |= 1u << (e - 32);
            }
        }
    }
    uint32_t count = 0;
    for (uint32_t e = 0; e < 64; ++e) {
        if ((e < 32 ? lo >> e : hi >> (e - 32)) & 1u) {
            ids[count++] = e;
        }
    }
    return count;
}
#endif

void kernel_main() {
    constexpr uint32_t stage = MOE_STAGE;  // define: the stages use different argument lists
    constexpr bool fused = stage == 0;
    constexpr uint32_t kt = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t bank_cols = get_named_compile_time_arg_val("bank_cols");  // columns per bank and expert
    constexpr uint32_t cols = get_named_compile_time_arg_val("core_cols");       // weight columns per block
    constexpr uint32_t splits = get_named_compile_time_arg_val("splits");        // cores per bank
    constexpr uint32_t banks = get_named_compile_time_arg_val("banks");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t tile_rows = get_named_compile_time_arg_val("tile_rows");
    constexpr uint32_t groups = get_named_compile_time_arg_val("row_groups");
    constexpr uint32_t width = get_named_compile_time_arg_val("metadata_width");
    constexpr uint32_t half = get_named_compile_time_arg_val("inter_tiles") / banks;  // activated tiles per bank
    constexpr uint32_t row_bytes = tile_rows * 64;
    constexpr uint32_t act_page_bytes = half * row_bytes;
    constexpr uint32_t chunks = fused ? half : 1;  // gate/up units per expert and bank (one gate/up pair each)
    static_assert(width % 32 == 0 && width - 1 >= (rows * top_k < 64 ? rows * top_k : 64));
    constexpr uint32_t senders = get_named_compile_time_arg_val("senders");
    static_assert(rows <= groups * tile_rows && groups <= 2 && rows >= 1);
    static_assert(tile_rows == 1 || tile_rows == 2 || tile_rows == 4 || tile_rows == 8);
    static_assert(fused ? (cols == 2 && bank_cols == 2 * half) : cols * splits == bank_cols);
    static_assert(cols * groups <= 4, "accumulators must fit four FP32 DEST slots");
    constexpr bool transposed = cols > 1;  // column-major weight blocks: one SrcB unpack serves all columns

#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t weight_addr = get_named_compile_time_arg_val("weight_addr");
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("weight_page_bytes");
    constexpr uint32_t pages = get_named_compile_time_arg_val("weight_pages");  // per contiguous run
    constexpr uint32_t run_bytes = get_named_compile_time_arg_val("weight_run_bytes");
    constexpr uint32_t expert_bytes = get_named_compile_time_arg_val("weight_expert_bytes");
    constexpr uint32_t buffers = get_named_compile_time_arg_val("weight_buffers");
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t part = get_arg_val<uint32_t>(2);
    const uint64_t base = get_noc_addr_from_bank_id<true>(bank, weight_addr);
    uint8_t ids[64];
    uint32_t count;
    if constexpr (fused) {
        DeviceZoneScopedN("MOE-WAIT");
        count = routed_union<rows, top_k, senders>(ids);
    } else {
        cb_wait_front(cb_index, 1);
        const auto* sel = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_index));
        count = sel[width - 1];
        for (uint32_t s = 0; s < count; ++s) {
            ids[s] = sel[s];
        }
    }
    constexpr uint32_t block_bytes = fused ? 2 * run_bytes : run_bytes;
    TridRing<cb_w, buffers, buffers - 2, kt * cols, block_bytes> ring;
    ring.start();
    noc_async_read_one_packet_set_state<true>(base, page_bytes, vc);
    if constexpr (fused) {
        constexpr uint32_t column_bytes = run_bytes;  // one column: kt tiles
        for (uint32_t u = part; u < count * chunks; u += splits) {
            const uint32_t s = u / chunks, c = u % chunks;
            uint32_t dst = ring.write;
            noc_async_read_set_trid(ring.trid);
            for (uint32_t run = 0; run < 2; ++run) {  // gate column c, then up column h + c
                uint32_t src = ids[s] * expert_bytes + (run * half + c) * column_bytes;
                for (uint32_t p = 0; p < pages; ++p) {
                    noc_async_read_one_packet_with_state_with_trid(static_cast<uint32_t>(base), src, dst, ring.trid);
                    src += page_bytes;
                    dst += page_bytes;
                }
            }
            ring.next();
        }
    } else {
        for (uint32_t s = 0; s < count; ++s) {
            uint32_t src = ids[s] * expert_bytes + part * run_bytes;
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
#elif defined(COMPILE_FOR_BRISC)
    const uint32_t scratch = get_write_ptr(cb_scratch);
#if MOE_STAGE == 0
    // args: x, selection, routing, activated, bank, part, writer
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto sel_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto route_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();
    constexpr auto act_args = TensorAccessorArgs<route_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), 2048);
    const auto sel_out = TensorAccessor(sel_args, get_arg_val<uint32_t>(1), width * 2);
    const auto route_out = TensorAccessor(route_args, get_arg_val<uint32_t>(2), width * 2);
    const auto act = TensorAccessor(act_args, get_arg_val<uint32_t>(3), act_page_bytes);
    const uint32_t bank = get_arg_val<uint32_t>(4);
    const uint32_t part = get_arg_val<uint32_t>(5);
    const bool writer = get_arg_val<uint32_t>(6) != 0;
    // in0 tiles: rows [g * tile_rows, ...) of each x tile's faces 0 and 1 (32 B per row and face); rows >= B zero
    constexpr uint32_t padded = groups * tile_rows;
    cb_reserve_back(cb_x, groups * kt);
    const uint32_t input = get_write_ptr(cb_x);
    if constexpr (rows < padded) {
        for (uint32_t off = 0; off < groups * kt * row_bytes; off += MEM_ZEROS_SIZE) {
            const uint32_t n = groups * kt * row_bytes - off < MEM_ZEROS_SIZE ? groups * kt * row_bytes - off
                                                                              : MEM_ZEROS_SIZE;
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), input + off, n);
        }
        noc_async_read_barrier();
    }
    for (uint32_t t = 0; t < kt; ++t) {
        const uint64_t src = get_noc_addr(t, x);
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t r0 = g * tile_rows;
            const uint32_t n = rows - r0 < tile_rows ? rows - r0 : tile_rows;
            const uint32_t dst = input + (g * kt + t) * row_bytes;
            noc_async_read(src + r0 * 32, dst, n * 32);
            noc_async_read(src + 512 + r0 * 32, dst + row_bytes / 2, n * 32);
        }
    }
    uint8_t ids[64];
    uint32_t count;
    {
        DeviceZoneScopedN("MOE-WAIT");
        count = routed_union<rows, top_k, senders>(ids);
    }
    // weight table [rows][width] BF16 + the count word after it (TRISC); union ids + count (selection, writer)
    cb_reserve_back(cb_route, 1);
    const uint32_t table = get_write_ptr(cb_route);
    {
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table);
        for (uint32_t i = 0; i < rows * width / 2; ++i) {
            words[i] = 0;
        }
        words[rows * width / 2] = count;
        uint8_t slot[64];
        for (uint32_t s = 0; s < count; ++s) {
            slot[ids[s]] = s;
        }
        const auto* land = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_land));
        auto* half_words = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(table);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t k = 0; k < top_k; ++k) {
                const uint32_t e = land[r * 8 + k];
                if (e < 64) {
                    half_words[r * width + slot[e]] = static_cast<uint16_t>(land[r * 8 + 4 + k]);
                }
            }
        }
    }
    if (writer) {
        auto* sel = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
        for (uint32_t i = 0; i < width; ++i) {
            sel[i] = i < count ? static_cast<uint16_t>(ids[i]) : 0;
        }
        sel[width - 1] = static_cast<uint16_t>(count);
        noc_async_write(scratch, get_noc_addr(0, sel_out), width * 2);
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_write(table + r * width * 2, get_noc_addr(r, route_out), width * 2);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, groups * kt);
    cb_push_back(cb_route, 1);
    // each unit's row-group tiles -> its activation pages
    for (uint32_t u = part; u < count * chunks; u += splits) {
        const uint32_t s = u / chunks, c = u % chunks;
        for (uint32_t g = 0; g < groups; ++g) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), get_noc_addr((s * groups + g) * banks + bank, act) + c * row_bytes,
                            row_bytes);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
#else
    // args: activated, selection, out, bank, part
    constexpr auto act_args = TensorAccessorArgs<0>();
    constexpr auto sel_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();
    const auto act = TensorAccessor(act_args, get_arg_val<uint32_t>(0), act_page_bytes);
    const auto sel = TensorAccessor(sel_args, get_arg_val<uint32_t>(1), width * 2);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(2), 2048);
    const uint32_t bank = get_arg_val<uint32_t>(3);
    const uint32_t part = get_arg_val<uint32_t>(4);
    cb_reserve_back(cb_index, 1);
    noc_async_read(get_noc_addr(0, sel), get_write_ptr(cb_index), width * 2);
    // zeroed staging tiles: output rows beyond the bucket stay zero
    for (uint32_t off = 0; off < cols * 2048; off += MEM_ZEROS_SIZE) {
        noc_async_read(get_noc_addr(MEM_ZEROS_BASE), scratch + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    const uint32_t count = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_index))[width - 1];
    cb_push_back(cb_index, 1);
    // every union expert's intermediate rows, gathered from its per-bank pages in K order
    TridRing<cb_x, 4, 2, groups * kt, groups * kt * row_bytes> ring;
    ring.start();
    for (uint32_t s = 0; s < count; ++s) {
        noc_async_read_set_trid(ring.trid);
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t b = 0; b < banks; ++b) {
                noc_async_read(get_noc_addr((s * groups + g) * banks + b, act),
                               ring.write + (g * kt + b * half) * row_bytes, act_page_bytes);
            }
        }
        ring.next();
    }
    ring.finish();
    noc_async_read_set_trid(0);
    cb_wait_front(cb_out, cols * groups);
    const uint32_t results = get_read_ptr(cb_out);
    for (uint32_t c = 0; c < cols; ++c) {
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t src = results + (c * groups + g) * row_bytes;
            const uint64_t dst = get_noc_addr(scratch + c * 2048 + g * tile_rows * 32);
            noc_async_write(src, dst, row_bytes / 2);
            noc_async_write(src + row_bytes / 2, dst + 512, row_bytes / 2);
        }
    }
    noc_async_write_barrier();
    for (uint32_t c = 0; c < cols; ++c) {
        noc_async_write(scratch + c * 2048, get_noc_addr(bank * bank_cols + part * cols + c, out), 2048);
    }
    cb_pop_front(cb_out, cols * groups);
    noc_async_write_barrier();
#endif
#elif defined(COMPILE_FOR_TRISC)
    static_assert(DST_ACCUM_MODE, "FP32 destination accumulation required");
    const uint32_t part = get_arg_val<uint32_t>(0);
    deepseek_compute_kernel_init();
    reconfig_data_format<SrcOrder::Reverse, true>(cb_x, cb_w);
    pack_reconfig_data_format<true>(cb_out);
    custom_mm_block_init_short<false, true, false>(cb_x, cb_w, cb_out, cols);
    if constexpr (fused) {
        PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false)));
        cb_wait_front(cb_route, 1);
        const uint32_t count = read_tile_value(cb_route, 0, rows * width / 2);
        cb_wait_front(cb_x, groups * kt);
        for (uint32_t u = part; u < count * chunks; u += splits) {
            const uint32_t s = u / chunks;
            cb_wait_front(cb_w, kt * cols);
            for (uint32_t g = 0; g < groups; ++g) {
                uint32_t weight_bits[tile_rows < 4 ? 4 : tile_rows] = {};
                for (uint32_t r = 0; r < tile_rows && g * tile_rows + r < rows; ++r) {
                    const uint32_t pair = read_tile_value(cb_route, 0, (g * tile_rows + r) * width / 2 + s / 2);
                    weight_bits[r] = s % 2 ? pair & 0xffff0000u : pair << 16;
                }
                cb_reserve_back(cb_out, 1);
                tile_regs_acquire();
                custom_mm_block<true, transposed>(cb_x, cb_w, g * kt, 0, 0, kt, cols);
                tile_regs_commit();
                PACK(TTI_SEMWAIT(p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK),
                                 p_stall::STALL_ON_ZERO));
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                PACK((_llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::calculate_rows_swiglu<tile_rows>, 0, 1, 0,
                                                            VectorMode::R, weight_bits)));
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                pack_tile(0, cb_out, 0);
                tile_regs_release();
                cb_push_back(cb_out, 1);
            }
            cb_pop_front(cb_w, kt * cols);
        }
        cb_pop_front(cb_x, groups * kt);
        cb_pop_front(cb_route, 1);
    } else {
        cb_wait_front(cb_index, 1);
        const uint32_t count = read_tile_value(cb_index, 0, (width - 1) / 2) >> 16;
        // DEST slot g * cols + c accumulates column c of row group g over every union expert (ascending ids)
        tile_regs_acquire();
        for (uint32_t s = 0; s < count; ++s) {
            cb_wait_front(cb_x, groups * kt);
            cb_wait_front(cb_w, kt * cols);
            for (uint32_t g = 0; g < groups; ++g) {
                if (s + 1 == count) {
                    custom_mm_block<true, transposed>(cb_x, cb_w, g * kt, 0, g * cols, kt, cols);
                } else {
                    custom_mm_block<false, transposed>(cb_x, cb_w, g * kt, 0, g * cols, kt, cols);
                }
            }
            cb_pop_front(cb_x, groups * kt);
            cb_pop_front(cb_w, kt * cols);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, cols * groups);
        for (uint32_t c = 0; c < cols; ++c) {
            for (uint32_t g = 0; g < groups; ++g) {
                pack_tile(g * cols + c, cb_out);
            }
        }
        tile_regs_release();
        cb_push_back(cb_out, cols * groups);
        cb_pop_front(cb_index, 1);
    }
    custom_mm_block_uninit<false>();
#endif
}
