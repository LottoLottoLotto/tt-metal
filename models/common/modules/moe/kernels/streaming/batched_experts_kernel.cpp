// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Union-expert decode, several worker cores per DRAM bank. Gate/up splits the union's
// experts across a bank's cores (each reads whole expert bank blocks once, shared by
// every row group). Down splits the bank's output columns: each core streams its
// columns of every expert once, accumulating all experts in FP32 DEST. B16 uses two
// 8-row subtiles against the same unpacked weight block.
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
    constexpr uint32_t kt = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t per_core_n = get_named_compile_time_arg_val("per_core_n");  // bank columns
    constexpr uint32_t cols = get_named_compile_time_arg_val("core_cols");         // columns per core
    constexpr uint32_t splits = get_named_compile_time_arg_val("splits");          // cores per bank
    constexpr uint32_t banks = get_named_compile_time_arg_val("banks");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t tile_rows = get_named_compile_time_arg_val("tile_rows");
    constexpr uint32_t groups = get_named_compile_time_arg_val("row_groups");
    constexpr uint32_t selected = get_named_compile_time_arg_val("selected");
    constexpr uint32_t width = get_named_compile_time_arg_val("metadata_width");
    constexpr bool fused = get_named_compile_time_arg_val("fused_swiglu");
    constexpr uint32_t row_bytes = tile_rows * 64;
    // Activation pages: one per (expert, row group, gate/up bank), holding that bank's
    // `half` intermediate row tiles. Each expert's down input is spread over many L1
    // banks, so concurrent down cores do not converge on one source core.
    constexpr uint32_t half = get_named_compile_time_arg_val("inter_tiles") / banks;
    constexpr uint32_t act_page_bytes = half * row_bytes;
    static_assert(tile_rows == 1 || tile_rows == 2 || tile_rows == 4 || tile_rows == 8);
    static_assert(rows == groups * tile_rows && groups <= 2);
    static_assert(width > selected && width % 32 == 0);
    static_assert(fused ? cols == per_core_n && per_core_n == 2 * half : cols * splits == per_core_n);
    static_assert(cols * (fused ? 1 : groups) <= 4, "accumulators must fit four FP32 DEST slots");
    // Weight blocks are column-major (K tiles contiguous per column): with several
    // columns, read them transposed so one SrcB input unpack serves all of them.
    constexpr bool transposed = cols > 1;

#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t weight_addr = get_named_compile_time_arg_val("weight_addr");
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("weight_page_bytes");
    constexpr uint32_t pages = get_named_compile_time_arg_val("weight_pages");
    constexpr uint32_t block_bytes = get_named_compile_time_arg_val("weight_block_bytes");
    constexpr uint32_t expert_bytes = get_named_compile_time_arg_val("weight_expert_bytes");
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t part = get_arg_val<uint32_t>(2);
    const uint64_t base = get_noc_addr_from_bank_id<true>(bank, weight_addr);
    cb_wait_front(cb_index, 1);
    auto* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_index));
    const uint32_t count = ids[width - 1];
    // Gate/up: this core's experts, one whole bank block each, retained while every row
    // group computes. Down: this core's column slice of every expert.
    constexpr uint32_t buffers = fused ? 3 : 8, ahead = buffers - 2;
    TridRing<cb_w, buffers, ahead, kt * cols, block_bytes> ring;
    ring.start();
    noc_async_read_one_packet_set_state<true>(base, page_bytes, vc);
    const uint32_t first = fused ? part : 0, stride = fused ? splits : 1;
    const uint32_t column_offset = fused ? 0 : part * block_bytes;
    for (uint32_t s = first; s < count; s += stride) {
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
#elif defined(COMPILE_FOR_BRISC)
    const uint32_t bank = get_arg_val<uint32_t>(4);
    const uint32_t part = get_arg_val<uint32_t>(5);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto ids_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto route_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<route_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), fused ? 2048 : act_page_bytes);
    const auto ids = TensorAccessor(ids_args, get_arg_val<uint32_t>(1), width * 2);
    const auto route = TensorAccessor(route_args, get_arg_val<uint32_t>(2), width * 2);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(3), fused ? act_page_bytes : 2048);
    cb_reserve_back(cb_index, 1);
    noc_async_read(get_noc_addr(0, ids), get_write_ptr(cb_index), width * 2);
    noc_async_read_barrier();
    const uint32_t count = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_index))[width - 1];
    cb_push_back(cb_index, 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);
    if constexpr (fused) {
        cb_reserve_back(cb_route, 1);
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_read(get_noc_addr(r, route), get_write_ptr(cb_route) + r * width * 2, width * 2);
        }
        constexpr uint32_t read_face_bytes = (rows < 2 ? 2 : rows) * 32;
        for (uint32_t t = 0; t < kt; ++t) {
            const uint64_t src = get_noc_addr(t, x);
            noc_async_read(src, scratch + t * 2 * read_face_bytes, read_face_bytes);
            noc_async_read(src + 512, scratch + (t * 2 + 1) * read_face_bytes, read_face_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_route, 1);
        cb_reserve_back(cb_x, groups * kt);
        const uint32_t input = get_write_ptr(cb_x);
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t t = 0; t < kt; ++t) {
                const uint32_t src = scratch + t * 2 * read_face_bytes + g * tile_rows * 32;
                const uint64_t dst = get_noc_addr(input + (g * kt + t) * row_bytes);
                noc_async_write(src, dst, row_bytes / 2);
                noc_async_write(src + read_face_bytes, dst + row_bytes / 2, row_bytes / 2);
            }
        }
        noc_async_write_barrier();
        cb_push_back(cb_x, groups * kt);
        for (uint32_t s = part; s < count; s += splits) {
            for (uint32_t g = 0; g < groups; ++g) {
                cb_wait_front(cb_out, half);
                noc_async_write(get_read_ptr(cb_out), get_noc_addr((s * groups + g) * banks + bank, out), act_page_bytes);
                noc_async_writes_flushed();
                cb_pop_front(cb_out, half);
            }
        }
    } else {
        // Zeroed staging tiles: output rows beyond the bucket stay zero.
        for (uint32_t off = 0; off < cols * 2048; off += MEM_ZEROS_SIZE) {
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), scratch + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        // Every expert's full intermediate input, gathered from its per-bank pages.
        TridRing<cb_x, 4, 2, groups * kt, groups * kt * row_bytes> ring;
        ring.start();
        for (uint32_t s = 0; s < count; ++s) {
            noc_async_read_set_trid(ring.trid);
            for (uint32_t g = 0; g < groups; ++g) {
                for (uint32_t b = 0; b < banks; ++b) {
                    noc_async_read(get_noc_addr((s * groups + g) * banks + b, x),
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
            noc_async_write(scratch + c * 2048, get_noc_addr(bank * per_core_n + part * cols + c, out), 2048);
        }
        cb_pop_front(cb_out, cols * groups);
    }
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    static_assert(DST_ACCUM_MODE, "batched experts require FP32 destination accumulation");
    const uint32_t part = get_arg_val<uint32_t>(0);
    deepseek_compute_kernel_init();
    reconfig_data_format<SrcOrder::Reverse, true>(cb_x, cb_w);
    pack_reconfig_data_format<true>(cb_out);
    custom_mm_block_init_short<false, true, false>(cb_x, cb_w, cb_out, cols);
    cb_wait_front(cb_index, 1);
    const uint32_t count = read_tile_value(cb_index, 0, (width - 1) / 2) >> 16;
    if constexpr (fused) {
        PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false)));
        cb_wait_front(cb_x, groups * kt);
        cb_wait_front(cb_route, 1);
        for (uint32_t s = part; s < count; s += splits) {
            cb_wait_front(cb_w, kt * per_core_n);
            for (uint32_t g = 0; g < groups; ++g) {
                uint32_t weight_bits[tile_rows < 4 ? 4 : tile_rows] = {};
                for (uint32_t r = 0; r < tile_rows; ++r) {
                    const uint32_t pair = read_tile_value(cb_route, 0, (g * tile_rows + r) * width / 2 + s / 2);
                    weight_bits[r] = s % 2 ? pair & 0xffff0000u : pair << 16;
                }
                cb_reserve_back(cb_out, half);
                tile_regs_acquire();
                custom_mm_block<true, transposed>(cb_x, cb_w, g * kt, 0, 0, kt, per_core_n);
                tile_regs_commit();
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
            }
            cb_pop_front(cb_w, kt * per_core_n);
        }
        cb_pop_front(cb_x, groups * kt);
    } else {
        // DEST slot g * cols + c accumulates column c of row group g over every expert,
        // in union order; the other rows' experts contribute exact zeros.
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
    }
    custom_mm_block_uninit<false>();
#endif
}
