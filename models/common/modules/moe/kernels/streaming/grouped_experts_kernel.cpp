// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// A DRAM bank's workers take the packed groups round-robin; each streams its group's
// expert shuffled BFP4 weight columns from that bank and applies them to the full
// 32-token group. Gate/up pairs fuse weighted SwiGLU in FP32 DEST; down preserves
// independent FP32 assignment outputs. The group count stays on device.
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#endif
#include "models/common/modules/moe/kernels/streaming/experts_matmul.hpp"

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_PACK)
namespace ckernel::sfpu {
// The binary SFPU wrapper visits all four faces (VectorMode::RC). Each face
// consists of eight vector slots. Tile 2 contains the token's repeated BF16 weight.
inline void calculate_grouped_swiglu(const uint dst_gate, const uint dst_up, const uint dst_out) {
#pragma GCC unroll 8
    for (int slot = 0; slot < 8; ++slot) {
        const sfpi::vFloat gate = sfpi::dst_reg[dst_gate * 32];
        const sfpi::vFloat up = sfpi::dst_reg[dst_up * 32];
        const sfpi::vFloat weight = sfpi::dst_reg[(dst_gate + 2) * 32];
        sfpi::dst_reg[dst_out * 32] = gate * _sfpu_sigmoid_<true>(gate) * up * weight;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_x = 0, cb_w = 1, cb_count = 2, cb_index = 3, cb_out = 4, cb_route = 5;

void kernel_main() {
    constexpr uint32_t kt = get_named_compile_time_arg_val("num_tiles_k");
    constexpr uint32_t nt = get_named_compile_time_arg_val("out_tiles");
    constexpr uint32_t per_bank_out = get_named_compile_time_arg_val("per_bank_out");
    constexpr uint32_t block_k = get_named_compile_time_arg_val("subblock_k");
    constexpr bool fused = get_named_compile_time_arg_val("fused_swiglu") != 0;
    constexpr uint32_t splits = get_named_compile_time_arg_val("splits");  // workers per bank
    constexpr uint32_t columns = per_bank_out * (fused ? 2 : 1);
    constexpr uint32_t k_blocks = kt / block_k;

#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t block_bytes = get_named_compile_time_arg_val("weight_block_bytes");
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("weight_page_bytes");
    constexpr uint32_t pages = get_named_compile_time_arg_val("weight_pages");
    constexpr uint32_t expert_bytes = columns * k_blocks * block_bytes;
    const uint32_t part = get_arg_val<uint32_t>(2);
    const uint64_t bank_base = get_noc_addr_from_bank_id<true>(
        get_arg_val<uint32_t>(0), get_named_compile_time_arg_val("weight_addr"));
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(bank_base, page_bytes, get_arg_val<uint32_t>(1));
    const uint32_t buffer_base = get_write_ptr(cb_w);
    const uint32_t buffer_end = buffer_base + 3 * block_bytes;
    cb_wait_front(cb_count, 1);
    const uint32_t count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_count))[0];
    for (uint32_t group = part; group < count; group += splits) {
        cb_wait_front(cb_index, 1);
        const uint32_t expert =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_index))[(group + 1) % 16];
        cb_pop_front(cb_index, 1);
        // Triple-buffered column blocks, with one read ahead of the published CB.
        // Flush at group boundaries: BRISC must drain this group's output before
        // publishing the next group's input and expert id.
        cb_reserve_back(cb_w, 2 * block_k);
        uint32_t write_addr = get_write_ptr(cb_w);
        uint32_t issue_trid = 1, wait_trid = 1;
        bool pending = false;
        for (uint32_t column = 0; column < columns; ++column) {
            // Resident gate/up bank layout is gate0, gate1, ..., up0, up1, ...;
            // stream paired columns so gate, up and routing weights fit FP32 DEST.
            const uint32_t stored_column = fused
                ? column / 2 + (column % 2) * per_bank_out : column;
            uint32_t read_offset = expert * expert_bytes + stored_column * k_blocks * block_bytes;
            for (uint32_t kb = 0; kb < k_blocks; ++kb) {
                noc_async_read_set_trid(issue_trid);
                for (uint32_t page = 0; page < pages; ++page) {
                    noc_async_read_one_packet_with_state_with_trid(bank_base, read_offset, write_addr, issue_trid);
                    read_offset += page_bytes;
                    write_addr += page_bytes;
                }
                if (pending) {
                    noc_async_read_barrier_with_trid(wait_trid);
                    cb_push_back(cb_w, block_k);
                    wait_trid = wait_trid == 3 ? 1 : wait_trid + 1;
                    cb_reserve_back(cb_w, 2 * block_k);
                } else {
                    pending = true;
                }
                issue_trid = issue_trid == 3 ? 1 : issue_trid + 1;
                if (write_addr == buffer_end) {
                    write_addr = buffer_base;
                }
            }
        }
        noc_async_read_barrier_with_trid(wait_trid);
        cb_push_back(cb_w, block_k);
    }

#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t output_bytes = fused ? 2048 : 4096;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto experts_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<experts_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), 2048);
    const auto experts = TensorAccessor(
        experts_args, get_arg_val<uint32_t>(1), get_named_compile_time_arg_val("metadata_bytes"));
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(2), output_bytes);
    const uint32_t bank = get_arg_val<uint32_t>(4);
    const uint32_t part = get_arg_val<uint32_t>(5);
    const uint64_t metadata = get_noc_addr(0, experts);
    cb_reserve_back(cb_count, 1);
    noc_async_read(metadata, get_write_ptr(cb_count), 64);
    noc_async_read_barrier();
    const uint32_t count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_count))[0];
    cb_push_back(cb_count, 1);
    for (uint32_t group = part; group < count; group += splits) {
        cb_reserve_back(cb_index, 1);
        noc_async_read(metadata + ((group + 1) / 16) * 64, get_write_ptr(cb_index), 64);
        noc_async_read_barrier();
        cb_push_back(cb_index, 1);
        cb_reserve_back(cb_x, kt);
        for (uint32_t k = 0; k < kt; ++k) {
            noc_async_read(get_noc_addr(group * kt + k, x), get_write_ptr(cb_x) + k * 2048, 2048);
        }
        if constexpr (fused) {
            constexpr auto route_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
            const auto route = TensorAccessor(route_args, get_arg_val<uint32_t>(3), 2048);
            cb_reserve_back(cb_route, 1);
            noc_async_read(get_noc_addr(group, route), get_write_ptr(cb_route), 2048);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, kt);
        if constexpr (fused) {
            cb_push_back(cb_route, 1);
        }
        for (uint32_t n = 0; n < per_bank_out; ++n) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), get_noc_addr(group * nt + bank * per_bank_out + n, out), output_bytes);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();

#elif defined(COMPILE_FOR_TRISC)
    static_assert(DST_ACCUM_MODE, "grouped expert matmuls require FP32 accumulation");
    deepseek_compute_kernel_init();
    // Reconfigure only the pack format/geometry: DST_ACCUM_MODE remains FP32.
    // cb_out is BF16 for weighted SwiGLU, Float32 for unrounded down results.
    pack_reconfig_data_format<true>(cb_out);
    if constexpr (fused) {
        PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false)));
    }
    const uint32_t part = get_arg_val<uint32_t>(0);
    cb_wait_front(cb_count, 1);
    const uint32_t count = read_tile_value(cb_count, 0, 0);
    for (uint32_t group = part; group < count; group += splits) {
        cb_wait_front(cb_x, kt);
        if constexpr (fused) {
            cb_wait_front(cb_route, 1);
        }
        for (uint32_t n = 0; n < per_bank_out; ++n) {
            cb_reserve_back(cb_out, 1);
            reconfig_data_format<SrcOrder::Reverse, true>(cb_x, cb_w);
            matmul_init(cb_x, cb_w, false);
            tile_regs_acquire();
            for (uint32_t operand = 0; operand < (fused ? 2u : 1u); ++operand) {
                for (uint32_t kb = 0; kb < k_blocks; ++kb) {
                    cb_wait_front(cb_w, block_k);
                    for (uint32_t k = 0; k < block_k; ++k) {
                        matmul_tiles(cb_x, cb_w, kb * block_k + k, k, operand);
                    }
                    cb_pop_front(cb_w, block_k);
                }
            }
            if constexpr (fused) {
                // No BF16 round trip of gate/up. Copy only the already-BF16 route
                // tile into the third FP32 DEST slot, then fuse on the pack thread.
                reconfig_data_format(cb_route, cb_route);
                copy_tile_init(cb_route);
                copy_tile(cb_route, 0, 2);
            }
            tile_regs_commit();
            if constexpr (fused) {
                PACK(TTI_SEMWAIT(
                    p_stall::STALL_TDMA | p_stall::STALL_CFG,
                    semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                PACK((_llk_math_eltwise_binary_sfpu_params_(
                    ckernel::sfpu::calculate_grouped_swiglu, 0, 1, 0, VectorMode::RC)));
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            } else {
                tile_regs_wait();
            }
            pack_tile(0, cb_out, 0);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_x, kt);
        if constexpr (fused) {
            cb_pop_front(cb_route, 1);
        }
    }
#endif
}
