// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode MoE gate/up projection fused with the routed SwiGLU, streamed from DRAM:
//   hidden[s][t] = SiLU(x[t] @ G[e_s]) * (x[t] @ U[e_s]) * weight[t][s]
// for the selected slots e_s. One worker core per DRAM bank (8 on P150). Weights are
// WIDTH_SHARDED [E*K, N] with each bank's per-expert [K, N/8] block stored column-major in
// tiles, so every output tile's K-tile column is one contiguous read from the core's own
// bank. The gate/up columns are interleaved per bank (interleave_gate_up_for_banks): bank
// b's per_core_n columns are gate tiles [b*h, b*h + h) then the matching up tiles, h =
// per_core_n / 2, so both operands of each SwiGLU tile land in the same core's DEST.
//
// rows = 1: one token, the router's top-k slots.
// rows = 2: two tokens (speculative verification); the slots are the union of both
//           tokens' selections, their count in selection[31]. Each streamed weight block
//           multiplies both token rows ([2,32] input tiles), and a token's zero routing
//           weight for the other token's experts yields exact zeros.
//
// BRISC:  reads the selected slot ids and routing weights; gathers rows 0..rows-1 of the
//         tiled input x into [rows,32] tiles; writes each activated [rows,32] tile into
//         the compact down-projection input: page s (selection entry) holds the entry's
//         intermediate tiles back to back.
// NCRISC: DRAMStreamingExpertsMatmul weight streaming (pipelined, transaction ids).
// TRISC:  per slot, the per_core_n gate/up columns accumulate in DEST (custom_mm, LoFi,
//         FP32); the pack thread then applies SiLU(gate) * up * weight on the token rows
//         in FP32 (no BF16 round trip of gate/up) and packs only the h activated tiles.
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#endif
#include "models/common/modules/moe/kernels/streaming/experts_matmul.hpp"
#if defined(COMPILE_FOR_TRISC) && defined(TRISC_PACK)
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {
// out = SiLU(gate) * up * weight on the first ITERATIONS 4-row x 8-column slots of a face.
// SFPU lane 8i + c holds row i of the slot (vConstTileId = 2 * lane), so lanes 0-7 are
// token row 0 and lanes 8-15 token row 1.
template <int ITERATIONS, bool two_rows>
inline void calculate_routed_swiglu(
    const uint dst_gate, const uint dst_up, const uint dst_out, const uint weight0, const uint weight1) {
    constexpr uint dst_tile_size_sfpi = 32;
    // Both constants are loaded before the lane predicate: a constant load inside the
    // v_if block crashed the SFPU compiler (rvtt_live internal error).
    const sfpi::vFloat weight_row0 = Converter::as_float(weight0);
    sfpi::vFloat weight = Converter::as_float(weight1);
    if constexpr (two_rows) {
        v_if(sfpi::vConstTileId < 16) { weight = weight_row0; }
        v_endif;
    } else {
        weight = weight_row0;
    }
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat gate = sfpi::dst_reg[dst_gate * dst_tile_size_sfpi];
        const sfpi::vFloat up = sfpi::dst_reg[dst_up * dst_tile_size_sfpi];
        sfpi::dst_reg[dst_out * dst_tile_size_sfpi] = gate * _sfpu_sigmoid_<true>(gate) * up * weight;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_index = 2, cb_scratch = 3, cb_out = 4, cb_weights = 5;

void kernel_main() {
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("num_tiles_k");
    constexpr uint32_t per_core_n = get_named_compile_time_arg_val("per_core_n");  // gate + up columns
    constexpr uint32_t half_n = per_core_n / 2;                                    // activated tiles
    constexpr uint32_t selected = get_named_compile_time_arg_val("selected");      // maximum slot count
    constexpr uint32_t subblock_k = get_named_compile_time_arg_val("subblock_k");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    static_assert(rows == 1 || rows == 2);
    static_assert(per_core_n % 2 == 0 && per_core_n <= 4, "gate/up pairs must fit one FP32 half-sync DEST");

#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingExpertsMatmul::ReaderCTArgs<
        cb_in1,
        cb_out,
        get_named_compile_time_arg_val("in1_addr"),
        get_named_compile_time_arg_val("in1_page_size"),
        get_named_compile_time_arg_val("in1_num_pages"),
        subblock_k,
        per_core_n,
        get_named_compile_time_arg_val("in1_block_bytes"),
        half_n * selected,
        num_tiles_k / subblock_k,
        get_named_compile_time_arg_val("bank_id"),
        get_named_compile_time_arg_val("vc"),
        1,         // enable_indexing
        cb_index,  // expert ids
        0,         // index_offset
        0,         // use_hardcoded_expert_index
        selected>;
    deepseek_b1_ops::DRAMStreamingExpertsMatmul::Op<CTArgs, true> op;
    if constexpr (rows == 1) {
        op();
    } else {
        cb_wait_front(cb_index, 1);
        const uint32_t count = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_index))[31];
        op(count, 0, num_tiles_k / subblock_k);
    }

#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t bank_index = get_named_compile_time_arg_val("bank_id");
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto sel_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t row_bytes = 64;  // [1, 32] 16-bit row
    // [rows, 32] BF16 tile: face 0 rows then face 1 rows (32 B per face row).
    constexpr uint32_t face_bytes = rows * 32;
    constexpr uint32_t row_tile_bytes = 2 * face_bytes;
    constexpr uint32_t inter_tiles = get_named_compile_time_arg_val("inter_tiles");
    constexpr uint32_t page_bytes = inter_tiles * row_tile_bytes;  // one entry's compact hidden row
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), tile_bytes);
    const auto selection = TensorAccessor(sel_args, get_arg_val<uint32_t>(1), row_bytes);
    const auto weights = TensorAccessor(w_args, get_arg_val<uint32_t>(2), row_bytes);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(3), page_bytes);

    // The expert ids gate NCRISC's first weight read: publish them before anything else.
    // The router wrote them ascending (shared slot last): one 64-byte read, no mask scan.
    cb_reserve_back(cb_index, 1);
    noc_async_read(get_noc_addr(0, selection), get_write_ptr(cb_index), row_bytes);
    noc_async_read_barrier();
    const uint32_t count =
        rows == 1 ? selected : reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_index))[31];
    cb_push_back(cb_index, 1);

    // Routing weight rows (one per token), read by the compute threads.
    cb_reserve_back(cb_weights, 1);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(get_noc_addr(r, weights), get_write_ptr(cb_weights) + r * row_bytes, row_bytes);
    }

    // Rows 0-1 of each x tile into its [rows,32] input tile. x may live in DRAM, whose reads
    // must keep source and destination 64 B aligned alike, so each face-row pair lands in
    // aligned scratch first; local L1->L1 NOC copies (16 B alignment) then place the
    // row halves. A BRISC word-copy repack instead measured 9.9 us.
    const uint32_t staging = get_write_ptr(cb_scratch);
    for (uint32_t t = 0; t < num_tiles_k; ++t) {
        const uint64_t src = get_noc_addr(t, x);
        noc_async_read(src, staging + t * 128, 64);             // rows 0-1, cols 0-15 (face 0)
        noc_async_read(src + 512, staging + t * 128 + 64, 64);  // rows 0-1, cols 16-31 (face 1)
    }
    noc_async_read_barrier();
    cb_push_back(cb_weights, 1);
    cb_reserve_back(cb_in0, num_tiles_k);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t t = 0; t < num_tiles_k; ++t) {
        const uint64_t dst = get_noc_addr(in0 + t * row_tile_bytes);
        noc_async_write(staging + t * 128, dst, face_bytes);
        noc_async_write(staging + t * 128 + 64, dst + face_bytes, face_bytes);
    }
    noc_async_write_barrier();
    cb_push_back(cb_in0, num_tiles_k);

    // Results arrive slot-major: (entry s, tile j) -> page s, intermediate tile bank*h + j.
    // The packed [rows,32] tile is already the compact layout: one contiguous write.
    for (uint32_t s = 0; s < count; ++s) {
        const uint64_t page = get_noc_addr(s, out) + bank_index * half_n * row_tile_bytes;
        for (uint32_t j = 0; j < half_n; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), page + j * row_tile_bytes, row_tile_bytes);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();

#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t num_subblocks_k = num_tiles_k / subblock_k;
    constexpr bool transpose = false, split_acc = true, dense_packing = false;
    static_assert(DST_ACCUM_MODE, "FP32 accumulation: the SwiGLU runs on FP32 DEST values");
    deepseek_compute_kernel_init();
    reconfig_data_format<SrcOrder::Reverse, /*is_tile_dim_reconfig_en=*/true>(cb_in0, cb_in1);
    pack_reconfig_data_format<true>(cb_out);
    custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_in0, cb_in1, cb_out);
    PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false /*APPROXIMATE*/)));

    cb_wait_front(cb_index, 1);
    const uint32_t count = rows == 1 ? selected : read_tile_value(cb_index, 0, 15) >> 16;  // selection[31]
    // Routing weights as FP32 bits: row r, entry s is BF16 element 32 r + s.
    cb_wait_front(cb_weights, 1);
    uint32_t weight_bits[rows][selected];
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t s = 0; s < count; s += 2) {
            const uint32_t pair = read_tile_value(cb_weights, 0, r * 16 + s / 2);
            weight_bits[r][s] = pair << 16;
            if (s + 1 < count) {
                weight_bits[r][s + 1] = pair & 0xffff0000u;
            }
        }
    }

    cb_wait_front(cb_in0, num_tiles_k);
    for (uint32_t s = 0; s < count; ++s) {
        cb_reserve_back(cb_out, half_n);
        tile_regs_acquire();
        for (uint32_t w = 0; w < per_core_n; ++w) {
            for (uint32_t sb_k = 0; sb_k + 1 < num_subblocks_k; ++sb_k) {
                cb_wait_front(cb_in1, subblock_k);
                custom_mm_block<false>(cb_in0, cb_in1, sb_k * subblock_k, 0, w, subblock_k);
                cb_pop_front(cb_in1, subblock_k);
            }
            cb_wait_front(cb_in1, subblock_k);
            custom_mm_block<true>(cb_in0, cb_in1, (num_subblocks_k - 1) * subblock_k, 0, w, subblock_k);
            cb_pop_front(cb_in1, subblock_k);
        }
        tile_regs_commit();

        // SwiGLU on the pack thread (as DRAMStreamingExpertsMatmul's fused SiLU): wait for
        // math, address the packer's DEST half, then only the token rows (rows 0-3 of faces
        // 0-1: VectorMode::R, two slots per face).
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t j = 0; j < half_n; ++j) {
            PACK((_llk_math_eltwise_binary_sfpu_params_(
                ckernel::sfpu::calculate_routed_swiglu<2, rows == 2>,
                j,
                half_n + j,
                j,
                VectorMode::R,
                weight_bits[0][s],
                weight_bits[rows - 1][s])));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        for (uint32_t j = 0; j < half_n; ++j) {
            pack_tile(j, cb_out, j);
        }
        tile_regs_release();
        cb_push_back(cb_out, half_n);
    }
    custom_mm_block_uninit<dense_packing>();
    cb_pop_front(cb_in0, num_tiles_k);
#endif
}
