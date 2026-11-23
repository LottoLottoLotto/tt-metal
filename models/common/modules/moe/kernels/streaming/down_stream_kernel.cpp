// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode MoE down projection plus expert reduction, streamed from DRAM:
//   out[t, c] = sum_s h_s[t] @ D_{e_s}[:, c]
// over the selected slots e_s. One worker core per DRAM bank (8 on P150); weights are
// WIDTH_SHARDED [E*K, N] with each bank's per-expert [K, N/8] block stored column-major
// in tiles (as for gate/up), so expert e's column c is one contiguous block.
//
// The selected slots' hidden rows are concatenated into one [rows, S*K] input, and
// each output column streams the S matching weight blocks back to back: the expert
// sum is the K accumulation of one matmul (split accumulation in FP32 dest), so no
// per-expert outputs or separate reduction exist.
//
// rows = 2 (speculative verification): the slots are the union of both tokens'
// selections, their count in selection[31]. A token's hidden row is exactly zero for a
// slot it did not select (zero routing weight), so each token's sum is unchanged, and
// every expert's weights stream once for both tokens.
//
// BRISC:  reads the selected slot ids (router's selection row) into cb_index; reads each
//         selected entry's compact hidden row (its [rows,32] input tiles back to back,
//         written in that layout by the selected SwiGLU) with one contiguous read;
//         out [1,1,rows,N].
// NCRISC: streams blocks (column c outer, slot s inner), pipelined with transaction ids.
// TRISC:  DRAMStreamingExpertsMatmul compute (custom_mm, LoFi, FP32 accumulation).
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#endif
#include "models/common/modules/moe/kernels/streaming/experts_matmul.hpp"

constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_index = 2, cb_scratch = 3, cb_out = 4;

void kernel_main() {
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");      // per expert (K / 32)
    constexpr uint32_t per_core_n = get_named_compile_time_arg_val("per_core_n");
    constexpr uint32_t selected = get_named_compile_time_arg_val("selected");    // maximum slot count
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    static_assert(rows == 1 || rows == 2);

#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t bank_id = get_named_compile_time_arg_val("bank_id");
    constexpr uint32_t vc = get_named_compile_time_arg_val("vc");
    constexpr uint32_t in1_addr = get_named_compile_time_arg_val("in1_addr");
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("in1_page_size");
    constexpr uint32_t num_pages = get_named_compile_time_arg_val("in1_num_pages");
    constexpr uint32_t block_bytes = get_named_compile_time_arg_val("in1_block_bytes");  // one expert column
    constexpr uint32_t expert_bytes = block_bytes * per_core_n;
    static_assert(page_bytes * num_pages == block_bytes);

    const uint64_t base = get_noc_addr_from_bank_id<true>(bank_id, in1_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(base, page_bytes, vc);

    // DRAMStreamingExpertsMatmul's pipeline with a deeper ring: one expert column is only
    // 9 KB, so num_buffers - 2 blocks stay in flight beyond the one being published;
    // transaction ids 1..num_buffers track the ring slots.
    constexpr uint32_t num_buffers = get_named_compile_time_arg_val("in1_buffers");
    constexpr uint32_t extra_in_flight = num_buffers - 2;
    static_assert(num_buffers >= 3 && num_buffers <= 15);
    cb_reserve_back(cb_in1, k_tiles * (extra_in_flight + 1));
    uint32_t write_addr = get_write_ptr(cb_in1);
    const uint32_t cb_base = write_addr;
    const uint32_t cb_end = cb_base + num_buffers * block_bytes;

    cb_wait_front(cb_index, 1);
    auto* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_index));
    const uint32_t count = rows == 1 ? selected : ids[31];
    uint32_t offsets[selected];
    for (uint32_t s = 0; s < count; ++s) {
        offsets[s] = static_cast<uint32_t>(ids[s]) * expert_bytes;
    }

    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    for (uint32_t c = 0; c < per_core_n; ++c) {
        for (uint32_t s = 0; s < count; ++s) {
            uint32_t src = offsets[s] + c * block_bytes;
            noc_async_read_set_trid(trid);
            for (uint32_t p = 0; p < num_pages; ++p) {
                noc_async_read_one_packet_with_state_with_trid(static_cast<uint32_t>(base), src, write_addr, trid);
                src += page_bytes;
                write_addr += page_bytes;
            }
            if (free_blocks == num_buffers - extra_in_flight) {
                noc_async_read_barrier_with_trid(trid_to_wait);
                cb_push_back(cb_in1, k_tiles);
                trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
                cb_reserve_back(cb_in1, k_tiles * (extra_in_flight + 1));
            } else {
                free_blocks -= 1;
            }
            trid = trid == num_buffers ? 1 : trid + 1;
            if (write_addr >= cb_end) {
                write_addr = cb_base;
            }
        }
    }
    for (uint32_t i = 0; i < extra_in_flight; ++i) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_in1, k_tiles);
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }

#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t bank_index = get_named_compile_time_arg_val("bank_id");
    constexpr auto h_args = TensorAccessorArgs<0>();
    constexpr auto sel_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t sel_bytes = 64;
    constexpr uint32_t page_bytes = k_tiles * rows * 64;  // one entry's [rows,32] tiles
    const auto hidden = TensorAccessor(h_args, get_arg_val<uint32_t>(0), page_bytes);
    const auto selection = TensorAccessor(sel_args, get_arg_val<uint32_t>(1), sel_bytes);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(2), tile_bytes);

    constexpr uint32_t face_bytes = rows * 32;          // rows 0..rows-1 of one face
    constexpr uint32_t row_tile_bytes = 2 * face_bytes;  // [rows, 32] BF16 tile
    const uint32_t stage_l1 = get_write_ptr(cb_scratch);
    constexpr uint32_t stages = 4;

    cb_reserve_back(cb_index, 1);
    noc_async_read(get_noc_addr(0, selection), get_write_ptr(cb_index), sel_bytes);
    noc_async_read_barrier();
    auto* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_index));
    const uint32_t count = rows == 1 ? selected : ids[31];
    cb_push_back(cb_index, 1);

    // hidden is compact, in selection order: page s holds entry s's k_tiles [rows,32] input
    // tiles back to back. One contiguous read per entry (two 32-64 B face-row reads per
    // tile measured ~7 us of this op's 22 us).
    cb_reserve_back(cb_in0, selected * k_tiles);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t s = 0; s < count; ++s) {
        noc_async_read(get_noc_addr(s, hidden), in0 + s * page_bytes, page_bytes);
    }
    for (uint32_t off = 0; off < stages * tile_bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(get_noc_addr(MEM_ZEROS_BASE), stage_l1 + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    cb_push_back(cb_in0, selected * k_tiles);

    // The output is a full tile row consumed as such (residual add), so each result's
    // rows are written into zeroed staging tiles by a local NOC copy (rows..31 stay zero),
    // then whole tiles go out. Copies land (barrier) before the tile writes read the
    // stages; the next batch reuses them only after those writes.
    static_assert(per_core_n % stages == 0 || per_core_n < stages);
    for (uint32_t c0 = 0; c0 < per_core_n; c0 += stages) {
        const uint32_t batch = per_core_n - c0 < stages ? per_core_n - c0 : stages;
        for (uint32_t i = 0; i < batch; ++i) {
            cb_wait_front(cb_out, 1);
            const uint64_t stage_noc = get_noc_addr(stage_l1 + i * tile_bytes);
            const uint32_t src = get_read_ptr(cb_out);
            noc_async_write(src, stage_noc, face_bytes);
            noc_async_write(src + face_bytes, stage_noc + 512, face_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
        for (uint32_t i = 0; i < batch; ++i) {
            noc_async_write(stage_l1 + i * tile_bytes, get_noc_addr(bank_index * per_core_n + c0 + i, out), tile_bytes);
        }
        noc_async_write_barrier();
    }
    noc_async_write_barrier();

#elif defined(COMPILE_FOR_TRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingExpertsMatmul::ComputeCTArgs<
        cb_in0,
        cb_in1,
        cb_out,
        k_tiles,     // one expert block per K subblock
        per_core_n,
        get_named_compile_time_arg_val("subblock_w"),
        selected,    // K subblocks = selected experts, accumulated (split_acc, finalize last)
        rows,        // in0 tile rows
        0,           // no fused SiLU
        1>;          // FP32 accumulation
    deepseek_compute_kernel_init();
    deepseek_b1_ops::DRAMStreamingExpertsMatmul::Op<CTArgs, true, false> op;
    if constexpr (rows == 1) {
        op();
    } else {
        cb_wait_front(cb_index, 1);
        const uint32_t count = read_tile_value(cb_index, 0, 15) >> 16;  // selection[31]
        op(0, per_core_n, count);
    }
    // in0 was pushed as the full maximum-size block; pop it whole.
    cb_pop_front(cb_in0, selected * k_tiles);
#endif
}
