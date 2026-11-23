// SPDX-License-Identifier: Apache-2.0
// Shared-KV verification SDPA writer: ttnn sdpa_decode's writer_decode_all.cpp (causal, DRAM
// output, GQA) with (1) per-row causal masks for the rows of this slot's (pass, row block) item
// (verify_sdpa_common.hpp; each packed Q row masks keys beyond its own position; rows outside the
// item reuse the representative row's mask and are discarded), and (2) the root scattering each
// item row of its KV head's 6 Q heads straight
// into the verification attention output [1, rows, 24*256]. Scalars, the tree reduction and all
// reduction transfers are the reference code.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "models/demos/blackhole/qwen36/tt/attention/verify_sdpa_common.hpp"

#define MAX_TREE_REDUCTION_ROUNDS 6

// Row r of the mask tiles masks key column c of K tile kt when kt*32 + c > boundary[r] (the
// row's position within the last chunk): 0 = keep, bf16 -inf = drop, as generate_mask fills.
template <uint32_t cb_mask_in, uint32_t PNHt, uint32_t Sk_chunk_t>
void generate_row_masks(const uint32_t (&boundary)[PNHt * 32]) {
    constexpr uint16_t NEG_INF = 0xFF80;
    CircularBuffer cb_mask(cb_mask_in);
    constexpr uint32_t tiles = PNHt * Sk_chunk_t;
    cb_mask.reserve_back(tiles);
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);
    const uint32_t base = cb_mask.get_write_ptr();
    for (uint32_t row_tile = 0; row_tile < PNHt; ++row_tile) {
        for (uint32_t k_tile = 0; k_tile < Sk_chunk_t; ++k_tile) {
            volatile tt_l1_ptr uint16_t* tile =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + (row_tile * Sk_chunk_t + k_tile) * tile_bytes);
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t limit = boundary[row_tile * 32 + row];
                for (uint32_t col = 0; col < 32; ++col) {
                    const uint32_t face = (row / 16) * 2 + col / 16;
                    tile[face * 256 + (row % 16) * 16 + col % 16] = (k_tile * 32 + col > limit) ? NEG_INF : 0;
                }
            }
        }
    }
    cb_mask.push_back(tiles);
}

void kernel_main() {
    Noc noc;

    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t PNHt = 1;
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(3);
    constexpr uint32_t num_cores = get_compile_time_arg_val(4);
    constexpr uint32_t reducer_semaphore_id = get_compile_time_arg_val(5);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();
    uint32_t reducer_semaphore_addr = get_semaphore(reducer_semaphore_id);

    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_11;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;
    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_worker = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_intermed_out = tt::CBIndex::c_19;
    constexpr uint32_t cb_out = tt::CBIndex::c_20;

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_head = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slot = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const bool is_tree_root = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t parent_core_in_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t send_at_round = get_arg_val<uint32_t>(arg_idx++);
    constexpr uint32_t step_semaphore_inc[6] = {1, 16, 256, 4096, 65536, 1048576};
    constexpr uint32_t step_semaphore_shift[6] = {0, 4, 8, 12, 16, 20};
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        children_per_round[r] = get_arg_val<uint32_t>(arg_idx++);
    }
    tt_l1_ptr uint32_t* reduction_group_core_xs = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_per_head;
    tt_l1_ptr uint32_t* reduction_group_core_ys = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_per_head;

    // This slot's item rows and representative position (reader filled the stick).
    uint32_t positions[T];
    uint32_t cur_pos = UINT32_MAX;
    uint32_t rows = 0;
    uint32_t block = 0;
    {
        CircularBuffer cb_index(cb_cur_pos);
        cb_index.wait_front(1);
        const volatile tt_l1_ptr uint32_t* index_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_index.get_read_ptr());
        for (uint32_t r = 0; r < T; ++r) {
            positions[r] = index_ptr[r];
        }
        const verify_sdpa::SlotWork work = verify_sdpa::slot_work<T, k_chunk_size>(index_ptr, slot);
        rows = work.rows;
        block = work.block;
        cur_pos = work.representative;
        cb_index.pop_front(1);
    }
    if (cur_pos == UINT32_MAX) {
        return;
    }

    const auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, 8>(cur_pos);
    const auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(cur_pos, slot, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic);
    if (k_chunk_start == k_chunk_end) {
        return;
    }

    uint32_t num_active_children = 0;
    uint32_t active_children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    uint32_t num_active_rounds = 0;
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        uint32_t child_id = children_per_round[r];
        if (child_id != UINT32_MAX && child_id < k_num_chunks) {
            active_children_per_round[r] = child_id;
            num_active_children++;
            num_active_rounds = r + 1;
        } else {
            active_children_per_round[r] = UINT32_MAX;
        }
    }
    const bool should_send_to_parent = (parent_core_in_group != UINT32_MAX);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    dataflow_kernel_lib::prepare_zero_tile<cb_zero_in>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);

    // Per-row causal masks of the last chunk (the reference mask of each row's own position).
    {
        uint32_t boundary[PNHt * 32];
        const uint32_t chunk_span = Sk_chunk_t_dynamic * 32;
        const uint32_t fallback = cur_pos % chunk_span;
        for (uint32_t i = 0; i < PNHt * 32; ++i) {
            const uint32_t tile = i / 32;
            const uint32_t in_tile = i % 32;
            const uint32_t row = (block + tile) * verify_sdpa::kRowsPerTile + in_tile / verify_sdpa::kHeadsPerKv;
            const bool mine = in_tile < verify_sdpa::kRowsPerTile * verify_sdpa::kHeadsPerKv && ((rows >> row) & 1);
            boundary[i] = mine ? positions[row] % chunk_span : fallback;
        }
        generate_row_masks<cb_mask_in, PNHt, Sk_chunk_t>(boundary);
    }

    constexpr uint32_t out_chunk_tiles = PNHt * vDHt;
    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);
    noc.async_write_barrier();  // #19201 BH hang workaround

    if (num_active_children > 0) {
        for (uint32_t round = 0; round < num_active_rounds; ++round) {
            uint32_t child_id = active_children_per_round[round];
            if (child_id != UINT32_MAX) {
                while (true) {
                    invalidate_l1_cache();
                    uint32_t sem_val = *in0_receiver_semaphore_addr_ptr;
                    uint8_t step_sem = (sem_val >> step_semaphore_shift[round]) & 0x0F;
                    if (step_sem >= 1) {
                        break;
                    }
                }
                constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
                constexpr uint32_t o_read_size = out_chunk_tiles * tile_bytes_intermed;
                constexpr uint32_t ml_read_size = PNHt * tile_bytes_intermed;
                uint32_t block_offset = round * (out_chunk_tiles + 2 * PNHt) * tile_bytes_intermed;
                CircularBuffer cb_intermed(cb_intermed_out);
                const uint8_t noc_id = noc.get_noc_id();
                const uint32_t my_noc_x = my_x[noc_id];
                const uint32_t my_noc_y = my_y[noc_id];
                uint32_t intermed_l1_read_addr = cb_intermed.get_read_ptr() + block_offset;
                UnicastEndpoint intermed_src;

                CircularBuffer cb_l(cb_l_in);
                cb_l.reserve_back(PNHt);
                noc.async_read(
                    intermed_src, CoreLocalMem<uint32_t>(cb_l.get_write_ptr()), ml_read_size,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr}, {});
                intermed_l1_read_addr += ml_read_size;
                noc.async_read_barrier();
                cb_l.push_back(PNHt);

                CircularBuffer cb_m(cb_m_in);
                cb_m.reserve_back(PNHt);
                noc.async_read(
                    intermed_src, CoreLocalMem<uint32_t>(cb_m.get_write_ptr()), ml_read_size,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr}, {});
                intermed_l1_read_addr += ml_read_size;
                noc.async_read_barrier();
                cb_m.push_back(PNHt);

                CircularBuffer cb_o(cb_out_o);
                cb_o.reserve_back(out_chunk_tiles);
                noc.async_read(
                    intermed_src, CoreLocalMem<uint32_t>(cb_o.get_write_ptr()), o_read_size,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr}, {});
                noc.async_read_barrier();
                cb_o.push_back(out_chunk_tiles);
            }
        }
    }

    if (!is_tree_root && should_send_to_parent) {
        CircularBuffer cb_out_w(cb_out_worker);
        CircularBuffer cb_out_m_buf(cb_out_m);
        CircularBuffer cb_out_l_buf(cb_out_l);
        CircularBuffer cb_intermed(cb_intermed_out);
        cb_out_w.wait_front(out_chunk_tiles);
        cb_out_m_buf.wait_front(PNHt);
        cb_out_l_buf.wait_front(PNHt);
        constexpr uint32_t tile_bytes = get_tile_size(cb_out_worker);
        uint32_t block_offset = send_at_round * (out_chunk_tiles + 2 * PNHt) * tile_bytes;
        constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes;
        constexpr uint32_t ml_write_size = PNHt * tile_bytes;
        uint32_t parent_noc_x = reduction_group_core_xs[parent_core_in_group];
        uint32_t parent_noc_y = reduction_group_core_ys[parent_core_in_group];
        uint32_t output_write_addr = cb_intermed.get_write_ptr() + block_offset;
        UnicastEndpoint output_dst;
        noc.async_write(
            CoreLocalMem<uint32_t>(cb_out_l_buf.get_read_ptr()), output_dst, ml_write_size, {},
            {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
        output_write_addr += ml_write_size;
        noc.async_write(
            CoreLocalMem<uint32_t>(cb_out_m_buf.get_read_ptr()), output_dst, ml_write_size, {},
            {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
        output_write_addr += ml_write_size;
        noc.async_write(
            CoreLocalMem<uint32_t>(cb_out_w.get_read_ptr()), output_dst, o_write_size, {},
            {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
        noc.async_write_barrier();
        Semaphore<>(reducer_semaphore_id).up(noc, parent_noc_x, parent_noc_y, step_semaphore_inc[send_at_round]);
        cb_out_w.pop_front(out_chunk_tiles);
        cb_out_m_buf.pop_front(PNHt);
        cb_out_l_buf.pop_front(PNHt);
        noc.async_atomic_barrier();
        return;
    }
    if (!is_tree_root) {
        return;
    }

    // Root: scatter each pass row's 6 head rows (32 BF16 = two 32-byte face-row stripes per tile)
    // into output tile (head * vDHt + d), row = verification row.
    CircularBuffer cb_out_buf(cb_out);
    cb_out_buf.wait_front(out_chunk_tiles);
    noc.async_writes_flushed();
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_writer = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t src_base = cb_out_buf.get_read_ptr();
    for (uint32_t tile = 0; tile < PNHt; ++tile) {
        for (uint32_t in_tile = 0; in_tile < verify_sdpa::kRowsPerTile * verify_sdpa::kHeadsPerKv; ++in_tile) {
            const uint32_t row = (block + tile) * verify_sdpa::kRowsPerTile + in_tile / verify_sdpa::kHeadsPerKv;
            if (((rows >> row) & 1) == 0) {
                continue;
            }
            const uint32_t head = cur_head * verify_sdpa::kHeadsPerKv + in_tile % verify_sdpa::kHeadsPerKv;
            for (uint32_t d = 0; d < vDHt; ++d) {
                const uint32_t src_tile = src_base + (tile * vDHt + d) * tile_bytes;
                const uint64_t dst_tile = out_writer.get_noc_addr(head * vDHt + d);
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t src = src_tile + ((in_tile / 16) * 2 + half) * 512 + (in_tile % 16) * 32;
                    const uint64_t dst = dst_tile + ((row / 16) * 2 + half) * 512 + (row % 16) * 32;
                    noc_async_write(src, dst, 32);
                }
            }
        }
    }
    noc.async_write_barrier();
    cb_out_buf.pop_front(out_chunk_tiles);
}
