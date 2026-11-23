// SPDX-License-Identifier: Apache-2.0
// Shared-KV verification SDPA reader: ttnn sdpa_decode's reader_decode_all.cpp (paged, causal,
// DRAM Q, no mask/sink/window/mcast) with (1) the slot rule of verify_sdpa_common.hpp selecting
// this core's (pass, row block) item and the representative position handed to compute,
// (2) per-(row block, KV head) packed queries: tile (head * blocks + block) holds rows
// 4*block .. +3 x that head's 6 Q heads, and (3) K/V chunk sharing: a core whose item has a
// partner (same pass, other row block, same KV head and member) reads only K (lower slot) or
// only V (higher slot) from DRAM and copies it into the partner's identical CB slot, so each
// chunk leaves DRAM once. CB contents equal the reference read_k/read_v placement (K transposed
// per chunk, V row-major). Unshared items keep the reference read barrier cadence; shared items
// tag each chunk's reads with a transaction id and stream `ahead` chunks in front of the exchange.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "models/demos/blackhole/qwen36/tt/attention/verify_sdpa_common.hpp"

// One K (transposed per chunk) or V (row-major) chunk into `write_ptr`, as read_k / read_v place
// it. trid == 0: reference cadence (barrier every `threshold` reads); otherwise all reads carry
// transaction id `trid` and the caller waits for them.
template <uint32_t DHt, uint32_t num_kv_heads, uint32_t block_size_t, uint32_t tile_bytes, uint32_t threshold,
          bool transpose, typename Reader>
inline void read_chunk(
    const Noc& noc, const Reader& reader, uint32_t write_ptr, uint32_t rows, uint32_t row0, uint32_t head,
    volatile tt_l1_ptr uint32_t* page_table, uint32_t& count, uint32_t trid = 0) {
    for (uint32_t row = 0; row < rows; ++row) {
        uint32_t tile = virtual_seq_tile_id_to_physical_tile_id<uint32_t, num_kv_heads, block_size_t, DHt, 0>(
            row0 + row, head, page_table);
        uint32_t ptr = transpose ? write_ptr + row * tile_bytes : write_ptr + row * DHt * tile_bytes;
        for (uint32_t col = 0; col < DHt; ++col) {
            if (trid != 0) {
                noc.async_read<NocOptions::TXN_ID>(
                    reader, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = tile}, {}, {.trid = trid});
            } else {
                noc.async_read(reader, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = tile}, {});
                if (++count == threshold) {
                    noc.async_read_barrier();
                    count = 0;
                }
            }
            tile += 1;
            ptr += transpose ? rows * tile_bytes : tile_bytes;
        }
    }
}

void kernel_main() {
    Noc noc;
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(4);
    constexpr uint32_t pos_stick_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(6);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(7);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(8);
    constexpr uint32_t slots = get_compile_time_arg_val(9);
    constexpr uint32_t barrier_threshold = get_compile_time_arg_val(10);
    constexpr uint32_t credit_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t arrived_sem_id = get_compile_time_arg_val(12);
    constexpr uint32_t kv_slots = get_compile_time_arg_val(13);  // chunk slots per K / V CB
    constexpr auto q_args = TensorAccessorArgs<14>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr uint32_t blocks = (T + verify_sdpa::kRowsPerTile - 1) / verify_sdpa::kRowsPerTile;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_writer_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_id_page_table = tt::CBIndex::c_9;
    constexpr uint32_t cb_compute_cur_pos = tt::CBIndex::c_15;

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_page_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_head = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slot = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    // NoC coordinates of this (KV head, member) core in every slot.
    tt_l1_ptr uint32_t* slot_xs = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    tt_l1_ptr uint32_t* slot_ys = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx + slots));

    // Positions of all rows -> writer (per-row masks and output rows); the representative
    // position of this slot's item -> compute (element `slot`, as reference B > 1).
    CircularBuffer cb_writer(cb_writer_cur_pos);
    cb_writer.reserve_back(1);
    const uint32_t writer_ptr = cb_writer.get_write_ptr();
    const auto pos_reader = TensorAccessor(pos_args, pos_addr);
    noc.async_read(pos_reader, CoreLocalMem<uint32_t>(writer_ptr), pos_stick_bytes, {.page_id = 0}, {});
    noc.async_read_barrier();
    const volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_ptr);
    const verify_sdpa::SlotWork work = verify_sdpa::slot_work<T, k_chunk_size>(positions, slot);
    const uint32_t cur_pos = work.representative;
    CircularBuffer cb_compute(cb_compute_cur_pos);
    cb_compute.reserve_back(1);
    volatile tt_l1_ptr uint32_t* compute_pos =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_compute.get_write_ptr());
    compute_pos[slot] = cur_pos;
    cb_writer.push_back(1);
    cb_compute.push_back(1);
    if (cur_pos == UINT32_MAX) {
        return;  // idle slot (compute and writer skip too)
    }

    const auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, 8>(cur_pos);
    const auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(cur_pos, slot, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic);
    if (k_chunk_start == k_chunk_end) {
        return;
    }

    constexpr uint32_t q_chunk_tiles = DHt;
    const uint32_t k_chunk_tiles = Sk_chunk_t_dynamic * DHt;
    const uint32_t v_chunk_tiles = Sk_chunk_t_dynamic * vDHt;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    {
        const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
        CircularBuffer cb_q(cb_q_in);
        cb_q.reserve_back(q_chunk_tiles);
        uint32_t q_write_ptr = cb_q.get_write_ptr();
        uint32_t q_tile_id = (cur_head * blocks + work.block) * DHt;
        for (uint32_t tile = 0; tile < q_chunk_tiles; ++tile) {
            noc.async_read(q_reader, CoreLocalMem<uint32_t>(q_write_ptr), q_tile_bytes, {.page_id = q_tile_id}, {});
            q_tile_id += 1;
            q_write_ptr += q_tile_bytes;
        }
        noc.async_read_barrier();
        cb_q.push_back(q_chunk_tiles);
    }

    const auto k_reader = TensorAccessor(k_args, k_addr);
    const auto v_reader = TensorAccessor(v_args, v_addr);
    // Every verification row shares one sequence page table: read row 0.
    CircularBuffer cb_page_table(cb_id_page_table);
    cb_page_table.reserve_back(1);
    volatile tt_l1_ptr uint32_t* page_table_ptr =
        read_page_table_for_batch(noc, cb_id_page_table, 0, page_table_args, page_table_addr, page_table_page_size);
    cb_page_table.push_back(1);

    CircularBuffer cb_k(cb_k_in);
    CircularBuffer cb_v(cb_v_in);
    uint32_t count = 0;
    const uint32_t chunks = k_chunk_end - k_chunk_start;
    const uint32_t k_bytes = k_chunk_tiles * k_tile_bytes;
    const uint32_t v_bytes = v_chunk_tiles * v_tile_bytes;
    const uint32_t k_base = cb_k.get_write_ptr();  // fresh CBs: slot 0
    const uint32_t v_base = cb_v.get_write_ptr();
    // Shared items: reads run `ahead` chunks in front of the exchange.
    constexpr uint32_t ahead = kv_slots > 2 ? kv_slots - 2 : 1;
    static_assert(ahead <= 3, "at most three chunks in flight");

    if (work.partner == verify_sdpa::kNoPartner) {
        // Unshared item: the reference read_k / read_v sequence and barrier cadence (other items
        // stream the same K/V concurrently, where deeper unthrottled reads measured slower).
        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            const uint32_t row0 = k_chunk * Sk_chunk_t_dynamic;
            cb_k.reserve_back(k_chunk_tiles);
            read_chunk<DHt, num_kv_heads, block_size_t, k_tile_bytes, barrier_threshold, true>(
                noc, k_reader, cb_k.get_write_ptr(), Sk_chunk_t_dynamic, row0, cur_head, page_table_ptr, count);
            noc.async_read_barrier();
            count = 0;
            cb_k.push_back(k_chunk_tiles);
            cb_v.reserve_back(v_chunk_tiles);
            read_chunk<vDHt, num_kv_heads, block_size_t, v_tile_bytes, barrier_threshold, false>(
                noc, v_reader, cb_v.get_write_ptr(), Sk_chunk_t_dynamic, row0, cur_head, page_table_ptr, count);
            noc.async_read_barrier();
            count = 0;
            cb_v.push_back(v_chunk_tiles);
        }
        return;
    }

    // Shared chunks: both partners run the same chunk sequence with identical CBs (KV_SLOTS chunk
    // slots each), so their write pointers agree and each copies its half into the partner's
    // matching slot. Semaphores count chunks: credit = partner reserved the slot, arrived =
    // partner's half landed. The DRAM reads of the next `ahead` chunks are in flight while chunk c
    // is exchanged.
    const bool reads_k = slot < work.partner;
    const uint32_t partner_x = slot_xs[work.partner];
    const uint32_t partner_y = slot_ys[work.partner];
    const uint32_t credit_addr = get_semaphore(credit_sem_id);
    const uint32_t arrived_addr = get_semaphore(arrived_sem_id);
    const uint64_t partner_credit = get_noc_addr(partner_x, partner_y, credit_addr);
    const uint64_t partner_arrived = get_noc_addr(partner_x, partner_y, arrived_addr);
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_addr);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrived_addr);
    const uint32_t half_bytes = reads_k ? k_bytes : v_bytes;

    // Exchange reads are tagged with transaction id 1 + index % 4.
    auto issue = [&](uint32_t index) {
        const uint32_t row0 = (k_chunk_start + index) * Sk_chunk_t_dynamic;
        const uint32_t ring = index % kv_slots;
        const uint32_t trid = 1 + (index & 3);
        if (reads_k) {
            read_chunk<DHt, num_kv_heads, block_size_t, k_tile_bytes, barrier_threshold, true>(
                noc, k_reader, k_base + ring * k_bytes, Sk_chunk_t_dynamic, row0, cur_head, page_table_ptr, count,
                trid);
        } else {
            read_chunk<vDHt, num_kv_heads, block_size_t, v_tile_bytes, barrier_threshold, false>(
                noc, v_reader, v_base + ring * v_bytes, Sk_chunk_t_dynamic, row0, cur_head, page_table_ptr, count,
                trid);
        }
    };

    // reserve_back(n) covers the n oldest unpushed slots; the K reader pushes K as soon as it lands.
    for (uint32_t index = 0; index < ahead && index < chunks; ++index) {
        cb_k.reserve_back((index + 1) * k_chunk_tiles);
        cb_v.reserve_back((index + 1) * v_chunk_tiles);
        noc_semaphore_inc(partner_credit, 1);
        issue(index);
    }
    for (uint32_t index = 0; index < chunks; ++index) {
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + (index & 3)});
        if (reads_k) {
            cb_k.push_back(k_chunk_tiles);  // compute starts QK while V is exchanged
        }
        const uint32_t ring = index % kv_slots;
        const uint32_t half_ptr = reads_k ? k_base + ring * k_bytes : v_base + ring * v_bytes;
        noc_semaphore_wait_min(credit, index + 1);
        noc_async_write(half_ptr, get_noc_addr(partner_x, partner_y, half_ptr), half_bytes);
        if (index + ahead < chunks) {
            cb_k.reserve_back((reads_k ? ahead : ahead + 1) * k_chunk_tiles);
            cb_v.reserve_back((ahead + 1) * v_chunk_tiles);
            noc_semaphore_inc(partner_credit, 1);
            issue(index + ahead);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(partner_arrived, 1);
        noc_semaphore_wait_min(arrived, index + 1);
        if (!reads_k) {
            cb_k.push_back(k_chunk_tiles);
        }
        cb_v.push_back(v_chunk_tiles);
    }
    noc_async_atomic_barrier();
}
