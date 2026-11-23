// SPDX-License-Identifier: Apache-2.0
// Verify-only copy of the recurrent reader: all rows' q/k are gathered once, normalized by
// compute as one multi-row tile set, and each row is handed back to compute in row 0.
// BATCHED_USERS: every row is a different user. Before each row the reader loads that user's
// TILE state [1,48,128,128] (16 tiles per head, DRAM) into cb_state; compute does not carry state
// between rows and the writer stores each row's new state back to the same user.
// GROUP_ROWS (= slot count): shared verify rounds. A row map (one word per row: active bit 31,
// slot << 16, group length << 8, offset in the group) groups consecutive rows by user: the first
// row of a group loads its slot's DRAM TILE state, later rows chain from the previous row as the
// single-user verify does. Each row's flags (bit 0 fresh state, bit 1 the next row continues the
// group) travel to compute in word 8 of its cb_g tile (rows >= 1 of that tile are otherwise zero
// and unused by the scalar broadcast). Inactive rows read candidate row t (never a user's state).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4, cb_state = 5;
constexpr uint32_t cb_decay_bias = 6, cb_decay_scale = 7, cb_ones = 26;
constexpr uint32_t cb_qall = 8, cb_kall = 9, cb_qnall = 10, cb_knall = 11, cb_stage = 12;
constexpr uint32_t cb_gall = 13, cb_ball = 14, cb_biasall = 15, cb_dall = 30, cb_sall = 31;

void kernel_main() {
#ifdef VERIFY_TOKENS
    constexpr uint32_t num_tokens = VERIFY_TOKENS;
    static_assert(num_tokens >= 2 && num_tokens <= 32);
#else
    constexpr uint32_t num_tokens = 2;
#endif
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr auto q_a = TensorAccessorArgs<5>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto a_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto db_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto ds_a = TensorAccessorArgs<db_a.next_compile_time_args_offset()>();
    constexpr auto ones_a = TensorAccessorArgs<ds_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t a_addr = get_arg_val<uint32_t>(4);
    const uint32_t b_addr = get_arg_val<uint32_t>(5);
    const uint32_t s_addr = get_arg_val<uint32_t>(6);
    const uint32_t db_addr = get_arg_val<uint32_t>(7);
    const uint32_t ds_addr = get_arg_val<uint32_t>(8);
    const uint32_t ones_addr = get_arg_val<uint32_t>(9);
    const uint32_t BH = get_arg_val<uint32_t>(10);

    const uint32_t tile_bytes = get_tile_size(cb_q);
    constexpr uint32_t state_stick_bytes = Vt * 32 * sizeof(uint16_t);
    const auto q_acc = TensorAccessor(q_a, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_a, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_a, v_addr, tile_bytes);
    const auto a_acc = TensorAccessor(a_a, a_addr, tile_bytes);
    const auto b_acc = TensorAccessor(b_a, b_addr, tile_bytes);
    const auto s_acc = TensorAccessor(s_a, s_addr, state_stick_bytes);
    const auto db_acc = TensorAccessor(db_a, db_addr, tile_bytes);
    const auto ds_acc = TensorAccessor(ds_a, ds_addr, tile_bytes);
    const auto ones_acc = TensorAccessor(ones_a, ones_addr, tile_bytes);
    Noc noc;

    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t count) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(count);
        for (uint32_t tile = 0; tile < count; ++tile) {
            noc.async_read(
                acc,
                cb,
                tile_bytes,
                {.page_id = base + tile},
                {.offset_bytes = tile * tile_bytes});
        }
        noc.async_read_barrier();
        cb.push_back(count);
    };

    auto read_state_into_tiles = [&]() {
        constexpr uint32_t tile_width = 32;
        constexpr uint32_t face_width = 16;
        constexpr uint32_t face_row_bytes = face_width * sizeof(uint16_t);
        constexpr uint32_t face_bytes = face_width * face_width * sizeof(uint16_t);
        CircularBuffer cb(cb_state);
        cb.reserve_back(Kt * Vt);
        for (uint32_t tile_row = 0; tile_row < Kt; ++tile_row) {
            for (uint32_t row = 0; row < tile_width; ++row) {
                const uint32_t state_row = h * Kt * tile_width + tile_row * tile_width + row;
                const uint32_t face_row = row % face_width;
                const uint32_t face_row_group = row / face_width;
                for (uint32_t tile_col = 0; tile_col < Vt; ++tile_col) {
                    const uint32_t tile_offset = (tile_row * Vt + tile_col) * tile_bytes;
                    for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                        const uint32_t source_offset =
                            (tile_col * tile_width + face_col * face_width) * sizeof(uint16_t);
                        const uint32_t face = face_row_group * 2 + face_col;
                        const uint32_t dest_offset =
                            tile_offset + face * face_bytes + face_row * face_row_bytes;
                        noc.async_read(
                            s_acc,
                            cb,
                            face_row_bytes,
                            {.page_id = state_row, .offset_bytes = source_offset},
                            {.offset_bytes = dest_offset});
                    }
                }
            }
        }
        noc.async_read_barrier();
        cb.push_back(Kt * Vt);
    };

    // q/k/v are [T, 1, BH, D] TILE, with ceil(BH / 32) tile rows per
    // token. a/b are [1, 1, T, BH], with the head on the tiled column.
    // Gather each head into row 0 so compute retains the T=1 arithmetic.
    constexpr uint32_t face_width = 16;
    constexpr uint32_t face_row_bytes = face_width * sizeof(uint16_t);
    constexpr uint32_t face_bytes = face_width * face_width * sizeof(uint16_t);
    auto row_offset = [](uint32_t row, uint32_t face_col) {
        return ((row / face_width) * 2 + face_col) * face_bytes + (row % face_width) * face_row_bytes;
    };
    // CB memory persists across programs, so every gathered tile is fully zeroed
    // before its row lands; compute treats these as one-row-per-tile operands.
    // Blackhole DRAM reads need 64-byte aligned source offsets; a 32-byte face
    // row at an odd row index is misaligned. Read the aligned 64-byte row pair
    // for both face columns into a scratch region of the destination tile
    // (face 2, bytes 1024..1151, otherwise zero), move the wanted row into row 0,
    // then re-zero the scratch. Issue all reads for a CB before one barrier.
    constexpr uint32_t pair_bytes = 2 * face_row_bytes;
    constexpr uint32_t scratch_bytes = 2 * face_bytes;  // face 2 start
    auto issue_row_reads = [&](const auto& acc, CircularBuffer& cb, uint32_t page, uint32_t src_row, uint32_t tile_offset) {
        for (uint32_t face_col = 0; face_col < 2; ++face_col) {
            const uint32_t src = (row_offset(src_row, face_col) / pair_bytes) * pair_bytes;
            noc.async_read(
                acc,
                cb,
                pair_bytes,
                {.page_id = page, .offset_bytes = src},
                {.offset_bytes = tile_offset + scratch_bytes + face_col * pair_bytes});
        }
    };
    auto settle_row = [&](CircularBuffer& cb, uint32_t src_row, uint32_t tile_offset) {
        auto* tile = reinterpret_cast<volatile uint64_t*>(cb.get_write_ptr() + tile_offset);
        const uint32_t pick = (src_row & 1) * (face_row_bytes / sizeof(uint64_t));
        for (uint32_t face_col = 0; face_col < 2; ++face_col) {
            const uint32_t src_w = (scratch_bytes + face_col * pair_bytes) / sizeof(uint64_t) + pick;
            const uint32_t dst_w = row_offset(0, face_col) / sizeof(uint64_t);
            tile[dst_w] = tile[src_w];
            tile[dst_w + 1] = tile[src_w + 1];
            tile[dst_w + 2] = tile[src_w + 2];
            tile[dst_w + 3] = tile[src_w + 3];
        }
        for (uint32_t i = 0; i < 2 * pair_bytes / sizeof(uint64_t); ++i) {
            tile[scratch_bytes / sizeof(uint64_t) + i] = 0;
        }
    };

    // Load the persistent state and per-head front-end constants once. Keeping
    // these resident also lets the one-tile input CBs stream t=0 then t=1
    // without the reader blocking before compute can start.
#ifdef VERIFY_BULK_STATE
    // The paired output worker bulk-reads this head's 128 contiguous state rows,
    // tilizes them as UInt16 words (bit-preserving) and writes the 16 tiles
    // straight into the reserved state CB while the operands below are gathered.
    static_assert(Kt == 4 && Vt == 4, "Bulk state requires 128x128 heads");
    const uint32_t ox = get_arg_val<uint32_t>(11);
    const uint32_t oy = get_arg_val<uint32_t>(12);
    {
        CircularBuffer cb(cb_state);
        cb.reserve_back(Kt * Vt);
        const uint32_t destination = cb.get_write_ptr();
        // Scratch word inside the reserved, unpublished state CB; the NoC write
        // completes before the output worker may overwrite it.
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination) = destination;
        noc_async_write(destination, get_noc_addr(ox, oy, get_semaphore(5)), sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(3)), 1);
    }
#elif defined(BATCHED_USERS)
    // Per-user state accessors share one layout; the per-row base addresses follow the
    // regular runtime arguments (one per row). Row 0's state is read below, in the row loop.
    constexpr auto user_state_args = TensorAccessorArgs<ones_a.next_compile_time_args_offset()>();
#elif defined(GROUP_ROWS)
    // Runtime args 11.. : GROUP_ROWS slot state addresses, the candidate tensor [T,48,128,128]
    // (same interleaved DRAM TILE layout), the row map [1,32] uint32.
    constexpr auto group_state_args = TensorAccessorArgs<ones_a.next_compile_time_args_offset()>();
    const uint32_t candidate_addr = get_arg_val<uint32_t>(11 + GROUP_ROWS);
    uint32_t row_map[num_tokens];
    {
        // The staging CB is reader-private scratch (reserved, never published); the row map lands
        // past the q/k/a/b staging area (bytes 0..17920).
        const uint32_t map_l1 = CircularBuffer(cb_stage).get_write_ptr() + 19456;
        const auto map = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_arg_val<uint32_t>(12 + GROUP_ROWS), 128);
        noc_async_read(get_noc_addr(0, map), map_l1, 128);
        noc_async_read_barrier();
        for (uint32_t t = 0; t < num_tokens; ++t) {
            row_map[t] = reinterpret_cast<volatile uint32_t*>(map_l1)[t];
        }
    }
#else
    read_state_into_tiles();
#endif
    read_into(db_acc, cb_decay_bias, h, 1);
    read_into(ds_acc, cb_decay_scale, h, 1);
    read_into(ones_acc, cb_ones, 0, 1);

    const uint32_t head_tiles = (BH + 31) / 32;
    const uint32_t head_tile = h / 32;
    const uint32_t head_row = h % 32;
    CircularBuffer q_cb(cb_q), k_cb(cb_k), v_cb(cb_v), g_cb(cb_g), beta_cb(cb_beta);
    // Every row's q/k is gathered once into multi-row tiles (row t = token t) that compute
    // normalizes in one pass (row-independent arithmetic). Row t of the normalized tiles
    // is then copied into row 0 of zeroed per-token operands, exactly the one-row layout
    // that per-token normalization produced.
    static_assert(num_tokens <= 16, "multi-row q/k staging holds 16 rows");
    {
        CircularBuffer qa(cb_qall), ka(cb_kall), stage(cb_stage);
        qa.reserve_back(Kt);
        ka.reserve_back(Kt);
        // Every row's g/beta (row t, column 0) and the decay bias replicated per row
        // feed compute's one-pass decay / sigmoid(beta) for all rows.
        CircularBuffer ga(cb_gall), ba(cb_ball), biasa(cb_biasall);
        stage.reserve_back(10);
        ga.reserve_back(1);
        ba.reserve_back(1);
        biasa.reserve_back(1);
        noc.async_write_zeros(qa, Kt * tile_bytes);
        noc.async_write_zeros(ka, Kt * tile_bytes);
        noc.async_write_zeros(ga, tile_bytes);
        noc.async_write_zeros(ba, tile_bytes);
        noc.write_zeros_l1_barrier();
        for (uint32_t token = 0; token < num_tokens; ++token) {
            const uint32_t qk_page = (token * head_tiles + head_tile) * Kt;
            for (uint32_t tile = 0; tile < Kt; ++tile) {
                for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                    const uint32_t src = (row_offset(head_row, face_col) / pair_bytes) * pair_bytes;
                    const uint32_t slot = (token * Kt + tile) * 2 * pair_bytes + face_col * pair_bytes;
                    noc.async_read(q_acc, stage, pair_bytes, {.page_id = qk_page + tile, .offset_bytes = src},
                                   {.offset_bytes = slot});
                    noc.async_read(k_acc, stage, pair_bytes, {.page_id = qk_page + tile, .offset_bytes = src},
                                   {.offset_bytes = 8192 + slot});
                }
            }
            // a/b are [1, 1, T, BH]: row `token`, column head_row of tile head_tile.
            const uint32_t scalar_src = (row_offset(token, head_row / face_width) / pair_bytes) * pair_bytes;
            noc.async_read(a_acc, stage, pair_bytes, {.page_id = head_tile, .offset_bytes = scalar_src},
                           {.offset_bytes = 16384 + token * pair_bytes});
            noc.async_read(b_acc, stage, pair_bytes, {.page_id = head_tile, .offset_bytes = scalar_src},
                           {.offset_bytes = 17408 + token * pair_bytes});
        }
        noc.async_read(db_acc, biasa, tile_bytes, {.page_id = h}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t pick = (head_row & 1) * (face_row_bytes / sizeof(uint64_t));
        auto* staged = reinterpret_cast<volatile uint64_t*>(stage.get_write_ptr());
        auto* qw = reinterpret_cast<volatile uint64_t*>(qa.get_write_ptr());
        auto* kw = reinterpret_cast<volatile uint64_t*>(ka.get_write_ptr());
        for (uint32_t token = 0; token < num_tokens; ++token) {
            for (uint32_t tile = 0; tile < Kt; ++tile) {
                for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                    const uint32_t src_w = ((token * Kt + tile) * 2 * pair_bytes + face_col * pair_bytes) / sizeof(uint64_t) + pick;
                    const uint32_t dst_w = (tile * tile_bytes + row_offset(token, face_col)) / sizeof(uint64_t);
                    for (uint32_t w = 0; w < 4; ++w) {
                        qw[dst_w + w] = staged[src_w + w];
                        kw[dst_w + w] = staged[8192 / sizeof(uint64_t) + src_w + w];
                    }
                }
            }
        }
        auto* staged16 = reinterpret_cast<volatile uint16_t*>(stage.get_write_ptr());
        auto* g16 = reinterpret_cast<volatile uint16_t*>(ga.get_write_ptr());
        auto* b16 = reinterpret_cast<volatile uint16_t*>(ba.get_write_ptr());
        auto* bias16 = reinterpret_cast<volatile uint16_t*>(biasa.get_write_ptr());
        for (uint32_t token = 0; token < num_tokens; ++token) {
            const uint32_t element = (token & 1) * face_width + head_row % face_width;
            const uint32_t dst = row_offset(token, 0) / sizeof(uint16_t);
            g16[dst] = staged16[(16384 + token * pair_bytes) / sizeof(uint16_t) + element];
            b16[dst] = staged16[(17408 + token * pair_bytes) / sizeof(uint16_t) + element];
            bias16[dst] = bias16[0];
        }
        qa.push_back(Kt);
        ka.push_back(Kt);
        ga.push_back(1);
        ba.push_back(1);
        biasa.push_back(1);
    }
    CircularBuffer qn_cb(cb_qnall), kn_cb(cb_knall), dn_cb(cb_dall), sn_cb(cb_sall);
    // Element (token, 0) of a one-tile all-rows result into (0, 0) of a zeroed operand.
    auto hand_scalar = [&](CircularBuffer& all, CircularBuffer& one, uint32_t token) {
        const auto* src = reinterpret_cast<volatile uint16_t*>(all.get_read_ptr());
        auto* dst = reinterpret_cast<volatile uint16_t*>(one.get_write_ptr());
        dst[0] = src[row_offset(token, 0) / sizeof(uint16_t)];
    };
    auto hand_row = [&](CircularBuffer& all, CircularBuffer& one, uint32_t token) {
        auto* src = reinterpret_cast<volatile uint64_t*>(all.get_read_ptr());
        auto* dst = reinterpret_cast<volatile uint64_t*>(one.get_write_ptr());
        for (uint32_t tile = 0; tile < Kt; ++tile) {
            for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                const uint32_t s = (tile * tile_bytes + row_offset(token, face_col)) / sizeof(uint64_t);
                const uint32_t d = (tile * tile_bytes + row_offset(0, face_col)) / sizeof(uint64_t);
                for (uint32_t w = 0; w < 4; ++w) {
                    dst[d + w] = src[s + w];
                }
            }
        }
    };
    for (uint32_t token = 0; token < num_tokens; ++token) {
        #ifdef TREE_FORK_POSITION
        if (token == 8) {
            auto* released = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
            noc_semaphore_wait(released, 1);
            noc_semaphore_set(released, 0);
            constexpr auto parent_args = TensorAccessorArgs<ones_a.next_compile_time_args_offset()>();
            const auto parent = TensorAccessor(parent_args, get_arg_val<uint32_t>(11), tile_bytes);
            read_into(parent, cb_state, h * Kt * Vt, Kt * Vt);
        }
        #endif
        #ifdef BATCHED_USERS
        {
            // Blocks until compute has released the previous row's state (one 16-tile buffer).
            const auto user_state = TensorAccessor(user_state_args, get_arg_val<uint32_t>(11 + token), tile_bytes);
            read_into(user_state, cb_state, h * Kt * Vt, Kt * Vt);
        }
        #endif
        #ifdef GROUP_ROWS
        const uint32_t word = row_map[token];
        const bool active = (word >> 31) != 0;
        const uint32_t offset = word & 0xff, length = (word >> 8) & 0xff, slot = (word >> 16) & 0xff;
        const bool fresh = token == 0 || !active || offset == 0;
        const bool carry = active && offset + 1 < length && token + 1 < num_tokens;
        if (fresh) {
            // Blocks until compute has released the previous group's state (one 16-tile buffer). An
            // inactive row computes from slot 0's state (read only; its result is discarded).
            read_into(TensorAccessor(group_state_args, get_arg_val<uint32_t>(11 + (active ? slot : 0)), tile_bytes),
                      cb_state, h * Kt * Vt, Kt * Vt);
        }
        #endif
        // One zero fill and one DRAM round trip per token for all five operands:
        // the same bytes as five sequential read_row_into/read_scalar_into calls.
        q_cb.reserve_back(Kt);
        k_cb.reserve_back(Kt);
        v_cb.reserve_back(Vt);
        g_cb.reserve_back(1);
        beta_cb.reserve_back(1);
        noc.async_write_zeros(q_cb, Kt * tile_bytes);
        noc.async_write_zeros(k_cb, Kt * tile_bytes);
        noc.async_write_zeros(v_cb, Vt * tile_bytes);
        noc.async_write_zeros(g_cb, tile_bytes);
        noc.async_write_zeros(beta_cb, tile_bytes);
        noc.write_zeros_l1_barrier();
        const uint32_t v_page = (token * head_tiles + head_tile) * Vt;
        for (uint32_t tile = 0; tile < Vt; ++tile) {
            issue_row_reads(v_acc, v_cb, v_page + tile, head_row, tile * tile_bytes);
        }
        if (token == 0) {
            qn_cb.wait_front(Kt);
            kn_cb.wait_front(Kt);
            dn_cb.wait_front(1);
            sn_cb.wait_front(1);
        }
        hand_row(qn_cb, q_cb, token);
        hand_row(kn_cb, k_cb, token);
        hand_scalar(dn_cb, g_cb, token);
        hand_scalar(sn_cb, beta_cb, token);
        #ifdef GROUP_ROWS
        reinterpret_cast<volatile uint32_t*>(g_cb.get_write_ptr())[8] = (fresh ? 1u : 0u) | (carry ? 2u : 0u);
        #endif
        noc.async_read_barrier();
        for (uint32_t tile = 0; tile < Vt; ++tile) {
            settle_row(v_cb, head_row, tile * tile_bytes);
        }
        q_cb.push_back(Kt);
        k_cb.push_back(Kt);
        v_cb.push_back(Vt);
        g_cb.push_back(1);
        beta_cb.push_back(1);
#ifdef VERIFY_BULK_STATE
        if (token == 0) {
            auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
            noc_semaphore_wait(ready, 1);
            noc_semaphore_set(ready, 0);
            CircularBuffer(cb_state).push_back(Kt * Vt);
        }
#endif
    }
    qn_cb.pop_front(Kt);
    kn_cb.pop_front(Kt);
    dn_cb.pop_front(1);
    sn_cb.pop_front(1);
    // Drain the state-transport increment before exit (see fused_verify_writer.cpp).
    noc_async_atomic_barrier();
}
