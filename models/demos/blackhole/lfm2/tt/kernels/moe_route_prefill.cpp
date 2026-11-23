// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 MoE prefill routing in one program: sigmoid + expert-bias top-k (bitwise sigmoid_routing.cpp, flags 3, through
// the decode router's exact score table) and the grouped-routing prefix (bitwise grouped_routing_prefix.cpp: same
// group map, experts metadata row and inverse rows), spread over every core instead of one.
//
// Phase 1 (worker w: tokens [w * T / W, (w + 1) * T / W)): route the tokens, count them per expert and bucket their
// (token, weight) entries by expert, all in this core's L1 (same address on every core); arrive on the master.
// The master (worker 0) waits for every worker and releases them.
// Phase 2: every worker reads all workers' counts (L1 -> L1) -> expert totals, group starts (expert order) and the
// rows each expert gets from earlier workers.  Token role: inverse[token][k] = group * 32 + row.  Expert role
// (worker e < E): gathers expert e's buckets from every worker in worker (= token) order into its group pages
// (32 token ids + 32 weights; padding rows 0xffffffff / 0).  The master also writes the metadata row.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#include "models/demos/blackhole/lfm2/tt/fast/kernels/sigmoid_math.hpp"

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t top_k = get_compile_time_arg_val(1);
    constexpr uint32_t tokens = get_compile_time_arg_val(2);
    constexpr uint32_t workers = get_compile_time_arg_val(3);
    constexpr uint32_t e_lo = get_compile_time_arg_val(4);
    constexpr uint32_t e_hi = get_compile_time_arg_val(5);
    constexpr uint32_t expert_width = get_compile_time_arg_val(6);
    constexpr uint32_t table_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t max_tokens = get_compile_time_arg_val(8);   // tokens of the largest worker share
    constexpr uint32_t max_groups = get_compile_time_arg_val(9);   // groups of one expert, upper bound
    constexpr auto logits_args = TensorAccessorArgs<10>();
    constexpr auto bias_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto map_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto experts_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    constexpr auto inverse_args = TensorAccessorArgs<experts_args.next_compile_time_args_offset()>();
    static_assert(E % 32 == 0 && E <= 64 && top_k <= 4 && tokens % 32 == 0);
    constexpr uint32_t columns = E / 32;
    constexpr uint32_t tile_bytes = 4096;  // FP32 32x32

    const auto logits = TensorAccessor(logits_args, get_arg_val<uint32_t>(0), tile_bytes);
    const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(1), E * 4);
    const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(2), table_bytes);
    const auto group_map = TensorAccessor(map_args, get_arg_val<uint32_t>(3), 256);
    const auto expert_ids = TensorAccessor(experts_args, get_arg_val<uint32_t>(4), expert_width * 4);
    const auto inverse = TensorAccessor(inverse_args, get_arg_val<uint32_t>(5), 128);
    const uint32_t worker = get_arg_val<uint32_t>(6);
    constexpr uint32_t xy_arg = 7;  // (x, y) of every worker, worker order
    const uint32_t arrive_sem = get_semaphore(0), go_sem = get_semaphore(1);
    auto core_addr = [&](uint32_t w, uint32_t addr) {
        return get_noc_addr(get_arg_val<uint32_t>(xy_arg + 2 * w), get_arg_val<uint32_t>(xy_arg + 2 * w + 1), addr);
    };

    // L1 layout (CB 0; identical on every core, so peers address each other's buffers directly)
    const uint32_t base = get_write_ptr(0);
    const uint32_t cnt_addr = base;                                        // E counts (this worker)
    const uint32_t bucket_addr = cnt_addr + E * 4;                         // top_k * max_tokens 16 B entries
    const uint32_t logits_addr = bucket_addr + top_k * max_tokens * 16;    // columns FP32 tiles
    const uint32_t table_addr = logits_addr + columns * tile_bytes;        // score table
    const uint32_t bias_addr = table_addr + ((table_bytes + 63) & ~63u);   // E int32 (Q24)
    const uint32_t row_addr = bias_addr + E * 4;                           // one token's logits
    const uint32_t rec_addr = row_addr + E * 4;                            // max_tokens x 4 route words
    const uint32_t all_cnt_addr = rec_addr + max_tokens * 16;              // workers x E counts
    const uint32_t start_addr = all_cnt_addr + workers * E * 4;            // first group of each expert
    const uint32_t pos_addr = start_addr + E * 4;                          // running row of each expert
    const uint32_t page_addr = pos_addr + E * 4;                           // inverse rows / group pages
    const uint32_t meta_addr = page_addr + ((max_tokens * 128 > max_groups * 256 ? max_tokens * 128 : max_groups * 256));
    auto* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_addr);
    auto* bucket = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bucket_addr);
    auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_addr);
    auto* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rec_addr);
    auto* all_cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(all_cnt_addr);
    auto* start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_addr);
    auto* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_addr);

    const uint32_t t0 = worker * tokens / workers, t1 = (worker + 1) * tokens / workers;
    const uint32_t n_tok = t1 - t0;

    // ---------------- phase 1: route this worker's tokens
    noc_async_read(get_noc_addr(0, table), table_addr, table_bytes);
    noc_async_read(get_noc_addr(0, bias), bias_addr, E * 4);
    for (uint32_t e = 0; e < E; ++e) cnt[e] = 0;
    uint32_t loaded = 0xffffffffu;
    auto* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(logits_addr);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t tr = t / 32, r = t % 32;
        if (tr != loaded) {
            for (uint32_t c = 0; c < columns; ++c) {
                noc_async_read(get_noc_addr(tr * columns + c, logits), logits_addr + c * tile_bytes, tile_bytes);
            }
            noc_async_read_barrier();
            loaded = tr;
        }
        for (uint32_t c = 0; c < columns; ++c) {
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t src = c * 1024 + (r / 16) * 512 + h * 256 + (r % 16) * 16;
                for (uint32_t j = 0; j < 16; ++j) row[c * 32 + h * 16 + j] = tile[src + j];
            }
        }
        uint32_t ids[top_k];
        uint16_t w[top_k];
        lfm_route<E, top_k, 3, e_lo, e_hi>(row, reinterpret_cast<volatile tt_l1_ptr int32_t*>(bias_addr),
                                           reinterpret_cast<volatile tt_l1_ptr uint16_t*>(table_addr), ids, w);
        for (uint32_t k = 0; k < top_k; ++k) {
            rec[(t - t0) * 4 + k] = ids[k] | (static_cast<uint32_t>(w[k]) << 16);
            cnt[ids[k]] = cnt[ids[k]] + 1;
        }
    }
    // bucket the entries by expert (token order inside each bucket)
    {
        uint32_t off = 0;
        for (uint32_t e = 0; e < E; ++e) {
            pos[e] = off;
            off += cnt[e];
        }
        for (uint32_t i = 0; i < n_tok; ++i) {
            for (uint32_t k = 0; k < top_k; ++k) {
                const uint32_t word = rec[i * 4 + k];
                const uint32_t e = word & 0xffffu;
                const uint32_t p = pos[e];
                pos[e] = p + 1;
                bucket[4 * p] = t0 + i;  // 16 B entries keep every NoC transfer 16 B aligned
                bucket[4 * p + 1] = word >> 16;
            }
        }
    }
    noc_semaphore_inc(core_addr(0, arrive_sem), 1);
    if (worker == 0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_sem), workers);
        for (uint32_t i = 0; i < workers; ++i) {
            noc_semaphore_inc(core_addr(i, go_sem), 1);
        }
        noc_async_atomic_barrier();
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), 1);

    // ---------------- phase 2
    for (uint32_t w = 0; w < workers; ++w) {
        noc_async_read(core_addr(w, cnt_addr), all_cnt_addr + w * E * 4, E * 4);
    }
    noc_async_read_barrier();
    uint32_t total_groups = 0;
    for (uint32_t e = 0; e < E; ++e) {
        uint32_t n = 0;
        for (uint32_t w = 0; w < workers; ++w) n += all_cnt[w * E + e];
        start[e] = total_groups;
        total_groups += (n + 31) / 32;
    }
    // token role: inverse rows of this worker's tokens
    if (n_tok != 0) {
        for (uint32_t e = 0; e < E; ++e) {
            uint32_t n = 0;
            for (uint32_t w = 0; w < worker; ++w) n += all_cnt[w * E + e];
            pos[e] = n;
        }
        auto* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_addr);
        for (uint32_t i = 0; i < n_tok; ++i) {
            for (uint32_t k = 0; k < 32; ++k) page[i * 32 + k] = 0;
            for (uint32_t k = 0; k < top_k; ++k) {
                const uint32_t e = rec[i * 4 + k] & 0xffffu;
                const uint32_t p = pos[e];
                pos[e] = p + 1;
                page[i * 32 + k] = (start[e] + p / 32) * 32 + p % 32;
            }
            noc_async_write(page_addr + i * 128, get_noc_addr(t0 + i, inverse), 128);
        }
        noc_async_write_barrier();
    }
    // expert role: group pages of expert `worker`
    if (worker < E) {
        const uint32_t e = worker;
        auto* group = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_addr);
        // entries land as (token, weight) pairs at the expert's running row; repacked into pages below
        const uint32_t gather_addr = meta_addr + ((expert_width * 4 + 63) & ~63u);
        auto* gathered = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_addr);
        uint32_t n = 0;
        for (uint32_t w = 0; w < workers; ++w) {
            const uint32_t c = all_cnt[w * E + e];
            if (c == 0) continue;
            uint32_t before = 0;
            for (uint32_t f = 0; f < e; ++f) before += all_cnt[w * E + f];
            noc_async_read(core_addr(w, bucket_addr + before * 16), gather_addr + n * 16, c * 16);
            n += c;
        }
        noc_async_read_barrier();
        const uint32_t groups = (n + 31) / 32;
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t i = g * 32 + r;
                group[g * 64 + r] = i < n ? gathered[4 * i] : 0xffffffffu;
                group[g * 64 + 32 + r] = i < n ? gathered[4 * i + 1] : 0;
            }
            noc_async_write(page_addr + g * 256, get_noc_addr(start[e] + g, group_map), 256);
        }
        noc_async_write_barrier();
    }
    // master: metadata row
    if (worker == 0) {
        auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_addr);
        for (uint32_t i = 0; i < expert_width; ++i) meta[i] = 0;
        meta[0] = total_groups;
        uint32_t entry = 1;
        for (uint32_t e = 0; e < E; ++e) {
            const uint32_t n = (e + 1 < E ? start[e + 1] : total_groups) - start[e];
            for (uint32_t j = 0; j < n; ++j) meta[entry++] = e;
        }
        noc_async_write(meta_addr, get_noc_addr(0, expert_ids), expert_width * 4);
        noc_async_write_barrier();
    }
}
