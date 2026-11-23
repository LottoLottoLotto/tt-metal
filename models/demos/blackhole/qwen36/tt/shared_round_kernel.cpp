// SPDX-License-Identifier: Apache-2.0
// Shared verify rounds: row tokens and per-group acceptance.
//
// SR_MODE 0 (110 cores x BRISC/NCRISC): per-row argmax of tiled BF16 logits rows [0, SR_ROWS) with ttnn.argmax's
//   order (verify_accept_kernel.cpp: bfloat16_greater on the bit pattern, equal bits keep the lower index), one
//   (key, index) record per row and worker.
// SR_MODE 1 (one core): row tokens t_r (the merged argmax, or with SR_SAMPLED the device sampler's draws), then per
//   group g (row map: rows r0 .. r0 + L - 1 of slot s; row 0 the pending token, rows 1.. the drafts) the accepted
//   count a = the longest prefix with t_{r0+j} == token_{r0+j+1} (j < L - 1; a row flagged for a host redraw ends
//   the group there). The group emits t_{r0} .. t_{r0+a}. Outputs:
//     result [1,64] int32: words 0..15 t_r; 16 + r0: a + 1 at each group's first row; 32 + r0: 1 if row r0 + a
//       needs a host redraw;
//     commit [1,32] uint32: count, then (slot, r0 + a) of every group with L > 1 (gdn/group_commit.py);
//     ring [1,32] uint32: row r -> bit 31 | slot << 16 when row r's input was accepted (r - r0 <= a): its target
//       features enter that slot's drafter context at the row's position;
//     drafter scalars: every slot of the round gets anchor_s = t_{r0+a} and position_s = p_{r0} + a + 1 ([1,1]
//       int32 each, the next draft block), and ring slot s < SR_RINGS its owners[p % 2048] = p for p in p_{r0} ..
//       p_{r0} + a (the drafter's committed context; the K/V rows follow in dflash_cache commit_round);
//   with SR_SAMPLED, the sampler history of every active group's slot (parameter row r0, word 11 < SR_SLOTS) counts
//   its emitted tokens (the verify rows ran with F_NOCOMMIT), unless the group ends in a host redraw (the host
//   re-syncs).
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

namespace {
constexpr uint32_t rows = SR_ROWS;
constexpr uint32_t mode = SR_MODE;
constexpr uint32_t workers = SR_WORKERS;
constexpr uint32_t vocab_tiles = SR_VOCAB_TILES;
constexpr uint32_t row_bytes = rows * 32;  // live rows of one 16-column face
constexpr uint32_t record_bytes = 128;     // (key, index) for up to 16 rows

inline uint32_t key(uint32_t bits) { return bits & 0x8000 ? (~bits & 0xffff) : (bits | 0x8000); }

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
inline void scan(uint32_t worker, uint32_t logits_address, uint32_t scratch, uint32_t partial_address) {
    const auto logits = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), logits_address, 2048);
    const auto partial = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), partial_address, record_bytes);
    const uint32_t slot[2] = {scratch, scratch + 2 * row_bytes};
    auto record = reinterpret_cast<volatile uint32_t*>(scratch + 4 * row_bytes);
    uint32_t best_key[rows], best_id[rows];
    for (uint32_t r = 0; r < rows; ++r) {
        best_key[r] = key(0xff80);
        best_id[r] = 0;
    }
    auto fetch = [&](uint32_t tile, uint32_t destination) {
        // Only the live rows of faces 0 and 1 (rows < 16 never touch faces 2 and 3).
        noc_async_read(logits.get_noc_addr(tile, 0), destination, row_bytes);
        noc_async_read(logits.get_noc_addr(tile, 512), destination + row_bytes, row_bytes);
    };
    uint32_t current = 0;
    if (worker < vocab_tiles) fetch(worker, slot[0]);
    for (uint32_t tile = worker; tile < vocab_tiles; tile += workers) {
        noc_async_read_barrier();
        if (tile + workers < vocab_tiles) fetch(tile + workers, slot[current ^ 1]);
        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t k = best_key[r], id = best_id[r];
            for (uint32_t face = 0; face < 2; ++face) {
                auto words = reinterpret_cast<volatile uint32_t*>(slot[current] + face * row_bytes + r * 32);
                const uint32_t base = tile * 32 + face * 16;
                for (uint32_t w = 0; w < 8; ++w) {
                    const uint32_t word = words[w];
                    if (k & 0x8000) {
                        // Positive best b: a half beats it only if it is positive and its bits exceed b.
                        const uint32_t c = (0x7fff - (k & 0x7fff)) * 0x00010001u;
                        if ((((word & 0x7fff7fffu) + c) & ~word & 0x80008000u) == 0) continue;
                    }
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t candidate = key((word >> (16 * half)) & 0xffff);
                        const uint32_t index = base + 2 * w + half;
                        if (candidate > k || (candidate == k && index < id)) {
                            k = candidate;
                            id = index;
                        }
                    }
                }
            }
            best_key[r] = k;
            best_id[r] = id;
        }
        current ^= 1;
    }
    noc_async_read_barrier();
    for (uint32_t r = 0; r < rows; ++r) {
        record[2 * r] = best_key[r];
        record[2 * r + 1] = best_id[r];
    }
    noc_async_write(scratch + 4 * row_bytes, partial.get_noc_addr(worker), record_bytes);
    noc_async_write_barrier();
}
#endif
}  // namespace

void kernel_main() {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    const uint32_t task = get_arg_val<uint32_t>(0);
    if constexpr (mode == 0) {
#if defined(COMPILE_FOR_BRISC)
        scan(2 * task, get_common_arg_val<uint32_t>(0), get_write_ptr(0), get_common_arg_val<uint32_t>(1));
#elif defined(COMPILE_FOR_NCRISC)
        scan(2 * task + 1, get_common_arg_val<uint32_t>(0), get_write_ptr(1), get_common_arg_val<uint32_t>(1));
#endif
    } else {
#if defined(COMPILE_FOR_BRISC)
        // Common args: source (argmax partial records, or the sampler's out [16,4] with SR_SAMPLED), tokens [1,W]
        // u32, positions [W] i32, row map [1,32] u32, result, commit, ring, SR_SLOTS anchor scalars, SR_SLOTS
        // position scalars (0: no drafter), SR_RINGS ring owners, then with SR_SAMPLED the sampler params [32,32]
        // and SR_SLOTS history tensors.
        constexpr uint32_t slots = SR_SLOTS;
        const uint32_t scratch = get_write_ptr(0);
        auto dram = [](uint32_t address, uint32_t page) {
            return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), address, page);
        };
        const uint32_t io = scratch + 16384;
        auto token = reinterpret_cast<volatile uint32_t*>(io);
        auto position = reinterpret_cast<volatile int32_t*>(io + 128);
        auto map = reinterpret_cast<volatile uint32_t*>(io + 256);
        auto result = reinterpret_cast<volatile int32_t*>(io + 384);    // 64 words
        auto commit = reinterpret_cast<volatile uint32_t*>(io + 640);   // 32 words
        auto ring = reinterpret_cast<volatile uint32_t*>(io + 768);     // 32 words
        const uint32_t page_l1 = io + 896;                               // 2 x 64 B scalar page staging
        const uint32_t hist_l1 = io + 1024;                              // 128 B history page staging
        const uint32_t params_l1 = io + 1152;                            // 32 x 128 B parameter rows
        const uint32_t owners_l1 = io + 5248;                            // 8 KB ring owners
        constexpr uint32_t rings = SR_RINGS;
        constexpr uint32_t sampled_base = 7 + 2 * slots + rings;
        noc_async_read(dram(get_common_arg_val<uint32_t>(1), rows * 4).get_noc_addr(0), io, rows * 4);
        noc_async_read(dram(get_common_arg_val<uint32_t>(2), rows * 4).get_noc_addr(0), io + 128, rows * 4);
        noc_async_read(dram(get_common_arg_val<uint32_t>(3), 128).get_noc_addr(0), io + 256, 128);
        uint32_t target[rows];
        bool redraw[rows];
#ifdef SR_SAMPLED
        // Sampler out [16,4] int32 (one 16-byte page per row, staged 64-byte aligned): token, flags (bit 0: truncated
        // tie group, host redraw); the round rows' parameter rows (one 128-byte page each).
        const auto out_pages = dram(get_common_arg_val<uint32_t>(0), 16);
        const auto param_pages = dram(get_common_arg_val<uint32_t>(sampled_base), 128);
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_read(out_pages.get_noc_addr(r), scratch + 64 * r, 16);
            noc_async_read(param_pages.get_noc_addr(r), params_l1 + 128 * r, 128);
        }
        noc_async_read_barrier();
        for (uint32_t r = 0; r < rows; ++r) {
            auto out = reinterpret_cast<volatile int32_t*>(scratch + 64 * r);
            target[r] = static_cast<uint32_t>(out[0]);
            redraw[r] = (out[1] & 1) != 0;
        }
#else
        const auto partial = dram(get_common_arg_val<uint32_t>(0), record_bytes);
        for (uint32_t w = 0; w < workers; ++w) noc_async_read(partial.get_noc_addr(w), scratch + w * 64, 64);
        noc_async_read_barrier();
        // Records are 128 B; rows 0..7 sit in the first 64 B of each. Re-read the second halves for rows >= 8.
        uint32_t best_key[rows], best_id[rows];
        for (uint32_t r = 0; r < rows; ++r) {
            best_key[r] = key(0xff80);
            best_id[r] = 0;
        }
        for (uint32_t half = 0; half < (rows + 7) / 8; ++half) {
            if (half) {
                for (uint32_t w = 0; w < workers; ++w) {
                    noc_async_read(partial.get_noc_addr(w, 64), scratch + w * 64, 64);
                }
                noc_async_read_barrier();
            }
            auto records = reinterpret_cast<volatile uint32_t*>(scratch);
            for (uint32_t r = 8 * half; r < rows && r < 8 * half + 8; ++r) {
                for (uint32_t w = 0; w < workers; ++w) {
                    const uint32_t candidate = records[w * 16 + 2 * (r - 8 * half)];
                    const uint32_t index = records[w * 16 + 2 * (r - 8 * half) + 1];
                    if (candidate > best_key[r] || (candidate == best_key[r] && index < best_id[r])) {
                        best_key[r] = candidate;
                        best_id[r] = index;
                    }
                }
            }
        }
        for (uint32_t r = 0; r < rows; ++r) {
            target[r] = best_id[r];
            redraw[r] = false;
        }
#endif
        for (uint32_t i = 0; i < 64; ++i) result[i] = 0;
        for (uint32_t i = 0; i < 32; ++i) commit[i] = ring[i] = 0;
        for (uint32_t r = 0; r < rows; ++r) result[r] = static_cast<int32_t>(target[r]);
        const bool drafter = get_common_arg_val<uint32_t>(7) != 0;
        uint32_t committed = 0;
        for (uint32_t r = 0; r < rows;) {
            const uint32_t word = map[r];
            if ((word >> 31) == 0) {
                ++r;
                continue;
            }
            const uint32_t length = (word >> 8) & 0xff, slot = (word >> 16) & 0xff;
            uint32_t a = 0;
            while (a + 1 < length && r + a + 1 < rows && !redraw[r + a] && target[r + a] == token[r + a + 1]) ++a;
            result[16 + r] = static_cast<int32_t>(a + 1);
            result[32 + r] = redraw[r + a] ? 1 : 0;
            if (length > 1 && committed < 8) {
                commit[1 + 2 * committed] = slot;
                commit[2 + 2 * committed] = r + a;
                ++committed;
            }
            for (uint32_t j = 0; j <= a; ++j) ring[r + j] = (1u << 31) | (slot << 16);
            if (drafter) {
                auto page = reinterpret_cast<volatile int32_t*>(page_l1);
                for (uint32_t i = 0; i < 32; ++i) page[i] = 0;
                page[0] = static_cast<int32_t>(target[r + a]);
                page[16] = position[r] + static_cast<int32_t>(a) + 1;
                noc_async_write(page_l1, dram(get_common_arg_val<uint32_t>(7 + slot), 32).get_noc_addr(0), 32);
                noc_async_write(page_l1 + 64, dram(get_common_arg_val<uint32_t>(7 + slots + slot), 32).get_noc_addr(0), 32);
                noc_async_write_barrier();
            }
            if (slot < rings) {
                const auto owners = dram(get_common_arg_val<uint32_t>(7 + 2 * slots + slot), 2048 * 4);
                noc_async_read(owners.get_noc_addr(0), owners_l1, 2048 * 4);
                noc_async_read_barrier();
                auto absolute = reinterpret_cast<volatile uint32_t*>(owners_l1);
                for (uint32_t j = 0; j <= a; ++j) {
                    const uint32_t p = static_cast<uint32_t>(position[r] + static_cast<int32_t>(j));
                    absolute[p % 2048] = p;
                }
                noc_async_write(owners_l1, owners.get_noc_addr(0), 2048 * 4);
                noc_async_write_barrier();
            }
#ifdef SR_SAMPLED
            // Parameter row r0 (the group's first row): word 0 flags (bit 0 active), word 11 the history slot (the
            // host keeps every request's history in step, as the non-speculative sampled decode does).
            auto params = reinterpret_cast<volatile uint32_t*>(params_l1 + r * 128);
            if ((params[0] & 1) && !redraw[r + a] && params[11] < slots) {
                const auto hist = dram(get_common_arg_val<uint32_t>(sampled_base + 1 + params[11]), 128);
                for (uint32_t j = 0; j <= a; ++j) {
                    const uint32_t id = target[r + j];
                    noc_async_read(hist.get_noc_addr(id / 32), hist_l1, 128);
                    noc_async_read_barrier();
                    auto counts = reinterpret_cast<volatile uint32_t*>(hist_l1);
                    counts[id % 32] = counts[id % 32] + 1;  // bit 31 (prompt flag) is untouched below 2^31
                    noc_async_write(hist_l1, hist.get_noc_addr(id / 32), 128);
                    noc_async_write_barrier();
                }
            }
#endif
            r += length;
        }
        commit[0] = committed;
        noc_async_write(io + 384, dram(get_common_arg_val<uint32_t>(4), 256).get_noc_addr(0), 256);
        noc_async_write(io + 640, dram(get_common_arg_val<uint32_t>(5), 128).get_noc_addr(0), 128);
        noc_async_write(io + 768, dram(get_common_arg_val<uint32_t>(6), 128).get_noc_addr(0), 128);
        noc_async_write_barrier();
#endif
    }
#endif
}
