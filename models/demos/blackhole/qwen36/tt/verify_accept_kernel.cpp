// SPDX-License-Identifier: Apache-2.0
// Exact greedy verification on tiled BF16 logits: per-row argmax with ttnn.argmax's order
// (bfloat16_greater is a total order on the BF16 bit pattern; equal bits keep the lower index;
// the scan starts from (-inf, index 0)), then the longest-accepted-prefix bookkeeping.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

namespace {
constexpr uint32_t rows = VA_ROWS;
constexpr uint32_t depth = rows - 1;
constexpr uint32_t mode = VA_MODE;
constexpr uint32_t workers = VA_WORKERS;
constexpr uint32_t vocab_tiles = VA_VOCAB_TILES;
constexpr uint32_t row_bytes = rows * 32;  // live rows of one 16-column face
constexpr uint32_t record_bytes = 64;      // (key, index) for up to 8 rows
constexpr uint32_t head_bytes = ((2 * rows - 1) * 4 + 31) / 32 * 32;

// Monotonic key of bfloat16_greater's order.
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
        // Args: partial, tokens [1,rows] u32, positions [rows] i32 (updated in place), head [1,2*rows-1],
        // accepted, next_token, advance, next_position, start ([1,1] int32 each).
        const auto partial = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), record_bytes);
        const auto tokens = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), 32);
        const auto positions = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 32);
        const auto head = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3), head_bytes);
        const uint32_t scratch = get_write_ptr(0);
        auto records = reinterpret_cast<volatile uint32_t*>(scratch);
        const uint32_t io = scratch + workers * record_bytes;
        // DRAM reads land on 64-byte aligned L1 slots (Blackhole DRAM read alignment).
        auto token = reinterpret_cast<volatile uint32_t*>(io);
        auto position = reinterpret_cast<volatile int32_t*>(io + 64);
        auto out = reinterpret_cast<volatile int32_t*>(io + 128);
        auto scalars = reinterpret_cast<volatile int32_t*>(io + 192);  // 5 x 32-byte pages
        for (uint32_t w = 0; w < workers; ++w) noc_async_read(partial.get_noc_addr(w), scratch + w * record_bytes, record_bytes);
        noc_async_read(tokens.get_noc_addr(0), io, 32);
        noc_async_read(positions.get_noc_addr(0), io + 64, 32);
        noc_async_read_barrier();
        int32_t target[rows];
        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t k = key(0xff80), id = 0;
            for (uint32_t w = 0; w < workers; ++w) {
                const uint32_t candidate = records[w * 16 + 2 * r], index = records[w * 16 + 2 * r + 1];
                if (candidate > k || (candidate == k && index < id)) {
                    k = candidate;
                    id = index;
                }
            }
            target[r] = static_cast<int32_t>(id);
        }
        for (uint32_t i = 0; i < 16; ++i) out[i] = 0;
        int32_t accepted = 0, chained = 1;
        for (uint32_t r = 0; r < rows; ++r) out[r] = target[r];
        for (uint32_t i = 0; i < depth; ++i) {
            chained = chained && target[i] == static_cast<int32_t>(token[i + 1]) ? 1 : 0;
            out[rows + i] = chained;
            accepted += chained;
        }
        const int32_t start = position[0];
        const int32_t values[5] = {accepted, target[accepted], accepted + 1, start + accepted + 1, start};
        for (uint32_t s = 0; s < 5; ++s) {
            for (uint32_t i = 0; i < 8; ++i) scalars[8 * s + i] = i ? 0 : values[s];
            const auto scalar = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4 + s), 32);
            noc_async_write(io + 192 + 32 * s, scalar.get_noc_addr(0), 32);
        }
        for (uint32_t r = 0; r < rows; ++r) position[r] = start + accepted + 1 + static_cast<int32_t>(r);
        noc_async_write(io + 128, head.get_noc_addr(0), head_bytes);
        noc_async_write(io + 64, positions.get_noc_addr(0), 32);
        noc_async_write_barrier();
#endif
    }
#endif
}
