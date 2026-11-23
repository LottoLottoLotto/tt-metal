// SPDX-License-Identifier: Apache-2.0
// Full-vocabulary hierarchical top16 and sequential path updates. All comparisons
// happen on device RISCs; the learned BF16 rank-256 dot products run on the FPU.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
namespace {
constexpr uint32_t k = 16;
constexpr uint32_t workers = 64;
constexpr uint32_t vocab_tiles = 248320 / 32;
constexpr uint32_t rows = DFLASH_ROWS;
constexpr uint32_t mode = DFLASH_MODE;
constexpr uint32_t invalid = 0xffffffff;
uint32_t offset(uint32_t r, uint32_t c) {
    return (r / 16) * 512 + (c / 16) * 256 + (r % 16) * 16 + c % 16;
}
// Monotonic BF16 key, including negative numbers. +/-0 tie and NaNs sort above
// infinity (torch topk's convention). Equal values choose the lower vocabulary
// ID: torch.topk(sorted=False) does not promise an order for tied values.
uint32_t key(uint32_t bits) {
    if ((bits & 0x7fff) == 0) bits = 0;
    if ((bits & 0x7f80) == 0x7f80 && (bits & 0x7f)) return 0x10000;
    return bits & 0x8000 ? (~bits & 0xffff) : (bits ^ 0x8000);
}
bool better(uint32_t v, uint32_t id, uint32_t old_v, uint32_t old_id) {
    return old_id == invalid || key(v) > key(old_v) || (key(v) == key(old_v) && id < old_id);
}
void insert(uint32_t v, uint32_t id, volatile uint32_t* values, volatile uint32_t* ids) {
    if (id == invalid || !better(v, id, values[k - 1], ids[k - 1])) return;
    uint32_t slot = k - 1;
    while (slot && better(v, id, values[slot - 1], ids[slot - 1])) {
        values[slot] = values[slot - 1];
        ids[slot] = ids[slot - 1];
        --slot;
    }
    values[slot] = v;
    ids[slot] = id;
}
}
#endif

void kernel_main() {
#if defined(COMPILE_FOR_BRISC)
    const uint32_t scratch = get_write_ptr(0);
    const uint32_t task = get_arg_val<uint32_t>(0);
    if constexpr (mode == 0) {
        const auto logits = TensorAccessor(tensor_accessor::make_interleaved_dspec<DFLASH_LOGITS_DRAM>(), get_common_arg_val<uint32_t>(0), 2048);
        const auto partial = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), 128);
        auto input = reinterpret_cast<volatile uint16_t*>(scratch);
        auto ids = reinterpret_cast<volatile uint32_t*>(scratch + 2048);
        auto values = ids + rows * k;
        auto record = reinterpret_cast<volatile uint32_t*>(scratch + 12288);
        for (uint32_t i = 0; i < rows * k; ++i) {
            ids[i] = invalid;
            values[i] = 0xff80;
        }
        // Disjoint striped vocabulary tiles: all 7760 tiles are visited once.
        for (uint32_t tile = task; tile < vocab_tiles; tile += workers) {
            noc_async_read(get_noc_addr(tile, logits), scratch, 2048);
            noc_async_read_barrier();
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t c = 0; c < 32; ++c) {
                    insert(input[offset(r, c)], tile * 32 + c, values + r * k, ids + r * k);
                }
            }
        }
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t i = 0; i < k; ++i) {
                record[i] = ids[r * k + i];
                record[k + i] = values[r * k + i];
            }
            noc_async_write(scratch + 12288, get_noc_addr(r * workers + task, partial), 128);
            noc_async_write_barrier();
        }
    } else if constexpr (mode == 1) {
        const auto partial = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), 128);
        const auto candidates = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1 + 2 * task), 64);
        const auto unary = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2 + 2 * task), 2048);
        auto record = reinterpret_cast<volatile uint32_t*>(scratch);
        auto ids = reinterpret_cast<volatile uint32_t*>(scratch + 128);
        auto values = reinterpret_cast<volatile uint32_t*>(scratch + 192);
        auto tile = reinterpret_cast<volatile uint16_t*>(scratch + 256);
        for (uint32_t i = 0; i < k; ++i) {
            ids[i] = invalid;
            values[i] = 0xff80;
        }
        // A globally top16 value must belong to its shard's local top16.
        for (uint32_t shard = 0; shard < workers; ++shard) {
            noc_async_read(get_noc_addr(task * workers + shard, partial), scratch, 128);
            noc_async_read_barrier();
            for (uint32_t i = 0; i < k; ++i) insert(record[k + i], record[i], values, ids);
        }
        for (uint32_t i = 0; i < 1024; ++i) tile[i] = 0;
        for (uint32_t i = 0; i < k; ++i) tile[i] = values[i];
        noc_async_write(scratch + 128, get_noc_addr(0, candidates), 64);
        noc_async_write(scratch + 256, get_noc_addr(0, unary), 2048);
        noc_async_write_barrier();
    } else if constexpr (mode == 2 || mode == 4) {
        constexpr uint32_t path_bytes = ((rows * 4 + 31) / 32) * 32;
        const auto scores = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), 2048);
        const auto candidates = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), 64);
        const auto next = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 32);
        const auto path = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3), path_bytes);
        auto s = reinterpret_cast<volatile uint16_t*>(scratch);
        auto ids = reinterpret_cast<volatile uint32_t*>(scratch + 2048);
        auto output = reinterpret_cast<volatile uint32_t*>(scratch + 2112);
        auto scalar = reinterpret_cast<volatile uint32_t*>(scratch + 2240);
        noc_async_read(get_noc_addr(0, scores), scratch, 2048);
        noc_async_read(get_noc_addr(0, candidates), scratch + 2048, 64);
        if constexpr (mode == 4) {
            // Rank-two fork: inherit only the already chosen prefix of A.
            const auto source_path = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4), path_bytes);
            noc_async_read(get_noc_addr(0, source_path), scratch + 2112, path_bytes);
        } else if (task) {
            noc_async_read(get_noc_addr(0, path), scratch + 2112, path_bytes);
        }
        noc_async_read_barrier();
        if (!task) for (uint32_t i = 0; i < path_bytes / 4; ++i) output[i] = 0;
        for (uint32_t i = 0; i < 8; ++i) scalar[i] = 0;
        uint32_t best = 0;
        for (uint32_t i = 1; i < k; ++i) if (better(s[i], ids[i], s[best], ids[best])) best = i;
        if constexpr (mode == 4) {
            const uint32_t first = best;
            best = first == 0 ? 1 : 0;
            for (uint32_t i = 0; i < k; ++i) {
                if (i != first && better(s[i], ids[i], s[best], ids[best])) best = i;
            }
            for (uint32_t i = task; i < path_bytes / 4; ++i) output[i] = 0;
        }
        scalar[0] = ids[best];
        output[task] = ids[best];
        noc_async_write(scratch + 2240, get_noc_addr(0, next), 32);
        noc_async_write(scratch + 2112, get_noc_addr(0, path), path_bytes);
        noc_async_write_barrier();
    } else if constexpr (mode == 3) {
        // One core per rank tile. Preserve the requested row bit-for-bit and
        // clear padding, without untilizing/retilizing the entire projection.
        constexpr uint32_t position = DFLASH_POSITION;
        const auto input = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), 2048);
        const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), 2048);
        auto source = reinterpret_cast<volatile uint16_t*>(scratch);
        auto destination = reinterpret_cast<volatile uint16_t*>(scratch + 2048);
        noc_async_read(get_noc_addr(task, input), scratch, 2048);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < 1024; ++i) destination[i] = 0;
        for (uint32_t c = 0; c < 32; ++c) destination[offset(0, c)] = source[offset(position, c)];
        noc_async_write(scratch + 2048, get_noc_addr(task, output), 2048);
        noc_async_write_barrier();
    }
#endif
}
