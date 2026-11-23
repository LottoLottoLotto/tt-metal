// SPDX-License-Identifier: Apache-2.0
// P150 T1..T32 Qwen3.8-27B: four-tap BF16 FIR, SiLU, and exact GQA3 scatter.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#endif

namespace native_gdn {
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kHalfRowBytes = 32;
constexpr uint32_t kHistoryPageBytes = 10240 * sizeof(uint16_t);
constexpr uint32_t kInputCb = 0;
constexpr uint32_t kWeightCb = 1;
constexpr uint32_t kScratchCb = 2;
constexpr uint32_t kHistoryCb = 3;
constexpr uint32_t kOutputCb = 16;
constexpr uint32_t kTempCb = 17;

FORCE_INLINE uint32_t tile_row_offset(uint32_t row, bool right) {
    return (row >= 16 ? 1024 : 0) + (right ? 512 : 0) + (row & 15) * kHalfRowBytes;
}

#if defined(COMPILE_FOR_BRISC)

FORCE_INLINE void copy_half_row(uint32_t source, uint32_t destination) {
    const volatile uint32_t* src = reinterpret_cast<const volatile uint32_t*>(source);
    volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(destination);
    for (uint32_t i = 0; i < kHalfRowBytes / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}
#endif
}  // namespace native_gdn

void kernel_main() {
    using namespace native_gdn;
    constexpr uint32_t num_tokens = get_named_compile_time_arg_val("num_tokens");
    static_assert(num_tokens >= 1 && num_tokens <= 32);
    constexpr uint32_t fork_position = get_named_compile_time_arg_val("fork_position");
#if defined(COMPILE_FOR_BRISC)
    // Ownership follows source heads, so GQA copies 30,31,32 share one reader.
    constexpr uint32_t worker = get_named_compile_time_arg_val("worker_id");
    constexpr uint32_t kind = worker < 16 ? 0 : (worker < 32 ? 1 : 2);
    constexpr uint32_t head = worker < 16 ? worker : (worker < 32 ? worker - 16 : worker - 32);
    constexpr uint32_t copies = kind < 2 ? 3 : 1;
    constexpr uint32_t flat_head = worker * 128;
    constexpr bool mega_is_dram = get_named_compile_time_arg_val("mega_is_dram") == 1;
    // Batched users: row t is user t. Its three history rows are its own DRAM state
    // (common arg 14 + t, the candidate slot), and the shifted history is written back there.
    constexpr bool batched = get_named_compile_time_arg_val("batched") == 1;
    // Shared verify rounds (groups = slot count): row map word t (active bit 31, slot << 16, length << 8,
    // offset) groups consecutive rows by user. Row t's tap source j = offset + tap is history row j of
    // its slot's DRAM state (common arg 15 + slot) when j < 3, else the group's own row t - 3 + tap. A
    // single-row group shifts its slot's history in place; other rows write their prefix history to
    // candidate rows 3t.. of the [1,3T,10240] tensor (common arg 14). Inactive rows read and write
    // only their candidate rows.
    constexpr uint32_t groups = get_named_compile_time_arg_val("groups");
    const uint32_t mega_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t state_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t state_x = get_common_arg_val<uint32_t>(7);
    const uint32_t state_y = get_common_arg_val<uint32_t>(8);
    const uint32_t scalar_addrs[2] = {get_common_arg_val<uint32_t>(9), get_common_arg_val<uint32_t>(10)};
    const uint32_t output_addrs[3] = {get_common_arg_val<uint32_t>(11), get_common_arg_val<uint32_t>(12),
                                    get_common_arg_val<uint32_t>(13)};
    const auto mega = TensorAccessor(tensor_accessor::make_interleaved_dspec<mega_is_dram>(), mega_addr, kTileBytes);
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), output_addrs[kind], kTileBytes);
    // Tokens are processed in groups of four: slot s of a group owns the even
    // tile rows 8s + 2d, so one elementwise tile pass covers four tokens (the
    // FIR/SiLU math is position independent, hence bitwise unchanged). Each
    // group stages all four taps plus bias with one zero fill and one read
    // round trip; two groups are in flight (input/weight CBs hold two groups),
    // so group g+1 is gathered while compute runs group g.
    // Staging (CB3) per in-flight group: 16 history rows (tap, slot) of 256 B,
    // then 64 row-pair slots (tap, slot, dim) of 128 B; a 64 B zero row follows.
    constexpr uint32_t kSlots = 4;
    constexpr uint32_t num_groups = (num_tokens + kSlots - 1) / kSlots;
    constexpr uint32_t kHistoryBytes = 4 * kSlots * 256;
    constexpr uint32_t kGroupStage = kHistoryBytes + 4 * kSlots * 4 * 128;
    const uint32_t stage_base = get_write_ptr(kHistoryCb);
    const uint32_t zero_src = stage_base + 2 * kGroupStage;
    for (uint32_t i = 0; i < 16; ++i) {
        reinterpret_cast<volatile uint32_t*>(zero_src)[i] = 0;
    }
    uint32_t row_map[num_tokens];
    if constexpr (groups > 0) {
        const uint32_t map_l1 = zero_src + 64;
        const auto map = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                        get_common_arg_val<uint32_t>(15 + groups), 128);
        noc_async_read(get_noc_addr(0, map), map_l1, 128);
        noc_async_read_barrier();
        for (uint32_t t = 0; t < num_tokens; ++t) {
            row_map[t] = reinterpret_cast<volatile uint32_t*>(map_l1)[t];
        }
    }
    Noc noc;

    auto issue_group = [&](uint32_t group) {
        const uint32_t first = group * kSlots;
        const uint32_t slots = num_tokens - first < kSlots ? num_tokens - first : kSlots;
        const uint32_t history = stage_base + (group & 1) * kGroupStage;
        const uint32_t pairs = history + kHistoryBytes;
        if (group >= 2) {
            // This staging buffer last fed group-2's history writes.
            noc_async_write_barrier();
        }
        // Tap weights and bias rows are gathered by NCRISC (below) in parallel.
        cb_reserve_back(kInputCb, 4);
        const uint32_t input_dst = get_write_ptr(kInputCb);
        // Preserve zero padding without clearing words on the reader.
        noc.async_write_zeros(CircularBuffer(kInputCb), 4 * kTileBytes);
        noc.write_zeros_l1_barrier();
        int32_t rows[4][kSlots];
        for (uint32_t tap = 0; tap < 4; ++tap) {
            for (uint32_t s = 0; s < slots; ++s) {
                const uint32_t token = first + s;
                const uint32_t history_staging = history + (tap * kSlots + s) * 256;
                // Walk the fixed tree to the FIR ancestor; negative indices are
                // the original three live history rows, never another branch.
                int32_t ancestor = token;
                for (uint32_t back = tap; back < 3; ++back) {
                    ancestor = (fork_position && ancestor == 8)
                        ? int32_t(fork_position) - 1 : ancestor - 1;
                }
                // Batched rows never see another row: taps 0..2 are the user's own history.
                int32_t history_index = batched ? (tap < 3 ? int32_t(tap) : int32_t(token) + 3) : ancestor + 3;
                if constexpr (groups > 0) {
                    const uint32_t word = row_map[token];
                    const uint32_t j = ((word >> 31) ? (word & 0xff) : 0u) + tap;
                    history_index = j < 3 ? int32_t(j) : int32_t(token) - 3 + int32_t(tap) + 3;
                }
                rows[tap][s] = history_index;
                if (history_index < 3) {
                    if constexpr (groups > 0) {
                        const uint32_t word = row_map[token];
                        const bool active = (word >> 31) != 0;
                        // An inactive row reads slot 0's history (read only; its result is discarded).
                        const auto source = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                            get_common_arg_val<uint32_t>(15 + (active ? ((word >> 16) & 0xff) : 0)), kHistoryPageBytes);
                        const uint32_t page = uint32_t(history_index);
                        noc_async_read(get_noc_addr(page, source) + flat_head * sizeof(uint16_t),
                            history_staging, 128 * sizeof(uint16_t));
                    } else if constexpr (batched) {
                        const auto user = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                            get_common_arg_val<uint32_t>(14 + token), kHistoryPageBytes);
                        noc_async_read(get_noc_addr(history_index, user) + flat_head * sizeof(uint16_t),
                            history_staging, 128 * sizeof(uint16_t));
                    } else {
                        // All three history rows live on this core, not at three equal addresses.
                        noc_async_read(get_noc_addr(state_x, state_y,
                            state_addr + history_index * kHistoryPageBytes + flat_head * sizeof(uint16_t)),
                            history_staging, 128 * sizeof(uint16_t));
                    }
                }
                const uint32_t row = history_index - 3;
                for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
                    const uint32_t flat = flat_head + dim_tile * 32;
                    if (history_index >= 3) {
                        const uint64_t input_src = get_noc_addr(flat / 32, mega);
                        const uint32_t pair = pairs + ((tap * kSlots + s) * 4 + dim_tile) * 128;
                        // Select an aligned pair from either face-row group. Only
                        // the requested 32-byte half-row participates in the FIR.
                        noc_async_read(input_src + tile_row_offset(row & ~1u, false), pair, 64);
                        noc_async_read(input_src + tile_row_offset(row & ~1u, true), pair + 64, 64);
                    }
                }
            }
        }
        noc_async_read_barrier();
        for (uint32_t tap = 0; tap < 4; ++tap) {
            for (uint32_t s = 0; s < slots; ++s) {
                const uint32_t history_staging = history + (tap * kSlots + s) * 256;
                if (rows[tap][s] >= 3) {
                    const uint32_t row_offset = ((rows[tap][s] - 3) & 1) * kHalfRowBytes;
                    for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
                        const uint32_t pair = pairs + ((tap * kSlots + s) * 4 + dim_tile) * 128;
                        copy_half_row(pair + row_offset, history_staging + dim_tile * 64);
                        copy_half_row(pair + 64 + row_offset, history_staging + dim_tile * 64 + 32);
                    }
                }
                const uint32_t dst = input_dst + tap * kTileBytes;
                for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
                    const uint32_t dst_row = 8 * s + dim_tile * 2;
                    copy_half_row(history_staging + dim_tile * 64, dst + tile_row_offset(dst_row, false));
                    copy_half_row(history_staging + dim_tile * 64 + 32, dst + tile_row_offset(dst_row, true));
                }
            }
        }
        cb_push_back(kInputCb, 4);
        // Every history read above completed before these writes (T=1 shifts live rows in place).
        for (uint32_t tap = 1; tap < 4; ++tap) {
            for (uint32_t s = 0; s < slots; ++s) {
                const uint32_t history_staging = history + (tap * kSlots + s) * 256;
                if constexpr (num_tokens == 1) {
                    noc_async_write(history_staging, get_noc_addr(state_x, state_y,
                        state_addr + (tap - 1) * kHistoryPageBytes + flat_head * sizeof(uint16_t)), 256);
                } else if constexpr (groups > 0) {
                    const uint32_t token = first + s;
                    const uint32_t word = row_map[token];
                    const bool active = (word >> 31) != 0;
                    const bool single = active && ((word >> 8) & 0xff) == 1;
                    if (active) {
                        const auto destination = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                            get_common_arg_val<uint32_t>(single ? 15 + ((word >> 16) & 0xff) : 14), kHistoryPageBytes);
                        const uint32_t page = single ? tap - 1 : 3 * token + tap - 1;
                        noc_async_write(history_staging, get_noc_addr(page, destination) + flat_head * sizeof(uint16_t), 256);
                    }
                } else {
                    // Verification never mutates live history; retain every prefix.
                    const auto candidate = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                        get_common_arg_val<uint32_t>(14 + first + s), kHistoryPageBytes);
                    noc_async_write(history_staging, get_noc_addr(tap - 1, candidate) + flat_head * sizeof(uint16_t), 256);
                }
            }
        }
    };

    auto finish_group = [&](uint32_t group) {
        const uint32_t first = group * kSlots;
        const uint32_t slots = num_tokens - first < kSlots ? num_tokens - first : kSlots;
        cb_wait_front(kOutputCb, 1);
        const uint32_t output_src = get_read_ptr(kOutputCb);
        for (uint32_t s = 0; s < slots; ++s) {
            const uint32_t token = first + s;
            for (uint32_t copy = 0; copy < copies; ++copy) {
                const uint32_t output_head = head * copies + copy;
                for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
                    const uint64_t dst = get_noc_addr(token * 8 + (output_head / 32) * 4 + dim_tile, output);
                    const uint32_t src_row = 8 * s + dim_tile * 2;
                    noc_async_write(output_src + tile_row_offset(src_row, false),
                                    dst + tile_row_offset(output_head % 32, false), kHalfRowBytes);
                    noc_async_write(output_src + tile_row_offset(src_row, true),
                                    dst + tile_row_offset(output_head % 32, true), kHalfRowBytes);
                }
            }
            if constexpr (head < 16) {
                for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
                    const uint64_t dst = get_noc_addr(token * 8 + 4 + dim_tile, output);
                    noc_async_write(zero_src, dst + tile_row_offset(16 + head, false), kHalfRowBytes);
                    noc_async_write(zero_src, dst + tile_row_offset(16 + head, true), kHalfRowBytes);
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(kOutputCb, 1);
        if constexpr (worker == 0) {
            // The row-pair area of this group's staging is free once its copies ran.
            const uint32_t scalar_staging = stage_base + (group & 1) * kGroupStage + kHistoryBytes;
            for (uint32_t s = 0; s < slots; ++s) {
                const uint32_t token = first + s;
                // Scalars retain two width tiles; only heads 0..47 are logical.
                for (uint32_t slot = 0; slot < 4; ++slot) {
                    const uint64_t src = get_noc_addr(320 + slot, mega);
                    noc_async_read(src + tile_row_offset(token & ~1u, false), scalar_staging + slot * 128, 64);
                    noc_async_read(src + tile_row_offset(token & ~1u, true), scalar_staging + slot * 128 + 64, 64);
                }
                noc_async_read_barrier();
                for (uint32_t slot = 0; slot < 4; ++slot) {
                    const uint32_t scalar = slot / 2, tile = slot % 2;
                    const auto dst_accessor = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), scalar_addrs[scalar], kTileBytes);
                    const uint64_t dst = get_noc_addr(tile, dst_accessor);
                    noc_async_write(scalar_staging + slot * 128 + (token & 1) * 32, dst + tile_row_offset(token, false), 32);
                    noc_async_write(scalar_staging + slot * 128 + 64 + (token & 1) * 32, dst + tile_row_offset(token, true), 32);
                }
                noc_async_write_barrier();
            }
        }
    };

    for (uint32_t group = 0; group < num_groups; ++group) {
        issue_group(group);
        if (group > 0) {
            finish_group(group - 1);
        }
    }
    finish_group(num_groups - 1);
#elif defined(COMPILE_FOR_NCRISC)
    // Weights: the four tap rows and the bias row, replicated into each slot's
    // even rows 8s + 2d exactly as the input rows are laid out.
    constexpr uint32_t worker = get_named_compile_time_arg_val("worker_id");
    constexpr uint32_t flat_head = worker * 128;
    constexpr bool tap_is_dram = get_named_compile_time_arg_val("tap_is_dram") == 1;
    constexpr bool bias_is_dram = get_named_compile_time_arg_val("bias_is_dram") == 1;
    constexpr uint32_t kSlots = 4;
    constexpr uint32_t num_groups = (num_tokens + kSlots - 1) / kSlots;
    Noc noc;
    for (uint32_t group = 0; group < num_groups; ++group) {
        const uint32_t first = group * kSlots;
        const uint32_t slots = num_tokens - first < kSlots ? num_tokens - first : kSlots;
        cb_reserve_back(kWeightCb, 5);
        const uint32_t weight_dst = get_write_ptr(kWeightCb);
        noc.async_write_zeros(CircularBuffer(kWeightCb), 5 * kTileBytes);
        noc.write_zeros_l1_barrier();
        for (uint32_t tap = 0; tap < 5; ++tap) {
            const uint32_t address = get_common_arg_val<uint32_t>(tap);
            const uint32_t dst = weight_dst + tap * kTileBytes;
            for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
                uint64_t src;
                if (tap < 4) {
                    src = get_noc_addr(flat_head / 32 + dim_tile,
                        TensorAccessor(tensor_accessor::make_interleaved_dspec<tap_is_dram>(), address, kTileBytes));
                } else {
                    src = get_noc_addr(flat_head / 32 + dim_tile,
                        TensorAccessor(tensor_accessor::make_interleaved_dspec<bias_is_dram>(), address, kTileBytes));
                }
                for (uint32_t s = 0; s < slots; ++s) {
                    const uint32_t dst_row = 8 * s + dim_tile * 2;
                    // Even destination rows are 64B aligned. Only their first half-row is used.
                    noc_async_read(src, dst + tile_row_offset(dst_row, false), 64);
                    noc_async_read(src + 512, dst + tile_row_offset(dst_row, true), 64);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(kWeightCb, 5);
    }
#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t num_groups = (num_tokens + 3) / 4;
    for (uint32_t group = 0; group < num_groups; ++group) {
    unary_op_init_common(kInputCb, kScratchCb);
    cb_wait_front(kInputCb, 4);
    cb_wait_front(kWeightCb, 5);

    // ttnn.multiply uses the precise SFPU binary path for BF16. Preserve its
    // per-op BF16 boundaries and leave the baseline BF16 adds on the FPU.
    mul_binary_tile_init();
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(kWeightCb, kInputCb);
    copy_tile(kInputCb, 3, 0);
    copy_tile_to_dst_init_short_with_dt(kInputCb, kWeightCb);
    copy_tile(kWeightCb, 3, 1);
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    cb_reserve_back(kScratchCb, 1);
    tile_regs_wait();
    pack_reconfig_data_format<true>(kScratchCb);
    pack_tile(0, kScratchCb);
    tile_regs_release();
    cb_push_back(kScratchCb, 1);

    for (uint32_t tap = 0; tap < 3; ++tap) {
        mul_binary_tile_init();
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(kWeightCb, kInputCb);
        copy_tile(kInputCb, tap, 0);
        copy_tile_to_dst_init_short_with_dt(kInputCb, kWeightCb);
        copy_tile(kWeightCb, tap, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(kTempCb, 1);
        tile_regs_wait();
        pack_reconfig_data_format<true>(kTempCb);
        pack_tile(0, kTempCb);
        tile_regs_release();
        cb_push_back(kTempCb, 1);

        cb_wait_front(kScratchCb, 1);
        cb_wait_front(kTempCb, 1);
        tile_regs_acquire();
        add_tiles_init(kScratchCb, kTempCb);
        add_tiles(kScratchCb, kTempCb, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(kScratchCb, 1);
        cb_pop_front(kTempCb, 1);
        cb_reserve_back(kScratchCb, 1);
        pack_reconfig_data_format<true>(kScratchCb);
        pack_tile(0, kScratchCb);
        tile_regs_release();
        cb_push_back(kScratchCb, 1);
    }

    cb_wait_front(kScratchCb, 1);
    tile_regs_acquire();
    add_tiles_init(kScratchCb, kWeightCb);
    add_tiles(kScratchCb, kWeightCb, 0, 4, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_pop_front(kScratchCb, 1);
    cb_reserve_back(kTempCb, 1);
    pack_reconfig_data_format<true>(kTempCb);
    pack_tile(0, kTempCb);
    tile_regs_release();
    cb_push_back(kTempCb, 1);

    cb_wait_front(kTempCb, 1);
    cb_reserve_back(kOutputCb, 1);
    tile_regs_acquire();
    copy_tile_init(kTempCb);
    copy_tile(kTempCb, 0, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format<true>(kOutputCb);
    pack_tile(0, kOutputCb);
    tile_regs_release();
    cb_pop_front(kTempCb, 1);
    cb_push_back(kOutputCb, 1);

    cb_pop_front(kInputCb, 4);
    cb_pop_front(kWeightCb, 5);
    }
#endif
}
