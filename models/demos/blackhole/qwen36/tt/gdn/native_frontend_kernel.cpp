// SPDX-License-Identifier: Apache-2.0
// P150 T1..T32 GDN frontend: four-tap BF16 FIR, SiLU, and exact GQA scatter.
// Geometry (key heads, GQA factor, value heads, flat QKV width, scalar tiles) is
// compile-time; every head is 128 wide. 27B: H=16, G=3, HV=48, W=10240.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
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
constexpr uint32_t kQkvWidth = get_named_compile_time_arg_val("qkv_width");
constexpr uint32_t kHistoryPageBytes = kQkvWidth * sizeof(uint16_t);
constexpr uint32_t kKeyHeads = get_named_compile_time_arg_val("key_heads");
constexpr uint32_t kGqa = get_named_compile_time_arg_val("gqa");
constexpr uint32_t kValueHeads = get_named_compile_time_arg_val("value_heads");
// Output q/k/v are [T, 1, HV, 128]: heads fill tile rows, 4 width tiles per tile row.
constexpr uint32_t kHeadRowTiles = (kValueHeads + 31) / 32;
constexpr uint32_t kTokenTiles = kHeadRowTiles * 4;
// Logical rows past HV in the last head tile row must stay zero.
constexpr uint32_t kPadRows = kHeadRowTiles * 32 - kValueHeads;
constexpr uint32_t kScalarTileBase = kQkvWidth / 32;
constexpr uint32_t kScalarTiles = get_named_compile_time_arg_val("scalar_tiles");
static_assert(kQkvWidth == (2 * kKeyHeads + kValueHeads) * 128);
static_assert(kKeyHeads * kGqa == kValueHeads);
constexpr uint32_t kInputCb = 0;
constexpr uint32_t kWeightCb = 1;
constexpr uint32_t kScratchCb = 2;
constexpr uint32_t kHistoryCb = 3;
constexpr uint32_t kOutputCb = 16;
constexpr uint32_t kTempCb = 17;
// Tokens share tiles: token t of a pass uses rows 8t + 2d (d = head dim tile), so one
// compute pass serves four tokens. Row placement does not change any element's math.
constexpr uint32_t kPassTokens = 4;

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
    constexpr uint32_t passes = (num_tokens + kPassTokens - 1) / kPassTokens;
    static_assert(num_tokens >= 1 && num_tokens <= 32);
    constexpr uint32_t fork_position = get_named_compile_time_arg_val("fork_position");
#if defined(COMPILE_FOR_BRISC)
    // Ownership follows source heads, so the GQA copies of one key head share one reader.
    // The worker id is a per-core runtime argument: a compile-time id gave every core its
    // own binary, and dispatching 64 distinct binaries cost ~20-35 us per launch.
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t kind = worker < kKeyHeads ? 0 : (worker < 2 * kKeyHeads ? 1 : 2);
    const uint32_t head = kind == 0 ? worker : (kind == 1 ? worker - kKeyHeads : worker - 2 * kKeyHeads);
    const uint32_t copies = kind < 2 ? kGqa : 1;
    const uint32_t flat_head = worker * 128;
    constexpr bool mega_is_dram = get_named_compile_time_arg_val("mega_is_dram") == 1;
    constexpr bool tap_is_dram = get_named_compile_time_arg_val("tap_is_dram") == 1;
    constexpr bool bias_is_dram = get_named_compile_time_arg_val("bias_is_dram") == 1;
    const uint32_t mega_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tap_addrs[4] = {get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2),
                                 get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4)};
    const uint32_t bias_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t state_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t state_x = get_common_arg_val<uint32_t>(7);
    const uint32_t state_y = get_common_arg_val<uint32_t>(8);
    const uint32_t scalar_addrs[2] = {get_common_arg_val<uint32_t>(9), get_common_arg_val<uint32_t>(10)};
    const uint32_t output_addrs[3] = {get_common_arg_val<uint32_t>(11), get_common_arg_val<uint32_t>(12),
                                    get_common_arg_val<uint32_t>(13)};
    const auto mega = TensorAccessor(tensor_accessor::make_interleaved_dspec<mega_is_dram>(), mega_addr, kTileBytes);
    const auto bias = TensorAccessor(tensor_accessor::make_interleaved_dspec<bias_is_dram>(), bias_addr, kTileBytes);
    // Verification candidates and q/k/v/a/b outputs reside in DRAM.
    const auto output = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), output_addrs[kind], kTileBytes);
    Noc noc;

    // Staging (kHistoryCb): every FIR source row of this head in row-major form (the three
    // live history rows, then one per token), the token rows' face-row blocks as read,
    // the four taps and the bias (rows 0-1 of both faces per dim tile), and a zero row.
    constexpr uint32_t kRowBytes = 128 * sizeof(uint16_t);
    constexpr uint32_t kSources = 3 + num_tokens;
    constexpr uint32_t kTopRows = num_tokens < 16 ? num_tokens : 16;
    constexpr uint32_t kBottomRows = num_tokens > 16 ? num_tokens - 16 : 0;
    constexpr uint32_t kTopBlock = kHalfRowBytes * ((kTopRows + 1) & ~1u);  // one face, 64 B multiple
    constexpr uint32_t kBottomBlock = kHalfRowBytes * ((kBottomRows + 1) & ~1u);
    constexpr uint32_t kDimBytes = 2 * (kTopBlock + kBottomBlock);
    static_assert(kScalarTiles <= 2, "scalar tiles reuse the four-tile token staging");
    const uint32_t rows_base = get_write_ptr(kHistoryCb);
    const uint32_t stage_base = rows_base + kSources * kRowBytes;
    const uint32_t param_base = stage_base + 4 * kDimBytes;
    const uint32_t zero_row = param_base + 5 * 4 * 128;

    // Face-row blocks of one tile row (rows 0..num_tokens-1), 64 B aligned on both ends.
    auto read_token_rows = [&](uint64_t src, uint32_t dst) {
        noc_async_read(src, dst, kTopBlock);
        noc_async_read(src + 512, dst + kTopBlock, kTopBlock);
        if constexpr (kBottomRows > 0) {
            noc_async_read(src + 1024, dst + 2 * kTopBlock, kBottomBlock);
            noc_async_read(src + 1536, dst + 2 * kTopBlock + kBottomBlock, kBottomBlock);
        }
    };
    auto staged_row = [&](uint32_t block, uint32_t token, bool right) {
        const uint32_t face = token < 16 ? (right ? kTopBlock : 0)
                                         : 2 * kTopBlock + (right ? kBottomBlock : 0);
        return block + face + (token & 15) * kHalfRowBytes;
    };

    // One read phase: all three history rows live on the state core.
    for (uint32_t row = 0; row < 3; ++row) {
        noc_async_read(get_noc_addr(state_x, state_y,
            state_addr + row * kHistoryPageBytes + flat_head * sizeof(uint16_t)),
            rows_base + row * kRowBytes, kRowBytes);
    }
    for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
        const uint32_t tile = flat_head / 32 + dim_tile;
        read_token_rows(get_noc_addr(tile, mega), stage_base + dim_tile * kDimBytes);
        for (uint32_t param = 0; param < 5; ++param) {
            const uint64_t src = param < 4
                ? get_noc_addr(tile, TensorAccessor(tensor_accessor::make_interleaved_dspec<tap_is_dram>(),
                                                    tap_addrs[param], kTileBytes))
                : get_noc_addr(tile, bias);
            const uint32_t dst = param_base + (param * 4 + dim_tile) * 128;
            noc_async_read(src, dst, 64);
            noc_async_read(src + 512, dst + 64, 64);
        }
    }
    for (uint32_t i = 0; i < 2 * kHalfRowBytes / sizeof(uint32_t); ++i) {
        reinterpret_cast<volatile uint32_t*>(zero_row)[i] = 0;
    }
    noc_async_read_barrier();
    for (uint32_t token = 0; token < num_tokens; ++token) {
        const uint32_t row = rows_base + (3 + token) * kRowBytes;
        for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
            const uint32_t block = stage_base + dim_tile * kDimBytes;
            copy_half_row(staged_row(block, token, false), row + dim_tile * 64);
            copy_half_row(staged_row(block, token, true), row + dim_tile * 64 + 32);
        }
    }
    if (worker == 0) {
        // Raw a/b rows: the token staging is free again once converted above.
        for (uint32_t tile = 0; tile < 2 * kScalarTiles; ++tile) {
            read_token_rows(get_noc_addr(kScalarTileBase + tile, mega), stage_base + tile * kDimBytes);
        }
    }

    // Walk the fixed tree to the FIR ancestor: sources 0-2 are the original live history
    // rows, never another branch; 3 + t is token t's row.
    auto source = [&](uint32_t token, uint32_t tap) {
        int32_t ancestor = token;
        for (uint32_t back = tap; back < 3; ++back) {
            ancestor = (fork_position && ancestor == 8) ? int32_t(fork_position) - 1 : ancestor - 1;
        }
        return rows_base + uint32_t(ancestor + 3) * kRowBytes;
    };
    // History transition: rows 1..3 of each token's window become its new rows 0..2.
    for (uint32_t token = 0; token < num_tokens; ++token) {
        for (uint32_t tap = 1; tap < 4; ++tap) {
            if constexpr (num_tokens == 1) {
                noc_async_write(source(token, tap), get_noc_addr(state_x, state_y,
                    state_addr + (tap - 1) * kHistoryPageBytes + flat_head * sizeof(uint16_t)), kRowBytes);
            } else {
                // Verification never mutates live history; retain every prefix.
                const auto candidate = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                       get_common_arg_val<uint32_t>(14 + token), kHistoryPageBytes);
                noc_async_write(source(token, tap), get_noc_addr(tap - 1, candidate) + flat_head * sizeof(uint16_t),
                                kRowBytes);
            }
        }
    }

    for (uint32_t pass = 0; pass < passes; ++pass) {
        const uint32_t first = pass * kPassTokens;
        const uint32_t count = num_tokens - first < kPassTokens ? num_tokens - first : kPassTokens;
        // Four taps (input and weight tiles), then the bias (weight tile only).
        for (uint32_t tap = 0; tap < 5; ++tap) {
            const bool bias_tile = tap == 4;
            if (!bias_tile) cb_reserve_back(kInputCb, 1);
            cb_reserve_back(kWeightCb, 1);
            const uint32_t input_dst = get_write_ptr(kInputCb);
            const uint32_t weight_dst = get_write_ptr(kWeightCb);
            // Preserve zero padding without clearing 1024 words on the reader.
            if (!bias_tile) noc.async_write_zeros(CircularBuffer(kInputCb), kTileBytes);
            noc.async_write_zeros(CircularBuffer(kWeightCb), kTileBytes);
            noc.write_zeros_l1_barrier();
            for (uint32_t slot = 0; slot < count; ++slot) {
                const uint32_t input = bias_tile ? 0 : source(first + slot, tap);
                for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
                    const uint32_t row = 8 * slot + 2 * dim_tile;
                    const uint32_t weight = param_base + (tap * 4 + dim_tile) * 128;
                    copy_half_row(weight, weight_dst + tile_row_offset(row, false));
                    copy_half_row(weight + 64, weight_dst + tile_row_offset(row, true));
                    if (!bias_tile) {
                        copy_half_row(input + dim_tile * 64, input_dst + tile_row_offset(row, false));
                        copy_half_row(input + dim_tile * 64 + 32, input_dst + tile_row_offset(row, true));
                    }
                }
            }
            if (!bias_tile) cb_push_back(kInputCb, 1);
            cb_push_back(kWeightCb, 1);
        }
        cb_wait_front(kOutputCb, 1);
        const uint32_t output_src = get_read_ptr(kOutputCb);
        for (uint32_t slot = 0; slot < count; ++slot) {
            const uint32_t token = first + slot;
            for (uint32_t copy = 0; copy < copies; ++copy) {
                const uint32_t output_head = head * copies + copy;
                for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
                    const uint32_t row = 8 * slot + 2 * dim_tile;
                    const uint64_t dst = get_noc_addr(token * kTokenTiles + (output_head / 32) * 4 + dim_tile, output);
                    noc_async_write(output_src + tile_row_offset(row, false),
                                    dst + tile_row_offset(output_head % 32, false), kHalfRowBytes);
                    noc_async_write(output_src + tile_row_offset(row, true),
                                    dst + tile_row_offset(output_head % 32, true), kHalfRowBytes);
                }
            }
            if (kPadRows > 0 && head < kPadRows) {
                for (uint32_t dim_tile = 0; dim_tile < 4; ++dim_tile) {
                    const uint64_t dst = get_noc_addr(token * kTokenTiles + (kHeadRowTiles - 1) * 4 + dim_tile, output);
                    noc_async_write(zero_row, dst + tile_row_offset(32 - kPadRows + head, false), kHalfRowBytes);
                    noc_async_write(zero_row, dst + tile_row_offset(32 - kPadRows + head, true), kHalfRowBytes);
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(kOutputCb, 1);
    }
    if (worker == 0) {
        // Scalar segments are padded to whole tiles; only heads 0..HV-1 are logical.
        noc_async_read_barrier();
        for (uint32_t tile = 0; tile < 2 * kScalarTiles; ++tile) {
            const uint32_t scalar = tile / kScalarTiles;
            const auto dst_accessor = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(),
                                                     scalar_addrs[scalar], kTileBytes);
            const uint64_t dst = get_noc_addr(tile % kScalarTiles, dst_accessor);
            const uint32_t block = stage_base + tile * kDimBytes;
            for (uint32_t token = 0; token < num_tokens; ++token) {
                noc_async_write(staged_row(block, token, false), dst + tile_row_offset(token, false), kHalfRowBytes);
                noc_async_write(staged_row(block, token, true), dst + tile_row_offset(token, true), kHalfRowBytes);
            }
        }
    }
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    for (uint32_t pass = 0; pass < passes; ++pass) {
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
