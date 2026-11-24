// SPDX-License-Identifier: Apache-2.0
// 4-bit V cache support (QWEN36_KV_CACHE_DTYPE=bfp8_v4r): Y[unit] = X[unit] @ M for 256-wide head
// blocks, M = the randomized Hadamard R (V in) or R^T (attention output), HiFi4 with FP32
// accumulation, BF16 result. A unit is one 32-row tile row of one 256-column head block (8 tiles);
// a work item is one unit and V_ROT_COLS of its 8 output column tiles. Tile placement is strided
// (common runtime args), so the same program reads/writes token-major [rows, heads*256] and
// head-major [heads, rows, 256] tensors.
//   V_ROT_ROUND:   the writer rounds every face row of the BF16 result to its BFP4 grid
//                  (v_rotate_rne.hpp, round-to-nearest-even), which the truncating packer, the
//                  BFP4 paged_update_cache and rope_kv_many's packer conversion store exactly.
//   V_ROT_PAGED:   (implies rounding) the writer encodes BFP4 tiles itself and writes them straight
//                  into the paged V cache through the chunk page table (fused paged_fill_cache).
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_buffer.h"
#include "models/demos/blackhole/qwen36/tt/attention/v_rotate_rne.hpp"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace v_rotate {
constexpr uint32_t kIn = 0;       // 8 BF16 input tiles per item (double-buffered)
constexpr uint32_t kM = 1;        // BFP4 M column tiles (k-major: k * COLS + c), resident per column group
constexpr uint32_t kOut = 2;      // BF16 result tiles
constexpr uint32_t kBfp4 = 3;     // writer-owned BFP4 tile staging (paged mode)
constexpr uint32_t kTable = 4;    // writer-owned chunk page-table row (paged mode)
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kBfp4Bytes = 576;  // 64 exponent bytes (face-major, one per face row) + 512 nibble bytes
constexpr uint32_t kCols = V_ROT_COLS;
constexpr uint32_t kGroupsPerUnitRow = 8 / kCols;

struct Item {
    uint32_t row_block;
    uint32_t group;
    uint32_t col0;
};

// Common runtime args shared by the three processors (same layout on each).
// 0 in_addr, 1 m_addr, 2 out_addr (paged: V cache), 3 table_addr, 4 groups, 5 in_base, 6 in_rb_stride,
// 7 in_g_stride, 8 out_rb_stride, 9 out_g_stride, 10 valid_row_blocks, 11 table page bytes,
// 12 table read bytes (page bytes rounded up to 64), 13 rounded rows per tile (V_ROT_ROUND: rows >= this
// are left unrounded; decode/verification launches only round their T live token rows)
inline Item decode_item(uint32_t item, uint32_t groups) {
    const uint32_t unit = item / kGroupsPerUnitRow;
    return {unit / groups, unit % groups, (item % kGroupsPerUnitRow) * kCols};
}
}  // namespace v_rotate

void kernel_main() {
    using namespace v_rotate;
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t groups = get_common_arg_val<uint32_t>(4);
#if defined(COMPILE_FOR_NCRISC)
    const auto input = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<V_ROT_IN_DRAM>(), get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto matrix = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), kBfp4Bytes);
    const uint32_t in_base = get_common_arg_val<uint32_t>(5);
    const uint32_t in_rb_stride = get_common_arg_val<uint32_t>(6);
    const uint32_t in_g_stride = get_common_arg_val<uint32_t>(7);
    uint32_t loaded = 0xFFFFFFFF;
    for (uint32_t item = first; item < first + count; ++item) {
        const Item it = decode_item(item, groups);
        if (it.col0 != loaded) {
            // Compute pops the previous column group before this reserve can succeed.
            cb_reserve_back(kM, 8 * kCols);
            uint32_t dst = get_write_ptr(kM);
            for (uint32_t k = 0; k < 8; ++k) {
                for (uint32_t c = 0; c < kCols; ++c) {
                    noc_async_read(get_noc_addr(k * 8 + it.col0 + c, matrix), dst, kBfp4Bytes);
                    dst += kBfp4Bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(kM, 8 * kCols);
            loaded = it.col0;
        }
        cb_reserve_back(kIn, 8);
        uint32_t dst = get_write_ptr(kIn);
        const uint32_t tile0 = in_base + it.row_block * in_rb_stride + it.group * in_g_stride;
        for (uint32_t k = 0; k < 8; ++k) {
            noc_async_read(get_noc_addr(tile0 + k, input), dst, kTileBytes);
            dst += kTileBytes;
        }
        noc_async_read_barrier();
        cb_push_back(kIn, 8);
    }
#elif defined(COMPILE_FOR_BRISC)
    const uint32_t out_rb_stride = get_common_arg_val<uint32_t>(8);
    const uint32_t out_g_stride = get_common_arg_val<uint32_t>(9);
    const uint32_t round_rows = get_common_arg_val<uint32_t>(13);
#if defined(V_ROT_PAGED)
    // Paged V cache [blocks, groups, page_rows, 256] in BFP4 tiles; this chunk's row block rb is
    // logical block rb / page_tiles of the chunk page table (-1 entries are skipped, as in
    // paged_fill_cache), and only the first valid_row_blocks row blocks are written.
    constexpr uint32_t page_tiles = V_ROT_PAGE_ROWS / 32;
    const auto cache = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), kBfp4Bytes);
    const auto table = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<V_ROT_TABLE_DRAM>(), get_common_arg_val<uint32_t>(3),
        get_common_arg_val<uint32_t>(11));
    const uint32_t valid_row_blocks = get_common_arg_val<uint32_t>(10);
    cb_reserve_back(kTable, 1);
    const uint32_t table_ptr = get_write_ptr(kTable);
    if (count != 0) {
        noc_async_read(get_noc_addr(0, table), table_ptr, get_common_arg_val<uint32_t>(12));
        noc_async_read_barrier();
    }
    cb_reserve_back(kBfp4, 2);
    const uint32_t stage0 = get_write_ptr(kBfp4);
    uint32_t slot = 0;
#else
    const auto output = TensorAccessor(
        tensor_accessor::make_interleaved_dspec<V_ROT_OUT_DRAM>(), get_common_arg_val<uint32_t>(2), kTileBytes);
#endif
    for (uint32_t item = first; item < first + count; ++item) {
        const Item it = decode_item(item, groups);
        for (uint32_t c = 0; c < kCols; ++c) {
            cb_wait_front(kOut, 1);
            const uint32_t src = get_read_ptr(kOut);
#if defined(V_ROT_PAGED)
            const uint32_t physical = reinterpret_cast<volatile uint32_t*>(table_ptr)[it.row_block / page_tiles];
            if (it.row_block < valid_row_blocks && physical != 0xFFFFFFFF) {
                // Two staging tiles alternate so encoding overlaps the previous tile's write; one
                // barrier before reusing slot 0 retires both outstanding writes.
                const uint32_t stage = stage0 + slot * kBfp4Bytes;
                if (slot == 0) {
                    noc_async_write_barrier();
                }
                volatile uint8_t* exponents = reinterpret_cast<volatile uint8_t*>(stage);
                for (uint32_t row = 0; row < 64; ++row) {  // face-major face rows: face = row / 16
                    exponents[row] = static_cast<uint8_t>(v_rotate_rne::encode_row_bfp4(
                        reinterpret_cast<volatile uint32_t*>(src + row * 32),
                        reinterpret_cast<volatile uint32_t*>(stage + 64 + row * 8)));
                }
                const uint32_t tile = ((physical * groups + it.group) * page_tiles + it.row_block % page_tiles) * 8 +
                                      it.col0 + c;
                noc_async_write(stage, get_noc_addr(tile, cache), kBfp4Bytes);
                slot ^= 1;
            }
#else
#if defined(V_ROT_ROUND)
            // Tile row r spans face rows (r / 16) * 2 * 16 + r % 16 (left face) and +16 (right face).
            for (uint32_t r = 0; r < round_rows; ++r) {
                const uint32_t face_row = (r / 16) * 32 + r % 16;
                v_rotate_rne::round_row_bf16(reinterpret_cast<volatile uint32_t*>(src + face_row * 32));
                v_rotate_rne::round_row_bf16(reinterpret_cast<volatile uint32_t*>(src + (face_row + 16) * 32));
            }
#endif
            const uint32_t tile = it.row_block * out_rb_stride + it.group * out_g_stride + it.col0 + c;
            noc_async_write(src, get_noc_addr(tile, output), kTileBytes);
            noc_async_write_barrier();
#endif
            cb_pop_front(kOut, 1);
        }
    }
#if defined(V_ROT_PAGED)
    noc_async_write_barrier();
#endif
#elif defined(COMPILE_FOR_TRISC)
    compute_kernel_hw_startup<SrcOrder::Reverse>(kIn, kM, kOut);
    matmul_init(kIn, kM);
    uint32_t loaded = 0xFFFFFFFF;
    for (uint32_t item = first; item < first + count; ++item) {
        const Item it = decode_item(item, groups);
        if (it.col0 != loaded) {
            if (loaded != 0xFFFFFFFF) {
                cb_pop_front(kM, 8 * kCols);
            }
            cb_wait_front(kM, 8 * kCols);
            loaded = it.col0;
        }
        cb_wait_front(kIn, 8);
        for (uint32_t c = 0; c < kCols; ++c) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < 8; ++k) {
                matmul_tiles(kIn, kM, k, k * kCols + c, 0);
            }
            tile_regs_commit();
            cb_reserve_back(kOut, 1);
            tile_regs_wait();
            pack_tile(0, kOut);
            tile_regs_release();
            cb_push_back(kOut, 1);
        }
        cb_pop_front(kIn, 8);
    }
    if (loaded != 0xFFFFFFFF) {
        cb_pop_front(kM, 8 * kCols);
    }
#endif
}
