// SPDX-License-Identifier: Apache-2.0
// One-launch paged K/V write for B1 decode (T=1) and MTP-1 verification (T=2): replaces, per token, the two
// position/page-table slices and the K and V paged_update_cache calls, and (T=2) the two copies of token 1's K/V
// shards to DRAM. Core (cache, head) owns one cache's head: for each token it reads the tile row holding the
// position (Wt tiles, read once when both tokens land in the same tile row), overwrites row pos % 32 with the
// token's head row and writes the tiles back. BF16 cache: the row's two face-row halves are copied. BFP8_b cache:
// each 16-value face row of the token is packed like the packer / host block-float path (shared exponent = max
// exponent, mantissa rounded to nearest even and saturated, sign dropped on zero mantissa) into its exponent byte
// and 16 mantissa bytes; the tile's other rows keep their bytes. Position -1 skips the token.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
// 16 BF16 values (8 words) -> one BFP8_b face row: shared exponent byte + 16 sign/magnitude mantissa bytes, as the
// host block-float packer (convert_u32_to_bfp<Bfp8_b, round-to-nearest-even>): max exponent over the row, the
// 24-bit mantissa (hidden bit set) shifted by the exponent gap, rounded to 7 bits (ties to even), saturated at 127;
// zero / denormal inputs give 0 and a zero mantissa drops the sign.
inline void pack_bfp8_row(volatile tt_l1_ptr uint32_t* in, volatile tt_l1_ptr uint8_t* exponent,
                          volatile tt_l1_ptr uint32_t* mantissas) {
    uint32_t v[16];
    uint32_t shared = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t pair = in[i];
        v[2 * i] = pair & 0xffffu;
        v[2 * i + 1] = pair >> 16;
    }
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t e = (v[i] >> 7) & 0xffu;
        shared = e > shared ? e : shared;
    }
    uint32_t words[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t e = (v[i] >> 7) & 0xffu;
        uint32_t byte = 0;
        if (e != 0) {
            uint32_t m = (1u << 23) | ((v[i] & 0x7fu) << 16);
            const uint32_t gap = shared - e;
            m = gap > 23 ? 0 : m >> gap;
            const uint32_t rest = m & 0x1ffffu;
            m >>= 17;
            if (rest > 0x10000u || (rest == 0x10000u && (m & 1u))) {
                ++m;
            }
            m = m > 127 ? 127 : m;
            byte = m == 0 ? 0 : (((v[i] >> 15) & 1u) << 7) | m;
        }
        words[i / 4] |= byte << (8 * (i % 4));
    }
    *exponent = static_cast<uint8_t>(shared);
    for (uint32_t i = 0; i < 4; ++i) {
        mantissas[i] = words[i];
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t tokens = get_compile_time_arg_val(0);
    constexpr uint32_t heads = get_compile_time_arg_val(1);
    constexpr uint32_t wt = get_compile_time_arg_val(2);
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t copy_last = get_compile_time_arg_val(4);
    constexpr uint32_t bfp8 = get_compile_time_arg_val(5);
    constexpr auto pos_args = TensorAccessorArgs<6>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto kc_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto ko_args = TensorAccessorArgs<vc_args.next_compile_time_args_offset()>();
    constexpr auto vo_args = TensorAccessorArgs<ko_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;                       // BF16 tiles (K/V shards, DRAM copies)
    constexpr uint32_t cache_tile_bytes = bfp8 ? 1088 : 2048;  // cache tiles
    constexpr uint32_t block_tiles = block_size / 32;
    static_assert(tokens >= 1 && tokens <= 2 && block_size % 32 == 0);

    const uint32_t pos_page = get_arg_val<uint32_t>(6);
    const uint32_t pt_page = get_arg_val<uint32_t>(7);
    const auto positions = TensorAccessor(pos_args, get_arg_val<uint32_t>(0), pos_page);
    const auto page_table = TensorAccessor(pt_args, get_arg_val<uint32_t>(1), pt_page);
    const auto k_cache = TensorAccessor(kc_args, get_arg_val<uint32_t>(2), cache_tile_bytes);
    const auto v_cache = TensorAccessor(vc_args, get_arg_val<uint32_t>(3), cache_tile_bytes);
    const auto k_out = TensorAccessor(ko_args, get_arg_val<uint32_t>(4), tile_bytes);
    const auto v_out = TensorAccessor(vo_args, get_arg_val<uint32_t>(5), tile_bytes);
    const uint32_t shard_x = get_arg_val<uint32_t>(8), shard_y = get_arg_val<uint32_t>(9);
    const uint32_t value = get_arg_val<uint32_t>(10);  // 0: K cache, 1: V cache
    const uint32_t head = get_arg_val<uint32_t>(11);
    uint32_t src[tokens];
    for (uint32_t t = 0; t < tokens; ++t) {
        src[t] = get_arg_val<uint32_t>(12 + 2 * t + value);  // token t's K / V shard address on the shard core
    }

    // Scratch: positions chunk [0, 64), page-table chunks [64, 64 + 64 * tokens), tile rows from 256 (two rows of
    // wt cache tiles), BFP8 input face rows (tokens x wt x 64 B), last-token copy staging.
    const uint32_t base = get_write_ptr(0);
    const uint32_t rows_base = base + 256;
    const uint32_t in_base = rows_base + 2 * wt * cache_tile_bytes;
    noc_async_read(get_noc_addr(0, positions), base, 64);
    noc_async_read_barrier();
    auto* pos_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    uint32_t pos[tokens];
    uint32_t chunk_word[tokens] = {};
    for (uint32_t t = 0; t < tokens; ++t) {
        pos[t] = pos_words[t];
        if (pos[t] == 0xffffffffu) {
            continue;
        }
        const uint32_t offset = (pos[t] / block_size) * 4;
        const uint32_t chunk = offset & ~63u;
        chunk_word[t] = (offset - chunk) / 4;
        noc_async_read(get_noc_addr(t, page_table) + chunk, base + 64 + 64 * t, 64);
    }
    noc_async_read_barrier();

    uint32_t row_id[tokens] = {};
    uint32_t buffer[tokens] = {};
    uint32_t buffers = 0;
    for (uint32_t t = 0; t < tokens; ++t) {
        if (pos[t] == 0xffffffffu) {
            continue;
        }
        const uint32_t block = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 64 + 64 * t)[chunk_word[t]];
        row_id[t] = ((block * heads + head) * block_tiles + (pos[t] % block_size) / 32) * wt;
        if (t == 1 && pos[0] != 0xffffffffu && row_id[1] == row_id[0]) {
            buffer[1] = buffer[0];
            continue;
        }
        buffer[t] = rows_base + buffers++ * wt * cache_tile_bytes;
        for (uint32_t c = 0; c < wt; ++c) {
            const uint64_t addr = value ? get_noc_addr(row_id[t] + c, v_cache) : get_noc_addr(row_id[t] + c, k_cache);
            noc_async_read(addr, buffer[t] + c * cache_tile_bytes, cache_tile_bytes);
        }
    }
    noc_async_read_barrier();
    for (uint32_t t = 0; t < tokens; ++t) {
        if (pos[t] == 0xffffffffu) {
            continue;
        }
        const uint32_t r = pos[t] % 32;
        if constexpr (bfp8) {
            // Token row h of each input tile: face 0 / face 1 halves -> 64 B per tile.
            for (uint32_t c = 0; c < wt; ++c) {
                const uint64_t in = get_noc_addr(shard_x, shard_y, src[t] + c * tile_bytes + head * 32);
                const uint32_t stage = in_base + (t * wt + c) * 64;
                noc_async_read(in, stage, 32);
                noc_async_read(in + 512, stage + 32, 32);
            }
        } else {
            const uint32_t face_row = r < 16 ? r * 32 : 1024 + (r - 16) * 32;
            for (uint32_t c = 0; c < wt; ++c) {
                const uint64_t in = get_noc_addr(shard_x, shard_y, src[t] + c * tile_bytes + head * 32);
                const uint32_t out = buffer[t] + c * cache_tile_bytes + face_row;
                noc_async_read(in, out, 32);
                noc_async_read(in + 512, out + 512, 32);
            }
        }
    }
    noc_async_read_barrier();
    if constexpr (bfp8) {
        for (uint32_t t = 0; t < tokens; ++t) {
            if (pos[t] == 0xffffffffu) {
                continue;
            }
            const uint32_t r = pos[t] % 32;
            const uint32_t face0 = r < 16 ? 0 : 2;
            const uint32_t face_row = r % 16;
            for (uint32_t c = 0; c < wt; ++c) {
                const uint32_t tile = buffer[t] + c * cache_tile_bytes;
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t face = face0 + half;
                    pack_bfp8_row(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_base + (t * wt + c) * 64 + half * 32),
                        reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile + face * 16 + face_row),
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + 64 + face * 256 + face_row * 16));
                }
            }
        }
    }
    for (uint32_t t = 0; t < tokens; ++t) {
        if (pos[t] == 0xffffffffu || (t == 1 && pos[0] != 0xffffffffu && row_id[1] == row_id[0])) {
            continue;
        }
        for (uint32_t c = 0; c < wt; ++c) {
            const uint64_t addr = value ? get_noc_addr(row_id[t] + c, v_cache) : get_noc_addr(row_id[t] + c, k_cache);
            noc_async_write(buffer[t] + c * cache_tile_bytes, addr, cache_tile_bytes);
        }
    }
    if constexpr (copy_last) {
        // The last token's K / V shard -> its DRAM tensor (commit candidate); this core copies its head's share.
        constexpr uint32_t share = wt / heads;
        const uint32_t stage = in_base + tokens * wt * 64;
        for (uint32_t c = 0; c < share; ++c) {
            noc_async_read(
                get_noc_addr(shard_x, shard_y, src[tokens - 1] + (head * share + c) * tile_bytes),
                stage + c * tile_bytes,
                tile_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t c = 0; c < share; ++c) {
            const uint32_t id = head * share + c;
            noc_async_write(
                stage + c * tile_bytes, value ? get_noc_addr(id, v_out) : get_noc_addr(id, k_out), tile_bytes);
        }
    }
    noc_async_write_barrier();
}
