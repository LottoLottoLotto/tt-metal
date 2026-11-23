// SPDX-License-Identifier: Apache-2.0
// Shared row-set rule of the shared-KV verification SDPA (verify_sdpa.py).
//
// Verification rows r = 0..T-1 attend causally at consecutive positions pos[r]. The reference
// (per-row paged SDPA decode, 8 cores per (row, KV head), k_chunk 64) splits each row's
// ceil((pos+1)/64) K chunks over its 8 cores; rows with equal floor(pos/64) therefore share
// every per-core chunk range and the whole tree-reduction schedule. A "pass" is the set of rows
// whose floor(pos/64) equals floor(pos[0]/64) + pass: all its rows use one partition, computed
// from a representative position, and differ only in the per-row causal mask of the last chunk.
// T <= 8 consecutive rows span at most two chunk counts, so two passes cover them.
//
// Q packs 4 rows x 6 heads per tile ("block" b holds rows 4b..4b+3). Every non-empty
// (pass, block) pair is one work item; items are enumerated pass-major and item s runs on core
// slot s (32 cores: 4 KV heads x 8). Neighbouring items of one pass read identical K/V chunk
// sequences, so their cores split each chunk's reads (lower slot K, higher slot V) and exchange
// the halves over the NoC.
#pragma once
#include <cstdint>

namespace verify_sdpa {

constexpr uint32_t kHeadsPerKv = 6;       // 24 Q heads / 4 KV heads
constexpr uint32_t kRowsPerTile = 4;      // 4 rows x 6 heads = 24 of 32 tile rows
constexpr uint32_t kNoPartner = 0xFFFFFFFF;

struct SlotWork {
    uint32_t rows = 0;                     // bit mask of this item's rows
    uint32_t block = 0;                    // Q row block (rows 4*block .. 4*block+3)
    uint32_t representative = UINT32_MAX;  // a row position of the item's pass (UINT32_MAX: idle slot)
    uint32_t partner = kNoPartner;         // slot sharing this item's K/V chunk sequence
};

template <uint32_t T, uint32_t k_chunk_size>
inline SlotWork slot_work(const volatile uint32_t* pos, uint32_t slot) {
    constexpr uint32_t blocks = (T + kRowsPerTile - 1) / kRowsPerTile;
    const uint32_t first_chunk = pos[0] / k_chunk_size;
    uint32_t item_rows[2 * blocks];
    uint32_t item_pass[2 * blocks];
    uint32_t item_block[2 * blocks];
    uint32_t item_rep[2 * blocks];
    uint32_t items = 0;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t block = 0; block < blocks; ++block) {
            uint32_t mask = 0;
            uint32_t rep = UINT32_MAX;
            for (uint32_t r = block * kRowsPerTile; r < T && r < (block + 1) * kRowsPerTile; ++r) {
                if (pos[r] / k_chunk_size == first_chunk + pass) {
                    mask |= 1u << r;
                    rep = pos[r];
                }
            }
            if (mask != 0) {
                item_rows[items] = mask;
                item_pass[items] = pass;
                item_block[items] = block;
                item_rep[items] = rep;
                ++items;
            }
        }
    }
    SlotWork work;
    if (slot >= items) {
        return work;
    }
    work.rows = item_rows[slot];
    work.block = item_block[slot];
    work.representative = item_rep[slot];
    // A pass spans at most two blocks, so at most one neighbouring item shares the pass.
    if (slot > 0 && item_pass[slot - 1] == item_pass[slot]) {
        work.partner = slot - 1;
    } else if (slot + 1 < items && item_pass[slot + 1] == item_pass[slot]) {
        work.partner = slot + 1;
    }
    return work;
}

}  // namespace verify_sdpa
