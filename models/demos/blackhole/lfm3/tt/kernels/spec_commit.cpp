// SPDX-License-Identifier: Apache-2.0
// Speculative-verification commit for LFM3 (tt/speculate.py SpecCommit; the LFM2.5 kernel with the 7-tap conv's six
// state rows, extended to shared rounds): runs after the verify step's sampler / argmax, inside the same trace.
//
// The verify step's rows are groups, one per sequence (the row map, tt/speculate.py map_word: offset k in bits 0-7,
// group length n in 8-15, conv-state slot in 16-23, bit 31 active).  Group g = rows r0 .. r0 + n - 1 fed tok[r0] (the
// sequence's last committed token) and its drafts tok[r0 + 1 .. r0 + n - 1] at consecutive positions.  Row r's sampled
// token is ids[r] (stride SC_STRIDE words; with the exact sampler word 1 holds its flags, R_HOST = 1: the host redraws
// that row).  Accepted drafts of a group:
//   a = the first j in [0, n - 1] with j == n - 1, or row r0 + j flagged, or ids[r0 + j] != tok[r0 + j + 1].
// The group emits tok[r0 + 1 .. r0 + a] and the final token ids[r0 + a] (the host's redraw when flagged).
//
// Conv state: the verify step wrote every row's Bx to the layer's candidate tensor (row r = verify row r) and left the
// state s0..s5 (row = slot) unchanged.  Rows r0 .. r0 + a of a group were consumed, so its slot's new state is the
// last six rows of c = [s0, ..., s5, Bx_r0, ..., Bx_{r0 + a}]:  s_j <- c[a + 1 + j]  (j = 0..5; ascending j only reads
// rows not yet replaced; every group touches only its own slot's row).  Every worker (core, RISC lane) computes every
// group's a itself and commits its share of the (layer, column tile) jobs: per job it reads rows [0, SC_ROWS) of the
// candidate and state tiles (every slot of the round: slot < SC_ROWS), patches each group's slot row and writes the
// rows back (one worker per job: no two writers of a tile).  Worker 0 also writes the result page and, with
// SC_SAMPLED, appends each penalized group's emitted tokens the device knows (drafts 1..a, + the final token unless
// flagged) to its slot's sampler history.
//
// Result page (u32 x 64): [0, 16) a at each group's first row, [16, 32) ids per row, [32, 48) flags per row,
// [48] groups, [49] live rows.
// Common args: tok, pos, ids, result, params, row map, SC_SLOTS history slots, then (cand, s0, ..., s5) per conv layer.
// Per-worker args: [first job, jobs, master].
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t rows = SC_ROWS;
    constexpr uint32_t stride = SC_STRIDE;
    constexpr uint32_t tiles = SC_TILES;  // column tiles per conv state (hidden / 32)
    constexpr uint32_t slots = SC_SLOTS;  // sampler history slots passed (0 without SC_SAMPLED)
    constexpr uint32_t batch = SC_BATCH;  // jobs per read round
    constexpr uint32_t kTile = 2048;
    constexpr uint32_t kState = 6;        // conv_L_cache 7 - 1
    constexpr uint32_t kLayerArgs = 1 + kState;
    constexpr uint32_t kLayer0 = 6 + slots;  // first layer's common arg
    constexpr uint32_t F_ACTIVE = 1, F_PENALTY = 4, R_HOST = 1;
    constexpr uint32_t kActive = 1u << 31;
    // rows [0, rows) of faces 0 / 1: 64 B granules (DRAM reads need 64 B alignment)
    constexpr uint32_t face_bytes = (rows * 32 + 63) / 64 * 64;
    static_assert(rows >= 1 && rows <= 16);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t jobs = get_arg_val<uint32_t>(1);
    const bool master = get_arg_val<uint32_t>(2) != 0;
#if defined(COMPILE_FOR_BRISC)
    const uint32_t base = get_write_ptr(0);
#else
    const uint32_t base = get_write_ptr(1);
#endif
    // Scratch: tok 64 | pos 64 | map 64 | ids 16 x 64 | result 256 | hist 128 | params 128 | jobs batch x (cand 1024 |
    // s0..s5 6 x 1024); a tile image holds faces 0 / 1 (rows 0-15) at + 0 / + 512, row r at + 32 r.
    const uint32_t tok_l1 = base, pos_l1 = base + 64, map_l1 = base + 128, ids_l1 = base + 192;
    const uint32_t res_l1 = ids_l1 + 16 * 64, hist_l1 = res_l1 + 256, prm_l1 = hist_l1 + 128, job_l1 = prm_l1 + 128;
    constexpr uint32_t img = 1024;
    constexpr uint32_t job_bytes = img * (1 + kState);
    auto dram = [](uint32_t i, uint32_t page) {
        return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(i), page);
    };
    noc_async_read(dram(0, rows * 4).get_noc_addr(0), tok_l1, 64);
    noc_async_read(dram(1, rows * 4).get_noc_addr(0), pos_l1, 64);
    noc_async_read(dram(5, 64).get_noc_addr(0), map_l1, 64);
    if constexpr (stride == 1) {
        noc_async_read(dram(2, rows * 4).get_noc_addr(0), ids_l1, 64);
    } else {
        const auto ids = dram(2, stride * 4);
        for (uint32_t j = 0; j < rows; ++j) noc_async_read(ids.get_noc_addr(j), ids_l1 + j * 64, stride * 4);
    }
    noc_async_read_barrier();
    const volatile uint32_t* tok = reinterpret_cast<volatile uint32_t*>(tok_l1);
    const volatile int32_t* pos = reinterpret_cast<volatile int32_t*>(pos_l1);
    const volatile uint32_t* map = reinterpret_cast<volatile uint32_t*>(map_l1);
    auto id_of = [&](uint32_t j) -> uint32_t {
        return reinterpret_cast<volatile uint32_t*>(ids_l1 + (stride == 1 ? j * 4 : j * 64))[0];
    };
    auto flags_of = [&](uint32_t j) -> uint32_t {
        return stride == 1 ? 0u : reinterpret_cast<volatile uint32_t*>(ids_l1 + j * 64)[1];
    };
    // groups: (first row, accepted count, slot)
    uint32_t g_r0[16], g_a[16], g_slot[16];
    uint32_t groups = 0, live = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t m = map[r];
        if (pos[r] >= 0) ++live;
        if (!(m & kActive) || (m & 0xFFu) != 0) continue;
        const uint32_t n = (m >> 8) & 0xFFu;
        uint32_t a = 0;
        while (a + 1 < n && !(flags_of(r + a) & R_HOST) && id_of(r + a) == tok[r + a + 1]) ++a;
        g_r0[groups] = r;
        g_a[groups] = a;
        g_slot[groups] = (m >> 16) & 0xFFu;
        ++groups;
    }

    // conv state commit: job = layer * tiles + column tile
    for (uint32_t j0 = 0; j0 < jobs && groups; j0 += batch) {
        const uint32_t nb = jobs - j0 < batch ? jobs - j0 : batch;
        for (uint32_t q = 0; q < nb; ++q) {
            const uint32_t job = first + j0 + q, layer = job / tiles, c = job % tiles;
            const uint32_t l1 = job_l1 + q * job_bytes;
            for (uint32_t t = 0; t <= kState; ++t) {  // t 0: the candidate, 1..6: s0..s5
                const uint64_t src = dram(kLayer0 + kLayerArgs * layer + t, kTile).get_noc_addr(c);
                noc_async_read(src, l1 + t * img, face_bytes);
                noc_async_read(src + 512, l1 + t * img + 512, face_bytes);
            }
        }
        noc_async_read_barrier();
        for (uint32_t q = 0; q < nb; ++q) {
            const uint32_t job = first + j0 + q, layer = job / tiles, c = job % tiles;
            const uint32_t l1 = job_l1 + q * job_bytes;
            for (uint32_t g = 0; g < groups; ++g) {
                const uint32_t sr = g_slot[g] * 32;
                for (uint32_t s = 0; s < kState; ++s) {
                    const uint32_t src = g_a[g] + 1 + s;  // index into c = [s0..s5, Bx_r0..Bx_{r0+a}]
                    const uint32_t from = src < kState ? l1 + (1 + src) * img + sr
                                                       : l1 + (g_r0[g] + src - kState) * 32;
                    const uint32_t to = l1 + (1 + s) * img + sr;
                    for (uint32_t f = 0; f < 2; ++f) {
                        auto* d = reinterpret_cast<volatile uint32_t*>(to + 512 * f);
                        const auto* from_w = reinterpret_cast<volatile uint32_t*>(from + 512 * f);
                        for (uint32_t w = 0; w < 8; ++w) d[w] = from_w[w];
                    }
                }
            }
            for (uint32_t s = 0; s < kState; ++s) {
                const uint64_t dst = dram(kLayer0 + kLayerArgs * layer + 1 + s, kTile).get_noc_addr(c);
                noc_async_write(l1 + (1 + s) * img, dst, face_bytes);
                noc_async_write(l1 + (1 + s) * img + 512, dst + 512, face_bytes);
            }
        }
        noc_async_write_barrier();
    }

    if (master) {
        auto* res = reinterpret_cast<volatile uint32_t*>(res_l1);
        for (uint32_t w = 0; w < 64; ++w) res[w] = 0;
        for (uint32_t g = 0; g < groups; ++g) res[g_r0[g]] = g_a[g];
        for (uint32_t j = 0; j < rows; ++j) {
            res[16 + j] = id_of(j);
            res[32 + j] = flags_of(j);
        }
        res[48] = groups;
        res[49] = live;
        noc_async_write(res_l1, dram(3, 256).get_noc_addr(0), 256);
#if SC_SAMPLED
        // each penalized group's slot history += the emitted tokens the device knows (sampling.py commit()'s
        // read-modify-write); the row's params page holds its flags
        const auto params = dram(4, 128);
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t r0 = g_r0[g], a = g_a[g];
            noc_async_read(params.get_noc_addr(r0), prm_l1, 128);
            noc_async_read_barrier();
            const uint32_t pflags = reinterpret_cast<volatile uint32_t*>(prm_l1)[0];
            if ((pflags & (F_ACTIVE | F_PENALTY)) != (F_ACTIVE | F_PENALTY) || g_slot[g] >= slots) continue;
            const auto hist = dram(6 + g_slot[g], 128);
            const uint32_t n = a + ((flags_of(r0 + a) & R_HOST) ? 0 : 1);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t token = i < a ? tok[r0 + 1 + i] : id_of(r0 + a);
                noc_async_read(hist.get_noc_addr(token / 32), hist_l1, 128);
                noc_async_read_barrier();
                auto* h = reinterpret_cast<volatile uint32_t*>(hist_l1);
                h[token % 32] = h[token % 32] + 1;
                noc_async_write(hist_l1, hist.get_noc_addr(token / 32), 128);
                noc_async_write_barrier();
            }
        }
#endif
        noc_async_write_barrier();
    }
#endif
}
