// SPDX-License-Identifier: Apache-2.0
// Speculative-verification commit (tt/speculate.py SpecCommit): runs after the verify step's sampler / argmax, inside
// the same trace.
//
// Rows j = 0..K of the verify step fed tok[0] (the last committed token) and the drafts tok[1..K] at positions
// pos[0] + j (rows with pos < 0 are inactive).  Row j's sampled token is ids[j] (stride SC_STRIDE words; with the
// exact sampler word 1 holds its flags, R_HOST = 1: the host redraws that row).  Accepted drafts:
//   a = the first j in [0, K] with j == K, or row j flagged, or ids[j] != tok[j + 1].
// The step emits tok[1..a] and the final token ids[a] (the host's redraw when flagged).
//
// Every worker (core, RISC lane) computes a itself and commits its share of the conv states, (layer, column tile)
// jobs: s0 <- Bx_{a-1} (a = 0: the old s1), s1 <- Bx_a, from the verify step's candidate rows (state row 0; the tile's
// other rows are kept).  Worker 0 also writes the result page [a, final token, final flags, live rows, ids...] and,
// with SC_SAMPLED and a penalized row 0, appends tok[1..a] (+ the final token unless flagged) to history slot 0.
//
// Common args: tok, pos, ids, result, params, history slot 0, then (cand, s0, s1) per conv layer.
// Per-worker args: [first job, jobs, master].
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t rows = SC_ROWS;
    constexpr uint32_t stride = SC_STRIDE;
    constexpr uint32_t tiles = SC_TILES;  // column tiles per conv state (hidden / 32)
    constexpr uint32_t batch = 8;         // jobs per read round
    constexpr uint32_t kTile = 2048;
    constexpr uint32_t F_ACTIVE = 1, F_PENALTY = 4, R_HOST = 1;
    static_assert(rows >= 1 && rows <= 16);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t jobs = get_arg_val<uint32_t>(1);
    const bool master = get_arg_val<uint32_t>(2) != 0;
#if defined(COMPILE_FOR_BRISC)
    const uint32_t base = get_write_ptr(0);
#else
    const uint32_t base = get_write_ptr(1);
#endif
    // Scratch: tok 64 | pos 64 | ids 16 x 64 | result 128 | hist 128 | jobs batch x (cand 1024 | s0 128 | s1 128)
    const uint32_t tok_l1 = base, pos_l1 = base + 64, ids_l1 = base + 128, res_l1 = ids_l1 + 16 * 64;
    const uint32_t hist_l1 = res_l1 + 128, job_l1 = hist_l1 + 128;
    constexpr uint32_t job_bytes = 1024 + 128 + 128;
    auto dram = [](uint32_t i, uint32_t page) {
        return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(i), page);
    };
    noc_async_read(dram(0, rows * 4).get_noc_addr(0), tok_l1, 64);
    noc_async_read(dram(1, rows * 4).get_noc_addr(0), pos_l1, 64);
    if constexpr (stride == 1) {
        noc_async_read(dram(2, rows * 4).get_noc_addr(0), ids_l1, 64);
    } else {
        const auto ids = dram(2, stride * 4);
        for (uint32_t j = 0; j < rows; ++j) noc_async_read(ids.get_noc_addr(j), ids_l1 + j * 64, stride * 4);
    }
    noc_async_read_barrier();
    const volatile uint32_t* tok = reinterpret_cast<volatile uint32_t*>(tok_l1);
    const volatile int32_t* pos = reinterpret_cast<volatile int32_t*>(pos_l1);
    auto id_of = [&](uint32_t j) -> uint32_t {
        return reinterpret_cast<volatile uint32_t*>(ids_l1 + (stride == 1 ? j * 4 : j * 64))[0];
    };
    auto flags_of = [&](uint32_t j) -> uint32_t {
        return stride == 1 ? 0u : reinterpret_cast<volatile uint32_t*>(ids_l1 + j * 64)[1];
    };
    uint32_t live = 0;
    while (live < rows && pos[live] >= 0) ++live;
    const uint32_t k = live ? live - 1 : 0;
    uint32_t a = 0;
    while (a < k && !(flags_of(a) & R_HOST) && id_of(a) == tok[a + 1]) ++a;

    // conv state commit: job = layer * tiles + column tile
    for (uint32_t j0 = 0; j0 < jobs; j0 += batch) {
        const uint32_t nb = jobs - j0 < batch ? jobs - j0 : batch;
        for (uint32_t q = 0; q < nb; ++q) {
            const uint32_t job = first + j0 + q, layer = job / tiles, c = job % tiles;
            const uint32_t l1 = job_l1 + q * job_bytes;
            const uint64_t cand = dram(6 + 3 * layer, kTile).get_noc_addr(c);
            const uint64_t s0 = dram(7 + 3 * layer, kTile).get_noc_addr(c);
            const uint64_t s1 = dram(8 + 3 * layer, kTile).get_noc_addr(c);
            noc_async_read(cand, l1, 512);
            noc_async_read(cand + 512, l1 + 512, 512);
            noc_async_read(s0, l1 + 1024, 64);
            noc_async_read(s0 + 512, l1 + 1024 + 64, 64);
            noc_async_read(s1, l1 + 1152, 64);
            noc_async_read(s1 + 512, l1 + 1152 + 64, 64);
        }
        noc_async_read_barrier();
        for (uint32_t q = 0; q < nb; ++q) {
            const uint32_t job = first + j0 + q, layer = job / tiles, c = job % tiles;
            const uint32_t l1 = job_l1 + q * job_bytes;
            for (uint32_t f = 0; f < 2; ++f) {
                // row 0 of face f: s0 at l1 + 1024 + 64 f, s1 at l1 + 1152 + 64 f; cand row r at l1 + 512 f + 32 r
                auto* d0 = reinterpret_cast<volatile uint32_t*>(l1 + 1024 + 64 * f);
                auto* d1 = reinterpret_cast<volatile uint32_t*>(l1 + 1152 + 64 * f);
                const auto* src0 = reinterpret_cast<volatile uint32_t*>(a ? l1 + 512 * f + 32 * (a - 1) : l1 + 1152 + 64 * f);
                const auto* src1 = reinterpret_cast<volatile uint32_t*>(l1 + 512 * f + 32 * a);
                for (uint32_t w = 0; w < 8; ++w) d0[w] = src0[w];  // before s1's row 0 is replaced
                for (uint32_t w = 0; w < 8; ++w) d1[w] = src1[w];
            }
            const uint64_t s0 = dram(7 + 3 * layer, kTile).get_noc_addr(c);
            const uint64_t s1 = dram(8 + 3 * layer, kTile).get_noc_addr(c);
            noc_async_write(l1 + 1024, s0, 64);
            noc_async_write(l1 + 1024 + 64, s0 + 512, 64);
            noc_async_write(l1 + 1152, s1, 64);
            noc_async_write(l1 + 1152 + 64, s1 + 512, 64);
        }
        noc_async_write_barrier();
    }

    if (master) {
        const uint32_t final_flags = flags_of(a);
        auto* res = reinterpret_cast<volatile uint32_t*>(res_l1);
        for (uint32_t w = 0; w < 32; ++w) res[w] = 0;
        res[0] = a;
        res[1] = id_of(a);
        res[2] = final_flags;
        res[3] = live;
        for (uint32_t j = 0; j < rows; ++j) res[4 + j] = id_of(j);
        noc_async_write(res_l1, dram(3, 128).get_noc_addr(0), 128);
#if SC_SAMPLED
        // history slot 0 += the emitted tokens the device knows (sampling.py commit()'s read-modify-write)
        noc_async_read(dram(4, 128).get_noc_addr(0), hist_l1, 128);
        noc_async_read_barrier();
        const uint32_t pflags = reinterpret_cast<volatile uint32_t*>(hist_l1)[0];
        if ((pflags & (F_ACTIVE | F_PENALTY)) == (F_ACTIVE | F_PENALTY)) {
            const auto hist = dram(5, 128);
            const uint32_t n = a + ((final_flags & R_HOST) ? 0 : 1);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t token = i < a ? tok[i + 1] : id_of(a);
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
