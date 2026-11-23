// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 attention prefill K / V front (fast/attn_prefill.py KVFront), writer (BRISC): each unit's 4 output tiles
// (kv head g, kind kv, aligned-frame tile row i' = position tile start / 32 + i') to
//   full:    the paged cache [nblocks + 1, 8, 64, 128] (position tiles >= 2 nblocks are skipped: past the cache;
//            nblocks = max_seq_len / 64); PAGED (batched serving): the pool, logical block j at entry j of the page
//            row (runtime arg after nu: [1, W] int32; the entries of blocks s_t / 2 .. (s_t + Tpt) / 2 are read once);
//            tiles without a valid row (position tile >= (start + valid) / 32) are not written: their blocks may not
//            belong to the sequence (the flash reads the zero block there);
//   sliding: the chunk's K / V [1, 8, (Tpt + 1) 32, 128] (row tile i'), the flash prefill's / ring write's input.
// Program start: the compute's reduce scaler tile (SUM, REDUCE_ROW).
#ifndef PAGED
#define PAGED 0
#endif
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t Tpt = get_compile_time_arg_val(0);
constexpr uint32_t cb_ok = get_compile_time_arg_val(1);
constexpr uint32_t cb_ov = get_compile_time_arg_val(2);
constexpr uint32_t ok_bytes = get_compile_time_arg_val(3);
constexpr uint32_t ov_bytes = get_compile_time_arg_val(4);
constexpr bool SLIDING = get_compile_time_arg_val(5) != 0;
constexpr uint32_t cb_ws = get_compile_time_arg_val(6);  // writer scratch (the position page)
constexpr uint32_t cb_one = get_compile_time_arg_val(7); // the compute's reduce scaler (1.0)

FORCE_INLINE auto dram(uint32_t addr, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), addr, page);
}

void kernel_main() {
    const uint32_t k_addr = get_arg_val<uint32_t>(0);
    const uint32_t v_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2);
    const uint32_t nblocks = get_arg_val<uint32_t>(3);
    const uint32_t u0 = get_arg_val<uint32_t>(4);
    const uint32_t nu = get_arg_val<uint32_t>(5);
    const auto kd = dram(k_addr, ok_bytes);
    const auto vd = dram(v_addr, ov_bytes);
    const auto ps = dram(pos_addr, 128);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_one, ckernel::PoolType::SUM,
                                                             ckernel::ReduceDim::REDUCE_ROW,
                                                             dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    const uint32_t scratch = get_write_ptr(cb_ws);
    noc_async_read(get_noc_addr(0, ps), scratch, 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[0];
    [[maybe_unused]] const uint32_t end = start + reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[1];
    const uint32_t s_t = start >> 5, o = start & 31;
#if PAGED
    // page-table entries b0 .. of blocks s_t / 2 .. (s_t + Tpt) / 2 (<= 66), from the 64 B-aligned entry b0
    const uint32_t b0 = (s_t >> 1) & ~15u;
    {
        const auto pt = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_arg_val<uint32_t>(6), 64);
        const uint32_t last = ((s_t + Tpt) >> 1) < nblocks ? ((s_t + Tpt) >> 1) : nblocks - 1;
        noc_async_read(get_noc_addr(0, pt, b0 * 4), scratch + 128, ((last + 1 - b0) * 4 + 63) & ~63u);
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    const volatile tt_l1_ptr int32_t* pte = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scratch + 128);
#endif

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t gk = u / (Tpt + 1);
        const uint32_t ip = u % (Tpt + 1);
        const uint32_t kv = gk >> 3, g = gk & 7;
        if (ip == Tpt && o == 0) {
            continue;
        }
        const uint32_t cb = kv == 0 ? cb_ok : cb_ov;
        const uint32_t bytes = kv == 0 ? ok_bytes : ov_bytes;
        cb_wait_front(cb, 4);
        uint32_t src = get_read_ptr(cb);
        const uint32_t pt = s_t + ip;
        const bool write = SLIDING || (pt < 2 * nblocks && (!PAGED || pt * 32 < end));
        for (uint32_t c = 0; c < 4; ++c) {
            uint32_t page;
            if constexpr (SLIDING) {
                page = (g * (Tpt + 1) + ip) * 4 + c;
            } else {
#if PAGED
                const uint32_t blk = write ? static_cast<uint32_t>(pte[(pt >> 1) - b0]) : 0;
#else
                const uint32_t blk = pt >> 1;
#endif
                page = ((blk * 8 + g) * 2 + (pt & 1)) * 4 + c;
            }
            if (write) {
                noc_async_write(src, get_noc_addr(page, kv == 0 ? kd : vd), bytes);
            }
            src += bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb, 4);
    }
}
