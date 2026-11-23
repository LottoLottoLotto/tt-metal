// SPDX-License-Identifier: Apache-2.0
// Self-feeding decode: advance the attention decode tables from position p (cur_pos) to q = p + 1 on the device and
// log the step's selected token.  One core: NCRISC writes the full-layer gather window, BRISC everything else.
// Every value equals attention.AttnDecodeTables.update(q) (host): cur_pos = q, ring_pos = q % 1088,
// ring_idx[s] = s <= q and d = (q - s) mod 1088 < 1024 ? d : MASKED (s < 1088; the 64-entry tail is constant),
// full_idx[c] = c > q ? MASKED : (q - c < 1024 ? q - c : ZERO) (c < full_len; the tail is constant): only the
// window c in [q - 1024, q] can differ from the table of p, so only it is rewritten (the table must hold p's values:
// the host seeds it with update()), tau = tau_table[q] (host-tabulated fp32, element (0, 0) of the tile; the rest of
// the first face row stays zero).  Log: page q % log_pages of `log` = [sel words 0..2 (token, flags, n), q].
// Common runtime args: cur_pos, ring_pos, ring_idx, full_idx, tau, tau_table, sel, log addresses.
#include <cstdint>
#include "read_vc.hpp"
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
    ink_set_read_vc();
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t RING = 1024, SLOTS = 1088, ZERO = 1024, MASKED = 1025;
    constexpr uint32_t full_len = get_named_compile_time_arg_val("full_len");
    constexpr uint32_t max_seq = get_named_compile_time_arg_val("max_seq");
    constexpr uint32_t log_pages = get_named_compile_time_arg_val("log_pages");
    constexpr uint32_t ring_page = get_named_compile_time_arg_val("ring_page");  // bytes of the ring_idx row
    constexpr uint32_t full_page = get_named_compile_time_arg_val("full_page");  // bytes of the full_idx row
    auto acc = [](uint32_t arg, uint32_t page) {
        return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(arg), page);
    };
    const auto cur_pos = acc(0, 4);
    const uint32_t base = get_write_ptr(0);  // CB 0: BRISC [0, 8 KB), NCRISC [8 KB, 16 KB)
    const uint32_t sem = get_semaphore(0);
#if defined(COMPILE_FOR_NCRISC)
    const uint32_t l1 = base + 8192;
    noc_async_read(cur_pos.get_noc_addr(0), l1, 64);
    noc_async_read_barrier();
    const uint32_t q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1)[0] + 1;
    // p is read: BRISC may overwrite cur_pos now (same core: a plain L1 store to the semaphore word).
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), 1);
    if (q < full_len) {
        const uint32_t c0 = (q > RING ? q - RING : 0) & ~15u;
        uint32_t c1 = (q + 16) & ~15u;
        if (c1 > full_len) c1 = full_len;
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 64);
        for (uint32_t c = c0; c < c1; ++c) {
            w[c - c0] = c > q ? MASKED : (q - c < RING ? q - c : ZERO);
        }
        const auto full_idx = acc(3, full_page);
        noc_async_write(l1 + 64, full_idx.get_noc_addr(0, c0 * 4), (c1 - c0) * 4);
        noc_async_write_barrier();
    }
#else
    const uint32_t l1 = base;
    const auto sel = acc(6, 64);
    const auto log = acc(7, 64);
    noc_async_read(cur_pos.get_noc_addr(0), l1, 64);
    noc_async_read(sel.get_noc_addr(0), l1 + 64, 64);
    noc_async_read_barrier();
    const uint32_t q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1)[0] + 1;
    volatile tt_l1_ptr uint32_t* lg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 64);
    lg[3] = q;
    noc_async_write(l1 + 64, log.get_noc_addr(q % log_pages), 64);
    if (q < max_seq) {
        // tau_table[q]: its 64 B-aligned chunk, then (q % 16)
        const auto tau_table = acc(5, max_seq * 4);
        noc_async_read(tau_table.get_noc_addr(0, (q & ~15u) * 4), l1 + 128, 64);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 192);
        t[0] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 128)[q & 15];
        for (uint32_t i = 1; i < 16; ++i) t[i] = 0;
        noc_async_write(l1 + 192, acc(4, 4096).get_noc_addr(0), 64);
    }
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 1024);
    const uint32_t qm = q % SLOTS;  // (q - s) mod 1088 = (qm - s) mod 1088
    for (uint32_t s = 0; s < SLOTS; ++s) {
        const uint32_t d = s <= qm ? qm - s : qm + SLOTS - s;
        r[s] = s <= q && d < RING ? d : MASKED;
    }
    noc_async_write(l1 + 1024, acc(2, ring_page).get_noc_addr(0), SLOTS * 4);
    volatile tt_l1_ptr uint32_t* pp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 256);
    pp[0] = q;
    pp[16] = qm;
    noc_async_write(l1 + 320, acc(1, 4).get_noc_addr(0), 4);
    // cur_pos last, after NCRISC has read p
    volatile tt_l1_ptr uint32_t* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    noc_semaphore_wait(count, 1);
    noc_semaphore_set(count, 0);
    noc_async_write(l1 + 256, cur_pos.get_noc_addr(0), 4);
    noc_async_write_barrier();
#endif
#endif
}
