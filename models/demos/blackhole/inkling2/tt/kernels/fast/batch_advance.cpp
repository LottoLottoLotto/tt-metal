// SPDX-License-Identifier: Apache-2.0
// Batched decode step tail (fast/batch_tables.py BatchAdvance): advance rows b < R of the batch tables from their
// positions p_b (cur_pos word b) to q_b = p_b + 1 and log the step's tokens.  One core (BRISC).
//   cur_pos word b = q_b;  tau element (0, b) = tau_table[q_b] (host-tabulated fp32, as DecodeAdvance; unchanged
//   when q_b >= max_seq);  from_sel: token b = word 0 of sel page b (the device sampler's rows) -> tok word b;
//   else token b = tok word b (the LM head's argmax);  log page q_0 % log_pages = [tokens 0..15, flags 0..15
//   (from_sel; else 0), q 0..15] (words b >= R: zero).
// Rows b >= R are never read nor written (cur_pos / tau / tok words >= R keep their values: every 64-byte write
// carries back what was read).  Pad rows (replicas of row 0) advance like row 0.
// Common runtime args: cur_pos, tau, tau_table, tok, sel, log addresses.
#include <cstdint>
#include "read_vc.hpp"
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

void kernel_main() {
    ink_set_read_vc();
#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t R = get_named_compile_time_arg_val("rows");
    constexpr uint32_t max_seq = get_named_compile_time_arg_val("max_seq");
    constexpr uint32_t log_pages = get_named_compile_time_arg_val("log_pages");
    constexpr uint32_t from_sel = get_named_compile_time_arg_val("from_sel");
    constexpr uint32_t sel_page = get_named_compile_time_arg_val("sel_page");
    constexpr uint32_t tok_page = get_named_compile_time_arg_val("tok_page");
    static_assert(R >= 1 && R <= 16, "batch rows: 1..16 (words 0..15 of one 64-byte chunk)");
    auto acc = [](uint32_t arg, uint32_t page) {
        return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(arg), page);
    };
    const auto cur_pos = acc(0, 128);
    const auto tau = acc(1, 4096);
    const auto tau_table = acc(2, max_seq * 4);
    const auto tok = acc(3, tok_page);
    const auto sel = acc(4, sel_page);
    const auto log = acc(5, 256);
    // Scratch (64-byte aligned): cur_pos 64 | tau 64 | tok 64 | log 256 | sel R * 64 | tau chunks R * 64
    const uint32_t base = get_write_ptr(0);
    const uint32_t pos_l1 = base, tau_l1 = base + 64, tok_l1 = base + 128, log_l1 = base + 192;
    const uint32_t sel_l1 = base + 448, chunk_l1 = sel_l1 + 16 * 64;
    noc_async_read(cur_pos.get_noc_addr(0), pos_l1, 64);
    noc_async_read(tau.get_noc_addr(0), tau_l1, 64);
    noc_async_read(tok.get_noc_addr(0), tok_l1, 64);
    if constexpr (from_sel) {
        for (uint32_t b = 0; b < R; ++b) noc_async_read(sel.get_noc_addr(b), sel_l1 + b * 64, 64);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);
    for (uint32_t b = 0; b < R; ++b) {
        const uint32_t q = pos[b] + 1;
        if (q < max_seq) noc_async_read(tau_table.get_noc_addr(0, (q & ~15u) * 4), chunk_l1 + b * 64, 64);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* tw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tau_l1);
    volatile tt_l1_ptr uint32_t* tk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_l1);
    volatile tt_l1_ptr uint32_t* lg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(log_l1);
    for (uint32_t i = 0; i < 64; ++i) lg[i] = 0;
    for (uint32_t b = 0; b < R; ++b) {
        const uint32_t q = pos[b] + 1;
        if (q < max_seq) tw[b] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chunk_l1 + b * 64)[q & 15];
        pos[b] = q;
        if constexpr (from_sel) {
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel_l1 + b * 64);
            tk[b] = s[0];
            lg[16 + b] = s[1];
        }
        lg[b] = tk[b];
        lg[32 + b] = q;
    }
    noc_async_write(pos_l1, cur_pos.get_noc_addr(0), 64);
    noc_async_write(tau_l1, tau.get_noc_addr(0), 64);
    if constexpr (from_sel) noc_async_write(tok_l1, tok.get_noc_addr(0), 64);
    noc_async_write(log_l1, log.get_noc_addr(pos[0] % log_pages), 256);
    noc_async_write_barrier();
#endif
}
