// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Decode (and rows-glue) kernels: the NoC read request VC, set explicitly at kernel start (fast/NOTES-prefill.md
// section 6).
//
// In dedicated-NOC mode a plain noc_async_read never writes its command buffer's NOC_CTRL, so it inherits the static
// read VC whatever ran last on that core / NOC left there (the firmware sets VC 1 once; a set_state<true>(.., vc)
// read in a prefill kernel changes it).  Decode speed then depended on the prefill that ran before (~1 %).
// The setters write the VC of this RISC's read command buffer on both NOCs (some kernels read on the other NOC, e.g.
// moe_decode's swapped stream cores; both RISCs write the same per-core value, so a shared command buffer is
// harmless); no-ops on the compute RISCs, so a kernel file shared by all RISCs calls one at the top of every
// kernel_main.  The outputs do not depend on the VC (bitwise).  x, y: absolute logical worker coordinates.
//
// ink_set_read_vc(): VC y & 3, every decode / verify / drafter data-movement kernel.  Fastest of the sweep (plain
// traced decode device ms at p128 / p2k, model 4096, logs/dev/vc_*.log): one VC on every core 11.16 / 11.57-11.58,
// (x + y) & 3 11.16 / 11.69, x & 3 11.19 / 11.68, VC alternating with the row (y & 3, y >> 1, y & 1, 2 (y & 1), ...)
// 10.99-11.04 / 11.45-11.47, this one 11.02 / 11.47; release runtime 11.05 / 11.52, no explicit VC after the current
// prefill 11.15 / 11.62.  At long context (native plan, decode ms at 128K / 256K, logs/dev/vc2_*.log) y & 3
// 26.95 / 31.30 is the best too (no explicit VC 26.90 / 31.80, VC 1 27.03 / 31.87, 2 (y & 1) 27.45-27.79 /
// 32.28-33.05).
// ink_set_read_vc_rows(): VC y & 1, glue_rows.cpp (prefill rows glue + speculative verify).  32K prefill (fast
// profile, chunk 4096, logs/dev/vc3_*.log): y & 1 9.100 s, 2 (y & 1) 9.115, y & 3 9.354, VC 0 9.378, none 9.509.
#pragma once

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE void ink_write_read_vc(uint32_t vc) {
    for (uint32_t noc = 0; noc < NUM_NOCS; ++noc) {
        while (!noc_cmd_buf_ready(noc, read_cmd_buf)) {
        }
        NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CTRL,
                              NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc));
    }
}

FORCE_INLINE void ink_set_read_vc() { ink_write_read_vc(get_absolute_logical_y() & 3); }
FORCE_INLINE void ink_set_read_vc_rows() { ink_write_read_vc(get_absolute_logical_y() & 1); }
#else
inline void ink_set_read_vc() {}
inline void ink_set_read_vc_rows() {}
#endif
