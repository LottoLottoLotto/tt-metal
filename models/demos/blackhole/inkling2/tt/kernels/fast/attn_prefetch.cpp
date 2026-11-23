// SPDX-License-Identifier: Apache-2.0
// Attention weight prefetch (fast/attn_decode.py AttnPrefetch, owner AttnFast): one data-movement kernel on the cores
// of the L1 prefetch buffer, added to the glue programs that precede an attention application (MID of the previous
// application, PRE of this one), which leave DRAM mostly idle.  Copies weight blocks of the next application's
// projection / o_proj workers from DRAM into this core's shard of the buffer; the stream workers then read those
// blocks from L1 (decode_matmul_stream.cpp, `pf`).
// The copies start `delay` wall-clock cycles after the kernel: the glue reads its own DRAM inputs first (~5 us), and
// queued behind megabytes of prefetch those small reads would stall the glue's critical path.
// Common args: [buffer address, delay].  Per-core args: [n, then n x (DRAM bank, bank address, bytes, shard offset)].
#include <cstdint>
#include "read_vc.hpp"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    ink_set_read_vc();
    const uint32_t base = get_common_arg_val<uint32_t>(0);
    const uint32_t delay = get_common_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_timestamp_32b();
    while (get_timestamp_32b() - t0 < delay) {
    }
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t a = 1 + 4 * i;
        noc_async_read(get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1)),
                       base + get_arg_val<uint32_t>(a + 3), get_arg_val<uint32_t>(a + 2));
    }
    noc_async_read_barrier();
}
