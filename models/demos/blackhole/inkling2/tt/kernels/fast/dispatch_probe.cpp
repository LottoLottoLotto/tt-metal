// SPDX-License-Identifier: Apache-2.0
// Dispatch-cost probe (tests/prof_dispatch.py, owner Profiler): BRISC / NCRISC spin for common runtime arg 0 loop
// iterations, TRISC returns at once.  PAD (bytes) inflates each binary's text with a skipped .space block, so the cost
// of shipping kernel binaries can be measured; TAG / RTA only make distinct programs (one per kernel group / RTA count).
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#define PROBE_STR2(x) #x
#define PROBE_STR(x) PROBE_STR2(x)

void kernel_main() {
#if PAD > 0
    asm volatile("j 1f\n .space " PROBE_STR(PAD) "\n1:");
#endif
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
    const uint32_t n = get_common_arg_val<uint32_t>(0) + TAG * 0;
    for (volatile uint32_t i = 0; i < n; i = i + 1) {
    }
#endif
}
