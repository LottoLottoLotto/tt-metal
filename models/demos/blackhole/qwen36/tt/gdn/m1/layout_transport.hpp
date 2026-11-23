// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Treat transport words as UInt16, not floating-point values. Live state and
// recurrence/output arithmetic remain BF16; only scratch CB interpretation
// changes. Native helpers select their integer layout paths automatically.
static_assert(!DST_ACCUM_MODE, "Bulk output compute configuration must remain unchanged");

inline void bulk_state_tilize() {
    compute_kernel_hw_startup(20, 21);
    compute_kernel_lib::tilize<4, 20, 21>(4);
}

inline void bulk_state_untilize() {
    // Reverse aliases have independent counters but reuse the same storage.
    compute_kernel_lib::untilize<4, 22, 23>(4);
}
