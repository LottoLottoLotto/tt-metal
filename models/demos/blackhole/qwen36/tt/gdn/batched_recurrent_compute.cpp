// SPDX-License-Identifier: Apache-2.0
// Keep BF16 rounding, normalization, decay, and tilize/untilize identical to B1. Only
// the key order differs: state rows arrive as DRAM-bank runs and q/k are permuted to
// match (see the reader), so key-dimension sums accumulate in a different order.
// The included routine consumes and releases every operand/state CB each time.
#define kernel_main recurrent_single_head
#include "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/recurrent_gated_delta_rule.cpp"
#undef kernel_main

void kernel_main() {
    const uint32_t jobs = get_arg_val<uint32_t>(0);
    for (uint32_t job = 0; job < jobs; ++job) recurrent_single_head();
}
