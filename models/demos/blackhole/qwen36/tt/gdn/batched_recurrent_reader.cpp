// SPDX-License-Identifier: Apache-2.0
// State stream for (slot, value-head) jobs; the writer core-mate supplies the operands.
// A head's 128 state rows of 256 B are interleaved over the 8 DRAM banks, so bank b
// holds rows b, b+8, ..., b+120 contiguously. One 4-KiB read per bank lands them at
// working rows 16b..16b+15: working row m holds key row (m / 16) + 8 * (m % 16). The
// writer stores the same runs back and permutes q and k identically. The state CB holds
// two heads, so this stream runs a job ahead of compute.
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t kBanks = 8, kRun = 128 / kBanks * 256;
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t stride = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const auto state = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_arg_val<uint32_t>(3), 256);
    for (uint32_t job = 0; job < count; ++job) {
        const uint32_t index = first + job * stride;
        cb_reserve_back(8, 16);
        const uint32_t rows = get_write_ptr(8);
        for (uint32_t bank = 0; bank < kBanks; ++bank) {
            noc_async_read(get_noc_addr(index * 128 + bank, state), rows + bank * kRun, kRun);
        }
        noc_async_read_barrier();
        cb_push_back(8, 16);
    }
}
