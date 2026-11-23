// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "api/dataflow/dataflow_api.h"

// Four packed NoC-zero x/y words followed by the first local row. Metadata is
// decoded by _state_routes from TensorAccessorArgs; no core-placement guesses.
// Chunk lengths follow from the fixed 128-row head / 56-row shard geometry.
template <uint32_t route_arg, bool write_state>
inline void bulk_state_transfer(uint32_t state_address, uint32_t local_address) {
    uint32_t remaining = 128;
    uint32_t first_row = get_arg_val<uint32_t>(route_arg + 4);
    uint32_t shard = 0;
    while (remaining != 0) {
        const uint32_t available = 56 - first_row;
        const uint32_t rows = remaining < available ? remaining : available;
        const uint32_t bytes = rows * 256;
        const uint32_t xy = get_arg_val<uint32_t>(route_arg + shard);
        const uint64_t remote = get_noc_addr(xy >> 8, xy & 0xff, state_address + first_row * 256);
        if constexpr (write_state) {
            noc_async_write(local_address, remote, bytes);
        } else {
            noc_async_read(remote, local_address, bytes);
        }
        local_address += bytes;
        remaining -= rows;
        first_row = 0;
        ++shard;
    }
}
