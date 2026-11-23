// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::core::distributed {

// How the sequence is split over the CP axis. With N CP devices, Contiguous gives device d global block d of N;
// ZigZag gives it blocks d and 2N-1-d of 2N, which balances causal attention across the ring.
enum class CpSharding : uint8_t { Contiguous = 0, ZigZag = 1 };

// Global position of the token at `device_order_pos` when the sequence is laid out over the CP devices in order.
inline uint32_t cp_global_position(CpSharding sharding, uint32_t device_order_pos, uint32_t seq_len, uint32_t cp_size) {
    if (sharding == CpSharding::Contiguous) {
        return device_order_pos;
    }
    const uint32_t block = seq_len / (2U * cp_size);
    const uint32_t slot = device_order_pos / block;
    const uint32_t rank = slot / 2U;
    const uint32_t global_block = (slot % 2U == 0U) ? rank : 2U * cp_size - 1U - rank;
    return global_block * block + device_order_pos % block;
}

}  // namespace ttml::core::distributed
