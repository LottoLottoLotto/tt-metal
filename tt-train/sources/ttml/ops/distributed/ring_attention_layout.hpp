// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::ops::distributed {

enum class RingAttentionLayout : uint8_t { Contiguous = 0, ZigZag = 1 };

inline uint32_t zigzag_global_position(uint32_t device_order_pos, uint32_t seq_len, uint32_t cp_size) {
    const uint32_t chunk = seq_len / (2U * cp_size);
    const uint32_t slot = device_order_pos / chunk;
    const uint32_t rank = slot / 2U;
    const uint32_t global_chunk = (slot % 2U == 0U) ? rank : 2U * cp_size - 1U - rank;
    return global_chunk * chunk + device_order_pos % chunk;
}

}  // namespace ttml::ops::distributed
