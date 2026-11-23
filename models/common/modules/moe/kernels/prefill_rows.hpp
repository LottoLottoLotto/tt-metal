// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Masked prefill rows: one interleaved DRAM UINT32 page [segment_rows, segments,
// length_0, ..., length_{segments-1}]. Row t is real iff it lies within the leading
// length rows of its segment; every other row is padding and receives no experts.
struct PrefillRows {
    volatile tt_l1_ptr uint32_t* meta;

    void load(uint32_t address, uint32_t l1) {
        const auto rows = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), address, 128);
        noc_async_read(get_noc_addr(0, rows), l1, 128);
        noc_async_read_barrier();
        meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    }

    bool valid(uint32_t row) const {
        const uint32_t segment = row / meta[0];
        return segment < meta[1] && row - segment * meta[0] < meta[2 + segment];
    }
};
