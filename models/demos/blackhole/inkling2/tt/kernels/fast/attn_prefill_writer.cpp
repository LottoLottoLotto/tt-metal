// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 attention prefill (fast/attn_prefill.py FlashPrefill), writer (BRISC): the compute's reduce scaler and
// column-identity tiles (as the stock SDPA writer), then each work item's gated output [Sq x DHt] tiles to
// out [1, 1, Tp, NQH * 128] (BF16, interleaved DRAM; head h = columns 128 h ..: the o_proj input layout).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t NQH = get_compile_time_arg_val(0);
constexpr uint32_t DHt = get_compile_time_arg_val(1);
constexpr uint32_t Sq = get_compile_time_arg_val(2);
constexpr uint32_t qn = get_compile_time_arg_val(3);
constexpr uint32_t cb_out = get_compile_time_arg_val(4);
constexpr uint32_t cb_identity_scale = get_compile_time_arg_val(5);
constexpr uint32_t cb_col_identity = get_compile_time_arg_val(6);
constexpr uint32_t identity_packed = get_compile_time_arg_val(7);
constexpr uint32_t cb_one = get_compile_time_arg_val(8);  // FP32 SUM scaler of the compute's row sums of squares
constexpr uint32_t kBf16 = 2048;
constexpr uint32_t kRowTiles = NQH * DHt;

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t global_q_start = get_arg_val<uint32_t>(1);
    const uint32_t global_q_count = get_arg_val<uint32_t>(2);
    const auto out = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), out_addr, kBf16);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_identity_scale, ckernel::PoolType::MAX,
                                                             ckernel::ReduceDim::REDUCE_ROW,
                                                             dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_packed);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_one, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW,
                                                             dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();

    constexpr uint32_t n = Sq * DHt;
    for (uint32_t it = 0; it < global_q_count; ++it) {
        const uint32_t gq = global_q_start + it;
        const uint32_t h = gq / qn;
        const uint32_t R0 = (gq % qn) * Sq;
        cb_wait_front(cb_out, n);
        uint32_t src = get_read_ptr(cb_out);
        for (uint32_t r = 0; r < Sq; ++r) {
            for (uint32_t d = 0; d < DHt; ++d) {
                noc_async_write(src, get_noc_addr((R0 + r) * kRowTiles + h * DHt + d, out), kBf16);
                src += kBf16;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, n);
    }
}
