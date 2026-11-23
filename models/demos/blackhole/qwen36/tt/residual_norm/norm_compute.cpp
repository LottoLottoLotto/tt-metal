// SPDX-License-Identifier: Apache-2.0
// No arithmetic edits: the original width-sharded RMSNorm runs after input DMA.
#define kernel_main residual_norm_native_compute
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp"
#undef kernel_main

void kernel_main() {
    cb_wait_front(0, get_named_compile_time_arg_val("rn_tiles"));
    residual_norm_native_compute();
}
