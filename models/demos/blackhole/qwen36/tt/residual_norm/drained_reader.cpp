// SPDX-License-Identifier: Apache-2.0
// The native sharded RMSNorm multicast reader, unchanged, then drained: its non-posted semaphore increments (and
// writes) complete before the kernel exits. Otherwise the next program on this core and NoC can take its NoC
// transaction-counter baseline before a late acknowledgement lands, and its own atomic barrier never returns (the
// LM head's in0 receivers end with one). Same communication and arithmetic as the native kernel.
#define kernel_main drained_native_reader
#ifdef DRAIN_SENDER
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp"
#else
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp"
#endif
#undef kernel_main

void kernel_main() {
    drained_native_reader();
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}
