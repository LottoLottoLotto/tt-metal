// SPDX-License-Identifier: Apache-2.0
// MTP-1 commit of every GDN layer in one launch: select candidate S1/S2 (and its conv rows) by the device accept
// flag, untilize, and write into each layer's live L1 state. Pure data movement + lossless BF16 untilize.
// Jobs j = layer * heads + head are dealt round-robin over the worker cores. Per job:
//   BRISC  reads the head's 16 candidate tiles (4 blocks of 4) into CB0, then copies the job's 512-byte conv stripes;
//   TRISC  untilizes each block CB0 -> CB16;
//   NCRISC writes each untilized 32-row block into the height-sharded live state shards.
// Common args: [flag, per layer (s1, s2, c1, c2, state, conv, conv_core_xy) ..., shard core xy ...]; xy = x | y << 16.
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#endif

constexpr uint32_t kHeads = get_named_compile_time_arg_val("heads");
constexpr uint32_t kLayers = get_named_compile_time_arg_val("layers");
constexpr uint32_t kWorkers = get_named_compile_time_arg_val("workers");
constexpr uint32_t kJobs = kLayers * kHeads;
constexpr uint32_t kLayerArgs = 7;
constexpr uint32_t kCoresOffset = 1 + kLayers * kLayerArgs;

FORCE_INLINE uint32_t job_count(uint32_t worker) { return worker < kJobs ? (kJobs - worker + kWorkers - 1) / kWorkers : 0; }

void kernel_main() {
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t jobs = job_count(worker);
#if defined(COMPILE_FOR_BRISC)
    constexpr bool flag_dram = get_named_compile_time_arg_val("flag_dram");
    constexpr uint32_t conv_row_bytes = get_named_compile_time_arg_val("conv_width") * sizeof(uint16_t);
    constexpr uint32_t stripe_bytes = 512;
    const uint32_t scratch = get_write_ptr(2);
    const auto flag = TensorAccessor(tensor_accessor::make_interleaved_dspec<flag_dram>(), get_common_arg_val<uint32_t>(0), 32);
    noc_async_read(get_noc_addr(0, flag), scratch, 32);
    noc_async_read_barrier();
    const uint32_t accepted = *reinterpret_cast<volatile uint32_t*>(scratch) != 0;
    for (uint32_t n = 0; n < jobs; ++n) {
        const uint32_t job = worker + n * kWorkers;
        const uint32_t layer = job / kHeads;
        const uint32_t head = job % kHeads;
        const uint32_t base = 1 + layer * kLayerArgs;
        const auto candidate = TensorAccessor(
            tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(base + accepted), 2048);
        for (uint32_t block = 0; block < 4; ++block) {
            cb_reserve_back(0, 4);
            const uint32_t input = get_write_ptr(0);
            for (uint32_t tile = 0; tile < 4; ++tile) {
                noc_async_read(get_noc_addr(head * 16 + block * 4 + tile, candidate), input + tile * 2048, 2048);
            }
            noc_async_read_barrier();
            cb_push_back(0, 4);
        }
        // Each head owns one disjoint 512-byte stripe of the layer's three conv rows (host checks coverage).
        const uint32_t stripe_offset = head * stripe_bytes;
        if (stripe_offset < conv_row_bytes) {
            const auto conv = TensorAccessor(
                tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(base + 2 + accepted),
                conv_row_bytes);
            const uint32_t conv_addr = get_common_arg_val<uint32_t>(base + 5);
            const uint32_t xy = get_common_arg_val<uint32_t>(base + 6);
            for (uint32_t row = 0; row < 3; ++row) {
                noc_async_read(get_noc_addr(row, conv) + stripe_offset, scratch + row * stripe_bytes, stripe_bytes);
            }
            noc_async_read_barrier();
            for (uint32_t row = 0; row < 3; ++row) {
                noc_async_write(
                    scratch + row * stripe_bytes,
                    get_noc_addr(xy & 0xFFFF, xy >> 16, conv_addr + row * conv_row_bytes + stripe_offset), stripe_bytes);
            }
            noc_async_write_barrier();
        }
    }
#elif defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t shard_rows = get_named_compile_time_arg_val("shard_rows");
    for (uint32_t n = 0; n < jobs; ++n) {
        const uint32_t job = worker + n * kWorkers;
        const uint32_t layer = job / kHeads;
        const uint32_t head = job % kHeads;
        const uint32_t state_addr = get_common_arg_val<uint32_t>(1 + layer * kLayerArgs + 4);
        for (uint32_t block = 0; block < 4; ++block) {
            cb_wait_front(16, 4);
            const uint32_t output = get_read_ptr(16);
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t global_row = head * 128 + block * 32 + row;
                const uint32_t xy = get_common_arg_val<uint32_t>(kCoresOffset + global_row / shard_rows);
                noc_async_write(
                    output + row * 256,
                    get_noc_addr(xy & 0xFFFF, xy >> 16, state_addr + (global_row % shard_rows) * 256), 256);
            }
            noc_async_writes_flushed();
            cb_pop_front(16, 4);
        }
    }
    noc_async_write_barrier();
#elif defined(COMPILE_FOR_TRISC)
    compute_kernel_hw_startup(0, 16);
    if (jobs) {
        compute_kernel_lib::untilize<4, 0, 16,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(jobs * 4);
    }
#endif
}
