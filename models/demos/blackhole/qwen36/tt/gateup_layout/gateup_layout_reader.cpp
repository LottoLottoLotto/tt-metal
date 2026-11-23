// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

namespace {
constexpr uint32_t kBanks = 8;
constexpr uint32_t kActiveWorkers = 96;
constexpr uint32_t kColumnsPerWorker = 8;
constexpr uint32_t kKTiles = 128;
constexpr uint32_t kTileBytes = 576;
constexpr uint32_t kRegionTiles = kColumnsPerWorker * kKTiles;
constexpr uint32_t kRegionBytes = kRegionTiles * kTileBytes;
constexpr uint32_t kWeightCb = 1;
constexpr uint32_t kDeliveryCb = 2;

#if defined(COMPILE_FOR_NCRISC)
uint32_t worker_id() {
    constexpr uint32_t grid_x = 11;
    return get_absolute_logical_x() + grid_x * get_absolute_logical_y();
}
#endif
}  // namespace

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t stage_k = get_named_compile_time_arg_val("stage_k");
    constexpr uint32_t delivery_mode = get_named_compile_time_arg_val("delivery_mode");
    static_assert(kKTiles % stage_k == 0);
    const uint32_t base_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t worker = worker_id();
    if (worker >= kActiveWorkers) {
        return;
    }
    const uint32_t bank = worker % kBanks;
    const uint32_t region = worker / kBanks;
    const uint32_t region_addr = base_addr + region * kRegionBytes;
    const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank, region_addr);
    const uint32_t scratch = get_write_ptr(kWeightCb);
    constexpr uint32_t column_bytes = kKTiles * kTileBytes;
    constexpr uint32_t stage_bytes = stage_k * kTileBytes;
    constexpr uint32_t buffer_bytes = kColumnsPerWorker * stage_bytes;

    DeviceZoneScopedN("QWEN_GATEUP_REPACKED_DELIVERY");
    uint32_t ping = 0;
    for (uint32_t kb = 0; kb < kKTiles; kb += stage_k) {
        const uint32_t dst = scratch + ping * buffer_bytes;
        if constexpr (delivery_mode == 2) {
            const uint32_t block = kb / stage_k;
            noc_async_read(bank_base + block * buffer_bytes, dst, buffer_bytes);
        } else {
            for (uint32_t column = 0; column < kColumnsPerWorker; ++column) {
                noc_async_read(
                    bank_base + column * column_bytes + kb * kTileBytes,
                    dst + column * stage_bytes,
                    stage_bytes);
            }
        }
        noc_async_read_barrier();
        if constexpr (delivery_mode == 1) {
            const uint32_t delivered = get_write_ptr(kDeliveryCb) + ping * buffer_bytes;
            for (uint32_t column = 0; column < kColumnsPerWorker; ++column) {
                for (uint32_t k = 0; k < stage_k; ++k) {
                    const uint32_t src = dst + (column * stage_k + k) * kTileBytes;
                    const uint32_t out = delivered + (k * kColumnsPerWorker + column) * kTileBytes;
                    noc_async_read(get_noc_addr(src), out, kTileBytes);
                }
            }
            noc_async_read_barrier();
        }
        ping ^= 1;
    }
#endif
}
