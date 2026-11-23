// SPDX-License-Identifier: Apache-2.0
// Optional phase timestamps for the fused residual norm kernels (define RN_PROFILE): each data-movement RISC
// records up to 16 wall-clock samples in CB 30 and writes them to page (2 * core + slot) of a row-major UINT32
// [2 * cores, 16] DRAM tensor (accessor args after the kernel's own, address = common runtime arg `crta`).
#pragma once

#ifdef RN_PROFILE
#define RN_PROF_INIT(slot)                                                                                     \
    volatile tt_l1_ptr uint32_t* rn_prof = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(30) + \
                                                                                        (slot) * 64);         \
    uint32_t rn_prof_n = 0;                                                                                    \
    rn_prof[rn_prof_n++] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L)
#define RN_STAMP() rn_prof[rn_prof_n++] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L)
#define RN_PROF_FLUSH(args_offset, crta, page)                                                                 \
    do {                                                                                                       \
        constexpr auto rn_pa = TensorAccessorArgs<args_offset>();                                              \
        const auto rn_acc = TensorAccessor(rn_pa, get_common_arg_val<uint32_t>(crta), 64);                     \
        noc_async_write(reinterpret_cast<uint32_t>(rn_prof), rn_acc.get_noc_addr(page), 64);                   \
        noc_async_write_barrier();                                                                             \
    } while (0)
#else
#define RN_PROF_INIT(slot)
#define RN_STAMP()
#define RN_PROF_FLUSH(args_offset, crta, page)
#endif
