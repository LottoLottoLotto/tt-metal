// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM2 prefill experts with row-group reuse (derived from models/common/modules/moe grouped_experts_kernel.cpp).
//
// Same inputs / outputs / per-tile math as GroupedExperts (bitwise): packed BF16 x tiles [groups*32, K], the
// grouped-routing metadata row (count, expert id per 32-row group; groups of one expert are consecutive), resident
// bank-shuffled BFP4 weights, FP32 DEST accumulation in the same k order, fused weighted SwiGLU for gate/up.
//
// Work split: a run of consecutive groups of one expert is cut into items of <= max_rows groups (balanced sizes);
// a unit is (item, DRAM bank b).  Units are ordered item-major and every worker takes a contiguous range of them
// with balanced cost (groups per unit), so a worker loads an item's x once for all of its banks and streams each
// (expert, bank) weight column block once per item (GroupedExperts re-streams it for every 32-row group).
// Each worker keeps the item's x (<= max_rows * K tiles) and one column (pair) of weights resident; the weight
// stream of the next column overlaps the math of the current one.
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#endif
#include "models/common/modules/moe/kernels/streaming/experts_matmul.hpp"

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_PACK)
namespace ckernel::sfpu {
inline void calculate_grouped_swiglu(const uint dst_gate, const uint dst_up, const uint dst_out) {
#pragma GCC unroll 8
    for (int slot = 0; slot < 8; ++slot) {
        const sfpi::vFloat gate = sfpi::dst_reg[dst_gate * 32];
        const sfpi::vFloat up = sfpi::dst_reg[dst_up * 32];
        const sfpi::vFloat weight = sfpi::dst_reg[(dst_gate + 2) * 32];
        sfpi::dst_reg[dst_out * 32] = gate * _sfpu_sigmoid_<true>(gate) * up * weight;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_x = 0, cb_w = 1, cb_meta = 2, cb_out = 4, cb_route = 5;

namespace {
// Walks every unit (item, bank) in order and reports this worker's contiguous share.  Unit cost (balancing) =
// 8 * groups + unit_cost: the per-unit weight stream costs about unit_cost / 8 groups of x / output traffic.
struct Units {
    volatile uint32_t* meta;  // [count, expert of group 0, expert of group 1, ...]
    uint32_t count, banks, max_rows, worker, workers, unit_cost;
    // state
    uint32_t run_start = 0, run_len = 0, items = 0, item = 0, item_g0 = 0, item_rows = 0, bank = 0;
    uint32_t cost_before = 0, total = 0, item_id = 0;
    bool started = false;

    void init(volatile uint32_t* m, uint32_t b, uint32_t rmax, uint32_t w, uint32_t nw, uint32_t uc) {
        meta = m;
        count = m[0];
        banks = b;
        max_rows = rmax;
        worker = w;
        workers = nw;
        unit_cost = uc;
        uint32_t n_items = 0;
        for (uint32_t g = 0; g < count;) {
            const uint32_t e = meta[1 + g];
            uint32_t n = 1;
            while (g + n < count && meta[1 + g + n] == e) ++n;
            n_items += (n + max_rows - 1) / max_rows;
            g += n;
        }
        total = (count * 8 + n_items * unit_cost) * banks;
    }
    void open_run() {
        const uint32_t e = meta[1 + run_start];
        run_len = 1;
        while (run_start + run_len < count && meta[1 + run_start + run_len] == e) {
            ++run_len;
        }
        items = (run_len + max_rows - 1) / max_rows;
        item = 0;
        item_g0 = run_start;
    }
    void open_item() {
        const uint32_t base = run_len / items, extra = run_len % items;
        item_rows = base + (item < extra ? 1 : 0);
    }
    // Advance to the next unit overall; false at the end.
    bool next_any() {
        if (!started) {
            started = true;
            if (count == 0) return false;
            open_run();
            open_item();
            bank = 0;
            return true;
        }
        cost_before += item_rows * 8 + unit_cost;
        if (++bank < banks) return true;
        bank = 0;
        ++item_id;
        item_g0 += item_rows;
        if (++item < items) {
            open_item();
            return true;
        }
        run_start += run_len;
        if (run_start >= count) return false;
        open_run();
        open_item();
        return true;
    }
    bool mine() const {
        const uint64_t p = uint64_t{cost_before} * workers;
        return p >= uint64_t{worker} * total && p < uint64_t{worker + 1} * total;
    }
    bool past() const { return uint64_t{cost_before} * workers >= uint64_t{worker + 1} * total; }
    uint32_t expert() const { return meta[1 + item_g0]; }
};
}  // namespace

void kernel_main() {
    constexpr uint32_t kt = get_named_compile_time_arg_val("num_tiles_k");
    constexpr uint32_t nt = get_named_compile_time_arg_val("out_tiles");
    constexpr uint32_t per_bank_out = get_named_compile_time_arg_val("per_bank_out");
    constexpr uint32_t block_k = get_named_compile_time_arg_val("subblock_k");
    constexpr bool fused = get_named_compile_time_arg_val("fused_swiglu") != 0;
    constexpr uint32_t banks = get_named_compile_time_arg_val("banks");
    constexpr uint32_t max_rows = get_named_compile_time_arg_val("max_rows");
    constexpr uint32_t workers = get_named_compile_time_arg_val("workers");
    constexpr uint32_t unit_cost = get_named_compile_time_arg_val("unit_cost");
    constexpr uint32_t columns = per_bank_out * (fused ? 2 : 1);   // streamed columns per (expert, bank)
    constexpr uint32_t k_blocks = kt / block_k;
    constexpr uint32_t col_tiles = kt * (fused ? 2 : 1);            // tiles per output column (pair)
    // Down only: outputs as row-major FP32 rows (pack_untilize; the combine reads token rows directly), rows-outer
    // order with all of the unit's columns resident.
    constexpr bool untilize = get_named_compile_time_arg_val("untilize_out") != 0;
    constexpr uint32_t res_tiles = untilize ? columns * kt : col_tiles;  // weight tiles held at once
    static_assert(!(untilize && fused), "the row-major output is the down projection's");
    static_assert(!untilize || max_rows <= 4, "matmul_block rows must fit half of the FP32 DEST");

#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t block_bytes = get_named_compile_time_arg_val("weight_block_bytes");
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("weight_page_bytes");
    constexpr uint32_t pages = get_named_compile_time_arg_val("weight_pages");
    constexpr uint32_t expert_bytes = columns * k_blocks * block_bytes;
    constexpr uint32_t weight_addr = get_named_compile_time_arg_val("weight_addr");
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint32_t buffer_base = get_write_ptr(cb_w);
    const uint32_t buffer_end = buffer_base + 2 * res_tiles * (block_bytes / block_k);
    cb_wait_front(cb_meta, 1);
    Units u;
    u.init(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta)), banks, max_rows, worker, workers, unit_cost);
    uint32_t write_addr = buffer_base;
    uint32_t issue_trid = 1, wait_trid = 1;
    bool pending = false;
    while (u.next_any()) {
        if (u.past()) break;
        if (!u.mine()) continue;
        const uint32_t expert = u.expert();
        const uint64_t bank_base = get_noc_addr_from_bank_id<true>(u.bank, weight_addr);
        noc_async_read_one_packet_set_state<true>(bank_base, page_bytes, vc);
        for (uint32_t column = 0; column < columns; ++column) {
            const uint32_t stored_column = fused ? column / 2 + (column % 2) * per_bank_out : column;
            uint32_t read_offset = expert * expert_bytes + stored_column * k_blocks * block_bytes;
            for (uint32_t kb = 0; kb < k_blocks; ++kb) {
                cb_reserve_back(cb_w, 2 * block_k);
                noc_async_read_set_trid(issue_trid);
                for (uint32_t page = 0; page < pages; ++page) {
                    noc_async_read_one_packet_with_state_with_trid(bank_base, read_offset, write_addr, issue_trid);
                    read_offset += page_bytes;
                    write_addr += page_bytes;
                }
                if (pending) {
                    noc_async_read_barrier_with_trid(wait_trid);
                    cb_push_back(cb_w, block_k);
                    wait_trid = wait_trid == 3 ? 1 : wait_trid + 1;
                } else {
                    pending = true;
                }
                issue_trid = issue_trid == 3 ? 1 : issue_trid + 1;
                if (write_addr == buffer_end) {
                    write_addr = buffer_base;
                }
            }
        }
    }
    if (pending) {
        noc_async_read_barrier_with_trid(wait_trid);
        cb_push_back(cb_w, block_k);
    }

#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t output_bytes = untilize ? nt * 32 * 4 : (fused ? 2048 : 4096);  // tile, or FP32 row
    constexpr uint32_t meta_bytes = get_named_compile_time_arg_val("metadata_bytes");
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto experts_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<experts_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0), 2048);
    const auto experts = TensorAccessor(experts_args, get_arg_val<uint32_t>(1), meta_bytes);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(2), output_bytes);
    const uint32_t worker = get_arg_val<uint32_t>(4);
    cb_reserve_back(cb_meta, 1);
    noc_async_read(get_noc_addr(0, experts), get_write_ptr(cb_meta), meta_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_meta, 1);
    Units u;
    u.init(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta)), banks, max_rows, worker, workers, unit_cost);
    uint32_t loaded_item = 0xffffffffu;
    while (u.next_any()) {
        if (u.past()) break;
        if (!u.mine()) continue;
        const uint32_t rows = u.item_rows, g0 = u.item_g0;
        if (u.item_id != loaded_item) {
            // A new item: its x groups (and route tiles) start at the CB bases; the whole capacity is pushed so
            // that every item begins at the base again (TRISC pops the whole capacity per item).
            loaded_item = u.item_id;
            cb_reserve_back(cb_x, max_rows * kt);
            const uint32_t xbase = get_write_ptr(cb_x);
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t k = 0; k < kt; ++k) {
                    noc_async_read(get_noc_addr((g0 + r) * kt + k, x), xbase + (r * kt + k) * 2048, 2048);
                }
            }
            if constexpr (fused) {
                constexpr auto route_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
                const auto route = TensorAccessor(route_args, get_arg_val<uint32_t>(3), 2048);
                cb_reserve_back(cb_route, max_rows);
                const uint32_t rbase = get_write_ptr(cb_route);
                for (uint32_t r = 0; r < rows; ++r) {
                    noc_async_read(get_noc_addr(g0 + r, route), rbase + r * 2048, 2048);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, max_rows * kt);
            if constexpr (fused) {
                cb_push_back(cb_route, max_rows);
            }
        }
        if constexpr (untilize) {
            constexpr uint32_t segment = per_bank_out * 32 * 4;  // one row's FP32 columns of this bank
            cb_wait_front(cb_out, max_rows * per_bank_out);
            const uint32_t src = get_read_ptr(cb_out);
            for (uint32_t i = 0; i < rows * 32; ++i) {
                noc_async_write(src + i * segment, get_noc_addr(g0 * 32 + i, out) + u.bank * segment, segment);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, max_rows * per_bank_out);
        } else {
            for (uint32_t n = 0; n < per_bank_out; ++n) {
                for (uint32_t r = 0; r < rows; ++r) {
                    cb_wait_front(cb_out, 1);
                    noc_async_write(get_read_ptr(cb_out),
                                    get_noc_addr((g0 + r) * nt + u.bank * per_bank_out + n, out), output_bytes);
                    noc_async_writes_flushed();
                    cb_pop_front(cb_out, 1);
                }
            }
        }
    }
    noc_async_write_barrier();

#elif defined(COMPILE_FOR_TRISC)
    static_assert(DST_ACCUM_MODE, "grouped expert matmuls require FP32 accumulation");
    deepseek_compute_kernel_init();
    pack_reconfig_data_format<true>(cb_out);
    if constexpr (untilize) {
        pack_untilize_dest_init<1, per_bank_out>(cb_out);
    }
    if constexpr (fused) {
        PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false)));
    }
    const uint32_t worker = get_arg_val<uint32_t>(0);
    cb_wait_front(cb_meta, 1);
    Units u;
    u.init(reinterpret_cast<volatile uint32_t*>(get_tile_address(cb_meta, 0)), banks, max_rows, worker, workers, unit_cost);
    uint32_t loaded_item = 0xffffffffu;
    bool holding = false;
    while (u.next_any()) {
        if (u.past()) break;
        if (!u.mine()) continue;
        const uint32_t rows = u.item_rows;
        if (u.item_id != loaded_item) {
            if (holding) {
                cb_pop_front(cb_x, max_rows * kt);
                if constexpr (fused) {
                    cb_pop_front(cb_route, max_rows);
                }
            }
            loaded_item = u.item_id;
            holding = true;
            cb_wait_front(cb_x, max_rows * kt);
            if constexpr (fused) {
                cb_wait_front(cb_route, max_rows);
            }
        }
        if constexpr (untilize) {
            // One weight tile feeds all `rows` row groups (matmul_block, rt_dim = rows <= 4: half of FP32 DEST);
            // per output column n the rows x 1 tiles are untilized into row block [rows * 32, per_bank_out * 32].
            cb_reserve_back(cb_out, max_rows * per_bank_out);
            for (uint32_t n = 0; n < per_bank_out; ++n) {
                reconfig_data_format<SrcOrder::Reverse, true>(cb_x, cb_w);
                matmul_block_init(cb_x, cb_w, false, 1, rows, kt);
                tile_regs_acquire();
                for (uint32_t kb = 0; kb < k_blocks; ++kb) {
                    cb_wait_front(cb_w, (n * k_blocks + kb + 1) * block_k);
                    for (uint32_t k = 0; k < block_k; ++k) {
                        matmul_block(cb_x, cb_w, kb * block_k + k, (n * k_blocks + kb) * block_k + k, 0, false, 1,
                                     rows, kt);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<1, per_bank_out>(cb_out, rows, n);
                tile_regs_release();
            }
            cb_push_back(cb_out, max_rows * per_bank_out);
            cb_pop_front(cb_w, res_tiles);
            continue;
        }
        for (uint32_t n = 0; n < per_bank_out; ++n) {
            for (uint32_t r = 0; r < rows; ++r) {
                cb_reserve_back(cb_out, 1);
                reconfig_data_format<SrcOrder::Reverse, true>(cb_x, cb_w);
                matmul_init(cb_x, cb_w, false);
                tile_regs_acquire();
                for (uint32_t operand = 0; operand < (fused ? 2u : 1u); ++operand) {
                    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
                        if (r == 0) {
                            cb_wait_front(cb_w, (operand * k_blocks + kb + 1) * block_k);
                        }
                        for (uint32_t k = 0; k < block_k; ++k) {
                            matmul_tiles(cb_x, cb_w, r * kt + kb * block_k + k, (operand * k_blocks + kb) * block_k + k,
                                         operand);
                        }
                    }
                }
                if constexpr (fused) {
                    reconfig_data_format(cb_route, cb_route);
                    copy_tile_init(cb_route);
                    copy_tile(cb_route, r, 2);
                }
                tile_regs_commit();
                if constexpr (fused) {
                    PACK(TTI_SEMWAIT(
                        p_stall::STALL_TDMA | p_stall::STALL_CFG,
                        semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
                    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                    PACK((_llk_math_eltwise_binary_sfpu_params_(
                        ckernel::sfpu::calculate_grouped_swiglu, 0, 1, 0, VectorMode::RC)));
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                } else {
                    tile_regs_wait();
                }
                pack_tile(0, cb_out, 0);
                tile_regs_release();
                cb_push_back(cb_out, 1);
            }
            cb_pop_front(cb_w, col_tiles);
        }
    }
    if (holding) {
        cb_pop_front(cb_x, max_rows * kt);
        if constexpr (fused) {
            cb_pop_front(cb_route, max_rows);
        }
    }
#endif
}
