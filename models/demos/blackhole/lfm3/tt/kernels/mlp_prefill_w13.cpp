// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 prefill dense MLP, first half: h = SiLU(x @ W1) * (x @ W3) in one launch (tt/prefill/mlp.py).
//
// x [M, K] BF16 tiles (DRAM interleaved), W13 BFP4/BFP8 in the panel layout of tt/prefill/mlp.py (panel g = h column
// tile g: for every K tile its W1 tile and its W3 tile, 2 * K tiles, pages [g * PT, (g + 1) * PT)), h [M, N] BF16.
//
// Grid: core (x, y) owns the x tile rows of M group x (<= P consecutive row tiles, resident in L1: in0) and the h
// column tiles (panels) of N group y.  in0 of a column is read from DRAM by its bottom core (BRISC) and multicast to
// the column (NOC 1); the panels of a row are read by its x = 0 core (NCRISC) and multicast along the row (NOC 0,
// ring of NB panels).  Every core accumulates each (row pair, panel) output block over the whole K in FP32 DEST
// (half sync: rows r, r + 1 x (gate, up) = 4 tiles), then the pack thread computes SiLU(gate) * up in DEST on the
// SFPU (overlapping the next block's matmul) and packs h (BF16): no L1 partials, no g / u round trip.
//
// NOC discipline (an earlier version hung intermittently when h writes overlapped the in0 multicasts on NOC 1):
// no core writes h before every column's in0 multicast has completed: each in0 sender, once its multicasts are
// acknowledged, bumps the master's count (semaphore 4); the master then multicasts "go" (semaphore 5) to the grid.
// Every receiver asks before it is sent anything (free counts), so no data lands in a core that is not ready.
//
// DRAM reads are bank runs: pages [b0, b0 + n) with b0 % 8 == 0 of an interleaved tensor are read as 8 contiguous
// runs (bank b: pages b0 + b, b0 + 8 + b, ...) into L1 slots [b * n / 8, (b + 1) * n / 8).  So L1 in0 slot j of a row
// holds K tile k(j) = 8 * (j % (KT / 8)) + j / (KT / 8), and the panel layout stores W tiles in the matching K order:
// panel L1 slot 2 * j + c (c = 0 gate, 1 up) is K tile k(j).
//
// Per-core runtime args (all three RISCs): row0, rows, panel0, npanels, in0 sender (1) / receiver (0), in0 sender
// NOC x, y, in0 multicast rectangle (NOC 0 corners x0, y0, x1, y1), in0 destinations, in1 role, in1 sender x, y, in1
// rectangle, in1 destinations, master (1/0), master x, y, grid rectangle (NOC 0 corners), grid destinations (cores
// - 1), in0 senders.  Common: NCRISC [w13 address], BRISC [x address, h address].
// Semaphores: 0 in0 data (rows delivered), 1 in0 free (receivers ready), 2 in1 data (panels delivered), 3 in1 free
// (receiver ring slots reserved), 4 in0 senders done (master), 5 go (h writes may start).
#include <cstdint>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#ifdef TRISC_PACK
#include "llk_math_eltwise_binary_sfpu_params.h"
#endif
#endif

namespace mlp {
constexpr uint32_t KT = MLP_KT;          // K tiles (hidden / 32)
constexpr uint32_t NT = MLP_NT;          // h width tiles (intermediate / 32)
constexpr uint32_t P = MLP_P;            // max rows (tiles) per core
constexpr uint32_t NB = MLP_NB;          // in1 ring panels
constexpr uint32_t W_TILE = MLP_W_TILE;  // weight tile bytes (576 BFP4, 1088 BFP8)
constexpr uint32_t HC = 1;              // h column tiles per panel (the BRISC write order and panel layout assume 1)
constexpr uint32_t PT = 2 * HC * KT;     // panel tiles
constexpr uint32_t BANKS = 8;
constexpr uint32_t X_TILE = 2048;
constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_out = 16;
constexpr uint32_t kArgs = 29;
static_assert(KT % BANKS == 0 && PT % BANKS == 0 && NB >= 2);

struct Args {
    uint32_t row0, rows, panel0, npanels;
    uint32_t in0_send, in0_sx, in0_sy, in0_x0, in0_y0, in0_x1, in0_y1, in0_n;
    uint32_t in1_send, in1_sx, in1_sy, in1_x0, in1_y0, in1_x1, in1_y1, in1_n;
    uint32_t master, mx, my, gx0, gy0, gx1, gy1, g_n, senders;
    void load() {
        uint32_t* a = &row0;
        for (uint32_t i = 0; i < kArgs; ++i) {
            a[i] = get_arg_val<uint32_t>(i);
        }
    }
};
}  // namespace mlp

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace mlp {
// pages [page0, page0 + n) (page0 % 8 == 0) of an interleaved DRAM tensor -> L1 as 8 bank runs
FORCE_INLINE void read_bank_runs(uint32_t base, uint32_t page0, uint32_t n, uint32_t page_bytes, uint32_t l1) {
    const uint32_t per = n / BANKS;
    const uint32_t off = base + (page0 / BANKS) * page_bytes;
    for (uint32_t b = 0; b < BANKS; ++b) {
        noc_async_read(get_noc_addr_from_bank_id<true>(b, off), l1 + b * per * page_bytes, per * page_bytes);
    }
}
// multicast address of a rectangle given by its NOC 0 corners (NOC 1: corners swapped)
FORCE_INLINE uint64_t mc_addr(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t addr) {
    return noc_index == 1 ? get_noc_multicast_addr(x1, y1, x0, y0, addr) : get_noc_multicast_addr(x0, y0, x1, y1, addr);
}
}  // namespace mlp
#endif

#if defined(COMPILE_FOR_NCRISC)
// in1: the panels of this core's N group; the row's x = 0 core reads and multicasts them.
void kernel_main() {
    using namespace mlp;
    Args a;
    a.load();
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t total = a.npanels;
    constexpr uint32_t panel_bytes = PT * W_TILE;
    const uint32_t sem_data = get_semaphore(2), sem_free = get_semaphore(3);
    volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_data);
    volatile tt_l1_ptr uint32_t* free_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_free);
    if (total == 0) {
        return;
    }
    if (a.in1_send) {
        // DRAM reads run up to PREFETCH panels ahead of the multicast (transaction id per ring slot), within the ring
        const uint32_t base = get_write_ptr(cb_in1);
        const uint64_t mc_data = mc_addr(a.in1_x0, a.in1_y0, a.in1_x1, a.in1_y1, sem_data);
        constexpr uint32_t D = MLP_PREFETCH < NB ? MLP_PREFETCH : NB - 1;
        auto slot = [&](uint32_t t) { return base + (t % NB) * panel_bytes; };
        auto issue = [&](uint32_t t) {
            noc_async_read_set_trid(1 + t % NB);
            read_bank_runs(w_addr, (a.panel0 + t) * PT, PT, W_TILE, slot(t));
        };
        uint32_t issued = 0;
        for (uint32_t t = 0; t < total; ++t) {
            while (issued < total && issued <= t + D) {
                cb_reserve_back(cb_in1, (issued - t + 1) * PT);
                issue(issued++);
            }
            noc_async_read_barrier_with_trid(1 + t % NB);
            const uint32_t wp = slot(t);
            if (a.in1_n) {
                *data = t + 1;  // the flag multicast's source value, stored before the NOC reads it
                noc_semaphore_wait_min(free_, (t + 1) * a.in1_n);
                noc_async_write_multicast(
                    wp, mc_addr(a.in1_x0, a.in1_y0, a.in1_x1, a.in1_y1, wp), panel_bytes, a.in1_n, true);
                noc_semaphore_set_multicast(sem_data, mc_data, a.in1_n);
                noc_async_writes_flushed();
            }
            cb_push_back(cb_in1, PT);
        }
        noc_async_read_set_trid(0);
        noc_async_write_barrier();
    } else {
        const uint64_t sender_free = get_noc_addr(a.in1_sx, a.in1_sy, sem_free);
        for (uint32_t t = 0; t < total; ++t) {
            cb_reserve_back(cb_in1, PT);
            noc_semaphore_inc(sender_free, 1);
            noc_semaphore_wait_min(data, t + 1);
            cb_push_back(cb_in1, PT);
        }
        noc_async_atomic_barrier();
    }
}

#elif defined(COMPILE_FOR_BRISC)
// in0 (this core's resident rows; the column's sender reads and multicasts them), the "go" barrier, then h -> DRAM.
void kernel_main() {
    using namespace mlp;
    Args a;
    a.load();
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t h_addr = get_common_arg_val<uint32_t>(1);
    constexpr uint32_t row_bytes = KT * X_TILE;
    const uint32_t sem_data = get_semaphore(0), sem_free = get_semaphore(1);
    const uint32_t sem_done = get_semaphore(4), sem_go = get_semaphore(5);
    volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_data);
    volatile tt_l1_ptr uint32_t* free_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_free);
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go);
    const uint32_t rows = a.rows;
    const uint32_t out_base = get_read_ptr(cb_out);  // (nothing pushed yet: the CB's start)
    cb_reserve_back(cb_in0, P * KT);
    const uint32_t base = get_write_ptr(cb_in0);
    if (a.in0_send) {
        const uint64_t mc_data = mc_addr(a.in0_x0, a.in0_y0, a.in0_x1, a.in0_y1, sem_data);
        if (rows) {
            read_bank_runs(x_addr, a.row0 * KT, KT, X_TILE, base);
        }
        if (a.in0_n) {
            noc_semaphore_wait_min(free_, a.in0_n);  // every receiver is running and asked for its rows
        }
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t l1 = base + r * row_bytes;
            noc_async_read_barrier();
            if (r + 1 < rows) {
                read_bank_runs(x_addr, (a.row0 + r + 1) * KT, KT, X_TILE, l1 + row_bytes);
            }
            if (a.in0_n) {
                *data = r + 1;
                noc_async_write_multicast(
                    l1, mc_addr(a.in0_x0, a.in0_y0, a.in0_x1, a.in0_y1, l1), row_bytes, a.in0_n, true);
                noc_semaphore_set_multicast(sem_data, mc_data, a.in0_n);
                noc_async_writes_flushed();
            }
            cb_push_back(cb_in0, KT);
        }
        noc_async_write_barrier();  // this column's multicasts are acknowledged
        noc_semaphore_inc(get_noc_addr(a.mx, a.my, sem_done), 1);
        noc_async_atomic_barrier();
    } else {
        noc_semaphore_inc(get_noc_addr(a.in0_sx, a.in0_sy, sem_free), 1);
        for (uint32_t r = 0; r < rows; ++r) {
            noc_semaphore_wait_min(data, r + 1);
            cb_push_back(cb_in0, KT);
        }
    }
    if (rows < P) {
        cb_push_back(cb_in0, (P - rows) * KT);  // (unused slots)
    }
    // go barrier: no h write while any in0 multicast is in flight
    if (a.master) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done), a.senders);
        *go = 1;
        if (a.g_n) {
            noc_semaphore_set_multicast(sem_go, mc_addr(a.gx0, a.gy0, a.gx1, a.gy1, sem_go), a.g_n);
            noc_async_write_barrier();
        }
    } else {
        noc_semaphore_wait_min(go, 1);
    }
    // h tiles -> DRAM; up to WB writes in flight before their CB slots are released (one flush per WB tiles)
    constexpr uint32_t WB = MLP_OUT_TILES / 2;
    const uint32_t out_end = out_base + MLP_OUT_TILES * X_TILE;
    uint32_t pending = 0;
    for (uint32_t j = 0; j < a.npanels; ++j) {
        const uint32_t col = a.panel0 + j;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t page = (a.row0 + r) * NT + col;
            cb_wait_front(cb_out, pending + 1);
            uint32_t src = get_read_ptr(cb_out) + pending * X_TILE;
            if (src >= out_end) {
                src -= MLP_OUT_TILES * X_TILE;
            }
#ifndef MLP_SKIP_WRITE
            noc_async_write(src, get_noc_addr_from_bank_id<true>(page % BANKS, h_addr + (page / BANKS) * X_TILE),
                            X_TILE);
#endif
            if (++pending == WB) {
                noc_async_writes_flushed();
                cb_pop_front(cb_out, WB);
                pending = 0;
            }
        }
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_out, pending);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}

#elif defined(COMPILE_FOR_TRISC)
#ifdef TRISC_PACK
namespace ckernel::sfpu {
// DEST tile g <- SiLU(g) * u = g * sigmoid(g) * u, FP32 (the stock SiLU's sigmoid: exp + 2 Newton reciprocal steps)
inline void mlp_swiglu(const uint dst_gate, const uint dst_up, const uint dst_out) {
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        const sfpi::vFloat g = sfpi::dst_reg[dst_gate * 32];
        const sfpi::vFloat u = sfpi::dst_reg[dst_up * 32];
        sfpi::dst_reg[dst_out * 32] = g * _sfpu_sigmoid_<MLP_SIG_FP32>(g) * u;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    using namespace mlp;
    Args a;
    a.load();
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    silu_tile_init_pack();
    // HC = 1: one rt x 2 matmul_block per K step over rows r0, r0 + 1 (rt <= 2), DEST (r, gate | up) = 2 r + {0, 1}.
    // HC = 2: one 1 x 4 block per K step over one row, DEST gate 0, 1 | up 2, 3 (fewer in0 unpacks per tile).
    // Either way an in0 tile is reused across the block and the weight tiles are streamed (SrcA).
    constexpr uint32_t RB = HC == 1 ? 2 : 1;  // rows per DEST block (4 FP32 tiles: half of DEST)
    constexpr uint32_t ct = 2 * HC;
    const uint32_t rows = a.rows;
    uint32_t cur_rt = 0;
    for (uint32_t jp = 0; jp < a.npanels; ++jp) {
        cb_wait_front(cb_in1, PT);
        for (uint32_t r0 = 0; r0 < rows; r0 += RB) {
            const uint32_t rt = rows - r0 >= RB ? RB : rows - r0;
            if (jp == 0) {
                cb_wait_front(cb_in0, (r0 + rt) * KT);
            }
            if (rt != cur_rt) {
                matmul_block_init(cb_in0, cb_in1, 0, ct, rt, KT);
                cur_rt = rt;
            }
            tile_regs_acquire();
#ifndef MLP_SKIP_MATH
            for (uint32_t j = 0; j < KT; ++j) {
                matmul_block(cb_in0, cb_in1, r0 * KT + j, ct * j, 0, 0, ct, rt, KT);
            }
#endif
            tile_regs_commit();
            // pack thread: wait for the math thread's DEST half, SwiGLU on the SFPU there, pack the h tiles
            PACK(TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO));
#ifndef MLP_SKIP_SFPU
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
            for (uint32_t r = 0; r < rt; ++r) {
                for (uint32_t c = 0; c < HC; ++c) {
                    PACK((_llk_math_eltwise_binary_sfpu_params_(
                        ckernel::sfpu::mlp_swiglu, r * ct + c, r * ct + HC + c, r * ct + c, VectorMode::RC)));
                }
            }
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#endif
            for (uint32_t r = 0; r < rt; ++r) {
                for (uint32_t c = 0; c < HC; ++c) {
                    cb_reserve_back(cb_out, 1);
                    pack_tile(r * ct + c, cb_out, 0);
                    cb_push_back(cb_out, 1);
                }
            }
            tile_regs_release();
        }
        cb_pop_front(cb_in1, PT);
    }
    cb_wait_front(cb_in0, P * KT);
    cb_pop_front(cb_in0, P * KT);
}
#endif
