// SPDX-License-Identifier: Apache-2.0
// Residual add + RMSNorm prologue for a unified (NCRISC/BRISC/TRISC) decode kernel on a gx x gy core rectangle
// (gx * gy = np_cores, row-major core index k = (y - y0) * gx + (x - x0), core k owns width tiles [k*t, (k+1)*t)):
//
//     h = a + b                (BF16; written to the h tensor; skipped with np_residual = 0: h = a)
//     n = h / rms(h) * gamma   (gathered into np_cb_x on EVERY core of the rectangle: the full [rows, W] row block,
//                              i.e. what the host-side ResidualRMSNorm would write to n, ready as a matmul in0)
//
// Same arithmetic as ResidualRMSNorm (residual_norm_*.cpp; bitwise identical h and n): per-core partial row sums
// of h^2 (diagonal of a HiFi4 h h^T matmul) -> master's CB_M column k (inline writes + semaphore) -> master reduce
// (1/W scaler) + eps + rsqrt -> FP32 rstd column tile multicast -> (h * rstd) * gamma.  Then every core writes
// its n rows (faces 0/1 for rows <= 16) into the master's np_cb_x + semaphore; the master multicasts the block to
// all other cores (one multicast source: concurrent multicasts from all 32 cores serialize on path reservations)
// + semaphore; np_dm() returns with np_cb_x pushed (np_kt tiles) on every core.
//
// Data movement runs on ONE RISC (call np_dm<ACC, RT>() on the RISC that does not stream weights, typically
// BRISC); compute: np_compute() before the consumer's own compute (it leaves the unpacker/packer configured for
// the norm: re-init the consumer's ops, e.g. compute_kernel_hw_startup / mm_block_init, after it).
//
// Embedding mode (np_embed = 1, np_residual = 0): a is gathered from a row-major [vocab, W] BF16 table at the ids of
// a row-major [1, rows] uint32 token tensor (passed in the b slot) and also written out as h (the residual x).
//
// Named compile-time args (both RISCs): np_tiles, np_cores, np_rows, np_gx, np_x0, np_y0, np_residual, np_embed, np_kt,
// np_cb_a, np_cb_b, np_cb_g, np_cb_scaler, np_cb_h, np_cb_n, np_cb_part, np_cb_m, np_cb_rstd, np_cb_x;
// DM only: np_master_x, np_master_y (NOC coords of core k = 0), np_mc_x0, np_mc_y0, np_mc_x1, np_mc_y1
// (NOC-0 multicast rectangle of all cores), np_scaler (bf16 1/W packed twice), np_sem_gather, np_sem_rstd,
// np_sem_xg, np_sem_x (4 semaphores on the rectangle, initial 0),
// np_write_n; compute only: np_eps (fp32 bits of the BF16-truncated eps).
// DM positional compile-time args from ACC: TensorAccessorArgs of a, b, gamma, h, n (b / h / n: pass a's when
// unused).  DM common runtime args from RT: a, b, gamma, h, n addresses.
#pragma once
#include <cstdint>

namespace np {
constexpr uint32_t tiles = get_named_compile_time_arg_val("np_tiles");
constexpr uint32_t n_cores = get_named_compile_time_arg_val("np_cores");
constexpr uint32_t rows = get_named_compile_time_arg_val("np_rows");
constexpr uint32_t gx = get_named_compile_time_arg_val("np_gx");
constexpr uint32_t x0 = get_named_compile_time_arg_val("np_x0");
constexpr uint32_t y0 = get_named_compile_time_arg_val("np_y0");
constexpr bool residual = get_named_compile_time_arg_val("np_residual") == 1;
constexpr bool embed = get_named_compile_time_arg_val("np_embed") == 1;  // a = embedding-table rows at token ids
static_assert(!(embed && residual));
constexpr uint32_t kt = get_named_compile_time_arg_val("np_kt");
constexpr uint32_t cb_a = get_named_compile_time_arg_val("np_cb_a");
constexpr uint32_t cb_b = get_named_compile_time_arg_val("np_cb_b");
constexpr uint32_t cb_g = get_named_compile_time_arg_val("np_cb_g");
constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("np_cb_scaler");
constexpr uint32_t cb_h = get_named_compile_time_arg_val("np_cb_h");
constexpr uint32_t cb_n = get_named_compile_time_arg_val("np_cb_n");
constexpr uint32_t cb_part = get_named_compile_time_arg_val("np_cb_part");
constexpr uint32_t cb_m = get_named_compile_time_arg_val("np_cb_m");
constexpr uint32_t cb_rstd = get_named_compile_time_arg_val("np_cb_rstd");
constexpr uint32_t cb_x = get_named_compile_time_arg_val("np_cb_x");
constexpr uint32_t m_tiles = (n_cores + 31) / 32;
constexpr uint32_t tb = 2048;  // BF16 tile bytes
static_assert(kt == tiles * n_cores);

FORCE_INLINE uint32_t core_index() {
    return (get_absolute_logical_y() - y0) * gx + (get_absolute_logical_x() - x0);
}
}  // namespace np

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)

namespace np {
// Local DMA zero fill from the hardware zero page; caller issues the read barrier.
FORCE_INLINE void zero_fill(uint32_t addr, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes) {
        const uint32_t n = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(zeros, addr, n);
        addr += n;
        bytes -= n;
    }
}
}  // namespace np

template <uint32_t ACC, uint32_t RT>
FORCE_INLINE void np_dm() {
    using namespace np;
    constexpr uint32_t master_x = get_named_compile_time_arg_val("np_master_x");
    constexpr uint32_t master_y = get_named_compile_time_arg_val("np_master_y");
    constexpr uint32_t scaler_bits = get_named_compile_time_arg_val("np_scaler");
    constexpr uint32_t sem_gather_id = get_named_compile_time_arg_val("np_sem_gather");
    constexpr uint32_t sem_rstd_id = get_named_compile_time_arg_val("np_sem_rstd");
    constexpr uint32_t sem_xg_id = get_named_compile_time_arg_val("np_sem_xg");
    constexpr uint32_t sem_x_id = get_named_compile_time_arg_val("np_sem_x");
    constexpr bool write_n = get_named_compile_time_arg_val("np_write_n") == 1;
    uint32_t mx0 = get_named_compile_time_arg_val("np_mc_x0"), my0 = get_named_compile_time_arg_val("np_mc_y0");
    uint32_t mx1 = get_named_compile_time_arg_val("np_mc_x1"), my1 = get_named_compile_time_arg_val("np_mc_y1");
    if (noc_index == 1) {
        const uint32_t tx = mx0, ty = my0;
        mx0 = mx1, my0 = my1, mx1 = tx, my1 = ty;
    }
    constexpr auto a_args = TensorAccessorArgs<ACC>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto n_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const auto g = TensorAccessor(g_args, get_common_arg_val<uint32_t>(RT + 2), tb);
    const uint32_t k = core_index();
    const uint32_t tile0 = k * tiles;
    const bool is_master = k == 0;

    // 1. inputs
    constexpr uint32_t cb_in = residual ? cb_a : cb_h;  // norm only: a goes straight to CB_H
    cb_reserve_back(cb_in, tiles);
    cb_reserve_back(cb_g, tiles);
    const uint32_t wa = get_write_ptr(cb_in);
    const uint32_t wg = get_write_ptr(cb_g);
    for (uint32_t i = 0; i < tiles; ++i) {
        noc_async_read_tile(tile0 + i, g, wg + i * tb);
    }
    if constexpr (embed) {
        // a = rows of the row-major [vocab, W] BF16 table at the token ids of the row-major [1, rows] uint32 tok
        // tensor (b slot): this core's tiles*32 columns of each row, scattered into tile layout (padding rows 0).
        constexpr uint32_t row_bytes = n_cores * tiles * 64;  // W * 2
        constexpr uint32_t chunk = tiles * 64;                // this core's bytes of one table row
        const auto table = TensorAccessor(a_args, get_common_arg_val<uint32_t>(RT + 0), row_bytes);
        const auto tok = TensorAccessor(b_args, get_common_arg_val<uint32_t>(RT + 1), (rows * 4 + 63) / 64 * 64);
        cb_reserve_back(cb_b, 2);  // scratch: token ids at +0, table chunks at +128
        const uint32_t scratch = get_write_ptr(cb_b);
        static_assert(128 + rows * chunk <= 2 * tb, "embedding scratch");
        noc_async_read(tok.get_noc_addr(0), scratch, (rows * 4 + 63) / 64 * 64);
        zero_fill(wa, tiles * tb);
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_read(table.get_noc_addr(ids[r], k * chunk), scratch + 128 + r * chunk, chunk);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t r = 0; r < rows; ++r) {
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 128 + r * chunk);
            for (uint32_t c = 0; c < tiles * 2; ++c) {  // 16-column (32 B) pieces: tile c / 2, face column c % 2
                const uint32_t dst = wa + (c >> 1) * tb + ((r >> 4) * 2 + (c & 1)) * 512 + (r & 15) * 32;
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                for (uint32_t w = 0; w < 8; ++w) {
                    d[w] = src[c * 8 + w];
                }
            }
        }
    } else {
        const auto a = TensorAccessor(a_args, get_common_arg_val<uint32_t>(RT + 0), tb);
        for (uint32_t i = 0; i < tiles; ++i) {
            noc_async_read_tile(tile0 + i, a, wa + i * tb);
        }
    }
    uint32_t wb = 0;
    if constexpr (residual) {
        const auto b = TensorAccessor(b_args, get_common_arg_val<uint32_t>(RT + 1), tb);
        cb_reserve_back(cb_b, tiles);
        wb = get_write_ptr(cb_b);
        for (uint32_t i = 0; i < tiles; ++i) {
            noc_async_read_tile(tile0 + i, b, wb + i * tb);
        }
    }
    const uint32_t m_base = get_write_ptr(cb_m);  // identical L1 address on every core of the rectangle
    uint32_t ws = 0;
    if (is_master) {
        cb_reserve_back(cb_scaler, 1);
        ws = get_write_ptr(cb_scaler);
        zero_fill(ws, 2048);
        cb_reserve_back(cb_m, m_tiles);
        for (uint32_t j = 0; j < m_tiles; ++j) {
            for (uint32_t f = 0; f < 4; ++f) {
                const uint32_t face = m_base + j * 4096 + f * 1024;
                const uint32_t r0 = (f >> 1) * 16, c0 = j * 32 + (f & 1) * 16;
                const uint32_t first = c0 >= n_cores ? 0 : (rows > r0 ? (rows - r0 < 16 ? rows - r0 : 16) : 0);
                if (first < 16) {
                    zero_fill(face + first * 64, (16 - first) * 64);
                }
            }
        }
    }
    noc_async_read_barrier();
    if constexpr (residual) {
        cb_push_back(cb_b, tiles);
    }
    cb_push_back(cb_in, tiles);
    cb_push_back(cb_g, tiles);
    if (is_master) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws);
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t c = 0; c < 8; ++c) {
                p[f * 128 + c] = scaler_bits;
            }
        }
        cb_push_back(cb_scaler, 1);
    }

    // 2. h out (residual: as soon as compute packed it; embed: the gathered rows = the layer's residual input x).
    if constexpr (residual || embed) {
        const auto h = TensorAccessor(h_args, get_common_arg_val<uint32_t>(RT + 3), tb);
        cb_wait_front(cb_h, tiles);
        const uint32_t rh = get_read_ptr(cb_h);
        for (uint32_t i = 0; i < tiles; ++i) {
            noc_async_write_tile(tile0 + i, h, rh + i * tb);
        }
    }

    // 3. partial row sums of h^2 (diagonal of CB_PART) -> master's CB_M column k.
    cb_wait_front(cb_part, 1);
    invalidate_l1_cache();
    {
        volatile tt_l1_ptr uint32_t* part = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_part));
        const uint32_t mt = k >> 5, kc = k & 31;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t src = (r >> 4) * 3 * 256 + (r & 15) * 17;
            const uint32_t dst = mt * 1024 + ((r >> 4) * 2 + (kc >> 4)) * 256 + (r & 15) * 16 + (kc & 15);
            noc_inline_dw_write<InlineWriteDst::L1>(get_noc_addr(master_x, master_y, m_base + dst * 4), part[src]);
        }
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(master_x, master_y, get_semaphore(sem_gather_id)), 1);
    cb_pop_front(cb_part, 1);

    // 4. 1/rms: master gathers + multicasts the FP32 column tile; the others receive it.
    if (is_master) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather_id));
        noc_semaphore_wait(sem, n_cores);
        cb_push_back(cb_m, m_tiles);
        if constexpr (n_cores > 1) {
            cb_wait_front(cb_rstd, 1);
            const uint32_t rr = get_read_ptr(cb_rstd);
            noc_async_write_multicast(rr, get_noc_multicast_addr(mx0, my0, mx1, my1, rr), 4096, n_cores - 1, true);
            const uint32_t s = get_semaphore(sem_rstd_id);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s) = 1;
            noc_semaphore_set_multicast(s, get_noc_multicast_addr(mx0, my0, mx1, my1, s), n_cores - 1);
        }
    } else {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rstd_id));
        noc_semaphore_wait(sem, 1);
        cb_reserve_back(cb_rstd, 1);
        cb_push_back(cb_rstd, 1);
    }

    // 5. n rows -> the master's np_cb_x (unicast) + gather semaphore; the master multicasts the assembled block
    //    to all other cores + the x semaphore.  (One multicast source: concurrent multicasts from all cores into
    //    the same rectangle serialize on path reservations, ~2.7 us each.)
    cb_wait_front(cb_n, tiles);
    const uint32_t rn = get_read_ptr(cb_n);
    const uint32_t xb = get_write_ptr(cb_x);  // identical on every core
    for (uint32_t i = 0; i < tiles; ++i) {
        const uint32_t src = rn + i * tb, dst = xb + (tile0 + i) * tb;
        if constexpr (rows <= 16) {
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write(src + f * 512, get_noc_addr(master_x, master_y, dst + f * 512), rows * 32);
            }
        } else {
            noc_async_write(src, get_noc_addr(master_x, master_y, dst), tb);
        }
    }
    if constexpr (write_n) {
        const auto n = TensorAccessor(n_args, get_common_arg_val<uint32_t>(RT + 4), tb);
        for (uint32_t i = 0; i < tiles; ++i) {
            noc_async_write_tile(tile0 + i, n, rn + i * tb);
        }
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(master_x, master_y, get_semaphore(sem_xg_id)), 1);
    cb_reserve_back(cb_x, kt);
    if (is_master) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_xg_id)), n_cores);
        if constexpr (n_cores > 1) {
            // rows <= 16: faces 0/1 (the first 1 KB of each tile) carry every valid row
            constexpr uint32_t span = rows <= 16 ? 1024 : tb;
            if constexpr (span == tb) {
                noc_async_write_multicast(xb, get_noc_multicast_addr(mx0, my0, mx1, my1, xb), kt * tb, n_cores - 1,
                                          true);
            } else {
                for (uint32_t t = 0; t < kt; ++t) {
                    noc_async_write_multicast(xb + t * tb, get_noc_multicast_addr(mx0, my0, mx1, my1, xb + t * tb),
                                              span, n_cores - 1, true);
                }
            }
            const uint32_t s = get_semaphore(sem_x_id);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s) = 1;
            noc_semaphore_set_multicast(s, get_noc_multicast_addr(mx0, my0, mx1, my1, s), n_cores - 1);
        }
    } else {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x_id)), 1);
    }
    cb_push_back(cb_x, kt);
    noc_async_write_barrier();
    cb_pop_front(cb_n, tiles);
}

#else  // compute

template <bool unused = true>
FORCE_INLINE void np_compute() {
    using namespace np;
    constexpr uint32_t eps_bits = get_named_compile_time_arg_val("np_eps");
    const bool is_master = core_index() == 0;
    if constexpr (residual) {
        binary_op_init_common(cb_a, cb_b, cb_h);
        add_tiles_init(cb_a, cb_b);
        cb_wait_front(cb_a, tiles);
        cb_wait_front(cb_b, tiles);
        cb_reserve_back(cb_h, tiles);
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles; ++i) {
            add_tiles(cb_a, cb_b, i, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < tiles; ++i) {
            pack_tile(i, cb_h);
        }
        tile_regs_release();
        cb_push_back(cb_h, tiles);
    } else {
        binary_op_init_common(cb_h, cb_h, cb_part);
    }

    pack_reconfig_data_format(cb_part);
    matmul_init(cb_h, cb_h, 1);
    cb_wait_front(cb_h, tiles);
    tile_regs_acquire();
    for (uint32_t i = 0; i < tiles; ++i) {
        matmul_tiles(cb_h, cb_h, i, i, 0);
    }
    tile_regs_commit();
    cb_reserve_back(cb_part, 1);
    tile_regs_wait();
    pack_tile(0, cb_part);
    tile_regs_release();
    cb_push_back(cb_part, 1);

    if (is_master) {
        reconfig_data_format(cb_scaler, cb_m);
        pack_reconfig_data_format(cb_rstd);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_m, cb_scaler, cb_rstd);
        cb_wait_front(cb_scaler, 1);
        cb_wait_front(cb_m, m_tiles);
        tile_regs_acquire();
        for (uint32_t j = 0; j < m_tiles; ++j) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_m, cb_scaler, j, 0, 0);
        }
        reduce_uninit();
        constexpr VectorMode vm = rows <= 16 ? VectorMode::RC_custom : VectorMode::C;
        binop_with_scalar_tile_init();
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar, (APPROX, 0 /*ADD*/, 8), 0,
                             vm, eps_bits));
        rsqrt_tile_init<false>();
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt,
                             (APPROX, 8, DST_ACCUM_MODE, false, false), 0, vm));
        tile_regs_commit();
        cb_reserve_back(cb_rstd, 1);
        tile_regs_wait();
        pack_tile(0, cb_rstd);
        tile_regs_release();
        cb_push_back(cb_rstd, 1);
    }

    cb_wait_front(cb_rstd, 1);
    cb_wait_front(cb_g, tiles);
    cb_reserve_back(cb_n, tiles);
    reconfig_data_format(cb_h, cb_rstd);
    pack_reconfig_data_format(cb_n);
    mul_bcast_cols_init_short(cb_h, cb_rstd);
    tile_regs_acquire();
    for (uint32_t i = 0; i < tiles; ++i) {
        mul_tiles_bcast_cols(cb_h, cb_rstd, i, 0, i);
    }
    reconfig_data_format(cb_rstd, cb_g);
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g);
    for (uint32_t i = 0; i < tiles; ++i) {
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < tiles; ++i) {
        pack_tile(i, cb_n);
    }
    tile_regs_release();
    cb_push_back(cb_n, tiles);
}

#endif
