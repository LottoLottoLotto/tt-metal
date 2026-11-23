// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 attention prefill K / V front (fast/attn_prefill.py KVFront), reader (NCRISC).
//
// Output tile row i' of kv head g, kind kv (0 = k, 1 = v) is position tile s_t + i' (s_t = start / 32, o = start %
// 32: the "aligned frame"; chunk row rho is position start + rho, so output row j' of tile i' is chunk row
// 32 i' + j' - o; rows j' < o of tile 0 are positions before the chunk: the cache's / ring's existing rows).
// Per work unit (g, kv, i') this reader pushes, per column tile c of the head (4):
//   X: the linear-output tiles of chunk row tiles i' - 2, i' - 1, i' (kv_lin [Tp, 2048], column tile kv 32 + g 4 +
//      c); row tile -1 = the conv-input ring tile rotated to positions start - 32 .. start - 1 (ring row (rho + o) %
//      32 -> row rho), other rows outside the chunk and chunk rows >= valid (padding): zero, so every cache / ring
//      row written is finite (decode masks additively);
//   taps: sliding: 4 FP32 tiles (lag 0..3, every row = the channel taps incl. residual and 1 / sqrt(5));
//         full:    4 BF16 static-tap tiles (rows replicated);
//   old (i' == 0 and o > 0 only): the cache / ring tile of position tile s_t (rows < o are kept).
// and per unit (full layers): the dyn rows D (dyn [Tp, 64] column tile kv) of chunk row tiles i' - 1, i'.
// Program start: params (start / 32, o, Tpt), the shift / selector matrices for the compute (reader-built BF16 0/1
// tiles), the per-kv-head dyn selectors J_{g, lag} (full).
// Batched serving (defines): SLOTTED: the chunk's sequence slot s = pos word 2: conv-input ring pages s * 64 .., ring
// (sliding) pages s * 1152 ..; PAGED (full layers): the cache is the pool, logical block j at page-table entry j
// (runtime arg after nu: the [1, W] int32 page row).
#ifndef SLOTTED
#define SLOTTED 0
#endif
#ifndef PAGED
#define PAGED 0
#endif
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t Tpt = get_compile_time_arg_val(0);
constexpr uint32_t cb_x = get_compile_time_arg_val(1);
constexpr uint32_t cb_tap = get_compile_time_arg_val(2);
constexpr uint32_t cb_old = get_compile_time_arg_val(3);   // k-format old tile (kv 0)
constexpr uint32_t cb_oldv = get_compile_time_arg_val(4);  // v-format old tile (kv 1)
constexpr uint32_t cb_dyn = get_compile_time_arg_val(5);
constexpr uint32_t cb_m = get_compile_time_arg_val(6);     // matrices (see kM* below)
constexpr uint32_t cb_p = get_compile_time_arg_val(7);
constexpr uint32_t cb_rs = get_compile_time_arg_val(11);  // reader scratch: raw ring tile
constexpr uint32_t ok_bytes = get_compile_time_arg_val(8);
constexpr uint32_t ov_bytes = get_compile_time_arg_val(9);
constexpr bool SLIDING = get_compile_time_arg_val(10) != 0;
constexpr uint32_t tap_bytes = SLIDING ? 4096 : 2048;
constexpr uint32_t kBf16 = 2048;
constexpr uint32_t kCols = 64;    // kv_lin column tiles
constexpr uint32_t kRingTiles = 36, kRingMod = 34;
constexpr uint16_t kOne = 0x3F80;
// matrix tiles in cb_m: [0..3] S_hi(lag), [4..7] S_lo(lag), 8 R_o (rows >= o all ones), 9 E_lt (diagonal rows < o),
// 10 S_up(o), 11 S_dn(o), 12 .. 43: J_{g, lag} (g * 4 + lag; full layers)
constexpr uint32_t kMats = SLIDING ? 12 : 44;

FORCE_INLINE auto dram(uint32_t addr, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), addr, page);
}

FORCE_INLINE void zero_l1(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + off, bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE);
    }
}

FORCE_INLINE uint32_t elem(uint32_t i, uint32_t j) {
    return ((i >> 4) * 2 + (j >> 4)) * 256 + (i & 15) * 16 + (j & 15);
}

FORCE_INLINE void one(uint32_t tile, uint32_t i, uint32_t j) {
    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile)[elem(i, j)] = kOne;
}

// row rho of dst = row src_row(rho) of src (BF16 tiles; 2 x 32 B per row)
FORCE_INLINE void copy_row(uint32_t dst, uint32_t drow, uint32_t src, uint32_t srow) {
    for (uint32_t h = 0; h < 2; ++h) {
        const uint32_t d = dst + ((drow >> 4) * 2 + h) * 512 + (drow & 15) * 32;
        const uint32_t s = src + ((srow >> 4) * 2 + h) * 512 + (srow & 15) * 32;
        volatile tt_l1_ptr uint32_t* dp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d);
        const volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(s);
        for (uint32_t w = 0; w < 8; ++w) {
            dp[w] = sp[w];
        }
    }
}

// rows j0 .. 31 of a BF16 tile = 0 (the chunk rows >= valid: padding)
FORCE_INLINE void zero_rows_from(uint32_t tile, uint32_t j0) {
    for (uint32_t j = j0; j < 32; ++j) {
        for (uint32_t h = 0; h < 2; ++h) {
            volatile tt_l1_ptr uint32_t* dp =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + ((j >> 4) * 2 + h) * 512 + (j & 15) * 32);
            for (uint32_t w = 0; w < 8; ++w) {
                dp[w] = 0;
            }
        }
    }
}

void kernel_main() {
    uint32_t a = 0;
    const uint32_t xk_addr = get_arg_val<uint32_t>(a++);  // k linear output (tensor, first column tile, width)
    const uint32_t xk_c0 = get_arg_val<uint32_t>(a++);
    const uint32_t xk_w = get_arg_val<uint32_t>(a++);
    const uint32_t xv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t xv_c0 = get_arg_val<uint32_t>(a++);
    const uint32_t xv_w = get_arg_val<uint32_t>(a++);
    const uint32_t ring_addr = get_arg_val<uint32_t>(a++);   // conv-input ring [1, 1, 32, 2048]
    const uint32_t tap_addr = get_arg_val<uint32_t>(a++);    // [1, 1, 128, 2048]: lag row tiles
    const uint32_t dk_addr = get_arg_val<uint32_t>(a++);  // full: dyn tap projections kd | vd ([Tp, 32] each)
    const uint32_t dv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ok_addr = get_arg_val<uint32_t>(a++);     // full: K cache; sliding: K ring
    const uint32_t ov_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(a++);
    const uint32_t u0 = get_arg_val<uint32_t>(a++);
    const uint32_t nu = get_arg_val<uint32_t>(a++);
#if PAGED
    const uint32_t pt_addr = get_arg_val<uint32_t>(a++);
#endif
    const auto xk = dram(xk_addr, kBf16);
    const auto xv = dram(xv_addr, kBf16);
    const auto rg = dram(ring_addr, kBf16);
    const auto tp = dram(tap_addr, tap_bytes);
    const auto dk = dram(dk_addr, kBf16);
    const auto dv = dram(dv_addr, kBf16);
    const auto okc = dram(ok_addr, ok_bytes);
    const auto ovc = dram(ov_addr, ov_bytes);
    const auto ps = dram(pos_addr, 128);

    // params + matrices
    cb_reserve_back(cb_p, 1);
    const uint32_t pp = get_write_ptr(cb_p);
    noc_async_read(get_noc_addr(0, ps), pp, 128);
    cb_reserve_back(cb_m, kMats);
    const uint32_t mm = get_write_ptr(cb_m);
    zero_l1(mm, kMats * kBf16);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp);
    const uint32_t start = pw[0];
    const uint32_t valid = pw[1];  // chunk rows >= valid are padding: read as zero (finite cache / ring rows)
    const uint32_t s_t = start >> 5, o = start & 31;
    const uint32_t ring_base = SLOTTED ? pw[2] * 64 : 0;                   // the slot's conv-input ring pages
    [[maybe_unused]] const uint32_t kv_base = SLOTTED && SLIDING ? pw[2] * 8 * kRingTiles * 4 : 0;  // its K / V ring
    [[maybe_unused]] uint32_t old_block = s_t >> 1;  // full: the cache block of position tile s_t (PAGED: pool block)
#if PAGED
    if (o != 0) {  // page-table entry s_t / 2 (64 B-aligned read into the params page's scratch words)
        const auto pt = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), pt_addr, 64);
        noc_async_read(get_noc_addr(0, pt, ((s_t >> 1) * 4) & ~63u), pp + 128, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        old_block = pw[32 + ((s_t >> 1) & 15)];
    }
#endif
    pw[0] = s_t;
    pw[1] = o;
    cb_push_back(cb_p, 1);
    for (uint32_t l = 0; l < 4; ++l) {
        const uint32_t up = (o + l) & 31;  // S_up(u'): row j >= u' <- row j - u'; S_dn(u'): row j < u' <- j - u' + 32
        for (uint32_t j = 0; j < 32; ++j) {
            if (j >= up) {
                one(mm + l * kBf16, j, j - up);
            } else {
                one(mm + (4 + l) * kBf16, j, j + 32 - up);
            }
        }
    }
    for (uint32_t j = 0; j < 32; ++j) {
        if (j >= o) {
            for (uint32_t c = 0; c < 32; ++c) {
                one(mm + 8 * kBf16, j, c);
            }
            one(mm + 10 * kBf16, j, j - o);
        } else {
            one(mm + 9 * kBf16, j, j);
            one(mm + 11 * kBf16, j, j + 32 - o);
        }
    }
    if constexpr (!SLIDING) {
        for (uint32_t gl = 0; gl < 32; ++gl) {  // J_{g, lag}: row g * 4 + lag all ones (column broadcast)
            for (uint32_t j = 0; j < 32; ++j) {
                one(mm + (12 + gl) * kBf16, gl, j);
            }
        }
    }
    cb_push_back(cb_m, kMats);

    const uint32_t rs = get_write_ptr(cb_rs);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t gk = u / (Tpt + 1);  // kv * 8 + g
        const uint32_t ip = u % (Tpt + 1);
        const uint32_t kv = gk >> 3, g = gk & 7;
        if (ip == Tpt && o == 0) {
            continue;  // the extra aligned-frame tile exists only for an unaligned start (compute skips it too)
        }
        const uint32_t c0 = kv * 32 + g * 4;  // first column tile of the head in kv_lin
        if constexpr (!SLIDING) {
            cb_reserve_back(cb_dyn, 2);
            const uint32_t dd = get_write_ptr(cb_dyn);
            for (uint32_t s = 0; s < 2; ++s) {  // slot 0: row tile i' - 1, slot 1: i'
                const int32_t row = static_cast<int32_t>(ip) - 1 + static_cast<int32_t>(s);
                if (row >= 0 && row < static_cast<int32_t>(Tpt) && 32 * static_cast<uint32_t>(row) < valid) {
                    noc_async_read(get_noc_addr(row, kv == 0 ? dk : dv), dd + s * kBf16, kBf16);
                } else {
                    zero_l1(dd + s * kBf16, kBf16);
                }
            }
            noc_async_read_barrier();
            for (uint32_t s = 0; s < 2; ++s) {
                const int32_t row = static_cast<int32_t>(ip) - 1 + static_cast<int32_t>(s);
                if (row >= 0 && 32 * static_cast<uint32_t>(row) < valid && 32 * static_cast<uint32_t>(row) + 32 > valid) {
                    zero_rows_from(dd + s * kBf16, valid - 32 * static_cast<uint32_t>(row));
                }
            }
            cb_push_back(cb_dyn, 2);
        }
        for (uint32_t c = 0; c < 4; ++c) {
            cb_reserve_back(cb_x, 3);
            const uint32_t xd = get_write_ptr(cb_x);
            uint32_t ring_dst = 0;
            for (uint32_t s = 0; s < 3; ++s) {  // slots: row tiles i' - 2, i' - 1, i'
                const int32_t row = static_cast<int32_t>(ip) - 2 + static_cast<int32_t>(s);
                if (row >= 0 && row < static_cast<int32_t>(Tpt) && 32 * static_cast<uint32_t>(row) < valid) {
                    noc_async_read(kv == 0 ? get_noc_addr(row * xk_w + xk_c0 + g * 4 + c, xk)
                                           : get_noc_addr(row * xv_w + xv_c0 + g * 4 + c, xv),
                                   xd + s * kBf16, kBf16);
                } else if (row == -1) {
                    ring_dst = xd + s * kBf16;
                    noc_async_read(get_noc_addr(ring_base + c0 + c, rg), rs, kBf16);  // raw ring tile -> scratch
                } else {
                    zero_l1(xd + s * kBf16, kBf16);
                }
            }
            cb_reserve_back(cb_tap, 4);
            const uint32_t td = get_write_ptr(cb_tap);
            for (uint32_t l = 0; l < 4; ++l) {
                noc_async_read(get_noc_addr(l * kCols + c0 + c, tp), td + l * tap_bytes, tap_bytes);
            }
            noc_async_read_barrier();
            if (ring_dst != 0) {  // row tile -1: positions start - 32 + rho = ring row (rho + o) % 32
                invalidate_l1_cache();
                for (uint32_t rr = 0; rr < 32; ++rr) {
                    copy_row(ring_dst, rr, rs, (rr + o) & 31);
                }
            }
            for (uint32_t s = 0; s < 3; ++s) {  // the valid boundary tile: rows >= valid zero
                const int32_t row = static_cast<int32_t>(ip) - 2 + static_cast<int32_t>(s);
                if (row >= 0 && 32 * static_cast<uint32_t>(row) < valid && 32 * static_cast<uint32_t>(row) + 32 > valid) {
                    zero_rows_from(xd + s * kBf16, valid - 32 * static_cast<uint32_t>(row));
                }
            }
            cb_push_back(cb_x, 3);
            cb_push_back(cb_tap, 4);
            if (ip == 0 && o != 0) {
                const uint32_t ob = kv == 0 ? ok_bytes : ov_bytes;
                const uint32_t cbo = kv == 0 ? cb_old : cb_oldv;
                cb_reserve_back(cbo, 1);
                uint32_t page;
                if constexpr (SLIDING) {
                    page = kv_base + (g * kRingTiles + s_t % kRingMod) * 4 + c;
                } else {
                    page = ((old_block * 8 + g) * 2 + (s_t & 1)) * 4 + c;
                }
                noc_async_read(get_noc_addr(page, kv == 0 ? okc : ovc), get_write_ptr(cbo), ob);
                noc_async_read_barrier();
                cb_push_back(cbo, 1);
            }
        }
    }
}
