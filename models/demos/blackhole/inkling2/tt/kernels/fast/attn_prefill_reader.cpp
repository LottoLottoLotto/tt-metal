// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling2 attention prefill (fast/attn_prefill.py FlashPrefill), reader (NCRISC).
//
// Program start: the chunk position from the device tensor `pos` (int32 words: start, valid; RowsGlue.pos) -> the
// compute's params tile (word 0 = start / 32, word 1 = o = start % 32); constant tiles for the compute (column masks);
// the Q~ CB's zero background.  Then per work item (q head h, q chunk of Sq tiles = chunk rows 32 (qc Sq + r) .., absolute
// position tile p0 + r), in the compute's order:
//   1. the projection outputs of the item's rows: q [Sq x DHt] and gate [Sq x DHt] tiles of qg [Tp, 6144]
//      (q | gate), the r tile of rs [Tp, 768] (head h: [Wr x (16) | 0 | Ws x | 0 ..]), full layers: the log-scale
//      factors tau (FP32, column 0: element (i, 0) = tau(position of row i), from the model's tau table);
//   2. Q~ = the diagonal expansion of the compute's finished r_aug rows: tile (r, k) = diag(r_aug[32 rows, k]),
//      k = 0..17, built here in L1 (zero background once; only the 32 diagonal BF16 values are rewritten);
//   3. the bias table slots of distance tiles t = tmin .. tmax (slot t + 1; tmin = -1 for an unaligned start, whose
//      band slots come from the shifted scratch table of the pre-pass), the sink and the NEG slot (only the tiles
//      the compute reads);
//   4. per key chunk, Sk virtual key tiles: K (transposed order, tile (c, d) at d * Sk + c) and V (c * DHt + d).
// Virtual key tile vt = 0 is the sink (zero K / V), vt in [1, nvt) key position tile kt = w0 + vt - 1 (w0 = max(0,
// p0 - 32) on sliding layers, 0 on full layers; the last one p0 + Sq when o != 0: the q tiles' rows are positions
// 32 (p0 + r) + o + i), beyond: padding (zero K / V).  Key sources:
//   full    (SLIDING 0): the paged cache [nblocks + 1, 8, 64, 128] (the chunk's rows already written; zero block z);
//   sliding (SLIDING 1): positions < start: the ring [1, 8, 1152, 128] (slot tile kt % 34; tile 34 = zero rows),
//                        positions >= start: the chunk's K / V [1, 8, Tp + 32, 128] (KVFront's aligned frame,
//                        row tile i = position tile start / 32 + i; same format as the ring).
// Batched serving (defines): SLOTTED (sliding): the ring of sequence slot s = pos word 2 ([S, 8, 1152, 128]: pages
// s * 1152 ..); PAGED (full): the cache is the pool, logical block j at entry j of the page row (runtime arg after
// gq_count: [1, W] int32), read per K chunk one chunk ahead (2 x 128 B after the params words); key tiles without a
// valid row (position tile >= (start + valid) / 32: KVFront did not write them, their blocks may be another
// sequence's) read the zero block (masked for every valid row).
#ifndef SLOTTED
#define SLOTTED 0
#endif
#ifndef PAGED
#define PAGED 0
#endif
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t NQH = get_compile_time_arg_val(0);
constexpr uint32_t NKH = get_compile_time_arg_val(1);
constexpr uint32_t DHt = get_compile_time_arg_val(2);
constexpr uint32_t Sq = get_compile_time_arg_val(3);
constexpr uint32_t Sk = get_compile_time_arg_val(4);
constexpr uint32_t qn = get_compile_time_arg_val(5);    // q chunks per head
constexpr uint32_t Tpt = get_compile_time_arg_val(6);   // chunk rows, tiles
constexpr uint32_t cb_qr = get_compile_time_arg_val(7);
constexpr uint32_t cb_rr = get_compile_time_arg_val(8);
constexpr uint32_t cb_g = get_compile_time_arg_val(9);
constexpr uint32_t cb_tau = get_compile_time_arg_val(10);
constexpr uint32_t cb_ra = get_compile_time_arg_val(11);  // finished r_aug tiles (from the compute)
constexpr uint32_t cb_k = get_compile_time_arg_val(12);
constexpr uint32_t cb_v = get_compile_time_arg_val(13);
constexpr uint32_t cb_qt = get_compile_time_arg_val(14);
constexpr uint32_t cb_tbl = get_compile_time_arg_val(15);
constexpr uint32_t cb_c = get_compile_time_arg_val(16);   // constants: M16, E17, E16
constexpr uint32_t cb_p = get_compile_time_arg_val(17);   // params tile
constexpr uint32_t k_bytes = get_compile_time_arg_val(18);
constexpr uint32_t v_bytes = get_compile_time_arg_val(19);
constexpr bool SLIDING = get_compile_time_arg_val(20) != 0;
constexpr uint32_t kBf16 = 2048, kF32 = 4096;
constexpr uint32_t kHeadsPerKv = NQH / NKH;
constexpr uint32_t kRsTiles = NQH;           // rs row tiles: one per head
constexpr uint32_t kSlot = 18;               // tiles per table slot (k = 0..17)
constexpr uint32_t kBand = 32;               // last band table (distance tile)
constexpr uint32_t kSink = 35, kNeg = 36;    // table slots (distance tile t at slot t + 1)
constexpr uint32_t kRingTiles = 36;  // ring tensor rows / 32 per kv head
constexpr uint32_t kRingMod = 34;    // ring modulus 1088 / 32
constexpr uint32_t kBarrier = 16;
constexpr uint16_t kOne = 0x3F80;    // BF16 1.0

FORCE_INLINE auto dram(uint32_t addr, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), addr, page);
}

FORCE_INLINE void zero_l1(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + off, bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE);
    }
}

// element (i, j) of a 32x32 tile (4 faces of 16x16, row-major faces and rows), in elements
FORCE_INLINE uint32_t elem(uint32_t i, uint32_t j) {
    return ((i >> 4) * 2 + (j >> 4)) * 256 + (i & 15) * 16 + (j & 15);
}

// table tiles the compute reads from slot s: [k0, k1) (band slots s = t + 1: the NEG-mask tile k = 16 only where a
// distance can be < 0 or >= 1024: t <= 0, t >= 31)
FORCE_INLINE void slot_range(uint32_t s, uint32_t& k0, uint32_t& k1) {
    if (s == kSink) {
        k0 = 16, k1 = 18;
    } else if (s == kNeg) {
        k0 = 0, k1 = 1;
    } else if (s <= 1 || s >= kBand) {
        k0 = 0, k1 = 17;
    } else {
        k0 = 0, k1 = 16;
    }
}

void kernel_main() {
    uint32_t a = 0;
    const uint32_t qg_addr = get_arg_val<uint32_t>(a++);
    const uint32_t rs_addr = get_arg_val<uint32_t>(a++);
    const uint32_t k_addr = get_arg_val<uint32_t>(a++);
    const uint32_t v_addr = get_arg_val<uint32_t>(a++);
    const uint32_t kn_addr = get_arg_val<uint32_t>(a++);
    const uint32_t vn_addr = get_arg_val<uint32_t>(a++);
    const uint32_t t_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ts_addr = get_arg_val<uint32_t>(a++);   // the pre-pass's shifted table (o != 0)
    const uint32_t pos_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tau_addr = get_arg_val<uint32_t>(a++);
    const uint32_t zero_block = get_arg_val<uint32_t>(a++);  // full: the cache's zero block
    const uint32_t qg_stride = get_arg_val<uint32_t>(a++);   // qg tensor width, tiles
    const uint32_t qg_col0 = get_arg_val<uint32_t>(a++);     // first q tile column (gate: + NQH DHt)
    const uint32_t gq_start = get_arg_val<uint32_t>(a++);
    const uint32_t gq_count = get_arg_val<uint32_t>(a++);
#if PAGED
    static_assert(Sk <= 32);
    const auto ptab = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_arg_val<uint32_t>(a++), 64);
#endif

    const auto qg = dram(qg_addr, kBf16);
    const auto rs = dram(rs_addr, kBf16);
    const auto kc = dram(k_addr, k_bytes);
    const auto vc = dram(v_addr, v_bytes);
    const auto kn = dram(kn_addr, k_bytes);
    const auto vn = dram(vn_addr, v_bytes);
    const auto tb = dram(t_addr, kBf16);
    const auto tbs = dram(ts_addr, kBf16);
    const auto ps = dram(pos_addr, 128);
    const auto tt = dram(tau_addr, 128);  // tau table: page p = positions 32 p .. 32 p + 31 (FP32)

    // params: chunk start (device tensor) -> compute
    cb_reserve_back(cb_p, 1);
    const uint32_t pp = get_write_ptr(cb_p);
    noc_async_read(get_noc_addr(0, ps), pp, 128);
    // constants: M16 (columns 0..15), E17 (column 17), E16 (column 16)
    cb_reserve_back(cb_c, 3);
    const uint32_t cc = get_write_ptr(cb_c);
    zero_l1(cc, 3 * kBf16);
    zero_l1(get_write_ptr(cb_qt), Sq * kSlot * kBf16);  // Q~ background (single-buffered: tiles reused)
    if constexpr (!SLIDING) {
        zero_l1(get_write_ptr(cb_tau), 2 * Sq * kF32);  // tau tiles: diagonal only (double-buffered)
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp)[0];
    [[maybe_unused]] const uint32_t end = start + reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp)[1];
    // SLOTTED: the slot's K / V ring pages
    [[maybe_unused]] const uint32_t kv_base =
        SLOTTED ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp)[2] * NKH * kRingTiles * DHt : 0;
    const uint32_t s_t = start >> 5, o = start & 31;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp)[0] = s_t;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp)[1] = o;
    cb_push_back(cb_p, 1);
#if PAGED
    // page-table entries of K chunk ch: 32 from the 64 B-aligned entry pt_lo(ch) & ~15 (blocks of key tiles
    // ch Sk - 1 .. ch Sk + Sk - 2) into params-page buffer ch & 1 (the compute reads only words 0 / 1)
    auto pt_lo = [&](uint32_t ch) -> uint32_t { return ch * Sk == 0 ? 0u : ((ch * Sk - 1) >> 1) & ~15u; };
    auto pt_fetch = [&](uint32_t ch) {
        noc_async_read(get_noc_addr(0, ptab, pt_lo(ch) * 4), pp + 128 + (ch & 1) * 128, 128);
    };
#endif
    {
        volatile tt_l1_ptr uint16_t* c = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cc);
        for (uint32_t i = 0; i < 32; ++i) {
            if (i < 16) {
                for (uint32_t j = 0; j < 32; ++j) {
                    c[elem(j, i)] = kOne;
                }
            }
            c[1024 + elem(i, 17)] = kOne;
            c[2 * 1024 + elem(i, 16)] = kOne;
        }
    }
    cb_push_back(cb_c, 3);

    for (uint32_t it = 0; it < gq_count; ++it) {
        const uint32_t gq = gq_start + it;
        const uint32_t h = gq / qn;
        const uint32_t qc = gq % qn;
        const uint32_t g = h / kHeadsPerKv;
        const uint32_t p0 = s_t + qc * Sq;
        const uint32_t w0 = SLIDING && p0 > kBand ? p0 - kBand : 0;
        const uint32_t nkeys = p0 + Sq - w0 + (o != 0 ? 1 : 0);
        const uint32_t nvt = 1 + nkeys;  // vt 0: the sink tile
        const uint32_t nch = (nvt + Sk - 1) / Sk;
        const uint32_t tmax = p0 + Sq - 1 - w0 < kBand ? p0 + Sq - 1 - w0 : kBand;
        const uint32_t smin = o != 0 ? 0 : 1;  // first band slot (t = -1 only for an unaligned start)
        const uint32_t R0 = qc * Sq;  // first chunk row tile of the item

        // 1. projection rows: q, r, gate (+ tau)
        cb_reserve_back(cb_qr, Sq * DHt);
        cb_reserve_back(cb_rr, Sq);
        {
            uint32_t dq = get_write_ptr(cb_qr);
            const uint32_t dr = get_write_ptr(cb_rr);
            for (uint32_t r = 0; r < Sq; ++r) {
                for (uint32_t d = 0; d < DHt; ++d) {
                    noc_async_read(get_noc_addr((R0 + r) * qg_stride + qg_col0 + h * DHt + d, qg), dq, kBf16);
                    dq += kBf16;
                }
                noc_async_read(get_noc_addr((R0 + r) * kRsTiles + h, rs), dr + r * kBf16, kBf16);
            }
        }
        uint32_t tau_buf = 0;
        if constexpr (!SLIDING) {
            cb_reserve_back(cb_tau, Sq);
            tau_buf = get_write_ptr(cb_tau);
            // positions 32 (p0 + r) .. + 63 (two table pages) land in the tile's last 64 words (scratch, face 3),
            // then tau(32 (p0 + r) + o + i) moves to column 0
            for (uint32_t r = 0; r < Sq; ++r) {
                noc_async_read(get_noc_addr(p0 + r, tt), tau_buf + r * kF32 + kF32 - 256, 128);
                noc_async_read(get_noc_addr(p0 + r + 1, tt), tau_buf + r * kF32 + kF32 - 128, 128);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_qr, Sq * DHt);
        cb_push_back(cb_rr, Sq);
        if constexpr (!SLIDING) {
            invalidate_l1_cache();
            for (uint32_t r = 0; r < Sq; ++r) {
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tau_buf + r * kF32);
                uint32_t v[32];
                for (uint32_t i = 0; i < 32; ++i) {
                    v[i] = t[1024 - 64 + o + i];
                }
                for (uint32_t i = 0; i < 64; ++i) {
                    t[1024 - 64 + i] = 0;
                }
                for (uint32_t i = 0; i < 32; ++i) {
                    t[elem(i, 0)] = v[i];
                }
            }
            cb_push_back(cb_tau, Sq);
        }
        cb_reserve_back(cb_g, Sq * DHt);
        {
            uint32_t dg = get_write_ptr(cb_g);
            for (uint32_t r = 0; r < Sq; ++r) {
                for (uint32_t d = 0; d < DHt; ++d) {
                    noc_async_read(get_noc_addr((R0 + r) * qg_stride + qg_col0 + NQH * DHt + h * DHt + d, qg), dg, kBf16);
                    dg += kBf16;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_g, Sq * DHt);

        // 2. Q~ from the compute's r_aug tiles
        cb_wait_front(cb_ra, Sq);
        invalidate_l1_cache();
        cb_reserve_back(cb_qt, Sq * kSlot);
        {
            // Q~[r][k] diagonal (i, i) = r_aug tile r element (i, k); offsets in BF16 elements
            const uint32_t ra = get_read_ptr(cb_ra);
            const uint32_t qt = get_write_ptr(cb_qt);
            for (uint32_t r = 0; r < Sq; ++r) {
                const volatile tt_l1_ptr uint16_t* src =
                    reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(ra + r * kBf16);
                volatile tt_l1_ptr uint16_t* dst =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(qt + r * kSlot * kBf16);
                for (uint32_t i = 0; i < 32; ++i) {
                    const uint32_t srow = (i >> 4) * 512 + (i & 15) * 16;  // face (i/16, 0) row i % 16
                    const uint32_t d = elem(i, i);
                    for (uint32_t k = 0; k < 16; ++k) {
                        dst[k * 1024 + d] = src[srow + k];
                    }
                    dst[16 * 1024 + d] = src[srow + 256];  // column 16: face (i/16, 1)
                    dst[17 * 1024 + d] = src[srow + 257];
                }
            }
        }
        cb_pop_front(cb_ra, Sq);
        cb_push_back(cb_qt, Sq * kSlot);

        // 3. table slots t + 1 = smin .. tmax + 1, sink, NEG
        for (uint32_t i = smin; i <= tmax + 3; ++i) {
            const uint32_t s = i <= tmax + 1 ? i : (i == tmax + 2 ? kSink : kNeg);
            const bool shifted = o != 0 && s < kSink;
            uint32_t k0, k1;
            slot_range(s, k0, k1);
            cb_reserve_back(cb_tbl, kSlot);
            const uint32_t dst = get_write_ptr(cb_tbl);
            for (uint32_t k = k0; k < k1; ++k) {
                noc_async_read(shifted ? get_noc_addr(s * kSlot + k, tbs) : get_noc_addr(s * kSlot + k, tb),
                               dst + k * kBf16, kBf16);
            }
            noc_async_read_barrier();
            cb_push_back(cb_tbl, kSlot);
        }
        // 4. K / V chunks
#if PAGED
        pt_fetch(0);
        noc_async_read_barrier();
#endif
        for (uint32_t ch = 0; ch < nch; ++ch) {
            cb_reserve_back(cb_k, Sk * DHt);
            cb_reserve_back(cb_v, Sk * DHt);
            const uint32_t kdst = get_write_ptr(cb_k);
            const uint32_t vdst = get_write_ptr(cb_v);
#if PAGED
            invalidate_l1_cache();
            const volatile tt_l1_ptr int32_t* pte =
                reinterpret_cast<volatile tt_l1_ptr int32_t*>(pp + 128 + (ch & 1) * 128);
            const uint32_t lo = pt_lo(ch);
            if (ch + 1 < nch) {
                pt_fetch(ch + 1);  // lands before this chunk's last barrier
            }
#endif
            uint32_t n = 0;
            for (uint32_t c = 0; c < Sk; ++c) {
                const uint32_t vt = ch * Sk + c;
                const uint32_t kt = w0 + vt - 1;
                // vt 0 (sink) and padding: zero K / V; PAGED: also key tiles without a valid row
                const bool real = vt >= 1 && vt < nvt && (!PAGED || kt * 32 < end);
                bool chunk = false;
                uint32_t base;
                if constexpr (SLIDING) {
                    if (!real) {
                        base = kv_base + (g * kRingTiles + kRingMod) * DHt;
                    } else if (kt < s_t) {
                        base = kv_base + (g * kRingTiles + kt % kRingMod) * DHt;
                    } else {
                        base = (g * (Tpt + 1) + kt - s_t) * DHt;  // aligned frame: Tpt + 1 row tiles
                        chunk = true;
                    }
                } else {
#if PAGED
                    const uint32_t block = real ? static_cast<uint32_t>(pte[(kt >> 1) - lo]) : zero_block;
#else
                    const uint32_t block = real ? kt >> 1 : zero_block;
#endif
                    const uint32_t sub = real ? kt & 1 : 0;
                    base = ((block * NKH + g) * 2 + sub) * DHt;
                }
                for (uint32_t d = 0; d < DHt; ++d) {
                    const uint64_t ka = chunk ? get_noc_addr(base + d, kn) : get_noc_addr(base + d, kc);
                    const uint64_t va = chunk ? get_noc_addr(base + d, vn) : get_noc_addr(base + d, vc);
                    noc_async_read(ka, kdst + (d * Sk + c) * k_bytes, k_bytes);
                    noc_async_read(va, vdst + (c * DHt + d) * v_bytes, v_bytes);
                    if (++n == kBarrier) {
                        noc_async_read_barrier();
                        n = 0;
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_k, Sk * DHt);
            cb_push_back(cb_v, Sk * DHt);
        }
    }
}
