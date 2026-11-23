// SPDX-FileCopyrightText: © 2026 Lottolabs
// SPDX-License-Identifier: Apache-2.0
// LFM3 prefill sandwich-norm residual (tt/prefill/norm.py PrefillResidualNorm), one launch per layer boundary:
//
//   p = rms_norm(o) * post   (BF16, = ttnn.rms_norm(o, post))       h = x + p   (BF16, = ttnn.add(x, p))
//   n = rms_norm(h) * next   (BF16, = ttnn.rms_norm(h, next); only with has_next)
//
// Bitwise the stock chain: every value goes through the same FPU / SFPU operations as ttnn.rms_norm's
// layernorm.cpp (RMSNORM, CK_NORM: HiFi4, FP32 DEST, FP32 intermediates) and ttnn.add:
//   o^2 = mul_tiles(o, o) packed FP32; sum = reduce_row(o^2, scaler 1.0) accumulated in DEST over the row's 64
//   column tiles IN COLUMN ORDER; mean = sum * (1/2048) (SFPU), packed FP32; rstd = rsqrt(mean + eps) (FPU add of the
//   BF16 eps column tile, non-legacy SFPU rsqrt), packed FP32; y = o * rstd[col bcast] (FPU, FP32 in DEST), then
//   y * gamma[row bcast] with y moved DEST -> srcA (the native kernel packs y as FP32 and unpacks it to srcA: same
//   bits); p packed BF16; h = add_tiles(x, p) packed BF16.
//
// Work split: the Tt x 64 tiles (row-major flat index t = row * 64 + column = the DRAM page) are cut into one
// contiguous range [a, a + T) per core (T <= 64), i.e. at most two segments: s0 = [a, end of a's row) and
// s1 = [start of the next row, a + T).  A row split over several cores keeps the stock summation order through a
// chain: the core holding a row's first columns reduces them (DEST starts at 0) and sends its raw FP32 partial DEST
// tile to the next core ("OUT"), which unpacks it straight into DEST (UnpackToDestFp32: exact) and keeps
// accumulating its own columns ("IN"); the core holding the row's last column finishes (1/N, eps, rsqrt) and sends
// the FP32 rstd tile to every earlier core of the row ("peers"; OUT segments receive it).  Per core and round, at
// most one partial in, one partial out, one rstd in.  Compute order per round: phase A = squares + reduce, s1 first
// (a row head never waits, so every chain moves from lower to higher core index), then s0; phase B = normalize s0
// (local rstd) then s1 (remote rstd).  Round 2 (n) repeats this on h.
//
// RISCs: NCRISC (NOC 0) reads o (phase-A order: s1 first), gamma rows, x, next gamma rows (8-tile blocks, each
// pushed when it landed), then runs the round-2 exchanges; BRISC (NOC 1) builds the scaler / eps tiles, runs the
// round-1 exchanges, writes h and n.  Exchange buffers / semaphores are per round, so every CB has one producer and
// one consumer RISC; remote writes target the same CB base address on every core; a flag write right behind the
// data write (same NOC, same VC, same destination: delivered in order) tells the receiver it landed.
//
// Compile-time args: [has_next, 1/N fp32 bits, eps1 fp32 bits, eps2 fp32 bits].
// Common runtime args (no per-core args: each core derives its range from its logical coordinates):
//   [o, x, gamma post, gamma next, h, n addresses, row tiles Tt, cores n, grid width, (noc x, noc y) of core 0..n-1]
// (gamma: the RMSNorm's [1, 1, 64, 32] row-major BF16 weight, 64-byte pages).  Core c = y * width + x owns tiles
// [c U / n, (c + 1) U / n) of U = 64 Tt; segment flags: 1 = IN, 2 = OUT.
// Ablation define (timing only, wrong values): RN_ABL_NOMATH drops every unpack / math / pack op (keeps all CB and
// DEST handshakes), i.e. the data-movement + synchronization floor.
#include <cstdint>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#endif

namespace rnp {
constexpr uint32_t kTile = 2048;    // BF16 tile bytes
constexpr uint32_t kTile32 = 4096;  // FP32 tile bytes
constexpr uint32_t kCols = 64;      // 2048 / 32
constexpr uint32_t kBlk = 4;        // FP32 DEST half: 4 tiles
constexpr uint32_t kReadBlk = 8;    // tiles per read barrier
constexpr uint32_t IN = 1, OUT = 2;
// CBs
constexpr uint32_t CB_O = 0, CB_X = 1, CB_G1 = 2, CB_G2 = 3, CB_SC = 4, CB_EPS = 5, CB_PIN1 = 6, CB_PIN2 = 7,
                   CB_RR1 = 8, CB_RR2 = 9;
constexpr uint32_t CB_H = 16, CB_N = 17, CB_P = 18, CB_POUT1 = 19, CB_POUT2 = 20, CB_RS1 = 21, CB_RS2 = 22,
                   CB_SQ = 24, CB_EX = 25, CB_RL = 26;
// semaphores
constexpr uint32_t SEM_PIN1 = 0, SEM_PIN2 = 1, SEM_RR1 = 2, SEM_RR2 = 3;

constexpr bool has_next = get_compile_time_arg_val(0) == 1;
constexpr uint32_t inv_n_bits = get_compile_time_arg_val(1);
constexpr uint32_t eps1_bits = get_compile_time_arg_val(2);
constexpr uint32_t eps2_bits = get_compile_time_arg_val(3);

struct Layout {
    uint32_t a, T, nseg, len[2], fl[2];
    // phase-A position (o CB) and natural position (x / gamma / h / n CBs, output pages a + pos) per segment
    uint32_t offA[2], offB[2];
    bool rot;                // phase-A order = s1, s0
    uint32_t core, peer0;    // this core's index; the row tail's peers are cores [peer0, core)
};

constexpr uint32_t kArgTable = 9;  // common arg index of core 0's noc x

FORCE_INLINE Layout layout() {
    Layout L;
    const uint32_t units = get_common_arg_val<uint32_t>(6) * kCols, n = get_common_arg_val<uint32_t>(7);
    const uint32_t c = get_absolute_logical_y() * get_common_arg_val<uint32_t>(8) + get_absolute_logical_x();
    const uint32_t a = c * units / n, b = (c + 1) * units / n;
    const uint32_t row_end = (a / kCols + 1) * kCols, cut = b < row_end ? b : row_end;
    L.a = a;
    L.T = b - a;
    L.nseg = cut < b ? 2 : 1;
    L.len[0] = cut - a;
    L.len[1] = b - cut;
    L.fl[0] = (a % kCols ? IN : 0) | (cut % kCols ? OUT : 0);
    L.fl[1] = L.nseg == 2 && b % kCols ? OUT : 0;
    L.core = c;
    // owner of the row's first tile t0: the largest k with k U / n <= t0
    const uint32_t t0 = a / kCols * kCols;
    L.peer0 = L.fl[0] == IN ? ((t0 + 1) * n + units - 1) / units - 1 : c;
    L.rot = L.nseg == 2;  // s1 is then a row head (OUT): reduce it first
    L.offB[0] = 0;
    L.offB[1] = L.len[0];
    L.offA[0] = L.rot ? L.len[1] : 0;
    L.offA[1] = 0;
    return L;
}

FORCE_INLINE uint32_t order(const Layout& L, uint32_t k) { return L.rot ? 1 - k : k; }  // phase-A segment k
}  // namespace rnp

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
namespace rnp {
FORCE_INLINE auto dram(uint32_t addr, uint32_t page) {
    return TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), addr, page);
}

// Tell a peer that the tile just written to it landed: the flag write follows the data write on the same NOC and VC
// to the same destination, so it arrives after it (no ack round trip).
FORCE_INLINE void signal(uint32_t px, uint32_t py, uint32_t sem) {
    noc_inline_dw_write<InlineWriteDst::L1>(get_noc_addr(px, py, get_semaphore(sem)), 1);
}

// Round-R exchanges of this core (phase-A order): push the incoming partial (IN), send the outgoing partial (OUT),
// send the rstd of a finished row to the row's earlier cores (IN without OUT).
template <uint32_t R>
FORCE_INLINE void exchanges(const Layout& L) {
    constexpr uint32_t cb_pin = R == 1 ? CB_PIN1 : CB_PIN2, cb_pout = R == 1 ? CB_POUT1 : CB_POUT2;
    constexpr uint32_t cb_rs = R == 1 ? CB_RS1 : CB_RS2, cb_rr = R == 1 ? CB_RR1 : CB_RR2;
    constexpr uint32_t sem_pin = R == 1 ? SEM_PIN1 : SEM_PIN2, sem_rr = R == 1 ? SEM_RR1 : SEM_RR2;
    auto noc_x = [](uint32_t c) { return get_common_arg_val<uint32_t>(kArgTable + 2 * c); };
    auto noc_y = [](uint32_t c) { return get_common_arg_val<uint32_t>(kArgTable + 2 * c + 1); };
    const uint32_t npeers = L.core - L.peer0;
    // remote targets: the (one-page) CB bases, identical on every core; read before this RISC pushes them
    const uint32_t pin_base = get_write_ptr(cb_pin), rr_base = get_write_ptr(cb_rr);
    for (uint32_t k = 0; k < L.nseg; ++k) {
        const uint32_t fl = L.fl[order(L, k)];
        if (fl & IN) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_pin)), 1);
            cb_reserve_back(cb_pin, 1);
            cb_push_back(cb_pin, 1);
        }
        if (fl & OUT) {  // the row continues on the next core
            const uint32_t nx = noc_x(L.core + 1), ny = noc_y(L.core + 1);
            cb_wait_front(cb_pout, 1);
            noc_async_write(get_read_ptr(cb_pout), get_noc_addr(nx, ny, pin_base), kTile32);
            signal(nx, ny, sem_pin);
            noc_async_writes_flushed();
            cb_pop_front(cb_pout, 1);
        } else if ((fl & IN) && npeers) {
            cb_wait_front(cb_rs, 1);
            const uint32_t src = get_read_ptr(cb_rs);
            for (uint32_t p = 0; p < npeers; ++p) {
                const uint32_t px = noc_x(L.peer0 + p), py = noc_y(L.peer0 + p);
                noc_async_write(src, get_noc_addr(px, py, rr_base), kTile32);
                signal(px, py, sem_rr);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_rs, 1);
        }
    }
}

// The remote rstd of an OUT segment, once it landed.
template <uint32_t R>
FORCE_INLINE void receive_rstd() {
    constexpr uint32_t cb_rr = R == 1 ? CB_RR1 : CB_RR2, sem_rr = R == 1 ? SEM_RR1 : SEM_RR2;
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rr)), 1);
    cb_reserve_back(cb_rr, 1);
    cb_push_back(cb_rr, 1);
}
}  // namespace rnp
#endif

#if defined(COMPILE_FOR_TRISC)
namespace rnp {
#if defined(RN_ABL_NOMATH)
#define RN_OP(...) \
    do {           \
    } while (0)
#else
#define RN_OP(...) __VA_ARGS__
#endif

// y (FP32 in DEST) * gamma[row 0 broadcast]: DEST -> srcA (dest reuse), gamma row tile unpacked to srcB.
FORCE_INLINE void gamma_mul_init(uint32_t cb_g) {
    UNPACK((llk_unpack_A_init<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(false, false, cb_g)));
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, MATH_FIDELITY,
                                       EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g, cb_g, false)));
}

FORCE_INLINE void gamma_mul(uint32_t cb_g, uint32_t tile, uint32_t dst) {
    UNPACK((llk_unpack_A<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g, tile)));
    MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, DST_ACCUM_MODE, MATH_FIDELITY,
                                  EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g, cb_g, dst, true)));
}

// in^2 of tiles [off, off + len) -> CB_SQ (FP32), full 4-tile blocks.
FORCE_INLINE void squares(uint32_t cb_in, uint32_t off, uint32_t len) {
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(CB_SQ);
    mul_tiles_init(cb_in, cb_in);
    for (uint32_t j = 0; j < len; j += kBlk) {
        const uint32_t nb = len - j < kBlk ? len - j : kBlk;
        cb_wait_front(cb_in, off + j + nb);
        cb_reserve_back(CB_SQ, kBlk);
        tile_regs_acquire();
        for (uint32_t i = 0; i < nb; ++i) {
            RN_OP(mul_tiles(cb_in, cb_in, off + j + i, off + j + i, i));
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < nb; ++i) {
            RN_OP(pack_tile(i, CB_SQ));
        }
        tile_regs_release();
        cb_push_back(CB_SQ, kBlk);
    }
}

// Row sums of CB_SQ's len tiles accumulated in DEST tile 0 (after the incoming partial when IN), in column order.
// OUT: raw partial -> CB_POUT; else 1/N, + eps, rsqrt -> CB_RL (and CB_RS when the row started on earlier cores).
template <uint32_t R>
FORCE_INLINE void reduce(uint32_t len, uint32_t fl, bool peers) {
    constexpr uint32_t cb_pin = R == 1 ? CB_PIN1 : CB_PIN2, cb_pout = R == 1 ? CB_POUT1 : CB_POUT2;
    constexpr uint32_t cb_rs = R == 1 ? CB_RS1 : CB_RS2;
    reconfig_data_format(CB_SC, CB_SQ);
    tile_regs_acquire();
    if (fl & IN) {
        cb_wait_front(cb_pin, 1);
        reconfig_data_format_srca(CB_SC, cb_pin);
        copy_tile_to_dst_init_short(cb_pin);
        RN_OP(copy_tile(cb_pin, 0, 0));
        cb_pop_front(cb_pin, 1);
        reconfig_data_format_srca(cb_pin, CB_SC);
    }
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SC, CB_EX);
    for (uint32_t j = 0; j < len; j += kBlk) {
        const uint32_t nb = len - j < kBlk ? len - j : kBlk;
        cb_wait_front(CB_SQ, kBlk);
        for (uint32_t i = 0; i < nb; ++i) {
            RN_OP(reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SC, i, 0, 0));
        }
        cb_pop_front(CB_SQ, kBlk);
    }
    reduce_uninit();
    if (fl & OUT) {
        tile_regs_commit();
        cb_reserve_back(cb_pout, 1);
        pack_reconfig_data_format(cb_pout);
        tile_regs_wait();
        RN_OP(pack_tile(0, cb_pout));
        tile_regs_release();
        cb_push_back(cb_pout, 1);
        return;
    }
    binop_with_scalar_tile_init();
    RN_OP(mul_unary_tile(0, inv_n_bits));
    tile_regs_commit();
    cb_reserve_back(CB_EX, 1);
    pack_reconfig_data_format(CB_EX);
    tile_regs_wait();
    RN_OP(pack_tile(0, CB_EX));
    tile_regs_release();
    cb_push_back(CB_EX, 1);

    cb_wait_front(CB_EX, 1);
    reconfig_data_format(CB_EX, CB_EPS);
    tile_regs_acquire();
    add_tiles_init(CB_EX, CB_EPS);
    RN_OP(add_tiles(CB_EX, CB_EPS, 0, R - 1, 0));
    rsqrt_tile_init<false>();
    RN_OP(rsqrt_tile<false>(0));
    tile_regs_commit();
    cb_pop_front(CB_EX, 1);
    cb_reserve_back(CB_RL, 1);
    if (peers) {
        cb_reserve_back(cb_rs, 1);
    }
    pack_reconfig_data_format(CB_RL);
    tile_regs_wait();
    RN_OP(pack_tile(0, CB_RL));
    if (peers) {
        RN_OP(pack_tile(0, cb_rs));
    }
    tile_regs_release();
    cb_push_back(CB_RL, 1);
    if (peers) {
        cb_push_back(cb_rs, 1);
    }
}

// Round 1: p = (o * rstd) * post -> CB_P, h = x + p -> CB_H.  Round 2: n = (h * rstd) * next -> CB_N.
// CB_P: full 4-tile blocks consumed here; CB_H / CB_N: resident (never popped), exact tile counts.
template <uint32_t R>
FORCE_INLINE void normalize(uint32_t cb_in, uint32_t off_in, uint32_t off, uint32_t len, uint32_t cb_r) {
    constexpr uint32_t cb_g = R == 1 ? CB_G1 : CB_G2, cb_out = R == 1 ? CB_P : CB_N;
    cb_wait_front(cb_r, 1);
    for (uint32_t j = 0; j < len; j += kBlk) {
        const uint32_t nb = len - j < kBlk ? len - j : kBlk;
        const uint32_t step = R == 1 ? kBlk : nb;
        cb_wait_front(cb_g, off + j + nb);
        reconfig_data_format(cb_in, cb_r);
        pack_reconfig_data_format(cb_out);
        cb_reserve_back(cb_out, step);
        tile_regs_acquire();
        mul_bcast_cols_init_short(cb_in, cb_r);
        for (uint32_t i = 0; i < nb; ++i) {
            RN_OP(mul_tiles_bcast_cols(cb_in, cb_r, off_in + j + i, 0, i));
        }
        reconfig_data_format(cb_r, cb_g);
        gamma_mul_init(cb_g);
        for (uint32_t i = 0; i < nb; ++i) {
            RN_OP(gamma_mul(cb_g, off + j + i, i));
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < nb; ++i) {
            RN_OP(pack_tile(i, cb_out));
        }
        tile_regs_release();
        cb_push_back(cb_out, step);
        if constexpr (R == 1) {
            cb_wait_front(CB_X, off + j + nb);
            cb_wait_front(CB_P, kBlk);
            reconfig_data_format(CB_X, CB_P);
            pack_reconfig_data_format(CB_H);
            cb_reserve_back(CB_H, nb);
            tile_regs_acquire();
            add_tiles_init(CB_X, CB_P);
            for (uint32_t i = 0; i < nb; ++i) {
                RN_OP(add_tiles(CB_X, CB_P, off + j + i, i, i));
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < nb; ++i) {
                RN_OP(pack_tile(i, CB_H));
            }
            tile_regs_release();
            cb_push_back(CB_H, nb);
            cb_pop_front(CB_P, kBlk);
        }
    }
    cb_pop_front(cb_r, 1);
}

template <uint32_t R>
FORCE_INLINE void round(const Layout& L) {
    constexpr uint32_t cb_in = R == 1 ? CB_O : CB_H, cb_rr = R == 1 ? CB_RR1 : CB_RR2;
    const bool any_peers = L.peer0 < L.core;
    for (uint32_t k = 0; k < L.nseg; ++k) {
        const uint32_t s = order(L, k);
        squares(cb_in, R == 1 ? L.offA[s] : L.offB[s], L.len[s]);
        reduce<R>(L.len[s], L.fl[s], (L.fl[s] & IN) && !(L.fl[s] & OUT) && any_peers);
    }
    for (uint32_t s = 0; s < L.nseg; ++s) {
        normalize<R>(cb_in, R == 1 ? L.offA[s] : L.offB[s], L.offB[s], L.len[s], (L.fl[s] & OUT) ? cb_rr : CB_RL);
    }
}
}  // namespace rnp
#endif

void kernel_main() {
    using namespace rnp;
    const Layout L = layout();
#if defined(COMPILE_FOR_NCRISC)
    // o (phase-A order), gamma rows, x, next gamma rows: the CBs hold the whole range (never popped)
    const auto o = dram(get_common_arg_val<uint32_t>(0), kTile);
    const auto x = dram(get_common_arg_val<uint32_t>(1), kTile);
    auto read_tiles = [](const auto& acc, uint32_t cb, uint32_t first, uint32_t n) {
        for (uint32_t j = 0; j < n; j += kReadBlk) {
            const uint32_t nb = n - j < kReadBlk ? n - j : kReadBlk;
            cb_reserve_back(cb, nb);
            const uint32_t w = get_write_ptr(cb);
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read(acc.get_noc_addr(first + j + i), w + i * kTile, kTile);
            }
            noc_async_read_barrier();
            cb_push_back(cb, nb);
        }
    };
    // gamma row tiles of columns (a + i) % 64: the 64-byte weight page -> face 0 row 0 (+ face 0 row 1, unused), its
    // second half copied to face 1 row 0 (the stock reader's layout; the other rows are never read)
    auto read_gamma = [&](uint32_t addr, uint32_t cb) {
        const auto g = dram(addr, 64);
        cb_reserve_back(cb, L.T);
        const uint32_t w = get_write_ptr(cb);
        for (uint32_t i = 0; i < L.T; ++i) {
            noc_async_read(g.get_noc_addr((L.a + i) % kCols), w + i * kTile, 64);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < L.T; ++i) {
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + i * kTile);
            for (uint32_t c = 0; c < 8; ++c) {
                t[128 + c] = t[8 + c];
            }
        }
        cb_push_back(cb, L.T);
    };
    if (L.rot) {
        read_tiles(o, CB_O, L.a + L.len[0], L.len[1]);
        read_tiles(o, CB_O, L.a, L.len[0]);
    } else {
        read_tiles(o, CB_O, L.a, L.T);
    }
    read_gamma(get_common_arg_val<uint32_t>(2), CB_G1);
    read_tiles(x, CB_X, L.a, L.T);
    if constexpr (has_next) {
        read_gamma(get_common_arg_val<uint32_t>(3), CB_G2);
        exchanges<2>(L);
        for (uint32_t s = 0; s < L.nseg; ++s) {
            if (L.fl[s] & OUT) {
                receive_rstd<2>();
            }
        }
        noc_async_write_barrier();
    }
#elif defined(COMPILE_FOR_BRISC)
    // reduce scaler (BF16 1.0 in row 0 of every face, zeros elsewhere) and the eps column tiles (BF16-truncated eps
    // in column 0 of faces 0 / 2), as the stock reader builds them
    {
        cb_reserve_back(CB_SC, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_SC));
        for (uint32_t i = 0; i < kTile / 4; ++i) {
            p[i] = 0;
        }
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t c = 0; c < 8; ++c) {
                p[f * 128 + c] = 0x3F803F80u;
            }
        }
        cb_push_back(CB_SC, 1);
        cb_reserve_back(CB_EPS, 2);
        volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_EPS));
        for (uint32_t i = 0; i < kTile / 2; ++i) {
            e[i] = 0;
        }
        volatile tt_l1_ptr uint16_t* e16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_EPS));
        for (uint32_t r = 0; r < 2; ++r) {
            const uint16_t v = static_cast<uint16_t>((r == 0 ? eps1_bits : eps2_bits) >> 16);
            for (uint32_t f = 0; f < 4; f += 2) {
                for (uint32_t j = 0; j < 256; j += 16) {
                    e16[r * (kTile / 2) + f * 256 + j] = v;
                }
            }
        }
        cb_push_back(CB_EPS, 2);
    }
    exchanges<1>(L);
    // h (round 1) then n (round 2) as compute packs them; CB_H / CB_N are never popped: tile i at base + i * kTile
    auto write_out = [&](uint32_t cb, uint32_t addr, bool rstd_first) {
        const auto out = dram(addr, kTile);
        const uint32_t base = get_read_ptr(cb);
        for (uint32_t s = 0; s < L.nseg; ++s) {
            if (rstd_first && (L.fl[s] & OUT)) {
                receive_rstd<1>();
            }
            for (uint32_t j = 0; j < L.len[s]; j += kBlk) {
                const uint32_t nb = L.len[s] - j < kBlk ? L.len[s] - j : kBlk;
                const uint32_t i0 = L.offB[s] + j;
                cb_wait_front(cb, i0 + nb);
                for (uint32_t i = 0; i < nb; ++i) {
                    noc_async_write(base + (i0 + i) * kTile, out.get_noc_addr(L.a + i0 + i), kTile);
                }
            }
        }
    };
    write_out(CB_H, get_common_arg_val<uint32_t>(4), true);
    if constexpr (has_next) {
        write_out(CB_N, get_common_arg_val<uint32_t>(5), false);
    }
    noc_async_write_barrier();
#else
    binary_op_init_common(CB_O, CB_O, CB_SQ);
    cb_wait_front(CB_SC, 1);
    cb_wait_front(CB_EPS, 2);
    round<1>(L);
    if constexpr (has_next) {
        round<2>(L);
    }
#endif
}
