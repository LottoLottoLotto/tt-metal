// SPDX-FileCopyrightText: © 2026 Lottolabs
//
// SPDX-License-Identifier: Apache-2.0

// Inkling-2-Mini prefill experts over 32-slot expert groups (moe_router_rows.cpp table: meta page = counts [E], group
// starts [E + 1]; map page per group = 32 slots [token | k << 16, bf16 routing weight], pads 0xffffffff), weights
// streamed from the resident bank layout of moe_layout.py.
//
// Work split (every core of the grid, identical arithmetic on NCRISC / BRISC): each expert's groups are cut into
// ceil(G / RB) near-equal row blocks; a row block has NU units (STAGE 0: the 36 gate/up pairs, STAGE 1: the 24 down
// column groups of `cols` tiles).  The units of all row blocks in expert order are cut into equal contiguous ranges,
// one per core.  A core loads each of its row blocks' activations once and streams every unit of its range once,
// applying it to all the block's groups: every weight byte is read once per row block, the activations ~once.
// STAGE 0 (gate/up + softcapped SwiGLU): BRISC gathers the block's x rows (row-major [T, H], one read per token row,
//   pads skipped: rows are independent in the matmul) and TRISC tilizes them; units = pairs [K, {gate, up}] from
//   their banks; per group gate / up accumulate in FP32 DEST, the group's routing weight tile (BRISC-built, weight
//   per row) lands in DEST 3, the pack thread applies g, u *= f13[e]; act = 4 tanh(g/4) sigmoid(g) 25 tanh(u/25) *
//   w_row; BF16 act tile -> act[group * I/32 + pair].
// STAGE 1 (down): BRISC reads the block's act tiles; units = [K = I/32, cols] blocks of bank c / (bank_cols / cols);
//   FP32 output [32, cols * 32] untilized by the packer; BRISC writes each real row (token t, pick k) to the combine
//   buffer: page ((t / 32) * col_groups + c) * top_k + k, row t % 32 (row-major [32, cols * 32] FP32 per page).
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#endif
#include "models/common/modules/moe/kernels/streaming/experts_matmul.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "models/demos/blackhole/inkling2/tt/kernels/moe/moe_route.hpp"
#endif

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_PACK)
#include "sfpu/ckernel_sfpu_converter.h"
namespace ckernel::sfpu {
sfpi_inline sfpi::vFloat ink_tanh(sfpi::vFloat y) {
    sfpi::vFloat t = _sfpu_sigmoid_<true>(y + y) * 2.0f - 1.0f;
    const sfpi::vFloat y2 = y * y;
    v_if(sfpi::abs(y) < 0.0625f) { t = y * (1.0f - y2 * (0.333333343f - y2 * 0.133333340f)); }
    v_endif;
    return t;
}

// All rows: dst_gate + 2 is scratch, dst_gate + 3 holds the per-row routing weight.
inline void calculate_ink_swiglu_rows(const uint dst_gate, const uint dst_up, const uint dst_out, const uint f_bits) {
    constexpr uint tile_stride = 32;
    const uint dst_tmp = dst_gate + 2, dst_w = dst_gate + 3;
#pragma GCC unroll 1
    for (int d = 0; d < 8; d++) {
        {
            const sfpi::vFloat u = sfpi::dst_reg[dst_up * tile_stride] * Converter::as_float(f_bits);
            sfpi::dst_reg[dst_up * tile_stride] = ink_tanh(u * 0.04f) * 25.0f;
        }
        {
            const sfpi::vFloat g = sfpi::dst_reg[dst_gate * tile_stride] * Converter::as_float(f_bits);
            sfpi::dst_reg[dst_tmp * tile_stride] = ink_tanh(g * 0.25f) * 4.0f;
        }
        {
            const sfpi::vFloat g = sfpi::dst_reg[dst_gate * tile_stride] * Converter::as_float(f_bits);
            sfpi::vFloat r = _sfpu_sigmoid_<true>(g) * sfpi::dst_reg[dst_tmp * tile_stride];
            r = r * sfpi::dst_reg[dst_up * tile_stride];
            sfpi::dst_reg[dst_out * tile_stride] = r * sfpi::dst_reg[dst_w * tile_stride];
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_x = 0, cb_w = 1, cb_meta = 2, cb_wt = 3, cb_out = 4, cb_info = 5, cb_bmeta = 6, cb_rows = 7,
                   cb_map = 8;
constexpr uint32_t END = 0xffffffffu;
constexpr uint32_t META_BYTES = 2048;  // counts [E], group starts [E + 1]

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
// This core's pieces: (expert, first group, groups, first unit, end unit) of every row block meeting its unit range;
// the core's half owns units [ub0, ub1) of every row block (lo / hi are absolute unit indices).
template <uint32_t E, uint32_t RB>
struct Pieces {
    const volatile tt_l1_ptr uint32_t* start;  // group starts [E + 1]
    uint32_t nu, off, u0 = 0, u1 = 0;

    Pieces(const volatile tt_l1_ptr uint32_t* s, uint32_t core, uint32_t cores, uint32_t ub0, uint32_t ub1) :
        start(s), nu(ub1 - ub0), off(ub0) {
        uint32_t U = 0;
        for (uint32_t e = 0; e < E; ++e) {
            const uint32_t G = start[e + 1] - start[e];
            U += (G + RB - 1) / RB * nu;
        }
        u0 = U * core / cores;
        u1 = U * (core + 1) / cores;
    }

    template <typename F>
    void run(F&& f) const {
        uint32_t ub = 0;
        for (uint32_t e = 0; e < E && ub < u1; ++e) {
            const uint32_t G = start[e + 1] - start[e];
            if (G == 0) {
                continue;
            }
            const uint32_t nb = (G + RB - 1) / RB, base = G / nb, extra = G % nb;
            uint32_t g = start[e];
            for (uint32_t r = 0; r < nb && ub < u1; ++r) {
                const uint32_t n = base + (r < extra ? 1 : 0);
                if (ub + nu > u0) {
                    const uint32_t lo = (u0 > ub ? u0 : ub) - ub, hi = (u1 < ub + nu ? u1 : ub + nu) - ub;
                    f(e, g, n, lo + off, hi + off);
                }
                ub += nu;
                g += n;
            }
        }
    }
};

template <uint32_t cb, uint32_t buffers, uint32_t ahead, uint32_t block_tiles, uint32_t block_bytes>
struct TridRing {
    static_assert(ahead >= 1 && ahead + 1 < buffers && buffers <= 15);
    uint32_t begin, end, write, free = buffers, trid = 1, wait = 1, pending = 0;

    void start() {
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
        cb_reserve_back(cb, block_tiles * (ahead + 1));
        begin = write = get_write_ptr(cb);
        end = begin + buffers * block_bytes;
    }
    void push_oldest() {
        noc_async_read_barrier_with_trid(wait);
        cb_push_back(cb, block_tiles);
        wait = wait == buffers ? 1 : wait + 1;
        --pending;
    }
    void next() {
        ++pending;
        write += block_bytes;
        if (write >= end) {
            write = begin;
        }
        if (free == buffers - ahead) {
            push_oldest();
            cb_reserve_back(cb, block_tiles * (ahead + 1));
        } else {
            --free;
        }
        trid = trid == buffers ? 1 : trid + 1;
    }
    void finish() {
        while (pending) {
            push_oldest();
        }
    }
};
#endif

void kernel_main() {
    constexpr uint32_t stage = STAGE;
    constexpr bool gate_up = stage == 0;
    constexpr uint32_t kt = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t cols = get_named_compile_time_arg_val("cols");
    constexpr uint32_t rb = get_named_compile_time_arg_val("rb");  // max groups per row block
    constexpr uint32_t E = get_named_compile_time_arg_val("experts");
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr uint32_t inter_tiles = get_named_compile_time_arg_val("inter_tiles");
    constexpr uint32_t bank_cols = get_named_compile_time_arg_val("bank_cols");  // down: output tiles per bank
    constexpr uint32_t pair_base = get_named_compile_time_arg_val("pair_base");
    constexpr uint32_t pair_odd = get_named_compile_time_arg_val("pair_odd");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t unit_tiles = kt * cols;
    constexpr uint32_t bank_groups = bank_cols / cols;              // down column groups per bank
    constexpr uint32_t col_groups = hidden_tiles / cols;            // down column groups
    // NOC halves: the west cores stream units [0, split) of every row block (gate/up pairs / down column groups in
    // order: the first half of the banks first), the east cores [split, units): DRAM responses mostly stay on the
    // half's own NOC0 rows / columns; split ~ units x west cores / cores (host) balances the cores' work.
    const uint32_t core = get_arg_val<uint32_t>(0);   // index within the half
    const uint32_t cores = get_arg_val<uint32_t>(1);  // cores of the half
    const uint32_t ub0 = get_arg_val<uint32_t>(2), ub1 = get_arg_val<uint32_t>(3);  // the half's unit window
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    using Place [[maybe_unused]] = ink_moe::PairPlacement<pair_base, pair_odd>;
#endif

#if defined(COMPILE_FOR_NCRISC)
    // args: core, cores, meta, factors, w_addr
    constexpr uint32_t tile_bytes = get_named_compile_time_arg_val("w_tile_bytes");
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("w_page_bytes");
    constexpr uint32_t pages = get_named_compile_time_arg_val("w_pages");
    constexpr uint32_t buffers = get_named_compile_time_arg_val("w_buffers");
    constexpr uint32_t unit_bytes = unit_tiles * tile_bytes;
    static_assert(pages * page_bytes == unit_bytes);
    constexpr auto meta_args = TensorAccessorArgs<0>();
    constexpr auto fac_args = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();
    const auto meta = TensorAccessor(meta_args, get_common_arg_val<uint32_t>(2), META_BYTES);
    const auto factors = TensorAccessor(fac_args, get_common_arg_val<uint32_t>(3), 2 * E * 4);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t vc = core & 3;
    const uint32_t scratch = get_write_ptr(cb_meta);
    const uint32_t fac_addr = scratch + META_BYTES;
    noc_async_read(meta.get_noc_addr(0), scratch, META_BYTES);
    noc_async_read(factors.get_noc_addr(0), fac_addr, 2 * E * 4);
    noc_async_read_barrier();
    auto* fac = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fac_addr);
    const Pieces<E, rb> pieces(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch) + E, core, cores, ub0, ub1);
    TridRing<cb_w, buffers, buffers - 2, unit_tiles, unit_bytes> ring;
    DeviceZoneScopedN("X-NC");
    ring.start();
    pieces.run([&](uint32_t e, uint32_t, uint32_t n, uint32_t lo, uint32_t hi) {
        {  // per-piece record for TRISC: [groups, f13 bits, units]
            cb_reserve_back(cb_info, 1);
            auto* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_info));
            info[0] = n;
            info[1] = fac[e];
            info[2] = hi - lo;
            cb_push_back(cb_info, 1);
        }
        for (uint32_t j = lo; j < hi; ++j) {
            uint32_t bank, src;
            if constexpr (gate_up) {  // pair j of expert e: the bank whose pair run holds it
                bank = 0;
                while (j >= Place::start(e, bank) + Place::pairs(e, bank)) {
                    ++bank;
                }
                src = (Place::before(e, bank) + j - Place::start(e, bank)) * unit_bytes;
            } else {
                bank = j / bank_groups;
                src = (e * bank_groups + j % bank_groups) * unit_bytes;
            }
            const uint64_t base = get_noc_addr_from_bank_id<true>(bank, w_addr);
            noc_async_read_one_packet_set_state<true>(base, page_bytes, vc);
            uint32_t dst = ring.write;
            noc_async_read_set_trid(ring.trid);
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read_one_packet_with_state_with_trid(static_cast<uint32_t>(base), src, dst, ring.trid);
                src += page_bytes;
                dst += page_bytes;
            }
            ring.next();
        }
    });
    ring.finish();
    noc_async_read_set_trid(0);
    cb_reserve_back(cb_info, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_info))[0] = END;
    cb_push_back(cb_info, 1);

#elif defined(COMPILE_FOR_BRISC)
    // args: core, cores, meta, map, in (x rows | act tiles), out (act tiles | combine rows)
    constexpr uint32_t row_bytes = hidden_tiles * 64;    // x row (BF16)
    constexpr uint32_t out_row_bytes = cols * 32 * 4;    // down output row segment (FP32)
    constexpr uint32_t out_page_bytes = 32 * out_row_bytes;
    constexpr auto meta_args = TensorAccessorArgs<0>();
    constexpr auto map_args = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();
    constexpr auto in_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto meta = TensorAccessor(meta_args, get_common_arg_val<uint32_t>(2), META_BYTES);
    const auto map = TensorAccessor(map_args, get_common_arg_val<uint32_t>(3), 256);
    const auto in = TensorAccessor(in_args, get_common_arg_val<uint32_t>(4), gate_up ? row_bytes : 2048);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(5), gate_up ? 2048 : out_page_bytes);
    const uint32_t scratch = get_write_ptr(cb_bmeta);
    noc_async_read(meta.get_noc_addr(0), scratch, META_BYTES);
    noc_async_read_barrier();
    const Pieces<E, rb> pieces(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch) + E, core, cores, ub0, ub1);
    const uint32_t mp = get_write_ptr(cb_map);
    auto* slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp);
    DeviceZoneScopedN("X-BR");
    pieces.run([&](uint32_t, uint32_t gf, uint32_t n, uint32_t lo, uint32_t hi) {
        for (uint32_t g = 0; g < n; ++g) {
            noc_async_read(map.get_noc_addr(gf + g), mp + g * 256, 256);
        }
        noc_async_read_barrier();
        if constexpr (gate_up) {
            cb_reserve_back(cb_wt, rb);  // fixed-size slots: the TRISC indices never wrap inside a piece
            auto* wt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_wt));
            for (uint32_t g = 0; g < n; ++g) {  // routing weight tile: row r = slot r's weight in every column
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t v = slots[64 * g + 2 * r + 1];
                    const uint32_t pair = v | (v << 16);
                    for (uint32_t h = 0; h < 2; ++h) {
                        const uint32_t d = 512 * g + (r / 16) * 256 + h * 128 + (r % 16) * 8;
                        for (uint32_t i = 0; i < 8; ++i) {
                            wt[d + i] = pair;
                        }
                    }
                }
            }
            cb_push_back(cb_wt, rb);
            for (uint32_t g = 0; g < n; ++g) {  // x rows of group g (pads not read)
                cb_reserve_back(cb_rows, kt);
                const uint32_t d = get_write_ptr(cb_rows);
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t s = slots[64 * g + 2 * r];
                    if (s != 0xffffffffu) {
                        noc_async_read(in.get_noc_addr(s & 0xffffu), d + r * row_bytes, row_bytes);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_rows, kt);
            }
            for (uint32_t j = lo; j < hi; ++j) {
                for (uint32_t g = 0; g < n; ++g) {
                    cb_wait_front(cb_out, 1);
                    noc_async_write(get_read_ptr(cb_out), out.get_noc_addr((gf + g) * inter_tiles + j), 2048);
                    noc_async_writes_flushed();
                    cb_pop_front(cb_out, 1);
                }
            }
        } else {
            cb_reserve_back(cb_x, rb * kt);
            const uint32_t xa = get_write_ptr(cb_x);
            for (uint32_t t = 0; t < n * kt; ++t) {
                noc_async_read(in.get_noc_addr(gf * kt + t), xa + t * 2048, 2048);
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, rb * kt);
            for (uint32_t c = lo; c < hi; ++c) {
                for (uint32_t g = 0; g < n; ++g) {
                    cb_wait_front(cb_out, cols);
                    const uint32_t src = get_read_ptr(cb_out);
                    for (uint32_t r = 0; r < 32; ++r) {
                        const uint32_t s = slots[64 * g + 2 * r];
                        if (s != 0xffffffffu) {
                            const uint32_t t = s & 0xffffu, k = s >> 16;
                            noc_async_write(src + r * out_row_bytes,
                                            out.get_noc_addr(((t / 32) * col_groups + c) * top_k + k) +
                                                (t % 32) * out_row_bytes,
                                            out_row_bytes);
                        }
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_out, cols);
                }
            }
        }
    });
    noc_async_write_barrier();

#elif defined(COMPILE_FOR_TRISC)
    static_assert(DST_ACCUM_MODE, "FP32 destination accumulation required");
    constexpr uint32_t transpose = K_ALIGNED;
    deepseek_compute_kernel_init();
    if constexpr (gate_up) {
        PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (false)));
    } else {
        reconfig_data_format<SrcOrder::Reverse, true>(cb_x, cb_w);
        matmul_block_init(cb_x, cb_w, transpose, cols, 1, kt);
        pack_reconfig_data_format<true>(cb_out);
        pack_untilize_dest_init<cols>(cb_out);
    }
    DeviceZoneScopedN("X-TR");
    while (true) {
        cb_wait_front(cb_info, 1);
        const uint32_t n = read_tile_value(cb_info, 0, 0);
        if (n == END) {
            cb_pop_front(cb_info, 1);
            break;
        }
        const uint32_t f_bits = read_tile_value(cb_info, 0, 1);
        const uint32_t nu = read_tile_value(cb_info, 0, 2);
        cb_pop_front(cb_info, 1);
        if constexpr (gate_up) {
            compute_kernel_lib::tilize<kt, cb_rows, cb_x>(n);
            if (n < rb) {  // fixed-size slots
                cb_reserve_back(cb_x, (rb - n) * kt);
                cb_push_back(cb_x, (rb - n) * kt);
            }
            cb_wait_front(cb_wt, rb);
            pack_reconfig_data_format<true>(cb_out);
        }
        cb_wait_front(cb_x, rb * kt);
        for (uint32_t l = 0; l < nu; ++l) {
            cb_wait_front(cb_w, unit_tiles);
            for (uint32_t g = 0; g < n; ++g) {
                if constexpr (gate_up) {
                    reconfig_data_format<SrcOrder::Reverse, true>(cb_x, cb_w);
                    matmul_block_init(cb_x, cb_w, transpose, cols, 1, kt);
                }
                tile_regs_acquire();
                for (uint32_t k = 0; k < kt; ++k) {
                    matmul_block(cb_x, cb_w, g * kt + k, k * cols, 0, transpose, cols, 1, kt);
                }
                if constexpr (gate_up) {
                    reconfig_data_format(cb_wt, cb_wt);
                    copy_tile_init(cb_wt);
                    copy_tile(cb_wt, g, 3);
                    tile_regs_commit();
                    cb_reserve_back(cb_out, 1);
                    PACK(TTI_SEMWAIT(p_stall::STALL_TDMA | p_stall::STALL_CFG,
                                     semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
                    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                    PACK((_llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::calculate_ink_swiglu_rows, 0, 1, 0,
                                                                VectorMode::RC, f_bits)));
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                    pack_tile(0, cb_out, 0);
                    tile_regs_release();
                    cb_push_back(cb_out, 1);
                } else {
                    tile_regs_commit();
                    cb_reserve_back(cb_out, cols);
                    tile_regs_wait();
                    pack_untilize_dest<cols>(cb_out);
                    tile_regs_release();
                    cb_push_back(cb_out, cols);
                }
            }
            cb_pop_front(cb_w, unit_tiles);
        }
        cb_pop_front(cb_x, rb * kt);
        if constexpr (gate_up) {
            cb_pop_front(cb_wt, rb);
        }
    }
#endif
}
