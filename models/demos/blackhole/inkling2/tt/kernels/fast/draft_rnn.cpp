// SPDX-License-Identifier: Apache-2.0
// One recurrent step of the drafter's Markov head, fused (fast/draft_ops.py MarkovRNN), 8 cores, core j owning
// state / latent column tile j:
//     raw    = ctx[row] + [state | w1[prev]] @ W_sp            (FP32: ctx row FP32, BF16 operands, FP32 DEST)
//     gate   = sigmoid(raw[:, 0:256]), cand = tanh(raw[:, 256:512]), latent = tanh(raw[:, 512:768])
//     state' = cand + gate * (state - cand)  (= gate * state + (1 - gate) * cand)       -> BF16 latent / state'
// Core j computes raw column tiles j, 8 + j, 16 + j (gate / cand / latent of state tile j).
// NCRISC: the 16 x 3 W_sp tiles of the core ([512, 768] BF16 TILE, page kt * 24 + ct) in (kt, c) order.
// BRISC:  prev id of user u (word u of a uint32 row) -> the w1 row (row-major [V, 256] BF16, one 512 B page per id)
//         into row u of x tiles 8..15; state rows 0..users-1 into x tiles 0..7 and the core's state tile; ctx row
//         crow * users + u (the batched block interleaves users' rows) of the core's 3 FP32 tiles moved to row u;
//         afterwards rows 0..users-1 of latent / state'
//         tile j -> DRAM.  users = 1 is the single-sequence program (row 0, ctx row crow).
// The state input and output must be different tensors (every core reads the whole input state).
// Common args: [prev ids, w1, state in, ctx, W_sp, latent out, state out].  Per-core: [j].
#include <cstdint>
#include "read_vc.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace {
constexpr uint32_t KT = 16;  // [state (8) | e (8)] K tiles
constexpr uint32_t cb_x = 0, cb_w = 1, cb_ctx = 2, cb_st = 3, cb_stage = 4, cb_lat = 16, cb_nst = 17;
}  // namespace

void kernel_main() {
    ink_set_read_vc();
    const uint32_t j = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_NCRISC)
    const auto w = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(4), 2048);
    cb_reserve_back(cb_w, KT * 3);
    const uint32_t dst = get_write_ptr(cb_w);
    for (uint32_t k = 0; k < KT; ++k) {
        for (uint32_t c = 0; c < 3; ++c) {
            noc_async_read(w.get_noc_addr(k * 24 + c * 8 + j), dst + (k * 3 + c) * 2048, 2048);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, KT * 3);
#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t crow = get_named_compile_time_arg_val("crow");  // block row (user u: ctx row crow * users + u)
    constexpr uint32_t users = get_named_compile_time_arg_val("users");
    static_assert(crow < 16 && users >= 1 && users <= 4 && crow * users + users - 1 < 32);
    constexpr uint32_t row_bytes = (users + 1) / 2 * 64;  // BF16 rows 0..users-1 (64 B row pairs)
    const uint32_t stage = (get_write_ptr(cb_stage) + 63) & ~63u;
    // prev ids -> w1 rows (512 B each)
    const auto ids = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(0), 128);
    noc_async_read(ids.get_noc_addr(0), stage, 64);
    noc_async_read_barrier();
    const auto w1 = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(1), 512);
    for (uint32_t u = 0; u < users; ++u) {
        const uint32_t id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[u];
        noc_async_read(w1.get_noc_addr(id), stage + 64 + u * 512, 512);
    }
    const auto st = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(2), 2048);
    const auto ctx = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(3), 4096);
    cb_reserve_back(cb_x, KT);
    cb_reserve_back(cb_ctx, 3);
    cb_reserve_back(cb_st, 1);
    const uint32_t x = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < 8; ++t) {  // state rows 0..users-1 (row pairs of faces 0 / 1)
        const uint64_t s = st.get_noc_addr(t);
        noc_async_read(s, x + t * 2048, row_bytes);
        noc_async_read(s + 512, x + t * 2048 + 512, row_bytes);
    }
    const uint32_t sj = get_write_ptr(cb_st);
    noc_async_read(st.get_noc_addr(j), sj, row_bytes);
    noc_async_read(st.get_noc_addr(j) + 512, sj + 512, row_bytes);
    const uint32_t c0 = get_write_ptr(cb_ctx);
    for (uint32_t c = 0; c < 3; ++c) {  // ctx row crow * users + u -> row u of faces 0 / 1 (FP32: 64 B face rows)
        const uint64_t s = ctx.get_noc_addr(c * 8 + j);
        for (uint32_t u = 0; u < users; ++u) {
            const uint32_t r = crow * users + u;
            const uint32_t off = (r >> 4) * 2048 + (r & 15) * 64;
            noc_async_read(s + off, c0 + c * 4096 + u * 64, 64);
            noc_async_read(s + off + 1024, c0 + c * 4096 + 1024 + u * 64, 64);
        }
    }
    noc_async_read_barrier();
    // e = w1[prev] -> row u of x tiles 8..15 (values 32 t .. 32 t + 15 -> face 0, + 16 .. + 31 -> face 1)
    for (uint32_t u = 0; u < users; ++u) {
        for (uint32_t t = 0; t < 8; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* d =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x + (8 + t) * 2048 + f * 512 + u * 32);
                volatile tt_l1_ptr uint32_t* s =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 64 + u * 512 + t * 64 + f * 32);
                for (uint32_t q = 0; q < 8; ++q) {
                    d[q] = s[q];
                }
            }
        }
    }
    cb_push_back(cb_x, KT);
    cb_push_back(cb_ctx, 3);
    cb_push_back(cb_st, 1);
    // outputs: rows 0..users-1 (faces 0 / 1) of tile j
    const auto lat = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(5), 2048);
    const auto nst = TensorAccessor(tensor_accessor::make_interleaved_dspec<true>(), get_common_arg_val<uint32_t>(6), 2048);
    cb_wait_front(cb_lat, 1);
    cb_wait_front(cb_nst, 1);
    const uint32_t lt = get_read_ptr(cb_lat);
    const uint32_t nt = get_read_ptr(cb_nst);
    noc_async_write(lt, lat.get_noc_addr(j), 32 * users);
    noc_async_write(lt + 512, lat.get_noc_addr(j) + 512, 32 * users);
    noc_async_write(nt, nst.get_noc_addr(j), 32 * users);
    noc_async_write(nt + 512, nst.get_noc_addr(j) + 512, 32 * users);
    noc_async_write_barrier();
    cb_pop_front(cb_lat, 1);
    cb_pop_front(cb_nst, 1);
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_lat);
    cb_wait_front(cb_x, KT);
    cb_wait_front(cb_w, KT * 3);
    cb_wait_front(cb_ctx, 3);
    cb_wait_front(cb_st, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_w, cb_ctx);
    for (uint32_t c = 0; c < 3; ++c) {
        copy_tile(cb_ctx, c, c);
    }
    copy_tile_to_dst_init_short_with_dt(cb_ctx, cb_st);
    copy_tile(cb_st, 0, 3);
    reconfig_data_format_srca(cb_st, cb_w);
    matmul_block_init(cb_x, cb_w, 0, 3, 1, KT);
    for (uint32_t k = 0; k < KT; ++k) {
        matmul_block(cb_x, cb_w, k, k * 3, 0, 0, 3, 1, KT);
    }
    sigmoid_tile_init();
    sigmoid_tile(0);
    tanh_tile_init();
    tanh_tile(1);
    tanh_tile(2);
    sub_binary_tile_init();
    sub_binary_tile(3, 1, 3);
    mul_binary_tile_init();
    mul_binary_tile(0, 3, 3);
    add_binary_tile_init();
    add_binary_tile(3, 1, 3);
    tile_regs_commit();
    cb_reserve_back(cb_lat, 1);
    cb_reserve_back(cb_nst, 1);
    tile_regs_wait();
    pack_tile(2, cb_lat, 0);
    pack_tile(3, cb_nst, 0);
    tile_regs_release();
    cb_push_back(cb_lat, 1);
    cb_push_back(cb_nst, 1);
    cb_pop_front(cb_x, KT);
    cb_pop_front(cb_w, KT * 3);
    cb_pop_front(cb_ctx, 3);
    cb_pop_front(cb_st, 1);
#endif
}
