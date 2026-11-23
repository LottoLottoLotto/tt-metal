// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/remote_circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

// Gemma decode weight reader (G4_IN1, set by the 1D in0-multicast factory from TT_MM1D_IN1_* and
// TT_MM1D_PACKED_IN1). Interleaved DRAM weight, no in1 multicast. Per K block:
//  - G4_IN1_NOCS == 2: the in1 reads use both NoCs (dynamic NoC mode). G4_IN1_SPLIT 1: interleaved weight
//    alternates tiles, packed weight splits every contiguous read into two DRAM-aligned byte halves; 2: like 1,
//    but the interleaved weight gives the first half of each block's tiles to one NoC and the rest to the other;
//    3: odd workers read everything on the second NoC; 4: odd K blocks are read on the second NoC.
//  - G4_IN1_D > 1: up to G4_IN1_D blocks in flight, one NoC transaction id per block; blocks are published
//    to compute in K order as their reads complete (the CB holds more than G4_IN1_D blocks).
//  - G4_PACKED_IN1: the weight is stored in a worker-bank packed layout for per_core_N = G4_PACKED_PN
//    (W = Nt / pn workers, worker w owns logical columns [w*pn, (w+1)*pn), B = G4_PACKED_BANKS). Logical
//    tile (k, n), w = n / pn, c = n % pn, lives in interleaved page  bank + B * slot  with
//      G4_PACKED_G == 0:  bank = map[w],               slot = rank(w) * Kt * pn + k * pn + c
//      G4_PACKED_G == G:  bank = (w + k / G) % B,      slot = w * (Kt / B) * pn + (k / G / B) * G * pn
//                                                             + (k % G) * pn + c
//    map = G4_PACKED_BANK_MAP (one bank per worker, W / B workers per bank) or w % B by default, and rank(w)
//    = #{w' < w : map[w'] == map[w]}. A worker's rows of one bank are contiguous and each K block is read with
//    one read per bank run.
// Tiles land in the CB in the same order as on the default path, so compute is unchanged.
#if defined(G4_IN1) && (!defined(SKIP_MCAST) || defined(IN1_SHARDED) || defined(ENABLE_GLOBAL_CB) || \
                        defined(IN1_DRAM_WIDTH_SHARDED) || defined(IN1_DRAM_HEIGHT_SHARDED))
#error "G4_IN1 requires the interleaved, non-multicast in1 reader"
#endif
void kernel_main() {
    // READER
    uint32_t rt_args_idx = 0;
    // in1 tensor args
    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // in1 mcast args
    const uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);

    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);

    // WRITER
    // out tensor args
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

    // padding args (READER)
    const uint32_t last_block_w = get_arg_val<uint32_t>(rt_args_idx++);
    // padding args (WRITER)
    const uint32_t out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_subblock_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_block_tiles_h_skip = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_subblock_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_subblock_tiles_addr_skip = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_block_tiles_w_skip = get_arg_val<uint32_t>(rt_args_idx++);

    // COMPILE TIME ARGS
    // READER
    // in1 tensor args
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in1_tensor_next_block_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in1_tensor_next_w_dim_block_stride = get_compile_time_arg_val(3);
    // in1 block args
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(5);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);

    // in1 mcast args
    constexpr uint32_t in1_mcast_num_dests = get_compile_time_arg_val(12);
    constexpr uint32_t in1_mcast_num_cores = get_compile_time_arg_val(13);
    // batch args
    constexpr uint32_t KtNt = get_compile_time_arg_val(14);
    constexpr uint32_t batch = get_compile_time_arg_val(15);
    constexpr uint32_t bcast_B = get_compile_time_arg_val(16);
    // sparsity args
    constexpr uint32_t batchB = get_compile_time_arg_val(17);
    constexpr uint32_t sparsity_pagesize = get_compile_time_arg_val(18);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(19);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(20);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(21);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(22);
    constexpr uint32_t out_tensor_next_w_dim_block_stride = get_compile_time_arg_val(23);
    constexpr uint32_t out_tensor_next_h_dim_block_stride = get_compile_time_arg_val(24);
    // out subblock args
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(26);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(27);
    // batch args
    constexpr uint32_t MtNt = get_compile_time_arg_val(28);  // if 0
    // Don't need batch; same as batch from READER args

    // When sparsity is disabled, we just loop once
    constexpr uint32_t batchB_lim = batchB == 0 ? 1u : batchB;

#ifdef FUSE_BIAS
    // in3 mcast args
    const uint32_t in3_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in3_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t in3_tensor_stride_w = get_compile_time_arg_val(29);

    constexpr uint32_t dfb_id_in3 = get_named_compile_time_arg_val("cb_bias");
    // Use the CB page size (padded to the DRAM alignment by the factory) for DRAM reads
    // and L1 write strides, NOT the raw tile size. On Blackhole, the DRAM read alignment
    // is 64B, so a sub-64B tile (e.g. 32B for a (1,16) bf16 bias tile) cannot be read
    // directly from DRAM, and 32B-strided L1 writes land at non-64B-aligned addresses
    // that disagree with the 64B-aligned DRAM source. The factory pads the CB page to
    // 64B; the unpacker still reads the actual 32B tile from the padded page via tile
    // dims. For tiles already >= dram_alignment (e.g. 32x32 bf16 = 2048B), the page size
    // equals the tile size, so this is a no-op. Mirrors how in0/in1 readers walk at the
    // aligned stride.
    const uint32_t bias_single_tile_size_bytes = get_local_cb_interface(dfb_id_in3).fifo_page_size;
    constexpr const uint32_t in3_tile_hw = get_tile_hw(dfb_id_in3);

#ifndef BIAS_SHARDED
    uint32_t l1_write_addr_in3;
    // Bias accessor will be defined later after TensorAccessor args
#endif  // BIAS_SHARDED
#else
    rt_args_idx += 2;  // Skip over placeholders
#endif  // FUSE_BIAS
#ifndef OUT_SHARDED
    const uint32_t last_num_blocks_w_dim = get_arg_val<uint32_t>(rt_args_idx++);
#endif  // OUT_SHARDED

    constexpr bool fuse_op_all_gather = (bool)get_compile_time_arg_val(30);
    constexpr bool fuse_op_reduce_scatter = (bool)get_compile_time_arg_val(31);

    MatmulOpReceiver fused_op_receiver;
    OpSignaler op_signaler;
    if constexpr (fuse_op_all_gather) {
        fused_op_receiver = MatmulOpReceiver(
            false, /* wait_for_op_signal */
            rt_args_idx,
            num_blocks_inner_dim,
            in1_block_h /* tiles_per_block (in the same dimension */
        );
    } else if constexpr (fuse_op_reduce_scatter) {
        op_signaler = OpSignaler(rt_args_idx);
    }

    constexpr auto in1_args = TensorAccessorArgs<32>();
    constexpr auto sparsity_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<sparsity_args.next_compile_time_args_offset()>();
#ifdef FUSE_BIAS
    constexpr auto bias_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto after_bias_offset = bias_args.next_compile_time_args_offset();
#else
    constexpr auto after_bias_offset = out_args.next_compile_time_args_offset();
#endif  // FUSE_BIAS

// RT and COMPILE TIME ARGS for DRAM sharded weights
#ifdef IN1_DRAM_WIDTH_SHARDED
    const uint32_t vc = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_dram_shards_to_read = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t dram_tensor_start_offset = get_arg_val<uint32_t>(rt_args_idx++);
    tt_l1_ptr uint32_t* in1_block_w_dram_stride_bytes = (tt_l1_ptr uint32_t*)get_arg_addr(rt_args_idx++);
    tt_l1_ptr uint32_t* current_dram_bank_id = (tt_l1_ptr uint32_t*)get_arg_addr(rt_args_idx++);

    constexpr uint32_t in1_dram_block_num_tiles = get_compile_time_arg_val(after_bias_offset);
    constexpr uint32_t in1_block_w_dram_bytes = get_compile_time_arg_val(after_bias_offset + 1);
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifdef IN1_DRAM_HEIGHT_SHARDED
    constexpr uint32_t in1_KtNt_per_batch = get_compile_time_arg_val(after_bias_offset);        // K*N tiles per batch
    constexpr uint32_t in1_batches_per_bank = get_compile_time_arg_val(after_bias_offset + 1);  // batches per DRAM bank
#endif  // IN1_DRAM_HEIGHT_SHARDED

#ifdef FUSE_BIAS
#ifndef BIAS_SHARDED
    const auto s3 = TensorAccessor(bias_args, in3_tensor_addr);
#endif  // BIAS_SHARDED
#endif  // FUSE_BIAS

    constexpr uint32_t dfb_id_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(dfb_id_in1);
    constexpr const uint32_t in1_tile_hw = get_tile_hw(dfb_id_in1);
    // Tiles whose size is not a multiple of the DRAM alignment are padded to it in DRAM, and the
    // interleaved in1 CB pages are sized to match (see the program factory). On the plain interleaved
    // path the NOC reads the unpadded tile of data into each padded slot and tiles are laid out /
    // multicast at the padded stride. No-op when already aligned. The sharded / DRAM-sharded paths
    // keep their natural (unpadded) stride.
    constexpr uint32_t in1_aligned_tile_size_bytes =
        (in1_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
#if !defined(IN1_SHARDED) && !defined(IN1_DRAM_WIDTH_SHARDED) && !defined(IN1_DRAM_HEIGHT_SHARDED) && \
    !defined(ENABLE_GLOBAL_CB)
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_aligned_tile_size_bytes;
#else
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;
#endif

    constexpr uint32_t dfb_id_out0 = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(dfb_id_out0);
    constexpr const uint32_t output_tile_hw = get_tile_hw(dfb_id_out0);

    Noc noc;
    DataflowBuffer dfb_in1(dfb_id_in1);
    DataflowBuffer dfb_out(dfb_id_out0);
    Semaphore<> sender_sem(get_compile_time_arg_val(10));
    Semaphore<> receiver_sem(get_compile_time_arg_val(11));
#ifdef FUSE_BIAS
    DataflowBuffer dfb_in3(dfb_id_in3);
#endif

//  READER
#ifdef IN1_SHARDED
    dfb_in1.reserve_back(in1_block_num_tiles * num_blocks_inner_dim);
    dfb_in1.push_back(in1_block_num_tiles * num_blocks_inner_dim);
#elif !defined(ENABLE_GLOBAL_CB)
    uint32_t l1_write_addr_in1;

    [[maybe_unused]] const auto s1 = TensorAccessor(in1_args, in1_tensor_addr);
#endif  // IN1_SHARDED / ENABLE_GLOBAL_CB

#ifdef ENABLE_GLOBAL_CB
    constexpr uint32_t remote_cb_id = tt::CBIndex::c_31;
    const uint32_t in1_fifo_tiles = get_local_cb_interface(dfb_id_in1).fifo_num_pages;
#endif

#ifdef G4_IN1
    constexpr uint32_t g4_d = G4_IN1_D;
    constexpr uint32_t g4_nocs = G4_IN1_NOCS;
    static_assert(g4_d >= 1 && g4_d < NOC_MAX_TRANSACTION_ID && (g4_nocs == 1 || g4_nocs == 2));
    const Noc g4_noc0(noc.get_noc_id());
    const Noc g4_noc1(noc.get_noc_id() ^ 1);
    // Whole-read NoC choice for G4_IN1_SPLIT 3 (per worker: odd workers, or G4_NOC_MAP[worker] when given; K block
    // parity for 4 is applied per block).
#ifdef G4_NOC_MAP
    constexpr uint8_t g4_noc_map[] = {G4_NOC_MAP};
    const bool g4_odd_worker = g4_noc_map[in1_tensor_start_tile_id / in1_block_w] != 0;
#else
    const bool g4_odd_worker = ((in1_tensor_start_tile_id / in1_block_w) & 1) != 0;
#endif
#if G4_IN1_SPLIT >= 5
    // G4_IN1_SPLIT 5/6: per DRAM bank, bit b of the mask selects the second NoC for reads from bank b. The factory
    // gives workers west / east of logical column G4_EAST_X the masks that keep each bank's data on the NoC whose
    // route from the bank's column to the worker is short.
    const uint32_t g4_noc1_banks =
        get_absolute_logical_x() < G4_EAST_X ? G4_NOC1_BANKS_WEST : G4_NOC1_BANKS_EAST;
#else
    constexpr uint32_t g4_noc1_banks = 0;
#endif
    const uint32_t g4_fifo_limit = get_local_cb_interface(dfb_id_in1).fifo_limit;
    const uint32_t g4_fifo_size = get_local_cb_interface(dfb_id_in1).fifo_size;
    uint32_t g4_pending = 0;  // blocks issued but not yet pushed
    uint32_t g4_issue_trid = 1;
    uint32_t g4_wait_trid = 1;
#ifdef G4_PACKED_IN1
    static_assert(in1_block_w == G4_PACKED_PN);
    // This core's worker index follows from its first output column (non-transposed weight).
    const uint32_t g4_worker = in1_tensor_start_tile_id / G4_PACKED_PN;
#if G4_PACKED_G == 0
#ifdef G4_PACKED_BANK_MAP
    constexpr uint8_t g4_bank_map[] = {G4_PACKED_BANK_MAP};
    const uint32_t g4_fixed_bank = g4_bank_map[g4_worker];
    uint32_t g4_rank = 0;
    for (uint32_t i = 0; i < g4_worker; ++i) {
        g4_rank += g4_bank_map[i] == g4_fixed_bank ? 1 : 0;
    }
#else
    const uint32_t g4_fixed_bank = g4_worker % G4_PACKED_BANKS;
    const uint32_t g4_rank = g4_worker / G4_PACKED_BANKS;
#endif
#endif
    AllocatorBank<AllocatorBankType::DRAM> g4_dram;
    auto g4_read_run = [&](const Noc& n, uint32_t bank, uint32_t src, uint32_t dst, uint32_t bytes) {
        if constexpr (g4_d > 1) {
            n.async_read<NocOptions::TXN_ID>(
                g4_dram, CoreLocalMem<uint32_t>(dst), bytes, {.bank_id = bank, .addr = src}, {},
                NocOptVals{.trid = g4_issue_trid});
        } else {
            n.async_read(g4_dram, CoreLocalMem<uint32_t>(dst), bytes, {.bank_id = bank, .addr = src}, {});
        }
    };
#else
    auto g4_read_page = [&](const Noc& n, uint32_t page, uint32_t dst) {
        if constexpr (g4_d > 1) {
            n.async_read<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(
                s1, CoreLocalMem<uint32_t>(dst), in1_single_tile_size_bytes, {.page_id = page}, {},
                NocOptVals{.trid = g4_issue_trid});
        } else {
            n.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                s1, CoreLocalMem<uint32_t>(dst), in1_single_tile_size_bytes, {.page_id = page}, {});
        }
    };
#endif
    auto g4_wait = [&]() {
        if constexpr (g4_d > 1) {
            g4_noc0.async_read_barrier<NocOptions::TXN_ID>({.trid = g4_wait_trid});
            if constexpr (g4_nocs == 2) {
                g4_noc1.async_read_barrier<NocOptions::TXN_ID>({.trid = g4_wait_trid});
            }
        } else {
            g4_noc0.async_read_barrier();
            if constexpr (g4_nocs == 2) {
                g4_noc1.async_read_barrier();
            }
        }
    };
#endif  // G4_IN1

    //  WRITER
    const auto s = TensorAccessor(out_args, out_tensor_addr);
    // `s` is only consumed inside the `#ifndef OUT_SHARDED` write path below; mark it used so
    // sharded builds don't warn (-Wunused-but-set-variable).
    (void)s;

    // sparsity accessor
    constexpr uint32_t dfb_id_sparsity = get_named_compile_time_arg_val("cb_sparsity");
    DataflowBuffer dfb_sparsity(dfb_id_sparsity);
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr);

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

#ifdef IN1_SHARDED
    uint64_t in1_start_address = dfb_in1.get_write_ptr();
#endif  // IN1_SHARDED
#endif  // SKIP_MCAST

    uint32_t l1_write_addr_sparsity = 0;
    if constexpr (batchB > 0) {
        dfb_sparsity.reserve_back(1);
        l1_write_addr_sparsity = dfb_sparsity.get_write_ptr();
    }

#ifdef IN1_DRAM_WIDTH_SHARDED
    constexpr uint32_t in1_dram_block_size_bytes = in1_dram_block_num_tiles * in1_single_tile_size_bytes;
    uint32_t in1_block_w_bytes = in1_block_w * in1_single_tile_size_bytes;
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifdef IN1_DRAM_HEIGHT_SHARDED
    constexpr uint32_t in1_batch_stride_bytes = in1_KtNt_per_batch * in1_single_tile_size_bytes;
#endif  // IN1_DRAM_HEIGHT_SHARDED

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in1_batch_tile_id = in1_tensor_start_tile_id;

#ifdef IN1_DRAM_HEIGHT_SHARDED
        // Compute DRAM bank and offset for this batch
        uint32_t in1_dram_bank_id = b / in1_batches_per_bank;
        uint32_t in1_batch_in_shard = b % in1_batches_per_bank;
        AllocatorBank<AllocatorBankType::DRAM> dram_src;
        uint32_t in1_dram_batch_offset = in1_batch_in_shard * in1_batch_stride_bytes;
#endif  // IN1_DRAM_HEIGHT_SHARDED

        if constexpr (batchB > 0) {
            noc.async_read(s_sparsity, dfb_sparsity, sparsity_pagesize, {.page_id = b}, {.offset_bytes = 0});
            noc.async_read_barrier();
        }

        for (uint32_t bB = 0; bB < batchB_lim; ++bB) {
            if constexpr (batchB > 0) {
                if (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr_sparsity)[bB] == 0) {
                    out_tensor_start_tile_id += MtNt;
                    in1_batch_tile_id += KtNt;
                    continue;
                }
            }

            uint32_t in1_tensor_current_h_dim_block_tile_id = in1_batch_tile_id;
            uint32_t out_tensor_current_h_dim_block_tile_id = out_tensor_start_tile_id;
            for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
                uint32_t in1_tensor_current_w_dim_block_tile_id = in1_tensor_current_h_dim_block_tile_id;
                uint32_t out_tensor_current_w_dim_block_tile_id = out_tensor_current_h_dim_block_tile_id;
#ifdef FUSE_BIAS
                uint32_t in3_tensor_current_w_dim_block_tile_id = in3_tensor_start_tile_id;
#endif  // FUSE_BIAS
                for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                    uint32_t in1_tensor_current_inner_dim_block_start_tile_id = in1_tensor_current_w_dim_block_tile_id;
#ifdef IN1_DRAM_WIDTH_SHARDED
                    // Reset DRAM read offset for each bh block — the inner dim loop
                    // advances through K, and each output row block re-reads the same
                    // in1 columns from K=0. (bw is always 1 for DRAM-sharded senders.)
                    uint32_t l1_read_addr_in1_offset = 0;
#endif  // IN1_DRAM_WIDTH_SHARDED

                    for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                        if constexpr (fuse_op_all_gather) {
                            fused_op_receiver.update_current_block_start_tile_id(
                                block, in1_tensor_current_inner_dim_block_start_tile_id, in1_batch_tile_id);
                        }
#if defined(ENABLE_GLOBAL_CB)
                        // The tensor prefetcher pushes this receiver's K-blocks in natural order.
                        // Keep one block of lookahead: publish the current block to compute, then
                        // wait for the unpack engine to drain the previous block before returning
                        // its remote-CB credit to the prefetcher.
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        experimental::remote_cb_wait_front(remote_cb_id, block == 0 ? 1u : 2u);
#elif defined(IN1_DRAM_WIDTH_SHARDED)
                        // Operand 1 - DRAM width sharded
                        dfb_in1.reserve_back(in1_block_num_tiles);

                        uint64_t in1_start_address =
                            dfb_in1.get_write_ptr();  // copy start address of block, to be used for mcasting

                        uint32_t l1_write_addr_in1_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        AllocatorBank<AllocatorBankType::DRAM> dram_bank;
                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint32_t shard_bank_id = current_dram_bank_id[next_bank_id_and_dram_stride_index];
                            uint32_t shard_base_addr = in1_tensor_addr;
                            if (i == 0) {
                                shard_base_addr += dram_tensor_start_offset;
                            }
                            noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                dram_bank,
                                in1_single_tile_size_bytes,
                                {.bank_id = shard_bank_id, .addr = shard_base_addr},
                                NocOptVals{.vc = vc});

                            uint32_t l1_read_addr_in1 = l1_read_addr_in1_offset;
                            uint32_t l1_write_addr_in1 = dfb_in1.get_write_ptr() + l1_write_addr_in1_offset;
                            uint32_t in1_block_w_dram =
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index] /
                                in1_single_tile_size_bytes;

                            for (uint32_t m = 0; m < in1_block_h; ++m) {
                                uint32_t l1_read_addr_in1_temp = l1_read_addr_in1;
                                uint32_t l1_write_addr_in1_temp = l1_write_addr_in1;
                                for (uint32_t w = 0; w < in1_block_w_dram; ++w) {
                                    noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                        dram_bank,
                                        CoreLocalMem<uint32_t>(l1_write_addr_in1_temp),
                                        in1_single_tile_size_bytes,
                                        {.bank_id = shard_bank_id, .addr = shard_base_addr + l1_read_addr_in1_temp},
                                        {},
                                        NocOptVals{.vc = vc});
                                    l1_read_addr_in1_temp += in1_single_tile_size_bytes;
                                    l1_write_addr_in1_temp += in1_single_tile_size_bytes;
                                }
                                l1_read_addr_in1 += in1_block_w_dram_bytes;
                                l1_write_addr_in1 += in1_block_w_bytes;
                            }
                            l1_write_addr_in1_offset +=
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index];
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        l1_read_addr_in1_offset += in1_dram_block_size_bytes;
                        noc.async_read_barrier();
#elif defined(IN1_DRAM_HEIGHT_SHARDED)
                        // Operand 1 - DRAM height sharded (batched)
                        // Each DRAM bank holds batches_per_bank complete [K, N] matrices
                        // Bank and offset computed at start of batch loop
                        dfb_in1.reserve_back(in1_block_num_tiles);

                        l1_write_addr_in1 = dfb_in1.get_write_ptr();
                        uint64_t in1_start_address =
                            l1_write_addr_in1;  // copy start address of block, to be used for mcasting

                        // Read in1 block from the correct DRAM bank
                        // Tile layout within a batch: row-major [K, N], same strides as interleaved
                        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in1_block_h; ++h) {
                            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in1_block_w; ++w) {
                                if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                    uint32_t tile_byte_offset =
                                        in1_dram_batch_offset + in1_tensor_tile_id * in1_single_tile_size_bytes;
                                    noc.async_read(
                                        dram_src,
                                        CoreLocalMem<uint32_t>(l1_write_addr_in1),
                                        in1_single_tile_size_bytes,
                                        {.bank_id = in1_dram_bank_id, .addr = in1_tensor_addr + tile_byte_offset},
                                        {});
                                }
                                l1_write_addr_in1 += in1_single_tile_size_bytes;
                                in1_tensor_tile_id += in1_tensor_stride_w;
                            }
                            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                        }
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#elif defined(G4_IN1)
                        // Operand 1 - Gemma decode reader (see G4_IN1 above): reserve room for every unpublished
                        // block plus this one, issue this block's reads and publish the oldest block once g4_d
                        // blocks are in flight.
                        dfb_in1.reserve_back(in1_block_num_tiles * (g4_pending + 1));
                        {
                            uint32_t g4_dst = dfb_in1.get_write_ptr() + g4_pending * in1_block_size_bytes;
                            if (g4_dst >= g4_fifo_limit) {
                                g4_dst -= g4_fifo_size;
                            }
                            const bool g4_block_other = g4_nocs == 2 && (G4_IN1_SPLIT == 3   ? g4_odd_worker
                                                                         : G4_IN1_SPLIT == 4 ? (block & 1) != 0
                                                                                             : false);
#ifdef G4_PACKED_IN1
                            constexpr uint32_t pn = G4_PACKED_PN;
                            constexpr uint32_t banks = G4_PACKED_BANKS;
                            uint32_t g4_k = block * in1_block_h;
                            for (uint32_t h = 0; h < in1_block_h;) {
#if G4_PACKED_G == 0
                                const uint32_t g4_bank = g4_fixed_bank;
                                const uint32_t g4_slot = g4_rank * G4_PACKED_KT * pn + g4_k * pn;
                                const uint32_t g4_rows = in1_block_h - h;
#else
                                constexpr uint32_t group = G4_PACKED_G;
                                const uint32_t g4_group = g4_k / group;
                                const uint32_t g4_row = g4_k - g4_group * group;
                                const uint32_t g4_bank = (g4_worker + g4_group) % banks;
                                const uint32_t g4_slot = g4_worker * (G4_PACKED_KT / banks) * pn +
                                                         (g4_group / banks) * group * pn + g4_row * pn;
                                const uint32_t g4_rows =
                                    in1_block_h - h < group - g4_row ? in1_block_h - h : group - g4_row;
#endif
                                const uint32_t g4_bytes = g4_rows * pn * in1_aligned_tile_size_bytes;
                                const uint32_t g4_src = in1_tensor_addr + g4_slot * in1_aligned_tile_size_bytes;
                                if constexpr (g4_nocs == 2 && G4_IN1_SPLIT <= 2) {
                                    const uint32_t first = (g4_bytes / 2) & ~(DRAM_ALIGNMENT - 1);
                                    g4_read_run(g4_noc0, g4_bank, g4_src, g4_dst, first);
                                    g4_read_run(g4_noc1, g4_bank, g4_src + first, g4_dst + first, g4_bytes - first);
                                } else {
                                    const bool g4_other = G4_IN1_SPLIT >= 5 ? ((g4_noc1_banks >> g4_bank) & 1) != 0
                                                                            : g4_block_other;
                                    g4_read_run(g4_other ? g4_noc1 : g4_noc0, g4_bank, g4_src, g4_dst, g4_bytes);
                                }
                                g4_dst += g4_bytes;
                                h += g4_rows;
                                g4_k += g4_rows;
                            }
#else
                            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                            uint32_t g4_t = 0;
                            for (uint32_t h = 0; h < in1_block_h; ++h) {
                                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                                for (uint32_t w = 0; w < in1_block_w; ++w, ++g4_t) {
                                    if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                        const bool other =
                                            g4_nocs == 2 &&
                                            (G4_IN1_SPLIT == 1   ? (g4_t & 1) != 0
                                             : G4_IN1_SPLIT == 2 ? g4_t >= in1_block_num_tiles / 2
                                             : G4_IN1_SPLIT >= 5
                                                 ? ((g4_noc1_banks >> (in1_tensor_tile_id % G4_IN1_BANKS)) & 1) != 0
                                                 : g4_block_other);
                                        g4_read_page(other ? g4_noc1 : g4_noc0, in1_tensor_tile_id, g4_dst);
                                    }
                                    g4_dst += in1_aligned_tile_size_bytes;
                                    in1_tensor_tile_id += in1_tensor_stride_w;
                                }
                                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                            }
#endif
                        }
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;
                        g4_issue_trid = g4_issue_trid == g4_d ? 1 : g4_issue_trid + 1;
                        if (++g4_pending == g4_d) {
                            g4_wait();
                            dfb_in1.push_back(in1_block_num_tiles);
                            g4_wait_trid = g4_wait_trid == g4_d ? 1 : g4_wait_trid + 1;
                            --g4_pending;
                        }
#elif !defined(IN1_SHARDED)
                        // Operand 1 - interleaved
                        dfb_in1.reserve_back(in1_block_num_tiles);
                        uint32_t in1_write_offset = 0;
                        uint64_t in1_start_address =
                            dfb_in1.get_write_ptr();  // copy start address of block, to be used for mcasting

                        // Copy in1 block into CB, as the default kernel
                        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in1_block_h; ++h) {
                            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                            for (uint32_t w = 0; w < in1_block_w; ++w) {
                                if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                    noc.async_read(
                                        s1,
                                        dfb_in1,
                                        in1_single_tile_size_bytes,
                                        {.page_id = in1_tensor_tile_id},
                                        {.offset_bytes = in1_write_offset});
                                }
                                in1_write_offset += in1_aligned_tile_size_bytes;
                                in1_tensor_tile_id += in1_tensor_stride_w;
                            }
                            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
                        }
                        in1_tensor_current_inner_dim_block_start_tile_id += in1_tensor_next_block_stride;

                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#endif  // IN1_DRAM_WIDTH_SHARDED / IN1_DRAM_HEIGHT_SHARDED / IN1_SHARDED

#ifndef SKIP_MCAST
                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        sender_sem.wait(in1_mcast_num_dests);
                        sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        MulticastEndpoint mcast_dst;
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(static_cast<uint32_t>(in1_start_address)),
                            mcast_dst,
                            in1_block_size_bytes,
                            in1_mcast_num_cores,
                            {},
                            {.noc_x_start = in1_mcast_dest_noc_start_x,
                             .noc_y_start = in1_mcast_dest_noc_start_y,
                             .noc_x_end = in1_mcast_dest_noc_end_x,
                             .noc_y_end = in1_mcast_dest_noc_end_y,
                             .addr = static_cast<uint32_t>(in1_start_address)},
                            true);

                        // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                        // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                        // statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                        // On Blackhole the flush is needed because NoC latency is higher than L1 <-> RISCV latency
                        // which means data could be changed before
                        //  write is issued.
                        noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE

                        // We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        receiver_sem.set_multicast(
                            noc,
                            in1_mcast_dest_noc_start_x,
                            in1_mcast_dest_noc_start_y,
                            in1_mcast_dest_noc_end_x,
                            in1_mcast_dest_noc_end_y,
                            in1_mcast_num_cores);
#endif  // SKIP_MCAST

#if !defined(IN1_SHARDED) && !defined(G4_IN1)
                        dfb_in1.push_back(in1_block_num_tiles);
#endif  // IN1_SHARDED
#ifdef ENABLE_GLOBAL_CB
                        if (block >= 1) {
                            while (!dfb_in1.pages_reservable_at_back(in1_fifo_tiles - in1_block_num_tiles)) {
                                invalidate_l1_cache();
                            }
                            experimental::remote_cb_pop_front(remote_cb_id, 1);
                        }
#endif
                    }
#ifdef G4_IN1
                    // Publish the blocks still in flight, oldest first.
                    for (; g4_pending > 0; --g4_pending) {
                        g4_wait();
                        dfb_in1.push_back(in1_block_num_tiles);
                        g4_wait_trid = g4_wait_trid == g4_d ? 1 : g4_wait_trid + 1;
                    }
                    if constexpr (g4_d > 1) {
                        // The packet tag register is sticky: leave later plain reads untagged.
                        noc_async_read_set_trid(0, g4_noc0.get_noc_id());
                        if constexpr (g4_nocs == 2) {
                            noc_async_read_set_trid(0, g4_noc1.get_noc_id());
                        }
                    }
#endif
#ifdef ENABLE_GLOBAL_CB
                    if (num_blocks_inner_dim > 0) {
                        while (!dfb_in1.pages_reservable_at_back(in1_fifo_tiles)) {
                            invalidate_l1_cache();
                        }
                        experimental::remote_cb_pop_front(remote_cb_id, 1);
                    }
#endif
#ifdef FUSE_BIAS
                    // Only read bias on first batch, or we have multiple output blocks
                    if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                        // Operand 1
#ifndef BIAS_SHARDED
                        dfb_in3.reserve_back(in1_block_w);
                        uint32_t in3_write_offset = 0;

                        uint64_t in3_start_address =
                            dfb_in3.get_write_ptr();        // copy start address of block, to be used for mcasting
                        uint32_t in3_block_size_bytes = 0;  // can be optimized later, pass it to kernel

#ifdef IN1_DRAM_WIDTH_SHARDED
                        uint32_t l1_write_addr_in3_offset = 0;
                        uint32_t next_bank_id_and_dram_stride_index = 0;

                        AllocatorBank<AllocatorBankType::DRAM> bias_dram_bank;
                        for (uint32_t i = 0; i < num_dram_shards_to_read; ++i) {
                            uint32_t bias_shard_bank_id = current_dram_bank_id[next_bank_id_and_dram_stride_index];
                            uint32_t bias_shard_base_addr = in3_tensor_addr;
                            if (i == 0) {
                                // dram_tensor_start_offset is in in1 tile bytes; convert to
                                // bias tile bytes since bias_dtype may differ from in1_dtype.
                                bias_shard_base_addr += (dram_tensor_start_offset / in1_single_tile_size_bytes) *
                                                        bias_single_tile_size_bytes;
                            }

                            noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                bias_dram_bank,
                                bias_single_tile_size_bytes,
                                {.bank_id = bias_shard_bank_id, .addr = bias_shard_base_addr},
                                NocOptVals{.vc = vc});

                            uint32_t l1_read_addr_in3 = 0;
                            l1_write_addr_in3 = dfb_in3.get_write_ptr() + l1_write_addr_in3_offset;
                            // in1_block_w_dram_stride_bytes is in in1 tile bytes, so divide
                            // by in1_single_tile_size_bytes (not bias) to get the tile count.
                            uint32_t in3_block_w_dram =
                                in1_block_w_dram_stride_bytes[next_bank_id_and_dram_stride_index] /
                                in1_single_tile_size_bytes;

                            for (uint32_t w = 0; w < in3_block_w_dram; ++w) {
                                noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                                    bias_dram_bank,
                                    CoreLocalMem<uint32_t>(l1_write_addr_in3),
                                    bias_single_tile_size_bytes,
                                    {.bank_id = bias_shard_bank_id, .addr = bias_shard_base_addr + l1_read_addr_in3},
                                    {},
                                    NocOptVals{.vc = vc});
                                l1_read_addr_in3 += bias_single_tile_size_bytes;
                                l1_write_addr_in3 += bias_single_tile_size_bytes;
                                in3_block_size_bytes += bias_single_tile_size_bytes;
                            }
                            // Advance L1 offset in bias tile bytes, not in1 stride bytes.
                            l1_write_addr_in3_offset += in3_block_w_dram * bias_single_tile_size_bytes;
                            next_bank_id_and_dram_stride_index += 2;
                        }
                        noc.async_read_barrier();
#else
                        // Copy in1 block into CB, as the default kernel
                        uint32_t in3_tensor_tile_id = in3_tensor_current_w_dim_block_tile_id;
                        for (uint32_t w = 0; w < in1_block_w; ++w) {
                            if (bw < num_blocks_w_dim - 1 || w < last_block_w) {
                                noc.async_read(
                                    s3,
                                    dfb_in3,
                                    bias_single_tile_size_bytes,
                                    {.page_id = in3_tensor_tile_id},
                                    {.offset_bytes = in3_write_offset});
                            }
                            in3_write_offset += bias_single_tile_size_bytes;
                            in3_tensor_tile_id += in3_tensor_stride_w;
                            in3_block_size_bytes += bias_single_tile_size_bytes;
                        }
                        // Barrier! make sure the reads are done
                        noc.async_read_barrier();
#endif  // IN1_DRAM_WIDTH_SHARDED

#ifndef SKIP_MCAST

                        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        sender_sem.wait(in1_mcast_num_dests);
                        sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        MulticastEndpoint mcast_dst;
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(static_cast<uint32_t>(in3_start_address)),
                            mcast_dst,
                            in3_block_size_bytes,
                            in1_mcast_num_cores,
                            {},
                            {.noc_x_start = in1_mcast_dest_noc_start_x,
                             .noc_y_start = in1_mcast_dest_noc_start_y,
                             .noc_x_end = in1_mcast_dest_noc_end_x,
                             .noc_y_end = in1_mcast_dest_noc_end_y,
                             .addr = static_cast<uint32_t>(in3_start_address)},
                            true);
                        // Note: no need for write barrier, since these two multicasts are done on the same noc id, same
                        // vc, same cmd_buf Also, this only works because we are setting VCs statically (using
                        // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                        // On Blackhole the flush is needed because NoC latency is higherthan L1 <-> RISCV
                        // latency which means data could be changed before write is issued.
                        noc.async_writes_flushed();
#endif  // ARCH_BLACKHOLE

                        // We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        receiver_sem.set_multicast(
                            noc,
                            in1_mcast_dest_noc_start_x,
                            in1_mcast_dest_noc_start_y,
                            in1_mcast_dest_noc_end_x,
                            in1_mcast_dest_noc_end_y,
                            in1_mcast_num_cores);
#endif  // SKIP_MCAST

                        dfb_in3.push_back(in1_block_w);
#else
                        dfb_in3.reserve_back(in1_block_w);
                        dfb_in3.push_back(in1_block_w);
#endif  // BIAS_SHARDED
                    }
#endif  // FUSE_BIAS

#ifndef OUT_SHARDED
                    // WRITER
                    uint32_t num_blocks_w_dim_ =
                        bw >= last_num_blocks_w_dim - 1 ? last_num_blocks_w_dim : num_blocks_w_dim;
                    uint32_t out_num_nonzero_subblocks_h_ = out_num_nonzero_subblocks_h;
                    uint32_t out_num_nonzero_subblocks_w_ = out_num_nonzero_subblocks_w;
                    if (bw == num_blocks_w_dim_ - 1) {
                        out_num_nonzero_subblocks_w_ = out_last_num_nonzero_subblocks_w;
                    }
                    uint32_t out_tensor_sbh_start_tile_id = out_tensor_current_w_dim_block_tile_id;
                    for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h_; ++sbh) {
                        uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
                        for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w_; ++sbw) {
                            uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                            uint32_t out_subblock_h_ = out_subblock_h;
                            uint32_t out_subblock_w_ = out_subblock_w;
                            uint32_t subblock_tiles_addr_skip = 0;
                            if (bh == num_blocks_h_dim - 1 && sbh == out_num_nonzero_subblocks_h - 1) {
                                out_subblock_h_ = out_last_subblock_h;
                            }
                            if (bw == num_blocks_w_dim_ - 1 && sbw == out_num_nonzero_subblocks_w_ - 1) {
                                out_subblock_w_ = out_last_subblock_w;
                                subblock_tiles_addr_skip = padded_subblock_tiles_addr_skip;
                            }

                            dfb_out.wait_front(out_subblock_tile_count);
                            uint32_t out_read_offset = 0;

                            for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                                uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                                for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                                    if (bw < num_blocks_w_dim_) {
                                        noc.async_write(
                                            dfb_out,
                                            s,
                                            output_single_tile_size_bytes,
                                            {.offset_bytes = out_read_offset},
                                            {.page_id = out_tensor_tile_id});
                                    }

                                    out_read_offset += output_single_tile_size_bytes;

                                    out_tensor_tile_id += out_tensor_stride_w;
                                }
                                // Skip padded tiles in subblock along row
                                out_read_offset += subblock_tiles_addr_skip;
                                out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                            }

                            noc.async_write_barrier();
                            dfb_out.pop_front(out_subblock_tile_count);
                            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
                        }
                        // Pop fully padded subblocks along the row
                        if (bw == num_blocks_w_dim_ - 1) {
                            dfb_out.wait_front(padded_block_tiles_w_skip);
                            dfb_out.pop_front(padded_block_tiles_w_skip);
                        }
                        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
                    }
                    // Pop row(s) of fully padded subblocks
                    if (bh == num_blocks_h_dim - 1) {
                        dfb_out.wait_front(padded_block_tiles_h_skip);
                        dfb_out.pop_front(padded_block_tiles_h_skip);
                    }

#endif
                    in1_tensor_current_w_dim_block_tile_id += in1_tensor_next_w_dim_block_stride;
                    out_tensor_current_w_dim_block_tile_id += out_tensor_next_w_dim_block_stride;
#ifdef FUSE_BIAS
                    in3_tensor_current_w_dim_block_tile_id += in1_block_w;
#endif
                }
                out_tensor_current_h_dim_block_tile_id += out_tensor_next_h_dim_block_stride;
            }
            out_tensor_start_tile_id += MtNt;
            in1_batch_tile_id += KtNt;
        }
        if constexpr (bcast_B == 0) {
#ifndef IN1_DRAM_HEIGHT_SHARDED
            // For height-sharded DRAM, tile IDs are relative within a batch;
            // batch offset is handled by switching DRAM banks
            in1_tensor_start_tile_id += KtNt;
#endif
        }

        if (fuse_op_reduce_scatter) {
            // Signal reduce_scatter to go
            op_signaler.synchronize_workers_and_signal_op(0);
        }
    }

#if OUT_SHARDED
    dfb_out.wait_front(
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
#endif
#ifdef ENABLE_GLOBAL_CB
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc.async_atomic_barrier();
#endif
    noc.async_write_barrier();
}
