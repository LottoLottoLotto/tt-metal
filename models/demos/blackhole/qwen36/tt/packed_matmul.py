# SPDX-License-Identifier: Apache-2.0
"""Bank-packed single-device M<=32 decode matmuls for any K (35B: K=64 tiles) on the P150 11-wide grid.

The program is the native 1D-mcast matmul descriptor of the stock decode call (same program config, compute kernel,
compile args, CBs and in0 multicast), with only the in1 weight reader replaced by the image's packed-layout reader
(`reader_bmm_tile_layout_qwen_lmhead_packed.cpp`, the one the C++ factory selects for the 27B K=128/160 LM head and
GDN mega projection; its C++ guards do not admit K=64). The weight must be repacked worker-major/bank-rotating
(`repack_for_worker_banks` with full_cols = per_core_N): every worker then streams contiguous K-group runs
from one DRAM bank at a time instead of per_core_N scattered tiles per K row. The compute kernel sees the identical
in1 block contents in the identical order, so outputs are bitwise those of the stock matmul.
"""
import math
import os
from pathlib import Path

import torch
import ttnn

_PACKED_READER = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_qwen_lmhead_packed.cpp"
_STOCK_READER = "reader_bmm_tile_layout_in1_sender_writer_padding.cpp"
_FACTORY = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMcast1DProgramFactory


def repack_for_worker_banks(weight, full_cols=71):
    """Canonical group-K4, eight-bank LM weight, shared by all token counts.

    Worker w (row-major over the 11x10 grid) owns output columns [w*full_cols, ...); its K stages rotate over the
    eight DRAM banks, contiguous within a bank (27B: K=128/160; 35B: K=64; draft heads: any N <= 110*full_cols)."""
    k_tiles = weight.shape[0] // 32
    n_tiles = weight.shape[1] // 32
    workers = -(-n_tiles // full_cols)
    if k_tiles % 32 or not 1 < workers <= 110:
        raise ValueError(f"packed LM-head reader requires K%32==0 and <=110 workers, got {k_tiles}x{n_tiles}/{full_cols}")
    last_cols = n_tiles - (workers - 1) * full_cols
    banks = 8
    group_k_tiles = 4
    pages_per_worker_per_bank = (k_tiles // banks) * full_cols
    target_to_source = torch.empty(k_tiles * n_tiles, dtype=torch.int64)
    for worker in range(workers):
        valid_cols = last_cols if worker == workers - 1 else full_cols
        for group in range(k_tiles // group_k_tiles):
            bank = (worker + group) % banks
            bank_page = (
                worker * pages_per_worker_per_bank
                + (group // banks) * group_k_tiles * valid_cols
            )
            for h in range(group_k_tiles):
                target_bank_page = bank_page + h * valid_cols
                target_pages = bank + banks * torch.arange(target_bank_page, target_bank_page + valid_cols)
                source_page = (group * group_k_tiles + h) * n_tiles + worker * full_cols
                target_to_source[target_pages] = torch.arange(source_page, source_page + valid_cols)

    source_tiles = (
        weight.reshape(k_tiles, 32, n_tiles, 32)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(k_tiles * n_tiles, 32, 32)
    )
    packed_tiles = source_tiles[target_to_source]
    return (
        packed_tiles.reshape(k_tiles, n_tiles, 32, 32)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(weight.shape)
    )


def native_descriptor(x, weight, out, program_config, compute_kernel_config, output_memory_config):
    """The stock 1D-mcast matmul program for out = x @ weight (ttnn.linear decode attributes)."""
    params = ttnn._ttnn.operations.matmul.MatmulParams()
    params.program_config = program_config
    params.compute_kernel_config = compute_kernel_config
    params.output_dtype = ttnn.bfloat16
    params.output_mem_config = output_memory_config
    params.bcast_batch = True
    params.transpose_a = False
    params.transpose_b = False
    params.untilize_out = False
    inputs = ttnn._ttnn.operations.matmul.MatmulInputs()
    inputs.input_tensors = [x, weight]
    inputs.optional_input_tensors = [None]
    inputs.optional_output_tensors = [None]
    return _FACTORY.create_descriptor(params, inputs, [out])


class PackedDecodeMatmul:
    """out[..., rows, N*32] = x[..., rows<=32, K*32] @ packed weight (one tile row), one generic_op per call."""

    def __init__(self, weight, program_config, compute_kernel_config, output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                 cb_slots=2, outstanding=1, dual_noc=False):
        """dual_noc: split every packed weight read across both NoCs (the image reader's QWEN_PACKED_DUAL_NOC path;
        all data-movement kernels then run in dynamic NoC mode, as the C++ factory sets it for the 27B variant)."""
        pc = program_config
        self.k_tiles = int(weight.padded_shape[-2]) // 32
        self.n_tiles = int(weight.padded_shape[-1]) // 32
        if (
            weight.dtype not in (ttnn.bfloat8_b, ttnn.bfloat4_b)
            or weight.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
            or weight.memory_config().buffer_type != ttnn.BufferType.DRAM
            or pc.compute_with_storage_grid_size.x != 11 or pc.compute_with_storage_grid_size.y > 10
            or pc.per_core_M != 1 or not pc.mcast_in0
            or pc.in0_block_w not in (1, 2, 4, 8) or self.k_tiles % pc.in0_block_w
            or self.k_tiles % 32 or not pc.per_core_N < self.n_tiles <= 110 * pc.per_core_N
        ):
            raise ValueError("PackedDecodeMatmul: unsupported weight/program geometry")
        if not 1 <= outstanding <= 4 or outstanding >= cb_slots or cb_slots > 8:
            raise ValueError("PackedDecodeMatmul: need 1 <= outstanding < cb_slots <= 8")
        self.weight = weight
        self.program_config = pc
        self.compute_kernel_config = compute_kernel_config
        self.output_memory_config = output_memory_config
        self.cb_slots = cb_slots
        self.outstanding = outstanding
        self.dual_noc = dual_noc
        self.tile_bytes = 1088 if weight.dtype == ttnn.bfloat8_b else 576

    def _program(self, x, out, compute_kernel_config, output_memory_config):
        desc = native_descriptor(x, self.weight, out, self.program_config, compute_kernel_config, output_memory_config)
        pc = self.program_config
        kernels = list(desc.kernels)
        readers = [k for k in kernels if Path(k.kernel_source).name == _STOCK_READER]
        if len(readers) != 1:
            raise ValueError("PackedDecodeMatmul: native in1 reader not found")
        reader = readers[0]
        defines = [(name, value) for name, value in reader.defines]
        if any(name.startswith("QWEN_") for name, _ in defines):
            raise ValueError("PackedDecodeMatmul: native reader already carries a Qwen packed layout")
        defines += [
            ("QWEN_LMHEAD_PACKED_LAYOUT", "1"),
            ("QWEN_LMHEAD_CB_SLOTS", str(self.cb_slots)),
            ("QWEN_LMHEAD_OUTSTANDING", str(self.outstanding)),
            ("QWEN_PACKED_LAST_WORKER", str((self.n_tiles - 1) // pc.per_core_N)),
            ("QWEN_PACKED_PAGES_PER_WORKER_PER_BANK", str((self.k_tiles // 8) * pc.per_core_N)),
        ]
        if self.outstanding > 1:
            defines.append(("QWEN_LMHEAD_PIPELINED", "1"))
        if self.dual_noc:
            defines.append(("QWEN_PACKED_DUAL_NOC", "1"))
            for kernel in kernels:
                cfg = kernel.config
                if isinstance(cfg, ttnn.DataMovementConfigDescriptor):
                    kernel.config = ttnn.DataMovementConfigDescriptor(
                        processor=cfg.processor, noc=cfg.noc, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC
                    )
                elif not isinstance(cfg, ttnn.ComputeConfigDescriptor):
                    raise ValueError(f"PackedDecodeMatmul: unexpected kernel config {type(cfg).__name__}")
        reader.defines = defines
        reader.kernel_source = _PACKED_READER
        desc.kernels = kernels
        block_bytes = pc.in0_block_w * pc.per_core_N * self.tile_bytes
        cbs = list(desc.cbs)
        found = 0
        for cb in cbs:
            if {f.buffer_index for f in cb.format_descriptors} == {1}:
                if cb.total_size != 2 * block_bytes:
                    raise ValueError(f"PackedDecodeMatmul: unexpected native in1 CB size {cb.total_size}")
                cb.total_size = self.cb_slots * block_bytes
                found += 1
        if found != 1:
            raise ValueError("PackedDecodeMatmul: native in1 CB not found")
        desc.cbs = cbs
        return desc

    def __call__(self, x, output_memory_config=None, compute_kernel_config=None):
        """output_memory_config / compute_kernel_config: per-call overrides of the construction defaults (the stock
        call site's own attributes)."""
        mc = output_memory_config or self.output_memory_config
        ckc = compute_kernel_config or self.compute_kernel_config
        shape = list(x.shape)
        rows = math.prod(shape[:-1])
        if rows > 32 or int(x.padded_shape[-2]) != 32 or math.prod(list(x.padded_shape)[:-2]) != 1 or shape[-1] != self.k_tiles * 32:
            raise ValueError("PackedDecodeMatmul: input must be one tile row of <= 32 rows")
        x4 = ttnn.reshape(x, [1, 1, rows, shape[-1]])
        out = ttnn.empty([1, 1, rows, self.n_tiles * 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=x.device(), memory_config=mc)
        try:
            ttnn.generic_op([x4, self.weight, out], self._program(x4, out, ckc, mc))
        except BaseException:
            ttnn.deallocate(out)
            raise
        return ttnn.reshape(out, shape[:-1] + [self.n_tiles * 32])


def packed_decode_runner(weight, program_config, compute_kernel_config, output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                         cb_slots=2, outstanding=1, dual_noc=True):
    """Decode-only bank-packed copy of an interleaved BFP4/BFP8 weight plus its dual-NoC PackedDecodeMatmul, computing
    exactly what the stock 1D-mcast `program_config` matmul computes: same in0_block_w (the only blocking that shapes
    the accumulation), per_core_N and subblocks, laid out row-major on an 11-wide grid. The original weight stays in
    place for prefill; the copy costs weight-size extra DRAM."""
    pc = program_config
    workers = -(-int(weight.padded_shape[-1]) // 32 // pc.per_core_N)
    grid_y = -(-workers // 11)
    config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(11, grid_y), in0_block_w=pc.in0_block_w, out_subblock_h=pc.out_subblock_h,
        out_subblock_w=pc.out_subblock_w, per_core_M=pc.per_core_M, per_core_N=pc.per_core_N, fuse_batch=True,
        fused_activation=None, mcast_in0=True,
    )
    host = ttnn.to_torch(weight)
    packed = ttnn.from_torch(
        repack_for_worker_banks(host.reshape(host.shape[-2], host.shape[-1]), full_cols=pc.per_core_N).reshape(
            host.shape),
        dtype=weight.dtype, layout=ttnn.TILE_LAYOUT, device=weight.device(), memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return PackedDecodeMatmul(packed, config, compute_kernel_config, output_memory_config=output_memory_config,
                              cb_slots=cb_slots, outstanding=outstanding, dual_noc=dual_noc)


def packed_projections():
    """QWEN36_PACKED_PROJ: comma list of single-request 35B decode projections served by packed_decode_runner
    (out = GDN out_proj, o = attention o_proj, qkv = attention fused QKV)."""
    return {p for p in os.environ.get("QWEN36_PACKED_PROJ", "").split(",") if p}
