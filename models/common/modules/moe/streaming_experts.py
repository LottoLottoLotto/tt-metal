# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM-streamed MoE expert projections for decode (one token, or two for verification).

The selected experts' weights are streamed on the eight DRAM-bank worker cores (one per
bank) with the DeepSeek V3 ``DRAMStreamingExpertsMatmul`` compute. Each bank holds
every expert's [K, N/8] column block, stored column-major in tiles, so an output tile's
K-tile column is one contiguous burst from the core's own bank.

``GateUpSwiGLUStream``: the gate/up projection (34 us vs 54 us for ``sparse_matmul`` on the
Qwen3.6 shape, 9 experts) fused with the routed SwiGLU. Each bank core owns matching gate
and up tiles (``interleave_gate_up_for_banks``), accumulates both in FP32 DEST, and applies
``SiLU(gate) * up * weight`` on the token rows before packing, so gate/up never leave DEST.
It writes the compact, selection-ordered hidden rows the down projection reads.

``DownStream``: the down projection and the expert sum in one op. The selected slots'
hidden rows form one [rows, S*K] input and each output column accumulates the S expert
blocks in FP32, so there are no per-expert outputs and no separate reduction.

With ``rows == 2`` the selection is the two tokens' union (count in entry 31): each
expert's weights stream once and multiply both token rows ([2, 32] input tiles), and a
token's zero routing weight for the other token's experts adds exact zeros.

The down projection reads one contiguous compact hidden row per selected entry.
"""

from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNELS = Path(__file__).resolve().parent / "kernels" / "streaming"
_NOC_MAX_BURST = 16384  # Blackhole


def _bank_cores(device) -> list:
    return device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)


def _bank_args(cores) -> tuple[list, list]:
    """Per-core ``bank_id`` and NOC virtual channel (cores sharing a row get distinct VCs)."""
    bank_ids, vcs = [], []
    for idx, core in enumerate(cores):
        vc = idx & 0x3
        for j in range(idx):
            if cores[j].y == core.y and (j & 0x3) == (idx & 0x3):
                vc = (vc + 1) & 0x3
                break
        bank_ids.append((core, idx))
        vcs.append((core, vc))
    return bank_ids, vcs


def _read_page(block_bytes: int, tile_bytes: int) -> int:
    """Largest whole-tile NOC burst that divides one streamed block."""
    page = (_NOC_MAX_BURST // tile_bytes) * tile_bytes
    while block_bytes % page:
        page -= tile_bytes
    return page


def _brisc_accessors(tensors) -> list:
    args = []
    for tensor in tensors:
        accessor = ttnn.TensorAccessorArgs(tensor)
        if accessor.get_common_runtime_args():
            raise ValueError("streamed expert accessors require compile-time metadata")
        args.extend(accessor.get_compile_time_args())
    return args


def _check_selection(selection: ttnn.Tensor) -> None:
    if (tuple(selection.shape) != (1, 1, 1, 32) or selection.dtype != ttnn.uint16
            or selection.layout != ttnn.ROW_MAJOR_LAYOUT):
        raise ValueError("selection must be row-major UINT16 [1, 1, 1, 32]")


def _cb(core_set, index, fmt, page_size, tiles, tile):
    return ttnn.CBDescriptor(
        total_size=page_size * tiles,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(index, fmt, page_size, tile)],
    )


def _scratch_cb(core_set, index, size):
    return ttnn.CBDescriptor(
        total_size=size,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, size)],
    )


def shuffle_for_banks(weights: torch.Tensor, banks: int) -> torch.Tensor:
    """``[E, K, N]`` -> ``[E*K, N]`` whose WIDTH_SHARDED bank shards are, per expert,
    the [K, N/banks] block with tiles in column-major order (K tiles contiguous)."""
    e, k, n = weights.shape
    kt, per_n_t = k // 32, n // banks // 32
    tiles = weights.reshape(e, kt, 32, banks, per_n_t, 32).permute(3, 0, 4, 1, 2, 5)  # bank, e, nt, kt, r, c
    tiles = tiles.reshape(banks, e * kt, per_n_t, 32, 32).permute(1, 3, 0, 2, 4)  # storage row tile, r, bank, ct, c
    return tiles.reshape(e * k, n).contiguous()


def interleave_gate_up_for_banks(weights: torch.Tensor, banks: int) -> torch.Tensor:
    """``[E, K, 2I]`` gate|up -> ``[E, K, 2I]`` whose bank b column block holds gate tiles
    ``[b*h, b*h + h)`` then up tiles ``[b*h, b*h + h)`` (h = I / 32 / banks), so each bank
    core computes both SwiGLU operands of its intermediate tiles."""
    e, k, n = weights.shape
    h = n // 2 // 32 // banks
    tiles = weights.reshape(e, k, 2, banks, h, 32).permute(0, 1, 3, 2, 4, 5)  # e, k, bank, gate|up, h, c
    return tiles.reshape(e, k, n).contiguous()


def stream_weight_memory_config(device, rows: int, width: int) -> ttnn.MemoryConfig:
    banks = device.dram_grid_size().x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, [rows, width // banks], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _row_tile(rows: int):
    return ttnn.Tile([rows, 32])


class GateUpSwiGLUStream:
    """Selected-expert gate/up fused with the routed SwiGLU.

    ``x[1,1,rows,K]``, the router's selection row and routing weights -> compact hidden
    ``[1, 1, selected, I_t * rows * 32]`` (row-major, selection order, entry s's ``[rows, 32]``
    intermediate tiles back to back), the input ``DownStream`` consumes. ``weights`` must be
    in the bank layout of ``interleave_gate_up_for_banks``.
    """

    def __init__(self, device, weights: ttnn.Tensor, *, stored_experts: int, hidden: int,
                 intermediate: int, selected: int, rows: int = 1) -> None:
        cores = _bank_cores(device)
        banks = device.dram_grid_size().x
        width = 2 * intermediate
        if len(cores) != banks:
            raise ValueError(f"expected one worker per DRAM bank, got {len(cores)} cores for {banks} banks")
        if intermediate % (32 * banks) or hidden % 32:
            raise ValueError("gate and up must each split into whole tiles per bank")
        if tuple(weights.shape)[-2:] != (stored_experts * hidden, width):
            raise ValueError(f"streamed weights must be [{stored_experts * hidden}, {width}], got {tuple(weights.shape)}")
        if rows not in (1, 2):
            raise ValueError(f"streamed experts support one or two token rows, got {rows}")
        self.device = device
        self.weights = weights
        self.hidden = hidden
        self.intermediate = intermediate
        self.selected = selected
        self.rows = rows
        self.cores = cores
        self.core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        self.kt = hidden // 32
        self.per_core_n = width // banks // 32  # gate tiles, then the matching up tiles
        if self.per_core_n > 4:
            raise ValueError(f"{self.per_core_n} gate/up tiles per bank exceed one FP32 half-sync DEST")
        self._program = None
        self._signature = None

    def _build(self, x, selection, weights, out):
        in1_tile = self.weights.get_tile()
        tile_bytes = in1_tile.get_tile_size(self.weights.dtype)
        subblock_k = self.kt  # whole-K columns measured fastest (sk64: 34.0 us)
        block_bytes = subblock_k * tile_bytes
        page = _read_page(block_bytes, tile_bytes)
        half_n = self.per_core_n // 2

        row_tile = ttnn.TileDescriptor(_row_tile(self.rows))
        row_bytes = _row_tile(self.rows).get_tile_size(ttnn.bfloat16)
        index_tile = ttnn.TileDescriptor(ttnn.Tile([1, 32]))
        index_bytes = ttnn.Tile([1, 32]).get_tile_size(ttnn.uint16)
        cs = self.core_set
        cbs = [
            _cb(cs, 0, ttnn.bfloat16, row_bytes, self.kt, row_tile),
            _cb(cs, 1, self.weights.dtype, tile_bytes, 3 * subblock_k, ttnn.TileDescriptor(in1_tile)),
            _cb(cs, 2, ttnn.uint16, index_bytes, 1, index_tile),
            _scratch_cb(cs, 3, self.kt * 128),  # 64 B-aligned x face-row pairs
            _cb(cs, 4, ttnn.bfloat16, row_bytes, 2 * half_n, row_tile),  # double-buffered activated tiles
            _cb(cs, 5, ttnn.bfloat16, 2 * index_bytes, 1, ttnn.TileDescriptor(ttnn.Tile([2, 32]))),
        ]
        common = [
            ("num_tiles_k", self.kt),
            ("per_core_n", self.per_core_n),
            ("selected", self.selected),
            ("subblock_k", subblock_k),
            ("rows", self.rows),
        ]
        bank_ids, vcs = _bank_args(self.cores)
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(_KERNELS / "gate_up_stream_kernel.cpp"),
            core_ranges=cs,
            ncrisc_named_compile_time_args=common + [
                ("in1_addr", self.weights.buffer_address()),
                ("in1_page_size", page),
                ("in1_num_pages", block_bytes // page),
                ("in1_block_bytes", block_bytes),
            ],
            brisc_compile_time_args=_brisc_accessors((x, selection, weights, out)),
            brisc_named_compile_time_args=common + [("inter_tiles", self.intermediate // 32)],
            trisc_named_compile_time_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
            ),
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(named_compile_time_arg="bank_id", core_values=bank_ids, other_value=0),
                PerCoreCompileTimeDescriptor(named_compile_time_arg="vc", core_values=vcs, other_value=0),
            ],
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=[
                    (c, [x.buffer_address(), selection.buffer_address(), weights.buffer_address(), out.buffer_address()])
                    for c in self.cores
                ]
            ),
        )
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, x: ttnn.Tensor, selection: ttnn.Tensor, weights: ttnn.Tensor) -> ttnn.Tensor:
        if (tuple(x.shape) != (1, 1, self.rows, self.hidden) or x.layout != ttnn.TILE_LAYOUT
                or x.dtype != ttnn.bfloat16):
            raise ValueError(f"x must be BF16 tiles [1, 1, {self.rows}, {self.hidden}]")
        if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError("x must be interleaved")
        _check_selection(selection)
        if (tuple(weights.shape) != (1, 1, self.rows, 32) or weights.dtype != ttnn.bfloat16
                or weights.layout != ttnn.ROW_MAJOR_LAYOUT):
            raise ValueError(f"routing weights must be row-major BF16 [1, 1, {self.rows}, 32]")
        out = ttnn.empty(
            [1, 1, self.selected, self.intermediate * self.rows],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tensors = [x, selection, weights, out, self.weights]
        signature = tuple((int(t.buffer_address()), str(t.memory_config())) for t in tensors)
        if self._program is None or signature != self._signature:
            self._program = self._build(x, selection, weights, out)
            self._signature = signature
        ttnn.generic_op(tensors, self._program)
        return out


class DownStream:
    """Down projection plus expert sum: compact hidden -> ``[1, 1, rows, N]``.

    ``hidden`` is ``GateUpSwiGLUStream``'s compact output, row-major
    ``[1, 1, selected, K_t * rows * 32]`` in selection order (entry s's ``[rows, 32]`` tiles
    back to back). ``out[t] = sum_s hidden[s][t] @ W[e_s]`` over the selection row's slots;
    the routing weights are already folded into ``hidden`` by the fused SwiGLU.
    """

    def __init__(self, device, weights: ttnn.Tensor, *, stored_experts: int, intermediate: int,
                 hidden: int, selected: int, rows: int = 1) -> None:
        cores = _bank_cores(device)
        banks = device.dram_grid_size().x
        if len(cores) != banks:
            raise ValueError(f"expected one worker per DRAM bank, got {len(cores)} cores for {banks} banks")
        if hidden % (32 * banks) or intermediate % 32:
            raise ValueError("down width must split into whole tiles per bank")
        if tuple(weights.shape)[-2:] != (stored_experts * intermediate, hidden):
            raise ValueError(
                f"streamed weights must be [{stored_experts * intermediate}, {hidden}], got {tuple(weights.shape)}"
            )
        if rows not in (1, 2):
            raise ValueError(f"streamed experts support one or two token rows, got {rows}")
        self.device = device
        self.weights = weights
        self.intermediate = intermediate
        self.hidden = hidden
        self.selected = selected
        self.rows = rows
        self.cores = cores
        self.core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        self.kt = intermediate // 32
        self.per_core_n = hidden // banks // 32
        # One expert column is only 9 KB (BFP4, K=512): a deeper read ring keeps enough
        # bytes in flight. Traced layer 5: 3 -> 27.5 us, 5 -> 21.8, 8 -> 21.7, 15 -> 22.2.
        self.buffers = 6
        self._program = None
        self._signature = None

    def _build(self, h, selection, out):
        in1_tile = self.weights.get_tile()
        tile_bytes = in1_tile.get_tile_size(self.weights.dtype)
        block_bytes = self.kt * tile_bytes  # one expert's K column: one contiguous burst
        page = _read_page(block_bytes, tile_bytes)
        subblock_w = max(w for w in range(1, 5) if self.per_core_n % w == 0)
        in_tiles = self.selected * self.kt

        row_tile = ttnn.TileDescriptor(_row_tile(self.rows))
        row_bytes = _row_tile(self.rows).get_tile_size(ttnn.bfloat16)
        index_tile = ttnn.TileDescriptor(ttnn.Tile([1, 32]))
        index_bytes = ttnn.Tile([1, 32]).get_tile_size(ttnn.uint16)
        cs = self.core_set
        cbs = [
            _cb(cs, 0, ttnn.bfloat16, row_bytes, in_tiles, row_tile),
            _cb(cs, 1, self.weights.dtype, tile_bytes, self.buffers * self.kt, ttnn.TileDescriptor(in1_tile)),
            _cb(cs, 2, ttnn.uint16, index_bytes, 1, index_tile),
            _scratch_cb(cs, 3, 4 * 2048),
            _cb(cs, 4, ttnn.bfloat16, row_bytes, self.per_core_n, row_tile),
        ]
        common = [
            ("k_tiles", self.kt), ("per_core_n", self.per_core_n), ("selected", self.selected), ("rows", self.rows)
        ]
        bank_ids, vcs = _bank_args(self.cores)
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(_KERNELS / "down_stream_kernel.cpp"),
            core_ranges=cs,
            ncrisc_named_compile_time_args=common + [
                ("in1_addr", self.weights.buffer_address()),
                ("in1_page_size", page),
                ("in1_num_pages", block_bytes // page),
                ("in1_block_bytes", block_bytes),
                ("in1_buffers", self.buffers),
            ],
            brisc_compile_time_args=_brisc_accessors((h, selection, out)),
            brisc_named_compile_time_args=common,
            trisc_named_compile_time_args=common + [("subblock_w", subblock_w)],
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
            ),
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(named_compile_time_arg="bank_id", core_values=bank_ids, other_value=0),
                PerCoreCompileTimeDescriptor(named_compile_time_arg="vc", core_values=vcs, other_value=0),
            ],
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=[(c, [h.buffer_address(), selection.buffer_address(), out.buffer_address()]) for c in self.cores]
            ),
        )
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])

    def __call__(self, h: ttnn.Tensor, selection: ttnn.Tensor) -> ttnn.Tensor:
        width = self.intermediate * self.rows
        if (tuple(h.shape) != (1, 1, self.selected, width) or h.layout != ttnn.ROW_MAJOR_LAYOUT
                or h.dtype != ttnn.bfloat16):
            raise ValueError(f"hidden must be compact BF16 row-major [1, 1, {self.selected}, {width}]")
        if h.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError("hidden must be interleaved")
        _check_selection(selection)
        out = ttnn.empty(
            [1, 1, self.rows, self.hidden],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tensors = [h, selection, out, self.weights]
        signature = tuple((int(t.buffer_address()), str(t.memory_config())) for t in tensors)
        if self._program is None or signature != self._signature:
            self._program = self._build(h, selection, out)
            self._signature = signature
        ttnn.generic_op(tensors, self._program)
        return out
