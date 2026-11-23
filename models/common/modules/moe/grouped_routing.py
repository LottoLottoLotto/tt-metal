# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-only prefill routing, expert-major tile packing, and FP32 expert reduction.

Only the four returned tensors belong to the caller. Route records and the group map
are invocation-local DRAM scratch, released after packing; L1 scratch belongs to each
program. Capacity is a host-computable upper bound, not a device-to-host count read.

Masked prefill (``valid_rows``, see ``prefill_valid_rows``) leaves padding rows out of
routing and grouping entirely: they cost no expert work and combine to zero rows.
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from models.common.modules.moe.selected_experts import _CachedProgram, _accessor_args, _cb
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNELS = Path(__file__).resolve().parent / "kernels"


def _workers(device, work):
    grid = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(min(work, grid.x * grid.y))]
    return cores, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _empty(device, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.empty(shape, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _data_program(name, tensors, constants, cores, core_set, scratch_bytes, extra=()):
    args = list(constants)
    for tensor in tensors:
        args.extend(_accessor_args(tensor))
    addresses = [tensor.buffer_address() for tensor in tensors]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / name),
        core_ranges=core_set,
        compile_time_args=args,
        runtime_args=[(core, addresses + [i, len(cores), *extra]) for i, core in enumerate(cores)],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(
        kernels=[kernel], cbs=[_cb(0, (scratch_bytes + 2047) // 2048, core_set)], semaphores=[]
    )


def _check_tiles(tensor, name, dtype=ttnn.bfloat16):
    if (tensor.dtype != dtype or tensor.layout != ttnn.TILE_LAYOUT
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or tuple(tensor.shape)[:2] != (1, 1)
            or tuple(tensor.get_tile().tile_shape) != (32, 32)):
        formats = "FLOAT32" if dtype == ttnn.float32 else "BF16"
        raise ValueError(f"{name} must be interleaved DRAM {formats} 32x32 tiles [1, 1, rows, width]")


def prefill_valid_rows(device, lengths, segment_rows: int) -> ttnn.Tensor:
    """Device row mask for masked prefill: segment s owns rows [s*segment_rows,
    (s+1)*segment_rows) and only its first lengths[s] rows are real. Returns an
    interleaved DRAM UINT32 row-major [1,1,1,32] page [segment_rows, segments, lengths...].
    Build it once per forward (it is a host write, so outside any trace capture) and pass
    it to every MoE layer; the caller owns and deallocates it."""
    lengths = [int(length) for length in lengths]
    if not 1 <= len(lengths) <= 30 or segment_rows < 1 or not all(0 <= n <= segment_rows for n in lengths):
        raise ValueError("need 1..30 segments with 0 <= length <= segment_rows")
    page = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
    page[..., 0], page[..., 1] = segment_rows, len(lengths)
    page[..., 2 : 2 + len(lengths)] = torch.tensor(lengths, dtype=torch.int32)
    return ttnn.from_torch(page, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _pack_program(device, tensors, groups, hidden, expert_width):
    """Group gather: whole-row token reads, compute tilize, tile writes (one group per step)."""
    cores, cs = _workers(device, groups)
    width_tiles = hidden // 32
    args = []
    for tensor in tensors:
        args.extend(_accessor_args(tensor))
    addresses = [tensor.buffer_address() for tensor in tensors]
    runtime = [(c, addresses + [i, len(cores)]) for i, c in enumerate(cores)]
    named = [("width_tiles", width_tiles), ("expert_width", expert_width)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=str(_KERNELS / "grouped_routing_pack.cpp"),
        core_ranges=cs,
        brisc_compile_time_args=args,
        ncrisc_compile_time_args=args,
        brisc_named_compile_time_args=named,
        ncrisc_named_compile_time_args=named,
        trisc_named_compile_time_args=named,
        trisc_compute_config=ttnn.ComputeConfigDescriptor(),
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            brisc_args=runtime, ncrisc_args=runtime, trisc_args=runtime),
    )
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))

    def buffer(index, pages):
        return ttnn.CBDescriptor(
            total_size=pages * 2048, core_ranges=cs,
            format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, 2048, tile)])

    # Rows are double-buffered so the next group's reads overlap this group's tilize.
    cbs = [buffer(0, 2 * width_tiles), buffer(1, width_tiles), _cb(2, 1, cs), _cb(3, 1, cs), _cb(4, 2, cs)]
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])


class GroupedRouting:
    """Route all tokens, then gather into 32-token groups in ascending expert order.

    ``x`` is [1,1,T,H], ``logits`` is [1,1,T,E+32]. Returns ``x`` [1,1,G*32,H],
    ``weights`` [1,1,G*32,32] (weight repeated across columns), ``experts`` UINT32
    row-major [1,1,1,ceil((G+1)/32)*32] (count followed by one expert ID per group),
    and ``inverse`` UINT32 row-major [1,1,T,32] (top-k ascending, shared last).
    G=min(T*top_k+ceil(T/32), ceil(T*(top_k+1)/32)+E+1). Unused capacity past the
    actual count is unspecified; padded rows in actual groups are zero. T >= 1.
    With ``valid_rows``, padding rows are not routed or grouped and their ``inverse``
    entries are 0xffffffff (combined as zero rows); real rows are unchanged.
    """

    def __init__(self, device, num_experts: int, top_k: int, shared_slot: int):
        if num_experts <= 0 or num_experts % 32 or not 1 <= top_k <= min(num_experts, 31):
            raise ValueError("need tile-aligned experts and 1 <= top_k <= min(num_experts, 31)")
        if not num_experts <= shared_slot < 65536:
            raise ValueError("shared_slot must be a UINT16 slot beyond the routed experts")
        self.device = device
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_slot = shared_slot
        self._shape = None

    def __call__(self, x, logits, valid_rows=None) -> dict:
        _check_tiles(x, "x")
        _check_tiles(logits, "logits")
        tokens, hidden = tuple(x.shape)[-2:]
        if tokens < 1 or hidden % 32 or tuple(logits.shape) != (1, 1, tokens, self.num_experts + 32):
            raise ValueError("need T >= 1, tile-aligned H, and logits [1,1,T,num_experts+32]")
        masked = valid_rows is not None
        if masked and (valid_rows.dtype != ttnn.uint32 or valid_rows.layout != ttnn.ROW_MAJOR_LAYOUT
                       or valid_rows.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                       or tuple(valid_rows.shape) != (1, 1, 1, 32)):
            raise ValueError("valid_rows must come from prefill_valid_rows")
        if self._shape != (tokens, hidden):
            # Logical tails sharing the same padded shape still change loop bounds.
            self._route = {False: _CachedProgram(), True: _CachedProgram()}
            self._prefix = {False: _CachedProgram(), True: _CachedProgram()}
            self._pack = _CachedProgram()
            self._shape = (tokens, hidden)
        # At small M there cannot be more groups than route records; the shared
        # expert contributes only ceil(T/32) groups rather than one per token.
        groups = min(
            tokens * self.top_k + (tokens + 31) // 32,
            (tokens * (self.top_k + 1) + 31) // 32 + self.num_experts + 1,
        )
        expert_width = ((groups + 32) // 32) * 32
        packed = _empty(self.device, [1, 1, groups * 32, hidden])
        weights = _empty(self.device, [1, 1, groups * 32, 32])
        experts = _empty(self.device, [1, 1, 1, expert_width], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        inverse = _empty(self.device, [1, 1, tokens, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        # A route word is (BF16 weight << 16) | expert ID. A map page contains
        # 32 source token IDs then 32 BF16 weights stored in UINT32 words.
        routes = _empty(self.device, [1, 1, tokens, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        group_map = _empty(self.device, [1, 1, groups, 64], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        # The mask page travels as a runtime address (and in the cache signature), so both
        # variants keep one accessor layout; the masked one reserves room for the page.
        mask = [valid_rows] if masked else []
        extra = [valid_rows.buffer_address()] if masked else []
        cores, cs = _workers(self.device, (tokens + 31) // 32)
        route_tensors = [logits, routes]
        self._route[masked].run([logits, *mask, routes], lambda: _data_program(
            "grouped_routing.cpp", route_tensors,
            [self.num_experts, self.top_k, self.shared_slot, tokens, int(masked)], cores, cs,
            (self.num_experts // 32 + 1) * 2048 + (self.num_experts + 32) * 2 + 256, extra))
        cores, cs = _workers(self.device, 1)
        prefix_tensors = [routes, group_map, experts, inverse]
        self._prefix[masked].run([routes, *mask, group_map, experts, inverse], lambda: _data_program(
            "grouped_routing_prefix.cpp", prefix_tensors,
            [self.num_experts, self.top_k, self.shared_slot, tokens, expert_width, int(masked)], cores, cs,
            (self.num_experts + 1) * (256 + 3 * 4) + 2 * 4096 + 128 + 192, extra))
        # Row-major tokens: the pack reads one contiguous row per token and tilizes on device.
        rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pack_tensors = [rows, group_map, experts, packed, weights]
        self._pack.run(pack_tensors, lambda: _pack_program(self.device, pack_tensors, groups, hidden, expert_width))
        ttnn.deallocate(rows)
        ttnn.deallocate(routes)
        ttnn.deallocate(group_map)
        return {"x": packed, "weights": weights, "experts": experts, "inverse": inverse}


def grouped_combine(packed_output, inverse, top_k: int = 8):
    """Gather FLOAT32 top_k routed outputs plus the shared output.

    Accumulates in FP32 and rounds to BF16 once, returning a newly allocated
    [1,1,T,H] interleaved DRAM tensor. Both arguments remain caller-owned.
    Padded rows in the last output tile are explicitly zero.
    """
    _check_tiles(packed_output, "packed_output", dtype=ttnn.float32)
    if (not 1 <= top_k <= 31 or inverse.dtype != ttnn.uint32 or inverse.layout != ttnn.ROW_MAJOR_LAYOUT
            or inverse.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or tuple(inverse.shape)[:2] != (1, 1) or inverse.shape[-1] != 32):
        raise ValueError("inverse must be interleaved DRAM UINT32 [1,1,T,32], 1 <= top_k <= 31")
    device = packed_output.device()
    tokens, hidden = inverse.shape[-2], packed_output.shape[-1]
    if tokens < 1 or hidden % 32 or packed_output.shape[-2] % 32:
        raise ValueError("need T >= 1 and whole packed-output tiles")
    out = _empty(device, [1, 1, tokens, hidden])
    # Row-major projections: each token-slot row segment is one contiguous read.
    rows = ttnn.to_layout(packed_output, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    width_tiles = hidden // 32
    chunk = next(c for c in (4, 2, 1) if width_tiles % c == 0)
    items = ((tokens + 31) // 32) * (width_tiles // chunk)
    cores, cs = _workers(device, items)
    tensors = [rows, inverse, out]
    addresses = [t.buffer_address() for t in tensors]
    args = []
    for tensor in tensors:
        args.extend(_accessor_args(tensor))
    runtime = [(c, addresses + [i, len(cores)]) for i, c in enumerate(cores)]
    named_args = [
        ("tokens", tokens), ("width_tiles", width_tiles), ("selected", top_k + 1), ("chunk", chunk),
    ]
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
    # FP32 destination accumulation alone still allows TF32 input unpacking.
    # CB0 must bypass SrcA and unpack directly to DST at full precision.
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    unpack[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode.extend(unpack)
    kernel = UnifiedKernelDescriptor(
        kernel_source=str(_KERNELS / "grouped_routing_combine.cpp"),
        core_ranges=cs,
        brisc_compile_time_args=args,
        ncrisc_compile_time_args=args,
        brisc_named_compile_time_args=named_args,
        ncrisc_named_compile_time_args=named_args,
        trisc_named_compile_time_args=named_args,
        trisc_compute_config=compute,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            brisc_args=runtime, ncrisc_args=runtime, trisc_args=runtime),
    )
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))

    def buffer(index, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=cs,
            format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

    selected = top_k + 1
    program = ttnn.ProgramDescriptor(
        kernels=kernel.get_kernel_descriptors().kernels,
        cbs=[buffer(0, ttnn.float32, 4096, 2 * selected * chunk), _cb(1, 2, cs),
             buffer(2, ttnn.bfloat16, 2048, chunk), buffer(16, ttnn.bfloat16, 2048, 2 * chunk)],
        semaphores=[])
    ttnn.generic_op(tensors, program)
    ttnn.deallocate(rows)
    return out
