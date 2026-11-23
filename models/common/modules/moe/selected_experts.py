# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace-compatible sparse MoE routing for single-stream and concurrent decode.

``TopKRouting``: top-k selection and routing weights as compact ``[1, 32]`` rows
(``selection``: UINT16 slot ids, ``weights``: BF16 routing weights, same order). With two
token rows the selection is the union of both tokens' slots (count in entry 31) and
``weights`` holds one row per token, zero where that token did not select a slot.

``BatchedTopKRouting``: explicitly concurrent 1/2/4/8/16-row routing, with independent
row workers and a device-only union merge. Its variable-width metadata is consumed by
``BatchedExperts``; it does not change the legacy one/two-row speculative contract.

``ParallelTopKRouting``: ``BatchedTopKRouting``'s exact output for 1-2 decode rows with
the top-k scan split over one core per logit tile.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

_KERNELS = Path(__file__).resolve().parent / "kernels"
_TILE_BYTES = 2048


def _accessor_args(tensor: ttnn.Tensor) -> list[int]:
    accessor = ttnn.TensorAccessorArgs(tensor)
    if accessor.get_common_runtime_args():
        raise ValueError("selected-expert accessors require compile-time metadata")
    return list(accessor.get_compile_time_args())


def _cb(index: int, tiles: int, cores: ttnn.CoreRangeSet) -> ttnn.CBDescriptor:
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    return ttnn.CBDescriptor(
        total_size=tiles * _TILE_BYTES,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, _TILE_BYTES, tile)],
    )


def _row_ranges(cores) -> ttnn.CoreRangeSet:
    """Cores as row-contiguous CoreRanges: program dispatch cost grows with the range count."""
    ranges = []
    for core in sorted(cores, key=lambda c: (c.y, c.x)):
        if ranges and ranges[-1][1] == core.y and ranges[-1][2] + 1 == core.x:
            ranges[-1][2] = core.x
        else:
            ranges.append([core.x, core.y, core.x])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y), ttnn.CoreCoord(x1, y)) for x0, y, x1 in ranges]
    )


class _CachedProgram:
    """Rebuild a program descriptor only when its tensors' storage changes."""

    def __init__(self) -> None:
        self.program = None
        self.signature = None

    def run(self, tensors: list[ttnn.Tensor], build) -> None:
        signature = tuple(
            (int(t.buffer_address()), tuple(t.padded_shape), str(t.memory_config())) for t in tensors
        )
        if self.program is None or signature != self.signature:
            self.program = build()
            self.signature = signature
        ttnn.generic_op(tensors, self.program)


class TopKRouting:
    """Router tail: BF16 tile logits -> compact ``(selection, weights)`` rows.

    One data-movement core selects the top_k of the first ``num_experts`` logits (ties to
    the lower expert id) and takes the softmax over them in fixed point, replacing topk,
    softmax, untilize and scatter (the topk alone is a 45 us single-core op). With
    ``shared_slot``, logit column ``num_experts`` is the shared expert's gate: that slot
    is always selected (last) with weight ``sigmoid(gate)``. With ``rows == 2`` the two
    tokens' selections merge into one ascending union (count in entry 31, remaining
    entries padded with the unused slots above the shared slot) and ``weights`` has one
    row per token.
    """

    def __init__(self, device, num_experts: int, top_k: int, shared_slot: int | None = None, rows: int = 1) -> None:
        if rows not in (1, 2):
            raise ValueError(f"router tail supports one or two token rows, got {rows}")
        if rows == 2 and shared_slot is None:
            raise ValueError("two-row routing pads the union with the slots above the shared slot")
        selected = rows * top_k + (shared_slot is not None)
        if num_experts % ttnn.TILE_SIZE or not 1 <= top_k or selected > ttnn.TILE_SIZE - (rows > 1):
            raise ValueError(f"need tile-aligned experts and at most 32 selected slots, got {num_experts}, {top_k}")
        if shared_slot is not None and shared_slot < num_experts:
            raise ValueError(f"shared slot {shared_slot} must not alias a routed expert (< {num_experts})")
        self.device = device
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_slot = shared_slot
        self.rows = rows
        self.logit_width = num_experts + (ttnn.TILE_SIZE if shared_slot is not None else 0)
        self.core = ttnn.CoreCoord(0, 0)
        self.core_set = ttnn.CoreRangeSet({ttnn.CoreRange(self.core, self.core)})
        self._cache = _CachedProgram()

    def _build(self, logits, selection, weights):
        compile_time_args = (
            [self.num_experts, self.top_k, self.shared_slot or 0, self.rows]
            + _accessor_args(logits)
            + _accessor_args(selection)
            + _accessor_args(weights)
        )
        runtime_args = [(self.core, [logits.buffer_address(), selection.buffer_address(), weights.buffer_address()])]
        reader = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "topk_routing.cpp"),
            core_ranges=self.core_set,
            compile_time_args=compile_time_args,
            runtime_args=runtime_args,
            config=ttnn.ReaderConfigDescriptor(),
        )
        # Scratch: each token's logits row (64 B per logit tile), the id row and the weight rows.
        scratch_bytes = self.rows * (self.logit_width // ttnn.TILE_SIZE) * 64 + 64 * (1 + self.rows)
        scratch_tiles = (scratch_bytes + _TILE_BYTES - 1) // _TILE_BYTES
        kernels, cbs = [reader], [_cb(0, scratch_tiles, self.core_set)]
        if self.rows == 2:
            # The other data-movement RISC routes row 1 concurrently (scalar top-k and
            # fixed-point softmax dominate) and hands the result over in CB 1.
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=str(_KERNELS / "topk_routing.cpp"),
                core_ranges=self.core_set,
                compile_time_args=compile_time_args,
                runtime_args=runtime_args,
                defines=[("ROUTING_HELPER", "1")],
                config=ttnn.WriterConfigDescriptor(),
            ))
            cbs.append(_cb(1, 1, self.core_set))
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])

    def __call__(self, logits: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Returns ``(selection, weights)``: the row-major ``[1, 1, 1, 32]`` row of the selected
        slot ids (UINT16, routed ascending, shared slot last) and the ``[1, 1, rows, 32]``
        routing weights (BF16), zero padded."""
        if (
            tuple(logits.shape) != (1, 1, self.rows, self.logit_width)
            or logits.dtype != ttnn.bfloat16
            or logits.layout != ttnn.TILE_LAYOUT
            or logits.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
        ):
            raise ValueError(f"logits must be interleaved BF16 tiles [1, 1, {self.rows}, {self.logit_width}]")

        def row(dtype, rows):
            return ttnn.empty(
                [1, 1, rows, ttnn.TILE_SIZE],
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        selection, weights = row(ttnn.uint16, 1), row(ttnn.bfloat16, self.rows)
        tensors = [logits, selection, weights]
        self._cache.run(tensors, lambda: self._build(*tensors))
        return selection, weights


class BatchedTopKRouting:
    """Concurrent decode routing for 1/2/4/8/16 independent rows.

    One program: each row routes on its own core and posts its record to row 0's core,
    which merges the union once every row has arrived. Metadata is UINT16
    [1,1,1,width], with the runtime union count at width-1; routing weights are BF16
    [1,1,rows,width], zero for unselected experts. width is the next multiple of 32
    containing rows*top_k+1 expert ids and the count. No host count read is needed.
    """

    def __init__(self, device, num_experts, top_k, shared_slot, rows):
        if rows not in (1, 2, 4, 8, 16):
            raise ValueError("batched routing requires a 1/2/4/8/16-row bucket")
        if num_experts % 32 or not 1 <= top_k <= min(num_experts, 31):
            raise ValueError("need tile-aligned experts and 1 <= top_k <= min(experts, 31)")
        if not num_experts <= shared_slot < 65536:
            raise ValueError("shared slot must be a UINT16 id beyond the routed experts")
        self.device = device
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_slot = shared_slot
        self.rows = rows
        self.width = ((rows * top_k + 2 + 31) // 32) * 32
        grid = device.compute_with_storage_grid_size()
        if grid.x * grid.y < rows:
            raise ValueError("batched routing needs one worker per token")
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(rows)]
        self._merge_core = device.worker_core_from_logical_core(self.cores[0])
        self._cache = _CachedProgram()

    def _build(self, tensors):
        cs = _row_ranges(self.cores)
        args = [self.num_experts, self.top_k, self.shared_slot, self.rows, self.width]
        for tensor in tensors:
            args.extend(_accessor_args(tensor))
        addresses = [tensor.buffer_address() for tensor in tensors]
        merge = [self._merge_core.x, self._merge_core.y]
        # Own logits row and record, then row 0's record inbox, union ids, weight rows and
        # expert positions (layout mirrored in batched_routing.cpp).
        scratch = ((self.num_experts // 32 + 1) * 64 + 128 + self.rows * 128
                   + (self.rows + 1) * self.width * 2 + self.num_experts)
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "batched_routing.cpp"), core_ranges=cs,
            compile_time_args=args,
            runtime_args=[(core, addresses + [row] + merge) for row, core in enumerate(self.cores)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(
            kernels=[kernel], cbs=[_cb(0, (scratch + 2047) // 2048, cs)],
            semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=cs, initial_value=0)],
        )

    def __call__(self, logits):
        if (tuple(logits.shape) != (1, 1, self.rows, self.num_experts + 32)
                or logits.dtype != ttnn.bfloat16 or logits.layout != ttnn.TILE_LAYOUT
                or logits.memory_config() != ttnn.L1_MEMORY_CONFIG):
            raise ValueError("batched logits must be interleaved L1 BF16 tiles [1,1,B,E+32]")

        def empty(rows, width, dtype):
            return ttnn.empty(
                [1, 1, rows, width], dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        selection = empty(1, self.width, ttnn.uint16)
        weights = empty(self.rows, self.width, ttnn.bfloat16)
        tensors = [logits, selection, weights]
        self._cache.run(tensors, lambda: self._build(tensors))
        return selection, weights


class ParallelTopKRouting:
    """``BatchedTopKRouting``'s exact (selection, weights) for 1-2 decode rows, with the
    top-k scan split over one core per 32-logit tile and row (cores ``(tile, row)``).

    Each scan core posts its tile's top-k survivors to its row's finisher (tile 0), which
    replays them through the same top-k (identical list by construction), runs the same
    fixed-point softmax / shared gate and posts the token record to row 0's finisher for
    the ascending union. Same output layout as ``BatchedTopKRouting``.
    """

    def __init__(self, device, num_experts, top_k, shared_slot, rows):
        if rows not in (1, 2):
            raise ValueError("parallel routing supports one or two token rows")
        if num_experts % 32 or not 1 <= top_k <= min(num_experts, 31):
            raise ValueError("need tile-aligned experts and 1 <= top_k <= min(experts, 31)")
        if not num_experts <= shared_slot < 65536:
            raise ValueError("shared slot must be a UINT16 id beyond the routed experts")
        grid = device.compute_with_storage_grid_size()
        self.tiles = num_experts // 32
        if self.tiles > grid.x or rows > grid.y:
            raise ValueError("parallel routing needs one core per logit tile and row")
        self.device = device
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_slot = shared_slot
        self.rows = rows
        self.width = ((rows * top_k + 2 + 31) // 32) * 32
        self.cores = [ttnn.CoreCoord(t, r) for r in range(rows) for t in range(self.tiles)]
        self.core_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.tiles - 1, rows - 1))}
        )
        self._finishers = [device.worker_core_from_logical_core(ttnn.CoreCoord(0, r)) for r in range(rows)]
        self._cache = _CachedProgram()

    def _build(self, tensors):
        args = [self.num_experts, self.top_k, self.shared_slot, self.rows, self.width]
        for tensor in tensors:
            args.extend(_accessor_args(tensor))
        addresses = [tensor.buffer_address() for tensor in tensors]
        union = self._finishers[0]
        # Layout mirrored in parallel_routing.cpp.
        align = lambda n: (n + 15) // 16 * 16  # noqa: E731
        record, token = align(self.top_k * 4), align((self.top_k + 1) * 4)
        scratch = (64 + record + self.tiles * record + align((self.num_experts + 32) * 2) + token
                   + self.rows * token + (self.rows + 1) * self.width * 2 + self.num_experts)
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "parallel_routing.cpp"), core_ranges=self.core_set,
            compile_time_args=args,
            runtime_args=[
                (c, addresses + [c.x, c.y, self._finishers[c.y].x, self._finishers[c.y].y, union.x, union.y])
                for c in self.cores
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(
            kernels=[kernel], cbs=[_cb(0, (scratch + _TILE_BYTES - 1) // _TILE_BYTES, self.core_set)],
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=self.core_set, initial_value=0) for i in (0, 1)],
        )

    def __call__(self, logits):
        if (tuple(logits.shape) != (1, 1, self.rows, self.num_experts + 32)
                or logits.dtype != ttnn.bfloat16 or logits.layout != ttnn.TILE_LAYOUT
                or logits.memory_config() != ttnn.L1_MEMORY_CONFIG):
            raise ValueError("parallel routing logits must be interleaved L1 BF16 tiles [1,1,R,E+32]")

        def empty(rows, dtype):
            return ttnn.empty([1, 1, rows, self.width], dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

        selection, weights = empty(1, ttnn.uint16), empty(self.rows, ttnn.bfloat16)
        tensors = [logits, selection, weights]
        self._cache.run(tensors, lambda: self._build(tensors))
        return selection, weights
