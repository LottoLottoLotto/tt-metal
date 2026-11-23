# SPDX-License-Identifier: Apache-2.0
"""Fixed-address committed DFlash KV; speculative K/V never touch the ring."""
from pathlib import Path

import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("dflash_cache_kernel.cpp").resolve())
DRAM = ttnn.DRAM_MEMORY_CONFIG


def device_scalar(tensor, name):
    if (tuple(tensor.shape) != (1, 1)
            or tensor.dtype not in (ttnn.int32, ttnn.uint32)
            or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
            or tensor.memory_config() != DRAM):
        raise ValueError(f"DFlash {name} must be row-major DRAM int32/uint32 [1,1]")


def _launch(device, tensors, mode, rows, lane=0, slots=0):
    # Mode 1 (draft prepare) builds one mask column tile per worker: 65 workers.
    workers = 32 if mode in (2, 6, 7) else 65 if mode == 1 else 1
    grid = device.compute_with_storage_grid_size()
    coordinates = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in coordinates})
    tasks = [(core, [i]) for i, core in enumerate(coordinates)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        brisc_named_compile_time_args=[("mode", mode), ("rows", rows), ("lane", lane), ("slots", slots)],
        brisc_common_runtime_args=[int(t.buffer_address()) for t in tensors],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=False, dst_full_sync_en=False,
        ),
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=tasks, brisc_args=tasks, trisc_args=tasks,
        ),
    )
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    program = ttnn.ProgramDescriptor(
        kernels=kernel.get_kernel_descriptors().kernels,
        cbs=[ttnn.CBDescriptor(
            total_size=2048 if mode in (2, 6, 7) else 12288,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048, tile)],
        )],
        semaphores=[],
    )
    ttnn.generic_op(tensors, program)


class CommittedKV:
    """Single-sequence ring with absolute positions, shared by all five layers.

    A valid append is a contiguous prefix [start,start+count), count <= input rows.
    For draft, start+count <= anchor position. The scheduler owns these values;
    they are read only by the device kernel. Future/stale owners are masked even
    when a caller intentionally rewinds the anchor. Reset is required on reuse
    for another request. No speculative append/eviction/rollback is performed.
    """
    window = 2048

    def __init__(self, device, block_size, mask_rows=None):
        """mask_rows: query rows of the draft mask (default block_size; 32 for a ring drafted in a multi-user
        block, prepare(lane=u) then masks every row but the user's)."""
        self.device = device
        self.block_size = block_size
        self._owned = []
        self.owners = self._empty((1, 2048), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.feature_positions = self._empty((1, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.noise_positions = self._empty((1, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.token_ids = self._empty((1, block_size), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.mask = self._empty((1, 1, mask_rows or block_size, 2080), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # Zero initialization is essential: masked NaN values could still poison
        # SDPA's probability/value multiplication in never-owned ring slots.
        self.layers = []
        for _ in range(5):
            # window committed rows + one 32-row tail tile row for the draft's noise K/V.
            pair = tuple(ttnn.zeros(
                (1, 8, 2080, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=DRAM,
            ) for _ in range(2))
            self._owned.extend(pair)
            self.layers.append(pair)
        self.reset()

    def _empty(self, shape, dtype, layout):
        tensor = ttnn.empty(shape, dtype=dtype, layout=layout, device=self.device, memory_config=DRAM)
        self._owned.append(tensor)
        return tensor

    def reset(self):
        # generic_op requires an input and output even for an in-place kernel.
        _launch(self.device, [self.owners, self.owners], 0, self.block_size)

    def prepare(self, anchor, position, start, count, *, prefill=False, lane=0):
        for name, tensor in (("anchor", anchor), ("position", position), ("feature_start", start), ("feature_count", count)):
            device_scalar(tensor, name)
        tensors = [self.owners, position, start, count, anchor, self.feature_positions]
        if not prefill:
            tensors.extend([self.noise_positions, self.token_ids, self.mask])
        _launch(self.device, tensors, 3 if prefill else 1, 32 if prefill else self.block_size, lane=lane)

    def commit(self, layer, key, value, start, count):
        rows = int(key.shape[-2])
        if not 1 <= rows <= 32 or tuple(key.shape) != (1, 8, rows, 128) or tuple(value.shape) != tuple(key.shape):
            raise ValueError("DFlash committed KV must have shape [1,8,1..32,128]")
        _launch(self.device, [*self.layers[layer], key, value, start, count], 2, rows)

    def commit_rows(self, layer, key, value, start, count, offset):
        """Prefill append of source rows [offset, offset+count) to ring positions [start, ...).

        start and offset are tile-row aligned (checked by the caller), so full tiles move whole."""
        rows = int(key.shape[-2])
        if rows % 32 or tuple(key.shape) != (1, 8, rows, 128) or tuple(value.shape) != tuple(key.shape):
            raise ValueError("DFlash prefill KV must have shape [1,8,32k,128]")
        _launch(self.device, [*self.layers[layer], key, value, start, count, offset], 6, rows)

    def attention_inputs(self, layer, noise_key, noise_value, rows=None):
        """Write noise K/V (rows >= block zeroed; rows=32 keeps every row of a multi-user block, whose mask
        selects the user's lane) into each ring's tail tile row; the ring itself is the attention input:
        exactly 32 noise rows after the committed window, so K-chunks and the dense absolute-position mask align."""
        from .dflash_ops import retile_heads

        rings = self.layers[layer]
        tiles = 128 // 32
        retile_heads(self.device, [
            (noise, (ring, tiles, (self.window // 32 + 1) * tiles, (self.window // 32) * tiles))
            for ring, noise in zip(rings, (noise_key, noise_value))
        ], rows or self.block_size)
        return rings

    def copy_from(self, other):
        """This ring <- other's committed context (owners and every layer's K/V; e.g. a prefill's context)."""
        ttnn.copy(other.owners, self.owners)
        for (key, value), (other_key, other_value) in zip(self.layers, other.layers):
            ttnn.copy(other_key, key)
            ttnn.copy(other_value, value)

    def host_scalar(self, value):
        """Prefill-only host metadata upload, never called in the draft path."""
        return ttnn.from_torch(torch.tensor([[value]], dtype=torch.int32), device=self.device,
                               dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)

    def release(self):
        for tensor in self._owned:
            ttnn.deallocate(tensor)
        self._owned.clear()
        self.layers.clear()


def commit_round(rings, layer, key, value, ring_table, positions):
    """Shared-round append (kernel mode 7): row r of key/value [1,8,W,128] goes to ring ring_table[r] at
    positions[r] (accepted rows only, see shared_round_kernel.cpp); owners are updated by the round kernel."""
    rows = int(key.shape[-2])
    if not 1 <= rows <= 16 or tuple(key.shape) != (1, 8, rows, 128) or tuple(value.shape) != tuple(key.shape):
        raise ValueError("DFlash round KV must have shape [1,8,1..16,128]")
    tensors = [*(r.layers[layer][0] for r in rings), *(r.layers[layer][1] for r in rings), key, value, ring_table, positions]
    _launch(rings[0].device, tensors, 7, rows, slots=len(rings))
