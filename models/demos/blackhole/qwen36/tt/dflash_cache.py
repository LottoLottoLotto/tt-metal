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


def _launch(device, tensors, mode, rows, feature_width, mask_token_id):
    workers = 64 if mode == 4 else 32 if mode == 2 else 1
    grid = device.compute_with_storage_grid_size()
    coordinates = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in coordinates})
    tasks = [(core, [i]) for i, core in enumerate(coordinates)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        brisc_named_compile_time_args=[
            ("mode", mode), ("rows", rows),
            ("feature_tiles", feature_width // 32), ("mask_token_id", mask_token_id),
        ],
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
            total_size=2048 if mode in (2, 4) else 12288,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048, tile)],
        )],
        semaphores=[],
    )
    ttnn.generic_op(tensors, program)


class CommittedKV:
    """Single-sequence ring with absolute positions, shared by all draft layers.

    A valid append is a contiguous prefix [start,start+count), count <= input rows.
    For draft, start+count <= anchor position. The scheduler owns these values;
    they are read only by the device kernel. Future/stale owners are masked even
    when a caller intentionally rewinds the anchor. Reset is required on reuse
    for another request. No speculative append/eviction/rollback is performed.
    """
    window = 2048

    def __init__(self, device, block_size, *, num_layers, feature_width, mask_token_id):
        self.device = device
        self.block_size = block_size
        self.feature_width = feature_width
        self.mask_token_id = mask_token_id
        self._owned = []
        self.owners = self._empty((1, 2048), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.feature_positions = self._empty((1, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.noise_positions = self._empty((1, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.token_ids = self._empty((1, block_size), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.mask = self._empty((1, 1, block_size, 2080), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.prefill_features = self._empty((1, 1, 32, feature_width), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # Zero initialization is essential: masked NaN values could still poison
        # SDPA's probability/value multiplication in never-owned ring slots.
        self.layers = []
        for _ in range(num_layers):
            pair = tuple(ttnn.zeros(
                (1, 8, 2048, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
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
        self._launch([self.owners, self.owners], 0, self.block_size)

    def _launch(self, tensors, mode, rows):
        _launch(self.device, tensors, mode, rows, self.feature_width, self.mask_token_id)

    def prepare(self, anchor, position, start, count, *, prefill=False):
        for name, tensor in (("anchor", anchor), ("position", position), ("feature_start", start), ("feature_count", count)):
            device_scalar(tensor, name)
        tensors = [self.owners, position, start, count, anchor, self.feature_positions]
        if not prefill:
            tensors.extend([self.noise_positions, self.token_ids, self.mask])
        self._launch(tensors, 3 if prefill else 1, 32 if prefill else self.block_size)

    def commit(self, layer, key, value, start, count):
        rows = int(key.shape[-2])
        if not 1 <= rows <= 32 or tuple(key.shape) != (1, 8, rows, 128) or tuple(value.shape) != tuple(key.shape):
            raise ValueError("DFlash committed KV must have shape [1,8,1..32,128]")
        self._launch([*self.layers[layer], key, value, start, count], 2, rows)

    def attention_inputs(self, layer, noise_key, noise_value):
        # Concatenation is temporary, not persistent cache storage. Force exactly
        # 32 noise rows so K-chunks and the dense absolute-position mask align.
        result = []
        for committed, noise in zip(self.layers[layer], (noise_key, noise_value)):
            padded = ttnn.pad(noise, ((0, 0), (0, 0), (0, 32 - self.block_size), (0, 0)), 0.0,
                              memory_config=DRAM)
            result.append(ttnn.concat([committed, padded], dim=2, memory_config=DRAM))
        return tuple(result)

    def stage_prefill(self, features, offset):
        """Copy one aligned tile row without shape-/offset-dependent programs."""
        self._launch([features, self.prefill_features, offset], 4, 32)
        return self.prefill_features

    def host_scalar(self, value):
        """Prefill-only host metadata upload, never called in the draft path."""
        return ttnn.from_torch(torch.tensor([[value]], dtype=torch.int32), device=self.device,
                               dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)

    def release(self):
        for tensor in self._owned:
            ttnn.deallocate(tensor)
        self._owned.clear()
        self.layers.clear()
