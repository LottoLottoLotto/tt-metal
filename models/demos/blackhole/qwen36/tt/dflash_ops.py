# SPDX-License-Identifier: Apache-2.0
"""P150 DFlash2 convolution and greedy, predecessor-conditioned candidate path.

Only checkpoint layout conversion happens on the host. Inference consists of
TTNN matmuls/embeddings and native kernels; even top16 and path decisions remain
on device. Objects are single-stream, non-reentrant owners. Warm every block width
before capture. Returned buffers have fixed addresses per width and are borrowed:
the next same-width call overwrites them. A trace and its input/output buffers
must be retired before release(); release does not synchronize the device.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_VOCAB = 248320
_RANK = 256
_WORKERS = 64
_CONV = str(Path(__file__).with_name("dflash_ops_conv.cpp").resolve())
_SELECT = str(Path(__file__).with_name("dflash_ops_select.cpp").resolve())


def _weight(name, tensor, shape):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
        raise ValueError(f"DFlash {name}: expected checkpoint shape {shape}")
    if tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu":
        raise ValueError(f"DFlash {name}: requires CPU BF16 checkpoint storage")


def _upload(device, tensor, layout=ttnn.TILE_LAYOUT):
    # Upload the supplied checkpoint, never silently reuse a cache with different
    # weights. The enclosing loader owns revision/hash-bound disk caching.
    return ttnn.from_torch(
        tensor.contiguous(), device=device, dtype=ttnn.bfloat16,
        layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _empty(device, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.empty(shape, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _cores(device, count):
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < _WORKERS:
        raise ValueError("DFlash native ops require at least 64 logical compute/storage cores")
    coordinates = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
    return coordinates, ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in coordinates})


def _check(device, name, tensor, width, rows=None):
    shape = tuple(int(x) for x in tensor.shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[-1] != width or not 1 <= shape[-2] <= 32:
        raise ValueError(f"DFlash {name}: expected [1,1,1..32,{width}], got {shape}")
    if rows is not None and shape[-2] != rows:
        raise ValueError(f"DFlash {name}: row count differs from hidden")
    if tuple(int(x) for x in tensor.padded_shape) != (1, 1, 32, width):
        raise ValueError(f"DFlash {name}: requires exactly one standard row tile")
    if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT or tensor.tile != ttnn.Tile((32, 32)):
        raise ValueError(f"DFlash {name}: requires standard tiled BF16")
    if tensor.device() != device or tensor.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
        raise ValueError(f"DFlash {name}: requires same-device interleaved DRAM/L1")
    return shape[-2]


def _compute():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=False,
    )


def _linear(hidden, weight, output, config):
    return ttnn.linear(
        hidden, weight, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=config, optional_output_tensor=output,
    )


@dataclass(frozen=True)
class _Dynamic:
    owner: object
    projection: object
    rows: int


class NativeDynamicConv:
    """Two bank/two tap/group16 convolution using actual learned BF16 weights.

    prepare returns (pre-convolved hidden, opaque post-bank handle). finish must
    use that handle on this instance, before the next same-width prepare. There
    is no history across blocks: tap one is zero at each block's first row.
    """

    def __init__(self, device, base_kernel_torch, projection_weight_torch, cache_prefix):
        self.hidden_size = int(base_kernel_torch.shape[-1])
        if self.hidden_size not in (2048, 5120):
            raise ValueError("DFlash convolution requires hidden width 2048 or 5120")
        self.projection_size = 4 * (self.hidden_size // 16)
        _weight("base_kernel", base_kernel_torch, (2, 2, self.hidden_size))
        _weight("kernel_projection", projection_weight_torch, (self.projection_size, self.hidden_size))
        _cores(device, _WORKERS)
        self.device = device
        self.cache_prefix = str(cache_prefix)
        self.base = _upload(device, base_kernel_torch.reshape(1, 1, 4, self.hidden_size))
        self.projection = _upload(device, projection_weight_torch.T)
        self.config = _compute()
        self._buffers = {}
        self._released = False

    def _storage(self, rows):
        if self._released:
            raise RuntimeError("DFlash convolution has been released")
        if rows not in self._buffers:
            self._buffers[rows] = (
                _empty(self.device, (1, 1, rows, self.projection_size)),
                _empty(self.device, (1, 1, rows, self.hidden_size)),
                _empty(self.device, (1, 1, rows, self.hidden_size)),
            )
        return self._buffers[rows]

    def _convolve(self, hidden, projection, output, rows, branch):
        coordinates, cores = _cores(self.device, _WORKERS)
        tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
        cbs = [
            ttnn.CBDescriptor(
                total_size=count * 2048, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.bfloat16, 2048, tile)],
            )
            for index, count in ((0, 2), (1, 2), (2, 2), (3, 3), (4, 1), (5, 1), (6, 1), (7, 1), (16, 1))
        ]
        compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, dst_full_sync_en=False,
        )
        unpack = [ttnn.UnpackToDestMode.Default] * 64
        for index in (0, 1, 2, 4, 5, 6, 7):
            unpack[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode.extend(unpack)
        tasks = [(core, [i]) for i, core in enumerate(coordinates)]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_CONV, core_ranges=cores,
            defines=[("DFLASH_ROWS", str(rows)), ("DFLASH_BRANCH", str(branch)),
                     ("DFLASH_HIDDEN", str(self.hidden_size)),
                     ("DFLASH_HIDDEN_DRAM", str(int(hidden.memory_config() == ttnn.DRAM_MEMORY_CONFIG)))],
            ncrisc_common_runtime_args=[int(t.buffer_address()) for t in (hidden, self.base, projection)],
            brisc_common_runtime_args=[int(output.buffer_address())],
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=tasks, brisc_args=tasks, trisc_args=tasks,
            ),
        )
        program = ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])
        ttnn.generic_op([hidden, self.base, projection, output], program)
        return output

    def prepare(self, hidden):
        rows = _check(self.device, "conv hidden", hidden, self.hidden_size)
        projected, prepared, _ = self._storage(rows)
        _linear(hidden, self.projection, projected, self.config)
        self._convolve(hidden, projected, prepared, rows, 0)
        return prepared, _Dynamic(self, projected, rows)

    def finish(self, hidden, dynamic):
        if not isinstance(dynamic, _Dynamic) or dynamic.owner is not self:
            raise ValueError("DFlash finish requires this convolution's opaque prepare handle")
        _check(self.device, "conv branch output", hidden, self.hidden_size, dynamic.rows)
        _, _, finished = self._storage(dynamic.rows)
        return self._convolve(hidden, dynamic.projection, finished, dynamic.rows, 1)

    def release(self):
        """Free owner storage after all outstanding commands/traces are retired."""
        if not self._released:
            for tensor in (self.base, self.projection):
                ttnn.deallocate(tensor)
            for buffers in self._buffers.values():
                for tensor in buffers:
                    ttnn.deallocate(tensor)
            self._buffers.clear()
            self._released = True


class NativeCandidateSelector:
    """Full-vocabulary top16 plus the learned, sequential greedy selector.

    Pinned TTNN topk only uses its multicore UInt16 algorithm below 65535
    columns and on power-of-two widths. Full 248320-column topk would select its
    L1-heavy single-core implementation. Instead 64 device workers reduce all
    7760 vocabulary tiles to 16 candidates each; one worker per position merges
    those 1024 entries to the exact global top16, with uint32 IDs throughout.

    Tied unary/scores use lower vocabulary ID, intentionally deterministic;
    PyTorch's sorted=False topk gives no stable tie-order guarantee. Rank products
    are rounded BF16 before HiFi4/FP32-accumulated dot products, then dot and unary
    addition each round BF16, matching the reference's tensor boundaries.

    select_branches is B8-only: A is this same greedy path; B shares its prefix,
    takes the second-ranked conditioned score at the fixed fork, and then uses
    its own predecessor. Both paths borrow separate stable buffers. Warm the
    selected fork before capture; neither selection API reads device values.
    """

    def __init__(self, device, predecessor_weight_torch, successor_weight_torch,
                 hidden_projection_weight_torch, cache_prefix, top_k=16):
        if top_k != 16:
            raise ValueError("DFlash2 checkpoint contract requires top_k=16")
        _weight("predecessor_codebook", predecessor_weight_torch, (_VOCAB, _RANK))
        _weight("successor_codebook", successor_weight_torch, (_VOCAB, _RANK))
        self.hidden_size = int(hidden_projection_weight_torch.shape[-1])
        if self.hidden_size not in (2048, 5120):
            raise ValueError("DFlash selector requires hidden width 2048 or 5120")
        _weight("hidden_projection", hidden_projection_weight_torch, (_RANK, self.hidden_size))
        _cores(device, _WORKERS)
        self.device = device
        self.cache_prefix = str(cache_prefix)
        self.top_k = top_k
        self.predecessor = _upload(device, predecessor_weight_torch, ttnn.ROW_MAJOR_LAYOUT)
        self.successor = _upload(device, successor_weight_torch, ttnn.ROW_MAJOR_LAYOUT)
        self.projection = _upload(device, hidden_projection_weight_torch.T)
        self.config = _compute()
        self._buffers = {}
        self._released = False

    def _storage(self, rows):
        if self._released:
            raise RuntimeError("DFlash selector has been released")
        if rows not in self._buffers:
            self._buffers[rows] = {
                "projection": _empty(self.device, (1, 1, rows, _RANK)),
                "row": _empty(self.device, (1, 1, 1, _RANK)),
                "partial": _empty(self.device, (rows * _WORKERS, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "path": _empty(self.device, (1, rows), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "candidates": [_empty(self.device, (1, 16), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) for _ in range(rows)],
                "unary": [_empty(self.device, (1, 1, 1, 16)) for _ in range(rows)],
                "next": [_empty(self.device, (1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) for _ in range(rows)],
            }
        return self._buffers[rows]

    def _kernel(self, tensors, rows, mode, count=1, position=0, logits_dram=True):
        coordinates, cores = _cores(self.device, count)
        tasks = [(core, [position if mode in (2, 4) else i]) for i, core in enumerate(coordinates)]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_SELECT, core_ranges=cores,
            defines=[("DFLASH_ROWS", str(rows)), ("DFLASH_MODE", str(mode)),
                     ("DFLASH_LOGITS_DRAM", str(int(logits_dram))), ("DFLASH_POSITION", str(position))],
            brisc_common_runtime_args=[int(t.buffer_address()) for t in tensors],
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(brisc_args=tasks),
        )
        cb = ttnn.CBDescriptor(
            total_size=16384, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                0, ttnn.bfloat16, 2048, ttnn.TileDescriptor(ttnn.Tile((32, 32))),
            )],
        )
        program = ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=[cb], semaphores=[])
        ttnn.generic_op(tensors, program)

    def select(self, hidden, logits, anchor_ids):
        """Return the original greedy path in borrowed stable device storage."""
        return self._select(hidden, logits, anchor_ids)

    def select_branches(self, hidden, logits, anchor_ids, fork_position):
        """Return borrowed uint32 [1,7] (A, B); fork_position is in 1..7."""
        if type(fork_position) is not int or not 1 <= fork_position <= 7:
            raise ValueError("DFlash branch fork_position must be a host integer in 1..7")
        return self._select(hidden, logits, anchor_ids, fork_position)

    def _select(self, hidden, logits, anchor_ids, fork_position=None):
        rows = _check(self.device, "selector hidden", hidden, self.hidden_size)
        if rows > 31:
            raise ValueError("DFlash selector requires B-1 <= 31")
        if fork_position is not None and rows != 7:
            raise ValueError("DFlash two-branch selection requires B8 hidden/logits with seven rows")
        _check(self.device, "selector logits", logits, _VOCAB, rows)
        if (tuple(anchor_ids.shape) != (1, 1) or anchor_ids.dtype not in (ttnn.uint32, ttnn.int32)
                or anchor_ids.layout != ttnn.ROW_MAJOR_LAYOUT or anchor_ids.device() != self.device
                or anchor_ids.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError("DFlash anchor_ids requires same-device DRAM row-major uint32/int32 [1,1]")
        storage = self._storage(rows)
        if fork_position is not None and "path_b" not in storage:
            storage["path_b"] = _empty(self.device, (1, rows), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            storage["next_b"] = [
                _empty(self.device, (1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) for _ in range(rows)
            ]
        projected = storage["projection"]
        _linear(hidden, self.projection, projected, self.config)
        self._kernel([logits, storage["partial"]], rows, 0, _WORKERS,
                     logits_dram=logits.memory_config() == ttnn.DRAM_MEMORY_CONFIG)
        merge = [storage["partial"]]
        for candidates, unary in zip(storage["candidates"], storage["unary"]):
            merge.extend((candidates, unary))
        self._kernel(merge, rows, 1, rows)
        predecessor = anchor_ids
        converted_anchor = None
        if anchor_ids.dtype == ttnn.int32:
            converted_anchor = ttnn.typecast(anchor_ids, ttnn.uint32)
            predecessor = converted_anchor
        predecessor_b = None
        # Only shape controls this loop. Every chosen predecessor remains a device
        # tensor and is consumed by the next position's actual embedding lookup.
        for position in range(rows):
            candidates = storage["candidates"][position]
            pred_embedding = ttnn.embedding(predecessor, self.predecessor, layout=ttnn.TILE_LAYOUT)
            succ_embedding = ttnn.embedding(candidates, self.successor, layout=ttnn.TILE_LAYOUT)
            pred = ttnn.reshape(pred_embedding, (1, 1, 1, _RANK))
            succ = ttnn.reshape(succ_embedding, (1, 1, 16, _RANK))
            row = storage["row"]
            self._kernel([projected, row], rows, 3, count=8, position=position)
            conditioned = ttnn.multiply(pred, row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            dot = ttnn.matmul(
                conditioned, succ, transpose_b=True, dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.config,
            )
            scores = ttnn.add(dot, storage["unary"][position], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            predecessor = storage["next"][position]
            self._kernel([scores, candidates, predecessor, storage["path"]], rows, 2, position=position)
            if fork_position is not None and position == fork_position - 1:
                # A and B have the same predecessor here, so reuse the exact
                # learned scores. The native rank-two update also copies A's
                # prefix; no host IDs or redundant prefix scoring are involved.
                predecessor_b = storage["next_b"][position]
                self._kernel(
                    [scores, candidates, predecessor_b, storage["path_b"], storage["path"]],
                    rows, 4, position=position,
                )
            elif fork_position is not None and position >= fork_position:
                # Candidate embeddings, unaries and the projected hidden row
                # are shared. Only predecessor conditioning belongs to B.
                pred_embedding_b = ttnn.embedding(predecessor_b, self.predecessor, layout=ttnn.TILE_LAYOUT)
                pred_b = ttnn.reshape(pred_embedding_b, (1, 1, 1, _RANK))
                conditioned_b = ttnn.multiply(pred_b, row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                dot_b = ttnn.matmul(
                    conditioned_b, succ, transpose_b=True, dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.config,
                )
                scores_b = ttnn.add(dot_b, storage["unary"][position], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                predecessor_b = storage["next_b"][position]
                self._kernel(
                    [scores_b, candidates, predecessor_b, storage["path_b"]], rows, 2, position=position,
                )
                for tensor in (pred_embedding_b, conditioned_b, dot_b, scores_b):
                    ttnn.deallocate(tensor)
            # Reshape tensors alias embeddings; deallocate each allocation once.
            for tensor in (pred_embedding, succ_embedding, conditioned, dot, scores):
                ttnn.deallocate(tensor)
        if converted_anchor is not None:
            ttnn.deallocate(converted_anchor)
        return storage["path"] if fork_position is None else (storage["path"], storage["path_b"])

    def release(self):
        """Invalidate borrowed paths and diagnostics; does not synchronize."""
        if not self._released:
            for tensor in (self.predecessor, self.successor, self.projection):
                ttnn.deallocate(tensor)
            for storage in self._buffers.values():
                for value in storage.values():
                    for tensor in value if isinstance(value, list) else (value,):
                        ttnn.deallocate(tensor)
            self._buffers.clear()
            self._released = True
