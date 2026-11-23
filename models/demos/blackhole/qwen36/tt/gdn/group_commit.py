# SPDX-License-Identifier: Apache-2.0
"""Shared verify rounds: per-group GDN commit into DRAM user slots (pure DMA, group_commit_kernel.cpp).

A round's multi-row groups leave one candidate per row (FusedVerifyOutput(groups=...) states
[T,48,128,128], native frontend histories [1,3T,10240]); the round bookkeeping kernel writes the
commit table ([1,32] uint32: count, then (slot, accepted row) pairs) and this copies each listed row's
candidates into its slot. Single-row groups were already written in place by the verify pass.
"""
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

_KERNEL = "models/demos/blackhole/qwen36/tt/gdn/group_commit_kernel.cpp"
_MAX_COMMON_ARGS = 300
_CORES = 64


def _make_program(table, layers, slots):
    grid = table.device().compute_with_storage_grid_size()
    workers = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(_CORES)]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in workers})
    args = [int(table.buffer_address())]
    for candidates, histories, states, convs in layers:
        args += [int(candidates.buffer_address()), int(histories.buffer_address()),
                 *[int(t.buffer_address()) for t in states], *[int(t.buffer_address()) for t in convs]]
    per_core = [(c, [i]) for i, c in enumerate(workers)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=cores,
        defines=[("GC_LAYERS", str(len(layers))), ("GC_SLOTS", str(slots)), ("GC_WORKERS", str(2 * _CORES))],
        ncrisc_common_runtime_args=args,
        brisc_common_runtime_args=args,
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(ncrisc_args=per_core, brisc_args=per_core),
    )
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    cbs = [ttnn.CBDescriptor(total_size=8 * 2048, core_ranges=cores,
                             format_descriptors=[ttnn.CBFormatDescriptor(cb, ttnn.bfloat16, 2048, tile)])
           for cb in (0, 1)]
    return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])


class GroupCommit:
    """layers: per GDN layer (candidate states, candidate histories, slot states, slot histories)."""

    def __init__(self, table, layers):
        self.table = table
        self.layers = [(c, h, tuple(s), tuple(v)) for c, h, s, v in layers]
        self.slots = len(self.layers[0][2])
        per_launch = max(1, (_MAX_COMMON_ARGS - 1) // (2 + 2 * self.slots))
        self.launches = []
        for start in range(0, len(self.layers), per_launch):
            chunk = self.layers[start:start + per_launch]
            tensors = [table] + [t for c, h, s, v in chunk for t in (c, h, *s, *v)]
            self.launches.append((tensors, _make_program(table, chunk, self.slots)))

    def __call__(self):
        for tensors, program in self.launches:
            ttnn.generic_op(tensors, program)
