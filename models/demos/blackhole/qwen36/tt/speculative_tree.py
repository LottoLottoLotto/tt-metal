# SPDX-License-Identifier: Apache-2.0
"""Exact B8 shared-prefix target verification with device-side branch selection."""
from pathlib import Path

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.speculative import TargetVerifier
from models.demos.blackhole.qwen36.tt.attention.verify import commit_verify_many
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("speculative_tree_kernel.cpp").resolve())


class TreeTargetVerifier(TargetVerifier):
    """One packed layer pass over [anchor,A1..A7,Bf..B7].

    Inputs/features are stable borrowed buffers. Selection tensors, selected
    features, and target hidden/logits are owned by the caller. Candidate state
    is single-use: commit frees recurrent and attention candidates; convolution
    candidates remain owned by their GDN runners, as in single-path verification.
    """

    def __init__(self, model, page_table_width, fork_position,
                 feature_layers=(5, 19, 33, 47, 61)):
        if not 1 <= fork_position <= 7:
            raise ValueError("B8 tree fork_position must be in 1..7")
        super().__init__(model, 16 - fork_position, page_table_width, feature_layers)
        self.fork_position = fork_position
        self.depth, self.block_size = 7, 8
        self.path_b = (*range(fork_position), *range(8, self.rows))
        self.position_offsets = (*range(8), *range(fork_position, 8))
        self.parents = (-1, *range(7), fork_position - 1, *range(8, self.rows - 1))

    def _op(self, mode, tensors, input_bytes=32, output_bytes=32):
        grid = self.device.compute_with_storage_grid_size()
        workers = min(grid.x * grid.y, len(self.features) * self.model.args.dim // 32) if mode == 2 else 1
        core_list = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in core_list})
        tasks = [(core, [i]) for i, core in enumerate(core_list)]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL, core_ranges=cores,
            brisc_named_compile_time_args=[
                ("mode", mode), ("fork_position", self.fork_position),
                ("flag_dram", int(tensors[0].memory_config() == ttnn.DRAM_MEMORY_CONFIG)),
                ("input_bytes", input_bytes), ("output_bytes", output_bytes),
                ("width_tiles", self.model.args.dim // 32), ("layers", len(self.features)),
                ("workers", workers),
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
        program = ttnn.ProgramDescriptor(
            kernels=kernel.get_kernel_descriptors().kernels,
            cbs=[ttnn.CBDescriptor(total_size=2048, core_ranges=cores, format_descriptors=[
                ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048, ttnn.TileDescriptor(ttnn.Tile((32, 32)))),
            ])], semaphores=[],
        )
        ttnn.generic_op(tensors, program)

    def pack_branches(self, anchor, branches):
        """Capture-safe packing from borrowed native selector outputs, no readback."""
        if len(branches) != 2:
            raise ValueError("Tree verification requires exactly two branches")
        for branch in branches:
            if (tuple(branch.shape) != (1, 7) or branch.dtype != ttnn.uint32
                    or branch.layout != ttnn.ROW_MAJOR_LAYOUT
                    or branch.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                    or branch.device() != self.device):
                raise ValueError("Tree branches must be same-device uint32 DRAM row-major [1,7]")
        if (tuple(anchor.shape) != (1, 1) or anchor.dtype not in (ttnn.int32, ttnn.uint32)
                or anchor.layout != ttnn.ROW_MAJOR_LAYOUT or anchor.device() != self.device
                or anchor.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)):
            raise ValueError("Tree anchor must be same-device interleaved int32/uint32 [1,1]")
        self._op(0, [anchor, *branches, self.tokens], output_bytes=int(self.tokens.padded_shape[-1]) * 4)

    def seed(self, anchor, position, page_table_row, branches):
        if len(branches) != 2:
            raise ValueError("Tree verification requires two complete branches")
        self.set_page_table(page_table_row)
        positions = ttnn.from_torch(
            torch.tensor([position + i for i in self.position_offsets], dtype=torch.int32),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(positions, self.positions)
        if isinstance(branches[0], ttnn.Tensor):
            token = ttnn.from_torch(torch.tensor([[anchor]], dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            self.pack_branches(token, branches)
            ttnn.deallocate(token)
        else:
            a, b = (torch.as_tensor(branch).reshape(-1).tolist() for branch in branches)
            if len(a) != 7 or len(b) != 7 or a[:self.fork_position - 1] != b[:self.fork_position - 1]:
                raise ValueError("Tree branches require seven tokens and an identical shared prefix")
            tokens = ttnn.from_torch(torch.tensor([[anchor, *a, *b[self.fork_position - 1:]]],
                                                  dtype=torch.int32),
                                     dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(tokens, self.tokens)
        return branches

    def target_pass(self):
        idx = ttnn.reshape(ttnn.typecast(self.positions, ttnn.uint32), [1, self.rows])
        cos = ttnn.embedding(idx, self.model._target_cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idx, self.model._target_sin_table, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(idx)
        x = self.model.embd(self.tokens)
        candidates = []
        for index, layer in enumerate(self.model.layers):
            previous = x
            x, candidate = layer.verify_many(
                x, cos, sin, self.positions, self.page_table, fork_position=self.fork_position,
            )
            ttnn.deallocate(previous)
            if index in self.features:
                ttnn.copy(x, self.features[index])
            candidates.append(candidate)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        hidden = self.model._final_norm_decode(x)
        ttnn.deallocate(x)
        return hidden, candidates

    def accept_branches(self, logits):
        logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        argmax = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        argmax = ttnn.reshape(argmax, [1, self.rows])
        def integer(width=1):
            return ttnn.empty((1, width), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        selection = {"targets": integer(8), "flags": [integer() for _ in range(7)]}
        selection.update({key: integer() for key in ("accepted", "winner", "accepted_a", "accepted_b", "commit_index")})
        outputs = [selection["targets"], *selection["flags"],
                   *(selection[key] for key in ("accepted", "winner", "accepted_a", "accepted_b", "commit_index"))]
        self._op(1, [argmax, self.tokens, *outputs], input_bytes=int(argmax.padded_shape[-1]) * 4,
                 output_bytes=int(self.tokens.padded_shape[-1]) * 4)
        ttnn.deallocate(argmax)
        return selection

    def commit(self, candidates, selection):
        if len(candidates) != len(self.model.layers):
            raise ValueError("Tree commit requires every layer's candidate state")
        for layer, candidate in zip(self.model.layers, candidates):
            if layer.is_full_attention:
                commit_verify_many(candidate, selection["accepted"], self.positions, self.page_table,
                                   layer.attention.paged_kv_cache_key, layer.attention.paged_kv_cache_value,
                                   winner=selection["winner"])
                ttnn.deallocate(candidate.key)
                ttnn.deallocate(candidate.value)
            else:
                states, conv = candidate
                layer.attention.commit_verify_many(selection["commit_index"], states, conv)
        candidates.clear()

    def selected_features(self, winner):
        if not self.features:
            raise ValueError("Selected feature gather requires configured target feature layers")
        if (tuple(winner.shape) != (1, 1) or winner.dtype != ttnn.int32
                or winner.layout != ttnn.ROW_MAJOR_LAYOUT or winner.device() != self.device
                or winner.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)):
            raise ValueError("Feature selection requires a same-device int32 winner [1,1]")
        selected = ttnn.empty((1, 1, 8, len(self.features) * self.model.args.dim),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._op(2, [winner, selected, *self.features.values()])
        return selected
