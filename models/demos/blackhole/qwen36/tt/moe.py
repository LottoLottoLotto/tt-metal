# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MoE routing plus single-P150 expert execution."""

from pathlib import Path
import torch

import ttnn

from models.common.modules.moe.tt_moe_local import TTLocalMoEDecode, TTLocalMoEGate, decode_linear
from models.common.modules.moe.batched_experts import BatchedExperts
from models.common.modules.moe.selected_experts import BatchedTopKRouting
from models.common.modules.moe.grouped_experts import GroupedExperts
from models.common.modules.moe.grouped_routing import GroupedRouting, grouped_combine, prefill_valid_rows  # noqa: F401
from models.demos.blackhole.qwen36.tt.weight_mapping import prepare_qwen35_moe_layer


class Qwen36MoE:
    """Single-P150 MoE with explicit concurrent decode and grouped prefill paths."""

    def __init__(self, mesh_device, state_dict, args, tensor_cache_path=None):
        if getattr(args, "num_devices", 1) != 1:
            raise ValueError("Qwen36MoE requires one P150")
        self.max_batch_size = getattr(args, "max_batch_size", 1)
        self._batched = {}
        native_experts = (
            getattr(state_dict, "has_native_tensor", lambda _key: False)("experts.gate_up_proj")
            and state_dict.has_native_tensor("experts.down_proj")
        )
        if native_experts:
            # Routed experts dominate checkpoint size. Their native files are
            # already BFP4, tiled, and oriented [1, expert, input, output].
            # Load them directly to device; only the small router/shared
            # tensors need the ordinary torch preparation path.
            gate_up = state_dict.native_tensor("experts.gate_up_proj", mesh_device)
            down = state_dict.native_tensor("experts.down_proj", mesh_device)
            shared_gate = state_dict["shared_expert.gate_proj.weight"].T
            shared_up = state_dict["shared_expert.up_proj.weight"].T
            weights = {
                "router": state_dict["gate.weight"].T,
                "gate_up": gate_up,
                "down": down,
                "shared_gate_up": torch.cat((shared_gate, shared_up), dim=-1),
                "shared_down": state_dict["shared_expert.down_proj.weight"].T,
                "shared_expert_gate": state_dict["shared_expert_gate.weight"].T,
            }
        else:
            weights = prepare_qwen35_moe_layer(state_dict)
        cache_dir = Path(tensor_cache_path) / "moe" if tensor_cache_path is not None else None
        self.experts = TTLocalMoEDecode(
            mesh_device,
            weights["gate_up"],
            weights["down"],
            weights["shared_gate_up"],
            weights["shared_down"],
            top_k=args.num_experts_per_tok,
            cache_dir=cache_dir,
        )
        self.gate = TTLocalMoEGate(
            mesh_device,
            weights["router"],
            weights["shared_expert_gate"],
            top_k=args.num_experts_per_tok,
            shared_slot=self.experts.shared_slot,
        )
        self._grouped_routing = GroupedRouting(
            mesh_device, self.gate.num_experts, self.gate.top_k, self.experts.shared_slot
        )
        self._grouped_experts = GroupedExperts(
            mesh_device, self.experts.gate_up, self.experts.down,
            hidden=self.experts.hidden_size, intermediate=self.experts.intermediate_size,
            stored_experts=self.experts.num_experts + 1,
        )
        self._prefill_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True,
        )

    def _forward_grouped(self, x, valid_rows=None):
        packed_input = x
        if x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            packed_input = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        logits = ttnn.linear(
            packed_input, self.gate.router, compute_kernel_config=self._prefill_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        packed = self._grouped_routing(packed_input, logits, valid_rows)
        if packed_input is not x:
            ttnn.deallocate(packed_input)
        ttnn.deallocate(logits)
        projected = self._grouped_experts(packed["x"], packed["weights"], packed["experts"])
        output = grouped_combine(projected, packed["inverse"], top_k=self.gate.top_k)
        ttnn.deallocate(projected)
        for tensor in packed.values():
            ttnn.deallocate(tensor)
        return output

    def _forward_one(self, x):
        selection, weights = self.gate.forward(x)
        output = self.experts.forward(x, selection, weights)
        ttnn.deallocate(selection)
        ttnn.deallocate(weights)
        return output

    def forward_batched(self, x):
        """Independent request rows [1,1,B,H], B in 1/2/4/8/16.

        Requires a model reserved with max_batch_size > 1. Expert weights are shared
        across the device-routed union; no request state lives in MoE. Padded/inactive
        rows may compute outputs but cannot affect any other row. Warm every serving
        bucket before trace capture; each bucket has its own program descriptors.
        """
        input_shape = list(x.shape)
        rows = 1
        for dimension in input_shape[:-1]:
            rows *= int(dimension)
        if (self.max_batch_size <= 1 or rows not in (1, 2, 4, 8, 16)
                or rows > self.max_batch_size or input_shape[-1] != self.experts.hidden_size):
            raise ValueError("batched MoE needs a reserved 1/2/4/8/16 bucket and the model hidden width")
        flat = ttnn.reshape(x, (1, 1, rows, self.experts.hidden_size))
        if rows == 1:
            return ttnn.reshape(self._forward_one(flat), input_shape)
        if rows not in self._batched:
            routing = BatchedTopKRouting(
                self.gate.mesh_device, self.gate.num_experts, self.gate.top_k, self.experts.shared_slot, rows,
            )
            experts = BatchedExperts(
                self.gate.mesh_device, self.experts.gate_up, self.experts.down,
                hidden=self.experts.hidden_size, intermediate=self.experts.intermediate_size,
                stored_experts=self.experts.num_experts + 1, top_k=self.gate.top_k, rows=rows,
            )
            self._batched[rows] = routing, experts
        routing, experts = self._batched[rows]
        packed = flat
        if flat.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            packed = ttnn.to_memory_config(flat, ttnn.L1_MEMORY_CONFIG)
        logits = decode_linear(packed, self.gate.router, self.gate.router_program)
        selection, weights = routing(logits)
        ttnn.deallocate(logits)
        output = experts(packed, selection, weights)
        for tensor in (selection, weights):
            ttnn.deallocate(tensor)
        if packed is not flat:
            ttnn.deallocate(packed)
        return ttnn.reshape(output, input_shape)

    def forward(self, x, mode="prefill", valid_rows=None):
        """``valid_rows`` (prefill only): a ``prefill_valid_rows`` device page marking each
        segment's real rows. Padding rows then take no experts and return zero rows; real
        rows are unchanged. None (full buckets, the traced chunk) keeps the unmasked path."""
        if mode not in ("prefill", "decode"):
            raise ValueError("MoE mode must be 'prefill' or 'decode'")
        if valid_rows is not None and mode != "prefill":
            raise ValueError("valid_rows applies to prefill only")
        if mode == "decode" and self.max_batch_size > 1:
            return self.forward_batched(x)
        input_shape = list(x.shape)
        hidden = input_shape[-1]
        if hidden != self.experts.hidden_size:
            raise ValueError(f"MoE input width must be {self.experts.hidden_size}, got {hidden}")
        rows = 1
        for dimension in input_shape[:-1]:
            rows *= int(dimension)
        flat = ttnn.reshape(x, (1, 1, rows, hidden))
        if rows <= 2 and valid_rows is None:
            # Decode, or both rows of a speculative verification in one expert pass.
            return ttnn.reshape(self._forward_one(flat), input_shape)
        # Pack all verification rows together. Each selected expert reads its
        # resident weights once per group, rather than once per token pair.
        return ttnn.reshape(self._forward_grouped(flat, valid_rows), input_shape)
