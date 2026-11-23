# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Shared/Dense MLP with GeGLU activation.

Each decoder layer has BOTH a shared MLP and routed MoE experts.
Architecture: down_proj(GELU(gate_proj(x)) * up_proj(x))
intermediate_size = 2112, no bias.

HF weight shapes:
  gate_proj.weight: [intermediate_size, hidden_size] = [2112, 2816]
  up_proj.weight:   [intermediate_size, hidden_size] = [2112, 2816]
  down_proj.weight: [hidden_size, intermediate_size] = [2816, 2112]
"""

import os

import ttnn
from models.demos.gemma4.tt import decode_mm
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.tt.precision import dtype_to_str, gelu_fast, matmul_compute_config
from models.demos.gemma4.utils.general_utils import get_cache_file_name


class SharedMLP:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        gate_up_dtype=None,
        down_dtype=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        tp = mesh_config.tp if mesh_config else 1
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

        # Tag the cache filenames with the weight dtype so that flipping a
        # SharedMLP weight's dtype (e.g. bf16 → bfp8 for DRAM-pressure relief)
        # doesn't collide with a previously-cached file that holds the same
        # logical weight at a different dtype. The rest of the model's cache
        # entries are unaffected and stay reusable across runs.
        gate_up_dtype = gate_up_dtype or dtype
        down_dtype = down_dtype or dtype
        gate_up_suffix = f"_{dtype_to_str(gate_up_dtype)}"
        down_suffix = f"_{dtype_to_str(down_dtype)}"
        self.gate_up_compute_config = matmul_compute_config(gate_up_dtype)
        self.down_compute_config = matmul_compute_config(down_dtype)

        if tp > 1:
            col_mapper = mesh_config.column_parallel(mesh_device)
            row_mapper = mesh_config.row_parallel(mesh_device)
        else:
            col_mapper = None
            row_mapper = None

        if state_dict:
            gate_proj_weight = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            up_proj_weight = state_dict["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            gate_proj_weight = None
            up_proj_weight = None
            down_proj_weight = None

        # gate/up: column-parallel (shard output dim across TP devices)
        self.gate_proj = ttnn.as_tensor(
            gate_proj_weight,
            device=mesh_device,
            dtype=gate_up_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj.weight{tp_suffix}{gate_up_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj = ttnn.as_tensor(
            up_proj_weight,
            device=mesh_device,
            dtype=gate_up_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj.weight{tp_suffix}{gate_up_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # down: row-parallel (shard input dim, allreduce after)
        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=mesh_device,
            dtype=down_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj.weight{tp_suffix}{down_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel, down is row-parallel + allreduce.
        """
        gate = decode_mm.linear(hidden_states, self.gate_proj, self.gate_up_compute_config)
        up = decode_mm.linear(hidden_states, self.up_proj, self.gate_up_compute_config)
        if os.environ.get("GEMMA4_FUSE_GELU_MUL") == "1" and not gelu_fast() and hidden_states.padded_shape[-2] <= 32:
            # hidden = GELU(gate) * up in one eltwise launch (GELU applied to input a in the binary op;
            # non-exact vs the separate gelu op).
            hidden = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)])
        else:
            gate_act = ttnn.gelu(gate, fast_and_approximate_mode=gelu_fast())
            gate.deallocate(True)
            gate = gate_act
            hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        # output = hidden @ down_proj
        output = decode_mm.linear(hidden, self.down_proj, self.down_compute_config)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
