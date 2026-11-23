# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import os

import torch
import ttnn


@dataclass(frozen=True)
class AttentionWeights:
    q_proj: ttnn.Tensor
    k_proj: ttnn.Tensor
    v_proj: ttnn.Tensor
    o_proj: ttnn.Tensor
    q_norm: ttnn.Tensor  # +1 pre-offset (zero-centered RMSNorm)
    k_norm: ttnn.Tensor
    qkv_fused: ttnn.Tensor | None = None


def load_attention_weights(mesh_device, state_dict, tensor_cache_path=None) -> AttentionWeights:
    linear_dtype = (
        ttnn.bfloat4_b
        if os.environ.get("QWEN36_BFP4_WEIGHTS") == "1"
        else ttnn.bfloat8_b
    )
    def load_2d(name):
        t = state_dict[f"{name}.weight"].T.contiguous()
        return ttnn.as_tensor(
            t,
            dtype=linear_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight") if tensor_cache_path else None,
        )

    def load_norm(name):
        t = state_dict[f"{name}.weight"] + 1.0
        return ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight_offset") if tensor_cache_path else None,
        )
    def load_fused_qkv():
        if os.environ.get("QWEN36_FUSED_QKV") != "1":
            return None
        fused = torch.cat(
            [
                state_dict["q_proj.weight"],
                state_dict["k_proj.weight"],
                state_dict["v_proj.weight"],
            ],
            dim=0,
        ).T.contiguous()
        return ttnn.as_tensor(
            fused,
            dtype=linear_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / "self_attn.qkv_fused.weight") if tensor_cache_path else None,
        )

    return AttentionWeights(
        q_proj=load_2d("q_proj"),
        k_proj=load_2d("k_proj"),
        v_proj=load_2d("v_proj"),
        o_proj=load_2d("o_proj"),
        q_norm=load_norm("q_norm"),
        k_norm=load_norm("k_norm"),
        qkv_fused=load_fused_qkv(),
    )
