# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import os
from pathlib import Path

import torch
import ttnn
from models.demos.blackhole.qwen36.tt import weight_prefetch
from models.demos.blackhole.qwen36.tt.attention.config import optimized_27b_attention


@dataclass(frozen=True)
class AttentionWeights:
    q_proj: ttnn.Tensor | None
    k_proj: ttnn.Tensor | None
    v_proj: ttnn.Tensor | None
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
    from models.demos.blackhole.qwen36.tt.model_config import evaluation_weight_dtype

    optimized = optimized_27b_attention()
    draft_layer = tensor_cache_path is not None and Path(tensor_cache_path).name.startswith("mtp.layers.")
    if optimized and draft_layer:
        # The draft layer is stored losslessly and is absent from the evaluator's target-layer
        # precision map; native attention runs it at the same BF8 contract as target layers.
        linear_dtype = ttnn.bfloat8_b
    else:
        linear_dtype = evaluation_weight_dtype("attention", tensor_cache_path, linear_dtype)
    if optimized:
        expected = {"q_proj": (12288, 5120), "k_proj": (1024, 5120),
                    "v_proj": (1024, 5120), "o_proj": (5120, 6144)}
        for name, shape in expected.items():
            if tuple(state_dict[f"{name}.weight"].shape) != shape:
                raise ValueError(f"Optimized 27B attention requires {name}.weight {shape}")
        if linear_dtype != ttnn.bfloat8_b:
            raise ValueError("Optimized 27B attention requires BF8 projection weights")
    def load_2d(name):
        t = state_dict[f"{name}.weight"].T.contiguous()
        logical = t
        suffix = ""
        if name == "o_proj" and optimized:
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            t, suffix = tpc.proj_o_rotating_pack(t, linear_dtype)
        weight = ttnn.as_tensor(
            t,
            dtype=linear_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight{suffix}") if tensor_cache_path else None,
        )
        if (
            name == "o_proj"
            and optimized
            and not draft_layer
            and tuple(logical.shape) == (6144, 5120)
            and linear_dtype == ttnn.bfloat8_b
            and weight_prefetch.enabled("proj_o")
        ):
            weight_prefetch.register(
                "proj_o", weight, logical, mesh_device,
                (tensor_cache_path / f"self_attn.{name}.weight.prefetch") if tensor_cache_path else None,
                in0_block_w=4,
            )
        return weight

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
        if not optimized and os.environ.get("QWEN36_FUSED_QKV") != "1":
            return None
        fused = torch.cat(
            [
                state_dict["q_proj.weight"],
                state_dict["k_proj.weight"],
                state_dict["v_proj.weight"],
            ],
            dim=0,
        ).T.contiguous()
        cache_name = "self_attn.qkv_fused.weight"
        if optimized:
            # Worker-bank page order for the packed decode reader: 90 workers x 5 columns
            # (gated_attention.py decode specs). Prefill reads it through the factory page map.
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            fused = tpc.repack_for_worker_banks(fused, workers=90, full_cols=5)
            cache_name += ".packed-worker-bank.c5"
        return ttnn.as_tensor(
            fused,
            dtype=linear_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / cache_name) if tensor_cache_path else None,
        )

    return AttentionWeights(
        q_proj=None if optimized else load_2d("q_proj"),
        k_proj=None if optimized else load_2d("k_proj"),
        v_proj=None if optimized else load_2d("v_proj"),
        o_proj=load_2d("o_proj"),
        q_norm=load_norm("q_norm"),
        k_norm=load_norm("k_norm"),
        qkv_fused=load_fused_qkv(),
    )
