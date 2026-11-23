# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Remap HuggingFace Qwen3.5-9B state dict to internal format.

Handles:
- Stripping 'model.language_model.' prefix
- Filtering out vision encoder and MTP weights
- Renaming combined in_proj_qkv → qkv_proj (DeltaNet layers; the op uses the fused weight)
- Splitting combined conv1d.weight into separate Q, K, V conv weights (DeltaNet layers)
- Renaming lm_head.weight → output.weight
- Renaming embed_tokens → tok_embeddings
"""
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Dict

import torch

# Layer indices that use full (softmax) attention
FULL_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}

# DeltaNet QKV split dimensions (used to split the combined conv1d.weight into
# per-stream Q/K/V conv weights — the QKV projection itself stays combined).
# Q: num_key_heads(16) × key_head_dim(128) = 2048
# K: num_key_heads(16) × key_head_dim(128) = 2048
# (V = num_value_heads(32) × value_head_dim(128) = 4096 is the remaining slice)
LINEAR_Q_DIM = 2048
LINEAR_K_DIM = 2048


def remap_qwen36_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap HF Qwen3.5-9B state dict to internal format.

    Args:
        state_dict: Raw HuggingFace state dict loaded from safetensors.

    Returns:
        Remapped state dict with internal naming convention.
    """
    remapped = {}

    for key, tensor in state_dict.items():
        # Filter out vision encoder weights (check original key — no prefix stripping yet)
        if "visual" in key or key.startswith("model.visual"):
            continue
        # Filter out MTP (multi-token prediction) weights (original key)
        if key.startswith("mtp"):
            continue

        # Strip the language-model prefix. Two checkpoint sources produce different
        # prefixes for the same internal weights:
        #   - raw sharded safetensors:           model.language_model.X
        #   - AutoModelForCausalLM.from_pretrained (text-only Qwen3_5ForCausalLM): model.X
        # Strip whichever matches, longest first, so BOTH sources yield identical
        # internal keys. Top-level weights like lm_head.weight have no prefix and are
        # matched against the original `key` below.
        new_key = key
        for prefix in ("model.language_model.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break

        # Rename top-level weights
        if new_key == "embed_tokens.weight":
            remapped["tok_embeddings.weight"] = tensor
            continue
        if key == "lm_head.weight":
            remapped["output.weight"] = tensor
            continue
        # Final norm (model.language_model.norm.weight)
        if new_key == "norm.weight":
            remapped["norm.weight"] = tensor
            continue

        # Handle per-layer weights
        if new_key.startswith("layers."):
            parts = new_key.split(".")
            layer_idx = int(parts[1])
            layer_prefix = f"layers.{layer_idx}"
            sub_key = ".".join(parts[2:])

            # DeltaNet layers: keep ONLY the combined QKV weight. The split q/k/v_proj
            # were dead — the op runs the fused QKV projection from the combined weight
            # (it only read the splits in a fallback reached when qkv_proj_weight is None,
            # which never happens for the 9B).
            if sub_key == "linear_attn.in_proj_qkv.weight":
                remapped[f"{layer_prefix}.linear_attn.qkv_proj.weight"] = tensor  # [8192, 4096]
                continue

            if sub_key == "linear_attn.conv1d.weight":
                conv = tensor  # [8192, 1, 4]
                q_conv = conv[:LINEAR_Q_DIM, :, :]
                k_conv = conv[LINEAR_Q_DIM : LINEAR_Q_DIM + LINEAR_K_DIM, :, :]
                v_conv = conv[LINEAR_Q_DIM + LINEAR_K_DIM :, :, :]
                remapped[f"{layer_prefix}.linear_attn.q_conv.weight"] = q_conv
                remapped[f"{layer_prefix}.linear_attn.k_conv.weight"] = k_conv
                remapped[f"{layer_prefix}.linear_attn.v_conv.weight"] = v_conv
                continue

            # All other keys pass through unchanged
            remapped[new_key] = tensor
            continue

        # Any remaining keys pass through
        remapped[new_key] = tensor

    return remapped

class LazyQwen35MoeStateDict(Mapping):
    """Read one remapped tensor at a time from the original sharded checkpoint."""

    def __init__(self, checkpoint: str | Path):
        from safetensors import safe_open

        self.checkpoint = Path(checkpoint)
        self._safe_open = safe_open
        with (self.checkpoint / "model.safetensors.index.json").open() as stream:
            index = json.load(stream)["weight_map"]
        self._entries = {}
        for source_name, shard_name in index.items():
            if "visual" in source_name or source_name.startswith("mtp."):
                continue
            remapped = self._remapped_names(source_name)
            for name, selector in remapped:
                if name in self._entries:
                    raise ValueError(f"Duplicate remapped checkpoint key: {name}")
                self._entries[name] = (source_name, shard_name, selector)

    @staticmethod
    def _remapped_names(source_name):
        name = source_name
        for prefix in ("model.language_model.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        if name == "embed_tokens.weight":
            return (("tok_embeddings.weight", None),)
        if source_name == "lm_head.weight":
            return (("output.weight", None),)
        if name.endswith(".linear_attn.in_proj_qkv.weight"):
            return ((name.replace(".linear_attn.in_proj_qkv.weight", ".linear_attn.qkv_proj.weight"), None),)
        if name.endswith(".linear_attn.conv1d.weight"):
            base = name.removesuffix("conv1d.weight")
            return (
                (base + "q_conv.weight", "q"),
                (base + "k_conv.weight", "k"),
                (base + "v_conv.weight", "v"),
            )
        return ((name, None),)

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, name):
        source_name, shard_name, selector = self._entries[name]
        with self._safe_open(str(self.checkpoint / shard_name), framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(source_name)
        if selector is None:
            return tensor
        q_end = LINEAR_Q_DIM
        k_end = q_end + LINEAR_K_DIM
        if selector == "q":
            return tensor[:q_end]
        if selector == "k":
            return tensor[q_end:k_end]
        return tensor[k_end:]


def prepare_qwen35_moe_layer(mlp_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert one HF Qwen3.5 MoE MLP to the local TT canonical orientation.

    HF stores linear weights as ``[out, in]`` and packs routed gate/up along the
    output axis. The local runtime consumes ``[in, out]``:

    * router: ``[hidden, experts]``
    * routed gate/up: ``[experts, hidden, 2 * intermediate]``
    * routed down: ``[experts, intermediate, hidden]``
    * shared gate/up and down use the same trailing orientations
    * shared-expert gate: ``[hidden, 1]``
    """
    required = {
        "gate.weight",
        "experts.gate_up_proj",
        "experts.down_proj",
        "shared_expert.gate_proj.weight",
        "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight",
        "shared_expert_gate.weight",
    }
    missing = sorted(required.difference(mlp_state))
    if missing:
        raise KeyError(f"Missing Qwen MoE weights: {missing}")

    router = mlp_state["gate.weight"].T
    gate_up = mlp_state["experts.gate_up_proj"].transpose(-2, -1)
    down = mlp_state["experts.down_proj"].transpose(-2, -1)
    shared_gate = mlp_state["shared_expert.gate_proj.weight"].T
    shared_up = mlp_state["shared_expert.up_proj.weight"].T
    shared_gate_up = torch.cat((shared_gate, shared_up), dim=-1)
    shared_down = mlp_state["shared_expert.down_proj.weight"].T
    shared_expert_gate = mlp_state["shared_expert_gate.weight"].T

    experts, hidden, packed_intermediate = gate_up.shape
    intermediate = packed_intermediate // 2
    expected = {
        "router": (hidden, experts),
        "down": (experts, intermediate, hidden),
        "shared_gate_up": (hidden, packed_intermediate),
        "shared_down": (intermediate, hidden),
        "shared_expert_gate": (hidden, 1),
    }
    tensors = {
        "router": router,
        "gate_up": gate_up,
        "down": down,
        "shared_gate_up": shared_gate_up,
        "shared_down": shared_down,
        "shared_expert_gate": shared_expert_gate,
    }
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensors[name].shape)} must be {shape}")
    return tensors


def load_qwen36_mtp_state_dict(model_path) -> Dict[str, torch.Tensor]:
    """Load only the checkpoint's small MTP subtree from sharded safetensors."""
    from safetensors import safe_open

    model_path = Path(model_path)
    with open(model_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    file_to_keys: Dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if key.startswith("mtp."):
            file_to_keys.setdefault(filename, []).append(key)

    mtp: Dict[str, torch.Tensor] = {}
    for filename, keys in file_to_keys.items():
        with safe_open(str(model_path / filename), framework="pt") as sf:
            for key in keys:
                mtp[key] = sf.get_tensor(key)
    if not mtp:
        raise ValueError(f"Qwen3.5 checkpoint at {model_path} has no mtp.* weights")
    return mtp


def is_fp8_checkpoint(model_path) -> bool:
    """True when the checkpoint dir holds block-wise FP8 safetensors.

    Detected by a ``*.weight_scale_inv`` entry in the safetensors index (the
    per-block dequant scales that accompany float8_e4m3fn weights). Such a
    checkpoint cannot be loaded via AutoModelForCausalLM here; use
    ``load_qwen36_state_dict_fp8`` instead.
    """
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    except (KeyError, ValueError, OSError):
        return False
    return any(k.endswith(".weight_scale_inv") for k in weight_map)


def load_qwen36_state_dict_fp8(model_path) -> Dict[str, torch.Tensor]:
    """Load Qwen3.5 FP8 weights: block-wise dequant + minimal key remap.

    Produces the SAME internal key scheme as ``remap_qwen36_state_dict`` for the
    shared/simple weights (``layers.N.mlp.*``, ``layers.N.self_attn.*``,
    ``input_layernorm`` / ``post_attention_layernorm``, ``tok_embeddings``,
    ``norm``, ``output``) so ``layer.py``'s substate extraction is unchanged.

    The one deliberate difference vs the single-device remap: GDN ``linear_attn.*``
    projections are kept RAW (fused ``in_proj_qkv``, fused ``conv1d``, plus
    ``in_proj_z`` / ``in_proj_a`` / ``in_proj_b`` / ``out_proj`` / ``A_log`` /
    ``dt_bias`` / ``norm.weight``) — NOT split or renamed — so the tensor-parallel
    GDN weight-prep helpers (prepare_gdn_qkv / prepare_conv_taps) can reorder and
    shard them per device. The TP module loaders branch on this raw layout.
    """
    from safetensors import safe_open

    from models.demos.blackhole.qwen36.tt.tp_common import dequant_fp8_block

    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    file_to_keys: Dict[str, list] = {}
    for key, filename in weight_map.items():
        file_to_keys.setdefault(filename, []).append(key)

    raw: Dict[str, torch.Tensor] = {}
    for filename, keys in file_to_keys.items():
        with safe_open(str(model_path / filename), framework="pt") as sf:
            present = set(sf.keys())
            for key in keys:
                if key in present:
                    raw[key] = sf.get_tensor(key)

    # Dequantize FP8 (skip the scale tensors themselves)
    dequantized: Dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if key.endswith(".weight_scale_inv"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale_inv"
            dequantized[key] = (
                dequant_fp8_block(tensor, raw[scale_key]) if scale_key in raw else tensor.to(torch.bfloat16)
            )
        else:
            dequantized[key] = tensor

    state_dict: Dict[str, torch.Tensor] = {}
    for key, tensor in dequantized.items():
        if "visual" in key or key.startswith("mtp"):
            continue
        short = key
        for prefix in ("model.language_model.", "model."):
            if short.startswith(prefix):
                short = short[len(prefix) :]
                break
        if "embed_tokens" in short:
            state_dict["tok_embeddings.weight"] = tensor
        elif key == "lm_head.weight" or short == "lm_head.weight":
            state_dict["output.weight"] = tensor
        else:
            # Everything else (layers.N.mlp.*, self_attn.*, linear_attn.* RAW,
            # input_layernorm/post_attention_layernorm, norm) passes through.
            state_dict[short] = tensor

    return state_dict
