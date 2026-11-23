# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM2.5-24B-A2B (Lfm2MoeForCausalLM) architecture constants, read from the HF config.json."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Development default only (serving passes the checkpoint directory): the BF16 snapshot when present, else the quant
# checkpoint (it carries the same config.json / tokenizer files), under LFM25_DEV_ROOT.
_DEV_ROOT = Path(os.environ.get("LFM25_DEV_ROOT", "lfm25"))
HF_DIR = next((p for p in (_DEV_ROOT / "hf", _DEV_ROOT / "quant/ckpt") if (p / "config.json").exists()),
              _DEV_ROOT / "hf")


@dataclass
class Lfm2Config:
    hidden: int = 2048
    vocab: int = 128000
    n_layers: int = 40
    layer_types: list = field(default_factory=list)
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 64
    rope_theta: float = 1e7
    norm_eps: float = 1e-5
    conv_l_cache: int = 3
    intermediate: int = 11776
    n_dense: int = 2
    n_experts: int = 64
    top_k: int = 4
    moe_intermediate: int = 1536
    norm_topk_prob: bool = True
    routed_scaling: float = 1.0
    bos: int = 124894
    eos: int = 124900
    pad: int = 124893

    @classmethod
    def from_hf(cls, hf_dir: str | Path = HF_DIR) -> "Lfm2Config":
        c = json.loads((Path(hf_dir) / "config.json").read_text())
        rope = c.get("rope_parameters") or {}
        return cls(
            hidden=c["hidden_size"], vocab=c["vocab_size"], n_layers=c["num_hidden_layers"],
            layer_types=list(c["layer_types"]), n_heads=c["num_attention_heads"],
            n_kv_heads=c["num_key_value_heads"], head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
            rope_theta=float(rope.get("rope_theta", c.get("rope_theta", 1e7))), norm_eps=c["norm_eps"],
            conv_l_cache=c["conv_L_cache"], intermediate=c["intermediate_size"], n_dense=c["num_dense_layers"],
            n_experts=c["num_experts"], top_k=c["num_experts_per_tok"], moe_intermediate=c["moe_intermediate_size"],
            norm_topk_prob=c["norm_topk_prob"], routed_scaling=c["routed_scaling_factor"],
            bos=c["bos_token_id"], eos=c["eos_token_id"], pad=c["pad_token_id"],
        )

    def is_attention(self, layer: int) -> bool:
        return self.layer_types[layer] == "full_attention"

    def is_moe(self, layer: int) -> bool:
        return layer >= self.n_dense

    @property
    def attention_layers(self) -> list[int]:
        return [i for i in range(self.n_layers) if self.is_attention(i)]

    @property
    def conv_layers(self) -> list[int]:
        return [i for i in range(self.n_layers) if not self.is_attention(i)]
