# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM3-1.7B (Lfm3ForCausalLM) architecture constants, read from the HF config.json."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Default for development tools: the BF16 snapshot or a quant checkpoint (it carries the same config.json / tokenizer
# files), LFM3_HF_DIR; serving passes the checkpoint directory explicitly.
HF_DIR = Path(os.environ.get("LFM3_HF_DIR", "hf"))

FULL, SLIDING, CONV = "full_attention", "sliding_attention", "conv"


@dataclass
class Lfm3Config:
    hidden: int = 2048
    vocab: int = 128000
    n_layers: int = 24
    layer_types: list = field(default_factory=list)
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 128
    # per attention layer type: {"rope_theta", "partial_rotary_factor"}
    rope: dict = field(default_factory=dict)
    sliding_window: int = 256
    norm_eps: float = 1e-5
    conv_l_cache: int = 7
    intermediate: int = 7680
    sandwich_norm: bool = True
    bos: int = 124894
    eos: int = 124900
    pad: int = 124893

    @classmethod
    def from_hf(cls, hf_dir: str | Path = HF_DIR) -> "Lfm3Config":
        c = json.loads((Path(hf_dir) / "config.json").read_text())
        rope = {t: {"rope_theta": float(p["rope_theta"]), "partial_rotary_factor": float(p.get("partial_rotary_factor", 1.0))}
                for t, p in c["rope_parameters"].items()}
        for p in c["rope_parameters"].values():
            if p.get("rope_type", "default") != "default":
                raise ValueError(f"rope_type {p['rope_type']} not supported")
        if c.get("conv_bias"):
            raise ValueError("conv_bias not supported")
        return cls(
            hidden=c["hidden_size"], vocab=c["vocab_size"], n_layers=c["num_hidden_layers"],
            layer_types=list(c["layer_types"]), n_heads=c["num_attention_heads"],
            n_kv_heads=c["num_key_value_heads"], head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
            rope=rope, sliding_window=int(c["sliding_window"]), norm_eps=c["norm_eps"],
            conv_l_cache=c["conv_L_cache"], intermediate=c["intermediate_size"],
            sandwich_norm=bool(c.get("use_sandwich_norm", False)),
            bos=c["bos_token_id"], eos=c["eos_token_id"], pad=c["pad_token_id"],
        )

    def is_attention(self, layer: int) -> bool:
        return self.layer_types[layer] in (FULL, SLIDING)

    def is_sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == SLIDING

    @property
    def attention_layers(self) -> list[int]:
        return [i for i in range(self.n_layers) if self.is_attention(i)]

    @property
    def conv_layers(self) -> list[int]:
        return [i for i in range(self.n_layers) if not self.is_attention(i)]
