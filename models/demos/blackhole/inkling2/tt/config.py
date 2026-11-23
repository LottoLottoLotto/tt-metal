# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini (Inkling2Loop text model) configuration, parsed from the HF snapshot's config.json (text_config).

Shared by every runtime module (`cfg` argument of Inkling2Attention / Inkling2MoE). Layer schedule:
layers 0..5 once, the loop layers 6..17 `loop_passes` (3) times, layers 18..23 once = `applications()` (48 entries of
(layer, pass_idx)); pass_idx is 0 outside the loop.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# the checkpoint directory (config.json, safetensors): MODEL_WEIGHTS_DIR, else ./checkpoint (the release repo layout)
HF_DIR = os.environ.get("MODEL_WEIGHTS_DIR") or "checkpoint"


@dataclass(frozen=True)
class Inkling2Config:
    hidden: int = 2304
    n_layers: int = 24
    vocab: int = 201024                 # embedding rows (padded vocabulary)
    unpadded_vocab: int = 200058        # unembedding rows used by the reference (lm_head)
    n_heads: int = 24
    n_kv_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    sliding_window: int = 1024
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    qk_l2_clip: float = 22.627416997969522
    rel_l2_clip: float = 8.0
    eps: float = 1e-6
    conv_kernel: int = 4                # sconv_kernel_size (post-attn / post-MLP / K / V short convs)
    n_experts: int = 128
    top_k: int = 8
    moe_intermediate: int = 1152
    route_scale: float = 8.0
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    logit_softcap: float = 30.0
    layers_per_block: int = 6           # n_layers_per_attn_res_block
    loop_start: int = 6
    loop_end: int = 18
    loop_passes: int = 3
    gate_rank: int = 64
    eos_token_id: int = 200006
    local_layers: tuple = field(default_factory=lambda: tuple(i for i in range(24) if (i + 1) % 6))

    @classmethod
    def from_hf(cls, hf_dir: str | Path = HF_DIR) -> "Inkling2Config":
        raw = json.loads((Path(hf_dir) / "config.json").read_text())
        t = raw.get("text_config", raw)
        loop = t.get("loop_config", {})
        return cls(
            hidden=t["hidden_size"], n_layers=t["num_hidden_layers"], vocab=t["vocab_size"],
            unpadded_vocab=t["unpadded_vocab_size"], n_heads=t["num_attention_heads"],
            n_kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"], d_rel=t["d_rel"],
            rel_extent=t["rel_extent"], sliding_window=t["sliding_window_size"],
            log_scaling_n_floor=t["log_scaling_n_floor"], log_scaling_alpha=t["log_scaling_alpha"],
            qk_l2_clip=t["qk_l2_clip_threshold"], rel_l2_clip=t["rel_l2_clip_threshold"], eps=t["rms_norm_eps"],
            conv_kernel=t["sconv_kernel_size"], n_experts=t["n_routed_experts"], top_k=t["num_experts_per_tok"],
            moe_intermediate=t["intermediate_size"], route_scale=t["route_scale"],
            mlp_gate_softcap=t["mlp_gate_softcap"], mlp_up_softcap=t["mlp_up_softcap"],
            logit_softcap=t["final_logit_softcapping"], layers_per_block=t["n_layers_per_attn_res_block"],
            loop_start=loop.get("core_start", 6), loop_end=loop.get("core_end", 18),
            loop_passes=loop.get("num_passes", 3), gate_rank=loop.get("gate_rank", 64),
            eos_token_id=raw.get("eos_token_id", t.get("eos_token_id", 200006)),
            local_layers=tuple(t["local_layer_ids"]),
        )

    # ------------------------------------------------------------------ layer kinds
    def is_sliding(self, layer: int) -> bool:
        return layer in self.local_layers

    def is_full(self, layer: int) -> bool:
        return layer not in self.local_layers

    def is_loop(self, layer: int) -> bool:
        return self.loop_start <= layer < self.loop_end

    def is_boundary(self, layer: int) -> bool:
        """Block boundary: the layer input is appended to the block states."""
        return layer % self.layers_per_block == 0

    def passes(self, layer: int) -> int:
        """Number of applications (KV / conv cache slots) of `layer` per token."""
        return self.loop_passes if self.is_loop(layer) else 1

    @property
    def n_loop_layers(self) -> int:
        return self.loop_end - self.loop_start

    def applications(self) -> list[tuple[int, int]]:
        """(layer, pass_idx) in execution order (48 for Inkling-2-Mini)."""
        out = [(i, 0) for i in range(self.loop_start)]
        for p in range(self.loop_passes):
            out += [(i, p) for i in range(self.loop_start, self.loop_end)]
        out += [(i, 0) for i in range(self.loop_end, self.n_layers)]
        return out

    @property
    def padded_unembed(self) -> int:
        """Unembedding columns on the device (unpadded vocab rounded up to a tile)."""
        return -(-self.unpadded_vocab // 32) * 32
