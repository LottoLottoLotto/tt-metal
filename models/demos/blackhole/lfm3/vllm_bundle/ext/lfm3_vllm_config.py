# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM3 config + architecture registration for vLLM without remote code (vLLM general plugin `lfm3_config`, entry
point declared in this directory's pyproject.toml, installed as the package lfm3-vllm-config).

The checkpoint's config.json names model_type "lfm3" with an auto_map to configuration_lfm3.py, which needs a newer
transformers (and Python >= 3.11) than the image's 5.12.1 / 3.10.  vLLM loads general plugins in every process before
it parses the model config; `register()` adds this class to vLLM's config registry (vLLM then builds the config with
it, no --trust-remote-code) and to transformers' AutoConfig (anything else that resolves the config).  Fields and
defaults follow configuration_lfm3.py; the TT runtime reads config.json itself (tt/config.py).
"""
from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

ADAPTER = "models.demos.blackhole.lfm3.tt.lfm3_vllm:Lfm3ForCausalLM"  # the TT Generator adapter (models.demos.blackhole.lfm3)


class Lfm3Config(PretrainedConfig):
    model_type = "lfm3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 128_000,
        hidden_size: int = 2048,
        intermediate_size: int = 7680,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 128_000,
        initializer_range: float = 0.02,
        norm_eps: float = 0.00001,
        use_cache: bool = True,
        pad_token_id: int | None = 124_893,
        bos_token_id: int | None = 124_894,
        eos_token_id: int | list[int] | None = 124_900,
        tie_word_embeddings: bool = True,
        rope_parameters: dict[str, Any] | None = None,
        conv_bias: bool = False,
        conv_L_cache: int = 7,
        layer_types: list[str] | None = None,
        head_dim: int | None = None,
        sliding_window: int | None = 256,
        use_sandwich_norm: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        if rope_parameters is None:
            rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            }
        self.rope_parameters = rope_parameters
        self.conv_bias = conv_bias
        self.conv_L_cache = conv_L_cache
        if layer_types is None:
            layer_types = ["full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv"
                           for i in range(num_hidden_layers)]
        if len(layer_types) != num_hidden_layers:
            raise ValueError(f"layer_types has {len(layer_types)} entries for {num_hidden_layers} layers")
        self.layer_types = layer_types
        # The 256-token window of the sliding_attention layers is kept as `attention_window`, and `sliding_window` is
        # None for vLLM: the TT runtime reads the window from config.json itself and declares one full-length paged KV
        # spec for all 11 attention layers, and vllm_tt_plugin would otherwise turn prefix caching off for a
        # "sliding-window model" (the model's exact prefix resume restores full KV pages + conv state).
        self.attention_window = sliding_window
        self.sliding_window = None
        self.use_sandwich_norm = use_sandwich_norm
        tie_word_embeddings = kwargs.pop("tie_embedding", tie_word_embeddings)
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)


def register() -> None:
    """vLLM general plugin: idempotent (vLLM may call it several times per process).  Also registers the plain
    architecture name with the TT adapter: vLLM validates config.json's `architectures` (Lfm3ForCausalLM, unknown to
    this vLLM) before vllm_tt_plugin rewrites it to the TT-prefixed name its EXTRA_MODELS_DIR bundle registers."""
    from transformers import AutoConfig
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY["lfm3"] = Lfm3Config
    try:
        AutoConfig.register("lfm3", Lfm3Config)
    except ValueError:  # already registered in this process
        pass
    if "Lfm3ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Lfm3ForCausalLM", ADAPTER)
