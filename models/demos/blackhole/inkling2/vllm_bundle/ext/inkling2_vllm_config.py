# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini for vLLM without remote code: vLLM general plugin `inkling2_config` (entry point in
inkling2_vllm_config-1.0.dist-info/entry_points.txt; installed into the image's site-packages with inkling2_encoding.py).

`register()` (vLLM calls it in every process before it parses the model config):
- config: model_type "inkling2_loop" (config.json, nested text_config / vision_config / mtp_config; transformers 5.12.1
  does not know it) -> Inkling2LoopConfig in vLLM's config registry and transformers' AutoConfig.  vLLM reads only the
  text config's shape fields; the TT runtime reads config.json itself (models.demos.blackhole.inkling2 tt/config.py).
  `sliding_window` is None for vLLM (the runtime owns every cache; vLLM's KV pages are not used) and
  max_position_embeddings is the text config's model_max_length (1,048,576).
- architecture: Inkling2LoopForConditionalGeneration (+ its TT-prefixed name) -> the TT adapter (text-only class: vLLM
  sees no multimodal interface).  vllm_tt_plugin's EXTRA_MODELS_DIR bundle registers the TT name too.
- tokenizer mode / renderer "inkling2" (serve with --tokenizer-mode inkling2): the HF tokenizer, and chat messages
  rendered straight to token ids by inkling2_encoding.render_messages (the Inkling2 reference rendering); image / audio
  / video parts are rejected with a 400 before anything is fetched.
"""
from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

ADAPTER = "models.demos.blackhole.inkling2.tt.inkling2_vllm:Inkling2ForCausalLM"
ARCH = "Inkling2LoopForConditionalGeneration"
MODE = "inkling2"


class Inkling2LoopTextConfig(PretrainedConfig):
    model_type = "inkling2_loop_text"
    base_config_key = "text_config"

    def __init__(self, vocab_size: int = 201024, hidden_size: int = 2304, num_hidden_layers: int = 24,
                 num_attention_heads: int = 24, num_key_value_heads: int = 8, head_dim: int = 128,
                 model_max_length: int = 1048576, sliding_window_size: int | None = 1024,
                 rms_norm_eps: float = 1e-6, unpadded_vocab_size: int = 200058, **kwargs: Any):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.model_max_length = model_max_length
        self.max_position_embeddings = model_max_length
        self.rms_norm_eps = rms_norm_eps
        self.unpadded_vocab_size = unpadded_vocab_size
        # the local layers' window, kept under the checkpoint's name; vLLM sees no sliding window (see module doc)
        self.sliding_window_size = sliding_window_size
        self.sliding_window = None
        kwargs.pop("torch_dtype", None)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


class Inkling2LoopConfig(PretrainedConfig):
    model_type = "inkling2_loop"
    sub_configs = {"text_config": Inkling2LoopTextConfig}

    def __init__(self, text_config: dict | Inkling2LoopTextConfig | None = None, vision_config: dict | None = None,
                 mtp_config: dict | None = None, eos_token_id: int | list[int] | None = 200006, **kwargs: Any):
        if not isinstance(text_config, Inkling2LoopTextConfig):
            text_config = Inkling2LoopTextConfig(**(text_config or {}))
        self.text_config = text_config
        self.vision_config = vision_config  # unused: images are rejected
        self.mtp_config = mtp_config  # the block drafter's config (mtp.safetensors); not used by vLLM
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(eos_token_id=eos_token_id, **kwargs)

    def get_text_config(self, *args, **kwargs) -> Inkling2LoopTextConfig:
        return self.text_config


def _renderer_cls():
    from vllm.entrypoints.chat_utils import parse_chat_messages, parse_chat_messages_async
    from vllm.renderers.hf import HfRenderer
    from vllm.renderers.inputs.preprocess import parse_dec_only_prompt

    import inkling2_encoding as enc

    class Inkling2Renderer(HfRenderer):
        """Chat messages -> Inkling-2 prompt token ids (inkling2_encoding.render_messages).  chat_template_kwargs:
        reasoning_effort (also the request's top-level reasoning_effort), continue_final_message, tools (vLLM passes
        the request's tools there).  Any custom chat template is ignored."""

        def __init__(self, config, tokenizer_kwargs):
            super().__init__(config, tokenizer_kwargs)
            model_type = getattr(config.hf_config, "model_type", None)
            if model_type != "inkling2_loop":
                raise ValueError(f"--tokenizer-mode {MODE} renders Inkling-2 chats; model_type is {model_type!r}")
            self._text_tokenizer = enc.HfTextTokenizer(self.get_tokenizer())

        def _render(self, messages, params) -> list[int]:
            kwargs = params.chat_template_kwargs or {}
            try:
                return enc.render_messages(
                    messages, self._text_tokenizer, tools=kwargs.get("tools"),
                    reasoning_effort=enc.resolve_reasoning_effort(kwargs.get("reasoning_effort")),
                    continue_final_message=bool(kwargs.get("continue_final_message")))
            except ValueError:
                raise
            except (TypeError, KeyError, AttributeError) as e:  # malformed request content -> HTTP 400
                raise ValueError(f"invalid Inkling-2 chat request: {e}") from e

        @staticmethod
        def _materialize(messages):
            """Copies with list tool_calls: a request's are a one-shot pydantic ValidatorIterator, which vLLM's
            parse_chat_messages would consume before the rendering reads them."""
            return [{**m, "tool_calls": list(m["tool_calls"])} if m.get("tool_calls") is not None else m
                    for m in messages]

        def render_messages(self, messages, params):
            messages = self._materialize(messages)
            enc.reject_media(messages)
            conversation, _, _ = parse_chat_messages(messages, self.config, content_format="string")
            return conversation, parse_dec_only_prompt(self._render(messages, params))

        async def render_messages_async(self, messages, params):
            messages = self._materialize(messages)
            enc.reject_media(messages)
            conversation, _, _ = await parse_chat_messages_async(messages, self.config, content_format="string")
            return conversation, parse_dec_only_prompt(self._render(messages, params))

    return Inkling2Renderer


def __getattr__(name: str):
    # vLLM's renderer registry imports "inkling2_vllm_config.Inkling2Renderer" lazily (after vLLM is importable)
    if name == "Inkling2Renderer":
        cls = _renderer_cls()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


def register() -> None:
    """vLLM general plugin; idempotent (vLLM may call it several times per process)."""
    from transformers import AutoConfig
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY["inkling2_loop"] = Inkling2LoopConfig
    for model_type, cls in (("inkling2_loop", Inkling2LoopConfig), ("inkling2_loop_text", Inkling2LoopTextConfig)):
        try:
            AutoConfig.register(model_type, cls)
        except ValueError:  # already registered in this process
            pass
    supported = ModelRegistry.get_supported_archs()
    for arch in (ARCH, "TT" + ARCH):
        if arch not in supported:
            ModelRegistry.register_model(arch, ADAPTER)
    if MODE not in TokenizerRegistry.tokenizers:
        TokenizerRegistry.register(MODE, "vllm.tokenizers.hf", "CachedHfTokenizer")
    if MODE not in RENDERER_REGISTRY.renderers:
        RENDERER_REGISTRY.register(MODE, "inkling2_vllm_config", "Inkling2Renderer")
