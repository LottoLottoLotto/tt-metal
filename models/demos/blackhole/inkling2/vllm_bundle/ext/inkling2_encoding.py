# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2 chat encoding: OpenAI-style messages + tools -> prompt token ids (pure; no vLLM imports).

Port of the Inkling2 path of upstream vLLM's `vllm/renderers/inkling_encoding.py` (`render_inkling2_messages`,
vllm-project/vllm main, Apache-2.0; PR #69028 routes model_type "inkling2_loop" to it), text-only:

- blocks `<|message_ROLE|>[author]<|content_KIND|>payload<|end_message|>`; an assistant turn ends with
  `<|content_model_end_sampling|>`; tools are declared first in one `<|message_system|>tool_declare<|content_xml|>`
  block (canonical JSON, keys sorted); `Thinking effort level: E` is a system text block placed before the first
  non-system message (default 0.9; names none/minimal/low/medium/high/xhigh/max or a number in [0, 0.99]).
- NO generation prefix: the model emits its own `<|message_model|>` (same as the HF chat template, which has no
  add_generation_prompt branch).
- payload text never encodes to framing tokens (`split_special_tokens`): user text that spells "<|end_message|>" stays
  text.  The HF chat template tokenizes its rendered string, so it differs there (and only there, for well-formed
  conversations; tests/test_render.py checks equality on the cases that matter).
- image / audio / video parts are rejected (Inkling-2-Mini is served text-only).

Deviations from upstream: no tool_choice / parallel_tool_calls instruction text (this vLLM's ChatParams does not carry
them; the HF template does not render them either); images rejected instead of emitting `<|content_image|>`.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol

MESSAGE_USER = "<|message_user|>"
MESSAGE_MODEL = "<|message_model|>"
MESSAGE_SYSTEM = "<|message_system|>"
MESSAGE_TOOL = "<|message_tool|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_MODEL_END_SAMPLING = "<|content_model_end_sampling|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_TOOL_ERROR = "<|content_tool_error|>"
CONTENT_XML = "<|content_xml|>"
CONTENT_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
END_MESSAGE = "<|end_message|>"

SPECIAL_TOKENS = (MESSAGE_USER, MESSAGE_MODEL, MESSAGE_SYSTEM, MESSAGE_TOOL, CONTENT_TEXT,
                  CONTENT_MODEL_END_SAMPLING, CONTENT_THINKING, CONTENT_TOOL_ERROR, CONTENT_XML,
                  CONTENT_INVOKE_TOOL_JSON, END_MESSAGE)

ROLE_MESSAGE_TOKENS: dict[str, str] = {
    "user": MESSAGE_USER,
    "assistant": MESSAGE_MODEL,
    "system": MESSAGE_SYSTEM,
    "developer": MESSAGE_SYSTEM,  # the developer role folds into the system role
    "tool": MESSAGE_TOOL,
}

_CONTENT_KIND_TOKENS: dict[str, str] = {
    "text": CONTENT_TEXT,
    "thinking": CONTENT_THINKING,
    "tool_error": CONTENT_TOOL_ERROR,
    "xml": CONTENT_XML,
    "invoke_tool_json": CONTENT_INVOKE_TOOL_JSON,
}

NAMED_REASONING_EFFORT = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99,
                          "max": 0.99}
DEFAULT_REASONING_EFFORT = 0.9
_MAX_REASONING_EFFORT = 0.99

_MEDIA_PART_TYPES = frozenset({"image", "input_image", "image_url", "image_pil", "image_embeds", "audio",
                               "input_audio", "audio_url", "video", "video_url"})
TEXT_ONLY_MESSAGE = "Inkling-2-Mini is served text-only here: image, audio and video inputs are not supported"


class TextTokenizer(Protocol):
    def encode_text(self, text: str) -> list[int]: ...

    def encode_special(self, token: str) -> int: ...


class HfTextTokenizer:
    """TextTokenizer over an HF tokenizer: special ids from its vocabulary, payload text encoded with
    split_special_tokens (never to framing tokens) and without BOS/EOS."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        missing = [t for t in SPECIAL_TOKENS if t not in vocab]
        if missing:
            raise ValueError(f"Inkling-2 tokenizer is missing special tokens: {missing}")
        self._special_ids = {t: vocab[t] for t in SPECIAL_TOKENS}

    def encode_text(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)

    def encode_special(self, token: str) -> int:
        return self._special_ids[token]


def resolve_reasoning_effort(value: object) -> float:
    """None -> 0.9; a name (NAMED_REASONING_EFFORT) or a number / numeric string in [0, 0.99]."""
    if value is None:
        return DEFAULT_REASONING_EFFORT
    if isinstance(value, str):
        if value in NAMED_REASONING_EFFORT:
            return NAMED_REASONING_EFFORT[value]
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"invalid Inkling reasoning_effort: {value!r}") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid Inkling reasoning_effort: {value!r}")
    return float(value)


def reject_media(messages: Sequence[Mapping[str, Any]]) -> None:
    """ValueError (HTTP 400) for any image / audio / video content part."""
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, (str, bytes)) or content is None:
            continue
        for part in content:
            if isinstance(part, Mapping) and part.get("type") in _MEDIA_PART_TYPES:
                raise ValueError(f"{TEXT_ONLY_MESSAGE} (content part type {part.get('type')!r})")


def render_messages(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: TextTokenizer,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    reasoning_effort: float = DEFAULT_REASONING_EFFORT,
    continue_final_message: bool = False,
) -> list[int]:
    """Render chat messages to Inkling-2 input ids.  With continue_final_message, a final assistant message holding only
    a text string is left open (`<|message_model|><|content_text|>` + its text) so generation continues it."""
    reject_media(messages)
    messages = list(messages)
    assistant_prefix = None
    if continue_final_message and messages:
        assistant_prefix = continuation_prefix(messages[-1])
        if assistant_prefix is not None:
            messages.pop()

    input_ids: list[int] = []
    declared_tools = _collect_tools(messages, tools)
    if declared_tools:
        _append_message(input_ids, tokenizer, "system", "xml", _tool_declare_json(declared_tools),
                        author_name="tool_declare")

    effort_text: str | None = _effort_text(reasoning_effort)
    tool_call_id_to_name: dict[str, str] = {}
    for message in messages:
        role = _expect_role(message)
        if effort_text is not None and role not in {"system", "developer"}:
            _append_message(input_ids, tokenizer, "system", "text", effort_text)
            effort_text = None

        author_name = None
        if role == "tool":
            name = message.get("name")
            author_name = (str(name) if name
                           else tool_call_id_to_name.get(str(message.get("tool_call_id") or "")))
        blocks = _blocks(message, role, tool_call_id_to_name)
        for kind, text, block_author in blocks:
            _append_message(input_ids, tokenizer, role, kind, text, author_name=block_author or author_name)
        if role == "assistant" and blocks:
            input_ids.append(tokenizer.encode_special(CONTENT_MODEL_END_SAMPLING))

    if effort_text is not None:
        _append_message(input_ids, tokenizer, "system", "text", effort_text)

    if assistant_prefix is not None:
        input_ids.append(tokenizer.encode_special(MESSAGE_MODEL))
        input_ids.append(tokenizer.encode_special(CONTENT_TEXT))
        input_ids.extend(tokenizer.encode_text(assistant_prefix))
    return input_ids


def continuation_prefix(message: Mapping[str, Any]) -> str | None:
    """The text a continue_final_message request extends: a plain-string assistant message without tool calls or
    reasoning; None means the message renders as a closed turn."""
    content = message.get("content")
    if (message.get("role") != "assistant" or not isinstance(content, str) or message.get("tool_calls")
            or _assistant_reasoning(message)):
        return None
    return content


def _collect_tools(messages, tools) -> list[Mapping[str, Any]]:
    """Request-level plus per-developer-message tools, declared in one leading system block."""
    all_tools = list(tools or [])
    for message in messages:
        if message.get("role") == "developer":
            all_tools.extend(message.get("tools") or [])
    return all_tools


def _effort_text(reasoning_effort: float) -> str:
    if not 0.0 <= reasoning_effort <= _MAX_REASONING_EFFORT:
        raise ValueError(f"Inkling reasoning_effort must be in [0.0, 0.99], got {reasoning_effort!r}")
    millieffort = round(reasoning_effort * 1000)
    effort = round((millieffort / 1000) * 100) / 100
    return f"Thinking effort level: {effort:g}"


def _append_message(input_ids, tokenizer, role, kind, text, *, author_name=None) -> None:
    input_ids.append(tokenizer.encode_special(ROLE_MESSAGE_TOKENS[role]))
    if author_name:
        input_ids.extend(tokenizer.encode_text(author_name))
    if kind not in _CONTENT_KIND_TOKENS:
        raise ValueError(f"unsupported Inkling render part kind: {kind!r}")
    input_ids.append(tokenizer.encode_special(_CONTENT_KIND_TOKENS[kind]))
    input_ids.extend(tokenizer.encode_text(text))
    input_ids.append(tokenizer.encode_special(END_MESSAGE))


def _assistant_reasoning(message: Mapping[str, Any]) -> Any:
    reasoning = message.get("reasoning")
    return message.get("reasoning_content") if reasoning is None else reasoning


def _blocks(message, role, tool_call_id_to_name) -> list[tuple[str, str, str | None]]:
    content = message.get("content")
    if role != "assistant":
        return [(kind, text, None) for kind, text in _iter_parts(content)]

    reasoning = _assistant_reasoning(message)
    if reasoning is not None and not isinstance(reasoning, str):
        raise TypeError("assistant reasoning_content must be a string")
    tool_calls = message.get("tool_calls") or []
    if content == "" and (tool_calls or reasoning):
        content = None

    blocks: list[tuple[str, str, str | None]] = [("thinking", reasoning, None)] if reasoning else []
    blocks.extend((kind, text, None) for kind, text in _iter_parts(content))
    for tool_call in tool_calls:
        name, args = _tool_call_name_and_args(tool_call)
        tool_call_id = _as_mapping(tool_call).get("id")
        if tool_call_id:
            tool_call_id_to_name[str(tool_call_id)] = name
        args = _tml_json_loads(json.dumps(args, allow_nan=False))
        args_json = _tml_json_dumps({key: args[key] for key in sorted(args)}, sort_keys=False)
        blocks.append(("invoke_tool_json", f'{{"name":{_tml_json_dumps(name)},"args":{args_json}}}', name))
    return blocks


def _iter_parts(content: Any) -> Iterator[tuple[str, str]]:
    if content is None:
        return
    if isinstance(content, str):
        yield ("text", content)
        return
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        raise TypeError("message content must be a string or a sequence of parts")
    for part in content:
        if isinstance(part, str):
            yield ("text", part)
            continue
        if not isinstance(part, Mapping):
            raise TypeError(f"content part must be mapping, got {type(part).__name__}")
        ptype = part.get("type")
        if ptype in (None, "text", "input_text"):
            yield ("text", part.get("text") or "")
        elif ptype == "thinking":
            yield ("thinking", part.get("thinking") or "")
        elif ptype == "tool_error":
            yield ("tool_error", part.get("text") or "")
        elif ptype in _MEDIA_PART_TYPES:
            raise ValueError(f"{TEXT_ONLY_MESSAGE} (content part type {ptype!r})")
        else:
            raise ValueError(f"unsupported Inkling2 content part type: {ptype!r}")


def _expect_role(message: Mapping[str, Any]) -> str:
    role = message.get("role")
    if role not in ROLE_MESSAGE_TOKENS:
        raise ValueError(f"unsupported Inkling message role {role!r}; expected one of {sorted(ROLE_MESSAGE_TOKENS)}")
    return str(role)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(f"expected mapping, got {type(value).__name__}")


def _tool_call_name_and_args(tool_call_value: Any) -> tuple[str, Mapping[str, Any]]:
    tool_call = _as_mapping(tool_call_value)
    function = _as_mapping(tool_call.get("function", {}))
    name = function.get("name")
    if not isinstance(name, str):
        raise TypeError("tool call function name must be a string")
    raw_args = function.get("arguments") or {}
    args = (json.loads(raw_args) if raw_args.strip() else {}) if isinstance(raw_args, str) else raw_args
    if not isinstance(args, Mapping):
        raise TypeError("tool call function arguments must decode to an object")
    return name, args


def _tool_declare_json(tools: Sequence[Mapping[str, Any]]) -> str:
    tool_specs = []
    for tool_value in tools:
        tool = _as_mapping(tool_value)
        function = {key: value for key, value in _as_mapping(tool.get("function", {})).items() if value is not None}
        parameters = function.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("tool parameters must be a JSON object")
        tool_specs.append({
            "description": function.get("description", ""),
            "name": function["name"],
            "parameters": _tml_json_loads(json.dumps(parameters, ensure_ascii=False, allow_nan=False)),
            "type": tool.get("type") or "function",
        })
    return _tml_json_dumps(tool_specs, legacy_floats=False)


def _tml_json_loads(text: str) -> Any:
    """Parse JSON as the Inkling2 reference does (64-bit ints, finite floats)."""

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"Invalid JSON constant: {value}")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON number out of range")
        return number

    def bounded_int(value: str) -> int | float:
        number = int(value)
        if value == "-0" or not -(1 << 63) <= number < 1 << 64:
            return finite_float(value)
        return number

    return json.loads(text, parse_constant=invalid_constant, parse_float=finite_float, parse_int=bounded_int)


def _tml_json_dumps(value: Any, *, legacy_floats: bool = True, sort_keys: bool = True) -> str:
    """Compact JSON with the Inkling2 reference float spelling."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return "{" + ",".join(
            _tml_json_dumps(key) + ":" + _tml_json_dumps(value[key], legacy_floats=legacy_floats, sort_keys=sort_keys)
            for key in (sorted(value) if sort_keys else value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_tml_json_dumps(item, legacy_floats=legacy_floats, sort_keys=sort_keys)
                              for item in value) + "]"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        result = repr(value)
        if "e" not in result:
            return result
        mantissa, exponent_text = result.split("e")
        exponent = int(exponent_text)
        if -5 <= exponent < 16:
            negative = mantissa.startswith("-")
            digits = mantissa.lstrip("-").replace(".", "")
            decimal = exponent + 1
            if decimal <= 0:
                result = "0." + "0" * -decimal + digits
            elif decimal >= len(digits):
                result = digits + "0" * (decimal - len(digits)) + ".0"
            else:
                result = digits[:decimal] + "." + digits[decimal:]
            return ("-" if negative else "") + result
        sign = "+" if exponent >= 0 and not legacy_floats else ""
        return mantissa + "e" + sign + str(exponent)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
