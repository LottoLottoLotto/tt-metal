# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""vLLM reasoning + tool-call parsers for Inkling-2 (load with --reasoning-parser-plugin / --tool-parser-plugin this
file; select with --reasoning-parser inkling2 --tool-call-parser inkling2).

Output format (every marker is one special token; blocks repeat in any order; the turn ends with
<|content_model_end_sampling|> = eos 200006):

    <|message_model|><|content_thinking|>reasoning<|end_message|>
    <|message_model|><|content_text|>visible text<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
    <|content_model_end_sampling|>

(<|content_invoke_tool_text|> and <|content_tool_error|> blocks render as visible text, like upstream vLLM's
`vllm/parser/inkling.py`.)  The markers are special tokens, so both parsers set skip_special_tokens=False in
adjust_request (the tool parser's is vLLM's own hook; the reasoning parser's is called by
src/docker/vllm-reasoning-adjust-request.patch) and work on the text: a special token arrives whole in a delta.

Reasoning parser: thinking-block payloads -> reasoning.  When the request has tools (tool_choice auto) the rest of the
output is handed to the tool parser as raw text: non-streamed, the non-thinking blocks keep their markers when the
output holds a well-formed tool call (otherwise vLLM uses the parser's content as is, so it is the visible text);
streamed, the reasoning phase ends at the first visible block (is_reasoning_end: a content_text / invoke_tool_* /
tool_error / end-sampling token after the prompt's last message) and that delta's raw remainder becomes the tool
parser's first text, as vLLM's serving layer expects.  Without tools it returns the visible text (markers removed).
Tool parser: visible blocks -> content, invoke_tool_json blocks -> tool calls (arguments = the payload's "args",
JSON), each tool call emitted whole when its block closes; a thinking block after the handoff streams as reasoning.
A tool block cut off by the length limit is dropped when streamed and returned as content when not.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

MESSAGE_MODEL = "<|message_model|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
CONTENT_INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>"
CONTENT_TOOL_ERROR = "<|content_tool_error|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
TURN_BREAKS = ("<|message_user|>", "<|message_system|>", "<|message_tool|>", "<|endoftext|>")

# marker -> role in the block grammar
_KIND = {MESSAGE_MODEL: "model", CONTENT_THINKING: "thinking", CONTENT_TEXT: "text",
         CONTENT_INVOKE_TOOL_JSON: "tool_json", CONTENT_INVOKE_TOOL_TEXT: "text", CONTENT_TOOL_ERROR: "text",
         END_MESSAGE: "end", END_SAMPLING: "stop", **{t: "stop" for t in TURN_BREAKS}}
_SPLIT = re.compile("(" + "|".join(re.escape(m) for m in sorted(_KIND, key=len, reverse=True)) + ")")
_ENDED = (CONTENT_TEXT, CONTENT_INVOKE_TOOL_JSON, CONTENT_INVOKE_TOOL_TEXT, CONTENT_TOOL_ERROR, END_SAMPLING)


class BlockScanner:
    """Incremental scanner of the block grammar.  feed(text) -> events (kind, payload, raw):
    ("loose", text, raw)            text outside any block (or a header that never opened a block)
    ("open:B", name, raw)           block B (thinking / text / tool_json) opens; name = the header's author name;
                                    raw = header + marker
    ("data", text, raw)             block payload
    ("close:B", "", raw)            block B ends (raw "" when implied by a new header / the end of the turn)
    ("stop", "", raw)               end of the turn (end-sampling / another role)."""

    def __init__(self):
        self.block: str | None = None  # the open block
        self.header: str | None = None  # the author-name text of a header in progress (None: no header open)
        self.header_raw = ""

    def feed(self, text: str) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for piece in _SPLIT.split(text):
            if not piece:
                continue
            kind = _KIND.get(piece)
            if kind is None:
                if self.header is not None:
                    self.header += piece
                    self.header_raw += piece
                elif self.block is None:
                    out.append(("loose", piece, piece))
                else:
                    out.append(("data", piece, piece))
            elif kind == "model":
                self._close(out, "")
                if self.header is None:
                    self.header, self.header_raw = "", ""
                self.header_raw += piece
            elif kind in ("thinking", "text", "tool_json"):
                if self.header is None and self.block == kind == "thinking":
                    out.append(("data", "", piece))  # a repeated thinking marker inside a thinking block
                    continue
                self._close(out, "")
                name, raw = self.header or "", self.header_raw + piece
                self.header, self.header_raw = None, ""
                self.block = kind
                out.append(("open:" + kind, name, raw))
            elif kind == "end":
                if self.block is not None:
                    self._close(out, piece)
                elif self.header is not None:  # a header without a content marker: its text is visible
                    out.append(("loose", self.header, self.header_raw + piece))
                    self.header, self.header_raw = None, ""
                else:
                    out.append(("loose", "", piece))
            else:  # stop
                self._close(out, "")
                if self.header is not None:
                    out.append(("loose", self.header, self.header_raw))
                    self.header, self.header_raw = None, ""
                out.append(("stop", "", piece))
        return out

    def _close(self, out, raw: str) -> None:
        if self.block is not None:
            out.append(("close:" + self.block, "", raw))
            self.block = None


def _tool_call(name: str, payload: str) -> tuple[str, str] | None:
    """(name, arguments JSON) of an invoke_tool_json payload {"name": ..., "args": {...}}; None if it is not one."""
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") if isinstance(obj.get("name"), str) and obj.get("name") else name.strip()
    if not name:
        return None
    args = next((obj[k] for k in ("args", "arguments", "parameters") if k in obj), {})
    return name, args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)


def _tools_active(request) -> bool:
    """The request's output goes through the tool parser (vLLM's auto tool-choice condition)."""
    return bool(getattr(request, "tools", None)) and getattr(request, "tool_choice", None) in ("auto", None)


class _Replay:
    """Per-stream scanner state that survives being driven with (previous_text, delta_text): a call whose
    previous_text is not what this instance has seen (another choice of an n>1 request) replays it first."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.scanner = BlockScanner()
        self.seen = ""

    def feed(self, previous_text: str, delta_text: str, on_event) -> None:
        if previous_text != self.seen:
            self.reset()
            for ev in self.scanner.feed(previous_text):
                on_event(ev, replay=True)
            self.seen = previous_text
        for ev in self.scanner.feed(delta_text):
            on_event(ev, replay=False)
        self.seen += delta_text


@ReasoningParserManager.register_module("inkling2")
class Inkling2ReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        vocab = self.vocab
        self._ended_ids = {vocab[t] for t in _ENDED}
        self._turn_ids = {vocab[t] for t in ("<|message_user|>", "<|message_system|>", "<|message_tool|>")}
        self._end_message_id = vocab[END_MESSAGE]
        self._model_id = vocab[MESSAGE_MODEL]
        self._tools = False
        self._stream = _Replay()
        self._handed_off = False

    def adjust_request(self, request):
        request.skip_special_tokens = False
        self._tools = _tools_active(request)
        return request

    # ---------------------------------------------------------------- token-id rules
    def _turn_start(self, input_ids: Sequence[int]) -> int:
        """Index after the prompt's last non-model message (0 for output-only ids)."""
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] in self._turn_ids:
                for j in range(i + 1, len(input_ids)):
                    if input_ids[j] == self._end_message_id:
                        return j + 1
                return len(input_ids)
        return 0

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """A visible block (text / tool call / tool error) or the end of the turn has started after the last
        non-model message.  A prompt (ends with a user / system / tool message) is never past its reasoning."""
        ended = self._ended_ids
        return any(t in ended for t in input_ids[self._turn_start(input_ids):])

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        start = self._turn_start(input_ids)
        for i in range(start, len(input_ids)):
            if input_ids[i] in self._ended_ids:
                j = i
                while j > start and input_ids[j - 1] != self._end_message_id and input_ids[j - 1] != self._model_id:
                    j -= 1  # the header's author-name tokens
                if j > start and input_ids[j - 1] == self._model_id:
                    j -= 1
                return list(input_ids[j:])
        return []

    # ---------------------------------------------------------------- text
    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        """(reasoning, content).  With tools active and at least one well-formed tool call the content is the raw
        non-thinking blocks (markers kept) for the tool parser; otherwise (vLLM then uses the content as is) it is the
        visible text, an unparsable tool block's payload included, as the streamed path delivers it."""
        reasoning, visible, raw_rest = [], [], []
        in_thinking, in_tool, tool_name, tool_buf, calls = False, False, "", [], 0
        for kind, payload, raw in BlockScanner().feed(model_output):
            if kind == "open:thinking":
                in_thinking = True
            elif kind == "close:thinking":
                in_thinking = False
            elif kind == "data" and in_thinking:
                reasoning.append(payload)
            else:
                raw_rest.append(raw)
                if kind == "open:tool_json":
                    in_tool, tool_name, tool_buf = True, payload, []
                elif kind == "close:tool_json":
                    calls += _tool_call(tool_name, "".join(tool_buf)) is not None
                    in_tool = False
                elif kind == "data" and in_tool:
                    tool_buf.append(payload)
                if kind in ("data", "loose"):
                    visible.append(payload)
        content = raw_rest if _tools_active(request) and calls else visible
        return "".join(reasoning) or None, "".join(content) or None

    def extract_reasoning_streaming(self, previous_text, current_text, delta_text, previous_token_ids,
                                    current_token_ids, delta_token_ids) -> DeltaMessage | None:
        reasoning, content = [], []
        state = self.__dict__.setdefault("_sstate", {"thinking": False})

        def on_event(ev, replay):
            kind, payload, raw = ev
            if self._tools and self._handed_off:
                if not replay:
                    content.append(raw)
                return
            if kind == "open:thinking":
                state["thinking"] = True
            elif kind == "close:thinking":
                state["thinking"] = False
            elif kind == "data" and state["thinking"]:
                if not replay:
                    reasoning.append(payload)
            elif kind.startswith("open:") or kind == "stop":
                if self._tools:
                    self._handed_off = True
                    if not replay:
                        content.append(raw)
            elif kind in ("data", "loose") and not replay:
                content.append(payload)

        if previous_text != self._stream.seen:
            state["thinking"] = False
            self._handed_off = False
        self._stream.feed(previous_text, delta_text, on_event)
        if not reasoning and not content:
            return None
        return DeltaMessage(reasoning="".join(reasoning) or None, content="".join(content) or None)


@ToolParserManager.register_module("inkling2")
class Inkling2ToolParser(ToolParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._stream = _Replay()
        self._block: str | None = None
        self._name = ""
        self._buf: list[str] = []

    def adjust_request(self, request):
        request = super().adjust_request(request)
        request.skip_special_tokens = False
        return request

    def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
        content, calls = [], []
        block, name, buf = None, "", []
        for kind, payload, raw in BlockScanner().feed(model_output):
            if kind.startswith("open:"):
                block, name, buf = kind[5:], payload, []
            elif kind.startswith("close:") or kind == "stop":
                if block == "tool_json":
                    call = _tool_call(name, "".join(buf))
                    if call is None:
                        content.append("".join(buf))
                    else:
                        calls.append(ToolCall(type="function", function=FunctionCall(name=call[0], arguments=call[1])))
                block = None
            elif kind == "data":
                if block == "tool_json":
                    buf.append(payload)
                elif block == "text":
                    content.append(payload)
            elif kind == "loose":
                content.append(payload)
        if block == "tool_json":  # unterminated tool block (length limit): keep its text
            content.append("".join(buf))
        text = "".join(content)
        return ExtractedToolCallInformation(tools_called=bool(calls), tool_calls=calls,
                                            content=text if text else None)

    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, previous_token_ids,
                                     current_token_ids, delta_token_ids, request) -> DeltaMessage | None:
        content, reasoning, entries = [], [], []

        def on_event(ev, replay):
            kind, payload, raw = ev
            if kind.startswith("open:"):
                self._block, self._name, self._buf = kind[5:], payload, []
            elif kind.startswith("close:") or kind == "stop":
                if self._block == "tool_json":
                    call = _tool_call(self._name, "".join(self._buf))
                    if call is None:
                        if not replay:
                            content.append("".join(self._buf))
                    else:
                        index = len(self.prev_tool_call_arr)
                        self.prev_tool_call_arr.append({"name": call[0], "arguments": call[1]})
                        self.streamed_args_for_tool.append(call[1])
                        self.current_tool_id = index
                        if not replay:
                            entries.append(DeltaToolCall(index=index, id=make_tool_call_id(), type="function",
                                                         function=DeltaFunctionCall(name=call[0],
                                                                                    arguments=call[1])))
                self._block = None
            elif kind == "data":
                if self._block == "tool_json":
                    self._buf.append(payload)
                elif not replay:
                    (reasoning if self._block == "thinking" else content).append(payload)
            elif kind == "loose" and not replay:
                content.append(payload)

        if previous_text != self._stream.seen:
            self._block, self._name, self._buf = None, "", []
            self.prev_tool_call_arr, self.streamed_args_for_tool, self.current_tool_id = [], [], -1
        self._stream.feed(previous_text, delta_text, on_event)
        if not (content or reasoning or entries):
            return None
        return DeltaMessage(content="".join(content) or None, reasoning="".join(reasoning) or None,
                            tool_calls=entries)
