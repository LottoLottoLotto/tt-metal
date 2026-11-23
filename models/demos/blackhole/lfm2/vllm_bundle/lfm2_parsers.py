# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""vLLM reasoning + tool-call parsers for LFM2.5 (load with --reasoning-parser-plugin / --tool-parser-plugin this file;
select with --reasoning-parser lfm2 --tool-call-parser lfm2).

Reasoning: the chat template's generation prompt ends in "<think>" (LFM2.5 always thinks), so the completion starts
inside the thinking block and closes it with "</think>" (both non-special single tokens 124901 / 124902).  That is
DeepSeek-R1's convention (reasoning first, start token usually absent from the output).

Tool calls: "<|tool_call_start|>[f(a=1), g(b='x')]<|tool_call_end|>" (non-special tokens 124905 / 124906): a pythonic
call list between markers, possibly after some text.  Non-streaming: every marker block is parsed with vLLM's
pythonic parser and the text outside the blocks is the content.  Streaming: text streams as content; a block is held
back until its end marker and then emitted as complete tool-call deltas.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
)
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
from vllm.tool_parsers.pythonic_tool_parser import PythonicToolParser

TOOL_START = "<|tool_call_start|>"
TOOL_END = "<|tool_call_end|>"


@ReasoningParserManager.register_module("lfm2")
class Lfm2ReasoningParser(DeepSeekR1ReasoningParser):
    """<think> ... </think>, with the opening tag supplied by the prompt."""


@ToolParserManager.register_module("lfm2")
class Lfm2ToolParser(ToolParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._pythonic = PythonicToolParser(tokenizer)
        self._emitted_blocks = 0   # streaming: complete marker blocks already sent
        self._content_sent = 0     # streaming: characters of current_text sent as content

    def _calls(self, block: str, request):
        info = self._pythonic.extract_tool_calls(block.strip(), request)
        return info.tool_calls if info.tools_called else None

    def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
        if TOOL_START not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        calls, text, rest = [], [], model_output
        while TOOL_START in rest:
            before, _, after = rest.partition(TOOL_START)
            text.append(before)
            block, found, rest = after.partition(TOOL_END)
            parsed = self._calls(block, request) if found else None
            if parsed is None:  # unterminated or unparsable block: keep it as text
                text.append(TOOL_START + block + (TOOL_END if found else ""))
                continue
            calls.extend(parsed)
        text.append(rest)
        content = "".join(text).strip() or None
        if not calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        return ExtractedToolCallInformation(tools_called=True, tool_calls=calls, content=content)

    def extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str,
                                     previous_token_ids: Sequence[int], current_token_ids: Sequence[int],
                                     delta_token_ids: Sequence[int], request) -> DeltaMessage | None:
        # Content = the text outside marker blocks; an open (unterminated) block is held back until its end marker.
        # The markers are single tokens, so a marker never arrives split across deltas.
        blocks = current_text.split(TOOL_START)
        visible = [blocks[0]]
        complete = []
        for part in blocks[1:]:
            body, found, tail = part.partition(TOOL_END)
            if found:
                complete.append(body)
                visible.append(tail)
        text = "".join(visible)
        content = text[self._content_sent:]
        self._content_sent = max(self._content_sent, len(text))
        tool_deltas = []
        for body in complete[self._emitted_blocks:]:
            for call in self._calls(body, request) or []:
                self.current_tool_id += 1
                args = call.function.arguments
                self.prev_tool_call_arr.append({"name": call.function.name, "arguments": json.loads(args or "{}")})
                self.streamed_args_for_tool.append(args)
                tool_deltas.append(DeltaToolCall(
                    index=self.current_tool_id, id=make_tool_call_id(), type="function",
                    function=DeltaFunctionCall(name=call.function.name, arguments=args).model_dump(exclude_none=True)))
        self._emitted_blocks = len(complete)
        if tool_deltas:
            return DeltaMessage(content=content or None, tool_calls=tool_deltas)
        if content:
            return DeltaMessage(content=content)
        return None
