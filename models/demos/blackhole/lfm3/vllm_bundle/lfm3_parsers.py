# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""vLLM reasoning + tool-call parsers for LFM3 (load with --reasoning-parser-plugin / --tool-parser-plugin this file;
select with --reasoning-parser lfm3 --tool-call-parser lfm3).

Reasoning: "<think> ... </think>" (single tokens 124901 / 124902).  The chat template's generation prompt ends in
"<think>" (thinking on, the default) or "<think></think>" (chat_template_kwargs {"enable_thinking": false}), so the
completion starts inside the thinking block, or after it.  That is Qwen3.5's convention: vLLM's Qwen3 parser
(thinking state from enable_thinking; a prompt that already closed the block is detected by the serving layer through
is_reasoning_end on the prompt ids, and the whole output is content; a missing "</think>" with thinking on means
truncated reasoning).  With thinking off, a stray "</think>" the model emits stays in the content (both paths).

Tool calls: Qwen3-Coder XML, "<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n</function>\\n</tool_call>"
(the six tags are single tokens).  Non-streamed: vLLM's qwen3_coder parser unchanged (parameter values converted with
the request's tool schema: strings verbatim, multi-line included; numbers / booleans / objects as JSON).  Streamed:
Lfm3ToolParser.extract_tool_calls_streaming (below) replaces vLLM's incremental state machine, which loses arguments
when a delta carries many tokens (prompt-lookup speculation: a whole call in 1-3 deltas -> arguments "{}").
"""
from __future__ import annotations

from vllm.entrypoints.openai.engine.protocol import DeltaFunctionCall, DeltaMessage, DeltaToolCall
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


@ReasoningParserManager.register_module("lfm3")
class Lfm3ReasoningParser(Qwen3ReasoningParser):
    """<think> ... </think> with the opening tag (or the whole closed block) supplied by the prompt.

    Thinking off (enable_thinking false: the prompt already closed the block): the whole output is content, also when
    the model emits a stray "</think>".  vLLM's streaming path does this from the prompt ids (is_reasoning_end on the
    prompt) without calling the parser, but its non-streaming path always calls extract_reasoning, and Qwen3's would
    split at that stray tag and return reasoning; this keeps both paths identical."""

    def extract_reasoning(self, model_output, request):
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def is_reasoning_end(self, input_ids):
        """Qwen3's rule (the last of <think> / </think> / <tool_call> decides; a <tool_call> is an implicit end, as
        in its non-streaming extract_reasoning), without its exception for a <tool_call> followed by </tool_call>
        (meant for template examples in a prompt): the serving layer asks this of each streamed delta, and a delta
        that holds a whole call (speculation) would otherwise never end the reasoning and stream the call as
        content.  LFM3 prompts end in "<think>" or "<think></think>", so a prompt is decided by its last tokens."""
        for t in reversed(input_ids):
            if t == self.start_token_id:
                return False
            if t == self.end_token_id or t == self._tool_call_token_id:
                return True
        return False


@ToolParserManager.register_module("lfm3")
class Lfm3ToolParser(Qwen3CoderToolParser):
    """LFM3's Qwen3-Coder-style XML tool calls.

    Streaming re-derives, at every delta, what vLLM's non-streaming extract_tool_calls makes of the text so far and
    sends only what can no longer change, so any split of the output into deltas (one token each, a whole call in
    one delta, several calls in one delta, a call split anywhere) streams the same calls and content:
    - content: the text before the first "<tool_call>" (before a bare "<function=" it is held back: the non-streaming
      parser treats such text as a call only when no "<tool_call>" follows);
    - a call: its header (id, name, arguments "") as soon as "<function=NAME>" is complete, its whole JSON arguments
      (the non-streaming parse of that call) once the call is final, i.e. closed by "</function>" or by its
      "</tool_call>";
    - while a call is open every delta carries an empty-arguments entry for it, and prev_tool_call_arr holds the
      non-streaming parse of the partial call: if the request then ends without closing it (max_tokens), vLLM's
      serving layer sends that remainder with the finish chunk (its "unstreamed tool arguments" check), as the
      non-streamed response would return it;
    - at the end-of-sequence token the non-streaming result of the whole text is emitted for whatever is left (text
      after a "<tool_call>" that held no call becomes content, bare-"<function=" calls are sent).
    A delta with content or several tool-call entries starts with an entry whose arguments are None, so the serving
    layer's finish check (which rebuilds the last delta around its first entry) leaves such a delta intact."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        stop = (getattr(self.model_tokenizer, "eos_token_id", None), self.vocab.get("<|im_end|>"))
        self._stop_ids = frozenset(i for i in stop if i is not None)
        self._reset_stream(None)

    def _reset_stream(self, request):
        self._request = request
        self._content_sent = 0  # chars of the text already sent as content
        self._final_calls: list[bool] = []  # per announced call: its arguments were sent
        self._broken = False  # a final call did not parse: no more calls until the end (non-streaming fallback)
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []

    def _functions(self, text):
        """The function-call strings _get_function_calls(text) returns when text holds a "<tool_call>", each with
        whether it is final (later text cannot change it): closed by "</function>", or inside a closed block."""
        out = []
        for block in self.tool_call_regex.finditer(text):
            closed_block = block.group(1) is not None
            body = block.group(1) if closed_block else block.group(2)
            for f in self.tool_call_function_regex.finditer(body):
                out.append((f.group(1), True) if f.group(1) is not None else (f.group(2), closed_block))
        return out

    def _non_streamed(self, text, tools):
        """(content, [(name, arguments)]) as extract_tool_calls(text) returns them (content "" for None)."""
        if self.tool_call_prefix not in text:
            return text, []
        try:
            calls = [self._parse_xml_function_call(f, tools) for f in self._get_function_calls(text)]
        except Exception:
            return text, []
        if not calls:
            return text, []
        i = text.find(self.tool_call_start_token)
        return text[:i if i >= 0 else text.find(self.tool_call_prefix)], [
            (c.function.name, c.function.arguments) for c in calls]

    def _announce(self, i, name, entries):
        self._final_calls.append(False)
        self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})
        self.streamed_args_for_tool.append("")
        entries.append(DeltaToolCall(index=i, id=self._generate_tool_call_id(), type="function",
                                     function=DeltaFunctionCall(name=name, arguments="")))

    def _finish_call(self, i, arguments, entries):
        self._final_calls[i] = True
        self.prev_tool_call_arr[i]["arguments"] = arguments
        self.streamed_args_for_tool[i] = arguments
        last = entries[-1] if entries else None
        if last is not None and last.index == i and last.function.arguments == "":
            last.function.arguments = arguments  # announced in this same delta
        else:
            entries.append(DeltaToolCall(index=i, function=DeltaFunctionCall(arguments=arguments)))

    def _message(self, content, entries):
        open_call = next((i for i, done in enumerate(self._final_calls) if not done), None)
        if not content and not entries:
            if open_call is None:
                return None
            entries = [DeltaToolCall(index=open_call, function=DeltaFunctionCall(arguments=""))]
        if entries and (content or len(entries) > 1) and entries[0].function.arguments is not None:
            first = entries[0]
            if first.id is not None:  # a new call: its header alone first, then its arguments
                arguments = first.function.arguments
                first.function = DeltaFunctionCall(name=first.function.name)
                if arguments:
                    entries.insert(1, DeltaToolCall(index=first.index, function=DeltaFunctionCall(arguments=arguments)))
            else:  # the rest of an announced call: lead with a no-op entry for that same call
                entries.insert(0, DeltaToolCall(index=first.index))
        return DeltaMessage(**({"content": content} if content else {}), tool_calls=entries)

    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, previous_token_ids,
                                     current_token_ids, delta_token_ids, request):
        if not previous_text:
            self._reset_stream(request)
        final = any(t in self._stop_ids for t in delta_token_ids)
        if not delta_text and not final:
            return self._message("", [])
        text = current_text
        tools = request.tools if request is not None else None
        entries: list[DeltaToolCall] = []
        if final:
            content, calls = self._non_streamed(text, tools)
            if not self._broken:
                for i, (name, arguments) in enumerate(calls):
                    if i < len(self._final_calls) and self._final_calls[i]:
                        continue
                    if i == len(self._final_calls):
                        self._announce(i, name, entries)
                    self._finish_call(i, arguments, entries)
        else:
            start = text.find(self.tool_call_start_token)
            if start >= 0:
                content = text[:start]
                funcs = self._functions(text)
            else:
                bare = text.find(self.tool_call_prefix)
                content = text if bare < 0 else text[:bare]
                funcs = []
            for i, (fstr, done) in enumerate(funcs):
                if self._broken:
                    break
                if i < len(self._final_calls) and self._final_calls[i]:
                    continue
                if i == len(self._final_calls):
                    name_end = fstr.find(">")
                    if name_end < 0:
                        self._broken = done  # a final call without a name fails the non-streaming parse
                        break
                    self._announce(i, fstr[:name_end], entries)
                try:
                    arguments = self._parse_xml_function_call(fstr, tools).function.arguments
                except Exception:
                    self._broken = done
                    break
                if done:
                    self._finish_call(i, arguments, entries)
                else:
                    self.prev_tool_call_arr[i]["arguments"] = arguments  # sent by the serving layer if truncated
        new_content = ""
        if len(content) > self._content_sent:
            new_content = content[self._content_sent:]
            self._content_sent = len(content)
        return self._message(new_content, entries)
