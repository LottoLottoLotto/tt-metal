# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for LFM2.5-24B-A2B on one P150 (vllm-tt-plugin; registered through EXTRA_MODELS_DIR, see
vllm_bundle/models/lfm2/vllm_metadata.json).

- Prefill is model-owned and eager: every request is prefilled (2048-token chunks, bucketed lengths) into its batch
  slot (conv-state row) and its KV pages; the slot comes from the runner's `empty_slots`.
- Decode runs through tt_transformers' Generator: `Lfm2Model.ttnn_decode_forward` is captured as a trace (one for
  logits, one with the greedy argmax on device) and replayed with refreshed host inputs every step.  With
  max_num_seqs > 1 the step runs at the smallest decode width (1, 2, 4, ..., max_num_seqs) holding the active rows
  (row b = conv-state slot b after the runner's slot remap): one trace set per width, all captured at warmup.
- Sampling: all-greedy unpenalized batches take the device argmax (`supports_greedy_only_sample_on_device`, needs
  `sample_on_device_mode: decode_only`); every other batch whose requests are all device-eligible takes the exact
  vLLM-equivalent device sampler (tt/sampling.py: any temperature / top_k incl. disabled / top_p / min_p / penalties,
  counter-based RNG; vllm_tt_plugin asks `exact_sampling_eligible` / `exact_sampling_spec`) bound to the logits
  trace, with single-token host redraws it flags; anything else returns logits to vLLM's host sampler.
- KV: one FullAttentionSpec (block 64, 8 KV heads, head 64) for the 10 attention layers; the short-conv state is
  model-owned (like Qwen36's GDN state).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.generator import Generator

from .config import Lfm2Config
from .model import BLOCK, PREFILL_BUCKETS, PREFILL_CHUNK, Lfm2Model
from .source import HFSource, QuantSource

# Fast decode parts (tt/fast/step.py), pinned (an upgrade re-runs the gates): MoEDecodeV2 (router fused, 1 row) and
# MoEDecodeRows (B 2..16, union of the rows' experts), the bitwise norm+matmul, fused conv-op and attention-front
# programs, and the custom paged SDPA decode (asdpa: not bitwise vs stock, closer to FP64; every decode width, work
# spread by the live rows' positions, batch-invariant; 5.3 ms/step vs 17.5 stock at pos 2000, 6.1 vs 19.7 at 32768).
FAST_PARTS = "moe2,moer,norm,conv,attn,mm,nmm,convf,afront,asdpa"


def open_weights(path: str):
    """A quant checkpoint directory (lfmq manifest) or an HF BF16 snapshot."""
    p = Path(path)
    manifest = p / "manifest.json"
    if manifest.is_file() and "lfmq" in manifest.read_text()[:400]:
        return QuantSource(p)
    return HFSource(p)


def _is_greedy(sampling_params) -> bool:
    temps = getattr(sampling_params, "temperature", None)
    if temps is None:
        return True
    temps = temps.tolist() if isinstance(temps, torch.Tensor) else (temps if isinstance(temps, (list, tuple)) else [temps])
    return all(float(t) == 0.0 for t in temps)


class Lfm2ForCausalLM(Generator):
    """vLLM-compatible wrapper for LFM2.5-24B-A2B (Lfm2MoeForCausalLM) on one Blackhole P150."""

    model_capabilities = {
        # Exact prefix resume (vllm_tt_plugin/prefix_resume.py): conv-state snapshots at 2048-token boundaries.
        "supports_prefix_caching": True,
        # The conv state is a running state: async decode's device-side token/position continuity is not supported.
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        "supports_greedy_only_sample_on_device": True,
        "supports_exact_device_sampling": True,
    }

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Paged attention for the 10 attention layers; the conv state is owned by the TT model."""
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        cache_config.block_size = BLOCK
        dtype = (model_config.dtype if cache_config.cache_dtype == "auto"
                 else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype])
        return {"attn": FullAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=model_config.get_num_kv_heads(vllm_config.parallel_config),
            head_size=model_config.get_head_size(),
            dtype=dtype,
            sliding_window=None,
        )}

    @classmethod
    def get_max_tokens_all_users(cls, model_name: str = "", num_devices: int = 1, tt_data_parallel: int = 1,
                                 max_model_len: int | None = None, max_num_seqs: int | None = None, **kwargs) -> int:
        """KV token pool shared by all users: LFM2_MAX_TOKENS_ALL_USERS, else max_model_len x max_num_seqs."""
        override = os.environ.get("LFM2_MAX_TOKENS_ALL_USERS")
        if override:
            return int(override)
        return int(max_model_len or 131072) * int(max_num_seqs or 1)

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1,
                              optimizations=None, **kwargs):
        if tt_data_parallel != 1 or mesh_device.get_num_devices() != 1:
            raise ValueError("LFM2.5 serving runs on one P150 (no data / tensor parallelism)")
        if not 1 <= max_batch_size <= 16:
            raise ValueError("max_num_seqs must be in 1..16 (device sampler and conv-state slots)")
        path = os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL") or hf_config._name_or_path
        src = open_weights(path)
        cache_root = os.environ.get("TT_CACHE_PATH")
        logger.info(f"LFM2.5: weights {path} ({src.identity}), tensor cache {cache_root}, "
                    f"max_batch_size {max_batch_size}, max_seq_len {max_seq_len}")
        cls._log_default_sampling()
        model = Lfm2Model(mesh_device, src, cfg=Lfm2Config.from_hf(path), max_seq_len=max_seq_len, batch=max_batch_size,
                          logits_dtype=ttnn.bfloat16, own_kv=False,
                          cache_root=(Path(cache_root) / "tensors") if cache_root else None,
                          log=lambda m: None)
        # Every decode step re-stages tokens/positions from the host (no device token feedback into the trace inputs).
        model._tt_vllm_always_refresh_decode_trace_inputs = True
        model.sampling = None
        args = SimpleNamespace(max_batch_size=max_batch_size, max_seq_len=max_seq_len, vocab_size=model.cfg.vocab,
                               mesh_device=mesh_device)
        return cls([model], [args], mesh_device)

    @staticmethod
    def _log_default_sampling():
        """State the server's default sampling (vLLM fills a request's unset fields from it) and its sampler path:
        tools/serve_lfm2.sh passes the resolved defaults as LFM2_DEFAULT_SAMPLING_JSON (the engine's vLLM config is not
        current in the worker when the model is built); else the current vLLM config, if any."""
        from vllm.config import get_current_vllm_config

        from .sampling import MAX_TOP_K

        if os.environ.get("LFM2_DEFAULT_SAMPLING_JSON"):
            d = json.loads(os.environ["LFM2_DEFAULT_SAMPLING_JSON"])
        else:
            try:
                model_config = get_current_vllm_config().model_config
            except AssertionError:  # no vLLM config set (standalone tests)
                return
            if model_config is None:
                return
            d = model_config.get_diff_sampling_param()
        t = float(d.get("temperature", 1.0))
        top_k = int(d.get("top_k", 0))
        path = ("device argmax" if t < 1e-5 and float(d.get("repetition_penalty", 1.0)) == 1.0 else
                "exact device sampler" + (", full vocabulary" if t >= 1e-5 and (top_k <= 0 or top_k > MAX_TOP_K)
                                          else ""))
        label = os.environ.get("LFM2_DEFAULT_SAMPLING_LABEL", "vLLM model config")
        logger.info(f"LFM2.5: default sampling for requests that set no sampling fields ({label}): {d or '{}'} "
                    f"(unset fields vLLM-neutral; decode path: {path})")

    # ------------------------------------------------------------------ KV / state
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        model = self.model[0]
        caches = model.allocate_kv_caches(kv_cache_shape)
        # LfmKern's fused decode step (tt/fast/step.py) unless LFM_FAST_DECODE=0; parts: LFM_FAST_PARTS (comma list).
        if os.environ.get("LFM_FAST_DECODE", "1") != "0":
            parts = [p for p in os.environ.get("LFM_FAST_PARTS", FAST_PARTS).split(",") if p]
            model.enable_fast_decode(parts)
            logger.info(f"LFM2: fast decode step, parts {parts}")
        if self.exact_sampling_available():
            model.init_exact_sampler(model.batch)
        self.speculation = None
        if os.environ.get("LFM2_SPECULATE", "0") == "1":
            if model.batch != 1 or model.fast is None:
                logger.warning("LFM2 speculation needs max_num_seqs 1 and the fast decode step: disabled")
            else:
                from .speculate import Speculation

                self.speculation = Speculation(model, -(-model.max_seq_len // BLOCK),
                                               sampler=getattr(model, "exact_sampler", None), log=logger.info)
                logger.info(f"LFM2: prompt-lookup speculation, verify widths {self.speculation.widths}")
        return caches

    # ------------------------------------------------------------------ prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, start_pos=None, empty_slots=None,
                        sampling_params=None, enable_trace=False, **kwargs):
        """Model-owned prefill of every row into its slot.  Returns last-token logits [B, 1, vocab] (float32), or the
        greedy token ids [B] when vLLM asked for device sampling (sample_on_device_mode 'all')."""
        self.mode = Mode.PREFILL
        model = self.model[0]
        B = int(tokens.shape[0])
        slots = list(range(B)) if empty_slots is None else [int(s) for s in empty_slots]
        self._log_decode_path = True  # the next decode step states its path (a line per admission, not per step)
        out = []
        for i in range(B):
            L = int(prompt_lens[i]) if prompt_lens is not None else int(tokens.shape[1])
            start = int(start_pos[i]) if start_pos is not None else 0
            if start % PREFILL_CHUNK:
                raise ValueError(f"LFM2 prefill resumes only at {PREFILL_CHUNK}-token boundaries (got {start})")
            resume, snapshot = self._prefix_resume_args(tokens[i], page_table[i], start)
            if resume is not None:
                model.restore_slot_state(slots[i], resume)
            logits = model.prefill(tokens[i, start:L], start=start, slot=slots[i], page_table_row=page_table[i],
                                   snapshot=snapshot)
            out.append(logits)
        logits = torch.stack(out).unsqueeze(1)  # [B, 1, V]
        if sampling_params is not None:
            if not _is_greedy(sampling_params):
                raise ValueError("LFM2 prefill device sampling supports greedy only")
            return logits[:, -1].argmax(-1).to(torch.int32)
        return logits

    @staticmethod
    def _prefix_resume_args(tokens, page_table_row, start):
        """(conv-state payload to restore, snapshot callback) for one request's prefill.  A nonzero start is the
        boundary the scheduler resumed this request from (its KV blocks below it are the snapshot's, written by the
        same 2048-chunk schedule); the callback publishes this prefill's last full-chunk boundary."""
        from vllm_tt_plugin.prefix_resume import REGISTRY

        resume = REGISTRY.take(tokens, start) if start else None
        snapshot = None
        if REGISTRY.active:
            row = page_table_row

            def snapshot(length, payload):
                REGISTRY.put(tokens, length, row, payload, sum(int(t.numel()) * 2 for t in payload))
        return resume, snapshot

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device=False, greedy_only=False, **kwargs):
        """Compile every prefill program the server can hit: each bucket at chunk start 0 and at a later chunk start
        (all later chunks share one chunked-SDPA program).  Prefill is never traced."""
        if self.already_warmed_up_prefill:
            return
        model = self.model[0]
        blocks = int(kv_cache[0][0].shape[0]) - 1 if kv_cache else model.trash_block
        need = (PREFILL_CHUNK + PREFILL_BUCKETS[-1]) // BLOCK
        if blocks < need:
            raise ValueError(f"KV pool of {blocks} blocks cannot hold the prefill warmup ({need} blocks)")
        row = torch.arange(need, dtype=torch.int32)
        lengths = list(PREFILL_BUCKETS) + [PREFILL_CHUNK + b for b in PREFILL_BUCKETS]
        if model.max_seq_len < lengths[-1]:
            lengths = [n for n in lengths if n <= model.max_seq_len]
        for n in lengths:
            logger.info(f"LFM2 prefill warmup: {n} tokens")
            model.prefill(torch.full((n,), 1000, dtype=torch.int64), slot=0, page_table_row=row)
        model.reset()
        from vllm_tt_plugin.prefix_resume import REGISTRY

        # Prefix resume: snapshots at 2048-token boundaries (active only when vLLM prefix caching is on).
        REGISTRY.configure(PREFILL_CHUNK, BLOCK, int(os.environ.get("LFM2_PREFIX_SNAPSHOTS", "16")))
        self.already_warmed_up_prefill = True

    # ------------------------------------------------------------------ decode
    def warmup_model_decode(self, *args, **kwargs):
        """Compile (enable_trace False) or capture (True) every decode width; the exact sampler per width too."""
        if kwargs.get("can_sample_on_device", False):
            kwargs["greedy_only"] = True  # the single-device argmax path; random sampling uses the exact sampler
        kwargs.pop("non_greedy_decoding_on_device", None)
        model = self.model[0]
        capture = kwargs.get("enable_trace", False)
        result = None
        for width in model.decode_widths:
            logger.info(f"LFM2 decode warmup: width {width} (capture={capture})")
            result = super().warmup_model_decode(*args, **dict(kwargs, max_batch_size=width))
            if self.exact_sampling_available():
                self._warmup_exact_sampler(width, capture)
        if capture and getattr(self, "speculation", None) is not None:
            self.speculation.capture()
        model.reset()  # warmup decoded garbage into every conv-state slot
        return result

    # ------------------------------------------------------------------ prompt-lookup speculation (tt/speculate.py)
    # vllm_tt_plugin (src/docker/plugin-speculate.patch) calls these for speculative_config method "ngram" with one
    # running request: lookup_drafts after every step, lookup_step for a step the scheduler gave drafts.
    def sampled_speculation_available(self) -> bool:
        """Sampled requests (the exact device sampler) may speculate too (asked before allocate_kv_cache)."""
        return os.environ.get("LFM2_SPECULATE", "0") == "1" and self.exact_sampling_available()

    def lookup_drafts(self, owner, prompt_ids, output_ids, max_drafts: int, n_min: int, n_max: int) -> list[int]:
        spec = getattr(self, "speculation", None)
        if spec is None or max_drafts <= 0:
            return []
        return spec.drafts(owner, prompt_ids, output_ids, max_drafts, n_min, n_max)

    def lookup_step(self, token, drafts, position, block_ids, request=None):
        """One verify step -> (emitted tokens, None).  request: the exact sampler's SamplingRequest (None: greedy)."""
        self.mode = Mode.DECODE
        # the sampler path ("argmax" / "exact"), shared with the plain steps that alternate with verify steps
        path = "argmax" if request is None else "exact"
        if self.__dict__.pop("_log_decode_path", False) or path != getattr(self, "_decode_path", None):
            logger.info(f"LFM2 decode path: {path} with prompt-lookup speculation (verify steps when drafts exist)")
            self._decode_path = path
        emitted = self.speculation.step(token, drafts, position, block_ids, request)
        return emitted, None

    def _use_width(self, width: int):
        """Point the Generator's decode trace stores at this width's (trace ids, inputs, outputs)."""
        stores = self.__dict__.setdefault("_width_traces", {})
        if width not in stores:
            stores[width] = (defaultdict(lambda: None), defaultdict(lambda: None), defaultdict(lambda: None))
        self.trace_ids_decode, self.trace_inputs_decode, self.trace_output_decode = stores[width]

    def decode_forward(self, tokens, start_pos, page_table=None, kv_cache=None, enable_trace=True,
                       read_from_device=True, sampling_params=None, reset_batch=False, prompt_tokens=None,
                       output_tokens=None, slot_remap=None, exact_sampling=None, **kwargs):
        """Returns host tensors: greedy token ids [B] (device argmax), sampled ids [rows] (exact sampler), or logits
        [B, 1, vocab]."""
        model = self.model[0]
        if getattr(self, "speculation", None) is not None:
            self.speculation.note_plain_step()
        if slot_remap is not None and model.batch > 1:
            model.remap_slots(slot_remap)
        B = int(tokens.shape[0])
        if B > model.batch:
            raise ValueError(f"{B} decode rows exceed the {model.batch} conv-state slots")
        tokens = torch.as_tensor(tokens).reshape(B, -1)
        start_pos = torch.as_tensor(start_pos).reshape(-1)
        live = (start_pos >= 0).nonzero()
        # rows [0, n) hold every active row; the step runs at the smallest decode width >= n, the extra rows inactive
        # (position -1: no KV write, no attention, no experts; their outputs are ignored)
        n = int(live.max()) + 1 if live.numel() else 1
        width = model.decode_width(n)
        if B > width:
            tokens, start_pos = tokens[:width], start_pos[:width]
            page_table = page_table[:width] if page_table is not None else None
        elif B < width:
            pad = width - B
            tokens = torch.cat([tokens, torch.zeros(pad, tokens.shape[1], dtype=tokens.dtype)])
            start_pos = torch.cat([start_pos, torch.full((pad,), -1, dtype=start_pos.dtype)])
            if page_table is not None:
                page_table = torch.cat([page_table, torch.zeros(pad, page_table.shape[1], dtype=page_table.dtype)])
        self._use_width(width)
        common = dict(page_table=page_table, kv_cache=kv_cache, enable_trace=enable_trace, read_from_device=False,
                      reset_batch=reset_batch, **{k: v for k, v in kwargs.items() if k == "skip_trace_precompile"})
        path = "exact" if exact_sampling is not None else ("argmax" if sampling_params is not None else "logits")
        admitted = self.__dict__.pop("_log_decode_path", False)
        if admitted or path != getattr(self, "_decode_path", None):
            logger.info(f"LFM2 decode path: {path} (device argmax | exact device sampler | host-sampled logits)")
            self._decode_path = path
        if path == "exact":
            self._sampler_last_positions = [int(p) for p in start_pos.tolist()]
            out = super().decode_forward(tokens, start_pos, **common)
            return self._decode_exact(out, exact_sampling, width)
        if path == "argmax":
            if not _is_greedy(sampling_params):
                raise ValueError("LFM2 decode: non-greedy device sampling goes through the exact sampler")
            out = super().decode_forward(tokens, start_pos, defer_device_sampling=True, **common)
            ids = out[0][0] if isinstance(out[0], tuple) else out[0]  # untraced runs wrap (ids, None)
            return self._pad_rows(model.process_output_decode(ids, min(B, width), is_tokens=True), B)
        out = super().decode_forward(tokens, start_pos, **common)
        return self._pad_rows(model.process_output_decode(out[0][0], min(B, width)), B)

    @staticmethod
    def _pad_rows(t, B: int):
        """Rows beyond the decode width (inactive rows the caller passed) as zeros."""
        if t.shape[0] >= B:
            return t
        return torch.cat([t, torch.zeros((B - t.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype)])

    # ------------------------------------------------------------------ exact device sampling (tt/sampling.py)
    def exact_sampling_available(self):
        return os.environ.get("LFM2_EXACT_SAMPLING", "1") == "1"

    def exact_sampling_eligible(self, params, num_output_tokens: int) -> bool:
        """vllm_tt_plugin hook: may this request's decode rows use the device sampler?"""
        from .sampling import device_eligible

        return device_eligible(params, num_output_tokens)

    def exact_sampling_spec(self, params, seed: int):
        """vllm_tt_plugin hook: the request's RowSampling (top_k normalized against the vocabulary)."""
        from .sampling import row_sampling

        return row_sampling(params, seed, self.model[0].cfg.vocab)

    def _warmup_exact_sampler(self, width: int, capture: bool):
        """Compile the sampler for `width` rows on scratch logits, then (capture) bind that width's two sampler traces
        (with and without the row-maximum program for unpenalized top_k-disabled rows) to its logits decode trace's
        output and position input (call right after the width's decode warmup)."""
        model = self.model[0]
        sampler = model.exact_sampler
        vocab = model.cfg.vocab
        if not capture:
            logits = ttnn.from_torch(torch.zeros(1, 1, width, vocab, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=model.device,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
            pos = ttnn.from_torch(torch.zeros(width, dtype=torch.int32), dtype=ttnn.int32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=model.device)
            sampler.set_params([None] * width)
            for full in (False, True):
                sampler.sample_decode(logits, width, pos, full)
            ttnn.synchronize_device(model.device)
            ttnn.deallocate(logits)
            ttnn.deallocate(pos)
            sampler.invalidate()
            return
        self._use_width(width)
        outputs, inputs = self.trace_output_decode[False], self.trace_inputs_decode[False]
        if outputs is None or inputs is None:
            raise RuntimeError(f"exact sampling: the width-{width} logits decode trace must be captured first")
        logits, pos = outputs[0][0], inputs[0][1]
        sampler.set_params([None] * width)
        traces = self.__dict__.setdefault("_sampler_traces", {})
        for full in (False, True):
            sampler.sample_decode(logits, width, pos, full)
            ttnn.synchronize_device(model.device)
            trace = ttnn.begin_trace_capture(model.device, cq_id=0)
            sampler.sample_decode(logits, width, pos, full)
            ttnn.end_trace_capture(model.device, trace, cq_id=0)
            traces[width, full] = trace
        self._sampler_logits = self.__dict__.get("_sampler_logits", {})
        self._sampler_logits[width] = logits
        sampler.invalidate()
        logger.info(f"LFM2: captured the exact-sampling traces (width {width})")

    def _decode_exact(self, tt_out, requests, width: int):
        """Draw this step's tokens on device from the decode trace's logits (decode width `width`); host-redraw any
        flagged row (R_HOST) from the same logits with the same Philox uniform."""
        from .sampling import R_CUT, R_HOST, R_TAIL, R_TIE, host_sample

        model = self.model[0]
        sampler = model.exact_sampler
        logits = self._sampler_logits[width]
        if tt_out[0][0] is not logits:
            raise RuntimeError("exact sampling requires the traced decode (trace_mode all / decode_only)")
        if any(r is not None for r in requests[width:]):
            raise ValueError(f"exact sampling: a request row beyond the decode width {width}")
        full = any(r is not None and r.spec.full and not r.spec.penalized for r in requests[:width])
        self._sampler_trace = trace, logits = self._sampler_traces[width, full], logits
        words = [None] * width
        for row, request in enumerate(requests[:width]):
            if request is not None:
                # The history (output counts, prompt flags) is read for penalized rows only; the device still
                # commits other rows' tokens into their slot, so that slot's recorded state is dropped.
                if request.spec.penalized:
                    sampler.ensure_history(row, request.owner, request.outputs, request.prompt or (),
                                           request.spec.uses_prompt)
                else:
                    sampler.invalidate(row)
                words[row] = request.spec.words(row)
        sampler.set_params(words)
        ttnn.execute_trace(model.device, trace, cq_id=0, blocking=False)
        out = sampler.read_out(width)
        tokens = out[:, 0].to(torch.int32).clone()
        host_rows = None
        for row, request in enumerate(requests[:width]):
            if request is None:
                continue
            flags = int(out[row, 1])
            if flags & R_HOST:
                if host_rows is None:
                    host_rows = ttnn.to_torch(logits).reshape(-1, model.cfg.vocab)
                position = int(self._sampler_last_positions[row]) + 1
                tokens[row] = host_sample(host_rows[row], request.spec, request.outputs, request.prompt or (),
                                          position)
                sampler.invalidate(row)
                reason = {R_TIE: "top-k tie group", R_TAIL: "draw in the tail", R_CUT: "cut beyond the candidates"}
                logger.info(f"LFM2 exact sampling: host redraw for row {row} at position {position} "
                            f"({', '.join(v for k, v in reason.items() if flags & k)})")
            elif request.spec.penalized:
                sampler.note_committed(row, request.owner, len(request.outputs) + 1)
        return self._pad_rows(tokens, len(requests))[: max(len(requests), 1)]
