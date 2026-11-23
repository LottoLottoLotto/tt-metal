# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for LFM3-1.7B on one P150 (vllm-tt-plugin; registered through EXTRA_MODELS_DIR, see
vllm_bundle/models/lfm3/vllm_metadata.json).  Ported from the LFM2.5 adapter (lfm2_vllm.py).

- Prefill is model-owned and eager: every request is prefilled (2048-token chunks, bucketed lengths) into its batch
  slot (conv-state row) and its KV pages; the slot comes from the runner's `empty_slots`.
- Decode runs through tt_transformers' Generator: `Lfm3Model.decode_forward` (stock step, or the fused fast step after
  `enable_fast_decode`) is captured as a trace (one for logits, one with the greedy argmax on device) and replayed
  with refreshed host inputs every step.  The step runs at the smallest decode width (model.decode_widths: the fast
  step's 1, 2, 4, ..., max_num_seqs; the stock step only max_num_seqs) holding the active rows (row b = conv-state
  slot b after the runner's slot remap): one trace set per width, all captured at warmup.
- Sampling: all-greedy unpenalized batches take the device argmax (`supports_greedy_only_sample_on_device`, needs
  `sample_on_device_mode: decode_only`); every other batch whose requests are all device-eligible takes the exact
  vLLM-equivalent device sampler (tt/sampling.py: any temperature / top_k incl. disabled / top_p / min_p / penalties,
  counter-based RNG; vllm_tt_plugin asks `exact_sampling_eligible` / `exact_sampling_spec`) bound to the logits
  trace, with single-token host redraws it flags; anything else returns logits to vLLM's host sampler.
- KV: one FullAttentionSpec (block 64, 4 KV heads, head 128) for the 11 attention layers (6 full + 5 sliding: the
  sliding layers keep full-length pages and read only their 256-token window); the short-conv state is model-owned.
- Prefix caching: vllm_tt_plugin's prefix_resume registry with conv-state snapshots (13 x [6, 2048] BF16) at the last
  full 2048-token boundary of every prefill; a resumed request restores it and prefills from there.
- Captured decode traces are replayed directly (inputs staged in place, the page table only when it changed); the
  running rows' next step is launched as soon as a step's tokens are read (decode-ahead, any max_num_seqs; the
  scheduler side is src/docker/plugin-decode-ahead.patch), hiding vLLM's per-step host work behind the device step.
- Prompt-lookup speculation in shared rounds (tt/speculate.py, LFM3_SPECULATE=1, any max_num_seqs): lookup_plan /
  lookup_round for vllm_tt_plugin's speculative_config method "ngram" (src/docker/plugin-speculate.patch); every
  running stream rides a verify round, its drafts priced by the round planner; plain rounds when no draft pays.
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
from vllm_tt_plugin import decode_ahead

from . import step_profile
from .config import Lfm3Config
from .model import BLOCK, PREFILL_BUCKETS, PREFILL_CHUNK, Lfm3Model
from .source import HFSource, QuantSource
from .step_profile import part


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


class _Hist:
    """A request's output ids before a step: base[:n] + extra.  base is the plugin's live list (append-only; the
    plugin appends a step's tokens after decode_forward returns), so a step launched ahead records its history as
    the list's current length plus the tokens it was launched with, without copying the list."""

    __slots__ = ("base", "n", "extra")

    def __init__(self, base, n: int, extra=()):
        self.base, self.n, self.extra = base, n, tuple(extra)

    def __len__(self):
        return self.n + len(self.extra)

    def __iter__(self):
        yield from self.base[: self.n]
        yield from self.extra


class _Step:
    """One launched traced decode step.  Rows are conv-state slots; tokens / positions are the staged inputs (int
    lists, position -1 = inactive row), requests / hists the exact sampler's rows; out (and the sampler's flags) once
    read; dead: rows a prefill rewrote while the step was pending; committed: the drafter commit ran after it; early:
    the EarlyRead handle of its output (enqueued before the commit), None: read it blocking."""

    __slots__ = ("width", "path", "tokens", "positions", "page_table", "requests", "hists", "full", "out", "flags",
                 "dead", "committed", "early")

    def __init__(self, width, path, tokens, positions, page_table, requests=None, hists=None):
        self.width, self.path, self.tokens, self.positions = width, path, tokens, positions
        self.page_table, self.requests, self.hists = page_table, requests, hists
        self.full, self.out, self.flags, self.dead = False, None, None, set()
        self.committed = False
        self.early = None



class Lfm3ServingModel(Lfm3Model):
    """Lfm3Model plus the tt_transformers Generator's model-side interface (decode input staging, the traced
    decode body, output readback) and the exact device sampler."""

    def __init__(self, device, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.mesh_device = device
        self.num_devices = 1

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        from models.tt_transformers.tt.common import copy_host_to_device

        return copy_host_to_device(self.prepare_decode_inputs_host(tokens, current_pos, page_table),
                                   mesh_device=self.device)

    def ttnn_decode_forward(self, tokens, current_pos, rot_mat_idxs=None, page_table=None, kv_cache=None,
                            on_device_logits=False, **kwargs):
        """Generator decode step: (logits, None), or the greedy token ids (bare tensor) when on_device_logits."""
        logits = self.decode_forward(tokens, current_pos, rot_mat_idxs, page_table)
        if on_device_logits:
            ids = self.greedy_tokens(logits)
            ttnn.deallocate(logits)
            return ids
        return logits, None

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        t = ttnn.to_torch(tt_out)
        if is_tokens:
            return t.reshape(-1)[:B].to(torch.int32)
        rows = t.float().reshape(-1, self.cfg.vocab)
        if rows.shape[0] < B * S:
            rows = torch.nn.functional.pad(rows, (0, 0, 0, B * S - rows.shape[0]))
        return rows[: B * S].view(B, S, self.cfg.vocab)

    def switch_mode(self, mode):
        return None

    def init_exact_sampler(self, num_slots: int):
        """The exact vLLM-equivalent device sampler (tt/sampling.py): fixed-address buffers, allocated before any
        trace is captured."""
        if getattr(self, "exact_sampler", None) is None:
            from .sampling import DeviceSampler

            self.exact_sampler = DeviceSampler(self.device, self.cfg.vocab, num_slots=num_slots)
        return self.exact_sampler

    @property
    def serving_widths(self) -> list[int]:
        """Decode widths: the model's (fast step: 1, 2, 4, ..., batch) or, without them, the full slot width."""
        return list(getattr(self, "decode_widths", None) or [self.batch])

    def serving_width(self, rows: int) -> int:
        return next(w for w in self.serving_widths if w >= max(1, rows))


class Lfm3ForCausalLM(Generator):
    """vLLM-compatible wrapper for LFM3-1.7B (Lfm3ForCausalLM) on one Blackhole P150."""

    model_capabilities = {
        # Exact prefix resume (vllm_tt_plugin/prefix_resume.py): conv-state snapshots at 2048-token boundaries.
        "supports_prefix_caching": True,
        # The conv state is a running state: async decode's device-side token/position continuity is not supported.
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        "supports_greedy_only_sample_on_device": True,
        "supports_exact_device_sampling": True,
        # The adapter launches the next decode step of the running rows before the scheduler has run (decode-ahead,
        # LFM3_DECODE_AHEAD=1): vllm_tt_plugin/decode_ahead.py, one lookahead KV slot per running request.
        "supports_decode_ahead": os.environ.get("LFM3_DECODE_AHEAD", "1") == "1",
    }

    # vLLM validates config.json's architecture (Lfm3ForCausalLM: no upstream vLLM implementation, registered to this
    # class by the lfm3_config general plugin) as a text-generation model before vllm_tt_plugin routes it to the TT
    # runner: `__init__` must take vllm_config, and `embed_input_ids` / `forward(input_ids, positions)` /
    # `compute_logits` must exist (as in the Gemma4 TT bridge).  The TT runner only calls prefill_forward /
    # decode_forward; these never run.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError("LFM3 runs on TT through prefill_forward / decode_forward")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("LFM3 runs on TT through prefill_forward / decode_forward")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("LFM3 runs on TT through prefill_forward / decode_forward")

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Paged attention for the 11 attention layers (full and sliding share the layout); the conv state is owned
        by the TT model."""
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
        """KV token pool shared by all users: LFM3_MAX_TOKENS_ALL_USERS, else max_model_len x max_num_seqs."""
        override = os.environ.get("LFM3_MAX_TOKENS_ALL_USERS")
        if override:
            return int(override)
        return int(max_model_len or 32768) * int(max_num_seqs or 1)

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1,
                              optimizations=None, **kwargs):
        if tt_data_parallel != 1 or mesh_device.get_num_devices() != 1:
            raise ValueError("LFM3 serving runs on one P150 (no data / tensor parallelism)")
        if max_batch_size not in (1, 2, 4, 8, 16):
            raise ValueError("max_num_seqs must be 1, 2, 4, 8 or 16 (decode widths, device sampler rows)")
        path = os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL") or hf_config._name_or_path
        src = open_weights(path)
        cache_root = os.environ.get("TT_CACHE_PATH")
        logger.info(f"LFM3: weights {path} ({src.identity}), tensor cache {cache_root}, "
                    f"max_batch_size {max_batch_size}, max_seq_len {max_seq_len}")
        cls._log_default_sampling()
        model = Lfm3ServingModel(mesh_device, src, cfg=Lfm3Config.from_hf(path), max_seq_len=max_seq_len,
                                 batch=max_batch_size, logits_dtype=ttnn.bfloat16, own_kv=False,
                                 cache_root=(Path(cache_root) / "tensors") if cache_root else None,
                                 log=lambda m: None)
        # Every decode step re-stages tokens/positions from the host (no device token feedback into the trace inputs).
        model._tt_vllm_always_refresh_decode_trace_inputs = True
        model.sampling = None
        args = SimpleNamespace(max_batch_size=max_batch_size, max_seq_len=max_seq_len, vocab_size=model.cfg.vocab,
                               mesh_device=mesh_device)
        step_profile.install()
        return cls([model], [args], mesh_device)

    @staticmethod
    def _log_default_sampling():
        """State the server's default sampling (vLLM fills a request's unset fields from it) and its sampler path:
        tools/serve_lfm3.sh passes the resolved defaults as LFM3_DEFAULT_SAMPLING_JSON (the engine's vLLM config is not
        current in the worker when the model is built); else the current vLLM config, if any."""
        from vllm.config import get_current_vllm_config

        from .sampling import MAX_TOP_K

        if os.environ.get("LFM3_DEFAULT_SAMPLING_JSON"):
            d = json.loads(os.environ["LFM3_DEFAULT_SAMPLING_JSON"])
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
        label = os.environ.get("LFM3_DEFAULT_SAMPLING_LABEL", "vLLM model config")
        logger.info(f"LFM3: default sampling for requests that set no sampling fields ({label}): {d or '{}'} "
                    f"(unset fields vLLM-neutral; decode path: {path})")

    # ------------------------------------------------------------------ KV / state
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        model = self.model[0]
        caches = model.allocate_kv_caches(kv_cache_shape)
        # The fused fast decode step (tt/fast/step.py) unless LFM_FAST_DECODE=0; parts: LFM_FAST_PARTS (comma list,
        # default the step's DEFAULT_PARTS).
        if os.environ.get("LFM_FAST_DECODE", "1") == "1":
            parts = [p for p in os.environ.get("LFM_FAST_PARTS", "").split(",") if p] or None
            fast = model.enable_fast_decode(parts)
            logger.info(f"LFM3: fast decode step, parts {list(getattr(fast, 'parts', parts or []))}")
        else:
            logger.info("LFM3: stock decode step")
        logger.info(f"LFM3: KV pool {int(kv_cache_shape[0])} blocks of {BLOCK} tokens x {len(caches)} attention "
                    f"layers; decode widths {model.serving_widths}")
        if self.exact_sampling_available():
            model.init_exact_sampler(model.batch)
        # Prompt-lookup speculation in shared rounds (tt/speculate.py): the fast step, LFM3_SPECULATE=1 (serve_lfm3.sh
        # passes 1 unless LFM3_SPECULATE=0).  The verify page table has the plain decode step's width (vllm_tt_plugin's
        # max_num_blocks_per_req): SDPADecode's partitions depend on it.
        self.speculation = None
        if os.environ.get("LFM3_SPECULATE", "0") == "1":
            if model.fast is None:
                logger.warning("LFM3 speculation needs the fast decode step: disabled")
            else:
                from .speculate import Speculation

                blocks = min(-(-model.max_seq_len // BLOCK), int(kv_cache_shape[0]))
                # The block drafter (tt/drafter.py): LFM3_SPEC_DRAFTER=1 (default) with an export at LFM3_DRAFTER;
                # its K/V share the pool's pages (3 more attention layers' worth of KV per block).
                from .drafter import DEFAULT_DIR

                path = Path(os.environ.get("LFM3_DRAFTER", DEFAULT_DIR))
                if os.environ.get("LFM3_SPEC_DRAFTER", "1") == "1" and (path / "config.json").exists():
                    d = model.enable_drafter(path, page_blocks=blocks, log=logger.info)
                    logger.info(f"LFM3: block drafter {path} (step {d.cfg.get('step')}, {d.identity}), K/V "
                                f"{3 * 2 * int(kv_cache_shape[0]) * 34816 / 2**30:.2f} GiB")
                self.speculation = Speculation(model, blocks, sampler=getattr(model, "exact_sampler", None),
                                               plain_widths=model.serving_widths, log=logger.info)
                logger.info(f"LFM3: speculation in shared rounds (prompt lookup"
                            f"{' + block drafter' if model.drafter is not None else ''}), verify widths "
                            f"{self.speculation.widths}, page table {blocks} blocks")
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
        if getattr(self, "speculation", None) is not None:
            self.speculation.admit(slots)  # their streams are registered after the prefill step
        # A step launched ahead (it runs before this prefill on the device): the prefilled slots' rows are dropped
        # (the prefill rewrites the slot's conv state; their KV write is in a block vLLM never cached), the rest stay
        # pending for the next decode step.
        ahead = self._ahead
        if ahead is not None:
            ahead.dead.update(slots)
            if all(p < 0 or j in ahead.dead for j, p in enumerate(ahead.positions)):
                self._drop_ahead()
        self._log_decode_path = True  # the next decode step states its path (a line per admission, not per step)
        out = []
        for i in range(B):
            L = int(prompt_lens[i]) if prompt_lens is not None else int(tokens.shape[1])
            start = int(start_pos[i]) if start_pos is not None else 0
            if start % PREFILL_CHUNK:
                raise ValueError(f"LFM3 prefill resumes only at {PREFILL_CHUNK}-token boundaries (got {start})")
            resume, snapshot = self._prefix_resume_args(tokens[i], page_table[i], start)
            if resume is not None:
                model.restore_slot_state(slots[i], resume)
            logits = model.prefill(tokens[i, start:L], start=start, slot=slots[i], page_table_row=page_table[i],
                                   snapshot=snapshot)
            out.append(logits)
        logits = torch.stack(out).unsqueeze(1)  # [B, 1, V]
        if sampling_params is not None:
            if not _is_greedy(sampling_params):
                raise ValueError("LFM3 prefill device sampling supports greedy only")
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
        """Compile every prefill program the server can hit: each bucket at chunk starts 0 (fresh conv state) and
        2048 (resumed state, KV in earlier pages).  Prefill programs depend only on the bucket: the chunk start is a
        device tensor read by the attention programs and the page-table row is padded to max_seq_len / 64 blocks.
        Prefill is never traced."""
        if self.already_warmed_up_prefill:
            return
        model = self.model[0]
        blocks = int(kv_cache[0][0].shape[0]) - 1 if kv_cache else model.trash_block
        starts = [0, PREFILL_CHUNK]
        last = PREFILL_BUCKETS[-1]
        need = (starts[-1] + last) // BLOCK
        if blocks < need:
            raise ValueError(f"KV pool of {blocks} blocks cannot hold the prefill warmup ({need} blocks)")
        row = torch.arange(need, dtype=torch.int32)
        for start in starts:
            for n in PREFILL_BUCKETS:
                if start + n > model.max_seq_len:
                    continue
                logger.info(f"LFM3 prefill warmup: {n} tokens at {start}")
                model.prefill(torch.full((n,), 1000, dtype=torch.int64), start=start, slot=0, page_table_row=row)
        model.reset()
        from vllm_tt_plugin.prefix_resume import REGISTRY

        # Prefix resume: snapshots at 2048-token boundaries (active only when vLLM prefix caching is on).
        REGISTRY.configure(PREFILL_CHUNK, BLOCK, int(os.environ.get("LFM3_PREFIX_SNAPSHOTS", "16")))
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
        for width in model.serving_widths:
            logger.info(f"LFM3 decode warmup: width {width} (capture={capture})")
            result = super().warmup_model_decode(*args, **dict(kwargs, max_batch_size=width))
            if self.exact_sampling_available():
                self._warmup_exact_sampler(width, capture)
        if capture and getattr(self, "speculation", None) is not None:
            self.speculation.capture()
            if model.drafter is not None:
                # the drafter commit after each plain decode trace, bound to its position / page-table inputs
                for width in model.serving_widths:
                    for mode in (True, False):
                        inputs = self._width_traces[width][1][mode]
                        if inputs is not None:
                            _, pos_d, _, pt_d = inputs[0]
                            model.drafter.capture_commit(("plain", width, mode), width, pos_d, pt_d)
                logger.info(f"LFM3: captured the drafter's commit traces ({len(model.drafter.commit_traces)})")
        if not capture and model.batch > 1:
            model.remap_slots(list(range(model.batch)))  # compile the device remap (rows condensed while serving)
        model.reset()  # warmup decoded garbage into every conv-state slot
        self._drop_ahead()
        # decode-ahead once every trace is captured; LFM3_DECODE_AHEAD=0 disables it
        self._ahead_ok = capture and os.environ.get("LFM3_DECODE_AHEAD", "1") == "1"
        self._ahead_stats = {"launched": 0, "used": 0, "skipped": 0, "drafts": 0, "completed": 0}
        if self._ahead_ok:
            logger.info("LFM3: decode-ahead on (the running rows' next step is launched as soon as a step's tokens are "
                        "read; the TT scheduler allocates one lookahead KV slot per running request)")
        return result

    # ------------------------------------------------------------------ prompt-lookup speculation (tt/speculate.py)
    # vllm_tt_plugin (src/docker/plugin-speculate.patch) calls these for speculative_config method "ngram": lookup_plan
    # after every step (every scheduled request that may speculate -> its drafts for the next round), lookup_round for a
    # decode step whose requests the scheduler gave drafts (a shared verify round: every running request rides it).
    def sampled_speculation_available(self) -> bool:
        """Sampled requests (the exact device sampler) may speculate too (asked before allocate_kv_cache)."""
        return os.environ.get("LFM3_SPECULATE", "0") == "1" and self.exact_sampling_available()

    def lookup_plan(self, entries, after_prefill: bool = False) -> list[list[int]]:
        """entries [(owner, slot, prompt ids, output ids, remaining tokens, depth, stop ids, n_min, n_max)] -> drafts
        per entry for the next round (Speculation.register): the plan made at the read point of the step just run
        (decode-ahead), else a fresh one; none while a plain step launched ahead is pending or after a prefill step
        (a new request's first round is plain)."""
        spec = getattr(self, "speculation", None)
        if spec is None:
            return [[] for _ in entries]
        return spec.register(entries, fresh=self._ahead is None and not after_prefill)

    def lookup_round(self, tokens, drafts, positions, page_table, requests=None, slot_remap=None):
        """One shared verify round over the running rows 0..N-1 (slot = row after slot_remap): tokens / positions [N],
        drafts per row (lists, possibly empty), page_table int32 [>= N, blocks] (row = request), requests: the exact
        sampler's SamplingRequest per row (None: every row greedy, device argmax) -> the emitted tokens per row
        (accepted drafts + one).  The next plain step is launched ahead when the next round has no drafts."""
        self.mode = Mode.DECODE
        if self._ahead is not None:
            raise RuntimeError("LFM3 decode-ahead: a verify round was scheduled while the next plain step is running")
        model, spec = self.model[0], self.speculation
        if slot_remap is not None and model.batch > 1:
            remap = [int(s) for s in torch.as_tensor(slot_remap).reshape(-1).tolist()]
            model.remap_slots(remap)
            spec.remap(remap)
        n = len(tokens)
        path = "argmax" if requests is None else "exact"
        if self.__dict__.pop("_log_decode_path", False) or path != getattr(self, "_decode_path", None):
            logger.info(f"LFM3 decode path: {path} with prompt-lookup speculation (shared verify rounds when drafts "
                        f"pay)")
            self._decode_path = path
        if step_profile.PROF is not None:
            step_profile.PROF.decoded(f"verify-{path}", n)
        pt = torch.as_tensor(page_table)
        pt = pt.to(torch.int32) if pt.dtype != torch.int32 else pt
        with part("stage"):
            rec = spec.launch(tokens, spec.trim(spec.resolve(drafts, positions)), positions, pt, requests)
        spec.prepare(range(n))
        with part("wait"):
            emitted = spec.collect(rec)
        if self._ahead_ok:
            width = model.serving_width(n)
            nxt = [int(positions[i]) + len(emitted[i]) for i in range(n)] + [-1] * (width - n)
            toks = [int(e[-1]) for e in emitted] + [0] * (width - n)
            self._launch_ahead(width, path, toks, nxt, self._width_page_table(pt, width), requests,
                               [tuple(int(t) for t in e) for e in emitted] + [()] * (width - n))
        return emitted

    @staticmethod
    def _width_page_table(pt, width: int):
        """The decode page table's first `width` rows (zero rows appended when it has fewer), int32 contiguous."""
        if pt.shape[0] >= width:
            return pt[:width].contiguous()
        return torch.cat([pt, torch.zeros(width - pt.shape[0], pt.shape[1], dtype=torch.int32)]).contiguous()

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
        [B, 1, vocab].  Captured widths / paths replay their trace directly (_decode_traced: inputs staged in place,
        the page table only when it changed, decode-ahead); the rest (warmup compile / capture, untraced runs) goes
        through the Generator."""
        model = self.model[0]
        if getattr(self, "speculation", None) is not None:
            self.speculation.note_plain_step()
        # A step launched ahead is read only after the host work that does not need its tokens (input normalization,
        # matching, staging the next step's inputs): its device time hides that work.  Its rows are the slots before
        # this call's remap.
        ahead, self._ahead = self._ahead, None
        if ahead is not None:
            decode_ahead.set_pending(False)
        remap = None
        if slot_remap is not None and model.batch > 1:
            remap = [int(s) for s in torch.as_tensor(slot_remap).reshape(-1).tolist()]
        B = int(tokens.shape[0])
        if B > model.batch:
            raise ValueError(f"{B} decode rows exceed the {model.batch} conv-state slots")
        toks = [int(t) for t in torch.as_tensor(tokens).reshape(B, -1)[:, 0].tolist()]
        poss = [int(p) for p in torch.as_tensor(start_pos).reshape(-1).tolist()]
        live = [i for i, p in enumerate(poss) if p >= 0]
        # rows [0, n) hold every active row; the step runs at the smallest decode width >= n, the extra rows inactive
        # (position -1: no KV write, no attention; their outputs are ignored)
        width = model.serving_width(live[-1] + 1 if live else 1)
        toks = (toks + [0] * width)[:width]
        poss = (poss + [-1] * width)[:width]
        if page_table is not None:
            pt = torch.as_tensor(page_table)
            page_table = self._width_page_table(pt.to(torch.int32) if pt.dtype != torch.int32 else pt, width)
        path = "exact" if exact_sampling is not None else ("argmax" if sampling_params is not None else "logits")
        admitted = self.__dict__.pop("_log_decode_path", False)
        if admitted or path != getattr(self, "_decode_path", None):
            logger.info(f"LFM3 decode path: {path} (device argmax | exact device sampler | host-sampled logits)")
            self._decode_path = path
        if path == "argmax" and not _is_greedy(sampling_params):
            raise ValueError("LFM3 decode: non-greedy device sampling goes through the exact sampler")
        spec = getattr(self, "speculation", None)
        if spec is not None:
            spec.mode = "argmax" if path == "argmax" else "exact"  # prices the next round
        requests = None
        if path == "exact":
            if any(r is not None for r in exact_sampling[width:]):
                raise ValueError(f"exact sampling: a request row beyond the decode width {width}")
            requests = (list(exact_sampling[:width]) + [None] * width)[:width]
        rows = min(B, width)
        if step_profile.PROF is not None:
            step_profile.PROF.decoded(path, len(live))
        if enable_trace and self._traced(width, path):
            out = self._decode_traced(width, path, toks, poss, page_table, requests, rows, ahead, remap)
        else:
            if ahead is not None:
                raise RuntimeError("LFM3 decode-ahead: an untraced decode step while a launched step is pending")
            if remap is not None:
                model.remap_slots(remap)
                if spec is not None:
                    spec.remap(remap)
            out = self._decode_generic(width, path, toks, poss, page_table, requests, rows, dict(
                kv_cache=kv_cache, enable_trace=enable_trace, reset_batch=reset_batch,
                **{k: v for k, v in kwargs.items() if k == "skip_trace_precompile"}))
        if path == "exact":
            return self._pad_rows(out, len(exact_sampling))[: max(len(exact_sampling), 1)]
        if path == "argmax":
            return self._pad_rows(out[:rows], B)
        return self._pad_rows(out, B)

    def _decode_generic(self, width, path, toks, poss, page_table, requests, rows, common):
        """The Generator's decode (warmup compile / trace capture, untraced runs)."""
        model = self.model[0]
        self._pt_staged = {}  # the Generator re-stages every trace input
        self._use_width(width)
        common = dict(common, page_table=page_table, read_from_device=False)
        tokens = torch.tensor(toks, dtype=torch.int32).reshape(width, 1)
        start_pos = torch.tensor(poss, dtype=torch.int32)
        if path == "exact":
            out = super().decode_forward(tokens, start_pos, **common)
            if out[0][0] is not self._sampler_logits[width]:
                raise RuntimeError("exact sampling requires the traced decode (trace_mode all / decode_only)")
            hists = [_Hist(r.outputs, len(r.outputs)) if r is not None else None for r in requests]
            step = _Step(width, path, toks, poss, page_table, requests, hists)
            step.full = self._exact_launch(requests, width, hists)
            self._read(step)
            self._finish_exact(step, range(width))
            return torch.tensor(step.out, dtype=torch.int32)
        if path == "argmax":
            out = super().decode_forward(tokens, start_pos, defer_device_sampling=True, **common)
            ids = out[0][0] if isinstance(out[0], tuple) else out[0]  # untraced runs wrap (ids, None)
            return model.process_output_decode(ids, width, is_tokens=True)
        out = super().decode_forward(tokens, start_pos, **common)
        return model.process_output_decode(out[0][0], rows)

    # ------------------------------------------------------------------ traced decode + decode-ahead
    # A captured (width, path) replays its trace directly: the token / position inputs are copied into the trace's
    # input buffers, the page table only when it changed and no other decode trace ran since it was staged (_launch),
    # the rope index only for steps that read it (the fast step's attention fronts take the position itself).
    #
    # Decode-ahead (LFM3_DECODE_AHEAD=1, any max_num_seqs; src/NOTES.md section 17): as soon as a step's tokens are
    # read, the next step of the same rows is launched (the tokens just read at positions + 1, the same pages and
    # sampling requests, each history extended by its token), so the device runs it while vLLM updates the requests,
    # schedules and builds the next input; the next decode_forward reads its tokens instead of launching.  Rows are
    # conv-state slots; the call's slot_remap (finished rows condensed away) maps each of its rows to the launched
    # step's row.  Per active row of the call:
    #  - covered by the launched step: it must be that request's next step (same token, position, pages up to its
    #    block, sampler owner / spec / history length; anything else raises, the row's state having advanced) -> the
    #    launched step's token;
    #  - not covered (a prefill admitted the request while the step was pending; in serving the scheduler runs the
    #    pending decode turn before any prefill, vllm_tt_plugin/decode_ahead.py): those rows run in a step of their
    #    own with the covered rows inactive and their conv state saved and restored around it (_complete).
    # Rows of the launched step the call does not have (finished, aborted, preempted, re-prefilled) are dropped: their
    # KV write went to position + 1 of the request's own last block, which vLLM neither shares nor caches before
    # that position is computed (a later owner of the block writes every position before reading it); their
    # conv-state slot is rewritten by the prefill that reuses it (fresh start or a restored snapshot); their sampler
    # history slot is invalidated.  The device RNG is counter-based (seed, position): no draw order to keep.
    # Not launched ahead: host-sampled (logits) steps, right before a prefill turn (decode_ahead.prefill_due), at the
    # last position, at a KV block boundary unless the scheduler allocates a lookahead slot (decode_ahead.lookahead),
    # and when the speculation planner gives the next round drafts (a shared verify round: decode-ahead runs for plain
    # rounds only; a verify round reads its result, then plans and may launch the next plain step ahead).
    _ahead = None  # the step launched ahead (_Step) or None
    _ahead_ok = False  # decode-ahead enabled (after the warmup captures)
    _ahead_stats = None

    def _traced(self, width: int, path: str) -> bool:
        stores = self.__dict__.get("_width_traces", {}).get(width)
        if stores is None or stores[0][path == "argmax"] is None:
            return False
        if path == "exact":
            return (width, False) in self.__dict__.get("_sampler_traces", {})
        return True

    def _needs_rope_index(self, width: int) -> bool:
        """Does the decode step at `width` read the rope-index input (the stock step / fallbacks do)?"""
        cache = self.__dict__.setdefault("_rope_index_needed", {})
        if width not in cache:
            model = self.model[0]
            f = model.fast
            cache[width] = f is None or any(layer.attn and not (f._attn_ok(layer, width) or f._anfront_ok(layer, width))
                                            for layer in model.layers)
        return cache[width]

    def _stage_inputs(self, width, path, positions, page_table):
        """Copy a step's positions (+ the rope index when the step reads it) and page table (only when it changed
        since it was staged for this trace and no other decode trace ran since: see _launch) into the width's trace
        inputs.  The copies queue behind whatever runs on the device (one command queue), so the next step's inputs
        may be staged while the step launched ahead still runs."""
        mode = path == "argmax"
        _, pos_d, ridx_d, pt_d = self._width_traces[width][1][mode][0]
        rk = dict(layout=ttnn.ROW_MAJOR_LAYOUT)
        with part("stage"):
            pos = torch.tensor(positions, dtype=torch.int32)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(pos, dtype=ttnn.int32, **rk), pos_d)
            if self._needs_rope_index(width):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(pos.clamp(min=0).reshape(1, -1), dtype=ttnn.uint32, **rk), ridx_d)
            if page_table is not None:
                staged = self.__dict__.setdefault("_pt_staged", {})
                last = staged.get((width, mode))
                if last is None or last.shape != page_table.shape or not torch.equal(last, page_table):
                    ttnn.copy_host_to_device_tensor(ttnn.from_torch(page_table, dtype=ttnn.int32, **rk), pt_d)
                    staged[(width, mode)] = page_table.clone()

    def _launch(self, width, path, tokens, positions, page_table, requests=None, hists=None, staged=False) -> _Step:
        """Stage one traced step's inputs (int lists [width]; page table int32 [width, blocks]; staged: positions /
        page table / sampler params already copied) and enqueue its trace (+ the exact sampler's for path exact:
        requests / hists [width])."""
        model = self.model[0]
        mode = path == "argmax"
        trace_ids, inputs, _ = self._width_traces[width]
        if not staged:
            self._stage_inputs(width, path, positions, page_table)
        with part("stage"):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.tensor([tokens], dtype=torch.int32), dtype=ttnn.uint32,
                                layout=ttnn.ROW_MAJOR_LAYOUT), inputs[mode][0][0])
        with part("launch"):
            ttnn.execute_trace(model.device, trace_ids[mode][0], cq_id=0, blocking=False)
        # The page table staged for another (width, path) is unknown from here on: the Generator allocates each decode
        # trace's inputs right before capturing it, after the earlier traces' intermediates were released, so a trace
        # captured later can have its inputs inside an earlier trace's intermediates, which every replay of that trace
        # overwrites (seen: the width-1 argmax trace overwrote the width-1 logits trace's page table; a later exact step
        # with the same pages skipped the copy and read KV through garbage block ids).  Tokens and positions are copied
        # before every launch.  Verify traces (captured after every decode-trace input existed) and the sampler traces
        # (no intermediates) cannot overwrite a decode trace's inputs.
        known = self.__dict__.get("_pt_staged")
        if known and (len(known) > 1 or (width, mode) not in known):
            keep = known.get((width, mode))
            known.clear()
            if keep is not None:
                known[(width, mode)] = keep
        step = _Step(width, path, tokens, positions, page_table, requests, hists)
        if path == "exact":
            step.full = self._exact_launch(requests, width, hists, params=not staged)
        spec = getattr(self, "speculation", None)
        if spec is not None and spec.drafter is not None:
            # the drafter's context: its commit right after the step (it reads the step's taps and inputs)
            live = [i for i, p in enumerate(positions) if p >= 0]
            key = ("plain", width, mode)
            step.committed = path != "logits" and key in spec.drafter.commit_traces and spec.commit_wanted(live)
            if step.committed:
                # the step's tokens are copied to the host before the commit, which runs while the host reads them
                out = (self._width_traces[width][2][True][0] if path == "argmax"
                       else self.model[0].exact_sampler.out)
                step.early = spec.early.start([out])
                with part("launch"):
                    spec.drafter.launch_commit(key)
            spec.note_step([(i, positions[i], 1) for i in live], step.committed)
        self._last_step = step
        return step

    def _read(self, step: _Step):
        """Blocking read of a launched argmax / exact step: step.out (token ids [width]), step.flags (exact)."""
        with part("wait"):
            host = self.speculation.early.read(step.early)[0] if step.early is not None else None
            if step.path == "argmax":
                out = host if host is not None else ttnn.to_torch(self._width_traces[step.width][2][True][0])
            else:
                out = self.model[0].exact_sampler.read_out(step.width, host)
        if step.path == "argmax":
            step.out = out.reshape(-1)[: step.width].to(torch.int32).tolist()
        else:
            step.out, step.flags = out[:, 0].tolist(), out[:, 1].tolist()

    def _read_logits(self, step: _Step, rows: int):
        with part("wait"):
            return self.model[0].process_output_decode(self._width_traces[step.width][2][False][0][0], rows)

    def _drop_ahead(self):
        """Discard the step launched ahead unread (warmup, a prefill over all its rows)."""
        ahead, self._ahead = self._ahead, None
        if ahead is None:
            return
        decode_ahead.set_pending(False)
        if ahead.requests is not None:
            for row, request in enumerate(ahead.requests):
                if request is not None:
                    self.model[0].exact_sampler.invalidate(row)

    def _decode_traced(self, width, path, toks, poss, page_table, requests, rows, ahead, remap):
        """One decode step over rows [0, width): its tokens (int32 [width]; logits [rows, 1, vocab] for path logits)
        from the step launched ahead, a fresh launch, or both (_complete); then the next step is launched ahead.
        With a step launched ahead covering every row, its read comes after the match and after staging the next
        step's positions / page table / sampler params (none of which needs its tokens)."""
        self.mode = Mode.DECODE
        used, need = {}, [i for i, p in enumerate(poss) if p >= 0]
        nxt = None
        if ahead is not None:
            with part("post"):
                used, need = self._match(ahead, width, path, toks, poss, page_table, requests, remap)
            if used and not need:
                nxt = [p + 1 if p >= 0 else -1 for p in poss]
                if self._ahead_allowed(width, path, nxt):
                    self._stage_inputs(width, path, nxt, page_table)
                    if path == "exact":
                        self._exact_params(requests, width)
                else:
                    nxt = None
            self._spec_prepare()
            self._read(ahead)
        if remap is not None:
            self.model[0].remap_slots(remap)
            if getattr(self, "speculation", None) is not None:
                self.speculation.remap(remap)
        if ahead is not None and ahead.path == "exact":
            with part("post"):
                self._finish_exact(ahead, used)
        # the drafter's ctx32 rows of the step just read (call row -> its row in that step; committed steps only)
        if used and not need:
            if path == "logits":
                raise RuntimeError("LFM3 decode-ahead: a host-sampled step over rows launched ahead")
            self._ahead_stats["used"] += 1
            out = [0] * width
            for j, i in used.items():
                out[i] = ahead.out[j]
            ctx_rows = {i: j for j, i in used.items()} if ahead.committed else {}
        elif used:
            out = self._complete(width, path, toks, poss, page_table, requests, rows, ahead, used, need)
            if path == "logits":
                return out
            ctx_rows = {i: i for i in need} if self._last_step.committed else {}
        else:
            hists = None
            if path == "exact":
                hists = [_Hist(r.outputs, len(r.outputs)) if r is not None else None for r in requests]
            step = self._launch(width, path, toks, poss, page_table, requests, hists)
            if path == "logits":
                return self._read_logits(step, rows)
            self._spec_prepare()
            self._read(step)
            if path == "exact":
                with part("post"):
                    self._finish_exact(step, range(width))
            out = step.out
            ctx_rows = {i: i for i in need} if step.committed else {}
        if self._ahead_ok:
            self._launch_ahead(width, path, [t if p >= 0 else 0 for t, p in zip(out, poss)],
                               nxt or [p + 1 if p >= 0 else -1 for p in poss], page_table, requests,
                               [(t,) for t in out], staged=nxt is not None, ctx_rows=ctx_rows)
        return torch.tensor(out, dtype=torch.int32)

    def _match(self, ahead: _Step, width, path, toks, poss, page_table, requests, remap):
        """Pair the call's active rows with the launched step's rows -> ({ahead row: call row} for the covered rows,
        [call rows not covered]).  A covered row that is not its request's next step raises."""
        used, need = {}, []
        apos, atok = ahead.positions, ahead.tokens
        same_pages = (remap is None and page_table is not None and ahead.page_table is not None
                      and ahead.page_table.shape == page_table.shape and torch.equal(ahead.page_table, page_table))
        for i, p in enumerate(poss):
            if p < 0:
                continue
            j = remap[i] if remap is not None else i
            if j >= ahead.width or apos[j] < 0 or j in ahead.dead:
                need.append(i)
                continue
            why = None
            if apos[j] != p or atok[j] != toks[i]:
                why = f"token {toks[i]} at {p} vs launched {atok[j]} at {apos[j]}"
            elif not same_pages and not self._same_pages(ahead.page_table, j, page_table, i, p):
                why = "pages differ"
            elif path != "logits" and not self._same_sampling(ahead, j, requests[i] if path == "exact" else None):
                why = "sampling request differs"
            if why is not None:
                raise RuntimeError(f"LFM3 decode-ahead: row {i} (launched as row {j}) is not the step launched "
                                   f"ahead: {why}")
            used[j] = i
        return used, need

    @staticmethod
    def _same_pages(mine, j, theirs, i, position) -> bool:
        """The pages up to the step's own block (the rest of the row does not change the step)."""
        if (mine is None) != (theirs is None):
            return False
        blocks = position // BLOCK + 1
        return mine is None or torch.equal(mine[j, :blocks], theirs[i, :blocks])

    @staticmethod
    def _same_sampling(ahead: _Step, j, request) -> bool:
        """Same sampler semantics for the row: the same request (owner, spec, history length), or greedy unpenalized
        on both sides when one side is the device argmax (exact-sampler greedy = argmax, ties to the lowest id).  (A
        host-sampled call carries no requests: its covered rows keep their device-sampled token, _complete.)"""
        mine = ahead.requests[j] if ahead.requests is not None else None
        if mine is not None and request is not None:
            return (mine.owner == request.owner and mine.spec == request.spec
                    and len(ahead.hists[j]) == len(request.outputs))
        r = mine if mine is not None else request
        return r is None or (r.spec.greedy and not r.spec.penalized)

    def _complete(self, width, path, toks, poss, page_table, requests, rows, ahead, used, need):
        """Rows the step launched ahead does not cover: run them in a step of their own (width enough for them), the
        covered rows inactive there (position -1: no KV write, no sampler commit) with their already advanced conv
        state saved before and restored after it.  -> tokens [width] (logits [rows, 1, vocab] for path logits: the
        covered rows as one-hot rows of their token)."""
        model = self.model[0]
        self._ahead_stats["completed"] += 1
        w = model.serving_width(max(need) + 1)
        needed = set(need)
        covered = sorted(i for i in used.values() if i < w)
        logger.info(f"LFM3 decode-ahead: {len(need)} row(s) admitted while a step was pending, completed in a step "
                    f"of width {w} ({len(covered)} covered row(s) inside it saved / restored)")
        saved = model.save_slots(covered) if covered else None
        toks2 = [toks[i] if i in needed else 0 for i in range(w)]
        poss2 = [poss[i] if i in needed else -1 for i in range(w)]
        reqs2 = hists2 = None
        if path == "exact":
            reqs2 = [requests[i] if i in needed else None for i in range(w)]
            hists2 = [_Hist(r.outputs, len(r.outputs)) if r is not None else None for r in reqs2]
        step = self._launch(w, path, toks2, poss2, page_table[:w] if page_table is not None else None, reqs2, hists2)
        if path == "logits":
            logits = self._read_logits(step, w)
        else:
            self._read(step)
            if path == "exact":
                self._finish_exact(step, needed)
        if saved is not None:
            model.restore_slots(covered, saved)
        if path == "logits":
            full = torch.full((rows, 1, model.cfg.vocab), float("-inf"), dtype=logits.dtype)
            for i in need:
                full[i] = logits[i]
            for j, i in used.items():
                if i < rows:
                    full[i, 0, ahead.out[j]] = 0.0
            return full
        out = [0] * width
        for i in need:
            out[i] = step.out[i]
        for j, i in used.items():
            out[i] = ahead.out[j]
        return out

    def _ahead_allowed(self, width, path, positions) -> bool:
        """May the rows' next step (positions: its inputs, -1 = inactive) be launched ahead?  (Not: decode-ahead off,
        host-sampled steps, right before a prefill turn, past the last position, across a KV block boundary without
        the scheduler's lookahead slot, or with no active row.)"""
        if not self._ahead_ok or path == "logits" or not self._traced(width, path) or decode_ahead.prefill_due():
            return False
        model = self.model[0]
        lookahead = decode_ahead.lookahead()
        active = False
        for p in positions:
            if p < 0:
                continue
            if p >= model.max_seq_len or (not lookahead and p % BLOCK == 0):
                return False
            active = True
        return active

    def _spec_prepare(self):
        """Speculation: index the streams' contexts while the device runs the step (the query after the read then
        costs only its keys)."""
        spec = getattr(self, "speculation", None)
        if spec is not None:
            with part("post"):
                spec.prepare(range(len(spec.rows)))

    def _launch_ahead(self, width, path, tokens, positions, page_table, requests, extras, staged=False,
                      ctx_rows=None):
        """Launch the next step of the rows (tokens / positions: its inputs, -1 = inactive; extras[row]: the tokens
        the row's request emitted since its live output list) unless it does not apply (see the section comment).
        With speculation the next round is planned first (Speculation.precompute: every live row's emitted tokens
        extend its stream, a draft pass may be launched for the streams the drafter serves, read when the verify round
        is launched (ctx_rows: each row's drafter context row after a plain step); the plan is cached for
        vllm_tt_plugin's lookup_plan): a round with drafts is a verify round, not launched ahead.  staged:
        _decode_traced already checked it and staged its positions / page table / sampler params."""
        st = self._ahead_stats
        spec = getattr(self, "speculation", None)
        if spec is not None and path != "logits":
            with part("post"):
                rows = [(i, extras[i], p - len(extras[i])) for i, p in enumerate(positions) if p >= 0]
                verify = spec.precompute(rows, page_table=page_table, ctx_rows=ctx_rows)
            if verify:
                st["drafts"] += 1  # the next round verifies drafts: not launched ahead
                return
        if not staged and not self._ahead_allowed(width, path, positions):
            st["skipped"] += 1
            return
        hists = None
        if path == "exact":
            requests = (list(requests[:width]) + [None] * width)[:width]
            hists = [_Hist(r.outputs, len(r.outputs), extras[row]) if r is not None else None
                     for row, r in enumerate(requests)]
        self._ahead = self._launch(width, path, tokens, positions, page_table, requests, hists, staged=staged)
        decode_ahead.set_pending(True)
        st["launched"] += 1
        if st["launched"] % 16384 == 0:
            logger.info(f"LFM3 decode-ahead: {st}")

    @staticmethod
    def _pad_rows(t, B: int):
        """Rows beyond the decode width (inactive rows the caller passed) as zeros."""
        if t.shape[0] >= B:
            return t
        return torch.cat([t, torch.zeros((B - t.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype)])

    # ------------------------------------------------------------------ exact device sampling (tt/sampling.py)
    def exact_sampling_available(self):
        return os.environ.get("LFM3_EXACT_SAMPLING", "1") == "1"

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
        logger.info(f"LFM3: captured the exact-sampling traces (width {width})")

    def _exact_params(self, requests, width: int):
        """Copy the rows' sampler parameters (one 32-word row each, cached per request spec and row; the copy is
        skipped when unchanged) into the sampler's params buffer.  Queued behind any running step, like
        _stage_inputs."""
        words = [None] * width
        cache = self.__dict__.setdefault("_words", {})
        if len(cache) > 4096:
            cache.clear()
        with part("stage"):
            for row, request in enumerate(requests[:width]):
                if request is not None:
                    w = cache.get((request.spec, row))
                    if w is None:
                        w = cache[request.spec, row] = request.spec.words(row)
                    words[row] = w
            self.model[0].exact_sampler.set_params(words)

    def _exact_launch(self, requests, width: int, hists, params: bool = True) -> bool:
        """Enqueue the exact sampler on the width's decode-trace logits (right after that trace): per-row params
        (params False: already staged by _exact_params) and penalty histories (hists[row]: the request's output ids
        before this step).  -> `full` (the sampler trace with the row-maximum program)."""
        model = self.model[0]
        sampler = model.exact_sampler
        full = any(r is not None and r.spec.full and not r.spec.penalized for r in requests[:width])
        self._sampler_trace = trace, _ = self._sampler_traces[width, full], self._sampler_logits[width]
        if params:
            self._exact_params(requests, width)
        with part("stage"):
            for row, request in enumerate(requests[:width]):
                if request is not None:
                    spec = request.spec
                    # The history (output counts, prompt flags) is read for penalized rows only; the device still
                    # commits other rows' tokens into their slot, so that slot's recorded state is dropped.
                    if spec.penalized:
                        sampler.ensure_history(row, request.owner, hists[row], request.prompt or (), spec.uses_prompt)
                    else:
                        sampler.invalidate(row)
        with part("launch"):
            ttnn.execute_trace(model.device, trace, cq_id=0, blocking=False)
        return full

    def _finish_exact(self, step: _Step, used):
        """Sampler bookkeeping of a read exact step.  Rows in `used` (their token is returned): a row the device
        flagged (R_HOST) is redrawn on the host from the same logits with the same Philox uniform (position + 1: the
        drawn token's position), else its history commit is noted; every other request row's history slot is
        invalidated (the device committed a token the request will not see)."""
        from .sampling import R_CUT, R_HOST, R_TAIL, R_TIE, host_sample

        model = self.model[0]
        sampler = model.exact_sampler
        host_rows = None
        for row, request in enumerate(step.requests[: step.width]):
            if request is None:
                continue
            if row not in used:
                sampler.invalidate(row)
                continue
            flags = int(step.flags[row])
            if flags & R_HOST:
                if host_rows is None:
                    host_rows = ttnn.to_torch(self._sampler_logits[step.width]).reshape(-1, model.cfg.vocab)
                position = step.positions[row] + 1
                step.out[row] = host_sample(host_rows[row], request.spec, step.hists[row], request.prompt or (),
                                            position)
                sampler.invalidate(row)
                reason = {R_TIE: "top-k tie group", R_TAIL: "draw in the tail", R_CUT: "cut beyond the candidates"}
                logger.info(f"LFM3 exact sampling: host redraw for row {row} at position {position} "
                            f"({', '.join(v for k, v in reason.items() if flags & k)})")
            elif request.spec.penalized:
                sampler.note_committed(row, request.owner, len(step.hists[row]) + 1)
