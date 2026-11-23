# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM wrapper for Qwen3.5/3.6 on Blackhole — a thin tt_transformers Generator subclass.

Hybrid model: 8 paged-KV attention + 24 GDN recurrent-state layers. GDN forbids token-padding and
isn't position-general, so prefill is model-owned (masked-bucket for short prompts / chunk-outer
trace for long, via prefill_dispatch) while Generator drives decode only. GDN + KV state is
model-bound, so the kv_cache contract param is accepted but unused.
"""

import math
import os
from collections import defaultdict
from typing import Mapping, Optional

import torch
from loguru import logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.demos.blackhole.qwen36.tt.common import create_tt_model
from models.demos.blackhole.qwen36.tt import weight_prefetch
from models.demos.blackhole.qwen36.tt.generator_interface import prefill_dispatch, warmup_decode_buckets
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.generator import Generator

_PREFILL_WARMUP_CHUNK = 2048
_PREFILL_WARMUP_BUCKET = 4096
_BLOCK_SIZE = 64
# Multi-user DFlash2 (shared verify rounds, shared_serving.py): decode slots served with the batched drafter.
SHARED_SLOTS = (4, 8)


class TT_Qwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        if os.environ.get("QWEN36_GENERIC_BASELINE") == "1":
            return {"image": 0, "video": 0}
        # Any number of images and videos per request, mixed (vLLM's default limit applies unless
        # --limit-mm-per-prompt sets one): the tower runs per image / video frame and the rows are
        # spliced in prompt order over both placeholder ids (Qwen36Model.get_visual_features).
        return {"image": None, "video": None}


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36ForCausalLM(Generator, SupportsMultiModal):
    """vLLM-compatible wrapper for Qwen3.5-9B on Blackhole P150."""

    # Decode bucketing keeps several traces live and refreshes the selected
    # bucket's inputs before replay, so their I/O buffers may safely overlap.
    _tt_allow_decode_trace_buffer_reuse = True

    # supports_async_decode=False: async decode assumes on-device token/position continuity, which
    # corrupts Qwen's GDN scan. The generic single-device sampler path is force-argmax only; random and
    # penalized requests use the exact vLLM-equivalent device sampler (device_sampling.py) when eligible,
    # including lossless speculative sampling for MTP chains and DFlash, else the runner's host sampler.
    model_capabilities = {
        # Exact prefix resume (prefix_resume.py): single-device, single-sequence traced prefill.
        "supports_prefix_caching": True,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        "supports_greedy_only_sample_on_device": True,
        # Exact on-device vLLM sampling for decode and (single sequence) sampled MTP / DFlash verification.
        "supports_exact_device_sampling": True,
    }

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Describe paged attention; GDN state is owned by the TT model."""
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        # Upstream Mamba page alignment is not applicable to our unpaged GDN
        # state. Keep KV blocks aligned with the fixed prefill chunk traces.
        cache_config.block_size = _BLOCK_SIZE
        dtype = model_config.dtype if cache_config.cache_dtype == "auto" else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        return {"foo": FullAttentionSpec(
            block_size=_BLOCK_SIZE,
            num_kv_heads=model_config.get_num_kv_heads(vllm_config.parallel_config),
            head_size=model_config.get_head_size(),
            dtype=dtype,
            sliding_window=model_config.get_sliding_window(),
        )}

    def _validate_device_sampling_request(self, requested):
        if not requested:
            return
        for model in self.model:
            if model.sampling is not None:
                continue
            mesh_shape = tuple(int(dim) for dim in model.mesh_device.shape)
            logits_per_device = math.ceil(model.args.vocab_size / model.num_devices)
            raise RuntimeError(
                "Qwen3.6 on-device sampling requires a certified TP topology (1x4 or 1x8) "
                f"with at most 65536 logits/device; got mesh={mesh_shape}, "
                f"vocab={model.args.vocab_size}, logits/device={logits_per_device}. "
                "Unset sample_on_device_mode for host sampling."
            )

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        """All-user KV capacity (the shared paged-KV token pool).

        QWEN36_MAX_TOKENS_ALL_USERS overrides it with a FIXED pool (set per device+model from the
        tt-inference-server spec's env_vars, mirroring GEMMA4_MAX_TOKENS_ALL_USERS). This decouples
        the pool from max_model_len × max_num_seqs so ONE config serves both a single long request
        (up to max_model_len) and a batch of shorter ones (sum of lengths ≤ pool) — e.g. 524288 =
        1×256K or 8×64K. Without the override, fall back to max_model_len × max_num_seqs (the old
        per-config product) so existing single-mode specs are unchanged."""
        override = os.environ.get("QWEN36_MAX_TOKENS_ALL_USERS")
        if override:
            return int(override)
        if max_model_len is not None:
            return int(max_model_len) * int(max_num_seqs or 1)
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_num_seqs=max_num_seqs,
            **kwargs,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        if os.environ.get("QWEN36_DFLASH") == "1":
            if os.environ.get("QWEN36_MTP", "0") != "0":
                raise ValueError("DFlash requires QWEN36_MTP=0; it is not an MTP mode")
            if mesh_device.get_num_devices() != 1 or tt_data_parallel != 1 or max_batch_size not in (1, *SHARED_SLOTS):
                raise ValueError("DFlash supports one device with one sequence, or shared verify rounds over "
                                 f"{' or '.join(map(str, SHARED_SLOTS))} sequences (16-user serving is plain)")
            if max_batch_size > 1 and int(os.environ.get("QWEN36_DFLASH_BRANCHES", "1")) != 1:
                raise ValueError("Shared verify rounds draft one DFlash path per user")
            block_size = int(os.environ.get("QWEN36_DFLASH_BLOCK_SIZE", "8"))
            if block_size not in (4, 8) or (max_batch_size > 1 and block_size != 8):
                raise ValueError("DFlash block size must be 4 or 8 (8 for shared verify rounds)")
            cls._dflash_branch_settings(block_size)
            if not os.environ.get("QWEN36_DFLASH_MODEL"):
                raise ValueError("DFlash requires a hash-verified QWEN36_DFLASH_MODEL checkpoint")
        # Weights dir: MODEL_WEIGHTS_DIR → HF_MODEL → hf_config._name_or_path; a hub id resolves to a local snapshot.
        name_or_path = os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL") or hf_config._name_or_path
        if name_or_path and not os.path.isdir(os.path.expanduser(name_or_path)):
            from huggingface_hub import snapshot_download

            # When offline/CI, resolve from the local cache only so snapshot_download
            # reads the cached refs instead of reaching the HF API (refused by HF_HUB_OFFLINE=1).
            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
            name_or_path = snapshot_download(name_or_path, local_files_only=offline)
        args, model, _ = create_tt_model(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, hf_model=name_or_path
        )
        # Native checkpoints carry text weights only (load_state_dict validates them as such); their
        # vision tower comes from QWEN36_VISION_WEIGHTS and is built only when QWEN36_VISION=1.
        native = os.path.isfile(os.path.join(args.CKPT_DIR, "native_manifest.json"))
        if os.environ.get("QWEN36_GENERIC_BASELINE") != "1" and (not native or os.environ.get("QWEN36_VISION") == "1"):
            model.init_vision_model()
        return cls([model], [args], mesh_device)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV (8 attn layers) + external GDN state; returns the 8 KV pairs.

        batch_size = max_batch_size (vLLM's max_num_seqs, threaded through initialize_vllm_model):
        the paged KV blocks (kv_cache_shape) already cover all users, and this sizes the per-slot
        GDN recurrent/conv state [B,...] + the decode kv grid. B==1 is the single-sequence path."""
        batch_size = self.model[0].args.max_batch_size
        caches = self.model[0].allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=batch_size)
        if self.exact_sampling_available():
            # Fixed-address sampler buffers precede every trace; one history slot per decode row.
            self.model[0].init_exact_sampler(batch_size)
        if self.mtp_available():
            self._init_mtp_buffers()
        if os.environ.get("QWEN36_DFLASH") == "1":
            if self.model[0].multiuser is not None:
                self._init_shared_drafter()
            else:
                self._init_dflash_buffers()
        return caches

    @staticmethod
    def _has_visual(kwargs, pixel_key):
        """True only when the request carries REAL visual data for this modality. vLLM attaches an
        empty pixel_values placeholder to text requests for a multimodal-registered model, so a
        plain ``is not None`` check misclassifies text as multimodal. Mirrors the emptiness test in
        _gather_user_visual (key absent / empty list / first item None => text-only)."""
        v = kwargs.get(pixel_key)
        return v is not None and len(v) > 0 and v[0] is not None

    @staticmethod
    def _gather_user_visual(kwargs, pixel_key, grid_key):
        """Pull this (B=1) user's patches + (t,h,w) grids for one modality out of the vLLM kwargs.

        Returns (pixel_values, grid_thw) or None when the request carries nothing for this modality.
        Multiple items for the user arrive as lists; concat the patches and stack the grids (the
        shapes get_visual_features expects: [num_patches, patch_dim] + [N, 3]).
        """
        if pixel_key not in kwargs or len(kwargs[pixel_key]) == 0 or kwargs[pixel_key][0] is None:
            return None
        pixel_values = kwargs[pixel_key][0]
        grid_thw = kwargs[grid_key][0]
        if isinstance(pixel_values, list) and len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values, dim=0)
            grid_thw = torch.stack([g.to(dtype=torch.int32) for g in grid_thw], dim=0)
        return pixel_values, grid_thw.reshape(-1, 3)

    def _compute_vision_tokens(self, model, kwargs, tokens, skip_before=0):
        """Run the vision tower for this (single-user, B=1) request, if it carries images or video.

        Mirrors the Qwen3-VL generator's multimodal check: pull this user's pixels + grids out of
        the vLLM kwargs and return the packed embeddings (host [num_vision_tokens, H], prompt
        order) for prefill to splice in. Returns None for a text-only request, so the whole
        multimodal path is skipped. Items whose placeholders all lie before ``skip_before`` (a
        prefix-resume boundary) are not encoded: their rows are never spliced.
        """
        images = self._gather_user_visual(kwargs, "pixel_values", "image_grid_thw")
        videos = self._gather_user_visual(kwargs, "pixel_values_videos", "video_grid_thw")
        if images is None and videos is None:
            return None
        return model.get_visual_features(tokens[0], images=images, videos=videos, skip_before=skip_before)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """All prefill is model-owned (Generator drives decode only)."""
        # Mirror Generator.prefill_forward_text: model-owned prefill bypasses
        # its bookkeeping, but replaces these slots' decode input ownership.
        self.mode = Mode.PREFILL
        if not hasattr(self, "_slots_prefilled_since_decode"):
            self._slots_prefilled_since_decode = set()
        empty_slots = kwargs.get("empty_slots")
        self._slots_prefilled_since_decode.update(
            range(tokens.shape[0]) if empty_slots is None else [int(s) for s in empty_slots]
        )
        model = self.model[0]
        # Prefill replays its own traces: the next plain decode restages every input (see decode_forward).
        self._foreign_trace_ran = True
        dflash = self.dflash_available()
        if dflash:
            # Request ownership changes before, never after, the new feature prefill.
            self.dflash_reset()
        if model.multiuser is not None:
            return self._prefill_forward_users(model, tokens, page_table, kv_cache, prompt_lens, empty_slots, kwargs)
        if model.args.max_batch_size > 1:
            # Batched text prefill into decode slots (MM is B=1). Require real visual data, not a
            # non-None empty pixel_values placeholder from vLLM on text requests.
            assert not self._has_visual(kwargs, "pixel_values") and not self._has_visual(
                kwargs, "pixel_values_videos"
            ), (
                "batched (max_num_seqs>1) serving is text-only; multimodal is single-sequence "
                "(max_concurrency=1). Run the model at max_num_seqs=1 for image/video requests."
            )
            return self._prefill_forward_tp_batched(model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"))
        return self._prefill_forward_single(model, tokens, page_table, prompt_lens, dflash, kwargs)

    # Per-request vLLM prefill inputs (one entry per scheduled request).
    _PER_REQUEST_KWARGS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "mm_keys",
                           "start_pos")
    # The plugin passes ``intermediate_prefill`` (rows whose chunk does not finish the prompt).
    accepts_intermediate_prefill = True

    def _prefill_forward_users(self, model, tokens, page_table, kv_cache, prompt_lens, empty_slots, kwargs):
        """Multi-user serving: prefill each scheduled request through the single-user path (live L1
        state as scratch, multimodal and prefix resume included), then load it into its decode slot.

        A long prompt arrives in 2,048-aligned chunks across scheduler steps, with decode steps of the
        other users in between. An intermediate chunk ends on a chunk boundary: its state there is kept
        on the host (the prefix-resume snapshot), and the next chunk resumes from it exactly as a
        prefix-cache hit does, so a chunked prompt runs the same chunks as a whole one."""
        count = int(tokens.shape[0])
        slots = list(range(count)) if empty_slots is None else [int(s) for s in empty_slots]
        table = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        intermediate = kwargs.get("intermediate_prefill") or [False] * count
        # Whole token row per request (the runner passes it with ``prefill_totals``): the vision tower and
        # M-RoPE need every placeholder of the prompt even when this step prefills only a chunk of it.
        totals = kwargs.get("prefill_totals")
        logits = []
        for row, slot in enumerate(slots):
            request = dict(kwargs)
            for key in self._PER_REQUEST_KWARGS:
                if request.get(key) is not None:
                    request[key] = request[key][row:row + 1]
            lens = prompt_lens[row:row + 1] if prompt_lens is not None else None
            row_tokens = tokens[row:row + 1, : int(totals[row])] if totals is not None else tokens[row:row + 1]
            row_logits, _ = self._prefill_forward_single(
                model, row_tokens, table[row:row + 1], lens, False, request,
                partial=bool(intermediate[row]), rope_tokens=row_tokens if totals is not None else None,
            )
            if not intermediate[row]:
                model.multiuser.load_slot(slot)
                if self.shared_available():
                    # The prefill ring (this prompt's drafter context) moves into the slot's ring.
                    keys = kwargs.get("request_ids")
                    self._shared.load_slot(slot, keys[row] if keys is not None else f"slot{slot}")
            logits.append(row_logits.reshape(1, 1, -1))
        return torch.cat(logits, dim=0), torch.zeros(count, dtype=torch.long)

    def _prefill_forward_single(self, model, tokens, page_table, prompt_lens, dflash, kwargs, partial=False,
                                rope_tokens=None):
        start_pos = kwargs.get("start_pos")
        start = int(start_pos[0]) if start_pos is not None else 0
        mm_keys = kwargs.get("mm_keys")
        mm = tuple(mm_keys[0]) if mm_keys and mm_keys[0] else ()
        vision_tokens = self._compute_vision_tokens(model, kwargs, tokens, skip_before=start)
        if model.num_devices > 1:
            if start:
                raise ValueError("prefix resume is single-device only")
            return self._prefill_forward_tp(model, tokens, page_table, prompt_lens, vision_tokens=vision_tokens)
        seq_len = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        traced = dflash or kwargs.get("enable_trace", False)
        if partial and seq_len % _PREFILL_WARMUP_CHUNK:
            raise ValueError(f"an intermediate prefill chunk must end on a {_PREFILL_WARMUP_CHUNK}-token boundary")
        resume, snapshot = self._prefix_resume_args(tokens, page_table, start, traced, mm,
                                                    self._take_partial(tokens, start, mm),
                                                    self._partial_store(tokens, mm) if partial else None)
        logger.info(f"Prefilling User 1 up to {seq_len} tokens" + (f" (resumed at {start})" if start else ""))
        # Multimodal works WITH the captured trace here: prefill_dispatch routes to the traced
        # path, which splices the image/video rows via a fixed-shape ttnn.where over persistent
        # buffers (compiled at warmup, updated per request by copy_host_to_device — no request-time
        # compile).
        if dflash:
            # Even without trace replay, use the fixed-bucket path: eager paged
            # prefill neither captures the five taps nor has bounded program shapes.
            logits = model.prefill_traced_chunked(
                tokens, page_table, actual_len=seq_len, vision_tokens=vision_tokens, resume=resume, snapshot=snapshot
            )
            self._dflash_position = seq_len
            self._dflash_context_ready = True
        elif traced:
            logits = model.prefill_traced_chunked(
                tokens, page_table, actual_len=seq_len, vision_tokens=vision_tokens, resume=resume, snapshot=snapshot,
                rope_tokens=rope_tokens,
            )
        else:
            logits = prefill_dispatch(
                model,
                tokens,
                page_table,
                prompt_lens,
                use_trace=kwargs.get("enable_trace", False),
                vision_tokens=vision_tokens,
            )
        logits = ttnn.to_torch(logits)
        # The vLLM runner unpacks (logits, rope_deltas) because the HF config has mrope_section.
        # Zero deltas are returned for all modalities: the multimodal M-RoPE delta is applied
        # entirely model-side (build_request_rope stashes self.rope.rope_delta during prefill, and
        # every decode path offsets the rope position by it), so the value handed back to vLLM is
        # unused for device-side rope and stays zero.
        rope_deltas = torch.zeros(logits.shape[0], dtype=torch.long)
        logger.info(f"Finished prefill up to {seq_len} tokens, starting decode...")
        return logits, rope_deltas

    @staticmethod
    def _prefix_resume_args(tokens, page_table, start, traced, mm, continuation=None, keep=None):
        """(resume payload, snapshot callback) for one single-sequence prefill.

        A nonzero start is the boundary the scheduler resumed this request from: the request's own
        previous chunk (``continuation``, its snapshot from _partial_store) or a prefix-cache hit (its
        blocks below it are the snapshot's). The snapshot callback publishes this prefill's own last
        full-chunk boundary to the prefix cache and, for an intermediate chunk, hands it to ``keep``.
        Only the traced chunk schedule is resumable. ``mm`` (the request's images / videos: placeholder
        offset, content identifier) is part of the key, as in the scheduler."""
        from vllm_tt_plugin.prefix_resume import REGISTRY

        if start and not traced:
            raise ValueError("prefix resume requires the traced chunk prefill")
        resume = None
        if continuation is not None:
            resume = dict(continuation, length=start)
        elif start:
            resume = dict(REGISTRY.take(tokens[0], start, mm), length=start)
        publish = traced and REGISTRY.active
        if not publish and keep is None:
            return resume, None
        row = page_table[0]

        def snapshot(length, payload, nbytes):
            if publish:
                REGISTRY.put(tokens[0], length, row, payload, nbytes, mm)
            if keep is not None:
                keep(length, payload)

        return resume, snapshot

    def _take_partial(self, tokens, start, mm):
        """The snapshot an earlier intermediate chunk of this request left at ``start``, if any."""
        if not start or not getattr(self, "_partial_prefills", None):
            return None
        from vllm_tt_plugin.prefix_resume import token_key

        return self._partial_prefills.pop(token_key(tokens[0], start, mm), None)

    def _partial_store(self, tokens, mm):
        """Snapshot sink of an intermediate chunk; at most one held per decode slot (an abandoned
        request's entry is dropped oldest-first)."""
        from collections import OrderedDict
        from vllm_tt_plugin.prefix_resume import token_key

        if getattr(self, "_partial_prefills", None) is None:
            self._partial_prefills = OrderedDict()
        held = self._partial_prefills
        limit = max(1, int(self.model[0].args.max_batch_size))

        def keep(length, payload):
            held[token_key(tokens[0], length, mm)] = payload
            while len(held) > limit:
                held.popitem(last=False)

        return keep

    def _prefill_forward_tp(self, model, tokens, page_table, prompt_lens, vision_tokens=None):
        """TP (B=1) paged prefill via the model-owned masked fixed-bucket path.

        prefill_traced_chunked rounds the prompt up to a fixed bucket and masks the GDN to the
        EXACT valid_len, so prefill runs one of a bounded, pre-warmed program set (the
        compile-clobbers-trace fix) — for <=2048 prompts it is entirely the masked bucket (no
        chunk trace needed). Longer prompts replay the chunk-outer trace (Milestone B). Returns
        host logits [1, 1, vocab] gathered to a single replica."""
        T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        if tokens.shape[1] > T:
            tokens = tokens[:, :T]
        logger.info(f"Prefilling User 1 up to {T} tokens (TP masked-bucket/chunked)")
        # Multimodal is supported on TP too: prefill_traced_chunked splices the image/video rows via
        # a fixed-shape ttnn.where over hidden-sharded persistent buffers (the vision rows are
        # gathered to full hidden on host, placed along seq, then re-sharded), so no request-time
        # compile clobbers the parked trace.
        logits = model.prefill_traced_chunked(
            tokens, page_table, actual_len=T, vision_tokens=vision_tokens
        )  # [1,1,vocab] replicated
        logits = (
            ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
            .reshape(-1, model.args.vocab_size)[:1]
            .float()
            .view(1, 1, -1)
        )
        logger.info(f"Finished prefill up to {T} tokens, starting decode...")
        return logits, torch.zeros(1, dtype=torch.long)

    def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots):
        """TP batched (max_num_seqs>1) prefill: prefill each request in this step into its decode slot.

        vLLM prefills new requests while other slots decode, so each user's B=1 state is written into
        row empty_slots[u] of the batched GDN buffers without disturbing the live rows (model-owned,
        via prefill_paged_slots). Attention fills each request's blocks via its page-table row.

        tokens:      torch [N, max_T] (rows are the N requests scheduled this prefill step).
        page_table:  torch [N, max_blocks] — row u = request u's blocks.
        prompt_lens: per-request real lengths (row u trimmed to prompt_lens[u]).
        empty_slots: per-request decode slot; defaults to range(N) (mirrors Generator.prefill_forward_text).
        Returns ([N, 1, vocab] host logits, [N] zero rope_deltas — text M-RoPE delta is 0, applied model-side).
        """
        N = tokens.shape[0]
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [tokens.shape[1]] * N
        if empty_slots is None:
            empty_slots = list(range(N))
        empty_slots = [int(s) for s in empty_slots]
        token_ids_list = [tokens[u : u + 1, : plens[u]].to(torch.int32) for u in range(N)]
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        logger.info(f"Prefilling {N} user(s) into slots {empty_slots} (TP batched masked-bucket)")
        host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)
        logits = torch.cat([hl.reshape(1, 1, -1) for hl in host_logits], dim=0)  # [N, 1, vocab]
        logger.info(f"Finished batched prefill of {N} user(s), starting decode...")
        return logits, torch.zeros(N, dtype=torch.long)

    # ---- Greedy, single-sequence native DFlash serving ----
    @staticmethod
    def _dflash_branch_settings(block_size):
        branches = int(os.environ.get("QWEN36_DFLASH_BRANCHES", "1"))
        fork = int(os.environ.get("QWEN36_DFLASH_FORK_POSITION", "3"))
        if branches not in (1, 2) or not 1 <= fork <= 7:
            raise ValueError("DFlash supports one/two branches and fork positions 1..7")
        if branches == 2 and block_size != 8:
            raise ValueError("Two-branch DFlash requires block size 8")
        return branches, fork

    def dflash_available(self):
        return getattr(self.model[0], "_dflash_cycle", None) is not None

    def _init_dflash_buffers(self):
        """Allocate all native owners before the first prefill/decode trace."""
        from models.demos.blackhole.qwen36.tt.dflash import NativeDFlash, load_dflash_checkpoint
        from models.demos.blackhole.qwen36.tt.speculative_dflash import DFlashCycle
        from models.demos.blackhole.qwen36.tt.speculative_features import PrefillFeatures

        model = self.model[0]
        if self.dflash_available():
            return model._dflash_cycle.width
        if model.num_devices != 1 or model.args.max_batch_size != 1:
            raise ValueError("DFlash requires a single-device, single-sequence target")
        if self.mtp_available() or os.environ.get("QWEN36_MTP", "0") != "0":
            raise ValueError("DFlash cannot share a target with MTP enabled")
        if (model.args.dim, model.args.vocab_size, model.args.n_layers) != (5120, 248320, 64):
            raise ValueError("The pinned DFlash checkpoint requires the Qwen3.8-27B target geometry")
        if tuple(model.layer_indices) != tuple(range(64)):
            raise ValueError("DFlash requires all target layers in their original order")
        if not 2048 <= model.args.max_seq_len <= 262144:
            raise ValueError("DFlash requires a target context between 2048 and 262144 tokens")
        if not model._paged_kv_caches:
            raise RuntimeError("Allocate paged KV before DFlash initialization")
        if getattr(model, "_chunked_prepared_key", None) is not None:
            raise RuntimeError("DFlash must be initialized before any prefill warmup")
        block_size = int(os.environ.get("QWEN36_DFLASH_BLOCK_SIZE", "8"))
        if block_size not in (4, 8):
            raise ValueError("DFlash block size must be 4 or 8")
        branches, fork = self._dflash_branch_settings(block_size)
        directory = os.environ.get("QWEN36_DFLASH_MODEL")
        if not directory:
            raise ValueError("DFlash requires QWEN36_DFLASH_MODEL")
        blocks = int(model._paged_kv_caches[0][0].shape[0])
        width = (blocks // 8) * 8
        if width < math.ceil(model.args.max_seq_len / int(model._paged_kv_caches[0][0].shape[2])):
            raise ValueError("DFlash aligned page-table capacity must cover the configured target context")
        state, config = load_dflash_checkpoint(os.path.expanduser(directory))

        def embed_tokens(token_ids):
            if tuple(token_ids.shape) != (1, block_size):
                raise ValueError("DFlash embedding requires native [1, block_size] token IDs")
            hidden = model.embd(token_ids)
            return ttnn.reshape(hidden, [1, 1, block_size, model.args.dim])

        draft_head = model.dflash_draft_head

        def output_head(hidden):
            rows = block_size - 1
            if tuple(hidden.shape) != (1, 1, rows, model.args.dim):
                raise ValueError("DFlash head requires native [1, 1, depth, hidden] input")
            # Do not apply the target's final RMSNorm: the drafter already applied its own norm.
            if draft_head is not None:
                weight, program_config, _ = draft_head
                return ttnn.linear(hidden, weight, program_config=program_config,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # The target head consumes its native 3D shape.
            logits = model._lm_head(ttnn.reshape(hidden, [1, rows, model.args.dim]))
            return ttnn.reshape(logits, [1, 1, rows, model.args.vocab_size])

        model.dflash = NativeDFlash(
            model.device, state, config, embed_tokens, output_head,
            model.args.weight_cache_path() / "dflash", block_size=block_size,
            draft_vocab=None if draft_head is None else draft_head[2],
        )
        del state
        try:
            buckets = sorted(set((*model._PREFILL_MASK_BUCKETS, _PREFILL_WARMUP_CHUNK)))
            model.prefill_features = PrefillFeatures(
                model, config["dflash_config"]["target_layer_ids"], buckets, model.dflash
            )
            if branches == 2:
                from models.demos.blackhole.qwen36.tt.speculative_dflash_tree import DFlashTreeCycle
                model._dflash_cycle = DFlashTreeCycle(model, model.dflash, width, fork)
            else:
                model._dflash_cycle = DFlashCycle(model, model.dflash, width)
                if self.exact_sampling_available():
                    model._dflash_cycle.enable_sampling(model.init_exact_sampler(1), self.dflash_sample_mode())
        except Exception:
            features = getattr(model, "prefill_features", None)
            if features is not None:
                features.release()
                model.prefill_features = None
            model.dflash.release()
            model.dflash = None
            raise
        self._dflash_context_ready = False
        self._dflash_position = None
        self._dflash_prefill_compiled = False
        self._dflash_decode_compiled = False
        self.already_warmed_up_prefill = False
        return width

    def dflash_reset(self):
        """Invalidate request ownership, retaining every fixed-address buffer."""
        self._dflash_context_ready = False
        self._dflash_position = None
        cycle = getattr(self.model[0], "_dflash_cycle", None)
        if cycle is not None:
            cycle.reset()

    def _dflash_request_cycle(self, position):
        cycle = getattr(self.model[0], "_dflash_cycle", None)
        if cycle is None or not cycle.compiled or not cycle.first_compiled:
            raise RuntimeError("DFlash must be initialized and warmed before serving")
        if not getattr(self, "_dflash_context_ready", False) or position != self._dflash_position:
            raise RuntimeError("DFlash requires fresh, contiguous target prefill features")
        if position < 0 or position + 2 * cycle.depth + 1 >= self.model[0].args.max_seq_len:
            raise ValueError("DFlash verification and next noisy block exceed the target context")
        return cycle

    def dflash_first_drafts(self, first_token, first_position, request=None):
        """Propose from the fresh prefill context without consuming the anchor.

        With `request` (a device_sampling.SamplingRequest) the proposals are those of the sampled cycle's draft
        mode (QWEN36_DFLASH_SAMPLE_DRAFT), whose distributions stay on device for the next verification."""
        token, position = int(first_token), int(first_position)
        cycle = self._dflash_request_cycle(position)
        if not 0 <= token < self.model[0].args.vocab_size:
            raise ValueError("DFlash seed token is outside the vocabulary")
        if request is not None:
            if not self.sampled_speculation_available():
                raise ValueError("Sampled DFlash requires the linear cycle and the exact device sampler")
            self.model[0].prepare_sampled_request(request, cycle.depth)
            return cycle.first_drafts_sampled(token, position)
        return cycle.first_drafts(token, position)

    def dflash_step(self, committed_token, drafts, position, page_table_row, request=None):
        """Execute one complete scheduler-owned verify/commit/draft transaction (sampled with `request`)."""
        model = self.model[0]
        token, position = int(committed_token), int(position)
        cycle = self._dflash_request_cycle(position)
        depth = cycle.depth
        proposals = [int(draft) for draft in drafts]
        if len(proposals) != depth:
            raise ValueError("DFlash requires the complete configured proposal block")
        if any(not 0 <= value < model.args.vocab_size for value in (token, *proposals)):
            raise ValueError("DFlash tokens are outside the vocabulary")
        pt = torch.as_tensor(page_table_row, dtype=torch.int64, device="cpu").reshape(-1)
        if pt.numel() > cycle.width:
            raise ValueError("DFlash request page table exceeds captured capacity")
        cache = model._paged_kv_caches[0][0]
        needed = (position + depth) // int(cache.shape[2]) + 1
        if pt.numel() < needed:
            raise ValueError("DFlash target block exceeds the request's allocated KV pages")
        active = pt[:needed]
        if torch.any(active < 0) or torch.any(active >= int(cache.shape[0])):
            raise ValueError("DFlash page table references an unallocated physical KV block")
        if torch.unique(active).numel() != needed:
            raise ValueError("DFlash active logical KV pages must have distinct physical blocks")
        # The runner supplies its actual owned block list, never a padded table.
        # Refresh it every transaction; retained device pages confer no ownership.
        if request is not None:
            if not self.sampled_speculation_available():
                raise ValueError("Sampled DFlash requires the linear cycle and the exact device sampler")
            model.prepare_sampled_request(request, depth)
        cycle.seed(token, position, active.to(torch.int32), proposals=proposals)
        try:
            if request is None:
                compact = ttnn.to_torch(cycle.run()).reshape(-1).tolist()
                emitted, next_drafts = cycle.decode_result(compact, position)
            else:
                emitted, next_drafts, redraw = cycle.decode_sampled(cycle.run_sampled())

                def redraft(context, anchor_position):
                    model.prepare_sampled_request(request, depth, outputs=list(request.outputs) + context)
                    return cycle.first_drafts_sampled(context[-1], anchor_position)

                emitted, redrafted = model.finish_sampled_cycle(request, position, emitted, redraw,
                                                                cycle.verify_logits, redraft)
                if redrafted is not None:
                    next_drafts = redrafted
        except Exception:
            self.dflash_reset()
            raise
        self._dflash_position = position + len(emitted)
        return emitted, next_drafts

    def dflash_branch_stats(self):
        return getattr(self.model[0]._dflash_cycle, "last_stats", None)

    def dflash_warmup(self, capture=False):
        """Compile first; capture only after target prefill/decode compilation."""
        model = self.model[0]
        cycle = model._dflash_cycle
        sampled = self.sampled_speculation_available()
        if (capture and cycle.trace is not None and cycle.first_trace is not None
                and (not sampled or cycle.sampled_trace is not None)):
            return
        saved = model._save_deltanet_states()
        try:
            table = torch.arange(cycle.width, dtype=torch.int32)
            if capture and weight_prefetch.arm(model.device):
                # Build the prefetcher-fed verify programs (and their GCBs, which take L1 that prefill
                # circular buffers use) after prefill capture and before any decode-phase capture.
                self.dflash_reset()
                cycle.seed_for_warmup(table)
                cycle.step()
                ttnn.synchronize_device(model.device)
            self.dflash_reset()
            cycle.seed_for_warmup(table)
            cycle.warmup_first_drafts(capture=capture)
            self.dflash_reset()
            cycle.seed_for_warmup(table)
            cycle.warmup(capture=capture)
            if sampled:
                # Inactive parameter rows: the sampler kernels run as no-ops on the synthetic warmup inputs.
                model.exact_sampler.set_params([None] * 32)
                self.dflash_reset()
                cycle.seed_for_warmup(table)
                cycle.warmup_sampled(capture=capture)
        finally:
            model._restore_deltanet_states(saved, model.device)
            self.dflash_reset()
            if sampled:
                model.exact_sampler.invalidate()

    # ---- Multi-user DFlash2: shared verify rounds (4 or 8 slots) ----
    # A multi-user server with DFlash2 serves every decode step either as the plain multi-user step (s27v6) or as one
    # shared verify round (shared_round.py) over every running request: its pending token plus the drafts the
    # scheduler carried from the previous round, followed by the batched drafter passes the round planner chose
    # (shared_serving.py), which give each request its drafts for the next round. The plugin offers every decode step
    # whose rows are on the device sampler (greedy, or exact device sampling: a sampled round, s27v8) to
    # shared_step(); None means no row carries drafts and the planner drafts nothing after the step, so the plain step
    # serves it (that skips the drafter context commit: a later round's drafter masks the missing rows).
    def shared_available(self):
        return getattr(self, "_shared", None) is not None

    def _dram_free(self, tag):
        view = ttnn.get_memory_view(self.model[0].device, ttnn.BufferType.DRAM)
        free = int(view.total_bytes_free_per_bank) * int(view.num_banks) / 1e9
        logger.info("Shared rounds DRAM after {}: {:.2f} GB free", tag, free)
        return free

    def _init_shared_drafter(self):
        """Shared rounds, part 1 (allocate_kv_cache, before any prefill program): the drafter (BFP8 backbone, draft
        head) and the prefill feature taps, which fill the single-sequence prefill ring a slot's ring is loaded from.
        The slot rings, round inputs and candidates (_init_shared_rounds) follow the prefill compile, whose transient
        DRAM would not fit beside them."""
        from models.demos.blackhole.qwen36.tt.dflash import NativeDFlash, load_dflash_checkpoint
        from models.demos.blackhole.qwen36.tt.speculative_features import PrefillFeatures

        model = self.model[0]
        if getattr(model, "dflash", None) is not None:
            return
        if model.multiuser is None or model.multiuser.slots not in SHARED_SLOTS:
            raise ValueError(f"shared verify rounds serve {' or '.join(map(str, SHARED_SLOTS))} decode slots")
        if self.mtp_available() or os.environ.get("QWEN36_MTP", "0") != "0":
            raise ValueError("DFlash cannot share a target with MTP enabled")
        if (model.args.dim, model.args.vocab_size, model.args.n_layers) != (5120, 248320, 64):
            raise ValueError("The pinned DFlash checkpoint requires the Qwen3.8-27B target geometry")
        if tuple(model.layer_indices) != tuple(range(64)):
            raise ValueError("DFlash requires all target layers in their original order")
        if not 2048 <= model.args.max_seq_len <= 262144:
            raise ValueError("DFlash requires a target context between 2048 and 262144 tokens")
        if getattr(model, "_chunked_prepared_key", None) is not None:
            raise RuntimeError("DFlash must be initialized before any prefill warmup")
        if int(os.environ.get("QWEN36_DFLASH_BLOCK_SIZE", "8")) != 8:
            raise ValueError("shared verify rounds draft blocks of 8")
        directory = os.environ.get("QWEN36_DFLASH_MODEL")
        if not directory:
            raise ValueError("DFlash requires QWEN36_DFLASH_MODEL")
        state, config = load_dflash_checkpoint(os.path.expanduser(directory))
        head = model.dflash_draft_head

        def embed_tokens(token_ids):
            return ttnn.reshape(model.embd(token_ids), [1, 1, int(token_ids.shape[-1]), model.args.dim])

        def output_head(hidden):
            # The drafter applied its own final norm; never the target's.
            if head is not None:
                weight, program_config, _ = head
                return ttnn.linear(hidden, weight, program_config=program_config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            rows = int(hidden.shape[-2])
            logits = model._lm_head(ttnn.reshape(hidden, [1, rows, model.args.dim]))
            return ttnn.reshape(logits, [1, 1, rows, model.args.vocab_size])

        model.dflash = NativeDFlash(model.device, state, config, embed_tokens, output_head,
                                    model.args.weight_cache_path() / "dflash", block_size=8,
                                    draft_vocab=None if head is None else head[2])
        del state
        try:
            buckets = sorted(set((*model._PREFILL_MASK_BUCKETS, _PREFILL_WARMUP_CHUNK)))
            model.prefill_features = PrefillFeatures(model, config["dflash_config"]["target_layer_ids"], buckets,
                                                     model.dflash)
        except Exception:
            model.dflash.release()
            model.dflash = None
            raise
        self._shared = None
        self._dram_free("the drafter")

    def _init_shared_rounds(self):
        """Shared rounds, part 2 (after the prefill compile, before any decode program): one drafter ring per slot,
        the fixed-address round inputs of the 4 / 8 / 16-row rounds and the candidate rows of multi-row groups
        (QWEN36_SHARED_CAPACITY, default 16: two 8-row groups). With the exact device sampler every width also gets a
        sampled round and every drafter pass a sampled variant (QWEN36_SHARED_SAMPLE_DRAFT, see shared_sample_mode)."""
        from models.demos.blackhole.qwen36.tt.dflash_slots import SlotDrafter
        from models.demos.blackhole.qwen36.tt.shared_round import SharedRounds
        from models.demos.blackhole.qwen36.tt.shared_serving import SharedServing

        if self.shared_available():
            return
        model = self.model[0]
        decoder = model.multiuser
        capacity = int(os.environ.get("QWEN36_SHARED_CAPACITY", "16"))
        sampler = model.exact_sampler if self.exact_sampling_available() else None
        drafter = SlotDrafter(model, model.dflash, decoder.slots, decoder.slots, sampler=sampler,
                              sample_mode=self.shared_sample_mode())
        rounds = SharedRounds(model, decoder, widths=(4, 8, 16), capacity=capacity, drafter=drafter, sampler=sampler)
        self._shared = SharedServing(model, decoder, rounds, drafter)
        logger.info("Shared verify rounds: slots={} rings={} widths={} candidate rows={} sampled rounds={} "
                    "sampled drafts={}", decoder.slots, drafter.rings, rounds.widths, rounds.capacity,
                    sampler is not None, drafter.sample_mode if sampler is not None else "off")
        self._dram_free("the rings and round buffers")

    @staticmethod
    def shared_sample_mode():
        """Drafts of random / penalized requests in shared rounds (QWEN36_SHARED_SAMPLE_DRAFT): point (the greedy
        walk) | sampled (default, DFlash2's probabilistic walk: softmax(scores / T) over the 16 candidates) | processed
        (after the request's top-k / top-p / min-p); sampled walks draw with the uniform the verifying row draws with.
        The three accepted alike in runs/sampled2 (8 slots, 1-8 users)."""
        return os.environ.get("QWEN36_SHARED_SAMPLE_DRAFT", "sampled")

    def shared_sampling_available(self):
        """Random / penalized requests ride shared rounds (sampled rounds); else their steps run plain."""
        return self.shared_available() and self._shared.rounds.sampler is not None

    def _shared_warmup(self, capture):
        """Compile (capture=False) or capture every round width (greedy and sampled) and drafter pass on synthetic
        layouts: rows on the scratch KV block, a multi-row group of slot 0 then one row per other slot; the sampler's
        parameter rows inactive. The rings are reset afterwards."""
        from models.demos.blackhole.qwen36.tt.dflash_slots import LANES
        from models.demos.blackhole.qwen36.tt.shared_round import row_word

        serving = self._shared
        rounds, drafter, slots = serving.rounds, serving.drafter, serving.decoder.slots
        sampler = rounds.sampler
        variants = (False, True) if sampler is not None else (False,)
        if sampler is not None:
            sampler.set_params([None] * 32)
        for width in rounds.widths:
            group = min(width, rounds.capacity, 8)
            words = [row_word(j, group, 0) if j < group else
                     (row_word(0, 1, 1 + j - group) if 1 + j - group < slots else 0) for j in range(width)]
            positions = [j if j < group else 0 for j in range(width)]
            rounds.stage(width, [0] * width, positions, positions, serving.scratch_row.repeat(width, 1), words)
            for sampled in variants:
                rounds.warmup(width, capture=capture, sampled=sampled)
        for quad in range(drafter.rings // 4):
            for lanes in LANES:
                for sampled in variants if drafter.samples else (False,):
                    drafter.warmup(quad, capture=capture, lanes=lanes, sampled=sampled)
        ttnn.synchronize_device(self.model[0].device)
        for slot in range(drafter.rings):
            drafter.reset_slot(slot)
        if sampler is not None:
            sampler.invalidate()
        if capture:
            self._dram_free("the round and drafter captures")

    def shared_step(self, rows, slot_remap=None, sampling=None):
        """One decode step of a shared-rounds server. rows: per decode row r, which is decode slot r once
        slot_remap is applied: (request id, pending token, drafts, position, page-table row, next_cap), next_cap
        bounding the request's drafts for the next round. sampling: None (every row greedy, unpenalized) or per row
        the request's device_sampling.SamplingRequest (a sampled round, shared_sampling_available()). Returns None
        when the plain multi-user step must serve this step (no row carries drafts and the planner drafts nothing
        after it; nothing ran, the remap included), else (emitted tokens, next drafts) per row."""
        serving = self._shared
        full = [(key, slot, int(token), [int(t) for t in drafts], int(position), table, int(cap))
                for slot, (key, token, drafts, position, table, cap) in enumerate(rows)]
        plan = serving.plan(full)
        if not any(row[3] for row in full) and not plan[1]:
            return None
        if sampling is not None and not self.shared_sampling_available():
            raise ValueError("sampled shared rounds need the exact device sampler")
        self._foreign_trace_ran = True
        return serving.round(full, slot_remap, plan, sampling=sampling)

    def shared_release(self, key):
        """A request finished, was aborted or preempted: forget its acceptance statistics and slot ownership."""
        if self.shared_available():
            self._shared.release(key)

    # ---- Exact on-device sampling ----
    def exact_sampling_available(self):
        model = self.model[0]
        return (
            model.num_devices == 1
            and os.environ.get("QWEN36_EXACT_SAMPLING", "1") == "1"
            and os.environ.get("QWEN36_GENERIC_BASELINE") != "1"
        )

    @staticmethod
    def dflash_sample_mode():
        """DFlash proposal for sampled requests (QWEN36_DFLASH_SAMPLE_DRAFT): point | sampled | processed."""
        return os.environ.get("QWEN36_DFLASH_SAMPLE_DRAFT", "sampled")

    def sampled_speculation_available(self):
        """Sampled speculative cycles: any MTP depth, or the linear (single-branch) DFlash cycle."""
        if not self.exact_sampling_available():
            return False
        if self.mtp_available():
            return True
        if os.environ.get("QWEN36_DFLASH") == "1":
            return int(os.environ.get("QWEN36_DFLASH_BRANCHES", "1")) == 1
        return False

    def decode_bucket_widths(self, max_batch_size):
        """Decode trace widths: powers of two up to max_batch_size, or the multi-user step widths."""
        multiuser = self.model[0].multiuser
        if multiuser is not None:
            return list(multiuser.widths)
        if os.environ.get("TT_DECODE_BUCKETING", "1") != "1" or max_batch_size <= 1:
            return [max_batch_size]
        widths, w = [], 1
        while w < max_batch_size:
            widths.append(w)
            w *= 2
        return widths + [max_batch_size]

    def _sampler_decode_widths(self):
        return self.decode_bucket_widths(int(self.model[0].args.max_batch_size))

    def _warmup_exact_sampler(self, capture):
        """Phase 1 compiles every decode sampler width on scratch logits; phase 2 captures one sampler trace per
        decode-bucket trace, bound to that trace's logits output and position input."""
        model = self.model[0]
        sampler = model.exact_sampler
        vocab = model.args.vocab_size
        if not capture:
            for width in self._sampler_decode_widths():
                for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
                    logits = ttnn.from_torch(
                        torch.zeros(1, 1, width, vocab, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                        layout=layout, device=model.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    pos = ttnn.from_torch(torch.zeros(width, dtype=torch.int32), dtype=ttnn.int32,
                                          layout=ttnn.ROW_MAJOR_LAYOUT, device=model.device)
                    sampler.set_params([None] * 32)
                    sampler.sample_decode(logits, width, pos)
                    ttnn.synchronize_device(model.device)
                    ttnn.deallocate(logits)
                    ttnn.deallocate(pos)
            sampler.invalidate()
            return
        traces = getattr(self, "_sampler_decode_traces", None)
        if traces is None:
            traces = self._sampler_decode_traces = {}
        for width, (_, inputs, outputs) in sorted(getattr(self, "_bucket_trace_store", {}).items()):
            if width in traces or outputs[False] is None or inputs[False] is None:
                continue
            logits = outputs[False][0][0]
            pos = inputs[False][0][1]
            if logits.memory_config().buffer_type != ttnn.BufferType.DRAM:
                raise RuntimeError("Exact sampling expects DRAM decode logits")
            view = ttnn.reshape(logits, (1, 1, width, vocab)) if logits.layout == ttnn.TILE_LAYOUT else logits
            sampler.set_params([None] * 32)
            sampler.sample_decode(view, width, pos)
            ttnn.synchronize_device(model.device)
            trace = ttnn.begin_trace_capture(model.device, cq_id=0)
            sampler.sample_decode(view, width, pos)
            ttnn.end_trace_capture(model.device, trace, cq_id=0)
            traces[width] = (trace, view)
            logger.info(f"Captured exact-sampling decode trace for bucket width {width}")
        sampler.invalidate()

    def _decode_exact(self, tt_out, requests):
        """Draw this step's tokens on device from the decode trace's logits; host-redraw any flagged row."""
        from models.demos.blackhole.qwen36.tt.device_sampling import host_sample

        model = self.model[0]
        sampler = model.exact_sampler
        logits = tt_out[0][0]
        dims = [int(d) for d in logits.shape]
        width = math.prod(dims[:-1])
        trace, view = self._sampler_decode_traces[width]
        rows = len(requests)
        words = [None] * 32
        for row, request in enumerate(requests):
            if request is not None:
                sampler.ensure_history(row, request.owner, request.outputs, request.prompt or (),
                                       request.spec.uses_prompt)
                words[row] = request.spec.words(row)
        sampler.set_params(words)
        ttnn.execute_trace(model.device, trace, cq_id=0, blocking=False)
        out = sampler.read_out(width)
        tokens = out[:, 0].to(torch.int32).clone()
        host_rows = None
        for row, request in enumerate(requests):
            if request is None:
                continue
            if int(out[row, 1]) & 1:
                if host_rows is None:
                    host_rows = ttnn.to_torch(view).reshape(-1, model.args.vocab_size)
                position = int(self._sampler_last_positions[row]) + 1
                tokens[row] = host_sample(host_rows[row], request.spec, request.outputs, request.prompt or (), position)
                sampler.invalidate(row)
                logger.info(f"Exact sampling: host redraw for row {row} at position {position} (truncated tie group)")
            else:
                sampler.note_committed(row, request.owner, len(request.outputs) + 1)
        return tokens[: max(rows, 1)]

    # ---- Greedy, single-sequence native MTP serving ----
    def mtp_available(self):
        return getattr(self.model[0], "mtp", None) is not None

    def _init_mtp_buffers(self):
        model = self.model[0]
        if not model._paged_kv_caches:
            raise RuntimeError("Allocate paged KV caches before MTP warmup")
        blocks = int(model._paged_kv_caches[0][0].shape[0])
        # Native paged_update_cache requires an aligned row no wider than the pool.
        width = (blocks // 8) * 8
        model.init_mtp_cycle(width)
        if model.mtp_depth > 1 and getattr(model, "_mtp_wide", None) is None:
            from models.demos.blackhole.qwen36.tt.mtp_wide import WideMTPCycle

            model._mtp_wide = WideMTPCycle(model, model.mtp_depth, width)
        if model.mtp_depth > 1 and self.exact_sampling_available():
            model._mtp_wide.enable_sampling(model.init_exact_sampler(1))
        return width

    def _mtp_cache_positions(self):
        model = self.model[0]
        return None if model.mtp_depth == 1 else model._mtp_wide.cache_positions

    def mtp_first_drafts(self, first_token, first_position, request=None):
        """Read a complete sequential chain from the current target hidden state.

        With `request` (a device_sampling.SamplingRequest) each draft is drawn from the processed draft
        distribution q_i, which stays on device for the next sampled verification."""
        model = self.model[0]
        token, position = int(first_token), int(first_position)
        depth = model.mtp_depth
        if not 0 <= token < model.args.vocab_size:
            raise ValueError("MTP seed token is outside the vocabulary")
        if not 0 <= position or position + depth - 1 >= model.args.max_seq_len:
            raise ValueError("Initial MTP chain exceeds the context")
        if request is not None:
            if not self.sampled_speculation_available():
                raise ValueError("Sampled MTP requires the exact device sampler")
            model.prepare_sampled_request(request, depth, depth)
            return model.mtp.first_chain_sampled(token, position, depth, self._mtp_cache_positions())
        if depth == 1:
            return [model.mtp.first_draft(token, position)]
        return model._mtp_wide.first_drafts(token, position)

    def mtp_step(self, committed_token, drafts, position, page_table_row, request=None):
        """Reseed from the scheduler and execute exactly one complete transaction.

        The caller must use ordinary decode for a partial scheduled chain or context tail. Random or
        penalized requests pass `request` and run the sampled (speculative rejection sampling) cycle.
        This API never advances target state to warm up.
        """
        model = self.model[0]
        depth = model.mtp_depth
        token, position = int(committed_token), int(position)
        proposals = [int(draft) for draft in drafts]
        if len(proposals) != depth:
            raise ValueError("MTP requires the complete configured proposal chain")
        if any(not 0 <= value < model.args.vocab_size for value in (token, *proposals)):
            raise ValueError("MTP tokens are outside the vocabulary")
        # Full acceptance drafts again from p+D+1 through p+2D, inclusive.
        if not 0 <= position or position + 2 * depth >= model.args.max_seq_len:
            raise ValueError("MTP verify and next draft chain exceed the context")
        pt = torch.as_tensor(page_table_row, dtype=torch.int64, device="cpu").reshape(-1)
        width = model._mtp_cycle_width
        if pt.numel() > width:
            raise ValueError(f"MTP page table has {pt.numel()} blocks; captured capacity is {width}")
        cache = model._paged_kv_caches[0][0]
        needed = (position + depth) // int(cache.shape[2]) + 1
        if pt.numel() < needed:
            raise ValueError("MTP target chain exceeds the request's allocated KV pages")
        active = pt[:needed]
        if torch.any(active < 0) or torch.any(active >= int(cache.shape[0])):
            raise ValueError("MTP page table references an unallocated physical KV block")
        if torch.unique(active).numel() != needed:
            raise ValueError("MTP active logical KV pages must have distinct physical blocks")
        # Ignore scheduler padding beyond the visible chain; every accessed page was
        # checked above. Always refresh ownership, even when the row width is unchanged.
        pt = active.to(torch.int32)
        if request is not None:
            if not self.sampled_speculation_available():
                raise ValueError("Sampled MTP requires the exact device sampler")
            if depth == 1:
                emitted, next_draft = model.mtp_step_sampled(token, proposals[0], position, pt, request)
                return emitted, [next_draft]
            cycle = model._mtp_wide
            model.prepare_sampled_request(request, depth, depth)
            cycle.seed(token, position, pt, proposals=proposals)
            words = cycle.run_sampled()
            accepted, vflags = int(words[16]), int(words[17])
            word = model.exact_sampler.CHAIN_WORD

            def redraft(context, anchor_position):
                model.prepare_sampled_request(request, depth, depth, outputs=list(request.outputs) + context)
                return model.mtp.first_chain_sampled(context[-1], anchor_position, depth, cycle.cache_positions)

            emitted, redrafted = model.finish_sampled_cycle(
                request, position, words[:accepted + 1], vflags - 1 if vflags else None, cycle.verify_logits, redraft)
            next_drafts = redrafted if redrafted is not None else [int(v) for v in words[word:word + depth]]
            return emitted, next_drafts
        if depth == 1:
            emitted, next_draft = model.mtp_step(token, proposals[0], position, pt)
            return emitted, [next_draft]
        cycle = model._mtp_wide
        cycle.seed(token, position, pt, proposals=proposals)
        compact = ttnn.to_torch(cycle.run()).reshape(-1).tolist()
        accepted = sum(int(flag) for flag in compact[depth + 1:2 * depth + 1])
        emitted = [int(value) for value in compact[:accepted + 1]]
        next_drafts = [int(value) for value in compact[2 * depth + 1:]]
        return emitted, next_drafts

    def mtp_warmup(self, page_table_width, capture=True):
        model = self.model[0]
        sampled = self.sampled_speculation_available()
        if model.mtp_depth == 1:
            model.mtp_cycle_warmup(page_table_width, capture=capture)
            model.mtp.warmup_first_draft(capture=capture)
            if sampled:
                model.mtp_cycle_warmup_sampled(page_table_width, capture=capture)
                model.exact_sampler.set_params([None] * 32)
                model.mtp.warmup_first_chain_sampled(1, None, capture=capture)
                model.exact_sampler.invalidate()
            return
        cycle = model._mtp_wide
        if cycle.trace is not None and cycle.first_trace is not None and (
                not sampled or cycle.sampled_trace is not None):
            return
        saved = model._save_deltanet_states()
        try:
            # Only synthetic warmup inputs are consumed here, never request tokens.
            table = torch.arange(page_table_width, dtype=torch.int32)
            cycle.seed(11, 0, table, proposals=[12] * model.mtp_depth)
            cycle.warmup_first_drafts(capture=capture)
            cycle.warmup(capture=capture)
            if sampled:
                # Inactive parameter rows: the sampler kernels run as no-ops on the synthetic warmup inputs.
                model.exact_sampler.set_params([None] * 32)
                cycle.seed(11, 0, table, proposals=[12] * model.mtp_depth)
                cycle.warmup_sampled(capture=capture)
                model.mtp.warmup_first_chain_sampled(model.mtp_depth, cycle.cache_positions, capture=capture)
        finally:
            model._restore_deltanet_states(saved, model.device)
            if sampled:
                model.exact_sampler.invalidate()

    def warmup_model_decode(self, *args, **kwargs):
        width = self._init_mtp_buffers() if self.mtp_available() else None
        dflash = self.dflash_available()
        capture = kwargs.get("enable_trace", False)
        if dflash and capture and not (
            self._dflash_prefill_compiled and self._dflash_decode_compiled
        ):
            raise RuntimeError("Compile all DFlash prefill and decode programs before capturing traces")
        model = self.model[0]
        saved = model._save_deltanet_states() if dflash else None
        try:
            result = self._warmup_model_decode_base(*args, **kwargs)
            if width is not None:
                logger.info("Warming Qwen MTP-{} device cycle (capture={}, page_table_width={})...",
                            model.mtp_depth, capture, width)
                self.mtp_warmup(width, capture=capture)
            if self.exact_sampling_available():
                logger.info("Warming exact on-device sampling (capture={})...", capture)
                self._warmup_exact_sampler(capture)
            if dflash:
                self.dflash_warmup(capture=capture)
                self._dflash_decode_compiled = True
            if self.shared_available():
                # After the plain multi-user traces: rounds, then drafter passes (every input predates every capture).
                logger.info("Warming shared verify rounds and drafter passes (capture={})...", capture)
                self._shared_warmup(capture)
        finally:
            if saved is not None:
                model._restore_deltanet_states(saved, model.device)
                self.dflash_reset()
        return result

    def decode_forward(self, *args, exact_sampling=None, **kwargs):
        # Traced decode (single-device and TP): trace captured at pos 0 in warmup, replayed here.
        # Valid for TP — GDN state is in fixed in-place buffers, and prefill only replays pre-warmed programs.
        if exact_sampling is not None:
            # Exact on-device sampling: replay the ordinary (logits) decode trace without reading it back,
            # then the bucket's sampler trace; only token ids and flags return to the host.
            for key in ("sampling_params", "prompt_tokens", "output_tokens", "reset_batch"):
                kwargs.pop(key, None)
            kwargs["read_from_device"] = False
            start_pos = kwargs.get("start_pos", args[1] if len(args) > 1 else None)
            self._sampler_last_positions = [int(p) for p in torch.as_tensor(start_pos).reshape(-1).tolist()]
            tt_out = self.decode_forward(*args, **kwargs)
            return self._decode_exact(tt_out, exact_sampling)
        if not getattr(self, "_decode_logged", False):
            self._decode_logged = True
            logger.info("Decode trace replay active (Qwen)")
        model = self.model[0]
        if self.dflash_available():
            # Ordinary target decode has no DFlash feature commit. Invalidate
            # until a fresh full prefill; unlike MTP, last-hidden cannot repair it.
            self.dflash_reset()
        # Batched serving: apply vLLM's condense slot_remap to the per-slot GDN recurrent/conv state
        # BEFORE the decode trace reads it. The plugin remaps its own buffers (and the seed RNG via
        # super().decode_forward), but GDN state is model-internal, so mirror the same reindex here.
        # slot_remap is passed through unchanged so the seed-RNG remap inside super() still runs.
        if model.multiuser is not None:
            slot_remap = kwargs.get("slot_remap")
            if slot_remap is not None:
                if self.shared_available():
                    self._shared.remap(slot_remap)  # GDN slots plus the slots' drafter rings and anchors
                else:
                    model.multiuser.remap(slot_remap)
        elif model.args.max_batch_size > 1:
            slot_remap = kwargs.get("slot_remap")
            if slot_remap is not None:
                model._remap_gdn_slots(slot_remap)
        # Decode bucketing (default on; TT_DECODE_BUCKETING=0 off): slice host inputs to the
        # smallest power-of-2 width >= active prefix [0:num_active) before the base forward.
        # No runner edit / output re-pad — plugin reads unpadded_batch_size in slot order.
        # Each width keeps its own trace metadata, inputs, and output.
        args = list(args)

        def _read(name, pos):
            if name in kwargs:
                return kwargs[name]
            return args[pos] if pos < len(args) else None

        def _write(name, pos, val):
            if name in kwargs:
                kwargs[name] = val
            elif pos < len(args):
                args[pos] = val

        tokens = _read("tokens", 0)
        if os.environ.get("TT_DECODE_BUCKETING", "1") == "1" and tokens is not None:
            start_pos = _read("start_pos", 1)
            width = int(tokens.shape[0])
            num_active = int((start_pos != -1).sum()) if start_pos is not None else width
            num_active = max(1, min(num_active, width))
            bucket = min(width, 1 << max(0, (num_active - 1).bit_length()))  # smallest pow2 >= num_active
            if model.multiuser is not None:
                from models.demos.blackhole.qwen36.tt.multiuser import user_width

                bucket = user_width(num_active)
            # Keep full width when slot_remap is set: remap indexes the full slot space (tokens /
            # GDN). Rare (row moves only); bucketing resumes next step. Check kw + positional #10.
            if _read("slot_remap", 10) is not None:
                bucket = width
            if bucket < width:
                _write("tokens", 0, tokens[:bucket])
                if start_pos is not None:
                    _write("start_pos", 1, start_pos[:bucket])
                page_table = _read("page_table", 2)
                if page_table is not None:
                    _write("page_table", 2, page_table[:bucket])
                tokens = tokens[:bucket]

        if tokens is not None:
            B = int(tokens.shape[0])
            store = getattr(self, "_bucket_trace_store", None)
            if store is None:
                store = self._bucket_trace_store = {}
            if B not in store:
                store[B] = (defaultdict(lambda: None), defaultdict(lambda: None), defaultdict(lambda: None))
            self.trace_ids_decode, self.trace_inputs_decode, self.trace_output_decode = store[B]
            # Key the sampling trace by bucket width too: Generator binds it to one logits tensor
            # by identity, and each decode-bucket width has its own.
            for _m in self.model:
                _sm = getattr(_m, "sampling", None)
                if _sm is not None and hasattr(_sm, "set_trace_bucket"):
                    _sm.set_trace_bucket(B)
        if model.multiuser is not None and tokens is not None and getattr(self, "_foreign_trace_ran", False):
            # Another trace replayed since the last plain step (a prefill, a shared round, a drafter pass). This
            # width's decode inputs were allocated after the prefill capture, so they may lie in a released
            # intermediate of an earlier-captured trace and hold its bytes now (the LFM3 trace-input bug): restage
            # every input from the host, and invalidate the device-fed positions so a reset never keeps a device
            # token (sync scheduling: the host tokens are authoritative).
            self._foreign_trace_ran = False
            kwargs["reset_batch"] = True
            inputs = self.trace_inputs_decode[True]
            if inputs:
                position = inputs[0][1]
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(torch.full(tuple(int(d) for d in position.shape), -2, dtype=torch.int32),
                                    dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), position)
        return super().decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        # Phase 1 compiles every chunk, masked bucket and fill width without parking a trace.
        # Phase 2 reuses those buffers and programs, then captures the chunk.
        if enable_trace and getattr(self, "already_warmed_up_prefill", False):
            return
        if self.mtp_available():
            # Prefill runs before decode in both warmup phases. Its trace must not
            # claim scratch addresses subsequently used for persistent wide inputs.
            self._init_mtp_buffers()
        if os.environ.get("QWEN36_DFLASH") == "1":
            if self.model[0].multiuser is not None:
                self._init_shared_drafter()
            else:
                self._init_dflash_buffers()
        vision = getattr(self.model[0], "vision_model", None)
        if vision is not None and not getattr(self, "_vision_warmed", False):
            # Compile the tower's kernels on a small image now, not on the first image request (seconds).
            patch_dim = vision.reference_model.patch_embed.proj.weight[0].numel()
            vision.encode(torch.zeros(16 * 16, patch_dim, dtype=torch.bfloat16), torch.tensor([[1, 16, 16]]))
            self._vision_warmed = True
        dflash = self.dflash_available()
        if dflash and enable_trace and not (
            self._dflash_prefill_compiled and self._dflash_decode_compiled
        ):
            raise RuntimeError("DFlash requires the complete compile phase before prefill trace capture")
        # Size the chunk-trace page table to the full KV cache (not a hardcoded 4096) so served ISL
        # isn't capped; still captures one chunk — just a bigger page-table tensor.
        if kv_cache:
            # Round to a multiple of 32: paged/chunked SDPA needs the page-table stick % 32 == 0.
            num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        else:
            num_blocks = math.ceil(_PREFILL_WARMUP_BUCKET / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        if kv_cache:
            page_table.clamp_(max=int(kv_cache[0][0].shape[0]) - 1)
        model = self.model[0]
        # Batched serving (max_num_seqs>1): the decode buffers are [B,...], but prefill runs B=1. Bind
        # the PERSISTENT B=1 GDN prefill scratch and capture the chunk trace against IT, so long prompts
        # (>chunk_size) replay the traced chunk-outer path per user instead of the slower eager fallback.
        # The scratch is not freed (prefill_paged_slots rebinds it per request); the batched decode
        # buffers are restored before the decode-trace warmup captures at [B,...].
        # Single-device multi-user serving prefills through the ordinary single-user path.
        batched = model.args.max_batch_size > 1 and model.num_devices > 1
        logger.info(
            f"Starting Qwen prefill warmup: chunk-prefill trace{' (batched, B=1 scratch)' if batched else ''} "
            f"(chunk={_PREFILL_WARMUP_CHUNK}, page_table_blocks={num_blocks})..."
        )
        prev = model._bind_gdn_prefill_scratch() if batched else None
        saved = model._save_deltanet_states() if dflash else None
        try:
            model.capture_prefill_trace_chunked(
                self.mesh_device, page_table, chunk_size=_PREFILL_WARMUP_CHUNK, capture_chunk_trace=enable_trace
            )
            # Prefix caching for this model is exact prefix resume only (prefix_resume.py); it is
            # declared here with no snapshot capacity and enabled once the chunk trace it resumes
            # is captured (single device, single sequence).
            from vllm_tt_plugin.prefix_resume import REGISTRY

            resumable = enable_trace and not batched and model.num_devices == 1
            if not enable_trace and not batched and model.num_devices == 1:
                # Compile the resume state round trip (DRAM-staged copies into the fixed state
                # buffers) before any trace is parked; the round trip itself is exact.
                model._restore_gdn_snapshot(model._save_gdn_snapshot())
            REGISTRY.configure(
                _PREFILL_WARMUP_CHUNK,
                _BLOCK_SIZE,
                int(os.environ.get("QWEN36_PREFIX_SNAPSHOTS", "8")) if resumable else 0,
            )
            if dflash:
                # Target warmup committed every feature bucket through the
                # drafter's fixed 32-row append. Now compile the B-row append,
                # noisy backbone, head/selector, and verifier before any capture.
                self.dflash_warmup(capture=False)
                self._dflash_prefill_compiled = True
        finally:
            if prev is not None:
                model._unbind_gdn_prefill_scratch(prev)
            if saved is not None:
                model._restore_deltanet_states(saved, model.device)
                self.dflash_reset()
        if not enable_trace and model.multiuser is not None and getattr(model, "dflash", None) is not None:
            # Shared rounds: rings, round inputs and candidates after the prefill compile, before any decode program.
            self._init_shared_rounds()
        self.already_warmed_up_prefill = enable_trace

    def _warmup_model_decode_base(self, *args, **kwargs):
        # Defer to WarmupForwardMixin, which warms the paged-SDPA + GDN decode path at pos 0.
        # Drop stale `non_greedy_decoding_on_device` from the old vLLM plugin; no-op for Qwen.
        kwargs.pop("non_greedy_decoding_on_device", None)
        self._validate_device_sampling_request(kwargs.get("can_sample_on_device", False))
        if kwargs.get("can_sample_on_device", False) and self.model[0].num_devices == 1:
            kwargs["greedy_only"] = True
        return warmup_decode_buckets(self, super().warmup_model_decode, *args, **kwargs)
