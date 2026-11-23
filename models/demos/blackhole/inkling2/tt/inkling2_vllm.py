# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for Inkling-2-Mini on one P150 (vllm_tt_plugin; registered through EXTRA_MODELS_DIR, see
vllm_bundle/models/inkling2/vllm_metadata.json, and by the inkling2_config general plugin in vllm_bundle/ext).

- One sequence at a time (max_num_seqs 1).  The runtime (tt/model.py Inkling2Model) owns every cache: 48 KV slots (one
  per layer application: the loop layers 6-17 have one per pass) and the short-conv states.  vLLM's KV pages are
  bookkeeping only: one FullAttentionSpec sized for max_model_len, and allocate_kv_cache allocates nothing.
- Prefill: every new request resets the runtime and prefills its whole prompt (the runtime chunks it: the plan's
  prefill_chunk rows, 4096 in every profile; INKLING2_PREFILL_CHUNK overrides).  Every prefill program is compiled at
  server start (warmup_model_prefill: 23 chunk shapes at 4096; kernel binaries persist in the mounted kernel cache, so
  only the first start of a new runtime pays the full compile).  Every request is computed from a clean state (A, B,
  A gives A's tokens twice; a cancelled request leaves nothing behind) or from an exact prefix snapshot:
- Prefix caching (NOTES-prefix.md, tt/prefix_cache.py; the plan's prefix_snapshots > 0 and vLLM
  --enable-prefix-caching): vllm_tt_plugin's exact prefix resume (prefix_resume.REGISTRY, patched:
  src/docker/vllm-tt-plugin-prefix.patch).  A prefill publishes a device snapshot of its slot's running state (sliding
  K / V rings, conv rings, glue rings; batched speculation: + the drafter's per-slot rings) at chunk boundaries a full
  prefill crosses (multiples of 2048 batched / of the prefill chunk single stream: its final segment's start, its
  aligned end, boundaries where cached prompts fork); a request whose prompt starts with a snapshot's tokens is
  scheduled from it (num_computed_tokens = S, the full-attention blocks of [0, S) shared), restores it into its slot
  and prefills only [S, L): bitwise what a full prefill gives.
  Single stream: the full / drafter caches are in identity order, so a snapshot is valid while no prefill rewrote its
  rows (prefix_cache.IdentityRows).
- Decode: the self-feeding decode traces of tt/fast/decode_loop.TokenDecoder (one per full-attention bucket): the
  device embeds the token, runs the 48 applications and the LM head, picks the next token (greedy argmax, or the
  exact on-device vLLM sampler for eligible sampled requests: fast/sampling.py, incl. the full 200K vocabulary for
  top_k unset, top_p, min_p, penalties, seeds) and advances its own position / attention tables; the host reads back
  one token id per step.  The vllm_tt_plugin asks `device_sampling_eligible` (fast/sampling.device_eligible; the
  image patches the plugin to use it).  Requests the device cannot sample exactly (logprobs, structured output, logit
  bias, logits processors, temperature > 8, top_k > 64 with top_p < 1, ...) get the step's FP32 logits (padded with
  -inf to vLLM's 201,024-entry vocabulary) and vLLM samples them on the host.  INKLING2_DEVICE_DECODE=0 restores the
  host-fed path (model.decode_traced + host sampling of every step).
- Positions are checked: a decode step must continue the sequence the runtime holds (its position = the number of
  tokens the runtime has consumed); anything else raises instead of producing tokens from a wrong state.
- Speculative decoding (INKLING2_SPEC=on, tt/spec.py SpecDecoder; default off): greedy and device-sampled requests
  run verify / draft cycles with the checkpoint's block drafter (hf/mtp.safetensors): INKLING2_SPEC_DRAFTS = auto
  (default: 1..3 drafts per cycle, picked per cycle from the request's recent acceptance and the cycle costs measured
  at capture) or m (fixed).  The verify embeds its rows on the device and the next cycle's verify + drafter context are
  enqueued behind each draft (the host's acceptance work and vLLM's per-token steps overlap device work).  A request
  whose last 64 cycles emitted fewer tokens per ms than 0.9 x plain decoding hands off to the plain TokenDecoder (a
  drafter that does not pay stops costing).  vLLM still asks one token per decode step: a cycle emits the target's greedy
  tokens of several positions at once and the following steps are answered from that queue (each checked against the
  (token, position) vLLM sends).  Prefill also writes the drafter's context.  Sampled / penalized requests the device
  samples run sampled cycles: every verify row is sampled on the device (its penalties counting the drafts before
  it) and the drafts are accepted while they equal the samples = speculative rejection sampling of the drafter's
  point-mass proposal, exact (a seeded request emits the tokens of plain decoding; fast/sampling.py, NOTES-spec.md
  "Sampled cycles").  Host-logits steps (logprobs, ...) decode plainly through the TokenDecoder; a request that
  leaves the speculative path continues plainly from the exact committed state, and so does a request near
  max_model_len (a cycle needs R + 8 free positions).
- Batched mode (vLLM --max-num-seqs S, 2..16; NOTES-batch.md): Inkling2Model(max_num_seqs=S, kv_blocks=NB) keeps S
  sequence slots (sliding / conv / glue rings, sampler history per slot) and one shared full-attention pool of NB vLLM
  blocks (NB = vllm_tt_plugin worker's count: get_max_tokens_all_users + 64 S tokens, / 64; the sequences' block tables
  address it).  The plugin runs its batched policy (capability supports_chunked_prefill_single_device_batched):
  decode steps of every running sequence alternate with prefill turns of <= 2048 new tokens (chunks; start_pos > 0 =
  continuation).  The plugin's logical state slots move (slot_remap); the device state never does: the adapter keeps
  phys[logical] (phys'[i] = phys[remap[i]]; prefill row i -> phys[empty_slots[i]], decode row b -> phys[b]).
  prefill_forward prefills each row's chunk into its slot (start 0: reset_slot first) -> logits [Bp, 1, vocab];
  decode_forward runs the fast/batch_decode.BatchDecoder step over the B real rows (pad rows start_pos -1): token ids
  [P] for all-greedy steps (sampling_params; the image's plugin patch selects decode_only sampling in batched mode) and
  exact_sampling steps, else the step's logits [P, 1, vocab] for vLLM's host sampler.  Positions are checked per slot
  (prefill continuation start and decode position = the tokens the slot holds).  The stock attention path is
  single-stream only.  INKLING2_MAX_TOKENS_ALL_USERS = the KV pool (default 262,144 tokens).
- Batched speculation (INKLING2_SPEC auto (default) | on | off with max_num_seqs > 1; tt/spec.py BatchSpec,
  NOTES-spec.md "Batched speculation"): the BatchDecoder steps carry verify groups (a sequence's anchor + drafts as
  consecutive rows) next to plain rows, the batched drafter drafts for up to 4 sequences per pass, every prefill chunk
  and decode row writes the drafter context.  vLLM still asks one token per sequence per step: decode_forward answers
  from each sequence's queue of accepted tokens and runs a device step only when some queue is empty (every other
  speculating sequence cycles in that step too).  auto: speculate for all sequences or none, per active-sequence count,
  whichever served faster (measured); host-sampled steps and penalized requests decode plainly.  Each sequence's
  tokens equal its tokens alone (verify rows are bitwise its decode rows).  The plugin allocates KV blocks
  TT_DECODE_LOOKAHEAD (16, launcher) tokens ahead of vLLM's position for the cycles that run ahead.

Context profile and memory: tt/kvplan.py `plan(max_model_len)` picks the expert precision, the attention projection
map, the full-attention K / V cache formats, the decode buckets (one trace per bucket) and refuses configurations that
do not fit the P150 (experts bfp4:gptq and attention q|gate GPTQ-BFP4 + k / v / o BFP8 in every profile; fast <=
65,536: BF16 KV; long <= 131,072: BFP8 KV; native <= 1,048,576: K BFP8 / V BFP4; NOTES-longctx.md).  After the load the adapter checks the free DRAM against the plan's prefill
reserve and refuses to serve below it.
Env: MODEL_WEIGHTS_DIR / HF_MODEL (checkpoint dir; default the served model dir), INKLING2_PRECISION (expert map
override; "quality" = BFP8 experts + BFP8 attention), INKLING2_ATTN_PRECISION (attention map override, e.g. "bfp8"), INKLING2_KV ("<k>:<v>" cache formats override: bf16 | bfp8 | bfp4), INKLING2_DEVICE_DECODE (1 default |
0; batched: 0 = no exact device sampler), INKLING2_EMBED (auto default: the BF16 embedding table goes to the device
when DRAM keeps EMBED_RESERVE_GIB free after it and after the speculative decoder's planned memory (INKLING2_SPEC=on:
kvplan SPEC_FIXED_GIB + drafter KV), else the host uploads each step's embedding rows | device | host; the same for
the BatchDecoder), INKLING2_SPEC (single stream: off default | on; batched: auto default | on | off),
INKLING2_SPEC_DRAFTS (single stream: auto default | 1..7; NOTES-spec.md "Spec5"),
INKLING2_PREFILL_CHUNK (prefill chunk rows override: <= 512 or a multiple of 512 up to 4096),
INKLING2_MAX_TOKENS_ALL_USERS (batched: the shared KV pool in tokens, default kvplan.DEFAULT_KV_TOKENS = 262,144),
INKLING2_PREFIX_SNAPSHOTS (auto default | N: device prefix snapshots, kvplan), INKLING2_PREFIX_CACHING (on | off).
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import torch
from loguru import logger

from .kvplan import (DEFAULT_KV_TOKENS, EMBED_RESERVE_GIB, MAX_SEQS, PREFILL_RESERVE_GIB, SPEC_BUCKETS, SPEC_FIXED_GIB,
                     SPEC_KV_PER_TOKEN, PlanError, batch_spec_gib, batch_spec_mode, plan)
from .model import Inkling2Model

BLOCK = 64  # vLLM page size (single stream: bookkeeping only; batched: the full-attention pool's block)


def _free_dram(mesh_device) -> int:
    import ttnn

    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return v.num_banks * v.total_bytes_free_per_bank


def _embed_mode(mesh_device, table_bytes: int, later_bytes: int = 0) -> str:
    """auto: the table goes to the device when EMBED_RESERVE_GIB stays free after it and after `later_bytes` still to
    be allocated (the speculative decoder's drafter + context, allocated after the TokenDecoder)."""
    mode = os.environ.get("INKLING2_EMBED", "auto")
    if mode in ("device", "host"):
        return mode
    return "device" if _free_dram(mesh_device) - table_bytes - later_bytes >= EMBED_RESERVE_GIB * 2 ** 30 else "host"


class Inkling2ForCausalLM:
    """vLLM-compatible wrapper for Inkling2LoopForConditionalGeneration (text) on one Blackhole P150."""

    model_capabilities = {
        # exact prefix resume (NOTES-prefix.md): vllm_tt_plugin.prefix_resume with the runtime's device snapshots
        "supports_prefix_caching": True,
        "supports_async_decode": False,
        # greedy unpenalized steps: device argmax; eligible sampled / penalized steps: exact device sampler
        "supports_sample_on_device": True,
        "supports_greedy_only_sample_on_device": True,
        "supports_exact_device_sampling": True,
        # max_num_seqs > 1: the plugin's batched policy (chunked prefill turns alternating with batched decode steps)
        "supports_chunked_prefill_single_device_batched": True,
    }

    def __init__(self, model: Inkling2Model, *, vocab_size: int, max_seq_len: int, max_num_seqs: int = 1,
                 **_vllm_kwargs):
        # (**_vllm_kwargs: vLLM's model-protocol check wants an __init__ that accepts vllm_config)
        self.model = model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_num_seqs = max_num_seqs
        self.model_args = [SimpleNamespace(max_batch_size=max_num_seqs, max_seq_len=max_seq_len,
                                           vocab_size=vocab_size)]
        self.already_warmed_up_prefill = False
        self.traced = False
        self.pos = None  # tokens the runtime has consumed for the current sequence (None: no sequence)
        self.decoder = None
        self.spec = None
        self.batch = None
        self.bspec = None
        self.prefix = None  # prefix_cache.PrefixCache (exact prefix resume; None: off)
        if max_num_seqs > 1:
            from .fast.batch_decode import BatchDecoder

            spec_mode = batch_spec_mode()
            embed = _embed_mode(model.device, model.embed.table.numel() * 2,
                                int(batch_spec_gib(model.kv_blocks, max_num_seqs) * 2 ** 30) if spec_mode != "off" else 0)
            logger.info(f"Inkling-2-Mini: batched decode of up to {max_num_seqs} sequences "
                        f"({model.kv_blocks} KV pool blocks), embedding table on the {embed}, speculation {spec_mode}")
            on = spec_mode != "off"
            self.batch = BatchDecoder(model, embed=embed, taps=on, extra_buckets=SPEC_BUCKETS if on else ())
            if on:
                from .spec import BatchSpec

                self.bspec = BatchSpec(model, self.batch, log=lambda s: logger.info(s),
                                       modes="auto" if spec_mode == "auto" else "spec")
            self.phys = list(range(max_num_seqs))  # physical slot of each logical (plugin) state slot
            self.slot_pos = [None] * max_num_seqs  # tokens the sequence in physical slot s holds (None: none)
            self.width = max_seq_len // BLOCK      # page-table row width of the runtime
            self._init_prefix()
            return
        if os.environ.get("INKLING2_DEVICE_DECODE", "1") != "0":
            from .fast.decode_loop import TokenDecoder

            spec_bytes = (int((SPEC_FIXED_GIB * 2 ** 30) + max_seq_len * SPEC_KV_PER_TOKEN)
                          if os.environ.get("INKLING2_SPEC", "off") == "on" else 0)
            embed = _embed_mode(model.device, model.embed.table.numel() * 2, spec_bytes)
            logger.info(f"Inkling-2-Mini: device decode (self-feeding traces), embedding table on the {embed}")
            self.decoder = TokenDecoder(model, embed=embed)
        self._spec_queue = []   # [(position, token in, token out)]: target tokens of the last cycle not yet asked for
        self._spec_mode = None  # this request: None (undecided) | "spec" | "plain"
        if os.environ.get("INKLING2_SPEC", "off") == "on":
            if self.decoder is None:
                raise ValueError("INKLING2_SPEC=on needs the device decode (INKLING2_DEVICE_DECODE=1)")
            from .spec import SpecDecoder

            m = os.environ.get("INKLING2_SPEC_DRAFTS", "auto")
            rows = (2, 3, 4) if m == "auto" else int(m) + 1
            logger.info(f"Inkling-2-Mini: speculative decoding of greedy and device-sampled requests, drafts per cycle: {m}")
            # table: the TokenDecoder's device embedding table (None in host-embedding mode) -> the verify embeds its
            # rows on the device and the next verify is enqueued behind each draft; plain: the hand-off decoder
            self.spec = SpecDecoder(model, rows=rows, log=lambda s: logger.info(s), table=self.decoder.table,
                                    plain=self.decoder)
        self._init_prefix()

    def _init_prefix(self, registry=None):
        """Prefix caching (NOTES-prefix.md): exact prefix resume through vllm_tt_plugin's registry (`registry`: a
        stand-in with its API in tests) when the model has a snapshot pool.  Chunk alignment = the boundaries a full
        prefill crosses: batched, the scheduler's non-final chunk ends (TT_PREFILL_CHUNK_ALIGN, 2048 = the turn
        budget); single stream, the runtime's prefill chunk.  The registry is active only once vLLM's scheduler
        attaches it (--enable-prefix-caching).  Without snapshots the registry is still configured (no snapshot
        capacity): an attached registry that was never configured would fall back to vLLM's KV-only prefix hits."""
        self.prefix = None
        if registry is None:
            try:
                from vllm_tt_plugin.prefix_resume import REGISTRY as registry
            except ImportError:
                logger.warning("Inkling-2-Mini: vllm_tt_plugin.prefix_resume not found: prefix caching off")
                return
        batched = self.batch is not None
        align = int(os.environ.get("TT_PREFILL_CHUNK_ALIGN", "2048")) if batched else self.model.prefill_chunk
        if self.model.snaps is None:
            registry.configure(align, BLOCK, 0)
            return
        from .prefix_cache import PrefixCache

        self.prefix = PrefixCache(self.model, registry, align, identity=not batched, fork_hints=True,
                                  log=lambda s: logger.info(s))
        logger.info(f"Inkling-2-Mini: prefix caching by exact resume: {self.model.snaps.count} device snapshots of "
                    f"{self.model.snaps.nbytes / 2 ** 20:.0f} MiB, resume points every {align} tokens")

    def _prefix_at(self, row_tokens, end: int, final: bool, resumed_at: int, slot: int, block_row, request_id):
        """prefill(at=...) hook of one prefill call: publish the slot's state at the chunk boundaries the prefix
        cache picks (PrefixCache.boundary).  None when prefix caching is inactive."""
        pc = self.prefix
        if pc is None or not pc.active:
            return None

        def at(pos):
            if pc.boundary(row_tokens, pos, end, final, resumed_at, request_id):
                pc.publish(row_tokens, pos, slot, block_row)

        return at

    def exact_sampling_available(self):
        """vllm_tt_plugin: exact on-device sampling of eligible sampled / penalized decode rows."""
        if self.batch is not None:
            return os.environ.get("INKLING2_DEVICE_DECODE", "1") != "0" and self.batch.sampler is not None
        return self.decoder is not None and self.decoder.sampler is not None

    @staticmethod
    def device_sampling_eligible(params, num_output_tokens=0):
        """vllm_tt_plugin: requests the device samples exactly (fast/sampling.py, incl. the full vocabulary)."""
        from .fast.sampling import device_eligible

        return device_eligible(params, num_output_tokens)

    @staticmethod
    def device_row_sampling(params, seed):
        """vllm_tt_plugin: the device sampler's parameters of one request (fast/sampling.RowSampling)."""
        from .fast.sampling import row_sampling

        return row_sampling(params, seed)

    # ------------------------------------------------------------------ vLLM model-protocol shims
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim (the TT runner never calls it)
        raise NotImplementedError

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError

    # ------------------------------------------------------------------ construction / KV bookkeeping
    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """One full-attention spec (block 64) so vLLM's scheduler accounts the tokens; single stream: bookkeeping only
        (the runtime owns the KV); batched: vLLM's block ids address the runtime's full-attention pool."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        model_config = vllm_config.model_config
        vllm_config.cache_config.block_size = BLOCK
        return {"attn": FullAttentionSpec(block_size=BLOCK,
                                          num_kv_heads=model_config.get_num_kv_heads(vllm_config.parallel_config),
                                          head_size=model_config.get_head_size(), dtype=torch.bfloat16,
                                          sliding_window=None)}

    @classmethod
    def get_max_tokens_all_users(cls, model_name: str = "", num_devices: int = 1, tt_data_parallel: int = 1,
                                 max_model_len: int | None = None, max_num_seqs: int | None = None, **kwargs) -> int:
        """vllm_tt_plugin worker: the KV pool in tokens (it adds 64 per sequence and divides by the block size).
        Single stream: max_model_len; batched: INKLING2_MAX_TOKENS_ALL_USERS (default DEFAULT_KV_TOKENS), shared by
        all sequences (one long request up to max_model_len or many shorter ones)."""
        if int(max_num_seqs or 1) > 1:
            return int(os.environ.get("INKLING2_MAX_TOKENS_ALL_USERS") or DEFAULT_KV_TOKENS)
        return int(max_model_len or 8192)

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1,
                              optimizations=None, **kwargs):
        if tt_data_parallel != 1 or mesh_device.get_num_devices() != 1:
            raise ValueError("Inkling-2-Mini serving runs on one P150 (no data / tensor parallelism)")
        S = int(max_batch_size)
        if not 1 <= S <= MAX_SEQS:
            raise ValueError(f"Inkling-2-Mini serves 1..{MAX_SEQS} sequences at a time (--max-num-seqs {S})")
        batched = {}
        if S > 1:
            if os.environ.get("INK_ATTN", "fast") == "stock":
                raise ValueError("Inkling-2-Mini: batched decode needs the fused attention (INK_ATTN=fast)")
            T = cls.get_max_tokens_all_users(max_model_len=int(max_seq_len), max_num_seqs=S)
            batched = dict(max_num_seqs=S, kv_tokens=T)
        path = os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL") or hf_config._name_or_path
        try:
            p = plan(int(max_seq_len), **batched)
        except PlanError as e:
            raise ValueError(f"Inkling-2-Mini: {e}") from None
        logger.info(f"Loading weights from {path}")
        logger.info(f"Inkling-2-Mini: weights {path}, {p.summary()}\n{p.table()}")
        # the model picks its attention decode path from INK_ATTN (tt/model.py): follow the plan unless set
        os.environ.setdefault("INK_ATTN", "fast" if p.attn_decode == "fused" else "stock")
        import ttnn

        fmt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
        slots = dict(max_num_seqs=S, kv_blocks=p.kv_blocks) if S > 1 else {}
        # prefix caching resumes the fused paths' state (rows glue rings, fused attention conv rings); the ttnn glue /
        # stock attention decode keep other states
        snaps = p.prefix_snapshots
        if snaps and (os.environ.get("INK_GLUE", "fast") == "ttnn" or os.environ["INK_ATTN"] == "stock"):
            logger.warning("Inkling-2-Mini: prefix caching needs the fused glue and attention: off")
            snaps = 0
        model = Inkling2Model(mesh_device, path, max_seq_len=max_seq_len, precision=p.precision,
                              attn_precision=p.attn_precision, attn_policy={"k": fmt[p.k_dtype], "v": fmt[p.v_dtype]}, buckets=p.buckets,
                              moe_balance=p.moe_balance, prefill_chunk=p.prefill_chunk, log=lambda m: logger.debug(m),
                              prefix_snapshots=snaps, **slots)
        vocab = int(getattr(getattr(hf_config, "text_config", hf_config), "vocab_size"))
        out = cls(model, vocab_size=vocab, max_seq_len=max_seq_len, max_num_seqs=S)
        free = _free_dram(mesh_device) / 2 ** 30
        logger.info(f"Inkling-2-Mini: {free:.2f} GiB DRAM free after the load (plan: {p.headroom_gib:.2f} GiB headroom "
                    f"+ {PREFILL_RESERVE_GIB} GiB prefill reserve)")
        if free < PREFILL_RESERVE_GIB:
            raise RuntimeError(f"Inkling-2-Mini: only {free:.2f} GiB DRAM free after the load, below the "
                               f"{PREFILL_RESERVE_GIB} GiB a prefill chunk needs ({p.summary()}): use a smaller "
                               "--max-model-len or 4-bit formats (INKLING2_PRECISION / INKLING2_KV)")
        return out

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Nothing to allocate (the runtime owns every cache); batched: vLLM's block count must be the pool's."""
        if self.batch is not None:
            nb, bs = int(kv_cache_shape[0]), int(kv_cache_shape[2])
            if (nb, bs) != (self.model.kv_blocks, BLOCK):
                raise ValueError(f"Inkling-2-Mini: vLLM allocates {nb} KV blocks of {bs} tokens, the runtime's pool has "
                                 f"{self.model.kv_blocks} blocks of {BLOCK} (get_max_tokens_all_users / --block-size)")
        return []

    # ------------------------------------------------------------------ helpers
    def _pad_vocab(self, logits: torch.Tensor) -> torch.Tensor:
        """[V_unpadded] -> [1, 1, vocab_size] float32 (-inf beyond the unembedding's rows)."""
        out = torch.full((1, 1, self.vocab_size), float("-inf"), dtype=torch.float32)
        out[0, 0, : logits.numel()] = logits
        return out

    def _page_row(self, row: torch.Tensor) -> torch.Tensor:
        """vLLM's block-table row (int32 [W'], W' = min(max_model_len / 64, KV blocks)) -> int32 [max_seq_len / 64]
        (columns past the sequence: don't care)."""
        w = int(row.shape[0])
        if w >= self.width:
            return row[: self.width]
        return torch.cat([row, torch.zeros(self.width - w, dtype=row.dtype)])

    def _sampling(self, req):
        """fast.sampling.RowSampling of a plugin SamplingRequest (None: greedy unpenalized = device argmax)."""
        from .fast.sampling import RowSampling

        s = req.spec
        spec = RowSampling(temperature=s.temperature, top_k=s.top_k, top_p=s.top_p, min_p=s.min_p,
                           presence=s.presence, frequency=s.frequency, repetition=s.repetition, seed=s.seed)
        return None if spec.greedy and not spec.penalized else spec

    def _prefill_batched(self, tokens, page_table, prompt_lens, start_pos, empty_slots, request_ids=None,
                         final_rows=None):
        """Prefill turn: row i's chunk tokens[i, start_pos[i]:prompt_lens[i]] into physical slot
        phys[empty_slots[i]] with its page row (start 0: a new sequence, reset_slot first; a start the scheduler
        resumed from a prefix snapshot (request_ids[i]): the snapshot restored into the slot first; else the
        continuation of the tokens the slot holds) -> logits [Bp, 1, vocab] of each row's last chunk token (the plugin
        reads the rows whose prompt ends here).  Prefix caching publishes the slot's state at the boundaries
        PrefixCache.boundary picks (final_rows[i]: row i's prompt ends in this turn)."""
        Bp = int(tokens.shape[0])
        if empty_slots is None or len(empty_slots) != Bp:
            raise ValueError(f"Inkling-2-Mini batched prefill needs one state slot per row (empty_slots {empty_slots})")
        if start_pos is None:
            start_pos = [0] * Bp
        model, V, pc = self.model, self.model.cfg.unpadded_vocab, self.prefix
        out = torch.full((Bp, 1, self.vocab_size), float("-inf"), dtype=torch.float32)
        for i in range(Bp):
            s = self.phys[int(empty_slots[i])]
            start, end = int(start_pos[i]), int(prompt_lens[i])
            if not 0 <= start < end <= self.max_seq_len:
                raise ValueError(f"Inkling-2-Mini prefill chunk {start}..{end} (max_seq_len {self.max_seq_len})")
            rid = request_ids[i] if request_ids is not None else None
            row_tokens = tokens[i, :end]
            resumed = pc is not None and pc.resuming(rid, start)
            if start == 0:
                model.reset_slot(s)
            elif resumed:
                self.slot_pos[s] = None
                pc.resume(rid, row_tokens, start, s)  # the slot's state = the snapshot of tokens[:start]
            elif self.slot_pos[s] != start:
                raise RuntimeError(f"Inkling-2-Mini prefill continues at {start} in slot {s} but the slot holds "
                                   f"{self.slot_pos[s]} tokens")
            self.slot_pos[s] = None  # until the chunk is in
            final = bool(final_rows[i]) if final_rows is not None else True
            at = self._prefix_at(row_tokens, end, final, start if resumed else -1, s, page_table[i], rid)
            if self.bspec is not None:  # + the drafter's context of the chunk's rows
                logits = self.bspec.prefill(tokens[i, start:end], start, s, self._page_row(page_table[i]), at=at)
            else:
                logits = model.prefill(tokens[i, start:end], start, slot=s, page_row=self._page_row(page_table[i]),
                                       at=at)
            self.slot_pos[s] = end
            self.batch.invalidate(s)
            out[i, 0, :V] = logits
        if self.bspec is not None:
            self.bspec.note_prefill()
        return out

    # ------------------------------------------------------------------ prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, start_pos=None, empty_slots=None,
                        sampling_params=None, enable_trace=False, prefill_request_ids=None, prefill_final_rows=None,
                        **kwargs):
        """Reset the runtime and prefill the request's prompt (start > 0: resume from the prefix snapshot the
        scheduler found, NOTES-prefix.md); returns its last-token logits [1, 1, vocab].  Batched: the prefill turn's
        rows (_prefill_batched)."""
        if sampling_params is not None:
            raise ValueError("Inkling-2-Mini samples prefill logits on the host (sample_on_device_mode decode_only)")
        if self.batch is not None:
            return self._prefill_batched(tokens, page_table, prompt_lens, start_pos, empty_slots,
                                         prefill_request_ids, prefill_final_rows)
        if int(tokens.shape[0]) != 1:
            raise ValueError(f"Inkling-2-Mini prefills one request at a time (got {int(tokens.shape[0])})")
        L = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[1])
        start = int(start_pos[0]) if start_pos is not None else 0
        rid = prefill_request_ids[0] if prefill_request_ids else None
        pc = self.prefix
        if start and not (pc is not None and pc.resuming(rid, start)):
            raise ValueError(f"Inkling-2-Mini prefills whole prompts or resumes a prefix snapshot (start position "
                             f"{start} without one: chunked prefill must be off)")
        model = self.model
        self.pos = None
        if self.decoder is not None:
            self.decoder.invalidate()
        self._spec_queue, self._spec_mode = [], None
        toks = tokens[0, :L]
        restore = None
        if start:
            def restore():
                pc.resume(rid, toks, start, 0)
        if pc is not None:
            pc.note_prefill(start, L)  # identity rows start..L-1 rewritten (snapshots above start go stale)
        at = self._prefix_at(toks, L, True, start if start else -1, 0,
                             page_table[0] if page_table is not None else None, rid)
        if self.spec is not None:  # + the drafter's context of every prompt row
            logits = self.spec.prefill(toks, start, restore, at)
        else:
            if start:
                restore()
            else:
                model.reset()
            logits = model.prefill(toks[start:], start, at=at)
        self.pos = L
        return self._pad_vocab(logits)

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device=False, greedy_only=False, **kwargs):
        """Compile every prefill program the server can hit (the runtime's warmup_prefill: every chunk length as a
        first and as a continuation chunk; with prefix caching the snapshot / restore copies); prefill is never
        traced."""
        if self.already_warmed_up_prefill:
            return
        logger.info(f"Warming up prefill (Inkling-2-Mini: every chunk shape of {self.model.prefill_chunk} rows)")
        t0 = time.perf_counter()
        self.model.warmup_prefill(log=logger.info)
        logger.info(f"Inkling-2-Mini: prefill warmup (chunk {self.model.prefill_chunk}) "
                    f"{time.perf_counter() - t0:.1f} s")
        if self.model.snaps is not None:  # compile the snapshot round trip before any trace is captured
            self.model.snapshot_slot(0, 0)
            self.model.restore_slot(0, 0)
        self.model.reset()
        self.pos = None
        if self.batch is not None:
            self.slot_pos = [None] * self.max_num_seqs
        if self.prefix is not None:
            self.prefix.invalidate_rows()
        self.already_warmed_up_prefill = True

    # ------------------------------------------------------------------ decode
    def warmup_model_decode(self, *args, enable_trace=False, **kwargs):
        """Compile (enable_trace False) or compile + capture (True) the decode step; both end with a reset runtime.
        Device decode: the TokenDecoder traces (main + greedy / sample tails per bucket) are the only decode traces."""
        model = self.model
        if self.batch is not None:  # batched: the BatchDecoder's bucket traces (compiled and captured in capture())
            if enable_trace and not self.traced:
                if self.bspec is not None:
                    logger.info("Warming up decode (Inkling-2-Mini: compiling the batched drafter programs)")
                    self.bspec.compile()  # before any capture
                logger.info("Inkling-2-Mini: Capturing the batched decode traces")
                self.batch.capture()  # ends with model.reset()
                if self.bspec is not None:
                    self.bspec.capture()
                self.traced = True
            self.model.reset()
            self.batch.invalidate()
            self.phys = list(range(self.max_num_seqs))
            self.slot_pos = [None] * self.max_num_seqs
            return
        if self.spec is not None and not self.spec._compiled:
            self.spec.compile()  # every spec program before any trace capture
        if enable_trace:
            if not self.traced:
                logger.info("Inkling-2-Mini: Capturing the decode traces")
                if self.decoder is not None:
                    self.decoder.capture()
                else:
                    model.capture_decode_trace()
                if self.spec is not None:
                    logger.info("Inkling-2-Mini: Capturing the speculative verify / commit / draft traces")
                    self.spec.capture()
                model.capture_reset_trace()  # each request's reset: one replay instead of ~400 dispatched ops
                self.traced = True
        else:
            model.reset()
            model.decode_step(1000, 0)
        model.reset()
        if self.decoder is not None:
            self.decoder.invalidate()
        self._spec_queue, self._spec_mode = [], None
        self.pos = None

    def _spec_next(self, token: int, pos: int, spec=None, req=None):
        """Greedy (spec None) or device-sampled (spec: RowSampling, req: the plugin's SamplingRequest) step through
        the speculative path: the next token, or None when the request continues plainly (the committed state is
        exact: the TokenDecoder seeds from (token, pos))."""
        if self._spec_queue:
            qpos, qtok, nxt = self._spec_queue.pop(0)
            if (qpos, qtok) != (pos, token):
                raise RuntimeError(f"Inkling-2-Mini: speculative queue holds ({qtok}, {qpos}), vLLM sent ({token}, "
                                   f"{pos})")
            return nxt
        sp = self.spec
        if self._spec_mode is None:
            if spec is not None and sp.sampler is None:
                self._spec_mode = "plain"
                return None
            sp.set_sampling(spec, owner=req.owner if req is not None else None,
                            output_ids=req.outputs if req is not None else (),
                            prompt_ids=(req.prompt or ()) if req is not None else ())
            sp.begin(token, pos)  # first draft (prefill wrote the drafter context)
            self._spec_mode = "spec"
        elif spec != sp.sampling:  # the request's sampling changed: continue plainly
            self._leave_spec()
            return None
        if not sp.active:
            self._leave_spec()
            return None
        em = sp.step(token, pos)
        self._spec_queue = [(pos + 1 + i, em[i], em[i + 1]) for i in range(len(em) - 1)]
        return em[0]

    def _decode_batched(self, tokens, start_pos, page_table, enable_trace, sampling_params, exact_sampling,
                        slot_remap):
        """One BatchDecoder step of the B real rows (front-packed; pad rows start_pos -1) -> token ids int32 [P]
        (all-greedy `sampling_params` or `exact_sampling` rows) or logits [P, 1, vocab] (host sampling)."""
        from .fast.batch_decode import BatchRow

        tokens = torch.as_tensor(tokens).reshape(-1)
        start_pos = torch.as_tensor(start_pos).reshape(-1)
        P, S = int(tokens.numel()), self.max_num_seqs
        if slot_remap is not None:  # logical slot i takes the state of logical slot remap[i]; no device copy
            remap = [int(r) for r in torch.as_tensor(slot_remap).reshape(-1)]
            if sorted(remap) != list(range(S)):
                raise ValueError(f"Inkling-2-Mini: slot_remap {remap} is not a permutation of 0..{S - 1}")
            self.phys = [self.phys[r] for r in remap]
        active = start_pos >= 0
        B = int(active.sum())
        if not 1 <= B <= S or not bool(active[:B].all()):
            raise ValueError(f"Inkling-2-Mini batched decode: real rows must be 1..{S} rows packed first "
                             f"(start_pos {start_pos.tolist()})")
        if not (enable_trace and self.traced):
            raise ValueError("Inkling-2-Mini batched decode runs only through the captured BatchDecoder traces")
        if exact_sampling is not None and len(exact_sampling) != B:
            raise ValueError(f"Inkling-2-Mini: {len(exact_sampling)} exact-sampling rows for {B} decode rows")
        if exact_sampling is None and sampling_params is not None and any(
                float(t) != 0.0 for t in list(sampling_params.temperature)[:B]):
            raise ValueError("Inkling-2-Mini: device sampling without exact_sampling rows is greedy only")
        pages = torch.as_tensor(page_table)[:B].clone()  # private copy: the rows are kept as the decoder's mirror
        rows = []
        for b in range(B):
            s, pos = self.phys[b], int(start_pos[b])
            if self.slot_pos[s] != pos:
                raise RuntimeError(f"Inkling-2-Mini decode row {b} at position {pos} but its slot {s} holds "
                                   f"{self.slot_pos[s]} tokens")
            if pos >= self.max_seq_len:
                raise ValueError(f"position {pos} beyond max_seq_len {self.max_seq_len}")
            kw = {}
            req = exact_sampling[b] if exact_sampling is not None else None
            if req is not None:
                kw = dict(sampling=self._sampling(req), owner=req.owner, outputs=req.outputs, prompt=req.prompt or ())
            rows.append(BatchRow(slot=s, token=int(tokens[b]), pos=pos, pages=self._page_row(pages[b]), **kw))
        if exact_sampling is not None or sampling_params is not None:
            out = torch.zeros(P, dtype=torch.int32)
            step = self.bspec.decode if self.bspec is not None else self.batch.step
            out[:B] = torch.as_tensor(step(rows), dtype=torch.int32)
        else:
            lg = self.bspec.decode_logits(rows) if self.bspec is not None else self.batch.step_logits(rows)
            out = torch.zeros((P, 1, self.vocab_size), dtype=torch.float32)
            out[:, :, lg.shape[-1]:] = float("-inf")
            out[:B, 0, : lg.shape[-1]] = lg
        for r in rows:
            self.slot_pos[r.slot] += 1
        return out

    def decode_forward(self, tokens, start_pos, page_table=None, kv_cache=None, enable_trace=True,
                       read_from_device=True, sampling_params=None, exact_sampling=None, slot_remap=None, **kwargs):
        """One decode step of the running sequence -> host logits [1, 1, vocab] (host sampling), or the next token
        id as int32 [rows] when the step samples on the device: `sampling_params` (all-greedy, unpenalized: device
        argmax) or `exact_sampling` (vllm_tt_plugin rows: device_sampling.SamplingRequest(owner, spec, outputs,
        prompt) -> the exact on-device vLLM sampler).  Batched: every running sequence (_decode_batched)."""
        if self.batch is not None:
            return self._decode_batched(tokens, start_pos, page_table, enable_trace, sampling_params,
                                        exact_sampling, slot_remap)
        tokens = torch.as_tensor(tokens).reshape(-1)
        start_pos = torch.as_tensor(start_pos).reshape(-1)
        pos = int(start_pos[0])
        if any(int(p) >= 0 for p in start_pos[1:]):
            raise ValueError("Inkling-2-Mini decodes one sequence at a time")
        if pos != self.pos:
            raise RuntimeError(f"Inkling-2-Mini decode at position {pos} but the runtime holds {self.pos} tokens of "
                               "the current sequence")
        if pos >= self.max_seq_len:
            raise ValueError(f"position {pos} beyond max_seq_len {self.max_seq_len}")
        model = self.model
        token = int(tokens[0])
        dec = self.decoder if enable_trace and self.traced else None
        on_device = exact_sampling is not None or sampling_params is not None
        if on_device:
            if dec is None:
                raise ValueError("Inkling-2-Mini samples on the device only through the device decode traces")
            spec = None
            if exact_sampling is not None:
                req = exact_sampling[0]
                spec = self._sampling(req)
            elif any(float(t) != 0.0 for t in list(sampling_params.temperature)[:1]):
                raise ValueError("Inkling-2-Mini: device sampling without exact_sampling rows is greedy only")
            nxt = None
            if self.spec is not None and self._spec_mode != "plain":
                nxt = self._spec_next(token, pos, spec, req if spec is not None else None)
            if nxt is None:
                self._leave_spec()
                if spec is not None:
                    dec.set_sampling(spec, owner=req.owner, output_ids=req.outputs, prompt_ids=req.prompt or ())
                else:
                    dec.set_sampling(None)
                nxt = dec.decode(token, pos)
            self.pos = pos + 1
            out = torch.zeros(tokens.numel(), dtype=torch.int32)  # padded rows (none expected at max_num_seqs 1)
            out[0] = nxt
            return out
        self._leave_spec()
        if dec is not None:
            logits = dec.step_logits(token, pos)
        elif enable_trace and self.traced:
            logits = model.logits_to_torch(model.decode_traced(token, pos))
        else:
            logits = model.decode_step(token, pos)
        self.pos = pos + 1
        out = self._pad_vocab(logits)
        if tokens.numel() > 1:  # padded rows (none expected at max_num_seqs 1)
            out = torch.cat([out, torch.zeros((tokens.numel() - 1, 1, self.vocab_size))])
        return out

    def _leave_spec(self):
        """The request continues on the plain path (the spec state is exact only between cycles)."""
        if self._spec_queue:
            raise RuntimeError("Inkling-2-Mini: a host-sampled step inside a speculative cycle's queued tokens")
        if self.spec is not None and self._spec_mode == "spec":
            self.spec.abandon()  # an enqueued sampled cycle committed tokens the request will not take
        self._spec_mode = "plain"
