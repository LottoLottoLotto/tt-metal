# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM (TT plugin) model for single-user gemma-4 serving at the full context on one device.

Runs tt/single_user.py (ring-buffered sliding-window KV, chunked prefill, traced decoders) behind the
TT plugin's model interface; selected by ``GEMMA4_SINGLE_USER=1`` (see the plugin's platform.py).

* Weights: the verified native checkpoint ``GEMMA4_NATIVE`` (native_checkpoint.prepare_native_runtime
  refuses a checkpoint whose hashes, equivalence proof or runtime identity do not match; the drafter
  likewise needs its speculative-equivalence proof). ``GEMMA4_ALLOW_UNPROVEN=1`` skips the proof checks
  for development trees only.
* KV: vLLM sees one FullAttentionSpec per layer and budgets ``max_model_len`` tokens of blocks; the
  full-attention pools are sized from vLLM's block count and addressed through vLLM's block table
  (copied in place into the persistent device page table before every step / trace replay). Sliding
  layers use model-owned rings of ``RING_TOKENS`` positions (vLLM's block ids never reach them).
* Prefill: vLLM chunked prefill with ``max_num_batched_tokens == PREFILL_CHUNK`` hands one runtime chunk
  per step (start = number of computed tokens).
* Prefix caching = exact prefix resume (vllm_tt_plugin/prefix_resume.py): a prompt publishes a snapshot
  (sliding K/V tails; its full-attention blocks stay in vLLM's pool) at its last chunk boundary, and at a
  boundary where it leaves another cached prompt; a later prompt with that prefix gets those blocks and
  resumes there, running the same chunks as a full prefill. ``GEMMA_PREFIX_SNAPSHOTS`` (default 4,
  ~0.34 GB host each) bounds the snapshots kept.
* Decode: speculative steps (``--speculative-config {"method": "gemma4_assistant", ...}``) run one
  draft + verify cycle of SpecGreedyDecoder: greedy unpenalized requests the exact greedy cycle (commit the
  accepted drafts plus the target token), random or penalized requests the device sampler can process
  exactly (device_sampling.device_eligible) the lossless speculative-sampling cycle. Every other decode
  step (the last K tokens of a request, context end, ineligible parameters) runs the traced LogitsDecoder:
  eligible random / penalized requests sample its logits on device (exact vLLM sampler, one 16-byte
  record read back), greedy unpenalized and ineligible requests read the logits row and use vLLM's host
  sampler (greedy = argmax, the same token the on-device argmax picks). ``GEMMA_EXACT_SAMPLING=0`` turns the
  device sampler off (every random request on the host sampler, no speculation for it).
Text only: the class is not multimodal, so vLLM rejects image / audio inputs.
"""

import math
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.single_user import FULL, SLIDING, SingleUserRuntime

RING_TOKENS = int(os.environ.get("GEMMA4_RING_TOKENS", 18432))
PREFILL_CHUNK = int(os.environ.get("GEMMA4_PREFILL_CHUNK", 16384))
STEP_LOG_EVERY = 64
BLOCK_SIZE = 64
TOOLS_DIR = os.environ.get("GEMMA4_TOOLS_DIR", "/home/container_app_user/gemma4-tools")
_ASSISTANT_DTYPES = {
    "bf16": (ttnn.bfloat16, None),
    "bfp8": (ttnn.bfloat8_b, None),
    "bfp4": (ttnn.bfloat4_b, None),
    "bfp8-lm4": (ttnn.bfloat8_b, ttnn.bfloat4_b),
}


def _native_checkpoint():
    if TOOLS_DIR not in sys.path:
        sys.path.insert(0, TOOLS_DIR)
    import native_checkpoint

    return native_checkpoint


class Gemma4SingleUserForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": True,
        "exact_prefix_resume": True,  # the platform keeps prefix caching on despite the sliding window
        "supports_async_decode": False,
        "supports_sample_on_device": False,
        # Exact vLLM-equivalent device sampling (device_sampling.py): decode rows and sampled speculation.
        "supports_exact_device_sampling": True,
    }

    def __init__(self, vllm_config=None, prefix="", *, model, mesh_device, max_seq_len, assistant, draft_len, identity):
        # vLLM's model-class inspection requires a ``vllm_config`` keyword; the TT loader constructs the
        # instance through initialize_vllm_model.
        self.model = model
        self.mesh = mesh_device
        self.max_seq_len = max_seq_len
        self.assistant = assistant
        self.draft_len = draft_len
        self.identity = identity
        self.runtime = None
        self._prefill_warm = False  # (the runner resets ``already_warmed_up_prefill`` between its phases)
        self.spec_stats = {"steps": 0, "tokens": 0, "s": 0.0, "gap_s": 0.0, "end": None}

    # ---- vLLM model protocol (class inspection only: the TT runner calls prefill_forward /
    # decode_forward / speculative_step) ---------------------------------------------------------
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("TT bridge: embeddings run on device inside prefill_forward / decode_forward")

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError("TT bridge: the TT runner calls prefill_forward / decode_forward")

    def compute_logits(self, hidden_states, **kwargs):
        raise NotImplementedError("TT bridge: logits come from prefill_forward / decode_forward")

    # ---- vLLM configuration hooks ---------------------------------------------------------------
    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=None, max_num_seqs=1, **kwargs):
        return int(max_model_len) * int(max_num_seqs)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """One FullAttentionSpec per layer with the layer's own KV geometry (a single uniform-type
        group): vLLM budgets max_model_len tokens and hands out the full-attention block ids."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        text = getattr(vllm_config.model_config.hf_config, "text_config", vllm_config.model_config.hf_config)
        dtype = vllm_config.model_config.dtype
        block_size = vllm_config.cache_config.block_size
        full = (text.num_global_key_value_heads or text.num_key_value_heads, text.global_head_dim or text.head_dim)
        sliding = (text.num_key_value_heads, text.head_dim)
        return {
            f"model.layers.{i}.self_attn": FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=(sliding if lt == SLIDING else full)[0],
                head_size=(sliding if lt == SLIDING else full)[1],
                dtype=dtype,
            )
            for i, lt in enumerate(text.layer_types)
        }

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1,
                              optimizations=None):
        from models.demos.gemma4.tt.common import create_tt_model

        if max_batch_size != 1 or tt_data_parallel != 1 or mesh_device.get_num_devices() != 1:
            raise ValueError("gemma-4 single-user serving: one device, max_num_seqs 1, no data parallel")
        if max_seq_len % PREFILL_CHUNK and max_seq_len > PREFILL_CHUNK:
            logger.info(f"max_model_len {max_seq_len} is not a multiple of the {PREFILL_CHUNK}-token prefill chunk")
        nc = _native_checkpoint()
        native = os.environ["GEMMA4_NATIVE"]
        unproven = os.environ.get("GEMMA4_ALLOW_UNPROVEN") == "1"
        if unproven:
            logger.warning("GEMMA4_ALLOW_UNPROVEN=1: loading without equivalence-proof checks (development only)")
        model_path, state_dict, cache_path, identity = nc.prepare_native_runtime(native, allow_unproven=unproven)
        os.environ["TT_CACHE_PATH"] = cache_path
        os.environ["GEMMA4_PRECISION_PLAN"] = str(Path(native) / "precision_plan.json")
        t0 = time.time()
        _, model, _, state_dict = create_tt_model(
            mesh_device=mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            model_path=model_path,
            create_kv_cache=False,
            state_dict=state_dict,
        )
        del state_dict
        assistant, draft_len = None, 0
        assistant_dir = os.environ.get("GEMMA4_ASSISTANT_DIR")
        if assistant_dir:
            assistant, draft_len = cls._load_assistant(nc, model, mesh_device, assistant_dir, identity, unproven)
        logger.info(f"gemma-4 single-user model loaded in {time.time() - t0:.1f}s (drafter K={draft_len})")
        return cls(model=model, mesh_device=mesh_device, max_seq_len=max_seq_len, assistant=assistant,
                   draft_len=draft_len, identity=identity)

    @staticmethod
    def _load_assistant(nc, model, mesh_device, assistant_dir, identity, unproven):
        from models.demos.gemma4.tt.assistant.model import Gemma4AssistantModel
        from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs

        _, cache, drafter = nc.prepare_drafter(assistant_dir, identity, allow_unproven=unproven)
        draft_len = int(os.environ["GEMMA4_DRAFT_LEN"])
        if not unproven and draft_len != drafter["draft_len"]:
            raise ValueError(f"drafter proof covers K={drafter['draft_len']}, requested K={draft_len}")
        dtype, lm_head_dtype = _ASSISTANT_DTYPES[drafter["dtype"]]
        a_args = Gemma4AssistantArgs.from_hf_config(Gemma4AssistantArgs.load_hf_config(assistant_dir))
        a_args.model_cache_path = Path(cache)
        sd = Gemma4AssistantArgs.load_state_dict(assistant_dir, dummy_weights=False)
        assistant = Gemma4AssistantModel(
            mesh_device=mesh_device,
            assistant_args=a_args,
            target_model=model,
            state_dict=sd,
            ccl_manager=model.ccl_manager,
            dtype=dtype,
            lm_head_dtype=lm_head_dtype,
            tensor_cache_path=str(a_args.weight_cache_path(dtype)),
            mesh_config=model.mesh_config,
        )
        return assistant, draft_len

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Full-attention pools of vLLM's block count, sliding rings; builds the runtime."""
        text = self.model.hf_config
        full_blocks = {int(shape[0]) for (shape, _, _), lt in zip(per_layer_specs, text.layer_types) if lt == FULL}
        if len(full_blocks) != 1:
            raise ValueError(f"unexpected KV specs {per_layer_specs[:2]}")
        num_blocks = full_blocks.pop()
        if num_blocks * BLOCK_SIZE < self.max_seq_len or int(per_layer_specs[0][0][2]) != BLOCK_SIZE:
            raise ValueError(f"KV cache of {num_blocks} blocks x {per_layer_specs[0][0][2]} for {self.max_seq_len} tokens")
        from models.tt_transformers.tt.common import PagedAttentionConfig

        pac = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=num_blocks)
        kv = self.model.create_kv_caches(pac, 1, sliding_ring_tokens=RING_TOKENS)
        self.runtime = SingleUserRuntime(self.model, self.mesh, kv, self.max_seq_len, block_size=BLOCK_SIZE,
                                         ring_tokens=RING_TOKENS, prefill_chunk=PREFILL_CHUNK,
                                         assistant=self.assistant, draft_len=self.draft_len)
        if os.environ.get("GEMMA_EXACT_SAMPLING", "1") == "1":
            # Fixed-address sampler buffers, allocated before any decode trace is captured.
            from models.demos.gemma4.tt.device_sampling import DeviceSampler

            self.runtime.attach_sampler(DeviceSampler(self.mesh, self.model.vocab_size, max_rows=self.draft_len + 1,
                                                      max_drafts=self.draft_len))
        mem = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        logger.info(f"KV: full {num_blocks} blocks x {BLOCK_SIZE}, sliding ring {RING_TOKENS} tokens; "
                    f"DRAM allocated {mem.total_bytes_allocated_per_bank * mem.num_banks / 1e9:.2f} GB")
        return kv

    # ---- warmup ---------------------------------------------------------------------------------
    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, can_sample_on_device=False, **kwargs):
        """Compile every prefill program a request can hit: first chunks of each padded length and
        continuation chunks (>= 1024 rows) at a nonzero offset."""
        if self._prefill_warm:
            return
        rt = self.runtime
        rt.set_full_blocks(list(range(rt.n_blocks)))
        t0 = time.time()
        lengths = [n for n in (128, 1024, 2048, 4096, 8192, 16384) if n <= PREFILL_CHUNK]
        for n in lengths:
            rt.prefill_chunk_logits([2] * n, 0)
        resumable = self.max_seq_len > PREFILL_CHUNK
        if resumable:
            # Compile the resume round trip (tail upload + ring refill) before any trace is captured.
            rt.restore_prefix(rt.prefix_snapshot()[0], PREFILL_CHUNK)
            for n in lengths[1:]:
                rt.prefill_chunk_logits([2] * n, PREFILL_CHUNK)
        rt.release_prefill_state()
        from vllm_tt_plugin.prefix_resume import REGISTRY

        REGISTRY.configure(PREFILL_CHUNK, BLOCK_SIZE,
                           int(os.environ.get("GEMMA_PREFIX_SNAPSHOTS", "4")) if resumable else 0, fork_hints=True)
        ttnn.synchronize_device(self.mesh)
        self._prefill_warm = True
        logger.info(f"prefill warmup {time.time() - t0:.1f}s")

    def warmup_model_decode(self, kv_cache=None, enable_trace=False, **kwargs):
        """Capture the decode traces (speculative cycle, logits step) once, all persistent buffers
        allocated before any capture."""
        if not enable_trace or self.runtime.logits_decoder.trace_id is not None:
            return
        t0 = time.time()
        self.runtime.capture_traces(token=2, pos=16)
        logger.info(f"decode trace capture {time.time() - t0:.1f}s")

    # ---- execution ------------------------------------------------------------------------------
    def prefill_forward(self, tokens, page_table, kv_cache=None, prompt_lens=None, start_pos=None,
                        enable_trace=False, sampling_params=None, empty_slots=None, **kwargs):
        if tokens.shape[0] != 1 or sampling_params is not None:
            raise ValueError("single-user prefill with host sampling only")
        end = int(prompt_lens[0])
        start = int(start_pos[0]) if start_pos is not None else 0
        self._set_blocks(page_table[0], end)
        from vllm_tt_plugin.prefix_resume import REGISTRY

        rt = self.runtime
        request_ids, final_rows = kwargs.get("prefill_request_ids"), kwargs.get("prefill_final_rows")
        prefix = REGISTRY.active and request_ids is not None and final_rows is not None
        final = prefix and bool(final_rows[0])
        row = tokens[0, :end]
        if prefix and start:
            if REGISTRY.resuming(request_ids[0]):
                # The scheduler resumed this request at ``start`` with the snapshot's blocks.
                rt.restore_prefix(REGISTRY.take(row, start, request_id=request_ids[0]), start)
            elif ((final and start == end // PREFILL_CHUNK * PREFILL_CHUNK) or REGISTRY.wanted(row, start)) \
                    and not REGISTRY.has(row, start):
                REGISTRY.put(row, start, page_table[0], *rt.prefix_snapshot())
        logits = rt.prefill_chunk_logits(tokens[0, start:end].tolist(), start)
        if final and end % PREFILL_CHUNK == 0:
            REGISTRY.put(row, end, page_table[0], *rt.prefix_snapshot())
        return logits.reshape(1, 1, -1)

    def _set_blocks(self, block_row, n_tokens):
        """Full-attention page table = the request's blocks holding positions < ``n_tokens`` (all
        allocated by the scheduler), every later entry vLLM's null block 0. Entries past the allocation
        are stale ids of earlier requests (vLLM frees blocks in reverse, so they can alias this
        request's own blocks) and prefill padding / speculative rows past ``n_tokens`` would write
        through them; block 0 is never allocated to a request, so those writes are harmless."""
        n = -(-n_tokens // self.runtime.block_size)
        self.runtime.set_full_blocks(block_row[:n])

    # ---- exact device sampling (device_sampling.py; queried by the runner right after load) ---------------
    def exact_sampling_available(self):
        return os.environ.get("GEMMA_EXACT_SAMPLING", "1") == "1"

    def sampled_speculation_available(self):
        """Sampled speculation runs the split (per-step drafter / per-P verify) traces, which need the
        one-launch batched verify for every row count."""
        from models.demos.gemma4.tt.attention.decode import verify_sdpa_rows

        return (self.exact_sampling_available() and self.assistant is not None and self.draft_len > 1
                and verify_sdpa_rows() == self.draft_len + 1)

    def decode_forward(self, tokens, start_pos, page_table=None, kv_cache=None, enable_trace=True,
                       read_from_device=True, sampling_params=None, exact_sampling=None, **kwargs):
        """Logits row of the step ([1, 1, vocab], host sampling), or with ``exact_sampling`` (the request's
        device_sampling.SamplingRequest) the next token drawn on device ([1] int32)."""
        if sampling_params is not None:
            raise ValueError("generic on-device sampling is not supported; decode returns logits or exact samples")
        self._set_blocks(page_table[0], int(start_pos[0]) + 1)
        token, pos = int(tokens[0, 0]), int(start_pos[0])
        if exact_sampling is not None:
            if len(exact_sampling) != 1:
                raise ValueError("single-user exact sampling takes one request")
            return torch.tensor([self.runtime.decode_sample(token, pos, exact_sampling[0])], dtype=torch.int32)
        return self.runtime.decode_logits(token, pos).reshape(1, 1, -1)

    def speculative_step(self, token, position, block_row, request=None):
        """One speculative cycle for ``token`` at ``position``: exact greedy, or with ``request`` (a
        device_sampling.SamplingRequest) lossless speculative sampling; returns the committed tokens
        (positions position+1 ..)."""
        rt = self.runtime
        t0 = time.perf_counter()
        st = self.spec_stats
        if st["end"] is not None:
            st["gap_s"] += t0 - st["end"]  # host time between steps (scheduler, detokenizer, streaming)
        self._set_blocks(block_row, position + self.draft_len + 1)
        committed = rt.spec_step(token, position) if request is None else rt.spec_step_sampled(token, position, request)[0]
        st["end"] = time.perf_counter()
        st["steps"] += 1
        st["tokens"] += len(committed)
        st["s"] += st["end"] - t0
        if st["steps"] % STEP_LOG_EVERY == 0:
            n = STEP_LOG_EVERY
            logger.info(f"speculative pos={position} last {n} steps: tokens/step={st['tokens'] / n:.2f} "
                        f"drafts/step={rt.spec.drafted / n:.2f} "
                        f"step_ms={1000 * st['s'] / n:.1f} host_gap_ms={1000 * st['gap_s'] / n:.1f}")
            rt.spec.drafted = 0
            st.update(tokens=0, s=0.0, gap_s=0.0)
        return committed
