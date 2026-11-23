# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-user gemma-4 runtime: KV page tables (ring-buffered sliding-window caches), prefill up to
the model's full context in chunks, and the traced greedy / speculative / logits decoders.

Sliding-window ring. Sliding layers only ever read the last ``sliding_window`` (W) positions, so their
cache pool holds ``ring_tokens`` positions and position ``p`` lives in slot ``p % ring_tokens``:
decode / verify KV writes and SDPA decode get ``cache_position_modulo=ring_tokens`` with the identity
page table [1, ring_blocks] (Gemma4AttentionConfig.ring_tokens; the drafter's reads of the target's
sliding cache too), and prefill fills each chunk through the ring-aliased table ``b -> b % ring_blocks``.
The wrap only changes physical addresses: SDPA decode walks the same absolute chunks with the same core
split and mask as with an unbounded cache. The ring is exact as long as the slots a step reads or
writes never alias each other:
  * decode / verify at p..p+K read [floor((p-W+1)/64)*64, p+K] (< W + 64 + K + 64 positions);
  * a prefill chunk [s, s+L) is written in one fill and must not overwrite the previous chunk's last
    window before decode reads it: ring_tokens >= prefill_chunk + W + 1024 covers both.
Below ``ring_tokens`` positions the wrap is the identity, i.e. the unbounded layout.

Exact prefix resume (vLLM prefix caching, vllm_tt_plugin/prefix_resume.py). Between chunks a prefill
carries only each sliding layer's K/V tail (the last window of the previous chunk; full-attention layers
read earlier chunks from their paged blocks). ``prefix_snapshot`` copies those tails to host at a chunk
boundary; ``restore_prefix`` puts them back and refills the ring slots of the window below the boundary
from them (the same fill op and values the full prefill wrote there), so a prefill resumed at the
boundary runs the identical chunk programs on identical inputs.

Prefill runs in chunks of ``prefill_chunk`` tokens (multiple of 2048, < 32768). Every chunk -- also a
single-chunk prompt -- runs the generator-level chunked path: full-attention layers read earlier
chunks from the paged cache (chunked SDPA with the chunk's absolute offset); sliding layers attend
[previous chunk's last window | chunk] from the carried K/V tail. The first chunk runs exactly the ops
of a single-shot prefill (plus saving the tail), so prompts up to one chunk are bit-identical to it.
"""

import math

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.attention.decode import verify_sdpa_rows
from models.demos.gemma4.tt.greedy_decode import GreedyDecoder, LogitsDecoder
from models.tt_transformers.tt.common import get_padded_prefill_len

FULL = "full_attention"
SLIDING = "sliding_attention"


class SingleUserRuntime:
    def __init__(self, model, mesh_device, kv_cache, max_seq_len, block_size=64, ring_tokens=None,
                 prefill_chunk=16384, assistant=None, draft_len=5):
        window = model.hf_config.sliding_window
        if prefill_chunk % 2048 or not 2048 <= prefill_chunk < 32768:
            raise ValueError(f"prefill_chunk {prefill_chunk}: multiple of 2048 below 32768")
        if ring_tokens is not None and (ring_tokens % block_size or ring_tokens < prefill_chunk + window + 1024):
            raise ValueError(f"ring_tokens {ring_tokens}: multiple of {block_size}, >= prefill_chunk + {window + 1024}")
        self.model = model
        self.mesh = mesh_device
        self.kv_cache = kv_cache
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.ring_tokens = ring_tokens
        self.prefill_chunk = prefill_chunk
        self.n_blocks = math.ceil(max_seq_len / block_size)
        self.ring_blocks = ring_tokens // block_size if ring_tokens else self.n_blocks
        blocks = torch.arange(self.n_blocks, dtype=torch.int32).reshape(1, -1)
        # Host tables per layer type in logical block order (prefill chunk tables are slices of them);
        # decode / verify use the device tables: full = the full row, sliding = the ring's identity.
        self.tables = {FULL: blocks.clone(), SLIDING: blocks % self.ring_blocks}
        self.page_tables = {FULL: self._dev(self.tables[FULL]), SLIDING: self._dev(blocks[:, : self.ring_blocks])}
        self.greedy = GreedyDecoder(model, mesh_device, kv_cache, self.page_tables)
        self.logits_decoder = LogitsDecoder(model, mesh_device, kv_cache, self.page_tables)
        self.spec = None
        self.page_tables_rep = None
        self.sampler = None  # device_sampling.DeviceSampler (attach_sampler)
        if assistant is not None:
            self.attach_assistant(assistant, draft_len)
        self._tails_live = False
        self._trace_programs = None  # program-cache size when the live traces were captured

    def attach_assistant(self, assistant, draft_len):
        """Exact speculative decoding with the it-assistant drafter (SpecGreedyDecoder); the drafter's
        sliding layers read the target's ring."""
        from models.demos.gemma4.tt.spec_greedy import SpecGreedyDecoder

        for lt, layer in zip(assistant.layer_types, assistant.layers):
            if lt == SLIDING:
                layer.self_attn.config.ring_tokens = self.ring_tokens
        rep = None
        if verify_sdpa_rows() == draft_len + 1:
            # One-launch verify SDPA: the page tables replicated to one row per verify row.
            rep = {k: self._dev(self._device_row(k).repeat(draft_len + 1, 1)) for k in self.page_tables}
        self.page_tables_rep = rep
        self.spec = SpecGreedyDecoder(self.model, assistant, self.mesh, self.kv_cache, self.page_tables, draft_len,
                                      page_tables_rep=rep)
        return self.spec

    def _device_row(self, layer_type):
        """Host copy of the decode-time device table of ``layer_type`` ([1, blocks])."""
        return self.tables[FULL] if layer_type == FULL else torch.arange(self.ring_blocks, dtype=torch.int32).reshape(1, -1)

    def _dev(self, table):
        return ttnn.from_torch(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh)

    # ---- page tables ---------------------------------------------------------------------------
    def set_full_blocks(self, block_ids):
        """Point the full-attention page table at an external block list (vLLM block table for the
        user); updated in place, so captured traces see it on their next replay."""
        row = torch.zeros(1, self.n_blocks, dtype=torch.int32)
        n = min(len(block_ids), self.n_blocks)
        row[0, :n] = torch.as_tensor(block_ids[:n], dtype=torch.int32)
        if torch.equal(row, self.tables[FULL]):
            return
        self.tables[FULL] = row
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self.page_tables[FULL]
        )
        if self.page_tables_rep is not None:
            rep = row.repeat(self.page_tables_rep[FULL].shape[0], 1)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(rep, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self.page_tables_rep[FULL]
            )

    # ---- prefill --------------------------------------------------------------------------------
    def prefill_chunk_logits(self, tokens, start):
        """Prefill ``tokens`` (list of ids) at absolute positions [start, start+len). ``start`` must be
        0 or a multiple of ``prefill_chunk`` following the previous chunk of the same prompt, and the
        chunk at most ``prefill_chunk`` long. Returns the float32 logits of the chunk's last token."""
        n = len(tokens)
        if not 0 < n <= self.prefill_chunk or start % self.prefill_chunk:
            raise ValueError(f"prefill chunk of {n} tokens at {start}: chunks of <= {self.prefill_chunk} at multiples of it")
        if self.spec is not None:
            self.spec.reset()  # a staged cycle belongs to the KV this prefill rewrites
        # A chunk after the first attends [tail | chunk] on sliding layers; SDPA needs 1024 + L to split
        # into its 256-row Q chunks, hence the 1024 floor.
        padded = get_padded_prefill_len(n) if start == 0 else max(1024, get_padded_prefill_len(n))
        if start + padded > self.n_blocks * self.block_size:
            raise ValueError(f"prefill of {start + n} tokens exceeds the {self.max_seq_len}-token context")
        ids = torch.zeros(1, padded, dtype=torch.long)
        ids[0, :n] = torch.as_tensor(tokens, dtype=torch.long)
        b0, b1 = start // self.block_size, (start + padded) // self.block_size
        # Full layers read earlier chunks through ``page_table``; sliding layers only fill (their
        # attention uses the carried tail), so their ``page_table`` entry is unused.
        page_table = {FULL: self._dev(self.tables[FULL][:, :b1]), SLIDING: self.page_tables[SLIDING]}
        chunk_table = {k: self._dev(t[:, b0:b1]) for k, t in self.tables.items()}
        x, *_ = self.model.prepare_inputs_prefill(ids, page_table=None)
        last = n - 1
        logits = self.model.ttnn_prefill_forward(
            x,
            user_id=0,
            page_table=page_table,
            chunk_page_table=chunk_table,
            chunk_start_idx=start,
            get_last_token=(last // 32) * 32,
            kv_cache=self.kv_cache,
            batch_size=1,
        )
        self._tails_live = True
        out = self.model.process_output_prefill(logits, last % 32).reshape(-1).float()
        logits.deallocate(True)
        for t in [page_table[FULL]] + list(chunk_table.values()):
            t.deallocate(True)
        return out

    def prefill(self, prompt_ids):
        """Whole-prompt prefill; returns the float32 logits of the last prompt token."""
        C = self.prefill_chunk
        for s in range(0, len(prompt_ids), C):
            logits = self.prefill_chunk_logits(prompt_ids[s : s + C], s)
        self.release_prefill_state()
        return logits

    def release_prefill_state(self):
        """Free the sliding-window K/V tails carried between prefill chunks."""
        if self._tails_live:
            for layer in self.model.layers:
                layer.self_attn._release_sliding_prefill_tail()
            self._tails_live = False

    # ---- exact prefix resume ----------------------------------------------------------------------
    def prefix_snapshot(self):
        """(host payload, bytes): every layer's live sliding K/V tail, in layer order (None for
        layers without one). Taken at a chunk boundary, between two chunks of a prefill."""
        if not self._tails_live:
            raise RuntimeError("prefix snapshot needs the tails of a prefill in progress")
        tails, nbytes = [], 0
        for layer in self.model.layers:
            tail = getattr(layer.self_attn, "_sliding_prefill_tail", None)
            if tail is None:
                tails.append(None)
                continue
            host = tuple((ttnn.to_torch(t), t.dtype) for t in tail)
            nbytes += sum(h.numel() * h.element_size() for h, _ in host)
            tails.append(host)
        return {"tails": tails}, nbytes

    def restore_prefix(self, saved, start):
        """Resume a prefill at chunk boundary ``start`` from a prefix_snapshot taken there: the sliding
        tails, and the ring slots of the positions [start - tail, start) (decode / verify read them
        while the prompt ends within a window of ``start``), refilled from the tails through the
        ring-aliased table exactly like the chunk fill that first wrote them."""
        from models.demos.gemma4.tt.attention.operations import effective_block_size

        if start <= 0 or start % self.prefill_chunk or len(saved["tails"]) != len(self.model.layers):
            raise ValueError(f"cannot resume a prefill at {start} from this snapshot")
        self.release_prefill_state()
        ring_table = None
        for i, (layer, tail) in enumerate(zip(self.model.layers, saved["tails"])):
            if tail is None:
                continue
            k, v = (
                ttnn.from_torch(h, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for h, dt in tail
            )
            attn = layer.self_attn
            attn._sliding_prefill_tail = (k, v)
            if attn.config.ring_tokens is None or i in self.model.kv_shared_layer_map:
                continue
            rows = int(k.shape[-2])
            if ring_table is None:
                b1 = start // self.block_size
                ring_table = self._dev(self.tables[SLIDING][:, b1 - rows // self.block_size : b1].contiguous())
            k_cache, v_cache = attn.kv_cache
            nkv = 1 if attn.weights.kv_replicated else attn.config.num_key_value_heads
            eff_bs = effective_block_size(k_cache, attn.config.head_dim, nkv)
            ttnn.experimental.paged_fill_cache(k_cache, k, ring_table, batch_idx=0, block_size=eff_bs)
            ttnn.experimental.paged_fill_cache(v_cache, v, ring_table, batch_idx=0, block_size=eff_bs)
        if ring_table is not None:
            ring_table.deallocate(True)
        self._tails_live = True

    # ---- decode ----------------------------------------------------------------------------------
    # Trace safety. A trace replays into the scratch addresses its intermediates had at capture;
    # they are free between replays, so anything allocated after a capture and alive across a
    # replay may be overwritten -- notably the DRAM kernel binaries of programs compiled later
    # (a prefill of a new shape, another decoder's first run). Corrupted binaries hang the device.
    # Hence every decoder use goes through ``_decoder``, which re-captures all live traces when the
    # program cache grew since they were captured, and a capture pass that compiled anything is
    # repeated, so each capture comes after the compile runs of all decoders captured with it.
    def _decoder(self, decoder, token, pos):
        self.release_prefill_state()
        live = [d for d in self._decoders() if d.trace_id is not None]
        if decoder in live and self.mesh.num_program_cache_entries() == self._trace_programs:
            return decoder
        self._capture(live + ([] if decoder in live else [decoder]), token, pos)
        return decoder

    def _decoders(self):
        return [self.greedy, self.logits_decoder] + ([self.spec] if self.spec is not None else [])

    def _capture(self, decoders, token, pos):
        """(Re-)capture ``decoders`` at ``token``/``pos``. Each capture runs one eager compile step and
        one captured step there: they write the KV rows of ``pos`` (and the speculative rows after
        it), which the following real step rewrites with the same values."""
        logger.info(f"decode trace (re)capture at pos {pos}: {[type(d).__name__ for d in decoders]}")
        if self.sampler is not None:
            self.sampler.set_params({})  # inactive rows: the sampler add-ons run as no-ops while capturing
        for _ in range(2):
            before = self.mesh.num_program_cache_entries()
            for d in self._decoders():
                d.release()
            for d in decoders:
                d.capture(token, pos)
            self._trace_programs = self.mesh.num_program_cache_entries()
            if self._trace_programs == before:
                return
        raise RuntimeError("decode trace capture keeps compiling new programs")

    def greedy_generate(self, token, pos, max_new, stop_ids=()):
        return self._decoder(self.greedy, token, pos).generate(token, pos, max_new, stop_ids=stop_ids)

    def spec_generate(self, token, pos, max_new, stop_ids=()):
        return self._decoder(self.spec, token, pos).generate(token, pos, max_new, stop_ids=stop_ids)

    def decode_logits(self, token, pos):
        return self._decoder(self.logits_decoder, token, pos).step(token, pos)

    def spec_step(self, token, pos):
        """One exact speculative cycle (serving); returns the committed tokens."""
        return self._decoder(self.spec, token, pos).step(token, pos)

    def capture_traces(self, token, pos):
        """Capture the serving traces (logits step, speculative cycle) at ``token``/``pos`` (KV rows
        there are overwritten later)."""
        self._capture([d for d in (self.logits_decoder, self.spec) if d is not None], token, pos)

    # ---- exact device sampling (device_sampling.py) ------------------------------------------------
    def attach_sampler(self, sampler):
        """Exact device sampler: the logits decoder gets a sampling add-on trace and the speculative decoder its
        sampled cycle (split traces + add-ons). Attach before any decode trace is captured."""
        if any(d.trace_id is not None for d in self._decoders()):
            raise RuntimeError("attach the device sampler before any decode trace is captured")
        self.sampler = sampler
        self.logits_decoder.sampler = sampler
        if self.spec is not None:
            self.spec.sampler = sampler

    def _load_request(self, request):
        """History = the request's output counts (+ prompt flags for repetition), params = its rows."""
        s = self.sampler
        s.ensure_history(request.owner, request.outputs, request.prompt or (), request.spec.uses_prompt)
        s.set_request(request.spec)

    def decode_sample(self, token, pos, request):
        """One device-sampled decode step for ``token`` at ``pos`` (device_sampling.SamplingRequest); returns the
        token at pos + 1. A truncated top-k tie group is drawn on the host from the step's logits."""
        from models.demos.gemma4.tt.device_sampling import R_INEXACT, host_sample

        dec = self._decoder(self.logits_decoder, token, pos)
        self._load_request(request)
        rec = dec.sample(token, pos)
        if int(rec[1]) & R_INEXACT:
            logits = ttnn.to_torch(dec.out).reshape(-1)[: self.model.vocab_size].float()
            tok = host_sample(logits, request.spec, request.outputs, request.prompt or (), pos + 1)
            self.sampler.invalidate()
            logger.info(f"exact sampling: host draw at position {pos + 1} (truncated top-k tie group)")
            return tok
        self.sampler.note_committed(request.owner, len(request.outputs) + 1)
        return int(rec[0])

    def spec_step_sampled(self, token, pos, request):
        """One lossless speculative-sampling cycle (serving); returns (committed tokens, drafts verified)."""
        from models.demos.gemma4.tt.device_sampling import V_HOST, host_sample

        spec = self._decoder(self.spec, token, pos)
        self._load_request(request)

        def host_draw(row, position, before):
            logger.info(f"exact sampling: host draw at position {position} (truncated top-k tie group)")
            return host_sample(row.float(), request.spec, list(request.outputs) + before, request.prompt or (), position)

        committed, _, k, flags = spec.step_sampled(token, pos, host_draw)
        if flags == V_HOST:
            self.sampler.invalidate()
        else:
            self.sampler.note_committed(request.owner, len(request.outputs) + len(committed))
        return committed, k
