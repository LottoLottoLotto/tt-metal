# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini (Inkling2LoopForCausalLM, text) on one P150: embedding + embed_norm, 48 decoder-layer applications
(layers 0-5, the loop layers 6-17 three times, layers 18-23), final block attention residual, norm, unembedding (BFP8)
and the logit softcap (30).

Orchestration = Inkling2LoopTextModel.forward:
  * block states (normed, FP32) are collected at the boundary layers 0, 6, 12, 18 (the layer INPUT is appended);
  * passes 2 and 3 over the loop layers restart from the pre-loop hidden / block states, and every loop layer of those
    passes mixes message_input messages (queries over [block state appended at layer 12 by the previous pass,
    previous pass's last hidden]: the reference's `block_states[..., num_pre_loop_blocks + 1:]` + hidden) into its
    q / k / v / MLP inputs through the message gates;
  * every application has its own KV cache and conv states (pass slot `pass_idx` of layers 6-17).

Entry points (batch 1, one sequence):
  prefill(token_ids, start_pos=0) -> torch float32 [unpadded_vocab] logits of the last token; chunked by
      prefill_schedule (prefill_chunk rows, then multiples of PREFILL_TAIL, then a last chunk padded to a multiple of
      32); start_pos > 0 continues the current sequence (after a prefill or decode steps that reached exactly
      start_pos).  prefill_logits(token_ids, positions, start_pos=0) -> logits [len(positions), unpadded_vocab] at the
      given absolute positions (teacher-forced evaluation).
  decode(emb_tt, cur_pos_tt, ids=None) -> device FP32 logits [1, 1, 1, padded_unembed] (device-only, trace-capturable;
      columns >= unpadded_vocab read -30); the LM head (fast/head.py FastHead) also writes the greedy token to word 0
      of ids (default self.ids).  emb_tt: BF16 [1, 1, 32, H] tile with the token's embedding row in row 0 (the
      embedding table lives on the host: decode_inputs_host(token));
      cur_pos_tt: [1] int32 row-major device tensor = `attn_tables.cur_pos`.  Before each call
      `set_decode_pos(pos)` selects the bucket and writes its attention decode tables (cur_pos included).
  decode_step(token, pos) -> torch logits (untraced); capture_decode_trace() then decode_traced(token, pos) (writes the
      token and the tables, replays the bucket's trace, returns its device logits; logits_to_torch() reads them).
  warmup_prefill() compiles every first / second prefill chunk shape; reset() starts a new sequence.
Batched entry points (max_num_seqs S > 1 or kv_blocks: tt/NOTES-batch.md): prefill(ids, start_pos, slot=s,
  page_row=pages) prefills one sequence chunk into slot s through its page row; reset_slot(s) starts a new sequence in
  slot s; batch_decode(e_rows, bt, R, ids) is the device-only step of R rows over the batch tables `bt`
  (fast/batch_tables.py, allocated here); fast/batch_decode.py BatchDecoder traces and drives it.
Prefix caching (prefix_snapshots = M > 0; NOTES-prefix.md, tt/prefix_cache.py): snapshot_slot(s, j) / restore_slot(j, s)
  copy a slot's running state (sliding K / V rings, K / V conv-input rings, glue conv rings) to / from a device pool of
  M snapshots; prefill(at=...) reports the chunk boundaries a full prefill crosses.

Decode glue: by default the 48 applications' glue (embed_norm, block / message residuals, the loop messages, gates,
norms, convs, residuals, final mix + norm) runs as the fused programs of tt/fast/glue.py (FastGlue, ~2 launches per
application around the attention / MoE modules; decode() hands it the raw embedding row; its conv states are
rows_glue's position rings); INK_GLUE=ttnn selects the ttnn glue path below (_forward, decode mode) for A/B.

Prefill / verify glue: with the fused glue, prefill chunks and spec-verify rows (mode "verify") run the glue of all 48
applications as the tokens-as-rows programs of tt/fast/glue_rows.py (RowsGlue, ~2 launches per application + a gate
matmul / GATE program on the message-gated ones); rows_glue owns the post-attention / post-MLP conv states as
position-indexed rings (row pos % 32), so a verify's rows are committed by advancing the position.  INK_GLUE=ttnn: the
ttnn ListStates path (_forward) and ShortConv states for everything.

Attention decode: by default the fused kernels of tt/fast/attn_decode.py (AttnDecodeFast, 4 launches per application;
INK_ATTN=stock: the ttnn path of attention.py for A/B).  The fused path reads only cur_pos / tau and its cost follows
the position, so the model then keeps ONE bucket (= max_seq_len) unless `buckets` is given, and its tables skip the
ring_idx / full_idx rows.  Stock path, full-attention decode buckets: the full layers' decode reads the first
`full_len` cache positions (SDPA decode over full_len keys, mask gathered for full_len columns), so its cost grows with
full_len, not with the position.  The model keeps one AttnDecodeTables per bucket (`buckets`, default powers of two
from 4096 up to max_seq_len) and one decode trace per bucket; set_decode_pos / decode_step / decode_traced pick the
smallest bucket > pos, so a sequence moves to the next bucket's trace by itself (all buckets share the caches).

Trace safety: a replay writes into the DRAM / L1 buffers the capture's intermediates had; any device buffer allocated
AFTER a capture can sit there and is overwritten by every replay.  So (1) every persistent device tensor (weights,
caches, conv states, the decode tables of every bucket, the prefill kernels' tau / shifted-table buffers, the
trace input) is allocated in the constructor; (2) capture_decode_trace compiles every bucket before capturing any;
(3) prefill (compiled lazily, possibly after the capture) uses no op that keeps device state with its cached program
(tile-layout ttnn.reshape builds and caches a page-mapping tensor);
prefill / decode allocate only tensors that die within the call.  Program binaries are allocated top-down in DRAM,
away from the bottom-up activations.  tests/test_serving.py checks that no bottom-up DRAM block outlives a sequence
after the capture.
"""
from __future__ import annotations

import contextlib
import os

import torch
import ttnn

from .common import CK_HIFI4, DRAM, DTYPES, WeightCache, free
from .precision import attn_weight_map, expert_map, expert_source
from .config import HF_DIR, Inkling2Config
from .kvplan import DEFAULT_ATTN_PRECISION, MAX_PREFILL_CHUNK, PREFILL_TAIL, decode_buckets, prefill_chunk_for
from .layers import (BF16, F32, BlockResidual, Embedding, ListStates, MessageGate, MessageResidual, RMSNorm,
                     RowConsts, RowStates, ShortConv, unorm)

DECODE_ROWS = 32
LOGIT_ROWS = 32  # prefill_logits: rows per LM head call (FastHead: one tile row)
GATE_NAMES = (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp"))  # message kind -> checkpoint controller name
MAX_SEQS = 16  # batched serving: sequence slots (MoE rows / LM head ids per decode step <= 16)


class LayerGlue:
    """Per-layer glue weights and state of one decoder layer (all passes)."""

    def __init__(self, device, cfg: Inkling2Config, src, layer: int):
        p = f"model.llm.layers.{layer}"
        t = src.tensor
        self.layer = layer
        self.full = cfg.is_full(layer)
        self.loop = cfg.is_loop(layer)
        self.boundary = cfg.is_boundary(layer)
        # attn_res.attn_queries.{i} belongs to layer i + 1 (layer 0 has no attention_input)
        self.attn_res = BlockResidual(device, t(f"model.llm.attn_res.attn_queries.{layer - 1}")) if layer else None
        self.mlp_res = BlockResidual(device, t(f"model.llm.attn_res.mlp_queries.{layer}"))
        if self.full:
            self.qkv_norms = [RMSNorm(device, t(f"{p}.{k}_input_norm.weight"), cfg.eps) for k in "qkv"]
        else:
            self.in_norm = RMSNorm(device, t(f"{p}.attn_norm.weight"), cfg.eps)
        self.post_attn_norm = RMSNorm(device, t(f"{p}.mlp_norm.weight"), cfg.eps)
        passes = cfg.passes(layer)
        self.attn_conv = ShortConv(device, t(f"{p}.attn_sconv.weight"), passes, cfg.conv_kernel)
        self.mlp_conv = ShortConv(device, t(f"{p}.mlp_sconv.weight"), passes, cfg.conv_kernel)
        self.gates = {}
        if self.loop:
            for kind, ck in GATE_NAMES:
                c = f"{p}.controllers.{ck}"
                self.gates[kind] = MessageGate(device, t(f"{c}.down_weight"), t(f"{c}.forget_weight"),
                                               t(f"{c}.gain_weight"))

    def reset(self):
        self.attn_conv.reset()
        self.mlp_conv.reset()


class Messages:
    """The messages of one pass over the loop layers; get(j, kind) -> FP32 message (a new tensor, caller frees).
    prefill (ListStates): s0 = the previous pass's block state from layer 12, elementwise FP32 mixing; gm() is None.
    decode (RowStates): row-stacked, all 48 messages in one [1,1,64,H] tensor, get() slices one row ([1,1,1,H]);
    the message halves of all 48 message gates' down projections in one matmul (wdm_all [H, 48 * 64]), gm() slices
    the gate's [1, 64]."""

    def __init__(self, mr: MessageResidual, states, n_pre: int, h_prev, wdm_all=None):
        self.mr = mr
        s1 = unorm(h_prev, dtype=BF16 if isinstance(states, RowStates) else F32)
        if isinstance(states, RowStates):
            # +1 skips the state the loop's first layer appended (= norm(pre-loop hidden), the same every pass)
            if states.n != n_pre + 2:
                raise RuntimeError(f"expected one previous-pass block state, got {states.n - n_pre - 1}")
            self.M = mr.prepare_rows(states.X, n_pre + 1, s1, states.c)
            nM = unorm(self.M, dtype=BF16)
            self.GM = ttnn.matmul(nM, wdm_all, dtype=F32, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
            ttnn.deallocate(nM)
            self.rows = True
        else:
            prev = states.items[n_pre + 1:]
            if len(prev) != 1:
                raise RuntimeError(f"expected one previous-pass block state, got {len(prev)}")
            self.s1 = s1
            self.w0, self.d = mr.prepare(prev[0], s1)
            self.rows = False

    def get(self, j: int, kind: str):
        k = MessageResidual.KINDS.index(kind)
        if self.rows:
            return self.mr.message_row(self.M, j, k)
        return MessageResidual.message(self.w0, self.d, self.s1, j, k)

    def gm(self, j: int, kind: str):
        if not self.rows:
            return None
        c = 4 * j + MessageResidual.KINDS.index(kind)
        r = self.GM.shape[-1] // (4 * self.mr.n_loop)
        return ttnn.slice(self.GM, [0, 0, c, c * r], [1, 1, c + 1, (c + 1) * r], memory_config=DRAM)

    def free(self):
        if self.rows:
            free(self.M, self.GM)
        else:
            free(self.w0, self.d, self.s1)


class Inkling2Model:
    """precision: expert weight format on the device ("bfp4:gptq" = the served default: GPTQ-rounded BFP4 experts | "bfp8"
    (quality profile) | "bfp4" | "mixed:<json>" per-layer map, see precision.py); attn_precision: attention projection
    weight map (precision.attn_weight_map; None = env INK_ATTN_PRECISION, else kvplan.DEFAULT_ATTN_PRECISION: q|gate
    GPTQ-BFP4 (variant gptq under INK_VARIANT_ROOT), k / v / o BFP8; "bfp8" = all BFP8); attn_policy: attention
    dtype policy dict (attention.DEFAULT_POLICY keys; None = its defaults); max_seq_len: bound of every position (full
    caches; multiple of 64); buckets: full-attention decode buckets (multiples of 64, the last = max_seq_len; default
    decode_buckets(max_seq_len)); prefill_chunk: largest prefill chunk (None = kvplan.prefill_chunk_for(max_seq_len),
    the served profile's: 4096 rows clipped to max_seq_len).
    Batched serving (NOTES-batch.md): max_num_seqs = S sequence slots (1..MAX_SEQS; every per-sequence state tensor gets
    a leading slot dimension), kv_blocks = NB blocks of the shared full-attention pool (vLLM's block ids; block NB =
    the zero block).  max_num_seqs 1 with kv_blocks None is the single-stream runtime (identity block order).
    prefix_snapshots = M > 0: a device pool of M snapshots of a sequence slot's running state (tt/prefix_cache.py,
    NOTES-prefix.md: exact prefix resume; restore_slot, prefill(at=...))."""

    def __init__(self, mesh_device, hf_dir: str = HF_DIR, *, max_seq_len: int = 8192, precision: str = "bfp4:gptq",
                 attn_policy: dict | None = None, buckets: list[int] | None = None,
                 prefill_chunk: int | None = None, source=None, use_cache: bool = True, moe_balance: bool = True,
                 attn_precision: str | None = None, max_num_seqs: int = 1, kv_blocks: int | None = None,
                 prefix_snapshots: int = 0, log=print):
        if max_seq_len % 64:
            raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of 64")
        if not 1 <= max_num_seqs <= MAX_SEQS:
            raise ValueError(f"max_num_seqs {max_num_seqs}: 1..{MAX_SEQS}")
        self.max_num_seqs, self.kv_blocks = max_num_seqs, kv_blocks
        self.batched = max_num_seqs > 1 or kv_blocks is not None  # slot-major state + the paged full-attention pool
        self.fast_attn = os.environ.get("INK_ATTN", "fast") != "stock"  # fused attention decode (module docstring)
        self.buckets = sorted(set(buckets or ([max_seq_len] if self.fast_attn else decode_buckets(max_seq_len))))
        if self.buckets[-1] != max_seq_len or any(b % 64 or b <= 0 for b in self.buckets):
            raise ValueError(f"buckets {self.buckets}: multiples of 64, the largest = max_seq_len {max_seq_len}")
        prefill_chunk = prefill_chunk or prefill_chunk_for(max_seq_len)
        if prefill_chunk % 32 or not 32 <= prefill_chunk <= MAX_PREFILL_CHUNK or (prefill_chunk > PREFILL_TAIL
                                                                                   and prefill_chunk % PREFILL_TAIL):
            raise ValueError(f"prefill_chunk {prefill_chunk}: a multiple of 32, of {PREFILL_TAIL} above it, at most "
                             f"{MAX_PREFILL_CHUNK}")
        self.prefill_chunk = prefill_chunk
        self.device = mesh_device
        self.cfg = cfg = Inkling2Config.from_hf(hf_dir)
        if source is None:
            from .source import CheckpointSource

            source = CheckpointSource(hf_dir)
        self.src = source
        self.max_seq_len = max_seq_len
        self.precision = precision
        self.moe_balance = moe_balance  # Inkling2MoE(balance=...): kvplan.Plan.moe_balance (+0.95 GiB BFP4 experts)
        self.expert_prec = expert_map(precision, cfg.n_layers)
        self.attn_policy = attn_policy
        self.attn_precision = attn_precision or os.environ.get("INK_ATTN_PRECISION") or DEFAULT_ATTN_PRECISION
        self.attn_wprec = attn_weight_map(self.attn_precision, cfg.n_layers)
        self.cache = WeightCache(mesh_device, enabled=use_cache)
        t = source.tensor
        self.embed = Embedding(mesh_device, t("model.llm.embed.weight"), t("model.llm.embed_norm.weight"), cfg.eps)
        self.final_res = BlockResidual(mesh_device, t("model.llm.attn_res.output_query"))
        self.norm = RMSNorm(mesh_device, t("model.llm.norm.weight"), cfg.eps)
        self.msg = MessageResidual(mesh_device, t("model.llm.middle_cycle_cross_pass_queries"),
                                   t("model.llm.middle_cycle_cross_pass_kv_queries"))
        self.row_consts = RowConsts(mesh_device)
        # LM head: the fused streamed head + softcap (+ greedy argmax) of fast/head.py, bitwise the stock
        # layers.LMHead (BFP8 unembedding, padding columns -30)
        from .fast.head import FastHead, head_weight, unembed_t

        self.head = FastHead(mesh_device, cfg, head_weight(self.cache, lambda: unembed_t(source, cfg)))
        self.ids = ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=DRAM)
        # one decode table set per bucket, all allocated now (trace safety: see the module docstring)
        self.tables = {b: self._attention_tables(b) for b in self.buckets}
        self._bucket = self.buckets[0]
        # the attention prefill kernels (fast/attn_prefill.py: K / V front, flash attention with the in-kernel bias /
        # masks / sink, any start and context length), shared by every layer
        from .fast.attn_prefill import FlashPrefill

        self.flash = FlashPrefill(mesh_device, cfg, max_seq_len)
        # batched: the page row of the sequence a prefill chunk belongs to (int32 [1, max_seq_len / 64]: column j =
        # pool block of positions 64 j ..; host-written before each prefill call); single stream: None (identity order)
        self.prefill_pt = None
        if self.batched:
            self.prefill_pt = ttnn.from_torch(torch.arange(max_seq_len // 64, dtype=torch.int32).reshape(1, -1),
                                              dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
                                              memory_config=DRAM)
        self._prefill_pt_host = None  # host mirror of prefill_pt (written only when the page row changes)
        # batched decode step: the batch tables (fast/batch_tables.py; NOTES-batch.md section 3) and the reset_slot
        # listeners (BatchDecoder: sampler history / host mirror of the slot)
        self.bt = None
        if self.batched:
            from .fast.batch_tables import BatchTables

            self.bt = BatchTables(mesh_device, cfg, max_seq_len)
        self.slot_listeners = []
        self.glue, self.attn, self.moe = [], [], []
        for layer in range(cfg.n_layers):
            log(f"loading layer {layer}")
            self.glue.append(LayerGlue(mesh_device, cfg, source, layer))
            self.attn.append(self._attention(layer))
            self.moe.append(self._moe(layer))
        self.app_hook = None  # tests: app_hook(app_idx, layer, pass_idx, h_out_dev) after every application (prefill)
        H = cfg.hidden

        def msg_wdm():
            # decode: the message halves of every message gate's down projection, column block 4 j + k (j = loop
            # layer, k = q, k, v, mlp) -> Messages.GM
            cols = [t(f"model.llm.layers.{cfg.loop_start + j}.controllers.{ck}.down_weight")[:, H:].T
                    for j in range(cfg.n_loop_layers) for _, ck in GATE_NAMES]
            return torch.cat(cols, 1).reshape(1, 1, H, -1)

        self.msg_wdm = self.cache.get("msg_gate_down_m", msg_wdm, ttnn.bfloat16)
        self._emb = ttnn.to_device(self.decode_inputs_host(0), mesh_device, memory_config=DRAM)  # trace input
        self._traces = {}  # bucket -> (trace id, device logits)
        self._reset_tid = None  # capture_reset_trace()
        self.fast_glue = self.rows_glue = None  # fused glue (INK_GLUE=ttnn: the ttnn glue path for everything)
        if os.environ.get("INK_GLUE", "fast") != "ttnn":
            from .fast.glue import FastGlue
            from .fast.glue_rows import RowsGlue

            self.rows_glue = RowsGlue(self)  # prefill / verify rows; owns the glue convs' position rings
            for a in self.attn:  # the chunk's start / valid rows: one device tensor, written once per chunk
                a.prefill_pos = self.rows_glue.pos
            self.fast_glue = FastGlue(self)
            # attention weight prefetch in MID / PRE (not for test models whose attention stand-ins have no fused path;
            # not in batched mode: the batch step does not use it and its 256 KB / core of L1 would only shrink the
            # batched programs' circular buffers, NOTES-batch.md)
            if (self.fast_attn and not self.batched and os.environ.get("INK_ATTN_PF", "1") != "0"
                    and all(hasattr(a, "fast") for a in self.attn)):
                from .fast.attn_decode import PF_VERIFY, AttnPrefetch

                # INK_ATTN_PF_VERIFY=0: decode only (the speculative verify streams its weights from DRAM)
                vrows = () if os.environ.get("INK_ATTN_PF_VERIFY", "1") == "0" else PF_VERIFY
                self.fast_glue.prefetch = AttnPrefetch(mesh_device, cfg, [a.fast for a in self.attn],
                                                       self.fast_glue.grid, verify_rows=vrows)
        # prefix caching (NOTES-prefix.md): the device pool of slot-state snapshots, allocated now (trace safety)
        self.snaps = None
        if prefix_snapshots:
            from .prefix_cache import PrefixSnapshots

            self.snaps = PrefixSnapshots(self, prefix_snapshots)

    # ------------------------------------------------------------------ module construction (real modules)
    def _attention_tables(self, full_len: int):
        from .attention import AttnDecodeTables

        return AttnDecodeTables(self.device, self.cfg, max_seq_len=self.max_seq_len, full_len=full_len,
                                idx_rows=not self.fast_attn)

    def _attention(self, layer: int):
        from .attention import Inkling2Attention

        kw = dict(seqs=self.max_num_seqs, kv_blocks=self.kv_blocks) if self.batched else {}
        a = Inkling2Attention(self.device, self.src, layer, self.cfg, max_seq_len=self.max_seq_len,
                              dtype_policy=self.attn_policy, tables=self.attn_tables, weight_cache=self.cache,
                              flash=self.flash, weight_precision=self.attn_wprec[layer], **kw)
        a.prefill_pt = self.prefill_pt
        if self.fast_attn:
            a.enable_fast(self.cache)
        return a

    def _moe(self, layer: int):
        from .moe import Inkling2MoE

        p = self.expert_prec[layer]
        src, cache = expert_source(self.src, self.cache, self.device, p)
        return Inkling2MoE(self.device, src, layer, self.cfg, expert_dtype=DTYPES[p.fmt], cache=cache,
                           balance=self.moe_balance)

    # ------------------------------------------------------------------ state
    def reset(self):
        """Start a new sequence: zero every conv state / conv-input ring (stale K / V rows are always masked).  After
        capture_reset_trace() one trace replay (non-blocking) instead of ~400 ops of host dispatch (~10 ms)."""
        if self._reset_tid is not None:
            ttnn.execute_trace(self.device, self._reset_tid, cq_id=0, blocking=False)
            return
        self._reset_ops()

    def _reset_ops(self):
        if self.rows_glue is None:
            for g in self.glue:
                g.reset()
        else:  # the fused glue keeps every post-attention / post-MLP conv state in rows_glue's rings
            self.rows_glue.reset()
        for a in self.attn:
            a.reset()

    def capture_reset_trace(self):
        """Capture reset() as one trace (needs a trace region; every op zeroes a persistent tensor in place, so the
        trace has no intermediates and later allocations are safe); reset() replays it from then on."""
        if self._reset_tid is None:
            self._reset_ops()  # compile
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._reset_ops()
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self._reset_tid = tid

    def reset_slot(self, s: int):
        """Batched: a new sequence in slot s: zero slot s's attention conv-input rings and glue conv rings (its K / V
        rings and pool blocks need no reset: stale rows are masked by position), then the slot listeners
        (BatchDecoder: forget slot s's sampler history / mirror).  The other slots' state is untouched."""
        if not self.batched:
            raise RuntimeError("reset_slot: single-stream model (use reset())")
        if not 0 <= s < self.max_num_seqs:
            raise ValueError(f"slot {s} outside 0..{self.max_num_seqs - 1}")
        for a in self.attn:
            a.reset_slot(s)
        self.rows_glue.reset_slot(s)
        for f in self.slot_listeners:
            f(s)

    def snapshot_slot(self, s: int, j: int):
        """Prefix caching: prefix snapshot j = slot s's running state now (tt/prefix_cache.py; s = 0 single stream)."""
        self._check_snap(s, j)
        self.snaps.save(s, j)

    def restore_slot(self, j: int, s: int = 0):
        """Prefix caching: a sequence resumes from prefix snapshot j in slot s (single stream: 0): the slot listeners
        run as for reset_slot (batched), then every ring of slot s is overwritten with the snapshot's (exact)."""
        self._check_snap(s, j)
        for f in self.slot_listeners:
            f(s)
        self.snaps.restore(j, s)

    def _check_snap(self, s: int, j: int):
        if self.snaps is None:
            raise RuntimeError("prefix snapshots: Inkling2Model(prefix_snapshots=M) with M > 0")
        if not 0 <= s < self.max_num_seqs or not 0 <= j < self.snaps.count:
            raise ValueError(f"prefix snapshot {j} / slot {s} outside 0..{self.snaps.count - 1} / "
                             f"0..{self.max_num_seqs - 1}")

    # ------------------------------------------------------------------ one application
    def _attend(self, layer: int, p: int, xq, xk, xv, ctx):
        if ctx["mode"] == "verify":  # spec decode: multi-row ops of tt/spec.py
            return ctx["ops"].attend(layer, p, xq, xk, xv)
        if ctx["mode"] == "decode":
            return self.attn[layer].decode(xq, xk, xv, ctx["cur_pos"], p)
        return self.attn[layer].prefill(xq, xk, xv, ctx["start"], ctx["valid"], p)

    def _conv(self, conv: ShortConv, x, p: int, ctx):
        if ctx["mode"] == "verify":
            return ctx["ops"].conv(conv, x, p)
        if ctx["mode"] == "decode":
            return conv.decode(x, p)
        return conv.prefill(x, ctx["valid"], p, ctx["conv_idx"])

    def _apply(self, layer: int, p: int, h, S, msgs: Messages | None, ctx):
        """One decoder-layer application (reference Inkling2Loop*DecoderLayer.forward): h (BF16 layer input) -> BF16
        output.  S: the block states (ListStates / RowStates); at block boundaries norm(h) is appended to S.
        Does not free h."""
        G = self.glue[layer]
        j = layer - self.cfg.loop_start

        def gate(kind, x, nx=None):
            m, gm = msgs.get(j, kind), msgs.gm(j, kind)
            out = G.gates[kind](x, m, nx, gm)
            free(m, gm)
            return out

        # normed layer input: the partial state of attention_input / the new block state (FP32; BF16 in decode,
        # where the row-stacked mixing takes BF16 operands)
        nh = unorm(h, dtype=ctx["state_dtype"])
        if G.attn_res is None:  # layer 0: no attention_input; input_layernorm(h)
            S.append(nh)
            xs = None
        else:
            xs = S.mix(G.attn_res, nh, keep=G.boundary)
        if G.full:
            if msgs is not None:
                gated = [gate(kind, x) for x, kind in zip(xs, "qkv")]
                free(*xs)
                xs = gated
            xq, xk, xv = (n(x) for n, x in zip(G.qkv_norms, xs))
            free(*xs)
            ins = [xq, xk, xv]
        elif xs is None:
            xq = xk = xv = G.in_norm(h)
            ins = [xq]
        elif msgs is not None:
            x = xs[0]
            nx = unorm(x, dtype=BF16)
            ins = []
            for kind in "qkv":
                g = gate(kind, x, nx)
                ins.append(G.in_norm(g))
                ttnn.deallocate(g)
            free(nx, x)
            xq, xk, xv = ins
        else:
            xq = xk = xv = G.in_norm(xs[0])
            ttnn.deallocate(xs[0])
            ins = [xq]
        a = self._attend(layer, p, xq, xk, xv, ctx)
        free(*ins)
        c = self._conv(G.attn_conv, a, p, ctx)
        ttnn.deallocate(a)
        if G.boundary:  # the block restarts from the attention output; h is a block state now
            r = ttnn.typecast(c, F32, memory_config=DRAM)
        else:
            r = ttnn.add(h, c, dtype=F32, memory_config=DRAM)
        ttnn.deallocate(c)
        y = S.mix(G.mlp_res, unorm(r))[0]
        if msgs is not None:
            y2 = gate("mlp", y)
            ttnn.deallocate(y)
            y = y2
        z = G.post_attn_norm(y)
        ttnn.deallocate(y)
        if ctx["mode"] == "verify":
            mo = ctx["ops"].moe(layer, z)
        else:
            mo = self.moe[layer](z, decode=ctx["mode"] == "decode")
        ttnn.deallocate(z)
        d = self._conv(G.mlp_conv, mo, p, ctx)
        ttnn.deallocate(mo)
        ttnn.add(r, d, dtype=F32, memory_config=DRAM, output_tensor=r)
        ttnn.deallocate(d)
        out = ttnn.typecast(r, BF16, memory_config=DRAM)
        ttnn.deallocate(r)
        return out

    # ------------------------------------------------------------------ the 48 applications
    def _forward(self, x, ctx):
        """x: BF16 embeddings [1,1,T,H] (consumed) -> normed final hidden BF16 [1,1,T,H].  With the fused glue,
        prefill / verify run on rows_glue (prefill writes the chunk's start / valid rows into rows_glue.pos; a verify
        caller writes them with rows_glue.set_pos before the replay)."""
        if self.rows_glue is not None and ctx["mode"] in ("prefill", "verify"):
            return self._forward_rows(x, ctx)
        cfg = self.cfg
        decode = ctx["mode"] == "decode"
        ctx["state_dtype"] = BF16 if decode else F32
        hook = ctx.get("tap") or (None if decode else self.app_hook)
        app = 0
        S = RowStates(self.row_consts) if decode else ListStates()
        h = x

        def run(layers, p, msgs, keep):
            nonlocal h, app
            for layer in layers:
                new = self._apply(layer, p, h, S, msgs, ctx)
                if hook is not None:
                    hook(app, layer, p, new)
                app += 1
                if h is not keep:
                    ttnn.deallocate(h)
                h = new

        run(range(cfg.loop_start), 0, None, None)
        pre_h, pre_S = h, S.snapshot()
        loop = range(cfg.loop_start, cfg.loop_end)
        run(loop, 0, None, pre_h)
        for p in range(1, cfg.loop_passes):
            # passes 2.. restart from the pre-loop hidden / block states, with messages from the previous pass
            msgs = Messages(self.msg, S, pre_S.n, h, self.msg_wdm)
            ttnn.deallocate(h)
            S.free_from(pre_S.n)
            S = pre_S.snapshot()
            h = pre_h
            run(loop, p, msgs, pre_h)
            msgs.free()
        ttnn.deallocate(pre_h)
        run(range(cfg.loop_end, cfg.n_layers), 0, None, None)
        f = S.mix(self.final_res, unorm(h, dtype=ctx["state_dtype"]))[0]
        ttnn.deallocate(h)
        if decode:
            S.free_all()
        else:
            S.free_from(0)
        out = self.norm(f)
        ttnn.deallocate(f)
        return out

    def _forward_rows(self, x, ctx):
        """_forward on the fused rows glue (tt/fast/glue_rows.py): attention / MoE from the prefill modules or, in
        mode "verify", ctx["ops"].  A verify of R <= fast_glue.max_rows rows runs the decode glue programs as R
        instances (FastGlue.verify; INK_VERIFY_GLUE=rows keeps RowsGlue); with ctx["emb"] = the raw embedding rows
        (row r = token r) layer 0 fuses embed_norm as the decode step does, so each row's glue is bitwise a decode
        step's."""
        rg = self.rows_glue
        if ctx["mode"] == "verify":
            ops = ctx["ops"]
            attend, moe = ops.attend, ops.moe
            fg = self.fast_glue
            if fg is not None and ops.R <= fg.max_rows and os.environ.get("INK_VERIFY_GLUE", "fast") != "rows":
                raw = ctx.get("emb")
                out = fg.verify(x if raw is None else raw, ops.R, attend, moe,
                                hook=ctx.get("tap") or self.app_hook, raw=raw is not None)
                ttnn.deallocate(x)
                return ttnn.clone(out, memory_config=DRAM)
        else:
            if self.batched:  # the chunk's slot (word 2 of rows_glue.pos)
                rg.set_pos(ctx["start"], ctx["valid"], slot=ctx["slot"])
            else:
                rg.set_pos(ctx["start"], ctx["valid"])
            start, valid = ctx["start"], ctx["valid"]

            def attend(layer, p, xq, xk, xv):
                return self.attn[layer].prefill(xq, xk, xv, start, valid, p)

            def moe(layer, z):
                return self.moe[layer](z, decode=False)

        pf = self.fast_glue.prefetch if ctx["mode"] == "prefill" and self.fast_glue is not None else None
        with pf.released() if pf is not None else contextlib.nullcontext():  # prefill CBs use the prefetch L1
            out = rg.forward(x, attend, moe, hook=ctx.get("tap") or self.app_hook)
        ttnn.deallocate(x)
        return out

    # ------------------------------------------------------------------ prefill
    def prefill(self, token_ids, start_pos: int = 0, chunk: int | None = None, *, slot: int = 0,
                page_row=None, at=None, tap=None, on_chunk=None) -> torch.Tensor:
        """Prefill token_ids at positions start_pos.. -> FP32 logits [unpadded_vocab] of the last token.  Batched
        models: into slot `slot` with the sequence's page row `page_row` (its vLLM block-table row: column j = pool
        block of positions 64 j ..; required); single stream: slot 0, page_row None.  at(pos): called at every chunk
        boundary (before the chunk starting at absolute position pos, and with pos = the end after the last chunk;
        prefix caching publishes the slot's state there).  tap(app, layer, pass, h): every application's output rows
        of each chunk (ctx["tap"]); on_chunk(start, n, out): after each chunk's forward (absolute start, valid rows,
        the final hidden [1, 1, Tp, H]; batched speculation: the drafter's context; it runs before the next at())."""
        n = torch.as_tensor(token_ids).numel()
        return self._prefill(token_ids, start_pos, chunk, [n - 1], slot, page_row, at, tap, on_chunk)[0]

    def prefill_logits(self, token_ids, positions, start_pos: int = 0, chunk: int | None = None, *, slot: int = 0,
                       page_row=None) -> torch.Tensor:
        """Prefill token_ids at positions start_pos.. -> FP32 logits [len(positions), unpadded_vocab] at the given
        absolute positions (ascending, within start_pos .. start_pos + len - 1); slot / page_row as prefill()."""
        rel = [int(p) - start_pos for p in positions]
        return self._prefill(token_ids, start_pos, chunk, rel, slot, page_row)

    def _prefill_target(self, slot: int, page_row):
        """Batched: the device page row of the prefill (prefill_pt, written when it changes); checks slot."""
        if not self.batched:
            if slot != 0 or page_row is not None:
                raise ValueError("single-stream model: prefill into slot 0 without a page row")
            return
        if self.rows_glue is None:
            raise RuntimeError("batched prefill needs the fused rows glue (INK_GLUE=ttnn is single-stream only)")
        if not 0 <= slot < self.max_num_seqs:
            raise ValueError(f"slot {slot} outside 0..{self.max_num_seqs - 1}")
        if page_row is None:
            raise ValueError("batched prefill needs the sequence's page row")
        row = torch.as_tensor(page_row, dtype=torch.int32).reshape(-1)
        W = self.max_seq_len // 64
        if row.numel() > W:
            raise ValueError(f"page row of {row.numel()} columns > max_seq_len / 64 = {W}")
        host = torch.zeros(1, W, dtype=torch.int32)
        host[0, :row.numel()] = row
        if self._prefill_pt_host is None or not torch.equal(host, self._prefill_pt_host):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                            self.prefill_pt)
            self._prefill_pt_host = host

    def prefill_schedule(self, L: int, chunk: int | None = None) -> list[tuple[int, int]]:
        """(offset, rows) of the chunks of an L-token prompt (any start position: the attention prefill takes
        unaligned starts in-kernel): `chunk` (default prefill_chunk) rows while that many remain, then the largest
        multiple of PREFILL_TAIL below the rest, then the last <= PREFILL_TAIL rows (padded to a multiple of 32 on the
        device).  So the chunk shapes are the multiples of PREFILL_TAIL up to chunk plus Tp = 32 .. PREFILL_TAIL (what
        warmup_prefill compiles), and from start 0 every chunk but the last starts at a multiple of PREFILL_TAIL
        (64-aligned, as the drafter's context fill needs)."""
        chunk = chunk or self.prefill_chunk
        out, pos = [], 0
        while pos < L:
            r = L - pos
            n = chunk if r >= chunk else (r // PREFILL_TAIL * PREFILL_TAIL if r > PREFILL_TAIL else r)
            out.append((pos, n))
            pos += n
        return out

    def chunk_input(self, ids, Tp: int):
        """embed_norm(embedding rows) of the chunk's token ids -> BF16 [1, 1, Tp, H] (rows >= len(ids) padding)."""
        e = ttnn.to_device(self.embed.rows(ids, Tp), self.device, memory_config=DRAM)
        x = self.embed(e)
        ttnn.deallocate(e)
        return x

    def rows_logits(self, out, rows: list[int]) -> torch.Tensor:
        """Host FP32 logits [len(rows), unpadded_vocab] of the given rows of a chunk's final hidden `out`
        ([1, 1, Tp, H]): the rows by a one-hot matmul (exact for BF16; the programs depend only on Tp and the block
        size), the LM head on LOGIT_ROWS rows at a time, untilized on the device (the read moves the valid rows
        only, not the 32-row tile padding: 0.8 MB instead of 25.6 MB per row block)."""
        V, Tp = self.cfg.unpadded_vocab, int(out.shape[-2])
        parts = []
        for r0 in range(0, len(rows), LOGIT_ROWS):
            blk = rows[r0:r0 + LOGIT_ROWS]
            R = -(-len(blk) // 32) * 32
            sel = torch.zeros(1, 1, R, Tp)
            sel[0, 0, torch.arange(len(blk)), torch.tensor(blk)] = 1
            sel = ttnn.from_torch(sel, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM)
            h = ttnn.matmul(sel, out, dtype=BF16, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
            ttnn.deallocate(sel)
            lg = self.head(h, rows=len(blk))
            rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            parts.append(ttnn.to_torch(rm).float().reshape(len(blk), -1)[:, :V])
            free(h, lg, rm)
        return torch.cat(parts)

    def _prefill(self, token_ids, start_pos: int, chunk: int | None, want: list[int], slot: int = 0,
                 page_row=None, at=None, tap=None, on_chunk=None) -> torch.Tensor:
        ids = torch.as_tensor(token_ids, dtype=torch.int64).reshape(-1)
        L = ids.numel()
        chunk = chunk or self.prefill_chunk
        if L == 0:
            raise ValueError("empty prompt")
        if start_pos < 0 or start_pos + L > self.max_seq_len:
            raise ValueError(f"prompt positions {start_pos}..{start_pos + L - 1} outside 0..{self.max_seq_len - 1}")
        if chunk % 32 or not 32 <= chunk <= self.prefill_chunk:
            raise ValueError(f"prefill chunk {chunk}: a multiple of 32 up to prefill_chunk {self.prefill_chunk}")
        if any(b <= a for a, b in zip(want, want[1:])) or not all(0 <= w < L for w in want):
            raise ValueError("logit positions must ascend within the prompt")
        self._prefill_target(slot, page_row)
        out_rows, wi = [], 0
        for pos, n in self.prefill_schedule(L, chunk):
            if at is not None:
                at(start_pos + pos)
            x = self.chunk_input(ids[pos:pos + n], -(-n // 32) * 32)
            if self.rows_glue is not None:
                ctx = {"mode": "prefill", "start": start_pos + pos, "valid": n, "slot": slot}
                if tap is not None:
                    ctx["tap"] = tap
                out = self._forward(x, ctx)
            else:
                ctx = {"mode": "prefill", "start": start_pos + pos, "valid": n,
                       "conv_idx": ShortConv.state_index(self.device, n, self.cfg.conv_kernel)}
                if tap is not None:
                    ctx["tap"] = tap
                out = self._forward(x, ctx)
                ttnn.deallocate(ctx["conv_idx"])
            if on_chunk is not None:
                on_chunk(start_pos + pos, n, out)
            rows = []
            while wi < len(want) and want[wi] < pos + n:
                rows.append(want[wi] - pos)
                wi += 1
            if rows:
                out_rows.append(self.rows_logits(out, rows))
            ttnn.deallocate(out)
        if at is not None:
            at(start_pos + L)
        return torch.cat(out_rows)

    def warmup_shapes(self, chunk: int | None = None) -> list[int]:
        """Chunk lengths prefill_schedule produces: Tp = 32 .. PREFILL_TAIL, then the multiples of PREFILL_TAIL up to
        chunk."""
        chunk = chunk or self.prefill_chunk
        return list(range(32, min(chunk, PREFILL_TAIL) + 1, 32)) + list(range(2 * PREFILL_TAIL, chunk + 1,
                                                                                PREFILL_TAIL))

    def warmup_prefill(self, chunk: int | None = None, log=print):
        """Compile every prefill program: each warmup_shapes() chunk length as a first chunk (start 0) and as a
        continuation chunk (start = chunk, or the last 32-aligned start that fits max_seq_len; prefilled directly: the
        earlier rows' cache contents do not matter for the programs).  The glue's / MoE's / attention's prefill
        programs depend only on Tp (the valid row count and the start position are runtime operands).  Batched models
        prefill slot 0 through an all-zero page row (vLLM's null block 0: never a sequence's).  Ends with reset()."""
        chunk = chunk or self.prefill_chunk
        kw = dict(slot=0, page_row=torch.zeros(self.max_seq_len // 64, dtype=torch.int32)) if self.batched else {}
        for Tp in self.warmup_shapes(chunk):
            log(f"warmup prefill Tp={Tp}")
            self.reset()
            self.prefill(torch.ones(Tp, dtype=torch.int64), 0, chunk=chunk, **kw)
            start = min(chunk, (self.max_seq_len - Tp) // 32 * 32)
            if start > 0:
                self.reset()
                self.prefill(torch.ones(Tp, dtype=torch.int64), start, chunk=chunk, **kw)
        self.reset()

    # ------------------------------------------------------------------ decode
    def decode(self, emb_tt, cur_pos_tt, ids=None):
        """Device-only decode step (trace-capturable) -> FP32 logits [1, 1, 1, padded_unembed] on the device; the
        greedy token (argmax of those logits, ties -> lowest id) goes to word 0 of `ids` (uint32 ROW_MAJOR DRAM;
        default self.ids).  emb_tt: the token's embedding row (row 0 of a [1, 1, 32, H] BF16 tile,
        decode_inputs_host); not consumed."""
        if self.fast_glue is not None:
            out = self.fast_glue.decode(emb_tt, cur_pos_tt)  # embed_norm fused; persistent output tensor
        else:
            out = self._forward(self.embed(emb_tt), {"mode": "decode", "cur_pos": cur_pos_tt})
        lg = self.head(out, rows=1, ids=self.ids if ids is None else ids)  # row 0 of the [1, 1, 32, H] output
        if self.fast_glue is None:
            ttnn.deallocate(out)
        return lg

    def batch_decode(self, e_rows, bt, R: int, ids, hook=None, fin_out=None):
        """Batched decode step (trace-capturable; NOTES-batch.md section 4) of rows 0..R-1 (R <= 16): e_rows = raw
        embedding rows (BF16 [1, 1, 32, H] TILE, row b = row b's input token; not consumed), bt = the batch tables
        (row b: slot[b]'s sequence at cur_pos[b], page_table row b) -> FP32 logits [1, 1, R, padded_unembed] on the
        device; the greedy token of row b (argmax, ties -> lowest id) -> word b of `ids`.  Row b is bitwise the
        single-stream decode of that sequence (each batched module computes a row exactly as its one-row form); rows
        of one slot at consecutive positions in consecutive rows are a verify group (batched speculation: each row
        sees its group's earlier rows, as the single-stream verify).  hook(app, layer, pass, h): every application's
        output rows (FastGlue.batch); fin_out: a copy of the final hidden rows (the drafter's features)."""
        if not self.batched or self.fast_glue is None:
            raise RuntimeError("batch_decode: batched model with the fused glue")
        attn, moe = self.attn, self.moe

        def attend(layer, p, xq, xk, xv):
            return attn[layer].fast.batch(xq, xk, xv, bt, R, p)

        def moe_rows(layer, z):
            return moe[layer](z, decode=True, rows=R)

        fin = self.fast_glue.batch(e_rows, bt, R, attend, moe_rows, hook=hook)  # persistent output rows
        if fin_out is not None:
            ttnn.copy(fin, fin_out)
        return self.head(fin, rows=R, ids=ids)

    def decode_inputs_host(self, token: int):
        """Host decode input: the token's embedding row as row 0 of a [1, 1, 32, H] BF16 TILE tensor."""
        return self.embed.rows([int(token)], DECODE_ROWS)

    @property
    def attn_tables(self):
        """The decode tables of the current bucket (cur_pos = the decode step's position operand)."""
        return self.tables[self._bucket]

    def bucket(self, pos: int) -> int:
        """The decode bucket of position pos: the smallest full_len > pos."""
        if not 0 <= pos < self.max_seq_len:
            raise ValueError(f"decode position {pos} outside 0..{self.max_seq_len - 1} (max_seq_len)")
        return next(b for b in self.buckets if pos < b)

    def _use_bucket(self, b: int):
        if b != self._bucket:
            tb = self.tables[b]
            for a in self.attn:
                a.tables = tb
            self._bucket = b
        return self.tables[b]

    def set_decode_pos(self, pos: int) -> int:
        """Select pos's bucket and write its attention decode tables (cur_pos included) from the host; before every
        decode step / replay.  Returns the bucket."""
        b = self.bucket(pos)
        self._use_bucket(b).update(pos)
        return b

    def decode_step(self, token: int, pos: int) -> torch.Tensor:
        """Untraced decode of one token at position pos -> torch float32 [unpadded_vocab]."""
        tok = ttnn.to_device(self.decode_inputs_host(token), self.device, memory_config=DRAM)
        self.set_decode_pos(pos)
        lg = self.decode(tok, self.attn_tables.cur_pos)
        out = self.logits_to_torch(lg)
        free(tok, lg)
        return out

    def capture_decode_trace(self, buckets: list[int] | None = None):
        """Compile and capture the decode step of every bucket (default all; the device needs trace_region_size for
        all of them).  All compile runs come first: a program compiled after a capture may allocate device state
        (e.g. a tiled reshape's page map) inside that trace's intermediates.  Each compile run decodes position 0
        (writes position 0 of every cache and shifts the conv states): call between sequences; ends with reset().
        Re-capturing releases the previous traces."""
        self.release_trace()
        buckets = buckets or self.buckets
        for b in buckets:
            tb = self._use_bucket(b)
            tb.update(0)
            ttnn.copy_host_to_device_tensor(self.decode_inputs_host(0), self._emb)
            ttnn.deallocate(self.decode(self._emb, tb.cur_pos))  # compile
        for b in buckets:
            tb = self._use_bucket(b)
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            lg = self.decode(self._emb, tb.cur_pos)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self._traces[b] = (tid, lg)
        ttnn.synchronize_device(self.device)
        self.reset()

    def decode_traced(self, token: int, pos: int):
        """Replay pos's bucket trace for (token, pos) -> device FP32 logits [1, 1, 1, padded_unembed] (that trace's
        persistent output: read it before the next replay of any bucket's trace)."""
        b = self.bucket(pos)
        if b not in self._traces:
            raise RuntimeError(f"no decode trace for bucket {b} (position {pos}): capture_decode_trace()")
        ttnn.copy_host_to_device_tensor(self.decode_inputs_host(token), self._emb)
        self.set_decode_pos(pos)
        tid, lg = self._traces[b]
        ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
        return lg

    def logits_to_torch(self, lg) -> torch.Tensor:
        return ttnn.to_torch(lg).float().reshape(-1)[: self.cfg.unpadded_vocab]

    def release_trace(self):
        for tid, lg in self._traces.values():
            ttnn.release_trace(self.device, tid)
            ttnn.deallocate(lg)
        self._traces.clear()


# Kernel config ring (tt-metal dispatch.cpp initialize_worker_config_buf_mgr): every program's per-core config (all
# RISC binaries + runtime args + CB configs) is staged in the L1 between the firmware map and the worker L1 (69 KB by
# default on Blackhole).  The fused glue / SDPA programs carry ~48 KB of binaries per core, so at 69 KB the dispatcher
# could stage only one program ahead and stalled ~1 ms per decode step (tt/fast/NOTES-profile.md; 192 KB: test_model
# bfp4:gptq 14.59 -> 13.55 ms, outputs bitwise unchanged).  worker_l1_size = default (1461376 B) - (RING - 69) KB;
# serving passes the same value (tools/serve_inkling2.sh additional-config "worker_l1_size").
KERNEL_CONFIG_RING_KB = 192
WORKER_L1_SIZE = 1461376 - (KERNEL_CONFIG_RING_KB - 69) * 1024


def open_device(trace_region_size: int = 0):
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, num_command_queues=1,
                                 trace_region_size=trace_region_size, worker_l1_size=WORKER_L1_SIZE)
