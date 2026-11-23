# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM3-1.7B on one P150: embedding, 24 decoder layers (13 gated short conv, 6 full-attention and 5 sliding-window
attention operators; sandwich norms; dense SwiGLU FFN in every layer), final norm (model.embedding_norm), tied LM head.

Prefill: eager, chunked (2048), per request into one batch slot (its conv-state row) and its KV pages.
Decode: `decode_forward` is device-only (trace-capturable) over B batch rows (row b = slot b).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import ttnn

from .common import CK_HIFI4, DTYPES, WeightCache, free
from .config import FULL, SLIDING, Lfm3Config
from .layers import DRAM, Attention, DenseMLP, RMSNorm, Rope, ShortConv

BLOCK = 64                              # KV page size (tokens); vLLM --block-size
PREFILL_CHUNK = 2048
PREFILL_BUCKETS = (128, 256, 512, 1024, 2048)   # padded prefill chunk lengths (bounded program set)
PAD_MULTIPLE = PREFILL_BUCKETS[0]       # tests: the smallest prefill bucket


def prefill_bucket(n: int) -> int:
    for b in PREFILL_BUCKETS:
        if n <= b:
            return b
    raise ValueError(f"prefill chunk of {n} rows exceeds {PREFILL_BUCKETS[-1]}")


class Layer:
    """h = x + post_operator_norm(op(operator_norm(x))); out = h + post_ffn_norm(mlp(ffn_norm(h))) (the post norms
    only with use_sandwich_norm)."""

    def __init__(self, device, cfg, src, i: int, cache, batch: int):
        p = f"model.layers.{i}"
        self.i = i
        self.attn = cfg.is_attention(i)
        self.type = cfg.layer_types[i]
        norm = lambda n: RMSNorm(device, src.get(f"{p}.{n}.weight"), cfg.norm_eps)  # noqa: E731
        self.op_norm, self.ffn_norm = norm("operator_norm"), norm("ffn_norm")
        self.post_op_norm = norm("post_operator_norm") if cfg.sandwich_norm else None
        self.post_ffn_norm = norm("post_ffn_norm") if cfg.sandwich_norm else None
        self.op = Attention(device, cfg, src, i, cache) if self.attn else ShortConv(device, cfg, src, i, cache, batch)
        self.ffn = DenseMLP(device, cfg, src, i, cache)
        self._rn = {}  # prefill residual-norm programs per (post norm, next norm)

    def _residual(self, x, o, post):
        if post is not None:
            n = post(o)
            ttnn.deallocate(o)
            o = n
        out = ttnn.add(x, o, memory_config=DRAM)
        free(x, o)
        return out

    def _ffn(self, h):
        n = self.ffn_norm(h)
        f = self.ffn(n)
        ttnn.deallocate(n)
        return self._residual(h, f, self.post_ffn_norm)

    def _residual_norm(self, post, nxt):
        """prefill/norm.PrefillResidualNorm (h = x + post(o), n = nxt(h): one program), cached per (post, nxt)."""
        key = (id(post), id(nxt))
        rn = self._rn.get(key)
        if rn is None:
            from .prefill.norm import PrefillResidualNorm

            if post is None:
                raise ValueError("LFM3 prefill needs the sandwich post norms")
            rn = self._rn[key] = PrefillResidualNorm(post, nxt)
        return rn

    def prefill(self, x, ctx):
        """One whole layer (tests): x -> out."""
        return self.prefill_fused(x, self.op_norm(x), ctx)[0]

    def prefill_fused(self, x, n, ctx, nxt=None):
        """x (residual stream) and n = op_norm(x) -> (out, nxt(out) or None); the residual adds and the norms around
        them are fused (two programs per layer: operator residual + ffn norm, ffn residual + the next layer's
        operator norm)."""
        if self.attn:
            o = self.op.prefill(n, ctx["rope"][self.type], ctx["page_table"], ctx["chunk_page_table"],
                                ctx["start_tensor"])
        else:
            o = self.op.prefill(n, ctx["valid"], slot=ctx.get("slot", 0), fresh=ctx.get("start", 0) == 0)
        ttnn.deallocate(n)
        h, n2 = self._residual_norm(self.post_op_norm, self.ffn_norm)(x, o)
        free(x, o)
        f = self.ffn.prefill(n2)
        ttnn.deallocate(n2)
        out, n_next = self._residual_norm(self.post_ffn_norm, nxt)(h, f)
        free(h, f)
        return out, n_next

    def decode(self, x, ctx):
        n = self.op_norm(x)
        if self.attn:
            cos, sin = ctx["cos_sin"][self.type]
            o = self.op.decode(n, cos, sin, ctx["rope"][self.type], ctx["cur_pos"], ctx["page_table"])
        else:
            o = self.op.decode(n)
        ttnn.deallocate(n)
        return self._ffn(self._residual(x, o, self.post_op_norm))


class Lfm3Model:
    """batch = decode rows / conv-state slots.  own_kv: allocate KV caches + an identity page table for standalone
    use (a serving adapter instead calls allocate_kv_caches with its block pool).  logits_dtype: FP32 (tests) or BF16
    (serving: device argmax / sampling take BF16 logits, like HF's BF16 lm_head)."""

    def __init__(self, device, src, cfg: Lfm3Config | None = None, max_seq_len: int = 8192, batch: int = 1,
                 layers: list[int] | None = None, use_cache: bool = True, lm_head: bool = True,
                 logits_dtype=ttnn.float32, own_kv: bool = True, cache_root=None, log=print):
        self.device, self.src = device, src
        self.cfg = cfg = cfg or Lfm3Config.from_hf()
        self.batch, self.max_seq_len = batch, max_seq_len
        self.logits_dtype = logits_dtype
        self.cache = (WeightCache(device, src.identity, enabled=use_cache) if cache_root is None
                      else WeightCache(device, src.identity, root=Path(cache_root), enabled=use_cache))
        self.layer_ids = list(range(cfg.n_layers)) if layers is None else list(layers)
        # Embedding table stays BF16 (exact rows), row-major for ttnn.embedding.
        self.embed = self.cache.get("embed", lambda: src.get("model.embed_tokens.weight"), ttnn.bfloat16,
                                    layout=ttnn.ROW_MAJOR_LAYOUT)
        self.final_norm = RMSNorm(device, src.get("model.embedding_norm.weight"), cfg.norm_eps)
        # Tied head: the quant checkpoint carries its BFP8-grid copy as lm_head.weight; the HF one only the table.
        head_key = "lm_head.weight" if "lm_head.weight" in set(src.keys()) else "model.embed_tokens.weight"
        head_fmt = src.fmt(head_key) if head_key == "lm_head.weight" else "bfp8"
        self.lm_head = (self.cache.get("lm_head", lambda: src.get(head_key).T, DTYPES[head_fmt]) if lm_head else None)
        types = {cfg.layer_types[i] for i in self.layer_ids} & {FULL, SLIDING}
        self.rope = {t: Rope(device, cfg, t, max_seq_len) for t in sorted(types)}
        self.layers = []
        for i in self.layer_ids:
            log(f"loading layer {i}")
            self.layers.append(Layer(device, cfg, src, i, self.cache, batch))
        self.kv = None
        self.trash_block = None
        self._argmax = None
        self.fast = None  # tt/fast/step.py FastStep (enable_fast_decode), used by decode_forward when set
        self.drafter = None  # tt/drafter.py Drafter (enable_drafter): prefill chunks and decode steps commit its K/V
        if own_kv:
            per_seq = math.ceil(max_seq_len / BLOCK)
            self.allocate_kv_caches([per_seq * batch, cfg.n_kv_heads, BLOCK, cfg.head_dim])
            self.page_table_host = torch.arange(per_seq * batch, dtype=torch.int32).reshape(batch, per_seq)
            self.page_table = ttnn.from_torch(self.page_table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                              device=device, memory_config=DRAM)

    # ------------------------------------------------------------------ state
    def allocate_kv_caches(self, kv_cache_shape):
        """Paged BFP8 K/V for every attention layer (full and sliding alike: same page table, full-length pages):
        [num_blocks, n_kv, block, head_dim] plus one private trash block (index num_blocks) that receives the padded
        prefill rows beyond a request's own pages.  Returns the [k, v] pairs (attention layers in order)."""
        blocks, nkv, block, hd = (int(d) for d in kv_cache_shape)
        if (nkv, block, hd) != (self.cfg.n_kv_heads, BLOCK, self.cfg.head_dim):
            raise ValueError(f"KV cache shape {tuple(kv_cache_shape)} does not match "
                             f"[*, {self.cfg.n_kv_heads}, {BLOCK}, {self.cfg.head_dim}]")
        self.trash_block = blocks
        self.kv = []
        for layer in self.layers:
            if layer.attn:
                k, v = (ttnn.zeros([blocks + 1, nkv, block, hd], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                   device=self.device, memory_config=DRAM) for _ in range(2))
                layer.op.set_kv_cache(k, v)
                self.kv.append([k, v])
        return self.kv

    def reset(self, slot: int | None = None):
        """Zero the conv state of one slot (None: all).  KV positions are overwritten before they are read."""
        for layer in self.layers:
            if not layer.attn:
                layer.op.reset(slot)

    def save_slot_state(self, slot: int) -> list:
        """Host copy of slot's conv state (13 x [6, H] BF16; the prefix-resume payload)."""
        return [layer.op.state_rows(slot) for layer in self.layers if not layer.attn]

    def restore_slot_state(self, slot: int, rows: list):
        for layer, r in zip([layer for layer in self.layers if not layer.attn], rows):
            layer.op.restore_slot(slot, r)

    def remap_slots(self, remap):
        """Decode rows moved (serving condense): new row r takes old slot remap[r] (rows beyond remap keep theirs).
        On the device (ShortConv.remap: one exact permutation matmul per state tensor), queued like a decode step."""
        idx = [int(i) for i in torch.as_tensor(remap).reshape(-1).tolist()]
        perm = torch.eye(self.batch, dtype=torch.bfloat16)
        perm[: len(idx)] = 0
        for r, k in enumerate(idx):
            perm[r, k] = 1
        perm_d = ttnn.from_torch(perm.reshape(1, 1, self.batch, self.batch), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM)
        for layer in self.layers:
            if not layer.attn:
                layer.op.remap(perm_d)
        ttnn.deallocate(perm_d)

    def save_slots(self, slots) -> list:
        """Host copies of several slots' conv state (one read per state tensor; exact BF16)."""
        idx = torch.as_tensor(slots, dtype=torch.long)
        return [[ttnn.to_torch(s).reshape(layer.op.batch, layer.op.H)[idx].clone() for s in layer.op.s]
                for layer in self.layers if not layer.attn]

    def restore_slots(self, slots, saved: list):
        """Write save_slots() copies back into those slots (the other slots keep their current state)."""
        idx = torch.as_tensor(slots, dtype=torch.long)
        for layer, rows in zip([layer for layer in self.layers if not layer.attn], saved):
            op = layer.op
            for s, r in zip(op.s, rows):
                host = ttnn.to_torch(s).reshape(op.batch, op.H).clone()
                host[idx] = r
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(host.reshape(1, 1, op.batch, op.H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                    s)

    # ------------------------------------------------------------------ helpers
    def _embed(self, ids: torch.Tensor):
        t = ttnn.from_torch(ids.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=self.device, memory_config=DRAM)
        e = ttnn.embedding(t, self.embed, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(t)
        n = ids.numel()
        return ttnn.reshape(e, [1, 1, n, self.cfg.hidden])

    def head(self, x_row):
        """[1, 1, R, H] -> logits [1, 1, R, vocab] (device, self.logits_dtype)."""
        n = self.final_norm(x_row)
        logits = ttnn.linear(n, self.lm_head, dtype=self.logits_dtype, compute_kernel_config=CK_HIFI4,
                             memory_config=DRAM)
        ttnn.deallocate(n)
        return logits

    def _prefill_page_rows(self, row: torch.Tensor, s: int, n: int, Tp: int):
        """(full page-table row, chunk page-table row) covering positions [0, s + Tp): the request's own pages for
        the rows it owns (ceil((s + n) / BLOCK) pages), the trash block for the padded remainder.  The full row is at
        least max_seq_len / BLOCK wide (a fixed width: the prefill SDPA program depends on it), rounded up to a multiple
        of 8 blocks (32-byte page-table rows)."""
        own = -(-(s + n) // BLOCK)
        total = (s + Tp) // BLOCK
        width = max(total, row.numel(), -(-self.max_seq_len // BLOCK))
        full = torch.full((1, -(-width // 8) * 8), self.trash_block, dtype=torch.int32)
        full[0, :own] = row[:own].to(torch.int32)
        return full, full[:, s // BLOCK: total].contiguous()

    # ------------------------------------------------------------------ prefill
    def prefill(self, tokens, start: int = 0, chunk: int = PREFILL_CHUNK, slot: int = 0, page_table_row=None,
                hidden_hook=None, return_logits=True, snapshot=None):
        """Prefill token ids at positions start.. (start % chunk == 0) into batch slot `slot` (conv-state row) and the
        KV pages of `page_table_row` (host int tensor; default: this model's own table row `slot`).  start == 0
        starts the slot's conv state from zero.  Returns the last token's logits (torch float [vocab]).
        hidden_hook(layer_id, x_dev, valid) is called after every layer (tests: per-layer hidden states)."""
        ids = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
        L = ids.numel()
        if start % chunk or chunk % BLOCK:
            raise ValueError("chunked prefill: chunk a multiple of the block, starts chunk aligned")
        row = (self.page_table_host[slot] if page_table_row is None else torch.as_tensor(page_table_row)).reshape(-1)
        pos = 0
        last = None
        while pos < L:
            n = min(chunk, L - pos)
            Tp = prefill_bucket(n)
            chunk_ids = torch.full((Tp,), self.cfg.pad, dtype=torch.int64)
            chunk_ids[:n] = ids[pos:pos + n]
            s = start + pos
            if s + n > self.max_seq_len:
                raise ValueError("prompt exceeds max_seq_len")
            full, part = self._prefill_page_rows(row, s, n, Tp)
            kw = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, memory_config=DRAM)
            page_table, chunk_pt = ttnn.from_torch(full, **kw), ttnn.from_torch(part, **kw)
            # chunk start as a device scalar: the attention programs read it at run time (one program per bucket)
            start_t = ttnn.from_torch(torch.tensor([s], dtype=torch.int32), **kw)
            ctx = {"start": s, "valid": n, "slot": slot, "rope": self.rope, "page_table": page_table,
                   "chunk_page_table": chunk_pt, "start_tensor": start_t}
            x = self._embed(chunk_ids)
            if hidden_hook is not None:
                hidden_hook(-1, x, n)
            nrm = self.layers[0].op_norm(x)
            taps = {}
            for li, layer in enumerate(self.layers):
                nxt = self.layers[li + 1].op_norm if li + 1 < len(self.layers) else None
                x, nrm = layer.prefill_fused(x, nrm, ctx, nxt)
                if hidden_hook is not None:
                    hidden_hook(layer.i, x, n)
                if self.drafter is not None and layer.i in self.drafter.taps:
                    taps[layer.i] = x if li + 1 == len(self.layers) else ttnn.clone(x, memory_config=DRAM)
            if self.drafter is not None:
                # tt/drafter.py: the chunk's rows into the drafter's K/V (same pages and trash rows as the target's)
                self.drafter.prefill_commit([taps[t] for t in self.drafter.taps], start_t, chunk_pt)
                free(*[t for t in taps.values() if t is not x])
            free(chunk_pt, page_table, start_t)
            pos += n
            if snapshot is not None and n == chunk and L - pos < chunk:
                # The last full-chunk boundary of this prompt: publish the slot's state there (prefix resume).
                snapshot(start + pos, self.save_slot_state(slot))
            if pos >= L and return_logits and self.lm_head is not None:
                xr = ttnn.slice(x, [0, 0, n - 1, 0], [1, 1, n, self.cfg.hidden])
                logits = self.head(xr)
                # untilized on device: the host reads one row (0.5 MB FP32), not the 32-row padded tile buffer
                # (16 MB + host untilize: ~3 ms per prompt; bitwise the same values)
                rm = ttnn.untilize(logits)
                last = ttnn.to_torch(rm).float().reshape(-1)[: self.cfg.vocab]
                free(xr, logits, rm)
            ttnn.deallocate(x)
        return last

    # ------------------------------------------------------------------ decode
    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Host decode inputs: tokens [1, B] u32, cur_pos [B] i32 (-1 = inactive row), rope idx [1, B] u32 (inactive
        rows read position 0), page table [B, blocks] i32."""
        tok = torch.as_tensor(tokens).reshape(1, -1).to(torch.int32)
        B = tok.shape[1]
        if isinstance(current_pos, torch.Tensor):
            pos = current_pos.reshape(-1).to(torch.int32)
        else:
            pos = torch.full((B,), int(current_pos), dtype=torch.int32)
        rk = dict(layout=ttnn.ROW_MAJOR_LAYOUT)
        return (ttnn.from_torch(tok, dtype=ttnn.uint32, **rk), ttnn.from_torch(pos, dtype=ttnn.int32, **rk),
                ttnn.from_torch(pos.clamp(min=0).reshape(1, -1), dtype=ttnn.uint32, **rk),
                ttnn.from_torch(torch.as_tensor(page_table).to(torch.int32), dtype=ttnn.int32, **rk)
                if page_table is not None else None)

    def enable_fast_decode(self, parts=None):
        """Route decode_forward through the fused decode modules (tt/fast/step.py FastStep).  parts: None ->
        env LFM_FAST_PARTS (comma list) or step.DEFAULT_PARTS.  Call after the KV caches are allocated and before any
        trace capture."""
        import os

        from .fast.step import DEFAULT_PARTS, FastStep

        if not parts:
            env = os.environ.get("LFM_FAST_PARTS", "").strip()
            parts = tuple(p for p in env.split(",") if p) if env else DEFAULT_PARTS
        self.fast = FastStep(self, parts=tuple(parts))
        return self.fast

    def enable_drafter(self, path=None, page_blocks: int | None = None, log=print):
        """Load the block drafter (tt/drafter.py; export dir `path`, default env LFM3_DRAFTER) over this model's KV
        pool (allocate_kv_caches first) and the fast step (enable_fast_decode first): every prefill chunk commits its
        K/V, every decode / verify step exports its taps (the commit is the drafter's own trace, Drafter.commit);
        draft passes via self.drafter.  Before any trace capture."""
        from .drafter import Drafter
        from .speculate import VERIFY_WIDTHS

        if self.fast is None or self.kv is None:
            raise ValueError("enable_drafter needs the KV caches and the fast decode step")
        d = Drafter(self, path, log=log)
        d.allocate(self.trash_block, page_blocks or -(-self.max_seq_len // BLOCK),
                   list(self.fast.widths) + list(VERIFY_WIDTHS))
        self.fast.enable_drafter(d)
        self.drafter = d
        return d

    @property
    def decode_widths(self) -> list[int]:
        """Widths the decode step runs at: the fast step's buckets (1, 2, 4, ..., batch); the stock step only the full
        slot width (its conv-state ops take every slot)."""
        return list(self.fast.widths) if self.fast is not None else [self.batch]

    def decode_width(self, rows: int) -> int:
        """The smallest decode width holding decode rows [0, rows) (row b = conv-state slot b)."""
        return next(w for w in self.decode_widths if w >= max(1, rows))

    def decode_forward(self, tok, cur_pos, rope_idx, page_table):
        """Device-only decode step (trace-capturable): logits [1, 1, B, vocab] on device (self.logits_dtype); the
        fast step when enabled (enable_fast_decode), else the stock step."""
        if self.fast is not None:
            return self.fast.decode_forward(tok, cur_pos, rope_idx, page_table)
        return self.decode_forward_stock(tok, cur_pos, rope_idx, page_table)

    def decode_forward_stock(self, tok, cur_pos, rope_idx, page_table):
        """Stock device-only decode step (trace-capturable) over B = tok.shape[-1] rows (= self.batch: the conv-state
        ops take every slot): logits [1, 1, B, vocab] on device."""
        B = tok.shape[-1]
        e = ttnn.embedding(tok, self.embed, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
        x = ttnn.reshape(e, [1, 1, B, self.cfg.hidden])
        cos_sin = {t: r.decode(rope_idx) for t, r in self.rope.items()}
        ctx = {"cos_sin": cos_sin, "rope": self.rope, "cur_pos": cur_pos, "page_table": page_table}
        for layer in self.layers:
            x = layer.decode(x, ctx)
        for cos, sin in cos_sin.values():
            free(cos, sin)
        logits = self.head(x)
        ttnn.deallocate(x)
        return logits

    def greedy_tokens(self, logits):
        """uint32 argmax over the vocabulary per row (ties -> lowest id), on device."""
        rows = math.prod(int(d) for d in list(logits.shape)[:-1])
        if logits.dtype != ttnn.bfloat16:
            logits = ttnn.typecast(logits, ttnn.bfloat16)
        if rows <= 2:
            if self._argmax is None:
                from models.demos.blackhole.qwen36.tt.tile_argmax import TileArgmax

                self._argmax = TileArgmax()
            return self._argmax(logits)
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        out = ttnn.argmax(rm, dim=-1, memory_config=DRAM)
        ttnn.deallocate(rm)
        return out

    # ------------------------------------------------------------------ standalone helpers
    def decode_inputs(self, tokens, positions):
        """Device decode inputs for standalone use (tok, cur_pos, rope_idx)."""
        tok, pos, ridx, _ = self.prepare_decode_inputs_host(torch.as_tensor(tokens), torch.as_tensor(positions))
        return tuple(ttnn.to_device(t, self.device, memory_config=DRAM) for t in (tok, pos, ridx))

    def decode(self, token: int, position: int):
        tok, cur_pos, rope_idx = self.decode_inputs([token], [position])
        logits = self.decode_forward(tok, cur_pos, rope_idx, self.page_table)
        out = ttnn.to_torch(logits).float().reshape(-1)[: self.cfg.vocab]
        free(tok, cur_pos, rope_idx, logits)
        return out


def open_device(trace_region_size: int = 0):
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, num_command_queues=1,
                                 trace_region_size=trace_region_size)
