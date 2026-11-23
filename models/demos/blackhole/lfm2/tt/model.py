# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM2.5-24B-A2B on one P150: embedding, 40 decoder layers (30 gated short conv + 10 GQA attention operators; dense
SwiGLU FFN in layers 0-1, 64-expert top-4 MoE in 2-39), final norm (model.embedding_norm), tied LM head.

Prefill: eager, chunked (2048), per request into one batch slot (its conv-state row) and its KV pages.
Decode: `decode_forward` is device-only (trace-capturable) over B batch rows (row b = slot b).
The Generator contract (prepare_decode_inputs_host / ttnn_decode_forward / process_output_decode / switch_mode) is
what tt_transformers' Generator calls for traced decode under vLLM (tt/lfm2_vllm.py).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import ttnn

from .common import CK_HIFI4, WeightCache, free
from .config import Lfm2Config
from .layers import DRAM, Attention, DenseMLP, RMSNorm, Rope, ShortConv
from .moe import MoE

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
    def __init__(self, device, cfg, src, i: int, cache, batch: int):
        p = f"model.layers.{i}"
        self.i = i
        self.attn = cfg.is_attention(i)
        self.op_norm = RMSNorm(device, src.get(f"{p}.operator_norm.weight"), cfg.norm_eps)
        self.ffn_norm = RMSNorm(device, src.get(f"{p}.ffn_norm.weight"), cfg.norm_eps)
        self.op = Attention(device, cfg, src, i, cache) if self.attn else ShortConv(device, cfg, src, i, cache, batch)
        self.ffn = MoE(device, cfg, src, i, cache) if cfg.is_moe(i) else DenseMLP(device, cfg, src, i, cache)

    def _ffn(self, h):
        n = self.ffn_norm(h)
        f = self.ffn(n)
        ttnn.deallocate(n)
        out = ttnn.add(h, f, memory_config=DRAM)
        free(h, f)
        return out

    def prefill(self, x, ctx):
        n = self.op_norm(x)
        if self.attn:
            o = self.op.prefill(n, ctx["start"], ctx["rope"], ctx["page_table"], ctx["chunk_page_table"])
        else:
            o = self.op.prefill(n, ctx["valid"], slot=ctx.get("slot", 0), fresh=ctx.get("start", 0) == 0)
        ttnn.deallocate(n)
        h = ttnn.add(x, o, memory_config=DRAM)
        free(o)
        return self._ffn(h)

    def decode(self, x, ctx):
        n = self.op_norm(x)
        if self.attn:
            o = self.op.decode(n, ctx["cos"], ctx["sin"], ctx["rope"], ctx["cur_pos"], ctx["page_table"])
        else:
            o = self.op.decode(n)
        ttnn.deallocate(n)
        h = ttnn.add(x, o, memory_config=DRAM)
        free(o)
        return self._ffn(h)


class Lfm2Model:
    """batch = decode rows / conv-state slots.  own_kv: allocate KV caches + an identity page table for standalone
    use; vLLM instead calls allocate_kv_caches with its block pool.  logits_dtype: FP32 (tests) or BF16 (serving:
    the device sampler and argmax take BF16 logits, like HF's BF16 lm_head)."""

    def __init__(self, device, src, cfg: Lfm2Config | None = None, max_seq_len: int = 8192, batch: int = 1,
                 layers: list[int] | None = None, use_cache: bool = True, lm_head: bool = True,
                 logits_dtype=ttnn.float32, own_kv: bool = True, cache_root=None, log=print):
        self.device, self.mesh_device, self.src = device, device, src
        self.cfg = cfg = cfg or Lfm2Config.from_hf()
        self.batch, self.max_seq_len = batch, max_seq_len
        self.logits_dtype = logits_dtype
        self.num_devices = 1
        self.cache = (WeightCache(device, src.identity, enabled=use_cache) if cache_root is None
                      else WeightCache(device, src.identity, root=Path(cache_root), enabled=use_cache))
        self.layer_ids = list(range(cfg.n_layers)) if layers is None else list(layers)
        # Embedding table stays BF16 (exact rows), row-major for ttnn.embedding.
        self.embed = self.cache.get("embed", lambda: src.get("model.embed_tokens.weight"), ttnn.bfloat16,
                                    layout=ttnn.ROW_MAJOR_LAYOUT)
        self.final_norm = RMSNorm(device, src.get("model.embedding_norm.weight"), cfg.norm_eps)
        # Tied head: the quant checkpoint carries its BFP8-grid copy as lm_head.weight; the HF one only the table.
        head_key = "lm_head.weight" if "lm_head.weight" in set(src.keys()) else "model.embed_tokens.weight"
        self.lm_head = (self.cache.get("lm_head", lambda: src.get(head_key).T, ttnn.bfloat8_b) if lm_head else None)
        self.rope = Rope(device, cfg, max_seq_len)
        self.layers = []
        for i in self.layer_ids:
            log(f"loading layer {i}")
            self.layers.append(Layer(device, cfg, src, i, self.cache, batch))
        self.kv = None
        self.trash_block = None
        self._argmax = None
        self.fast = None  # tt/fast/step.py FastStep (enable_fast_decode), used by decode_forward when set
        if own_kv:
            per_seq = math.ceil(max_seq_len / BLOCK)
            self.allocate_kv_caches([per_seq * batch, cfg.n_kv_heads, BLOCK, cfg.head_dim])
            self.page_table_host = torch.arange(per_seq * batch, dtype=torch.int32).reshape(batch, per_seq)
            self.page_table = ttnn.from_torch(self.page_table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                              device=device, memory_config=DRAM)

    # ------------------------------------------------------------------ state
    def allocate_kv_caches(self, kv_cache_shape):
        """Paged BFP8 K/V for the attention layers: [num_blocks, n_kv, block, head_dim] plus one private trash
        block (index num_blocks) that receives the padded prefill rows beyond a request's own pages.  Returns the
        [k, v] pairs (attention layers in order)."""
        blocks, nkv, block, hd = (int(d) for d in kv_cache_shape)
        if (nkv, block, hd) != (self.cfg.n_kv_heads, BLOCK, self.cfg.head_dim):
            raise ValueError(f"KV cache shape {tuple(kv_cache_shape)} does not match [*, 8, {BLOCK}, 64]")
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
        """Host copy of slot's conv state (30 x [2, H] BF16; the prefix-resume payload)."""
        return [layer.op.save_slot(slot) for layer in self.layers if not layer.attn]

    def restore_slot_state(self, slot: int, rows: list):
        for layer, r in zip([layer for layer in self.layers if not layer.attn], rows):
            layer.op.restore_slot(slot, r)

    def remap_slots(self, remap):
        """Decode rows moved (vLLM condense): new row r takes old slot remap[r] (host round trip; rare)."""
        idx = torch.as_tensor(remap, dtype=torch.long).reshape(-1)
        for layer in self.layers:
            if not layer.attn:
                layer.op.remap(idx)

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
        the rows it owns (ceil((s + n) / BLOCK) pages), the trash block for the padded remainder."""
        own = -(-(s + n) // BLOCK)
        total = (s + Tp) // BLOCK
        full = torch.full((1, max(total, row.numel())), self.trash_block, dtype=torch.int32)
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
        if start % chunk:
            raise ValueError("chunked prefill starts must be chunk aligned")
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
            ctx = {"start": s, "valid": n, "slot": slot, "rope": self.rope, "page_table": page_table,
                   "chunk_page_table": chunk_pt}
            x = self._embed(chunk_ids)
            if hidden_hook is not None:
                hidden_hook(-1, x, n)
            for layer in self.layers:
                x = layer.prefill(x, ctx)
                if hidden_hook is not None:
                    hidden_hook(layer.i, x, n)
            free(chunk_pt, page_table)
            pos += n
            if snapshot is not None and n == chunk and L - pos < chunk:
                # The last full-chunk boundary of this prompt: publish the slot's state there (prefix resume).
                snapshot(start + pos, self.save_slot_state(slot))
            if pos >= L and return_logits and self.lm_head is not None:
                xr = ttnn.slice(x, [0, 0, n - 1, 0], [1, 1, n, self.cfg.hidden])
                logits = self.head(xr)
                last = ttnn.to_torch(logits).float().reshape(-1)[: self.cfg.vocab]
                free(xr, logits)
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

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        from models.tt_transformers.tt.common import copy_host_to_device

        return copy_host_to_device(self.prepare_decode_inputs_host(tokens, current_pos, page_table),
                                   mesh_device=self.device)

    def enable_fast_decode(self, parts=None):
        """Route decode_forward through LfmKern's fused decode modules (tt/fast/step.py FastStep).  Call after the KV
        caches are allocated and before any trace capture."""
        from .fast.step import DEFAULT_PARTS, FastStep

        self.fast = FastStep(self, parts=tuple(parts) if parts else DEFAULT_PARTS)
        return self.fast

    @property
    def decode_widths(self) -> list[int]:
        """Widths the decode step runs at: the fast step's buckets (1, 2, 4, ..., batch); the stock step only the full
        slot width (its conv-state ops take every slot)."""
        return list(self.fast.widths) if self.fast is not None else [self.batch]

    def decode_width(self, rows: int) -> int:
        """The smallest decode width holding decode rows [0, rows) (row b = conv-state slot b)."""
        return next(w for w in self.decode_widths if w >= max(1, rows))

    def decode_forward(self, tok, cur_pos, rope_idx, page_table):
        """Device-only decode step (trace-capturable): logits [1, 1, B, vocab] on device."""
        if self.fast is not None:
            return self.fast.decode_forward(tok, cur_pos, rope_idx, page_table)
        return self.decode_forward_stock(tok, cur_pos, rope_idx, page_table)

    def decode_forward_stock(self, tok, cur_pos, rope_idx, page_table):
        B = tok.shape[-1]
        e = ttnn.embedding(tok, self.embed, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
        x = ttnn.reshape(e, [1, 1, B, self.cfg.hidden])
        cos, sin = self.rope.decode(rope_idx)
        ctx = {"cos": cos, "sin": sin, "rope": self.rope, "cur_pos": cur_pos, "page_table": page_table}
        for layer in self.layers:
            x = layer.decode(x, ctx)
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
        """The exact vLLM-equivalent device sampler (tt/sampling.py: Qwen36's sampler plus top_k-disabled rows):
        fixed-address buffers, allocated before any trace is captured."""
        if getattr(self, "exact_sampler", None) is None:
            from .sampling import DeviceSampler

            self.exact_sampler = DeviceSampler(self.device, self.cfg.vocab, num_slots=num_slots)
        return self.exact_sampler

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
