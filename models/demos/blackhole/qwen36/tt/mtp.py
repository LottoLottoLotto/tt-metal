# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 multi-token predictor for single-device TT speculative decode."""

import torch
import ttnn

from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.tt_transformers.tt.common import Mode


class Qwen36MTP:
    """One-layer Qwen3.5 MTP head using the target embedding and LM head."""

    def __init__(self, model, state_dict, tensor_cache_path=None):
        if model.num_devices != 1:
            raise NotImplementedError("Qwen3.5 MTP currently supports one TT device")
        self.model = model
        self.device = model.device
        self.args = model.args
        self.num_steps = 4

        self.pre_fc_norm_embedding = self._norm(
            state_dict, "pre_fc_norm_embedding", tensor_cache_path
        )
        self.pre_fc_norm_hidden = self._norm(
            state_dict, "pre_fc_norm_hidden", tensor_cache_path
        )
        self.norm = self._norm(state_dict, "norm", tensor_cache_path)

        fc_weight = state_dict["mtp.fc.weight"].to(torch.bfloat16).T.contiguous()
        self.fc_weight = ttnn.as_tensor(
            fc_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / "mtp.fc.weight.bf16")
            if tensor_cache_path
            else None,
        )
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self.fc_decode_pc = tpc.create_matmul_1d_decode_progcfg(
            1,
            self.fc_weight.shape[-2],
            self.fc_weight.shape[-1],
            num_cores=44,
            grid_w=11,
        )
        if self.args.dim == 5120:
            # BF16 FC streams K=10240 beside the permanent recurrent-state pool.
            self.fc_decode_pc.in0_block_w = 2
        self.layer = Qwen36DecoderLayer(
            self.device,
            self.args,
            state_dict,
            0,
            tensor_cache_path,
            state_prefix="mtp.layers.0",
            force_full_attention=True,
        )
        cache_shape = (1, self.args.n_kv_heads, 64, self.args.head_dim)
        # One 64-row page. The draft shares the target attention contract (BF16 for the
        # optimized 27B; BF8 for the sealed 9B), so its paged SDPA program config matches.
        cache_dtype = ttnn.bfloat16 if self.layer.attention.optimized_27b else ttnn.bfloat8_b
        key_cache = ttnn.zeros(
            cache_shape,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        value_cache = ttnn.zeros(
            cache_shape,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.layer.attention.set_paged_kv_cache(key_cache, value_cache)
        self.page_table = ttnn.from_torch(
            torch.zeros((1, 1), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        self._draft_cache_pos = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
        )
        # Allocate before any serving trace is captured; addresses remain fixed.
        self._first_token = ttnn.from_torch(
            torch.zeros((1, 1), dtype=torch.int32),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
        )
        self._first_position = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
        )
        self._first_draft_trace = None
        self._first_draft_output = None
        self._build_draft_head(state_dict, tensor_cache_path)

    def _build_draft_head(self, state_dict, tensor_cache_path):
        """Optional compact draft LM head: BFP4 rows [0, V) plus the special-token tail.

        Drafts only need the likely next token; the target verifies every draft with its
        full head, so output is unchanged. QWEN36_MTP_DRAFT_VOCAB=V keeps the first V
        vocabulary rows (BPE ids are roughly frequency ordered) and the tile-aligned tail
        holding the tokenizer's added special tokens, cutting the draft head's weight read.
        Unset or 0: the draft shares the target LM head.
        """
        import json
        import os
        from pathlib import Path

        self.draft_vocab = int(os.environ.get("QWEN36_MTP_DRAFT_VOCAB", "0"))
        self.draft_lm_head = None
        if not self.draft_vocab:
            return
        vocab = self.args.vocab_size
        tokenizer_config = json.loads((Path(self.args.CKPT_DIR) / "tokenizer_config.json").read_text())
        special_start = min(int(k) for k in tokenizer_config["added_tokens_decoder"])
        tail_start = (special_start // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if self.draft_vocab % ttnn.TILE_SIZE or not 0 < self.draft_vocab < tail_start < vocab:
            raise ValueError(f"draft vocab {self.draft_vocab} must be tile aligned and below {tail_start}")
        # Index remap after argmax: draft columns >= V are vocabulary ids tail_start + (i - V).
        self._draft_tail_offset = tail_start - self.draft_vocab
        weight = state_dict["output.weight"]  # [vocab, dim]
        rows = torch.cat((weight[: self.draft_vocab], weight[tail_start:vocab]))
        self.draft_lm_head = ttnn.as_tensor(
            rows.T.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"mtp.draft_lm_head.v{self.draft_vocab}.t{tail_start}.bfp4")
            if tensor_cache_path
            else None,
        )
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._draft_lm_head_pc = tpc.create_matmul_1d_decode_progcfg(
            1, self.args.dim, int(self.draft_lm_head.shape[-1]), num_cores=110, grid_w=11
        )
        self._draft_lm_head_pc.in0_block_w = 2

    def _draft_logits(self, hidden):
        """Draft LM-head logits for the final-norm draft hidden row (compact head when configured)."""
        if self.draft_lm_head is None:
            return self.model._lm_head(hidden)
        return ttnn.linear(
            hidden, self.draft_lm_head, program_config=self._draft_lm_head_pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def draft_layout(self):
        """Column layout of the draft logits for the device sampler: width, the tile where the compact head's
        special-token tail starts, and that tail's vocabulary-id offset."""
        vocab = self.args.vocab_size
        if self.draft_lm_head is None:
            return dict(width=vocab, split_tile=vocab // ttnn.TILE_SIZE, id_offset=0)
        width = self.draft_vocab + (vocab - (self.draft_vocab + self._draft_tail_offset))
        return dict(width=width, split_tile=self.draft_vocab // ttnn.TILE_SIZE, id_offset=self._draft_tail_offset)

    def _draft_argmax(self, hidden):
        """Greedy draft token id (int32 [1, 1]) from the final-norm draft hidden row."""
        logits = self._draft_logits(hidden)
        logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        selected = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        out = ttnn.typecast(ttnn.reshape(selected, [1, 1]), ttnn.int32)
        ttnn.deallocate(selected)
        if self.draft_lm_head is not None:
            in_tail = ttnn.ge(out, self.draft_vocab)
            remapped = ttnn.add(out, ttnn.multiply(in_tail, self._draft_tail_offset))
            ttnn.deallocate(in_tail)
            ttnn.deallocate(out)
            out = remapped
        return out

    def sample_draft(self, hidden, position_i32, out):
        """Draw the draft token from the processed draft distribution q on device (q stays in the sampler)."""
        logits = self._draft_logits(hidden)
        layout = self.draft_layout()
        view = ttnn.reshape(logits, (1, 1, 1, layout["width"]))
        self.model.exact_sampler.draft(
            view, pos=position_i32, pos_word=0, pos_add=1, out=out,
            point_draft=self.model.mtp_point_draft, **layout,
        )
        ttnn.deallocate(logits)

    def _norm(self, state_dict, name, tensor_cache_path):
        return RMSNorm(
            device=self.device,
            dim=self.args.dim,
            state_dict=state_dict,
            weight_key=name,
            state_dict_prefix="mtp.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=self.args.norm_eps,
        )


    def reset_cache(self):
        """The fixed one-block cache is overwritten from local position zero."""

    def warmup(self):
        hidden = ttnn.zeros(
            (1, 1, self.args.dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.propose(0, hidden, 0, self.num_steps)
        ttnn.deallocate(hidden)

    def step(self, token_id, hidden, position, cache_position):
        """Predict one draft token and return (token_id, normalized hidden)."""

        token = torch.tensor([[int(token_id)]], dtype=torch.int32)
        token_tt = ttnn.from_torch(
            token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        embedding = self.model.embd(token_tt)
        ttnn.deallocate(token_tt)

        embedding_norm = self.pre_fc_norm_embedding(embedding, mode=Mode.DECODE)
        hidden_norm = self.pre_fc_norm_hidden(hidden, mode=Mode.DECODE)
        fused = ttnn.concat([embedding_norm, hidden_norm], dim=-1)
        ttnn.deallocate(embedding)
        ttnn.deallocate(embedding_norm)
        ttnn.deallocate(hidden_norm)
        x = ttnn.linear(
            fused,
            self.fc_weight,
            program_config=self.fc_decode_pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(fused)
        ttnn.synchronize_device(self.device)

        pos = torch.tensor([[int(position)]], dtype=torch.long)
        cos, sin = self.model.rope.get_rot_mats(pos)
        layer_input = x
        cache_pos = ttnn.from_torch(
            torch.full((1,), int(cache_position), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        x = self.layer.forward(
            layer_input,
            cos=cos,
            sin=sin,
            mode="decode",
            position_tensor=cache_pos,
            page_table=self.page_table,
        )
        ttnn.deallocate(layer_input)
        ttnn.deallocate(cache_pos)
        ttnn.synchronize_device(self.device)
        hidden_out = self.norm(x, mode=Mode.DECODE)
        ttnn.deallocate(x)
        ttnn.synchronize_device(self.device)
        logits = self.model._lm_head(hidden_out)
        ttnn.synchronize_device(self.device)
        host_logits = ttnn.to_torch(logits).reshape(-1, self.model.vocab_size).float()
        next_token = int(host_logits[-1].argmax().item())
        ttnn.deallocate(logits)
        return next_token, hidden_out

    def propose(self, first_token, target_hidden, first_position, num_steps=None):
        """Produce a greedy draft chain. Cache is local to this proposal."""
        self.reset_cache()
        steps = self.num_steps if num_steps is None else int(num_steps)
        drafts = []
        token = int(first_token)
        hidden = target_hidden
        owned_hidden = False
        for offset in range(steps):
            token, new_hidden = self.step(
                token, hidden, first_position + offset, offset
            )
            if owned_hidden:
                ttnn.deallocate(hidden)
            hidden = new_hidden
            owned_hidden = True
            drafts.append(token)
        if owned_hidden:
            ttnn.deallocate(hidden)
        return drafts

    def _first_draft_step(self):
        hidden = ttnn.reshape(self.model._last_hidden, (1, 32, self.args.dim))
        active = ttnn.slice(hidden, (0, 0, 0), (1, 1, self.args.dim))
        result = self.draft_device(self._first_token, active, self._first_position)
        ttnn.deallocate(active)
        return result

    def warmup_first_draft(self, capture=True):
        """Compile before any trace is parked; capture only in the second warmup phase."""
        if self._first_draft_trace is not None:
            return
        warm = self._first_draft_step()
        ttnn.deallocate(warm)
        ttnn.synchronize_device(self.device)
        if not capture:
            self._first_draft_compiled = True
            return
        trace = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._first_draft_output = self._first_draft_step()
        ttnn.end_trace_capture(self.device, trace, cq_id=0)
        self._first_draft_trace = trace
        self._first_draft_compiled = True
        ttnn.execute_trace(self.device, trace, cq_id=0, blocking=True)

    def first_draft(self, token, position):
        """Replay the first proposal against the current persistent target hidden."""
        if not getattr(self, "_first_draft_compiled", False):
            raise RuntimeError("First-draft programs must be warmed before serving")
        token_host = ttnn.from_torch(
            torch.tensor([[int(token)]], dtype=torch.int32),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        position_host = ttnn.from_torch(
            torch.tensor([int(position)], dtype=torch.int32),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(token_host, self._first_token)
        ttnn.copy_host_to_device_tensor(position_host, self._first_position)
        if self._first_draft_trace is None:
            output = self._first_draft_step()
            result = int(ttnn.to_torch(output).item())
            ttnn.deallocate(output)
            return result
        ttnn.execute_trace(self.device, self._first_draft_trace, cq_id=0, blocking=False)
        return int(ttnn.to_torch(self._first_draft_output).item())

    def release_trace(self):
        """Release captured commands before model-owned cache storage is freed."""
        if self._first_draft_trace is not None:
            ttnn.synchronize_device(self.device)
            ttnn.release_trace(self.device, self._first_draft_trace)
            self._first_draft_trace = None
        if self._first_draft_output is not None:
            ttnn.deallocate(self._first_draft_output)
            self._first_draft_output = None
        self._first_draft_compiled = False
        if getattr(self, "_first_draft_sampled_trace", None) is not None:
            ttnn.synchronize_device(self.device)
            ttnn.release_trace(self.device, self._first_draft_sampled_trace)
            self._first_draft_sampled_trace = None
        self._first_draft_sampled_compiled = False

    def draft_hidden_device(self, token_i32, hidden, position_i32, cache_position=None):
        """One exact device draft layer step: the final-norm draft hidden row for (token, hidden, position)."""
        token_u32 = ttnn.typecast(token_i32, ttnn.uint32)
        embedding = self.model.embd(token_u32)
        ttnn.deallocate(token_u32)
        embedding_norm = self._decode_norm(self.pre_fc_norm_embedding, embedding)
        hidden_norm = self._decode_norm(self.pre_fc_norm_hidden, hidden)
        fused = ttnn.concat([embedding_norm, hidden_norm], dim=-1)
        ttnn.deallocate(embedding)
        ttnn.deallocate(embedding_norm)
        ttnn.deallocate(hidden_norm)
        x = ttnn.linear(fused, self.fc_weight, program_config=self.fc_decode_pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(fused)
        idx = ttnn.reshape(ttnn.typecast(position_i32, ttnn.uint32), [1, 1])
        cos = ttnn.embedding(idx, self.model._target_cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idx, self.model._target_sin_table, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(idx)
        y = self.layer.forward(
            x, cos=cos, sin=sin, mode="decode",
            position_tensor=self._draft_cache_pos if cache_position is None else cache_position,
            page_table=self.page_table,
        )
        ttnn.deallocate(x)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        hidden_out = self._decode_norm(self.norm, y)
        ttnn.deallocate(y)
        return hidden_out

    def draft_device(self, token_i32, hidden, position_i32, cache_position=None, return_hidden=False):
        """One exact device draft step; optional local prefix position and hidden carry."""
        hidden_out = self.draft_hidden_device(token_i32, hidden, position_i32, cache_position)
        out = self._draft_argmax(hidden_out)
        if not return_hidden:
            ttnn.deallocate(hidden_out)
        return (out, hidden_out) if return_hidden else out

    # ---- Sampled first draft (exact on-device sampling; q stays in the sampler for verification) ----
    def _first_draft_sampled_step(self):
        hidden = ttnn.reshape(self.model._last_hidden, (1, 32, self.args.dim))
        active = ttnn.slice(hidden, (0, 0, 0), (1, 1, self.args.dim))
        hidden_out = self.draft_hidden_device(self._first_token, active, self._first_position)
        ttnn.deallocate(active)
        self.sample_draft(hidden_out, self._first_position, self.model.exact_sampler.first)
        ttnn.deallocate(hidden_out)

    def warmup_first_draft_sampled(self, capture=True):
        """Compile before any trace is parked; capture only in the second warmup phase."""
        if getattr(self, "_first_draft_sampled_trace", None) is not None:
            return
        self._first_draft_sampled_step()
        ttnn.synchronize_device(self.device)
        if not capture:
            self._first_draft_sampled_compiled = True
            return
        trace = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._first_draft_sampled_step()
        ttnn.end_trace_capture(self.device, trace, cq_id=0)
        self._first_draft_sampled_trace = trace
        self._first_draft_sampled_compiled = True
        ttnn.execute_trace(self.device, trace, cq_id=0, blocking=True)

    def first_draft_sampled(self, token, position):
        """Draw the first draft for (token at position) from the current persistent target hidden."""
        if not getattr(self, "_first_draft_sampled_compiled", False):
            raise RuntimeError("Sampled first-draft programs must be warmed before serving")
        token_host = ttnn.from_torch(
            torch.tensor([[int(token)]], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        position_host = ttnn.from_torch(
            torch.tensor([int(position)], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(token_host, self._first_token)
        ttnn.copy_host_to_device_tensor(position_host, self._first_position)
        if getattr(self, "_first_draft_sampled_trace", None) is None:
            self._first_draft_sampled_step()
        else:
            ttnn.execute_trace(self.device, self._first_draft_sampled_trace, cq_id=0, blocking=False)
        return int(ttnn.to_torch(self.model.exact_sampler.first).reshape(-1)[4])

    def _decode_norm(self, norm, x):
        """Width-sharded multi-core RMSNorm (the framework default runs one core, ~33 us)."""
        from models.demos.blackhole.qwen36.tt.rms_norm import get_decode_norm_config

        nc = get_decode_norm_config(self.args)
        return norm(
            ttnn.to_memory_config(x, nc["sharded_output_config"]),
            mode=Mode.DECODE,
            in_sharded=True,
            out_sharded=False,
            norm_config=nc,
        )
