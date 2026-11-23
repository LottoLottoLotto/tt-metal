# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 multi-token predictor for single-device TT speculative decode."""

import os

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
        # Draft-only precision (default BFP8; QWEN36_MTP_DRAFT_FC_DTYPE=bf16 restores the
        # lossless FC): proposals never change the verified greedy stream.
        fc_tag = os.environ.get("QWEN36_MTP_DRAFT_FC_DTYPE", "bfp8")
        fc_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[fc_tag]
        self.fc_weight = ttnn.as_tensor(
            fc_weight,
            dtype=fc_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"mtp.fc.weight.{fc_tag}")
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
            # The FC streams K=10240 beside the permanent recurrent-state pool.
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
        self._init_draft_head(state_dict, tensor_cache_path)

    # Draft proposals never change emitted tokens: verification is exact greedy
    # over the full target vocabulary. A leading-vocabulary draft head only trades
    # a few rare-token proposals for a much smaller second LM-head read.
    _SPECIAL_TAIL_START = 248032  # tile-aligned start covering every added/special id

    def _init_draft_head(self, state_dict, tensor_cache_path):
        self.draft_head_weight = None
        self._draft_vocab_main = int(os.environ.get("QWEN36_MTP_DRAFT_VOCAB", "65536"))
        if not self._draft_vocab_main:
            return
        head = state_dict["output.weight"]  # [vocab, dim]
        vocab = int(head.shape[0])
        main, tail = self._draft_vocab_main, self._SPECIAL_TAIL_START
        if main % 32 or not 0 < main < tail or vocab % 32 or vocab <= tail:
            raise ValueError("QWEN36_MTP_DRAFT_VOCAB must be a tile multiple below the special-token tail")
        subset = torch.cat([head[:main], head[tail:]], dim=0).T.contiguous()
        # Draft-only precision (default BFP4; QWEN36_MTP_DRAFT_HEAD_DTYPE=bfp8 matches the
        # target head): proposals never change the verified greedy stream.
        dtype = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[os.environ.get("QWEN36_MTP_DRAFT_HEAD_DTYPE", "bfp4")]
        name = f"mtp.draft_head.{main}.{'bfp4' if dtype == ttnn.bfloat4_b else 'bfp8'}"
        self.draft_head_weight = ttnn.as_tensor(
            subset,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / name) if tensor_cache_path else None,
        )
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self.draft_head_pc = tpc.create_matmul_1d_decode_progcfg(
            1, self.args.dim, int(subset.shape[-1]), num_cores=110, grid_w=11
        )
        self.draft_head_pc.in0_block_w = 2

    def _draft_logits(self, hidden_out):
        """Draft-head logits for the normalized MTP hidden row (compact leading vocabulary + special tail)."""
        if self.draft_head_weight is None:
            return self.model._lm_head(hidden_out)
        return ttnn.linear(
            hidden_out, self.draft_head_weight, program_config=self.draft_head_pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def draft_layout(self):
        """Column layout of the draft logits for the device sampler: width, the tile where the special-token tail
        starts, and that tail's vocabulary-id offset."""
        vocab = self.args.vocab_size
        if self.draft_head_weight is None:
            return dict(width=vocab, split_tile=vocab // ttnn.TILE_SIZE, id_offset=0)
        main, tail = self._draft_vocab_main, self._SPECIAL_TAIL_START
        return dict(width=main + vocab - tail, split_tile=main // ttnn.TILE_SIZE, id_offset=tail - main)

    def sample_draft(self, hidden_out, step, position_i32, out):
        """Draw chain step `step` from the processed draft distribution q on device (q stays in the sampler); the
        token lands in word CHAIN_WORD + step of `out` (position_i32 holds the chain's anchor position)."""
        logits = self._draft_logits(hidden_out)
        layout = self.draft_layout()
        view = ttnn.reshape(logits, (1, 1, 1, layout["width"]))
        self.model.exact_sampler.draft_step(view, step, pos=position_i32, out=out,
                                            point_draft=self.model.mtp_point_draft, **layout)
        ttnn.deallocate(logits)

    def _draft_argmax(self, hidden_out):
        """int32 [1,1] greedy draft token from the normalized MTP hidden row."""
        logits = self._draft_logits(hidden_out)
        logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        selected = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        out = ttnn.typecast(ttnn.reshape(selected, [1, 1]), ttnn.int32)
        ttnn.deallocate(selected)
        if self.draft_head_weight is None:
            return out
        # Subset column j maps to token j below main, else tail + (j - main).
        is_tail = ttnn.typecast(ttnn.ge(out, self._draft_vocab_main), ttnn.int32)
        shifted = ttnn.add(out, ttnn.multiply(is_tail, self._SPECIAL_TAIL_START - self._draft_vocab_main))
        ttnn.deallocate(out)
        ttnn.deallocate(is_tail)
        return shifted

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
        if getattr(self, "_first_sampled_trace", None) is not None:
            ttnn.synchronize_device(self.device)
            ttnn.release_trace(self.device, self._first_sampled_trace)
            self._first_sampled_trace = None
        self._first_sampled_compiled = False

    def draft_device(self, token_i32, hidden, position_i32, cache_position=None, return_hidden=False):
        """One exact device draft step; optional local prefix position and hidden carry."""
        hidden_out = self.draft_hidden_device(token_i32, hidden, position_i32, cache_position)
        out = self._draft_argmax(hidden_out)
        if not return_hidden:
            ttnn.deallocate(hidden_out)
        return (out, hidden_out) if return_hidden else out

    def draft_hidden_device(self, token_i32, hidden, position_i32, cache_position=None):
        """The normalized MTP hidden row for (token, hidden, position): one device draft layer step."""
        token_u32 = ttnn.typecast(token_i32, ttnn.uint32)
        embedding = self.model.embd(token_u32)
        ttnn.deallocate(token_u32)
        embedding_norm = self.pre_fc_norm_embedding(embedding, mode=Mode.DECODE)
        hidden_norm = self.pre_fc_norm_hidden(hidden, mode=Mode.DECODE)
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
        hidden_out = self.norm(y, mode=Mode.DECODE)
        ttnn.deallocate(y)
        return hidden_out

    # ---- Sampled draft chains (exact speculative sampling; q stays in the sampler for verification) ----
    def sample_chain(self, token_i32, hidden, anchor_position, depth, out, cache_positions=None):
        """Draw a depth-step draft chain after `token_i32` at anchor_position ([1] or [1,1] int32): step i is drawn
        from q_i at anchor_position + 1 + i and written to out word CHAIN_WORD + i. Returns nothing; the caller
        reads the chain from `out`."""
        word = self.model.exact_sampler.CHAIN_WORD
        token = token_i32
        for step in range(depth):
            shifted = None if step == 0 else ttnn.add(anchor_position, step)
            position = ttnn.reshape(anchor_position if shifted is None else shifted, [1])
            cache = None if cache_positions is None else cache_positions[step]
            hidden_out = self.draft_hidden_device(token, hidden, position, cache_position=cache)
            if shifted is not None:
                ttnn.deallocate(shifted)
            if step:
                ttnn.deallocate(hidden)
                ttnn.deallocate(token)
            self.sample_draft(hidden_out, step, anchor_position, out)
            hidden = hidden_out
            token = ttnn.slice(out, (0, word + step), (1, word + step + 1))
        ttnn.deallocate(hidden)
        ttnn.deallocate(token)

    def _first_chain_sampled_step(self, depth, cache_positions=None):
        hidden = ttnn.reshape(self.model._last_hidden, (1, 32, self.args.dim))
        active = ttnn.slice(hidden, (0, 0, 0), (1, 1, self.args.dim))
        self.sample_chain(self._first_token, active, self._first_position, depth, self.model.exact_sampler.first,
                          cache_positions)

    def warmup_first_chain_sampled(self, depth, cache_positions=None, capture=True):
        """Compile before any trace is parked; capture only in the second warmup phase."""
        if getattr(self, "_first_sampled_trace", None) is not None:
            return
        self._first_chain_sampled_step(depth, cache_positions)
        ttnn.synchronize_device(self.device)
        self._first_sampled_compiled = True
        if not capture:
            return
        trace = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._first_chain_sampled_step(depth, cache_positions)
        ttnn.end_trace_capture(self.device, trace, cq_id=0)
        self._first_sampled_trace = trace
        ttnn.execute_trace(self.device, trace, cq_id=0, blocking=True)

    def first_chain_sampled(self, token, position, depth, cache_positions=None):
        """Draw the first sampled draft chain for (token at position) from the current persistent target hidden."""
        if not getattr(self, "_first_sampled_compiled", False):
            raise RuntimeError("Sampled first-draft programs must be warmed before serving")
        token_host = ttnn.from_torch(torch.tensor([[int(token)]], dtype=torch.int32),
                                     dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        position_host = ttnn.from_torch(torch.tensor([int(position)], dtype=torch.int32),
                                        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(token_host, self._first_token)
        ttnn.copy_host_to_device_tensor(position_host, self._first_position)
        if getattr(self, "_first_sampled_trace", None) is None:
            self._first_chain_sampled_step(depth, cache_positions)
        else:
            ttnn.execute_trace(self.device, self._first_sampled_trace, cq_id=0, blocking=False)
        word = self.model.exact_sampler.CHAIN_WORD
        return [int(v) for v in ttnn.to_torch(self.model.exact_sampler.first).reshape(-1)[word:word + depth].tolist()]
