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
        key_cache = ttnn.zeros(
            cache_shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        value_cache = ttnn.zeros(
            cache_shape,
            dtype=ttnn.bfloat8_b,
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
        self._draft_cache_pos = None
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

    def warmup_first_draft(self):
        """Capture after the cycle warmup; it initializes the shared RoPE tables."""
        if self._first_draft_trace is not None:
            ttnn.release_trace(self.device, self._first_draft_trace)
            self._first_draft_trace = None
            self._first_draft_output = None
        warm = self._first_draft_step()
        ttnn.deallocate(warm)
        ttnn.synchronize_device(self.device)
        trace = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._first_draft_output = self._first_draft_step()
        ttnn.end_trace_capture(self.device, trace, cq_id=0)
        self._first_draft_trace = trace
        ttnn.execute_trace(self.device, trace, cq_id=0, blocking=True)

    def first_draft(self, token, position):
        """Replay the first proposal against the current persistent target hidden."""
        if self._first_draft_trace is None:
            raise RuntimeError("First-draft trace must be warmed before serving")
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
        ttnn.execute_trace(self.device, self._first_draft_trace, cq_id=0, blocking=False)
        return int(ttnn.to_torch(self._first_draft_output).item())

    def draft_device(self, token_i32, hidden, position_i32):
        """One draft token entirely on device: token [1,1] int32, hidden [1,1,dim] TILE,
        position [1] int32. Returns the argmax draft as int32 [1,1]. Trace-safe: the
        one-block draft KV cache is always written at local position 0."""
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
        cos = ttnn.embedding(idx, self.model._mtp_cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idx, self.model._mtp_sin_table, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(idx)
        if self._draft_cache_pos is None:
            self._draft_cache_pos = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        y = self.layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=self._draft_cache_pos, page_table=self.page_table)
        ttnn.deallocate(x)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        hidden_out = self.norm(y, mode=Mode.DECODE)
        ttnn.deallocate(y)
        logits = self.model._lm_head(hidden_out)
        ttnn.deallocate(hidden_out)
        logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        selected = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        out = ttnn.typecast(ttnn.reshape(selected, [1, 1]), ttnn.int32)
        ttnn.deallocate(selected)
        return out
