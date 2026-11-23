# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tensor-parallel ("Megatron-style") qwen35_27b vision attention.

Mirrors the LLM TP convention from `tt_transformers.tt.attention`:

  in:  replicated x_11SH (the wrapping DistributedLayerNorm produced this)
  ──▶ column-sharded W_qkv (head-fractured) ──▶ per-device n_local_heads
  ──▶ SDPA → nlp_concat_heads
  ──▶ row-sharded W_o ──▶ partial sums
  ──▶ tt_all_reduce(dim=3)  -> on T3K/QB2 this is a reduce_scatter
  out: fractured along dim=3 (each device owns dim/TP)

The fractured output then re-enters the next block's DistributedLayerNorm,
which gathers it back to replicated.
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

from . import lean


class VisionAttention(LightweightModule):
    def __init__(self, *args, **kwargs):
        kwargs["causal_mask"] = False
        self.__init(*args, **kwargs)

    def forward(
        self,
        x,
        rot_mats,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        pad_marks=None,
    ):
        return self.forward_prefill(
            x,
            rot_mats=rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=None,
            pad_marks=pad_marks,
        )

    def __init(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        causal_mask=True,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.configuration = configuration
        self.cluster_shape = configuration.cluster_shape
        # We TP across cluster axis 1.
        self.tp = self.cluster_shape[1]
        # `tt_all_reduce` for T3K/QB2 ignores the supplied cluster_axis and
        # reduce_scatters across the non-1 axis; keep this at 0 to avoid the
        # cluster_axis==1 short-circuit.
        self.ccl_cluster_axis = 0

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = paged_attention_config
        self.causal_mask = causal_mask
        self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
        self.ccl_dtype = configuration.ccl_dtype
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        self.tile_size = configuration.tile_size

        # Each device holds n_heads / tp heads.
        self.n_local_heads = self.n_heads // self.tp
        self.n_local_kv_heads = self.n_kv_heads // self.tp
        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size
        # Per-device qkv width = (n_local_heads + 2*n_local_kv_heads) * padded_head_dim
        self.local_qkv_size = (self.n_local_heads + 2 * self.n_local_kv_heads) * self.padded_head_dim

        self.dtype = dtype
        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.transformation_mats = transformation_mats
        self.decoders_optimizations = configuration.decoders_optimizations
        self.model_config = configuration.get_model_config()
        self.ccl_topology = configuration.ccl_topology()
        self.is_multichip = configuration.is_multichip
        self.activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        self.kv_cache_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.KV_CACHE
        )
        self.sdpa_prefill_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_PREFILL, configuration=configuration
        )
        self.li_qkv_prefill_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_PREFILL, configuration=configuration
        )
        self.li_o_prefill_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_PREFILL, configuration=configuration
        )
        if lean.HIFI_CFG is not None:
            self.sdpa_prefill_compute_kernel_cfg = lean.HIFI_CFG
            self.li_qkv_prefill_compute_kernel_cfg = lean.HIFI_CFG
            self.li_o_prefill_compute_kernel_cfg = lean.HIFI_CFG

        layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}.tp{self.tp}")

        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        # Initialise bias placeholders.
        self.wqkv_bias_prefill = None
        self.wo_bias_prefill = None

        # ---- wqkv weight + bias (column / head sharded) ----------------------------
        def pad_head_chunk(t, n_local, last_dim_is_in: bool):
            """Reshape [n_local*head_dim, in] (or [n_local*head_dim] for bias)
            so that head_dim can be padded to padded_head_dim, then flattened back."""
            if self.head_dim == self.padded_head_dim:
                return t
            if last_dim_is_in:  # weight chunk shape [n_local*head_dim, in]
                t = t.reshape(n_local, self.head_dim, -1)
                t = torch.nn.functional.pad(t, (0, 0, 0, self.padded_head_dim - self.head_dim))
                return t.reshape(n_local * self.padded_head_dim, -1)
            else:  # bias chunk shape [n_local*head_dim]
                t = t.reshape(n_local, self.head_dim)
                t = torch.nn.functional.pad(t, (0, self.padded_head_dim - self.head_dim))
                return t.reshape(-1)

        # Build the *full* wqkv weight tensor laid out so that consecutive blocks of
        # `local_qkv_size` columns belong to consecutive devices. Then ShardTensor2dMesh
        # along dim=-1 gives each device its own [Q_local | K_local | V_local].
        qkv_chunks = []
        for i in range(self.tp):
            wq_i = pad_head_chunk(
                torch.chunk(self.state_dict[f"{wq_str}.weight"], self.tp, dim=0)[i], self.n_local_heads, True
            )
            wk_i = pad_head_chunk(
                torch.chunk(self.state_dict[f"{wk_str}.weight"], self.tp, dim=0)[i], self.n_local_kv_heads, True
            )
            wv_i = pad_head_chunk(
                torch.chunk(self.state_dict[f"{wv_str}.weight"], self.tp, dim=0)[i], self.n_local_kv_heads, True
            )
            qkv_i = torch.cat(
                [torch.transpose(wq_i, -2, -1), torch.transpose(wk_i, -2, -1), torch.transpose(wv_i, -2, -1)], dim=-1
            )
            qkv_chunks.append(qkv_i)
        qkv_cat = torch.cat(qkv_chunks, dim=-1).unsqueeze(0).unsqueeze(0)
        # qkv_cat shape: [1, 1, dim, tp * local_qkv_size]; shard dim=-1 across cluster axis 1.
        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=lean.WEIGHT_DTYPE or ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.cluster_shape),
            cache_file_name=cache_name("wqkv_col"),
        )

        if f"{wq_str}.bias" in self.state_dict:
            bias_chunks = []
            for i in range(self.tp):
                bq_i = pad_head_chunk(
                    torch.chunk(self.state_dict[f"{wq_str}.bias"], self.tp)[i], self.n_local_heads, False
                )
                bk_i = pad_head_chunk(
                    torch.chunk(self.state_dict[f"{wk_str}.bias"], self.tp)[i], self.n_local_kv_heads, False
                )
                bv_i = pad_head_chunk(
                    torch.chunk(self.state_dict[f"{wv_str}.bias"], self.tp)[i], self.n_local_kv_heads, False
                )
                bias_chunks.append(torch.cat([bq_i, bk_i, bv_i], dim=-1))
            qkv_bias = torch.cat(bias_chunks, dim=-1)
            self.wqkv_bias_prefill = ttnn.as_tensor(
                qkv_bias,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.cluster_shape),
                dtype=lean.WEIGHT_DTYPE or ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("wqkv_bias_col"),
            )

        # ---- q_norm / k_norm -------------------------------------------------------
        # The Qwen3.5-27B vision tower does not ship q_norm/k_norm (verified via
        # state-dict keys), but we keep the no-op fallback so the forward path is
        # unchanged. If this ever needs to support q/k norm, mirror the RMSNorm
        # path with a replicated norm weight (each head uses the same head_dim weight).
        if f"{q_norm_str}.weight" in self.state_dict:
            raise NotImplementedError("VisionAttention does not yet support q_norm; add an RMSNorm here when needed.")
        if f"{k_norm_str}.weight" in self.state_dict:
            raise NotImplementedError("VisionAttention does not yet support k_norm; add an RMSNorm here when needed.")
        self.q_norm = lambda x, mode: x
        self.k_norm = lambda x, mode: x

        # ---- wo weight + bias (row sharded along contraction dim) ------------------
        pt_wo_t = self.state_dict[f"{wo_str}.weight"]  # [dim, n_heads*head_dim]
        if self.head_dim != self.padded_head_dim:
            heads = pt_wo_t.reshape(-1, self.n_heads, self.head_dim)
            heads = torch.nn.functional.pad(heads, (0, self.padded_head_dim - self.head_dim))
            pt_wo = heads.reshape(1, 1, -1, self.n_heads * self.padded_head_dim).transpose(-1, -2)
        else:
            pt_wo = pt_wo_t.transpose(-1, -2).unsqueeze(0).unsqueeze(0)
        # pt_wo shape: [1, 1, n_heads*padded_head_dim, dim]; shard dim=-2 across axis 1.
        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -2), mesh_shape=self.cluster_shape),
            cache_file_name=cache_name("wo_row"),
        )

        if f"{wo_str}.bias" in self.state_dict:
            # The block output is fractured along dim=3 (post reduce_scatter),
            # so the bias has to be fractured to match. Each device gets dim/TP
            # contiguous channels.
            self.wo_bias_prefill = ttnn.as_tensor(
                self.state_dict[f"{wo_str}.bias"],
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.cluster_shape),
                dtype=self.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("wo_bias_frac"),
            )

        self.scale = self.head_dim**-0.5

        # Per-device qkv matmul program config: each device's qkv_size is
        # `local_qkv_size`. Match the existing per-device 8x8 grid layout.
        dram_shard_grid_width = 8
        self.xqkv_prefill_progcfg = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=max(
                1,
                8 if seq_len >= self.MAX_QKV_MM_SEQ_LEN else math.ceil(seq_len / self.tile_size / 8),
            ),
            per_core_N=math.ceil(self.local_qkv_size / 32 / dram_shard_grid_width),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=seq_len <= self.MAX_QKV_MM_SEQ_LEN,
        )

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        pad_marks=None,
    ):
        if lean.LEAN and x_11SH.shape[-2] > lean.CHUNK_ROWS:
            return self._forward_row_chunks(x_11SH, rot_mats, pad_marks)
        q, k, v = self._project(x_11SH, rot_mats, pad_marks)
        attn_output_1QSD = self._attend(q, k, v)
        for t in (q, k, v):
            ttnn.deallocate(t)
        return self._output(attn_output_1QSD)

    def _forward_row_chunks(self, x_11SH, rot_mats, pad_marks):
        """Same result as one pass, in lean.CHUNK_ROWS-row pieces so device memory stays bounded for
        large images: Q/K/V are projected per row chunk, K/V are concatenated, and each Q chunk
        attends over the whole frame before its output projection."""
        seq_len, width = x_11SH.shape[-2], x_11SH.shape[-1]
        rows = lambda t, a, b: ttnn.slice(t, [0, 0, a, 0], [1, 1, b, t.shape[-1]])  # noqa: E731
        qs, ks, vs = [], [], []
        for a in range(0, seq_len, lean.CHUNK_ROWS):
            b = min(a + lean.CHUNK_ROWS, seq_len)
            rot = [rows(t, a, b) for t in rot_mats]
            marks = tuple(rows(t, a, b) for t in pad_marks) if pad_marks is not None else None
            q, k, v = self._project(ttnn.slice(x_11SH, [0, 0, a, 0], [1, 1, b, width]), rot, marks)
            for t in rot + list(marks or ()):
                ttnn.deallocate(t)
            qs.append(q)
            ks.append(k)
            vs.append(v)
        ttnn.deallocate(x_11SH)
        k = ttnn.concat(ks, dim=2)
        v = ttnn.concat(vs, dim=2)
        for t in ks + vs:
            ttnn.deallocate(t)
        outs = []
        for q in qs:
            outs.append(self._output(self._attend(q, k, v)))
            ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        out = ttnn.concat(outs, dim=2)
        for t in outs:
            ttnn.deallocate(t)
        return out

    def _project(self, x_11SH, rot_mats, pad_marks):
        """QKV projection, heads, q/k norms, rotary, padding marks and SDPA dtypes for x's rows.
        Deallocates x_11SH; returns (q, k, v) as [1, heads, rows, padded_head_dim]."""
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        # ---- QKV matmul (column / head sharded) -----------------------------------
        if lean.LEAN:
            xqkv_fused = lean.linear(
                x_11SH, self.wqkv, dtype=self.activation_dtype, compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg
            )
        else:
            if seq_len > self.MAX_QKV_MM_SEQ_LEN:
                if seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
                    raise ValueError(f"seq_len {seq_len} must be divisible by {self.MAX_QKV_MM_SEQ_LEN}")
                x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

            xqkv_fused = ttnn.linear(
                x_11SH,
                self.wqkv,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                program_config=self.xqkv_prefill_progcfg(seq_len),
            )

        if self.wqkv_bias_prefill is not None:
            xqkv_fused = xqkv_fused + self.wqkv_bias_prefill

        if seq_len > self.MAX_QKV_MM_SEQ_LEN and not lean.LEAN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

        # Each device owns local_qkv_size columns -> n_local_heads / n_local_kv_heads.
        (
            q_heads_1QSD_pre_rot,
            k_heads_1KSD_pre_rot,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode=Mode.PREFILL)
        k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode=Mode.PREFILL)

        ttnn.deallocate(xqkv_fused)

        # ---- Rotary embeddings ----------------------------------------------------
        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)

        if self.head_dim != self.padded_head_dim:
            pad_dim = lambda x, v: ttnn.pad(
                x, (x.shape[0], x.shape[1], x.shape[2], self.padded_head_dim), (0, 0, 0, 0), v
            )
            rot_mats = [pad_dim(rot_mats[0], 1.0), pad_dim(rot_mats[1], 0.0)]

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(q_heads_1QSD_pre_rot)

        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        if pad_marks is not None:
            # The sequence is padded up to a multiple of 128 and padding keys must get no weight. An
            # all-zero padded head-dim column carries that mask instead of an S x S tensor: q holds
            # 1 there, real keys 0 and padding keys -8192, so real scores are unchanged and padding
            # keys score -8192 * scale (exp underflows to 0). See vision_pad_marks.
            q_mark, k_mark = pad_marks
            q_marked = ttnn.add(q_heads_1QSD, q_mark)
            ttnn.deallocate(q_heads_1QSD)
            q_heads_1QSD = q_marked
            k_marked = ttnn.add(k_heads_1KSD, k_mark)
            ttnn.deallocate(k_heads_1KSD)
            k_heads_1KSD = k_marked

        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=lean.QKV_DTYPE or ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=lean.QKV_DTYPE or self.kv_cache_dtype)
        ttnn.deallocate(k_heads_1KSD)

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=lean.QKV_DTYPE or ttnn.bfloat8_b)
        ttnn.deallocate(v_heads_1VSD)

        return q_heads_1QSD_8b, k_heads_1KSD_8b, v_heads_1VSD_8b

    def _attend(self, q_heads_1QSD_8b, k_heads_1KSD_8b, v_heads_1VSD_8b):
        """Non-causal SDPA of q's rows over every k/v row (purely local: each device runs its own
        n_local_heads); returns [1, heads, q rows, padded_head_dim]."""
        attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_1QSD_8b,
            k_heads_1KSD_8b,
            v_heads_1VSD_8b,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
            program_config=lean.SDPA_CONFIG(self.mesh_device)
            if lean.LEAN
            else self.configuration.get_attn_sdpa_program_config(Mode.PREFILL, q_heads_1QSD_8b.shape[-2], None, None),
        )
        return ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.padded_head_dim])

    def _output(self, attn_output_1QSD):
        """Concatenate heads, output projection and all-reduce for the attention rows."""
        seq_len = attn_output_1QSD.shape[-2]

        # ---- WO matmul (row sharded along contraction) ----------------------------
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)

        if lean.LEAN:
            output_partial = lean.linear(
                attn_output_11SH, self.wo, dtype=self.activation_dtype or ttnn.bfloat8_b,
                compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
            )
            ttnn.deallocate(attn_output_11SH)
        else:
            if seq_len > 1024:
                attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

            # Each device contributes a partial sum of the full output dim.
            output_partial = ttnn.linear(
                attn_output_11SH,
                self.wo,
                compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
                dtype=self.activation_dtype or ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.model_config["VISION_WO_PREFILL_PROGCFG"](seq_len),
            )
            ttnn.deallocate(attn_output_11SH)

            if seq_len > 1024:
                output_partial = ttnn.reshape(output_partial, [1, 1, seq_len, -1])

        # On T3K/QB2 `tt_all_reduce(dim=3)` is implemented as a
        # reduce_scatter, so the result is fractured along dim=3 -- exactly
        # the block I/O contract that the LLM uses.
        output_frac = tt_all_reduce(
            output_partial,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=self.ccl_cluster_axis,
            dim=3,
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.ccl_dtype,
            topology=self.ccl_topology,
        )
        if output_frac is not output_partial:
            ttnn.deallocate(output_partial)

        # Bias is fractured along dim=3 to match.
        if self.wo_bias_prefill is not None:
            output_frac = output_frac + self.wo_bias_prefill

        return output_frac
