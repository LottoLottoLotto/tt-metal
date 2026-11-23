# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode

from . import lean
from .vision_attention import VisionAttention
from .vision_distributed_layernorm import DistributedLayerNorm
from .vision_mlp import MLP


class VisionBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        tt_ccl,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.model_config = args.get_model_config()
        self.tt_ccl = tt_ccl

        if self.tt_ccl is None:
            raise ValueError("VisionBlock requires a `tt_ccl` instance")

        self.layer_num = layer_num

        self.attention = VisionAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
        )
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
        )
        # Block I/O is fractured along dim=3, so the norms all-gather first
        # (mirrors `DistributedNorm` in the LLM).
        ln_kwargs = dict(
            device=mesh_device,
            dim=args.dim,
            eps=1e-6,  # Qwen2_5_VLVisionBlock hard-codes this
            state_dict=state_dict,
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            tt_ccl=tt_ccl,
            ccl_topology=args.ccl_topology(),
        )
        self.attention_norm = DistributedLayerNorm(
            state_dict_prefix=args.get_state_dict_prefix("norm1", layer_num),
            **ln_kwargs,
        )
        self.ff_norm = DistributedLayerNorm(
            state_dict_prefix=args.get_state_dict_prefix("norm2", layer_num),
            **ln_kwargs,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats,
        pad_marks=None,
    ) -> ttnn.Tensor:
        """Run the vision block.

        I/O contract: ``x`` is fractured along dim=3 (each device holds dim/TP),
        output is fractured along dim=3. Norms internally all-gather to a
        replicated tensor; attention/MLP end with a ``reduce_scatter`` (via
        ``tt_all_reduce``), restoring the fracture.
        """
        skip_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"VisionBlock input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        # The norm gathers along dim=3 and outputs a replicated tensor.
        attn_in = self._by_rows(self.attention_norm, x, keep_input=True)
        # Attention takes replicated input and produces a tensor fractured
        # along dim=3 (because tt_all_reduce reduce-scatters on T3K/QB2).
        attn_out = self.attention.forward(
            attn_in,
            rot_mats=rot_mats,
            pad_marks=pad_marks,
        )

        # Residual + attn_out: both fractured along dim=3.
        h = ttnn.add(x, attn_out, memory_config=skip_mem_cfg, dtype=None)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(x)
        return self._by_rows(self._feed_forward, h)

    def _feed_forward(self, h):
        ff_in = self.ff_norm(h)
        ff_out = self.feed_forward.forward(ff_in, mode=Mode.PREFILL)
        ttnn.deallocate(ff_in)
        out = ttnn.add(
            h,
            ff_out,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out

    @staticmethod
    def _by_rows(fn, x, keep_input=False):
        """fn(x) for a row-wise fn; frames longer than lean.CHUNK_ROWS run it per row chunk (bounded
        device memory) and concatenate. fn may deallocate its input; x itself is deallocated unless
        keep_input."""
        seq_len = x.shape[-2]
        if not lean.LEAN or seq_len <= lean.CHUNK_ROWS:
            return fn(x)
        outs = [
            fn(ttnn.slice(x, [0, 0, a, 0], [1, 1, min(a + lean.CHUNK_ROWS, seq_len), x.shape[-1]]))
            for a in range(0, seq_len, lean.CHUNK_ROWS)
        ]
        if not keep_input:
            ttnn.deallocate(x)
        out = ttnn.concat(outs, dim=2)
        for t in outs:
            ttnn.deallocate(t)
        return out
