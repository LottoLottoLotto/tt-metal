# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.vision.functional import qwen3_5_vision_transformer_preprocess
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_rope_style_hf_to_meta,
    standardize_hf_keys_multimodal,
)

from . import lean
from .patch_merger import PatchMerger
from .vision_block import VisionBlock
from .vision_model_config import VisionModelArgs


class VisionTransformer(LightweightModule):
    """
    Vision Transformer model for Qwen 3 VL.
    This implements only the transformer blocks part of the vision transformer.
    Patch embedding and merging should be done outside this class.
    """

    def __init__(
        self,
        args,
        dtype,
        state_dict,
        weight_cache_path,
        tt_ccl=None,
    ):
        """
        Initialize the Vision Transformer model.

        Args:
            args (VisionModelArgs): Model arguments
            dtype (ttnn.dtype): Data type for computations
            state_dict (dict): State dictionary containing model weights
            weight_cache_path (str): Path to weight cache
            tt_ccl (TT_CCL, optional): CCL helper. The vision tower is always
                tensor-parallel, so one is constructed internally if not given.
        """
        super().__init__()
        self.args = args
        self.dtype = dtype
        self.weight_cache_path = weight_cache_path

        # The vision tower is always tensor-parallel and needs a TT_CCL on hand.
        self.tt_ccl = tt_ccl if tt_ccl is not None else TT_CCL(args.mesh_device)

        # Create transformation matrix for RoPE QK prefill
        transformation_mat_torch = get_rot_transformation_mat(
            args.head_dim
        )  # todo)) args.head_dim is ignored inside the function
        self.transformation_mats = {
            "prefill": ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=args.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(args.mesh_device),
            )
        }

        # Create vision blocks
        self.blocks = []
        for i in range(args.hf_config.vision_config.depth):
            block = VisionBlock(
                mesh_device=args.mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=self.transformation_mats,
                args=args,
                tt_ccl=self.tt_ccl,
            )
            self.blocks.append(block)

        # The Megatron-style PatchMerger consumes a fractured-along-dim=3 tensor
        # directly (no pre-merger all-gather) and produces a fractured-along-dim=3
        # output, mirroring the LLM's DistributedNorm + LMHead final stretch in
        # `tt_transformers.tt.model`.
        self.patch_merger = PatchMerger(
            mesh_device=args.mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=args.get_state_dict_prefix("PatchMerger"),
            dtype=dtype,
            tt_ccl=self.tt_ccl,
        )

    def prepare_input(self, patch_input, seq_len=None):
        """Convert a patchified torch input to a ttnn tensor
        Args:
            patch_input (torch.Tensor): Patchified input tensor
            seq_len (int): Sequence length

        Returns:
            ttnn.Tensor: Prepared input tensor
        """
        patch_seq_len, _ = patch_input.shape
        x = patch_input
        seq_len = ((patch_seq_len // 128) + 1) * 128 if seq_len is None else seq_len
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_len - patch_seq_len)).unsqueeze(0)
        x = self.args.prepare_residual_tensor_prefill(x)
        return x

    def forward(
        self,
        x,
        unpadded_seq_len,
        rot_mats,
        pad_marks=None,
    ):
        """
        Forward pass through the Vision Transformer blocks for ONE image or video frame.

        Args:
            x (ttnn.Tensor): Input tensor [batch_size, 1, seq_len, hidden_dim].
                This arrives fractured along dim=3 (the hidden dim) because
                `prepare_residual_tensor_prefill` shards at load time.
            unpadded_seq_len (int): Real sequence length before padding.
            rot_mats (list): Rotation matrices for positional embeddings.
            pad_marks (tuple): (q_mark, k_mark) from vision_pad_marks, or None when the sequence
                has no padding rows.

        Returns:
            ttnn.Tensor: Output tensor, fractured along dim=3 (each device owns
                out_hidden_size/TP), matching the LLM lm_head contract.
        """
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                rot_mats=rot_mats,
                pad_marks=pad_marks,
            )

        # The PatchMerger consumes the block output fractured along dim=3
        # directly (its first op is a DistributedLayerNorm that all-gathers
        # internally) and produces a fractured-along-dim=3 output. No
        # pre-merger all-gather is needed.
        x = x[:, :, :unpadded_seq_len, :]
        x = self.patch_merger(x)
        return x


def vision_pad_marks(unpadded_seq_len, seq_len, head_dim, padded_head_dim, mesh_device):
    """Padding mask for one segment, carried in a zero padded head-dim column (no S x S tensor).

    Returns (q_mark, k_mark), each [1, 1, seq_len, padded_head_dim] BF16 and added to every head:
    q_mark is 1 in the marker column; k_mark is -8192 there on padding rows and 0 on real rows.
    The marker column starts a 16-wide block of padding columns, so BFP8 typecasts of q/k never
    share an exponent between it and real values. None when there is no padding."""
    if unpadded_seq_len == seq_len:
        return None
    col = math.ceil(head_dim / 16) * 16
    assert col < padded_head_dim, f"no zero head-dim column for the padding mask ({head_dim} -> {padded_head_dim})"
    q = torch.zeros(1, 1, seq_len, padded_head_dim, dtype=torch.bfloat16)
    k = torch.zeros(1, 1, seq_len, padded_head_dim, dtype=torch.bfloat16)
    q[..., col] = 1.0
    k[..., unpadded_seq_len:, col] = -8192.0
    upload = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return upload(q), upload(k)


class DropInVisionTransformer(torch.nn.Module):
    """Wraps VisionTransformer to be a drop-in replacement for
    Qwen2_5_VisionTransformerPretrainedModel. It uses the reference model
    for certain preprocessing steps like patch embedding and index calculation.
    """

    def __init__(
        self,
        reference_model,
        model_args: VisionModelArgs,
        dtype=ttnn.bfloat8_b,
        debug=False,
        tt_ccl=None,
    ):
        """
        Initialize the TorchVisionTransformer wrapper.

        Args:
            reference_model (Qwen2_5_VisionTransformerPretrainedModel): Initialized reference HF model instance.
            model_args (VisionModelArgs): Model configuration arguments.
            dtype (ttnn.dtype): Compute dtype for weights.
            debug (bool): If True, run the reference path alongside and report PCC.
            tt_ccl (TT_CCL, optional): Reuse a CCL helper across multiple models.
        """
        super().__init__()
        self.reference_model = reference_model
        self.model_args = model_args
        self.debug = debug

        state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
        state_dict = convert_hf_to_meta(state_dict, model_args.head_dim)
        state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
        state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

        # Initialize TT model
        self.tt_model = VisionTransformer(
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
            tt_ccl=tt_ccl,
        )

    @property
    def dtype(self):
        return self.reference_model.dtype

    @property
    def spatial_merge_size(self):
        return self.model_args.hf_config.vision_config.spatial_merge_size

    def encode(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Run the tower over every image / video frame; returns host BF16 ``[N, out_hidden_size]``.

        Args:
            pixel_values (torch.Tensor): patchified pixels of all items, ``[num_patches, patch_dim]``.
            grid_thw (torch.Tensor): per-item grid ``(t, h, w)``, ``[num_items, 3]``.

        Patches attend only within their own image or video frame (the upstream ``cu_seqlens``
        segments), and patch embedding, position embedding, rotary and the patch merger are
        per-frame, so each frame runs as its own tower pass: exact, and no cross-frame mask.
        """
        grids = [tuple(int(v) for v in g) for g in grid_thw.reshape(-1, 3).tolist()]
        assert sum(t * h * w for t, h, w in grids) == pixel_values.shape[0], "pixel rows do not match grid_thw"
        rows, offset = [], 0
        for t, h, w in grids:
            for _ in range(t):
                rows.append(self._encode_frame(pixel_values[offset : offset + h * w], torch.tensor([[1, h, w]])))
                offset += h * w
        return torch.cat(rows, dim=0)

    def _encode_frame(self, pixel_values, grid_thw):
        """One image or video frame (grid ``[[1, h, w]]``) through the TT tower -> host ``[h*w/4, H]``."""
        mesh = self.model_args.mesh_device
        unpadded_seq_len = int(grid_thw.prod())
        # Pad to a multiple of 2048 for the upstream attention configs; the lean path only needs 128.
        seq_len = math.ceil(unpadded_seq_len / 128) * 128 if lean.LEAN else ((unpadded_seq_len // 2048) + 1) * 2048
        _, position_embeddings = qwen3_5_vision_transformer_preprocess(
            seq_len=unpadded_seq_len,
            grid_thw=grid_thw,
            head_dim=self.model_args.head_dim,
            spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
        )
        patch_input = self.reference_model.patch_embed(pixel_values)
        patch_input = patch_input + self.reference_model.fast_pos_embed_interpolate(grid_thw)

        # Rotary tables padded with the identity rotation (cos = 1, sin = 0).
        cos_orig, sin_orig = convert_rope_style_hf_to_meta(*position_embeddings)
        pad = (0, 0, 0, seq_len - unpadded_seq_len)
        rot_mats = [
            ttnn.from_torch(
                torch.nn.functional.pad(t, pad, value=v)[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for t, v in ((cos_orig, 1.0), (sin_orig, 0.0))
        ]
        attention = self.tt_model.blocks[0].attention
        pad_marks = vision_pad_marks(unpadded_seq_len, seq_len, attention.head_dim, attention.padded_head_dim, mesh)
        tt_input = self.tt_model.prepare_input(patch_input, seq_len)
        tt_out = self.tt_model(tt_input, unpadded_seq_len=unpadded_seq_len, rot_mats=rot_mats, pad_marks=pad_marks)
        ttnn.deallocate(tt_input)
        for t in rot_mats + list(pad_marks or ()):
            ttnn.deallocate(t)

        # The merger output is fractured along dim=3 on a mesh (replicated across mesh rows).
        out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=self.model_args.cluster_shape),
        )
        ttnn.deallocate(tt_out)
        out_hidden_size = self.model_args.hf_config.vision_config.out_hidden_size
        out = out[0, 0, : unpadded_seq_len // self.spatial_merge_size**2, :out_hidden_size].to(torch.bfloat16)
        if self.debug:
            reference_output = self.reference_model.forward(pixel_values, grid_thw)
            _, pcc = comp_pcc(reference_output, out)
            logger.info(f"DropInVisionTransformer: PCC to reference model: {pcc}")
        return out
