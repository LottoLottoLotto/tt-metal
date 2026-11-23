# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Configuration-derived Qwen hybrid text model for Blackhole P150.

Token embeddings -> decoder layers -> RMSNorm -> LM head.
Full-attention layers own KV caches; DeltaNet layers own recurrent state.
"""

import math
import os

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup
from models.demos.blackhole.qwen36.tt.rms_norm import get_decode_norm_config
from models.demos.blackhole.qwen36.tt.packed_matmul import repack_for_worker_banks
from models.tt_transformers.tt.common import Mode, get_block_size, num_blocks_in_seq


class DeviceEmbedding:
    """Batch-one embedding with a device-resident table.

    Token lookups stay on device: the MTP cycle derives its next tokens on device, and prefill
    and decode trace capture forbid device-to-host reads. ``weight`` stays the host table for
    host-side single-row callers.
    """

    def __init__(self, mesh_device, weight):
        self.mesh_device = mesh_device
        self.weight = weight
        self.device_weight = ttnn.from_torch(
            weight.to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, token_ids):
        return ttnn.embedding(
            token_ids, self.device_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )


class Qwen36Model:
    """Qwen hybrid text LM on Blackhole P150. HF_MODEL selects the checkpoint."""

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        self.args = args
        self.device = mesh_device
        self.mesh_device = mesh_device  # Generator reads model.mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.layer_indices = getattr(args, "layer_indices", None) or list(range(args.n_layers))
        self._generic_baseline = os.environ.get("QWEN36_GENERIC_BASELINE") == "1"
        if self._generic_baseline and self.num_devices != 1:
            raise ValueError("Generic baseline currently supports a single P150 only")
        from models.demos.blackhole.qwen36.tt.gdn.gated_deltanet import _effective_native_gdn_config

        self._native_gdn_state = _effective_native_gdn_config()[0]
        self._deltanet_conv_pool = None
        # Native GDN state is allocated after all asynchronous weight uploads.
        # Reserving it here can deadlock a later DRAM transfer on P150.
        self._reserved_gdn_batch_size = int(args.max_batch_size)
        self._batched_single_device = self.num_devices == 1 and self._reserved_gdn_batch_size > 1
        self._deltanet_external_states = []
        # CCL for multi-device all-reduce; None on single device (ops no-op).
        if self.num_devices > 1:
            from models.tt_transformers.tt.ccl import TT_CCL

            self.tt_ccl = TT_CCL(mesh_device)
        else:
            self.tt_ccl = None
        self.configuration = args  # Generator reads model.configuration.max_seq_len
        self.sampling_dp = 1
        # RoPE is host-recomputed each step, so refresh all decode trace inputs.
        self._tt_vllm_always_refresh_decode_trace_inputs = True
        # Full top-k sampling needs at most 64K logits/device, so it remains
        # restricted to the certified 1x4/1x8 TP meshes. Single-device callers
        # consume local logits directly; constructing SamplingGenerator after
        # reserving recurrent L1 state can block on its redundant device buffers.
        mesh_shape = tuple(int(dim) for dim in mesh_device.shape)
        self._supports_on_device_sampling = not self._generic_baseline and (
            mesh_shape in ((1, 4), (1, 8))
            and args.vocab_size % self.num_devices == 0
            and (args.vocab_size // self.num_devices <= 64 * 1024)
        )
        if self._supports_on_device_sampling:
            from models.common.sampling.generator import SamplingGenerator

            # vocab/num_devices isn't a power of 2; the multi-device TopK kernel needs it padded.
            args.pad_logits_to_power_of_2 = True
            # force_argmax (the cheap 1-all-gather greedy path) is enabled on the base
            # SAMPLING_AG_CONFIG in model_config.py and runs IN-TRACE (faster than eager). Decode
            # bucketing is made compatible with the in-trace sampler by namespacing the sampling
            # trace per bucket width (SamplingGenerator.set_trace_bucket, driven from
            # qwen36_vllm.decode_forward) — see generator._validate_trace_inputs.
            self.sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=self.tt_ccl)
        else:
            self.sampling = None

        # Framework Embedding (mesh-aware; replicates on 1-device mesh).
        from models.tt_transformers.tt.embedding import Embedding
        if self.num_devices == 1:
            # The vocabulary table stays on device in every mode: traced prefill/decode and the
            # MTP cycle (which picks its tokens on device) cannot round-trip token ids via host.
            self.embd = DeviceEmbedding(mesh_device, state_dict["tok_embeddings.weight"])
        else:
            from models.tt_transformers.tt.embedding import Embedding

            self.embd = Embedding(
                mesh_device=mesh_device,
                args=args,
                weight_cache_path=tensor_cache_path,
                state_dict=state_dict,
                dtype=ttnn.bfloat16,
            )
        self.rope = Qwen36RoPESetup(mesh_device, args)


        # Per-request vision grid (t,h,w), stashed by get_image_features / get_video_features so the
        # prefill paths can build the multimodal 3D RoPE (M-RoPE) position ids without threading
        # grid_thw through every prefill signature. Exactly one is non-None for a multimodal request
        # (image XOR video); both None => text-only. The active one also selects which placeholder
        # token id (image_token_id vs video_token_id) the vision-splice paths look for.
        self._req_image_grid_thw = None
        self._req_video_grid_thw = None

        # Transformer layers
        logger.info(f"Loading {len(self.layer_indices)} transformer layers (indices={self.layer_indices})...")
        self.layers = []
        for i in tqdm(self.layer_indices, desc="Loading layers"):
            layer = Qwen36DecoderLayer(mesh_device, args, state_dict, i, tensor_cache_path, tt_ccl=self.tt_ccl)
            self.layers.append(layer)
        if self.num_devices == 1:
            self._deltanet_external_states = self._allocate_deltanet_state_buffers(
                self._reserved_gdn_batch_size
            )
        if self.num_devices == 1:
            gdn_state_index = 0
            for layer in self.layers:
                if not layer.is_full_attention:
                    bind_state = (
                        layer.attention.set_batched_external_state
                        if self._batched_single_device else layer.attention.set_external_state
                    )
                    bind_state(*self._deltanet_external_states[gdn_state_index])
                    gdn_state_index += 1
            assert gdn_state_index == len(self._deltanet_external_states), (
                f"GDN layer/state count mismatch: layers={gdn_state_index}, "
                f"states={len(self._deltanet_external_states)}"
            )

        # Framework RMSNorm (add_unit_offset=True). Single device: is_distributed=None.
        # 27B TP: hidden is sharded -> pass is_distributed + tt_ccl or use DistributedNorm.
        self.norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key="norm",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=self.tt_ccl)
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            # TP: DistributedNorm all-gathers fractured hidden for LM head.
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            self.norm = DistributedNorm(self.norm, args, tt_ccl=self.tt_ccl, TG=args.is_galaxy)

        # LM head [in,out]. Mesh: vocab-sharded (dim=-1); _lm_head all-gathers logits.
        # M=1 decode is weight-read-bound (~1.3GB/token), so sharding cuts bandwidth;
        # gather moves only the logit row. REPLICATED fallback if vocab indivisible.
        lm_head_weight = state_dict["output.weight"].T.contiguous()  # [dim, vocab_size]
        self._lmhead_vocab_sharded = self.num_devices > 1 and lm_head_weight.shape[-1] % self.num_devices == 0
        if self.num_devices > 1 and not self._lmhead_vocab_sharded:
            logger.warning(
                f"LM-head vocab {lm_head_weight.shape[-1]} not divisible by num_devices "
                f"{self.num_devices}; falling back to replicated LM head."
            )
        lm_head_bfp4 = self.num_devices == 1 and os.environ.get("QWEN36_BFP4_LM_HEAD") == "1"
        from models.demos.blackhole.qwen36.tt.model_config import evaluation_weight_dtype

        lm_head_dtype = evaluation_weight_dtype(
            "lm_head", tensor_cache_path, ttnn.bfloat4_b if lm_head_bfp4 else ttnn.bfloat8_b
        )
        lm_head_bfp4 = lm_head_dtype == ttnn.bfloat4_b
        # 35B (dim 2048, K=64 tiles): bank-packed weight read by PackedLMHead (the C++ packed path admits only
        # K=128/160); bitwise the stock decode LM head.
        self._lmhead_packed_generic = (
            args.dim == 2048 and os.environ.get("QWEN36_SINGLE_1D_DECODE") == "1"
            and os.environ.get("QWEN36_LM_HEAD_PACKED") != "0"
        )
        self._lmhead_packed = (
            (os.environ.get("QWEN36_LM_HEAD_PACKED") == "1" or self._lmhead_packed_generic)
            and lm_head_dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b)
            and self.num_devices == 1
        )
        self._lmhead_packed_generic = self._lmhead_packed_generic and self._lmhead_packed
        self._lmhead_packed_runner = None
        if self._lmhead_packed:
            lm_head_weight = repack_for_worker_banks(lm_head_weight)
        lm_cache_suffix = (
            f".{'bfp4' if lm_head_bfp4 else 'bfp8'}.lmhead_packed_g4_k{args.dim // 32}"
            if self._lmhead_packed else (".bfp4" if lm_head_bfp4 else "")
        )
        if self._lmhead_vocab_sharded:
            # Separate cache (.vshard): as_tensor ignores mesh_mapper on reload.
            lm_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
            lm_cache = tensor_cache_path / f"output.weight.vshard{lm_cache_suffix}" if tensor_cache_path else None
        else:
            lm_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.num_devices > 1 else None
            lm_cache = tensor_cache_path / f"output.weight{lm_cache_suffix}" if tensor_cache_path else None
        if self.num_devices == 1:
            # The P150 DMA staging path cannot retire the monolithic ~500 MiB
            # vocabulary matrix in one host transfer. Upload bounded blocks, then
            # join them on device so decode runs one tuned matmul.
            part_count = max(1, (lm_head_weight.numel() + 7_999_999) // 8_000_000)
            # Whole-tile chunk widths: concat of block-float parts requires tile-aligned
            # inputs (248320 / 64 = 3880 is not; BFP8 only worked via its BF16 fallback).
            tile_cols = lm_head_weight.shape[-1] // ttnn.TILE_SIZE
            chunk = ((tile_cols + part_count - 1) // part_count) * ttnn.TILE_SIZE
            parts = []
            for index, part in enumerate(torch.split(lm_head_weight, chunk, dim=-1)):
                host = ttnn.as_tensor(
                    part.contiguous(),
                    dtype=lm_head_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    cache_file_name=(tensor_cache_path / f"output.weight.tpart{chunk // ttnn.TILE_SIZE}_{index}{lm_cache_suffix}")
                    if tensor_cache_path else None,
                )
                parts.append(ttnn.to_device(host, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                ttnn.synchronize_device(mesh_device, cq_id=0)
            joined = ttnn.concat(parts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for part in parts:
                ttnn.deallocate(part)
            # Many-input concat returns BF16 for block-float inputs; restore the
            # stored precision so decode reads half the bytes.
            if joined.dtype != lm_head_dtype:
                self.lm_head_weight = ttnn.typecast(joined, lm_head_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(joined)
            else:
                self.lm_head_weight = joined
        else:
            self.lm_head_weight = ttnn.as_tensor(
                lm_head_weight,
                dtype=lm_head_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=lm_cache,
                **(dict(mesh_mapper=lm_mapper) if lm_mapper is not None else {}),
            )
        self._lmhead_decode_pc = None
        self._lmhead_prefill_configs = {}
        if (
            self.num_devices == 1
            and os.environ.get("QWEN36_SINGLE_1D_DECODE") == "1"
        ):
            if self._lmhead_packed:
                self._lmhead_decode_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(11, 10),
                    in0_block_w=1 if args.dim == 5120 else 2,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=71,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                if self._lmhead_packed_generic:
                    from models.demos.blackhole.qwen36.tt.packed_matmul import PackedDecodeMatmul

                    # The stock call's default compute config for this BFP8 x BF16 matmul.
                    self._lmhead_packed_runner = PackedDecodeMatmul(
                        self.lm_head_weight, self._lmhead_decode_pc,
                        ttnn.WormholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                            fp32_dest_acc_en=False, packer_l1_acc=True,
                        ),
                        cb_slots=3, outstanding=2, dual_noc=True,
                    )
            else:
                from models.demos.blackhole.qwen36.tt import tp_common as tpc

                self._lmhead_decode_pc = tpc.create_matmul_1d_decode_progcfg(
                    1,
                    self.lm_head_weight.shape[-2],
                    self.lm_head_weight.shape[-1],
                    num_cores=int(os.environ.get("QWEN36_LM_HEAD_CORES", "110")),
                    grid_w=int(os.environ.get("QWEN36_DECODE_GRID_W", "11")),
                )
                # Bound BF16 LM-head circular buffers; use the same block size for all generic precisions.
                self._lmhead_decode_pc.in0_block_w = 2

        self.vocab_size = args.vocab_size
        # True: return pre-gather vocab-sharded logits for per-shard argmax + host combine.
        self._ondev_argmax = False
        self._fuse_final_residual_norm = (
            os.environ.get("QWEN36_FUSED_FINAL_RESIDUAL_NORM") == "1"
        )
        self._paged_kv_caches = None
        # Positions in self.layers of full-attn layers (not checkpoint indices); drives KV cache bind.
        self._attention_layer_indices = [pos for pos, layer in enumerate(self.layers) if layer.is_full_attention]
        # Batch-one GDN buffers were reserved before model construction and stay L1-resident.
        # Shared zero buffers for in-place DN reset between traced replays.
        self._dn_zero_recurrent = None
        self._dn_zero_conv = None
        self._dn_zero_buffers = {}
        self._gdn_remap_scratch = None
        # Chunk-outer trace: one all-layer chunk captured, replayed per chunk via DMA inputs.
        # Persistent buffers below; addresses baked into trace.
        self._chunked_trace_id = None
        self._chunked_trace_output = None
        self._chunked_chunk_size = None
        self._chunk_token_buf = None
        self._chunk_start_idx_tensor = None
        self._chunk_page_table_buf = None
        self._chunk_full_page_table_buf = None
        self._chunk_cos_buf = None
        self._chunk_sin_buf = None
        # Traced batched short-prompt (bucket) prefill: one B=1 full-bucket trace replayed
        # once per user (see capture_prefill_trace_bucket / prefill_traced_bucket_batched).
        self._bucket_trace_id = None
        self._bucket_trace_output = None
        self._bucket_size = None
        self._bucket_token_buf = None
        self._bucket_start_idx_tensor = None
        self._bucket_page_table_buf = None
        self._bucket_full_page_table_buf = None
        self._bucket_cos_buf = None
        self._bucket_sin_buf = None
        self._gdn_batched_prev = None  # batched GDN bindings saved during bucket-trace capture
        # Persistent B=1 GDN prefill scratch (batched serving): allocated once at warmup, its buffer
        # addresses are baked into the chunk-prefill trace and reused by every prefill_paged_slots
        # replay, so it is never freed/reallocated (only zeroed in place). See _bind_gdn_prefill_scratch.
        self._gdn_prefill_scratch = None
        # Packed fresh-prompt prefill (batched single device): per-group-size G-row GDN state
        # and zero sources, allocated by warmup_prefill_packed; packing is only planned for the
        # (bucket, G) shapes that warmup compiled.
        self._gdn_packed_scratch = None
        self._packed_prefill_warmed = set()

        # Optional vision tower (DropInVisionTransformer), attached lazily by
        # init_vision_model() for the multimodal serving path. None on the text-only path.
        self.vision_model = None
        self.vision_args = None

        # Trace-safe vision splice (traced serving path). The chunk/masked-bucket forwards run a
        # FIXED-shape ttnn.where(mask, vision, text) over these persistent buffers — compiled once
        # at warmup, then updated per request via copy_host_to_device, so no per-request program
        # ever compiles to clobber a parked trace. Allocated (single device only) in
        # capture_prefill_trace_chunked; None means "no traced path" -> the where is skipped.
        self._vis_buf = None  # [1, chunk_size, dim] bf16, image rows placed at their positions
        self._vis_mask_buf = None  # [1, chunk_size, 1] bf16, 1 at image positions else 0
        self._vis_zero_mask_host = None  # cached host zero mask for the clear (text/tail) path
        self._last_hidden = None
        # Exact on-device vLLM sampling (device_sampling.py): allocated before any trace by init_exact_sampler.
        self.exact_sampler = None
        # Sampled MTP draft: "sampled" draws x ~ q; "point" uses the processed draft argmax as a point-mass q.
        self.mtp_point_draft = os.environ.get("QWEN36_MTP_DRAFT_MODE", "sampled") == "point"
        self.mtp = None
        if os.environ.get("QWEN36_MTP") == "1":
            self.mtp_depth = int(os.environ.get("QWEN36_MTP_DEPTH", "1"))
            if self.mtp_depth not in (1, 2, 3):
                raise ValueError("Serving MTP depth must be 1, 2, or 3")
            if args.max_batch_size != 1:
                raise ValueError("Native MTP requires a single active sequence")
            from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP

            self.mtp = Qwen36MTP(self, state_dict, tensor_cache_path)

    def init_vision_model(self, reference_visual=None, vision_args=None, dtype=ttnn.bfloat8_b, debug=False):
        """Build and attach the TT vision tower (DropInVisionTransformer).

        The vision tower runs on the SAME mesh as the text model. It still needs the HF
        reference visual for the patch embed / positional-interpolation steps that are not
        ported to TT; if ``reference_visual`` is not supplied it is loaded here via
        ``VisionModelArgs.reference_vision_model``. Idempotent — returns the existing tower
        if already built.

        Args:
            reference_visual: HF ``model.model.visual`` to wrap. Loaded internally if None.
            vision_args (VisionModelArgs): vision config on this mesh. Built internally if None.
            dtype (ttnn.dtype): compute dtype for the vision weights.
            debug (bool): run the reference vision path alongside and log PCC.

        Returns:
            DropInVisionTransformer: the attached vision tower.
        """
        if self.vision_model is not None:
            return self.vision_model
        from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
        from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs

        if vision_args is None:
            vision_args = VisionModelArgs(
                self.mesh_device,
                max_batch_size=self.args.max_batch_size,
                max_seq_len=self.args.max_seq_len,
            )
        if reference_visual is None:
            reference_visual = vision_args.reference_vision_model(depth=vision_args.hf_config.vision_config.depth)
        self.vision_args = vision_args
        self.vision_model = DropInVisionTransformer(reference_visual, vision_args, dtype=dtype, debug=debug)
        return self.vision_model

    def get_image_features(self, pixel_values, image_grid_thw):
        """Run the vision tower over a single user's images.

        Mirrors the HF reference's ``get_image_features`` seam: pixel patches in, packed
        image embeddings out — one row per image-placeholder token, ready to be spliced
        into the text embeddings by ``_scatter_vision_tokens``.

        Args:
            pixel_values (torch.Tensor): patchified pixels ``[num_patches, patch_dim]``.
            image_grid_thw (torch.Tensor): per-image grid ``(t, h, w)``, ``[num_images, 3]``.

        Returns:
            ttnn.Tensor: ``[num_image_tokens, H]`` image embeddings, hidden-fractured along
            the last dim on a mesh (same sharding as the text embeddings).
        """
        assert self.vision_model is not None, "init_vision_model() must be called before get_image_features()"
        # Stash the grid (as an IMAGE grid) so the prefill paths can build M-RoPE position ids for
        # this request (the splice positions in input_ids + the (t,h,w) grid are all M-RoPE needs).
        # Clear any stale video grid so the modality (and thus the placeholder token id) is image.
        self._req_image_grid_thw = image_grid_thw
        self._req_video_grid_thw = None
        image_features = self.vision_model.forward(pixel_values, grid_thw=image_grid_thw)
        # The vision tower returns [1, B, S, H]; flatten the leading (batch/seq) dims to the
        # packed [num_image_tokens, H] rows the text-model splice (_scatter_vision_tokens /
        # _set_vision_merge) expects. The hidden dim is unchanged so the mesh hidden-fracture
        # is preserved. B == 1 for now.
        hidden = image_features.shape[-1]
        return ttnn.reshape(image_features, (-1, hidden))

    def get_video_features(self, pixel_values_videos, video_grid_thw):
        """Run the vision tower over a single user's video frames.

        Mirrors the HF reference's ``get_video_features`` seam, which is just ``get_image_features``
        on the video pixels/grid — the vision tower forward is identical for image and video. The
        only differences are downstream: M-RoPE treats the grid as a VIDEO grid (split per frame by
        timestamps, modality==2), and the embeddings splice into ``video_token_id`` placeholders
        rather than ``image_token_id``. Both are selected by stashing the grid here as a video grid.

        Args:
            pixel_values_videos (torch.Tensor): patchified video pixels ``[num_patches, patch_dim]``.
            video_grid_thw (torch.Tensor): per-video grid ``(t, h, w)``, ``[num_videos, 3]``.

        Returns:
            ttnn.Tensor: ``[num_video_tokens, H]`` video embeddings, hidden-fractured along the last
            dim on a mesh (same sharding as the text embeddings).
        """
        assert self.vision_model is not None, "init_vision_model() must be called before get_video_features()"
        # Stash the grid as a VIDEO grid; clear any stale image grid so the modality (and the
        # placeholder token id) is video.
        self._req_video_grid_thw = video_grid_thw
        self._req_image_grid_thw = None
        video_features = self.vision_model.forward(pixel_values_videos, grid_thw=video_grid_thw)
        hidden = video_features.shape[-1]
        return ttnn.reshape(video_features, (-1, hidden))

    def _vision_placeholder_token_id(self):
        """The input-id the current request's vision embeddings splice into: ``video_token_id`` for
        a video request (video grid stashed), else ``image_token_id``. The vision-splice paths
        (_scatter_vision_tokens / _set_vision_merge / _vis_row_offset_for) use this to locate the
        placeholder positions, mirroring HF's ``input_ids == image_token_id`` /
        ``input_ids == video_token_id`` masks."""
        if self._req_video_grid_thw is not None:
            return int(self.args.hf_config.video_token_id)
        return int(self.args.hf_config.image_token_id)

    def _build_request_rope(self, token_ids, vision_tokens):
        """Stage the per-request RoPE for this prefill: M-RoPE (3D position ids + rope_delta) when
        the request is multimodal (vision_tokens present -> use the grid stashed by
        get_image_features / get_video_features), else clear to ordinary 1D RoPE. Call once at a
        prefill entry point with the REAL token ids (token_ids[:, :actual_len]); the chunk/tail
        seams then slice the staged table by sequence position and decode offsets by rope_delta."""
        image_grid = self._req_image_grid_thw if vision_tokens is not None else None
        video_grid = self._req_video_grid_thw if vision_tokens is not None else None
        self.rope.build_request_rope(token_ids, image_grid_thw=image_grid, video_grid_thw=video_grid)

    def _alloc_vision_merge_buffers(self, device, chunk_size):
        """Allocate the persistent vision-splice buffers used by the traced prefill path.

        ``_vis_buf`` holds the image embeddings placed at their token positions (zeros elsewhere);
        ``_vis_mask_buf`` is the 0/1 image mask. Both are zero-initialised, so the ttnn.where baked
        into the captured forward is the identity until a real multimodal request stages them.
        Allocating before warmup means the where compiles in the warmup pass (and is then
        captured), never at request time.

        Shapes/sharding match the activations of the forward that consumes them:
          - single device: vis [1, chunk_size, dim], mask [1, chunk_size, 1] (the 3D embd output);
          - TP: vis [1, 1, chunk_size, dim] HIDDEN-SHARDED across the mesh exactly like embd
            fractures its output (ShardTensor2dMesh dims=(None, -1)), so each device's where sees
            its own [.., dim/TP] vision columns; mask [1, 1, chunk_size, 1] REPLICATED (it
            broadcasts over the sharded hidden dim). DropInVisionTransformer fractures its output
            the same way, so the per-device columns line up.
        """
        if self._vis_buf is not None:
            return
        H = self.args.dim
        if self.num_devices > 1:
            shard = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.args.cluster_shape)
            rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
            self._vis_buf = ttnn.from_torch(
                torch.zeros(1, 1, chunk_size, H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard,
            )
            self._vis_mask_buf = ttnn.from_torch(
                torch.zeros(1, 1, chunk_size, 1, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            self._vis_zero_mask_host = ttnn.from_torch(
                torch.zeros(1, 1, chunk_size, 1, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            return
        self._vis_buf = ttnn.from_torch(
            torch.zeros(1, chunk_size, H, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._vis_mask_buf = ttnn.from_torch(
            torch.zeros(1, chunk_size, 1, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._vis_zero_mask_host = ttnn.from_torch(
            torch.zeros(1, chunk_size, 1, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def _apply_vision_merge(self, x, length):
        """Final text-vs-vision SELECTION of the trace-safe splice: where(mask, vision, x).

        This is NOT a scatter and is not a substitute for one. The scatter — placing the n
        (variable) vision rows at their n image positions — already happened on host in
        _set_vision_merge, which is why ``_vis_buf`` here is a FULL [1, chunk_size, dim] buffer
        (image rows at their positions, zeros elsewhere), the same length as ``x``, NOT the raw
        [n, dim] vision tensor. So n << seq_len (a few image tokens in a long text prompt) is fine:
        the [.,1] mask broadcasts over the hidden dim and is 1 only at those n positions, so
        out == vision there and out == x (text) everywhere else (mask==0 is the exact identity).

        The placement is forced onto the host because aligning a variable n to fixed positions is
        inherently variable-shape; any on-device form (ttnn.scatter / pad / slice-copy) recompiles
        per request and would clobber the parked trace. ``length`` selects the segment (chunk_size
        for a full chunk, bucket for the masked path); the buffers slice down to it. No-op when the
        buffers are unallocated (text-only / non-traced deployment, which uses the device scatter)."""
        if self._vis_buf is None:
            return x
        if self._vis_buf.shape.rank == 4:
            # TP: buffers are [1, 1, chunk_size, dim(/TP)], matching the 4D TP activations.
            full = self._vis_buf.shape[2] == length
            mask = self._vis_mask_buf if full else self._vis_mask_buf[:, :, :length, :]
            vis = self._vis_buf if full else self._vis_buf[:, :, :length, :]
        else:
            full = self._vis_buf.shape[1] == length
            mask = self._vis_mask_buf if full else self._vis_mask_buf[:, :length, :]
            vis = self._vis_buf if full else self._vis_buf[:, :length, :]
        out = ttnn.where(mask, vis, x)
        ttnn.deallocate(x)
        return out

    def _set_vision_merge(self, ids_host, vision_tokens, vis_row_offset=0):
        """Stage the persistent vision buffers for the next forward (host -> device copy only;
        no program compiles). ``vision_tokens`` None clears the mask (the where becomes identity,
        for text-only requests); otherwise the packed image rows are read back to host, placed at
        their token positions in a zero [1, chunk_size, dim] buffer, and uploaded along with the
        0/1 mask. ``ids_host`` is the segment's token ids (torch), used to locate the image
        placeholders (== hf_config.image_token_id).

        ``vis_row_offset`` is the number of image-placeholder tokens that appear in the prompt
        BEFORE this segment, i.e. the index of the first packed vision row belonging to it. A
        large image whose placeholders span multiple prefill chunks (or spill into the tail) is
        thus spliced correctly: each segment consumes its own slice
        ``vis_host[vis_row_offset : vis_row_offset + n]`` of the packed rows. A segment with no
        image placeholders (text-only chunk / tail) clears the mask (identity merge)."""
        if self._vis_buf is None:
            # Fail loudly rather than silently drop the image: a multimodal request must run on a
            # path with the trace-safe buffers (capture_prefill_trace_chunked, single device) or
            # the on-device scatter (non-traced prefill_paged).
            assert vision_tokens is None, "vision merge requested but the trace-safe buffers are not allocated"
            return
        if vision_tokens is None:
            ttnn.copy_host_to_device_tensor(self._vis_zero_mask_host, self._vis_mask_buf)
            return
        tp = self.num_devices > 1
        cs = self._vis_buf.shape[-2]  # seq dim: dim 1 (3D single) / dim 2 (4D TP)
        Hg = self.args.dim  # global hidden (the buffer's last dim is dim/TP on a mesh)
        flat = ids_host.reshape(-1)
        pos = torch.nonzero(flat[:cs] == self._vision_placeholder_token_id(), as_tuple=False).reshape(-1)
        n = int(pos.numel())
        # No image placeholders in this segment (text-only chunk, or a tail that holds none of the
        # image rows): the merge is the identity, so just clear the mask.
        if n == 0:
            ttnn.copy_host_to_device_tensor(self._vis_zero_mask_host, self._vis_mask_buf)
            return
        assert vis_row_offset + n <= int(vision_tokens.shape[0]), (
            f"vision splice out of range: row offset {vis_row_offset} + {n} image positions in this "
            f"segment exceeds {int(vision_tokens.shape[0])} packed vision rows"
        )
        # Gather the (hidden-fractured on a mesh) vision rows to full [num_image_tokens, Hg] on
        # host, then take this segment's slice. The placement is along the SEQ dim, orthogonal to
        # the hidden fracture, so the round-trip gather->place->reshard preserves the per-device
        # columns. ConcatMeshToTensor(dim=1) over the 2D [rows, dim/TP] is the inverse of the
        # dims=(None,-1) hidden shard used on re-upload.
        if tp:
            vis_host = ttnn.to_torch(vision_tokens, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=1)).to(
                torch.bfloat16
            )
        else:
            vis_host = ttnn.to_torch(vision_tokens).to(torch.bfloat16)  # [num_image_tokens, Hg]
        seg = vis_host[vis_row_offset : vis_row_offset + n]
        if tp:
            shard = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.args.cluster_shape)
            rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
            vis_full = torch.zeros(1, 1, cs, Hg, dtype=torch.bfloat16)
            vis_full[0, 0, pos] = seg
            mask = torch.zeros(1, 1, cs, 1, dtype=torch.bfloat16)
            mask[0, 0, pos, 0] = 1.0
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(vis_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=shard),
                self._vis_buf,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep),
                self._vis_mask_buf,
            )
            return
        vis_full = torch.zeros(1, cs, Hg, dtype=torch.bfloat16)
        vis_full[0, pos] = seg
        mask = torch.zeros(1, cs, 1, dtype=torch.bfloat16)
        mask[0, pos, 0] = 1.0
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(vis_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._vis_buf
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._vis_mask_buf
        )

    def _vis_row_offset_for(self, token_ids, chunk_start):
        """Packed-vision-row offset for the prefill segment starting at absolute position
        ``chunk_start``: the number of image-placeholder tokens before it. The vision rows are
        packed in image-placeholder order, so this is the index of the first row this segment
        owns — used to splice a large image whose placeholders span multiple chunks / the tail."""
        if chunk_start <= 0:
            return 0
        return int((token_ids[:, :chunk_start] == self._vision_placeholder_token_id()).sum())

    def switch_mode(self, mode):
        """Generator mode-change hook; no-op (no prefetcher)."""
        return None

    def _lm_head_linear(self, x):
        original_shape = list(x.shape)
        rows = 1
        for dim in original_shape[:-1]:
            rows *= int(dim)
        if self._lmhead_packed_runner is not None:
            if rows > 32:
                raise ValueError("The K=64 packed LM head serves at most 32 rows per call")
            if rows == 1:
                return self._lmhead_packed_runner(x)
            logits = self._lmhead_packed_runner(ttnn.reshape(x, [1, 1, rows, original_shape[-1]]))
            return ttnn.reshape(logits, original_shape[:-1] + [logits.shape[-1]])
        if self._lmhead_packed:
            program_config = self._lmhead_decode_pc
            if rows > 32:
                program_config = self._lmhead_prefill_configs.get(rows)
                if program_config is None:
                    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(11, 10),
                        in0_block_w=4,
                        out_subblock_h=1,
                        out_subblock_w=2,
                        per_core_M=max(1, math.ceil(math.ceil(rows / 32) / 110)),
                        per_core_N=7760,
                        out_block_h=1,
                        out_block_w=8,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=False,
                    )
                    self._lmhead_prefill_configs[rows] = program_config
            packed_input = ttnn.reshape(x, [1, 1, rows, original_shape[-1]])
            logits = ttnn.linear(
                packed_input, self.lm_head_weight, program_config=program_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.reshape(logits, original_shape[:-1] + [logits.shape[-1]])
        if self.num_devices == 1 and rows > 1:
            x = ttnn.reshape(x, [1, 1, rows, original_shape[-1]])
            logits = ttnn.linear(x, self.lm_head_weight, program_config=self._lmhead_decode_pc)
            return ttnn.reshape(logits, original_shape[:-1] + [logits.shape[-1]])
        program_config = self._lmhead_decode_pc if x.shape[-2] <= ttnn.TILE_SIZE else None
        return ttnn.linear(x, self.lm_head_weight, program_config=program_config)

    def _lm_head(self, x):
        """LM-head matmul. Vocab-sharded mesh: partial logits + all-gather to full replicated.
        Single device: plain matmul."""
        logits = self._lm_head_linear(x)
        if self._lmhead_vocab_sharded:
            from models.tt_transformers.tt.ccl import tt_all_gather

            logits = tt_all_gather(
                logits,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=None,
                dim=len(logits.shape) - 1,
                topology=self.args.ccl_topology(),
            )
        return logits

    def _final_norm_residual_decode(self, x, residual):
        """Fuse the final decoder residual add into the model RMSNorm."""
        if os.environ.get("QWEN36_SHARDED_FINAL_NORM") == "1" or os.environ.get("QWEN36_OPTIMIZED_27B") == "1":
            nc = get_decode_norm_config(self.args)
            shard_config = nc["sharded_output_config"]
            original_shape = list(x.shape)
            rows = 1
            for dim in original_shape[:-1]:
                rows *= int(dim)
            if rows > 1:
                x = ttnn.reshape(x, [1, 1, rows, original_shape[-1]])
                residual = ttnn.reshape(residual, [1, 1, rows, original_shape[-1]])
            x_sharded = ttnn.to_memory_config(x, shard_config)
            residual_sharded = ttnn.to_memory_config(residual, shard_config)
            output_sharded = ttnn.rms_norm(
                x_sharded,
                residual_input_tensor=residual_sharded,
                epsilon=self.norm.eps,
                weight=self.norm.weight,
                program_config=nc["sharded_program_config"],
                memory_config=shard_config,
                compute_kernel_config=self.norm.compute_kernel_config_hifi2,
            )
            ttnn.deallocate(x_sharded)
            ttnn.deallocate(residual_sharded)
            output = ttnn.sharded_to_interleaved(output_sharded)
            if rows > 1:
                output = ttnn.reshape(output, original_shape)
            ttnn.deallocate(output_sharded)
            return output
        return ttnn.rms_norm(
            x,
            residual_input_tensor=residual,
            epsilon=self.norm.eps,
            weight=self.norm.weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.norm.compute_kernel_config_hifi2,
        )

    def verify_final_norm(self, layer_index):
        """Final-norm hook for a verification layer pass: the last layer of a fused-final-norm model
        applies decode's fused residual+RMSNorm (same program, row-local), so every verification row
        reaches the LM head with the bits a single-row decode produces. None elsewhere."""
        if self._fuse_final_residual_norm and layer_index == len(self.layers) - 1:
            return self._final_norm_residual_decode
        return None

    def verify_hidden(self, x):
        """LM-head input from the last verification layer output (see verify_final_norm)."""
        return x if self._fuse_final_residual_norm else self._final_norm_decode(x)

    def _final_norm_decode(self, x):
        """Final RMSNorm before the LM head (TP decode).

        The bare `self.norm(x, DECODE)` runs plain ttnn.rms_norm on a DRAM-interleaved [32,dim]
        tensor -> single tile-row -> 1 core (~80us/token). Passing the framework's 'lm_head' norm
        config runs the sharded multi-core norm across lm_head_core_grid instead; output_mem_config
        is forced back to DRAM so the LM-head matmul input is byte-identical (layout-only change).
        """
        if self.num_devices == 1 and os.environ.get("QWEN36_OPTIMIZED_27B") == "1":
            # Bounded 8x5 width-sharded norm for M=1 and the M=2 MTP verify row pair alike.
            from models.demos.blackhole.qwen36.tt.rms_norm import bounded_rms_norm

            return bounded_rms_norm(x, self.norm.weight, self.norm.eps, self.norm.compute_kernel_config_hifi2)
        if self.num_devices == 1 and os.environ.get("QWEN36_SHARDED_FINAL_NORM") == "1":
            nc = get_decode_norm_config(self.args)
            x_sharded = ttnn.to_memory_config(x, nc["sharded_output_config"])
            return self.norm(
                x_sharded,
                mode=Mode.DECODE,
                in_sharded=True,
                out_sharded=False,
                norm_config=nc,
            )
        if self.num_devices > 1:
            nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
            nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
            return self.norm(x, mode=Mode.DECODE, norm_config=nc)
        return self.norm(x, mode=Mode.DECODE)

    @classmethod
    def from_pretrained(
        cls, device, max_batch_size=1, max_seq_len=2048, n_layers=None, layer_indices=None, hf_model=None
    ):
        # HF_MODEL env var (hub or local path) is canonical; hf_model sets it for back-compat.
        if hf_model is not None:
            import os

            os.environ["HF_MODEL"] = hf_model

        args = Qwen36ModelArgs(
            mesh_device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # layer_indices: run only these checkpoint layers (e.g. [0,3,31]) for profiling.
        # Each keeps its real type via full attention_type_list. Overrides n_layers truncation.
        if layer_indices is not None:
            layer_indices = list(layer_indices)
            assert layer_indices, "layer_indices must be non-empty"
            assert all(
                0 <= i < len(args.attention_type_list) for i in layer_indices
            ), f"layer_indices {layer_indices} out of range [0, {len(args.attention_type_list)})"
            args.layer_indices = layer_indices
            args.n_layers = len(layer_indices)
        elif n_layers is not None:
            args.n_layers = n_layers
            args.attention_type_list = args.attention_type_list[:n_layers]

        logger.info("Loading + remapping weights via Qwen36ModelArgs.load_state_dict()...")
        state_dict = args.load_state_dict()

        cache_path = args.weight_cache_path()
        return cls(device, args, state_dict, tensor_cache_path=cache_path)

    def prefill_tp(self, token_ids, valid_len=None, vision_tokens=None):
        """Tensor-parallel full-model prefill (num_devices>1). Stateless: runs the
        whole sequence from scratch through the fractured-residual TP layers and
        returns the next-token logits at position valid_len-1.

        token_ids: torch [1, T] (pad T to a multiple of 128 for the GDN chunk
        kernel; right-padding does not affect the causal logit at valid_len-1).
        Returns ttnn logits [1, 1, 1, vocab_size] (host).
        """
        B, T = token_ids.shape
        assert B == 1, "prefill_tp is single-sequence"
        valid_len = valid_len or T

        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text), then build cos/sin from
        # that staged sequence table — same source as the traced TP path (_rope_tp_cos_sin_torch).
        self._build_request_rope(token_ids[:, :valid_len], vision_tokens)
        tok = ttnn.from_torch(
            token_ids.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)  # [1, T, dim_frac] (hidden dim sharded across mesh)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
        x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, T)
        rep = ttnn.ReplicateTensorToMesh(self.device)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)

        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill", chunk_size=128, valid_len=valid_len)

        # Last real position via one-hot matmul (not slice): bare slice breaks at long T (~49k+).
        sel = torch.zeros(1, 1, 1, T, dtype=torch.float32)
        sel[0, 0, 0, valid_len - 1] = 1.0
        sel_tt = ttnn.from_torch(
            sel,
            dtype=x.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x_last = ttnn.matmul(sel_tt, x)  # [1,1,1,dim_frac]
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)  # DistributedNorm on selected row
        self._store_last_hidden(x_last)
        logits = self._lm_head(x_last)
        # Replicated logits; read one replica -> torch [vocab_size].
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        return lt[0].reshape(-1)[: self.vocab_size]

    def reset_tp(self):
        """Reset TP layer KV cache / GDN state for a new sequence."""
        for layer in self.layers:
            layer.attention.reset_state()

    def decode_tp(self, token_id, pos):
        """Single-token TP decode at position `pos` (B=1). Uses KV + GDN from prefill/decode."""
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

        tok = ttnn.from_torch(
            torch.tensor([[int(token_id)]], dtype=torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)  # [1,1,dim_frac]
        x = ttnn.reshape(x, (1, 1, 1, x.shape[-1]))  # [1,1,B=1,dim_frac]
        # RoPE position offset by rope_delta for multimodal (KV position cur_pos_tt stays `pos`).
        cos, sin = rot_mats_decode(
            self.device,
            self.args.rope_head_dim,
            self.args.max_seq_len,
            self.args.rope_theta,
            torch.tensor([pos + self.rope.rope_delta], dtype=torch.int32),
        )
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            dtype=ttnn.int32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=cur_pos_tt)
        x = self._final_norm_decode(x)
        logits = self._lm_head(x)
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        return lt[0].reshape(-1)[: self.vocab_size]

    def generate_tp(self, prompt_ids, max_new_tokens=20):
        """TP greedy generation: prefill prompt, then decode. Returns new token ids."""
        import math as _math

        self.reset_tp()
        T = len(prompt_ids)
        T_pad = max(128, _math.ceil(T / 128) * 128)
        padded = prompt_ids + [0] * (T_pad - T)
        logits = self.prefill_tp(torch.tensor([padded], dtype=torch.long), valid_len=T)
        nxt = int(torch.argmax(logits).item())
        out = [nxt]
        for pos in range(T, T + max_new_tokens - 1):
            logits = self.decode_tp(nxt, pos)
            nxt = int(torch.argmax(logits).item())
            out.append(nxt)
        return out

    def _scatter_vision_tokens(self, x, token_ids, vision_tokens):
        """Splice vision-model embeddings into the text token embeddings, on device.

        On-device equivalent of the HF reference's
        ``inputs_embeds.masked_scatter(image_mask, image_embeds)``. The embedding is
        flattened to ``[rows, H]``; the packed ``vision_tokens`` are placed into a zero
        buffer at the image-placeholder rows with a dim-0 ``ttnn.scatter``, then merged
        with the text embeddings via
        ``ttnn.where(special_image_mask, vision, text)``. The embeddings/vision never
        leave the device. No-op when ``vision_tokens`` is None or the prompt has no
        image tokens (the text-only path).

        The image-placeholder mask and placement positions are computed on host from the
        token ids (which already live on host at every call site), mirroring the HF
        reference's ``special_image_mask = input_ids == self.config.image_token_id``, then
        uploaded. The two tiny derived tensors uploaded are the ``[n, H]`` scatter index
        (hidden-sharded on a mesh, like the embedding activations) and the ``[rows, 1]``
        where-predicate (replicated on a mesh — it broadcasts over the sharded hidden dim).

        Args:
            x (ttnn.Tensor): text embeddings from ``self.embd`` — ``[B, T, H]`` (the
                raw embedding output), hidden fractured along the last dim on a mesh.
            token_ids (torch.Tensor): the prefill token ids (``input_ids``), ``[B, T]`` on
                host; image placeholders are the entries equal to ``hf_config.image_token_id``.
            vision_tokens (ttnn.Tensor): ``[num_image_tokens, H]`` produced by the
                vision tower (fractured along hidden on a mesh, like ``x``), one row
                per image placeholder token.

        Returns:
            ttnn.Tensor: same logical shape / layout / sharding as ``x`` with the
            vision embeddings scattered in.
        """
        if vision_tokens is None:
            return x

        orig_shape = tuple(x.shape)
        hidden = orig_shape[-1]
        rows = 1
        for d in orig_shape[:-1]:
            rows *= d

        # special_image_mask = input_ids == image_token_id, computed on host from the
        # token ids and uploaded. torch.nonzero gives the placement positions directly.
        flat_ids = token_ids.reshape(-1)
        mask_bool = flat_ids == self._vision_placeholder_token_id()
        pos = torch.nonzero(mask_bool, as_tuple=False).reshape(-1)
        n = int(pos.numel())
        if n == 0:
            return x
        assert n == int(
            vision_tokens.shape[0]
        ), f"input_ids has {n} image-token positions but vision_tokens has {int(vision_tokens.shape[0])} rows"

        # Placement index: the dim-0 rows of the flattened [rows, H] embedding to fill,
        # repeated across the hidden dim so the whole hidden vector at each row is written.
        # ttnn.scatter mirrors torch.scatter: out[index[i, h], h] = src[i, h], with
        # index/src/input the same rank.
        index = pos.view(n, 1).expand(n, hidden).contiguous().to(torch.int32)
        # where-predicate: [rows, 1], broadcasts over hidden in ttnn.where.
        mask_col = mask_bool.view(rows, 1)

        if self.num_devices > 1:
            # Shard the index along hidden the same way the embedding shards its
            # activations, so each device's [n, H/TP] index matches its local x/vision
            # shard (the hidden columns are identical, so splitting is free). The predicate
            # broadcasts over the sharded hidden dim, so it is replicated.
            index_tt = ttnn.from_torch(
                index,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, 1), mesh_shape=self.args.cluster_shape
                ),
            )
            mask_tt = ttnn.from_torch(
                mask_col,
                dtype=x.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        else:
            index_tt = ttnn.from_torch(index, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            mask_tt = ttnn.from_torch(mask_col, dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

        # ttnn.scatter requires input.dtype == src.dtype.
        src = vision_tokens if vision_tokens.dtype == x.dtype else ttnn.typecast(vision_tokens, x.dtype)

        # Place the packed vision rows into a zero buffer, then select per row:
        # vision at image positions, original text embedding everywhere else.
        x_2d = ttnn.reshape(x, (rows, hidden))
        vision_placed = ttnn.scatter(ttnn.zeros_like(x_2d), 0, index_tt, src)
        ttnn.deallocate(index_tt)
        out = ttnn.where(mask_tt, vision_placed, x_2d)
        ttnn.deallocate(vision_placed)
        ttnn.deallocate(mask_tt)
        return ttnn.reshape(out, orig_shape)

    def prefill(self, token_ids, vision_tokens=None):
        B, T = token_ids.shape

        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text) before any cos/sin seam.
        self._build_request_rope(token_ids, vision_tokens)

        if T > 1024:
            return self.prefill_layer_chunked(token_ids, chunk_size=2048, vision_tokens=vision_tokens)

        # Short sequences (<=1024)
        self.reset_state(batch_size=B)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)

        cos, sin = self.rope.get_prefill_rot_mats(0, T)

        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

        x = self.norm(x, mode=Mode.PREFILL)

        x_last = x[:, -1:, :]
        self._store_last_hidden(x_last)
        logits = self._lm_head(x_last)

        return logits

    def prefill_layer_chunked(self, token_ids, chunk_size=2048, page_table=None, vision_tokens=None):
        """Prefill long sequences using layer-at-a-time chunked processing.

        DeltaNet uses larger chunk_size (256 vs 64) to limit Neumann-series error
        (4096 tokens -> 16 sub-chunks, PCC >0.98). page_table enables paged prefill."""
        B, T = token_ids.shape
        self.reset_state(batch_size=B)
        features = getattr(self, "prefill_features", None)
        if features is not None:
            features.begin_sequence()

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(token_ids_ttnn)

        # Attn layers: chunk_size>=4096 (no Neumann limit; fewer SDPA compilations).
        attn_chunk_size = max(chunk_size, 4096)

        page_table_tt = None
        if page_table is not None:
            page_table_tt = ttnn.from_torch(
                page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )

        for layer_idx, layer in enumerate(self.layers):
            layer_chunk_size = attn_chunk_size if layer.is_full_attention else chunk_size

            chunks_out = []
            for chunk_start in range(0, T, layer_chunk_size):
                chunk_end = min(chunk_start + layer_chunk_size, T)

                x_chunk = x[:, chunk_start:chunk_end, :]
                x_chunk = ttnn.to_layout(x_chunk, ttnn.TILE_LAYOUT)

                if layer.is_full_attention and page_table is not None:
                    # Paged prefill path. M-RoPE-aware cos/sin for sequence positions of this chunk
                    # (slices the staged per-request table for multimodal; 1D RoPE otherwise).
                    cos, sin = self.rope.get_prefill_rot_mats(chunk_start, chunk_end - chunk_start)

                    block_size = 64
                    chunk_blocks_end = math.ceil(chunk_end / block_size)
                    chunk_page_table = page_table[:, chunk_start // block_size : chunk_blocks_end]
                    chunk_page_table_tt = ttnn.from_torch(
                        chunk_page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                    )

                    x_chunk = layer.forward(
                        x_chunk,
                        cos=cos,
                        sin=sin,
                        mode="prefill",
                        page_table=page_table_tt,
                        chunk_page_table=chunk_page_table_tt,
                        chunk_start_idx=chunk_start,
                    )

                elif layer.is_full_attention:
                    # Original concat path (non-paged prefill). M-RoPE-aware cos/sin (per-request
                    # table slice for multimodal; 1D RoPE otherwise).
                    cos, sin = self.rope.get_prefill_rot_mats(chunk_start, chunk_end - chunk_start)
                    x_chunk = layer.forward(x_chunk, cos=cos, sin=sin, mode="prefill")
                else:
                    x_chunk = layer.forward(
                        x_chunk,
                        cos=None,
                        sin=None,
                        mode="prefill",
                        chunk_size=layer.attention.long_prefill_chunk_size,
                    )

                chunks_out.append(x_chunk)

            # Last layer: save last token from last chunk before concat (avoids L1 clash on long T).
            is_last_layer = layer_idx == len(self.layers) - 1
            if is_last_layer:
                x_last = chunks_out[-1][:, -1:, :]
                x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)

            if len(chunks_out) == 1:
                x_new = chunks_out[0]
            else:
                x_new = ttnn.concat(chunks_out, dim=1)
                for c in chunks_out:
                    ttnn.deallocate(c)
            x_new = ttnn.to_memory_config(x_new, ttnn.DRAM_MEMORY_CONFIG)

            ttnn.deallocate(x)
            x = x_new
            if features is not None:
                features.capture_tail(layer_idx, x, T)

        if features is not None:
            features.commit_tail(T)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        self._store_last_hidden(x_last)
        logits = self._lm_head(x_last)
        ttnn.deallocate(x)

        return logits

    def _store_last_hidden(self, hidden):
        if self.mtp is None:
            return
        rows = math.prod(list(hidden.shape)[:-1])
        source = ttnn.reshape(hidden, (1, 1, rows, self.args.dim))
        if rows < 32:
            padded = ttnn.pad(
                source,
                [(0, 0), (0, 0), (0, 32 - rows), (0, 0)],
                value=0.0,
            )
            source = padded
        if self._last_hidden is None:
            self._last_hidden = ttnn.clone(source, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy(source, self._last_hidden)

    def propose_mtp(self, first_token, first_position, num_steps=1):
        if self.mtp is None or self._last_hidden is None:
            return []
        hidden = ttnn.reshape(self._last_hidden, (1, 32, self.args.dim))
        active = ttnn.slice(hidden, (0, 0, 0), (1, 1, self.args.dim))
        drafts = self.mtp.propose(first_token, active, first_position, num_steps)
        ttnn.deallocate(active)
        return drafts

    def verify_two_target(
        self,
        token_ids,
        positions_host,
        position_tensor,
        page_table,
        shadow_fill_page_table,
        shadow_source_page,
        shadow_same_page,
        cos=None,
        sin=None,
    ):
        """Run one genuine M=2 target pass and retain transactional layer candidates."""
        x = self.embd(token_ids)
        if cos is None or sin is None:
            cos, sin = self.rope.get_rot_mats(positions_host.reshape(1, 2).to(torch.long))
        candidates = []
        for index, layer in enumerate(self.layers):
            x, layer_candidates = layer.verify_two(
                x,
                cos,
                sin,
                position_tensor,
                page_table,
                shadow_fill_page_table=shadow_fill_page_table if layer.is_full_attention else None,
                shadow_source_page=shadow_source_page,
                shadow_same_page=shadow_same_page,
                final_norm=self.verify_final_norm(index),
            )
            candidates.append(layer_candidates)
        hidden = self.verify_hidden(x)
        logits = self._lm_head(hidden)
        logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        target_tokens = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        draft_token = token_ids[:, 1:2]
        accepted = ttnn.eq(target_tokens[:, 0:1], draft_token)
        return logits, target_tokens, accepted, hidden, candidates

    def commit_verify_two(
        self,
        accept_second,
        hidden,
        candidates,
        second_position_tensor,
        live_page_table,
    ):
        """Commit S1 universally and S2/KV1 only when the device comparison accepted."""
        selected = 1 if accept_second else 0
        for layer, layer_candidates in zip(self.layers, candidates):
            if layer.is_full_attention:
                if not accept_second:
                    continue
                key1, value1 = layer_candidates
                heads = layer.attention.config.num_kv_heads
                dim = layer.attention.config.head_dim
                grid = self.device.compute_with_storage_grid_size()
                shard_grid = ttnn.num_cores_to_corerangeset(1, grid, row_wise=True)
                shard_spec = ttnn.ShardSpec(shard_grid, [32, dim], ttnn.ShardOrientation.ROW_MAJOR)
                sharded_mc = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
                )
                key1 = ttnn.to_memory_config(ttnn.reshape(key1, [1, 1, heads, dim]), sharded_mc)
                value1 = ttnn.to_memory_config(ttnn.reshape(value1, [1, 1, heads, dim]), sharded_mc)
                ttnn.experimental.paged_update_cache(
                    layer.attention.paged_kv_cache_key,
                    key1,
                    update_idxs_tensor=second_position_tensor,
                    page_table=live_page_table,
                )
                ttnn.experimental.paged_update_cache(
                    layer.attention.paged_kv_cache_value,
                    value1,
                    update_idxs_tensor=second_position_tensor,
                    page_table=live_page_table,
                )
            else:
                recurrent_candidates, conv_candidates = layer_candidates
                # Candidates are TILE [H, K, V]; the live state is row-major sharded.
                chosen = ttnn.to_layout(recurrent_candidates[selected], ttnn.ROW_MAJOR_LAYOUT)
                ttnn.copy(ttnn.reshape(chosen, layer.attention.recurrent_state.shape), layer.attention.recurrent_state)
                ttnn.deallocate(chosen)
                ttnn.copy(conv_candidates[selected], layer.attention.fused_conv_state)
        selected_hidden = hidden[:, selected : selected + 1, :]
        self._store_last_hidden(selected_hidden)

    # ------------------------------------------------------------------
    # Device-resident MTP-1 cycle (trace-replayed; no host round trip).
    # ------------------------------------------------------------------
    def init_mtp_cycle(self, page_table_width):
        """Allocate the fixed device buffers the traced MTP cycle reads and writes.

        Per step the graph consumes tokens [committed, draft], their two positions and a
        one-row page table, verifies both at once, commits the selected recurrent/conv
        state on device, drafts the next token from the selected hidden, and rewrites
        the token/position buffers for the next replay. The host reads back only
        ``cycle_out`` = [target0, target1, accepted, draft_next].
        """
        assert self.mtp is not None, "init_mtp_cycle requires QWEN36_MTP=1"
        if page_table_width < 8 or page_table_width % 8:
            raise ValueError("MTP page-table width must be a positive multiple of eight blocks")
        if self._paged_kv_caches and page_table_width > int(self._paged_kv_caches[0][0].shape[0]):
            raise ValueError("MTP page-table width exceeds allocated KV blocks")
        if getattr(self, "_mtp_tokens", None) is not None:
            if self._mtp_cycle_width != page_table_width:
                raise ValueError("MTP page-table width cannot change after buffer allocation")
            # Warmup runs twice. Reallocating after prefill trace capture can put
            # read-only RoPE tables inside that trace's replay scratch footprint.
            return
        dev = self.device
        i32 = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self._mtp_tokens = ttnn.from_torch(torch.zeros(1, 2, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self._mtp_positions = ttnn.from_torch(torch.zeros(2, dtype=torch.int32), **i32)
        # Two identical rows: both positions live in the same request's page table. The
        # draft's KV slot (pos+1) is dead on rejection and is overwritten by the next
        # committed token before any read, so no shadow page is needed.
        self._mtp_page_table = ttnn.from_torch(torch.zeros(2, page_table_width, dtype=torch.int32), **i32)
        self._mtp_out = ttnn.from_torch(torch.zeros(1, 4, dtype=torch.int32), **i32)
        if self._last_hidden is None:
            # Fixed-address buffer so decode traces captured later bake a stable destination.
            self._last_hidden = ttnn.from_torch(torch.zeros(1, 1, 32, self.args.dim, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._mtp_cycle_trace = None
        self._mtp_cycle_width = page_table_width
        self.init_target_verification()
        self.init_exact_sampler(1)

    def init_dflash(
        self, checkpoint, cache_prefix, block_size=8, feature_buckets=None, page_table_width=None
    ):
        """Allocate device drafting and feature buffers before any prefill/cycle trace.

        Returns the fixed-address DFlashCycle. Warm both its first_drafts and
        transaction programs before capture; a fresh prefill resets draft context.
        """
        from models.demos.blackhole.qwen36.tt.dflash import NativeDFlash, load_dflash_checkpoint
        from models.demos.blackhole.qwen36.tt.speculative_dflash import DFlashCycle
        from models.demos.blackhole.qwen36.tt.speculative_features import PrefillFeatures

        if self.num_devices != 1 or not self._paged_kv_caches:
            raise RuntimeError("DFlash requires one device and allocated paged KV caches")
        if any(cache.dtype != ttnn.bfloat16 for pair in self._paged_kv_caches for cache in pair):
            raise ValueError("DFlash accepted-prefix KV transactions require QWEN_SDPA_BF8=0")
        if getattr(self, "_dflash_cycle", None) is not None:
            raise RuntimeError("DFlash is already initialized")
        if getattr(self, "_chunked_prepared_key", None) is not None:
            raise RuntimeError("Initialize DFlash before prefill warmup or capture")
        state, config = load_dflash_checkpoint(checkpoint)
        if config["hidden_size"] != self.args.dim or config["vocab_size"] != self.args.vocab_size:
            raise ValueError("Draft hidden/vocabulary dimensions do not match the target")
        taps = config["dflash_config"]["target_layer_ids"]
        if max(taps) >= len(self.layers):
            raise ValueError("Draft feature taps exceed the target layer count")
        buckets = self._PREFILL_MASK_BUCKETS if feature_buckets is None else tuple(feature_buckets)
        if not buckets or max(buckets) < config["sliding_window"]:
            raise ValueError("Feature buckets must cover the draft sliding window")
        width = int(self._paged_kv_caches[0][0].shape[0]) if page_table_width is None else page_table_width
        self.dflash = NativeDFlash(
            self.device, state, config, self.embd, self._lm_head, cache_prefix, block_size=block_size
        )
        del state
        self.prefill_features = PrefillFeatures(self, self.dflash.feature_layers, buckets, self.dflash)
        self._dflash_cycle = DFlashCycle(self, self.dflash, width)
        return self._dflash_cycle

    def init_target_verification(self):
        """Allocate shared RoPE and fused GDN outputs before any speculative trace."""
        if getattr(self, "_target_cos_table", None) is not None:
            return
        dev = self.device
        # Rope tables as row-major [max_seq, rope_dim] for device gather by position. Single-device
        # rope keeps only one device row, so take the full table from the host copy.
        cos_t = self.rope.cos_cpu
        sin_t = self.rope.sin_cpu
        self._target_cos_table = ttnn.from_torch(cos_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self._target_sin_table = ttnn.from_torch(sin_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        if GDNConfig.from_args(self.args).num_v_heads == 48:
            from models.demos.blackhole.qwen36.tt.gdn.fused_verify_output import FusedVerifyOutput

            self._target_verify_inbox = None
            for layer in self.layers:
                if layer.is_full_attention:
                    continue
                dn = layer.attention
                if dn.native_frontend and dn.native_output and dn.native_recurrent:
                    runner = FusedVerifyOutput(dev, inbox=self._target_verify_inbox)
                    self._target_verify_inbox = runner.inbox
                    dn._fused_verify_output = runner

    def _mtp_rope_for_positions(self, positions_i32):
        """cos/sin [1, 2, rope_dim] TILE gathered on device from the two positions."""
        idx = ttnn.reshape(ttnn.typecast(positions_i32, ttnn.uint32), [1, 2])
        cos = ttnn.embedding(idx, self._target_cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idx, self._target_sin_table, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(idx)
        return cos, sin

    def _mtp_select_row(self, two_rows, flag_tile):
        """Pick row 1 when flag else row 0 from a [1, 2, D] TILE tensor -> [1, 1, D]."""
        return ttnn.where(flag_tile, two_rows[:, 1:2, :], two_rows[:, 0:1, :])

    def _verify_two_layers(self, x, cos, sin, positions, page_table):
        """All decoder layers at M=2 (MTP verify) without committing; returns (last output, per-layer candidates)."""
        candidates = []
        for index, layer in enumerate(self.layers):
            previous = x
            x, layer_candidates = layer.verify_two(
                x, cos, sin, positions, page_table,
                shadow_fill_page_table=None, shadow_source_page=0, shadow_same_page=False,
                final_norm=self.verify_final_norm(index),
            )
            ttnn.deallocate(previous)
            candidates.append(layer_candidates)
        return x, candidates

    def _greedy_argmax(self, logits):
        """uint32 argmax over the last dim of 1-2 tiled logit rows (TileArgmax; same ids as untilize + ttnn.argmax)."""
        if getattr(self, "_tile_argmax", None) is None:
            from models.demos.blackhole.qwen36.tt.tile_argmax import TileArgmax

            self._tile_argmax = TileArgmax()
        return self._tile_argmax(logits)

    def _commit_verify_two(self, accepted, candidates):
        """Commit every GDN layer's accepted M=2 candidate (S1 or S2, by the device flag) into its live L1 state in
        one launch, then release the candidates. KV for both positions already sits in the live pages."""
        from models.demos.blackhole.qwen36.tt.gdn.fused_commit_all import FusedCommitAll

        entries = []
        for layer, layer_candidates in zip(self.layers, candidates):
            if layer.is_full_attention:
                continue
            (s1, s2), (c1, c2) = layer_candidates
            entries.append((s1, s2, c1, c2, layer.attention.recurrent_state, layer.attention.fused_conv_state))
        if not entries:
            return
        if getattr(self, "_fused_mtp_commit_all", None) is None:
            self._fused_mtp_commit_all = FusedCommitAll()
        self._fused_mtp_commit_all(accepted, entries)
        for s1, s2, *_ in entries:
            ttnn.deallocate(s1)
            ttnn.deallocate(s2)

    def mtp_cycle_step(self):
        """One full MTP transaction on device: verify M=2, commit, draft next. Trace-safe."""
        tokens = self._mtp_tokens
        positions = self._mtp_positions
        page_table = self._mtp_page_table
        cos, sin = self._mtp_rope_for_positions(positions)
        x = self.embd(tokens)
        x, candidates = self._verify_two_layers(x, cos, sin, positions, page_table)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        hidden = self.verify_hidden(x)
        if hidden is not x:
            ttnn.deallocate(x)
        logits = self._lm_head(hidden)
        targets = self._greedy_argmax(logits)  # [1, 2] uint32
        ttnn.deallocate(logits)
        targets_i32 = ttnn.typecast(targets, ttnn.int32)
        draft = ttnn.typecast(tokens[:, 1:2], ttnn.int32)
        accepted = ttnn.eq(targets_i32[:, 0:1], draft)  # [1,1] int32 0/1
        ttnn.deallocate(draft)
        # Tile-layout flag for where() over tiled operands.
        flag_tile = ttnn.to_layout(ttnn.typecast(ttnn.reshape(accepted, [1, 1, 1]), ttnn.bfloat16), ttnn.TILE_LAYOUT)

        # Select and untilize directly into the fixed live GDN state buffers.
        self._commit_verify_two(accepted, candidates)

        # Selected hidden and next committed token/position.
        sel_hidden = self._mtp_select_row(hidden, flag_tile)  # [1,1,dim]
        ttnn.deallocate(hidden)
        # Row-major int select: t0 + acc * (t1 - t0) (where() needs TILE operands).
        t0 = targets_i32[:, 0:1]
        t1 = targets_i32[:, 1:2]
        next_token = ttnn.add(t0, ttnn.multiply(accepted, ttnn.subtract(t1, t0)))  # int32 [1,1]
        advance = ttnn.add(accepted, 1)  # 1 or 2
        next_pos = ttnn.add(positions[0:1], ttnn.reshape(advance, [1]))

        # Native draft from (next_token, sel_hidden) at next_pos.
        draft_next = self.mtp.draft_device(next_token, sel_hidden, next_pos)  # int32 [1,1]
        ttnn.deallocate(sel_hidden)

        # Rewrite cycle inputs for the next replay and the compact host-visible output.
        new_tokens = ttnn.typecast(ttnn.concat([next_token, draft_next], dim=1), ttnn.uint32)
        ttnn.copy(new_tokens, tokens)
        ttnn.deallocate(new_tokens)
        new_pos = ttnn.concat([ttnn.reshape(next_pos, [1]), ttnn.reshape(ttnn.add(next_pos, 1), [1])], dim=0)
        ttnn.copy(new_pos, positions)
        ttnn.deallocate(new_pos)
        out = ttnn.concat([targets_i32, accepted, draft_next], dim=1)  # [1,4]
        ttnn.copy(out, self._mtp_out)
        ttnn.deallocate(out)
        ttnn.deallocate(targets_i32)
        ttnn.deallocate(targets)
        ttnn.deallocate(accepted)
        ttnn.deallocate(flag_tile)
        ttnn.deallocate(next_token)
        ttnn.deallocate(draft_next)
        ttnn.deallocate(advance)
        ttnn.deallocate(next_pos)
        return self._mtp_out

    def mtp_cycle_seed(self, committed_token, draft_token, position, page_table_row):
        """Load the cycle inputs from host (after prefill or a rejected/finished request)."""
        token_host = ttnn.from_torch(
            torch.tensor([[committed_token, draft_token]], dtype=torch.int32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        position_host = ttnn.from_torch(
            torch.tensor([position, position + 1], dtype=torch.int32),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(token_host, self._mtp_tokens)
        ttnn.copy_host_to_device_tensor(position_host, self._mtp_positions)
        self.mtp_cycle_set_page_table(page_table_row)

    def mtp_cycle_set_page_table(self, page_table_row):
        row = page_table_row.to(torch.int32).reshape(1, -1)
        if row.shape[1] > self._mtp_cycle_width:
            raise ValueError("MTP page table exceeds the fixed cycle capacity")
        if row.shape[1] < self._mtp_cycle_width:
            row = torch.cat([row, torch.zeros(1, self._mtp_cycle_width - row.shape[1], dtype=torch.int32)], dim=1)
        host = ttnn.from_torch(row.repeat(2, 1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self._mtp_page_table)

    def mtp_cycle_run(self):
        """Execute exactly one transaction; capture is exclusively a warmup operation."""
        if not getattr(self, "_mtp_cycle_compiled", False):
            raise RuntimeError("MTP programs must be warmed before serving")
        if self._mtp_cycle_trace is None or os.environ.get("QWEN36_MTP_EAGER") == "1":
            self.mtp_cycle_step()
            ttnn.synchronize_device(self.device)
        else:
            ttnn.execute_trace(self.device, self._mtp_cycle_trace, cq_id=0, blocking=True)
        return ttnn.to_torch(self._mtp_out).reshape(-1).tolist()

    def mtp_cycle_warmup(self, page_table_width, capture=True):
        """Compile M=2 before any trace; capture later without changing persistent addresses."""
        self.init_mtp_cycle(page_table_width)
        if self._mtp_cycle_trace is not None:
            return
        saved = self._save_deltanet_states()
        table = torch.arange(page_table_width, dtype=torch.int32)
        try:
            self.mtp_cycle_seed(11, 12, 62, table)
            if not getattr(self, "_mtp_cycle_compiled", False):
                self.mtp_cycle_step()
                ttnn.synchronize_device(self.device)
                self._mtp_cycle_compiled = True
            if capture:
                self.mtp_cycle_seed(11, 12, 62, table)
                trace = ttnn.begin_trace_capture(self.device, cq_id=0)
                self.mtp_cycle_step()
                ttnn.end_trace_capture(self.device, trace, cq_id=0)
                self._mtp_cycle_trace = trace
                ttnn.execute_trace(self.device, trace, cq_id=0, blocking=True)
        finally:
            self._restore_deltanet_states(saved, self.device)

    def mtp_step(self, committed_token, draft_token, position, page_table_row):
        """Run one device MTP transaction for the request at `position`.

        Returns (emitted_tokens, next_draft). The scheduler is authoritative:
        reseed every call, including request reuse with an identical token/position
        tuple and same-width page-table reassignment.
        """
        if not 0 <= int(position) < self.args.max_seq_len - 2:
            raise ValueError("MTP verify/draft requires two lookahead positions inside the context")
        if os.environ.get("QWEN36_MTP_TRACE") == "1":
            logger.info(f"MTP step in=({int(committed_token)},{int(draft_token)},{int(position)})")
        self.mtp_cycle_seed(int(committed_token), int(draft_token), int(position), page_table_row)
        t0, t1, acc, next_draft = self.mtp_cycle_run()
        emitted = [int(t0), int(t1)] if acc else [int(t0)]
        if os.environ.get("QWEN36_MTP_TRACE") == "1":
            logger.info(f"MTP step out targets=({int(t0)},{int(t1)}) acc={int(acc)} draft={int(next_draft)}")
        return emitted, int(next_draft)

    # ------------------------------------------------------------------
    # Exact on-device sampling (device_sampling.py) and the sampled MTP-1 cycle.
    # ------------------------------------------------------------------
    def init_exact_sampler(self, num_slots=1):
        """Allocate the sampler's fixed-address buffers before any trace is captured."""
        if self.exact_sampler is not None:
            if self.exact_sampler.num_slots < num_slots:
                raise ValueError("The exact sampler was allocated for fewer history slots")
            return self.exact_sampler
        if self.num_devices != 1:
            raise ValueError("Exact on-device sampling supports a single device")
        from models.demos.blackhole.qwen36.tt.device_sampling import DeviceSampler

        self.exact_sampler = DeviceSampler(self.device, self.args.vocab_size, num_slots=num_slots)
        self._mtp_sampled_trace = None
        self._mtp_verify_logits = None
        return self.exact_sampler

    def prepare_sampled_request(self, request, outputs=None):
        """MTP: bring history slot 0 to the request's outputs and load its verify/draft parameter rows (row 1
        also counts the draft token in its history)."""
        sampler = self.exact_sampler
        outputs = request.outputs if outputs is None else outputs
        sampler.ensure_history(0, request.owner, outputs, request.prompt or (), request.spec.uses_prompt)
        sampler.set_params([request.spec.words(0), request.spec.words(0, extra=True)])

    def mtp_cycle_step_sampled(self):
        """One sampled MTP transaction on device: verify M=2 with speculative rejection sampling, commit, draw the
        next draft from q. Trace-safe; the host reads back only the sampler's cycle words."""
        tokens = self._mtp_tokens
        positions = self._mtp_positions
        page_table = self._mtp_page_table
        sampler = self.exact_sampler
        cos, sin = self._mtp_rope_for_positions(positions)
        x = self.embd(tokens)
        x, candidates = self._verify_two_layers(x, cos, sin, positions, page_table)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        hidden = self.verify_hidden(x)
        if hidden is not x:
            ttnn.deallocate(x)
        logits = self._lm_head(hidden)
        # Retained (fixed address in the trace): the host redraws from it when a tie group was truncated.
        self._mtp_verify_logits = ttnn.reshape(logits, (1, 1, 2, self.args.vocab_size))
        sampler.verify_two(self._mtp_verify_logits, tokens, positions)
        cyc = sampler.cyc
        targets_i32 = ttnn.slice(cyc, (0, 0), (1, 2))
        accepted = ttnn.slice(cyc, (0, 2), (1, 3))  # [1,1] int32 0/1
        flag_tile = ttnn.to_layout(ttnn.typecast(ttnn.reshape(accepted, [1, 1, 1]), ttnn.bfloat16), ttnn.TILE_LAYOUT)

        self._commit_verify_two(accepted, candidates)

        sel_hidden = self._mtp_select_row(hidden, flag_tile)  # [1,1,dim]
        ttnn.deallocate(hidden)
        # The host redraw path drafts again from the committed row's hidden.
        self._store_last_hidden(sel_hidden)
        t0 = targets_i32[:, 0:1]
        t1 = targets_i32[:, 1:2]
        next_token = ttnn.add(t0, ttnn.multiply(accepted, ttnn.subtract(t1, t0)))  # int32 [1,1]
        advance = ttnn.add(accepted, 1)
        next_pos = ttnn.add(positions[0:1], ttnn.reshape(advance, [1]))

        hidden_out = self.mtp.draft_hidden_device(next_token, sel_hidden, next_pos)
        ttnn.deallocate(sel_hidden)
        self.mtp.sample_draft(hidden_out, next_pos, cyc)
        ttnn.deallocate(hidden_out)
        draft_next = ttnn.slice(cyc, (0, 4), (1, 5))

        new_tokens = ttnn.typecast(ttnn.concat([next_token, draft_next], dim=1), ttnn.uint32)
        ttnn.copy(new_tokens, tokens)
        ttnn.deallocate(new_tokens)
        new_pos = ttnn.concat([ttnn.reshape(next_pos, [1]), ttnn.reshape(ttnn.add(next_pos, 1), [1])], dim=0)
        ttnn.copy(new_pos, positions)
        ttnn.deallocate(new_pos)
        for tensor in (targets_i32, accepted, flag_tile, next_token, draft_next, advance, next_pos):
            ttnn.deallocate(tensor)

    def mtp_cycle_warmup_sampled(self, page_table_width, capture=True):
        """Compile the sampled cycle before any trace; capture later at fixed addresses."""
        self.init_mtp_cycle(page_table_width)
        if self._mtp_sampled_trace is not None:
            return
        saved = self._save_deltanet_states()
        table = torch.arange(page_table_width, dtype=torch.int32)
        try:
            self.exact_sampler.set_params([None, None])  # inactive rows: the sampler kernels are no-ops
            self.mtp_cycle_seed(11, 12, 62, table)
            if not getattr(self, "_mtp_sampled_compiled", False):
                self.mtp_cycle_step_sampled()
                ttnn.synchronize_device(self.device)
                self._mtp_sampled_compiled = True
            if capture:
                if self._mtp_verify_logits is not None:
                    ttnn.deallocate(self._mtp_verify_logits)
                    self._mtp_verify_logits = None
                self.mtp_cycle_seed(11, 12, 62, table)
                trace = ttnn.begin_trace_capture(self.device, cq_id=0)
                self.mtp_cycle_step_sampled()
                ttnn.end_trace_capture(self.device, trace, cq_id=0)
                self._mtp_sampled_trace = trace
                ttnn.execute_trace(self.device, trace, cq_id=0, blocking=True)
        finally:
            self._restore_deltanet_states(saved, self.device)
            self.exact_sampler.invalidate()

    def mtp_step_sampled(self, committed_token, draft_token, position, page_table_row, request):
        """Run one sampled MTP transaction; returns (emitted_tokens, next_draft).

        A truncated top-k tie group (R_INEXACT) forces rejection on device; the host then draws that position
        exactly from the retained verify logits and redraws the next draft."""
        from models.demos.blackhole.qwen36.tt.device_sampling import host_sample

        if not 0 <= int(position) < self.args.max_seq_len - 2:
            raise ValueError("MTP verify/draft requires two lookahead positions inside the context")
        if not getattr(self, "_mtp_sampled_compiled", False):
            raise RuntimeError("Sampled MTP programs must be warmed before serving")
        position = int(position)
        sampler = self.exact_sampler
        self.prepare_sampled_request(request)
        self.mtp_cycle_seed(int(committed_token), int(draft_token), position, page_table_row)
        if self._mtp_sampled_trace is None or os.environ.get("QWEN36_MTP_EAGER") == "1":
            self.mtp_cycle_step_sampled()
        else:
            ttnn.execute_trace(self.device, self._mtp_sampled_trace, cq_id=0, blocking=False)
        t0, t1, acc, vflags, next_draft = (int(v) for v in ttnn.to_torch(sampler.cyc).reshape(-1)[:5].tolist())
        outputs = request.outputs
        if vflags == 0:
            emitted = [t0, t1] if acc else [t0]
            sampler.note_committed(0, request.owner, len(outputs) + len(emitted))
            return emitted, next_draft
        rows = ttnn.to_torch(self._mtp_verify_logits).reshape(-1, self.args.vocab_size)
        if vflags == 1:
            emitted = [host_sample(rows[0], request.spec, outputs, request.prompt or (), position + 1)]
        else:
            emitted = [t0, host_sample(rows[1], request.spec, list(outputs) + [t0], request.prompt or (), position + 2)]
        logger.info(f"Exact sampling: host redraw at position {position + len(emitted)} (truncated top-k tie group)")
        sampler.invalidate(0)
        self.prepare_sampled_request(request, outputs=list(outputs) + emitted)
        return emitted, self.mtp.first_draft_sampled(emitted[-1], position + len(emitted))

    def decode(self, token_ids, current_pos):
        B = token_ids.shape[0]

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        ttnn.deallocate(token_ids_ttnn)
        flat_decode = self.num_devices == 1 and os.environ.get("QWEN36_FLAT_DECODE_BATCH") == "1"
        if flat_decode:
            x = ttnn.reshape(x, (1, 1, B, x.shape[-1]))

        # RoPE position is offset by rope_delta for a multimodal request (image tokens compress the
        # position space); the KV/cache position (cur_pos_tensor below) stays the true sequence pos.
        position_ids = torch.full((B, 1), current_pos + self.rope.rope_delta, dtype=torch.long)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # cur_pos for SDPA decode + paged_update_cache ([B*n_kv] after cache reshape).
        n_kv = self.args.n_kv_heads
        cur_pos_tensor = ttnn.from_torch(
            torch.full((B * n_kv,), current_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        for i, layer in enumerate(self.layers):
            final_norm = (
                self._final_norm_residual_decode
                if self._fuse_final_residual_norm and i == len(self.layers) - 1
                else None
            )
            x = layer.forward(
                x,
                cos=cos,
                sin=sin,
                mode="decode",
                position_tensor=cur_pos_tensor,
                final_norm=final_norm,
            )

        if not self._fuse_final_residual_norm:
            x = self._final_norm_decode(x)
        if flat_decode:
            x = ttnn.reshape(x, (B, 1, x.shape[-1]))
        self._store_last_hidden(x)
        if self._ondev_argmax:
            # Pre-gather vocab-sharded logits; caller argmaxes shards, skips all-gather + readback.
            logits = self._lm_head_linear(x)
        else:
            logits = self._lm_head(x)
        ttnn.deallocate(x)

        return logits

    def _forward_decode(self, token_ids_buf, cos, sin, cur_pos_tensor, page_table, sharded_lm_head=False):
        """Trace-safe paged decode. All inputs are device tensors.

        sharded_lm_head=True: return the pre-gather vocab-sharded logits (no all-gather)
        for the on-device sampler, which does its own cross-device top-k + gather.
        """
        x = self.embd(token_ids_buf)
        flat_decode = self._batched_single_device or (
            self.num_devices == 1 and os.environ.get("QWEN36_FLAT_DECODE_BATCH") == "1"
        )
        if self.num_devices > 1 or flat_decode:
            # Keep serving slots in one physical M tile throughout decode. TP requires
            # this layout; single-device batching uses it to avoid one padded tile per slot.
            x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))
        for i, layer in enumerate(self.layers):
            final_norm = (
                self._final_norm_residual_decode
                if self._fuse_final_residual_norm and i == len(self.layers) - 1
                else None
            )
            if layer.is_full_attention:
                x = layer.forward(
                    x,
                    cos,
                    sin,
                    position_tensor=cur_pos_tensor,
                    page_table=page_table,
                    mode="decode",
                    final_norm=final_norm,
                )
            else:
                x = layer.forward(x, mode="decode", position_tensor=cur_pos_tensor, final_norm=final_norm)
        if not self._fuse_final_residual_norm:
            x = self._final_norm_decode(x)
        self._store_last_hidden(x)
        if sharded_lm_head or self._ondev_argmax:
            # Pre-gather vocab-sharded logits (on-device sampling / greedy argmax).
            logits = self._lm_head_linear(x)
        else:
            logits = self._lm_head(x)
        ttnn.deallocate(x)
        return logits

    def _forward_prefill_chunk(
        self, token_buf, cos_buf, sin_buf, chunk_start_idx_tensor, full_page_table, chunk_page_table
    ):
        """Trace-safe single-chunk prefill. Updates paged KV + GDN state in place.
        Returns last-layer hidden [1, chunk_size, hidden_size]."""
        x = self.embd(token_buf)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # Trace-safe vision splice (fixed-shape where over persistent buffers; identity when the
        # mask buffer is zero, which is the case for every text-only chunk and request). The caller
        # stages the buffers before replaying chunk 0 of a multimodal prompt; chunks>0 are cleared.
        x = self._apply_vision_merge(x, length=x.shape[1])
        features = getattr(self, "prefill_features", None)
        for index, layer in enumerate(self.layers):
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos_buf,
                    sin=sin_buf,
                    mode="prefill",
                    page_table=full_page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size)
            ttnn.deallocate(x)
            x = x_new
            if features is not None:
                features.capture(index, x)
        return x

    def _rope_tp_cos_sin_torch(self, start, length):
        """Torch cos/sin tables [1, 1, length, rope_head_dim] for SEQUENCE positions
        [start, start+length), in the rope_tp (HF split-halves) format consumed by
        apply_partial_rope_prefill. Single source of truth for the TP masked-bucket and
        traced chunk-outer prefill paths (so the captured trace's cos/sin are byte-identical
        to the eager path's). M-RoPE-aware: when a multimodal request staged a per-sequence
        table (build_request_rope) this slices it; otherwise it is ordinary 1D RoPE at
        positions [start, start+length) — byte-identical to the pre-M-RoPE behaviour."""
        rd = self.args.rope_head_dim
        cos_t, sin_t = self.rope.prefill_cos_sin_torch(start, length)  # [length, rd] bf16
        cos = cos_t.reshape(1, 1, length, rd)
        sin = sin_t.reshape(1, 1, length, rd)
        return cos, sin

    def _forward_prefill_chunk_tp(
        self, token_buf, cos_buf, sin_buf, chunk_start_idx_tensor, full_page_table, chunk_page_table
    ):
        """TP trace-safe single-chunk prefill (replicated persistent buffers).
        Full chunk (valid_len==chunk_size); flexible SDPA via device chunk_start_idx.
        Returns hidden [1,1,chunk_size,dim]."""
        chunk_size = self._chunked_chunk_size
        x = self.embd(token_buf)
        x = ttnn.reshape(x, (1, 1, chunk_size, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # Trace-safe vision splice (fixed-shape where over the hidden-sharded persistent buffers;
        # identity when the mask is zero, i.e. every text-only chunk). The caller stages the
        # buffers before replaying chunk 0 of a multimodal prompt; later chunks are cleared.
        x = self._apply_vision_merge(x, length=chunk_size)
        for layer in self.layers:
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos_buf,
                    sin=sin_buf,
                    mode="prefill",
                    page_table=full_page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                )
            else:
                # valid_len=None: no GDN mask; trace-safe static conv capture (matches valid_len==chunk_size).
                x_new = layer.forward(x, mode="prefill", chunk_size=self.args.gdn_chunk_size, valid_len=None)
            ttnn.deallocate(x)
            x = x_new
        return x

    def capture_prefill_trace_chunked(
        self, device, page_table, chunk_size=2048, warmup_masked_buckets=True, capture_chunk_trace=True
    ):
        """Capture one chunk's all-layer prefill as a trace; replayed per chunk.

        Chunk-outer prefill stays under the 4 GiB trace limit at long context.
        Flexible SDPA (runtime chunk_start) makes one trace serve all chunk positions.

        capture_chunk_trace=False warms the masked-bucket programs but skips parking the chunk trace.
        The batched (B>1) vLLM path passes capture_chunk_trace=True with the PERSISTENT B=1 prefill
        scratch bound (_bind_gdn_prefill_scratch), so the trace bakes that scratch's addresses and
        long prompts replay the traced chunk path per user (prefill_paged_slots rebinds the scratch)."""
        if self.num_devices > 1:
            return self._capture_prefill_trace_chunked_tp(
                device,
                page_table,
                chunk_size=chunk_size,
                warmup_masked_buckets=warmup_masked_buckets,
                capture_chunk_trace=capture_chunk_trace,
            )
        assert self._deltanet_external_states is not None, "Call allocate_kv_caches first"
        assert chunk_size % 128 == 0, f"chunk_size {chunk_size} must be a multiple of 128"
        B = 1
        block_size = get_block_size(self._paged_kv_caches)
        assert chunk_size % block_size == 0, "Prefill chunks must align with KV cache blocks"
        blocks_per_chunk = chunk_size // block_size
        prepared_key = (chunk_size, tuple(page_table.shape))
        if getattr(self, "_chunked_prepared_key", None) == prepared_key:
            if capture_chunk_trace and self._chunked_trace_id is None:
                self._capture_prepared_prefill_chunk(device)
            return
        if getattr(self, "_chunked_prepared_key", None) is not None:
            raise ValueError("Prefill trace geometry cannot change while its persistent buffers are live")

        if self._chunked_trace_id is not None:
            ttnn.release_trace(device, self._chunked_trace_id)
            self._chunked_trace_id = None

        self._chunked_chunk_size = chunk_size

        # Allocate the vision-splice buffers BEFORE warmup so the fixed-shape ttnn.where in
        # _forward_prefill_chunk / _forward_prefill_chunk_masked compiles in the warmup pass (and
        # is captured), never at request time. Zero-initialised -> identity for text-only.
        self._alloc_vision_merge_buffers(device, chunk_size)

        # ---- Persistent per-chunk input buffers (addresses baked into the trace) ----
        self._chunk_token_buf = ttnn.from_torch(
            torch.zeros(B, chunk_size, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self._chunk_start_idx_tensor = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._chunk_full_page_table_buf = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._chunk_page_table_buf = ttnn.from_torch(
            page_table[:, :blocks_per_chunk].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        # TP handoff: add ReplicateTensorToMesh for cos/sin (parity with tt/rope.py).
        self._chunk_cos_buf = ttnn.from_torch(
            self.rope.cos_cpu[:chunk_size].unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._chunk_sin_buf = ttnn.from_torch(
            self.rope.sin_cpu[:chunk_size].unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Keep the currently-bound state: normal serving uses external B=1 buffers, while
        # batched serving binds its persistent B=1 prefill scratch before entering here.
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            dn.conv_state_q = None
            dn.conv_state_k = None
            dn.conv_state_v = None
            if dn.split_conv_state is not None:
                for buf in dn.split_conv_state:
                    ttnn.deallocate(buf)
                dn.split_conv_state = None
            dn._chunk_inplace_state = True
        self._init_dn_zero_buffers()

        # Warmup outside trace: compile per-chunk programs.
        self._reset_dn_state_inplace()
        warmup_out = self._forward_prefill_chunk(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.deallocate(warmup_out)
        ttnn.synchronize_device(device)

        # Warmup masked-bucket programs outside trace (same GDN mode as serving).
        # Dummy prefills dirty state/KV; reset below before capture.
        if warmup_masked_buckets:
            self.warmup_prefill_masked_buckets(page_table)

        self._chunked_prepared_key = prepared_key
        self._reset_dn_state_inplace()
        if capture_chunk_trace:
            self._capture_prepared_prefill_chunk(device)

    def _capture_prepared_prefill_chunk(self, device):
        """Warm at the final allocator layout, then capture with persistent inputs."""
        # Decode/MTP warmup may allocate persistent L1 buffers between prefill
        # preparation and capture. Auto-selected matmul programs depend on the
        # available L1, so Phase 1's programs alone do not cover that layout.
        self._reset_dn_state_inplace()
        warmup_out = self._forward_prefill_chunk(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.deallocate(warmup_out)
        ttnn.synchronize_device(device)
        self._reset_dn_state_inplace()
        self._chunked_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._chunked_trace_output = self._forward_prefill_chunk(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.end_trace_capture(device, self._chunked_trace_id, cq_id=0)
        logger.info("Chunked prefill trace captured successfully!")

    def _capture_prefill_trace_chunked_tp(
        self, device, page_table, chunk_size=2048, warmup_masked_buckets=True, capture_chunk_trace=True
    ):
        """TP fork of capture_prefill_trace_chunked.

        Replicated persistent buffers; rope_tp cos/sin; GDN uses _stable_state (not external buffers).
        Trace replays _forward_prefill_chunk_tp."""
        assert self._deltanet_external_states is not None, "Call allocate_kv_caches first"
        assert chunk_size % 128 == 0, f"chunk_size {chunk_size} must be a multiple of 128"
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size

        if self._chunked_trace_id is not None:
            ttnn.release_trace(device, self._chunked_trace_id)
            self._chunked_trace_id = None
        self._chunked_chunk_size = chunk_size

        # Allocate the hidden-sharded vision-splice buffers BEFORE warmup so the fixed-shape
        # ttnn.where in the TP forwards compiles in the warmup pass (and is captured), never at
        # request time. Zero-initialised -> identity for text-only.
        self._alloc_vision_merge_buffers(device, chunk_size)

        rep = ttnn.ReplicateTensorToMesh(device)
        B = 1
        # Persistent per-chunk inputs (replicated; addresses baked into trace).
        self._chunk_token_buf = ttnn.from_torch(
            torch.zeros(B, chunk_size, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._chunk_start_idx_tensor = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._chunk_full_page_table_buf = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=rep
        )
        self._chunk_page_table_buf = ttnn.from_torch(
            page_table[:, :blocks_per_chunk].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, chunk_size)
        self._chunk_cos_buf = ttnn.from_torch(
            cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )
        self._chunk_sin_buf = ttnn.from_torch(
            sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )

        # Warmup outside trace: compile per-chunk programs.
        self._reset_gdn_state_for_new_sequence()
        warmup_out = self._forward_prefill_chunk_tp(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.deallocate(warmup_out)
        ttnn.synchronize_device(device)

        # Warmup masked-bucket/tail programs outside trace (same GDN mode; avoids trace clobber).
        if warmup_masked_buckets:
            self.warmup_prefill_masked_buckets(page_table)

        if not capture_chunk_trace:
            # Batched (B>1) vLLM path: masked-bucket programs are warmed above; skip parking the
            # chunk trace (it would bake the B=1 prefill scratch that is freed after warmup, and
            # batched serving handles short prompts only). num_full==0 prompts never need it.
            self._chunked_trace_id = None
            self._reset_gdn_state_for_new_sequence()
            logger.info("Masked-bucket prefill programs (TP) warmed; chunk trace skipped (batched path).")
            return

        # Capture trace.
        self._reset_gdn_state_for_new_sequence()
        self._chunked_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._chunked_trace_output = self._forward_prefill_chunk_tp(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.end_trace_capture(device, self._chunked_trace_id, cq_id=0)
        logger.info("Chunked prefill trace (TP) captured successfully!")

    # ----------------------------------------------------------------------- #
    # Traced batched SHORT-prompt prefill (B=32 / ISL<=128)
    # ----------------------------------------------------------------------- #
    # The traced chunk body and GDN chunk-seq kernel are B=1 (the kernel caps
    # BH=B*Nv_tp at ~32 => B<=4 at TP=4). So capture ONE B=1 full-bucket(128) trace and
    # replay it once per user: each replay DMAs the user's token slice + page-table row
    # into persistent buffers, execute_trace, then copy the B=1 GDN state into row u of
    # the batched [B,...] decode buffer. Full bucket => valid_len=None (trace-safe; a
    # one-hot mask in-trace TT_FATALs). Avoids per-layer host dispatch and per-op from_torch.

    def _alloc_gdn_scratch_b1(self):
        """Allocate a dedicated B=1 GDN state set on every GDN layer, distinct from the
        batched [B,...] decode buffer. Returns the prior batched bindings for the caller to
        restore for decode. MUST run before trace capture (allocates buffers)."""
        prev = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            prev.append(
                (
                    dn,
                    dn.B,
                    dn.rec_state,
                    dn.conv_states,
                    dn.conv_carry,
                    dn._zero_conv0,
                    dn._stable_state,
                )
            )
            # reset_state allocates against self.B, so set B=1 first.
            dn.B = 1
            dn.reset_state()  # builds rec_state [1,Nv,Dk,Dv], conv_states[*] [1,1,D], conv_carry, _zero_conv0
            dn._stable_state = True  # in-place carry so the trace's baked addresses survive replays
        return prev

    def _restore_gdn_batched(self, prev):
        """Restore the batched [B,...] GDN bindings saved by _alloc_gdn_scratch_b1 and free
        the B=1 scratch, so decode reads the assembled batched state."""
        for dn, B_b, rec_b, conv_b, carry_b, zero0_b, stable_b in prev:
            # Free the B=1 scratch allocated for the prefill trace.
            if dn.rec_state is not None:
                ttnn.deallocate(dn.rec_state)
            for cs in dn.conv_states or []:
                ttnn.deallocate(cs)
            if dn.conv_carry is not None:
                ttnn.deallocate(dn.conv_carry)
            if dn._zero_conv0 is not None:
                ttnn.deallocate(dn._zero_conv0)
            # Rebind the batched decode buffers.
            dn.B = B_b
            dn.rec_state = rec_b
            dn.conv_states = conv_b
            dn.conv_carry = carry_b
            dn._zero_conv0 = zero0_b
            dn._stable_state = stable_b

    def _ensure_gdn_prefill_scratch(self):
        """Allocate persistent B=1 state buffers used by the prefill trace."""
        if self._gdn_prefill_scratch is not None:
            return
        scratch = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            rec_shape = [1] + list(dn.recurrent_state.shape)[1:]
            conv_shape = [1] + list(dn.fused_conv_state.shape)[1:]
            rec = ttnn.zeros(
                rec_shape,
                dtype=dn.recurrent_state.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT if self._batched_single_device else ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            conv = ttnn.zeros(
                conv_shape,
                dtype=dn.fused_conv_state.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT if self._batched_single_device else ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            scratch.append((dn, rec, conv))
        self._gdn_prefill_scratch = scratch
        if self._batched_single_device:
            first_dn = scratch[0][0]
            self._gdn_remap_scratch = tuple(
                ttnn.zeros(list(tensor.shape), dtype=tensor.dtype, layout=tensor.layout,
                           device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for tensor in (first_dn.recurrent_state, first_dn.fused_conv_state)
            )
            # Compile transfers before any traces are parked. Slot indices are runtime
            # arguments, so one representative warms all arrivals and remappings.
            from models.demos.blackhole.qwen36.tt.batch_state import copy_state_rows
            rows = list(range(self._reserved_gdn_batch_size))
            for live, prefill, remap in zip(
                (first_dn.recurrent_state, first_dn.fused_conv_state),
                scratch[0][1:], self._gdn_remap_scratch,
            ):
                copy_state_rows(live, prefill, [0], [0])
                copy_state_rows(prefill, live, [0], [0])
                copy_state_rows(live, remap, rows, rows)
                copy_state_rows(remap, live, rows, rows)

    def _bind_gdn_prefill_scratch(self):
        """Bind persistent B=1 state while prefill runs one request."""
        self._ensure_gdn_prefill_scratch()
        prev = []
        for dn, rec, conv in self._gdn_prefill_scratch:
            prev.append(
                (
                    dn,
                    dn.recurrent_state,
                    dn.fused_conv_state,
                    dn.split_conv_state,
                    dn.use_inplace_state,
                    dn._chunk_inplace_state,
                )
            )
            dn.recurrent_state = rec
            dn.fused_conv_state = conv
            dn.split_conv_state = None
            dn.use_inplace_state = True
            dn._chunk_inplace_state = True
        self._init_dn_zero_buffers()
        return prev

    def _unbind_gdn_prefill_scratch(self, prev):
        """Restore batched decode state without freeing persistent prefill buffers."""
        for dn, rec, conv, split, use_inplace, chunk_inplace in prev:
            dn.recurrent_state = rec
            dn.fused_conv_state = conv
            dn.split_conv_state = split
            dn.use_inplace_state = use_inplace
            dn._chunk_inplace_state = chunk_inplace
        self._init_dn_zero_buffers()

    def _snapshot_gdn_scratch(self):
        """Snapshot bound B=1 state before a throwaway trace-capture execution."""
        return [
            (ttnn.to_torch(dn.recurrent_state), ttnn.to_torch(dn.fused_conv_state))
            for layer in self.layers
            if not layer.is_full_attention
            for dn in [layer.attention]
        ]

    def _restore_gdn_scratch(self, snap):
        """Restore bound B=1 state in place, preserving trace-captured addresses."""
        for layer, (rec, conv) in zip((l for l in self.layers if not l.is_full_attention), snap):
            dn = layer.attention
            rec_device = ttnn.from_torch(
                rec,
                dtype=dn.recurrent_state.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            conv_device = ttnn.from_torch(
                conv,
                dtype=dn.fused_conv_state.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            ttnn.copy(rec_device, dn.recurrent_state)
            ttnn.copy(conv_device, dn.fused_conv_state)
            ttnn.deallocate(rec_device)
            ttnn.deallocate(conv_device)

    def capture_prefill_trace_bucket(self, device, page_table, bucket=128):
        """Capture ONE B=1 full-bucket prefill trace (all-layer forward, valid_len=None) for
        batched serving of prompts whose length is EXACTLY the bucket. GDN points at a B=1
        scratch. Replay once per user via prefill_traced_bucket_batched.

        Only full-bucket prompts are traced: valid_len cannot be masked inside a trace, so a
        short prompt would pad through the GDN recurrence and corrupt the decode state. Callers
        route actual_len < bucket prompts to eager prefill_paged_peruser.

        Args:
          device: mesh device.
          page_table: torch.Tensor [1, bpu] int32 — one user's row (buffer width fixed across
            replays; each replay DMAs a different user's row in).
          bucket: fixed bucket length (128 for ISL==128; must be a multiple of 128).
        """
        assert self.num_devices > 1, "capture_prefill_trace_bucket is the TP (num_devices>1) path"
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        assert bucket % 128 == 0, f"bucket {bucket} must be a multiple of 128 (GDN sub-chunk)"
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_bucket = bucket // block_size

        if getattr(self, "_bucket_trace_id", None) is not None:
            ttnn.release_trace(device, self._bucket_trace_id)
            self._bucket_trace_id = None
        self._bucket_size = bucket
        # _forward_prefill_chunk_tp sizes its reshape/loop from _chunked_chunk_size; point it
        # at the bucket. Save the prior value so release_prefill_trace_bucket can restore it
        # (a later chunked prefill on the same model must not be left at 128).
        self._chunked_chunk_size_prebucket = self._chunked_chunk_size
        self._chunked_chunk_size = bucket

        # Swap GDN to a dedicated B=1 scratch for capture + replay (the trace writes B=1).
        self._gdn_batched_prev = self._alloc_gdn_scratch_b1()

        rep = ttnn.ReplicateTensorToMesh(device)
        B = 1
        # Full-page-table buffer width MUST be a 32-multiple >= the SDPA's target_blocks so
        # forward_prefill_paged's zero-PAD branch (ttnn.zeros + ttnn.concat — a host write that
        # TT_FATALs in a trace) never runs during replay. Replays' _fit_pt_row pad to this width.
        buf_blocks = max(32, ((page_table.shape[-1] + 31) // 32) * 32)
        if page_table.shape[-1] < buf_blocks:
            page_table = torch.cat(
                [page_table, torch.zeros(page_table.shape[0], buf_blocks - page_table.shape[-1], dtype=torch.int32)],
                dim=1,
            )
        self._bucket_buf_blocks = buf_blocks
        # Persistent per-replay input buffers (replicated; addresses baked into the trace).
        self._bucket_token_buf = ttnn.from_torch(
            torch.zeros(B, bucket, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._bucket_start_idx_tensor = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._bucket_full_page_table_buf = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=rep
        )
        self._bucket_page_table_buf = ttnn.from_torch(
            page_table[:, :blocks_per_bucket].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, bucket)
        self._bucket_cos_buf = ttnn.from_torch(
            cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )
        self._bucket_sin_buf = ttnn.from_torch(
            sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )

        # Warmup OUTSIDE the trace: compile every program the capture + replays run (a compile
        # during replay would clobber the parked trace). Two sets: (a) the full-bucket forward
        # (the trace body); (b) the logit-select run eagerly after each replay.
        self._reset_gdn_state_for_new_sequence()
        warmup_out = self._forward_prefill_chunk_tp(
            self._bucket_token_buf,
            self._bucket_cos_buf,
            self._bucket_sin_buf,
            self._bucket_start_idx_tensor,
            self._bucket_full_page_table_buf,
            self._bucket_page_table_buf,
        )
        # Warm the logit-select at actual_len=bucket. Its program is fixed per bucket; actual_len
        # only changes the one-hot values (a host write), so one warmup covers every actual_len.
        warm_logits = self._masked_bucket_logits_tp(warmup_out, bucket, bucket)
        ttnn.deallocate(warm_logits)
        ttnn.deallocate(warmup_out)
        ttnn.synchronize_device(device)

        # Capture: snapshot the B=1 scratch, run the throwaway compile+capture passes, restore
        # so the addresses the trace baked in stay valid (both passes advance the in-place GDN
        # recurrence; KV at block 0 is harmlessly overwritten by the first real replay).
        self._reset_gdn_state_for_new_sequence()
        gdn_snap = self._snapshot_gdn_scratch()
        self._forward_prefill_chunk_tp(
            self._bucket_token_buf,
            self._bucket_cos_buf,
            self._bucket_sin_buf,
            self._bucket_start_idx_tensor,
            self._bucket_full_page_table_buf,
            self._bucket_page_table_buf,
        )
        self._bucket_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._bucket_trace_output = self._forward_prefill_chunk_tp(
            self._bucket_token_buf,
            self._bucket_cos_buf,
            self._bucket_sin_buf,
            self._bucket_start_idx_tensor,
            self._bucket_full_page_table_buf,
            self._bucket_page_table_buf,
        )
        ttnn.end_trace_capture(device, self._bucket_trace_id, cq_id=0)
        self._restore_gdn_scratch(gdn_snap)
        logger.info(f"Bucket({bucket}) prefill trace (TP) captured successfully!")

    def release_prefill_trace_bucket(self):
        """Release the captured bucket prefill trace + persistent buffers and restore the
        batched GDN bindings for decode. Called after prefill_traced_bucket_batched."""
        if getattr(self, "_bucket_trace_id", None) is not None:
            ttnn.release_trace(self.device, self._bucket_trace_id)
            self._bucket_trace_id = None
        for buf in (
            getattr(self, "_bucket_token_buf", None),
            getattr(self, "_bucket_start_idx_tensor", None),
            getattr(self, "_bucket_full_page_table_buf", None),
            getattr(self, "_bucket_page_table_buf", None),
            getattr(self, "_bucket_cos_buf", None),
            getattr(self, "_bucket_sin_buf", None),
        ):
            if buf is not None:
                ttnn.deallocate(buf)
        self._bucket_token_buf = None
        self._bucket_start_idx_tensor = None
        self._bucket_full_page_table_buf = None
        self._bucket_page_table_buf = None
        self._bucket_cos_buf = None
        self._bucket_sin_buf = None
        self._bucket_trace_output = None
        # Restore _chunked_chunk_size (capture pointed it at the bucket) for a later chunked prefill.
        if hasattr(self, "_chunked_chunk_size_prebucket"):
            self._chunked_chunk_size = self._chunked_chunk_size_prebucket
            del self._chunked_chunk_size_prebucket
        # Restore the batched [B,...] GDN decode buffers the prefill assembled into.
        if getattr(self, "_gdn_batched_prev", None) is not None:
            self._restore_gdn_batched(self._gdn_batched_prev)
            self._gdn_batched_prev = None

    def prefill_traced_bucket_batched(self, token_ids_list, page_table, valid_lens=None):
        """Traced batched short-prompt prefill: replay the captured B=1 full-bucket trace once
        per user, stitching each replay's B=1 GDN state into row u of the batched [B,...] decode
        buffer. Attention fills each user's physical blocks via the per-user page-table row
        (batch_idx=0 baked into the trace). Returns a list of B device logits [1, 1, vocab].

        CORRECTNESS CONTRACT: every user's actual_len MUST equal the captured bucket. The trace
        runs valid_len=None (no GDN mask); for a short prompt padded to the bucket that would push
        padding tokens through the GDN recurrence and corrupt the decode state. Short prompts must
        be routed to eager prefill_paged_peruser. Asserts actual_len == bucket and never pads.
        Call capture_prefill_trace_bucket first; release_prefill_trace_bucket before decode.
        """
        assert self.num_devices > 1, "prefill_traced_bucket_batched is the TP (num_devices>1) path"
        assert getattr(self, "_bucket_trace_id", None) is not None, "Call capture_prefill_trace_bucket first"
        bucket = self._bucket_size
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_bucket = bucket // block_size
        rep = ttnn.ReplicateTensorToMesh(self.device)

        B = len(token_ids_list)
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert page_table_torch.shape[0] == B, "page_table must have one row per user"

        # Pad/clip each request's page-table row to the captured buffer width. Keep the
        # [1, buf_blocks] batch dim — copy_host_to_device_tensor requires identical logical shapes.
        buf_blocks = int(self._bucket_full_page_table_buf.shape[-1])

        def _fit_pt_row(row):
            row = row.reshape(1, -1)  # [1, bpu] -> ensure 2D
            if row.shape[1] < buf_blocks:
                row = torch.cat([row, torch.zeros(1, buf_blocks - row.shape[1], dtype=row.dtype)], dim=1)
            elif row.shape[1] > buf_blocks:
                row = row[:, :buf_blocks]
            return row.contiguous()

        # Per-user logits are read to HOST during the loop and re-uploaded at the end: each replay
        # overwrites the persistent trace output, and the post-loop assembly churns device memory.
        host_logits = []  # torch [1, 1, vocab] (one replica) per user
        # Collect each replay's B=1 GDN state to assemble into the batched decode buffer. Snapshot
        # via a host to_torch round trip (NOT ttnn.clone() — that allocates from the same general
        # device pool the captured trace's own baked-address intermediates draw from, so the next
        # user's execute_trace() silently overwrites the "cloned" snapshot; confirmed via
        # checksumming a live tensor changing value with nothing writing to it, ruled out as a race
        # since an added synchronize_device() didn't change the deterministic wrong result).
        per_user_rec = []
        per_user_conv = []
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_states = [layer.attention for layer in self.layers if not layer.is_full_attention]

        try:
            for u in range(B):
                toks = token_ids_list[u]
                assert toks.shape[0] == 1, f"user {u}: token_ids must be [1, T_u]"
                actual = valid_lens[u] if valid_lens is not None else toks.shape[1]
                # CORRECTNESS: traced path serves ONLY full-bucket prompts (see docstring); short
                # prompts must be routed to eager prefill_paged_peruser.
                assert actual == bucket, (
                    f"user {u}: actual_len {actual} != bucket {bucket}; the traced bucket prefill "
                    f"only serves full-bucket prompts — route short prompts to prefill_paged_peruser"
                )

                # Zero the B=1 GDN scratch before each user (address-stable; the trace baked these in).
                self._reset_gdn_state_for_new_sequence()

                # Full bucket, no padding (actual == bucket).
                token_buf = toks[:, :bucket].to(torch.int32)

                # DMA this user's inputs into the persistent buffers (addresses preserved).
                tok_host = ttnn.from_torch(
                    token_buf, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
                )
                ttnn.copy_host_to_device_tensor(tok_host, self._bucket_token_buf)

                row = _fit_pt_row(page_table_torch[u])
                pt_host = ttnn.from_torch(
                    row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
                )
                ttnn.copy_host_to_device_tensor(pt_host, self._bucket_full_page_table_buf)
                cpt_host = ttnn.from_torch(
                    row[:, :blocks_per_bucket].contiguous(),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=None,
                    mesh_mapper=rep,
                )
                ttnn.copy_host_to_device_tensor(cpt_host, self._bucket_page_table_buf)

                # chunk_start=0 and cos/sin for [0, bucket) are baked in; no per-replay DMA needed.
                ttnn.execute_trace(self.device, self._bucket_trace_id, cq_id=0, blocking=False)
                # Sync before reading this user's state/logit: the trace writes hidden/rec_state/
                # conv_states in place and the next replay would overwrite them.
                ttnn.synchronize_device(self.device)

                # Gather this replay's B=1 GDN state for assembly after the loop. The next replay
                # resets the scratch IN PLACE, so snapshot now via host to_torch (see comment above).
                per_user_rec.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_states])
                per_user_conv.append(
                    [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_states]
                )

                # Logit at actual_len-1, read to HOST immediately (before the next replay overwrites
                # the trace output). Re-uploaded at the end.
                lg = self._masked_bucket_logits_tp(self._bucket_trace_output, actual, bucket)
                host_logits.append(ttnn.to_torch(lg, mesh_composer=comp)[0:1].clone())  # [1,1,vocab] one replica
                ttnn.deallocate(lg)

            ttnn.synchronize_device(self.device)
        finally:
            # Rebind the batched buffers regardless of success or failure (restore was deferred so
            # the loop could use the B=1 scratch) — otherwise a mid-loop assertion/exception (e.g. a
            # short prompt hitting the actual_len == bucket check) leaves every GDN layer pointed at
            # the B=1 scratch instead of the batched decode buffers.
            if getattr(self, "_gdn_batched_prev", None) is not None:
                self._restore_gdn_batched(self._gdn_batched_prev)
                self._gdn_batched_prev = None

        # Assemble the per-user states into row u in place (_stable_state path).
        self._assemble_per_user_gdn(per_user_rec, per_user_conv)

        # Re-upload the per-user logits as stable device tensors after all allocations.
        return self._reupload_host_logits(host_logits)

    def _assemble_per_user_gdn(self, per_user_rec, per_user_conv):
        """Stitch B per-user B=1 GDN states (host torch) into row u of the batched [B,...] decode
        buffers via assemble_batched_state. The batched GDN bindings MUST already be rebound (writes
        in place under _stable_state). Shared by prefill_traced_bucket_batched/prefill_chunked_peruser.

        per_user_rec[u][li]:  host rec_state snapshot for user u, GDN layer li (mesh dim 0 = devices).
        per_user_conv[u][li]: list of K host conv_states snapshots for user u, GDN layer li.
        """
        B = len(per_user_rec)
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        for li, dn in enumerate(dn_layers):
            K = dn.K
            D = dn.qkv_dim_tp
            rec_list = [
                ttnn.from_torch(
                    per_user_rec[u][li],
                    dtype=dn.rec_state.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=mapper,
                )
                for u in range(B)
            ]
            # Rebuild the [1, K-1, D] carry tensor the assembler expects from the per-slot
            # conv_states snapshot (slots 1..K-1; slot 0 is the shifted-out zero). Each slot
            # snapshot is [1, 1, D]; concat along dim 1 -> [1, K-1, D].
            conv_carry_list = []
            staging = []  # (per-slot tensors) to deallocate after assembly
            for u in range(B):
                slots = [
                    ttnn.from_torch(
                        per_user_conv[u][li][m],
                        dtype=dn.conv_states[m].dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        mesh_mapper=mapper,
                    )
                    for m in range(1, K)
                ]
                staging.extend(slots)
                conv_carry_list.append(ttnn.concat(slots, dim=1) if K - 1 > 1 else ttnn.reshape(slots[0], (1, 1, D)))
            # assemble_batched_state takes ownership of rec_list + conv_carry_list and deallocates
            # them (gdn/tp.py); we only free the per-slot staging tensors it never sees.
            dn.assemble_batched_state(rec_list, conv_carry_list)
            for t in staging:
                ttnn.deallocate(t)

    def _reupload_host_logits(self, host_logits):
        """Re-upload per-user host logits (read to host during a batched prefill loop) as stable
        replicated device tensors [1, 1, vocab] — the prefill_paged_peruser return contract.
        Done after all per-user execute/assembly allocations so the returned tensors are stable."""
        return [
            ttnn.from_torch(
                hl,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for hl in host_logits
        ]

    def prefill_paged_slots(self, token_ids_list, page_table, empty_slots, valid_lens=None, start_positions=None):
        """Process prompt segments into independent live request slots.

        Tokens contain the full prefix; valid_lens is the cumulative end and
        start_positions the already-computed prefix length (zero for a new slot).
        State transfers remain on device and preserve the other decode slots.
        """
        N = len(token_ids_list)
        assert len(empty_slots) == N, "one slot per request"
        starts = [0] * N if start_positions is None else [int(p) for p in start_positions]
        assert len(starts) == N, "one start position per request"
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert pt.shape[0] == N, "page_table must have one row per request"

        actuals = []
        for u in range(N):
            toks = token_ids_list[u]
            assert toks.shape[0] == 1, f"request {u}: token_ids must be [1, T_u]"
            actual = int(valid_lens[u]) if valid_lens is not None else toks.shape[1]
            assert actual >= 1, f"request {u}: empty prompt (actual_len={actual})"
            assert 0 <= starts[u] < actual <= toks.shape[1]
            if starts[u] and not self._batched_single_device:
                raise ValueError("Resumable prefill requires single-device batched serving")
            actuals.append(actual)
        packs, serial = self._plan_packed_prefill(actuals, starts)

        # Bind the persistent B=1 GDN prefill scratch: prefill runs B=1 and its per-sequence reset
        # would otherwise zero the batched [B,...] decode buffer (clobbering the live rows). This is
        # the SAME scratch whose addresses the chunk-prefill trace baked at warmup, so the traced
        # long-prompt path replays correctly; short prompts use the masked bucket on the same scratch.
        prev = self._bind_gdn_prefill_scratch()
        host_logits = [None] * N
        try:
            # Fresh short prompts sharing a bucket run as one packed forward; each row's GDN state
            # is committed straight into its decode slot.
            for bucket, members in packs:
                logits = self._prefill_packed(
                    bucket,
                    [token_ids_list[u] for u in members],
                    [actuals[u] for u in members],
                    [self._fit_page_table_width(pt[u : u + 1]) for u in members],
                    [int(empty_slots[u]) for u in members],
                    prev,
                )
                for u, row_logits in zip(members, logits):
                    host_logits[u] = row_logits
            for u in serial:
                toks = token_ids_list[u]
                actual = actuals[u]
                start = starts[u]
                if start:
                    from models.demos.blackhole.qwen36.tt.batch_state import copy_state_rows
                    for dn, recurrent, convolution, *_ in prev:
                        copy_state_rows(recurrent, dn.recurrent_state, [int(empty_slots[u])], [0])
                        copy_state_rows(convolution, dn.fused_conv_state, [int(empty_slots[u])], [0])
                lg = self.prefill_traced_chunked(
                    toks[:, :actual], pt[u : u + 1], actual_len=actual, start_position=start
                )
                host_logits[u] = ttnn.to_torch(lg).reshape(-1, self.args.vocab_size)[:1].float().view(1, 1, -1)
                ttnn.deallocate(lg)
                # Copy the B=1 scratch state directly into this request's persistent
                # decode slot while the batched buffers remain reachable through prev.
                self._write_bound_gdn_scratch_slot(prev, int(empty_slots[u]))
        finally:
            # Always rebind the batched decode buffers (a mid-loop assert must not leave GDN on scratch).
            # Does NOT free the scratch — it persists for the next request and keeps the trace valid.
            self._unbind_gdn_prefill_scratch(prev)
            if not self._batched_single_device:
                for dn, *_ in prev:
                    dn._restore_split_conv_from_fused()

        return host_logits

    # Packed fresh-prompt prefill (single-device batched serving). Short prefill is bound by
    # expert-weight reads, so G fresh prompts sharing a masked bucket run as ONE forward:
    # embeddings, norms, residuals and MoE see the flattened [1, G*bucket] rows (the shapes of the
    # serial G*bucket bucket), GDN runs B=G with per-row valid_len masks, and full attention runs
    # per row with that row's page table. Every shape here has G*bucket <= 2048.
    _PREFILL_PACKED_SHAPES = {128: (16, 8, 4, 2), 256: (8, 4, 2), 512: (4, 2), 1024: (2,)}

    def _plan_packed_prefill(self, lengths, starts):
        """Split request indices into packed (bucket, members) groups and serial leftovers.

        Only fresh requests (start 0) of a bucket with warmed packed shapes are packed, greedily
        into the largest warmed group size; resumed segments, long prompts and leftovers run serially."""
        if not self._packed_prefill_warmed:
            return [], list(range(len(lengths)))
        by_bucket = {}
        serial = []
        for u, (length, start) in enumerate(zip(lengths, starts)):
            bucket = self._mask_bucket_for(length)
            if start == 0 and any((bucket, g) in self._packed_prefill_warmed for g in self._PREFILL_PACKED_SHAPES.get(bucket, ())):
                by_bucket.setdefault(bucket, []).append(u)
            else:
                serial.append(u)
        packs = []
        for bucket, members in by_bucket.items():
            while members:
                group = next(
                    (g for g in self._PREFILL_PACKED_SHAPES[bucket] if g <= len(members) and (bucket, g) in self._packed_prefill_warmed),
                    1,
                )
                if group == 1:
                    serial.extend(members)
                    break
                packs.append((bucket, members[:group]))
                members = members[group:]
        return packs, sorted(serial)

    def _fit_page_table_width(self, page_table):
        """Pad/clip page-table columns to the warmed prefill width (vLLM pads to its own width).
        Trailing entries index blocks past the prompt and are never read by causal SDPA."""
        buf = getattr(self, "_chunk_full_page_table_buf", None)
        if buf is None:
            return page_table
        width = int(buf.shape[-1])
        if page_table.shape[1] < width:
            pad = torch.zeros(page_table.shape[0], width - page_table.shape[1], dtype=page_table.dtype)
            return torch.cat([page_table, pad], dim=1)
        return page_table[:, :width]

    def _ensure_gdn_packed_scratch(self):
        """Allocate the GDN state rows (plus zero sources) one packed GDN call works on, shared by
        every GDN layer: each call zeroes them, runs, and commits its rows to the decode slots.

        The sequential-scan kernel runs one (row, value head) pair per core (BH <= grid cores), so
        a packed group's GDN runs in calls of _gdn_pack_rows rows: the largest count that fits the
        grid and divides every packed group size."""
        if self._gdn_packed_scratch is not None:
            return
        self._ensure_gdn_prefill_scratch()
        dn, rec_template, conv_template = self._gdn_prefill_scratch[0]
        grid = self.device.compute_with_storage_grid_size()
        groups = [g for sizes in self._PREFILL_PACKED_SHAPES.values() for g in sizes]
        fit = max(1, int(grid.x) * int(grid.y) // int(dn.num_v_heads))
        self._gdn_pack_rows = max(r for r in range(1, fit + 1) if all(g % r == 0 for g in groups))
        self._gdn_packed_scratch = tuple(
            ttnn.zeros(
                [self._gdn_pack_rows] + list(template.shape)[1:],
                dtype=template.dtype,
                layout=template.layout,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for template in (rec_template, conv_template, rec_template, conv_template)
        )

    def _prefill_packed(self, bucket, token_rows, lengths, page_rows, slots, bound_state):
        """One masked forward over G fresh prompts padded to `bucket`.

        Row u holds token_rows[u][:, :lengths[u]] at packed rows [u*bucket, u*bucket+lengths[u]).
        Each GDN layer's final state for row u is written into decode slot slots[u] of the live
        buffers in bound_state (the prev of _bind_gdn_prefill_scratch). Returns host logits
        [1, 1, vocab] per row, in order."""
        from models.demos.blackhole.qwen36.tt.batch_state import copy_state_rows

        G = len(lengths)
        rows = G * bucket
        dim = self.args.dim
        dram = ttnn.DRAM_MEMORY_CONFIG
        rec_p, conv_p, rec_zero, conv_zero = self._gdn_packed_scratch
        pack_rows = self._gdn_pack_rows
        block_size = get_block_size(self._paged_kv_caches)
        self._mtp_seeded = None
        # Text-only packed rows: plain 1D RoPE, and every row starts at position 0.
        self._build_request_rope(token_rows[0][:, : lengths[0]], None)

        tokens = torch.zeros(1, rows, dtype=torch.int32)
        for u, (row_tokens, length) in enumerate(zip(token_rows, lengths)):
            tokens[0, u * bucket : u * bucket + length] = row_tokens[0, :length].to(torch.int32)
        tok = ttnn.from_torch(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        x = self.embd(tok)
        x = ttnn.to_memory_config(x, dram)
        ttnn.deallocate(tok)
        cos, sin = self.rope.get_prefill_rot_mats(0, bucket)
        csi = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        full_pts, chunk_pts = [], []
        for row, length in zip(page_rows, lengths):
            full_pts.append(ttnn.from_torch(row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device))
            # Fill K/V only for real blocks; padded writes would land in other requests' blocks.
            chunk_pts.append(
                ttnn.from_torch(
                    row[:, : num_blocks_in_seq(length, block_size)].contiguous(),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                )
            )
        live = {id(dn): (recurrent, convolution) for dn, recurrent, convolution, *_ in bound_state}
        # Padding rows past each segment's length take no MoE expert work. Always passed (also
        # for full segments) so warmup compiles the masked MoE programs serving uses.
        moe_rows = self._moe_valid_rows(lengths, bucket)

        def segment(tensor, first, count):
            """Rows [first, first+count) of the packed [1, rows, dim] tensor (whole tensor if all)."""
            if count == G:
                return tensor
            return ttnn.slice(
                tensor, (0, first * bucket, 0), (1, (first + count) * bucket, dim), memory_config=dram
            )

        for layer in self.layers:
            attn_in = layer.attention_norm(x, mode=Mode.PREFILL, norm_config=None)
            if layer.is_full_attention:
                outputs = []
                for u in range(G):
                    row_in = segment(attn_in, u, 1)
                    outputs.append(
                        layer.attention.forward(
                            row_in,
                            cos,
                            sin,
                            page_table=full_pts[u],
                            chunk_page_table=chunk_pts[u],
                            chunk_start_idx_tensor=csi,
                        )
                    )
                    ttnn.deallocate(row_in)
            else:
                dn = layer.attention
                recurrent, convolution = live[id(dn)]
                bound = (dn.recurrent_state, dn.fused_conv_state)
                outputs = []
                try:
                    for first in range(0, G, pack_rows):
                        part = segment(attn_in, first, pack_rows)
                        ttnn.copy(rec_zero, rec_p)
                        ttnn.copy(conv_zero, conv_p)
                        dn.recurrent_state, dn.fused_conv_state = rec_p, conv_p
                        grouped = dn.forward(
                            ttnn.reshape(part, [pack_rows, bucket, dim]),
                            mode="chunk",
                            chunk_size=dn.long_prefill_chunk_size,
                            valid_len=list(lengths[first : first + pack_rows]),
                        )
                        dn.recurrent_state, dn.fused_conv_state = bound
                        if part is not attn_in:
                            ttnn.deallocate(part)
                        outputs.append(ttnn.reshape(grouped, [1, pack_rows * bucket, dim]))
                        copy_state_rows(rec_p, recurrent, list(range(pack_rows)), slots[first : first + pack_rows])
                        copy_state_rows(conv_p, convolution, list(range(pack_rows)), slots[first : first + pack_rows])
                finally:
                    dn.recurrent_state, dn.fused_conv_state = bound
            if len(outputs) == 1:
                attn_out = outputs[0]
            else:
                attn_out = ttnn.concat(outputs, dim=1, memory_config=dram)
                for output in outputs:
                    ttnn.deallocate(output)
            ttnn.deallocate(attn_in)
            h = ttnn.add(x, attn_out)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(x)
            ff_in = layer.ffn_norm(h, mode=Mode.PREFILL, norm_config=None)
            ff_out = (
                layer.feed_forward.forward(ff_in)
                if moe_rows is None
                else layer.feed_forward.forward(ff_in, valid_rows=moe_rows)
            )
            ttnn.deallocate(ff_in)
            x = ttnn.add(h, ff_out)
            ttnn.deallocate(h)
            ttnn.deallocate(ff_out)

        for tensor in (cos, sin, csi, *full_pts, *chunk_pts):
            ttnn.deallocate(tensor)
        if moe_rows is not None:
            ttnn.deallocate(moe_rows)
        # One-hot row selection (fixed program per packed shape) of each row's last real token.
        sel = torch.zeros(1, G, rows, dtype=torch.float32)
        for u, length in enumerate(lengths):
            sel[0, u, u * bucket + length - 1] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        last = ttnn.matmul(sel_tt, x)
        ttnn.deallocate(sel_tt)
        ttnn.deallocate(x)
        last_dram = ttnn.to_memory_config(last, dram)
        normalized = self.norm(last_dram, mode=Mode.PREFILL)
        ttnn.deallocate(last_dram)
        logits = self._lm_head(normalized)
        ttnn.deallocate(normalized)
        host = ttnn.to_torch(logits).reshape(-1, self.args.vocab_size)[:G].float()
        ttnn.deallocate(logits)
        return [host[u].view(1, 1, -1) for u in range(G)]

    def warmup_prefill_packed(self, page_table, bound_state):
        """Compile every packed prefill shape before any trace is parked.

        Call in batched single-device serving with the B=1 prefill scratch bound (bound_state is
        _bind_gdn_prefill_scratch's return). Dummy rows write KV through page_table row 0 and GDN
        state into slots 0..G-1; serving overwrites both before use."""
        if not self._batched_single_device or self.mtp is not None:
            return
        self._ensure_gdn_packed_scratch()
        row = page_table[:1].contiguous()
        for bucket, groups in self._PREFILL_PACKED_SHAPES.items():
            for group in groups:
                if group > self._reserved_gdn_batch_size:
                    continue
                lengths = [max(1, bucket - (u * bucket) // group) for u in range(group)]
                tokens = [torch.zeros(1, length, dtype=torch.int32) for length in lengths]
                self._prefill_packed(bucket, tokens, lengths, [row] * group, list(range(group)), bound_state)
                self._packed_prefill_warmed.add((bucket, group))
        ttnn.synchronize_device(self.device)

    @staticmethod
    def _replace_device_row(destination, source, row):
        """Replace one dim-0 row without moving the destination through host memory."""
        batch = int(destination.shape[0])
        assert 0 <= row < batch
        if batch == 1:
            ttnn.copy(source, destination)
            return

        pieces = []
        temporary = []
        rank = len(destination.shape)
        if row:
            start = [0] * rank
            end = list(destination.shape)
            end[0] = row
            prefix = ttnn.slice(destination, start, end, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            pieces.append(prefix)
            temporary.append(prefix)
        pieces.append(source)
        if row + 1 < batch:
            start = [0] * rank
            start[0] = row + 1
            end = list(destination.shape)
            suffix = ttnn.slice(destination, start, end, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            pieces.append(suffix)
            temporary.append(suffix)

        updated = ttnn.concat(pieces, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(updated, destination)
        ttnn.deallocate(updated)
        for tensor in temporary:
            ttnn.deallocate(tensor)

    def _write_bound_gdn_scratch_slot(self, bound_state, slot):
        """Write the currently bound B=1 scratch into one persistent decode slot."""
        for dn, recurrent, convolution, *_ in bound_state:
            if self._batched_single_device:
                from models.demos.blackhole.qwen36.tt.batch_state import copy_state_rows
                copy_state_rows(dn.recurrent_state, recurrent, [0], [slot])
                copy_state_rows(dn.fused_conv_state, convolution, [0], [slot])
            else:
                self._replace_device_row(recurrent, dn.recurrent_state, slot)
                self._replace_device_row(convolution, dn.fused_conv_state, slot)

    @staticmethod
    def _remap_device_rows(state, indices):
        """Reorder dim-0 rows using device slices and one device concatenation."""
        if indices == list(range(len(indices))):
            return

        runs = []
        run_start = indices[0]
        run_end = run_start + 1
        for source in indices[1:]:
            if source == run_end:
                run_end += 1
            else:
                runs.append((run_start, run_end))
                run_start = source
                run_end = source + 1
        runs.append((run_start, run_end))

        rank = len(state.shape)
        pieces = []
        for start_row, end_row in runs:
            start = [0] * rank
            start[0] = start_row
            end = list(state.shape)
            end[0] = end_row
            pieces.append(ttnn.slice(state, start, end, memory_config=ttnn.DRAM_MEMORY_CONFIG))

        updated = ttnn.concat(pieces, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(updated, state)
        ttnn.deallocate(updated)
        for piece in pieces:
            ttnn.deallocate(piece)

    def _remap_gdn_slots(self, remap):
        """Apply vLLM's destination-to-source slot remap entirely on device."""
        remap_host = remap if isinstance(remap, torch.Tensor) else torch.as_tensor(remap)
        batch = int(remap_host.numel())
        indices = [
            int(src) if 0 <= int(src) < batch else dst
            for dst, src in enumerate(remap_host.reshape(-1).tolist())
        ]
        if indices == list(range(batch)):
            return
        if self._batched_single_device:
            from models.demos.blackhole.qwen36.tt.batch_state import copy_state_rows
            self._ensure_gdn_prefill_scratch()
            destinations = list(range(batch))
            for recurrent, convolution in self._deltanet_external_states:
                for state, scratch in zip((recurrent, convolution), self._gdn_remap_scratch):
                    copy_state_rows(state, scratch, indices, destinations)
                    copy_state_rows(scratch, state, destinations, destinations)
            return
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            for state in (dn.recurrent_state, dn.fused_conv_state):
                assert int(state.shape[0]) == batch
                self._remap_device_rows(state, indices)
            dn._restore_split_conv_from_fused()

    def prefill_chunked_peruser(self, token_ids_list, page_table, valid_lens=None):
        """Batched per-user LONG-prefill (TP, eager). Runs the single-user chunk-outer path
        (prefill_traced_chunked) for each user into a B=1 GDN scratch, then stitches each user's
        final GDN state into row u of the batched [B,...] decode buffers.

        Handles ANY prompt length with exact valid_len masking (no last-token padding through the
        GDN recurrence): short -> masked bucket; long -> chunk-outer + masked tail. The long-prompt
        counterpart to prefill_traced_bucket_batched (full-bucket-only) and prefill_paged_peruser
        (single-pass). Call allocate_kv_caches(batch_size=B) first.

        token_ids_list: list of B torch.Tensor [1, T_u] (lengths may differ).
        page_table:      torch.Tensor [B, max_blocks_per_seq] int32 — row u = user u's blocks.
                         IMPORTANT: max_blocks_per_seq MUST be a multiple of 8. The chunked SDPA
                         reads each row as a ROW_MAJOR int32 stick requiring stick_size
                         (width * 4 bytes) % 32 == 0, i.e. width % 8 == 0. A misaligned width
                         makes the SDPA read the wrong KV.
        valid_lens:      optional list of B ints (real token counts); defaults to each T_u.
        Returns:         list of B ttnn logits [1, 1, vocab] (replicated; at valid_len-1).
        """
        assert self.num_devices > 1, "prefill_chunked_peruser is the TP (num_devices>1) path"
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        # The chunked path keys its chunk math on _chunked_chunk_size (default 2048); a parked
        # bucket trace leaves it at 128, breaking num_full/tail sizing. Require it released first.
        assert getattr(self, "_bucket_trace_id", None) is None, (
            "release the bucket prefill trace before prefill_chunked_peruser " "(_chunked_chunk_size would be wrong)"
        )
        assert self._chunked_chunk_size in (None, 2048), (
            f"prefill_chunked_peruser expects the 2048-token chunk; got _chunked_chunk_size="
            f"{self._chunked_chunk_size}"
        )

        B = len(token_ids_list)
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert page_table_torch.shape[0] == B, "page_table must have one row per user"

        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]

        # Swap every GDN layer to a B=1 scratch (per-user path is B=1); assembled into the batched
        # buffers after the loop. prev holds the batched bindings for restore.
        prev = self._alloc_gdn_scratch_b1()
        host_logits = []  # torch [1, 1, vocab] (one replica) per user
        # Snapshot each user's B=1 GDN state via a host to_torch round trip (NOT ttnn.clone() — see
        # prefill_traced_bucket_batched for why the device-side clone path is broken: it aliases
        # the captured trace's own baked-address intermediates and gets silently overwritten by
        # the next user's execute_trace()).
        per_user_rec = []  # per user, list over GDN layers of host rec_state snapshots
        per_user_conv = []  # per user, list over GDN layers of [list of K conv snapshots]
        try:
            for u in range(B):
                toks = token_ids_list[u]
                assert toks.shape[0] == 1, f"user {u}: token_ids must be [1, T_u]"
                vlen = valid_lens[u] if valid_lens is not None else toks.shape[1]
                assert 1 <= vlen <= toks.shape[1], f"user {u}: valid_len {vlen} not in [1, {toks.shape[1]}]"

                # Per-user long path (from scratch) into the B=1 scratch: carries state across
                # chunks, masks the tail exactly, writes user u's KV via the page-table row.
                lg = self.prefill_traced_chunked(toks, page_table_torch[u : u + 1].contiguous(), actual_len=vlen)
                # Read the logit to HOST immediately (the next prefill + post-loop assembly churn
                # device memory and would otherwise corrupt the returned tensor).
                ttnn.synchronize_device(self.device)
                host_logits.append(ttnn.to_torch(lg, mesh_composer=comp)[0:1].clone())  # [1,1,vocab] one replica
                ttnn.deallocate(lg)

                # Snapshot this user's B=1 GDN state for assembly after the loop (host round trip —
                # the B=1 scratch is reset IN PLACE for the next user). slot 0 of conv_states is
                # the zeroed shifted-out tap; only slots 1..K-1 carry state.
                per_user_rec.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_layers])
                per_user_conv.append(
                    [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_layers]
                )
        finally:
            # Restore the batched [B,...] GDN decode buffers and free the B=1 scratch. The clones
            # are independent allocations, so freeing the scratch here does not touch them.
            self._restore_gdn_batched(prev)

        ttnn.synchronize_device(self.device)
        # Stitch the per-user states into row u of the (now-rebound) batched decode buffers.
        self._assemble_per_user_gdn(per_user_rec, per_user_conv)
        # Re-upload the per-user logits as stable device tensors (prefill_paged_peruser contract).
        return self._reupload_host_logits(host_logits)

    def _forward_prefill_chunk_eager(self, token_slice, chunk_start, page_table):
        """Eager final partial-chunk prefill (< chunk_size). GDN zero-pads to 128-multiple internally
        (not bucket padding). Returns hidden [1,T_tail_padded,hidden_size]."""
        T_tail = token_slice.shape[1]
        block_size = 64
        tok = ttnn.from_torch(
            token_slice.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        x = self.embd(tok)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        cos, sin = self.rope.get_prefill_rot_mats(chunk_start, T_tail)
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        blk0 = chunk_start // block_size
        blkN = math.ceil((chunk_start + T_tail) / block_size)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        for layer in self.layers:
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="prefill",
                    page_table=full_pt,
                    chunk_page_table=chunk_pt,
                    chunk_start_idx=chunk_start,
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size)
            ttnn.deallocate(x)
            x = x_new
        return x

    # Fixed buckets for masked tail/short prefill. Lengths round up here -> bounded compile set.
    # All 128-multiples (GDN sub-chunk). Masked GDN in DRAM avoids L1 clash at bucket 512.
    # Diverges from get_padded_prefill_len: 256/512 for short TTFT; GDN needs exact valid_len mask.
    _PREFILL_MASK_BUCKETS = (128, 256, 512, 1024, 2048)

    @classmethod
    def _mask_bucket_for(cls, length):
        """Smallest fixed bucket >= length (falls back to the next 128-multiple)."""
        for b in cls._PREFILL_MASK_BUCKETS:
            if length <= b:
                return b
        return ((length + 127) // 128) * 128

    def _forward_prefill_chunk_masked(
        self, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=True, vision_tokens=None
    ):
        """Single masked fixed-bucket prefill forward over `bucket` positions.

        First valid_len tokens real; rest padded. Attn runs full bucket; GDN masks via valid_len.
        Returns hidden [1,bucket,hidden] or [1,1,bucket,hidden] (TP)."""
        if self.num_devices > 1:
            return self._forward_prefill_chunk_masked_tp(
                token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=flex_sdpa, vision_tokens=vision_tokens
            )
        block_size = get_block_size(self._paged_kv_caches)
        tok = ttnn.from_torch(
            token_buf.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        x = self.embd(tok)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        # Trace-safe vision splice: a fixed-shape ttnn.where over persistent buffers (compiled
        # at warmup; mask==0 -> identity, so text-only is unchanged). The caller stages the
        # buffers (prefill_masked_bucket -> _set_vision_merge). No-op until a trace is captured.
        x = self._apply_vision_merge(x, length=bucket)
        cos, sin = self.rope.get_prefill_rot_mats(chunk_start, bucket)
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        blk0 = chunk_start // block_size
        # Fill K/V only for real blocks (ceil(valid_len/64)); padded writes would corrupt block 0.
        blkN = num_blocks_in_seq(chunk_start + valid_len, block_size)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        # Flexible SDPA (device chunk_start): one program per bucket for any tail position.
        # Host-int chunk_start compiles per position and can clobber parked trace.
        csi_tensor = ttnn.from_torch(
            torch.tensor([chunk_start], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        features = getattr(self, "prefill_features", None)
        # Padding rows past valid_len take no MoE expert work. Always passed (also when the
        # bucket is full) so warmup compiles the masked MoE programs serving uses.
        moe_rows = self._moe_valid_rows([valid_len], bucket)
        for index, layer in enumerate(self.layers):
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="prefill",
                    page_table=full_pt,
                    chunk_page_table=chunk_pt,
                    chunk_start_idx_tensor=csi_tensor,
                    moe_valid_rows=moe_rows,
                )
            else:
                x_new = layer.forward(
                    x,
                    mode="prefill",
                    chunk_size=layer.attention.long_prefill_chunk_size,
                    valid_len=valid_len,
                    moe_valid_rows=moe_rows,
                )
            ttnn.deallocate(x)
            x = x_new
            if features is not None:
                features.capture(index, x)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(full_pt)
        ttnn.deallocate(chunk_pt)
        ttnn.deallocate(csi_tensor)
        if moe_rows is not None:
            ttnn.deallocate(moe_rows)
        return x

    def _moe_valid_rows(self, lengths, segment_rows):
        """MoE padding-row page for an eager masked prefill (a host write: never inside a
        trace), or None for dense-MLP models."""
        if not getattr(self.args, "is_moe", False):
            return None
        from models.demos.blackhole.qwen36.tt.moe import prefill_valid_rows

        return prefill_valid_rows(self.device, lengths, segment_rows)

    def _forward_prefill_chunk_masked_tp(
        self, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=True, vision_tokens=None
    ):
        """TP (num_devices>1) masked fixed-bucket single-chunk prefill forward.

        flex_sdpa=True: flexible chunked SDPA (serving). flex_sdpa=False: host-int path (debug).
        Fills K/V for real blocks only. Returns hidden [1,1,bucket,dim]."""
        block_size = get_block_size(self._paged_kv_caches)
        tok = ttnn.from_torch(
            token_buf.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)
        x = ttnn.reshape(x, (1, 1, bucket, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        # Trace-safe vision splice (fixed-shape where over the hidden-sharded persistent buffers,
        # sliced to bucket; identity when the mask is zero). The caller stages the buffers
        # (prefill_masked_bucket -> _set_vision_merge). No-op until a trace is captured.
        x = self._apply_vision_merge(x, length=bucket)
        # rope_tp cos/sin for absolute positions [chunk_start, chunk_start+bucket).
        cos_t, sin_t = self._rope_tp_cos_sin_torch(chunk_start, bucket)
        cos = ttnn.from_torch(
            cos_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        sin = ttnn.from_torch(
            sin_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        blk0 = chunk_start // block_size
        blkN = num_blocks_in_seq(chunk_start + valid_len, block_size)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        csi_tensor = (
            ttnn.from_torch(
                torch.tensor([chunk_start], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            if flex_sdpa
            else None
        )
        for layer in self.layers:
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="prefill",
                    page_table=full_pt,
                    chunk_page_table=chunk_pt,
                    chunk_start_idx=chunk_start,
                    chunk_start_idx_tensor=csi_tensor,
                    valid_len=valid_len,  # unused by full attention
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=self.args.gdn_chunk_size, valid_len=valid_len)
            ttnn.deallocate(x)
            x = x_new
        # Deallocate per-chunk inputs; only hidden survives (avoids OOM in eager 64k loop).
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(full_pt)
        ttnn.deallocate(chunk_pt)
        if csi_tensor is not None:
            ttnn.deallocate(csi_tensor)
        return x

    def prefill_masked_bucket(
        self,
        token_ids,
        page_table,
        actual_len,
        chunk_start=0,
        bucket=None,
        flex_sdpa=True,
        vision_tokens=None,
        vis_row_offset=0,
    ):
        """Masked fixed-bucket prefill for a segment of `actual_len` real tokens.

        Pads the segment up to a fixed bucket length, runs all layers ONCE, and masks the GDN
        recurrent + conv state so they reflect exactly `actual_len` real tokens — numerically
        equivalent to the eager exact-length path (prefill_paged) but using one of only a few
        bucket-sized programs instead of compiling a fresh program per prompt length. That
        bounded program set is what makes warmup able to compile every code path before a trace
        is parked, so a short request can never trigger the compile-clobbers-trace hang.

        `chunk_start` is the segment's absolute start position (0 for a from-scratch short
        prompt; num_full*chunk_size for the tail of a long prompt — the carried GDN/KV state
        must already be in place). `vis_row_offset` is the number of image-placeholder tokens
        before this segment (so a tail that holds the bottom of a large image splices the right
        slice of the packed vision rows). Returns ttnn.Tensor (host) [1, 1, vocab_size]: the
        logit after position actual_len-1.
        """
        B_batch, _ = token_ids.shape
        assert B_batch == 1, "masked-bucket prefill is single-sequence"
        if bucket is None:
            bucket = self._mask_bucket_for(actual_len)
        assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"

        if chunk_start == 0:
            # chunk_start==0: new sequence, re-zero GDN. chunk_start>0: tail, keep carried state.
            self._reset_gdn_state_for_new_sequence()
            # Stage the per-request RoPE for this segment (M-RoPE for multimodal, 1D for text).
            # Only at the sequence start; a carried tail (chunk_start>0) keeps the table the
            # long-prompt entry (prefill_traced_chunked) already staged.
            self._build_request_rope(token_ids[:, :actual_len], vision_tokens)
            features = getattr(self, "prefill_features", None)
            if features is not None:
                features.begin_sequence()


        real = token_ids[:, :actual_len].to(torch.int32)
        if bucket > actual_len:
            pad = torch.zeros(1, bucket - actual_len, dtype=torch.int32)
            token_buf = torch.cat([real, pad], dim=1)
        else:
            token_buf = real

        # Stage the trace-safe vision buffers (host->device copy only). A segment splices its own
        # slice of the packed vision rows (vis_row_offset); a segment with no image placeholders
        # (text-only prompt, or a tail past the image) clears the mask inside _set_vision_merge so
        # the where is the identity. No-op without buffers.
        self._set_vision_merge(token_buf, vision_tokens, vis_row_offset)

        hidden = self._forward_prefill_chunk_masked(
            token_buf, actual_len, chunk_start, page_table, bucket, flex_sdpa=flex_sdpa, vision_tokens=vision_tokens
        )
        ttnn.synchronize_device(self.device)
        features = getattr(self, "prefill_features", None)
        if features is not None:
            features.commit(chunk_start, actual_len, bucket)

        if self.num_devices > 1:
            return self._masked_bucket_logits_tp(hidden, actual_len, bucket)
        logits = self._masked_bucket_logits(hidden, actual_len, bucket)
        ttnn.deallocate(hidden)
        return logits

    def _masked_bucket_logits(self, hidden, actual_len, bucket):
        """Fixed-shape row selection shared by masked tails and exact full chunks."""

        # One-hot matmul for last row (fixed program per bucket; slice would recompile per length).
        sel = torch.zeros(1, 1, bucket, dtype=torch.float32)
        sel[0, 0, actual_len - 1] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=hidden.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        x_last = ttnn.matmul(sel_tt, hidden)
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        normalized = self.norm(x_last, mode=Mode.PREFILL)
        ttnn.deallocate(x_last)
        self._store_last_hidden(normalized)
        logits = self._lm_head(normalized)
        ttnn.deallocate(normalized)
        host_logits = logits.cpu()
        ttnn.deallocate(logits)
        return host_logits

    def _masked_bucket_logits_tp(self, hidden, actual_len, bucket):
        """TP: one-hot select row actual_len-1, norm, lm_head. Returns replicated [1,1,vocab]."""
        sel = torch.zeros(1, 1, 1, bucket, dtype=torch.float32)
        sel[0, 0, 0, actual_len - 1] = 1.0
        sel_tt = ttnn.from_torch(
            sel,
            dtype=hidden.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x_last = ttnn.matmul(sel_tt, hidden)  # [1, 1, 1, dim]
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        self._store_last_hidden(x_last)
        logits = self._lm_head(x_last)
        return ttnn.reshape(logits, (1, 1, logits.shape[-1]))

    def warmup_prefill_masked_buckets(self, page_table, buckets=None):
        """Compile every masked-bucket prefill program up front, so a request never compiles after
        a trace is parked (a post-park compile clobbers the trace -> second-request hang, #48536).

        Two program kinds:
          * bucket-keyed (SDPA / GDN mask / norm / MLP-or-MoE): one per (bucket, is_full). Warmed
            by a dummy forward at each bucket, once masked (actual_len < bucket) and once full (==).
          * fill-width-keyed (paged_fill_cache): hashes on the fill shape, so it recompiles per
            fill width. Warmed directly by _warmup_paged_fill_widths (no full forward).

        MUST run in GDN serving state, before any trace is parked (capture_prefill_trace_chunked
        calls this just before begin_trace_capture). page_table must cover the largest bucket."""
        if buckets is None:
            buckets = self._PREFILL_MASK_BUCKETS
        block_size = get_block_size(self._paged_kv_caches)

        # Bucket-keyed programs: one masked + one no-mask forward per bucket.
        seen = set()
        for bucket in sorted(buckets):
            for actual_len in (max(1, bucket // 2), bucket):
                actual_len = max(1, min(actual_len, bucket))
                key = (bucket, actual_len == bucket)
                if key in seen:
                    continue
                seen.add(key)
                toks = torch.zeros(1, actual_len, dtype=torch.int32)
                self.prefill_masked_bucket(toks, page_table, actual_len=actual_len, bucket=bucket)
        # Fill-width-keyed programs: warm every width directly (no full forward).
        self._warmup_paged_fill_widths(page_table, buckets, block_size)
        ttnn.synchronize_device(self.device)

    def _warmup_paged_fill_widths(self, page_table, buckets, block_size):
        """Warm the per-fill-width programs in TPAttention.forward_prefill_paged's KV-fill sub-path
        (ttnn.slice + paged_fill_cache) without a full-model forward. Both hash on the fill shape
        (seq = fill_blocks * block_size), so each width is a fresh program; an un-warmed width would
        compile after the trace is parked and clobber it (hang).

        The ops are shape-keyed, so warming one layer's cache serves every layer -- far cheaper than
        the old per-width all-layer forward. Cover EVERY fill-width-dependent op here; a new one is
        caught by test_prefill_warmup_no_recompile (width sweep under misses-disallowed)."""
        if not self._paged_kv_caches:
            return
        if self.num_devices == 1:
            self._warmup_paged_fill_widths_single(page_table, buckets, block_size)
            return
        k_cache, v_cache = self._paged_kv_caches[0]
        nkv, hd = k_cache.shape[1], k_cache.shape[3]
        mapper = ttnn.ReplicateTensorToMesh(self.device)
        seen = set()
        for bucket in sorted(buckets):
            k_full = ttnn.from_torch(
                torch.zeros(1, nkv, bucket, hd, dtype=torch.bfloat16),
                dtype=k_cache.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                # L1 to match the request-time fill's L1 K/V (slice's cache key is buffer-type-specific).
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            for w in range(1, num_blocks_in_seq(bucket, block_size) + 1):
                page_len = min(w * block_size, bucket)
                key = (bucket, page_len)
                if key in seen:
                    continue
                seen.add(key)
                pt = ttnn.from_torch(
                    page_table[:, :w].contiguous(),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    mesh_mapper=mapper,
                )
                if page_len < bucket:
                    fill = ttnn.slice(k_full, (0, 0, 0, 0), (1, nkv, page_len, hd))
                else:
                    fill = k_full
                ttnn.experimental.paged_fill_cache(k_cache, fill, pt, batch_idx=0)
                ttnn.experimental.paged_fill_cache(v_cache, fill, pt, batch_idx=0)
                if page_len < bucket:
                    ttnn.deallocate(fill)
                ttnn.deallocate(pt)
            ttnn.deallocate(k_full)

    def _warmup_paged_fill_widths_single(self, page_table, buckets, block_size):
        """Single device: the fill slices BF16 K/V activations and casts them to the cache dtype
        per width, with memory configs set by the attention forward itself. Warm every width by
        running one full-attention layer's prefill forward (the ops are shape-keyed, so one layer
        serves all), rather than restating those ops here."""
        layer = next(layer for layer in self.layers if layer.is_full_attention)
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        csi = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        for bucket in sorted(buckets):
            hidden = ttnn.zeros(
                [1, bucket, self.args.dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x = layer.attention_norm(hidden, mode=Mode.PREFILL, norm_config=None)
            ttnn.deallocate(hidden)
            cos, sin = self.rope.get_prefill_rot_mats(0, bucket)
            for width in range(1, num_blocks_in_seq(bucket, block_size) + 1):
                chunk_pt = ttnn.from_torch(
                    page_table[:, :width].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                )
                out = layer.attention.forward(
                    x, cos, sin, page_table=full_pt, chunk_page_table=chunk_pt, chunk_start_idx_tensor=csi
                )
                ttnn.deallocate(out)
                ttnn.deallocate(chunk_pt)
            for tensor in (x, cos, sin):
                ttnn.deallocate(tensor)
        ttnn.deallocate(full_pt)
        ttnn.deallocate(csi)

    def prefix_snapshot(self, chunk_size):
        """Host copy of everything a prefill carries past a completed full-chunk boundary.

        Taken right after the trace replay of the chunk ending at the boundary (before any tail):
        the GDN recurrent + convolution state, and (DFlash) the draft feature taps of that last
        chunk, the only skipped chunk whose rows a resumed prefill's drafter context can need.
        Paged KV below the boundary stays in its blocks (prefix_resume.py owns their validity)."""
        gdn = self._save_gdn_snapshot()
        features = getattr(self, "prefill_features", None)
        taps = features.save(chunk_size) if features is not None else None
        tensors = [*gdn["recurrent"], *(gdn["conv"] or ()), *(taps or ())]
        if gdn["conv_pool"] is not None:
            tensors.append(gdn["conv_pool"])
        return {"gdn": gdn, "features": taps}, sum(t.numel() * t.element_size() for t in tensors)

    def prefill_traced_chunked(
        self, token_ids, page_table, actual_len, vision_tokens=None, start_position=0, resume=None, snapshot=None
    ):
        """Prefill by replaying the captured per-chunk trace for each FULL 2048-token chunk,
        then processing the final partial chunk eagerly with minimal padding.

        Only the real prompt (token_ids[:, :actual_len]) is processed; any bucket padding in
        token_ids is ignored. Full chunks (num_full = actual_len // chunk_size) are replayed
        from the trace; the remaining tail (< chunk_size tokens) is run eagerly so the GDN
        kernel zero-pads it to the next multiple of 128 (matching the non-traced path) instead
        of repeating the bucket padding through the recurrence — which corrupts the decode
        state at long context. actual_len is the real prompt length; the next-token logit is
        extracted at actual_len-1. Returns ttnn.Tensor (host) [1, 1, vocab_size].

        vision_tokens (multimodal) are spliced trace-safely: the captured forward runs a
        FIXED-shape ttnn.where(mask, vision, text) over persistent buffers (_vis_buf /
        _vis_mask_buf) that this method stages per chunk via copy_host_to_device — no program
        compiles at request time, so the parked trace is never clobbered. Each chunk (and the tail)
        splices its own slice of the packed vision rows (vis_row_offset = image tokens before the
        chunk), so a large image whose placeholders span multiple chunks is handled; a segment with
        no image tokens clears the mask (the where becomes the identity). Works on both single
        device (3D buffers) and TP (4D hidden-sharded buffers; the vision rows are gathered to full
        hidden on host in _set_vision_merge, placed along seq, then re-sharded — see
        _alloc_vision_merge_buffers).
        """
        # Default to the standard 2048-token chunk when no trace is captured (e.g. the TP MVP,
        # which serves <=2048 prompts entirely via the masked bucket below and so needs no chunk
        # trace). The chunk trace is only required once there is at least one full chunk to replay.
        chunk_size = self._chunked_chunk_size or 2048
        B, T = token_ids.shape
        assert 1 <= actual_len <= T, f"actual_len {actual_len} not in [1, {T}]"
        assert 0 <= start_position < actual_len
        assert start_position % 128 == 0, "Resumed prefill starts must align to 128 tokens"
        if start_position and (self.num_devices != 1 or vision_tokens is not None):
            raise ValueError("Resumed prefill supports single-device text serving")
        if resume is not None and (int(resume["length"]) != start_position or start_position % 2048):
            # Exact prefix resume (prefix_resume.py) restarts the SAME 2048-aligned chunk schedule a
            # full prefill runs, at a boundary whose state a full prefill of the prefix produced.
            raise ValueError(f"cannot resume a {actual_len}-token prefill at {resume['length']}")
        if snapshot is not None and (vision_tokens is not None or self.num_devices > 1):
            snapshot = None
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size
        remaining = actual_len - start_position
        num_full = remaining // chunk_size
        tail_real = remaining - num_full * chunk_size
        assert (
            num_full == 0 or self.num_devices > 1 or self._chunked_trace_id is not None
        ), "Call capture_prefill_trace_chunked first"

        # Stage the per-request RoPE once for the whole prompt (M-RoPE for multimodal, 1D for text).
        # The chunk-replay loops + the masked tail then slice this sequence-indexed table by chunk
        # position, and decode offsets by the stored rope_delta. (The num_full==0 short path below
        # re-stages it inside prefill_masked_bucket; that is idempotent.)
        self._build_request_rope(token_ids[:, :actual_len], vision_tokens)
        features = getattr(self, "prefill_features", None)
        if resume is not None:
            self._reset_gdn_state_for_new_sequence()
            self._restore_gdn_snapshot(resume["gdn"])
            if features is not None:
                # A full prefill commits only rows inside the final draft window; of the skipped
                # chunks only the one ending at the boundary can reach it. Replay its commit.
                features.begin_sequence()
                features.restore(chunk_size, resume["features"])
                features.commit(start_position - chunk_size, chunk_size, chunk_size, sequence_length=actual_len)

        # Short prompt (no full chunks): route the whole prompt through the SAME masked
        # fixed-bucket path the long-prompt tail uses. chunk_start=0 makes prefill_masked_bucket
        # do the sequence-start GDN reset and run one masked forward — there is no trace to replay,
        # so the chunk-input plumbing below is skipped. This is the single bucketed+masked path
        # shared by short prompts and the long-prompt tail; prefill_dispatch routes every traced
        # prefill here so the short/long seam is defined once.
        if num_full == 0:
            # Pad/clip the SDPA page table to the warmed/captured width so the short-prompt forward
            # REPLAYS the pre-warmed programs instead of recompiling at request time (which clobbers
            # parked decode/chunk traces -> second-request hang). No-op when no chunk buffer was
            # captured or the widths already match.
            page_table = self._fit_page_table_width(page_table)
            return self.prefill_masked_bucket(
                token_ids[:, start_position:actual_len], page_table, actual_len=remaining,
                chunk_start=start_position, vision_tokens=vision_tokens
            )

        if self.num_devices > 1:
            # TP long prompt: traced replay preferred; eager masked-bucket fallback if no trace.
            if self._chunked_trace_id is not None:
                return self._prefill_traced_chunked_tp(
                    token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=vision_tokens
                )
            # Eager fallback: flexible qk=64 SDPA matches traced path.
            return self._prefill_chunked_eager_tp(
                token_ids,
                page_table,
                actual_len,
                num_full,
                chunk_size,
                tail_real,
                flex_sdpa=True,
                vision_tokens=vision_tokens,
            )

        # Re-zero GDN once; carries across replays + masked tail (chunk_start>0 skips reset).
        if start_position == 0:
            self._reset_gdn_state_for_new_sequence()
        # Pad/clip page_table to captured buffer width (vLLM may differ). Trailing blocks unused.
        page_table = self._fit_page_table_width(page_table)
        pt_host = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(pt_host, self._chunk_full_page_table_buf)

        # The drafter attends only to its final context window. The target still
        # captures every chunk, but projecting older features into the draft KV
        # ring is pure overwrite work.
        if features is not None and resume is None:
            features.begin_sequence()
        for c in range(num_full):
            cs = start_position + c * chunk_size
            tok_host = ttnn.from_torch(
                token_ids[:, cs : cs + chunk_size].to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(tok_host, self._chunk_token_buf)

            csi_host = ttnn.from_torch(
                torch.tensor([cs], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(csi_host, self._chunk_start_idx_tensor)

            blk0 = cs // block_size
            cpt_host = ttnn.from_torch(
                page_table[:, blk0 : blk0 + blocks_per_chunk].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(cpt_host, self._chunk_page_table_buf)

            # M-RoPE-aware per-chunk cos/sin (slices the staged per-request table for multimodal;
            # 1D RoPE for text). Updated into the persistent buffer per chunk via host->device copy,
            # so it stays trace-safe.
            cos_seq, sin_seq = self.rope.prefill_cos_sin_torch(cs, chunk_size)
            cos_host = ttnn.from_torch(
                cos_seq.unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            sin_host = ttnn.from_torch(
                sin_seq.unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(cos_host, self._chunk_cos_buf)
            ttnn.copy_host_to_device_tensor(sin_host, self._chunk_sin_buf)

            # Stage the trace-safe vision buffers: each chunk splices its own slice of the packed
            # vision rows (vis_row_offset = image tokens before cs); a chunk with no image tokens
            # clears the mask so the captured where is the identity. host->device copy only (no
            # compile), so the parked trace is untouched. Handles a large image whose placeholders
            # span multiple chunks.
            self._set_vision_merge(
                token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
            )

            ttnn.execute_trace(self.device, self._chunked_trace_id, cq_id=0, blocking=False)
            if features is not None:
                features.commit(cs, chunk_size, chunk_size, sequence_length=actual_len)

        ttnn.synchronize_device(self.device)
        if snapshot is not None and num_full > 0 and start_position % chunk_size == 0:
            snapshot(start_position + num_full * chunk_size, *self.prefix_snapshot(chunk_size))

        # Tail via masked bucket (or last full chunk hidden if exact multiple of chunk_size).
        if tail_real > 0:
            cs = start_position + num_full * chunk_size
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
        # Use the same warmed row-selector program as a full masked bucket.
        # A request-only slice/untilize here would compile after traces were parked.
        return self._masked_bucket_logits(self._chunked_trace_output, chunk_size, chunk_size)

    def _prefill_chunked_eager_tp(
        self, token_ids, page_table, actual_len, num_full, chunk_size, tail_real, flex_sdpa=True, vision_tokens=None
    ):
        """TP eager long-prompt prefill via warmed bucket=chunk_size programs.
        Returns logits [1,1,vocab] at actual_len-1."""
        # Re-zero GDN at sequence start; tail (chunk_start>0) keeps carried state.
        self._reset_gdn_state_for_new_sequence()
        last_hidden = None
        for c in range(num_full):
            cs = c * chunk_size
            if last_hidden is not None:
                ttnn.deallocate(last_hidden)
            # Stage the vision buffers: each chunk splices its own slice of the packed vision rows
            # (vis_row_offset = image tokens before cs); a chunk with no image tokens clears the
            # mask so the where is the identity (host->device copy only, no compile).
            self._set_vision_merge(
                token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
            )
            # Full chunk: valid_len == bucket == chunk_size (no padding/masking).
            last_hidden = self._forward_prefill_chunk_masked_tp(
                token_ids[:, cs : cs + chunk_size], chunk_size, cs, page_table, chunk_size, flex_sdpa=flex_sdpa
            )
            ttnn.synchronize_device(self.device)
        if tail_real > 0:
            ttnn.deallocate(last_hidden)
            cs = num_full * chunk_size
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                flex_sdpa=flex_sdpa,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
        # Exact multiple of chunk_size: logit from last full chunk.
        logits = self._masked_bucket_logits_tp(last_hidden, chunk_size, chunk_size)
        ttnn.deallocate(last_hidden)
        return logits

    def _prefill_traced_chunked_tp(
        self, token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=None
    ):
        """TP traced chunk-outer prefill: replay the captured per-chunk trace
        (_forward_prefill_chunk_tp) for each FULL chunk, then run the partial tail through the
        masked bucket. The TP analog of the single-device loop in prefill_traced_chunked: each
        chunk's inputs are DMA'd into the REPLICATED persistent buffers via
        copy_host_to_device_tensor (no per-chunk program dispatch / device allocation — only one
        execute_trace per chunk), so GDN recurrent/conv + paged-KV state carry in place across
        replays and host pressure stays bounded at 128K. The tail's chunk_start>0 skips the GDN
        reset so the carried state continues. Returns logits [1, 1, vocab] at actual_len-1."""
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size
        rep = ttnn.ReplicateTensorToMesh(self.device)

        # Re-zero GDN once; carries across replays + tail (chunk_start>0 skips reset).
        self._reset_gdn_state_for_new_sequence()

        # Pad/clip page_table to captured width; write once (constant across chunks).
        buf_blocks = int(self._chunk_full_page_table_buf.shape[-1])
        if page_table.shape[1] < buf_blocks:
            page_table = torch.cat(
                [
                    page_table,
                    torch.zeros(page_table.shape[0], buf_blocks - page_table.shape[1], dtype=page_table.dtype),
                ],
                dim=1,
            )
        elif page_table.shape[1] > buf_blocks:
            page_table = page_table[:, :buf_blocks]
        pt_host = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
        )
        ttnn.copy_host_to_device_tensor(pt_host, self._chunk_full_page_table_buf)

        # Replay trace per full chunk. Host input-prep (from_torch tilize of cos/sin + DMA copies) and
        # device trace exec share cq_id=0, so the queue already ORDERS each chunk's copies AFTER the
        # prior chunk's trace reads them (no double-buffering needed). The old code did a full
        # synchronize_device every chunk, which forced the host to wait and serialized chunk N+1's CPU
        # prep behind chunk N's device exec. Instead we hold host-tensor refs alive (so their in-flight
        # DMAs aren't GC'd) and sync only every _SYNC_EVERY chunks — the host software-pipelines chunk
        # N+1's from_torch/tilize over chunk N's device exec. Periodic (not fully removed) sync bounds
        # in-flight queue depth so very long context (e.g. traced_128k = 64 chunks) can't overrun the
        # command queue. QWEN36_PREFILL_OVERLAP=0 restores the per-chunk sync.
        _log_every = max(1, num_full // 4)
        _overlap = os.environ.get("QWEN36_PREFILL_OVERLAP", "1") != "0"
        _SYNC_EVERY = 8 if _overlap else 1
        _host_refs = []  # keep host tensors alive until the next sync frees their DMAs
        for c in range(num_full):
            cs = c * chunk_size
            tok_host = ttnn.from_torch(
                token_ids[:, cs : cs + chunk_size].to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(tok_host, self._chunk_token_buf)

            csi_host = ttnn.from_torch(
                torch.tensor([cs], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(csi_host, self._chunk_start_idx_tensor)

            blk0 = cs // block_size
            cpt_host = ttnn.from_torch(
                page_table[:, blk0 : blk0 + blocks_per_chunk].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(cpt_host, self._chunk_page_table_buf)

            cos_t, sin_t = self._rope_tp_cos_sin_torch(cs, chunk_size)
            cos_host = ttnn.from_torch(
                cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep
            )
            sin_host = ttnn.from_torch(
                sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep
            )
            ttnn.copy_host_to_device_tensor(cos_host, self._chunk_cos_buf)
            ttnn.copy_host_to_device_tensor(sin_host, self._chunk_sin_buf)
            _host_refs += [tok_host, csi_host, cpt_host, cos_host, sin_host]

            # Stage the hidden-sharded vision buffers: each chunk splices its own slice of the
            # packed vision rows (vis_row_offset = image tokens before cs); a chunk with no image
            # tokens clears the mask so the captured where is the identity. host->device copy only
            # (no compile), so the parked trace is untouched.
            self._set_vision_merge(
                token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
            )

            ttnn.execute_trace(self.device, self._chunked_trace_id, cq_id=0, blocking=False)

            # Bound in-flight depth; after a sync the completed DMAs' host tensors can be released.
            if (c + 1) % _SYNC_EVERY == 0:
                ttnn.synchronize_device(self.device)
                _host_refs.clear()
            if (c + 1) % _log_every == 0:
                logger.info(f"[TP chunk-replay] {c + 1}/{num_full} chunks")

        # Drain any still-in-flight chunk DMAs before returning, so the loop's host input tensors
        # (_host_refs) are not GC'd while a non-blocking execute_trace is still reading them — a
        # use-after-free that hangs the device. When num_full < _SYNC_EVERY the loop never synced,
        # so the no-tail return below (which issues no further blocking work) would otherwise race.
        if _host_refs:
            ttnn.synchronize_device(self.device)
            _host_refs.clear()

        # Tail via masked bucket, or _masked_bucket_logits_tp if no tail (TP 4D hidden).
        if tail_real > 0:
            cs = num_full * chunk_size
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
        return self._masked_bucket_logits_tp(self._chunked_trace_output, chunk_size, chunk_size)

    def reset_state(self, batch_size=None):
        """Reset layer state for a new sequence (eager/pre-trace path; trace uses _reset_dn_state_inplace)."""
        if self.num_devices == 1 and self._deltanet_external_states:
            if batch_size is not None and batch_size != self._reserved_gdn_batch_size:
                raise ValueError("Cannot resize persistent single-stream GDN state")
            self._init_dn_zero_buffers()
            self._reset_dn_state_inplace()
            for layer in self.layers:
                if layer.is_full_attention:
                    layer.attention.reset_cache()
            return
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.reset_cache()
            else:
                layer.attention.reset_state(batch_size)

    def _reset_gdn_state_for_new_sequence(self):
        """Zero GDN recurrent+conv at sequence start.

        Trace capture runs forward twice; GDN state is non-idempotent. Must re-zero before each
        real sequence. In-place buffers (_chunk_inplace_state) use _reset_dn_state_inplace."""
        self._mtp_seeded = None
        if self.num_devices > 1:
            # TP: reset_state_inplace preserves decode-trace baked addresses.
            for layer in self.layers:
                if not layer.is_full_attention:
                    layer.attention.reset_state_inplace()
            return
        inplace = any(
            (not l.is_full_attention) and getattr(l.attention, "_chunk_inplace_state", False) for l in self.layers
        )
        if inplace:
            self._reset_dn_state_inplace()
        else:
            self.reset_state(batch_size=1)

    def _reset_dn_state_inplace(self):
        """Zero DN state in place via pre-allocated zero buffers (trace addresses fixed)."""
        assert self._dn_zero_recurrent is not None, "Call _init_dn_zero_buffers first"
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            ttnn.copy(self._dn_zero_recurrent, dn.recurrent_state)
            if self._deltanet_conv_pool is None:
                ttnn.copy(self._dn_zero_conv, dn.fused_conv_state)
            # split_conv_state rebuilt lazily on first decode.
            if dn.split_conv_state is not None:
                for buf in dn.split_conv_state:
                    ttnn.deallocate(buf)
                dn.split_conv_state = None
        # The per-layer convolution tensors are narrow views into one sharded pool.
        # Reset the backing allocation once; copying into each view can lose its
        # shard offset and leave stale state in alternating requests.
        if self._deltanet_conv_pool is not None:
            ttnn.copy(self._dn_zero_conv, self._deltanet_conv_pool)

    def _init_dn_zero_buffers(self):
        """Allocate shared zero buffers for DN recurrent and conv shapes."""
        first_dn = next(layer.attention for layer in self.layers if not layer.is_full_attention)
        key = (tuple(first_dn.recurrent_state.shape), first_dn.recurrent_state.layout,
               first_dn.recurrent_state.dtype, tuple(first_dn.fused_conv_state.shape),
               first_dn.fused_conv_state.layout)
        if key in self._dn_zero_buffers:
            self._dn_zero_recurrent, self._dn_zero_conv = self._dn_zero_buffers[key]
            return
        # Recurrent tensors are per-layer; convolution state is one pooled allocation.
        rec_shape = list(first_dn.recurrent_state.shape)
        conv_shape = list(
            (self._deltanet_conv_pool if self._deltanet_conv_pool is not None else first_dn.fused_conv_state).shape
        )
        self._dn_zero_recurrent = ttnn.zeros(
            rec_shape,
            dtype=first_dn.recurrent_state.dtype,
            layout=first_dn.recurrent_state.layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._dn_zero_conv = ttnn.zeros(
            conv_shape,
            dtype=ttnn.bfloat16,
            layout=first_dn.fused_conv_state.layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._dn_zero_buffers[key] = self._dn_zero_recurrent, self._dn_zero_conv

    def _allocate_deltanet_state_buffers(self, batch_size):
        """Reserve stable state; only native kernels require the compact L1 pool."""
        if self.num_devices > 1:
            return []
        cfg = GDNConfig.from_args(self.args)
        gdn_layer_count = sum(
            self.args.attention_type_list[layer_index] != "full_attention"
            for layer_index in self.layer_indices
        )
        if not self._native_gdn_state or self._batched_single_device:
            state_dtype = (
                ttnn.float32 if not self._batched_single_device and self.args.mamba_ssm_dtype == "float32"
                else ttnn.bfloat16
            )
            return [
                (
                    ttnn.zeros(
                        [batch_size, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim],
                        dtype=state_dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT if self._batched_single_device else ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    ttnn.zeros(
                        [batch_size, cfg.conv_kernel_size - 1, cfg.q_dim + cfg.k_dim + cfg.v_dim],
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                )
                for _ in range(gdn_layer_count)
            ]
        state_grid = self.device.compute_with_storage_grid_size()
        grid_y = int(state_grid.y)
        core_count = int(state_grid.x) * grid_y
        all_cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(int(state_grid.x) - 1, grid_y - 1),
                )
            }
        )
        recurrent_rows = batch_size * cfg.num_v_heads * cfg.head_k_dim
        recurrent_shard_rows = (recurrent_rows + core_count - 1) // core_count
        recurrent_mem_config = ttnn.create_sharded_memory_config(
            shape=(recurrent_shard_rows, cfg.head_v_dim),
            core_grid=all_cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        recurrent_states = []
        for _ in range(gdn_layer_count):
            recurrent_states.append(
                ttnn.from_torch(
                    torch.zeros(
                        batch_size,
                        cfg.num_v_heads,
                        cfg.head_k_dim,
                        cfg.head_v_dim,
                        dtype=torch.bfloat16,
                    ),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=recurrent_mem_config,
                )
            )

        conv_width = cfg.q_dim + cfg.k_dim + cfg.v_dim
        conv_rows_per_layer = batch_size * (cfg.conv_kernel_size - 1)
        conv_rows = gdn_layer_count * conv_rows_per_layer
        # The native frontend reads a layer's whole history from one core; it is
        # enabled whenever the fused frontend is (27B mega or decode-only mega).
        frontend_fused = os.environ.get("QWEN_GDN_FRONTEND_FUSED") == "1"
        conv_shard_rows = conv_rows_per_layer if (cfg.num_v_heads == 48 or frontend_fused) else 1
        conv_core_count = conv_rows // conv_shard_rows
        assert conv_core_count <= core_count, (
            f"compact GDN convolution pool needs {conv_core_count} cores, device has {core_count}"
        )
        conv_cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(core_index // grid_y, core_index % grid_y),
                    ttnn.CoreCoord(core_index // grid_y, core_index % grid_y),
                )
                for core_index in range(conv_core_count)
            }
        )
        conv_pool_mem_config = ttnn.create_sharded_memory_config(
            shape=(conv_shard_rows, conv_width),
            core_grid=conv_cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._deltanet_conv_pool = ttnn.from_torch(
            torch.zeros(1, conv_rows, conv_width, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=conv_pool_mem_config,
        )
        states = []
        for state_index, recurrent in enumerate(recurrent_states):
            convolution = ttnn.narrow(
                self._deltanet_conv_pool,
                1,
                state_index * conv_rows_per_layer,
                conv_rows_per_layer,
            )
            states.append((recurrent, convolution))
        l1_view = ttnn.get_memory_view(self.device, ttnn.BufferType.L1)
        logger.info(
            "Reserved early GDN L1 state: layers={} batch={} grid={}x{} "
            "allocated_per_bank={} largest_free_per_bank={} first_rec_addr={} last_conv_addr={}",
            len(states),
            batch_size,
            int(state_grid.x),
            grid_y,
            l1_view.total_bytes_allocated_per_bank,
            l1_view.largest_contiguous_bytes_free_per_bank,
            states[0][0].buffer_address() if states else None,
            states[-1][1].buffer_address() if states else None,
        )
        return states

    def set_paged_kv_caches(self, kv_caches):
        """Attach paged KV caches to the 8 attention layers."""
        self._paged_kv_caches = kv_caches
        for cache_idx, layer_idx in enumerate(self._attention_layer_indices):
            k_cache, v_cache = kv_caches[cache_idx]
            self.layers[layer_idx].attention.set_paged_kv_cache(k_cache, v_cache)

    def allocate_kv_caches(self, kv_cache_shape, dtype, batch_size=1):
        """Allocate caches for all 32 layers. Returns only the attention KV caches (for vLLM)."""
        assert self._paged_kv_caches is None, "allocate_kv_caches already called; deallocate first"
        assert batch_size == self._reserved_gdn_batch_size, (
            f"GDN state reserved for batch={self._reserved_gdn_batch_size}, "
            f"got batch={batch_size}"
        )
        # QWEN_SDPA_BF8: bf8 paged KV for SDPA; halves KV memory (gated — validate PCC at long ctx).
        if os.environ.get("QWEN_SDPA_BF8", "0") == "1":
            dtype = ttnn.bfloat8_b
        if self.num_devices > 1:
            return self._allocate_kv_caches_tp(kv_cache_shape, dtype, batch_size)

        kv_caches = []
        for idx in self._attention_layer_indices:
            k_cache = ttnn.zeros(kv_cache_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            v_cache = ttnn.zeros(kv_cache_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            kv_caches.append([k_cache, v_cache])
        self.set_paged_kv_caches(kv_caches)
        if self._batched_single_device:
            self._init_dn_zero_buffers()
            self._ensure_gdn_prefill_scratch()
            logger.info("Batched GDN persistent state: layers={} slots={} BF16 row-major DRAM",
                        len(self._deltanet_external_states), batch_size)
            return kv_caches
        if not self._native_gdn_state:
            logger.info("Generic GDN persistent state: layers={} recurrent_dtype={} placement=DRAM",
                        len(self._deltanet_external_states), self.args.mamba_ssm_dtype)
            return kv_caches

        cfg = GDNConfig.from_args(self.args)
        state_grid = self.device.compute_with_storage_grid_size()
        state_core_count = int(state_grid.x) * int(state_grid.y)
        recurrent_rows = batch_size * cfg.num_v_heads * cfg.head_k_dim
        recurrent_shard_rows = (recurrent_rows + state_core_count - 1) // state_core_count
        conv_width = cfg.q_dim + cfg.k_dim + cfg.v_dim
        recurrent_bytes = sum(int(rec.volume()) * 2 for rec, _ in self._deltanet_external_states)
        convolution_bytes = sum(int(conv.volume()) * 2 for _, conv in self._deltanet_external_states)
        compute_cores = state_core_count
        persistent_per_core = (recurrent_bytes + convolution_bytes + compute_cores - 1) // compute_cores
        recurrent_max_per_core = (
            len(self._deltanet_external_states) * recurrent_shard_rows * cfg.head_v_dim * 2
        )
        convolution_max_per_core = int(self._deltanet_conv_pool.memory_config().shard_spec.shape[0]) * conv_width * 2
        persistent_max_per_core = recurrent_max_per_core + convolution_max_per_core
        physical_l1_bytes = 1_572_864
        program_scratch_per_core = 0
        if os.environ.get("QWEN36_OPTIMIZED_27B") == "1":
            # Pinned Blackhole HAL: MEM_MAP_END 40832 + kernel-config region 70656.
            static_cb_origin = 111_488
            # Packed BF8 LM, K-block 1: in0 + in1 + BF16 output + FP32 partials;
            # deliberately take no credit for output/partial lifetime aliasing.
            lm_cb_bytes = 4_096 + 154_496 + 145_408 + 290_816
            # gated_delta_attn_seq streams two input matrices and aliases state/final:
            # 145 physical FP32 tiles at C=K=V=128, including all phase scratch.
            ct = cfg.long_prefill_chunk_size // 32
            kt, vt = cfg.head_k_dim // 32, cfg.head_v_dim // 32
            seq_cb_tiles = ct * ct + 6 * ct * vt + kt * vt + 3 * max(kt, vt) + 5
            static_cb_high_water = static_cb_origin + max(lm_cb_bytes, seq_cb_tiles * 4096)
            # Shared native output buffers (one M tile for T=1 and T=2), plus the
            # model-owned row-major verify mailbox. Reserve even the optional T=2 buffer.
            output_tiles = math.ceil(cfg.v_dim / 32)
            program_scratch_per_core = 2 * math.ceil(output_tiles / state_core_count) * 2048
            if self.mtp is not None:
                program_scratch_per_core += cfg.head_v_dim * 2
        elif self.args.dim == 5120:
            # Sealed legacy measurement, prior to the bounded full-27B norm port.
            static_cb_high_water = 1_258_368
        else:
            # No sealed CB measurement for this geometry; Metal rejects any
            # program whose static CBs overlap the L1 state pool at launch.
            static_cb_high_water = None
        persistent_low_address = min(
            int(buf.buffer_address()) for state in self._deltanet_external_states for buf in state
        )
        safety_margin = None
        if static_cb_high_water is not None:
            safety_margin = min(
                physical_l1_bytes - persistent_max_per_core, persistent_low_address
            ) - static_cb_high_water - program_scratch_per_core
            assert safety_margin >= 60 * 1024, (
                f"GDN L1 map leaves only {safety_margin} bytes/core; require at least 61440 "
                f"(state={persistent_max_per_core}, program_scratch={program_scratch_per_core}, "
                f"static_cb_end={static_cb_high_water}, lowest_state_address={persistent_low_address})"
            )
        assert all(
            rec.dtype == ttnn.bfloat16
            and conv.dtype == ttnn.bfloat16
            and rec.layout == ttnn.ROW_MAJOR_LAYOUT
            and rec.memory_config().buffer_type == ttnn.BufferType.L1
            and conv.memory_config().buffer_type == ttnn.BufferType.L1
            for rec, conv in self._deltanet_external_states
        ), "GDN persistent state must be BF16 and L1-resident"
        logger.info(
            "GDN persistent state: layers={} dtype=BF16 placement=L1 recurrent_bytes={} "
            "convolution_bytes={} compute_cores={} average_bytes_per_core={} "
            "max_bytes_per_core={} static_cb_high_water={} safety_margin={}",
            len(self._deltanet_external_states),
            recurrent_bytes,
            convolution_bytes,
            compute_cores,
            persistent_per_core,
            persistent_max_per_core,
            static_cb_high_water,
            safety_margin,
        )

        return kv_caches

    def free_kv_caches(self):
        """Release paged KV caches while retaining fixed-address persistent GDN state."""
        ttnn.synchronize_device(self.device)
        if getattr(self, "_dflash_cycle", None) is not None:
            self._dflash_cycle.release()
            self._dflash_cycle = None
        if getattr(self, "_mtp_wide", None) is not None:
            self._mtp_wide.release()
        if self.mtp is not None:
            self.mtp.release_trace()
        if getattr(self, "_mtp_cycle_trace", None) is not None:
            ttnn.release_trace(self.device, self._mtp_cycle_trace)
            self._mtp_cycle_trace = None
        self._mtp_cycle_compiled = False
        if getattr(self, "_mtp_sampled_trace", None) is not None:
            ttnn.release_trace(self.device, self._mtp_sampled_trace)
            self._mtp_sampled_trace = None
        self._mtp_sampled_compiled = False
        if getattr(self, "_chunked_trace_id", None) is not None:
            ttnn.release_trace(self.device, self._chunked_trace_id)
            self._chunked_trace_id = None
        if getattr(self, "_chunked_trace_output", None) is not None:
            ttnn.deallocate(self._chunked_trace_output)
            self._chunked_trace_output = None
        self._chunked_prepared_key = None
        if getattr(self, "prefill_features", None) is not None:
            self.prefill_features.release()
            self.prefill_features = None
        if getattr(self, "dflash", None) is not None:
            self.dflash.release()
            self.dflash = None
        # GDN state is a model-lifetime L1 reservation; generation resets zero it in place.
        if getattr(self, "_paged_kv_caches", None) is not None:
            for k_cache, v_cache in self._paged_kv_caches:
                ttnn.deallocate(k_cache)
                ttnn.deallocate(v_cache)
            self._paged_kv_caches = None

    def _allocate_kv_caches_tp(self, kv_cache_shape, dtype, batch_size):
        """TP paged KV allocation (B=1). Replicated per device; GDN self-manages state."""

        def _mk():
            return ttnn.as_tensor(
                torch.zeros(kv_cache_shape, dtype=torch.bfloat16),
                device=self.device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )

        kv_caches = [[_mk(), _mk()] for _ in self._attention_layer_indices]
        self.set_paged_kv_caches(kv_caches)  # binds via TPAttention.set_paged_kv_cache
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.B = batch_size
                layer.attention.reset_state()
                # Fixed-address GDN state for decode trace compatibility.
                layer.attention._stable_state = True
        # Marker for re-entry assert; TP GDN state lives in module, not external buffers.
        self._deltanet_external_states = []
        return kv_caches

    def _prefill_paged_tp(self, token_ids, page_table, valid_len=None, vision_tokens=None, gdn_collect=False):
        """TP (num_devices>1) paged prefill, B=1. Mirrors the demo prefill_tp but routes
        the full-attention layers through the paged KV cache (forward_prefill_paged) so
        decode can read it via page_table. GDN layers capture their recurrent/conv state
        as in the demo. Returns logits [1, 1, vocab] at position valid_len-1.
        """
        B, T = token_ids.shape
        assert B == 1, "TP prefill is single-sequence (B=1); batched serving prefills one user at a time"
        vlen = valid_len or T
        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text).
        self._build_request_rope(token_ids[:, :vlen], vision_tokens)
        pt_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        page_table_tt = ttnn.from_torch(pt_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        tok = ttnn.from_torch(
            token_ids.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
        x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, T)
        rep = ttnn.ReplicateTensorToMesh(self.device)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        for layer in self.layers:
            x = layer.forward(
                x,
                cos=cos,
                sin=sin,
                mode="prefill",
                chunk_size=self.args.gdn_chunk_size,
                valid_len=vlen,
                page_table=page_table_tt,
                chunk_page_table=page_table_tt,
                chunk_start_idx=0,
                gdn_collect=gdn_collect,
            )
        x = self.norm(x, mode=Mode.PREFILL)
        x_last = x[:, :, vlen - 1 : vlen, :]
        self._store_last_hidden(x_last)
        logits = self._lm_head(x_last)
        ttnn.deallocate(x)
        return ttnn.reshape(logits, (1, 1, logits.shape[-1]))

    def prefill_paged_peruser(self, token_ids_list, page_table, valid_lens=None):
        """Batched per-user TP prefill (the batched serving contract).

        Prefills B users into ONE shared paged KV cache + the batched GDN decode state, one user at
        a time. Each user's full-attention layers fill their own blocks via the per-user page-table
        row; each GDN layer collects that user's from-scratch state, stitched into row u of the
        batched decode buffers by finalize_pending(). Call allocate_kv_caches(batch_size=B) first.

        token_ids_list: list of B torch.Tensor [1, T_u] (lengths may differ).
        page_table:      torch.Tensor [B, max_blocks_per_seq] int32 — row u = user u's blocks.
        valid_lens:      optional list of B ints (real token counts); defaults to each T_u.
        Returns:         list of B ttnn logits [1, 1, vocab_size] (one per user, at valid_len-1).
        """
        assert self.num_devices > 1, "prefill_paged_peruser is the TP (num_devices>1) path"
        B = len(token_ids_list)
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert page_table_torch.shape[0] == B, "page_table must have one row per user"

        # Fresh GDN per-user accumulators (cleared at the end by finalize_pending).
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention._pending = []

        logits = []
        for u in range(B):
            vlen = valid_lens[u] if valid_lens is not None else None
            # Each user's prefill is the validated B=1 paged path, pointed at user u's blocks.
            lg = self._prefill_paged_tp(
                token_ids_list[u], page_table_torch[u : u + 1], valid_len=vlen, gdn_collect=True
            )
            logits.append(lg)

        # Stitch every user's collected GDN state into the batched decode buffers (row u = user u).
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.finalize_pending()
        return logits

    def _alloc_gdn_scratch_b(self, bg):
        """Like _alloc_gdn_scratch_b1 but for a group of `bg` users: allocate a dedicated
        [bg,...] GDN state on every GDN layer, distinct from the real batched [B,...] decode
        buffer. Returns the prior batched bindings for _restore_gdn_batched. Used by the grouped
        batched prefill (forward_prefill_batched writes [bg,...] into these in place)."""
        prev = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            prev.append((dn, dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state))
            dn.B = bg
            dn.reset_state()  # builds rec_state [bg,Nv,Dk,Dv], conv_states[*] [1,bg,D], carry, zero0
            dn._stable_state = True  # forward_prefill_batched writes state in place under this flag
        return prev

    def _assemble_groups_gdn_dev(self, group_rec_dev, group_conv_dev):
        """Assemble per-GROUP GDN states (each already batched [bg,...] on device, from
        forward_prefill_batched) into the full [B,...] batched decode buffers via device-side
        concat — no host round-trip. rec: concat groups along dim 0 -> [B,Nv,Dk,Dv]; conv_states[m]:
        concat groups along dim 1 -> [1,B,D]. The batched GDN bindings MUST already be rebound
        (writes in place under _stable_state). Row u == user u because groups are contiguous
        (group g = users [g*group_size : ...])."""
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        ng = len(group_rec_dev)
        for li, dn in enumerate(dn_layers):
            rec_full = ttnn.concat([group_rec_dev[g][li] for g in range(ng)], dim=0)  # [B, Nv, Dk, Dv]
            rec_src = rec_full if rec_full.dtype == dn.rec_state.dtype else ttnn.typecast(rec_full, dn.rec_state.dtype)
            ttnn.copy(rec_src, dn.rec_state)
            if rec_src is not rec_full:
                ttnn.deallocate(rec_src)
            ttnn.deallocate(rec_full)
            for g in range(ng):
                ttnn.deallocate(group_rec_dev[g][li])
            for m in range(dn.K):
                conv_full = ttnn.concat([group_conv_dev[g][li][m] for g in range(ng)], dim=1)  # [1, B, D]
                ttnn.copy(conv_full, dn.conv_states[m])
                ttnn.deallocate(conv_full)
                for g in range(ng):
                    ttnn.deallocate(group_conv_dev[g][li][m])

    def prefill_paged_grouped(self, token_ids_list, page_table, valid_lens=None, group_size=4):
        """Grouped batched SHORT-prompt prefill (single-pass, every valid_len <= one GDN bucket):
        process users in groups of <= group_size through ONE hybrid forward per group instead of B
        sequential B=1 forwards. Within a group the GDN layers run BATCHED (forward_prefill_batched,
        per-row valid_len masking — bit-exact per user, see test_gdn_tp_batched_prefill) and the
        full-attention layers run PER-USER (attention prefill is B=1 only). Groups of <=4 respect the
        GDN kernel cap BH=B*Nv_tp<=32. Numerically the batched GDN + per-user attention is the same
        math as prefill_paged_peruser, so per-user output (incl. DIFFERENT prompts/lengths) is
        unchanged; it just amortizes the underutilized GDN over the group.

        token_ids_list: list of B torch.Tensor [1, T_u] (lengths may differ).
        page_table:      torch.Tensor [B, blocks_per_user] int32 (row u = user u's blocks).
        valid_lens:      optional list of B ints; defaults to each T_u. Every valid_len MUST be <=
                         the derived bucket (a single GDN chunk-set); callers route longer prompts
                         to the chunked path.
        Returns:         list of B ttnn logits [1, 1, vocab] (prefill_paged_peruser contract).
        """
        assert self.num_devices > 1, "prefill_paged_grouped is the TP (num_devices>1) path"
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        B = len(token_ids_list)
        pt_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert pt_torch.shape[0] == B, "page_table must have one row per user"
        vlens = list(valid_lens) if valid_lens is not None else [int(t.shape[1]) for t in token_ids_list]

        gdn_chunk = self.args.gdn_chunk_size
        block_size = get_block_size(self._paged_kv_caches)
        # Common bucket for the group forward: round the longest prompt up to a GDN-chunk multiple.
        bucket = max(gdn_chunk, ((max(vlens) + gdn_chunk - 1) // gdn_chunk) * gdn_chunk)
        assert all(v <= bucket for v in vlens), "every valid_len must fit the single-pass bucket"

        # The fused chunk_gated_delta_rule op caps the group by its SCAN, which maps one (head,
        # v-block) row per core: BH = B*Nv_tp must stay <= the compute grid (~96-104 cores on P150).
        # With Nv_tp=12 that's B <= 8, and — unlike the old gated_delta_attn_seq kernel — it is
        # bucket-independent (SCAN L1 is state-sized, not chunk-count-sized). Validated bit-exact vs
        # per-user at B=8 for bucket 128 and 256 (test_gdn_fused_batch: ceiling + large-group). Buckets
        # >256 aren't produced here (callers route T>256 to per-user), so cap them at 1 defensively.
        gdn_max_bg = 8 if bucket <= 2 * gdn_chunk else 1
        group_size = max(1, min(group_size, gdn_max_bg))

        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        rep = ttnn.ReplicateTensorToMesh(self.device)
        # cos/sin for absolute positions [0, bucket) — shared by all users (single pass from pos 0).
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, bucket)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        csi = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

        group_rec_dev, group_conv_dev = [], []
        host_logits = [None] * B
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        for g0 in range(0, B, group_size):
            grp = list(range(g0, min(g0 + group_size, B)))
            Bg = len(grp)
            prev = self._alloc_gdn_scratch_b(Bg)
            try:
                # Batched embedding: [1, Bg, bucket, dim] (pad each user's tokens to the bucket).
                tok_bg = torch.zeros(Bg, bucket, dtype=torch.int32)
                for i, u in enumerate(grp):
                    t = token_ids_list[u][0, : vlens[u]].to(torch.int32)
                    tok_bg[i, : t.shape[0]] = t
                tok = ttnn.from_torch(tok_bg, dtype=ttnn.uint32, device=self.device, mesh_mapper=rep)
                x = self.embd(tok)  # [Bg, bucket, d]
                d = x.shape[-1]
                # Canonical residual-stream shape [1, 1, Bg*bucket, d] (dim1==1) so the framework
                # norm / MLP / residual add see the SAME layout as the validated per-user path
                # (a [1,Bg,bucket,d] shape trips "invalid subtile broadcast" in the norm/residual).
                # Reshaped to [Bg, bucket, d] only for the batched GDN, and split to [1,Bg,bucket,d]
                # to slice each user for the per-user attention. Row order is user-major (user u owns
                # rows [u*bucket : (u+1)*bucket]), matching the group state assembly.
                x = ttnn.reshape(x, (1, 1, Bg * bucket, d))
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(tok)
                # Per-user device page tables (full + real-blocks-only for the KV fill).
                full_pts, chunk_pts = [], []
                for u in grp:
                    row = pt_torch[u : u + 1].contiguous()
                    full_pts.append(
                        ttnn.from_torch(row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
                    )
                    blkN = num_blocks_in_seq(vlens[u], block_size)
                    chunk_pts.append(
                        ttnn.from_torch(
                            row[:, :blkN].contiguous(),
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=self.device,
                        )
                    )

                for layer in self.layers:
                    # The DistributedNorm all-gathers to the FULL hidden dim for the module (dn), so
                    # attn_in's last dim is the full dim (!= d, the fractured residual-stream dim).
                    attn_in = layer.attention_norm(x, mode=Mode.PREFILL)  # [1, 1, Bg*bucket, full]
                    full = attn_in.shape[-1]
                    if layer.is_full_attention:
                        attn_in_b = ttnn.reshape(attn_in, (1, Bg, bucket, full))  # split users for slicing
                        outs = []
                        for i, u in enumerate(grp):
                            xi = ttnn.reshape(attn_in_b[:, i : i + 1, :, :], (1, 1, bucket, full))
                            oi = layer.attention.forward_prefill_paged(
                                xi,
                                cos,
                                sin,
                                full_pts[i],
                                chunk_page_table=chunk_pts[i],
                                chunk_start_idx=0,
                                chunk_start_idx_tensor=csi,
                                user_id=0,  # per-user page tables are single-row; blocks route via values
                            )
                            ttnn.deallocate(xi)  # per-user slice copy
                            outs.append(ttnn.reshape(oi, (1, 1, bucket, oi.shape[-1])))
                        # Concat user outputs along the seq dim -> [1, 1, Bg*bucket, d_out] (user-major).
                        attn_out = ttnn.concat(outs, dim=2) if Bg > 1 else outs[0]
                        for o in outs:
                            if o is not attn_out:
                                ttnn.deallocate(o)
                    else:
                        # Batched GDN over the group (per-row valid_len masking, from scratch).
                        gdn_in = ttnn.reshape(attn_in, (Bg, bucket, full))
                        attn_out = layer.attention.forward_prefill_batched(
                            gdn_in, chunk_size=gdn_chunk, valid_lens=[vlens[u] for u in grp], carry=False
                        )  # [1, Bg, bucket, d_out]
                        attn_out = ttnn.reshape(attn_out, (1, 1, Bg * bucket, attn_out.shape[-1]))
                    ttnn.deallocate(attn_in)
                    h = ttnn.add(x, attn_out)  # both [1, 1, Bg*bucket, d]
                    ttnn.deallocate(x)
                    ttnn.deallocate(attn_out)
                    ff_in = layer.ffn_norm(h, mode=Mode.PREFILL)
                    ff_out = layer.feed_forward.forward(ff_in)
                    ttnn.deallocate(ff_in)
                    x = ttnn.add(h, ff_out)
                    ttnn.deallocate(h)
                    ttnn.deallocate(ff_out)

                # Final norm + per-user next-token logit at valid_len-1, read to host immediately.
                xn = self.norm(x, mode=Mode.PREFILL)  # [1, 1, Bg*bucket, full]
                ttnn.deallocate(x)
                xn_b = ttnn.reshape(xn, (1, Bg, bucket, xn.shape[-1]))
                for i, u in enumerate(grp):
                    x_last = xn_b[:, i : i + 1, vlens[u] - 1 : vlens[u], :]  # [1,1,1,full] (slice copy)
                    lg = self._lm_head_linear(x_last)
                    ttnn.deallocate(x_last)
                    host_logits[u] = (
                        ttnn.to_torch(lg, mesh_composer=comp).reshape(1, 1, -1)[:, :, : self.args.vocab_size].clone()
                    )
                    ttnn.deallocate(lg)
                ttnn.deallocate(xn)
                for t in full_pts + chunk_pts:
                    ttnn.deallocate(t)

                # Clone the group's batched GDN state (survives the next group's scratch reset).
                group_rec_dev.append([ttnn.clone(dn.rec_state) for dn in dn_layers])
                group_conv_dev.append([[ttnn.clone(dn.conv_states[m]) for m in range(dn.K)] for dn in dn_layers])
            finally:
                self._restore_gdn_batched(prev)

        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(csi)
        ttnn.synchronize_device(self.device)
        # Stitch the per-group states into the full [B,...] batched decode buffers (row u = user u).
        self._assemble_groups_gdn_dev(group_rec_dev, group_conv_dev)
        return self._reupload_host_logits(host_logits)

    def _fill_paged_cache_from_prefill(self, page_table):
        """Copy concat K/V into paged cache after prefill (one layer at a time to limit memory)."""
        for cache_idx, layer_idx in enumerate(self._attention_layer_indices):
            attn = self.layers[layer_idx].attention
            if attn.past_key is not None:
                k_cache, v_cache = self._paged_kv_caches[cache_idx]
                key = (
                    ttnn.typecast(attn.past_key, k_cache.dtype)
                    if attn.past_key.dtype != k_cache.dtype else attn.past_key
                )
                value = (
                    ttnn.typecast(attn.past_value, v_cache.dtype)
                    if attn.past_value.dtype != v_cache.dtype else attn.past_value
                )
                ttnn.experimental.paged_fill_cache(k_cache, key, page_table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(v_cache, value, page_table, batch_idx=0)
                if key is not attn.past_key:
                    ttnn.deallocate(key)
                if value is not attn.past_value:
                    ttnn.deallocate(value)
                ttnn.deallocate(attn.past_key)
                ttnn.deallocate(attn.past_value)
                attn.past_key = None
                attn.past_value = None

    def prefill_paged(self, token_ids, page_table, valid_len=None, vision_tokens=None):
        """Prefill using paged attention for long sequences, concat for short.

        For T > 1024: uses paged prefill (paged_fill_cache + chunked_sdpa)
        via prefill_layer_chunked with page_table.
        For T <= 1024: uses direct concat prefill + post-hoc paged cache fill.

        Args:
            token_ids: torch.Tensor [B, T] token IDs
            page_table: torch.Tensor or ttnn.Tensor [B, max_blocks_per_seq] int32
        Returns:
            logits: ttnn.Tensor [B, 1, vocab_size]
        """
        if self.num_devices > 1:
            return self._prefill_paged_tp(token_ids, page_table, valid_len=valid_len, vision_tokens=vision_tokens)

        B, T = token_ids.shape
        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text) before any cos/sin seam;
        # prefill_layer_chunked (T>1024) inherits the staged table.
        self._build_request_rope(token_ids[:, :valid_len] if valid_len else token_ids, vision_tokens)
        # Keep page_table as torch.Tensor for CPU slicing in prefill_layer_chunked.
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        self.reset_state(batch_size=B)
        features = getattr(self, "prefill_features", None)
        if features is not None:
            features.begin_sequence()

        # Concat-based prefill for SDPA.
        if T > 1024:
            logits = self.prefill_layer_chunked(
                token_ids, chunk_size=2048, page_table=page_table_torch, vision_tokens=vision_tokens
            )
        else:
            token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
            x = self.embd(token_ids_ttnn)
            x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
            ttnn.deallocate(token_ids_ttnn)

            cos, sin = self.rope.get_prefill_rot_mats(0, T)

            for index, layer in enumerate(self.layers):
                x = layer.forward(x, cos=cos, sin=sin, mode="prefill")
                if features is not None:
                    features.capture_tail(index, x, T)
            if features is not None:
                features.commit_tail(T)

            x = self.norm(x, mode=Mode.PREFILL)
            x_last = x[:, -1:, :]
            self._store_last_hidden(x_last)
            logits = self._lm_head(x_last)
            ttnn.deallocate(x)

        # Post-prefill: paged_fill no-op if already paged (T>1024); copies concat KV otherwise.
        page_table_device = ttnn.from_torch(
            page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        self._fill_paged_cache_from_prefill(page_table_device)

        # Fuse DeltaNet conv states for decode.
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                if dn.fused_conv_state is None and dn.conv_state_q is not None:
                    dn.fused_conv_state = ttnn.concat([dn.conv_state_q, dn.conv_state_k, dn.conv_state_v], dim=2)
                    dn.fused_conv_state = ttnn.to_layout(dn.fused_conv_state, ttnn.TILE_LAYOUT)

        # Copy DeltaNet state into external pre-allocated buffers.
        if self._deltanet_external_states is not None:
            dn_idx = 0
            for layer in self.layers:
                if not layer.is_full_attention:
                    dn = layer.attention
                    ext_rec, ext_conv = self._deltanet_external_states[dn_idx]
                    if dn.recurrent_state is not ext_rec:
                        ttnn.copy(dn.recurrent_state, ext_rec)
                    if dn.fused_conv_state is not None and dn.fused_conv_state is not ext_conv:
                        ttnn.copy(dn.fused_conv_state, ext_conv)
                    dn_idx += 1

        return logits

    def decode_paged(self, token_ids, current_pos, page_table):
        """Single-token paged decode. Returns logits [B,1,vocab_size]."""
        B = token_ids.shape[0]
        # Accept torch or ttnn page_table.
        if isinstance(page_table, torch.Tensor):
            page_table = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        ttnn.deallocate(token_ids_ttnn)

        # RoPE position offset by rope_delta for multimodal (KV position stays the true seq pos).
        position_ids = torch.full((B, 1), current_pos + self.rope.rope_delta, dtype=torch.long)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # cur_pos [B] for paged ops (not [B*n_kv] like non-paged decode).
        cur_pos_tensor = ttnn.from_torch(
            torch.full((B,), current_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        for layer in self.layers:
            if layer.is_full_attention:
                x = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="decode",
                    position_tensor=cur_pos_tensor,
                    page_table=page_table,
                )
            else:
                x = layer.forward(x, cos=cos, sin=sin, mode="decode")

        x = self._final_norm_decode(x)
        self._store_last_hidden(x)
        logits = self._lm_head(x)
        ttnn.deallocate(x)

        return logits

    def _decode_layers_paged(self, x, cos, sin, cur_pos_tensor, page_table):
        for layer in self.layers:
            if layer.is_full_attention:
                x = layer.forward(
                    x, cos=cos, sin=sin, mode="decode", position_tensor=cur_pos_tensor, page_table=page_table
                )
            else:
                x = layer.forward(x, cos=cos, sin=sin, mode="decode")
        x = self._final_norm_decode(x)
        logits = self._lm_head(x)
        # Greedy token computed in the same trace: a 4-byte readback per token instead
        # of the full 248k-entry logits row. The LM head emits exactly vocab_size columns.
        logits_rows = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        token = ttnn.argmax(logits_rows, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(logits_rows)
        if getattr(self, "_decode_keep_hidden", False):
            # Final-norm hidden row (the MTP draft input) stays a trace output.
            return logits, token, x
        ttnn.deallocate(x)
        return logits, token

    def decode_traced(self, token_id, current_pos, page_table):
        """Single-device, batch-one paged decode replayed from one captured trace.

        Only the embedding row, RoPE row, and position are written per token; every
        layer and the LM head replay from the trace. The first call is a real eager
        step that compiles every program; the second call captures the trace (capture
        records without executing) and then replays it once. ``page_table`` must stay fixed.
        Returns device ``(logits, greedy_token)``, valid until the next call; the token is a
        uint32 ``[1, 1, 1]`` argmax over the real vocabulary.
        """
        if self.num_devices != 1:
            raise ValueError("decode_traced supports a single device")
        if self.mtp is not None:
            raise ValueError("decode_traced does not capture the MTP hidden-state hook")
        embedded = self.embd.weight[int(token_id)].reshape(1, 1, -1).to(torch.bfloat16)
        embed_host = ttnn.from_torch(embedded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        pos_host = ttnn.from_torch(torch.tensor([current_pos], dtype=torch.int32), dtype=ttnn.int32)
        cos_host, sin_host = self.rope.get_cos_sin_host(current_pos + self.rope.rope_delta)
        state = getattr(self, "_decode_trace_state", None)
        if state is None:
            if isinstance(page_table, torch.Tensor):
                page_table = ttnn.from_torch(
                    page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                )
            self._decode_page_table = page_table
            self._decode_embed = ttnn.to_device(embed_host, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._decode_pos = ttnn.to_device(pos_host, self.device)
        else:
            ttnn.copy_host_to_device_tensor(embed_host, self._decode_embed)
            ttnn.copy_host_to_device_tensor(pos_host, self._decode_pos)
        # RoPE decode rows are the stable buffers owned by Qwen36RoPESetup.
        ttnn.copy_host_to_device_tensor(cos_host, self.rope.cos_device)
        ttnn.copy_host_to_device_tensor(sin_host, self.rope.sin_device)
        args = (self._decode_embed, self.rope.cos_device, self.rope.sin_device, self._decode_pos,
                self._decode_page_table)
        if state is None:
            self._decode_trace_state = "compiled"
            self._decode_outputs = self._decode_layers_paged(*args)
        elif state == "compiled":
            for tensor in self._decode_outputs:
                ttnn.deallocate(tensor)
            trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._decode_outputs = self._decode_layers_paged(*args)
            ttnn.end_trace_capture(self.device, trace, cq_id=0)
            self._decode_trace = trace
            self._decode_trace_state = "captured"
        if state is not None:
            ttnn.execute_trace(self.device, self._decode_trace, cq_id=0, blocking=False)
        return self._decode_outputs

    def release_decode_trace(self):
        if getattr(self, "_decode_trace_state", None) == "captured":
            ttnn.release_trace(self.device, self._decode_trace)
        self._decode_trace_state = None

    # Generator contract — decode

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Build HOST decode inputs: (tokens_tt, cur_pos_tt, rope_packed, page_table_tt)."""
        from models.demos.blackhole.qwen36.tt.generator_interface import pack_rope_host

        B = tokens.shape[0]
        tokens_tt = ttnn.from_torch(tokens.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Per-user positions: current_pos may be a [B] tensor (each user at its own position) or a
        # scalar (lockstep). Build a [B] int32 vector so cur_pos and rope carry one rotation per user.
        if isinstance(current_pos, torch.Tensor):
            pos_vec = current_pos.to(torch.int32).reshape(-1)
            assert pos_vec.shape[0] == B, f"current_pos length {pos_vec.shape[0]} != batch {B}"
        else:
            pos_vec = torch.full((B,), int(current_pos), dtype=torch.int32)
        # RoPE position is the KV position offset by rope_delta (multimodal compresses the position
        # space; post-image text has t==h==w so 1D RoPE at rope_pos is correct). cur_pos_tt below
        # stays the true KV position. rope_delta is 0 for text, so this is a no-op there.
        rope_pos_vec = pos_vec + self.rope.rope_delta
        if self.num_devices > 1 or self._batched_single_device:
            # TP: rope_tp cos/sin [1,B,1,rope_dim] packed on host.
            rd = self.args.rope_head_dim
            inv_freq = 1.0 / (self.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
            freqs = torch.outer(rope_pos_vec.float(), inv_freq)  # [B, rd/2], per-user rotation
            emb = torch.cat([freqs, freqs], dim=-1)
            shape = (1, 1, B, rd) if self._batched_single_device else (1, B, 1, rd)
            cos = emb.cos().reshape(shape).to(torch.bfloat16)
            sin = emb.sin().reshape(shape).to(torch.bfloat16)
            rope_packed = ttnn.from_torch(torch.cat([cos, sin], dim=0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:
            cos_host, sin_host = self.rope.get_cos_sin_host(int(rope_pos_vec[0]))  # HOST ttnn [1,1,rope_head_dim]
            rope_packed = pack_rope_host(cos_host, sin_host)  # torch-based (host)
        cur_pos_tt = ttnn.from_torch(pos_vec, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        page_table_tt = (
            ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            if page_table is not None
            else None
        )
        return tokens_tt, cur_pos_tt, rope_packed, page_table_tt

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        """Host-to-device transfer for decode inputs."""
        from models.tt_transformers.tt.common import copy_host_to_device

        host = self.prepare_decode_inputs_host(tokens, current_pos, page_table=page_table)
        return copy_host_to_device(host, mesh_device=self.mesh_device)

    def ttnn_decode_forward(
        self,
        tokens,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        on_device_logits=False,
        **kwargs,
    ):
        """Generator decode forward. kv_cache accepted but unused (state is model-bound).

        on_device_logits=True: return the raw vocab-sharded shard for the on-device sampler.
        """
        from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope

        cos, sin = unpack_rope(rot_mat_idxs)
        if on_device_logits:
            assert self.sampling is not None, "on_device_logits=True but self.sampling is None"
            logits = self._forward_decode(tokens, cos, sin, current_pos, page_table, sharded_lm_head=True)
            # Sampler runs >=32-wide; pad B up to it (else shape mismatch). Extra slots unused.
            sampler_batch = self.sampling.tt_sampling.max_batch_size
            B = logits.shape[2]
            if B < sampler_batch:
                logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, sampler_batch - B), (0, 0)], value=0.0)
            # Bare tensor (not a tuple): the traced path passes this straight to capture_trace().
            return logits
        logits = self._forward_decode(tokens, cos, sin, current_pos, page_table)
        return logits, None

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        """Convert decode output to host torch. Host-sampling returns logits [B,S,vocab];
        on-device sampling returns sampled token ids or sampled-token log-probs.
        """
        if is_tokens or is_log_probs:
            # Sampled ids and old-path sampled-token log-probs are replicated across devices.
            if self.num_devices > 1:
                return ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(-1)[:B]
            return ttnn.to_torch(tt_out).reshape(-1)[:B]
        if self.num_devices > 1:
            # TP: read one replica (get_device_tensors[0]), not ConcatMeshToTensor (~4x readback).
            full = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        else:
            full = ttnn.to_torch(tt_out).float()
        rows = full.reshape(-1, self.args.vocab_size)
        required_rows = B * S
        if rows.shape[0] < required_rows:
            # Decode bucketing returns only the active prefix. The shared
            # generator requests the fixed serving width, so add neutral rows
            # instead of splitting each vocabulary row during ``view(B, S, -1)``.
            rows = torch.nn.functional.pad(rows, (0, 0, 0, required_rows - rows.shape[0]))
        return rows[:required_rows].view(B, S, self.args.vocab_size)

    def _gdn_layers(self):
        return [layer.attention for layer in self.layers if not layer.is_full_attention]

    def _save_gdn_snapshot(self):
        """Bit-exact host copy of the B=1 GDN recurrent + convolution state (the pooled
        convolution allocation is read whole, never through its per-layer views)."""
        pool = self._deltanet_conv_pool
        layers = self._gdn_layers()
        return {
            "recurrent": [ttnn.to_torch(dn.recurrent_state) for dn in layers],
            "conv_pool": ttnn.to_torch(pool) if pool is not None else None,
            "conv": None if pool is not None else [ttnn.to_torch(dn.fused_conv_state) for dn in layers],
        }

    def _restore_gdn_snapshot(self, saved):
        """Write a _save_gdn_snapshot back into the fixed (trace-baked) state buffers."""

        def put(host, destination):
            # Stage through DRAM-interleaved: a direct copy into a sharded buffer is not reliable.
            staged = ttnn.from_torch(
                host, dtype=destination.dtype, layout=destination.layout, device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.copy(staged, destination)
            ttnn.deallocate(staged)

        layers = self._gdn_layers()
        if len(saved["recurrent"]) != len(layers):
            raise ValueError("GDN snapshot does not match this model")
        if saved["conv_pool"] is not None:
            put(saved["conv_pool"], self._deltanet_conv_pool)
        for index, dn in enumerate(layers):
            put(saved["recurrent"][index], dn.recurrent_state)
            if saved["conv"] is not None:
                put(saved["conv"][index], dn.fused_conv_state)
            if getattr(dn, "split_conv_state", None) is not None:
                dn._restore_split_conv_from_fused()

    def _save_deltanet_states(self):
        """Snapshot GDN state to host (guard across decode-trace capture's double forward)."""
        saved = []
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                saved.append(
                    {
                        "recurrent": ttnn.to_torch(dn.recurrent_state),
                        "conv": ttnn.to_torch(dn.fused_conv_state) if dn.fused_conv_state is not None else None,
                    }
                )
        return saved

    def _restore_deltanet_states(self, saved_states, device):
        """Restore GDN state via ttnn.copy (preserves trace-baked buffer addresses)."""
        if self._deltanet_conv_pool is not None:
            # Restore the pool, not narrow aliases: device copies into a view can lose
            # its shard offset and silently overwrite another layer's convolution rows.
            conv_host = torch.cat([state["conv"] for state in saved_states], dim=1)
            restored_pool = ttnn.from_torch(
                conv_host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.copy(restored_pool, self._deltanet_conv_pool)
            ttnn.deallocate(restored_pool)
        idx = 0
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                saved = saved_states[idx]
                # Source must be DRAM-interleaved: a sharded-L1 source copies stale data
                # into the sharded persistent buffer once other L1 programs have run.
                restored = ttnn.from_torch(
                    saved["recurrent"], dtype=dn.recurrent_state.dtype, layout=dn.recurrent_state.layout, device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.copy(restored, dn.recurrent_state)
                ttnn.deallocate(restored)
                if saved["conv"] is not None and self._deltanet_conv_pool is None:
                    restored_conv = ttnn.from_torch(
                        saved["conv"], dtype=ttnn.bfloat16, layout=dn.fused_conv_state.layout, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    ttnn.copy(restored_conv, dn.fused_conv_state)
                    ttnn.deallocate(restored_conv)
                if getattr(dn, "split_conv_state", None) is not None:
                    dn._restore_split_conv_from_fused()
                idx += 1
