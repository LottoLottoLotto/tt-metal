# SPDX-License-Identifier: Apache-2.0
"""Native BF16 DFlash2 for hash-pinned Qwen3.8-27B and Qwen3.6-35B-A3B drafts.

Only checkpoint initialization and prefill metadata use Torch/host computation.
Draft consumes device scalars and never reads tensor values back to the host.
The target owns embeddings, vocabulary head, capture, and request scheduling.
"""
import hashlib
import json
from pathlib import Path

import torch
import ttnn

from .dflash_cache import CommittedKV, DRAM
from .dflash_ops import NativeCandidateSelector, NativeDynamicConv
from .rms_norm import bounded_rms_norm

# Identity is coupled to the complete supported architecture, not just file size.
# 27B: z-lab revision 50307d4c; 35B: incoai revision 51ef7b6923ad6c14cb1bb41c37a9041446496ab6.
_CHECKPOINTS = {
    5120: {
        "sha256": "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c",
        "bytes": 3848817896, "intermediate_size": 17408, "num_hidden_layers": 5,
        "mask_token_id": 248070, "target_layer_ids": [5, 19, 33, 47, 61],
    },
    2048: {
        "sha256": "d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d",
        "bytes": 1052513456, "intermediate_size": 6144, "num_hidden_layers": 6,
        "mask_token_id": 248077, "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
    },
}


def load_dflash_checkpoint(directory):
    """Load the hash-pinned raw safetensors state dictionary and its config.

    The caller may instead supply a state_dict from its own validated loader.
    This loader deliberately does not remap the two raw selector codebook keys.
    """
    from safetensors.torch import load_file

    directory = Path(directory)
    config = _validate_config(json.loads((directory / "config.json").read_text()))
    identity = _CHECKPOINTS[config["hidden_size"]]
    path = directory / "model.safetensors"
    before = path.stat()
    if before.st_size != identity["bytes"]:
        raise ValueError("DFlash checkpoint size differs from the pinned checkpoint for this config")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != identity["sha256"]:
        raise ValueError(f"DFlash checkpoint SHA256 mismatch: {digest}")
    state = load_file(str(path), device="cpu")
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("DFlash checkpoint changed during loading")
    return state, config


def _configuration(config):
    return config if isinstance(config, dict) else config.to_dict()


def _validate_config(config):
    config = _configuration(config)
    hidden_size = config.get("hidden_size")
    if hidden_size not in _CHECKPOINTS:
        raise ValueError(f"Unsupported DFlash hidden_size={hidden_size!r}")
    identity = _CHECKPOINTS[hidden_size]
    expected = {
        "hidden_size": hidden_size, "intermediate_size": identity["intermediate_size"],
        "num_hidden_layers": identity["num_hidden_layers"], "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 248320, "sliding_window": 2048,
        "is_causal": False, "attention_bias": False,
        "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"DFlash requires {key}={value!r}, got {config.get(key)!r}")
    if config.get("layer_types") != ["sliding_attention"] * identity["num_hidden_layers"]:
        raise ValueError("DFlash requires sliding attention in every trained layer")
    if hidden_size == 2048 and config.get("num_target_layers") != 40:
        raise ValueError("The 35B DFlash checkpoint requires a 40-layer target")
    rope = config.get("rope_parameters", {})
    if rope.get("rope_theta") != 10000000 or rope.get("rope_type") != "default":
        raise ValueError("DFlash requires full default RoPE with theta=1e7")
    draft = config.get("dflash_config", {})
    for key, value in {
        "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": identity["mask_token_id"],
        "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": identity["target_layer_ids"],
    }.items():
        if draft.get(key) != value:
            raise ValueError(f"DFlash requires dflash_config.{key}={value!r}")
    # The pinned checkpoint has neither scaling nor softcapping. Do not silently
    # accept a different model's output policy under this checkpoint identity.
    for key, value in (("output_multiplier", 1.0), ("final_logit_softcapping", None)):
        if draft.get(key, config.get(key, value)) != value:
            raise ValueError(f"The pinned DFlash checkpoint requires {key}={value!r}")
    return config


class NativeDFlash:
    """Trained single-sequence DFlash2 with a committed-only 2048-row ring.

    Public ownership/shape contract:
      * state_dict: real BF16 raw checkpoint tensors from a validated loader;
        host weights are not retained after initialization.
      * embed_tokens([1,B] IDs) -> BF16 hidden with width hidden_size; output_head
        takes [1,1,B-1,hidden_size] and returns full BF16 vocabulary logits.
      * prefill(features,start_position,valid_length): features [1,1,L,feature_width],
        host integer metadata; append only those valid rows, no noisy backbone.
      * draft(anchor_ids,position,features,feature_start,feature_count): metadata
        all row-major DRAM int32/uint32 [1,1], features [1,1,B,feature_width].
        Committed features cover [feature_start,feature_start+feature_count),
        with count in [0,B] and end <= position. Position is the known anchor's
        absolute position, not a private inferred cursor. Returns native uint32
        [1,B-1], borrowed until the next draft of this instance.
      * draft_branches(...,fork_position): same metadata as draft, B8 only.
        Returns two simultaneously live borrowed uint32 [1,7] paths after one
        feature append, backbone, vocabulary head, and shared selector setup.
      * reset invalidates ownership without reallocating; release frees only
        this instance's weights/caches, never target callback-owned resources.

    Ring ownership, masks, and ID staging have fixed addresses. Native operation
    temporaries are trace-capture allocations; Main owns warmup/capture/replay.
    All commands, including reset and target callbacks, must use the same queue.
    """
    def __init__(self, device, state_dict, config, embed_tokens, output_head, cache_prefix, block_size=8):
        if block_size not in (4, 8):
            raise ValueError("Native DFlash supports block sizes 4 and 8")
        self.config = _validate_config(config)
        self.hidden_size = self.config["hidden_size"]
        self.feature_layers = tuple(self.config["dflash_config"]["target_layer_ids"])
        self.feature_width = self.hidden_size * len(self.feature_layers)
        self.device = device
        self.block_size = block_size
        self.embed_tokens = embed_tokens
        self.output_head = output_head
        self._owned = []
        self.layers = []
        self.selector = None
        self.cache = None
        self._released = False
        grid = device.compute_with_storage_grid_size()
        if grid.x < 8 or grid.y < 8:
            raise ValueError("Native DFlash requires at least an 8x8 worker grid")
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=False,
        )
        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
            q_chunk_size=32, k_chunk_size=32,
        )
        self._matmul_configs = {}
        prefix = str(Path(cache_prefix) / _CHECKPOINTS[self.hidden_size]["sha256"] / "native_bf16")
        hidden_size = self.hidden_size
        intermediate_size = self.config["intermediate_size"]

        def raw(key, shape):
            tensor = state_dict[key]
            if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                raise ValueError(f"DFlash weight {key} must be BF16 {shape}")
            return tensor

        def linear(key, out_features, in_features):
            return self._upload(raw(key, (out_features, in_features)).T.contiguous().reshape(
                1, 1, in_features, out_features))

        def norm(key, width):
            # Qwen3 norms are ordinary learned weights, not target Qwen3.5's 1+w.
            return self._upload(raw(key, (width,)).reshape(1, 1, 1, width))

        try:
            self.fc = linear("fc.weight", hidden_size, self.feature_width)
            self.hidden_norm = norm("hidden_norm.weight", hidden_size)
            self.final_norm = norm("norm.weight", hidden_size)
            for index in range(self.config["num_hidden_layers"]):
                p = f"layers.{index}"
                layer = {}
                self.layers.append(layer)
                layer["input_norm"] = norm(f"{p}.input_layernorm.weight", hidden_size)
                layer["post_norm"] = norm(f"{p}.post_attention_layernorm.weight", hidden_size)
                layer["q_norm"] = norm(f"{p}.self_attn.q_norm.weight", 128)
                layer["k_norm"] = norm(f"{p}.self_attn.k_norm.weight", 128)
                for name, out_width, in_width in (
                    ("q_proj", 4096, hidden_size), ("k_proj", 1024, hidden_size),
                    ("v_proj", 1024, hidden_size), ("o_proj", hidden_size, 4096),
                ):
                    layer[name] = linear(f"{p}.self_attn.{name}.weight", out_width, in_width)
                for name, out_width, in_width in (
                    ("gate_proj", intermediate_size, hidden_size), ("up_proj", intermediate_size, hidden_size),
                    ("down_proj", hidden_size, intermediate_size),
                ):
                    layer[name] = linear(f"{p}.mlp.{name}.weight", out_width, in_width)
                for name in ("attention_conv", "mlp_conv"):
                    layer[name] = NativeDynamicConv(
                        device, raw(f"{p}.{name}.base_kernel", (2, 2, hidden_size)),
                        raw(f"{p}.{name}.kernel_projection.weight", (hidden_size // 4, hidden_size)),
                        f"{prefix}/{p}.{name}",
                    )
            self.selector = NativeCandidateSelector(
                device,
                raw("candidate_selector.predecessor_codebook", (248320, 256)),
                raw("candidate_selector.successor_codebook", (248320, 256)),
                raw("candidate_selector.hidden_projection.weight", (256, hidden_size)),
                f"{prefix}/candidate_selector", top_k=16,
            )
            # Constant RoPE tables are initialization, not host inference. Build
            # FP32 frequencies then cast cos/sin to BF16 exactly as the reference.
            inverse = 1.0 / (10000000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
            positions = torch.arange(262144 + 32, dtype=torch.float32)
            angles = positions[:, None] * inverse[None, :]
            angles = torch.cat((angles, angles), dim=-1)
            self.cos_table = self._upload(angles.cos().to(torch.bfloat16), row_major=True)
            self.sin_table = self._upload(angles.sin().to(torch.bfloat16), row_major=True)
            self.cache = CommittedKV(
                device, block_size, num_layers=len(self.layers), feature_width=self.feature_width,
                mask_token_id=self.config["dflash_config"]["mask_token_id"],
            )
        except Exception:
            self.release()
            raise

    def _upload(self, tensor, *, row_major=False):
        result = ttnn.from_torch(tensor, device=self.device, dtype=ttnn.bfloat16,
                                 layout=ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT,
                                 memory_config=DRAM)
        self._owned.append(result)
        return result

    def _check_live(self):
        if self._released:
            raise RuntimeError("NativeDFlash has been released")

    def _hidden(self, tensor, rows, width=None):
        if width is None:
            width = self.hidden_size
        if (tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT
                or int(tensor.shape[-1]) != width):
            raise ValueError(f"DFlash requires tiled BF16 hidden width {width}")
        logical = tuple(int(dim) for dim in tensor.shape)
        if logical not in ((1, rows, width), (1, 1, rows, width)):
            raise ValueError(f"DFlash expected one sequence with {rows} rows, got {logical}")
        tensor = ttnn.reshape(tensor, (1, 1, rows, width))
        return ttnn.to_memory_config(tensor, DRAM)

    def _linear(self, hidden, weight):
        width = int(weight.shape[-1])
        config = self._matmul_configs.get(width)
        if config is None:
            config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 8), in0_block_w=1,
                out_subblock_h=1, out_subblock_w=1, per_core_M=1,
                per_core_N=((width // 32) + 63) // 64,
                fuse_batch=True, fused_activation=None, mcast_in0=True,
            )
            self._matmul_configs[width] = config
        return ttnn.linear(hidden, weight, dtype=ttnn.bfloat16, memory_config=DRAM,
                            program_config=config, compute_kernel_config=self.compute)

    def _norm(self, hidden, weight):
        # Preserve reference's BF16 cast between normalization and learned scale.
        # Reuse the target's bounded 40-core norm geometry (without its weights
        # or zero-centering policy); interleaved H5120 norm can overflow L1.
        if int(hidden.shape[-1]) == 5120:
            normalized = bounded_rms_norm(hidden, None, 1e-6, self.compute)
        else:
            normalized = ttnn.rms_norm(hidden, epsilon=1e-6, memory_config=DRAM,
                                       compute_kernel_config=self.compute)
        return ttnn.multiply(normalized, weight, memory_config=DRAM)

    def _rope_tables(self, indices, rows):
        result = []
        for table in (self.cos_table, self.sin_table):
            embedded = ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT,
                                       dtype=ttnn.bfloat16, memory_config=DRAM)
            embedded = ttnn.reshape(embedded, (1, 1, 32, 128))
            result.append(embedded[:, :, :rows, :])
        return tuple(result)

    @staticmethod
    def _rope(hidden, tables):
        cos, sin = tables
        first = hidden[:, :, :, :64]
        second = hidden[:, :, :, 64:]
        rotated = ttnn.concat([ttnn.neg(second, memory_config=DRAM), first], dim=-1, memory_config=DRAM)
        # Separate BF16 products and BF16 add: no fused multiply-add contraction.
        return ttnn.add(ttnn.multiply(hidden, cos, memory_config=DRAM),
                         ttnn.multiply(rotated, sin, memory_config=DRAM), memory_config=DRAM)

    @staticmethod
    def _heads(query, key, value, query_heads):
        kv = ttnn.concat([key, value], dim=-1, memory_config=DRAM)
        return ttnn.experimental.nlp_create_qkv_heads(
            query, kv, num_heads=query_heads, num_kv_heads=8,
            transpose_k_heads=False, memory_config=DRAM,
        )

    def _context_kv(self, layer, hidden, tables):
        key = self._linear(hidden, layer["k_proj"])
        value = self._linear(hidden, layer["v_proj"])
        # Use the supported native head shuffle, not a padded-row reshape that
        # could reinterpret sequence tile padding as head channels.
        unused, key, value = self._heads(key, key, value, 8)
        del unused
        return self._rope(self._norm(key, layer["k_norm"]), tables), value

    def _commit_features(self, features, rows, start, count):
        target_hidden = self._norm(self._linear(features, self.fc), self.hidden_norm)
        tables = self._rope_tables(self.cache.feature_positions, rows)
        for index, layer in enumerate(self.layers):
            key, value = self._context_kv(layer, target_hidden, tables)
            self.cache.commit(index, key, value, start, count)

    def prefill(self, features, start_position, valid_length, source_offset=0):
        """Append a prompt feature suffix; metadata may be host integers here."""
        self._check_live()
        if not all(isinstance(value, int) for value in (start_position, valid_length, source_offset)):
            raise ValueError("DFlash prefill position, length, and source offset must be host integers")
        length = int(features.shape[-2])
        features = self._hidden(features, length, self.feature_width)
        if (
            start_position < 0
            or not 0 <= source_offset <= valid_length <= length
            or start_position + valid_length > 262144
        ):
            raise ValueError("DFlash prefill valid rows exceed the input or position limit")
        for offset in range(source_offset, valid_length, 32):
            count = min(32, valid_length - offset)
            offset_tt = self.cache.host_scalar(offset)
            chunk = self.cache.stage_prefill(features, offset_tt)
            start_tt = self.cache.host_scalar(start_position + offset)
            count_tt = self.cache.host_scalar(count)
            self.cache.prepare(start_tt, start_tt, start_tt, count_tt, prefill=True)
            self._commit_features(chunk, 32, start_tt, count_tt)

    def reset(self):
        self._check_live()
        self.cache.reset()

    def draft(self, anchor_ids, position, features, feature_start, feature_count):
        """Commit device-counted features and propose B-1 tokens, entirely native."""
        proposed_hidden, logits = self._draft_hidden(anchor_ids, position, features, feature_start, feature_count)
        return self.selector.select(proposed_hidden, logits, anchor_ids)

    def draft_branches(self, anchor_ids, position, features, feature_start, feature_count, fork_position):
        """Propose A and a learned rank-two fork B with a single draft backbone."""
        self._check_live()
        if self.block_size != 8 or type(fork_position) is not int or not 1 <= fork_position <= 7:
            raise ValueError("DFlash draft_branches requires B8 and a host fork_position in 1..7")
        proposed_hidden, logits = self._draft_hidden(anchor_ids, position, features, feature_start, feature_count)
        return self.selector.select_branches(proposed_hidden, logits, anchor_ids, fork_position)

    def _draft_hidden(self, anchor_ids, position, features, feature_start, feature_count):
        self._check_live()
        features = self._hidden(features, self.block_size, self.feature_width)
        self.cache.prepare(anchor_ids, position, feature_start, feature_count)
        self._commit_features(features, self.block_size, feature_start, feature_count)
        hidden = self._hidden(self.embed_tokens(self.cache.token_ids), self.block_size)
        tables = self._rope_tables(self.cache.noise_positions, self.block_size)
        for index, layer in enumerate(self.layers):
            residual = hidden
            normalized = self._norm(hidden, layer["input_norm"])
            branch, dynamic = layer["attention_conv"].prepare(normalized)
            q, k, v = self._heads(
                self._linear(branch, layer["q_proj"]),
                self._linear(branch, layer["k_proj"]),
                self._linear(branch, layer["v_proj"]), 32,
            )
            q = self._rope(self._norm(q, layer["q_norm"]), tables)
            k = self._rope(self._norm(k, layer["k_norm"]), tables)
            key, value = self.cache.attention_inputs(index, k, v)
            attended = ttnn.transformer.scaled_dot_product_attention(
                q, key, value, attn_mask=self.cache.mask, is_causal=False,
                scale=128 ** -0.5, memory_config=DRAM, program_config=self.sdpa_config,
                compute_kernel_config=self.compute,
            )
            attended = ttnn.experimental.nlp_concat_heads(attended, memory_config=DRAM)
            attended = self._hidden(attended, self.block_size, 4096)
            branch = self._linear(attended, layer["o_proj"])
            branch = layer["attention_conv"].finish(branch, dynamic)
            hidden = ttnn.add(residual, branch, memory_config=DRAM)
            residual = hidden
            normalized = self._norm(hidden, layer["post_norm"])
            branch, dynamic = layer["mlp_conv"].prepare(normalized)
            gate = ttnn.silu(self._linear(branch, layer["gate_proj"]), memory_config=DRAM)
            up = self._linear(branch, layer["up_proj"])
            branch = self._linear(ttnn.multiply(gate, up, memory_config=DRAM), layer["down_proj"])
            branch = layer["mlp_conv"].finish(branch, dynamic)
            hidden = ttnn.add(residual, branch, memory_config=DRAM)
        hidden = self._norm(hidden, self.final_norm)
        proposed_hidden = hidden[:, :, 1:self.block_size, :]
        logits = self._hidden(self.output_head(proposed_hidden), self.block_size - 1, 248320)
        return proposed_hidden, logits

    def release(self):
        if self._released:
            return
        if self.cache is not None:
            self.cache.release()
        if self.selector is not None:
            self.selector.release()
        for layer in self.layers:
            for name in ("attention_conv", "mlp_conv"):
                if name in layer:
                    layer[name].release()
        for tensor in self._owned:
            ttnn.deallocate(tensor)
        self._owned.clear()
        self.layers.clear()
        self._released = True
