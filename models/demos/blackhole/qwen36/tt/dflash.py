# SPDX-License-Identifier: Apache-2.0
"""Native BF16 Qwen3.8-27B DFlash2, pinned to z-lab revision 50307d4c.

Only checkpoint initialization and prefill metadata use Torch/host computation.
Draft consumes device scalars and never reads tensor values back to the host.
The target owns embeddings, vocabulary head, capture, and request scheduling.
"""
import hashlib
import os
import json
from pathlib import Path

import torch
import ttnn

from .dflash_cache import CommittedKV, DRAM
from .dflash_ops import NativeCandidateSelector, NativeDynamicConv, retile_heads
from .rms_norm import bounded_rms_norm

DRAFT_SHA256 = "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
DRAFT_BYTES = 3848817896


def load_dflash_checkpoint(directory):
    """Load the hash-pinned raw safetensors state dictionary and its config.

    The caller may instead supply a state_dict from its own validated loader.
    This loader deliberately does not remap the two raw selector codebook keys.
    """
    from safetensors.torch import load_file

    directory = Path(directory)
    path = directory / "model.safetensors"
    before = path.stat()
    if before.st_size != DRAFT_BYTES:
        raise ValueError("DFlash checkpoint size differs from the pinned checkpoint")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != DRAFT_SHA256:
        raise ValueError(f"DFlash checkpoint SHA256 mismatch: {digest}")
    state = load_file(str(path), device="cpu")
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("DFlash checkpoint changed during loading")
    config = json.loads((directory / "config.json").read_text())
    _validate_config(config)
    return state, config


def _configuration(config):
    return config if isinstance(config, dict) else config.to_dict()


def _validate_config(config):
    config = _configuration(config)
    expected = {
        "hidden_size": 5120, "intermediate_size": 17408,
        "num_hidden_layers": 5, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 248320, "sliding_window": 2048,
        "is_causal": False, "attention_bias": False,
        "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"DFlash requires {key}={value!r}, got {config.get(key)!r}")
    if config.get("layer_types") != ["sliding_attention"] * 5:
        raise ValueError("DFlash requires five sliding-attention layers")
    rope = config.get("rope_parameters", {})
    if rope.get("rope_theta") != 10000000 or rope.get("rope_type") != "default":
        raise ValueError("DFlash requires full default RoPE with theta=1e7")
    draft = config.get("dflash_config", {})
    for key, value in {
        "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248070,
        "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 19, 33, 47, 61],
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
    """Five-layer single-sequence DFlash2 with a committed-only 2048-row ring.

    Public ownership/shape contract:
      * state_dict: real BF16 raw checkpoint tensors from a validated loader;
        host weights are not retained after initialization.
      * embed_tokens([1,B] IDs) -> BF16 hidden with width 5120; output_head takes
        [1,1,B-1,5120] and returns full BF16 vocabulary logits.
      * prefill(features,start_position,valid_length): features [1,1,L,25600],
        host integer metadata; append only those valid rows, no noisy backbone.
      * draft(anchor_ids,position,features,feature_start,feature_count): metadata
        all row-major DRAM int32/uint32 [1,1], features [1,1,B,25600].
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
    def __init__(self, device, state_dict, config, embed_tokens, output_head, cache_prefix, block_size=8,
                 draft_vocab=None):
        if block_size not in (4, 8):
            raise ValueError("Native DFlash supports block sizes 4 and 8")
        self.config = _validate_config(config)
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
        prefix = str(Path(cache_prefix) / DRAFT_SHA256 / "native_bf16")

        def raw(key, shape):
            tensor = state_dict[key]
            if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                raise ValueError(f"DFlash weight {key} must be BF16 {shape}")
            return tensor

        # Draft-only BFP8 projections: proposals never bypass exact target verification,
        # so weight precision changes acceptance only, never the emitted stream. Native-context
        # serving may select (QWEN36_DFLASH_DRAFT_WEIGHT_DTYPE) bfp4: BFP4 fc, attention and
        # gate/up, BFP8 down_proj; or bfp4_all: BFP4 down_proj too. BFP4 with packer L1
        # accumulation at down_proj's [17408 -> 5120] M=1 shape is reserved for the target's
        # packed MLP-down path (QWEN36_MLP_DOWN_REQUIRE_PACKED), so a BFP4 draft down_proj
        # accumulates in the FP32 destination without packer L1 accumulation.
        self.linear_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        self.down_bfp4_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=False,
        )
        policy = os.environ.get("QWEN36_DFLASH_DRAFT_WEIGHT_DTYPE", "bfp8")
        if policy not in ("bfp8", "bfp4", "bfp4_all"):
            raise ValueError("QWEN36_DFLASH_DRAFT_WEIGHT_DTYPE must be bfp8, bfp4 or bfp4_all")
        weight_dtype = ttnn.bfloat8_b if policy == "bfp8" else ttnn.bfloat4_b
        down_dtype = ttnn.bfloat4_b if policy == "bfp4_all" else ttnn.bfloat8_b

        def linear(key, out_features, in_features, dtype=None):
            return self._upload(raw(key, (out_features, in_features)).T.contiguous().reshape(
                1, 1, in_features, out_features), dtype=weight_dtype if dtype is None else dtype)

        def norm(key, width):
            # Qwen3 norms are ordinary learned weights, not target Qwen3.5's 1+w.
            return self._upload(raw(key, (width,)).reshape(1, 1, 1, width))

        try:
            self.fc = linear("fc.weight", 5120, 25600)
            self.hidden_norm = norm("hidden_norm.weight", 5120)
            self.final_norm = norm("norm.weight", 5120)
            for index in range(5):
                p = f"layers.{index}"
                layer = {}
                self.layers.append(layer)
                layer["input_norm"] = norm(f"{p}.input_layernorm.weight", 5120)
                layer["post_norm"] = norm(f"{p}.post_attention_layernorm.weight", 5120)
                layer["q_norm"] = norm(f"{p}.self_attn.q_norm.weight", 128)
                layer["k_norm"] = norm(f"{p}.self_attn.k_norm.weight", 128)
                for name, out_width, in_width in (
                    ("q_proj", 4096, 5120), ("k_proj", 1024, 5120),
                    ("v_proj", 1024, 5120), ("o_proj", 5120, 4096),
                ):
                    layer[name] = linear(f"{p}.self_attn.{name}.weight", out_width, in_width)
                for name, out_width, in_width, dtype in (
                    ("gate_proj", 17408, 5120, None), ("up_proj", 17408, 5120, None),
                    ("down_proj", 5120, 17408, down_dtype),
                ):
                    layer[name] = linear(f"{p}.mlp.{name}.weight", out_width, in_width, dtype)
                for name in ("attention_conv", "mlp_conv"):
                    layer[name] = NativeDynamicConv(
                        device, raw(f"{p}.{name}.base_kernel", (2, 2, 5120)),
                        raw(f"{p}.{name}.kernel_projection.weight", (1280, 5120)),
                        f"{prefix}/{p}.{name}",
                    )
            self.selector = NativeCandidateSelector(
                device,
                raw("candidate_selector.predecessor_codebook", (248320, 256)),
                raw("candidate_selector.successor_codebook", (248320, 256)),
                raw("candidate_selector.hidden_projection.weight", (256, 5120)),
                f"{prefix}/candidate_selector", top_k=16, draft_vocab=draft_vocab,
            )
            # Constant RoPE tables are initialization, not host inference. Build
            # FP32 frequencies then cast cos/sin to BF16 exactly as the reference.
            inverse = 1.0 / (10000000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
            positions = torch.arange(262144 + 32, dtype=torch.float32)
            angles = positions[:, None] * inverse[None, :]
            angles = torch.cat((angles, angles), dim=-1)
            self.cos_table = self._upload(angles.cos().to(torch.bfloat16), row_major=True)
            self.sin_table = self._upload(angles.sin().to(torch.bfloat16), row_major=True)
            self.cache = CommittedKV(device, block_size)
        except Exception:
            self.release()
            raise

    def _upload(self, tensor, *, row_major=False, dtype=ttnn.bfloat16):
        result = ttnn.from_torch(tensor, device=self.device, dtype=dtype,
                                 layout=ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT,
                                 memory_config=DRAM)
        self._owned.append(result)
        return result

    def _check_live(self):
        if self._released:
            raise RuntimeError("NativeDFlash has been released")

    def _hidden(self, tensor, rows, width=5120):
        if (tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT
                or int(tensor.shape[-1]) != width):
            raise ValueError(f"DFlash requires tiled BF16 hidden width {width}")
        logical = tuple(int(dim) for dim in tensor.shape)
        if logical not in ((1, rows, width), (1, 1, rows, width)):
            raise ValueError(f"DFlash expected one sequence with {rows} rows, got {logical}")
        tensor = ttnn.reshape(tensor, (1, 1, rows, width))
        return ttnn.to_memory_config(tensor, DRAM)

    def _linear(self, hidden, weight, output=None):
        """Draft weights: whole-grid 1D multicast with deep K blocks, HiFi2/FP32 accumulation."""
        depth, width = int(weight.shape[-2]) // 32, int(weight.shape[-1]) // 32
        config = self._matmul_configs.get((depth, width))
        if config is None:
            grid = self.device.compute_with_storage_grid_size()
            per_core_n = -(-width // (grid.x * grid.y))
            config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=max(d for d in range(1, 9) if depth % d == 0),
                out_subblock_h=1,
                out_subblock_w=max(d for d in range(1, 5) if per_core_n % d == 0),
                per_core_M=1, per_core_N=per_core_n,
                fuse_batch=True, fused_activation=None, mcast_in0=True,
            )
            self._matmul_configs[(depth, width)] = config
        compute = self.linear_compute
        if weight.dtype == ttnn.bfloat4_b and (depth, width) == (544, 160):
            compute = self.down_bfp4_compute
        return ttnn.linear(hidden, weight, dtype=ttnn.bfloat16, memory_config=DRAM,
                           program_config=config, compute_kernel_config=compute,
                           optional_output_tensor=output)

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

    def _norm_tile_rows(self, hidden, weight):
        """_norm one 32-row tile row at a time: the bounded 5120-wide norm picks its sharding by
        row count, and only the 32-row (draft append) geometry reproduces the draft's bits."""
        rows = int(hidden.shape[-2])
        if rows == 32:
            return self._norm(hidden, weight)
        parts = []
        for start in range(0, rows, 32):
            part = ttnn.slice(hidden, (0, 0, start, 0), (1, 1, start + 32, int(hidden.shape[-1])), memory_config=DRAM)
            parts.append(self._norm(part, weight))
            ttnn.deallocate(part)
        output = ttnn.concat(parts, dim=2, memory_config=DRAM)
        for part in parts:
            ttnn.deallocate(part)
        return output

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

    def _heads(self, *projections):
        """[1,1,B,H*128] projections -> [1,H,B,128] heads (same tiles as nlp_create_qkv_heads)."""
        rows = int(projections[0].shape[-2])
        return retile_heads(self.device, [(p, (1, int(p.shape[-1]) // 128, rows, 128)) for p in projections], rows)

    def _context_kv(self, layer, hidden, tables):
        key = self._linear(hidden, layer["k_proj"])
        value = self._linear(hidden, layer["v_proj"])
        key, value = self._heads(key, value)
        return self._rope(self._norm(key, layer["k_norm"]), tables), value

    def _project_rows(self, features, weight):
        """One matmul over every prompt row (weights read once, not once per 32 rows).

        Same K blocking (in0_block_w), fidelity and packer accumulation as _linear, so each
        row equals the per-32-row projection bit for bit.
        """
        rows = int(features.shape[-2])
        depth, width = int(weight.shape[-2]) // 32, int(weight.shape[-1]) // 32
        m_tiles = rows // 32
        grid_x, grid_y = 10, 8
        per_core_m = -(-m_tiles // grid_y)
        per_core_n = -(-width // grid_x)
        out_block_w = next(w for w in (8, 4, 2, 1) if per_core_n % w == 0)
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=max(d for d in range(1, 9) if depth % d == 0),
            out_subblock_h=1,
            out_subblock_w=4 if out_block_w % 4 == 0 else out_block_w,
            out_block_h=per_core_m if per_core_m <= 7 else 1,
            out_block_w=out_block_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
        )
        return ttnn.linear(features, weight, dtype=ttnn.bfloat16, memory_config=DRAM,
                           program_config=config, compute_kernel_config=self.linear_compute)

    def _commit_features(self, features, rows, start, count):
        target_hidden = self._norm(self._linear(features, self.fc), self.hidden_norm)
        tables = self._rope_tables(self.cache.feature_positions, rows)
        for index, layer in enumerate(self.layers):
            key, value = self._context_kv(layer, target_hidden, tables)
            self.cache.commit(index, key, value, start, count)

    def prefill(self, features, start_position, valid_length, source_offset=0):
        """Append prompt feature rows [source_offset, valid_length) at start_position+source_offset.

        Every row of the segment goes through one fc projection, one hidden norm and, per layer,
        one K/V projection, K norm and RoPE; the ring append then copies whole tiles (prefill row
        starts are tile-aligned). Row for row this is the arithmetic of the draft's 32-row append
        (same K blocking, fidelity and accumulation; norms and RoPE are row-local)."""
        self._check_live()
        if not all(isinstance(value, int) for value in (start_position, valid_length, source_offset)):
            raise ValueError("DFlash prefill position, length, and source offset must be host integers")
        length = int(features.shape[-2])
        features = self._hidden(features, length, 25600)
        if (
            start_position < 0
            or not 0 <= source_offset <= valid_length <= length
            or start_position + valid_length > 262144
        ):
            raise ValueError("DFlash prefill valid rows exceed the input or position limit")
        if start_position % 32 or source_offset % 32 or length % 32:
            raise ValueError("DFlash prefill segments must start on a tile row")
        if source_offset == valid_length:
            return
        positions = torch.zeros((1, length), dtype=torch.int32)
        positions[0, :valid_length] = torch.arange(start_position, start_position + valid_length, dtype=torch.int32)
        indices = ttnn.from_torch(positions, device=self.device, dtype=ttnn.uint32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        tables = []
        for table in (self.cos_table, self.sin_table):
            embedded = ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
            tables.append(ttnn.reshape(embedded, (1, 1, length, 128)))
        ttnn.deallocate(indices)
        projected = self._project_rows(features, self.fc)
        target_hidden = self._norm_tile_rows(projected, self.hidden_norm)
        ttnn.deallocate(projected)
        start_tt = self.cache.host_scalar(start_position + source_offset)
        count_tt = self.cache.host_scalar(valid_length - source_offset)
        offset_tt = self.cache.host_scalar(source_offset)
        self.cache.prepare(start_tt, start_tt, start_tt, count_tt, prefill=True)
        heads = int(self.layers[0]["k_proj"].shape[-1]) // 128
        for index, layer in enumerate(self.layers):
            key, value = (
                ttnn.permute(ttnn.reshape(self._project_rows(target_hidden, layer[name]), (1, length, heads, 128)),
                             (0, 2, 1, 3), memory_config=DRAM)
                for name in ("k_proj", "v_proj")
            )
            key = self._rope(self._norm(key, layer["k_norm"]), tables)
            self.cache.commit_rows(index, key, value, start_tt, count_tt, offset_tt)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
        ttnn.deallocate(target_hidden)
        for tensor in (*tables, start_tt, count_tt, offset_tt):
            ttnn.deallocate(tensor)

    def reset(self):
        self._check_live()
        self.cache.reset()

    def draft(self, anchor_ids, position, features, feature_start, feature_count, sampler=None,
              sample_position=None, processed=False):
        """Commit device-counted features and propose B-1 tokens, entirely native. With `sampler` (exact speculative
        sampling) the selector walk draws each proposal from its candidate distribution instead of taking the
        argmax; the distributions stay on device for verification. `sample_position` is the anchor's position."""
        proposed_hidden, logits = self._draft_hidden(anchor_ids, position, features, feature_start, feature_count)
        return self.selector.select(proposed_hidden, logits, anchor_ids, sampler=sampler,
                                    sample_position=sample_position, processed=processed)

    def draft_branches(self, anchor_ids, position, features, feature_start, feature_count, fork_position):
        """Propose A and a learned rank-two fork B with a single draft backbone."""
        self._check_live()
        if self.block_size != 8 or type(fork_position) is not int or not 1 <= fork_position <= 7:
            raise ValueError("DFlash draft_branches requires B8 and a host fork_position in 1..7")
        proposed_hidden, logits = self._draft_hidden(anchor_ids, position, features, feature_start, feature_count)
        return self.selector.select_branches(proposed_hidden, logits, anchor_ids, fork_position)

    def _draft_hidden(self, anchor_ids, position, features, feature_start, feature_count):
        self._check_live()
        features = self._hidden(features, self.block_size, 25600)
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
                self._linear(branch, layer["v_proj"]),
            )
            q = self._rope(self._norm(q, layer["q_norm"]), tables)
            k = self._rope(self._norm(k, layer["k_norm"]), tables)
            key, value = self.cache.attention_inputs(index, k, v)
            attended = ttnn.transformer.scaled_dot_product_attention(
                q, key, value, attn_mask=self.cache.mask, is_causal=False,
                scale=128 ** -0.5, memory_config=DRAM, program_config=self.sdpa_config,
                compute_kernel_config=self.compute,
            )
            (attended,) = retile_heads(self.device, [(attended, (1, 1, self.block_size, 4096))], self.block_size)
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
        logits = self._hidden(self.output_head(proposed_hidden), self.block_size - 1, self.selector.logits_width)
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
