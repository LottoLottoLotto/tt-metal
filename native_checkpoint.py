#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""TT-native text-only checkpoint contract and lossless host reconstruction.

Native matrices are TTNN TILE dumps in [input, output] orientation. Loading
returns remapped torch [output, input] BF16 tensors; the runtime requantizes
them. This is NOT direct device-cache loading, GGUF, QAT, or an official
Unsloth quant. Exact per-matrix idempotence is necessary but insufficient:
GDN AB/mega tensors concatenate quant-rounded components, and packed/fused
layouts can regroup block exponents. End-to-end equivalence is required for
each supported runtime configuration. This format supports single-device
Qwen3.8-27B text execution with optional MTP-1; vision and tensor parallelism
are excluded. All original MTP tensors are stored losslessly.

TTNN host conversion may initialize driver metadata: even host-only commands
need exclusive device ownership in this environment. No original HF checkpoint
is used by this loader. The installed compatible TT runtime is still required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

MANIFEST = "native_manifest.json"
FORMAT = "qwen38-27b-ttnn-native-text-v1"
MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DTYPES = {"bf16": "bfloat16", "bfp8": "bfloat8_b", "bfp4": "bfloat4_b"}
MTP_KEYS = frozenset({
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
})


def validate_config(config):
    """Validate the pinned official architecture, not the inherited 9B defaults."""
    text = config.get("text_config", {})
    expected = {
        "model_type": "qwen3_5_text", "num_hidden_layers": 64,
        "hidden_size": 5120, "intermediate_size": 17408, "vocab_size": 248320,
        "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
        "tie_word_embeddings": False, "attention_bias": False,
        "attn_output_gate": True, "output_gate_type": "swish", "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "dtype": "bfloat16",
    }
    if config.get("model_type") != "qwen3_5" or config.get("quantization_config"):
        raise ValueError(f"Expected original unquantized {MODEL_ID} configuration")
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f"Unsupported {MODEL_ID} config {key}: {text.get(key)!r}, expected {value!r}")
    layer_types = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]
    if text.get("layer_types") != layer_types:
        raise ValueError("Unexpected 64-layer attention schedule")
    rope = text.get("rope_parameters", {})
    if (rope.get("rope_type") != "default" or rope.get("rope_theta") != 10000000
            or rope.get("partial_rotary_factor") != 0.25
            or rope.get("mrope_section") != [11, 11, 10] or rope.get("mrope_interleaved") is not True):
        raise ValueError("Unexpected pinned RoPE configuration")
    return text


def source_tensor_shapes(config):
    """Complete text/MTP inventory derived from the validated official config."""
    text = validate_config(config)
    hidden, intermediate, vocab = text["hidden_size"], text["intermediate_size"], text["vocab_size"]
    head = text["head_dim"]
    query, kv = text["num_attention_heads"] * head, text["num_key_value_heads"] * head
    key = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    shapes = {"lm_head.weight": [vocab, hidden],
              "model.language_model.embed_tokens.weight": [vocab, hidden],
              "model.language_model.norm.weight": [hidden]}

    def layer(prefix, kind):
        shapes.update({prefix + "input_layernorm.weight": [hidden],
                       prefix + "post_attention_layernorm.weight": [hidden],
                       prefix + "mlp.gate_proj.weight": [intermediate, hidden],
                       prefix + "mlp.up_proj.weight": [intermediate, hidden],
                       prefix + "mlp.down_proj.weight": [hidden, intermediate]})
        if kind == "full_attention":
            shapes.update({prefix + "self_attn.q_proj.weight": [2 * query, hidden],
                           prefix + "self_attn.k_proj.weight": [kv, hidden],
                           prefix + "self_attn.v_proj.weight": [kv, hidden],
                           prefix + "self_attn.o_proj.weight": [hidden, query],
                           prefix + "self_attn.q_norm.weight": [head],
                           prefix + "self_attn.k_norm.weight": [head]})
        else:
            prefix += "linear_attn."
            shapes.update({prefix + "in_proj_qkv.weight": [2 * key + value, hidden],
                           prefix + "in_proj_z.weight": [value, hidden],
                           prefix + "in_proj_a.weight": [text["linear_num_value_heads"], hidden],
                           prefix + "in_proj_b.weight": [text["linear_num_value_heads"], hidden],
                           prefix + "out_proj.weight": [hidden, value],
                           prefix + "conv1d.weight": [2 * key + value, 1, text["linear_conv_kernel_dim"]],
                           prefix + "A_log": [text["linear_num_value_heads"]],
                           prefix + "dt_bias": [text["linear_num_value_heads"]],
                           prefix + "norm.weight": [text["linear_value_head_dim"]]})

    for index, kind in enumerate(text["layer_types"]):
        layer(f"model.language_model.layers.{index}.", kind)
    layer("mtp.layers.0.", "full_attention")
    shapes.update({"mtp.fc.weight": [hidden, 2 * hidden], "mtp.norm.weight": [hidden],
                   "mtp.pre_fc_norm_embedding.weight": [hidden], "mtp.pre_fc_norm_hidden.weight": [hidden]})
    validate_mtp_keys(shapes)
    return shapes


def remapped_tensor_shapes(config):
    shapes = {}
    text = validate_config(config)
    key = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    for name, shape in source_tensor_shapes(config).items():
        name = name.removeprefix("model.language_model.")
        name = {"lm_head.weight": "output.weight", "embed_tokens.weight": "tok_embeddings.weight"}.get(name, name)
        if name.endswith(".linear_attn.conv1d.weight"):
            prefix = name.removesuffix("conv1d.weight")
            for stream, width in (("q", key), ("k", key), ("v", shape[0] - 2 * key)):
                shapes[prefix + stream + "_conv.weight"] = [width, *shape[1:]]
        else:
            shapes[name.replace(".linear_attn.in_proj_qkv.weight", ".linear_attn.qkv_proj.weight")] = shape
    return shapes


def validate_source_index(index, config):
    expected = source_tensor_shapes(config)
    actual = {name for name in index if not name.startswith("model.visual.")}
    if actual != set(expected):
        raise ValueError(f"Source tensor inventory mismatch: missing={sorted(set(expected)-actual)}, unexpected={sorted(actual-set(expected))}")
    return expected


def revision_evidence(root, names):
    evidence = {}
    for name in names:
        sidecar = local_file(root, ".cache/huggingface/download/" + name + ".metadata")
        observed = sidecar.read_text().splitlines()[0]
        if observed != REVISION:
            raise ValueError(f"Source revision mismatch for {name}: {observed}")
        evidence[name] = observed
    return evidence


def tensor_hash(tensor):
    import torch

    raw = tensor.contiguous().view(torch.uint8).reshape(-1).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def validate_mtp_keys(keys):
    actual = {name for name in keys if name.startswith("mtp.")}
    if actual != MTP_KEYS:
        raise ValueError(f"Incomplete/unsupported one-layer MTP subtree: missing={sorted(MTP_KEYS - actual)}, unexpected={sorted(actual - MTP_KEYS)}")


def equivalence_filename(environment):
    mode = environment.get("QWEN36_MTP")
    if mode not in ("0", "1"):
        raise ValueError("Native execution requires explicit QWEN36_MTP=0 or 1")
    return "equivalence-mtp.json" if mode == "1" else "equivalence.json"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def local_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path == root:
        raise ValueError(f"Artifact path escapes checkpoint: {name}")
    if path.is_relative_to(root):
        return path
    if hf_cache_blob(root, name, path):
        # The snapshot entry itself, not its blob: loaders key on the file name (e.g. ttnn's .tensorbin check).
        return Path(os.path.normpath(root / name))
    raise ValueError(f"Artifact path escapes checkpoint: {name}")


def hf_cache_blob(root, name, path):
    """A Hugging Face cache snapshot (<cache>/models--<org>--<name>/snapshots/<revision>/...) stores every file as a
    symlink into the cache's blob store (the repo's blobs/, which may link on into a cache-wide store); accept a
    target inside that cache, for a name that stays inside the checkpoint lexically."""
    if not Path(os.path.normpath(root / name)).is_relative_to(root):
        return False
    for parent in (root, *root.parents):
        if parent.parent.name == "snapshots" and parent.parent.parent.name.startswith("models--"):
            return path.is_relative_to(parent.parent.parent.parent)
    return False


def read_manifest(root):
    manifest = json.loads((Path(root) / MANIFEST).read_text())
    if manifest.get("format") != FORMAT or manifest.get("scope") != "text-only-no-vision-with-mtp":
        raise ValueError("Unsupported native checkpoint format/scope")
    if not manifest.get("tensors") or not manifest.get("roundtrip", {}).get("all_passed"):
        raise ValueError("Incomplete native checkpoint or failed tensor roundtrip")
    mtp = manifest.get("mtp", {})
    if (mtp.get("enabled") is not True or type(mtp.get("num_speculative_tokens")) is not int
            or mtp["num_speculative_tokens"] != 1 or type(mtp.get("source_tensor_count")) is not int
            or mtp["source_tensor_count"] != len(MTP_KEYS) or mtp.get("storage") != "lossless"):
        raise ValueError("Checkpoint requires complete lossless MTP storage for MTP-1")
    validate_mtp_keys(manifest["tensors"])
    for name in MTP_KEYS:
        entry = manifest["tensors"][name]
        source_hash = entry.get("source_tensor_sha256")
        file_info = manifest.get("files", {}).get(entry.get("file"), {})
        if (entry.get("storage") != "safetensors-lossless" or entry.get("source_name") != name
                or not entry.get("source_shape") or entry["source_shape"] != entry.get("shape")
                or not entry.get("source_torch_dtype")
                or entry["source_torch_dtype"] != entry.get("restored_torch_dtype")
                or not isinstance(source_hash, str) or len(source_hash) != 64
                or any(c not in "0123456789abcdef" for c in source_hash)
                or source_hash != entry.get("tensor_sha256")
                or not file_info.get("sha256") or entry.get("sha256") != file_info["sha256"]
                or entry.get("roundtrip", {}).get("exact_values") is not True
                or entry.get("roundtrip", {}).get("exact_dtype") is not True):
            raise ValueError(f"MTP tensor is not bound to lossless original values/dtype: {name}")
    if (manifest.get("source", {}).get("model_id") != MODEL_ID
            or manifest["source"].get("declared_revision") != REVISION):
        raise ValueError("Native checkpoint model/revision mismatch")
    config = json.loads(local_file(root, "config.json").read_text())
    expected = remapped_tensor_shapes(config)
    if set(manifest["tensors"]) != set(expected):
        raise ValueError("Native checkpoint has incomplete/unexpected text/MTP tensor inventory")
    sources = source_tensor_shapes(config)
    for name, entry in manifest["tensors"].items():
        file_info = manifest.get("files", {}).get(entry.get("file"), {})
        if (entry.get("shape") != expected[name]
                or entry.get("source_shape") != sources.get(entry.get("source_name"))
                or entry.get("source_torch_dtype") not in ("torch.bfloat16", "torch.float32")
                or entry.get("sha256") != file_info.get("sha256")):
            raise ValueError(f"Native tensor metadata disagrees with source architecture/inventory: {name}")
        precision = None if name.startswith("mtp.") else tensor_precision(name, manifest["precision"])
        if precision is None:
            if (entry.get("storage") != "safetensors-lossless"
                    or entry.get("restored_torch_dtype") != entry.get("source_torch_dtype")
                    or entry.get("roundtrip") != {"exact_values": True, "exact_dtype": True}):
                raise ValueError(f"Nonmatrix/MTP tensor must preserve source dtype and values: {name}")
        elif (entry.get("storage") != "ttnn-tile" or entry.get("precision") != precision
                or entry.get("native_shape") != list(reversed(expected[name]))
                or entry.get("native_dtype") != DTYPES[precision]
                or entry.get("layout") != "TILE_LAYOUT" or entry.get("orientation") != "input,output"
                or entry.get("roundtrip") != {"exact_values": True, "exact_serialized_bytes": True}):
            raise ValueError(f"Matrix lacks matching native precision/roundtrip contract: {name}")
    return manifest


def verify_files(root, manifest):
    for name, info in manifest["files"].items():
        path = local_file(root, name)
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Checkpoint file hash/size mismatch: {name}")


# The release ships one checkpoint and one pair of reload proofs per runtime image (checkpoint/ for the 8K / 128K
# image, proofs-262k/ for the 262K image); its launcher mounts the image's pair over the checkpoint's.  This variable
# names the directory holding the pair recorded on the running runtime instead; unset, the checkpoint's own.  The
# proof is still bound to the checkpoint's manifest, precision and MTP mode, and validate_runtime still compares its
# environment and every source it names.
PROOFS_ENV = "QWEN36_NATIVE_PROOFS"


def require_equivalence(root, manifest):
    proofs = os.environ.get(PROOFS_ENV) or root
    proof = json.loads((Path(proofs) / equivalence_filename(os.environ)).read_text())
    if (proof.get("manifest_sha256") != digest(Path(root) / MANIFEST)
            or proof.get("exact_logits_equal") is not True or proof.get("tokens_compared", 0) < 1):
        raise ValueError("Native checkpoint lacks matching exact end-to-end equivalence evidence")
    if proof.get("precision") != manifest["precision"]:
        raise ValueError("Equivalence evidence precision mismatch")
    environment = proof.get("runtime_environment", {})
    if equivalence_filename(environment) != equivalence_filename(os.environ):
        raise ValueError("Equivalence evidence MTP mode mismatch")
    return proof


def runtime_support_identity(runtime_root):
    """Bind installed overlay sources and compiled TTNN libraries to evidence."""
    metal_root = Path(runtime_root).resolve().parents[3]
    sources = json.loads((metal_root / "qwen-runtime-support.json").read_text())
    for name, expected in sources.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Installed runtime support source differs: {name}")
    binaries = {
        str(metal_root / name): digest(metal_root / name)
        for name in ("build/lib/_ttnncpp.so", "build/lib/_ttnn.so", "ttnn/ttnn/_ttnn.so")
    }
    return {"sources": sources, "binaries": binaries}


# Source installs of this runtime (e.g. a tt-model container package) put the same files at other paths and compile
# the TTNN libraries from this tree, so the release image's absolute paths and binary digests cannot match.  This
# variable names a relocation file (qwen-runtime-relocation.json) that maps the release install prefixes to this
# install and lists every proof-bound file whose bytes legitimately differ; every other file is still compared.
RELOCATION_ENV = "QWEN36_RUNTIME_RELOCATION"


def runtime_relocation():
    path = os.environ.get(RELOCATION_ENV)
    return json.loads(Path(path).read_text()) if path else None


def relocated_path(name, relocation, metal_root):
    for prefix in sorted(relocation["prefixes"], key=len, reverse=True):
        if name.startswith(prefix):
            target = relocation["prefixes"][prefix]
            if target.startswith("module:"):
                import importlib.util

                spec = importlib.util.find_spec(target[len("module:"):])
                if spec is None or not spec.submodule_search_locations:
                    raise ValueError(f"Relocation target package is not installed: {target}")
                base = Path(next(iter(spec.submodule_search_locations)))
            elif target == "metal:":
                base = metal_root
            else:
                raise ValueError(f"Unsupported relocation target: {target}")
            return base / name[len(prefix):]
    raise ValueError(f"No relocation prefix covers the runtime file: {name}")


def relocated_digest(name, recorded, relocation):
    """The digest the relocated file must have; None for a library compiled from this tree (not comparable)."""
    entry = relocation["files"].get(name)
    if entry is None:
        return recorded
    if entry.get("release_sha256") != recorded:
        raise ValueError(f"Relocation entry does not describe the proof-bound file: {name}")
    return entry.get("sha256")


def validate_relocated_runtime(runtime_root, proof, relocation):
    """validate_runtime's source and library checks for a relocated source install."""
    metal_root = Path(runtime_root).resolve().parents[3]
    support = proof.get("runtime_support") or {}
    if json.loads((metal_root / "qwen-runtime-support.json").read_text()) != support.get("sources"):
        raise ValueError("Installed runtime support inventory differs from native equivalence")
    for name, value in proof["runtime_sources"].items():
        expected = relocated_digest(relocation["runtime_root"] + name, value, relocation)
        if expected is not None and digest(local_file(runtime_root, name)) != expected:
            raise ValueError(f"Runtime source differs from native equivalence: {name}")
    rebuilt = []
    for name, value in {**support.get("sources", {}), **support.get("binaries", {})}.items():
        expected = relocated_digest(name, value, relocation)
        if expected is None:
            rebuilt.append(name)
        elif digest(relocated_path(name, relocation, metal_root)) != expected:
            raise ValueError(f"Installed runtime support source differs: {name}")
    from loguru import logger

    logger.info("Native equivalence: relocated runtime verified ({} files by digest; compiled from source, not "
                "digest-compared: {})", len(proof["runtime_sources"]) + len(support.get("sources", {}))
                + len(support.get("binaries", {})) - len(rebuilt), ", ".join(sorted(rebuilt)))


def validate_runtime(root, runtime_root, num_devices):
    """Reject using equivalence evidence under a different execution contract."""
    if num_devices != 1:
        raise ValueError("Native checkpoint equivalence covers one device only")
    proof = require_equivalence(root, read_manifest(root))
    for name, value in proof["runtime_environment"].items():
        current = os.environ.get(name)
        if name == "QWEN36_MAX_TOKENS_ALL_USERS" and current is not None and current.isdigit() and value.isdigit():
            # Paged-KV pool capacity only: a larger single-stream pool serves longer contexts with the
            # same per-position arithmetic; never smaller than the evaluated pool.
            if int(current) < int(value):
                raise ValueError(f"Runtime environment differs from native equivalence: {name}")
            continue
        if name == "QWEN36_WEIGHT_PREFETCH" and current == "none":
            # DRAM-core weight prefetch streams the same K blocks into L1 (bitwise-equal outputs);
            # native-context serving may disable it to return its DRAM weight copies to the KV pool.
            continue
        if current != value:
            raise ValueError(f"Runtime environment differs from native equivalence: {name}")
    relocation = runtime_relocation()
    if relocation is not None:
        validate_relocated_runtime(runtime_root, proof, relocation)
        return
    for name, value in proof["runtime_sources"].items():
        if digest(local_file(runtime_root, name)) != value:
            raise ValueError(f"Runtime source differs from native equivalence: {name}")
    if runtime_support_identity(runtime_root) != proof.get("runtime_support"):
        raise ValueError("Runtime overlay or compiled libraries differ from native equivalence")


def matrix_family(name):
    if name == "output.weight":
        return "lm_head"
    if not name.startswith("layers.") or not name.endswith(".weight"):
        return None
    suffix = ".".join(name.split(".")[2:])
    if suffix in {f"self_attn.{p}_proj.weight" for p in ("q", "k", "v", "o")}:
        return "attention"
    if suffix == "linear_attn.out_proj.weight":
        return "gdn_output"
    if suffix in {f"linear_attn.{p}.weight" for p in ("qkv_proj", "in_proj_a", "in_proj_b", "in_proj_z")}:
        return "gdn"
    if suffix in {"mlp.gate_proj.weight", "mlp.up_proj.weight"}:
        return "mlp_gate_up"
    if suffix == "mlp.down_proj.weight":
        return "mlp_down"
    return None


def tensor_precision(name, plan):
    family = matrix_family(name)
    if family is None:
        return None
    if family == "lm_head":
        return plan[family]
    values = plan["layers"][name.split(".")[1]]
    return values.get("gdn_output", values["gdn"]) if family == "gdn_output" else values[family]


def load_native_checkpoint(root, verify_hashes=True, allow_unverified=False):
    """Reconstruct text and MTP tensors; allow_unverified is for equivalence runs only.

    Small/nonlinear tensors retain original values/dtypes. Embeddings and all
    MTP tensors are stored losslessly. This loader
    materializes one remapped model, never an HF model or an original copy.
    """
    import torch
    import ttnn
    from safetensors import safe_open

    root = Path(root).resolve()
    manifest = read_manifest(root)
    if verify_hashes:
        verify_files(root, manifest)
    if not allow_unverified:
        require_equivalence(root, manifest)
    result = {}
    for name, entry in manifest["tensors"].items():
        path = local_file(root, entry["file"])
        if entry["storage"] == "ttnn-tile":
            if entry["precision"] != tensor_precision(name, manifest["precision"]):
                raise ValueError(f"Tensor precision disagrees with checkpoint plan: {name}")
            tensor = ttnn.load_tensor(str(path))
            if list(tensor.shape) != entry["native_shape"] or tensor.dtype != getattr(ttnn, DTYPES[entry["precision"]]):
                raise ValueError(f"Native tensor shape/dtype mismatch: {name}")
            if tensor.layout != ttnn.TILE_LAYOUT:
                raise ValueError(f"Native tensor layout mismatch: {name}")
            value = ttnn.to_torch(tensor).to(torch.bfloat16).T.contiguous()
            del tensor
        elif entry["storage"] == "safetensors-lossless":
            with safe_open(str(path), framework="pt", device="cpu") as source:
                value = source.get_tensor(name)
        else:
            raise ValueError(f"Unsupported tensor storage: {name}")
        if list(value.shape) != entry["shape"] or str(value.dtype) != entry["restored_torch_dtype"]:
            raise ValueError(f"Restored tensor shape/dtype mismatch: {name}")
        if entry["storage"] == "safetensors-lossless" and tensor_hash(value) != entry["tensor_sha256"]:
            raise ValueError(f"Restored lossless tensor differs from exported values: {name}")
        result[name] = value
    if not {"tok_embeddings.weight", "output.weight", "norm.weight"} <= result.keys():
        raise ValueError("Native checkpoint is missing required top-level tensors")
    return result


def record_equivalence(root, baseline, restored):
    """Compare actual full-vocabulary runner artifacts, not weight-error proxies.

    Baseline must evaluate ORIGINAL weights at the SAME chosen precision and
    runtime, not baseline-bf16 against a mixed candidate. Exact logits equality
    is deliberately strict. Evidence covers the supplied records only.
    The proof is bound to the recorded execution environment and MTP mode,
    never the environment of this proof-writing process.
    """
    import numpy as np

    root, baseline, restored = map(Path, (root, baseline, restored))
    manifest = read_manifest(root)
    verify_files(root, manifest)
    left = json.loads((baseline / "metadata.json").read_text())
    right = json.loads((restored / "metadata.json").read_text())
    for metadata in (left, right):
        if metadata.get("status") != "complete" or metadata.get("backend") != "ttnn-native":
            raise ValueError("Both evaluations must be completed TTNN runs")
        if metadata.get("precision") != manifest["precision"] or not metadata.get("full_vocabulary"):
            raise ValueError("Both evaluations must use the exported precision and full logits")
    for key in ("precision", "selection", "input_sha256", "runtime_environment", "alignment", "max_seq_len"):
        if left.get(key) != right.get(key):
            raise ValueError(f"End-to-end comparison configuration mismatch: {key}")
    proof_filename = equivalence_filename(left.get("runtime_environment", {}))
    if left["source_identity"]["sources"] != right["source_identity"]["sources"]:
        raise ValueError("Runtime source hashes differ between equivalence runs")
    if right["source_identity"].get("native_manifest_sha256") != digest(root / MANIFEST):
        raise ValueError("Restored evaluation is not bound to this native manifest")
    if left["source_identity"].get("runtime_support") != right["source_identity"].get("runtime_support"):
        raise ValueError("Runtime overlay or compiled libraries differ between equivalence runs")
    if not left["source_identity"].get("runtime_support"):
        raise ValueError("Missing runtime overlay and compiled library identity")
    if left["source_identity"].get("native_manifest_sha256") is not None:
        raise ValueError("Baseline must use the original HF checkpoint")
    if left["source_identity"].get("index_sha256") != manifest["source"]["original_index_sha256"]:
        raise ValueError("Baseline original checkpoint index differs from exported source")
    if left["source_identity"].get("shards") != manifest["source"]["shards"]:
        raise ValueError("Baseline source shard contents differ from exported source")
    if (left["runtime_environment"].get("QWEN36_OPTIMIZED_27B") != "1"
            or left["runtime_environment"].get("QWEN36_GENERIC_BASELINE") != "0"):
        raise ValueError("Equivalence must use the optimized 27B execution contract")
    for metadata in (left, right):
        if metadata["source_identity"].get("config_sha256") != manifest["files"]["config.json"]["sha256"]:
            raise ValueError("Evaluation config differs from native checkpoint")
    rows_a = [json.loads(line) for line in (baseline / "records.jsonl").read_text().splitlines() if line.strip()]
    rows_b = [json.loads(line) for line in (restored / "records.jsonl").read_text().splitlines() if line.strip()]
    if not rows_a or len(rows_a) != len(rows_b):
        raise ValueError("Evaluation record coverage differs or is empty")
    for metadata, rows in ((left, rows_a), (right, rows_b)):
        if metadata.get("completed_records") != len(rows) or [row["id"] for row in rows] != metadata["selection"]["record_ids"]:
            raise ValueError("Evaluation records do not cover the declared completed selection")
    evidence, tokens = [], 0
    for a, b in zip(rows_a, rows_b):
        for key in ("id", "split", "token_ids"):
            if a[key] != b[key]:
                raise ValueError(f"Evaluation record mismatch: {key}")
        file_a = local_file(baseline, a["logits_file"])
        file_b = local_file(restored, b["logits_file"])
        x, y = np.load(file_a, mmap_mode="r"), np.load(file_b, mmap_mode="r")
        expected = (len(a["token_ids"]) - 1, left["vocab_size"])
        if x.shape != expected or y.shape != expected:
            raise ValueError(f"Unexpected full-logits shape for record {a['id']}")
        for start in range(0, x.shape[0], 16):
            if not np.isfinite(x[start:start + 16]).all() or not np.array_equal(x[start:start + 16], y[start:start + 16]):
                raise ValueError(f"Native reload logits differ for record {a['id']} at chunk {start}")
        tokens += x.shape[0]
        evidence.append({"id": a["id"], "baseline_sha256": digest(file_a), "restored_sha256": digest(file_b)})
    proof = {"manifest_sha256": digest(root / MANIFEST), "exact_logits_equal": True,
             "tokens_compared": tokens, "records": evidence, "precision": manifest["precision"],
             "runtime_sources": left["source_identity"]["sources"],
             "runtime_support": left["source_identity"]["runtime_support"],
             "runtime_environment": left["runtime_environment"],
             "baseline_metadata_sha256": digest(baseline / "metadata.json"),
             "restored_metadata_sha256": digest(restored / "metadata.json"),
             "scope": "Exact parity only on recorded full-vocabulary teacher-forced sequences; not a quality certification"}
    save_json(root / proof_filename, proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--restored", type=Path)
    args = parser.parse_args()
    if bool(args.baseline) != bool(args.restored):
        parser.error("--baseline and --restored must be supplied together")
    if args.baseline:
        print(json.dumps(record_equivalence(args.checkpoint, args.baseline, args.restored), indent=2))
    else:
        manifest = read_manifest(args.checkpoint)
        verify_files(args.checkpoint, manifest)
        print(json.dumps({"files_verified": True, "tensors": len(manifest["tensors"]),
                          "equivalence": require_equivalence(args.checkpoint, manifest)}, indent=2))


if __name__ == "__main__":
    main()
