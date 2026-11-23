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
each supported runtime configuration. Single-device generic or dtype-gated
mixed packed execution with optional one-token MTP can be verified; vision
and tensor parallelism are excluded. MTP source tensors are stored losslessly.

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
FORMAT = "qwen35-9b-ttnn-native-text-v1"
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
        raise ValueError("Equivalence requires explicit QWEN36_MTP=0 or QWEN36_MTP=1")
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
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Artifact path escapes checkpoint: {name}")
    return path


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
        raise ValueError("Checkpoint requires complete lossless MTP-1 metadata")
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
    return manifest


def verify_files(root, manifest):
    for name, info in manifest["files"].items():
        path = local_file(root, name)
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Checkpoint file hash/size mismatch: {name}")


def require_equivalence(root, manifest):
    proof = json.loads((Path(root) / equivalence_filename(os.environ)).read_text())
    if (proof.get("manifest_sha256") != digest(Path(root) / MANIFEST)
            or proof.get("exact_logits_equal") is not True or proof.get("tokens_compared", 0) < 1):
        raise ValueError("Native checkpoint lacks matching exact end-to-end equivalence evidence")
    if proof.get("precision") != manifest["precision"]:
        raise ValueError("Equivalence evidence precision mismatch")
    environment = proof.get("runtime_environment", {})
    if equivalence_filename(environment) != equivalence_filename(os.environ):
        raise ValueError("Equivalence evidence MTP mode mismatch")
    return proof


def validate_runtime(root, runtime_root, num_devices):
    """Reject using equivalence evidence under a different execution contract."""
    if num_devices != 1:
        raise ValueError("Native checkpoint equivalence covers one device only")
    proof = require_equivalence(root, read_manifest(root))
    for name, value in proof["runtime_environment"].items():
        if os.environ.get(name) != value:
            raise ValueError(f"Runtime environment differs from native equivalence: {name}")
    for name, value in proof["runtime_sources"].items():
        if digest(local_file(runtime_root, name)) != value:
            raise ValueError(f"Runtime source differs from native equivalence: {name}")


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

    Small/nonlinear tensors (including original FP32 A_log/dt_bias) retain
    original values/dtypes. Embeddings and all MTP tensors are stored losslessly. This loader
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
        if name in MTP_KEYS and tensor_hash(value) != entry["source_tensor_sha256"]:
            raise ValueError(f"Restored MTP tensor differs from original source hash: {name}")
        result[name] = value
    if not {"tok_embeddings.weight", "output.weight", "norm.weight"} <= result.keys():
        raise ValueError("Native checkpoint is missing required top-level tensors")
    return result


def record_equivalence(root, baseline, restored):
    """Compare actual full-vocabulary runner artifacts, not weight-error proxies.

    Baseline must evaluate ORIGINAL weights at the SAME chosen precision and
    runtime, not baseline-bf16 against a mixed candidate. Exact logits equality
    is deliberately strict. Evidence covers the supplied records only.
    The destination is selected by the recorded MTP runtime environment, never
    by the environment of this proof-writing process. MTP proof does not replace
    a separate live speculative-cycle verification.
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
    if left["source_identity"].get("native_manifest_sha256") is not None:
        raise ValueError("Baseline must use the original HF checkpoint")
    if left["source_identity"].get("index_sha256") != manifest["source"]["original_index_sha256"]:
        raise ValueError("Baseline original checkpoint index differs from exported source")
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
