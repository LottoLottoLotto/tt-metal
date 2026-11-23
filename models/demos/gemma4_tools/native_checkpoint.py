#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""TT-native text-only checkpoint for google/gemma-4-12B-it on one P150 (MeshShape 1x1).

Layout of a checkpoint directory ``<root>`` (its basename must contain "12B" so the
runtime's per-model policy key resolves, e.g. ``.../gemma-4-12B-it``):

  native_manifest.json   format, source identity, precision plan, every file's sha256/bytes
  precision_plan.json    the Gemma4Precision plan the tensors were quantized with
  config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json
  small.safetensors      every 1-D text tensor (norm weights, layer scalars), lossless BF16
  tensors/tensor_cache_bf16/**.tensorbin
                         the exact TTNN host tensors (TILE bfp8/bfp4/bf16 matrices, row-major
                         bf16 embedding, norm tiles) that the runtime uploads to DRAM; produced by
                         the runtime's own ttnn.as_tensor cache writer from the ORIGINAL weights
  equivalence.json       written only by ``record`` from two real runs (never by hand)

Loading never touches the original HF safetensors: 2-D matrices are passed to the model
constructor as meta-device placeholders (shape only) and every ttnn.as_tensor call hits its
.tensorbin file; a missing/corrupt file makes the fallback conversion fail loudly on the meta
tensor. The loader refuses to serve without an equivalence proof that matches the manifest,
the runtime source tree and the TTNN binaries.

Subcommands (run inside the TT image):
  finalize <root> --source <hf dir> --fetch-verification <json>
      after the builder populated tensors/: write small.safetensors, copy config/tokenizer,
      hash everything into native_manifest.json.
  record <root> --baseline <teacher-run dir> --restored <teacher-run dir>
      compare the two runs' full logits (original weights quantized at load vs native reload,
      identical code/plan/input) and write equivalence.json iff bit-exact.
  verify <root>
      verify hashes + proof + runtime identity (what the loader does).

Speculative-decoding drafter (google/gemma-4-12B-it-assistant) directory ``<droot>``:
  drafter_manifest.json  source identity (fetch verification, config/safetensors sha256), draft weight
                         dtype, target checkpoint manifest, every file's sha256/bytes
  config.json, model.safetensors (hardlink of the sha-verified HF file; small tensors are read from it)
  tensors/assistant_tensor_cache_<dtype>/**.tensorbin   TTNN drafter tensors written by the runtime
  spec_equivalence.json  written only by ``spec-record``: greedy streams of a speculative run equal the
                         non-speculative streams of the same proven target runtime
  drafter-finalize <droot> --source <hf dir> --target <target root> --dtype bfp8
  spec-record <droot> --target <root> --reference <generate run> --speculative <generate run>
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

FORMAT = "gemma4-12b-it-ttnn-native-text-1x1-v1"
MODEL_ID = "google/gemma-4-12B-it"
MANIFEST = "native_manifest.json"
PROOF = "equivalence.json"
DRAFTER_FORMAT = "gemma4-12b-it-assistant-ttnn-drafter-1x1-v1"
DRAFTER_MANIFEST = "drafter_manifest.json"
SPEC_PROOF = "spec_equivalence.json"
META_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
TT_ROOT = Path("/home/container_app_user/tt-metal")
TREE = TT_ROOT / "models/demos/gemma4"
BINARIES = ("build/lib/_ttnncpp.so", "build/lib/libtt_metal.so", "ttnn/ttnn/_ttnn.so")
TEXT_PREFIX = "model.language_model."


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def validate_config(cfg):
    t = cfg["text_config"]
    expected = {
        "hidden_size": 3840,
        "num_hidden_layers": 48,
        "intermediate_size": 15360,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "vocab_size": 262144,
        "sliding_window": 1024,
        "attention_k_eq_v": True,
        "final_logit_softcapping": 30.0,
        "tie_word_embeddings": True,
        "enable_moe_block": False,
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
    }
    if cfg.get("model_type") != "gemma4_unified":
        raise ValueError("expected gemma4_unified config")
    for k, v in expected.items():
        if t.get(k) != v:
            raise ValueError(f"unsupported config {k}={t.get(k)!r} (expected {v!r})")
    return t


def text_shapes(cfg):
    """Every text tensor of the source checkpoint (name -> shape)."""
    t = validate_config(cfg)
    H, I, V = t["hidden_size"], t["intermediate_size"], t["vocab_size"]
    shapes = {TEXT_PREFIX + "embed_tokens.weight": [V, H], TEXT_PREFIX + "norm.weight": [H]}
    for i, lt in enumerate(t["layer_types"]):
        p = f"{TEXT_PREFIX}layers.{i}."
        full = lt == "full_attention"
        hd = t["global_head_dim"] if full else t["head_dim"]
        nkv = t["num_global_key_value_heads"] if full else t["num_key_value_heads"]
        shapes.update(
            {
                p + "input_layernorm.weight": [H],
                p + "post_attention_layernorm.weight": [H],
                p + "pre_feedforward_layernorm.weight": [H],
                p + "post_feedforward_layernorm.weight": [H],
                p + "layer_scalar": [1],
                p + "mlp.gate_proj.weight": [I, H],
                p + "mlp.up_proj.weight": [I, H],
                p + "mlp.down_proj.weight": [H, I],
                p + "self_attn.q_proj.weight": [t["num_attention_heads"] * hd, H],
                p + "self_attn.k_proj.weight": [nkv * hd, H],
                p + "self_attn.o_proj.weight": [H, t["num_attention_heads"] * hd],
                p + "self_attn.q_norm.weight": [hd],
                p + "self_attn.k_norm.weight": [hd],
            }
        )
        if not full:  # K=V tied on full-attention layers: no v_proj
            shapes[p + "self_attn.v_proj.weight"] = [nkv * hd, H]
    return shapes


def runtime_identity():
    files = sorted(p for ext in ("*.py", "*.cpp", "*.hpp") for p in TREE.rglob(ext) if "__pycache__" not in p.parts and "tests" not in p.parts)
    return {
        "gemma4_tree": {str(p.relative_to(TREE)): digest(p) for p in files},
        "binaries": {b: digest(TT_ROOT / b) for b in BINARIES if (TT_ROOT / b).exists()},
    }


def read_manifest(root):
    root = Path(root)
    m = json.loads((root / MANIFEST).read_text())
    if m.get("format") != FORMAT or m.get("scope") != "text-only, MeshShape 1x1 (single P150)":
        raise ValueError("unsupported native checkpoint format/scope")
    if m.get("source", {}).get("model_id") != MODEL_ID:
        raise ValueError("native checkpoint model mismatch")
    cfg = json.loads((root / "config.json").read_text())
    shapes = text_shapes(cfg)
    if m["text_tensor_shapes"] != shapes:
        raise ValueError("manifest tensor inventory disagrees with config")
    if json.loads((root / "precision_plan.json").read_text()) != m["precision_plan"]:
        raise ValueError("precision_plan.json differs from manifest")
    return m


def verify_files(root, m):
    root = Path(root).resolve()
    listed = set(m["files"])
    present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {MANIFEST, PROOF}
    if present != listed:
        raise ValueError(f"file set mismatch: extra={sorted(present - listed)[:5]} missing={sorted(listed - present)[:5]}")
    for name, info in m["files"].items():
        p = root / name
        if p.stat().st_size != info["bytes"] or digest(p) != info["sha256"]:
            raise ValueError(f"hash/size mismatch: {name}")


def require_proof(root, m):
    root = Path(root)
    proof = json.loads((root / PROOF).read_text())
    if proof.get("manifest_sha256") != digest(root / MANIFEST):
        raise ValueError("equivalence proof does not match this manifest")
    if proof.get("exact_logits_equal") is not True or proof.get("tokens_compared", 0) < 1:
        raise ValueError("equivalence proof is not an exact-logits proof")
    if proof.get("precision_plan") != m["precision_plan"]:
        raise ValueError("equivalence proof precision plan mismatch")
    return proof


# GEMMA4_* variables that select paths / the serving entry point and never change numerics; every other
# GEMMA4_* variable (matmul accumulation, norms, GELU, ring / prefill chunk, ...) must match the proof.
# GEMMA4_DRAFT_LEN is checked against the drafter's speculative proof (draft_len) by the serving loader.
SERVING_SELECTORS = frozenset({"GEMMA4_NATIVE", "GEMMA4_SINGLE_USER", "GEMMA4_ASSISTANT_DIR", "GEMMA4_DRAFT_LEN",
                               "GEMMA4_TOOLS_DIR"})


def prepare_native_runtime(root, allow_unproven=False, verify_hashes=True):
    """Loader entry: returns (model_path, state_dict, tt_cache_path, identity)."""
    import torch
    from safetensors.torch import load_file

    root = Path(root).resolve()
    m = read_manifest(root)
    if verify_hashes:
        verify_files(root, m)
    identity = {"manifest_sha256": digest(root / MANIFEST), "proof": None}
    if not allow_unproven:
        proof = require_proof(root, m)
        now = runtime_identity()
        if now != proof["runtime_identity"]:
            diff = [k for k in now["gemma4_tree"] if proof["runtime_identity"]["gemma4_tree"].get(k) != now["gemma4_tree"][k]]
            raise ValueError(f"runtime differs from the one the equivalence proof covers: {diff[:8] or 'binaries'}")
        env_now = {k: v for k, v in os.environ.items()
                   if k.startswith("GEMMA4_") and k != "GEMMA4_PRECISION_PLAN" and k not in SERVING_SELECTORS}
        if env_now != proof["runtime_env"]:
            raise ValueError(f"GEMMA4_* runtime environment {env_now} differs from the proof's {proof['runtime_env']}")
        identity["proof"] = digest(root / PROOF)
    small = load_file(str(root / "small.safetensors"))
    state_dict = {}
    for name, shape in m["text_tensor_shapes"].items():
        if len(shape) == 2:
            state_dict[name] = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        else:
            state_dict[name] = small[name]
    return str(root), state_dict, str(root / "tensors"), identity


def finalize(root, source, fetch_verification):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    root, source = Path(root).resolve(), Path(source).resolve()
    cfg = json.loads((source / "config.json").read_text())
    shapes = text_shapes(cfg)
    for f in META_FILES:
        shutil.copyfile(source / f, root / f)
    small = {}
    with safe_open(str(source / "model.safetensors"), framework="pt") as fh:
        keys = set(fh.keys())
        missing = [k for k in shapes if k not in keys]
        if missing:
            raise ValueError(f"source lacks text tensors: {missing[:5]}")
        for name, shape in shapes.items():
            if list(fh.get_slice(name).get_shape()) != shape:
                raise ValueError(f"source shape mismatch {name}")
            if len(shape) != 2:
                small[name] = fh.get_tensor(name).to(torch.bfloat16).contiguous()
    save_file(small, str(root / "small.safetensors"))
    tensor_files = sorted(p for p in (root / "tensors").rglob("*.tensorbin"))
    if not tensor_files:
        raise ValueError("builder produced no tensors")
    files = {}
    for p in sorted(q for q in root.rglob("*") if q.is_file()):
        rel = str(p.relative_to(root))
        if rel in (MANIFEST, PROOF):
            continue
        files[rel] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    fv = json.loads(Path(fetch_verification).read_text())
    manifest = {
        "format": FORMAT,
        "scope": "text-only, MeshShape 1x1 (single P150)",
        "source": {
            "model_id": MODEL_ID,
            "fetch_verification": fv,
            "source_dir": str(source),
            "config_sha256": digest(source / "config.json"),
        },
        "precision_plan": json.loads((root / "precision_plan.json").read_text()),
        "text_tensor_shapes": shapes,
        "tensor_files": len(tensor_files),
        "tensor_bytes": sum(p.stat().st_size for p in tensor_files),
        "files": files,
        "builder_runtime_identity": runtime_identity(),
    }
    save_json(root / MANIFEST, manifest)
    return {"files": len(files), "tensor_files": len(tensor_files), "tensor_GB": manifest["tensor_bytes"] / 1e9}


def record(root, baseline, restored):
    import numpy as np

    root, baseline, restored = Path(root).resolve(), Path(baseline), Path(restored)
    m = read_manifest(root)
    verify_files(root, m)
    a = json.loads((baseline / "metadata.json").read_text())
    b = json.loads((restored / "metadata.json").read_text())
    for meta in (a, b):
        if meta.get("status") != "complete" or meta.get("mode") != "teacher" or not meta.get("saved_logits"):
            raise ValueError("both runs must be completed teacher runs with saved full logits")
    for key in ("input_sha256", "records", "saved_logits"):
        if a[key] != b[key]:
            raise ValueError(f"run configuration mismatch: {key}")
    if a["runtime"] != b["runtime"]:
        raise ValueError("runtime (source tree/tool/env/plan) differs between the two runs")
    if a["runtime"]["precision_plan"] != m["precision_plan"]:
        raise ValueError("baseline plan differs from the checkpoint plan")
    if a["weights"]["native"] or not a["weights"]["model_dir"]:
        raise ValueError("baseline must load the original HF weights")
    if Path(a["weights"]["cache"]).resolve() == (root / "tensors").resolve():
        raise ValueError("baseline must quantize from the original weights, not reuse the checkpoint tensors")
    if not b["weights"]["native"] or b["native_identity"].get("manifest_sha256") != digest(root / MANIFEST):
        raise ValueError("restored run is not bound to this native checkpoint")
    tokens, recs = 0, []
    for rid in a["saved_logits"]:
        x = np.load(baseline / f"{rid}.logits.npy", mmap_mode="r")
        y = np.load(restored / f"{rid}.logits.npy", mmap_mode="r")
        if x.shape != y.shape or x.shape[1] != 262144:
            raise ValueError(f"logit shape mismatch for {rid}")
        for s in range(0, x.shape[0], 64):
            if not np.array_equal(np.asarray(x[s : s + 64]), np.asarray(y[s : s + 64])):
                raise ValueError(f"native reload logits differ for {rid} at row {s}")
        tokens += x.shape[0]
        recs.append(
            {
                "id": rid,
                "rows": int(x.shape[0]),
                "baseline_sha256": digest(baseline / f"{rid}.logits.npy"),
                "restored_sha256": digest(restored / f"{rid}.logits.npy"),
            }
        )
    for rid in a["records"]:
        za, zb = np.load(baseline / f"{rid}.npz"), np.load(restored / f"{rid}.npz")
        if not (np.array_equal(za["nll"], zb["nll"]) and np.array_equal(za["argmax"], zb["argmax"])):
            raise ValueError(f"teacher-forced NLL/argmax differ for {rid}")
    proof = {
        "manifest_sha256": digest(root / MANIFEST),
        "exact_logits_equal": True,
        "tokens_compared": tokens,
        "records_full_logits": recs,
        "records_nll_argmax_equal": a["records"],
        "precision_plan": m["precision_plan"],
        "runtime_identity": runtime_identity(),
        "runtime_run_sha256": {"gemma4_tree_sha256": a["runtime"]["gemma4_tree_sha256"], "tool_sha256": a["runtime"]["tool_sha256"]},
        "runtime_env": a["runtime"]["env"],
        "baseline_metadata_sha256": digest(baseline / "metadata.json"),
        "restored_metadata_sha256": digest(restored / "metadata.json"),
        "scope": "bit-exact full-vocabulary logits, original weights quantized at load vs native reload, same runtime; not a quality certification",
    }
    save_json(root / PROOF, proof)
    return {k: proof[k] for k in ("exact_logits_equal", "tokens_compared", "manifest_sha256")}


def _drafter_files(droot):
    return {
        str(p.relative_to(droot)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(q for q in droot.rglob("*") if q.is_file())
        if str(p.relative_to(droot)) not in (DRAFTER_MANIFEST, SPEC_PROOF)
    }


def drafter_finalize(droot, source, target, dtype):
    droot, source, target = Path(droot).resolve(), Path(source).resolve(), Path(target).resolve()
    fv = json.loads((source / "fetch-verification.json").read_text())
    if not list((droot / "tensors").rglob("*.tensorbin")):
        raise ValueError("drafter tensors/ is empty: run the runtime once with --assistant-cache <droot>/tensors")
    for name in ("config.json", "model.safetensors"):
        if digest(droot / name) != digest(source / name):
            raise ValueError(f"{name} differs from the source")
    manifest = {
        "format": DRAFTER_FORMAT,
        "source": {"model_id": "google/gemma-4-12B-it-assistant", "fetch_verification": fv, "source_dir": str(source),
                   "config_sha256": digest(source / "config.json"), "safetensors_sha256": digest(source / "model.safetensors")},
        "dtype": dtype,
        "target_manifest_sha256": digest(target / MANIFEST),
        "files": _drafter_files(droot),
        "builder_runtime_identity": runtime_identity(),
    }
    save_json(droot / DRAFTER_MANIFEST, manifest)
    return {"files": len(manifest["files"]), "dtype": dtype}


def prepare_drafter(droot, target_identity, allow_unproven=False):
    """Loader entry for the drafter: verify files (+ speculative proof); returns (dir, cache path, identity)."""
    droot = Path(droot).resolve()
    m = json.loads((droot / DRAFTER_MANIFEST).read_text())
    if m.get("format") != DRAFTER_FORMAT:
        raise ValueError("unsupported drafter format")
    if _drafter_files(droot) != m["files"]:
        raise ValueError("drafter file set / hash mismatch")
    if target_identity.get("manifest_sha256") != m["target_manifest_sha256"]:
        raise ValueError("drafter was built for a different target checkpoint")
    identity = {"drafter_manifest_sha256": digest(droot / DRAFTER_MANIFEST), "spec_proof": None, "dtype": m["dtype"]}
    if not allow_unproven:
        proof = json.loads((droot / SPEC_PROOF).read_text())
        if proof.get("drafter_manifest_sha256") != identity["drafter_manifest_sha256"] or proof.get("streams_identical") is not True:
            raise ValueError("speculative proof does not cover this drafter")
        if proof.get("target_proof_sha256") != target_identity.get("proof"):
            raise ValueError("speculative proof was made against a different target proof")
        if proof["runtime_identity"] != runtime_identity():
            raise ValueError("runtime differs from the one the speculative proof covers")
        identity["spec_proof"] = digest(droot / SPEC_PROOF)
        identity["draft_len"] = proof["draft_len"]
    return str(droot), str(droot / "tensors"), identity


def spec_record(droot, target, reference, speculative):
    droot, target = Path(droot).resolve(), Path(target).resolve()
    a = json.loads((Path(reference) / "metadata.json").read_text())
    b = json.loads((Path(speculative) / "metadata.json").read_text())
    for meta in (a, b):
        if meta.get("status") != "complete" or meta.get("mode") != "generate":
            raise ValueError("both runs must be completed generate runs")
        ident = meta.get("native_identity") or {}
        if ident.get("manifest_sha256") != digest(target / MANIFEST) or not ident.get("proof"):
            raise ValueError("runs must load the proven target checkpoint (no --allow-unproven)")
    if a["runtime"] != b["runtime"] or a["native_identity"] != b["native_identity"]:
        raise ValueError("reference and speculative runs differ in runtime / target identity")
    if a.get("spec") or not b.get("spec"):
        raise ValueError("reference must be non-speculative and the second run speculative")
    if b["spec"]["drafter"]["drafter_manifest_sha256"] != digest(droot / DRAFTER_MANIFEST):
        raise ValueError("speculative run did not use this drafter")
    if a["prompts_sha256"] != b["prompts_sha256"] or a["max_new"] != b["max_new"]:
        raise ValueError("runs used different prompts / lengths")
    ra = {r["id"]: r["gen_ids"] for r in map(json.loads, open(Path(reference) / "generations.jsonl"))}
    rb = {r["id"]: r["gen_ids"] for r in map(json.loads, open(Path(speculative) / "generations.jsonl"))}
    if ra.keys() != rb.keys() or any(ra[k] != rb[k] for k in ra):
        raise ValueError("speculative greedy streams differ from the non-speculative ones")
    proof = {
        "drafter_manifest_sha256": digest(droot / DRAFTER_MANIFEST),
        "target_manifest_sha256": digest(target / MANIFEST),
        "target_proof_sha256": a["native_identity"]["proof"],
        "streams_identical": True,
        "prompts_sha256": a["prompts_sha256"],
        "max_new": a["max_new"],
        "draft_len": b["spec"]["draft_len"],
        "tokens_compared": sum(len(v) for v in ra.values()),
        "streams_sha256": {k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in sorted(ra.items())},
        "runtime_identity": runtime_identity(),
        "runtime_env": a["runtime"]["env"],
        "reference_metadata_sha256": digest(Path(reference) / "metadata.json"),
        "speculative_metadata_sha256": digest(Path(speculative) / "metadata.json"),
        "scope": "greedy token streams of speculative decoding equal non-speculative greedy decoding on the recorded prompts, same proven runtime",
    }
    save_json(droot / SPEC_PROOF, proof)
    return {k: proof[k] for k in ("streams_identical", "tokens_compared", "draft_len")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("finalize")
    f.add_argument("root")
    f.add_argument("--source", required=True)
    f.add_argument("--fetch-verification", required=True)
    r = sub.add_parser("record")
    r.add_argument("root")
    r.add_argument("--baseline", required=True)
    r.add_argument("--restored", required=True)
    v = sub.add_parser("verify")
    v.add_argument("root")
    df = sub.add_parser("drafter-finalize")
    df.add_argument("root")
    df.add_argument("--source", required=True)
    df.add_argument("--target", required=True)
    df.add_argument("--dtype", default="bfp8")
    sr = sub.add_parser("spec-record")
    sr.add_argument("root")
    sr.add_argument("--target", required=True)
    sr.add_argument("--reference", required=True)
    sr.add_argument("--speculative", required=True)
    a = ap.parse_args()
    if a.cmd == "drafter-finalize":
        print(json.dumps(drafter_finalize(a.root, a.source, a.target, a.dtype), indent=1))
        return
    if a.cmd == "spec-record":
        print(json.dumps(spec_record(a.root, a.target, a.reference, a.speculative), indent=1))
        return
    if a.cmd == "finalize":
        print(json.dumps(finalize(a.root, a.source, a.fetch_verification), indent=1))
    elif a.cmd == "record":
        print(json.dumps(record(a.root, a.baseline, a.restored), indent=1))
    else:
        _, sd, cache, ident = prepare_native_runtime(a.root)
        print(json.dumps({"verified": True, "tensors": len(sd), "cache": cache, "identity": ident}, indent=1))


if __name__ == "__main__":
    main()
