# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Tensor sources: where the runtime reads checkpoint values from.

A source exposes ``keys()``, ``fmt(name)`` ('bfp4' | 'bfp8' | 'bf16' | 'fp32': the format the values were prepared
for), ``get(name) -> torch.Tensor`` (HF tensor name, HF shape and orientation) and ``identity`` (a short string that
keys the device tensor cache: two sources with the same identity must return identical values).

* ``HFSource``: the BF16 HF safetensors (values are the BF16 originals). ``fmt`` is the RTN plan: Linear weights
  'bfp8' (MLP w1/w2/w3: ``LFM_HF_MLP_FMT``, default 'bfp8'), everything else 'bf16'; ttnn's round-to-nearest does the
  conversion.
* ``QuantSource``: the quant workstream's lfmq-1 checkpoint through lfmq/reader.py (vllm_bundle/lfmq)
  (``open_checkpoint``); its BFP values are already on the BFP grid, so ttnn's conversion reproduces them exactly, and
  ``fmt`` names the device format per tensor.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

HF_DIR = Path(os.environ.get("LFM3_HF_DIR", "hf"))  # the BF16 HF snapshot (development tools)
HF_REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
# The checkpoint reader package (lfmq/); LFM_QUANT_PKG names the directory holding lfmq/ (default: this package's
# vllm_bundle/).
QUANT_READER = (Path(os.environ.get("LFM_QUANT_PKG", str(Path(__file__).resolve().parents[1] / "vllm_bundle")))
                / "lfmq" / "reader.py")
QUANT_CKPT = Path(os.environ.get("LFM3_QUANT_CKPT", "quant/ckpt"))


class HFSource:
    def __init__(self, hf_dir: str | Path = HF_DIR):
        from safetensors import safe_open

        self.dir = Path(hf_dir)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:  # single-file snapshot
            with safe_open(str(self.dir / "model.safetensors"), framework="pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._open = safe_open
        self._files = {}
        self.identity = f"hf-{HF_REVISION[:12]}"

    def keys(self):
        return self.weight_map.keys()

    def _file(self, shard: str):
        f = self._files.get(shard)
        if f is None:
            f = self._files[shard] = self._open(str(self.dir / shard), framework="pt")
        return f

    def get(self, name: str) -> torch.Tensor:
        return self._file(self.weight_map[name]).get_tensor(name)

    def fmt(self, name: str) -> str:
        if name.endswith((".w1.weight", ".w2.weight", ".w3.weight")):
            return os.environ.get("LFM_HF_MLP_FMT", "bfp8")
        if name.endswith("_proj.weight"):
            return "bfp8"
        return "bf16"


class QuantSource:
    """Adapter over the quant workstream's reader (open_checkpoint(path) -> obj with keys/fmt/get)."""

    def __init__(self, ckpt: str | Path = QUANT_CKPT, reader: str | Path = QUANT_READER):
        pkg_root = str(Path(reader).resolve().parent.parent)  # .../quant (lfmq is a package there)
        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)
        from lfmq.reader import open_checkpoint

        self.ckpt = open_checkpoint(str(ckpt))
        ident = Path(ckpt)
        manifest = next((p for p in (ident / "manifest.json", ident / "index.json") if p.exists()), None)
        digest = hashlib.sha1((manifest.read_bytes() if manifest else str(ident.resolve()).encode())).hexdigest()
        self.identity = f"q-{digest[:10]}"  # content-keyed: the same checkpoint mounted anywhere shares the cache

    def keys(self):
        return self.ckpt.keys()

    def get(self, name: str) -> torch.Tensor:
        return self.ckpt.get(name)

    def fmt(self, name: str) -> str:
        return self.ckpt.fmt(name)


def open_source(spec: str | None = None):
    """'hf' (default) or a quant checkpoint directory ('q' = LFM3_QUANT_CKPT, default quant/ckpt)."""
    if spec in (None, "", "hf"):
        return HFSource()
    return QuantSource(QUANT_CKPT if spec == "q" else spec)
