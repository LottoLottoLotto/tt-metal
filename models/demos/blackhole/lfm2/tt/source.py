# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Tensor sources: where the runtime reads checkpoint values from.

A source exposes ``keys()``, ``fmt(name)`` ('bfp4' | 'bfp8' | 'bf16' | 'fp32': the format the values were prepared
for) , ``get(name) -> torch.Tensor`` (HF tensor name, HF shape and orientation) and ``identity`` (a short string that
keys the device tensor cache: two sources with the same identity must return identical values).

* ``HFSource``: the BF16 HF safetensors shards (values are the BF16 originals; the runtime's BFP4/BFP8 conversion is
  ttnn's round-to-nearest).
* ``QuantSource``: the quant checkpoint through lfmq/reader.py (``open_checkpoint``);
  its BFP values are already on the BFP grid, so ttnn's conversion reproduces them exactly.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

# Development defaults only (serving passes the checkpoint directory), under LFM25_DEV_ROOT.
_DEV_ROOT = Path(os.environ.get("LFM25_DEV_ROOT", "lfm25"))
HF_DIR = _DEV_ROOT / "hf"
HF_REVISION = "1769516ac588c4c558089c94f27d96b0dc67b183"
# The quant checkpoint's reader package (lfmq/); LFM_QUANT_PKG names the directory holding lfmq/ (default: this
# package's vllm_bundle/).
QUANT_READER = (Path(os.environ.get("LFM_QUANT_PKG", Path(__file__).resolve().parents[1] / "vllm_bundle"))
                / "lfmq" / "reader.py")


class HFSource:
    def __init__(self, hf_dir: str | Path = HF_DIR):
        from safetensors import safe_open

        self.dir = Path(hf_dir)
        index = self.dir / "model.safetensors.index.json"
        if not index.exists():
            index = _DEV_ROOT / "index.json"
        self.weight_map = json.loads(index.read_text())["weight_map"]
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
        return "fp32" if name.endswith("expert_bias") else "bf16"

    def available(self, name: str) -> bool:
        """True when the shard holding `name` is fully downloaded (lets tests run on partial downloads)."""
        return (self.dir / self.weight_map[name]).exists()


class QuantSource:
    """Adapter over the quant agent's reader (open_checkpoint(path) -> obj with keys/fmt/get)."""
    def __init__(self, ckpt: str | Path, reader: str | Path = QUANT_READER):
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

    def available(self, name: str) -> bool:
        return True


def open_source(spec: str | None = None):
    """'hf' (default) or a quant checkpoint directory."""
    if spec in (None, "", "hf"):
        return HFSource()
    return QuantSource(spec)
