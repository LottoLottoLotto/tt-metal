# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Per-layer expert precision map (owner: Prec4; numbers in quant/NOTES.md sections 6-7).

`Inkling2Model(precision=...)` accepts
  "bfp8" | "bfp4"            every layer's experts in that ttnn format (RTN = ttnn's own host conversion), or
  "mixed:<json>"             a JSON object {"default": SPEC, "<layer>" | "<a>-<b>": SPEC, ...} (later keys win),
SPEC = "bfp8" | "bfp4" | "bfp4:<variant>".  A variant `V` is resolved under VARIANT_ROOT (env INK_VARIANT_ROOT,
default: native/ next to the checkpoint directory MODEL_WEIGHTS_DIR, the release repo layout), first match wins:
  1. native package  VARIANT_ROOT/V/native_manifest.json + VARIANT_ROOT/V/tensors/*.tensorbin: the expert tensors
     exactly as Inkling2MoE stores them on the device (ttnn TILE dumps, WeightCache file naming, moe_layout tile
     order), read directly (no conversion, never written; quant/native_pkg.py builds / verifies it), plus attn/ (the
     attention variant below).  A tensor the package lacks (e.g. a changed moe layout tag) is an error.
     INK_VERIFY_PACKAGE=1 re-hashes every file (experts and attention) at load.
  2. codes           VARIANT_ROOT/V/L{nn}.safetensors written by quant/gptq_ink.py: per tensor t in gate [E, I, H],
     up [E, I, H], down [E, H, I] (nn.Linear [out, in]) `t.q` int8 mantissas (-7..7) and `t.e` uint8 [E, out / 16,
     in] shared biased exponents, one per 16 consecutive OUTPUT rows at one input column (the BFP block of the W^T
     orientation moe.py hands to ttnn); value q * 2^(e - 127 - 2).  NVFP4 layers hold the factored values (code *
     block scale; the per-expert factor stays outside, as moe.py's factored=True), MXFP8 layers the plain values.
     Every value is BFP4-exact for that orientation, so ttnn.from_torch(bfloat4_b) reproduces it bit for bit.
     Converted once into the device cache CACHE_ROOT/variants/V (the moe.py cache tags do not encode the variant);
  3. device cache    CACHE_ROOT/variants/V alone (codes absent): loads as long as every tensor is cached.

Attention projection weights (attn_weight_map / attn_weight; quant/NOTES.md section 8): `Inkling2Model(attn_precision=
...)` / INK_ATTN_PRECISION, default model.DEFAULT_ATTN_PRECISION, per layer and family (q = wq * q_scale, gate, k, v,
o): "bfp8" | "bfp4" | "bfp4:<V>" (every family) or "mixed:<json>" {"default": X, "<layer>" | "<a>-<b>": X, ...}
(later keys win), X = SPEC (every family) or {"<family>": SPEC, ...} (the listed families; the others keep earlier
values).  Variant V = BFP4-exact W^T [K, N] bf16 tensors "q", "gate", "k", "v", "o" in VARIANT_ROOT/V/attn/
L{nn}.safetensors (quant/gptq_attn.py; in a native package: hashed in its manifest's "attention" section).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

VARIANT_ROOT = Path(os.environ.get("INK_VARIANT_ROOT")
                    or Path(os.environ.get("MODEL_WEIGHTS_DIR") or "checkpoint").parent / "native")
FORMATS = ("bfp8", "bfp4")
MANIFEST = "native_manifest.json"
PACKAGE_FORMAT = "inkling2-mini-tt-v2"  # quant/native_pkg.py: experts (tensors/) + attention projections (attn/)


@dataclass(frozen=True)
class ExpertPrecision:
    fmt: str            # "bfp8" | "bfp4"
    variant: str = ""   # "" = RTN of the checkpoint values

    def __str__(self):
        return self.fmt + (f":{self.variant}" if self.variant else "")


def _entry(spec: str) -> ExpertPrecision:
    fmt, _, variant = spec.partition(":")
    if fmt not in FORMATS or (variant and fmt != "bfp4"):
        raise ValueError(f"expert precision {spec!r}: expected bfp8 | bfp4 | bfp4:<variant>")
    if variant and not variant.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"expert precision {spec!r}: bad variant name")
    return ExpertPrecision(fmt, variant)


def expert_map(precision: str, n_layers: int) -> list[ExpertPrecision]:
    """precision string -> one ExpertPrecision per decoder layer."""
    if not precision.startswith("mixed:"):
        return [_entry(precision)] * n_layers
    spec = json.loads(precision[len("mixed:"):])
    out = [_entry(spec.pop("default", "bfp8"))] * n_layers
    for key, val in spec.items():
        a, _, b = key.partition("-")
        lo, hi = int(a), int(b or a)
        if not 0 <= lo <= hi < n_layers:
            raise ValueError(f"expert precision map: bad layer range {key!r}")
        for layer in range(lo, hi + 1):
            out[layer] = _entry(val)
    return out


def decode_planes(q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """q int8 [..., R, C], e uint8 [..., R // 16, C] -> float32 values q * 2^(e - 129)."""
    ex = e.to(torch.int32).repeat_interleave(16, dim=-2) - 129
    return torch.ldexp(q.float(), ex)


class VariantSource:
    """CheckpointSource whose expert tensors come from a variant directory (everything else delegated)."""

    def __init__(self, base, variant: str):
        self.base = base
        self.dir = VARIANT_ROOT / variant

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _planes(self, layer: int):
        from safetensors.torch import load_file

        t = load_file(str(self.dir / f"L{layer:02d}.safetensors"))
        return tuple(decode_planes(t[f"{k}.q"], t[f"{k}.e"]) for k in ("gate", "up", "down"))

    def experts(self, layer: int, experts=None, dtype=torch.float32):
        if experts is not None or self.base.expert_format(layer) != "mxfp8":
            raise ValueError("VariantSource.experts: whole MXFP8 layers only (NVFP4 layers: experts_factored)")
        return tuple(x.to(dtype) for x in self._planes(layer))

    def experts_factored(self, layer: int, experts=None):
        if experts is not None or self.base.expert_format(layer) != "nvfp4":
            raise ValueError("VariantSource.experts_factored: whole NVFP4 layers only")
        p = f"model.llm.layers.{layer}.mlp.experts."
        # per-expert factors exactly as CheckpointSource._dequant: fl32(amax / denominator)
        f13, f2 = (self.base.tensor(p + f"{w}_weight_amax").float()
                   / self.base.tensor(p + f"{w}_weight_outer_scale_denominator").float() for w in ("w13", "w2"))
        return (*self._planes(layer), f13, f2)


def sha256_file(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=None)
def open_package(pkg_dir: str) -> dict:
    """Read and check a native package's manifest (format, every expert / attention file present with its size; all
    sha256 with INK_VERIFY_PACKAGE=1).  Returns the manifest."""
    d = Path(pkg_dir)
    m = json.loads((d / MANIFEST).read_text())
    if m.get("format") != PACKAGE_FORMAT:
        raise ValueError(f"{d / MANIFEST}: format {m.get('format')!r}, expected {PACKAGE_FORMAT!r}")
    verify = os.environ.get("INK_VERIFY_PACKAGE", "0") == "1"
    for name, t in {**m["tensors"], **m["attention"]["files"]}.items():
        p = d / t["file"]
        if not p.is_file() or p.stat().st_size != t["bytes"]:
            raise ValueError(f"native package {d}: {t['file']} missing or wrong size")
        if verify and sha256_file(p) != t["sha256"]:
            raise ValueError(f"native package {d}: {t['file']} sha256 mismatch")
    return m


def expert_source(base_src, base_cache, device, prec: ExpertPrecision):
    """(source, WeightCache) for one layer's Inkling2MoE under precision `prec`."""
    if not prec.variant:
        return base_src, base_cache
    from .common import CACHE_ROOT, WeightCache

    vdir = VARIANT_ROOT / prec.variant
    if (vdir / MANIFEST).is_file():
        open_package(str(vdir))
        return base_src, PackageCache(device, vdir / "tensors")
    cdir = CACHE_ROOT / "variants" / prec.variant
    if not ((vdir / "L00.safetensors").is_file() or cdir.is_dir()):
        raise ValueError(f"expert variant {prec.variant!r}: no native package or codes in {vdir}, no device cache {cdir}")
    return VariantSource(base_src, prec.variant), WeightCache(device, root=cdir, enabled=base_cache.enabled)


def _package_cache_cls():
    from .common import WeightCache

    class _PackageCache(WeightCache):
        """Read-only WeightCache over a native package's tensors/ (a missing tensor is an error, never built)."""

        def __init__(self, device, root: Path):
            super().__init__(device, root=root, enabled=False)
            self.enabled = True

        def get(self, key, build, dtype, layout=None, memory_config=None):
            import ttnn

            layout = ttnn.TILE_LAYOUT if layout is None else layout
            path = self.path(key, dtype, layout)
            if not path.is_file():
                raise ValueError(f"native expert package {self.dir.parent} has no {path.name} (built for another "
                                 f"expert layout / dtype?): rebuild it with quant/native_pkg.py build")
            kw = {} if memory_config is None else {"memory_config": memory_config}
            return super().get(key, build, dtype, layout, **kw)

    return _PackageCache


def PackageCache(device, root: Path):
    return _package_cache_cls()(device, root)


# ---------------------------------------------------------------- attention projection weights
ATTN_FAMILIES = ("q", "gate", "k", "v", "o")


def attn_weight_map(precision: str, n_layers: int) -> list[dict[str, ExpertPrecision]]:
    """attention weight precision string (module docstring) -> per decoder layer {family: ExpertPrecision}."""
    def apply(cur: dict, val) -> dict:
        if isinstance(val, str):
            return {f: _entry(val) for f in ATTN_FAMILIES}
        bad = set(val) - set(ATTN_FAMILIES)
        if bad:
            raise ValueError(f"attention precision map: unknown families {sorted(bad)}")
        return {**cur, **{f: _entry(s) for f, s in val.items()}}

    base = apply({}, "bfp8")
    if not precision.startswith("mixed:"):
        return [apply(base, precision)] * n_layers
    spec = json.loads(precision[len("mixed:"):])
    out = [apply(base, spec.pop("default", "bfp8"))] * n_layers
    for key, val in spec.items():
        a, _, b = key.partition("-")
        lo, hi = int(a), int(b or a)
        if not 0 <= lo <= hi < n_layers:
            raise ValueError(f"attention precision map: bad layer range {key!r}")
        for layer in range(lo, hi + 1):
            out[layer] = apply(out[layer], val)
    return out


def attn_tag(prec: dict[str, ExpertPrecision], families=ATTN_FAMILIES) -> str:
    """WeightCache key suffix of `families` under `prec` ("" = all BFP8 RTN, the historical keys)."""
    parts = [f"{f}{p.fmt[3:]}" + (f"-{p.variant}" if p.variant else "") for f in families
             if str(p := prec[f]) != "bfp8"]
    return "." + "_".join(parts) if parts else ""


def attn_weight(wc, key: str, build, parts, prec: dict[str, ExpertPrecision], layer: int, base_dtype):
    """One attention projection tensor through WeightCache `wc`.  build() -> W^T [K, N] (checkpoint values) whose column
    ranges are `parts` = [(family, n_cols), ...] in order (tile-aligned).  All families BFP8 RTN: as before (base_dtype,
    the attention dtype policy's).  Otherwise the tensor is bfloat4_b if every family is BFP4, else base_dtype, under
    key + attn_tag: RTN-BFP4 families inside a wider tensor are snapped on the host with ttnn's own BFP4 conversion
    (BFP4 values are exact in BFP8 / BF16), variant families are read from VARIANT_ROOT/V/attn/L{nn}.safetensors."""
    import ttnn

    fams = [f for f, _ in parts]
    tag = attn_tag(prec, fams)
    if not tag:
        return wc.get(key, build, base_dtype)
    all4 = all(prec[f].fmt == "bfp4" for f in fams)
    dtype = ttnn.bfloat4_b if all4 else base_dtype

    def snap4(w):
        return ttnn.to_torch(ttnn.from_torch(w.to(torch.bfloat16).contiguous(), dtype=ttnn.bfloat4_b,
                                             layout=ttnn.TILE_LAYOUT)).float()

    def built():
        from safetensors import safe_open

        w = build().float()
        cols, c0 = [], 0
        for f, n in parts:
            p, part = prec[f], w[:, c0:c0 + n]
            c0 += n
            if p.variant:
                vdir = VARIANT_ROOT / p.variant
                if (vdir / MANIFEST).is_file():
                    open_package(str(vdir))
                path = vdir / "attn" / f"L{layer:02d}.safetensors"
                with safe_open(str(path), "pt") as fh:
                    v = fh.get_tensor(f).float()
                if v.shape != part.shape or not torch.equal(snap4(v), v):
                    raise ValueError(f"{path}:{f}: shape {tuple(v.shape)} (want {tuple(part.shape)}) or not BFP4-exact")
                part = v
            elif p.fmt == "bfp4" and not all4:
                part = snap4(part)
            cols.append(part)
        assert c0 == w.shape[1]
        return torch.cat(cols, 1)

    return wc.get(key + tag, built, dtype)
