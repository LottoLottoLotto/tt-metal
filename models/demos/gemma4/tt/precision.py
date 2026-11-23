# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-module dtype/precision overrides for Gemma4.

Reads precision_overrides.json and resolves a (model variant, mesh shape) tuple
into a {module_name: ttnn.DataType} mapping. Modules without an override use a
caller-supplied default (typically bfloat16).

The JSON file is the single source of truth for per-system precision tweaks
(e.g. dropping shared_mlp to bfp8 on Gemma4-31B at 1x2 to fit DRAM). New entries
are added there rather than in code.
"""

import json
import os

import ttnn

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precision_overrides.json")

# Module names that may be overridden — keep in sync with the JSON schema and
# with the constructors that accept these kwargs (Gemma4Model and below).
KNOWN_MODULES = ("shared_mlp", "mlp_gate_up", "mlp_down", "attention", "experts", "router", "lm_head", "embedding")

# Environment override: path to a JSON precision plan
#   {"default": {module: dtype, ...}, "layers": {"<idx>": {module: dtype, ...}, ...}}
# When set it replaces the precision_overrides.json lookup entirely (quantization
# studies and native checkpoints pin their exact plan this way).
PLAN_ENV = "GEMMA4_PRECISION_PLAN"

_DTYPE_BY_NAME = {
    "bf16": ttnn.bfloat16,
    "bfloat16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "fp32": ttnn.float32,
    "float32": ttnn.float32,
}


def dtype_to_str(dtype):
    """Short stable string for cache-filename suffixes ("bf16" / "bfp8" / "fp32").

    Cache filenames embed the dtype string so flipping a module's dtype in
    precision_overrides.json doesn't reuse a stale cached tensor at the
    previous precision.
    """
    if dtype == ttnn.bfloat16:
        return "bf16"
    if dtype == ttnn.bfloat8_b:
        return "bfp8"
    if dtype == ttnn.bfloat4_b:
        return "bfp4"
    if dtype == ttnn.float32:
        return "fp32"
    raise ValueError(f"No cache-suffix mapping for dtype {dtype}")


class Gemma4Precision:
    """Per-module dtype mapping. Construct via ``Gemma4Precision.load(...)``
    or directly with ``Gemma4Precision({...})``."""

    def __init__(self, overrides=None, layer_overrides=None):
        self._overrides = dict(overrides) if overrides else {}
        self._layer_overrides = {int(k): dict(v) for k, v in (layer_overrides or {}).items()}

    def get(self, module_name, default=ttnn.bfloat16, layer_idx=None):
        """Resolve a module dtype: per-layer entry, then model-wide entry, then
        ``default``. ``mlp_gate_up``/``mlp_down`` fall back to ``shared_mlp``."""
        names = (module_name, "shared_mlp") if module_name in ("mlp_gate_up", "mlp_down") else (module_name,)
        for name in names:
            if layer_idx is not None and name in self._layer_overrides.get(int(layer_idx), {}):
                return self._layer_overrides[int(layer_idx)][name]
            if name in self._overrides:
                return self._overrides[name]
        return default

    def __repr__(self):
        return f"Gemma4Precision({self._overrides!r}, layers={self._layer_overrides!r})"

    @classmethod
    def from_plan(cls, plan):
        """Build from a plan dict ``{"default": {...}, "layers": {"i": {...}}}``."""

        def resolve(raw, where):
            out = {}
            for k, v in raw.items():
                if k not in KNOWN_MODULES:
                    raise ValueError(f"precision plan {where}: unknown module {k!r}")
                if v not in _DTYPE_BY_NAME:
                    raise ValueError(f"precision plan {where}.{k}={v!r}: unknown dtype")
                out[k] = _DTYPE_BY_NAME[v]
            return out

        default = resolve(plan.get("default", {}), "default")
        layers = {int(i): resolve(v, f"layers.{i}") for i, v in plan.get("layers", {}).items()}
        return cls(default, layers)

    @classmethod
    def load(cls, model_path, mesh_shape):
        """Resolve overrides for the given (model, mesh).

        model_path: full path to the HF checkpoint; we key on the basename.
        mesh_shape: (rows, cols) tuple, formatted as "RxC" for the JSON key.
        """
        plan_path = os.environ.get(PLAN_ENV)
        if plan_path:
            with open(plan_path) as f:
                return cls.from_plan(json.load(f))
        model_key = os.path.basename(str(model_path).rstrip("/"))
        mesh_key = f"{mesh_shape[0]}x{mesh_shape[1]}"

        try:
            with open(_PATH) as f:
                table = json.load(f)
        except FileNotFoundError:
            return cls({})

        model_entry = table.get(model_key)
        if not model_entry:
            return cls({})

        # Mesh-specific override wins over "default"
        raw = model_entry.get(mesh_key) or model_entry.get("default") or {}
        resolved = {}
        for k, v in raw.items():
            if k not in KNOWN_MODULES:
                continue  # ignore unknown / future keys silently
            if v not in _DTYPE_BY_NAME:
                raise ValueError(
                    f"precision_overrides.json[{model_key}][{mesh_key}][{k}]={v!r} — "
                    f"unknown dtype; expected one of {sorted(_DTYPE_BY_NAME)}"
                )
            resolved[k] = _DTYPE_BY_NAME[v]
        return cls(resolved)


_FIDELITY = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi3": ttnn.MathFidelity.HiFi3,
    "hifi4": ttnn.MathFidelity.HiFi4,
}


def matmul_compute_config(weight_dtype):
    """Explicit compute config for weight matmuls.

    ttnn's implicit default depends on whether a program config is passed
    (HiFi2 without one, LoFi with one — e.g. the LM head), which silently mixes
    fidelities. Pin one: ``GEMMA4_MM_FIDELITY_<DTYPE>`` (e.g. ``..._BFP4``)
    overrides ``GEMMA4_MM_FIDELITY`` (default hifi2). FP32 dest accumulation by default
    (``GEMMA4_MM_FP32ACC=0`` disables): with bf16 accumulation the K=15360 down projection
    alone showed ~4% relative error vs HF on bf16 weights; FP32 accumulation brings it to <1%.
    L1 packer accumulation on.
    """
    name = os.environ.get(f"GEMMA4_MM_FIDELITY_{dtype_to_str(weight_dtype).upper()}") or os.environ.get(
        "GEMMA4_MM_FIDELITY", "hifi2"
    )
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=_FIDELITY[name.lower()],
        math_approx_mode=False,
        fp32_dest_acc_en=os.environ.get("GEMMA4_MM_FP32ACC", "1") == "1",
        packer_l1_acc=True,
    )


def norm_compute_config():
    """RMSNorm compute config: HiFi4 with FP32 dest accumulation (``GEMMA4_NORM_FP32=0`` restores
    the op default). Gemma residual streams reach |x|~1e3+, so the sum of squares needs FP32."""
    if os.environ.get("GEMMA4_NORM_FP32", "1") == "0":
        return None
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def gelu_fast():
    """``GEMMA4_GELU_FAST=1`` keeps ttnn's fast/approximate GELU; default is the accurate one."""
    return os.environ.get("GEMMA4_GELU_FAST", "0") == "1"
