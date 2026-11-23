# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Lazy reader of the Inkling-2-Mini HF checkpoint, by ORIGINAL tensor name, with exact expert dequantization.

Plain torch + safetensors (host venv and the tt-metal image, torch >= 2.1 with float8_e4m3fn). Nothing is read at
construction except the safetensors headers; every call reads only the bytes it needs (`safe_open` mmap slices), so the
19.5 GB `model-00001-of-00001.safetensors` is never loaded at once.

  src = CheckpointSource("checkpoint")                     # the HF checkpoint directory
  src.tensor("model.llm.layers.0.attn.wq_du.weight")     # torch.bfloat16 [3072, 2304], as stored
  w_gate, w_up, w_down = src.experts(5)                    # float32, see below
  src.expert_format(20)                                    # "nvfp4" (layers 0-19) | "mxfp8" (layers 20-23)

`.tensor(name)` returns the stored tensor unchanged: BF16 -> torch.bfloat16, F32 -> torch.float32, F8_E4M3 ->
torch.float8_e4m3fn, U8 -> torch.uint8 (packed NVFP4 codes / UE8M0 scales). Names of `mtp.safetensors` (drafter:
`fuse_weight`, `layers.N...`, ...) resolve too when that file is present.

Expert weights (`.experts(layer, experts=None, dtype=torch.float32)`):
  returns (w_gate, w_up, w_down), nn.Linear orientation [out, in], experts along dim 0:
    w_gate [E, 1152, 2304]   gate = rows 0::2 of the checkpoint's w13 (rows alternate gate, up)
    w_up   [E, 1152, 2304]   up   = rows 1::2 of w13
    w_down [E, 2304, 1152]   w2
  E = 128, or len(experts) for `experts=` (a slice / index list / int of expert ids). The reference forward is
  gate_up = x @ [w_gate; w_up]^T, act(gate, up), y = act @ w_down^T  (transformers PR: Interleave(dim=1) then
  gate_up.chunk(2) -> gate first; vLLM's kernel reads gate at column 2j and up at 2j+1 of the interleaved product).

  NVFP4 (layers 0-19; `weight_scale_format: amax_denominator`, `weight_block_shape [1, 16]`):
    w13_weight U8 [E, 2304, 1152], w2_weight U8 [E, 2304, 576]: two E2M1 codes per byte, EVEN element (along K) in
    the LOW nibble; E2M1 code c -> (+/-){0, .5, 1, 1.5, 2, 3, 4, 6}[c & 7], bit 3 = sign.
    w13_weight_scale / w2_weight_scale F8_E4M3 [E, 2304, K/16]: one scale per 16 consecutive K (input) values.
    w*_weight_amax F32 [E], w*_weight_outer_scale_denominator F32 [1] (= 2688 = 6 * 448 in this checkpoint):
    per-expert factor f[e] = fl32(amax[e] / denominator).
    W[e, r, k] = e2m1(code) * scale[e, r, k // 16] * f[e]
    computed exactly as the PR loader does: fl32(fl32(e2m1 * scale) * f) (the first product is exact in fp32: <= 6
    significant bits), then cast to `dtype` (float32: bit-identical to the PR before its cast; bfloat16: bit-identical
    to the PR loader into a bf16 model). `fouroversix_mae` only decided which scale (block max -> 6 or -> 4) the
    quantizer chose; decoding is the same.
  MXFP8 (layers 20-23; `weight_block_shape [32, 32]`, UE8M0 scales):
    w13_weight F8_E4M3 [E, 2304, 2304], w2_weight F8_E4M3 [E, 2304, 1152],
    w*_weight_scale U8 [E, 2304, K/32]: one scale per row per 32 consecutive K values (the stored layout is per row,
    the PR loader indexes it so);  W = e4m3 * 2^(scale - 127)  (exact in fp32 and in bf16: 4 significant bits).

`.experts_factored(layer, experts=None)` -> (g, u, d, f13, f2): the same tensors in float32 WITHOUT the NVFP4 per-expert
factor (g/u/d = e2m1 * scale, exact; f13/f2 float32 [E]; both None for MXFP8). W = g * f13[:, None, None] etc. This is
the lossless form for BFP storage: e2m1 * E4M3 values have at most 6 significant bits, the factor is applied to the
expert's matmul output.
"""
from __future__ import annotations

import json
import os
import struct

import torch
from safetensors import safe_open

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
_CHUNK = 8  # experts dequantized per step (bounds the int64 code-index temporaries to ~170 MB)

MAIN_FILE = "model-00001-of-00001.safetensors"
MTP_FILE = "mtp.safetensors"
_PREFIX = "model.llm.layers.{}.mlp.experts."


def _read_header(path: str) -> dict:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def _expert_index(experts, num_experts: int):
    if experts is None:
        return slice(0, num_experts)
    if isinstance(experts, int):
        return [experts]
    if isinstance(experts, range):
        return list(experts)
    return experts


class CheckpointSource:
    """Lazy, by-original-name access to the Inkling-2-Mini checkpoint (see module docstring)."""

    def __init__(self, hf_dir: str, files: tuple[str, ...] | None = None):
        self.hf_dir = hf_dir
        files = files or tuple(f for f in (MAIN_FILE, MTP_FILE) if os.path.exists(os.path.join(hf_dir, f)))
        if not files:
            raise FileNotFoundError(f"no safetensors file in {hf_dir}")
        self._file_of: dict[str, str] = {}
        self._meta: dict[str, dict] = {}
        for f in files:
            path = os.path.join(hf_dir, f)
            for name, meta in _read_header(path).items():
                if name in self._file_of:
                    raise ValueError(f"tensor {name} in both {self._file_of[name]} and {path}")
                self._file_of[name] = path
                self._meta[name] = meta
        self._handles: dict[str, object] = {}

    # ---- generic tensors ----
    def names(self) -> list[str]:
        return list(self._file_of)

    def __contains__(self, name: str) -> bool:
        return name in self._file_of

    def shape(self, name: str) -> list[int]:
        return list(self._meta[name]["shape"])

    def dtype(self, name: str) -> str:
        """safetensors dtype string: 'BF16', 'F32', 'F8_E4M3', 'U8'."""
        return self._meta[name]["dtype"]

    def _handle(self, path: str):
        h = self._handles.get(path)
        if h is None:
            h = self._handles[path] = safe_open(path, framework="pt")
        return h

    def tensor(self, name: str) -> torch.Tensor:
        """The stored tensor, unchanged dtype (BF16/F32 as stored)."""
        if name not in self._file_of:
            raise KeyError(name)
        return self._handle(self._file_of[name]).get_tensor(name)

    def _rows(self, name: str, idx) -> torch.Tensor:
        """name[idx] along dim 0, reading only those rows."""
        if isinstance(idx, slice):
            return self._handle(self._file_of[name]).get_slice(name)[idx]
        sl = self._handle(self._file_of[name]).get_slice(name)
        return torch.stack([sl[i : i + 1][0] for i in idx])

    # ---- experts ----
    def expert_format(self, layer: int) -> str:
        return "nvfp4" if (_PREFIX.format(layer) + "w13_weight_amax") in self._file_of else "mxfp8"

    def _dequant(self, layer: int, which: str, idx, with_factor: bool):
        """(values float32 [n, rows, K], factor float32 [n] or None). values include the factor iff with_factor."""
        p = _PREFIX.format(layer) + f"{which}_weight"
        nvfp4 = self.expert_format(layer) == "nvfp4"
        if nvfp4:
            amax = self.tensor(p + "_amax").float()
            den = self.tensor(p + "_outer_scale_denominator").float()
            factor_all = amax / den  # fl32, as the PR loader (amax.float() / denominator.float())
            factor = factor_all[idx] if isinstance(idx, slice) else factor_all[torch.as_tensor(idx)]
            e2m1 = torch.tensor(E2M1_VALUES, dtype=torch.float32)
            byte = torch.arange(256)
            byte_to_values = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)  # [256, 2]
        else:
            factor = None
        ids = list(range(*idx.indices(self.shape(p)[0]))) if isinstance(idx, slice) else list(idx)
        rows = self.shape(p)[1]
        cols = self.shape(p)[2] * (2 if nvfp4 else 1)
        out = torch.empty(len(ids), rows, cols, dtype=torch.float32)
        for s in range(0, len(ids), _CHUNK):
            sub = ids[s : s + _CHUNK]
            sel = slice(sub[0], sub[-1] + 1) if sub == list(range(sub[0], sub[-1] + 1)) else sub
            w = self._rows(p, sel)
            scale = self._rows(p + "_scale", sel).float()
            if nvfp4:
                v = byte_to_values[w.long()].view(*scale.shape, NVFP4_BLOCK) * scale[..., None]
                if with_factor:
                    v = v * factor[s : s + len(sub), None, None, None]
            else:
                assert float(scale.max()) < 255, "UE8M0 NaN scale"
                v = w.float().view(*scale.shape, MXFP8_BLOCK) * torch.exp2(scale - 127)[..., None]
            out[s : s + len(sub)] = v.reshape(len(sub), rows, cols)
        return out, factor

    def experts(self, layer: int, experts=None, dtype: torch.dtype = torch.float32):
        """(w_gate [E,1152,2304], w_up [E,1152,2304], w_down [E,2304,1152]) exact dequantized, see module docstring."""
        idx = _expert_index(experts, self.shape(_PREFIX.format(layer) + "w13_weight")[0])
        w13, _ = self._dequant(layer, "w13", idx, with_factor=True)
        w2, _ = self._dequant(layer, "w2", idx, with_factor=True)
        return (
            w13[:, 0::2].to(dtype).contiguous(),
            w13[:, 1::2].to(dtype).contiguous(),
            w2.to(dtype),
        )

    def experts_factored(self, layer: int, experts=None):
        """(g, u, d, f13, f2): float32 weights without the NVFP4 per-expert factor (exact), factors float32 [E] or
        None (MXFP8). w_gate == g * f13[:, None, None] (fp32 product, bit-identical to `.experts`)."""
        idx = _expert_index(experts, self.shape(_PREFIX.format(layer) + "w13_weight")[0])
        w13, f13 = self._dequant(layer, "w13", idx, with_factor=False)
        w2, f2 = self._dequant(layer, "w2", idx, with_factor=False)
        return w13[:, 0::2].contiguous(), w13[:, 1::2].contiguous(), w2, f13, f2
