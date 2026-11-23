# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Packed BFP codes of a TT-snapped weight (torch only; shared by the builders and reader.py).

Geometry (tt-metal bfloat8_b / bfloat4_b): the runtime hands ttnn W^T = [in, out];
a block is 16 consecutive OUTPUT channels at one input index. For an HF weight W [..., out, in]:
  e  uint8 [..., in, out/16]   shared biased fp32 exponent per block
  q  bfp8: int8  [..., in, out]       signed codes
     bfp4: uint8 [..., in, out/2]     two 4-bit sign/magnitude codes per byte (bit 3 = sign), even index in the low nibble
value = q * 2^(e - 127 - (W - 1)), W = 7 (bfp8) / 3 (bfp4); every value is exactly bf16-representable.
"""
import torch

MANT = {"bfp8": 7, "bfp4": 3}
BLOCK = 16


def encode(q, E, fmt):
    """q int32 signed codes [..., in, out] (|q| <= 2^W - 1), E int [..., in, out/16] -> (q_stored, e_stored)."""
    if fmt == "bfp4":
        c = (q.abs() | ((q < 0).to(q.dtype) << 3)).to(torch.uint8)
        qs = (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()
    else:
        qs = q.to(torch.int8).contiguous()
    return qs, E.to(torch.uint8).contiguous()


def codes(qs, fmt):
    """stored q -> int32 signed codes [..., in, out]."""
    if fmt == "bfp4":
        c = torch.stack([qs & 15, qs >> 4], -1).reshape(*qs.shape[:-1], qs.shape[-1] * 2).to(torch.int32)
        return torch.where((c & 8) != 0, -(c & 7), c & 7)
    return qs.view(torch.int8).to(torch.int32)


def decode(qs, es, fmt):
    """stored (q, e) -> bf16 HF-orientation weight [..., out, in] (exact)."""
    q = codes(qs, fmt)
    *lead, inn, out = q.shape
    step = torch.ldexp(torch.ones(es.shape, dtype=torch.float32), es.to(torch.int32) - 127 - (MANT[fmt] - 1))
    v = q.reshape(*lead, inn, out // BLOCK, BLOCK).to(torch.float32) * step.unsqueeze(-1)  # exact: |q| < 2^8
    return v.reshape(*lead, inn, out).transpose(-2, -1).contiguous().to(torch.bfloat16)
