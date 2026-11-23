# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Shared device helpers: compute configs, cached weight upload."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
import ttnn

CACHE_ROOT = Path(os.environ.get("LFM_TENSOR_CACHE", str(Path.home() / ".cache" / "lfm3" / "tensors")))
DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16, "fp32": ttnn.float32}
_DTYPE_TAG = {ttnn.bfloat4_b: "bfp4", ttnn.bfloat8_b: "bfp8", ttnn.bfloat16: "bf16", ttnn.float32: "fp32"}


def compute_config(fidelity=ttnn.MathFidelity.HiFi2, fp32_acc: bool = True, packer_l1_acc: bool = True,
                   approx: bool = False):
    return ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE, math_fidelity=fidelity, math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc, packer_l1_acc=packer_l1_acc,
    )


# Matmuls with BFP8 weights: HiFi2 keeps the activation's full BF16 mantissa through two passes; FP32 accumulation.
CK_BFP8 = compute_config(ttnn.MathFidelity.HiFi2)
# LM head, RoPE rotate_half and gate-expansion matmuls: HiFi4 + FP32 accumulation.
CK_HIFI4 = compute_config(ttnn.MathFidelity.HiFi4)
# Elementwise / norm configs.
CK_NORM = compute_config(ttnn.MathFidelity.HiFi4, packer_l1_acc=False)


class WeightCache:
    """Device weights keyed by (source identity, tensor key, dtype): first use converts on the host and dumps the
    tilized host tensor (ttnn.dump_tensor); later loads read that file and skip the source entirely."""

    def __init__(self, device, identity: str, root: Path = CACHE_ROOT, enabled: bool = True):
        self.device = device
        self.dir = root / identity
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str, dtype) -> Path:
        return self.dir / f"{key}.{_DTYPE_TAG[dtype]}.tensorbin"

    def get(self, key: str, build: Callable[[], torch.Tensor], dtype, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, store: bool = True):
        path = self.path(key, dtype)
        if self.enabled and path.exists():
            host = ttnn.load_tensor(str(path))
        else:
            t = build()
            if dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16):
                t = t.to(torch.bfloat16)
            else:
                t = t.to(torch.float32)
            host = ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout)
            if self.enabled and store:
                tmp = path.with_name(path.name.replace(".tensorbin", ".partial.tensorbin"))
                ttnn.dump_tensor(str(tmp), host)
                os.replace(tmp, path)
        if self.device is None:
            return host
        return ttnn.to_device(host, self.device, memory_config=memory_config)


def to_dev(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def free(*ts):
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)
