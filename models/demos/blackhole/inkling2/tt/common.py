# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Shared device helpers: compute kernel configs, host->device upload, cached weight upload."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
CACHE_ROOT = Path(os.environ.get("INK_TENSOR_CACHE") or Path.home() / ".cache" / "inkling2-mini" / "tensors")
DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16, "fp32": ttnn.float32}
_DTYPE_TAG = {ttnn.bfloat4_b: "bfp4", ttnn.bfloat8_b: "bfp8", ttnn.bfloat16: "bf16", ttnn.float32: "fp32"}


def compute_config(fidelity=ttnn.MathFidelity.HiFi4, fp32_acc: bool = True, packer_l1_acc: bool = True,
                   approx: bool = False):
    return ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE, math_fidelity=fidelity, math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc, packer_l1_acc=packer_l1_acc,
    )


CK_HIFI4 = compute_config(ttnn.MathFidelity.HiFi4)                      # fp32/bf16 matmuls, LM head
CK_HIFI2 = compute_config(ttnn.MathFidelity.HiFi2)                      # BFP8-weight matmuls
CK_NORM = compute_config(ttnn.MathFidelity.HiFi4, packer_l1_acc=False)  # norms / elementwise


def to_dev(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=DRAM):
    if dtype in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b):
        t = t.to(torch.bfloat16)
    elif dtype == ttnn.float32:
        t = t.to(torch.float32)
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def free(*ts):
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


class WeightCache:
    """Device weights keyed by (tensor key, dtype, layout): the first use converts on the host and dumps the tilized host
    tensor (ttnn.dump_tensor) under root/; later loads read that file and skip the checkpoint."""

    def __init__(self, device, root: Path = CACHE_ROOT, enabled: bool = True):
        self.device = device
        self.dir = Path(root)
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str, dtype, layout) -> Path:
        lay = "rm" if layout == ttnn.ROW_MAJOR_LAYOUT else "tile"
        return self.dir / f"{key}.{_DTYPE_TAG[dtype]}.{lay}.tensorbin"

    def get(self, key: str, build: Callable[[], torch.Tensor], dtype, layout=ttnn.TILE_LAYOUT, memory_config=DRAM):
        path = self.path(key, dtype, layout)
        if self.enabled and path.exists():
            host = ttnn.load_tensor(str(path))
        else:
            t = build()
            t = t.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
            host = ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout)
            if self.enabled:
                tmp = path.with_name(path.name.replace(".tensorbin", f".{os.getpid()}.partial.tensorbin"))
                ttnn.dump_tensor(str(tmp), host)
                os.replace(tmp, path)
        if self.device is None:
            return host
        return ttnn.to_device(host, self.device, memory_config=memory_config)
