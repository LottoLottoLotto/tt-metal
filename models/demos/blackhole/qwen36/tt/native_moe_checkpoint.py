# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Streaming loader for the single-P150 Qwen3.6 MoE native checkpoint."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

FORMAT = "qwen36-35b-a3b-ttnn-native-v1"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
MANIFEST = "native_manifest.json"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_manifest(root, verify_files=False):
    root = Path(root)
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("format") != FORMAT or manifest.get("revision") != REVISION:
        raise ValueError("Unsupported Qwen3.6 MoE native checkpoint")
    if manifest.get("status") != "tensor_roundtrip_verified" or not manifest.get("tensors"):
        raise ValueError("Incomplete Qwen3.6 MoE native checkpoint")
    roundtrip = manifest.get("roundtrip", {})
    tensors = manifest.get("tensors", {})
    if (
        roundtrip.get("all_passed") is not True
        or roundtrip.get("native", 0) + roundtrip.get("lossless", 0) != len(tensors)
    ):
        raise ValueError("Native checkpoint roundtrip inventory is inconsistent")
    for name, entry in tensors.items():
        if (
            entry.get("storage") not in ("ttnn-tile", "safetensors-lossless")
            or entry.get("roundtrip_verified") is not True
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
            or not entry.get("source_shape")
        ):
            raise ValueError(f"Invalid native tensor metadata: {name}")
    if verify_files:
        for entry in manifest["tensors"].values():
            path = root / entry["file"]
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError(f"Native tensor file mismatch: {entry['file']}")
        for name, info in manifest.get("assets", {}).items():
            path = root / name
            if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
                raise ValueError(f"Native checkpoint asset mismatch: {name}")
        loader = manifest.get("loader")
        if loader:
            path = root / loader["file"]
            if path.stat().st_size != loader["bytes"] or digest(path) != loader["sha256"]:
                raise ValueError("Native checkpoint loader mismatch")
    return manifest


class NativeExpertGroups:
    """Grouped native BFP4 experts, uploaded group-by-group to bound host staging."""

    def __init__(self, group_dir, source_shape, group_size=16):
        self.group_dir = Path(group_dir)
        self.group_size = group_size
        experts, output_size, input_size = source_shape
        self.shape = (1, experts, input_size, output_size)

    def load_all(self, device):
        """Upload every expert once; return one stacked device tensor [1, E, K, N].

        Groups are uploaded one at a time and synchronized, bounding host DMA
        staging (monolithic multi-GiB uploads hang the P150 staging ring).
        """
        import ttnn

        groups = []
        for group in range((self.shape[1] + self.group_size - 1) // self.group_size):
            packed_group = ttnn.load_tensor(str(self.group_dir / f"{group:02d}.tensorbin"))
            groups.append(ttnn.to_device(packed_group, device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.synchronize_device(device)
        stacked = ttnn.concat(groups, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for packed_group in groups:
            ttnn.deallocate(packed_group)
        if tuple(stacked.shape) != self.shape:
            raise ValueError(f"stacked experts {tuple(stacked.shape)} != manifest {self.shape}")
        return stacked


class LazyNativeQwen35StateDict(Mapping):
    """Restore one remapped tensor at a time; never expand the whole model in RAM."""

    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        self.manifest = read_manifest(self.checkpoint)
        self._entries = self.manifest["tensors"]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def has_native_tensor(self, name):
        return self._entries[name]["storage"] == "ttnn-tile"

    def native_tensor(self, name, device):
        import ttnn

        entry = self._entries[name]
        if entry["storage"] != "ttnn-tile":
            raise TypeError(f"{name} is not stored as a native TTNN tensor")
        group_dir = self.checkpoint / "expert_groups" / Path(entry["file"]).stem
        if group_dir.is_dir():
            return NativeExpertGroups(group_dir, entry["source_shape"])
        raise FileNotFoundError(f"Grouped expert checkpoint is missing: {group_dir}")

    def __getitem__(self, name):
        import ttnn
        from safetensors import safe_open

        entry = self._entries[name]
        path = self.checkpoint / entry["file"]
        if entry["storage"] == "safetensors-lossless":
            with safe_open(str(path), framework="pt", device="cpu") as source:
                return source.get_tensor(name)
        native = ttnn.load_tensor(str(path))
        restored = ttnn.to_torch(native)
        orientation = entry["orientation"]
        if orientation == "input,output":
            return restored.T.contiguous()
        if orientation == "singleton,expert,input,output":
            return restored.squeeze(0).transpose(-2, -1).contiguous()
        raise ValueError(f"Unsupported native orientation for {name}: {orientation}")
