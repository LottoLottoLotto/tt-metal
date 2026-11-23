# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for slicing nested state dicts."""
from __future__ import annotations
from collections.abc import Mapping
import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class PrefixState(Mapping):
    """Lazy view over one state-dict prefix."""

    def __init__(self, state, prefix):
        self.state = state
        self.prefix = f"{prefix}."
        self._keys = tuple(
            key[len(self.prefix) :] for key in state.keys() if key.startswith(self.prefix)
        )
        self._key_set = frozenset(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, key):
        if key not in self._key_set:
            raise KeyError(key)
        return self.state[self.prefix + key]

    def __contains__(self, key):
        return key in self._key_set

    def native_tensor(self, key, device):
        loader = getattr(self.state, "native_tensor", None)
        if loader is None:
            raise TypeError("Underlying state does not expose native tensors")
        return loader(self.prefix + key, device)

    def has_native_tensor(self, key):
        checker = getattr(self.state, "has_native_tensor", None)
        return checker is not None and checker(self.prefix + key)


def substate(state: dict[str, "torch.Tensor"], key: str) -> dict[str, "torch.Tensor"]:
    """Return a subtree while preserving lazy checkpoint access."""
    if not isinstance(state, dict):
        return PrefixState(state, key)
    prefix = f"{key}."
    prefix_len = len(prefix)
    return {k[prefix_len:]: state[k] for k in state.keys() if k.startswith(prefix)}


def has_substate(state: dict[str, "torch.Tensor"], key: str) -> bool:
    """True if any key starts with `key.`."""
    prefix = f"{key}."
    return any(k.startswith(prefix) for k in state)


def indexed_substates(state: dict[str, "torch.Tensor"], key: str) -> list[dict[str, "torch.Tensor"]]:
    """Extract a list of indexed sub-states (e.g. `key.0`, `key.1`, ...)."""
    result = []
    for i in itertools.count():
        s = substate(state, f"{key}.{i}")
        if not s:
            return result
        result.append(s)
    return []
