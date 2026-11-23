# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Shared helper of the per-token device ops (fast/decode_loop.py, fast/sampling.py): a trace-safe cached generic_op
program.  (The greedy argmax is fused into the LM head: fast/head.py.)
"""
from __future__ import annotations

import ttnn


class CachedProgram:
    """Program descriptor rebuilt only when the tensors' addresses / shapes or the extra key change (trace-safe)."""

    def __init__(self):
        self.program, self.signature = None, None

    def run(self, tensors, build, key=()):
        sig = (tuple((int(t.buffer_address()), tuple(t.shape), tuple(t.padded_shape)) for t in tensors), key)
        if self.program is None or sig != self.signature:
            self.program, self.signature = build(), sig
        ttnn.generic_op(list(tensors), self.program)
