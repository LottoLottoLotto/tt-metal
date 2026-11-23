# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import os


def optimized_27b_attention():
    return (
        os.environ.get("QWEN36_OPTIMIZED_27B") == "1"
        and os.environ.get("QWEN36_GENERIC_BASELINE") != "1"
    )


@dataclass(frozen=True)
class AttentionConfig:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    norm_eps: float
    max_seq_len: int

    @classmethod
    def from_args(cls, args) -> "AttentionConfig":
        return cls(
            num_heads=args.n_heads,
            num_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            norm_eps=args.norm_eps,
            max_seq_len=args.max_seq_len,
        )
