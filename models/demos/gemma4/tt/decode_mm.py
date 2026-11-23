# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit program configs for the decode weight matmuls (activation = one tile row, M <= 32).

Every output tile is reduced over the full K on one core in blocks of in0_block_w=2 with the same
compute config (HiFi2, FP32 dest, packer L1 accumulation) as ttnn.linear's default path, so outputs
are bit-identical to it. Larger K blocks change the FP32 accumulation order: measured on real
decode activations they flip isolated bf16 output bits and change long-context greedy streams.

At in0_block_w=2 the tall projections (MLP down, attention o_proj) were bound by the per-K-block
activation multicast handshake, not by DRAM. The rebuilt TTNN of this image (phase-2 p2b) adds
TT_MM1D_IN0_CHUNK_TILES to the in0-multicast 1D matmul: the activation is read and multicast C K
blocks per handshake while compute still consumes in0_block_w tiles per block (K blocking and
accumulation order unchanged). With it, spreading those projections over 60 cores (2 output tiles
each) takes MLP down from 285 to ~390 GB/s and o_proj from ~220-270 to ~370-390 GB/s (tools/pipe_bench.py).

``GEMMA4_DECODE_MM=0`` restores the default heuristic.
"""

import functools
import os

import ttnn

# Read by the TTNN matmul factory when a program is created (so before the first matmul): multicast
# the activation 16 tiles (8 K blocks at in0_block_w=2) per handshake. Data movement only.
os.environ.setdefault("TT_MM1D_IN0_CHUNK_TILES", "16")

# (K tiles, N tiles) -> (grid x, grid y, in0_block_w, per_core_N). in0_block_w stays 2: the default
# heuristic's value, and the only one that keeps the FP32 accumulation order (and so the bf16 output
# bits) of the default path on real activations.
_TABLE = {
    (120, 480): (11, 3, 2, 15),  # MLP gate / up        3840 -> 15360
    (480, 120): (11, 6, 2, 2),  # MLP down              15360 -> 3840
    (120, 256): (11, 6, 2, 4),  # sliding fused QKV    3840 -> 4096+2048+2048
    (120, 288): (11, 3, 2, 9),  # global fused QKV     3840 -> 8192+512+512
    (128, 120): (11, 6, 2, 2),  # sliding o_proj        4096 -> 3840
    (256, 120): (11, 6, 2, 2),  # global o_proj         8192 -> 3840
    # it-assistant drafter shapes (never on the target path; drafts only steer speculation, so any
    # K block is allowed): o_proj, MLP down, pre/post projections, q(+zero kv) projections.
    (128, 32): (8, 4, 8, 1),  # sliding o_proj          4096 -> 1024
    (256, 32): (11, 2, 8, 2),  # global o_proj / down   8192 -> 1024
    (240, 32): (11, 2, 8, 2),  # pre_projection         7680 -> 1024
    (32, 120): (8, 8, 8, 2),  # post_projection         1024 -> 3840
    (32, 256): (11, 6, 2, 4),  # sliding qkv / gate / up 1024 -> 8192
    (32, 288): (11, 3, 2, 9),  # global qkv             1024 -> 9216
}

# GEMMA4_DECODE_MM_DN=1 (needs a TTNN build with the per-worker dual-NoC in1 reader, image ttnn-dn*;
# ttnn/overlay branch dn): grids spanning all 10 rows, per-grid NoC-1 column masks, 16-slot weight CB.
# Measured 431-472 GB/s per shape (tools/dn_bench.py) vs 372-398 with _TABLE. Interleaved weights.
_DN_TABLE = {
    (120, 480): (10, 10, 2, 5),  # MLP gate / up
    (480, 120): (6, 10, 4, 2),  # MLP down
    (120, 256): (7, 10, 2, 4),  # sliding fused QKV
    (120, 288): (10, 10, 2, 3),  # global fused QKV
    (128, 120): (8, 8, 4, 2),  # sliding o_proj
    (256, 120): (7, 9, 4, 2),  # global o_proj
}
_DN_ENV = {
    "TT_MM1D_IN1_DUAL_NOC": "3",
    "TT_MM1D_IN1_NOC_COLS": "10x10:0000011111,7x10:1000101,6x10:100100,7x9:1100000,8x8:10010000",
    "TT_MM1D_IN1_CB_SLOTS": "16",
    "TT_MM1D_IN0_CHUNK_TILES": "64",
}
if os.environ.get("GEMMA4_DECODE_MM_DN") == "1":
    for _k, _v in _DN_ENV.items():
        os.environ[_k] = _v
    _TABLE.update(_DN_TABLE)


def enabled():
    return os.environ.get("GEMMA4_DECODE_MM", "1") == "1"


@functools.lru_cache(maxsize=None)
def _config(gx, gy, kb, pn):
    sub = max(d for d in (4, 3, 2, 1) if pn % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=kb,
        out_subblock_h=1,
        out_subblock_w=sub,
        per_core_M=1,
        per_core_N=pn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _fast_kb():
    """``GEMMA4_DECODE_MM_KB=N`` (non-exact): widen in0_block_w of the target shapes to the largest
    divisor of K tiles <= N. FP32 accumulation is kept; only the K-block order changes."""
    return int(os.environ.get("GEMMA4_DECODE_MM_KB", "0"))


def program_config(x, w):
    """Tuned config for ``x @ w`` when ``x`` is a single tile row, else None (default heuristic)."""
    if not enabled() or x.padded_shape[-2] > 32 or x.padded_shape[0] * x.padded_shape[1] != 1:
        return None
    key = (w.padded_shape[-2] // 32, w.padded_shape[-1] // 32)
    entry = _TABLE.get(key)
    if entry is None:
        return None
    gx, gy, kb, pn = entry
    n = _fast_kb()
    if n > kb:
        kb = max(d for d in range(1, n + 1) if key[0] % d == 0)
    return _config(gx, gy, kb, pn)


def linear(x, w, compute_kernel_config, memory_config=None):
    pc = program_config(x, w)
    if pc is None:
        return ttnn.linear(x, w, compute_kernel_config=compute_kernel_config, memory_config=memory_config)
    return ttnn.linear(
        x,
        w,
        compute_kernel_config=compute_kernel_config,
        program_config=pc,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )
