# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Lean-L1 execution for the vision tower beside the native text runtime.

The native Qwen3.8-27B runtime keeps the GDN recurrent state (and sampler / draft buffers) resident in
L1, so every program it runs must keep its static circular buffers below that resident region (about
600 KB of the 1.5 MB per core). The vision tower's default matmul and norm configurations size their
buffers for an empty L1. With QWEN36_VISION_L1_LEAN=1 its matmuls run on small fixed blocks: rows are
processed MM_GRID_Y tiles at a time as a batch over the full width of the core grid, and each core holds
one row tile by a few dozen output tiles. SDPA runs on the whole core grid with 64-row Q / 128-row K
chunks (larger K chunks overrun the L1 budget or hang in the row-chunked path).
"""
import math
import os

import ttnn

LEAN = os.environ.get("QWEN36_VISION_L1_LEAN") == "1"
MM_GRID_Y = 8  # grid rows of the lean matmul (per_core_M = 1, so 256 rows per batch); measured best of 4, 8, 10
MAX_IN1_TILES = 40  # in0_block_w * per_core_N bound on the weight block (double-buffered)
# Frames longer than this run the attention projections, SDPA query blocks and the MLP in row chunks of
# this size (16,384-token images have 65,536 patch rows; their full-width QKV alone is 604 MB).
CHUNK_ROWS = 8192
# QWEN36_VISION_PRECISION: "bf16" keeps vision weights and the SDPA Q/K/V in BF16 (default BFP8 as upstream).
_PRECISION = os.environ.get("QWEN36_VISION_PRECISION", "bfp8")
WEIGHT_DTYPE = ttnn.bfloat16 if _PRECISION == "bf16" else None  # None: the module's upstream choice
QKV_DTYPE = ttnn.bfloat16 if _PRECISION == "bf16" else None
# QWEN36_VISION_HIFI=1: every vision matmul / SDPA in HiFi2 with FP32 accumulation (HiFi4 measured no more
# accurate against an FP32 reference tower with BFP8 operands, and 1.2x slower).
HIFI_CFG = (
    ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    if os.environ.get("QWEN36_VISION_HIFI") == "1"
    else None
)


def SDPA_CONFIG(device):
    """Non-causal vision SDPA over the device's whole core grid on 64-row Q / 128-row K chunks (head_dim 72
    padded to 96)."""
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(), q_chunk_size=64, k_chunk_size=128,
        exp_approx_mode=False,
    )


def linear(x, w, *, bias=None, activation=None, dtype=None, compute_kernel_config=None):
    """ttnn.linear over x [..., S, K] with lean program blocks.

    Returns [1, 1, S_total, N] in DRAM (rows are zero-padded to whole row batches for the matmul and
    sliced back). ``activation`` is applied as a separate op ("gelu" only).
    """
    k = x.shape[-1]
    rows = 1
    for i in range(len(x.shape) - 1):
        rows *= x.shape[i]
    grid = (x.device().compute_with_storage_grid_size().x, MM_GRID_Y)
    rows_per_batch = 32 * grid[1]
    padded = math.ceil(rows / rows_per_batch) * rows_per_batch
    n = w.shape[-1]
    k_tiles, n_tiles = k // 32, math.ceil(n / 32)
    per_core_n = math.ceil(n_tiles / grid[0])
    in0_block_w = next(d for d in (4, 3, 2, 1) if k_tiles % d == 0 and d * per_core_n <= MAX_IN1_TILES)
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    xb = ttnn.reshape(x, [1, 1, rows, k])
    if padded != rows:
        xb = ttnn.pad(xb, padding=((0, 0), (0, 0), (0, padded - rows), (0, 0)), value=0.0)
    xb = ttnn.reshape(xb, [1, padded // rows_per_batch, rows_per_batch, k])
    out = ttnn.linear(
        xb, w, bias=bias, dtype=dtype, program_config=cfg, compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.reshape(out, [1, 1, padded, n])
    if padded != rows:
        out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, rows, n])
    if activation == "gelu":
        act = ttnn.gelu(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        out = act
    elif activation is not None:
        raise ValueError(f"lean vision linear: unsupported activation {activation!r}")
    return out


def mm(x, w, *, bias=None, activation=None, dtype=None, compute_kernel_config=None,
       memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=None):
    """ttnn.linear, or the lean-block linear above when QWEN36_VISION_L1_LEAN=1."""
    compute_kernel_config = HIFI_CFG or compute_kernel_config
    if LEAN:
        return linear(x, w, bias=bias, activation=activation, dtype=dtype, compute_kernel_config=compute_kernel_config)
    return ttnn.linear(
        x, w, bias=bias, activation=activation, dtype=dtype, compute_kernel_config=compute_kernel_config,
        memory_config=memory_config, program_config=program_config,
    )
