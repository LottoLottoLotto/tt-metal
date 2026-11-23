# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zero-centered RMSNorm for Qwen3.5 (weights pre-offset by +1; fused ttnn.rms_norm)."""
import math
import os
from functools import lru_cache

import ttnn

from models.tt_transformers.tt.common import Mode


def bounded_norm_enabled():
    return (
        os.environ.get("QWEN36_OPTIMIZED_27B") == "1"
        and os.environ.get("QWEN36_GENERIC_BASELINE") != "1"
    )


@lru_cache(maxsize=9)
def _bounded_norm_config(rows):
    """One tile row per core; never stage a full 5120-wide row on a core.

    Decode uses 40 width shards (160 tiles / 40 = 4), not the 9B
    64-core configuration, which cannot divide the 27B hidden width.
    Prefill uses up to eight independent rows of eight width shards.
    """
    if not 1 <= rows <= 256:
        raise ValueError(f"Bounded RMSNorm expects 1..256 rows, got {rows}")
    if rows <= 32:
        grid = ttnn.CoreGrid(x=8, y=5)
        shard_width = 128
        strategy = ttnn.ShardStrategy.WIDTH
    else:
        grid = ttnn.CoreGrid(x=8, y=(rows + 31) // 32)
        shard_width = 640
        strategy = ttnn.ShardStrategy.BLOCK
    shard_config = ttnn.create_sharded_memory_config(
        shape=(32, shard_width),
        core_grid=grid,
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return {
        "sharded_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=4,
            block_h=1,
            block_w=shard_width // 32,
            inplace=False,
        ),
        "sharded_output_config": shard_config,
        "output_mem_config": ttnn.DRAM_MEMORY_CONFIG,
    }


def get_decode_norm_config(args):
    """Framework-compatible config for M=1/M=2, including fused residual RMSNorm."""
    if bounded_norm_enabled() and args.dim == 5120 and getattr(args, "num_devices", 1) == 1:
        return dict(_bounded_norm_config(1))
    return args.get_norm_config("lm_head", Mode.DECODE)


def bounded_rms_norm(x, weight, eps, compute_kernel_config):
    """Bound every full-27B interleaved framework norm, including prefill/MTP.

    The pinned native sharded RMSNorm's static CB allocations are 114688 B
    per core for decode and 262144 B for a prefill chunk. BF16 sharded
    input/output add 16384/81920 B respectively; fused decode adds another
    8192 B for the residual. These are CB/tensor bytes, not absolute L1
    addresses: firmware/program reservation must be budgeted separately.

    Source: layernorm/device/sharded_layernorm_factory_helpers.cpp,
    CBSizeParams::compute and add_cb_descriptors. For block_h=1:
      gamma + 2*x + scalar/epsilon/global-scaler + 4*statistics + external
    Decode: 8192 + 32768 + 8192 + 16384 + (8+5-1)*4096.
    Prefill: 40960 + 163840 + 8192 + 16384 + 8*4096.
    Chunking keeps this bound independent of prefill length (128..2048).
    The pinned Blackhole default CB origin is 111488 B: MEM_MAP_END=40832
    (dev_mem_map.h) plus 69 KiB kernel config (bh_hal_tensix.cpp). Thus the
    maximum norm working-set reservation including origin and BF16 tensors
    is 455552 B/core, leaving 367744 B with 749568 B resident GDN state
    under physical L1=1572864 B. Nondefault kernel reservations must be
    accounted for separately by the caller's capacity guard.
    Weight is already pre-offset by framework RMSNorm; never add one here.
    """
    original_shape = tuple(x.shape)
    rows = math.prod(original_shape[:-1])
    flat = ttnn.reshape(x, (1, 1, rows, 5120))
    outputs = []
    for start in range(0, rows, 256):
        chunk_rows = min(256, rows - start)
        chunk = (
            flat
            if rows <= 256
            else ttnn.slice(
                flat,
                (0, 0, start, 0),
                (1, 1, start + chunk_rows, 5120),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        config = _bounded_norm_config(chunk_rows)
        sharded = ttnn.to_memory_config(chunk, config["sharded_output_config"])
        if rows > 256:
            ttnn.deallocate(chunk)
        normed = ttnn.rms_norm(
            sharded,
            weight=weight,
            epsilon=eps,
            program_config=config["sharded_program_config"],
            memory_config=config["sharded_output_config"],
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.deallocate(sharded)
        outputs.append(ttnn.sharded_to_interleaved(normed, ttnn.DRAM_MEMORY_CONFIG))
        ttnn.deallocate(normed)
    if len(outputs) == 1:
        output = outputs[0]
    else:
        output = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in outputs:
            ttnn.deallocate(chunk)
    return ttnn.reshape(output, original_shape)


def rms_norm_ttnn(x, weight, eps=1e-6, memory_config=None):
    """Zero-centered RMSNorm using fused ttnn.rms_norm.

    Qwen3.5 uses zero-centered RMSNorm for ALL layer norms:
      output = x * rsqrt(mean(x^2) + eps) * (1 + weight)

    The weight should be pre-offset by +1 so we can use the standard fused op.
    Verified: fused ttnn.rms_norm works for all decode shapes on Blackhole P150:
      [1, 1, 4096], [1, 1, 16, 256], [1, 1, 32, 128]
    """
    return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=memory_config)
