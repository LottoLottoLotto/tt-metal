# SPDX-License-Identifier: Apache-2.0
"""4-bit V cache (QWEN36_KV_CACHE_DTYPE=bfp8_v4r): K stays bfloat8_b; V is rotated by a randomized
Hadamard R (256x256, random signs, seed 20261002) and stored as bfloat4_b, rounded to nearest-even.
The attention output is rotated back by R^T per Q head before the sigmoid output gate
(softmax(QK^T) (V R) R^T = softmax(QK^T) V), so o_proj and the checkpoint stay unchanged.

One program family (v_rotate_kernel.cpp) does every rotation: X @ M for 256-wide head blocks, HiFi4
with FP32 accumulation, BF16 result, M = R or R^T held as bfloat4_b (every entry is +-1/16, exact).
Each output row depends only on its own input row and the fixed K order, so a token's rotated V and
an attention row's R^T product are bit-identical whichever path (prefill, M=1 decode, MTP/DFlash
verification) computes them.

V writes: the device packer TRUNCATES BF16 -> BFP4, so the rotated V is rounded on its BFP4 grid by
the kernel's writer (v_rotate_rne.hpp; round-to-nearest-even, bit-exact to ttquant.bfp.rtn) and then
stored exactly: by this kernel straight into the paged cache for prefill chunks, by
paged_update_cache for M=1 decode, and by rope_kv_many's packer conversion for verification rows.
"""

from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_KERNEL = str(Path(__file__).with_name("v_rotate_kernel.cpp").resolve())
SEED = 20261002
HEAD_DIM = 256
_TILE = 2048
_BFP4_TILE = 576
_QKV_V_TILE = 416  # fused Q+gate (12288) and K (1024) columns precede V: 13312 / 32
_MATRICES = {}


def rotation(n=HEAD_DIM, seed=SEED):
    """R = diag(random signs) @ H / sqrt(n): orthogonal, entries +-1/16 (Phase-1 kvquant rot.py)."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=generator).double() * 2 - 1
    return (signs[:, None] * h) / n**0.5


def matrices(device):
    """(R, R^T) as bfloat4_b [256,256] DRAM tiles; created once per device, before any trace capture."""
    key = id(device)
    if key not in _MATRICES:
        r = rotation().float()

        def upload(m):
            return ttnn.from_torch(m.contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

        _MATRICES[key] = (upload(r), upload(r.T))
    return _MATRICES[key]


def _shape(tensor, padded=False):
    return tuple(int(v) for v in (tensor.padded_shape if padded else tensor.shape))


def _check_bf16_tiles(tensor, name):
    if (tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT or tensor.tile != ttnn.Tile((32, 32))
            or tensor.is_sharded()):
        raise ValueError(f"v_rotate {name}: requires interleaved BF16 32x32 tiles")


def _is_dram(tensor):
    return "1" if tensor.memory_config().buffer_type == ttnn.BufferType.DRAM else "0"


def _run(source, matrix, target, *, groups, row_blocks, cols, cores, in_base, in_rb_stride, in_g_stride,
         out_rb_stride=0, out_g_stride=0, rounding=False, table=None, valid_row_blocks=0, round_rows=32, extra=()):
    device = source.device()
    grid = device.compute_with_storage_grid_size()
    if cores > grid.x * grid.y:
        raise ValueError("v_rotate needs more workers than the device grid has")
    items = row_blocks * groups * (8 // cols)
    core_list = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(cores)]
    core_set = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in core_list})
    base, rem = divmod(items, cores)
    tasks, start = [], 0
    for i, core in enumerate(core_list):
        n = base + (1 if i < rem else 0)
        tasks.append((core, [start, n]))
        start += n
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))

    def cb(index, tiles, dtype, page):
        return ttnn.CBDescriptor(total_size=tiles * page, core_ranges=core_set,
                                 format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

    in_tiles = 16 if cols == 8 else 8
    cbs = [cb(0, in_tiles, ttnn.bfloat16, _TILE), cb(1, 8 * cols, ttnn.bfloat4_b, _BFP4_TILE),
           cb(2, 4 if cols > 1 else 2, ttnn.bfloat16, _TILE)]
    defines = [("V_ROT_COLS", str(cols)), ("V_ROT_IN_DRAM", _is_dram(source))]
    table_bytes = read_bytes = 0
    if table is not None:
        table_bytes = int(table.padded_shape[-1]) * 4
        read_bytes = (table_bytes + 63) // 64 * 64
        if read_bytes > 1024:
            raise ValueError("v_rotate paged fill supports chunk page tables of at most 256 blocks")
        cbs += [cb(3, 2, ttnn.bfloat4_b, _BFP4_TILE),
                ttnn.CBDescriptor(total_size=1024, core_ranges=core_set,
                                  format_descriptors=[ttnn.CBFormatDescriptor(4, ttnn.int32, 1024)])]
        defines += [("V_ROT_PAGED", "1"), ("V_ROT_PAGE_ROWS", str(int(target.shape[2]))),
                    ("V_ROT_TABLE_DRAM", _is_dram(table))]
    else:
        defines.append(("V_ROT_OUT_DRAM", _is_dram(target)))
        if rounding:
            defines.append(("V_ROT_ROUND", "1"))
    common = [int(source.buffer_address()), int(matrix.buffer_address()), int(target.buffer_address()),
              int(table.buffer_address()) if table is not None else 0, groups, in_base, in_rb_stride, in_g_stride,
              out_rb_stride, out_g_stride, valid_row_blocks, table_bytes, read_bytes, round_rows]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=core_set,
        defines=defines,
        ncrisc_common_runtime_args=common,
        brisc_common_runtime_args=common,
        trisc_common_runtime_args=common,
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
            dst_full_sync_en=False),
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            ncrisc_args=tasks, brisc_args=tasks, trisc_args=tasks),
    )
    program = ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])
    io = [source, matrix, target] + ([table] if table is not None else []) + list(extra)
    ttnn.generic_op(io, program)


def _workers(device):
    grid = device.compute_with_storage_grid_size()
    return int(grid.x) * int(grid.y)


def rotate_v_rows(qkv, rounding=True):
    """Verification / M=1 decode rows: V columns of the fused QKV projection [1,T,14336] (T <= 32)
    -> rotated V [1,T,1024] (token rows, head h in columns 256h..), BF16 values on their BFP4 grid
    (rows >= T of the padded tile are not rounded and never read)."""
    if _shape(qkv)[-1] != 14336 or len(_shape(qkv)) != 3 or _shape(qkv, True)[-2] != 32:
        raise ValueError("rotate_v_rows expects the fused QKV projection [1,T<=32,14336]")
    _check_bf16_tiles(qkv, "qkv")
    r, _ = matrices(qkv.device())
    out = ttnn.empty((1, _shape(qkv)[1], 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=qkv.device(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _run(qkv, r, out, groups=4, row_blocks=1, cols=1, cores=32, in_base=_QKV_V_TILE, in_rb_stride=448,
         in_g_stride=8, out_rb_stride=32, out_g_stride=8, rounding=rounding, round_rows=_shape(qkv)[1])
    return out


def fill_v_paged(qkv, cache_value, chunk_page_table, page_len):
    """Prefill chunk: V columns of the fused QKV projection [1,T,14336] (T % 32 == 0) -> rotated,
    RNE-rounded BFP4 tiles written straight into the paged V cache through the chunk page table
    (rows >= page_len are not written; -1 table entries are skipped), replacing typecast+fill."""
    shape = _shape(qkv)
    if len(shape) != 3 or shape[0] != 1 or shape[2] != 14336 or shape[1] % 32 or _shape(qkv, True) != shape:
        raise ValueError("fill_v_paged expects a tile-aligned fused QKV projection [1,T,14336]")
    _check_bf16_tiles(qkv, "qkv")
    cache_shape = _shape(cache_value)
    if (cache_value.dtype != ttnn.bfloat4_b or len(cache_shape) != 4 or cache_shape[1] != 4
            or cache_shape[3] != HEAD_DIM or cache_shape[2] % 32
            or cache_value.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
        raise ValueError("fill_v_paged expects a DRAM bfloat4_b paged V cache [blocks,4,page,256]")
    if (chunk_page_table.dtype != ttnn.int32 or len(chunk_page_table.shape) != 2
            or int(chunk_page_table.shape[0]) != 1 or chunk_page_table.layout != ttnn.ROW_MAJOR_LAYOUT):
        raise ValueError("fill_v_paged expects an int32 row-major chunk page table [1, blocks]")
    if page_len % 32 or not 0 < page_len <= shape[1] or page_len > int(chunk_page_table.shape[1]) * cache_shape[2]:
        raise ValueError("fill_v_paged page_len must be tile-aligned and covered by the input and page table")
    r, _ = matrices(qkv.device())
    _run(qkv, r, cache_value, groups=4, row_blocks=shape[1] // 32, cols=8, cores=_workers(qkv.device()),
         in_base=_QKV_V_TILE, in_rb_stride=448, in_g_stride=8, table=chunk_page_table,
         valid_row_blocks=page_len // 32)


def unrotate_heads_major(attn):
    """Prefill: chunked SDPA output [1,24,T,256] (T % 32 == 0) -> (attn @ R^T) as the token-major
    [1,T,6144] row block the gate multiply consumes (also replaces the head relayout)."""
    shape = _shape(attn)
    if len(shape) != 4 or shape[0] != 1 or shape[3] != HEAD_DIM or shape[2] % 32 or _shape(attn, True) != shape:
        raise ValueError("unrotate_heads_major expects a tile-aligned [1,H,T,256] attention output")
    _check_bf16_tiles(attn, "attn")
    heads, tokens = shape[1], shape[2]
    _, rt = matrices(attn.device())
    out = ttnn.empty((1, tokens, heads * HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=attn.device(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _run(attn, rt, out, groups=heads, row_blocks=tokens // 32, cols=8, cores=_workers(attn.device()),
         in_base=0, in_rb_stride=8, in_g_stride=(tokens // 32) * 8, out_rb_stride=heads * 8, out_g_stride=8)
    return out


def unrotate_rows(attn):
    """Decode / verification: attention rows [..., R, 24*256] (token-major, R <= 32 rows, e.g. the
    verify SDPA's [1,T,6144]) or the paged decode SDPA's [1,1,24,256] (rows = Q heads) -> attn @ R^T
    per 256-column head block, same shape."""
    shape, padded = _shape(attn), _shape(attn, True)
    if padded[-2] != 32 or shape[-1] % HEAD_DIM or any(d != 1 for d in padded[:-2]):
        raise ValueError("unrotate_rows expects one 32-row tile row of 256-wide head blocks")
    _check_bf16_tiles(attn, "attn")
    groups = shape[-1] // HEAD_DIM
    _, rt = matrices(attn.device())
    out = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=attn.device(),
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cols = 1 if groups <= 4 else 2
    items = groups * (8 // cols)
    _run(attn, rt, out, groups=groups, row_blocks=1, cols=cols, cores=min(items, _workers(attn.device())),
         in_base=0, in_rb_stride=groups * 8, in_g_stride=8, out_rb_stride=groups * 8, out_g_stride=8)
    return out
