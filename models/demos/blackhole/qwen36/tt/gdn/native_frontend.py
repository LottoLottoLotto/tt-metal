# SPDX-License-Identifier: Apache-2.0
"""P150 GDN frontend: source-owned GQA FIR and T1..T32 state transitions.

One worker per source head (``2*H`` key/query heads + ``HV`` value heads, 128 wide each)
reads its four FIR taps, the three-row L1 history and the current mega-projection row,
emits SiLU(conv) scattered into head-major q/k/v ``[T, 1, HV, 128]`` (GQA copies written
by the owning key head), and advances the history. Worker 0 also extracts the raw a/b
scalar rows. The geometry is taken from the tensors: 27B is H=16, HV=48, W=10240.
"""
from dataclasses import dataclass

import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_TILE_BYTES = 2048
_HEAD_DIM = 128
_KERNEL = "models/demos/blackhole/qwen36/tt/gdn/native_frontend_kernel.cpp"
_GRID_X = 10


@dataclass(frozen=True)
class FrontendGeometry:
    key_heads: int
    value_heads: int

    @property
    def gqa(self):
        return self.value_heads // self.key_heads

    @property
    def workers(self):
        return 2 * self.key_heads + self.value_heads

    @property
    def qkv_width(self):
        return self.workers * _HEAD_DIM

    @property
    def scalar_width(self):
        """a/b segments of the mega projection, each padded to whole tiles."""
        return ((self.value_heads + 31) // 32) * 32

    @property
    def mega_width(self):
        return self.qkv_width + 2 * self.scalar_width + self.value_heads * _HEAD_DIM


GEOMETRY_27B = FrontendGeometry(key_heads=16, value_heads=48)


def _core(worker):
    return ttnn.CoreCoord(worker % _GRID_X, worker // _GRID_X)


def _core_set(geometry):
    rows = (geometry.workers + _GRID_X - 1) // _GRID_X
    ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_GRID_X - 1, rows - 2))] if rows > 1 else []
    last = geometry.workers - (rows - 1) * _GRID_X
    ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, rows - 1), ttnn.CoreCoord(last - 1, rows - 1)))
    return ttnn.CoreRangeSet(set(ranges))


@dataclass
class NativeFrontend:
    q: ttnn.Tensor
    k: ttnn.Tensor
    v: ttnn.Tensor
    a: ttnn.Tensor
    b: ttnn.Tensor
    zero_bias: ttnn.Tensor
    geometry: FrontendGeometry
    conv_s1: ttnn.Tensor | None = None
    conv_s2: ttnn.Tensor | None = None
    conv_candidates: tuple = ()
    program: object = None
    program_signature: tuple[int, ...] | None = None
    fork_position: int = 0


def _allocate(device, num_tokens, scratch=None, geometry=GEOMETRY_27B):
    def zeros(shape, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if not 1 <= num_tokens <= 32:
        raise ValueError("native frontend supports one tile row (1..32 tokens)")
    hv, width = geometry.value_heads, geometry.qkv_width
    scalar_shape = (1, 1, hv) if num_tokens == 1 else (1, 1, num_tokens, hv)
    candidates = tuple(
        zeros((1, 3, width), ttnn.ROW_MAJOR_LAYOUT) for _ in range(num_tokens)
    ) if num_tokens > 1 else ()
    return NativeFrontend(
        q=scratch.q if scratch is not None else zeros((num_tokens, 1, hv, _HEAD_DIM)),
        k=scratch.k if scratch is not None else zeros((num_tokens, 1, hv, _HEAD_DIM)),
        v=scratch.v if scratch is not None else zeros((num_tokens, 1, hv, _HEAD_DIM)),
        a=scratch.a if scratch is not None else zeros(scalar_shape),
        b=scratch.b if scratch is not None else zeros(scalar_shape),
        zero_bias=scratch.zero_bias if scratch is not None else zeros((1, 1, width)),
        geometry=geometry,
        conv_s1=candidates[0] if candidates else None,
        conv_s2=candidates[1] if candidates else None,
        conv_candidates=candidates,
    )


def allocate_native_frontend(device, geometry=GEOMETRY_27B):
    return _allocate(device, 1, geometry=geometry)


def allocate_native_frontend_m2(device, geometry=GEOMETRY_27B):
    return _allocate(device, 2, geometry=geometry)


def allocate_native_frontend_many(device, num_tokens, scratch=None, geometry=GEOMETRY_27B):
    """Share sequential-layer operands, but retain each layer's prefix histories."""
    if not 2 <= num_tokens <= 32:
        raise ValueError("verification requires 2..32 rows")
    return _allocate(device, num_tokens, scratch, geometry)


def _staging_bytes(num_tokens):
    """Kernel staging (CB 3): 3 + T row-major source rows (256 B), each dim tile's face-row
    blocks of the T token rows (64 B-aligned), 4 taps + bias (4 x 128 B each), a zero row."""
    top = 32 * ((min(num_tokens, 16) + 1) & ~1)
    bottom = 32 * ((max(num_tokens - 16, 0) + 1) & ~1)
    return (3 + num_tokens) * 256 + 4 * 2 * (top + bottom) + 5 * 4 * 128 + 64


def _cb_descriptors(geometry, num_tokens):
    cores = _core_set(geometry)
    full = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    fmt = lambda cb: ttnn.CBFormatDescriptor(cb, ttnn.bfloat16, _TILE_BYTES, full)
    return [
        *[ttnn.CBDescriptor(total_size=count * _TILE_BYTES, core_ranges=cores,
                           format_descriptors=[fmt(cb)])
          for cb, count in ((0, 4), (1, 5), (2, 1), (16, 1), (17, 1))],
        ttnn.CBDescriptor(total_size=_staging_bytes(num_tokens), core_ranges=cores, format_descriptors=[
            ttnn.CBFormatDescriptor(3, ttnn.bfloat16, 64, ttnn.TileDescriptor(ttnn.Tile((1, 32))))
        ]),
    ]


def _make_program(device, mega, taps, bias, conv_state, output):
    geometry = output.geometry
    logical = ttnn.corerange_to_cores(conv_state.memory_config().shard_spec.grid, row_wise=True)
    if len(logical) != 1 or tuple(conv_state.memory_config().shard_spec.shape) != (3, geometry.qkv_width):
        raise RuntimeError(f"native GDN frontend requires one L1 convolution shard [3,{geometry.qkv_width}]")
    state_core = device.worker_core_from_logical_core(logical[0])
    token_args = [("num_tokens", int(output.q.shape[0])), ("fork_position", output.fork_position)]
    geometry_args = [
        ("qkv_width", geometry.qkv_width),
        ("key_heads", geometry.key_heads),
        ("gqa", geometry.gqa),
        ("value_heads", geometry.value_heads),
        ("scalar_tiles", geometry.scalar_width // 32),
    ]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=_core_set(geometry),
        ncrisc_named_compile_time_args=[*token_args, *geometry_args],
        brisc_named_compile_time_args=[
            *token_args,
            *geometry_args,
            ("mega_is_dram", int(mega.memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("tap_is_dram", int(taps[0].memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("bias_is_dram", int(bias.memory_config().buffer_type == ttnn.BufferType.DRAM)),
        ],
        trisc_named_compile_time_args=[*token_args, *geometry_args],
        brisc_common_runtime_args=[
            int(mega.buffer_address()), *[int(t.buffer_address()) for t in taps],
            int(bias.buffer_address()), int(conv_state.buffer_address()), state_core.x, state_core.y,
            int(output.a.buffer_address()), int(output.b.buffer_address()),
            int(output.q.buffer_address()), int(output.k.buffer_address()), int(output.v.buffer_address()),
            *([int(t.buffer_address()) for t in output.conv_candidates] or [0, 0]),
        ],
        ncrisc_common_runtime_args=[],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=False, dst_full_sync_en=False,
        ),
        unified_compile_time_core_descriptors=[],
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            brisc_args=[(_core(i), [i]) for i in range(geometry.workers)]
        ),
    )
    descriptors = kernel.get_kernel_descriptors()
    return ttnn.ProgramDescriptor(
        kernels=descriptors.kernels, cbs=_cb_descriptors(geometry, int(output.q.shape[0])), semaphores=[]
    )


def native_frontend(mega, taps, bias, conv_state, output):
    geometry = output.geometry
    width, mega_width = geometry.qkv_width, geometry.mega_width
    num_tokens = int(output.q.shape[0])
    if not 1 <= num_tokens <= 32 or tuple(mega.shape) not in (
        (1, num_tokens, mega_width), (1, 1, num_tokens, mega_width)
    ):
        raise RuntimeError(f"native GDN frontend requires {num_tokens} packed {mega_width}-wide rows, got {mega.shape}")
    if len(taps) != 4 or any(tuple(t.shape) != (1, 1, width) for t in taps):
        raise RuntimeError(f"native GDN frontend requires four flat BF16 tiled taps [1,1,{width}]")
    if bias is None:
        bias = output.zero_bias
    if tuple(bias.shape) != (1, 1, width):
        raise RuntimeError(f"native GDN frontend requires flat bias [1,1,{width}]")
    if (tuple(conv_state.shape) != (1, 3, width) or conv_state.layout != ttnn.ROW_MAJOR_LAYOUT
            or conv_state.dtype != ttnn.bfloat16 or not conv_state.is_sharded()
            or conv_state.memory_config().buffer_type != ttnn.BufferType.L1):
        raise RuntimeError(f"native GDN frontend requires BF16 row-major L1 history [1,3,{width}]")
    tensors = [mega, *taps, bias, conv_state, output.q, output.k, output.v, output.a, output.b]
    tensors.extend(output.conv_candidates)
    signature = tuple(int(t.buffer_address()) for t in tensors)
    # Each layer owns exactly one descriptor per token count, not an address-keyed cache.
    if output.program is None or output.program_signature != signature:
        output.program = _make_program(mega.device(), mega, taps, bias, conv_state, output)
        output.program_signature = signature
    ttnn.generic_op(tensors, output.program)
    return output.q, output.k, output.v, output.a, output.b
