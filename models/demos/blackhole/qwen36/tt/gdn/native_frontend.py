# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape P150 batch-one GDN convolution frontend."""
from dataclasses import dataclass

import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

_GRID_X = 4
_GRID_Y = 3
_TILE_BYTES = 2048
_KERNEL = "models/demos/blackhole/qwen36/tt/gdn/native_frontend_kernel.cpp"


def _core(worker, num_tokens):
    width = _GRID_X * num_tokens
    return ttnn.CoreCoord(worker % width, worker // width)


def _core_set(num_tokens):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_GRID_X * num_tokens - 1, _GRID_Y - 1))})


def _one_core(core):
    return ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})


@dataclass
class NativeFrontend:
    q: ttnn.Tensor
    k: ttnn.Tensor
    v: ttnn.Tensor
    a: ttnn.Tensor
    b: ttnn.Tensor
    zero_bias: ttnn.Tensor
    conv_s1: ttnn.Tensor | None = None
    conv_s2: ttnn.Tensor | None = None
    program: object = None
    program_signature: tuple[int, ...] | None = None


def allocate_native_frontend(device):
    def zeros(shape, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

    return NativeFrontend(
        q=zeros((1, 1, 32, 128)),
        k=zeros((1, 1, 32, 128)),
        v=zeros((1, 1, 32, 128)),
        a=zeros((1, 1, 32)),
        b=zeros((1, 1, 32)),
        zero_bias=zeros((1, 1, 384, 32), ttnn.DRAM_MEMORY_CONFIG),
    )


def _cb_descriptors(num_tokens):
    cores = _core_set(num_tokens)
    full = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    fmt = lambda cb: ttnn.CBFormatDescriptor(cb, ttnn.bfloat16, _TILE_BYTES, full)
    return [
        ttnn.CBDescriptor(total_size=4 * _TILE_BYTES, core_ranges=cores, format_descriptors=[fmt(0)]),
        ttnn.CBDescriptor(total_size=5 * _TILE_BYTES, core_ranges=cores, format_descriptors=[fmt(1)]),
        ttnn.CBDescriptor(total_size=_TILE_BYTES, core_ranges=cores, format_descriptors=[fmt(2)]),
        ttnn.CBDescriptor(
            total_size=128,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(3, ttnn.bfloat16, 64, ttnn.TileDescriptor(ttnn.Tile((1, 32))))
            ],
        ),
        ttnn.CBDescriptor(total_size=_TILE_BYTES, core_ranges=cores, format_descriptors=[fmt(16)]),
        ttnn.CBDescriptor(total_size=_TILE_BYTES, core_ranges=cores, format_descriptors=[fmt(17)]),
    ]


def _state_physical_cores(device, conv_state):
    logical = ttnn.corerange_to_cores(conv_state.memory_config().shard_spec.grid, row_wise=True)
    if len(logical) != 3:
        raise RuntimeError(f"native GDN frontend requires three convolution-state cores, got {len(logical)}")
    return [device.worker_core_from_logical_core(core) for core in logical]


def _make_program(device, mega, taps, bias, conv_state, output):
    num_tokens = int(output.q.shape[0])
    cores = _core_set(num_tokens)
    state_cores = _state_physical_cores(device, conv_state)
    per_core_kind = [(_core(i, num_tokens), (i % 12) // 4) for i in range(12 * num_tokens)]
    per_core_dim_tile = [(_core(i, num_tokens), i % 4) for i in range(12 * num_tokens)]
    per_core_token = [(_core(i, num_tokens), i // 12) for i in range(12 * num_tokens)]

    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        ncrisc_named_compile_time_args=[("num_tokens", int(output.q.shape[0]))],
        brisc_named_compile_time_args=[
            ("num_tokens", int(output.q.shape[0])),
            ("mega_is_dram", int(mega.memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("tap_is_dram", int(taps[0].memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("bias_is_dram", int(bias.memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("output_is_dram", int(output.q.memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("candidate_is_dram", int(
                output.conv_s1 is not None
                and output.conv_s1.memory_config().buffer_type == ttnn.BufferType.DRAM
            )),
        ],
        trisc_named_compile_time_args=[("num_tokens", int(output.q.shape[0]))],
        brisc_common_runtime_args=[
            int(mega.buffer_address()),
            *[int(t.buffer_address()) for t in taps],
            int(bias.buffer_address()),
            int(conv_state.buffer_address()),
            *[coord.x for coord in state_cores],
            *[coord.y for coord in state_cores],
            int(output.a.buffer_address()),
            int(output.b.buffer_address()),
            int(output.q.buffer_address()),
            int(output.k.buffer_address()),
            int(output.v.buffer_address()),
            int(output.conv_s1.buffer_address()) if output.conv_s1 is not None else 0,
            int(output.conv_s2.buffer_address()) if output.conv_s2 is not None else 0,
        ],
        ncrisc_common_runtime_args=[],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        unified_compile_time_core_descriptors=[],
        per_core_compile_time_descriptors=[
            PerCoreCompileTimeDescriptor("output_kind", per_core_kind, 1),
            PerCoreCompileTimeDescriptor("dim_tile", per_core_dim_tile, 1),
            PerCoreCompileTimeDescriptor("token_index", per_core_token, 1),
        ],
    )
    descriptors = kernel.get_kernel_descriptors()
    return ttnn.ProgramDescriptor(kernels=descriptors.kernels, cbs=_cb_descriptors(num_tokens), semaphores=[])


def native_frontend(mega, taps, bias, conv_state, output):
    num_tokens = int(output.q.shape[0])
    expected_mega_shapes = {
        1: ((1, 1, 12352), (1, 1, 1, 12352)),
        2: ((1, 2, 12352), (1, 1, 2, 12352)),
    }
    if num_tokens not in expected_mega_shapes or tuple(int(v) for v in mega.shape) not in expected_mega_shapes[num_tokens]:
        raise RuntimeError(f"native GDN frontend requires {num_tokens} packed decode rows, got {mega.shape}")
    if len(taps) != 4:
        raise RuntimeError("native GDN frontend requires four fused taps")
    expected_packed_shape = (1, 1, 384, 32)
    if any(tuple(int(v) for v in tap.shape) != expected_packed_shape for tap in taps):
        raise RuntimeError("native GDN frontend requires four pre-expanded 12-tile taps")
    if bias is not None and tuple(int(v) for v in bias.shape) != expected_packed_shape:
        raise RuntimeError("native GDN frontend requires a pre-expanded 12-tile bias")
    if bias is None:
        bias = output.zero_bias
    program_tensors = [
        mega,
        *taps,
        bias,
        conv_state,
        output.a,
        output.b,
        output.q,
        output.k,
        output.v,
    ]
    if output.conv_s1 is not None:
        program_tensors.extend([output.conv_s1, output.conv_s2])
    signature = tuple(int(tensor.buffer_address()) for tensor in program_tensors)
    if output.program is None or output.program_signature != signature:
        output.program = _make_program(mega.device(), mega, taps, bias, conv_state, output)
        output.program_signature = signature
    tensors = [mega, *taps, bias, conv_state, output.q, output.k, output.v, output.a, output.b]
    if output.conv_s1 is not None:
        tensors.append(output.conv_s1)
    if output.conv_s2 is not None:
        tensors.append(output.conv_s2)
    ttnn.generic_op(tensors, output.program)
    return output.q, output.k, output.v, output.a, output.b


def allocate_native_frontend_m2(device, conv_state):
    def zeros(shape, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
            memory_config=memory_config,
        )

    return NativeFrontend(
        q=zeros((2, 1, 32, 128)),
        k=zeros((2, 1, 32, 128)),
        v=zeros((2, 1, 32, 128)),
        a=zeros((1, 1, 2, 32)),
        b=zeros((1, 1, 2, 32)),
        zero_bias=zeros((1, 1, 384, 32)),
        conv_s1=zeros(tuple(int(v) for v in conv_state.shape), ttnn.ROW_MAJOR_LAYOUT),
        conv_s2=zeros(
            tuple(int(v) for v in conv_state.shape),
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
