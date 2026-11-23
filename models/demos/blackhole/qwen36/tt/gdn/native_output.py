# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape P150 batch-one GDN output preparation."""
from dataclasses import dataclass

import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

_GRID_X = 8
_GRID_Y = 6
_TILE_BYTES = 2048
_KERNEL = "models/demos/blackhole/qwen36/tt/gdn/native_output_kernel.cpp"


def _core(worker):
    return ttnn.CoreCoord(worker % _GRID_X, worker // _GRID_X)


def _core_set():
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_GRID_X - 1, _GRID_Y - 1))})


@dataclass
class NativeOutput:
    value: ttnn.Tensor
    program: object = None
    input_addresses: tuple[int, int, int] | None = None


def allocate_native_output(device):
    value = ttnn.from_torch(
        torch.zeros((1, 1, 6144), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return NativeOutput(value=value)


def _cb_descriptors():
    cores = _core_set()
    tile = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    fmt = lambda cb: ttnn.CBFormatDescriptor(cb, ttnn.bfloat16, _TILE_BYTES, tile)
    # Liveness aliases preserve 24 KiB/core: x→clipped gate, gamma→output,
    # and RMS square→fusion. The aliases never overlap in execution.
    four_tile_aliases = ((0, 12), (1,), (2, 16), (7, 10), (8,))
    one_tile_cbs = (3, 4, 5, 6, 9, 11, 15)
    return [
        *[
            ttnn.CBDescriptor(
                total_size=4 * _TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[fmt(cb) for cb in aliases],
            )
            for aliases in four_tile_aliases
        ],
        *[
            ttnn.CBDescriptor(
                total_size=_TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[fmt(cb)],
            )
            for cb in one_tile_cbs
        ],
    ]


def _token_layout(value):
    """Return token count and packing for [T,1,6144] or [1,T,6144]."""
    shape = tuple(int(v) for v in value.shape)
    if shape[0] == 1 and shape[-1] == 6144 and len(shape) == 3:
        return shape[1], shape[1] > 1
    return shape[0], False


def _make_program(recurrent, gate, gamma, output):
    cores = _core_set()
    heads = [(_core(i), [i]) for i in range(48)]
    compute_args = [(_core(i), [1]) for i in range(48)]
    num_tokens, packed_rows = _token_layout(output.value)
    token_args = [("num_tokens", num_tokens), ("packed_rows", int(packed_rows))]
    cb_args = [
        ("cb_scaler", 3), ("cb_eps", 4), ("cb_in", 0), ("cb_inb", 15),
        ("cb_out", 8), ("cb_gamma", 2), ("cb_beta", 15), ("cb_xmm", 0),
        ("cb_ex", 5), ("cb_ex2", 6), ("cb_xmm2", 7), ("cb_ex2pe", 9),
        ("cb_fusion", 10), ("cb_in_rm", 15),
    ]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL,
        core_ranges=cores,
        ncrisc_named_compile_time_args=token_args,
        brisc_named_compile_time_args=[
            ("recurrent_is_dram", int(recurrent.memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("gate_is_dram", int(gate.memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("gamma_is_dram", int(gamma.memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("output_is_dram", int(output.value.memory_config().buffer_type == ttnn.BufferType.DRAM)),
            *token_args,
        ],
        trisc_compile_time_args=[4, 4, 1, 0, 0, 0, 0, 128, 32],
        trisc_named_compile_time_args=[*cb_args, *token_args],
        brisc_common_runtime_args=[
            int(recurrent.buffer_address()),
            int(gate.buffer_address()),
            int(gamma.buffer_address()),
            int(output.value.buffer_address()),
        ],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
            brisc_args=heads, trisc_args=compute_args
        ),
        defines=[("RMSNORM", "1")],
    )
    descriptors = kernel.get_kernel_descriptors()
    return ttnn.ProgramDescriptor(kernels=descriptors.kernels, cbs=_cb_descriptors(), semaphores=[])


def native_output(recurrent, gate, gamma, output):
    num_tokens, packed_rows = _token_layout(output.value)
    expected_recurrent = (num_tokens, 1, 48, 128)
    if tuple(int(v) for v in recurrent.shape) != expected_recurrent:
        raise RuntimeError(f"native GDN output requires recurrent {expected_recurrent}, got {recurrent.shape}")
    expected_gate = (1, num_tokens, 6144) if packed_rows else (num_tokens, 1, 6144)
    if tuple(int(v) for v in gate.shape) != expected_gate:
        raise RuntimeError(f"native GDN output requires gate {list(expected_gate)}, got {gate.shape}")
    input_addresses = tuple(tensor.buffer_address() for tensor in (recurrent, gate, gamma))
    if output.program is None or output.input_addresses != input_addresses:
        output.program = _make_program(recurrent, gate, gamma, output)
        output.input_addresses = input_addresses
    ttnn.generic_op([recurrent, gate, gamma, output.value], output.program)
    return output.value


def allocate_native_output_m2(device):
    """Packed-row M=2 output: [1,2,6144], directly consumable by o_proj."""
    value = ttnn.from_torch(
        torch.zeros((1, 2, 6144), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return NativeOutput(value=value)
