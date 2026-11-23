# SPDX-License-Identifier: Apache-2.0
"""P150 27B GDN frontend: source-owned GQA3 FIR and T1..T32 state transitions."""
from dataclasses import dataclass

import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedKernelDescriptor,
)

_TILE_BYTES = 2048
_KERNEL = "models/demos/blackhole/qwen36/tt/gdn/native_frontend_kernel.cpp"


def _core(worker):
    return ttnn.CoreCoord(worker % 10, worker // 10)


def _core_set():
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))})


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
    conv_candidates: tuple = ()
    program: object = None
    program_signature: tuple[int, ...] | None = None
    fork_position: int = 0
    # Row t is user t; conv_candidates are then the users' own DRAM histories (updated in place).
    batched: bool = False
    # Shared verify rounds: slot count; conv_rows is the [1,3T,10240] prefix-history tensor (row t's
    # history after it at rows 3t..3t+2); the slots' histories and the row map are bound per call.
    groups: int = 0
    conv_rows: ttnn.Tensor | None = None


def _allocate(device, num_tokens, scratch=None, histories=True):
    def zeros(shape, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if not 1 <= num_tokens <= 32:
        raise ValueError("native frontend supports one tile row (1..32 tokens)")
    scalar_shape = (1, 1, 48) if num_tokens == 1 else (1, 1, num_tokens, 48)
    candidates = tuple(
        zeros((1, 3, 10240), ttnn.ROW_MAJOR_LAYOUT) for _ in range(num_tokens)
    ) if num_tokens > 1 and histories else ()
    return NativeFrontend(
        q=scratch.q if scratch is not None else zeros((num_tokens, 1, 48, 128)),
        k=scratch.k if scratch is not None else zeros((num_tokens, 1, 48, 128)),
        v=scratch.v if scratch is not None else zeros((num_tokens, 1, 48, 128)),
        a=scratch.a if scratch is not None else zeros(scalar_shape),
        b=scratch.b if scratch is not None else zeros(scalar_shape),
        zero_bias=scratch.zero_bias if scratch is not None else zeros((1, 1, 10240)),
        conv_s1=candidates[0] if candidates else None,
        conv_s2=candidates[1] if candidates else None,
        conv_candidates=candidates,
    )


def allocate_native_frontend(device):
    return _allocate(device, 1)


def allocate_native_frontend_m2(device, conv_state):
    return _allocate(device, 2)


def allocate_native_frontend_many(device, num_tokens, scratch=None):
    """Share sequential-layer operands, but retain each layer's prefix histories."""
    if not 2 <= num_tokens <= 32:
        raise ValueError("verification requires 2..32 rows")
    return _allocate(device, num_tokens, scratch)


def allocate_native_frontend_batched(device, num_tokens, scratch=None):
    """Batched-user operands; bind the users' histories per call via native_frontend(conv_state=...)."""
    if not 2 <= num_tokens <= 16:
        raise ValueError("batched users require 2..16 rows")
    output = _allocate(device, num_tokens, scratch)
    for candidate in output.conv_candidates:
        ttnn.deallocate(candidate)
    output.conv_s1 = output.conv_s2 = None
    output.conv_candidates = ()
    output.batched = True
    return output


def allocate_native_frontend_groups(device, num_tokens, slots, scratch=None, conv_rows=None):
    """Shared-round operands; conv_rows is the layer's [1,3C,10240] prefix-history tensor (C candidate rows: the
    rows of multi-row groups come first and must lie below C), shared by every round width of the layer."""
    if not 2 <= num_tokens <= 16:
        raise ValueError("shared rounds require 2..16 rows")
    if conv_rows is None or int(conv_rows.shape[-2]) % 3 or int(conv_rows.shape[-1]) != 10240:
        raise ValueError("shared rounds require a [1,3C,10240] prefix-history tensor")
    output = _allocate(device, num_tokens, scratch, histories=False)
    output.conv_s1 = output.conv_s2 = None
    output.conv_candidates = ()
    output.groups = int(slots)
    output.conv_rows = conv_rows
    return output


def _cb_descriptors():
    cores = _core_set()
    full = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    fmt = lambda cb: ttnn.CBFormatDescriptor(cb, ttnn.bfloat16, _TILE_BYTES, full)
    return [
        *[ttnn.CBDescriptor(total_size=count * _TILE_BYTES, core_ranges=cores,
                           format_descriptors=[fmt(cb)])
          for cb, count in ((0, 8), (1, 10), (2, 1), (16, 2), (17, 1))],
        # Two in-flight groups x (16 (tap, slot) history rows of 256 B + 64 row-pair
        # slots of 128 B), then one 64 B zero half-row source and a 128 B row-map copy.
        ttnn.CBDescriptor(total_size=2 * 12288 + 64 + 128, core_ranges=cores, format_descriptors=[
            ttnn.CBFormatDescriptor(3, ttnn.bfloat16, 64, ttnn.TileDescriptor(ttnn.Tile((1, 32))))
        ]),
    ]


def _make_program(device, mega, taps, bias, conv_state, output, group=None):
    if output.batched or output.groups:
        # Unused by the batched / grouped kernel (histories come from DRAM tensors).
        state_core = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    else:
        logical = ttnn.corerange_to_cores(conv_state.memory_config().shard_spec.grid, row_wise=True)
        if len(logical) != 1 or tuple(conv_state.memory_config().shard_spec.shape) != (3, 10240):
            raise RuntimeError("native GDN frontend requires one L1 convolution shard [3,10240]")
        state_core = device.worker_core_from_logical_core(logical[0])
    if output.groups:
        slots, row_map = group
        tail = [int(output.conv_rows.buffer_address()), *[int(t.buffer_address()) for t in slots],
                int(row_map.buffer_address())]
    else:
        tail = [int(t.buffer_address()) for t in output.conv_candidates] or [0, 0]
    token_args = [("num_tokens", int(output.q.shape[0])), ("fork_position", output.fork_position)]
    kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL, core_ranges=_core_set(),
        ncrisc_named_compile_time_args=[
            *token_args,
            ("tap_is_dram", int(taps[0].memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("bias_is_dram", int(bias.memory_config().buffer_type == ttnn.BufferType.DRAM)),
        ],
        brisc_named_compile_time_args=[
            *token_args,
            ("mega_is_dram", int(mega.memory_config().buffer_type == ttnn.BufferType.DRAM)),
            ("batched", int(output.batched)),
            ("groups", int(output.groups)),
        ],
        trisc_named_compile_time_args=token_args,
        brisc_common_runtime_args=[
            int(mega.buffer_address()), *[int(t.buffer_address()) for t in taps],
            int(bias.buffer_address()), int(conv_state.buffer_address()), state_core.x, state_core.y,
            int(output.a.buffer_address()), int(output.b.buffer_address()),
            int(output.q.buffer_address()), int(output.k.buffer_address()), int(output.v.buffer_address()),
            *tail,
        ],
        ncrisc_common_runtime_args=[*[int(t.buffer_address()) for t in taps], int(bias.buffer_address())],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=False, dst_full_sync_en=False,
        ),
        unified_compile_time_core_descriptors=[],
        per_core_compile_time_descriptors=[
            PerCoreCompileTimeDescriptor("worker_id", [(_core(i), i) for i in range(80)], 1)
        ],
    )
    descriptors = kernel.get_kernel_descriptors()
    return ttnn.ProgramDescriptor(kernels=descriptors.kernels, cbs=_cb_descriptors(), semaphores=[])


def native_frontend(mega, taps, bias, conv_state, output):
    """conv_state: the live L1 history, (batched) a tuple with one DRAM history per row, or (groups)
    (slot histories, row map [1,32] uint32)."""
    num_tokens = int(output.q.shape[0])
    if not 1 <= num_tokens <= 32 or tuple(mega.shape) not in ((1, num_tokens, 16512), (1, 1, num_tokens, 16512)):
        raise RuntimeError(f"native GDN frontend requires {num_tokens} packed 16512-wide rows, got {mega.shape}")
    if len(taps) != 4 or any(tuple(t.shape) != (1, 1, 10240) for t in taps):
        raise RuntimeError("native GDN frontend requires four flat BF16 tiled taps [1,1,10240]")
    if bias is None:
        bias = output.zero_bias
    if tuple(bias.shape) != (1, 1, 10240):
        raise RuntimeError("native GDN frontend requires flat bias [1,1,10240]")
    group = None
    if output.groups:
        slots, row_map = conv_state
        slots = tuple(slots)
        if len(slots) != output.groups or any(
                tuple(u.shape) != (1, 3, 10240) or u.layout != ttnn.ROW_MAJOR_LAYOUT or u.dtype != ttnn.bfloat16
                or u.memory_config() != ttnn.DRAM_MEMORY_CONFIG for u in slots):
            raise RuntimeError("grouped GDN frontend requires one BF16 row-major DRAM history [1,3,10240] per slot")
        group = (slots, row_map)
        conv_state = output.conv_rows
    elif output.batched:
        users = tuple(conv_state)
        if len(users) != num_tokens or num_tokens < 2:
            raise RuntimeError("batched GDN frontend requires one user history per row (>= 2 rows)")
        if any(tuple(u.shape) != (1, 3, 10240) or u.layout != ttnn.ROW_MAJOR_LAYOUT or u.dtype != ttnn.bfloat16
               or u.memory_config() != ttnn.DRAM_MEMORY_CONFIG for u in users):
            raise RuntimeError("batched GDN frontend requires BF16 row-major DRAM user histories [1,3,10240]")
        output.conv_candidates = users
        conv_state = users[0]
    elif (tuple(conv_state.shape) != (1, 3, 10240) or conv_state.layout != ttnn.ROW_MAJOR_LAYOUT
            or conv_state.dtype != ttnn.bfloat16 or not conv_state.is_sharded()
            or conv_state.memory_config().buffer_type != ttnn.BufferType.L1):
        raise RuntimeError("native GDN frontend requires BF16 row-major L1 history [1,3,10240]")
    tensors = [mega, *taps, bias, conv_state, output.q, output.k, output.v, output.a, output.b]
    if group is not None:
        tensors.extend((*group[0], group[1]))
    else:
        tensors.extend(output.conv_candidates[1:] if output.batched else output.conv_candidates)
    signature = tuple(int(t.buffer_address()) for t in tensors)
    # Each layer owns exactly one descriptor per token count, not an address-keyed cache.
    if output.program is None or output.program_signature != signature:
        output.program = _make_program(mega.device(), mega, taps, bias, conv_state, output, group)
        output.program_signature = signature
    ttnn.generic_op(tensors, output.program)
    return output.q, output.k, output.v, output.a, output.b
