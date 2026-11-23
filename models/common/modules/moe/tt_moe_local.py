# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device, batch-one MoE decode without fabric dispatch.

Routing (linear -> top-k -> softmax) and expert execution stay on device: the routed
experts and the shared expert are one stacked resident BFP4 tensor per projection,
consumed by ``ttnn.sparse_matmul`` with a row-major mask selecting the top-k routed
experts plus the always-on shared expert. No routing metadata crosses to the host, so
the decode step is trace-compatible.
"""

from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from models.common.modules.moe.selected_experts import TopKRouting
from models.common.modules.moe.streaming_experts import (
    DownStream,
    GateUpSwiGLUStream,
    interleave_gate_up_for_banks,
    shuffle_for_banks,
    stream_weight_memory_config,
)


def decode_program_config(n: int, k: int, cores: tuple[int, int], max_k_block: int = 64):
    """1D multicast config for an M=1 decode linear with N split across ``cores``."""
    num_cores = cores[0] * cores[1]
    n_tiles = (n + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    k_tiles = k // ttnn.TILE_SIZE
    per_core_n = (n_tiles + num_cores - 1) // num_cores
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*cores),
        in0_block_w=max(d for d in range(1, min(k_tiles, max_k_block) + 1) if k_tiles % d == 0),
        out_subblock_h=1,
        out_subblock_w=max(d for d in (4, 2, 1) if per_core_n % d == 0),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


_DECODE_LINEAR_COMPUTE = None


def decode_linear(x: ttnn.Tensor, weight: ttnn.Tensor, program_config) -> ttnn.Tensor:
    global _DECODE_LINEAR_COMPUTE
    if _DECODE_LINEAR_COMPUTE is None:
        _DECODE_LINEAR_COMPUTE = ttnn.init_device_compute_kernel_config(
            x.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    return ttnn.linear(
        x,
        weight,
        program_config=program_config,
        compute_kernel_config=_DECODE_LINEAR_COMPUTE,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

class TTLocalMoEGate:
    """Device top-k router with the shared expert as one more always-selected slot.

    The router weight is ``[hidden, experts]``; ``shared_gate`` (``[hidden, 1]``) becomes
    its logit column ``experts``. Returns ``(selection, weights)``: ROW_MAJOR ``[1, 1, 1,
    32]`` rows listing the ``top_k + 1`` selected slot ids (UINT16; routed ascending, the
    shared slot last) and their weights (BF16; the softmax over the top-k routed logits,
    Qwen3.5 ``norm_topk_prob``, and ``sigmoid(shared gate)``). The id count is exact even
    if a weight underflows; the expert programs rely on that.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        router: torch.Tensor,
        shared_gate: torch.Tensor,
        *,
        top_k: int = 8,
        shared_slot: int,
    ) -> None:
        if router.ndim != 2:
            raise ValueError(f"router must be rank 2 [hidden, experts], got {tuple(router.shape)}")
        self.mesh_device = mesh_device
        self.hidden_size, self.num_experts = router.shape
        if tuple(shared_gate.shape) != (self.hidden_size, 1):
            raise ValueError(f"shared_gate shape must be ({self.hidden_size}, 1), got {tuple(shared_gate.shape)}")
        self.top_k = top_k
        if not 1 <= top_k <= self.num_experts:
            raise ValueError(f"top_k must be in [1, {self.num_experts}], got {top_k}")
        # One tile of extra logit columns: column 0 is the shared-expert gate.
        extra = torch.zeros(self.hidden_size, ttnn.TILE_SIZE, dtype=shared_gate.dtype)
        extra[:, :1] = shared_gate
        self.router = ttnn.from_torch(
            torch.cat((router, extra.to(router.dtype)), dim=-1).to(torch.bfloat16).contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Traced sweep (P150): 8x2 cores with whole-K blocks, 11.4 us vs 12.2 us on 8x1.
        self.router_program = decode_program_config(self.num_experts + ttnn.TILE_SIZE, self.hidden_size, (8, 2))
        self._topk_routing = TopKRouting(mesh_device, self.num_experts, top_k, shared_slot)
        # Two-token verification: one union selection with per-token weights.
        self._topk_routing_2 = TopKRouting(mesh_device, self.num_experts, top_k, shared_slot, rows=2)

    def forward(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``x`` is ``[1, 1, rows, hidden]`` with one row, or two for speculative verification."""
        rows = int(x.shape[-2])
        if tuple(x.shape) != (1, 1, rows, self.hidden_size) or rows not in (1, 2):
            raise ValueError(f"x must have shape (1, 1, 1|2, {self.hidden_size}), got {tuple(x.shape)}")
        logits = decode_linear(x, self.router, self.router_program)
        routing = self._topk_routing if rows == 1 else self._topk_routing_2
        selection, weights = routing(logits)
        ttnn.deallocate(logits)
        return selection, weights


class TTLocalMoEDecode:
    """One P150 MoE decode for one token and one layer.

    Host weights use their final TT matmul orientation:
      * ``gate_up``: ``[experts, hidden, 2 * intermediate]``, gate then up
      * ``down``: ``[experts, intermediate, hidden]``
      * shared tensors have the same trailing dimensions without an expert axis (the
        shared-expert gate belongs to ``TTLocalMoEGate``)

    MLP weights are BFP4. The shared expert is stored as slot ``experts`` of the stacked
    routed tensors: it then runs inside the same streamed expert ops as the top-k routed
    experts, weighted by ``sigmoid(shared_gate)`` from the router (see ``TTLocalMoEGate``),
    instead of as five separate dense ops.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        shared_gate_up: torch.Tensor,
        shared_down: torch.Tensor,
        *,
        top_k: int = 8,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.top_k = top_k
        native_experts = hasattr(gate_up, "load_all") and hasattr(down, "load_all")
        gate_up_shape = tuple(gate_up.shape)
        down_shape = tuple(down.shape)
        if native_experts:
            if len(gate_up_shape) != 4 or gate_up_shape[0] != 1:
                raise ValueError(f"native gate_up shape must be [1, experts, hidden, packed], got {gate_up_shape}")
            _, self.num_experts, self.hidden_size, packed_intermediate = gate_up_shape
        else:
            self.num_experts, self.hidden_size, packed_intermediate = gate_up_shape
        if packed_intermediate % 2:
            raise ValueError("gate_up output dimension must be even")
        self.intermediate_size = packed_intermediate // 2
        expected_down = (
            (1, self.num_experts, self.intermediate_size, self.hidden_size)
            if native_experts
            else (self.num_experts, self.intermediate_size, self.hidden_size)
        )
        if down_shape != expected_down:
            raise ValueError(f"down shape {down_shape} must be {expected_down}")
        if tuple(shared_gate_up.shape) != (self.hidden_size, packed_intermediate):
            raise ValueError(
                "shared_gate_up shape must be "
                f"({self.hidden_size}, {packed_intermediate}), got {tuple(shared_gate_up.shape)}"
            )
        if tuple(shared_down.shape) != (self.intermediate_size, self.hidden_size):
            raise ValueError(
                "shared_down shape must be "
                f"({self.intermediate_size}, {self.hidden_size}), got {tuple(shared_down.shape)}"
            )
        if not 1 <= top_k <= self.num_experts:
            raise ValueError(f"top_k must be in [1, {self.num_experts}], got {top_k}")
        for name, tensor in {
            "gate_up": gate_up,
            "down": down,
            "shared_gate_up": shared_gate_up,
            "shared_down": shared_down,
        }.items():
            if tensor.shape[-1] % ttnn.TILE_SIZE or tensor.shape[-2] % ttnn.TILE_SIZE:
                raise ValueError(f"{name} trailing dimensions must be tile aligned, got {tuple(tensor.shape)}")

        cache_root = Path(cache_dir) if cache_dir is not None else None

        def upload(name: str, tensor: torch.Tensor, dtype) -> ttnn.Tensor:
            cache_file = str(cache_root / name) if cache_root is not None else None
            return ttnn.as_tensor(
                tensor.to(torch.bfloat16).contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_file,
            )

        # Native checkpoints already contain tiled BFP4 expert tensors in the
        # exact rank-4 device orientation; upload them without a torch round trip.
        # All routed experts stay resident (~18 GB BFP4) for dynamic routing.
        if native_experts:
            self.gate_up = gate_up.load_all(mesh_device)
            self.down = down.load_all(mesh_device)
        else:
            self.gate_up = upload("gate_up", gate_up.unsqueeze(0), ttnn.bfloat4_b)
            self.down = upload("down", down.unsqueeze(0), ttnn.bfloat4_b)
        # Fold the shared expert in as slot ``num_experts``; zero slots pad the stack to a
        # tile so the mask row stays tile aligned. The routed prefix is bit-identical.
        self.num_slots = self.num_experts + ttnn.TILE_SIZE
        self.shared_slot = self.num_experts
        self.gate_up = self._append_slot(
            self.gate_up, upload("shared_gate_up_slot", shared_gate_up[None, None], ttnn.bfloat4_b)
        )
        self.down = self._append_slot(self.down, upload("shared_down_slot", shared_down[None, None], ttnn.bfloat4_b))
        # Both expert projections stream from DRAM on the eight bank cores (device
        # profile, one layer: gate/up 36 us vs 52 us for sparse_matmul): the stacked slots
        # [0, experts] (the zero pad slots are never selected) are converted to the
        # per-bank column-major layout once and cached. Gate/up columns are interleaved
        # per bank so the SwiGLU fuses into the gate/up stream.
        stored = self.num_experts + 1
        self.gate_up = self._bank_layout(self.gate_up, "gate_up_swiglu", stored, cache_root, interleave=True)
        self.down = self._bank_layout(self.down, "down", stored, cache_root)
        self._gate_up_stream = GateUpSwiGLUStream(
            mesh_device,
            self.gate_up,
            stored_experts=stored,
            hidden=self.hidden_size,
            intermediate=self.intermediate_size,
            selected=top_k + 1,
        )
        self._down_stream = DownStream(
            mesh_device,
            self.down,
            stored_experts=stored,
            intermediate=self.intermediate_size,
            hidden=self.hidden_size,
            selected=top_k + 1,
        )
        # Two-token verification streams the union of both tokens' experts (at most
        # 2 * top_k routed plus the shared slot) once, multiplying both token rows.
        union = 2 * top_k + 1
        self._gate_up_stream_2 = GateUpSwiGLUStream(
            mesh_device,
            self.gate_up,
            stored_experts=stored,
            hidden=self.hidden_size,
            intermediate=self.intermediate_size,
            selected=union,
            rows=2,
        )
        self._down_stream_2 = DownStream(
            mesh_device,
            self.down,
            stored_experts=stored,
            intermediate=self.intermediate_size,
            hidden=self.hidden_size,
            selected=union,
            rows=2,
        )

    def _bank_layout(
        self, stack: ttnn.Tensor, name: str, stored: int, cache_root: Path | None, interleave: bool = False
    ) -> ttnn.Tensor:
        """``[1, slots, K, N]`` -> streamed ``[stored*K, N]`` bank layout, freeing ``stack``.

        The conversion is a host round trip of ~0.5-1 GB per tensor, so the result is
        cached; the cache key names the layout and the source shape."""
        _, _, k, n = stack.shape
        banks = self.mesh_device.dram_grid_size().x
        memory_config = stream_weight_memory_config(self.mesh_device, stored * k, n)
        cache = cache_root / f"{name}_stream_b{banks}_e{stored}_k{k}_n{n}.tensorbin" if cache_root else None
        if cache is not None and cache.exists():
            ttnn.deallocate(stack)
            streamed = ttnn.load_tensor(str(cache), device=self.mesh_device)
            if streamed.memory_config() != memory_config:
                streamed = ttnn.to_memory_config(streamed, memory_config)
            return streamed
        experts = ttnn.to_torch(stack).float()[0, :stored]
        ttnn.deallocate(stack)
        if interleave:
            experts = interleave_gate_up_for_banks(experts, banks)
        host = ttnn.from_torch(
            shuffle_for_banks(experts, banks).to(torch.bfloat16), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT
        )
        del experts
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            ttnn.dump_tensor(str(cache), host)
        return ttnn.to_device(host, self.mesh_device, memory_config=memory_config)

    @staticmethod
    def _append_slot(stack: ttnn.Tensor, slot: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, E, K, N]`` + ``[1, 1, K, N]`` -> ``[1, E + 32, K, N]`` (zero-padded), freeing inputs."""
        _, _, k, n = stack.shape
        pad = ttnn.zeros(
            [1, ttnn.TILE_SIZE - 1, k, n],
            dtype=stack.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=stack.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        stacked = ttnn.concat([stack, slot, pad], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for tensor in (stack, slot, pad):
            ttnn.deallocate(tensor)
        return stacked

    def forward(self, x: ttnn.Tensor, selection: ttnn.Tensor, weights: ttnn.Tensor) -> ttnn.Tensor:
        """Run the selected slots (top-k routed experts plus the shared expert).

        ``selection``/``weights`` come from ``TTLocalMoEGate``. The fused gate/up + SwiGLU
        stream reads the selected slots' weights from DRAM on the eight bank cores, applies
        ``SiLU(gate) * up * weight`` (the routing weight is linear through the down
        projection) in DEST, and writes the compact hidden rows in selection order. The
        streamed down projection reads one contiguous row per entry and sums the selected
        experts in FP32 in one op. With two token rows the slots are both tokens' union and
        each expert's weights stream once.
        """
        rows = int(x.shape[-2])
        if tuple(x.shape) != (1, 1, rows, self.hidden_size) or rows not in (1, 2):
            raise ValueError(f"x must have shape (1, 1, 1|2, {self.hidden_size}), got {tuple(x.shape)}")
        gate_up, down = (
            (self._gate_up_stream, self._down_stream) if rows == 1 else (self._gate_up_stream_2, self._down_stream_2)
        )
        hidden = gate_up(x, selection, weights)
        output = down(hidden, selection)
        ttnn.deallocate(hidden)
        return output
