# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM2 MoE block on one P150: BF16 router (FP32 logits), sigmoid + expert-bias top-4 routing kernel, all 64 experts
resident in BFP4 (DRAM bank-shuffled layout), expert-grouped gate/up+SwiGLU and down projections, FP32 combine.

The same device-only path serves prefill (T rows) and decode (T = 1): no host reads, trace-compatible.  Reused from
models/common/modules/moe (Qwen36): the grouped prefix / pack programs (grouped_routing.py), GroupedExperts and
grouped_combine.  New: kernels/sigmoid_routing.cpp (LFM2's sigmoid + bias selection, norm_topk_prob weights; it
writes the route records the stock prefix consumes, called with top_k - 1 and an unused shared slot so all top_k
records are routed experts).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import ttnn

from models.common.modules.moe.grouped_experts import _SPLITS, GroupedExperts
from models.common.modules.moe.grouped_routing import _data_program, _empty, _pack_program, _workers, grouped_combine
from models.common.modules.moe.selected_experts import _CachedProgram
from models.common.modules.moe.streaming_experts import (
    _bank_cores,
    interleave_gate_up_for_banks,
    shuffle_for_banks,
    stream_weight_memory_config,
)

from .common import CK_HIFI4, free
from .moe_prefill import CombineRows, GroupedExpertsM, RoutePrefill

KERNELS = Path(__file__).resolve().parent / "kernels"
# Routing flags (sigmoid_routing.cpp): bit 0 rounds the FP32 router logits to BF16 (HF's BF16 Linear output), bit 1
# rounds the sigmoid scores to BF16 (HF computes sigmoid on BF16 and adds the FP32 bias to the BF16 score).
ROUTING_FLAGS = int(os.environ.get("LFM_ROUTING_FLAGS", "3"))
BIAS_SCALE = float(1 << 24)  # Q24
# Expert weight format: bfp4 (the precision plan).  LFM_EXPERT_DTYPE=bfp8 is a diagnostic variant (24.4 GB of
# experts, fits next to everything else; not cached on disk) that separates BFP4 quantization error from
# implementation error in full-model comparisons.
EXPERT_DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
EXPERT_DTYPE = os.environ.get("LFM_EXPERT_DTYPE", "bfp4")

# Rows from which MoE.__call__ takes the prefill programs (RoutePrefill + GroupedExpertsM); decode rows (stock
# B-row decode step) keep the grouped path.
PREFILL_MIN_ROWS = 64


def grouped_experts(device, gate_up, down, hidden: int, intermediate: int, stored: int) -> GroupedExperts:
    """GroupedExperts over the resident bank tensors.  Its constructor only accepts BFP4; the kernel itself takes
    the weight format from the tensor (CB format and tile size), so the BFP8 diagnostic variant skips that check."""
    if gate_up.dtype == ttnn.bfloat4_b:
        return GroupedExperts(device, gate_up, down, hidden=hidden, intermediate=intermediate, stored_experts=stored)
    ge = GroupedExperts.__new__(GroupedExperts)
    ge.device, ge.gate_up, ge.down, ge.hidden, ge.intermediate = device, gate_up, down, hidden, intermediate
    ge.cores, ge.banks, ge.splits = _bank_cores(device), device.dram_grid_size().x, _SPLITS
    ge._programs, ge._signatures = [None, None], [None, None]
    return ge


class SigmoidGroupedRouting:
    """x [1,1,T,H] BF16, FP32 logits [1,1,T,E], Q24 bias page -> grouped_routing's packed dict (x, weights, experts,
    inverse) for GroupedExperts + grouped_combine(top_k - 1)."""

    def __init__(self, device, num_experts: int, top_k: int, flags: int = ROUTING_FLAGS):
        if num_experts % 32 or not 2 <= top_k <= 31:
            raise ValueError("need tile-aligned experts and 2 <= top_k <= 31")
        self.device, self.num_experts, self.top_k, self.flags = device, num_experts, top_k, flags
        self._shape = None
        self.keep_routes = False  # tests: also return the route records as packed["routes"]

    def capacity(self, tokens: int) -> int:
        """Upper bound of 32-row expert groups: every expert with any rows adds at most one partial group."""
        return min(tokens * self.top_k, (tokens * self.top_k + 31) // 32 + self.num_experts)

    def __call__(self, x, logits, bias) -> dict:
        tokens, hidden = tuple(x.shape)[-2:]
        if tuple(logits.shape) != (1, 1, tokens, self.num_experts) or logits.dtype != ttnn.float32:
            raise ValueError(f"logits must be FP32 tiles [1,1,{tokens},{self.num_experts}]")
        if self._shape != (tokens, hidden):
            self._route, self._prefix, self._pack = _CachedProgram(), _CachedProgram(), _CachedProgram()
            self._shape = (tokens, hidden)
        groups = self.capacity(tokens)
        expert_width = ((groups + 32) // 32) * 32
        packed = _empty(self.device, [1, 1, groups * 32, hidden])
        weights = _empty(self.device, [1, 1, groups * 32, 32])
        experts = _empty(self.device, [1, 1, 1, expert_width], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        inverse = _empty(self.device, [1, 1, tokens, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        routes = _empty(self.device, [1, 1, tokens, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        group_map = _empty(self.device, [1, 1, groups, 64], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

        cores, cs = _workers(self.device, (tokens + 31) // 32)
        route_tensors = [logits, bias, routes]
        columns = self.num_experts // 32
        self._route.run(route_tensors, lambda: _data_program(
            str(KERNELS / "sigmoid_routing.cpp"), route_tensors,
            [self.num_experts, self.top_k, tokens, self.flags], cores, cs,
            columns * 4096 + self.num_experts * 4 + 128))
        # Stock prefix with top_k - 1: it reads top_k records per token; shared_slot = num_experts never occurs.
        cores, cs = _workers(self.device, 1)
        prefix_tensors = [routes, group_map, experts, inverse]
        self._prefix.run(prefix_tensors, lambda: _data_program(
            "grouped_routing_prefix.cpp", prefix_tensors,
            [self.num_experts, self.top_k - 1, self.num_experts, tokens, expert_width, 0], cores, cs,
            (self.num_experts + 1) * (256 + 3 * 4) + 2 * 4096 + 128 + 192))
        rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pack_tensors = [rows, group_map, experts, packed, weights]
        self._pack.run(pack_tensors, lambda: _pack_program(self.device, pack_tensors, groups, hidden, expert_width))
        free(rows, group_map)
        out = {"x": packed, "weights": weights, "experts": experts, "inverse": inverse}
        if self.keep_routes:
            out["routes"] = routes
        else:
            ttnn.deallocate(routes)
        return out


def bias_q24(expert_bias: torch.Tensor) -> torch.Tensor:
    q = torch.round(expert_bias.double() * BIAS_SCALE)
    if q.abs().max() >= 2**31 - 2**25:
        raise ValueError("expert_bias out of Q24 range")
    return q.to(torch.int32).reshape(1, 1, 1, -1)


def expert_bank_tensors(src, prefix: str, n_experts: int, banks: int):
    """Host BF16 [E*H, 2I] (gate|up, interleaved per bank) and [E*I, H] bank-shuffled layouts from the source's
    per-expert HF weights (w1 gate [I,H], w3 up [I,H], w2 down [H,I])."""
    gate_up = torch.stack([
        torch.cat([src.get(f"{prefix}.experts.{e}.w1.weight").T, src.get(f"{prefix}.experts.{e}.w3.weight").T], dim=1)
        for e in range(n_experts)
    ]).to(torch.bfloat16)
    gate_up = shuffle_for_banks(interleave_gate_up_for_banks(gate_up, banks), banks)
    down = torch.stack([src.get(f"{prefix}.experts.{e}.w2.weight").T for e in range(n_experts)]).to(torch.bfloat16)
    down = shuffle_for_banks(down, banks)
    return gate_up, down


class MoE:
    def __init__(self, device, cfg, src, layer: int, cache):
        self.device, self.cfg = device, cfg
        prefix = f"model.layers.{layer}.feed_forward"
        E, H, I = cfg.n_experts, cfg.hidden, cfg.moe_intermediate
        self.router = cache.get(f"L{layer}.router", lambda: src.get(f"{prefix}.gate.weight").T, ttnn.bfloat16)
        self.bias = ttnn.from_torch(bias_q24(src.get(f"{prefix}.expert_bias")), dtype=ttnn.int32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        banks = device.dram_grid_size().x
        built = {}

        def build(which):
            if not built:
                built["gate_up"], built["down"] = expert_bank_tensors(src, prefix, E, banks)
            return built.pop(which)

        dt = EXPERT_DTYPES[EXPERT_DTYPE]
        store = dt == ttnn.bfloat4_b  # the BFP8 diagnostic variant (24 GB) is never written to the tensor cache
        self.gate_up = cache.get(f"L{layer}.experts.gate_up_b{banks}", lambda: build("gate_up"), dt,
                                 memory_config=stream_weight_memory_config(device, E * H, 2 * I), store=store)
        self.down = cache.get(f"L{layer}.experts.down_b{banks}", lambda: build("down"), dt,
                              memory_config=stream_weight_memory_config(device, E * I, H), store=store)
        built.clear()
        self.routing = SigmoidGroupedRouting(device, E, cfg.top_k)
        self.experts = grouped_experts(device, self.gate_up, self.down, H, I, E)
        # Prefill (whole 32-row tiles, >= PREFILL_MIN_ROWS rows): many-core routing + prefix, the row-group reusing
        # experts program with row-major down rows, and the combine without to_layout; all bitwise the stock grouped
        # path above (tests/test_moe_prefill.py).
        self.prefill_routing = RoutePrefill(device, E, cfg.top_k) if ROUTING_FLAGS == 3 else None
        self.prefill_experts = GroupedExpertsM(device, self.gate_up, self.down, H, I)
        self.prefill_combine = CombineRows(device, cfg.top_k)

    def route(self, x):
        """Router logits (FP32) and the packed routing of x [1,1,T,H] (DRAM interleaved BF16)."""
        logits = ttnn.linear(x, self.router, dtype=ttnn.float32, compute_kernel_config=CK_HIFI4,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        packed = self.routing(x, logits, self.bias)
        return logits, packed

    def __call__(self, x, debug: dict | None = None):
        """debug (tests): receives host copies of the FP32 router logits and the route records (stock path)."""
        if x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        T = x.shape[-2]
        prefill = debug is None and T >= PREFILL_MIN_ROWS and T % 32 == 0
        if prefill and self.prefill_routing is not None:
            logits = ttnn.linear(x, self.router, dtype=ttnn.float32, compute_kernel_config=CK_HIFI4,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            packed = self.prefill_routing(x, logits, self.bias)
        else:
            self.routing.keep_routes = debug is not None
            logits, packed = self.route(x)
        if debug is not None:
            debug["logits"] = ttnn.to_torch(logits).float().reshape(-1, self.cfg.n_experts)
            routes = packed.pop("routes")
            debug["routes"] = ttnn.to_torch(routes).reshape(-1, 32)
            ttnn.deallocate(routes)
        ttnn.deallocate(logits)
        if prefill:
            projected = self.prefill_experts(packed["x"], packed["weights"], packed["experts"])
            out = self.prefill_combine(projected, packed["inverse"])
        else:
            projected = self.experts(packed["x"], packed["weights"], packed["experts"])
            out = grouped_combine(projected, packed["inverse"], top_k=self.cfg.top_k - 1)
        ttnn.deallocate(projected)
        for t in packed.values():
            ttnn.deallocate(t)
        return out
