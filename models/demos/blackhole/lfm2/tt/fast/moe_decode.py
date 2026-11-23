# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Single-row decode MoE for LFM2 (64 experts, top-4, I = 1536) on one P150, two launches:
router logits -> [router tail + streamed gate/up + SwiGLU + streamed down + expert sum].  No prefix / pack / combine.

Replaces, for decode rows (T = 1), `moe.MoE.__call__` (router linear -> SigmoidGroupedRouting (sigmoid_routing +
grouped prefix + to_layout + pack) -> GroupedExperts -> grouped_combine).  It reads the same resident expert tensors
(BFP4 gate_up [64*2048, 3072] / down [64*1536, 2048] in the streaming bank layout, router, Q24 bias page).

1. router logits, FP32, HiFi4 + FP32 acc, bitwise the default `ttnn.linear` of MoE.route (so the routing is bitwise
   the current path): `router_kind="stream"` (default) = decode_matmul.StreamDecodeMatmul on a stream-layout copy of
   the router (zero-padded to N=256 BF16: +1 MB DRAM per layer) in the stock in0_block_w=2 summation order;
   `router_kind="linear"` = ttnn.linear with the 1-core in0_block_w=2 program config (no extra memory, slower).
2. kernels/moe_decode_experts.cpp, one program, two roles:
   gate/up role (MOE_STAGE 0): per DRAM bank `gate_splits` workers.  NCRISC reads the logits row and runs the integer
   routing of kernels/sigmoid_routing.cpp (sigmoid_math.hpp: with the default flags 3 the scores come from the exact
   host-tabulated sigmoid_table, else an exact two-pass evaluation; bitwise the same ids and BF16 weights), then
   streams (expert, chunk of `pairs` gate/up column pairs) units: two contiguous column runs, gate/up accumulated in
   FP32 DEST (LoFi: exact for BFP4 x BF16), SiLU(g) * u * w on the pack thread, [1,32] activation tiles to an L1
   page per (expert, bank); after each unit every down worker's semaphore for that expert is incremented.
   down role (MOE_STAGE 1): per bank `down_splits` workers (row neighbours of the gate/up ones) route the token
   themselves, prefetch their column slice of the 4 selected experts' down weights while gate/up runs, gather each
   expert's activation row once its units are all written, accumulate the 4 experts in FP32 DEST and write the
   [1,1,1,2048] BF16 output tile (rows 1-31 zero).
"""
from __future__ import annotations

from pathlib import Path

import ttnn

from models.common.modules.moe.batched_experts import _bank_workers
from models.common.modules.moe.selected_experts import _CachedProgram, _accessor_args, _row_ranges
from models.common.modules.moe.streaming_experts import _bank_args, _bank_cores, _brisc_accessors, _cb, _read_page, _scratch_cb
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from ..common import CK_HIFI4
from ..moe import ROUTING_FLAGS
from .decode_matmul import PRESETS, StreamDecodeMatmul, cached_stream_weight, stream_weight
from . import sigmoid_table

_SCORE_TABLES: dict = {}


def score_table(device):
    """The flags-3 router score table (sigmoid_table.build()) as one row-major UINT16 DRAM page, one per device."""
    key = id(device)
    if key not in _SCORE_TABLES:
        t = sigmoid_table.build().reshape(1, 1, 1, -1)
        _SCORE_TABLES[key] = ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return _SCORE_TABLES[key]


KERNELS = Path(__file__).resolve().parent / "kernels"
L1 = ttnn.L1_MEMORY_CONFIG


def router_program_config():
    """One core, both 32-expert output tiles, in0_block_w=2: bitwise the default ttnn.linear of MoE.route."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(1, 1), in0_block_w=2, out_subblock_h=1, out_subblock_w=2,
        per_core_M=1, per_core_N=2, fuse_batch=True, fused_activation=None, mcast_in0=True)


class MoEDecode:
    """x [1,1,1,H] BF16 tile (DRAM or L1 interleaved) -> MoE output [1,1,1,H] BF16 tile (`memory_config`)."""

    def __init__(self, device, router, bias, gate_up, down, *, hidden: int, intermediate: int, num_experts: int,
                 top_k: int, flags: int = ROUTING_FLAGS, router_kind: str = "stream", router_stream=None,
                 gate_splits: int = 3, pairs: int = 2, down_splits: int = 2, gate_buffers: int = 3):
        banks = device.dram_grid_size().x
        self.device, self.router, self.bias, self.gate_up, self.down = device, router, bias, gate_up, down
        self.hidden, self.intermediate, self.E, self.top_k, self.flags = hidden, intermediate, num_experts, top_k, flags
        self.banks = banks
        self.half = intermediate // 32 // banks
        if (intermediate % (32 * banks) or hidden % (32 * banks) or self.half % pairs or 2 * pairs > 4
                or num_experts != 64 or tuple(router.shape)[-2:] != (hidden, num_experts)):
            raise ValueError("unsupported MoE geometry for the decode streams")
        down_cols = hidden // 32 // banks
        if down_cols % down_splits or down_cols // down_splits > 4:
            raise ValueError("down columns per core must fit four FP32 DEST slots")
        for t, k, n in ((gate_up, hidden, 2 * intermediate), (down, intermediate, hidden)):
            mc = t.memory_config()
            if (tuple(t.shape)[-2:] != (num_experts * k, n) or t.layout != ttnn.TILE_LAYOUT
                    or mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED or mc.buffer_type != ttnn.BufferType.DRAM):
                raise ValueError("expert weights must be the resident WIDTH_SHARDED DRAM bank layout")
        self.gate_splits, self.pairs, self.down_splits, self.gate_buffers = gate_splits, pairs, down_splits, gate_buffers
        if router_kind == "stream":
            # router_stream: the stream-layout router (decode_matmul.cached_stream_weight(cache, f"L{i}.router",
            # PRESETS["router"], dtype=ttnn.bfloat16)); built from `router` when not given.
            weight = router_stream if router_stream is not None else stream_weight(
                router, PRESETS["router"], device=device, dtype=ttnn.bfloat16)
            self._router = StreamDecodeMatmul(device, weight, PRESETS["router"], CK_HIFI4, out_dtype=ttnn.float32,
                                              output_memory_config=L1, n_out=num_experts)
        elif router_kind == "linear":
            self._router = None
            self.router_pc = router_program_config()
        else:
            raise ValueError(f"router_kind must be 'stream' or 'linear', got {router_kind}")
        self.bank_cores = _bank_cores(device)
        self.table = score_table(device)
        self._prog = _CachedProgram()
        self.keep_routing = False  # tests: __call__ also returns the (ids, weights) rows

    @classmethod
    def from_moe(cls, moe, cache=None, layer: int | None = None, **kw) -> "MoEDecode":
        """From a moe.MoE; with the runtime's WeightCache and the layer index the stream-layout router is cached
        on disk (key f"L{layer}.router.<preset tag>")."""
        cfg = moe.cfg
        if cache is not None and kw.get("router_kind", "stream") == "stream":
            kw["router_stream"] = cached_stream_weight(cache, f"L{layer}.router", PRESETS["router"],
                                                       dtype=ttnn.bfloat16)
        return cls(moe.device, moe.router, moe.bias, moe.gate_up, moe.down, hidden=cfg.hidden,
                   intermediate=cfg.moe_intermediate, num_experts=cfg.n_experts, top_k=cfg.top_k, **kw)

    # ------------------------------------------------------------------ programs
    def _placement(self):
        """(gate/up workers, down workers): per bank the bank core and its row neighbours, gate/up first."""
        _, vcs = _bank_args(self.bank_cores)
        rows = _bank_workers(self.device, self.bank_cores, self.gate_splits + self.down_splits)
        gate, down = [], []
        for bank, (row, (_, vc)) in enumerate(zip(rows, vcs)):
            for p in range(self.gate_splits):
                gate.append((row[p], bank, p, (vc + p) & 0x3))
            for p in range(self.down_splits):
                q = self.gate_splits + p
                down.append((row[q], bank, p, (vc + q) & 0x3))
        return gate, down

    def _role_kernel(self, fused, workers, routing_tensors, brisc_tensors, resident, extra_brisc):
        splits = self.gate_splits if fused else self.down_splits
        kt = (self.hidden if fused else self.intermediate) // 32
        bank_cols = (2 * self.intermediate if fused else self.hidden) // self.banks // 32
        cols = 2 * self.pairs if fused else bank_cols // splits
        wt = resident.get_tile()
        tb = wt.get_tile_size(resident.dtype)
        run_bytes = kt * (self.pairs if fused else cols) * tb
        page = _read_page(run_bytes, tb)
        buffers = self.gate_buffers if fused else self.top_k + 1
        row_tile = ttnn.TileDescriptor(ttnn.Tile([1, 32]))
        cs = _row_ranges([c for c, *_ in workers])
        cbs = [
            _cb(cs, 0, ttnn.bfloat16, 64, kt if fused else 2 * kt, row_tile),
            _cb(cs, 1, resident.dtype, tb, buffers * kt * cols, ttnn.TileDescriptor(wt)),
            _scratch_cb(cs, 2, (self.E // 32) * 128 + self.E * 4 + 64 + int(self.table.shape[-1]) * 2),
            _scratch_cb(cs, 3, kt * 128 if fused else cols * 2048),
            _cb(cs, 4, ttnn.bfloat16, 64, 2 * self.pairs if fused else cols, row_tile),
            _scratch_cb(cs, 5, 64),
        ]
        common = [("k_tiles", kt), ("bank_cols", bank_cols), ("core_cols", cols),
                  ("splits", splits), ("banks", self.banks), ("top_k", self.top_k),
                  ("inter_tiles", self.intermediate // 32),
                  ("units_per_expert", self.banks * self.half // self.pairs)]
        ncrisc_ct = []
        for t in routing_tensors:
            ncrisc_ct += _accessor_args(t)
        routing = [t.buffer_address() for t in routing_tensors]
        addresses = [t.buffer_address() for t in brisc_tensors]
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "moe_decode_experts.cpp"), core_ranges=cs,
            defines=[("MOE_STAGE", "0" if fused else "1")],
            ncrisc_compile_time_args=ncrisc_ct,
            ncrisc_named_compile_time_args=common + [
                ("weight_addr", resident.buffer_address()), ("weight_page_bytes", page),
                ("weight_pages", run_bytes // page), ("weight_run_bytes", run_bytes),
                ("weight_expert_bytes", kt * bank_cols * tb), ("weight_buffers", buffers),
                ("experts", self.E), ("route_flags", self.flags),
                ("score_e_lo", sigmoid_table.E_LO), ("score_e_hi", sigmoid_table.E_HI)],
            brisc_compile_time_args=_brisc_accessors(brisc_tensors),
            brisc_named_compile_time_args=common,
            trisc_named_compile_time_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=[(core, [bank, vc, part] + routing + [int(fused and bank == 0 and part == 0)])
                             for core, bank, part, vc in workers],
                brisc_args=[(core, addresses + [bank, part] + extra_brisc) for core, bank, part, _ in workers],
                trisc_args=[(core, [part]) for core, _, part, _ in workers]),
        )
        return kernel.get_kernel_descriptors().kernels, cbs

    def _program(self, tensors):
        x, logits, bias, ids, route, act, out = tensors[:7]
        gate, down = self._placement()
        noc = [self.device.worker_core_from_logical_core(c) for c, *_ in down]
        down_xy = [len(noc)] + [v for c in noc for v in (c.x, c.y)]
        routing = (logits, bias, ids, route, self.table)
        gk, gcbs = self._role_kernel(True, gate, routing, (x, act), self.gate_up, down_xy)
        dk, dcbs = self._role_kernel(False, down, routing, (act, out), self.down, [])
        every = _row_ranges([c for c, *_ in gate + down])
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=every, initial_value=0) for s in range(self.top_k)]
        return ttnn.ProgramDescriptor(kernels=gk + dk, cbs=gcbs + dcbs, semaphores=sems)

    # ------------------------------------------------------------------ forward
    def logits(self, x):
        if self._router is not None:
            return self._router(x)
        return ttnn.linear(x, self.router, dtype=ttnn.float32, compute_kernel_config=CK_HIFI4, memory_config=L1,
                           program_config=self.router_pc)

    def _row(self, dtype):
        return ttnn.empty([1, 1, 1, 32], dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
                          memory_config=L1)

    def experts(self, x, logits, memory_config=L1):
        """Gate/up + down on precomputed FP32 logits [1,1,1,E] (one launch) -> (out, ids, weights)."""
        ids, route = self._row(ttnn.uint16), self._row(ttnn.bfloat16)
        act = ttnn.empty([1, 1, self.top_k * self.banks, self.half * 32], dtype=ttnn.bfloat16,
                         layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, memory_config=L1)
        out = ttnn.empty([1, 1, 1, self.hidden], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                         memory_config=memory_config)
        tensors = [x, logits, self.bias, ids, route, act, out, self.gate_up, self.down]
        self._prog.run(tensors, lambda: self._program(tensors))
        ttnn.deallocate(act)
        return out, ids, route

    def __call__(self, x, memory_config=L1):
        if tuple(x.shape) != (1, 1, 1, self.hidden) or x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"MoEDecode takes one BF16 tile row [1,1,1,{self.hidden}]")
        logits = self.logits(x)
        out, ids, route = self.experts(x, logits, memory_config)
        ttnn.deallocate(logits)
        if self.keep_routing:
            return out, ids, route
        ttnn.deallocate(ids)
        ttnn.deallocate(route)
        return out
