# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Single-row decode MoE v2 for LFM2 (64 experts, top-4, I = 1536) on one P150: ONE launch per layer.

MoEDecode (moe_decode.py) runs two programs: router logits (StreamDecodeMatmul, 2 cores, 13 us) and the experts
program, in which every one of the 40 workers re-runs the whole routing (4.25 us) before streaming.  V2:

router role (kernels/moe_decode2_router.cpp), one core per 32-expert column (2 cores beside worker rows):
  logits bitwise the router preset of StreamDecodeMatmul (= MoE.route's ttnn.linear: in0_block_w = 2 chunks, HiFi4,
  FP32 L1 accumulation of the partials), but the partial stays in place (no per-chunk CB round trip through the
  unpacker), `router_group` chunks share a DEST acquire, in0 is a [router_rows, 32] tile filled by one 576 B read per
  x tile (split over BRISC and NCRISC, progressive pushes), and the whole weight column is fetched at once.  BRISC and
  NCRISC score 16 experts each (flags-3 table), merge, the two router cores swap their column lists, both form the
  final ids + BF16 weights (moe_decode2_route.hpp / moe_decode2_math.hpp: exact 32-bit-only division, bitwise
  lfm_route) and the 4 router RISCs write them to 10 workers each (payload, flush, route semaphore).
workers (kernels/moe_decode2_experts.cpp): moe_decode_experts.cpp's gate/up and down roles, with
  - the routing delivered (no per-worker routing),
  - one DRAM bank per worker row (`bank_rows="auto"`: two banks' streams in one row ran one bank at ~3/4 speed),
  - VC: the down workers of a bank share one VC (`vc_mode=1`), gate/up units of 1 pair (`pairs=1`),
  - gate/up activations written straight into every down worker's cb_act + a per-bank byte-lane semaphore
    (no intermediate tensor, no gather),
  - down workers accumulating each expert one bank piece (6 K tiles) at a time as pieces arrive, weights in blocks
    of `down_pieces` pieces (bitwise one custom_mm per expert), output zero-filled early, row 0 written at the end.

Same resident tensors as MoEDecode (expert banks, the stream-layout router copy `L{i}.router.stream-c1w1`, Q24 bias,
the score table); no new weight copies.  Only flags 3 (the default ROUTING_FLAGS) is supported.
Measured (tests/fast/test_moe_decode2.py, prof_moe2.py, layer 6, load ~1-2): rep_ms 72.8 (MoEDecode) -> ~54 us; output
bitwise MoEDecode, logits / ids / weights bitwise MoE.route.
"""
from __future__ import annotations

import ttnn

from models.common.modules.moe.batched_experts import _bank_workers
from models.common.modules.moe.selected_experts import _CachedProgram, _accessor_args, _row_ranges
from models.common.modules.moe.streaming_experts import _bank_args, _bank_cores, _brisc_accessors, _cb, _read_page, _scratch_cb
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from ..moe import ROUTING_FLAGS
from . import sigmoid_table
from .decode_matmul import PRESETS, cached_stream_weight, stream_weight
from .moe_decode import KERNELS, L1, score_table

LAND_CB, ACT_CB = 7, 6  # CBs written remotely (same index and address on every core of the program)


class MoEDecodeV2:
    """x [1,1,1,H] BF16 tile (DRAM or L1 interleaved) -> MoE output [1,1,1,H] BF16 tile (`memory_config`)."""

    def __init__(self, device, router, bias, gate_up, down, *, hidden: int, intermediate: int, num_experts: int,
                 top_k: int, flags: int = ROUTING_FLAGS, router_stream=None, gate_splits: int = 3, pairs: int = 1,
                 down_splits: int = 2, gate_buffers: int = 3, router_rows: int = 16, router_kb: int = 8,
                 router_group: int = 4, bank_rows: str = "auto", vc_mode: int = 1,
                 down_pieces: int = 4):
        banks = device.dram_grid_size().x
        self.device, self.bias, self.gate_up, self.down = device, bias, gate_up, down
        self.hidden, self.intermediate, self.E, self.top_k = hidden, intermediate, num_experts, top_k
        self.banks = banks
        self.half = intermediate // 32 // banks
        if flags != 3:
            raise ValueError("MoEDecodeV2 implements the flags-3 routing (BF16 logits and scores) only")
        if (intermediate % (32 * banks) or hidden % (32 * banks) or self.half % pairs or 2 * pairs > 4
                or num_experts != 64 or top_k > 4 or tuple(router.shape)[-2:] != (hidden, num_experts)):
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
        p = PRESETS["router"]
        if (p.cores_per_bank, p.w, p.k_split) != (1, 1, 2):
            raise ValueError("the router role reproduces the c1w1, k_split=2 router preset")
        # router_stream: decode_matmul.cached_stream_weight(cache, f"L{i}.router", PRESETS["router"], dtype=bf16)
        # (the same entry MoEDecode uses); column c of the router lives in DRAM bank c as one contiguous K run.
        self.router_w = router_stream if router_stream is not None else stream_weight(
            router, p, device=device, dtype=ttnn.bfloat16)
        kt = hidden // 32
        if router_rows not in (16, 32) or kt % router_kb or router_kb % 2 or not 1 <= router_group <= 4:
            raise ValueError("bad router geometry")
        self.router_rows, self.router_kb, self.router_group = router_rows, router_kb, router_group
        self.columns = num_experts // 32
        self.vc_mode = vc_mode
        if banks % down_pieces or not 2 <= top_k * banks // down_pieces + 1 <= 15 or self.half % 2:
            raise ValueError("down_pieces must split the banks into <= 14 blocks per worker (one transaction id each)")
        self.down_pieces = down_pieces
        self.bank_cores = list(_bank_cores(device))
        if bank_rows == "auto":
            # One DRAM bank per worker row: with two banks' streams in one row (the P150's NoC-0 anchors put banks 0
            # and 4 in row 9), one of them slowed to ~3/4 of the others' rate once all banks start together
            # (prof_moe2.py: 73.6 vs 60.5 us).  The lower-index bank of a shared row moves to the nearest free row.
            grid_y = device.compute_with_storage_grid_size().y
            for b, a in enumerate(self.bank_cores):
                used = {c.y for i, c in enumerate(self.bank_cores) if i != b}
                if a.y in {c.y for c in self.bank_cores[b + 1:]}:
                    y = min((y for y in range(grid_y) if y not in used), key=lambda y: (abs(y - a.y), y))
                    self.bank_cores[b] = ttnn.CoreCoord(a.x, y)
        else:
            for item in filter(None, bank_rows.split(";")):  # "bank:row;...": a bank's workers in another row
                b, y = (int(v) for v in item.split(":"))
                self.bank_cores[b] = ttnn.CoreCoord(self.bank_cores[b].x, y)
        self.table = score_table(device)
        self._prog = _CachedProgram()
        self.keep_routing = False  # tests: __call__ also returns (logits, ids, weights)

    @classmethod
    def from_moe(cls, moe, cache=None, layer: int | None = None, **kw) -> "MoEDecodeV2":
        """From a moe.MoE; with the runtime's WeightCache and the layer index the stream-layout router comes from
        the cache entry f"L{layer}.router.<preset tag>" (shared with MoEDecode)."""
        cfg = moe.cfg
        if cache is not None:
            kw["router_stream"] = cached_stream_weight(cache, f"L{layer}.router", PRESETS["router"],
                                                       dtype=ttnn.bfloat16)
        return cls(moe.device, moe.router, moe.bias, moe.gate_up, moe.down, hidden=cfg.hidden,
                   intermediate=cfg.moe_intermediate, num_experts=cfg.n_experts, top_k=cfg.top_k, **kw)

    # ------------------------------------------------------------------ placement
    def _placement(self):
        """(gate/up workers, down workers, routers): per bank the bank core and its row neighbours, gate/up first;
        router c beside a worker row (no extra core range), nearest to DRAM bank c's core."""
        _, vcs = _bank_args(self.bank_cores)
        rows = _bank_workers(self.device, self.bank_cores, self.gate_splits + self.down_splits)
        gate, down = [], []
        for bank, (row, (_, vc)) in enumerate(zip(rows, vcs)):
            for p in range(self.gate_splits):
                gate.append((row[p], bank, p, (vc + p) & 0x3))
            for p in range(self.down_splits):
                q = self.gate_splits + p
                # vc_mode 1: the down workers share one VC, the gate/up streams keep their own
                down.append((row[q], bank, p, (vc + (self.gate_splits if self.vc_mode == 1 else q)) & 0x3))
        grid = self.device.compute_with_storage_grid_size()
        taken = {(c.x, c.y) for r in rows for c in r}
        workers = set(taken)
        routers = []
        for col in range(self.columns):
            a = self.bank_cores[col]
            free = [(x, y) for x in range(grid.x) for y in range(grid.y) if (x, y) not in taken]
            beside = [c for c in free if (c[0] - 1, c[1]) in workers or (c[0] + 1, c[1]) in workers]
            best = min(beside or free, key=lambda c: (abs(c[0] - a.x) + abs(c[1] - a.y), c[1], c[0]))
            taken.add(best)
            workers.add(best)
            routers.append((ttnn.CoreCoord(*best), col))
        return gate, down, routers

    # ------------------------------------------------------------------ programs
    def _role_kernel(self, fused, workers, ncrisc_tensors, brisc_tensors, resident, extra_brisc):
        splits = self.gate_splits if fused else self.down_splits
        kt = (self.hidden if fused else self.intermediate) // 32
        bank_cols = (2 * self.intermediate if fused else self.hidden) // self.banks // 32
        cols = 2 * self.pairs if fused else bank_cols // splits
        wt = resident.get_tile()
        tb = wt.get_tile_size(resident.dtype)
        run_bytes = kt * (self.pairs if fused else cols) * tb
        # down workers read each column's K slice per bank piece (half tiles) into [piece][column][K] blocks of
        # `down_pieces` pieces (one transaction id each: all of a worker's blocks in flight)
        page = _read_page(run_bytes, tb) if fused else self.half * tb
        block = kt * cols if fused else self.down_pieces * self.half * cols
        buffers = self.gate_buffers if fused else self.top_k * self.banks // self.down_pieces + 1
        row_tile = ttnn.TileDescriptor(ttnn.Tile([1, 32]))
        cs = _row_ranges([c for c, *_ in workers])
        cbs = ([_cb(cs, 0, ttnn.bfloat16, 64, kt, row_tile)] if fused else []) + [
            _cb(cs, 1, resident.dtype, tb, buffers * block, ttnn.TileDescriptor(wt)),
            _scratch_cb(cs, 2, 64),
            _scratch_cb(cs, 3, kt * 128 if fused else cols * 2048),
            _cb(cs, 4, ttnn.bfloat16, 64, 2 * self.pairs if fused else cols, row_tile),
            _scratch_cb(cs, 5, 64),
        ]
        common = [("k_tiles", kt), ("bank_cols", bank_cols), ("core_cols", cols),
                  ("splits", splits), ("banks", self.banks), ("top_k", self.top_k),
                  ("inter_tiles", self.intermediate // 32), ("down_pieces", self.down_pieces)]
        ncrisc_ct = []
        for t in ncrisc_tensors:
            ncrisc_ct += _accessor_args(t)
        nc_addrs = [t.buffer_address() for t in ncrisc_tensors]
        addresses = [t.buffer_address() for t in brisc_tensors]
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "moe_decode2_experts.cpp"), core_ranges=cs,
            defines=[("MOE_STAGE", "0" if fused else "1")],
            ncrisc_compile_time_args=ncrisc_ct,
            ncrisc_named_compile_time_args=common + [
                ("weight_addr", resident.buffer_address()), ("weight_page_bytes", page),
                ("weight_pages", run_bytes // page), ("weight_run_bytes", run_bytes),
                ("weight_expert_bytes", kt * bank_cols * tb), ("weight_buffers", buffers),
                ("route_sem", self.banks)],
            brisc_compile_time_args=_brisc_accessors(brisc_tensors),
            brisc_named_compile_time_args=common + [
                ("out_l1", int(brisc_tensors[-1].memory_config().buffer_type == ttnn.BufferType.L1)),
                ("bank_sem", 0), ("gate_chunks", self.half // self.pairs)],
            trisc_named_compile_time_args=common,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=[(core, [bank, vc, part] + nc_addrs + [int(fused and bank == 0 and part == 0)])
                             for core, bank, part, vc in workers],
                brisc_args=[(core, addresses + [bank, part] + extra_brisc) for core, bank, part, _ in workers],
                trisc_args=[(core, [part]) for core, _, part, _ in workers]),
        )
        return kernel.get_kernel_descriptors().kernels, cbs

    def _senders(self, routers, gate, down):
        """Per router core: [(other router NoC x, y, its workers' NoC (x, y)) for BRISC, then for NCRISC].  The 4
        router RISCs take the workers round-robin, gate/up workers first (their streams start the earliest)."""
        noc = [self.device.worker_core_from_logical_core(c) for c, *_ in gate + down]
        xy = [(n.x, n.y) for n in noc]
        senders = len(routers) * 2
        rnoc = [self.device.worker_core_from_logical_core(c) for c, _ in routers]
        out = []
        for r in range(len(routers)):
            other = rnoc[1 - r]
            args = []
            for risc in range(2):
                mine = xy[2 * r + risc::senders]
                args += [other.x, other.y, len(mine)] + [v for p in mine for v in p]
            out.append(args)
        return out

    def _router_kernel(self, routers, senders, x, logits):
        kt = self.hidden // 32
        rows = self.router_rows
        in0_tile = ttnn.Tile([rows, 32])
        out_tile_bytes = rows * 32 * 4
        out_face_bytes = out_tile_bytes // 2 if rows <= 16 else 1024
        table_bytes = int(self.table.shape[-1]) * 2
        if table_bytes != 2 * (sigmoid_table.E_HI - sigmoid_table.E_LO + 1) * 128 * 2:
            raise ValueError("score table size does not match sigmoid_table.E_LO/E_HI")
        area = (table_bytes + self.E * 4 + 63) // 64 * 64
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _ in routers])

        def cb(index, dtype, page, pages, tile):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cs,
                                     format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page,
                                                                                 ttnn.TileDescriptor(tile))])

        cbs = [cb(0, ttnn.bfloat16, rows * 32 * 2, kt // 2, in0_tile),
               cb(2, ttnn.bfloat16, rows * 32 * 2, kt // 2, in0_tile),
               cb(1, ttnn.bfloat16, 2048, kt, ttnn.Tile([32, 32])),
               _scratch_cb(cs, 3, 2 * area + 8 * 64),
               cb(16, ttnn.float32, out_tile_bytes, 1, in0_tile),
               cb(24, ttnn.float32, out_tile_bytes, 1, in0_tile)]
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        mode[24] = ttnn.UnpackToDestMode.UnpackToDestFp32  # exact reload of the FP32 partial
        compute.unpack_to_dest_mode.extend(mode)
        common = [("kt", kt), ("kb", self.router_kb), ("ks", 2), ("group", self.router_group)]
        dm = common + [("top_k", self.top_k), ("experts", self.E), ("score_e_lo", sigmoid_table.E_LO),
                       ("score_e_hi", sigmoid_table.E_HI), ("out_face_bytes", out_face_bytes),
                       ("in0_tile_bytes", rows * 32 * 2), ("route_sem", self.banks), ("xchg_sem", self.banks + 1),
                       ("cross_sem", self.banks + 3)]
        acc = {name: _accessor_args(t) for name, t in
               (("x", x), ("logits", logits), ("table", self.table), ("bias", self.bias))}

        def accessors(names):
            ct, named = [], []
            for name in names:
                named.append((f"{name}_cta", len(ct)))
                ct += acc[name]
            return ct, named

        nc_ct, nc_named = accessors(("x", "table", "bias"))
        br_ct, br_named = accessors(("x", "logits", "table", "bias"))
        addrs = [int(t.buffer_address()) for t in (self.router_w, x, logits, self.table, self.bias)]
        per_core = [(core, [col] + args) for (core, col), args in zip(routers, senders)]
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "moe_decode2_router.cpp"), core_ranges=cs,
            ncrisc_compile_time_args=nc_ct, ncrisc_named_compile_time_args=dm + nc_named,
            brisc_compile_time_args=br_ct, brisc_named_compile_time_args=dm + br_named,
            trisc_named_compile_time_args=common,
            ncrisc_common_runtime_args=addrs, brisc_common_runtime_args=addrs,
            trisc_compute_config=compute,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                ncrisc_args=per_core, brisc_args=per_core, trisc_args=[(c, [col]) for c, col in routers]),
        )
        return kernel.get_kernel_descriptors().kernels, cbs

    def _program(self, tensors):
        x, logits, ids, route, out = tensors[:5]
        gate, down, routers = self._placement()
        noc = [self.device.worker_core_from_logical_core(c) for c, *_ in down]
        down_xy = [len(noc)] + [v for c in noc for v in (c.x, c.y)]
        gk, gcbs = self._role_kernel(True, gate, (ids, route), (x,), self.gate_up, down_xy)
        dk, dcbs = self._role_kernel(False, down, (ids, route), (out,), self.down, [])
        rk, rcbs = self._router_kernel(routers, self._senders(routers, gate, down), x, logits)
        workers = _row_ranges([c for c, *_ in gate + down])
        every = _row_ranges([c for c, *_ in gate + down] + [c for c, _ in routers])
        # One CB over all the cores that are written remotely, before any per-role CB: the same L1 address on all.
        land = _scratch_cb(every, LAND_CB, 64)
        act = _cb(workers, ACT_CB, ttnn.bfloat16, 64, self.top_k * self.intermediate // 32,
                  ttnn.TileDescriptor(ttnn.Tile([1, 32])))
        # semaphores: 0..banks-1 activation pieces per bank (down workers), then the routing ones (router kernel)
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=every, initial_value=0) for s in range(self.banks + 4)]
        return ttnn.ProgramDescriptor(kernels=gk + dk + rk, cbs=[land, act] + gcbs + dcbs + rcbs, semaphores=sems)

    # ------------------------------------------------------------------ forward
    def _row(self, dtype):
        return ttnn.empty([1, 1, 1, 32], dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
                          memory_config=L1)

    def forward(self, x, memory_config=L1):
        """-> (out, logits FP32 [1,1,1,E] (row 0 only), ids UINT16 row, weights BF16 row)."""
        if tuple(x.shape) != (1, 1, 1, self.hidden) or x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"MoEDecodeV2 takes one BF16 tile row [1,1,1,{self.hidden}]")
        ids, route = self._row(ttnn.uint16), self._row(ttnn.bfloat16)
        logits = ttnn.empty([1, 1, 1, self.E], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device,
                            memory_config=L1)
        out = ttnn.empty([1, 1, 1, self.hidden], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                         memory_config=memory_config)
        tensors = [x, logits, ids, route, out, self.gate_up, self.down, self.router_w, self.bias, self.table]
        self._prog.run(tensors, lambda: self._program(tensors))
        return out, logits, ids, route

    def __call__(self, x, memory_config=L1):
        out, logits, ids, route = self.forward(x, memory_config)
        if self.keep_routing:
            return out, logits, ids, route
        for t in (logits, ids, route):
            ttnn.deallocate(t)
        return out
