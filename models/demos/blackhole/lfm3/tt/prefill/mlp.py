# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Prefill dense SwiGLU MLP (Lfm3MLP: w2(silu(w1 x) * w3 x)) for chunk rows M >= 32 (whole tiles).

`PrefillMLP(mlp)` wraps a layers.DenseMLP: h = SiLU(x W1) * (x W3) comes from ONE launch of kernels/mlp_prefill_w13.cpp
(both projections accumulated over the whole K in FP32 DEST, SiLU(gate) * up on the SFPU of the pack thread, h packed
once to BF16; no g / u DRAM round trip, no separate SiLU*mul program), then y = h @ w2 (stock 2D multicast
matmul, program config with a larger in0_block_w than layers.linear's).

The kernel reads W1 | W3 in a panel layout (one extra device copy of w1 + w3 per layer, built from the DenseMLP's
device tensors, block-float values unchanged): panel g = h column tile g = for every K tile its W1 tile and its W3
tile, stored so that one panel is 8 contiguous DRAM bank runs (see the kernel header).  DenseMLP's w1 / w3 stay for
the decode paths.

Work split (M-dependent, cached per M): core (x, y) of a gx x gy rectangle of the compute grid owns the row tiles of
M group x (gx = min(grid x, M / 32), <= 6 rows resident in L1) and the panels of N group y (gy = grid y).  More than
gx * 6 row tiles (M > 2112): one launch per block of <= gx * 6 row tiles (M = 4096: two launches of 2048 rows).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import ttnn

from ..fast.conv_decode import CachedProgram, cb_descriptors
from ..common import CK_BFP8
from ..layers import CK_BFP4_PREFILL, DRAM

KERNEL = str(Path(__file__).resolve().parent.parent / "kernels" / "mlp_prefill_w13.cpp")
BANKS = 8
X_TILE = 2048
W_TILE = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088}
MAX_ROWS = 6               # resident in0 rows (tiles) per core: 6 x K/32 BF16 tiles (786 KB at K = 2048)
L1_BUDGET = 1250 * 1024    # CB bytes per core
OUT_TILES = 16
PREFETCH = 3              # weight panels read ahead of the multicast (in1 sender)
W2_L1_BUDGET = 1350 * 1024  # w2 matmul CBs per core (as layers.prefill_matmul_config)


def bank_run_order(n: int) -> torch.Tensor:
    """Page offset (within an 8-aligned run of n pages of a DRAM-interleaved tensor) of L1 slot s when the run is
    read as 8 contiguous bank runs (kernel read_bank_runs): slot s = b * n/8 + i holds page 8 i + b."""
    s = torch.arange(n)
    per = n // BANKS
    return BANKS * (s % per) + s // per


def panel_layout(w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """W1, W3 (W^T [K, N]) -> the kernel's panel tensor [N/32 * 2K/8, 256] (tile grid 8 wide: page p at tile row
    p // 8, tile column p % 8).  Panel g = pages [g * 2Kt, (g + 1) * 2Kt); its L1 slot 2 j + c (read as bank runs)
    holds K tile k(j) (the in0 row's bank-run order) of W1 (c = 0) / W3 (c = 1) at h column g."""
    K, N = w1.shape
    kt, nt = K // 32, N // 32
    kj = bank_run_order(kt)  # in0 L1 slot j -> K tile
    t1 = w1.reshape(kt, 32, nt, 32).permute(2, 0, 1, 3)[:, kj]  # [nt, kt (slot order), 32, 32]
    t3 = w3.reshape(kt, 32, nt, 32).permute(2, 0, 1, 3)[:, kj]
    slots = torch.stack([t1, t3], dim=2).reshape(nt, 2 * kt, 32, 32)  # slot 2 j + c
    pages = torch.empty_like(slots)
    pages[:, bank_run_order(2 * kt)] = slots
    return pages.reshape(nt * 2 * kt // BANKS, BANKS, 32, 32).permute(0, 2, 1, 3).reshape(-1, BANKS * 32)


def _split(n: int, parts: int) -> list[tuple[int, int]]:
    """n items into `parts` contiguous balanced ranges (start, count)."""
    base, extra = divmod(n, parts)
    out, s = [], 0
    for i in range(parts):
        c = base + (i < extra)
        out.append((s, c))
        s += c
    return out


@dataclass
class _Block:
    """One launch: row tiles [m0, m0 + mt) of x / h."""
    gx: int
    gy: int
    p: int        # max rows per core
    nb: int       # in1 ring panels
    args: list    # (CoreCoord, per-core runtime args)


class PrefillMLP:
    """y = w2(SiLU(w1 x) * w3 x) for prefill chunks: x [1, 1, M, H] BF16 TILE DRAM interleaved (M % 32 == 0) ->
    y [1, 1, M, H] BF16 DRAM.  cache / key: WeightCache entry for the panel copy of w1 | w3 (e.g. f"L{i}.mlp.w13p")."""

    extra_defines: tuple = ()  # kernel ablations (tests/prefill/mlp.py --ablate): MLP_SKIP_MATH, MLP_SKIP_SFPU, ...

    def __init__(self, mlp, cache=None, key: str | None = None):
        if mlp.w1.dtype != mlp.w3.dtype or mlp.w1.dtype not in W_TILE:
            raise ValueError(f"PrefillMLP: w1 / w3 must share a BFP4 / BFP8 format ({mlp.w1.dtype}, {mlp.w3.dtype})")
        self.mlp, self.w2 = mlp, mlp.w2
        self.device = mlp.w1.device()
        self.K, self.N = mlp.w1.shape[-2], mlp.w1.shape[-1]
        if self.K % (32 * BANKS) or self.N % 32:
            raise ValueError("PrefillMLP: hidden must be a multiple of 256, intermediate of 32")
        self.dtype = mlp.w1.dtype

        def build():
            return panel_layout(ttnn.to_torch(mlp.w1).reshape(self.K, self.N),
                                ttnn.to_torch(mlp.w3).reshape(self.K, self.N))

        if cache is not None and key is not None:
            self.w13 = cache.get(key, build, self.dtype)
        else:
            self.w13 = ttnn.from_torch(build().to(torch.bfloat16), dtype=self.dtype, layout=ttnn.TILE_LAYOUT,
                                       device=self.device, memory_config=DRAM)
        self.fidelity = ttnn.MathFidelity.LoFi if self.dtype == ttnn.bfloat4_b else ttnn.MathFidelity.HiFi2
        self._plans: dict = {}
        self._progs: dict = {}
        self._w2_pc: dict = {}

    def extra_dram_bytes(self) -> int:
        return 2 * self.K * self.N // 1024 * W_TILE[self.dtype]

    def plan(self, M: int) -> list[_Block]:
        """The launches of an M-row call."""
        pl = self._plans.get(M)
        if pl is not None:
            return pl
        grid = self.device.compute_with_storage_grid_size()
        mt = M // 32
        per = grid.x * MAX_ROWS
        pl = [self._block(m0, n) for m0, n in _split(mt, -(-mt // per))]
        self._plans[M] = pl
        return pl

    def _block(self, m0: int, mt: int) -> _Block:
        dev = self.device
        grid = dev.compute_with_storage_grid_size()
        kt, nt = self.K // 32, self.N // 32
        gx, gy = min(grid.x, mt), min(grid.y, nt)
        rows = _split(mt, gx)
        p = max(c for _, c in rows)
        panel_bytes = 2 * kt * W_TILE[self.dtype]
        nb = min(8, (L1_BUDGET - p * kt * X_TILE - OUT_TILES * X_TILE) // panel_bytes)
        if nb < 2:
            raise ValueError(f"PrefillMLP: no L1 room for the weight ring ({mt} row tiles)")
        cols = _split(nt, gy)

        def phys(x, y):
            return dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y))

        g0, g1 = phys(0, 0), phys(gx - 1, gy - 1)
        master = phys(0, gy - 1)
        args = []
        for y in range(gy):
            for x in range(gx):
                r0, nr = rows[x]
                c0, nc = cols[y]
                # in0: the column's bottom core reads x and multicasts it to the rows above (NOC 1); in1: the row's
                # x = 0 core multicasts east (NOC 0).  Master (the bottom-left core): the h-write "go" barrier.
                s0, s1 = phys(x, gy - 1), phys(0, y)
                a0, z0 = phys(x, 0), phys(x, max(gy - 2, 0))
                a1, z1 = phys(min(1, gx - 1), y), phys(gx - 1, y)
                args.append((ttnn.CoreCoord(x, y),
                             [m0 + r0, nr, c0, nc,
                              int(y == gy - 1), s0.x, s0.y, a0.x, a0.y, z0.x, z0.y, gy - 1,
                              int(x == 0), s1.x, s1.y, a1.x, a1.y, z1.x, z1.y, gx - 1,
                              int(x == 0 and y == gy - 1), master.x, master.y, g0.x, g0.y, g1.x, g1.y,
                              gx * gy - 1, gx]))
        return _Block(gx, gy, p, nb, args)

    def _build(self, x, h, b: _Block):
        kt, nt = self.K // 32, self.N // 32
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(b.gx - 1, b.gy - 1))])
        defines = dict([("MLP_KT", str(kt)), ("MLP_NT", str(nt)), ("MLP_P", str(b.p)), ("MLP_NB", str(b.nb)),
                        ("MLP_W_TILE", str(W_TILE[self.dtype])), ("MLP_SIG_FP32", "true"),
                        ("MLP_PREFETCH", str(PREFETCH)), ("MLP_OUT_TILES", str(OUT_TILES))])
        defines.update(dict(self.extra_defines))
        defines = list(defines.items())

        def rt(per):
            r = ttnn.RuntimeArgs()
            for core, a in per:
                r[core.x][core.y] = list(a)
            return r

        dm = dict(kernel_source=KERNEL, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=cores,
                  defines=defines)
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, math_approx_mode=False,
                                               fp32_dest_acc_en=True, dst_full_sync_en=False)
        kernels = [
            ttnn.KernelDescriptor(**dm, common_runtime_args=[int(self.w13.buffer_address())], runtime_args=rt(b.args),
                                  config=ttnn.DataMovementConfigDescriptor(
                                      processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default)),
            ttnn.KernelDescriptor(**dm, common_runtime_args=[int(x.buffer_address()), int(h.buffer_address())],
                                  runtime_args=rt(b.args),
                                  config=ttnn.DataMovementConfigDescriptor(
                                      processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default)),
            ttnn.KernelDescriptor(**dm, runtime_args=rt(b.args), config=compute),
        ]
        cbs = (cb_descriptors((0,), b.p * kt, cores)
               + cb_descriptors((1,), b.nb * 2 * kt, cores, dtype=self.dtype, page=W_TILE[self.dtype])
               + cb_descriptors((16,), OUT_TILES, cores))
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in range(6)]
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)

    def swiglu(self, x):
        """h = SiLU(x W1) * (x W3): [1, 1, M, N] BF16 DRAM."""
        M = x.shape[-2]
        if (len(x.shape) != 4 or x.shape[-1] != self.K or M % 32 or x.dtype != ttnn.bfloat16
                or x.layout != ttnn.TILE_LAYOUT or x.memory_config() != DRAM):
            raise ValueError(f"PrefillMLP: x must be BF16 TILE DRAM interleaved [1, 1, M, {self.K}], got {x.shape}")
        h = ttnn.empty([1, 1, M, self.N], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                       memory_config=DRAM)
        for i, b in enumerate(self.plan(M)):
            prog = self._progs.setdefault((M, i), CachedProgram())
            prog.run([x, self.w13, h], lambda b=b: self._build(x, h, b))
        return h

    def w2_config(self, M: int):
        """2D-multicast config of y = h @ w2: layers.prefill_matmul_config's grid and per-core block, with the
        largest in0_block_w that fits L1 (fewer FP32 partial round trips) and subblock widths up to 4 (incl. 3)."""
        pc = self._w2_pc.get(M)
        if pc is None:
            grid = self.device.compute_with_storage_grid_size()
            mt, kt, nt = M // 32, self.N // 32, self.w2.shape[-1] // 32
            pm = -(-mt // min(grid.y, mt))
            pn = -(-nt // min(grid.x, nt))
            gy, gx = -(-mt // pm), -(-nt // pn)
            sw = max(d for d in (4, 3, 2, 1) if pn % d == 0)
            sh = max(d for d in (4, 3, 2, 1) if pm % d == 0 and d * sw <= 4)
            wt = W_TILE.get(self.w2.dtype, 2048)
            ibw = next((d for d in (24, 20, 16, 12, 10, 8, 6, 4, 2, 1)
                        if kt % d == 0 and pm * pn * 6144 + 2 * d * (pm * X_TILE + pn * wt) <= W2_L1_BUDGET), None)
            pc = False if ibw is None else ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
                per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None, fuse_batch=True)
            self._w2_pc[M] = pc
        return pc or None

    def __call__(self, x):
        h = self.swiglu(x)
        ck = CK_BFP4_PREFILL if self.w2.dtype == ttnn.bfloat4_b else CK_BFP8
        y = ttnn.linear(h, self.w2, dtype=ttnn.bfloat16, compute_kernel_config=ck, memory_config=DRAM,
                        program_config=self.w2_config(h.shape[-2]))
        ttnn.deallocate(h)
        return y
