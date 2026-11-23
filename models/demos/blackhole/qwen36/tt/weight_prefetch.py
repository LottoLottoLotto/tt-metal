# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRISC tensor prefetcher for exact single-device decode matmuls on Blackhole.

The Blackhole DRAM cores stream receiver-contiguous weights (one NdShardSpec shard per matmul worker,
round-robin over the eight banks) into a per-family global circular buffer (GCB). The consuming 1D
matmul multicasts the activation exactly like the DRAM-reading path and pops weight K-blocks in natural
order with the same in0_block_w, so K blocking and accumulation order are unchanged (bitwise-equal
outputs). Because the stream runs on the DRAM cores, the next prefetched matmul's first blocks land in
worker L1 while the preceding non-matmul work (GDN recurrence, attention, norms) runs.

Every prefetched matmul queues its own request immediately before dispatch. Eager calls send the
request now; during trace capture the request is recorded and re-sent by every execute_trace, so each
replay pushes exactly the pages its matmuls consume.

Families (enable with QWEN36_WEIGHT_PREFETCH=<comma list>, single device only):
  proj_o  [6144, 5120] BF8 GDN out_proj (K block 8) and attention o_proj (K block 4), 80 workers x 2 cols
  down    [17408, 5120] BFP4 MLP down (K block 8), 80 workers x 2 cols
  gate_up [5120, 34816] BFP4 MLP [up|gate] (K block 2), 64 workers x 17 cols
The routed decode matmul keeps its original DRAM weight for every other caller (prefill, fallbacks).
"""
import os

import ttnn

_BANKS = 8
_GRID_W = 11
_started = []  # meshes with a running prefetcher
_routes = {}  # id(original weight) -> (family, prefetch weight, original weight)
_armed = []  # meshes whose decode matmuls route through the prefetcher
_reserved = {}  # family name -> L1 reservation tensor covering its GCB FIFO (held outside prefill)

_SPECS = {
    # name: (K, N, dtype, receivers, K blocks of the consumers, default per-receiver GCB KB)
    "proj_o": (6144, 5120, "bf8", 80, (4, 8), 196),
    "down": (17408, 5120, "bf4", 80, (8,), 36),
    "gate_up": (5120, 34816, "bf4", 64, (2,), 150),
}


def enabled_families(num_devices=1):
    if num_devices != 1:
        return ()
    return tuple(f for f in os.environ.get("QWEN36_WEIGHT_PREFETCH", "").split(",") if f in _SPECS)


def enabled(name, num_devices=1):
    return name in enabled_families(num_devices)


class PrefetchFamily:
    """One weight shape fed through one GCB; one program config per consumer K block."""

    def __init__(self, name):
        k, n, dtype, receivers, kbs, gcb_kb = _SPECS[name]
        gcb_kb = int(os.environ.get(f"QWEN36_WEIGHT_PREFETCH_{name.upper()}_KB", gcb_kb))
        self.dtype = ttnn.bfloat8_b if dtype == "bf8" else ttnn.bfloat4_b
        tile_bytes = 1088 if dtype == "bf8" else 576
        n_tiles = n // 32
        per_core_n = n_tiles // receivers
        if per_core_n * receivers != n_tiles or any((k // 32) % kb for kb in kbs):
            raise ValueError(f"{name}: [{k}, {n}] does not split over {receivers} receivers / K blocks {kbs}")
        self.name, self.k, self.n, self.receivers = name, k, n, receivers
        self.program_configs = {
            kb: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(_GRID_W, (receivers + _GRID_W - 1) // _GRID_W),
                in0_block_w=kb,
                out_subblock_h=1,
                out_subblock_w=max(d for d in range(1, 5) if per_core_n % d == 0),  # fp32 dest: <= 4 tiles
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            for kb in kbs
        }
        self.memory_config = ttnn.MemoryConfig(
            ttnn.BufferType.DRAM,
            ttnn.NdShardSpec(
                ttnn.Shape([k, n // receivers]),
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_BANKS - 1, 0))}),
                ttnn.ShardOrientation.ROW_MAJOR,
                ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
            ),
        )
        page = max(kbs) * per_core_n * tile_bytes  # a multiple of every consumer's K-block page
        self.gcb_bytes = max(2, gcb_kb * 1024 // page) * page
        self.gcb = None
        self.fifo_address = None
        self.block_counts = {}
        # Decode consumption order (registration == layer order): (prefetch weight, consumer K block).
        self.sequence = []
        self.calls = 0

    def upload(self, torch_weight, mesh, cache_path):
        """Upload a logical [K, N] weight in the receiver-contiguous layout (tile values unchanged)."""
        return ttnn.as_tensor(
            torch_weight,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=self.memory_config,
            cache_file_name=cache_path,
        )

    def bind(self, mesh, weight):
        """Create the GCB once, from the first uploaded weight."""
        if self.gcb is not None:
            return
        cores = [ttnn.CoreCoord(i % _GRID_W, i // _GRID_W) for i in range(self.receivers)]
        # Shard s (matmul worker s in row-major order) lives on bank s % 8.
        bank_to_receivers = [
            (bank, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores[bank::_BANKS]])) for bank in range(_BANKS)
        ]
        pcs = list(self.program_configs.values())
        # The FIFO is a non-owning view over an L1 reservation that is held while decode programs run
        # and released for prefill (see reserve/release); the GCB config block lives in L1_SMALL.
        self.fifo_address = _reserve(mesh, self)
        os.environ["TT_GCB_UNTRACKED_DATA_ADDR"] = str(self.fifo_address)
        try:
            self.gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
                mesh, pcs, [weight] * len(pcs), bank_to_receivers, self.gcb_bytes
            )
        finally:
            del os.environ["TT_GCB_UNTRACKED_DATA_ADDR"]
        self.block_counts = {
            kb: ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(pc, weight, self.gcb)
            for kb, pc in self.program_configs.items()
        }

    def linear(self, x, weight, in0_block_w, **kwargs):
        mesh = x.device()
        self.bind(mesh, weight)
        _ensure_started(mesh)
        if len(_active_families()) == 1:
            # Sole prefetched family: one request per decode pass streams every layer's weight in
            # consumption order (a single host->DRISC message per trace replay instead of one per layer).
            index = self.calls % len(self.sequence)
            expected, expected_kb = self.sequence[index]
            if expected is not weight or expected_kb != in0_block_w:
                raise RuntimeError(f"weight prefetch {self.name}: decode call {index} is out of layer order")
            if index == 0:
                ttnn.experimental.queue_tensor_prefetcher_request(
                    mesh, [(w, self.block_counts[kb]) for w, kb in self.sequence], self.gcb
                )
            self.calls += 1
        else:
            # Several families interleave per layer; the DRISC serves requests in order, so each
            # matmul queues its own block stream.
            ttnn.experimental.queue_tensor_prefetcher_request(
                mesh, [(weight, self.block_counts[in0_block_w])], self.gcb
            )
        return ttnn.linear(x, weight, program_config=self.program_configs[in0_block_w], global_cb=self.gcb, **kwargs)


_families = {}


def family(name):
    if name not in _families:
        _families[name] = PrefetchFamily(name)
    return _families[name]


def register(name, original, torch_weight, mesh, cache_path, in0_block_w):
    """Upload the prefetch copy of `original` ([K, N] logical torch weight) and route its decode matmuls
    (K block `in0_block_w`). Register in decode layer order."""
    fam = family(name)
    pf = fam.upload(torch_weight, mesh, cache_path)
    _routes[id(original)] = (fam, pf, original)
    fam.sequence.append((pf, in0_block_w))
    return pf


def linear(x, weight, program_config=None, **kwargs):
    """ttnn.linear, routed through the prefetcher for registered weights at their decode K block."""
    route = _routes.get(id(weight))
    if (
        route is not None
        and any(m is x.device() for m in _armed)
        and route[2] is weight
        and program_config is not None
        and getattr(program_config, "mcast_in0", False)
        and program_config.per_core_M == 1
        and program_config.in0_block_w in route[0].program_configs
        and _rows(x) <= 32
    ):
        return route[0].linear(x, route[1], program_config.in0_block_w, **kwargs)
    return ttnn.linear(x, weight, program_config=program_config, **kwargs)


def _active_families():
    return {id(fam): fam for fam, _, _ in _routes.values()}


def _rows(x):
    rows = 1
    for d in list(x.padded_shape)[:-1]:
        rows *= int(d)
    return rows


def _ensure_started(mesh):
    if any(m is mesh for m in _started):
        return
    ttnn.experimental.start_tensor_prefetcher(mesh)
    # Weights were uploaded over CQ0; the stream must not read DRAM before those writes land.
    ttnn.experimental.wait_for_cq_on_tensor_prefetcher(mesh, 0)
    _started.append(mesh)
    _wrap_close()


def arm(mesh):
    """Route registered decode matmuls through the prefetcher from now on. Returns True when this call
    armed it, so the caller must run one eager step (building the GCB-fed programs) before a capture.

    Arm only after every prefill program is compiled and captured: the GCBs are created at the first
    routed matmul and occupy L1 that prefill circular buffers use. Prefill trace replays are not
    re-validated and may overwrite the (then empty) GCB region; fence() orders the next stream after them.
    """
    if not _routes or any(m is mesh for m in _armed):
        return False
    # Allocate every GCB now, before decode transients exist, so they pack against persistent L1.
    for fam, pf, _ in _routes.values():
        fam.bind(mesh, pf)
    _armed.append(mesh)
    return True


def _reserve(mesh, fam):
    """Allocate the L1 reservation for `fam`'s FIFO (every worker, one shard of gcb_bytes per core)."""
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    rows = fam.gcb_bytes // 64  # [rows, 32] BF16 row-major shard == gcb_bytes
    reservation = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows * grid.x * grid.y, 32]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                          ttnn.ShardSpec(cores, [rows, 32], ttnn.ShardOrientation.ROW_MAJOR)),
    )
    _reserved[fam.name] = reservation
    return reservation.buffer_address()


def release(mesh):
    """Free the FIFO reservations for prefill. Call only with the prefetcher idle (after decode completed:
    every queued block has been consumed by its matmul, which precedes the prefill in CQ order)."""
    if not any(m is mesh for m in _armed):
        return
    for name in list(_reserved):
        ttnn.deallocate(_reserved.pop(name))


def reacquire(mesh):
    """After prefill: fence the next stream behind it and re-reserve every FIFO at its fixed address."""
    if not any(m is mesh for m in _armed):
        return
    fence(mesh)
    for fam in _families.values():
        if fam.gcb is not None and fam.name not in _reserved:
            address = _reserve(mesh, fam)
            if address != fam.fifo_address:
                raise RuntimeError(
                    f"weight prefetch {fam.name}: L1 reservation moved {fam.fifo_address} -> {address}; "
                    "a prefill-lifetime L1 tensor is still allocated"
                )


def fence(mesh):
    """Order later prefetch streams after all work already enqueued on CQ0 (e.g. a prefill replay whose
    circular buffers overlap the GCB FIFOs)."""
    if any(m is mesh for m in _started):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(mesh, 0)


def stop(mesh):
    for i, m in enumerate(_started):
        if m is mesh:
            ttnn.experimental.stop_tensor_prefetcher(mesh)
            del _started[i]
            return


def _wrap_close():
    """The prefetcher must be stopped before its device closes; hook ttnn.close_mesh_device once."""
    close = ttnn.close_mesh_device
    if getattr(close, "_qwen_prefetch_stop", False):
        return

    def close_mesh_device(mesh_device, *args, **kwargs):
        stop(mesh_device)
        return close(mesh_device, *args, **kwargs)

    close_mesh_device._qwen_prefetch_stop = True
    ttnn.close_mesh_device = close_mesh_device
