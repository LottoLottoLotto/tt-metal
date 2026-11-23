# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Prefix caching: exact prefix resume from device snapshots of a sequence slot's running state (tt/NOTES-prefix.md).

A prefill chunk [S, ..) of a sequence reads, besides the full-attention K / V of positions < S, only the slot's running
state: the 40 sliding K / V rings, the 48 K / V conv-input rings and the 96 glue conv rings (+ registered extras, e.g.
the batched drafter's rings).  The state a prefill reaches at a chunk boundary S therefore resumes any prompt that
starts with the same S tokens, bitwise, when the resumed prefill restarts the same chunk schedule (S a multiple of the
server's chunk alignment).

  PrefixSnapshots(model, M)  device pool of M snapshots (every slot-major state tensor mirrored with M slots, allocated
                             at construction: trace safety); save(slot, j) / restore(j, slot): exact copies (ttnn.slice
                             of the source slot + ttnn.fill_cache into the target slot; call-local temporaries only).
  IdentityRows               single stream: the full-attention / drafter caches are in identity order (vLLM's blocks are
                             bookkeeping), so a snapshot of [0, S) is valid while no prefill rewrote a row below S.
  PrefixCache                the serving side over vllm_tt_plugin.prefix_resume.REGISTRY (or a stand-in with its API):
                             which boundaries a prefill publishes, publish (pool slot, LRU eviction through the
                             registry), resume (take + restore).
"""
from __future__ import annotations

import ttnn

from .common import DRAM

_BYTES = {ttnn.bfloat16: 2.0, ttnn.float32: 4.0, ttnn.bfloat8_b: 1088 / 1024, ttnn.bfloat4_b: 576 / 1024}


def slot_state(model) -> list:
    """The model's slot-major running-state tensors (dim 0 = sequence slot; S = 1 in single stream): every sliding
    application's K / V ring, every application's K / V conv-input ring, the rows glue's conv rings."""
    if model.rows_glue is None:
        raise ValueError("prefix snapshots need the fused rows glue (INK_GLUE=ttnn keeps ShortConv states)")
    out = []
    for a in model.attn:
        for p in range(a.n_slots):
            if not a.full:
                out += [a.k_cache[p], a.v_cache[p]]
            out.append(a.conv_ring[p])
    return out + list(model.rows_glue.rings.values())


def _nbytes(t) -> int:
    n = 1
    for d in list(t.shape)[1:]:
        n *= int(d)
    return int(n * _BYTES[t.dtype])


def _copy(src, i: int, dst, j: int):
    """dst[j] = src[i] (dim 0), exact."""
    x = src
    if int(src.shape[0]) != 1:
        shape = list(src.shape)
        x = ttnn.slice(src, [i, 0, 0, 0], [i + 1] + [int(d) for d in shape[1:]], memory_config=DRAM)
    ttnn.fill_cache(dst, x, j)
    if x is not src:
        ttnn.deallocate(x)


class PrefixSnapshots:
    """Device pool of `count` snapshots of one sequence slot's running state.  Construct before any trace capture
    (the model constructor); register(tensors) adds slot-major extras (also before any capture)."""

    def __init__(self, model, count: int):
        if count < 1:
            raise ValueError(f"prefix snapshots: count {count} < 1")
        self.device, self.count = model.device, count
        self.src, self.pool = [], []
        self.nbytes = 0  # bytes of one snapshot
        self.free = list(range(count - 1, -1, -1))  # pop() -> the lowest free index
        self.register(slot_state(model))

    def register(self, tensors):
        for t in tensors:
            shape = [self.count] + [int(d) for d in list(t.shape)[1:]]
            self.pool.append(ttnn.zeros(shape, dtype=t.dtype, layout=ttnn.TILE_LAYOUT, device=self.device,
                                        memory_config=DRAM))
            self.src.append(t)
            self.nbytes += _nbytes(t)

    def acquire(self) -> int | None:
        return self.free.pop() if self.free else None

    def release(self, j: int):
        if not 0 <= j < self.count or j in self.free:
            raise RuntimeError(f"prefix snapshot {j} released twice / unknown")
        self.free.append(j)
        self.free.sort(reverse=True)

    def save(self, slot: int, j: int):
        """Snapshot j = the state of sequence slot `slot` now."""
        for t, p in zip(self.src, self.pool):
            _copy(t, slot, p, j)

    def restore(self, j: int, slot: int):
        """Sequence slot `slot` = snapshot j (every ring of the slot overwritten)."""
        for t, p in zip(self.src, self.pool):
            _copy(p, j, t, slot)


class IdentityRows:
    """Single stream: per 64-row block of the identity-order caches, the sequence number of the last prefill that
    wrote it.  write(start, end) before a prefill of rows start..end-1 (rounded out to whole blocks: the last chunk's
    padding rows and the drafter's whole-block context writes); a snapshot of [0, S) stamped at publish time stays
    valid while every block below S is older than the stamp.  Decode / verify rows lie above the writing sequence's
    prompt, so a valid snapshot (S <= that prompt's resume point) never sees them: prefill writes are enough."""

    BLOCK = 64

    def __init__(self, max_seq_len: int):
        self.ver = [0] * (-(-max_seq_len // self.BLOCK) + 1)
        self.seq = 0

    def write(self, start: int, end: int):
        self.seq += 1
        for b in range(start // self.BLOCK, min(len(self.ver), -(-end // self.BLOCK))):
            self.ver[b] = self.seq

    def clear(self):
        """Every row rewritten (warmup / capture): no snapshot survives."""
        self.write(0, len(self.ver) * self.BLOCK)

    def valid(self, stamp: int, length: int) -> bool:
        return max(self.ver[:length // self.BLOCK], default=0) <= stamp


class Handle:
    """Registry payload: pool slot `index` of the PrefixCache `owner`; valid() (single stream) = its identity rows are
    intact."""

    __slots__ = ("owner", "index", "length", "_stamp")

    def __init__(self, owner: "PrefixCache", index: int, length: int):
        self.owner, self.index, self.length = owner, index, length
        self._stamp = owner.rows.seq if owner.rows is not None else 0

    def valid(self) -> bool:
        rows = self.owner.rows
        return rows is None or rows.valid(self._stamp, self.length)


class PrefixCache:
    """Exact prefix resume for one Inkling2Model (tt/NOTES-prefix.md).  registry: vllm_tt_plugin.prefix_resume.REGISTRY
    (configure / active / put / take / resuming / has / wanted / evict_one / shareable, payload valid() and on_drop;
    configure drops every snapshot of an earlier configuration); align: the server's chunk alignment (batched: 2048 =
    the prefill turn budget, single stream: the model's prefill_chunk); identity: the model's full caches are in
    identity order (single stream) -> IdentityRows validity."""

    def __init__(self, model, registry, align: int, *, identity: bool, block: int = 64, fork_hints: bool = False,
                 log=print):
        if model.snaps is None:
            raise ValueError("prefix caching needs the model's snapshot pool (Inkling2Model(prefix_snapshots=M))")
        if align % block or align % 32:
            raise ValueError(f"prefix resume alignment {align}: a multiple of the {block}-token KV block")
        self.model, self.snaps, self.reg, self.align, self.log = model, model.snaps, registry, align, log
        self.rows = IdentityRows(model.max_seq_len) if identity else None
        self.stats = {"published": 0, "resumed": 0, "skipped_full": 0}
        registry.configure(align, block, self.snaps.count, fork_hints=fork_hints, on_drop=self._dropped)

    @property
    def active(self) -> bool:
        return bool(self.reg.active)

    def _dropped(self, payload):
        if isinstance(payload, Handle) and payload.owner is self:
            self.snaps.release(payload.index)

    def invalidate_rows(self):
        """Single stream: the identity caches were overwritten outside a tracked prefill (warmup, capture)."""
        if self.rows is not None:
            self.rows.clear()

    # ---- one prefill call (rows start..end-1 of a prompt of the request; final: the prompt ends here)
    def resuming(self, request_id, start: int) -> bool:
        """Whether this prefill continues from a snapshot the scheduler found (vs a chunk continuation)."""
        if not start or not self.active:
            return False
        if request_id is None:
            raise RuntimeError("prefix resume at a nonzero start needs the request id (vllm_tt_plugin prefix patch)")
        return bool(self.reg.resuming(request_id))

    def resume(self, request_id, tokens, start: int, slot: int):
        """Take the snapshot of tokens[:start] the scheduler resumed this request from and restore it into `slot`."""
        h = self.reg.take(tokens, start, request_id=request_id)
        if not isinstance(h, Handle) or h.length != start:
            raise RuntimeError(f"prefix resume: no snapshot of the scheduled {start}-token prefix")
        if not h.valid():
            raise RuntimeError(f"prefix resume: the {start}-token snapshot's identity rows were overwritten")
        self.model.restore_slot(h.index, slot)
        self.stats["resumed"] += 1

    def boundary(self, tokens, pos: int, end: int, final: bool, resumed_at: int, request_id=None) -> bool:
        """At chunk boundary pos of a prefill ending at `end`: publish the state there?  The final segment's start
        (pos = end // align * align of a final prefill), the prompt's end when aligned, any boundary the scheduler
        asked for (fork hints: another cached prompt shares the prefix up to it); never the state just restored, never
        a request whose state must not be shared (cache_salt / LoRA / multimodal: registry.shareable)."""
        if not self.active or pos <= 0 or pos % self.align or pos == resumed_at:
            return False
        if request_id is not None and not self.reg.shareable(request_id):
            return False
        if pos == end:
            ok = final
        else:
            ok = (final and pos == end // self.align * self.align) or bool(self.reg.wanted(tokens, pos))
        return ok and not self.reg.has(tokens, pos)

    def publish(self, tokens, pos: int, slot: int, block_ids) -> bool:
        """Snapshot slot `slot` (holding tokens[:pos]) and register it with the KV blocks of [0, pos)."""
        j = self.snaps.acquire()
        if j is None and self.reg.evict_one():
            j = self.snaps.acquire()
        if j is None:
            self.stats["skipped_full"] += 1
            return False
        self.model.snapshot_slot(slot, j)
        try:
            self.reg.put(tokens, pos, block_ids, Handle(self, j, pos), self.snaps.nbytes)
        except Exception:
            self.snaps.release(j)
            raise
        self.stats["published"] += 1
        return True

    def note_prefill(self, start: int, end: int):
        """Single stream: rows start..end-1 are about to be (re)written by a prefill."""
        if self.rows is not None:
            self.rows.write(start, end)
