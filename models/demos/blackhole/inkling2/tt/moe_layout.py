# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Host side of the Inkling-2-Mini MoE: resident expert weight layout (DRAM bank streams) and the torch reference.

Pure torch (host venv and the image).  Used by moe.py (device upload) and the tests (layout emulation, reference).

Expert weights are stored per layer as two WIDTH_SHARDED DRAM tensors, one shard per DRAM bank (8 on the P150).
Every stored 32x32 tile is a weight block in the HF Linear orientation [out, in] (`k_aligned`, the default): BFP
shared exponents then run along K (16 consecutive inputs), the same blocks as the checkpoint's NVFP4 scales, so the
factored NVFP4 values (e2m1 * E4M3 scale) convert to BFP8 (near-)losslessly.  The kernels multiply with in1 tiles
transposed.  `k_aligned=False` stores W^T tiles (exponents along N) for comparison runs.

gate_up  [E*H, 2I] (H = 2304 = 72 k-tiles, I = 1152 = 36 intermediate tiles = 36 gate/up "pairs"):
  bank b holds, per expert e, `pairs(e, b)` consecutive pairs j in [start(e, b), start(e, b) + pairs(e, b)):
  36 pairs over 8 banks = 4.5 per bank, so banks alternate 5 / 4 pairs by the parity of (e + b) (each bank holds
  exactly 4.5 pairs per expert on average, the per-token imbalance is the parity mix of the selected experts).
  A pair is K row-major (k, {gate, up}) tiles (144 tiles, contiguous): one streamed unit = gate and up of one
  intermediate tile, so the softcapped SwiGLU runs on the core that owns both.
  Bank-b tile stream: expert-major, pair-major, k-major, {gate, up}.
down     [E*I, H]: bank b owns output tiles [9b, 9b + 9), split into `down_groups` column groups of `down_cols`
  tiles; per expert and group the [K=36, down_cols] block row-major (contiguous).
  Bank-b tile stream: expert-major, group-major, k-major, column.
A bank's tile stream fills its shard [rows, 9 tiles] row-major (tile L -> shard tile (L // 9, L % 9)).

Per-expert NVFP4 factors (`f13`, `f2`, source.experts_factored) are not folded into the stored values: the kernels
scale the gate/up accumulators by f13[e] (before the SwiGLU, as the reference's weights include it) and the routing
weight by f2[e] (down is linear).  MXFP8 layers and unfactored loads use factors of 1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BANKS = 8
TILE = 32


# ---------------------------------------------------------------- gate/up pair placement (kernels: same formulas)
def gu_base(inter_tiles: int, banks: int = BANKS) -> int:
    base, rem = divmod(inter_tiles, banks)
    if rem not in (0, banks // 2):
        raise ValueError(f"{inter_tiles} intermediate tiles over {banks} banks: need a remainder of 0 or banks/2")
    return base


def gu_odd(inter_tiles: int, banks: int = BANKS) -> int:
    """1 when the banks alternate base+1 / base pairs per expert (I/32 % banks == banks/2), else 0."""
    return int(inter_tiles % banks != 0)


def gu_pairs(e: int, b: int, inter_tiles: int, banks: int = BANKS) -> int:
    return gu_base(inter_tiles, banks) + (gu_odd(inter_tiles, banks) and (e + b) % 2 == 0)


def gu_start(e: int, b: int, inter_tiles: int, banks: int = BANKS) -> int:
    """First global pair index of bank b for expert e."""
    odd = gu_odd(inter_tiles, banks)
    return b * gu_base(inter_tiles, banks) + odd * ((b + (1 if e % 2 == 0 else 0)) // 2)


def gu_before(e: int, b: int, inter_tiles: int, banks: int = BANKS) -> int:
    """Pairs stored in bank b before expert e."""
    odd = gu_odd(inter_tiles, banks)
    return e * gu_base(inter_tiles, banks) + odd * ((e + (1 if b % 2 == 0 else 0)) // 2)


def _tiles(w: torch.Tensor) -> torch.Tensor:
    """[E, N, K] (HF Linear) -> [E, N/32, K/32, 32, 32] tiles of the [out, in] blocks."""
    e, n, k = w.shape
    return w.reshape(e, n // TILE, TILE, k // TILE, TILE).permute(0, 1, 3, 2, 4)


def _shard(stream: torch.Tensor, width_tiles: int) -> torch.Tensor:
    """Bank tile stream [L, 32, 32] -> shard [L/width*32, width*32] (tile L at (L // width, L % width))."""
    return stream.reshape(-1, width_tiles, TILE, TILE).permute(0, 2, 1, 3).reshape(-1, width_tiles * TILE)


def gate_up_layout(w_gate: torch.Tensor, w_up: torch.Tensor, banks: int = BANKS, k_aligned: bool = True):
    """w_gate, w_up [E, I, H] -> stored gate_up [E*H, 2I] (dtype kept)."""
    E, I, H = w_gate.shape
    it = I // TILE
    g, u = _tiles(w_gate), _tiles(w_up)  # e, j, k, 32n, 32k
    pair = torch.stack([g, u], dim=3)  # e, j, k, {g,u}, 32, 32
    if not k_aligned:
        pair = pair.transpose(-1, -2)
    width = 2 * it // banks
    if 2 * it % banks:
        raise ValueError("2I/32 must split evenly over the banks")
    shards = []
    for b in range(banks):
        parts = []
        for e in range(E):
            s, p = gu_start(e, b, it, banks), gu_pairs(e, b, it, banks)
            parts.append(pair[e, s : s + p].reshape(-1, TILE, TILE))
        shards.append(_shard(torch.cat(parts), width))
    return torch.cat(shards, dim=1)


def gate_up_dup_layout(w_gate: torch.Tensor, w_up: torch.Tensor, banks: int = BANKS, k_aligned: bool = True):
    """Floating-pair copies for the bank-balanced decode (kernels: moe_route.hpp balance_floating / bank_units):
    w_gate, w_up [E, I, H] -> [E/2 * 2H * banks / (2I/32), 2I] (Inkling: [32768, 2304]), stored like gate_up (one
    shard per bank).  Bank b holds, for every expert e with (e + b) odd in ascending e (dup unit e // 2), the last pair
    of bank b-1's run of e (pair gu_start(e, b-1) + base, bank b-1's (base+1)-th pair): either bank can stream it."""
    E, I, H = w_gate.shape
    it = I // TILE
    base = gu_base(it, banks)
    if not gu_odd(it, banks):
        raise ValueError("no floating pairs: the intermediate tiles split evenly over the banks")
    g, u = _tiles(w_gate), _tiles(w_up)
    pair = torch.stack([g, u], dim=3)  # e, j, k, {g,u}, 32, 32
    if not k_aligned:
        pair = pair.transpose(-1, -2)
    width = 2 * it // banks
    shards = []
    for b in range(banks):
        prev = (b - 1) % banks
        parts = [pair[e, gu_start(e, prev, it, banks) + base].reshape(-1, TILE, TILE)
                 for e in range(E) if (e + b) % 2 == 1]
        shards.append(_shard(torch.cat(parts), width))
    return torch.cat(shards, dim=1)


def down_layout(w_down: torch.Tensor, banks: int = BANKS, cols: int = 3, k_aligned: bool = True):
    """w_down [E, H, I] -> stored down [E*I, H]."""
    E, H, I = w_down.shape
    nt = H // TILE
    per_bank = nt // banks
    if nt % banks or per_bank % cols:
        raise ValueError("output tiles must split into banks x column groups")
    t = _tiles(w_down)  # e, n, k, 32n, 32k
    if not k_aligned:
        t = t.transpose(-1, -2)
    kt = I // TILE
    t = t.reshape(E, banks, per_bank // cols, cols, kt, TILE, TILE).permute(1, 0, 2, 4, 3, 5, 6)
    return torch.cat([_shard(t[b].reshape(-1, TILE, TILE), per_bank) for b in range(banks)], dim=1)


# ---------------------------------------------------------------- reference (modeling_inkling2_loop.py semantics)
def ref_route(x: torch.Tensor, router_w: torch.Tensor, bias: torch.Tensor, top_k: int = 8, scale: float = 8.0):
    """Inkling2LoopTopKRouter: x [T, H] -> (scores fp32 [T, E], weights fp32 [T, k], ids [T, k])."""
    logits = F.linear(x.float(), router_w.float())
    scores = torch.sigmoid(logits)
    ids = torch.topk(scores + bias.float(), top_k, dim=-1, sorted=False).indices
    w = scores.gather(1, ids)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return scores, w * scale, ids


def softcapped_swiglu(gate: torch.Tensor, up: torch.Tensor, gcap: float = 4.0, ucap: float = 25.0):
    dt = gate.dtype
    g, u = gate.float(), up.float()
    return (torch.tanh(g / gcap) * gcap * torch.sigmoid(g) * torch.tanh(u / ucap) * ucap).to(dt)


def ref_experts(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, w_gate, w_up, w_down) -> torch.Tensor:
    """Inkling2LoopExperts in the model dtype of the weights (BF16 in the HF model): x [T, H], ids/weights [T, k];
    w_* indexed by expert id (tensors [E, ...] or a dict id -> tensor).  fp32 accumulation, output in x.dtype."""
    dt = w_gate[int(ids[0, 0])].dtype
    xx = x.to(dt)
    out = torch.zeros(x.shape, dtype=torch.float32)
    for e in torch.unique(ids).tolist():
        tok, pos = torch.where(ids == e)
        gate = F.linear(xx[tok], w_gate[e])
        up = F.linear(xx[tok], w_up[e])
        h = F.linear(softcapped_swiglu(gate, up), w_down[e])
        out.index_add_(0, tok, (h * weights[tok, pos, None]).float())
    return out.to(x.dtype)
