# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""LFM3 block drafter on the P150 (DFlash-style; trained off-device, see Lottolabs/LFM3-1.7B-DFlash-Drafter).

One drafter pass proposes 7 tokens for a stream from the target's hidden states (the residual stream after layers
2, 12 and 23 = "taps"):

    ctx[p]  = hidden_norm(fc(cat(rms(tap2[p]), rms(tap12[p]), rms(tap23[p]))))         every committed position p
    K/V[l][p] = k_norm(k_l(ctx[p])) roped at p, v_l(ctx[p])                          per drafter layer l (3)
    block rows i = 0..7 at positions a + i (a = the anchor: the stream's pending token):
      h_i = embed(token[a]) + ctx_gain * ctx[a - 1] + mask[i]
      3 x (pre-norm attention over K/V of positions (a + 7 - 2048, a - 1] and the 8 block rows (no mask inside the
           block), pre-norm SwiGLU 2048 -> 6144)
      draft i = argmax over the 65,536-token draft vocabulary of norm(h_i) @ head^T     (rows 1..7; head = the
           target's lm_head rows of the draft vocabulary)

Device layout (this module):
  * K/V: a paged BFP8 cache per drafter layer with the target's page table and block size (vLLM's pages of a request
    carry both models' K/V, so prefix caching keeps the drafter context too): [blocks + 1 + streams, 4, 64, 128];
    block `blocks` is the target's trash block (padded prefill rows); the last `streams` blocks take the draft rows of
    a pass whose positions lie in pages holding no committed position yet (one per drafted stream).
  * taps: every target decode / verify step copies its rows' taps into fixed per-width buffers (FastStep,
    enable_drafter; the target's values are only read).
  * commit (a separate trace per step kind, launched right after the target step when the drafter context is
    kept): the taps -> ctx and the three layers' K/V rows of every live row at its position (AttnNormQKVFront with
    the hidden norm as its norm and the zero-gated q|k|v|g weight; its q output is dropped; verify steps group the
    rows by the target's row map), ctx_gain * ctx of every row -> `ctx32` (row r = decode row r of the step) for the
    next pass's ctx[a - 1].  Rows past a verify group's accepted count write K/V at positions >= the next anchor:
    rewritten by the next step before any pass reads them (as the target's own KV).
  * prefill_commit (eager, every prefill chunk): the same for the chunk's rows (AttnFrontPrefill, as the target's
    prefill attention front).
  * draft (traced per stream count n, R = 8 n rows): the anchor's embedding + one selection matmul (ctx_gain * ctx
    row of the stream + mask row of the block row), three layers of AttnNormQKVFront (verify mode: groups of 8
    consecutive positions; writes the block K/V) + SDPADecode (cur_pos a + 7 for every row of a group, window 2048:
    the block is non-causal) + o_proj + FusedMLPDecode, final norm, BFP4 head over the draft vocabulary, top-2 per row
    (kernels/drafter_argmax.cpp: the draft and its logit margin over the runner-up, the confidence feature).

The output gate of the reused attention kernels is fed zero gate weights (sigmoid(0) = 0.5 exactly); o_proj carries
the compensating factor 2 (exact in block float).  Weights: export_tt.py's BF16 safetensors converted here (FMT: fc and
attention BFP8, MLP and head BFP4: the values drafter.py's snap_device evaluates offline).
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from .common import CK_BFP8, CK_HIFI4, CK_NORM, WeightCache, free
from .config import SLIDING
from .layers import DRAM, Attention, linear, norm_weight

L1 = ttnn.L1_MEMORY_CONFIG
KERNELS = Path(__file__).resolve().parent / "kernels"
DEFAULT_DIR = "drafter"  # the export directory (config.json, model.safetensors, ...); serving sets LFM3_DRAFTER
BLOCK_ROWS = 8          # anchor + 7 drafts (= verify width 8)
DRAFTS = BLOCK_ROWS - 1
WINDOW = 2048           # SDPADecode window (multiple of 64): keys (a + 7 - 2048, a + 7] -> 2040 context rows
MAX_STREAMS = 2         # drafted streams per pass (R = 8 n <= 16 rows: the attention kernels' limit)
TAP_EPS = 1e-6
FMT = {"fc": ttnn.bfloat8_b, "attn": ttnn.bfloat8_b, "mlp": ttnn.bfloat4_b, "head": ttnn.bfloat4_b}
_FMT_NAME = {ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}


def snap(t: torch.Tensor, dtype) -> torch.Tensor:
    """BF16 values of `t` after ttnn's host conversion to `dtype` (what the device holds; blocks along the last dim)."""
    t = t.to(torch.bfloat16).contiguous()
    if dtype == ttnn.bfloat16:
        return t
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).reshape(t.shape).to(torch.bfloat16)


def key_values(keys: np.ndarray) -> np.ndarray:
    """drafter_argmax.cpp's keys (monotonic map of BF16 bits, + 1) -> the BF16 values (float32)."""
    k = keys.astype(np.int64) - 1
    u = np.where(k & 0x8000, k & 0x7FFF, (~k) & 0xFFFF).astype(np.uint32)
    return (u << 16).view(np.float32)


class _Source:
    """The drafter's attention weights under the target's tensor names (layers.Attention / load_linear)."""

    def __init__(self, sd: dict, identity: str):
        self.identity = identity
        self.t = {}
        for j in range(3):
            p, a = f"layers.{j}", f"model.layers.{j}.self_attn"
            for n in ("q", "k", "v"):
                self.t[f"{a}.{n}_proj.weight"] = sd[f"{p}.{n}.weight"]
            self.t[f"{a}.gate_proj.weight"] = torch.zeros(16, sd[f"{p}.q.weight"].shape[1], dtype=torch.bfloat16)
            self.t[f"{a}.o_proj.weight"] = (sd[f"{p}.o.weight"].float() * 2).to(torch.bfloat16)  # sigmoid(0) = 1/2
            self.t[f"{a}.q_norm.weight"] = sd[f"{p}.q_norm.weight"]
            self.t[f"{a}.k_norm.weight"] = sd[f"{p}.k_norm.weight"]

    def keys(self):
        return list(self.t)

    def get(self, name):
        return self.t[name]

    def fmt(self, name):
        return _FMT_NAME[FMT["attn"]]


class DrafterArgmax:
    """kernels/drafter_argmax.cpp: per row (rows <= 16) of BF16 TILE logits [1, 1, rows, V] -> uint32 [1, 48] page
    (idx, key1, key2 per row)."""

    def __init__(self, device):
        self.device = device
        self._progs = {}

    def _build(self, logits, out, rows, tiles):
        dev = self.device
        grid = dev.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        workers = 2 * len(cores)
        gather = dev.worker_core_from_logical_core(cores[0])
        named = [("rows", rows), ("tiles", tiles), ("workers", workers), ("gather_x", gather.x),
                 ("gather_y", gather.y)]
        common = [int(logits.buffer_address()), int(out.buffer_address())]
        per_core = [(c, [i]) for i, c in enumerate(cores)]
        core_set = ttnn.CoreRangeSet({ttnn.CoreRange(cores[0], cores[-1])})
        kernel = UnifiedKernelDescriptor(
            kernel_source=str(KERNELS / "drafter_argmax.cpp"), core_ranges=core_set,
            brisc_named_compile_time_args=named, ncrisc_named_compile_time_args=named,
            brisc_common_runtime_args=common, ncrisc_common_runtime_args=common,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(brisc_args=per_core, ncrisc_args=per_core),
        )
        rb = (rows * 32 + 63) // 64 * 64
        span = max(1, math.ceil(tiles / workers)) * 2 * rb
        slot = (rows * 12 + 15) // 16 * 16
        cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=core_set,
                                 format_descriptors=[ttnn.CBFormatDescriptor(cb, ttnn.uint32, size)])
               for cb, size in ((0, max(span, slot)), (1, span), (2, workers * slot))]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=core_set, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=sems)

    def __call__(self, logits, rows: int, out):
        if (logits.dtype != ttnn.bfloat16 or logits.layout != ttnn.TILE_LAYOUT or not 1 <= rows <= 16
                or int(logits.padded_shape[-2]) != 32 or logits.memory_config() != DRAM):
            raise ValueError("DrafterArgmax needs DRAM BF16 TILE logits with <= 16 rows in one tile row")
        tiles = int(logits.padded_shape[-1]) // 32
        sig = (int(logits.buffer_address()), int(out.buffer_address()), rows, tiles)
        if self._progs.get(rows, (None,))[0] != sig:
            self._progs[rows] = (sig, self._build(logits, out, rows, tiles))
        ttnn.generic_op([logits, out], self._progs[rows][1])
        return out


class Drafter:
    """The drafter of an Lfm3Model with the fast decode step (see the module docstring).

    Lifecycle: Drafter(model) (weights) -> allocate(num_blocks, page_blocks, widths) (K/V caches, tap buffers, fixed
    trace inputs; before any trace capture) -> FastStep.enable_drafter(self) (the target steps export their taps) ->
    capture() (draft traces) and capture_commit(...) per target step kind -> per round: launch_commit(key) after a
    target step; stage(...) + launch(n) + read(n) for a draft pass."""

    def __init__(self, model, path: str | None = None, max_streams: int = MAX_STREAMS, log=print):
        from safetensors.torch import load_file

        from .fast.attn_fused import AttnNormQKVFront, SDPADecode
        from .fast.decode_matmul import PRESETS, Preset, StreamDecodeMatmul, cached_stream_weight
        from .fast.mlp_fused import W13_PRESETS, FusedMLPDecode
        from .fast.residual_norm import ResidualRMSNorm
        from .speculate import map_word

        self.path = Path(path or os.environ.get("LFM3_DRAFTER", DEFAULT_DIR))
        self.m, self.device, self.log = model, model.device, log
        dev = self.device
        self.cfg = json.loads((self.path / "config.json").read_text())
        c = self.cfg
        if (c["hidden"], c["heads"], c["kv_heads"], c["head_dim"], c["layers"], c["block"], c["theta"]) != (
                2048, 16, 4, 128, 3, BLOCK_ROWS, 10000.0) or not c.get("direct"):
            raise ValueError(f"drafter config {c} is not the supported layout")
        rope = model.cfg.rope[SLIDING]
        if rope["rope_theta"] != c["theta"] or rope["partial_rotary_factor"] != 1:
            raise ValueError("the drafter's RoPE must equal the target's sliding-attention RoPE")
        self.taps = [int(t) for t in c["taps"]]
        self.vocab = torch.tensor(json.loads((self.path / "draft_vocab.json").read_text()), dtype=torch.int64)
        self._vocab_np = self.vocab.numpy()
        sd = load_file(str(self.path / "model.safetensors"))
        fmts = "-".join(_FMT_NAME[FMT[k]] for k in ("fc", "attn", "mlp", "head"))
        self.identity = f"drafter-{c['sha256'][:12]}-{fmts}"
        cache_dir = model.cache.dir.parent if model.cache.enabled else None
        self.cache = (WeightCache(dev, self.identity, root=cache_dir) if cache_dir is not None
                      else WeightCache(dev, self.identity, enabled=False))
        self.max_streams = max_streams

        def w(name):
            return sd[name].float()

        # attention layers (q|k|v|zero gate fused, o_proj x 2), the target's sliding-attention RoPE tables
        dcfg = copy.copy(model.cfg)
        dcfg.layer_types = [SLIDING] * 3
        dcfg.sliding_window = WINDOW
        src = _Source(sd, self.identity)
        self.attn = [Attention(dev, dcfg, src, j, self.cache) for j in range(3)]
        self.rope = model.rope[SLIDING]

        def stream(key, preset, dtype, build, key3=None, build3=None):
            return cached_stream_weight(self.cache, key, preset, dtype, key3=key3, build=build, build3=build3)

        # fc 6144 -> 2048: stream copy (decode commit) + resident (prefill commit)
        fc_t = lambda: snap(w("fc.weight").T, FMT["fc"])  # noqa: E731
        self.fc_res = self.cache.get("fc", fc_t, FMT["fc"])
        p_fc = Preset(8, 1, 8, 3, 2)
        self.fc = StreamDecodeMatmul(dev, stream("fc", p_fc, FMT["fc"], fc_t), p_fc, CK_BFP8)
        self.tap_norm = ResidualRMSNorm(dev, torch.ones(2048), TAP_EPS, n_memory_config=L1)
        self.hidden_norm_w = w("hidden_norm.weight")
        self.hidden_norm_rm = norm_weight(dev, self.hidden_norm_w)  # prefill commit (ttnn.rms_norm)
        # ctx_gain * ctx in one norm: rms_norm(y) * (hidden_norm * ctx_gain)
        self.gctx_norm = ResidualRMSNorm(dev, self.hidden_norm_w * w("ctx_gain"), 1e-5, n_memory_config=DRAM)
        # per layer: commit front (hidden-norm prologue), draft front (n1), o_proj, fused MLP (n2)
        p13, p2 = W13_PRESETS[FMT["mlp"]], Preset(8, 1, 8, 2, 1)
        self.front_c, self.front_d, self.wo, self.mlp = [], [], [], []
        self.front_cv = None  # verify-mode commit fronts on the target's row map (bind_verify)
        for j, att in enumerate(self.attn):
            qkvg = stream(f"L{j}.attn.qkvg", PRESETS["attn_qkv"], FMT["attn"], lambda att=att: att.wqkv)
            self.front_c.append(AttnNormQKVFront(att, self.rope, self.hidden_norm_w, 1e-5, post=None, weight=qkvg))
            self.front_d.append(AttnNormQKVFront(att, self.rope, w(f"layers.{j}.n1.weight"), 1e-5, post=None,
                                                 weight=qkvg))
            self.wo.append(StreamDecodeMatmul(dev, stream(f"L{j}.attn.o_proj", PRESETS["attn_out"], FMT["attn"],
                                                          lambda att=att: att.wo),
                                              PRESETS["attn_out"], CK_BFP8))
            w13 = stream(f"L{j}.mlp.w1", p13, FMT["mlp"], lambda j=j: snap(w(f"layers.{j}.gate.weight").T, FMT["mlp"]),
                         key3=f"L{j}.mlp.w3", build3=lambda j=j: snap(w(f"layers.{j}.up.weight").T, FMT["mlp"]))
            w2 = stream(f"L{j}.mlp.w2", p2, FMT["mlp"], lambda j=j: snap(w(f"layers.{j}.down.weight").T, FMT["mlp"]))
            self.mlp.append(FusedMLPDecode(dev, w(f"layers.{j}.n2.weight"), 1e-5, w13, p13, w2, p2, CK_BFP8))
        self.final_norm = ResidualRMSNorm(dev, w("norm.weight"), 1e-5, h_memory_config=L1, n_memory_config=L1)
        # head: the target checkpoint's lm_head rows of the draft vocabulary
        head_key = "lm_head.weight" if "lm_head.weight" in set(model.src.keys()) else "model.embed_tokens.weight"

        def head_t():
            return snap(model.src.get(head_key)[self.vocab].float().T.contiguous(), FMT["head"])

        self.head = StreamDecodeMatmul(dev, stream("head", PRESETS["lm_head"], FMT["head"], head_t),
                                       PRESETS["lm_head"], CK_BFP8)
        self.argmax = DrafterArgmax(dev)
        self.mask_rows = sd["mask"].to(torch.bfloat16)  # [8, 2048]
        self._SDPADecode, self._AttnNormQKVFront, self._map_word = SDPADecode, AttnNormQKVFront, map_word
        self.sdpa = {}
        self.kv = None
        self.inputs, self.outs, self.traces, self.commit_traces = {}, {}, {}, {}
        self._staged = {}

    # ------------------------------------------------------------------ allocation
    def allocate(self, num_blocks: int, page_blocks: int, widths):
        """K/V caches ([num_blocks + 1 + max_streams, 4, 64, 128] BFP8 per layer: the pool, the target's trash block,
        one overflow block per drafted stream), the tap buffers of every step width in `widths`, ctx32 and every draft
        trace's fixed-address inputs.  page_blocks: the decode page-table width.  Before any trace capture."""
        dev = self.device
        self.num_blocks, self.page_blocks = int(num_blocks), int(page_blocks)
        self.overflow = self.num_blocks + 1
        self.kv = []
        for att in self.attn:
            k, v = (ttnn.zeros([self.num_blocks + 1 + self.max_streams, 4, 64, 128], dtype=ttnn.bfloat8_b,
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM) for _ in range(2))
            att.set_kv_cache(k, v)
            self.kv.append((k, v))
        tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        rk = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM)
        self.tap_bufs = {W: [ttnn.from_torch(torch.zeros(1, 1, W, 2048), **tile) for _ in self.taps]
                         for W in sorted(set(int(x) for x in widths))}
        self.ctx32 = ttnn.from_torch(torch.zeros(1, 1, 32, 2048), **tile)
        mask32 = torch.zeros(1, 1, 32, 2048)
        mask32[0, 0, :BLOCK_ROWS] = self.mask_rows.float()
        self.mask32 = ttnn.from_torch(mask32, **tile)
        for n in range(1, self.max_streams + 1):
            R = BLOCK_ROWS * n
            self.inputs[n] = dict(
                tok=ttnn.from_torch(torch.zeros(1, R, dtype=torch.int32), dtype=ttnn.uint32, **rk),
                sel=ttnn.from_torch(torch.zeros(1, 1, R, 64), **tile),
                pos=ttnn.from_torch(torch.full((R,), -1, dtype=torch.int32), dtype=ttnn.int32, **rk),
                spos=ttnn.from_torch(torch.full((R,), -1, dtype=torch.int32), dtype=ttnn.int32, **rk),
                pt=ttnn.from_torch(torch.full((R, self.page_blocks), self.num_blocks, dtype=torch.int32),
                                   dtype=ttnn.int32, **rk),
            )
            words = [self._map_word(r % BLOCK_ROWS, BLOCK_ROWS, 0) for r in range(R)] + [0] * (16 - R)
            vm = torch.tensor(np.asarray(words, dtype=np.uint32).view(np.int32)).reshape(1, 16)
            self.inputs[n]["vmap"] = ttnn.from_torch(vm, dtype=ttnn.uint32, **rk)
            self.outs[n] = dict(top2=ttnn.from_torch(torch.zeros(1, 48, dtype=torch.int32), dtype=ttnn.uint32, **rk))
            if R not in self.sdpa:
                self.sdpa[R] = self._SDPADecode(dev, rows=R)
            self._staged[n] = None
        self.front_dv = {n: [self._AttnNormQKVFront(None, None, verify=True, share=self.front_d[j],
                                                    vmap=self.inputs[n]["vmap"]) for j in range(3)]
                         for n in self.inputs}
        return self.kv

    def bind_verify(self, vmap):
        """The verify-mode commit fronts on the target's verify row map (FastStep.vmap)."""
        self.front_cv = [self._AttnNormQKVFront(None, None, verify=True, share=self.front_c[j], vmap=vmap)
                         for j in range(3)]

    # ------------------------------------------------------------------ commit (after a target decode step)
    def commit(self, W: int, cur_pos, page_table, verify: bool = False):
        """The width-W step's taps (tap_bufs[W]) at positions cur_pos [W] int32 (-1 = inactive row) and pages
        page_table [W, blocks] int32 -> K/V rows + ctx32; verify: rows grouped by the target's row map (bind_verify).
        Device-only, trace-capturable."""
        if verify and self.front_cv is None:
            raise RuntimeError("Drafter.commit(verify=True) before bind_verify")
        ns = []
        for t in self.tap_bufs[W]:
            _, n = self.tap_norm(t)
            ns.append(n)
        x = ttnn.concat(ns, dim=-1, memory_config=L1)
        free(*ns)
        y = self.fc(x, output_memory_config=L1)
        ttnn.deallocate(x)
        for f in (self.front_cv if verify else self.front_c):
            _, q, g = f(y, None, cur_pos, page_table)
            free(q, g)
        _, gc = self.gctx_norm(y)
        ttnn.deallocate(y)
        if W < 32:
            pad = ttnn.pad(gc, [(0, 0), (0, 0), (0, 32 - W), (0, 0)], 0.0)
            ttnn.deallocate(gc)
            gc = pad
        ttnn.copy(gc, self.ctx32)
        ttnn.deallocate(gc)

    def capture_commit(self, key, W: int, cur_pos, page_table, verify: bool = False):
        """Compile then capture the commit trace of one target step kind `key` (its fixed position / page-table
        inputs)."""
        self.commit(W, cur_pos, page_table, verify)
        ttnn.synchronize_device(self.device)
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.commit(W, cur_pos, page_table, verify)
        ttnn.end_trace_capture(self.device, tid, cq_id=0)
        old = self.commit_traces.pop(key, None)
        if old is not None:
            ttnn.release_trace(self.device, old)
        self.commit_traces[key] = tid

    def launch_commit(self, key):
        ttnn.execute_trace(self.device, self.commit_traces[key], cq_id=0, blocking=False)

    # ------------------------------------------------------------------ prefill commit (eager)
    def prefill_commit(self, taps, start_tensor, chunk_page_table):
        """taps [1, 1, Tp, 2048] x 3 (prefill rows at start.., chunk_page_table as the target's chunk write); writes
        the chunk's K/V rows (AttnFrontPrefill: the target's prefill front)."""
        ns = [ttnn.rms_norm(t, epsilon=TAP_EPS, compute_kernel_config=CK_NORM, memory_config=DRAM) for t in taps]
        x = ttnn.concat(ns, dim=-1, memory_config=DRAM)
        free(*ns)
        y = linear(x, self.fc_res)
        ttnn.deallocate(x)
        ctx = ttnn.rms_norm(y, weight=self.hidden_norm_rm, epsilon=1e-5, compute_kernel_config=CK_NORM,
                            memory_config=DRAM)
        ttnn.deallocate(y)
        for att in self.attn:
            qkv = linear(ctx, att.wqkv)
            q = att._front(self.rope)(qkv, start_tensor, chunk_page_table)
            free(q, qkv)
        ttnn.deallocate(ctx)

    # ------------------------------------------------------------------ draft pass
    def _forward(self, n: int):
        """The traced draft pass of n streams (fixed inputs self.inputs[n]) -> logits [1, 1, 8n, 65536] BF16 DRAM;
        top-2 page into self.outs[n]['top2']."""
        I = self.inputs[n]
        R = BLOCK_ROWS * n
        e = ttnn.reshape(ttnn.embedding(I["tok"], self.m.embed, layout=ttnn.TILE_LAYOUT, memory_config=DRAM),
                         [1, 1, R, 2048])
        g = ttnn.concat([self.ctx32, self.mask32], dim=2, memory_config=DRAM)
        gm = ttnn.matmul(I["sel"], g, compute_kernel_config=CK_HIFI4, memory_config=DRAM)  # gain ctx[a-1] + mask[i]
        ttnn.deallocate(g)
        a, b = e, gm
        for j in range(3):
            h, q, gate = self.front_dv[n][j](a, b, I["pos"], I["pt"])
            free(a, b)
            att = self.attn[j]
            flat = self.sdpa[R](q, att.k_cache, att.v_cache, I["spos"], I["pt"], gate=gate, window=WINDOW)
            free(q, gate)
            o = self.wo[j](flat, output_memory_config=L1)
            ttnn.deallocate(flat)
            a, b = self.mlp[j](h, o)
            free(h, o)
        hf, nf = self.final_norm(a, b)
        free(a, b, hf)
        logits = self.head(nf)
        ttnn.deallocate(nf)
        self.argmax(logits, R, self.outs[n]["top2"])
        return logits

    def capture(self, log=print):
        """Compile then capture the draft trace of every stream count."""
        for n in self.inputs:
            old = self.outs[n].pop("logits", None)
            if old is not None:
                ttnn.deallocate(old)
            ttnn.deallocate(self._forward(n))
            ttnn.synchronize_device(self.device)
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            self.outs[n]["logits"] = self._forward(n)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self.traces[n] = tid
            log(f"LFM3 drafter: captured the draft trace ({n} stream(s), {BLOCK_ROWS * n} rows)")

    def stage(self, n: int, streams):
        """streams: per drafted stream (anchor position a, anchor token, its row in the last committed step (ctx32 row
        of position a - 1), page-table row int32 [>= page_blocks or fewer] of its request) -> the n-stream inputs."""
        if len(streams) != n:
            raise ValueError("stage: one entry per drafted stream")
        R = BLOCK_ROWS * n
        anchors = np.asarray([int(a) for a, _, _, _ in streams], dtype=np.int64)
        if int(anchors.max()) + DRAFTS >= self.page_blocks * 64:
            raise ValueError(f"draft rows beyond the page table at anchors {anchors.tolist()}")
        offs = np.arange(BLOCK_ROWS)
        tok = np.repeat(np.asarray([int(t) for _, t, _, _ in streams], dtype=np.int32), BLOCK_ROWS).reshape(1, R)
        pos = (anchors[:, None] + offs).reshape(R).astype(np.int32)
        spos = np.repeat(anchors + DRAFTS, BLOCK_ROWS).astype(np.int32)
        sel = np.zeros((1, 1, R, 64), dtype=np.float32)
        sel[0, 0, np.arange(R), np.repeat([int(r) for _, _, r, _ in streams], BLOCK_ROWS)] = 1.0
        sel[0, 0, np.arange(R), 32 + np.tile(offs, n)] = 1.0
        pt = np.full((R, self.page_blocks), self.num_blocks, dtype=np.int32)
        for s, (a, _, _, pt_row) in enumerate(streams):
            src = np.asarray(pt_row, dtype=np.int32).reshape(-1)[: self.page_blocks]
            rows = pt[s * BLOCK_ROWS:(s + 1) * BLOCK_ROWS]
            rows[:, : src.size] = src
            # pages holding no committed position (every position >= a) -> this stream's overflow block
            rows[:, -(-a // 64): (a + DRAFTS) // 64 + 1] = self.overflow + s
        I = self.inputs[n]
        rk = dict(layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(tok), dtype=ttnn.uint32, **rk), I["tok"])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(sel), dtype=ttnn.bfloat16,
                                                        layout=ttnn.TILE_LAYOUT), I["sel"])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(pos), dtype=ttnn.int32, **rk), I["pos"])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(spos), dtype=ttnn.int32, **rk), I["spos"])
        if self._staged[n] is None or not np.array_equal(self._staged[n], pt):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(pt), dtype=ttnn.int32, **rk), I["pt"])
            self._staged[n] = pt

    def launch(self, n: int):
        ttnn.execute_trace(self.device, self.traces[n], cq_id=0, blocking=False)

    def read(self, n: int, page=None):
        """-> (drafts [n][7] target token ids, margins [n][7]: top-1 minus top-2 logit) of the last n-stream pass
        (page: its top-2 page already copied to the host, speculate.EarlyRead; None: a blocking read)."""
        R = BLOCK_ROWS * n
        if page is None:
            page = ttnn.to_torch(self.outs[n]["top2"])
        page = page.reshape(-1)[: 3 * R].to(torch.int64).numpy() & 0xFFFFFFFF
        page = page.reshape(R, 3)
        ids = self._vocab_np[page[:, 0]].reshape(n, BLOCK_ROWS)[:, 1:]
        vals = key_values(page[:, 1:]).reshape(n, BLOCK_ROWS, 2)[:, 1:]
        return ids.tolist(), (vals[..., 0] - vals[..., 1]).tolist()
