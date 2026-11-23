# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini block drafter (`hf/mtp.safetensors`, DSpark "rnn_hidden" Markov head) on the P150, fused.

Semantics = the CPU reference `ref/drafter/drafter.py` (from vLLM `models/inkling2/nvidia/dspark.py` + the DSpark /
DFlash speculator flow; summary in its docstring and in tt/NOTES-spec.md).  The drafter only proposes tokens: any
deviation from the reference costs acceptance, never correctness (the target verifies every draft).

    dr = Drafter(device, cfg, cache, max_seq_len, source=src)
    dr.ctx_inp[R].write(positions)         # host: positions of the R context rows (rope rows + cache rows)
    dr.context(feats, R)                   # context K/V of R rows (feats: 4 BF16 [1, 1, 32, H] tensors, rows
                                           # 0..R-1 = target features of those positions); traceable
    dr.inp.write(x0, p, anchor, sel_row)   # host: block inputs of the anchor at position p
    dr.draft(feats, steps=m)               # block forward + m Markov steps -> step_ids (read_drafts()); traceable

Batched configuration (slots=S <= 16, kv_blocks=NB: the batched server's speculative decoding; the single-stream
configuration slots=None above is unchanged):
    dr = Drafter(device, cfg, cache, max_seq_len, source=src, slots=S, kv_blocks=NB)
    caches (BF16): full-context layer (1): ONE pool [NB + 1, 8, 64, 128] per K / V addressed by vLLM block ids (page
        row = the target's block-table row, W = max_seq_len / 64 columns; block NB = zero / scratch); local layers
        (0, 2): per-slot rings [S * 32, 8, 64, 128] (logical block j of slot s -> block s*32 + j % 32: 2048 positions).
    dr.slot_tensors()                                   # [K L0, V L0, K L2, V L2] slot-major views [S, 256, 64, 128]
    dr.context_fill(feats, start, slot=s, page_row=pt)  # prompt context, T <= 4096 rows (rings in 2048-row parts)
    dr.ctx_rows_inp[N].write(slots, positions, pages)   # host, N = 1..16 rows (pad rows: replicas of row 0)
    dr.context_rows(feats, N)                           # traceable: context K/V of feats rows 0..N-1
    dr.users_inp.write([(slot, p, anchor, x0, sel_row, page_row), ...])   # host, U <= 4 users
    dr.draft_users(feats, U, steps)                     # traceable: one block pass of 8U rows + `steps` Markov steps
    dr.read_drafts_users(U, steps) -> [[drafts of user u]]
Every user is computed exactly as the single-stream drafter computes its sequence (bitwise: row-independent stream
matmuls / norms / rope, one block SDPA call per user with the single-sequence core split, multi-row MarkovRNN /
MarkovArgmax programs with one fold stream per step for all users); cache rows are addressed by physical positions
(block * 64 + pos % 64) computed on the host per layer kind.
Precision (constructor args / DraftPrecision.from_env: INK_DRAFT_LAYERS, INK_DRAFT_BASE, INK_DRAFT_FOLD = bfp8 | bfp4):
the 3 block layers' + context projections, the base-logit unembedding copy, the Markov fold.

Per-cycle inputs (`DraftInputs`, persistent device tensors written from the host with `DraftInputs.write`):
    x0      BF16 [1, 1, 8, H]     block input: row 0 = normed target embedding of the anchor, rows 1..7 mask_embed
    pos8    int32 [8]             p .. p+7 (block K/V cache rows); cur8 [8] = p + 7 (SDPA: keys <= p+7)
    cos/sin BF16 [1, 1, 32, 128]  rope rows of the block (row u = position p + u)
    sel     BF16 [1, 1, 8, 32]    one-hot row selector: every row picks the feature row of position p-1 (f_rnn)
    ctx_inp[R]: positions int32 [R] + cos/sin [1, 1, 32, 128] of R context rows.

Device formulation (per block layer 13 launches, weights streamed from DRAM):
- residual BF16 (as the reference); add + RMSNorm in one launch (fast/residual_norm.py ResidualRMSNorm, 24 cores).
- [q | k | v] = StreamDecodeMatmul(x) (8 rows); RowMove (fast/draft_ops.py) q / k rows -> heads-major tensors Qh
  [1, 24, 32, 128] / Kh [1, 8, 32, 128] (row h*32 + u = head h of block row u) and v rows -> the paged V cache rows p+u
  (32 B face-row writes: no read-modify-write race between rows of one tile).
- q / k: rms_norm(weight q_norm / k_norm) + ttnn.experimental.rotary_embedding (neox rotate_half; cos / sin rows =
  the block positions, BF16); RowMove: q rows -> the SDPA query, k rows -> the paged K cache rows p+u.
- block attention: one paged SDPA decode with the 8 KV HEADS as the batch (MQA: user j = kv head j, 24 query rows =
  (block row u, group g), the cache viewed as [blocks * 8, 1, 64, 128] with page-table row j = pages b * 8 + j), so
  every K/V row is read once; CAUSAL with cur_pos = p + 7 (= the non-causal block; is_causal False would ignore cur_pos
  and attend to the whole page table), local layers sliding_window_size 1024 + 8; RowMove heads -> rows;
  o = StreamDecodeMatmul.
- MLP: gate|up StreamDecodeMatmul with SiLU(g) * u fused in FP32 DEST (glu preset), down StreamDecodeMatmul.
- base logits of the 8 rows = StreamDecodeMatmul(hid) over a stream-layout copy of the unembedding, BF16 (the
  reference rounds base to BF16).
- Markov head: ctx_i = [h_i | f_rnn | 1] [W_h; W_f; b] (one StreamDecodeMatmul, FP32 out), then per step two
  launches: MarkovRNN (w1[prev] gather, raw =
  ctx_i + [state, w1[prev]] W_sp^T in FP32 DEST, sigmoid / tanh gates, state update; 8 cores) and MarkovArgmax =
  argmax bf16(base_i + fold_b + latent fold) with the fold streamed once and the add / argmax fused (no FP32 [32, V]
  passes); the id row is the next step's prev.
- context K/V: mix (FP32 -> BF16) + RMSNorm -> kv_ctx StreamDecodeMatmul -> RowMove (k -> heads-major, v -> cache)
  -> k norm + rotary -> RowMove (k -> cache).  Prompt context (`context_fill`, untraced) keeps stock ops.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import ttnn

from .common import CK_HIFI2, CK_HIFI4, DRAM, DTYPES, free
from .config import HF_DIR
from .fast.decode_matmul import Preset, StreamDecodeMatmul, cached_stream_weight
from .fast.draft_ops import MarkovArgmax, MarkovRNN, RowMove
from .fast.residual_norm import ResidualRMSNorm

BF16, F32 = ttnn.bfloat16, ttnn.float32
BLOCK = 8
PAGE = 64
RING = 32           # batched local-layer ring: 32 blocks = 2048 positions per slot (window 1024 + 8, prefill parts)
MAX_SLOTS = 16      # batched configuration: sequence slots
MAX_USERS = 4       # users per draft_users pass (8 rows each: one 32-row stream pass)
MAX_CTX_ROWS = 16   # rows per context_rows call
SILU = [ttnn.UnaryOpType.SILU]

# stream presets (K = 2304 = 72 tiles unless noted)
P_QKV = Preset(5, 4, 8)              # 2304 x 5120: 40 workers x 4 columns
P_O = Preset(3, 3, 8)                # 3072 x 2304: 24 workers x 3 columns
P_GU = Preset(6, 4, 8, glu=True)     # 2304 x 2 * 9216 (SiLU(g) * u in DEST): 48 workers
P_DOWN = Preset(3, 3, 8)             # 9216 x 2304
P_KVCTX = Preset(4, 2, 8)            # 2304 x 2048
P_BASE = Preset(8, 4, 8, 3, 2)       # 2304 x 200064 (unembedding copy): 64 workers (417 GB/s BFP8, 401 BFP4)
P_FOLD = Preset(8, 2, 8)             # 256 x 200064 (Markov fold): 64 workers, w = 2 (base + fold_b in DEST)
P_FOLD_U = Preset(13, 2, 8)          # the fold for multi-user Markov steps (batched drafter): 104 workers
P_MCTX = Preset(3, 1, 5)             # [hid | f_rnn | 1] (4640 = 145 tiles) x 768, BF16 weight, FP32 out


def mtp_config(hf_dir) -> dict:
    return json.loads((Path(hf_dir) / "config.json").read_text())["mtp_config"]


@dataclass(frozen=True)
class DraftPrecision:
    layers: str = "bfp8"   # block layers + context projections
    base: str = "bfp4"     # unembedding copy of the base logits (BFP4: -0.5 ms per draft, acceptance unchanged on the
    #                        served prompts, NOTES-spec.md "Spec5")
    fold: str = "bfp8"     # Markov fold (unembed @ hidden_out)

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("INK_DRAFT_LAYERS", cls.layers), os.environ.get("INK_DRAFT_BASE", cls.base),
                   os.environ.get("INK_DRAFT_FOLD", cls.fold))

    @property
    def tag(self) -> str:
        return f"layers={self.layers} base={self.base} fold={self.fold}"


def rope_tables(pos: torch.Tensor, head_dim: int = 128, theta: float = 1e6):
    """fp32 cos / sin [N, head_dim] (neox halves duplicated) of positions pos [N]."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    f = pos.to(torch.float32)[:, None] * inv[None]
    return torch.cat([f.cos(), f.cos()], -1), torch.cat([f.sin(), f.sin()], -1)


class Drafter:
    def __init__(self, device, cfg, cache, max_seq_len: int, hf_dir: str = HF_DIR,
                 source=None, ctx_widths=(BLOCK, 32), precision: DraftPrecision | None = None,
                 slots: int | None = None, kv_blocks: int | None = None):
        from safetensors import safe_open

        self.device, self.cfg = device, cfg
        self.prec = precision or DraftPrecision.from_env()
        mc = mtp_config(hf_dir)
        self.mc = mc
        self.H, self.D = mc["hidden_size"], mc["head_dim"]
        self.nq, self.nkv = mc["num_attention_heads"], mc["num_key_value_heads"]
        self.L = mc["num_hidden_layers"]
        self.eps = mc["rms_norm_eps"]
        self.rank = mc["markov_rank"]
        self.window = mc["sliding_window_size"]
        self.local = set(mc["local_layer_ids"])
        self.theta = mc["rope_theta"]
        assert mc["block_size"] == BLOCK and self.nq % self.nkv == 0 and self.nkv == BLOCK
        assert max_seq_len % PAGE == 0
        self.max_seq_len = max_seq_len
        self.V, self.Vp = cfg.unpadded_vocab, cfg.padded_unembed
        f = safe_open(str(Path(hf_dir) / "mtp.safetensors"), "pt")
        g = f.get_tensor
        d = device
        H, D, nq, nkv, r = self.H, self.D, self.nq, self.nkv, self.rank
        wdt = DTYPES[self.prec.layers]

        def SW(key, preset, build, dtype=wdt, build3=None):
            return cached_stream_weight(cache, f"mtp.{key}", preset, build, dtype, build3)

        def vec(t, rows=1):
            return ttnn.from_torch(t.float().reshape(1, 1, rows, -1), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=d,
                                   memory_config=DRAM)

        def rn(key):
            return ResidualRMSNorm(d, g(key).float(), self.eps)

        self.alpha = torch.softmax(g("fuse_weight").float(), -1)  # [4 rows, 4 taps]
        self.layers = []
        for i in range(self.L):
            p = f"layers.{i}."
            w_qkv = SW(f"L{i}.qkv", P_QKV, lambda p=p: torch.cat([g(p + f"self_attn.{n}_proj.weight") for n in "qkv"]).T)
            w_o = SW(f"L{i}.o", P_O, lambda p=p: g(p + "self_attn.o_proj.weight").T)
            w_gu = SW(f"L{i}.gu", P_GU, lambda p=p: g(p + "mlp.gate_proj.weight").T,
                      build3=lambda p=p: g(p + "mlp.up_proj.weight").T)
            w_down = SW(f"L{i}.down", P_DOWN, lambda p=p: g(p + "mlp.down_proj.weight").T)
            w_kv = SW(f"L{i}.kv_ctx", P_KVCTX,
                      lambda p=p: torch.cat([g(p + f"self_attn.{n}_proj_ctx.weight") for n in "kv"]).T)
            lw = {
                "qkv": StreamDecodeMatmul(d, w_qkv, P_QKV, CK_HIFI2),
                "o": StreamDecodeMatmul(d, w_o, P_O, CK_HIFI2),
                "gu": StreamDecodeMatmul(d, w_gu, P_GU, CK_HIFI2),
                "down": StreamDecodeMatmul(d, w_down, P_DOWN, CK_HIFI2),
                "kv_ctx": StreamDecodeMatmul(d, w_kv, P_KVCTX, CK_HIFI2),
                # stock-layout copy for the prompt context fill (any row count)
                "kv_ctx_stock": cache.get(f"mtp.L{i}.kv_ctx", lambda p=p: torch.cat(
                    [g(p + f"self_attn.{n}_proj_ctx.weight") for n in "kv"]).T, ttnn.bfloat8_b),
                "rn_in": rn(p + "input_layernorm.weight"),
                "rn_post": rn(p + "post_attention_layernorm.weight"),
                "rn_mix": rn(p + "dflare_norm.weight"),
                "dflare": vec(g(p + "dflare_norm.weight")),
                "q_norm": vec(g(p + "self_attn.q_norm.weight")),
                "k_norm": vec(g(p + "self_attn.k_norm.weight")),
                "k_norm_ctx": vec(g(p + "self_attn.k_norm_ctx.weight")),
            }
            self.layers.append(lw)
        self.rn_final = rn("norm.weight")
        self.rn_m = rn("markov_head.dflare_norm.weight")
        rp = g("markov_head.rnn_proj.weight").float()  # [768, 5120]: [state | prev | hidden | f_rnn]
        Wc = lambda key, build, dtype=BF16: cache.get(f"mtp.{key}", build, dtype)  # noqa: E731

        def w_mctx():
            """[W_hid; W_frnn; bias; 0] = [4640, 768]: x = [h_i | f_rnn | 1, 0 x 31]."""
            w = torch.zeros(2 * H + 32, 3 * r)
            w[: 2 * H] = rp[:, 2 * r:].T
            w[2 * H] = g("markov_head.rnn_proj.bias").float()
            return w

        self.mm_ctx = StreamDecodeMatmul(d, cached_stream_weight(cache, "mtp.rnn.ctx", P_MCTX, w_mctx, BF16), P_MCTX,
                                         CK_HIFI4, out_dtype=F32)
        one = torch.zeros(1, 1, BLOCK, 32)
        one[..., 0] = 1
        self.ones = ttnn.from_torch(one, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=DRAM)
        # Markov embedding table (row-major for ttnn.embedding): [vocab, 256] BF16 (103 MB)
        self.w1 = ttnn.from_torch(g("markov_head.markov_w1.weight"), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=d, memory_config=DRAM)
        unembed = source.tensor("model.llm.unembed.weight") if source is not None else None
        # target embed_norm weight (host fp32): the block's anchor row = embed_norm(embed[anchor])
        self.embed_norm_w = source.tensor("model.llm.embed_norm.weight").float() if source is not None else None
        self._unembed = unembed
        self._cache = cache
        # base logits: stream copy of the unembedding [H, Vp]
        w_base = cached_stream_weight(cache, "mtp.base", P_BASE, self._unembed_t, DTYPES[self.prec.base])
        self.mm_base = StreamDecodeMatmul(d, w_base, P_BASE, CK_HIFI2, n_out=self.Vp)
        # Markov fold [256, Vp] (stream layout) + fold_b row 0 of [1, 1, 32, Vp] BF16
        w_fold = cached_stream_weight(cache, "mtp.fold", P_FOLD, lambda: self._fold_w(g), DTYPES[self.prec.fold])
        self.fold_b = cache.get("mtp.fold_b32", lambda: self._fold_b(g), BF16)
        self.markov = [MarkovArgmax(d, w_fold, P_FOLD, self.V, row, CK_HIFI2) for row in range(BLOCK)]
        w_sp = Wc("rnn.sp", lambda: rp[:, :2 * r].T)  # [state | prev] rows of rnn_proj^T: [512, 768]
        self.rnn = [MarkovRNN(d, w_sp, row) for row in range(BLOCK)]
        self.w_fold, self.w_sp = w_fold, w_sp
        self.mask_embed = g("mask_embed")
        nb = max_seq_len // PAGE
        self.nb = nb
        z = lambda shape, dt=BF16: ttnn.zeros(shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=d,  # noqa: E731
                                              memory_config=DRAM)
        self.slots = slots
        if slots is None:
            # caches: per layer paged K / V [S/64, 8, 64, 128] BF16, identity page tables
            self.k_cache = [z([nb, nkv, PAGE, D]) for _ in range(self.L)]
            self.v_cache = [z([nb, nkv, PAGE, D]) for _ in range(self.L)]
            # MQA views for the block SDPA: user j = kv head j, pages b * nkv + j
            self.k_mqa = [ttnn.reshape(c, [nb * nkv, 1, PAGE, D]) for c in self.k_cache]
            self.v_mqa = [ttnn.reshape(c, [nb * nkv, 1, PAGE, D]) for c in self.v_cache]
            pt = (torch.arange(nb, dtype=torch.int32).reshape(1, -1) * nkv
                  + torch.arange(nkv, dtype=torch.int32)[:, None])
            self.pt_mqa = ttnn.from_torch(pt.contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=d,
                                          memory_config=DRAM)
            # persistent block intermediates (RowMove destinations)
            gq = nq // nkv
            self.qh = z([1, nq, 32, D])             # row h*32 + u: q head h of block row u
            self.kh = z([1, nkv, 32, D])            # row j*32 + u: k head j of block row u
            self.q = z([1, nkv, BLOCK * gq, D])     # SDPA query: user j, row u*gq + g = q head j*gq + g of block row u
            self.o_rows = z([1, 1, BLOCK, nq * D])  # attention output rows
            self.mv_qkv = [RowMove(d, self._moves_qkv()) for _ in range(self.L)]
            self.mv_qk = RowMove(d, [(0, h * 32 + u, 0, 2, (h // gq) * 32 + u * gq + h % gq, 0, 4)
                                     for u in range(BLOCK) for h in range(nq)]
                                 + [(1, j * 32 + u, 0, 3, ("cache", u, j, nkv), 0, 4)
                                    for u in range(BLOCK) for j in range(nkv)])
            self.mv_rows = RowMove(d, [(0, j * 32 + u * gq + gg, 0, 1, u, (j * gq + gg) * 4, 4)
                                       for u in range(BLOCK) for j in range(nkv) for gg in range(gq)])
            self.ctx_widths = sorted({BLOCK, *ctx_widths})
            self.ctx_k = z([1, nkv, 32, D])          # row j*32 + x: k head j of context row x
            self.mv_ctx_kv = {u: RowMove(d, self._moves_ctx_kv(u)) for u in self.ctx_widths}
            self.mv_ctx_k = {u: RowMove(d, self._moves_ctx_k(u)) for u in self.ctx_widths}
            # ---- persistent per-cycle inputs (host-written) ----
            self.inp = DraftInputs(d, self)
            self.ctx_inp = {u: CtxInputs(d, self, u) for u in self.ctx_widths}
        else:
            self._init_batched(slots, kv_blocks, z, g)
        # the host unembedding is needed until every fold copy is built (the batched one above: a fresh tensor cache
        # has no other copy to rebuild it from)
        self._unembed = None
        # draft token i (1..7) -> word 0 of step_ids[i - 1] (MarkovArgmax outputs, the next step's prev) and word 0 of
        # row i - 1 of the draft log (read_drafts(): one read; the spec verify embeds its rows on the device)
        self.step_ids = [ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=d, memory_config=DRAM)
                         for _ in range(BLOCK - 1)]
        self.log = ttnn.from_torch(torch.zeros(BLOCK - 1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=d, memory_config=DRAM)
        zr = lambda: ttnn.zeros([1, 1, 32, r], dtype=BF16, layout=ttnn.TILE_LAYOUT, device=d,  # noqa: E731
                                memory_config=DRAM)
        self.zero_state, self.latent, self.state = zr(), zr(), [zr(), zr()]  # Markov state ping-pong
        self.grid = d.compute_with_storage_grid_size()
        self.debug = bool(int(os.environ.get("INK_SPEC_DEBUG", "0")))  # sync + print per stage
        self.dump = None  # dict: per-stage host copies (tests/dbg_spec_drafter.py; untraced only)
        self.sdpa_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(self.grid.x, self.grid.y),
                                               q_chunk_size=32, k_chunk_size=PAGE, exp_approx_mode=False)

    # ------------------------------------------------------------------ host-side derived weights
    def _unembed_t(self):
        """[H, Vp] W^T of the unembedding (rows >= V zero)."""
        u = self._unembed if self._unembed is not None else _unembed_from_cache(self._cache)
        out = torch.zeros(self.H, self.Vp, dtype=torch.bfloat16)
        out[:, : self.V] = u[: self.V].to(torch.bfloat16).T
        return out

    def _fold_w(self, g):
        """[256, Vp] fold W^T = (unembed @ hidden_out)^T (fp64, BF16-rounded as the reference stores it)."""
        P = g("markov_head.hidden_out.weight").double()  # [H, r]
        u = self._unembed if self._unembed is not None else _unembed_from_cache(self._cache)
        out = torch.zeros(self.rank, self.Vp)
        for s in range(0, self.V, 8192):
            e = min(s + 8192, self.V)
            out[:, s:e] = (u[s:e].double() @ P).to(torch.bfloat16).float().T
        return out

    def _fold_b(self, g):
        """[1, 1, 32, Vp] with row 0 = unembed @ hidden_out.bias (BF16-rounded)."""
        b = g("markov_head.hidden_out.bias").double()
        u = self._unembed if self._unembed is not None else _unembed_from_cache(self._cache)
        out = torch.zeros(1, 1, 32, self.Vp)
        out[0, 0, 0, : self.V] = (u[: self.V].double() @ b).to(torch.bfloat16).float()
        return out

    def _moves_qkv(self, rows: int = BLOCK):
        """qkv rows (tensor 0) -> q heads-major (1), k heads-major (2), v rows -> the v cache (3)."""
        nq, nkv = self.nq, self.nkv
        mv = []
        for u in range(rows):
            mv += [(0, u, h * 4, 1, h * 32 + u, 0, 4) for h in range(nq)]
            mv += [(0, u, (nq + j) * 4, 2, j * 32 + u, 0, 4) for j in range(nkv)]
            mv += [(0, u, (nq + nkv + j) * 4, 3, ("cache", u, j, nkv), 0, 4) for j in range(nkv)]
        return mv

    def _moves_ctx_kv(self, rows: int):
        """Context kv rows (0) -> k heads-major ctx_k (1), v rows -> the v cache (2) at pos[x]."""
        nkv = self.nkv
        return ([(0, x, j * 4, 1, j * 32 + x, 0, 4) for x in range(rows) for j in range(nkv)]
                + [(0, x, (nkv + j) * 4, 2, ("cache", x, j, nkv), 0, 4) for x in range(rows) for j in range(nkv)])

    def _moves_ctx_k(self, rows: int):
        """Normed + rotated context k heads-major (0) -> the k cache (1) at pos[x]."""
        nkv = self.nkv
        return [(0, j * 32 + x, 0, 1, ("cache", x, j, nkv), 0, 4) for x in range(rows) for j in range(nkv)]

    # ------------------------------------------------------------------ batched configuration
    def _init_batched(self, slots: int, kv_blocks: int, z, g):
        """Caches + persistent inputs of the batched drafter (slots S <= 16, kv_blocks NB = vLLM pool blocks).
        Full-context layers: ONE pool [NB + 1, 8, 64, 128] per K / V addressed by vLLM block ids (block NB: zero /
        scratch).  Local layers: per-slot rings [S * 32, 8, 64, 128]: logical block j of slot s -> block s*32 + j%32."""
        if not 1 <= slots <= MAX_SLOTS or kv_blocks is None or kv_blocks < 1:
            raise ValueError(f"Drafter: slots {slots} (1..{MAX_SLOTS}) and kv_blocks {kv_blocks} >= 1 required")
        d, D, nq, nkv = self.device, self.D, self.nq, self.nkv
        self.S, self.NB, self.W = slots, kv_blocks, self.nb
        self.kinds = ["ring" if i in self.local else "pool" for i in range(self.L)]
        shape = {"ring": [slots * RING, nkv, PAGE, D], "pool": [kv_blocks + 1, nkv, PAGE, D]}
        self.k_cache = [z(shape[k]) for k in self.kinds]
        self.v_cache = [z(shape[k]) for k in self.kinds]
        self.k_mqa = [ttnn.reshape(c, [shape[k][0] * nkv, 1, PAGE, D]) for c, k in zip(self.k_cache, self.kinds)]
        self.v_mqa = [ttnn.reshape(c, [shape[k][0] * nkv, 1, PAGE, D]) for c, k in zip(self.v_cache, self.kinds)]
        gq = nq // nkv
        self.qh = z([1, nq, 32, D])        # row h*32 + x: q head h of block row x = r*U + u (user u, block row r)
        self.kh = z([1, nkv, 32, D])
        self.qu = [z([1, nkv, BLOCK * gq, D]) for _ in range(MAX_USERS)]  # per-user SDPA query (as self.q)
        self.o_rows = z([1, 1, 32, nq * D])
        self.ctx_k = z([1, nkv, 32, D])
        one = torch.zeros(1, 1, BLOCK * MAX_USERS, 32)
        one[..., 0] = 1
        self.ones_u = {u: ttnn.from_torch(one[:, :, : BLOCK * u].contiguous(), dtype=BF16, layout=ttnn.TILE_LAYOUT,
                                          device=d, memory_config=DRAM) for u in range(2, MAX_USERS + 1)}
        self.ones_u[1] = self.ones
        self._ops = {}  # (kind, ...) -> RowMove / MarkovArgmax / MarkovRNN, built on first use (host only)
        # MarkovArgmax with users > 1 reads fold_b row brow * U + u in place: the bias replicated in all 32 rows
        self.fold_b_rep = self._cache.get("mtp.fold_b_rep32", lambda: ttnn.to_torch(self.fold_b).float()[..., :1, :]
                                          .expand(1, 1, 32, -1).contiguous(), BF16)
        self.users_inp = UserInputs(d, self)
        self.ctx_rows_inp = {n: CtxRowInputs(d, self, n) for n in range(1, MAX_CTX_ROWS + 1)}
        # Markov fold for users > 1 in a 104-worker stream layout (+54 MB BFP8): the per-user argmax scans, not the
        # stream, bound the multi-user MarkovArgmax; same per-column values and worker order = column order
        self.w_fold_u = cached_stream_weight(self._cache, "mtp.fold", P_FOLD_U, lambda: self._fold_w(g),
                                             DTYPES[self.prec.fold])

    def _op(self, key, build):
        op = self._ops.get(key)
        if op is None:
            op = self._ops[key] = build()
        return op

    def _rowmove(self, key, moves):
        def build():
            mv = moves()
            # many-row moves (users > 1) spread over every core (no effect on values)
            return RowMove(self.device, mv, workers=64 if len(mv) <= BLOCK * 40 else 2 * self.grid.x * self.grid.y)
        return self._op(("rm",) + key, build)

    def _markov_u(self, users: int, brow: int):
        if users == 1:
            return self.markov[brow]
        # users > 1: the fold in the 104-worker layout (P_FOLD_U: the per-user argmax scans are the bottleneck)
        return self._op(("markov", users, brow), lambda: MarkovArgmax(self.device, self.w_fold_u, P_FOLD_U, self.V,
                                                                      brow, CK_HIFI2, users=users))

    def _rnn_u(self, users: int, crow: int):
        if users == 1:
            return self.rnn[crow]
        return self._op(("rnn", users, crow), lambda: MarkovRNN(self.device, self.w_sp, crow, users=users))

    def page_blocks(self, page_row) -> torch.Tensor:
        """Host page row (vLLM block ids [<= W] or longer) -> int64 [W] pool blocks (missing / out-of-range: NB)."""
        row = torch.as_tensor(page_row, dtype=torch.int64).reshape(-1)[: self.W]
        if row.numel() < self.W:
            row = torch.cat([row, torch.full((self.W - row.numel(),), self.NB, dtype=torch.int64)])
        return torch.where((row >= 0) & (row <= self.NB), row, torch.full_like(row, self.NB))

    def phys(self, kind: str, slot: int, q: int, page_row) -> int:
        """Physical cache row (block * 64 + q % 64) of position q of a sequence in `slot` with page row `page_row`
        (host vLLM block ids; a missing / out-of-range block -> NB)."""
        lb = q // PAGE
        if kind == "ring":
            blk = slot * RING + lb % RING
        else:
            blk = int(page_row[lb]) if lb < len(page_row) else self.NB
            blk = blk if 0 <= blk <= self.NB else self.NB
        return blk * PAGE + q % PAGE

    def slot_tensors(self):
        """Slot-major views [S, 32 * 8, 64, 128] of the local layers' rings: [K L0, V L0, K L2, V L2] (local layers in
        order, K then V); slot s of each is that slot's whole drafter ring state (prefix-cache snapshots)."""
        out = []
        for i, kind in enumerate(self.kinds):
            if kind == "ring":
                out += [ttnn.reshape(c, [self.S, RING * self.nkv, PAGE, self.D]) for c in (self.k_cache[i],
                                                                                            self.v_cache[i])]
        return out

    def context_rows(self, feats, rows: int):
        """Batched context(): drafter context K/V of rows 0..rows-1 of feats (4 BF16 [1, 1, 32, H]) at each row's
        (slot, position, page row) from ctx_rows_inp[rows].write.  rows <= 16.  Traceable."""
        ci = self.ctx_rows_inp[rows]
        for i, lw in enumerate(self.layers):
            kind = self.kinds[i]
            mix = self._mix(feats, i, lw["rn_mix"])
            kv = lw["kv_ctx"](mix, rows=rows, x_first=rows > BLOCK)
            ttnn.deallocate(mix)
            self._rowmove(("ctx_kv", rows, i), lambda: self._moves_ctx_kv(rows))(
                [kv, self.ctx_k, self.v_cache[i]], ci.pos[kind])
            ttnn.deallocate(kv)
            kr = self._norm_rope(self.ctx_k, lw["k_norm_ctx"], ci.cos, ci.sin)
            self._rowmove(("ctx_k", rows, i), lambda: self._moves_ctx_k(rows))([kr, self.k_cache[i]], ci.pos[kind])
            ttnn.deallocate(kr)
            self._dbg(f"ctx rows L{i} rows={rows}")

    def block_users(self, users: int):
        """Batched block(): block forward of `users` users (rows interleaved: row r*U + u = block row r of user u; the
        multi-user Markov programs read a step's U rows as one contiguous run) on users_inp.x0[users] ->
        final-norm hidden BF16 [1, 1, 8 * users, H] (writes each user's block K/V at its p..p+7).  Stream matmuls of
        more than 8 rows gather their input first (x_first: same values)."""
        ui, U = self.users_inp, users
        rows, nq, nkv = BLOCK * U, self.nq, self.nkv
        gq = nq // nkv
        xf = rows > BLOCK
        h, m = ui.x0[U], None
        for i, lw in enumerate(self.layers):
            kind = self.kinds[i]
            if m is None:
                _, xb = lw["rn_in"](h)
            else:
                hn, xb = lw["rn_in"](h, m)
                free(m)
                if h is not ui.x0[U]:
                    ttnn.deallocate(h)
                h = hn
            self._d(f"L{i}.x", xb)
            qkv = lw["qkv"](xb, rows=rows, x_first=xf)
            ttnn.deallocate(xb)
            self._rowmove(("qkv", U, i), lambda: self._moves_qkv(rows))(
                [qkv, self.qh, self.kh, self.v_cache[i]], ui.pos[kind])
            ttnn.deallocate(qkv)
            qr = self._norm_rope(self.qh, lw["q_norm"], ui.cos, ui.sin)
            kr = self._norm_rope(self.kh, lw["k_norm"], ui.cos, ui.sin)
            self._d(f"L{i}.qr", qr)
            self._rowmove(("qk", U, i), lambda: (
                [(0, h_ * 32 + r * U + u, 0, 2 + u, (h_ // gq) * 32 + r * gq + h_ % gq, 0, 4)
                 for u in range(U) for r in range(BLOCK) for h_ in range(nq)]
                + [(1, j * 32 + x, 0, 2 + U, ("cache", x, j, nkv), 0, 4) for x in range(rows) for j in range(nkv)]))(
                [qr, kr, *self.qu[:U], self.k_cache[i]], ui.pos[kind])
            free(qr, kr)
            self._dbg(f"L{i} q / k / v users={U}")
            # one SDPA per user (8 kv heads as the batch, exactly the single-sequence call: same core split)
            kw = {"sliding_window_size": self.window + BLOCK} if i in self.local else {}
            atts = [ttnn.transformer.paged_scaled_dot_product_attention_decode(
                self.qu[u], self.k_mqa[i], self.v_mqa[i], ui.pt[kind][u], is_causal=True, cur_pos_tensor=ui.cur[u],
                scale=self.D ** -0.5, program_config=self.sdpa_cfg, compute_kernel_config=CK_HIFI4,
                memory_config=DRAM, **kw) for u in range(U)]
            self._rowmove(("rows", U, i), lambda: [
                (u, j * 32 + r * gq + gg, 0, U, r * U + u, (j * gq + gg) * 4, 4)
                for u in range(U) for r in range(BLOCK) for j in range(nkv) for gg in range(gq)])(
                [*atts, self.o_rows])
            free(*atts)
            self._d(f"L{i}.att", self.o_rows)
            self._dbg(f"L{i} sdpa users={U}")
            a = lw["o"](self.o_rows, rows=rows, x_first=xf)
            hn, xb = lw["rn_post"](h, a)
            free(a)
            if h is not ui.x0[U]:
                ttnn.deallocate(h)
            h = hn
            act = lw["gu"](xb, rows=rows, x_first=xf)
            ttnn.deallocate(xb)
            m = lw["down"](act, rows=rows, x_first=xf)
            ttnn.deallocate(act)
            self._d(f"L{i}.h", m)
            self._dbg(f"L{i} mlp users={U}")
        hn, out = self.rn_final(h, m)
        free(m, hn)
        if h is not ui.x0[U]:
            ttnn.deallocate(h)
        self._d("hid", out)
        return out

    def draft_users(self, feats, users: int, steps: int = BLOCK - 1, hidden=None):
        """Batched draft(): block of `users` users (users_inp.write) in one pass, base logits, f_rnn of user u =
        feature row sel_row[u] of feats, `steps` Markov steps for every user (draft i of user u -> word u of
        step_ids[i - 1] and of draft-log row i - 1; read_drafts_users()).  Returns the block hidden [1, 1, 8 * users,
        H] (row r * users + u = block row r of user u) when `hidden` (tests), else frees it.  Traceable."""
        ui, U = self.users_inp, users
        rows = BLOCK * U
        xf = rows > BLOCK
        hid = self.block_users(U)
        base = self.mm_base(hid, rows=rows, x_first=xf)  # BF16 [1, 1, 8U, Vp]
        self._dbg("base logits")
        fm = self._mix(feats, self.L, self.rn_m)
        fr = ttnn.matmul(ui.sel[U], fm, dtype=BF16, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        ttnn.deallocate(fm)
        self._d("f_rnn", fr)
        x = ttnn.concat([hid, fr, self.ones_u[U]], dim=-1, memory_config=DRAM)
        ttnn.deallocate(fr)
        ctx = self.mm_ctx(x, rows=rows, x_first=xf)  # FP32 [1, 1, 8U, 768]
        ttnn.deallocate(x)
        self._d("mctx", ctx)
        fold_b = self.fold_b if U == 1 else self.fold_b_rep
        prev_ids = ui.anchor_id
        for i in range(steps):
            s_in = self.zero_state if i == 0 else self.state[(i - 1) % 2]
            self._rnn_u(U, i + 1)(prev_ids, self.w1, s_in, ctx, self.latent, self.state[i % 2])
            self._markov_u(U, i + 1)(self.latent, base, fold_b, self.step_ids[i], self.log)
            self._dbg(f"markov step {i} users={U}")
            prev_ids = self.step_ids[i]
        free(ctx, base)
        if hidden:
            return hid
        ttnn.deallocate(hid)
        return None

    def read_drafts_users(self, users: int, steps: int = BLOCK - 1) -> list[list[int]]:
        """The last draft_users' first `steps` drafts per user (one device read of the draft log)."""
        if steps <= 0:
            return [[] for _ in range(users)]
        log = ttnn.to_torch(self.log).reshape(BLOCK - 1, 32)
        return [[int(x) for x in log[:steps, u]] for u in range(users)]

    def _dbg(self, stage: str):
        if self.debug:
            ttnn.synchronize_device(self.device)
            print(f"[drafter] {stage} ok", flush=True)

    def _d(self, name: str, t):
        """Stage dump for the per-stage bisect (tests/dbg_spec_drafter.py): never set during trace capture."""
        if self.dump is not None:
            self.dump[name] = ttnn.to_torch(t).float()

    def reset(self):
        """New sequence: nothing to clear (every cache row is written before it is read)."""

    # ------------------------------------------------------------------ helpers
    def _mix_sum(self, feats, row: int):
        """sum_t alpha[row, t] feats[t] (FP32) -> FP32 [1, 1, T, H]."""
        acc = None
        for t, x in enumerate(feats):
            y = ttnn.mul(x, float(self.alpha[row, t]), dtype=F32, memory_config=DRAM)
            if acc is None:
                acc = y
            else:
                ttnn.add(acc, y, output_tensor=acc)
                ttnn.deallocate(y)
        return acc

    def _mix(self, feats, row: int, rn):
        """RMSNorm(sum_t alpha[row, t] feats[t]) -> BF16 [1, 1, 32, H] (all 32 feature rows; the FP32 mix is rounded
        to BF16 before the one-launch norm)."""
        acc = self._mix_sum(feats, row)
        mb = ttnn.typecast(acc, BF16, memory_config=DRAM)
        ttnn.deallocate(acc)
        _, out = rn(mb)
        ttnn.deallocate(mb)
        return out

    def _norm_rope(self, x, w, cos, sin):
        """Per-row RMSNorm * w, then RoPE (rotate_half): x BF16 heads-major [1, heads, 32, 128] (row = position row
        of cos / sin) -> BF16."""
        n = ttnn.rms_norm(x, epsilon=self.eps, weight=w, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        out = ttnn.experimental.rotary_embedding(n, cos, sin, memory_config=DRAM)
        ttnn.deallocate(n)
        return out

    # ------------------------------------------------------------------ context K/V
    def context(self, feats, users: int):
        """Context K/V of `users` rows: feats = 4 BF16 [1, 1, 32, H] tensors (rows 0..users-1 = the target features
        of positions ctx_inp[users].pos); uses the CtxInputs of `users` rows.  Traceable."""
        ci = self.ctx_inp[users]
        for i, lw in enumerate(self.layers):
            mix = self._mix(feats, i, lw["rn_mix"])
            self._d(f"ctx{i}.mix", mix)
            kv = lw["kv_ctx"](mix, rows=users)
            ttnn.deallocate(mix)
            kt = self.ctx_k
            self.mv_ctx_kv[users]([kv, kt, self.v_cache[i]], ci.pos)
            ttnn.deallocate(kv)
            kr = self._norm_rope(kt, lw["k_norm_ctx"], ci.cos, ci.sin)
            self._d(f"ctx{i}.k", kr)
            self.mv_ctx_k[users]([kr, self.k_cache[i]], ci.pos)
            ttnn.deallocate(kr)
            self._dbg(f"ctx L{i} users={users}")

    def context_fill(self, feats, start: int, slot: int | None = None, page_row=None):
        """Prefill form of context(): the context K/V of the T rows of feats (4 BF16 [1, 1, T, H] tensors, T a
        multiple of 32) at positions start .. start+T-1 (start a multiple of 64), written by one paged_fill_cache per
        layer and K / V (whole cache blocks; rows past the prompt hold junk that the first block / verify context
        rewrites before any read).  Batched configuration: slot and page_row (host vLLM block ids [W]) are required;
        the rings are written in <= 2048-row parts (T <= 4096).  Untraced (allocates per-call position tensors)."""
        T = feats[0].shape[-2]
        if start % PAGE or T % 32 or start + T > self.max_seq_len:
            raise ValueError(f"context_fill rows {start}..{start + T - 1}: start a multiple of {PAGE}, T of 32")
        d, D, nkv = self.device, self.D, self.nkv
        cos, sin = rope_tables(torch.arange(start, start + T), D, self.theta)
        dev = lambda t: ttnn.from_torch(t.reshape(1, 1, T, D), dtype=BF16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                        device=d, memory_config=DRAM)
        c, s = dev(cos), dev(sin)
        i32 = lambda v: ttnn.from_torch(torch.tensor(v, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32,  # noqa
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=d, memory_config=DRAM)
        if self.slots is None:
            if slot is not None or page_row is not None:
                raise ValueError("context_fill: slot / page_row need the batched configuration")
            nb = -(-T // PAGE)
            pt = ttnn.from_torch(torch.arange(start // PAGE, start // PAGE + nb, dtype=torch.int32).reshape(1, -1),
                                 dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=d, memory_config=DRAM)
            kinds, parts = [None] * self.L, {None: [(0, T, pt)]}
        else:
            if slot is None or page_row is None or not 0 <= slot < self.S or T > 2 * RING * PAGE:
                raise ValueError(f"context_fill: slot (< {self.S}) and page_row required, T <= {2 * RING * PAGE}")
            blocks = self.page_blocks(page_row)
            kinds, parts = self.kinds, {}
            for kind in sorted(set(kinds)):
                step = RING * PAGE if kind == "ring" else T  # a ring part never aliases inside itself
                parts[kind] = []
                for a in range(0, T, step):
                    b = min(T, a + step)
                    lbs = range((start + a) // PAGE, (start + b - 1) // PAGE + 1)
                    blks = [slot * RING + lb % RING if kind == "ring" else int(blocks[lb]) for lb in lbs]
                    parts[kind].append((a, b, i32(blks)))
        for i, lw in enumerate(self.layers):
            acc = self._mix_sum(feats, i)
            mf = ttnn.rms_norm(acc, epsilon=self.eps, weight=lw["dflare"], compute_kernel_config=CK_HIFI4,
                               memory_config=DRAM)
            mix = ttnn.typecast(mf, BF16, memory_config=DRAM)
            free(acc, mf)
            kv = ttnn.linear(mix, lw["kv_ctx_stock"], dtype=BF16, compute_kernel_config=CK_HIFI2, memory_config=DRAM)
            ttnn.deallocate(mix)
            kvw = nkv * D
            k = _users(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, T, kvw]), T, nkv, D)
            v = _users(ttnn.slice(kv, [0, 0, 0, kvw], [1, 1, T, 2 * kvw]), T, nkv, D)
            ttnn.deallocate(kv)
            kn = ttnn.rms_norm(k, epsilon=self.eps, weight=lw["k_norm_ctx"], compute_kernel_config=CK_HIFI4,
                               memory_config=DRAM)
            ttnn.deallocate(k)
            kh = ttnn.permute(kn, (0, 2, 1, 3), memory_config=DRAM)  # [1, 8, T, 128]
            kr = ttnn.experimental.rotary_embedding(kh, c, s, memory_config=DRAM)
            vh = ttnn.permute(v, (0, 2, 1, 3), memory_config=DRAM)
            free(kn, kh, v)
            for cache, x in ((self.k_cache[i], kr), (self.v_cache[i], vh)):
                for a, b, pt in parts[kinds[i]]:
                    xp = x if (a, b) == (0, T) else ttnn.slice(x, [0, 0, a, 0], [1, nkv, b, D])
                    ttnn.experimental.paged_fill_cache(cache, xp, pt, batch_idx=0)
                    if xp is not x:
                        ttnn.deallocate(xp)
            free(kr, vh)
        free(c, s, *(pt for ps in parts.values() for _, _, pt in ps))

    # ------------------------------------------------------------------ block forward + Markov chain
    def block(self):
        """Block forward on inp.x0 -> final-norm hidden BF16 [1, 1, 8, H] (writes the block K/V at p..p+7)."""
        inp = self.inp
        h, m = inp.x0, None
        for i, lw in enumerate(self.layers):
            if m is None:
                _, xb = lw["rn_in"](h)
            else:
                hn, xb = lw["rn_in"](h, m)
                free(m)
                if h is not inp.x0:
                    ttnn.deallocate(h)
                h = hn
            self._d(f"L{i}.x", xb)
            qkv = lw["qkv"](xb, rows=BLOCK)
            ttnn.deallocate(xb)
            self.mv_qkv[i]([qkv, self.qh, self.kh, self.v_cache[i]], inp.pos8)
            ttnn.deallocate(qkv)
            qr = self._norm_rope(self.qh, lw["q_norm"], inp.cos, inp.sin)
            kr = self._norm_rope(self.kh, lw["k_norm"], inp.cos, inp.sin)
            self._d(f"L{i}.qr", qr)
            self.mv_qk([qr, kr, self.q, self.k_cache[i]], inp.pos8)
            free(qr, kr)
            self._dbg(f"L{i} q / k / v")
            # is_causal=True: SDPA decode reads cur_pos only in causal mode (non-causal attends to the WHOLE page
            # table).  Causal with cur_pos = p + 7 for every query row = keys <= p + 7 = the non-causal block.
            kw = {"sliding_window_size": self.window + BLOCK} if i in self.local else {}
            att = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                self.q, self.k_mqa[i], self.v_mqa[i], self.pt_mqa, is_causal=True, cur_pos_tensor=inp.cur8,
                scale=self.D ** -0.5, program_config=self.sdpa_cfg, compute_kernel_config=CK_HIFI4,
                memory_config=DRAM, **kw)
            self.mv_rows([att, self.o_rows])
            ttnn.deallocate(att)
            self._d(f"L{i}.att", self.o_rows)
            self._dbg(f"L{i} sdpa")
            a = lw["o"](self.o_rows, rows=BLOCK)
            hn, xb = lw["rn_post"](h, a)
            free(a)
            if h is not inp.x0:
                ttnn.deallocate(h)
            h = hn
            act = lw["gu"](xb, rows=BLOCK)
            ttnn.deallocate(xb)
            m = lw["down"](act, rows=BLOCK)
            ttnn.deallocate(act)
            self._d(f"L{i}.h", m)
            self._dbg(f"L{i} mlp")
        hn, out = self.rn_final(h, m)
        free(m, hn)
        if h is not inp.x0:
            ttnn.deallocate(h)
        self._d("hid", out)
        return out

    def read_drafts(self, steps: int = BLOCK - 1) -> list[int]:
        """The last draft's first `steps` drafts (one device read of the draft log)."""
        if steps <= 0:
            return []
        return [int(x) for x in ttnn.to_torch(self.log).reshape(BLOCK - 1, 32)[:steps, 0]]

    def draft(self, feats, hidden=None, steps: int = BLOCK - 1):
        """Full draft: block forward, base logits of the 8 rows, f_rnn from feats (row selected by inp.sel), `steps`
        Markov steps -> step_ids (read_drafts()).  Returns the block hidden when `hidden` (tests), else frees it."""
        inp = self.inp
        hid = self.block()
        base = self.mm_base(hid, rows=BLOCK)  # BF16 [1, 1, 8, Vp]
        self._dbg("base logits")
        # f_rnn: mix every feature row with alpha[3], then select row sel into all 8 rows (one-hot matmul, exact)
        fm = self._mix(feats, self.L, self.rn_m)
        fr = ttnn.matmul(inp.sel, fm, dtype=BF16, compute_kernel_config=CK_HIFI4, memory_config=DRAM)  # [1, 1, 8, H]
        ttnn.deallocate(fm)
        self._d("f_rnn", fr)
        x = ttnn.concat([hid, fr, self.ones], dim=-1, memory_config=DRAM)  # [h_i | f_rnn | 1]
        ttnn.deallocate(fr)
        ctx = self.mm_ctx(x, rows=BLOCK)  # FP32 [1, 1, 8, 768]
        ttnn.deallocate(x)
        self._d("mctx", ctx)
        prev_ids = inp.anchor_id
        for i in range(steps):  # only the drafts the verify will use (Markov steps are sequential)
            s_in = self.zero_state if i == 0 else self.state[(i - 1) % 2]
            self.rnn[i + 1](prev_ids, self.w1, s_in, ctx, self.latent, self.state[i % 2])
            self.markov[i + 1](self.latent, base, self.fold_b, self.step_ids[i], self.log)
            self._dbg(f"markov step {i}")
            prev_ids = self.step_ids[i]
        free(ctx, base)
        if hidden:
            return hid
        ttnn.deallocate(hid)
        return None


def _users(rows_hd, users: int, nkv: int, D: int):
    """[1, 1, users, nkv*D] -> [1, users, nkv, D] TILE through row-major (frees the input)."""
    rm = ttnn.to_layout(rows_hd, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(rows_hd)
    rs = ttnn.reshape(rm, [1, users, nkv, D])
    t = ttnn.to_layout(rs, ttnn.TILE_LAYOUT)
    free(rm)
    return t


def _unembed_from_cache(cache):
    path = cache.path("unembed", ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    t = ttnn.to_torch(ttnn.load_tensor(str(path)))  # [1, 1, H, V_pad]
    return t.reshape(t.shape[-2], t.shape[-1]).T.contiguous()


def _i32(device, n):
    return ttnn.from_torch(torch.zeros(n, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=device, memory_config=DRAM)


def _rope_rows(cos, sin):
    """[n <= 32, 128] -> BF16 host tensors [1, 1, 32, 128] (row u = position u's rope row, padding rows 0)."""
    def f(t):
        x = torch.zeros(32, t.shape[-1])
        x[: t.shape[0]] = t
        return ttnn.from_torch(x.reshape(1, 1, 32, -1), dtype=BF16, layout=ttnn.TILE_LAYOUT)
    return f(cos), f(sin)


class DraftInputs:
    """Persistent per-cycle block inputs (host-written with DraftInputs.write)."""

    def __init__(self, device, dr: Drafter):
        self.device, self.dr = device, dr
        H, D = dr.H, dr.D
        z = lambda shape, dt=BF16: ttnn.zeros(shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device,  # noqa
                                              memory_config=DRAM)
        self.x0 = z([1, 1, BLOCK, H])
        self.cos = z([1, 1, 32, D])
        self.sin = z([1, 1, 32, D])
        self.sel = z([1, 1, BLOCK, 32])
        self.pos8 = _i32(device, BLOCK)
        self.cur8 = _i32(device, BLOCK)
        self.anchor_id = ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)

    def host_pos(self, p: int, sel_row: int):
        """Host tensors of a block at anchor position p that do not depend on the anchor token (sel_row: the feature
        row of position p-1 in the feats passed to draft()): [(host, device)] pairs for write_pairs()."""
        dr = self.dr
        pos = torch.arange(p, p + BLOCK)
        cos, sin = rope_tables(pos, dr.D, dr.theta)
        c, s = _rope_rows(cos, sin)
        sel = torch.zeros(BLOCK, 32)
        sel[:, sel_row] = 1
        i32 = lambda v: ttnn.from_torch(torch.as_tensor(v, dtype=torch.int32).reshape(-1), dtype=ttnn.int32,  # noqa
                                        layout=ttnn.ROW_MAJOR_LAYOUT)
        return [(c, self.cos), (s, self.sin),
                (ttnn.from_torch(sel.reshape(1, 1, BLOCK, 32), dtype=BF16, layout=ttnn.TILE_LAYOUT), self.sel),
                (i32(pos), self.pos8), (i32([p + BLOCK - 1] * BLOCK), self.cur8)]

    def host_anchor_id(self, anchor: int):
        """Host tensor of the anchor id (MarkovRNN's first prev; the spec verify's device embedding of row 0)."""
        aid = torch.zeros(1, 32, dtype=torch.int32)
        aid[0, 0] = anchor
        return [(ttnn.from_torch(aid, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.anchor_id)]

    def host_anchor(self, x0: torch.Tensor, anchor: int):
        """Host tensors of the anchor token: x0 [8, H] BF16 block input and the anchor id."""
        return [(ttnn.from_torch(x0.to(torch.bfloat16).reshape(1, 1, BLOCK, self.dr.H), dtype=BF16,
                                 layout=ttnn.TILE_LAYOUT), self.x0)] + self.host_anchor_id(anchor)

    @staticmethod
    def write_pairs(pairs):
        for h, d in pairs:
            ttnn.copy_host_to_device_tensor(h, d)

    def write(self, x0, p, anchor, sel_row):
        """Block inputs of the anchor `anchor` at position p (x0 [8, H] block input)."""
        self.write_pairs(self.host_pos(p, sel_row) + self.host_anchor(x0, anchor))


class CtxInputs:
    """Persistent inputs of a context write of `users` rows: positions and rope rows."""

    def __init__(self, device, dr: Drafter, users: int):
        self.dr, self.users = dr, users
        z = lambda: ttnn.zeros([1, 1, 32, dr.D], dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,  # noqa
                               memory_config=DRAM)
        self.cos, self.sin = z(), z()
        self.pos = _i32(device, users)

    def write(self, pos: torch.Tensor):
        """pos [users] positions (padding rows: a scratch position that is never read)."""
        dr = self.dr
        pos = torch.as_tensor(pos, dtype=torch.int64).reshape(-1)
        assert pos.numel() == self.users
        cos, sin = rope_tables(pos, dr.D, dr.theta)
        c, s = _rope_rows(cos, sin)
        hosts = [(c, self.cos), (s, self.sin),
                 (ttnn.from_torch(pos.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self.pos)]
        for h, dev in hosts:
            ttnn.copy_host_to_device_tensor(h, dev)


def _hi32(v):
    """Host int32 row-major tensor of v (any shape)."""
    return ttnn.from_torch(torch.as_tensor(v, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


class UserInputs:
    """Persistent block inputs of up to MAX_USERS users of the batched drafter (host-written with write()).
        x0[U]   BF16 [1, 1, 8U, H]   block inputs (row r*U + u = block row r of user u)
        sel[U]  BF16 [1, 1, 8U, 32]  one-hot: user u's rows pick feature row sel_row[u] (f_rnn)
        cos/sin BF16 [1, 1, 32, 128] rope rows (row r*U + u = position p_u + r)
        pos[kind] int32 [32]          physical cache rows of the 8U block rows (ring / pool layers)
        cur[u]  int32 [8]             p_u + 7 (SDPA cur_pos of user u's 8 kv-head entries)
        pt[kind][u] int32 [8, W]      user u's MQA page table (row j = block * 8 + j); written only when it changes
        anchor_id uint32 [1, 32]      word u = user u's anchor (first Markov prev)"""

    def __init__(self, device, dr: Drafter):
        self.dr = dr
        H, D = dr.H, dr.D
        z = lambda shape: ttnn.zeros(shape, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,  # noqa: E731
                                     memory_config=DRAM)
        self.x0 = {u: z([1, 1, BLOCK * u, H]) for u in range(1, MAX_USERS + 1)}
        self.sel = {u: z([1, 1, BLOCK * u, 32]) for u in range(1, MAX_USERS + 1)}
        self.cos, self.sin = z([1, 1, 32, D]), z([1, 1, 32, D])
        kinds = sorted(set(dr.kinds))
        self.pos = {k: _i32(device, 32) for k in kinds}
        self.cur = [_i32(device, BLOCK) for _ in range(MAX_USERS)]
        self.pt = {k: [ttnn.from_torch(torch.full((dr.nkv, dr.W), dr.NB * dr.nkv, dtype=torch.int32),
                                       dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                       memory_config=DRAM) for _ in range(MAX_USERS)] for k in kinds}
        self._pt_rows = {}  # (kind, u) -> host block row last written
        self.anchor_id = ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)

    def write(self, users):
        """users: U <= 4 tuples (slot, p, anchor, x0 [8, H], sel_row, page_row): user u's block rows at positions
        p..p+7 of the sequence in `slot` with vLLM page row `page_row` (host int [W]); f_rnn from feature row sel_row
        of the feats passed to draft_users."""
        dr = self.dr
        U = len(users)
        if not 1 <= U <= MAX_USERS:
            raise ValueError(f"UserInputs: 1..{MAX_USERS} users, got {U}")
        n, nkv = BLOCK * U, dr.nkv
        x0 = torch.zeros(n, dr.H, dtype=torch.bfloat16)
        sel = torch.zeros(n, 32)
        pos = torch.zeros(n, dtype=torch.int64)
        phys = {k: torch.zeros(32, dtype=torch.int32) for k in self.pos}
        aid = torch.zeros(1, 32, dtype=torch.int32)
        pairs = []
        for u, (slot, p, anchor, xu, sel_row, page_row) in enumerate(users):
            if not 0 <= slot < dr.S or p < 0 or p + BLOCK > dr.max_seq_len or not 0 <= sel_row < 32:
                raise ValueError(f"UserInputs: user {u} slot {slot} p {p} sel_row {sel_row} out of range")
            rows = [r * U + u for r in range(BLOCK)]
            x0[rows] = torch.as_tensor(xu).to(torch.bfloat16).reshape(BLOCK, dr.H)
            sel[rows, sel_row] = 1
            aid[0, u] = int(anchor)
            for r in range(BLOCK):
                pos[r * U + u] = p + r
                for k in phys:
                    phys[k][r * U + u] = dr.phys(k, slot, p + r, page_row)
            pairs.append((_hi32([p + BLOCK - 1] * BLOCK), self.cur[u]))
            for k in self.pt:
                blocks = (slot * RING + torch.arange(dr.W, dtype=torch.int64) % RING if k == "ring"
                          else dr.page_blocks(page_row))
                last = self._pt_rows.get((k, u))
                if last is None or not torch.equal(last, blocks):
                    self._pt_rows[(k, u)] = blocks
                    pt = blocks[None, :] * nkv + torch.arange(nkv, dtype=torch.int64)[:, None]
                    pairs.append((_hi32(pt), self.pt[k][u]))
        cos, sin = rope_tables(pos, dr.D, dr.theta)
        c, s = _rope_rows(cos, sin)
        bf = lambda t, rows, w: ttnn.from_torch(t.reshape(1, 1, rows, w), dtype=BF16,  # noqa: E731
                                                layout=ttnn.TILE_LAYOUT)
        pairs += [(bf(x0, n, dr.H), self.x0[U]), (bf(sel, n, 32), self.sel[U]), (c, self.cos), (s, self.sin),
                  (ttnn.from_torch(aid, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.anchor_id)]
        pairs += [(_hi32(phys[k]), self.pos[k]) for k in phys]
        for h, dev in pairs:
            ttnn.copy_host_to_device_tensor(h, dev)


class CtxRowInputs:
    """Persistent inputs of a batched context write of `rows` rows: physical cache rows per layer kind + rope rows."""

    def __init__(self, device, dr: Drafter, rows: int):
        self.dr, self.rows = dr, rows
        z = lambda: ttnn.zeros([1, 1, 32, dr.D], dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,  # noqa
                               memory_config=DRAM)
        self.cos, self.sin = z(), z()
        self.pos = {k: _i32(device, 32) for k in sorted(set(dr.kinds))}

    def write(self, slots, positions, pages):
        """Row x: slot slots[x], position positions[x], page row pages[x] (host vLLM block ids; pages: [rows, W]
        tensor or a sequence of rows).  Pad rows: any position rewritten before it is read (e.g. replicas of row 0)."""
        dr = self.dr
        slots = [int(v) for v in torch.as_tensor(slots).reshape(-1)]
        pos = torch.as_tensor(positions, dtype=torch.int64).reshape(-1)
        if len(slots) != self.rows or pos.numel() != self.rows or len(pages) < self.rows:
            raise ValueError(f"CtxRowInputs: {self.rows} rows expected")
        phys = {k: torch.zeros(32, dtype=torch.int32) for k in self.pos}
        for x in range(self.rows):
            if not 0 <= slots[x] < dr.S or not 0 <= int(pos[x]) < dr.max_seq_len:
                raise ValueError(f"CtxRowInputs: row {x} slot {slots[x]} position {int(pos[x])} out of range")
            for k in phys:
                phys[k][x] = dr.phys(k, slots[x], int(pos[x]), pages[x])
        cos, sin = rope_tables(pos, dr.D, dr.theta)
        c, s = _rope_rows(cos, sin)
        for h, dev in [(c, self.cos), (s, self.sin)] + [(_hi32(phys[k]), self.pos[k]) for k in phys]:
            ttnn.copy_host_to_device_tensor(h, dev)
