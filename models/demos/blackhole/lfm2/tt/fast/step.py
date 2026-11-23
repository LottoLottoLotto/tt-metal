# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fast decode step for Lfm2Model: `Lfm2Model.decode_forward` with the tt/fast modules swapped in (model.py is not
modified; FastStep shares the model's weights, KV caches, conv state and page table).

    from models.demos.blackhole.lfm2.tt.fast.step import FastStep
    fast = FastStep(model)                                  # parts=DEFAULT_PARTS, after the model is built and the
                                                            # KV caches are allocated, before any trace capture
    logits = fast.decode_forward(tok, cur_pos, rope_idx, page_table)   # same inputs / output as the stock step

Output: logits [1, 1, B, vocab] in model.logits_dtype (FP32 tests, BF16 serving).  Device-only, trace-capturable
(every program is address-keyed: rebuilt only when a tensor address changes, no host reads).

Parts (any subset; unknown names are an error):
  moe   MoEDecode per MoE layer (single row: B > 1 falls back to the stock MoE)
  moe2  MoEDecodeV2 (moe_decode2.py, MoEFuse): router inside the experts program (one launch), bitwise MoEDecode;
        overrides moe (single row)
  moer  MoEDecodeRows (moe_rows.py) for every decode width 2..16: the union of the rows' experts streamed once, each
        row bitwise MoEDecodeV2 on that row alone (two launches); rows with cur_pos < 0 select no expert
  norm  ResidualRMSNorm chain: every residual add fused with the following norm (h / n in L1)
  conv  ConvDecode tail between the conv projections (every decode width <= 16)
  attn  AttnDecode front (qkv split, q/k norm, RoPE, paged KV write; cur_pos < 0 rows skip the write; B <= 16)
  mm    StreamDecodeMatmul for conv in/out, attention qkv/out, dense w1|w3 + SwiGLU and w2, LM head (B <= 32)
  nmm   (round 2, bitwise vs the chain above) NormStreamDecodeMatmul: operator norm + conv in_proj / attention qkv
        and ffn norm + dense w1|w3 in one program each; layer 0 also folds the embedding gather (B <= 16)
  sdpa  SDPA decode program config `sdpa` (grid x, y, k_chunk; default (8, 8, 512)) instead of the stock
        (8, 8, 64): not bitwise (k-chunked online softmax), PCC 0.99996 vs stock config; faster at long context
  convf ConvOpDecode (conv_fused.py, ConvFuse): conv in_proj + tail + out_proj in one program, bitwise; with norm
        also the operator norm (norm prologue, placement (0, 6)): h, out = op(a, b); overrides nmm on conv layers
        (layer 0 then keeps the separate embedding); every decode width
  afront AttnQKVFront (attn_fused.py, AttnFuse): qkv matmul + front in one program, bitwise (takes n: overrides
        nmm on attention layers; B <= 8)
  anfront AttnNormQKVFront (attn_fused.py, AttnFuse): operator norm + qkv matmul + front in one program, bitwise
        (with norm; overrides afront / nmm on attention layers; B <= 8); ~0.1 ms/step SLOWER than afront in the
        whole step (round 3, logs/fast_step_r3a.json): not in the defaults
  asdpa SDPADecode (attn_fused.py, AttnFuse): custom paged SDPA decode writing the flat [1, 1, B, 2048] output (no
        reshape) for every decode width <= 16: work spread over all cores by the live rows' positions (idle rows
        cost nothing) and batch-invariant; not bitwise vs stock (PCC vs FP64 0.99996-0.99999, stock 0.99964-0.99996)
Modules a part needs but that do not apply to the current B fall back to the stock code per call.

Decode widths (vLLM decode buckets): the step runs at any width in `decode_widths(model.batch)` (1, 2, 4, ... and
model.batch).  Row-count-specific modules (moer, conv, convf, asdpa) are built once per width (sharing weights); a
width w < model.batch processes decode rows / conv-state slots [0, w) and leaves the other slots' state unchanged.
"""
from __future__ import annotations

import copy

import ttnn

from ..common import CK_BFP8, CK_HIFI4, free
from ..layers import DRAM, linear, sdpa_compute
from ..moe import MoE
from .attn_decode import AttnDecode
from .conv_decode import ConvDecode
from .decode_matmul import PRESETS, StreamDecodeMatmul, cached_stream_weight
from .moe_decode import MoEDecode
from .norm_matmul import NormStreamDecodeMatmul
from .residual_norm import ResidualRMSNorm

L1 = ttnn.L1_MEMORY_CONFIG
PARTS = ("moe", "moe2", "moer", "norm", "conv", "attn", "mm", "nmm", "sdpa", "convf", "afront", "anfront", "asdpa")
# bitwise-equivalent to the round-1 set ("moe", "norm", "conv", "attn", "mm"), round 3 (logs/fast_step_r3a.json:
# 64/64 bitwise logits vs round 2 at pos 128 / 2000 / 32000); sdpa / asdpa are opt-in (not bitwise)
DEFAULT_PARTS = ("moe2", "moer", "norm", "conv", "attn", "mm", "nmm", "convf", "afront")
SDPA_TUNED = (8, 8, 512)
STOCK_SDPA = (8, 8, 64)


def decode_widths(batch: int) -> list[int]:
    """The decode widths of a `batch`-slot model: powers of two below it, then batch itself."""
    widths, w = [], 1
    while w < batch:
        widths.append(w)
        w *= 2
    return widths + [batch]


def B_rows(tok) -> int:
    """Decode rows of a [1, B] token tensor."""
    return int(tok.shape[-1])



class FastStep:
    """decode_forward of `model` with the selected fast parts (see the module docstring)."""

    def __init__(self, model, parts=DEFAULT_PARTS, src=None, sdpa=SDPA_TUNED):
        parts = set(parts)
        if parts - set(PARTS):
            raise ValueError(f"unknown parts {sorted(parts - set(PARTS))}; known {PARTS}")
        src = src if src is not None else model.src
        self.m, self.parts, self.sdpa = model, parts, tuple(sdpa)
        dev, cfg, cache = model.device, model.cfg, model.cache
        self.widths = decode_widths(model.batch)
        self.moe, self.attn, self.op_rn, self.ffn_rn, self.mm, self.nmm = {}, {}, {}, {}, {}, {}
        self.afront, self.anfront, self.moe2 = {}, {}, {}
        # per decode width w: {layer: module} (moer / conv / convf), SDPADecode (asdpa)
        self.moer = {w: {} for w in self.widths}
        self.conv = {w: {} for w in self.widths}
        self.convf = {w: {} for w in self.widths}
        self.asdpa = {}
        self.convf_norm = "norm" in parts  # convf built with the operator-norm prologue
        # sibling modules (still evolving): imported only when requested
        if parts & {"afront", "anfront", "asdpa"}:
            from .attn_fused import AttnNormQKVFront, AttnQKVFront, SDPADecode
        if "convf" in parts:
            from .conv_fused import ConvOpDecode
        if "asdpa" in parts:
            self.asdpa = {w: SDPADecode(dev, rows=w) for w in self.widths}
        for layer in model.layers:
            p, i = f"model.layers.{layer.i}", layer.i
            dense = not isinstance(layer.ffn, MoE)
            if "moe2" in parts and not dense:
                from .moe_decode2 import MoEDecodeV2

                self.moe2[i] = MoEDecodeV2.from_moe(layer.ffn, cache=cache, layer=i)
            if "moer" in parts and not dense:
                from .moe_rows import MoEDecodeRows

                router = self.moe2[i].router_w if i in self.moe2 else None
                for w in self.widths:
                    if 2 <= w <= 16:
                        if router is None:
                            mr = MoEDecodeRows.from_moe(layer.ffn, w, cache=cache, layer=i)
                            router = mr.router_w
                        else:
                            mr = MoEDecodeRows.from_moe(layer.ffn, w, router_stream=router)
                        self.moer[w][i] = mr
            if "moe" in parts and not dense:
                self.moe[i] = MoEDecode.from_moe(layer.ffn, cache=cache, layer=i)
            if "mm" in parts:
                if layer.attn:
                    shapes = {"qkv": ("attn_qkv", f"L{i}.attn.qkv"), "wo": ("attn_out", f"L{i}.attn.out_proj")}
                else:
                    shapes = {"in": ("conv_in", f"L{i}.conv.in_proj"), "out": ("conv_out", f"L{i}.conv.out_proj")}
                if dense:
                    shapes["w2"] = ("mlp_w2", f"L{i}.mlp.w2")
                for name, (preset, key) in shapes.items():
                    self.mm[(i, name)] = StreamDecodeMatmul(
                        dev, cached_stream_weight(cache, key, PRESETS[preset]), PRESETS[preset], CK_BFP8)
                if dense:
                    self.mm[(i, "w13")] = StreamDecodeMatmul(
                        dev, cached_stream_weight(cache, f"L{i}.mlp.w1", PRESETS["mlp_w13"], key3=f"L{i}.mlp.w3"),
                        PRESETS["mlp_w13"], CK_BFP8)
            if "conv" in parts and not layer.attn:
                taps = None
                for w in self.widths:
                    self.conv[w][i] = ConvDecode(layer.op, rows=w, taps=taps)
                    taps = self.conv[w][i].taps
            if "convf" in parts and not layer.attn:
                ws = [self.mm[(i, k)].weight if (i, k) in self.mm else cached_stream_weight(cache, key, PRESETS[pr])
                      for k, key, pr in (("in", f"L{i}.conv.in_proj", "conv_in"), ("out", f"L{i}.conv.out_proj",
                                                                                   "conv_out"))]
                kw = dict(norm=(src.get(f"{p}.operator_norm.weight"), cfg.norm_eps), placement=(0, 6)) \
                    if "norm" in parts else {}
                taps = None
                for w in self.widths:
                    self.convf[w][i] = ConvOpDecode(layer.op, *ws, rows=w, taps=taps, **kw)
                    taps = self.convf[w][i].taps
            if "anfront" in parts and "norm" in parts and layer.attn:
                w = self.mm[(i, "qkv")].weight if (i, "qkv") in self.mm else None
                self.anfront[i] = AttnNormQKVFront(layer.op, model.rope, src.get(f"{p}.operator_norm.weight"),
                                                   cfg.norm_eps, cache=cache, layer=i, weight=w)
            if "afront" in parts and layer.attn:
                w = self.mm[(i, "qkv")].weight if (i, "qkv") in self.mm else None
                self.afront[i] = AttnQKVFront(layer.op, model.rope, cache, i, weight=w)
            if "attn" in parts and layer.attn:
                self.attn[i] = AttnDecode(layer.op, model.rope)
            if "norm" in parts:
                kw = dict(h_memory_config=L1, n_memory_config=L1)
                self.op_rn[i] = ResidualRMSNorm(dev, src.get(f"{p}.operator_norm.weight"), cfg.norm_eps, **kw)
                self.ffn_rn[i] = ResidualRMSNorm(dev, src.get(f"{p}.ffn_norm.weight"), cfg.norm_eps, **kw)
            if "nmm" in parts:
                specs = [("qkv", "attn_qkv", f"L{i}.attn.qkv", None, "operator_norm")] if layer.attn else [
                    ("in", "conv_in", f"L{i}.conv.in_proj", None, "operator_norm")]
                if dense:
                    specs.append(("w13", "mlp_w13", f"L{i}.mlp.w1", f"L{i}.mlp.w3", "ffn_norm"))
                for name, preset, key, key3, norm in specs:
                    w = self.mm[(i, name)].weight if (i, name) in self.mm else cached_stream_weight(
                        cache, key, PRESETS[preset], key3=key3)
                    self.nmm[(i, name)] = NormStreamDecodeMatmul(
                        dev, src.get(f"{p}.{norm}.weight"), cfg.norm_eps, w, PRESETS[preset], CK_BFP8,
                        output_memory_config=L1 if name == "w13" else DRAM)
        if "norm" in parts:
            self.final_rn = ResidualRMSNorm(dev, src.get("model.embedding_norm.weight"), cfg.norm_eps,
                                            h_memory_config=L1, n_memory_config=L1)
        if "mm" in parts and model.lm_head is not None:
            self.mm[(-1, "head")] = StreamDecodeMatmul(dev, cached_stream_weight(cache, "lm_head", PRESETS["lm_head"]),
                                                       PRESETS["lm_head"], CK_HIFI4, out_dtype=model.logits_dtype)
        # speculative verification (enable_verify): candidate Bx rows per conv layer, verify-mode modules per width
        self.cand, self.convf_v, self.afront_v, self.verify_widths = {}, {}, {}, []
        self._verify = False

    def enable_verify(self, widths) -> None:
        """Build the verify-step modules (decode_forward(..., verify=True)) for these row counts (2..8): the rows are
        consecutive tokens of the sequence in conv-state row 0.  Needs the parts moe2, moer, norm, convf, afront,
        asdpa; shares every weight.  Conv layers write each row's Bx to self.cand[layer] (one [1, 1, 32, 2048] BF16
        tensor per conv layer) instead of updating the state; tt/speculate.py SpecCommit commits the accepted row."""
        need = {"moe2", "moer", "norm", "convf", "afront", "asdpa", "mm", "nmm"}
        if need - self.parts:
            raise ValueError(f"FastStep.enable_verify needs the parts {sorted(need)}")
        from .attn_fused import AttnQKVFront, SDPADecode
        from .conv_fused import ConvOpDecode
        from .moe_rows import MoEDecodeRows

        m, dev = self.m, self.m.device
        for layer in m.layers:
            i = layer.i
            if layer.attn:
                if i not in self.afront_v:
                    self.afront_v[i] = AttnQKVFront(layer.op, m.rope, m.cache, i, verify=True, share=self.afront[i])
            elif i not in self.cand:
                self.cand[i] = ttnn.zeros([1, 1, 32, m.cfg.hidden], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                          device=dev, memory_config=DRAM)
        for w in widths:
            if not 2 <= w <= 8:
                raise ValueError(f"FastStep.enable_verify: {w} rows (2..8)")
            if w in self.verify_widths:
                continue
            if w not in self.asdpa:
                self.asdpa[w] = SDPADecode(dev, rows=w)
            moer = self.moer.setdefault(w, {})
            convf = self.convf_v.setdefault(w, {})
            for layer in m.layers:
                i = layer.i
                if i in self.moe2 and i not in moer:
                    moer[i] = MoEDecodeRows.from_moe(layer.ffn, w, router_stream=self.moe2[i].router_w)
                if not layer.attn:
                    ref = self.convf[self.widths[0]][i]
                    p = f"model.layers.{i}"
                    convf[i] = ConvOpDecode(layer.op, ref.w_in, ref.w_out, rows=w, taps=ref.taps,
                                            norm=(m.src.get(f"{p}.operator_norm.weight"), m.cfg.norm_eps),
                                            placement=(0, 6), verify=self.cand[i])
            self.verify_widths.append(w)

    def view(self, parts) -> "FastStep":
        """Same modules restricted to `parts` (a subset of the constructed ones): ablations without rebuilding."""
        v = copy.copy(self)
        v.parts = set(parts)
        v.moe = self.moe if "moe" in parts else {}
        v.moe2 = self.moe2 if "moe2" in parts else {}
        empty = {w: {} for w in self.widths}
        v.moer = self.moer if "moer" in parts else empty
        v.conv = self.conv if "conv" in parts else empty
        v.attn = self.attn if "attn" in parts else {}
        v.mm = self.mm if "mm" in parts else {}
        v.nmm = self.nmm if "nmm" in parts else {}
        v.convf = self.convf if "convf" in parts else empty
        v.afront = self.afront if "afront" in parts else {}
        v.anfront = self.anfront if "anfront" in parts and "norm" in parts else {}
        v.asdpa = self.asdpa if "asdpa" in parts else {}
        return v

    # ------------------------------------------------------------------ pieces
    def _lin(self, i, name, x, w, **kw):
        mm = self.mm.get((i, name))
        return mm(x, **kw) if mm is not None else linear(x, w)

    def _sdpa_cfg(self):
        gx, gy, kc = self.sdpa if "sdpa" in self.parts else STOCK_SDPA
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), q_chunk_size=32,
                                      k_chunk_size=kc, exp_approx_mode=False)

    def _conv_ok(self, layer, B):
        return not self._verify and layer.i in self.conv.get(B, ()) and B <= 16

    def _convs(self, B):
        """The fused conv operators of width B (verify-mode ones during a verify step)."""
        return self.convf_v.get(B, {}) if self._verify else self.convf.get(B, {})

    def _convf_ok(self, layer, B):
        return layer.i in self._convs(B) and ("norm" in self.parts) == self.convf_norm

    def _attn_ok(self, layer, B):
        return (not self._verify and layer.i in self.attn and B <= 16) or self._afront_ok(layer, B)

    def _afront_ok(self, layer, B):
        return layer.i in (self.afront_v if self._verify else self.afront) and B <= 8

    def _anfront_ok(self, layer, B):
        return layer.i in self.anfront and B <= 8

    def _op(self, layer, n, ctx, first=None, qd=None):
        """Operator on the normed input n; `first`: its first projection (qkv / conv in_proj) already computed by a
        fused norm + matmul, or `qd`: the attention's SDPA query (AttnNormQKVFront); n unused then."""
        op, B = layer.op, ctx["B"]
        if layer.attn:
            if qd is None and not self._attn_ok(layer, B):
                return op.decode(n, ctx["cos"], ctx["sin"], ctx["rope"], ctx["cur_pos"], ctx["page_table"])
            if qd is not None:
                pass
            elif first is None and self._afront_ok(layer, B):
                front = self.afront_v if self._verify else self.afront
                qd = front[layer.i](n, ctx["cur_pos"], ctx["page_table"])
            else:
                qkv = first if first is not None else self._lin(layer.i, "qkv", n, op.wqkv)
                qd = self.attn[layer.i](qkv, ctx["cur_pos"], ctx["page_table"])
                ttnn.deallocate(qkv)
            sdpa = self.asdpa.get(B)
            if sdpa is not None:
                flat = sdpa(qd, op.k_cache, op.v_cache, ctx["cur_pos"], ctx["page_table"])
                ttnn.deallocate(qd)
                y = self._lin(layer.i, "wo", flat, op.wo)
                ttnn.deallocate(flat)
                return y
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                qd, op.k_cache, op.v_cache, cur_pos_tensor=ctx["cur_pos"], page_table_tensor=ctx["page_table"],
                is_causal=True, scale=op.scale, program_config=self._sdpa_cfg(), compute_kernel_config=ctx["sdpa_ck"])
            ttnn.deallocate(qd)
            flat = ttnn.reshape(out, [1, 1, B, op.nh * op.hd])
            y = self._lin(layer.i, "wo", flat, op.wo)
            free(out)
            return y
        if first is None and self._convf_ok(layer, B):
            return self._convs(B)[layer.i](n)
        if not self._conv_ok(layer, B):
            if B != op.batch:
                raise ValueError(f"FastStep: decode width {B} needs the conv / convf part (conv state of {op.batch})")
            return op.decode(n)
        bcx = first if first is not None else self._lin(layer.i, "in", n, op.in_proj)
        y = self.conv[B][layer.i](bcx)
        ttnn.deallocate(bcx)
        out = self._lin(layer.i, "out", y, op.out_proj)
        ttnn.deallocate(y)
        return out

    def _ffn(self, layer, n, cur_pos=None):
        moer = self.moer.get(n.shape[-2], {}).get(layer.i)
        if moer is not None:
            return moer(n, cur_pos)  # inactive rows (cur_pos < 0) select no expert
        if layer.i in self.moe2 and n.shape[-2] == 1:
            return self.moe2[layer.i](n)
        if layer.i in self.moe and n.shape[-2] == 1:
            return self.moe[layer.i](n)
        if (layer.i, "w13") in self.mm:
            h = self.mm[(layer.i, "w13")](n, output_memory_config=L1)
            y = self.mm[(layer.i, "w2")](h)
            ttnn.deallocate(h)
            return y
        return layer.ffn(n)

    def _fused_op_norm(self, layer, B):
        """NormStreamDecodeMatmul of this layer's operator norm + first projection, when it applies."""
        if self._afront_ok(layer, B) if layer.attn else self._convf_ok(layer, B):
            return None  # the fused operator takes the normed input
        ok = self._attn_ok(layer, B) if layer.attn else self._conv_ok(layer, B)
        return self.nmm.get((layer.i, "qkv" if layer.attn else "in")) if ok else None

    def _head(self, n):
        head = self.mm.get((-1, "head"))
        if head is not None:
            return head(n)
        return ttnn.linear(n, self.m.lm_head, dtype=self.m.logits_dtype, compute_kernel_config=CK_HIFI4,
                           memory_config=DRAM)

    # ------------------------------------------------------------------ step
    def decode_forward(self, tok, cur_pos, rope_idx, page_table, verify: bool = False):
        """Device-only decode step (trace-capturable): logits [1, 1, B, vocab] (model.logits_dtype).  verify: the B
        rows are consecutive tokens of the sequence in conv-state row 0 (enable_verify): KV written for every row,
        the conv state left unchanged (each row's Bx -> self.cand)."""
        if verify and B_rows(tok) not in self.verify_widths:
            raise ValueError(f"FastStep: no verify modules for {B_rows(tok)} rows (enable_verify)")
        self._verify = verify
        try:
            return self._decode_forward(tok, cur_pos, rope_idx, page_table)
        finally:
            self._verify = False

    def _decode_forward(self, tok, cur_pos, rope_idx, page_table):
        m = self.m
        B = B_rows(tok)
        layers = m.layers
        # layer 0: the embedding gather folded into the norm + first projection whenever nmm applies, even where
        # convf / afront would take the other layers (saves the ~25 us embedding program + reshape)
        l0 = layers[0]
        ok0 = self._attn_ok(l0, B) and l0.i in self.attn if l0.attn else self._conv_ok(l0, B)
        fused0 = self.nmm.get((l0.i, "qkv" if l0.attn else "in")) if "norm" in self.parts and ok0 else None
        embed0 = fused0 is not None and B <= 16
        if not embed0:
            e = ttnn.embedding(tok, m.embed, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
            x = ttnn.reshape(e, [1, 1, B, m.cfg.hidden])
        need_rope = any(layer.attn and not (self._attn_ok(layer, B) or self._anfront_ok(layer, B)) for layer in layers)
        cos, sin = m.rope.decode(rope_idx) if need_rope else (None, None)
        ctx = {"cos": cos, "sin": sin, "rope": m.rope, "cur_pos": cur_pos, "page_table": page_table,
               "sdpa_ck": sdpa_compute(), "B": B}
        if "norm" not in self.parts:
            for layer in layers:
                nn = layer.op_norm(x)
                o = self._op(layer, nn, ctx)
                ttnn.deallocate(nn)
                h = ttnn.add(x, o, memory_config=DRAM)
                free(x, o)
                n2 = layer.ffn_norm(h)
                f = self._ffn(layer, n2, cur_pos)
                ttnn.deallocate(n2)
                x = ttnn.add(h, f, memory_config=DRAM)
                free(h, f)
            free(cos, sin)
            n = m.final_norm(x)
            ttnn.deallocate(x)
            logits = self._head(n)
            ttnn.deallocate(n)
            return logits
        # pending residual (a, b): the next norm consumes a + b (b None: a alone, the embedding)
        a, b = (None, None) if embed0 else (x, None)
        for j, layer in enumerate(layers):
            fused = fused0 if j == 0 else self._fused_op_norm(layer, B)
            if j == 0 and embed0:
                xr, first = fused0.embed(tok, m.embed)
                o = self._op(layer, None, ctx, first=first)
            elif layer.attn and self._anfront_ok(layer, B):
                xr, qd = self.anfront[layer.i](a, b, cur_pos, page_table)
                o = self._op(layer, None, ctx, qd=qd)
            elif not layer.attn and self._convf_ok(layer, B):
                op = self._convs(B)[layer.i]
                xr, o = op(a, b) if b is not None else op(a)
            elif fused is not None:
                xr, first = fused(a, b)
                o = self._op(layer, None, ctx, first=first)
            else:
                xr, n = self.op_rn[layer.i](a, b)
                o = self._op(layer, n, ctx)
                ttnn.deallocate(n)
            if b is not None:
                free(a, b)
            w13 = self.nmm.get((layer.i, "w13"))
            if w13 is not None and (layer.i, "w2") in self.mm:
                h, hm = w13(xr, o)
                f = self.mm[(layer.i, "w2")](hm)
                ttnn.deallocate(hm)
            else:
                h, n2 = self.ffn_rn[layer.i](xr, o)
                f = self._ffn(layer, n2, cur_pos)
                ttnn.deallocate(n2)
            free(xr, o)
            a, b = h, f
        _, n = self.final_rn(a, b)
        free(a, b, cos, sin)
        logits = self._head(n)
        ttnn.deallocate(n)
        return logits
