# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Fast decode step for Lfm3Model: `Lfm3Model.decode_forward_stock` with the tt/fast modules swapped in.  FastStep
shares the model's weights (stream-layout copies are extra), KV caches, conv state and page table.

    model.enable_fast_decode(parts)                         # or FastStep(model, parts) directly; after the KV caches
                                                            # are allocated, before any trace capture
    logits = model.decode_forward(tok, cur_pos, rope_idx, page_table)  # same inputs / output as the stock step

Output: logits [1, 1, B, vocab] in model.logits_dtype (FP32 tests, BF16 serving).  Device-only, trace-capturable
(every program is address-keyed: rebuilt only when a tensor address changes, no host reads).

LFM3 layer (sandwich norm): h = x + post_op_norm(op(op_norm(x))); out = h + post_ffn_norm(mlp(ffn_norm(h))).  With
the norm part the step carries a pending residual (a, b, post): the next norm consumes a + post(b) (b None: a alone).

Parts (any subset; unknown names are an error):
  norm   ResidualRMSNorm(post_weight=...) chain: every residual add fused with the post norm before it and the norm
         after it (one launch: h = a + post(b), n = norm(h); h / n in L1)
  mm     StreamDecodeMatmul (DRAM-streaming, BFP8 / BFP4 weights) for conv in/out, attention qkvg / o_proj, dense
         w1|w3 + SwiGLU and w2, LM head
  nmm    NormStreamDecodeMatmul (needs norm): sandwich residual + operator norm + conv in_proj / attention qkvg, and
         + ffn norm + dense w1|w3 in one program each; layer 0 also folds the embedding gather
  conv   ConvDecode: B*x, 7-tap conv with the 6-row state, C*, state shift in one launch (every decode width)
  convf  ConvOpDecode: conv in_proj + tail + out_proj in one program (with norm: + the sandwich residual / operator
         norm prologue); overrides nmm / conv on conv layers (layer 0 keeps nmm.embed + conv when those apply)
  attn   AttnDecode front: qkvg split, q/k norm, RoPE, paged KV write, sigmoid gate
  afront AttnQKVFront: qkvg matmul + AttnDecode front in one program (overrides nmm on attention layers)
  asdpa  SDPADecode: paged SDPA decode (sliding layers read only their 256-token window) with the head gate applied,
         flat [1, 1, B, 2048] output, batch-invariant (used with attn / afront)
  anfront AttnNormQKVFront (needs norm): sandwich residual + operator norm + qkvg matmul + front in one program
         (overrides afront / nmm on attention layers; bitwise ResidualRMSNorm + AttnQKVFront)
  mlpf   FusedMLPDecode (needs norm): sandwich residual + ffn norm + w1|w3 SwiGLU + w2 in ONE program (the w2 workers
         stream their weights while w1|w3 runs); overrides the nmm w13 + mm w2 pair (bitwise it)
Modules a part needs but that do not apply to the current B fall back to the stock code per call.

Decode widths: the step runs at any width in `decode_widths(model.batch)` (1, 2, 4, ... and model.batch);
row-count-specific modules (conv, convf, asdpa) are built once per width (sharing weights); a width w < model.batch
processes decode rows / conv-state slots [0, w) and leaves the other slots' state unchanged.

Verify mode (speculative decoding, tt/speculate.py): enable_verify(widths) builds verify-mode modules for 2..16 rows;
decode_forward(..., verify=True) runs the rows as groups of consecutive tokens, one group per sequence (a shared
round; the row map self.vmap gives each row its group offset, group length and conv-state slot): the plain width-1
module sequence, conv layers chain each group's rows from its slot's state and write every row's Bx to self.cand
instead of the state, one front worker per 32-row KV tile of a group patches its rows.  Row j's logits are bitwise
its sequence's sequential 1-row steps', whatever the other groups.
"""
from __future__ import annotations

import copy

import torch
import ttnn

from ..common import CK_BFP8, CK_HIFI4, free
from ..layers import DRAM, GATE_PAD, linear

L1 = ttnn.L1_MEMORY_CONFIG
PARTS = ("norm", "mm", "nmm", "conv", "convf", "attn", "afront", "asdpa", "mlpf", "anfront")
DEFAULT_PARTS = ("norm", "mm", "nmm", "conv", "convf", "attn", "afront", "asdpa", "mlpf", "anfront")
MAX_ROWS = 16


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

    def __init__(self, model, parts=DEFAULT_PARTS, src=None):
        from .decode_matmul import StreamDecodeMatmul, cached_stream_weight, preset_for
        from .norm_matmul import NormStreamDecodeMatmul
        from .residual_norm import ResidualRMSNorm

        parts = set(parts)
        if parts - set(PARTS):
            raise ValueError(f"unknown parts {sorted(parts - set(PARTS))}; known {PARTS}")
        if model.batch > MAX_ROWS:
            raise ValueError(f"FastStep: at most {MAX_ROWS} decode rows (model.batch {model.batch})")
        src = src if src is not None else model.src
        self.m, self.parts = model, parts
        dev, cfg, cache = model.device, model.cfg, model.cache
        self.widths = decode_widths(model.batch)
        self.mm, self.nmm, self.op_rn, self.ffn_rn = {}, {}, {}, {}
        self.attn, self.afront, self.anfront, self.mlpf = {}, {}, {}, {}
        self.conv = {w: {} for w in self.widths}
        self._convf_kw = {}  # per conv layer: ConvOpDecode's norm / post / placement kwargs (verify modules reuse them)
        self.convf = {w: {} for w in self.widths}
        self.asdpa = {}
        self.convf_norm = "norm" in parts  # convf built with the sandwich / operator-norm prologue
        if parts & {"attn", "afront", "anfront", "asdpa"}:
            from .attn_decode import AttnDecode
            from .attn_fused import AttnNormQKVFront, AttnQKVFront, SDPADecode
        if parts & {"conv", "convf"}:
            from .conv_decode import ConvDecode, replicated_taps
            from .conv_fused import ConvOpDecode
        if "asdpa" in parts:
            self.asdpa = {w: SDPADecode(dev, rows=w) for w in self.widths}

        def nw(name):
            return src.get(f"{name}.weight").float()

        def stream(key, name, like, key3=None, preset=None):
            """(stream-layout copy of WeightCache entry `key` in its real device format, its preset) for projection
            `name` (like: the resident tensor; the preset is tuned per format), one device copy per layout shared by
            every module.  With mlpf the w1|w3 modules use the fused MLP's layout (one copy)."""
            if preset is None:
                preset = preset_for(name, like.dtype)
                if name == "mlp_w13" and "mlpf" in parts:
                    fused = fused_presets(like.dtype)[0]
                    preset = preset if preset.tag == fused.tag else fused
            k = (key, preset.tag, like.dtype)
            if k not in self._streams:
                self._streams[k] = cached_stream_weight(cache, key, preset, dtype=like.dtype, key3=key3)
            return self._streams[k], preset

        if "mlpf" in parts:
            from .mlp_fused import FusedMLPDecode, fused_presets
        self._streams = {}
        prev_post = None  # post_ffn_norm of the previous layer: the post norm of the next residual's b
        for layer in model.layers:
            p, i, op = f"model.layers.{layer.i}", layer.i, layer.op
            post_op, post_ffn = nw(f"{p}.post_operator_norm"), nw(f"{p}.post_ffn_norm")
            op_norm, ffn_norm = nw(f"{p}.operator_norm"), nw(f"{p}.ffn_norm")
            qkv_ok = layer.attn and op.fused_gate
            n_qkv = op.qkv_width + GATE_PAD if qkv_ok else None  # logical qkvg width (the stream pads to 104 tiles)
            # name: (preset name, cache key, resident tensor, w3 key, logical out width)
            if layer.attn:
                ws = {"qkv": ("attn_qkv", f"L{i}.attn.qkvg", op.wqkv, None, n_qkv)} if qkv_ok else {}
                ws["wo"] = ("attn_out", f"L{i}.attn.o_proj", op.wo, None, None)
            else:
                ws = {"in": ("conv_in", f"L{i}.conv.in_proj", op.in_proj, None, None),
                      "out": ("conv_out", f"L{i}.conv.out_proj", op.out_proj, None, None)}
            ws["w2"] = ("mlp_w2", f"L{i}.mlp.w2", layer.ffn.w2, None, None)
            ws["w13"] = ("mlp_w13", f"L{i}.mlp.w1", layer.ffn.w1, f"L{i}.mlp.w3", None)
            if "mm" in parts:
                for name, (pname, key, like, key3, n_out) in ws.items():
                    w, preset = stream(key, pname, like, key3)
                    self.mm[(i, name)] = StreamDecodeMatmul(dev, w, preset, CK_BFP8, n_out=n_out)
            taps = replicated_taps(op) if parts & {"conv", "convf"} and not layer.attn else None
            if "conv" in parts and not layer.attn:
                for w in self.widths:
                    self.conv[w][i] = ConvDecode(op, rows=w, taps=taps)
            if "convf" in parts and not layer.attn:
                w_in, _ = stream(f"L{i}.conv.in_proj", "conv_in", op.in_proj)
                w_out, _ = stream(f"L{i}.conv.out_proj", "conv_out", op.out_proj)
                kw = dict(norm=(op_norm, cfg.norm_eps), post=prev_post, placement=(0, 6)) if "norm" in parts else {}
                self._convf_kw[i] = kw
                for w in self.widths:
                    self.convf[w][i] = ConvOpDecode(op, w_in, w_out, rows=w, taps=taps, **kw)
            if "afront" in parts and qkv_ok:
                self.afront[i] = AttnQKVFront(op, model.rope[layer.type], cache, i,
                                              weight=stream(f"L{i}.attn.qkvg", "attn_qkv", op.wqkv)[0])
            if "anfront" in parts and "norm" in parts and qkv_ok:
                self.anfront[i] = AttnNormQKVFront(op, model.rope[layer.type], op_norm, cfg.norm_eps, post=prev_post,
                                                   weight=stream(f"L{i}.attn.qkvg", "attn_qkv", op.wqkv)[0])
            if "attn" in parts and layer.attn:
                self.attn[i] = AttnDecode(op, model.rope[layer.type])
            if "norm" in parts:
                kw = dict(h_memory_config=L1, n_memory_config=L1)
                self.op_rn[i] = ResidualRMSNorm(dev, op_norm, cfg.norm_eps, post_weight=prev_post, **kw)
                self.ffn_rn[i] = ResidualRMSNorm(dev, ffn_norm, cfg.norm_eps, post_weight=post_op, **kw)
            if "nmm" in parts:
                specs = [("w13", ffn_norm, post_op)]
                if qkv_ok:
                    specs.append(("qkv", op_norm, prev_post))
                elif not layer.attn:
                    specs.append(("in", op_norm, prev_post))
                for name, norm_w, post_w in specs:
                    pname, key, like, key3, n_out = ws[name]
                    w, preset = stream(key, pname, like, key3)
                    self.nmm[(i, name)] = NormStreamDecodeMatmul(
                        dev, norm_w, cfg.norm_eps, w, preset, CK_BFP8,
                        output_memory_config=L1 if name == "w13" else DRAM, post_weight=post_w, n_out=n_out)
            if "mlpf" in parts and "norm" in parts:
                p13, p2 = fused_presets(layer.ffn.w1.dtype)
                w13, _ = stream(f"L{i}.mlp.w1", "mlp_w13", layer.ffn.w1, f"L{i}.mlp.w3", preset=p13)
                w2, _ = stream(f"L{i}.mlp.w2", "mlp_w2", layer.ffn.w2, preset=p2)
                self.mlpf[i] = FusedMLPDecode(dev, ffn_norm, cfg.norm_eps, w13, p13, w2, p2, CK_BFP8,
                                              post_weight=post_op)
            prev_post = post_ffn
        if "norm" in parts:
            self.final_rn = ResidualRMSNorm(dev, nw("model.embedding_norm"), cfg.norm_eps, post_weight=prev_post,
                                            h_memory_config=L1, n_memory_config=L1)
        if "mm" in parts and model.lm_head is not None:
            w, preset = stream("lm_head", "lm_head", model.lm_head)
            self.mm[(-1, "head")] = StreamDecodeMatmul(dev, w, preset, CK_HIFI4, out_dtype=model.logits_dtype)
        # speculative verification (enable_verify): candidate Bx rows per conv layer, the row map, verify-mode modules
        # per width
        self.cand, self.conv_v, self.convf_v, self.anfront_v, self.verify_widths = {}, {}, {}, {}, []
        self.vmap = None
        # tt/drafter.py: every step copies the residual rows after the drafter's tap layers into the drafter's tap
        # buffers of its width (enable_drafter); the target's values are only read
        self.drafter = None
        self._tap_in, self._tap_final = {}, None

    def enable_drafter(self, drafter) -> None:
        """Every decode / verify step (traces captured afterwards) also copies its taps -- the residual rows after
        layers drafter.taps (the input of the next layer's operator, the final norm's for the last layer) -- into
        drafter.tap_bufs[width] for the drafter's commit (Drafter.commit, its own trace).  The target path is unchanged
        (the copies only read its tensors)."""
        last = self.m.layers[-1].i
        self._tap_in, self._tap_final = {}, None
        for k, t in enumerate(drafter.taps):
            if t == last:
                self._tap_final = k
            else:
                self._tap_in[t + 1] = k
        self.drafter = drafter
        if self.vmap is not None:
            drafter.bind_verify(self.vmap)

    def _tap(self, B, k, x):
        ttnn.copy(x, self.drafter.tap_bufs[B][k])

    def enable_verify(self, widths) -> None:
        """Build the verify-step modules (decode_forward(..., verify=True)) for these row counts (2..16): the rows are
        groups of consecutive tokens, one per sequence, described by the row map self.vmap ([1, 16] uint32 ROW_MAJOR
        DRAM, staged per step: tt/speculate.py map_word).  The verify step runs exactly the module sequence of the
        width-1 plain step (layer 0: nmm.embed + ConvDecode + out_proj stream; conv layers ConvOpDecode; attention
        layers AttnNormQKVFront + SDPADecode + o_proj; FusedMLPDecode; final norm + head), in verify mode where a
        module carries state: conv layers read each group's slot state plus the group's earlier rows and write every
        row's Bx to self.cand[layer] (one [1, 1, 32, 2048] BF16 tensor per conv layer) instead of updating the state
        (tt/speculate.py SpecCommit commits each group's accepted rows); one front worker per 32-row KV cache tile of
        a group patches all of the group's rows in it.  The other modules are batch-invariant (each row its own
        position and page-table row), so row j's logits are bitwise its sequence's sequential 1-row step's.  Shares
        every weight."""
        from .attn_fused import AttnNormQKVFront, SDPADecode
        from .conv_decode import ConvDecode
        from .conv_fused import ConvOpDecode

        m, dev = self.m, self.m.device
        l0 = m.layers[0]
        plain = all((self._anfront_ok(layer, 1) if layer.attn else
                     (j == 0 and self._conv_ok(layer, 1) and (layer.i, "in") in self.nmm)
                     or (j > 0 and self._convf_ok(layer, 1))) and layer.i in self.mlpf
                    for j, layer in enumerate(m.layers))
        if l0.attn or not plain or 1 not in self.asdpa or "norm" not in self.parts or (-1, "head") not in self.mm:
            raise ValueError("FastStep.enable_verify needs the default parts (layer 0 nmm.embed + conv, convf, "
                             "anfront, asdpa, mlpf, norm, mm)")
        if self.vmap is None:
            self.vmap = ttnn.from_torch(torch.zeros(1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM)
            if self.drafter is not None:
                self.drafter.bind_verify(self.vmap)
        for layer in m.layers:
            i = layer.i
            if layer.attn:
                if i not in self.anfront_v:
                    self.anfront_v[i] = AttnNormQKVFront(layer.op, None, verify=True, share=self.anfront[i],
                                                         vmap=self.vmap)
            elif i not in self.cand:
                self.cand[i] = ttnn.zeros([1, 1, 32, m.cfg.hidden], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                          device=dev, memory_config=DRAM)
        for w in widths:
            if not 2 <= w <= MAX_ROWS:
                raise ValueError(f"FastStep.enable_verify: {w} rows (2..{MAX_ROWS})")
            if w in self.verify_widths:
                continue
            if w not in self.asdpa:
                self.asdpa[w] = SDPADecode(dev, rows=w)
            conv, convf = self.conv_v.setdefault(w, {}), self.convf_v.setdefault(w, {})
            for j, layer in enumerate(m.layers):
                if layer.attn:
                    continue
                i = layer.i
                if j == 0:
                    conv[i] = ConvDecode(layer.op, rows=w, taps=self.conv[1][i].taps, verify=self.cand[i],
                                         vmap=self.vmap)
                else:
                    ref = self.convf[1][i]
                    convf[i] = ConvOpDecode(layer.op, ref.w_in, ref.w_out, rows=w, taps=ref.taps,
                                            verify=self.cand[i], vmap=self.vmap, **self._convf_kw[i])
            self.verify_widths.append(w)

    def view(self, parts) -> "FastStep":
        """Same modules restricted to `parts` (a subset of the constructed ones): ablations without rebuilding."""
        parts = set(parts)
        if parts - self.parts:
            raise ValueError(f"view: parts {sorted(parts - self.parts)} were not constructed")
        v = copy.copy(self)
        v.parts = parts
        empty = {w: {} for w in self.widths}
        v.mm = self.mm if "mm" in parts else {}
        v.nmm = self.nmm if {"nmm", "norm"} <= parts else {}
        v.conv = self.conv if "conv" in parts else empty
        v.convf = self.convf if "convf" in parts else empty
        v.attn = self.attn if "attn" in parts else {}
        v.afront = self.afront if "afront" in parts else {}
        v.anfront = self.anfront if {"anfront", "norm"} <= parts else {}
        v.asdpa = self.asdpa if "asdpa" in parts else {}
        v.mlpf = self.mlpf if {"mlpf", "norm"} <= parts else {}
        return v

    # ------------------------------------------------------------------ pieces
    def _lin(self, i, name, x, w, **kw):
        mm = self.mm.get((i, name))
        return mm(x, **kw) if mm is not None else linear(x, w)

    def _conv_ok(self, layer, B):
        return layer.i in self.conv.get(B, ())

    def _convf_ok(self, layer, B):
        return layer.i in self.convf.get(B, ()) and ("norm" in self.parts) == self.convf_norm

    def _afront_ok(self, layer, B):
        return layer.i in self.afront and B <= self.afront[layer.i].max_rows

    def _anfront_ok(self, layer, B):
        return layer.i in self.anfront and B <= self.anfront[layer.i].max_rows

    def _attn_ok(self, layer, B):
        return (layer.i in self.attn and B <= self.attn[layer.i].max_rows) or self._afront_ok(layer, B)

    def _op(self, layer, n, ctx, first=None, front=None):
        """Operator on the normed input n; `first`: its first projection (qkvg / conv in_proj) already computed by a
        fused norm + matmul, or `front`: the attention's (SDPA query, gate) from AttnNormQKVFront (n unused then)."""
        op, B = layer.op, ctx["B"]
        if layer.attn:
            if front is None and not self._attn_ok(layer, B):
                cos, sin = ctx["cos_sin"][layer.type]
                return op.decode(n, cos, sin, ctx["rope"][layer.type], ctx["cur_pos"], ctx["page_table"])
            if front is not None:
                qd, gate = front
            elif first is None and self._afront_ok(layer, B):
                qd, gate = self.afront[layer.i](n, ctx["cur_pos"], ctx["page_table"])
            else:
                qkv = first if first is not None else self._lin(layer.i, "qkv", n, op.wqkv)
                if op.fused_gate:
                    qd, gate = self.attn[layer.i](qkv, ctx["cur_pos"], ctx["page_table"])
                else:  # separate gate_proj (format differs from q/k/v): its logits go in as a kwarg
                    g = linear(n, op.wg)
                    qd, gate = self.attn[layer.i](qkv, ctx["cur_pos"], ctx["page_table"], gate=g)
                    ttnn.deallocate(g)
                ttnn.deallocate(qkv)
            sdpa = self.asdpa.get(B)
            if sdpa is not None:
                flat = sdpa(qd, op.k_cache, op.v_cache, ctx["cur_pos"], ctx["page_table"], gate=gate,
                            window=op.window)
            else:
                from .attn_fused import stock_sdpa_gated

                flat = stock_sdpa_gated(op, qd, gate, ctx["cur_pos"], ctx["page_table"])
            free(qd, gate)
            y = self._lin(layer.i, "wo", flat, op.wo)
            ttnn.deallocate(flat)
            return y
        if first is None and self._convf_ok(layer, B):
            return self.convf[B][layer.i](n)
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

    def _mlp(self, layer, n):
        if (layer.i, "w13") in self.mm:
            h = self.mm[(layer.i, "w13")](n, output_memory_config=L1)
            y = self.mm[(layer.i, "w2")](h)
            ttnn.deallocate(h)
            return y
        return layer.ffn(n)

    def _fused_op_norm(self, layer, B):
        """NormStreamDecodeMatmul of this layer's sandwich residual + operator norm + first projection, if it
        applies."""
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
        m = self.m
        B = B_rows(tok)
        if verify:
            if B not in self.verify_widths:
                raise ValueError(f"FastStep: no verify modules for {B} rows (enable_verify)")
            return self._forward_verify(tok, {"rope": m.rope, "cur_pos": cur_pos, "page_table": page_table, "B": B})
        if B not in self.widths:
            raise ValueError(f"FastStep: decode width {B} not in {self.widths}")
        layers = m.layers
        need_rope = {layer.type for layer in layers
                     if layer.attn and not (self._attn_ok(layer, B) or self._anfront_ok(layer, B))}
        cos_sin = {t: m.rope[t].decode(rope_idx) for t in sorted(need_rope)}
        ctx = {"cos_sin": cos_sin, "rope": m.rope, "cur_pos": cur_pos, "page_table": page_table, "B": B}
        try:
            if "norm" not in self.parts:
                return self._forward_stock_norms(tok, ctx)
            return self._forward_fused_norms(tok, ctx)
        finally:
            for cos, sin in cos_sin.values():
                free(cos, sin)

    def _embed(self, tok, B):
        e = ttnn.embedding(tok, self.m.embed, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
        return ttnn.reshape(e, [1, 1, B, self.m.cfg.hidden])

    def _forward_stock_norms(self, tok, ctx):
        """Stock norms / residual adds (layer.* helpers) around the fast operators and MLP."""
        m, B = self.m, ctx["B"]
        x = self._embed(tok, B)
        for layer in m.layers:
            n = layer.op_norm(x)
            o = self._op(layer, n, ctx)
            ttnn.deallocate(n)
            h = layer._residual(x, o, layer.post_op_norm)
            n2 = layer.ffn_norm(h)
            f = self._mlp(layer, n2)
            ttnn.deallocate(n2)
            x = layer._residual(h, f, layer.post_ffn_norm)
        n = m.final_norm(x)
        ttnn.deallocate(x)
        logits = self._head(n)
        ttnn.deallocate(n)
        return logits

    def _forward_verify(self, tok, ctx):
        """The verify step: _forward_fused_norms' width-1 module sequence with the verify-mode conv / attention-front
        modules (enable_verify)."""
        m, B = self.m, ctx["B"]
        a = b = None
        tapping = self.drafter is not None
        for j, layer in enumerate(m.layers):
            i = layer.i
            if j == 0:
                xr, first = self.nmm[(i, "in")].embed(tok, m.embed)
                y = self.conv_v[B][i](first)
                ttnn.deallocate(first)
                o = self._lin(i, "out", y, layer.op.out_proj)
                ttnn.deallocate(y)
            elif layer.attn:
                xr, qd, gate = self.anfront_v[i](a, b, ctx["cur_pos"], ctx["page_table"])
                o = self._op(layer, None, ctx, front=(qd, gate))
            else:
                xr, o = self.convf_v[B][i](a, b)
            if b is not None:
                free(a, b)
            if tapping and i in self._tap_in:
                self._tap(B, self._tap_in[i], xr)
            h, f = self.mlpf[i](xr, o)
            free(xr, o)
            a, b = h, f
        hf, n = self.final_rn(a, b)
        free(a, b)
        if tapping and self._tap_final is not None:
            self._tap(B, self._tap_final, hf)
        logits = self._head(n)
        ttnn.deallocate(n)
        return logits

    def _forward_fused_norms(self, tok, ctx):
        m, B = self.m, ctx["B"]
        layers = m.layers
        # layer 0: the embedding gather folded into the norm + first projection whenever nmm applies there (also
        # where convf would take the layer: saves the embedding program + reshape)
        l0 = layers[0]
        ok0 = (l0.i in self.attn and self._attn_ok(l0, B)) if l0.attn else self._conv_ok(l0, B)
        fused0 = self.nmm.get((l0.i, "qkv" if l0.attn else "in")) if ok0 else None
        # pending residual (a, b): the next norm consumes a + post(b) (b None: a alone, the embedding)
        a, b = (None, None) if fused0 is not None else (self._embed(tok, B), None)
        tapping = self.drafter is not None
        for j, layer in enumerate(layers):
            if j == 0 and fused0 is not None:
                xr, first = fused0.embed(tok, m.embed)
                o = self._op(layer, None, ctx, first=first)
            elif layer.attn and self._anfront_ok(layer, B):
                xr, qd, gate = self.anfront[layer.i](a, b, ctx["cur_pos"], ctx["page_table"])
                o = self._op(layer, None, ctx, front=(qd, gate))
            elif not layer.attn and self._convf_ok(layer, B):
                xr, o = self.convf[B][layer.i](a, b)
            else:
                fused = self._fused_op_norm(layer, B)
                if fused is not None:
                    xr, first = fused(a, b)
                    o = self._op(layer, None, ctx, first=first)
                else:
                    xr, n = self.op_rn[layer.i](a, b)
                    o = self._op(layer, n, ctx)
                    ttnn.deallocate(n)
            if b is not None:
                free(a, b)
            if tapping and layer.i in self._tap_in:
                self._tap(B, self._tap_in[layer.i], xr)
            w13 = self.nmm.get((layer.i, "w13"))
            if layer.i in self.mlpf:
                h, f = self.mlpf[layer.i](xr, o)
            elif w13 is not None and (layer.i, "w2") in self.mm:
                h, hm = w13(xr, o)
                f = self.mm[(layer.i, "w2")](hm)
                ttnn.deallocate(hm)
            else:
                h, n2 = self.ffn_rn[layer.i](xr, o)
                f = self._mlp(layer, n2)
                ttnn.deallocate(n2)
            free(xr, o)
            a, b = h, f
        hf, n = self.final_rn(a, b)
        free(a, b)
        if tapping and self._tap_final is not None:
            self._tap(B, self._tap_final, hf)
        logits = self._head(n)
        ttnn.deallocate(n)
        return logits
