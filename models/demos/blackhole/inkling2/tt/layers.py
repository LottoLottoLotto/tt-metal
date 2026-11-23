# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini glue modules on the device (everything but attention and the routed experts).

Activations are [1, 1, T, 2304] TILE tensors in DRAM: T = prefill chunk rows, or 32 in decode (row 0 = the token).
Precision follows the reference (modeling_inkling2_loop.py): the residual stream between layers is BF16, everything
inside a layer that the reference runs in FP32 (block / message attention residuals, message gates, short convs, the
in-layer residual) is FP32 here.

  RMSNorm           weighted RMSNorm (input/q/k/v_input_layernorm, post_attention_layernorm, embed_norm, norm)
  unorm             unweighted RMSNorm (Inkling2LoopUnweightedRMSNorm), FP32 out
  BlockResidual     Inkling2LoopBlockAttentionResidual: softmax over [block states..., partial] with 1 or 3 queries;
                    block states are kept NORMED (FP32), the only form any consumer reads
  MessageResidual   Inkling2LoopMessageAttentionResidual: per pass, 48 queries (12 loop layers x q/k/v/mlp) over the
                    previous pass's [block state of layer 12, final hidden]
  MessageGate       Inkling2LoopMessageGate: (1 - tanh(F g)) * x + tanh(G g) * m, g = down(cat(norm x, norm m))
  ShortConv         Inkling2LoopShortConvolution (post-attention conv idx 2 / post-MLP conv idx 3): depthwise causal
                    kernel-4 conv + residual, / sqrt(5); one conv state (last 3 inputs) per pass slot
  Embedding / LMHead
"""
from __future__ import annotations

import math

import torch
import ttnn

from .common import CK_HIFI4, CK_NORM, DRAM, free, to_dev

F32 = ttnn.float32
BF16 = ttnn.bfloat16


def norm_weight(device, w: torch.Tensor):
    """RMSNorm gamma in the [1, 1, dim/32, 32] row-major form ttnn.rms_norm takes (BF16, as stored)."""
    return ttnn.from_torch(w.to(torch.bfloat16).reshape(1, 1, -1, 32), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=device, memory_config=DRAM)


def as_f32(x):
    """FP32 copy of x (a new tensor even when x already is FP32)."""
    if x.dtype == F32:
        return ttnn.clone(x, memory_config=DRAM)
    return ttnn.typecast(x, F32, memory_config=DRAM)


class RMSNorm:
    """weight * x * rsqrt(mean(x^2) + eps), computed in FP32; output BF16 (the reference casts to the stream dtype)."""

    def __init__(self, device, w: torch.Tensor, eps: float):
        self.w, self.eps = norm_weight(device, w), eps

    def __call__(self, x, dtype=BF16):
        y = ttnn.rms_norm(x, weight=self.w, epsilon=self.eps, compute_kernel_config=CK_NORM, memory_config=DRAM)
        if y.dtype != dtype:
            z = ttnn.typecast(y, dtype, memory_config=DRAM)
            ttnn.deallocate(y)
            y = z
        return y


def unorm(x, eps: float = 1e-6, dtype=F32):
    """Unweighted RMSNorm of x (BF16 or FP32) -> dtype; computed from FP32 unless both are BF16 (the decode path:
    one op, the norm's internal accumulation is FP32 either way)."""
    if x.dtype == BF16 and dtype == BF16:
        return ttnn.rms_norm(x, epsilon=eps, compute_kernel_config=CK_NORM, memory_config=DRAM)
    xf = x if x.dtype == F32 else ttnn.typecast(x, F32, memory_config=DRAM)
    y = ttnn.rms_norm(xf, epsilon=eps, compute_kernel_config=CK_NORM, memory_config=DRAM)
    if xf is not x:
        ttnn.deallocate(xf)
    if dtype != F32:
        z = ttnn.typecast(y, dtype, memory_config=DRAM)
        ttnn.deallocate(y)
        y = z
    return y


def _col(t, c: int):
    """Column c of a [1, 1, T, W] tile tensor as [1, 1, T, 1] (broadcasts along the width)."""
    T = t.shape[-2]
    return ttnn.slice(t, [0, 0, 0, c], [1, 1, T, c + 1])


def _softmax_list(logits: list):
    """Softmax across a list of same-shape FP32 tensors (elementwise over positions), returns the weights list."""
    m = logits[0]
    for lg in logits[1:]:
        m2 = ttnn.maximum(m, lg, memory_config=DRAM)
        if m is not logits[0]:
            ttnn.deallocate(m)
        m = m2
    es = []
    for lg in logits:
        d = ttnn.sub(lg, m, dtype=F32, memory_config=DRAM)
        es.append(ttnn.exp(d, memory_config=DRAM))
        ttnn.deallocate(d)
    if m is not logits[0]:
        ttnn.deallocate(m)
    z = es[0]
    for e in es[1:]:
        z2 = ttnn.add(z, e, dtype=F32, memory_config=DRAM)
        if z is not es[0]:
            ttnn.deallocate(z)
        z = z2
    inv = ttnn.reciprocal(z, memory_config=DRAM)
    if z is not es[0]:
        ttnn.deallocate(z)
    ws = [ttnn.mul(e, inv, dtype=F32, memory_config=DRAM) for e in es]
    free(inv, *es)
    return ws


class RowConsts:
    """Decode-mode constants for row-stacked states ([1, 1, 32, H], row i = state i, row 0 of every activation is the
    token): row-move matrices R(src, dst) (one-hot [32, 32], R[dst, src] = 1) and softmax column masks."""

    def __init__(self, device):
        self.device = device
        self._r, self._m = {}, {}

    def move(self, src: int, dst: int):
        key = (src, dst)
        if key not in self._r:
            r = torch.zeros(1, 1, 32, 32)
            r[..., dst, src] = 1
            self._r[key] = to_dev(r, self.device, BF16)
        return self._r[key]

    def mask(self, n_valid: int):
        """Softmax column mask row [1, 1, 1, 32] (FP32): 0 for the first n_valid columns, -1e4 after."""
        if n_valid not in self._m:
            m = torch.zeros(1, 1, 1, 32)
            m[..., n_valid:] = -1e4
            self._m[n_valid] = to_dev(m, self.device, F32)
        return self._m[n_valid]


def _rows_mix(qrows, X, n_valid: int, consts: RowConsts, dtype=BF16):
    """softmax_i(q . X_i) X over the first n_valid rows of X, per query row of qrows ([1,1,rows,H] BF16, pre-scaled);
    X BF16 -> dtype [1,1,rows,H].  Logits (mask as the matmul bias) and softmax in FP32; the matmuls take BF16 / FP32
    operands at BF16-level precision."""
    L = ttnn.linear(qrows, X, transpose_b=True, bias=consts.mask(n_valid), dtype=F32, compute_kernel_config=CK_HIFI4,
                    memory_config=DRAM)
    W = ttnn.softmax(L, dim=-1, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
    ttnn.deallocate(L)
    O = ttnn.matmul(W, X, dtype=dtype, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
    ttnn.deallocate(W)
    return O


class BlockResidual:
    """Inkling2LoopBlockAttentionResidual: out_q = sum_i softmax_i(q . s_i / sqrt(H)) s_i over the normed stacked
    states s = [block states..., norm(partial)] (FP32).

    __call__ (prefill / any T): states as a list of [1,1,T,H] tensors, elementwise FP32 mixing; the queries are the
    columns of one [H, 32] FP32 matrix (1 or 3 used), pre-scaled by H^-0.5.
    rows (decode, row 0 = the token): states stacked as the rows of one [1,1,32,H] tensor X; per query
    softmax(q X^T) X with two small matmuls (BF16-level precision of the mixing weights)."""

    def __init__(self, device, q: torch.Tensor):
        q = q.float().reshape(-1, q.shape[-1])
        self.nq, H = q.shape
        qt = torch.zeros(H, 32)
        qt[:, : self.nq] = q.T * H ** -0.5
        self.qt = to_dev(qt.reshape(1, 1, *qt.shape), device, F32)
        self.qrows = []
        for i in range(self.nq):
            r = torch.zeros(1, 1, 32, H)
            r[..., 0, :] = q[i] * H ** -0.5
            self.qrows.append(to_dev(r, device, BF16))

    def __call__(self, states: list, partial_n) -> list:
        """states: normed block states (FP32 [1,1,T,H]); partial_n: the normed partial state -> nq FP32 outputs."""
        allst = list(states) + [partial_n]
        logits = [ttnn.matmul(s, self.qt, dtype=F32, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
                  for s in allst]
        ws = _softmax_list(logits)
        free(*logits)
        outs = []
        for q in range(self.nq):
            acc = None
            for s, w in zip(allst, ws):
                wc = _col(w, q)
                term = ttnn.mul(s, wc, dtype=F32, memory_config=DRAM)
                ttnn.deallocate(wc)
                if acc is None:
                    acc = term
                else:
                    ttnn.add(acc, term, dtype=F32, memory_config=DRAM, output_tensor=acc)
                    ttnn.deallocate(term)
            outs.append(acc)
        free(*ws)
        return outs

    def rows(self, X, n_valid: int, consts: RowConsts) -> list:
        """X: [1,1,32,H] stacked normed states (rows 0..n_valid-1) -> nq outputs [1,1,32,H] FP32 (row 0)."""
        return [_rows_mix(qr, X, n_valid, consts) for qr in self.qrows]


class ListStates:
    """Block states for any T: a list of normed FP32 [1,1,T,H] tensors (owned)."""

    def __init__(self, items=None):
        self.items = list(items or [])

    @property
    def n(self):
        return len(self.items)

    def snapshot(self):
        return ListStates(self.items)

    def append(self, p):
        """Takes ownership of p."""
        self.items.append(p)

    def mix(self, res: BlockResidual, p, keep: bool = False):
        """res over [states..., p]; keep: p becomes a block state (ownership taken), else p is freed."""
        outs = res(self.items, p)
        if keep:
            self.items.append(p)
        else:
            ttnn.deallocate(p)
        return outs

    def state(self, i: int):
        return self.items[i]

    def free_from(self, k: int):
        free(*self.items[k:])
        del self.items[k:]


class RowStates:
    """Decode block states: rows 0..n-1 of X [1,1,32,H] BF16 (rows >= n zero); every X built is kept in the shared
    `owned` list until free_all (a decode step creates a handful)."""

    def __init__(self, consts: RowConsts, X=None, n: int = 0, owned=None):
        self.c, self.X, self.n = consts, X, n
        self.owned = [] if owned is None else owned

    def snapshot(self):
        return RowStates(self.c, self.X, self.n, self.owned)

    def _stack(self, p):
        """X with p's row 0 as row n (a new tensor); frees p."""
        s = ttnn.matmul(self.c.move(0, self.n), p, dtype=BF16, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        ttnn.deallocate(p)
        if self.X is not None:
            ttnn.add(s, self.X, output_tensor=s)
        return s

    def append(self, p):
        self.X = self._stack(p)
        self.n += 1
        self.owned.append(self.X)

    def mix(self, res: BlockResidual, p, keep: bool = False):
        X = self._stack(p)
        outs = res.rows(X, self.n + 1, self.c)
        if keep:
            self.X = X
            self.n += 1
            self.owned.append(X)
        else:
            ttnn.deallocate(X)
        return outs

    def free_from(self, k: int):
        pass  # the X tensors are freed together (free_all)

    def free_all(self):
        free(*self.owned)
        self.owned.clear()


class MessageResidual:
    """Inkling2LoopMessageAttentionResidual for one pass: states [s0, s1] = [normed block state appended by the previous
    pass at layer 12, norm(previous pass's final hidden)]; the message of (loop layer j, kind k) is
    w0 s0 + (1 - w0) s1 = s1 + w0 (s0 - s1), w0 = sigmoid(l0 - l1) = 1 / (1 + exp(l1 - l0)).
    Query column 4 j + k (k = q, k, v, mlp), pre-scaled by H^-0.5; FP32.

    Decode (rows): Xm = [s0; s1] stacked as rows, M = softmax(Q Xm^T) Xm for all 48 queries at once ([1,1,64,H]);
    message(j, k) = row 4 j + k of M as a [1,1,1,H] tensor (it broadcasts over the token tile's rows)."""

    KINDS = ("q", "k", "v", "mlp")

    def __init__(self, device, queries: torch.Tensor, kv_queries: torch.Tensor):
        # checkpoint: queries rows (2j, 2j+1) = (q, mlp) of loop layer j; kv_queries rows (2j, 2j+1) = (k, v)
        q = queries.float().unflatten(0, (-1, 2))
        kv = kv_queries.float().unflatten(0, (-1, 2))
        merged = torch.cat([q[:, :1], kv, q[:, 1:]], dim=1)  # [n_loop, 4, H]: q, k, v, mlp
        self.n_loop, _, H = merged.shape
        cols = -(-4 * self.n_loop // 32) * 32
        qt = torch.zeros(H, cols)
        qt[:, : 4 * self.n_loop] = merged.reshape(-1, H).T * H ** -0.5
        self.qt = to_dev(qt.reshape(1, 1, *qt.shape), device, F32)
        self.qrows = to_dev(qt.T.reshape(1, 1, cols, H).contiguous(), device, BF16)

    def prepare_rows(self, X_prev, row: int, s1, consts: RowConsts):
        """Decode: s0 = row `row` of the previous pass's stacked block states X_prev, s1 = normed previous hidden
        (freed) -> M [1,1,cols,H] BF16 (row 4 j + k = message (j, k))."""
        Xm = ttnn.matmul(consts.move(row, 0), X_prev, dtype=BF16, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        b = ttnn.matmul(consts.move(0, 1), s1, dtype=BF16, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        ttnn.deallocate(s1)
        ttnn.add(Xm, b, output_tensor=Xm)
        ttnn.deallocate(b)
        M = _rows_mix(self.qrows, Xm, 2, consts)
        ttnn.deallocate(Xm)
        return M

    def message_row(self, M, j: int, k: int):
        c = 4 * j + k
        return ttnn.slice(M, [0, 0, c, 0], [1, 1, c + 1, M.shape[-1]], memory_config=DRAM)

    def prepare(self, s0, s1):
        """-> (w0 [1,1,T,cols] FP32, d = s0 - s1, s1); the caller owns s1 (kept alive) and frees w0 / d."""
        l0 = ttnn.matmul(s0, self.qt, dtype=F32, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        l1 = ttnn.matmul(s1, self.qt, dtype=F32, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        diff = ttnn.sub(l1, l0, dtype=F32, memory_config=DRAM)
        free(l0, l1)
        e = ttnn.exp(diff, memory_config=DRAM)
        ttnn.deallocate(diff)
        ttnn.add(e, 1.0, output_tensor=e)
        w0 = ttnn.reciprocal(e, memory_config=DRAM)
        ttnn.deallocate(e)
        d = ttnn.sub(s0, s1, dtype=F32, memory_config=DRAM)
        return w0, d

    @staticmethod
    def message(w0, d, s1, j: int, k: int):
        wc = _col(w0, 4 * j + k)
        t = ttnn.mul(d, wc, dtype=F32, memory_config=DRAM)
        ttnn.deallocate(wc)
        ttnn.add(t, s1, dtype=F32, memory_config=DRAM, output_tensor=t)
        return t


class MessageGate:
    """Inkling2LoopMessageGate: g = bf16(down(cat(norm x, norm m))) [T, 64]; out = (1 - tanh(g F^T)) x + tanh(g G^T) m
    (FP32; the reference keeps forget/gain in FP32, its weights are BF16 values)."""

    def __init__(self, device, down: torch.Tensor, forget: torch.Tensor, gain: torch.Tensor):
        H = forget.shape[0]
        w4 = lambda t: to_dev(t.reshape(1, 1, *t.shape), device, BF16)  # noqa: E731
        self.wdx = w4(down[:, :H].T)        # [H, 64]
        self.wdm = w4(down[:, H:].T)        # [H, 64]
        self.wf = w4(forget.T)              # [64, H]
        self.wg = w4(gain.T)

    def __call__(self, x, m, nx=None, gm=None):
        """x, m FP32 [1,1,T,H] (m may be one row, broadcast); nx: bf16 unweighted norm of x if already computed (shared
        across q/k/v gates); gm: the message half of the down projection, norm(m) @ down[:, H:]^T, if precomputed
        (decode: Messages.gm, all 48 messages of a pass in one matmul)."""
        own_nx = nx is None
        if own_nx:
            nx = unorm(x, dtype=BF16)
        if gm is None:
            gx = ttnn.matmul(nx, self.wdx, dtype=F32, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
            nm = unorm(m, dtype=BF16)
            gmm = ttnn.matmul(nm, self.wdm, dtype=F32, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
            ttnn.deallocate(nm)
            ttnn.add(gx, gmm, dtype=F32, memory_config=DRAM, output_tensor=gx)
            ttnn.deallocate(gmm)
            g = ttnn.typecast(gx, BF16, memory_config=DRAM)
            ttnn.deallocate(gx)
        else:  # the precomputed message half as the bias row; BF16 out = the reference's bf16 down_proj output
            g = ttnn.linear(nx, self.wdx, bias=gm, dtype=BF16, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        if own_nx:
            ttnn.deallocate(nx)
        fz = ttnn.matmul(g, self.wf, dtype=F32, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        gz = ttnn.matmul(g, self.wg, dtype=F32, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        ttnn.deallocate(g)
        f = ttnn.tanh(fz, memory_config=DRAM)
        gn = ttnn.tanh(gz, memory_config=DRAM)
        free(fz, gz)
        fx = ttnn.mul(f, x, dtype=F32, memory_config=DRAM)
        out = ttnn.sub(x, fx, dtype=F32, memory_config=DRAM)
        gm2 = ttnn.mul(gn, m, dtype=F32, memory_config=DRAM)
        ttnn.add(out, gm2, dtype=F32, memory_config=DRAM, output_tensor=out)
        free(f, gn, fx, gm2)
        return out


class ShortConv:
    """Inkling2LoopShortConvolution on the hidden stream: y = (sum_j w_j x[t-3+j] + x[t]) / sqrt(5) in FP32, BF16 out.
    Taps are folded on the host: t_j = w_j / sqrt(5) (j < 3), t_3 = (w_3 + 1) / sqrt(5) (FP32 [1, 1, 1, H]).

    Conv state per pass slot p: s[p][j] = x[t-3+j] (j = 0..2), [1, 1, 32, H] FP32 TILE holding BF16 values (exact),
    the row in every row.  Decode (row 0 of a 32-row tile): y = typecast(x t_3 + sum_j addcmul(s_j t_j)), state shifted
    in place (trace-safe copies).  Prefill reads row 0 of the states as the chunk's left context and writes back the
    chunk's last three valid rows (gathered by a device row index)."""

    def __init__(self, device, w: torch.Tensor, passes: int, kernel: int = 4):
        H = w.shape[0]
        w = w.float().reshape(H, kernel)
        self.H, self.K = H, kernel
        scale = 1.0 / math.sqrt(kernel + 1)
        taps = [w[:, j] * scale for j in range(kernel)]
        taps[-1] = (w[:, -1] + 1.0) * scale
        self.taps = [to_dev(t.reshape(1, 1, 1, H), device, F32) for t in taps]  # prefill: broadcast over rows
        self.taps32 = [to_dev(t.reshape(1, 1, 1, H).expand(1, 1, 32, H), device, F32) for t in taps]  # decode
        self.s = [[to_dev(torch.zeros(1, 1, 32, H), device, F32) for _ in range(kernel - 1)] for _ in range(passes)]

    def reset(self):
        for st in self.s:
            for t in st:
                ttnn.mul(t, 0.0, output_tensor=t)

    @staticmethod
    def _sum(rows, taps):
        """rows: K tensors (x[t-3], x[t-2], x[t-1], x[t]) -> BF16 sum_j taps_j * rows_j (FP32 accumulation)."""
        acc = ttnn.mul(rows[-1], taps[-1], dtype=F32, memory_config=DRAM)
        for r, t in zip(rows[:-1], taps[:-1]):
            p = ttnn.mul(r, t, dtype=F32, memory_config=DRAM)
            ttnn.add(acc, p, dtype=F32, memory_config=DRAM, output_tensor=acc)
            ttnn.deallocate(p)
        y = ttnn.typecast(acc, BF16, memory_config=DRAM)
        ttnn.deallocate(acc)
        return y

    def decode(self, x, p: int):
        """x [1, 1, 32, H] BF16 (row 0 = the token) -> BF16 [1, 1, 32, H]."""
        st = self.s[p]
        xf = ttnn.typecast(x, F32, memory_config=DRAM)
        acc = ttnn.mul(xf, self.taps32[-1], dtype=F32, memory_config=DRAM)
        for s, t in zip(st, self.taps32[:-1]):
            nxt = ttnn.addcmul(acc, s, t, value=1.0, memory_config=DRAM)
            ttnn.deallocate(acc)
            acc = nxt
        y = ttnn.typecast(acc, BF16, memory_config=DRAM)
        ttnn.deallocate(acc)
        for j in range(self.K - 2):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(xf, st[-1])
        ttnn.deallocate(xf)
        return y

    @staticmethod
    def state_index(device, valid: int, kernel: int = 4):
        """Device row index [1, 32 (K-1)] uint32: 32 copies of valid + j for j = 0..K-2, the rows of [state; chunk]
        that become the new conv state tiles (a runtime operand: the prefill programs depend only on the chunk
        length)."""
        idx = torch.arange(valid, valid + kernel - 1, dtype=torch.int32).repeat_interleave(32).reshape(1, -1)
        return ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                               memory_config=DRAM)

    def prefill(self, x, valid: int, p: int, idx=None):
        """x [1, 1, Tp, H] BF16 (rows >= valid are padding); idx: state_index(valid) (built here when None)."""
        st = self.s[p]
        Tp = x.shape[-2]
        k1 = self.K - 1
        parts = []
        for s in st:
            sb = ttnn.typecast(s, BF16, memory_config=DRAM)
            u = ttnn.untilize(sb)
            parts.append(ttnn.slice(u, [0, 0, 0, 0], [1, 1, 1, self.H]))
            free(sb, u)
        parts.append(ttnn.untilize(x))
        cat = ttnn.concat(parts, dim=2, memory_config=DRAM)  # rows: x[-3], x[-2], x[-1], x[0], ..., x[Tp-1]
        free(*parts)
        rows = []
        for j in range(k1):
            sl = ttnn.slice(cat, [0, 0, j, 0], [1, 1, j + Tp, self.H])
            rows.append(ttnn.tilize(sl, memory_config=DRAM))
            ttnn.deallocate(sl)
        y = self._sum(rows + [x], self.taps)
        free(*rows)
        own = idx is None
        if own:
            idx = self.state_index(x.device(), valid, self.K)
        table = ttnn.reshape(cat, [Tp + k1, self.H])
        g = ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, 32 (K-1), H]
        g4 = ttnn.reshape(g, [1, 1, 32 * k1, self.H])
        for j in range(k1):  # state j = x[valid-3+j] in all 32 rows
            t = ttnn.slice(g4, [0, 0, 32 * j, 0], [1, 1, 32 * j + 32, self.H])
            tf = ttnn.typecast(t, F32, memory_config=DRAM)
            ttnn.copy(tf, st[j])
            ttnn.deallocate(tf)
            ttnn.deallocate(t)
        free(g, cat)
        if own:
            ttnn.deallocate(idx)
        return y


class Embedding:
    """NormedEmbedding: embed_norm(embed[ids]) -> BF16 [1, 1, T, H].  The table stays on the HOST (BF16, 0.93 GB;
    saves that much device DRAM): `rows(ids, T)` gathers the rows into a host TILE tensor (zero rows up to T), the
    caller uploads it (prefill: to_device; decode trace: copy_host_to_device_tensor into the persistent input), and
    `__call__` applies embed_norm on the device."""

    def __init__(self, device, table: torch.Tensor, norm_w: torch.Tensor, eps: float):
        self.H = table.shape[1]
        self.table = table.to(torch.bfloat16).contiguous()
        self.norm = RMSNorm(device, norm_w, eps)

    def rows(self, ids: torch.Tensor, T: int):
        ids = torch.as_tensor(ids, dtype=torch.int64).reshape(-1)
        x = torch.zeros(1, 1, T, self.H, dtype=torch.bfloat16)
        x[0, 0, : ids.numel()] = self.table[ids]
        return ttnn.from_torch(x, dtype=BF16, layout=ttnn.TILE_LAYOUT)

    def __call__(self, x_dev):
        """x_dev: [1, 1, T, H] BF16 TILE device tensor of embedding rows -> embed_norm(x) (a new tensor)."""
        return self.norm(x_dev)


class LMHead:
    """logits = softcap(x @ unembed^T) = 30 tanh(l / 30), FP32 [1, 1, R, V_pad] (V_pad = unpadded vocab rounded up to
    32; the padding columns get a -1e4 bias, so they read -30 after the softcap)."""

    def __init__(self, device, w_t, bias, softcap: float):
        self.w, self.bias, self.cap = w_t, bias, softcap

    def __call__(self, x):
        lg = ttnn.linear(x, self.w, bias=self.bias, dtype=F32, compute_kernel_config=CK_HIFI4, memory_config=DRAM)
        ttnn.mul(lg, 1.0 / self.cap, output_tensor=lg)
        th = ttnn.tanh(lg, memory_config=DRAM)
        ttnn.deallocate(lg)
        ttnn.mul(th, self.cap, output_tensor=th)
        return th
