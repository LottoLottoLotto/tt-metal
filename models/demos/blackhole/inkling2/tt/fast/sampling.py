# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Exact on-device vLLM sampling of Inkling's FP32 softcapped logits (decode: one row; speculative verify: R rows;
batched decode: R <= 16 rows of different requests, fast/batch_decode.py: params page b = row b's words(slot=its
history slot), position word b of the batch tables' cur_pos, pads / unpenalized rows with F_NOCOMMIT).

Ported from qwen36 `tt/device_sampling.py` (BF16 logits; MTP draft / verify dropped): kernels/fast/sampling.cpp +
kernels/fast/sampling_math.hpp (qwen36 device_sampling_math.hpp + the integer full-vocabulary helpers).  Processing
follows vLLM's v1 Sampler for one request row:

    float32 logits -> repetition, frequency, presence penalties (output-token counts and the prompt mask live on
    device) -> temperature (logits / T) -> min_p -> top-k (ties at the k-th value kept) -> top-p (ascending cumulative
    probability <= 1 - p masked, the largest always kept) -> softmax -> draw.

Top-K rows (greedy, or 1 <= top_k <= 64): two launches, scan (220 workers keep the exact top-(top_k + 8) candidates
of the penalized logits by (value desc, id asc); columns >= `valid` = the unembedding padding are skipped) and row
(merge, sampler, Philox4x32-10 draw keyed by the request seed with counter (absolute position of the drawn token,
purpose)).  A top-k tie group that might continue beyond the candidates is flagged R_INEXACT (not committed): the
host redraws that token with `host_sample` on the retained logits.

Full-vocabulary rows (temperature > 0 with top_k unset or > 64; `RowSampling.full`): scan (+ the penalized row to
`plog`, top-72 candidates), row (merge: maximum + candidates), weights (every token's exp((l - max) / T) as an exact
Q18.46 integer, min_p applied; per-tile and per-stripe sums), draw (inverse CDF of u * Z over the integer sums in
stripe order; top-p decided on the candidates when they reach the nucleus edge, else rejection sampling), and for
top-p / top-k > 64 rows tail + final (rejection: up to 4 independent full-distribution draws, the first inside the
nucleus / top-k set wins: exact; all 4 outside -> R_INEXACT, the host redraws with `host_sample(purpose=P_HOST)`).
Weights below 2^-46 of the maximum (> 31.9 T nats) are zero: a total mass below 3e-9 is dropped, under the float32
rounding of vLLM's own softmax.  `host_sample_full` is the bit-exact host model of this path (tests).

The token goes to word 0 of `tok` (the decode trace's embedding index row: the next step's input) and [token, flags,
n, 0] to page `row` of `sel`; the token is added to the device history (penalties of the next steps).  Speculative
verify (`verify`): R rows at positions p .. p + R - 1, row r counting the drafts d_1..d_r in its penalties (words 0
of the draft log's rows), no commit per row; then `accept` (one core): k = longest prefix with sampled(row i) ==
d_{i+1}, history += d_1..d_k + the row-k token, and the R sampled tokens -> the verify ids row.  Sampling every row
and accepting while the sample equals the draft IS speculative rejection sampling for the drafter's point-mass
proposal q = delta(d) (accept d with probability min(1, p(d) / q(d)) = p(d); on rejection the sample is distributed
as the residual max(0, p - q) / (1 - p(d)) = p without d), and since row i draws with the same counter (position)
as plain decoding, a seeded spec request emits exactly the tokens of the plain request.
Host-only (`device_eligible` False): logprobs, logit bias, allowed / bad words, structured output, logits processors,
an unsatisfied min_tokens, temperature > 8, and 64 < top_k < vocabulary combined with top_p < 1.
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

from .token_ops import CachedProgram

K = 72
MAX_TOP_K = 64
MAX_FULL_TEMPERATURE = 8.0  # full-vocabulary rows: weights vanish exactly where max - l >= 256 (T <= 8)
SAMPLING_EPS = 1e-5  # vLLM: temperature below this is greedy
_KERNEL = str(Path(__file__).resolve().parent.parent / "kernels" / "fast" / "sampling.cpp")
_HIST_COLS = 32
_PARAM_ROWS = 32
_RS_WORDS = 256       # rowstate page: 64 state words + K (value, id) candidate pairs
_ND = 4               # rejection draws

F_ACTIVE, F_GREEDY, F_PENALTY, F_REP, F_EXTRA, F_NOCOMMIT, F_MONO = 1, 2, 4, 8, 16, 32, 64
F_FULL, F_TOPP, F_TOPK = 128, 256, 512
R_INEXACT = 1
P_SAMPLE, P_REJ, P_HOST = 1, 5, 9  # draw / rejection draws 2..4 (P_REJ + i) / host redraw after 4 rejections
_MASK32 = 0xFFFFFFFF
W_ONE = 1 << 46


def _f32(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def philox4x32(counter, key):
    c = [int(v) & _MASK32 for v in counter]
    k0, k1 = (int(v) & _MASK32 for v in key)
    for _ in range(10):
        p0 = 0xD2511F53 * c[0]
        p1 = 0xCD9E8D57 * c[2]
        c = [((p1 >> 32) ^ c[1] ^ k0) & _MASK32, p1 & _MASK32, ((p0 >> 32) ^ c[3] ^ k1) & _MASK32, p0 & _MASK32]
        k0 = (k0 + 0x9E3779B9) & _MASK32
        k1 = (k1 + 0xBB67AE85) & _MASK32
    return c


def uniform_bits(seed, position, purpose=P_SAMPLE):
    """The device's 52 random bits: Philox4x32-10(counter=(position, purpose, 0, 0), key=seed)."""
    c = philox4x32((position, purpose, 0, 0), (seed & _MASK32, (seed >> 32) & _MASK32))
    return ((c[0] << 20) | (c[1] >> 12)) & ((1 << 52) - 1)


def uniform(seed, position, purpose=P_SAMPLE):
    """The device's uniform draw in [0, 1): uniform_bits / 2^52."""
    return uniform_bits(seed, position, purpose) / float(1 << 52)


@dataclass(frozen=True)
class RowSampling:
    """One request's sampling parameters (vLLM SamplingParams semantics; top_k <= 0 = unset) plus its resolved 64-bit
    seed."""

    temperature: float
    top_k: int
    top_p: float = 1.0
    min_p: float = 0.0
    presence: float = 0.0
    frequency: float = 0.0
    repetition: float = 1.0
    seed: int = 0

    @property
    def greedy(self):
        return self.temperature < SAMPLING_EPS

    @property
    def full(self):
        """Full-vocabulary device path (top_k unset or above the candidate cap)."""
        return not self.greedy and not 1 <= int(self.top_k) <= MAX_TOP_K

    @property
    def rejection(self):
        """Full-vocabulary row that may need the rejection launches (top-p, or top-k > 64)."""
        return self.full and (self.top_p < 1.0 or int(self.top_k) > MAX_TOP_K)

    @property
    def penalized(self):
        return self.presence != 0.0 or self.frequency != 0.0 or self.repetition != 1.0

    @property
    def uses_prompt(self):
        return self.repetition != 1.0

    @property
    def monotone(self):
        """Penalties never raise a logit (lets the scan skip history lookups below its threshold)."""
        return self.presence >= 0.0 and self.frequency >= 0.0 and self.repetition >= 1.0

    @property
    def candidates(self):
        """Device candidate count: the top-k plus an 8-entry margin that detects truncated tie groups."""
        if self.greedy:
            return 1
        return K if self.full else min(K, int(self.top_k) + 8)

    def tmin(self):
        """min_p threshold on the Q18.46 weights: vLLM masks probs < max_prob * min_p <=> w < min_p * 2^46."""
        return math.ceil(Fraction(float(np.float32(self.min_p))) * W_ONE) if self.min_p > 0 else 0

    def one_minus_p(self):
        """vLLM's float32 1 - p as (m, s): exactly m / 2^s."""
        f = Fraction(float(np.float32(1.0) - np.float32(self.top_p)))
        if f == 0:
            return 0, 0
        return f.numerator, f.denominator.bit_length() - 1

    def words(self, slot=0, nocommit=False, extra=0):
        """Params page of a row; extra: the row counts the first `extra` draft-log tokens in its penalties."""
        f32 = np.float32
        flags = F_ACTIVE
        flags |= F_GREEDY if self.greedy else 0
        flags |= F_PENALTY if self.penalized else 0
        flags |= F_REP if self.uses_prompt else 0
        flags |= F_EXTRA if extra and self.penalized else 0
        flags |= F_NOCOMMIT if nocommit else 0
        flags |= F_MONO if self.monotone else 0
        if self.full:
            flags |= F_FULL
            flags |= F_TOPP if self.top_p < 1.0 else 0
            flags |= F_TOPK if int(self.top_k) > MAX_TOP_K else 0
        rep = f32(self.repetition)
        top_p = f32(self.top_p)
        w = [0] * 32
        w[0] = flags
        w[1] = _f32(f32(self.temperature)) if not self.greedy else _f32(1.0)
        w[2] = int(self.top_k) if not self.greedy and int(self.top_k) > 0 else (1 if self.greedy else 0)
        w[3] = _f32(top_p)
        w[4] = _f32(f32(self.min_p))
        w[5] = _f32(f32(self.presence))
        w[6] = _f32(f32(self.frequency))
        w[7] = _f32(rep)
        w[8] = _f32(f32(1.0) / rep)  # torch: 1.0 / penalties in float32
        w[9] = self.seed & _MASK32
        w[10] = (self.seed >> 32) & _MASK32
        w[11] = int(slot)
        w[13] = _f32(f32(1.0) - top_p)  # vLLM: 1 - p in float32
        w[14] = self.candidates
        # log2(e) / T as unsigned Q32.32 for the integer exp2 of the candidate / vocabulary weights.
        c = self.c_q32()
        w[18] = c & _MASK32
        w[19] = (c >> 32) & _MASK32
        w[20] = int(extra)
        t = self.tmin()
        w[21], w[22] = t & _MASK32, (t >> 32) & _MASK32
        w[23], w[24] = self.one_minus_p() if self.full and self.top_p < 1.0 else (0, 0)
        return w

    def c_q32(self):
        if self.greedy:
            return 0
        return round((1.0 / (math.log(2.0) * float(np.float32(self.temperature)))) * (1 << 32))


def device_eligible(params, num_output_tokens=0, top_k_cap=MAX_TOP_K):
    """True when the device reproduces vLLM's sampler for this request exactly (else the host path runs).

    Host-only: logprobs / prompt logprobs, allowed_token_ids, bad_words, logit_bias, an unsatisfied min_tokens,
    structured outputs, logits processors, temperature > 8, and top_k above the candidate cap together with top_p < 1.
    (The vllm_tt_plugin picks this through Inkling2ForCausalLM.device_sampling_eligible.)"""
    if params.logprobs is not None or params.prompt_logprobs is not None:
        return False
    if params.allowed_token_ids or params.bad_words_token_ids or params.logit_bias:
        return False
    if params.min_tokens and num_output_tokens < params.min_tokens:
        return False
    if getattr(params, "structured_outputs", None) is not None or getattr(params, "guided_decoding", None) is not None:
        return False
    if getattr(params, "logits_processors", None):
        return False
    if params.temperature < SAMPLING_EPS:
        return True
    k = int(params.top_k)
    if 1 <= k <= top_k_cap:
        return True
    if params.temperature > MAX_FULL_TEMPERATURE:
        return False
    return k <= 0 or float(params.top_p) >= 1.0


def row_sampling(params, seed):
    top_k = int(params.top_k)
    return RowSampling(
        temperature=float(params.temperature),
        top_k=top_k,
        top_p=float(params.top_p) if top_k != 1 else 0.0,
        min_p=float(params.min_p),
        presence=float(params.presence_penalty),
        frequency=float(params.frequency_penalty),
        repetition=float(params.repetition_penalty),
        seed=int(seed) & ((1 << 64) - 1),
    )


def _penalized(logits, spec: RowSampling, output_ids, prompt_ids):
    from vllm.v1.sample.ops.penalties import apply_all_penalties

    x = logits.reshape(1, -1).float().clone()
    if spec.penalized:
        vocab = x.shape[-1]
        prompt = torch.tensor([list(prompt_ids) if spec.uses_prompt else []], dtype=torch.int64)
        if prompt.numel() == 0:
            prompt = torch.full((1, 1), vocab, dtype=torch.int64)
        x = apply_all_penalties(
            x, prompt, torch.tensor([spec.presence]), torch.tensor([spec.frequency]),
            torch.tensor([spec.repetition]), [list(output_ids)],
        )
    return x


def host_probs(logits, spec: RowSampling, output_ids=(), prompt_ids=()):
    """vLLM's sampler distribution (float64 probabilities over the row): penalties, temperature, min_p, top-k (None
    when unset), top-p (None at 1.0), softmax."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    x = _penalized(logits, spec, output_ids, prompt_ids)
    x = x / torch.tensor([[spec.temperature]], dtype=torch.float32)
    if spec.min_p > 0:
        probs = x.softmax(dim=-1)
        x = x.masked_fill(probs < probs.amax(dim=-1, keepdim=True) * spec.min_p, float("-inf"))
    k = torch.tensor([int(spec.top_k)]) if int(spec.top_k) > 0 else None
    p = torch.tensor([spec.top_p], dtype=torch.float32) if spec.top_p < 1.0 else None
    x = apply_top_k_top_p(x, k, p)
    return x.softmax(dim=-1, dtype=torch.float64).reshape(-1)


def host_sample(logits, spec: RowSampling, output_ids, prompt_ids, position, purpose=P_SAMPLE):
    """vLLM's float32 sampler on one host logit row ([valid] FP32), drawn with the device's counter-based uniform
    (position = the drawn token's absolute position) by inverse CDF in descending-probability order.  Used when the
    device reports R_INEXACT (top-K rows: a truncated tie group, purpose P_SAMPLE; full-vocabulary rows: 4 rejected
    draws, purpose P_HOST); the same distribution as the device path (and the reference of the top-K path's tests)."""
    if spec.greedy:
        return int(_penalized(logits, spec, output_ids, prompt_ids).argmax(dim=-1))
    probs = host_probs(logits, spec, output_ids, prompt_ids)
    support = torch.nonzero(probs > 0).reshape(-1)
    order = support[torch.argsort(-probs[support], stable=True)]
    cum = torch.cumsum(probs[order], 0)
    u = uniform(spec.seed, position, purpose) * float(cum[-1])
    index = int(torch.searchsorted(cum, torch.tensor([u], dtype=torch.float64), right=True))
    return int(order[min(index, order.numel() - 1)])


# ---------------------------------------------------------------------------------------------- full-vocabulary model
_EXP2_TABLE = np.array([
    0x80000000, 0x7e9f0606, 0x7d41d96e, 0x7be86fba, 0x7a92be8b, 0x7940bb9e, 0x77f25cce, 0x76a7980f,
    0x75606374, 0x741cb528, 0x72dc8374, 0x719fc4b9, 0x70666f76, 0x6f307a41, 0x6dfddbcc, 0x6cce8ae1,
    0x6ba27e65, 0x6a79ad56, 0x69540ec9, 0x683199ed, 0x6712460b, 0x65f60a7f, 0x64dcdec3, 0x63c6ba64,
    0x62b39509, 0x61a3666d, 0x60962665, 0x5f8bccdb, 0x5e8451d0, 0x5d7fad59, 0x5c7dd7a4, 0x5b7ec8f2,
    0x5a82799a, 0x5988e209, 0x5891fac1, 0x579dbc57, 0x56ac1f75, 0x55bd1cdb, 0x54d0ad5a, 0x53e6c9da,
    0x52ff6b55, 0x521a8ad7, 0x51382182, 0x50582888, 0x4f7a9930, 0x4e9f6cd4, 0x4dc69cdd, 0x4cf022ca,
    0x4c1bf829, 0x4b4a169c, 0x4a7a77d4, 0x49ad1598, 0x48e1e9ba, 0x4818ee22, 0x47521cc6, 0x468d6fae,
    0x45cae0f2, 0x450a6abb, 0x444c0740, 0x438fb0cb, 0x42d561b4, 0x421d1462, 0x4166c34c, 0x40b268fa,
], dtype=np.uint64)


def fixq24(v: np.ndarray) -> np.ndarray:
    """sampling_math.hpp fixq24: float32 values -> trunc(v * 2^24) as int64."""
    b = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32).astype(np.int64)
    e = (b >> 23) & 0xFF
    m = (b & 0x7FFFFF) | 0x800000
    sh = e - 126
    left = m << np.clip(sh, 0, 63)
    right = m >> np.clip(-sh, 0, 63)
    out = np.where(sh >= 0, left, right)
    out = np.where(e < 103, 0, out)
    return np.where(b >> 31 != 0, -out, out)


def wfix46(mq: int, lq: np.ndarray, c: int) -> np.ndarray:
    """sampling_math.hpp wfix46 (vectorized): Q18.46 weights of fixq24 logits below the maximum mq."""
    d = mq - lq
    out = np.zeros(d.shape, dtype=np.uint64)
    one = d <= 0
    live = (d > 0) & (d < (256 << 24))
    dq = d[live].astype(np.uint64)
    y = dq * np.uint64(c >> 32) + ((dq * np.uint64(c & _MASK32)) >> np.uint64(32))
    ok = y < np.uint64(46 << 24)
    y = y[ok]
    n = y >> np.uint64(24)
    frac = y & np.uint64(0xFFFFFF)
    k = (frac >> np.uint64(18)).astype(np.int64)
    rq = (frac & np.uint64(0x3FFFF)) << np.uint64(6)
    a = (rq * np.uint64(0x58B90BFC)) >> np.uint64(30)
    a2 = (a * a) >> np.uint64(31)
    a3 = (a2 * a) >> np.uint64(31)
    a4 = (a3 * a) >> np.uint64(31)
    poly = (np.uint64(0x80000000) - a + (a2 >> np.uint64(1)) - a3 // np.uint64(6) + a4 // np.uint64(24)) & np.uint64(_MASK32)
    mant = (_EXP2_TABLE[k] * poly) >> np.uint64(31)
    w = (mant << np.uint64(15)) >> n
    vals = np.zeros(dq.shape, dtype=np.uint64)
    vals[ok] = w
    out[live] = vals
    out[one] = np.uint64(W_ONE)
    return out


def _fkey(v: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32).copy()
    b[b == 0x80000000] = 0
    return b ^ (((b.astype(np.int32) >> 31).astype(np.uint32)) | np.uint32(0x80000000))


def stripe_order(valid: int, tiles: int, workers: int) -> np.ndarray:
    """Token ids in the device's walk order: stripes (worker w: tiles w, w + NW, ...) in worker order."""
    t = np.concatenate([np.arange(w, tiles, workers) for w in range(workers)])
    ids = (t[:, None] * 32 + np.arange(32)[None, :]).reshape(-1)
    return ids[ids < valid]


def host_sample_full(logits, spec: RowSampling, output_ids, prompt_ids, position, workers: int, tiles: int):
    """Bit-exact host model of the device's full-vocabulary path -> (token, inexact)."""
    v = _penalized(logits, spec, output_ids, prompt_ids).reshape(-1).numpy().astype(np.float32)
    V = v.shape[0]
    key = _fkey(v)
    ids = np.arange(V)
    order = np.lexsort((ids, -key.astype(np.int64)))  # key desc, id asc
    cand = order[:K]
    mq = int(fixq24(v[cand[:1]])[0])
    c = spec.c_q32()
    w = wfix46(mq, fixq24(v), c)
    tmin = spec.tmin()
    if tmin:
        w[w < np.uint64(tmin)] = 0
    Z = int(w.sum(dtype=np.uint64))
    perm = stripe_order(V, tiles, workers)
    cum = np.cumsum(w[perm], dtype=np.uint64)

    def walk(target):
        return int(perm[int(np.searchsorted(cum, np.uint64(target), side="right"))])

    if spec.top_p < 1.0:
        m, s = spec.one_minus_p()
        Lz = (Z * m) >> s
        C = np.cumsum(w[cand].astype(object))
        hit = np.nonzero(C >= Z - Lz)[0]
        if hit.size:
            kept = int(hit[0]) + 1
            target = (int(C[kept - 1]) * uniform_bits(spec.seed, position, P_SAMPLE)) >> 52
            cc = np.cumsum(w[cand[:kept]].astype(object))
            return int(cand[next(i for i in range(kept) if cc[i] > target)]), False
    else:
        Lz = 0
    topk = int(spec.top_k) > MAX_TOP_K
    if spec.top_p >= 1.0 and not topk:
        return walk((Z * uniform_bits(spec.seed, position, P_SAMPLE)) >> 52), False
    cpos = {int(t): i for i, t in enumerate(cand)}
    for i in range(_ND):
        purpose = P_SAMPLE if i == 0 else P_REJ + i - 1
        x = walk((Z * uniform_bits(spec.seed, position, purpose)) >> 52)
        if x in cpos:
            greater = int((key[cand] > key[x]).sum())
            ok = not topk or greater < int(spec.top_k)
        else:
            after = (key < key[x]) | ((key == key[x]) & (ids >= x))
            tail = int(w[after].sum(dtype=np.uint64))
            greater = int((key > key[x]).sum())
            ok = (spec.top_p >= 1.0 or tail > Lz) and (not topk or greater < int(spec.top_k))
        if ok:
            return x, False
    return None, True


# ---------------------------------------------------------------------------------------------- device sampler
class DeviceSampler:
    """Fixed-address device buffers and the sampling kernels.  width = logits width (tile aligned), valid = sampled
    columns (the unpadded vocabulary), num_slots = penalty history slots (batched: one per sequence slot), rows = the
    most rows of one call (speculative verify; batched decode: 16 = SAMPLE_ROWS <= 16 reads face rows 0..15 only)."""

    def __init__(self, device, width: int, valid: int, num_slots: int = 1, cores: int | None = None, rows: int = 4):
        if width % 32:
            raise ValueError("device sampling requires a tile-aligned logit width")
        grid = device.compute_with_storage_grid_size()
        count = min(grid.x * grid.y, int(cores or os.environ.get("INK_SAMPLING_CORES", "110")))
        self.device = device
        self.vocab = valid
        self.vocab_tiles = width // 32
        self.num_slots = num_slots
        self.max_rows = rows
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
        self.workers = 2 * count
        nw = self.workers
        self.stripe = -(-self.vocab_tiles // nw)
        self.tsum_page = -(-self.stripe * 8 // 64) * 64
        u32 = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def zeros(r, cols):
            return ttnn.from_torch(torch.zeros(r, cols, dtype=torch.int32), **u32)

        self.hist = [zeros(self.vocab_tiles, _HIST_COLS) for _ in range(num_slots)]
        self.params = zeros(_PARAM_ROWS, 32)
        self.partial = zeros(16 * nw, 2 * K)
        self.heads = zeros(16, 16 * nw)
        self.plog = zeros(rows * self.vocab_tiles, 32)
        self.rowstate = zeros(rows, _RS_WORDS)
        self.wsum = zeros(rows, 16 * nw)
        self.tsum = zeros(rows * nw, self.tsum_page // 4)
        self.tailpart = zeros(rows, 16 * nw)
        self._params_host = None
        self._hist_state = [None] * num_slots  # slot -> (owner, outputs incorporated, prompt flags present)
        self._progs = {}

    # ---------------------------------------------------------------- host state
    def set_params(self, rows):
        """rows: list of 32-word param rows (row r -> params page r); inactive rows are zero."""
        table = np.zeros((_PARAM_ROWS, 32), dtype=np.uint32)
        for r, words in enumerate(rows):
            if words is not None:
                table[r] = np.asarray(words, dtype=np.uint64).astype(np.uint32)
        if self._params_host is not None and np.array_equal(table, self._params_host):
            return
        host = ttnn.from_torch(torch.from_numpy(table.view(np.int32)), dtype=ttnn.uint32,
                               layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.params)
        self._params_host = table

    def ensure_history(self, slot, owner, output_ids, prompt_ids, use_prompt):
        """Make history slot `slot` hold exactly `output_ids` counts (+ prompt flags when repetition is on)."""
        state = (owner, len(output_ids), bool(use_prompt))
        if self._hist_state[slot] == state:
            return False
        width = self.vocab_tiles * 32
        counts = np.bincount(np.asarray(output_ids, dtype=np.int64), minlength=width)[:width]
        table = counts.astype(np.uint32)
        if use_prompt and len(prompt_ids):
            ids = np.asarray(prompt_ids, dtype=np.int64)
            mask = np.zeros(width, dtype=bool)
            mask[ids[ids < width]] = True  # ids past the logit width can never be sampled
            table |= mask.astype(np.uint32) << np.uint32(31)
        host = ttnn.from_torch(torch.from_numpy(table.view(np.int32).reshape(self.vocab_tiles, _HIST_COLS)),
                               dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.hist[slot])
        self._hist_state[slot] = state
        return True

    def note_committed(self, slot, owner, total_outputs):
        """The device appended tokens to slot's history; it now holds `total_outputs` outputs of `owner`."""
        state = self._hist_state[slot]
        if state is not None and state[0] == owner:
            self._hist_state[slot] = (owner, int(total_outputs), state[2])

    def invalidate(self, slot=None):
        for s in range(self.num_slots) if slot is None else (slot,):
            self._hist_state[s] = None

    # ---------------------------------------------------------------- kernels
    def _run(self, mode, rows, cores, defines, worker_args, cb_bytes, logits, sel, tok, pos, aux, spec=False):
        """Launch kernel mode `mode`: worker_args -> per-core (brisc, ncrisc) args of every core of the rectangle
        (the first len(cores) cores row-major)."""
        aux = aux if aux is not None else self.params
        tensors = [logits, self.params, self.partial, self.heads, sel, tok, pos, aux, self.plog, self.rowstate,
                   self.wsum, self.tsum, self.tailpart] + self.hist
        common = [int(logits.buffer_address()), int(self.heads.buffer_address()), int(self.params.buffer_address()),
                  int(self.partial.buffer_address()), int(sel.buffer_address()), int(tok.buffer_address()),
                  int(pos.buffer_address()), int(aux.buffer_address()), int(self.plog.buffer_address()),
                  int(self.rowstate.buffer_address()), int(self.wsum.buffer_address()),
                  int(self.tsum.buffer_address()), int(self.tailpart.buffer_address())]
        common += [int(h.buffer_address()) for h in self.hist]
        sel_page = int(sel.padded_shape[-1]) * 4
        tok_page = int(tok.padded_shape[-1]) * 4

        def build():
            core_set = ttnn.num_cores_to_corerangeset(len(cores), self.device.compute_with_storage_grid_size(),
                                                      row_wise=True)
            kernel = UnifiedKernelDescriptor(
                kernel_source=_KERNEL,
                core_ranges=core_set,
                defines=[("SAMPLE_MODE", str(mode)), ("SAMPLE_K", str(K)), ("SAMPLE_WORKERS", str(self.workers)),
                         ("SAMPLE_TILES", str(self.vocab_tiles)), ("SAMPLE_VALID", str(self.vocab)),
                         ("SAMPLE_ROWS", str(rows)), ("SAMPLE_SEL_PAGE", str(sel_page)),
                         ("SAMPLE_TOK_PAGE", str(tok_page)), ("SAMPLE_SPEC", str(int(spec)))]
                + [(k, str(v)) for k, v in defines],
                brisc_common_runtime_args=common,
                ncrisc_common_runtime_args=common,
                per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                    brisc_args=[(c, [a[0]]) for c, a in zip(cores, worker_args)],
                    ncrisc_args=[(c, [a[1]]) for c, a in zip(cores, worker_args)]),
            )
            cb = ttnn.CBDescriptor(
                total_size=cb_bytes, core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048,
                                                            ttnn.TileDescriptor(ttnn.Tile((32, 32))))],
            )
            return ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=[cb], semaphores=[])

        prog = self._progs.setdefault((mode, rows, spec), CachedProgram())
        prog.run(tensors, build)

    @staticmethod
    def _cb(nbytes):
        return ((nbytes + 2047) // 2048) * 2048

    def _workers(self):
        n = len(self.cores)
        return self.cores, [(i, n + i) for i in range(n)]

    def scan(self, logits, sel, tok, pos, rows=1, aux=None, spec=False):
        """Mode 0: penalized per-worker top-K of logit rows 0..rows-1 (FP32 TILE [1, 1, rows, width], DRAM) +
        F_FULL rows' penalized logits -> plog."""
        if (logits.dtype != ttnn.float32 or logits.layout != ttnn.TILE_LAYOUT
                or int(logits.padded_shape[-1]) != self.vocab_tiles * 32 or int(logits.padded_shape[-2]) != 32
                or logits.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError("device sampling needs DRAM FP32 TILE logits of one tile row")
        if not 1 <= rows <= self.max_rows:
            raise ValueError(f"rows must be in 1..{self.max_rows}")
        batch = 8 if rows <= 2 else 4
        half = 128 * rows + 8 * rows * K + batch * 2 * rows * 64 + batch * rows * 128 + 8 * 128
        half = ((half + 63) // 64) * 64
        cores, args = self._workers()
        self._run(0, rows, cores, [("SAMPLE_BATCH", batch), ("SAMPLE_SCRATCH_HALF", half)], args,
                  self._cb(2 * half), logits, sel, tok, pos, aux, spec)

    def row(self, logits, sel, tok, pos, rows=1, spec=False):
        """Mode 1: merge per row (one core each); top-K rows: sampler + emit; F_FULL rows: max + candidates."""
        scratch = 128 + 64 + 256 + self.workers * 64 + 64 + 1024 + K * 16 + 64
        self._run(1, rows, self.cores[:rows], [("SAMPLE_BATCH", 1), ("SAMPLE_SCRATCH_HALF", 0)],
                  [(r, r) for r in range(rows)], self._cb(scratch), logits, sel, tok, pos, None, spec)

    def _stripe_mode(self, mode, logits, sel, tok, pos, rows, spec):
        batch = 32
        half = 128 + 256 + 64 + self.tsum_page + batch * 128
        half = ((half + 63) // 64) * 64
        cores, args = self._workers()
        self._run(mode, rows, cores, [("SAMPLE_BATCH", batch), ("SAMPLE_SCRATCH_HALF", half)], args,
                  self._cb(2 * half), logits, sel, tok, pos, None, spec)

    def _row_mode(self, mode, logits, sel, tok, pos, rows, spec):
        scratch = 128 + 64 + 256 + 1024 + self.tsum_page + 128 + self.workers * 64
        self._run(mode, rows, self.cores[:rows], [("SAMPLE_BATCH", 1), ("SAMPLE_SCRATCH_HALF", 0)],
                  [(r, r) for r in range(rows)], self._cb(scratch), logits, sel, tok, pos, None, spec)

    def sample(self, logits, sel, tok, pos, *, rows=1, kind="sample", aux=None, spec=False):
        """kind: "sample" (top-K rows), "full" (full-vocabulary rows without top-p / top-k), "full_reject" (+ the
        rejection launches); rows: logit rows (row r draws position pos word r + 1)."""
        self.scan(logits, sel, tok, pos, rows, aux, spec)
        self.row(logits, sel, tok, pos, rows, spec)
        if kind in ("full", "full_reject"):
            self._stripe_mode(2, logits, sel, tok, pos, rows, spec)
            self._row_mode(3, logits, sel, tok, pos, rows, spec)
            if kind == "full_reject":
                self._stripe_mode(4, logits, sel, tok, pos, rows, spec)
                self._row_mode(5, logits, sel, tok, pos, rows, spec)

    def accept(self, logits, sel, ids, pos, draft_log, rows):
        """Mode 6 (speculative verify of `rows` rows, sel pages 0..rows-1): history commit of the accepted prefix +
        the row tokens -> ids words 0..rows-1."""
        scratch = 128 + rows * 64 + rows * 128 + 64 + 128
        self._run(6, rows, self.cores[:1], [("SAMPLE_BATCH", 1), ("SAMPLE_SCRATCH_HALF", 0)], [(0, 0)],
                  self._cb(scratch), logits, sel, ids, pos, draft_log, True)

    def __call__(self, logits, sel, tok, pos, kind="sample"):
        """Sample row 0 of logits: token -> tok word 0, [token, flags, n, 0] -> sel words 0..3.  pos: int32 device
        tensor whose word 0 is the decode input position (the drawn token's position - 1)."""
        self.sample(logits, sel, tok, pos, kind=kind)

    def release(self):
        for t in [*self.hist, self.params, self.partial, self.heads, self.plog, self.rowstate, self.wsum, self.tsum,
                  self.tailpart]:
            ttnn.deallocate(t)


def kind_of(spec: RowSampling | None) -> str:
    """Decode tail kind of a request: greedy (device argmax) | sample (top-K rows) | full | full_reject."""
    if spec is None or (spec.greedy and not spec.penalized):
        return "greedy"
    if not spec.full:
        return "sample"
    return "full_reject" if spec.rejection else "full"
