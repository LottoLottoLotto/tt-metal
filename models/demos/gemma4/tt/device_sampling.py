# SPDX-License-Identifier: Apache-2.0
"""Exact on-device vLLM sampling and lossless speculative sampling for gemma-4 (single user, one P150).

Ported from the Qwen3.6-35B implementation (Sample35). Processing follows vLLM's v1 Sampler
(vllm/v1/sample/sampler.py) for one request row, on the bf16 softcapped logits the host sampler sees as float32
(gemma-4 applies final_logit_softcapping = 30 inside the model's LM head, before any sampler):

    float32 logits -> repetition, frequency, presence penalties (apply_all_penalties; output-token counts and the
    prompt mask live on device) -> temperature (logits / T) -> min_p -> top-k (ties at the k-th value kept) ->
    top-p (ascending cumulative probability <= 1 - p masked, the largest always kept) -> softmax -> draw.

The device keeps the exact global top-(top_k + 8) candidates (at most K = 72) of the penalized logits by (value desc,
id asc), so every request with 1 <= top_k <= 64 is processed exactly; a top-k tie group that might continue beyond
the candidates is reported (R_INEXACT) and that one token is drawn on the host from the retained logits. Only
token ids, flags and accept counts are read back.

Randomness is Philox4x32-10 keyed by the request seed, counter (absolute token position, purpose): a seeded
request reproduces its output; unseeded requests draw a random 64-bit base seed.

Speculative sampling (drafter chain of k <= K drafts): draft x_i is drawn from the processed draft distribution q_i
(the same settings on the drafter's logits; its penalties count x_1..x_{i-1}), or q_i is the point mass on the
processed draft argmax. Verification walks i = 1..k: accept x_i with probability min(1, p_{i-1}(x_i)/q_i(x_i))
(p_{i-1}'s penalties count x_1..x_{i-1}); at the first rejection emit a draw from normalize(max(0, p_{i-1} - q_i))
and stop; after k acceptances emit the bonus from p_k. Each emitted token is distributed as non-speculative
sampling at its position given the emitted prefix. The draft count k may depend on anything known before the
verification (the confidence gate stops on the drafter's max q): that only decides at which position verification
falls back to a plain draw, so the output distribution is unchanged.
"""

import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

K = 72
MAX_TOP_K = 64
MAX_DRAFTS = 8
SAMPLING_EPS = 1e-5  # vLLM: temperature below this is greedy
_KERNEL = str(Path(__file__).with_name("device_sampling_kernel.cpp").resolve())
_HIST_COLS = 32
_PARAM_ROWS = 32
_ROW_WORDS = 256
DRAFT_PARAM_BASE = 16  # params pages of draft step i: DRAFT_PARAM_BASE + i - 1 (verify / decode rows: 0..15)

F_ACTIVE, F_GREEDY, F_PENALTY, F_REP, F_EXTRA, F_NOCOMMIT, F_MONO = 1, 2, 4, 8, 16, 32, 64
R_INEXACT = 1
V_HOST, V_QMISS = 1, 2  # verify flags: the host draws the position after the accepted drafts; q mismatch
P_SAMPLE, P_ACCEPT, P_RESIDUAL, P_DRAFT = 1, 2, 3, 4
_MASK32 = 0xFFFFFFFF


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


def uniform(seed, position, purpose):
    """The device's uniform draw in [0, 1): 52 bits of Philox4x32-10(counter=(position, purpose, 0, 0))."""
    c = philox4x32((position, purpose, 0, 0), (seed & _MASK32, (seed >> 32) & _MASK32))
    mant = ((c[0] << 20) | (c[1] >> 12)) & ((1 << 52) - 1)
    return mant / float(1 << 52)


@dataclass(frozen=True)
class RowSampling:
    """One request's sampling parameters (vLLM SamplingParams semantics) plus its resolved 64-bit seed."""

    temperature: float
    top_k: int
    top_p: float
    min_p: float
    presence: float
    frequency: float
    repetition: float
    seed: int

    @property
    def greedy(self):
        return self.temperature < SAMPLING_EPS

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
        return 1 if self.greedy else min(K, int(self.top_k) + 8)

    def words(self, slot, position=0, extras=0, nocommit=False):
        """Params page: ``extras`` = number of draft tokens in front of this row that its penalties also count."""
        if not 0 <= extras <= MAX_DRAFTS:
            raise ValueError(f"{extras} extra tokens (at most {MAX_DRAFTS})")
        f32 = np.float32
        flags = F_ACTIVE
        flags |= F_GREEDY if self.greedy else 0
        flags |= F_PENALTY if self.penalized else 0
        flags |= F_REP if self.uses_prompt else 0
        flags |= F_EXTRA if extras else 0
        flags |= F_NOCOMMIT if nocommit else 0
        flags |= F_MONO if self.monotone else 0
        rep = f32(self.repetition)
        top_p = f32(self.top_p)
        w = [0] * 32
        w[0] = flags
        w[1] = _f32(f32(self.temperature)) if not self.greedy else _f32(1.0)
        w[2] = int(self.top_k) if not self.greedy else 1
        w[3] = _f32(top_p)
        w[4] = _f32(f32(self.min_p))
        w[5] = _f32(f32(self.presence))
        w[6] = _f32(f32(self.frequency))
        w[7] = _f32(rep)
        w[8] = _f32(f32(1.0) / rep)  # torch: 1.0 / penalties in float32
        w[9] = self.seed & _MASK32
        w[10] = (self.seed >> 32) & _MASK32
        w[11] = int(slot)
        w[12] = int(position) & _MASK32
        w[13] = _f32(f32(1.0) - top_p)  # vLLM: 1 - p in float32
        w[14] = self.candidates
        # log2(e) / T as unsigned Q32.32 for the integer exp2 of the candidate weights.
        c = round((1.0 / (math.log(2.0) * float(f32(self.temperature)))) * (1 << 32)) if not self.greedy else 0
        w[18] = c & _MASK32
        w[19] = (c >> 32) & _MASK32
        w[20] = int(extras)
        return w


@dataclass
class SamplingRequest:
    """A request routed to the device sampler: identity, parameters, live output ids, prompt ids (repetition)."""

    owner: object
    spec: RowSampling
    outputs: list
    prompt: object = None


def device_eligible(params, num_output_tokens=0, top_k_cap=MAX_TOP_K):
    """True when the device reproduces vLLM's sampler for this request exactly (else the host path runs).

    Host-only: logprobs / prompt logprobs, allowed_token_ids, bad_words, logit_bias, an unsatisfied min_tokens,
    and random sampling without a top_k in [1, 64] (the full-vocabulary nucleus is not on device)."""
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
    return 1 <= int(params.top_k) <= top_k_cap


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


def host_sample(logits, spec, output_ids, prompt_ids, position, purpose=P_SAMPLE):
    """vLLM's float32 sampler on one host logit row, drawn with the device's counter-based uniform.

    Used when the device reports a tie group it cannot see completely; identical semantics to the host path."""
    from vllm.v1.sample.ops.penalties import apply_all_penalties
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    x = logits.reshape(1, -1).float().clone()
    vocab = x.shape[-1]
    if spec.penalized:
        prompt = torch.tensor([list(prompt_ids) if spec.uses_prompt else []], dtype=torch.int64)
        if prompt.numel() == 0:
            prompt = torch.full((1, 1), vocab, dtype=torch.int64)
        x = apply_all_penalties(
            x, prompt, torch.tensor([spec.presence]), torch.tensor([spec.frequency]),
            torch.tensor([spec.repetition]), [list(output_ids)],
        )
    if spec.greedy:
        return int(x.argmax(dim=-1))
    x = x / torch.tensor([[spec.temperature]], dtype=torch.float32)
    if spec.min_p > 0:
        probs = x.softmax(dim=-1)
        x = x.masked_fill(probs < probs.amax(dim=-1, keepdim=True) * spec.min_p, float("-inf"))
    x = apply_top_k_top_p(x, torch.tensor([spec.top_k]), torch.tensor([spec.top_p], dtype=torch.float32))
    probs = x.softmax(dim=-1, dtype=torch.float64).reshape(-1)
    support = torch.nonzero(probs > 0).reshape(-1)
    order = support[torch.argsort(-probs[support], stable=True)]
    cum = torch.cumsum(probs[order], 0)
    u = uniform(spec.seed, position, purpose) * float(cum[-1])
    index = int(torch.searchsorted(cum, torch.tensor([u], dtype=torch.float64), right=True))
    return int(order[min(index, order.numel() - 1)])


def _dram(t):
    """1 for a DRAM tensor, 0 for L1; the kernels address every buffer as interleaved pages."""
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"device sampling needs interleaved buffers, got {t.memory_config()}")
    return int(t.memory_config().buffer_type == ttnn.BufferType.DRAM)


class DeviceSampler:
    """Fixed-address device buffers and the three native kernels (one history slot: single user). Allocate before
    any trace capture; every kernel launch only refers to these buffers and to trace-owned tensors by address."""

    def __init__(self, device, vocab_size, max_rows=6, max_drafts=5, cores=None):
        if vocab_size % 32:
            raise ValueError("device sampling requires a tile-aligned vocabulary")
        if not 1 <= max_rows <= 16 or not 0 <= max_drafts <= MAX_DRAFTS:
            raise ValueError(f"unsupported sampler geometry rows={max_rows} drafts={max_drafts}")
        grid = device.compute_with_storage_grid_size()
        count = min(grid.x * grid.y, int(cores or os.environ.get("GEMMA_SAMPLING_CORES", "110")))
        self.device = device
        self.vocab = vocab_size
        self.vocab_tiles = vocab_size // 32
        self.num_slots = 1
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
        self.workers = 2 * count
        u32 = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def zeros(rows, cols):
            return ttnn.from_torch(torch.zeros(rows, cols, dtype=torch.int32), **u32)

        self.hist = [zeros(self.vocab_tiles, _HIST_COLS)]
        self.params = zeros(_PARAM_ROWS, 32)
        self.partial = zeros(16 * self.workers, 2 * K)
        self.heads = zeros(16, 16 * self.workers)
        self.rowbuf = zeros(max_rows, _ROW_WORDS)
        self.qbuf = zeros(max(max_drafts, 1), _ROW_WORDS)
        self.out = zeros(16, 4)  # 16-byte records: decode row (op SAMPLE) / draft step i at row i-1 (op DRAFT)
        self.cyc = zeros(1, 32)  # verify record: m, emitted, flags, 0, tokens...
        self.dlist = zeros(1, 16)  # word i = draft x_i (the extras of the later draft rows)
        self.dummy = zeros(1, 16)
        self.max_rows = max_rows
        self.max_drafts = max_drafts
        self._params_host = None
        self._hist_state = None  # (owner, outputs incorporated, prompt flags present)

    # ---------------------------------------------------------------- host state
    def set_request(self, spec):
        """Params of every row for one request: page j = verify row j (counts drafts x_1..x_j; page 0 is also the
        decode row), page DRAFT_PARAM_BASE + i - 1 = draft step i (counts x_1..x_{i-1})."""
        pages = {j: spec.words(0, extras=j) for j in range(self.max_rows)}
        for i in range(1, self.max_drafts + 1):
            pages[DRAFT_PARAM_BASE + i - 1] = spec.words(0, extras=i - 1)
        self.set_params(pages)

    def set_params(self, pages):
        """pages: {page index: 32 words}; every other page is inactive (zero)."""
        table = np.zeros((_PARAM_ROWS, 32), dtype=np.uint32)
        for r, words in pages.items():
            if words is not None:
                table[r] = np.asarray(words, dtype=np.uint64).astype(np.uint32)
        if self._params_host is not None and np.array_equal(table, self._params_host):
            return
        host = ttnn.from_torch(torch.from_numpy(table.view(np.int32)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.params)
        self._params_host = table

    def ensure_history(self, owner, output_ids, prompt_ids, use_prompt):
        """Make the history hold exactly `output_ids` counts (+ prompt flags when repetition is on)."""
        state = (owner, len(output_ids), bool(use_prompt))
        if self._hist_state == state:
            return False
        counts = np.bincount(np.asarray(output_ids, dtype=np.int64), minlength=self.vocab)[: self.vocab]
        table = counts.astype(np.uint32)
        if use_prompt and len(prompt_ids):
            mask = np.zeros(self.vocab, dtype=bool)
            mask[np.asarray(prompt_ids, dtype=np.int64)] = True
            table |= mask.astype(np.uint32) << np.uint32(31)
        host = ttnn.from_torch(
            torch.from_numpy(table.view(np.int32).reshape(self.vocab_tiles, _HIST_COLS)),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host, self.hist[0])
        self._hist_state = state
        return True

    def note_committed(self, owner, total_outputs):
        """The device appended tokens to the history; it now holds `total_outputs` outputs of `owner`."""
        if self._hist_state is not None and self._hist_state[0] == owner:
            self._hist_state = (owner, int(total_outputs), self._hist_state[2])

    def invalidate(self):
        self._hist_state = None

    # ---------------------------------------------------------------- kernels
    def _launch(self, mode, cores, tensors, defines, args, brisc_args, ncrisc_args=None, cb_bytes=16384):
        core_set = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores})
        common = [int(a) for a in args] + [int(h.buffer_address()) for h in self.hist]
        base = {"SAMPLE_PARAM_BASE": 0, "SAMPLE_AUX_DRAM": 1, "SAMPLE_POS_DRAM": 1, "SAMPLE_OUT_DRAM": 1,
                "SAMPLE_TOKDST": 0, "SAMPLE_TOKDST_DRAM": 1, "SAMPLE_DLIST_WORD": 0, "SAMPLE_Q_PAGE": 0,
                "SAMPLE_OUT_ROW": 0, "SAMPLE_POS_MODE": 0, "SAMPLE_POS_WORD": 0, "SAMPLE_POS_ADD": 0,
                "SAMPLE_ROW_OP": 0, "SAMPLE_POINT_DRAFT": 0, "SAMPLE_EXTRA_WORD": 1, "SAMPLE_SCRATCH_HALF": 0,
                "SAMPLE_LOGITS_DRAM": 1, "SAMPLE_LOGITS_RM": 0, "SAMPLE_BATCH": 1, "SAMPLE_ID_OFFSET": 0,
                "SAMPLE_TILES": self.vocab_tiles, "SAMPLE_SPLIT_TILE": self.vocab_tiles}
        base.update(dict(defines))
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=core_set,
            defines=[("SAMPLE_MODE", str(mode)), ("SAMPLE_K", str(K)), ("SAMPLE_WORKERS", str(self.workers)),
                     ("SAMPLE_VOCAB_TILES", str(self.vocab_tiles)), ("SAMPLE_NSLOTS", str(self.num_slots)),
                     ("SAMPLE_PROFILE", os.environ.get("GEMMA_SAMPLING_PROFILE", "0"))]
            + [(k, str(int(v))) for k, v in sorted(base.items())],
            brisc_common_runtime_args=common,
            ncrisc_common_runtime_args=common,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=brisc_args, ncrisc_args=ncrisc_args or [(c, [0]) for c in cores]
            ),
        )
        cb = ttnn.CBDescriptor(
            total_size=cb_bytes,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, 2048, ttnn.TileDescriptor(ttnn.Tile((32, 32))))],
        )
        program = ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=[cb], semaphores=[])
        ttnn.generic_op(list(tensors), program)

    def _addr(self, t):
        return int(t.buffer_address()) if t is not None else int(self.dummy.buffer_address())

    def _common(self, logits=None, out=None, aux=None, pos=None, tokdst=None):
        return [self._addr(logits), self._addr(self.heads), self._addr(self.params), self._addr(self.partial),
                self._addr(self.rowbuf), self._addr(self.qbuf), self._addr(out), self._addr(aux), self._addr(pos),
                self.num_slots, self._addr(tokdst), self._addr(self.dlist)]

    def scan(self, logits, rows, *, param_base=0, aux=None, extra_word=1):
        """Mode 0: penalized per-worker top-K of logits rows [0, rows): BF16, one 32-row tile (TILE) or one page per
        row (ROW_MAJOR, e.g. an untilized logits row). Params pages param_base + r."""
        shape = tuple(int(x) for x in logits.shape)
        padded = tuple(int(x) for x in logits.padded_shape)
        width = shape[-1]
        row_major = logits.layout == ttnn.ROW_MAJOR_LAYOUT
        if logits.dtype != ttnn.bfloat16 or not (row_major or logits.layout == ttnn.TILE_LAYOUT):
            raise ValueError("device sampling scan requires BF16 tiled or row-major logits")
        if width != self.vocab or padded[-1] != width or not 1 <= rows <= 16 or (not row_major and padded[-2] != 32):
            raise ValueError(f"device sampling scan requires one row tile of <=16 vocab rows, got {shape}/{padded}")
        if row_major and math.prod(shape[:-1]) < rows:
            raise ValueError("device sampling scan: fewer logit rows than sampled rows")
        tiles = width // 32
        rb = ((rows * 32 + 63) // 64) * 64
        batch = 8 if rows <= 2 else 4
        tile_bytes = rows * 64 if row_major else 2 * rb
        half = 128 * rows + 64 + 8 * rows * K + batch * tile_bytes + batch * rows * 128
        half = ((half + 63) // 64) * 64
        n = len(self.cores)
        aux_t = aux if aux is not None else self.dummy
        defines = {"SAMPLE_ROWS": rows, "SAMPLE_TILES": tiles, "SAMPLE_SPLIT_TILE": tiles,
                   "SAMPLE_LOGITS_DRAM": _dram(logits), "SAMPLE_BATCH": batch, "SAMPLE_LOGITS_RM": int(row_major),
                   "SAMPLE_SCRATCH_HALF": half, "SAMPLE_EXTRA_WORD": extra_word, "SAMPLE_PARAM_BASE": param_base,
                   "SAMPLE_AUX_DRAM": _dram(aux_t)}
        cb_bytes = ((2 * half + 2047) // 2048) * 2048
        self._launch(
            0, self.cores, [logits, self.params, self.partial], defines.items(), self._common(logits=logits, aux=aux_t),
            brisc_args=[(c, [i]) for i, c in enumerate(self.cores)],
            ncrisc_args=[(c, [n + i]) for i, c in enumerate(self.cores)],
            cb_bytes=cb_bytes,
        )

    def rows(self, rows, op, *, param_base=0, out=None, pos=None, pos_word=0, pos_add=0, out_row=0, q_page=0,
             tokdst=None, dlist_word=0, point_draft=False):
        """Mode 1: merge + vLLM sampler per row. op 0 SAMPLE -> record out[out_row + row] (+ history);
        1 DRAFT -> q in qbuf[q_page], record out[out_row] and tokdst, draft list word dlist_word; 2 PREP -> rowbuf."""
        out = self.out if out is None else out
        pos_t = pos if pos is not None else self.dummy
        tok_t = tokdst if tokdst is not None else self.dummy
        defines = {"SAMPLE_ROWS": rows, "SAMPLE_POS_MODE": 1 if pos is not None else 0, "SAMPLE_POS_WORD": pos_word,
                   "SAMPLE_POS_ADD": pos_add, "SAMPLE_ROW_OP": op, "SAMPLE_OUT_ROW": out_row, "SAMPLE_Q_PAGE": q_page,
                   "SAMPLE_POINT_DRAFT": int(point_draft), "SAMPLE_PARAM_BASE": param_base,
                   "SAMPLE_POS_DRAM": _dram(pos_t), "SAMPLE_OUT_DRAM": _dram(out), "SAMPLE_TOKDST": int(tokdst is not None),
                   "SAMPLE_TOKDST_DRAM": _dram(tok_t), "SAMPLE_DLIST_WORD": dlist_word}
        cores = self.cores[:rows]
        scratch = 512 + self.workers * 64 + 64 + 1024 + K * 16 + 64
        self._launch(
            1, cores, [self.partial, self.params, out], defines.items(),
            self._common(out=out, pos=pos_t, tokdst=tok_t),
            brisc_args=[(c, [r]) for r, c in enumerate(cores)],
            cb_bytes=((scratch + 2047) // 2048) * 2048,
        )

    def verify(self, rows, *, pos, aux, pos_word=0, pos_add=0):
        """Mode 2: sequential rejection sampling of the rows - 1 drafts (aux words 1..) against rowbuf rows -> cyc."""
        defines = {"SAMPLE_ROWS": rows, "SAMPLE_POS_MODE": 1, "SAMPLE_POS_WORD": pos_word, "SAMPLE_POS_ADD": pos_add,
                   "SAMPLE_ROW_OP": 2, "SAMPLE_POS_DRAM": _dram(pos), "SAMPLE_OUT_DRAM": _dram(self.cyc),
                   "SAMPLE_AUX_DRAM": _dram(aux)}
        cores = self.cores[:1]
        scratch = 1024 + rows * 1024 + (rows - 1) * 1024 + K * 8
        self._launch(
            2, cores, [self.rowbuf, self.qbuf, self.params, self.cyc], defines.items(),
            self._common(out=self.cyc, pos=pos, aux=aux),
            brisc_args=[(cores[0], [0])],
            cb_bytes=((scratch + 2047) // 2048) * 2048,
        )

    # ---------------------------------------------------------------- composite device sequences
    def sample_decode(self, logits, pos):
        """Non-speculative decode of row 0 (params page 0); the drawn token sits at pos[0] + 1 -> out row 0."""
        self.scan(logits, 1)
        self.rows(1, 0, pos=pos, pos_add=1)

    def draft(self, logits, step, *, pos, tokdst, point_draft=False):
        """Draft step `step` (1-based) of the cycle whose context token sits at pos[0]: x_step ~ q_step (penalties
        count x_1..x_{step-1}) at pos[0] + step -> q in qbuf[step - 1], record out[step - 1], token into `tokdst`."""
        base = DRAFT_PARAM_BASE + step - 1
        self.scan(logits, 1, param_base=base, aux=self.dlist, extra_word=1)
        self.rows(1, 1, param_base=base, pos=pos, pos_add=step, out_row=step - 1, q_page=step - 1, tokdst=tokdst,
                  dlist_word=step, point_draft=point_draft)

    def verify_rows(self, logits, rows, *, tokens, pos):
        """Verify rows [c, x_1..x_{rows-1}] (`tokens`: the verify input, words 0..rows-1) at pos[0].. -> cyc."""
        self.scan(logits, rows, aux=tokens, extra_word=1)
        self.rows(rows, 2)
        self.verify(rows, pos=pos, aux=tokens)

    def read_out(self):
        return ttnn.to_torch(self.out).reshape(16, 4).to(torch.int64)

    def read_cyc(self):
        return [int(v) for v in ttnn.to_torch(self.cyc).reshape(-1).tolist()]

    def release(self):
        for t in [*self.hist, self.params, self.partial, self.heads, self.rowbuf, self.qbuf, self.out, self.cyc,
                  self.dlist, self.dummy]:
            ttnn.deallocate(t)
