# SPDX-License-Identifier: Apache-2.0
"""Exact on-device vLLM sampling and lossless speculative sampling (MTP chains, DFlash) for Qwen on one P150.

Processing follows vLLM's v1 Sampler (vllm/v1/sample/sampler.py) for one request row:

    float32 logits -> repetition, frequency, presence penalties (apply_all_penalties; output-token counts and the
    prompt mask live on device) -> temperature (logits / T) -> min_p -> top-k (ties at the k-th value kept) ->
    top-p (ascending cumulative probability <= 1 - p masked, the largest always kept) -> softmax -> draw.

The device keeps the exact global top-(top_k + 8) candidates (at most K = 72) of the penalized logits by (value desc,
id asc), so every request with 1 <= top_k <= 64 is processed exactly; a top-k tie group that might continue beyond
the candidates is reported (R_INEXACT) and that one token is redrawn on the host from the retained logits. Only
token ids, flags and accept counts are read back.

Randomness is Philox4x32-10 keyed by the request seed, counter (absolute token position, purpose): a seeded
request reproduces its output; unseeded requests draw a random 64-bit base seed.

Speculative sampling (sequential chain of D drafts x_1..x_D verified by D + 1 target rows): x_i is accepted with
probability min(1, p_i(x_i) / q_i(x_i)), where p_i is the processed target distribution whose penalty history
includes x_1..x_i-1 and q_i the distribution x_i was drawn from; the first rejection draws from
normalize(max(0, p_i - q_i)) and ends the chain; full acceptance draws the bonus from p_D. q_i is either a stored
sampled distribution (MTP drafts from the processed draft head, DFlash2 drafts from the selector's candidate
softmax) or the point mass on x_i (any deterministic proposer, e.g. DFlash2's greedy selector path). The emitted
distribution equals non-speculative sampling at the same settings.
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
SAMPLING_EPS = 1e-5  # vLLM: temperature below this is greedy
_KERNEL = str(Path(__file__).with_name("device_sampling_kernel.cpp").resolve())
_HIST_COLS = 32
_PARAM_ROWS = 32
_ROW_WORDS = 256
# Every kernel define (mode-specific values override these).
_DEFAULT_DEFINES = {
    "SAMPLE_ROWS": 1, "SAMPLE_TILES": 1, "SAMPLE_SPLIT_TILE": 1, "SAMPLE_ID_OFFSET": 0, "SAMPLE_LOGITS_DRAM": 1,
    "SAMPLE_LOGITS_RM": 0, "SAMPLE_BATCH": 1, "SAMPLE_SCRATCH_HALF": 0, "SAMPLE_EXTRA_WORD": 0, "SAMPLE_POS_MODE": 0,
    "SAMPLE_POS_WORD": 0, "SAMPLE_POS_ADD": 0, "SAMPLE_ROW_OP": 0, "SAMPLE_OUT_PAGE": 16, "SAMPLE_OUT_OFFSET": 0,
    "SAMPLE_POINT_DRAFT": 0, "SAMPLE_PARAM_BASE": 0, "SAMPLE_Q_PAGE": 0, "SAMPLE_OUT_WORD": 0, "SAMPLE_POINT_Q": 0,
    "SAMPLE_VA": 0, "SAMPLE_SEL_TASK": 0, "SAMPLE_SEL_ROWS": 1, "SAMPLE_SEL_PROC": 0, "SAMPLE_SEL_LANE": 0,
    "SAMPLE_SEL_PURPOSE": 4,
}

F_ACTIVE, F_GREEDY, F_PENALTY, F_REP, F_EXTRA, F_NOCOMMIT, F_MONO = 1, 2, 4, 8, 16, 32, 64
R_INEXACT = 1
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

    def words(self, slot, position=0, n_extra=0, nocommit=False, extra_base=0):
        f32 = np.float32
        flags = F_ACTIVE
        flags |= F_GREEDY if self.greedy else 0
        flags |= F_PENALTY if self.penalized else 0
        flags |= F_REP if self.uses_prompt else 0
        flags |= F_EXTRA if n_extra else 0
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
        w[20] = int(n_extra)  # this row also counts that many speculative extra tokens (aux words) as outputs
        w[21] = int(extra_base)  # ... starting at aux word SAMPLE_EXTRA_WORD + extra_base (shared-round groups)
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


class DeviceSampler:
    """Fixed-address device buffers and the native kernels. Single-stream, non-reentrant owner.

    Parameter rows (32 words each): rows 0..R-1 are the decode rows or the verify rows of a speculative cycle (verify
    row r counts the drafts x_1..x_r as extra outputs); rows DRAFT_ROW + i are draft step i of an MTP chain (counting
    the chain's earlier drafts). Draft chains write their tokens to words CHAIN_WORD.. of `cyc` (cycle) or `first`
    (first drafts after a prefill / host redraw); verification writes emitted tokens to cyc words 0..R-1, the accepted
    count to word 16 and the host-redraw flag (1 + chain index) to word 17."""

    DRAFT_ROW = 16
    CHAIN_WORD = 20

    def __init__(self, device, vocab_size, num_slots=1, cores=None):
        if vocab_size % 32:
            raise ValueError("device sampling requires a tile-aligned vocabulary")
        grid = device.compute_with_storage_grid_size()
        count = min(grid.x * grid.y, int(cores or os.environ.get("QWEN36_SAMPLING_CORES", "110")))
        self.device = device
        self.vocab = vocab_size
        self.vocab_tiles = vocab_size // 32
        self.num_slots = num_slots
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
        self.workers = 2 * count
        u32 = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def zeros(rows, cols, dtype=ttnn.uint32):
            return ttnn.from_torch(torch.zeros(rows, cols, dtype=torch.int32), **dict(u32, dtype=dtype))

        self._zeros = zeros
        self.hist = [zeros(self.vocab_tiles, _HIST_COLS) for _ in range(num_slots)]
        self.params = zeros(_PARAM_ROWS, 32)
        self.partial = zeros(16 * self.workers, 2 * K)
        self.heads = zeros(16, 16 * self.workers)
        self.rowbuf = zeros(16, _ROW_WORDS)
        self.qbuf = zeros(16, _ROW_WORDS)
        self.out = zeros(16, 4)
        self.cyc = zeros(1, 32, ttnn.int32)
        self.first = zeros(1, 32, ttnn.int32)
        self.dummy = zeros(1, 16)
        # NativeVerifyAccept-compatible outputs per verify width R: head [1, 2R-1] + 5 scalars [1, 1].
        self._va = {}
        self._params_host = None
        # slot -> (owner, outputs incorporated, prompt flags present)
        self._hist_state = [None] * num_slots

    def va_buffers(self, rows):
        """Allocate (before any trace) the fixed verify outputs for width `rows`: (head, accepted, next_token,
        advance, next_position, start)."""
        if rows not in self._va:
            i32 = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._va[rows] = tuple(
                ttnn.from_torch(torch.zeros(shape, dtype=torch.int32), **i32)
                for shape in [(1, 2 * rows - 1)] + [(1, 1)] * 5
            )
        return self._va[rows]

    # ---------------------------------------------------------------- host state
    def set_params(self, rows):
        """rows: list of 32-word param rows (row r -> params page r); inactive rows are zero."""
        table = np.zeros((_PARAM_ROWS, 32), dtype=np.uint32)
        for r, words in enumerate(rows):
            if words is not None:
                table[r] = np.asarray(words, dtype=np.uint64).astype(np.uint32)
        if self._params_host is not None and np.array_equal(table, self._params_host):
            return
        host = ttnn.from_torch(torch.from_numpy(table.view(np.int32)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.params)
        self._params_host = table

    def ensure_history(self, slot, owner, output_ids, prompt_ids, use_prompt):
        """Make history slot `slot` hold exactly `output_ids` counts (+ prompt flags when repetition is on)."""
        state = (owner, len(output_ids), bool(use_prompt))
        if self._hist_state[slot] == state:
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
    def _launch(self, mode, cores, tensors, defines, args, brisc_args, ncrisc_args=None, cb_bytes=16384, extra=()):
        core_set = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores})
        common = [int(a) for a in args] + [int(h.buffer_address()) for h in self.hist] + [self._addr(t) for t in extra]
        settings = dict(_DEFAULT_DEFINES)
        settings.update(dict(defines))
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=core_set,
            defines=[("SAMPLE_MODE", str(mode)), ("SAMPLE_K", str(K)), ("SAMPLE_WORKERS", str(self.workers)),
                     ("SAMPLE_VOCAB_TILES", str(self.vocab_tiles)), ("SAMPLE_NSLOTS", str(self.num_slots)),
                     ("SAMPLE_PROFILE", os.environ.get("QWEN36_SAMPLING_PROFILE", "0"))]
            + [(k, str(v)) for k, v in settings.items()],
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
        ttnn.generic_op(list(tensors) + [t for t in extra if t is not None], program)

    def _addr(self, t):
        return int(t.buffer_address()) if t is not None else int(self.dummy.buffer_address())

    def _common(self, logits=None, out=None, aux=None, pos=None):
        return [self._addr(logits), self._addr(self.heads), self._addr(self.params), self._addr(self.partial), self._addr(self.rowbuf),
                self._addr(self.qbuf), self._addr(out), self._addr(aux), self._addr(pos), self.num_slots]

    def scan(self, logits, rows, *, width=None, split_tile=None, id_offset=0, aux=None, extra_word=0, param_base=0):
        """Mode 0: penalized per-worker top-K of logits rows [0, rows) with parameter rows [param_base, +rows): BF16,
        one 32-row tile (TILE) or one page per row (ROW_MAJOR, e.g. untilized decode logits). Rows with n_extra > 0
        also count aux words [extra_word, extra_word + n_extra) as outputs."""
        shape = tuple(int(x) for x in logits.shape)
        padded = tuple(int(x) for x in logits.padded_shape)
        width = shape[-1] if width is None else width
        row_major = logits.layout == ttnn.ROW_MAJOR_LAYOUT
        if logits.dtype != ttnn.bfloat16 or not (row_major or logits.layout == ttnn.TILE_LAYOUT):
            raise ValueError("device sampling scan requires BF16 tiled or row-major logits")
        if padded[-1] != width or width % 32 or not 1 <= rows <= 16 or (not row_major and padded[-2] != 32):
            raise ValueError(f"device sampling scan requires one row tile of <=16 rows, got {shape}/{padded}")
        if row_major and math.prod(shape[:-1]) < rows:
            raise ValueError("device sampling scan: fewer logit rows than sampled rows")
        tiles = width // 32
        split = tiles if split_tile is None else split_tile
        rb = ((rows * 32 + 63) // 64) * 64
        batch = 8 if rows <= 2 else 4
        tile_bytes = rows * 64 if row_major else 2 * rb
        half = 128 * rows + 128 + 8 * rows * K + batch * tile_bytes + batch * rows * 128
        half = ((half + 63) // 64) * 64
        dram = logits.memory_config().buffer_type == ttnn.BufferType.DRAM
        n = len(self.cores)
        defines = [("SAMPLE_ROWS", rows), ("SAMPLE_TILES", tiles), ("SAMPLE_SPLIT_TILE", split),
                   ("SAMPLE_ID_OFFSET", id_offset), ("SAMPLE_LOGITS_DRAM", int(dram)), ("SAMPLE_BATCH", batch),
                   ("SAMPLE_LOGITS_RM", int(row_major)), ("SAMPLE_SCRATCH_HALF", half), ("SAMPLE_EXTRA_WORD", extra_word),
                   ("SAMPLE_PARAM_BASE", param_base)]
        cb_bytes = ((2 * half + 2047) // 2048) * 2048
        self._launch(
            0, self.cores, [logits, self.params, self.partial], defines, self._common(logits=logits, aux=aux),
            brisc_args=[(c, [i]) for i, c in enumerate(self.cores)],
            ncrisc_args=[(c, [n + i]) for i, c in enumerate(self.cores)],
            cb_bytes=cb_bytes,
        )

    def rows(self, rows, op, *, out=None, pos=None, pos_mode=0, pos_word=0, pos_add=0, param_base=0, q_page=0,
             out_word=0, point_draft=False):
        """Mode 1: merge + vLLM sampler per row. op 0 SAMPLE -> out [16,4]; 1 DRAFT -> q in qbuf page q_page, token in
        word out_word of `out` ([1,32] int32); 2 PREP -> rowbuf page row."""
        out = self.out if out is None else out
        out_page = 16 if op == 0 else 128
        defines = [("SAMPLE_ROWS", rows), ("SAMPLE_TILES", self.vocab_tiles), ("SAMPLE_SPLIT_TILE", self.vocab_tiles),
                   ("SAMPLE_POS_MODE", pos_mode), ("SAMPLE_POS_WORD", pos_word), ("SAMPLE_POS_ADD", pos_add),
                   ("SAMPLE_ROW_OP", op), ("SAMPLE_OUT_PAGE", out_page), ("SAMPLE_POINT_DRAFT", int(point_draft)),
                   ("SAMPLE_PARAM_BASE", param_base), ("SAMPLE_Q_PAGE", q_page), ("SAMPLE_OUT_WORD", out_word)]
        cores = self.cores[:rows]
        scratch = 512 + self.workers * 64 + 64 + 1024 + K * 16 + 64
        self._launch(
            1, cores, [self.partial, self.params, out], defines, self._common(out=out, pos=pos),
            brisc_args=[(c, [r]) for r, c in enumerate(cores)],
            cb_bytes=((scratch + 2047) // 2048) * 2048,
        )

    def verify_chain(self, rows, tokens, positions, *, point_q=False, va=False):
        """Mode 2: sequential rejection sampling of tokens[1..rows-1] against rowbuf rows 0..rows-1 -> cyc words
        (+ NativeVerifyAccept-format outputs and positions advanced in place when va)."""
        defines = [("SAMPLE_ROWS", rows), ("SAMPLE_POINT_Q", int(point_q)), ("SAMPLE_VA", int(va))]
        cores = self.cores[:1]
        extra = self.va_buffers(rows) if va else ()
        scratch = 1024 + ((K * 8 + 1023) // 1024) * 1024 + (2 * rows - 1) * 1024
        self._launch(
            2, cores, [self.rowbuf, self.qbuf, self.params, self.cyc, tokens, positions], defines,
            self._common(out=self.cyc, pos=positions, aux=tokens),
            brisc_args=[(cores[0], [0])],
            cb_bytes=((scratch + 2047) // 2048) * 2048, extra=extra,
        )
        return extra

    def select_draw(self, scores, candidates, next_token, path, task, rows, pos, *, processed=False):
        """Mode 3: DFlash2 sampled selector step `task` (draft position pos + 1 + task): x ~ q over the 16 candidates,
        q -> qbuf page task, x -> next_token and path[task]."""
        defines = [("SAMPLE_SEL_TASK", task), ("SAMPLE_SEL_ROWS", rows), ("SAMPLE_SEL_PROC", int(processed))]
        cores = self.cores[:1]
        self._launch(
            3, cores, [self.params, self.qbuf, pos], defines, self._common(pos=pos),
            brisc_args=[(cores[0], [0])], cb_bytes=4096, extra=(scores, candidates, next_token, path),
        )

    def lane_draw(self, scores, candidates, next_token, path, task, rows, pos, param_row, offset, *, processed=False,
                  purpose=P_SAMPLE):
        """Mode 3 for one lane of a batched drafter pass (shared rounds): the draft at anchor position pos[0] + offset
        of the request in parameter row `param_row`, x ~ q over the 16 candidates drawn with Philox purpose
        `purpose` (P_SAMPLE: the same uniform the target row verifying it draws with) -> next_token and path[task]
        (q is not kept). A greedy or inactive parameter row takes the best candidate (the greedy walk's choice)."""
        defines = [("SAMPLE_SEL_LANE", 1), ("SAMPLE_SEL_ROWS", rows), ("SAMPLE_SEL_PROC", int(processed)),
                   ("SAMPLE_SEL_PURPOSE", purpose)]
        cores = self.cores[:1]
        self._launch(
            3, cores, [self.params, pos], defines, self._common(pos=pos),
            brisc_args=[(cores[0], [0, int(task), int(param_row), int(offset)])], cb_bytes=4096,
            extra=(scores, candidates, next_token, path),
        )

    # ---------------------------------------------------------------- composite device sequences
    def sample_decode(self, logits, rows, pos):
        """Non-speculative decode: rows [0, rows) of logits; positions = pos[row] + 1 (decode input positions)."""
        self.scan(logits, rows)
        self.rows(rows, 0, pos=pos, pos_mode=2, pos_add=1)

    def draft_step(self, logits, step, *, pos, out, width=None, split_tile=None, id_offset=0, point_draft=False):
        """MTP draft chain step `step` (token position pos[0] + 1 + step): q -> qbuf page step, token -> word
        CHAIN_WORD + step of `out`; the scan counts the chain's earlier drafts (out words CHAIN_WORD..)."""
        self.scan(logits, 1, width=width, split_tile=split_tile, id_offset=id_offset, aux=out,
                  extra_word=self.CHAIN_WORD, param_base=self.DRAFT_ROW + step)
        self.rows(1, 1, out=out, pos=pos, pos_mode=1, pos_word=0, pos_add=1 + step, param_base=self.DRAFT_ROW + step,
                  q_page=step, out_word=self.CHAIN_WORD + step, point_draft=point_draft)

    def verify(self, logits, tokens, positions, rows, *, point_q=False, va=True):
        """Verify `rows` target rows (row r counts the drafts tokens[1..r]) -> cyc / VA outputs + history."""
        self.scan(logits, rows, aux=tokens, extra_word=1)
        self.rows(rows, 2)
        return self.verify_chain(rows, tokens, positions, point_q=point_q, va=va)

    def read_out(self, rows):
        return ttnn.to_torch(self.out).reshape(16, 4)[:rows].to(torch.int64)

    def read_cycle(self, buffer=None):
        return [int(v) for v in ttnn.to_torch(self.cyc if buffer is None else buffer).reshape(-1).tolist()]

    def request_rows(self, spec, depth, draft_steps=0):
        """Parameter rows for one request: verify rows 0..depth (row r counts r drafts), draft steps at DRAFT_ROW."""
        rows = [spec.words(0, n_extra=r) for r in range(depth + 1)]
        rows += [None] * (self.DRAFT_ROW - len(rows))
        rows += [spec.words(0, n_extra=i) for i in range(draft_steps)]
        return rows

    def release(self):
        for t in [*self.hist, self.params, self.partial, self.heads, self.rowbuf, self.qbuf, self.out, self.cyc, self.first,
                  self.dummy, *(t for bufs in self._va.values() for t in bufs)]:
            ttnn.deallocate(t)
