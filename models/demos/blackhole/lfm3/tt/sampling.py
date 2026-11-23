# SPDX-License-Identifier: Apache-2.0
"""Exact on-device vLLM sampling for LFM3 on one P150 (kernels/sampling.cpp, kernels/sampling_math.hpp; copied
unchanged from the LFM2.5 runtime lfm2/tt, vocabulary-generic: LFM3's 128000 = 4000 tiles).

Forked from Qwen36's device sampler (models/demos/blackhole/qwen36/tt/device_sampling.py: same arithmetic, Philox
draws and history buffers; the speculative modes dropped) and extended to top_k disabled / above 64.

Processing follows vLLM's v1 Sampler (vllm/v1/sample/sampler.py) for one request row:

    float32 logits -> repetition, frequency, presence penalties (apply_all_penalties; output-token counts and the
    prompt mask live on device) -> temperature (logits / T) -> min_p -> top-k (ties at the k-th value kept) ->
    top-p (ascending cumulative probability <= 1 - p masked, the largest always kept) -> softmax -> draw.

The device keeps the exact global top-(top_k + 8) candidates (at most K = 72) of the penalized logits by (value desc,
id asc).  Rows with 1 <= top_k <= 64 are processed from the candidates alone.  Rows with top_k disabled (0 / -1 /
>= vocab) or above 64 ("full" rows) also get the full-vocabulary mass Z = sum_i exp((l_i - l_max) / T) (fixed-point
sums per worker, merged per row), so the probability left beyond the candidates is known: a min_p or top-p cut
inside the candidates is exact, and an uncut tail is sampled with its true mass.  The host redraws one token from
the step's logits (vLLM's float32 sampler on the same Philox uniform, `host_sample`) when the device cannot resolve
it: the draw lands in the uncut tail (R_TAIL), the nucleus / min_p set / a top_k > 64 reaches beyond the candidates
(R_CUT), or a top-k tie group may continue beyond them (R_TIE).  Only token ids and flags are read back.

Randomness is Philox4x32-10 keyed by the request seed, counter (absolute token position, purpose): a seeded
request reproduces its output; unseeded requests draw a random 64-bit base seed.
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
KW = 24  # kernels/sampling.cpp: entries a worker keeps per row (its best min(candidates, KW))
MASS_BITS = 46  # kernels/sampling_math.hpp: mass weights below 2^-46 of the maximum are dropped
SAMPLING_EPS = 1e-5  # vLLM: temperature below this is greedy
_KERNEL = str(Path(__file__).with_name("kernels").joinpath("sampling.cpp").resolve())
_HIST_COLS = 32
PARAM_ROWS = 32  # params pages (rows of the uint32 [PARAM_ROWS, 32] params table)
_ROW_WORDS = 256
_MASS_SLOT = 64
_RMAX_COPIES = 8  # kernels/sampling.cpp RMAX_COPIES: row-maximum pages, one per DRAM bank (each scan worker reads one)

F_ACTIVE, F_GREEDY, F_PENALTY, F_REP, F_NOCOMMIT, F_MONO, F_FULL = 1, 2, 4, 8, 32, 64, 128
# params words: a row's history also counts aux tokens [b, b + n), n = word W_EXTRA, b = word W_EXTRA_BASE (shared
# speculative verify rounds: the drafts of the row's group before the row)
W_EXTRA, W_EXTRA_BASE = 12, 15
R_HOST, R_TIE, R_TAIL, R_CUT = 1, 2, 4, 8
P_SAMPLE = 1
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


def uniform(seed, position, purpose=P_SAMPLE):
    """The device's uniform draw in [0, 1): 52 bits of Philox4x32-10(counter=(position, purpose, 0, 0))."""
    c = philox4x32((position, purpose, 0, 0), (seed & _MASK32, (seed >> 32) & _MASK32))
    mant = ((c[0] << 20) | (c[1] >> 12)) & ((1 << 52) - 1)
    return mant / float(1 << 52)


@dataclass(frozen=True)
class RowSampling:
    """One request's sampling parameters (vLLM SamplingParams semantics) plus its resolved 64-bit seed.
    top_k 0 = disabled."""

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
    def full(self):
        """top_k disabled or above MAX_TOP_K: the device also needs the full-vocabulary mass."""
        return not self.greedy and (self.top_k == 0 or self.top_k > MAX_TOP_K)

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
        """Device candidate count: the top-k plus an 8-entry margin that detects truncated tie groups (K for full
        rows)."""
        if self.greedy:
            return 1
        return K if self.full else min(K, int(self.top_k) + 8)

    def words(self, slot):
        f32 = np.float32
        flags = F_ACTIVE
        flags |= F_GREEDY if self.greedy else 0
        flags |= F_PENALTY if self.penalized else 0
        flags |= F_REP if self.uses_prompt else 0
        flags |= F_MONO if self.monotone else 0
        flags |= F_FULL if self.full else 0
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
        w[13] = _f32(f32(1.0) - top_p)  # vLLM: 1 - p in float32
        w[14] = self.candidates
        if not self.greedy:
            t = float(f32(self.temperature))
            # log2(e) / T as unsigned Q32.32 for the integer exp2 of the candidate weights and the mass.
            c = round((1.0 / (math.log(2.0) * t)) * (1 << 32))
            w[18] = c & _MASK32
            w[19] = (c >> 32) & _MASK32
            # Mass window in Q.24 logits: tokens further below the maximum weigh < 2^-MASS_BITS (2 units of margin).
            win = math.ceil(MASS_BITS * math.log(2.0) * t * (1 << 24)) + 2
            w[20] = win & _MASK32
            w[21] = (win >> 32) & _MASK32
        return w


@dataclass
class SamplingRequest:
    """A request routed to the device sampler: identity, parameters, live output ids, prompt ids (repetition)."""

    owner: object
    spec: RowSampling
    outputs: list
    prompt: object = None


def device_eligible(params, num_output_tokens=0):
    """True when the device sampler reproduces vLLM's sampler for this request (else the host path runs).

    Host-only: logprobs / prompt logprobs, allowed_token_ids, bad_words, logit_bias, an unsatisfied min_tokens,
    structured outputs (guided decoding) and logits processors.  Every temperature / top_k / top_p / min_p /
    penalty / seed combination is eligible (single tokens may still be redrawn on the host, see the module doc)."""
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
    return True


def row_sampling(params, seed, vocab):
    top_k = int(params.top_k)
    if top_k <= 0 or top_k >= vocab:
        top_k = 0
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

    Used for the rows the device flags (R_HOST); identical semantics to the host path."""
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
    k = torch.tensor([spec.top_k]) if spec.top_k else None
    p = torch.tensor([spec.top_p], dtype=torch.float32) if spec.top_p < 1.0 else None
    x = apply_top_k_top_p(x, k, p)
    probs = x.softmax(dim=-1, dtype=torch.float64).reshape(-1)
    support = torch.nonzero(probs > 0).reshape(-1)
    order = support[torch.argsort(-probs[support], stable=True)]
    cum = torch.cumsum(probs[order], 0)
    u = uniform(spec.seed, position, purpose) * float(cum[-1])
    index = int(torch.searchsorted(cum, torch.tensor([u], dtype=torch.float64), right=True))
    return int(order[min(index, order.numel() - 1)])


class DeviceSampler:
    """Fixed-address device buffers and the native kernels (scan, row). Single-stream, non-reentrant owner."""

    def __init__(self, device, vocab_size, num_slots=1, cores=None):
        if vocab_size % 32:
            raise ValueError("device sampling requires a tile-aligned vocabulary")
        grid = device.compute_with_storage_grid_size()
        count = min(grid.x * grid.y, int(cores or os.environ.get("LFM_SAMPLING_CORES", "110")))
        self.device = device
        self.vocab = vocab_size
        self.vocab_tiles = vocab_size // 32
        self.num_slots = num_slots
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
        self.workers = 2 * count
        if count % grid.x:
            raise ValueError("device sampling uses whole rows of the core grid")
        u32 = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def zeros(rows, cols, dtype=ttnn.uint32):
            return ttnn.from_torch(torch.zeros(rows, cols, dtype=torch.int32), **dict(u32, dtype=dtype))

        self.hist = [zeros(self.vocab_tiles, _HIST_COLS) for _ in range(num_slots)]
        self.params = zeros(PARAM_ROWS, 32)
        self.partial = zeros(16 * self.workers, 2 * KW)
        self.heads = zeros(16, 16 * self.workers)
        self.mass = zeros(16, (_MASS_SLOT // 4) * self.workers)
        self.rowbuf = zeros(16, _ROW_WORDS)
        self.out = zeros(16, 4)
        self.dummy = zeros(1, 16)
        self.rmax = zeros(_RMAX_COPIES, 16 * self.workers)  # every worker's stripe maxima per row (mode 2), copies
        self._params_host = None
        # slot -> (owner, outputs incorporated, prompt flags present)
        self._hist_state = [None] * num_slots

    # ---------------------------------------------------------------- host state
    def set_params(self, rows):
        """rows: list of 32-word param rows (row r -> params page r); inactive rows are zero."""
        table = np.zeros((PARAM_ROWS, 32), dtype=np.uint32)
        for r, words in enumerate(rows):
            if words is not None:
                table[r] = np.asarray(words, dtype=np.uint64).astype(np.uint32)
        self.set_param_table(table)

    def set_param_table(self, table):
        """table: uint32 [PARAM_ROWS, 32] (row r -> params page r; inactive rows zero), copied when it changed."""
        if self._params_host is not None and np.array_equal(table, self._params_host):
            return
        host = ttnn.from_torch(torch.from_numpy(table.view(np.int32)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.params)
        self._params_host = table

    def ensure_history(self, slot, owner, output_ids, prompt_ids, use_prompt):
        """Make history slot `slot` hold exactly `output_ids` counts (+ prompt flags when repetition is on).
        output_ids: any sized iterable of ids (read only when the slot's recorded state differs)."""
        state = (owner, len(output_ids), bool(use_prompt))
        if self._hist_state[slot] == state:
            return False
        ids = np.fromiter(output_ids, dtype=np.int64, count=len(output_ids))
        counts = np.bincount(ids, minlength=self.vocab)[: self.vocab]
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
    def _launch(self, mode, cores, tensors, defines, args, brisc_args, ncrisc_args=None, cb_bytes=16384):
        # Cores are a row-major prefix of the grid: whole grid rows as one rectangle (multicast dispatch).
        gx = self.device.compute_with_storage_grid_size().x
        n = len(cores)
        ranges = set()
        if n >= gx:
            ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, n // gx - 1)))
        if n % gx:
            ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, n // gx), ttnn.CoreCoord(n % gx - 1, n // gx)))
        core_set = ttnn.CoreRangeSet(ranges)
        common = [int(a) for a in args] + [int(h.buffer_address()) for h in self.hist]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=core_set,
            defines=[("SAMPLE_MODE", str(mode)), ("SAMPLE_K", str(K)), ("SAMPLE_WORKERS", str(self.workers)),
                     ("SAMPLE_NSLOTS", str(self.num_slots)),
                     ("SAMPLE_PROFILE", os.environ.get("LFM_SAMPLING_PROFILE", "0"))]
            + [(k, str(v)) for k, v in defines],
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

    def _common(self, logits=None, pos=None, aux=None):
        return [self._addr(logits), self._addr(self.heads), self._addr(self.params), self._addr(self.partial),
                self._addr(self.rowbuf), self._addr(self.out), self._addr(self.mass), self._addr(pos),
                self._addr(self.rmax), self._addr(aux)]

    def _stripe(self, logits, rows):
        """(bytes per staged vocabulary tile, tiles per worker) after validating the logits."""
        shape = tuple(int(x) for x in logits.shape)
        padded = tuple(int(x) for x in logits.padded_shape)
        if logits.dtype != ttnn.bfloat16 or logits.layout != ttnn.TILE_LAYOUT:
            raise ValueError("device sampling scan requires BF16 tiled logits")
        if padded[-1] != self.vocab or not 1 <= rows <= 16 or padded[-2] != 32:
            raise ValueError(f"device sampling scan requires one row tile of <=16 rows, got {shape}/{padded}")
        # kernels/sampling.cpp TILE_BYTES: face 0 + the rows' 64-byte-aligned block of face 1, one read per tile
        return 512 + ((rows * 32 + 63) // 64) * 64, -(-self.vocab_tiles // self.workers)

    def needs_global_max(self):
        """Some active row is unpenalized F_FULL (its mass window starts at the row's global maximum)."""
        if self._params_host is None:
            return False
        flags = self._params_host[:16, 0]
        return bool(((flags & (F_ACTIVE | F_FULL | F_PENALTY)) == (F_ACTIVE | F_FULL)).any())

    def scan(self, logits, rows, full=None, aux=None):
        """Mode 0: penalized per-worker top-K and F_FULL masses of logits rows [0, rows) (BF16, one 32-row tile).
        `full`: first run the row-maximum program (mode 2) so unpenalized F_FULL rows use their global maximum as the
        mass reference (a traced sequence fixes it; None decides from the current params).  aux: u32 token tensor
        (one page) whose tokens [b, b + n) (params words W_EXTRA_BASE / W_EXTRA) a penalized row also counts as
        outputs."""
        if full is None:
            full = self.needs_global_max()
        tile_bytes, stripe = self._stripe(logits, rows)
        n = len(self.cores)
        dram = logits.memory_config().buffer_type == ttnn.BufferType.DRAM
        brisc = [(c, [i]) for i, c in enumerate(self.cores)]
        ncrisc = [(c, [n + i]) for i, c in enumerate(self.cores)]
        common = [("SAMPLE_ROWS", rows), ("SAMPLE_TILES", self.vocab_tiles), ("SAMPLE_LOGITS_DRAM", int(dram))]
        if full:
            # kernels/sampling.cpp mode 2, per RISC half: params | the worker's stripe of tiles | stage.
            half = ((128 * rows + stripe * tile_bytes + 64 + 63) // 64) * 64
            self._launch(2, self.cores, [logits, self.params, self.rmax], common + [("SAMPLE_SCRATCH_HALF", half)],
                         self._common(logits=logits), brisc_args=brisc, ncrisc_args=ncrisc,
                         cb_bytes=((2 * half + 2047) // 2048) * 2048)
        # mode 0, per RISC half: params | the worker's stripe of tiles | its history pages per row | stage | (full)
        # every worker's stripe maxima.
        half = 128 * rows + stripe * tile_bytes + rows * stripe * 128 + K * 8 + 64 + (self.workers * 64 if full else 0)
        half = ((half + 63) // 64) * 64
        self._launch(
            0, self.cores, [logits, self.params, self.partial], common + [
                ("SAMPLE_SCRATCH_HALF", half), ("SAMPLE_ROW_OP", 0), ("SAMPLE_GLOBAL_MAX", int(bool(full)))],
            self._common(logits=logits, aux=aux), brisc_args=brisc, ncrisc_args=ncrisc,
            cb_bytes=((2 * half + 2047) // 2048) * 2048,
        )

    def rows(self, rows, op, *, pos=None):
        """Mode 1: merge + vLLM sampler per row; op 0 SAMPLE -> out [16, 4] (positions = pos[row] + 1), 2 PREP ->
        rowbuf[row] (processed distribution)."""
        defines = [("SAMPLE_ROWS", rows), ("SAMPLE_TILES", self.vocab_tiles), ("SAMPLE_LOGITS_DRAM", 1),
                   ("SAMPLE_SCRATCH_HALF", 0), ("SAMPLE_ROW_OP", op)]
        cores = self.cores[:rows]
        # mode 1: params/pos/out/hist 384 | heads | mass | 32 worker lists | result | merged candidates
        scratch = 384 + self.workers * (64 + _MASS_SLOT) + 32 * KW * 8 + 4 * _ROW_WORDS + K * 12 + 64
        self._launch(
            1, cores, [self.partial, self.params, self.out], defines, self._common(pos=pos),
            brisc_args=[(c, [r]) for r, c in enumerate(cores)],
            cb_bytes=((scratch + 2047) // 2048) * 2048,
        )

    # ---------------------------------------------------------------- composite device sequences
    def sample_decode(self, logits, rows, pos, full=None, aux=None):
        """Decode: rows [0, rows) of logits; positions = pos[row] + 1 (decode input positions). `full`: see scan.
        aux: see scan (speculative verify rows)."""
        self.scan(logits, rows, full, aux)
        self.rows(rows, 0, pos=pos)

    def read_out(self, rows, host=None):
        """The sampled rows (int64 [rows, 4]); host: `out` already copied to the host (speculate.EarlyRead)."""
        return (host if host is not None else ttnn.to_torch(self.out)).reshape(16, 4)[:rows].to(torch.int64)

    def release(self):
        for t in [*self.hist, self.params, self.partial, self.heads, self.mass, self.rowbuf, self.out, self.dummy,
                  self.rmax]:
            ttnn.deallocate(t)
