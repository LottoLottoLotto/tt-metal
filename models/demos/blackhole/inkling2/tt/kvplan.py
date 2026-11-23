# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Context profiles and the P150 DRAM plan of Inkling-2-Mini (owner LongCtx; standard library only).

The runtime (inkling2_vllm.py) and the launcher (tools/serve_inkling2.sh, `python3 kvplan.py --max-model-len N`)
call `plan(max_seq_len)`; it picks the profile, the expert / KV-cache formats, decode buckets, trace region and
embedding placement, and refuses (PlanError, exit code 2) when the configuration does not fit the card.

Profiles by max_seq_len; experts DEFAULT_PRECISION (Prec4's GPTQ-snapped BFP4, bfp4:gptq) in every profile (Main's
decision: 3.72 GB / token, all KL 0.0109) and attention DEFAULT_ATTN_PRECISION (q|gate GPTQ-BFP4, k / v / o BFP8:
-340 MB / token, all KL 0.0116 vs 0.0106; fast/NOTES-attn.md Round 6).  Env overrides: INKLING2_PRECISION = expert
map, or "quality" = BFP8 experts + BFP8 attention unless INKLING2_ATTN_PRECISION is given (QUALITY_PRECISION: all KL
0.0030, 1.4x the expert bytes; fits max_seq_len <= 283,008, <= 108,800 with INKLING2_SPEC=on, the KV formats above);
INKLING2_ATTN_PRECISION = attention map (e.g. bfp8); INKLING2_KV = "<k>:<v>" cache
formats; INKLING2_MOE_BALANCE = auto (default) | on | off; INKLING2_PREFILL_CHUNK = prefill chunk rows (<= 512 or a
multiple of 512, at most MAX_PREFILL_CHUNK).
  fast    <= 65,536    full-attention K / V caches BF16            (decode: fused, one bucket)
  long    <= 131,072   K / V BFP8 (measured lossless)
  native  <= 1,048,576 K BFP8 / V BFP4                            (the model's 1,048,576 tokens)
Prefill chunk (Plan.prefill_chunk, prefill_chunk_for): 4096 rows in every profile (PrefillLand, fast/NOTES-prefill.md:
TTFT -10 % at 8K-32K vs 2048, the same at <= 2K; activations 0.68 GiB at 4096, inside the reserve also at 1M; 8192
does not build: the MoE route program's circular buffers exceed L1), clipped to max_seq_len.
(BFP8 experts leave ~3.4 GiB for the full KV: 111K positions in BF16, 209K in BFP8; a 1M K BFP8 / V BFP4 cache is
13 GiB: 4-bit experts.)
Measured quality of each option: src/inkling2/tt/NOTES-longctx.md.

DRAM model (GiB = 2^30 B; TILE storage: BF16 2 B / value, BFP8 1088/1024, BFP4 576/1024):
  experts       24 layers x 128 experts x 3 x 1152 x 2304 values in the layer's format
  other weights attention (BFP8 projections + BF16 small ones), unembed BFP8, message gates, routers, norms,
                prefill gather indices: OTHER_GIB, measured (DRAM after load minus the computed parts) with all-BFP8
                attention; minus the attention map's BFP4 tensors (attn_saving_gib; the fused decode's stream copies
                FUSED_DECODE_GIB likewise)
  embedding     BF16 table 0.86 GiB on the device when EMBED_RESERVE_GIB stays free, else on the host
  full KV       8 slots (layers 5, 23 and 3 passes of 11, 17) x (max_seq_len + 64) x 8 heads x 128 x (K + V format)
  sliding KV    40 slots x 1152 rows (ring 1088 + zero sink block 64) x 8 x 128 x 2 (K, V) BF16
  tables        per decode bucket: full-cache gather row (4 B / position) + page table; tau table 4 B / position
  MoE balance   Inkling2MoE(balance=True) (MoE4/MoE5): each expert's floating gate/up pair also stored in the next
                DRAM bank (gate_up_dup, 2/27 of the expert values in the layer's format: +0.95 GiB BFP4, +1.79 GiB
                BFP8) so decode streams equal gate/up bytes from every bank; auto = on when the plan keeps
                BALANCE_MIN_HEADROOM_GIB headroom with it (bfp4:gptq: fast / long / native up to 1,015,168; off at
                1,048,576 and for BFP8 experts at 32K+)
  reserve       PREFILL_RESERVE_GIB: prefill chunk activations (measured peak 0.18 / 0.34 / 0.68 GiB at chunk 1024 /
                2048 / 4096), program binaries
Allocatable DRAM = 31.83 GiB (P150, measured with no trace region; quant/results/dev_bfp_mem.json) minus the trace
region minus ALLOC_SLACK_GIB (fragmentation: 31.25 GiB of 31.33 allocatable at a 512 MiB region).
Batched serving (`plan(..., max_num_seqs=S, kv_tokens=T)`, S 2..16; NOTES-batch.md section 6; S = 1 is the plan
above, unchanged): K / V formats from the max_seq_len profile; full KV = ONE pool of kv_blocks_for(T, S) + 1 blocks
(vLLM's num_gpu_blocks: T + 64 S tokens / 64, + the zero block) x 64 positions per full application instead of the
per-sequence caches; sliding KV x S; slot states (S - 1) x SLOT_STATE_BYTES (conv / glue rings, sampler history);
batch page table [32, W] + the prefill page row; trace region = TRACE_FIXED + BATCH_TRACE_MARGIN x the measured
batch_trace_bytes(R) of every BatchDecoder bucket R (batch_buckets(S): 45.6 MB at S 16 -> a 128 MiB region); embedding
placement as above (BatchDecoder(embed=...)).  Batched speculation (batch_spec_mode(): INKLING2_SPEC, NOTES-spec.md
"Batched speculation"): + batch_spec_gib (the drafter, its layer-1 pool, its per-slot rings), the SPEC_BUCKETS step
buckets and BATCH_SPEC_TRACE; a prefix snapshot then also holds the slot's drafter rings (PREFIX_DRAFTER_BYTES).
CLI: --max-num-seqs S --kv-tokens T.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field

GiB = 2 ** 30
NATIVE_MAX = 1_048_576          # hf/config.json text_config.model_max_length
FAST_MAX = 65_536
LONG_MAX = 131_072
MIN_LEN = 1024
MIN_BUCKET = 4096
BPV = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}
EXPERT_VALUES = 128 * 3 * 1152 * 2304   # per layer
N_LAYERS = 24
FULL_SLOTS, SLIDING_SLOTS = 8, 40
KV_ROW = 8 * 128                        # values per position per slot (K or V)
DRAM_TOTAL_GIB = 31.83
ALLOC_SLACK_GIB = 0.08
OTHER_GIB = 1.68                        # measured: DRAM after load - experts - KV (NOTES-longctx.md "memory")
FUSED_DECODE_GIB = 0.74                # AttnFast stream-layout attention copies (0.72) + FastGlue buffers (0.015)
SPEC_FIXED_GIB = 0.50                  # INKLING2_SPEC=on (Spec2)
SPEC_KV_PER_TOKEN = 3 * 2 * 8 * 128 * 2  # drafter K / V, 12 KiB per position
EMBED_GIB = 200064 * 2304 * 2 / GiB     # 0.86
EMBED_RESERVE_GIB = 1.5                 # free DRAM kept when the embedding table goes to the device
PREFILL_RESERVE_GIB = 1.25
TRACE_PER_BUCKET = 72 << 20             # TokenDecoder main + tail traces of one bucket (measured, rounded up)
TRACE_FIXED = 64 << 20
DEFAULT_PRECISION = "bfp4:gptq"        # Prec4's GPTQ-snapped BFP4 experts (same bytes as RTN BFP4, all KL 0.0109 vs 0.0151)
QUALITY_PRECISION = "bfp8"             # INKLING2_PRECISION=quality
# attention projection weights (tt/precision.py attn_weight_map grammar; quant/NOTES.md section 8, fast/NOTES-attn.md
# Round 6): q|gate GPTQ-BFP4 (the fused decode streams q|gate in BFP4 and k|v in BFP8 as two worker groups), k / v / o
# BFP8, in every profile.  Overrides: INKLING2_ATTN_PRECISION (plan / serving), INK_ATTN_PRECISION (tt/model.py)
DEFAULT_ATTN_PRECISION = 'mixed:{"default":{"q":"bfp4:gptq","gate":"bfp4:gptq"}}'
ATTN_FAMILIES = {"q": 2304 * 3072, "gate": 2304 * 3072, "k": 2304 * 1024, "v": 2304 * 1024, "o": 3072 * 2304}
MOE_BALANCE_FRACTION = 2 / 27          # gate_up_dup values / expert values (1/9 of gate / up, moe_layout.py)
BALANCE_MIN_HEADROOM_GIB = 0.5         # auto MoE balance: headroom kept after the dup tensors
# batched serving (NOTES-batch.md section 6; Inkling2Model(max_num_seqs=S, kv_blocks=NB), fast/batch_decode.BatchDecoder)
MAX_SEQS = 16                           # tt/model.py MAX_SEQS (MoE rows, LM head ids per decode step)
DEFAULT_KV_TOKENS = 262_144             # vLLM's shared KV pool when max_num_seqs > 1 (INKLING2_MAX_TOKENS_ALL_USERS)
KV_BLOCK = 64                           # vLLM block = one full-attention cache block
SLOT_RING_BYTES = SLIDING_SLOTS * 1152 * KV_ROW * 2 * 2   # sliding K / V rings of one slot (BF16): 189 MB
SLOT_STATE_BYTES = (48 * 32 * 2048 * 2  # attention K / V conv-input rings (BF16) of one slot: 6.3 MB
                    + 96 * 32 * 2304 * 4  # glue post-attention / post-MLP conv rings (FP32): 28.3 MB
                    + 4 * 200064)       # sampler history (DeviceSampler.hist[s], 4 B per vocabulary entry): 0.8 MB
# trace region of one BatchDecoder bucket R (main trace + the 4 sampling-kind tails + the logits untilize), measured by
# BatchStep with the final multi-row glue / attention / MoE (S 16, bfp4:gptq; logs/batch/batchstep/test_batch.json):
# independent of max_seq_len (the programs depend on R only); BatchDecoder captures nothing else.  A bucket between
# measured ones (S not a power of two) takes the next larger measured bucket's bytes; the plan adds BATCH_TRACE_MARGIN.
BATCH_TRACE_BYTES = {1: 5_799_936, 2: 7_307_264, 4: 9_076_736, 8: 11_501_568, 16: 11_902_976}
BATCH_TRACE_MARGIN = 1.25
# batched speculation (tt/spec.py BatchSpec; NOTES-spec.md "Batched speculation"): INKLING2_SPEC in batched serving =
# auto (default: the adaptive policy) | on (speculate whenever every sequence can) | off (no drafter)
SPEC_BUCKETS = (3, 6, 12, 16)           # extra BatchDecoder step buckets (verify rows: up to 16 at any max_num_seqs)
BATCH_SPEC_FIXED_GIB = 0.80             # drafter weights (layers BFP8, base BFP4, fold, Markov table) + block buffers
BATCH_SPEC_TRACE = 64 << 20             # drafter context (per bucket) / draft (users x drafts) traces
DRAFT_POOL_PER_TOKEN = 2 * 8 * 128 * 2  # drafter layer 1 K / V by vLLM block id: 4 KiB per pool position
DRAFT_RING_BYTES = 2 * 2 * 2048 * 8 * 128 * 2  # drafter layers 0 / 2 K / V rings of one slot (2048 positions): 16.8 MB
# prefix caching (NOTES-prefix.md): device snapshots of one slot's running state = its sliding K / V rings + K / V
# conv-input rings + glue conv rings (tt/prefix_cache.PrefixSnapshots; no sampler history): 223 MB each
# (PREFIX_SNAPSHOT_BYTES); with batched speculation each snapshot also holds the slot's drafter rings (BatchSpec
# registers Drafter.slot_tensors(): layers 0 / 2 K / V, PREFIX_DRAFTER_BYTES = DRAFT_RING_BYTES, 16.8 MB).  Count:
# INKLING2_PREFIX_SNAPSHOTS (auto: as many as leave PREFIX_MIN_HEADROOM_GIB headroom, at most PREFIX_MAX_SNAPSHOTS);
# INKLING2_PREFIX_CACHING=off -> 0
PREFIX_SNAPSHOT_BYTES = SLOT_RING_BYTES + 48 * 32 * 2048 * 2 + 96 * 32 * 2304 * 4
PREFIX_DRAFTER_BYTES = DRAFT_RING_BYTES
PREFIX_MAX_SNAPSHOTS = 16
PREFIX_MIN_HEADROOM_GIB = 1.0

PROFILES = (  # (name, max_seq_len bound, k, v, prefill chunk)
    ("fast", FAST_MAX, "bf16", "bf16", 4096),
    ("long", LONG_MAX, "bfp8", "bfp8", 4096),
    ("native", NATIVE_MAX, "bfp8", "bfp4", 4096),
)
MAX_PREFILL_CHUNK = 4096                # the MoE prefill route program's L1 limit (T = 8192 does not build)
PREFILL_TAIL = 512                      # tt/model.py prefill_schedule: chunks above this are multiples of it


class PlanError(ValueError):
    """The requested context / formats do not fit the P150 (message = the reason and the memory table)."""


@dataclass
class Plan:
    max_seq_len: int
    profile: str
    precision: str
    k_dtype: str
    v_dtype: str
    buckets: list
    trace_region: int
    embed: str
    attn_decode: str = "fused"  # "fused" (tt/fast/attn_decode, one bucket) | "stock" (attention.py ttnn path, buckets)
    memory_gib: dict = field(default_factory=dict)
    allocatable_gib: float = 0.0
    headroom_gib: float = 0.0
    moe_balance: bool = False  # Inkling2MoE(balance=...): bank-balanced decode gate/up streams (gate_up_dup tensors)
    attn_precision: str = DEFAULT_ATTN_PRECISION  # Inkling2Model(attn_precision=...)
    prefill_chunk: int = 512  # Inkling2Model(prefill_chunk=...): rows per prefill chunk (prefill_chunk_for)
    max_num_seqs: int = 1  # Inkling2Model(max_num_seqs=...): sequence slots (> 1: batched serving)
    kv_tokens: int = 0  # batched: vLLM's shared KV pool tokens (get_max_tokens_all_users); 0 = single stream
    kv_blocks: int | None = None  # batched: Inkling2Model(kv_blocks=...) = vLLM's num_gpu_blocks (kv_blocks_for)
    batch_buckets: list = field(default_factory=list)  # batched: BatchDecoder buckets (rows per decode trace)
    prefix_snapshots: int = 0  # Inkling2Model(prefix_snapshots=...): device snapshots for prefix caching (0 = off)
    prefix_snapshot_bytes: int = 0  # bytes of one snapshot (+ PREFIX_DRAFTER_BYTES with batched speculation)

    def table(self) -> str:
        rows = [f"  {k:<26s} {v:7.2f} GiB" for k, v in self.memory_gib.items()]
        return "\n".join(rows + [f"  {'allocatable':<26s} {self.allocatable_gib:7.2f} GiB",
                                 f"  {'headroom':<26s} {self.headroom_gib:7.2f} GiB"])

    def summary(self) -> str:
        s = (f"profile {self.profile}: max_seq_len {self.max_seq_len}, experts {self.precision}, attention "
             f"{self.attn_precision}, KV K {self.k_dtype} / V {self.v_dtype}, {self.attn_decode} attention decode, "
             f"{len(self.buckets)} decode buckets {self.buckets[0]}..{self.buckets[-1]}, trace region "
             f"{self.trace_region >> 20} MiB, embedding on the {self.embed}, MoE bank balance "
             f"{'on' if self.moe_balance else 'off'}, prefill chunk {self.prefill_chunk}")
        if self.max_num_seqs > 1:
            s += (f", batched: {self.max_num_seqs} sequence slots, KV pool {self.kv_tokens} tokens = "
                  f"{self.kv_blocks} blocks, batch buckets {self.batch_buckets}")
        if self.prefix_snapshots:
            s += (f", prefix snapshots {self.prefix_snapshots} x {self.prefix_snapshot_bytes / 2 ** 20:.0f} MiB"
                  + (" (incl. drafter rings)" if self.prefix_snapshot_bytes > PREFIX_SNAPSHOT_BYTES else ""))
        else:
            s += ", prefix caching off"
        return s


def decode_buckets(max_seq_len: int, smallest: int = MIN_BUCKET) -> list:
    """Full-attention decode buckets: powers of two from `smallest` below max_seq_len, then max_seq_len."""
    out, b = [], smallest
    while b < max_seq_len:
        out.append(b)
        b *= 2
    return out + [max_seq_len]


def batch_buckets(max_num_seqs: int) -> list:
    """BatchDecoder buckets (decode rows per trace): 1, 2, 4, 8, 16 up to max_num_seqs, plus max_num_seqs."""
    return sorted({b for b in (1, 2, 4, 8, 16) if b <= max_num_seqs} | {max_num_seqs})


def kv_blocks_for(kv_tokens: int, max_num_seqs: int) -> int:
    """vLLM's KV block count for a pool of kv_tokens (vllm_tt_plugin worker.get_num_available_blocks_tt: the model's
    get_max_tokens_all_users + one block of headroom per sequence, ceil / block size); block 0 is vLLM's null block."""
    return -(-(kv_tokens + KV_BLOCK * max_num_seqs) // KV_BLOCK)


def batch_spec_mode(env=os.environ) -> str:
    """Batched serving: INKLING2_SPEC auto (default) | on | off."""
    v = env.get("INKLING2_SPEC", "auto")
    if v not in ("auto", "on", "off"):
        raise PlanError(f"INKLING2_SPEC {v!r}: auto | on | off")
    return v


def batch_spec_gib(kv_blocks: int, max_num_seqs: int) -> float:
    """Device memory of batched speculation: the drafter, its layer-1 pool (one block per vLLM block + a scratch
    block) and its per-slot layer-0 / 2 rings."""
    return BATCH_SPEC_FIXED_GIB + ((kv_blocks + 1) * KV_BLOCK * DRAFT_POOL_PER_TOKEN
                                   + max_num_seqs * DRAFT_RING_BYTES) / GiB


def pool_kv_gib(kv_blocks: int, k: str, v: str) -> float:
    """The batched full-attention pool: kv_blocks + 1 (the zero block) blocks of 64 positions per full application."""
    return FULL_SLOTS * (kv_blocks + 1) * KV_BLOCK * KV_ROW * (BPV[k] + BPV[v]) / GiB


def batch_trace_bytes(bucket: int) -> int:
    """Trace region of one batch bucket (main + tails, BATCH_TRACE_BYTES; between measured buckets: the next larger)."""
    return BATCH_TRACE_BYTES[min(b for b in BATCH_TRACE_BYTES if b >= bucket)]


def expert_formats(precision: str) -> list:
    """Per-layer expert format of an Inkling2Model precision string (precision.py grammar; parsed here without torch)."""
    if precision.startswith("mixed:"):
        spec = json.loads(precision[len("mixed:"):])
        fmts = [spec.pop("default", "bfp8").partition(":")[0]] * N_LAYERS
        for key, val in spec.items():
            a, _, b = key.partition("-")
            for layer in range(int(a), int(b or a) + 1):
                fmts[layer] = val.partition(":")[0]
    else:
        fmts = [precision.partition(":")[0]] * N_LAYERS
    for f in fmts:
        if f not in ("bfp8", "bfp4"):
            raise PlanError(f"expert precision {precision!r}: formats bfp8 | bfp4")
    return fmts


def expert_gib(precision: str) -> float:
    """Expert DRAM of an Inkling2Model precision string."""
    return sum(EXPERT_VALUES * BPV[f] for f in expert_formats(precision)) / GiB


def attn_formats(attn_precision: str) -> list:
    """Per-layer {family: format} of an attention precision map (precision.attn_weight_map grammar, without torch)."""
    def fmt(spec):
        f = spec.partition(":")[0]
        if f not in ("bfp8", "bfp4"):
            raise PlanError(f"attention precision {attn_precision!r}: formats bfp8 | bfp4")
        return f

    def apply(cur, val):
        if isinstance(val, str):
            return {k: fmt(val) for k in ATTN_FAMILIES}
        return {**cur, **{k: fmt(s) for k, s in val.items()}}

    base = apply({}, "bfp8")
    if not attn_precision.startswith("mixed:"):
        return [apply(base, attn_precision)] * N_LAYERS
    spec = json.loads(attn_precision[len("mixed:"):])
    out = [apply(base, spec.pop("default", "bfp8"))] * N_LAYERS
    for key, val in spec.items():
        a, _, b = key.partition("-")
        for layer in range(int(a), int(b or a) + 1):
            out[layer] = apply(out[layer], val)
    return out


def attn_saving_gib(attn_precision: str) -> tuple:
    """(attention module tensors, fused decode stream copies) DRAM saved by the map's BFP4 families against all-BFP8
    (the measured OTHER_GIB / FUSED_DECODE_GIB): a tensor is BFP4 only if all its families are (module: q|gate, k, v,
    o; stream copies: q|gate and k|v as two groups when their formats differ, else one q|gate|k|v group, and o)."""
    def saved(groups, f):
        return sum(sum(ATTN_FAMILIES[x] for x in g) for g in groups if all(f[x] == "bfp4" for x in g))

    mod = stream = 0
    for f in attn_formats(attn_precision):
        mod += saved((("q", "gate"), ("k",), ("v",), ("o",)), f)
        two = {f["q"], f["gate"]} != {f["k"], f["v"]}
        stream += saved((("q", "gate"), ("k", "v"), ("o",)) if two else (("q", "gate", "k", "v"), ("o",)), f)
    d = (BPV["bfp8"] - BPV["bfp4"]) / GiB
    return mod * d, stream * d


def full_kv_gib(max_seq_len: int, k: str, v: str) -> float:
    return FULL_SLOTS * (max_seq_len + 64) * KV_ROW * (BPV[k] + BPV[v]) / GiB


def profile_for(max_seq_len: int):
    for p in PROFILES:
        if max_seq_len <= p[1]:
            return p
    raise PlanError(f"max_seq_len {max_seq_len} > the model's native context {NATIVE_MAX}")


def prefill_chunk_for(max_seq_len: int, env=None) -> int:
    """Prefill chunk rows of max_seq_len: env INKLING2_PREFILL_CHUNK (env=None: no override) or the profile's,
    clipped to the largest multiple of PREFILL_TAIL <= max_seq_len (a chunk never exceeds the context)."""
    v = (env or {}).get("INKLING2_PREFILL_CHUNK")
    chunk = int(v) if v else profile_for(max_seq_len)[4]
    if chunk % 32 or not 32 <= chunk <= MAX_PREFILL_CHUNK or (chunk > PREFILL_TAIL and chunk % PREFILL_TAIL):
        raise PlanError(f"prefill chunk {chunk}: a multiple of 32 up to {PREFILL_TAIL}, or of {PREFILL_TAIL} up to "
                        f"{MAX_PREFILL_CHUNK}")
    return max(min(chunk, max_seq_len // PREFILL_TAIL * PREFILL_TAIL), min(chunk, PREFILL_TAIL))


def plan(max_seq_len: int, precision: str | None = None, kv: str | None = None, embed: str | None = None,
         trace_region: int | None = None, moe_balance: str | None = None, attn_precision: str | None = None,
         env=os.environ, max_num_seqs: int = 1, kv_tokens: int | None = None,
         prefix_snapshots: str | int | None = None) -> Plan:
    """The serving plan for max_seq_len (see the module docstring); raises PlanError when it does not fit.
    max_num_seqs > 1: batched serving (NOTES-batch.md section 6) with a shared full-attention pool of kv_tokens
    (default DEFAULT_KV_TOKENS) + one block per sequence; max_num_seqs 1: the single-stream plan.  prefix_snapshots:
    "auto" | N (default env INKLING2_PREFIX_SNAPSHOTS, else auto; INKLING2_PREFIX_CACHING=off -> 0): the device
    snapshot pool of prefix caching, sized after everything else (auto: PREFIX_MIN_HEADROOM_GIB stays free)."""
    if not 1 <= max_num_seqs <= MAX_SEQS:
        raise PlanError(f"max_num_seqs {max_num_seqs}: 1..{MAX_SEQS}")
    batched = max_num_seqs > 1
    if batched:
        kv_tokens = DEFAULT_KV_TOKENS if kv_tokens is None else kv_tokens
        if kv_tokens < MIN_LEN:
            raise PlanError(f"KV pool {kv_tokens} tokens: at least {MIN_LEN}")
        spec_on = batch_spec_mode(env) != "off"
    if not MIN_LEN <= max_seq_len <= NATIVE_MAX or max_seq_len % 64:
        raise PlanError(f"max_seq_len {max_seq_len}: a multiple of 64 in [{MIN_LEN}, {NATIVE_MAX}]")
    name, _, p_k, p_v, _ = profile_for(max_seq_len)
    prefill_chunk = prefill_chunk_for(max_seq_len, env)
    precision = precision or env.get("INKLING2_PRECISION") or DEFAULT_PRECISION
    quality = precision == "quality"
    if quality:
        precision = QUALITY_PRECISION
    # quality = closest to the checkpoint: BFP8 experts AND BFP8 attention (all KL 0.0030 vs 0.0046 with q|gate BFP4,
    # logs/eval/final-{quality,qualityserve}.json) unless the attention map is given explicitly
    attn_precision = (attn_precision or env.get("INKLING2_ATTN_PRECISION")
                      or ("bfp8" if quality else DEFAULT_ATTN_PRECISION))
    attn_mod, attn_stream = attn_saving_gib(attn_precision)
    kv = kv or env.get("INKLING2_KV") or f"{p_k}:{p_v}"
    k, _, v = kv.partition(":")
    v = v or k
    if k not in BPV or v not in BPV:
        raise PlanError(f"KV formats {kv!r}: <k>:<v> with k, v in bf16 | bfp8 | bfp4")
    # attention decode: the fused path (tt/fast/attn_decode, AttnFast2: 2-level SDPA reduction) reads every K / V format
    # combination up to NATIVE_MAX positions; its cost follows the position -> one bucket.  INK_ATTN=stock: the stock
    # ttnn path with power-of-two decode buckets (one trace each).
    fused = env.get("INK_ATTN", "fast") != "stock"
    buckets = [max_seq_len] if fused else decode_buckets(max_seq_len)
    kv_blocks = kv_blocks_for(kv_tokens, max_num_seqs) if batched else None
    bbuckets = batch_buckets(max_num_seqs) if batched else []
    if batched and spec_on:
        bbuckets = sorted(set(bbuckets) | set(SPEC_BUCKETS))
    if trace_region is None:
        # single stream: TokenDecoder main + tails per decode bucket; batched: the BatchDecoder bucket traces x
        # BATCH_TRACE_MARGIN instead (+ the drafter's with batched speculation); TRACE_FIXED on top in both
        traces = (int(sum(batch_trace_bytes(b) for b in bbuckets) * BATCH_TRACE_MARGIN) if batched
                  else TRACE_PER_BUCKET * len(buckets))
        if batched and spec_on:
            traces += BATCH_SPEC_TRACE
        trace_region = -(-(TRACE_FIXED + traces) // (64 << 20)) * (64 << 20)
    # batched: the pool replaces the per-sequence full caches; every slot has its sliding rings; slots 1.. add their
    # conv / glue rings and sampler history (slot 0's are in OTHER_GIB); batch tables: page table [32, W] + prefill row
    full = (("full KV pool", pool_kv_gib(kv_blocks, k, v)) if batched
            else ("full KV", full_kv_gib(max_seq_len, k, v)))
    mem = {
        "experts": expert_gib(precision),
        "other weights": OTHER_GIB - attn_mod,
        "fused decode copies": FUSED_DECODE_GIB - attn_stream if fused else 0.0,
        full[0]: full[1],
        "sliding KV": max_num_seqs * SLOT_RING_BYTES / GiB,
        "decode tables": (sum(b + 128 for b in buckets) * 4 + max_seq_len * 4
                          + (33 * (max_seq_len // KV_BLOCK) * 4 if batched else 0)) / GiB,
        "prefill reserve": PREFILL_RESERVE_GIB,
    }
    if batched:
        mem["slot states"] = (max_num_seqs - 1) * SLOT_STATE_BYTES / GiB
    if batched and spec_on:
        mem["batched speculation"] = batch_spec_gib(kv_blocks, max_num_seqs)
    elif not batched and env.get("INKLING2_SPEC", "off") == "on":
        # Spec2: drafter + Markov table + fold + verify buffers, and the drafter's 3-layer BF16 K / V per position
        mem["speculative decoding"] = SPEC_FIXED_GIB + max_seq_len * SPEC_KV_PER_TOKEN / GiB
    alloc = DRAM_TOTAL_GIB - trace_region / GiB - ALLOC_SLACK_GIB
    need = sum(mem.values())
    embed = embed or env.get("INKLING2_EMBED", "auto")
    if embed == "auto":
        embed = "device" if alloc - need - EMBED_GIB >= EMBED_RESERVE_GIB - PREFILL_RESERVE_GIB else "host"
    if embed == "device":
        mem["embedding (device)"] = EMBED_GIB
    elif embed != "host":
        raise PlanError(f"embedding placement {embed!r}: auto | device | host")
    # MoE bank balance after the embedding: the dup tensors only take room the plan does not otherwise use
    balance_gib = expert_gib(precision) * MOE_BALANCE_FRACTION
    moe_balance = moe_balance or env.get("INKLING2_MOE_BALANCE", "auto")
    if moe_balance == "auto":
        moe_balance = "on" if alloc - sum(mem.values()) - balance_gib >= BALANCE_MIN_HEADROOM_GIB else "off"
    if moe_balance == "on":
        mem["MoE balance copies"] = balance_gib
    elif moe_balance != "off":
        raise PlanError(f"MoE balance {moe_balance!r}: auto | on | off")
    # prefix caching snapshots last: they take only room nothing above uses.  Batched speculation: each snapshot also
    # holds one slot's drafter rings (BatchSpec registers them with the pool)
    snap_bytes = PREFIX_SNAPSHOT_BYTES + (PREFIX_DRAFTER_BYTES if batched and spec_on else 0)
    snap_gib = snap_bytes / GiB
    snaps = prefix_snapshots if prefix_snapshots is not None else env.get("INKLING2_PREFIX_SNAPSHOTS", "auto")
    if env.get("INKLING2_PREFIX_CACHING", "on") == "off":
        snaps = 0
    elif snaps == "auto":
        snaps = max(0, min(PREFIX_MAX_SNAPSHOTS, int((alloc - sum(mem.values()) - PREFIX_MIN_HEADROOM_GIB) // snap_gib)))
    elif not str(snaps).isdigit():
        raise PlanError(f"prefix snapshots {snaps!r}: auto | a count")
    snaps = int(snaps)
    if snaps:
        mem["prefix snapshots"] = snaps * snap_gib
    p = Plan(max_seq_len, name, precision, k, v, buckets, trace_region, embed, "fused" if fused else "stock",
             {key: round(val, 3) for key, val in mem.items()}, round(alloc, 3), round(alloc - sum(mem.values()), 3),
             moe_balance == "on", attn_precision, prefill_chunk, max_num_seqs, kv_tokens if batched else 0, kv_blocks,
             bbuckets, snaps, snap_bytes if snaps else 0)
    if p.headroom_gib < 0:
        hint = ("a smaller --max-model-len, or 4-bit experts / KV (INKLING2_PRECISION, INKLING2_KV)" if not batched
                else "fewer sequences (--max-num-seqs), a smaller KV pool (INKLING2_MAX_TOKENS_ALL_USERS), or 4-bit "
                     "experts / KV (INKLING2_PRECISION, INKLING2_KV)")
        raise PlanError(f"{p.summary()} does not fit the P150 ({-p.headroom_gib:.2f} GiB short):\n{p.table()}\n"
                        f"use {hint}")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--precision")
    ap.add_argument("--kv")
    ap.add_argument("--moe-balance", choices=("auto", "on", "off"))
    ap.add_argument("--max-num-seqs", type=int, default=1, help="sequence slots (> 1: batched serving)")
    ap.add_argument("--kv-tokens", type=int, help=f"batched: shared KV pool tokens (default {DEFAULT_KV_TOKENS})")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON (launcher)")
    ap.add_argument("--prefix-snapshots", help="prefix caching snapshots: auto | N (0 = prefix caching off)")
    a = ap.parse_args(argv)
    try:
        p = plan(a.max_model_len, a.precision, a.kv, moe_balance=a.moe_balance, max_num_seqs=a.max_num_seqs,
                 kv_tokens=a.kv_tokens, prefix_snapshots=a.prefix_snapshots)
    except PlanError as e:
        print(f"inkling2 memory plan: {e}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(p)) if a.json else f"{p.summary()}\n{p.table()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
