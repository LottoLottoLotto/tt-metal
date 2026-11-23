# Qwen hybrid models on Blackhole

This isolated workspace adds **Qwen3.8-27B on one P150** to the inherited
hybrid Gated DeltaNet / Gated Full Attention runtime. The upstream configurations are:

| Model            | `HF_MODEL`             | Mesh / `MESH_DEVICE` | Parallelism            |
| ---------------- | ---------------------- | -------------------- | ---------------------- |
| Qwen3.5-9B       | `Qwen/Qwen3.5-9B`      | single P150 — `P150` | single device          |
| Qwen3.5-27B      | `Qwen/Qwen3.5-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x8 — `P150x8`    | 8-way tensor parallel  |
| Qwen3.8-27B      | `Qwen/Qwen3.8-27B`     | single P150 — `P150` | single-device native checkpoint |

- The **9B** runs on a **single Blackhole P150** device. It uses the validated
  single-device forward path (no collectives).
- The **27B** variants (both Qwen3.5-27B and Qwen3.6-27B) run on a **P150x4**
  (a `(1, 4)` Blackhole mesh) using **4-way tensor parallelism (TP)**. The TP
  path needs `FABRIC_1D` for the cross-device collectives (all-reduce /
  reduce-scatter) and a trace region for the captured chunk-outer prefill trace.
- **Qwen3.6-27B additionally runs at TP=8** on a `(1, 8)` mesh (`P150x8`).
  Because it has only **4 KV heads**, TP=8 cannot give each device its own head:
  each head is instead **replicated across the device pair holding its GQA query
  group** (devices 0-1 share KV head 0, 2-3 head 1, and so on), so
  `n_local_kv_heads` is 1 at both TP=4 and TP=8 and the whole runtime KV path is
  unchanged. See `tp_common.replicate_kv_weight` and
  `ModelArgs.SUPPORTS_KV_REPLICATION`.

Everything model-specific (hybrid layer dispatch, DeltaNet head/conv dims,
partial rotary factor, vocab, layer count) is read from the parsed HF config, so
the single code base adapts to each checkpoint. The device count alone
(`num_devices > 1`) switches between the single-device and TP code paths — see
`tt/model_config.py` and `tt/tp_common.py`.

## Qwen3.8-27B optimized single-device execution

`QWEN36_OPTIMIZED_27B=1` (with `QWEN36_GENERIC_BASELINE=0`) selects the complete
sealed-9B optimization family at 27B geometry: packed BF4 gate/up + native SwiGLU,
packed BF4 down, one canonical packed BF8 GDN mega projection shared by prefill and
decode, packed BF8 LM head, fused BF8 QKV / SDPA, bounded width-sharded RMSNorms,
native BF16 L1-resident GDN state (48 layers, 749,568 B/core), native
frontend/recurrence/output kernels, shared wide target verification, optional MTP
or native DFlash 2, and trace-mode-all serving with two-phase warmup.
`tools/tt_eval.py --execution optimized` and `tools/serve_native.py` own the full
switch list; do not hand-assemble flags. Unsupported geometry fails rather than
falling back. The evaluator's `--execution generic` and the older workspaces remain
separate rollback paths; the FP32 `native27`/`stage27` kernels were removed.

Per-core L1 admission is derived, not measured-and-hoped: the pinned HAL origin
(111,488 B) plus the largest static circular-buffer program (packed BF8 LM head at
K-block 1, 594,816 B; the sequential GDN prefill scan is 593,920 B after its CB
aliasing) must leave at least 61,440 B beside the resident state. Programs that
cannot satisfy this on the 27B (`concatenate_heads` at H=24/D=256, L1 prefill
scratch, the framework's height-sharded norm on the two-row MTP verify input) take
interleaved/DRAM/bounded-sharded paths under the optimized flag.

Prefill over the canonical packed weights runs the 2D multicast matmul
(`matmul_multicore_reuse_mcast_2d_program_factory.cpp` overlay): the in1 sender
translates logical tiles to bank-packed pages and each N slab is read once and
multicast to every M row-block. The former streamed-N 1D reader re-read the whole
payload per 32-row block. Block sizes in `mlp._packed_prefill_program_config` and
`_packed_mega_prefill_program_config` are bounded to fit beside the resident state;
`fused_horner.supports` admits the 27B's 3072 diagonal blocks per 2048-token chunk.

M=2 GDN verification uses a separate 33-core BF8 output-projection program with
K-block 2 (`out_verify_program_config`); M=1 retains its existing K-block 8.
Reusing M=1's program for M=2 was faster but introduced rare projection rounding
differences and recurrent-state drift over repeated traces. K-block 2 passed exact
target/draft-logit and committed-state comparisons for accepted/rejected drafts,
including the 127→128 KV-page boundary, and repeated 20-cycle trace comparisons.
The final resident-model A/B measured 70.825→69.180 ms per MTP cycle (2.32% lower
latency), not API token throughput. See `../../probes/m2_projection_ab.py` and
`../../evidence/m2-projection-final.json`.

The packed BF8 27B mega projection `[5120,16512]` uses K-block 2 for M=1 and
M=2 decode, retaining the 11x10 grid, six output tiles per core, subblock width 2,
and two reader buffer slots. Other shapes/dtypes retain K-block 4. A resident-model
sweep rejected larger buffers, alternate subblocks, and K-block 8 on performance.
The final interleaved K-block-4/2 A/B measured 69.181→68.925 ms per MTP cycle
(0.37% lower latency), with the M=2 output-projection optimization already active.
M1 full logits, accepted/rejected MTP logits and committed states, the 127→128
page-boundary cases, and repeated 20-cycle traces matched exactly in the probe.
See `../../evidence/mega-projection-final.json` and
`../../probes/mega_projection_ab.py`; these short synthetic checks are not a
long-context quality certification or an API-throughput measurement.

The canonical environment in `tools/tt_eval.py` selects three buffer slots for
packed BF4 MLP down decode, retaining one outstanding read, K-block 8, and output
subblock width 2. The interleaved resident-model confirmation measured
68.9234→68.9012 ms/MTP cycle (0.032% lower latency), with exact full logits and
committed states in the exercised cases. This is a minor trace-level gain, not
a measured API-throughput increase. The original 2–4-outstanding-read path failed
projection parity because its producer reserved only one block despite pending
unpublished DMA reads. Reserving the entire unpublished window fixes the overwrite:
all 15 formerly failing settings now pass, including full MTP logits and states.
Multi-read execution remains disabled by default because none beat three slots /
one read in the corrected sweep (68.894 ms versus best multi-read 68.908 ms).
See `../../evidence/down-pipeline-optimization.json` and the hardware regression
probe `../../probes/down_pipeline_sweep.py`. The kernel fix is source-validated,
not packaged/deployed; the shared 9B path was not exercised in this 27B sweep.
See also `../../evidence/down-projection-optimization.json`. Existing environment-bound
checkpoint proofs predate the three-slot setting and must be regenerated.

The native frontend and M=2 recurrence reader clear their padded operand tiles
with synchronized NoC zero-fill instead of CPU store loops. The zero-fill barrier
completes before operand reads; full-tile padding, BF16 arithmetic boundaries,
layouts, and CB allocations are unchanged. The resident-model interleaved A/B
measured 68.916 ms/MTP cycle before, 68.145 ms with frontend-only, 68.474 ms with
recurrence-only, and 67.537 ms with both changes (2.00% lower latency).
All three candidates matched full M1 logits/states and accepted/rejected MTP
target/draft logits and committed states at positions 120 and 127; twelve
20-cycle trace comparisons were exact. The full-state native regression passed
all six stages and 5,396 comparisons, including 32-step recurrence/verification.
See `../../evidence/gdn-zero-fill-optimization.json` and
`../../probes/gdn_zero_fill_ab.py`; the A/B reference kernels are retained under
`../../evidence/gdn-kernel-baseline/`. These are short synthetic hardware checks,
not an API-throughput measurement or long-context quality certification.

All retained decode optimizations are packaged and HTTP-benchmarked in the separate
local image `qwen38-27b-tt:integrated-decode-20260909`, with fresh proofs under
`../../checkpoint-integrated/`. The original published image tag and original
`../../checkpoint/` proofs remain unchanged. See
`../../evidence/integrated-decode-summary.json` for the immutable image identity,
configuration, validation results, and benchmark scope.

On one P150, three-repeat streamed API medians with 512 actual completion tokens
were 25.53 tok/s for technical prose, 25.09 for Python, and 25.35 for a numbered
list with MTP on; MTP off sustained about 15.70 tok/s. The existing 128-token
benchmark improved from the previous image's recorded 23.14 to 24.28 tok/s.
A synthetic single-word repetition reached 29.21 tok/s at 99.2% draft acceptance;
a three-word repetition reached only 17.71 tok/s at 20.5% acceptance. Neither is
a general task-throughput claim. Repeats are deterministic within each mode;
MTP-on and MTP-off do not generate bit-identical completions.

Both modes passed long-context retrieval and request isolation. The behavior
suite passed 10/11 in each mode: the JSON-only answer still includes Markdown
fences, exactly as in the previous-image artifacts. This is a known failure,
not an all-pass quality certification. Reproduction uses `../../tools/api_validate.py`
and `../../probes/integrated_decode_benchmark.py`; benchmark servers were stopped.

The subsequent irregular packed-reader rewrite supports bounded 2..4-read
transaction-ID lookahead for gate/up, GDN mega and LM head. It reserves all
unpublished DMA blocks before issuing another read and publishes completed blocks
in K order. The existing single-read path and serving defaults remain unchanged.
Hardware sweeps passed exact M1/M2 projections, full-model M1 logits/state,
accepted/rejected MTP target/draft logits/state, and 113 repeated full-cycle trace
state checks. These used a deterministic 120-token prompt, not a quality corpus.

Only LM-head lookahead improved full-cycle timing: five CB slots/four reads
confirmed 67.551 to 67.429 ms per MTP cycle (0.18%). Gate/up and GDN mega
multi-read configurations were slower; enabling five slots/four reads for all
three families cost 69.057 ms. The rewrite remains opt-in, not a serving default
or an HTTP-throughput claim. The kernel-tested image
`qwen38-27b-tt:irregular-pipeline-tested` contains the measured runtime plus
updated source-provenance metadata. Existing integrated-image reload proofs do
not certify this new image or nondefault environment.

See `../../evidence/irregular-pipeline-optimization.json` for exact commands,
image identities, build recipe and measurements. Reproduction uses
`../../probes/irregular_pipeline_sweep.py --families lmhead --configs 5:4`
with its required weights/cache/output/device-ownership arguments. The underlying
opt-in settings are `QWEN36_LM_HEAD_CB_SLOTS=5` and
`QWEN36_LM_HEAD_OUTSTANDING=4`; gate/up and GDN mega should remain at 2/1.

The dual-NoC runtime enables both NoCs for eligible Blackhole K160 packed gate/up,
GDN mega, and LM-head projections with one physical output tile row (M=1..32).
It retains deployed weight formats, arithmetic, K-block sizes, buffer slots, and
the single-outstanding-read default. Each weight read is partitioned into disjoint,
64-byte-aligned ranges; both NoCs complete before the block is published.
Both data-movement processors use dynamic NoC mode. Activation multicasts are
unlinked, with data acknowledgements before readiness flags; sharded senders also
acknowledge loopback flags before reusing their semaphore. Linked multicast cannot
coexist with another RISC's unicast reads on the same NoC.

`QWEN36_PACKED_DUAL_NOC` is a descriptor-time control: unset selects
`gateup,mega,lm_head,qkv`, `0` disables it, and a comma-separated, duplicate-free subset
selects individual families. Whitespace, empty tokens, and unknown names are
rejected. Set it before creating programs and use fresh processes for A/B runs;
existing program caches do not support changing the policy in place. Other
geometries, streamed/2D prefill, and the down projection retain their existing
paths. This requires rebuilding the compiled host factory, not merely copying JIT
sources; `runtime/build_serving_chain.py` now performs that rebuild.
With `QWEN36_OPTIMIZED_27B=1` the fused attention QKV weight ([5120,14336] BFP8) is
stored in the same rotating worker-bank page order (90 workers x 5 columns, repacked at
load from the checkpoint tensor; quantization is tile-local) and M=1 programs use the
dual-NoC bank reader: 207 -> 176 us per projection, bitwise-equal outputs. Prefill and any
other program read it through the factory page map.

Wide MTP verification (T=3/4) runs RoPE for every row in one launch
(`attention/rope_kv_many_kernel.cpp`) and writes the K/V head rows straight into the
paged caches, replacing T RoPE launches plus 2T `paged_update_cache` calls.

The MTP draft FC and draft LM head are quantized for the draft only (FC BFP8, head BFP4;
`QWEN36_MTP_DRAFT_FC_DTYPE=bf16` / `QWEN36_MTP_DRAFT_HEAD_DTYPE=bfp8` restore the previous
formats). Verification stays full-vocabulary BF8, so the emitted greedy stream is unchanged.

`../../evidence/dual-noc-results.json` records the immutable runtime, byte-exact
projection/reload checks, and matched serving measurements. Projection bandwidth
improvements are not by themselves an end-to-end decode speedup.

Serving requires fresh source/environment-bound checkpoint reload evidence for the
selected MTP mode; do not reuse a proof from another image or disable
`validate_runtime`. `../../checkpoint-integrated/equivalence.json` (MTP off) and
`equivalence-mtp.json` (MTP on) prove exact full-logit reload parity across 95
teacher-forced positions against original weights under identical runtime sources,
overlay sources, compiled TTNN libraries and environment. This is serialization
equivalence, not equivalence to generic execution and not a quality certification.

Kernel-level evidence: `../../probes/native_gdn_smoke.py` (frontend, recurrence,
output, fused verify/commit, sequential scan; optional `--full-state` residency),
`../../probes/packed_smoke.py` (every packed family at M=1/2/128/256 with trace
replay), `../../probes/prefill_warmup_probe.py` (serving warmup outside vLLM).
Measured serving behavior, throughput against the control, memory bounds and the
immutable image ID are recorded in `../../optimization-results.json`.

### Native DFlash 2 serving

The separate `method=dflash` path runs the complete trained five-layer BF16
drafter, grouped dynamic convolutions, full-vocabulary top-16 candidate search,
and learned predecessor-conditioned selector on P150. It shares target
verification with MTP; it is not repeated MTP or CPU-backed drafting.
Five committed target-feature taps feed a 2048-slot draft KV ring. Noisy draft KV
never enters persistent context.

From the workspace root, use the certified image and checkpoint:

```bash
python3 tools/serve_native.py \
  --checkpoint checkpoint-serving-dflash \
  --cache-root dflash-serving-cache \
  --image sha256:c74a0dabb3774f04f8adc73fce08b5987ae65198b8e8e9ea239ad3bb245c01bf \
  --dflash dflash-draft --dflash-block-size 8 \
  --device-ownership-confirmed
```

The local tag is `qwen38-27b-tt:dflash-serving`; the immutable image argument above
also reuses the measured B8 cache. Block size includes the anchor; `4` is supported.
The launcher selects MTP-off reload evidence and mounts the draft read-only.
DFlash supports one P150, one active sequence, and a configured context of
2048–8192 tokens; 8192 was exercised. Native speculation is greedy-only.
Noneligible sampling, incomplete budgets, and context tails use ordinary TT
decode, never host drafting. Prefix caching, scheduler-chunked prefill, remote KV,
and async scheduling are disabled; internal target chunk replay remains enabled.

Both DFlash sizes and the source-matched MTP1 control passed all 53 HTTP lifecycle
scenarios plus 11 workload/context-tail requests each. Native rollback matched
every acceptance index across six cases, including 47 full-state checkpoints.
A separate actual-vLLM B8 run forbade post-warmup compilation: all 15 requests
retained 1,078 program-cache entries and identical allocator snapshots.

B8 measured 38.76 / 43.45 / 35.91 HTTP decode tokens/s on the code, reasoning-prose,
and 3037-token retrieval fixtures, versus MTP1's 26.73 / 26.72 / 23.02.
It was slower on the separate API-suite prose prompt: 20.41 versus 25.68 tokens/s.
Keep MTP1 as the default; DFlash is opt-in, not a universal speedup.
The previous MTP image/checkpoint remain untouched. See
`../../probes/dflash/PLAN.md`, `../../probes/dflash/serving-command.json`, and
`../../evidence/dflash-serving-summary.json` for full timings, limits, source
bindings, regression evidence, and failed attempts.

#### Opt-in two-path B8 verification

`--dflash-branches 2` produces two alternatives from one drafter backbone pass.
The second path shares the first path's prefix, chooses the second-ranked learned
candidate at a fixed fork, then follows its own predecessor-conditioned suffix.
A packed target tree evaluates the common prefix once, selects the path with
greater acceptance (ties prefer the first), and commits only that path's
recurrent state, KV, and features. No second resident L1 recurrent state is needed.

```bash
python3 tools/serve_native.py \
  --checkpoint checkpoint-serving-dflash-branches \
  --cache-root dflash-branches-serving-cache \
  --image sha256:2072038b6551a2b20e21e26564ccda29bf653b05c3d6c27d4edeb85c5222e596 \
  --dflash dflash-draft --dflash-block-size 8 \
  --dflash-branches 2 --dflash-fork-position 6 \
  --device-ownership-confirmed
```

Branches default to `1`; two paths require B8. Fork positions are `1..7`
(default `3`), fixed per warmed server and isolated in the cache identity.
The tree has `16 - fork_position` target rows but still reserves at most eight
sequence positions. Existing sampling, stop, reset, and context-tail rules apply.
The launcher rejects branch mode with the older single-path checkpoint proof.

All seven forks pass exact target/state and rollback checks: 252 traced commits
and 62,576,640 logit elements. Twelve real learned-branch/control cases preserve
1,536 frozen-oracle output tokens. Single B8, fork 3, and fork 6 each pass
53 HTTP lifecycle scenarios plus 11 workload/context-tail requests.

A separate real-vLLM fork-6 run forbids post-warmup program-cache misses.
All 15 requests retain 1,081 entries and identical DRAM/L1/TRACE allocator
snapshots. These snapshots are not execution peaks.

**Keep single-path B8 for performance.** Median HTTP decode tokens/s for
code / reasoning prose / retrieval are `38.77 / 43.46 / 35.92` for single B8,
`34.47 / 40.66 / 31.40` for fork 3, and `36.10 / 40.50 / 34.44` for fork 6.
Both branch configurations also lose on the separate API-suite prose and code
prompts. Extra acceptance does not repay extra verification work on these inputs.
The previous DFlash and MTP images/checkpoints remain unchanged.
See `../../probes/dflash/branch-command.json` and
`../../evidence/dflash-branches-summary.json` for source-bound evidence and limits.

## Architecture

Assembly: `tok_embeddings → N × Qwen36DecoderLayer → RMSNorm → LM Head`.

Each model interleaves two attention block types (read from the HF
`layer_types`): **Gated DeltaNet** (linear-attention, recurrent + causal conv
state) layers and **Gated Full Attention** (paged KV cache) layers. The 9B has
32 layers (24 DeltaNet + 8 full-attention). Qwen3.5 uses zero-centered RMSNorm
everywhere and **partial** RoPE (only a fraction of each head is rotated).

## Environment setup

Before running **any** test, export the two environment variables that select
the checkpoint and the device mesh.

**9B (single P150):**

```bash
export HF_MODEL=Qwen/Qwen3.5-9B
export MESH_DEVICE=P150
```

**27B (P150x4):**

```bash
# Qwen3.6-27B
export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=P150x4

# …or Qwen3.5-27B
export HF_MODEL=Qwen/Qwen3.5-27B
export MESH_DEVICE=P150x4
```

`HF_MODEL` is the single source of truth for the checkpoint — it may be a Hugging
Face hub id (resolved via `snapshot_download`) or a local checkpoint directory.
`MESH_DEVICE` selects the mesh shape (`P150` → `(1,1)`, `P150x4` → `(1,4)`,
`P150x8` → `(1,8)`).

Optional flags:

```bash
# Run SDPA in BF8 (faster; slightly lower precision).
export QWEN_SDPA_BF8=1
```


## End-to-end demo test (`demo/text_demo.py`)

The e2e text-generation test lives in `demo/text_demo.py`. It is a single
parametrized test (`test_demo_text`) covering a range of input sequence lengths
(ISLs): 128, 4k, 8k, 16k, 32k, 64k, 128k, and 256k tokens. Each ISL runs prefill
+ decode and validates output (non-degenerate generation) and per-ISL
performance gates (TTFT and decode tok/s).

Two execution variants exist per ISL, identified by the test id prefix:

- **`traced_*`** — captures the prefill (chunk-outer) and decode forward passes
  as device traces and replays them. This is the **preferred** path and the one
  vLLM serves; run these by default.
- **`paged_*`** — non-traced paged path, useful as an eager reference/fallback.

Run the preferred traced cases (the env vars above must already be exported):

```bash
# All traced ISLs
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced"

# A single ISL, e.g. the short 128-token traced case
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128"

# Medium / long traced ISLs
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_4k"
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_64k"
```

The **same command works for both 9B and 27B** — only the exported `HF_MODEL` /
`MESH_DEVICE` differ. On a single device the test takes the validated 9B path; on
the `(1,4)` mesh it routes through the TP chunk-outer traced prefill + paged
traced decode path automatically.

> Long-context cases (64k+) download a public-domain corpus (Frankenstein, War
> and Peace) on first run and cache it under `demo/sample_prompts/.context_cache`.

## Tests

There are two tiers of tests under `tests/`.

### Single-device unit / component tests — **9B**

These run on a single P150 against the 9B checkpoint (the conftests
`setdefault HF_MODEL=Qwen/Qwen3.5-9B`). They validate each component's forward
pass against a torch reference (PCC) — see `tests/pcc_thresholds.json` for the
per-test thresholds.

`tests/unit/` (component PCC vs torch):

| Test                       | Validates                                            |
| -------------------------- | ---------------------------------------------------- |
| `test_embedding.py`        | token embedding                                      |
| `test_rms_norm.py`         | zero-centered RMSNorm (the "+1" fold)                |
| `test_rope.py`             | partial-rotary RoPE (host freqs + on-device lookup)  |
| `test_mlp.py`              | single-device SwiGLU MLP (layer 0)                   |
| `test_attention.py`        | single-device gated full attention (layer 3)         |
| `test_gdn.py`              | single-device Gated DeltaNet (layer 0)               |
| `test_lm_head.py`          | LM head logits (bf8 vs bf16)                          |
| `test_layer.py`            | full decoder-block sanity (no NaN/Inf, non-constant) |
| `test_model.py`            | Generator decode contract: traced vs paged decode    |
| `test_substate.py`         | weight `substate` helper (pure CPU, no device)       |

`tests/` (single-device, also 9B):

| Test                     | Validates                                                  |
| ------------------------ | ---------------------------------------------------------- |
| `test_prefill.py`        | masked fixed-bucket + chunk-outer prefill vs `prefill_paged` |
| `test_weight_mapping.py` | HF → internal weight key remapping (pure CPU)              |

Run the 9B unit suite (with `HF_MODEL=Qwen/Qwen3.5-9B`, `MESH_DEVICE=P150`):

```bash
pytest models/demos/blackhole/qwen36/tests/unit/ -v -s
pytest models/demos/blackhole/qwen36/tests/test_prefill.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_weight_mapping.py -v -s
```

> `test_prefill.py` auto-skips cases longer than `--max-prefill` (default 8192).
> Raise it to exercise long-context prefill, e.g. `--max-prefill 131072`.

### Tensor-parallel tests — **27B (P150x4)**

The `*_tp` tests exercise the multi-device TP path and default to the 27B
checkpoint. They must run on the `(1,4)` mesh with `FABRIC_1D` (the
`parametrize_mesh_tp` helper wires this from `MESH_DEVICE`). PCC thresholds are in
`tests/pcc_thresholds.json`.

| Test                  | Validates                                                            |
| --------------------- | ------------------------------------------------------------------- |
| `test_mlp_tp.py`      | TP SwiGLU MLP (column/row-parallel + reduce-scatter)                |
| `test_attention_tp.py`| TP gated full attention: decode / prefill / paged-KV contract       |
| `test_gdn_tp.py`      | TP Gated DeltaNet: decode + chunk-prefill                           |
| `test_model_tp.py`    | full-model TP contract: paged+traced path matches the bespoke oracle |
| `test_generate_tp.py` | full-model bespoke `generate_tp` on a real prompt (answer oracle)   |

Run the 27B TP suite (with `HF_MODEL=Qwen/Qwen3.6-27B` or `Qwen/Qwen3.5-27B`,
`MESH_DEVICE=P150x4`):

```bash
pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_model_tp.py -svq
pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py -v -s
```

> `test_substate.py` and `test_weight_mapping.py` are pure-CPU and need no device.
> `test_weight_mapping.py`'s shape constants assume the 9B checkpoint.

### Decode matmul data movement (DFlash4)

Set by `tt/__init__.py` when `QWEN36_OPTIMIZED_27B=1` (each overridable; all read by the rebuilt 1D
matmul factory at program creation, K blocking and accumulation order unchanged, outputs bitwise equal):

- `QWEN36_IN0_MCAST_CHUNK_TILES=32`: for one interleaved M tile row, the activation sender reads and
  multicasts up to 32 K tiles (a whole number of K blocks) per semaphore handshake into a two-chunk CB;
  the compute kernel still consumes `in0_block_w` tiles per K block. LM head excluded. `0` disables.
- `QWEN36_MLP_DOWN_DUAL_NOC=1`: bank-stripe packed MLP down splits each contiguous K-block read into two
  DRAM-aligned halves, one per NoC (dynamic NoC mode).
- `QWEN36_PROJ_O_ROTATING_COLS=2`: GDN out and attention o_proj ([6144,5120] BF8) stored in the rotating
  worker-bank page order (80 workers x 2 columns, repacked at load, cache suffix `.packed-worker-bank.o2`)
  and read by the dual-NoC worker-bank reader; other programs read it through the factory page map.

T=8 microbench (traced, µs): GDN out 90.5 -> 79.3, attention o_proj 91.2 -> 78.2, MLP down 126.4 -> 120.3,
gate/up 216.1 -> 210, GDN mega 206.2 -> 201.3.
