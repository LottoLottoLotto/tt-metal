<!-- SPDX-FileCopyrightText: © 2026 Lottolabs
     SPDX-License-Identifier: Apache-2.0 -->

# tt/fast — decode hot-spot replacements (owner: LfmKern)

Standalone drop-in modules for the 1-row traced decode step; LfmTT integrates them into model.py / moe.py / layers.py.
Tests: `src/lfm2/tests/fast/` (`tools/devrun.sh NAME -m models.demos.blackhole.lfm2.tests.fast.<test>`); helpers in
`tests/fast/harness.py` (Env, `rep_ms` = 16 calls per captured trace, device time per call; `traced_ms` = 1 call per
trace, host-enqueue bound below ~60 us), `tests/fast/devprof.py` (device-profiler per-program / per-zone durations).
Weights: GPTQ checkpoint (cache q-ckpt-b947979e32).

## 0. Whole step (tests/fast/fast_step.py, pos 2000, traced, logs/fast_step.json / logs/dev/f_step3.log, load 1-2)

| step variant | ms/token | tok/s |
|---|---|---|
| stock `Lfm2Model.decode_forward` | 17.54 | 57 |
| + MoEDecode | 13.45 | 74 |
| + ResidualRMSNorm chain | 11.70 | 85 |
| + ConvDecode tail | 11.14 | 90 |
| + AttnDecode front | 10.01 | 100 |
| + StreamDecodeMatmul (conv/attn proj, dense MLP, LM head) | **6.67** | **150** |

Correctness (16 teacher-forced steps after a 2000-token prefill, same tokens through stock and fast step): top-1
16/16, step-0 logits PCC 0.99991 (max abs 0.18); later steps drift to PCC 0.98-0.999 because both paths feed their own
KV / conv state forward (numerics-level differences compound; no step disagreed on the argmax).

Remaining gap to the ~4 ms weight-streaming roofline (~1.9 GB/token at ~470 GB/s): ~2.7 ms.  Per token now (from
the module numbers): MoE 38 x ~73 us = 2.8 ms (DRAM floor ~1.7), conv 30 x ~(32+13+8) = 1.6, attention 10 x
~(17+16+31+13) = 0.8, LM head 0.57, dense MLP 0.38, norms 81 x 5 = 0.4.  Most small ops are now ~50% fixed launch
cost (~5-7 us per generic_op).

## 1. MoE decode: `moe_decode.MoEDecode` (kernels/moe_decode_experts.cpp, kernels/sigmoid_math.hpp, sigmoid_table.py)

Two launches per layer, one token row: router logits (StreamDecodeMatmul, bitwise ttnn.linear) -> one program with
two roles: gate/up workers (3 per DRAM bank) and down workers (2 per bank, row neighbours).
- Routing on every worker's NCRISC (no extra launch), bitwise sigmoid_routing.cpp: with flags 3 (default) the score of
  a BF16 logit is read from a host-tabulated exact table (sigmoid_table.py emulates the integer arithmetic; constant
  ranges asserted for every BF16 magnitude); other flags use an exact Newton floor division + a provably-safe
  approximate first pass.  12.9 us (libgcc 64-bit division) -> 4.25 us.
- Gate/up: unit = (expert, 2 gate/up column pairs) = 4 FP32 DEST tiles (LFM2's 12 tiles per bank do not fit DEST),
  transposed custom_mm (LoFi, exact for BFP4 x BF16), SiLU(g)*u*w on the pack thread, [1,32] activation tiles to L1;
  each unit increments a per-expert semaphore on every down worker.
- Down workers prefetch their 4 experts' column slices while gate/up runs, start each expert when its 24 units are
  in, accumulate all experts in FP32 DEST, one BF16 rounding.
- Same resident expert tensors as moe.MoE; extra DRAM: router stream copy 1 MB/layer (+ 9.7 KB score table).

Correctness (test_moe_decode.py, layers 6/7/39 x 64 rows, logs/dev/f_moe6.log): logits, ids, BF16 weights bitwise
192/192 vs MoE.route + route records; output PCC 0.999976-0.999987 vs MoE.__call__; vs a host FP32 reference on the
device weight values PCC 0.99998 (current path 0.999986).
Timing (rep_ms, layer 6, load 19.5): **MoE 177.2 -> 72.8 us** (router 13.3; experts launch ~60).  Device profile:
experts program 64.8 us span, kernels <= 58 us, routing 4.25, ~21.3 MB in ~50 us = ~420 GB/s (DRAM-bound).
Split sweep (exp_moe_sweep.py): 1-3 gate workers/bank, 1-2 pairs, buffers 3-4 all 57-61 us: bandwidth-limited.

Integration (moe.py), decode rows only (T == 1; prefill / B > 1 keep the grouped path):

    from .fast.moe_decode import MoEDecode
    # MoE.__init__ (needs the WeightCache + layer index to cache the router's stream copy):
    self.decode_moe = MoEDecode.from_moe(self, cache=cache, layer=layer)
    # MoE.__call__, first lines:
    if x.shape[-2] == 1 and debug is None:
        return self.decode_moe(x)   # x [1,1,1,H] BF16 tile DRAM/L1 -> [1,1,1,H] BF16 tile in L1

## 2. Fused residual add + RMSNorm: `residual_norm.ResidualRMSNorm` (FastNorm; kernels/residual_norm_*.cpp)

One launch on 32 cores: h = a + b (bitwise ttnn.add, 72/72 cases), n = rms_norm(h) * w (<= 1 BF16 ulp vs
ttnn.rms_norm, min PCC 0.9999987; same error class vs FP64).  rep_ms 36.7 (add + rms_norm) -> 5.0-5.3 us.
Construct decode-only objects per norm weight (`ResidualRMSNorm(device, w, cfg.norm_eps)`); chain in decode: layer 0
`_, n = op_rn0(x)`; per layer `o = op(n); h, n2 = ffn_rn(x, o); f = ffn(n2); x, n = next_rn(h, f)` (next layer's
op_rn, the last one the final norm) -> LM head on n.  Use h/n in L1 (feeds StreamDecodeMatmul's fast case).
Details: agent://LfmKern.FastNorm, logs/fast_residual_norm.json.

## 3. Decode matmuls: `decode_matmul.StreamDecodeMatmul` (FastMatmul; kernels/decode_matmul_stream.cpp)

One custom DRAM-streaming kernel; weights get a one-time bank-contiguous tile permutation (same BFP8 values;
`cached_stream_weight(cache, key, PRESETS[...])`).  rep_ms before -> after (x in L1):
conv in_proj 49.8 -> 31.5 (bitwise), conv out_proj 38.3 -> 12.7 (bitwise), attn qkv 48.2 -> 16.8 (bitwise),
attn out 38.4 -> 12.7 (bitwise), dense w2 207.7 -> 73.8 (bitwise), dense w1|w3 + SwiGLU fused 154.5 -> 116.2 (PCC
0.999993, closer to FP64 than stock), LM head 836 -> 572 (487 GB/s; max abs 4.3e-6, top-5 equal), router 25.3 ->
13.2 (bitwise).  Extra device DRAM ~1.08 GB (prefill keeps the original layouts).  Integration snippets:
agent://LfmKern.FastMatmul (ShortConv in_dec/out_dec, Attention qkv_dec/wo_dec, DenseMLP w13_dec/w2_dec,
Lfm2Model head_dec); fast_step.py `FastStep.__init__` shows all constructions.

## 4. Conv tail / attention front (FastConvAttn)

- `conv_decode.ConvDecode(shortconv)`: B*x, 3-tap conv with state, C*, state shift in one launch on 64 cores,
  bitwise (y, s0, s1; B=1 and 4).  Tail 53.4 -> 8.2 us.  `y = fast(bcx)` between in_proj and out_proj; state tensors
  and prefill unchanged.
- `attn_decode.AttnDecode(attention, rope)`: qkv split, per-head q/k RMSNorm, RoPE, paged BFP8 KV write, q permute in
  one launch, bitwise (q, full K/V caches, op output at 11 positions, B=1 and 4).  Front 130.4 -> 15.9 us;
  `qd = fast(qkv, cur_pos, page_table)` then the unchanged SDPA decode; Rope.decode gathers no longer needed.
Details: agent://LfmKern.FastConvAttn, logs/fast_conv_decode.json, logs/fast_attn_decode.json.

## 5. Next steps after round 1 (by expected gain; 1-3 and the B > 1 part of 4 were done in round 2, section 6)
1. Launch count: ~6 us fixed per generic_op dominates the small ops.  Fuse conv: in_proj + tail + out_proj (one
   program), attention: qkv + front, and the norm into the following matmul's in0 producer.
2. MoE: fold the router (2 cores, 13 us + launch) into the experts program as a third role signalling the routing
   workers by semaphore (~-6 us/layer); split the routing pass over NCRISC+BRISC (~-2 us).
3. SDPA decode (31 us x 10) for head_dim 64 / B=1 with the K/V write fused.
4. B > 1 (vLLM max-num-seqs > 1): MoEDecode is single-row; ConvDecode/AttnDecode/StreamDecodeMatmul/ResidualRMSNorm
   take B <= 16-32 rows.

## 6. Round 2: runtime step, fused programs, B > 1 row independence

### 6.1 `step.FastStep` (StepOwner): the runtime decode step
`FastStep(model, parts=DEFAULT_PARTS, src=None, sdpa=(8, 8, 512))` after the KV caches are allocated, before trace
capture; `logits = fast.decode_forward(tok, cur_pos, rope_idx, page_table)` (same inputs / output as
`Lfm2Model.decode_forward`, logits in model.logits_dtype).  It shares the model's weights, KV and conv state.  It is
device-only and address-keyed (trace-safe).  Per call, a part that does not apply to the current B falls back to the
stock code.  `.view(parts)` gives ablations without rebuilding.  `Lfm2Model.enable_fast_decode(parts)` routes
`decode_forward` through it.  The serving adapter (lfm2_vllm.py) pins its parts in `FAST_PARTS` / `LFM_FAST_PARTS`.

| part | module | B | vs round-1 chain |
|---|---|---|---|
| moe | moe_decode.MoEDecode | 1 | (round 1) |
| moe2 | moe_decode2.MoEDecodeV2: router role + experts in one launch | 1 | bitwise |
| norm, conv, attn, mm | round-1 modules (sections 2-4) | <= 32 / 16 / 16 / 32 | (round 1) |
| nmm | norm_matmul.NormStreamDecodeMatmul: operator norm + conv in_proj / attn qkv, ffn norm + dense w1\|w3; layer 0 also folds the embedding gather | <= 16 | bitwise |
| convf | conv_fused.ConvOpDecode: conv in_proj + tail + out_proj (+ operator norm prologue with norm) | == conv-state rows | bitwise |
| afront | attn_fused.AttnQKVFront: qkv matmul + AttnDecode front | <= 8 | bitwise |
| anfront | attn_fused.AttnNormQKVFront: operator norm + qkv + front | <= 8 | bitwise; slower in the step (6.3) |
| sdpa | stock SDPA decode program config (8, 8, 512) instead of (8, 8, 64) | any | not bitwise (PCC 0.99996) |
| asdpa | attn_fused.SDPADecode: custom paged SDPA decode | == model.batch | not bitwise (PCC vs FP64 0.99996, stock 0.99959) |

### 6.2 Modules (rep_ms = 16 calls per trace; load average in brackets)
- `norm_matmul.NormStreamDecodeMatmul` and `NormPrologue` (kernels/decode_matmul_norm.cpp, norm_prologue.hpp,
  StepOwner): ResidualRMSNorm + StreamDecodeMatmul in one program on an 8 x 4 rectangle at (0, 6).  The weight
  streams during the norm, and the master core gathers n and multicasts it once.  The reusable prologue is used by
  conv_fused and attn_fused, so its API must stay stable.  conv_in pair 35.7 -> 29.6 us, attn_qkv 21.0 -> 16.4, dense
  w13 120.8 -> 112.9, embedding + norm + conv_in 42.7 -> 29.5 (load 1.0-1.5).  h and y are bitwise at 1 and 4 rows
  (logs/fast_norm_matmul.json).  Placement matters: at (0, 0) the stream is ~3.5 us slower than at (0, 6), because
  the DRAM-bank anchors sit in columns 0 and 6.
- `conv_fused.ConvOpDecode` (ConvFuse): 3-launch conv 57.1 -> 40.2 us; ResidualRMSNorm + 3-launch 61.3 -> 40.9 us
  with the norm prologue (load 1.9).  out / s0 / s1 / h are bitwise vs the 3-launch chain at B = 1, 4 and 20
  (logs/fast_conv_fused.json).  The out_proj in0 tile height is 8 / 16 / 32 rows by B.
- `moe_decode2.MoEDecodeV2` (MoEFuse): router logits are computed in the experts program, and the routing is
  delivered to the workers by semaphore.  One launch: MoEDecode 72.8 -> 53.7 us (x L1; load 1.5-2.0).  The output
  is bitwise MoEDecode, and logits / ids / weights are bitwise MoE.route (logs/fast_moe_decode2.json).
- `attn_fused` (AttnFuse): AttnQKVFront and AttnNormQKVFront give bitwise q and K/V at B = 1 and 4.  The
  AttnNormQKVFront module is 31.8 -> 25.9 us, but the step is ~0.1 ms slower with it (6.3).  SDPADecode: 14.8 ->
  11.2 / 33.0 -> 17.3 / 247 -> 92 us at pos 128 / 2000 / 32000 (logs/fast_attn_fused_r3.json).

### 6.3 Whole step (tests/fast/fast_step.py; traced ms/token, 64 teacher-forced steps on the book prompt)
Sources: logs/fast_step_r2.json (load 0.9-2.0) for the rows up to + afront and the r2 full set.
logs/fast_step_r3a.json / r3b.json (LfmKern3, load 1.1-2.0) for the moe2 / anfront / asdpa rows.

| step variant | pos 128 | pos 2000 | pos 32000 |
|---|---|---|---|
| stock `Lfm2Model.decode_forward` | 17.37 | 17.56 | 19.69 |
| round-1 parts (moe, norm, conv, attn, mm) | 6.50 | 6.70 | 8.84 |
| + nmm | 6.21 | 6.40 | 8.54 |
| + convf | 5.88 | 6.08 | 8.22 |
| + afront (the bitwise base) | 5.77 | 5.97 | 8.11 |
| + moe2 | **5.28** | **5.47** | **7.62** |
| + moe2 + anfront | 5.38 | 5.57 | 7.72 |
| + moe2 + anfront + asdpa (not bitwise) | 5.21 | 5.24 | 5.98 |
| r2 set with sdpa (8, 8, 512) + asdpa, no moe2 (not bitwise) | 5.68 | 5.75 | 6.52 |

moe2 whole-step check: base + moe2 + anfront vs base gives 64/64 steps with bitwise FP32 logits at 128, 2000 and
32000.  So MoEDecodeV2 (and AttnNormQKVFront) are bitwise in the step, not only per module.  Top-1 vs stock:
58/64, 63/64 and 61/64, the same as the base.  Every disagreement is a near-tie with fast's argmax in stock's top-5.
asdpa vs stock: 60/64, 63/64, 62/64.

### 6.4 B > 1: row independence (LfmTT's report: batch_vs_single B = 4, slot 3 diverged at step 49)
Root cause:
- The fast step at B > 1 was a hybrid.  moe / moe2 are single-row, so at B > 1 they fell back to the stock grouped
  MoE, which is not bitwise MoEDecode (PCC 0.99998).  Slot b at B > 1 therefore ran a different step from both B = 1
  references (fast with MoEDecode, and stock with fast-vs-stock norm / w13 / head differences).  The greedy chat
  sequence drifted to a near-tie flip at step 49.
- The B = 3 per-part PCCs (kern/StepOwner/st_d.json) compared fast against stock at the same B.  Those comparisons
  are expected to be non-bitwise for norm (<= 1 ulp vs rms_norm) and mm (w13 GLU, head).  They did not show a row
  bug.

The suspected row bugs are absent.  `tests/fast/test_rows.py` (logs/fast_rows.json, logs/fast_rows_conv.json) places
slot j's data in row j at B = 1, 2, 3, 4 and 8.  The other rows hold other slots' data or zeros.  The input padding
rows are zero, 3e4 or NaN (ttnn.Tensor pad_value, standing in for the undefined padding that fast ops leave).
Every row must be bitwise its B = 1 output.  0 failures in 4.8k checked rows (load 1.1-1.3):
- ResidualRMSNorm: h and n, residual and norm-only, inputs L1 / DRAM, outputs L1 / DRAM.
- StreamDecodeMatmul: all 8 presets, x L1 / DRAM, w13 out L1.  The 64 B-rounded transfers only touch padding rows.
- NormStreamDecodeMatmul: conv_in, attn_qkv and mlp_w13 (h and y), plus the embed form.
- ConvOpDecode: norm (0, 6), residual and norm-only, and rows placement.  out, s0 and s1 are checked, with slot j's
  conv state in row j.
- ConvDecode, and stock rms_norm / linear / add.

Undefined padding rows are never consumed into valid rows.  The sum of squares reads only the diagonal of h h^T for
rows < B, and the matmuls are row-local.

Whole-step check: `tests/fast/test_step_rows.py` (logs/fast_step_rows_sdpa_default.json, fast_step_rows_sdpa16.json),
parts norm, conv, attn, mm, nmm, convf, afront, anfront with the stock MoE on both sides.  Even rows other than the
slot are live slots; odd rows are inactive (cur_pos -1).
- Default SDPA config at both B: slot 3 at B = 4 is not bitwise vs B = 1 from step 0.  Logits max abs: fast 0.82,
  stock 0.87.
- Stock is also not bitwise at B > 1: the stock paged SDPA decode splits its grid over the users, so each user gets
  64 / B cores and a different K split and reduction order.
- With 16 cores per user on both sides (B = 1 with SDPA (8, 2, 64), B = 4 with (8, 8, 64)), the fast logits are
  bitwise 8/8 steps.
So every fast part is row-independent.  The only B-dependence left is the stock SDPA's per-user core split, and it
is shared with the stock step.  It also explains the fast-vs-stock step-0 PCC of ~0.985 at B > 1 in
logs/fast_step_r3_b*.json.

Fix: LfmKern3's `moe_rows.MoEDecodeRows` (part moer) is per row bitwise MoEDecodeV2, so the MoE at B > 1 is the
B = 1 MoE.  No change was needed in the round-1 / round-2 kernels.  `tests/batch_vs_single.py` with LFM_FAST_DECODE=1
and parts moe2, moer, norm, conv, attn, mm, nmm, convf, afront is 64/64 vs both B = 1 references (stock and fast)
at B = 4 slot 3, B = 4 slot 0, B = 2 slot 1 and B = 2 slot 0 (logs/dev/k3_gate_b4.log, bf_bvs_4_0.log,
k3_gate_b2.log, bf_bvs_2_0.log).  Before the fix, B = 4 slot 3 diverged at step 49 (bvs_1_4.log).

Remaining: slot logits at B > 1 are not bitwise equal to B = 1, in fast and stock alike, because of the SDPA core
split.  A per-B SDPA grid with fixed cores per user (16: B = 1 (8, 2), B = 2 (8, 4), B = 4 (8, 8)) would make them
bitwise up to B = 4, but it slows B = 1 attention at long context.  It was not adopted.

## 7. Round 3 (LfmKern3): multi-row MoE decode, verified defaults, batched throughput

### 7.1 `moe_rows.MoEDecodeRows` (part moer; kernels/moe_rows_router.cpp, moe_rows_experts.cpp)
B = 2..16 token rows (`model.batch`), two launches per MoE layer.  The union of the rows' selected experts is streamed
once, in ascending expert id order; every row is computed against every union expert with its own routing weight (0
where it did not select the expert: exact zero contributions), so each row sums its own experts in ascending id order
exactly as MoEDecodeV2 does on that row alone.
- program 1: router role (2 cores, moe_decode2's bitwise router matmul with in0 = rows 0-15 of x; the cores swap
  their [16, 32] FP32 logits tiles; the 4 router RISCs route contiguous row ranges with moe_decode2_route.hpp, bitwise
  lfm_route, and write per-row records + a semaphore increment to every gate/up worker) + gate/up workers (3 per
  bank, one bank per worker row as V2): union, per-row weight table, units (union expert, one gate/up column pair)
  round-robin, per row group of [tile_rows, 32] in0 tiles one transposed custom_mm + SiLU(g) * u * w_row (per-row
  weight by SFPU lane, batched_experts_kernel.cpp), result into an L1-interleaved activation tensor.
- program 2: down workers (4 per bank, 2 columns each) stream their columns of every union expert and accumulate in
  FP32 DEST per row group (batched_experts_kernel.cpp's down role).  tile_rows = min(pow2 >= B, 8), 2 row groups at
  B > 8.
- `__call__(x, cur_pos=None)`: with the step's cur_pos, rows with cur_pos < 0 (idle slots) select no expert, so idle
  slots cost no expert bandwidth (their output rows are zero).
- Same resident tensors as MoEDecodeV2; no new weight copies.  Extra L1 per call: the activation tensor
  (<= 64 experts x 2 groups x 8 banks x 3 KB, spread over the grid).

Correctness (tests/fast/test_moe_rows.py, layers 6 and 39, B = 2, 3, 4, 8, 16, 4 trials each incl. cur_pos with the
last row inactive; logs/fast_moe_rows.json, logs/dev/k3_mr4.log): every active row's output, logits and routing
(ids + BF16 weights) BITWISE MoEDecodeV2 on that row alone (244/244 rows); union = sorted union of the per-row
selections 40/40; inactive rows zero with no union entry 20/20; PCC vs the stock grouped MoE on the same rows
>= 0.999977.

rep_ms per MoE layer (layer 6, x L1, load 1.3-1.4): unions of 7-38 experts (5.3 MB each) stream at ~420 GB/s.

| B | stock MoE (grouped) | MoEDecodeRows | 1-row MoEDecodeV2 |
|---|---|---|---|
| 2 | 276 us | 101 us | 54 us |
| 3 | 360 | 143 | |
| 4 | 445 | 175 | |
| 8 | 699 | 284 | |
| 16 | 1096 | 440 | |

Split / buffer sweep (gate_splits 3-5, gate_buffers 3-4, down_buffers 6-8, down_splits 2-4): all within 1-2%
(DRAM-bound); defaults kept.

### 7.2 Defaults and the B > 1 gate
`DEFAULT_PARTS = moe2, moer, norm, conv, attn, mm, nmm, convf, afront` (step.py; LfmTT was told to pin the same list
in lfm2_vllm.py FAST_PARTS).  B = 1: bitwise round 2 (6.3).  B > 1: tests/batch_vs_single.py 64/64 vs both B = 1
references (6.4).  anfront stays out (slower in the step); asdpa is opt-in (not bitwise; B <= 13: it needs one core
per row and kv head, larger batches keep the stock SDPA).

SDPADecode at the serving page-table width (max_model_len 131072 = 2048 blocks per sequence): it used to hold S, P and
V of a worker's whole share in L1 (sized by table width / workers: 2.7 MB per core at 2048 blocks, build error).
Now a share longer than `chunk_blocks` (40) runs as several exact two-pass partitions (blocks [p n / parts,
(p + 1) n / parts), parts = active workers x ceil(n / (active x 40))), each sent to the group reducer like a worker
(mailbox slot / gather rows p); the reducer only reads the live gather tiles.  Contexts up to 13 x 40 blocks
(B = 1) keep one partition per worker and are bitwise the previous kernel (identical r3 metrics).
tests/fast/test_attn_fused.py --parts sdpa,sdpa_long (logs/fast_attn_sdpa_long.json, logs/dev/k3_af4.log): PCC vs
FP64 >= 0.99996 at pos 0 .. 131071 for B = 1, B = 4 and chunk_blocks = 8 (stock 0.9906 at B = 4, pos 131000).
rep_ms B = 1 at the 2048-block table, SDPADecode vs stock SDPA + reshape: 11.1 vs 19.7 us (pos 128), 17.6 vs 42.3
(2000), 91.7 vs 256.5 (32000), 348 vs 959 (131071) (load 2.5).

### 7.3 Whole step at B > 1 (tests/fast/fast_step.py --batch B --ablate --no-check; every row active with its own
token at the position, traced ms/step, aggregate tok/s = B / step time; logs/fast_step_r3_batch{2,4,8,16}.json)

pos 2000 (B = 1 from 6.3):

| B | stock | before: fast with the stock MoE (all-moer) | defaults (moer) | defaults + asdpa | load |
|---|---|---|---|---|---|
| 1 | 17.54 ms / 57 tok/s | 5.97 / 168 | 5.47 / 183 | 5.24 / 191 (+anfront) | 1.1 |
| 2 | 20.44 / 98 | 12.83 / 156 (+asdpa) | 6.69 / 299 | 6.49 / 308 | 3.9 |
| 4 | 25.79 / 155 | 18.16 / 220 (+asdpa) | 8.94 / 447 | 8.62 / 464 | 2.4 |
| 8 | 33.58 / 238 | 25.92 / 309 (+asdpa) | 12.14 / 659 | 11.57 / 692 | 1.5 |
| 16 | 47.34 / 338 | 40.24 / 398 | 17.55 / 912 | (asdpa does not fit) | 2.3 |

pos 128: B = 2 stock 20.18, defaults 6.45, + asdpa 6.38 ms; B = 4 25.02 / 8.28 / 8.16; B = 8 31.95 / 10.49 / 10.32;
B = 16 44.69 / 14.97 ms (1069 tok/s).
(The earlier logs/fast_step_r3_b*.json fed the same token to every row, i.e. one expert union: optimistic, superseded.)

### 7.4 Remaining gap (B = 1, pos 2000)
Weight streaming per token: experts 38 x 4 x 5.3 MB = 0.81 GB + dense BFP8 projections / head ~1.08 GB = 1.9 GB,
~3.8-4.0 ms at 470-500 GB/s.  Now 5.47 ms bitwise (5.24 with asdpa): ~1.3-1.5 ms above the roofline.  MoE 38 x 53.7 us
= 2.04 ms (floor ~1.6); the remaining ~3.4 ms covers ~1.08 GB of dense weights (floor ~2.2 ms) plus ~170 launches.
Next: fold the ffn residual + norm into the MoE router cores (~5 us x 38); fewer conv / attention programs; for
B > 1 a single-program MoEDecodeRows (down workers prefetching their union columns during gate/up, act hand-off by
semaphore as in V2) would save one launch + pipeline fill per layer (~10 us x 38).

## 8. Round 4 (LfmBatchDecode): decode widths and a batch SDPA decode on every core

### 8.1 Decode widths
`step.decode_widths(batch)` = 1, 2, 4, ... and `batch`; FastStep builds the row-count-specific modules once per width
(moer, conv, convf, asdpa; weights shared: MoEDecodeRows takes MoEDecodeV2's router stream copy, ConvDecode /
ConvOpDecode share the replicated taps).  A width w < model.batch runs decode rows / conv-state slots [0, w).
ConvDecode / ConvOpDecode take `rows=` (default the state's rows); an odd row count < 16 moves rows + 1 (64 B
granules), so with a wider state the kernels write the extra state row back unchanged (`KEEP_ROW` / `keep_row`).
`Lfm2Model.decode_widths` / `decode_width(n)` expose the buckets; the stock step keeps the full width.

### 8.2 `attn_fused.SDPADecode` (kernels/attn_sdpa.cpp), rewritten schedule
Before: W = min(13, 110 // (8 B)) cores per (row, kv head) group, B <= 13 (the step fell back to the stock SDPA at
16 rows, which gives each user 64 / 16 = 4 cores whatever its activity: 1.9 ms per layer for ONE active row at 16K).
Now, for 1..16 rows on `cores` cores (default 104 = 8 x 13 for one row, all 110 for more):
- partitions per group depend only on that row's position: min(13, n) partitions up to 13 x C blocks, then
  13 x ceil(n / (13 C)) (C = 40 at the serving 2048-block table; narrower tables give the same partitions below
  13 x C blocks).  One row's result is therefore bitwise independent of the batch it runs in (`row vs alone`
  bitwise in every case of tests/fast/test_attn_fused.py --parts sdpa_batch,sdpa_batch_narrow) and B = 1 is the
  previous kernel's split (model-level greedy 64/64 identical to dev-m8).
- schedule: all groups' partitions in (b, g, p) order weighted by blocks + k0 (k0 = 2) are cut into `cores` equal
  ranges; every core derives the plan from cur_pos (no host work per step).  Rows at -1 have no partitions: idle
  rows cost nothing.  A core runs its partitions as segments (one per group: Q + page-table slice, double-buffered
  Q / mask CBs).
- reducer of group G = core G % cores, slot G // cores (2 slots at 16 rows on 110 cores); idle slots pass their
  gather region through the CB so the slot regions stay aligned with the CB front (and cAf is rewound after a slot
  with fewer live tiles).
- the FUSE_WO variant (out projection in the same program; never adopted in the step) was removed.

rep_us at the serving table width (logs/fast_attn_sdpa_batch.json; load 2.3-3.4):

| rows (all active) | 4096 | 16384 | 32767 | 65535 | 122880 | stock at 16 rows |
|---|---|---|---|---|---|---|
| 1 | 24.1 | 54.1 | 95.5 | 181.8 | 328.5 | |
| 2 | 35.5 | 100.9 | 188.2 | 361.3 | 668.1 | |
| 4 | 60.2 | 186.9 | 355.9 | 699.3 | 1297 | |
| 8 | 99.8 | 341 | 656 | 1294 | 2381 | |
| 16 | 194 | 674 | 1320 | 2634 | 4982 | 502 / 1929 / 3826 / 7631 / 14302 |

(rows 2-16 measured on 110 cores; one active row in a 16-row call: 24-346 us, i.e. its B = 1 time.)  16 rows at
122880: 2.1 GB of K/V in 4.98 ms = 429 GB/s (DRAM-bound).  B = 1 vs the dev-m8 kernel (same session):
12.2 / 18.8 / 93.5 / 348.6 us vs 11.1 / 17.5 / 91.8 / 348.5 at 128 / 2000 / 32000 / 131071.

### 8.3 Whole step, served shape (tests/batch_decode_bench.py: 16-slot adapter model, 16400-block pool, 2048-block
tables; n active rows at the position; logs/bdb/m8_time.json, m9_time.json): see src/NOTES.md section 16.
