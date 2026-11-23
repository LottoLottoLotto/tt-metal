# tt/fast — LFM3 fused traced decode step (owner Lfm3Fast)

Ported from the LFM2.5 kernels (`lfm2/tt/fast`) and adapted to LFM3 (sandwich norms,
7-tap conv with a 6-row state, head_dim 128 attention with q/k norm + sigmoid head gate + 256-token sliding window,
dense SwiGLU 7680 in every layer, BFP4 MLP weights, 128000-wide head).  Module work: FastNorm3 (norm / matmul /
fused MLP), FastConv3 (conv), FastAttn3 (attention), integration + whole-step measurements: Lfm3Fast.

Tests (development tree, not shipped here): `lfm3/tests/fast/` via `tools/devrun.sh NAME -m models.demos.blackhole.lfm3.tests.fast.<test>`;
weights from `LFM_FAST_SRC` (default `quant/ckpt-bfp8`; the GPTQ default checkpoint is `quant/ckpt`:
w1/w3 BFP4, everything else BFP8).  `harness.rep_ms` = 16 calls per captured trace (device time per call).
Checkpoints: GPTQ `quant/ckpt` (cache q-cf2a8ba32c, 1.459 GB streamed per token), RTN `quant/ckpt-bfp8`
(q-f68ad7c280, 1.836 GB).  DRAM roofline below = weight bytes / 470 GB/s (the P150 streaming rate measured in
lfm2; the fused MLP reaches 478-485 GB/s, the LM head 490): GPTQ 3.10 ms, ckpt-bfp8 3.91 ms.

## 0. Whole step (tests/fast/fast_step.py; traced ms/step, aggregate tok/s = B / step; DEFAULT_PARTS)

B = 1 (logs/fast_step_q_b1.json, logs/fast_step_bfp8_b1.json; load 0.7-2.0):

| checkpoint | pos | stock ms | fast ms | fast tok/s | fast / roofline |
|---|---|---|---|---|---|
| GPTQ `quant/ckpt` | 128 | 15.10 | **3.45** | **290** | 1.11 |
| GPTQ | 2000 | 15.13 | **3.48** | **287** | 1.12 |
| GPTQ | 32000 | 15.80 | **3.93** | **255** | (+ K/V reads of the 6 full layers) |
| `quant/ckpt-bfp8` | 128 | 15.15 | **4.21** | **238** | 1.08 |
| ckpt-bfp8 | 2000 | 15.19 | **4.24** | **236** | 1.08 |
| ckpt-bfp8 | 32000 | 15.86 | **4.68** | **214** | |

B > 1: every row active at the position with its own token (logs/fast_step_{q,bfp8}_b{2,4,8,16}.json; load 1.2-2.1):

| B | GPTQ pos 128: ms / tok/s | GPTQ 2000 | GPTQ 32000 | bfp8 128 | bfp8 2000 | bfp8 32000 |
|---|---|---|---|---|---|---|
| 1 | 3.45 / 290 | 3.48 / 287 | 3.93 / 255 | 4.21 / 238 | 4.24 / 236 | 4.68 / 214 |
| 2 | 3.49 / 574 | 3.54 / 564 | 4.43 / 451 | 4.23 / 473 | 4.29 / 466 | 5.18 / 386 |
| 4 | 3.55 / 1128 | 3.67 / 1091 | 5.40 / 741 | 4.30 / 930 | 4.42 / 905 | 6.15 / 651 |
| 8 | 3.67 / 2178 | 3.92 / 2039 | 7.35 / 1089 | 4.44 / 1802 | 4.69 / 1707 | 8.11 / 986 |
| 16 | 4.00 / 3999 | 4.47 / 3582 | 11.41 / 1402 | 4.76 / 3359 | 5.23 / 3060 | 12.17 / 1314 |
| stock 1 | 15.10 / 66 | 15.13 / 66 | 15.80 / 63 | 15.15 / 66 | 15.19 / 66 | 15.86 / 63 |
| stock 16 | 15.34 / 1043 | 16.16 / 990 | 26.64 / 601 | 15.87 / 1008 | 16.21 / 987 | 26.70 / 599 |

The step is weight-streaming bound up to B = 16 at short context; at 32000 the 6 full-attention layers' K/V reads
grow with B (16 rows x 32000 positions x 6 layers = 2.7 GB of BFP8 K/V per step; SDPADecode streams it at ~457 GB/s).

## 1. Parts (`step.py`, `FastStep(model, parts)`)

`DEFAULT_PARTS = norm, mm, nmm, conv, convf, attn, afront, asdpa, mlpf, anfront`.
`Lfm3Model.enable_fast_decode(parts=None)` (None -> env `LFM_FAST_PARTS` or `DEFAULT_PARTS`) routes
`decode_forward(tok, cur_pos, rope_idx, page_table)` through `FastStep.decode_forward` (same I/O as the stock
`decode_forward_stock`; logits in `model.logits_dtype`, BF16 for the serving sampler).  Device-only,
trace-capturable, programs address-keyed.  Widths 1, 2, 4, ..., model.batch (<= 16); width w runs rows / conv slots
[0, w) and leaves the other slots' state untouched; one trace per width.  `FastStep.view(parts)` = ablation without
rebuilding.

Sandwich chain: with `norm` the step carries a pending residual (a, b); the next fused module computes
h = a + bf16(rms_norm(b) * post), n = rms_norm(h) * w in its prologue (two reductions), post = the previous layer's
post_ffn_norm (operator side) or the layer's post_operator_norm (ffn side); final: post_ffn_norm[23] +
embedding_norm.  Default step at B <= 16, 74 programs (stock ~1000):
- layer 0 (conv): NormStreamDecodeMatmul.embed (embedding gather + operator norm + in_proj) -> ConvDecode -> out_proj
  stream -> FusedMLPDecode
- conv layer: ConvOpDecode (sandwich + operator norm + in_proj + 7-tap tail + out_proj) -> FusedMLPDecode (sandwich
  + ffn norm + w1|w3 SwiGLU + w2)
- attention layer: AttnNormQKVFront (sandwich + operator norm + qkvg + q/k norm + RoPE + paged KV write + gate) ->
  SDPADecode (gated output) -> o_proj stream -> FusedMLPDecode
- ResidualRMSNorm (final sandwich + embedding_norm) -> LM head stream.

| part | module (file) | what | rows |
|---|---|---|---|
| norm | residual_norm.ResidualRMSNorm(post_weight=) | sandwich residual + norm, one launch | <= 32 |
| mm | decode_matmul.StreamDecodeMatmul | DRAM-streaming matmul, bank-contiguous stream copies (BFP8/BFP4/BF16), glu SwiGLU, x multicast from 9 rows | <= 32 |
| nmm | norm_matmul.NormStreamDecodeMatmul (+ NormPrologue) | sandwich norm prologue + conv in_proj / qkvg / w1\|w3; `.embed` for layer 0 | <= 32 |
| conv | conv_decode.ConvDecode | B*x, 7-tap conv over the 6-row state, C*, state shift (in-kernel) | <= 32 |
| convf | conv_fused.ConvOpDecode | prologue + in_proj + tail + out_proj in one program | <= 32 |
| attn | attn_decode.AttnDecode | qkvg split, q/k RMSNorm, RoPE (tables read at cur_pos), BFP8 paged KV write, gate copy | <= 16 |
| afront | attn_fused.AttnQKVFront | qkvg stream matmul + AttnDecode front in one program | <= 16 |
| anfront | attn_fused.AttnNormQKVFront | sandwich norm prologue + AttnQKVFront (overrides afront / nmm on attention layers) | <= 16 |
| asdpa | attn_fused.SDPADecode | paged SDPA decode, head_dim 128, GQA 4, window 256 (<= 5 blocks read), sigmoid gate in an FP32 epilogue, batch-invariant partitions, idle rows free | <= 16 |
| mlpf | mlp_fused.FusedMLPDecode | sandwich + ffn norm + w1\|w3 SwiGLU + w2 in one program on 64 cores (w2 weights stream while w1\|w3 runs; g gathered and multicast once) | <= 16 |

`attn` / `afront` / `nmm` (except the layer-0 embed form) / `conv` (except layer 0) are fallbacks the default set no
longer takes at B <= 16 (leave-one-out below).  Stream-layout copies live in the WeightCache under
`<key>.stream-c<cpb>w<w>[-glu]` (one device copy per layout, shared by every module; ~1.4-1.8 GB of device DRAM;
prefill keeps the original layouts).  `decode_matmul.preset_for(name, dtype)` / `mlp_fused.fused_presets(dtype)`
pick the tuned presets per format.

## 2. Module timings (rep_us, B = 1 unless stated, x in L1, GPTQ ckpt; module logs below)

| module | fast | stock equivalent | evidence |
|---|---|---|---|
| ResidualRMSNorm sandwich (B 1/16) | 7.2 / 10.5 | 69.9 (rms_norm(post) + add + rms_norm) | logs/fast_residual_norm.json |
| StreamDecodeMatmul conv_in 2048x6144 | 31.4 (425 GB/s) | 49.8 | logs/fast_decode_matmul.json |
| conv_out / o_proj 2048x2048 | 11.9 / 12.0 | 38.4 / 38.3 | " |
| qkvg 2048x3104 | 17.9 | 48.6 | " |
| w2 7680x2048 BFP8 | 40.8 (409 GB/s) | 136.3 | " |
| w1\|w3 + SwiGLU, BFP4 / BFP8 | 44.3 / 78.1 | MLP 234-237 (whole) | " |
| LM head 2048x128000 FP32 / BF16 out | 569.6 / 568.9 (490 GB/s) | 835.6 / 785.1 | " |
| nmm conv_in / qkvg / w13 BFP4 / w13 BFP8 (sandwich) | 30.2 / 17.9 / 42.5 / 74.9 | chain 119 / 118 / - / - | logs/fast_norm_matmul.json |
| nmm.embed (embedding + norm + conv_in) | 30.6 | 42.6-50.2 | " |
| FusedMLPDecode GPTQ / bfp8 (B 1/16) | 72.3-77.6 (478 GB/s) / 103.7-109.5 (485 GB/s) | pair nmm w13 + w2: 82.6-86.7 / 115.6-117.9 | logs/fast_mlp_fused.json, fast_mlp_fused_bfp8src.json |
| ConvDecode tail | 8.1 | 131 (stock middle) | logs/fast_conv_decode.json |
| ConvOpDecode no-norm / norm+post (B 1/16) | 39.7 / 40.3-45.8 | ShortConv.decode 190.9; chain 259.9 | logs/fast_conv_fused.json |
| AttnQKVFront (pos 2000, x DRAM / L1) | 29.9 / 25.0 | qkvg linear + front 169.5 | logs/fast_attn_fused_r2.json |
| AttnNormQKVFront vs ResidualRMSNorm + AttnQKVFront (B 1/4/8/16) | 26.1 / 30.8 / 36.2 / 40.6 | 31.9 / 33.5 / 36.1 / 48.7 | " |
| SDPADecode full, pos 128 / 2000 / 32000 / 131071 | 12.0 / 16.8 / 90.3 / 340.0 | 28.7 / 47.3 / 159.0 / 510.4 | " |
| SDPADecode sliding (any pos) | 11.7-12.0 | 28.6-31.7 | " |
| SDPADecode B=16 full / sliding, pos 2000 | 97.5 / 25.6 | 151.4 / 51.8 | " |
| whole attention op incl. sandwich norm, pos 2000, full / sliding (B=1) | 55.2 / 50.1 | 330.0 / 324.2 | " |

## 3. Ablation (traced ms, B = 1; cumulative in DEFAULT_PARTS order, then leave-one-out of the full set)

| variant | GPTQ 128 | GPTQ 2000 | GPTQ 32000 | bfp8 2000 |
|---|---|---|---|---|
| stock | 15.10 | 15.13 | 15.80 | 15.19 |
| norm | 13.15 | 13.21 | 13.88 | 13.26 |
| + mm | 8.17 | 8.22 | 8.89 | 8.96 |
| + nmm | 7.94 | 7.97 | 8.65 | 8.75 |
| + conv | 6.04 | 6.09 | 6.76 | 6.87 |
| + convf | 5.92 | 5.96 | 6.63 | 6.74 |
| + attn (front; stock SDPA + gate) | 4.03 | 4.11 | 4.78 | 4.89 |
| + afront | 4.02 | 4.09 | 4.76 | 4.87 |
| + asdpa | 3.77 | 3.80 | 4.24 | 4.58 |
| + mlpf | 3.53 | 3.56 | 4.00 | 4.31 |
| + anfront (= all) | **3.45** | **3.48** | **3.93** | **4.24** |
| all - norm | 7.66 | 7.69 | 8.13 | 8.44 |
| all - mm | 5.15 | 5.18 | 5.62 | 5.93 |
| all - nmm (layer 0 then uses embedding + convf) | 3.47 | 3.50 | 3.94 | 4.25 |
| all - conv | 3.47 | 3.50 | 3.94 | 4.25 |
| all - convf | 3.64 | 3.67 | 4.11 | 4.42 |
| all - attn | 3.45 | 3.48 | 3.93 | 4.23 |
| all - afront | 3.45 | 3.48 | 3.93 | 4.23 |
| all - asdpa | 3.73 | 3.81 | 4.48 | 4.56 |
| all - mlpf | 3.70 | 3.73 | 4.18 | 4.52 |
| all - anfront | 3.53 | 3.56 | 4.00 | 4.31 |

(With mlpf in the set, the nmm / mm w1|w3 fallbacks use the fused MLP's weight layout; for BFP8 that is the 8-cores-
per-bank layout with a 64-core norm, so "all - mlpf" on bfp8 is the 64-core-norm pair, not the round-1 32-core one.)

## 4. Correctness

- Fast vs stock step, 64 teacher-forced steps on the book prompt (stock greedy tokens fed to both; each path feeds
  its own KV / conv state forward; FP32 logits; logs/fast_step_{q,bfp8}_b1.json): GPTQ top-1 62/64, 64/64, 64/64 at
  pos 128 / 2000 / 32000, step-0 logits PCC 0.99994 / 0.99991 / 0.99991 (max abs 0.17-0.25), min PCC over the 64
  steps >= 0.99974; the misses (pos 128 steps 21, 32) are at stock top1-top2 gaps of 0.024 and 0.0019 with fast's
  token in stock's top-5.  ckpt-bfp8: 63/64, 64/64, 63/64, step-0 PCC 0.99993 / 0.99995 / 0.99990, misses at stock
  gaps 0.0019 and 0.093.  Not bitwise by design: SDPADecode is closer to FP64 than the stock SDPA, the fused glu / w2
  / head accumulate whole K in FP32 (rare 1-ulp differences vs ttnn.linear), the norms are <= 1 BF16 ulp.
- Round-2 parts in the step: + mlpf vs the round-1 set and + anfront vs + mlpf: FP32 logits bitwise 64/64 steps at
  pos 128 / 2000 / 32000 on GPTQ (logs/fast_step_q_mlpf.json, logs/fast_step_q_anf.json; bfp8 mlpf vs its pair
  logs/fast_step_bfp8_mlpf.json).
- B > 1 vs stock at the same B (one step re-feeding each prompt's last token, rows at 128 / 2000 / 32000, last row
  inactive): PCC >= 0.99949 every row; top-1 equal on every GPTQ row; on bfp8 the pos-128 rows miss at a stock gap of
  0.0048 (logs/fast_step_bfp8_b2_gap.json), every other row equal (logs/fast_step_*_b{2,4,8,16}.json "batch_check").
- Batch invariance (tests/fast/test_step_rows.py, logs/fast_step_rows_{q,bfp8}.json): the book prompt in slot W-1 of a
  width-W step (even rows live with their own sequences, odd rows inactive) gives FP32 logits **bitwise equal** to
  the same prompt alone at width 1 for 32/32 steps at W = 2, 4, 8, 16, both checkpoints.
- HF goldens (tests/test_model.py --fast default; greedy match length / teacher-forced top-1 on chat, code, prose,
  long; the prefill-side metrics are identical since the fast step only changes decode):

| checkpoint, step | chat | code | prose | long | log |
|---|---|---|---|---|---|
| GPTQ stock | 37 / 0.922 | 64 / 1.000 | 17 / 0.891 | 24 / 0.922 | logs/prefill/t_model_q_window.json |
| GPTQ fast | 37 / 0.938 | 64 / 1.000 | 17 / 0.906 | 24 / 0.922 | logs/t_model_q_fast.json |
| ckpt-bfp8 stock | 46 / 0.984 | 64 / 1.000 | 63 / 0.984 | 39 / 0.984 | logs/t_model_bfp8_stock.json |
| ckpt-bfp8 fast | 46 / 0.969 | 64 / 1.000 | 58 / 0.953 | 39 / 0.984 | logs/t_model_bfp8_fast.json |

  The bfp8 prose divergence at step 58 is a near-tie (fast margin 0.008, golden margin 0.127); the extra
  teacher-forced misses are at golden margins 0.005-0.18.
- Modules (owners' tests, both checkpoints): matmuls bitwise ttnn.linear for conv_in / conv_out / qkvg / o_proj at
  B 1-16, w2 1 element 1 ulp at B=4, head FP32 max abs 4.3e-6 with equal top-5 (`exact_preset` = stock accumulation
  order, bitwise everywhere, +3-9 us per projection); stream copies hold exactly the original block-float values
  (checked when built); glu PCC 0.99999 vs the stock silu*mul chain and closer to FP64; ResidualRMSNorm /
  NormPrologue <= 1 ulp per norm, nmm bitwise ResidualRMSNorm + StreamDecodeMatmul; FusedMLPDecode h / f bitwise the
  nmm w13 + w2 pair (B 1-16, 3 modes; PCC >= 0.99998 vs stock DenseMLP); ConvDecode / ConvOpDecode bitwise vs the
  stock conv middle + all six state rows (B 1-16, narrower widths, BFP8 and BFP4); attention fronts (incl. anfront)
  bitwise vs stock q / gate / K / V except one 1-ulp element at 7e-37; SDPADecode PCC vs FP64 >= 0.99993 over
  pos 0-131071 incl. window / block edges, B 1-16 (stock 0.99941); row independence (row j of a B-row call bitwise
  the B = 1 call on row j's data, padding rows zero / 3e4 / NaN): conv 4146/4146 (logs/fast_rows_conv.json),
  tests/fast/test_rows.py (ResidualRMSNorm sandwich / plain / norm-only, all 7 matmul presets in the GPTQ formats,
  NormStreamDecodeMatmul sandwich conv_in / qkvg / w13 BFP4 + embed, stock ops) 3358/3358 (logs/fast_rows.json),
  fused MLP 90/90; module logs logs/fast_*.json, device logs logs/dev/fn_*, fc_*, fa_*.

## 5. Where the time goes (device profile of one traced step, tests/fast/prof_step.py, B = 1, pos 2000)

logs/prof_step_q.json (GPTQ): 3.52 ms, 74 programs, 3.42 ms busy, 0.11 ms of inter-program gaps
(logs/prof_step_bfp8.json: 4.28 ms, same structure, MLP 104.2 us per layer).

| kind | n | us total | us each | weight floor each (470 GB/s) |
|---|---|---|---|---|
| FusedMLPDecode | 24 | 1740 | 72.5 | 73.2 (34.4 MB: w1\|w3 BFP4 + w2 BFP8) |
| LM head | 1 | 580 | 580 | 593 (278 MB) |
| ConvOpDecode | 12 | 479 | 40.0 | 37.9 (17.8 MB) |
| AttnNormQKVFront | 11 | 278 | 25.3 | 14.4 (6.8 MB) |
| SDPADecode (+ 3.3 us mean gap before it) | 11 | 153 | 13.9 | ~4.6 full / ~0.6 sliding (K/V) |
| o_proj (+ layer-0 conv out_proj) | 12 | 141 | 11.7 | 9.5 |
| layer 0: nmm.embed (+ in_proj) + ConvDecode (out_proj in the o_proj row) | 2 | 38 | | 28.5 (in_proj 13.4 MB) |
| final ResidualRMSNorm | 1 | 7 | 7 | - |

Every weight-streaming program now runs at 420-490 GB/s.  Remaining gap to the roofline (3.48 traced vs 3.13 ms incl.
K/V): the attention chain's fixed latency (front hand-offs and norm ~11 us, SDPA reduction ~10-14 us, its gaps:
~30 us per attention layer beyond its weights = ~0.33 ms) and ~0.1 ms of inter-program gaps.  SDPA + o_proj in one
program was built and measured slower (the o_proj prefetch starves the K/V readers on the same NOC; 48 vs 23 us,
FastAttn3 round 2), so it was removed.  Further B = 1 gains need a different attention schedule (e.g. starting the
o_proj stream only after SDPA's K/V reads) or fewer launches per attention layer; B > 1 at long context is bound by
K/V streaming.

## 6. Commands

    LFM_FAST_SRC=quant/ckpt tools/devrun.sh fs_q_b1 -m models.demos.blackhole.lfm3.tests.fast.fast_step --pos 128,2000,32000 --steps 64 --ablate --out logs/fast_step_q_b1.json
    LFM_FAST_SRC=quant/ckpt tools/devrun.sh fs_q_b4 -m models.demos.blackhole.lfm3.tests.fast.fast_step --batch 4 --pos 128,2000,32000 --out logs/fast_step_q_b4.json
    LFM_FAST_SRC=quant/ckpt tools/devrun.sh fs_rows_q -m models.demos.blackhole.lfm3.tests.fast.test_step_rows --batch 16 --widths 2,4,8,16 --out logs/fast_step_rows_q.json
    LFM_FAST_SRC=quant/ckpt tools/devrun.sh fs_prof_q -m models.demos.blackhole.lfm3.tests.fast.prof_step --pos 2000 --out logs/prof_step_q.json
    tools/devrun.sh tm_q_fast -m models.demos.blackhole.lfm3.tests.test_model --src q --fast default --out logs/t_model_q_fast.json
    logs/final_campaign.sh   # all of the above for both checkpoints (B = 1..16, rows, goldens, profile)
    module tests: test_residual_norm, test_decode_matmul, test_norm_matmul, test_mlp_fused, test_rows (norm / mm /
    nmm), test_conv_decode, test_conv_fused [--fmt bfp4], test_rows_conv, test_attn_decode,
    test_attn_fused --parts front,anfront,sdpa,sdpa_chunk,time (all `-m models.demos.blackhole.lfm3.tests.fast.<name>`)

## 7. Verify mode for prompt-lookup speculation (owner Lfm3Spec; shared rounds: src/NOTES.md sections 12 and 18)

`FastStep.enable_verify(widths)` + `decode_forward(..., verify=True)`: the rows are groups of consecutive tokens, one
group per sequence (a shared round, release s13 on; s8..s12: one sequence in conv-state row 0), described by the row map
`self.vmap` ([1, 16] uint32, staged per round: tt/speculate.py `map_word` = offset in the group, group length, the
group's conv-state slot); the module sequence is the width-1 plain step's (`_forward_verify`), with three verify
variants:
- ConvDecode / ConvOpDecode `verify=cand, vmap=...` (kernels conv_decode.cpp VERIFY, conv_fused.cpp CF_VERIFY,
  conv_tail.hpp `bx_only` / `verify_context`): TRISC computes every row's Bx right after in_proj; the DM side stages the
  state rows [0, W) in a scratch CB (7 / 19) and builds each row's left context from its group's slot state plus the
  group's earlier Bx rows (row j of a group: c[j .. j + 5] of c = [s0..s5 of the slot, Bx_r0, Bx_r0+1, ...]), then the
  usual state partial + tail runs (row j = the sequential step's arithmetic); every row's Bx goes to
  `self.cand[layer]`, the state is not written.
- AttnNormQKVFront `verify=True, vmap=...` (attn_decode.cpp FRONT_VERIFY, reader common arg 11 = the row map): per
  32-row K/V cache tile of a group one leader (the group's first row in that tile) reads, patches and writes the tile;
  the group's other rows in it send their packed BFP8 head rows into the leader's CB 26 and increment its semaphore 1.
  Each row has its own position and page-table row (its sequence's pages).
- SDPADecode, FusedMLPDecode, norms, head at width 2 / 4 / 8 / 16: batch-invariant as before.
tt/speculate.py SpecCommit (kernels/spec_commit.cpp) then sets, per group, its slot's s_j <- c[a + 1 + j] for the
group's accepted count a.  Evidence: tests/test_speculate.py part (1) (logs/spec13/t_full.json): 15 rounds of 1-8
sequences at widths 2-16 with permuted slots, rows up to each group's accepted count bitwise the sequential 1-row
steps, committed state bitwise, every row bitwise the same group verified alone; rows crossing 32-row KV tiles and
64-token pages.  Device cost (back-to-back trace replays, every row live, logs/spec13/timing.json; tt/speculate.py
StepCost: base by width + 0.0144 ms per live row and 1K positions, within 1.5%): verify argmax 2 / 4 / 8 / 16 rows at
position 128: 3.64 / 3.78 / 4.13 / 4.77 ms, at 32768: 4.61 / 5.68 / 7.95 / 12.47 ms (the plain step: 1 / 16 rows
3.49 / 4.34 ms at 128, 4.00 / 12.04 at 32768).
