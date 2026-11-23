#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4-12B single-P150 (1x1) runner: teacher-forced quality eval, greedy generation, perf.

Runs inside the TT image with the work tree bind-mounted over models/demos/gemma4.

Weights come either from an HF directory (``--model-dir``, quantized at load per the
precision plan ``--plan`` into the TT tensor cache ``--cache``) or from a native checkpoint
(``--native``; see native_checkpoint.py; refuses to run without a matching equivalence proof
unless ``--allow-unproven`` is given, which only the proof recipe uses).

Modes
  teacher : for each record of --input (jsonl {id, domain, token_ids}) run one padded prefill,
            take post-norm hidden states, run the on-device LM head + softcap in row chunks,
            read logits to host (fp32 log_softmax) and write <out>/<id>.npz
            {tokens, nll, argmax, kl} (kl = KL(ref || tt) vs --ref top-64 + tail bucket).
            --save-logits N stores the full bf16 logits of the first N records (proof runs).
  generate: greedy chat generation for --prompts (jsonl {id, prompt} or gsm8k prompts.jsonl
            with prompt_ids) through SingleUserRuntime (chunked prefill + traced greedy or
            speculative decode), writes <out>/generations.jsonl.
  perf    : prefill + traced greedy decode timing at given context lengths (--perf-ctx).
  decodetf: prefill + teacher-forced traced decode logits (LogitsDecoder).
KV: --max-seq-len positions of full-attention cache; --ring-tokens N makes the sliding-window caches
rings of N positions (see tt/single_user.py), --prefill-chunk the prefill chunk.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import ttnn

EOS_IDS = {1, 106, 50}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def dram(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {
        "allocated_GB": round(v.total_bytes_allocated_per_bank * v.num_banks / 1e9, 3),
        "free_GB": round(v.total_bytes_free_per_bank * v.num_banks / 1e9, 3),
        "largest_free_GB": round(v.largest_contiguous_bytes_free_per_bank * v.num_banks / 1e9, 3),
    }


def kl_from_ref(ref_ids, ref_lp, tt_lp):
    """KL(ref || tt) over the reference top-k plus one tail bucket. tt_lp: [N, V] fp32 log-probs."""
    p = np.exp(ref_lp.astype(np.float64))
    q_lp = np.take_along_axis(tt_lp, ref_ids.astype(np.int64), axis=1).astype(np.float64)
    q = np.exp(q_lp)
    tail_p = np.clip(1.0 - p.sum(1), 0.0, None)
    tail_q = np.clip(1.0 - q.sum(1), 1e-12, None)
    kl = (p * (ref_lp - q_lp)).sum(1) + tail_p * np.log(np.clip(tail_p, 1e-12, None) / tail_q)
    return np.clip(kl, 0.0, None).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# model construction


def build(args, max_seq_len, block_size=64):
    from models.demos.gemma4.tt.common import create_tt_model
    from models.demos.gemma4.tt.generator import _load_text_tokenizer
    from models.tt_transformers.tt.common import PagedAttentionConfig

    trace_region = int(os.environ.get("GEMMA4_TRACE_REGION_SIZE", 200_000_000))
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1), num_command_queues=1, trace_region_size=trace_region
    )
    n_blocks = math.ceil(max_seq_len / block_size)
    pac = PagedAttentionConfig(block_size=block_size, max_num_blocks=n_blocks)

    identity = {}
    if args.native:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import native_checkpoint as nc

        model_path, state_dict, cache_path, identity = nc.prepare_native_runtime(
            args.native, allow_unproven=args.allow_unproven
        )
        os.environ["TT_CACHE_PATH"] = cache_path
        os.environ["GEMMA4_PRECISION_PLAN"] = str(Path(args.native) / "precision_plan.json")
    else:
        model_path = args.model_dir
        state_dict = dummy_state_dict(model_path, args.dummy_layers) if args.dummy_layers else None
        if args.override_from:
            state_dict = mixed_state_dict(model_path, args.override_from, args.override_regex)
        if args.cache_only:
            state_dict = cache_only_state_dict(model_path)
        if args.cache:
            os.environ["TT_CACHE_PATH"] = args.cache
        if args.plan:
            os.environ["GEMMA4_PRECISION_PLAN"] = args.plan
    t0 = time.time()
    tokenizer = _load_text_tokenizer(model_path)
    tokenizer.stop_tokens = sorted(EOS_IDS)
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=pac,
        state_dict=state_dict,
        num_layers=args.dummy_layers or None,
        sliding_ring_tokens=args.ring_tokens or None,
    )
    del state_dict
    load_s = time.time() - t0
    mem = dram(mesh)
    logger.info(f"LOADED in {load_s:.1f}s dram={mem}")
    from models.demos.gemma4.tt.single_user import SingleUserRuntime

    rt = SingleUserRuntime(model, mesh, tt_kv_cache, max_seq_len, block_size=block_size,
                           ring_tokens=args.ring_tokens or None, prefill_chunk=args.prefill_chunk)
    return dict(
        mesh=mesh,
        model=model,
        rt=rt,
        kv=[tt_kv_cache],
        tokenizer=tokenizer,
        load_s=load_s,
        dram_after_load=mem,
        identity=identity,
        model_path=model_path,
    )


def dummy_state_dict(model_path, n_layers):
    """Random-weight text state dict for plumbing smoke tests (first n_layers layers)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import native_checkpoint as nc

    cfg = json.load(open(Path(model_path) / "config.json"))
    g = torch.Generator().manual_seed(0)
    sd = {}
    for name, shape in nc.text_shapes(cfg).items():
        if ".layers." in name and int(name.split(".layers.")[1].split(".")[0]) >= n_layers:
            continue
        if len(shape) == 2:
            sd[name] = (torch.randn(shape, generator=g) * 0.02).to(torch.bfloat16)
        else:
            sd[name] = torch.ones(shape, dtype=torch.bfloat16)
    return sd


def cache_only_state_dict(model_path):
    """Warm-cache load: 1-D tensors from the checkpoint, 2-D matrices as meta placeholders (every
    matrix must then come from the TT tensor cache; a miss fails loudly on the meta tensor)."""
    from safetensors import safe_open

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import native_checkpoint as nc

    cfg = json.load(open(Path(model_path) / "config.json"))
    sd = {}
    with safe_open(str(Path(model_path) / "model.safetensors"), framework="pt") as fh:
        for name, shape in nc.text_shapes(cfg).items():
            sd[name] = torch.empty(shape, dtype=torch.bfloat16, device="meta") if len(shape) == 2 else fh.get_tensor(name)
    return sd


def mixed_state_dict(base_dir, override_dir, regex):
    """Base checkpoint tensors, with every text tensor whose name matches ``regex`` taken from
    ``override_dir`` instead (e.g. MLP weights from the QAT checkpoint)."""
    import re

    from safetensors import safe_open

    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    sd = Gemma4ModelArgs.load_state_dict(base_dir)
    pat = re.compile(regex)
    n = 0
    for f in sorted(Path(override_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                if k in sd and pat.search(k):
                    t = fh.get_tensor(k).to(torch.bfloat16)
                    assert t.shape == sd[k].shape, k
                    sd[k] = t
                    n += 1
    logger.info(f"override: {n} tensors matching {regex!r} taken from {override_dir}")
    assert n > 0
    return sd


def runtime_identity(args):
    root = Path("/home/container_app_user/tt-metal/models/demos/gemma4")
    files = sorted(p for ext in ("*.py", "*.cpp", "*.hpp") for p in root.rglob(ext) if "__pycache__" not in p.parts and "tests" not in p.parts)
    src = hashlib.sha256()
    for p in files:
        src.update(str(p.relative_to(root)).encode())
        src.update(sha256_file(p).encode())
    env = {k: v for k, v in os.environ.items() if k.startswith("GEMMA4_") and k != "GEMMA4_PRECISION_PLAN"}
    plan = os.environ.get("GEMMA4_PRECISION_PLAN")
    return {
        "gemma4_tree_sha256": src.hexdigest(),
        "tool_sha256": sha256_file(__file__),
        "env": env,
        "precision_plan": json.load(open(plan)) if plan else None,
    }


# ----------------------------------------------------------------------------------------------
# teacher forcing


def padded_len(n):
    from models.tt_transformers.tt.common import get_padded_prefill_len

    return get_padded_prefill_len(n)


def teacher_logits_rows(ctx, token_ids, rows=256):
    """Yield (start, fp32 logits [r, V]) for positions 0..T-2 after one padded prefill."""
    model = ctx["model"]
    T = len(token_ids)
    P = padded_len(T)
    toks = torch.zeros(1, P, dtype=torch.long)
    toks[0, :T] = torch.tensor(token_ids)
    if P > ctx["rt"].prefill_chunk:
        raise ValueError(f"teacher mode prefills one chunk; record of {T} tokens")
    x, *_ = model.prepare_inputs_prefill(toks, page_table=None)
    tt_pt = {k: ttnn.from_torch(t[:, : P // ctx["rt"].block_size], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                device=ctx["mesh"]) for k, t in ctx["rt"].tables.items()}
    model._prefill_trace_mode = True
    try:
        hidden = model.ttnn_prefill_forward(x, page_table=tt_pt, kv_cache=ctx["kv"][0], get_last_token=-1)
    finally:
        model._prefill_trace_mode = False
    ttnn.synchronize_device(ctx["mesh"])
    V = model.vocab_size
    for s in range(0, T - 1, rows):
        e = min(s + rows, T - 1)
        r0 = s
        r1 = min(((e + 31) // 32) * 32, P)
        sl = ttnn.slice(hidden, (0, 0, r0, 0), (1, 1, r1, hidden.shape[-1]))
        logits = model._apply_lm_head(sl, is_decode=False)
        out = ttnn.to_torch(logits).reshape(-1, logits.shape[-1])[: e - s, :V].float()
        logits.deallocate(True)
        yield s, out
    hidden.deallocate(True)


def run_teacher(args, ctx):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.limit:
        records = records[: args.limit]
    ref = Path(args.ref) if args.ref else None
    done = []
    t_all = time.time()
    for n, rec in enumerate(records):
        path = out / f"{rec['id']}.npz"
        ids = rec["token_ids"]
        T = len(ids)
        if path.exists() and n >= args.save_logits:
            done.append(rec["id"])
            continue
        rpath = ref / f"{rec['id']}.npz" if ref else None
        r = np.load(rpath) if rpath is not None and rpath.exists() else None
        nll = np.zeros(T - 1, np.float32)
        am = np.zeros(T - 1, np.int32)
        kl = np.full(T - 1, np.nan, np.float32)
        top_ids = np.zeros((T - 1, 256), np.int32)
        top_lp = np.zeros((T - 1, 256), np.float16)
        full = np.zeros((T - 1, ctx["model"].vocab_size), np.uint16) if n < args.save_logits else None
        tgt = torch.tensor(ids[1:], dtype=torch.long)
        t0 = time.time()
        for s, lg in teacher_logits_rows(ctx, ids):
            e = s + lg.shape[0]
            if full is not None:
                full[s:e] = lg.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            lp = torch.log_softmax(lg, dim=-1)
            nll[s:e] = (-lp.gather(1, tgt[s:e, None])[:, 0]).numpy()
            am[s:e] = lg.argmax(-1).numpy()
            tv, ti = lp.topk(256, dim=-1)
            top_ids[s:e] = ti.numpy()
            top_lp[s:e] = tv.numpy().astype(np.float16)
            if r is not None:
                kl[s:e] = kl_from_ref(r["top_ids"][s:e], r["top_lp"][s:e], lp.numpy())
        dt = time.time() - t0
        np.savez(path, tokens=np.asarray(ids, np.int32), nll=nll, argmax=am, kl=kl, top_ids=top_ids, top_lp=top_lp)
        if full is not None:
            np.save(out / f"{rec['id']}.logits.npy", full)
        msg = f"REC {n+1}/{len(records)} {rec['id']} T={T} ppl={math.exp(float(nll.mean())):.4f} {dt:.1f}s"
        if r is not None:
            msg += f" ref_ppl={math.exp(float(r['nll'].mean())):.4f} top1={float((am == r['argmax']).mean()):.4f} kl={float(kl.mean()):.4f}"
        logger.info(msg)
        done.append(rec["id"])
    meta = {
        "status": "complete",
        "mode": "teacher",
        "records": done,
        "input": args.input,
        "input_sha256": sha256_file(args.input),
        "seconds": time.time() - t_all,
        "load_seconds": ctx["load_s"],
        "dram_after_load": ctx["dram_after_load"],
        "runtime": runtime_identity(args),
        "weights": {"native": args.native, "model_dir": args.model_dir, "cache": args.cache,
                    "override_from": args.override_from, "override_regex": args.override_regex if args.override_from else None},
        "native_identity": ctx["identity"],
        "saved_logits": [r["id"] for r in records[: args.save_logits]],
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))


def run_decodetf(args, ctx):
    """Teacher-forced DECODE check: prefill tokens[:T-D], then feed the next D reference tokens
    one at a time through the (traced) decode path; score positions T-D-1 .. T-2 against the ref."""
    model = ctx["model"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in open(args.input) if l.strip()][: args.limit or None]
    D = args.decode_tf
    rows = []
    for rec in records:
        ids = rec["token_ids"]
        T = len(ids)
        P0 = T - D
        r = np.load(Path(args.ref) / f"{rec['id']}.npz") if args.ref else None
        rt = ctx["rt"]
        logits = [rt.prefill(ids[:P0])]
        for i in range(D - 1):
            logits.append(rt.decode_logits(ids[P0 + i], P0 + i))
        L = torch.stack(logits)
        lp = torch.log_softmax(L, -1)
        tgt = torch.tensor(ids[P0:], dtype=torch.long)
        s0 = P0 - 1
        nll = (-lp.gather(1, tgt[:, None])[:, 0]).numpy()
        am = L.argmax(-1).numpy().astype(np.int32)
        bits = L.to(torch.bfloat16).view(torch.int16).numpy()
        extra = {"logits_bf16_bits": bits} if args.save_logits else {}
        # per-row sha256 of the bf16 logits: bitwise comparison without storing D x vocab values
        extra["row_sha256"] = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in bits])
        row = {"id": rec["id"], "positions": [s0, s0 + D], "nll_mean": float(nll.mean())}
        if r is not None:
            kl = kl_from_ref(r["top_ids"][s0 : s0 + D], r["top_lp"][s0 : s0 + D], lp.numpy())
            extra["kl"] = kl
            row.update(dNLL=float(nll.mean() - r["nll"][s0 : s0 + D].mean()),
                       top1=float((am == r["argmax"][s0 : s0 + D]).mean()), kl_mean=float(kl.mean()))
        np.savez(out / f"{rec['id']}.npz", start=s0, nll=nll, argmax=am, **extra)
        logger.info(f"DTF {row}")
        rows.append(row)
    (out / "decodetf.json").write_text(json.dumps({"rows": rows, "runtime": runtime_identity(args)}, indent=1))


# ----------------------------------------------------------------------------------------------
# generation


def build_assistant(args, ctx):
    """Load the it-assistant drafter (weights from --assistant, TT tensor cache under /work/cache)."""
    from models.demos.gemma4.tt.assistant.model import Gemma4AssistantModel
    from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs

    model = ctx["model"]
    cache = args.assistant_cache or f"/work/cache/assistant-{args.assistant_dtype}"
    ctx["drafter_identity"] = {"hf_dir": args.assistant, "dtype": args.assistant_dtype}
    if (Path(args.assistant) / "drafter_manifest.json").exists():
        import native_checkpoint as nc

        _, cache, ctx["drafter_identity"] = nc.prepare_drafter(args.assistant, ctx["identity"],
                                                            allow_unproven=args.allow_unproven or args.allow_unproven_drafter)
        if ctx["drafter_identity"]["dtype"] != args.assistant_dtype:
            raise ValueError(f"drafter dir holds {ctx['drafter_identity']['dtype']} tensors, --assistant-dtype {args.assistant_dtype}")
    # bfp8-lm4: BFP8 drafter with a BFP4 LM head (drafter precision only moves acceptance, never output)
    dtype, lm_head_dtype = {"bf16": (ttnn.bfloat16, None), "bfp8": (ttnn.bfloat8_b, None),
                            "bfp4": (ttnn.bfloat4_b, None), "bfp8-lm4": (ttnn.bfloat8_b, ttnn.bfloat4_b)}[args.assistant_dtype]
    hf_config = Gemma4AssistantArgs.load_hf_config(args.assistant)
    a_args = Gemma4AssistantArgs.from_hf_config(hf_config)
    a_args.model_cache_path = Path(cache)
    t0 = time.time()
    sd = Gemma4AssistantArgs.load_state_dict(args.assistant, dummy_weights=False)
    assistant = Gemma4AssistantModel(
        mesh_device=ctx["mesh"],
        assistant_args=a_args,
        target_model=model,
        state_dict=sd,
        ccl_manager=model.ccl_manager,
        dtype=dtype,
        lm_head_dtype=lm_head_dtype,
        tensor_cache_path=str(a_args.weight_cache_path(dtype)),
        mesh_config=model.mesh_config,
    )
    del sd
    ctx["spec"] = ctx["rt"].attach_assistant(assistant, args.draft_len)
    logger.info(f"ASSISTANT loaded in {time.time() - t0:.1f}s dtype={args.assistant_dtype} K={args.draft_len} dram={dram(ctx['mesh'])}")


def get_decoder(ctx):
    return ctx["rt"].greedy


def prefill_token(ctx, prompt_ids):
    """Prefill; returns (first greedy token, prefill seconds)."""
    t0 = time.time()
    logits = ctx["rt"].prefill(list(prompt_ids))
    return int(logits.argmax()), time.time() - t0


def greedy(ctx, prompt_ids, max_new, stop=True):
    """Returns (tokens, prefill s, per-step seconds): SingleUserRuntime prefill, then the traced
    speculative decoder (with --assistant) or the traced on-device-argmax GreedyDecoder."""
    T = len(prompt_ids)
    nxt, t_prefill = prefill_token(ctx, prompt_ids)
    out = [nxt]
    if max_new <= 1 or (stop and nxt in EOS_IDS):
        return out, t_prefill, []
    stop_ids = EOS_IDS if stop else ()
    t1 = time.time()
    if "spec" in ctx:
        toks, cycles = ctx["rt"].spec_generate(nxt, T, max_new - 1, stop_ids=stop_ids)
        ctx["last_cycles"] = cycles
        out += toks
        dt = time.time() - t1
        return out, t_prefill, [dt / max(1, len(toks))] * len(toks)
    toks, times = ctx["rt"].greedy_generate(nxt, T, max_new - 1, stop_ids=stop_ids)
    out += toks
    return out, t_prefill, list(np.diff([t1] + times))


def run_generate(args, ctx):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = ctx["tokenizer"]
    rows = [json.loads(l) for l in open(args.prompts) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    res = []
    with open(out / "generations.jsonl", "w") as fh:
        for r in rows:
            if "prompt_ids" in r:
                ids = r["prompt_ids"]
            else:
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=True
                )
                if isinstance(ids, dict) or hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
                ids = list(ids)
            gen_ids, tp, steps = greedy(ctx, ids, args.max_new, stop=not args.no_stop)
            text = tok.decode([t for t in gen_ids if t not in EOS_IDS], skip_special_tokens=False)
            steady = steps[1:] if len(steps) > 1 else steps
            row = {
                "id": r.get("id", r.get("idx")),
                "idx": r.get("idx"),
                "prompt_tokens": len(ids),
                "prompt_ids": list(ids),
                "gen_ids": gen_ids,
                "text": text,
                "prefill_s": tp,
                "decode_tok_s": (len(steady) / sum(steady)) if steady else None,
            }
            if "last_cycles" in ctx:
                cyc = ctx.pop("last_cycles")
                row["spec"] = {"cycles": len(cyc), "tokens_per_cycle": len(gen_ids[1:]) / max(1, len(cyc)),
                               "accepted": [c["accepted"] for c in cyc],
                               "cycle_ms_median": 1000 * float(np.median([c["s"] for c in cyc[1:]] or [0]))}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            sp = row.get("spec")
            spec_s = f" tok/cycle={sp['tokens_per_cycle']:.2f} cycle_ms={sp['cycle_ms_median']:.1f}" if sp else ""
            logger.info(f"GEN {row['id']} n={len(gen_ids)} tok/s={row['decode_tok_s']}{spec_s} :: {text[:160]!r}")
            res.append(row)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "mode": "generate",
                "n": len(res),
                "dram_after_load": ctx["dram_after_load"],
                "runtime": runtime_identity(args),
                "weights": {"native": args.native, "model_dir": args.model_dir, "cache": args.cache},
                "native_identity": ctx["identity"],
                "prompts_sha256": sha256_file(args.prompts),
                "max_new": args.max_new,
                "decoder": "spec" if "spec" in ctx else "greedy",
                "ring_tokens": args.ring_tokens,
                "prefill_chunk": args.prefill_chunk,
                "max_seq_len": args.max_seq_len,
                "no_stop": args.no_stop,
                "spec": {"draft_len": args.draft_len, "drafter": ctx["drafter_identity"]} if "spec" in ctx else None,
            },
            indent=1,
        )
    )


# ----------------------------------------------------------------------------------------------
# perf


def run_perf(args, ctx):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"dram_after_load": ctx["dram_after_load"], "runtime": runtime_identity(args), "runs": []}
    rng = np.random.default_rng(0)
    for L in [int(x) for x in args.perf_ctx.split(",")]:
        ids = [2] + rng.integers(1000, 200000, L - 1).tolist()
        # warm (compile / capture), then measured run
        for rep in range(2):
            nxt, tp = prefill_token(ctx, ids)
            t1 = time.time()
            _, stamps = ctx["rt"].greedy_generate(nxt, L, args.perf_steps)
            times = list(np.diff([t1] + stamps))
            steady = times[3:]
            row = {
                "ctx": L,
                "mode": "greedy-trace",
                "rep": rep,
                "prefill_s": tp,
                "prefill_tok_s": L / tp,
                "decode_ms_median": 1000 * float(np.median(steady)),
                "decode_tok_s": len(steady) / float(np.sum(steady)),
                "decode_ms_mean": 1000 * float(np.mean(steady)),
                "dram": dram(ctx["mesh"]),
            }
            logger.info(f"PERF {row}")
            results["runs"].append(row)
    results["dram_end"] = dram(ctx["mesh"])
    (out / "perf.json").write_text(json.dumps(results, indent=1))


PASSKEYS = {"begin": "CEDAR-7419-QUARTZ", "middle": "HARBOR-5096-VIOLET", "end": "ORBIT-2863-MARBLE"}


def passkey_prompt(tok, context):
    """Exactly ``context`` token ids: Gemma chat turn holding a numbered-facts archive with passkeys at
    its begin, middle (a record boundary) and end, asking for the three keys."""
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    prefix = [2] + enc("<|turn>user\nRead this factual archive and retrieve its three passkeys. Ignore ordinary "
                       f"numbered facts.\nBEGIN_PASSKEY: {PASSKEYS['begin']}\nArchive facts follow:\n")
    middle = enc(f"MIDDLE_PASSKEY: {PASSKEYS['middle']}\n")
    suffix = enc(f"\nEnd of archive.\nEND_PASSKEY: {PASSKEYS['end']}\nReturn only the BEGIN_PASSKEY, MIDDLE_PASSKEY "
                 "and END_PASSKEY values from this archive, in that order, one per line. Do not explain.<turn|>\n"
                 "<|turn>model\n<|channel>thought\n<channel|>")
    body = context - len(prefix) - len(middle) - len(suffix)
    filler, bounds, n = [], [0], 1
    while len(filler) < body:
        filler += enc("".join(f"Record {k:06d}: {k} plus 1 equals {k + 1}.\n" for k in range(n, n + 32)))
        bounds.append(len(filler))
        n += 32
    filler = filler[:body]
    half = min((b for b in bounds if b <= body), key=lambda b: abs(b - body // 2))
    ids = prefix + filler[:half] + middle + filler[half:] + suffix
    assert len(ids) == context
    return ids


def run_passkey(args, ctx):
    """Passkey retrieval at exact contexts (--perf-ctx): chunked prefill timing, greedy (speculative
    with --assistant) decode of --max-new tokens, retrieved keys."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = ctx["tokenizer"]
    rows = []
    for L in [int(x) for x in args.perf_ctx.split(",")]:
        ids = passkey_prompt(tok, L)
        gen_ids, tp, steps = greedy(ctx, ids, args.max_new)
        text = tok.decode([t for t in gen_ids if t not in EOS_IDS], skip_special_tokens=False)
        row = {"ctx": L, "prefill_s": tp, "prefill_tok_s": L / tp, "n": len(gen_ids),
               "decode_tok_s": (len(steps) / sum(steps)) if steps else None,
               "retrieved": {k: v in text for k, v in PASSKEYS.items()}, "text": text, "gen_ids": gen_ids,
               "dram": dram(ctx["mesh"])}
        if "last_cycles" in ctx:
            cyc = ctx.pop("last_cycles")
            row["spec"] = {"cycles": len(cyc), "tokens_per_cycle": len(gen_ids[1:]) / max(1, len(cyc)),
                           "cycle_ms_median": 1000 * float(np.median([c["s"] for c in cyc[1:]] or [0]))}
        logger.info(f"PASSKEY { {k: v for k, v in row.items() if k != 'gen_ids'} }")
        rows.append(row)
        (out / "passkey.json").write_text(json.dumps({"rows": rows, "runtime": runtime_identity(args)}, indent=1))


_NOT_OPS = {"to_torch", "from_torch", "synchronize_device", "open_mesh_device", "close_mesh_device", "as_tensor",
            "copy_host_to_device_tensor", "deallocate", "get_memory_view", "begin_trace_capture", "end_trace_capture",
            "execute_trace", "release_trace", "load_tensor", "dump_tensor", "ReadDeviceProfiler"}


def _category(stack):
    for fr in stack:
        f = fr.filename
        if "/models/demos/gemma4/tt/" not in f:
            continue
        rel = f.split("/models/demos/gemma4/tt/")[1]
        if rel == "shared_mlp.py":
            return "mlp"
        if rel.startswith("attention/"):
            return "attention"
        if rel == "rms_norm.py":
            return "norm"
        if rel == "model.py" and fr.function == "_apply_lm_head":
            return "lm_head"
        return rel.replace(".py", "") + ":" + fr.function
    return "other"


def _device_program_ms(csv_path, skip_ids):
    """{run host ID: device ms} from the device profiler CSV (FW zone span over all cores)."""
    import csv

    freq = 1350.0
    span = {}
    with open(csv_path) as f:
        head = f.readline()
        if "CHIP_FREQ[MHz]:" in head:
            freq = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rd = csv.reader(f)
        cols = [c.strip() for c in next(rd)]
        i_t, i_id, i_zone, i_type = (cols.index(c) for c in ("time[cycles since reset]", "run host ID", "zone name", "type"))
        for row in rd:
            if len(row) <= i_type or not row[i_zone].strip().endswith("-FW"):
                continue
            rid = int(row[i_id])
            if rid in skip_ids:
                continue
            t = int(row[i_t])
            lo, hi = span.get(rid, (t, t))
            span[rid] = (min(lo, t), max(hi, t))
    return {k: (v[1] - v[0]) / freq / 1000.0 for k, v in sorted(span.items())}


def run_specparts(args, ctx):
    """Traced timing of the speculative cycle parts (drafter K steps / P-row verify / whole cycle)."""
    spec, mesh = ctx["spec"], ctx["mesh"]
    L = int(args.perf_ctx.split(",")[0])
    ids = [2] + np.random.default_rng(0).integers(1000, 200000, L - 1).tolist()
    nxt, _ = prefill_token(ctx, ids)
    spec.prepare_eager(nxt, L)
    b = spec.buf

    def draft():
        h = ttnn.unsqueeze_to_4D(ttnn.embedding(b["row"], b["hsrc"], layout=ttnn.TILE_LAYOUT))
        tok = b["tok"]
        outs = []
        for _ in range(spec.K):
            logits, h2 = spec.assistant.step(tok, h, spec.shared_kv, spec.page_tables, b["d_pu"], b["d_pi"])
            h.deallocate(True)
            h = h2
            tok = ttnn.reshape(spec._argmax_rows(logits, 1), (1, 1))
            outs.append(tok)
        return outs + [h]

    vx0 = ttnn.from_torch(torch.full((1, spec.P), 100, dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

    def verify():
        if getattr(spec, "batched", False):
            lg, hid = spec.model.ttnn_rows_forward(vx0, b["v_pu"], None, spec.page_tables, kv_cache=spec.kv_cache,
                                                   batched=(b["v_pall"], spec.page_tables_rep))
        else:
            lg, hid = spec.model.ttnn_rows_forward(vx0, b["v_pu"], [b[f"v_pi{i}"] for i in range(spec.P)], spec.page_tables,
                                                   kv_cache=spec.kv_cache)
        return [spec._argmax_rows(lg, spec.P), hid]

    dec = get_decoder(ctx)
    dec.start(nxt, L)

    def decode():
        return [dec._forward()]

    res = {}
    parts = [("decode_fwd", decode), ("verify", verify)]
    if spec.K:
        parts += [("draft", draft), ("cycle", spec._body)]
    for name, fn in parts:
        for t in fn():
            t.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs = fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ts.append(time.perf_counter() - t0)
        ttnn.release_trace(mesh, tid)
        for t in outs:
            t.deallocate(True)
        res[name] = 1000 * float(np.median(ts))
    logger.info(f"SPECPARTS ctx={L} K={spec.K} {res}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "specparts.json").write_text(json.dumps({"ctx": L, "K": spec.K, "ms": res}, indent=1))


def run_profile(args, ctx):
    """Per-op decode profile: eager decode steps with a synchronize after every ttnn op (host wall
    per op, including dispatch) and, if TT_METAL_DEVICE_PROFILER=1, device kernel time per program
    matched to the op sequence. Categories come from the calling gemma4 module."""
    import inspect
    from collections import defaultdict

    model, mesh, rt = ctx["model"], ctx["mesh"], ctx["rt"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L = int(args.perf_ctx.split(",")[0])
    ids = [2] + np.random.default_rng(0).integers(1000, 200000, L - 1).tolist()
    rt.prefill(ids)
    pos = torch.tensor([L])
    dec = rt.logits_decoder

    def step(tok):  # eager decode forward at pos
        dec.inputs = [ttnn.to_device(h, device=mesh) for h in dec._host_inputs(tok, int(pos))]
        dec._forward().deallocate(True)
    if "spec" in ctx:  # one eager speculative cycle (K drafter steps + P-row verify)
        spec = ctx["spec"]
        spec.prepare_eager(100, L)

        def step(tok):
            for t in spec._body():
                t.deallocate(True)
    for _ in range(3):  # compile
        step(100)
        pos += 1
    ttnn.synchronize_device(mesh)
    eager = []
    for _ in range(5):
        t0 = time.time()
        step(100)
        ttnn.synchronize_device(mesh)
        eager.append(time.time() - t0)
        pos += 1
    prof = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    csv_path = "/home/container_app_user/tt-metal/generated/profiler/.logs/profile_log_device.csv"
    calls = []
    patched = {}

    def wrap(ns, name, fn):
        def w(*a, **k):
            stack = inspect.stack(0)[1:30]
            cat = _category(stack[:11])
            if any(fr.filename.endswith("assistant/model.py") for fr in stack):
                cat = "draft/" + cat
            ttnn.synchronize_device(mesh)
            id0 = ttnn._ttnn.get_device_operation_id()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            ttnn.synchronize_device(mesh)
            dt = time.perf_counter() - t0
            calls.append((cat, f"{ns.__name__.replace('ttnn.', '')}.{name}", dt, id0, ttnn._ttnn.get_device_operation_id()))
            return r

        return w

    for ns in (ttnn, ttnn.experimental, ttnn.transformer):
        for name in dir(ns):
            fn = getattr(ns, name)
            if type(fn).__name__ == "FastOperation" and name not in _NOT_OPS:
                patched[(ns, name)] = fn
                setattr(ns, name, wrap(ns, name, fn))
    try:
        n_steps = 3
        if prof:
            ttnn.ReadDeviceProfiler(mesh)  # drain device-side buffers before the profiled steps
        for _ in range(n_steps):
            step(100)
            pos += 1
            if prof:
                patched_read = patched[(ttnn, "ReadDeviceProfiler")] if (ttnn, "ReadDeviceProfiler") in patched else ttnn.ReadDeviceProfiler
                patched_read(mesh)
    finally:
        for (ns, name), fn in patched.items():
            setattr(ns, name, fn)
    ctx["profile_pending"] = dict(calls=calls, n_steps=n_steps, eager=eager, L=L, out=out, prof=prof, csv=csv_path)


def finish_profile(p):
    """Runs after close_mesh_device (the device profiler CSV is written at close). The patched
    steps are the last device work, so the op sequence is matched to the LAST programs."""
    from collections import defaultdict

    calls, n_steps = p["calls"], p["n_steps"]
    raw = _device_program_ms(p["csv"], set()) if p["prof"] and os.path.exists(p["csv"]) else {}
    dev = defaultdict(float)
    for rid, ms in raw.items():  # run host ID = device_operation_id * 1024 + program index
        dev[rid >> 10] += ms
    ids = sorted(dev)
    import bisect

    per = defaultdict(lambda: [0, 0.0, 0.0])
    covered = 0
    for cat, op, dt, id0, id1 in calls:
        lo, hi = bisect.bisect_left(ids, id0), bisect.bisect_left(ids, id1)
        d = sum(dev[i] for i in ids[lo:hi])
        covered += hi - lo
        e = per[(cat, op)]
        e[0] += 1
        e[1] += dt
        e[2] += d
    matched = bool(dev) and covered > 0
    logger.info(f"device programs total={len(dev)} attributed to profiled ops={covered} op calls={len(calls)}")
    table = sorted(
        ({"category": c, "op": o, "calls_per_step": n / n_steps, "host_sync_ms_per_step": 1000 * t / n_steps,
          "device_ms_per_step": (d / n_steps) if matched else None} for (c, o), (n, t, d) in per.items()),
        key=lambda r: -r["host_sync_ms_per_step"],
    )
    cats = defaultdict(lambda: [0.0, 0.0, 0])
    for r in table:
        key = r["category"].split(":")[0] + ("/matmul" if r["op"].endswith((".linear", ".matmul")) else "")
        cats[key][0] += r["host_sync_ms_per_step"]
        cats[key][1] += r["device_ms_per_step"] or 0.0
        cats[key][2] += r["calls_per_step"]
    res = {
        "ctx": p["L"],
        "eager_step_ms": 1000 * float(np.median(p["eager"])),
        "ops_per_step": len(calls) / n_steps,
        "device_programs_matched": matched,
        "host_sync_total_ms_per_step": sum(r["host_sync_ms_per_step"] for r in table),
        "device_total_ms_per_step": (sum(r["device_ms_per_step"] for r in table)) if matched else None,
        "programs_attributed": covered,
        "by_category": {k: {"host_sync_ms": round(v[0], 3), "device_ms": round(v[1], 3), "calls": v[2]} for k, v in sorted(cats.items())},
        "ops": table,
    }
    (p["out"] / "profile.json").write_text(json.dumps(res, indent=1))
    logger.info(f"PROFILE eager={res['eager_step_ms']:.2f}ms ops/step={res['ops_per_step']} matched={matched} "
                f"dev_total={res['device_total_ms_per_step']} cats={res['by_category']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "generate", "perf", "decodetf", "profile", "specparts", "passkey"],
                    required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--plan")
    ap.add_argument("--cache")
    ap.add_argument("--native")
    ap.add_argument("--allow-unproven", action="store_true")
    ap.add_argument("--input")
    ap.add_argument("--ref")
    ap.add_argument("--prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-logits", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--no-stop", action="store_true", help="generate: ignore EOS (fixed-length streams)")
    ap.add_argument("--ring-tokens", type=int, default=0,
                    help="sliding-window caches as rings of this many positions (0: unbounded)")
    ap.add_argument("--prefill-chunk", type=int, default=16384)
    ap.add_argument("--assistant", help="it-assistant HF dir: exact speculative decoding (SpecGreedyDecoder)")
    ap.add_argument("--allow-unproven-drafter", action="store_true", help="drafter dir without spec_equivalence.json (spec-record runs)")
    ap.add_argument("--assistant-dtype", choices=["bf16", "bfp8", "bfp4", "bfp8-lm4"], default="bf16")
    ap.add_argument("--assistant-cache")
    ap.add_argument("--draft-len", type=int, default=3)
    ap.add_argument("--perf-ctx", default="128,2048")
    ap.add_argument("--perf-steps", type=int, default=40)
    ap.add_argument("--decode-tf", type=int, default=64)
    ap.add_argument("--override-from", help="take tensors matching --override-regex from this HF dir")
    ap.add_argument("--override-regex", default=r"\.mlp\.")
    ap.add_argument("--cache-only", action="store_true", help="matrices must come from the warm TT cache")
    ap.add_argument("--dummy-layers", type=int, default=0, help="random weights, first N layers (smoke only)")
    args = ap.parse_args()
    assert bool(args.native) != bool(args.model_dir), "exactly one of --native / --model-dir"
    torch.set_num_threads(16)
    ctx = build(args, args.max_seq_len)
    if args.assistant:
        build_assistant(args, ctx)
    try:
        {"teacher": run_teacher, "generate": run_generate, "perf": run_perf, "decodetf": run_decodetf, "profile": run_profile,
         "specparts": run_specparts, "passkey": run_passkey}[args.mode](args, ctx)
    finally:
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
            ttnn.ReadDeviceProfiler(ctx["mesh"])
        ttnn.close_mesh_device(ctx["mesh"])
    if "profile_pending" in ctx:
        finish_profile(ctx["profile_pending"])
    logger.info("DONE")


if __name__ == "__main__":
    main()
