# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Per-step host-time profile of the vLLM engine loop around the LFM3 adapter (src/NOTES.md section 17).

    LFM3_STEP_PROFILE=N[:CSV] tools/serve_lfm3.sh ...   every N decode steps the EngineCore logs one line per
                                                       (decode path, active rows) seen in that window; CSV (a path
                                                       inside the container): one line per decode step

Installed by the adapter in the EngineCore process (vllm_tt_plugin forces vLLM's uni executor: scheduler, model runner
and adapter share one process); off (no hooks) unless the variable is set.  Parts, mean ms per decode step:
  period   schedule() start to the next schedule() start (decode steps only; a prefill in between drops the step)
  sched    TTScheduler.schedule
  build    TTModelRunner.build_model_input (_update_states + the decode input tensors / exact-sampling rows)
  adapter  Lfm3ForCausalLM.decode_forward, split into
           stage  host->device input copies, sampler params / histories
           launch execute_trace enqueues (non-blocking)
           wait   blocking readback of the step's tokens / logits (device time not hidden by the host work)
           post   host work on the read tokens (sampler flags, redraws, padding, decode-ahead matching)
           other  the rest (input normalization, width choice, slot remap)
  runner   execute_model minus build and adapter
  sample   TTModelRunner.sample_tokens (runner output, state append)
  update   Scheduler.update_from_output
  loop     period minus the parts above (engine loop, grammar bitmask, input / output queues, the output thread's
           serialization holding the GIL)
"""
from __future__ import annotations

import contextlib
import os
import time
from collections import defaultdict

from loguru import logger

PROF: "StepProfile | None" = None
PARTS = ("sched", "build", "adapter", "stage", "launch", "wait", "post", "runner", "sample", "update")
_NULL = contextlib.nullcontext()


class _Part:
    __slots__ = ("name", "t0")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        PROF.add(self.name, time.perf_counter() - self.t0)


def part(name: str):
    """`with part("stage"):` times its body into the current step's part (a shared no-op when profiling is off)."""
    return _NULL if PROF is None else _Part(name)


class StepProfile:
    def __init__(self, every: int, csv_path: str | None = None):
        self.every = max(1, every)
        self.cur = defaultdict(float)
        self.key = None  # (path, rows) when the current step decoded
        self.t_sched = None
        self.acc = {}
        self.steps = 0
        self.csv = None
        if csv_path:
            self.csv = open(csv_path, "a", buffering=1 << 16)
            self.csv.write("wall,path,rows,period," + ",".join(PARTS) + ",other,loop\n")

    def add(self, part: str, dt: float):
        self.cur[part] += dt

    def decoded(self, path: str, rows: int):
        self.key = (path, rows)

    def step_start(self, now: float):
        """Called at every schedule(): closes the previous step (recorded when it was a decode step)."""
        if self.t_sched is not None and self.key is not None:
            period = now - self.t_sched
            c = self.cur
            loop = period - sum(c[p] for p in ("sched", "build", "adapter", "runner", "sample", "update"))
            other = c["adapter"] - sum(c[p] for p in ("stage", "launch", "wait", "post"))
            row = self.acc.setdefault(self.key, defaultdict(float))
            row["n"] += 1
            row["period"] += period
            row["loop"] += loop
            for p in PARTS:
                row[p] += c[p]
            row["other"] += other
            if self.csv is not None:
                self.csv.write(f"{time.time():.4f},{self.key[0]},{self.key[1]},{1000 * period:.4f},"
                               + ",".join(f"{1000 * c[p]:.4f}" for p in PARTS)
                               + f",{1000 * other:.4f},{1000 * loop:.4f}\n")
            self.steps += 1
            if self.steps % self.every == 0:
                self.report()
                if self.csv is not None:
                    self.csv.flush()
        self.cur = defaultdict(float)
        self.key = None
        self.t_sched = now

    def report(self):
        for (path, rows), r in sorted(self.acc.items()):
            n = r["n"]
            ms = {k: 1000 * v / n for k, v in r.items() if k != "n"}
            logger.info(
                f"LFM3 step profile: path {path} rows {rows} steps {int(n)} period {ms['period']:.3f} ms | sched "
                f"{ms['sched']:.3f} build {ms['build']:.3f} adapter {ms['adapter']:.3f} (stage {ms['stage']:.3f} "
                f"launch {ms['launch']:.3f} wait {ms['wait']:.3f} post {ms['post']:.3f} other {ms['other']:.3f}) "
                f"runner {ms['runner']:.3f} sample {ms['sample']:.3f} update {ms['update']:.3f} loop {ms['loop']:.3f}")
        self.acc = {}


def _wrap(cls, name: str, part: str, start: bool = False, net: tuple = ()):
    """Time cls.name into `part` (net: parts measured inside it are subtracted: `runner` = execute_model minus build
    and adapter)."""
    fn = getattr(cls, name)

    def timed(self, *args, **kwargs):
        prof = PROF
        t0 = time.perf_counter()
        if start:
            prof.step_start(t0)
        inner0 = sum(prof.cur[p] for p in net)
        try:
            return fn(self, *args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            prof.add(part, dt - (sum(prof.cur[p] for p in net) - inner0))

    timed.__wrapped__ = fn
    setattr(cls, name, timed)


def install():
    """LFM3_STEP_PROFILE=N[:CSV]: hook the scheduler, the model runner and the adapter (once per process); every N
    decode steps log the means per (path, rows); CSV: also append one line per decode step to that file."""
    global PROF
    every, _, csv_path = os.environ.get("LFM3_STEP_PROFILE", "").partition(":")
    if not every or int(every) <= 0 or PROF is not None:
        return
    from vllm_tt_plugin.model_runner import TTModelRunner
    from vllm_tt_plugin.scheduler import TTScheduler

    from .lfm3_vllm import Lfm3ForCausalLM

    PROF = StepProfile(int(every), csv_path or None)
    _wrap(TTScheduler, "schedule", "sched", start=True)
    _wrap(TTScheduler, "update_from_output", "update")
    _wrap(TTModelRunner, "build_model_input", "build")
    _wrap(TTModelRunner, "execute_model", "runner", net=("build", "adapter"))
    _wrap(TTModelRunner, "sample_tokens", "sample")
    _wrap(Lfm3ForCausalLM, "decode_forward", "adapter")
    logger.info(f"LFM3: step profile on (a report every {every} decode steps)")
