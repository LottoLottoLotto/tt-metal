# SPDX-License-Identifier: Apache-2.0
"""Serving side of shared verify rounds (s27v7; sampled rounds s27v8): round layout, draft planning, per-slot
bookkeeping.

One decode step of a multi-user server with DFlash2 is one shared round (shared_round.SharedRounds): every running
request rides as its pending token plus the drafts the scheduler carried from the previous step (none: one row).
After the round the batched drafter (dflash_slots.SlotDrafter) runs the draft passes the planner chose (per quad of
slots, over its first 1, 2 or 4 lanes), and each request gets its drafts for the next round, cut to its planned depth.

RoundPlanner prices a round with the measured device cost table (BASE_MS per width, KV read ms per row and 1K
positions, DRAFT_MS per pass by lanes) and each user's acceptance by depth (decayed counts backed off to a prior):
q(u, i) = P(drafts 1..i of u accepted), Q_u = sum_i q(u, i). A plan maximizes sum_u log(1 + Q_u) - N log(cost)
(proportional fairness across users) over every width and greedy draft prefix; it must beat the plain round's
-N log(cost) by 3% per user, otherwise the round is plain (every user one row, no drafter pass). A user whose
drafts keep failing gets a one-draft probe every PROBE rounds."""
import math

import torch
import ttnn

from .device_sampling import RowSampling, host_sample
from .dflash_slots import LANES, PARAM_ROW, QUAD, pass_lanes
from .shared_round import row_word

# Device ms (traced, P150; measured planner costs, 8 slots, BFP8 KV): the shared
# round at 4 / 8 / 16 rows (one group of up to 8 rows + single rows; verify bookkeeping, GDN group commit and the
# drafter ring commit included) less its KV term, the KV read growth per row and 1K positions, and one batched
# drafter pass of 1 / 2 / 4 lanes.
BASE_MS = {4: 54.66, 8: 60.16, 16: 79.12}
KV_MS_PER_ROW_1K = 0.093
ROUND_EXTRA_MS = 0.0       # inside BASE_MS (measured rounds, not the plain step)
DRAFT_MS = {1: 9.72, 2: 11.33, 4: 14.58}
MARGIN = 1.03
PRIOR = 0.75               # conditional acceptance of draft i given 1..i-1 accepted, before any data
PRIOR_WEIGHT = 2.0
DECAY = 0.9                # per observation of a draft position
PROBE = 16
MAX_DEPTH = 7
# Parameter rows of greedy rows that carry no request (benches): greedy, no history slot.
GREEDY = RowSampling(temperature=0.0, top_k=1, top_p=1.0, min_p=0.0, presence=0.0, frequency=0.0, repetition=1.0,
                     seed=0)
NO_HISTORY = 0xFF


def _plain(request):
    """A row the greedy round serves: no request (bench) or greedy without penalties."""
    return request is None or (request.spec.greedy and not request.spec.penalized)


class UserStats:
    def __init__(self):
        self.offered = [0.0] * (MAX_DEPTH + 1)
        self.accepted = [0.0] * (MAX_DEPTH + 1)
        self.rounds_without = 0

    def update(self, depth, accepted):
        """A round offered `depth` drafts and accepted the first `accepted`. Draft i was tested only if drafts
        1..i-1 were accepted (i <= accepted + 1): later drafts carry no information about the conditional
        acceptance P(i | 1..i-1), and a position's average moves only when it is tested."""
        for i in range(1, min(depth, accepted + 1) + 1):
            self.offered[i] = self.offered[i] * DECAY + 1.0
            self.accepted[i] = self.accepted[i] * DECAY + (1.0 if i <= accepted else 0.0)

    def chain(self):
        """q[i] = P(drafts 1..i accepted), i = 1..MAX_DEPTH."""
        q, p = [0.0], 1.0
        for i in range(1, MAX_DEPTH + 1):
            p *= (self.accepted[i] + PRIOR * PRIOR_WEIGHT) / (self.offered[i] + PRIOR_WEIGHT)
            q.append(p)
        return q


def round_cost(rows, kv_positions, passes, widths=None):
    """Device ms of a round of `rows` rows (the smallest of `widths` holding them) whose rows read KV up to
    kv_positions (one entry per row), plus the drafter passes `passes` ({quad: lanes})."""
    width = next((w for w in sorted(widths or BASE_MS) if rows <= w), None)
    if width is None:
        return math.inf
    return (BASE_MS[width] + ROUND_EXTRA_MS + KV_MS_PER_ROW_1K * sum(kv_positions) / 1000.0
            + sum(DRAFT_MS[lanes] for lanes in passes.values()))


class RoundPlanner:
    def __init__(self, capacity, rings, widths=None):
        self.capacity, self.rings = capacity, rings
        self.widths = tuple(sorted(widths or BASE_MS))
        if any(w not in BASE_MS for w in self.widths):
            raise ValueError(f"round widths {self.widths} need costs in BASE_MS")
        self.stats = {}

    def forget(self, key):
        self.stats.pop(key, None)

    def observe(self, key, depth, accepted):
        stats = self.stats.setdefault(key, UserStats())
        stats.update(depth, accepted)
        stats.rounds_without = 0 if depth else stats.rounds_without + 1

    def plan(self, users):
        """users: (key, slot, position, max_depth) per running request (max_depth 0: may not draft).
        Returns ({key: depth}, {quad: lanes} drafter passes)."""
        n = len(users)
        if n == 0:
            return {}, {}
        positions = [p for _, _, p, _ in users]
        plain = round_cost(n, positions, {}, self.widths)
        # A drafting plan must beat the plain round by MARGIN per user: plain scores as if MARGIN times cheaper.
        best = ({k: 0 for k, _, _, _ in users}, {}, -n * math.log(plain / MARGIN))
        chains = {k: self.stats.setdefault(k, UserStats()).chain() for k, _, _, _ in users}
        # Greedy: each step adds the draft row (some user's next draft) with the best resulting score; every prefix
        # of every width's sequence is priced and the best kept.
        for width in self.widths:
            if width < n:
                continue
            budget = min(width - n, self.capacity)
            depth = {k: 0 for k, _, _, _ in users}
            value = {k: 1.0 for k, _, _, _ in users}
            passes = {}
            kv = list(positions)
            multi_rows = 0
            while len(kv) - n < budget:
                base = sum(math.log(v) for v in value.values())
                choice = None
                for key, slot, position, cap in users:
                    d = depth[key]
                    if d >= min(cap, MAX_DEPTH) or slot >= self.rings:
                        continue
                    if multi_rows + (1 if d else 2) > self.capacity:  # a first draft also makes its row a candidate
                        continue
                    trial = dict(passes)
                    trial[slot // QUAD] = max(trial.get(slot // QUAD, 0), pass_lanes(slot % QUAD))
                    gained = value[key] + chains[key][d + 1]
                    # a draft row reads its user's KV up to its own position
                    cost = round_cost(len(kv) + 1, kv + [position + d + 1], trial, self.widths)
                    score = base - math.log(value[key]) + math.log(gained) - n * math.log(cost)
                    if choice is None or score > choice[1]:
                        choice = (key, score, trial, position + d + 1)
                if choice is None:
                    break
                key, score, passes, kv_position = choice
                multi_rows += 1 if depth[key] else 2
                value[key] += chains[key][depth[key] + 1]
                depth[key] += 1
                kv.append(kv_position)
                if score > best[2]:
                    best = (dict(depth), dict(passes), score)
        depths, passes, _ = best
        # Probes: a drafting-capable user that has not drafted for PROBE rounds tries one draft when a planned pass
        # covers its lane and the probe fits the plan's width and the candidate rows.
        if passes:
            rows = n + sum(depths.values())
            width = next(w for w in self.widths if rows <= w)
            multi_rows = sum(d + 1 for d in depths.values() if d)
            for key, slot, _, cap in users:
                stats = self.stats[key]
                if (depths[key] == 0 and cap > 0 and slot < self.rings and stats.rounds_without >= PROBE
                        and slot % QUAD < passes.get(slot // QUAD, 0) and rows + 1 <= width
                        and multi_rows + 2 <= self.capacity):
                    depths[key] = 1
                    rows += 1
                    multi_rows += 2
        return depths, passes


def layout(groups, width, scratch_row):
    """Round rows for groups = [(slot, tokens, first position, rope offset, page-table row)] (tokens: the pending
    token then the drafts). Multi-row groups first (candidate rows), then single rows, then inactive padding.
    Returns (order, tokens, positions, rope positions, page table rows, row map words)."""
    order = sorted(range(len(groups)), key=lambda i: len(groups[i][1]) == 1)
    tokens, positions, rope, table, words = [], [], [], [], []
    for i in order:
        slot, toks, first, delta, row = groups[i]
        length = len(toks)
        tokens += [int(t) for t in toks]
        positions += [first + j for j in range(length)]
        rope += [first + j + delta for j in range(length)]
        table += [row] * length
        words += [row_word(j, length, slot) for j in range(length)]
    pad = width - len(tokens)
    tokens += [0] * pad
    positions += [0] * pad
    rope += [0] * pad
    table += [scratch_row] * pad
    words += [0] * pad
    return order, tokens, positions, rope, torch.stack(table), words



class SharedServing:
    """Shared rounds of a MultiUserDecoder with per-slot drafter contexts (one per model, single P150)."""

    def __init__(self, model, decoder, rounds, drafter, depth=MAX_DEPTH):
        self.model, self.decoder, self.rounds, self.drafter = model, decoder, rounds, drafter
        self.depth = depth
        self.planner = RoundPlanner(rounds.capacity, drafter.rings if drafter is not None else 0, widths=rounds.widths)
        self.last_width = None
        self.force = None  # (depth, lanes): forced drafting for drafter checks, see forced()
        self.scratch_row = torch.full((decoder.page_table_width,), model._kv_scratch_block, dtype=torch.int32)
        self.owner = [None] * decoder.slots
        self.stats = {"rounds": 0, "sampled_rounds": 0, "rows": 0, "emitted": 0, "drafted_rounds": 0, "passes": 0,
                      **{f"passes_x{lanes}": 0 for lanes in LANES}}

    # ---- slot lifecycle ----
    def load_slot(self, slot, key):
        """A request's prefill finished into `slot` (MultiUserDecoder.load_slot ran): its drafter context is the
        prefill ring; its first round rides without drafts. A slot's previous request (if any) is forgotten."""
        if self.drafter is not None:
            if slot < self.drafter.rings:
                self.drafter.contexts[slot].copy_from(self.drafter.drafter.cache)
        if self.owner[slot] is not None and self.owner[slot] != key:
            self.planner.forget(self.owner[slot])
        for other, owner in enumerate(self.owner):
            if owner == key and other != slot:  # a re-prefilled (preempted / resumed) request left its old slot
                self.owner[other] = None
        self.owner[slot] = key

    def release(self, key):
        self.planner.forget(key)
        for slot, owner in enumerate(self.owner):
            if owner == key:
                self.owner[slot] = None

    def remap(self, slot_remap):
        """vLLM condensing: slot i takes over slot_remap[i]'s request (MultiUserDecoder.remap semantics)."""
        remap = [int(s) for s in torch.as_tensor(slot_remap).reshape(-1).tolist()][: self.decoder.slots]
        moves = [(src, dst) for dst, src in enumerate(remap) if 0 <= src < self.decoder.slots and src != dst]
        if not moves:
            return
        self.decoder.remap(slot_remap)
        if self.drafter is not None:
            self.drafter.remap(moves)
        owners = list(self.owner)
        destinations = {dst for _, dst in moves}
        for src, dst in moves:
            self.owner[dst] = owners[src]
        # A vacated source holds no request (a later load_slot there must not forget a moved request's stats).
        for src, _ in moves:
            if src not in destinations:
                self.owner[src] = None

    # ---- one round ----
    def fit(self, rows):
        """Cut drafts (the longest group's last draft first) until the round fits the widest round and the
        candidate rows. Drafts were planned for the previous round's requests: requests that joined since then
        ride as one row each and can push the plan past the limits."""
        drafts = [list(row[3]) for row in rows]
        limit, capacity = self.rounds.widths[-1], self.rounds.capacity
        if len(rows) > limit:
            raise RuntimeError(f"{len(rows)} requests exceed the {limit}-row round")
        while (sum(1 + len(d) for d in drafts) > limit
               or sum(1 + len(d) for d in drafts if d) > capacity):
            longest = max(range(len(drafts)), key=lambda i: len(drafts[i]))
            drafts[longest].pop()
        return [(*row[:3], d, *row[4:]) for row, d in zip(rows, drafts)]

    def forced(self, users, depth, lanes=None):
        """A plan without the planner (drafter checks): every drafting-capable user drafts `depth` (in slot order,
        while the round width and candidate rows allow), passes cover the drafting users (at least `lanes`)."""
        depths, passes = {k: 0 for k, _, _, _ in users}, {}
        rows, multi = len(users), 0
        for key, slot, _, cap in sorted(users, key=lambda u: u[1]):
            d = min(depth, cap, MAX_DEPTH)
            if d <= 0 or slot >= (self.drafter.rings if self.drafter is not None else 0):
                continue
            d = min(d, self.rounds.widths[-1] - rows, self.rounds.capacity - multi - 1)
            if d <= 0:
                continue
            depths[key] = d
            rows += d
            multi += d + 1
            quad = slot // QUAD
            passes[quad] = max(passes.get(quad, 0), pass_lanes(slot % QUAD), lanes or 0)
        return depths, passes

    def plan(self, rows):
        """Plan made before a round of `rows` (round()'s rows, slots after any remap): the next round's draft depth
        per request and the drafter passes after this round, ({key: depth}, {quad: lanes}). With no drafts in
        `rows` and no passes the step can be the plain multi-user step instead (which commits no drafter context)."""
        users = [(key, slot, int(position) + 1 + len(drafts), cap if self.drafter is not None else 0)
                 for key, slot, _, drafts, position, _, cap in self.fit(rows)]
        if self.force is not None and self.drafter is not None:
            return self.forced(users, *self.force)
        return self.planner.plan(users)

    def _stage_sampling(self, order, rows, sampling):
        """Sampler parameter rows of a sampled round: round row r0 + j of a request's group draws its token with the
        request's spec, counting the group's drafts 1..j (round tokens r0 + 1 ..) as outputs and committing nothing
        (the round kernel commits the emitted tokens); row PARAM_ROW + slot is the request's drafting lane. Every
        request's history slot (= its decode slot) holds its outputs first, as for a non-speculative sampled step."""
        sampler = self.rounds.sampler
        params = [None] * 32
        r = 0
        for i in order:
            _, slot, _, drafts, _, _, _ = rows[i]
            request = sampling[i]
            if request is None:
                spec, history = GREEDY, NO_HISTORY
            else:
                spec, history = request.spec, slot
                sampler.ensure_history(slot, request.owner, request.outputs, request.prompt or (), spec.uses_prompt)
                if not _plain(request):
                    params[PARAM_ROW + slot] = spec.words(slot)
            for j in range(len(drafts) + 1):
                params[r + j] = spec.words(history, n_extra=j, nocommit=True, extra_base=r + 1)
            r += len(drafts) + 1
        sampler.set_params(params)

    def _host_redraw(self, logits, row, request, accepted, position):
        """The token of round row `row` drawn on the host from the round's retained logits (exact sampler path)."""
        values = ttnn.to_torch(logits).reshape(-1, self.model.args.vocab_size)[row]
        spec = request.spec if request is not None else GREEDY
        outputs = list(request.outputs) + list(accepted) if request is not None else list(accepted)
        prompt = (request.prompt or ()) if request is not None else ()
        return host_sample(values, spec, outputs, prompt, position)

    def round(self, rows, slot_remap=None, plan=None, sampling=None):
        """rows: per scheduled request (key, slot, pending token, drafts, position, page-table row, next_cap) where
        next_cap bounds its drafts for the next round (0: none). Returns (emitted tokens, next drafts) per row.
        plan: plan(rows) made before this call (slots after slot_remap); None plans while the round runs.
        With self.force = (depth, lanes) the next round's drafts come from forced() instead of the planner.
        sampling: per row a device_sampling.SamplingRequest (its spec, history owner, outputs and prompt) or None
        (greedy, unpenalized). A round with a random or penalized request is a sampled round (SharedRounds
        step(sampled=True)): every row's token is drawn by the device sampler, and drafting passes over such a
        request's lane draw its drafts (SlotDrafter sampled passes). A row whose top-k tie group the device cannot
        see whole is redrawn here from the round's logits (vLLM's float32 sampler, same uniform)."""
        if slot_remap is not None:
            self.remap(slot_remap)
        rows = self.fit(rows)
        decoder, rounds, drafter = self.decoder, self.rounds, self.drafter
        groups = []
        for key, slot, token, drafts, position, table, _ in rows:
            if self.owner[slot] != key:
                raise RuntimeError(f"slot {slot} does not hold request {key}")
            row = torch.as_tensor(table, dtype=torch.int32).reshape(-1)[: decoder.page_table_width]
            if row.numel() < decoder.page_table_width:
                row = torch.cat([row, torch.zeros(decoder.page_table_width - row.numel(), dtype=torch.int32)])
            groups.append((slot, [int(token), *[int(t) for t in drafts]], int(position),
                           int(decoder.rope_deltas[slot]), row))
        total = sum(len(g[1]) for g in groups)
        width = next(w for w in rounds.widths if total <= w)
        self.last_width = width
        order, tokens, positions, rope, table, words = layout(groups, width, self.scratch_row)
        rounds.stage(width, tokens, positions, rope, table, words)
        sampled = sampling is not None and any(not _plain(s) for s in sampling)
        if sampled:
            self._stage_sampling(order, rows, sampling)
        logits = rounds.launch(width, sampled)
        # Without a plan, plan the next round while this one runs: it decides which draft passes follow this round.
        depths, passes = plan if plan is not None else self.plan(rows)
        self.last_plan = (depths, passes)
        for quad, lanes in sorted(passes.items()):
            drafter.launch(quad, lanes, sampled=sampled and drafter.samples and any(
                not _plain(request) and row[1] // QUAD == quad and row[1] % QUAD < lanes
                for row, request in zip(rows, sampling)))
        result = rounds.read_result()
        proposals = {quad: drafter.read(quad, lanes) for quad, lanes in sorted(passes.items())}
        emitted, next_drafts = [None] * len(rows), [[] for _ in rows]
        r = 0
        for i in order:
            key, slot, _, drafts, _, _, _ = rows[i]
            count = result[16 + r]
            emitted[i] = result[r:r + count]
            self.planner.observe(key, len(drafts), count - 1)
            request = sampling[i] if sampled else None
            redrawn = bool(result[32 + r])
            if redrawn:
                # The device stopped the group at a row whose top-k tie group continues past its candidates: that
                # token is drawn here (its history was not committed: the next ensure_history re-syncs it).
                emitted[i][-1] = self._host_redraw(logits, r + count - 1, request, emitted[i][:-1],
                                                   positions[r + count - 1] + 1)
                rounds.sampler.invalidate(slot)
            elif request is not None:
                rounds.sampler.note_committed(slot, request.owner, len(request.outputs) + count)
            lanes = proposals.get(slot // QUAD)
            # A redrawn token was not the drafter's anchor: those drafts were proposed after another token.
            if lanes is not None and slot % QUAD < len(lanes) and depths.get(key, 0) and not redrawn:
                next_drafts[i] = lanes[slot % QUAD][: depths[key]]
            r += len(drafts) + 1
        self.stats["rounds"] += 1
        self.stats["sampled_rounds"] += sampled
        self.stats["rows"] += total
        self.stats["emitted"] += sum(len(e) for e in emitted)
        self.stats["drafted_rounds"] += bool(passes)
        self.stats["passes"] += len(passes)
        for lanes in passes.values():
            self.stats[f"passes_x{lanes}"] += 1
        return emitted, next_drafts