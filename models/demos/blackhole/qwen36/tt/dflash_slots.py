# SPDX-License-Identifier: Apache-2.0
"""Per-user DFlash2 draft contexts and batched drafting for shared verify rounds (s27v7; sampled passes s27v8).

Ring slots 0..R-1 (the multi-user decode slots that may draft) each own a committed 2048-row context
(dflash_cache.CommittedKV with a 32-row draft mask). A round commits every accepted row's target features to its
user's ring (`commit`: one fc / norm / K-V projection pass over the round's rows, then kernel mode 7 scatters the
rows; the round kernel updates the ring owners and writes each slot's next anchor token and position). A draft pass
(`launch(quad, lanes)`) runs the five-layer backbone once over one 32-row tile holding up to four users' blocks
(anchor + 7 mask tokens each: the weights stream once), the first `lanes` (1, 2 or 4) users of the quad attending
their own rings through lane masks, then the draft head and `lanes` greedy selector chains over those users' rows
only (fewer lanes: fewer SDPAs, a shorter vocabulary scan and selector walk). A prefilled request's context moves
from the single-sequence prefill ring into its slot (`load_slot`); vLLM condensing moves rings (`remap`).

Drafts only propose: verification is exact whatever they are, so the batched pass need not reproduce the
single-user drafter's bits. A sampled pass (`launch(quad, lanes, sampled=True)`, quads holding a random or penalized
request) walks each lane with the device sampler instead of the greedy selector: the request in sampler parameter row
PARAM_ROW + slot draws its drafts from the candidates' distribution with the uniform its target rows will draw with
(dflash_ops.select_chains), so a draft is the token the round samples whenever the two distributions agree there."""
import torch
import ttnn

from .dflash_cache import DRAM, CommittedKV, commit_round
from .dflash_ops import retile_heads

BLOCK = 8
QUAD = 4
LANES = (1, 2, 4)
PARAM_ROW = 16  # sampler parameter row of slot s's lane: PARAM_ROW + s (rows 0..15 belong to the round's rows)
SAMPLE_MODES = ("point", "sampled", "processed")


def pass_lanes(lane):
    """Lanes of the smallest draft pass covering quad lane `lane` (0..3)."""
    return next(n for n in LANES if lane < n)


class SlotDrafter:
    def __init__(self, model, drafter, slots, rings, sampler=None, sample_mode="sampled"):
        """sampler / sample_mode: drafts of random or penalized requests ("point": the greedy walk; "sampled": drawn
        from softmax(scores / T) over the 16 candidates; "processed": after the request's top-k / top-p / min-p)."""
        if drafter.block_size != BLOCK:
            raise ValueError("batched drafting requires the block-8 drafter")
        if rings % QUAD or not 0 < rings <= slots:
            raise ValueError("ring slots must be a positive multiple of four, at most the decode slots")
        if sample_mode not in SAMPLE_MODES:
            raise ValueError(f"shared-round draft sampling mode must be one of {SAMPLE_MODES}")
        if sampler is not None and PARAM_ROW + slots > 32:
            raise ValueError("sampled drafting needs a sampler parameter row per slot")
        self.model, self.drafter, self.device = model, drafter, model.device
        self.slots, self.rings = slots, rings
        self.sampler = sampler if sample_mode != "point" else None
        self.sample_mode = sample_mode
        dev = self.device

        def scalar():
            return ttnn.from_torch(torch.zeros((1, 1), dtype=torch.int32), dtype=ttnn.int32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM)

        self.contexts = [CommittedKV(dev, BLOCK, mask_rows=QUAD * BLOCK) for _ in range(rings)]
        self.anchors = [scalar() for _ in range(slots)]
        self.positions = [scalar() for _ in range(slots)]
        self.zero = scalar()
        # Lane selectors: 1 on lane u's 8 query rows of [1,32,32,128], 0 elsewhere.
        self.lane_select = []
        for lane in range(QUAD):
            mask = torch.zeros(1, 32, QUAD * BLOCK, 128, dtype=torch.bfloat16)
            mask[:, :, lane * BLOCK:(lane + 1) * BLOCK, :] = 1
            self.lane_select.append(ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                                    memory_config=DRAM))
        self.out = {(quad, lanes): ttnn.from_torch(torch.zeros((1, lanes * BLOCK), dtype=torch.int32), dtype=ttnn.uint32,
                                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM)
                    for quad in range(rings // QUAD) for lanes in LANES}
        self.traces = {}
        self.compiled = set()

    @property
    def owners(self):
        return [context.owners for context in self.contexts]

    def round_tables(self):
        """(anchor scalars, anchor position scalars, ring owners) for RoundAccept."""
        return self.anchors, self.positions, self.owners

    # ---- context management (host-ordered, outside traces) ----
    def load_slot(self, slot, anchor, position):
        """A prefilled request takes `slot`: its prompt context (the prefill ring) and its next draft block."""
        if slot < self.rings:
            self.contexts[slot].copy_from(self.drafter.cache)
        self.set_anchor(slot, anchor, position)

    def set_anchor(self, slot, anchor, position):
        for value, tensor in ((anchor, self.anchors[slot]), (position, self.positions[slot])):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.tensor([[int(value)]], dtype=torch.int32), dtype=ttnn.int32,
                                layout=ttnn.ROW_MAJOR_LAYOUT), tensor)

    def reset_slot(self, slot):
        if slot < self.rings:
            self.contexts[slot].reset()

    def remap(self, moves):
        """moves: (source, destination) slot pairs (vLLM condensing); sources are read before any write."""
        moves = [(s, d) for s, d in moves if s != d]
        if not moves:
            return
        contexts, scalars = {}, {}
        for source, _ in moves:
            if source < self.rings and source not in contexts:
                context = self.contexts[source]
                contexts[source] = [ttnn.clone(context.owners),
                                    *(ttnn.clone(t) for pair in context.layers for t in pair)]
            scalars.setdefault(source, (ttnn.clone(self.anchors[source]), ttnn.clone(self.positions[source])))
        for source, destination in moves:
            anchor, position = scalars[source]
            ttnn.copy(anchor, self.anchors[destination])
            ttnn.copy(position, self.positions[destination])
            if destination < self.rings:
                context = self.contexts[destination]
                if source in contexts:
                    owners, *layers = contexts[source]
                    ttnn.copy(owners, context.owners)
                    for staged, live in zip(layers, (t for pair in context.layers for t in pair)):
                        ttnn.copy(staged, live)
                else:
                    context.reset()  # no context beyond the ring slots: it refills from the next commits
        for tensor in (*(t for ts in contexts.values() for t in ts), *(t for ts in scalars.values() for t in ts)):
            ttnn.deallocate(tensor)

    # ---- device steps (trace-safe) ----
    def commit(self, features, positions, ring_table):
        """Round rows' target features [1,1,W,25600] -> K/V of every drafter layer -> each accepted row's ring."""
        d = self.drafter
        rows = int(features.shape[-2])
        target_hidden = d._norm(d._linear(features, d.fc), d.hidden_norm)
        idx = ttnn.reshape(ttnn.typecast(positions, ttnn.uint32), [1, rows])
        tables = []
        for table in (d.cos_table, d.sin_table):
            embedded = ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
            tables.append(ttnn.reshape(embedded, (1, 1, rows, 128)))
        ttnn.deallocate(idx)
        for index, layer in enumerate(d.layers):
            key, value = d._context_kv(layer, target_hidden, tables)
            commit_round(self.contexts, index, key, value, ring_table, positions)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
        for tensor in (target_hidden, *tables):
            ttnn.deallocate(tensor)

    def _quad(self, quad, lanes, sampled=False):
        d, model = self.drafter, self.model
        first = quad * QUAD
        slots = list(range(first, first + lanes))
        contexts = [self.contexts[s] for s in slots]
        rows = QUAD * BLOCK  # the backbone always runs one full tile row; lanes >= `lanes` are inert padding
        for lane, (context, slot) in enumerate(zip(contexts, slots)):
            context.prepare(self.anchors[slot], self.positions[slot], self.positions[slot], self.zero, lane=lane)
        padded = contexts + [contexts[0]] * (QUAD - lanes)
        tokens = ttnn.concat([c.token_ids for c in padded], dim=1)
        hidden = d._hidden(ttnn.reshape(model.embd(tokens), [1, 1, rows, model.args.dim]), rows)
        noise = ttnn.concat([ttnn.slice(c.noise_positions, (0, 0), (1, BLOCK)) for c in padded], dim=1)
        tables = []
        for table in (d.cos_table, d.sin_table):
            embedded = ttnn.embedding(noise, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
            tables.append(ttnn.reshape(embedded, (1, 1, rows, 128)))
        for index, layer in enumerate(d.layers):
            residual = hidden
            normalized = d._norm(hidden, layer["input_norm"])
            branch, dynamic = layer["attention_conv"].prepare(normalized, block=BLOCK)
            q, k, v = d._heads(
                d._linear(branch, layer["q_proj"]),
                d._linear(branch, layer["k_proj"]),
                d._linear(branch, layer["v_proj"]),
            )
            q = d._rope(d._norm(q, layer["q_norm"]), tables)
            k = d._rope(d._norm(k, layer["k_norm"]), tables)
            attended = None
            for lane, context in enumerate(contexts):
                key, value = context.attention_inputs(index, k, v, rows=rows)
                out = ttnn.transformer.scaled_dot_product_attention(
                    q, key, value, attn_mask=context.mask, is_causal=False,
                    scale=128 ** -0.5, memory_config=DRAM, program_config=d.sdpa_config,
                    compute_kernel_config=d.compute,
                )
                if lanes == 1:
                    attended = out  # rows past lane 0 are inert padding (lane 0's mask keeps them finite)
                    continue
                part = ttnn.multiply(out, self.lane_select[lane], memory_config=DRAM)
                ttnn.deallocate(out)
                if attended is None:
                    attended = part
                else:
                    summed = ttnn.add(attended, part, memory_config=DRAM)
                    ttnn.deallocate(attended)
                    ttnn.deallocate(part)
                    attended = summed
            (attended,) = retile_heads(self.device, [(attended, (1, 1, rows, 4096))], rows)
            branch = d._linear(attended, layer["o_proj"])
            branch = layer["attention_conv"].finish(branch, dynamic)
            hidden = ttnn.add(residual, branch, memory_config=DRAM)
            residual = hidden
            normalized = d._norm(hidden, layer["post_norm"])
            branch, dynamic = layer["mlp_conv"].prepare(normalized, block=BLOCK)
            gate = ttnn.silu(d._linear(branch, layer["gate_proj"]), memory_config=DRAM)
            up = d._linear(branch, layer["up_proj"])
            branch = d._linear(ttnn.multiply(gate, up, memory_config=DRAM), layer["down_proj"])
            branch = layer["mlp_conv"].finish(branch, dynamic)
            hidden = ttnn.add(residual, branch, memory_config=DRAM)
        hidden = d._norm(hidden, d.final_norm)
        live = lanes * BLOCK
        if live < rows:
            hidden = ttnn.slice(hidden, (0, 0, 0, 0), (1, 1, live, model.args.dim), memory_config=DRAM)
        head = model.dflash_draft_head
        if head is not None:
            weight, program_config, _ = head
            logits = ttnn.linear(hidden, weight, program_config=program_config, memory_config=DRAM)
        else:
            logits = ttnn.reshape(model._lm_head(ttnn.reshape(hidden, [1, live, model.args.dim])),
                                  [1, 1, live, model.args.vocab_size])
        logits = d._hidden(logits, live, d.selector.logits_width)
        if sampled:
            path = d.selector.select_chains(hidden, logits, [self.anchors[s] for s in slots], BLOCK,
                                            sampler=self.sampler, positions=[self.positions[s] for s in slots],
                                            param_rows=[PARAM_ROW + s for s in slots],
                                            processed=self.sample_mode == "processed")
        else:
            path = d.selector.select_chains(hidden, logits, [self.anchors[s] for s in slots], BLOCK)
        ttnn.copy(path, self.out[(quad, lanes)])
        return self.out[(quad, lanes)]

    @property
    def samples(self):
        """True when random / penalized requests get sampled drafting passes."""
        return self.sampler is not None

    def warmup(self, quad, capture, lanes=QUAD, sampled=False):
        key = (quad, lanes, bool(sampled))
        if sampled and self.sampler is None:
            raise ValueError("sampled drafting passes need the device sampler")
        if key not in self.compiled:
            self._quad(quad, lanes, sampled)
            ttnn.synchronize_device(self.device)
            self.compiled.add(key)
        if capture and key not in self.traces:
            trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._quad(quad, lanes, sampled)
            ttnn.end_trace_capture(self.device, trace, cq_id=0)
            self.traces[key] = trace

    def launch(self, quad, lanes=QUAD, sampled=False):
        """Draft for slots quad*4 .. quad*4 + lanes - 1 (their anchors / positions as the last round left them);
        sampled: lanes of random / penalized requests draw (their sampler parameter rows must be set)."""
        key = (quad, lanes, bool(sampled))
        if key in self.traces:
            ttnn.execute_trace(self.device, self.traces[key], cq_id=0, blocking=False)
        else:
            self._quad(quad, lanes, sampled)
        return self.out[(quad, lanes)]

    def read(self, quad, lanes=QUAD):
        """Drafts per lane: 7 uint32 each (block rows 1..7)."""
        values = [int(v) for v in ttnn.to_torch(self.out[(quad, lanes)]).reshape(-1).tolist()]
        return [values[lane * BLOCK + 1:(lane + 1) * BLOCK] for lane in range(lanes)]

    def release(self):
        for trace in self.traces.values():
            ttnn.release_trace(self.device, trace)
        self.traces.clear()
        for context in self.contexts:
            context.release()
        for tensor in (*self.anchors, *self.positions, self.zero, *self.lane_select, *self.out.values()):
            ttnn.deallocate(tensor)
