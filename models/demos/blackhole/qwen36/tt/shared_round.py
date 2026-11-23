# SPDX-License-Identifier: Apache-2.0
"""Shared verify rounds for single-P150 multi-user serving (s27v7; sampled rounds s27v8).

A round is one traced step of W = 4, 8 or 16 rows over every running user: per user a group of consecutive rows =
its pending token then its drafts at consecutive positions (a user without drafts rides as one row). The row map
(`row_word`) gives every row its offset in its group, the group length and the group's slot (the user's decode row,
which owns its DRAM GDN state, see multiuser.py). Every module is the multi-user step's (multiuser.py: projections
over all rows, per-row attention through each row's page table, K/V written at each row's position), except GDN:

  FusedVerifyOutput(groups=S) / the grouped native front end: a group's first row starts from its slot's DRAM
  recurrent and convolution state and later rows chain from the previous row, exactly as the single-user verify;
  a single-row group writes its slot in place (bitwise the plain step), a longer group leaves one candidate per row;
  RoundAccept: per-row argmax (ttnn.argmax order), or in a sampled round the device sampler's per-row draws, and per
  group the accepted prefix a (drafts equal to the previous row's target), the commit table, the drafter tables;
  GroupCommit: the accepted row's candidates -> the group's slot.

Every row's logits are bitwise those of its user's sequential single-row steps whatever the other rows hold (the
multi-user step's row invariance plus the chained recurrence of the single-user verify), so a group emits its
accepted drafts and its row r0 + a target: the tokens of plain decoding. In a sampled round row r0 + j draws the
token at position p_{r0} + j + 1 from its request's processed distribution (penalty history = the outputs plus the
group's drafts 1..j) with the uniform of (seed, that position): a non-speculative step draws the same token from the
same distribution once drafts 1..j were emitted, so the accepted prefix plus the r0 + a draw are the spec-off tokens
(lossless for unseeded requests, identical for seeded ones). Rejected rows' K/V lie at positions the user's next
round rewrites before any query reads them (a query at position q reads keys <= q of its own pages)."""
from pathlib import Path

import torch
import ttnn

from models.demos.blackhole.qwen36.tt.attention.batched import USER_WIDTHS, release_groups, user_groups
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

_KERNEL = str(Path(__file__).with_name("shared_round_kernel.cpp").resolve())
_CORES = 110
MAP_ACTIVE = 1 << 31
DRAM = ttnn.DRAM_MEMORY_CONFIG


def row_word(offset, length, slot):
    """Row-map word of a round row (gdn/verify_recurrence_reader.cpp, native_frontend_kernel.cpp, shared_round_kernel.cpp)."""
    return MAP_ACTIVE | (slot << 16) | (length << 8) | offset


def _device(device, shape, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, value=0):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.int32
    return ttnn.from_torch(torch.full(shape, value, dtype=torch_dtype), dtype=dtype, layout=layout, device=device,
                           memory_config=DRAM)


class RoundAccept:
    """shared_round_kernel.cpp: row tokens (argmax of the round's logits, or the device sampler's draws) and the
    per-group acceptance bookkeeping of one round width."""

    def __init__(self, device, rows, vocab, slots, rings=0):
        if not 1 <= rows <= 16 or vocab % 32:
            raise ValueError("round acceptance supports 1..16 rows and a tile-multiple vocabulary")
        grid = device.compute_with_storage_grid_size()
        self.device, self.rows, self.vocab, self.slots, self.rings = device, rows, vocab, slots, rings
        self.coordinates = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(_CORES)]
        self.workers = 2 * _CORES
        self.partial = _device(device, (self.workers, 32), ttnn.uint32)

    def _launch(self, tensors, addresses, mode, coordinates, ncrisc, defines=()):
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in coordinates})
        tasks = [(core, [i]) for i, core in enumerate(coordinates)]
        kernel = UnifiedKernelDescriptor(
            kernel_source=_KERNEL, core_ranges=cores,
            defines=[("SR_ROWS", str(self.rows)), ("SR_MODE", str(mode)), ("SR_WORKERS", str(self.workers)),
                     ("SR_VOCAB_TILES", str(self.vocab // 32)), ("SR_SLOTS", str(self.slots)),
                     ("SR_RINGS", str(self.rings)), *defines],
            brisc_common_runtime_args=addresses,
            ncrisc_common_runtime_args=addresses if ncrisc else [],
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=tasks, **({"ncrisc_args": tasks} if ncrisc else {})),
        )
        sizes = ((0, 4096), (1, 4096)) if mode == 0 else ((0, 32768),)
        cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(
            index, ttnn.bfloat16, 2048, ttnn.TileDescriptor(ttnn.Tile((32, 32))))]) for index, size in sizes]
        program = ttnn.ProgramDescriptor(kernels=kernel.get_kernel_descriptors().kernels, cbs=cbs, semaphores=[])
        ttnn.generic_op(list(tensors), program)

    def __call__(self, logits, tokens, positions, row_map, result, commit, ring, drafter=None, sampler=None):
        """drafter: (per-slot anchor scalars, per-slot position scalars, ring owners) of the slots' draft contexts;
        sampler: the DeviceSampler whose out[16,4] holds this round's draws (sampled round), else argmax."""
        if sampler is None:
            shape = tuple(int(x) for x in logits.padded_shape)
            if (shape[-1] != self.vocab or shape[-2] != 32 or int(logits.shape[-2]) != self.rows
                    or logits.layout != ttnn.TILE_LAYOUT or logits.dtype != ttnn.bfloat16
                    or logits.memory_config() != DRAM):
                raise ValueError(f"round acceptance requires DRAM tiled BF16 [.., {self.rows}, {self.vocab}] logits")
            self._launch([logits, self.partial], [int(logits.buffer_address()), int(self.partial.buffer_address())],
                         0, self.coordinates, True)
            source, extra, defines = self.partial, [], ()
        else:
            source, defines = sampler.out, (("SR_SAMPLED", "1"),)
            extra = [sampler.params, *sampler.hist]
        tensors = [source, tokens, positions, row_map, result, commit, ring]
        addresses = [int(t.buffer_address()) for t in tensors]
        if drafter is not None:
            anchors, anchor_positions, owners = drafter
            if len(anchors) != self.slots or len(anchor_positions) != self.slots or len(owners) != self.rings:
                raise ValueError("round drafter tables must cover every slot and ring")
            scalars = [*anchors, *anchor_positions, *owners]
            tensors += scalars
            addresses += [int(t.buffer_address()) for t in scalars]
        else:
            if self.rings:
                raise ValueError("ring owners require the drafter tables")
            addresses += [0] * (2 * self.slots)
        tensors += extra
        addresses += [int(t.buffer_address()) for t in extra]
        self._launch(tensors, addresses, 1, self.coordinates[:1], False, defines)

    def release(self):
        ttnn.deallocate(self.partial)


class SharedRounds:
    """Fixed-address inputs, candidates, bookkeeping outputs and the captured round traces of a MultiUserDecoder.

    capacity: candidate rows (rows of multi-row groups, which the host places first; single-row groups update their
    slot in place). drafter: a dflash_slots.SlotDrafter whose rings receive every accepted row's target features
    (taps of layers 5, 19, 33, 47, 61) and whose per-slot anchors the round writes. sampler: the model's
    device_sampling.DeviceSampler (one history slot per decode slot); with it every width also has a sampled round
    (step(width, sampled=True)): the sampler draws each row's token from the request's processed distribution with
    the Philox uniform of (seed, the token's position), exactly as a non-speculative decode step draws it, so a
    group accepts its drafts while they equal those draws and a seeded request emits its spec-off tokens."""

    FEATURE_LAYERS = (5, 19, 33, 47, 61)

    def __init__(self, model, decoder, widths=None, capacity=8, drafter=None, sampler=None):
        from models.demos.blackhole.qwen36.tt.gdn.group_commit import GroupCommit

        self.model, self.decoder, self.device, self.drafter = model, decoder, model.device, drafter
        if sampler is not None and sampler.num_slots < decoder.slots:
            raise ValueError("sampled rounds need one sampler history slot per decode slot")
        self.sampler = sampler
        self.widths = tuple(widths or decoder.widths)
        if any(w not in USER_WIDTHS or w > 16 for w in self.widths):
            raise ValueError(f"shared round widths must be in {USER_WIDTHS} and at most 16")
        self.slots = decoder.slots
        self.capacity = int(capacity)
        if not 2 <= self.capacity <= 16:
            raise ValueError("shared round candidate capacity must be 2..16 rows")
        feature_layers = self.FEATURE_LAYERS if drafter is not None else ()
        dev = self.device
        gdn = decoder.gdn_layers
        # Candidates shared by every width: row t of [C,48,128,128] / rows 3t..3t+2 of [1,3C,10240].
        self.candidate_states = [ttnn.empty([self.capacity, 48, 128, 128], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                            device=dev, memory_config=DRAM) for _ in gdn]
        self.candidate_convs = [_device(dev, (1, 3 * self.capacity, 10240), ttnn.bfloat16) for _ in gdn]
        self.row_map = _device(dev, (1, 32), ttnn.uint32)
        self.result = _device(dev, (1, 64), ttnn.int32)
        self.commit = _device(dev, (1, 32), ttnn.uint32)
        self.ring = _device(dev, (1, 32), ttnn.uint32)
        self.inputs = {}
        self.accept = {}
        self.features = {}
        for width in self.widths:
            self.inputs[width] = (
                _device(dev, (1, width), ttnn.uint32),
                _device(dev, (width,), ttnn.int32),
                _device(dev, (width,), ttnn.int32),
                _device(dev, (width, decoder.page_table_width), ttnn.int32),
            )
            self.accept[width] = RoundAccept(dev, width, model.args.vocab_size, self.slots,
                                             drafter.rings if drafter is not None else 0)
            self.features[width] = {index: _device(dev, (1, width, model.args.dim), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                                    for index in feature_layers}
        self.group_commit = GroupCommit(self.commit, [
            (self.candidate_states[i], self.candidate_convs[i], decoder.recurrent[i], decoder.conv[i])
            for i in range(len(gdn))
        ])
        self.traces = {}
        self.logits = {}
        self.compiled = set()
        self._eager_logits = None

    # ---- device step (trace-safe) ----
    def step(self, width, sampled=False):
        model, decoder = self.model, self.decoder
        tokens, positions, rope, table = self.inputs[width]
        idx = ttnn.reshape(ttnn.typecast(rope, ttnn.uint32), [1, width])
        cos = ttnn.embedding(idx, decoder.cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idx, decoder.sin_table, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(idx)
        groups = user_groups(positions, table)
        features = self.features[width]
        x = model.embd(tokens)
        gdn_index = 0
        for index, layer in enumerate(model.layers):
            previous = x
            users = None
            if not layer.is_full_attention:
                users = ("groups", (decoder.recurrent[gdn_index], decoder.conv[gdn_index],
                                    self.candidate_states[gdn_index], self.candidate_convs[gdn_index], self.row_map))
                gdn_index += 1
            x = layer.users_forward(x, cos, sin, positions, table, groups, users,
                                    final_norm=model.verify_final_norm(index))
            ttnn.deallocate(previous)
            if index in features:
                ttnn.copy(x, features[index])
        release_groups(groups)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        hidden = model.verify_hidden(x)
        if hidden is not x:
            ttnn.deallocate(x)
        logits = model._lm_head(hidden)
        ttnn.deallocate(hidden)
        drafter = self.drafter
        sampler = self.sampler if sampled else None
        if sampled:
            if sampler is None:
                raise ValueError("sampled rounds need the device sampler")
            # Parameter row r of the round row r (shared_serving.py): a group's row j counts the group's drafts 1..j
            # (round tokens r0 + 1 ..) as outputs and does not commit; the round kernel commits the emitted tokens.
            sampler.scan(logits, width, aux=tokens, extra_word=0)
            sampler.rows(width, 0, pos=positions, pos_mode=2, pos_add=1)
        self.accept[width](logits, tokens, positions, self.row_map, self.result, self.commit, self.ring,
                           drafter.round_tables() if drafter is not None else None, sampler=sampler)
        self.group_commit()
        if drafter is not None:
            taps = ttnn.concat([features[index] for index in self.FEATURE_LAYERS], dim=-1, memory_config=DRAM)
            drafter.commit(ttnn.reshape(taps, [1, 1, width, len(self.FEATURE_LAYERS) * model.args.dim]), positions,
                           self.ring)
            ttnn.deallocate(taps)
        return logits

    # ---- host side ----
    def stage(self, width, tokens, positions, rope_positions, page_table, row_map):
        """Copy one round's host inputs into the width's fixed device inputs (every input, every round)."""
        tokens_dev, positions_dev, rope_dev, table_dev = self.inputs[width]
        table = torch.as_tensor(page_table).to(torch.int32)
        if tuple(table.shape) != (width, self.decoder.page_table_width):
            raise ValueError("round page table must be [width, page_table_width]")
        words = torch.zeros(1, 32, dtype=torch.int64)
        words[0, :len(row_map)] = torch.as_tensor(row_map, dtype=torch.int64)
        for row, word in enumerate(int(w) for w in row_map):
            if word & MAP_ACTIVE and (word >> 8) & 0xFF > 1 and row >= self.capacity:
                raise ValueError(f"multi-row groups must occupy the first {self.capacity} round rows")
        for host, device, dtype in (
            (torch.as_tensor(tokens).reshape(1, width).to(torch.int32), tokens_dev, ttnn.uint32),
            (torch.as_tensor(positions).reshape(width).to(torch.int32), positions_dev, ttnn.int32),
            (torch.as_tensor(rope_positions).reshape(width).to(torch.int32), rope_dev, ttnn.int32),
            (table, table_dev, ttnn.int32),
            (words.to(torch.int32), self.row_map, ttnn.uint32),
        ):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT), device)

    def warmup(self, width, capture, sampled=False):
        """Compile (eager step) and optionally capture the width's round trace (greedy or sampled)."""
        key = (width, bool(sampled))
        if key not in self.compiled:
            logits = self.step(width, sampled)
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(logits)
            self.compiled.add(key)
        if capture and key not in self.traces:
            trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            self.logits[key] = self.step(width, sampled)
            ttnn.end_trace_capture(self.device, trace, cq_id=0)
            self.traces[key] = trace

    def launch(self, width, sampled=False):
        """Run one round; returns its logits (the trace's fixed output; eager: kept until the next eager round),
        which a sampled round's host redraw reads."""
        key = (width, bool(sampled))
        if key in self.traces:
            ttnn.execute_trace(self.device, self.traces[key], cq_id=0, blocking=False)
            return self.logits[key]
        if self._eager_logits is not None:
            ttnn.deallocate(self._eager_logits)
        self._eager_logits = self.step(width, sampled)
        return self._eager_logits

    def read_result(self):
        return [int(v) for v in ttnn.to_torch(self.result).reshape(-1).tolist()]

    def release(self):
        for trace in self.traces.values():
            ttnn.release_trace(self.device, trace)
        self.traces.clear()
        if self._eager_logits is not None:
            ttnn.deallocate(self._eager_logits)
            self._eager_logits = None
        for accept in self.accept.values():
            accept.release()
        for tensor in (*self.candidate_states, *self.candidate_convs, self.row_map, self.result, self.commit, self.ring,
                       *(t for inputs in self.inputs.values() for t in inputs),
                       *(t for f in self.features.values() for t in f.values())):
            ttnn.deallocate(tensor)
