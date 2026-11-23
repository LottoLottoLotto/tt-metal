# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recurrent / chunked DeltaNet forward (the `forward` dispatch).

Behavior-preserving extraction of the original `Qwen36GatedDeltaNet.forward` body.
Operates on the gdn instance: reads weights from `gdn.weights`, config dims from
`gdn.cfg`, mirrored scalar attrs + runtime state from `gdn`. Every ttnn op,
memory_config, and the `gated_deltanet_forward_ttnn` kwargs are verbatim.
"""
import ttnn
from models.demos.blackhole.qwen36.tt.gdn.native_frontend import native_frontend
from models.demos.blackhole.qwen36.tt.gdn.native_output import native_output
from models.demos.blackhole.qwen36.tt.gdn.state import init_recurrent_state, split_fused_conv_state
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import gated_deltanet_forward_ttnn


def ensure_native_frontend_many(gdn, T, fork_position=0):
    """This layer's verify_many front-end outputs (per-layer prefix histories, shared operands).

    Created on first use, or up front by TargetVerifier.__init__ so these long-lived per-layer
    buffers are placed before the warmup forwards instead of between their transient tensors
    (which would fragment DRAM for later trace captures)."""
    verify_key = (T, fork_position) if fork_position else T
    if verify_key not in gdn.native_frontend_output_many:
        from models.demos.blackhole.qwen36.tt.gdn.native_frontend import allocate_native_frontend_many
        key = (id(gdn.device), T)
        shared = type(gdn)._shared_native_frontend_many.get(key)
        frontend_output = allocate_native_frontend_many(gdn.device, T, shared)
        frontend_output.fork_position = fork_position
        if shared is None:
            type(gdn)._shared_native_frontend_many[key] = frontend_output
        gdn.native_frontend_output_many[verify_key] = frontend_output
    return gdn.native_frontend_output_many[verify_key]


def ensure_native_frontend_users(gdn, T):
    """This layer's multi-user front-end program (operands shared by every layer of width T)."""
    outputs = gdn.__dict__.setdefault("native_frontend_output_users", {})
    if T not in outputs:
        from models.demos.blackhole.qwen36.tt.gdn.native_frontend import allocate_native_frontend_batched
        key = (id(gdn.device), "users", T)
        shared = type(gdn)._shared_native_frontend_many.get(key)
        outputs[T] = allocate_native_frontend_batched(gdn.device, T, shared)
        if shared is None:
            type(gdn)._shared_native_frontend_many[key] = outputs[T]
    return outputs[T]


def ensure_verify_users(gdn, T):
    """This layer's multi-user recurrence + output program (verify inbox shared by every layer)."""
    runners = gdn.__dict__.setdefault("_fused_verify_output_users", {})
    if T not in runners:
        from models.demos.blackhole.qwen36.tt.gdn.fused_verify_output import FusedVerifyOutput
        key = id(gdn.device)
        shared = type(gdn)._shared_verify_many_inbox.get(key)
        if shared is None and hasattr(gdn, "_fused_verify_output"):
            shared = gdn._fused_verify_output.inbox
        runners[T] = FusedVerifyOutput(gdn.device, inbox=shared, batched=True)
        type(gdn)._shared_verify_many_inbox[key] = runners[T].inbox
    return runners[T]

def ensure_native_frontend_groups(gdn, T, slots, conv_rows):
    """This layer's shared-round front-end program over the layer's [1,3R,10240] prefix histories."""
    outputs = gdn.__dict__.setdefault("native_frontend_output_groups", {})
    if T not in outputs:
        from models.demos.blackhole.qwen36.tt.gdn.native_frontend import allocate_native_frontend_groups
        key = (id(gdn.device), "groups", T)
        shared = type(gdn)._shared_native_frontend_many.get(key)
        outputs[T] = allocate_native_frontend_groups(gdn.device, T, slots, shared, conv_rows=conv_rows)
        if shared is None:
            type(gdn)._shared_native_frontend_many[key] = outputs[T]
    return outputs[T]


def ensure_verify_groups(gdn, T, slots):
    """This layer's shared-round recurrence + output program (verify inbox shared by every layer)."""
    runners = gdn.__dict__.setdefault("_fused_verify_output_groups", {})
    if T not in runners:
        from models.demos.blackhole.qwen36.tt.gdn.fused_verify_output import FusedVerifyOutput
        key = id(gdn.device)
        shared = type(gdn)._shared_verify_many_inbox.get(key)
        if shared is None and hasattr(gdn, "_fused_verify_output"):
            shared = gdn._fused_verify_output.inbox
        runners[T] = FusedVerifyOutput(gdn.device, inbox=shared, groups=slots)
        type(gdn)._shared_verify_many_inbox[key] = runners[T].inbox
    return runners[T]




def recurrent_forward(gdn, x, mode="recurrent", chunk_size=None, valid_len=None, fork_position=0, users=None,
                      groups=None):
    """Non-kernel GDN forward. mode='chunk' (prefill, may delegate to the prefill kernel) or
    'recurrent' (single-token decode). Reads weights/state/dims off the gdn instance; updates
    gdn's recurrent + conv state in place or by reassignment per the trace-capture flags.

    valid_len: for fixed-bucket masked prefill — x is right-padded to T but only the first
    valid_len positions are real (see gated_deltanet_forward_ttnn). None = no padding.

    users: (recurrent_states, conv_states), one DRAM pair per row, with mode='verify_many':
    row t is user t, read from and written back to its own state (multi-user decode).

    groups: (slot recurrent states, slot conv states, candidate states [R,48,128,128], candidate histories
    [1,3R,10240], row map [1,32]) with mode='verify_many' (R >= T): a shared verify round (FusedVerifyOutput
    groups mode); returns (output, (), candidate histories)."""
    w = gdn.weights
    verifying = mode in ("verify_two", "verify_many")
    if chunk_size is None:
        chunk_size = gdn.long_prefill_chunk_size if mode == "chunk" else 64

    if gdn.recurrent_state is None:
        shape = x.shape
        batch_size = shape[0] if len(shape) == 3 else 1
        init_recurrent_state(gdn, batch_size)

    T = x.shape[1]
    if fork_position and (mode != "verify_many" or T != 16 - fork_position):
        raise ValueError("Tree recurrence requires its fixed verify_many topology")
    verify_key = (T, fork_position) if fork_position else T

    # After prefill, fuse separate conv states into one for efficient decode
    if T == 1 and gdn.fused_conv_state is None and gdn.conv_state_q is not None:
        gdn.fused_conv_state = ttnn.concat([gdn.conv_state_q, gdn.conv_state_k, gdn.conv_state_v], dim=2)
        gdn.fused_conv_state = ttnn.to_layout(gdn.fused_conv_state, ttnn.TILE_LAYOUT)
        split_fused_conv_state(gdn)

    # Chunk-parallel prefill via the C++ gated_delta_attn_seq kernel (float32, chunk_size=128).
    seq_masks = w.chunk_seq_masks_long
    persistent_recurrent_state = gdn.recurrent_state
    recurrent_input = persistent_recurrent_state
    recurrent_row_dram = None
    if mode == "chunk" and persistent_recurrent_state.layout == ttnn.ROW_MAJOR_LAYOUT:
        # The legacy prefill kernel still consumes tiles. This one-time
        # prefill-boundary conversion is not present on the decode path.
        recurrent_row_dram = ttnn.to_memory_config(
            persistent_recurrent_state,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        recurrent_input = ttnn.to_layout(
            recurrent_row_dram,
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    native_state_address = None
    native_conv_address = None
    if T == 1 and gdn.native_recurrent:
        assert gdn.recurrent_state is not None, "native GDN requested without recurrent state"
        assert gdn.recurrent_state.dtype == ttnn.bfloat16, "native GDN requires BF16 recurrent state"
        assert gdn.recurrent_state.memory_config().buffer_type == ttnn.BufferType.L1, (
            "native GDN requested but persistent recurrent state is not L1-resident"
        )
        if gdn.native_frontend:
            assert w.A_neg is not None, "native GDN front end requested without precomputed decay scale"
        native_state_address = int(persistent_recurrent_state.buffer_address())
        assert gdn.fused_conv_state is not None, "native GDN requested without fused convolution state"
        assert gdn.fused_conv_state.memory_config().buffer_type == ttnn.BufferType.L1, (
            "native GDN requested but persistent convolution state is not L1-resident"
        )
        native_conv_address = int(gdn.fused_conv_state.buffer_address())

    native_frontend_runner = None
    if groups is not None:
        if mode != "verify_many" or fork_position or users is not None or not (
                gdn.native_frontend and gdn.native_output and gdn.native_recurrent):
            raise RuntimeError("shared verify rounds require the native verify_many path")
        slot_states, slot_convs, candidate_states, candidate_convs, row_map = groups
        frontend_output = ensure_native_frontend_groups(gdn, T, len(slot_states), candidate_convs)

        def native_frontend_runner(mega, _taps, _bias, _conv_state):
            return native_frontend(
                mega, w.native_frontend_weight_taps, w.native_frontend_bias, (slot_convs, row_map), frontend_output,
            )
        native_frontend_runner.conv_s1 = native_frontend_runner.conv_s2 = None
        native_frontend_runner.conv_candidates = frontend_output.conv_rows
    elif users is not None:
        if mode != "verify_many" or fork_position or not (gdn.native_frontend and gdn.native_output and gdn.native_recurrent):
            raise RuntimeError("multi-user GDN decode requires the native verify_many path")
        recurrent_users, conv_users = users
        if not len(recurrent_users) == len(conv_users) == T:
            raise ValueError("multi-user GDN decode requires one state pair per row")
        frontend_output = ensure_native_frontend_users(gdn, T)

        def native_frontend_runner(mega, _taps, _bias, _conv_state):
            return native_frontend(
                mega, w.native_frontend_weight_taps, w.native_frontend_bias, conv_users, frontend_output,
            )
        native_frontend_runner.conv_s1 = native_frontend_runner.conv_s2 = None
        native_frontend_runner.conv_candidates = tuple(conv_users)
    elif (T == 1 or verifying) and gdn.native_frontend:
        if T == 1:
            if gdn.native_frontend_output is None:
                from models.demos.blackhole.qwen36.tt.gdn.native_frontend import allocate_native_frontend
                gdn.native_frontend_output = allocate_native_frontend(gdn.device)
            frontend_output = gdn.native_frontend_output
        elif mode == "verify_many":
            frontend_output = ensure_native_frontend_many(gdn, T, fork_position)
        else:
            if gdn.native_frontend_output_m2 is None:
                from models.demos.blackhole.qwen36.tt.gdn.native_frontend import allocate_native_frontend_m2
                gdn.native_frontend_output_m2 = allocate_native_frontend_m2(gdn.device, gdn.fused_conv_state)
            frontend_output = gdn.native_frontend_output_m2
        def native_frontend_runner(mega, _taps, _bias, conv_state):
            return native_frontend(
                mega,
                w.native_frontend_weight_taps,
                w.native_frontend_bias,
                conv_state,
                frontend_output,
            )
        native_frontend_runner.conv_s1 = frontend_output.conv_s1
        native_frontend_runner.conv_s2 = frontend_output.conv_s2
        native_frontend_runner.conv_candidates = frontend_output.conv_candidates
    native_verify_output_runner = None
    if groups is not None:
        group_runner = ensure_verify_groups(gdn, T, len(slot_states))

        def native_verify_output_runner(q, k, v, a, b, db, ds, ones, _live_state, gate, gamma):
            # Single-row groups write their slot in place; longer groups fill candidate_states.
            return (group_runner(q, k, v, a, b, db, ds, ones, (slot_states, candidate_states, row_map), gate, gamma),)
    elif users is not None:
        runner = ensure_verify_users(gdn, T)

        def native_verify_output_runner(q, k, v, a, b, db, ds, ones, _live_state, gate, gamma):
            # No candidates: every row's new state is already written back to its user.
            return (runner(q, k, v, a, b, db, ds, ones, recurrent_users, gate, gamma),)
    elif verifying and gdn.native_frontend and gdn.native_output and gdn.native_recurrent:
        if mode == "verify_many":
            if verify_key not in gdn._fused_verify_output_many:
                from models.demos.blackhole.qwen36.tt.gdn.fused_verify_output import FusedVerifyOutput
                key = id(gdn.device)
                shared = type(gdn)._shared_verify_many_inbox.get(key)
                if shared is None and hasattr(gdn, "_fused_verify_output"):
                    shared = gdn._fused_verify_output.inbox
                runner = FusedVerifyOutput(gdn.device, inbox=shared, fork_position=fork_position)
                type(gdn)._shared_verify_many_inbox[key] = runner.inbox
                gdn._fused_verify_output_many[verify_key] = runner
            native_verify_output_runner = gdn._fused_verify_output_many[verify_key]
        else:
            if not hasattr(gdn, "_fused_verify_output"):
                from models.demos.blackhole.qwen36.tt.gdn.fused_verify_output import FusedVerifyOutput
                gdn._fused_verify_output = FusedVerifyOutput(gdn.device)
            native_verify_output_runner = gdn._fused_verify_output

    native_output_runner = None
    if (T == 1 or mode == "verify_two") and gdn.native_output and native_verify_output_runner is None:
        if T == 1:
            if gdn.native_output_output is None:
                from models.demos.blackhole.qwen36.tt.gdn.native_output import NativeOutput, allocate_native_output
                shared = type(gdn)._shared_native_output_value
                if shared is None:
                    shared = allocate_native_output(gdn.device).value
                    type(gdn)._shared_native_output_value = shared
                gdn.native_output_output = NativeOutput(value=shared)
            output_state = gdn.native_output_output
        else:
            if gdn.native_output_output_m2 is None:
                from models.demos.blackhole.qwen36.tt.gdn.native_output import NativeOutput, allocate_native_output_m2
                shared = type(gdn)._shared_native_output_value_m2
                if shared is None:
                    shared = allocate_native_output_m2(gdn.device).value
                    type(gdn)._shared_native_output_value_m2 = shared
                gdn.native_output_output_m2 = NativeOutput(value=shared)
            output_state = gdn.native_output_output_m2

        def native_output_runner(recurrent, gate, gamma):
            return native_output(recurrent, gate, gamma, output_state)

    native_m1_output_runner = None
    grid = gdn.device.compute_with_storage_grid_size()
    if (
        mode == "recurrent" and T == 1 and x.shape[0] == 1
        and gdn.device.get_num_devices() == 1 and (grid.x, grid.y) == (11, 10)
        and gdn.native_recurrent and gdn.native_frontend and gdn.native_output
        and (gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim) == (48, 128, 128)
        and (w.mega_qkv_dim, w.mega_a_dim, w.mega_b_dim, w.mega_g_dim) == (10240, 64, 64, 6144)
    ):
        if gdn._fused_m1_output is None:
            from models.demos.blackhole.qwen36.tt.gdn.m1.fused_output import FusedM1Output
            key = id(gdn.device)
            runner = FusedM1Output(
                gdn.device, inbox=type(gdn)._shared_m1_inbox.get(key),
                gate_column_offset=w.mega_qkv_dim + w.mega_a_dim + w.mega_b_dim,
            )
            type(gdn)._shared_m1_inbox[key] = runner.inbox
            gdn._fused_m1_output = runner

        def native_m1_output_runner(q, k, v, a, b, db, ds, ones, state, gate, gamma):
            # Binding is per layer, not a process-global address lookup. New
            # request slots may bind new live state; a captured trace retains
            # its own descriptor and original addresses without mutation.
            if state is not persistent_recurrent_state:
                raise RuntimeError("M1 fusion must operate on this layer's live state")
            return gdn._fused_m1_output(
                q, k, v, a, b, db, ds, ones, state, gate, gamma,
                output=gdn.native_output_output.value,
            )

    output, new_state, new_conv_q, new_conv_k, new_conv_v, new_fused_conv = gated_deltanet_forward_ttnn(
        hidden_states=x,
        native_m1_output_runner=native_m1_output_runner,
        q_proj_weight=w.q_proj_weight,
        k_proj_weight=w.k_proj_weight,
        v_proj_weight=w.v_proj_weight,
        a_proj_weight=w.a_proj_weight,
        b_proj_weight=w.b_proj_weight,
        o_proj_weight=w.o_proj_weight,
        q_conv_weight=w.q_conv_weight,
        k_conv_weight=w.k_conv_weight,
        v_conv_weight=w.v_conv_weight,
        q_conv_bias=w.q_conv_bias,
        k_conv_bias=w.k_conv_bias,
        v_conv_bias=w.v_conv_bias,
        A_log=w.A_log,
        dt_bias=w.dt_bias,
        o_norm_weight=w.o_norm_weight,
        g_proj_weight=w.g_proj_weight,
        num_heads=gdn.num_heads,
        num_v_heads=gdn.num_v_heads,
        head_k_dim=gdn.head_k_dim,
        head_v_dim=gdn.head_v_dim,
        conv_kernel_size=gdn.conv_kernel_size,
        use_gate=True,
        norm_eps=gdn.norm_eps,
        output_gate_type=gdn.cfg.output_gate_type,
        device=gdn.device,
        recurrent_state=recurrent_input,
        conv_state_q=gdn.conv_state_q,
        conv_state_k=gdn.conv_state_k,
        conv_state_v=gdn.conv_state_v,
        mode=mode,
        chunk_size=chunk_size,
        q_weight_taps=w.q_weight_taps,
        k_weight_taps=w.k_weight_taps,
        v_weight_taps=w.v_weight_taps,
        q_bias_dev=w.q_bias_dev,
        k_bias_dev=w.k_bias_dev,
        v_bias_dev=w.v_bias_dev,
        qkv_proj_weight=w.qkv_proj_weight,
        q_dim=gdn.cfg.q_dim,
        k_dim=gdn.cfg.k_dim,
        compute_kernel_config=gdn.compute_kernel_config_decode if mode == "recurrent" or verifying else gdn.compute_kernel_config,
        A_neg_precomputed=w.A_neg,
        fused_conv_weight_taps=w.fused_conv_weight_taps,
        fused_conv_bias_dev=w.fused_conv_bias_dev,
        fused_conv_state=gdn.fused_conv_state,
        fused_conv_state_split=getattr(gdn, "split_conv_state", None),
        ab_proj_weight=w.ab_proj_weight,
        mega_fused_weight=w.mega_fused_weight_decode if mode == "recurrent" or verifying else w.mega_fused_weight,
        mega_qkv_dim=w.mega_qkv_dim,
        mega_a_dim=w.mega_a_dim,
        mega_b_dim=w.mega_b_dim,
        mega_g_dim=w.mega_g_dim,
        mega_decode_program_config=gdn.mega_decode_program_config if mode == "recurrent" or verifying else None,
        # Verification rows (T<=32, one tile row) share M=1's program: same K blocking, same bits.
        out_decode_program_config=gdn.out_decode_program_config,
        use_inplace_state=gdn.use_inplace_state,
        chunk_seq_masks=seq_masks,
        valid_len=valid_len,
        native_frontend_runner=native_frontend_runner,
        native_output_runner=native_output_runner,
        native_verify_output_runner=native_verify_output_runner,
        native_decay_bias=w.native_decay_bias,
        native_decay_scale=w.native_decay_scale,
    )
    if verifying:
        return output, new_state, new_fused_conv

    if recurrent_input is not persistent_recurrent_state:
        if list(new_state.shape) != list(persistent_recurrent_state.shape):
            new_state = ttnn.reshape(new_state, list(persistent_recurrent_state.shape))
        new_state_row = ttnn.to_layout(
            new_state,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(new_state_row, persistent_recurrent_state)
        ttnn.deallocate(new_state_row)
        ttnn.deallocate(new_state)
        ttnn.deallocate(recurrent_input)
        ttnn.deallocate(recurrent_row_dram)
        new_state = persistent_recurrent_state
    if (gdn._chunk_inplace_state or gdn.use_inplace_state) and mode == "chunk":
        # Per-chunk traced-prefill replay: write state into the persistent external
        # buffers in place so it carries across execute_trace() calls. gdn.recurrent_state
        # and gdn.fused_conv_state keep pointing at the same (baked) buffer addresses.
        if new_state is not gdn.recurrent_state:
            if list(new_state.shape) != list(gdn.recurrent_state.shape):
                new_state = ttnn.reshape(new_state, list(gdn.recurrent_state.shape))
            ttnn.copy(new_state, gdn.recurrent_state)
            ttnn.deallocate(new_state)
        if new_fused_conv is not None and new_fused_conv is not gdn.fused_conv_state and not isinstance(new_fused_conv, list):
            if new_fused_conv.layout != gdn.fused_conv_state.layout:
                conv_row_dram = ttnn.to_layout(
                    new_fused_conv,
                    gdn.fused_conv_state.layout,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(new_fused_conv)
                if conv_row_dram.memory_config() == gdn.fused_conv_state.memory_config():
                    new_fused_conv = conv_row_dram
                else:
                    new_fused_conv = ttnn.to_memory_config(
                        conv_row_dram,
                        gdn.fused_conv_state.memory_config(),
                    )
                    ttnn.deallocate(conv_row_dram)
            ttnn.copy(new_fused_conv, gdn.fused_conv_state)
            ttnn.deallocate(new_fused_conv)
        return output

    if native_state_address is not None:
        assert int(new_state.buffer_address()) == native_state_address, (
            "native GDN replaced the persistent recurrent-state buffer"
        )
        assert new_state.memory_config().buffer_type == ttnn.BufferType.L1, (
            "native GDN moved persistent recurrent state out of L1"
        )
        assert new_fused_conv is not None and not isinstance(new_fused_conv, list), (
            "native GDN replaced fused convolution state with split buffers"
        )
        assert int(new_fused_conv.buffer_address()) == native_conv_address, (
            "native GDN replaced the persistent convolution-state buffer"
        )

    gdn.recurrent_state = new_state
    if isinstance(new_fused_conv, list):
        gdn.split_conv_state = new_fused_conv
    elif new_fused_conv is not None:
        gdn.fused_conv_state = new_fused_conv
    else:
        gdn.conv_state_q = new_conv_q
        gdn.conv_state_k = new_conv_k
        gdn.conv_state_v = new_conv_v
    return output
