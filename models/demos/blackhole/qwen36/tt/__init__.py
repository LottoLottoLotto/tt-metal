"""Qwen hybrid runtime execution policy.

Set QWEN36_GENERIC_BASELINE=1 before importing the runtime. This overrides
inherited release-image optimization flags; precision is still supplied by the
checkpoint/evaluator, independently of execution policy.
"""
import os


if os.environ.get("QWEN36_GENERIC_BASELINE") == "1":
    for _flag in (
        "QWEN36_MTP",
        "QWEN36_SINGLE_1D_DECODE",
        "QWEN36_FUSED_QKV",
        "QWEN36_FUSED_GATE_UP",
        "QWEN_GDN_RECURRENT_FUSED",
        "QWEN_GDN_FRONTEND_FUSED",
        "QWEN_GDN_OUTPUT_FUSED",
        "QWEN36_SHARDED_FINAL_NORM",
        "QWEN36_FUSED_FINAL_RESIDUAL_NORM",
        "QWEN36_FLAT_DECODE_BATCH",
        "QWEN36_PREFILL_OVERLAP",
        "QWEN_SDPA_BF8",
        "QWEN_GDN_INV_DOUBLING",
        "QWEN9B_SDPA_QK64",
        "TT_DECODE_BUCKETING",
    ):
        os.environ[_flag] = "0"
    # No model-independent diagonal damping in the reference delta rule.
    os.environ["QWEN_GDN_DIAG_ALPHA"] = "0"
    # The installed C++ packed readers use getenv presence, not its value.
    for _flag in (
        "QWEN36_FUSED_GATE_UP_PACKED",
        "QWEN36_MLP_DOWN_PACKED",
        "QWEN36_LM_HEAD_PACKED",
        "QWEN_GDN_MEGA_PACKED",
        "QWEN36_GATEUP_REQUIRE_PACKED",
        "QWEN36_LM_HEAD_REQUIRE_PACKED",
        "QWEN36_MLP_DOWN_REQUIRE_PACKED",
        "QWEN_GDN_MEGA_REQUIRE_PACKED",
        "QWEN36_GATEUP_INSTRUMENT",
        "QWEN_GDN_DUMP",
        "QWEN36_IN0_MCAST_CHUNK_TILES",
        "QWEN36_MLP_DOWN_DUAL_NOC",
        "QWEN36_PROJ_O_ROTATING_COLS",
        "QWEN36_WEIGHT_PREFETCH",
    ):
        os.environ.pop(_flag, None)
    del _flag
elif os.environ.get("QWEN36_OPTIMIZED_27B") == "1":
    # Exact decode-matmul data-movement policy (read by the rebuilt matmul factory when programs are
    # created; each can be overridden or disabled with "0"/unset-equivalent values in the environment):
    # activation multicast in chunks of up to 32 K tiles per handshake (K blocking unchanged),
    # MLP-down weight reads split over both NoCs, and [6144, 5120] BF8 output projections in the
    # rotating worker-bank layout with two columns per worker (80 workers). Decode verification streams
    # the GDN/attention output projections through the Blackhole DRAM-core tensor prefetcher
    # (weight_prefetch.py; same K blocks, bitwise-equal outputs) so their first weight blocks land in L1
    # during the preceding recurrence/attention work.
    for _flag, _value in (
        ("QWEN36_IN0_MCAST_CHUNK_TILES", "32"),
        ("QWEN36_MLP_DOWN_DUAL_NOC", "1"),
        ("QWEN36_PROJ_O_ROTATING_COLS", "2"),
        ("QWEN36_WEIGHT_PREFETCH", "proj_o"),
    ):
        os.environ.setdefault(_flag, _value)
    del _flag, _value