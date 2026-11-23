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
    ):
        os.environ.pop(_flag, None)
    del _flag
# HF Gated DeltaNet prefill rule: no diagonal damping of the chunked delta-rule L matrix (the runtime default was
# 0.25). Device eval, d8, all corpora vs CPU BF16: dPPL +7.1% / KL .104 at 0.25 -> +5.0% / .081 at 0 (Release35D).
# An explicit QWEN_GDN_DIAG_ALPHA in the environment still wins.
os.environ.setdefault("QWEN_GDN_DIAG_ALPHA", "0")
