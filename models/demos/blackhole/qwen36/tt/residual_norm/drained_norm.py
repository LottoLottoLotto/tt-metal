# SPDX-License-Identifier: Apache-2.0
"""ttnn.rms_norm(x, residual_input_tensor=residual) on width-sharded L1 inputs, program for program (the native
sharded factory's descriptor: FUSE_PRE_ADD, compute, writer and circular buffers unchanged), except that its two
multicast readers drain their non-posted semaphore increments before they exit (drained_reader.cpp).

The model's final residual RMSNorm runs right before the LM head, whose in0 receivers end with an atomic barrier.
A receiver of the native norm exits with its last increment possibly in flight; when the acknowledgement lands after
the LM head's in0 receiver on that core took its NoC counter baseline, that receiver's barrier never completes and
the device hangs (observed: one LM-head core with NCRISC in noc_async_atomic_barrier, tools/tt-triage). The per-layer
fused residual norms (fused_residual_norm.py, norm_reader.cpp) already drain the same native readers."""
from pathlib import Path

import ttnn

_WRAPPER = str(Path(__file__).resolve().with_name("drained_reader.cpp"))
_SENDER = "reader_mcast_sender_unary_sharded_ln.cpp"
_RECEIVER = "reader_mcast_receiver_unary_sharded_ln.cpp"


def drained_rms_norm(x, residual, weight, eps, program_config, memory_config, compute_kernel_config):
    """x, residual: width-sharded BF16 L1 [.., rows, hidden] on memory_config's grid; returns the sharded output."""
    output = ttnn.empty(list(x.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x.device(),
                        memory_config=memory_config)
    params = ttnn.LayerNormParams()
    params.norm_type = ttnn.LayerNormType.RMSNORM
    params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    params.eps = float(eps)
    params.output_mem_config = memory_config
    params.program_config = program_config
    params.compute_kernel_config = compute_kernel_config
    params.dtype = ttnn.bfloat16
    inputs = ttnn.LayerNormInputs(x)
    inputs.residual_input_tensor = residual
    inputs.weight = weight
    program = ttnn.LayerNormShardedProgramFactory.create_descriptor(params, inputs, output)
    kernels = list(program.kernels)
    readers = 0
    for kernel in kernels:
        name = Path(kernel.kernel_source).name
        if name in (_SENDER, _RECEIVER):
            kernel.defines = list(kernel.defines) + ([("DRAIN_SENDER", "1")] if name == _SENDER else [])
            kernel.kernel_source = _WRAPPER
            readers += 1
    if not readers:
        raise RuntimeError("the native sharded RMSNorm descriptor has no multicast reader to drain")
    program.kernels = kernels
    ttnn.generic_op([x, residual, weight, output], program)
    return output
