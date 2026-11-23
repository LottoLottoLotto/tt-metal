# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0
import os

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32
SHARD_HEIGHT = TILE


class LayerNorm(LightweightModule):
    def __init__(
        self,
        device,
        dim,
        state_dict,
        state_dict_prefix,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat8_b,
        model_config=None,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.device = device
        self.eps = eps

        torch_weight = (
            state_dict[f"{state_dict_prefix}.weight"].unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim])
        )
        torch_bias = state_dict[f"{state_dict_prefix}.bias"].unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim])
        if weight_cache_path is None:
            cache_name = lambda *_: None
        else:
            cache_name = lambda suffix: weight_cache_path / (state_dict_prefix + f"{suffix}")

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name("weight"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.bias = ttnn.as_tensor(
            torch_bias,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name("bias"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        # QWEN36_VISION_L1_LEAN=1: the vision tower shares L1 with the native text runtime's resident
        # state, so the interleaved norm runs as streamed ops (see _lean_forward) on [1, 1, 1, dim] rows.
        self.lean = os.environ.get("QWEN36_VISION_L1_LEAN") == "1"
        if self.lean:
            row = lambda t: t.reshape(1, 1, 1, dim)
            self.lean_weight = ttnn.as_tensor(
                row(state_dict[f"{state_dict_prefix}.weight"]), device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=weight_memory_config,
                cache_file_name=cache_name("lean_weight"),
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
            )
            self.lean_bias = ttnn.as_tensor(
                row(state_dict[f"{state_dict_prefix}.bias"]), device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=weight_memory_config,
                cache_file_name=cache_name("lean_bias"),
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
            )

        if model_config:
            self.sharded_input_config = model_config["SHARDED_NORM_INPUT_MEMCFG"]
            self.sharded_program_config = model_config["SHARDED_NORM_PRGM_CFG"]
            self.sharded_output_config = model_config["SHARDED_NORM_OUTPUT_MEMCFG"]
        else:
            assert (
                dim % SHARD_HEIGHT == 0
            ), f"Input dimension dim ({dim}) must be a multiple of SHARD_HEIGHT ({SHARD_HEIGHT})"
            shard_width_hidden_dim_across_32_cores = dim // SHARD_HEIGHT
            core_grid = ttnn.CoreGrid(x=8, y=SHARD_HEIGHT // 8)
            # core_grid = ttnn.CoreGrid(x=8, y=8)
            self.sharded_input_config = ttnn.create_sharded_memory_config(
                shape=(SHARD_HEIGHT, shard_width_hidden_dim_across_32_cores),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.sharded_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[core_grid.x, core_grid.y],
                subblock_w=shard_width_hidden_dim_across_32_cores // TILE,
                block_h=SHARD_HEIGHT // TILE,
                block_w=shard_width_hidden_dim_across_32_cores // TILE,
                inplace=False,
            )
            self.sharded_output_config = self.sharded_input_config

    def forward(self, x: ttnn.Tensor, in_sharded=False, out_sharded=False) -> ttnn.Tensor:
        if in_sharded:
            x = ttnn.layer_norm(
                x,
                epsilon=self.eps,
                weight=self.weight,
                bias=self.bias,
                program_config=self.sharded_program_config,
                memory_config=self.sharded_output_config,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            )
            if out_sharded:
                return x
            x_interleaved = ttnn.sharded_to_interleaved(x)
            x.deallocate(True)
            return x_interleaved
        else:  # Interleaved rmsnorm does not need program or memory configs
            assert not out_sharded, "Non-sharded version of RMSNorm cannot output a sharded tensor"
            if self.lean:
                return self._lean_forward(x)
            x = ttnn.layer_norm(
                x,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                ),
            )
            return x

    def _lean_forward(self, x):
        """LayerNorm from reductions and elementwise ops. ttnn.layer_norm keeps whole rows (dim/32
        tiles) in each of its ~6 circular buffers, which overruns the L1 left free beside the text
        model's resident GDN state; these ops stream a few tiles at a time."""
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        mean = ttnn.mean(x, dim=-1, keepdim=True, compute_kernel_config=cfg)
        xc = ttnn.subtract(x, mean)
        ttnn.deallocate(mean)
        sq = ttnn.multiply(xc, xc)
        var = ttnn.mean(sq, dim=-1, keepdim=True, compute_kernel_config=cfg)
        ttnn.deallocate(sq)
        inv = ttnn.rsqrt(ttnn.add(var, self.eps))
        ttnn.deallocate(var)
        y = ttnn.multiply(xc, inv)
        ttnn.deallocate(xc)
        ttnn.deallocate(inv)
        out = ttnn.add(ttnn.multiply(y, self.lean_weight), self.lean_bias)
        ttnn.deallocate(y)
        return out
