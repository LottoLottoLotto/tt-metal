# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-address post-layer prefill features for a context-conditioned drafter."""
import torch
import ttnn


class PrefillFeatures:
    def __init__(self, model, layer_indices, buckets, consumer):
        if model.num_devices != 1:
            raise ValueError("Draft feature capture currently requires one device")
        if getattr(model, "_chunked_prepared_key", None) is not None:
            raise RuntimeError("Allocate feature buffers before prefill warmup and trace capture")
        self.model, self.consumer = model, consumer
        self.layers = tuple(layer_indices)
        if not self.layers or len(set(self.layers)) != len(self.layers) or any(
            index < 0 or index >= len(model.layers) for index in self.layers
        ):
            raise ValueError("Feature layers must be distinct valid target indices")
        self.buffers = {
            int(bucket): {
                index: ttnn.from_torch(torch.zeros((1, bucket, model.args.dim), dtype=torch.bfloat16),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=model.device,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for index in self.layers
            }
            for bucket in buckets
        }
        if not self.buffers or any(bucket <= 0 or bucket % 32 for bucket in self.buffers):
            raise ValueError("Feature buckets must be positive tile multiples")

    def begin_sequence(self):
        """Discard the previous request before selectively committing the visible suffix."""
        self.consumer.reset()

    def capture(self, index, hidden):
        if index in self.layers:
            bucket = int(hidden.shape[-2])
            ttnn.copy(hidden, self.buffers[bucket][index])

    def save(self, bucket):
        """Host copies of the last capture at this bucket, in feature-layer order."""
        return [ttnn.to_torch(self.buffers[bucket][index]) for index in self.layers]

    def restore(self, bucket, saved):
        """Write a save() back into the fixed capture buffers (host->device, no allocation)."""
        if saved is None or len(saved) != len(self.layers):
            raise ValueError("Prefill feature snapshot does not match the feature layers")
        for index, host in zip(self.layers, saved):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.buffers[bucket][index]
            )

    def commit(self, start_position, valid_length, bucket, sequence_length=None):
        if not 0 < valid_length <= bucket:
            raise ValueError("Only real prefill rows may enter the draft context")
        source_offset = 0
        if sequence_length is not None:
            if not start_position + valid_length <= sequence_length:
                raise ValueError("Prefill feature segment exceeds the sequence")
            invisible = max(0, sequence_length - self.consumer.cache.window - start_position)
            source_offset = (invisible // 32) * 32
            if source_offset >= valid_length:
                return
        buffers = self.buffers[bucket]
        features = ttnn.concat([buffers[index] for index in self.layers], dim=-1,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        try:
            self.consumer.prefill(ttnn.reshape(features, [1, 1, bucket, len(self.layers) * self.model.args.dim]),
                                  start_position, valid_length, source_offset)
        finally:
            ttnn.deallocate(features)

    def release(self):
        for buffers in self.buffers.values():
            for tensor in buffers.values():
                ttnn.deallocate(tensor)
        self.buffers.clear()
