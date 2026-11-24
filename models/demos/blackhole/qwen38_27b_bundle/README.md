<!-- SPDX-FileCopyrightText: © 2026 Lottolabs -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Qwen3.8-27B P150: tt-model container package support files

- `vllm-model-redirect.json` (`VLLM_MODEL_REDIRECT_PATH`): `tt-model serve` hands vLLM the weights repo id; the
  loadable Hugging Face folder of the release (config, tokenizer, processor, generation config, native checkpoint)
  is its `checkpoint/` sub-folder.  vLLM's model redirect maps the repo id, and the snapshot root that vLLM substitutes
  for it under `HF_HUB_OFFLINE=1`, to that folder of the pinned snapshot in the container's Hugging Face cache (`/hf/hub`, i.e. `HF_HUB_CACHE` left at `$HF_HOME/hub`).
- `../../../../qwen-runtime-relocation.json` (`QWEN36_RUNTIME_RELOCATION`): see `native_checkpoint.validate_runtime`.
- `QWEN36_NATIVE_PROOFS` (`native_checkpoint.require_equivalence`): this tree is the runtime of the release's 262K image
  (ctx6-kv4-s27v8; its 8K / 128K settings never touch the 4-bit V cache code), so its reload proofs are the release's
  `proofs-262k/` pair, not the pair in `checkpoint/` (recorded on the 8K / 128K image).  The release's launcher mounts
  that pair over the checkpoint's; the package names the directory instead.
