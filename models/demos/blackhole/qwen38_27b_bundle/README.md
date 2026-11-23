<!-- SPDX-FileCopyrightText: © 2026 Lottolabs -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Qwen3.8-27B P150: tt-model container package support files

- `vllm-model-redirect.json` (`VLLM_MODEL_REDIRECT_PATH`): `tt-model serve` hands vLLM the weights repo id; the
  loadable Hugging Face folder of the release (config, tokenizer, processor, generation config, native checkpoint)
  is its `checkpoint/` sub-folder.  vLLM's model redirect maps the repo id to that folder of the pinned snapshot
  in the container's Hugging Face cache (`/hf/hub`, i.e. `HF_HUB_CACHE` left at `$HF_HOME/hub`).
- `../../../../qwen-runtime-relocation.json` (`QWEN36_RUNTIME_RELOCATION`): see `native_checkpoint.validate_runtime`.
