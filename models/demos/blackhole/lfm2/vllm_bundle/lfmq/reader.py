# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Reader for the TT-native quantized LFM2.5-24B-A2B checkpoint (the release's checkpoint/). Needs torch + safetensors only.

    from lfmq.reader import open_checkpoint          # sys.path += [<directory holding lfmq/>]
    ck = open_checkpoint("<release>/checkpoint")
    ck.keys()        # HF tensor names (the 7610 of model.safetensors.index.json) + "lm_head.weight"
    ck.fmt(name)     # 'bfp4' | 'bfp8' | 'bf16' | 'fp32'
    ck.shape(name)   # HF shape
    ck.get(name)     # exact tensor in HF name / shape / orientation ([out, in] for Linears):
                     #   bfp4/bfp8 -> bfloat16 values already on the TT grid: ttnn.from_torch(W.T, bfloat4_b/bfloat8_b)
                     #   reproduces them bit-exactly (blocks = 16 consecutive output channels at one input index, i.e. 16
                     #   consecutive elements along N of W^T = [in, out]); bf16 -> original; fp32 -> original (expert_bias)
    ck.codes(name)   # (q int32 [in, out], e int32 [in, out/16]) stored BFP codes of a bfp tensor (W^T orientation)

"lm_head.weight" is the LM head at BFP8 (RTN of the tied embedding, [128000, 2048]; ttnn gets its transpose
[2048, 128000]); "model.embed_tokens.weight" stays the original BF16 table.
Manifest (manifest.json): per tensor fmt, shape, file, sha256 of the stored bytes (q then e for bfp, raw otherwise) and
value_sha256 = sha256 of get(name)'s raw bytes; per file sha256. verify(name) checks both.
"""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

try:
    from . import codec
except ImportError:  # loaded as a plain module
    import importlib.util

    _spec = importlib.util.spec_from_file_location("lfmq_codec", Path(__file__).with_name("codec.py"))
    codec = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(codec)


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self._t = self.manifest["tensors"]
        self._fh = {}

    def keys(self):
        return list(self._t)

    def __contains__(self, name):
        return name in self._t

    def fmt(self, name):
        return self._t[name]["fmt"]

    def shape(self, name):
        return tuple(self._t[name]["shape"])

    def _file(self, f):
        if f not in self._fh:
            self._fh[f] = safe_open(str(self.path / f), framework="pt")
        return self._fh[f]

    def _stored(self, name):
        m = self._t[name]
        fh = self._file(m["file"])
        if m["fmt"] in codec.MANT:
            return fh.get_tensor(name + ".q"), fh.get_tensor(name + ".e")
        return (fh.get_tensor(name),)

    def codes(self, name):
        m = self._t[name]
        assert m["fmt"] in codec.MANT, (name, m["fmt"])
        qs, es = self._stored(name)
        return codec.codes(qs, m["fmt"]), es.to(torch.int32)

    def get(self, name):
        m = self._t[name]
        st = self._stored(name)
        if m["fmt"] in codec.MANT:
            v = codec.decode(st[0], st[1], m["fmt"])
        else:
            v = st[0]
            assert v.dtype == {"bf16": torch.bfloat16, "fp32": torch.float32}[m["fmt"]], (name, v.dtype)
        assert tuple(v.shape) == tuple(m["shape"]), (name, v.shape, m["shape"])
        return v

    def verify(self, name):
        """True iff the stored bytes and the decoded values match the manifest hashes."""
        m = self._t[name]
        h = hashlib.sha256()
        for t in self._stored(name):
            h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
        v = self.get(name)
        hv = hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        return h.hexdigest() == m["sha256"] and hv == m["value_sha256"]


def open_checkpoint(path):
    return Checkpoint(path)
