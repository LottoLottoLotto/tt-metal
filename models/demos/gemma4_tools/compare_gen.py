#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Lottolabs
# SPDX-License-Identifier: Apache-2.0
"""Compare greedy token streams of two generate runs: compare_gen.py REF_DIR NEW_DIR [--json OUT]."""
import json
import sys
from pathlib import Path


def load(d):
    return {r["id"]: r for r in (json.loads(l) for l in open(Path(d) / "generations.jsonl") if l.strip())}


def main():
    ref, new = load(sys.argv[1]), load(sys.argv[2])
    rows, ok = [], True
    for k, r in ref.items():
        n = new.get(k)
        if n is None:
            rows.append({"id": k, "status": "missing"})
            ok = False
            continue
        a, b = r["gen_ids"], n["gen_ids"]
        div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        same = a == b
        ok &= same
        rows.append({"id": k, "tokens": len(a), "identical": same, "first_divergence": div if not same else None,
                     "ref_tok_s": r.get("decode_tok_s"), "new_tok_s": n.get("decode_tok_s")})
    res = {"identical_all": ok, "n": len(rows), "tokens": sum(r.get("tokens", 0) for r in rows), "rows": rows}
    for r in rows:
        print(r)
    print("IDENTICAL_ALL", ok, "tokens", res["tokens"])
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
