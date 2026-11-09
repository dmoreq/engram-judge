#!/usr/bin/env python3
"""Export a Laya checkpoint into the layout `eg-laya` loads, plus golden fixtures.

Offline development tool. Runtime never imports this module; the Rust binary reads only
`model.safetensors`, `encoder/config.json`, `tokenizer/tokenizer.json` and `lazy.json`.

Usage:
    .venv/bin/python scripts/export_laya.py --checkpoint laya-multilingual
    .venv/bin/python scripts/export_laya.py --checkpoint both --fixtures 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request

CHECKPOINTS = {
    "laya": "convaiinnovations/laya",
    "laya-multilingual": "convaiinnovations/laya-multilingual",
}

FILES = [
    "config.json",
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Code-pair cases: two snippets under one question. `label` is only for corpus bookkeeping and
# is not used by the parity fixtures.
DUPLICATE_CASES = [
    {
        "id": "dup-rename",
        "label": 1,
        "ins": "Both code snippets are pasted below. Do they implement the same algorithm, "
        "ignoring identifier names, formatting and comment changes?",
        "false_desc": "the two snippets compute different results",
        "true_desc": "the two snippets are behaviourally equivalent",
        "state": "snippet A:\n"
        "fn total(xs: &[u32]) -> u32 {\n"
        "    let mut acc = 0;\n"
        "    for x in xs { acc += *x; }\n"
        "    acc\n"
        "}\n"
        "snippet B:\n"
        "fn sum_all(values: &[u32]) -> u32 {\n"
        "    let mut total = 0;\n"
        "    for value in values { total += *value; }\n"
        "    total\n"
        "}\n",
    },
    {
        "id": "dup-loop-vs-iter",
        "label": 1,
        "ins": "Both code snippets are pasted below. Do they implement the same algorithm, "
        "ignoring identifier names, formatting and comment changes?",
        "false_desc": "the two snippets compute different results",
        "true_desc": "the two snippets are behaviourally equivalent",
        "state": "snippet A:\n"
        "def names(rows):\n"
        "    out = []\n"
        "    for row in rows:\n"
        "        out.append(row[\"name\"].strip())\n"
        "    return out\n"
        "snippet B:\n"
        "def collect(rows):\n"
        "    return [r[\"name\"].strip() for r in rows]\n",
    },
    {
        "id": "dup-reordered-guards",
        "label": 1,
        "ins": "Both code snippets are pasted below. Do they implement the same algorithm, "
        "ignoring identifier names, formatting and comment changes?",
        "false_desc": "the two snippets compute different results",
        "true_desc": "the two snippets are behaviourally equivalent",
        "state": "snippet A:\n"
        "function clamp(value, low, high) {\n"
        "  if (value < low) return low;\n"
        "  if (value > high) return high;\n"
        "  return value;\n"
        "}\n"
        "snippet B:\n"
        "function bound(v, lo, hi) {\n"
        "  if (v > hi) return hi;\n"
        "  if (v < lo) return lo;\n"
        "  return v;\n"
        "}\n",
    },
    {
        "id": "near-miss-off-by-one",
        "label": 0,
        "ins": "Both code snippets are pasted below. Do they implement the same algorithm, "
        "ignoring identifier names, formatting and comment changes?",
        "false_desc": "the two snippets compute different results",
        "true_desc": "the two snippets are behaviourally equivalent",
        "state": "snippet A:\n"
        "def page(items, size):\n"
        "    out = []\n"
        "    for i in range(0, len(items), size):\n"
        "        out.append(items[i:i + size])\n"
        "    return out\n"
        "snippet B:\n"
        "def chunk(values, step):\n"
        "    out = []\n"
        "    for i in range(0, len(values), step):\n"
        "        out.append(values[i:i + step - 1])\n"
        "    return out\n",
    },
    {
        "id": "near-miss-inverted-predicate",
        "label": 0,
        "ins": "Both code snippets are pasted below. Do they implement the same algorithm, "
        "ignoring identifier names, formatting and comment changes?",
        "false_desc": "the two snippets compute different results",
        "true_desc": "the two snippets are behaviourally equivalent",
        "state": "snippet A:\n"
        "fn is_ready(flags: &Flags) -> bool {\n"
        "    flags.enabled && !flags.paused\n"
        "}\n"
        "snippet B:\n"
        "fn can_start(flags: &Flags) -> bool {\n"
        "    flags.enabled && flags.paused\n"
        "}\n",
    },
    {
        "id": "unrelated",
        "label": 0,
        "ins": "Both code snippets are pasted below. Do they implement the same algorithm, "
        "ignoring identifier names, formatting and comment changes?",
        "false_desc": "the two snippets compute different results",
        "true_desc": "the two snippets are behaviourally equivalent",
        "state": "snippet A:\n"
        "def sha256_hex(data):\n"
        "    return hashlib.sha256(data).hexdigest()\n"
        "snippet B:\n"
        "def median(values):\n"
        "    ordered = sorted(values)\n"
        "    mid = len(ordered) // 2\n"
        "    return ordered[mid]\n",
    },
]

SUPERSEDE_CASES = [
    {
        "id": "mem-revision",
        "label": 1,
        "ins": "Both notes describe the same project fact. Does the new note replace the old "
        "one, that is, does it make the old note obsolete?",
        "false_desc": "the notes describe different facts and both stay valid",
        "true_desc": "the new note supersedes the old note",
        "state": "old note:\n"
        "The index schema version is 9; rebuild the database after any schema change.\n"
        "new note:\n"
        "The index schema version is 10; rebuild the database after any schema change.\n",
    },
    {
        "id": "mem-complementary",
        "label": 0,
        "ins": "Both notes describe the same project fact. Does the new note replace the old "
        "one, that is, does it make the old note obsolete?",
        "false_desc": "the notes describe different facts and both stay valid",
        "true_desc": "the new note supersedes the old note",
        "state": "old note:\n"
        "Symbol search uses SQLite FTS5 over the symbols table.\n"
        "new note:\n"
        "Caller counts are stored in the edges table and ranked with bm25.\n",
    },
    {
        "id": "mem-scope-narrower",
        "label": 0,
        "ins": "Both notes describe the same project fact. Does the new note replace the old "
        "one, that is, does it make the old note obsolete?",
        "false_desc": "the notes describe different facts and both stay valid",
        "true_desc": "the new note supersedes the old note",
        "state": "old note:\n"
        "Memory notes are confined to .engram/memory.db and ~/.engram/memory.db.\n"
        "new note:\n"
        "Global memory writes go to $ENGRAM_GLOBAL_DIR when that variable is set.\n",
    },
]


def fetch(repo: str, rel_path: str, dest: str) -> str:
    """Download one Hub file with a coarse progress report, reusing a local copy."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  cached {rel_path} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{rel_path}"
    tmp = dest + ".part"
    started = time.time()
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done, next_mark = 0, 0.0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                if pct >= next_mark:
                    elapsed = time.time() - started
                    print(f"  {rel_path} {pct:5.1f}% {done / 1e6:7.1f}/{total / 1e6:.1f} MB "
                          f"{done / 1e6 / max(elapsed, 1e-6):5.1f} MB/s", flush=True)
                    next_mark += 5.0
    os.replace(tmp, dest)
    return dest


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(checkpoint: str, root: str) -> str:
    """Materialise `<root>/<checkpoint>/` with only the files the runtime needs."""
    repo = CHECKPOINTS[checkpoint]
    cache = os.path.join(root, ".cache", checkpoint)
    out = os.path.join(root, checkpoint)
    os.makedirs(out, exist_ok=True)
    print(f"[{checkpoint}] downloading {repo}")
    for rel in FILES:
        fetch(repo, rel, os.path.join(cache, rel))
    for rel in FILES:
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(os.path.join(cache, rel), dst)
        except OSError:
            shutil.copy2(os.path.join(cache, rel), dst)
    with open(os.path.join(cache, "rl_agent_config.json")) as fh:
        model_cfg = json.load(fh)
    with open(os.path.join(cache, "encoder/config.json")) as fh:
        encoder_cfg = json.load(fh)
    lazy = {
        "checkpoint": repo,
        "head_layers": model_cfg.get("head_layers", 2),
        "temperature": model_cfg.get("temperature", [1.0, 1.0, 1.0]),
        "qtype_noul": 2,
        "encoder": {
            "hidden_size": encoder_cfg["hidden_size"],
            "num_hidden_layers": encoder_cfg["num_hidden_layers"],
            "num_attention_heads": encoder_cfg["num_attention_heads"],
            "intermediate_size": encoder_cfg["intermediate_size"],
            "global_attn_every_n_layers": encoder_cfg["global_attn_every_n_layers"],
            "local_attention": encoder_cfg.get("local_attention", encoder_cfg.get("sliding_window", 128)),
            "sliding_window": encoder_cfg.get("sliding_window"),
            "layer_types": encoder_cfg.get("layer_types"),
            "global_rope_theta": encoder_cfg.get("global_rope_theta")
            or encoder_cfg.get("rope_parameters", {}).get("full_attention", {}).get("rope_theta"),
            "local_rope_theta": encoder_cfg.get("local_rope_theta")
            or encoder_cfg.get("rope_parameters", {}).get("sliding_attention", {}).get("rope_theta"),
        },
        "weights_sha256": sha256(os.path.join(out, "model.safetensors")),
    }
    with open(os.path.join(out, "lazy.json"), "w") as fh:
        json.dump(lazy, fh, indent=2, sort_keys=True)
        fh.write("\n")
    total = sum(os.path.getsize(os.path.join(out, f)) for f in FILES if os.path.exists(os.path.join(out, f)))
    print(f"[{checkpoint}] ready at {out} ({total / 1e6:.1f} MB, "
          f"global_rope={lazy['encoder']['global_rope_theta']}, "
          f"local_rope={lazy['encoder']['local_rope_theta']}, "
          f"sliding_window={lazy['encoder']['sliding_window']})")
    return out


def golden(checkpoint_dir: str, fixtures_path: str, cases: list, limit: int) -> None:
    """Run the reference PyTorch model and dump ids, logits and probabilities."""
    os.environ.setdefault("USE_TF", "0")
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import build_model, build_sequence

    with open(os.path.join(checkpoint_dir, "rl_agent_config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(checkpoint_dir, "lazy.json")) as fh:
        lazy = json.load(fh)
    enc_dir = os.path.join(checkpoint_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir, pretrained=False)
    model.load_state_dict(load_file(os.path.join(checkpoint_dir, "model.safetensors")), strict=True)
    model.eval()
    tok_dir = os.path.join(checkpoint_dir, "tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_dir)
    temperature = float(lazy["temperature"][2])
    max_len, head_max_len = 1024, 192

    rows = []
    for case in cases[:limit]:
        q = {"t": "noul", "ins": case["ins"],
             "crit": {"false": case["false_desc"], "true": case["true_desc"]}}
        ids, markers = build_sequence(tok, case["state"], q, max_len=max_len, head_max_len=head_max_len)
        tensors = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "marker_pos": torch.tensor([markers], dtype=torch.long),
            "marker_mask": torch.ones((1, len(markers)), dtype=torch.bool),
            "qtype": torch.tensor([2], dtype=torch.long),
        }
        with torch.no_grad():
            logits, _act = model(**tensors)
        z = logits[0].double().numpy() / temperature
        p = np.exp(z - z.max())
        p = p / p.sum()
        rows.append({
            "id": case["id"],
            "label": case["label"],
            "state": case["state"],
            "ins": case["ins"],
            "false_desc": case["false_desc"],
            "true_desc": case["true_desc"],
            "input_ids": ids,
            "markers": markers,
            "logits": [float(v) for v in logits[0].float().numpy()],
            "probs": [float(v) for v in p],
            "temperature": temperature,
            "tokens": len(ids),
        })
        print(f"  {case['id']}: {len(ids)} tokens, p(true)={p[1]:.6f}")

    os.makedirs(os.path.dirname(fixtures_path), exist_ok=True)
    with open(fixtures_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[{os.path.basename(checkpoint_dir)}] wrote {len(rows)} fixtures to {fixtures_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="both",
                        choices=["laya", "laya-multilingual", "both"])
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, ".models"))
    parser.add_argument("--fixtures", type=int, default=0,
                        help="number of duplicate cases per checkpoint to dump as golden fixtures")
    args = parser.parse_args()

    names = list(CHECKPOINTS) if args.checkpoint == "both" else [args.checkpoint]
    for name in names:
        out = prepare(name, args.root)
        if args.fixtures:
            fixtures = os.path.join(REPO_ROOT, "tests", "fixtures", f"golden_{name.replace('-', '_')}.jsonl")
            cases = DUPLICATE_CASES + (SUPERSEDE_CASES if args.fixtures > len(DUPLICATE_CASES) else [])
            golden(out, fixtures, cases, args.fixtures)
    with open(os.path.join(args.root, "cases.json"), "w") as fh:
        json.dump({"duplicate": DUPLICATE_CASES, "supersede": SUPERSEDE_CASES}, fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
