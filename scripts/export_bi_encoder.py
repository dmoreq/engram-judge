#!/usr/bin/env python3
"""Wrap a plain ModernBERT/sentence-encoder checkout into the layout `eg-laya embed` loads.

The judge layout expects `encoder/config.json`, `tokenizer/`, `model.safetensors` with Laya's
`encoder.*` prefixes. A public sentence-encoder ships `config.json`, `tokenizer.json` and a
standard `model.*` checkpoint. This script normalises it: hardlinks the weights, copies the config
and tokenizer, and writes `lazy.json` with the weight prefix and pooling mode.

Usage:
    .venv/bin/python scripts/export_bi_encoder.py --model .models/owl \
        --checkpoint Shuu12121/CodeCloneDetection-ModernBERT-Owl --pooling cls
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local sentence-encoder directory")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--prefix", default="", help="weight prefix in model.safetensors")
    args = parser.parse_args()

    root = os.path.abspath(args.model)
    config_src = os.path.join(root, "config.json")
    weights_src = os.path.join(root, "model.safetensors")
    tokenizer_src = os.path.join(root, "tokenizer.json")
    for path in (config_src, weights_src, tokenizer_src):
        if not os.path.exists(path):
            print(f"missing {path}", file=sys.stderr)
            return 1

    os.makedirs(os.path.join(root, "encoder"), exist_ok=True)
    os.makedirs(os.path.join(root, "tokenizer"), exist_ok=True)
    shutil.copy2(config_src, os.path.join(root, "encoder", "config.json"))
    if os.path.abspath(tokenizer_src) != os.path.join(root, "tokenizer", "tokenizer.json"):
        shutil.copy2(tokenizer_src, os.path.join(root, "tokenizer", "tokenizer.json"))

    with open(config_src) as fh:
        config = json.load(fh)
    lazy = {
        "checkpoint": args.checkpoint or os.path.basename(root),
        "head_layers": 0,
        "temperature": [1.0, 1.0, 1.0],
        "qtype_noul": 2,
        "weight_prefix": args.prefix,
        "pooling": args.pooling,
        "encoder": {
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "intermediate_size": config["intermediate_size"],
            "global_attn_every_n_layers": config["global_attn_every_n_layers"],
            "local_attention": config.get("local_attention", 128),
            "global_rope_theta": config.get("global_rope_theta")
            or config.get("rope_parameters", {}).get("full_attention", {}).get("rope_theta"),
            "local_rope_theta": config.get("local_rope_theta")
            or config.get("rope_parameters", {}).get("sliding_attention", {}).get("rope_theta"),
        },
    }
    with open(os.path.join(root, "lazy.json"), "w") as fh:
        json.dump(lazy, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"prepared {root} (prefix={args.prefix}, pooling={args.pooling})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
