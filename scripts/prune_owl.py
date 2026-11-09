#!/usr/bin/env python3
"""Write layer-pruned variants of a ModernBERT checkpoint (keeps the first N layers).

Disk shrinks with the layer count because dropped layers are not written. `encoder/config.json`
is rewritten with `num_hidden_layers = N` so the loader allocates only that many blocks.

Usage:
    .venv/bin/python scripts/prune_owl.py --model .models/owl --layers 8,6
    .venv/bin/python scripts/prune_owl.py --model .models/owl-int4 --layers 8,6 --suffix int4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from safetensors.torch import load_file, save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def keep(name: str, layers: int) -> bool:
    if not name.startswith("layers."):
        return True
    return int(name.split(".")[1]) < layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.join(REPO_ROOT, ".models", "owl"))
    parser.add_argument("--layers", default="8,6")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    root = os.path.abspath(args.model)
    state = load_file(os.path.join(root, "model.safetensors"))
    for count in (int(value) for value in args.layers.split(",")):
        out = f"{root}-{count}l" + (f"-{args.suffix}" if args.suffix else "")
        tensors = {name: tensor for name, tensor in state.items() if keep(name, count)}
        os.makedirs(os.path.join(out, "encoder"), exist_ok=True)
        os.makedirs(os.path.join(out, "tokenizer"), exist_ok=True)
        save_file(tensors, os.path.join(out, "model.safetensors"))
        shutil.copy2(os.path.join(root, "tokenizer", "tokenizer.json"), os.path.join(out, "tokenizer", "tokenizer.json"))
        with open(os.path.join(root, "encoder", "config.json")) as handle:
            config = json.load(handle)
        config["num_hidden_layers"] = count
        if isinstance(config.get("layer_types"), list):
            config["layer_types"] = config["layer_types"][:count]
        with open(os.path.join(out, "encoder", "config.json"), "w") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
        with open(os.path.join(root, "lazy.json")) as handle:
            lazy = json.load(handle)
        with open(os.path.join(out, "lazy.json"), "w") as handle:
            json.dump(lazy, handle, indent=2, sort_keys=True)
            handle.write("\n")
        size = os.path.getsize(os.path.join(out, "model.safetensors"))
        print(f"{count:2} layers -> {out}  {size / 1e6:6.1f} MB  ({len(tensors)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
