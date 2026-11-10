#!/usr/bin/env python3
"""Write FP16, INT8 and INT4-group64 variants of a sentence-encoder checkpoint.

Layout produced (safetensors):

- FP16: every tensor cast to F16.
- INT8: `name.q8` (I8) + `name.scale` (F32, one per output row); norms and 1-D tensors stay F16.
- INT4: `name.q4` (U8, two signed nibbles per byte, low nibble first) + `name.scale`
  (F32, one per group of 64 columns); norms and 1-D tensors stay F16.

`eg-judge` dequantises at load time, so disk drops while resident memory stays at the F16 floor.

Usage:
    .venv/bin/python scripts/quantize_owl.py --model .models/owl --formats fp16,int8,int4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUP = 64


def quantise_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = tensor.shape[0]
    flat = tensor.reshape(rows, -1).float()
    scale = flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    quantised = (flat / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return quantised, scale


def quantise_int4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = tensor.shape[0]
    cols = tensor.shape[1]
    if cols % GROUP != 0:
        raise ValueError(f"columns {cols} not divisible by group {GROUP}")
    groups = cols // GROUP
    grouped = tensor.float().reshape(rows, groups, GROUP)
    scale = grouped.abs().amax(dim=2).clamp(min=1e-8) / 7.0
    quantised = (grouped / scale.unsqueeze(2)).round().clamp(-7, 7)
    packed = quantised.reshape(rows, groups * GROUP).to(torch.int16) & 0x0F
    packed = (packed[:, 0::2] | (packed[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale.reshape(rows, groups)


def is_quantisable(name: str, tensor: torch.Tensor) -> bool:
    # Embeddings are quantised too: at 38.6 M parameters they dominate the 80 MB tier cap.
    return tensor.dim() == 2 and tensor.shape[1] % GROUP == 0


def convert(source: torch.Tensor, fmt: str, name: str) -> dict[str, torch.Tensor]:
    if not is_quantisable(name, source):
        return {name: source.to(torch.float16)}
    if fmt == "fp16":
        return {name: source.to(torch.float16)}
    if fmt == "int8":
        q, s = quantise_int8(source)
        # safetensors readers in candle do not decode I8; store the same bytes as U8.
        return {f"{name}.q8": q.view(torch.uint8), f"{name}.scale": s}
    q, s = quantise_int4(source)
    return {f"{name}.q4": q, f"{name}.scale": s}


def write_variant(root: str, out: str, fmt: str, state: dict[str, torch.Tensor]) -> int:
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        tensors.update(convert(tensor, fmt, name))
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    with open(os.path.join(root, "config.json")) as handle:
        config = json.load(handle)
    os.makedirs(os.path.join(out, "encoder"), exist_ok=True)
    shutil.copy2(os.path.join(root, "encoder", "config.json"), os.path.join(out, "encoder", "config.json"))
    os.makedirs(os.path.join(out, "tokenizer"), exist_ok=True)
    shutil.copy2(os.path.join(root, "tokenizer", "tokenizer.json"), os.path.join(out, "tokenizer", "tokenizer.json"))
    with open(os.path.join(root, "lazy.json")) as handle:
        lazy = json.load(handle)
    lazy["weight_format"] = fmt
    with open(os.path.join(out, "lazy.json"), "w") as handle:
        json.dump(lazy, handle, indent=2, sort_keys=True)
        handle.write("\n")
    size = os.path.getsize(os.path.join(out, "model.safetensors"))
    print(f"{fmt:5} -> {out}  {size / 1e6:6.1f} MB")
    return size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.join(REPO_ROOT, ".models", "owl"))
    parser.add_argument("--formats", default="fp16,int8,int4")
    args = parser.parse_args()

    root = os.path.abspath(args.model)
    state = load_file(os.path.join(root, "model.safetensors"))
    total = sum(value.numel() for value in state.values())
    print(f"source {root}: {len(state)} tensors, {total / 1e6:.1f}M parameters")
    for fmt in args.formats.split(","):
        write_variant(root, f"{root}-{fmt}", fmt.strip(), state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
