#!/usr/bin/env python3
"""Turn a matrix of `eg-judge bench` reports into the optimization benchmark table.

Reads `.tmp/matrix/<variant>_<corpus>.json`, measures each checkpoint's disk size, and writes
`docs/reports/2026-11-09_owl-optimization-benchmark.md`.

Usage:
    .venv/bin/python scripts/run_matrix_benchmark.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANTS = [
    ("owl", "F32", "12"),
    ("owl-fp16", "F16", "12"),
    ("owl-int8", "INT8", "12"),
    ("owl-int4", "INT4", "12"),
    ("owl-8l", "F32", "8"),
    ("owl-6l", "F32", "6"),
    ("owl-int4-8l", "INT4", "8"),
    ("owl-int4-6l", "INT4", "6"),
]


def load(directory: str, variant: str, corpus: str) -> dict | None:
    path = os.path.join(directory, f"{variant}_{corpus}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def peak_rss_mb(variant: str) -> float:
    path = os.path.join(REPO_ROOT, ".tmp", f"time_{variant}.txt")
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                if "maximum resident set size" in line:
                    return round(int(line.split()[0]) / 1048576, 1)
    return 0.0


def disk_mb(variant: str) -> float:
    path = os.path.join(REPO_ROOT, ".models", variant, "model.safetensors")
    if not os.path.exists(path):
        return 0.0
    return round(os.path.getsize(path) / 1e6, 1)


def fmt(row: dict | None, keys: list[str]) -> list[str]:
    if row is None:
        return ["-"] * len(keys)
    return [str(row.get(key, "-")) for key in keys]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=os.path.join(REPO_ROOT, ".tmp", "matrix"))
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "docs", "reports", "2026-11-09_owl-optimization-benchmark.md"))
    args = parser.parse_args()

    rows = []
    for variant, precision, layers in VARIANTS:
        short = load(args.matrix, variant, "pairs")
        long = load(args.matrix, variant, "bigclonebench_400")
        rows.append({
            "variant": variant,
            "precision": precision,
            "layers": layers,
            "disk_mb": disk_mb(variant),
            "rss_mb": peak_rss_mb(variant),
            "short_p50": (short or {}).get("p50_ms", 0.0),
            "short_p95": (short or {}).get("p95_ms", 0.0),
            "short_f1": (short or {}).get("f1", 0.0),
            "long_p50": (long or {}).get("p50_ms", 0.0),
            "long_p95": (long or {}).get("p95_ms", 0.0),
            "long_f1": (long or {}).get("f1", 0.0),
            "throughput": (short or {}).get("throughput_pairs_per_s", 0.0),
        })

    baseline = rows[0]
    lines = []
    lines.append("# Owl bi-encoder: shrink and speedup benchmark\n")
    lines.append("Date: 2026-11-09\n")
    lines.append("Corpora: `tests/corpus/pairs.jsonl` (500 near-miss pairs) and "
                 "`tests/corpus/bigclonebench_400.jsonl` (400 real clones). Machine: arm64 Apple "
                 "Silicon, Metal F16 compute, in-process timing, one iteration per pair.\n")
    lines.append("## Matrix\n")
    lines.append("| variant | precision | layers | disk MB | RSS MB | near-miss p50 / p95 (ms) | "
                 "BCB p50 / p95 (ms) | F1 near-miss | F1 BCB | pairs/s |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | {row['precision']} | {row['layers']} | {row['disk_mb']} | "
            f"{row['rss_mb']} | {row['short_p50']} / {row['short_p95']} | "
            f"{row['long_p50']} / {row['long_p95']} | {row['short_f1']:.4f} | "
            f"{row['long_f1']:.4f} | {row['throughput']} |")
    lines.append("")
    lines.append("## Deltas against the F32 baseline\n")
    lines.append("| variant | disk change | F1 BCB change | p50 BCB change |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in rows[1:]:
        disk = (row["disk_mb"] / baseline["disk_mb"] - 1.0) * 100.0 if baseline["disk_mb"] else 0.0
        f1 = row["long_f1"] - baseline["long_f1"]
        latency = (row["long_p50"] / baseline["long_p50"] - 1.0) * 100.0 if baseline["long_p50"] else 0.0
        lines.append(f"| `{row['variant']}` | {disk:+.1f}% | {f1:+.4f} | {latency:+.1f}% |")
    lines.append("")

    tier0 = os.path.join(REPO_ROOT, ".tmp", "tokendiff_synth.json")
    tier0_long = os.path.join(REPO_ROOT, ".tmp", "tokendiff_bcb.json")
    if os.path.exists(tier0) and os.path.exists(tier0_long):
        with open(tier0) as handle:
            short = json.load(handle)
        with open(tier0_long) as handle:
            long = json.load(handle)
        lines.append("## Tier-0 baseline\n")
        lines.append("| variant | precision | disk MB | near-miss p50 (ms) | BCB p50 (ms) | "
                     "F1 near-miss | F1 BCB |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        lines.append(
            f"| `normalised-token fingerprint` | none | 0 | {short['pair_ms_median']} | "
            f"{long['pair_ms_median']} | {short['best_threshold']['f1']} | "
            f"{long['best_threshold']['f1']} |")
        lines.append("")

    lines.append("## Method\n")
    lines.append("- Quantisation: `scripts/quantize_owl.py` writes F16, per-row INT8 and "
                 "INT4 group-64 (two signed nibbles per byte) checkpoints; the loader "
                 "dequantises at load, so RSS stays at the F16 floor and the gain is disk.")
    lines.append("- Pruning: `scripts/prune_owl.py` keeps the first N layers and rewrites "
                 "`num_hidden_layers`.")
    lines.append("- Kernel path: candle's fused `softmax_last_dim`, zero-bias fused layer norm "
                 "and Metal `sdpa` for unmasked global layers. Copying `laya-candle`'s custom "
                 "Metal kernels would require `unsafe` FFI, which this repository's hygiene gate "
                 "forbids in `src/`.")
    lines.append("- Command per row: `eg-judge bench --model .models/<variant> --corpus "
                 "tests/corpus/<corpus>.jsonl --iterations 1 --device metal --dtype f16`.")
    lines.append("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    for row in rows:
        print(f"{row['variant']:14} disk={row['disk_mb']:6.1f}MB bcb_p50={row['long_p50']:7.1f}ms "
              f"f1_bcb={row['long_f1']:.4f} f1_near={row['short_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
