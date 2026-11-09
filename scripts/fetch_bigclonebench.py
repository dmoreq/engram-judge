#!/usr/bin/env python3
"""Fetch a reproducible BigCloneBench slice (CodeXGLUE test split) as labelled pairs.

Writes `tests/corpus/bigclonebench_sample.jsonl` (4000 pairs, natural imbalance), a balanced subset,
and a deterministic 400-pair slice used by the model-comparison table. Only the 400-pair slice is
tracked; the larger files are gitignored.

Usage:
    .venv/bin/python scripts/fetch_bigclonebench.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(REPO_ROOT, "tests", "corpus")
BASE = ("https://datasets-server.huggingface.co/rows?dataset="
        "google%2Fcode_x_glue_cc_clone_detection_big_clone_bench&config=default&split=test")
TOTAL = 4000
PAGE = 100


def fetch(offset: int) -> list[dict]:
    url = f"{BASE}&offset={offset}&length={PAGE}"
    request = urllib.request.Request(url, headers={"User-Agent": "engram-laya-research"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return [row["row"] for row in json.load(response)["rows"]]


def write(path: str, rows: list[dict]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps({
                "a": row["func1"],
                "b": row["func2"],
                "label": int(bool(row["label"])),
                "kind": "duplicate",
                "source": "bigclonebench",
            }, sort_keys=True) + "\n")


def main() -> int:
    rows: list[dict] = []
    for offset in range(0, TOTAL, PAGE):
        rows.extend(fetch(offset))
        print(f"  fetched {len(rows)}/{TOTAL}", flush=True)
        time.sleep(0.2)
    os.makedirs(CORPUS, exist_ok=True)
    write(os.path.join(CORPUS, "bigclonebench_sample.jsonl"), rows)

    positives = [row for row in rows if bool(row["label"])]
    negatives = [row for row in rows if not bool(row["label"])]
    size = min(len(positives), len(negatives))
    balanced = positives[:size] + negatives[:size]
    write(os.path.join(CORPUS, "bigclonebench_balanced.jsonl"), balanced)

    random.Random(0).shuffle(balanced)
    write(os.path.join(CORPUS, "bigclonebench_400.jsonl"), balanced[:400])
    print(f"wrote {len(rows)} sample, {len(balanced)} balanced, 400 tracked pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
