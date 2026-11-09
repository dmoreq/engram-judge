#!/usr/bin/env python3
"""Classic, non-neural baseline: normalised-token similarity with no model at all.

Strips comments, maps identifiers to `ID` and literals to `LIT`, then scores a pair with the Dice
ratio of the longest common subsequence (`difflib`) and with 3-gram Jaccard. This is the tier-0
candidate: if it matches a cross-encoder on near-miss pairs at microsecond cost, the neural arm is
only needed for the residual.

Usage:
    .venv/bin/python scripts/eval_token_diff.py --corpus tests/corpus/pairs.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import statistics
import sys
import time

KEYWORDS = set("""
if else for while return fn def let mut const var function class struct impl trait pub use mod
match case break continue new true false null none some ok err and or not in is as async await
try catch throw import from self this super static void int i32 u32 u64 f64 str bool string number
""".split())

COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|#[^\n]*", re.S)
STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\sA-Za-z0-9_]")


def normalise(code: str) -> list[str]:
    code = COMMENT.sub(" ", code)
    code = STRING.sub(" LIT ", code)
    code = NUMBER.sub(" LIT ", code)
    tokens = []
    for token in TOKEN.findall(code):
        if token in KEYWORDS:
            tokens.append(token)
        elif re.match(r"[A-Za-z_]", token):
            tokens.append("ID")
        else:
            tokens.append(token)
    return tokens


def dice(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    return matcher.ratio()


def jaccard(left: list[str], right: list[str], n: int = 3) -> float:
    if len(left) < n or len(right) < n:
        return dice(left, right)
    a = {tuple(left[i:i + n]) for i in range(len(left) - n + 1)}
    b = {tuple(right[i:i + n]) for i in range(len(right) - n + 1)}
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def metrics(rows, threshold):
    tp = fp = fn = correct = 0
    for label, score in rows:
        predicted = score >= threshold
        actual = label == 1
        correct += int(predicted == actual)
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, correct / max(len(rows), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()

    with open(args.corpus) as handle:
        pairs = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        pairs = pairs[: args.limit]

    rows = []
    latencies = []
    for pair in pairs:
        started = time.perf_counter()
        left = normalise(pair["a"])
        right = normalise(pair["b"])
        score = max(dice(left, right), jaccard(left, right))
        latencies.append((time.perf_counter() - started) * 1000.0)
        rows.append((int(pair["label"]), score))

    best = (0.0, 0.5, 0.0, 0.0, 0.0)
    for step in range(1, 100):
        threshold = step / 100
        precision, recall, f1, accuracy = metrics(rows, threshold)
        if f1 > best[0]:
            best = (f1, threshold, precision, recall, accuracy)
    precision, recall, f1, accuracy = metrics(rows, 0.5)
    report = {
        "model": "token-diff",
        "reference": "normalised tokens",
        "cases": len(rows),
        "threshold_0.5": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        },
        "best_threshold": {
            "threshold": best[1],
            "precision": round(best[2], 4),
            "recall": round(best[3], 4),
            "f1": round(best[0], 4),
            "accuracy": round(best[4], 4),
        },
        "pair_ms_median": round(statistics.median(latencies), 3),
        "pair_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3),
        "rows": [{"label": label, "probability": score} for label, score in rows],
    }
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(json.dumps(report, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
