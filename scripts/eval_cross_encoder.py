#!/usr/bin/env python3
"""Score a cross-encoder clone detector on a labelled pair corpus.

Feeds `(a, b)` as a sentence pair to `AutoModelForSequenceClassification`, takes
`softmax(logits)[1]` as P(clone), sweeps the threshold and reports latency. Models without a
two-class head are rejected instead of silently scoring a random projection.

Usage:
    .venv/bin/python scripts/eval_cross_encoder.py --model .models/codet5-clone \
        --corpus tests/corpus/bigclonebench_400.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--out")
    args = parser.parse_args()

    os.environ.setdefault("USE_TF", "0")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = os.path.abspath(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    labels = getattr(model.config, "num_labels", 0)
    if labels != 2:
        print(f"{args.model}: {labels} labels, not a binary cross-encoder", file=sys.stderr)
        return 1

    with open(args.corpus) as handle:
        pairs = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        pairs = pairs[: args.limit]

    rows = []
    latencies = []
    for index, pair in enumerate(pairs):
        batch = tokenizer(pair["a"], pair["b"], return_tensors="pt",
                          truncation=True, max_length=args.max_len)
        started = time.perf_counter()
        with torch.no_grad():
            logits = model(**batch).logits[0]
        latencies.append((time.perf_counter() - started) * 1000.0)
        probability = float(torch.softmax(logits, -1)[1])
        rows.append((int(pair["label"]), probability))
        if (index + 1) % 50 == 0:
            print(f"  scored {index + 1}/{len(pairs)}", flush=True)

    best = (0.0, 0.5, 0.0, 0.0, 0.0)
    for step in range(1, 100):
        threshold = step / 100
        precision, recall, f1, accuracy = metrics(rows, threshold)
        if f1 > best[0]:
            best = (f1, threshold, precision, recall, accuracy)
    precision, recall, f1, accuracy = metrics(rows, 0.5)
    report = {
        "model": os.path.basename(model_dir),
        "reference": "cross-encoder",
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
        "pair_ms_median": round(statistics.median(latencies), 1),
        "pair_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1),
        "rows": [{"label": label, "probability": score} for label, score in rows],
    }
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(json.dumps(report, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
