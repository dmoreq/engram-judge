#!/usr/bin/env python3
"""Score a sentence-transformer clone detector (bi-encoder) on a labelled pair corpus.

Encodes each side once, takes the cosine of the mean-pooled, L2-normalised vectors, and sweeps
the similarity threshold. This is the reference arm for the "replace the model" experiment: no
`noul` head, no pair sequence, and embeddings that can be cached per snippet.

Usage:
    .venv/bin/python scripts/eval_owl.py --model .models/owl \
        --corpus tests/corpus/bigclonebench_balanced.jsonl
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
    parser.add_argument("--corpus", default=os.path.join(REPO_ROOT, "tests", "corpus", "bigclonebench_balanced.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--out")
    args = parser.parse_args()

    os.environ.setdefault("USE_TF", "0")
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = os.path.abspath(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()

    with open(args.corpus) as fh:
        pairs = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        pairs = pairs[: args.limit]

    def encode(text: str) -> "torch.Tensor":
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_len)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        if args.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(pooled, dim=-1)[0]

    rows = []
    encode_ms = []
    for index, pair in enumerate(pairs):
        started = time.perf_counter()
        left = encode(pair["a"])
        right = encode(pair["b"])
        encode_ms.append((time.perf_counter() - started) * 1000.0)
        rows.append((int(pair["label"]), float(torch.dot(left, right))))
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
        "reference": "bi-encoder cosine",
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
        "encode_pair_ms_median": round(statistics.median(encode_ms), 1),
        "encode_pair_ms_p95": round(sorted(encode_ms)[int(len(encode_ms) * 0.95) - 1], 1),
        "rows": [{"label": label, "probability": score} for label, score in rows],
    }
    line = json.dumps(report, sort_keys=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(line + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
