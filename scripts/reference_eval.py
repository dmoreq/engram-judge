#!/usr/bin/env python3
"""Score a corpus with the PyTorch reference checkpoint (the Laya-FP16 baseline).

Prints the same metric block `eg-laya bench` prints, so `scripts/keep_gate.py` can compare the
two numerically without translating schemas.

Usage:
    .venv/bin/python scripts/reference_eval.py --model .models/laya-multilingual \
        --corpus tests/corpus/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QUESTION_TEXT = {
    "duplicate": {
        "ins": "Both code snippets are pasted below. Do they implement the same algorithm, "
               "ignoring identifier names, formatting and comment changes?",
        "false": "the two snippets compute different results",
        "true": "the two snippets are behaviourally equivalent",
        "labels": ("snippet A", "snippet B"),
    },
    "supersede": {
        "ins": "Both notes describe the same project fact. Does the new note replace the old one, "
               "that is, does it make the old note obsolete?",
        "false": "the notes describe different facts and both stay valid",
        "true": "the new note supersedes the old note",
        "labels": ("old note", "new note"),
    },
}


def pair_state(kind: str, first: str, second: str) -> str:
    label_a, label_b = QUESTION_TEXT[kind]["labels"]
    return "%s:\n%s\n%s:\n%s\n" % (label_a, first.rstrip(), label_b, second.rstrip())


def classification(rows, threshold):
    tp = fp = fn = correct = 0
    for label, probability in rows:
        predicted = probability >= threshold
        actual = label == 1
        correct += int(predicted == actual)
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif actual:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, correct / max(len(rows), 1)


def ece(rows, bins=15):
    if not rows:
        return 0.0
    counts = [0] * bins
    confidence = [0.0] * bins
    accuracy = [0.0] * bins
    for label, probability in rows:
        index = min(int(probability * bins), bins - 1)
        counts[index] += 1
        confidence[index] += probability
        accuracy[index] += label
    total = len(rows)
    error = 0.0
    for bin_index in range(bins):
        if not counts[bin_index]:
            continue
        weight = counts[bin_index] / total
        error += weight * abs(confidence[bin_index] / counts[bin_index] - accuracy[bin_index] / counts[bin_index])
    return round(error, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default=os.path.join(REPO_ROOT, "tests", "corpus", "pairs.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out")
    args = parser.parse_args()

    os.environ.setdefault("USE_TF", "0")
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import build_model, build_sequence

    model_dir = os.path.abspath(args.model)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(model_dir, "lazy.json")) as fh:
        lazy = json.load(fh)
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"), pretrained=False)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    temperature = float(lazy["temperature"][2])

    rows = []
    with open(args.corpus) as fh:
        pairs = [json.loads(line) for line in fh if line.strip()]
    for index, pair in enumerate(pairs):
        kind = pair.get("kind", "duplicate")
        text = QUESTION_TEXT[kind]
        question = {"t": "noul", "ins": text["ins"],
                    "crit": {"false": text["false"], "true": text["true"]}}
        ids, markers = build_sequence(tokenizer, pair_state(kind, pair["a"], pair["b"]), question,
                                      max_len=1024, head_max_len=192)
        with torch.no_grad():
            logits, _ = model(
                input_ids=torch.tensor([ids], dtype=torch.long),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                marker_pos=torch.tensor([markers], dtype=torch.long),
                marker_mask=torch.ones((1, len(markers)), dtype=torch.bool),
                qtype=torch.tensor([2], dtype=torch.long),
            )
        z = logits[0].double().numpy() / temperature
        p = np.exp(z - z.max())
        p = p / p.sum()
        rows.append((int(pair["label"]), float(round(p[1], 3))))
        if (index + 1) % 50 == 0:
            print(f"  scored {index + 1}/{len(pairs)}", flush=True)

    precision, recall, f1, accuracy = classification(rows, args.threshold)
    report = {
        "reference": "laya-fp16",
        "cases": len(rows),
        "threshold": args.threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "ece": ece(rows),
        "rows": [{"label": label, "probability": probability} for label, probability in rows],
    }
    line = json.dumps(report, sort_keys=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
