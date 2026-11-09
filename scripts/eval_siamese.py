#!/usr/bin/env python3
"""Evaluate a siamese clone detector: shared encoder, per-side pooling, concat head.

Several released clone checkpoints are not `AutoModelForSequenceClassification` artefacts: they
ship a shared encoder plus a `classifier.dense(2*d -> d) + out_proj(d -> 2)` head trained on the
concatenation of the two pooled vectors. This script wires that shape explicitly, verifies the
head tensors exist in the checkpoint, and reports the same metric block as the other evaluators.

Usage:
    .venv/bin/python scripts/eval_siamese.py --model .models/codebert-clone \
        --base roberta --corpus tests/corpus/bigclonebench_400.jsonl
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
    parser.add_argument("--base", choices=["roberta", "t5"], default="roberta")
    parser.add_argument("--pooling", choices=["cls", "mean", "last"], default="cls")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--out")
    args = parser.parse_args()

    os.environ.setdefault("USE_TF", "0")
    import torch
    from transformers import AutoConfig, AutoTokenizer

    model_dir = os.path.abspath(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    state = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            if name.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(path)
            else:
                state = torch.load(path, map_location="cpu", weights_only=False)
            break
    if state is None:
        print("no weights", file=sys.stderr)
        return 1
    if args.base == "roberta" and config.model_type in {"roberta", "bert", "modernbert"}:
        from transformers import AutoModel
        base = AutoModel.from_config(config)
    elif args.base == "t5":
        from transformers import T5EncoderModel
        base = T5EncoderModel(config).float()
        vocab = next((tensor.shape[0] for key, tensor in state.items()
                      if key.endswith("embed_tokens.weight") or key == "shared.weight"), None)
        if vocab is not None and base.get_input_embeddings().weight.shape[0] != vocab:
            base.resize_token_embeddings(vocab)
    else:
        print(f"unsupported base {config.model_type}", file=sys.stderr)
        return 1
    # The released checkpoints wrap the backbone under an `encoder` attribute, so their tensors
    # carry an extra `encoder.` prefix that `AutoModel` does not expect. Strip it and load only
    # the keys the backbone declares; the classification head is wired separately below.
    expected = set(base.state_dict().keys())
    backbone = {}
    for key, tensor in state.items():
        for candidate in (key, key[len("encoder."):] if key.startswith("encoder.") else key):
            if candidate in expected:
                backbone[candidate] = tensor
                break
    missing = expected - set(backbone)
    loaded = base.load_state_dict(backbone, strict=False)
    if len(missing) > 0.1 * len(expected):
        print(f"{args.model}: only {len(backbone)}/{len(expected)} backbone tensors matched",
              file=sys.stderr)
        return 1
    base.eval()

    if "classifier.dense.weight" not in state:
        print(f"{args.model}: no classifier.dense head", file=sys.stderr)
        return 1

    dense = torch.nn.Linear(state["classifier.dense.weight"].shape[1],
                           state["classifier.dense.weight"].shape[0])
    dense.weight.data = state["classifier.dense.weight"]
    dense.bias.data = state["classifier.dense.bias"]
    out_proj = torch.nn.Linear(state["classifier.out_proj.weight"].shape[1], 2)
    out_proj.weight.data = state["classifier.out_proj.weight"]
    out_proj.bias.data = state["classifier.out_proj.bias"]
    dense.eval()
    out_proj.eval()

    def pool(text: str):
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_len)
        with torch.no_grad():
            hidden = base(**batch).last_hidden_state
        if args.pooling == "cls":
            return hidden[:, 0]
        if args.pooling == "last":
            length = int(batch["attention_mask"].sum()) - 1
            return hidden[:, length]
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    with open(args.corpus) as handle:
        pairs = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        pairs = pairs[: args.limit]

    rows = []
    latencies = []
    for index, pair in enumerate(pairs):
        started = time.perf_counter()
        features = torch.cat([pool(pair["a"]), pool(pair["b"])], dim=-1)
        with torch.no_grad():
            logits = out_proj(torch.tanh(dense(features)))[0]
        latencies.append((time.perf_counter() - started) * 1000.0)
        rows.append((int(pair["label"]), float(torch.softmax(logits, -1)[1])))
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
        "reference": f"siamese/{args.base}/{args.pooling}",
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
