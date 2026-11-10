#!/usr/bin/env python3
"""Fit temperature scaling and a decision threshold, then score on a held-out half.

Input is the `eg-judge bench` report, which carries the raw two-option logits per pair. Fitting and
scoring use different halves so the reported ECE and F1 are out of sample.

Plan section 2.2 asks for temperature scaling before any retraining decision, and for a threshold
that keeps precision >= 0.90. This script reports both: the max-F1 threshold and the
precision-first threshold.

Usage:
    scripts/fit_calibration.py --report .tmp/candidate.json --out .tmp/recipe.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def softmax_true(logits: list[float], temperature: float) -> float:
    z = [float(value) / temperature for value in logits[:2]]
    top = max(z)
    exps = [math.exp(value - top) for value in z]
    return exps[1] / sum(exps)


def metrics(rows, temperature: float, threshold: float) -> dict:
    tp = fp = fn = correct = 0
    nll = 0.0
    confidences = []
    for row in rows:
        probability = softmax_true(row["logits"], temperature)
        predicted = probability >= threshold
        actual = row["label"] == 1
        correct += int(predicted == actual)
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        target = probability if actual else 1.0 - probability
        nll -= math.log(max(target, 1e-12))
        confidences.append((probability, row["label"]))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": correct / max(len(rows), 1),
        "nll": nll / max(len(rows), 1),
        "ece": ece(confidences),
    }


def ece(pairs, bins: int = 15) -> float:
    if not pairs:
        return 0.0
    counts = [0] * bins
    confidence = [0.0] * bins
    accuracy = [0.0] * bins
    for probability, label in pairs:
        index = min(int(probability * bins), bins - 1)
        counts[index] += 1
        confidence[index] += probability
        accuracy[index] += label
    total = len(pairs)
    error = 0.0
    for index in range(bins):
        if not counts[index]:
            continue
        weight = counts[index] / total
        error += weight * abs(confidence[index] / counts[index] - accuracy[index] / counts[index])
    return error


def best_threshold(rows, temperature: float, min_precision: float | None) -> float:
    best, best_score = 0.5, -1.0
    steps = [index / 200 for index in range(1, 200)]
    for candidate in steps:
        result = metrics(rows, temperature, candidate)
        if min_precision is not None and result["precision"] < min_precision:
            continue
        score = result["f1"]
        if score > best_score:
            best, best_score = candidate, score
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out")
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.report) as fh:
        report = json.load(fh)
    rows = [row for row in report["rows"] if len(row.get("logits", [])) >= 2]
    if len(rows) < 20:
        print("not enough rows with logits; rerun eg-judge bench", file=sys.stderr)
        return 1
    split = int(len(rows) * (1.0 - args.holdout_fraction))
    fit_rows, holdout = rows[:split], rows[split:]

    best_temperature, best_nll = 1.0, float("inf")
    for step in range(5, 501):
        temperature = step / 100.0
        value = metrics(fit_rows, temperature, 0.5)["nll"]
        if value < best_nll:
            best_temperature, best_nll = temperature, value

    baseline = metrics(holdout, 1.0, 0.5)
    tuned = metrics(holdout, best_temperature, best_threshold(fit_rows, best_temperature, None))
    precision_first = metrics(
        holdout, best_temperature, best_threshold(fit_rows, best_temperature, 0.90)
    )
    recipe = {
        "temperature": round(best_temperature, 3),
        "threshold": round(best_threshold(fit_rows, best_temperature, None), 3),
        "note": "fitted on %d rows, scored on %d" % (len(fit_rows), len(holdout)),
    }
    output = {
        "temperature": recipe["temperature"],
        "threshold": recipe["threshold"],
        "baseline_holdout": baseline,
        "tuned_holdout": tuned,
        "precision_first_holdout": precision_first,
    }
    print(json.dumps(output, sort_keys=True, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(recipe, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
