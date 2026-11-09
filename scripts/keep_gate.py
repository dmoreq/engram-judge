#!/usr/bin/env python3
"""Apply the keep/reject matrix to a candidate report and its reference baseline.

Matrix (plan section 3):
  F1 parity   keep >= 0.95 x reference, reject < 0.90 x reference
  ECE         keep <= 0.06, reject > 0.12
  p95 latency keep <= 10 ms, reject > 25 ms          (skipped when the report has no latency)
  footprint   keep model <= 220 MB (Tier 1) / <= 80 MB (Tier 2), reject > 400 MB

Usage:
    scripts/keep_gate.py --reference ref.json --candidate cand.json --model-size-mb 161
"""

from __future__ import annotations

import argparse
import json
import sys


def axis(name: str, ok: bool, reject: bool, detail: str) -> dict:
    verdict = "KEEP" if ok else ("REJECT" if reject else "REVIEW")
    return {"axis": name, "verdict": verdict, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model-size-mb", type=float)
    parser.add_argument("--tier", choices=["1", "2"], default="2")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with open(args.reference) as fh:
        reference = json.load(fh)
    with open(args.candidate) as fh:
        candidate = json.load(fh)

    axes = []
    reference_f1 = float(reference["f1"])
    candidate_f1 = float(candidate["f1"])
    ratio = candidate_f1 / reference_f1 if reference_f1 else 0.0
    axes.append(axis("f1-parity", ratio >= 0.95, ratio < 0.90,
                     f"candidate {candidate_f1:.4f} vs reference {reference_f1:.4f} (ratio {ratio:.3f})"))
    candidate_ece = float(candidate["ece"])
    axes.append(axis("calibration", candidate_ece <= 0.06, candidate_ece > 0.12,
                     f"ECE {candidate_ece:.4f}"))
    if "p95_ms" in candidate:
        p95 = float(candidate["p95_ms"])
        axes.append(axis("latency", p95 <= 10.0, p95 > 25.0, f"p95 {p95:.1f} ms"))
    if args.model_size_mb is not None:
        cap = 220.0 if args.tier == "1" else 80.0
        size = args.model_size_mb
        axes.append(axis("footprint", size <= cap, size > 400.0,
                         f"model {size:.1f} MB (tier {args.tier} cap {cap:.0f} MB)"))

    overall = "KEEP"
    if any(item["verdict"] == "REJECT" for item in axes):
        overall = "REJECT"
    elif any(item["verdict"] == "REVIEW" for item in axes):
        overall = "REVIEW"

    if args.json:
        print(json.dumps({"overall": overall, "axes": axes}, sort_keys=True))
    else:
        for item in axes:
            print(f"{item['verdict']:6} {item['axis']:12} {item['detail']}")
        print(f"\noverall: {overall}")
    return 0 if overall != "REJECT" else 1


if __name__ == "__main__":
    sys.exit(main())
