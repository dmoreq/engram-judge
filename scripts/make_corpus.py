#!/usr/bin/env python3
"""Generate a deterministic labelled pair corpus for the keep/reject gates.

Positive pairs are semantics-preserving rewrites (identifier renames, comment noise, reordering of
independent top-level items). Negative pairs carry exactly one semantic mutation (`<` to `<=`,
`+` to `-`, `==` to `!=`, `&&` to `||`, an incremented constant). Both kinds stay textually close,
which is the gray zone the judge is meant to arbitrate.

Usage:
    .venv/bin/python scripts/make_corpus.py --count 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_SNIPPETS = [
    (
        "rust-sum",
        ["xs", "acc", "x", "total"],
        [("+=", "-=")],
        """fn sum_values(xs: &[u32]) -> u32 {
    let mut acc = 0;
    for x in xs {
        acc += *x;
    }
    acc
}""",
    ),
    (
        "rust-clamp",
        ["value", "low", "high", "v", "lo", "hi"],
        [("< low", "<= low")],
        """fn clamp_range(value: i64, low: i64, high: i64) -> i64 {
    if value < low {
        return low;
    }
    if value > high {
        return high;
    }
    value
}""",
    ),
    (
        "rust-first-even",
        ["values", "candidate", "found"],
        [("% 2 == 0", "% 2 != 0")],
        """fn first_even(values: &[i64]) -> Option<i64> {
    let mut found = None;
    for candidate in values {
        if candidate % 2 == 0 {
            found = Some(*candidate);
            break;
        }
    }
    found
}""",
    ),
    (
        "rust-normalize",
        ["text", "parts", "part", "cleaned"],
        [("push('-')", "push('_')")],
        """fn normalize(text: &str) -> String {
    let mut cleaned = String::new();
    for part in text.split(' ') {
        if !part.is_empty() {
            cleaned.push_str(part);
            cleaned.push('-');
        }
    }
    cleaned
}""",
    ),
    (
        "python-chunk",
        ["items", "size", "chunks", "index", "out"],
        [("index + size", "index + size - 1")],
        """def chunk(items, size):
    out = []
    for index in range(0, len(items), size):
        out.append(items[index:index + size])
    return out""",
    ),
    (
        "python-names",
        ["rows", "row", "names", "out"],
        [('.strip()', '.lstrip()')],
        """def names(rows):
    out = []
    for row in rows:
        out.append(row["name"].strip())
    return out""",
    ),
    (
        "python-sha",
        ["blob", "digest", "data"],
        [("sha256", "sha1")],
        """def sha(blob):
    digest = hashlib.sha256(blob).hexdigest()
    return digest""",
    ),
    (
        "python-bucket",
        ["value", "limit", "bounds", "total", "fallback"],
        [("value > fallback", "value >= fallback")],
        """def bucket(value, limit):
    total = 0
    bounds = [0, limit]
    for fallback in bounds:
        if value > fallback:
            total = fallback
    return total""",
    ),
    (
        "js-debounce",
        ["handle", "wait", "timer", "callback"],
        [("clearTimeout", "clearInterval")],
        """function debounce(callback, wait) {
  let timer = null;
  return function () {
    clearTimeout(timer);
    timer = setTimeout(callback, wait);
  };
}""",
    ),
    (
        "js-flatten",
        ["rows", "row", "flat", "out"],
        [("...row", "...[]")],
        """function flatten(rows) {
  const flat = [];
  for (const row of rows) {
    flat.push(...row);
  }
  return flat;
}""",
    ),
    (
        "js-merge",
        ["left", "right", "merged", "key"],
        [("merged[key] = left[key];", "merged[key] = right[key];")],
        """function merge(left, right) {
  const merged = {};
  for (const key of Object.keys(left)) {
    merged[key] = left[key];
  }
  for (const key of Object.keys(right)) {
    merged[key] = right[key];
  }
  return merged;
}""",
    ),
    (
        "js-percent",
        ["part", "whole", "ratio", "result"],
        [("ratio * 100", "ratio * 1000")],
        """function percent(part, whole) {
  let ratio = part / whole;
  const result = ratio * 100;
  return result;
}""",
    ),
]

MUTATIONS = [
    ("<", "<="),
    ("<=", "<"),
    ("+", "-"),
    ("-", "+"),
    ("==", "!="),
    ("!=", "=="),
    ("&&", "||"),
    (">", ">="),
]

SUFFIXES = ["_a", "_b", "_alt", "_v2", "_impl", "_copy"]


def rename(code: str, names: list[str], rng: random.Random, heavy: bool) -> str:
    shuffled = names[:]
    rng.shuffle(shuffled)
    for index, name in enumerate(names):
        if not heavy and index % 2 == 1:
            continue
        replacement = shuffled[index % len(shuffled)] + rng.choice(SUFFIXES)
        code = re.sub(rf"\b{re.escape(name)}\b", replacement, code)
    return code


def decorate(code: str, rng: random.Random) -> str:
    """Add semantics-free noise: blank lines, comments, and reordered top-level items."""
    lines = code.splitlines()
    if rng.random() < 0.5 and len(lines) > 3:
        split = rng.randint(1, len(lines) - 1)
        lines.insert(split, "")
    if rng.random() < 0.5:
        lines.insert(0, "// reviewed")
    if rng.random() < 0.3:
        lines.append("")
    return "\n".join(lines)


def mutate(code: str, mutations: list[tuple[str, str]]) -> str:
    """Apply exactly one semantic mutation that is guaranteed to be present."""
    old, new = mutations[0]
    if old not in code:
        raise ValueError(f"mutation {old!r} not present in snippet")
    return code.replace(old, new, 1)


def build(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    while len(rows) < count:
        snippet_id, names, mutations, code = rng.choice(BASE_SNIPPETS)
        roll = rng.random()
        label = 0 if roll < 0.45 else 1
        heavy = label == 1 and roll > 0.35
        left = decorate(rename(code, names, rng, heavy=False), rng)
        right_source = code if label == 1 else mutate(code, mutations)
        right = decorate(rename(right_source, names, rng, heavy=heavy), rng)
        rows.append({
            "a": left,
            "b": right,
            "label": label,
            "kind": "duplicate",
            "source": snippet_id,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "tests", "corpus", "pairs.jsonl"))
    args = parser.parse_args()
    rows = build(args.count, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    positives = sum(row["label"] for row in rows)
    print(f"wrote {len(rows)} pairs ({positives} duplicate, {len(rows) - positives} distinct) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
