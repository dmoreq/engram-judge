# engram-judge

Offline semantic judge for `engram`'s two gray-zone questions:

- **Anti-reinvention**: are these two snippets the same algorithm, after renames and reflow?
- **Memory bloat**: does this new note supersede the old one?

The binary, `eg-judge`, runs a local model on Metal or the CPU, with no network, no daemon and
no Python at runtime. One JSON line per call.

## Shipping artifact

`.models/owl-int4`: a 12-layer ModernBERT bi-encoder quantised to INT4 group-64.

| Property | Value |
| --- | --- |
| Disk | 85.4 MB |
| BigCloneBench F1 | 0.9592 (F32 baseline 0.9618) |
| Embedding cosine to F32 | 0.9569 |
| p50 | 40 ms short pair / 133 ms at BigCloneBench length |
| Peak RSS | ~1.6 GB (Metal unified, load-time dequant) |

The three reports under `docs/reports/` record how it was selected; every other checkpoint that
was benchmarked has been removed and is reproducible from `scripts/`.

## Quick start

```sh
export EG_JUDGE_MODEL="$(pwd)/.models/owl-int4"

cargo run --release -- similarity \
    --text-a "fn total(x: &[i32]) -> i32 { x.iter().sum() }" \
    --text-b "fn sum_all(arr: &[i32]) -> i32 { let mut s = 0; for v in arr { s += v; } s }"
```

```json
{"decision":"duplicate","similarity":0.868,"threshold":0.5,"latency_ms":72.3,"device":"metal"}
```

`similarity` takes `--text-a`/`--text-b`, their `-file` forms, or `{"text_a": ..., "text_b": ...}`
on stdin. `embed --text ...` prints the pooled vector for a cache index. `bench --corpus FILE`
runs a labelled JSONL corpus and reports F1, ECE and latency percentiles.

## Contract

| item | value |
| --- | --- |
| stdout | exactly one JSON line |
| exit `0` | verdict produced |
| exit `1` | malformed input (empty text, bad JSON, bad recipe or device) |
| exit `2` | invalid CLI usage (clap) |
| exit `6` | model, tokenizer, I/O or device failure |
| network | none |
| writes | none, except `bench --out FILE` |

`--device auto|cpu|metal` selects the backend; a Metal failure warns on stderr and falls back to
CPU. `--dtype f32|f16` picks the weight dtype. `--max-len` caps tokens per side (default 512).

## Other checkpoints

The same binary still drives Laya `noul` checkpoints through `judge-duplicate` and
`judge-supersede`: a 12- or 22-layer encoder plus a two-layer decision head. Those measure
F1 0.6755 on BigCloneBench and 0.7084 on the near-miss corpus, so they are supported but not
shipped. Point `EG_JUDGE_MODEL` at an exported checkpoint to use them.

## Measured results

| arm | BigCloneBench F1 | near-miss F1 | p50 BCB | p50 short | weights |
| --- | ---: | ---: | ---: | ---: | ---: |
| `owl-int4` bi-encoder | 0.9592 | 0.7139 | 133.0 ms | 39.9 ms | 85.4 MB |
| `owl` F32 bi-encoder | 0.9618 | 0.7130 | 140.0 ms | 41.6 ms | 606.7 MB |
| Laya `noul` 22L | 0.6755 | 0.6780 | 544.9 ms | 79.7 ms | 644 MB |
| normalised-token fingerprint | 0.6813 | 0.8203 | 0.90 ms | 0.03 ms | 0 MB |

Quantisation keeps BigCloneBench F1 to -0.0026. Layer pruning does not: 8 and 6 layers both drop
F1 to 0.6755, the all-positive baseline, so no pruned variant ships.

## Verification

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
bash tests/scripts/allowlist_check.sh Cargo.lock
bash tests/scripts/write_path_check.sh
bash tests/scripts/hygiene_check.sh
```

Numerical parity against the PyTorch reference needs a checkpoint:

```sh
EG_JUDGE_MODEL=.models/owl cargo test --release --test parity -- --nocapture
```

The parity suite rebuilds the prompt with the Rust tokenizer, runs the Rust encoder and head, and
requires `|delta P| < 2e-3`; the bi-encoder path matches the reference embedding to 2e-7.

## Layout

```
src/
  model/encoder.rs   ModernBERT forward: embeddings, RoPE, GeGLU, global + sliding attention
  model/head.rs      two-layer pre-norm decision head and per-marker scorer
  model/norm.rs      fused layer norm for bias-free checkpoints
  model/mod.rs       checkpoint loader, INT8/INT4 dequant, forward, embedding
  tokenizer.rs       tokenizer wrapper, Laya prompt layout, BERT-style encode
  judge.rs           Laya questions, state layout, verdict
  bi_encoder.rs      pooled embedding, cosine similarity
  bench.rs           latency percentiles, throughput, RSS, F1, ECE
  corpus.rs          labelled pair corpus
  run.rs, cli.rs     subcommands, stdin/file inputs, exit codes
scripts/
  quantize_owl.py    FP16 / INT8 / INT4 variants
  eval_owl.py        bi-encoder reference arm
  eval_token_diff.py deterministic tier-0 baseline
  fetch_bigclonebench.py  reproducible BigCloneBench slice
  publish_model.sh   upload the shipping checkpoint to the Hub
tests/
  parity.rs          token and probability parity against the reference
  cli.rs             exit codes and command surface
  fixtures/          golden ids, logits and probabilities
  corpus/            500 near-miss pairs plus a tracked BigCloneBench slice
```

## Limits

- The bi-encoder loses single-token differences: near-miss F1 is 0.71 against 0.96 for real clones.
  `engram`'s deterministic detectors and its code graph resolve most of that band.
- Batch size is one; the encoder skips padding masks.
- INT4 shrinks the file, not the arithmetic: weights are dequantised at load, so RSS stays at the
  F16 floor.
