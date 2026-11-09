# engram-laya

Native Rust judge for the two gray-zone decisions `engram` cannot settle with static heuristics:

- **Anti-reinvention**: are these two snippets the same algorithm, after renames and reflow?
- **Memory bloat**: does this new note supersede the old one?

The binary, `eg-laya`, runs the proposition (`noul`) head of a Laya checkpoint on Apple Metal or
the CPU, offline, with no Python at runtime. It answers one JSON line per call and exits.

## What the checkpoint actually is

The model is **not** a pairwise cross-encoder with a binary classifier. Laya is an
option-marker scorer:

```
[CLS] "noul question: <instructions>" [SEP] [MASK] false-option [MASK] true-option [SEP] <state> [SEP]
```

The encoder reads that whole sequence once. A two-layer `nn.TransformerEncoder` head runs over
the sequence, `scorer` turns the two `[MASK]` positions into logits, and a softmax over the two
options gives `P(true)`. `type_emb[noul]` is added to every token before the head.

| checkpoint | backbone | hidden / layers | vocab | FP16 weights |
| --- | --- | --- | --- | --- |
| `convaiinnovations/laya` (English) | ModernBERT-large | 1024 / 28 | 50,368 | 842.6 MB |
| `convaiinnovations/laya-multilingual` | mmBERT-base | 768 / 22 | 256,000 | 643.8 MB |

Both heads are identical: `head.layers.0..1`, `scorer.{0,1,3}`, `type_emb`, `temperature[3]`.
`act_head` is dropped: it only feeds the act/escalate decision this tool never reports.

## Quick start

```sh
# 1. Reference environment (offline tooling only, never shipped)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "torch>=2.6" "transformers==5.22.2" \
    "safetensors>=0.4" "numpy>=1.26" "huggingface_hub>=0.20"
uv pip install --python .venv/bin/python --no-deps laya==0.4.30

# 2. Download + export the checkpoint and golden fixtures
.venv/bin/python scripts/export_laya.py --checkpoint laya-multilingual --fixtures 9

# 3. Build
cargo build --release

# 4. Judge
target/release/eg-laya judge-duplicate \
    --code-a "fn a(x: u32) -> u32 { x + 1 }" \
    --code-b "fn b(y: u32) -> u32 { y + 1 }" \
    --model .models/laya-multilingual
```

```json
{"decision":"distinct","probability":0.51,"calibrated":false,"latency_ms":7.3,"device":"metal"}
```

`--code-a`/`--code-b` accept inline text; `--code-a-file`/`--code-b-file` read from disk; with
neither, the command accepts `{"code_a": "...", "code_b": "..."}` on stdin. `judge-supersede`
takes `--old-note`/`--new-note` or `{"old_note": ..., "new_note": ...}`.

## Contract

| item | value |
| --- | --- |
| stdout | exactly one JSON line |
| fields | `decision`, `probability`, `calibrated`, `latency_ms`, `device` (`logits` only with `--with-logits`) |
| exit `0` | verdict produced |
| exit `1` | malformed input (empty snippet, bad JSON, bad recipe, bad device name) |
| exit `2` | invalid CLI usage (clap) |
| exit `6` | model, tokenizer, I/O or device failure |
| network | none: no socket, no daemon, no Hub call |
| writes | none: the runtime only reads the model directory |

`--device auto|cpu|metal` selects the backend; a Metal failure warns on stderr and falls back to
CPU rather than aborting. `--dtype f32|f16` picks the weight dtype. `--device-info` prints the
resolved device, checkpoint, hidden size, temperature and threshold to stderr. Every verdict
also carries `"device"`, so a silent CPU fallback is visible in the output.

## Calibration

`--recipe FILE` loads `{"temperature": 0.83, "threshold": 0.62}`; `--threshold` overrides the
threshold. `calibrated` is `true` only when a recipe actually supplies a value. Without one the
checkpoint ships `temperature = 1.0` and `threshold = 0.5`, which Laya documents as
over-confident.

## Gates

```sh
# Reference baseline (PyTorch, FP16)
.venv/bin/python scripts/reference_eval.py --model .models/laya-multilingual \
    --corpus tests/corpus/pairs.jsonl --out .tmp/reference.json

# Candidate (Rust)
target/release/eg-laya bench --model .models/laya-multilingual \
    --corpus tests/corpus/pairs.jsonl --iterations 10 > .tmp/candidate.json

# Keep / reject matrix
.venv/bin/python scripts/keep_gate.py --reference .tmp/reference.json \
    --candidate .tmp/candidate.json --model-size-mb 161 --tier 2
```

`scripts/make_corpus.py` regenerates the 500-pair corpus deterministically: positives differ only
by identifier names, comments and whitespace; negatives carry exactly one semantic mutation
(`<` to `<=`, `+` to `-`, `==` to `!=`, `&&` to `||`, and so on).

## Measured results

Machine: arm64 Apple Silicon, macOS, candle 0.11.1 with the Metal backend, mmBERT-base
multilingual checkpoint, in-process latency.

**Numerical parity** (9 golden fixtures, reference = PyTorch 2.15 + transformers 5.22 + laya 0.4.30):
token ids exact, marker positions exact, worst `|delta P(true)| = 1e-6`. The Rust encoder, head,
scorer, RoPE convention, sliding-window band and prompt layout reproduce the reference.

**Gate** on 500 labelled pairs (277 duplicate, 223 distinct; negatives differ by exactly one
operator or literal):

| axis | shipped defaults (T=1, thr=0.5) | after temperature scaling | gate |
| --- | --- | --- | --- |
| F1 parity | 0.7084 vs reference 0.7075 (1.001x) | 0.7424 (1.049x) | KEEP at >= 0.95x |
| ECE | 0.2202 | 0.1047 | REJECT at > 0.12, REVIEW after tuning, KEEP needs <= 0.06 |
| p95 latency | 123.2 ms (174.7 before kernel fusion) | 123.2 ms | REJECT at > 25 ms |
| footprint | 644 MB | 644 MB | REJECT at > 400 MB |

Overall: **REJECT**. Fitted recipe (temperature 5.0, threshold 0.525) is written by
`scripts/fit_calibration.py`, which fits on half the corpus and scores the other half.

### Checkpoint comparison

Same 200 labelled pairs, same machine, Metal F16, in-process. Reference is the same checkpoint in
PyTorch FP16.

| checkpoint | weights | params | p50 | p95 | F1 | ECE | worst parity delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `laya-multilingual` (mmBERT-base) | 644 MB | 322 M | 85.9 ms | 123.2 ms | 0.678 | 0.273 | 1e-6 |
| `laya` (ModernBERT-large) | 843 MB | 421 M | 181.0 ms | 268.5 ms | 0.703 | 0.190 | 2e-6 |

The multilingual row is the current engine after two kernel fusions (fused `softmax_last_dim`,
fused zero-bias layer norm): p50 115.9 -> 85.9 ms, p95 180.0 -> 123.2 ms, F1 unchanged, parity
still 1e-6. The English row predates the fusions.

The English checkpoint is 1.0 F1 point better and 0.08 less over-confident; the multilingual one
is 35% faster and 25% smaller. Both reproduce their PyTorch reference (token ids exact, |delta P|
<= 3e-6) and both fail calibration, latency and footprint.

Latency breakdown, measured by removing parts of the graph: 22 encoder layers ~86% (one layer
~3.5 ms at ~212 tokens), decision head ~5%, tokenizer/embeddings/scorer ~3%. A single-shot CLI
invocation is 645 ms wall: 643 MB of safetensors are read and converted before the first token.

### Latency research

Measured on this machine unless marked *reported*.

| path | change | p50 | p95 | evidence |
| --- | --- | --- | --- | --- |
| baseline candle Metal F16 | - | 115.9 ms | 180.0 ms | measured |
| fused softmax + fused layer norm | 2 call sites | 85.9 ms | 123.2 ms | measured, F1 unchanged |
| 8 of 22 encoder layers | config only | 36.0 ms | 51.0 ms | measured, F1 0.689 on this corpus |
| MLX F16, same weights | runtime swap | ~29 ms | ~35 ms (est.) | measured encoder 23.6 ms at 172 tokens |
| ORT CPU int8, MiniLM-L6 proxy | runtime swap | 7.4 ms @ 212 tok | 9.4 ms | measured, 12.1x fewer FLOPs than here |
| ORT CoreML EP, same proxy | ANE path | 35.2 ms | 36.7 ms | measured; 5x slower than ORT CPU |
| 3 MB distilled CodeBERT (Compressor, ASE 2022) | model swap | - | - | reported: 4.3x faster than CodeBERT, 99.2% of its clone F1 |
| ModernBERT sentence-clone (CodeModernBERT-Owl) | model swap | - | - | reported: F1 0.963 on BigCloneBench, 12 layers |

Arithmetic: ~40-49 GFLOP per forward. p95 <= 10 ms needs >= 4 TFLOPS sustained; candle measures
~2.3 and MLX ~2.9 on M=172 GEMMs. The CoreML/ANE arm is not a shortcut: a 6-layer INT8 proxy ran
5x slower under the CoreML EP than under the ORT CPU EP, because partition copies dominate at this
model size.

### Why 10 ms is out of reach on candle

Two independent reviews were run because the latency gate was failing.

The systems review measured that candle batches 50 dispatches per command buffer and waits 29
times per forward, so dispatch overhead is under 8% of runtime. The wall is arithmetic: ~49 GFLOP
per forward at 212 tokens against ~2.3 TFLOPS for M=172 GEMMs on this GPU, i.e. 18 ms for the GEMM
chain alone. The three cheapest fusions it measured (`sdpa` + fused softmax, fused RoPE, fused
zero-bias layer norm) take p50 from 113 ms to a best case of 27-30 ms, still 3x over target.
Nothing that keeps candle reaches 10 ms at this model size.

The contrarian review attacked the formulation instead, and found the gate itself unsound on this
corpus: with 41% of the synthetic negatives previously unchanged (now fixed), the reference sat
0.4 F1 points above an all-positive constant, and a 4-layer truncation that predicts a constant
passed both quality axes. Its conclusion: the decision should be resolved by a deterministic
normalized-AST/token fingerprint plus a tiny model on the residual, and the 22-layer encoder should
not be in the per-pair path at all.

Both agree on one point: **the checkpoint must leave the hot path, or the latency target must be
restated.**

### Open decisions

1. Latency: restate the gate at p95 <= 35 ms for the candle path, or change the execution layer
   (MLX FFI measured ~29 ms p50, CoreML/ANE unproven), or move the encoder out of the per-pair
   path with a fingerprint cascade.
2. Footprint: 220 MB is below the encoder layers alone at F16 (221 MB) and below INT8-everything
   (322 MB). Meeting it needs INT8 plus embedding factorisation or vocabulary pruning, which
   changes the trained function.
3. Corpus: replace the synthetic pairs with real Type-3/4 clones and real supersede notes before
   any KEEP verdict is treated as evidence of quality.
4. Gate binding: `tests/parity.rs` requires logit parity with the PyTorch checkpoint; the keep
   matrix requires F1-ratio parity. Architecture changes break the first and can satisfy the
   second, so one of them must be declared non-binding for any distillation work.

## Verification

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
bash tests/scripts/allowlist_check.sh Cargo.lock
bash tests/scripts/write_path_check.sh
bash tests/scripts/hygiene_check.sh
```

Numerical parity needs the weights:

```sh
EG_LAYA_MODEL=.models/laya-multilingual cargo test --release --test parity -- --nocapture
```

The parity suite rebuilds the prompt with the Rust tokenizer (ids and marker positions must match
exactly), runs the Rust encoder and head, and requires `|P(true) - P(true)reference| < 2e-3` on
the golden fixtures.

## Layout

```
src/
  model/encoder.rs   ModernBERT forward: embeddings, RoPE, GeGLU, global + sliding attention
  model/head.rs      2-layer pre-norm encoder head and the per-marker scorer
  model/mod.rs       checkpoint loader, forward, temperature-scaled softmax
  tokenizer.rs       tokenizer.json wrapper and the exact Laya prompt layout
  judge.rs           instructions, state layout, verdict
  bench.rs           latency percentiles, F1, ECE
  corpus.rs          labelled pair corpus
  run.rs, cli.rs     subcommands, stdin/file inputs, exit codes
scripts/
  export_laya.py     download, export, golden fixtures
  reference_eval.py  PyTorch baseline metrics
  keep_gate.py       keep/reject matrix
  make_corpus.py     deterministic 500-pair corpus
tests/
  parity.rs          token and probability parity against the reference
  cli.rs             exit codes and command surface
  fixtures/          golden ids, logits and probabilities
  corpus/pairs.jsonl 500 labelled pairs
```

## Limits

- The checkpoints are general decision models, not code-duplication models. Zero-shot quality on
  code is the open question the keep/reject gate answers; treat an uncalibrated verdict as
  advisory until the gate passes.
- Batch size is one. The encoder skips padding masks entirely, which is correct only for a single
  unpadded sequence.
- INT8/INT4 weight paths are not implemented yet. Loading F32 doubles resident memory relative to
  the FP16 checkpoint; `--dtype f16` restores the checkpoint footprint.
