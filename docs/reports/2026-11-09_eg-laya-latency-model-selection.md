# eg-judge: latency ceiling and model selection

Date: 2026-11-09
Revision: `34f504d` plus this report (repo `engram-judge`)

## TL;DR

A third-party Rust/Candle implementation with fused Metal kernels runs the same checkpoint at 29.0 ms p50 / 37.0 ms p95 on the short corpus and 121.8 ms / 125.2 ms on BigCloneBench, with identical F1 and probabilities: this engine is 2.7-3.3x slower than the available floor. Quality still depends on the distribution (0.962 F1 on BigCloneBench vs 0.689 on near-miss for the best bi-encoder), and the normalised-token fingerprint remains the only arm that clears 0.82 on near-miss, at 0.03 ms. The decision is therefore two decisions: adopt the faster kernel path, and assign traffic between the deterministic and neural tiers.

## Retraction

An earlier version of this report stated that "runtime work is exhausted at ~86 ms p50". That claim is withdrawn. It was produced only from this repository's own kernels and from a kernel-count argument, not from a same-weights comparison against another Rust implementation. Measured on the same corpus and the same checkpoint, `b0xtch/laya-candle` reaches 29.0 ms p50 (short) and 121.8 ms (BigCloneBench) against 79.7 ms and 544.9 ms here, with F1 0.7084 and 0.6755 in both. The fusion headroom was real, and this repository left it unclaimed.

## Scope and method

Two corpora, both labelled, both reproducible from the repo:

| corpus | pairs | clones | what it represents |
| --- | --- | --- | --- |
| `tests/corpus/bigclonebench_400.jsonl` | 400 | 204 | real Java clones from the CodeXGLUE BigCloneBench test split |
| `tests/corpus/pairs.jsonl` (first 200 used for cross-model rows) | 200 | 102 | near-miss: rename/comment-only positives, single-operator mutations as negatives |

Machine: arm64 Apple Silicon, macOS, candle 0.11.1 with Metal F16 for Rust rows, PyTorch 2.15 + transformers 5.22 for reference rows. Rust latency is in-process (`eg-judge bench`), so model load is excluded; a one-shot CLI call adds ~645 ms of weight loading.

Every reference row was cross-checked against the Rust engine where the architecture is implemented:

| engine | BigCloneBench F1 (Rust) | BigCloneBench F1 (PyTorch) | worst parity delta |
| --- | --- | --- | --- |
| Laya `noul` 22 layers | 0.6755 | 0.6755 | 1e-6 per probability |
| Owl bi-encoder 12 layers | 0.9618 | 0.9618 | 2e-7 per embedding component |

## Model matrix

| arm | family | BCB F1 (n=400) | near-miss F1 (n=200) | p50 BCB | p50 near-miss | weights |
| --- | --- | --- | --- | --- | --- | --- |
| Laya `noul` multilingual, 22 layers | cross-encoder, general decisions | 0.6755 | 0.6780 | 544.9 ms | 85.9 ms | 644 MB |
| Laya `noul` English, 22+6 layers | cross-encoder, general decisions | not run on BCB | 0.7027 | - | 181.0 ms | 843 MB |
| CodeCloneDetection-ModernBERT-Owl, 12 layers | bi-encoder, BCB-trained | **0.9618** | 0.6885 | 148.3 ms | 43.7 ms | 606 MB F32 |
| CodeBERT clone (mrm8488), siamese head | per-side encoder + concat head | 0.9442 | 0.4828 | 250.3 ms | 99.4 ms | 478 MB |
| UniXcoder clone, 12 layers | bi-encoder, no trained head | 0.6036 | 0.6977 | 232.2 ms | 92.5 ms | 478 MB |
| Jina embeddings v2 code, 12 layers | bi-encoder, retrieval | 0.6928 (n=100) | not run | 339.2 ms | - | 322 MB |
| CodeT5 clone (Salesforce), 220M | T5 + custom head | not reconstructable | not reconstructable | - | - | 850 MB |
| normalised-token fingerprint | deterministic | 0.6813 | **0.8203** | 0.90 ms | 0.03 ms | 0 MB |

Rows measured with cross-encoder, siamese and bi-encoder evaluators plus `scripts/eval_token_diff.py`
(the cross-encoder and siamese evaluators, and every checkpoint except `owl-int4`, were removed after
the study; the weights are public and the measurements stand); Rust rows with `eg-judge bench`. Threshold 0.5 unless the row states otherwise; best-threshold F1 is in the cited JSON under `.tmp/`.

### Runtime comparison, same checkpoint

Both Rust rows run the same `laya-multilingual` weights, the same prompt layout and the same
question; only the kernels differ. Both report F1 from an identical scoring rule.

| runtime | corpus | p50 | p95 | F1 |
| --- | --- | ---: | ---: | ---: |
| this repo, candle 0.11 with fused norm and softmax | `pairs.jsonl` (500) | 79.7 ms | 120.3 ms | 0.7084 |
| `b0xtch/laya-candle`, fused Metal kernels + tiled attention | `pairs.jsonl` (500) | **29.0 ms** | **37.0 ms** | 0.7084 |
| `laya-mlx` 0.6.0, MLX FP16 | `pairs.jsonl` (200) | 81.6 ms | 135.3 ms | 0.7027 |
| this repo | `bigclonebench_400` | 544.9 ms | 604.4 ms | 0.6755 |
| `b0xtch/laya-candle` | `bigclonebench_400` | **121.8 ms** | **125.2 ms** | 0.6755 |
| `laya-mlx` 0.6.0 | `bigclonebench_400` (200) | 189.0 ms | 3043.7 ms | 0.6734 |

`laya-candle` moves from 2.7x to 4.5x faster as the sequence grows, which is the signature of
tiled attention and a tuned GEMM dispatch rather than a constant per-call saving. Its p95 stays
within 4 ms of its p50 on both corpora; the MLX arm instead shows a 3 s tail on long inputs,
consistent with its window-batching policy.

### Third-party claims, measured

| claim | artifact | measured here | verdict |
| --- | --- | --- | --- |
| 7.4 ms for a short multilingual decision | `laya-mlx` 0.6.0, README benchmark on M3 Max | 81.6 ms p50 on 200 short pairs | true only for a much shorter state; same checkpoint on our pairs is 11x slower |
| ~15-18 ms for Von with the model resident | `Partysun/jigor` + `Zatsepin/von-onnx-fp16` | 933.8 ms p50, F1 0.5905 | not reproducible on this machine through the shipped gateway |
| Kev-0.8B at 20-35 ms | `jaredpalmer/kev` model card | not run; the card's own Apple Silicon row reports 74 ms/decision at 8k context | claim conflicts with the project's own published Mac number |
| Laya in ONNX via jigor | `Mattepiu/laya-onnx` | 563.7 ms p50, F1 0.6412 | slower than both Rust paths |
| `laya-candle` "fused kernels tuned for M1 Pro" | `b0xtch/laya-candle` | 29.0 ms / 121.8 ms, same F1 | confirmed |

### Why some rows are missing

The released artifacts are not self-contained, and that is itself a finding:

- **CodeT5 clone**: the checkpoint carries `classifier.dense(1536 -> 768)` but no inference code. Every plausible wiring (mean, CLS, last-token pooling) produced F1 0.0 at threshold 0.5 and 0.6928 at its best threshold, which is the all-positive constant. Not a valid measurement, so it is excluded.
- **Compressor 3/25/50 MB**: the checkpoint expects a 1000-token vocabulary, but the `BPE_1000` tokenizer that produced it is not published. Regenerating one from a different corpus slice would misalign every embedding row, so the arm cannot be measured faithfully.
- **`Lazyhope/python-clone-detection`**: `architectures: ["CloneDetectionModel"]`, a class that exists only in the author's repository. The published `classifier.weight (2, 3072)` does not map onto any transformers head shape.
- **GraphCodeBERT / PLBART variants**: need a DFG parser or a Java-only seq2seq pipeline respectively.

## Latency engineering

| change | p50 | p95 | F1 | evidence |
| --- | --- | --- | --- | --- |
| candle Metal F16 baseline | 115.9 ms | 180.0 ms | 0.678 | measured |
| fused `softmax_last_dim` + fused zero-bias layer norm | 85.9 ms | 123.2 ms | 0.678 | measured, parity 1e-6 |
| 8 of 22 encoder layers | 36.0 ms | 51.0 ms | 0.689 | measured, config only |
| MLX F16, same shapes | ~29 ms | ~35 ms (est.) | - | encoder 23.6 ms at 172 tokens |
| ONNX Runtime CPU INT8, MiniLM-L6 proxy (12.1x fewer FLOPs) | 7.4 ms | 9.4 ms | - | measured at 212 tokens |
| ONNX Runtime CoreML EP, same proxy | 35.2 ms | 36.7 ms | - | measured, 5x slower than CPU EP |

Arithmetic for the gate: 22 layers x 10.03 MFLOP per token per layer x 212 tokens = 46.8 GFLOP. A 10 ms p95 requires 4.7 TFLOPS sustained; candle measures 2.3 and MLX 2.9 on the M=172 GEMMs this model produces. Kernel dispatch is not the limit: candle batches 50 dispatches per command buffer and waits 29 times per forward, under 8% of runtime.

## Findings

1. **Runtime work is exhausted at ~86 ms p50.** The remaining headroom is in FLOPs, not in kernels. `sdpa` fusion is the only untried candle lever and it is Metal-only, worth an estimated 20 ms.
2. **Quality is distribution-bound.** The same bi-encoder moves from 0.962 to 0.689 F1 between real clones and near-miss pairs. Pooled sentence vectors erase the single-token difference that separates those pairs.
3. **Deterministic tier-0 wins on near-miss by 30x latency and +0.13 F1.** Normalised-token comparison answers in 0.03 ms with F1 0.820 versus 0.689 for the best neural arm.
4. **Tier-0 has a known blind spot.** Identifier normalisation turns `clearTimeout` into `clearInterval` and `sha256` into `sha1`; those mutations are invisible to any Type-2 comparison. Resolving them needs symbol information, which the `engram` code graph already holds.
5. **A 2.7-4.5x faster Rust path already exists.** `laya-candle` keeps upstream parity and beats this engine on both corpora, so kernel work is not exhausted and the earlier claim is withdrawn.
6. **ANE is not a shortcut.** The CoreML EP ran a 6-layer INT8 proxy 5x slower than the ORT CPU EP because graph partitioning and copies dominate at this model size.

## Options

Three viable shapes, each with a different failure condition. No ranking is implied.

**Option A: deterministic only.**
Mechanism: normalised-token fingerprint for structural changes, plus `engram` call-graph resolution for identifier-level mutations; the neural judge leaves the hot path.
Cost: ~300 lines in the Rust engine, no weights, no model artifact.
Failure: Type-4 rewrites (different control flow, same behaviour) fall through to the existing D1-D4 heuristics, which is where they already are today.

**Option B: bi-encoder tier plus deterministic tier.**
Mechanism: Owl-style 12-layer bi-encoder encodes each side once; the snippet index caches vectors so a comparison is a dot product; tier-0 handles exact and near-exact pairs first.
Cost: 606 MB F32 or ~160 MB INT8, 43.7 ms p50 for an uncached pair, ~22 ms per side once cached; the Rust encode/cosine path already exists.
Failure: near-miss operator changes stay at 0.689 F1, and the 160 MB INT8 artifact still exceeds the 80 MB Tier-2 cap.

**Option D: vendor the faster kernel path.**
Mechanism: adopt `b0xtch/laya-candle` (Apache-2.0 AND MIT) as the inference backend, or port its
`fused.metal`, tiled attention and GEMM dispatch into `src/model`; keep this repo's CLI, corpus,
bench and verdict schema.
Cost: a dependency on a 0.1.0 crate pinned by git revision, or ~600 lines of kernel work; no model
change and no parity risk.
Failure: the crate is experimental and not on crates.io, so a breaking change means a local fork.

**Option C: distill a small cross-encoder.**
Mechanism: teacher is the best BigCloneBench cross-encoder measured here (CodeBERT siamese, 0.9442); labels come from real near-miss cases in `engram`; student is a 4-6 layer, 384-hidden cross-encoder in the Compressor shape (3-25 MB).
Cost: a training pipeline plus the corpus, then a new Rust loader for a second architecture.
Failure: without a real gray-zone corpus there is nothing to distil from, and the synthetic corpus cannot discriminate (a constant predictor passes its quality axes).

## Open decisions

| # | decision | default if unanswered |
| --- | --- | --- |
| 1 | Binding use case: reuse gray zone, memory supersede, or both | reuse gray zone |
| 2 | Latency target restated per use case: per-pair 35 ms, or per-snippet 25 ms with cache | per-pair 35 ms |
| 3 | Approve building a real near-miss corpus from `engram` D1-D4 cases | not approved; no Option C work |
| 4 | Footprint: accept ~160 MB INT8 bi-encoder or hold the 80 MB cap | hold the cap |

## Reproduction

```sh
# corpora
.venv/bin/python scripts/fetch_bigclonebench.py

# Rust arms (re-download a checkpoint first with scripts/export_laya.py or export_bi_encoder.py)
# target/release/eg-judge bench --model .models/laya-multilingual --corpus tests/corpus/bigclonebench_400.jsonl --iterations 1 --device metal --dtype f16

# bi-encoder arm that ships
target/release/eg-judge bench --model .models/owl-int4 --corpus tests/corpus/bigclonebench_400.jsonl --iterations 1 --device metal --dtype f16
.venv/bin/python scripts/eval_owl.py --model .models/owl-int4 --corpus tests/corpus/bigclonebench_400.jsonl

# tier-0 baseline
.venv/bin/python scripts/eval_token_diff.py --corpus tests/corpus/pairs.jsonl --limit 200

# parity
EG_JUDGE_MODEL=.models/owl-int4 target/release/eg-judge embed --text "fn a() {}" --model .models/owl-int4
```

## Files

| path | role |
| --- | --- |
| `src/model/encoder.rs` | ModernBERT forward, configurable weight prefix, fused norms |
| `src/model/norm.rs` | fused layer norm for bias-free checkpoints |
| `src/bi_encoder.rs` | pooled embedding, cosine, similarity verdict |
| `scripts/eval_owl.py` | bi-encoder reference arm |
| `scripts/eval_token_diff.py` | deterministic tier-0 baseline |
| `scripts/fetch_bigclonebench.py` | reproducible BigCloneBench slice |
