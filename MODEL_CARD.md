---
language: code
license: apache-2.0
tags: [code, clone-detection, bi-encoder, modernbert, int4, engram]
metrics: [f1]
model-index:
  - name: owl-int4
    results:
      - task: {type: text-classification}
        dataset: {name: BigCloneBench, type: bigclonebench}
        metrics:
          - {type: f1, value: 0.9592}
---

# owl-int4: code clone detection bi-encoder (INT4)

Offline-first INT4-quantised ModernBERT bi-encoder for code duplicate detection. Runs entirely
locally through [`engram-judge`](https://github.com/dmoreq/engram-judge): no Python, no network,
no daemon.

## Model details

| Property | Value |
| --- | --- |
| Architecture | ModernBERT-base bi-encoder, 12 layers, 768 hidden, CLS pooling |
| Quantisation | INT4 group-64, two signed nibbles per byte, per-group scales |
| Disk | 85.4 MB |
| BigCloneBench F1 | 0.9592 (F32 baseline 0.9618, delta -0.0026) |
| Embedding cosine to F32 | 0.9569 |
| p50 latency | 40 ms short pair, 133 ms at BigCloneBench length (Apple Metal) |
| Peak RSS | ~1.6 GB (Metal unified memory, weights dequantised at load) |

## Usage

```sh
export EG_JUDGE_MODEL=/path/to/owl-int4
eg-judge similarity --text-a "$A" --text-b "$B"
```

```json
{"decision":"duplicate","similarity":0.868,"threshold":0.5,"latency_ms":72.3,"device":"metal"}
```

`eg-judge embed --text "$A"` returns the pooled, L2-normalised vector so an index can cache one
side and compare with a dot product.

## Files

| File | Purpose |
| --- | --- |
| `model.safetensors` | INT4 group-64 weights: `*.q4` (packed nibbles) plus `*.scale` per group |
| `encoder/config.json` | ModernBERT hyper-parameters |
| `tokenizer/tokenizer.json` | BPE tokenizer, CLS pooling |
| `lazy.json` | Runtime manifest: weight prefix, pooling mode, head layers |

Load with `eg-judge similarity --model <this directory>`; the loader dequantises at load.

## Training data and provenance

Fine-tuned by the checkpoint author on
[BigCloneBench](https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench)
from the public sentence-transformer `Shuu12121/CodeCloneDetection-ModernBERT-Owl`. This repository
only quantised it; no additional training was performed.

The upstream checkpoint is Apache-2.0, so this derivative carries the same licence and its
attribution below.

```bibtex
@misc{shuu12121_codernnclone_modernbert_owl,
  title  = {CodeCloneDetection-ModernBERT-Owl},
  author = {Shuu12121},
  note   = {Sentence-Transformers checkpoint fine-tuned on BigCloneBench},
  year   = {2025}
}
```

## Limits

- Near-miss pairs, which differ by one operator, score F1 0.7139. The pooled vector erases that
  difference; use a token or AST diff for that band.
- BigCloneBench is Java. Performance on Python, Rust and TypeScript is unmeasured.
- INT4 reduces the file, not the arithmetic: inference runs at the dequantised F16 width.
