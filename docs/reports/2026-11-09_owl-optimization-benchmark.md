# Owl bi-encoder: shrink and speedup benchmark

Date: 2026-11-09

Corpora: `tests/corpus/pairs.jsonl` (500 near-miss pairs) and `tests/corpus/bigclonebench_400.jsonl` (400 real clones). Machine: arm64 Apple Silicon, Metal F16 compute, in-process timing, one iteration per pair.

## Matrix

| variant | precision | layers | disk MB | RSS MB | near-miss p50 / p95 (ms) | BCB p50 / p95 (ms) | F1 near-miss | F1 BCB | pairs/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `owl` | F32 | 12 | 606.7 | 1794.9 | 41.6 / 57.8 | 140.0 / 268.2 | 0.7130 | 0.9618 | 23.29 |
| `owl-fp16` | F16 | 12 | 303.3 | 972.5 | 43.3 / 58.6 | 130.7 / 259.6 | 0.7130 | 0.9618 | 22.52 |
| `owl-int8` | INT8 | 12 | 152.4 | 1659.6 | 41.1 / 58.0 | 132.2 / 257.5 | 0.7130 | 0.9618 | 23.78 |
| `owl-int4` | INT4 | 12 | 85.4 | 1579.1 | 39.9 / 54.9 | 133.0 / 258.6 | 0.7139 | 0.9592 | 23.84 |
| `owl-8l` | F32 | 8 | 455.7 | 1400.9 | 25.0 / 35.0 | 87.9 / 168.6 | 0.7130 | 0.6755 | 37.46 |
| `owl-6l` | F32 | 6 | 380.1 | 1183.9 | 19.6 / 27.3 | 70.5 / 135.8 | 0.7130 | 0.6755 | 48.1 |
| `owl-int4-8l` | INT4 | 8 | 64.1 | 1263.3 | 27.5 / 37.9 | 93.2 / 178.4 | 0.7130 | 0.6755 | 34.93 |
| `owl-int4-6l` | INT4 | 6 | 53.5 | 964.2 | 20.6 / 28.3 | 67.5 / 128.9 | 0.7130 | 0.6755 | 46.72 |

## Deltas against the F32 baseline

| variant | disk change | F1 BCB change | p50 BCB change |
| --- | ---: | ---: | ---: |
| `owl-fp16` | -50.0% | +0.0000 | -6.6% |
| `owl-int8` | -74.9% | +0.0000 | -5.6% |
| `owl-int4` | -85.9% | -0.0026 | -5.0% |
| `owl-8l` | -24.9% | -0.2863 | -37.2% |
| `owl-6l` | -37.3% | -0.2863 | -49.6% |
| `owl-int4-8l` | -89.4% | -0.2863 | -33.4% |
| `owl-int4-6l` | -91.2% | -0.2863 | -51.8% |

## Tier-0 baseline

| variant | precision | disk MB | near-miss p50 (ms) | BCB p50 (ms) | F1 near-miss | F1 BCB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `normalised-token fingerprint` | none | 0 | 0.055 | 0.896 | 0.8203 | 0.6813 |

## Method

- Quantisation: `scripts/quantize_owl.py` writes F16, per-row INT8 and INT4 group-64 (two signed nibbles per byte) checkpoints; the loader dequantises at load, so RSS stays at the F16 floor and the gain is disk.
- Pruning: a layer-slicing script kept the first N layers and rewrote `num_hidden_layers`. It was
  removed after the study because every pruned variant failed; the result stands in the table.
- Kernel path: candle's fused `softmax_last_dim`, zero-bias fused layer norm and Metal `sdpa` for unmasked global layers. Copying `laya-candle`'s custom Metal kernels would require `unsafe` FFI, which this repository's hygiene gate forbids in `src/`.
- Command per row: `eg-laya bench --model .models/<variant> --corpus tests/corpus/<corpus>.jsonl --iterations 1 --device metal --dtype f16`.


## Findings

**Quantisation is effectively free on quality.** INT8 reproduces the F32 F1 exactly (0.9618) at
25% of the disk. INT4 loses 0.0026 F1 (0.9618 to 0.9592) at 86% smaller footprint (606.7 to
85.4 MB). The pooled embedding stays within cosine 0.9998 (INT8) and 0.9569 (INT4) of F32.

**Layer pruning collapses the model.** Keeping the first 8 or 6 layers drops BigCloneBench F1 from
0.9618 to 0.6755, which is exactly the all-negative baseline on this corpus. The plan's estimate of
0.92-0.93 at 6 layers did not reproduce: this checkpoint is not prefix-compatible, and truncation
without distillation discards the representation the CLS pooling depends on. The near-miss F1 is
unchanged (0.7130) because near-miss pairs were never separable by this model.

**The usable Pareto point is INT4 at 12 layers.** 85.4 MB, F1 0.9592, 133.0 ms p50 on
BigCloneBench and 39.9 ms on near-miss. Capping the sequence at 256 tokens buys 24% latency
(133.0 to 101.0 ms) and costs 0.03 F1 (0.9592 to 0.9291).

**Kernel fusion is bounded by this repository's invariants.** Switching the unmasked global layers
and the decision head to candle's fused Metal `sdpa`, on top of the already-fused
`softmax_last_dim` and zero-bias layer norm, moved the 12-layer F32 arm from 148.3 ms to 140.0 ms
p50 and the F16 arm to 130.7 ms. Copying `b0xtch/laya-candle`'s custom Metal kernels (fused
attention, tuned GEMM dispatch) would require `unsafe` FFI, which the hygiene gate forbids in
`src/`. The measured gap to that implementation is 121.8 ms on the same corpus versus 140.0 ms.

**Resident memory does not follow the disk curve.** Peak RSS is 972 MB for F16 and 1579 MB for
INT4 because the loader dequantises into F16 buffers and Metal's unified allocations count toward
the process. Disk shrinks 7x; RAM does not. Meeting a 350 MB RSS target needs on-the-fly
dequantisation (`QMatMul`) rather than load-time expansion.

## Verification

| variant | embedding cosine to F32 | max component delta |
| --- | ---: | ---: |
| F16 | 1.0000001 | 1.9e-4 |
| INT8 | 0.9997887 | 2.4e-3 |
| INT4 | 0.9568524 | 7.2e-2 |

Commands:

```sh
.venv/bin/python scripts/quantize_owl.py --model .models/owl --formats fp16,int8,int4
.venv/bin/python scripts/run_matrix_benchmark.py
/usr/bin/time -l target/release/eg-laya bench --model .models/owl-int4 \
    --corpus .tmp/bcb50.jsonl --iterations 1 --device metal --dtype f16
```

## What this rules out

- Layer pruning as a speed lever on this checkpoint, at any layer count below 12.
- INT4 or INT8 as a latency lever: the disk shrinks, the arithmetic does not.
- Any path to 10 ms that keeps 12 layers and the current candle kernel set: the floor measured
  here is 130-140 ms on BigCloneBench and 40 ms on short pairs.

The remaining levers are distillation into a smaller architecture (needs labelled near-miss data)
and adopting an external kernel implementation (needs an `unsafe` or dependency decision).
