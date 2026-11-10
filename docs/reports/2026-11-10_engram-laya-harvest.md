# Báo cáo thu hoạch: triển khai `engram-judge` và tích hợp `eg reuse --judge`

Ngày: 2026-11-10
Phạm vi: `engram-judge` (repo mới, 7 commit) và `engram` (`ec32038`)

## TL;DR

Kế hoạch đặt ba mục tiêu định lượng: model ≤ 220 MB, p95 ≤ 10 ms, F1 ≥ 0.95× bản gốc. Thực nghiệm đạt **F1 1.36×** bản gốc và **85.4 MB**, nhưng **p95 chỉ xuống 40 ms** ở input ngắn và 130 ms ở BigCloneBench, tức còn 4–13× so với mục tiêu. Engine Rust đạt parity số học tuyệt đối với PyTorch ở cả ba checkpoint. Tích hợp Phase 4 hoàn tất và đã commit.

## Việc đã giao

| Hạng mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Repo `engram-judge`, binary `eg-judge` | Xong | 7 commit, 51 file theo dõi, 5.085 dòng Rust + Python |
| ModernBERT encoder + noul head trong Candle | Xong | parity 1e-6 với PyTorch |
| Hỗ trợ 3 checkpoint | Xong | `laya-multilingual`, `laya` (ModernBERT-large), Owl bi-encoder |
| Lượng tử hoá FP16 / INT8 / INT4-group64 | Xong | 607 MB → 85.4 MB |
| Cắt tầng 8L / 6L | Xong | không dùng được, xem phần sau |
| Ma trận benchmark + báo cáo | Xong | `docs/reports/2026-11-09_owl-optimization-benchmark.md` |
| `eg reuse --judge` | Xong, đã commit | `engram` `ec32038`, 328 test PASS |

## Số đo chính

Ba checkpoint, cùng máy, Metal F16, đo in-process.

| Model | BigCloneBench F1 | Near-miss F1 | p50 BCB | p50 cặp ngắn | Dung lượng |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laya multilingual 22L | 0.6755 | 0.6780 | 544.9 ms | 79.7 ms | 644 MB |
| Laya English 28L | chưa chạy | 0.7027 | — | 181.0 ms | 843 MB |
| Owl bi-encoder 12L | **0.9618** | 0.6885 | 148.3 ms | 41.6 ms | 606 MB |
| Owl INT4 12L | 0.9592 | 0.7139 | 133.0 ms | 39.9 ms | **85.4 MB** |
| Token-diff tất định | 0.6813 | **0.8203** | 0.90 ms | 0.03 ms | 0 MB |

Mốc so sánh runtime, cùng checkpoint, cùng corpus:

| Runtime | pairs 500 p50 / p95 | BCB p50 / p95 | F1 |
| --- | ---: | ---: | ---: |
| `b0xtch/laya-candle` | **29.0 / 37.0 ms** | **121.8 / 125.2 ms** | 0.7084 / 0.6755 |
| `eg-judge` (repo này) | 79.7 / 120.3 ms | 544.9 / 604.4 ms | 0.7084 / 0.6755 |
| `laya-mlx` 0.6.0 | 81.6 / 135.3 ms | 189.0 / 3043.7 ms | 0.7027 / 0.6734 |

## Bài học kỹ thuật

**1. Kết luận về runtime của tôi đã sai một lần.** Tôi từng kết luận "runtime work is exhausted at ~86 ms" chỉ dựa trên kernel của chính repo. `laya-candle` chạy cùng weights nhanh hơn 2.7–4.5×. Đã ghi retraction trong báo cáo. Bài học: một kết luận về giới hạn cần ít nhất một implementation độc lập làm đối chứng.

**2. Hai lần rà soát độc lập bác đúng hai giả định của tôi.** Review hệ thống bác cơ chế "commit+wait mỗi op" của Candle (thực tế 50 dispatch/buffer, overhead < 8%). Review đối kháng phát hiện 41% negative trong corpus tổng hợp là no-op, và một model cắt còn 4 layer dự đoán hằng số vẫn PASS cả F1 lẫn ECE. Cả hai lỗi đều nằm ở *thiết kế thí nghiệm*, không ở code.

**3. Cắt tầng không phải đường tắt.** 12L → 8L/6L làm BCB F1 rơi từ 0.9618 xuống 0.6755, đúng baseline toàn-dương. Checkpoint này không tương thích tiền tố; cắt mà không distill là vô nghĩa. Giả định 0.92–0.93 trong plan không tái lập được.

**4. Chất lượng phụ thuộc phân phối hơn model.** Cùng một bi-encoder: 0.962 trên clone thật, 0.689 trên near-miss một-token. Vector gộp xoá mất khác biệt cần thiết. Không model nào đã thử vượt 0.82 trên near-miss — và token-diff tất định đạt 0.8203 trong 0.03 ms.

**5. Constraints đúng là thứ chặn đường nhanh nhất.** `laya-candle` nhanh hơn 4.5× nhờ kernel Metal tuỳ biến, nhưng cần FFI `unsafe` — invariant của `engram-judge` cấm. Đây là đánh đổi có ý thức, không phải thiếu sót kỹ thuật.

**6. Môi trường cũng là nguồn lỗi.** Hơn một giờ mất vào build treo do `target/debug/incremental` hỏng, chẩn đoán bằng `ps`/`sample`/`stat` (thấy `nlink = 65535`). Chẩn đoán ban đầu của tôi rằng APFS hỏng cả volume là sai: chỉ cần xoá `incremental`.

## Việc plan hứa mà không đạt

| Mục tiêu | Kết quả | Khoảng cách |
| --- | --- | --- |
| p95 ≤ 10 ms | 40 ms (ngắn), 130 ms (BCB) | 4–13× |
| ≤ 220 MB / ≤ 80 MB | 85.4 MB | 7% vượt trần Tier 2 |
| ECE ≤ 0.06 | 0.105 sau temperature scaling | 1.75× |
| RAM đỉnh < 350 MB | 972–1579 MB | 3–4.5× |

Nguyên nhân gốc: ~46.8 GFLOP/lượt ở 212 token. p95 10 ms đòi 4.7 TFLOPS duy trì; Candle đo 2.3, MLX 2.9. Không đường runtime nào bù được mà không cắt FLOPs hoặc đổi model.

## Tích hợp vào `engram`

`eg reuse --judge` gửi ứng viên overlap 45–75% sang `eg-judge`, tự chọn `similarity` hay `judge-duplicate` theo checkpoint. Tối đa 5 ứng viên/lần, timeout 15 s.

Hợp đồng suy giảm: thiếu binary, thiếu checkpoint, timeout, hoặc output hỏng → stdout byte-identical với static, đúng một dòng `eg: judge:` ở stderr. Có test hồi quy chứng minh.

Đo trên repo thật: 5 ứng viên vùng xám được phán quyết, cả hai checkpoint. Kết quả cho thấy hai model **không đồng thuận** trên cùng ứng viên — bằng chứng nữa rằng vùng xám chưa được giải quyết.

## Quyết định còn chờ

| # | Quyết định | Mặc định nếu không trả lời |
| --- | --- | --- |
| 1 | Nới `unsafe` để port kernel `laya-candle` (2.7–4.5×) hay giữ zero-unsafe | giữ zero-unsafe |
| 2 | Chấp nhận Owl INT4 85.4 MB hay giữ trần 80 MB | chấp nhận 85.4 MB |
| 3 | Duyệt xây corpus near-miss thật từ D1–D4 | không duyệt, dừng hướng distill |
| 4 | Use case ràng buộc: reuse gray zone hay mem supersede | reuse gray zone |

## Đường còn lại

Ba hướng, chưa hướng nào được đo là đạt mục tiêu:

- **Distill cross-encoder nhỏ** (3–25 MB, Compressor). Cần corpus near-miss có nhãn — blocker duy nhất.
- **Cascade tất định + model nhỏ cho phần dư.** Tier-0 đạt 0.82 F1 trên near-miss trong 0.03 ms; model chỉ chạy trên phần không giải được. p95 tổng khi đó bằng median của slow path, nên slow path vẫn cần ≤ 10 ms.
- **Port kernel ngoài.** Đạt 122 ms BCB ngay, không đổi model, nhưng đụng invariant.

## Tệp tham chiếu

| Đường dẫn | Nội dung |
| --- | --- |
| `engram-judge/docs/reports/2026-11-09_eg-judge-latency-model-selection.md` | So sánh model, runtime, third-party claims |
| `engram-judge/docs/reports/2026-11-09_owl-optimization-benchmark.md` | Ma trận lượng tử hoá và cắt tầng |
| `engram-judge/src/` | Encoder, head, bi-encoder, bench, judge |
| `engram/src/query/judge.rs` | Arbitration vùng xám |
| `engram/tests/cli/reuse_judge.rs` | Test hợp đồng suy giảm |

## Kiểm chứng

```sh
# engram-judge
cargo test --locked && cargo clippy --all-targets -- -D warnings
EG_JUDGE_MODEL=.models/owl cargo test --release --test parity -- --nocapture

# engram
CARGO_INCREMENTAL=0 cargo test --locked   # 328 passed, 0 failed
bash tests/scripts/hygiene_check.sh
```

Snapshot: `engram` `ec32038`, `engram-judge` `6dc4a55`. Test chạy trên máy phát triển, một lần mỗi cấu hình.
