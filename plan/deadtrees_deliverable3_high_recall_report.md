# DeadTrees Deliverable 3 — High-recall SAM2 và kết quả end-to-end

Ngày chạy: 2026-08-03

## 1. GPU và môi trường thực thi

GPU vật lý của máy:

- NVIDIA RTX 5880 Ada Generation;
- tổng VRAM 49.140 MiB;
- VRAM trống trước khi chạy 34.161 MiB;
- driver 580.173.02.

Lần kiểm tra đầu trong Codex sandbox trả `torch.cuda.is_available() == False` vì bubblewrap không map `/dev/nvidia*`. Đây không phải thiếu VRAM.

Sau khi chạy ngoài sandbox, môi trường Anaconda mặc định vẫn lỗi `torchvision::nms` do binary mismatch:

- default: PyTorch 2.10.0+cu128, torchvision ops lỗi;
- `cs2_venv`: PyTorch 2.4.0+cu121, torchvision 0.19.0+cu121, CUDA và NMS hoạt động.

Generation cuối cùng dùng `cs2_venv`.

## 2. Cấu hình high-recall định trước

| Tham số | Baseline | High-recall v1 |
|---|---:|---:|
| points per side | 32 | 64 |
| predicted-IoU threshold | 0,86 | 0,75 |
| stability threshold | 0,92 | 0,85 |
| minimum region area | 100 px | 20 px |

High-recall được ghi vào `DeadTrees/sam2_masks_high_recall_v1`, không ghi đè baseline.

Kết quả generation:

- đủ 100/100 ảnh;
- 17.264 proposal trước evaluator area filter;
- thời gian 595,7 giây, khoảng 9 phút 56 giây;
- không OOM.

## 3. Raw proposal comparison

| Metric | Original raw | High-recall raw | Thay đổi |
|---|---:|---:|---:|
| Predictions sau lọc 50 px | 5.052 | 17.233 | +241,1% |
| Mean best IoU / GT | 0,1364 | 0,2957 | +0,1593 |
| Proposal recall @0,25 | 0,2129 | 0,4746 | +0,2617 |
| Proposal recall @0,50 | 0,1374 | 0,3045 | +0,1671 |
| Object F1 @0,25 | **0,1202** | 0,0980 | -0,0222 |
| Object F1 @0,50 | **0,0776** | 0,0630 | -0,0146 |
| GT miss rate | 0,4841 | **0,1344** | -0,3498 |
| Split rate | **0,1585** | 0,5445 | +0,3860 |
| Boundary F1, matched @0,50 | 0,3672 | **0,4294** | +0,0622 |
| ASSD | 6,05 px | **5,89 px** | -0,16 px |

High-recall giải quyết đúng bottleneck miss nhưng tạo quá nhiều duplicate/split/background proposals. Vì vậy proposal recall tăng mạnh trong khi raw detection F1 giảm.

### Recall theo kích thước tại IoU 0,50

| Size quartile | Original raw | High-recall raw |
|---|---:|---:|
| Q1 <= 374,5 px | 0,0080 | **0,0704** |
| Q2 | 0,1006 | **0,2857** |
| Q3 | 0,2621 | **0,5161** |
| Q4 large | 0,1791 | **0,3461** |

Object nhỏ vẫn khó, nhưng recall Q1 tăng khoảng 8,8 lần. Đây là bằng chứng cấu hình generator có tác động đúng hướng, không chỉ tăng mask ngẫu nhiên.

## 4. Classification trên high-recall universe

Nhãn proposal:

- 942 positives tại Hungarian IoU >= 0,25;
- 13.431 clean negatives;
- 2.860 ambiguous;
- 1.987 GT positives cho diagnostic upper bound.

Macro leave-one-site-out:

| Features | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| Shape | 0,1384 | 0,0978 | 0,6281 |
| RGB | 0,1022 | 0,1631 | 0,7349 |
| Shape+RGB | **0,2137** | **0,3164** | **0,7866** |

Shape+RGB tiếp tục tốt hơn Shape-only: +0,0753 macro F1 và +0,2186 PR-AUC. Universe high-recall khó hơn vì số clean negatives tăng hơn ba lần.

## 5. Strict OOF end-to-end result

Mỗi site được lọc bởi model chưa train trên site đó. Classifier threshold được chọn trong inner leave-one-training-site-out, không nhìn outer test.

### IoU 0,50

| Pipeline | Predictions | Precision | Recall | Object F1 |
|---|---:|---:|---:|---:|
| Original raw baseline | 5.052 | 0,0540 | **0,1374** | 0,0776 |
| High-recall raw | 17.233 | 0,0351 | 0,3045 | 0,0630 |
| High-recall + Shape | 6.887 | 0,0485 | 0,1681 | 0,0753 |
| High-recall + RGB | 3.962 | 0,0477 | 0,0951 | 0,0635 |
| High-recall + Shape+RGB | 1.657 | **0,1098** | 0,0916 | **0,0999** |

High-recall + Shape+RGB so với original raw baseline:

- precision tăng 0,0540 → 0,1098;
- Object F1@0,50 tăng 0,0776 → 0,0999;
- absolute F1 gain +0,0223;
- relative F1 gain khoảng **+28,8%**.

### Nhiều IoU thresholds

| IoU | Original raw F1 | High-recall + Shape+RGB F1 | Absolute gain |
|---|---:|---:|---:|
| 0,25 | 0,1202 | **0,1575** | +0,0373 |
| 0,50 | 0,0776 | **0,0999** | +0,0223 |
| 0,75 | 0,0207 | **0,0236** | +0,0029 |

Đây là cấu hình đầu tiên thắng baseline end-to-end trên cùng 100 ảnh và cùng GT universe.

## 6. Diễn giải đúng mức

Kết quả ủng hộ ba mệnh đề:

1. High-recall SAM2 prompting/thresholds tăng mạnh proposal recall và giảm miss.
2. RGB bổ sung tín hiệu cho Shape trong phân loại deadwood proposals.
3. Kết hợp generator recall cao với Shape+RGB filtering cải thiện object F1 end-to-end.

Nhưng pipeline cuối vẫn hy sinh recall so với original raw sau classifier filtering: 0,0916 so với 0,1374 tại IoU 0,50. Threshold hiện được tối ưu cho proposal classification F1, không phải object-detection F1. Đây là phần có thể cải thiện tiếp mà không cần đổi architecture.

## 7. Bước tiếp theo

Ưu tiên tiếp theo:

1. chọn OOF classifier threshold theo end-to-end object F1 trên inner training sites;
2. xử lý duplicate/split proposals bằng NMS hoặc containment suppression dựa trên score trước classifier;
3. giữ nguyên generator v1, không tiếp tục mở grid/giảm threshold cho đến khi xử lý được 54,5% split rate;
4. báo per-site end-to-end để kiểm tra improvement có bị chi phối bởi một site;
5. sau đó mới thử boundary refinement, vì miss đã giảm nhưng split/duplicate đang là lỗi chính.

## 8. Artifacts

- `DeadTrees/sam2_masks_high_recall_v1/manifest.json`;
- `DeadTrees/experiments/high_recall_v1/summary.json`;
- `DeadTrees/experiments/high_recall_v1/per_gt.csv`;
- `DeadTrees/experiments/classification_high_recall_v1/summary.json`;
- `DeadTrees/experiments/classification_high_recall_v1/fold_metrics.csv`;
- `DeadTrees/experiments/classification_high_recall_v1/oof_predictions.csv`;
- `DeadTrees/experiments/classification_high_recall_v1/end_to_end_metrics.json`.
