# Research Log: MoE4SCD + SECOND-OC

Nhật ký nghiên cứu theo session. Mỗi session ghi: việc đã làm, phát hiện quan trọng, kết quả, bước tiếp theo.

---

## 2026-07-01 — Session 3: SECOND-OC Benchmark + Git Setup

### Việc đã làm
- Literature review 24 paper về object-level SCD, change captioning (2023-2025)
  - Đọc full-text: AnyChange (NeurIPS 2024), SECOND-CC (2025), RCD (2025), SCanNet
  - Kết quả: chốt novelty "exhaustive per-object enumeration trên 1.694 test pairs"
- Viết `plan/Second oc plan v2.md` (Plan v2) và `plan/ Second oc agent plan.md` (Agent Plan với runnable code)
- Phát hiện blocker: `tokens_T1_test/*.pt` không có key "masks", chỉ có features/centroids/areas
- Viết Phase -1 → Phase 1 pipeline scripts:
  - `generate_sam2_masks_test.py` — chạy SAM2 gen masks, lưu `SECOND/sam2_masks_T1_test/*.npz`
  - `SECOND-OC/config.py` — RGB→class mapping (7 màu SECOND), shared config
  - `SECOND-OC/sam2_loader.py` — load masks từ .npz
  - `SECOND-OC/phase0_verify.py` — verify 1694 pairs, check GT label format
  - `SECOND-OC/phase1_extract_instances.py` — SAM2 mask ∩ GT label → instance JSON
  - `SECOND-OC/phase1_visualize.py` — spot-check visualization

### Phát hiện quan trọng
- GT labels của SECOND là **RGB không phải grayscale**: 7 màu: tree=[0,128,0], buildings=[128,0,0], water=[0,0,255], non_veg=[128,128,128], playground=[255,255,255], low_veg=[0,255,0], other=[255,0,0]
- `_rgb_to_class()` cần exact lookup + nearest-color fallback (có trong code `run_multiscale_token_graph_reasoning.py:87`)
- SAM2 masks tiêu tốn ~1.5-2.2s/pair trên RTX 5880 Ada; ~60-90 phút cho 1694 pairs

### Trạng thái cuối session
- SAM2 mask generation: **đang chạy** (PID 2599468), ~97/1694 khi kết thúc session
- Monitor: `ls SECOND/sam2_masks_T1_test/ | wc -l`
- Log: `/tmp/sam2_masks_gen.log` (mất sau reboot)

### Bước tiếp theo
```bash
# Khi generation xong:
python3.11 SECOND-OC/phase0_verify.py
python3.11 SECOND-OC/phase1_extract_instances.py
python3.11 SECOND-OC/phase1_visualize.py   # view phase1_viz_*.png
```

---

## 2026-04-10 — Session 2: MOB-GCN + Phuong Dao Integration

### Việc đã làm
- Đọc và tích hợp 2 bài báo vào pipeline:
  - **MOB-GCN (2025)**: Multiresolution Graph Network với Gumbel-Softmax pooling
  - **Phuong D. Dao (ISPRS 2021)**: Inverse noise weighting (CV) + MAD outlier removal
- Viết `mob_gcn_modules.py`: `GumbelSoftmaxPool`, `MGNLayer`, `compute_smoothness_loss`
- Viết `token_hierarchical_reasoner.py`: `HierarchicalGraphReasoner`, `HierarchicalChangeReasoner`
- Chạy thực nghiệm multiscale graph vs baseline:
  - `run_multiscale_token_graph_reasoning.py`
  - Kết quả: Baseline F1=0.4763 → Multiscale F1=0.4880 (ΔIoU=+0.0101)

### Kết quả
- MOB-GCN thêm ΔIoU=+0.01 so với baseline graph — cải thiện nhỏ nhưng có ý nghĩa
- Expert specialization analysis: experts chuyên biệt theo **task axis** (change/stability), không phải class

### Bước tiếp theo ghi lại lúc đó
- Run longer training (30+ epochs) cho MOB-GCN
- Fix contrastive crash (disk full at step 1300/2968)
- Implement Rich Token (shape descriptor + spectral signature)

---

## 2026-04-10 — Session 1: Initial Setup + Training + Test Evaluation

### Việc đã làm
- Setup pipeline: SAM2 features → Tokenization → Hungarian Matching → Transformer+GNN+MoE
- Train model `stage5_6_semantic` (best checkpoint)
- Chạy test set evaluation lần đầu với `eval_test_set.py`

### Kết quả chính
```
Test set: 1694 pairs (official SECOND split)
F1     = 0.542
IoU    = 0.372
P      = 0.414   ← quá nhiều FP
R      = 0.784
TP=41,552  FP=58,765  FN=11,452  TN=199,221
```

### Phát hiện
- Token coverage: SAM2 chỉ cover 68.5% pixels, overlap 20%, có 25 giant masks
- Tokenization Projection Loss = 0.178 F1 (gap giữa token F1=0.542 → pixel F1=0.364)
- GNN > MoE routing — spatial context quan trọng hơn expert routing

### Bước tiếp theo ghi lại lúc đó
- Tìm hiểu SECOND GT labels để eval thật (không phải proxy labels)
- Oracle tokenizer experiment (dùng GT masks thay vì SAM2)

---

## Model Checkpoint Inventory

| Checkpoint | F1 (val) | F1 (test proxy) | Notes |
|---|---|---|---|
| `SECOND/stage5_6_semantic/best_model.pt` | 0.547 | 0.542 | Best overall, semantic routing |
| `SECOND/stage4/best_model.pt` | ~0.48 | — | Baseline transformer |
| `SECOND/stage4B/best_model.pt` | ~0.50 | — | +GNN |

## Key Files

| File | Vai trò |
|---|---|
| `tokenize_regions.py` | SAM2 tokenization (Stage 2) |
| `token_matching.py` | Hungarian matching (Stage 3) |
| `token_change_reasoner_moe.py` | MoE model architecture |
| `train_reasoner.py` | Training script |
| `eval_test_set.py` | Test set evaluation |
| `generate_sam2_masks_test.py` | Phase -1: gen masks for SECOND-OC |
| `SECOND-OC/` | Object-centric benchmark pipeline |
| `plan/` | Research plans (v2 + agent-executable) |
