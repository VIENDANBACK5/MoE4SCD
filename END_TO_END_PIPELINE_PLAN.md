# 🚀 Master Pipeline: End-to-End Execution Plan (iSAID/WHU/SECOND)

This document provides a step-by-step guide to running the entire semantic change detection pipeline, from raw imagery to final hierarchical change heatmaps.

---

## 🛠 Preparation: Directory Structure
Before starting, organize your data as follows:
```text
data/
├── raw/
│   ├── isaid/ (images, semantic_png, instance_id_RGB)
│   └── whu/ (train, val, test folders with T1/T2/label)
└── processed/
    └── isaid_tiles/ (will be created by tiling)
```

---

## 📍 Stage 0: Tiling (Cắt ảnh lớn)
*Mục tiêu: Chia ảnh vệ tinh khổng lồ thành các mảnh 512x512.*

### For iSAID:
```bash
python scripts/preprocess_isaid.py \
    --input_dir data/raw/isaid/images \
    --mask_dir data/raw/isaid/semantic_png \
    --output_dir data/processed/isaid_tiles
```
**🧐 Cách xem In/Out:**
- **Input:** Kiểm tra ảnh gốc trong `data/raw/isaid/images`.
- **Output:** Kiểm tra thư mục `data/processed/isaid_tiles`. Bạn sẽ thấy hàng nghìn ảnh nhỏ 512x512.

---

## 📍 Stage 1: SAM2 Feature Extraction (Trích xuất đặc trưng)
*Mục tiêu: Chạy Encoder của SAM2 để lấy đặc trưng tri giác (embeddings).*

```bash
python extract_sam2_features.py \
    --dataset_root data/processed/isaid_tiles \
    --device cuda
```
**🧐 Cách xem In/Out:**
- **In:** Các file `.png` trong tiles.
- **Out:** Các file `.pt` trong `embeddings_T1` và `embeddings_T2`. (Dùng `torch.load` để xem tensor nếu cần).

---

## 📍 Stage 2: Tokenization & Autotuning (Chia vùng & Tối ưu)
*Mục tiêu: Biến pixels thành các Semantic Tokens và tìm mật độ tối ưu (PPS).*

### 2.1 Autotune (Tìm PPS tốt nhất):
```bash
python run_stage2_autotune.py \
    --dataset_root data/processed/isaid_tiles \
    --gt_dir data/raw/isaid/instance_masks \
    --limit 30
```
- **Output:** Xem file `autotune_results.json` để biết PPS nào (16, 24, 32...) cho chỉ số US/OS tốt nhất.

### 2.2 Tokenize chính thức:
```bash
python tokenize_regions.py \
    --dataset_root data/processed/isaid_tiles \
    --points_per_side 24 \
    --save_masks \
    --visualize --num_vis 10
```
**🧐 Cách xem In/Out:**
- **Input:** Ảnh và Embeddings.
- **Output:** 
  - File `.pt` chứa tokens, areas, centroids, CVs.
  - **Visual:** Xem thư mục `visualizations/stage2/` để thấy các mặt nạ (masks) của SAM2 phủ lên ảnh. Đây là căn cứ để giáo sư đánh giá độ trung thực của vùng.

---

## 📍 Stage 3: Token Matching (Khớp đối tượng)
*Mục tiêu: Tìm các cặp đối tượng tương ứng giữa hai thời điểm.*

```bash
python token_matching.py \
    --dataset_root data/processed/isaid_tiles \
    --visualize --num_vis 5
```
**🧐 Cách xem In/Out:**
- **In:** Token files từ Stage 2.
- **Out:** File `matches/*.pt`.
- **Visual:** Xem `visualizations/stage3/`. Các đường kẻ nối các vùng từ T1 sang T2 sẽ cho thấy thuật toán Hungarian + Spatial Gating có khớp đúng xe hơi, tòa nhà hay không.

---

## 📍 Stage 4: Hierarchical Training (Huấn luyện MOB-GCN)
*Mục tiêu: Dùng GNN phân tầng để suy luận thay đổi.*

```bash
python train_reasoner.py \
    --tokens_T1 data/processed/isaid_tiles/tokens_T1 \
    --tokens_T2 data/processed/isaid_tiles/tokens_T2 \
    --matches data/processed/isaid_tiles/matches \
    --model_type hierarchical \
    --alpha_cv 1.0 \
    --gt_dir data/raw/isaid/instance_masks \
    --epochs 50 \
    --output experiments/isaid_v1
```
**🧐 Cách xem In/Out:**
- **Out:** Kiểm tra `experiments/isaid_v1/training_log.csv`. 
- **Theo dõi:** Cột `val_f1` và các chỉ số OS/US/ED của Cluster sẽ thay đổi sau mỗi epoch.

---

## 📍 Stage 5: Inference & Global Heatmap (Kết quả cuối cùng)
*Mục tiêu: Tạo bản đồ nhiệt (Heatmap) xác suất thay đổi trên toàn bộ ảnh.*

```bash
python inference_visualizer.py \
    --img_p data/processed/isaid_tiles/im1/sample_001.png \
    --tok1 data/processed/isaid_tiles/tokens_T1/sample_001.pt \
    --tok2 data/processed/isaid_tiles/tokens_T2/sample_001.pt \
    --match data/processed/isaid_tiles/matches/sample_001_matches.pt \
    --ckpt experiments/isaid_v1/best_model.pt \
    --model_type hierarchical \
    --output final_results/heatmap_001.png
```
**🧐 Cách xem In/Out:**
- **Output:** File `final_results/heatmap_001.png`.
- **Nghiệm thu:** Đây là kết quả cuối cùng để đưa vào báo cáo. Bản đồ nhiệt màu Đỏ thể hiện khu vực thay đổi mạnh, màu Xanh là không đổi.

---

## 📂 Danh sách các file Visualizations cần lưu cho báo cáo:
1. `visualizations/stage2/*.png`: Chứng minh SAM2 phân đoạn đúng đối tượng.
2. `visualizations/stage3/*.png`: Chứng minh Matching khớp đúng đối tượng qua thời gian.
3. `experiments/isaid_v1/training_log.csv`: Đồ thị loss và accuracy.
4. `final_results/*.png`: Bản đồ thay đổi cuối cùng (Heatmap).
