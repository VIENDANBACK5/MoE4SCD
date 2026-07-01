# MoE4SCD — Token-based Semantic Change Detection with Mixture of Experts

> **Phát hiện thay đổi ngữ nghĩa** (Semantic Change Detection) cho ảnh viễn thám bằng kiến trúc Token + GNN + MoE, sử dụng bộ mã hóa SAM2.

---

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Kiến trúc Pipeline](#kiến-trúc-pipeline)
- [Dataset](#dataset)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Hướng dẫn chạy từng Stage](#hướng-dẫn-chạy-từng-stage)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Phân tích chuyên gia MoE](#phân-tích-chuyên-gia-moe)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Câu hỏi nghiên cứu & Hướng tiếp theo](#câu-hỏi-nghiên-cứu--hướng-tiếp-theo)

---

## Giới thiệu

Dự án này xây dựng pipeline **Semantic Change Detection (SCD)** tiên tiến cho ảnh viễn thám bằng cách thay thế phương pháp pixel-level truyền thống (UNet/CNN) bằng **Token-based Object-Centric reasoning**.

### Ý tưởng cốt lõi

Thay vì so sánh từng pixel giữa 2 ảnh (T1, T2), hệ thống:
1. **Phân đoạn** ảnh thành các vùng có nghĩa (token) bằng SAM2
2. **Khớp** các token tương ứng giữa T1 và T2
3. **Suy luận** sự thay đổi qua Transformer → GNN → MoE

### Câu hỏi nghiên cứu chính

> Liệu **Semantic-guided routing** (tiêm nhãn lớp vào router) có tạo ra chuyên môn hóa chuyên gia tốt hơn **Dynamic routing** (chỉ dựa trên embedding)?

---

## Kiến trúc Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                     INPUT: Ảnh T1 & T2 (512×512)            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: SAM2 Feature Extraction                            │
│  SAM2 Hiera-Large → embeddings (1, 256, 64, 64)             │
│  Script: extract_sam2_features.py                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: Region Tokenization                                │
│  SAM2AutoMaskGenerator → masks → masked avg pool            │
│  Output: tokens (N, 256), centroids (N, 2), areas (N)       │
│  Script: tokenize_regions.py                                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3: Token Matching                                     │
│  Hungarian matching + Top-K pruning + Spatial gating        │
│  Script: token_matching.py                                   │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 4+: Token Change Reasoner                             │
│                                                              │
│  TokenEncoder (Linear + time_embed + pos_mlp + area_mlp)    │
│       ↓                                                      │
│  TransformerReasoner (4 layers, 8 heads, d=384, ff=1536)    │
│       ↓                                                      │
│  GraphReasoner — GNN (k=6 neighbors, 2 layers)              │
│       ↓                                                      │
│  MoELayer (4 experts, dim=512, Top-1 routing, residual)     │
│       ↓                                                      │
│  ChangePredictionHead + DeltaHead                            │
│                                                              │
│  Script: token_change_reasoner_moe.py + train_reasoner.py   │
└──────────────────────────────────────────────────────────────┘
```

---

## Dataset

Sử dụng bộ dữ liệu **[SECOND](https://captain-whu.github.io/SCD/)** (Semantic Change Detection in Remote Sensing).

| Thuộc tính | Giá trị |
|------------|---------|
| Train pairs | 2,968 cặp ảnh (512×512) |
| Test pairs | 1,694 cặp ảnh (512×512) |
| Classes | 7: background, water, soil/impervious, vegetation, building, farmland, low\_veg |
| Format | im1/ (T1), im2/ (T2), label1/, label2/ (PNG semantic maps) |

> ⚠️ **Data không được lưu trong repo** (154GB+). Tải SECOND dataset về và đặt vào thư mục `SECOND/` theo cấu trúc bên dưới.

### Cấu trúc SECOND/ sau khi chạy pipeline

```
SECOND/
├── im1/                  # Ảnh gốc T1
├── im2/                  # Ảnh gốc T2
├── label1/               # Nhãn semantic T1
├── label2/               # Nhãn semantic T2
├── embeddings_T1/        # SAM2 features T1 train (Stage 1)
├── embeddings_T2/        # SAM2 features T2 train (Stage 1)
├── highres_T1/           # High-res features T1 train (Stage 1)
├── highres_T2/           # High-res features T2 train (Stage 1)
├── tokens_T1/            # Region tokens T1 train (Stage 2)
├── tokens_T2/            # Region tokens T2 train (Stage 2)
├── matches/              # Token matches (Stage 3) — 2,968 files
├── stage4/               # Baseline Transformer checkpoints
├── stage4B/              # + GNN checkpoints
├── stage4C/              # + MoE checkpoints
├── stage5_6_dynamic/     # Dynamic routing checkpoints
└── stage5_6_semantic/    # Semantic routing checkpoints ← BEST
```

---

## Cài đặt môi trường

### Yêu cầu

- Python 3.10+
- CUDA GPU (khuyến nghị ≥ 16GB VRAM cho SAM2)
- ~200GB disk space (data + features)

### Cài đặt

```bash
# Clone repo
git clone https://github.com/VIENDANBACK5/MoE4SCD.git
cd MoE4SCD

# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Cài dependencies
pip install -r requirements.txt

# Clone SAM2 và tải checkpoint
git clone https://github.com/facebookresearch/sam2.git sam2
# Tải sam2.1_hiera_large.pt từ Meta và đặt vào sam2/checkpoints/
```

### Cài SAM2 checkpoint

```bash
mkdir -p sam2/checkpoints
# Tải từ: https://github.com/facebookresearch/sam2#model-checkpoints
# File cần: sam2.1_hiera_large.pt (~900MB)
```

---

## Hướng dẫn chạy từng Stage

### Stage 1 — Trích xuất SAM2 features

```bash
python extract_sam2_features.py --split train
python extract_sam2_features.py --split test
```
**Thời gian:** ~16 min (train) + ~9 min (test)  
**Output:** `SECOND/embeddings_T1/`, `SECOND/embeddings_T2/`, `SECOND/highres_T1/`, `SECOND/highres_T2/`

---

### Stage 2 — Tokenization vùng ảnh

```bash
python tokenize_regions.py --split train
python tokenize_regions.py --split test
```
**Thời gian:** ~2h 13m (train) + ~1h 13m (test)  
**Output:** `SECOND/tokens_T1/`, `SECOND/tokens_T2/` — trung bình ~97–101 tokens/ảnh

> **Lưu ý:** Cảnh báo `UserWarning: cannot import name '_C' from 'sam2'` là vô hại (bỏ qua C++ postprocessing).

---

### Stage 3 — Token Matching

```bash
python token_matching.py
```
**Thời gian:** Cực nhanh — 0.00157s/pair  
**Output:** `SECOND/matches/` — 2,968 files `*_matches.pt` + `matching_report.json`

**Chỉ số matching:**
- Avg matches/pair: **88.42** (unmatched ratio: 11.85%)
- Spatial gating pruned: **79.1%** token pairs theo khoảng cách

---

### Stage 4+ — Huấn luyện mô hình

```bash
# Baseline Transformer
python train_reasoner.py --model stage4 --epochs 30

# + GNN (kết quả tốt nhất trong nhóm non-MoE)
python train_reasoner.py --model stage4B --epochs 30

# + MoE (4 experts)
python train_reasoner.py --model stage4C --epochs 30

# A/B test: Dynamic vs Semantic routing
python train_reasoner.py --model stage5_6_dynamic --epochs 30
python train_reasoner.py --model stage5_6_semantic --epochs 30
```

**Hyperparameters chuẩn:**

| Tham số | Giá trị |
|---------|---------|
| token\_dim | 256 |
| hidden\_dim | 384 |
| num\_layers | 4 |
| num\_heads | 8 |
| ff\_dim | 1536 |
| dropout | 0.1 |
| delta\_loss\_weight | 0.2 |
| Optimizer | AdamW + Cosine LR |
| Mixed precision | AMP (fp16) |

---

### Phân tích & Diagnostics

```bash
# Phân tích chuyên môn hóa expert
python stage4/analyze_specialization.py

# Kiểm tra expert diversity collapse
python stage4/analyze_expert_diversity.py

# Visualize expert assignment map
python stage4/visualize_expert_map.py

# Diagnostics dataset
python dataset_diagnostics.py
python matching_diagnostics.py
```

---

## Kết quả thực nghiệm

| Mô hình | Cải tiến chính | val\_change | val\_delta | val\_F1 | val\_IoU |
|---------|----------------|:-----------:|:----------:|:-------:|:--------:|
| **stage4** | Baseline Transformer | 0.6912 | 0.4135 | — | — |
| **stage4B** | + GNN (k=6, 2 layers) | 0.8051 | 0.5890 | — | — |
| **stage4B\_v2** | + weighted matching | 0.7912 | 0.5879 | — | — |
| **stage4C** | + MoE (4 experts) | 0.7908 | 0.5567 | tracked | tracked |
| **stage5\_6\_dynamic** | + Dynamic routing | 0.7879 | 0.4583 | 0.5469 | 0.3764 |
| **stage5\_6\_semantic** | + Semantic routing | 0.8234 | 0.4585 | **0.5470** | **0.3764** |

**🏆 Best model: `stage5_6_semantic` — F1=0.547, IoU=0.376**

### Nhận xét kết quả

- **GNN mang lại bước nhảy lớn nhất** (stage4→stage4B): `val_change` tăng từ 0.691 → 0.805. Cung cấp ngữ cảnh địa lý giúp mô hình phân biệt thay đổi thực sự vs nhiễu.
- **MoE duy trì hiệu suất** nhưng không cải thiện đáng kể — do class imbalance.
- **Semantic vs Dynamic routing**: chênh lệch F1 chỉ **0.0001** — không có khác biệt có ý nghĩa thống kê.

---

## Phân tích chuyên gia MoE

### Kết quả chuyên môn hóa

Dù router MoE ổn định và không collapse, các expert **không chuyên môn hóa theo lớp ngữ nghĩa** (Water, Building...). Nguyên nhân: dataset SECOND có ~80% token thuộc lớp `low_veg`, buộc mọi expert phải xử lý `low_veg`.

**Tuy nhiên, experts phân cực theo chiều "task":**

| Expert | Change Ratio | Vai trò |
|--------|:------------:|---------|
| Expert 0 | 0.408 | **Change processor** — nhạy với vùng thay đổi |
| Expert 1 | 0.197 | **Stability validator** — chuyên vùng ổn định |
| Expert 2 | 0.207 | **Stability validator** |
| Expert 3 | 0.408 | **Change processor** |

### Diversity Collapse Test

| Metric | Giá trị | Kết luận |
|--------|:-------:|----------|
| Off-diagonal cosine similarity | 0.224 | ✅ Không collapse (< 0.9) |
| Router Entropy | 0.062 | Routing rất sắc nét (near one-hot) |
| Expert output similarity | 0.224 | Experts khác biệt về mặt toán học |

---

## Cấu trúc thư mục

```
MoE4SCD/
├── extract_sam2_features.py        # Stage 1: SAM2 feature extraction
├── tokenize_regions.py             # Stage 2: Region tokenization
├── token_matching.py               # Stage 3: Hungarian token matching
├── token_matching_utils.py         # Vectorized math: similarity, Sinkhorn
├── token_change_reasoner.py        # Baseline model + MatchDataset
├── token_change_reasoner_graph.py  # GraphReasoner (GNN)
├── token_change_reasoner_moe.py    # MoELayer + router v1/v2/v3
├── train_reasoner.py               # Training loop: AdamW, AMP, cosine LR
├── diagnostics.py                  # Token dataset diagnostics
├── dataset_diagnostics.py          # Full 10-section diagnostics
├── matching_diagnostics.py         # Stage 3 match quality diagnostics
├── stage4_diagnostics.py           # Stage 4 diagnostics
├── run_expert_ablation.py          # Expert ablation experiments
├── run_expert_behavior.py          # Expert behavior analysis
├── run_expert_scaling.py           # Expert scaling experiments
├── run_multiscale_token_graph_reasoning.py  # Multiscale GNN experiments
├── run_change_map_reconstruction.py         # Stage 7-8 change map
├── stage4/
│   ├── analyze_specialization.py   # Expert-class purity scoring
│   ├── analyze_expert_diversity.py # Diversity collapse test
│   └── visualize_expert_map.py     # Spatial expert assignment viz
├── stage5/                         # Expert analysis outputs
├── stage6/                         # Expert behavior reports
├── stage7/                         # Change map reconstruction
├── stage8/                         # SAM mask alignment experiments
├── tests/                          # Unit tests
├── notebooks/                      # Jupyter notebooks
├── project_summary.md              # Full research summary (handover doc)
├── summarier.md                    # Quick reference summary
├── requirements.txt                # Python dependencies
└── .gitignore
```

---

## Câu hỏi nghiên cứu & Hướng tiếp theo

### Kết luận nghiên cứu

| Câu hỏi | Câu trả lời |
|---------|------------|
| GNN có giúp ích? | ✅ Rõ ràng — +11% val_change |
| MoE có giúp ích? | ⚠️ Duy trì hiệu suất, không cải thiện |
| Semantic > Dynamic routing? | ❌ Không đo được sự khác biệt (ΔF1=0.0001) |
| Expert có chuyên môn hóa? | ❌ Theo class: Không. Theo task (change/stable): ✅ Có |
| Bottleneck chính? | Class imbalance (80% low_veg token) |

### Hướng mở rộng tiếp theo

1. **Imbalance-aware routing** — Bỏ load-balancing loss, dùng class-weighted expert loss
2. **Longer training** — 30 epochs chưa hội tụ với MoE, thử 60-100 epochs
3. **Test set evaluation** — Chạy `best_model.pt` trên 1,694 test pairs
4. **Expert capacity** — Tăng `expert_dim` (512→1024) hoặc thêm experts cho rare classes
5. **Phase 2** — Dùng token embeddings cho pixel-level decoder (UperNet / Mask2Former)
6. **Early semantic injection** — Tiêm semantic labels vào TokenEncoder (early fusion) thay vì chỉ router

---

## Tài liệu tham khảo

- **SECOND Dataset**: [https://captain-whu.github.io/SCD/](https://captain-whu.github.io/SCD/)
- **SAM2 (Segment Anything Model 2)**: [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **Mixture of Experts**: Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017)
- **GraphSAGE**: Hamilton et al., *Inductive Representation Learning on Large Graphs* (2017)

---

## Liên hệ

Dự án được thực hiện trong khuôn khổ nghiên cứu về Remote Sensing & Change Detection.  
GitHub: [@VIENDANBACK5](https://github.com/VIENDANBACK5)
