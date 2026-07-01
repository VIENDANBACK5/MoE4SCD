# Implementation Plan: SECOND-OC Benchmark — Version 2.0

> **Bản cập nhật** sau literature review đầy đủ (24 paper, đọc full-text 4 paper áp sát nhất). Xem `second-oc-literature-review.md` để tra cứu nguồn. Sửa toàn bộ 6 vấn đề nghiêm trọng từ critique ban đầu.

---

## Định vị novelty (đã chốt, sau khi đối chiếu kỹ với 3 paper cạnh tranh)

SECOND-OC khác với 3 paper áp sát như sau:

| Paper | Họ làm | SECOND-OC khác gì |
|---|---|---|
| **SECOND-CC** (1/2025) | 1 nhãn class-transition (thay đổi nổi bật nhất) + 5 caption/ảnh, split 256×256 quadrant tự chế | Exhaustive per-object (mọi object thay đổi) + split 512×512 chuẩn (1.694 test pairs, khớp với toàn bộ SCD baseline) |
| **AnyChange** (NeurIPS 2024) | Object proposal binary (mask AR@1000), không có class-transition label, không caption | Semantic class-transition label tường minh + free-text description theo từng object |
| **RCD** (12/2025) | Text-prompted binary change map theo class, pixel-level dense, không sinh ngôn ngữ | Object-level (mask instance, không phải pixel-dense) + description tự nhiên |

**Claim novelty cụ thể (không còn "first benchmark" chung chung):**
> "SECOND-OC là annotation đầu tiên liệt kê đầy đủ mọi object thay đổi trong từng ảnh (exhaustive per-object enumeration) kết hợp nhãn class-transition có cấu trúc và mô tả ngôn ngữ tự nhiên, trên đúng split chuẩn 512×512 của SECOND (1.694 test pairs) dùng để so sánh với toàn bộ SCD literature."

---

## Tổng quan Pipeline (đã sửa)

```
SECOND test set (1,694 pairs)
    im1/, im2/           ← RGB images 512×512
    label1/, label2/     ← GT semantic labels (pixel-level, 6 classes)
    tokens_T1_test/      ← SAM2 tokens đã có
    tokens_T2_test/      ← SAM2 tokens đã có
    matches_test/        ← Token matching đã có
         │
         ▼
  [PHASE 0] Baseline Verification  ← MỚI — làm trước mọi thứ
  Xác minh split 1.694 test pairs có khớp TaCo/ChangeStar2 không
         │
         ▼
  [PHASE 1] Object Instance Extraction  ← ĐÃ SỬA: SAM2 mask ∩ GT label
  SAM2 tokens → object masks; GT label → dominant class per mask
         │
         ▼
  [PHASE 2] Change Type Classification  ← ĐÃ SỬA: không còn IoU matching
  Mỗi SAM2 mask T1 → áp đúng mask lên GT label T2 → xác định class chuyển
         │
         ▼
  [PHASE 3] Semantic Description Generation
  matched pairs → template caption + (optional) VLM caption
         │
         ▼
  [PHASE 4] Benchmark Packaging + Evaluation Protocol
  JSON annotations + Binary-Object-F1 + Semantic-Object-F1
```

---

## PHASE 0: Baseline Verification (MỚI — 2-3 ngày)

**Vấn đề phát hiện:** 2 dòng paper SCD dùng split khác nhau trên SECOND:
- **SCanNet/Bi-SRNet dòng cũ (≤ 2022):** 2.375 train + 593 test (= 2.968 GT pairs)
- **TaCo/ChangeStar2/GSTM-SCD dòng mới (≥ 2023):** 2.375 train + 593 val + 1.694 test (= 4.662 pairs)

Không thể đưa số từ 2 dòng này vào cùng bảng benchmark mà không kiểm tra.

**Việc cần làm:**
```python
# 1. Xác nhận test split bạn đang dùng
import os
test_pairs = os.listdir("SECOND/test/im1/")
print(f"Số test pairs: {len(test_pairs)}")  # cần = 1694

# 2. Check TaCo split file nếu repo của họ có sẵn
# 3. Ghi lại: baseline nào dùng split nào → quyết định cái nào cần tự chạy lại

# Nếu bạn dùng 1694 pairs → SCanNet/Bi-SRNet numbers KHÔNG thể so sánh trực tiếp
# → cần tự chạy SCanNet trên 1694 pairs, hoặc chỉ compare với TaCo/ChangeStar2 dòng
```

**Deliverable:** Một file `split_verification.md` ghi rõ baseline nào dùng được, baseline nào cần chạy lại.

---

## PHASE 1: Object Instance Extraction (ĐÃ SỬA)

**Vấn đề gốc:** Plan cũ dùng Connected Component labeling trên GT semantic mask → sẽ ghép nhiều building sát nhau thành 1 blob, không phải true instance.

**Fix đã xác nhận:** Dùng SAM2 masks (đã có sẵn trong `tokens_T1_test/`) làm ranh giới object, sau đó gán class từ GT label.

**File:** `build_object_instances_v2.py` (NEW)

**Thuật toán:**
```python
CLASSES = {0: "low_veg", 1: "buildings", 2: "water",
           3: "barren", 4: "hard_surface", 5: "high_veg"}
# Lưu ý: SECOND gốc dùng 6 class khác tên — cần map từ GT labels của dataset cụ thể bạn đang dùng

def extract_instances_from_sam2(stem, sam2_tokens_dir, gt_label_path, min_area=100):
    """
    Dùng SAM2 masks thay vì CC labeling.
    SAM2 masks đã encode ranh giới object thật,
    không bị vấn đề 'nhiều building liền nhau = 1 blob'.
    """
    # 1. Load SAM2 masks từ tokens directory
    sam2_masks = load_sam2_masks(sam2_tokens_dir, stem)  # list of binary mask (H×W)
    
    # 2. Load GT semantic label
    gt_label = np.array(Image.open(gt_label_path))  # shape (512, 512)
    
    instances = []
    for mask_id, mask in enumerate(sam2_masks):
        if mask.sum() < min_area:
            continue
        
        # 3. Gán class bằng mode của GT label trong vùng mask (dominant class)
        pixels_in_mask = gt_label[mask.astype(bool)]
        if len(pixels_in_mask) == 0:
            continue
        
        values, counts = np.unique(pixels_in_mask, return_counts=True)
        # Bỏ qua no-change class (0) nếu dataset encode no-change vào GT
        dominant_class_id = values[np.argmax(counts)]
        dominant_class_name = CLASSES.get(dominant_class_id, "unknown")
        
        ys, xs = np.where(mask)
        instances.append({
            "instance_id": f"{stem}_T1_{mask_id:03d}",
            "sam2_mask_id": mask_id,
            "class_id": int(dominant_class_id),
            "class_name": dominant_class_name,
            "centroid": [float(xs.mean()) / 512, float(ys.mean()) / 512],
            "area_px": int(mask.sum()),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            # mask lưu riêng dạng .npy hoặc encoded RLE để tiết kiệm dung lượng
        })
    
    return instances
```

**Lưu ý thiết kế:** Nếu SAM2 tokens trong project không phải dạng binary mask mà dạng feature embedding khác, cần đọc format của `tokens_T1_test/` trước. Đây là bước cần check thực tế với data.

**Ước tính output:** 1,694 pairs × ~15 SAM2 masks/ảnh = ~25,000 object instances

---

## PHASE 2: Change Type Classification (ĐÃ SỬA HOÀN TOÀN)

**Vấn đề gốc:** Plan cũ dùng IoU-based bipartite matching giữa mask T1 và mask T2. Từ AnyChange paper (bảng số liệu trực tiếp trên SECOND): SAM+Mask Match (đúng cách này) chỉ đạt F1=14.2, mask AR=3.7, trong khi latent matching đạt F1=41.8, AR=29.0.

**Fix:** Tận dụng T1/T2 đã **co-registered** — không cần "tìm" mask tương ứng ở ảnh kia. Đơn giản áp đúng mask T1 lên GT label T2 tại cùng vị trí địa lý.

**File:** `build_change_labels.py` (NEW, đơn giản hơn nhiều)

**Thuật toán:**
```python
def classify_change_per_object(t1_instance, gt_label_t1, gt_label_t2, 
                                iou_threshold_for_disappeared=0.05):
    """
    Vì T1/T2 co-registered: dùng đúng mask của T1 instance để
    lookup dominant class ở T2 — không cần bipartite matching.
    
    Lý do đơn giản hơn AnyChange: ta có GT label (ground truth),
    không cần dùng embedding similarity để ước lượng semantic change.
    """
    mask = t1_instance["mask"]  # binary mask 512×512
    
    # Class của object này ở T1 (đã có từ Phase 1)
    class_t1 = t1_instance["class_name"]
    
    # Class của cùng vị trí đó ở T2 (lookup GT label T2)
    pixels_at_t2 = gt_label_t2[mask.astype(bool)]
    
    if len(pixels_at_t2) == 0:
        return {"change_type": "disappeared", "class_t2": None}
    
    values, counts = np.unique(pixels_at_t2, return_counts=True)
    dominant_class_t2_id = values[np.argmax(counts)]
    class_t2 = CLASSES.get(dominant_class_t2_id, "unknown")
    
    # Nếu T2 label là mostly "no-change" (unlabeled) → disappeared
    # Cần check encoding của GT: SECOND annotates only changed pixels
    no_label_ratio = (pixels_at_t2 == NO_CHANGE_CLASS_ID).sum() / len(pixels_at_t2)
    if no_label_ratio > (1 - iou_threshold_for_disappeared):
        return {"change_type": "disappeared", "class_t1": class_t1, "class_t2": None}
    
    # Xác định loại thay đổi
    if class_t1 == class_t2:
        change_type = "no_change"
        change_label = "unchanged"
    else:
        change_type = "semantic_change"
        change_label = f"{class_t1} → {class_t2}"
    
    return {
        "change_type": change_type,
        "class_t1": class_t1,
        "class_t2": class_t2,
        "change_label": change_label,
        "t2_class_confidence": float(counts.max()) / len(pixels_at_t2)
    }
```

**Lưu ý quan trọng về encoding SECOND GT:** SECOND chỉ annotate changed pixels — no-change pixels có thể không có label hoặc có label riêng tùy implementation. Cần kiểm tra `label1/`, `label2/` của data thực tế để biết giá trị pixel no-change = bao nhiêu.

**4 loại thay đổi (giữ nguyên từ plan cũ, nhưng logic assign đã sửa):**

| Loại | Ký hiệu | Định nghĩa |
|---|:---:|---|
| No change | `NC` | class T1 == class T2 ở cùng vị trí địa lý |
| Semantic change | `SC` | class T1 ≠ class T2 ở cùng vị trí địa lý |
| Appeared | `AP` | Vùng T2 có object nhưng T1 tại đó là no-change/different |
| Disappeared | `DP` | Vùng T1 có object nhưng T2 tại đó là no-change/different |

---

## PHASE 3: Semantic Description Generation (ít thay đổi)

### 3A — Template-based (giữ nguyên)
Dùng 30 class-transition combinations → template caption chuẩn hóa.

```python
CHANGE_TEMPLATES = {
    ("low_veg", "buildings"): "Low vegetation cleared and replaced by buildings (construction)",
    ("buildings", "hard_surface"): "Buildings demolished, area converted to hard surface",
    ("barren", "buildings"): "Bare land developed into buildings",
    # ... 30 combinations từ 6 classes
    # Thêm modifier cho intensity: partial change nếu t2_class_confidence < 0.8
}

def generate_template_caption(change_type, class_t1, class_t2, confidence=1.0):
    if change_type == "no_change":
        return f"No land cover change: {class_t1} remains unchanged"
    elif change_type == "semantic_change":
        key = (class_t1, class_t2)
        base = CHANGE_TEMPLATES.get(key, f"{class_t1} changed to {class_t2}")
        if confidence < 0.7:
            return f"[Partial] {base} (partial change, mixed land cover)"
        return base
    elif change_type == "appeared":
        return f"New {class_t2} area appeared (previously no significant cover)"
    elif change_type == "disappeared":
        return f"{class_t1} area disappeared or replaced"
```

### 3B — VLM-based (Tier 1, optional, 300 pairs)
**Sửa từ plan cũ:** Không dùng LLaVA/InternVL2 gốc (không được train trên RS imagery). Dùng RS-specialized model:
- **GeoChat** (Kuckreja et al. 2023) — VLM cho RS, grounded descriptions
- **RemoteCLIP** (Liu et al. 2024) — đã được cite trong paper Token-MoE chính, có thể đã quen
- **SkySenseGPT** nếu accessible

**Bước verification bắt buộc cho Tier 1:**
- Human-verify 50/300 mẫu (inter-annotator agreement, Cohen's κ ≥ 0.7)
- Report VLM hallucination rate (số lần VLM mô tả sai class so với GT label)

---

## PHASE 4: Benchmark Packaging + Evaluation Protocol (ĐÃ SỬA METRICS)

### 4A — Metrics (sửa lại từ critique #4)

**Vấn đề gốc:** Plan cũ chỉ có 1 metric Object-F1, "semantic matching optional."

**Fix:** 2 metric tách biệt, cả hai đều mandatory, liên kết với metrics chuẩn SCD:

```python
def compute_binary_object_f1(predictions, ground_truth, iou_threshold=0.5):
    """
    Binary-Object-F1: Phát hiện đúng object nào thay đổi
    (không yêu cầu đúng loại thay đổi).
    Tương đương pixel-level mIoU nhưng ở cấp object.
    """
    ...

def compute_semantic_object_f1(predictions, ground_truth, iou_threshold=0.5):
    """
    Semantic-Object-F1: Đúng cả object VÀ class-transition (from → to).
    Tương đương SeK nhưng ở cấp object.
    TP = predicted mask overlaps GT changed object (IoU ≥ threshold)
         AND predicted change_label == GT change_type
    """
    ...
```

**3 metrics chính của benchmark:**

| Metric | Tương đương pixel-level | Thay thế cho |
|---|---|---|
| **Binary-Object-F1** | mIoU (binary change) | Pixel Binary F1 |
| **Semantic-Object-F1** | SeK coefficient | Pixel SeK |
| **Change-IoU** | Pixel IoU | Pixel IoU |

### 4B — Baseline table (sửa lại sau khi xác minh split)

Chỉ đưa vào bảng baseline các number đã xác minh chạy trên đúng 1.694 test pairs:
- **TaCo** (arXiv 2511.20306) — tự nhận đúng split 1.694, đáng tin
- **ChangeStar2** (arXiv 2406.15694) — cần verify
- **AnyChange** (NeurIPS 2024) — chạy object proposal evaluation (mask AR@1000) trên SECOND
- **ScanNet/Bi-SRNet** — cần verify split, nếu dùng 593 thì cần chạy lại

### 4C — JSON Format (giữ nguyên từ plan cũ, bổ sung một số field)

```json
{
  "stem": "00004",
  "image_t1": "im1/00004.png",
  "image_t2": "im2/00004.png",
  "split": "official_512x512_1694",
  "objects_t1": [
    {
      "instance_id": "00004_T1_001",
      "sam2_mask_id": 1,
      "class_id": 1,
      "class_name": "buildings",
      "centroid": [0.45, 0.32],
      "area_px": 3420,
      "bbox": [180, 140, 280, 220],
      "mask_file": "masks/00004_T1_001.npy"
    }
  ],
  "objects_t2": [...],
  "change_annotations": [
    {
      "t1_instance_id": "00004_T1_001",
      "change_type": "semantic_change",
      "class_t1": "low_veg",
      "class_t2": "buildings",
      "change_label": "low_veg → buildings",
      "t2_class_confidence": 0.87,
      "description": "Low vegetation cleared and replaced by buildings (construction)",
      "vlm_description": null,
      "tier": 2
    }
  ],
  "summary": {
    "n_changed": 3,
    "n_unchanged": 8,
    "n_appeared": 1,
    "n_disappeared": 2,
    "change_types": ["low_veg→buildings", "barren→hard_surface"],
    "diff_from_second_cc": "exhaustive_all_objects_not_single_prominent"
  }
}
```

---

## Lộ trình (5 tuần, tăng 1 tuần so với plan cũ)

### Tuần 0 (THÊM MỚI): Pre-work + Verification
- [ ] **Phase 0:** Chạy `len(os.listdir("SECOND/test/im1/"))` → xác nhận 1.694 test pairs
- [ ] Check format `tokens_T1_test/`: là binary mask, feature tensor, hay format khác?
- [ ] Xác minh split: TaCo/ChangeStar2 dùng đúng 1.694 hay khác?
- [ ] Literature: Đọc SAGE-CC (arXiv 2511.21420) để tránh trùng Phase 1→3 pipeline design
- [ ] Chốt venue: TGRS (faster, cùng nơi SECOND gốc + SCanNet publish) hay NeurIPS D&B 2026?
- **Deliverable:** `split_verification.md` + format spec cho SAM2 tokens

### Tuần 1: Phase 1 — SAM2-based Object Extraction
- [ ] Viết `build_object_instances_v2.py` — SAM2 mask ∩ GT label
- [ ] Test trên 10 mẫu, visualize: một building = một instance? (không phải một blob)
- [ ] Kiểm tra phân phối: số instances/ảnh, diện tích, class distribution
- [ ] Quyết định `min_area` dựa trên thực tế (ablate trên 50, 100, 200 pixels) — justify bằng data
- **Deliverable:** `instances_T1.json` + `instances_T2.json` (1,694 entries mỗi file)

### Tuần 2: Phase 2 + Phase 3A — Change Labels + Template Captions
- [ ] Viết `build_change_labels.py` — lookup class T2 tại vùng mask T1 (co-registered)
- [ ] Xử lý edge case: GT encoding no-change là gì? (xem format của `label1/`, `label2/`)
- [ ] Chạy toàn bộ 1,694 pairs → kiểm tra phân phối NC/SC/AP/DP
- [ ] Kiểm tra class transition matrix: pair nào có quá ít mẫu? (đặc biệt playground, water)
- [ ] Viết template captions cho 30 combinations
- **Deliverable:** `change_annotations.json` + `template_captions.json`

### Tuần 3: Phase 4 — Eval Script + Baseline Numbers
- [ ] Viết `object_eval.py` với Binary-Object-F1 + Semantic-Object-F1
- [ ] Chạy stage5_6_dynamic (model hiện tại) → lấy Object-F1 làm baseline của chính bạn
- [ ] Chạy 1-2 baseline SCD chuẩn (ChangeStar2 hoặc TaCo) → verify số trên đúng 1.694 split
- [ ] Ghi rõ mỗi baseline chạy trên split nào trong bảng
- **Deliverable:** `eval_results_baseline.json` + `eval/object_eval.py`

### Tuần 4: Tier 1 + Packaging
- [ ] Curate 300 pairs cho Tier 1 (stratified sampling theo change_type)
- [ ] (Optional) Chạy GeoChat/RemoteCLIP trên 300 pairs → VLM descriptions
- [ ] Human-verify 50/300 mẫu nếu làm Tier 1 VLM
- [ ] Đóng gói SECOND-OC theo format chuẩn
- **Deliverable:** `tier1/curated_300.json` + full dataset package

### Tuần 5: Paper Section
- [ ] Viết Dataset section: statistics, construction method, eval protocol, so sánh với SECOND-CC / AnyChange / RCD
- [ ] Tạo figures: so sánh với 3 paper cạnh tranh, ví dụ exhaustive vs single-label, class distribution
- [ ] Bảng so sánh dataset: SECOND-OC vs SECOND-CC vs AnyChange object eval vs RCD
- **Deliverable:** Paper-ready Dataset section draft

---

## Related Work — Positioning Statement (dùng khi viết paper)

> "Existing work approaches object-level change analysis from three angles. AnyChange [NeurIPS 2024] proposes zero-shot object change proposals via SAM's latent matching, but yields only binary class-agnostic masks without semantic transition labels or natural language descriptions. SECOND-CC [2025] augments the SECOND dataset with natural language captions and a single class-transition label per image pair for the most prominent change, using a custom 256×256 quadrant split incompatible with standard SCD evaluation protocols. RCD [2025] enables class-prompted binary change maps via language guidance but operates at the pixel level and does not generate free-form descriptions. SECOND-OC bridges these gaps by providing exhaustive per-object annotations — enumerating all changed instances in each image rather than the single most prominent change — with structured class-transition labels and natural language descriptions, on the official 512×512 split enabling direct comparison with the SCD literature. We additionally introduce Binary-Object-F1 and Semantic-Object-F1 metrics that complement the standard OA/mIoU/SeK/F_scd suite."

---

## File Map (cập nhật)

```
Image Segmentation/
├── split_verification.md        [NEW Tuần 0] Ghi rõ split + format SAM2 tokens
├── build_object_instances_v2.py [NEW Phase 1] SAM2 mask ∩ GT label
├── build_change_labels.py       [NEW Phase 2] Co-registered lookup, không IoU matching
├── generate_descriptions.py     [NEW Phase 3] Template + (optional) VLM
├── object_eval.py               [NEW Phase 4] Binary-Object-F1 + Semantic-Object-F1
└── SECOND-OC/
    ├── annotations/
    │   ├── instances_T1.json
    │   ├── instances_T2.json
    │   └── change_annotations.json
    ├── tier1/
    │   └── curated_300.json
    └── eval/
        ├── object_eval.py
        └── README.md            ← Nói rõ split chuẩn (1.694 pairs, 512×512)
```

---

## Số liệu ước tính (sau khi sửa plan)

| Thống kê | Ước tính | Ghi chú |
|---|---|---|
| Total image pairs | 1,694 | Chính xác — official SECOND test split |
| Tier 1 (curated, human-verified) | 300 | Stratified theo change_type |
| Total object instances | ~25,000 | SAM2 masks × 1.694 pairs |
| Changed objects | ~8,000 (32%) | Ước lượng từ 19.87% change pixels |
| Unique change transitions | 30 | Từ 6 classes × 6 classes - diagonal |
| Template descriptions | ~8,000 | Cho changed objects |
| VLM descriptions (Tier 1 only) | ~3,000 | Tier 1 chỉ có changed objects |
| Split compatibility | 100% | Khớp với TaCo, ChangeStar2, AnyChange SECOND eval |

---

## Checklist quyết định trước khi code (Version 2)

- [ ] Đã xác minh format của `tokens_T1_test/` — là mask hay tensor?
- [ ] Đã xác minh 1.694 test pairs khớp split chuẩn
- [ ] Đã đọc SAGE-CC (arXiv 2511.21420) — tránh trùng pipeline design
- [ ] Đã chọn venue (TGRS vs NeurIPS D&B vs workshop)
- [ ] Nếu làm Tier 1 VLM: đã chọn GeoChat hay RemoteCLIP hay tool khác?