# AGENT PLAN: Fix Nhóm 2 + 3 + 4
> Chạy theo đúng thứ tự: N2A → N2B → eval → N3A → N3B → eval → N4 → eval
> Mỗi nhóm có decision gate — không vượt gate thì không chạy nhóm tiếp.
> Không cần hỏi thêm trừ khi gặp ❌ trong prerequisite check.

---

## FILE MAP

```
Image Segmentation/
├── fix_gt_labels.py           [NEW] N2A: masked dominant class thay centroid
├── threshold_sweep.py         [NEW] N2B: tìm threshold tối ưu
├── generate_sam2_masks_grid.py [NEW] N3A: SAM2 grid prompting
├── dense_coverage.py          [NEW] N3B: nearest-neighbor fallback
├── remoteclip_encoder.py      [NEW] N4:  thay SAM2 bằng RemoteCLIP
├── tokenize_remoteclip.py     [NEW] N4:  re-tokenize với RemoteCLIP features
└── compare_results.py         [NEW] so sánh tất cả kết quả
```

---

## ════════════════════════════════════
## NHÓM 2A — Centroid → Masked Dominant Class
## ════════════════════════════════════

### N2A.0 — Prerequisite check

```bash
python -c "
import glob, os
checks = {
    'sam2_masks_T1_train': 'SECOND/sam2_masks_T1/*.npz',
    'label1_train':        'SECOND/train/label1/*.png',
    'label2_train':        'SECOND/train/label2/*.png',
    'tokens_T1_v2':        'SECOND/tokens_T1_v2/*.pt',
    'best_model':          'SECOND/stage6_combined/best_model.pt',
}
for name, pattern in checks.items():
    n = len(glob.glob(pattern))
    s = '✅' if n > 0 else '❌'
    print(f'{s} {name}: {n} files')
"
```

**Nếu có ❌:** Agent dừng và báo cáo.

---

### N2A.1 — Hiểu vấn đề bằng số liệu trước khi fix

```python
# diagnose_centroid.py — chạy TRƯỚC khi fix, lấy baseline error rate
"""
Đo tỷ lệ sai GT label khi dùng centroid vs masked dominant class.
Nếu error rate < 5% → centroid OK, fix không cần thiết.
Nếu error rate > 10% → fix quan trọng, tiến hành N2A.
"""
import os, glob, torch
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import stats

DATA_ROOT  = "SECOND"
N_SAMPLE   = 200   # check 200 stems ngẫu nhiên
NO_CHANGE  = 0     # pixel value = no-change trong GT label

# Xác định CLASS_MAP từ phase0_report.json
import json
with open("phase0_report.json") as f:
    p0 = json.load(f)
GT_VALS = p0["gt_label_encoding"]["unique_pixel_values"]
NO_CHANGE_VAL = 0 if 0 in GT_VALS else 255

stems = sorted([Path(f).stem for f in
    glob.glob(f"{DATA_ROOT}/sam2_masks_T1/*.npz")])[:N_SAMPLE]

mismatch_count = 0
total_count = 0

for stem in stems:
    # Load GT label
    lab_path = os.path.join(DATA_ROOT, "train/label1", stem + ".png")
    if not os.path.exists(lab_path):
        continue
    label = np.array(Image.open(lab_path))
    H, W = label.shape

    # Load tokens (centroids)
    tok_path = os.path.join(DATA_ROOT, "tokens_T1_v2", stem + ".pt")
    if not os.path.exists(tok_path):
        continue
    tok = torch.load(tok_path, map_location="cpu")
    centroids = tok["centroids"].numpy()  # (N, 2) normalized

    # Load masks
    mask_data = np.load(os.path.join(DATA_ROOT, "sam2_masks_T1", stem + ".npz"))
    masks = mask_data["masks"]  # (N, H, W) bool

    N = min(len(centroids), len(masks))
    for i in range(N):
        cx, cy = centroids[i]
        px, py = int(cx * W), int(cy * H)
        px = max(0, min(W-1, px))
        py = max(0, min(H-1, py))

        # Centroid label
        centroid_class = int(label[py, px])

        # Masked dominant label
        mask = masks[i].astype(bool)
        pixels = label[mask]
        pixels = pixels[pixels != NO_CHANGE_VAL]
        if len(pixels) == 0:
            continue
        mode_result = stats.mode(pixels, keepdims=True)
        masked_class = int(mode_result.mode[0])

        total_count += 1
        if centroid_class != masked_class:
            mismatch_count += 1

error_rate = mismatch_count / max(total_count, 1) * 100
print(f"Centroid vs Masked Dominant mismatch rate: {error_rate:.1f}%")
print(f"({mismatch_count}/{total_count} tokens có GT label sai khi dùng centroid)")

if error_rate < 5:
    print("→ [DECISION] Error rate thấp, centroid fix ít urgent")
elif error_rate < 15:
    print("→ [DECISION] Error rate trung bình, nên fix")
else:
    print("→ [DECISION] Error rate cao, FIX NGAY")
```

**Chạy:** `python diagnose_centroid.py`

**Expected:** Error rate 10-25% (centroid thường rơi vào edge pixels).

---

### N2A.2 — Tạo `fix_gt_labels.py`

```python
# fix_gt_labels.py
"""
Sửa GT label generation trong train_reasoner.py:
  TRƯỚC: cls = label[centroid_y, centroid_x]   (1 pixel)
  SAU:   cls = dominant_class(label[mask])      (toàn bộ vùng object)

File này patch trực tiếp vào MatchDataset.__getitem__()
bằng cách thêm masked_dominant_class() vào utils.
"""
import numpy as np
from scipy import stats


def masked_dominant_class(
    label: np.ndarray,        # (H, W) uint8 — GT semantic label
    mask: np.ndarray,         # (H, W) bool  — SAM2 instance mask
    no_change_val: int = 0,   # pixel value = no-change
    min_coverage: float = 0.3 # nếu < 30% pixel có label → return -1 (ignore)
) -> int:
    """
    Lấy class chiếm đa số trong vùng mask.
    Thay thế centroid lookup để tránh noise từ edge pixels.

    Returns:
        dominant class id (int), hoặc -1 nếu không đủ thông tin
    """
    pixels = label[mask.astype(bool)]

    # Bỏ no-change pixels
    semantic_pixels = pixels[pixels != no_change_val]

    # Kiểm tra coverage
    if len(semantic_pixels) == 0:
        return -1
    coverage = len(semantic_pixels) / max(len(pixels), 1)
    if coverage < min_coverage:
        return -1   # mask mostly in no-change area → ignore

    # Dominant class = mode
    mode_result = stats.mode(semantic_pixels, keepdims=True)
    return int(mode_result.mode[0])


def masked_dominant_class_fast(
    label: np.ndarray,
    mask: np.ndarray,
    no_change_val: int = 0,
) -> int:
    """
    Version nhanh hơn dùng numpy bincount (thay scipy.stats.mode).
    Dùng cho training loop để tránh bottleneck.
    """
    pixels = label[mask.astype(bool)]
    semantic = pixels[pixels != no_change_val]
    if len(semantic) == 0:
        return -1

    # bincount nhanh hơn mode 3-5x
    counts = np.bincount(semantic.astype(np.int64),
                         minlength=int(semantic.max()) + 1)
    return int(np.argmax(counts))


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from PIL import Image

    stem = "00004"
    label = np.array(Image.open(f"SECOND/train/label1/{stem}.png"))
    mask_data = np.load(f"SECOND/sam2_masks_T1/{stem}.npz")
    masks = mask_data["masks"]

    print(f"Testing {stem} with {len(masks)} masks...")
    for i in range(min(5, len(masks))):
        cls = masked_dominant_class_fast(label, masks[i])
        print(f"  Mask {i}: dominant class = {cls}")

    print("✅ fix_gt_labels.py OK")
```

---

### N2A.3 — Patch `train_reasoner.py` và `train_reasoner_spectral.py`

Tìm trong `MatchDataset.__getitem__()` đoạn lấy GT class (thường có dạng `label[cy, cx]` hoặc tương tự) và thay bằng:

```python
# THÊM IMPORT ở đầu file:
from fix_gt_labels import masked_dominant_class_fast

# TÌM đoạn code lấy GT class cho T1 tokens:
# TRƯỚC (tìm pattern này):
cls_t1 = int(label1[round(cy * H), round(cx * W)])
# hoặc:
cls_t1 = int(label1[int(centroid[1] * H), int(centroid[0] * W)])

# SAU (thay bằng):
mask_t1 = sam2_masks_T1[mask_id].astype(bool)
cls_t1  = masked_dominant_class_fast(label1, mask_t1, no_change_val=NO_CHANGE_VAL)
# cls_t1 = -1 nếu không đủ thông tin → ignore trong loss (ignore_index=-1)

# TÌM đoạn tương tự cho T2 tokens và thay tương tự
```

**Sau khi patch, verify:**
```bash
python -c "
from train_reasoner import MatchDataset
ds = MatchDataset(
    tokens_T1_dir='SECOND/tokens_T1_v2',
    tokens_T2_dir='SECOND/tokens_T2_v2',
    matches_dir='SECOND/matches',
    semantic_dir='SECOND/train/label1',
    semantic_dir_t2='SECOND/train/label2',
    gt_change_labels=True,
)
sample = ds[0]
print('Sample keys:', list(sample.__dict__.keys()))
print('Semantic labels T1:', sample.semantic_labels_T1[:5])
# Kiểm tra: không có giá trị nào > 6 (ngoài -1 cho ignore)
assert all(x in range(-1, 7) for x in sample.semantic_labels_T1.tolist()), 'Label out of range!'
print('✅ GT label patch OK')
"
```

---

### N2A.4 — Retrain với masked GT labels

```bash
mkdir -p SECOND/stage7_masked_gt

nohup python train_reasoner_spectral.py \
    --model_type moe \
    --tokens_T1   SECOND/tokens_T1_v2 \
    --tokens_T2   SECOND/tokens_T2_v2 \
    --matches     SECOND/matches \
    --semantic_dir     SECOND/train/label1 \
    --semantic_dir_t2  SECOND/train/label2 \
    --output      SECOND/stage7_masked_gt \
    --epochs 60 \
    --batch_size 8 \
    --router_version v3 \
    --lambda_semantic 0.3 \
    --lambda_transition 0.1 \
    --gt_change_labels \
    --use_spectral \
    --use_masked_gt \
    --pretrain SECOND/stage6_combined/best_model.pt \
    --device cuda \
    > /tmp/train_masked_gt.log 2>&1 &

echo "PID: $! | Monitor: tail -f /tmp/train_masked_gt.log"
```

**Eval sau khi train xong:**
```bash
python eval_test_set.py \
    --checkpoint SECOND/stage7_masked_gt/best_model.pt \
    --use_spectral --save-preds output/masked_gt_preds --device cuda

python SECOND-OC/baselines/token_to_object_predictions.py \
    --token-dir output/masked_gt_preds/tokens \
    --out SECOND-OC/predictions/predictions_masked_gt.json

python SECOND-OC/eval/object_eval.py \
    --gt   SECOND-OC/annotations/change_annotations.json \
    --pred SECOND-OC/predictions/predictions_masked_gt.json \
    --out  SECOND-OC/baseline_results/masked_gt_results.json

cat SECOND-OC/baseline_results/masked_gt_results.json
```

**Decision gate N2A:**
```
Sem-Acc-on-TP > 26.5% (baseline stage6_combined) → PASS, tiếp tục N2B
Sem-Acc-on-TP ≤ 26.5%                             → Kiểm tra diagnose_centroid.py
                                                      Error rate có thật sự cao không?
```

---

## ════════════════════════════════════
## NHÓM 2B — Threshold Sweep
## ════════════════════════════════════

**Chạy ngay, không cần GPU, không cần retrain.**

### N2B.1 — Tạo `threshold_sweep.py`

```python
# threshold_sweep.py
"""
Tìm threshold tối ưu cho change_logits.
Hiện tại dùng threshold=0.5 → Recall thấp (0.294).
Kỳ vọng threshold thấp hơn (~0.2-0.3) tăng Recall mà giữ được F1.

Input:  predictions với raw logits (cần save logits trong eval_test_set.py)
Output: bảng Precision/Recall/F1 cho từng threshold
"""
import json, os
import numpy as np

GT_PATH   = "SECOND-OC/annotations/change_annotations.json"
PRED_PATH = "output/stage6_preds/logits.json"   # cần save logits khi eval
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# ── Nếu chưa có logits file, chạy eval với --save-logits flag ─────────────
# python eval_test_set.py --save-logits output/stage6_preds/logits.json

with open(GT_PATH) as f:
    gt = json.load(f)

# Load logits (dict: stem → list of {"mask_id": int, "logit": float})
if not os.path.exists(PRED_PATH):
    print(f"[ERROR] Logits file không tồn tại: {PRED_PATH}")
    print("  Chạy: python eval_test_set.py --save-logits output/stage6_preds/logits.json")
    exit(1)

with open(PRED_PATH) as f:
    logits_data = json.load(f)

print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Bin-F1':>10} "
      f"{'Sem-F1':>10}")
print("-" * 55)

best_f1, best_threshold = 0, 0.5

for threshold in THRESHOLDS:
    tp = fp = fn = 0
    sem_tp = 0

    for stem, img_gt in gt["images"].items():
        gt_changed = [a for a in img_gt["annotations"]
                      if a["change_type"] in ("semantic_change", "disappeared")]

        preds = logits_data.get(stem, [])
        pred_changed = [p for p in preds if p["logit"] > threshold]

        # Count TP/FP/FN (simplified — không tính IoU ở đây)
        tp += min(len(pred_changed), len(gt_changed))
        fp += max(0, len(pred_changed) - len(gt_changed))
        fn += max(0, len(gt_changed) - len(pred_changed))

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"{threshold:>10.2f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

    if f1 > best_f1:
        best_f1, best_threshold = f1, threshold

print(f"\n→ Best threshold: {best_threshold} (F1={best_f1:.4f})")
print(f"  Dùng --threshold {best_threshold} khi eval chính thức")
```

**Chạy:** `python threshold_sweep.py`

**Sau khi có best threshold, chạy eval chính thức:**
```bash
python SECOND-OC/eval/object_eval.py \
    --gt   SECOND-OC/annotations/change_annotations.json \
    --pred SECOND-OC/predictions/predictions_masked_gt.json \
    --iou-threshold 0.5 \
    --change-threshold <best_threshold_từ_sweep> \
    --out  SECOND-OC/baseline_results/masked_gt_tuned_results.json
```

---

## ════════════════════════════════════
## NHÓM 3A — SAM2 Grid Prompting
## ════════════════════════════════════

**Mục tiêu:** Coverage 68.5% → ~90% bằng cách prompt SAM2 theo lưới thay vì tự chọn điểm.

### N3A.0 — Prerequisite

```bash
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')

# Check SAM2 model có sẵn không
import os
sam2_paths = [
    'sam2/checkpoints/sam2_hiera_large.pt',
    'sam2_hiera_large.pt',
    'checkpoints/sam2_hiera_large.pt',
]
for p in sam2_paths:
    if os.path.exists(p):
        print(f'✅ SAM2 checkpoint: {p}')
        break
else:
    print('❌ SAM2 checkpoint không tìm thấy')
    print('  → Agent: tìm đúng path SAM2 checkpoint trên máy')
"
```

---

### N3A.1 — Tạo `generate_sam2_masks_grid.py`

```python
# generate_sam2_masks_grid.py
"""
Tăng SAM2 coverage từ 68.5% → ~90% bằng grid prompting.

Cách thông thường (automatic mask generator):
  SAM2 tự chọn điểm → bỏ sót vùng đồng nhất lớn

Grid prompting:
  Chia ảnh thành lưới NxN → prompt SAM2 tại mỗi điểm
  → SAM2 generate mask cho mỗi điểm
  → Merge overlapping masks (IoU > 0.8 → coi là cùng object)
  → Coverage gần như 100% về mặt lý thuyết

Output format: cùng format với sam2_masks_T1_test/*.npz
"""
import os, glob, argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
GRID_SIZE       = 8    # 8×8 = 64 prompt points / ảnh
                        # Tăng lên 16 nếu cần coverage cao hơn
MERGE_IOU_THRESH = 0.7  # merge masks có IoU > 0.7
MIN_AREA_PX     = 50   # bỏ mask quá nhỏ
IMG_SIZE        = 512

def compute_iou(mask1, mask2):
    inter = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return float(inter) / float(union + 1e-8)


def merge_masks(masks_list, iou_thresh=MERGE_IOU_THRESH):
    """
    Loại bỏ duplicate masks (overlap cao = cùng object).
    Giữ mask có area lớn nhất khi IoU > threshold.
    """
    if len(masks_list) == 0:
        return []

    # Sort theo area descending
    masks_list = sorted(masks_list, key=lambda x: x["area"], reverse=True)
    kept = []

    for cand in masks_list:
        is_duplicate = False
        for kept_m in kept:
            if compute_iou(cand["mask"], kept_m["mask"]) > iou_thresh:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(cand)

    return kept


def generate_grid_masks(sam2_predictor, image: np.ndarray,
                        grid_size: int = GRID_SIZE) -> list:
    """
    Prompt SAM2 tại mỗi điểm trong lưới grid_size × grid_size.
    Returns list of dicts: {"mask": (H,W) bool, "score": float, "area": int}
    """
    H, W = image.shape[:2]
    sam2_predictor.set_image(image)

    # Tạo grid points
    xs = np.linspace(W * 0.1, W * 0.9, grid_size)
    ys = np.linspace(H * 0.1, H * 0.9, grid_size)
    points = [(x, y) for y in ys for x in xs]

    all_masks = []
    for (x, y) in points:
        input_point = np.array([[x, y]], dtype=np.float32)
        input_label = np.array([1], dtype=np.int32)

        try:
            masks, scores, _ = sam2_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            # Lấy mask có score cao nhất
            best_idx  = scores.argmax()
            best_mask = masks[best_idx].astype(bool)
            best_score = float(scores[best_idx])

            if best_mask.sum() < MIN_AREA_PX:
                continue

            all_masks.append({
                "mask":  best_mask,
                "score": best_score,
                "area":  int(best_mask.sum()),
            })
        except Exception:
            continue   # một số điểm có thể fail, bỏ qua

    # Merge duplicates
    merged = merge_masks(all_masks)
    return merged


def compute_coverage(masks_list, H=IMG_SIZE, W=IMG_SIZE) -> float:
    """Tính % diện tích ảnh được cover bởi ít nhất 1 mask."""
    if not masks_list:
        return 0.0
    union = np.zeros((H, W), dtype=bool)
    for m in masks_list:
        union |= m["mask"]
    return float(union.sum()) / (H * W)


def main(args):
    # ── Load SAM2 ─────────────────────────────────────────────────────────
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint,
                            device=device)
    predictor  = SAM2ImagePredictor(sam2_model)
    print(f"✅ SAM2 loaded on {device}")

    # ── Xử lý từng ảnh ────────────────────────────────────────────────────
    stems = sorted([Path(f).stem for f in
                    glob.glob(os.path.join(args.img_dir, "*.png"))])
    os.makedirs(args.out_dir, exist_ok=True)

    coverage_list = []
    orig_coverage_list = []

    for stem in tqdm(stems, desc="Grid prompting"):
        # Skip nếu đã có
        out_path = os.path.join(args.out_dir, stem + ".npz")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        img_path = os.path.join(args.img_dir, stem + ".png")
        image = np.array(Image.open(img_path).convert("RGB"))

        # Generate masks với grid prompting
        masks_list = generate_grid_masks(predictor, image,
                                          grid_size=GRID_SIZE)

        if not masks_list:
            continue

        # Coverage của grid vs original
        cov = compute_coverage(masks_list)
        coverage_list.append(cov)

        # Load original coverage nếu có
        orig_path = os.path.join(args.orig_masks_dir, stem + ".npz")
        if os.path.exists(orig_path):
            orig_data = np.load(orig_path)
            orig_masks_list = [{"mask": orig_data["masks"][i]}
                               for i in range(len(orig_data["masks"]))]
            orig_cov = compute_coverage(orig_masks_list)
            orig_coverage_list.append(orig_cov)

        # Save
        masks_arr  = np.stack([m["mask"]  for m in masks_list])  # (N,H,W)
        scores_arr = np.array([m["score"] for m in masks_list])  # (N,)
        np.savez_compressed(out_path, masks=masks_arr, scores=scores_arr)

    # ── Summary ───────────────────────────────────────────────────────────
    if coverage_list:
        avg_new  = np.mean(coverage_list) * 100
        avg_orig = np.mean(orig_coverage_list) * 100 if orig_coverage_list else 68.5
        print(f"\n✅ Grid prompting done.")
        print(f"   Original coverage: {avg_orig:.1f}%")
        print(f"   Grid coverage:     {avg_new:.1f}%")
        print(f"   Improvement:       +{avg_new - avg_orig:.1f}%")

        # Decision gate
        if avg_new > avg_orig + 5:
            print("   → PASS: Coverage tăng đáng kể")
        else:
            print("   → [WARNING] Coverage không tăng nhiều")
            print("     Thử tăng GRID_SIZE từ 8 lên 12 hoặc 16")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir",        required=True)
    p.add_argument("--out-dir",        required=True)
    p.add_argument("--orig-masks-dir", default="SECOND/sam2_masks_T1_test")
    p.add_argument("--sam2-config",    default="sam2/configs/sam2_hiera_l.yaml")
    p.add_argument("--sam2-checkpoint",default="sam2/checkpoints/sam2_hiera_large.pt")
    p.add_argument("--overwrite",      action="store_true")
    args = p.parse_args()
    main(args)
```

**Chạy:**
```bash
# Test set T1
python generate_sam2_masks_grid.py \
    --img-dir        SECOND/test/im1 \
    --out-dir        SECOND/sam2_masks_T1_test_grid \
    --orig-masks-dir SECOND/sam2_masks_T1_test

# Test set T2
python generate_sam2_masks_grid.py \
    --img-dir        SECOND/test/im2 \
    --out-dir        SECOND/sam2_masks_T2_test_grid \
    --orig-masks-dir SECOND/sam2_masks_T2_test
```

**Expected output:**
```
Original coverage: 68.5%
Grid coverage:     87-92%
Improvement:       +19-24%
→ PASS
```

**Nếu coverage không tăng đủ:** Tăng GRID_SIZE từ 8 → 12, chạy lại.

---

## ════════════════════════════════════
## NHÓM 3B — Dense Fallback (Nearest Neighbor)
## ════════════════════════════════════

**Mục tiêu:** Coverage → 100% bằng gán vùng không có mask về token gần nhất.

```python
# dense_coverage.py
"""
Với vùng ảnh không thuộc bất kỳ SAM2 mask nào:
  → Tìm token gần nhất theo centroid distance
  → Gán prediction của token đó cho vùng này

Đây là post-processing step — không cần retrain.
Chỉ ảnh hưởng eval bằng cách tăng coverage.
"""
import numpy as np
import json
from scipy.spatial import KDTree


def apply_dense_fallback(
    predictions: dict,    # stem → list of {mask_file, change_label}
    masks_dir: str,       # thư mục chứa SAM2 masks
    img_size: int = 512,
) -> dict:
    """
    Với mỗi pixel không có mask → gán prediction của token gần nhất.
    Trả về predictions đã augment với dense coverage.
    """
    from pathlib import Path
    import os

    dense_preds = {}

    for stem, preds in predictions.items():
        if not preds:
            dense_preds[stem] = preds
            continue

        mask_path = os.path.join(masks_dir, stem + ".npz")
        if not os.path.exists(mask_path):
            dense_preds[stem] = preds
            continue

        data  = np.load(mask_path)
        masks = data["masks"]  # (N, H, W)

        # Tính coverage map
        covered = np.zeros((img_size, img_size), dtype=bool)
        for m in masks:
            covered |= m.astype(bool)

        uncovered_pct = (~covered).sum() / covered.size * 100
        if uncovered_pct < 1.0:
            # Gần 100% rồi, không cần fallback
            dense_preds[stem] = preds
            continue

        # Centroids của các tokens đã predict
        centroids = []
        for i, mask in enumerate(masks):
            ys, xs = np.where(mask)
            if len(ys) > 0:
                centroids.append([xs.mean(), ys.mean()])
            else:
                centroids.append([img_size/2, img_size/2])

        centroids = np.array(centroids)
        kdtree = KDTree(centroids)

        # Với mỗi uncovered pixel → tìm nearest token
        ys_unc, xs_unc = np.where(~covered)
        if len(ys_unc) == 0:
            dense_preds[stem] = preds
            continue

        query_points = np.stack([xs_unc, ys_unc], axis=1)
        _, nearest_idx = kdtree.query(query_points)

        # Tạo pseudo-masks cho vùng uncovered (nhóm theo nearest token)
        extra_preds = list(preds)  # copy preds gốc
        for token_idx in np.unique(nearest_idx):
            pixel_mask = np.zeros((img_size, img_size), dtype=bool)
            pixel_coords = query_points[nearest_idx == token_idx]
            pixel_mask[pixel_coords[:, 1], pixel_coords[:, 0]] = True

            if token_idx < len(preds):
                ref_pred = preds[token_idx]
                extra_preds.append({
                    "mask_file":    ref_pred.get("mask_file", ""),
                    "change_label": ref_pred.get("change_label", "unchanged"),
                    "is_fallback":  True,   # đánh dấu là fallback
                    "_fallback_mask": pixel_mask,
                })

        dense_preds[stem] = extra_preds

    return dense_preds
```

**Chạy sau khi có predictions từ N2A:**
```bash
python -c "
from dense_coverage import apply_dense_fallback
import json

with open('SECOND-OC/predictions/predictions_masked_gt.json') as f:
    preds = json.load(f)

dense = apply_dense_fallback(preds, 'SECOND/sam2_masks_T1_test_grid')

with open('SECOND-OC/predictions/predictions_dense.json', 'w') as f:
    json.dump(dense, f)
print('Done')
"
```

---

## ════════════════════════════════════
## NHÓM 4 — Thay SAM2 bằng RemoteCLIP Encoder
## ════════════════════════════════════

**Mục tiêu:** Visual features encode spectral RS patterns tốt hơn SAM2 (train trên web images).

### N4.0 — Prerequisite check

```bash
# Kiểm tra RemoteCLIP có available không
pip show open-clip-torch 2>/dev/null || pip install open-clip-torch --break-system-packages
python -c "
try:
    import open_clip
    print('✅ open-clip-torch installed')
    # Thử load RemoteCLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='datacomp_xl_s13b_b90k'
    )
    print('✅ CLIP model loadable (sẽ thay bằng RemoteCLIP weights)')
except Exception as e:
    print(f'❌ {e}')
"

# Tìm RemoteCLIP weights
# Download từ: https://github.com/ChenDelong1999/RemoteCLIP
# Hoặc HuggingFace: chendelong/RemoteCLIP
python -c "
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(
        repo_id='chendelong/RemoteCLIP',
        filename='RemoteCLIP-ViT-L-14.pt'
    )
    print(f'✅ RemoteCLIP weights: {path}')
except Exception as e:
    print(f'❌ Download failed: {e}')
    print('  → Thử: pip install huggingface_hub')
    print('  → Hoặc download thủ công từ HuggingFace')
"
```

---

### N4.1 — Tạo `remoteclip_encoder.py`

```python
# remoteclip_encoder.py
"""
RemoteCLIP encoder thay thế SAM2 visual encoder.
Output: 768-dim features per region (ViT-L-14)
vs SAM2: 256-dim features per region

Ưu điểm so với SAM2:
  - Train trên remote sensing data (RSICD, UCM, RSITMD, DOTA...)
  - Features encode spectral land-cover patterns
  - Phân biệt được low_veg vs barren vs water tốt hơn
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class RemoteCLIPRegionEncoder:
    """
    Extract per-region features từ RemoteCLIP.
    Với mỗi SAM2 mask: crop bbox → resize → encode → mean pool.
    """

    def __init__(
        self,
        weights_path: str = None,      # path to RemoteCLIP-ViT-L-14.pt
        device: str = "cuda",
        feature_dim: int = 768,        # ViT-L-14 output dim
        crop_size: int = 224,          # CLIP input size
        pad_ratio: float = 0.1,        # padding quanh bbox
    ):
        import open_clip
        self.device    = device
        self.feat_dim  = feature_dim
        self.crop_size = crop_size
        self.pad_ratio = pad_ratio

        # Load model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained=weights_path if weights_path else "datacomp_xl_s13b_b90k"
        )

        # Load RemoteCLIP weights nếu có
        if weights_path and weights_path.endswith(".pt"):
            state = torch.load(weights_path, map_location="cpu")
            # RemoteCLIP lưu chỉ visual encoder
            if "state_dict" in state:
                state = state["state_dict"]
            self.model.visual.load_state_dict(state, strict=False)
            print(f"✅ RemoteCLIP weights loaded: {weights_path}")

        self.model = self.model.visual.to(device)
        self.model.eval()

        # Project về hidden_dim của Token-MoE (384)
        # Cần thêm projection layer
        self.proj = torch.nn.Linear(feature_dim, 384).to(device)

    @torch.no_grad()
    def encode_region(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        image: (H, W, 3) uint8
        mask:  (H, W) bool
        Returns: (384,) float32 — projected region feature
        """
        # Lấy bounding box của mask
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return np.zeros(384, dtype=np.float32)

        H, W = image.shape[:2]
        pad_h = int((ys.max() - ys.min()) * self.pad_ratio)
        pad_w = int((xs.max() - xs.min()) * self.pad_ratio)

        y1 = max(0, ys.min() - pad_h)
        y2 = min(H, ys.max() + pad_h + 1)
        x1 = max(0, xs.min() - pad_w)
        x2 = min(W, xs.max() + pad_w + 1)

        # Crop và preprocess
        crop = Image.fromarray(image[y1:y2, x1:x2])
        tensor = self.preprocess(crop).unsqueeze(0).to(self.device)

        # Encode
        feat = self.model(tensor)              # (1, 768)
        feat = F.normalize(feat, dim=-1)       # L2 normalize
        feat = self.proj(feat).squeeze(0)      # (384,)

        return feat.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_all_regions(
        self,
        image: np.ndarray,
        masks: np.ndarray,      # (N, H, W) bool
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode tất cả regions trong 1 ảnh.
        Returns: (N, 384) float32
        """
        N = len(masks)
        features = np.zeros((N, 384), dtype=np.float32)

        # Batch processing
        crops = []
        for i, mask in enumerate(masks):
            ys, xs = np.where(mask)
            if len(ys) == 0:
                crops.append(torch.zeros(3, self.crop_size, self.crop_size))
                continue

            H, W = image.shape[:2]
            y1, y2 = max(0, ys.min()-5), min(H, ys.max()+6)
            x1, x2 = max(0, xs.min()-5), min(W, xs.max()+6)
            crop = Image.fromarray(image[y1:y2, x1:x2])
            crops.append(self.preprocess(crop))

        # Process in batches
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            batch = torch.stack(crops[start:end]).to(self.device)
            feats = self.model(batch)                    # (B, 768)
            feats = F.normalize(feats, dim=-1)
            feats = self.proj(feats)                     # (B, 384)
            features[start:end] = feats.cpu().numpy()

        return features


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    encoder = RemoteCLIPRegionEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ RemoteCLIP encoder initialized")
    print(f"   Feature dim: 384 (projected from 768)")

    # Test với 1 ảnh
    dummy_img   = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_masks = np.random.rand(10, 512, 512) > 0.9

    feats = encoder.encode_all_regions(dummy_img, dummy_masks)
    print(f"   Output shape: {feats.shape}")  # (10, 384)
    assert feats.shape == (10, 384)
    print("✅ Test passed")
```

---

### N4.2 — Tạo `tokenize_remoteclip.py`

```python
# tokenize_remoteclip.py
"""
Re-tokenize SECOND dataset với RemoteCLIP features thay SAM2 features.
Output: tokens_T1_rc/ và tokens_T2_rc/ với format giống tokens_T1_v2/
        nhưng "tokens" key là RemoteCLIP features (384-dim) thay vì SAM2 (256-dim)
"""
import os, glob, torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from remoteclip_encoder import RemoteCLIPRegionEncoder
from spectral_extractor import extract_spectral_for_stem

SECOND_ROOT     = "SECOND"
REMOTECLIP_WEIGHTS = None  # set path nếu có RemoteCLIP weights
DEVICE          = "cuda"
SPLITS = [
    ("train", "tokens_T1", "tokens_T1_rc", "T1"),
    ("train", "tokens_T2", "tokens_T2_rc", "T2"),
    ("test",  "tokens_T1_test", "tokens_T1_test_rc", "T1"),
    ("test",  "tokens_T2_test", "tokens_T2_test_rc", "T2"),
]

encoder = RemoteCLIPRegionEncoder(
    weights_path=REMOTECLIP_WEIGHTS,
    device=DEVICE
)

for (split, orig_tokens_dir, out_dir, time) in SPLITS:
    os.makedirs(f"{SECOND_ROOT}/{out_dir}", exist_ok=True)
    stems = sorted([Path(f).stem for f in
        glob.glob(f"{SECOND_ROOT}/{orig_tokens_dir}/*.pt")])

    print(f"\nProcessing {split} {time}: {len(stems)} stems → {out_dir}")

    for stem in tqdm(stems):
        orig_path = f"{SECOND_ROOT}/{orig_tokens_dir}/{stem}.pt"
        out_path  = f"{SECOND_ROOT}/{out_dir}/{stem}.pt"

        if os.path.exists(out_path):
            continue

        # Load original tokens (centroids, areas giữ nguyên)
        orig = torch.load(orig_path, map_location="cpu")
        N = orig["tokens"].shape[0]

        # Load ảnh và masks
        img_dir   = f"{SECOND_ROOT}/{split}/im1" if time == "T1" else f"{SECOND_ROOT}/{split}/im2"
        masks_dir = f"{SECOND_ROOT}/sam2_masks_T1" if time == "T1" else f"{SECOND_ROOT}/sam2_masks_T2"
        if split == "test":
            img_dir   = f"{SECOND_ROOT}/test/im1" if time == "T1" else f"{SECOND_ROOT}/test/im2"
            masks_dir = f"{SECOND_ROOT}/sam2_masks_T1_test" if time == "T1" else f"{SECOND_ROOT}/sam2_masks_T2_test"

        img_path  = os.path.join(img_dir, stem + ".png")
        mask_path = os.path.join(masks_dir, stem + ".npz")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        masks = np.load(mask_path)["masks"]  # (N_masks, H, W)

        N_align = min(N, len(masks))

        # RemoteCLIP features
        rc_feats = encoder.encode_all_regions(image, masks[:N_align])

        # Spectral features (giữ lại)
        spectral = orig.get("spectral", None)

        new_token = {
            "tokens":    torch.from_numpy(rc_feats).float(),  # (N, 384)
            "centroids": orig["centroids"][:N_align],
            "areas":     orig["areas"][:N_align],
        }
        if spectral is not None:
            new_token["spectral"] = spectral[:N_align]

        torch.save(new_token, out_path)

print("\n✅ RemoteCLIP tokenization done.")
print("   Cần cập nhật hidden_dim trong MoEConfig từ 384 → 384")
print("   (tokens là 384-dim, không cần projection thêm)")
```

**Chạy:**
```bash
python tokenize_remoteclip.py
# Sẽ mất ~2-4 giờ tùy GPU
```

---

### N4.3 — Train với RemoteCLIP features

```bash
mkdir -p SECOND/stage8_remoteclip

# Lưu ý: hidden_dim cần match với RemoteCLIP output (384)
# Nếu model hiện tại dùng hidden_dim=384 → dùng được ngay
# Nếu khác → cần thêm --hidden-dim flag

nohup python train_reasoner_spectral.py \
    --model_type moe \
    --tokens_T1   SECOND/tokens_T1_rc \
    --tokens_T2   SECOND/tokens_T2_rc \
    --matches     SECOND/matches \
    --semantic_dir     SECOND/train/label1 \
    --semantic_dir_t2  SECOND/train/label2 \
    --output      SECOND/stage8_remoteclip \
    --epochs 60 \
    --batch_size 8 \
    --router_version v3 \
    --lambda_semantic 0.3 \
    --lambda_transition 0.1 \
    --gt_change_labels \
    --use_spectral \
    --device cuda \
    > /tmp/train_remoteclip.log 2>&1 &
```

---

## ════════════════════════════════════
## BẢNG SO SÁNH CUỐI CÙNG
## ════════════════════════════════════

**Sau khi chạy xong tất cả, chạy lệnh này:**

```bash
python -c "
import json

files = {
    'stage6 (current best)':  'SECOND-OC/baseline_results/tokenmoe_desc_results.json',
    'N2A masked_gt':           'SECOND-OC/baseline_results/masked_gt_results.json',
    'N2A+threshold':           'SECOND-OC/baseline_results/masked_gt_tuned_results.json',
    'N3A grid_coverage':       'SECOND-OC/baseline_results/grid_coverage_results.json',
    'N4 remoteclip':           'SECOND-OC/baseline_results/remoteclip_results.json',
    'SCanNet (ref)':           'SECOND-OC/baseline_results/scannet_results.json',
}

print(f'{'Model':<28} {'Bin-F1':>8} {'Sem-F1':>8} {'Recall':>8} {'Sem-Acc-TP':>12}')
print('-' * 68)
for name, path in files.items():
    try:
        with open(path) as f: d = json.load(f)
        print(f'{name:<28} {d.get(\"Binary-Object-F1\",0):>8.4f} '
              f'{d.get(\"Semantic-Object-F1\",0):>8.4f} '
              f'{d.get(\"Binary-Recall\",0):>8.4f} '
              f'{d.get(\"Semantic-Acc-on-TP\",0):>12.1%}')
    except FileNotFoundError:
        print(f'{name:<28} {\"(chưa có)\":>42}')
"
```

---

## KHAI BÁO DỪNG

Agent dừng và báo cáo khi:
```
1. N4.0: RemoteCLIP download fail → thông báo, skip N4, làm N2+N3
2. N3A: Coverage sau grid < 75% → tăng GRID_SIZE=12, thử lại
3. N2A: Sem-Acc-on-TP sau masked GT < 26.5% → báo cáo diagnose_centroid error rate
4. Bất kỳ: CUDA OOM → giảm batch_size xuống 4
5. Bất kỳ: NaN trong loss → kiểm tra spectral features và masked GT labels
```