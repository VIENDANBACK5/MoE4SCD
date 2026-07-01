# SECOND-OC: Agent-Executable Implementation Plan
> Mỗi bước được viết để AI coding agent có thể chạy trực tiếp, không cần hỏi thêm.
> Mỗi bước có: input rõ ràng → code đầy đủ → output mong đợi → assert validation.

---

## Cấu trúc thư mục giả định (agent phải verify ở Phase 0)

```
SECOND/
└── test/
    ├── im1/          # RGB T1, *.png, 512×512
    ├── im2/          # RGB T2, *.png, 512×512
    ├── label1/       # GT semantic T1, *.png pixel-level
    ├── label2/       # GT semantic T2, *.png pixel-level
    ├── tokens_T1_test/  # SAM2 tokens T1 (format cần sniff ở Phase 0)
    ├── tokens_T2_test/  # SAM2 tokens T2
    └── matches_test/    # SAM2 cross-temporal matches

SECOND-OC/            # Thư mục output (tạo mới)
├── annotations/
├── masks/
│   ├── T1/
│   └── T2/
├── tier1/
└── eval/
```

---

## ════════════════════════════════════
## PHASE 0 — Environment & Format Sniffing
## ════════════════════════════════════

### Bước 0.1 — Tạo thư mục output + verify split

**File:** `phase0_verify.py`

```python
#!/usr/bin/env python3
"""
Phase 0: Verify data layout, split size, SAM2 token format, GT label encoding.
Outputs: phase0_report.json (machine-readable), phase0_report.txt (human-readable)
Agent: chạy file này TRƯỚC mọi phase khác, đọc output trước khi tiếp tục.
"""
import os, json, glob, struct
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

# ── Cấu hình đường dẫn (agent sửa nếu cần) ──────────────────────────────────
DATA_ROOT   = "SECOND/test"          # sửa nếu cần
OUT_ROOT    = "SECOND-OC"
REPORT_PATH = "phase0_report.json"
# ─────────────────────────────────────────────────────────────────────────────

def sniff_token_format(token_dir: str) -> dict:
    """Tự động phát hiện format file trong token_dir."""
    files = list(Path(token_dir).iterdir())
    if not files:
        return {"format": "EMPTY", "count": 0}

    sample = files[0]
    ext = sample.suffix.lower()

    info = {"count": len(files), "ext": ext, "sample_name": sample.name}

    if ext == ".npy":
        arr = np.load(sample, allow_pickle=False)
        info.update({"format": "npy", "shape": list(arr.shape), "dtype": str(arr.dtype)})
        # Phân loại: binary mask hoặc feature tensor
        if arr.ndim == 2 and arr.dtype == bool:
            info["subtype"] = "binary_mask_single"
        elif arr.ndim == 3 and arr.shape[0] < 200:
            info["subtype"] = "mask_stack"      # (N_masks, H, W)
        elif arr.ndim == 3 and arr.shape[0] >= 200:
            info["subtype"] = "feature_tensor"  # (C, H, W)
        else:
            info["subtype"] = "unknown"

    elif ext == ".npz":
        arr = np.load(sample, allow_pickle=False)
        info.update({"format": "npz", "keys": list(arr.keys())})

    elif ext in (".pt", ".pth"):
        try:
            import torch
            data = torch.load(sample, map_location="cpu")
            if isinstance(data, dict):
                info.update({"format": "torch_dict", "keys": list(data.keys())})
            elif isinstance(data, torch.Tensor):
                info.update({"format": "torch_tensor", "shape": list(data.shape)})
            else:
                info.update({"format": "torch_other", "type": type(data).__name__})
        except Exception as e:
            info.update({"format": "torch_ERROR", "error": str(e)})

    elif ext == ".pkl":
        import pickle
        with open(sample, "rb") as f:
            data = pickle.load(f)
        info.update({"format": "pkl", "type": type(data).__name__})
        if isinstance(data, (list, dict)):
            info["len"] = len(data)

    elif ext == ".json":
        with open(sample) as f:
            data = json.load(f)
        info.update({"format": "json", "top_keys": list(data.keys())[:5] if isinstance(data, dict) else "list"})

    else:
        info["format"] = f"UNKNOWN_EXT_{ext}"

    return info


def sniff_gt_label_encoding(label_dir: str, n_sample: int = 20) -> dict:
    """Tìm ra unique pixel values trong GT labels và số lớp."""
    all_values = set()
    files = sorted(Path(label_dir).glob("*.png"))[:n_sample]
    for f in files:
        arr = np.array(Image.open(f))
        all_values.update(np.unique(arr).tolist())

    vals = sorted(all_values)
    return {
        "unique_pixel_values": vals,
        "n_classes_including_nochange": len(vals),
        "min": min(vals),
        "max": max(vals),
        "note": ("0=no-change + 1-6=semantic" if 0 in vals and max(vals) <= 6
                 else "check manually — unexpected values")
    }


def main():
    report = {}

    # ── 1. Kiểm tra thư mục tồn tại ─────────────────────────────────────────
    dirs = {k: os.path.join(DATA_ROOT, k) for k in
            ["im1", "im2", "label1", "label2",
             "tokens_T1_test", "tokens_T2_test", "matches_test"]}
    report["dirs_exist"] = {k: os.path.isdir(v) for k, v in dirs.items()}

    missing = [k for k, v in report["dirs_exist"].items() if not v]
    if missing:
        print(f"[ERROR] Thiếu thư mục: {missing}")
        print("  → Agent: sửa DATA_ROOT trong phase0_verify.py rồi chạy lại")

    # ── 2. Đếm số cặp ảnh ────────────────────────────────────────────────────
    stems_im1    = {Path(f).stem for f in glob.glob(os.path.join(dirs["im1"], "*.png"))}
    stems_label1 = {Path(f).stem for f in glob.glob(os.path.join(dirs["label1"], "*.png"))}
    common_stems = sorted(stems_im1 & stems_label1)

    report["n_test_pairs"]     = len(common_stems)
    report["split_check_1694"] = len(common_stems) == 1694
    report["split_check_593"]  = len(common_stems) == 593
    report["sample_stems"]     = common_stems[:5]

    print(f"[0.2] Test pairs found: {len(common_stems)}")
    if report["split_check_1694"]:
        print("  ✅ Đúng 1694 pairs — khớp split TaCo/ChangeStar2/GSTM-SCD")
    elif report["split_check_593"]:
        print("  ⚠️  593 pairs — là split SCanNet/Bi-SRNet cũ (2022-)")
        print("  → Phase 4 baseline: chỉ dùng TaCo/ChangeStar2 nếu có thêm 1694-split numbers")
    else:
        print(f"  ⚠️  {len(common_stems)} pairs — không khớp split nào đã biết, kiểm tra lại")

    # ── 3. Kiểm tra kích thước ảnh ───────────────────────────────────────────
    sample_im = Image.open(os.path.join(dirs["im1"], common_stems[0] + ".png"))
    report["image_size"] = list(sample_im.size)  # (W, H)
    assert sample_im.size == (512, 512), f"Ảnh không phải 512×512: {sample_im.size}"
    print(f"[0.3] Image size: {sample_im.size} ✅")

    # ── 4. Sniff GT label encoding ────────────────────────────────────────────
    report["gt_label_encoding"] = sniff_gt_label_encoding(dirs["label1"])
    enc = report["gt_label_encoding"]
    print(f"[0.4] GT label unique values: {enc['unique_pixel_values']}")
    print(f"      Ghi chú: {enc['note']}")

    # ── 5. Sniff SAM2 token format ────────────────────────────────────────────
    report["tokens_T1_format"] = sniff_token_format(dirs["tokens_T1_test"])
    report["tokens_T2_format"] = sniff_token_format(dirs["tokens_T2_test"])
    report["matches_format"]   = sniff_token_format(dirs["matches_test"])

    print(f"[0.5] tokens_T1 format: {report['tokens_T1_format']}")
    print(f"[0.5] tokens_T2 format: {report['tokens_T2_format']}")
    print(f"[0.5] matches format:   {report['matches_format']}")

    # ── 6. Kiểm tra 1 sample end-to-end ─────────────────────────────────────
    stem = common_stems[0]
    im1_path  = os.path.join(dirs["im1"],    stem + ".png")
    lab1_path = os.path.join(dirs["label1"], stem + ".png")
    lab2_path = os.path.join(dirs["label2"], stem + ".png")

    lab1 = np.array(Image.open(lab1_path))
    lab2 = np.array(Image.open(lab2_path))
    report["sample_check"] = {
        "stem": stem,
        "lab1_shape": list(lab1.shape),
        "lab2_shape": list(lab2.shape),
        "lab1_unique": np.unique(lab1).tolist(),
        "lab2_unique": np.unique(lab2).tolist(),
    }
    print(f"[0.6] Sample {stem}: lab1 unique={np.unique(lab1).tolist()}, "
          f"lab2 unique={np.unique(lab2).tolist()}")

    # ── 7. Tạo thư mục output ────────────────────────────────────────────────
    for d in [OUT_ROOT,
              f"{OUT_ROOT}/annotations",
              f"{OUT_ROOT}/masks/T1",
              f"{OUT_ROOT}/masks/T2",
              f"{OUT_ROOT}/tier1",
              f"{OUT_ROOT}/eval"]:
        os.makedirs(d, exist_ok=True)
    print(f"[0.7] Output dirs created under {OUT_ROOT}/")

    # ── 8. Ghi report ─────────────────────────────────────────────────────────
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Phase 0 done. Report saved: {REPORT_PATH}")
    print("   Agent: đọc report này trước khi chạy Phase 1.")

    # ── 9. In quyết định cho agent ────────────────────────────────────────────
    fmt = report["tokens_T1_format"].get("format", "UNKNOWN")
    subtype = report["tokens_T1_format"].get("subtype", "")
    print(f"\n[AGENT DECISION] SAM2 token format = '{fmt}', subtype = '{subtype}'")
    if fmt == "npy" and subtype in ("mask_stack", "binary_mask_single"):
        print("  → Dùng nhánh A (Phase 1): load mask trực tiếp từ .npy")
    elif fmt in ("torch_dict", "torch_tensor"):
        print("  → Dùng nhánh B (Phase 1): extract masks từ torch dict")
    elif fmt == "npz":
        print("  → Dùng nhánh C (Phase 1): load từ .npz, check key 'masks' hoặc 'segmentation'")
    elif fmt == "json":
        print("  → Dùng nhánh D (Phase 1): parse JSON masks (COCO-style RLE hoặc polygon)")
    else:
        print(f"  → ⚠️  Format chưa biết: {fmt}. Agent: kiểm tra file thủ công rồi báo cáo.")

    return report


if __name__ == "__main__":
    main()
```

**Chạy:** `python phase0_verify.py`

**Expected output (nếu đúng):**
```
[0.2] Test pairs found: 1694  ✅
[0.3] Image size: (512, 512)  ✅
[0.4] GT label unique values: [0, 1, 2, 3, 4, 5, 6]  (hoặc [0..5])
[0.5] tokens_T1 format: {'format': 'npy', 'subtype': 'mask_stack', ...}
[AGENT DECISION] → Dùng nhánh A
✅ Phase 0 done. Report saved: phase0_report.json
```

**Nếu format SAM2 không phải npy/mask:** agent dừng lại, paste `phase0_report.json` để xin hướng dẫn trước khi sang Phase 1.

---

## ════════════════════════════════════
## PHASE 1 — Object Instance Extraction
## ════════════════════════════════════

### Bước 1.0 — Đọc phase0_report.json, xác định CLASS_MAP

**File:** `config.py` (tạo 1 lần, import ở mọi phase)

```python
# config.py
import json

# ── Đọc Phase 0 report ───────────────────────────────────────────────────────
with open("phase0_report.json") as f:
    P0 = json.load(f)

DATA_ROOT  = "SECOND/test"
OUT_ROOT   = "SECOND-OC"
N_PAIRS    = P0["n_test_pairs"]        # 1694 hoặc 593
STEMS      = None                      # populated by loader

# ── GT Label encoding ─────────────────────────────────────────────────────────
# SECOND gốc: pixel=0 là no-change, 1-6 là 6 classes
# Một số implementation encode 0-5 (không có no-change class riêng)
GT_VALS = P0["gt_label_encoding"]["unique_pixel_values"]

if 0 in GT_VALS and max(GT_VALS) == 6:
    # Standard SECOND encoding
    NO_CHANGE_VAL = 0
    CLASS_MAP = {
        1: "non_veg_ground",   # non-vegetated ground surface (impervious + bare)
        2: "tree",
        3: "low_vegetation",
        4: "water",
        5: "buildings",
        6: "playgrounds",
    }
elif 0 in GT_VALS and max(GT_VALS) == 5:
    # Alternative: 0-5
    NO_CHANGE_VAL = 255   # không có trong label — mọi pixel đều có class
    CLASS_MAP = {
        0: "non_veg_ground",
        1: "tree",
        2: "low_vegetation",
        3: "water",
        4: "buildings",
        5: "playgrounds",
    }
else:
    raise ValueError(
        f"GT label values {GT_VALS} không khớp cấu hình biết. "
        "Agent: kiểm tra phase0_report.json > gt_label_encoding rồi sửa CLASS_MAP thủ công."
    )

# ── SAM2 token format (từ Phase 0) ───────────────────────────────────────────
TOKEN_FORMAT  = P0["tokens_T1_format"]["format"]    # 'npy', 'torch_dict', ...
TOKEN_SUBTYPE = P0["tokens_T1_format"].get("subtype", "")
TOKEN_SHAPE   = P0["tokens_T1_format"].get("shape", [])

# ── Hyperparams ───────────────────────────────────────────────────────────────
MIN_AREA_PX = 100          # bỏ qua object nhỏ hơn 100 pixels
DOMINANT_CLASS_THRESH = 0.5  # mask phải có >= 50% pixel thuộc 1 class
```

### Bước 1.1 — Loader SAM2 masks (có 4 nhánh theo format)

**File:** `sam2_loader.py`

```python
# sam2_loader.py
"""
Load SAM2 masks cho 1 stem, trả về list of dict:
  [{"mask": np.ndarray bool (512,512), "score": float, "mask_id": int}, ...]
Tự động chọn nhánh theo TOKEN_FORMAT từ config.py.
"""
import os
import numpy as np
from pathlib import Path
from config import TOKEN_FORMAT, TOKEN_SUBTYPE, TOKEN_SHAPE, DATA_ROOT


def load_sam2_masks(stem: str, time: str = "T1") -> list[dict]:
    """
    stem: e.g. "00004"
    time: "T1" hoặc "T2"
    Returns: list of {"mask": bool array (512,512), "score": float, "mask_id": int}
    """
    token_dir = os.path.join(DATA_ROOT,
                             "tokens_T1_test" if time == "T1" else "tokens_T2_test")

    # ── NHÁNH A: .npy mask stack ──────────────────────────────────────────────
    if TOKEN_FORMAT == "npy" and TOKEN_SUBTYPE in ("mask_stack", "binary_mask_single"):
        path = os.path.join(token_dir, stem + ".npy")
        if not os.path.exists(path):
            return []
        data = np.load(path, allow_pickle=False)
        # data shape: (N_masks, H, W) với dtype bool hoặc uint8
        if data.ndim == 2:
            data = data[np.newaxis, ...]   # single mask → (1, H, W)
        masks = []
        for i in range(data.shape[0]):
            m = data[i].astype(bool)
            masks.append({"mask": m, "score": 1.0, "mask_id": i})
        return masks

    # ── NHÁNH B: .npy với allow_pickle (chứa list of dict) ───────────────────
    if TOKEN_FORMAT == "npy" and TOKEN_SUBTYPE == "unknown":
        path = os.path.join(token_dir, stem + ".npy")
        data = np.load(path, allow_pickle=True).item()   # hoặc .tolist()
        if isinstance(data, list):
            masks = []
            for i, item in enumerate(data):
                if isinstance(item, dict) and "segmentation" in item:
                    m = np.array(item["segmentation"]).astype(bool)
                    masks.append({"mask": m,
                                  "score": float(item.get("stability_score", 1.0)),
                                  "mask_id": i})
            return masks
        raise ValueError(f"Nhánh B: format dict không nhận ra. Keys: {list(data.keys())}")

    # ── NHÁNH C: .npz ─────────────────────────────────────────────────────────
    if TOKEN_FORMAT == "npz":
        path = os.path.join(token_dir, stem + ".npz")
        data = np.load(path, allow_pickle=False)
        keys = list(data.keys())
        # Thử các key phổ biến
        mask_key = next((k for k in ["masks", "segmentation", "mask"] if k in keys), None)
        if mask_key is None:
            raise ValueError(f"Nhánh C: .npz không có key masks/segmentation. Keys: {keys}")
        arr = data[mask_key]  # (N, H, W)
        scores = data.get("scores", np.ones(arr.shape[0]))
        return [{"mask": arr[i].astype(bool), "score": float(scores[i]), "mask_id": i}
                for i in range(arr.shape[0])]

    # ── NHÁNH D: torch .pt/.pth ───────────────────────────────────────────────
    if TOKEN_FORMAT in ("torch_dict", "torch_tensor"):
        import torch
        path_pt  = os.path.join(token_dir, stem + ".pt")
        path_pth = os.path.join(token_dir, stem + ".pth")
        path = path_pt if os.path.exists(path_pt) else path_pth
        data = torch.load(path, map_location="cpu")
        if isinstance(data, torch.Tensor):
            arr = data.bool().numpy()
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            return [{"mask": arr[i], "score": 1.0, "mask_id": i}
                    for i in range(arr.shape[0])]
        if isinstance(data, dict):
            arr = data.get("masks", data.get("segmentation"))
            if arr is None:
                raise ValueError(f"Nhánh D torch dict, keys: {list(data.keys())}")
            if isinstance(arr, torch.Tensor):
                arr = arr.bool().numpy()
            scores = data.get("scores", np.ones(arr.shape[0]))
            if isinstance(scores, torch.Tensor):
                scores = scores.numpy()
            return [{"mask": arr[i], "score": float(scores[i]), "mask_id": i}
                    for i in range(arr.shape[0])]

    # ── NHÁNH E: JSON (COCO-style RLE hoặc polygon) ──────────────────────────
    if TOKEN_FORMAT == "json":
        import json
        from pycocotools import mask as mask_util
        path = os.path.join(token_dir, stem + ".json")
        with open(path) as f:
            anns = json.load(f)
        masks = []
        for i, ann in enumerate(anns):
            if "segmentation" in ann:
                seg = ann["segmentation"]
                if isinstance(seg, dict):   # RLE
                    m = mask_util.decode(seg).astype(bool)
                else:                       # polygon
                    from pycocotools.mask import frPyObjects, merge
                    rle = merge(frPyObjects(seg, 512, 512))
                    m = mask_util.decode(rle).astype(bool)
                masks.append({"mask": m,
                              "score": float(ann.get("score", 1.0)),
                              "mask_id": i})
        return masks

    raise NotImplementedError(
        f"Format '{TOKEN_FORMAT}' chưa được implement. "
        "Agent: báo cáo token format từ phase0_report.json."
    )
```

### Bước 1.2 — Extractor chính

**File:** `phase1_extract_instances.py`

```python
#!/usr/bin/env python3
"""
Phase 1: Extract object instances từ SAM2 masks ∩ GT labels.
Input:  tokens_T1_test/, tokens_T2_test/, label1/, label2/
Output: SECOND-OC/annotations/instances_T1.json
        SECOND-OC/annotations/instances_T2.json
        SECOND-OC/masks/T1/{stem}_{mask_id:03d}.npy   (binary mask per object)
        SECOND-OC/masks/T2/{stem}_{mask_id:03d}.npy
"""
import os, json, glob
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from config import (DATA_ROOT, OUT_ROOT, CLASS_MAP, NO_CHANGE_VAL,
                    MIN_AREA_PX, DOMINANT_CLASS_THRESH)
from sam2_loader import load_sam2_masks


def extract_instances_for_stem(stem: str, time: str) -> list[dict]:
    """
    Trả về list of instance dicts cho 1 ảnh (T1 hoặc T2).
    Không lưu mask ra disk — caller quyết định có lưu không.
    """
    label_dir = "label1" if time == "T1" else "label2"
    label_path = os.path.join(DATA_ROOT, label_dir, stem + ".png")

    if not os.path.exists(label_path):
        return []

    gt_label = np.array(Image.open(label_path))   # (512, 512) uint8

    # Load SAM2 masks
    sam2_masks = load_sam2_masks(stem, time)

    if not sam2_masks:
        return []

    instances = []
    for item in sam2_masks:
        mask = item["mask"]        # bool (512, 512)
        mask_id = item["mask_id"]
        score = item["score"]

        # ── Filter 1: diện tích tối thiểu ──────────────────────────────────
        area_px = int(mask.sum())
        if area_px < MIN_AREA_PX:
            continue

        # ── Gán class từ GT label ───────────────────────────────────────────
        pixels = gt_label[mask]

        # Bỏ no-change pixels (nếu dataset có)
        if NO_CHANGE_VAL != 255:
            semantic_pixels = pixels[pixels != NO_CHANGE_VAL]
        else:
            semantic_pixels = pixels

        if len(semantic_pixels) == 0:
            continue   # mask hoàn toàn nằm trong vùng no-change → bỏ qua

        values, counts = np.unique(semantic_pixels, return_counts=True)
        dominant_idx   = np.argmax(counts)
        dominant_class = int(values[dominant_idx])
        dominant_ratio = float(counts[dominant_idx]) / len(semantic_pixels)

        # ── Filter 2: class phải rõ ràng (dominant > threshold) ────────────
        if dominant_ratio < DOMINANT_CLASS_THRESH:
            continue   # vùng quá hỗn hợp class → bỏ qua (không phải 1 object)

        class_name = CLASS_MAP.get(dominant_class, f"class_{dominant_class}")

        # ── Tính bbox, centroid ─────────────────────────────────────────────
        ys, xs = np.where(mask)
        bbox     = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        centroid = [float(xs.mean()) / 512, float(ys.mean()) / 512]

        instance_id = f"{stem}_{time}_{mask_id:03d}"

        instances.append({
            "instance_id":       instance_id,
            "stem":              stem,
            "time":              time,
            "sam2_mask_id":      mask_id,
            "sam2_score":        round(score, 4),
            "class_id":          dominant_class,
            "class_name":        class_name,
            "dominant_ratio":    round(dominant_ratio, 3),
            "area_px":           area_px,
            "centroid":          centroid,
            "bbox":              bbox,      # [x_min, y_min, x_max, y_max]
            "mask_file":         f"masks/{time}/{instance_id}.npy",
        })

    return instances


def main():
    stems = sorted({Path(f).stem
                    for f in glob.glob(os.path.join(DATA_ROOT, "im1", "*.png"))})
    print(f"Processing {len(stems)} stems...")

    all_T1, all_T2 = {}, {}
    skipped_T1, skipped_T2 = 0, 0

    for stem in tqdm(stems, desc="Phase 1"):
        # ── T1 ──────────────────────────────────────────────────────────────
        insts_T1 = extract_instances_for_stem(stem, "T1")
        all_T1[stem] = insts_T1
        if not insts_T1:
            skipped_T1 += 1

        # ── T2 ──────────────────────────────────────────────────────────────
        insts_T2 = extract_instances_for_stem(stem, "T2")
        all_T2[stem] = insts_T2
        if not insts_T2:
            skipped_T2 += 1

        # ── Lưu masks ra disk (npy per instance) ────────────────────────────
        label1_path = os.path.join(DATA_ROOT, "label1", stem + ".png")
        label2_path = os.path.join(DATA_ROOT, "label2", stem + ".png")
        gt1 = np.array(Image.open(label1_path))
        gt2 = np.array(Image.open(label2_path))

        sam2_T1 = {item["mask_id"]: item["mask"] for item in load_sam2_masks(stem, "T1")}
        sam2_T2 = {item["mask_id"]: item["mask"] for item in load_sam2_masks(stem, "T2")}

        for inst in insts_T1:
            mask = sam2_T1[inst["sam2_mask_id"]]
            np.save(os.path.join(OUT_ROOT, inst["mask_file"]), mask)

        for inst in insts_T2:
            mask = sam2_T2[inst["sam2_mask_id"]]
            np.save(os.path.join(OUT_ROOT, inst["mask_file"]), mask)

    # ── Serialise (mask fields bỏ numpy arrays trước) ──────────────────────
    out_T1 = {"n_images": len(all_T1), "instances": all_T1}
    out_T2 = {"n_images": len(all_T2), "instances": all_T2}

    with open(f"{OUT_ROOT}/annotations/instances_T1.json", "w") as f:
        json.dump(out_T1, f, indent=2)
    with open(f"{OUT_ROOT}/annotations/instances_T2.json", "w") as f:
        json.dump(out_T2, f, indent=2)

    # ── Stats ────────────────────────────────────────────────────────────────
    total_T1 = sum(len(v) for v in all_T1.values())
    total_T2 = sum(len(v) for v in all_T2.values())
    print(f"\n✅ Phase 1 done.")
    print(f"   T1 instances: {total_T1}  ({skipped_T1} images with 0 instances)")
    print(f"   T2 instances: {total_T2}  ({skipped_T2} images with 0 instances)")
    print(f"   Expected: ~12,000–25,000 per split (depends on SAM2 mask density)")

    # ── Validation assertions ────────────────────────────────────────────────
    assert total_T1 > 5000, f"[FAIL] T1 instance count {total_T1} quá thấp, kiểm tra SAM2 loader"
    assert total_T2 > 5000, f"[FAIL] T2 instance count {total_T2} quá thấp"
    assert skipped_T1 < len(stems) * 0.2, \
        f"[FAIL] {skipped_T1}/{len(stems)} images không có instance — kiểm tra MIN_AREA_PX và loader"
    print("   ✅ Validation passed")


if __name__ == "__main__":
    main()
```

**Chạy:** `python phase1_extract_instances.py`

**Expected output:**
```
Processing 1694 stems...
Phase 1: 100%|████████| 1694/1694
✅ Phase 1 done.
   T1 instances: ~18,000   (< 50 images with 0 instances)
   T2 instances: ~18,000
   ✅ Validation passed
```

### Bước 1.3 — Visualize spot check (chạy trên 5 mẫu)

**File:** `phase1_visualize.py`

```python
#!/usr/bin/env python3
"""Quick visual check: vẽ instance masks lên ảnh, save ra PNG."""
import json, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import DATA_ROOT, OUT_ROOT, CLASS_MAP

COLORS = {
    "non_veg_ground": (200, 180, 140),
    "tree":           (34, 139, 34),
    "low_vegetation": (144, 238, 144),
    "water":          (30, 144, 255),
    "buildings":      (255, 99, 71),
    "playgrounds":    (255, 215, 0),
}

with open(f"{OUT_ROOT}/annotations/instances_T1.json") as f:
    data = json.load(f)

stems = random.sample(list(data["instances"].keys()), 5)

for stem in stems:
    img = Image.open(os.path.join(DATA_ROOT, "im1", stem + ".png")).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for inst in data["instances"][stem]:
        mask = np.load(os.path.join(OUT_ROOT, inst["mask_file"]))
        color = COLORS.get(inst["class_name"], (128, 128, 128)) + (80,)
        ys, xs = np.where(mask)
        for y, x in zip(ys[::3], xs[::3]):   # downsample for speed
            draw.point((x, y), fill=color)

        # Label bbox
        x1, y1, x2, y2 = inst["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (200,), width=2)
        draw.text((x1 + 2, y1 + 2), inst["class_name"][:4], fill=(255,255,255,255))

    result = Image.alpha_composite(img, overlay).convert("RGB")
    out_path = f"phase1_viz_{stem}.png"
    result.save(out_path)
    print(f"Saved: {out_path} ({len(data['instances'][stem])} instances)")
```

**Agent phải:** mở 5 PNG output, xác nhận từng tòa nhà = 1 instance riêng biệt (không bị gộp thành blob). Nếu blob → SAM2 mask chưa phân tách đúng → kiểm tra lại sam2_loader.py.

---

## ════════════════════════════════════
## PHASE 2 — Change Type Classification
## ════════════════════════════════════

### Bước 2.1 — Co-registered GT label lookup

**File:** `phase2_classify_changes.py`

```python
#!/usr/bin/env python3
"""
Phase 2: Với mỗi T1 instance, lookup class ở T2 tại cùng vị trí địa lý (co-registered).
Input:  instances_T1.json + GT label T2
Output: SECOND-OC/annotations/change_annotations.json

Không dùng IoU matching (đã chứng minh kém — AnyChange ablation F1=14.2 vs 41.8).
Dùng trực tiếp co-registration: mask T1 áp lên GT T2 tại cùng pixel coordinates.
"""
import os, json, glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from config import DATA_ROOT, OUT_ROOT, CLASS_MAP, NO_CHANGE_VAL

# ── Threshold ─────────────────────────────────────────────────────────────────
T2_CLASS_CONFIDENCE_MIN = 0.4   # mask phải có ≥ 40% pixel thuộc class T2 để gán
DISAPPEARED_NOCHANGE_THRESH = 0.8  # nếu >80% vùng mask là no-change ở T2 → disappeared

# ── Templates cho 30 class-transition combinations ────────────────────────────
# key: (class_t1, class_t2)
TRANSITION_TEMPLATES = {
    ("non_veg_ground", "buildings"):    "Bare land or impervious surface developed into buildings",
    ("non_veg_ground", "tree"):         "Bare ground revegetated with trees",
    ("non_veg_ground", "low_vegetation"): "Bare ground covered by low vegetation growth",
    ("non_veg_ground", "water"):        "Bare land flooded or converted to water body",
    ("non_veg_ground", "playgrounds"): "Bare ground converted to playground or sports facility",
    ("buildings", "non_veg_ground"):   "Buildings demolished, area converted to bare ground",
    ("buildings", "tree"):             "Buildings replaced by trees (unlikely — check annotation)",
    ("buildings", "low_vegetation"):   "Buildings cleared, vegetation regrowth observed",
    ("buildings", "water"):            "Buildings flooded or submerged",
    ("buildings", "playgrounds"):      "Buildings converted to playground or sports facility",
    ("tree", "non_veg_ground"):        "Tree cover cleared, area now bare ground",
    ("tree", "buildings"):             "Trees cleared and replaced by buildings",
    ("tree", "low_vegetation"):        "Tree canopy reduced to low vegetation",
    ("tree", "water"):                 "Tree area converted to water body",
    ("tree", "playgrounds"):           "Trees cleared for playground construction",
    ("low_vegetation", "non_veg_ground"): "Low vegetation removed, area now bare ground",
    ("low_vegetation", "buildings"):   "Low vegetation cleared and replaced by buildings (construction)",
    ("low_vegetation", "tree"):        "Low vegetation area matured into tree cover",
    ("low_vegetation", "water"):       "Vegetated area inundated or converted to water",
    ("low_vegetation", "playgrounds"): "Low vegetation converted to playground",
    ("water", "non_veg_ground"):       "Water body drained or dried into bare land",
    ("water", "buildings"):            "Water body filled and built upon",
    ("water", "tree"):                 "Former water area now covered by trees",
    ("water", "low_vegetation"):       "Water body receded, low vegetation established",
    ("water", "playgrounds"):          "Water body converted to sports facility",
    ("playgrounds", "non_veg_ground"): "Playground demolished, area reverted to bare ground",
    ("playgrounds", "buildings"):      "Playground replaced by buildings",
    ("playgrounds", "tree"):           "Playground area became tree cover",
    ("playgrounds", "low_vegetation"): "Playground area became low vegetation",
    ("playgrounds", "water"):          "Playground area became water body",
}

APPEARED_TEMPLATES = {
    "non_veg_ground":  "New bare or impervious surface area appeared",
    "buildings":       "New building structure appeared",
    "tree":            "New tree cover appeared",
    "low_vegetation":  "New low vegetation area appeared",
    "water":           "New water body appeared or expanded",
    "playgrounds":     "New playground or sports facility appeared",
}

DISAPPEARED_TEMPLATES = {
    "non_veg_ground":  "Bare ground area disappeared or was covered",
    "buildings":       "Building structure disappeared",
    "tree":            "Tree cover disappeared",
    "low_vegetation":  "Low vegetation area disappeared",
    "water":           "Water body disappeared or receded",
    "playgrounds":     "Playground disappeared",
}


def classify_instance_change(inst: dict, gt_label_t2: np.ndarray) -> dict:
    """
    Classify 1 T1 instance bằng cách lookup GT label T2 tại cùng tọa độ.
    Returns: dict với change_type, class_t2, change_label, description, confidence
    """
    mask_path = os.path.join(OUT_ROOT, inst["mask_file"])
    if not os.path.exists(mask_path):
        return None
    mask = np.load(mask_path).astype(bool)

    class_t1   = inst["class_name"]
    class_id_t1 = inst["class_id"]

    # Pixels ở T2 tại vùng mask T1
    pixels_t2 = gt_label_t2[mask]

    if len(pixels_t2) == 0:
        return None

    # Phân loại no-change pixels tại T2
    if NO_CHANGE_VAL != 255:
        semantic_t2 = pixels_t2[pixels_t2 != NO_CHANGE_VAL]
        nochange_ratio = float((pixels_t2 == NO_CHANGE_VAL).sum()) / len(pixels_t2)
    else:
        semantic_t2 = pixels_t2
        nochange_ratio = 0.0

    # ── Kiểm tra Disappeared: T2 mostly no-change ──────────────────────────
    if nochange_ratio >= DISAPPEARED_NOCHANGE_THRESH:
        return {
            "change_type":       "disappeared",
            "class_t1":          class_t1,
            "class_t2":          None,
            "change_label":      f"{class_t1} → [disappeared]",
            "t2_nochange_ratio": round(nochange_ratio, 3),
            "t2_class_confidence": 0.0,
            "description": DISAPPEARED_TEMPLATES.get(class_t1,
                                                      f"{class_t1} area disappeared"),
        }

    if len(semantic_t2) == 0:
        return None  # không đủ thông tin

    # ── Tìm dominant class T2 ─────────────────────────────────────────────
    values, counts = np.unique(semantic_t2, return_counts=True)
    dom_idx      = np.argmax(counts)
    class_id_t2  = int(values[dom_idx])
    confidence   = float(counts[dom_idx]) / len(semantic_t2)
    class_t2     = CLASS_MAP.get(class_id_t2, f"class_{class_id_t2}")

    if confidence < T2_CLASS_CONFIDENCE_MIN:
        # Vùng quá hỗn hợp — gán partial change
        change_type = "partial_change"
        change_label = f"{class_t1} → {class_t2} [partial, conf={confidence:.2f}]"
        description = f"[Partial] {class_t1} area partially changed toward {class_t2}"
    elif class_id_t1 == class_id_t2:
        change_type  = "no_change"
        change_label = f"{class_t1} → unchanged"
        description  = f"No land cover change: {class_t1} remains unchanged"
    else:
        change_type  = "semantic_change"
        change_label = f"{class_t1} → {class_t2}"
        key = (class_t1, class_t2)
        description = TRANSITION_TEMPLATES.get(
            key, f"{class_t1} changed to {class_t2}"
        )

    return {
        "change_type":         change_type,
        "class_t1":            class_t1,
        "class_t2":            class_t2,
        "class_id_t1":         class_id_t1,
        "class_id_t2":         class_id_t2,
        "change_label":        change_label,
        "t2_class_confidence": round(confidence, 3),
        "t2_nochange_ratio":   round(nochange_ratio, 3),
        "description":         description,
        "vlm_description":     None,   # Phase 3B sẽ fill nếu làm Tier 1
        "tier":                2,
    }


def main():
    # Load T1 instances
    with open(f"{OUT_ROOT}/annotations/instances_T1.json") as f:
        t1_data = json.load(f)

    stems = sorted(t1_data["instances"].keys())
    all_annotations = {}

    # Counters
    cnt = {"no_change": 0, "semantic_change": 0, "disappeared": 0,
           "partial_change": 0, "skip": 0}

    # Transition matrix: from-class → to-class → count
    transition_matrix = {}

    for stem in tqdm(stems, desc="Phase 2"):
        insts_T1 = t1_data["instances"][stem]
        if not insts_T1:
            continue

        # Load GT label T2
        lab2_path = os.path.join(DATA_ROOT, "label2", stem + ".png")
        if not os.path.exists(lab2_path):
            continue
        gt_label_t2 = np.array(Image.open(lab2_path))

        image_annotations = []
        for inst in insts_T1:
            result = classify_instance_change(inst, gt_label_t2)
            if result is None:
                cnt["skip"] += 1
                continue

            ann = {
                "instance_id": inst["instance_id"],
                "stem":        inst["stem"],
                "bbox":        inst["bbox"],
                "area_px":     inst["area_px"],
                **result
            }
            image_annotations.append(ann)
            cnt[result["change_type"]] = cnt.get(result["change_type"], 0) + 1

            # Update transition matrix
            c1 = result["class_t1"]
            c2 = result.get("class_t2") or "disappeared"
            transition_matrix.setdefault(c1, {})
            transition_matrix[c1][c2] = transition_matrix[c1].get(c2, 0) + 1

        # Summary per image
        n_changed = sum(1 for a in image_annotations if a["change_type"] == "semantic_change")
        n_unchanged = sum(1 for a in image_annotations if a["change_type"] == "no_change")
        n_disapp = sum(1 for a in image_annotations if a["change_type"] == "disappeared")
        n_partial = sum(1 for a in image_annotations if a["change_type"] == "partial_change")

        all_annotations[stem] = {
            "annotations": image_annotations,
            "summary": {
                "n_total":          len(image_annotations),
                "n_changed":        n_changed,
                "n_unchanged":      n_unchanged,
                "n_disappeared":    n_disapp,
                "n_partial":        n_partial,
                "change_labels":    list({a["change_label"] for a in image_annotations
                                          if a["change_type"] == "semantic_change"}),
            }
        }

    # ── Save ───────────────────────────────────────────────────────────────
    output = {
        "meta": {
            "n_images":          len(all_annotations),
            "split":             "official_SECOND_512x512",
            "n_pairs":           len(stems),
            "class_map":         CLASS_MAP,
            "change_type_counts": cnt,
            "transition_matrix": transition_matrix,
        },
        "images": all_annotations
    }

    with open(f"{OUT_ROOT}/annotations/change_annotations.json", "w") as f:
        json.dump(output, f, indent=2)

    # ── Print stats ─────────────────────────────────────────────────────────
    total = sum(cnt.values())
    print(f"\n✅ Phase 2 done. Total annotations: {total}")
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        pct = v / total * 100 if total else 0
        print(f"   {k:20s}: {v:6d}  ({pct:.1f}%)")

    print(f"\n   Transition matrix top-5:")
    flat = [(f"{c1}→{c2}", n)
            for c1, d in transition_matrix.items()
            for c2, n in d.items()]
    for pair, n in sorted(flat, key=lambda x: -x[1])[:5]:
        print(f"   {pair:40s}: {n}")

    # ── Validation ──────────────────────────────────────────────────────────
    assert cnt.get("semantic_change", 0) > 100, \
        "[FAIL] Quá ít semantic_change — kiểm tra GT label encoding và NO_CHANGE_VAL"
    assert len(transition_matrix) >= 3, \
        "[FAIL] Ít hơn 3 from-class trong transition matrix — kiểm tra CLASS_MAP"
    print("   ✅ Validation passed")


if __name__ == "__main__":
    main()
```

**Chạy:** `python phase2_classify_changes.py`

**Expected output:**
```
✅ Phase 2 done. Total annotations: ~18,000
   no_change       :  ~11,000  (61%)
   semantic_change :   ~5,500  (31%)
   disappeared     :     ~900   (5%)
   partial_change  :     ~500   (3%)

   Transition matrix top-5:
   non_veg_ground→buildings         : ~1,800
   low_vegetation→non_veg_ground    : ~1,200
   non_veg_ground→low_vegetation    : ~900
   low_vegetation→buildings         : ~800
   buildings→non_veg_ground         : ~600
```

---

## ════════════════════════════════════
## PHASE 3A — Template Description Generation
## ════════════════════════════════════

> Template descriptions đã được viết inline trong Phase 2 (field `description`).
> Phase 3A chỉ cần 1 validation pass + export clean caption file.

**File:** `phase3a_validate_descriptions.py`

```python
#!/usr/bin/env python3
"""
Phase 3A: Validate và export caption-only file cho dễ dùng.
Input:  change_annotations.json
Output: SECOND-OC/annotations/captions.jsonl  (1 line per annotation)
"""
import json

with open("SECOND-OC/annotations/change_annotations.json") as f:
    data = json.load(f)

missing_desc = 0
captions_out = []

for stem, img_data in data["images"].items():
    for ann in img_data["annotations"]:
        if not ann.get("description"):
            missing_desc += 1
            continue
        captions_out.append({
            "instance_id":  ann["instance_id"],
            "stem":         stem,
            "change_type":  ann["change_type"],
            "change_label": ann["change_label"],
            "description":  ann["description"],
            "tier":         ann.get("tier", 2),
        })

with open("SECOND-OC/annotations/captions.jsonl", "w") as f:
    for item in captions_out:
        f.write(json.dumps(item) + "\n")

print(f"✅ Phase 3A done.")
print(f"   Captions exported: {len(captions_out)}")
print(f"   Missing descriptions: {missing_desc}")
assert missing_desc < 50, f"[FAIL] {missing_desc} annotations không có description"
```

---

## ════════════════════════════════════
## PHASE 3B — Tier 1 Curation + VLM Descriptions (Optional)
## ════════════════════════════════════

### Bước 3B.1 — Chọn 300 pairs cho Tier 1 (stratified sampling)

**File:** `phase3b_curate_tier1.py`

```python
#!/usr/bin/env python3
"""
Phase 3B-1: Stratified sampling 300 pairs cho Tier 1.
Chiến lược: đảm bảo đủ đại diện tất cả change_type và top change_label.
Output: SECOND-OC/tier1/tier1_stems.json  (list of 300 stems)
"""
import json, random
from collections import defaultdict

TIER1_SIZE = 300
random.seed(42)

with open("SECOND-OC/annotations/change_annotations.json") as f:
    data = json.load(f)

# Nhóm stems theo change types nổi bật nhất trong ảnh đó
stems_by_type = defaultdict(list)
for stem, img_data in data["images"].items():
    anns = img_data["annotations"]
    # Ảnh có semantic_change → ưu tiên
    if any(a["change_type"] == "semantic_change" for a in anns):
        # Lấy change_label phổ biến nhất trong ảnh
        labels = [a["change_label"] for a in anns if a["change_type"] == "semantic_change"]
        top_label = max(set(labels), key=labels.count)
        stems_by_type[top_label].append(stem)
    elif any(a["change_type"] == "disappeared" for a in anns):
        stems_by_type["disappeared"].append(stem)
    else:
        stems_by_type["no_change_only"].append(stem)

# Stratified sampling
selected = []
types = [k for k in stems_by_type if k != "no_change_only"]
per_type = max(1, (TIER1_SIZE // 2) // len(types))  # ~50% từ semantic change stems

for t in types:
    pool = stems_by_type[t]
    n = min(per_type, len(pool))
    selected.extend(random.sample(pool, n))

# Fill còn lại từ no_change_only hoặc bất kỳ
remaining = TIER1_SIZE - len(selected)
extra_pool = [s for s in stems_by_type["no_change_only"] if s not in selected]
selected.extend(random.sample(extra_pool, min(remaining, len(extra_pool))))
selected = list(set(selected))[:TIER1_SIZE]

with open("SECOND-OC/tier1/tier1_stems.json", "w") as f:
    json.dump({"n": len(selected), "stems": sorted(selected)}, f, indent=2)

print(f"✅ Tier 1 curated: {len(selected)} stems")
print(f"   Change types represented: {len(types)}")
```

### Bước 3B.2 — VLM Description (GeoChat hoặc RemoteCLIP)

> **Chạy nếu và chỉ nếu GPU có sẵn và GeoChat/RemoteCLIP đã được install.**
> Nếu không có GPU → bỏ qua, Tier 1 vẫn có template descriptions từ Phase 3A.

**File:** `phase3b_vlm_descriptions.py`

```python
#!/usr/bin/env python3
"""
Phase 3B-2: Generate VLM descriptions cho Tier 1 changed objects.
Dùng GeoChat (ưu tiên) hoặc fallback sang template description nếu VLM fail.

Yêu cầu: pip install geochat  (hoặc theo README của GeoChat repo)
Output: SECOND-OC/tier1/vlm_descriptions.jsonl
"""
import json, os
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_ROOT = "SECOND/test"
OUT_ROOT  = "SECOND-OC"

# ── Load Tier 1 stems ────────────────────────────────────────────────────────
with open(f"{OUT_ROOT}/tier1/tier1_stems.json") as f:
    tier1_stems = set(json.load(f)["stems"])

# ── Load annotations ─────────────────────────────────────────────────────────
with open(f"{OUT_ROOT}/annotations/change_annotations.json") as f:
    all_data = json.load(f)

# ── Load VLM (GeoChat) ───────────────────────────────────────────────────────
# Agent: nếu import fail → comment toàn bộ khối VLM, dùng template_only=True
TEMPLATE_ONLY = False
try:
    # Thử load GeoChat nếu có
    # from geochat import GeoChat
    # vlm = GeoChat.from_pretrained("GeoChat-7B")
    raise ImportError("GeoChat not installed — using template fallback")
except ImportError:
    TEMPLATE_ONLY = True
    print("[WARNING] VLM không load được. Dùng template description cho Tier 1.")


def crop_bbox(img: Image.Image, bbox: list, pad: int = 20) -> Image.Image:
    """Crop ảnh theo bbox với padding."""
    w, h = img.size
    x1 = max(0, bbox[0] - pad)
    y1 = max(0, bbox[1] - pad)
    x2 = min(w, bbox[2] + pad)
    y2 = min(h, bbox[3] + pad)
    return img.crop((x1, y1, x2, y2))


def generate_vlm_description(crop_t1: Image.Image,
                              crop_t2: Image.Image,
                              change_label: str,
                              template: str) -> str:
    if TEMPLATE_ONLY:
        return template   # fallback

    prompt = (
        f"These are satellite image crops of the same location at two times.\n"
        f"Known land cover change: {change_label}\n"
        f"Describe what visually changed between T1 and T2 in 1-2 concise sentences, "
        f"focusing on observable differences in color, texture, or structure."
    )
    try:
        # response = vlm.generate(prompt, images=[crop_t1, crop_t2], max_new_tokens=80)
        # return response.strip()
        return template   # placeholder — replace với actual VLM call
    except Exception as e:
        print(f"  [VLM ERROR] {e} — using template")
        return template


# ── Main loop ─────────────────────────────────────────────────────────────────
results = []
hallucination_checks = []   # Untuk validasi: VLM class vs GT class

for stem in tqdm(sorted(tier1_stems), desc="Phase 3B VLM"):
    if stem not in all_data["images"]:
        continue

    img_t1 = Image.open(os.path.join(DATA_ROOT, "im1", stem + ".png"))
    img_t2 = Image.open(os.path.join(DATA_ROOT, "im2", stem + ".png"))

    for ann in all_data["images"][stem]["annotations"]:
        if ann["change_type"] not in ("semantic_change", "disappeared"):
            continue   # hanya Tier 1 changed objects yang dapat VLM desc

        crop_t1 = crop_bbox(img_t1, ann["bbox"])
        crop_t2 = crop_bbox(img_t2, ann["bbox"])

        vlm_desc = generate_vlm_description(
            crop_t1, crop_t2,
            ann["change_label"],
            ann["description"]   # template làm fallback
        )

        results.append({
            "instance_id":    ann["instance_id"],
            "stem":           stem,
            "change_label":   ann["change_label"],
            "template_desc":  ann["description"],
            "vlm_description": vlm_desc,
            "vlm_source":    "template_fallback" if TEMPLATE_ONLY else "geochat",
        })

# ── Save ─────────────────────────────────────────────────────────────────────
with open(f"{OUT_ROOT}/tier1/vlm_descriptions.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"✅ Phase 3B done. VLM descriptions: {len(results)}")
print(f"   Source: {'template_fallback' if TEMPLATE_ONLY else 'geochat'}")
```

---

## ════════════════════════════════════
## PHASE 4 — Evaluation Script + Packaging
## ════════════════════════════════════

### Bước 4.1 — Object-level Evaluation Metrics

**File:** `SECOND-OC/eval/object_eval.py`

```python
#!/usr/bin/env python3
"""
SECOND-OC Evaluation Script.
Metrics:
  - Binary-Object-F1     : detect đúng object thay đổi (không cần đúng class)
  - Semantic-Object-F1   : detect đúng object + đúng change_label
  - Change-IoU           : overlap vùng thay đổi ở cấp object

Usage:
    python object_eval.py \
        --gt  SECOND-OC/annotations/change_annotations.json \
        --pred predictions.json \
        --iou-threshold 0.5

predictions.json format:
{
  "stem1": [
    {"mask": "path/to/mask.npy",
     "change_label": "low_vegetation → buildings"},
    ...
  ],
  ...
}
"""
import argparse, json
import numpy as np
from pathlib import Path


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return float(inter) / float(union + 1e-8)


def evaluate(gt_path: str, pred_path: str,
             iou_threshold: float = 0.5) -> dict:

    with open(gt_path) as f:
        gt_data = json.load(f)
    with open(pred_path) as f:
        pred_data = json.load(f)

    # ── Counters ──────────────────────────────────────────────────────────
    binary_tp = binary_fp = binary_fn = 0
    sem_tp = sem_fp = sem_fn = 0
    iou_sum = iou_count = 0

    for stem, img_gt in gt_data["images"].items():
        gt_changed = [
            a for a in img_gt["annotations"]
            if a["change_type"] in ("semantic_change", "disappeared")
        ]
        preds = pred_data.get(stem, [])

        # Load GT masks
        gt_masks = []
        for ann in gt_changed:
            mask_path = Path("SECOND-OC") / ann["instance_id"].replace("_", "/", 2)
            # Fallback: cố tìm mask theo mask_file nếu có
            mask_file = ann.get("mask_file", "")
            full_path = Path("SECOND-OC") / mask_file if mask_file else None
            if full_path and full_path.exists():
                m = np.load(full_path).astype(bool)
            else:
                continue
            gt_masks.append({"mask": m, "change_label": ann["change_label"]})

        # Load Pred masks
        pred_masks = []
        for p in preds:
            if not Path(p["mask"]).exists():
                continue
            m = np.load(p["mask"]).astype(bool)
            pred_masks.append({"mask": m, "change_label": p.get("change_label", "")})

        # ── Matching: greedy, highest IoU first ───────────────────────────
        matched_gt   = set()
        matched_pred = set()

        for pi, pred in enumerate(pred_masks):
            best_iou = 0.0
            best_gi  = -1
            for gi, gt in enumerate(gt_masks):
                if gi in matched_gt:
                    continue
                iou = compute_mask_iou(pred["mask"], gt["mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_gi  = gi

            if best_iou >= iou_threshold and best_gi >= 0:
                matched_gt.add(best_gi)
                matched_pred.add(pi)

                # Binary TP
                binary_tp += 1
                iou_sum   += best_iou
                iou_count += 1

                # Semantic TP: change_label phải khớp
                gt_label   = gt_masks[best_gi]["change_label"].strip().lower()
                pred_label = pred_masks[pi]["change_label"].strip().lower()
                if gt_label == pred_label:
                    sem_tp += 1
                else:
                    sem_fp += 1   # matched spatially but wrong class

            else:
                binary_fp += 1
                sem_fp    += 1

        # FN: GT changed objects không được predict
        binary_fn += len(gt_masks) - len(matched_gt)
        sem_fn    += len(gt_masks) - sum(
            1 for gi, gt in enumerate(gt_masks) if gi in matched_gt and
            gt["change_label"].lower() ==
            pred_masks[list(matched_pred)[list(matched_gt).index(gi)]]["change_label"].lower()
            if gi in matched_gt
        )

    def f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        return 2 * p * r / (p + r + 1e-8), p, r

    bin_f1, bin_p, bin_r = f1(binary_tp, binary_fp, binary_fn)
    sem_f1, sem_p, sem_r = f1(sem_tp, sem_fp, sem_fn)
    mean_iou = iou_sum / iou_count if iou_count > 0 else 0.0

    results = {
        "Binary-Object-F1":    round(bin_f1, 4),
        "Binary-Precision":    round(bin_p, 4),
        "Binary-Recall":       round(bin_r, 4),
        "Semantic-Object-F1":  round(sem_f1, 4),
        "Semantic-Precision":  round(sem_p, 4),
        "Semantic-Recall":     round(sem_r, 4),
        "Change-IoU":          round(mean_iou, 4),
        "iou_threshold":       iou_threshold,
        "binary_TP":           binary_tp,
        "binary_FP":           binary_fp,
        "binary_FN":           binary_fn,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",            required=True)
    parser.add_argument("--pred",          required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    results = evaluate(args.gt, args.pred, args.iou_threshold)
    print(json.dumps(results, indent=2))
```

### Bước 4.2 — Đóng gói final JSON

**File:** `phase4_package.py`

```python
#!/usr/bin/env python3
"""
Phase 4: Tạo file benchmark cuối cùng SECOND-OC/annotations/benchmark.json
Merge instances + change_annotations + tier1 flags vào 1 file duy nhất.
"""
import json, os

OUT_ROOT = "SECOND-OC"

with open(f"{OUT_ROOT}/annotations/instances_T1.json") as f:
    inst_T1 = json.load(f)["instances"]
with open(f"{OUT_ROOT}/annotations/instances_T2.json") as f:
    inst_T2 = json.load(f)["instances"]
with open(f"{OUT_ROOT}/annotations/change_annotations.json") as f:
    change_data = json.load(f)

# Load Tier 1
tier1_stems = set()
tier1_path = f"{OUT_ROOT}/tier1/tier1_stems.json"
if os.path.exists(tier1_path):
    with open(tier1_path) as f:
        tier1_stems = set(json.load(f)["stems"])

# Load VLM descriptions nếu có
vlm_desc_map = {}
vlm_path = f"{OUT_ROOT}/tier1/vlm_descriptions.jsonl"
if os.path.exists(vlm_path):
    with open(vlm_path) as f:
        for line in f:
            item = json.loads(line)
            vlm_desc_map[item["instance_id"]] = item.get("vlm_description", "")

# ── Build benchmark entries ──────────────────────────────────────────────────
benchmark = {"meta": change_data["meta"], "entries": []}

for stem, img_data in change_data["images"].items():
    entry = {
        "stem":     stem,
        "image_t1": f"im1/{stem}.png",
        "image_t2": f"im2/{stem}.png",
        "tier":     1 if stem in tier1_stems else 2,
        "objects_t1": inst_T1.get(stem, []),
        "objects_t2": inst_T2.get(stem, []),
        "change_annotations": [],
        "summary": img_data["summary"],
    }

    for ann in img_data["annotations"]:
        iid = ann["instance_id"]
        ann_out = dict(ann)
        if iid in vlm_desc_map:
            ann_out["vlm_description"] = vlm_desc_map[iid]
        entry["change_annotations"].append(ann_out)

    benchmark["entries"].append(entry)

with open(f"{OUT_ROOT}/annotations/benchmark.json", "w") as f:
    json.dump(benchmark, f, indent=2)

n = len(benchmark["entries"])
n_changed = sum(
    e["summary"]["n_changed"] for e in benchmark["entries"]
)
print(f"✅ Phase 4 done.")
print(f"   benchmark.json: {n} images, {n_changed} changed object pairs")
print(f"   Tier 1: {sum(1 for e in benchmark['entries'] if e['tier']==1)} images")
print(f"   Output: {OUT_ROOT}/annotations/benchmark.json")
```

---

## ════════════════════════════════════
## RUN ORDER & CHECKLIST
## ════════════════════════════════════

```bash
# 1. Setup
pip install numpy pillow tqdm scipy

# 2. Phase 0 — WAJIB chạy trước
python phase0_verify.py
# → Đọc output, xác nhận format SAM2 tokens trước khi tiếp tục

# 3. Phase 1 — Instance extraction
python phase1_extract_instances.py
python phase1_visualize.py          # visual check 5 mẫu

# 4. Phase 2 — Change classification
python phase2_classify_changes.py
# → Check transition matrix: có >= 5 from-class, >= 15 unique pairs

# 5. Phase 3A — Template descriptions (already embedded in Phase 2)
python phase3a_validate_descriptions.py

# 6. Phase 3B — VLM Tier 1 (Optional, cần GPU + GeoChat)
python phase3b_curate_tier1.py
python phase3b_vlm_descriptions.py  # bỏ qua nếu không có GPU

# 7. Phase 4 — Package
python phase4_package.py

# 8. Test eval script
python SECOND-OC/eval/object_eval.py \
    --gt SECOND-OC/annotations/change_annotations.json \
    --pred dummy_predictions.json \
    --iou-threshold 0.5
```

## Validation checkpoints (agent tự kiểm tra sau mỗi phase)

| Phase | File output | Assert |
|---|---|---|
| 0 | `phase0_report.json` | `n_test_pairs == 1694` |
| 1 | `instances_T1.json` | `total_T1 > 5000` |
| 1 | `masks/T1/*.npy` | file count ≈ total_T1 |
| 2 | `change_annotations.json` | `semantic_change > 100`, `len(transition_matrix) >= 3` |
| 3A | `captions.jsonl` | `missing_descriptions < 50` |
| 3B | `tier1_stems.json` | `n == 300` |
| 4 | `benchmark.json` | `n_images == 1694` |

## Khi nào phải dừng và báo cáo

Agent dừng và paste output lên chat nếu:
- Phase 0: SAM2 format không khớp 5 nhánh đã viết
- Phase 1: `total_T1 < 5000` (SAM2 loader trả về quá ít mask)
- Phase 1: `skipped > 30%` số ảnh (không load được label hoặc token)
- Phase 2: `semantic_change == 0` (encoding GT label sai)
- Phase 2: `transition_matrix` chỉ có 1-2 from-class (CLASS_MAP sai)

Mọi trường hợp khác: agent tự xử lý theo nhánh decision đã ghi trong code.