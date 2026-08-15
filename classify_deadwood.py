# classify_deadwood.py
"""
Bài toán đơn giản: deadwood vs other (binary classification)
So sánh: shape-only features vs shape+spectral features

Không cần MoE/Transformer phức tạp — dùng Random Forest đơn giản
để validate finding CORE: spectral có giúp classification không.
"""
import os, glob, json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import rasterio

IMG_DIR  = "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
GT_DIR   = "DeadTrees/masks_gt"
SAM2_DIR = "DeadTrees/sam2_masks"
OUT_DIR  = "DeadTrees/results"
os.makedirs(OUT_DIR, exist_ok=True)


def extract_shape_features(mask):
    """Shape-only: area, perimeter, compactness, bbox ratio."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros(4)
    area = mask.sum()
    bbox_h = ys.max() - ys.min() + 1
    bbox_w = xs.max() - xs.min() + 1
    bbox_ratio = bbox_h / max(bbox_w, 1)

    # Perimeter approx qua boundary
    boundary = mask ^ ndimage.binary_erosion(mask)
    perimeter = boundary.sum()
    compactness = (4 * np.pi * area) / max(perimeter**2, 1)

    return np.array([np.log(area + 1), bbox_ratio, compactness, perimeter])


def extract_spectral_features(image, mask):
    """Spectral: mean/std RGB."""
    pixels = image[mask.astype(bool)].astype(np.float32)
    if len(pixels) == 0:
        return np.zeros(6)
    return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])


# Map stems to absolute image paths dynamically
img_files = glob.glob(f"{IMG_DIR}/**/*.tif", recursive=True)
stem_to_img_path = {Path(f).stem: f for f in img_files}

# --- Build dataset ---
X_shape, X_spectral, y = [], [], []

gt_files = glob.glob(f"{GT_DIR}/*_deadwood.png")
for gt_path in tqdm(gt_files, desc="Building classification dataset"):
    stem = Path(gt_path).stem.replace("_deadwood", "")

    img_path = stem_to_img_path.get(stem)
    if img_path is None:
        continue

    sam2_path = f"{SAM2_DIR}/{stem}.npz"
    if not os.path.exists(sam2_path):
        continue

    with rasterio.open(img_path) as src:
        image = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
        if image.dtype != np.uint8:
            image = ((image - image.min()) /
                    (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

    gt_mask    = np.array(Image.open(gt_path)) > 127
    sam2_masks = np.load(sam2_path)["masks"]

    for sam_mask in sam2_masks:
        sam_mask = sam_mask.astype(bool)
        if sam_mask.sum() < 50:  # Skip very small regions
            continue

        # Label: overlap > 50% với GT deadwood → positive
        overlap = (sam_mask & gt_mask).sum()
        label = 1 if overlap / sam_mask.sum() > 0.5 else 0

        X_shape.append(extract_shape_features(sam_mask))
        X_spectral.append(extract_spectral_features(image, sam_mask))
        y.append(label)

X_shape    = np.array(X_shape)
X_spectral = np.array(X_spectral)
y          = np.array(y)

print(f"\nDataset: {len(y)} samples, {y.sum()} positive (deadwood), "
      f"{len(y)-y.sum()} negative")

# --- Train/test split ---
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)

def evaluate_features(X, name):
    clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                 class_weight="balanced")
    clf.fit(X[idx_train], y[idx_train])
    pred = clf.predict(X[idx_test])

    acc = accuracy_score(y[idx_test], pred)
    f1  = f1_score(y[idx_test], pred)

    print(f"\n{'='*50}")
    print(f"Features: {name}")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print(classification_report(y[idx_test], pred,
                                target_names=["other", "deadwood"]))
    return {"accuracy": acc, "f1": f1}


# --- Ablation: shape-only vs shape+spectral ---
result_shape = evaluate_features(X_shape, "Shape-only (baseline)")

X_combined = np.concatenate([X_shape, X_spectral], axis=1)
result_combined = evaluate_features(X_combined, "Shape + Spectral")

# --- Summary ---
print(f"\n{'='*50}")
print("ABLATION SUMMARY")
print(f"{'='*50}")
print(f"Shape-only:       Acc={result_shape['accuracy']:.4f}, "
      f"F1={result_shape['f1']:.4f}")
print(f"Shape + Spectral: Acc={result_combined['accuracy']:.4f}, "
      f"F1={result_combined['f1']:.4f}")

delta_acc = result_combined['accuracy'] - result_shape['accuracy']
delta_f1  = result_combined['f1'] - result_shape['f1']
print(f"\nImprovement: +{delta_acc:.4f} accuracy, +{delta_f1:.4f} F1")

with open(f"{OUT_DIR}/classification_ablation.json", "w") as f:
    json.dump({
        "shape_only": result_shape,
        "shape_spectral": result_combined,
        "delta_accuracy": delta_acc,
        "delta_f1": delta_f1,
    }, f, indent=2)

# Validation
assert delta_f1 >= 0, \
    "[UNEXPECTED] Spectral features không cải thiện — cần điều tra thêm"
print(f"\n✅ Validation: spectral features improve classification "
      f"(+{delta_f1:.1%} F1)")
