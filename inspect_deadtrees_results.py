# inspect_deadtrees_results.py
import os, glob
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import rasterio

IMG_DIR  = "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
GT_DIR   = "DeadTrees/masks_gt"
SAM2_DIR = "DeadTrees/sam2_masks"

def extract_shape_features(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros(4)
    area = mask.sum()
    bbox_h = ys.max() - ys.min() + 1
    bbox_w = xs.max() - xs.min() + 1
    bbox_ratio = bbox_h / max(bbox_w, 1)
    boundary = mask ^ ndimage.binary_erosion(mask)
    perimeter = boundary.sum()
    compactness = (4 * np.pi * area) / max(perimeter**2, 1)
    return np.array([np.log(area + 1), bbox_ratio, compactness, perimeter])

def extract_spectral_features(image, mask):
    pixels = image[mask.astype(bool)].astype(np.float32)
    if len(pixels) == 0:
        return np.zeros(6)
    return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])

# Map stems to absolute image paths dynamically
img_files = glob.glob(f"{IMG_DIR}/**/*.tif", recursive=True)
stem_to_img_path = {Path(f).stem: f for f in img_files}

# --- 1. Distinct images containing GT objects ---
gt_files = glob.glob(f"{GT_DIR}/*_deadwood.png")
images_with_gt = []
total_gt_objects = 0
for gt_path in gt_files:
    gt_mask = np.array(Image.open(gt_path)) > 127
    if gt_mask.sum() == 0:
        continue
    # Count components
    labeled_gt, n_components = ndimage.label(gt_mask)
    n_valid = 0
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled_gt == comp_id
        if comp_mask.sum() >= 20:
            n_valid += 1
    if n_valid > 0:
        images_with_gt.append(Path(gt_path).stem.replace("_deadwood", ""))
        total_gt_objects += n_valid

print("=== SEGMENTATION DIAGNOSTICS ===")
print(f"Total GT objects (size >= 20px): {total_gt_objects}")
print(f"Number of distinct images containing these GT objects: {len(images_with_gt)}")
print(f"Images with GT: {images_with_gt}")
print()

# --- 2. Build dataset for classification ---
X_shape, X_spectral, y = [], [], []

for gt_path in gt_files:
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
        if sam_mask.sum() < 50:
            continue

        overlap = (sam_mask & gt_mask).sum()
        label = 1 if overlap / sam_mask.sum() > 0.5 else 0

        X_shape.append(extract_shape_features(sam_mask))
        X_spectral.append(extract_spectral_features(image, sam_mask))
        y.append(label)

X_shape    = np.array(X_shape)
X_spectral = np.array(X_spectral)
y          = np.array(y)

idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)

print("=== CLASSIFICATION DIAGNOSTICS ===")
print(f"Total samples (SAM2 masks >= 50px): {len(y)}")
print(f"Total deadwood positive samples in dataset: {y.sum()} ({y.sum()/len(y):.2%})")
print(f"Total negative samples: {len(y) - y.sum()}")
print(f"Train positive samples: {y[idx_train].sum()} / {len(idx_train)}")
print(f"Test positive samples:  {y[idx_test].sum()} / {len(idx_test)}")
print()

# Train & evaluate shape-only
clf_shape = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf_shape.fit(X_shape[idx_train], y[idx_train])
pred_shape = clf_shape.predict(X_shape[idx_test])
cm_shape = confusion_matrix(y[idx_test], pred_shape)

print("--- CONFUSION MATRIX: SHAPE-ONLY ---")
tn, fp, fn, tp = cm_shape.ravel()
print(f"TN: {tn}, FP: {fp}")
print(f"FN: {fn}, TP: {tp}")
print(classification_report(y[idx_test], pred_shape, target_names=["other", "deadwood"]))

# Train & evaluate shape + spectral
X_combined = np.concatenate([X_shape, X_spectral], axis=1)
clf_combined = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf_combined.fit(X_combined[idx_train], y[idx_train])
pred_combined = clf_combined.predict(X_combined[idx_test])
cm_combined = confusion_matrix(y[idx_test], pred_combined)

print("--- CONFUSION MATRIX: SHAPE + SPECTRAL ---")
tn, fp, fn, tp = cm_combined.ravel()
print(f"TN: {tn}, FP: {fp}")
print(f"FN: {fn}, TP: {tp}")
print(classification_report(y[idx_test], pred_combined, target_names=["other", "deadwood"]))

# Check details of positive predictions in combined
print("Combined model prediction breakdown:")
print("Predicted positive:", (pred_combined == 1).sum())
print("True positive (TP):", tp)
print("False positive (FP):", fp)
print("False negative (FN):", fn)
