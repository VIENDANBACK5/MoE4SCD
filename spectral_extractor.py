# spectral_extractor.py
"""
Extract 24 spectral features từ ảnh RGB gốc cho mỗi SAM2 mask.
Dựa trên Dao et al. 2021 (ISPRS): objects carry richer spectral
and spatial information than pixels.

24 features per token:
  mean RGB T1:      3 dims  (spectral identity T1)
  std RGB T1:       3 dims  (texture/roughness T1)
  mean RGB T2:      3 dims  (spectral identity T2)
  std RGB T2:       3 dims  (texture/roughness T2)
  delta mean:       3 dims  (spectral change magnitude) ← QUAN TRỌNG NHẤT
  delta std:        3 dims  (texture change)
  CV T1:            3 dims  (relative variability T1)
  CV T2:            3 dims  (relative variability T2)
  ─────────────────────────
  Tổng:            24 dims
"""
import os
import numpy as np
from PIL import Image


def extract_spectral_features(
    image_t1: np.ndarray,   # (H, W, 3) float32, range [0, 1]
    image_t2: np.ndarray,   # (H, W, 3) float32, range [0, 1]
    mask: np.ndarray,        # (H, W) bool
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Extract 24 spectral features cho 1 SAM2 mask.
    Returns: np.ndarray shape (24,) float32
    """
    # Lấy pixel values trong vùng mask
    region_t1 = image_t1[mask]  # (N_pixels, 3)
    region_t2 = image_t2[mask]  # (N_pixels, 3)

    if len(region_t1) == 0:
        return np.zeros(24, dtype=np.float32)

    # Mean RGB (spectral identity)
    mean_t1 = region_t1.mean(axis=0)   # (3,)
    mean_t2 = region_t2.mean(axis=0)

    # Std RGB (texture proxy)
    std_t1 = region_t1.std(axis=0)     # (3,)
    std_t2 = region_t2.std(axis=0)

    # Temporal delta (change magnitude) — quan trọng nhất
    delta_mean = mean_t2 - mean_t1     # (3,)
    delta_std  = std_t2 - std_t1       # (3,)

    # Coefficient of Variation (Dao et al. 2021)
    # CV = std / mean — đo độ biến thiên tương đối
    cv_t1 = std_t1 / (mean_t1 + eps)   # (3,)
    cv_t2 = std_t2 / (mean_t2 + eps)   # (3,)

    features = np.concatenate([
        mean_t1, std_t1,       # 6 dims: spectral identity + texture T1
        mean_t2, std_t2,       # 6 dims: spectral identity + texture T2
        delta_mean, delta_std, # 6 dims: temporal change
        cv_t1, cv_t2,          # 6 dims: relative variability
    ]).astype(np.float32)

    assert features.shape == (24,), f"Expected (24,), got {features.shape}"
    return features


def extract_spectral_for_stem(
    stem: str,
    im1_dir: str,
    im2_dir: str,
    masks_T1_dir: str,
    masks_T2_dir: str,
) -> dict:
    """
    Extract spectral features cho toàn bộ masks của 1 stem.
    Returns dict: {
        'spectral_T1': np.ndarray (N_masks_T1, 24),
        'spectral_T2': np.ndarray (N_masks_T2, 24),
    }
    """
    # Load ảnh
    img_t1 = np.array(Image.open(
        os.path.join(im1_dir, stem + ".png")
    ).convert("RGB")).astype(np.float32) / 255.0

    img_t2 = np.array(Image.open(
        os.path.join(im2_dir, stem + ".png")
    ).convert("RGB")).astype(np.float32) / 255.0

    # Load SAM2 masks
    data_t1 = np.load(os.path.join(masks_T1_dir, stem + ".npz"))
    data_t2 = np.load(os.path.join(masks_T2_dir, stem + ".npz"))
    masks_t1 = data_t1["masks"]  # (N1, H, W) bool
    masks_t2 = data_t2["masks"]  # (N2, H, W) bool

    # Extract features cho từng mask
    spectral_T1 = np.stack([
        extract_spectral_features(img_t1, img_t2, masks_t1[i].astype(bool))
        for i in range(len(masks_t1))
    ])  # (N1, 24)

    spectral_T2 = np.stack([
        extract_spectral_features(img_t2, img_t1, masks_t2[i].astype(bool))
        for i in range(len(masks_t2))
    ])  # (N2, 24)

    return {"spectral_T1": spectral_T1, "spectral_T2": spectral_T2}


# ── Test nhanh với 1 stem ────────────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_spectral_for_stem(
        stem="00003",
        im1_dir="SECOND/im1",
        im2_dir="SECOND/im2",
        masks_T1_dir="SECOND/sam2_masks_T1",
        masks_T2_dir="SECOND/sam2_masks_T2",
    )
    print(f"T1 spectral features shape: {result['spectral_T1'].shape}")
    print(f"T2 spectral features shape: {result['spectral_T2'].shape}")
    print(f"Sample T1 features (first mask): {result['spectral_T1'][0]}")

    # Validation
    assert result['spectral_T1'].shape[1] == 24, "Phải có 24 features"
    assert not np.any(np.isnan(result['spectral_T1'])), "Không được có NaN"
    print("✅ spectral_extractor.py validation passed")
