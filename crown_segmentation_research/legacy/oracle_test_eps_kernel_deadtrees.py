"""Oracle Test: Anisotropic eps-Kernel vs Standard StarDist Encoding on DeadTrees GT.

This is a representation-capacity test. It evaluates how faithfully real DeadTrees
instances (fallen logs, branching snags, irregular deadwood) can be reconstructed
from:
  1. Standard StarDist: 16 equiangular radial rays cast from instance centroid.
  2. Anisotropic eps-Kernel: Directional extent support function along principal
     axes (u_parallel, u_perp) + canonical directional basis in lifting space.

Runs on all held-out DeadTrees val npz files (site 5737, 20 images).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

VAL_DIR = Path("DeadTrees/star_convex_targets_v1/val")
MIN_GT_AREA = 20


def mask_to_polygon(mask: np.ndarray) -> Polygon | MultiPolygon | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    polys = []
    for c in contours:
        if len(c) >= 3:
            pts = c.squeeze()
            if pts.ndim == 2 and len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area >= 5.0:
                    polys.append(poly)
    if not polys:
        return None
    return unary_union(polys) if len(polys) > 1 else polys[0]


def encode_decode_stardist(mask: np.ndarray, n_rays: int = 16) -> Polygon | None:
    """Standard StarDist: equi-angular rays from centroid."""
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return None
    cy = float(np.mean(y_indices))
    cx = float(np.mean(x_indices))

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    poly = mask_to_polygon(mask)
    if poly is None or poly.is_empty:
        return None

    # Cast rays
    h, w = mask.shape
    max_diag = np.sqrt(h**2 + w**2)
    radial_pts = []
    
    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        # Sample along ray to find boundary
        ts = np.linspace(0, max_diag, int(max_diag * 2))
        xs = np.clip(np.round(cx + ts * dx), 0, w - 1).astype(int)
        ys = np.clip(np.round(cy + ts * dy), 0, h - 1).astype(int)
        
        in_mask = mask[ys, xs]
        if not in_mask[0]:
            # Centroid outside mask (non-convex shape failure)
            radial_pts.append((cx, cy))
            continue
            
        # Find last contiguous pixel inside mask
        diff = np.diff(in_mask.astype(int))
        exit_indices = np.where(diff == -1)[0]
        if len(exit_indices) > 0:
            last_t = ts[exit_indices[0]]
        else:
            last_t = 0.0
        radial_pts.append((cx + last_t * dx, cy + last_t * dy))

    if len(radial_pts) < 3:
        return None
    recon = Polygon(radial_pts)
    return recon if recon.is_valid and recon.area > 0 else recon.buffer(0)


def encode_decode_eps_kernel(mask: np.ndarray, n_directions: int = 16) -> Polygon | None:
    """Anisotropic eps-Kernel: Principal orientation + Directional Support Extents.
    
    Computes directional support function h_P(u) = max_{p in P} <p, u> along an
    anisotropic basis aligned with the principal inertia axes of the deadwood.
    """
    y_indices, x_indices = np.where(mask)
    if len(y_indices) < 3:
        return None
    
    coords = np.stack([x_indices, y_indices], axis=1).astype(np.float64) # (N, 2)
    center = np.mean(coords, axis=0, keepdims=True)
    centered = coords - center
    
    # 1. Principal Inertia Axes (PCA / Second Moments)
    cov = (centered.T @ centered) / len(coords)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Principal longitudinal axis
    u_major = eigvecs[:, 1]
    phi = np.arctan2(u_major[1], u_major[0])
    
    # 2. Adaptive Anisotropic Directional Basis
    # Dense sampling along longitudinal ends + transverse width
    canonical_angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    # Rotate basis by principal angle phi
    rotated_angles = canonical_angles + phi
    directions = np.stack([np.cos(rotated_angles), np.sin(rotated_angles)], axis=1) # (K, 2)
    
    # 3. Directional Support Extent & Extreme Coreset Points
    # For each direction u_k, find extreme support value and point
    projections = coords @ directions.T # (N, K)
    extreme_indices = np.argmax(projections, axis=0) # (K,)
    coreset_points = coords[extreme_indices] # (K, 2)
    
    # 4. Reconstruct Polytope Envelope via Convex Hull of Coreset Points
    # (or ordered boundary in lifting space)
    unique_pts = np.unique(coreset_points, axis=0)
    if len(unique_pts) < 3:
        return None
        
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(unique_pts)
        hull_pts = unique_pts[hull.vertices]
        recon = Polygon(hull_pts)
        return recon if recon.is_valid and recon.area > 0 else recon.buffer(0)
    except Exception:
        # Fallback to ordered points around center
        angles_pts = np.arctan2(unique_pts[:, 1] - center[0, 1], unique_pts[:, 0] - center[0, 0])
        sorted_idx = np.argsort(angles_pts)
        recon = Polygon(unique_pts[sorted_idx])
        return recon if recon.is_valid and recon.area > 0 else recon.buffer(0)


def compute_iou(poly1: Polygon | MultiPolygon | None, poly2: Polygon | MultiPolygon | None) -> float:
    if poly1 is None or poly2 is None or poly1.is_empty or poly2.is_empty:
        return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def main():
    print("=" * 80)
    print("ORACLE TEST: ANISOTROPIC eps-KERNEL VS STARDIST ON DEADTREES GT")
    print("=" * 80)

    npz_files = sorted(VAL_DIR.glob("*.npz"))
    print(f"Loaded {len(npz_files)} validation tiles from {VAL_DIR}\n")

    results = []
    
    for npz_path in npz_files:
        data = np.load(npz_path)
        label_map = data["instance_label"]
        labels = np.unique(label_map)
        labels = labels[labels != 0]
        
        for lbl in labels:
            mask = label_map == lbl
            if mask.sum() < MIN_GT_AREA:
                continue
                
            gt_poly = mask_to_polygon(mask)
            if gt_poly is None or gt_poly.is_empty:
                continue

            # Compute geometric properties
            y_idx, x_idx = np.where(mask)
            coords = np.stack([x_idx, y_idx], axis=1).astype(np.float64)
            centered = coords - np.mean(coords, axis=0, keepdims=True)
            cov = (centered.T @ centered) / len(coords)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            aspect_ratio = np.sqrt(eigvals[1] / eigvals[0])
            
            # 1. StarDist reconstruction
            stardist_poly = encode_decode_stardist(mask, n_rays=16)
            stardist_iou = compute_iou(gt_poly, stardist_poly)
            
            # 2. Anisotropic eps-Kernel reconstruction
            epskernel_poly = encode_decode_eps_kernel(mask, n_directions=16)
            epskernel_iou = compute_iou(gt_poly, epskernel_poly)
            
            results.append({
                "tile": npz_path.stem,
                "label": int(lbl),
                "area": int(mask.sum()),
                "aspect_ratio": float(aspect_ratio),
                "stardist_iou": float(stardist_iou),
                "epskernel_iou": float(epskernel_iou),
            })

    print(f"Total evaluated DeadTrees instances: {len(results)}\n")

    # Aggregate Analysis
    aspect_bins = [
        ("Compact snags (AR <= 2.0)", lambda r: r["aspect_ratio"] <= 2.0),
        ("Moderate elongated (2.0 < AR <= 5.0)", lambda r: 2.0 < r["aspect_ratio"] <= 5.0),
        ("Fallen logs / High AR (AR > 5.0)", lambda r: r["aspect_ratio"] > 5.0),
        ("All DeadTrees instances", lambda r: True),
    ]

    print(f"{'Instance Category':<35} | {'Count':<6} | {'StarDist IoU':<14} | {'eps-Kernel IoU':<14} | {'Delta IoU'}")
    print("-" * 88)

    summary = {}
    for name, predicate in aspect_bins:
        sub = [r for r in results if predicate(r)]
        if not sub:
            continue
        sd_mean = np.mean([r["stardist_iou"] for r in sub])
        ek_mean = np.mean([r["epskernel_iou"] for r in sub])
        delta = ek_mean - sd_mean
        print(f"{name:<35} | {len(sub):<6} | {sd_mean:<14.4f} | {ek_mean:<14.4f} | {delta:+.4f} ({delta*100:+.1f}%)")
        summary[name] = {
            "count": len(sub),
            "stardist_mean_iou": float(sd_mean),
            "epskernel_mean_iou": float(ek_mean),
            "delta_iou": float(delta),
        }

    # Severe failure rate (IoU < 0.50)
    sd_failures = np.mean([r["stardist_iou"] < 0.50 for r in results]) * 100
    ek_failures = np.mean([r["epskernel_iou"] < 0.50 for r in results]) * 100
    print("-" * 88)
    print(f"Severe Failure Rate (IoU < 0.50): StarDist = {sd_failures:.1f}% vs eps-Kernel = {ek_failures:.1f}% (-{sd_failures - ek_failures:.1f}% reduction)")

    out_file = Path("DeadTrees/experiments/oracle_eps_kernel_deadtrees_result.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"summary": summary, "per_instance": results}, indent=2))
    print(f"\nSaved full results to {out_file}")


if __name__ == "__main__":
    main()
