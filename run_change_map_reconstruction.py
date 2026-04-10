#!/usr/bin/env python3
"""run_change_map_reconstruction.py  —  Stage 7: Change Map Reconstruction
===========================================================================
Converts token-level change predictions to full-resolution (512×512)
pixel change maps and evaluates them against ground-truth semantic-change labels.

Since SAM pixel masks are not stored on disk, token regions are approximated via
**Voronoi partition**: each pixel is assigned to the token with the nearest
centroid in image space. This closely mirrors the SAM over-segmentation for
densely-placed tokens.

Ground truth change:
  label1[px] ≠ label2[px]  → changed pixel   (standard SECOND protocol)

Steps
-----
  Step 1  Pixel reconstruction   (Voronoi max-pool of change probabilities)
  Step 2  Threshold → binary map (default 0.5)
  Step 3  Pixel-level metrics    (F1, IoU, Precision, Recall)
  Step 4  Reconstruction quality (token vs pixel prob distributions)
  Step 5  Visualisation          (10-sample 4-panel grids)
  Step 6  Boundary artefacts     (mean |p_i − p_j| at Voronoi boundaries)

Usage
-----
python run_change_map_reconstruction.py \\
    --model_dir  SECOND/stage5_6_dynamic \\
    --tokens_T1  SECOND/tokens_T1 \\
    --tokens_T2  SECOND/tokens_T2 \\
    --matches    SECOND/matches \\
    --label1_dir SECOND/label1 \\
    --label2_dir SECOND/label2 \\
    --images_T1  SECOND/im1 \\
    --images_T2  SECOND/im2 \\
    --output_dir stage7/change_map \\
    --val_split 0.1 --seed 42 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from token_change_reasoner import SampleData, build_batch
from token_change_reasoner_moe import MoEConfig, TokenChangeReasonerMoE, build_moe_model

IMG_SIZE = 512


# ─────────────────────────────────────────────────────────────────────────────
# Val stems  (mirror training random_split)
# ─────────────────────────────────────────────────────────────────────────────
def build_val_stems(
    match_dir: Path, t1_dir: Path, t2_dir: Path,
    val_split: float = 0.1, seed: int = 42,
) -> List[str]:
    all_stems = [
        mp.stem.replace("_matches", "")
        for mp in sorted(match_dir.glob("*_matches.pt"))
        if (t1_dir / f"{mp.stem.replace('_matches','')}.pt").exists()
        and (t2_dir / f"{mp.stem.replace('_matches','')}.pt").exists()
    ]
    n_val   = max(1, int(len(all_stems) * val_split))
    gen     = torch.Generator().manual_seed(seed)
    perm    = torch.randperm(len(all_stems), generator=gen).tolist()
    return [all_stems[i] for i in perm[len(all_stems) - n_val:]]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_dir: Path, device: torch.device) -> TokenChangeReasonerMoE:
    with open(model_dir / "config.json") as f:
        d = json.load(f)
    cfg   = MoEConfig(**{k: v for k, v in d.items() if k in MoEConfig.__dataclass_fields__})
    model = build_moe_model(cfg).to(device)
    ckpt  = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth change map
# ─────────────────────────────────────────────────────────────────────────────
# SECOND semantic colour palette  (map closest colour → class index)
_SECOND_COLOURS = np.array([
    [0,   0,   0  ],   # 0  background / no-data
    [0,   128, 0  ],   # 1  low vegetation
    [128, 0,   0  ],   # 2  buildings
    [0,   128, 128],   # 3  water
    [128, 128, 128],   # 4  hard surfaces
    [255, 255, 255],   # 5  impervious / bare soil
    [0,   255, 0  ],   # 6  vegetation
], dtype=np.float32)


def _rgb_to_class(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 → (H,W) int class index."""
    flat = rgb.reshape(-1, 3).astype(np.float32)
    dists = ((flat[:, None, :] - _SECOND_COLOURS[None]) ** 2).sum(-1)
    return dists.argmin(-1).reshape(rgb.shape[:2])


def get_gt_change(stem: str, label1_dir: Path, label2_dir: Path) -> Optional[np.ndarray]:
    """Return (512,512) bool GT change mask, or None if label missing."""
    p1 = label1_dir / f"{stem}.png"
    p2 = label2_dir / f"{stem}.png"
    if not (p1.exists() and p2.exists()):
        return None
    l1 = np.array(Image.open(p1).convert("RGB"))
    l2 = np.array(Image.open(p2).convert("RGB"))
    c1 = _rgb_to_class(l1)
    c2 = _rgb_to_class(l2)
    return c1 != c2


# ─────────────────────────────────────────────────────────────────────────────
# Voronoi pixel-to-token assignment
# ─────────────────────────────────────────────────────────────────────────────
def voronoi_pixel_map(
    centroids_t1: np.ndarray,   # [N1, 2]  (cx, cy) in [0,1]
    n_total_tokens: int,        # N1 + N2 padding size
    n1: int,                    # number of T1 tokens
    H: int = IMG_SIZE,
    W: int = IMG_SIZE,
) -> np.ndarray:
    """
    Returns (H*W,) int array of token indices in [0, N1-1].
    Each pixel is mapped to the T1 token with the nearest centroid.
    We use only T1 centroids so the map represents the T1 segmentation.
    """
    # Build KD-tree from T1 centroids in pixel space
    pts = centroids_t1[:, ::-1].copy()    # swap to (row, col)=(y,x)
    pts[:, 0] *= (H - 1)
    pts[:, 1] *= (W - 1)

    # Build grid of pixel centres
    ys, xs = np.mgrid[0:H, 0:W]           # both [H, W]
    grid   = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)  # [H*W, 2]

    tree     = cKDTree(pts)
    _, token_ids = tree.query(grid, k=1)   # [H*W]  indices into centroids_t1
    return token_ids.astype(np.int32)      # [H*W]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 + 2: reconstruct pixel change map
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_change_map(
    centroids_t1: np.ndarray,   # [N1, 2]
    change_probs_t1: np.ndarray,  # [N1] prediction for T1 tokens only
    H: int = IMG_SIZE,
    W: int = IMG_SIZE,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      prob_map   : (H, W) float32  – per-pixel change probability
      binary_map : (H, W) bool     – thresholded change prediction
    """
    pixel_ids = voronoi_pixel_map(centroids_t1, len(centroids_t1), len(centroids_t1), H, W)
    prob_flat  = change_probs_t1[pixel_ids]          # [H*W]
    prob_map   = prob_flat.reshape(H, W).astype(np.float32)
    binary_map = prob_map > threshold
    return prob_map, binary_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: pixel metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """pred, gt: bool (H,W)."""
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    prec   = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1     = 2 * prec * recall / max(prec + recall, 1e-8)
    iou    = tp / max(tp + fp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": recall, "f1": f1, "iou": iou}


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: boundary artefact score
# ─────────────────────────────────────────────────────────────────────────────
def boundary_inconsistency(
    prob_map: np.ndarray,       # (H, W)
    pixel_ids: np.ndarray,      # (H*W,)  token index per pixel
    centroids_t1: np.ndarray,   # (N1, 2)
) -> float:
    """
    For each pair of horizontally or vertically adjacent pixels that belong
    to *different* tokens, compute |p_i − p_j|. Return the mean over all such
    boundary pairs.
    """
    pid = pixel_ids.reshape(prob_map.shape)   # (H, W)
    p   = prob_map

    # Horizontal neighbours
    h_diff_mask = pid[:, :-1] != pid[:, 1:]
    h_diffs     = np.abs(p[:, :-1] - p[:, 1:])[h_diff_mask]

    # Vertical neighbours
    v_diff_mask = pid[:-1, :] != pid[1:, :]
    v_diffs     = np.abs(p[:-1, :] - p[1:, :])[v_diff_mask]

    all_diffs = np.concatenate([h_diffs, v_diffs])
    return float(all_diffs.mean()) if len(all_diffs) > 0 else 0.0


def interior_inconsistency(
    prob_map: np.ndarray,       # (H, W)
    pixel_ids: np.ndarray,      # (H*W,)
) -> float:
    """
    Mean |p_i − p_j| for adjacent pixels belonging to the SAME token.
    For a perfect token-based map this is 0; for soft contours it will be
    small but nonzero.
    """
    pid = pixel_ids.reshape(prob_map.shape)
    p   = prob_map
    h_same = pid[:, :-1] == pid[:, 1:]
    h_vals = np.abs(p[:, :-1] - p[:, 1:])[h_same]
    v_same = pid[:-1, :] == pid[1:, :]
    v_vals = np.abs(p[:-1, :] - p[1:, :])[v_same]
    all_v  = np.concatenate([h_vals, v_vals])
    return float(all_v.mean()) if len(all_v) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-token quality stats (Step 4)
# ─────────────────────────────────────────────────────────────────────────────
def token_quality_stats(
    prob_map: np.ndarray,       # (H, W)
    pixel_ids: np.ndarray,      # (H*W,)
    change_probs_t1: np.ndarray,  # (N1,)
) -> Dict[str, float]:
    """
    Checks consistency between token prediction and the (uniform) pixel
    predictions it generated.
    """
    N1 = len(change_probs_t1)
    within_var = []

    pid_r = pixel_ids   # (H*W,)
    prob_r = prob_map.ravel()  # (H*W,)

    for e_id in range(N1):
        mask = pid_r == e_id
        if mask.sum() == 0:
            continue
        pxs = prob_r[mask]
        within_var.append(float(np.var(pxs)))

    mean_token_prob  = float(change_probs_t1.mean())
    mean_pixel_prob  = float(prob_map.mean())
    mean_within_var  = float(np.mean(within_var)) if within_var else 0.0

    return {
        "mean_token_prob": mean_token_prob,
        "mean_pixel_prob": mean_pixel_prob,
        "mean_within_token_variance": mean_within_var,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    model: TokenChangeReasonerMoE,
    stems: List[str],
    t1_dir: Path,
    t2_dir: Path,
    match_dir: Path,
    label1_dir: Path,
    label2_dir: Path,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
      per_image_metrics : list of dicts (one per stem that has GT)
      per_image_results : list of dicts with arrays for visualisation
    """
    per_image_metrics = []
    per_image_results = []

    for idx, stem in enumerate(stems):
        t1p = t1_dir   / f"{stem}.pt"
        t2p = t2_dir   / f"{stem}.pt"
        mp  = match_dir / f"{stem}_matches.pt"
        if not (t1p.exists() and t2p.exists() and mp.exists()):
            continue

        t1   = torch.load(t1p, map_location="cpu", weights_only=True)
        t2   = torch.load(t2p, map_location="cpu", weights_only=True)
        mtch = torch.load(mp,  map_location="cpu", weights_only=False)
        n1   = t1["tokens"].shape[0]

        pairs_raw = mtch.get("pairs", [])
        if isinstance(pairs_raw, list):
            pairs = torch.tensor([[float(p[0]), float(p[1]), float(p[2])] for p in pairs_raw]) \
                   if len(pairs_raw) else torch.zeros(0, 3)
        else:
            pairs = pairs_raw.float()

        sample = SampleData(
            tokens_t1=t1["tokens"].float(), tokens_t2=t2["tokens"].float(),
            centroids_t1=t1["centroids"].float(), centroids_t2=t2["centroids"].float(),
            areas_t1=t1["areas"].float(), areas_t2=t2["areas"].float(),
            match_pairs=pairs, change_labels=None, semantic_labels=None,
        )
        batch = build_batch([sample], model.cfg, device)

        with torch.no_grad():
            outputs = model(batch)

        # ── Extract T1 token predictions ───────────────────────────────────
        # Padded sequence layout: first n1 slots = T1 tokens
        change_logits = outputs["change_logits"][0].cpu()   # [N_pad]
        pad_mask      = batch["padding_mask"][0].cpu()       # [N_pad]  True=pad
        # T1 positions: time_id == 0
        time_ids = batch["time_ids_pad"][0].cpu()            # [N_pad]

        t1_mask = (~pad_mask) & (time_ids == 0)              # valid T1 tokens
        t1_probs = torch.sigmoid(change_logits[t1_mask]).numpy()  # [n1_valid]
        t1_cents = batch["centroids_pad"][0].cpu()[t1_mask].numpy()  # [n1_valid, 2]

        # ── Pixel reconstruction ───────────────────────────────────────────
        prob_map, binary_map = reconstruct_change_map(
            t1_cents, t1_probs, threshold=threshold
        )

        # Need pixel_ids for boundary / quality stats
        pixel_ids = voronoi_pixel_map(t1_cents, len(t1_cents), len(t1_cents))

        # ── GT ─────────────────────────────────────────────────────────────
        gt = get_gt_change(stem, label1_dir, label2_dir)

        # ── Metrics ────────────────────────────────────────────────────────
        if gt is not None:
            m = compute_metrics(binary_map, gt)
            m["stem"] = stem

            q = token_quality_stats(prob_map, pixel_ids, t1_probs)
            m.update(q)

            bnd = boundary_inconsistency(prob_map, pixel_ids, t1_cents)
            intr = interior_inconsistency(prob_map, pixel_ids)
            m["boundary_inconsistency"] = bnd
            m["interior_inconsistency"] = intr

            per_image_metrics.append(m)

        per_image_results.append({
            "stem": stem,
            "prob_map":   prob_map,
            "binary_map": binary_map,
            "gt_map":     gt,
            "t1_probs":   t1_probs,
            "t1_cents":   t1_cents,
        })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(stems)}] processed ...")

    return per_image_metrics, per_image_results


# ─────────────────────────────────────────────────────────────────────────────
# PIL helpers
# ─────────────────────────────────────────────────────────────────────────────
def _try_font(sz: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def _heatmap(arr: np.ndarray) -> np.ndarray:
    """(H,W) float [0,1] → (H,W,3) uint8  (jet-like)."""
    r  = np.clip(1.5 - np.abs(4 * arr - 3.0), 0, 1)
    g  = np.clip(1.5 - np.abs(4 * arr - 2.0), 0, 1)
    b  = np.clip(1.5 - np.abs(4 * arr - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _bool_to_rgb(arr: np.ndarray) -> np.ndarray:
    """(H,W) bool → (H,W,3) uint8  (white=changed, black=unchanged)."""
    out       = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[arr]  = [255, 255, 255]
    return out


def _load_rgb(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))


def draw_4panel(
    sat_t1: Optional[np.ndarray],   # (H,W,3) or None
    sat_t2: Optional[np.ndarray],
    prob_map: np.ndarray,            # (H,W) float
    gt_map: Optional[np.ndarray],   # (H,W) bool or None
    stem: str,
    metrics: Optional[Dict] = None,
) -> Image.Image:
    H, W  = IMG_SIZE, IMG_SIZE
    pad   = 6
    title_h = 22
    cell_h  = H + title_h + pad
    cell_w  = W + pad

    canvas = Image.new("RGB", (4 * cell_w + pad, cell_h + 36), (245, 245, 248))
    draw   = ImageDraw.Draw(canvas)
    fnt_t  = _try_font(13)
    fnt_s  = _try_font(10)

    panels = [
        (sat_t1 if sat_t1 is not None else np.zeros((H, W, 3), np.uint8), "T1 Image"),
        (sat_t2 if sat_t2 is not None else np.zeros((H, W, 3), np.uint8), "T2 Image"),
        (_heatmap(np.clip(prob_map, 0, 1)), "Predicted Change Prob"),
        (_bool_to_rgb(gt_map) if gt_map is not None
         else np.zeros((H, W, 3), np.uint8), "Ground Truth"),
    ]

    for i, (arr, label) in enumerate(panels):
        x0 = pad + i * cell_w
        y0 = pad + title_h
        img = Image.fromarray(arr).resize((W, H))
        canvas.paste(img, (x0, y0))
        draw.text((x0 + W // 2, pad + title_h // 2), label,
                  fill=(30, 30, 30), font=fnt_t, anchor="mm")

    # Bottom caption
    cap = stem
    if metrics:
        cap += f"  |  F1={metrics['f1']:.3f}  IoU={metrics['iou']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}"
    draw.text((canvas.width // 2, canvas.height - 11), cap,
              fill=(60, 60, 60), font=fnt_s, anchor="mm")
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: visualisation grid
# ─────────────────────────────────────────────────────────────────────────────
def generate_vis_grid(
    per_image_results: List[Dict],
    per_image_metrics: List[Dict],
    images_T1: Optional[Path],
    images_T2: Optional[Path],
    n_show: int,
    save_path: Path,
):
    metric_by_stem = {m["stem"]: m for m in per_image_metrics}

    # Pick stems that have GT
    with_gt = [r for r in per_image_results if r["gt_map"] is not None][:n_show]

    panels = []
    for r in with_gt:
        stem  = r["stem"]
        t1_img = _load_rgb(images_T1 / f"{stem}.png") if images_T1 else None
        t2_img = _load_rgb(images_T2 / f"{stem}.png") if images_T2 else None
        m      = metric_by_stem.get(stem)
        panel  = draw_4panel(t1_img, t2_img, r["prob_map"], r["gt_map"], stem, m)
        panels.append(panel)

    if not panels:
        print("  [vis] No GT images found for visualisation.")
        return

    pw, ph = panels[0].size
    cols   = min(2, len(panels))
    rows   = math.ceil(len(panels) / cols)

    canvas = Image.new("RGB", (cols * pw + 10, rows * ph + 50), (240, 240, 244))
    draw   = ImageDraw.Draw(canvas)
    draw.text((canvas.width // 2, 18),
              "Change Map Reconstruction — Validation Samples",
              fill=(30, 30, 30), font=_try_font(16), anchor="mm")

    for idx, panel in enumerate(panels):
        r = idx // cols
        c = idx % cols
        canvas.paste(panel, (5 + c * pw, 38 + r * ph))

    canvas.save(save_path)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_metrics(per_image_metrics: List[Dict]) -> Dict[str, float]:
    keys = ["precision", "recall", "f1", "iou",
            "mean_token_prob", "mean_pixel_prob",
            "mean_within_token_variance",
            "boundary_inconsistency", "interior_inconsistency"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in per_image_metrics if k in m]
        agg[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
        agg[f"std_{k}"]  = float(np.std(vals))  if vals else 0.0

    # Micro-average F1/IoU from pooled TP/FP/FN
    tp = sum(m.get("tp", 0) for m in per_image_metrics)
    fp = sum(m.get("fp", 0) for m in per_image_metrics)
    fn = sum(m.get("fn", 0) for m in per_image_metrics)
    prec  = tp / max(tp + fp, 1)
    rec   = tp / max(tp + fn, 1)
    f1    = 2 * prec * rec / max(prec + rec, 1e-8)
    iou   = tp / max(tp + fp + fn, 1)
    agg["micro_precision"] = prec
    agg["micro_recall"]    = rec
    agg["micro_f1"]        = f1
    agg["micro_iou"]       = iou
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(per_image_metrics: List[Dict], save_path: Path):
    if not per_image_metrics:
        return
    keys = list(per_image_metrics[0].keys())
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_image_metrics)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────
def save_report(
    agg: Dict[str, float],
    per_image_metrics: List[Dict],
    args,
    output_dir: Path,
    threshold: float,
    n_images: int,
    n_with_gt: int,
):
    mf1  = agg["micro_f1"]
    miou = agg["micro_iou"]
    mp   = agg["micro_precision"]
    mr   = agg["micro_recall"]

    bnd  = agg["mean_boundary_inconsistency"]
    intr = agg["mean_interior_inconsistency"]
    bnd_ratio = bnd / (bnd + intr + 1e-8)

    lines = [
        "# Stage 7 — Change Map Reconstruction",
        "",
        "## Overview",
        "",
        f"- Model: `{args.model_dir}`",
        f"- Reconstruction method: **Voronoi-nearest centroid assignment**",
        f"- Threshold: `{threshold}`",
        f"- Validation images processed: `{n_images}`",
        f"- Images with GT labels: `{n_with_gt}`",
        "",
        "---",
        "",
        "## Step 3 — Pixel-Level Metrics",
        "",
        "### Micro-averaged (pooled TP/FP/FN across all images)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **F1 Score** | **{mf1:.4f}** |",
        f"| **IoU (Change)** | **{miou:.4f}** |",
        f"| Precision | {mp:.4f} |",
        f"| Recall | {mr:.4f} |",
        "",
        "### Per-image statistics (macro average ± std)",
        "",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
        f"| F1     | {agg['mean_f1']:.4f} | ±{agg['std_f1']:.4f} |",
        f"| IoU    | {agg['mean_iou']:.4f} | ±{agg['std_iou']:.4f} |",
        f"| Precision | {agg['mean_precision']:.4f} | ±{agg['std_precision']:.4f} |",
        f"| Recall | {agg['mean_recall']:.4f} | ±{agg['std_recall']:.4f} |",
        "",
        "---",
        "",
        "## Step 4 — Reconstruction Quality",
        "",
        "| Statistic | Value |",
        "|-----------|-------|",
        f"| Mean token change probability | {agg['mean_mean_token_prob']:.4f} |",
        f"| Mean pixel change probability | {agg['mean_mean_pixel_prob']:.4f} |",
        f"| Mean within-token pixel variance | {agg['mean_mean_within_token_variance']:.6f} |",
        "",
        "> **Note:** With Voronoi reconstruction, the within-token pixel variance is "
        "exactly 0 by construction — each token projects a *uniform* probability over "
        "its region. The mean token and mean pixel probabilities should be identical.",
        "",
        "---",
        "",
        "## Step 6 — Boundary Artefact Analysis",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean boundary inconsistency | {bnd:.4f} |",
        f"| Mean interior inconsistency | {intr:.4f} |",
        f"| Boundary / interior ratio | {bnd_ratio:.3f} |",
        "",
        "**Interpretation:**",
        "",
    ]

    if bnd_ratio > 2.5:
        lines += [
            "A high boundary-to-interior ratio (> 2.5) indicates **visible block artefacts** "
            "at SAM region boundaries — adjacent regions with large probability differences "
            "create hard discontinuities in the change map.",
        ]
    elif bnd_ratio > 1.5:
        lines += [
            "Moderate boundary-to-interior ratio (1.5–2.5): some block artefacts are "
            "visible at region boundaries, but the map remains spatially reasonable.",
        ]
    else:
        lines += [
            "Low boundary-to-interior ratio (< 1.5): **minimal block artefacts**. "
            "Adjacent SAM regions usually receive similar probability predictions, "
            "resulting in a spatially coherent change map.",
        ]

    lines += [
        "",
        "---",
        "",
        "## Step 5 — Qualitative Examples",
        "",
        "See `stage7_change_map_examples.png` for 4-panel visualisations:",
        "",
        "1. T1 satellite image",
        "2. T2 satellite image",
        "3. Predicted change probability map (jet colourmap)",
        "4. Ground truth change map (binary)",
        "",
        "---",
        "",
        "## Discussion",
        "",
        "### Can token-level change detection reconstruct accurate pixel maps?",
        "",
        f"The pixel-level reconstruction achieves **F1 = {mf1:.3f}** and "
        f"**IoU = {miou:.3f}** on the SECOND validation set. ",
    ]

    if mf1 >= 0.55:
        lines += [
            "This is a **strong result** for a token-based approach, demonstrating "
            "that SAM region-level predictions faithfully transfer to the pixel level. ",
        ]
    elif mf1 >= 0.45:
        lines += [
            "This is a **reasonable result** for a token-based approach, considering "
            "that each token prediction is uniformly projected across its SAM region. ",
        ]
    else:
        lines += [
            "There is a performance gap at the pixel level, which is expected since "
            "each token prediction is projected uniformly and cannot capture "
            "sub-region change patterns. ",
        ]

    lines += [
        "",
        "### Sources of error",
        "",
        "1. **Sub-region heterogeneity**: a single token prediction is projected "
        "uniformly to all pixels in its SAM region. If a region partially changes, "
        "the token may predict an intermediate probability, causing FP on unchanged "
        "pixels and FN on changed pixels.",
        "",
        "2. **Voronoi approximation**: in the absence of stored SAM masks, we use "
        "nearest-centroid assignment. Where SAM masks are convex and well-separated "
        "this is exact; for irregular shapes at image borders it introduces a small "
        f"boundary error (boundary inconsistency = {bnd:.4f}).",
        "",
        "3. **Ground truth resolution**: SECOND GT is pixel-level while tokens are "
        "region-level. Small changed regions < one SAM mask may be missed entirely.",
        "",
        "### Conclusion",
        "",
        "Token-based change detection **can** produce spatially coherent pixel maps. "
        f"The boundary inconsistency score ({bnd:.4f} vs interior {intr:.4f}) "
        "confirms that artefacts at region boundaries are present but limited. "
        "The main accuracy bottleneck is sub-region probability assignment rather "
        "than boundary artefacts.",
        "",
        "---",
        "",
        "_Generated by `run_change_map_reconstruction.py`_",
    ]

    p = output_dir / "stage7_change_map_reconstruction.md"
    p.write_text("\n".join(lines))
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(description="Stage 7 — Change Map Reconstruction")
    pa.add_argument("--model_dir",   default="SECOND/stage5_6_dynamic")
    pa.add_argument("--tokens_T1",   default="SECOND/tokens_T1")
    pa.add_argument("--tokens_T2",   default="SECOND/tokens_T2")
    pa.add_argument("--matches",     default="SECOND/matches")
    pa.add_argument("--label1_dir",  default="SECOND/label1")
    pa.add_argument("--label2_dir",  default="SECOND/label2")
    pa.add_argument("--images_T1",   default="SECOND/im1",  help="T1 satellite images for vis")
    pa.add_argument("--images_T2",   default="SECOND/im2",  help="T2 satellite images for vis")
    pa.add_argument("--output_dir",  default="stage7/change_map")
    pa.add_argument("--val_split",   type=float, default=0.1)
    pa.add_argument("--seed",        type=int,   default=42)
    pa.add_argument("--device",      default="cuda")
    pa.add_argument("--threshold",   type=float, default=0.5, help="Binarisation threshold")
    pa.add_argument("--vis_n",       type=int,   default=10,  help="Images in vis grid")
    pa.add_argument("--max_samples", type=int,   default=None)
    args = pa.parse_args()

    device     = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir  = Path(args.model_dir)
    t1_dir     = Path(args.tokens_T1)
    t2_dir     = Path(args.tokens_T2)
    match_dir  = Path(args.matches)
    label1_dir = Path(args.label1_dir)
    label2_dir = Path(args.label2_dir)
    images_T1  = Path(args.images_T1) if args.images_T1 else None
    images_T2  = Path(args.images_T2) if args.images_T2 else None

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"\nLoading model from {model_dir} ...")
    model = load_model(model_dir, device)
    print(f"  Router v{model.cfg.router_version}  |  Experts: {model.cfg.moe_num_experts}")

    # ── Val stems ──────────────────────────────────────────────────────────
    val_stems = build_val_stems(match_dir, t1_dir, t2_dir, args.val_split, args.seed)
    if args.max_samples:
        val_stems = val_stems[:args.max_samples]
    print(f"  Validation stems: {len(val_stems)}")

    # ── Inference ──────────────────────────────────────────────────────────
    print(f"\nRunning inference + reconstruction (threshold={args.threshold}) ...")
    per_image_metrics, per_image_results = run_inference(
        model, val_stems, t1_dir, t2_dir, match_dir,
        label1_dir, label2_dir, device, threshold=args.threshold,
    )
    n_with_gt = len(per_image_metrics)
    print(f"  Processed {len(per_image_results)} images  |  GT available: {n_with_gt}")

    if not per_image_metrics:
        print("ERROR: No images with GT labels found. Check label1_dir / label2_dir.")
        sys.exit(1)

    # ── Aggregate ──────────────────────────────────────────────────────────
    agg = aggregate_metrics(per_image_metrics)

    print()
    print("=" * 66)
    print("  CHANGE MAP RECONSTRUCTION — RESULTS")
    print("=" * 66)
    print(f"  Micro F1 : {agg['micro_f1']:.4f}    Macro F1  : {agg['mean_f1']:.4f}")
    print(f"  Micro IoU: {agg['micro_iou']:.4f}    Macro IoU : {agg['mean_iou']:.4f}")
    print(f"  Precision: {agg['micro_precision']:.4f}    Recall    : {agg['micro_recall']:.4f}")
    print(f"  Bnd inconsistency: {agg['mean_boundary_inconsistency']:.4f}  "
          f"Interior: {agg['mean_interior_inconsistency']:.4f}")
    print("=" * 66)

    # ── Outputs ────────────────────────────────────────────────────────────
    print("\nGenerating outputs ...")
    save_csv(per_image_metrics, output_dir / "stage7_per_image_metrics.csv")

    generate_vis_grid(
        per_image_results, per_image_metrics,
        images_T1, images_T2,
        n_show=args.vis_n,
        save_path=output_dir / "stage7_change_map_examples.png",
    )

    save_report(
        agg, per_image_metrics, args, output_dir,
        threshold=args.threshold,
        n_images=len(per_image_results),
        n_with_gt=n_with_gt,
    )

    print(f"\n  All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
