#!/usr/bin/env python3
"""Stage 8C: Compare Voronoi, raw SAM, and SAM-partition reconstruction.

Goal:
Convert overlapping SAM proposals into a full non-overlapping partition and test
whether partition-based reconstruction improves pixel-level change detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
SAM2_REPO = ROOT / "sam2"
if SAM2_REPO.is_dir():
    sys.path.insert(0, str(SAM2_REPO))

from token_change_reasoner import SampleData, build_batch
from token_change_reasoner_moe import MoEConfig, TokenChangeReasonerMoE, build_moe_model
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

IMG_SIZE = 512
_SECOND_COLOURS = np.array(
    [
        [0, 0, 0],
        [0, 128, 0],
        [128, 0, 0],
        [0, 128, 128],
        [128, 128, 128],
        [255, 255, 255],
        [0, 255, 0],
    ],
    dtype=np.float32,
)


@dataclass
class MaskItem:
    mask: np.ndarray
    centroid: np.ndarray
    area_ratio: float
    score: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rgb_to_class(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    d = ((flat[:, None, :] - _SECOND_COLOURS[None, :, :]) ** 2).sum(-1)
    return d.argmin(-1).reshape(rgb.shape[:2])


def get_gt_change(stem: str, label1_dir: Path, label2_dir: Path) -> Optional[np.ndarray]:
    p1 = label1_dir / f"{stem}.png"
    p2 = label2_dir / f"{stem}.png"
    if not (p1.exists() and p2.exists()):
        return None
    l1 = np.array(Image.open(p1).convert("RGB"))
    l2 = np.array(Image.open(p2).convert("RGB"))
    return _rgb_to_class(l1) != _rgb_to_class(l2)


def build_val_stems(match_dir: Path, t1_dir: Path, t2_dir: Path, val_split: float, seed: int) -> List[str]:
    stems = [
        mp.stem.replace("_matches", "")
        for mp in sorted(match_dir.glob("*_matches.pt"))
        if (t1_dir / f"{mp.stem.replace('_matches', '')}.pt").exists()
        and (t2_dir / f"{mp.stem.replace('_matches', '')}.pt").exists()
    ]
    n_val = max(1, int(len(stems) * val_split))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(stems), generator=gen).tolist()
    return [stems[i] for i in perm[len(stems) - n_val :]]


def load_model(model_dir: Path, device: torch.device) -> TokenChangeReasonerMoE:
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        cfg_d = json.load(f)
    cfg = MoEConfig(**{k: v for k, v in cfg_d.items() if k in MoEConfig.__dataclass_fields__})
    model = build_moe_model(cfg).to(device)
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_amg(args, device: torch.device) -> SAM2AutomaticMaskGenerator:
    ckpt = Path(args.sam2_ckpt)
    if not ckpt.is_absolute():
        ckpt = SAM2_REPO / args.sam2_ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")

    sam2_model = build_sam2(args.sam2_config, str(ckpt), device=str(device))
    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        min_mask_region_area=args.min_mask_area,
        output_mode="binary_mask",
    )


def generate_masks(image_rgb: np.ndarray, amg: SAM2AutomaticMaskGenerator) -> List[MaskItem]:
    raw = amg.generate(image_rgb)
    items: List[MaskItem] = []
    for m in raw:
        seg = m["segmentation"]
        if seg.dtype != np.bool_:
            seg = seg.astype(bool)
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        cx = float(xs.mean()) / (IMG_SIZE - 1)
        cy = float(ys.mean()) / (IMG_SIZE - 1)
        area_ratio = float(seg.sum()) / (IMG_SIZE * IMG_SIZE)
        score = float(m.get("predicted_iou", 0.0))
        items.append(MaskItem(mask=seg, centroid=np.array([cx, cy], dtype=np.float32), area_ratio=area_ratio, score=score))

    if not items:
        full = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)
        items.append(MaskItem(mask=full, centroid=np.array([0.5, 0.5], dtype=np.float32), area_ratio=1.0, score=0.0))

    return items


def align_masks_to_tokens(mask_items: List[MaskItem], token_centroids: np.ndarray, token_areas: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Align regenerated masks to stored token order by centroid+area Hungarian matching."""
    m_cent = np.stack([m.centroid for m in mask_items], axis=0)
    m_area = np.array([m.area_ratio for m in mask_items], dtype=np.float32)
    m_score = np.array([m.score for m in mask_items], dtype=np.float32)

    d_cent = np.sqrt(((token_centroids[:, None, :] - m_cent[None, :, :]) ** 2).sum(axis=2))
    d_area = np.abs(token_areas[:, None] - m_area[None, :])
    cost = 0.7 * d_cent + 0.3 * d_area

    r, c = linear_sum_assignment(cost)
    aligned_masks: List[Optional[np.ndarray]] = [None] * len(token_centroids)
    aligned_scores = np.zeros(len(token_centroids), dtype=np.float32)
    aligned_areas = np.zeros(len(token_centroids), dtype=np.float32)

    for i, j in zip(r.tolist(), c.tolist()):
        aligned_masks[i] = mask_items[j].mask
        aligned_scores[i] = m_score[j]
        aligned_areas[i] = m_area[j]

    nearest = np.argmin(d_cent, axis=1)
    for i in range(len(aligned_masks)):
        if aligned_masks[i] is None:
            j = int(nearest[i])
            aligned_masks[i] = mask_items[j].mask
            aligned_scores[i] = m_score[j]
            aligned_areas[i] = m_area[j]

    return [m.astype(bool) for m in aligned_masks], aligned_scores, aligned_areas


def compute_token_probs(model: TokenChangeReasonerMoE, t1: Dict, t2: Dict, mtch: Dict, device: torch.device) -> np.ndarray:
    pairs_raw = mtch.get("pairs", [])
    if isinstance(pairs_raw, list):
        pairs = (
            torch.tensor([[float(p[0]), float(p[1]), float(p[2])] for p in pairs_raw], dtype=torch.float32)
            if len(pairs_raw)
            else torch.zeros((0, 3), dtype=torch.float32)
        )
    else:
        pairs = pairs_raw.float()

    sample = SampleData(
        tokens_t1=t1["tokens"].float(),
        tokens_t2=t2["tokens"].float(),
        centroids_t1=t1["centroids"].float(),
        centroids_t2=t2["centroids"].float(),
        areas_t1=t1["areas"].float(),
        areas_t2=t2["areas"].float(),
        match_pairs=pairs,
        change_labels=None,
        semantic_labels=None,
    )
    batch = build_batch([sample], model.cfg, device)
    with torch.no_grad():
        out = model(batch)

    logits = out["change_logits"][0].cpu()
    pad = batch["padding_mask"][0].cpu()
    tid = batch["time_ids_pad"][0].cpu()
    m = (~pad) & (tid == 0)
    return torch.sigmoid(logits[m]).numpy()


def voronoi_partition(centroids: np.ndarray, H: int = IMG_SIZE, W: int = IMG_SIZE) -> np.ndarray:
    pts = centroids[:, ::-1].copy()
    pts[:, 0] *= (H - 1)
    pts[:, 1] *= (W - 1)
    ys, xs = np.mgrid[0:H, 0:W]
    grid = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    tree = cKDTree(pts)
    _, ids = tree.query(grid, k=1)
    return ids.astype(np.int32).reshape(H, W)


def raw_sam_projection(masks: Sequence[np.ndarray], token_probs: np.ndarray, H: int = IMG_SIZE, W: int = IMG_SIZE) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.float32)
    for i, m in enumerate(masks):
        p = float(token_probs[i])
        out[m] = np.maximum(out[m], p)
    return out


def sam_partition_map(
    masks: Sequence[np.ndarray],
    centroids: np.ndarray,
    scores: Optional[np.ndarray] = None,
    areas: Optional[np.ndarray] = None,
    chooser: str = "area",
    H: int = IMG_SIZE,
    W: int = IMG_SIZE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Convert overlapping proposals into an all-pixel non-overlapping partition.

    chooser:
      - area: highest-area covering mask wins overlap
      - score: highest-score covering mask wins overlap
    """
    N = len(masks)
    stack = np.stack([m.astype(np.uint8) for m in masks], axis=0)  # [N,H,W]
    cover_count = stack.sum(axis=0).astype(np.int32)

    if areas is None:
        areas = np.array([m.sum() / float(H * W) for m in masks], dtype=np.float32)
    if scores is None:
        scores = np.zeros(N, dtype=np.float32)

    if chooser == "score":
        priority = scores
    else:
        priority = areas

    order = np.argsort(priority)  # ascending; later overwrite means higher wins
    part = np.full((H, W), -1, dtype=np.int32)
    for idx in order.tolist():
        part[masks[idx]] = idx

    # Fill uncovered pixels by nearest centroid.
    uncovered = part < 0
    if uncovered.any():
        pts = centroids[:, ::-1].copy()
        pts[:, 0] *= (H - 1)
        pts[:, 1] *= (W - 1)
        tree = cKDTree(pts)
        ys, xs = np.where(uncovered)
        q = np.stack([ys, xs], axis=1).astype(np.float32)
        _, ids = tree.query(q, k=1)
        part[ys, xs] = ids.astype(np.int32)

    overlap_before = float((cover_count > 1).mean())
    overlap_after = float(np.mean(False))  # guaranteed non-overlap partition
    coverage_before = float((cover_count > 0).mean())
    coverage_after = float(np.mean(part >= 0))

    diag = {
        "partition_coverage": coverage_after,
        "coverage_before": coverage_before,
        "overlap_before": overlap_before,
        "overlap_after": overlap_after,
        "num_masks": float(N),
        "avg_mask_area": float(np.mean([m.sum() for m in masks])) if N else 0.0,
    }
    return part, diag


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "iou": float(iou),
    }


def aggregate(rows: List[Dict], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ["precision", "recall", "f1", "iou"]:
        vals = [r[f"{prefix}_{k}"] for r in rows]
        out[f"macro_{prefix}_{k}"] = float(np.mean(vals)) if vals else 0.0
        out[f"std_{prefix}_{k}"] = float(np.std(vals)) if vals else 0.0

    tp = sum(int(r[f"{prefix}_tp"]) for r in rows)
    fp = sum(int(r[f"{prefix}_fp"]) for r in rows)
    fn = sum(int(r[f"{prefix}_fn"]) for r in rows)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    out[f"micro_{prefix}_precision"] = float(p)
    out[f"micro_{prefix}_recall"] = float(r)
    out[f"micro_{prefix}_f1"] = float(f1)
    out[f"micro_{prefix}_iou"] = float(iou)
    return out


def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def partition_to_color(part: np.ndarray) -> np.ndarray:
    n = int(part.max()) + 1
    rng = np.random.default_rng(123)
    palette = rng.integers(0, 255, size=(max(n, 1), 3), dtype=np.uint8)
    return palette[part]


def draw_mask_boundaries(img: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray:
    out = img.copy()
    for m in masks:
        mm = np.ascontiguousarray(m.astype(np.uint8))
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, (255, 255, 0), 1)
    return out


def heatmap(arr: np.ndarray) -> np.ndarray:
    a = np.clip(arr, 0, 1)
    return (plt.get_cmap("jet")(a)[..., :3] * 255).astype(np.uint8)


def make_visual_grid(records: List[Dict], save_path: Path, n_show: int) -> None:
    picks = records[:n_show]
    if not picks:
        return

    cols = 5
    rows = len(picks) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, r in enumerate(picks):
        # First row block
        ax0 = axes[2 * i + 0]
        ax0[0].imshow(r["t1"])
        ax0[0].set_title(f"{r['stem']} T1")
        ax0[1].imshow(r["t2"])
        ax0[1].set_title("T2")
        ax0[2].imshow(r["sam_overlay"])
        ax0[2].set_title("SAM proposals")
        ax0[3].imshow(r["partition_rgb"])
        ax0[3].set_title("SAM partition")
        ax0[4].imshow(r["vor_rgb"])
        ax0[4].set_title("Voronoi regions")

        # Second row block
        ax1 = axes[2 * i + 1]
        ax1[0].imshow(r["pred_vor"], cmap="jet", vmin=0, vmax=1)
        ax1[0].set_title("Pred Voronoi")
        ax1[1].imshow(r["pred_raw"], cmap="jet", vmin=0, vmax=1)
        ax1[1].set_title("Pred Raw SAM")
        ax1[2].imshow(r["pred_part"], cmap="jet", vmin=0, vmax=1)
        ax1[2].set_title("Pred SAM Partition")
        ax1[3].imshow(r["gt"].astype(np.uint8), cmap="gray")
        ax1[3].set_title("GT change")
        txt = (
            f"IoU V={r['vor_iou']:.3f} R={r['raw_iou']:.3f} P={r['part_iou']:.3f}\n"
            f"F1  V={r['vor_f1']:.3f} R={r['raw_f1']:.3f} P={r['part_f1']:.3f}"
        )
        ax1[4].text(0.02, 0.5, txt, fontsize=11)
        ax1[4].axis("off")

        for a in np.concatenate([ax0, ax1]):
            a.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def save_report(path: Path, args, agg: Dict[str, float], diag: Dict[str, float], n_proc: int, n_gt: int) -> None:
    def micro_row(name: str, pfx: str) -> str:
        return (
            f"| {name} | {agg[f'micro_{pfx}_f1']:.4f} | {agg[f'micro_{pfx}_iou']:.4f} | "
            f"{agg[f'micro_{pfx}_precision']:.4f} | {agg[f'micro_{pfx}_recall']:.4f} |"
        )

    lines = [
        "# Stage 8C - SAM Partition Reconstruction",
        "",
        "## Overview",
        "",
        f"- Model: `{args.model_dir}`",
        f"- Processed validation images: `{n_proc}`",
        f"- Images with GT: `{n_gt}`",
        f"- Threshold: `{args.threshold}`",
        f"- Partition chooser: `{args.partition_chooser}`",
        "",
        "---",
        "",
        "## Metrics (Micro)",
        "",
        "| Reconstruction | F1 | IoU | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
        micro_row("Voronoi", "vor"),
        micro_row("Raw SAM projection", "raw"),
        micro_row("SAM partition", "part"),
        "",
        "## Metrics (Macro mean +- std)",
        "",
        "| Reconstruction | F1 | IoU | Precision | Recall |",
        "|---|---|---|---|---|",
        (
            f"| Voronoi | {agg['macro_vor_f1']:.4f} +- {agg['std_vor_f1']:.4f} "
            f"| {agg['macro_vor_iou']:.4f} +- {agg['std_vor_iou']:.4f} "
            f"| {agg['macro_vor_precision']:.4f} +- {agg['std_vor_precision']:.4f} "
            f"| {agg['macro_vor_recall']:.4f} +- {agg['std_vor_recall']:.4f} |"
        ),
        (
            f"| Raw SAM projection | {agg['macro_raw_f1']:.4f} +- {agg['std_raw_f1']:.4f} "
            f"| {agg['macro_raw_iou']:.4f} +- {agg['std_raw_iou']:.4f} "
            f"| {agg['macro_raw_precision']:.4f} +- {agg['std_raw_precision']:.4f} "
            f"| {agg['macro_raw_recall']:.4f} +- {agg['std_raw_recall']:.4f} |"
        ),
        (
            f"| SAM partition | {agg['macro_part_f1']:.4f} +- {agg['std_part_f1']:.4f} "
            f"| {agg['macro_part_iou']:.4f} +- {agg['std_part_iou']:.4f} "
            f"| {agg['macro_part_precision']:.4f} +- {agg['std_part_precision']:.4f} "
            f"| {agg['macro_part_recall']:.4f} +- {agg['std_part_recall']:.4f} |"
        ),
        "",
        "---",
        "",
        "## Partition Diagnostics",
        "",
        f"- partition_coverage: {diag['partition_coverage']:.4f}",
        f"- overlap_before: {diag['overlap_before']:.4f}",
        f"- overlap_after: {diag['overlap_after']:.4f}",
        f"- num_masks: {diag['num_masks']:.2f}",
        f"- avg_mask_area: {diag['avg_mask_area']:.2f}",
        "",
        "---",
        "",
        "## Visualisation",
        "",
        "See `stage8_partition_comparison.png` for sample panels:",
        "- T1, T2",
        "- SAM proposals",
        "- SAM partition",
        "- Voronoi regions",
        "- prediction maps (Voronoi, raw SAM, partition)",
        "- GT change",
        "",
        "_Generated by `run_stage8c_partition_reconstruction.py`_",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    t1_dir = Path(args.tokens_T1)
    t2_dir = Path(args.tokens_T2)
    match_dir = Path(args.matches)
    img1_dir = Path(args.images_T1)
    img2_dir = Path(args.images_T2)
    label1_dir = Path(args.label1_dir)
    label2_dir = Path(args.label2_dir)

    print(f"Loading model from {model_dir} ...")
    model = load_model(model_dir, device)
    print(f"  Router v{model.cfg.router_version} | Experts: {model.cfg.moe_num_experts}")

    print("Loading SAM generator ...")
    amg = load_amg(args, device)

    stems = build_val_stems(match_dir, t1_dir, t2_dir, args.val_split, args.seed)
    if args.max_samples is not None:
        stems = stems[: args.max_samples]
    print(f"Validation stems: {len(stems)}")

    rows: List[Dict] = []
    vis_records: List[Dict] = []
    diag_acc = {
        "partition_coverage": [],
        "overlap_before": [],
        "overlap_after": [],
        "num_masks": [],
        "avg_mask_area": [],
    }

    for i, stem in enumerate(stems):
        t1p = t1_dir / f"{stem}.pt"
        t2p = t2_dir / f"{stem}.pt"
        mp = match_dir / f"{stem}_matches.pt"
        p1 = img1_dir / f"{stem}.png"
        p2 = img2_dir / f"{stem}.png"
        if not (t1p.exists() and t2p.exists() and mp.exists() and p1.exists() and p2.exists()):
            continue

        t1 = torch.load(t1p, map_location="cpu", weights_only=True)
        t2 = torch.load(t2p, map_location="cpu", weights_only=True)
        mt = torch.load(mp, map_location="cpu", weights_only=False)

        t1_img = np.array(Image.open(p1).convert("RGB"))
        t2_img = np.array(Image.open(p2).convert("RGB"))
        gt = get_gt_change(stem, label1_dir, label2_dir)
        if gt is None:
            continue

        token_probs = compute_token_probs(model, t1, t2, mt, device)
        centroids = t1["centroids"].numpy().astype(np.float32)
        areas = t1["areas"].numpy().astype(np.float32)

        mask_items = generate_masks(t1_img, amg)
        aligned_masks, aligned_scores, aligned_areas = align_masks_to_tokens(mask_items, centroids, areas)

        # Reconstructions.
        vor_map_idx = voronoi_partition(centroids)
        pred_vor = token_probs[vor_map_idx]

        pred_raw = raw_sam_projection(aligned_masks, token_probs)

        part_map, diag = sam_partition_map(
            aligned_masks,
            centroids,
            scores=aligned_scores,
            areas=aligned_areas,
            chooser=args.partition_chooser,
        )
        pred_part = token_probs[part_map]

        b_v = pred_vor > args.threshold
        b_r = pred_raw > args.threshold
        b_p = pred_part > args.threshold

        m_v = compute_metrics(b_v, gt)
        m_r = compute_metrics(b_r, gt)
        m_p = compute_metrics(b_p, gt)

        row = {
            "stem": stem,
            "vor_f1": m_v["f1"],
            "vor_iou": m_v["iou"],
            "vor_precision": m_v["precision"],
            "vor_recall": m_v["recall"],
            "vor_tp": m_v["tp"],
            "vor_fp": m_v["fp"],
            "vor_fn": m_v["fn"],
            "vor_tn": m_v["tn"],
            "raw_f1": m_r["f1"],
            "raw_iou": m_r["iou"],
            "raw_precision": m_r["precision"],
            "raw_recall": m_r["recall"],
            "raw_tp": m_r["tp"],
            "raw_fp": m_r["fp"],
            "raw_fn": m_r["fn"],
            "raw_tn": m_r["tn"],
            "part_f1": m_p["f1"],
            "part_iou": m_p["iou"],
            "part_precision": m_p["precision"],
            "part_recall": m_p["recall"],
            "part_tp": m_p["tp"],
            "part_fp": m_p["fp"],
            "part_fn": m_p["fn"],
            "part_tn": m_p["tn"],
            "delta_iou_part_vs_vor": m_p["iou"] - m_v["iou"],
            "delta_iou_part_vs_raw": m_p["iou"] - m_r["iou"],
            "partition_coverage": diag["partition_coverage"],
            "overlap_before": diag["overlap_before"],
            "overlap_after": diag["overlap_after"],
            "num_masks": diag["num_masks"],
            "avg_mask_area": diag["avg_mask_area"],
        }
        rows.append(row)

        for k in diag_acc.keys():
            diag_acc[k].append(diag[k])

        if len(vis_records) < args.vis_n:
            vis_records.append(
                {
                    "stem": stem,
                    "t1": t1_img,
                    "t2": t2_img,
                    "sam_overlay": draw_mask_boundaries(t1_img, aligned_masks),
                    "partition_rgb": partition_to_color(part_map),
                    "vor_rgb": partition_to_color(vor_map_idx),
                    "pred_vor": pred_vor,
                    "pred_raw": pred_raw,
                    "pred_part": pred_part,
                    "gt": gt,
                    "vor_iou": m_v["iou"],
                    "raw_iou": m_r["iou"],
                    "part_iou": m_p["iou"],
                    "vor_f1": m_v["f1"],
                    "raw_f1": m_r["f1"],
                    "part_f1": m_p["f1"],
                }
            )

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(stems)}] processed ...")

    if not rows:
        raise RuntimeError("No valid samples processed.")

    agg = {}
    agg.update(aggregate(rows, "vor"))
    agg.update(aggregate(rows, "raw"))
    agg.update(aggregate(rows, "part"))

    diag_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in diag_acc.items()}

    print("\n" + "=" * 74)
    print("STAGE 8C PARTITION COMPARISON")
    print("=" * 74)
    print(
        f"Voronoi      : F1={agg['micro_vor_f1']:.4f} IoU={agg['micro_vor_iou']:.4f} "
        f"P={agg['micro_vor_precision']:.4f} R={agg['micro_vor_recall']:.4f}"
    )
    print(
        f"Raw SAM      : F1={agg['micro_raw_f1']:.4f} IoU={agg['micro_raw_iou']:.4f} "
        f"P={agg['micro_raw_precision']:.4f} R={agg['micro_raw_recall']:.4f}"
    )
    print(
        f"SAM Partition: F1={agg['micro_part_f1']:.4f} IoU={agg['micro_part_iou']:.4f} "
        f"P={agg['micro_part_precision']:.4f} R={agg['micro_part_recall']:.4f}"
    )
    print(f"Delta IoU partition-voronoi: {agg['micro_part_iou'] - agg['micro_vor_iou']:.4f}")
    print(f"Delta IoU partition-rawSAM : {agg['micro_part_iou'] - agg['micro_raw_iou']:.4f}")
    print("=" * 74)

    csv_path = out_dir / "stage8c_per_image_metrics.csv"
    json_path = out_dir / "stage8c_partition_diagnostics.json"
    png_path = out_dir / "stage8_partition_comparison.png"
    md_path = out_dir / "stage8c_partition_reconstruction.md"

    save_csv(rows, csv_path)
    json_path.write_text(json.dumps({"diagnostics_mean": diag_mean, "aggregate": agg}, indent=2), encoding="utf-8")
    make_visual_grid(vis_records, png_path, args.vis_n)
    save_report(md_path, args, agg, diag_mean, n_proc=len(stems), n_gt=len(rows))

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 8C - SAM partition reconstruction")
    p.add_argument("--model_dir", default="SECOND/stage5_6_dynamic")
    p.add_argument("--tokens_T1", default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2", default="SECOND/tokens_T2")
    p.add_argument("--matches", default="SECOND/matches")
    p.add_argument("--images_T1", default="SECOND/im1")
    p.add_argument("--images_T2", default="SECOND/im2")
    p.add_argument("--label1_dir", default="SECOND/label1")
    p.add_argument("--label2_dir", default="SECOND/label2")
    p.add_argument("--output_dir", default="stage8/partition_compare")

    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--vis_n", type=int, default=6)

    p.add_argument("--partition_chooser", choices=["area", "score"], default="area")

    p.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--pred_iou_thresh", type=float, default=0.75)
    p.add_argument("--stability_thresh", type=float, default=0.85)
    p.add_argument("--min_mask_area", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
