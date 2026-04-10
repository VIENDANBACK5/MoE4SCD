#!/usr/bin/env python3
"""Stage 8: Voronoi vs SAM-mask reconstruction comparison.

This script evaluates pixel-level reconstruction quality for a token-based
change detector using two strategies on the SAME validation split:

1) Voronoi nearest-centroid projection
2) Direct SAM mask projection (max over overlaps)

Outputs:
  - stage8_reconstruction_comparison.png
  - stage8_sam_mask_reconstruction.md
  - stage8_per_image_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

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
EMB_SIZE = 64
SCALE = IMG_SIZE // EMB_SIZE

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
    mask: np.ndarray  # bool [H,W]
    centroid: np.ndarray  # [2] normalized (cx, cy)
    area: float  # fraction of image


def build_val_stems(match_dir: Path, t1_dir: Path, t2_dir: Path, val_split: float, seed: int) -> List[str]:
    all_stems = [
        mp.stem.replace("_matches", "")
        for mp in sorted(match_dir.glob("*_matches.pt"))
        if (t1_dir / f"{mp.stem.replace('_matches', '')}.pt").exists()
        and (t2_dir / f"{mp.stem.replace('_matches', '')}.pt").exists()
    ]
    n_val = max(1, int(len(all_stems) * val_split))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(all_stems), generator=gen).tolist()
    return [all_stems[i] for i in perm[len(all_stems) - n_val :]]


def load_model(model_dir: Path, device: torch.device) -> TokenChangeReasonerMoE:
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    cfg = MoEConfig(**{k: v for k, v in d.items() if k in MoEConfig.__dataclass_fields__})
    model = build_moe_model(cfg).to(device)
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_amg(args, device: torch.device) -> SAM2AutomaticMaskGenerator:
    ckpt_path = Path(args.sam2_ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = SAM2_REPO / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")

    model = build_sam2(args.sam2_config, str(ckpt_path), device=str(device))
    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        min_mask_region_area=args.min_mask_area,
        output_mode="binary_mask",
    )


def downsample_mask(mask: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    small = F.max_pool2d(t, kernel_size=SCALE, stride=SCALE)
    return small.squeeze() > 0.0


def masks_from_image(image_rgb: np.ndarray, amg: SAM2AutomaticMaskGenerator) -> List[MaskItem]:
    raw_masks = amg.generate(image_rgb)
    items: List[MaskItem] = []
    for m in raw_masks:
        seg = m["segmentation"]
        if seg.dtype != np.bool_:
            seg = seg.astype(bool)
        # Mirror tokenize_regions.py: skip masks that vanish after 64x64 pooling.
        if int(downsample_mask(seg).sum().item()) == 0:
            continue
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        cx = float(xs.mean()) / (IMG_SIZE - 1)
        cy = float(ys.mean()) / (IMG_SIZE - 1)
        area = float(seg.sum()) / (IMG_SIZE * IMG_SIZE)
        items.append(MaskItem(mask=seg, centroid=np.array([cx, cy], dtype=np.float32), area=area))

    if not items:
        full = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)
        items.append(MaskItem(mask=full, centroid=np.array([0.5, 0.5], dtype=np.float32), area=1.0))
    return items


def align_masks_to_tokens(mask_items: List[MaskItem], token_centroids: np.ndarray, token_areas: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """Align regenerated SAM masks to stored token ordering using Hungarian matching."""
    n_tok = token_centroids.shape[0]
    n_mask = len(mask_items)

    m_cent = np.stack([m.centroid for m in mask_items], axis=0)
    m_area = np.array([m.area for m in mask_items], dtype=np.float32)

    # Combined cost in normalized coordinate/area space.
    d_cent = np.sqrt(((token_centroids[:, None, :] - m_cent[None, :, :]) ** 2).sum(axis=2))
    d_area = np.abs(token_areas[:, None] - m_area[None, :])
    cost = 0.7 * d_cent + 0.3 * d_area

    r_idx, c_idx = linear_sum_assignment(cost)

    aligned = [None] * n_tok
    used_masks = set()
    cent_errs: List[float] = []
    area_errs: List[float] = []

    for r, c in zip(r_idx.tolist(), c_idx.tolist()):
        aligned[r] = mask_items[c].mask
        used_masks.add(c)
        cent_errs.append(float(d_cent[r, c]))
        area_errs.append(float(d_area[r, c]))

    # If token count differs from regenerated mask count, backfill with nearest centroid mask.
    if any(x is None for x in aligned):
        nearest = np.argmin(d_cent, axis=1)
        for i in range(n_tok):
            if aligned[i] is None:
                c = int(nearest[i])
                aligned[i] = mask_items[c].mask
                cent_errs.append(float(d_cent[i, c]))
                area_errs.append(float(d_area[i, c]))

    stats = {
        "n_tokens": float(n_tok),
        "n_regen_masks": float(n_mask),
        "mean_centroid_error": float(np.mean(cent_errs)) if cent_errs else 0.0,
        "mean_area_error": float(np.mean(area_errs)) if area_errs else 0.0,
        "count_gap": float(abs(n_tok - n_mask)),
        "reuse_rate": float(sum(1 for m in aligned if m is None) / max(n_tok, 1)),
    }

    return [m.astype(bool) for m in aligned], stats


def _rgb_to_class(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    dists = ((flat[:, None, :] - _SECOND_COLOURS[None, :, :]) ** 2).sum(-1)
    return dists.argmin(-1).reshape(rgb.shape[:2])


def get_gt_change(stem: str, label1_dir: Path, label2_dir: Path) -> Optional[np.ndarray]:
    p1 = label1_dir / f"{stem}.png"
    p2 = label2_dir / f"{stem}.png"
    if not (p1.exists() and p2.exists()):
        return None
    l1 = np.array(Image.open(p1).convert("RGB"))
    l2 = np.array(Image.open(p2).convert("RGB"))
    c1 = _rgb_to_class(l1)
    c2 = _rgb_to_class(l2)
    return c1 != c2


def voronoi_pixel_ids(centroids_t1: np.ndarray, H: int = IMG_SIZE, W: int = IMG_SIZE) -> np.ndarray:
    pts = centroids_t1[:, ::-1].copy()
    pts[:, 0] *= (H - 1)
    pts[:, 1] *= (W - 1)
    ys, xs = np.mgrid[0:H, 0:W]
    grid = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    tree = cKDTree(pts)
    _, token_ids = tree.query(grid, k=1)
    return token_ids.astype(np.int32)


def reconstruct_voronoi(centroids_t1: np.ndarray, t1_probs: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    ids = voronoi_pixel_ids(centroids_t1)
    prob = t1_probs[ids].reshape(IMG_SIZE, IMG_SIZE).astype(np.float32)
    pred = prob > threshold
    return prob, pred


def reconstruct_sam_masks(token_masks: Sequence[np.ndarray], t1_probs: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    prob = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, m in enumerate(token_masks):
        p = float(t1_probs[i])
        prob[m] = np.maximum(prob[m], p)
    pred = prob > threshold
    return prob, pred


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def aggregate(per_image: List[Dict[str, float]], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ["precision", "recall", "f1", "iou"]:
        vals = [m[f"{prefix}_{k}"] for m in per_image]
        out[f"macro_{prefix}_{k}"] = float(np.mean(vals)) if vals else 0.0
        out[f"std_{prefix}_{k}"] = float(np.std(vals)) if vals else 0.0

    tp = sum(int(m[f"{prefix}_tp"]) for m in per_image)
    fp = sum(int(m[f"{prefix}_fp"]) for m in per_image)
    fn = sum(int(m[f"{prefix}_fn"]) for m in per_image)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    out[f"micro_{prefix}_precision"] = float(precision)
    out[f"micro_{prefix}_recall"] = float(recall)
    out[f"micro_{prefix}_f1"] = float(f1)
    out[f"micro_{prefix}_iou"] = float(iou)
    return out


def _font(sz: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def _heatmap(arr: np.ndarray) -> np.ndarray:
    a = np.clip(arr, 0, 1)
    r = np.clip(1.5 - np.abs(4 * a - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4 * a - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4 * a - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _bool_to_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[arr] = [255, 255, 255]
    return out


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))


def draw_5panel(t1: np.ndarray, t2: np.ndarray, vor_prob: np.ndarray, sam_prob: np.ndarray, gt: np.ndarray, stem: str, metrics: Dict[str, float]) -> Image.Image:
    H = IMG_SIZE
    W = IMG_SIZE
    pad = 6
    title_h = 22
    cell_w = W + pad
    cell_h = H + title_h + pad

    canvas = Image.new("RGB", (5 * cell_w + pad, cell_h + 40), (245, 245, 248))
    draw = ImageDraw.Draw(canvas)

    panels = [
        (t1, "T1"),
        (t2, "T2"),
        (_heatmap(vor_prob), "Voronoi"),
        (_heatmap(sam_prob), "SAM-mask"),
        (_bool_to_rgb(gt), "GT"),
    ]

    for i, (arr, title) in enumerate(panels):
        x0 = pad + i * cell_w
        y0 = pad + title_h
        canvas.paste(Image.fromarray(arr), (x0, y0))
        draw.text((x0 + W // 2, pad + title_h // 2), title, fill=(30, 30, 30), font=_font(13), anchor="mm")

    cap = (
        f"{stem} | Vor(F1={metrics['voronoi_f1']:.3f},IoU={metrics['voronoi_iou']:.3f}) "
        f"SAM(F1={metrics['sam_f1']:.3f},IoU={metrics['sam_iou']:.3f})"
    )
    draw.text((canvas.width // 2, canvas.height - 12), cap, fill=(60, 60, 60), font=_font(10), anchor="mm")
    return canvas


def save_vis_grid(records: List[Dict], save_path: Path, n_show: int) -> None:
    picks = records[:n_show]
    if not picks:
        return
    panels = [draw_5panel(r["t1_img"], r["t2_img"], r["vor_prob"], r["sam_prob"], r["gt"], r["stem"], r["metrics"]) for r in picks]

    pw, ph = panels[0].size
    cols = 1 if len(panels) <= 3 else 2
    rows = math.ceil(len(panels) / cols)
    canvas = Image.new("RGB", (cols * pw + 10, rows * ph + 52), (240, 240, 244))
    draw = ImageDraw.Draw(canvas)
    draw.text((canvas.width // 2, 18), "Stage 8: Voronoi vs SAM-mask Reconstruction", fill=(30, 30, 30), font=_font(16), anchor="mm")

    for i, p in enumerate(panels):
        r = i // cols
        c = i % cols
        canvas.paste(p, (5 + c * pw, 40 + r * ph))

    canvas.save(save_path)


def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_report(path: Path, args, n_proc: int, n_gt: int, agg: Dict[str, float]) -> None:
    mi_v_f1 = agg["micro_voronoi_f1"]
    mi_v_iou = agg["micro_voronoi_iou"]
    mi_v_p = agg["micro_voronoi_precision"]
    mi_v_r = agg["micro_voronoi_recall"]

    mi_s_f1 = agg["micro_sam_f1"]
    mi_s_iou = agg["micro_sam_iou"]
    mi_s_p = agg["micro_sam_precision"]
    mi_s_r = agg["micro_sam_recall"]

    delta_iou = mi_s_iou - mi_v_iou

    lines = [
        "# Stage 8 — SAM Mask Reconstruction Comparison",
        "",
        "## Overview",
        "",
        f"- Model: `{args.model_dir}`",
        "- Dataset: SECOND validation split",
        f"- Validation images processed: `{n_proc}`",
        f"- Images with GT labels: `{n_gt}`",
        "- Reconstruction methods: Voronoi nearest-centroid vs direct SAM mask projection",
        f"- Threshold: `{args.threshold}`",
        "",
        "---",
        "",
        "## Metrics Comparison",
        "",
        "### Micro-averaged",
        "",
        "| Reconstruction | F1 | IoU | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
        f"| Voronoi | {mi_v_f1:.4f} | {mi_v_iou:.4f} | {mi_v_p:.4f} | {mi_v_r:.4f} |",
        f"| SAM mask | {mi_s_f1:.4f} | {mi_s_iou:.4f} | {mi_s_p:.4f} | {mi_s_r:.4f} |",
        "",
        "### Macro-averaged (mean ± std)",
        "",
        "| Reconstruction | F1 | IoU | Precision | Recall |",
        "|---|---|---|---|---|",
        (
            f"| Voronoi | {agg['macro_voronoi_f1']:.4f} ± {agg['std_voronoi_f1']:.4f} "
            f"| {agg['macro_voronoi_iou']:.4f} ± {agg['std_voronoi_iou']:.4f} "
            f"| {agg['macro_voronoi_precision']:.4f} ± {agg['std_voronoi_precision']:.4f} "
            f"| {agg['macro_voronoi_recall']:.4f} ± {agg['std_voronoi_recall']:.4f} |"
        ),
        (
            f"| SAM mask | {agg['macro_sam_f1']:.4f} ± {agg['std_sam_f1']:.4f} "
            f"| {agg['macro_sam_iou']:.4f} ± {agg['std_sam_iou']:.4f} "
            f"| {agg['macro_sam_precision']:.4f} ± {agg['std_sam_precision']:.4f} "
            f"| {agg['macro_sam_recall']:.4f} ± {agg['std_sam_recall']:.4f} |"
        ),
        "",
        f"**IoU improvement (micro):** $\\Delta IoU = IoU_{{mask}} - IoU_{{voronoi}} = {delta_iou:.4f}$",
        "",
        "---",
        "",
        "## Visual Comparison",
        "",
        "See `stage8_reconstruction_comparison.png` (10 samples, panel order: T1, T2, Voronoi, SAM-mask, GT).",
        "",
        "---",
        "",
        "## Analysis",
        "",
    ]

    if delta_iou > 0.01:
        lines.append(
            "SAM-mask projection provides a clear spatial accuracy gain, indicating Voronoi assignment introduces meaningful boundary/shape errors in this setup."
        )
    elif delta_iou < -0.01:
        lines.append(
            "SAM-mask projection underperforms Voronoi here, suggesting mismatch between regenerated masks and stored token ordering or unstable SAM mask regeneration."
        )
    else:
        lines.append(
            "The two methods are very close, indicating Voronoi approximation is already a good proxy for spatial projection on SECOND at this token granularity."
        )

    lines += [
        "",
        "Key interpretation: any gain from SAM-mask projection isolates spatial reconstruction error, while remaining error reflects token-level prediction limits (region heterogeneity and class ambiguity).",
        "",
        "---",
        "",
        "_Generated by `run_stage8_sam_mask_reconstruction.py`_",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_dir = Path(args.model_dir)
    t1_dir = Path(args.tokens_T1)
    t2_dir = Path(args.tokens_T2)
    match_dir = Path(args.matches)
    label1_dir = Path(args.label1_dir)
    label2_dir = Path(args.label2_dir)
    img1_dir = Path(args.images_T1)
    img2_dir = Path(args.images_T2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir} ...")
    model = load_model(model_dir, device)
    print(f"  Router v{model.cfg.router_version} | Experts: {model.cfg.moe_num_experts}")

    print("Loading SAM2 mask generator ...")
    amg = load_amg(args, device)

    stems = build_val_stems(match_dir, t1_dir, t2_dir, args.val_split, args.seed)
    if args.max_samples is not None:
        stems = stems[: args.max_samples]
    print(f"Validation stems: {len(stems)}")

    per_image_rows: List[Dict] = []
    vis_records: List[Dict] = []

    for idx, stem in enumerate(stems):
        t1p = t1_dir / f"{stem}.pt"
        t2p = t2_dir / f"{stem}.pt"
        mp = match_dir / f"{stem}_matches.pt"
        p1 = img1_dir / f"{stem}.png"
        p2 = img2_dir / f"{stem}.png"
        if not (t1p.exists() and t2p.exists() and mp.exists() and p1.exists() and p2.exists()):
            continue

        t1 = torch.load(t1p, map_location="cpu", weights_only=True)
        t2 = torch.load(t2p, map_location="cpu", weights_only=True)
        mtch = torch.load(mp, map_location="cpu", weights_only=False)

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
            outputs = model(batch)

        change_logits = outputs["change_logits"][0].cpu()
        pad_mask = batch["padding_mask"][0].cpu()
        time_ids = batch["time_ids_pad"][0].cpu()

        t1_mask = (~pad_mask) & (time_ids == 0)
        t1_probs = torch.sigmoid(change_logits[t1_mask]).numpy()

        # Use stored token ordering for reconstruction.
        t1_centroids = t1["centroids"].numpy().astype(np.float32)
        t1_areas = t1["areas"].numpy().astype(np.float32)

        vor_prob, vor_bin = reconstruct_voronoi(t1_centroids, t1_probs, args.threshold)

        img_t1 = np.array(Image.open(p1).convert("RGB"))
        regen_masks = masks_from_image(img_t1, amg)
        aligned_masks, align_stats = align_masks_to_tokens(regen_masks, t1_centroids, t1_areas)
        sam_prob, sam_bin = reconstruct_sam_masks(aligned_masks, t1_probs, args.threshold)

        gt = get_gt_change(stem, label1_dir, label2_dir)
        if gt is None:
            continue

        m_v = compute_metrics(vor_bin, gt)
        m_s = compute_metrics(sam_bin, gt)

        row = {
            "stem": stem,
            "voronoi_f1": m_v["f1"],
            "voronoi_iou": m_v["iou"],
            "voronoi_precision": m_v["precision"],
            "voronoi_recall": m_v["recall"],
            "voronoi_tp": m_v["tp"],
            "voronoi_fp": m_v["fp"],
            "voronoi_fn": m_v["fn"],
            "voronoi_tn": m_v["tn"],
            "sam_f1": m_s["f1"],
            "sam_iou": m_s["iou"],
            "sam_precision": m_s["precision"],
            "sam_recall": m_s["recall"],
            "sam_tp": m_s["tp"],
            "sam_fp": m_s["fp"],
            "sam_fn": m_s["fn"],
            "sam_tn": m_s["tn"],
            "delta_iou": m_s["iou"] - m_v["iou"],
            "n_tokens": int(t1_centroids.shape[0]),
            "n_regen_masks": int(len(regen_masks)),
            "align_mean_centroid_error": align_stats["mean_centroid_error"],
            "align_mean_area_error": align_stats["mean_area_error"],
            "align_count_gap": align_stats["count_gap"],
        }
        per_image_rows.append(row)

        if len(vis_records) < max(args.vis_n, 10):
            vis_records.append(
                {
                    "stem": stem,
                    "t1_img": img_t1,
                    "t2_img": np.array(Image.open(p2).convert("RGB")),
                    "vor_prob": vor_prob,
                    "sam_prob": sam_prob,
                    "gt": gt,
                    "metrics": row,
                }
            )

        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(stems)}] processed ...")

    if not per_image_rows:
        raise RuntimeError("No evaluated images with GT labels were processed.")

    agg_v = aggregate(per_image_rows, "voronoi")
    agg_s = aggregate(per_image_rows, "sam")
    agg = {**agg_v, **agg_s}

    print("\n" + "=" * 72)
    print("STAGE 8 — RECONSTRUCTION COMPARISON")
    print("=" * 72)
    print(
        f"Voronoi  : F1={agg['micro_voronoi_f1']:.4f} IoU={agg['micro_voronoi_iou']:.4f} "
        f"P={agg['micro_voronoi_precision']:.4f} R={agg['micro_voronoi_recall']:.4f}"
    )
    print(
        f"SAM mask : F1={agg['micro_sam_f1']:.4f} IoU={agg['micro_sam_iou']:.4f} "
        f"P={agg['micro_sam_precision']:.4f} R={agg['micro_sam_recall']:.4f}"
    )
    print(f"Delta IoU (SAM - Voronoi): {agg['micro_sam_iou'] - agg['micro_voronoi_iou']:.4f}")
    print("=" * 72)

    csv_path = out_dir / "stage8_per_image_metrics.csv"
    vis_path = out_dir / "stage8_reconstruction_comparison.png"
    md_path = out_dir / "stage8_sam_mask_reconstruction.md"

    save_csv(per_image_rows, csv_path)
    save_vis_grid(vis_records, vis_path, args.vis_n)
    save_report(md_path, args, n_proc=len(stems), n_gt=len(per_image_rows), agg=agg)

    print(f"Saved: {csv_path}")
    print(f"Saved: {vis_path}")
    print(f"Saved: {md_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 8: Voronoi vs SAM-mask reconstruction comparison")
    p.add_argument("--model_dir", default="SECOND/stage5_6_dynamic")
    p.add_argument("--tokens_T1", default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2", default="SECOND/tokens_T2")
    p.add_argument("--matches", default="SECOND/matches")
    p.add_argument("--label1_dir", default="SECOND/label1")
    p.add_argument("--label2_dir", default="SECOND/label2")
    p.add_argument("--images_T1", default="SECOND/im1")
    p.add_argument("--images_T2", default="SECOND/im2")
    p.add_argument("--output_dir", default="stage8/reconstruction")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--vis_n", type=int, default=10)
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--pred_iou_thresh", type=float, default=0.75)
    p.add_argument("--stability_thresh", type=float, default=0.85)
    p.add_argument("--min_mask_area", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
