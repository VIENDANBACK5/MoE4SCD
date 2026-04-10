#!/usr/bin/env python3
"""Stage 8B: Token-mask consistency debugging for token-based change detection.

Diagnostics implemented:
1) Token-mask centroid alignment check
2) Token-mask ordering consistency via feature re-pooling similarity
3) Mask stability check (reference vs evaluation mask regeneration)
4) Spatial coverage + overlap check
5) Visualization debug panels
6) Structured report generation
"""

from __future__ import annotations

import argparse
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
import seaborn as sns
import torch
import torch.nn.functional as F
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
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

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
    centroid: np.ndarray  # [2], normalized (cx, cy)
    area_ratio: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return _rgb_to_class(l1) != _rgb_to_class(l2)


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


def load_moe_model(model_dir: Path, device: torch.device) -> TokenChangeReasonerMoE:
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = MoEConfig(**{k: v for k, v in cfg_dict.items() if k in MoEConfig.__dataclass_fields__})
    model = build_moe_model(cfg).to(device)
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_amg(
    sam2_config: str,
    sam2_ckpt: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_thresh: float,
    min_mask_area: int,
    device: torch.device,
) -> SAM2AutomaticMaskGenerator:
    ckpt_path = Path(sam2_ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = SAM2_REPO / sam2_ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")

    sam2_model = build_sam2(sam2_config, str(ckpt_path), device=str(device))
    amg = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_thresh,
        min_mask_region_area=min_mask_area,
        output_mode="binary_mask",
    )
    return amg


def downsample_mask(mask: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(t, kernel_size=SCALE, stride=SCALE)
    return pooled.squeeze() > 0.0


def generate_masks(image_rgb: np.ndarray, amg: SAM2AutomaticMaskGenerator) -> List[MaskItem]:
    raw = amg.generate(image_rgb)
    items: List[MaskItem] = []
    for m in raw:
        seg = m["segmentation"]
        if seg.dtype != np.bool_:
            seg = seg.astype(bool)
        # Mirror tokenization behavior.
        if int(downsample_mask(seg).sum().item()) == 0:
            continue
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        cx = float(xs.mean()) / (IMG_SIZE - 1)
        cy = float(ys.mean()) / (IMG_SIZE - 1)
        area_ratio = float(seg.sum()) / (IMG_SIZE * IMG_SIZE)
        items.append(MaskItem(mask=seg, centroid=np.array([cx, cy], dtype=np.float32), area_ratio=area_ratio))

    if not items:
        full = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)
        items.append(MaskItem(mask=full, centroid=np.array([0.5, 0.5], dtype=np.float32), area_ratio=1.0))
    return items


def pool_feature_from_mask(mask: np.ndarray, embedding: torch.Tensor) -> torch.Tensor:
    """Re-pool feature from mask region at embedding resolution.

    Args:
        mask: bool [512,512]
        embedding: [256,64,64] or [1,256,64,64]
    Returns:
        [256] float32
    """
    emb = embedding.squeeze(0)
    mask_small = downsample_mask(mask)
    if int(mask_small.sum().item()) == 0:
        return emb.mean(dim=(1, 2))
    region = emb[:, mask_small]
    return region.mean(dim=1)


def cosine_similarity_matrix(tokens: torch.Tensor, pooled_masks: torch.Tensor) -> torch.Tensor:
    """Compute cosine sim matrix [num_tokens, num_masks]."""
    t = F.normalize(tokens.float(), dim=1)
    m = F.normalize(pooled_masks.float(), dim=1)
    return t @ m.T


def centroid_inside_mask_stats(centroids: np.ndarray, masks: Sequence[np.ndarray]) -> Tuple[int, int, List[int]]:
    total = min(len(centroids), len(masks))
    inside = 0
    mismatched: List[int] = []
    for i in range(total):
        cx = int(round(float(centroids[i, 0]) * (IMG_SIZE - 1)))
        cy = int(round(float(centroids[i, 1]) * (IMG_SIZE - 1)))
        cx = max(0, min(IMG_SIZE - 1, cx))
        cy = max(0, min(IMG_SIZE - 1, cy))
        if bool(masks[i][cy, cx]):
            inside += 1
        else:
            mismatched.append(i)
    return inside, total - inside, mismatched


def align_by_iou(ref_masks: Sequence[np.ndarray], eval_masks: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian alignment maximizing IoU between mask sets."""
    n_ref = len(ref_masks)
    n_eval = len(eval_masks)
    n = max(n_ref, n_eval)
    iou_mat = np.zeros((n_ref, n_eval), dtype=np.float32)

    for i in range(n_ref):
        rm = ref_masks[i]
        for j in range(n_eval):
            em = eval_masks[j]
            inter = int((rm & em).sum())
            union = int((rm | em).sum())
            iou_mat[i, j] = inter / max(union, 1)

    cost = 1.0 - iou_mat
    r, c = linear_sum_assignment(cost)
    # Keep only valid matched pairs.
    valid = (r < n_ref) & (c < n_eval)
    r = r[valid]
    c = c[valid]
    vals = iou_mat[r, c]
    return r, c, vals


def coverage_and_overlap(masks: Sequence[np.ndarray]) -> Dict[str, float]:
    if not masks:
        return {
            "coverage_ratio": 0.0,
            "sum_mask_ratio": 0.0,
            "overlap_pixel_ratio": 0.0,
            "overlap_excess_ratio": 0.0,
        }

    stack = np.stack([m.astype(np.uint8) for m in masks], axis=0)
    count = stack.sum(axis=0).astype(np.int32)
    image_pixels = float(IMG_SIZE * IMG_SIZE)

    covered = float((count > 0).sum())
    sum_mask_pixels = float(count.sum())
    overlap_pixels = float((count > 1).sum())
    overlap_excess = float(np.clip(count - 1, 0, None).sum())

    return {
        "coverage_ratio": covered / image_pixels,
        "sum_mask_ratio": sum_mask_pixels / image_pixels,
        "overlap_pixel_ratio": overlap_pixels / image_pixels,
        "overlap_excess_ratio": overlap_excess / image_pixels,
    }


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
    time_ids = batch["time_ids_pad"][0].cpu()
    t1_mask = (~pad) & (time_ids == 0)
    probs = torch.sigmoid(logits[t1_mask]).numpy()
    return probs


def voronoi_ids(centroids: np.ndarray) -> np.ndarray:
    pts = centroids[:, ::-1].copy()
    pts[:, 0] *= (IMG_SIZE - 1)
    pts[:, 1] *= (IMG_SIZE - 1)
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    grid = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    tree = cKDTree(pts)
    _, idx = tree.query(grid, k=1)
    return idx.astype(np.int32).reshape(IMG_SIZE, IMG_SIZE)


def mask_boundaries(mask: np.ndarray) -> np.ndarray:
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.ascontiguousarray(np.zeros(m.shape, dtype=np.uint8))
    cv2.drawContours(out, contours, -1, 1, 1)
    return out.astype(bool)


def draw_debug_panel(
    stem: str,
    t1_img: np.ndarray,
    t2_img: np.ndarray,
    masks: Sequence[np.ndarray],
    centroids: np.ndarray,
    probs: np.ndarray,
    gt_change: Optional[np.ndarray],
    save_path: Path,
) -> None:
    n_show_masks = min(25, len(masks))
    pick = np.random.choice(len(masks), size=n_show_masks, replace=False) if len(masks) > n_show_masks else np.arange(len(masks))

    # Build SAM boundary overlay.
    sam_overlay = t1_img.copy()
    for idx in pick.tolist():
        b = mask_boundaries(masks[idx])
        color = np.array([0, 255, 255], dtype=np.uint8)
        sam_overlay[b] = color

    # Centroid overlay.
    cent_overlay = sam_overlay.copy()
    for c in centroids:
        x = int(round(float(c[0]) * (IMG_SIZE - 1)))
        y = int(round(float(c[1]) * (IMG_SIZE - 1)))
        cv2.circle(cent_overlay, (x, y), 2, (255, 0, 0), -1)

    # Voronoi region map.
    vid = voronoi_ids(centroids)
    rng = np.random.default_rng(123)
    palette = rng.integers(0, 255, size=(max(vid.max() + 1, 1), 3), dtype=np.uint8)
    vor_rgb = palette[vid]

    # Prediction map from Voronoi projection.
    pred = probs[vid]

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    ax = axes.ravel()

    ax[0].imshow(t1_img)
    ax[0].set_title("T1")
    ax[1].imshow(t2_img)
    ax[1].set_title("T2")
    ax[2].imshow(sam_overlay)
    ax[2].set_title("SAM masks (boundaries)")
    ax[3].imshow(cent_overlay)
    ax[3].set_title("Token centroids")
    ax[4].imshow(vor_rgb)
    ax[4].set_title("Voronoi regions")

    if gt_change is None:
        ax[5].imshow(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8), cmap="gray")
        ax[5].set_title("GT change (missing)")
    else:
        ax[5].imshow(gt_change.astype(np.uint8), cmap="gray")
        ax[5].set_title("GT change")

    im = ax[6].imshow(pred, cmap="jet", vmin=0, vmax=1)
    ax[6].set_title("Prediction map")
    fig.colorbar(im, ax=ax[6], fraction=0.046, pad=0.04)

    # Highest-probability tokens overlay.
    rank = np.argsort(-probs)[:10]
    hp = t1_img.copy()
    for r in rank.tolist():
        b = mask_boundaries(masks[r])
        hp[b] = [255, 255, 0]
        x = int(round(float(centroids[r, 0]) * (IMG_SIZE - 1)))
        y = int(round(float(centroids[r, 1]) * (IMG_SIZE - 1)))
        cv2.circle(hp, (x, y), 3, (255, 0, 0), -1)
        cv2.putText(hp, f"{r}:{probs[r]:.2f}", (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    ax[7].imshow(hp)
    ax[7].set_title("Top probability tokens")

    for a in ax:
        a.axis("off")

    fig.suptitle(f"Stage8B Debug Panel - {stem}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_similarity_heatmap(sim: np.ndarray, stem: str, save_path: Path) -> None:
    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(sim, cmap="viridis", cbar=True)
    plt.title(f"Token-Mask Cosine Similarity ({stem})")
    plt.xlabel("Mask index")
    plt.ylabel("Token index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def write_report_md(summary: Dict, output_path: Path) -> None:
    lines = [
        "# Stage 8B Token-Mask Debug Report",
        "",
        "## Structured Output",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
    ]

    errs = summary.get("detected_mapping_errors", [])
    if errs:
        lines.append("Detected potential mapping errors:")
        for e in errs:
            lines.append(f"- {e}")
    else:
        lines.append("No major mapping error signatures detected under current diagnostics.")

    lines += [
        "",
        "## Diagnostic Attribution",
        "",
        "- A: token-mask index mismatch",
        "- B: unstable SAM mask regeneration",
        "- C: incorrect centroid calculation",
        "- D: reconstruction projection bug",
        "",
        "The most likely causes are inferred in detected_mapping_errors from measured thresholds.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    t1_dir = Path(args.tokens_T1)
    t2_dir = Path(args.tokens_T2)
    emb_t1_dir = Path(args.embeddings_T1)
    match_dir = Path(args.matches)
    img1_dir = Path(args.images_T1)
    img2_dir = Path(args.images_T2)
    label1_dir = Path(args.label1_dir)
    label2_dir = Path(args.label2_dir)

    print(f"Loading model from {model_dir} ...")
    model = load_moe_model(model_dir, device)
    print(f"  Router v{model.cfg.router_version} | Experts: {model.cfg.moe_num_experts}")

    print("Loading SAM generators (reference + evaluation) ...")
    amg_ref = load_amg(
        sam2_config=args.sam2_config,
        sam2_ckpt=args.sam2_ckpt,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_thresh=args.stability_thresh,
        min_mask_area=args.min_mask_area,
        device=device,
    )
    amg_eval = load_amg(
        sam2_config=args.sam2_config,
        sam2_ckpt=args.sam2_ckpt,
        points_per_side=args.eval_points_per_side,
        pred_iou_thresh=args.eval_pred_iou_thresh,
        stability_thresh=args.eval_stability_thresh,
        min_mask_area=args.eval_min_mask_area,
        device=device,
    )

    stems = build_val_stems(match_dir, t1_dir, t2_dir, args.val_split, args.seed)
    if args.max_samples is not None:
        stems = stems[: args.max_samples]
    print(f"Validation stems: {len(stems)}")

    total_tokens = 0
    centroid_inside_total = 0
    centroid_outside_total = 0
    mismatched_indices_examples: Dict[str, List[int]] = {}

    count_mismatch_images = 0
    per_image_count_gap: List[int] = []

    mask_areas: List[int] = []
    zero_area_masks = 0
    huge_masks = 0

    all_best_not_same = 0
    all_best_total = 0
    sim_diag_vals: List[float] = []
    sim_best_vals: List[float] = []

    stability_iou_vals: List[float] = []

    coverage_ratios: List[float] = []
    sum_mask_ratios: List[float] = []
    overlap_pixel_ratios: List[float] = []

    debug_panel_stems = random.sample(stems, k=min(args.vis_n, len(stems))) if stems else []

    for idx, stem in enumerate(stems):
        t1_path = t1_dir / f"{stem}.pt"
        t2_path = t2_dir / f"{stem}.pt"
        emb_path = emb_t1_dir / f"{stem}.pt"
        match_path = match_dir / f"{stem}_matches.pt"
        img1_path = img1_dir / f"{stem}.png"
        img2_path = img2_dir / f"{stem}.png"

        if not (t1_path.exists() and t2_path.exists() and emb_path.exists() and match_path.exists() and img1_path.exists() and img2_path.exists()):
            continue

        t1 = torch.load(t1_path, map_location="cpu", weights_only=True)
        t2 = torch.load(t2_path, map_location="cpu", weights_only=True)
        emb = torch.load(emb_path, map_location="cpu", weights_only=True)
        mtch = torch.load(match_path, map_location="cpu", weights_only=False)

        tokens = t1["tokens"].float()
        centroids = t1["centroids"].numpy().astype(np.float32)

        t1_img = np.array(Image.open(img1_path).convert("RGB"))
        t2_img = np.array(Image.open(img2_path).convert("RGB"))

        ref_masks_items = generate_masks(t1_img, amg_ref)
        eval_masks_items = generate_masks(t1_img, amg_eval)

        ref_masks = [m.mask for m in ref_masks_items]
        eval_masks = [m.mask for m in eval_masks_items]

        # 1) Token-mask centroid alignment by index.
        inside, outside, mismatch_idx = centroid_inside_mask_stats(centroids, ref_masks)
        centroid_inside_total += inside
        centroid_outside_total += outside
        total_tokens += min(len(centroids), len(ref_masks))
        if mismatch_idx and len(mismatched_indices_examples) < 20:
            mismatched_indices_examples[stem] = mismatch_idx[:50]

        # 2) Count consistency.
        gap = abs(len(tokens) - len(ref_masks))
        per_image_count_gap.append(gap)
        if gap != 0:
            count_mismatch_images += 1

        # 3) Mask areas.
        for m in ref_masks:
            area = int(m.sum())
            mask_areas.append(area)
            if area == 0:
                zero_area_masks += 1
            if area > (IMG_SIZE * IMG_SIZE // 2):
                huge_masks += 1

        # 2) Ordering consistency via similarity matrix.
        pooled = torch.stack([pool_feature_from_mask(m, emb).cpu() for m in ref_masks], dim=0)
        sim = cosine_similarity_matrix(tokens.cpu(), pooled).numpy()

        n_tok, n_mask = sim.shape
        min_n = min(n_tok, n_mask)
        diag = sim[np.arange(min_n), np.arange(min_n)]
        sim_diag_vals.extend(diag.tolist())
        best_idx = np.argmax(sim, axis=1)
        best_vals = np.max(sim, axis=1)
        sim_best_vals.extend(best_vals.tolist())
        all_best_not_same += int((best_idx[:min_n] != np.arange(min_n)).sum())
        all_best_total += int(min_n)

        if idx < args.heatmap_n:
            save_similarity_heatmap(sim, stem, out_dir / f"stage8b_similarity_heatmap_{stem}.png")

        # 3) Mask stability (ref vs eval).
        _, _, iou_vals = align_by_iou(ref_masks, eval_masks)
        stability_iou_vals.extend(iou_vals.tolist())

        # 4) Spatial coverage.
        cov = coverage_and_overlap(ref_masks)
        coverage_ratios.append(cov["coverage_ratio"])
        sum_mask_ratios.append(cov["sum_mask_ratio"])
        overlap_pixel_ratios.append(cov["overlap_pixel_ratio"])

        # 5) Visualization panel.
        if stem in debug_panel_stems:
            probs = compute_token_probs(model, t1, t2, mtch, device)
            gt = get_gt_change(stem, label1_dir, label2_dir)
            draw_debug_panel(
                stem=stem,
                t1_img=t1_img,
                t2_img=t2_img,
                masks=ref_masks,
                centroids=centroids,
                probs=probs,
                gt_change=gt,
                save_path=out_dir / f"stage8b_debug_panel_{stem}.png",
            )

        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(stems)}] processed ...")

    centroid_alignment_ratio = float(centroid_inside_total / max(total_tokens, 1))
    best_not_same_ratio = float(all_best_not_same / max(all_best_total, 1))

    sim_diag_mean = float(np.mean(sim_diag_vals)) if sim_diag_vals else 0.0
    sim_best_mean = float(np.mean(sim_best_vals)) if sim_best_vals else 0.0
    token_mask_similarity_mean = sim_diag_mean

    mask_iou_mean = float(np.mean(stability_iou_vals)) if stability_iou_vals else 0.0
    mask_iou_std = float(np.std(stability_iou_vals)) if stability_iou_vals else 0.0
    mask_iou_lt_08_ratio = float(np.mean(np.array(stability_iou_vals) < 0.8)) if stability_iou_vals else 0.0

    coverage_ratio = float(np.mean(coverage_ratios)) if coverage_ratios else 0.0
    sum_mask_ratio = float(np.mean(sum_mask_ratios)) if sum_mask_ratios else 0.0
    overlap_ratio = float(np.mean(overlap_pixel_ratios)) if overlap_pixel_ratios else 0.0

    detected_mapping_errors: List[str] = []
    if centroid_alignment_ratio < 0.95:
        detected_mapping_errors.append("C: centroid-in-mask ratio below 0.95 suggests centroid/mask inconsistency")
    if count_mismatch_images > 0:
        detected_mapping_errors.append("A: token-mask count mismatch detected")
    if best_not_same_ratio > 0.1:
        detected_mapping_errors.append("A: feature similarity best-match index differs from identity for many tokens")
    if mask_iou_mean < 0.8 or mask_iou_lt_08_ratio > 0.2:
        detected_mapping_errors.append("B: SAM mask regeneration appears unstable (low reference/eval IoU)")
    if coverage_ratio < 0.95:
        detected_mapping_errors.append("D: masks do not cover image adequately; projection may be biased")

    summary = {
        "centroid_alignment_ratio": centroid_alignment_ratio,
        "token_mask_similarity_mean": token_mask_similarity_mean,
        "token_mask_similarity_best_mean": sim_best_mean,
        "token_mask_best_not_identity_ratio": best_not_same_ratio,
        "mask_iou_mean": mask_iou_mean,
        "mask_iou_std": mask_iou_std,
        "mask_iou_lt_0_8_ratio": mask_iou_lt_08_ratio,
        "coverage_ratio": coverage_ratio,
        "sum_mask_ratio": sum_mask_ratio,
        "overlap_pixel_ratio": overlap_ratio,
        "count_mismatch_images": int(count_mismatch_images),
        "mean_count_gap": float(np.mean(per_image_count_gap)) if per_image_count_gap else 0.0,
        "max_count_gap": int(np.max(per_image_count_gap)) if per_image_count_gap else 0,
        "mask_area_mean": float(np.mean(mask_areas)) if mask_areas else 0.0,
        "mask_area_min": int(np.min(mask_areas)) if mask_areas else 0,
        "mask_area_max": int(np.max(mask_areas)) if mask_areas else 0,
        "zero_area_masks": int(zero_area_masks),
        "huge_masks_over_50pct": int(huge_masks),
        "total_tokens": int(total_tokens),
        "centroid_inside_mask_ratio": centroid_alignment_ratio,
        "mismatched_indices_examples": mismatched_indices_examples,
        "detected_mapping_errors": detected_mapping_errors,
    }

    json_path = out_dir / "stage8b_report.json"
    md_path = out_dir / "stage8b_mask_consistency_debug.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report_md(summary, md_path)

    print("\n" + "=" * 78)
    print("STAGE 8B DEBUG SUMMARY")
    print("=" * 78)
    print(f"centroid_alignment_ratio     : {summary['centroid_alignment_ratio']:.6f}")
    print(f"token_mask_similarity_mean   : {summary['token_mask_similarity_mean']:.6f}")
    print(f"mask_iou_mean                : {summary['mask_iou_mean']:.6f}")
    print(f"coverage_ratio               : {summary['coverage_ratio']:.6f}")
    print(f"detected_mapping_errors      : {len(summary['detected_mapping_errors'])}")
    for e in summary["detected_mapping_errors"]:
        print(f"  - {e}")
    print("=" * 78)
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 8B token-mask consistency debugging")

    p.add_argument("--model_dir", default="SECOND/stage5_6_dynamic")
    p.add_argument("--tokens_T1", default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2", default="SECOND/tokens_T2")
    p.add_argument("--embeddings_T1", default="SECOND/embeddings_T1")
    p.add_argument("--matches", default="SECOND/matches")
    p.add_argument("--images_T1", default="SECOND/im1")
    p.add_argument("--images_T2", default="SECOND/im2")
    p.add_argument("--label1_dir", default="SECOND/label1")
    p.add_argument("--label2_dir", default="SECOND/label2")
    p.add_argument("--output_dir", default="stage8/stage8b_debug")

    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vis_n", type=int, default=6)
    p.add_argument("--heatmap_n", type=int, default=3)

    p.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")

    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--pred_iou_thresh", type=float, default=0.75)
    p.add_argument("--stability_thresh", type=float, default=0.85)
    p.add_argument("--min_mask_area", type=int, default=256)

    # Eval-regeneration settings for stability test.
    p.add_argument("--eval_points_per_side", type=int, default=32)
    p.add_argument("--eval_pred_iou_thresh", type=float, default=0.75)
    p.add_argument("--eval_stability_thresh", type=float, default=0.85)
    p.add_argument("--eval_min_mask_area", type=int, default=256)

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
