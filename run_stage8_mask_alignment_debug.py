#!/usr/bin/env python3
"""Stage 8 debug: token-mask alignment diagnostics.

Runs systematic checks to diagnose whether token probabilities are assigned to
correct SAM masks when reconstructing pixel maps.
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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
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
EMB_SIZE = 64
SCALE = IMG_SIZE // EMB_SIZE


@dataclass
class MaskItem:
    mask: np.ndarray
    centroid: np.ndarray
    area: float


def _font(sz: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


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
    pooled = F.max_pool2d(t, kernel_size=SCALE, stride=SCALE)
    return pooled.squeeze() > 0.0


def masks_from_image(image_rgb: np.ndarray, amg: SAM2AutomaticMaskGenerator) -> List[MaskItem]:
    raw_masks = amg.generate(image_rgb)
    items: List[MaskItem] = []
    for m in raw_masks:
        seg = m["segmentation"]
        if seg.dtype != np.bool_:
            seg = seg.astype(bool)
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
    n_tok = token_centroids.shape[0]
    n_mask = len(mask_items)
    m_cent = np.stack([m.centroid for m in mask_items], axis=0)
    m_area = np.array([m.area for m in mask_items], dtype=np.float32)

    d_cent = np.sqrt(((token_centroids[:, None, :] - m_cent[None, :, :]) ** 2).sum(axis=2))
    d_area = np.abs(token_areas[:, None] - m_area[None, :])
    cost = 0.7 * d_cent + 0.3 * d_area

    r_idx, c_idx = linear_sum_assignment(cost)

    aligned: List[Optional[np.ndarray]] = [None] * n_tok
    cent_errs: List[float] = []
    area_errs: List[float] = []

    for r, c in zip(r_idx.tolist(), c_idx.tolist()):
        aligned[r] = mask_items[c].mask
        cent_errs.append(float(d_cent[r, c]))
        area_errs.append(float(d_area[r, c]))

    nearest = np.argmin(d_cent, axis=1)
    for i in range(n_tok):
        if aligned[i] is None:
            c = int(nearest[i])
            aligned[i] = mask_items[c].mask
            cent_errs.append(float(d_cent[i, c]))
            area_errs.append(float(d_area[i, c]))

    stats = {
        "n_tokens": float(n_tok),
        "n_masks": float(n_mask),
        "mean_centroid_error": float(np.mean(cent_errs)) if cent_errs else 0.0,
        "mean_area_error": float(np.mean(area_errs)) if area_errs else 0.0,
        "count_gap": float(abs(n_tok - n_mask)),
    }
    return [m.astype(bool) for m in aligned], stats


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
        outputs = model(batch)

    logits = outputs["change_logits"][0].cpu()
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
    _, ids = tree.query(grid, k=1)
    return ids.astype(np.int32).reshape(IMG_SIZE, IMG_SIZE)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    up = np.roll(m, -1, axis=0)
    dn = np.roll(m, 1, axis=0)
    lf = np.roll(m, -1, axis=1)
    rt = np.roll(m, 1, axis=1)
    b = (m != up) | (m != dn) | (m != lf) | (m != rt)
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = False
    return b & mask


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_centroid(img: Image.Image, cx: int, cy: int, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
    d = ImageDraw.Draw(img)
    r = 5
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)
    d.line((cx - 8, cy, cx + 8, cy), fill=color, width=2)
    d.line((cx, cy - 8, cx, cy + 8), fill=color, width=2)


def make_random_token_debug_grid(samples: List[Dict], save_path: Path) -> None:
    if not samples:
        return
    W, H = IMG_SIZE, IMG_SIZE
    pad = 6
    title_h = 20
    row_h = H + title_h + 28
    canvas = Image.new("RGB", (5 * (W + pad) + pad, len(samples) * row_h + 20), (242, 242, 246))
    draw = ImageDraw.Draw(canvas)
    headers = ["T1", "T2", "SAM mask overlay", "Mask+centroid", "Probability"]

    for r, s in enumerate(samples):
        y_base = 10 + r * row_h
        panels = [s["t1"], s["t2"], s["ov1"], s["ov2"], s["prob_panel"]]

        for c, arr in enumerate(panels):
            x = pad + c * (W + pad)
            draw.text((x + W // 2, y_base + 10), headers[c], fill=(20, 20, 20), font=_font(12), anchor="mm")
            if isinstance(arr, np.ndarray):
                img = Image.fromarray(arr)
            else:
                img = arr
            canvas.paste(img, (x, y_base + title_h))

        meta = f"stem={s['stem']} token={s['token_id']} p={s['prob']:.3f} centroid=({s['cx']},{s['cy']})"
        draw.text((canvas.width // 2, y_base + title_h + H + 12), meta, fill=(60, 60, 60), font=_font(10), anchor="mm")

    canvas.save(save_path)


def make_image_sanity_grid(samples: List[Image.Image], save_path: Path) -> None:
    if not samples:
        return
    pw, ph = samples[0].size
    cols = 1 if len(samples) < 3 else 2
    rows = math.ceil(len(samples) / cols)
    canvas = Image.new("RGB", (cols * pw + 8, rows * ph + 40), (240, 240, 244))
    draw = ImageDraw.Draw(canvas)
    draw.text((canvas.width // 2, 16), "Token probability assignment sanity examples", fill=(30, 30, 30), font=_font(14), anchor="mm")
    for i, p in enumerate(samples):
        r = i // cols
        c = i % cols
        canvas.paste(p, (4 + c * pw, 30 + r * ph))
    canvas.save(save_path)


def make_sanity_panel(stem: str, t1: np.ndarray, t2: np.ndarray, centroids: np.ndarray, masks: Sequence[np.ndarray], probs: np.ndarray) -> Image.Image:
    top_idx = np.argsort(-probs)[:5].tolist()
    ov = t1.copy()
    for idx in top_idx:
        m = masks[idx]
        color = (255, 80 + (idx * 33) % 160, 60 + (idx * 57) % 180)
        ov = overlay_mask(ov, m, color, alpha=0.25)
        b = mask_boundary(m)
        ov[b] = [255, 255, 0]
        cx = int(round(float(centroids[idx, 0]) * (IMG_SIZE - 1)))
        cy = int(round(float(centroids[idx, 1]) * (IMG_SIZE - 1)))
        cv = Image.fromarray(ov)
        d = ImageDraw.Draw(cv)
        d.text((cx + 6, cy - 6), f"{idx}:{probs[idx]:.2f}", fill=(255, 255, 255), font=_font(12))
        ov = np.array(cv)

    # Put centroids for top-k.
    cent = Image.fromarray(ov)
    for idx in top_idx:
        cx = int(round(float(centroids[idx, 0]) * (IMG_SIZE - 1)))
        cy = int(round(float(centroids[idx, 1]) * (IMG_SIZE - 1)))
        draw_centroid(cent, cx, cy, (255, 0, 0))

    W, H = IMG_SIZE, IMG_SIZE
    pad = 6
    title_h = 20
    canvas = Image.new("RGB", (3 * (W + pad) + pad, H + title_h + 28), (245, 245, 248))
    draw = ImageDraw.Draw(canvas)
    titles = ["T1", "T2", "T1 + SAM boundaries + p"]
    imgs = [Image.fromarray(t1), Image.fromarray(t2), cent]
    for i, im in enumerate(imgs):
        x = pad + i * (W + pad)
        draw.text((x + W // 2, 10), titles[i], fill=(30, 30, 30), font=_font(12), anchor="mm")
        canvas.paste(im, (x, title_h))
    draw.text((canvas.width // 2, canvas.height - 12), f"{stem} | top-5 probability tokens annotated", fill=(60, 60, 60), font=_font(10), anchor="mm")
    return canvas


def write_report(path: Path, stats: Dict[str, float], args, vis_paths: Dict[str, Path]) -> None:
    lines = [
        "# Stage 8 — Mask Alignment Debug Report",
        "",
        "## Setup",
        "",
        f"- Model: `{args.model_dir}`",
        "- Dataset: SECOND validation split",
        f"- Images processed: `{int(stats['images_processed'])}`",
        f"- Total tokens checked: `{int(stats['total_tokens'])}`",
        "",
        "---",
        "",
        "## Test 1 — Centroid-in-Mask Validation",
        "",
        f"- Inside count: `{int(stats['centroid_inside'])}`",
        f"- Outside count: `{int(stats['centroid_outside'])}`",
        f"- Centroid inside ratio: `{stats['centroid_inside_ratio']:.6f}`",
        "",
        "Interpretation: ratio should be close to 1.0; values < 0.95 indicate likely token-mask misalignment.",
        "",
        "---",
        "",
        "## Test 2 — Token-Mask Count Consistency",
        "",
        f"- Images with count mismatch: `{int(stats['count_mismatch_images'])}`",
        f"- Mean absolute count gap: `{stats['mean_count_gap']:.4f}`",
        f"- Max count gap: `{int(stats['max_count_gap'])}`",
        "",
        "---",
        "",
        "## Test 3 — Mask Area Consistency",
        "",
        f"- Mean mask area (pixels): `{stats['mean_mask_area_px']:.2f}`",
        f"- Min mask area (pixels): `{int(stats['min_mask_area_px'])}`",
        f"- Max mask area (pixels): `{int(stats['max_mask_area_px'])}`",
        f"- Zero-area masks: `{int(stats['zero_area_masks'])}`",
        f"- Extremely large masks (>50% image): `{int(stats['huge_masks'])}`",
        "",
        "---",
        "",
        "## Test 4 — Probability Assignment Sanity",
        "",
        f"- Saved qualitative sanity overlays: `{vis_paths['sanity'].name}`",
        "- Each panel annotates top-probability tokens with mask boundaries and probability labels.",
        "",
        "---",
        "",
        "## Test 5 — Random Token Visualisation",
        "",
        f"- Saved random token grid: `{vis_paths['random_tokens'].name}`",
        "- Shows T1, T2, SAM mask overlays, centroid, and token probability for 10 random tokens.",
        "",
        "---",
        "",
        "## Test 6 — Voronoi vs SAM Overlap",
        "",
        f"- Mean IoU(Voronoi region, SAM mask): `{stats['vor_sam_iou_mean']:.4f}`",
        f"- Std IoU: `{stats['vor_sam_iou_std']:.4f}`",
        f"- Fraction IoU < 0.6: `{stats['vor_sam_iou_lt_06_ratio']:.4f}`",
        "",
        "---",
        "",
        "## Overall Conclusion",
        "",
    ]

    if stats["centroid_inside_ratio"] < 0.95 or stats["count_mismatch_images"] > 0:
        lines.append(
            "Potential alignment bug detected: centroid-inside ratio or count consistency indicates mask-token mismatch."
        )
    elif stats["vor_sam_iou_mean"] < 0.6:
        lines.append(
            "No direct ordering mismatch detected, but Voronoi and SAM shapes differ strongly (low IoU), so reconstruction quality can diverge due to spatial geometry."
        )
    else:
        lines.append(
            "No strong evidence of token-mask misalignment bug in these checks; SAM underperformance likely comes from projection behavior and threshold/calibration effects rather than wrong token-to-mask assignment."
        )

    lines += [
        "",
        "_Generated by `run_stage8_mask_alignment_debug.py`_",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_dir = Path(args.model_dir)
    t1_dir = Path(args.tokens_T1)
    t2_dir = Path(args.tokens_T2)
    match_dir = Path(args.matches)
    img1_dir = Path(args.images_T1)
    img2_dir = Path(args.images_T2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir} ...")
    model = load_model(model_dir, device)
    print(f"  Router v{model.cfg.router_version} | Experts: {model.cfg.moe_num_experts}")

    print("Loading SAM2 generator ...")
    amg = load_amg(args, device)

    stems = build_val_stems(match_dir, t1_dir, t2_dir, args.val_split, args.seed)
    if args.max_samples is not None:
        stems = stems[: args.max_samples]
    print(f"Validation stems: {len(stems)}")

    centroid_inside = 0
    centroid_outside = 0
    total_tokens = 0

    count_gaps: List[int] = []
    area_values: List[int] = []
    zero_area_masks = 0
    huge_masks = 0
    all_iou: List[float] = []

    random_tokens: List[Dict] = []
    random_seen = 0

    sanity_panels: List[Image.Image] = []

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

        t1_centroids = t1["centroids"].numpy().astype(np.float32)
        t1_areas = t1["areas"].numpy().astype(np.float32)
        probs = compute_token_probs(model, t1, t2, mtch, device)

        img_t1 = np.array(Image.open(p1).convert("RGB"))
        img_t2 = np.array(Image.open(p2).convert("RGB"))

        regen_masks = masks_from_image(img_t1, amg)
        aligned_masks, align_stats = align_masks_to_tokens(regen_masks, t1_centroids, t1_areas)

        # Test 2
        gap = abs(len(regen_masks) - len(t1_centroids))
        count_gaps.append(gap)

        # Test 1 + Test 3
        for token_id, m in enumerate(aligned_masks):
            cx = int(round(float(t1_centroids[token_id, 0]) * (IMG_SIZE - 1)))
            cy = int(round(float(t1_centroids[token_id, 1]) * (IMG_SIZE - 1)))
            cx = min(max(cx, 0), IMG_SIZE - 1)
            cy = min(max(cy, 0), IMG_SIZE - 1)

            inside = bool(m[cy, cx])
            if inside:
                centroid_inside += 1
            else:
                centroid_outside += 1
            total_tokens += 1

            area = int(m.sum())
            area_values.append(area)
            if area == 0:
                zero_area_masks += 1
            if area > (IMG_SIZE * IMG_SIZE // 2):
                huge_masks += 1

            # Reservoir sampling for random token visualization
            random_seen += 1
            candidate = {
                "stem": stem,
                "token_id": token_id,
                "prob": float(probs[token_id]),
                "cx": cx,
                "cy": cy,
                "mask": m.copy(),
                "t1": img_t1.copy(),
                "t2": img_t2.copy(),
            }
            if len(random_tokens) < 10:
                random_tokens.append(candidate)
            else:
                j = random.randint(0, random_seen - 1)
                if j < 10:
                    random_tokens[j] = candidate

        # Test 6
        v_ids = voronoi_ids(t1_centroids)
        for token_id, m in enumerate(aligned_masks):
            v = v_ids == token_id
            inter = int((v & m).sum())
            union = int((v | m).sum())
            iou = inter / max(union, 1)
            all_iou.append(iou)

        # Test 4 samples
        if len(sanity_panels) < args.sanity_n:
            panel = make_sanity_panel(stem, img_t1, img_t2, t1_centroids, aligned_masks, probs)
            sanity_panels.append(panel)

        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(stems)}] processed ...")

    if total_tokens == 0:
        raise RuntimeError("No tokens processed.")

    # Build random token visual figure
    random_token_rows: List[Dict] = []
    for s in random_tokens:
        m = s["mask"]
        ov1 = overlay_mask(s["t1"], m, (255, 0, 0), alpha=0.35)
        b = mask_boundary(m)
        ov2 = s["t1"].copy()
        ov2[b] = [255, 255, 0]
        pimg = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (32, 36, 45))
        d = ImageDraw.Draw(pimg)
        d.text((IMG_SIZE // 2, IMG_SIZE // 2 - 20), "Token probability", fill=(230, 230, 230), font=_font(24), anchor="mm")
        d.text((IMG_SIZE // 2, IMG_SIZE // 2 + 25), f"p = {s['prob']:.4f}", fill=(255, 200, 80), font=_font(34), anchor="mm")
        draw_centroid(Image.fromarray(ov1), s["cx"], s["cy"], (0, 255, 255))

        ov1_img = Image.fromarray(ov1)
        draw_centroid(ov1_img, s["cx"], s["cy"], (0, 255, 255))

        ov2_img = Image.fromarray(ov2)
        draw_centroid(ov2_img, s["cx"], s["cy"], (255, 0, 0))

        random_token_rows.append(
            {
                "stem": s["stem"],
                "token_id": s["token_id"],
                "prob": s["prob"],
                "cx": s["cx"],
                "cy": s["cy"],
                "t1": s["t1"],
                "t2": s["t2"],
                "ov1": np.array(ov1_img),
                "ov2": np.array(ov2_img),
                "prob_panel": pimg,
            }
        )

    stats = {
        "images_processed": float(len(stems)),
        "total_tokens": float(total_tokens),
        "centroid_inside": float(centroid_inside),
        "centroid_outside": float(centroid_outside),
        "centroid_inside_ratio": float(centroid_inside / max(total_tokens, 1)),
        "count_mismatch_images": float(sum(1 for g in count_gaps if g != 0)),
        "mean_count_gap": float(np.mean(count_gaps)) if count_gaps else 0.0,
        "max_count_gap": float(np.max(count_gaps)) if count_gaps else 0.0,
        "mean_mask_area_px": float(np.mean(area_values)) if area_values else 0.0,
        "min_mask_area_px": float(np.min(area_values)) if area_values else 0.0,
        "max_mask_area_px": float(np.max(area_values)) if area_values else 0.0,
        "zero_area_masks": float(zero_area_masks),
        "huge_masks": float(huge_masks),
        "vor_sam_iou_mean": float(np.mean(all_iou)) if all_iou else 0.0,
        "vor_sam_iou_std": float(np.std(all_iou)) if all_iou else 0.0,
        "vor_sam_iou_lt_06_ratio": float(np.mean(np.array(all_iou) < 0.6)) if all_iou else 0.0,
    }

    random_vis_path = out_dir / "debug_token_mask_alignment.png"
    sanity_vis_path = out_dir / "debug_probability_assignment_sanity.png"
    report_path = out_dir / "stage8_mask_alignment_debug.md"

    make_random_token_debug_grid(random_token_rows, random_vis_path)
    make_image_sanity_grid(sanity_panels, sanity_vis_path)
    write_report(
        report_path,
        stats,
        args,
        {
            "random_tokens": random_vis_path,
            "sanity": sanity_vis_path,
        },
    )

    print("\n" + "=" * 72)
    print("STAGE 8 MASK ALIGNMENT DEBUG SUMMARY")
    print("=" * 72)
    print(f"Centroid inside ratio: {stats['centroid_inside_ratio']:.6f}")
    print(f"Count mismatches: {int(stats['count_mismatch_images'])} images")
    print(
        f"Mask area px: mean={stats['mean_mask_area_px']:.2f}, min={int(stats['min_mask_area_px'])}, "
        f"max={int(stats['max_mask_area_px'])}, zero={int(stats['zero_area_masks'])}, huge(>50%)={int(stats['huge_masks'])}"
    )
    print(
        f"Voronoi vs SAM IoU: mean={stats['vor_sam_iou_mean']:.4f}, std={stats['vor_sam_iou_std']:.4f}, "
        f"frac<0.6={stats['vor_sam_iou_lt_06_ratio']:.4f}"
    )
    print("=" * 72)
    print(f"Saved: {random_vis_path}")
    print(f"Saved: {sanity_vis_path}")
    print(f"Saved: {report_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 8 token-mask alignment debug")
    p.add_argument("--model_dir", default="SECOND/stage5_6_dynamic")
    p.add_argument("--tokens_T1", default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2", default="SECOND/tokens_T2")
    p.add_argument("--matches", default="SECOND/matches")
    p.add_argument("--images_T1", default="SECOND/im1")
    p.add_argument("--images_T2", default="SECOND/im2")
    p.add_argument("--output_dir", default="stage8/reconstruction")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--sanity_n", type=int, default=8)

    p.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--pred_iou_thresh", type=float, default=0.75)
    p.add_argument("--stability_thresh", type=float, default=0.85)
    p.add_argument("--min_mask_area", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
