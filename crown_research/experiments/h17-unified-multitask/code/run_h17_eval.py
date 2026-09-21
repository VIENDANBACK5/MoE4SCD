"""Evaluation, Diagnostic, and Benchmark Script for H17: Unified Multi-Task Architecture.

Evaluates:
1. Multi-split performance across evalhold60, test2 (in-domain), and test1 (out-of-domain Hain-AOI).
2. Seeding comparison: Global NMS vs Dynamic Scale-Adaptive NMS (H16) on both surface and adaptive_apex.
3. Head diagnostics: Boundary AUROC & seam separation; small-crown vs large-crown apex response.
4. Comprehensive 101-point COCO AP, AP50, AP75, and F1@50 metrics.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p != _local_p and p != _conda_p]
sys.path.insert(0, _conda_p)
sys.path.append(_local_p)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from model_h17 import CanopyWatershedNet4Head
from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_research.src.coco_ap import APAccumulator, instance_scores_from_surface
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

CACHE_DIR = Path("crown_research/data/cache")
OUT = Path("crown_research/experiments/h17-unified-multitask/results")
FIG = Path("crown_research/to_human/figures")
CKPT = "DeadTrees/experiments/canopy_watershed/best_unified_v2_watershed.pth"


def compute_superlevel_radius(surface_f: np.ndarray, y: int, x: int, r_thresh: float = 0.50, max_r: int = 150) -> float:
    """Estimates local crown support radius around local peak (y, x)."""
    H, W = surface_f.shape
    peak_val = float(surface_f[y, x])
    thresh = max(0.10, peak_val * r_thresh)
    y0, y1 = max(0, y - max_r), min(H, y + max_r + 1)
    x0, x1 = max(0, x - max_r), min(W, x + max_r + 1)
    sub = surface_f[y0:y1, x0:x1] >= thresh
    area = float(np.sum(sub))
    return float(np.sqrt(area / np.pi))


def adaptive_nms_seeds(
    field: np.ndarray,
    canopy: np.ndarray,
    gamma: float = 1.2,
    r_thresh: float = 0.45,
    r_min: float = 35.0,
    min_apex: float = 0.45,
) -> list[tuple[int, int, float]]:
    """Scale-adaptive local NMS seeding."""
    mask = (canopy > 0.5) & (field >= min_apex)
    candidates: list[tuple[float, int, int]] = []
    py, px = np.where(mask)
    for y, x in zip(py, px):
        candidates.append((float(field[y, x]), int(y), int(x)))
    candidates.sort(reverse=True, key=lambda c: c[0])

    seeds: list[tuple[int, int, float]] = []
    suppressed = np.zeros(field.shape, dtype=bool)

    for val, y, x in candidates:
        if suppressed[y, x]:
            continue
        seeds.append((y, x, val))
        r_supp = max(r_min, gamma * compute_superlevel_radius(field, y, x, r_thresh=r_thresh))
        r_int = int(np.ceil(r_supp))
        y0, y1 = max(0, y - r_int), min(field.shape[0], y + r_int + 1)
        x0, x1 = max(0, x - r_int), min(field.shape[1], x + r_int + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist_sq = (yy - y) ** 2 + (xx - x) ** 2
        suppressed[y0:y1, x0:x1] |= (dist_sq <= r_supp ** 2)

    return seeds


def build_h17_cache(split: str, num_crops: int, start_idx: int, tag: str, ckpt_path: str = CKPT) -> Path:
    out = CACHE_DIR / tag
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CanopyWatershedNet4Head(pretrained=False).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()

    ds = BAMCanopyWatershedDataset(split=split, crop_size=1024, augment=False, compute_targets=False)
    n = min(start_idx + num_crops, len(ds))
    kept = 0

    print(f"Building cache for {tag} ({split}, n={num_crops}, start={start_idx})...", flush=True)
    for idx in range(start_idx, n):
        f = out / f"crop_{idx:04d}.npz"
        if f.exists():
            kept += 1
            continue
        item = ds[idx]
        raw_gt = item["instance_label"].squeeze().numpy().astype(np.int32)
        gt_map = np.zeros_like(raw_gt)
        gt_map[raw_gt > 0] = raw_gt[raw_gt > 0] + 1
        if len(np.unique(raw_gt[raw_gt > 0])) == 0:
            continue

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model(item["image"].unsqueeze(0).to(device))

        np.savez_compressed(
            f,
            surface=preds["surface"].squeeze().float().cpu().numpy().astype(np.float16),
            boundary=preds["boundary"].squeeze().float().cpu().numpy().astype(np.float16),
            canopy=preds["canopy"].squeeze().float().cpu().numpy().astype(np.float16),
            adaptive_apex=preds["adaptive_apex"].squeeze().float().cpu().numpy().astype(np.float16),
            gt_map=gt_map.astype(np.int16),
            rgb=(item["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8),
        )
        kept += 1
        if (kept % 10 == 0) or (idx == n - 1):
            print(f"  --> cached {kept}/{num_crops} crops...", flush=True)

    print(f"[cache] Finished {tag}: {kept} crops saved at {out}")
    return out


def load_h17_cache(tag: str):
    """Yields (surface, boundary, canopy, adaptive_apex, gt_map, rgb)."""
    out = CACHE_DIR / tag
    for f in sorted(out.glob("crop_*.npz")):
        z = np.load(f)
        yield (
            z["surface"].astype(np.float32),
            z["boundary"].astype(np.float32),
            z["canopy"].astype(np.float32),
            z["adaptive_apex"].astype(np.float32),
            z["gt_map"].astype(np.int32),
            z["rgb"],
        )


def seam_band(gt: np.ndarray, r: int = 3) -> np.ndarray:
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gt.astype(np.float32), cv2.MORPH_GRADIENT, k3) > 0
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate((grad & (gt >= 2)).astype(np.uint8), kd) > 0


def head_diagnostic(crops: list) -> dict[str, float]:
    B, sy = [], []
    for _, bound, _, _, gt, _ in crops:
        sub = (slice(None, None, 3), slice(None, None, 3))
        B.append(bound[sub].ravel())
        sy.append(seam_band(gt)[sub].ravel())
    B, sy = np.concatenate(B), np.concatenate(sy)
    return {
        "B_on_seam": float(B[sy].mean()),
        "B_off_seam": float(B[~sy].mean()),
        "separation": float(B[sy].mean() / max(B[~sy].mean(), 1e-9)),
        "B_AUROC": float(roc_auc_score(sy, B)),
    }


def small_vs_large_crown_apex_audit(crops: list) -> dict[str, float]:
    small_apexes, large_apexes = [], []
    for _, _, _, apex, gt, _ in crops:
        gids = np.unique(gt)
        gids = gids[gids >= 2]
        for g in gids:
            mask = (gt == g)
            area = float(np.sum(mask))
            max_a = float(apex[mask].max()) if np.sum(mask) > 0 else 0.0
            if area < 300:
                small_apexes.append(max_a)
            elif area > 2000:
                large_apexes.append(max_a)

    return {
        "small_crown_n": len(small_apexes),
        "small_crown_mean_apex": float(np.mean(small_apexes)) if small_apexes else 0.0,
        "small_crown_lit_rate_pct": float(100.0 * np.mean(np.array(small_apexes) >= 0.40)) if small_apexes else 0.0,
        "large_crown_n": len(large_apexes),
        "large_crown_mean_apex": float(np.mean(large_apexes)) if large_apexes else 0.0,
        "large_crown_lit_rate_pct": float(100.0 * np.mean(np.array(large_apexes) >= 0.40)) if large_apexes else 0.0,
    }


def evaluate_crop_configuration(crops: list, seeding_fn, score_field_idx: int = 0) -> dict[str, float]:
    acc = APAccumulator()
    f1_list = []
    tot_gt = 0
    tot_seeds = 0
    tot_hit = 0
    tot_dup = 0
    tot_orphan = 0

    for item in crops:
        surf, bound, canopy, apex, gt, _ = item
        seeds = seeding_fn(surf, bound, canopy, apex)
        m, insts = flood(seeds, bound, canopy)

        score_field = surf if score_field_idx == 0 else apex
        insts = instance_scores_from_surface(m, insts, score_field)
        acc.add(m, insts, gt)
        f1_list.append(evaluate_metrics(m, insts, gt)["ap50"])

        gids = np.unique(gt)
        gids = gids[gids >= 2]
        tot_gt += len(gids)
        tot_seeds += len(seeds)

        seen = {}
        for y, x, _ in seeds:
            g = int(gt[y, x])
            if g < 2:
                tot_orphan += 1
            else:
                seen[g] = seen.get(g, 0) + 1
        tot_hit += len(seen)
        tot_dup += sum(v - 1 for v in seen.values())

    r = acc.ap()
    ratio = tot_seeds / max(1, tot_gt)
    hit_rate = 100.0 * tot_hit / max(1, tot_gt)
    dup_per_gt = tot_dup / max(1, tot_gt)
    orphan_per_gt = tot_orphan / max(1, tot_gt)

    return {
        "COCO AP": r["AP"],
        "AP50": r["AP50"],
        "AP75": r["AP75"],
        "F1@50": float(np.mean(f1_list)),
        "seed_ratio": ratio,
        "hit_rate_%": hit_rate,
        "dup_per_gt": dup_per_gt,
        "orphan_per_gt": orphan_per_gt,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    if not Path(CKPT).exists():
        print(f"Error: checkpoint {CKPT} not found!")
        return

    splits = {
        "evalhold60": ("eval", 60, 100, "evalhold60_h17"),
        "test2": ("test2", 40, 0, "test2_h17"),
        "test1": ("test1", 39, 0, "test1_h17"),
    }

    # 1. Build Caches
    for label, (sp, nc, st, tag) in splits.items():
        build_h17_cache(sp, nc, st, tag, ckpt_path=CKPT)

    # 2. Diagnostic Audits
    hd_rows = []
    apex_rows = []
    crops_dict = {}

    for label, (_, _, _, tag) in splits.items():
        crops = list(load_h17_cache(tag))
        crops_dict[label] = crops
        hd = head_diagnostic(crops)
        hd_rows.append({"split": label, **hd})
        apex_audit = small_vs_large_crown_apex_audit(crops)
        apex_rows.append({"split": label, **apex_audit})

    df_hd = pd.DataFrame(hd_rows).round(4)
    df_hd.to_csv(OUT / "h17_head_diagnostic.csv", index=False)
    print("\n" + "=" * 90)
    print("H17 Head Diagnostic (Boundary Seam Separation & AUROC):")
    print(df_hd.to_string(index=False))

    df_apex = pd.DataFrame(apex_rows).round(3)
    df_apex.to_csv(OUT / "h17_small_crown_audit.csv", index=False)
    print("\n" + "=" * 90)
    print("H17 Apex Responsiveness on Small vs. Large Crowns:")
    print(df_apex.to_string(index=False))
    print("=" * 90)

    # 3. Comprehensive Seeding & Benchmark Evaluation
    sweep_records = []

    # Configs to sweep:
    # A. Global NMS on Surface
    global_surf_grid = list(product([100, 140, 160, 180, 200], [0.40, 0.50, 0.60]))
    # B. Global NMS on Adaptive Apex
    global_apex_grid = list(product([40, 60, 80, 100, 140], [0.35, 0.45, 0.55]))
    # C. Dynamic Adaptive NMS on Adaptive Apex
    adaptive_apex_grid = [
        {"gamma": 1.2, "r_t": 0.45, "r_min": 35.0, "m_apex": 0.40},
        {"gamma": 1.4, "r_t": 0.45, "r_min": 35.0, "m_apex": 0.40},
        {"gamma": 1.6, "r_t": 0.40, "r_min": 40.0, "m_apex": 0.40},
        {"gamma": 1.6, "r_t": 0.45, "r_min": 30.0, "m_apex": 0.45},
        {"gamma": 1.8, "r_t": 0.40, "r_min": 40.0, "m_apex": 0.40},
    ]

    for label, crops in crops_dict.items():
        print(f"\nEvaluating seeding strategies on {label} ({len(crops)} crops)...", flush=True)

        # A. Global NMS on Surface
        for md, ma in global_surf_grid:
            res = evaluate_crop_configuration(
                crops,
                lambda s, b, c, a, md=md, ma=ma: global_nms_seeds(s, c, float(md), ma),
                score_field_idx=0,
            )
            sweep_records.append({
                "split": label,
                "strategy": "Global_NMS_Surface",
                "params": f"md={md},ma={ma}",
                **res,
            })

        # B. Global NMS on Adaptive Apex
        for md, ma in global_apex_grid:
            res = evaluate_crop_configuration(
                crops,
                lambda s, b, c, a, md=md, ma=ma: global_nms_seeds(a, c, float(md), ma),
                score_field_idx=0,
            )
            sweep_records.append({
                "split": label,
                "strategy": "Global_NMS_Apex",
                "params": f"md={md},ma={ma}",
                **res,
            })

        # C. Dynamic Scale-Adaptive NMS on Adaptive Apex
        for cfg in adaptive_apex_grid:
            res = evaluate_crop_configuration(
                crops,
                lambda s, b, c, a, cfg=cfg: adaptive_nms_seeds(
                    a, c, gamma=cfg["gamma"], r_thresh=cfg["r_t"], r_min=cfg["r_min"], min_apex=cfg["m_apex"]
                ),
                score_field_idx=0,
            )
            sweep_records.append({
                "split": label,
                "strategy": "Adaptive_NMS_Apex",
                "params": f"g={cfg['gamma']},rt={cfg['r_t']},rmin={cfg['r_min']},ma={cfg['m_apex']}",
                **res,
            })

    df_sweep = pd.DataFrame(sweep_records).round(3)
    df_sweep.to_csv(OUT / "h17_full_sweep.csv", index=False)

    # 4. Generate Best Summary Comparison
    best_summary = []
    for label in splits:
        for strat in ["Global_NMS_Surface", "Global_NMS_Apex", "Adaptive_NMS_Apex"]:
            sub = df_sweep[(df_sweep.split == label) & (df_sweep.strategy == strat)]
            if sub.empty:
                continue
            b = sub.sort_values("COCO AP", ascending=False).iloc[0]
            best_summary.append({
                "split": label,
                "strategy": strat,
                "best_params": b["params"],
                "COCO_AP": b["COCO AP"],
                "AP50": b["AP50"],
                "AP75": b["AP75"],
                "F1@50": b["F1@50"],
                "seed_ratio": b["seed_ratio"],
                "hit_rate_%": b["hit_rate_%"],
                "dup_per_gt": b["dup_per_gt"],
            })

    df_best = pd.DataFrame(best_summary)
    df_best.to_csv(OUT / "h17_benchmark_summary.csv", index=False)
    print("\n" + "=" * 105)
    print("H17 Multi-Split Seeding Benchmark Comparison:")
    print(df_best.to_string(index=False))
    print("=" * 105)


if __name__ == "__main__":
    main()
