#!/usr/bin/env python3
"""Independent branch: feature-guided pixel refinement decoder experiment.

Pipeline:
T1,T2 -> token model -> coarse Voronoi map -> feature-guided UNet refinement.
Compare on validation split:
1) Voronoi reconstruction
2) SAM partition reconstruction
3) Feature-guided UNet refinement
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
SAM2_REPO = ROOT / "sam2"
if SAM2_REPO.is_dir():
    sys.path.insert(0, str(SAM2_REPO))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from token_change_reasoner import SampleData, build_batch
from token_change_reasoner_moe import MoEConfig, TokenChangeReasonerMoE, build_moe_model

IMG = 512
LOW = 256

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


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RefinementUNet(nn.Module):
    """Lightweight UNet per requested structure."""

    def __init__(self, in_ch: int):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return torch.sigmoid(self.out(d2))


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target [B,1,H,W]
        p = pred.reshape(pred.shape[0], -1)
        t = target.reshape(target.shape[0], -1)
        inter = (p * t).sum(dim=1)
        den = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    """Gradient-based boundary loss using Sobel magnitude alignment."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gp = self._grad_mag(pred)
        gt = self._grad_mag(target)
        return F.l1_loss(gp, gt)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_splits(match_dir: Path, t1_dir: Path, t2_dir: Path, val_split: float, seed: int):
    stems = [
        mp.stem.replace("_matches", "")
        for mp in sorted(match_dir.glob("*_matches.pt"))
        if (t1_dir / f"{mp.stem.replace('_matches', '')}.pt").exists()
        and (t2_dir / f"{mp.stem.replace('_matches', '')}.pt").exists()
    ]
    n_val = max(1, int(len(stems) * val_split))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(stems), generator=g).tolist()
    val = [stems[i] for i in perm[len(stems) - n_val :]]
    val_set = set(val)
    train = [s for s in stems if s not in val_set]
    return train, val


def load_model(model_dir: Path, device: torch.device) -> TokenChangeReasonerMoE:
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        cfgd = json.load(f)
    cfg = MoEConfig(**{k: v for k, v in cfgd.items() if k in MoEConfig.__dataclass_fields__})
    model = build_moe_model(cfg).to(device)
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_amg(args, device: torch.device) -> SAM2AutomaticMaskGenerator:
    ck = Path(args.sam2_ckpt)
    if not ck.is_absolute():
        ck = SAM2_REPO / args.sam2_ckpt
    if not ck.exists():
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {ck}")
    sam2_model = build_sam2(args.sam2_config, str(ck), device=str(device))
    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        min_mask_region_area=args.min_mask_area,
        output_mode="binary_mask",
    )


def _rgb_to_class(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    d = ((flat[:, None, :] - _SECOND_COLOURS[None, :, :]) ** 2).sum(-1)
    return d.argmin(-1).reshape(rgb.shape[:2])


def get_gt_change(stem: str, label1_dir: Path, label2_dir: Path) -> np.ndarray:
    a = np.array(Image.open(label1_dir / f"{stem}.png").convert("RGB"))
    b = np.array(Image.open(label2_dir / f"{stem}.png").convert("RGB"))
    return _rgb_to_class(a) != _rgb_to_class(b)


def token_probs_for_stem(stem: str, model: TokenChangeReasonerMoE, t1_dir: Path, t2_dir: Path, match_dir: Path, device: torch.device):
    t1 = torch.load(t1_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
    t2 = torch.load(t2_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
    mt = torch.load(match_dir / f"{stem}_matches.pt", map_location="cpu", weights_only=False)

    pairs_raw = mt.get("pairs", [])
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
    mask_t1 = (~pad) & (tid == 0)
    probs = torch.sigmoid(logits[mask_t1]).numpy()
    cents = t1["centroids"].numpy().astype(np.float32)
    areas = t1["areas"].numpy().astype(np.float32)
    return probs, cents, areas


def voronoi_map(centroids: np.ndarray, H: int = IMG, W: int = IMG) -> np.ndarray:
    pts = centroids[:, ::-1].copy()
    pts[:, 0] *= (H - 1)
    pts[:, 1] *= (W - 1)
    ys, xs = np.mgrid[0:H, 0:W]
    q = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    tree = cKDTree(pts)
    _, idx = tree.query(q, k=1)
    return idx.astype(np.int32).reshape(H, W)


def generate_masks(img: np.ndarray, amg: SAM2AutomaticMaskGenerator) -> List[MaskItem]:
    raw = amg.generate(img)
    out: List[MaskItem] = []
    for m in raw:
        seg = m["segmentation"]
        if seg.dtype != np.bool_:
            seg = seg.astype(bool)
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        cx = float(xs.mean()) / (IMG - 1)
        cy = float(ys.mean()) / (IMG - 1)
        ar = float(seg.sum()) / (IMG * IMG)
        sc = float(m.get("predicted_iou", 0.0))
        out.append(MaskItem(seg, np.array([cx, cy], dtype=np.float32), ar, sc))
    if not out:
        full = np.ones((IMG, IMG), dtype=bool)
        out.append(MaskItem(full, np.array([0.5, 0.5], dtype=np.float32), 1.0, 0.0))
    return out


def align_masks(mask_items: List[MaskItem], token_centroids: np.ndarray, token_areas: np.ndarray):
    mc = np.stack([m.centroid for m in mask_items], axis=0)
    ma = np.array([m.area_ratio for m in mask_items], dtype=np.float32)
    ms = np.array([m.score for m in mask_items], dtype=np.float32)
    dc = np.sqrt(((token_centroids[:, None, :] - mc[None, :, :]) ** 2).sum(axis=2))
    da = np.abs(token_areas[:, None] - ma[None, :])
    cost = 0.7 * dc + 0.3 * da
    r, c = linear_sum_assignment(cost)

    aligned_masks: List[Optional[np.ndarray]] = [None] * len(token_centroids)
    aligned_scores = np.zeros(len(token_centroids), dtype=np.float32)
    aligned_areas = np.zeros(len(token_centroids), dtype=np.float32)
    for i, j in zip(r.tolist(), c.tolist()):
        aligned_masks[i] = mask_items[j].mask
        aligned_scores[i] = ms[j]
        aligned_areas[i] = ma[j]

    nearest = np.argmin(dc, axis=1)
    for i in range(len(aligned_masks)):
        if aligned_masks[i] is None:
            j = int(nearest[i])
            aligned_masks[i] = mask_items[j].mask
            aligned_scores[i] = ms[j]
            aligned_areas[i] = ma[j]

    return [m.astype(bool) for m in aligned_masks], aligned_scores, aligned_areas


def sam_partition_map(
    masks: Sequence[np.ndarray],
    centroids: np.ndarray,
    scores: np.ndarray,
    areas: np.ndarray,
    chooser: str = "area",
    H: int = IMG,
    W: int = IMG,
) -> np.ndarray:
    priority = scores if chooser == "score" else areas
    order = np.argsort(priority)
    part = np.full((H, W), -1, dtype=np.int32)
    for i in order.tolist():
        part[masks[i]] = i

    miss = part < 0
    if miss.any():
        pts = centroids[:, ::-1].copy()
        pts[:, 0] *= (H - 1)
        pts[:, 1] *= (W - 1)
        tree = cKDTree(pts)
        ys, xs = np.where(miss)
        q = np.stack([ys, xs], axis=1).astype(np.float32)
        _, idx = tree.query(q, k=1)
        part[ys, xs] = idx.astype(np.int32)

    return part


def image_gradient_edge(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    d = cv2.absdiff(t1, t2)
    g = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    mag = mag / (mag.max() + 1e-6)
    return mag.astype(np.float32)


def build_input_tensor(
    stem: str,
    coarse256: np.ndarray,
    highres_t1_dir: Path,
    highres_t2_dir: Path,
    img1_dir: Path,
    img2_dir: Path,
    use_edge: bool,
) -> torch.Tensor:
    h1 = torch.load(highres_t1_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
    h2 = torch.load(highres_t2_dir / f"{stem}.pt", map_location="cpu", weights_only=True)

    f256_a = h1[0].float().squeeze(0)  # [32,256,256]
    f128_a = h1[1].float()             # [1,64,128,128]
    f256_b = h2[0].float().squeeze(0)
    f128_b = h2[1].float()

    d256 = torch.abs(f256_a - f256_b)                          # [32,256,256]
    d128 = torch.abs(f128_a - f128_b)                          # [1,64,128,128]
    d128u = F.interpolate(d128, size=(LOW, LOW), mode="bilinear", align_corners=False).squeeze(0)  # [64,256,256]

    c = torch.from_numpy(coarse256.astype(np.float32)).unsqueeze(0)

    xs = [c, d256, d128u]

    if use_edge:
        t1 = np.array(Image.open(img1_dir / f"{stem}.png").convert("RGB"))
        t2 = np.array(Image.open(img2_dir / f"{stem}.png").convert("RGB"))
        e = image_gradient_edge(t1, t2)
        e = cv2.resize(e, (LOW, LOW), interpolation=cv2.INTER_LINEAR)
        xs.append(torch.from_numpy(e).unsqueeze(0))

    return torch.cat(xs, dim=0)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1, "iou": iou}


def aggregate(rows: List[Dict], pfx: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ["precision", "recall", "f1", "iou"]:
        vals = [r[f"{pfx}_{k}"] for r in rows]
        out[f"macro_{pfx}_{k}"] = float(np.mean(vals)) if vals else 0.0
        out[f"std_{pfx}_{k}"] = float(np.std(vals)) if vals else 0.0

    tp = sum(int(r[f"{pfx}_tp"]) for r in rows)
    fp = sum(int(r[f"{pfx}_fp"]) for r in rows)
    fn = sum(int(r[f"{pfx}_fn"]) for r in rows)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    out[f"micro_{pfx}_precision"] = p
    out[f"micro_{pfx}_recall"] = r
    out[f"micro_{pfx}_f1"] = f1
    out[f"micro_{pfx}_iou"] = iou
    return out


class RefinementDataset(Dataset):
    def __init__(
        self,
        stems: Sequence[str],
        coarse_cache_dir: Path,
        highres_t1_dir: Path,
        highres_t2_dir: Path,
        img1_dir: Path,
        img2_dir: Path,
        label1_dir: Path,
        label2_dir: Path,
        use_edge: bool,
    ):
        self.stems = list(stems)
        self.cache = coarse_cache_dir
        self.h1 = highres_t1_dir
        self.h2 = highres_t2_dir
        self.i1 = img1_dir
        self.i2 = img2_dir
        self.l1 = label1_dir
        self.l2 = label2_dir
        self.use_edge = use_edge

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        s = self.stems[idx]
        c = np.load(self.cache / f"{s}_coarse256.npy")
        x = build_input_tensor(s, c, self.h1, self.h2, self.i1, self.i2, self.use_edge)

        gt = get_gt_change(s, self.l1, self.l2).astype(np.float32)
        gt256 = cv2.resize(gt, (LOW, LOW), interpolation=cv2.INTER_NEAREST)
        y = torch.from_numpy(gt256).unsqueeze(0)
        return x, y, s


def ensure_coarse_cache(
    stems: Sequence[str],
    cache_dir: Path,
    token_model: TokenChangeReasonerMoE,
    t1_dir: Path,
    t2_dir: Path,
    match_dir: Path,
    device: torch.device,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(stems):
        p = cache_dir / f"{s}_coarse256.npy"
        if p.exists():
            continue
        probs, cents, _ = token_probs_for_stem(s, token_model, t1_dir, t2_dir, match_dir, device)
        vid = voronoi_map(cents)
        coarse = probs[vid].astype(np.float32)
        coarse256 = cv2.resize(coarse, (LOW, LOW), interpolation=cv2.INTER_LINEAR)
        np.save(p, coarse256.astype(np.float32))
        if (i + 1) % 50 == 0:
            print(f"  coarse cache [{i + 1}/{len(stems)}] ...")


def evaluate(
    stems: Sequence[str],
    refine_model: RefinementUNet,
    token_model: TokenChangeReasonerMoE,
    amg: SAM2AutomaticMaskGenerator,
    args,
    device: torch.device,
    coarse_cache_dir: Path,
):
    t1_dir = Path(args.tokens_T1)
    t2_dir = Path(args.tokens_T2)
    match_dir = Path(args.matches)
    img1_dir = Path(args.images_T1)
    img2_dir = Path(args.images_T2)
    label1_dir = Path(args.label1_dir)
    label2_dir = Path(args.label2_dir)
    highres_t1_dir = Path(args.highres_T1)
    highres_t2_dir = Path(args.highres_T2)

    rows: List[Dict] = []
    vis: List[Dict] = []

    refine_model.eval()

    for i, s in enumerate(stems):
        p1 = img1_dir / f"{s}.png"
        p2 = img2_dir / f"{s}.png"
        if not (p1.exists() and p2.exists()):
            continue

        gt = get_gt_change(s, label1_dir, label2_dir)

        probs, cents, areas = token_probs_for_stem(s, token_model, t1_dir, t2_dir, match_dir, device)
        vid = voronoi_map(cents)
        pred_vor = probs[vid]

        # SAM partition baseline.
        t1_img = np.array(Image.open(p1).convert("RGB"))
        masks = generate_masks(t1_img, amg)
        aligned_masks, aligned_scores, aligned_areas = align_masks(masks, cents, areas)
        part = sam_partition_map(
            aligned_masks,
            cents,
            scores=aligned_scores,
            areas=aligned_areas,
            chooser=args.partition_chooser,
        )
        pred_part = probs[part]

        # Refinement.
        c256 = np.load(coarse_cache_dir / f"{s}_coarse256.npy")
        x = build_input_tensor(s, c256, highres_t1_dir, highres_t2_dir, img1_dir, img2_dir, args.use_edge_map)
        with torch.no_grad():
            y = refine_model(x.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu().numpy()
        pred_ref = cv2.resize(y, (IMG, IMG), interpolation=cv2.INTER_LINEAR)

        b_v = pred_vor > args.threshold
        b_p = pred_part > args.threshold
        b_r = pred_ref > args.threshold

        mv = compute_metrics(b_v, gt)
        mp = compute_metrics(b_p, gt)
        mr = compute_metrics(b_r, gt)

        row = {
            "stem": s,
            "vor_f1": mv["f1"],
            "vor_iou": mv["iou"],
            "vor_precision": mv["precision"],
            "vor_recall": mv["recall"],
            "vor_tp": mv["tp"],
            "vor_fp": mv["fp"],
            "vor_fn": mv["fn"],
            "vor_tn": mv["tn"],
            "part_f1": mp["f1"],
            "part_iou": mp["iou"],
            "part_precision": mp["precision"],
            "part_recall": mp["recall"],
            "part_tp": mp["tp"],
            "part_fp": mp["fp"],
            "part_fn": mp["fn"],
            "part_tn": mp["tn"],
            "ref_f1": mr["f1"],
            "ref_iou": mr["iou"],
            "ref_precision": mr["precision"],
            "ref_recall": mr["recall"],
            "ref_tp": mr["tp"],
            "ref_fp": mr["fp"],
            "ref_fn": mr["fn"],
            "ref_tn": mr["tn"],
            "delta_iou_ref_vs_vor": mr["iou"] - mv["iou"],
            "delta_iou_ref_vs_part": mr["iou"] - mp["iou"],
        }
        rows.append(row)

        if len(vis) < args.vis_n:
            t2_img = np.array(Image.open(p2).convert("RGB"))
            vis.append(
                {
                    "stem": s,
                    "t1": t1_img,
                    "t2": t2_img,
                    "coarse": cv2.resize(c256, (IMG, IMG), interpolation=cv2.INTER_LINEAR),
                    "refined": pred_ref,
                    "gt": gt,
                }
            )

        if (i + 1) % 20 == 0:
            print(f"  eval [{i + 1}/{len(stems)}] ...")

    return rows, vis


def draw_boundary_overlay(base: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = base.copy()
    pb = cv2.Canny((pred * 255).astype(np.uint8), 60, 120) > 0
    gb = cv2.Canny((gt.astype(np.uint8) * 255), 60, 120) > 0
    out[gb] = [0, 255, 0]
    out[pb] = [255, 0, 0]
    return out


def save_visuals(vis: List[Dict], path: Path):
    if not vis:
        return
    rows = len(vis)
    fig, axes = plt.subplots(rows, 6, figsize=(22, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, v in enumerate(vis):
        gt = v["gt"]
        co = v["coarse"]
        rf = v["refined"]
        ov = draw_boundary_overlay(v["t1"], rf > 0.5, gt)

        titles = ["T1", "T2", "Coarse", "Refined", "GT", "Boundary diff"]
        imgs = [v["t1"], v["t2"], co, rf, gt.astype(np.uint8), ov]
        cmaps = [None, None, "jet", "jet", "gray", None]

        for j in range(6):
            ax = axes[i, j]
            if cmaps[j] is None:
                ax.imshow(imgs[j])
            else:
                ax.imshow(imgs[j], cmap=cmaps[j], vmin=0, vmax=1 if j in [2, 3] else None)
            ax.set_title(f"{v['stem']} - {titles[j]}" if j == 0 else titles[j])
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    import csv

    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_report(path: Path, args, agg: Dict[str, float], n_train: int, n_val: int):
    lines = [
        "# Feature-Guided Refinement Decoder Experiment",
        "",
        "## Setup",
        "",
        f"- Model: `{args.model_dir}`",
        f"- Train samples: `{n_train}`",
        f"- Val samples: `{n_val}`",
        f"- Input channels: `{args.input_channels}`",
        f"- Epochs: `{args.epochs}`",
        f"- Threshold: `{args.threshold}`",
        "",
        "## Micro Metrics",
        "",
        "| Decoder | F1 | IoU | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
        f"| Voronoi | {agg['micro_vor_f1']:.4f} | {agg['micro_vor_iou']:.4f} | {agg['micro_vor_precision']:.4f} | {agg['micro_vor_recall']:.4f} |",
        f"| SAM partition | {agg['micro_part_f1']:.4f} | {agg['micro_part_iou']:.4f} | {agg['micro_part_precision']:.4f} | {agg['micro_part_recall']:.4f} |",
        f"| Feature-guided refine | {agg['micro_ref_f1']:.4f} | {agg['micro_ref_iou']:.4f} | {agg['micro_ref_precision']:.4f} | {agg['micro_ref_recall']:.4f} |",
        "",
        "## Macro Metrics (mean +- std)",
        "",
        "| Decoder | F1 | IoU | Precision | Recall |",
        "|---|---|---|---|---|",
        (
            f"| Voronoi | {agg['macro_vor_f1']:.4f} +- {agg['std_vor_f1']:.4f} "
            f"| {agg['macro_vor_iou']:.4f} +- {agg['std_vor_iou']:.4f} "
            f"| {agg['macro_vor_precision']:.4f} +- {agg['std_vor_precision']:.4f} "
            f"| {agg['macro_vor_recall']:.4f} +- {agg['std_vor_recall']:.4f} |"
        ),
        (
            f"| SAM partition | {agg['macro_part_f1']:.4f} +- {agg['std_part_f1']:.4f} "
            f"| {agg['macro_part_iou']:.4f} +- {agg['std_part_iou']:.4f} "
            f"| {agg['macro_part_precision']:.4f} +- {agg['std_part_precision']:.4f} "
            f"| {agg['macro_part_recall']:.4f} +- {agg['std_part_recall']:.4f} |"
        ),
        (
            f"| Feature-guided refine | {agg['macro_ref_f1']:.4f} +- {agg['std_ref_f1']:.4f} "
            f"| {agg['macro_ref_iou']:.4f} +- {agg['std_ref_iou']:.4f} "
            f"| {agg['macro_ref_precision']:.4f} +- {agg['std_ref_precision']:.4f} "
            f"| {agg['macro_ref_recall']:.4f} +- {agg['std_ref_recall']:.4f} |"
        ),
        "",
        f"Delta IoU refine-voronoi: {agg['micro_ref_iou'] - agg['micro_vor_iou']:.4f}",
        f"Delta IoU refine-partition: {agg['micro_ref_iou'] - agg['micro_part_iou']:.4f}",
        "",
        "See `refinement_visual_comparison.png` for boundary-focused qualitative results.",
        "",
        "_Generated by `run_feature_guided_refinement_experiment.py`_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_refiner(model: RefinementUNet, loader: DataLoader, optimizer, bce, dice, boundary, device, w_bce, w_dice, w_bnd):
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        p = model(x)
        loss = w_bce * bce(p, y) + w_dice * dice(p, y) + w_bnd * boundary(p, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def main():
    pa = argparse.ArgumentParser(description="Feature-guided UNet refinement experiment")
    pa.add_argument("--model_dir", default="SECOND/stage5_6_dynamic")
    pa.add_argument("--tokens_T1", default="SECOND/tokens_T1")
    pa.add_argument("--tokens_T2", default="SECOND/tokens_T2")
    pa.add_argument("--matches", default="SECOND/matches")
    pa.add_argument("--images_T1", default="SECOND/im1")
    pa.add_argument("--images_T2", default="SECOND/im2")
    pa.add_argument("--label1_dir", default="SECOND/label1")
    pa.add_argument("--label2_dir", default="SECOND/label2")
    pa.add_argument("--highres_T1", default="SECOND/highres_T1")
    pa.add_argument("--highres_T2", default="SECOND/highres_T2")

    pa.add_argument("--output_dir", default="feature_refine_branch")
    pa.add_argument("--val_split", type=float, default=0.1)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--device", default="cuda")

    pa.add_argument("--epochs", type=int, default=8)
    pa.add_argument("--batch_size", type=int, default=4)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--num_workers", type=int, default=2)

    pa.add_argument("--w_bce", type=float, default=1.0)
    pa.add_argument("--w_dice", type=float, default=1.0)
    pa.add_argument("--w_boundary", type=float, default=0.5)

    pa.add_argument("--threshold", type=float, default=0.5)
    pa.add_argument("--use_edge_map", action="store_true")

    pa.add_argument("--train_max_samples", type=int, default=None)
    pa.add_argument("--val_max_samples", type=int, default=None)
    pa.add_argument("--vis_n", type=int, default=10)

    pa.add_argument("--partition_chooser", choices=["area", "score"], default="area")

    pa.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    pa.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    pa.add_argument("--points_per_side", type=int, default=32)
    pa.add_argument("--pred_iou_thresh", type=float, default=0.75)
    pa.add_argument("--stability_thresh", type=float, default=0.85)
    pa.add_argument("--min_mask_area", type=int, default=256)

    args = pa.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "coarse_cache_256"
    cache_dir.mkdir(parents=True, exist_ok=True)

    t1_dir = Path(args.tokens_T1)
    t2_dir = Path(args.tokens_T2)
    match_dir = Path(args.matches)

    print("Loading token model ...")
    token_model = load_model(Path(args.model_dir), device)
    print(f"  Router v{token_model.cfg.router_version} | Experts: {token_model.cfg.moe_num_experts}")

    print("Building train/val splits ...")
    train_stems, val_stems = build_splits(match_dir, t1_dir, t2_dir, args.val_split, args.seed)

    if args.train_max_samples is not None:
        train_stems = train_stems[: args.train_max_samples]
    if args.val_max_samples is not None:
        val_stems = val_stems[: args.val_max_samples]

    print(f"  train={len(train_stems)} val={len(val_stems)}")

    print("Preparing coarse Voronoi cache ...")
    ensure_coarse_cache(train_stems + val_stems, cache_dir, token_model, t1_dir, t2_dir, match_dir, device)

    in_ch = 1 + 32 + 64 + (1 if args.use_edge_map else 0)
    args.input_channels = in_ch

    train_ds = RefinementDataset(
        stems=train_stems,
        coarse_cache_dir=cache_dir,
        highres_t1_dir=Path(args.highres_T1),
        highres_t2_dir=Path(args.highres_T2),
        img1_dir=Path(args.images_T1),
        img2_dir=Path(args.images_T2),
        label1_dir=Path(args.label1_dir),
        label2_dir=Path(args.label2_dir),
        use_edge=args.use_edge_map,
    )

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    refiner = RefinementUNet(in_ch).to(device)
    opt = torch.optim.Adam(refiner.parameters(), lr=args.lr)
    bce = nn.BCELoss()
    dice = DiceLoss()
    bnd = BoundaryLoss().to(device)

    print("Training refinement UNet ...")
    hist = []
    for ep in range(1, args.epochs + 1):
        loss = train_refiner(refiner, loader, opt, bce, dice, bnd, device, args.w_bce, args.w_dice, args.w_boundary)
        hist.append(loss)
        print(f"  epoch {ep:02d}/{args.epochs} | loss={loss:.4f}")

    torch.save(
        {
            "model_state": refiner.state_dict(),
            "args": vars(args),
            "train_loss": hist,
        },
        out_dir / "refiner_checkpoint.pt",
    )

    print("Loading SAM generator for partition baseline ...")
    amg = load_amg(args, device)

    print("Evaluating three decoders on validation ...")
    rows, vis = evaluate(val_stems, refiner, token_model, amg, args, device, cache_dir)

    agg = {}
    agg.update(aggregate(rows, "vor"))
    agg.update(aggregate(rows, "part"))
    agg.update(aggregate(rows, "ref"))

    print("\n" + "=" * 74)
    print("FEATURE-GUIDED REFINEMENT RESULTS")
    print("=" * 74)
    print(
        f"Voronoi   : F1={agg['micro_vor_f1']:.4f} IoU={agg['micro_vor_iou']:.4f} "
        f"P={agg['micro_vor_precision']:.4f} R={agg['micro_vor_recall']:.4f}"
    )
    print(
        f"Partition : F1={agg['micro_part_f1']:.4f} IoU={agg['micro_part_iou']:.4f} "
        f"P={agg['micro_part_precision']:.4f} R={agg['micro_part_recall']:.4f}"
    )
    print(
        f"Refined   : F1={agg['micro_ref_f1']:.4f} IoU={agg['micro_ref_iou']:.4f} "
        f"P={agg['micro_ref_precision']:.4f} R={agg['micro_ref_recall']:.4f}"
    )
    print(f"Delta IoU refine-voronoi  : {agg['micro_ref_iou'] - agg['micro_vor_iou']:.4f}")
    print(f"Delta IoU refine-partition: {agg['micro_ref_iou'] - agg['micro_part_iou']:.4f}")
    print("=" * 74)

    save_csv(rows, out_dir / "decoder_comparison_metrics.csv")
    save_visuals(vis, out_dir / "refinement_visual_comparison.png")

    (out_dir / "aggregate_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    save_report(out_dir / "feature_guided_refinement_report.md", args, agg, len(train_stems), len(val_stems))

    print(f"Saved: {out_dir / 'decoder_comparison_metrics.csv'}")
    print(f"Saved: {out_dir / 'aggregate_metrics.json'}")
    print(f"Saved: {out_dir / 'refinement_visual_comparison.png'}")
    print(f"Saved: {out_dir / 'feature_guided_refinement_report.md'}")


if __name__ == "__main__":
    main()
