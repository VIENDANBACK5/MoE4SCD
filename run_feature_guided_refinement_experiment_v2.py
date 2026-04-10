#!/usr/bin/env python3
"""Feature-guided refinement decoder v2 (boundary-constrained residual branch).

This experiment redesigns refinement to avoid degenerate all-change predictions:
- residual delta prediction from coarse map
- coarse-aware gating outside coarse regions
- boundary-focused supervision
"""

from __future__ import annotations

import argparse
import csv
import json
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rgb_to_class(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    d = ((flat[:, None, :] - _SECOND_COLOURS[None, :, :]) ** 2).sum(-1)
    return d.argmin(-1).reshape(rgb.shape[:2])


def get_gt_change(stem: str, label1_dir: Path, label2_dir: Path) -> np.ndarray:
    a = np.array(Image.open(label1_dir / f"{stem}.png").convert("RGB"))
    b = np.array(Image.open(label2_dir / f"{stem}.png").convert("RGB"))
    return _rgb_to_class(a) != _rgb_to_class(b)


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


def load_token_model(model_dir: Path, device: torch.device) -> TokenChangeReasonerMoE:
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
    t1_mask = (~pad) & (tid == 0)
    probs = torch.sigmoid(logits[t1_mask]).numpy()
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


def coarse_boundary_map(coarse256: np.ndarray) -> np.ndarray:
    u8 = np.clip(coarse256 * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, 50, 120)
    return (edges > 0).astype(np.float32)


def edge_map_from_images(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    d = cv2.absdiff(t1, t2)
    g = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    mag = mag / (mag.max() + 1e-6)
    return mag.astype(np.float32)


def build_input_channels(
    stem: str,
    coarse256: np.ndarray,
    highres_t1_dir: Path,
    highres_t2_dir: Path,
    img1_dir: Path,
    img2_dir: Path,
):
    h1 = torch.load(highres_t1_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
    h2 = torch.load(highres_t2_dir / f"{stem}.pt", map_location="cpu", weights_only=True)

    f256_a = h1[0].float().squeeze(0)  # [32,256,256]
    f128_a = h1[1].float()             # [1,64,128,128]
    f256_b = h2[0].float().squeeze(0)
    f128_b = h2[1].float()

    d256 = torch.abs(f256_a - f256_b)
    d128 = torch.abs(f128_a - f128_b)
    d128u = F.interpolate(d128, size=(LOW, LOW), mode="bilinear", align_corners=False).squeeze(0)

    t1 = np.array(Image.open(img1_dir / f"{stem}.png").convert("RGB"))
    t2 = np.array(Image.open(img2_dir / f"{stem}.png").convert("RGB"))

    edge = edge_map_from_images(t1, t2)
    edge = cv2.resize(edge, (LOW, LOW), interpolation=cv2.INTER_LINEAR)
    bnd = coarse_boundary_map(coarse256)

    coarse_ch = torch.from_numpy(coarse256.astype(np.float32)).unsqueeze(0)
    bnd_ch = torch.from_numpy(bnd.astype(np.float32)).unsqueeze(0)
    edge_ch = torch.from_numpy(edge.astype(np.float32)).unsqueeze(0)

    # [1 + 1 + 1 + 32 + 64, 256, 256] = [99,256,256]
    x = torch.cat([coarse_ch, bnd_ch, edge_ch, d256, d128u], dim=0)
    return x


def apply_train_augmentation(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random crop/flip/brightness shift at 256x256."""
    H = x.shape[1]
    W = x.shape[2]

    # Random crop then resize back.
    crop = random.choice([192, 208, 224, 240, 256])
    if crop < H:
        y0 = random.randint(0, H - crop)
        x0 = random.randint(0, W - crop)
        x = x[:, y0 : y0 + crop, x0 : x0 + crop]
        y = y[:, y0 : y0 + crop, x0 : x0 + crop]
        x = F.interpolate(x.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)

    # Random flips.
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])
        y = torch.flip(y, dims=[2])
    if random.random() < 0.5:
        x = torch.flip(x, dims=[1])
        y = torch.flip(y, dims=[1])

    # Brightness shift proxy on edge channel (channel index 2).
    edge = x[2:3]
    alpha = 1.0 + random.uniform(-0.2, 0.2)
    beta = random.uniform(-0.1, 0.1)
    edge = torch.clamp(edge * alpha + beta, 0.0, 1.0)
    x[2:3] = edge

    return x, y


class RefinementDatasetV2(Dataset):
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
        train_aug: bool,
    ):
        self.stems = list(stems)
        self.cache = coarse_cache_dir
        self.h1 = highres_t1_dir
        self.h2 = highres_t2_dir
        self.i1 = img1_dir
        self.i2 = img2_dir
        self.l1 = label1_dir
        self.l2 = label2_dir
        self.train_aug = train_aug

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        s = self.stems[idx]
        coarse = np.load(self.cache / f"{s}_coarse256.npy")
        x = build_input_channels(s, coarse, self.h1, self.h2, self.i1, self.i2)

        gt = get_gt_change(s, self.l1, self.l2).astype(np.float32)
        gt256 = cv2.resize(gt, (LOW, LOW), interpolation=cv2.INTER_NEAREST)
        y = torch.from_numpy(gt256).unsqueeze(0)

        if self.train_aug:
            x, y = apply_train_augmentation(x, y)

        return x, y, torch.from_numpy(coarse.astype(np.float32)).unsqueeze(0), s


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(x + h, inplace=True)


class ResidualRefinementUNet(nn.Module):
    """Projection + residual UNet predicting delta map."""

    def __init__(self, in_ch: int):
        super().__init__()
        # projection layer
        self.proj = nn.Conv2d(in_ch, 32, kernel_size=1)

        # encoder
        self.e1_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.e1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.e1_res = ResidualBlock(64)

        self.e2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.e2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.e2_res = ResidualBlock(128)

        self.bottleneck = nn.Conv2d(128, 256, 3, padding=1)

        # decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d1 = nn.Conv2d(128 + 128, 128, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d2 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.proj(x), inplace=True)

        e1 = F.relu(self.e1_1(x), inplace=True)
        e1 = F.relu(self.e1_2(e1), inplace=True)
        e1 = self.e1_res(e1)

        p1 = F.max_pool2d(e1, 2)

        e2 = F.relu(self.e2_1(p1), inplace=True)
        e2 = F.relu(self.e2_2(e2), inplace=True)
        e2 = self.e2_res(e2)

        p2 = F.max_pool2d(e2, 2)

        b = F.relu(self.bottleneck(p2), inplace=True)

        u1 = self.up1(b)
        u1 = F.relu(self.d1(torch.cat([u1, e2], dim=1)), inplace=True)

        u2 = self.up2(u1)
        u2 = F.relu(self.d2(torch.cat([u2, e1], dim=1)), inplace=True)

        delta = torch.sigmoid(self.out(u2))
        return delta


def apply_residual_refine(coarse: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Residual refinement + coarse-aware gating.

    coarse: [B,1,256,256]
    delta : [B,1,256,256] in [0,1]
    """
    refined = torch.clamp(coarse + 0.5 * delta, 0.0, 1.0)
    gated = coarse * refined + (1.0 - coarse) * 0.1 * refined
    return torch.clamp(gated, 0.0, 1.0)


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.reshape(pred.shape[0], -1)
        t = target.reshape(target.shape[0], -1)
        inter = (p * t).sum(dim=1)
        den = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
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
        if (i + 1) % 100 == 0:
            print(f"  coarse cache [{i + 1}/{len(stems)}] ...")


def train_one_epoch(
    model: ResidualRefinementUNet,
    loader: DataLoader,
    opt,
    bce: nn.Module,
    dice: nn.Module,
    bnd: nn.Module,
    device: torch.device,
):
    model.train()
    total = 0.0
    mean_pred = 0.0
    mean_coarse = 0.0
    n = 0

    for x, y, coarse, _ in loader:
        x = x.to(device)
        y = y.to(device)
        coarse = coarse.to(device)

        delta = model(x)
        refined = apply_residual_refine(coarse, delta)

        loss = 0.5 * bce(refined, y) + 0.3 * dice(refined, y) + 0.2 * bnd(refined, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += float(loss.item())
        mean_pred += float(refined.mean().item())
        mean_coarse += float(coarse.mean().item())
        n += 1

    return {
        "loss": total / max(n, 1),
        "mean_pred": mean_pred / max(n, 1),
        "mean_coarse": mean_coarse / max(n, 1),
    }


def evaluate(
    stems: Sequence[str],
    refine_model: ResidualRefinementUNet,
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
    visuals: List[Dict] = []

    refine_model.eval()

    with torch.no_grad():
        for i, s in enumerate(stems):
            p1 = img1_dir / f"{s}.png"
            p2 = img2_dir / f"{s}.png"
            if not (p1.exists() and p2.exists()):
                continue

            gt = get_gt_change(s, label1_dir, label2_dir)
            probs, cents, areas = token_probs_for_stem(s, token_model, t1_dir, t2_dir, match_dir, device)

            # Voronoi baseline
            vid = voronoi_map(cents)
            pred_vor = probs[vid]

            # SAM partition baseline
            t1_img = np.array(Image.open(p1).convert("RGB"))
            t2_img = np.array(Image.open(p2).convert("RGB"))
            masks = generate_masks(t1_img, amg)
            aligned_masks, aligned_scores, aligned_areas = align_masks(masks, cents, areas)
            part = sam_partition_map(aligned_masks, cents, aligned_scores, aligned_areas, chooser=args.partition_chooser)
            pred_part = probs[part]

            # Residual refinement from coarse Voronoi map
            coarse256 = np.load(coarse_cache_dir / f"{s}_coarse256.npy")
            x = build_input_channels(s, coarse256, highres_t1_dir, highres_t2_dir, img1_dir, img2_dir).unsqueeze(0).to(device)
            coarse_t = torch.from_numpy(coarse256.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            delta = refine_model(x)
            ref256 = apply_residual_refine(coarse_t, delta).squeeze(0).squeeze(0).cpu().numpy()
            pred_ref = cv2.resize(ref256, (IMG, IMG), interpolation=cv2.INTER_LINEAR)

            b_v = pred_vor > args.threshold
            b_p = pred_part > args.threshold
            b_r = pred_ref > args.threshold

            mv = compute_metrics(b_v, gt)
            mp = compute_metrics(b_p, gt)
            mr = compute_metrics(b_r, gt)

            row = {
                "stem": s,
                "vor_f1": mv["f1"], "vor_iou": mv["iou"], "vor_precision": mv["precision"], "vor_recall": mv["recall"],
                "vor_tp": mv["tp"], "vor_fp": mv["fp"], "vor_fn": mv["fn"], "vor_tn": mv["tn"],
                "part_f1": mp["f1"], "part_iou": mp["iou"], "part_precision": mp["precision"], "part_recall": mp["recall"],
                "part_tp": mp["tp"], "part_fp": mp["fp"], "part_fn": mp["fn"], "part_tn": mp["tn"],
                "ref_f1": mr["f1"], "ref_iou": mr["iou"], "ref_precision": mr["precision"], "ref_recall": mr["recall"],
                "ref_tp": mr["tp"], "ref_fp": mr["fp"], "ref_fn": mr["fn"], "ref_tn": mr["tn"],
                "mean_pred_refined": float(pred_ref.mean()),
                "mean_coarse": float(pred_vor.mean()),
                "delta_iou_ref_vs_vor": mr["iou"] - mv["iou"],
                "delta_iou_ref_vs_part": mr["iou"] - mp["iou"],
            }
            rows.append(row)

            if len(visuals) < args.vis_n:
                visuals.append(
                    {
                        "stem": s,
                        "t1": t1_img,
                        "t2": t2_img,
                        "coarse": pred_vor,
                        "refined": pred_ref,
                        "gt": gt,
                    }
                )

            if (i + 1) % 20 == 0:
                print(f"  eval [{i + 1}/{len(stems)}] ...")

    return rows, visuals


def draw_boundary_overlay(t1: np.ndarray, coarse: np.ndarray, refined: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = t1.copy()
    cb = cv2.Canny((coarse * 255).astype(np.uint8), 60, 120) > 0
    rb = cv2.Canny((refined * 255).astype(np.uint8), 60, 120) > 0
    gb = cv2.Canny((gt.astype(np.uint8) * 255), 60, 120) > 0

    # Green = GT boundary, Blue = coarse-only, Red = refined-only, Yellow = refined∩GT.
    out[gb] = [0, 255, 0]
    out[cb & ~rb] = [0, 0, 255]
    out[rb & ~gb] = [255, 0, 0]
    out[rb & gb] = [255, 255, 0]
    return out


def save_visuals(visuals: List[Dict], save_path: Path):
    if not visuals:
        return
    rows = len(visuals)
    fig, axes = plt.subplots(rows, 6, figsize=(22, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, v in enumerate(visuals):
        ov = draw_boundary_overlay(v["t1"], v["coarse"], v["refined"], v["gt"])
        imgs = [v["t1"], v["t2"], v["coarse"], v["refined"], v["gt"].astype(np.uint8), ov]
        titles = ["T1", "T2", "Coarse", "Refined", "GT", "Boundary overlay"]
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
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_report(path: Path, args, agg: Dict[str, float], train_n: int, val_n: int, pred_mean: float, coarse_mean: float, iou_gain: float):
    lines = [
        "# Feature-Guided Residual Refinement v2",
        "",
        "## Setup",
        "",
        f"- Model: `{args.model_dir}`",
        f"- Train samples: `{train_n}`",
        f"- Val samples: `{val_n}`",
        f"- Epochs: `{args.epochs}`",
        f"- LR: `{args.lr}`",
        f"- Weight decay: `{args.weight_decay}`",
        f"- Threshold: `{args.threshold}`",
        "",
        "## Micro Metrics",
        "",
        "| Decoder | F1 | IoU | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
        f"| Voronoi | {agg['micro_vor_f1']:.4f} | {agg['micro_vor_iou']:.4f} | {agg['micro_vor_precision']:.4f} | {agg['micro_vor_recall']:.4f} |",
        f"| SAM partition | {agg['micro_part_f1']:.4f} | {agg['micro_part_iou']:.4f} | {agg['micro_part_precision']:.4f} | {agg['micro_part_recall']:.4f} |",
        f"| Residual refine v2 | {agg['micro_ref_f1']:.4f} | {agg['micro_ref_iou']:.4f} | {agg['micro_ref_precision']:.4f} | {agg['micro_ref_recall']:.4f} |",
        "",
        "## Diagnostics",
        "",
        f"- mean(prediction): `{pred_mean:.4f}`",
        f"- mean(coarse_change_map): `{coarse_mean:.4f}`",
        f"- IoU improvement vs coarse (Voronoi): `{iou_gain:.4f}`",
        "",
        "Check: mean(prediction) should not saturate near 1.0.",
        "",
        "See `refinement_v2_visual_comparison.png` for qualitative boundary alignment analysis.",
        "",
        "_Generated by `run_feature_guided_refinement_experiment_v2.py`_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    pa = argparse.ArgumentParser(description="Feature-guided residual refinement decoder v2")

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

    pa.add_argument("--output_dir", default="feature_refine_branch_v2")
    pa.add_argument("--val_split", type=float, default=0.1)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--device", default="cuda")

    pa.add_argument("--epochs", type=int, default=12)
    pa.add_argument("--batch_size", type=int, default=4)
    pa.add_argument("--num_workers", type=int, default=2)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--weight_decay", type=float, default=1e-4)

    pa.add_argument("--threshold", type=float, default=0.5)
    pa.add_argument("--partition_chooser", choices=["area", "score"], default="area")

    pa.add_argument("--train_max_samples", type=int, default=None)
    pa.add_argument("--val_max_samples", type=int, default=None)
    pa.add_argument("--vis_n", type=int, default=10)

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
    token_model = load_token_model(Path(args.model_dir), device)
    print(f"  Router v{token_model.cfg.router_version} | Experts: {token_model.cfg.moe_num_experts}")

    print("Building split ...")
    train_stems, val_stems = build_splits(match_dir, t1_dir, t2_dir, args.val_split, args.seed)
    if args.train_max_samples is not None:
        train_stems = train_stems[: args.train_max_samples]
    if args.val_max_samples is not None:
        val_stems = val_stems[: args.val_max_samples]
    print(f"  train={len(train_stems)} val={len(val_stems)}")

    print("Preparing coarse cache ...")
    ensure_coarse_cache(train_stems + val_stems, cache_dir, token_model, t1_dir, t2_dir, match_dir, device)

    in_ch = 99
    refiner = ResidualRefinementUNet(in_ch).to(device)

    train_ds = RefinementDatasetV2(
        stems=train_stems,
        coarse_cache_dir=cache_dir,
        highres_t1_dir=Path(args.highres_T1),
        highres_t2_dir=Path(args.highres_T2),
        img1_dir=Path(args.images_T1),
        img2_dir=Path(args.images_T2),
        label1_dir=Path(args.label1_dir),
        label2_dir=Path(args.label2_dir),
        train_aug=True,
    )

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCELoss()
    dice = DiceLoss()
    bnd = BoundaryLoss().to(device)

    print("Training residual refinement v2 ...")
    history = []
    for ep in range(1, args.epochs + 1):
        st = train_one_epoch(refiner, loader, opt, bce, dice, bnd, device)
        history.append(st)
        print(
            f"  epoch {ep:02d}/{args.epochs} | loss={st['loss']:.4f} "
            f"| mean(pred)={st['mean_pred']:.4f} mean(coarse)={st['mean_coarse']:.4f}"
        )

    torch.save(
        {
            "model_state": refiner.state_dict(),
            "args": vars(args),
            "train_history": history,
        },
        out_dir / "refiner_v2_checkpoint.pt",
    )

    print("Loading SAM generator ...")
    amg = load_amg(args, device)

    print("Evaluating Voronoi vs partition vs residual refine ...")
    rows, visuals = evaluate(val_stems, refiner, token_model, amg, args, device, cache_dir)

    agg = {}
    agg.update(aggregate(rows, "vor"))
    agg.update(aggregate(rows, "part"))
    agg.update(aggregate(rows, "ref"))

    pred_mean = float(np.mean([r["mean_pred_refined"] for r in rows])) if rows else 0.0
    coarse_mean = float(np.mean([r["mean_coarse"] for r in rows])) if rows else 0.0
    iou_gain = agg["micro_ref_iou"] - agg["micro_vor_iou"]

    print("\n" + "=" * 74)
    print("FEATURE-GUIDED RESIDUAL REFINEMENT V2")
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
        f"Refine v2 : F1={agg['micro_ref_f1']:.4f} IoU={agg['micro_ref_iou']:.4f} "
        f"P={agg['micro_ref_precision']:.4f} R={agg['micro_ref_recall']:.4f}"
    )
    print(f"mean(prediction)={pred_mean:.4f} | mean(coarse)={coarse_mean:.4f}")
    print(f"IoU improvement vs coarse={iou_gain:.4f}")
    print("=" * 74)

    save_csv(rows, out_dir / "decoder_comparison_metrics_v2.csv")
    save_visuals(visuals, out_dir / "refinement_v2_visual_comparison.png")
    (out_dir / "aggregate_metrics_v2.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out_dir / "train_history_v2.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_report(
        out_dir / "feature_guided_refinement_v2_report.md",
        args,
        agg,
        len(train_stems),
        len(val_stems),
        pred_mean,
        coarse_mean,
        iou_gain,
    )

    print(f"Saved: {out_dir / 'decoder_comparison_metrics_v2.csv'}")
    print(f"Saved: {out_dir / 'aggregate_metrics_v2.json'}")
    print(f"Saved: {out_dir / 'train_history_v2.json'}")
    print(f"Saved: {out_dir / 'refinement_v2_visual_comparison.png'}")
    print(f"Saved: {out_dir / 'feature_guided_refinement_v2_report.md'}")


if __name__ == "__main__":
    main()
