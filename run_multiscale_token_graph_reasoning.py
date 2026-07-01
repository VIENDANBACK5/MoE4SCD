#!/usr/bin/env python3
"""Multi-Scale Token Graph Reasoning experiment.

Implements:
1) Coarse + fine token extraction via SAM (different mask densities)
2) Token features via masked pooling from backbone embeddings
3) Hierarchical graph (coarse-coarse, fine-fine, cross-scale parent edges)
4) Baseline coarse-graph vs multi-scale graph training
5) Pixel reconstruction and metric comparison

This is a standalone research branch compatible with the existing phase1 pipeline.
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
SAM2_REPO = ROOT / "sam2"
if SAM2_REPO.is_dir():
    sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

IMG = 512
EMB = 64
SCALE = IMG // EMB

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
class TokenSet:
    feats: torch.Tensor       # [N, D]
    cents: torch.Tensor       # [N, 2] in [0,1]
    areas: torch.Tensor       # [N]
    masks: torch.Tensor       # [N, H, W] bool
    labels: torch.Tensor      # [N] 0/1


@dataclass
class PairSample:
    stem: str
    coarse: TokenSet
    fine: TokenSet


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


def load_gt_change(stem: str, label1_dir: Path, label2_dir: Path) -> np.ndarray:
    l1 = np.array(Image.open(label1_dir / f"{stem}.png").convert("RGB"))
    l2 = np.array(Image.open(label2_dir / f"{stem}.png").convert("RGB"))
    return (_rgb_to_class(l1) != _rgb_to_class(l2)).astype(np.uint8)


def build_stems(tokens_t1_dir: Path, tokens_t2_dir: Path, match_dir: Path, val_split: float, seed: int):
    stems = [
        mp.stem.replace("_matches", "")
        for mp in sorted(match_dir.glob("*_matches.pt"))
        if (tokens_t1_dir / f"{mp.stem.replace('_matches', '')}.pt").exists()
        and (tokens_t2_dir / f"{mp.stem.replace('_matches', '')}.pt").exists()
    ]
    n_val = max(1, int(len(stems) * val_split))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(stems), generator=g).tolist()
    val = [stems[i] for i in perm[len(stems) - n_val :]]
    val_set = set(val)
    train = [s for s in stems if s not in val_set]
    return train, val


def load_amg(config: str, ckpt: str, points_per_side: int, pred_iou_thresh: float, stability_thresh: float, min_mask_area: int, device: str):
    ck = Path(ckpt)
    if not ck.is_absolute():
        ck = SAM2_REPO / ck
    model = build_sam2(config, str(ck), device=device)
    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_thresh,
        min_mask_region_area=min_mask_area,
        output_mode="binary_mask",
    )


def downsample_mask(mask: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(t, kernel_size=SCALE, stride=SCALE)
    return pooled.squeeze() > 0.0


def extract_tokens_from_masks(
    masks: List[np.ndarray],
    emb_t1: torch.Tensor,  # [1,256,64,64]
    emb_t2: torch.Tensor,  # [1,256,64,64]
    gt_change: np.ndarray, # [H,W] bool/uint8
) -> TokenSet:
    emb1 = emb_t1.squeeze(0)
    emb2 = emb_t2.squeeze(0)

    feats = []
    cents = []
    areas = []
    out_masks = []
    labels = []

    for m in masks:
        if m.dtype != np.bool_:
            m = m.astype(bool)
        if m.sum() == 0:
            continue

        ms = downsample_mask(m)
        if int(ms.sum().item()) == 0:
            continue

        f1 = emb1[:, ms].mean(dim=1)
        f2 = emb2[:, ms].mean(dim=1)
        # Change embedding: absolute feature difference
        feat = torch.abs(f2 - f1)

        ys, xs = np.where(m)
        cx = float(xs.mean()) / (IMG - 1)
        cy = float(ys.mean()) / (IMG - 1)
        area = float(m.sum()) / (IMG * IMG)

        chg_ratio = float(gt_change[m].mean())
        lbl = 1.0 if chg_ratio >= 0.5 else 0.0

        feats.append(feat)
        cents.append(torch.tensor([cx, cy], dtype=torch.float32))
        areas.append(area)
        out_masks.append(torch.from_numpy(m))
        labels.append(lbl)

    if not feats:
        # fallback single global token
        m = np.ones((IMG, IMG), dtype=bool)
        f1 = emb1.mean(dim=(1, 2))
        f2 = emb2.mean(dim=(1, 2))
        feat = torch.abs(f2 - f1)
        feats = [feat]
        cents = [torch.tensor([0.5, 0.5], dtype=torch.float32)]
        areas = [1.0]
        out_masks = [torch.from_numpy(m)]
        labels = [float(gt_change.mean() >= 0.5)]

    return TokenSet(
        feats=torch.stack(feats).float(),
        cents=torch.stack(cents).float(),
        areas=torch.tensor(areas, dtype=torch.float32),
        masks=torch.stack(out_masks).bool(),
        labels=torch.tensor(labels, dtype=torch.float32),
    )


def extract_single_time_tokens(
    masks: List[np.ndarray],
    emb: torch.Tensor,       # [1,256,64,64]
    gt_change: np.ndarray,   # [H,W] bool/uint8
) -> TokenSet:
    """Extract per-timestamp token embeddings for contrastive learning."""
    emb_s = emb.squeeze(0)

    feats = []
    cents = []
    areas = []
    out_masks = []
    labels = []

    for m in masks:
        if m.dtype != np.bool_:
            m = m.astype(bool)
        if m.sum() == 0:
            continue

        ms = downsample_mask(m)
        if int(ms.sum().item()) == 0:
            continue

        feat = emb_s[:, ms].mean(dim=1)
        ys, xs = np.where(m)
        cx = float(xs.mean()) / (IMG - 1)
        cy = float(ys.mean()) / (IMG - 1)
        area = float(m.sum()) / (IMG * IMG)

        # Token is "changed" if most of its area overlaps change map
        chg_ratio = float(gt_change[m].mean())
        lbl = 1.0 if chg_ratio >= 0.5 else 0.0

        feats.append(feat)
        cents.append(torch.tensor([cx, cy], dtype=torch.float32))
        areas.append(area)
        out_masks.append(torch.from_numpy(m))
        labels.append(lbl)

    if not feats:
        m = np.ones((IMG, IMG), dtype=bool)
        feat = emb_s.mean(dim=(1, 2))
        feats = [feat]
        cents = [torch.tensor([0.5, 0.5], dtype=torch.float32)]
        areas = [1.0]
        out_masks = [torch.from_numpy(m)]
        labels = [float(gt_change.mean() >= 0.5)]

    return TokenSet(
        feats=torch.stack(feats).float(),
        cents=torch.stack(cents).float(),
        areas=torch.tensor(areas, dtype=torch.float32),
        masks=torch.stack(out_masks).bool(),
        labels=torch.tensor(labels, dtype=torch.float32),
    )


def sam_masks_for_scale(amg: SAM2AutomaticMaskGenerator, img: np.ndarray) -> List[np.ndarray]:
    raw = amg.generate(img)
    out = []
    for m in raw:
        seg = m["segmentation"]
        if seg.dtype != np.bool_:
            seg = seg.astype(bool)
        out.append(seg)
    return out


def prepare_cached_samples(
    stems: Sequence[str],
    cache_dir: Path,
    img1_dir: Path,
    img2_dir: Path,
    emb1_dir: Path,
    emb2_dir: Path,
    label1_dir: Path,
    label2_dir: Path,
    amg_coarse: SAM2AutomaticMaskGenerator,
    amg_fine: SAM2AutomaticMaskGenerator,
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(stems):
        outp = cache_dir / f"{s}.pt"
        if outp.exists():
            continue

        t1 = np.array(Image.open(img1_dir / f"{s}.png").convert("RGB"))
        t2 = np.array(Image.open(img2_dir / f"{s}.png").convert("RGB"))

        emb1 = torch.load(emb1_dir / f"{s}.pt", map_location="cpu", weights_only=True).float()
        emb2 = torch.load(emb2_dir / f"{s}.pt", map_location="cpu", weights_only=True).float()
        gt = load_gt_change(s, label1_dir, label2_dir)

        cmasks_t1 = sam_masks_for_scale(amg_coarse, t1)
        fmasks_t1 = sam_masks_for_scale(amg_fine, t1)
        cmasks_t2 = sam_masks_for_scale(amg_coarse, t2)
        fmasks_t2 = sam_masks_for_scale(amg_fine, t2)

        coarse = extract_tokens_from_masks(cmasks_t1, emb1, emb2, gt)
        fine = extract_tokens_from_masks(fmasks_t1, emb1, emb2, gt)

        # Additional timestamp-specific tokens for contrastive alignment
        coarse_t1 = extract_single_time_tokens(cmasks_t1, emb1, gt)
        coarse_t2 = extract_single_time_tokens(cmasks_t2, emb2, gt)
        fine_t1 = extract_single_time_tokens(fmasks_t1, emb1, gt)
        fine_t2 = extract_single_time_tokens(fmasks_t2, emb2, gt)

        pack = {
            "stem": s,
            "coarse": {
                "feats": coarse.feats,
                "cents": coarse.cents,
                "areas": coarse.areas,
                "masks": coarse.masks,
                "labels": coarse.labels,
            },
            "fine": {
                "feats": fine.feats,
                "cents": fine.cents,
                "areas": fine.areas,
                "masks": fine.masks,
                "labels": fine.labels,
            },
            "contrastive": {
                "coarse": {
                    "t1_feats": coarse_t1.feats,
                    "t1_cents": coarse_t1.cents,
                    "t1_masks": coarse_t1.masks,
                    "t1_labels": coarse_t1.labels,
                    "t2_feats": coarse_t2.feats,
                    "t2_cents": coarse_t2.cents,
                    "t2_masks": coarse_t2.masks,
                    "t2_labels": coarse_t2.labels,
                },
                "fine": {
                    "t1_feats": fine_t1.feats,
                    "t1_cents": fine_t1.cents,
                    "t1_masks": fine_t1.masks,
                    "t1_labels": fine_t1.labels,
                    "t2_feats": fine_t2.feats,
                    "t2_cents": fine_t2.cents,
                    "t2_masks": fine_t2.masks,
                    "t2_labels": fine_t2.labels,
                },
            },
        }
        torch.save(pack, outp)

        if (i + 1) % 50 == 0:
            print(f"  cache [{i + 1}/{len(stems)}] ...")


class CachedPairDataset(Dataset):
    def __init__(self, stems: Sequence[str], cache_dir: Path):
        self.stems = list(stems)
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        s = self.stems[idx]
        d = torch.load(self.cache_dir / f"{s}.pt", map_location="cpu", weights_only=False)
        return d


def knn_graph(cents: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    N = cents.shape[0]
    if N <= 1:
        nbr_idx = torch.zeros((N, 1), dtype=torch.long)
        nbr_w = torch.ones((N, 1), dtype=torch.float32)
        return nbr_idx, nbr_w

    k = min(k, N - 1)
    dist = torch.cdist(cents, cents, p=2)
    dist.fill_diagonal_(1e9)
    d, idx = dist.topk(k, dim=1, largest=False)  # [N,k]

    inv = 1.0 / d.clamp(min=1e-4)
    w = inv / inv.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return idx.long(), w.float()


def cross_scale_parent(fine_masks: torch.Tensor, coarse_masks: torch.Tensor, fine_cents: torch.Tensor, coarse_cents: torch.Tensor) -> torch.Tensor:
    """Return parent coarse index per fine token.

    If overlap(fine, coarse) / area(fine) > 0.5, assign that coarse.
    Else fallback to nearest coarse centroid.
    """
    Nf = fine_masks.shape[0]
    Nc = coarse_masks.shape[0]
    parent = torch.zeros(Nf, dtype=torch.long)

    if Nc == 1:
        parent[:] = 0
        return parent

    coarse_flat = coarse_masks.view(Nc, -1).float()
    fine_flat = fine_masks.view(Nf, -1).float()

    inter = fine_flat @ coarse_flat.T  # [Nf, Nc]
    fine_area = fine_flat.sum(dim=1, keepdim=True).clamp(min=1.0)
    ratio = inter / fine_area

    has = (ratio.max(dim=1).values > 0.5)
    best = ratio.argmax(dim=1)

    # nearest centroid fallback
    d = torch.cdist(fine_cents, coarse_cents, p=2)
    near = d.argmin(dim=1)

    parent = torch.where(has, best, near)
    return parent.long()


class GraphSAGELayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.w_self = nn.Linear(dim, dim, bias=False)
        self.w_nei = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, nbr_idx: torch.Tensor, nbr_w: torch.Tensor) -> torch.Tensor:
        # h:[N,D], nbr_idx:[N,k], nbr_w:[N,k]
        N, D = h.shape
        k = nbr_idx.shape[1]
        nei = h[nbr_idx.reshape(-1)].reshape(N, k, D)
        agg = (nbr_w.unsqueeze(-1) * nei).sum(dim=1)
        out = F.gelu(self.w_self(h) + self.w_nei(agg))
        out = self.norm(self.drop(out))
        return out


class BaselineCoarseGraphModel(nn.Module):
    def __init__(self, in_dim: int = 256, hidden: int = 256, k: int = 6, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(in_dim, hidden)
        self.g1 = GraphSAGELayer(hidden, dropout)
        self.g2 = GraphSAGELayer(hidden, dropout)
        self.cls = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, feats: torch.Tensor, cents: torch.Tensor) -> torch.Tensor:
        # feats:[Nc,D]
        h = F.gelu(self.enc(feats))
        idx, w = knn_graph(cents, self.k)
        idx = idx.to(h.device)
        w = w.to(h.device)
        h = h + self.g1(h, idx, w)
        h = h + self.g2(h, idx, w)
        return torch.sigmoid(self.cls(h)).squeeze(-1)  # [Nc]


class MultiScaleGraphModel(nn.Module):
    def __init__(self, in_dim: int = 256, hidden: int = 256, k: int = 6, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.enc_c = nn.Linear(in_dim, hidden)
        self.enc_f = nn.Linear(in_dim, hidden)

        self.g_f = GraphSAGELayer(hidden, dropout)
        self.g_c = GraphSAGELayer(hidden, dropout)

        # cross-scale transforms
        self.f2c = nn.Linear(hidden, hidden)
        self.c2f = nn.Linear(hidden, hidden)

        self.cls_c = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.cls_f = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

        # Projection head for token contrastive embeddings
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 128),
        )

    def forward(
        self,
        c_feats: torch.Tensor,
        c_cents: torch.Tensor,
        c_masks: torch.Tensor,
        f_feats: torch.Tensor,
        f_cents: torch.Tensor,
        f_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hc = F.gelu(self.enc_c(c_feats))
        hf = F.gelu(self.enc_f(f_feats))

        # Layer 1 fine graph
        f_idx, f_w = knn_graph(f_cents, self.k)
        hf = hf + self.g_f(hf, f_idx.to(hf.device), f_w.to(hf.device))

        # Layer 2 coarse graph
        c_idx, c_w = knn_graph(c_cents, self.k)
        hc = hc + self.g_c(hc, c_idx.to(hc.device), c_w.to(hc.device))

        # Layer 3 fine -> coarse (children aggregate to parent coarse)
        parent = cross_scale_parent(f_masks, c_masks, f_cents, c_cents).to(hc.device)  # [Nf]
        Nc = hc.shape[0]
        agg_c = torch.zeros_like(hc)
        cnt_c = torch.zeros((Nc, 1), device=hc.device)
        agg_c.index_add_(0, parent, self.f2c(hf))
        cnt_c.index_add_(0, parent, torch.ones((hf.shape[0], 1), device=hc.device))
        agg_c = agg_c / cnt_c.clamp(min=1.0)
        hc = hc + agg_c

        # Layer 4 coarse -> fine (parent coarse message to each fine)
        hf = hf + self.c2f(hc[parent])

        pc = torch.sigmoid(self.cls_c(hc)).squeeze(-1)
        pf = torch.sigmoid(self.cls_f(hf)).squeeze(-1)
        return pc, pf

    def project_tokens(self, feats: torch.Tensor) -> torch.Tensor:
        """Project 256-d token embeddings to 128-d contrastive space."""
        z = self.proj(feats)
        return F.normalize(z, dim=-1)


def _flatten_masks_bool(masks: torch.Tensor) -> torch.Tensor:
    if masks.dtype != torch.bool:
        masks = masks > 0
    return masks.view(masks.shape[0], -1)


def build_contrastive_pairs(
    t1_masks: torch.Tensor,
    t2_masks: torch.Tensor,
    t1_labels: torch.Tensor,
    t2_labels: torch.Tensor,
    t1_cents: torch.Tensor,
    t2_cents: torch.Tensor,
    iou_thresh: float,
    spatial_neg_dist: float,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Returns (positives, changed_negatives, far_negatives) using IoU matching.

    positives: matched IoU>thresh and both labels are no-change
    changed_negatives: matched IoU>thresh but at least one side is changed
    far_negatives: spatially distant pairs (centroid distance > threshold)
    """
    m1 = _flatten_masks_bool(t1_masks).float()
    m2 = _flatten_masks_bool(t2_masks).float()
    inter = m1 @ m2.T
    area1 = m1.sum(dim=1, keepdim=True)
    area2 = m2.sum(dim=1, keepdim=True).T
    union = (area1 + area2 - inter).clamp(min=1.0)
    iou = inter / union

    best_iou, best_j = iou.max(dim=1)
    matched = best_iou > iou_thresh

    positives: List[Tuple[int, int]] = []
    changed_negs: List[Tuple[int, int]] = []
    for i in torch.where(matched)[0].tolist():
        j = int(best_j[i].item())
        unchanged = (float(t1_labels[i].item()) < 0.5) and (float(t2_labels[j].item()) < 0.5)
        if unchanged:
            positives.append((i, j))
        else:
            changed_negs.append((i, j))

    d = torch.cdist(t1_cents.float(), t2_cents.float(), p=2)
    far = torch.where(d > spatial_neg_dist)
    far_negs = list(zip(far[0].tolist(), far[1].tolist()))
    return positives, changed_negs, far_negs


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    positives: List[Tuple[int, int]],
    changed_negs: List[Tuple[int, int]],
    far_negs: List[Tuple[int, int]],
    tau: float,
) -> torch.Tensor:
    """Cross-time NT-Xent: anchors from z1, positives in z2, negatives in denominator."""
    if len(positives) == 0:
        return z1.new_zeros(())

    sim = z1 @ z2.T  # cosine because z vectors are normalized
    losses = []
    all_cols = torch.arange(z2.shape[0], device=z1.device)

    # Fast lookup for optional hard-negative emphasis
    changed_lookup = {}
    for i, j in changed_negs:
        changed_lookup.setdefault(i, set()).add(j)
    far_lookup = {}
    for i, j in far_negs:
        far_lookup.setdefault(i, set()).add(j)

    for i, j in positives:
        num = torch.exp(sim[i, j] / tau)

        # Denominator includes all candidates from batch z2 (SimCLR style).
        den = torch.exp(sim[i, all_cols] / tau).sum()

        # Optional hard-negative upweight by re-adding changed/far negatives once.
        hard_cols = set()
        if i in changed_lookup:
            hard_cols.update(changed_lookup[i])
        if i in far_lookup:
            hard_cols.update(far_lookup[i])
        if hard_cols:
            hard_idx = torch.tensor(sorted(hard_cols), dtype=torch.long, device=z1.device)
            den = den + torch.exp(sim[i, hard_idx] / tau).sum()

        losses.append(-torch.log(num / den.clamp(min=1e-12)))

    return torch.stack(losses).mean()


def compute_batch_contrastive_loss(
    model: MultiScaleGraphModel,
    batch: Sequence[Dict],
    device: torch.device,
    iou_thresh: float,
    spatial_neg_dist: float,
    tau: float,
) -> Tuple[torch.Tensor, int, int]:
    """Build token pools across the batch and compute NT-Xent on matched pairs."""
    t1_chunks: List[torch.Tensor] = []
    t2_chunks: List[torch.Tensor] = []
    positives_global: List[Tuple[int, int]] = []
    changed_global: List[Tuple[int, int]] = []
    far_global: List[Tuple[int, int]] = []
    off1 = 0
    off2 = 0

    for item in batch:
        if "contrastive" not in item:
            continue

        for scale in ["coarse", "fine"]:
            sc = item["contrastive"][scale]
            t1f = sc["t1_feats"].to(device)
            t2f = sc["t2_feats"].to(device)
            t1_chunks.append(t1f)
            t2_chunks.append(t2f)

            pos, cneg, fneg = build_contrastive_pairs(
                t1_masks=sc["t1_masks"],
                t2_masks=sc["t2_masks"],
                t1_labels=sc["t1_labels"],
                t2_labels=sc["t2_labels"],
                t1_cents=sc["t1_cents"],
                t2_cents=sc["t2_cents"],
                iou_thresh=iou_thresh,
                spatial_neg_dist=spatial_neg_dist,
            )

            positives_global.extend([(i + off1, j + off2) for i, j in pos])
            changed_global.extend([(i + off1, j + off2) for i, j in cneg])
            far_global.extend([(i + off1, j + off2) for i, j in fneg])

            off1 += t1f.shape[0]
            off2 += t2f.shape[0]

    if not t1_chunks or not t2_chunks or not positives_global:
        return torch.zeros((), device=device), 0, 0

    z1 = model.project_tokens(torch.cat(t1_chunks, dim=0))
    z2 = model.project_tokens(torch.cat(t2_chunks, dim=0))
    loss = nt_xent_loss(z1, z2, positives_global, changed_global, far_global, tau)
    return loss, len(positives_global), len(changed_global)


def coarse_projection(coarse_masks: torch.Tensor, coarse_probs: torch.Tensor) -> np.ndarray:
    out = np.zeros((IMG, IMG), dtype=np.float32)
    for i in range(coarse_masks.shape[0]):
        m = coarse_masks[i].numpy()
        p = float(coarse_probs[i])
        out[m] = np.maximum(out[m], p)
    return out


def multiscale_projection(coarse_masks: torch.Tensor, coarse_probs: torch.Tensor, fine_masks: torch.Tensor, fine_probs: torch.Tensor) -> np.ndarray:
    out = np.zeros((IMG, IMG), dtype=np.float32)
    # coarse first then fine (fine can overwrite via max)
    for i in range(coarse_masks.shape[0]):
        m = coarse_masks[i].numpy()
        p = float(coarse_probs[i])
        out[m] = np.maximum(out[m], p)
    for i in range(fine_masks.shape[0]):
        m = fine_masks[i].numpy()
        p = float(fine_probs[i])
        out[m] = np.maximum(out[m], p)
    return out


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pb = pred > 0.5
    gb = gt.astype(bool)
    tp = int((pb & gb).sum())
    fp = int((pb & ~gb).sum())
    fn = int((~pb & gb).sum())
    tn = int((~pb & ~gb).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1, "iou": iou}


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

    out[f"micro_{prefix}_precision"] = p
    out[f"micro_{prefix}_recall"] = r
    out[f"micro_{prefix}_f1"] = f1
    out[f"micro_{prefix}_iou"] = iou
    return out


def train_epoch_baseline(model, loader, opt, device):
    model.train()
    bce = nn.BCELoss()
    total = 0.0
    n = 0
    for batch in loader:
        loss_sum = 0.0
        for item in batch:
            c = item["coarse"]
            feats = c["feats"].to(device)
            cents = c["cents"].to(device)
            gt = c["labels"].to(device)

            pred = model(feats, cents)
            loss = bce(pred, gt)
            loss_sum = loss_sum + loss

        loss_mean = loss_sum / max(len(batch), 1)
        opt.zero_grad(set_to_none=True)
        loss_mean.backward()
        opt.step()

        total += float(loss_mean.item())
        n += 1
    return total / max(n, 1)


def train_epoch_multiscale(
    model,
    loader,
    opt,
    device,
    fine_weight: float,
    contrastive_weight: float,
    contrastive_tau: float,
    pair_iou_thresh: float,
    spatial_neg_dist: float,
):
    model.train()
    bce = nn.BCELoss()
    total = 0.0
    total_bce = 0.0
    total_ctr = 0.0
    total_pos = 0
    total_changed_neg = 0
    n = 0
    for batch in loader:
        loss_sum = 0.0
        for item in batch:
            c = item["coarse"]
            f = item["fine"]

            c_feats = c["feats"].to(device)
            c_cents = c["cents"].to(device)
            c_masks = c["masks"].to(device)
            c_gt = c["labels"].to(device)

            f_feats = f["feats"].to(device)
            f_cents = f["cents"].to(device)
            f_masks = f["masks"].to(device)
            f_gt = f["labels"].to(device)

            pc, pf = model(c_feats, c_cents, c_masks, f_feats, f_cents, f_masks)
            lc = bce(pc, c_gt)
            lf = bce(pf, f_gt)
            loss = lc + fine_weight * lf
            loss_sum = loss_sum + loss

        bce_mean = loss_sum / max(len(batch), 1)
        ctr_loss, n_pos, n_cneg = compute_batch_contrastive_loss(
            model=model,
            batch=batch,
            device=device,
            iou_thresh=pair_iou_thresh,
            spatial_neg_dist=spatial_neg_dist,
            tau=contrastive_tau,
        )
        loss_mean = bce_mean + contrastive_weight * ctr_loss

        opt.zero_grad(set_to_none=True)
        loss_mean.backward()
        opt.step()

        total += float(loss_mean.item())
        total_bce += float(bce_mean.item())
        total_ctr += float(ctr_loss.item())
        total_pos += n_pos
        total_changed_neg += n_cneg
        n += 1
    return {
        "loss_total": total / max(n, 1),
        "loss_bce": total_bce / max(n, 1),
        "loss_contrastive": total_ctr / max(n, 1),
        "avg_positive_pairs": total_pos / max(n, 1),
        "avg_changed_neg_pairs": total_changed_neg / max(n, 1),
    }


def evaluate_models(base_model, ms_model, dataset: CachedPairDataset, device: torch.device, label1_dir: Path, label2_dir: Path, vis_n: int, output_dir: Path):
    base_model.eval()
    ms_model.eval()

    rows: List[Dict] = []
    vis = []

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            stem = item["stem"]
            c = item["coarse"]
            f = item["fine"]

            gt = load_gt_change(stem, label1_dir, label2_dir)

            # baseline coarse graph
            b_pred = base_model(c["feats"].to(device), c["cents"].to(device)).cpu()
            b_map = coarse_projection(c["masks"], b_pred)

            # multiscale
            pc, pf = ms_model(
                c["feats"].to(device), c["cents"].to(device), c["masks"].to(device),
                f["feats"].to(device), f["cents"].to(device), f["masks"].to(device),
            )
            m_map = multiscale_projection(c["masks"], pc.cpu(), f["masks"], pf.cpu())

            mb = compute_metrics(b_map, gt)
            mm = compute_metrics(m_map, gt)

            row = {
                "stem": stem,
                "baseline_f1": mb["f1"], "baseline_iou": mb["iou"], "baseline_precision": mb["precision"], "baseline_recall": mb["recall"],
                "baseline_tp": mb["tp"], "baseline_fp": mb["fp"], "baseline_fn": mb["fn"], "baseline_tn": mb["tn"],
                "multiscale_f1": mm["f1"], "multiscale_iou": mm["iou"], "multiscale_precision": mm["precision"], "multiscale_recall": mm["recall"],
                "multiscale_tp": mm["tp"], "multiscale_fp": mm["fp"], "multiscale_fn": mm["fn"], "multiscale_tn": mm["tn"],
                "delta_iou": mm["iou"] - mb["iou"],
            }
            rows.append(row)

            if len(vis) < vis_n:
                vis.append((stem, b_map, m_map, gt))

    # visuals
    if vis:
        fig, axes = plt.subplots(len(vis), 3, figsize=(14, 4 * len(vis)))
        if len(vis) == 1:
            axes = np.expand_dims(axes, 0)
        for i, (stem, bmap, mmap, gt) in enumerate(vis):
            axes[i, 0].imshow(bmap, cmap="jet", vmin=0, vmax=1)
            axes[i, 0].set_title(f"{stem} baseline")
            axes[i, 1].imshow(mmap, cmap="jet", vmin=0, vmax=1)
            axes[i, 1].set_title(f"{stem} multiscale")
            axes[i, 2].imshow(gt, cmap="gray")
            axes[i, 2].set_title("GT")
            for j in range(3):
                axes[i, j].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "multiscale_vs_baseline_vis.png", dpi=160)
        plt.close(fig)

    return rows


def save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_report(path: Path, agg: Dict[str, float], args, n_train: int, n_val: int):
    lines = [
        "# Multi-Scale Token Graph Reasoning",
        "",
        f"- Train samples: `{n_train}`",
        f"- Val samples: `{n_val}`",
        f"- Epochs: `{args.epochs}`",
        f"- Coarse kNN k: `{args.k}`",
        f"- Fine weight: `{args.fine_weight}`",
        f"- Contrastive weight: `{args.contrastive_weight}`",
        f"- Contrastive tau: `{args.contrastive_tau}`",
        f"- Pair IoU threshold: `{args.pair_iou_thresh}`",
        "",
        "## Micro metrics",
        "",
        "| Model | F1 | IoU | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline graph | {agg['micro_baseline_f1']:.4f} | {agg['micro_baseline_iou']:.4f} | {agg['micro_baseline_precision']:.4f} | {agg['micro_baseline_recall']:.4f} |",
        f"| Multi-scale graph | {agg['micro_multiscale_f1']:.4f} | {agg['micro_multiscale_iou']:.4f} | {agg['micro_multiscale_precision']:.4f} | {agg['micro_multiscale_recall']:.4f} |",
        "",
        f"Delta IoU (multiscale - baseline): {agg['micro_multiscale_iou'] - agg['micro_baseline_iou']:.4f}",
        "",
        "Generated by `run_multiscale_token_graph_reasoning.py`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    pa = argparse.ArgumentParser(description="Multi-scale token graph reasoning")
    pa.add_argument("--tokens_T1", default="SECOND/tokens_T1")
    pa.add_argument("--tokens_T2", default="SECOND/tokens_T2")
    pa.add_argument("--matches", default="SECOND/matches")
    pa.add_argument("--images_T1", default="SECOND/im1")
    pa.add_argument("--images_T2", default="SECOND/im2")
    pa.add_argument("--embeddings_T1", default="SECOND/embeddings_T1")
    pa.add_argument("--embeddings_T2", default="SECOND/embeddings_T2")
    pa.add_argument("--label1_dir", default="SECOND/label1")
    pa.add_argument("--label2_dir", default="SECOND/label2")
    pa.add_argument("--output_dir", default="multiscale_graph_branch")

    pa.add_argument("--val_split", type=float, default=0.1)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--device", default="cuda")

    pa.add_argument("--epochs", type=int, default=8)
    pa.add_argument("--batch_size", type=int, default=4)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--weight_decay", type=float, default=1e-4)

    pa.add_argument("--k", type=int, default=6)
    pa.add_argument("--fine_weight", type=float, default=1.2)

    # Contrastive training options
    pa.add_argument("--contrastive_weight", type=float, default=0.2)
    pa.add_argument("--contrastive_tau", type=float, default=0.07)
    pa.add_argument("--pair_iou_thresh", type=float, default=0.5)
    pa.add_argument("--spatial_neg_dist", type=float, default=0.35)

    pa.add_argument("--coarse_points_per_side", type=int, default=24)
    pa.add_argument("--fine_points_per_side", type=int, default=48)
    pa.add_argument("--pred_iou_thresh", type=float, default=0.75)
    pa.add_argument("--stability_thresh", type=float, default=0.85)
    pa.add_argument("--min_mask_area", type=int, default=256)

    pa.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    pa.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")

    pa.add_argument("--train_max_samples", type=int, default=None)
    pa.add_argument("--val_max_samples", type=int, default=None)
    pa.add_argument("--vis_n", type=int, default=8)
    pa.add_argument("--refresh_cache", action="store_true", help="Rebuild cache even if files already exist")

    args = pa.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_dir = Path(args.tokens_T1)
    t2_dir = Path(args.tokens_T2)
    match_dir = Path(args.matches)
    img1_dir = Path(args.images_T1)
    img2_dir = Path(args.images_T2)
    emb1_dir = Path(args.embeddings_T1)
    emb2_dir = Path(args.embeddings_T2)
    label1_dir = Path(args.label1_dir)
    label2_dir = Path(args.label2_dir)

    train_stems, val_stems = build_stems(t1_dir, t2_dir, match_dir, args.val_split, args.seed)
    if args.train_max_samples is not None:
        train_stems = train_stems[: args.train_max_samples]
    if args.val_max_samples is not None:
        val_stems = val_stems[: args.val_max_samples]

    print(f"train={len(train_stems)} val={len(val_stems)}")

    print("Loading SAM generators (coarse/fine)...")
    amg_coarse = load_amg(
        config=args.sam2_config,
        ckpt=args.sam2_ckpt,
        points_per_side=args.coarse_points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_thresh=args.stability_thresh,
        min_mask_area=args.min_mask_area,
        device=str(device),
    )
    amg_fine = load_amg(
        config=args.sam2_config,
        ckpt=args.sam2_ckpt,
        points_per_side=args.fine_points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_thresh=args.stability_thresh,
        min_mask_area=max(64, args.min_mask_area // 4),
        device=str(device),
    )

    cache_dir = out_dir / "sample_cache"
    print("Preparing cached multi-scale tokens...")
    if args.refresh_cache and cache_dir.exists():
        for p in cache_dir.glob("*.pt"):
            p.unlink()
    prepare_cached_samples(train_stems + val_stems, cache_dir, img1_dir, img2_dir, emb1_dir, emb2_dir, label1_dir, label2_dir, amg_coarse, amg_fine)

    # Backward compatibility: old cache does not contain contrastive fields.
    sample_file = cache_dir / f"{train_stems[0]}.pt"
    sample_obj = torch.load(sample_file, map_location="cpu", weights_only=False)
    has_ctr = "contrastive" in sample_obj
    if not has_ctr:
        print("[WARN] Cache files do not contain contrastive tokens. Rebuild with --refresh_cache to enable contrastive learning.")

    train_ds = CachedPairDataset(train_stems, cache_dir)
    val_ds = CachedPairDataset(val_stems, cache_dir)

    # collate as list (variable token counts)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: x)

    base_model = BaselineCoarseGraphModel(in_dim=256, hidden=256, k=args.k).to(device)
    ms_model = MultiScaleGraphModel(in_dim=256, hidden=256, k=args.k).to(device)

    opt_b = torch.optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_m = torch.optim.AdamW(ms_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Training baseline + multiscale models...")
    hist = []
    for ep in range(1, args.epochs + 1):
        lb = train_epoch_baseline(base_model, train_loader, opt_b, device)
        lm = train_epoch_multiscale(
            ms_model,
            train_loader,
            opt_m,
            device,
            fine_weight=args.fine_weight,
            contrastive_weight=args.contrastive_weight if has_ctr else 0.0,
            contrastive_tau=args.contrastive_tau,
            pair_iou_thresh=args.pair_iou_thresh,
            spatial_neg_dist=args.spatial_neg_dist,
        )
        hist.append(
            {
                "epoch": ep,
                "baseline_loss": lb,
                "multiscale_loss_total": lm["loss_total"],
                "multiscale_loss_bce": lm["loss_bce"],
                "multiscale_loss_contrastive": lm["loss_contrastive"],
                "avg_positive_pairs": lm["avg_positive_pairs"],
                "avg_changed_neg_pairs": lm["avg_changed_neg_pairs"],
            }
        )
        print(
            f"  epoch {ep:02d}/{args.epochs} | baseline={lb:.4f} "
            f"multiscale_total={lm['loss_total']:.4f} "
            f"(bce={lm['loss_bce']:.4f}, ctr={lm['loss_contrastive']:.4f}) "
            f"pos/batch={lm['avg_positive_pairs']:.1f}"
        )

    rows = evaluate_models(base_model, ms_model, val_ds, device, label1_dir, label2_dir, args.vis_n, out_dir)

    agg = {}
    agg.update(aggregate(rows, "baseline"))
    agg.update(aggregate(rows, "multiscale"))

    print("\n" + "=" * 70)
    print("MULTI-SCALE TOKEN GRAPH RESULTS")
    print("=" * 70)
    print(
        f"Baseline   : F1={agg['micro_baseline_f1']:.4f} IoU={agg['micro_baseline_iou']:.4f} "
        f"P={agg['micro_baseline_precision']:.4f} R={agg['micro_baseline_recall']:.4f}"
    )
    print(
        f"Multiscale : F1={agg['micro_multiscale_f1']:.4f} IoU={agg['micro_multiscale_iou']:.4f} "
        f"P={agg['micro_multiscale_precision']:.4f} R={agg['micro_multiscale_recall']:.4f}"
    )
    print(f"Delta IoU  : {agg['micro_multiscale_iou'] - agg['micro_baseline_iou']:.4f}")
    print("=" * 70)

    save_csv(rows, out_dir / "multiscale_vs_baseline_metrics.csv")
    (out_dir / "aggregate_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out_dir / "train_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    save_report(out_dir / "multiscale_token_graph_report.md", agg, args, len(train_stems), len(val_stems))

    torch.save({"state": base_model.state_dict(), "args": vars(args)}, out_dir / "baseline_graph.pt")
    torch.save({"state": ms_model.state_dict(), "args": vars(args)}, out_dir / "multiscale_graph.pt")

    print(f"Saved: {out_dir / 'multiscale_vs_baseline_metrics.csv'}")
    print(f"Saved: {out_dir / 'aggregate_metrics.json'}")
    print(f"Saved: {out_dir / 'train_history.json'}")
    print(f"Saved: {out_dir / 'multiscale_token_graph_report.md'}")
    print(f"Saved: {out_dir / 'multiscale_vs_baseline_vis.png'}")


if __name__ == "__main__":
    main()
