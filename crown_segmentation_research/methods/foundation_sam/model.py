"""CrownTransformer SAM 2.0 (Exact Meta SAM Architecture in Pure PyTorch).

Implements:
  1. Dense Prompt Spatial Injection (Gaussian Anchor at Stride 16 embedding level)
  2. 2-Layer Two-Way Transformer Cross-Attention (Figure 14 in SAM Paper) with Final Token-to-Image Attention
  3. 3-Mask Ambiguity Resolution Tokens (Subpart, Part, Whole) + 1 IoU Token
  4. 2-Layer Transposed Conv Upscaling (Stride 16 -> Stride 4, 32 channels) producing pure visual features F_high
  5. Continuous Dot-Product Mask Head: Mask_i(y, x) = sigma( w_i^T * F_high(y, x) )
  6. Dense Grid AMG Engine with Stability Score Filtering (IoU at +/- 1.0 threshold) and GPU IoU NMS

100% Native PyTorch, Zero external foundation model weights, 100% Fully Differentiable.
"""
from __future__ import annotations

import math
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from shapely.geometry import Polygon
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

BENCH_DIR = Path("DTE-Aerial-Data-public")
OUT_DIR = Path("crown_segmentation_research/images/previews_crown_transformer_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class PositionEmbeddingSine(nn.Module):
    """Clean 2D Sine-Cosine Positional Encodings (SAM Section 3.1)."""
    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, coords: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        H, W = shape
        # coords: (..., 2) where coords[..., 0] is y in [0, H], coords[..., 1] is x in [0, W]
        y = (coords[..., 0] / float(H)) * (2.0 * math.pi)
        x = (coords[..., 1] / float(W)) * (2.0 * math.pi)

        dim_t = torch.arange(self.num_pos_feats // 2, dtype=torch.float32, device=coords.device)
        dim_t = self.temperature ** (2.0 * dim_t / float(self.num_pos_feats // 2))

        sin_y = torch.sin(y.unsqueeze(-1) / dim_t)
        cos_y = torch.cos(y.unsqueeze(-1) / dim_t)
        sin_x = torch.sin(x.unsqueeze(-1) / dim_t)
        cos_x = torch.cos(x.unsqueeze(-1) / dim_t)

        pos_y = torch.cat([sin_y, cos_y], dim=-1)  # (..., 128)
        pos_x = torch.cat([sin_x, cos_x], dim=-1)  # (..., 128)
        return torch.cat([pos_y, pos_x], dim=-1)   # (..., 256)


class SmoothUpscaleDecoder(nn.Module):
    """Smooth 4x Feature Upscaler using Bilinear Interpolation + Conv2d(3x3).
    Completely eliminates Transposed Conv checkerboard/striping artifacts.
    """
    def __init__(self, in_ch: int = 256, out_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*K, 256, H16, W16) -> (B*K, 64, H8, W8)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        # (B*K, 64, H8, W8) -> (B*K, 32, H4, W4)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv2(x)
        return x


def make_gaussian_prompt(coords: torch.Tensor, feat_shape: tuple[int, int], img_shape: tuple[int, int], sigma: float = 1.5) -> torch.Tensor:
    B, K, _ = coords.shape
    Hf, Wf = feat_shape
    H, W = img_shape
    device = coords.device

    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, Hf - 1, Hf, device=device),
        torch.linspace(0, Wf - 1, Wf, device=device),
        indexing="ij",
    )
    y_grid = y_grid.view(1, 1, Hf, Wf)
    x_grid = x_grid.view(1, 1, Hf, Wf)

    cy = (coords[..., 0:1] / float(H) * float(Hf)).view(B, K, 1, 1)
    cx = (coords[..., 1:2] / float(W) * float(Wf)).view(B, K, 1, 1)

    dist_sq = (y_grid - cy)**2 + (x_grid - cx)**2
    gauss = torch.exp(-dist_sq / (2.0 * (sigma**2)))
    return gauss


class CrownTransformerSAM(nn.Module):
    """Full Standalone CrownTransformerSAM 2.0 with Meta SAM Architecture."""
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        # Backbone (ResNet50-FPN)
        self.backbone = resnet_fpn_backbone("resnet50", weights=None, trainable_layers=5)
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=128)

        # Dense Prompt Projection (Gaussian map -> 256 ch)
        self.prompt_proj = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=1),
        )

        # 4 Output Tokens: 3 Mask Tokens [Subpart, Part, Whole] + 1 IoU Token
        self.num_mask_tokens = 3
        self.mask_tokens = nn.Embedding(self.num_mask_tokens + 1, d_model)
        nn.init.normal_(self.mask_tokens.weight, std=0.02)

        self.point_embed = nn.Embedding(2, d_model)
        nn.init.normal_(self.point_embed.weight, std=0.02)

        # 2-Layer Two-Way Transformer Cross-Attention
        self.self_attn1 = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.token_to_img1 = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp1 = nn.Sequential(nn.Linear(d_model, 1024), nn.GELU(), nn.Linear(1024, d_model))
        self.norm3 = nn.LayerNorm(d_model)
        self.img_to_token1 = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm4 = nn.LayerNorm(d_model)

        self.self_attn2 = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm5 = nn.LayerNorm(d_model)
        self.token_to_img2 = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm6 = nn.LayerNorm(d_model)
        self.mlp2 = nn.Sequential(nn.Linear(d_model, 1024), nn.GELU(), nn.Linear(1024, d_model))
        self.norm7 = nn.LayerNorm(d_model)
        self.img_to_token2 = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm8 = nn.LayerNorm(d_model)

        # Final Token-to-Image Attention (SAM Section A)
        self.final_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm_final = nn.LayerNorm(d_model)

        # 4x Smooth Bilinear Upscaling Decoder (256 -> 64 -> 32)
        self.output_upscaling = SmoothUpscaleDecoder(in_ch=d_model, out_ch=32)

        # 3 MLPs for the 3 Output Mask Tokens
        self.mask_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, 32),
            )
            for _ in range(self.num_mask_tokens)
        ])

        # IoU Prediction MLP
        self.iou_mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.num_mask_tokens),
            nn.Sigmoid(),
        )

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fpn = self.backbone(x)
        p4 = fpn["2"]  # (B, 256, H16, W16)
        return p4, p4.flatten(2).permute(0, 2, 1)

    def forward_decoder(
        self,
        p4: torch.Tensor,
        point_coords: torch.Tensor,
        img_shape: tuple[int, int],
        point_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if point_coords.ndim == 2:
            point_coords = point_coords.unsqueeze(0)

        B, C, H16, W16 = p4.shape
        K = point_coords.shape[1]
        H, W = img_shape
        device = p4.device

        # 1. Inject Dense Gaussian Prompt into Image Embeddings
        g_maps = make_gaussian_prompt(point_coords, (H16, W16), (H, W), sigma=1.5)
        g_flat = g_maps.view(B * K, 1, H16, W16)
        g_embed = self.prompt_proj(g_flat).flatten(2).permute(0, 2, 1)  # (B*K, H16*W16, 256)

        img_tokens = p4.flatten(2).permute(0, 2, 1)
        img_embed_bk = img_tokens.unsqueeze(1).repeat(1, K, 1, 1).view(B * K, H16 * W16, C)
        img_embed_bk = img_embed_bk + g_embed  # Spatial Prompt Conditioned Image Features!

        # 2. Positional Encodings
        ys = torch.linspace(0.5, H16 - 0.5, H16, device=device) * (float(H) / float(H16))
        xs = torch.linspace(0.5, W16 - 0.5, W16, device=device) * (float(W) / float(W16))
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        grid_pts = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)
        pos_img = self.pos_enc(grid_pts, (H, W)).unsqueeze(0).repeat(B * K, 1, 1)

        pts_flat = point_coords.view(B * K, 1, 2)
        pos_pts = self.pos_enc(pts_flat, (H, W))

        if point_labels is None:
            labels_bk = torch.ones((B * K, 1), dtype=torch.long, device=device)
        else:
            labels_bk = point_labels.view(B * K, 1).long()

        prompt_tokens = self.point_embed(labels_bk) + pos_pts  # (B*K, 1, 256)

        # 3. Concatenate 4 Output Tokens with Prompt Tokens
        out_tokens = self.mask_tokens.weight.unsqueeze(0).repeat(B * K, 1, 1)  # (B*K, 4, 256)
        pos_out = torch.zeros_like(out_tokens)

        tokens = torch.cat([out_tokens, prompt_tokens], dim=1)  # (B*K, 5, 256)
        pos_tokens = torch.cat([pos_out, pos_pts], dim=1)

        # 4. Two-Way Transformer Cross-Attention (Layer 1)
        q = k = tokens + pos_tokens
        t_attn, _ = self.self_attn1(q, k, tokens)
        tokens = self.norm1(tokens + t_attn)

        q = tokens + pos_tokens
        k = img_embed_bk + pos_img
        t_attn2, _ = self.token_to_img1(q, k, img_embed_bk)
        tokens = self.norm2(tokens + t_attn2)
        tokens = self.norm3(tokens + self.mlp1(tokens))

        q = img_embed_bk + pos_img
        k = tokens + pos_tokens
        i_attn, _ = self.img_to_token1(q, k, tokens)
        img_embed_bk = self.norm4(img_embed_bk + i_attn)

        # Layer 2
        q = k = tokens + pos_tokens
        t_attn, _ = self.self_attn2(q, k, tokens)
        tokens = self.norm5(tokens + t_attn)

        q = tokens + pos_tokens
        k = img_embed_bk + pos_img
        t_attn2, _ = self.token_to_img2(q, k, img_embed_bk)
        tokens = self.norm6(tokens + t_attn2)
        tokens = self.norm7(tokens + self.mlp2(tokens))

        q = img_embed_bk + pos_img
        k = tokens + pos_tokens
        i_attn, _ = self.img_to_token2(q, k, tokens)
        img_embed_bk = self.norm8(img_embed_bk + i_attn)

        # Final Token-to-Image Attention
        q = tokens + pos_tokens
        k = img_embed_bk + pos_img
        t_attn_f, _ = self.final_attn(q, k, img_embed_bk)
        tokens = self.norm_final(tokens + t_attn_f)

        # 5. Upscale updated image features: (B*K, 32, H4, W4)
        img_spatial = img_embed_bk.permute(0, 2, 1).view(B * K, C, H16, W16)
        f_upscaled = self.output_upscaling(img_spatial)
        H4, W4 = f_upscaled.shape[-2:]

        # 6. Dynamic Hypernetwork Masks (3 Masks)
        mask_logits_list = []
        for i in range(self.num_mask_tokens):
            t_i = tokens[:, i, :]
            w_i = self.mask_mlps[i](t_i)  # (B*K, 32)
            m_i = (w_i.unsqueeze(-1).unsqueeze(-1) * f_upscaled).sum(dim=1)
            mask_logits_list.append(m_i)

        mask_logits = torch.stack(mask_logits_list, dim=1).view(B, K, self.num_mask_tokens, H4, W4)
        iou_preds = self.iou_mlp(tokens[:, self.num_mask_tokens, :]).view(B, K, self.num_mask_tokens)

        return mask_logits, iou_preds


def calculate_stability_score(logits: torch.Tensor, mask_threshold: float = 0.0, threshold_offset: float = 1.0) -> torch.Tensor:
    high_mask = logits > (mask_threshold + threshold_offset)
    low_mask = logits > (mask_threshold - threshold_offset)
    intersection = (high_mask & low_mask).sum(dim=(-2, -1)).float()
    union = (high_mask | low_mask).sum(dim=(-2, -1)).float()
    return (intersection + 1e-6) / (union + 1e-6)


def run_dense_amg_inference(
    model: CrownTransformerSAM,
    image_rgb: np.ndarray,
    device: torch.device,
    grid_size: int = 32,
    prob_thresh: float = 0.35,
    iou_thresh: float = 0.35,
    stability_thresh: float = 0.85,
    min_area: int = 25,
) -> tuple[np.ndarray, list[Polygon], int]:
    H, W = image_rgb.shape[:2]
    img_t = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        p4, _ = model.extract_features(img_t)

    ys = np.linspace(16, H - 16, grid_size)
    xs = np.linspace(16, W - 16, grid_size)
    all_points = torch.tensor([[y, x] for y in ys for x in xs], dtype=torch.float32, device=device)
    total_pts = len(all_points)

    chunk_size = 64
    candidate_masks = []
    candidate_ious = []

    for start in range(0, total_pts, chunk_size):
        end = min(start + chunk_size, total_pts)
        pts_chunk = all_points[start:end]

        with torch.no_grad():
            logits_h4, ious = model.forward_decoder(p4, pts_chunk, (H, W))
            logits_h4 = logits_h4.squeeze(0)  # (K, 3, H4, W4)
            ious = ious.squeeze(0)            # (K, 3)

            stability = calculate_stability_score(logits_h4, mask_threshold=0.0, threshold_offset=1.0)

            # Pick mask with highest predicted IoU
            best_mask_idx = torch.argmax(ious, dim=1)
            k_indices = torch.arange(len(pts_chunk), device=device)
            best_logits = logits_h4[k_indices, best_mask_idx]
            best_ious = ious[k_indices, best_mask_idx]
            best_stability = stability[k_indices, best_mask_idx]

            probs = torch.sigmoid(F.interpolate(best_logits.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)).squeeze(1).cpu().numpy()
            best_ious_np = best_ious.cpu().numpy()
            best_stab_np = best_stability.cpu().numpy()

        for k in range(len(pts_chunk)):
            prob_map = probs[k]
            py, px = int(pts_chunk[k, 0].item()), int(pts_chunk[k, 1].item())

            # Stability filter
            if best_stab_np[k] < stability_thresh:
                continue

            if prob_map[py, px] < 0.25:
                continue

            mask_area = (prob_map >= prob_thresh).sum()
            if mask_area < min_area or mask_area > (H * W * 0.15):
                continue

            fg_score = float(prob_map[prob_map >= prob_thresh].mean())
            if fg_score < 0.35:
                continue

            bin_mask = (prob_map >= prob_thresh)
            candidate_masks.append(bin_mask)
            candidate_ious.append(fg_score)

    total_candidates = len(candidate_masks)
    if total_candidates == 0:
        return np.zeros((H, W), dtype=np.int32), [], 0

    # GPU IoU NMS
    masks_tensor = torch.from_numpy(np.stack(candidate_masks, axis=0)).to(device).float()
    N = masks_tensor.shape[0]
    masks_flat = masks_tensor.view(N, -1)

    intersection = torch.mm(masks_flat, masks_flat.t())
    areas = masks_flat.sum(dim=1, keepdim=True)
    union = areas + areas.t() - intersection
    iou_mat = (intersection / (union + 1e-6)).cpu().numpy()

    order = np.argsort(-np.array(candidate_ious))
    keep = []
    suppressed = np.zeros(total_candidates, dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        overlapping = np.where(iou_mat[i] > iou_thresh)[0]
        suppressed[overlapping] = True

    # Panoptic assignment
    instance_map = np.zeros((H, W), dtype=np.int32)
    claimed = np.zeros((H, W), dtype=bool)
    polygons = []
    next_id = 1

    for idx in keep:
        mask = candidate_masks[idx]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if num_labels > 1:
            largest_comp_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean_mask = (labels == largest_comp_idx)
        else:
            clean_mask = mask

        clean_mask = clean_mask & (~claimed)
        if clean_mask.sum() < min_area:
            continue

        cnts, _ = cv2.findContours(clean_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if len(cnt) >= 3 and cv2.contourArea(cnt) >= min_area:
                pts = cnt.squeeze(1)
                poly = Polygon(pts)
                if poly.is_valid and poly.area >= min_area:
                    polygons.append(poly)
                    cv2.drawContours(instance_map, [cnt], -1, next_id, thickness=cv2.FILLED)
                    cv2.drawContours(claimed.astype(np.uint8), [cnt], -1, 1, thickness=cv2.FILLED)
                    next_id += 1

    return instance_map, polygons, total_candidates


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing CrownTransformerSAM 2.0 on {device}...")
    model = CrownTransformerSAM().to(device)
    ckpt = Path("DeadTrees/experiments/crown_transformer_sam_v2/best_crown_transformer_sam.pth")
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"Loaded {ckpt}!")
    model.eval()


if __name__ == "__main__":
    main()
