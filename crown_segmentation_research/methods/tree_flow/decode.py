"""
Euler-integration flow decoder for TreeFlowNet.

Given a centripetal vector field (vy, vx), iteratively advects each foreground
pixel toward its topological sink (crown center). Pixels converging to the same
sink form one instance.

Algorithm (from research.md Addendum 12 + Cellpose / Omnipose):
  1. Identify foreground pixels: canopy > canopy_thresh AND sdt > sdt_thresh
  2. For each foreground pixel p:
       p_{t+1} = round(p_t + eta * v_normalized(p_t))
       Repeat for T steps
  3. Final positions form clusters → one cluster = one instance
  4. Each instance contour extracted from clustered mask
  5. NMS on (centroid_heatmap peak, instance area) to remove duplicates
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import measure


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized Euler integration (GPU-accelerated)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_flow(flow: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize 2-channel flow field to unit vectors. flow: [2, H, W]"""
    mag = torch.sqrt((flow ** 2).sum(0, keepdim=True) + eps)
    return flow / mag


def euler_integrate(
    flow: torch.Tensor,       # [2, H, W]  raw flow logits
    foreground: torch.Tensor, # [H, W]     bool mask
    eta: float = 1.0,
    n_steps: int = 30,
    sigma_sink: float = 2.0,
    min_sink_density: float = 2.0,
) -> np.ndarray:
    """
    Vectorized Euler integration for all foreground pixels with sink peak clustering.
    Returns integer array [H, W] of cluster IDs (0 = background).
    """
    H, W = flow.shape[1:]

    flow_norm = _normalize_flow(flow)  # [2, H, W]

    # Starting positions for all foreground pixels
    fy, fx = foreground.nonzero(as_tuple=True)  # (N,) each
    if len(fy) == 0:
        return np.zeros((H, W), dtype=np.int32)

    # Positions as float [2, N] (row=y, col=x)
    pos = torch.stack([fy.float(), fx.float()], dim=0)  # [2, N]

    for _ in range(n_steps):
        py = pos[0].clamp(0, H - 1)
        px = pos[1].clamp(0, W - 1)

        grid_y = (py / (H - 1)) * 2 - 1
        grid_x = (px / (W - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)[None, None, :, :]  # [1,1,N,2]
        sampled = F.grid_sample(
            flow_norm[None],
            grid,
            mode='bilinear',
            align_corners=True,
        )
        v = sampled[0, :, 0, :]
        pos = pos + eta * v

    # Round final positions to integer grid coords
    final_y = pos[0].round().long().clamp(0, H - 1).cpu().numpy()
    final_x = pos[1].round().long().clamp(0, W - 1).cpu().numpy()

    # Form sink density map
    sink_density = np.zeros((H, W), dtype=np.float32)
    np.add.at(sink_density, (final_y, final_x), 1.0)

    # Gaussian smoothing
    smoothed = ndimage.gaussian_filter(sink_density, sigma=sigma_sink)

    # Find local maxima peaks
    local_max = (smoothed == ndimage.maximum_filter(smoothed, size=7)) & (smoothed >= min_sink_density)
    sink_y, sink_x = np.where(local_max)

    label_img = np.zeros((H, W), dtype=np.int32)
    fy_np = fy.cpu().numpy()
    fx_np = fx.cpu().numpy()

    if len(sink_y) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.stack([sink_y, sink_x], axis=1))
        _, nearest_sink_idx = tree.query(np.stack([final_y, final_x], axis=1))
        label_img[fy_np, fx_np] = nearest_sink_idx + 1
    else:
        # Fallback to connected components
        labeled_fg, _ = ndimage.label(foreground.cpu().numpy())
        label_img = labeled_fg

    return label_img


# ──────────────────────────────────────────────────────────────────────────────
# Centroid NMS — remove duplicate instances, keep highest centroid score
# ──────────────────────────────────────────────────────────────────────────────

def _instance_polygons_from_labels(
    label_img: np.ndarray,
    centroid_map: np.ndarray,
    min_area: int = 30,
) -> list[dict]:
    """
    Convert integer label image to list of instance dicts:
      {'mask': bool[H,W], 'score': float, 'centroid': (y,x), 'bbox': (r,c,h,w)}
    """
    instances = []
    for region in measure.regionprops(label_img):
        if region.area < min_area:
            continue
        r, c, r2, c2 = region.bbox
        sub_label = label_img[r:r2, c:c2]
        sub_mask = (sub_label == region.label)
        sub_cent = centroid_map[r:r2, c:c2]
        score = float(sub_cent[sub_mask].max()) if sub_mask.any() else 0.5

        full_mask = np.zeros(label_img.shape, dtype=bool)
        full_mask[r:r2, c:c2] = sub_mask

        cy, cx = region.centroid
        instances.append({
            'mask': full_mask,
            'score': score,
            'centroid': (cy, cx),
            'bbox': (r, c, r2 - r, c2 - c),
            'area': region.area,
        })
    return instances


def nms_instances(
    instances: list[dict],
    iou_thresh: float = 0.3,
) -> list[dict]:
    """Fast IoU-NMS on instance masks with bounding box pre-filtering."""
    instances = sorted(instances, key=lambda x: x['score'], reverse=True)
    keep = []
    suppressed = [False] * len(instances)
    for i, inst in enumerate(instances):
        if suppressed[i]:
            continue
        keep.append(inst)
        r1, c1, h1, w1 = inst['bbox']
        m1 = inst['mask']
        for j in range(i + 1, len(instances)):
            if suppressed[j]:
                continue
            r2, c2, h2, w2 = instances[j]['bbox']
            # Fast bbox overlap check
            if r1 >= r2 + h2 or r1 + h1 <= r2 or c1 >= c2 + w2 or c1 + w1 <= c2:
                continue
            m2 = instances[j]['mask']
            inter = (m1 & m2).sum()
            union = (m1 | m2).sum()
            if inter / (union + 1e-6) > iou_thresh:
                suppressed[j] = True
    return keep


# ──────────────────────────────────────────────────────────────────────────────
# Main decode function
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode(
    outputs: dict,
    canopy_thresh: float = 0.40,
    sdt_thresh: float = -0.50,    # SDT > thresh = inside a crown
    eta: float = 1.0,
    n_steps: int = 30,
    min_area: int = 30,
    nms_iou: float = 0.30,
    device: str = 'cpu',
) -> list[dict]:
    """
    Full decode pipeline: flow integration → clustering → NMS.

    Args:
        outputs: dict from TreeFlowNet.forward() for a SINGLE image
                 (all tensors [1, C, H, W] or [C, H, W])

    Returns:
        list of instance dicts: {'mask', 'score', 'centroid', 'bbox', 'area'}
    """
    def _sq(t):
        return t.squeeze(0) if t.dim() == 4 else t

    flow     = _sq(outputs['flow']).to(device)      # [2, H, W]
    sdt      = _sq(outputs['sdt']).to(device)       # [1, H, W]
    canopy   = _sq(outputs['canopy']).to(device)    # [1, H, W]
    centroid = _sq(outputs['centroid']).to(device)  # [1, H, W]

    # Foreground mask: inside canopy AND inside a crown (SDT > thresh)
    fg = (canopy[0] > canopy_thresh) & (sdt[0] > sdt_thresh)  # [H, W] bool

    if fg.sum() == 0:
        return []

    # Euler integration with sink clustering
    label_img = euler_integrate(flow, fg, eta=eta, n_steps=n_steps)

    # Build instance list from flow sink clusters
    centroid_np = centroid[0].cpu().numpy()
    instances = _instance_polygons_from_labels(label_img, centroid_np, min_area=min_area)

    # NMS
    instances = nms_instances(instances, iou_thresh=nms_iou)

    return instances


# ──────────────────────────────────────────────────────────────────────────────
# Organic TopoFlow-Watershed Decode (H19 Breakthrough)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_organic(
    outputs: dict,
    img_rgb: np.ndarray | None = None,
    min_sink_dist: int = 44,       # Physical crown scale exclusion radius (px) tuned in H20
    canopy_thresh: float = 0.30,   # Foreground canopy confidence threshold tuned in H20
    sdt_thresh: float = -0.15,     # Truncated signed distance threshold tuned in H20
    eta: float = 1.0,
    n_steps: int = 30,
    min_area: int = 150,           # Minimum tree crown area (px) tuned in H20
    grad_weight: float = 0.4,
    gradient_mode: str = "dizenzo", # "dizenzo", "sobel", or "none"
    refine_mode: str = "guided_filter", # "none", "guided_filter", "morph_snake", or "combined"
    sink_mode: str = "fixed",       # "fixed" (optimal) or "adaptive" (H22 scale-adaptive)
    r_min: float = 16.0,
    r_max: float = 52.0,
    score_mode: str = "calibrated", # "calibrated" (H26 Bayesian), "raw_apex", or "fixed"
    soft_nms: bool = True,          # H26 Gaussian Soft-NMS
    stream_burning: bool = False,   # H27 Disabled (gains +1.86 AP points)
    gorge_weight: float = 0.5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> list[dict]:
    """
    Advanced Organic TopoFlow-Watershed Decoder (H19 & H21).
    Resolves Voronoi straight cuts & touching-crown over-segmentation:
      1. Physical crown scale sink NMS (eliminates intra-crown multi-sink splits).
      2. Topographic Marker-Controlled Watershed on (1-SDT) + Di Zenzo color tensor gradient.
      3. High-resolution optical boundary refinement (Guided Filter / MorphSnakes).
    """
    import cv2
    from skimage import morphology, segmentation
    from crown_segmentation_research.methods.boundary_refinement.dizenzo_gradient import compute_dizenzo_gradient
    from crown_segmentation_research.methods.boundary_refinement.guided_filter_refine import refine_mask_guided_filter
    from crown_segmentation_research.methods.boundary_refinement.morph_snakes import refine_single_mask_morph_snake


    def _sq(t):
        return t.squeeze(0) if t.dim() == 4 else t

    flow     = _sq(outputs['flow']).to(device)
    sdt      = _sq(outputs['sdt']).to(device)
    canopy   = _sq(outputs['canopy']).to(device)
    centroid = _sq(outputs['centroid']).to(device)

    H, W = flow.shape[1:]
    sdt_np = sdt[0].cpu().numpy()
    canopy_np = canopy.sigmoid()[0].cpu().numpy() if canopy.max() > 1.0 or canopy.min() < 0.0 else canopy[0].cpu().numpy()
    cent_np = centroid.sigmoid()[0].cpu().numpy() if centroid.max() > 1.0 or centroid.min() < 0.0 else centroid[0].cpu().numpy()

    fg = (canopy_np > canopy_thresh) & (sdt_np > sdt_thresh)
    if fg.sum() < min_area:
        return []

    # Vectorized Euler advection
    mag = torch.sqrt((flow ** 2).sum(0, keepdim=True) + 1e-6)
    flow_norm = flow / mag

    fy, fx = torch.from_numpy(fg).nonzero(as_tuple=True)
    pos = torch.stack([fy.float().to(device), fx.float().to(device)], dim=0)

    for _ in range(n_steps):
        py = pos[0].clamp(0, H - 1)
        px = pos[1].clamp(0, W - 1)
        grid_y = (py / (H - 1)) * 2 - 1
        grid_x = (px / (W - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)[None, None, :, :]
        v = torch.nn.functional.grid_sample(flow_norm[None], grid, mode='bilinear', align_corners=True)[0, :, 0, :]
        pos = pos + eta * v

    final_y = pos[0].round().long().clamp(0, H - 1).cpu().numpy()
    final_x = pos[1].round().long().clamp(0, W - 1).cpu().numpy()

    # Saliency-weighted apex sink density
    sink_density = np.zeros((H, W), dtype=np.float32)
    np.add.at(sink_density, (final_y, final_x), 1.0)
    smoothed_sink = ndimage.gaussian_filter(sink_density, sigma=2.5)
    apex_score = smoothed_sink * (1.0 + 2.0 * cent_np + np.clip(sdt_np, 0, 1))

    # Scale-adaptive non-maximum suppression
    peak_candidates = (apex_score == ndimage.maximum_filter(apex_score, size=9)) & (smoothed_sink >= 1.5)
    py, px = np.where(peak_candidates)
    scores = apex_score[py, px]
    order = np.argsort(-scores)
    py, px = py[order], px[order]

    keep_sinks_y, keep_sinks_x = [], []
    keep_radii = []

    if sink_mode == "fixed":
        for y, x in zip(py, px):
            too_close = False
            for ky, kx in zip(keep_sinks_y, keep_sinks_x):
                dist = np.sqrt((y - ky)**2 + (x - kx)**2)
                if dist < min_sink_dist:
                    too_close = True
                    break
            if not too_close:
                keep_sinks_y.append(y)
                keep_sinks_x.append(x)
    else:
        # Scale-Adaptive Allometric Sink NMS (H22)
        # Dynamically scales exclusion radius based on predicted SDT and centroid peak
        for y, x in zip(py, px):
            local_sdt = max(0.0, float(sdt_np[y, x]))
            local_cent = float(cent_np[y, x])
            r_cand = r_min + (r_max - r_min) * (0.65 * local_sdt + 0.35 * min(1.0, 2.0 * local_cent))
            r_cand = float(np.clip(r_cand, r_min, r_max))

            too_close = False
            for ky, kx, kr in zip(keep_sinks_y, keep_sinks_x, keep_radii):
                dist = np.sqrt((y - ky)**2 + (x - kx)**2)
                thresh = (r_cand + kr) * 0.5  # average scale threshold
                if dist < thresh:
                    too_close = True
                    break
            if not too_close:
                keep_sinks_y.append(y)
                keep_sinks_x.append(x)
                keep_radii.append(r_cand)

    num_sinks = len(keep_sinks_y)
    if num_sinks == 0:
        return []

    # Construct Topographic Watershed Cost Surface
    if img_rgb is not None:
        if gradient_mode == "dizenzo":
            grad_norm = compute_dizenzo_gradient(img_rgb, sigma=1.0)
        elif gradient_mode == "sobel":
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_mag = cv2.GaussianBlur(grad_mag, (3, 3), 1.0)
            grad_norm = grad_mag / (grad_mag.max() + 1e-6)
        else:
            grad_norm = 0.0
    else:
        grad_norm = 0.0

    cost_surface = (1.0 - sdt_np) + 0.3 * (1.0 - canopy_np) - grad_weight * grad_norm

    # H27: Hydrological Stream-Burning Gorge Conditioning
    if img_rgb is not None and stream_burning:
        from crown_segmentation_research.methods.boundary_refinement.stream_burning import compute_stream_burning_gorge
        gorge_wall = compute_stream_burning_gorge(img_rgb)
        cost_surface = cost_surface + gorge_weight * gorge_wall

    markers = np.zeros((H, W), dtype=np.int32)
    for idx, (y, x) in enumerate(zip(keep_sinks_y, keep_sinks_x)):
        markers[y, x] = idx + 1

    labels = segmentation.watershed(cost_surface, markers, mask=fg)

    disk_struct = morphology.disk(2)
    instances = []
    for idx in range(1, num_sinks + 1):
        m = (labels == idx)
        if m.sum() < min_area:
            continue

        # Fast morphological smoothing on bounding box to avoid raster staircases
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        y0, y1 = max(0, int(ys.min()) - 2), min(H, int(ys.max()) + 3)
        x0, x1 = max(0, int(xs.min()) - 2), min(W, int(xs.max()) + 3)

        sub_m = m[y0:y1, x0:x1]
        sub_closed = ndimage.binary_closing(sub_m, structure=disk_struct)
        sub_clean = ndimage.binary_fill_holes(sub_closed)

        m_clean = np.zeros((H, W), dtype=bool)
        m_clean[y0:y1, x0:x1] = sub_clean

        # Optional High-Resolution Optical Boundary Refinement
        if img_rgb is not None and refine_mode in ("guided_filter", "combined"):
            m_clean = refine_mask_guided_filter(m_clean, img_rgb, radius=4, eps=1e-3, threshold=0.5)
        elif img_rgb is not None and refine_mode == "morph_snake":
            m_clean = refine_single_mask_morph_snake(m_clean, img_rgb, iterations=3, balloon=0.2)

        if m_clean.sum() < min_area:
            continue

        # H26: Bayesian Multi-Factor Confidence Scoring
        if score_mode == "calibrated":
            peak_cent = float(cent_np[m_clean].max()) if m_clean.sum() > 0 else 0.0
            mean_can = float(canopy_np[m_clean].mean()) if m_clean.sum() > 0 else 0.0
            mean_sdt = float(np.clip(sdt_np[m_clean].mean(), 0.0, 1.0)) if m_clean.sum() > 0 else 0.0
            calibrated_score = float(np.clip((peak_cent ** 0.55) * (mean_can ** 0.30) * (0.6 + 0.4 * mean_sdt), 0.01, 0.99))
        elif score_mode == "fixed":
            calibrated_score = 0.90
        else:
            calibrated_score = float(apex_score[keep_sinks_y[idx-1], keep_sinks_x[idx-1]])

        cy, cx = ndimage.center_of_mass(m_clean)
        instances.append({
            "mask": m_clean,
            "score": calibrated_score,
            "centroid": (float(cy), float(cx)),
            "area": int(m_clean.sum()),
            "bbox": (y0, x0, y1, x1),
        })

    # H26: Gaussian Soft-NMS Overlap Attenuation (Optimized with Fast Bounding Box Pre-Filtering)
    if soft_nms and len(instances) > 1:
        instances.sort(key=lambda x: x["score"], reverse=True)
        N = len(instances)
        for i in range(N):
            y0_i, x0_i, y1_i, x1_i = instances[i]["bbox"]
            mi = instances[i]["mask"]
            ai = instances[i]["area"]
            for j in range(i + 1, N):
                y0_j, x0_j, y1_j, x1_j = instances[j]["bbox"]
                # 1. Fast Disjoint Check
                if y0_i >= y1_j or y1_i <= y0_j or x0_i >= x1_j or x1_i <= x0_j:
                    continue
                # 2. Compute Intersection Only on Overlap Crop
                y_min = max(y0_i, y0_j)
                y_max = min(y1_i, y1_j)
                x_min = max(x0_i, x0_j)
                x_max = min(x1_i, x1_j)
                inter = np.logical_and(
                    mi[y_min:y_max, x_min:x_max],
                    instances[j]["mask"][y_min:y_max, x_min:x_max]
                ).sum()
                if inter > 0:
                    aj = instances[j]["area"]
                    iou = inter / (ai + aj - inter + 1e-6)
                    if iou > 0.15:
                        instances[j]["score"] *= float(np.exp(-(iou ** 2) / 0.5))

        instances = [inst for inst in instances if inst["score"] >= 0.05]

    return instances



# ──────────────────────────────────────────────────────────────────────────────
# Sweep helper for threshold tuning
# ──────────────────────────────────────────────────────────────────────────────

DECODE_DEFAULTS = dict(
    canopy_thresh=0.40,
    sdt_thresh=-0.50,
    eta=1.0,
    n_steps=30,
    min_area=30,
    nms_iou=0.30,
)

