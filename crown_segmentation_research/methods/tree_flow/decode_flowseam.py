"""
Two decoders that use the flow head for *pixel assignment*, not only for seeding.

Why: the oracle factorial (experiments/large_scale_v2/oracle_headroom.py) showed that the largest single
loss of decode_organic is its flood surface (1 - SDT + ...): the SDT target is clamped at 15 px, so the
surface is flat inside crowns and has no wall between touching crowns. The flow target, however, flips
direction across the seam between two crowns. Two published decoders exploit exactly that signal:

  flowwall   (HoVer-Net, Graham et al. 2019, MedIA): the Sobel gradient of the distance-to-centre maps is
             large on the seam and is used to build the watershed energy. Here: cost += w * relu(div v),
             since unit vectors on both sides of a seam point away from each other (positive divergence).
  flowassign (Cellpose, Stringer et al. 2021, Nature Methods): each pixel belongs to the sink its own
             trajectory converges to; no flooding. Here: pixels are assigned to the nearest seed of their
             final advected position (seeds = decode_organic's sink NMS), within the seed radius.

decode_organic is untouched; both functions share its seeding so only the assignment step differs.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def _probs(outputs):
    def _sq(t):
        return t.squeeze(0) if t.dim() == 4 else t
    flow, sdt = _sq(outputs["flow"]).float(), _sq(outputs["sdt"]).float()[0].numpy()
    can, cen = _sq(outputs["canopy"]).float()[0].numpy(), _sq(outputs["centroid"]).float()[0].numpy()
    if can.max() > 1.0 or can.min() < 0.0:
        can = 1 / (1 + np.exp(-can))
    if cen.max() > 1.0 or cen.min() < 0.0:
        cen = 1 / (1 + np.exp(-cen))
    return flow, sdt, can, cen


def _advect(flow: torch.Tensor, fg: np.ndarray, n_steps: int, device: str = "cpu"):
    # device="cuda" for single-tile evaluation (clean_benchmark/evaluate.py); the large-scale pipeline decodes many
    # tiles in parallel CPU worker processes and keeps the default
    H, W = fg.shape
    flow = flow.to(device)
    fn = flow / torch.sqrt((flow ** 2).sum(0, keepdim=True) + 1e-6)
    fy, fx = np.nonzero(fg)
    pos = torch.stack([torch.from_numpy(fy).float(), torch.from_numpy(fx).float()]).to(device)
    for _ in range(n_steps):
        py, px = pos[0].clamp(0, H - 1), pos[1].clamp(0, W - 1)
        grid = torch.stack([(px / (W - 1)) * 2 - 1, (py / (H - 1)) * 2 - 1], -1)[None, None]
        pos = pos + torch.nn.functional.grid_sample(fn[None], grid, mode="bilinear", align_corners=True)[0, :, 0, :]
    fy2 = pos[0].round().long().clamp(0, H - 1).cpu().numpy()
    fx2 = pos[1].round().long().clamp(0, W - 1).cpu().numpy()
    return fy, fx, fy2, fx2, fn.cpu()


def _seeds(fy2, fx2, shape, sdt, cen, seed_px: int):
    """Sink density + saliency + fixed-radius NMS, identical to decode_organic."""
    dens = np.zeros(shape, np.float32)
    np.add.at(dens, (fy2, fx2), 1.0)
    sm = ndimage.gaussian_filter(dens, 2.5)
    apex = sm * (1.0 + 2.0 * cen + np.clip(sdt, 0, 1))
    py, px = np.where((apex == ndimage.maximum_filter(apex, size=9)) & (sm >= 1.5))
    order = np.argsort(-apex[py, px])
    seeds = []
    for y, x in zip(py[order], px[order]):
        if all((y - ky) ** 2 + (x - kx) ** 2 >= seed_px ** 2 for ky, kx in seeds):
            seeds.append((int(y), int(x)))
    return seeds


def _instances_from_labels(labels, n, cen, can, min_area):
    """Per-label cleanup: close + fill holes, keep the largest component, drop small crowns."""
    disk = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], bool)
    out = []
    for lab, sl in enumerate(ndimage.find_objects(labels, max_label=n), start=1):
        if sl is None:
            continue
        m = labels[sl] == lab
        m = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(m, 2), structure=disk))[2:-2, 2:-2]
        cc, k = ndimage.label(m)
        if k > 1:
            m = cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)
        if m.sum() < min_area:
            continue
        full = np.zeros(labels.shape, bool)
        full[sl] = m
        ys, xs = np.nonzero(m)
        y0, x0 = sl[0].start, sl[1].start
        out.append({"mask": full, "score": float(np.clip(cen[sl][m].max() ** 0.55 * can[sl][m].mean() ** 0.3, 0.01, 0.99)),
                    "centroid": (float(ys.mean() + y0), float(xs.mean() + x0)), "area": int(m.sum()),
                    "bbox": (y0, x0, sl[0].stop, sl[1].stop)})
    return out


def decode_flowwall(outputs, img_rgb=None, min_sink_dist=44, canopy_thresh=0.30, sdt_thresh=-0.15,
                    n_steps=30, min_area=150, wall_weight=1.0, grad_weight=0.4, wall_sigma=1.5, device="cpu"):
    from skimage import segmentation
    flow, sdt, can, cen = _probs(outputs)
    fg = (can > canopy_thresh) & (sdt > sdt_thresh)
    if fg.sum() < min_area:
        return []
    _, _, fy2, fx2, fn = _advect(flow, fg, n_steps, device)
    seeds = _seeds(fy2, fx2, fg.shape, sdt, cen, min_sink_dist)
    if not seeds:
        return []
    vy, vx = fn[0].numpy(), fn[1].numpy()
    div = np.gradient(vx, axis=1) + np.gradient(vy, axis=0)
    wall = ndimage.gaussian_filter(np.clip(div, 0, None), wall_sigma)
    cost = (1.0 - sdt) + 0.3 * (1.0 - can) + wall_weight * wall
    if img_rgb is not None and grad_weight > 0:
        from crown_segmentation_research.methods.boundary_refinement.dizenzo_gradient import compute_dizenzo_gradient
        cost = cost - grad_weight * compute_dizenzo_gradient(img_rgb, sigma=1.0)
    markers = np.zeros(fg.shape, np.int32)
    for i, (y, x) in enumerate(seeds):
        markers[y, x] = i + 1
    labels = segmentation.watershed(cost, markers, mask=fg)
    return _instances_from_labels(labels, len(seeds), cen, can, min_area)


def decode_flowassign(outputs, img_rgb=None, min_sink_dist=44, canopy_thresh=0.30, sdt_thresh=-0.15,
                      n_steps=30, min_area=150, assign_radius=None, device="cpu"):
    from scipy.spatial import cKDTree
    flow, sdt, can, cen = _probs(outputs)
    fg = (can > canopy_thresh) & (sdt > sdt_thresh)
    if fg.sum() < min_area:
        return []
    fy, fx, fy2, fx2, _ = _advect(flow, fg, n_steps, device)
    seeds = _seeds(fy2, fx2, fg.shape, sdt, cen, min_sink_dist)
    if not seeds:
        return []
    r = assign_radius if assign_radius is not None else min_sink_dist
    d, idx = cKDTree(np.array(seeds, float)).query(np.stack([fy2, fx2], 1).astype(float), distance_upper_bound=r)
    ok = np.isfinite(d)
    labels = np.zeros(fg.shape, np.int32)
    labels[fy[ok], fx[ok]] = idx[ok] + 1
    return _instances_from_labels(labels, len(seeds), cen, can, min_area)
