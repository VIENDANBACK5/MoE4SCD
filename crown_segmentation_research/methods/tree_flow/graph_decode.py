"""Neuro-Flow-Graph (NFG) Decoder for Tree Crown and Deadwood Instance Segmentation.

Enhanced with:
  1. GSD-Adaptive Physical Scaling: Automatically tunes min_area, peak_distance, step_size,
     and bridge_distance from ground sampling distance (5cm, 10cm, 20cm).
  2. Divergence-Driven Singularities: Uses analytical vector divergence div(v) = dv_x/dx + dv_y/dy
     to detect subtle centripetal sinks in low-contrast broadleaf / mixed forest trees.
  3. Medial Axis Bellman-Ford Bridging: Shortest-path DP relaxation along principal
     orientations to bridge shadow gaps and connect fragmented fallen logs.
  4. Spectral Modularity Cut: Power iteration on the modularity matrix B to disentangle
     crisscrossing / tangled fallen wood piles.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon


def compute_divergence_field(flow: np.ndarray) -> np.ndarray:
    """Compute analytical vector field divergence div(v) = dv_x/dx + dv_y/dy.
    
    Negative divergence indicates strong centripetal convergence (sinks / tree centers).
    Positive divergence indicates boundaries and diverging flows.
    """
    vy, vx = flow[0], flow[1]
    
    # Central difference approximation
    dvx_dx = cv2.Sobel(vx, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dvy_dy = cv2.Sobel(vy, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    
    div = dvx_dx + dvy_dy
    # Negative divergence map in [0, 1] normalized
    neg_div = np.clip(-div, 0.0, None)
    max_val = float(np.max(neg_div))
    if max_val > 1e-4:
        neg_div = neg_div / max_val
    return neg_div.astype(np.float32)


def compute_orientation_field(flow: np.ndarray) -> np.ndarray:
    """Compute local undirected orientation angle phi in [-pi/2, pi/2] from flow field."""
    vy, vx = flow[0], flow[1]
    sin_2phi = 2.0 * vx * vy
    cos_2phi = vx**2 - vy**2
    phi = 0.5 * np.arctan2(sin_2phi, cos_2phi)
    return phi.astype(np.float32)


def get_adaptive_gsd_parameters(resolution: str | float) -> dict[str, float | int]:
    """Calculate physically invariant parameters based on Ground Sampling Distance (GSD)."""
    if isinstance(resolution, str):
        res_str = resolution.lower().strip()
        if "5cm" in res_str or "0.05" in res_str:
            gsd = 0.05
        elif "10cm" in res_str or "0.1" in res_str:
            gsd = 0.10
        elif "20cm" in res_str or "0.2" in res_str:
            gsd = 0.20
        else:
            gsd = 0.10
    else:
        gsd = float(resolution)

    # Physical scaling:
    # 1. Minimum instance area ~ 0.5 m^2 (allowing small snags / crown tops)
    min_area = max(int(0.5 / (gsd ** 2)), 6)
    # 2. Minimum peak distance ~ 0.25 m between distinct tree centers
    min_peak_dist = max(int(round(0.25 / gsd)), 1)
    # 3. Euler flow integration parameters
    if gsd <= 0.06:
        n_steps = 22
        step_size = 1.5
        max_bridge_dist = 45.0
    elif gsd <= 0.12:
        n_steps = 18
        step_size = 1.2
        max_bridge_dist = 25.0
    else:
        n_steps = 14
        step_size = 1.0
        max_bridge_dist = 14.0

    return {
        "gsd": gsd,
        "min_instance_area": min_area,
        "min_peak_distance": min_peak_dist,
        "n_steps": n_steps,
        "step_size": step_size,
        "max_bridge_distance": max_bridge_dist,
    }


def get_instance_endpoints_and_orientation(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute centroid, endpoints (p1, p2), aspect ratio, and principal orientation."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 5:
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        return np.array([cy, cx]), np.array([cy, cx]), 1.0, 0.0

    coords = np.stack([ys, xs], axis=1).astype(np.float32)
    center = np.mean(coords, axis=0)
    centered = coords - center

    # Covariance matrix PCA
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2) or np.isnan(cov).any():
        return center, center, 1.0, 0.0

    evals, evecs = np.linalg.eigh(cov)
    primary_axis = evecs[:, 1]  # (dy, dx)
    aspect_ratio = float(np.sqrt(max(evals[1], 1e-4) / max(evals[0], 1e-4)))
    orientation = float(np.arctan2(primary_axis[0], primary_axis[1]))

    proj = np.dot(centered, primary_axis)
    min_idx, max_idx = np.argmin(proj), np.argmax(proj)
    p1 = coords[min_idx]
    p2 = coords[max_idx]

    return p1, p2, aspect_ratio, orientation


def bridge_medial_axis_bellman_ford(
    instance_map: np.ndarray,
    canopy_prob: np.ndarray,
    max_bridge_distance: float = 35.0,
    max_angle_diff: float = np.deg2rad(35.0),
    min_area_to_bridge: int = 15,
) -> np.ndarray:
    """Medial Axis Shortest-Path / Bellman-Ford Bridging of fragmented logs across shadow gaps."""
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids != 0]

    if len(unique_ids) < 2:
        return instance_map

    height, width = instance_map.shape
    segments = {}

    for inst_id in unique_ids:
        mask = instance_map == inst_id
        area = int(mask.sum())
        if area < min_area_to_bridge:
            continue
        p1, p2, ar, orient = get_instance_endpoints_and_orientation(mask)
        center = 0.5 * (p1 + p2)
        segments[inst_id] = {
            "p1": p1,
            "p2": p2,
            "center": center,
            "ar": ar,
            "orient": orient,
            "area": area,
        }

    seg_ids = list(segments.keys())
    n_segs = len(seg_ids)
    if n_segs < 2:
        return instance_map

    parent = {sid: sid for sid in unique_ids}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n_segs):
        id_i = seg_ids[i]
        s_i = segments[id_i]
        for j in range(i + 1, n_segs):
            id_j = seg_ids[j]
            s_j = segments[id_j]

            d_angle = abs(s_i["orient"] - s_j["orient"])
            if d_angle > np.pi / 2:
                d_angle = np.pi - d_angle

            if d_angle > max_angle_diff:
                continue

            endpoints_i = [s_i["p1"], s_i["p2"]]
            endpoints_j = [s_j["p1"], s_j["p2"]]
            min_dist = 1e9
            best_pair = (None, None)

            for p_a in endpoints_i:
                for p_b in endpoints_j:
                    d = np.linalg.norm(p_a - p_b)
                    if d < min_dist:
                        min_dist = d
                        best_pair = (p_a, p_b)

            if min_dist > max_bridge_distance:
                continue

            p_a, p_b = best_pair
            bridge_vector = p_b - p_a
            bridge_angle = np.arctan2(bridge_vector[0], bridge_vector[1])
            d_bridge_angle = abs(bridge_angle - s_i["orient"])
            if d_bridge_angle > np.pi / 2:
                d_bridge_angle = np.pi - d_bridge_angle

            if d_bridge_angle > np.deg2rad(45.0) and min_dist > 10.0:
                continue

            num_samples = max(int(min_dist), 2)
            ys_line = np.linspace(p_a[0], p_b[0], num_samples).round().astype(np.int32)
            xs_line = np.linspace(p_a[1], p_b[1], num_samples).round().astype(np.int32)
            ys_line = np.clip(ys_line, 0, height - 1)
            xs_line = np.clip(xs_line, 0, width - 1)

            mean_canopy = float(np.mean(canopy_prob[ys_line, xs_line]))
            if mean_canopy >= 0.12 or min_dist <= 12.0:
                union(id_i, id_j)
                cv2.line(
                    instance_map,
                    (int(p_a[1]), int(p_a[0])),
                    (int(p_b[1]), int(p_b[0])),
                    int(id_i),
                    thickness=2,
                )

    new_map = np.zeros_like(instance_map)
    for uid in unique_ids:
        root_id = find(uid)
        new_map[instance_map == uid] = root_id

    return new_map


def spectral_modularity_cut_cluster(
    mask: np.ndarray,
    flow: np.ndarray,
    max_pts: int = 150,
    modularity_threshold: float = 0.12,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Spectral Modularity Bi-partitioning on tangled deadwood clusters."""
    ys, xs = np.nonzero(mask)
    n_pts = len(ys)
    if n_pts < 100:
        return None

    if n_pts > max_pts:
        step = n_pts // max_pts
        idx = np.arange(0, n_pts, step)[:max_pts]
        sub_ys = ys[idx]
        sub_xs = xs[idx]
    else:
        sub_ys = ys
        sub_xs = xs

    N = len(sub_ys)
    pts = np.stack([sub_ys, sub_xs], axis=1).astype(np.float32)
    v_vecs = np.stack([flow[0, sub_ys, sub_xs], flow[1, sub_ys, sub_xs]], axis=1)

    dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)
    sigma_spatial = 20.0
    spatial_aff = np.exp(-(dists**2) / (2.0 * sigma_spatial**2))

    dot_prods = np.clip(np.abs(np.dot(v_vecs, v_vecs.T)), 0.0, 1.0)
    orient_aff = dot_prods**2

    A = spatial_aff * orient_aff
    np.fill_diagonal(A, 0.0)

    deg = np.sum(A, axis=1)
    two_m = np.sum(deg)
    if two_m < 1e-4:
        return None

    B = A - np.outer(deg, deg) / two_m

    v = np.random.randn(N).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)

    for _ in range(15):
        v_next = np.dot(B, v)
        norm = np.linalg.norm(v_next)
        if norm < 1e-6:
            break
        v = v_next / norm

    lam = float(np.dot(v, np.dot(B, v)))
    if lam < modularity_threshold:
        return None

    group1_pts = pts[v > 0]
    group2_pts = pts[v <= 0]

    if len(group1_pts) < 20 or len(group2_pts) < 20:
        return None

    c1 = np.mean(group1_pts, axis=0)
    c2 = np.mean(group2_pts, axis=0)

    all_pts = np.stack([ys, xs], axis=1).astype(np.float32)
    d1 = np.sum((all_pts - c1) ** 2, axis=1)
    d2 = np.sum((all_pts - c2) ** 2, axis=1)

    mask1 = np.zeros_like(mask, dtype=bool)
    mask2 = np.zeros_like(mask, dtype=bool)

    assign_1 = d1 <= d2
    mask1[ys[assign_1], xs[assign_1]] = True
    mask2[ys[~assign_1], xs[~assign_1]] = True

    return mask1, mask2


def decode_neuro_flow_graph(
    flow: np.ndarray,
    canopy: np.ndarray,
    sdt: np.ndarray,
    centroid: np.ndarray,
    resolution: str | float = "5cm",
    canopy_threshold: float = 0.40,
    sdt_threshold: float = -0.30,
    centroid_threshold: float = 0.15,
    use_divergence_seeds: bool = True,
    use_bellman_ford: bool = True,
    use_spectral_cut: bool = True,
    min_instance_area: int | None = None,
    min_peak_distance: int | None = None,
    max_bridge_distance: float | None = None,
) -> tuple[np.ndarray, list[Polygon]]:
    """Complete GSD-Adaptive Neuro-Flow-Graph (NFG) Decoder."""
    if canopy.ndim == 3:
        canopy = canopy.squeeze(0)
    if sdt.ndim == 3:
        sdt = sdt.squeeze(0)
    if centroid.ndim == 3:
        centroid = centroid.squeeze(0)

    # 1. Calculate GSD Adaptive Parameters
    gsd_params = get_adaptive_gsd_parameters(resolution)
    if min_instance_area is None:
        min_instance_area = gsd_params["min_instance_area"]
    if min_peak_distance is None:
        min_peak_distance = gsd_params["min_peak_distance"]
    if max_bridge_distance is None:
        max_bridge_distance = gsd_params["max_bridge_distance"]

    n_steps = int(gsd_params["n_steps"])
    step_size = float(gsd_params["step_size"])

    height, width = canopy.shape
    fg_mask = (canopy >= canopy_threshold) & (sdt >= sdt_threshold)
    if not np.any(fg_mask):
        return np.zeros((height, width), dtype=np.int32), []

    # 2. Combine Centroid Heatmap + Negative Divergence Field div(v) < 0
    if use_divergence_seeds:
        neg_div = compute_divergence_field(flow)
        # Hybrid seed response: strong learned centroid OR geometric flow convergence
        seed_response = np.maximum(centroid, 0.70 * neg_div * (canopy >= 0.25))
    else:
        seed_response = centroid

    peak_filter = ndimage.maximum_filter(seed_response, size=min_peak_distance * 2 + 1)
    peaks = (seed_response == peak_filter) & (seed_response >= centroid_threshold) & fg_mask
    seed_ys, seed_xs = np.nonzero(peaks)

    # 3. Vectorized Euler flow integration
    fg_ys, fg_xs = np.nonzero(fg_mask)
    cur_ys = fg_ys.astype(np.float32)
    cur_xs = fg_xs.astype(np.float32)

    vy_map = flow[0]
    vx_map = flow[1]

    for _ in range(n_steps):
        iy = np.clip(np.round(cur_ys).astype(np.int32), 0, height - 1)
        ix = np.clip(np.round(cur_xs).astype(np.int32), 0, width - 1)
        cur_ys += step_size * vy_map[iy, ix]
        cur_xs += step_size * vx_map[iy, ix]
        cur_ys = np.clip(cur_ys, 0, height - 1)
        cur_xs = np.clip(cur_xs, 0, width - 1)

    end_ys = cur_ys
    end_xs = cur_xs

    # 4. Initial instance map from flow sinks
    instance_map = np.zeros((height, width), dtype=np.int32)
    if len(seed_ys) > 0:
        seeds = np.stack([seed_ys, seed_xs], axis=1).astype(np.float32)
        endpoints = np.stack([end_ys, end_xs], axis=1).astype(np.float32)
        chunk_size = 50000
        best_seeds = np.zeros(len(endpoints), dtype=np.int32)
        for start in range(0, len(endpoints), chunk_size):
            end = min(start + chunk_size, len(endpoints))
            dists = np.sum((endpoints[start:end, np.newaxis, :] - seeds[np.newaxis, :, :]) ** 2, axis=-1)
            best_seeds[start:end] = np.argmin(dists, axis=-1) + 1
        instance_map[fg_ys, fg_xs] = best_seeds
    else:
        num_labels, labels = cv2.connectedComponents(fg_mask.astype(np.uint8))
        instance_map = labels

    # 5. Medial Axis Bellman-Ford Bridging across shadow gaps
    if use_bellman_ford:
        instance_map = bridge_medial_axis_bellman_ford(
            instance_map,
            canopy_prob=canopy,
            max_bridge_distance=max_bridge_distance,
            min_area_to_bridge=max(min_instance_area // 2, 4),
        )

    # 6. Spectral Modularity Cut for tangled clusters (with aspect ratio & size gating)
    next_id = int(instance_map.max()) + 1
    if use_spectral_cut:
        current_ids = list(np.unique(instance_map))
        for inst_id in current_ids:
            if inst_id == 0:
                continue
            inst_mask = instance_map == inst_id
            if inst_mask.sum() > max(min_instance_area * 10, 300):
                # Check aspect ratio / variance to avoid splitting compact snags
                _, _, ar, _ = get_instance_endpoints_and_orientation(inst_mask)
                if ar > 2.0:  # Only cut elongated or tangled clusters
                    split_res = spectral_modularity_cut_cluster(
                        inst_mask,
                        flow,
                        modularity_threshold=0.12,
                    )
                    if split_res is not None:
                        m1, m2 = split_res
                        instance_map[m1] = inst_id
                        instance_map[m2] = next_id
                        next_id += 1

    # 7. Extract Shapely Polygons and clean instance map
    polygons = []
    clean_instance_map = np.zeros((height, width), dtype=np.int32)
    final_id = 1

    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids != 0]

    for inst_id in unique_ids:
        inst_mask = (instance_map == inst_id).astype(np.uint8)
        if inst_mask.sum() < min_instance_area:
            continue

        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt.squeeze(1)
                poly = Polygon(pts)
                if poly.is_valid and poly.area >= min_instance_area:
                    polygons.append(poly)
                    clean_instance_map[inst_mask > 0] = final_id
                    final_id += 1

    return clean_instance_map, polygons
