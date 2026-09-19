
# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import numpy as np
from shapely.geometry import Polygon

from crown_segmentation_research.methods.star_convex.targets import (
    boundary_weight_map,
    build_targets,
    object_probability_map,
    ray_distance_maps,
)
from crown_segmentation_research.methods.star_convex.decode import decode


def _disk(shape, cy, cx, radius):
    ys, xs = np.ogrid[: shape[0], : shape[1]]
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= radius**2


def test_ray_distance_at_disk_center_matches_radius():
    shape = (60, 60)
    mask = _disk(shape, 30, 30, 15)
    rays = ray_distance_maps(mask, n_rays=8)
    center_rays = rays[:, 30, 30]
    assert np.allclose(center_rays, 15, atol=1.5)


def test_probability_peaks_near_disk_center():
    shape = (60, 60)
    mask = _disk(shape, 30, 30, 15)
    probability = object_probability_map(mask)
    peak_y, peak_x = np.unravel_index(np.argmax(probability), probability.shape)
    assert abs(peak_y - 30) <= 2 and abs(peak_x - 30) <= 2
    assert probability.max() == 1.0
    assert probability[~mask].max() == 0.0


def test_end_to_end_decode_recovers_two_disks():
    shape = (60, 100)
    left = _disk(shape, 30, 25, 18)
    right = _disk(shape, 30, 75, 18)
    probability, rays = build_targets([left, right], n_rays=32)

    polygons = decode(probability, rays, n_rays=32, prob_threshold=0.3, min_peak_distance=5)
    assert len(polygons) == 2
    areas = sorted(p.area for p in polygons)
    expected_area = left.sum()
    for area in areas:
        assert abs(area - expected_area) / expected_area < 0.15


def test_single_disk_decodes_to_one_polygon():
    shape = (60, 60)
    mask = _disk(shape, 30, 30, 18)
    probability, rays = build_targets([mask], n_rays=32)
    polygons = decode(probability, rays, n_rays=32, prob_threshold=0.3, min_peak_distance=5)
    assert len(polygons) == 1


def test_boundary_weight_peaks_between_two_close_disks():
    shape = (60, 100)
    left = _disk(shape, 30, 25, 18)  # spans columns 7-43
    right = _disk(shape, 30, 65, 18)  # spans columns 47-83, leaving a 3px gap (44-46)
    weight = boundary_weight_map([left, right], w0=10.0, sigma=10.0)
    assert weight.shape == shape
    # weight is 0 on every foreground (instance) pixel
    assert np.all(weight[left] == 0.0)
    assert np.all(weight[right] == 0.0)
    # weight peaks somewhere in the background gap between the two disks
    gap_columns = slice(44, 47)
    assert weight[30, gap_columns].max() > 0.0
    assert weight[30, gap_columns].max() == weight.max()
    # far from any instance, weight decays back to ~0
    assert weight[0, 0] < 1e-6


def test_boundary_weight_is_zero_for_a_single_instance():
    shape = (60, 60)
    mask = _disk(shape, 30, 30, 18)
    weight = boundary_weight_map([mask])
    assert np.all(weight == 0.0)


def test_embedding_aware_nms_keeps_high_iou_different_instances():
    from crown_segmentation_research.methods.star_convex.decode import polygon_nms

    # two nearly-identical (high-IoU) squares, but their peak embeddings say
    # they are different instances -- should both survive NMS
    square_a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    square_b = Polygon([(1, 0), (11, 0), (11, 10), (1, 10)])
    polygons = [square_a, square_b]
    scores = [0.9, 0.8]

    kept_plain = polygon_nms(polygons, scores, iou_threshold=0.3)
    assert kept_plain == [0]  # plain IoU-NMS suppresses the second (high overlap)

    peak_embeddings = [np.array([0.0, 0.0]), np.array([10.0, 10.0])]  # far apart
    kept_embedding_aware = polygon_nms(polygons, scores, iou_threshold=0.3, peak_embeddings=peak_embeddings, embedding_delta_d=1.5)
    assert set(kept_embedding_aware) == {0, 1}  # embedding says: different instances, keep both


def test_embedding_aware_nms_still_suppresses_same_instance():
    from crown_segmentation_research.methods.star_convex.decode import polygon_nms

    square_a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    square_b = Polygon([(1, 0), (11, 0), (11, 10), (1, 10)])
    polygons = [square_a, square_b]
    scores = [0.9, 0.8]

    peak_embeddings = [np.array([0.0, 0.0]), np.array([0.1, 0.1])]  # close together
    kept = polygon_nms(polygons, scores, iou_threshold=0.3, peak_embeddings=peak_embeddings, embedding_delta_d=1.5)
    assert kept == [0]  # embedding agrees they're the same instance -- suppress as usual
