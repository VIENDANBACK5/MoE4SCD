
# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import numpy as np

from crown_segmentation_research.legacy.dense_hv_representation import (
    compute_hv_targets,
    decode_instances,
)


def _square(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_hv_targets_are_zero_at_centroid_and_bounded():
    mask = _square((30, 30), 5, 25, 5, 25)
    p_fg, h_map, v_map = compute_hv_targets([mask])
    assert p_fg.sum() == mask.sum()
    assert np.all(h_map[~mask] == 0) and np.all(v_map[~mask] == 0)
    assert h_map.min() >= -1.0 and h_map.max() <= 1.0
    assert v_map.min() >= -1.0 and v_map.max() <= 1.0


def test_decode_recovers_two_touching_squares():
    shape = (40, 60)
    left = _square(shape, 5, 35, 5, 30)
    right = _square(shape, 5, 35, 30, 55)
    p_fg, h_map, v_map = compute_hv_targets([left, right])

    instances = decode_instances(p_fg, h_map, v_map)
    assert len(instances) == 2
    recovered = sorted(instances, key=lambda m: m.sum())
    expected = sorted([left, right], key=lambda m: m.sum())
    for pred, gt in zip(recovered, expected):
        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        assert intersection / union > 0.85


def test_decode_recovers_single_instance_as_one_piece():
    shape = (30, 30)
    mask = _square(shape, 5, 25, 5, 25)
    p_fg, h_map, v_map = compute_hv_targets([mask])
    instances = decode_instances(p_fg, h_map, v_map)
    assert len(instances) == 1
    iou = (instances[0] & mask).sum() / (instances[0] | mask).sum()
    assert iou > 0.9


def test_empty_input_returns_empty_list():
    assert compute_hv_targets([])[0].shape == (0, 0)
