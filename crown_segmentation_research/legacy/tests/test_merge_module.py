
# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import numpy as np

from crown_segmentation_research.evaluation.merge_module import merge_oversegmented_instances


def _square_mask(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_adjacent_same_color_fragments_merge():
    shape = (40, 40)
    image = np.full((*shape, 3), 120, dtype=np.uint8)  # uniform color everywhere
    left = _square_mask(shape, 5, 20, 5, 20)
    right = _square_mask(shape, 5, 20, 20, 35)  # touches left at x=20

    merged_masks, merged_scores = merge_oversegmented_instances(
        [left, right], [0.9, 0.8], image, gsd_cm=1.7,
        delta_appearance=0.05, delta_distance_m=100.0,
    )
    assert len(merged_masks) == 1
    assert merged_scores == [0.9]
    assert np.array_equal(merged_masks[0], left | right)


def test_far_apart_fragments_do_not_merge():
    shape = (100, 100)
    image = np.full((*shape, 3), 120, dtype=np.uint8)
    a = _square_mask(shape, 0, 5, 0, 5)
    b = _square_mask(shape, 90, 95, 90, 95)

    merged_masks, _ = merge_oversegmented_instances(
        [a, b], [0.9, 0.8], image, gsd_cm=1.7,
        delta_appearance=1.0, delta_distance_m=0.5,
    )
    assert len(merged_masks) == 2


def test_adjacent_but_different_appearance_does_not_merge():
    shape = (40, 40)
    image = np.full((*shape, 3), 0, dtype=np.uint8)
    left = _square_mask(shape, 5, 20, 5, 20)
    right = _square_mask(shape, 5, 20, 20, 35)
    image[right] = 255  # sharply different appearance

    merged_masks, _ = merge_oversegmented_instances(
        [left, right], [0.9, 0.8], image, gsd_cm=1.7,
        delta_appearance=0.05, delta_distance_m=100.0,
    )
    assert len(merged_masks) == 2


def test_transitive_merge_across_three_fragments():
    shape = (60, 60)
    image = np.full((*shape, 3), 120, dtype=np.uint8)
    a = _square_mask(shape, 5, 20, 5, 20)
    b = _square_mask(shape, 5, 20, 20, 35)
    c = _square_mask(shape, 5, 20, 35, 50)

    merged_masks, _ = merge_oversegmented_instances(
        [a, b, c], [0.9, 0.8, 0.7], image, gsd_cm=1.7,
        delta_appearance=0.05, delta_distance_m=100.0,
    )
    assert len(merged_masks) == 1
    assert np.array_equal(merged_masks[0], a | b | c)


def test_single_or_empty_input_is_passthrough():
    shape = (10, 10)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    single = [_square_mask(shape, 0, 5, 0, 5)]
    masks, scores = merge_oversegmented_instances(
        single, [0.5], image, gsd_cm=1.7, delta_appearance=1.0, delta_distance_m=1.0
    )
    assert len(masks) == 1
    masks, scores = merge_oversegmented_instances(
        [], [], image, gsd_cm=1.7, delta_appearance=1.0, delta_distance_m=1.0
    )
    assert masks == [] and scores == []
