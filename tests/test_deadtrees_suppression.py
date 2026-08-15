import numpy as np

from deadtrees_pipeline.suppress_masks import suppress_overlapping_masks


def _mask(y0, y1, x0, x1):
    mask = np.zeros((30, 30), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_mask_nms_keeps_higher_quality_duplicate():
    masks = np.stack([_mask(2, 12, 2, 12), _mask(2, 12, 2, 12)])
    keep, stats = suppress_overlapping_masks(masks, np.array([0.8, 0.9]), min_area_px=1)
    assert keep.tolist() == [1]
    assert stats["mask_iou_nms"] == 1


def test_containment_guard_preserves_much_smaller_mask():
    large = _mask(2, 22, 2, 22)
    small = _mask(4, 9, 4, 9)
    keep, stats = suppress_overlapping_masks(
        np.stack([large, small]), np.array([0.9, 0.8]), min_area_px=1
    )
    assert sorted(keep.tolist()) == [0, 1]
    assert stats.get("containment", 0) == 0


def test_area_filter_is_applied_before_nms():
    masks = np.stack([_mask(1, 3, 1, 3), _mask(5, 15, 5, 15)])
    keep, stats = suppress_overlapping_masks(masks, np.array([0.99, 0.5]), min_area_px=10)
    assert keep.tolist() == [1]
    assert stats["below_min_area"] == 1
