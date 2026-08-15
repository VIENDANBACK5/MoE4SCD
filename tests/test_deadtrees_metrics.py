import numpy as np

from deadtrees_pipeline.metrics import (
    boundary_metrics,
    hungarian_match,
    overlap_matrices,
    structural_errors,
)


def rectangle(y1, y2, x1, x2, size=32):
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_perfect_match_has_perfect_object_and_boundary_metrics():
    mask = rectangle(5, 20, 7, 23)
    iou, intersections, _, _ = overlap_matrices(mask[None], mask[None])
    match = hungarian_match(iou, 0.5)
    boundary = boundary_metrics(mask, mask, tolerance_px=1)
    assert intersections[0, 0] == mask.sum()
    assert match.tp == 1 and match.fp == 0 and match.fn == 0
    assert match.f1 == 1.0
    assert boundary["boundary_f1"] == 1.0
    assert boundary["assd_px"] == 0.0
    assert boundary["hd95_px"] == 0.0


def test_hungarian_matching_is_one_to_one():
    iou = np.array([[0.9, 0.8], [0.85, 0.0]])
    match = hungarian_match(iou, 0.5)
    assert match.tp == 2
    assert {(i, j) for i, j, _ in match.pairs} == {(0, 1), (1, 0)}


def test_structural_split_and_merge_are_visible_before_hungarian():
    # GT0 is split across pred0/pred1; pred2 merges GT0 and GT1.
    intersections = np.array([[40, 40, 20], [0, 0, 80]])
    gt_areas = np.array([100, 100])
    result = structural_errors(intersections, gt_areas, significant_fraction=0.10)
    assert result["split_gt_count"] == 1
    assert result["merge_pred_count"] == 1


def test_empty_predictions_count_all_gt_as_false_negative():
    match = hungarian_match(np.zeros((3, 0)), 0.5)
    assert match.tp == 0
    assert match.fp == 0
    assert match.fn == 3

