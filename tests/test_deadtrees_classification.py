import numpy as np

from deadtrees_pipeline.classify_objects import (
    ALIVE_MIN_TREECOVER_OVERLAP,
    CLEAN_NEGATIVE_OVERLAP,
    RGB_FEATURES,
    SHAPE_FEATURES,
    classify_condition,
    extract_rgb_features,
    extract_shape_features,
)


def test_shape_features_for_solid_rectangle():
    mask = np.zeros((12, 14), dtype=bool)
    mask[2:10, 3:11] = True
    features = extract_shape_features(mask)

    assert set(features) == set(SHAPE_FEATURES)
    assert np.isclose(features["shape_extent"], 1.0)
    assert np.isclose(features["shape_solidity"], 1.0)
    assert np.isclose(features["shape_hole_fraction"], 0.0)
    assert np.isclose(features["shape_log_components"], np.log(2.0))


def test_rgb_features_use_only_mask_pixels_and_scale_to_unit_interval():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[1, 1] = [255, 128, 0]
    image[1, 2] = [255, 0, 0]
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1:3] = True
    features = extract_rgb_features(image, mask)

    assert set(features) == set(RGB_FEATURES)
    assert np.isclose(features["rgb_mean_r"], 1.0)
    assert np.isclose(features["rgb_mean_g"], 64 / 255)
    assert np.isclose(features["rgb_mean_b"], 0.0)
    assert all(np.isfinite(value) for value in features.values())


def test_classify_condition_dead_positive_wins_regardless_of_canopy_overlap():
    label, name = classify_condition(
        is_dead_positive=True, dead_overlap_pred=0.0, treecover_overlap_pred=1.0
    )
    assert (label, name) == (1, "dead")


def test_classify_condition_significant_dead_overlap_is_ambiguous_not_alive():
    label, name = classify_condition(
        is_dead_positive=False,
        dead_overlap_pred=CLEAN_NEGATIVE_OVERLAP,
        treecover_overlap_pred=1.0,
    )
    assert (label, name) == (None, "ambiguous_dead_overlap")


def test_classify_condition_clear_of_deadwood_and_mostly_canopy_is_alive():
    label, name = classify_condition(
        is_dead_positive=False,
        dead_overlap_pred=0.0,
        treecover_overlap_pred=ALIVE_MIN_TREECOVER_OVERLAP,
    )
    assert (label, name) == (0, "alive")


def test_classify_condition_neither_canopy_nor_dead_is_background():
    label, name = classify_condition(
        is_dead_positive=False,
        dead_overlap_pred=0.0,
        treecover_overlap_pred=ALIVE_MIN_TREECOVER_OVERLAP - 0.01,
    )
    assert (label, name) == (None, "background")
