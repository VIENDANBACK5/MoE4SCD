import numpy as np

from deadtrees_pipeline.classify_objects import (
    RGB_FEATURES,
    SHAPE_FEATURES,
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
