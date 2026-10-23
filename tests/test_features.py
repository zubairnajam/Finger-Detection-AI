"""Feature builder: invariance to where the hand is in the frame."""

import numpy as np
import pytest

from src.features import FEATURE_SIZE, build_features, image_aspects, mirror_landmarks


@pytest.fixture
def hand():
    rng = np.random.default_rng(1)
    points = rng.uniform(0.3, 0.6, size=(21, 3)).astype(np.float32)
    points[:, 2] -= 0.45
    return points.ravel()


def _move(raw, dx, dy, scale):
    points = raw.reshape(21, 3).copy()
    centre = points[0, :2].copy()
    points[:, :2] = (points[:, :2] - centre) * scale + centre + (dx, dy)
    points[:, 2] *= scale
    return points.ravel()


def test_output_shape(hand):
    assert build_features(hand).shape == (FEATURE_SIZE,)
    assert build_features(np.stack([hand, hand])).shape == (2, FEATURE_SIZE)


def test_wrist_is_origin_and_scale_is_unit(hand):
    points = build_features(hand)[:63].reshape(21, 3)
    np.testing.assert_allclose(points[0], 0, atol=1e-6)
    assert np.linalg.norm(points[9, :2]) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("dx,dy,scale", [(0.2, -0.1, 1.0), (-0.15, 0.2, 0.6), (0.05, 0.05, 1.4)])
def test_invariant_to_position_and_size(hand, dx, dy, scale):
    np.testing.assert_allclose(build_features(_move(hand, dx, dy, scale)), build_features(hand), atol=1e-4)


def test_aspect_per_row_matches_scalar(hand):
    batch = build_features(np.stack([hand, hand]), np.array([1.0, 4 / 3]))
    np.testing.assert_allclose(batch[1], build_features(hand, 4 / 3), atol=1e-6)


def test_mirror_keeps_finger_shape(hand):
    # Bend angles and distances describe shape only, so a mirrored hand matches.
    np.testing.assert_allclose(build_features(mirror_landmarks(hand))[63:], build_features(hand)[63:], atol=1e-5)


def test_image_aspects_by_source():
    np.testing.assert_allclose(
        image_aspects([r"data\raw\3\a.jpg", "archive/fingers/test/b_3R.png"]), [640 / 480, 1.0])
