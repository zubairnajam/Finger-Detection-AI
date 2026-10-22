"""Landmark flattening and HUD rendering (smoke tests at several resolutions)."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import CountdownState, StablePrediction
from src.hand_detection import flatten_landmarks
from src.ui_overlay import (
    Telemetry,
    draw_camera_overlay,
    draw_flash_effect,
    draw_landmarks,
    draw_panel,
)
from tests.conftest import make_detection_result


def test_flatten_returns_63_features_in_xyz_order(detection_result):
    features = flatten_landmarks(detection_result)
    assert features.shape == (63,)
    assert features.dtype == np.float32
    first = detection_result.hand_landmarks[0][0]
    np.testing.assert_allclose(features[:3], [first.x, first.y, first.z], rtol=1e-6)


def test_flatten_uses_first_hand_only():
    result = make_detection_result(num_hands=2)
    assert flatten_landmarks(result).shape == (63,)


def test_flatten_without_hand_returns_none():
    assert flatten_landmarks(SimpleNamespace(hand_landmarks=[])) is None


@pytest.mark.parametrize("size", [(480, 640), (720, 1280), (960, 1280), (1080, 1920), (240, 320)])
@pytest.mark.parametrize("with_countdown", [False, True])
def test_overlay_renders_at_common_resolutions(size, with_countdown, detection_result):
    h, w = size
    frame = np.full((h, w, 3), 90, dtype=np.uint8)
    probs = np.array([0.01, 0.02, 0.9, 0.05, 0.01, 0.01])
    countdown = CountdownState(label=3, duration_seconds=3, started_at=0.0, confidence=0.9)

    draw_landmarks(frame, detection_result)
    draw_camera_overlay(
        frame,
        raw_prediction=2,
        raw_confidence=0.9,
        stable_prediction=StablePrediction(label=2, confidence=0.9, votes=6),
        active_countdown=countdown if with_countdown else None,
        countdown_remaining=1.4 if with_countdown else None,
        captures_saved=3,
        fps=29.7,
        status_text="A very long status message " * 5,
        hand_detected=True,
        probabilities=probs,
        telemetry=Telemetry(fps=29.7, detect_ms=12.3, classify_ms=0.04),
    )
    assert frame.shape == (h, w, 3)
    assert frame.dtype == np.uint8
    assert (frame != 90).any()  # something was drawn


def test_overlay_without_optional_args_is_backward_compatible():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_camera_overlay(
        frame,
        raw_prediction=None,
        raw_confidence=0.0,
        stable_prediction=StablePrediction(label=None, confidence=0.0, votes=0),
        active_countdown=None,
        countdown_remaining=None,
        captures_saved=0,
        fps=0.0,
        status_text="idle",
        hand_detected=False,
    )


def test_hidden_hud_leaves_frame_untouched():
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    draw_camera_overlay(
        frame,
        raw_prediction=None,
        raw_confidence=0.0,
        stable_prediction=StablePrediction(label=None, confidence=0.0, votes=0),
        active_countdown=None,
        countdown_remaining=None,
        captures_saved=0,
        fps=0.0,
        status_text="idle",
        hand_detected=False,
        show_hud=False,
    )
    assert (frame == 50).all()


def test_panel_clips_to_frame_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_panel(frame, (-50, -50), (500, 500), alpha=0.5, corner_radius=20,
               border_color=(255, 255, 255))
    draw_panel(frame, (200, 200), (300, 300))  # fully outside: no-op
    assert frame.shape == (100, 100, 3)


def test_flash_effect_brightens_frame():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    draw_flash_effect(frame, flash_until=time.time() + 0.1)
    assert frame.mean() > 0
