"""Gesture -> countdown -> capture state machine."""

import pytest

from src.config import ROLLING_WINDOW_SIZE
from src.gesture_controller import GestureController


def feed(controller, label, frames, start, confidence=0.95, dt=0.033):
    """Push `frames` identical predictions; return (last update, end time)."""
    now = start
    update = None
    for _ in range(frames):
        prediction = None if label is None else (label, confidence)
        update = controller.update(prediction, now=now)
        now += dt
    return update, now


@pytest.fixture
def controller():
    # Arm instantly so tests can focus on countdown logic; arm-hold has its own tests.
    return GestureController(cooldown_seconds=1.0, arm_hold_seconds=0.0)


def test_no_hand_does_nothing(controller):
    update, _ = feed(controller, None, 10, start=0.0)
    assert update.stable_prediction.label is None
    assert update.countdown_remaining is None
    assert update.capture_request is None
    assert controller.active_countdown is None


def test_stable_gesture_starts_matching_countdown(controller):
    update, _ = feed(controller, 3, ROLLING_WINDOW_SIZE, start=10.0)
    assert controller.active_countdown is not None
    assert controller.active_countdown.duration_seconds == 3
    assert 0.0 < update.countdown_remaining <= 3.0


def test_countdown_expires_into_capture(controller):
    _, now = feed(controller, 2, ROLLING_WINDOW_SIZE, start=10.0)
    update = controller.update((2, 0.95), now=now + 2.5)
    assert update.capture_request is not None
    assert update.capture_request.trigger == "gesture_countdown"
    assert update.capture_request.gesture_label == 2
    assert update.capture_request.countdown_seconds == 2
    assert controller.active_countdown is None
    assert len(controller.window) == 0


def test_same_gesture_does_not_rearm_after_capture(controller):
    _, now = feed(controller, 1, ROLLING_WINDOW_SIZE, start=10.0)
    assert controller.update((1, 0.95), now=now + 1.5).capture_request is not None
    controller.mark_captured(now + 1.5, "x.jpg")

    feed(controller, 1, ROLLING_WINDOW_SIZE * 2, start=now + 1.6)
    assert controller.active_countdown is None


def test_gesture_rearms_after_hand_leaves(controller):
    _, now = feed(controller, 1, ROLLING_WINDOW_SIZE, start=10.0)
    controller.update((1, 0.95), now=now + 1.5)
    controller.mark_captured(now + 1.5, "x.jpg")

    _, now = feed(controller, None, 3, start=now + 2.0)
    feed(controller, 1, ROLLING_WINDOW_SIZE, start=now)
    assert controller.active_countdown is not None


def test_countdown_is_locked_against_other_gestures(controller):
    _, now = feed(controller, 5, ROLLING_WINDOW_SIZE, start=10.0)
    started_at = controller.active_countdown.started_at
    for label in (2, 3, 4, 1):
        _, now = feed(controller, label, ROLLING_WINDOW_SIZE, start=now)
    assert controller.active_countdown.label == 5
    assert controller.active_countdown.duration_seconds == 5
    assert controller.active_countdown.started_at == started_at


def test_locked_countdown_still_captures_with_original_label(controller):
    _, now = feed(controller, 3, ROLLING_WINDOW_SIZE, start=10.0)
    feed(controller, 1, ROLLING_WINDOW_SIZE, start=now)
    update = controller.update((1, 0.95), now=now + 3.5)
    assert update.capture_request is not None
    assert update.capture_request.gesture_label == 3


def test_gesture_must_be_held_before_arming():
    controller = GestureController(arm_hold_seconds=0.6)
    _, now = feed(controller, 3, ROLLING_WINDOW_SIZE, start=10.0)  # ~0.26 s
    assert controller.active_countdown is None
    assert "hold" in controller.status_text.lower()
    feed(controller, 3, 15, start=now)  # another ~0.5 s
    assert controller.active_countdown is not None
    assert controller.active_countdown.label == 3


def test_transient_counts_while_raising_hand_do_not_arm():
    controller = GestureController(arm_hold_seconds=0.6)
    now = 10.0
    # Fingers coming up one by one, each visible for ~0.3 s.
    for label in (1, 2, 3, 4):
        _, now = feed(controller, label, 9, start=now)
        assert controller.active_countdown is None
    feed(controller, 5, 30, start=now)
    assert controller.active_countdown.label == 5


def test_fist_cancels_countdown(controller):
    _, now = feed(controller, 4, ROLLING_WINDOW_SIZE, start=10.0)
    assert controller.active_countdown is not None
    statuses = []
    for _ in range(ROLLING_WINDOW_SIZE):
        controller.update((0, 0.95), now=now)
        statuses.append(controller.status_text)
        now += 0.033
    assert controller.active_countdown is None
    assert any("cancelled" in s.lower() for s in statuses)


def test_countdown_survives_hand_leaving_frame(controller):
    _, now = feed(controller, 3, ROLLING_WINDOW_SIZE, start=10.0)
    update, _ = feed(controller, None, 5, start=now)
    assert controller.active_countdown is not None
    assert update.countdown_remaining is not None


def test_cooldown_blocks_back_to_back_capture(controller):
    controller.mark_captured(100.0, "manual.jpg")
    _, now = feed(controller, 1, ROLLING_WINDOW_SIZE, start=99.0)
    # Countdown would expire at ~100.2 but cooldown lasts until 101.0
    update = controller.update((1, 0.95), now=100.5)
    assert update.capture_request is None
    assert update.countdown_remaining == 0.0
    assert "cooldown" in controller.status_text.lower()

    update = controller.update((1, 0.95), now=101.1)
    assert update.capture_request is not None


def test_low_confidence_never_triggers(controller):
    feed(controller, 3, ROLLING_WINDOW_SIZE * 3, start=0.0, confidence=0.3)
    assert controller.active_countdown is None


def test_manual_capture_resets_state(controller):
    feed(controller, 3, ROLLING_WINDOW_SIZE, start=10.0)
    request = controller.manual_capture_request()
    assert request.trigger == "manual_capture"
    assert request.gesture_label is None
    assert controller.active_countdown is None
    assert len(controller.window) == 0


def test_cancel_and_reset(controller):
    feed(controller, 3, ROLLING_WINDOW_SIZE, start=10.0)
    controller.cancel()
    assert controller.active_countdown is None
    assert controller.status_text == "Countdown cancelled."

    feed(controller, 3, ROLLING_WINDOW_SIZE, start=20.0)
    controller.reset()
    assert controller.active_countdown is None
    assert controller.last_committed_label is None
