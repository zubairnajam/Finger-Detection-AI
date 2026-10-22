"""
Gesture-driven camera state machine, independent of OpenCV and the webcam.

Keeping this logic free of I/O makes it unit-testable: tests feed in
synthetic predictions and timestamps and assert on countdowns and captures.

Rules:
    * Hold 1-5 fingers steadily for ARM_HOLD_SECONDS -> start an N-second countdown.
    * Once started, the countdown is locked: other finger counts are ignored
      until the photo is taken.
    * Hold a fist (0) -> cancel the countdown.
    * After a capture, the same gesture will not re-arm until the hand changes
      gesture or leaves the frame (prevents accidental burst shots).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from src.config import (
    ARM_HOLD_SECONDS,
    CAPTURE_COOLDOWN_SECONDS,
    MIN_CONFIDENCE,
    MIN_STABLE_VOTES,
    ROLLING_WINDOW_SIZE,
    CaptureRequest,
    CountdownState,
    StablePrediction,
)
from src.prediction import resolve_stable_prediction

COUNTDOWN_LABELS = frozenset({1, 2, 3, 4, 5})


@dataclass
class FrameUpdate:
    """Everything the UI needs to know after processing one frame."""
    stable_prediction: StablePrediction
    countdown_remaining: Optional[float]
    capture_request: Optional[CaptureRequest]


class GestureController:
    """Turns a stream of per-frame predictions into countdowns and captures."""

    def __init__(
        self,
        *,
        window_size: int = ROLLING_WINDOW_SIZE,
        min_votes: int = MIN_STABLE_VOTES,
        min_confidence: float = MIN_CONFIDENCE,
        cooldown_seconds: float = CAPTURE_COOLDOWN_SECONDS,
        arm_hold_seconds: float = ARM_HOLD_SECONDS,
    ) -> None:
        self.window: deque[tuple[int, float]] = deque(maxlen=window_size)
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self.arm_hold_seconds = arm_hold_seconds

        self.active_countdown: Optional[CountdownState] = None
        # Label currently being held towards arming, and when it was first seen.
        self.arming_label: Optional[int] = None
        self.arming_since = 0.0
        self.last_committed_label: Optional[int] = None
        self.last_capture_time = 0.0
        self.status_text = "Press SPACE for an instant photo or hold up 1-5 fingers for a countdown."

    # ── Per-frame update ────────────────────────────────────────────────
    def update(self, prediction: Optional[tuple[int, float]], now: float) -> FrameUpdate:
        """Advance the state machine by one frame.

        ``prediction`` is ``(label, confidence)`` when a hand was classified,
        or ``None`` when no hand is visible.
        """
        if prediction is not None:
            self.window.append(prediction)
            if self.active_countdown is None:
                self.status_text = "Gesture detected. Hold steady to arm the countdown."
        else:
            self.window.clear()
            if self.active_countdown is None:
                self.last_committed_label = None
                self.status_text = "Show 1-5 fingers for countdown, or press SPACE to snap."

        stable = resolve_stable_prediction(
            self.window, min_votes=self.min_votes, min_confidence=self.min_confidence,
        )
        self._apply_gesture(stable, hand_visible=prediction is not None, now=now)
        remaining, request = self._tick_countdown(now)
        return FrameUpdate(stable, remaining, request)

    def _apply_gesture(self, stable: StablePrediction, *, hand_visible: bool, now: float) -> None:
        if stable.label not in COUNTDOWN_LABELS or self.active_countdown is not None:
            self.arming_label = None

        if stable.label == 0:
            self.last_committed_label = None
            if self.active_countdown is not None:
                self.active_countdown = None
                self.status_text = "Countdown cancelled by 0-finger gesture."
        elif stable.label in COUNTDOWN_LABELS:
            # A running countdown is locked; only a fist or a key can stop it.
            if self.active_countdown is None and stable.label != self.last_committed_label:
                if stable.label != self.arming_label:
                    self.arming_label = stable.label
                    self.arming_since = now
                if now - self.arming_since >= self.arm_hold_seconds:
                    self.arming_label = None
                    self._start_countdown(stable, now)
                    self.status_text = f"{stable.label}-second countdown locked in."
                else:
                    self.status_text = f"Hold {stable.label} finger(s) steady..."
        elif stable.label is None and self.active_countdown is None and hand_visible:
            self.status_text = "Gesture seen, but it is not stable yet."

    def _start_countdown(self, stable: StablePrediction, now: float) -> None:
        self.active_countdown = CountdownState(
            label=stable.label,
            duration_seconds=stable.label,
            started_at=now,
            confidence=stable.confidence,
        )

    def _tick_countdown(self, now: float) -> tuple[Optional[float], Optional[CaptureRequest]]:
        countdown = self.active_countdown
        if countdown is None:
            return None, None

        remaining = max(0.0, countdown.duration_seconds - (now - countdown.started_at))
        if remaining > 0.0:
            return remaining, None

        if now - self.last_capture_time < self.cooldown_seconds:
            self.status_text = "Camera cooldown active. Hold your pose."
            return 0.0, None

        request = CaptureRequest(
            trigger="gesture_countdown",
            gesture_label=countdown.label,
            countdown_seconds=countdown.duration_seconds,
            confidence=countdown.confidence,
        )
        self.last_committed_label = countdown.label
        self.active_countdown = None
        self.window.clear()
        self.status_text = "Gesture countdown complete. Capturing photo."
        return None, request

    # ── User actions ────────────────────────────────────────────────────
    def reset(self, status_text: str = "Gesture state reset.") -> None:
        """Drop any countdown and prediction history."""
        self.active_countdown = None
        self.window.clear()
        self.last_committed_label = None
        self.arming_label = None
        self.status_text = status_text

    def cancel(self) -> None:
        self.reset("Countdown cancelled.")

    def manual_capture_request(self) -> CaptureRequest:
        """Reset state and build a request for an instant (shutter-key) photo."""
        self.reset(self.status_text)
        return CaptureRequest(
            trigger="manual_capture",
            gesture_label=None,
            countdown_seconds=None,
            confidence=None,
        )

    def mark_captured(self, now: float, filename: str) -> None:
        """Record that a photo was written to disk."""
        self.last_capture_time = now
        self.status_text = f"Saved {filename}."
