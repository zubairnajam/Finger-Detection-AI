"""
Final deliverable camera app with manual shutter and gesture-driven countdown.

Usage:
    python src/final_camera_app.py

Controls:
    SPACE / s  Take an instant photo like a normal camera
    x          Cancel the active countdown
    r          Reset gesture history and countdown state
    q          Quit

Gesture mode:
    Hold up 1-5 fingers steadily to start a countdown for that many seconds.
    Hold up 0 fingers to cancel the current countdown.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Optional
import json
import time

import cv2
import mediapipe as mp
import numpy as np

# Ensure project root is on sys.path so `src.*` imports work when invoked directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CAPTURE_COOLDOWN_SECONDS,
    CAPTURE_DIR,
    FLASH_DURATION_SECONDS,
    FRAME_TIME_MS,
    PROJECT_ROOT,
    ROLLING_WINDOW_SIZE,
    WINDOW_NAME,
    CaptureRequest,
    CountdownState,
    StablePrediction,
)
from src.hand_detection import build_detector, flatten_landmarks
from src.prediction import load_model_and_scaler, predict_finger_count, resolve_stable_prediction
from src.ui_overlay import draw_camera_overlay, draw_flash_effect, draw_landmarks


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create runtime output directories."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


def setup_window(window_name: str) -> None:
    """Open the camera preview window in full-screen mode when supported."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass


def save_capture(
    raw_frame_bgr,
    preview_frame_bgr,
    request: CaptureRequest,
    raw_prediction: Optional[int],
    raw_confidence: float,
    stable_prediction: StablePrediction,
) -> tuple[str, str]:
    """Save both the clean photo and the annotated preview with metadata."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{time.time_ns()}_{request.trigger}"
    image_path = CAPTURE_DIR / f"{stem}.jpg"
    preview_path = CAPTURE_DIR / f"{stem}_preview.jpg"
    metadata_path = CAPTURE_DIR / f"{stem}.json"

    cv2.imwrite(str(image_path), raw_frame_bgr)
    cv2.imwrite(str(preview_path), preview_frame_bgr)

    metadata = {
        "trigger": request.trigger,
        "gesture_label": request.gesture_label,
        "countdown_seconds": request.countdown_seconds,
        "request_confidence": request.confidence,
        "raw_prediction": raw_prediction,
        "raw_confidence": raw_confidence,
        "stable_prediction": stable_prediction.label,
        "stable_confidence": stable_prediction.confidence,
        "stable_votes": stable_prediction.votes,
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "preview_path": str(preview_path.relative_to(PROJECT_ROOT)),
        "saved_at": timestamp,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return image_path.name, preview_path.name


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the final deliverable camera app."""
    ensure_dirs()
    setup_window(WINDOW_NAME)
    model, mean, std = load_model_and_scaler()
    detector = build_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Could not open webcam.")

    prediction_window: deque[tuple[int, float]] = deque(maxlen=ROLLING_WINDOW_SIZE)
    active_countdown: CountdownState | None = None
    last_committed_gesture_label: Optional[int] = None
    last_capture_time = 0.0
    flash_until = 0.0
    captures_saved = 0
    status_text = "Press SPACE for an instant photo or hold up 1-5 fingers for a countdown."
    frame_count = 0
    fps = 0.0
    fps_timer_start = time.time()
    fps_frame_counter = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                status_text = "Could not read webcam frame."
                break

            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()
            frame_count += 1
            fps_frame_counter += 1

            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect_for_video(mp_image, frame_count * FRAME_TIME_MS)

            hand_detected = bool(detection_result.hand_landmarks)
            raw_prediction = None
            raw_confidence = 0.0
            features = flatten_landmarks(detection_result)

            if features is not None:
                raw_prediction, raw_confidence = predict_finger_count(features, model, mean, std)
                prediction_window.append((raw_prediction, raw_confidence))
                if active_countdown is None:
                    status_text = "Gesture detected. Hold steady to arm the countdown."
            else:
                prediction_window.clear()
                if active_countdown is None:
                    last_committed_gesture_label = None
                    status_text = "Show 1-5 fingers for countdown, or press SPACE to snap."

            stable_prediction = resolve_stable_prediction(prediction_window)
            pending_capture: CaptureRequest | None = None
            countdown_remaining: Optional[float] = None
            now = time.time()

            # ── Gesture state machine ───────────────────────────────────
            if stable_prediction.label == 0:
                last_committed_gesture_label = None
                if active_countdown is not None:
                    active_countdown = None
                    status_text = "Countdown cancelled by 0-finger gesture."
            elif stable_prediction.label in {1, 2, 3, 4, 5}:
                if active_countdown is None:
                    if stable_prediction.label != last_committed_gesture_label:
                        active_countdown = CountdownState(
                            label=stable_prediction.label,
                            duration_seconds=stable_prediction.label,
                            started_at=now,
                            confidence=stable_prediction.confidence,
                        )
                        status_text = f"{stable_prediction.label}-second countdown started."
                elif stable_prediction.label != active_countdown.label:
                    active_countdown = CountdownState(
                        label=stable_prediction.label,
                        duration_seconds=stable_prediction.label,
                        started_at=now,
                        confidence=stable_prediction.confidence,
                    )
                    status_text = f"Countdown updated to {stable_prediction.label} seconds."
            elif stable_prediction.label is None and active_countdown is None and features is not None:
                status_text = "Gesture seen, but it is not stable yet."

            # ── Countdown logic ─────────────────────────────────────────
            if active_countdown is not None:
                elapsed = now - active_countdown.started_at
                countdown_remaining = max(0.0, active_countdown.duration_seconds - elapsed)
                if countdown_remaining <= 0.0:
                    ready_for_capture = now - last_capture_time >= CAPTURE_COOLDOWN_SECONDS
                    if ready_for_capture:
                        pending_capture = CaptureRequest(
                            trigger="gesture_countdown",
                            gesture_label=active_countdown.label,
                            countdown_seconds=active_countdown.duration_seconds,
                            confidence=active_countdown.confidence,
                        )
                        last_committed_gesture_label = active_countdown.label
                        active_countdown = None
                        prediction_window.clear()
                        countdown_remaining = None
                        status_text = "Gesture countdown complete. Capturing photo."
                    else:
                        countdown_remaining = 0.0
                        status_text = "Camera cooldown active. Hold your pose."

            # ── Drawing ─────────────────────────────────────────────────
            draw_landmarks(frame, detection_result)

            elapsed_for_fps = time.time() - fps_timer_start
            if elapsed_for_fps >= 1.0:
                fps = fps_frame_counter / elapsed_for_fps
                fps_timer_start = time.time()
                fps_frame_counter = 0

            draw_camera_overlay(
                frame,
                raw_prediction=raw_prediction,
                raw_confidence=raw_confidence,
                stable_prediction=stable_prediction,
                active_countdown=active_countdown,
                countdown_remaining=countdown_remaining,
                captures_saved=captures_saved,
                fps=fps,
                status_text=status_text,
                hand_detected=hand_detected,
            )
            draw_flash_effect(frame, flash_until)

            # ── Capture ─────────────────────────────────────────────────
            if pending_capture is not None:
                filename, preview_name = save_capture(
                    raw_frame, frame, pending_capture,
                    raw_prediction, raw_confidence, stable_prediction,
                )
                captures_saved += 1
                last_capture_time = time.time()
                flash_until = last_capture_time + FLASH_DURATION_SECONDS
                status_text = f"Saved {filename}."

            cv2.imshow(WINDOW_NAME, frame)

            # ── Keyboard ────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord(" "), ord("s")):
                active_countdown = None
                prediction_window.clear()
                last_committed_gesture_label = None
                manual_request = CaptureRequest(
                    trigger="manual_capture",
                    gesture_label=None,
                    countdown_seconds=None,
                    confidence=None,
                )
                filename, preview_name = save_capture(
                    raw_frame, frame, manual_request,
                    raw_prediction, raw_confidence, stable_prediction,
                )
                captures_saved += 1
                last_capture_time = time.time()
                flash_until = last_capture_time + FLASH_DURATION_SECONDS
                status_text = f"Saved {filename}."
                continue
            if key == ord("x"):
                active_countdown = None
                prediction_window.clear()
                last_committed_gesture_label = None
                status_text = "Countdown cancelled."
                continue
            if key == ord("r"):
                active_countdown = None
                prediction_window.clear()
                last_committed_gesture_label = None
                status_text = "Gesture state reset."
                continue

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
