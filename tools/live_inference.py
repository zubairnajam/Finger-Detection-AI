"""
Legacy real-time finger-count prototype with smoothing and auto-capture.

Usage:
    python tools/live_inference.py

Controls:
    q  Quit
    r  Reset countdown and prediction history
    c  Toggle auto-capture on/off
    s  Save a manual screenshot/capture
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

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    FRAME_TIME_MS,
    LEGACY_CAPTURE_DIR,
    PROJECT_ROOT,
    ROLLING_WINDOW_SIZE,
    StablePrediction,
)
from src.hand_detection import build_detector, flatten_landmarks
from src.prediction import load_model_and_scaler, predict_finger_count, resolve_stable_prediction
from src.ui_overlay import draw_landmarks, draw_panel, draw_text

WINDOW_NAME = "Legacy Gesture-Controlled Prototype"
COUNTDOWN_SECONDS = 3.0
CAPTURE_COOLDOWN_SECONDS = 2.0

# Stricter stabilisation thresholds for legacy mode. (These used to be
# monkey-patched onto src.config, which had no effect because prediction.py
# had already imported the values; they are now passed explicitly.)
LEGACY_MIN_STABLE_VOTES = 6
LEGACY_MIN_CONFIDENCE = 0.65


def ensure_dirs() -> None:
    LEGACY_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


def setup_window(window_name: str) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass


def save_capture(
    raw_frame_bgr,
    preview_frame_bgr,
    stable_label: int,
    confidence: float,
) -> tuple[str, str]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{time.time_ns()}"
    image_path = LEGACY_CAPTURE_DIR / f"{stem}_label_{stable_label}.jpg"
    screenshot_path = LEGACY_CAPTURE_DIR / f"{stem}_label_{stable_label}_screenshot.jpg"
    metadata_path = LEGACY_CAPTURE_DIR / f"{stem}_label_{stable_label}.json"

    cv2.imwrite(str(image_path), raw_frame_bgr)
    cv2.imwrite(str(screenshot_path), preview_frame_bgr)
    metadata = {
        "label": stable_label,
        "confidence": confidence,
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "screenshot_path": str(screenshot_path.relative_to(PROJECT_ROOT)),
        "saved_at": timestamp,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return image_path.name, screenshot_path.name


def draw_overlay(
    frame,
    raw_prediction: Optional[int],
    raw_confidence: float,
    stable_prediction: StablePrediction,
    countdown_remaining: Optional[float],
    auto_capture_enabled: bool,
    captures_saved: int,
    fps: float,
    status_text: str,
) -> None:
    countdown_text = f"{countdown_remaining:.1f}s" if countdown_remaining is not None else "-"
    lines = [
        f"Raw prediction: {raw_prediction if raw_prediction is not None else '-'} ({raw_confidence:.2f})",
        f"Stable prediction: {stable_prediction.label if stable_prediction.label is not None else '-'} "
        f"| votes={stable_prediction.votes}/{ROLLING_WINDOW_SIZE} | conf={stable_prediction.confidence:.2f}",
        f"Countdown: {countdown_text}",
        f"Auto capture: {'ON' if auto_capture_enabled else 'OFF'} | Saved: {captures_saved}",
        f"FPS: {fps:.1f}",
        f"Status: {status_text}",
        "Keys: q quit | r reset | c auto-capture | s screenshot",
    ]

    draw_panel(frame, (8, 8), (632, 220), alpha=0.55, border_color=(90, 180, 255))

    for idx, line in enumerate(lines):
        y = 35 + idx * 28
        draw_text(frame, line, (20, y), scale=0.62, thickness=2)

    if stable_prediction.label is not None and countdown_remaining is not None:
        draw_text(
            frame, str(stable_prediction.label),
            (frame.shape[1] - 140, 140), scale=4.0, color=(0, 255, 120), thickness=6,
        )


def main() -> None:
    ensure_dirs()
    setup_window(WINDOW_NAME)
    model, mean, std = load_model_and_scaler()
    detector = build_detector(min_detection_confidence=0.5, min_presence_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Could not open webcam.")

    prediction_window: deque[tuple[int, float]] = deque(maxlen=ROLLING_WINDOW_SIZE)
    current_countdown_label: Optional[int] = None
    countdown_start_time: Optional[float] = None
    last_capture_time = 0.0
    auto_capture_enabled = True
    captures_saved = 0
    status_text = "Show one hand to the camera."
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

            raw_prediction = None
            raw_confidence = 0.0
            features = flatten_landmarks(detection_result)
            if features is not None:
                aspect = raw_frame.shape[1] / raw_frame.shape[0]
                raw_prediction, raw_confidence = predict_finger_count(features, model, mean, std, aspect)
                prediction_window.append((raw_prediction, raw_confidence))
                status_text = "Hand detected."
            else:
                prediction_window.clear()
                current_countdown_label = None
                countdown_start_time = None
                status_text = "No hand detected."

            stable_prediction = resolve_stable_prediction(
                prediction_window,
                min_votes=LEGACY_MIN_STABLE_VOTES,
                min_confidence=LEGACY_MIN_CONFIDENCE,
            )
            countdown_remaining: Optional[float] = None
            pending_auto_capture: tuple[int, float] | None = None

            if stable_prediction.label is not None:
                if current_countdown_label != stable_prediction.label:
                    current_countdown_label = stable_prediction.label
                    countdown_start_time = time.time()
                    status_text = f"Stable {stable_prediction.label} detected. Countdown started."

                if countdown_start_time is not None:
                    elapsed = time.time() - countdown_start_time
                    countdown_remaining = max(0.0, COUNTDOWN_SECONDS - elapsed)

                    if countdown_remaining <= 0.0:
                        ready_for_capture = time.time() - last_capture_time >= CAPTURE_COOLDOWN_SECONDS
                        if auto_capture_enabled and ready_for_capture:
                            pending_auto_capture = (stable_prediction.label, stable_prediction.confidence)
                            status_text = "Stable prediction captured."
                        elif not auto_capture_enabled:
                            status_text = "Stable prediction ready, but auto-capture is OFF."
                        else:
                            status_text = "Capture cooldown active."

                        prediction_window.clear()
                        current_countdown_label = None
                        countdown_start_time = None
                        countdown_remaining = None
            else:
                current_countdown_label = None
                countdown_start_time = None
                if features is not None:
                    status_text = "Waiting for a stable prediction."

            draw_landmarks(frame, detection_result)

            elapsed_for_fps = time.time() - fps_timer_start
            if elapsed_for_fps >= 1.0:
                fps = fps_frame_counter / elapsed_for_fps
                fps_timer_start = time.time()
                fps_frame_counter = 0

            draw_overlay(
                frame, raw_prediction, raw_confidence, stable_prediction,
                countdown_remaining, auto_capture_enabled, captures_saved, fps, status_text,
            )

            if pending_auto_capture is not None:
                capture_label, capture_confidence = pending_auto_capture
                filename, screenshot_name = save_capture(raw_frame, frame, capture_label, capture_confidence)
                captures_saved += 1
                last_capture_time = time.time()
                status_text = f"Captured {filename} and {screenshot_name}"

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                prediction_window.clear()
                current_countdown_label = None
                countdown_start_time = None
                status_text = "Prediction state reset."
            if key == ord("c"):
                auto_capture_enabled = not auto_capture_enabled
                status_text = f"Auto-capture {'enabled' if auto_capture_enabled else 'disabled'}."
            if key == ord("s"):
                if stable_prediction.label is None:
                    status_text = "Manual capture requires a stable finger prediction."
                else:
                    filename, screenshot_name = save_capture(
                        raw_frame, frame, stable_prediction.label, stable_prediction.confidence,
                    )
                    captures_saved += 1
                    last_capture_time = time.time()
                    status_text = f"Manual capture saved {filename} and {screenshot_name}."

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
