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

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import json
import math
import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "finger_count_model.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "feature_scaler.npz"
HAND_LANDMARKER_PATH = PROJECT_ROOT / "hand_landmarker.task"
CAPTURE_DIR = PROJECT_ROOT / "outputs" / "final_camera"
WINDOW_NAME = "Smart Camera"

ROLLING_WINDOW_SIZE = 8
MIN_STABLE_VOTES = 4
MIN_CONFIDENCE = 0.55
CAPTURE_COOLDOWN_SECONDS = 1.0
FLASH_DURATION_SECONDS = 0.18
FRAME_TIME_MS = 33
LABELS = [0, 1, 2, 3, 4, 5]


@dataclass
class StablePrediction:
    label: Optional[int]
    confidence: float
    votes: int


@dataclass
class CountdownState:
    label: int
    duration_seconds: int
    started_at: float
    confidence: float


@dataclass
class CaptureRequest:
    trigger: str
    gesture_label: Optional[int]
    countdown_seconds: Optional[int]
    confidence: Optional[float]


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


def load_model_and_scaler():
    """Load the trained classifier and feature scaling statistics."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = np.load(SCALER_PATH)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    std[std == 0] = 1.0
    return model, mean, std


def build_detector() -> HandLandmarker:
    """Create the MediaPipe hand detector for webcam frames."""
    if not HAND_LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"Missing MediaPipe task file: {HAND_LANDMARKER_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return vision.HandLandmarker.create_from_options(options)


def flatten_landmarks(detection_result) -> Optional[np.ndarray]:
    """Flatten the first detected hand into a 63-value feature vector."""
    if not detection_result.hand_landmarks:
        return None

    features = []
    for landmark in detection_result.hand_landmarks[0]:
        features.extend([landmark.x, landmark.y, landmark.z])
    return np.array(features, dtype=np.float32)


def predict_finger_count(
    features: np.ndarray,
    model: tf.keras.Model,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[int, float]:
    """Predict finger count and confidence from landmark features."""
    scaled = (features - mean) / std
    probabilities = model.predict(scaled.reshape(1, -1), verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return LABELS[predicted_index], float(probabilities[predicted_index])


def resolve_stable_prediction(
    prediction_window: deque[tuple[int, float]],
) -> StablePrediction:
    """Return a stable label only if enough recent votes agree."""
    if not prediction_window:
        return StablePrediction(label=None, confidence=0.0, votes=0)

    vote_counts = Counter(label for label, _ in prediction_window)
    label, votes = vote_counts.most_common(1)[0]
    confidences = [conf for pred_label, conf in prediction_window if pred_label == label]
    average_confidence = float(sum(confidences) / len(confidences))

    if votes < MIN_STABLE_VOTES or average_confidence < MIN_CONFIDENCE:
        return StablePrediction(label=None, confidence=average_confidence, votes=votes)

    return StablePrediction(label=label, confidence=average_confidence, votes=votes)


def draw_landmarks(frame_bgr, detection_result) -> None:
    """Draw hand landmarks in-place on the preview frame."""
    if not detection_result.hand_landmarks:
        return

    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    for hand_landmarks in detection_result.hand_landmarks:
        vision.drawing_utils.draw_landmarks(
            rgb_frame,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )
    frame_bgr[:] = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)


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


def draw_flash_effect(frame_bgr, flash_until: float) -> None:
    """Overlay a short white flash after taking a photo."""
    now = time.time()
    if now >= flash_until:
        return

    remaining_ratio = (flash_until - now) / FLASH_DURATION_SECONDS
    alpha = min(max(0.2 + 0.55 * remaining_ratio, 0.0), 0.75)
    flash_overlay = np.full_like(frame_bgr, 255)
    cv2.addWeighted(flash_overlay, alpha, frame_bgr, 1.0 - alpha, 0.0, frame_bgr)


def draw_panel(
    frame_bgr,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    alpha: float = 0.45,
    fill_color: tuple[int, int, int] = (18, 18, 18),
    border_color: tuple[int, int, int] = (240, 240, 240),
) -> None:
    """Draw a compact translucent camera-style panel."""
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, top_left, bottom_right, fill_color, -1)
    cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0, frame_bgr)
    cv2.rectangle(frame_bgr, top_left, bottom_right, border_color, 1)


def draw_overlay(
    frame,
    raw_prediction: Optional[int],
    raw_confidence: float,
    stable_prediction: StablePrediction,
    active_countdown: CountdownState | None,
    countdown_remaining: Optional[float],
    captures_saved: int,
    fps: float,
    status_text: str,
) -> None:
    """Render a cleaner camera-style overlay without blocking the frame."""
    top_left_bottom = (20, 20), (615, 112)
    draw_panel(frame, top_left_bottom[0], top_left_bottom[1], alpha=0.34)
    cv2.putText(
        frame,
        "SMART CAMERA",
        (36, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status_text,
        (36, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    # right_panel_width = 230
    # right_top_left = (frame.shape[1] - right_panel_width - 20, 20)
    # right_bottom_right = (frame.shape[1] - 20, 148)
    # draw_panel(frame, right_top_left, right_bottom_right, alpha=0.34)

    # stable_text = (
    #     str(stable_prediction.label)
    #     if stable_prediction.label is not None
    #     else "-"
    # )
    # raw_text = str(raw_prediction) if raw_prediction is not None else "-"
    # countdown_text = (
    #     f"{countdown_remaining:.1f}s"
    #     if active_countdown is not None and countdown_remaining is not None
    #     else "Ready"
    # )
    # info_lines = [
    #     f"Gesture: ON",
    #     f"Raw: {raw_text} ({raw_confidence:.2f})",
    #     f"Stable: {stable_text}  votes {stable_prediction.votes}/{ROLLING_WINDOW_SIZE}",
    #     f"Countdown: {countdown_text}",
    #     f"Shots: {captures_saved}   FPS: {fps:.1f}",
    # ]
    # for idx, line in enumerate(info_lines):
    #     y = 48 + idx * 22
    #     cv2.putText(
    #         frame,
    #         line,
    #         (right_top_left[0] + 16, y),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.55,
    #         (255, 255, 255),
    #         1,
    #         cv2.LINE_AA,
    #     )

    # bottom_panel_width = 420
    # bottom_panel_height = 42
    # bottom_top_left = ((frame.shape[1] - bottom_panel_width) // 2, frame.shape[0] - 68)
    # bottom_bottom_right = (
    #     bottom_top_left[0] + bottom_panel_width,
    #     bottom_top_left[1] + bottom_panel_height,
    # )
    # draw_panel(frame, bottom_top_left, bottom_bottom_right, alpha=0.34)
    # cv2.putText(
    #     frame,
    #     "SPACE shoot   x cancel countdown   r reset   q quit",
    #     (bottom_top_left[0] + 18, bottom_top_left[1] + 28),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.56,
    #     (255, 255, 255),
    #     1,
    #     cv2.LINE_AA,
    # )

    if active_countdown is not None and countdown_remaining is not None:
        large_text = str(max(0, math.ceil(countdown_remaining)))
        text_size, _ = cv2.getTextSize(large_text, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 8)
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(
            frame,
            large_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            5.0,
            (0, 255, 120),
            8,
            cv2.LINE_AA,
        )

        hint = f"{active_countdown.duration_seconds}-second countdown armed"
        hint_size, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        hint_x = (frame.shape[1] - hint_size[0]) // 2
        hint_y = max(170, text_y - 120)
        draw_panel(
            frame,
            (hint_x - 16, hint_y - 28),
            (hint_x + hint_size[0] + 16, hint_y + 10),
            alpha=0.32,
        )
        cv2.putText(
            frame,
            hint,
            (hint_x, hint_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


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
                        status_text = (
                            f"{stable_prediction.label}-second countdown started. Move into position."
                        )
                elif stable_prediction.label != active_countdown.label:
                    active_countdown = CountdownState(
                        label=stable_prediction.label,
                        duration_seconds=stable_prediction.label,
                        started_at=now,
                        confidence=stable_prediction.confidence,
                    )
                    status_text = (
                        f"Countdown updated to {stable_prediction.label} seconds."
                    )
            elif stable_prediction.label is None and active_countdown is None and features is not None:
                status_text = "Gesture seen, but it is not stable yet."

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

            draw_landmarks(frame, detection_result)

            elapsed_for_fps = time.time() - fps_timer_start
            if elapsed_for_fps >= 1.0:
                fps = fps_frame_counter / elapsed_for_fps
                fps_timer_start = time.time()
                fps_frame_counter = 0

            draw_overlay(
                frame,
                raw_prediction,
                raw_confidence,
                stable_prediction,
                active_countdown,
                countdown_remaining,
                captures_saved,
                fps,
                status_text,
            )
            draw_flash_effect(frame, flash_until)

            if pending_capture is not None:
                filename, preview_name = save_capture(
                    raw_frame,
                    frame,
                    pending_capture,
                    raw_prediction,
                    raw_confidence,
                    stable_prediction,
                )
                captures_saved += 1
                last_capture_time = time.time()
                flash_until = last_capture_time + FLASH_DURATION_SECONDS
                status_text = f"Saved {filename} and {preview_name}."

            cv2.imshow(WINDOW_NAME, frame)

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
                    raw_frame,
                    frame,
                    manual_request,
                    raw_prediction,
                    raw_confidence,
                    stable_prediction,
                )
                captures_saved += 1
                last_capture_time = time.time()
                flash_until = last_capture_time + FLASH_DURATION_SECONDS
                status_text = f"Saved {filename} and {preview_name}."
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
