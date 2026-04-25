"""
Real-time finger-count inference with smoothing, countdown, and auto-capture.

Usage:
    python src/live_inference.py

Controls:
    q  Quit
    r  Reset countdown and prediction history
    c  Toggle auto-capture on/off
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import json
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
CAPTURE_DIR = PROJECT_ROOT / "outputs" / "captures"

ROLLING_WINDOW_SIZE = 8
MIN_STABLE_VOTES = 6
MIN_CONFIDENCE = 0.65
COUNTDOWN_SECONDS = 3.0
CAPTURE_COOLDOWN_SECONDS = 2.0
FRAME_TIME_MS = 33
LABELS = [1, 2, 3, 4, 5]


@dataclass
class StablePrediction:
    label: Optional[int]
    confidence: float
    votes: int


def ensure_dirs() -> None:
    """Create runtime output directories."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


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
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
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
    frame_bgr,
    stable_label: int,
    confidence: float,
) -> str:
    """Save an auto-captured image and metadata."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{time.time_ns()}"
    image_path = CAPTURE_DIR / f"{stem}_label_{stable_label}.jpg"
    metadata_path = CAPTURE_DIR / f"{stem}_label_{stable_label}.json"

    cv2.imwrite(str(image_path), frame_bgr)
    metadata = {
        "label": stable_label,
        "confidence": confidence,
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "saved_at": timestamp,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return image_path.name


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
    """Render the current inference state on the webcam preview."""
    countdown_text = (
        f"{countdown_remaining:.1f}s"
        if countdown_remaining is not None
        else "-"
    )

    lines = [
        f"Raw prediction: {raw_prediction if raw_prediction is not None else '-'} ({raw_confidence:.2f})",
        f"Stable prediction: {stable_prediction.label if stable_prediction.label is not None else '-'} "
        f"| votes={stable_prediction.votes}/{ROLLING_WINDOW_SIZE} | conf={stable_prediction.confidence:.2f}",
        f"Countdown: {countdown_text}",
        f"Auto capture: {'ON' if auto_capture_enabled else 'OFF'} | Saved: {captures_saved}",
        f"FPS: {fps:.1f}",
        f"Status: {status_text}",
        "Keys: q quit | r reset | c auto-capture",
    ]

    cv2.rectangle(frame, (8, 8), (632, 220), (25, 25, 25), -1)
    cv2.rectangle(frame, (8, 8), (632, 220), (90, 180, 255), 2)

    for idx, line in enumerate(lines):
        y = 35 + idx * 28
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if stable_prediction.label is not None and countdown_remaining is not None:
        cv2.putText(
            frame,
            str(stable_prediction.label),
            (frame.shape[1] - 140, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            4.0,
            (0, 255, 120),
            6,
            cv2.LINE_AA,
        )


def main() -> None:
    """Run the real-time inference loop."""
    ensure_dirs()
    model, mean, std = load_model_and_scaler()
    detector = build_detector()
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
                raw_prediction, raw_confidence = predict_finger_count(features, model, mean, std)
                prediction_window.append((raw_prediction, raw_confidence))
                status_text = "Hand detected."
            else:
                prediction_window.clear()
                current_countdown_label = None
                countdown_start_time = None
                status_text = "No hand detected."

            stable_prediction = resolve_stable_prediction(prediction_window)
            countdown_remaining: Optional[float] = None

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
                            filename = save_capture(raw_frame, stable_prediction.label, stable_prediction.confidence)
                            captures_saved += 1
                            last_capture_time = time.time()
                            status_text = f"Captured {filename}"
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
                frame,
                raw_prediction,
                raw_confidence,
                stable_prediction,
                countdown_remaining,
                auto_capture_enabled,
                captures_saved,
                fps,
                status_text,
            )
            cv2.imshow("Gesture-Controlled Smart Camera", frame)

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

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
