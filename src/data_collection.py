"""
Webcam-based data collection tool for finger-count classification.

The script saves the clean raw webcam frame, not the annotated preview, so the
dataset can be reprocessed later. It also stores a sidecar JSON file containing
the detected landmarks for that exact frame.

Usage:
    python src/data_collection.py

Controls:
    1-5  Select the current finger-count label
    s    Save the current frame for the selected label
    a    Toggle auto-capture mode
    c    Clear the selected label
    q    Quit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import time
import json

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
VALID_LABELS = ("1", "2", "3", "4", "5")
AUTO_CAPTURE_INTERVAL_SECONDS = 0.7


@dataclass
class SaveResult:
    ok: bool
    message: str


def ensure_directories() -> None:
    """Create the expected raw-data folders if they do not already exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for label in VALID_LABELS:
        (RAW_DATA_DIR / label).mkdir(parents=True, exist_ok=True)


def build_detector() -> vision.HandLandmarker:
    """Create a MediaPipe hand landmarker configured for webcam video."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Download hand_landmarker.task first."
        )

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def count_saved_samples() -> Dict[str, int]:
    """Return the current sample counts for each label directory."""
    counts: Dict[str, int] = {}
    for label in VALID_LABELS:
        label_dir = RAW_DATA_DIR / label
        counts[label] = sum(1 for path in label_dir.iterdir() if path.is_file())
    return counts


def draw_landmarks(frame_bgr, detection_result) -> None:
    """Draw detected hand landmarks onto the live frame in-place."""
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

    annotated_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    frame_bgr[:] = annotated_bgr


def draw_overlay(
    frame,
    selected_label: Optional[str],
    auto_mode: bool,
    status_message: str,
    sample_counts: Dict[str, int],
) -> None:
    """Render instructions, capture status, and class counts."""
    lines = [
        f"Selected label: {selected_label or 'None'}",
        f"Auto capture: {'ON' if auto_mode else 'OFF'}",
        "Keys: 1-5 label | s save | a auto | c clear | q quit",
        f"Status: {status_message}",
        "Sample counts: " + "  ".join(f"{label}={sample_counts[label]}" for label in VALID_LABELS),
    ]

    for idx, text in enumerate(lines):
        y = 30 + idx * 28
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def serialize_landmarks(detection_result) -> List[List[Dict[str, float]]]:
    """Convert MediaPipe landmark objects into JSON-serializable dictionaries."""
    serialized: List[List[Dict[str, float]]] = []
    for hand_landmarks in detection_result.hand_landmarks:
        serialized.append(
            [
                {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                for landmark in hand_landmarks
            ]
        )
    return serialized


def save_frame(
    raw_frame,
    label: Optional[str],
    hand_detected: bool,
    detection_result,
) -> SaveResult:
    """Save the current raw frame and a landmark sidecar to the selected label folder."""
    if label is None:
        return SaveResult(False, "Select a label before saving.")

    if not hand_detected:
        return SaveResult(False, "No hand detected. Sample not saved.")

    label_dir = RAW_DATA_DIR / label
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{time.time_ns()}"
    image_path = label_dir / f"{stem}.jpg"
    json_path = label_dir / f"{stem}.json"

    saved = cv2.imwrite(str(image_path), raw_frame)
    if not saved:
        return SaveResult(False, "OpenCV could not write the image file.")

    sidecar = {
        "label": int(label),
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "hand_landmarks": serialize_landmarks(detection_result),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)

    return SaveResult(True, f"Saved sample to data/raw/{label}/{image_path.name}")


def main() -> None:
    """Run the interactive webcam data collection loop."""
    ensure_directories()
    detector = build_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Could not open webcam.")

    selected_label: Optional[str] = None
    auto_mode = False
    last_auto_capture_time = 0.0
    status_message = "Ready. Select a label with keys 1-5."
    frame_count = 0
    sample_counts = count_saved_samples()

    print("Phase 2 data collection started.")
    print("Select a label with keys 1-5, then press 's' to save or 'a' for auto-capture.")
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                status_message = "Webcam frame read failed."
                break

            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()
            frame_count += 1

            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = frame_count * 33
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            hand_detected = bool(detection_result.hand_landmarks)

            draw_landmarks(frame, detection_result)

            current_time = time.time()
            if (
                auto_mode
                and selected_label is not None
                and hand_detected
                and current_time - last_auto_capture_time >= AUTO_CAPTURE_INTERVAL_SECONDS
            ):
                save_result = save_frame(
                    raw_frame,
                    selected_label,
                    hand_detected,
                    detection_result,
                )
                status_message = save_result.message
                if save_result.ok:
                    sample_counts[selected_label] += 1
                    last_auto_capture_time = current_time

            draw_overlay(frame, selected_label, auto_mode, status_message, sample_counts)
            cv2.imshow("Finger Count Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                selected_label = None
                auto_mode = False
                status_message = "Label cleared."
                continue
            if key == ord("a"):
                auto_mode = not auto_mode
                status_message = f"Auto capture {'enabled' if auto_mode else 'disabled'}."
                continue
            if key == ord("s"):
                save_result = save_frame(
                    raw_frame,
                    selected_label,
                    hand_detected,
                    detection_result,
                )
                status_message = save_result.message
                if save_result.ok and selected_label is not None:
                    sample_counts[selected_label] += 1
                continue

            pressed_key = chr(key) if 32 <= key <= 126 else ""
            if pressed_key in VALID_LABELS:
                selected_label = pressed_key
                status_message = f"Selected label {selected_label}."

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
