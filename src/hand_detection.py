"""
MediaPipe hand detection utilities shared by the camera app and tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import HAND_LANDMARKER_PATH

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker


def build_detector(
    *,
    running_mode: str = "VIDEO",
    min_detection_confidence: float = 0.35,
    min_presence_confidence: float = 0.35,
    min_tracking_confidence: float = 0.35,
) -> "HandLandmarker":
    """Create a MediaPipe HandLandmarker with the given confidence thresholds.

    Parameters
    ----------
    running_mode : str
        "VIDEO" for webcam frames (default) or "IMAGE" for still images.
    """
    if not HAND_LANDMARKER_PATH.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe task file: {HAND_LANDMARKER_PATH}"
        )

    mode_map = {"VIDEO": vision.RunningMode.VIDEO, "IMAGE": vision.RunningMode.IMAGE}
    mode = mode_map.get(running_mode, vision.RunningMode.VIDEO)

    base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mode,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def flatten_landmarks(detection_result) -> Optional[np.ndarray]:
    """Flatten the first detected hand into a 63-value feature vector."""
    if not detection_result.hand_landmarks:
        return None

    features: list[float] = []
    for landmark in detection_result.hand_landmarks[0]:
        features.extend([landmark.x, landmark.y, landmark.z])
    return np.array(features, dtype=np.float32)
