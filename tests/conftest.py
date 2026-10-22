"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_detection_result(num_hands: int = 1, seed: int = 0) -> SimpleNamespace:
    """Build an object shaped like a MediaPipe HandLandmarkerResult."""
    rng = np.random.default_rng(seed)
    hands = []
    for _ in range(num_hands):
        coords = rng.uniform(0.2, 0.8, size=(21, 3))
        hands.append([SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in coords])
    return SimpleNamespace(hand_landmarks=hands)


@pytest.fixture
def detection_result() -> SimpleNamespace:
    return make_detection_result()


@pytest.fixture(scope="session")
def keras_model():
    """The trained Keras model (skips if TensorFlow or the model is unavailable)."""
    tf = pytest.importorskip("tensorflow")
    from src.config import MODEL_PATH

    if not MODEL_PATH.exists():
        pytest.skip("Trained model not present")
    return tf.keras.models.load_model(MODEL_PATH)
