"""
Model loading, inference, and prediction stabilisation logic.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Iterable, Union

import numpy as np

from src.config import (
    LABELS,
    MIN_CONFIDENCE,
    MIN_STABLE_VOTES,
    MODEL_PATH,
    SCALER_PATH,
    StablePrediction,
)
from src.inference import NumpyMLP, UnsupportedModelError

if TYPE_CHECKING:
    import tensorflow as tf

    Classifier = Union[tf.keras.Model, NumpyMLP]

logger = logging.getLogger(__name__)

BACKENDS = ("numpy", "keras")


def load_model_and_scaler(
    backend: str = "numpy",
) -> tuple["Classifier", np.ndarray, np.ndarray]:
    """Load the trained classifier and feature scaling statistics.

    Parameters
    ----------
    backend : str
        ``"numpy"`` (default) runs the forward pass in NumPy, which is ~1000x
        faster per frame than ``keras.Model.predict``. ``"keras"`` returns the
        original TensorFlow model. Both expose ``predict(x, verbose=0)``.

    Returns
    -------
    model, mean, std
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")

    import tensorflow as tf  # deferred: importing TF takes several seconds

    model = tf.keras.models.load_model(MODEL_PATH)
    if backend == "numpy":
        try:
            model = NumpyMLP.from_keras(model)
        except UnsupportedModelError as exc:
            logger.warning("NumPy backend unavailable (%s); falling back to Keras.", exc)

    scaler = np.load(SCALER_PATH)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    std[std == 0] = 1.0
    return model, mean, std


def backend_name(model: "Classifier") -> str:
    """Human-readable name of the inference backend in use."""
    return getattr(model, "backend_name", "keras")


def predict_probabilities(
    features: np.ndarray,
    model: "Classifier",
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Return the per-class probability vector for one landmark feature vector."""
    scaled = (features - mean) / std
    return np.asarray(model.predict(scaled.reshape(1, -1), verbose=0)[0])


def predict_finger_count(
    features: np.ndarray,
    model: "Classifier",
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[int, float]:
    """Predict finger count and confidence from landmark features."""
    probabilities = predict_probabilities(features, model, mean, std)
    predicted_index = int(np.argmax(probabilities))
    return LABELS[predicted_index], float(probabilities[predicted_index])


def resolve_stable_prediction(
    prediction_window: Iterable[tuple[int, float]],
    *,
    min_votes: int = MIN_STABLE_VOTES,
    min_confidence: float = MIN_CONFIDENCE,
) -> StablePrediction:
    """Return a stable label only if enough recent votes agree."""
    window = list(prediction_window)
    if not window:
        return StablePrediction(label=None, confidence=0.0, votes=0)

    vote_counts = Counter(label for label, _ in window)
    label, votes = vote_counts.most_common(1)[0]
    confidences = [conf for pred_label, conf in window if pred_label == label]
    average_confidence = float(sum(confidences) / len(confidences))

    if votes < min_votes or average_confidence < min_confidence:
        return StablePrediction(label=None, confidence=average_confidence, votes=votes)

    return StablePrediction(label=label, confidence=average_confidence, votes=votes)


__all__ = [
    "BACKENDS",
    "backend_name",
    "load_model_and_scaler",
    "predict_finger_count",
    "predict_probabilities",
    "resolve_stable_prediction",
]
