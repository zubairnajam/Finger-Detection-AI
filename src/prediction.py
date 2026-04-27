"""
Model loading, inference, and prediction stabilisation logic.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
import tensorflow as tf

from src.config import (
    LABELS,
    MIN_CONFIDENCE,
    MIN_STABLE_VOTES,
    MODEL_PATH,
    SCALER_PATH,
    StablePrediction,
)


def load_model_and_scaler() -> tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """Load the trained classifier and feature scaling statistics.

    Returns
    -------
    model, mean, std
    """
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
