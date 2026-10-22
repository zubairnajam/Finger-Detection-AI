"""
Lightweight NumPy inference engine for the finger-count MLP.

`tf.keras.Model.predict()` carries ~60-70 ms of fixed per-call overhead
(graph dispatch, data-adapter setup), which dominates single-sample latency in
a real-time loop. The classifier itself is only a few small dense layers, so
we extract the trained weights once and run the forward pass directly in
NumPy. Outputs match Keras to within ~1e-6 (verified in tests/).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import tensorflow as tf


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _linear(x: np.ndarray) -> np.ndarray:
    return x


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": _relu,
    "linear": _linear,
    "softmax": _softmax,
}

# Layers that are no-ops at inference time
PASSTHROUGH_LAYERS = {"Dropout", "InputLayer", "GaussianNoise", "GaussianDropout"}


class UnsupportedModelError(ValueError):
    """Raised when a Keras model contains layers the NumPy engine cannot run."""


@dataclass(frozen=True)
class DenseLayer:
    weights: np.ndarray
    bias: np.ndarray
    activation: str


class NumpyMLP:
    """Dense-only feed-forward network evaluated with NumPy.

    Exposes a Keras-compatible ``predict(x, verbose=0)`` so it can be used
    anywhere the original Keras model was used.
    """

    backend_name = "numpy"

    def __init__(self, layers: list[DenseLayer]):
        if not layers:
            raise ValueError("NumpyMLP needs at least one dense layer.")
        self.layers = layers

    # ── Construction ────────────────────────────────────────────────────
    @classmethod
    def from_keras(cls, model: "tf.keras.Model") -> "NumpyMLP":
        """Extract weights from a Sequential Dense/Dropout Keras model."""
        layers: list[DenseLayer] = []
        for layer in model.layers:
            kind = type(layer).__name__
            if kind in PASSTHROUGH_LAYERS:
                continue
            if kind != "Dense":
                raise UnsupportedModelError(f"Unsupported layer type: {kind}")

            activation = layer.activation.__name__
            if activation not in ACTIVATIONS:
                raise UnsupportedModelError(f"Unsupported activation: {activation}")

            weights, bias = layer.get_weights()
            layers.append(
                DenseLayer(
                    weights=weights.astype(np.float32),
                    bias=bias.astype(np.float32),
                    activation=activation,
                )
            )
        return cls(layers)

    @classmethod
    def load(cls, path: Path) -> "NumpyMLP":
        """Load weights previously written by :meth:`save`."""
        data = np.load(path)
        count = int(data["num_layers"])
        layers = [
            DenseLayer(
                weights=data[f"w{i}"],
                bias=data[f"b{i}"],
                activation=str(data[f"a{i}"]),
            )
            for i in range(count)
        ]
        return cls(layers)

    def save(self, path: Path) -> None:
        """Serialise weights to a single .npz file (no TensorFlow needed to load)."""
        arrays: dict[str, np.ndarray] = {"num_layers": np.array(len(self.layers))}
        for i, layer in enumerate(self.layers):
            arrays[f"w{i}"] = layer.weights
            arrays[f"b{i}"] = layer.bias
            arrays[f"a{i}"] = np.array(layer.activation)
        np.savez(path, **arrays)

    # ── Inference ───────────────────────────────────────────────────────
    @property
    def input_dim(self) -> int:
        return int(self.layers[0].weights.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.layers[-1].weights.shape[1])

    def count_params(self) -> int:
        return int(sum(layer.weights.size + layer.bias.size for layer in self.layers))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for a (n, features) or (features,) array."""
        h = np.asarray(x, dtype=np.float32)
        if h.ndim == 1:
            h = h.reshape(1, -1)
        for layer in self.layers:
            h = ACTIVATIONS[layer.activation](h @ layer.weights + layer.bias)
        return h

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:  # noqa: ARG002
        """Keras-compatible alias for :meth:`predict_proba`."""
        return self.predict_proba(x)

    __call__ = predict_proba
