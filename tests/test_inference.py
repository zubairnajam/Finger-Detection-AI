"""NumPy inference engine: correctness against Keras and serialisation."""

import numpy as np
import pytest

from src.config import LABELS, TEST_DIR
from src.inference import DenseLayer, NumpyMLP, UnsupportedModelError
from src.prediction import (
    load_model_and_scaler,
    predict_finger_count,
    predict_probabilities,
)


def _tiny_mlp(seed=0) -> NumpyMLP:
    rng = np.random.default_rng(seed)
    return NumpyMLP([
        DenseLayer(rng.normal(size=(4, 8)).astype(np.float32), np.zeros(8, np.float32), "relu"),
        DenseLayer(rng.normal(size=(8, 3)).astype(np.float32), np.zeros(3, np.float32), "softmax"),
    ])


def test_probabilities_sum_to_one():
    mlp = _tiny_mlp()
    probs = mlp.predict_proba(np.random.default_rng(1).normal(size=(10, 4)))
    assert probs.shape == (10, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    assert (probs >= 0).all()


def test_single_vector_is_promoted_to_batch():
    mlp = _tiny_mlp()
    assert mlp.predict_proba(np.ones(4)).shape == (1, 3)


def test_softmax_is_numerically_stable():
    mlp = NumpyMLP([DenseLayer(np.eye(2, dtype=np.float32) * 1e4,
                               np.zeros(2, np.float32), "softmax")])
    probs = mlp.predict_proba(np.array([[1.0, -1.0]]))
    assert np.isfinite(probs).all()


def test_save_and_load_roundtrip(tmp_path):
    mlp = _tiny_mlp()
    path = tmp_path / "weights.npz"
    mlp.save(path)
    restored = NumpyMLP.load(path)
    x = np.random.default_rng(2).normal(size=(5, 4))
    np.testing.assert_array_equal(mlp.predict_proba(x), restored.predict_proba(x))
    assert restored.count_params() == mlp.count_params()


def test_empty_network_rejected():
    with pytest.raises(ValueError):
        NumpyMLP([])


def test_matches_keras_on_trained_model(keras_model):
    mlp = NumpyMLP.from_keras(keras_model)
    assert mlp.input_dim == 63
    assert mlp.num_classes == len(LABELS)
    assert mlp.count_params() == keras_model.count_params()

    x = np.random.default_rng(0).normal(size=(256, 63)).astype(np.float32)
    expected = keras_model.predict(x, verbose=0)
    actual = mlp.predict(x, verbose=0)
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert (actual.argmax(1) == expected.argmax(1)).all()


def test_matches_keras_on_real_test_set(keras_model):
    x_path, y_path = TEST_DIR / "X_test.npy", TEST_DIR / "y_test.npy"
    if not x_path.exists():
        pytest.skip("Test split not present (data/ is git-ignored)")

    model, mean, std = load_model_and_scaler(backend="numpy")
    x = (np.load(x_path) - mean) / std
    y = np.load(y_path)

    keras_pred = keras_model.predict(x, verbose=0).argmax(1)
    numpy_pred = model.predict(x).argmax(1)
    assert (keras_pred == numpy_pred).all()
    assert (numpy_pred == y).mean() > 0.95


def test_unsupported_layer_raises():
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    with pytest.raises(UnsupportedModelError):
        NumpyMLP.from_keras(model)


@pytest.mark.parametrize("backend", ["numpy", "keras"])
def test_prediction_helpers_agree_across_backends(keras_model, backend):
    model, mean, std = load_model_and_scaler(backend=backend)
    features = np.random.default_rng(3).uniform(0, 1, size=63).astype(np.float32)
    label, confidence = predict_finger_count(features, model, mean, std)
    probs = predict_probabilities(features, model, mean, std)
    assert label in LABELS
    assert probs.shape == (len(LABELS),)
    assert abs(confidence - probs.max()) < 1e-6


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        load_model_and_scaler(backend="onnx")
