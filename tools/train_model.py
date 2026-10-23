"""
Train a finger-count classifier on MediaPipe landmark features.
Uses TensorFlow/Keras to build and train a neural network (MLP) on the landmarks extracted by the preprocessor. It outputs a trained model (finger_count_model.keras), along with a scaler and evaluation metrics like a confusion matrix to the outputs directory.

Usage:
    python tools/train_model.py

This script expects the dataset created by tools/preprocess_landmarks.py.
It is safe to rerun after collecting more data and rebuilding the dataset.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features, image_aspects, mirror_landmarks  # noqa: E402

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "training"

RANDOM_STATE = 42
BATCH_SIZE = 128
EPOCHS = 200
JITTER_COPIES = 2
USE_CLASS_WEIGHTS = False  # mirrored + jittered data is already near-balanced
JITTER_MAX_PIXELS = 4.0  # landmark noise, in pixels at 640 px frame width
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-3
LABELS = [0, 1, 2, 3, 4, 5]


def set_seed() -> None:
    """Keep runs reasonably reproducible."""
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


def ensure_output_dirs() -> None:
    """Create artifact directories."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def jitter_landmarks(raw: np.ndarray, max_pixels: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to landmark x/y, with a random strength per sample up to ``max_pixels``."""
    sigma = rng.uniform(0, max_pixels / 640, size=(len(raw), 1))
    noise = rng.normal(size=raw.shape).astype(np.float32) * sigma
    noise[:, 2::3] = 0.0  # leave depth alone
    return (raw + noise).astype(np.float32)


def load_aspects(csv_path: Path, n_rows: int) -> np.ndarray:
    """Per-row image aspect ratio, from the image paths in the preprocessing CSV."""
    if not csv_path.exists():
        return np.ones(n_rows, dtype=np.float32)
    with csv_path.open(encoding="utf-8") as handle:
        return image_aspects(row["image_path"] for row in csv.DictReader(handle))


def load_dataset():
    """Load the raw landmark arrays created during preprocessing, plus image aspects."""
    x_train = np.load(TRAIN_DIR / "X_train.npy")
    y_train = np.load(TRAIN_DIR / "y_train.npy")
    x_test = np.load(TEST_DIR / "X_test.npy")
    y_test = np.load(TEST_DIR / "y_test.npy")
    a_train = load_aspects(TRAIN_DIR / "train_landmarks.csv", len(x_train))
    a_test = load_aspects(TEST_DIR / "test_landmarks.csv", len(x_test))
    return x_train, y_train, a_train, x_test, y_test, a_test


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    """Create a small MLP suited to landmark vectors."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Balance training contribution when some labels are much rarer than others."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = int(counts.sum())
    return {
        int(label): float(total_samples / (len(unique_labels) * count))
        for label, count in zip(unique_labels, counts)
    }


def plot_training_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save loss and accuracy curves from training."""
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history_dict["loss"], label="Train loss")
    axes[0].plot(epochs, history_dict["val_loss"], label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history_dict["accuracy"], label="Train accuracy")
    axes[1].plot(epochs, history_dict["val_accuracy"], label="Val accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Train the model and save artifacts."""
    set_seed()
    ensure_output_dirs()

    raw_train, y_train, a_train, raw_test, y_test, a_test = load_dataset()
    if len(raw_train) < 10 or len(raw_test) < 5:
        raise RuntimeError("Dataset is too small to train reliably. Collect more samples first.")

    raw_fit, raw_val, y_fit, y_val, a_fit, a_val = train_test_split(
        raw_train,
        y_train,
        a_train,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    # Mirrored copies teach the model left and right hands equally; jittered
    # copies make it tolerant of the frame-to-frame wobble in live tracking.
    raw_fit = np.vstack([raw_fit, mirror_landmarks(raw_fit)])
    y_fit = np.concatenate([y_fit, y_fit])
    a_fit = np.concatenate([a_fit, a_fit])
    rng = np.random.default_rng(RANDOM_STATE)
    copies = [raw_fit] + [jitter_landmarks(raw_fit, JITTER_MAX_PIXELS, rng) for _ in range(JITTER_COPIES)]
    x_fit = build_features(np.vstack(copies), np.tile(a_fit, len(copies)))
    y_fit = np.tile(y_fit, len(copies))
    x_val = build_features(raw_val, a_val)
    x_test = build_features(raw_test, a_test)

    mean = x_fit.mean(axis=0)
    std = x_fit.std(axis=0)
    std[std == 0] = 1.0
    x_fit = (x_fit - mean) / std
    x_val = (x_val - mean) / std
    x_test_scaled = (x_test - mean) / std

    model = build_model(input_dim=x_fit.shape[1], num_classes=len(LABELS))
    class_weights = build_class_weights(y_fit)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
        class_weight=class_weights if USE_CLASS_WEIGHTS else None,
    )

    probabilities = model.predict(x_test_scaled, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    test_accuracy = accuracy_score(y_test, y_pred)

    model_path = MODELS_DIR / "finger_count_model.keras"
    model.save(model_path)
    np.savez(MODELS_DIR / "feature_scaler.npz", mean=mean, std=std)

    plot_training_curves(history, OUTPUTS_DIR / "training_curves.png")
    plot_confusion_matrix(y_test, y_pred, OUTPUTS_DIR / "confusion_matrix.png")

    report_text = classification_report(
        y_test,
        y_pred,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )
    with (OUTPUTS_DIR / "classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(report_text)

    summary = {
        "train_samples": int(len(raw_train)),
        "augmented_fit_samples": int(len(x_fit)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "feature_size": int(x_fit.shape[1]),
        "epochs_ran": int(len(history.history["loss"])),
        "test_accuracy": float(test_accuracy),
        "class_weights": {str(label): weight for label, weight in class_weights.items()},
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "scaler_path": str((MODELS_DIR / "feature_scaler.npz").relative_to(PROJECT_ROOT)),
        "rerun_note": "After collecting more samples, rerun tools/preprocess_landmarks.py and then this script.",
    }
    with (OUTPUTS_DIR / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Training complete.")
    print(f"Train samples: {len(raw_train)} (+ mirrored copies)")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Epochs run: {len(history.history['loss'])}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Artifacts saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
