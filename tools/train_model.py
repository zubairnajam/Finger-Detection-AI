"""
Train a finger-count classifier on MediaPipe landmark features.

Usage:
    python tools/train_model.py

This script expects the dataset created by tools/preprocess_landmarks.py.
It is safe to rerun after collecting more data and rebuilding the dataset.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "training"

RANDOM_STATE = 42
BATCH_SIZE = 8
EPOCHS = 100
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


def load_dataset():
    """Load the train/test arrays created during preprocessing."""
    x_train = np.load(TRAIN_DIR / "X_train.npy")
    y_train = np.load(TRAIN_DIR / "y_train.npy")
    x_test = np.load(TEST_DIR / "X_test.npy")
    y_test = np.load(TEST_DIR / "y_test.npy")
    return x_train, y_train, x_test, y_test


def standardize(train_features: np.ndarray, test_features: np.ndarray):
    """Standardize features using train-set statistics only."""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0

    x_train_scaled = (train_features - mean) / std
    x_test_scaled = (test_features - mean) / std
    return x_train_scaled, x_test_scaled, mean, std


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

    x_train, y_train, x_test, y_test = load_dataset()
    if len(x_train) < 10 or len(x_test) < 5:
        raise RuntimeError("Dataset is too small to train reliably. Collect more samples first.")

    x_train_scaled, x_test_scaled, mean, std = standardize(x_train, x_test)

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train_scaled,
        y_train,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    model = build_model(input_dim=x_train_scaled.shape[1], num_classes=len(LABELS))
    class_weights = build_class_weights(y_fit)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
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
        class_weight=class_weights,
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
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "feature_size": int(x_train.shape[1]),
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
    print(f"Train samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Epochs run: {len(history.history['loss'])}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Artifacts saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
