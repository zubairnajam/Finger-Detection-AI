"""
Standalone model evaluation and latency benchmark.

Loads the trained model, runs it against the test set, reports accuracy,
per-class precision/recall/F1, accuracy per data source (public archive vs
this project's own webcam captures), confusion matrix, and inference latency
for both the production NumPy engine and keras.predict().

Usage:
    python tools/evaluate_model.py

Outputs are saved to outputs/evaluation/.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CLASS_NAMES, EVALUATION_DIR, LABELS, TEST_DIR
from src.features import build_features, image_aspects
from src.prediction import load_model_and_scaler


def ensure_dirs() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def load_test_set():
    """Return model-ready test features (see src/features.py) and labels."""
    raw_test = np.load(TEST_DIR / "X_test.npy")
    y_test = np.load(TEST_DIR / "y_test.npy")
    aspects = np.ones(len(raw_test), dtype=np.float32)
    csv_path = TEST_DIR / "test_landmarks.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as handle:
            from_csv = image_aspects(row["image_path"] for row in csv.DictReader(handle))
        if len(from_csv) == len(raw_test):
            aspects = from_csv
    return build_features(raw_test, aspects), y_test


def load_test_sources(n_rows: int) -> list[str] | None:
    """Return 'webcam' or 'archive' for each test row, or None if unavailable."""
    csv_path = TEST_DIR / "test_landmarks.csv"
    if not csv_path.exists():
        return None
    with csv_path.open(encoding="utf-8") as handle:
        sources = [
            "webcam" if row["image_path"].replace("\\", "/").startswith("data/raw/") else "archive"
            for row in csv.DictReader(handle)
        ]
    return sources if len(sources) == n_rows else None


def benchmark_latency(
    model, mean, std, sample_features: np.ndarray, n_runs: int = 1000
) -> dict:
    """Time single-sample predictions to measure inference latency."""
    sample = ((sample_features[0] - mean) / std).reshape(1, -1)

    # Warm-up
    for _ in range(10):
        model.predict(sample, verbose=0)

    times_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(sample, verbose=0)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return {
        "n_runs": n_runs,
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
    }


def plot_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
    )
    plt.title("Confusion Matrix — Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(report_dict: dict, output_path: Path) -> None:
    """Bar chart of per-class F1 scores."""
    labels_str = [str(label) for label in LABELS]
    f1_scores = [report_dict[label_str]["f1-score"] for label_str in labels_str]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels_str, f1_scores, color="#2a78d6", edgecolor="white")
    for bar, score in zip(bars, f1_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{score:.3f}", ha="center", va="bottom", fontsize=10,
        )
    plt.ylim(0.9, 1.005)
    plt.xlabel("Finger Count Label")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Score — Test Set")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_dirs()
    model, mean, std = load_model_and_scaler(backend="numpy")
    keras_model, _, _ = load_model_and_scaler(backend="keras")
    x_test, y_test = load_test_set()

    x_test_scaled = (x_test - mean) / std

    # Predict
    probabilities = model.predict(x_test_scaled, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    # Classification report (text + dict)
    report_text = classification_report(y_test, y_pred, labels=LABELS, digits=4, zero_division=0)
    report_dict = classification_report(y_test, y_pred, labels=LABELS, digits=4, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    # Accuracy per data source (public archive vs this project's own webcam captures)
    per_source: dict[str, dict] = {}
    sources = load_test_sources(len(y_test))
    if sources is not None:
        src = np.array(sources)
        for name in ("archive", "webcam"):
            mask = src == name
            if mask.any():
                per_source[name] = {
                    "samples": int(mask.sum()),
                    "accuracy": float(accuracy_score(y_test[mask], y_pred[mask])),
                }

    # Latency: production NumPy engine vs the original keras.predict() path
    print("Running latency benchmark (NumPy engine x1000, keras.predict x200)...")
    latency = benchmark_latency(model, mean, std, x_test)
    latency_keras = benchmark_latency(keras_model, mean, std, x_test, n_runs=200)

    # Per-class summary table
    print("\n" + "=" * 60)
    print("  FINGER COUNT DETECTION — EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Test samples:     {len(y_test)}")
    print(f"  Overall accuracy: {accuracy:.4f}  ({accuracy:.1%})")
    print(f"  Macro F1:         {report_dict['macro avg']['f1-score']:.4f}")
    for name, stats in per_source.items():
        print(f"    {name:<8} accuracy {stats['accuracy']:.4f}  ({stats['samples']} samples)")
    print("\n  Inference latency (NumPy engine, used by the app):")
    print(f"    Mean:    {latency['mean_ms']:.4f} ms")
    print(f"    Median:  {latency['median_ms']:.4f} ms")
    print(f"    P95:     {latency['p95_ms']:.4f} ms")
    print(f"    P99:     {latency['p99_ms']:.4f} ms")
    print(f"  keras.predict() for comparison: {latency_keras['mean_ms']:.2f} ms mean "
          f"({latency_keras['mean_ms'] / latency['mean_ms']:,.0f}x slower)")
    print("\n  Per-class report:\n")
    print(report_text)

    # Plots
    plot_confusion_matrix(y_test, y_pred, EVALUATION_DIR / "confusion_matrix.png")
    plot_per_class_f1(report_dict, EVALUATION_DIR / "per_class_f1.png")

    # Save JSON report
    evaluation_summary = {
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy),
        "macro_avg": {
            "precision": report_dict["macro avg"]["precision"],
            "recall": report_dict["macro avg"]["recall"],
            "f1_score": report_dict["macro avg"]["f1-score"],
        },
        "per_source": per_source,
        "confusion_matrix": cm.tolist(),
        "model_parameters": int(model.count_params()),
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "per_class": {
            str(label): {
                "precision": report_dict[str(label)]["precision"],
                "recall": report_dict[str(label)]["recall"],
                "f1_score": report_dict[str(label)]["f1-score"],
                "support": report_dict[str(label)]["support"],
            }
            for label in LABELS
        },
        "latency": latency,
        "latency_backend": "numpy",
        "latency_keras_predict": latency_keras,
        "approach": {
            "hand_detection": "MediaPipe Hand Landmarker (float16)",
            "feature_extraction": "21 3D landmarks → 63 features (x, y, z)",
            "classifier": "TensorFlow/Keras MLP (128 → 64 → 32 → 6, softmax), served by a NumPy engine",
            "stabilisation": "Rolling-window majority vote (8-frame window, min 4 votes, min 55% confidence)",
        },
    }
    with (EVALUATION_DIR / "evaluation_report.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    (EVALUATION_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    print(f"\nArtifacts saved to: {EVALUATION_DIR.relative_to(_PROJECT_ROOT)}/")
    print("  - evaluation_report.json")
    print("  - classification_report.txt")
    print("  - confusion_matrix.png")
    print("  - per_class_f1.png")


if __name__ == "__main__":
    main()
