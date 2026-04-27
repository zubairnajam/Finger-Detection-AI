"""
Standalone model evaluation and latency benchmark.

Loads the trained model, runs it against the test set, reports accuracy,
per-class precision/recall/F1, confusion matrix, and inference latency.

Usage:
    python tools/evaluate_model.py

Outputs are saved to outputs/evaluation/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import EVALUATION_DIR, LABELS, TEST_DIR, TRAIN_DIR
from src.prediction import load_model_and_scaler


def ensure_dirs() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def load_test_set():
    x_test = np.load(TEST_DIR / "X_test.npy")
    y_test = np.load(TEST_DIR / "y_test.npy")
    return x_test, y_test


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
    bars = plt.bar(labels_str, f1_scores, color="#4A90D9", edgecolor="white")
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
    model, mean, std = load_model_and_scaler()
    x_test, y_test = load_test_set()

    # Scale test features
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    x_test_scaled = (x_test - mean) / std_safe

    # Predict
    probabilities = model.predict(x_test_scaled, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    # Classification report (text + dict)
    report_text = classification_report(y_test, y_pred, labels=LABELS, digits=4, zero_division=0)
    report_dict = classification_report(y_test, y_pred, labels=LABELS, digits=4, zero_division=0, output_dict=True)

    # Latency benchmark
    print("Running latency benchmark (1000 predictions)...")
    latency = benchmark_latency(model, mean, std, x_test)

    # Per-class summary table
    print("\n" + "=" * 60)
    print("  FINGER COUNT DETECTION — EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Test samples:    {len(y_test)}")
    print(f"  Overall accuracy: {accuracy:.4f}  ({accuracy:.1%})")
    print(f"\n  Inference latency:")
    print(f"    Mean:    {latency['mean_ms']:.2f} ms")
    print(f"    Median:  {latency['median_ms']:.2f} ms")
    print(f"    P95:     {latency['p95_ms']:.2f} ms")
    print(f"    P99:     {latency['p99_ms']:.2f} ms")
    print(f"\n  Per-class report:\n")
    print(report_text)

    # Plots
    plot_confusion_matrix(y_test, y_pred, EVALUATION_DIR / "confusion_matrix.png")
    plot_per_class_f1(report_dict, EVALUATION_DIR / "per_class_f1.png")

    # Save JSON report
    evaluation_summary = {
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy),
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
        "approach": {
            "hand_detection": "MediaPipe Hand Landmarker (float16)",
            "feature_extraction": "21 3D landmarks → 63 features (x, y, z)",
            "classifier": "TensorFlow/Keras MLP (128 → 64 → 32 → 6, softmax)",
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
