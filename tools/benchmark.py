"""
Reproducible performance & robustness benchmark suite.

Measures, on the current machine:
  1. Classifier latency   - keras.predict vs keras.__call__ vs the NumPy engine
  2. Pipeline latency     - MediaPipe hand detection + features + classifier on
                            real held-out webcam frames -> achievable FPS
  3. Lighting robustness  - detection rate & accuracy on held-out webcam frames
                            under simulated dim / overexposed / noisy / blurred
                            conditions
  4. Temporal stability   - per-frame vs rolling-vote accuracy and false-trigger
                            rate under landmark jitter (full 3,550-sample test set)

Usage:
    python tools/benchmark.py              # full run (~2-3 min)
    python tools/benchmark.py --quick      # fewer repetitions (~40 s)
    python tools/benchmark.py --skip-images

Outputs (outputs/benchmarks/):
    benchmark_report.json   machine-readable results (consumed by sync_web_metrics.py)
    benchmark_report.md     human-readable tables
    latency.png, lighting_robustness.png, temporal_stability.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    BENCHMARK_DIR,
    LABELS,
    MIN_CONFIDENCE,
    MIN_STABLE_VOTES,
    PROJECT_ROOT,
    ROLLING_WINDOW_SIZE,
    TEST_DIR,
)
from src.features import build_features, image_aspects
from src.hand_detection import build_detector, flatten_landmarks
from src.inference import NumpyMLP

# Reference categorical palette (validated for CVD separation) + text inks
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

LIGHTING_CONDITIONS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "Original": lambda img, rng: img,
    "Dim (40%)": lambda img, rng: cv2.convertScaleAbs(img, alpha=0.40),
    "Very dim (15%)": lambda img, rng: cv2.convertScaleAbs(img, alpha=0.15),
    "Overexposed": lambda img, rng: cv2.convertScaleAbs(img, alpha=1.3, beta=50),
    "Low contrast": lambda img, rng: cv2.convertScaleAbs(img, alpha=0.35, beta=110),
    "Sensor noise": lambda img, rng: np.clip(
        img.astype(np.float32) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8),
    "Motion blur": lambda img, rng: cv2.filter2D(img, -1, np.full((1, 21), 1 / 21)),
}
# Image-plane jitter in pixels at 640 px width. Applied to x/y only: MediaPipe z is
# a relative depth estimate on a different scale (and wrist z is always ~0).
JITTER_PIXELS = (0, 2, 4, 8, 12, 16)
REFERENCE_WIDTH_PX = 640


# ── Helpers ──────────────────────────────────────────────────────────────────

def timing_stats(samples_ms: list[float]) -> dict:
    arr = np.asarray(samples_ms)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def time_fn(fn: Callable[[], object], runs: int, warmup: int = 10) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return timing_stats(samples)


def system_info() -> dict:
    import tensorflow as tf

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "mediapipe": mp.__version__,
        "opencv": cv2.__version__,
    }


def load_models():
    import tensorflow as tf

    from src.config import MODEL_PATH, SCALER_PATH

    keras_model = tf.keras.models.load_model(MODEL_PATH)
    numpy_model = NumpyMLP.from_keras(keras_model)
    scaler = np.load(SCALER_PATH)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    std[std == 0] = 1.0
    return keras_model, numpy_model, mean, std


def held_out_webcam_images() -> list[tuple[Path, int]]:
    """Webcam frames that are in the *test* split (never seen in training)."""
    csv_path = TEST_DIR / "test_landmarks.csv"
    if not csv_path.exists():
        return []
    items = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rel = row["image_path"].replace("\\", "/")
            path = PROJECT_ROOT / rel
            if rel.startswith("data/raw/") and path.exists():
                items.append((path, int(row["label"])))
    return items


# ── 1. Classifier latency ────────────────────────────────────────────────────

def bench_classifier(keras_model, numpy_model, x_test: np.ndarray, runs: int) -> dict:
    import tensorflow as tf

    sample = x_test[:1]
    keras_runs = max(50, runs // 4)  # predict() is ~70 ms/call; keep wall time sane
    print(f"  keras.predict()      x{keras_runs}")
    k_predict = time_fn(lambda: keras_model.predict(sample, verbose=0), keras_runs)
    print(f"  keras.__call__()     x{runs}")
    k_call = time_fn(lambda: keras_model(sample, training=False), runs)
    print(f"  NumPy engine         x{runs * 10}")
    np_single = time_fn(lambda: numpy_model.predict_proba(sample), runs * 10)

    t0 = time.perf_counter()
    keras_model.predict(x_test, verbose=0, batch_size=len(x_test))
    k_batch_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(10):
        numpy_model.predict_proba(x_test)
    np_batch_s = (time.perf_counter() - t0) / 10

    del tf
    return {
        "single_sample": {
            "keras_predict": k_predict,
            "keras_call": k_call,
            "numpy": np_single,
        },
        "speedup_vs_keras_predict": k_predict["mean_ms"] / np_single["mean_ms"],
        "batch_throughput_samples_per_s": {
            "keras_predict": len(x_test) / k_batch_s,
            "numpy": len(x_test) / np_batch_s,
        },
        "batch_size": int(len(x_test)),
    }


# ── 2 & 3. Pipeline latency and lighting robustness ──────────────────────────

def bench_images(numpy_model, mean, std, images: list[tuple[Path, int]], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    detector = build_detector(running_mode="IMAGE")
    detect_ms: list[float] = []
    feature_ms: list[float] = []
    classify_ms: list[float] = []
    lighting: dict[str, dict] = {}

    try:
        frames = [(cv2.imread(str(p)), label) for p, label in images]
        for condition, transform in LIGHTING_CONDITIONS.items():
            detected = correct = 0
            for frame, label in frames:
                img = transform(frame, rng)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                t0 = time.perf_counter()
                result = detector.detect(mp_image)
                t1 = time.perf_counter()
                raw = flatten_landmarks(result)
                features = None if raw is None else build_features(raw, img.shape[1] / img.shape[0])
                t2 = time.perf_counter()
                if features is None:
                    continue
                probs = numpy_model.predict_proba((features - mean) / std)[0]
                t3 = time.perf_counter()

                if condition == "Original":
                    detect_ms.append((t1 - t0) * 1000)
                    feature_ms.append((t2 - t1) * 1000)
                    classify_ms.append((t3 - t2) * 1000)

                detected += 1
                correct += int(LABELS[int(np.argmax(probs))] == label)

            n = len(frames)
            lighting[condition] = {
                "images": n,
                "detection_rate": detected / n,
                "accuracy_when_detected": correct / detected if detected else 0.0,
                "end_to_end_accuracy": correct / n,
            }
            print(f"  {condition:<16} detect {detected / n:6.1%}   "
                  f"acc|det {lighting[condition]['accuracy_when_detected']:6.1%}   "
                  f"end-to-end {correct / n:6.1%}")
    finally:
        detector.close()

    detect = timing_stats(detect_ms)
    total_mean = detect["mean_ms"] + np.mean(feature_ms) + np.mean(classify_ms)
    return {
        "pipeline": {
            "frame_resolution": f"{frames[0][0].shape[1]}x{frames[0][0].shape[0]}",
            "hand_detection": detect,
            "feature_extraction": timing_stats(feature_ms),
            "classification": timing_stats(classify_ms),
            "total_mean_ms": float(total_mean),
            "max_fps": float(1000.0 / total_mean),
        },
        "lighting": lighting,
        "lighting_note": (
            f"{len(frames)} held-out webcam frames (labels 1-5) from the test split; "
            "conditions are simulated with photometric transforms."
        ),
    }


# ── 4. Temporal stability under landmark jitter ──────────────────────────────

def stable_vote(labels: np.ndarray, confs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised rolling-window vote over axis 1. Returns (label | -1, committed mask)."""
    n = labels.shape[0]
    counts = np.zeros((n, len(LABELS)), dtype=np.int32)
    conf_sum = np.zeros((n, len(LABELS)), dtype=np.float64)
    for k in range(labels.shape[1]):
        np.add.at(counts, (np.arange(n), labels[:, k]), 1)
        np.add.at(conf_sum, (np.arange(n), labels[:, k]), confs[:, k])
    winner = counts.argmax(1)
    votes = counts[np.arange(n), winner]
    avg_conf = conf_sum[np.arange(n), winner] / np.maximum(votes, 1)
    committed = (votes >= MIN_STABLE_VOTES) & (avg_conf >= MIN_CONFIDENCE)
    return np.where(committed, winner, -1), committed


def bench_temporal(numpy_model, mean, std, x_raw: np.ndarray, aspects: np.ndarray, y: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    results = []
    xy_mask = np.ones(x_raw.shape[1], dtype=np.float32)
    xy_mask[2::3] = 0.0
    for pixels in JITTER_PIXELS:
        sigma = pixels / REFERENCE_WIDTH_PX
        noise = rng.normal(0, sigma, (len(x_raw), ROLLING_WINDOW_SIZE, x_raw.shape[1])) * xy_mask
        frames = x_raw[:, None, :] + noise
        feats = build_features(frames.reshape(-1, x_raw.shape[1]), np.repeat(aspects, ROLLING_WINDOW_SIZE))
        probs = numpy_model.predict_proba((feats - mean) / std)
        probs = probs.reshape(len(x_raw), ROLLING_WINDOW_SIZE, -1)
        frame_labels = probs.argmax(2)
        frame_conf = probs.max(2)

        per_frame_acc = float((frame_labels == y[:, None]).mean())
        voted, committed = stable_vote(frame_labels, frame_conf)
        wrong_commit = committed & (voted != y)
        results.append({
            "sigma": sigma,
            "sigma_pixels_at_640": float(pixels),
            "per_frame_accuracy": per_frame_acc,
            "per_frame_error_rate": 1 - per_frame_acc,
            "stabilised_accuracy_when_committed": float((voted[committed] == y[committed]).mean())
            if committed.any() else 0.0,
            "stabilised_false_trigger_rate": float(wrong_commit.mean()),
            "abstain_rate": float(1 - committed.mean()),
        })
        r = results[-1]
        print(f"  jitter {pixels:>2}px  per-frame err "
              f"{r['per_frame_error_rate']:6.2%}  ->  stabilised false-trigger "
              f"{r['stabilised_false_trigger_rate']:6.2%}  (abstain {r['abstain_rate']:5.1%})")
    return {
        "samples": int(len(x_raw)),
        "window": ROLLING_WINDOW_SIZE,
        "min_votes": MIN_STABLE_VOTES,
        "min_confidence": MIN_CONFIDENCE,
        "results": results,
        "note": "Each test sample is replayed as an 8-frame window with i.i.d. Gaussian "
                "jitter added to every landmark's x/y position (std. dev. in pixels at 640 px width).",
    }


# ── 4. Framing robustness ───────────────────────────────────────────────────

# name -> (max shift as a fraction of the frame, min scale, max scale)
FRAMING_LEVELS = {
    "As captured": (0.0, 1.0, 1.0),
    "Off-centre": (0.2, 1.0, 1.0),
    "Further away": (0.0, 0.55, 0.75),
    "Closer": (0.0, 1.25, 1.4),
    "Mixed": (0.25, 0.6, 1.3),
}


def bench_framing(numpy_model, mean, std, x_raw: np.ndarray, aspects: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Accuracy when test hands are moved and resized in the frame, as happens on a real webcam."""
    rng = np.random.default_rng(seed)
    results = {}
    for name, (shift, lo, hi) in FRAMING_LEVELS.items():
        points = x_raw.reshape(-1, 21, 3).copy()
        scale = rng.uniform(lo, hi, len(points))[:, None, None]
        wrist = points[:, :1, :2].copy()
        points[:, :, :2] = (points[:, :, :2] - wrist) * scale + wrist
        points[:, :, 2:] *= scale
        points[:, :, :2] += rng.uniform(-shift, shift, (len(points), 1, 2))
        feats = build_features(points.reshape(len(points), -1), aspects)
        pred = np.array(LABELS)[numpy_model.predict_proba((feats - mean) / std).argmax(1)]
        results[name] = {
            "accuracy": float((pred == y).mean()),
            "four_read_as_five": int(((y == 4) & (pred == 5)).sum()),
        }
        print(f"  {name:<13} accuracy {results[name]['accuracy']:6.2%}")
    return {
        "samples": int(len(x_raw)),
        "results": results,
        "note": "Test-set hands are shifted (up to the given fraction of the frame) and re-scaled "
                "about the wrist before classification.",
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_latency(classifier: dict, out: Path) -> None:
    single = classifier["single_sample"]
    names = ["Keras model.predict()", "Keras model(x)", "NumPy engine (ours)"]
    values = [single["keras_predict"]["mean_ms"], single["keras_call"]["mean_ms"],
              single["numpy"]["mean_ms"]]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    bars = ax.barh(names, values, color=BLUE, height=0.55)
    ax.set_xscale("log")
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        ax.text(v * 1.15, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f} ms" if v < 1 else f"{v:.1f} ms", va="center", color=INK, fontsize=9)
    ax.set_xlim(min(values) / 2, max(values) * 6)
    ax.set_xlabel("Mean single-sample latency (ms, log scale)", color=INK_MUTED, fontsize=9)
    _style(ax, f"Classifier latency  -  {classifier['speedup_vs_keras_predict']:,.0f}x faster")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_lighting(lighting: dict, out: Path) -> None:
    conditions = list(lighting)
    det = [lighting[c]["detection_rate"] * 100 for c in conditions]
    acc = [lighting[c]["end_to_end_accuracy"] * 100 for c in conditions]
    x = np.arange(len(conditions))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(x - w / 2 - 0.01, det, w, color=BLUE, label="Hand detected")
    ax.bar(x + w / 2 + 0.01, acc, w, color=ORANGE, label="Correct finger count (end-to-end)")
    for xi, v in zip(x + w / 2 + 0.01, acc):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, conditions, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of held-out frames", color=INK_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncols=2, bbox_to_anchor=(1, 1.13))
    _style(ax, "Lighting robustness")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_temporal(temporal: dict, out: Path) -> None:
    rows = temporal["results"]
    px = [r["sigma_pixels_at_640"] for r in rows]
    frame_err = [r["per_frame_error_rate"] * 100 for r in rows]
    stable_err = [r["stabilised_false_trigger_rate"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(px, frame_err, color=ORANGE, linewidth=2, marker="o", markersize=6,
            label="Single frame (no voting)")
    ax.plot(px, stable_err, color=BLUE, linewidth=2, marker="o", markersize=6,
            label="8-frame rolling vote")
    ax.annotate("Single frame", (px[-1], frame_err[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.annotate("Rolling vote", (px[-1], stable_err[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Landmark jitter (std. dev., pixels at 640 px width)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Wrong-gesture rate (%)", color=INK_MUTED, fontsize=9)
    ax.set_xlim(-0.5, px[-1] * 1.22)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, "Temporal voting suppresses false triggers")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ── Report ───────────────────────────────────────────────────────────────────

def write_markdown(report: dict, out: Path) -> None:
    c = report["classifier"]
    s = c["single_sample"]
    lines = [
        "# Benchmark Report",
        "",
        f"Generated {report['generated_at']} on `{report['system']['processor']}` "
        f"({report['system']['cpu_count']} logical CPUs), Python {report['system']['python']}, "
        f"TensorFlow {report['system']['tensorflow']}, MediaPipe {report['system']['mediapipe']}.",
        "",
        "## Classifier latency (single sample, CPU)",
        "",
        "| Backend | Mean | Median | P95 | P99 |",
        "|---|---|---|---|---|",
    ]
    for name, key in [("Keras `model.predict()`", "keras_predict"), ("Keras `model(x)`", "keras_call"),
                      ("**NumPy engine**", "numpy")]:
        t = s[key]
        lines.append(f"| {name} | {t['mean_ms']:.3f} ms | {t['median_ms']:.3f} ms | "
                     f"{t['p95_ms']:.3f} ms | {t['p99_ms']:.3f} ms |")
    tp = c["batch_throughput_samples_per_s"]
    lines += [
        "",
        f"**Speed-up vs `model.predict()`: {c['speedup_vs_keras_predict']:,.0f}x.** "
        f"Batch throughput ({c['batch_size']} samples): NumPy {tp['numpy']:,.0f}/s vs "
        f"Keras {tp['keras_predict']:,.0f}/s.",
        "",
    ]

    if "pipeline" in report:
        p = report["pipeline"]
        lines += [
            f"## End-to-end pipeline ({p['frame_resolution']} webcam frames)",
            "",
            "| Stage | Mean | P95 |",
            "|---|---|---|",
            f"| MediaPipe hand detection | {p['hand_detection']['mean_ms']:.2f} ms | {p['hand_detection']['p95_ms']:.2f} ms |",
            f"| Feature extraction | {p['feature_extraction']['mean_ms']:.3f} ms | {p['feature_extraction']['p95_ms']:.3f} ms |",
            f"| MLP classification | {p['classification']['mean_ms']:.3f} ms | {p['classification']['p95_ms']:.3f} ms |",
            f"| **Total** | **{p['total_mean_ms']:.2f} ms** | |",
            "",
            f"Compute ceiling: **{p['max_fps']:.0f} FPS** (the webcam's own frame rate, typically 30 FPS, is the practical limit).",
            "",
            "## Lighting robustness",
            "",
            report["lighting_note"],
            "",
            "| Condition | Hand detected | Accuracy when detected | End-to-end accuracy |",
            "|---|---|---|---|",
        ]
        for name, r in report["lighting"].items():
            lines.append(f"| {name} | {r['detection_rate']:.1%} | {r['accuracy_when_detected']:.1%} | "
                         f"{r['end_to_end_accuracy']:.1%} |")
        lines.append("")

    t = report["temporal"]
    lines += [
        "## Temporal stability under landmark jitter",
        "",
        t["note"],
        "",
        "| Jitter (px @ 640) | Per-frame error | Rolling-vote false trigger | Abstain |",
        "|---|---|---|---|",
    ]
    for r in t["results"]:
        lines.append(f"| {r['sigma_pixels_at_640']:.1f} | {r['per_frame_error_rate']:.2%} | "
                     f"{r['stabilised_false_trigger_rate']:.2%} | {r['abstain_rate']:.1%} |")
    lines.append("")

    if "framing" in report:
        f = report["framing"]
        lines += ["## Framing robustness", "", f["note"], "", "| Framing | Accuracy | 4 read as 5 |", "|---|---|---|"]
        for name, r in f["results"].items():
            lines.append(f"| {name} | {r['accuracy']:.1%} | {r['four_read_as_five']} |")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1000, help="Timed runs for fast backends")
    parser.add_argument("--quick", action="store_true", help="Fewer repetitions")
    parser.add_argument("--skip-images", action="store_true", help="Skip MediaPipe image benchmarks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    runs = 200 if args.quick else args.runs

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    x_path = TEST_DIR / "X_test.npy"
    if not x_path.exists():
        sys.exit("Missing data/test/X_test.npy - run tools/preprocess_landmarks.py first.")

    print("Loading models...")
    keras_model, numpy_model, mean, std = load_models()
    x_raw = np.load(x_path).astype(np.float32)
    y = np.load(TEST_DIR / "y_test.npy")
    aspects = np.ones(len(x_raw), dtype=np.float32)
    csv_path = TEST_DIR / "test_landmarks.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as handle:
            aspects = image_aspects(row["image_path"] for row in csv.DictReader(handle))
    x_scaled = (build_features(x_raw, aspects) - mean) / std

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": system_info(),
        "model_parameters": numpy_model.count_params(),
    }

    print("\n[1/4] Classifier latency")
    report["classifier"] = bench_classifier(keras_model, numpy_model, x_scaled, runs)
    print(f"  -> {report['classifier']['speedup_vs_keras_predict']:,.0f}x speed-up")

    images = [] if args.skip_images else held_out_webcam_images()
    if images:
        print(f"\n[2/4] Pipeline latency + lighting robustness ({len(images)} held-out frames)")
        report.update(bench_images(numpy_model, mean, std, images, args.seed))
    else:
        print("\n[2/4] Skipped image benchmarks (no held-out webcam frames found or --skip-images)")

    print("\n[3/4] Temporal stability under landmark jitter")
    report["temporal"] = bench_temporal(numpy_model, mean, std, x_raw, aspects, y, args.seed)

    print("\n[4/4] Framing robustness (hand moved / resized in frame)")
    report["framing"] = bench_framing(numpy_model, mean, std, x_raw, aspects, y, args.seed)

    (BENCHMARK_DIR / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, BENCHMARK_DIR / "benchmark_report.md")
    plot_latency(report["classifier"], BENCHMARK_DIR / "latency.png")
    if "lighting" in report:
        plot_lighting(report["lighting"], BENCHMARK_DIR / "lighting_robustness.png")
    plot_temporal(report["temporal"], BENCHMARK_DIR / "temporal_stability.png")

    print(f"\nReports written to {BENCHMARK_DIR.relative_to(PROJECT_ROOT)}/")
    if "pipeline" in report:
        p = report["pipeline"]
        print(f"  End-to-end: {p['total_mean_ms']:.1f} ms/frame -> {p['max_fps']:.0f} FPS ceiling")


if __name__ == "__main__":
    main()
