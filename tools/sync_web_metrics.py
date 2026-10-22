"""
sync_web_metrics.py
───────────────────
Reads the latest evaluation_report.json, training_summary.json and (if present)
benchmark_report.json and writes a single web/metrics.json that the website
loads at runtime.

Run this automatically at the end of every training + evaluation cycle:
    python tools/sync_web_metrics.py

The run_pipeline.py script calls this automatically.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
EVAL_REPORT     = PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_report.json"
TRAIN_SUMMARY   = PROJECT_ROOT / "outputs" / "training"   / "training_summary.json"
BENCH_REPORT    = PROJECT_ROOT / "outputs" / "benchmarks" / "benchmark_report.json"
WEB_METRICS_OUT = PROJECT_ROOT / "web" / "metrics.json"

LABEL_NAMES = {
    "0": "Fist", "1": "Index", "2": "Peace",
    "3": "Three", "4": "Four",  "5": "Open Hand",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[WARNING] Not found, skipping: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_benchmarks(bench: dict) -> dict | None:
    """Condense benchmark_report.json into what the website charts need."""
    if not bench:
        return None

    single = bench["classifier"]["single_sample"]
    out = {
        "generated_at": bench.get("generated_at"),
        "processor": bench.get("system", {}).get("processor"),
        "classifier_latency_ms": {
            "keras_predict": round(single["keras_predict"]["mean_ms"], 3),
            "keras_call":    round(single["keras_call"]["mean_ms"], 3),
            "numpy":         round(single["numpy"]["mean_ms"], 4),
        },
        "speedup": round(bench["classifier"]["speedup_vs_keras_predict"]),
    }
    if "pipeline" in bench:
        p = bench["pipeline"]
        out["pipeline"] = {
            "resolution":     p["frame_resolution"],
            "detect_ms":      round(p["hand_detection"]["mean_ms"], 2),
            "classify_ms":    round(p["classification"]["mean_ms"], 4),
            "total_ms":       round(p["total_mean_ms"], 2),
            "max_fps":        round(p["max_fps"]),
        }
        out["lighting"] = [
            {
                "condition":      name,
                "detection_rate": round(r["detection_rate"] * 100, 1),
                "accuracy":       round(r["end_to_end_accuracy"] * 100, 1),
            }
            for name, r in bench["lighting"].items()
        ]
        out["lighting_note"] = bench.get("lighting_note")
    if "temporal" in bench:
        out["temporal"] = [
            {
                "jitter_px":       r["sigma_pixels_at_640"],
                "single_frame_err": round(r["per_frame_error_rate"] * 100, 2),
                "voted_err":        round(r["stabilised_false_trigger_rate"] * 100, 2),
            }
            for r in bench["temporal"]["results"]
        ]
    return out


def build_metrics(eval_data: dict, train_data: dict, bench_data: dict | None = None) -> dict:
    per_class = eval_data.get("per_class", {})

    # Build per-class list sorted by label
    classes = []
    for label_str in sorted(per_class.keys(), key=int):
        d = per_class[label_str]
        classes.append({
            "label":     int(label_str),
            "name":      LABEL_NAMES.get(label_str, label_str),
            "precision": round(d.get("precision", 0), 6),
            "recall":    round(d.get("recall",    0), 6),
            "f1":        round(d.get("f1_score",  0), 6),
            "support":   int(d.get("support", 0)),
        })

    latency = eval_data.get("latency", {})
    latency_keras = eval_data.get("latency_keras_predict", {})
    macro = eval_data.get("macro_avg", {})

    return {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "accuracy":      round(eval_data.get("accuracy", 0) * 100, 2),   # percent
        "test_samples":  eval_data.get("test_samples", 0),
        "train_samples": train_data.get("train_samples"),
        "macro_precision": round(macro.get("precision", 0) * 100, 2),
        "macro_recall":    round(macro.get("recall", 0) * 100, 2),
        "macro_f1":        round(macro.get("f1_score", 0) * 100, 2),
        "mean_latency_ms": round(latency.get("mean_ms", 0), 4),
        "p95_latency_ms":  round(latency.get("p95_ms", 0), 4),
        "latency_backend": eval_data.get("latency_backend", "keras"),
        "keras_predict_latency_ms": round(latency_keras.get("mean_ms", 0), 2) or None,
        "per_source": {
            name: {"samples": s["samples"], "accuracy": round(s["accuracy"] * 100, 2)}
            for name, s in eval_data.get("per_source", {}).items()
        },
        "confusion_matrix": eval_data.get("confusion_matrix"),
        "classes":       classes,
        # Training info (from training_summary.json if present)
        "train_accuracy":   round(train_data.get("test_accuracy",  0) * 100, 2),
        "train_epochs":     train_data.get("epochs_ran", train_data.get("epochs_trained")),
        "model_parameters": eval_data.get("model_parameters", train_data.get("model_parameters")),
        # Raw F1 list (convenient for Chart.js)
        "f1_scores": [c["f1"] for c in classes],
        "f1_labels": [f"{c['label']} ({c['name']})" for c in classes],
        "benchmarks": build_benchmarks(bench_data or {}),
    }


def main():
    print("[sync_web_metrics] Reading evaluation, training and benchmark reports...")

    eval_data  = load_json(EVAL_REPORT)
    train_data = load_json(TRAIN_SUMMARY)
    bench_data = load_json(BENCH_REPORT)

    if not eval_data:
        print("[ERROR] evaluation_report.json is missing.")
        print("  Run:  python tools/evaluate_model.py")
        sys.exit(1)

    metrics = build_metrics(eval_data, train_data, bench_data)

    WEB_METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(WEB_METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[sync_web_metrics] OK Wrote web/metrics.json")
    print(f"  Accuracy  : {metrics['accuracy']}%")
    print(f"  Samples   : {metrics['test_samples']}")
    print(f"  Latency   : {metrics['mean_latency_ms']} ms ({metrics['latency_backend']})")
    print(f"  Benchmarks: {'included' if metrics['benchmarks'] else 'not found (run tools/benchmark.py)'}")
    for c in metrics["classes"]:
        print(f"  Class {c['label']} ({c['name']:<10}): F1 = {c['f1']:.4f}")


if __name__ == "__main__":
    main()
