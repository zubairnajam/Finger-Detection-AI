"""
sync_web_metrics.py
───────────────────
Reads the latest evaluation_report.json and training_summary.json and writes
a single web/metrics.json that the website loads at runtime.

Run this automatically at the end of every training + evaluation cycle:
    python tools/sync_web_metrics.py

The run_pipeline.py script calls this automatically as its final stage.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
EVAL_REPORT     = PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_report.json"
TRAIN_SUMMARY   = PROJECT_ROOT / "outputs" / "training"   / "training_summary.json"
WEB_METRICS_OUT = PROJECT_ROOT / "web" / "metrics.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[WARNING] Not found, skipping: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def build_metrics(eval_data: dict, train_data: dict) -> dict:
    per_class = eval_data.get("per_class", {})

    # Build per-class list sorted by label
    classes = []
    for label_str in sorted(per_class.keys(), key=int):
        d = per_class[label_str]
        label_names = {
            "0": "Fist", "1": "Index", "2": "Peace",
            "3": "Three", "4": "Four",  "5": "Open Hand",
        }
        classes.append({
            "label":     int(label_str),
            "name":      label_names.get(label_str, label_str),
            "precision": round(d.get("precision", 0), 6),
            "recall":    round(d.get("recall",    0), 6),
            "f1":        round(d.get("f1_score",  0), 6),
            "support":   int(d.get("support", 0)),
        })

    latency = eval_data.get("latency", {})

    return {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "accuracy":      round(eval_data.get("accuracy", 0) * 100, 2),   # percent
        "test_samples":  eval_data.get("test_samples", 0),
        "mean_latency_ms": round(latency.get("mean_ms", 0), 1),
        "p95_latency_ms":  round(latency.get("p95_ms", 0), 1),
        "classes":       classes,
        # Training info (from training_summary.json if present)
        "train_accuracy":   round(train_data.get("test_accuracy",  0) * 100, 2),
        "train_epochs":     train_data.get("epochs_trained", None),
        "model_parameters": train_data.get("model_parameters", None),
        # Raw F1 list (convenient for Chart.js)
        "f1_scores": [c["f1"] for c in classes],
        "f1_labels": [f"{c['label']} ({c['name']})" for c in classes],
    }


def main():
    print("[sync_web_metrics] Reading evaluation and training reports...")

    eval_data  = load_json(EVAL_REPORT)
    train_data = load_json(TRAIN_SUMMARY)

    if not eval_data:
        print("[ERROR] evaluation_report.json is missing.")
        print("  Run:  python tools/evaluate_model.py")
        sys.exit(1)

    metrics = build_metrics(eval_data, train_data)

    WEB_METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(WEB_METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[sync_web_metrics] OK Wrote web/metrics.json")
    print(f"  Accuracy  : {metrics['accuracy']}%")
    print(f"  Samples   : {metrics['test_samples']}")
    print(f"  Latency   : {metrics['mean_latency_ms']} ms")
    for c in metrics["classes"]:
        print(f"  Class {c['label']} ({c['name']:<10}): F1 = {c['f1']:.4f}")


if __name__ == "__main__":
    main()
