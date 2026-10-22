"""
Export the trained classifier for the browser demo - no tensorflowjs needed.

Writes:
    web/model/mlp.json   layer shapes, activations, scaler stats, metadata
    web/model/mlp.bin    little-endian float32 weights (kernel then bias, per layer)

The website runs the forward pass in ~40 lines of plain JavaScript
(web/mlp.js), so it does not need to download the TF.js runtime.

Usage:
    python tools/export_web_model.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CLASS_NAMES, LABELS, WEB_DIR
from src.inference import NumpyMLP
from src.prediction import load_model_and_scaler

OUT_DIR = WEB_DIR / "model"


def export(model: NumpyMLP, mean: np.ndarray, std: np.ndarray, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[np.ndarray] = []
    layers = []
    for layer in model.layers:
        layers.append({
            "inputs": int(layer.weights.shape[0]),
            "units": int(layer.weights.shape[1]),
            "activation": layer.activation,
        })
        chunks += [layer.weights.astype("<f4").ravel(order="C"), layer.bias.astype("<f4")]

    blob = np.concatenate(chunks).astype("<f4")
    bin_path = out_dir / "mlp.bin"
    json_path = out_dir / "mlp.json"
    bin_path.write_bytes(blob.tobytes())

    manifest = {
        "format": "finger-mlp/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights_file": bin_path.name,
        "weights_float32_count": int(blob.size),
        "parameters": model.count_params(),
        "labels": LABELS,
        "class_names": [CLASS_NAMES[label] for label in LABELS],
        "layers": layers,
        "scaler": {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()},
    }
    json_path.write_text(json.dumps(manifest), encoding="utf-8")
    return json_path, bin_path


def main() -> None:
    model, mean, std = load_model_and_scaler(backend="numpy")
    if not isinstance(model, NumpyMLP):
        sys.exit("Model architecture is not supported by the web engine (Dense layers only).")
    json_path, bin_path = export(model, mean, std)
    print(f"Exported {model.count_params():,} parameters")
    print(f"  {json_path.relative_to(_PROJECT_ROOT)}  ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {bin_path.relative_to(_PROJECT_ROOT)}   ({bin_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
