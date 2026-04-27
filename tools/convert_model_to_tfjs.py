"""
convert_model_to_tfjs.py
────────────────────────
Converts the trained Keras model + scaler to TF.js format so the website
can run inference directly in the browser.

Run this once after training (or any time you retrain):
    python tools/convert_model_to_tfjs.py

Requirements (install once):
    pip install tensorflowjs
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "finger_count_model.keras"
SCALER_PATH  = PROJECT_ROOT / "models" / "feature_scaler.npz"
TFJS_OUT_DIR = PROJECT_ROOT / "web" / "model"          # served as static files
SCALER_JSON  = PROJECT_ROOT / "web" / "model" / "scaler.json"


def check_tensorflowjs():
    try:
        import tensorflowjs  # noqa: F401
    except ImportError:
        print("[ERROR] tensorflowjs is not installed.")
        print("  Run:  pip install tensorflowjs")
        sys.exit(1)


def convert_model():
    print(f"[1/3] Converting Keras model → TF.js...")
    print(f"      Source : {MODEL_PATH}")
    print(f"      Target : {TFJS_OUT_DIR}")

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  Run the training pipeline first: python run_pipeline.py")
        sys.exit(1)

    TFJS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "tensorflowjs.converters.converter",
            "--input_format=keras",
            str(MODEL_PATH),
            str(TFJS_OUT_DIR),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[ERROR] Conversion failed:")
        print(result.stderr)
        sys.exit(1)
    print("      ✓ Model converted successfully")


def export_scaler():
    print(f"[2/3] Exporting scaler parameters → scaler.json...")

    if not SCALER_PATH.exists():
        print(f"[WARNING] Scaler not found at {SCALER_PATH}. Skipping scaler export.")
        print("          The website will skip feature normalisation.")
        return

    data = np.load(SCALER_PATH)
    scaler_dict = {
        "mean": data["mean"].tolist(),
        "scale": data["scale"].tolist(),
    }
    SCALER_JSON.write_text(json.dumps(scaler_dict, indent=2))
    print(f"      ✓ Scaler exported ({len(scaler_dict['mean'])} features)")


def print_summary():
    print("[3/3] Summary of files written to web/model/:")
    for f in sorted(TFJS_OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"      {f.name:<35} {size_kb:>7.1f} KB")
    print()
    print("✅  Done! The website will now use your trained AI model.")
    print("   Next step → run:  python tools/sync_web_metrics.py")


if __name__ == "__main__":
    check_tensorflowjs()
    convert_model()
    export_scaler()
    print_summary()
