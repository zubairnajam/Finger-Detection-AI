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
        import numpy as np
        np.object = object
        np.bool = np.bool_
        np.complex = complex
        
        import sys
        from unittest.mock import MagicMock
        sys.modules['tensorflow_hub'] = MagicMock()

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

    wrapper_path = PROJECT_ROOT / "tfjs_wrapper.py"
    wrapper_path.write_text(
        "import sys\n"
        "import numpy as np\n"
        "np.object = object\n"
        "np.bool = np.bool_\n"
        "np.complex = complex\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules['tensorflow_hub'] = MagicMock()\n"
        "import tensorflow as tf\n"
        "import tensorflowjs as tfjs\n"
        "import tensorflowjs.converters.keras_h5_conversion as cvt\n"
        "cvt._check_version = lambda h5file: None\n"
        "model = tf.keras.models.load_model(sys.argv[1])\n"
        "tfjs.converters.save_keras_model(model, sys.argv[2])\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(wrapper_path),
            str(MODEL_PATH),
            str(TFJS_OUT_DIR),
        ],
        capture_output=True,
        text=True,
    )
    if wrapper_path.exists():
        wrapper_path.unlink()

    if result.returncode != 0:
        print("[ERROR] Conversion failed:")
        print(result.stderr)
        sys.exit(1)

    # FIX: TF.js v3.18/v4 converter has a bug with Keras 3 models where it exports "batch_shape" 
    # instead of "batchInputShape" for the InputLayer, causing a load failure in the browser.
    model_json_path = TFJS_OUT_DIR / "model.json"
    if model_json_path.exists():
        content = model_json_path.read_text(encoding="utf-8")
        content = content.replace('"batch_shape":', '"batchInputShape":')
        model_json_path.write_text(content, encoding="utf-8")

    print("      ✓ Model converted successfully")


def export_scaler():
    print(f"[2/3] Exporting scaler parameters → scaler.json...")

    if not SCALER_PATH.exists():
        print(f"[WARNING] Scaler not found at {SCALER_PATH}. Skipping scaler export.")
        print("          The website will skip feature normalisation.")
        return

    # train_model.py stores the standard deviation as "std"; the web app reads
    # it under the key "scale" (sklearn naming). Zero std is mapped to 1.
    data = np.load(SCALER_PATH)
    std = data["std"].astype(float)
    std[std == 0] = 1.0
    scaler_dict = {
        "mean": data["mean"].tolist(),
        "scale": std.tolist(),
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