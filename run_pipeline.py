"""
An end-to-end pipeline script that runs the entire finger detection workflow sequentially.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run_script(script_path: str, optional: bool = False):
    """Run a Python script as a subprocess."""
    print(f"\n{'='*60}")
    print(f"▶️ Starting: {script_path}")
    print(f"{'='*60}\n")

    try:
        subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            check=True
        )
    except subprocess.CalledProcessError as e:
        if optional:
            print(f"\n⚠️  Optional step '{script_path}' failed (exit {e.returncode}). Continuing...")
        else:
            print(f"\n❌ Error: {script_path} failed with exit code {e.returncode}.")
            print("Pipeline aborted.")
            sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n⚠️ Process {script_path} interrupted by user.")
        print("Pipeline aborted. You can run individual stages later if needed.")
        sys.exit(1)

def main():
    print("🚀 Starting the full Finger Detection AI pipeline...")
    print("The pipeline will run the following stages in order:")
    print("1. Data Collection     (Press 'q' when done recording)")
    print("2. Preprocess          (Automatic feature extraction)")
    print("3. Dataset Health      (Summary of processed data)")
    print("4. Train Model         (AI training)")
    print("5. Evaluate Model      (Accuracy, F1, latency benchmark)")
    print("6. Sync Web Metrics    (Update website charts & stats)")
    print("7. Convert to TF.js    (Update browser model)")
    print("8. Smart Camera        (Launch the final camera app)")

    # Core stages — must succeed
    core_stages = [
        "tools/data_collection.py",
        "tools/preprocess_landmarks.py",
        "tools/dataset_health_check.py",
        "tools/train_model.py",
        "tools/evaluate_model.py",
    ]

    # Web sync stages
    web_stages = [
        ("tools/sync_web_metrics.py",       False),  # required — writes web/metrics.json
        ("tools/convert_model_to_tfjs.py",  False),   # optional — needs: pip install tensorflowjs
    ]

    for script in core_stages:
        run_script(script)

    for script, optional in web_stages:
        run_script(script, optional=optional)

    run_script("src/final_camera_app.py")

    print(f"\n{'='*60}")
    print("🎉 Pipeline completed successfully!")
    print("   • web/metrics.json  → website metrics are up to date")
    print("   • web/model/        → browser model updated (if tensorflowjs installed)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
