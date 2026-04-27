"""
An end-to-end pipeline script that runs the entire finger detection workflow sequentially.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run_script(script_path: str):
    """Run a Python script as a subprocess."""
    print(f"\n{'='*60}")
    print(f"▶️ Starting: {script_path}")
    print(f"{'='*60}\n")
    
    try:
        # Run the script, passing output through to the console
        subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            check=True
        )
    except subprocess.CalledProcessError as e:
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
    print("1. Data Collection (Press 'q' when you are done recording)")
    print("2. Preprocess Landmarks (Automatic feature extraction)")
    print("3. Dataset Health Check (Summary of your processed data)")
    print("4. Train Model (Automatic AI training)")
    print("5. Evaluate Model (Accuracy, per-class metrics, latency benchmark)")
    print("6. Smart Camera (Launch the final camera app)")
    
    scripts_to_run = [
        "tools/data_collection.py",
        "tools/preprocess_landmarks.py",
        "tools/dataset_health_check.py",
        "tools/train_model.py",
        "tools/evaluate_model.py",
        "src/final_camera_app.py",
    ]
    
    for script in scripts_to_run:
        run_script(script)
        
    print(f"\n{'='*60}")
    print("🎉 Pipeline completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
