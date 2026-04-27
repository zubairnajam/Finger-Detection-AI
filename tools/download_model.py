"""
Download the MediaPipe hand landmarker task file used by the project.
Downloads the pre-trained MediaPipe hand landmark detection model (hand_landmarker.task) from Google. This is a setup script to ensure you have the base model needed for the project to work.

Usage:
    python tools/download_model.py
"""

from pathlib import Path
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"


def download_model() -> bool:
    """Download the MediaPipe hand landmarker model if it is not present."""
    if MODEL_PATH.exists():
        file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {MODEL_PATH.name} ({file_size_mb:.2f} MB)")
        return True

    print("Downloading MediaPipe hand landmarker model...")
    print(f"URL: {MODEL_URL}")

    try:
        def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(
                f"\rProgress: {percent:.1f}% ({mb_downloaded:.2f} MB / {mb_total:.2f} MB)",
                end="",
            )

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, progress_hook)
        print()

        if MODEL_PATH.exists():
            file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            print(f"Download complete: {MODEL_PATH.name} ({file_size_mb:.2f} MB)")
            return True

        print("Download failed: file not found after download.")
        return False
    except Exception as exc:
        print(f"\nError downloading model: {exc}")
        print("\nManual download:")
        print(f"1. Open: {MODEL_URL}")
        print(f"2. Save as: {MODEL_PATH.name}")
        print(f"3. Place it in: {PROJECT_ROOT}")
        return False


if __name__ == "__main__":
    download_model()
