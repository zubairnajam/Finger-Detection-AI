"""
Quick dataset inspection for raw captures and processed landmark outputs.

Usage:
    python tools/dataset_health_check.py
"""

from __future__ import annotations

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
VALID_LABELS = tuple(str(label) for label in range(6))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_label_files(label: str) -> tuple[int, int]:
    """Return image count and sidecar-json count for one label."""
    label_dir = RAW_DATA_DIR / label
    if not label_dir.exists():
        return 0, 0

    image_count = sum(
        1 for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    json_count = sum(
        1 for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )
    return image_count, json_count


def count_archive_split_images(split: str) -> int:
    """Return the number of archive images in a supported split directory."""
    for candidate in (ARCHIVE_DIR / "fingers" / split, ARCHIVE_DIR / split):
        if not candidate.exists():
            continue

        image_count = sum(
            1
            for path in candidate.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_count:
            return image_count

    return 0


def main() -> None:
    print("Raw capture health")
    total_images = 0
    total_sidecars = 0
    for label in VALID_LABELS:
        image_count, json_count = count_label_files(label)
        total_images += image_count
        total_sidecars += json_count
        print(f"  Label {label}: {image_count} images, {json_count} landmark sidecars")

    print(f"  Total: {total_images} images, {total_sidecars} landmark sidecars")

    archive_train_images = count_archive_split_images("train")
    archive_test_images = count_archive_split_images("test")
    print("\nArchive dataset health")
    print(f"  Train images: {archive_train_images}")
    print(f"  Test images: {archive_test_images}")

    summary_path = PROCESSED_DIR / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        print("\nLatest processed summary")
        print(f"  Processed samples: {summary.get('processed_samples_total', 0)}")
        print(f"  Skipped samples: {summary.get('skipped_samples_total', 0)}")
        print(f"  Train samples: {summary.get('train_samples', 0)}")
        print(f"  Test samples: {summary.get('test_samples', 0)}")
        print(f"  Feature size: {summary.get('feature_size', 0)}")
        print("  Processed by source:")
        for source, count in summary.get("processed_samples_per_source", {}).items():
            print(f"    {source}: {count}")
        print("  Processed by variant:")
        for variant, count in summary.get("processed_samples_per_variant", {}).items():
            print(f"    {variant}: {count}")
        print("  Processed per label:")
        for label, count in summary.get("processed_samples_per_label", {}).items():
            print(f"    {label}: {count}")
    else:
        print("\nNo processed summary yet. Run python tools\\preprocess_landmarks.py after collecting data.")

    skipped_path = PROCESSED_DIR / "skipped_samples.json"
    if skipped_path.exists():
        with skipped_path.open("r", encoding="utf-8") as handle:
            skipped = json.load(handle)
        print(f"\nSkipped sample records: {len(skipped)}")
    else:
        print("\nNo skipped-sample log yet.")

    print("\nExpected next step:")
    print("  1. Capture clean samples with python tools\\data_collection.py")
    print("  2. Rebuild the dataset with python tools\\preprocess_landmarks.py")
    print("  3. Retrain the model with python tools\\train_model.py")


if __name__ == "__main__":
    main()
