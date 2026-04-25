"""
Extract MediaPipe hand landmarks from raw images and create train/test datasets.

This script is intentionally rerunnable: whenever you add more images to data/raw,
run it again and it will rebuild the processed dataset from scratch.

Usage:
    python src/preprocess_landmarks.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import csv
import json

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"
VALID_LABELS = ("1", "2", "3", "4", "5")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
TEST_SIZE = 0.2
RANDOM_STATE = 42


def ensure_output_dirs() -> None:
    """Create dataset output folders if they do not already exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def build_detector() -> vision.HandLandmarker:
    """Create a MediaPipe detector configured for still images."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Download hand_landmarker.task first."
        )

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_features(image_path: Path, detector: vision.HandLandmarker) -> List[float] | None:
    """Return flattened 63-value landmark features for the first detected hand."""
    sidecar_path = image_path.with_suffix(".json")
    if sidecar_path.exists():
        return load_features_from_sidecar(sidecar_path)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]
    features: List[float] = []
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    return features


def load_features_from_sidecar(sidecar_path: Path) -> List[float] | None:
    """Load flattened landmark features from a capture-time JSON sidecar."""
    with sidecar_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    hand_landmarks = payload.get("hand_landmarks") or []
    if not hand_landmarks:
        return None

    first_hand = hand_landmarks[0]
    if len(first_hand) != 21:
        return None

    features: List[float] = []
    for landmark in first_hand:
        try:
            features.extend(
                [float(landmark["x"]), float(landmark["y"]), float(landmark["z"])]
            )
        except (KeyError, TypeError, ValueError):
            return None
    return features


def build_header() -> List[str]:
    """Create CSV column names for 21 x/y/z landmarks plus metadata."""
    header = ["image_path", "label"]
    for idx in range(21):
        header.extend([f"x_{idx}", f"y_{idx}", f"z_{idx}"])
    return header


def save_csv(rows: List[List[float | str]], output_path: Path) -> None:
    """Write rows to CSV with a stable header."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_header())
        writer.writerows(rows)


def main() -> None:
    """Extract landmarks, save a full dataset, and create a train/test split."""
    ensure_output_dirs()
    detector = build_detector()

    records: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []
    per_label_counts = {label: 0 for label in VALID_LABELS}

    try:
        for label in VALID_LABELS:
            label_dir = RAW_DATA_DIR / label
            if not label_dir.exists():
                continue

            image_paths = sorted(
                path for path in label_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in image_paths:
                features = extract_features(image_path, detector)
                if features is None:
                    reason = (
                        "No hand landmarks detected and no usable sidecar found"
                        if not image_path.with_suffix(".json").exists()
                        else "Landmark sidecar exists but is invalid"
                    )
                    skipped.append(
                        {
                            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                            "reason": reason,
                        }
                    )
                    continue

                records.append(
                    {
                        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                        "label": int(label),
                        "features": features,
                    }
                )
                per_label_counts[label] += 1
    finally:
        detector.close()

    if not records:
        with (PROCESSED_DIR / "skipped_samples.json").open("w", encoding="utf-8") as skipped_file:
            json.dump(skipped, skipped_file, indent=2)
        raise RuntimeError(
            "No valid samples were processed. Capture new samples with the updated "
            "collector so raw images and landmark sidecars are saved."
        )

    X = np.array([record["features"] for record in records], dtype=np.float32)
    y = np.array([record["label"] for record in records], dtype=np.int64)
    image_paths = np.array([record["image_path"] for record in records], dtype=object)

    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X,
        y,
        image_paths,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    full_rows = [[record["image_path"], record["label"], *record["features"]] for record in records]
    train_rows = [[path, label, *features] for path, label, features in zip(paths_train, y_train, X_train)]
    test_rows = [[path, label, *features] for path, label, features in zip(paths_test, y_test, X_test)]

    save_csv(full_rows, PROCESSED_DIR / "landmarks.csv")
    save_csv(train_rows, TRAIN_DIR / "train_landmarks.csv")
    save_csv(test_rows, TEST_DIR / "test_landmarks.csv")

    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y)
    np.save(TRAIN_DIR / "X_train.npy", X_train)
    np.save(TRAIN_DIR / "y_train.npy", y_train)
    np.save(TEST_DIR / "X_test.npy", X_test)
    np.save(TEST_DIR / "y_test.npy", y_test)

    summary = {
        "raw_samples_per_label": {
            label: sum(
                1
                for path in (RAW_DATA_DIR / label).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if (RAW_DATA_DIR / label).exists() else 0
            for label in VALID_LABELS
        },
        "processed_samples_per_label": per_label_counts,
        "processed_samples_total": int(len(records)),
        "skipped_samples_total": int(len(skipped)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_size": int(X.shape[1]),
        "rerun_note": "Add more images to data/raw and rerun this script to rebuild the dataset.",
    }

    with (PROCESSED_DIR / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    with (PROCESSED_DIR / "skipped_samples.json").open("w", encoding="utf-8") as skipped_file:
        json.dump(skipped, skipped_file, indent=2)

    print("Landmark preprocessing complete.")
    print(f"Processed samples: {len(records)}")
    print(f"Skipped samples: {len(skipped)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("Outputs written to data/processed, data/train, and data/test.")


if __name__ == "__main__":
    main()
