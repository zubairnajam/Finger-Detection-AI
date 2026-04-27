"""
Extract MediaPipe hand landmarks from local raw captures and archive images.
Takes the raw images you captured (and any archived images), extracts the 3D coordinates (x, y, z) of your hand landmarks using MediaPipe, and processes them into structured feature arrays (.npy and .csv files). These files are split into training and testing sets inside the data directory.

This script is intentionally rerunnable: whenever you add more images to
data/raw or place more samples inside archive/, run it again and it will
rebuild the processed dataset from scratch.

If an external archive already provides train/test folders, that split is kept
intact. Local raw captures are split into train/test during preprocessing.

Usage:
    python tools/preprocess_landmarks.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List
import csv
import json
import re

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"
VALID_LABELS = tuple(str(label) for label in range(6))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
TEST_SIZE = 0.2
RANDOM_STATE = 42
DETECTION_CONFIDENCE = 0.1
RESCUE_IMAGE_MAX_DIMENSION = 192
RESCUE_SCALE = 3.0
RESCUE_TARGET_SIZE = 512
RESCUE_ROTATIONS = (12.0, -12.0)
ARCHIVE_LABEL_PATTERN = re.compile(r"_(\d+)[A-Za-z]+$")
ARCHIVE_SPLIT_DIR_CANDIDATES = {
    "train": (ARCHIVE_DIR / "fingers" / "train", ARCHIVE_DIR / "train"),
    "test": (ARCHIVE_DIR / "fingers" / "test", ARCHIVE_DIR / "test"),
}


def ensure_output_dirs() -> None:
    """Create dataset output folders if they do not already exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def build_detector() -> HandLandmarker:
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
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=DETECTION_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def list_image_paths(directory: Path) -> List[Path]:
    """Return all supported image files below a directory in a stable order."""
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def convert_image_to_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize grayscale/BGR/BGRA images into RGB for MediaPipe."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_features_in_rgb_image(
    image_rgb: np.ndarray,
    detector: HandLandmarker,
) -> List[float] | None:
    """Run MediaPipe on an RGB image and flatten the first detected hand."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    features: List[float] = []
    for landmark in result.hand_landmarks[0]:
        features.extend([landmark.x, landmark.y, landmark.z])
    return features


def make_white_padded_rescue_image(
    image_rgb: np.ndarray,
    rotation_degrees: float = 0.0,
) -> np.ndarray:
    """Upscale a small crop and center it on a white square canvas."""
    working = image_rgb
    if rotation_degrees:
        height, width = working.shape[:2]
        center = (width / 2.0, height / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
        working = cv2.warpAffine(
            working,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    resized = cv2.resize(
        working,
        None,
        fx=RESCUE_SCALE,
        fy=RESCUE_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    height, width = resized.shape[:2]
    top = max((RESCUE_TARGET_SIZE - height) // 2, 0)
    bottom = max(RESCUE_TARGET_SIZE - height - top, 0)
    left = max((RESCUE_TARGET_SIZE - width) // 2, 0)
    right = max(RESCUE_TARGET_SIZE - width - left, 0)
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def generate_detection_variants(
    image_path: Path,
    image_rgb: np.ndarray,
) -> List[tuple[str, np.ndarray]]:
    """Build an ordered list of image variants to maximize landmark recovery."""
    variants: List[tuple[str, np.ndarray]] = [("original", image_rgb)]
    is_small_image = max(image_rgb.shape[:2]) <= RESCUE_IMAGE_MAX_DIMENSION
    use_rescue_variants = image_path.is_relative_to(ARCHIVE_DIR) or is_small_image
    if not use_rescue_variants:
        return variants

    variants.append(("rescue_white_pad", make_white_padded_rescue_image(image_rgb)))
    for rotation_degrees in RESCUE_ROTATIONS:
        rotation_suffix = str(int(rotation_degrees)).replace("-", "neg_")
        variants.append(
            (
                f"rescue_white_pad_rot_{rotation_suffix}",
                make_white_padded_rescue_image(
                    image_rgb,
                    rotation_degrees=rotation_degrees,
                ),
            )
        )
    return variants


def extract_features(
    image_path: Path,
    detector: HandLandmarker,
) -> tuple[List[float] | None, str | None]:
    """Return flattened features plus the variant that successfully produced them."""
    sidecar_path = image_path.with_suffix(".json")
    if sidecar_path.exists():
        sidecar_features = load_features_from_sidecar(sidecar_path)
        if sidecar_features is not None:
            return sidecar_features, "sidecar"

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None

    image_rgb = convert_image_to_rgb(image)
    for variant_name, variant_rgb in generate_detection_variants(image_path, image_rgb):
        features = detect_features_in_rgb_image(variant_rgb, detector)
        if features is not None:
            return features, variant_name

    return None, None


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


def infer_archive_label(image_path: Path) -> str | None:
    """Extract a numeric label from archive filenames like *_3L.png or *_0R.jpg."""
    match = ARCHIVE_LABEL_PATTERN.search(image_path.stem)
    if match is None:
        return None

    label = match.group(1)
    if label not in VALID_LABELS:
        return None
    return label


def resolve_archive_split_dir(split: str) -> Path | None:
    """Pick one canonical archive split directory and avoid duplicate layouts."""
    for candidate_dir in ARCHIVE_SPLIT_DIR_CANDIDATES.get(split, ()):
        if not candidate_dir.exists():
            continue

        has_images = any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in candidate_dir.rglob("*")
        )
        if has_images:
            return candidate_dir

    return None


def discover_archive_image_paths(split: str) -> List[Path]:
    """Find all archive images for a given split across supported layouts."""
    candidate_dir = resolve_archive_split_dir(split)
    if candidate_dir is None:
        return []
    return list_image_paths(candidate_dir)


def count_local_raw_images_per_label() -> Dict[str, int]:
    """Count local raw images stored in data/raw by label."""
    counts = {label: 0 for label in VALID_LABELS}
    for label in VALID_LABELS:
        label_dir = RAW_DATA_DIR / label
        counts[label] = sum(
            1
            for path in label_dir.iterdir()
            if label_dir.exists()
            and path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if label_dir.exists() else 0
    return counts


def count_archive_images_per_label(split: str) -> Dict[str, int]:
    """Count archive images for a split after inferring labels from filenames."""
    counts = {label: 0 for label in VALID_LABELS}
    for image_path in discover_archive_image_paths(split):
        label = infer_archive_label(image_path)
        if label is not None:
            counts[label] += 1
    return counts


def append_skipped_sample(
    skipped: List[Dict[str, str]],
    image_path: Path,
    source: str,
    split: str,
    reason: str,
) -> None:
    """Record a sample that could not be converted into landmarks."""
    skipped.append(
        {
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "source": source,
            "split": split,
            "reason": reason,
        }
    )


def build_record(
    image_path: Path,
    label: str,
    features: List[float],
    source: str,
    split: str,
    detection_variant: str | None,
) -> Dict[str, object]:
    """Create one dataset record with metadata and extracted features."""
    return {
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "label": int(label),
        "features": features,
        "source": source,
        "split": split,
        "detection_variant": detection_variant,
    }


def collect_local_records(
    detector: HandLandmarker,
    skipped: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    """Process local raw captures stored under data/raw/<label>/."""
    records: List[Dict[str, object]] = []

    for label in VALID_LABELS:
        label_dir = RAW_DATA_DIR / label
        for image_path in list_image_paths(label_dir):
            features, detection_variant = extract_features(image_path, detector)
            if features is None:
                reason = (
                    "No hand landmarks detected after fallback variants and no usable sidecar found"
                    if not image_path.with_suffix(".json").exists()
                    else "Landmark sidecar exists but is invalid and fallback image detection failed"
                )
                append_skipped_sample(skipped, image_path, "local_raw", "unsplit", reason)
                continue

            records.append(
                build_record(
                    image_path=image_path,
                    label=label,
                    features=features,
                    source="local_raw",
                    split="unsplit",
                    detection_variant=detection_variant,
                )
            )

    return records


def collect_archive_records(
    detector: HandLandmarker,
    split: str,
    skipped: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    """Process archive images while preserving their existing train/test split."""
    records: List[Dict[str, object]] = []

    for image_path in discover_archive_image_paths(split):
        label = infer_archive_label(image_path)
        if label is None:
            append_skipped_sample(
                skipped,
                image_path,
                "archive",
                split,
                "Could not infer a valid label from the filename",
            )
            continue

        features, detection_variant = extract_features(image_path, detector)
        if features is None:
            append_skipped_sample(
                skipped,
                image_path,
                "archive",
                split,
                "No hand landmarks detected after fallback variants",
            )
            continue

        records.append(
            build_record(
                image_path=image_path,
                label=label,
                features=features,
                source="archive",
                split=split,
                detection_variant=detection_variant,
            )
        )

    return records


def split_local_records(
    records: List[Dict[str, object]],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Split local raw records into train/test, using stratification when possible."""
    if not records:
        return [], []

    if len(records) < 2:
        for record in records:
            record["split"] = "train"
        return records, []

    labels = [int(record["label"]) for record in records]
    label_counts = Counter(labels)
    stratify = (
        labels
        if len(label_counts) > 1 and all(count >= 2 for count in label_counts.values())
        else None
    )

    try:
        train_records, test_records = train_test_split(
            records,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError:
        if len(records) < 3:
            train_records, test_records = records, []
        else:
            train_records, test_records = train_test_split(
                records,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

    for record in train_records:
        record["split"] = "train"
    for record in test_records:
        record["split"] = "test"

    return list(train_records), list(test_records)


def count_processed_records_per_label(records: List[Dict[str, object]]) -> Dict[str, int]:
    """Count processed records by label using the current project label range."""
    counts = {label: 0 for label in VALID_LABELS}
    for record in records:
        counts[str(record["label"])] += 1
    return counts


def count_processed_records_per_variant(records: List[Dict[str, object]]) -> Dict[str, int]:
    """Count how often each extraction path succeeded."""
    counts = Counter(
        str(record["detection_variant"] or "unknown")
        for record in records
    )
    return dict(sorted(counts.items()))


def main() -> None:
    """Extract landmarks, save a full dataset, and create combined train/test arrays."""
    ensure_output_dirs()
    detector = build_detector()

    skipped: List[Dict[str, str]] = []
    local_records: List[Dict[str, object]] = []
    archive_train_records: List[Dict[str, object]] = []
    archive_test_records: List[Dict[str, object]] = []

    try:
        local_records = collect_local_records(detector, skipped)
        archive_train_records = collect_archive_records(detector, "train", skipped)
        archive_test_records = collect_archive_records(detector, "test", skipped)
    finally:
        detector.close()

    local_train_records, local_test_records = split_local_records(local_records)
    train_records = local_train_records + archive_train_records
    test_records = local_test_records + archive_test_records
    records = train_records + test_records

    if not records:
        with (PROCESSED_DIR / "skipped_samples.json").open("w", encoding="utf-8") as skipped_file:
            json.dump(skipped, skipped_file, indent=2)
        raise RuntimeError(
            "No valid samples were processed. Add local captures to data/raw or place "
            "archive images under archive/fingers/train and archive/fingers/test."
        )

    if not train_records or not test_records:
        with (PROCESSED_DIR / "skipped_samples.json").open("w", encoding="utf-8") as skipped_file:
            json.dump(skipped, skipped_file, indent=2)
        raise RuntimeError(
            "Dataset build requires at least one training sample and one test sample."
        )

    X = np.array([record["features"] for record in records], dtype=np.float32)
    y = np.array([record["label"] for record in records], dtype=np.int64)
    X_train = np.array([record["features"] for record in train_records], dtype=np.float32)
    y_train = np.array([record["label"] for record in train_records], dtype=np.int64)
    X_test = np.array([record["features"] for record in test_records], dtype=np.float32)
    y_test = np.array([record["label"] for record in test_records], dtype=np.int64)

    full_rows = [[record["image_path"], record["label"], *record["features"]] for record in records]
    train_rows = [[record["image_path"], record["label"], *record["features"]] for record in train_records]
    test_rows = [[record["image_path"], record["label"], *record["features"]] for record in test_records]

    save_csv(full_rows, PROCESSED_DIR / "landmarks.csv")
    save_csv(train_rows, TRAIN_DIR / "train_landmarks.csv")
    save_csv(test_rows, TEST_DIR / "test_landmarks.csv")

    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y)
    np.save(TRAIN_DIR / "X_train.npy", X_train)
    np.save(TRAIN_DIR / "y_train.npy", y_train)
    np.save(TEST_DIR / "X_test.npy", X_test)
    np.save(TEST_DIR / "y_test.npy", y_test)

    processed_samples_per_source = {
        "local_raw_train": sum(1 for record in train_records if record["source"] == "local_raw"),
        "local_raw_test": sum(1 for record in test_records if record["source"] == "local_raw"),
        "archive_train": sum(1 for record in train_records if record["source"] == "archive"),
        "archive_test": sum(1 for record in test_records if record["source"] == "archive"),
    }

    summary = {
        "labels": [int(label) for label in VALID_LABELS],
        "local_raw_images_per_label": count_local_raw_images_per_label(),
        "archive_images_per_split_per_label": {
            "train": count_archive_images_per_label("train"),
            "test": count_archive_images_per_label("test"),
        },
        "processed_samples_per_label": count_processed_records_per_label(records),
        "train_samples_per_label": count_processed_records_per_label(train_records),
        "test_samples_per_label": count_processed_records_per_label(test_records),
        "processed_samples_per_variant": count_processed_records_per_variant(records),
        "processed_samples_per_source": processed_samples_per_source,
        "processed_samples_total": int(len(records)),
        "skipped_samples_total": int(len(skipped)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_size": int(X.shape[1]),
        "rerun_note": (
            "Add more images to data/raw or archive and rerun this script to rebuild "
            "the dataset."
        ),
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
    print(
        "Sources: "
        f"local_raw_train={processed_samples_per_source['local_raw_train']}, "
        f"local_raw_test={processed_samples_per_source['local_raw_test']}, "
        f"archive_train={processed_samples_per_source['archive_train']}, "
        f"archive_test={processed_samples_per_source['archive_test']}"
    )
    print(
        "Variants: "
        + ", ".join(
            f"{name}={count}"
            for name, count in count_processed_records_per_variant(records).items()
        )
    )
    print("Outputs written to data/processed, data/train, and data/test.")


if __name__ == "__main__":
    main()
