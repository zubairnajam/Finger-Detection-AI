"""
Turn raw MediaPipe landmarks into position- and scale-invariant features.

MediaPipe returns landmark x/y in image coordinates (0-1), so the raw vector
encodes *where* the hand is in the frame as much as *what shape* it makes.
The training images are tightly centred, which meant a hand held off-centre or
further from the webcam looked unfamiliar to the classifier (e.g. 4 read as 5).

Features (85 values):
    * 63  landmarks relative to the wrist, scaled by wrist -> middle-knuckle length
    * 15  joint bend cosines (3 joints x 5 fingers)
    *  5  fingertip -> wrist distances
    *  2  thumb-tip -> index knuckle / pinky knuckle distances (thumb tucked or out)

``web/mlp.js`` mirrors this function exactly; ``tests/test_web_engine.py``
checks the two stay in sync.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
RAW_FEATURE_SIZE = NUM_LANDMARKS * 3
FEATURE_SIZE = 85
FEATURE_VERSION = "normalized-v1"

WRIST, MIDDLE_MCP = 0, 9
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
FINGERTIPS = (4, 8, 12, 16, 20)
THUMB_TIP, INDEX_MCP, PINKY_MCP = 4, 5, 17
EPS = 1e-6
WEBCAM_ASPECT = 640 / 480
ARCHIVE_ASPECT = 1.0


def build_features(landmarks: np.ndarray, aspect: float | np.ndarray = 1.0) -> np.ndarray:
    """Convert raw ``(63,)`` or ``(N, 63)`` landmark vectors to ``FEATURE_SIZE`` features.

    ``aspect`` is the source image's width / height. MediaPipe normalises x by
    width and y by height, so x is rescaled to keep the hand's true proportions.
    It may be a scalar or one value per row.
    """
    raw = np.asarray(landmarks, dtype=np.float32)
    single = raw.ndim == 1
    points = raw.reshape(-1, NUM_LANDMARKS, 3).copy()

    aspect = np.broadcast_to(np.asarray(aspect, dtype=np.float32), (points.shape[0],))
    points[:, :, 0] *= aspect[:, None]
    points -= points[:, WRIST:WRIST + 1]
    scale = np.linalg.norm(points[:, MIDDLE_MCP, :2], axis=1)
    points /= np.maximum(scale, EPS)[:, None, None]

    extra = []
    for chain in FINGER_CHAINS:
        for a, b, c in zip(chain, chain[1:], chain[2:]):
            u = points[:, a] - points[:, b]
            v = points[:, c] - points[:, b]
            denom = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + EPS
            extra.append(np.sum(u * v, axis=1) / denom)
    for tip in FINGERTIPS:
        extra.append(np.linalg.norm(points[:, tip], axis=1))
    extra.append(np.linalg.norm(points[:, THUMB_TIP] - points[:, INDEX_MCP], axis=1))
    extra.append(np.linalg.norm(points[:, THUMB_TIP] - points[:, PINKY_MCP], axis=1))

    features = np.hstack([points.reshape(-1, RAW_FEATURE_SIZE), np.stack(extra, axis=1)])
    features = features.astype(np.float32)
    return features[0] if single else features


def image_aspects(image_paths) -> np.ndarray:
    """Width / height of each dataset image, inferred from where it came from.

    Webcam captures in ``data/raw/`` are 640x480; the public archive images are square.
    """
    return np.array(
        [WEBCAM_ASPECT if str(p).replace("\\", "/").startswith("data/raw/") else ARCHIVE_ASPECT
         for p in image_paths],
        dtype=np.float32,
    )


def mirror_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Flip raw landmarks horizontally (left hand <-> right hand) for augmentation."""
    mirrored = np.asarray(landmarks, dtype=np.float32).reshape(-1, NUM_LANDMARKS, 3).copy()
    mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]
    return mirrored.reshape(np.shape(landmarks))
