"""
Centralized configuration: paths, constants, labels, and shared dataclasses.

Every module in the project should import paths and tuning knobs from here
so they stay consistent across the camera app, tools, and evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "finger_count_model.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "feature_scaler.npz"
HAND_LANDMARKER_PATH = PROJECT_ROOT / "hand_landmarker.task"
CAPTURE_DIR = PROJECT_ROOT / "outputs" / "final_camera"
LEGACY_CAPTURE_DIR = PROJECT_ROOT / "outputs" / "legacy_captures"
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
TRAINING_DIR = PROJECT_ROOT / "outputs" / "training"
BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "benchmarks"
WEB_DIR = PROJECT_ROOT / "web"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"

# ── Labels & model ──────────────────────────────────────────────────────────
LABELS = [0, 1, 2, 3, 4, 5]
VALID_LABEL_STRINGS = tuple(str(label) for label in LABELS)
CLASS_NAMES = {0: "Fist", 1: "Index", 2: "Peace", 3: "Three", 4: "Four", 5: "Open Hand"}

# ── Prediction stabilisation ────────────────────────────────────────────────
ROLLING_WINDOW_SIZE = 8
MIN_STABLE_VOTES = 4
MIN_CONFIDENCE = 0.55

# ── Camera timing ───────────────────────────────────────────────────────────
CAPTURE_COOLDOWN_SECONDS = 1.0
# A gesture must stay stable this long before it locks in a countdown, so the
# in-between finger counts seen while raising a hand don't arm the wrong one.
ARM_HOLD_SECONDS = 0.6
FLASH_DURATION_SECONDS = 0.18
FRAME_TIME_MS = 33

# ── Appearance ──────────────────────────────────────────────────────────────
WINDOW_NAME = "Smart Camera"

# ── Shared colours (BGR) ────────────────────────────────────────────────────
COLOR_WHITE = (255, 255, 255)
COLOR_LIGHT_GRAY = (230, 230, 230)
COLOR_DARK_BG = (18, 18, 18)
COLOR_GREEN = (0, 255, 120)
COLOR_RED = (60, 60, 255)
COLOR_ACCENT = (255, 180, 50)
COLOR_CYAN = (255, 212, 0)
COLOR_YELLOW = (0, 210, 255)
COLOR_MUTED = (150, 150, 150)
COLOR_PANEL_BORDER = (70, 70, 70)


# ── Shared dataclasses ──────────────────────────────────────────────────────
@dataclass
class StablePrediction:
    """Result of the rolling-window vote stabiliser."""
    label: Optional[int]
    confidence: float
    votes: int


@dataclass
class CountdownState:
    """Active gesture-triggered countdown timer."""
    label: int
    duration_seconds: int
    started_at: float
    confidence: float


@dataclass
class CaptureRequest:
    """Represents a pending photo capture."""
    trigger: str
    gesture_label: Optional[int]
    countdown_seconds: Optional[int]
    confidence: Optional[float]
