"""
Shared UI drawing helpers for OpenCV camera overlays.

Provides reusable functions for panels, text, landmarks, flash effects,
and the full enhanced camera overlay used by the final camera app.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from src.config import (
    COLOR_ACCENT,
    COLOR_DARK_BG,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_RED,
    COLOR_WHITE,
    FLASH_DURATION_SECONDS,
    ROLLING_WINDOW_SIZE,
    CountdownState,
    StablePrediction,
)


# ── Low-level primitives ────────────────────────────────────────────────────

def draw_panel(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    *,
    alpha: float = 0.45,
    fill_color: tuple[int, int, int] = COLOR_DARK_BG,
    border_color: tuple[int, int, int] | None = None,
    border_thickness: int = 1,
    corner_radius: int = 0,
) -> None:
    """Draw a translucent rounded-rectangle panel."""
    overlay = frame.copy()
    if corner_radius > 0:
        # Use a mask for rounded corners
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(
            mask,
            (top_left[0] + corner_radius, top_left[1]),
            (bottom_right[0] - corner_radius, bottom_right[1]),
            255, -1,
        )
        cv2.rectangle(
            mask,
            (top_left[0], top_left[1] + corner_radius),
            (bottom_right[0], bottom_right[1] - corner_radius),
            255, -1,
        )
        for cx, cy in [
            (top_left[0] + corner_radius, top_left[1] + corner_radius),
            (bottom_right[0] - corner_radius, top_left[1] + corner_radius),
            (top_left[0] + corner_radius, bottom_right[1] - corner_radius),
            (bottom_right[0] - corner_radius, bottom_right[1] - corner_radius),
        ]:
            cv2.circle(mask, (cx, cy), corner_radius, 255, -1)
        overlay[mask > 0] = fill_color
    else:
        cv2.rectangle(overlay, top_left, bottom_right, fill_color, -1)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)

    if border_color is not None:
        cv2.rectangle(frame, top_left, bottom_right, border_color, border_thickness)


def draw_text(
    frame,
    text: str,
    position: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = COLOR_WHITE,
    thickness: int = 1,
) -> None:
    """Draw anti-aliased text at the given position."""
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
    )


def draw_landmarks(frame_bgr, detection_result) -> None:
    """Draw hand landmarks in-place on the preview frame."""
    if not detection_result.hand_landmarks:
        return

    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    for hand_landmarks in detection_result.hand_landmarks:
        vision.drawing_utils.draw_landmarks(
            rgb_frame,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )
    frame_bgr[:] = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)


def draw_flash_effect(frame_bgr, flash_until: float) -> None:
    """Overlay a short white flash after taking a photo."""
    now = time.time()
    if now >= flash_until:
        return

    remaining_ratio = (flash_until - now) / FLASH_DURATION_SECONDS
    alpha = min(max(0.2 + 0.55 * remaining_ratio, 0.0), 0.75)
    flash_overlay = np.full_like(frame_bgr, 255)
    cv2.addWeighted(flash_overlay, alpha, frame_bgr, 1.0 - alpha, 0.0, frame_bgr)


# ── Countdown arc ───────────────────────────────────────────────────────────

def _draw_countdown_arc(
    frame,
    countdown_remaining: float,
    total_seconds: int,
) -> None:
    """Draw a circular progress arc and large countdown number in the center."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 6

    # Background circle
    cv2.circle(frame, (cx, cy), radius + 4, COLOR_DARK_BG, -1)
    cv2.circle(frame, (cx, cy), radius + 4, (60, 60, 60), 2)

    # Progress arc (0° = top, sweeps clockwise)
    fraction = max(0.0, countdown_remaining / total_seconds)
    sweep_angle = int(360 * (1.0 - fraction))

    # Colour lerp: green → yellow → red
    if fraction > 0.5:
        arc_color = COLOR_GREEN
    elif fraction > 0.2:
        arc_color = (0, 230, 230)  # yellow-ish
    else:
        arc_color = COLOR_RED

    cv2.ellipse(
        frame, (cx, cy), (radius, radius),
        -90,  # start at top
        0, sweep_angle,
        arc_color, 6, cv2.LINE_AA,
    )

    # Large countdown number
    display_number = str(max(0, math.ceil(countdown_remaining)))
    text_size, _ = cv2.getTextSize(display_number, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)
    text_x = cx - text_size[0] // 2
    text_y = cy + text_size[1] // 2
    draw_text(frame, display_number, (text_x, text_y), scale=4.0, color=arc_color, thickness=6)


# ── Detection indicator ────────────────────────────────────────────────────

def _draw_hand_indicator(frame, hand_detected: bool) -> None:
    """Small coloured circle in the top-right corner: green = hand, red = none."""
    h, w = frame.shape[:2]
    color = COLOR_GREEN if hand_detected else COLOR_RED
    cx = w - 36
    cy = 36
    cv2.circle(frame, (cx, cy), 10, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 10, COLOR_WHITE, 1, cv2.LINE_AA)
    label = "HAND" if hand_detected else "NO HAND"
    draw_text(frame, label, (cx - 62, cy + 4), scale=0.38, color=color)


# ── Full overlay ────────────────────────────────────────────────────────────

def draw_camera_overlay(
    frame,
    *,
    raw_prediction: Optional[int],
    raw_confidence: float,
    stable_prediction: StablePrediction,
    active_countdown: CountdownState | None,
    countdown_remaining: Optional[float],
    captures_saved: int,
    fps: float,
    status_text: str,
    hand_detected: bool,
) -> None:
    """Render the full enhanced camera-style overlay."""
    h, w = frame.shape[:2]

    # ── Top-left: Title + status ────────────────────────────────────────
    tl_w, tl_h = 560, 92
    draw_panel(frame, (16, 12), (16 + tl_w, 12 + tl_h), alpha=0.38,
               border_color=(80, 80, 80), corner_radius=8)
    draw_text(frame, "SMART CAMERA", (32, 42), scale=0.78, color=COLOR_WHITE, thickness=2)
    # Truncate long status text
    max_chars = 68
    displayed_status = status_text[:max_chars] + "..." if len(status_text) > max_chars else status_text
    draw_text(frame, displayed_status, (32, 72), scale=0.48, color=COLOR_LIGHT_GRAY)

    # ── Top-right: Gesture info panel ───────────────────────────────────
    panel_w = 260
    tr_x = w - panel_w - 16
    tr_y = 12
    draw_panel(frame, (tr_x, tr_y), (tr_x + panel_w, tr_y + 112), alpha=0.38,
               border_color=(80, 80, 80), corner_radius=8)

    stable_text = str(stable_prediction.label) if stable_prediction.label is not None else "-"
    raw_text = str(raw_prediction) if raw_prediction is not None else "-"

    info_lines = [
        f"Gesture: {stable_text}  ({stable_prediction.confidence:.0%})",
        f"Raw: {raw_text}  ({raw_confidence:.0%})",
        f"Votes: {stable_prediction.votes}/{ROLLING_WINDOW_SIZE}",
        f"Shots: {captures_saved}   FPS: {fps:.0f}",
    ]
    for idx, line in enumerate(info_lines):
        y = tr_y + 28 + idx * 22
        draw_text(frame, line, (tr_x + 14, y), scale=0.46, color=COLOR_WHITE)

    # ── Hand detection indicator ────────────────────────────────────────
    _draw_hand_indicator(frame, hand_detected)

    # ── Bottom: Keyboard shortcuts ──────────────────────────────────────
    bar_text = "SPACE shoot  |  x cancel  |  r reset  |  q quit"
    bar_w = 440
    bar_h = 36
    bx = (w - bar_w) // 2
    by = h - 52
    draw_panel(frame, (bx, by), (bx + bar_w, by + bar_h), alpha=0.38,
               border_color=(80, 80, 80), corner_radius=6)
    draw_text(frame, bar_text, (bx + 16, by + 24), scale=0.44, color=COLOR_LIGHT_GRAY)

    # ── Center: Countdown arc ───────────────────────────────────────────
    if active_countdown is not None and countdown_remaining is not None:
        _draw_countdown_arc(frame, countdown_remaining, active_countdown.duration_seconds)

        hint = f"{active_countdown.duration_seconds}s countdown"
        hint_size, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        hint_x = (w - hint_size[0]) // 2
        hint_y = h // 2 + min(w, h) // 6 + 40
        draw_text(frame, hint, (hint_x, hint_y), scale=0.7, color=COLOR_ACCENT, thickness=2)
