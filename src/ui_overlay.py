"""
Shared UI drawing helpers for OpenCV camera overlays.

Provides reusable functions for panels, text, landmarks, flash effects,
and the full HUD used by the final camera app. Layout is defined for a
1280-px-wide canvas and scaled proportionally for other widths.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from src.config import (
    CLASS_NAMES,
    COLOR_ACCENT,
    COLOR_CYAN,
    COLOR_DARK_BG,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_MUTED,
    COLOR_PANEL_BORDER,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    FLASH_DURATION_SECONDS,
    LABELS,
    ROLLING_WINDOW_SIZE,
    CountdownState,
    StablePrediction,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
REFERENCE_WIDTH = 1280

# MediaPipe 21-point hand topology, grouped by finger for colour-coding.
FINGER_CHAINS: dict[str, tuple[int, ...]] = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
PALM_CHAIN = (5, 9, 13, 17)
FINGERTIPS = (4, 8, 12, 16, 20)
FINGER_COLORS = {  # BGR
    "thumb": (90, 120, 255),
    "index": (60, 210, 255),
    "middle": (120, 255, 120),
    "ring": (255, 210, 60),
    "pinky": (255, 120, 200),
}


@dataclass
class Telemetry:
    """Per-stage timings (milliseconds) and throughput shown in the HUD."""
    fps: float = 0.0
    detect_ms: float = 0.0
    classify_ms: float = 0.0
    backend: str = "numpy"


# ── Low-level primitives ────────────────────────────────────────────────────

def _rounded_mask(height: int, width: int, radius: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = max(0, min(radius, height // 2, width // 2))
    if radius == 0:
        mask[:] = 255
        return mask
    cv2.rectangle(mask, (radius, 0), (width - 1 - radius, height - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - 1 - radius), 255, -1)
    for cx, cy in (
        (radius, radius),
        (width - 1 - radius, radius),
        (radius, height - 1 - radius),
        (width - 1 - radius, height - 1 - radius),
    ):
        cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    return mask


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
    """Draw a translucent (optionally rounded) panel.

    Only the panel's region of interest is blended, rather than the full
    frame, which keeps the cost proportional to the panel size.
    """
    h, w = frame.shape[:2]
    x1, y1 = max(0, top_left[0]), max(0, top_left[1])
    x2, y2 = min(w, bottom_right[0]), min(h, bottom_right[1])
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    fill = np.empty_like(roi)
    fill[:] = fill_color
    blended = cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0.0)

    if corner_radius > 0:
        mask = _rounded_mask(y2 - y1, x2 - x1, corner_radius)
        roi[mask > 0] = blended[mask > 0]
    else:
        roi[:] = blended

    if border_color is not None:
        if corner_radius > 0:
            _draw_rounded_border(frame, (x1, y1), (x2 - 1, y2 - 1), corner_radius,
                                 border_color, border_thickness)
        else:
            cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), border_color, border_thickness)


def _draw_rounded_border(frame, tl, br, radius, color, thickness) -> None:
    x1, y1 = tl
    x2, y2 = br
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_text(
    frame,
    text: str,
    position: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = COLOR_WHITE,
    thickness: int = 1,
    font: int = FONT,
) -> None:
    """Draw anti-aliased text at the given position."""
    cv2.putText(frame, text, position, font, scale, color, thickness, cv2.LINE_AA)


def _text_width(text: str, scale: float, thickness: int, font: int = FONT) -> int:
    return cv2.getTextSize(text, font, scale, thickness)[0][0]


def _fit_text(text: str, max_width: int, scale: float, thickness: int = 1) -> str:
    """Truncate text with an ellipsis so it fits within max_width pixels."""
    if _text_width(text, scale, thickness) <= max_width:
        return text
    while text and _text_width(text + "...", scale, thickness) > max_width:
        text = text[:-1]
    return text + "..."


def _draw_bar(frame, x: int, y: int, width: int, height: int, fraction: float,
              color: tuple[int, int, int]) -> None:
    """Horizontal progress bar with a dim track."""
    fraction = min(max(fraction, 0.0), 1.0)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (55, 55, 55), -1)
    filled = int(round(width * fraction))
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)


def _confidence_color(confidence: float) -> tuple[int, int, int]:
    if confidence >= 0.8:
        return COLOR_GREEN
    if confidence >= 0.55:
        return COLOR_YELLOW
    return COLOR_RED


# ── Landmarks ───────────────────────────────────────────────────────────────

def draw_landmarks(frame_bgr, detection_result) -> None:
    """Draw a colour-coded hand skeleton in-place on the preview frame."""
    if not detection_result.hand_landmarks:
        return

    h, w = frame_bgr.shape[:2]
    s = max(w / REFERENCE_WIDTH, 0.5)
    bone = max(1, int(round(3 * s)))
    joint = max(2, int(round(4 * s)))
    tip = max(3, int(round(7 * s)))

    for hand_landmarks in detection_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        if len(pts) < 21:
            continue

        for a, b in zip(PALM_CHAIN, PALM_CHAIN[1:]):
            cv2.line(frame_bgr, pts[a], pts[b], (200, 200, 200), bone, cv2.LINE_AA)
        for finger, chain in FINGER_CHAINS.items():
            color = FINGER_COLORS[finger]
            for a, b in zip(chain, chain[1:]):
                cv2.line(frame_bgr, pts[a], pts[b], color, bone, cv2.LINE_AA)

        for point in pts:
            cv2.circle(frame_bgr, point, joint, COLOR_WHITE, -1, cv2.LINE_AA)
        for finger, tip_idx in zip(FINGER_CHAINS, FINGERTIPS):
            cv2.circle(frame_bgr, pts[tip_idx], tip, FINGER_COLORS[finger], -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, pts[tip_idx], tip, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_flash_effect(frame_bgr, flash_until: float) -> None:
    """Overlay a short white flash after taking a photo."""
    now = time.time()
    if now >= flash_until:
        return

    remaining_ratio = (flash_until - now) / FLASH_DURATION_SECONDS
    alpha = min(max(0.2 + 0.55 * remaining_ratio, 0.0), 0.75)
    flash_overlay = np.full_like(frame_bgr, 255)
    cv2.addWeighted(flash_overlay, alpha, frame_bgr, 1.0 - alpha, 0.0, frame_bgr)


# ── Countdown ring ──────────────────────────────────────────────────────────

def _draw_countdown_arc(
    frame,
    countdown_remaining: float,
    total_seconds: int,
) -> None:
    """Draw a circular progress ring and large countdown number in the center."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 6
    s = max(w / REFERENCE_WIDTH, 0.5)
    ring = max(4, int(round(10 * s)))

    draw_panel(frame, (cx - radius - ring * 2, cy - radius - ring * 2),
               (cx + radius + ring * 2, cy + radius + ring * 2),
               alpha=0.55, corner_radius=radius + ring * 2)
    cv2.circle(frame, (cx, cy), radius, (60, 60, 60), ring, cv2.LINE_AA)

    fraction = max(0.0, min(1.0, countdown_remaining / max(total_seconds, 1e-6)))
    if fraction > 0.5:
        arc_color = COLOR_GREEN
    elif fraction > 0.2:
        arc_color = COLOR_YELLOW
    else:
        arc_color = COLOR_RED

    # Remaining time drains clockwise from 12 o'clock
    sweep = 360.0 * fraction
    if sweep > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, sweep,
                    arc_color, ring, cv2.LINE_AA)

    display_number = str(max(0, math.ceil(countdown_remaining)))
    num_scale = 4.0 * s
    num_thick = max(2, int(round(7 * s)))
    (tw, th), _ = cv2.getTextSize(display_number, FONT_BOLD, num_scale, num_thick)
    draw_text(frame, display_number, (cx - tw // 2, cy + th // 2),
              scale=num_scale, color=arc_color, thickness=num_thick, font=FONT_BOLD)


# ── HUD sections ────────────────────────────────────────────────────────────

def _draw_header(frame, s: float, status_text: str, hand_detected: bool, backend: str) -> None:
    pad = int(16 * s)
    x1, y1 = pad, pad
    x2, y2 = x1 + int(560 * s), y1 + int(92 * s)
    draw_panel(frame, (x1, y1), (x2, y2), alpha=0.55,
               border_color=COLOR_PANEL_BORDER, corner_radius=int(12 * s))

    draw_text(frame, "SMART CAMERA", (x1 + int(18 * s), y1 + int(36 * s)),
              scale=0.8 * s, color=COLOR_WHITE, thickness=max(1, int(2 * s)), font=FONT_BOLD)
    draw_text(frame, f"{backend} engine",
              (x1 + int(236 * s), y1 + int(34 * s)), scale=0.45 * s, color=COLOR_MUTED)

    pill_color = COLOR_GREEN if hand_detected else COLOR_RED
    pill_text = "HAND" if hand_detected else "NO HAND"
    pill_w = int(104 * s)
    px2 = x2 - int(14 * s)
    px1 = px2 - pill_w
    py1, py2 = y1 + int(14 * s), y1 + int(42 * s)
    draw_panel(frame, (px1, py1), (px2, py2), alpha=0.25, fill_color=pill_color,
               corner_radius=int(14 * s))
    cv2.circle(frame, (px1 + int(16 * s), (py1 + py2) // 2), max(3, int(6 * s)),
               pill_color, -1, cv2.LINE_AA)
    draw_text(frame, pill_text, (px1 + int(30 * s), py2 - int(9 * s)),
              scale=0.45 * s, color=COLOR_WHITE, thickness=1)

    status = _fit_text(status_text, x2 - x1 - int(36 * s), 0.52 * s)
    draw_text(frame, status, (x1 + int(18 * s), y1 + int(72 * s)),
              scale=0.52 * s, color=COLOR_LIGHT_GRAY)


def _draw_gesture_panel(
    frame,
    s: float,
    *,
    raw_prediction: Optional[int],
    raw_confidence: float,
    stable_prediction: StablePrediction,
    probabilities: Optional[Sequence[float]],
) -> None:
    h, w = frame.shape[:2]
    pad = int(16 * s)
    panel_w = int(300 * s)
    x1, y1 = w - panel_w - pad, pad
    inner = x1 + int(18 * s)
    inner_w = panel_w - int(36 * s)
    has_probs = probabilities is not None and len(probabilities) == len(LABELS)
    panel_h = int((372 if has_probs else 214) * s)
    draw_panel(frame, (x1, y1), (x1 + panel_w, y1 + panel_h), alpha=0.55,
               border_color=COLOR_PANEL_BORDER, corner_radius=int(12 * s))

    draw_text(frame, "GESTURE", (inner, y1 + int(28 * s)), scale=0.45 * s, color=COLOR_MUTED)

    stable = stable_prediction.label
    if stable is not None:
        big, big_color = str(stable), COLOR_CYAN
        name = CLASS_NAMES.get(stable, "")
    elif raw_prediction is not None:
        big, big_color = str(raw_prediction), COLOR_MUTED
        name = "stabilising..."
    else:
        big, big_color, name = "-", COLOR_MUTED, "no hand"

    big_scale = 2.4 * s
    big_thick = max(2, int(4 * s))
    draw_text(frame, big, (inner, y1 + int(100 * s)), scale=big_scale, color=big_color,
              thickness=big_thick, font=FONT_BOLD)
    label_x = inner + int(70 * s)
    draw_text(frame, name, (label_x, y1 + int(66 * s)), scale=0.62 * s,
              color=COLOR_WHITE, thickness=max(1, int(2 * s)))
    conf = stable_prediction.confidence if stable is not None else raw_confidence
    draw_text(frame, f"{conf:.0%} confidence", (label_x, y1 + int(92 * s)),
              scale=0.45 * s, color=COLOR_LIGHT_GRAY)

    # Confidence bar
    bar_y = y1 + int(116 * s)
    _draw_bar(frame, inner, bar_y, inner_w, max(3, int(6 * s)), conf, _confidence_color(conf))

    # Vote meter: one segment per frame in the rolling window
    draw_text(frame, f"STABILITY  {stable_prediction.votes}/{ROLLING_WINDOW_SIZE} votes",
              (inner, y1 + int(152 * s)), scale=0.42 * s, color=COLOR_MUTED)
    seg_gap = max(2, int(4 * s))
    seg_w = (inner_w - seg_gap * (ROLLING_WINDOW_SIZE - 1)) // ROLLING_WINDOW_SIZE
    seg_y = y1 + int(164 * s)
    seg_color = COLOR_GREEN if stable is not None else COLOR_YELLOW
    for i in range(ROLLING_WINDOW_SIZE):
        sx = inner + i * (seg_w + seg_gap)
        color = seg_color if i < stable_prediction.votes else (55, 55, 55)
        cv2.rectangle(frame, (sx, seg_y), (sx + seg_w, seg_y + max(4, int(10 * s))), color, -1)

    if not has_probs:
        return

    # Class probability distribution
    draw_text(frame, "CLASS PROBABILITIES", (inner, y1 + int(208 * s)),
              scale=0.42 * s, color=COLOR_MUTED)
    top = int(np.argmax(probabilities))
    row_h = int(24 * s)
    label_w = int(96 * s)
    pct_w = int(44 * s)
    for i, label in enumerate(LABELS):
        ry = y1 + int(222 * s) + i * row_h
        p = float(probabilities[i])
        color = COLOR_CYAN if i == top else (120, 120, 120)
        draw_text(frame, f"{label} {CLASS_NAMES[label]}", (inner, ry + int(13 * s)),
                  scale=0.42 * s, color=COLOR_WHITE if i == top else COLOR_LIGHT_GRAY)
        _draw_bar(frame, inner + label_w, ry + int(4 * s), inner_w - label_w - pct_w,
                  max(3, int(10 * s)), p, color)
        draw_text(frame, f"{p:4.0%}", (inner + inner_w - pct_w + int(6 * s), ry + int(13 * s)),
                  scale=0.4 * s, color=COLOR_LIGHT_GRAY)


def _draw_telemetry(frame, s: float, telemetry: Telemetry, captures_saved: int) -> None:
    h, _ = frame.shape[:2]
    pad = int(16 * s)
    panel_w, panel_h = int(250 * s), int(118 * s)
    x1, y1 = pad, h - panel_h - pad
    draw_panel(frame, (x1, y1), (x1 + panel_w, y1 + panel_h), alpha=0.55,
               border_color=COLOR_PANEL_BORDER, corner_radius=int(12 * s))

    fps_color = COLOR_GREEN if telemetry.fps >= 24 else COLOR_YELLOW if telemetry.fps >= 15 else COLOR_RED
    rows = [
        ("FPS", f"{telemetry.fps:.0f}", fps_color),
        ("Hand detect", f"{telemetry.detect_ms:.1f} ms", COLOR_WHITE),
        ("Classifier", f"{telemetry.classify_ms:.2f} ms", COLOR_WHITE),
        ("Photos", str(captures_saved), COLOR_WHITE),
    ]
    for i, (label, value, color) in enumerate(rows):
        ry = y1 + int(28 * s) + i * int(24 * s)
        draw_text(frame, label, (x1 + int(16 * s), ry), scale=0.45 * s, color=COLOR_MUTED)
        vw = _text_width(value, 0.5 * s, 1)
        draw_text(frame, value, (x1 + panel_w - int(16 * s) - vw, ry), scale=0.5 * s, color=color)


def _draw_shortcuts(frame, s: float) -> None:
    h, w = frame.shape[:2]
    keys = [("SPACE", "shoot"), ("X", "cancel"), ("R", "reset"), ("H", "hide HUD"), ("Q", "quit")]
    scale = 0.45 * s
    gap = int(18 * s)
    key_pad = int(8 * s)
    widths = [_text_width(k, scale, 1) + key_pad * 2 + int(6 * s) + _text_width(d, scale, 1)
              for k, d in keys]
    bar_w = sum(widths) + gap * (len(keys) - 1) + int(32 * s)
    bar_h = int(40 * s)
    pad = int(16 * s)
    bx = (w - bar_w) // 2
    by = h - bar_h - pad
    draw_panel(frame, (bx, by), (bx + bar_w, by + bar_h), alpha=0.55,
               border_color=COLOR_PANEL_BORDER, corner_radius=int(10 * s))

    cx = bx + int(16 * s)
    text_y = by + bar_h // 2 + int(6 * s)
    for (key, desc), width in zip(keys, widths):
        kw = _text_width(key, scale, 1) + key_pad * 2
        cv2.rectangle(frame, (cx, by + int(9 * s)), (cx + kw, by + bar_h - int(9 * s)),
                      (90, 90, 90), 1, cv2.LINE_AA)
        draw_text(frame, key, (cx + key_pad, text_y), scale=scale, color=COLOR_WHITE)
        draw_text(frame, desc, (cx + kw + int(6 * s), text_y), scale=scale, color=COLOR_LIGHT_GRAY)
        cx += width + gap


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
    probabilities: Optional[Sequence[float]] = None,
    telemetry: Optional[Telemetry] = None,
    show_hud: bool = True,
) -> None:
    """Render the full camera HUD in-place."""
    h, w = frame.shape[:2]
    s = max(w / REFERENCE_WIDTH, 0.5)
    if telemetry is None:
        telemetry = Telemetry(fps=fps)

    if show_hud:
        _draw_header(frame, s, status_text, hand_detected, telemetry.backend)
        _draw_gesture_panel(
            frame, s,
            raw_prediction=raw_prediction,
            raw_confidence=raw_confidence,
            stable_prediction=stable_prediction,
            probabilities=probabilities,
        )
        _draw_telemetry(frame, s, telemetry, captures_saved)
        _draw_shortcuts(frame, s)

    if active_countdown is not None and countdown_remaining is not None:
        _draw_countdown_arc(frame, countdown_remaining, active_countdown.duration_seconds)

        hint = f"{active_countdown.duration_seconds}s countdown  -  make a fist to cancel"
        hint_scale = 0.7 * s
        hint_w = _text_width(hint, hint_scale, 2)
        hint_y = h // 2 + min(w, h) // 6 + int(70 * s)
        hint_x = (w - hint_w) // 2
        pad = int(14 * s)
        draw_panel(frame, (hint_x - pad, hint_y - int(30 * s)), (hint_x + hint_w + pad, hint_y + int(14 * s)),
                   alpha=0.6, corner_radius=int(10 * s))
        draw_text(frame, hint, (hint_x, hint_y), scale=hint_scale, color=COLOR_ACCENT, thickness=2)
