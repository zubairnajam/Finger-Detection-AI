"""
Final deliverable camera app with manual shutter and gesture-driven countdown.

Usage:
    python src/final_camera_app.py [--camera 0] [--width 1280 --height 720]
                                   [--windowed] [--backend numpy|keras]

Controls:
    SPACE / s  Take an instant photo like a normal camera
    x          Cancel the active countdown
    r          Reset gesture history and countdown state
    h          Show / hide the HUD
    q / ESC    Quit

Gesture mode:
    Hold up 1-5 fingers steadily to start a countdown for that many seconds.
    Hold up 0 fingers to cancel the current countdown.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# Ensure project root is on sys.path so `src.*` imports work when invoked directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CAPTURE_DIR,
    FLASH_DURATION_SECONDS,
    FRAME_TIME_MS,
    LABELS,
    PROJECT_ROOT,
    WINDOW_NAME,
    CaptureRequest,
    StablePrediction,
)
from src.gesture_controller import GestureController
from src.hand_detection import build_detector, flatten_landmarks
from src.prediction import BACKENDS, backend_name, load_model_and_scaler, predict_probabilities
from src.ui_overlay import Telemetry, draw_camera_overlay, draw_flash_effect, draw_landmarks

logger = logging.getLogger("smart_camera")

EMA_ALPHA = 0.1  # smoothing for the on-screen timing readouts
KEY_ESC = 27


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create runtime output directories."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


def setup_window(window_name: str, fullscreen: bool = True) -> None:
    """Open the camera preview window, full-screen when supported and requested."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if not fullscreen:
        return
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass


def open_camera(index: int, width: Optional[int], height: Optional[int]) -> cv2.VideoCapture:
    """Open a webcam and optionally request a capture resolution."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam #{index}. Check that it is connected and not in use "
            "by another application, or pick another device with --camera."
        )
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def save_capture(
    raw_frame_bgr,
    preview_frame_bgr,
    request: CaptureRequest,
    raw_prediction: Optional[int],
    raw_confidence: float,
    stable_prediction: StablePrediction,
) -> tuple[str, str]:
    """Save both the clean photo and the annotated preview with metadata."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{time.time_ns()}_{request.trigger}"
    image_path = CAPTURE_DIR / f"{stem}.jpg"
    preview_path = CAPTURE_DIR / f"{stem}_preview.jpg"
    metadata_path = CAPTURE_DIR / f"{stem}.json"

    if not cv2.imwrite(str(image_path), raw_frame_bgr):
        raise OSError(f"Failed to write {image_path}")
    cv2.imwrite(str(preview_path), preview_frame_bgr)

    metadata = {
        "trigger": request.trigger,
        "gesture_label": request.gesture_label,
        "countdown_seconds": request.countdown_seconds,
        "request_confidence": request.confidence,
        "raw_prediction": raw_prediction,
        "raw_confidence": raw_confidence,
        "stable_prediction": stable_prediction.label,
        "stable_confidence": stable_prediction.confidence,
        "stable_votes": stable_prediction.votes,
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "preview_path": str(preview_path.relative_to(PROJECT_ROOT)),
        "saved_at": timestamp,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return image_path.name, preview_path.name


def _ema(previous: float, sample: float) -> float:
    return sample if previous == 0.0 else (1 - EMA_ALPHA) * previous + EMA_ALPHA * sample


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture-controlled smart camera.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=None, help="Requested capture width")
    parser.add_argument("--height", type=int, default=None, help="Requested capture height")
    parser.add_argument("--display-width", type=int, default=1280,
                        help="Upscale small frames to this width for a crisp HUD (default: 1280)")
    parser.add_argument("--windowed", action="store_true", help="Do not start in full-screen")
    parser.add_argument("--backend", choices=BACKENDS, default="numpy",
                        help="Classifier inference backend (default: numpy)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


# ── Main loop ────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    """Run the final deliverable camera app."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    ensure_dirs()
    logger.info("Loading classifier (%s backend)...", args.backend)
    model, mean, std = load_model_and_scaler(backend=args.backend)
    engine = backend_name(model)
    detector = build_detector()
    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError:
        detector.close()
        raise

    setup_window(WINDOW_NAME, fullscreen=not args.windowed)
    controller = GestureController()
    telemetry = Telemetry(backend=engine)
    flash_until = 0.0
    captures_saved = 0
    frame_count = 0
    show_hud = True
    window_seen_visible = False
    fps_timer_start = time.time()
    fps_frame_counter = 0
    session_start = time.time()
    detect_total_ms = 0.0
    classify_total_ms = 0.0
    classified_frames = 0

    logger.info("Camera running. Press q or ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Could not read a frame from the webcam; stopping.")
                break

            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()
            frame_count += 1
            fps_frame_counter += 1

            # ── Hand detection ──────────────────────────────────────────
            t0 = time.perf_counter()
            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect_for_video(mp_image, frame_count * FRAME_TIME_MS)
            detect_ms = (time.perf_counter() - t0) * 1000.0
            telemetry.detect_ms = _ema(telemetry.detect_ms, detect_ms)
            detect_total_ms += detect_ms

            hand_detected = bool(detection_result.hand_landmarks)
            raw_prediction = None
            raw_confidence = 0.0
            probabilities: Optional[np.ndarray] = None
            features = flatten_landmarks(detection_result)

            # ── Classification ──────────────────────────────────────────
            if features is not None:
                t0 = time.perf_counter()
                probabilities = predict_probabilities(features, model, mean, std)
                classify_ms = (time.perf_counter() - t0) * 1000.0
                telemetry.classify_ms = _ema(telemetry.classify_ms, classify_ms)
                classify_total_ms += classify_ms
                classified_frames += 1
                predicted_index = int(np.argmax(probabilities))
                raw_prediction = LABELS[predicted_index]
                raw_confidence = float(probabilities[predicted_index])

            prediction = (raw_prediction, raw_confidence) if raw_prediction is not None else None
            update = controller.update(prediction, now=time.time())

            # ── Drawing ─────────────────────────────────────────────────
            if frame.shape[1] < args.display_width:
                scale = args.display_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            draw_landmarks(frame, detection_result)

            elapsed_for_fps = time.time() - fps_timer_start
            if elapsed_for_fps >= 1.0:
                telemetry.fps = fps_frame_counter / elapsed_for_fps
                fps_timer_start = time.time()
                fps_frame_counter = 0

            draw_camera_overlay(
                frame,
                raw_prediction=raw_prediction,
                raw_confidence=raw_confidence,
                stable_prediction=update.stable_prediction,
                active_countdown=controller.active_countdown,
                countdown_remaining=update.countdown_remaining,
                captures_saved=captures_saved,
                fps=telemetry.fps,
                status_text=controller.status_text,
                hand_detected=hand_detected,
                probabilities=probabilities,
                telemetry=telemetry,
                show_hud=show_hud,
            )
            draw_flash_effect(frame, flash_until)

            def take_photo(request: CaptureRequest) -> None:
                nonlocal captures_saved, flash_until
                try:
                    filename, _ = save_capture(
                        raw_frame, frame, request,
                        raw_prediction, raw_confidence, update.stable_prediction,
                    )
                except OSError as exc:
                    logger.error("Could not save photo: %s", exc)
                    controller.status_text = "Could not save photo - check disk space."
                    return
                captures_saved += 1
                controller.mark_captured(time.time(), filename)
                flash_until = controller.last_capture_time + FLASH_DURATION_SECONDS
                logger.info("Saved %s (%s)", filename, request.trigger)

            # ── Capture ─────────────────────────────────────────────────
            if update.capture_request is not None:
                take_photo(update.capture_request)

            cv2.imshow(WINDOW_NAME, frame)

            # ── Keyboard ────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), KEY_ESC):
                break
            if key in (ord(" "), ord("s")):
                take_photo(controller.manual_capture_request())
            elif key == ord("x"):
                controller.cancel()
            elif key == ord("r"):
                controller.reset()
            elif key == ord("h"):
                show_hud = not show_hud

            # Stop cleanly when the user closes the window with the mouse. Some
            # OpenCV backends always report -1, so only trust the property once
            # it has been seen reporting a visible window.
            visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            if visible >= 1:
                window_seen_visible = True
            elif window_seen_visible:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    duration = max(time.time() - session_start, 1e-6)
    logger.info(
        "Session summary: %d frames in %.1fs (%.1f FPS avg) | hand detect %.1f ms avg | "
        "classifier %.3f ms avg (%s) | %d photo(s) saved to %s",
        frame_count, duration, frame_count / duration,
        detect_total_ms / max(frame_count, 1),
        classify_total_ms / max(classified_frames, 1), engine,
        captures_saved, CAPTURE_DIR.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        logging.getLogger("smart_camera").error("%s", exc)
        sys.exit(1)
