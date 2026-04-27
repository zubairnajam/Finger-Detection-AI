# Finger Detection Smart Camera

A gesture-controlled camera that uses AI to detect finger counts (0–5) in
real-time and trigger photo captures with a countdown timer.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download the MediaPipe hand model (one-time)
python tools/download_model.py

# Launch the camera app
python src/final_camera_app.py
```

**Controls:** `SPACE` instant photo · `1-5 fingers` countdown capture ·
`x` cancel countdown · `r` reset · `q` quit

---

## Gesture Recognition Approach

| Layer | Technology | Details |
|-------|-----------|---------|
| **Hand Detection** | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (float16) | Detects 21 3D landmarks per hand |
| **Feature Extraction** | NumPy | Flattens 21 landmarks × 3 coords = **63 features** |
| **Classifier** | TensorFlow / Keras MLP | Dense(128) → Dense(64) → Dense(32) → Dense(6, softmax) |
| **Stabilisation** | Rolling-window majority vote | 8-frame window, ≥ 4 votes, ≥ 55% average confidence |

The system avoids hand-crafted heuristics by learning finger-count patterns
directly from landmark coordinates via a small neural network (~260 KB).

---

## Evaluation Metrics

### Gesture Recognition Accuracy

| Label | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (fist) | 0.9895 | 0.9958 | 0.9927 | 475 |
| 1 finger | 0.9816 | 0.9832 | 0.9824 | 596 |
| 2 fingers | 0.9581 | 0.9565 | 0.9573 | 598 |
| 3 fingers | 0.9557 | 0.9589 | 0.9573 | 608 |
| 4 fingers | 0.9785 | 0.9689 | 0.9737 | 611 |
| 5 fingers | 0.9935 | 0.9951 | 0.9943 | 612 |
| **Overall** | **—** | **—** | **0.9757** | **3 500** |

### Inference Latency

Run `python tools/evaluate_model.py` to benchmark on your hardware.
The MLP classifier infers a single sample in **< 5 ms** on a modern CPU,
enabling **30+ FPS** real-time operation.

---

## Real-Time Performance Under Different Lighting

MediaPipe's hand detector normalises brightness internally, making the
system robust across a range of conditions:

| Condition | Expected Performance |
|-----------|---------------------|
| **Well-lit room** (≥ 300 lux) | Best accuracy; confident detections |
| **Moderate / indoor** (100-300 lux) | Reliable; slight confidence drop |
| **Dim lighting** (< 100 lux) | Detection rate drops; lower thresholds (0.35) help but noise increases |
| **Backlighting / silhouette** | Hand shape may be lost; reposition so hand is lit from the front |
| **Mixed / flickering light** | Rolling-window stabiliser absorbs transient mis-predictions |

**Tips:** Face a light source · avoid silhouetting your hand · use the
rolling-vote stabiliser's natural smoothing to ride out brief flickers.

---

## Pipeline Diagram

See [`docs/pipeline_diagram.md`](docs/pipeline_diagram.md) for the full
Mermaid diagram. Summary:

```
Webcam → MediaPipe Hand Landmarker → 63 Features → TF/Keras MLP
       → Raw Prediction → Rolling-Window Vote → Countdown → Capture
```

---

## Project Structure

```
src/                       Final runtime application
  final_camera_app.py      Main camera app (deliverable)
  config.py                Shared paths, constants, dataclasses
  hand_detection.py        MediaPipe detector + landmark flattener
  prediction.py            Model loading + stabilisation logic
  ui_overlay.py            All UI drawing helpers

tools/                     Data pipeline & utilities
  data_collection.py       Webcam data collection
  preprocess_landmarks.py  Feature extraction from images
  train_model.py           Model training (MLP)
  evaluate_model.py        Evaluation & latency benchmark
  dataset_health_check.py  Dataset diagnostics
  download_model.py        MediaPipe model downloader

web/                       Browser-based demo site
  index.html               Website structure
  style.css                Dark-theme styling
  app.js                   MediaPipe JS + finger counting

docs/                      Documentation
  pipeline_diagram.md      Mermaid pipeline diagram

data/                      Datasets (raw, processed, train, test)
models/                    Trained model artifacts
outputs/                   Training reports, evaluation, captures
```

## Commands

```bash
# Main app
python src/final_camera_app.py

# Web demo (open http://localhost:8765)
python -m http.server 8765 --directory web

# Evaluation & metrics
python tools/evaluate_model.py

# Training pipeline
python tools/data_collection.py
python tools/preprocess_landmarks.py
python tools/train_model.py
python tools/dataset_health_check.py
python tools/download_model.py

# Full pipeline (all stages sequentially)
python run_pipeline.py
```

## Notes

- The final app (`src/final_camera_app.py`) is the only script needed for the deliverable demo.
- The `tools/` scripts are kept for retraining, evaluation, and debugging.
- `archive/` (external dataset) is intentionally left untouched.
