# Finger Detection Smart Camera

A gesture-controlled camera that detects finger counts (0–5) in real time and
triggers photo captures with a countdown. It runs as a desktop app (Python +
OpenCV) and as a fully in-browser web demo with no server-side inference.

**Highlights**

- **97.7% test accuracy** on 3,550 held-out samples (macro F1 97.8%)
- **25 ms end-to-end per frame** (40 FPS compute ceiling) on a 6-core laptop CPU
- **2,618× faster classifier** by serving the trained Keras MLP through a
  dependency-free NumPy engine (0.03 ms vs 76 ms for `model.predict()`)
- **~76 KB browser model** run by ~40 lines of plain JavaScript (no TF.js runtime),
  verified numerically against Python to within 1e-5
- **47 automated tests** covering inference parity, the gesture state machine,
  stabilisation and the web engine
- **Reproducible benchmark suite** for latency, lighting robustness and temporal stability

---

## Quick Start

```bash
pip install -r requirements.txt
python tools/download_model.py        # MediaPipe hand model (one-time)
python src/final_camera_app.py        # launch the camera app
```

**Gestures:** hold up 1–5 fingers to start an N-second countdown · switch count to
restart it · make a fist to cancel.

**Keys:** `SPACE`/`s` instant photo · `x` cancel countdown · `r` reset ·
`h` toggle HUD · `q`/`Esc` quit

**Options:**

```bash
python src/final_camera_app.py --camera 1 --windowed --backend numpy -v
```

| Flag | Purpose |
|------|---------|
| `--camera N` | Webcam device index |
| `--width`, `--height` | Requested capture resolution |
| `--display-width` | Window width (default 1280) |
| `--windowed` | Don't start full-screen |
| `--backend {numpy,keras}` | Classifier backend (default `numpy`) |
| `-v`, `--verbose` | Debug logging |

### Web demo

```bash
python -m http.server 8765 --directory web    # open http://localhost:8765
```

The site loads its metrics from `web/metrics.json`, so the numbers shown always
match the latest evaluation. It is deployable as-is to Vercel (`vercel.json`).

---

## Architecture

```
Webcam → MediaPipe Hand Landmarker → 63 features → MLP classifier
       → rolling-window vote → gesture state machine → countdown → capture
```

| Layer | Technology | Details |
|-------|-----------|---------|
| **Hand detection** | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (float16) | 21 3D landmarks per hand |
| **Features** | NumPy | 21 landmarks × (x, y, z) = **63 features**, standardised |
| **Classifier** | Keras MLP, served by NumPy | Dense(128) → Dense(64) → Dense(32) → Dense(6, softmax), 18.7k params |
| **Stabilisation** | Rolling-window majority vote | 8-frame window, ≥ 4 votes, ≥ 55% mean confidence |
| **Control logic** | `GestureController` state machine | I/O-free and unit-tested; prevents accidental burst shots |

See [`docs/pipeline_diagram.md`](docs/pipeline_diagram.md) for the full diagram.

**Why a NumPy engine?** `model.predict()` has ~60–70 ms of fixed per-call overhead
(graph dispatch, data-adapter setup), which dominated the real-time loop. The
network is just a few dense layers, so the weights are extracted once and the
forward pass runs directly in NumPy. Outputs match Keras to ~1e-6 (tested).
The same weights are exported to a flat binary for the browser.

---

## Evaluation

Run `python tools/evaluate_model.py`. Results are written to `outputs/evaluation/`.

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 · Fist | 0.988 | 0.996 | 0.992 | 475 |
| 1 · Index | 0.984 | 0.980 | 0.982 | 611 |
| 2 · Peace | 0.956 | 0.964 | 0.960 | 608 |
| 3 · Three | 0.967 | 0.953 | 0.960 | 617 |
| 4 · Four | 0.976 | 0.977 | 0.977 | 620 |
| 5 · Open hand | 0.995 | 0.997 | 0.996 | 619 |
| **Overall** | **0.978** | **0.978** | **0.978** | **3,550** |

**By data source:** 98.0% on the public dataset (3,444 samples) and 89.6% on
custom webcam captures (106 samples). The gap shows domain shift, and
collecting more webcam data is the clearest next improvement.

Most errors are between adjacent counts (2↔3, 3↔4). The confusion matrix is in
`outputs/evaluation/confusion_matrix.png`.

---

## Benchmarks

Run `python tools/benchmark.py` (`--quick` for a faster pass). The report and
charts go to `outputs/benchmarks/`. Measured on an Intel 8th/9th-gen CPU (6 logical cores),
Python 3.13, TensorFlow 2.21, MediaPipe 0.10.33.

### Classifier latency (single sample)

| Backend | Mean | P95 |
|---|---|---|
| Keras `model.predict()` | 75.8 ms | 95.5 ms |
| Keras `model(x)` | 5.66 ms | 8.93 ms |
| **NumPy engine** | **0.029 ms** | **0.032 ms** |

Batch throughput: about 909k samples/s (NumPy) vs 36k/s (Keras).

### End-to-end pipeline (640×480)

| Stage | Mean |
|---|---|
| MediaPipe hand detection | 24.88 ms |
| Feature extraction | 0.02 ms |
| MLP classification | 0.09 ms |
| **Total** | **24.99 ms (≈ 40 FPS)** |

Hand detection now accounts for 99.5% of frame time, so the classifier is no
longer a bottleneck.

### Lighting robustness

56 held-out webcam frames, with lighting conditions simulated by photometric transforms.

| Condition | Hand detected | End-to-end accuracy |
|---|---|---|
| Original | 98.2% | 76.8% |
| Dim (40%) | 96.4% | 64.3% |
| Very dim (15%) | 14.3% | 7.1% |
| Overexposed | 85.7% | 62.5% |
| Low contrast | 76.8% | 55.4% |
| Sensor noise | 62.5% | 44.6% |
| Motion blur | 89.3% | 64.3% |

Detection holds up down to about 40% brightness and breaks down in very dim
light. Accuracy on single raw webcam frames is lower than on the test set. In
the live app, the rolling-window vote smooths most of these per-frame errors.

### Temporal stability

With 16 px of landmark jitter, the rolling vote cuts wrong-gesture triggers from
5.34% (per frame) to 3.21%, a **40% reduction**, while abstaining on only 0.5% of windows.

---

## Testing

```bash
pip install pytest
python -m pytest
```

| Suite | Covers |
|-------|--------|
| `test_inference.py` | NumPy engine vs Keras numerical parity |
| `test_gesture_controller.py` | Countdown start/restart/cancel, re-arm rules |
| `test_stabilizer.py` | Rolling-window voting thresholds |
| `test_ui_and_features.py` | Feature extraction, overlay rendering |
| `test_web_engine.py` | JavaScript engine vs Python parity (264 vectors, needs Node) |

---

## Project Structure

```
src/                         Runtime application
  final_camera_app.py        Main camera app (CLI flags, HUD, session summary)
  gesture_controller.py      Gesture → countdown → capture state machine
  inference.py               NumPy inference engine
  prediction.py              Model loading + rolling-window stabiliser
  hand_detection.py          MediaPipe detector + landmark flattener
  ui_overlay.py              On-screen UI drawing
  config.py                  Paths, constants, dataclasses

tools/                       Data pipeline & utilities
  data_collection.py         Webcam data collection
  preprocess_landmarks.py    Feature extraction from images
  train_model.py             Model training
  evaluate_model.py          Evaluation (per-class, per-source, latency)
  benchmark.py               Latency / lighting / stability benchmarks
  export_web_model.py        Export weights for the browser (no tensorflowjs)
  sync_web_metrics.py        Write web/metrics.json from latest reports
  dataset_health_check.py    Dataset diagnostics
  download_model.py          MediaPipe model downloader

web/                         Browser demo
  index.html, style.css      Site
  app.js                     Camera, overlay, capture, gallery, charts
  mlp.js                     Dependency-free JS inference engine
  model/                     Exported weights (mlp.json + mlp.bin)

tests/                       pytest suite
docs/                        Pipeline diagram
outputs/                     Training, evaluation and benchmark reports
```

## Full pipeline

```bash
python run_pipeline.py    # collect → preprocess → train → evaluate → sync metrics → export web model
```

Steps can also be run individually with the scripts in `tools/`.
