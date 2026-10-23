# Finger Detection Smart Camera

A gesture-controlled camera that detects finger counts (0–5) in real time and
triggers photo captures with a countdown. It runs as a desktop app (Python +
OpenCV) and as a fully in-browser web demo with no server-side inference.

**Highlights**

- **96.7% test accuracy** on 3,550 held-out samples, and **96.2% on webcam images**
  (up from 89.6%) after switching to position-invariant features
- **Same accuracy wherever the hand is in the frame**: the previous model fell to
  62–83% when the hand moved off-centre or closer, often reading 4 as 5
- **~30 ms end-to-end per frame** on a 6-core laptop CPU
- **2,365× faster classifier** by serving the trained Keras MLP through a
  dependency-free NumPy engine (0.03 ms vs 68 ms for `model.predict()`)
- **~88 KB browser model** run by plain JavaScript (no TF.js runtime), with the
  feature builder and network checked against Python in tests
- **58 automated tests** covering features, inference parity, the gesture state
  machine, stabilisation and the web engine
- **Reproducible benchmark suite** for latency, lighting, jitter and hand framing

---

## Quick Start

```bash
pip install -r requirements.txt
python tools/download_model.py        # MediaPipe hand model (one-time)
python src/final_camera_app.py        # launch the camera app
```

**Gestures:** hold 1–5 fingers steady for 0.6 s to lock in an N-second countdown.
Once locked, other finger counts are ignored until the photo is taken. Make a fist to cancel.

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
Webcam → MediaPipe Hand Landmarker → 85 features → MLP classifier
       → rolling-window vote → gesture state machine → countdown → capture
```

| Layer | Technology | Details |
|-------|-----------|---------|
| **Hand detection** | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (float16) | 21 3D landmarks per hand |
| **Features** | NumPy (`src/features.py`) | **85 features**: landmarks relative to the wrist and scaled to hand size, 15 joint-bend angles, fingertip and thumb distances |
| **Classifier** | Keras MLP, served by NumPy | Dense(128) → Dense(64) → Dense(32) → Dense(6, softmax), 21.5k params |
| **Stabilisation** | Rolling-window majority vote | 8-frame window, ≥ 4 votes, ≥ 55% mean confidence |
| **Control logic** | `GestureController` state machine | I/O-free and unit-tested; prevents accidental burst shots |

See [`docs/pipeline_diagram.md`](docs/pipeline_diagram.md) for the full diagram.

**Why normalised features?** MediaPipe returns landmark positions in image
coordinates, so a raw-coordinate model learns *where* the hand is as well as
*what shape* it makes. The training images are tightly centred, which meant a
hand held off-centre or close to the webcam looked unfamiliar, and thumb-tucked
4s were read as 5s. Features are now relative to the wrist, scaled by palm size
and corrected for the frame's aspect ratio, plus finger-bend angles that
describe the shape directly. Training adds mirrored copies (left and right
hands) and copies with small landmark jitter (tracking wobble).

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
| 0 · Fist | 0.971 | 0.987 | 0.979 | 475 |
| 1 · Index | 0.971 | 0.977 | 0.974 | 611 |
| 2 · Peace | 0.950 | 0.934 | 0.942 | 608 |
| 3 · Three | 0.927 | 0.951 | 0.939 | 617 |
| 4 · Four | 0.990 | 0.958 | 0.974 | 620 |
| 5 · Open hand | 0.994 | 0.997 | 0.995 | 619 |
| **Overall** | **0.967** | **0.968** | **0.967** | **3,550** |

**By data source:** 96.7% on the public dataset (3,444 samples) and 96.2% on
custom webcam captures (106 samples), up from 89.6% with raw-coordinate features.
The old model scored 97.7% on the centred public images partly by using the
hand's position, which stopped working as soon as the hand moved (see
*Framing robustness* below).

Most errors are between adjacent counts (2↔3, 3↔4). The confusion matrix is in
`outputs/evaluation/confusion_matrix.png`.

---

## Benchmarks

Run `python tools/benchmark.py` (`--quick` for a faster pass). The report and
charts go to `outputs/benchmarks/`. Measured on an Intel 8th/9th-gen CPU (6 logical cores),
Python 3.13, TensorFlow 2.21, MediaPipe 0.10.33. Timings vary by a few ms between runs.

### Classifier latency (single sample)

| Backend | Mean | P95 |
|---|---|---|
| Keras `model.predict()` | 68.3 ms | 100.1 ms |
| Keras `model(x)` | 4.48 ms | 5.45 ms |
| **NumPy engine** | **0.029 ms** | **0.031 ms** |

### End-to-end pipeline (640×480)

| Stage | Mean |
|---|---|
| MediaPipe hand detection | 30.0 ms |
| Feature extraction | 0.59 ms |
| MLP classification | 0.07 ms |
| **Total** | **30.7 ms (≈ 33 FPS)** |

Hand detection accounts for 98% of frame time, so the classifier is not a bottleneck.

### Lighting robustness

56 held-out webcam frames, with lighting conditions simulated by photometric transforms.

| Condition | Hand detected | End-to-end accuracy |
|---|---|---|
| Original | 98.2% | 85.7% (was 76.8%) |
| Dim (40%) | 96.4% | 73.2% (was 64.3%) |
| Very dim (15%) | 14.3% | 8.9% (was 7.1%) |
| Overexposed | 85.7% | 71.4% (was 62.5%) |
| Low contrast | 76.8% | 57.1% (was 55.4%) |
| Sensor noise | 62.5% | 44.6% (unchanged) |
| Motion blur | 89.3% | 73.2% (was 64.3%) |

Detection holds up down to about 40% brightness and breaks down in very dim
light. Accuracy on single raw webcam frames is lower than on the test set. In
the live app, the rolling-window vote smooths most of these per-frame errors.

### Temporal stability

| Jitter (px @ 640) | Per-frame error | With rolling vote |
|---|---|---|
| 0 | 3.3% | 2.5% |
| 4 | 4.5% | 3.5% |
| 8 | 9.3% | 5.1% |
| 16 | 31.9% | 15.2% |

The finger-angle features are more sensitive to heavy jitter than raw
coordinates were. Training on jittered copies largely offsets this in the
realistic 0–4 px range, and the 8-frame vote roughly halves wrong triggers at
higher noise.

### Framing robustness

Test hands moved and resized within the frame, as happens with a real webcam:

| Framing | Previous model | Current model | 4 read as 5 (prev → now) |
|---|---|---|---|
| As captured | 97.7% | 96.7% | 1 → 2 |
| Off-centre | 76.6% | 96.7% | 19 → 2 |
| Further away | 62.5% | 96.7% | 1 → 2 |
| Closer | 82.7% | 96.7% | 253 → 2 |
| Mixed | 69.4% | 96.7% | 30 → 2 |

---

## Testing

```bash
pip install pytest
python -m pytest
```

| Suite | Covers |
|-------|--------|
| `test_features.py` | Feature builder: position/size invariance, mirroring, aspect ratio |
| `test_inference.py` | NumPy engine vs Keras numerical parity |
| `test_gesture_controller.py` | Arm-hold, countdown lock, cancel, re-arm rules |
| `test_stabilizer.py` | Rolling-window voting thresholds |
| `test_ui_and_features.py` | Feature extraction, overlay rendering |
| `test_web_engine.py` | JavaScript features and network vs Python (needs Node) |

---

## Project Structure

```
src/                         Runtime application
  final_camera_app.py        Main camera app (CLI flags, HUD, session summary)
  gesture_controller.py      Gesture → countdown → capture state machine
  features.py                Position-invariant feature builder
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
  visuals.js                 Interactive background, cursor and hero animation
  mlp.js                     JS feature builder + inference engine
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
