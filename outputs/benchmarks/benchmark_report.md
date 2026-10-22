# Benchmark Report

Generated 2026-10-22T16:51:29+00:00 on `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel` (6 logical CPUs), Python 3.13.7, TensorFlow 2.21.0, MediaPipe 0.10.33.

## Classifier latency (single sample, CPU)

| Backend | Mean | Median | P95 | P99 |
|---|---|---|---|---|
| Keras `model.predict()` | 75.835 ms | 71.420 ms | 95.532 ms | 162.442 ms |
| Keras `model(x)` | 5.657 ms | 4.947 ms | 8.931 ms | 16.036 ms |
| **NumPy engine** | 0.029 ms | 0.028 ms | 0.032 ms | 0.049 ms |

**Speed-up vs `model.predict()`: 2,618x.** Batch throughput (3550 samples): NumPy 909,203/s vs Keras 35,744/s.

## End-to-end pipeline (640x480 webcam frames)

| Stage | Mean | P95 |
|---|---|---|
| MediaPipe hand detection | 24.88 ms | 25.92 ms |
| Feature extraction | 0.023 ms | 0.025 ms |
| MLP classification | 0.086 ms | 0.090 ms |
| **Total** | **24.99 ms** | |

Compute ceiling: **40 FPS** (the webcam's own frame rate, typically 30 FPS, is the practical limit).

## Lighting robustness

56 held-out webcam frames (labels 1-5) from the test split; conditions are simulated with photometric transforms.

| Condition | Hand detected | Accuracy when detected | End-to-end accuracy |
|---|---|---|---|
| Original | 98.2% | 78.2% | 76.8% |
| Dim (40%) | 96.4% | 66.7% | 64.3% |
| Very dim (15%) | 14.3% | 50.0% | 7.1% |
| Overexposed | 85.7% | 72.9% | 62.5% |
| Low contrast | 76.8% | 72.1% | 55.4% |
| Sensor noise | 62.5% | 71.4% | 44.6% |
| Motion blur | 89.3% | 72.0% | 64.3% |

## Temporal stability under landmark jitter

Each test sample is replayed as an 8-frame window with i.i.d. Gaussian jitter added to every landmark's x/y position (std. dev. in pixels at 640 px width).

| Jitter (px @ 640) | Per-frame error | Rolling-vote false trigger | Abstain |
|---|---|---|---|
| 0.0 | 2.28% | 1.97% | 0.8% |
| 2.0 | 2.44% | 2.14% | 0.7% |
| 4.0 | 2.55% | 2.25% | 0.6% |
| 8.0 | 3.14% | 2.42% | 0.4% |
| 12.0 | 3.94% | 2.73% | 0.5% |
| 16.0 | 5.34% | 3.21% | 0.5% |
