# Benchmark Report

Generated 2026-10-23T04:45:58+00:00 on `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel` (6 logical CPUs), Python 3.13.7, TensorFlow 2.21.0, MediaPipe 0.10.33.

## Classifier latency (single sample, CPU)

| Backend | Mean | Median | P95 | P99 |
|---|---|---|---|---|
| Keras `model.predict()` | 68.257 ms | 63.477 ms | 100.057 ms | 143.648 ms |
| Keras `model(x)` | 4.476 ms | 4.280 ms | 5.454 ms | 7.617 ms |
| **NumPy engine** | 0.029 ms | 0.028 ms | 0.031 ms | 0.036 ms |

**Speed-up vs `model.predict()`: 2,365x.** Batch throughput (3550 samples): NumPy 901,771/s vs Keras 35,575/s.

## End-to-end pipeline (640x480 webcam frames)

| Stage | Mean | P95 |
|---|---|---|
| MediaPipe hand detection | 30.00 ms | 32.14 ms |
| Feature extraction | 0.595 ms | 0.677 ms |
| MLP classification | 0.070 ms | 0.079 ms |
| **Total** | **30.66 ms** | |

Compute ceiling: **33 FPS** (the webcam's own frame rate, typically 30 FPS, is the practical limit).

## Lighting robustness

56 held-out webcam frames (labels 1-5) from the test split; conditions are simulated with photometric transforms.

| Condition | Hand detected | Accuracy when detected | End-to-end accuracy |
|---|---|---|---|
| Original | 98.2% | 87.3% | 85.7% |
| Dim (40%) | 96.4% | 75.9% | 73.2% |
| Very dim (15%) | 14.3% | 62.5% | 8.9% |
| Overexposed | 85.7% | 83.3% | 71.4% |
| Low contrast | 76.8% | 74.4% | 57.1% |
| Sensor noise | 62.5% | 71.4% | 44.6% |
| Motion blur | 89.3% | 82.0% | 73.2% |

## Temporal stability under landmark jitter

Each test sample is replayed as an 8-frame window with i.i.d. Gaussian jitter added to every landmark's x/y position (std. dev. in pixels at 640 px width).

| Jitter (px @ 640) | Per-frame error | Rolling-vote false trigger | Abstain |
|---|---|---|---|
| 0.0 | 3.32% | 2.45% | 1.4% |
| 2.0 | 3.55% | 2.82% | 0.9% |
| 4.0 | 4.49% | 3.46% | 0.7% |
| 8.0 | 9.33% | 5.13% | 0.9% |
| 12.0 | 19.19% | 9.32% | 2.2% |
| 16.0 | 31.93% | 15.24% | 5.8% |

## Framing robustness

Test-set hands are shifted (up to the given fraction of the frame) and re-scaled about the wrist before classification.

| Framing | Accuracy | 4 read as 5 |
|---|---|---|
| As captured | 96.7% | 2 |
| Off-centre | 96.7% | 2 |
| Further away | 96.7% | 2 |
| Closer | 96.7% | 2 |
| Mixed | 96.7% | 2 |
