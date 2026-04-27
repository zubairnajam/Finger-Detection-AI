# Gesture-to-Camera-Control Pipeline

This diagram shows how the system processes each webcam frame from raw input to
camera capture.

```mermaid
flowchart LR
    A["📷 Webcam\nFrame"] --> B["🖐️ MediaPipe\nHand Landmarker"]
    B --> C["📐 Flatten\n21 Landmarks → 63 Features"]
    C --> D["🧠 TF/Keras MLP\nClassifier"]
    D --> E["📊 Raw Prediction\n(0-5 + confidence)"]
    E --> F["🔄 Rolling-Window\nStabiliser"]
    F --> G{"Stable\nGesture?"}
    G -- "Yes (1-5)" --> H["⏱️ Start N-second\nCountdown"]
    G -- "No" --> I["🔁 Continue\nDetecting"]
    G -- "0 fingers" --> J["❌ Cancel\nCountdown"]
    H --> K{"Timer\nExpired?"}
    K -- "Yes" --> L["📸 Capture\nPhoto"]
    K -- "No" --> M["🖥️ Show Countdown\nOverlay"]
    M --> A
    L --> N["💾 Save Image\n+ Metadata JSON"]
    N --> A
    I --> A
    J --> A
```

## Component Details

| Stage | Technology | Details |
|-------|-----------|---------|
| Hand Detection | MediaPipe Hand Landmarker | float16 model, 0.35 min confidence |
| Feature Extraction | NumPy | 21 landmarks × 3 coords (x, y, z) = 63 features |
| Classification | TensorFlow / Keras | MLP: 128 → 64 → 32 → 6 (softmax), ~260 KB |
| Stabilisation | Rolling Window Vote | 8-frame window, min 4 votes, 55% confidence threshold |
| Camera Control | OpenCV | Countdown timer → auto-capture with flash effect |
