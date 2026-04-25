# Finger Detection Smart Camera

Final deliverable:
- `src/final_camera_app.py` is the main camera app.
- It works like a normal camera with `SPACE`, and it also supports finger-count countdown capture.

Project layout:
- `src/` final runtime app
- `tools/` data collection, preprocessing, training, health check, model download, and legacy prototype scripts
- `docs/` archived planning notes and reference material
- `data/` raw and processed datasets
- `models/` trained model artifacts
- `outputs/final_camera/` final camera photos
- `outputs/training/` training reports and plots
- `outputs/legacy_captures/` captures from the older prototype
- `archive/` external dataset archive kept intact

Main commands:

```bash
python src/final_camera_app.py
python tools/data_collection.py
python tools/preprocess_landmarks.py
python tools/train_model.py
python tools/dataset_health_check.py
python tools/download_model.py
```

Notes:
- The final app is the only script you need for the deliverable demo.
- The `tools/` scripts are kept for retraining, data expansion, and debugging.
- `archive/` was intentionally left untouched.
