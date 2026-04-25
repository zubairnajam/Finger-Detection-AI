# Gesture-Controlled Smart Camera - Development Roadmap

## Team Members

* Muhammad Sarim Khan Ghouri (23k-0720) - Model & Training
* Muhammad Zubair Najam (23k-0693) - Integration & UI

---

## Phase 1: Environment Setup & Basic Testing (Week 1)

### Tasks

* [X] Install Python 3.10+
* [ ] Install all dependencies from requirements.txt
* [ ] Run setup_project.py to create folder structure
* [ ] Test webcam access (test_webcam.py)
* [ ] Test MediaPipe hand detection (test_mediapipe.py)

### Deliverable

Working webcam feed with MediaPipe hand landmark visualization

---

## Phase 2: Data Collection (Week 1-2)

### Tasks

* [ ] Create data collection script
* [ ] Collect 500-1000 images per finger count (1-5 fingers)
* [ ] Vary lighting conditions, backgrounds, hand orientations
* [ ] Extract MediaPipe landmarks from each image
* [ ] Save landmarks as CSV/NumPy arrays with labels
* [ ] Split data: 80% train, 20% test

### Deliverable

Processed dataset with landmark features and labels

---

## Phase 3: Model Development (Week 2-3)

### Sarim's Responsibilities

* [ ] Design CNN/MLP architecture
* [ ] Implement data preprocessing pipeline
* [ ] Implement data augmentation strategies
* [ ] Set up training pipeline with TensorFlow/Keras
* [ ] Train model and tune hyperparameters
* [ ] Generate training curves (loss, accuracy)
* [ ] Create confusion matrix on test set
* [ ] Achieve target: 95%+ accuracy

### Deliverable

Trained model saved as .h5 or .keras file

---

## Phase 4: Real-Time Integration (Week 3-4)

### Zubair's Responsibilities

* [ ] Build webcam capture loop with MediaPipe
* [ ] Integrate trained model for inference
* [ ] Implement rolling window for stable finger count detection
* [ ] Build countdown timer logic
* [ ] Design UI overlay (finger count, countdown, status)
* [ ] Implement auto-capture on countdown completion
* [ ] Save captured images with timestamps

### Deliverable

Working end-to-end prototype

---

## Phase 5: Testing & Optimization (Week 4-5)

### Joint Tasks

* [ ] Test under various lighting conditions
* [ ] Test with different backgrounds
* [ ] Test with different hand sizes
* [ ] Measure system latency (frames per second)
* [ ] Calculate precision, recall, accuracy metrics
* [ ] Fix edge cases and bugs
* [ ] Optimize for smooth real-time performance

### Deliverable

Robust, tested system meeting all objectives

---

## Phase 6: Documentation & Presentation (Week 5-6)

### Joint Tasks

* [ ] Write final project report
* [ ] Create demo video
* [ ] Prepare presentation slides
* [ ] Practice live demonstration
* [ ] Document code with comments and docstrings
* [ ] Create README with installation and usage instructions

### Deliverable

Complete project submission package

---

## Key Milestones

| Milestone                     | Target Date   | Status     |
| ----------------------------- | ------------- | ---------- |
| Environment setup complete    | End of Week 1 | ⏳ Pending |
| Dataset collected             | End of Week 2 | ⏳ Pending |
| Model trained (95%+ accuracy) | End of Week 3 | ⏳ Pending |
| Real-time system working      | End of Week 4 | ⏳ Pending |
| Testing complete              | End of Week 5 | ⏳ Pending |
| Final submission ready        | End of Week 6 | ⏳ Pending |

---

## Success Criteria

✅ Model achieves 95%+ accuracy on test set
✅ System runs at 15+ FPS on standard laptop
✅ Countdown timer is smooth and accurate
✅ Auto-capture works reliably
✅ System handles various lighting and backgrounds
✅ Live demo is successful

---

## Next Immediate Steps

1. Install all dependencies
2. Run test scripts to verify setup
3. Start data collection script development
4. Begin collecting initial dataset samples
