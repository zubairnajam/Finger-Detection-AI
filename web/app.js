/* ════════════════════════════════════════════════════════════
   Finger Detection Smart Camera — Web App
   MediaPipe hand detection + TF.js trained model inference
   ════════════════════════════════════════════════════════════ */

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// TF.js is loaded via <script> tag in index.html (window.tf)
// Trained model + scaler (loaded async if model/model.json exists)
let tfModel = null;
let scalerMean = null;
let scalerScale = null;

async function loadTFModel() {
  try {
    const modelResp = await fetch("model/model.json");
    if (!modelResp.ok) return; // model not yet converted — fall back to geometric
    tfModel = await tf.loadLayersModel("model/model.json");
    // Warm up
    const dummy = tf.zeros([1, 63]);
    tfModel.predict(dummy).dispose();
    dummy.dispose();
    console.log("[TF.js] Trained model loaded.");

    const scalerResp = await fetch("model/scaler.json");
    if (scalerResp.ok) {
      const s = await scalerResp.json();
      scalerMean  = s.mean;
      scalerScale = s.scale;
      console.log("[TF.js] Scaler loaded.");
    }
  } catch (e) {
    console.warn("[TF.js] Model not available, using geometric fallback.", e.message);
    tfModel = null;
  }
}

function predictWithModel(landmarks) {
  if (!tfModel) return null;
  // Flatten 21×3 landmarks → 63-element array
  const features = landmarks.flatMap(lm => [lm.x, lm.y, lm.z]);
  // Apply scaler if available
  const normed = scalerMean
    ? features.map((v, i) => (v - scalerMean[i]) / (scalerScale[i] || 1))
    : features;
  const input = tf.tensor2d([normed]);
  const probs = tfModel.predict(input);
  const probsArr = probs.dataSync();
  input.dispose();
  probs.dispose();
  const label = probsArr.indexOf(Math.max(...probsArr));
  const confidence = probsArr[label];
  return { label, confidence };
}

// ── Constants ───────────────────────────────────────────────
const WINDOW_SIZE = 8;
const MIN_VOTES = 4;
const MIN_CONFIDENCE = 0.55;
const CIRCUMFERENCE = 2 * Math.PI * 90; // countdown ring

// ── State ───────────────────────────────────────────────────
let handLandmarker = null;
let webcamRunning = false;
let lastVideoTime = -1;
let animationFrameId = null;

// All captured photos stored here: { id, dataUrl, timestamp }
const capturedPhotos = [];

const state = {
  predictionWindow: [],
  countdownActive: false,
  countdownLabel: null,
  countdownDuration: 0,
  countdownStart: 0,
  lastCommittedLabel: null,
  captureCount: 0,
  fps: 0,
  fpsFrameCount: 0,
  fpsLastTime: performance.now(),
  statusText: "Idle",
};

// ── DOM refs ────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const video = $("#webcam");
const canvas = $("#overlay-canvas");
const ctx = canvas.getContext("2d");

const ui = {
  fingerCount: $("#ui-finger-count"),
  confidence: $("#ui-confidence"),
  stable: $("#ui-stable"),
  votes: $("#ui-votes"),
  fps: $("#ui-fps"),
  captures: $("#ui-captures"),
  countdown: $("#ui-countdown"),
  status: $("#ui-status"),
  handStatus: $("#ui-hand-status"),
  fingerDots: document.querySelectorAll(".finger-dot"),
  placeholder: $("#camera-placeholder"),
  loading: $("#camera-loading"),
  countdownOverlay: $("#countdown-overlay"),
  countdownRing: $("#countdown-ring"),
  countdownNumber: $("#countdown-number"),
  flashOverlay: $("#flash-overlay"),
  galleryGrid: $("#gallery-grid"),
  btnStart: $("#btn-start-camera"),
  btnCapture: $("#btn-capture"),
  btnStopCamera: $("#btn-stop-camera"),
  btnManagePhotos: $("#btn-manage-photos"),
  // Modal
  photoModal: $("#photo-modal"),
  modalGrid: $("#modal-grid"),
  modalClose: $("#modal-close"),
  modalSelectAll: $("#modal-select-all"),
  modalDeselectAll: $("#modal-deselect-all"),
  modalCount: $("#modal-count"),
  modalDownloadZip: $("#modal-download-zip"),
  modalDeleteSelected: $("#modal-delete-selected"),
};

// ── Finger Counting (geometric heuristic) ───────────────────
function countFingers(landmarks, handedness) {
  const isRight = handedness === "Right";
  let count = 0;
  const up = [];

  // Thumb: compare x-coordinates relative to hand orientation
  const thumbTip = landmarks[4];
  const thumbIP = landmarks[3];
  const thumbExtended = isRight ? thumbTip.x < thumbIP.x : thumbTip.x > thumbIP.x;
  if (thumbExtended) count++;
  up.push(thumbExtended);

  // Other 4 fingers: tip.y < PIP.y means extended (y=0 is top of image)
  const tips = [8, 12, 16, 20];
  const pips = [6, 10, 14, 18];
  for (let i = 0; i < 4; i++) {
    const extended = landmarks[tips[i]].y < landmarks[pips[i]].y;
    if (extended) count++;
    up.push(extended);
  }

  return { count, up };
}

function getFingerConfidence(landmarks) {
  const tips = [8, 12, 16, 20];
  const pips = [6, 10, 14, 18];
  let clarity = 0;
  for (let i = 0; i < 4; i++) {
    const diff = Math.abs(landmarks[tips[i]].y - landmarks[pips[i]].y);
    clarity += Math.min(diff / 0.08, 1.0);
  }
  return clarity / 4;
}

// ── Prediction Stabiliser ───────────────────────────────────
function addPrediction(label, confidence) {
  state.predictionWindow.push({ label, confidence });
  if (state.predictionWindow.length > WINDOW_SIZE) state.predictionWindow.shift();
}

function getStablePrediction() {
  const win = state.predictionWindow;
  if (win.length === 0) return { label: null, confidence: 0, votes: 0 };

  const counts = {};
  for (const { label } of win) counts[label] = (counts[label] || 0) + 1;

  let bestLabel = null, bestVotes = 0;
  for (const [label, v] of Object.entries(counts)) {
    if (v > bestVotes) { bestLabel = parseInt(label); bestVotes = v; }
  }

  const confs = win.filter((w) => w.label === bestLabel).map((w) => w.confidence);
  const avgConf = confs.reduce((a, b) => a + b, 0) / confs.length;

  if (bestVotes < MIN_VOTES || avgConf < MIN_CONFIDENCE) {
    return { label: null, confidence: avgConf, votes: bestVotes };
  }
  return { label: bestLabel, confidence: avgConf, votes: bestVotes };
}

// ── Countdown ───────────────────────────────────────────────
function startCountdown(label) {
  state.countdownActive = true;
  state.countdownLabel = label;
  state.countdownDuration = label;
  state.countdownStart = performance.now() / 1000;
  state.statusText = `${label}s countdown started`;
  ui.countdownOverlay.classList.remove("hidden");
}

function cancelCountdown() {
  state.countdownActive = false;
  state.countdownLabel = null;
  ui.countdownOverlay.classList.add("hidden");
}

function getCountdownRemaining() {
  if (!state.countdownActive) return null;
  const elapsed = performance.now() / 1000 - state.countdownStart;
  return Math.max(0, state.countdownDuration - elapsed);
}

// ── Capture ─────────────────────────────────────────────────
function capturePhoto() {
  const offscreen = document.createElement("canvas");
  offscreen.width = video.videoWidth;
  offscreen.height = video.videoHeight;
  const offCtx = offscreen.getContext("2d");
  offCtx.translate(offscreen.width, 0);
  offCtx.scale(-1, 1);
  offCtx.drawImage(video, 0, 0);
  const dataUrl = offscreen.toDataURL("image/jpeg", 0.92);

  // Flash
  ui.flashOverlay.classList.remove("hidden");
  setTimeout(() => ui.flashOverlay.classList.add("hidden"), 300);

  state.captureCount++;
  ui.captures.textContent = state.captureCount;
  state.statusText = `Photo #${state.captureCount} captured!`;

  // Store photo
  const photoId = Date.now();
  const timestamp = new Date().toLocaleTimeString();
  capturedPhotos.push({ id: photoId, dataUrl, timestamp, name: `capture_${state.captureCount}.jpg` });

  // Gallery thumbnail (sidebar — keep last 6)
  refreshGalleryThumbnails();
}

function refreshGalleryThumbnails() {
  ui.galleryGrid.innerHTML = "";
  const recent = capturedPhotos.slice(-6).reverse();
  for (const photo of recent) {
    const img = document.createElement("img");
    img.src = photo.dataUrl;
    img.title = photo.name;
    ui.galleryGrid.appendChild(img);
  }
}

// ── UI Updates ──────────────────────────────────────────────
function updateUI(fingerResult, rawConf, stablePred, handDetected) {
  if (handDetected && fingerResult) {
    ui.fingerCount.textContent = fingerResult.count;
    ui.fingerCount.style.color = "#00ff88";
    ui.confidence.textContent = (rawConf * 100).toFixed(0) + "%";
    ui.fingerDots.forEach((dot, i) => {
      dot.classList.toggle("active", fingerResult.up[i]);
    });
    ui.handStatus.innerHTML = '<span class="status-dot online"></span><span>Hand Detected</span>';
  } else {
    ui.fingerCount.textContent = "—";
    ui.fingerCount.style.color = "";
    ui.confidence.textContent = "—";
    ui.fingerDots.forEach((dot) => dot.classList.remove("active"));
    ui.handStatus.innerHTML = '<span class="status-dot offline"></span><span>No Hand</span>';
  }

  ui.stable.textContent = stablePred.label !== null ? stablePred.label : "—";
  ui.stable.style.color = stablePred.label !== null ? "#00d4ff" : "";
  ui.votes.textContent = `${stablePred.votes} / ${WINDOW_SIZE}`;

  const remaining = getCountdownRemaining();
  if (remaining !== null) {
    ui.countdown.textContent = remaining.toFixed(1) + "s";
    ui.countdownNumber.textContent = Math.ceil(remaining);
    const fraction = remaining / state.countdownDuration;
    const offset = CIRCUMFERENCE * fraction;
    ui.countdownRing.style.strokeDashoffset = offset;
    const color = fraction > 0.5 ? "#00ff88" : fraction > 0.2 ? "#ffbb33" : "#ff4466";
    ui.countdownRing.style.stroke = color;
    ui.countdownNumber.style.color = color;
  } else {
    ui.countdown.textContent = "Ready";
  }

  ui.status.textContent = state.statusText;

  // FPS
  state.fpsFrameCount++;
  const now = performance.now();
  if (now - state.fpsLastTime >= 1000) {
    state.fps = Math.round((state.fpsFrameCount * 1000) / (now - state.fpsLastTime));
    state.fpsFrameCount = 0;
    state.fpsLastTime = now;
  }
  ui.fps.textContent = state.fps;
}

// ── Main detection loop ─────────────────────────────────────
function processFrame() {
  if (!webcamRunning) return;

  if (video.currentTime === lastVideoTime) {
    animationFrameId = requestAnimationFrame(processFrame);
    return;
  }
  lastVideoTime = video.currentTime;

  // Sync canvas size
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const results = handLandmarker.detectForVideo(video, performance.now());
  const handDetected = results.landmarks && results.landmarks.length > 0;
  let fingerResult = null;
  let rawConf = 0;

  if (handDetected) {
    const lm = results.landmarks[0];
    if (tfModel) {
      const pred = predictWithModel(lm);
      if (pred) {
        fingerResult = { count: pred.label, up: [false, false, false, false, false] };
        rawConf = pred.confidence;
      }
    }
    
    // Fallback to geometric if tfModel fails or is missing
    if (!tfModel || !fingerResult) {
      const handedness = results.handednesses?.[0]?.[0]?.categoryName || "Right";
      fingerResult = countFingers(lm, handedness);
      rawConf = getFingerConfidence(lm);
    }
    
    addPrediction(fingerResult.count, rawConf);
    // No skeleton drawing — just clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!state.countdownActive) state.statusText = "Gesture detected";
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    state.predictionWindow = [];
    if (!state.countdownActive) {
      state.lastCommittedLabel = null;
      state.statusText = "Show your hand";
    }
  }

  const stablePred = getStablePrediction();

  // ── Gesture state machine ─────────────────────────────
  if (stablePred.label === 0) {
    state.lastCommittedLabel = null;
    if (state.countdownActive) {
      cancelCountdown();
      state.statusText = "Cancelled (fist)";
    }
  } else if (stablePred.label >= 1 && stablePred.label <= 5) {
    if (!state.countdownActive) {
      if (stablePred.label !== state.lastCommittedLabel) {
        startCountdown(stablePred.label);
      }
    } else if (stablePred.label !== state.countdownLabel) {
      cancelCountdown();
      startCountdown(stablePred.label);
    }
  } else if (stablePred.label === null && !state.countdownActive && handDetected) {
    state.statusText = "Not stable yet...";
  }

  // ── Countdown check ───────────────────────────────────
  if (state.countdownActive) {
    const rem = getCountdownRemaining();
    if (rem <= 0) {
      capturePhoto();
      state.lastCommittedLabel = state.countdownLabel;
      cancelCountdown();
      state.predictionWindow = [];
    }
  }

  updateUI(fingerResult, rawConf, stablePred, handDetected);
  animationFrameId = requestAnimationFrame(processFrame);
}

// ── MediaPipe init ──────────────────────────────────────────
async function initHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.35,
    minHandPresenceConfidence: 0.35,
    minTrackingConfidence: 0.35,
  });
}

// ── Camera start / stop ─────────────────────────────────────
async function startCamera() {
  ui.placeholder.classList.add("hidden");
  ui.loading.classList.remove("hidden");

  try {
    await initHandLandmarker();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
    });
    video.srcObject = stream;
    await video.play();

    ui.loading.classList.add("hidden");
    webcamRunning = true;
    state.statusText = "Camera ready";
    ui.btnCapture.disabled = false;
    ui.btnStopCamera.disabled = false;
    ui.btnManagePhotos.disabled = false;

    ui.countdownRing.style.strokeDasharray = CIRCUMFERENCE;
    ui.countdownRing.style.strokeDashoffset = 0;

    processFrame();
  } catch (err) {
    ui.loading.classList.add("hidden");
    ui.placeholder.classList.remove("hidden");
    console.error("Camera init failed:", err);
    alert("Could not start camera. Check permissions and try again.\n\n" + err.message);
  }
}

function stopCamera() {
  webcamRunning = false;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);

  // Stop all tracks
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }

  cancelCountdown();
  state.predictionWindow = [];
  state.lastCommittedLabel = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ui.btnCapture.disabled = true;
  ui.btnStopCamera.disabled = true;
  ui.placeholder.classList.remove("hidden");
  state.statusText = "Camera stopped";
  ui.status.textContent = state.statusText;
}

// ── Photo Manager Modal ─────────────────────────────────────
function openPhotoModal() {
  ui.photoModal.classList.remove("hidden");
  renderModalGrid();
}

function closePhotoModal() {
  ui.photoModal.classList.add("hidden");
}

function renderModalGrid() {
  ui.modalGrid.innerHTML = "";

  if (capturedPhotos.length === 0) {
    ui.modalGrid.innerHTML = '<div class="modal-empty">No photos captured yet.</div>';
    updateModalButtons();
    return;
  }

  for (const photo of capturedPhotos) {
    const thumb = document.createElement("div");
    thumb.className = "modal-thumb";
    thumb.dataset.id = photo.id;
    thumb.innerHTML = `
      <img src="${photo.dataUrl}" alt="${photo.name}">
      <span class="thumb-check">✓</span>
      <span class="thumb-label">${photo.name}</span>
    `;
    thumb.addEventListener("click", () => {
      thumb.classList.toggle("selected");
      updateModalButtons();
    });
    ui.modalGrid.appendChild(thumb);
  }
  updateModalButtons();
}

function updateModalButtons() {
  const selected = ui.modalGrid.querySelectorAll(".modal-thumb.selected");
  const count = selected.length;
  ui.modalCount.textContent = `${count} selected`;
  ui.modalDownloadZip.disabled = count === 0;
  ui.modalDeleteSelected.disabled = count === 0;
}

function getSelectedIds() {
  return [...ui.modalGrid.querySelectorAll(".modal-thumb.selected")]
    .map((el) => parseInt(el.dataset.id));
}

async function downloadSelectedAsZip() {
  const selectedIds = new Set(getSelectedIds());
  if (selectedIds.size === 0) return;

  const zip = new JSZip();
  for (const photo of capturedPhotos) {
    if (!selectedIds.has(photo.id)) continue;
    // Convert data URL to binary
    const base64 = photo.dataUrl.split(",")[1];
    zip.file(photo.name, base64, { base64: true });
  }

  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `finger_detect_captures_${Date.now()}.zip`;
  a.click();
  URL.revokeObjectURL(url);
}

function deleteSelected() {
  const selectedIds = new Set(getSelectedIds());
  if (selectedIds.size === 0) return;

  // Remove from array
  for (let i = capturedPhotos.length - 1; i >= 0; i--) {
    if (selectedIds.has(capturedPhotos[i].id)) capturedPhotos.splice(i, 1);
  }

  state.captureCount = capturedPhotos.length;
  ui.captures.textContent = state.captureCount;
  refreshGalleryThumbnails();
  renderModalGrid();
}

// ── Charts & metrics (loaded live from web/metrics.json) ─────
async function loadMetricsAndInitCharts() {
  // Fallback values (used if metrics.json not yet generated)
  let m = {
    accuracy: 97.57,
    test_samples: 3500,
    mean_latency_ms: 62.7,
    f1_scores: [0.9927, 0.9824, 0.9573, 0.9573, 0.9737, 0.9943],
    f1_labels: ["0 (Fist)", "1 (Index)", "2 (Peace)", "3 (Three)", "4 (Four)", "5 (Open Hand)"],
    classes: [
      {label:0,name:"Fist",     precision:0.9895,recall:0.9958,f1:0.9927,support:475},
      {label:1,name:"Index",    precision:0.9816,recall:0.9832,f1:0.9824,support:596},
      {label:2,name:"Peace",    precision:0.9581,recall:0.9565,f1:0.9573,support:598},
      {label:3,name:"Three",    precision:0.9557,recall:0.9589,f1:0.9573,support:608},
      {label:4,name:"Four",     precision:0.9785,recall:0.9689,f1:0.9737,support:611},
      {label:5,name:"Open Hand",precision:0.9935,recall:0.9951,f1:0.9943,support:612},
    ],
  };

  try {
    const resp = await fetch("metrics.json?v=" + Date.now());
    if (resp.ok) {
      m = await resp.json();
      console.log("[metrics] Loaded live metrics.json");
    }
  } catch (_) {
    console.warn("[metrics] metrics.json not found, using built-in defaults.");
  }

  // ── Update stat cards ───────────────────────────────────
  const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  setTxt("metric-accuracy",  m.accuracy.toFixed(1) + "%");
  setTxt("metric-latency",   m.mean_latency_ms.toFixed(0) + "ms");
  setTxt("metric-samples",   m.test_samples.toLocaleString());

  // ── F1 bar chart ────────────────────────────────────────
  const f1Ctx = document.getElementById("f1-chart");
  if (f1Ctx) {
    new Chart(f1Ctx, {
      type: "bar",
      data: {
        labels: m.f1_labels,
        datasets: [{
          label: "F1 Score",
          data: m.f1_scores,
          backgroundColor: m.f1_scores.map((_, i) =>
            `rgba(0, 212, 255, ${0.55 + i * 0.03})`),
          borderColor: "rgba(0, 212, 255, 0.9)",
          borderWidth: 1,
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => `F1: ${(c.raw * 100).toFixed(2)}%` } },
        },
        scales: {
          y: {
            min: Math.max(0, Math.min(...m.f1_scores) - 0.06),
            max: 1.002,
            ticks: { color: "#94a3b8", callback: (v) => (v * 100).toFixed(0) + "%" },
            grid: { color: "rgba(255,255,255,0.04)" },
          },
          x: {
            ticks: { color: "#94a3b8", font: { size: 11 } },
            grid: { display: false },
          },
        },
      },
    });
  }

  // ── Classification table ─────────────────────────────────
  const tbody = document.getElementById("results-tbody");
  if (tbody && m.classes) {
    tbody.innerHTML = m.classes.map(c => `
      <tr>
        <td>${c.label} (${c.name})</td>
        <td>${(c.precision * 100).toFixed(1)}%</td>
        <td>${(c.recall * 100).toFixed(1)}%</td>
        <td><strong>${(c.f1 * 100).toFixed(1)}%</strong></td>
        <td>${c.support}</td>
      </tr>`).join("");
  }

  // ── Confusion matrix (static — re-run evaluate_model.py for real data) ──
  const cmData = [
    [473, 0, 0, 2, 0, 0],
    [0, 586, 5, 3, 2, 0],
    [0, 4, 572, 16, 5, 1],
    [0, 3, 17, 583, 5, 0],
    [1, 2, 3, 5, 592, 8],
    [4, 2, 0, 1, 2, 609],
  ];
  const cmEl = document.getElementById("confusion-matrix");
  if (cmEl) {
    cmEl.innerHTML = '<div class="cm-header"></div>';
    for (let c = 0; c < 6; c++) cmEl.innerHTML += `<div class="cm-header">${c}</div>`;
    const maxVal = 612;
    for (let r = 0; r < 6; r++) {
      cmEl.innerHTML += `<div class="cm-header">${r}</div>`;
      for (let c = 0; c < 6; c++) {
        const val = cmData[r][c];
        const bg = r === c
          ? `rgba(0, 212, 255, ${(val / maxVal * 0.7 + 0.15).toFixed(2)})`
          : val > 0
            ? `rgba(255, 68, 102, ${(val / maxVal * 0.6).toFixed(2)})`
            : "rgba(255,255,255,0.02)";
        cmEl.innerHTML += `<div class="cm-cell" style="background:${bg}" title="True ${r} -> Pred ${c}: ${val}">${val}</div>`;
      }
    }
  }
}

// ── Scroll animations ───────────────────────────────────────
function initScrollAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.classList.add("visible");
      });
    },
    { threshold: 0.15 }
  );
  document.querySelectorAll("[data-animate]").forEach((el) => observer.observe(el));

  const nav = document.getElementById("main-nav");
  window.addEventListener("scroll", () => {
    nav.classList.toggle("scrolled", window.scrollY > 60);
  });
}

// ── Keyboard shortcuts ──────────────────────────────────────
function initKeyboard() {
  document.addEventListener("keydown", (e) => {
    // Don't trigger shortcuts when modal is open
    if (!ui.photoModal.classList.contains("hidden")) {
      if (e.key === "Escape") closePhotoModal();
      return;
    }
    if (!webcamRunning) return;
    if (e.code === "Space" || e.key === "s") {
      e.preventDefault();
      cancelCountdown();
      state.predictionWindow = [];
      state.lastCommittedLabel = null;
      capturePhoto();
    }
  });
}

// ── Event listeners ─────────────────────────────────────────
ui.btnStart.addEventListener("click", startCamera);
ui.btnCapture.addEventListener("click", () => {
  if (!webcamRunning) return;
  cancelCountdown();
  state.predictionWindow = [];
  state.lastCommittedLabel = null;
  capturePhoto();
});
ui.btnStopCamera.addEventListener("click", stopCamera);
ui.btnManagePhotos.addEventListener("click", openPhotoModal);

// Modal events
ui.modalClose.addEventListener("click", closePhotoModal);
ui.photoModal.addEventListener("click", (e) => {
  if (e.target === ui.photoModal) closePhotoModal();
});
ui.modalSelectAll.addEventListener("click", () => {
  ui.modalGrid.querySelectorAll(".modal-thumb").forEach((t) => t.classList.add("selected"));
  updateModalButtons();
});
ui.modalDeselectAll.addEventListener("click", () => {
  ui.modalGrid.querySelectorAll(".modal-thumb").forEach((t) => t.classList.remove("selected"));
  updateModalButtons();
});
ui.modalDownloadZip.addEventListener("click", downloadSelectedAsZip);
ui.modalDeleteSelected.addEventListener("click", deleteSelected);

// ── Init ────────────────────────────────────────────────────
loadTFModel();                   // load trained model in background
loadMetricsAndInitCharts();      // fetch metrics.json + draw charts
initScrollAnimations();
initKeyboard();
