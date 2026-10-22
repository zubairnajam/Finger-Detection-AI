/* ════════════════════════════════════════════════════════════
   Finger Detection Smart Camera — Web App
   MediaPipe hand detection + dependency-free MLP inference
   ════════════════════════════════════════════════════════════ */

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";
import { MLP } from "./mlp.js";

// ── Constants ───────────────────────────────────────────────
const WINDOW_SIZE = 8;
const MIN_VOTES = 4;
const MIN_CONFIDENCE = 0.55;
const CIRCUMFERENCE = 2 * Math.PI * 90; // countdown ring
const CLASS_NAMES = ["Fist", "Index", "Peace", "Three", "Four", "Open Hand"];
const EMA = 0.1;
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Hand topology (MediaPipe 21-point), grouped by finger for colour-coding
const FINGERS = [
  { chain: [0, 1, 2, 3, 4], color: "#ff7a5a" },
  { chain: [0, 5, 6, 7, 8], color: "#ffd23c" },
  { chain: [9, 10, 11, 12], color: "#78ff78" },
  { chain: [13, 14, 15, 16], color: "#3cd2ff" },
  { chain: [0, 17, 18, 19, 20], color: "#c878ff" },
];
const PALM = [5, 9, 13, 17];

// Chart palette (validated categorical slots, dark-surface steps) + inks
const C_BLUE = "#3987e5";
const C_ORANGE = "#d95926";
const INK_2 = "#94a3b8";
const GRID = "rgba(255,255,255,0.06)";

// ── State ───────────────────────────────────────────────────
let mlp = null;                 // trained classifier (JS engine)
let handLandmarker = null;
let webcamRunning = false;
let lastVideoTime = -1;
let animationFrameId = null;
const capturedPhotos = [];      // { id, dataUrl, timestamp, name }

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
  detectMs: 0,
  classifyMs: 0,
  statusText: "Idle",
  showSkeleton: true,
};

// ── DOM refs ────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const video = $("#webcam");
const canvas = $("#overlay-canvas");
const ctx = canvas.getContext("2d");

const ui = {
  fingerCount: $("#ui-finger-count"),
  confidence: $("#ui-confidence"),
  confBar: $("#ui-conf-bar"),
  stable: $("#ui-stable"),
  gestureName: $("#ui-gesture-name"),
  votes: $("#ui-votes"),
  voteMeter: $("#ui-vote-meter"),
  probs: $("#ui-probs"),
  fps: $("#ui-fps"),
  detectMs: $("#ui-detect-ms"),
  classifyMs: $("#ui-classify-ms"),
  engine: $("#ui-engine"),
  captures: $("#ui-captures"),
  countdown: $("#ui-countdown"),
  status: $("#ui-status"),
  handStatus: $("#ui-hand-status"),
  placeholder: $("#camera-placeholder"),
  loading: $("#camera-loading"),
  loadingText: $("#camera-loading-text"),
  chip: $("#camera-chip"),
  countdownOverlay: $("#countdown-overlay"),
  countdownRing: $("#countdown-ring"),
  countdownNumber: $("#countdown-number"),
  flashOverlay: $("#flash-overlay"),
  galleryGrid: $("#gallery-grid"),
  btnStart: $("#btn-start-camera"),
  btnCapture: $("#btn-capture"),
  btnStopCamera: $("#btn-stop-camera"),
  btnManagePhotos: $("#btn-manage-photos"),
  toggleSkeleton: $("#toggle-skeleton"),
  photoModal: $("#photo-modal"),
  modalGrid: $("#modal-grid"),
  modalClose: $("#modal-close"),
  modalSelectAll: $("#modal-select-all"),
  modalDeselectAll: $("#modal-deselect-all"),
  modalCount: $("#modal-count"),
  modalDownloadZip: $("#modal-download-zip"),
  modalDeleteSelected: $("#modal-delete-selected"),
};

// Build static panel widgets once
const voteSegments = Array.from({ length: WINDOW_SIZE }, () => {
  const seg = document.createElement("span");
  seg.className = "vote-seg";
  ui.voteMeter.appendChild(seg);
  return seg;
});
const probRows = CLASS_NAMES.map((name, i) => {
  const li = document.createElement("li");
  li.className = "prob-row";
  li.innerHTML = `<span class="prob-label">${i} ${name}</span>
    <span class="prob-track"><span class="prob-fill"></span></span>
    <span class="prob-pct">—</span>`;
  ui.probs.appendChild(li);
  return { li, fill: li.querySelector(".prob-fill"), pct: li.querySelector(".prob-pct") };
});

// ── Model ───────────────────────────────────────────────────
async function loadModel() {
  try {
    mlp = await MLP.load("model/mlp.json");
    ui.engine.textContent = "Neural net";
    ui.engine.title = `${mlp.parameters.toLocaleString()}-parameter MLP, plain-JS inference`;
    console.log(`[model] Loaded MLP (${mlp.parameters} params)`);
  } catch (e) {
    mlp = null;
    ui.engine.textContent = "Geometric";
    ui.engine.title = "Trained model unavailable — using a rule-based fallback";
    console.warn("[model] Falling back to geometric finger counting:", e.message);
  }
}

// ── Geometric fallback (and per-finger visualisation) ───────
function countFingers(landmarks, handedness) {
  const isRight = handedness === "Right";
  const up = [];
  const thumbTip = landmarks[4];
  const thumbIP = landmarks[3];
  up.push(isRight ? thumbTip.x < thumbIP.x : thumbTip.x > thumbIP.x);
  const tips = [8, 12, 16, 20];
  const pips = [6, 10, 14, 18];
  for (let i = 0; i < 4; i++) up.push(landmarks[tips[i]].y < landmarks[pips[i]].y);
  return { count: up.filter(Boolean).length, up };
}

function geometricConfidence(landmarks) {
  const tips = [8, 12, 16, 20];
  const pips = [6, 10, 14, 18];
  let clarity = 0;
  for (let i = 0; i < 4; i++) {
    clarity += Math.min(Math.abs(landmarks[tips[i]].y - landmarks[pips[i]].y) / 0.08, 1.0);
  }
  return clarity / 4;
}

// ── Prediction stabiliser (mirrors src/prediction.py) ───────
function addPrediction(label, confidence) {
  state.predictionWindow.push({ label, confidence });
  if (state.predictionWindow.length > WINDOW_SIZE) state.predictionWindow.shift();
}

function getStablePrediction() {
  const win = state.predictionWindow;
  if (win.length === 0) return { label: null, confidence: 0, votes: 0 };

  const counts = new Map();
  for (const { label } of win) counts.set(label, (counts.get(label) || 0) + 1);
  let bestLabel = null, bestVotes = 0;
  for (const [label, v] of counts) if (v > bestVotes) { bestLabel = label; bestVotes = v; }

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
  state.statusText = `${label}-second countdown started`;
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

  ui.flashOverlay.classList.remove("hidden");
  setTimeout(() => ui.flashOverlay.classList.add("hidden"), 300);

  state.captureCount++;
  ui.captures.textContent = state.captureCount;
  state.statusText = `Photo #${state.captureCount} captured`;

  const now = new Date();
  const stamp = now.toISOString().replace(/[-:]/g, "").replace("T", "_").slice(0, 15);
  capturedPhotos.push({
    id: now.getTime() + Math.random(),
    dataUrl,
    timestamp: now.toLocaleTimeString(),
    name: `capture_${stamp}_${state.captureCount}.jpg`,
  });
  refreshGalleryThumbnails();
}

function refreshGalleryThumbnails() {
  ui.galleryGrid.innerHTML = "";
  if (capturedPhotos.length === 0) {
    ui.galleryGrid.innerHTML = '<p class="gallery-empty">No photos yet</p>';
    return;
  }
  for (const photo of capturedPhotos.slice(-6).reverse()) {
    const img = document.createElement("img");
    img.src = photo.dataUrl;
    img.alt = `Captured at ${photo.timestamp}`;
    img.title = photo.name;
    ui.galleryGrid.appendChild(img);
  }
}

// ── Drawing ─────────────────────────────────────────────────
function drawSkeleton(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks || !state.showSkeleton) return;

  const w = canvas.width, h = canvas.height;
  const s = Math.max(w / 1280, 0.5);
  const pt = (i) => [landmarks[i].x * w, landmarks[i].y * h];
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 3 * s;
  ctx.beginPath();
  PALM.forEach((i, k) => (k ? ctx.lineTo(...pt(i)) : ctx.moveTo(...pt(i))));
  ctx.stroke();

  for (const { chain, color } of FINGERS) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 4 * s;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8 * s;
    ctx.beginPath();
    chain.forEach((i, k) => (k ? ctx.lineTo(...pt(i)) : ctx.moveTo(...pt(i))));
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  ctx.fillStyle = "#fff";
  for (let i = 0; i < landmarks.length; i++) {
    const [x, y] = pt(i);
    ctx.beginPath();
    ctx.arc(x, y, ([4, 8, 12, 16, 20].includes(i) ? 6 : 3.5) * s, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ── UI Updates ──────────────────────────────────────────────
function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

function updateUI(raw, stablePred, handDetected) {
  if (handDetected && raw) {
    setText(ui.fingerCount, String(raw.label));
    setText(ui.confidence, `${Math.round(raw.confidence * 100)}%`);
    ui.handStatus.innerHTML = '<span class="status-dot online"></span><span>Hand detected</span>';
  } else {
    setText(ui.fingerCount, "—");
    setText(ui.confidence, "—");
    ui.handStatus.innerHTML = '<span class="status-dot offline"></span><span>No hand</span>';
  }

  // Headline gesture: stable label if committed, else raw (dimmed)
  const stable = stablePred.label;
  const shown = stable ?? (raw ? raw.label : null);
  setText(ui.stable, shown === null ? "—" : String(shown));
  ui.stable.classList.toggle("is-stable", stable !== null);
  setText(ui.gestureName, stable !== null ? CLASS_NAMES[stable] : raw ? "Stabilising…" : "No hand");

  const conf = stable !== null ? stablePred.confidence : raw ? raw.confidence : 0;
  ui.confBar.style.width = `${(conf * 100).toFixed(1)}%`;
  ui.confBar.dataset.level = conf >= 0.8 ? "high" : conf >= MIN_CONFIDENCE ? "mid" : "low";

  setText(ui.votes, `${stablePred.votes} / ${WINDOW_SIZE}`);
  voteSegments.forEach((seg, i) => {
    seg.classList.toggle("on", i < stablePred.votes);
    seg.classList.toggle("stable", stable !== null);
  });

  const probs = raw?.probs;
  let top = -1;
  if (probs) for (let i = 0; i < probs.length; i++) if (top < 0 || probs[i] > probs[top]) top = i;
  probRows.forEach((row, i) => {
    const p = probs ? probs[i] : 0;
    row.fill.style.width = `${(p * 100).toFixed(1)}%`;
    setText(row.pct, probs ? `${Math.round(p * 100)}%` : "—");
    row.li.classList.toggle("top", i === top);
  });

  const remaining = getCountdownRemaining();
  if (remaining !== null) {
    setText(ui.countdown, `${remaining.toFixed(1)}s`);
    setText(ui.countdownNumber, String(Math.ceil(remaining)));
    const fraction = remaining / state.countdownDuration;
    ui.countdownRing.style.strokeDashoffset = CIRCUMFERENCE * (1 - fraction);
    const color = fraction > 0.5 ? "var(--green)" : fraction > 0.2 ? "var(--yellow)" : "var(--red)";
    ui.countdownRing.style.stroke = color;
    ui.countdownNumber.style.color = color;
  } else {
    setText(ui.countdown, "Ready");
  }

  setText(ui.status, state.statusText);

  state.fpsFrameCount++;
  const now = performance.now();
  if (now - state.fpsLastTime >= 1000) {
    state.fps = Math.round((state.fpsFrameCount * 1000) / (now - state.fpsLastTime));
    state.fpsFrameCount = 0;
    state.fpsLastTime = now;
    setText(ui.fps, String(state.fps));
    setText(ui.detectMs, `${state.detectMs.toFixed(1)} ms`);
    setText(ui.classifyMs, state.classifyMs ? `${state.classifyMs.toFixed(3)} ms` : "—");
  }
}

// ── Main detection loop ─────────────────────────────────────
function processFrame() {
  if (!webcamRunning) return;

  if (video.currentTime === lastVideoTime) {
    animationFrameId = requestAnimationFrame(processFrame);
    return;
  }
  lastVideoTime = video.currentTime;

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  const t0 = performance.now();
  const results = handLandmarker.detectForVideo(video, t0);
  const t1 = performance.now();
  state.detectMs = state.detectMs ? (1 - EMA) * state.detectMs + EMA * (t1 - t0) : t1 - t0;

  const handDetected = results.landmarks && results.landmarks.length > 0;
  let raw = null;

  if (handDetected) {
    const lm = results.landmarks[0];
    if (mlp) {
      const t2 = performance.now();
      const features = new Float32Array(63);
      for (let i = 0; i < 21; i++) {
        features[i * 3] = lm[i].x;
        features[i * 3 + 1] = lm[i].y;
        features[i * 3 + 2] = lm[i].z;
      }
      raw = mlp.predict(features);
      const dt = performance.now() - t2;
      state.classifyMs = state.classifyMs ? (1 - EMA) * state.classifyMs + EMA * dt : dt;
    } else {
      const handedness = results.handednesses?.[0]?.[0]?.categoryName || "Right";
      const geo = countFingers(lm, handedness);
      raw = { label: geo.count, confidence: geometricConfidence(lm), probs: null };
    }
    addPrediction(raw.label, raw.confidence);
    drawSkeleton(lm);
    if (!state.countdownActive) state.statusText = "Gesture detected — hold steady";
  } else {
    drawSkeleton(null);
    state.predictionWindow = [];
    if (!state.countdownActive) {
      state.lastCommittedLabel = null;
      state.statusText = "Show your hand to the camera";
    }
  }

  const stablePred = getStablePrediction();

  // ── Gesture state machine (mirrors src/gesture_controller.py) ──
  if (stablePred.label === 0) {
    state.lastCommittedLabel = null;
    if (state.countdownActive) {
      cancelCountdown();
      state.statusText = "Countdown cancelled (fist)";
    }
  } else if (stablePred.label >= 1 && stablePred.label <= 5) {
    if (!state.countdownActive) {
      if (stablePred.label !== state.lastCommittedLabel) startCountdown(stablePred.label);
    } else if (stablePred.label !== state.countdownLabel) {
      cancelCountdown();
      startCountdown(stablePred.label);
      state.statusText = `Countdown updated to ${stablePred.label}s`;
    }
  } else if (stablePred.label === null && !state.countdownActive && handDetected) {
    state.statusText = "Not stable yet…";
  }

  if (state.countdownActive && getCountdownRemaining() <= 0) {
    capturePhoto();
    state.lastCommittedLabel = state.countdownLabel;
    cancelCountdown();
    state.predictionWindow = [];
  }

  updateUI(raw, stablePred, handDetected);
  animationFrameId = requestAnimationFrame(processFrame);
}

// ── MediaPipe init (cached; GPU with CPU fallback) ──────────
async function initHandLandmarker() {
  if (handLandmarker) return;
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.35,
    minHandPresenceConfidence: 0.35,
    minTrackingConfidence: 0.35,
  });
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, options("GPU"));
  } catch (e) {
    console.warn("[mediapipe] GPU delegate unavailable, using CPU:", e.message);
    handLandmarker = await HandLandmarker.createFromOptions(vision, options("CPU"));
  }
}

function describeCameraError(err) {
  switch (err?.name) {
    case "NotAllowedError":
      return "Camera permission was denied. Allow camera access in your browser's site settings and try again.";
    case "NotFoundError":
      return "No camera was found. Connect a webcam and try again.";
    case "NotReadableError":
      return "The camera is in use by another application. Close it and try again.";
    default:
      return `Could not start the camera: ${err?.message || err}`;
  }
}

// ── Camera start / stop ─────────────────────────────────────
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("This browser does not support webcam access. Try a recent Chrome, Edge, Firefox or Safari over HTTPS.");
    return;
  }
  ui.placeholder.classList.add("hidden");
  ui.loading.classList.remove("hidden");
  ui.btnStart.disabled = true;

  try {
    ui.loadingText.textContent = "Loading hand-tracking model…";
    await initHandLandmarker();
    ui.loadingText.textContent = "Waiting for camera permission…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    ui.loading.classList.add("hidden");
    ui.chip.classList.remove("hidden");
    webcamRunning = true;
    lastVideoTime = -1;
    state.statusText = "Camera ready — show 1–5 fingers";
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
    alert(describeCameraError(err));
  } finally {
    ui.btnStart.disabled = false;
  }
}

function stopCamera() {
  webcamRunning = false;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
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
  ui.chip.classList.add("hidden");
  ui.placeholder.classList.remove("hidden");
  state.statusText = "Camera stopped";
  ui.status.textContent = state.statusText;
}

function manualCapture() {
  if (!webcamRunning) return;
  cancelCountdown();
  state.predictionWindow = [];
  state.lastCommittedLabel = null;
  capturePhoto();
}

// ── Photo Manager Modal ─────────────────────────────────────
function openPhotoModal() {
  ui.photoModal.classList.remove("hidden");
  renderModalGrid();
  ui.modalClose.focus();
}

function closePhotoModal() {
  ui.photoModal.classList.add("hidden");
  ui.btnManagePhotos.focus();
}

function renderModalGrid() {
  ui.modalGrid.innerHTML = "";
  if (capturedPhotos.length === 0) {
    ui.modalGrid.innerHTML = '<div class="modal-empty">No photos captured yet.</div>';
    updateModalButtons();
    return;
  }
  for (const photo of capturedPhotos) {
    const thumb = document.createElement("button");
    thumb.type = "button";
    thumb.className = "modal-thumb";
    thumb.dataset.id = photo.id;
    thumb.setAttribute("aria-pressed", "false");
    const img = document.createElement("img");
    img.src = photo.dataUrl;
    img.alt = photo.name;
    const check = document.createElement("span");
    check.className = "thumb-check";
    check.textContent = "✓";
    const label = document.createElement("span");
    label.className = "thumb-label";
    label.textContent = photo.timestamp;
    thumb.append(img, check, label);
    thumb.addEventListener("click", () => {
      const selected = thumb.classList.toggle("selected");
      thumb.setAttribute("aria-pressed", String(selected));
      updateModalButtons();
    });
    ui.modalGrid.appendChild(thumb);
  }
  updateModalButtons();
}

function updateModalButtons() {
  const count = ui.modalGrid.querySelectorAll(".modal-thumb.selected").length;
  ui.modalCount.textContent = `${count} selected`;
  ui.modalDownloadZip.disabled = count === 0;
  ui.modalDeleteSelected.disabled = count === 0;
}

function getSelectedIds() {
  return [...ui.modalGrid.querySelectorAll(".modal-thumb.selected")].map((el) => Number(el.dataset.id));
}

async function downloadSelectedAsZip() {
  const selectedIds = new Set(getSelectedIds());
  if (selectedIds.size === 0) return;
  if (typeof JSZip === "undefined") {
    alert("ZIP library failed to load. Check your connection and try again.");
    return;
  }
  const zip = new JSZip();
  for (const photo of capturedPhotos) {
    if (selectedIds.has(photo.id)) zip.file(photo.name, photo.dataUrl.split(",")[1], { base64: true });
  }
  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `finger_detect_captures_${Date.now()}.zip`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function deleteSelected() {
  const selectedIds = new Set(getSelectedIds());
  if (selectedIds.size === 0) return;
  for (let i = capturedPhotos.length - 1; i >= 0; i--) {
    if (selectedIds.has(capturedPhotos[i].id)) capturedPhotos.splice(i, 1);
  }
  state.captureCount = capturedPhotos.length;
  ui.captures.textContent = state.captureCount;
  refreshGalleryThumbnails();
  renderModalGrid();
}

// ── Metrics, charts & benchmarks (from web/metrics.json) ────
const FALLBACK_METRICS = {
  accuracy: 97.72, test_samples: 3550, macro_f1: 97.77, model_parameters: 18726,
  mean_latency_ms: 0.0366, keras_predict_latency_ms: 93.87,
  per_source: { archive: { samples: 3444, accuracy: 97.97 }, webcam: { samples: 106, accuracy: 89.62 } },
  f1_scores: [0.9916, 0.9820, 0.9599, 0.9600, 0.9766, 0.9960],
  f1_labels: ["0 (Fist)", "1 (Index)", "2 (Peace)", "3 (Three)", "4 (Four)", "5 (Open Hand)"],
  classes: [],
  confusion_matrix: null,
  benchmarks: null,
};

const fmt = {
  ms: (v) => (v >= 10 ? `${v.toFixed(0)} ms` : v >= 1 ? `${v.toFixed(1)} ms` : `${v.toFixed(3)} ms`),
  pct: (v, d = 1) => `${Number(v).toFixed(d)}%`,
  int: (v) => Number(v).toLocaleString(),
};

function setTxt(id, v) {
  const el = document.getElementById(id);
  if (el && v !== undefined && v !== null) el.textContent = v;
}

function baseChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 700 },
    plugins: {
      legend: { labels: { color: INK_2, boxWidth: 12, boxHeight: 12, useBorderRadius: true, borderRadius: 3 } },
      tooltip: { backgroundColor: "rgba(15,20,35,0.95)", borderColor: "rgba(255,255,255,0.1)", borderWidth: 1, padding: 10 },
    },
    scales: {
      x: { ticks: { color: INK_2, font: { size: 11 } }, grid: { display: false } },
      y: { ticks: { color: INK_2 }, grid: { color: GRID } },
    },
  };
}

function renderHeadlineMetrics(m) {
  const b = m.benchmarks;
  setTxt("hero-accuracy", fmt.pct(m.accuracy));
  setTxt("metric-accuracy", fmt.pct(m.accuracy));
  document.getElementById("metric-accuracy-bar")?.style.setProperty("--fill", `${m.accuracy}%`);
  setTxt("metric-accuracy-detail",
    `${fmt.int(m.test_samples)} held-out samples${m.macro_f1 ? ` · macro F1 ${fmt.pct(m.macro_f1)}` : ""}`);
  setTxt("metric-params", m.model_parameters ? `${(m.model_parameters / 1000).toFixed(1)}k` : null);

  if (m.mean_latency_ms) {
    setTxt("metric-latency", fmt.ms(m.mean_latency_ms));
    if (m.keras_predict_latency_ms) {
      setTxt("metric-latency-detail", `NumPy engine vs ${fmt.ms(m.keras_predict_latency_ms)} keras.predict()`);
    }
  }
  if (b?.pipeline) {
    setTxt("hero-latency", fmt.ms(b.pipeline.total_ms));
    setTxt("metric-pipeline", fmt.ms(b.pipeline.total_ms));
    setTxt("metric-pipeline-detail", `Detection + classification · ${b.pipeline.max_fps} FPS ceiling`);
  }
  if (b?.speedup) {
    setTxt("hero-speedup", `${fmt.int(b.speedup)}×`);
    setTxt("bench-speedup-badge", `${fmt.int(b.speedup)}× faster`);
  }
  if (b?.processor) setTxt("bench-machine", `Last run: ${b.processor.split(",")[0]}, ${new Date(b.generated_at).toLocaleDateString()}.`);
  if (m.generated_at) setTxt("footer-generated", `Metrics generated ${new Date(m.generated_at).toLocaleString()}`);
}

function renderBenchmarks(b) {
  if (!b) {
    document.getElementById("benchmarks")?.classList.add("no-data");
    return;
  }

  // Latency (log scale; single series, direct labels via tooltip + axis)
  const lat = b.classifier_latency_ms;
  const latEl = document.getElementById("latency-chart");
  if (latEl && lat) {
    const opts = baseChartOptions();
    opts.indexAxis = "y";
    opts.plugins.legend.display = false;
    opts.plugins.tooltip.callbacks = { label: (c) => ` ${fmt.ms(c.raw)} per prediction` };
    opts.scales = {
      x: {
        type: "logarithmic",
        ticks: { color: INK_2, maxRotation: 0, autoSkip: false, callback: (v) => ([0.01, 0.1, 1, 10, 100].includes(v) ? `${v} ms` : "") },
        grid: { color: GRID },
      },
      y: {
        ticks: { color: "#e2e8f0", font: { size: 12 }, padding: 8 },
        grid: { display: false },
        afterFit: (axis) => { axis.width = Math.max(axis.width, window.innerWidth < 600 ? 150 : 210); },
      },
    };
    new Chart(latEl, {
      type: "bar",
      data: {
        labels: [`keras.predict()  ·  ${fmt.ms(lat.keras_predict)}`, `keras model(x)  ·  ${fmt.ms(lat.keras_call)}`, `NumPy engine  ·  ${fmt.ms(lat.numpy)}`],
        datasets: [{ data: [lat.keras_predict, lat.keras_call, lat.numpy], backgroundColor: [C_BLUE, C_BLUE, "#00d4ff"], borderRadius: 4, barThickness: 22 }],
      },
      options: opts,
    });
  }

  // Lighting robustness (2 series, grouped bars)
  const lightEl = document.getElementById("lighting-chart");
  if (lightEl && b.lighting) {
    setTxt("lighting-note", b.lighting_note);
    const opts = baseChartOptions();
    opts.scales.y.min = 0;
    opts.scales.y.max = 100;
    opts.scales.y.ticks.callback = (v) => `${v}%`;
    opts.plugins.tooltip.callbacks = { label: (c) => ` ${c.dataset.label}: ${c.raw}%` };
    new Chart(lightEl, {
      type: "bar",
      data: {
        labels: b.lighting.map((r) => r.condition),
        datasets: [
          { label: "Hand detected", data: b.lighting.map((r) => r.detection_rate), backgroundColor: C_BLUE, borderRadius: 4 },
          { label: "Correct count (end-to-end)", data: b.lighting.map((r) => r.accuracy), backgroundColor: C_ORANGE, borderRadius: 4 },
        ],
      },
      options: opts,
    });
    const table = document.getElementById("lighting-table");
    if (table) {
      table.innerHTML = "<thead><tr><th>Condition</th><th>Hand detected</th><th>End-to-end accuracy</th></tr></thead><tbody>" +
        b.lighting.map((r) => `<tr><td>${r.condition}</td><td>${r.detection_rate}%</td><td>${r.accuracy}%</td></tr>`).join("") + "</tbody>";
    }
  }

  // Temporal stability (2 lines)
  const tempEl = document.getElementById("temporal-chart");
  if (tempEl && b.temporal) {
    const opts = baseChartOptions();
    opts.interaction = { mode: "index", intersect: false };
    opts.scales.x.title = { display: true, text: "Landmark jitter (px at 640 px width)", color: INK_2 };
    opts.scales.y.title = { display: true, text: "Wrong-gesture rate", color: INK_2 };
    opts.scales.y.min = 0;
    opts.scales.y.ticks.callback = (v) => `${v}%`;
    opts.plugins.tooltip.callbacks = { title: (c) => `${c[0].label} px jitter`, label: (c) => ` ${c.dataset.label}: ${c.raw}%` };
    const line = (label, data, color) => ({
      label, data, borderColor: color, backgroundColor: color, borderWidth: 2,
      pointRadius: 4, pointHoverRadius: 6, pointBorderColor: "#0a0e1a", pointBorderWidth: 2, tension: 0.25,
    });
    new Chart(tempEl, {
      type: "line",
      data: {
        labels: b.temporal.map((r) => r.jitter_px),
        datasets: [
          line("Single frame", b.temporal.map((r) => r.single_frame_err), C_ORANGE),
          line("8-frame rolling vote", b.temporal.map((r) => r.voted_err), C_BLUE),
        ],
      },
      options: opts,
    });
    const table = document.getElementById("temporal-table");
    if (table) {
      table.innerHTML = "<thead><tr><th>Jitter (px)</th><th>Single frame</th><th>Rolling vote</th></tr></thead><tbody>" +
        b.temporal.map((r) => `<tr><td>${r.jitter_px}</td><td>${r.single_frame_err}%</td><td>${r.voted_err}%</td></tr>`).join("") + "</tbody>";
    }
  }

  // Pipeline time breakdown
  const p = b.pipeline;
  const bar = document.getElementById("stack-bar");
  const legend = document.getElementById("stack-legend");
  if (p && bar && legend) {
    const other = Math.max(p.total_ms - p.detect_ms - p.classify_ms, 0);
    const parts = [
      { name: "Hand detection (MediaPipe)", ms: p.detect_ms, color: C_BLUE },
      { name: "Feature extraction", ms: other, color: "#64748b" },
      { name: "MLP classification", ms: p.classify_ms, color: "#00d4ff" },
    ];
    bar.innerHTML = parts.map((x) =>
      `<span style="flex-grow:${Math.max(x.ms, p.total_ms * 0.006)};background:${x.color}" title="${x.name}: ${fmt.ms(x.ms)}"></span>`).join("");
    legend.innerHTML = parts.map((x) =>
      `<li><span class="swatch" style="background:${x.color}"></span>${x.name} <strong>${fmt.ms(x.ms)}</strong></li>`).join("") +
      `<li class="stack-total">Total <strong>${fmt.ms(p.total_ms)}</strong> → ${p.max_fps} FPS ceiling</li>`;
  } else {
    document.getElementById("pipeline-breakdown")?.classList.add("hidden");
  }
}

function renderResults(m) {
  const f1El = document.getElementById("f1-chart");
  if (f1El && m.f1_scores) {
    const opts = baseChartOptions();
    opts.plugins.legend.display = false;
    opts.plugins.tooltip.callbacks = { label: (c) => ` F1: ${(c.raw * 100).toFixed(2)}%` };
    opts.scales.y.min = Math.max(0, Math.floor((Math.min(...m.f1_scores) - 0.03) * 100) / 100);
    opts.scales.y.max = 1;
    opts.scales.y.ticks.callback = (v) => `${(v * 100).toFixed(0)}%`;
    new Chart(f1El, {
      type: "bar",
      data: { labels: m.f1_labels, datasets: [{ label: "F1", data: m.f1_scores, backgroundColor: C_BLUE, borderRadius: 4, maxBarThickness: 48 }] },
      options: opts,
    });
  }

  const tbody = document.getElementById("results-tbody");
  if (tbody && m.classes?.length) {
    const p = (v) => `${(v * 100).toFixed(2)}%`;
    tbody.innerHTML = m.classes.map((c) => `
      <tr><td>${c.label}</td><td>${c.name}</td><td>${p(c.precision)}</td><td>${p(c.recall)}</td>
      <td><strong>${p(c.f1)}</strong></td><td>${fmt.int(c.support)}</td></tr>`).join("") +
      (m.macro_f1 ? `<tr class="row-total"><td colspan="2"><strong>Macro avg</strong></td>
        <td>${fmt.pct(m.macro_precision, 2)}</td><td>${fmt.pct(m.macro_recall, 2)}</td>
        <td>${fmt.pct(m.macro_f1, 2)}</td><td>${fmt.int(m.test_samples)}</td></tr>` : "");
  }

  // Confusion matrix (row-normalised colour, raw counts as text)
  const cm = m.confusion_matrix;
  const cmEl = document.getElementById("confusion-matrix");
  if (cmEl && cm) {
    let html = '<div class="cm-header"></div>';
    for (let c = 0; c < cm.length; c++) html += `<div class="cm-header">${c}</div>`;
    for (let r = 0; r < cm.length; r++) {
      const rowTotal = cm[r].reduce((a, b) => a + b, 0) || 1;
      html += `<div class="cm-header">${r}</div>`;
      for (let c = 0; c < cm.length; c++) {
        const val = cm[r][c];
        const frac = val / rowTotal;
        const bg = r === c
          ? `rgba(57,135,229,${(0.2 + frac * 0.75).toFixed(2)})`
          : val > 0 ? `rgba(217,89,38,${Math.min(0.15 + frac * 6, 0.9).toFixed(2)})` : "rgba(255,255,255,0.025)";
        html += `<div class="cm-cell${val === 0 ? " zero" : ""}" style="background:${bg}"
          title="True ${r} (${CLASS_NAMES[r]}) → predicted ${c} (${CLASS_NAMES[c]}): ${val} (${(frac * 100).toFixed(1)}% of row)">${val}</div>`;
      }
    }
    cmEl.innerHTML = html;
  } else if (cmEl) {
    cmEl.innerHTML = '<p class="card-sub">Run <code>python tools/evaluate_model.py</code> and <code>tools/sync_web_metrics.py</code> to populate.</p>';
  }

  // Accuracy per data source
  const domain = document.getElementById("domain-bars");
  const src = m.per_source || {};
  if (domain && Object.keys(src).length) {
    const label = { archive: "Public dataset (archive)", webcam: "Own webcam captures" };
    domain.innerHTML = Object.entries(src).map(([k, v]) => `
      <div class="domain-row">
        <span class="domain-label">${label[k] || k} <small>${fmt.int(v.samples)} samples</small></span>
        <span class="domain-track"><span class="domain-fill ${k}" style="width:${v.accuracy}%"></span></span>
        <strong class="domain-value">${fmt.pct(v.accuracy)}</strong>
      </div>`).join("");
  } else {
    document.getElementById("domain-note")?.classList.add("hidden");
  }
}

async function loadMetricsAndInitCharts() {
  let m = FALLBACK_METRICS;
  try {
    const resp = await fetch(`metrics.json?v=${Date.now()}`);
    if (resp.ok) m = { ...FALLBACK_METRICS, ...(await resp.json()) };
  } catch (_) {
    console.warn("[metrics] metrics.json not found, using built-in defaults.");
  }
  if (typeof Chart !== "undefined") {
    Chart.defaults.font.family = "Inter, system-ui, sans-serif";
    Chart.defaults.color = INK_2;
  }
  renderHeadlineMetrics(m);
  if (typeof Chart === "undefined") {
    console.warn("[charts] Chart.js failed to load; skipping charts.");
    return;
  }
  renderBenchmarks(m.benchmarks);
  renderResults(m);
}

// ── Scroll animations ───────────────────────────────────────
function initScrollAnimations() {
  const observer = new IntersectionObserver(
    (entries) => entries.forEach((entry) => entry.isIntersecting && entry.target.classList.add("visible")),
    { threshold: 0.15 },
  );
  document.querySelectorAll("[data-animate]").forEach((el) => observer.observe(el));
  const nav = document.getElementById("main-nav");
  window.addEventListener("scroll", () => nav.classList.toggle("scrolled", window.scrollY > 60), { passive: true });
}

// ── Keyboard shortcuts ──────────────────────────────────────
function initKeyboard() {
  document.addEventListener("keydown", (e) => {
    if (!ui.photoModal.classList.contains("hidden")) {
      if (e.key === "Escape") closePhotoModal();
      return;
    }
    if (!webcamRunning) return;
    if (e.target.closest?.("input, textarea, select")) return;
    if (e.code === "Space" || e.key === "s") {
      e.preventDefault();
      manualCapture();
    } else if (e.key === "Escape" || e.key === "x") {
      cancelCountdown();
      state.predictionWindow = [];
      state.lastCommittedLabel = null;
      state.statusText = "Countdown cancelled";
    }
  });
}

// ── Event listeners ─────────────────────────────────────────
ui.btnStart.addEventListener("click", startCamera);
ui.btnCapture.addEventListener("click", manualCapture);
ui.btnStopCamera.addEventListener("click", stopCamera);
ui.btnManagePhotos.addEventListener("click", openPhotoModal);
ui.toggleSkeleton.addEventListener("change", (e) => {
  state.showSkeleton = e.target.checked;
  if (!state.showSkeleton) ctx.clearRect(0, 0, canvas.width, canvas.height);
});

ui.modalClose.addEventListener("click", closePhotoModal);
ui.photoModal.addEventListener("click", (e) => { if (e.target === ui.photoModal) closePhotoModal(); });
ui.modalSelectAll.addEventListener("click", () => {
  ui.modalGrid.querySelectorAll(".modal-thumb").forEach((t) => { t.classList.add("selected"); t.setAttribute("aria-pressed", "true"); });
  updateModalButtons();
});
ui.modalDeselectAll.addEventListener("click", () => {
  ui.modalGrid.querySelectorAll(".modal-thumb").forEach((t) => { t.classList.remove("selected"); t.setAttribute("aria-pressed", "false"); });
  updateModalButtons();
});
ui.modalDownloadZip.addEventListener("click", downloadSelectedAsZip);
ui.modalDeleteSelected.addEventListener("click", deleteSelected);

// Release the camera when the tab is hidden for good
window.addEventListener("pagehide", () => { if (webcamRunning) stopCamera(); });

// ── Init ────────────────────────────────────────────────────
loadModel();
loadMetricsAndInitCharts();
initScrollAnimations();
initKeyboard();
