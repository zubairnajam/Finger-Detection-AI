/* ════════════════════════════════════════════════════════════
   Finger Detection Smart Camera — Web App
   Uses MediaPipe Hand Landmarker (JS) + geometric finger counting
   ════════════════════════════════════════════════════════════ */

import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// ── Constants ───────────────────────────────────────────────
const WINDOW_SIZE = 8;
const MIN_VOTES = 4;
const MIN_CONFIDENCE = 0.55;
const CIRCUMFERENCE = 2 * Math.PI * 90; // countdown ring

// MediaPipe hand connections for drawing
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

// ── State ───────────────────────────────────────────────────
let handLandmarker = null;
let webcamRunning = false;
let lastVideoTime = -1;
let animationFrameId = null;

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
  btnCancel: $("#btn-cancel"),
  btnReset: $("#btn-reset"),
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
  // Confidence based on how "clearly" fingers are up or down
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
  // Draw current frame to an offscreen canvas
  const offscreen = document.createElement("canvas");
  offscreen.width = video.videoWidth;
  offscreen.height = video.videoHeight;
  const offCtx = offscreen.getContext("2d");
  offCtx.translate(offscreen.width, 0);
  offCtx.scale(-1, 1); // mirror
  offCtx.drawImage(video, 0, 0);
  const dataUrl = offscreen.toDataURL("image/jpeg", 0.92);

  // Flash
  ui.flashOverlay.classList.remove("hidden");
  setTimeout(() => ui.flashOverlay.classList.add("hidden"), 300);

  state.captureCount++;
  ui.captures.textContent = state.captureCount;
  state.statusText = `Photo #${state.captureCount} captured!`;

  // Gallery thumbnail
  const img = document.createElement("img");
  img.src = dataUrl;
  img.title = `Capture #${state.captureCount}`;
  img.addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `capture_${state.captureCount}.jpg`;
    a.click();
  });
  ui.galleryGrid.prepend(img);
  // Keep last 6
  while (ui.galleryGrid.children.length > 6) ui.galleryGrid.lastChild.remove();
}

// ── Drawing ─────────────────────────────────────────────────
function drawHandLandmarks(landmarks) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Connections
  ctx.strokeStyle = "rgba(0, 212, 255, 0.5)";
  ctx.lineWidth = 2;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }

  // Landmarks
  for (let i = 0; i < landmarks.length; i++) {
    const x = landmarks[i].x * w;
    const y = landmarks[i].y * h;
    const isTip = [4, 8, 12, 16, 20].includes(i);
    const radius = isTip ? 5 : 3;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = isTip ? "#00ff88" : "#00d4ff";
    ctx.fill();
    if (isTip) {
      ctx.strokeStyle = "rgba(0, 255, 136, 0.3)";
      ctx.lineWidth = 4;
      ctx.stroke();
    }
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
    // Colour shift
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
    const handedness = results.handednesses?.[0]?.[0]?.categoryName || "Right";
    fingerResult = countFingers(lm, handedness);
    rawConf = getFingerConfidence(lm);
    addPrediction(fingerResult.count, rawConf);
    drawHandLandmarks(lm);
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

// ── Camera start ────────────────────────────────────────────
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
    ui.btnCancel.disabled = false;
    ui.btnReset.disabled = false;

    // Set countdown ring dasharray
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

// ── Charts (results section) ────────────────────────────────
function initCharts() {
  // F1 bar chart
  const f1Ctx = document.getElementById("f1-chart");
  if (f1Ctx) {
    new Chart(f1Ctx, {
      type: "bar",
      data: {
        labels: ["0 (Fist)", "1 (Index)", "2 (Peace)", "3 (Three)", "4 (Four)", "5 (Open)"],
        datasets: [{
          label: "F1 Score",
          data: [0.9927, 0.9824, 0.9573, 0.9573, 0.9737, 0.9943],
          backgroundColor: [
            "rgba(0, 212, 255, 0.7)", "rgba(0, 212, 255, 0.65)",
            "rgba(0, 212, 255, 0.55)", "rgba(0, 212, 255, 0.55)",
            "rgba(0, 212, 255, 0.62)", "rgba(0, 212, 255, 0.72)",
          ],
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
            min: 0.94, max: 1.002,
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

  // Confusion matrix
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
    // Header row
    cmEl.innerHTML = '<div class="cm-header"></div>';
    for (let c = 0; c < 6; c++) cmEl.innerHTML += `<div class="cm-header">${c}</div>`;
    // Data rows
    const maxVal = 612;
    for (let r = 0; r < 6; r++) {
      cmEl.innerHTML += `<div class="cm-header">${r}</div>`;
      for (let c = 0; c < 6; c++) {
        const val = cmData[r][c];
        const intensity = r === c ? Math.round((val / maxVal) * 200 + 55) : Math.min(Math.round((val / maxVal) * 180), 180);
        const bg = r === c
          ? `rgba(0, 212, 255, ${(val / maxVal * 0.7 + 0.15).toFixed(2)})`
          : val > 0
            ? `rgba(255, 68, 102, ${(val / maxVal * 0.6).toFixed(2)})`
            : "rgba(255,255,255,0.02)";
        cmEl.innerHTML += `<div class="cm-cell" style="background:${bg}" title="True ${r} → Pred ${c}: ${val}">${val}</div>`;
      }
    }
  }
}

// ── Scroll animations ───────────────────────────────────────
function initScrollAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
        }
      });
    },
    { threshold: 0.15 }
  );
  document.querySelectorAll("[data-animate]").forEach((el) => observer.observe(el));

  // Nav background on scroll
  const nav = document.getElementById("main-nav");
  window.addEventListener("scroll", () => {
    nav.classList.toggle("scrolled", window.scrollY > 60);
  });
}

// ── Keyboard shortcuts ──────────────────────────────────────
function initKeyboard() {
  document.addEventListener("keydown", (e) => {
    if (!webcamRunning) return;
    if (e.code === "Space" || e.key === "s") {
      e.preventDefault();
      cancelCountdown();
      state.predictionWindow = [];
      state.lastCommittedLabel = null;
      capturePhoto();
    }
    if (e.key === "x") { cancelCountdown(); state.predictionWindow = []; state.lastCommittedLabel = null; state.statusText = "Cancelled"; }
    if (e.key === "r") { cancelCountdown(); state.predictionWindow = []; state.lastCommittedLabel = null; state.statusText = "Reset"; }
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
ui.btnCancel.addEventListener("click", () => {
  cancelCountdown();
  state.predictionWindow = [];
  state.lastCommittedLabel = null;
  state.statusText = "Cancelled";
});
ui.btnReset.addEventListener("click", () => {
  cancelCountdown();
  state.predictionWindow = [];
  state.lastCommittedLabel = null;
  state.statusText = "Reset";
});

// ── Init ────────────────────────────────────────────────────
initCharts();
initScrollAnimations();
initKeyboard();
