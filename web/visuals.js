/* ════════════════════════════════════════════════════════════
   Decorative visuals: interactive dot field, cursor ring and the
   hero hand. Independent of the camera app (app.js).
   ════════════════════════════════════════════════════════════ */

const ACCENT = [255, 107, 61];
const INK = [237, 237, 237];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const finePointer = window.matchMedia("(pointer: fine)").matches;

const pointer = { x: -9999, y: -9999, sx: -9999, sy: -9999, active: false };
window.addEventListener("pointermove", (e) => {
  pointer.x = e.clientX;
  pointer.y = e.clientY;
  if (!pointer.active) { pointer.sx = e.clientX; pointer.sy = e.clientY; pointer.active = true; }
  wake();
}, { passive: true });
document.addEventListener("pointerleave", () => { pointer.active = false; wake(); });
window.addEventListener("scroll", () => wake(), { passive: true });

const mix = (a, b, t) => a + (b - a) * t;
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

// ── Background dot field ────────────────────────────────────
// Dots sit on a grid; near the cursor they are pushed outward and lit,
// like landmarks being picked up by a tracker.
const field = document.getElementById("bg-field");
const fctx = field.getContext("2d");
const SPACING = 34;
const RADIUS = 190;
let fw = 0, fh = 0, dpr = 1;

function resizeField() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  fw = window.innerWidth;
  fh = window.innerHeight;
  field.width = fw * dpr;
  field.height = fh * dpr;
  fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  wake();
}

function drawField() {
  fctx.clearRect(0, 0, fw, fh);
  const px = pointer.sx, py = pointer.sy;
  const on = pointer.active && finePointer && !reducedMotion;

  if (on) {
    const glow = fctx.createRadialGradient(px, py, 0, px, py, RADIUS * 1.6);
    glow.addColorStop(0, rgba(ACCENT, 0.07));
    glow.addColorStop(1, rgba(ACCENT, 0));
    fctx.fillStyle = glow;
    fctx.fillRect(px - RADIUS * 2, py - RADIUS * 2, RADIUS * 4, RADIUS * 4);
  }

  const offset = -((window.scrollY * 0.35) % SPACING);
  for (let y = offset; y < fh + SPACING; y += SPACING) {
    for (let x = SPACING / 2; x < fw; x += SPACING) {
      let dx = 0, dy = 0, t = 0;
      if (on) {
        const vx = x - px, vy = y - py;
        const d = Math.hypot(vx, vy);
        if (d < RADIUS) {
          t = 1 - d / RADIUS;
          t *= t;
          const push = t * 14 / (d || 1);
          dx = vx * push;
          dy = vy * push;
        }
      }
      const c = t > 0 ? [mix(INK[0], ACCENT[0], t), mix(INK[1], ACCENT[1], t), mix(INK[2], ACCENT[2], t)] : INK;
      fctx.fillStyle = rgba(c, 0.09 + t * 0.6);
      fctx.beginPath();
      fctx.arc(x + dx, y + dy, 0.9 + t * 1.3, 0, Math.PI * 2);
      fctx.fill();
    }
  }
}

// ── Cursor ring ─────────────────────────────────────────────
const ring = document.getElementById("cursor-ring");
const ringPos = { x: -100, y: -100 };
if (finePointer && !reducedMotion) {
  document.documentElement.classList.add("has-cursor-ring");
  document.addEventListener("pointerover", (e) => {
    ring.classList.toggle("is-hover", !!e.target.closest("a, button, label, summary, [role='button']"));
  });
  document.addEventListener("pointerdown", () => ring.classList.add("is-down"));
  document.addEventListener("pointerup", () => ring.classList.remove("is-down"));
}

// ── Hero hand ───────────────────────────────────────────────
// One representative webcam sample per class from the training set, in
// wrist-relative units (wrist -> middle knuckle = 1), as in src/features.py.
const POSES = [[[0.0,0.0],[-0.328,-0.3],[-0.504,-0.724],[-0.27,-0.999],[0.01,-0.917],[-0.249,-1.08],[-0.162,-1.337],[-0.248,-1.03],[-0.257,-0.87],[0.016,-1.0],[0.097,-1.24],[-0.002,-0.905],[-0.011,-0.785],[0.256,-0.89],[0.321,-1.069],[0.186,-0.778],[0.143,-0.662],[0.48,-0.735],[0.513,-0.909],[0.387,-0.722],[0.334,-0.612]],[[-0.0,0.0],[-0.254,-0.247],[-0.423,-0.654],[-0.396,-1.005],[-0.147,-1.132],[-0.235,-1.071],[-0.291,-1.514],[-0.327,-1.791],[-0.312,-2.034],[0.005,-1.0],[-0.011,-1.276],[-0.114,-0.898],[-0.131,-0.709],[0.222,-0.867],[0.205,-1.039],[0.055,-0.679],[0.036,-0.523],[0.442,-0.689],[0.415,-0.859],[0.249,-0.62],[0.21,-0.484]],[[0.0,0.0],[-0.253,-0.253],[-0.368,-0.557],[-0.238,-0.823],[-0.018,-0.866],[-0.299,-0.999],[-0.402,-1.387],[-0.449,-1.615],[-0.477,-1.811],[-0.068,-0.998],[-0.021,-1.472],[0.001,-1.767],[0.036,-2.014],[0.143,-0.873],[0.168,-1.143],[0.022,-0.889],[-0.066,-0.685],[0.31,-0.671],[0.286,-0.871],[0.142,-0.711],[0.041,-0.549]],[[-0.0,0.0],[-0.227,-0.161],[-0.302,-0.478],[-0.056,-0.713],[0.201,-0.838],[-0.315,-0.934],[-0.452,-1.304],[-0.519,-1.519],[-0.56,-1.721],[-0.103,-0.995],[-0.206,-1.421],[-0.255,-1.664],[-0.292,-1.877],[0.101,-0.943],[0.164,-1.336],[0.191,-1.597],[0.214,-1.835],[0.281,-0.803],[0.366,-1.035],[0.31,-0.934],[0.248,-0.804]],[[0.0,0.0],[-0.251,-0.19],[-0.403,-0.522],[-0.223,-0.773],[0.024,-0.802],[-0.384,-0.925],[-0.536,-1.259],[-0.614,-1.464],[-0.664,-1.637],[-0.162,-0.987],[-0.188,-1.413],[-0.202,-1.676],[-0.204,-1.885],[0.044,-0.926],[0.137,-1.315],[0.199,-1.563],[0.253,-1.77],[0.229,-0.778],[0.423,-1.013],[0.562,-1.17],[0.684,-1.299]],[[0.0,0.0],[-0.353,-0.167],[-0.623,-0.486],[-0.802,-0.757],[-0.974,-0.945],[-0.335,-0.956],[-0.414,-1.313],[-0.456,-1.526],[-0.476,-1.718],[-0.11,-0.994],[-0.139,-1.4],[-0.152,-1.649],[-0.158,-1.858],[0.102,-0.942],[0.128,-1.328],[0.139,-1.559],[0.141,-1.766],[0.312,-0.824],[0.394,-1.113],[0.434,-1.307],[0.465,-1.496]]];
const NAMES = ["Fist", "Index", "Peace", "Three", "Four", "Open hand"];
const RAISED = [[], [8], [8, 12], [8, 12, 16], [8, 12, 16, 20], [4, 8, 12, 16, 20]];
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
const SEQUENCE = [1, 2, 3, 4, 5, 0];
const HOLD_MS = 1700, MORPH_MS = 650;

const hand = document.getElementById("hero-hand");
const hctx = hand?.getContext("2d");
const countEl = document.getElementById("hero-hand-count");
const nameEl = document.getElementById("hero-hand-name");
let hw = 0, hh = 0, heroVisible = true, heroStart = performance.now();
const tilt = { x: 0, y: 0 };

function resizeHand() {
  if (!hand) return;
  const r = hand.getBoundingClientRect();
  hw = r.width;
  hh = r.height;
  hand.width = hw * dpr;
  hand.height = hh * dpr;
  hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  wake();
}

const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function heroPose(now) {
  if (reducedMotion) return { pts: POSES[5], cls: 5, raisedT: 1 };
  const cycle = HOLD_MS + MORPH_MS;
  const elapsed = now - heroStart;
  const step = Math.floor(elapsed / cycle);
  const within = elapsed % cycle;
  const from = SEQUENCE[step % SEQUENCE.length];
  const to = SEQUENCE[(step + 1) % SEQUENCE.length];
  const t = within < HOLD_MS ? 0 : (within - HOLD_MS) / MORPH_MS;
  const pts = POSES[from].map((p, i) => {
    const finger = i === 0 ? 0 : Math.floor((i - 1) / 4);       // stagger fingers slightly
    const k = ease(Math.min(Math.max(t * 1.25 - finger * 0.06, 0), 1));
    return [mix(p[0], POSES[to][i][0], k), mix(p[1], POSES[to][i][1], k)];
  });
  return { pts, cls: t < 0.5 ? from : to, raisedT: Math.abs(t - 0.5) * 2 };
}

let shownClass = -1;
function drawHand(now) {
  if (!hctx || !hw) return;
  hctx.clearRect(0, 0, hw, hh);
  const { pts, cls, raisedT } = heroPose(now);

  if (cls !== shownClass) {
    shownClass = cls;
    countEl.textContent = cls;
    nameEl.textContent = NAMES[cls];
  }

  // Tilt toward the cursor
  const r = hand.getBoundingClientRect();
  const tx = pointer.active ? (pointer.sx - (r.left + hw / 2)) / window.innerWidth : 0;
  const ty = pointer.active ? (pointer.sy - (r.top + hh / 2)) / window.innerHeight : 0;
  tilt.x = mix(tilt.x, tx, 0.08);
  tilt.y = mix(tilt.y, ty, 0.08);

  const scale = Math.min(hh * 0.32, hw * 0.32);
  const ox = hw / 2 + tilt.x * 40;
  const oy = hh * 0.84 + tilt.y * 24;
  const rot = tilt.x * 0.35;
  const cos = Math.cos(rot), sin = Math.sin(rot);
  const P = pts.map(([x, y]) => [ox + (x * cos - y * sin) * scale, oy + (x * sin + y * cos) * scale]);

  // Detector box
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of P) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); }
  const pad = 18;
  minX -= pad; minY -= pad; maxX += pad; maxY += pad;
  const L = 14;
  hctx.strokeStyle = rgba(ACCENT, 0.85);
  hctx.lineWidth = 1.5;
  hctx.beginPath();
  for (const [cx, cy, sx, sy] of [[minX, minY, 1, 1], [maxX, minY, -1, 1], [minX, maxY, 1, -1], [maxX, maxY, -1, -1]]) {
    hctx.moveTo(cx, cy + sy * L); hctx.lineTo(cx, cy); hctx.lineTo(cx + sx * L, cy);
  }
  hctx.stroke();
  hctx.font = "500 11px 'Geist Mono', ui-monospace, monospace";
  hctx.fillStyle = rgba(ACCENT, 0.9);
  hctx.fillText(`hand · ${NAMES[cls].toLowerCase()}`, minX, Math.max(minY - 8, 52));

  // Bones
  hctx.lineCap = "round";
  hctx.strokeStyle = rgba(INK, 0.5);
  hctx.lineWidth = 1.5;
  hctx.beginPath();
  for (const [a, b] of BONES) { hctx.moveTo(...P[a]); hctx.lineTo(...P[b]); }
  hctx.stroke();

  // Joints; raised fingertips light up, nearby joints react to the cursor
  const raised = RAISED[cls];
  for (let i = 0; i < 21; i++) {
    const [x, y] = P[i];
    const near = pointer.active ? Math.max(0, 1 - Math.hypot(pointer.sx - r.left - x, pointer.sy - r.top - y) / 120) : 0;
    const isTip = raised.includes(i);
    if (isTip) {
      hctx.strokeStyle = rgba(ACCENT, 0.35 * raisedT);
      hctx.lineWidth = 1;
      hctx.beginPath();
      hctx.arc(x, y, 9 + raisedT * 3, 0, Math.PI * 2);
      hctx.stroke();
    }
    hctx.fillStyle = isTip ? rgba(ACCENT, 0.6 + 0.4 * raisedT) : rgba(INK, 0.75 + near * 0.25);
    hctx.beginPath();
    hctx.arc(x, y, (isTip ? 3.6 : 2.4) + near * 2.5, 0, Math.PI * 2);
    hctx.fill();
  }
}

// ── Single animation loop; sleeps when nothing is changing ──
let raf = 0, lastActivity = performance.now();
function wake() {
  lastActivity = performance.now();
  if (!raf) raf = requestAnimationFrame(frame);
}

function frame(now) {
  raf = 0;
  pointer.sx = mix(pointer.sx, pointer.x, 0.18);
  pointer.sy = mix(pointer.sy, pointer.y, 0.18);
  ringPos.x = mix(ringPos.x, pointer.x, 0.22);
  ringPos.y = mix(ringPos.y, pointer.y, 0.22);
  if (ring) ring.style.transform = `translate3d(${ringPos.x}px, ${ringPos.y}px, 0)`;

  drawField();
  if (heroVisible) drawHand(now);

  const settling = Math.abs(pointer.sx - pointer.x) + Math.abs(pointer.sy - pointer.y) > 0.5;
  const heroAnimating = heroVisible && !reducedMotion;
  if (settling || heroAnimating || now - lastActivity < 300) raf = requestAnimationFrame(frame);
}

if (hand) {
  new IntersectionObserver(([entry]) => { heroVisible = entry.isIntersecting; wake(); }).observe(hand);
  new ResizeObserver(resizeHand).observe(hand);
}
window.addEventListener("resize", resizeField);
resizeField();
resizeHand();
