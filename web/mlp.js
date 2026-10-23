/* ════════════════════════════════════════════════════════════
   Tiny dependency-free MLP inference engine
   Loads weights exported by tools/export_web_model.py and runs
   the forward pass in plain JS — no TF.js runtime download.
   ════════════════════════════════════════════════════════════ */

const ACTIVATIONS = {
  relu: (v) => { for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0; return v; },
  linear: (v) => v,
  softmax: (v) => {
    let max = -Infinity;
    for (const x of v) if (x > max) max = x;
    let sum = 0;
    for (let i = 0; i < v.length; i++) { v[i] = Math.exp(v[i] - max); sum += v[i]; }
    for (let i = 0; i < v.length; i++) v[i] /= sum;
    return v;
  },
};

/* Mirrors src/features.py build_features() exactly (checked by tests/test_web_engine.py).
   Makes the input independent of where the hand is in the frame and how big it is. */
const FINGER_CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];
const FINGERTIPS = [4, 8, 12, 16, 20];
const EPS = 1e-6;

/**
 * @param {ArrayLike<number>} raw  63 values: x, y, z for each of the 21 landmarks
 * @param {number} aspect  source frame width / height
 * @returns {Float32Array} 85 model-ready (unscaled) features
 */
export function buildFeatures(raw, aspect = 1) {
  const p = new Float32Array(63);
  for (let i = 0; i < 21; i++) {
    p[i * 3] = raw[i * 3] * aspect;
    p[i * 3 + 1] = raw[i * 3 + 1];
    p[i * 3 + 2] = raw[i * 3 + 2];
  }
  const wx = p[0], wy = p[1], wz = p[2];
  for (let i = 0; i < 21; i++) { p[i * 3] -= wx; p[i * 3 + 1] -= wy; p[i * 3 + 2] -= wz; }
  const scale = Math.max(Math.hypot(p[27], p[28]), EPS); // wrist -> middle knuckle (landmark 9)
  for (let i = 0; i < 63; i++) p[i] /= scale;

  const dist = (a, b) => Math.hypot(p[a * 3] - p[b * 3], p[a * 3 + 1] - p[b * 3 + 1], p[a * 3 + 2] - p[b * 3 + 2]);
  const out = new Float32Array(85);
  out.set(p);
  let k = 63;
  for (const chain of FINGER_CHAINS) {
    for (let j = 0; j + 2 < chain.length; j++) {
      const [a, b, c] = [chain[j] * 3, chain[j + 1] * 3, chain[j + 2] * 3];
      const ux = p[a] - p[b], uy = p[a + 1] - p[b + 1], uz = p[a + 2] - p[b + 2];
      const vx = p[c] - p[b], vy = p[c + 1] - p[b + 1], vz = p[c + 2] - p[b + 2];
      out[k++] = (ux * vx + uy * vy + uz * vz) / (Math.hypot(ux, uy, uz) * Math.hypot(vx, vy, vz) + EPS);
    }
  }
  for (const tip of FINGERTIPS) out[k++] = Math.hypot(p[tip * 3], p[tip * 3 + 1], p[tip * 3 + 2]);
  out[k++] = dist(4, 5);
  out[k++] = dist(4, 17);
  return out;
}

export class MLP {
  /**
   * @param {object} manifest  parsed mlp.json
   * @param {ArrayBuffer} buffer  contents of mlp.bin (little-endian float32)
   */
  constructor(manifest, buffer) {
    const floats = new Float32Array(buffer);
    if (floats.length !== manifest.weights_float32_count) {
      throw new Error(`Weight file has ${floats.length} floats, expected ${manifest.weights_float32_count}`);
    }
    let offset = 0;
    this.layers = manifest.layers.map(({ inputs, units, activation }) => {
      if (!ACTIVATIONS[activation]) throw new Error(`Unsupported activation: ${activation}`);
      const kernel = floats.subarray(offset, (offset += inputs * units)); // row-major [inputs][units]
      const bias = floats.subarray(offset, (offset += units));
      return { inputs, units, kernel, bias, activation };
    });
    this.mean = Float32Array.from(manifest.scaler.mean);
    this.std = Float32Array.from(manifest.scaler.std, (s) => (s === 0 ? 1 : s));
    this.labels = manifest.labels;
    this.classNames = manifest.class_names;
    this.parameters = manifest.parameters;
  }

  static async load(url = "model/mlp.json") {
    const manifestResp = await fetch(url);
    if (!manifestResp.ok) throw new Error(`HTTP ${manifestResp.status} for ${url}`);
    const manifest = await manifestResp.json();
    const binUrl = new URL(manifest.weights_file, new URL(url, location.href)).href;
    const binResp = await fetch(binUrl);
    if (!binResp.ok) throw new Error(`HTTP ${binResp.status} for ${binUrl}`);
    return new MLP(manifest, await binResp.arrayBuffer());
  }

  /** @param {ArrayLike<number>} features output of buildFeatures() (unscaled) */
  predictProba(features) {
    let h = new Float32Array(features.length);
    for (let i = 0; i < h.length; i++) h[i] = (features[i] - this.mean[i]) / this.std[i];

    for (const { inputs, units, kernel, bias, activation } of this.layers) {
      const out = Float32Array.from(bias);
      for (let i = 0; i < inputs; i++) {
        const x = h[i];
        if (x === 0) continue;
        const row = i * units;
        for (let j = 0; j < units; j++) out[j] += x * kernel[row + j];
      }
      h = ACTIVATIONS[activation](out);
    }
    return h;
  }

  /** @param {ArrayLike<number>} landmarks raw 63-value landmark vector; @param {number} aspect frame width / height */
  predict(landmarks, aspect = 1) {
    const probs = this.predictProba(buildFeatures(landmarks, aspect));
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { label: this.labels[best], confidence: probs[best], probs };
  }
}
