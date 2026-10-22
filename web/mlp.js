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

  /** @param {ArrayLike<number>} features raw (unscaled) 63-value landmark vector */
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

  predict(features) {
    const probs = this.predictProba(features);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { label: this.labels[best], confidence: probs[best], probs };
  }
}
