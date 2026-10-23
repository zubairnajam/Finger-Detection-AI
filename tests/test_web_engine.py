"""Cross-language parity: the browser's JS MLP must match the Python engine."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.features import build_features
from src.prediction import load_model_and_scaler

NODE = shutil.which("node")

NODE_SCRIPT = r"""
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [modulePath, manifestPath, binPath, inputPath] = process.argv.slice(2);
const { MLP, buildFeatures } = await import(pathToFileURL(modulePath).href);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const buf = readFileSync(binPath);
const mlp = new MLP(manifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const { x, aspects } = JSON.parse(readFileSync(inputPath, "utf8"));
console.log(JSON.stringify({
  features: x.map((v, i) => Array.from(buildFeatures(v, aspects[i]))),
  probs: x.map((v, i) => Array.from(mlp.predict(v, aspects[i]).probs)),
}));
"""


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_js_engine_matches_python(tmp_path, keras_model):
    from tools.export_web_model import export

    model, mean, std = load_model_and_scaler(backend="numpy")
    json_path, bin_path = export(model, mean, std, out_dir=tmp_path)

    rng = np.random.default_rng(0)
    x = rng.uniform(-0.1, 1.1, size=(64, 63)).astype(np.float32)
    test_x = PROJECT_ROOT / "data" / "test" / "X_test.npy"
    if test_x.exists():
        x = np.vstack([x, np.load(test_x)[:200]])

    input_path = tmp_path / "inputs.json"
    aspects = np.where(np.arange(len(x)) % 2 == 0, 1.0, 640 / 480)
    input_path.write_text(json.dumps({"x": x.tolist(), "aspects": aspects.tolist()}))
    script = tmp_path / "run.mjs"
    script.write_text(NODE_SCRIPT)

    result = subprocess.run(
        [NODE, str(script), str(PROJECT_ROOT / "web" / "mlp.js"), str(json_path), str(bin_path), str(input_path)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    out = json.loads(result.stdout)
    js_features, js_probs = np.array(out["features"]), np.array(out["probs"])
    py_features = build_features(x, aspects)
    py_probs = model.predict_proba((py_features - mean) / std)

    np.testing.assert_allclose(js_features, py_features, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(js_probs, py_probs, atol=1e-5)
    assert (js_probs.argmax(1) == py_probs.argmax(1)).all()
