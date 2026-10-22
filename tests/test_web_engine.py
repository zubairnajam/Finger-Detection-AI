"""Cross-language parity: the browser's JS MLP must match the Python engine."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.prediction import load_model_and_scaler

NODE = shutil.which("node")

NODE_SCRIPT = r"""
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [modulePath, manifestPath, binPath, inputPath] = process.argv.slice(2);
const { MLP } = await import(pathToFileURL(modulePath).href);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const buf = readFileSync(binPath);
const mlp = new MLP(manifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const inputs = JSON.parse(readFileSync(inputPath, "utf8"));
console.log(JSON.stringify(inputs.map((x) => Array.from(mlp.predictProba(x)))));
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
    input_path.write_text(json.dumps(x.tolist()))
    script = tmp_path / "run.mjs"
    script.write_text(NODE_SCRIPT)

    result = subprocess.run(
        [NODE, str(script), str(PROJECT_ROOT / "web" / "mlp.js"), str(json_path), str(bin_path), str(input_path)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    js_probs = np.array(json.loads(result.stdout))
    py_probs = model.predict_proba((x - mean) / std)

    np.testing.assert_allclose(js_probs, py_probs, atol=1e-5)
    assert (js_probs.argmax(1) == py_probs.argmax(1)).all()
