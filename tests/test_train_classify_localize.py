from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import train_classify_localize as pipeline


def test_class_activation_model_and_calibration() -> None:
    trainer, exporter = pipeline.build_models((32, 32))
    image = np.zeros((1, 32, 32, 1), np.float32)
    score, heatmap = exporter(image, training=False)
    assert score.shape == (1, 1)
    assert len(heatmap.shape) == 4 and heatmap.shape[-1] == 1
    assert heatmap.shape[1] > 1
    before = float(score.numpy()[0, 0])
    pipeline.calibrate_threshold(trainer, 0.6)
    after = float(exporter(image, training=False)[0].numpy()[0, 0])
    assert after < before


def test_uint8_export_has_scalar_and_map(tmp_path: Path) -> None:
    # Use the production input size because the public TFLite contract is fixed.
    _, exporter = pipeline.build_models()
    def representative():
        for level in (0, 127, 255):
            yield [np.full((1, 300, 300, 1), level, np.float32)]
    path = tmp_path / "tiny.tflite"
    contract = pipeline.convert_model(exporter, representative, path)
    assert path.exists()
    assert contract["input"]["shape"] == [1, 300, 300, 1]
    assert contract["ok_probability"]["shape"] == [1, 1]
    assert contract["defect_heatmap"]["shape"][1] > 1
    assert all(contract[key]["dtype"] == "uint8" for key in ("input", "ok_probability", "defect_heatmap"))
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    score, heatmap = pipeline.identify_outputs(interpreter.get_output_details())
    assert score["index"] != heatmap["index"]
