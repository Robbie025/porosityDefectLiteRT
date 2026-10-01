from __future__ import annotations

from pathlib import Path
import shutil

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import train_model


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((24, 24, 3), value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def _make_dataset(root: Path, count: int = 10) -> Path:
    for split in ("train", "test"):
        for class_name, value in (("def_front", 20), ("ok_front", 230)):
            for index in range(count):
                split_offset = 0 if split == "train" else 10
                direction = 1 if class_name == "def_front" else -1
                unique_value = value + direction * (split_offset + index)
                _write_image(
                    root / split / class_name / f"{split}_{class_name}_{index}.jpeg",
                    unique_value,
                )
    return root


def test_dataset_split_is_deterministic_and_stratified(tmp_path: Path) -> None:
    data_dir = _make_dataset(tmp_path / "casting")
    first = train_model.discover_dataset(data_dir, 0.2, seed=42)
    second = train_model.discover_dataset(data_dir, 0.2, seed=42)

    assert [record.path for record in first.train] == [record.path for record in second.train]
    assert [record.path for record in first.validation] == [
        record.path for record in second.validation
    ]
    assert train_model.split_summary(first) == {
        "train": {"def_front": 8, "ok_front": 8, "total": 16},
        "validation": {"def_front": 2, "ok_front": 2, "total": 4},
        "test": {"def_front": 10, "ok_front": 10, "total": 20},
    }


def test_dataset_removes_train_test_content_overlap(tmp_path: Path) -> None:
    data_dir = _make_dataset(tmp_path / "casting", count=2)
    duplicate = data_dir / "test" / "def_front" / "train_def_front_0.jpeg"
    shutil.copyfile(data_dir / "train" / "def_front" / "train_def_front_0.jpeg", duplicate)
    splits = train_model.discover_dataset(data_dir, 0.2, seed=42)
    assert splits.audit["removed_train_images_duplicated_in_test"] == 1
    assert all(record.path.name != "train_def_front_0.jpeg" for record in splits.train)
    assert all(record.path.name != "train_def_front_0.jpeg" for record in splits.validation)


def test_image_variants_have_expected_shape_and_range(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpeg"
    _write_image(image_path, 200)
    decoded = train_model.decode_image(image_path)
    assert decoded.shape == (224, 224, 3)
    for name in ("original", "zoom", "brightness"):
        variant = train_model.image_variant(decoded, name)
        assert variant.shape == (224, 224, 3)
        assert float(tf.reduce_min(variant)) >= 0.0
        assert float(tf.reduce_max(variant)) <= 255.0
    assert float(tf.reduce_mean(train_model.image_variant(decoded, "brightness"))) < float(
        tf.reduce_mean(decoded)
    )


def test_classification_head_contract() -> None:
    head = train_model.build_classification_head(feature_dim=16)
    result = head(np.ones((3, 16), dtype=np.float32), training=False).numpy()
    assert result.shape == (3, 1)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_binary_metrics() -> None:
    metrics = train_model.binary_metrics(
        np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.8, 0.7, 0.9])
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_representative_dataset_anchors_uint8_range(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpeg"
    _write_image(image_path, 120)
    records = (train_model.ImageRecord(image_path, 0),)
    samples = list(train_model.representative_dataset(records)())
    assert len(samples) == 3
    assert np.all(samples[0][0] == 0.0)
    assert np.all(samples[1][0] == 255.0)
    assert samples[2][0].shape == (1, 224, 224, 3)


def test_full_int8_conversion_and_inference(tmp_path: Path) -> None:
    inputs = keras.Input(shape=(8, 8, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    samples = [np.full((1, 8, 8, 3), value, np.float32) for value in (0, 64, 128, 255)]

    def representative():
        for sample in samples:
            yield [sample]

    result = train_model.convert_to_tflite(model, representative, tmp_path / "tiny.tflite")
    assert result.model_path.exists()
    assert result.input_details["dtype"] == "uint8"
    assert result.output_details["dtype"] == "uint8"
    assert result.input_details["shape"] == [1, 8, 8, 3]

    interpreter = tf.lite.Interpreter(model_path=str(result.model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(
        input_details["index"], train_model.quantize_for_tensor(samples[-1], input_details)
    )
    interpreter.invoke()
    score = train_model.dequantize_tensor(
        interpreter.get_tensor(output_details["index"]), output_details
    )
    assert score.shape == (1, 1)
