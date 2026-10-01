from __future__ import annotations

from pathlib import Path
import shutil

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import train_tflite_cnn


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((24, 24), value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def _make_dataset(root: Path, count: int = 10) -> Path:
    for split in ("train", "test"):
        for class_name, value in (("def_front", 20), ("ok_front", 230)):
            for index in range(count):
                split_offset = 0 if split == "train" else 10
                direction = 1 if class_name == "def_front" else -1
                _write_image(
                    root / split / class_name / f"{split}_{class_name}_{index}.jpeg",
                    value + direction * (split_offset + index),
                )
    return root


def test_dataset_split_is_stratified_and_removes_test_leakage(tmp_path: Path) -> None:
    data_dir = _make_dataset(tmp_path / "casting")
    shutil.copyfile(
        data_dir / "train" / "def_front" / "train_def_front_0.jpeg",
        data_dir / "test" / "def_front" / "duplicate.jpeg",
    )

    splits = train_tflite_cnn.discover_dataset(data_dir, 0.2, seed=42)

    assert train_tflite_cnn.split_summary(splits) == {
        "train": {"def_front": 7, "ok_front": 8, "total": 15},
        "validation": {"def_front": 2, "ok_front": 2, "total": 4},
        "test": {"def_front": 11, "ok_front": 10, "total": 21},
    }
    assert splits.audit["removed_train_images_duplicated_in_test"] == 1


def test_grayscale_decode_and_augmentation_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpeg"
    _write_image(image_path, 200)
    image = train_tflite_cnn.decode_image(image_path)
    augmented = train_tflite_cnn.augment_image(image, tf.constant(1), seed=42)

    assert image.shape == (300, 300, 1)
    assert augmented.shape == (300, 300, 1)
    assert float(tf.reduce_min(augmented)) >= 0.0
    assert float(tf.reduce_max(augmented)) <= 255.0


def test_notebook_cnn_contract() -> None:
    model = train_tflite_cnn.build_model()
    result = model(np.zeros((1, 300, 300, 1), dtype=np.float32), training=False).numpy()

    assert model.input_shape == (None, 300, 300, 1)
    assert model.output_shape == (None, 1)
    assert model.count_params() == 4_669_249
    assert result.shape == (1, 1)
    assert 0.0 <= float(result[0, 0]) <= 1.0


def test_representative_dataset_anchors_uint8_range(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpeg"
    _write_image(image_path, 120)
    records = (train_tflite_cnn.ImageRecord(image_path, 0),)
    samples = list(train_tflite_cnn.representative_dataset(records)())

    assert len(samples) == 3
    assert np.all(samples[0][0] == 0.0)
    assert np.all(samples[1][0] == 255.0)
    assert samples[2][0].shape == (1, 300, 300, 1)


def test_builtin_full_int8_conversion(tmp_path: Path) -> None:
    inputs = keras.Input(shape=(8, 8, 1))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    samples = [np.full((1, 8, 8, 1), value, np.float32) for value in (0, 64, 255)]

    def representative():
        for sample in samples:
            yield [sample]

    result = train_tflite_cnn.convert_to_tflite(
        model, representative, tmp_path / "tiny_int8.tflite"
    )

    assert result.model_path.exists()
    assert result.input_details["shape"] == [1, 8, 8, 1]
    assert result.input_details["dtype"] == "uint8"
    assert result.output_details["dtype"] == "uint8"
    assert not any(operator.startswith("Flex") for operator in result.operators)


def test_acceptance_gate() -> None:
    passing = {"metrics": {"tflite": {"accuracy": 0.99}, "accuracy_drop": 0.005}}
    failing = {"metrics": {"tflite": {"accuracy": 0.97}, "accuracy_drop": 0.02}}

    assert train_tflite_cnn.enforce_acceptance(passing, allow_low_accuracy=False)
    with pytest.raises(RuntimeError):
        train_tflite_cnn.enforce_acceptance(failing, allow_low_accuracy=False)
    assert not train_tflite_cnn.enforce_acceptance(failing, allow_low_accuracy=True)


def test_validation_threshold_is_mapped_to_half() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.6, 0.7, 0.9])
    threshold = train_tflite_cnn.select_accuracy_threshold(labels, scores)
    model = train_tflite_cnn.build_model(image_size=(32, 32))
    output_layer = model.get_layer("ok_probability")
    kernel, bias = output_layer.get_weights()
    kernel.fill(0.0)
    bias.fill(np.log(threshold / (1.0 - threshold)))
    output_layer.set_weights([kernel, bias])

    adjustment = train_tflite_cnn.calibrate_model_threshold(model, threshold)
    calibrated = model(np.zeros((1, 32, 32, 1), np.float32), training=False).numpy()

    assert threshold == pytest.approx(0.65)
    assert adjustment == pytest.approx(-np.log(0.65 / 0.35))
    assert float(calibrated[0, 0]) == pytest.approx(0.5, abs=1e-6)
